from flask_restful import Resource, reqparse
from flask_jwt_extended import jwt_required
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from flask import current_app

import os
from models import Image, Annotation # 从 models.py 导入 Image 和 Annotation

from extensions import db
import cv2
from PIL import Image as PILImage, ImageDraw, ImageFont

def get_defect_color(defect_type):
    colors = {
        '气孔': (0, 0, 255),    # 红色 (BGR)
        '气泡': (255, 0, 0),    # 蓝色 (BGR)
        '水纹': (0, 255, 0),    # 绿色 (BGR)
        '新缺陷': (128, 0, 128), # 紫色 (BGR)
        # 可以根据需要添加更多缺陷类型和颜色
    }
    return colors.get(defect_type, (0, 165, 255)) # 默认橙色 (BGR)
# 假设检测脚本和模型在 cascadetect_models 文件夹中


from cascadetect_models.models.cascade_detector import CascadeDetector # 从 models.cascade_detector.py 导入 CascadeDetector

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# UPLOAD_FOLDER is now configured in app.py and accessed via app.config['UPLOAD_FOLDER']

# The check for UPLOAD_FOLDER existence and creation is also handled in app.py

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 初始化检测器 (根据实际模型路径调整)
# 注意：这里的路径是相对于 cascadetect_backend 文件夹的
yolo_model_path = os.path.join(os.path.dirname(__file__), '..', 'cascadetect_models', 'runs', 'train', 'yolo', 'exp', 'weights', 'best.pt')
cnn_model_path = os.path.join(os.path.dirname(__file__), '..', 'cascadetect_models', 'runs', 'train', 'cnn', '1', 'cnn_model.pt') # 假设CNN模型也保存在类似路径

# 检查模型文件是否存在，如果不存在，则尝试使用默认的预训练模型路径
if not os.path.exists(yolo_model_path):
    yolo_model_path = os.path.join(os.path.dirname(__file__), '..', 'cascadetect_models', 'yolov8n.pt')
    print(f"警告: YOLO 模型 {os.path.join(os.path.dirname(__file__), '..', 'cascadetect_models', 'runs', 'train', 'yolo', 'exp', 'weights', 'best.pt')} 不存在，将使用默认模型 {yolo_model_path}")

if not os.path.exists(cnn_model_path):
    # 如果CNN模型不存在，这里可以设置一个None或者一个默认的CNN模型路径（如果cascadetect_models中有的话）
    # 这里暂时设置为None，表示如果自定义CNN模型不存在，则CascadeDetector内部可能会使用其默认逻辑或报错
    cnn_model_path = None # 或者指向一个默认的CNN模型
    print(f"警告: CNN 模型 {os.path.join(os.path.dirname(__file__), '..', 'cascadetect_models', 'runs', 'train', 'cnn_model', 'best_cnn_model.pt')} 不存在，将不加载CNN模型或使用默认CNN模型（如果CascadeDetector支持）。")

# 尝试从 data.yaml 获取类别名称
class_names = ['正常', '气孔', '裂纹', '夹渣', '其他'] # 默认类别
try:
    data_yaml_path = os.path.join(os.path.dirname(__file__), '..', 'cascadetect_models', 'datasets', 'yolo_v8', '03_yolo_standard', 'data.yaml')
    if os.path.exists(data_yaml_path):
        import yaml
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
            if 'names' in data_config:
                class_names = data_config['names']
                print(f"从 {data_yaml_path} 加载类别名称: {class_names}")
except Exception as e:
    print(f"加载类别名称失败: {e}，使用默认类别: {class_names}")


detector = CascadeDetector(
    yolo_model_path=yolo_model_path,
    cnn_model_path=cnn_model_path, # 如果为None，CascadeDetector内部会处理
    class_names=class_names
)

class ImageUpload(Resource):
    #@jwt_required()
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('files', type=FileStorage, location='files', action='append')
        args = parser.parse_args()
        
        uploaded_files = args['files']
        if not uploaded_files:
            return {'message': 'No file part'}, 400

        saved_images_info = []
        for file_storage in uploaded_files:
            if file_storage and allowed_file(file_storage.filename):
                original_filename = secure_filename(file_storage.filename)
                filename_base, filename_ext = os.path.splitext(original_filename)
                counter = 1
                filename = original_filename
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                # 如果文件已存在，则尝试添加后缀直到文件名唯一
                while os.path.exists(filepath):
                    filename = f"{filename_base}_{counter}{filename_ext}"
                    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                    counter += 1
                file_storage.save(filepath)

                # 将图片信息存入数据库
                new_image = Image(filename=filename)
                db.session.add(new_image)
                db.session.commit()

                # 调用模型进行缺陷检测
                # 注意：run_detection 是我们从 detect.py 导入的函数
                # 它内部会调用 detector.detect 和 detector.visualize
                # 我们需要确保 run_detection 返回的是结构化的标注数据
                # 或者直接在这里调用 detector.detect
                import cv2
                img_cv = cv2.imread(filepath)
                if img_cv is None:
                    # 如果图片读取失败，跳过这张图片
                    print(f"无法读取图片: {filepath}")
                    continue
                
                detection_results = detector.detect(img_cv) # 直接调用 detector.detect

                annotations_data = []
                for res in detection_results:
                    x1, y1, x2, y2, yolo_conf, yolo_cls, cnn_cls, cnn_conf = res
                    # CascadeDetector的class_names是基于CNN分类结果的
                    defect_type_name = detector.class_names[min(int(cnn_cls), len(detector.class_names) - 1)]
                    
                    annotation = Annotation(
                        image_id=new_image.id,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        defect_type=defect_type_name,
                        confidence=cnn_conf
                    )
                    db.session.add(annotation)
                    annotations_data.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'defect_type': defect_type_name,
                        'confidence': cnn_conf
                    })
                db.session.commit()
                saved_images_info.append({
                    'id': new_image.id,
                    'filename': filename,
                    'upload_time': new_image.upload_time.isoformat(),
                    'annotations': annotations_data
                })
            else:
                # 如果文件不允许或不存在，可以记录日志或返回错误信息
                print(f"文件类型不允许或文件不存在: {file_storage.filename if file_storage else 'N/A'}")

        if not saved_images_info:
             return {'message': 'No valid files uploaded or processed'}, 400

        return {'message': 'Files uploaded and processed successfully', 'images': saved_images_info}, 201

class ImageList(Resource):
    #@jwt_required()
    def get(self):
        images = db.session.query(Image).all()
        return [{'id': img.id, 'filename': img.filename, 'upload_time': img.upload_time.isoformat()} for img in images]

class ImageDetail(Resource):
    #@jwt_required()
    def get(self, image_id):
        image = db.session.query(Image).get_or_404(image_id)
        annotations = db.session.query(Annotation).filter_by(image_id=image.id).all()
        return {
            'id': image.id,
            'filename': image.filename,
            'upload_time': image.upload_time.isoformat(),
            'annotations': [
                {'id': ann.id, 'x1': ann.x1, 'y1': ann.y1, 'x2': ann.x2, 'y2': ann.y2, 'defect_type': ann.defect_type, 'confidence': ann.confidence}
                for ann in annotations
            ]
        }

    #@jwt_required()
    def delete(self, image_id):
        image = db.session.query(Image).get_or_404(image_id)
        # 只删除服务器上的物理文件
        try:
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], image.filename)
            if os.path.exists(filepath):
                os.remove(filepath)
            # 删除数据库中与该图片关联的所有标注信息
            db.session.query(Annotation).filter_by(image_id=image.id).delete()
            # 删除图片本身的记录
            db.session.delete(image)
            db.session.commit()
            return {'message': 'Image and associated data deleted successfully'}, 200
        except Exception as e:
            # 记录文件删除错误
            print(f"Error deleting file {filepath}: {e}")
            db.session.rollback() # 如果发生错误，回滚事务
            return {'message': f'Error deleting file or database record: {e}'}, 500

class AnnotationUpdate(Resource):
    #@jwt_required()
    def put(self, annotation_id):
        parser = reqparse.RequestParser()
        parser.add_argument('x1', type=float)
        parser.add_argument('y1', type=float)
        parser.add_argument('x2', type=float)
        parser.add_argument('y2', type=float)
        parser.add_argument('defect_type', type=str)
        args = parser.parse_args()

        annotation = db.session.query(Annotation).get_or_404(annotation_id)
        if args['x1'] is not None: annotation.x1 = args['x1']
        if args['y1'] is not None: annotation.y1 = args['y1']
        if args['x2'] is not None: annotation.x2 = args['x2']
        if args['y2'] is not None: annotation.y2 = args['y2']
        if args['defect_type'] is not None: annotation.defect_type = args['defect_type']
        
        db.session.commit()
        return {'message': 'Annotation updated successfully'}

    #@jwt_required()
    def delete(self, annotation_id):
        annotation = db.session.query(Annotation).get_or_404(annotation_id)
        db.session.delete(annotation)
        db.session.commit()
        return {'message': 'Annotation deleted successfully'}

class AnnotationCreate(Resource):
    #@jwt_required()
    def post(self, image_id):
        parser = reqparse.RequestParser()
        parser.add_argument('x1', type=float, required=True, help="x1 coordinate cannot be blank!")
        parser.add_argument('y1', type=float, required=True, help="y1 coordinate cannot be blank!")
        parser.add_argument('x2', type=float, required=True, help="x2 coordinate cannot be blank!")
        parser.add_argument('y2', type=float, required=True, help="y2 coordinate cannot be blank!")
        parser.add_argument('defect_type', type=str, required=True, help="Defect type cannot be blank!")
        parser.add_argument('confidence', type=float, required=True, help="Confidence cannot be blank!") # 通常由用户创建时，置信度可以默认为1或由用户输入
        args = parser.parse_args()

        image = db.session.query(Image).get_or_404(image_id)
        new_annotation = Annotation(
            image_id=image.id,
            x1=args['x1'], y1=args['y1'], x2=args['x2'], y2=args['y2'],
            defect_type=args['defect_type'],
            confidence=args['confidence']
        )
        db.session.add(new_annotation)
        db.session.commit()
        return {'message': 'Annotation created successfully', 'annotation_id': new_annotation.id}, 201

from flask import send_file
import io
import cv2
import numpy as np

class ImageExport(Resource):
    #@jwt_required()
    def get(self, image_id):
        image = db.session.query(Image).get_or_404(image_id)
        annotations = db.session.query(Annotation).filter_by(image_id=image.id).all()
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], image.filename)

        if not os.path.exists(filepath):
            return {'message': 'Image file not found'}, 404

        try:
                with open(filepath, 'rb') as f:
                    img_bytes = io.BytesIO(f.read())

                # 将图片转换为PIL格式以支持中文
                img_pil = PILImage.open(img_bytes)
                draw = ImageDraw.Draw(img_pil)

                # 尝试加载字体，确保支持中文
                try:
                    # 假设字体文件在项目根目录或某个可访问的路径
                    # 生产环境中请确保字体文件存在且路径正确
                    font_path = os.path.join(os.path.dirname(__file__), '..', 'cascadetect_models', 'assets', 'fonts', 'simsun.ttc') # 示例字体路径
                    if not os.path.exists(font_path):
                        # 如果simsun.ttc不存在，尝试其他常见字体或提供警告
                        print(f"警告: 未找到字体文件 {font_path}，尝试使用系统默认字体或指定其他字体。")
                        # 尝试使用Windows系统自带的字体
                        if os.name == 'nt': # Windows系统
                            font_path = 'C:/Windows/Fonts/simhei.ttf' # 黑体
                            if not os.path.exists(font_path):
                                font_path = 'C:/Windows/Fonts/msyh.ttc' # 微软雅黑

                    font = ImageFont.truetype(font_path, 20) if os.path.exists(font_path) else ImageFont.load_default()
                except Exception as e:
                    print(f"加载字体失败: {e}，使用默认字体。")
                    font = ImageFont.load_default()

                # 遍历标注，绘制矩形框和缺陷类型
                for ann in annotations:
                    x1, y1, x2, y2 = int(ann.x1), int(ann.y1), int(ann.x2), int(ann.y2)
                    color_bgr = get_defect_color(ann.defect_type)
                    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0]) # BGR转RGB

                    draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=2)

                    text = f"{ann.defect_type} ({ann.confidence:.2f})"
                    # 计算文本大小以确定文本背景框位置
                    text_bbox = draw.textbbox((0,0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    # 绘制文本背景（可选，为了文字更清晰）
                    text_bg_x1 = x1
                    text_bg_y1 = y1 - text_height - 5 if y1 - text_height - 5 > 0 else y1 + 5 # 避免超出图片顶部
                    text_bg_x2 = x1 + text_width + 5
                    text_bg_y2 = y1 - 5 if y1 - 5 > 0 else y1 + text_height + 10
                    draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2], fill=color_rgb)

                    # 绘制文本
                    draw.text((x1 + 2, text_bg_y1 + 2), text, font=font, fill=(255, 255, 255)) # 白色文字

                # 将PIL图像转换回OpenCV格式以便编码
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                # 将修改后的图片编码为JPEG格式
                _, img_encoded = cv2.imencode('.jpg', img_cv)
                img_bytes = io.BytesIO(img_encoded.tobytes())

                return send_file(img_bytes, mimetype='image/jpeg', as_attachment=True, download_name=f'annotated_{image.filename}')

        except Exception as e:
            print(f"Error processing image for export: {e}")
            return {'message': f'Error processing image for export: {e}'}, 500