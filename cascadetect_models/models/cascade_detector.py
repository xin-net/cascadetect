import torch
import cv2
import numpy as np
from cascadetect_models.models.yolo_model import YOLODetector
from cascadetect_models.models.cnn_model import CNNClassifier

import torch.utils.tensorboard as tensorboard
import os
from PIL import Image, ImageDraw, ImageFont

class CascadeDetector:
    """级联焊缝缺陷检测器，结合YOLOv8和CNN模型"""
    
    def __init__(self, yolo_model_path=None, cnn_model_path=None, num_classes=None,
                 conf_threshold=0.25, iou_threshold=0.45, cnn_input_size=(224, 224),
                 class_names=None):
        """
        初始化级联检测器
        
        参数:
            yolo_model_path: YOLOv8模型路径
            cnn_model_path: CNN模型路径
            num_classes: 类别数量
            conf_threshold: YOLOv8置信度阈值
            iou_threshold: YOLOv8 IOU阈值
            cnn_input_size: CNN输入图像尺寸
            class_names: 类别名称列表，如果为None则使用默认名称
        """
        # 初始化YOLOv8检测器
        self.yolo_detector = YOLODetector(
            model_path=yolo_model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # 如果num_classes未指定，尝试从模型加载
        if num_classes is None and cnn_model_path:
            # 临时创建一个CNN分类器来检查模型参数
            temp_classifier = CNNClassifier(num_classes=5)  # 临时使用5个类别
            try:
                state_dict = torch.load(cnn_model_path, map_location=temp_classifier.device)
                # 从fc.3.weight的形状获取类别数
                if "fc.3.weight" in state_dict:
                    num_classes = state_dict["fc.3.weight"].shape[0]
                    print(f"从模型自动检测到类别数: {num_classes}")
            except Exception as e:
                print(f"无法从模型自动检测类别数: {e}")
                num_classes = 5  # 默认为5
        
        # 初始化CNN分类器
        self.cnn_classifier = CNNClassifier(
            num_classes=num_classes,
            model_path=cnn_model_path,
            input_size=cnn_input_size
        )
        
        # 设置缺陷类别名称
        if class_names:
            self.class_names = class_names
        else:
            # 默认类别名称
            default_names = ['正常', '气孔', '裂纹', '夹渣', '未知缺陷']
            # 确保类别名称长度不小于num_classes
            self.class_names = default_names[:num_classes] if num_classes <= len(default_names) else \
                               default_names + [f'未知类型{i+1}' for i in range(len(default_names), num_classes)]
        
        print(f"级联检测器已初始化，类别数量: {num_classes}，类别名称: {self.class_names}")
    
    def detect(self, image):
        """
        对输入图像进行级联检测
        
        参数:
            image: 输入图像(OpenCV BGR格式)
            
        返回:
            results: 包含检测结果的列表，每个元素为[x1, y1, x2, y2, yolo_conf, yolo_class_id, cnn_class_id, cnn_conf]
        """
        # 第一阶段：YOLOv8检测
        bboxes, crops = self.yolo_detector.detect(image)
        
        if not bboxes:
            return []
        
        # 第二阶段：CNN精细分类
        refined_results = []
        
        for i, (bbox, crop) in enumerate(zip(bboxes, crops)):
            x1, y1, x2, y2, yolo_conf, yolo_class_id = bbox
            
            # 确保裁剪区域有效
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            
            # CNN分类
            cnn_class_id, cnn_conf = self.cnn_classifier.classify(crop)
            
            # 合并结果
            refined_results.append([
                x1, y1, x2, y2, 
                yolo_conf, 
                yolo_class_id,
                cnn_class_id, 
                cnn_conf
            ])
        
        return refined_results
    
    def visualize(self, image, results):
        """
        可视化检测结果
        
        参数:
            image: 输入图像
            results: 检测结果
            
        返回:
            vis_image: 带有检测框和标签的图像
        """
        vis_image = image.copy()
        
        # 颜色映射
        colors = [
            (0, 255, 0),    # 正常 - 绿色
            (0, 165, 255),  # 气孔 - 橙色
            (0, 0, 255),    # 裂纹 - 红色
            (255, 0, 0),    # 夹渣 - 蓝色
            (255, 0, 255)   # 其他类型 - 紫色
        ]
        
        # 扩展颜色列表，如果类别数超过预定义颜色数
        if len(self.class_names) > len(colors):
            import random
            for _ in range(len(self.class_names) - len(colors)):
                # 生成随机颜色
                colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        
        # 如果没有检测结果，直接返回原图
        if not results:
            return vis_image
        
        # 转换为PIL图像以支持中文
        pil_img = Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 尝试加载字体，支持中文显示
        font_size = 20
        font = None
        
        # Windows常见中文字体路径
        windows_fonts = [
            "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",    # 黑体
            "C:/Windows/Fonts/simsun.ttc",    # 宋体
            "C:/Windows/Fonts/simkai.ttf",    # 楷体
            "C:/Windows/Fonts/STKAITI.TTF",   # 华文楷体
        ]
        
        # Linux常见中文字体路径
        linux_fonts = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
            "/usr/share/fonts/truetype/arphic/uming.ttc",      # AR PL UMing
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # Noto Sans CJK
        ]
        
        # 首先尝试Windows字体
        for font_path in windows_fonts:
            try:
                font = ImageFont.truetype(font_path, font_size)
                print(f"使用字体: {font_path}")
                break
            except IOError:
                continue
        
        # 如果Windows字体加载失败，尝试Linux字体
        if font is None:
            for font_path in linux_fonts:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    print(f"使用字体: {font_path}")
                    break
                except IOError:
                    continue
        
        # 如果仍然失败，尝试从系统字体目录加载任何支持中文的字体
        if font is None:
            try:
                import matplotlib.font_manager as fm
                # 获取系统字体列表
                font_paths = fm.findSystemFonts()
                
                # 筛选可能支持中文的字体
                for font_path in font_paths:
                    if any(keyword in font_path.lower() for keyword in 
                           ["chinese", "cjk", "msyh", "simsun", "simhei", "yahei", "wqy", "noto"]):
                        try:
                            font = ImageFont.truetype(font_path, font_size)
                            print(f"使用系统字体: {font_path}")
                            break
                        except IOError:
                            continue
            except ImportError:
                print("警告: 无法导入matplotlib.font_manager来查找系统字体")
        
        # 如果所有尝试都失败，使用默认字体
        if font is None:
            font = ImageFont.load_default()
            print("警告: 未能加载中文字体，将使用默认字体，中文可能无法正确显示")
        
        # 获取当前项目根目录，尝试从项目中加载字体
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_dir = os.path.join(current_dir, "assets")
        project_fonts = [
            os.path.join(assets_dir, "fonts", "simhei.ttf"),  # 项目内黑体
            os.path.join(assets_dir, "fonts", "msyh.ttc"),    # 项目内微软雅黑
            os.path.join(assets_dir, "fonts", "simsun.ttc"),  # 项目内宋体
            os.path.join(current_dir, "fonts", "simhei.ttf"), # 另一种可能的项目结构
            os.path.join(current_dir, "resources", "fonts", "simhei.ttf"),
        ]
        
        # 尝试从项目目录加载字体
        if font is None:
            for font_path in project_fonts:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    print(f"使用项目字体: {font_path}")
                    break
                except IOError:
                    continue
        
        for res in results:
            x1, y1, x2, y2, yolo_conf, yolo_class_id, cnn_class_id, cnn_conf = res
            
            # 确保类别索引在有效范围内
            cnn_class_id = min(cnn_class_id, len(self.class_names) - 1)
            
            # 获取颜色和类别名称
            color = colors[cnn_class_id % len(colors)]
            label = f"{self.class_names[cnn_class_id]}: {cnn_conf:.2f}"
            
            # 在PIL图像上绘制矩形框 (注意PIL使用RGB顺序)
            rgb_color = (color[2], color[1], color[0])  # BGR转RGB
            
            # 绘制矩形框
            draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=rgb_color, width=2)
            
            # 获取文本大小
            try:
                # 新版PIL中的方法
                _, _, text_width, text_height = draw.textbbox((0, 0), label, font=font)
            except AttributeError:
                # 兼容旧版PIL
                text_width, text_height = draw.textsize(label, font=font)
            
            # 绘制标签背景
            draw.rectangle(
                [int(x1), int(y1) - text_height - 5, int(x1) + text_width, int(y1)],
                fill=rgb_color
            )
            
            # 绘制标签文本
            draw.text(
                (int(x1), int(y1) - text_height - 3),
                label,
                fill=(255, 255, 255),
                font=font
            )
        
        # 将PIL图像转回OpenCV格式
        vis_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        return vis_image
    
    def save_models(self, yolo_path, cnn_path):
        """保存模型"""
        self.yolo_detector.save(yolo_path)
        self.cnn_classifier.save(cnn_path)
    
    def load_models(self, yolo_path, cnn_path):
        """加载模型"""
        self.yolo_detector.load(yolo_path)
        self.cnn_classifier.load(cnn_path)