from ultralytics import YOLO
import torch
import numpy as np
import cv2
import os
import yaml
from pathlib import Path

class YOLODetector:
    """YOLOv8焊缝缺陷检测模型类"""
    
    def __init__(self, model_path=None, conf_threshold=0.25, iou_threshold=0.45):
        """
        初始化YOLOv8检测器
        
        参数:
            model_path: YOLOv8模型路径，如果为None则加载预训练模型
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
        """
        if model_path:
            self.model = YOLO(model_path)
        else:
            # 默认使用本地YOLOv8n模型 - 使用相对路径
            # 查找当前目录下的yolov8n.pt文件
            root_dir = Path(__file__).resolve().parents[1]  # 获取项目根目录
            yolo_model = root_dir / 'yolov8n.pt'
            
            if not yolo_model.exists():
                # 如果没有找到，使用ultralytics的预训练模型
                print(f"警告: 未找到本地模型文件: {yolo_model}，将使用预训练模型")
                self.model = YOLO('yolov8n.pt')  # 使用预训练模型
            else:
                print(f"使用本地模型文件: {yolo_model}")
                self.model = YOLO(str(yolo_model))
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"YOLOv8检测器已初始化，使用设备: {self.device}")
    
    def detect(self, image):
        """
        对输入图像进行缺陷检测，返回检测框
        
        参数:
            image: 输入图像，OpenCV格式(BGR)
            
        返回:
            bboxes: 边界框列表，每个元素为[x1, y1, x2, y2, conf, class_id]
            crops: 裁剪的候选区域图像列表
        """
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)
        
        # 获取检测结果
        bboxes = []
        crops = []
        
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                # 获取位置信息 [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # 获取置信度和类别
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # 存储结果
                bboxes.append([x1, y1, x2, y2, conf, cls_id])
                
                # 裁剪检测区域
                crop = image[y1:y2, x1:x2]
                crops.append(crop)
        
        return bboxes, crops
    
    def train(self, data_yaml, epochs=100, batch_size=16, imgsz=640):
        """
        训练YOLOv8模型
        
        参数:
            data_yaml: 数据配置文件路径
            epochs: 训练轮数
            batch_size: 批量大小
            imgsz: 图像大小
        """
        # 检查当前目录下是否有预训练模型
        root_dir = Path(__file__).resolve().parents[1]  # 获取项目根目录
        yolo_model = root_dir / 'yolov8n.pt'
        
        if yolo_model.exists():
            self.model = YOLO(str(yolo_model))
            print(f"使用本地预训练模型: {yolo_model}")
        else:
            self.model = YOLO('yolov8n.pt')
            print("使用官方预训练模型")
        
        # 检查data_yaml文件路径
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"数据配置文件不存在: {data_yaml}")
        
        # 读取并验证YAML文件
        with open(data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        # 检查关键路径
        yaml_dir = os.path.dirname(os.path.abspath(data_yaml))
        required_dirs = []
        
        # 获取训练和验证目录并检查它们是否存在
        for key in ['train', 'val']:
            if key in config:
                path = config[key]
                if not path.startswith('/'):  # 相对路径
                    full_path = os.path.join(yaml_dir, path.lstrip('./'))
                    required_dirs.append(full_path)
                    print(f"检查{key}目录: {full_path}")
        
        # 验证目录是否存在
        missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
        if missing_dirs:
            raise FileNotFoundError(f"以下目录不存在: {missing_dirs}")
        
        # 创建临时配置文件，但不修改原始文件
        temp_yaml = data_yaml.replace('.yaml', '_temp.yaml')
        
        # 创建配置副本
        modified_config = config.copy()
        
        # 将相对路径转换为绝对路径用于临时配置
        for key in ['train', 'val', 'test']:
            if key in modified_config and not modified_config[key].startswith('/'):
                modified_config[key] = os.path.join(yaml_dir, modified_config[key].lstrip('./'))
                print(f"转换{key}路径: {config[key]} -> {modified_config[key]}")
        
        # 保存临时配置
        with open(temp_yaml, 'w') as f:
            yaml.dump(modified_config, f, default_flow_style=False)
        
        print(f"已创建临时配置文件: {temp_yaml}")
        print(f"原始配置文件保持不变: {data_yaml}")
        
        try:
            # 使用绝对路径的临时配置文件进行训练
            # TensorBoard 会由 Ultralytics 根据全局设置自动处理
            self.model.train(
                data=temp_yaml,  # 使用临时配置文件路径
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                device='0' if torch.cuda.is_available() else 'cpu',  # 使用字符串格式的设备名称
                plots=True,
                save=True,
                project='runs/train/yolo',  # 修改: 指定项目目录到yolo子文件夹
                name='exp'  # 修改: 指定实验名称为exp，会自动递增 (exp, exp2, ...)
            )
            
            save_dir = Path(self.model.trainer.save_dir) # 获取实际保存目录
            print(f"TensorBoard日志和训练结果已保存到: {save_dir}") # 修改: 更新日志消息
            print(f"可以使用 'tensorboard --logdir {save_dir.parent}' 命令启动TensorBoard查看不同实验的结果")
            print(f"或者使用 'tensorboard --logdir {save_dir}' 查看当前实验的结果")
            
            # 返回保存的最佳模型路径
            best_model_path = save_dir / 'weights' / 'best.pt' # 修改: 更新最佳模型路径
            return str(best_model_path)
        finally:
            # 训练完成后删除临时文件
            if os.path.exists(temp_yaml):
                os.remove(temp_yaml)
                print(f"已删除临时配置文件: {temp_yaml}")
    
    def save(self, path):
        """保存模型"""
        self.model.save(path)
    
    def load(self, path):
        """加载模型"""
        self.model = YOLO(path)
        return self