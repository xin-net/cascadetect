#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
焊缝缺陷检测系统训练脚本
"""

import os
import argparse
import torch
from pathlib import Path
from models.yolo_model import YOLODetector
from models.cnn_model import CNNClassifier
from models.cascade_detector import CascadeDetector
from utils.data_utils import create_dataloaders
import yaml

def get_next_run_dir(model_type, base_dir):
    """
    获取下一个训练运行目录
    
    参数:
        model_type: 模型类型 ('yolo' 或 'cnn')
        base_dir: 基础目录
        
    返回:
        下一个运行目录路径
    """
    # 确保base_dir存在
    os.makedirs(base_dir, exist_ok=True)
    
    model_dir = os.path.join(base_dir, model_type)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, '1')
    
    # 查找现有的run目录
    existing_runs = [d for d in os.listdir(model_dir) 
                      if os.path.isdir(os.path.join(model_dir, d)) and d.isdigit()]
    
    if not existing_runs:
        return os.path.join(model_dir, '1')
    
    # 找到最大的run编号并加1
    max_run = max(map(int, existing_runs))
    next_run = max_run + 1
    
    return os.path.join(model_dir, str(next_run))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='焊缝缺陷检测系统训练脚本')
    
    # 数据集参数
    parser.add_argument('--data-dir', type=str, default='datasets',
                        help='数据集根目录')
    parser.add_argument('--yolo-dataset', type=str, default='yolo_v8/03_yolo_standard',
                        help='YOLO数据集路径 (相对于数据集根目录)')
    parser.add_argument('--cnn-dataset', type=str, default='cnn/resnet50_standard',
                        help='CNN数据集路径 (相对于数据集根目录)')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批量大小')
    parser.add_argument('--img-size', type=int, default=640,
                        help='YOLOv8训练图像大小')
    parser.add_argument('--cnn-size', type=int, default=224,
                        help='CNN训练图像大小')
    parser.add_argument('--yolo-epochs', type=int, default=100,
                        help='YOLOv8训练轮数')
    parser.add_argument('--cnn-epochs', type=int, default=50,
                        help='CNN训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='CNN学习率')
    
    # 模型参数
    parser.add_argument('--num-classes', type=int, default=5,
                        help='类别数量')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='YOLOv8置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='YOLOv8 IOU阈值')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='runs/train',
                        help='输出目录')
    parser.add_argument('--yolo-only', action='store_true',
                        help='仅训练YOLOv8模型')
    parser.add_argument('--cnn-only', action='store_true',
                        help='仅训练CNN模型')
    
    return parser.parse_args()


def train_yolo(args):
    """训练YOLOv8模型"""
    print("="*50)
    print("开始训练YOLOv8模型")
    print("="*50)
    
    # 准备数据
    # 修复路径拼接问题 - 避免重复的yolo_v8目录
    # 检查args.yolo_dataset是否已经包含了yolo_v8前缀
    if args.yolo_dataset.startswith('yolo_v8/'):
        # 如果已经包含前缀，直接使用相对路径
        data_yaml = os.path.join(args.data_dir, args.yolo_dataset, 'data.yaml')
    else:
        # 否则，添加yolo_v8前缀
        data_yaml = os.path.join(args.data_dir, 'yolo_v8', args.yolo_dataset, 'data.yaml')
    
    print(f"使用YOLO数据集配置文件: {data_yaml}")
    
    # 初始化检测器
    model = YOLODetector()
    
    # 训练模型
    model_save_path = model.train(
        data_yaml=data_yaml,
        epochs=args.yolo_epochs,
        batch_size=args.batch_size,
        imgsz=args.img_size
    )
    
    # 获取类别数量
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    num_classes = len(data_config['names'])
    
    return model_save_path, num_classes


def train_cnn(args):
    """训练CNN模型"""
    print("=" * 50)
    print("开始训练CNN模型")
    print("=" * 50)
    
    # 使用指定的CNN数据集目录
    cnn_dir = os.path.join(args.data_dir, args.cnn_dataset)
    
    print(f"使用CNN数据集: {cnn_dir}")
    
    # 确保数据集目录存在
    if not os.path.exists(cnn_dir):
        raise FileNotFoundError(f"CNN数据集目录不存在: {cnn_dir}")
    
    # 检查是否是按类别组织的目录结构
    train_dir = os.path.join(cnn_dir, 'train')
    valid_dir = os.path.join(cnn_dir, 'valid')
    test_dir = os.path.join(cnn_dir, 'test')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"训练集目录不存在: {train_dir}")
    
    # 获取类别列表（从train目录中的子目录名称）
    class_names = [d for d in os.listdir(train_dir) 
                   if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')]
    
    if not class_names:
        raise ValueError(f"在训练集目录中没有找到类别子目录: {train_dir}")
    
    num_classes = len(class_names)
    print(f"检测到的类别: {class_names}")
    print(f"类别数量: {num_classes}")
    
    # 初始化CNN模型
    model = CNNClassifier(num_classes=num_classes)
    
    # 创建数据加载器，直接使用按类别组织的数据集结构
    from utils.data_utils import create_dataloaders
    train_loader, valid_loader, _ = create_dataloaders(
        data_dir=cnn_dir,
        batch_size=args.batch_size,
        input_size=(args.cnn_size, args.cnn_size)
    )
    epochs = args.cnn_epochs
    
    # 获取下一个cnn训练目录
    cnn_run_dir = get_next_run_dir('cnn', args.output_dir)
    os.makedirs(cnn_run_dir, exist_ok=True)
    
    # 训练模型
    model.train(
        train_loader=train_loader,
        val_loader=valid_loader,
        epochs=epochs,
        learning_rate=args.lr,
        log_dir=cnn_run_dir
    )
    
    # 保存模型
    model_save_path = os.path.join(cnn_run_dir, 'cnn_model.pt')
    model.save(model_save_path)
    print(f"CNN模型已保存到 {model_save_path}")
    
    return model_save_path, num_classes


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练模型
    yolo_model_path = None
    cnn_model_path = None
    num_classes = args.num_classes
    
    # 训练YOLOv8模型
    if not args.cnn_only:
        yolo_result = train_yolo(args)
        if isinstance(yolo_result, tuple):
            yolo_model_path, yolo_num_classes = yolo_result
            if num_classes == args.num_classes:  # 如果还没有从CNN更新过
                num_classes = yolo_num_classes
        else:
            yolo_model_path = yolo_result
    
    # 训练CNN模型
    if not args.yolo_only:
        cnn_model_path, num_classes = train_cnn(args)
    
    # 如果两个模型都训练了，创建级联检测器并保存
    if yolo_model_path and cnn_model_path:
        print("=" * 50)
        print("创建级联检测器")
        print("=" * 50)
        
        # 初始化级联检测器
        cascade_detector = CascadeDetector(
            yolo_model_path=yolo_model_path,
            cnn_model_path=cnn_model_path,
            num_classes=num_classes
        )
        
        # 保存级联检测器配置
        cascade_save_dir = os.path.join(args.output_dir, 'cascade')
        os.makedirs(cascade_save_dir, exist_ok=True)
        cascade_config_path = os.path.join(cascade_save_dir, 'cascade_config.yaml')
        
        config = {
            'yolo_model_path': yolo_model_path,
            'cnn_model_path': cnn_model_path,
            'num_classes': num_classes,
            'conf_thres': args.conf_thres,
            'iou_thres': args.iou_thres
        }
        
        with open(cascade_config_path, 'w') as f:
            yaml.dump(config, f)
            
        print(f"级联检测器配置已保存到 {cascade_config_path}")
        
    print("\n训练完成!")
    if yolo_model_path:
        print(f"YOLOv8模型已保存到: {yolo_model_path}")
    if cnn_model_path:
        print(f"CNN模型已保存到: {cnn_model_path}")
    if yolo_model_path and cnn_model_path:
        print(f"级联检测器配置已保存到: {cascade_config_path}")
    
    return 0

if __name__ == "__main__":
    main()