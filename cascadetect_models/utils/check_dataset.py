#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据集检查和修复工具
用于检查YOLOv8数据集路径并修复可能的问题
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_directory_exists(dir_path):
    """检查目录是否存在，如果不存在则创建"""
    if not os.path.exists(dir_path):
        logger.warning(f"目录不存在: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"已创建目录: {dir_path}")
        return False
    return True

def check_and_fix_yaml(yaml_path):
    """检查YAML配置文件的路径，验证相对路径但不修改原文件"""
    if not os.path.exists(yaml_path):
        logger.error(f"YAML文件不存在: {yaml_path}")
        return False
    
    try:
        # 读取YAML文件
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"当前YAML配置: {config}")
        
        # 检查路径并验证目录
        yolo_dir = os.path.dirname(yaml_path)
        yolo_dir_abs = os.path.abspath(yolo_dir)
        
        # 检查train、val和test路径
        for key in ['train', 'val', 'test']:
            if key in config:
                original_path = config[key]
                logger.info(f"检查{key}路径: {original_path}")
                
                # 构建绝对路径以验证目录是否存在
                if not original_path.startswith('/'):
                    # 相对路径
                    rel_path = original_path.lstrip('./')
                    abs_path = os.path.join(yolo_dir, rel_path)
                    
                    # 检查目录是否存在
                    if os.path.exists(abs_path):
                        logger.info(f"✓ 目录存在: {abs_path}")
                    else:
                        logger.warning(f"✗ 目录不存在: {abs_path}")
                        check_directory_exists(os.path.dirname(abs_path))
                        logger.info(f"已创建目录结构: {os.path.dirname(abs_path)}")
                else:
                    # 绝对路径
                    if os.path.exists(original_path):
                        logger.info(f"✓ 目录存在: {original_path}")
                    else:
                        logger.warning(f"✗ 目录不存在: {original_path}")
                        check_directory_exists(os.path.dirname(original_path))
                        logger.info(f"已创建目录结构: {os.path.dirname(original_path)}")
        
        # 验证类别信息
        if 'names' in config:
            logger.info(f"类别名称: {config['names']}")
        else:
            logger.warning("YAML文件中未找到类别名称 (names)")
        
        if 'nc' in config:
            logger.info(f"类别数量: {config['nc']}")
        else:
            logger.warning("YAML文件中未找到类别数量 (nc)")
        
        # 不再修改原始YAML文件
        logger.info(f"验证完成，保持原始配置文件不变: {yaml_path}")
        
        return True
    except Exception as e:
        logger.exception(f"处理YAML文件时出错: {e}")
        return False

def verify_dataset_structure(dataset_dir):
    """验证数据集目录结构"""
    # 检查主要目录
    train_images = os.path.join(dataset_dir, 'train', 'images')
    train_labels = os.path.join(dataset_dir, 'train', 'labels')
    valid_images = os.path.join(dataset_dir, 'valid', 'images')
    valid_labels = os.path.join(dataset_dir, 'valid', 'labels')
    test_images = os.path.join(dataset_dir, 'test', 'images')
    test_labels = os.path.join(dataset_dir, 'test', 'labels')
    
    # 确保所有目录存在
    check_directory_exists(train_images)
    check_directory_exists(train_labels)
    check_directory_exists(valid_images)
    check_directory_exists(valid_labels)
    check_directory_exists(test_images)
    check_directory_exists(test_labels)
    
    # 检查图像和标签文件
    train_img_count = len([f for f in os.listdir(train_images) if f.endswith(('.jpg', '.jpeg', '.png'))])
    train_lbl_count = len([f for f in os.listdir(train_labels) if f.endswith('.txt')])
    valid_img_count = len([f for f in os.listdir(valid_images) if f.endswith(('.jpg', '.jpeg', '.png'))])
    valid_lbl_count = len([f for f in os.listdir(valid_labels) if f.endswith('.txt')])
    test_img_count = len([f for f in os.listdir(test_images) if f.endswith(('.jpg', '.jpeg', '.png'))])
    test_lbl_count = len([f for f in os.listdir(test_labels) if f.endswith('.txt')])
    
    logger.info(f"训练集: {train_img_count}张图像, {train_lbl_count}个标签文件")
    logger.info(f"验证集: {valid_img_count}张图像, {valid_lbl_count}个标签文件")
    logger.info(f"测试集: {test_img_count}张图像, {test_lbl_count}个标签文件")
    
    if train_img_count == 0 or valid_img_count == 0:
        logger.warning("训练集或验证集中没有图像文件！")
    
    return {
        'train_images': train_img_count,
        'train_labels': train_lbl_count,
        'valid_images': valid_img_count,
        'valid_labels': valid_lbl_count,
        'test_images': test_img_count,
        'test_labels': test_lbl_count
    }

def main():
    """主函数"""
    # 设置数据集路径
    dataset_dir = 'datasets/yolo_v8/03_yolo_standard'
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    
    logger.info(f"检查数据集: {dataset_dir}")
    
    # 检查目录结构
    check_directory_exists(dataset_dir)
    
    # 验证数据集结构
    stats = verify_dataset_structure(dataset_dir)
    
    # 检查并修复YAML文件
    if check_and_fix_yaml(yaml_path):
        logger.info("YAML文件检查完成")
    
    # 总结
    logger.info("数据集检查完成")
    for k, v in stats.items():
        logger.info(f"{k}: {v}")

if __name__ == "__main__":
    main() 