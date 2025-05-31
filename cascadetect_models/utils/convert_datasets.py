#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据集处理工具 - 多阶段处理
1. 将多边形标注转换为矩形框
2. 划分数据集为训练集、验证集和测试集
3. 将YOLOv8标准格式数据集转换为ResNet50所需的CNN数据集格式
4. 将多种缺陷类别合并为单一"缺陷"类别
"""

import os
import sys
import yaml
import argparse
import logging
import numpy as np
import shutil
import random
import cv2
from pathlib import Path
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_dir(dir_path):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"创建目录: {dir_path}")

def read_yaml_config(yaml_path):
    """读取YAML配置文件"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def write_yaml_config(config, yaml_path):
    """写入YAML配置文件"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

def get_min_enclosing_rect(points):
    """获取多边形的最小包围矩形
    
    Args:
        points: 多边形顶点坐标，形状为 [N, 2]
    
    Returns:
        [x_min, y_min, x_max, y_max]: 矩形框坐标 (归一化)
    """
    points = np.array(points)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    
    # 确保边界在图像范围内
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(1, x_max)
    y_max = min(1, y_max)
    
    return [x_min, y_min, x_max, y_max]

def convert_polygon_to_rectangle(input_label_path, output_label_path):
    """将多边形标注转换为矩形框
    
    Args:
        input_label_path: 输入标签文件路径
        output_label_path: 输出标签文件路径
    
    Returns:
        bool: 转换是否成功
    """
    if not os.path.exists(input_label_path):
        logger.warning(f"标签文件不存在: {input_label_path}")
        return False
    
    # 读取标签文件
    with open(input_label_path, 'r') as f:
        lines = f.readlines()
    
    # 转换每行数据
    output_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        
        # 读取类别
        class_id = int(parts[0])
        
        # 读取多边形坐标点
        polygon_points = []
        for i in range(1, len(parts), 2):
            if i+1 < len(parts):
                x = float(parts[i])
                y = float(parts[i+1])
                polygon_points.append([x, y])
        
        if not polygon_points:
            continue
            
        # 获取最小外接矩形
        x_min, y_min, x_max, y_max = get_min_enclosing_rect(polygon_points)
        
        # 计算中心点和宽高
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        # YOLO格式: <class_id> <center_x> <center_y> <width> <height>
        output_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
        output_lines.append(output_line)
    
    # 写入输出文件
    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
    with open(output_label_path, 'w') as f:
        f.write('\n'.join(output_lines))
    
    return True

def step1_polygon_to_rectangle(source_dir, target_dir):
    """第一步：将多边形标注转换为矩形框
    
    Args:
        source_dir: 源目录，包含原始多边形标注
        target_dir: 目标目录，存储转换后的矩形框标注
    """
    logger.info("=== 第一步：多边形转矩形 ===")
    
    # 确保目录存在
    source_train_images = os.path.join(source_dir, 'train', 'images')
    source_train_labels = os.path.join(source_dir, 'train', 'labels')
    source_yaml = os.path.join(source_dir, 'data.yaml')
    
    target_train_images = os.path.join(target_dir, 'train', 'images')
    target_train_labels = os.path.join(target_dir, 'train', 'labels')
    target_yaml = os.path.join(target_dir, 'data.yaml')
    
    # 检查源目录
    if not os.path.exists(source_train_images):
        logger.error(f"源图像目录不存在: {source_train_images}")
        return False
    
    if not os.path.exists(source_train_labels):
        logger.error(f"源标签目录不存在: {source_train_labels}")
        return False
    
    if not os.path.exists(source_yaml):
        logger.error(f"源配置文件不存在: {source_yaml}")
        return False
    
    # 创建目标目录
    ensure_dir(target_train_images)
    ensure_dir(target_train_labels)
    
    # 复制图像文件
    logger.info(f"复制图像文件从 {source_train_images} 到 {target_train_images}")
    image_files = [f for f in os.listdir(source_train_images) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for img_file in tqdm(image_files, desc="复制图像"):
        src_path = os.path.join(source_train_images, img_file)
        dst_path = os.path.join(target_train_images, img_file)
        shutil.copy2(src_path, dst_path)
    
    # 转换标签文件
    logger.info(f"转换标签文件从 {source_train_labels} 到 {target_train_labels}")
    label_files = [f for f in os.listdir(source_train_labels) if f.endswith('.txt')]
    success_count = 0
    for label_file in tqdm(label_files, desc="转换标签"):
        src_path = os.path.join(source_train_labels, label_file)
        dst_path = os.path.join(target_train_labels, label_file)
        if convert_polygon_to_rectangle(src_path, dst_path):
            success_count += 1
    
    logger.info(f"成功转换 {success_count}/{len(label_files)} 个标签文件")
    
    # 复制并更新配置文件
    logger.info(f"复制配置文件从 {source_yaml} 到 {target_yaml}")
    shutil.copy2(source_yaml, target_yaml)
    
    logger.info("第一步处理完成！")
    return True

def split_dataset(files, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1, seed=42):
    """将数据集分割为训练集、验证集和测试集
    
    Args:
        files: 文件路径列表
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        train_files, valid_files, test_files: 分割后的文件列表
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-5, "比例总和必须为1"
    
    random.seed(seed)
    random.shuffle(files)
    
    n = len(files)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)
    
    train_files = files[:train_end]
    valid_files = files[train_end:valid_end]
    test_files = files[valid_end:]
    
    return train_files, valid_files, test_files

def step2_split_dataset(source_dir, target_dir, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1, seed=42):
    """第二步：划分数据集为训练集、验证集和测试集
    
    Args:
        source_dir: 源目录，包含矩形框标注
        target_dir: 目标目录，存储划分后的数据集
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    logger.info("=== 第二步：划分数据集 ===")
    
    # 确保目录存在
    source_train_images = os.path.join(source_dir, 'train', 'images')
    source_train_labels = os.path.join(source_dir, 'train', 'labels')
    source_yaml = os.path.join(source_dir, 'data.yaml')
    
    target_train_images = os.path.join(target_dir, 'train', 'images')
    target_train_labels = os.path.join(target_dir, 'train', 'labels')
    
    target_valid_images = os.path.join(target_dir, 'valid', 'images')
    target_valid_labels = os.path.join(target_dir, 'valid', 'labels')
    
    target_test_images = os.path.join(target_dir, 'test', 'images')
    target_test_labels = os.path.join(target_dir, 'test', 'labels')
    
    target_yaml = os.path.join(target_dir, 'data.yaml')
    
    # 检查源目录
    if not os.path.exists(source_train_images):
        logger.error(f"源图像目录不存在: {source_train_images}")
        return False
    
    if not os.path.exists(source_train_labels):
        logger.error(f"源标签目录不存在: {source_train_labels}")
        return False
    
    if not os.path.exists(source_yaml):
        logger.error(f"源配置文件不存在: {source_yaml}")
        return False
    
    # 创建目标目录
    ensure_dir(target_train_images)
    ensure_dir(target_train_labels)
    ensure_dir(target_valid_images)
    ensure_dir(target_valid_labels)
    ensure_dir(target_test_images)
    ensure_dir(target_test_labels)
    
    # 获取图像文件列表
    image_files = [f for f in os.listdir(source_train_images) if f.endswith(('.jpg', '.jpeg', '.png'))]
    logger.info(f"找到 {len(image_files)} 个图像文件")
    
    # 划分数据集
    train_images, valid_images, test_images = split_dataset(
        image_files, 
        train_ratio=train_ratio, 
        valid_ratio=valid_ratio, 
        test_ratio=test_ratio,
        seed=seed
    )
    
    logger.info(f"数据集划分: 训练集 {len(train_images)}张, 验证集 {len(valid_images)}张, 测试集 {len(test_images)}张")
    
    # 复制训练集
    logger.info("复制训练集...")
    for img_file in tqdm(train_images, desc="复制训练图像"):
        # 复制图像
        src_img_path = os.path.join(source_train_images, img_file)
        dst_img_path = os.path.join(target_train_images, img_file)
        shutil.copy2(src_img_path, dst_img_path)
        
        # 复制标签
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        src_label_path = os.path.join(source_train_labels, label_file)
        dst_label_path = os.path.join(target_train_labels, label_file)
        
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)
    
    # 复制验证集
    logger.info("复制验证集...")
    for img_file in tqdm(valid_images, desc="复制验证图像"):
        # 复制图像
        src_img_path = os.path.join(source_train_images, img_file)
        dst_img_path = os.path.join(target_valid_images, img_file)
        shutil.copy2(src_img_path, dst_img_path)
        
        # 复制标签
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        src_label_path = os.path.join(source_train_labels, label_file)
        dst_label_path = os.path.join(target_valid_labels, label_file)
        
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)
    
    # 复制测试集
    logger.info("复制测试集...")
    for img_file in tqdm(test_images, desc="复制测试图像"):
        # 复制图像
        src_img_path = os.path.join(source_train_images, img_file)
        dst_img_path = os.path.join(target_test_images, img_file)
        shutil.copy2(src_img_path, dst_img_path)
        
        # 复制标签
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        src_label_path = os.path.join(source_train_labels, label_file)
        dst_label_path = os.path.join(target_test_labels, label_file)
        
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)
    
    # 读取原始YAML配置
    config = read_yaml_config(source_yaml)
    
    # 更新配置文件
    config['train'] = './train/images'
    config['val'] = './valid/images'
    config['test'] = './test/images'
    
    # 保存新的YAML配置
    write_yaml_config(config, target_yaml)
    logger.info(f"已更新配置文件: {target_yaml}")
    
    logger.info("第二步处理完成！")
    return True

def step3_yolo_to_cnn(source_dir, target_dir, split_data=False, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1, seed=42):
    """第三步：将YOLOv8标准格式数据集转换为CNN所需的数据集格式
    
    Args:
        source_dir: 源目录，包含YOLOv8标准格式数据集
        target_dir: 目标目录，存储CNN格式数据集
        split_data: 是否需要重新分割数据集，通常设为False因为YOLOv8标准格式已经分割好了
        train_ratio: 训练集比例 (如果需要重新分割)
        valid_ratio: 验证集比例 (如果需要重新分割)
        test_ratio: 测试集比例 (如果需要重新分割)
        seed: 随机种子 (如果需要重新分割)
    """
    logger.info("=== 第三步：YOLOv8转CNN ===")
    
    # 确保源目录存在
    source_yaml = os.path.join(source_dir, 'data.yaml')
    if not os.path.exists(source_yaml):
        logger.error(f"源配置文件不存在: {source_yaml}")
        return False
    
    # 读取类别名称
    config = read_yaml_config(source_yaml)
    class_names = config.get('names', [])
    if not class_names:
        logger.error("无法获取类别名称")
        return False
    
    logger.info(f"类别名称: {class_names}")
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 创建类别目录
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(target_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    # 处理各个数据集分割
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(source_dir, split, 'images')
        label_dir = os.path.join(source_dir, split, 'labels')
        
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            logger.warning(f"{split}集图像或标签目录不存在，跳过: {img_dir}, {label_dir}")
            continue
        
        images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        logger.info(f"找到 {len(images)} 个{split}集图像")
        
        if not images:
            continue
            
        logger.info(f"处理{split}集...")
        
        for img_file in tqdm(images, desc=f"处理{split}集图像"):
            img_path = os.path.join(img_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"无法读取图像: {img_path}")
                continue
                
            height, width = img.shape[:2]
            
            # 读取标签文件
            if not os.path.exists(label_path):
                logger.warning(f"标签文件不存在: {label_path}")
                continue
            
            boxes = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    box_width = float(parts[3])
                    box_height = float(parts[4])
                    
                    # 转换为左上角和右下角坐标
                    x_min = center_x - box_width / 2
                    y_min = center_y - box_height / 2
                    x_max = center_x + box_width / 2
                    y_max = center_y + box_height / 2
                    
                    # 确保坐标在图像范围内
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(1, x_max)
                    y_max = min(1, y_max)
                    
                    # 只有当边界框有效时才添加
                    if x_min < x_max and y_min < y_max:
                        boxes.append([class_id, x_min, y_min, x_max, y_max])
            
            if not boxes:
                logger.warning(f"图像没有有效标签: {img_path}")
                continue
            
            # 对每个标注框，裁剪并保存到相应类别目录
            for i, box in enumerate(boxes):
                class_id, x_min, y_min, x_max, y_max = box
                
                if class_id >= len(class_names):
                    logger.warning(f"类别ID超出范围: {class_id}, 最大ID应为 {len(class_names)-1}")
                    continue
                    
                class_name = class_names[class_id]
                
                # 计算像素坐标
                x_min_px = int(x_min * width)
                y_min_px = int(y_min * height)
                x_max_px = int(x_max * width)
                y_max_px = int(y_max * height)
                
                # 确保至少有1个像素的宽高
                if x_max_px <= x_min_px or y_max_px <= y_min_px:
                    logger.warning(f"无效的边界框大小: {[x_min_px, y_min_px, x_max_px, y_max_px]}")
                    continue
                
                # 裁剪图像
                try:
                    crop = img[y_min_px:y_max_px, x_min_px:x_max_px]
                    if crop.size == 0:
                        logger.warning(f"裁剪得到空图像: {[y_min_px, y_max_px, x_min_px, x_max_px]}")
                        continue
                except Exception as e:
                    logger.error(f"裁剪图像时出错: {e}")
                    continue
                
                # 保存裁剪的图像
                output_dir = os.path.join(target_dir, split, class_name)
                output_file = f"{os.path.splitext(img_file)[0]}_crop{i}.jpg"
                output_path_full = os.path.join(output_dir, output_file)
                
                try:
                    cv2.imwrite(output_path_full, crop)
                except Exception as e:
                    logger.error(f"保存图像时出错 {output_path_full}: {e}")
    
    # 创建CNN数据集README文件
    readme_path = os.path.join(target_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("CNN数据集目录结构\n")
        f.write("------------------------\n\n")
        f.write("数据集按以下结构组织:\n")
        f.write("cnn_data_root/\n")
        f.write("  ├── train/\n")
        for class_name in class_names:
            f.write(f"  │   ├── {class_name}/\n")
        f.write("  ├── valid/\n")
        for class_name in class_names:
            f.write(f"  │   ├── {class_name}/\n")
        f.write("  └── test/\n")
        for class_name in class_names:
            f.write(f"      ├── {class_name}/\n")
    
    logger.info(f"已创建README文件: {readme_path}")
    logger.info("第三步处理完成！")
    return True

def step4_merge_categories(source_dir, target_dir):
    """第四步：将多种缺陷类别合并为单一"缺陷"类别
    
    Args:
        source_dir: 源目录，包含YOLOv8标准格式数据集（多类别）
        target_dir: 目标目录，存储合并类别后的YOLOv8数据集
    """
    logger.info("=== 第四步：合并缺陷类别 ===")
    
    # 确保源目录存在
    source_yaml = os.path.join(source_dir, 'data.yaml')
    if not os.path.exists(source_yaml):
        logger.error(f"源配置文件不存在: {source_yaml}")
        return False
    
    # 读取类别名称（用于日志输出）
    config = read_yaml_config(source_yaml)
    class_names = config.get('names', [])
    if not class_names:
        logger.error("无法获取类别名称")
        return False
    
    logger.info(f"原始类别: {class_names}")
    logger.info("合并为单一类别: 'defect'")
    
    # 创建目标目录结构
    for split in ['train', 'valid', 'test']:
        for subdir in ['images', 'labels']:
            target_subdir = os.path.join(target_dir, split, subdir)
            os.makedirs(target_subdir, exist_ok=True)
    
    # 处理各个数据集分割
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(source_dir, split, 'images')
        label_dir = os.path.join(source_dir, split, 'labels')
        
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            logger.warning(f"{split}集图像或标签目录不存在，跳过: {img_dir}, {label_dir}")
            continue
        
        images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        logger.info(f"找到 {len(images)} 个{split}集图像")
        
        if not images:
            continue
            
        logger.info(f"处理{split}集...")
        
        # 设置目标目录
        target_img_dir = os.path.join(target_dir, split, 'images')
        target_label_dir = os.path.join(target_dir, split, 'labels')
        
        for img_file in tqdm(images, desc=f"处理{split}集图像"):
            # 复制图像文件
            src_img_path = os.path.join(img_dir, img_file)
            dst_img_path = os.path.join(target_img_dir, img_file)
            
            try:
                shutil.copy2(src_img_path, dst_img_path)
            except Exception as e:
                logger.error(f"复制图像文件失败 {src_img_path}: {e}")
                continue
            
            # 处理标签文件
            label_file = os.path.splitext(img_file)[0] + '.txt'
            src_label_path = os.path.join(label_dir, label_file)
            dst_label_path = os.path.join(target_label_dir, label_file)
            
            if not os.path.exists(src_label_path):
                logger.warning(f"标签文件不存在: {src_label_path}")
                continue
            
            # 读取原始标签
            with open(src_label_path, 'r') as f:
                lines = f.readlines()
            
            # 将所有类别ID修改为0（单一"缺陷"类别）
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # 将类别ID替换为0
                    parts[0] = '0'
                    new_line = ' '.join(parts)
                    new_lines.append(new_line)
            
            # 写入新标签文件
            with open(dst_label_path, 'w') as f:
                f.write('\n'.join(new_lines))
    
    # 创建新的data.yaml文件
    new_config = config.copy()
    new_config['names'] = ['defect']  # 单一"缺陷"类别
    new_config['nc'] = 1  # 类别数为1
    
    # 保存新的YAML配置
    target_yaml = os.path.join(target_dir, 'data.yaml')
    write_yaml_config(new_config, target_yaml)
    logger.info(f"已创建配置文件: {target_yaml}")
    
    logger.info("第四步处理完成！")
    return True

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='数据集处理工具')
    
    parser.add_argument('--step1', action='store_true',
                        help='执行第一步：多边形转矩形')
    parser.add_argument('--step2', action='store_true',
                        help='执行第二步：划分数据集')
    parser.add_argument('--step3', action='store_true',
                        help='执行第三步：YOLOv8转CNN')
    parser.add_argument('--step4', action='store_true',
                        help='执行第四步：合并缺陷类别')
    parser.add_argument('--all', action='store_true',
                        help='执行所有步骤')
    
    parser.add_argument('--source1', type=str, default='datasets/yolo_v8/01_yolo_roboflow',
                        help='第一步源目录 (默认: datasets/yolo_v8/01_yolo_roboflow)')
    parser.add_argument('--target1', type=str, default='datasets/yolo_v8/02_yolo_rectangle',
                        help='第一步目标目录 (默认: datasets/yolo_v8/02_yolo_rectangle)')
    
    parser.add_argument('--source2', type=str, default='datasets/yolo_v8/02_yolo_rectangle',
                        help='第二步源目录 (默认: datasets/yolo_v8/02_yolo_rectangle)')
    parser.add_argument('--target2', type=str, default='datasets/yolo_v8/03_yolo_standard',
                        help='第二步目标目录 (默认: datasets/yolo_v8/03_yolo_standard)')
    
    parser.add_argument('--source3', type=str, default='datasets/yolo_v8/03_yolo_standard',
                        help='第三步源目录 (默认: datasets/yolo_v8/03_yolo_standard)')
    parser.add_argument('--target3', type=str, default='datasets/cnn/resnet50_standard',
                        help='第三步目标目录 (默认: datasets/cnn/resnet50_standard)')
    
    parser.add_argument('--source4', type=str, default='datasets/yolo_v8/03_yolo_standard',
                        help='第四步源目录 (默认: datasets/yolo_v8/03_yolo_standard)')
    parser.add_argument('--target4', type=str, default='datasets/yolo_v8/04_yolo_nocategory',
                        help='第四步目标目录 (默认: datasets/yolo_v8/04_yolo_nocategory)')
    
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例 (默认: 0.7)')
    parser.add_argument('--valid-ratio', type=float, default=0.2,
                        help='验证集比例 (默认: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='测试集比例 (默认: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 如果未指定任何步骤，默认执行所有步骤
    if not (args.step1 or args.step2 or args.step3 or args.step4 or args.all):
        args.all = True
    
    # 执行第一步：多边形转矩形
    if args.step1 or args.all:
        if not step1_polygon_to_rectangle(args.source1, args.target1):
            logger.error("第一步处理失败，程序退出")
            return 1
    
    # 执行第二步：划分数据集
    if args.step2 or args.all:
        if not step2_split_dataset(args.source2, args.target2, 
                                  train_ratio=args.train_ratio,
                                  valid_ratio=args.valid_ratio,
                                  test_ratio=args.test_ratio,
                                  seed=args.seed):
            logger.error("第二步处理失败，程序退出")
            return 1
    
    # 执行第三步：YOLOv8转CNN
    if args.step3 or args.all:
        if not step3_yolo_to_cnn(args.source3, args.target3,
                               split_data=False,
                               train_ratio=args.train_ratio,
                               valid_ratio=args.valid_ratio,
                               test_ratio=args.test_ratio,
                               seed=args.seed):
            logger.error("第三步处理失败，程序退出")
            return 1
    
    # 执行第四步：合并缺陷类别
    if args.step4 or args.all:
        if not step4_merge_categories(args.source4, args.target4):
            logger.error("第四步处理失败，程序退出")
            return 1
    
    logger.info("所有处理步骤完成！")
    return 0

# 为兼容旧代码，将原convert_yolo_to_cnn.py中核心函数保留为独立函数
def create_cnn_dataset(yolo_dataset_path, output_path, class_names, split_data=True):
    """创建CNN数据集（兼容旧接口用于被其他模块调用）
    
    Args:
        yolo_dataset_path: YOLO数据集路径
        output_path: 输出路径
        class_names: 类别名称
        split_data: 是否分割数据集
    """
    logger.info(f"使用兼容接口创建CNN数据集: {yolo_dataset_path} -> {output_path}")
    logger.info(f"类别名称: {class_names}")
    
    # 调用第三步实现
    return step3_yolo_to_cnn(
        source_dir=yolo_dataset_path,
        target_dir=output_path,
        split_data=split_data
    )

# 为兼容旧代码，保留move_yolo_files函数
def move_yolo_files(yolo_dataset_path, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    """将YOLO数据集按比例分割并移动文件（兼容旧接口）
    
    Args:
        yolo_dataset_path: YOLO数据集路径
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    logger.info(f"使用兼容接口分割YOLO数据集: {yolo_dataset_path}")
    
    # 由于旧接口结构不太一样，这里需要进行特殊处理
    # 旧接口假设数据在images和labels目录，新接口假设数据在train/images和train/labels目录
    
    # 检查旧格式数据结构
    images_dir = os.path.join(yolo_dataset_path, 'images')
    labels_dir = os.path.join(yolo_dataset_path, 'labels')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        logger.error(f"旧格式目录不存在: {images_dir}, {labels_dir}")
        return False
    
    # 创建目标目录结构
    for split in ['train', 'valid', 'test']:
        for subdir in ['images', 'labels']:
            target_subdir = os.path.join(yolo_dataset_path, split, subdir)
            os.makedirs(target_subdir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 分割数据集
    train_files, valid_files, test_files = split_dataset(
        image_files, 
        train_ratio=train_ratio, 
        valid_ratio=valid_ratio, 
        test_ratio=test_ratio
    )
    
    # 移动文件
    splits = {
        'train': train_files,
        'valid': valid_files,
        'test': test_files
    }
    
    for split, files in splits.items():
        logger.info(f"移动{split}集文件...")
        target_img_dir = os.path.join(yolo_dataset_path, split, 'images')
        target_label_dir = os.path.join(yolo_dataset_path, split, 'labels')
        
        for img_file in tqdm(files, desc=f"移动{split}集"):
            # 移动图像文件
            src_img_path = os.path.join(images_dir, img_file)
            dst_img_path = os.path.join(target_img_dir, img_file)
            
            # 移动标签文件 (如果存在)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            src_label_path = os.path.join(labels_dir, label_file)
            dst_label_path = os.path.join(target_label_dir, label_file)
            
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)
            
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dst_label_path)
    
    logger.info("文件移动完成")
    return True

# 添加数据集检查功能，替代原有check_dataset.py
def check_dataset(data_dir='datasets', yolo_dir='yolov8', cnn_dir='cnn'):
    """检查数据集是否符合预期结构
    
    Args:
        data_dir: 数据目录
        yolo_dir: YOLO数据集子目录
        cnn_dir: CNN数据集子目录
    """
    logger.info("=== 检查数据集 ===")
    
    yolo_path = os.path.join(data_dir, yolo_dir)
    cnn_path = os.path.join(data_dir, cnn_dir)
    
    # 检查YOLO数据集
    logger.info(f"检查YOLOv8数据集: {yolo_path}")
    
    if not os.path.exists(yolo_path):
        logger.error(f"YOLOv8数据集目录不存在: {yolo_path}")
    else:
        # 检查data.yaml
        yaml_path = os.path.join(yolo_path, 'data.yaml')
        if not os.path.exists(yaml_path):
            logger.error(f"YOLOv8配置文件不存在: {yaml_path}")
        else:
            # 读取配置文件
            config = read_yaml_config(yaml_path)
            class_names = config.get('names', [])
            num_classes = config.get('nc', len(class_names))
            
            logger.info(f"类别数量: {num_classes}")
            logger.info(f"类别名称: {class_names}")
            
            # 检查训练集/验证集/测试集
            for split in ['train', 'valid', 'test']:
                img_dir = os.path.join(yolo_path, split, 'images')
                label_dir = os.path.join(yolo_path, split, 'labels')
                
                if not os.path.exists(img_dir):
                    logger.warning(f"{split}集图像目录不存在: {img_dir}")
                else:
                    img_count = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
                    logger.info(f"{split}集图像数量: {img_count}")
                
                if not os.path.exists(label_dir):
                    logger.warning(f"{split}集标签目录不存在: {label_dir}")
                else:
                    label_count = len([f for f in os.listdir(label_dir) if f.endswith('.txt')])
                    logger.info(f"{split}集标签数量: {label_count}")
    
    # 检查CNN数据集
    logger.info(f"检查CNN数据集: {cnn_path}")
    
    if not os.path.exists(cnn_path):
        logger.error(f"CNN数据集目录不存在: {cnn_path}")
    else:
        # 检查目录结构
        class_counts = {}
        
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(cnn_path, split)
            
            if not os.path.exists(split_dir):
                logger.warning(f"CNN {split}集目录不存在: {split_dir}")
                continue
            
            # 统计每个类别的样本数量
            class_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
            
            if not class_dirs:
                logger.warning(f"CNN {split}集没有类别目录: {split_dir}")
                continue
                
            logger.info(f"CNN {split}集类别: {class_dirs}")
            
            class_counts[split] = {}
            total_samples = 0
            
            for class_name in class_dirs:
                class_dir = os.path.join(split_dir, class_name)
                sample_count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
                class_counts[split][class_name] = sample_count
                total_samples += sample_count
            
            logger.info(f"CNN {split}集总样本数: {total_samples}")
            
            # 打印每个类别的样本数量
            for class_name, count in class_counts[split].items():
                logger.info(f"  - {class_name}: {count}张图像")
    
    logger.info("数据集检查完成！")
    return True

# 支持从命令行直接调用数据集检查功能
def check_dataset_main():
    """数据集检查主函数"""
    parser = argparse.ArgumentParser(description='检查数据集结构')
    parser.add_argument('--data-dir', type=str, default='datasets',
                        help='数据目录')
    parser.add_argument('--yolo-dir', type=str, default='yolov8',
                        help='YOLO数据集子目录')
    parser.add_argument('--cnn-dir', type=str, default='cnn',
                        help='CNN数据集子目录')
    
    args = parser.parse_args()
    
    return check_dataset(args.data_dir, args.yolo_dir, args.cnn_dir)

if __name__ == '__main__':
    sys.exit(main())
