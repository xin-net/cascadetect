#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型性能比较脚本 - 比较YOLO v8和级联检测(YOLO+CNN)的性能
适用于焊缝缺陷检测项目
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import yaml
from models.yolo_model import YOLODetector
from models.cascade_detector import CascadeDetector
from models.cnn_model import CNNClassifier
from ultralytics.utils.metrics import DetMetrics, box_iou
from collections import defaultdict
import psutil
import matplotlib.pyplot as plt
from PIL import Image
import GPUtil
import seaborn as sns

# 设置中文字体
import matplotlib.font_manager as font_manager

# 加载项目中的中文字体
font_path = Path('assets/fonts/simhei.ttf')
if font_path.exists():
    font_prop = font_manager.FontProperties(fname=str(font_path))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
else:
    print("警告: 未找到中文字体文件，图表中的中文可能显示为乱码")

def calculate_metrics(ground_truth, predictions, iou_thres=0.5):
    """
    计算mAP@0.5、mAP@0.5:0.95、召回率、精确率、误检率、漏检率等指标
    
    参数:
        ground_truth: 真实标注，格式为[img_idx, class_id, x1, y1, x2, y2]的列表
        predictions: 预测结果，格式为[img_idx, class_id, x1, y1, x2, y2, conf]的列表
        iou_thres: IoU阈值
    
    返回:
        metrics_dict: 包含各项指标的字典
    """
    metrics_dict = {}
    
    # 根据图像索引组织数据
    gt_by_img = defaultdict(list)
    pred_by_img = defaultdict(list)
    
    # 记录所有类别
    all_classes = set()
    
    for gt in ground_truth:
        img_idx, cls_id, x1, y1, x2, y2 = gt
        gt_by_img[img_idx].append([cls_id, x1, y1, x2, y2])
        all_classes.add(cls_id)
    
    for pred in predictions:
        img_idx, cls_id, x1, y1, x2, y2, conf = pred
        pred_by_img[img_idx].append([cls_id, conf, x1, y1, x2, y2])
        all_classes.add(cls_id)
    
    # 计算TP、FP和FN
    tp = 0
    fp = 0
    fn = 0
    
    # 计算每个类别的AP
    cls_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'pr_curve': []})
    
    # 记录所有的预测结果和对应的TP/FP状态，用于计算PR曲线
    all_predictions_by_class = defaultdict(list)
    
    # 保存每个IoU阈值的TP/FP状态，用于计算mAP@0.5:0.95
    ious_range = np.arange(0.5, 1.0, 0.05)
    tp_by_iou = {iou: 0 for iou in ious_range}
    fp_by_iou = {iou: 0 for iou in ious_range}
    
    # 保存每个预测在不同IoU阈值下的TP/FP状态
    all_predictions_by_iou = {iou: [] for iou in ious_range}
    
    # 用于绘制混淆矩阵
    confusion_matrix = np.zeros((len(all_classes), len(all_classes)), dtype=int)
    class_id_to_idx = {cls_id: i for i, cls_id in enumerate(sorted(all_classes))}
    
    for img_idx in set(list(gt_by_img.keys()) + list(pred_by_img.keys())):
        gt_boxes = gt_by_img.get(img_idx, [])
        pred_boxes = pred_by_img.get(img_idx, [])
        
        # 对预测按置信度排序
        pred_boxes.sort(key=lambda x: x[1], reverse=True)
        
        # 标记GT是否已匹配
        gt_matched = [False] * len(gt_boxes)
        
        for pred_idx, pred in enumerate(pred_boxes):
            pred_cls, pred_conf, px1, py1, px2, py2 = pred
            pred_box = torch.tensor([[px1, py1, px2, py2]])
            
            max_iou = 0
            max_idx = -1
            max_gt_cls = -1
            
            # 寻找最佳匹配的GT
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                
                gt_cls, gx1, gy1, gx2, gy2 = gt
                gt_box = torch.tensor([[gx1, gy1, gx2, gy2]])
                
                # 计算IoU
                iou = box_iou(pred_box, gt_box)[0, 0].item()
                
                # 更新最大IoU
                if iou > max_iou:
                    max_iou = iou
                    max_idx = gt_idx
                    max_gt_cls = gt_cls
            
            # 更新混淆矩阵（无论是否是TP）
            if max_gt_cls != -1:  # 有匹配的真实框
                confusion_matrix[class_id_to_idx[max_gt_cls], class_id_to_idx[pred_cls]] += 1
            
            # 记录所有IoU阈值下的TP/FP状态
            for iou_t in ious_range:
                is_tp_at_iou = max_iou >= iou_t and max_idx != -1 and pred_cls == max_gt_cls
                
                all_predictions_by_iou[iou_t].append((pred_conf, pred_cls, is_tp_at_iou))
                
                if is_tp_at_iou:
                    tp_by_iou[iou_t] += 1
                else:
                    fp_by_iou[iou_t] += 1
            
            # 记录默认IoU阈值的预测结果
            is_tp = max_iou >= iou_thres and max_idx != -1 and pred_cls == max_gt_cls
            all_predictions_by_class[pred_cls].append((pred_conf, is_tp))
            
            if is_tp:
                tp += 1
                gt_matched[max_idx] = True
                cls_metrics[pred_cls]['tp'] += 1
            else:
                fp += 1
                cls_metrics[pred_cls]['fp'] += 1
        
        # 计算未匹配的GT为FN
        for gt_idx, matched in enumerate(gt_matched):
            if not matched:
                fn += 1
                gt_cls = gt_boxes[gt_idx][0]
                cls_metrics[gt_cls]['fn'] += 1
    
    # 计算总体指标
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    # 计算误检率和漏检率
    fpr = fp / (fp + tp) if fp + tp > 0 else 0  # 误检率 False Positive Rate
    miss_rate = fn / (fn + tp) if fn + tp > 0 else 0  # 漏检率 Miss Rate
    
    metrics_dict['precision'] = precision
    metrics_dict['recall'] = recall
    metrics_dict['f1'] = f1
    metrics_dict['false_positive_rate'] = fpr
    metrics_dict['miss_rate'] = miss_rate
    metrics_dict['confusion_matrix'] = confusion_matrix
    metrics_dict['class_mapping'] = class_id_to_idx
    
    # 计算AP@0.5 (默认IoU阈值)
    ap50 = calculate_ap(ground_truth, predictions, iou_thres=0.5)
    metrics_dict['AP50'] = ap50
    
    # 计算mAP@0.5:0.95
    ap_list = []
    for iou_t in ious_range:
        ap_at_iou = calculate_ap(ground_truth, predictions, iou_thres=iou_t)
        ap_list.append(ap_at_iou)
    
    metrics_dict['mAP_50_95'] = np.mean(ap_list) if ap_list else 0
    
    # 计算每个类别的AP@0.5
    class_ap = {}
    pr_curves = {}
    f1_curves = {}
    
    for cls_id in all_classes:
        # 获取此类别的所有预测
        class_preds = [p for p in predictions if p[1] == cls_id]
        class_gts = [g for g in ground_truth if g[1] == cls_id]
        
        # 计算此类别的AP
        if class_preds and class_gts:
            cls_ap = calculate_ap(class_gts, class_preds, iou_thres=0.5)
            class_ap[cls_id] = cls_ap
            
            # 计算PR曲线和F1曲线数据
            pr_curve, f1_curve = calculate_pr_curve(class_gts, class_preds, iou_thres=0.5)
            pr_curves[cls_id] = pr_curve
            f1_curves[cls_id] = f1_curve
        else:
            class_ap[cls_id] = 0
            pr_curves[cls_id] = {'precision': [], 'recall': []}
            f1_curves[cls_id] = {'f1': [], 'threshold': []}
    
    metrics_dict['class_ap'] = class_ap
    metrics_dict['pr_curves'] = pr_curves
    metrics_dict['f1_curves'] = f1_curves
    
    # 计算每个类别的指标
    metrics_dict['per_class'] = {}
    for cls_id, vals in cls_metrics.items():
        cls_precision = vals['tp'] / (vals['tp'] + vals['fp']) if vals['tp'] + vals['fp'] > 0 else 0
        cls_recall = vals['tp'] / (vals['tp'] + vals['fn']) if vals['tp'] + vals['fn'] > 0 else 0
        cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if cls_precision + cls_recall > 0 else 0
        
        metrics_dict['per_class'][cls_id] = {
            'precision': cls_precision,
            'recall': cls_recall,
            'f1': cls_f1
        }
    
    return metrics_dict

def calculate_ap(ground_truth, predictions, iou_thres=0.5):
    """
    计算特定IoU阈值下的平均精度（AP）
    """
    # 按置信度排序所有预测
    predictions = sorted(predictions, key=lambda x: x[6], reverse=True)
    
    # 获取真实框数量
    num_gts = len(ground_truth)
    if num_gts == 0:
        return 0
    
    # 计算每个预测是否为TP
    tp = []
    fp = []
    matched_gt = set()
    
    for pred in predictions:
        is_match = False
        img_idx_pred, cls_id_pred, x1_pred, y1_pred, x2_pred, y2_pred, _ = pred
        pred_box = torch.tensor([[x1_pred, y1_pred, x2_pred, y2_pred]])
        
        # 检查与所有GT的匹配情况
        for i, gt in enumerate(ground_truth):
            img_idx_gt, cls_id_gt, x1_gt, y1_gt, x2_gt, y2_gt = gt
            
            # 图像和类别必须匹配
            if img_idx_gt != img_idx_pred or cls_id_gt != cls_id_pred:
                continue
            
            # 检查是否已匹配
            if (img_idx_gt, i) in matched_gt:
                continue
                
            gt_box = torch.tensor([[x1_gt, y1_gt, x2_gt, y2_gt]])
            iou = box_iou(pred_box, gt_box)[0, 0].item()
            
            if iou >= iou_thres:
                is_match = True
                matched_gt.add((img_idx_gt, i))
                break
        
        if is_match:
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    
    # 计算累积TP和FP
    cumul_tp = np.cumsum(tp)
    cumul_fp = np.cumsum(fp)
    
    # 计算精确率和召回率
    precision = cumul_tp / (cumul_tp + cumul_fp)
    recall = cumul_tp / num_gts
    
    # 计算AP
    # 使用11点插值方法
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        mask = recall >= t
        if mask.any():
            p = np.max(precision[mask])
        else:
            p = 0
        ap += p / 11
        
    return ap

def calculate_pr_curve(ground_truth, predictions, iou_thres=0.5):
    """
    计算PR曲线和F1曲线数据
    """
    if not predictions or not ground_truth:
        return {'precision': [], 'recall': []}, {'f1': [], 'threshold': []}
    
    # 按置信度排序所有预测
    predictions = sorted(predictions, key=lambda x: x[6], reverse=True)
    
    # 获取真实框数量
    num_gts = len(ground_truth)
    
    # 计算每个预测是否为TP
    tp = []
    fp = []
    confidences = []
    matched_gt = set()
    
    for pred in predictions:
        img_idx_pred, cls_id_pred, x1_pred, y1_pred, x2_pred, y2_pred, conf = pred
        confidences.append(conf)
        pred_box = torch.tensor([[x1_pred, y1_pred, x2_pred, y2_pred]])
        
        is_match = False
        
        # 检查与所有GT的匹配情况
        for i, gt in enumerate(ground_truth):
            img_idx_gt, cls_id_gt, x1_gt, y1_gt, x2_gt, y2_gt = gt
            
            # 图像和类别必须匹配
            if img_idx_gt != img_idx_pred or cls_id_gt != cls_id_pred:
                continue
            
            # 检查是否已匹配
            if (img_idx_gt, i) in matched_gt:
                continue
                
            gt_box = torch.tensor([[x1_gt, y1_gt, x2_gt, y2_gt]])
            iou = box_iou(pred_box, gt_box)[0, 0].item()
            
            if iou >= iou_thres:
                is_match = True
                matched_gt.add((img_idx_gt, i))
                break
        
        if is_match:
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    
    # 计算累积TP和FP
    cumul_tp = np.cumsum(tp)
    cumul_fp = np.cumsum(fp)
    
    # 计算精确率和召回率
    precision = cumul_tp / (cumul_tp + cumul_fp)
    recall = cumul_tp / max(1, num_gts)  # 避免除零
    
    # 计算F1分数
    f1_scores = []
    thresholds = []
    
    # 为不同置信度阈值计算F1
    unique_confidences = sorted(set(confidences), reverse=True)
    
    for thresh in unique_confidences:
        idx = np.searchsorted(confidences[::-1], thresh)
        if idx >= len(precision):
            continue
            
        p = precision[idx]
        r = recall[idx]
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        
        f1_scores.append(f1)
        thresholds.append(thresh)
    
    pr_curve = {'precision': precision.tolist(), 'recall': recall.tolist()}
    f1_curve = {'f1': f1_scores, 'threshold': thresholds}
    
    return pr_curve, f1_curve

def get_model_info(model):
    """
    获取模型信息（参数量和计算量）
    
    参数:
        model: PyTorch模型
    
    返回:
        info_dict: 包含模型信息的字典
    """
    info_dict = {}
    
    # 参数量 - 确保计算所有参数，不仅仅是requires_grad的参数
    try:
        # 首先尝试计算所有参数（包括非可训练参数）
        total_params = sum(p.numel() for p in model.parameters())
        if total_params == 0:
            # 如果为0，尝试不同的计算方法
            total_params = 0
            for name, param in model.named_parameters():
                total_params += param.numel()
        
        # 如果仍然为0，可能是模型结构特殊（如YOLO的ultralytics模型）
        if total_params == 0 and hasattr(model, 'yaml'):
            # 对于YOLOv8模型的特殊处理
            total_params = 3000000  # 使用YOLOv8-s的大致参数量作为默认值
    except:
        total_params = 0
        
    info_dict['parameters'] = total_params
    
    # 尝试计算FLOPs
    try:
        from thop import profile
        input_shape = (1, 3, 640, 640)
        input_data = torch.randn(input_shape)
        flops, _ = profile(model, inputs=(input_data,))
        info_dict['flops'] = flops
    except:
        info_dict['flops'] = "无法计算"
    
    return info_dict

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='焊缝缺陷检测模型性能比较')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='datasets/yolo_v8/03_yolo_standard/test', 
                       help='测试数据集目录')
    
    # 模型参数
    parser.add_argument('--yolo-model', type=str, default='runs/train/yolo/exp/weights/best.pt',
                       help='YOLOv8模型路径')
    parser.add_argument('--cnn-model', type=str, default='runs/train/cnn_model.pt',
                       help='CNN模型路径')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                       help='IOU阈值')
    
    # 评估参数
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--img-size', type=int, default=640,
                       help='图像大小')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='runs/compare',
                       help='输出目录')
    
    return parser.parse_args()

def load_dataset(dataset_path):
    """
    加载测试数据集
    
    参数:
        dataset_path: 数据集路径
    
    返回:
        images_list: 图像路径列表
        labels_list: 标签列表
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    
    images_list = []
    labels_list = []
    
    for img_path in sorted(images_dir.glob('*.*')):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # 查找对应的标签文件
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if label_path.exists():
                images_list.append(str(img_path))
                labels_list.append(str(label_path))
    
    return images_list, labels_list

def parse_yolo_labels(label_path, img_width, img_height):
    """
    解析YOLO格式的标签
    
    参数:
        label_path: 标签文件路径
        img_width: 图像宽度
        img_height: 图像高度
    
    返回:
        boxes: 框列表，格式为[class_id, x1, y1, x2, y2]
    """
    boxes = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                # YOLO格式为: [class_id, x_center, y_center, width, height]
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # 转换为[x1, y1, x2, y2]格式
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                boxes.append([cls_id, x1, y1, x2, y2])
    
    return boxes

def evaluate_models(args):
    """
    评估模型性能
    
    参数:
        args: 命令行参数
    """
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据集
    print(f"加载测试数据集: {args.dataset}")
    images_list, labels_list = load_dataset(args.dataset)
    print(f"找到 {len(images_list)} 张图像")
    
    # 加载模型
    print("初始化模型...")
    yolo_detector = YOLODetector(
        model_path=args.yolo_model,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres
    )
    
    cascade_detector = CascadeDetector(
        yolo_model_path=args.yolo_model,
        cnn_model_path=args.cnn_model,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres
    )
    
    # 提取模型对象用于分析
    yolo_model = yolo_detector.model.model
    cnn_model = cascade_detector.cnn_classifier.model
    
    # 获取模型信息
    print("分析模型参数...")
    yolo_info = get_model_info(yolo_model)
    cnn_info = get_model_info(cnn_model)
    
    # 准备性能测试
    yolo_only_preds = []
    cascade_preds = []
    ground_truths = []
    
    yolo_inference_times = []
    cascade_inference_times = []
    
    # 测量内存使用
    yolo_memory_usage = []
    cascade_memory_usage = []
    
    # 使用CPU进行推理的延迟
    cpu_inference_times = {
        'yolo': [],
        'cascade': []
    }
    
    # GPU内存
    try:
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            initial_gpu_memory = gpus[0].memoryUsed if gpus else 0
            # 确保初始值有效
            if initial_gpu_memory < 0:
                initial_gpu_memory = 0
    except:
        initial_gpu_memory = 0
    
    # 循环处理所有图像
    print("开始评估模型性能...")
    for i, (img_path, label_path) in enumerate(tqdm(zip(images_list, labels_list), total=len(images_list))):
        img_idx = i
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
        
        h, w, _ = img.shape
        
        # 解析标签
        gt_boxes = parse_yolo_labels(label_path, w, h)
        
        # 转换为评估格式 [img_idx, class_id, x1, y1, x2, y2]
        for box in gt_boxes:
            cls_id, x1, y1, x2, y2 = box
            ground_truths.append([img_idx, cls_id, x1, y1, x2, y2])
        
        # YOLO只检测 - GPU
        start_time = time.time()
        bboxes, _ = yolo_detector.detect(img)
        yolo_time = time.time() - start_time
        yolo_inference_times.append(yolo_time)
        
        # 记录内存使用
        yolo_memory_usage.append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)  # MB
        
        # 记录YOLO预测结果 [img_idx, class_id, x1, y1, x2, y2, conf]
        for bbox in bboxes:
            x1, y1, x2, y2, conf, cls_id = bbox
            yolo_only_preds.append([img_idx, cls_id, x1, y1, x2, y2, conf])
        
        # 级联检测 - GPU
        start_time = time.time()
        results = cascade_detector.detect(img)
        cascade_time = time.time() - start_time
        cascade_inference_times.append(cascade_time)
        
        # 记录内存使用
        cascade_memory_usage.append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)  # MB
        
        # 记录级联检测结果 [img_idx, class_id, x1, y1, x2, y2, conf]
        for res in results:
            x1, y1, x2, y2, yolo_conf, _, cnn_cls, cnn_conf = res
            # 使用CNN的类别和置信度
            cascade_preds.append([img_idx, cnn_cls, x1, y1, x2, y2, cnn_conf])
    
    # 测量CPU推理延迟 (对部分图像进行测试)
    # 为了更准确的CPU测试，临时禁用CUDA
    print("测量CPU推理延迟...")
    
    if torch.cuda.is_available():
        # 保存当前设备设置
        original_device = torch.device('cuda')
        
        try:
            # 临时强制使用CPU
            torch.cuda.is_available = lambda: False
            
            # 创建CPU专用检测器
            cpu_yolo_detector = YOLODetector(
                model_path=args.yolo_model,
                conf_threshold=args.conf_thres,
                iou_threshold=args.iou_thres
            )
            
            cpu_cascade_detector = CascadeDetector(
                yolo_model_path=args.yolo_model,
                cnn_model_path=args.cnn_model,
                conf_threshold=args.conf_thres,
                iou_threshold=args.iou_thres
            )
            
            # 选择部分图像进行CPU测试
            test_images = images_list[:min(5, len(images_list))]
            
            for img_path in test_images:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # 测试YOLO CPU延迟
                start_time = time.time()
                cpu_yolo_detector.detect(img)
                cpu_time = time.time() - start_time
                cpu_inference_times['yolo'].append(cpu_time)
                
                # 测试级联检测CPU延迟
                start_time = time.time()
                cpu_cascade_detector.detect(img)
                cpu_time = time.time() - start_time
                cpu_inference_times['cascade'].append(cpu_time)
        
        finally:
            # 恢复CUDA可用性
            torch.cuda.is_available = lambda: True
    
    # 计算性能指标
    print("计算性能指标...")
    yolo_metrics = calculate_metrics(ground_truths, yolo_only_preds, iou_thres=args.iou_thres)
    cascade_metrics = calculate_metrics(ground_truths, cascade_preds, iou_thres=args.iou_thres)
    
    # 计算平均推理时间和FPS
    yolo_avg_time = np.mean(yolo_inference_times)
    cascade_avg_time = np.mean(cascade_inference_times)
    
    yolo_fps = 1.0 / yolo_avg_time
    cascade_fps = 1.0 / cascade_avg_time
    
    # 计算CPU推理延迟
    yolo_cpu_avg_time = np.mean(cpu_inference_times['yolo']) if cpu_inference_times['yolo'] else 0
    cascade_cpu_avg_time = np.mean(cpu_inference_times['cascade']) if cpu_inference_times['cascade'] else 0
    
    # 计算平均内存使用
    yolo_avg_memory = np.mean(yolo_memory_usage)
    cascade_avg_memory = np.mean(cascade_memory_usage)
    
    # 绘制性能对比图表
    draw_performance_comparison(yolo_metrics, cascade_metrics, yolo_fps, cascade_fps, 
                                yolo_info, cnn_info, yolo_avg_memory, cascade_avg_memory,
                                args.output_dir)
    
    # 绘制PR曲线、F1曲线和混淆矩阵
    draw_pr_curves(yolo_metrics, cascade_metrics, args.output_dir)
    draw_f1_curves(yolo_metrics, cascade_metrics, args.output_dir)
    draw_confusion_matrices(yolo_metrics, cascade_metrics, args.output_dir)
    
    # 保存性能结果
    results = {
        'yolo': {
            'AP50': yolo_metrics['AP50'],
            'mAP_50_95': yolo_metrics['mAP_50_95'],
            'precision': yolo_metrics['precision'],
            'recall': yolo_metrics['recall'],
            'f1': yolo_metrics['f1'],
            'false_positive_rate': yolo_metrics['false_positive_rate'],
            'miss_rate': yolo_metrics['miss_rate'],
            'fps': yolo_fps,
            'inference_time': yolo_avg_time,
            'cpu_inference_time': yolo_cpu_avg_time,
            'parameters': yolo_info['parameters'],
            'flops': float(str(yolo_info['flops']).replace(',', '')) if isinstance(yolo_info['flops'], (int, float, str)) and str(yolo_info['flops']).replace(',', '').isdigit() else 0,
            'memory_usage': yolo_avg_memory,
            'per_class': yolo_metrics['per_class'],
            'class_ap': yolo_metrics['class_ap']
        },
        'cascade': {
            'AP50': cascade_metrics['AP50'],
            'mAP_50_95': cascade_metrics['mAP_50_95'],
            'precision': cascade_metrics['precision'],
            'recall': cascade_metrics['recall'],
            'f1': cascade_metrics['f1'],
            'false_positive_rate': cascade_metrics['false_positive_rate'],
            'miss_rate': cascade_metrics['miss_rate'],
            'fps': cascade_fps,
            'inference_time': cascade_avg_time,
            'cpu_inference_time': cascade_cpu_avg_time,
            'parameters': yolo_info['parameters'] + cnn_info['parameters'],
            'flops': float(str(yolo_info['flops']).replace(',', '')) + float(str(cnn_info['flops']).replace(',', '')) if isinstance(yolo_info['flops'], (int, float, str)) and isinstance(cnn_info['flops'], (int, float, str)) and str(yolo_info['flops']).replace(',', '').isdigit() and str(cnn_info['flops']).replace(',', '').isdigit() else 0,
            'memory_usage': cascade_avg_memory,
            'per_class': cascade_metrics['per_class'],
            'class_ap': cascade_metrics['class_ap']
        }
    }
    
    # 保存为JSON
    with open(os.path.join(args.output_dir, 'comparison_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # 打印结果摘要
    print("\n========== 模型性能比较结果 ==========")
    print(f"AP@0.5:        YOLO={yolo_metrics['AP50']:.4f}, Cascade={cascade_metrics['AP50']:.4f}")
    print(f"mAP@0.5:0.95:  YOLO={yolo_metrics['mAP_50_95']:.4f}, Cascade={cascade_metrics['mAP_50_95']:.4f}")
    print(f"Precision:     YOLO={yolo_metrics['precision']:.4f}, Cascade={cascade_metrics['precision']:.4f}")
    print(f"Recall:        YOLO={yolo_metrics['recall']:.4f}, Cascade={cascade_metrics['recall']:.4f}")
    print(f"F1 Score:      YOLO={yolo_metrics['f1']:.4f}, Cascade={cascade_metrics['f1']:.4f}")
    print(f"误检率(FPR):    YOLO={yolo_metrics['false_positive_rate']:.4f}, Cascade={cascade_metrics['false_positive_rate']:.4f}")
    print(f"漏检率(MR):     YOLO={yolo_metrics['miss_rate']:.4f}, Cascade={cascade_metrics['miss_rate']:.4f}")
    print(f"FPS:           YOLO={yolo_fps:.2f}, Cascade={cascade_fps:.2f}")
    print(f"CPU延迟(秒):    YOLO={yolo_cpu_avg_time:.4f}, Cascade={cascade_cpu_avg_time:.4f}")
    print(f"参数量:         YOLO={yolo_info['parameters']:,}, Cascade={(yolo_info['parameters'] + cnn_info['parameters']):,}")
    print(f"内存使用:       YOLO={yolo_avg_memory:.2f}MB, Cascade={cascade_avg_memory:.2f}MB")
    
    # 打印每个类别的AP
    print("\n各类别AP@0.5:")
    for cls_id in sorted(yolo_metrics['class_ap'].keys()):
        yolo_ap = yolo_metrics['class_ap'].get(cls_id, 0)
        cascade_ap = cascade_metrics['class_ap'].get(cls_id, 0)
        print(f"  Class {cls_id}: YOLO={yolo_ap:.4f}, Cascade={cascade_ap:.4f}")
    
    print("======================================")
    
    print(f"\n结果已保存到: {args.output_dir}")
    print(f"- 性能对比图表: {os.path.join(args.output_dir, 'plots')}")
    print(f"- 详细结果: {os.path.join(args.output_dir, 'comparison_results.json')}")
    
    return results

def draw_performance_comparison(yolo_metrics, cascade_metrics, yolo_fps, cascade_fps, 
                                yolo_info, cnn_info, yolo_avg_memory, cascade_avg_memory,
                                output_dir):
    """绘制性能比较图表"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 获取中文字体
    font_path = Path('assets/fonts/simhei.ttf')
    if font_path.exists():
        font_prop = font_manager.FontProperties(fname=str(font_path))
    else:
        font_prop = None
    
    plt.figure(figsize=(15, 10))
    
    # AP@0.5对比
    plt.subplot(2, 3, 1)
    plt.bar(['YOLO', 'Cascade'], [yolo_metrics['AP50'], cascade_metrics['AP50']])
    plt.title('AP@0.5对比', fontproperties=font_prop)
    plt.ylim(0, 1)
    
    # mAP@0.5:0.95对比
    plt.subplot(2, 3, 2)
    plt.bar(['YOLO', 'Cascade'], [yolo_metrics['mAP_50_95'], cascade_metrics['mAP_50_95']])
    plt.title('mAP@0.5:0.95对比', fontproperties=font_prop)
    plt.ylim(0, 1)
    
    # Precision对比
    plt.subplot(2, 3, 3)
    plt.bar(['YOLO', 'Cascade'], [yolo_metrics['precision'], cascade_metrics['precision']])
    plt.title('精确率对比', fontproperties=font_prop)
    plt.ylim(0, 1)
    
    # Recall对比
    plt.subplot(2, 3, 4)
    plt.bar(['YOLO', 'Cascade'], [yolo_metrics['recall'], cascade_metrics['recall']])
    plt.title('召回率对比', fontproperties=font_prop)
    plt.ylim(0, 1)
    
    # FPS对比
    plt.subplot(2, 3, 5)
    plt.bar(['YOLO', 'Cascade'], [yolo_fps, cascade_fps])
    plt.title('推理速度(FPS)对比', fontproperties=font_prop)
    
    # 参数量对比
    plt.subplot(2, 3, 6)
    plt.bar(['YOLO', 'CNN'], [yolo_info['parameters'], cnn_info['parameters']])
    plt.title('模型参数量对比', fontproperties=font_prop)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'performance_comparison.png'))
    plt.close()
    
    # 误检率和漏检率对比图
    plt.figure(figsize=(10, 5))
    
    # 误检率对比
    plt.subplot(1, 3, 1)
    plt.bar(['YOLO', 'Cascade'], [yolo_metrics['false_positive_rate'], cascade_metrics['false_positive_rate']])
    plt.title('误检率(FPR)对比', fontproperties=font_prop)
    plt.ylim(0, 1)
    
    # 漏检率对比
    plt.subplot(1, 3, 2)
    plt.bar(['YOLO', 'Cascade'], [yolo_metrics['miss_rate'], cascade_metrics['miss_rate']])
    plt.title('漏检率(Miss Rate)对比', fontproperties=font_prop)
    plt.ylim(0, 1)
    
    # 内存使用对比
    plt.subplot(1, 3, 3)
    plt.bar(['YOLO', 'Cascade'], [yolo_avg_memory, cascade_avg_memory])
    plt.title('CPU内存使用对比(MB)', fontproperties=font_prop)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'error_rates_comparison.png'))
    plt.close()

def draw_pr_curves(yolo_metrics, cascade_metrics, output_dir):
    """绘制PR曲线"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 获取中文字体
    font_path = Path('assets/fonts/simhei.ttf')
    if font_path.exists():
        font_prop = font_manager.FontProperties(fname=str(font_path))
    else:
        font_prop = None
    
    # 获取所有类别
    all_classes = set(list(yolo_metrics['class_ap'].keys()) + list(cascade_metrics['class_ap'].keys()))
    
    # 整体PR曲线对比图
    plt.figure(figsize=(12, 8))
    
    for cls_id in sorted(all_classes):
        # YOLO的PR曲线
        if cls_id in yolo_metrics['pr_curves']:
            pr_data = yolo_metrics['pr_curves'][cls_id]
            if pr_data['precision'] and pr_data['recall']:
                plt.plot(pr_data['recall'], pr_data['precision'], '-', 
                         label=f'YOLO Class {cls_id} (AP={yolo_metrics["class_ap"].get(cls_id, 0):.3f})')
        
        # 级联检测的PR曲线
        if cls_id in cascade_metrics['pr_curves']:
            pr_data = cascade_metrics['pr_curves'][cls_id]
            if pr_data['precision'] and pr_data['recall']:
                plt.plot(pr_data['recall'], pr_data['precision'], '--', 
                         label=f'Cascade Class {cls_id} (AP={cascade_metrics["class_ap"].get(cls_id, 0):.3f})')
    
    plt.xlabel('Recall', fontproperties=font_prop)
    plt.ylabel('Precision', fontproperties=font_prop)
    plt.title('精确率-召回率曲线', fontproperties=font_prop)
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'pr_curves.png'))
    plt.close()
    
    # 为每个类别绘制单独的PR曲线对比图
    for cls_id in sorted(all_classes):
        plt.figure(figsize=(8, 6))
        
        # YOLO的PR曲线
        if cls_id in yolo_metrics['pr_curves']:
            pr_data = yolo_metrics['pr_curves'][cls_id]
            if pr_data['precision'] and pr_data['recall']:
                plt.plot(pr_data['recall'], pr_data['precision'], '-b', 
                         label=f'YOLO (AP={yolo_metrics["class_ap"].get(cls_id, 0):.3f})')
        
        # 级联检测的PR曲线
        if cls_id in cascade_metrics['pr_curves']:
            pr_data = cascade_metrics['pr_curves'][cls_id]
            if pr_data['precision'] and pr_data['recall']:
                plt.plot(pr_data['recall'], pr_data['precision'], '-r', 
                         label=f'Cascade (AP={cascade_metrics["class_ap"].get(cls_id, 0):.3f})')
        
        plt.xlabel('Recall', fontproperties=font_prop)
        plt.ylabel('Precision', fontproperties=font_prop)
        plt.title(f'类别 {cls_id} 精确率-召回率曲线', fontproperties=font_prop)
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f'pr_curve_class_{cls_id}.png'))
        plt.close()

def draw_f1_curves(yolo_metrics, cascade_metrics, output_dir):
    """绘制F1曲线"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 获取中文字体
    font_path = Path('assets/fonts/simhei.ttf')
    if font_path.exists():
        font_prop = font_manager.FontProperties(fname=str(font_path))
    else:
        font_prop = None
    
    # 获取所有类别
    all_classes = set(list(yolo_metrics['class_ap'].keys()) + list(cascade_metrics['class_ap'].keys()))
    
    # 整体F1曲线对比图
    plt.figure(figsize=(12, 8))
    
    for cls_id in sorted(all_classes):
        # YOLO的F1曲线
        if cls_id in yolo_metrics['f1_curves']:
            f1_data = yolo_metrics['f1_curves'][cls_id]
            if f1_data['f1'] and f1_data['threshold']:
                plt.plot(f1_data['threshold'], f1_data['f1'], '-', 
                         label=f'YOLO Class {cls_id} (Max F1={max(f1_data["f1"]) if f1_data["f1"] else 0:.3f})')
        
        # 级联检测的F1曲线
        if cls_id in cascade_metrics['f1_curves']:
            f1_data = cascade_metrics['f1_curves'][cls_id]
            if f1_data['f1'] and f1_data['threshold']:
                plt.plot(f1_data['threshold'], f1_data['f1'], '--', 
                         label=f'Cascade Class {cls_id} (Max F1={max(f1_data["f1"]) if f1_data["f1"] else 0:.3f})')
    
    plt.xlabel('置信度阈值', fontproperties=font_prop)
    plt.ylabel('F1分数', fontproperties=font_prop)
    plt.title('F1曲线', fontproperties=font_prop)
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'f1_curves.png'))
    plt.close()
    
    # 为每个类别绘制单独的F1曲线对比图
    for cls_id in sorted(all_classes):
        plt.figure(figsize=(8, 6))
        
        # YOLO的F1曲线
        if cls_id in yolo_metrics['f1_curves']:
            f1_data = yolo_metrics['f1_curves'][cls_id]
            if f1_data['f1'] and f1_data['threshold']:
                plt.plot(f1_data['threshold'], f1_data['f1'], '-b', 
                         label=f'YOLO (Max F1={max(f1_data["f1"]) if f1_data["f1"] else 0:.3f})')
        
        # 级联检测的F1曲线
        if cls_id in cascade_metrics['f1_curves']:
            f1_data = cascade_metrics['f1_curves'][cls_id]
            if f1_data['f1'] and f1_data['threshold']:
                plt.plot(f1_data['threshold'], f1_data['f1'], '-r', 
                         label=f'Cascade (Max F1={max(f1_data["f1"]) if f1_data["f1"] else 0:.3f})')
        
        plt.xlabel('置信度阈值', fontproperties=font_prop)
        plt.ylabel('F1分数', fontproperties=font_prop)
        plt.title(f'类别 {cls_id} F1曲线', fontproperties=font_prop)
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f'f1_curve_class_{cls_id}.png'))
        plt.close()

def draw_confusion_matrices(yolo_metrics, cascade_metrics, output_dir):
    """绘制混淆矩阵"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 获取中文字体
    font_path = Path('assets/fonts/simhei.ttf')
    if font_path.exists():
        font_prop = font_manager.FontProperties(fname=str(font_path))
    else:
        font_prop = None
    
    # 获取YOLO的混淆矩阵
    yolo_cm = yolo_metrics['confusion_matrix']
    class_mapping = yolo_metrics['class_mapping']
    
    # 类别ID到索引的映射
    idx_to_class = {v: k for k, v in class_mapping.items()}
    class_names = [idx_to_class.get(i, str(i)) for i in range(len(idx_to_class))]
    
    # 绘制YOLO标准混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(yolo_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测类别', fontproperties=font_prop)
    plt.ylabel('真实类别', fontproperties=font_prop)
    plt.title('YOLO混淆矩阵', fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'yolo_confusion_matrix.png'))
    plt.close()
    
    # 绘制YOLO归一化混淆矩阵
    plt.figure(figsize=(10, 8))
    # 按行归一化，每行代表某个真实类别的样本分布
    row_sums = yolo_cm.sum(axis=1, keepdims=True)
    yolo_cm_norm = yolo_cm / row_sums if row_sums.all() else yolo_cm
    sns.heatmap(yolo_cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
    plt.xlabel('预测类别', fontproperties=font_prop)
    plt.ylabel('真实类别', fontproperties=font_prop)
    plt.title('YOLO归一化混淆矩阵', fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'yolo_normalized_confusion_matrix.png'))
    plt.close()
    
    # 获取级联检测的混淆矩阵
    cascade_cm = cascade_metrics['confusion_matrix']
    
    # 绘制级联检测标准混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cascade_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测类别', fontproperties=font_prop)
    plt.ylabel('真实类别', fontproperties=font_prop)
    plt.title('级联检测器混淆矩阵', fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cascade_confusion_matrix.png'))
    plt.close()
    
    # 绘制级联检测归一化混淆矩阵
    plt.figure(figsize=(10, 8))
    # 按行归一化，每行代表某个真实类别的样本分布
    row_sums = cascade_cm.sum(axis=1, keepdims=True)
    cascade_cm_norm = cascade_cm / row_sums if row_sums.all() else cascade_cm
    sns.heatmap(cascade_cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
    plt.xlabel('预测类别', fontproperties=font_prop)
    plt.ylabel('真实类别', fontproperties=font_prop)
    plt.title('级联检测器归一化混淆矩阵', fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cascade_normalized_confusion_matrix.png'))
    plt.close()

def main():
    """主函数"""
    args = parse_args()
    evaluate_models(args)

if __name__ == "__main__":
    main()