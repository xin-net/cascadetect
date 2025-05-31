#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
焊缝缺陷检测系统检测脚本
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from models.cascade_detector import CascadeDetector
from models.yolo_model import YOLODetector
from pathlib import Path
import time
from PIL import Image, ImageDraw, ImageFont

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='焊缝缺陷检测系统检测脚本')
    
    # 输入参数
    parser.add_argument('--source', type=str, required=True,
                        help='输入图像或视频的路径，或者0表示摄像头')
    
    # 模型参数
    parser.add_argument('--yolo-model', type=str, default='runs/train/yolo_model.pt',
                        help='YOLOv8模型路径')
    parser.add_argument('--cnn-model', type=str, default='runs/train/cnn_model.pt',
                        help='CNN模型路径')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IOU阈值')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='类别数量，如果不指定则自动从模型中检测')
    parser.add_argument('--class-names', type=str, default=None,
                        help='类别名称，用逗号分隔，例如：正常,气孔,裂纹,夹渣,其他')
    parser.add_argument('--yolo-only', action='store_true',
                        help='仅使用YOLO模型进行单级检测，不使用CNN进行二级分类')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='runs/detect',
                        help='输出目录')
    parser.add_argument('--save-txt', action='store_true',
                        help='保存检测结果为文本文件')
    parser.add_argument('--save-csv', action='store_true',
                        help='保存检测结果为CSV文件')
    parser.add_argument('--save-json', action='store_true',
                        help='保存检测结果为JSON文件')
    parser.add_argument('--view-img', action='store_true',
                        help='显示检测结果')
    
    # 性能参数
    parser.add_argument('--img-size', type=int, default=640,
                        help='输入图像大小')
    
    return parser.parse_args()


def detect_image(detector, image_path, output_dir, view_img=False, save_txt=False, save_csv=False, save_json=False):
    """
    检测单张图像
    
    参数:
        detector: 级联检测器
        image_path: 图像路径
        output_dir: 输出目录
        view_img: 是否显示图像
        save_txt: 是否保存结果为文本文件
        save_csv: 是否保存结果为CSV文件
        save_json: 是否保存结果为JSON文件
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像 {image_path}")
        return
    
    # 执行检测
    results = detector.detect(image)
    
    # 可视化结果
    vis_image = detector.visualize(image, results)
    
    # 准备输出路径
    filename = os.path.basename(image_path)
    basename = os.path.splitext(filename)[0]
    
    # 保存可视化图像
    output_img_path = os.path.join(output_dir, f"{basename}_result.jpg")
    cv2.imwrite(output_img_path, vis_image)
    print(f"结果已保存到 {output_img_path}")
    
    # 保存文本结果
    if save_txt:
        output_txt_path = os.path.join(output_dir, f"{basename}_result.txt")
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write("# 格式: x1 y1 x2 y2 YOLO置信度 YOLO类别ID CNN类别ID CNN置信度 CNN类别名称\n")
            for res in results:
                x1, y1, x2, y2, yolo_conf, yolo_cls, cnn_cls, cnn_conf = res
                # 获取CNN类别名称
                cnn_class_id = min(int(cnn_cls), len(detector.class_names) - 1)
                cnn_class_name = detector.class_names[cnn_class_id]
                # 格式: <x1> <y1> <x2> <y2> <yolo_conf> <yolo_class_id> <cnn_class_id> <cnn_conf> <cnn_class_name>
                f.write(f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {yolo_conf:.4f} {int(yolo_cls)} {int(cnn_cls)} {cnn_conf:.4f} {cnn_class_name}\n")
        print(f"检测结果已保存到 {output_txt_path}")
    
    # 保存CSV结果
    if save_csv:
        import csv
        output_csv_path = os.path.join(output_dir, f"{basename}_result.csv")
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            # 写入表头
            csv_writer.writerow(["x1", "y1", "x2", "y2", "宽度", "高度", "中心X", "中心Y", 
                                 "YOLO置信度", "YOLO类别ID", "CNN类别ID", "CNN置信度", "CNN类别名称"])
            for res in results:
                x1, y1, x2, y2, yolo_conf, yolo_cls, cnn_cls, cnn_conf = res
                # 计算额外信息
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # 获取CNN类别名称
                cnn_class_id = min(int(cnn_cls), len(detector.class_names) - 1)
                cnn_class_name = detector.class_names[cnn_class_id]
                # 写入行
                csv_writer.writerow([
                    f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", 
                    f"{width:.1f}", f"{height:.1f}", f"{center_x:.1f}", f"{center_y:.1f}",
                    f"{yolo_conf:.4f}", int(yolo_cls), int(cnn_cls), f"{cnn_conf:.4f}", cnn_class_name
                ])
        print(f"CSV格式检测结果已保存到 {output_csv_path}")
    
    # 保存JSON结果
    if save_json:
        import json
        output_json_path = os.path.join(output_dir, f"{basename}_result.json")
        json_data = {
            "image": image_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detections": []
        }
        for res in results:
            x1, y1, x2, y2, yolo_conf, yolo_cls, cnn_cls, cnn_conf = res
            # 计算额外信息
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            # 获取CNN类别名称
            cnn_class_id = min(int(cnn_cls), len(detector.class_names) - 1)
            cnn_class_name = detector.class_names[cnn_class_id]
            
            # 添加检测结果
            json_data["detections"].append({
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "width": float(width),
                    "height": float(height),
                    "center_x": float(center_x),
                    "center_y": float(center_y)
                },
                "yolo": {
                    "confidence": float(yolo_conf),
                    "class_id": int(yolo_cls)
                },
                "cnn": {
                    "confidence": float(cnn_conf),
                    "class_id": int(cnn_cls),
                    "class_name": cnn_class_name
                }
            })
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"JSON格式检测结果已保存到 {output_json_path}")
    
    # 显示结果
    if view_img:
        cv2.imshow('Detection Result', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return results


def detect_video(detector, video_path, output_dir, view_img=False, save_txt=False, save_csv=False, save_json=False):
    """
    检测视频
    
    参数:
        detector: 级联检测器
        video_path: 视频路径
        output_dir: 输出目录
        view_img: 是否显示视频
        save_txt: 是否保存结果为文本文件
        save_csv: 是否保存结果为CSV文件
        save_json: 是否保存结果为JSON文件
    """
    # 打开视频
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
        video_name = 'camera'
    else:
        cap = cv2.VideoCapture(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if not cap.isOpened():
        print(f"无法打开视频 {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建输出视频写入器
    output_video_path = os.path.join(output_dir, f"{video_name}_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 预备文本输出文件
    txt_file = None
    csv_writer = None
    csv_file = None
    json_data = {
        "video": video_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fps": fps,
        "frames": []
    }
    
    if save_txt:
        txt_path = os.path.join(output_dir, f"{video_name}_result.txt")
        txt_file = open(txt_path, 'w', encoding='utf-8')
        txt_file.write("# 格式: 帧号 x1 y1 x2 y2 YOLO置信度 YOLO类别ID CNN类别ID CNN置信度 CNN类别名称\n")
    
    if save_csv:
        import csv
        csv_path = os.path.join(output_dir, f"{video_name}_result.csv")
        csv_file = open(csv_path, 'w', encoding='utf-8', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["帧号", "x1", "y1", "x2", "y2", "宽度", "高度", "中心X", "中心Y", 
                            "YOLO置信度", "YOLO类别ID", "CNN类别ID", "CNN置信度", "CNN类别名称"])
    
    # 处理视频帧
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 执行检测
        results = detector.detect(frame)
        
        # 保存文本结果
        if save_txt and txt_file:
            for res in results:
                x1, y1, x2, y2, yolo_conf, yolo_cls, cnn_cls, cnn_conf = res
                # 获取CNN类别名称
                cnn_class_id = min(int(cnn_cls), len(detector.class_names) - 1)
                cnn_class_name = detector.class_names[cnn_class_id]
                # 格式: <帧号> <x1> <y1> <x2> <y2> <yolo_conf> <yolo_class_id> <cnn_class_id> <cnn_conf> <cnn_class_name>
                txt_file.write(f"{frame_idx} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {yolo_conf:.4f} {int(yolo_cls)} {int(cnn_cls)} {cnn_conf:.4f} {cnn_class_name}\n")
        
        # 保存CSV结果
        if save_csv and csv_writer:
            for res in results:
                x1, y1, x2, y2, yolo_conf, yolo_cls, cnn_cls, cnn_conf = res
                # 计算额外信息
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # 获取CNN类别名称
                cnn_class_id = min(int(cnn_cls), len(detector.class_names) - 1)
                cnn_class_name = detector.class_names[cnn_class_id]
                # 写入行
                csv_writer.writerow([
                    frame_idx, f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", 
                    f"{width:.1f}", f"{height:.1f}", f"{center_x:.1f}", f"{center_y:.1f}",
                    f"{yolo_conf:.4f}", int(yolo_cls), int(cnn_cls), f"{cnn_conf:.4f}", cnn_class_name
                ])
        
        # 保存JSON数据
        if save_json:
            frame_data = {
                "frame_idx": frame_idx,
                "detections": []
            }
            for res in results:
                x1, y1, x2, y2, yolo_conf, yolo_cls, cnn_cls, cnn_conf = res
                # 计算额外信息
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # 获取CNN类别名称
                cnn_class_id = min(int(cnn_cls), len(detector.class_names) - 1)
                cnn_class_name = detector.class_names[cnn_class_id]
                
                # 添加检测结果
                frame_data["detections"].append({
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width": float(width),
                        "height": float(height),
                        "center_x": float(center_x),
                        "center_y": float(center_y)
                    },
                    "yolo": {
                        "confidence": float(yolo_conf),
                        "class_id": int(yolo_cls)
                    },
                    "cnn": {
                        "confidence": float(cnn_conf),
                        "class_id": int(cnn_cls),
                        "class_name": cnn_class_name
                    }
                })
            json_data["frames"].append(frame_data)
        
        # 可视化结果
        vis_frame = detector.visualize(frame, results)
        
        # 写入输出视频
        writer.write(vis_frame)
        
        # 显示结果
        if view_img:
            cv2.imshow('Detection Result', vis_frame)
            if cv2.waitKey(1) == ord('q'):  # 按q键退出
                break
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"处理帧 {frame_idx}")
    
    # 关闭所有文件
    if save_txt and txt_file:
        txt_file.close()
        print(f"文本检测结果已保存到 {txt_path}")
    
    if save_csv and csv_file:
        csv_file.close()
        print(f"CSV格式检测结果已保存到 {csv_path}")
    
    if save_json:
        json_path = os.path.join(output_dir, f"{video_name}_result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"JSON格式检测结果已保存到 {json_path}")
    
    # 释放资源
    cap.release()
    writer.release()
    if view_img:
        cv2.destroyAllWindows()
    
    print(f"结果视频已保存到 {output_video_path}")


def detect_image_yolo_only(detector, image_path, output_dir, class_names=None, view_img=False, save_txt=False, save_csv=False, save_json=False):
    """
    仅使用YOLO模型检测单张图像
    
    参数:
        detector: YOLO检测器
        image_path: 图像路径
        output_dir: 输出目录
        class_names: 类别名称列表
        view_img: 是否显示图像
        save_txt: 是否保存结果为文本文件
        save_csv: 是否保存结果为CSV文件
        save_json: 是否保存结果为JSON文件
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像 {image_path}")
        return
    
    # 执行检测
    bboxes, crops = detector.detect(image)
    
    # 如果未指定类别名称，使用默认的COCO类别名称
    if class_names is None:
        class_names = [
            '正常', '气孔', '裂纹', '夹渣', '其他缺陷',
            '未知缺陷1', '未知缺陷2', '未知缺陷3', '未知缺陷4', '未知缺陷5'
        ]
    
    # 可视化结果
    vis_image = image.copy()
    
    # 颜色映射
    colors = [
        (0, 255, 0),    # 正常 - 绿色
        (0, 165, 255),  # 气孔 - 橙色
        (0, 0, 255),    # 裂纹 - 红色
        (255, 0, 0),    # 夹渣 - 蓝色
        (255, 0, 255)   # 其他类型 - 紫色
    ]
    
    # 扩展颜色列表
    if len(class_names) > len(colors):
        import random
        for _ in range(len(class_names) - len(colors)):
            colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    
    # 如果没有检测结果，直接返回原图
    if not bboxes:
        pass
    else:
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
        
        # 绘制检测框和标签
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2, conf, cls_id = bbox
            
            # 确保类别索引在有效范围内
            cls_id = min(int(cls_id), len(class_names) - 1)
            
            # 获取颜色和类别名称
            color = colors[cls_id % len(colors)]
            label = f"{class_names[cls_id]}: {conf:.2f}"
            
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
    
    # 准备输出路径
    filename = os.path.basename(image_path)
    basename = os.path.splitext(filename)[0]
    
    # 保存可视化图像
    output_img_path = os.path.join(output_dir, f"{basename}_yolo_result.jpg")
    cv2.imwrite(output_img_path, vis_image)
    print(f"结果已保存到 {output_img_path}")
    
    # 保存文本结果
    if save_txt:
        output_txt_path = os.path.join(output_dir, f"{basename}_yolo_result.txt")
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write("# 格式: x1 y1 x2 y2 YOLO置信度 YOLO类别ID YOLO类别名称\n")
            for bbox in bboxes:
                x1, y1, x2, y2, conf, cls_id = bbox
                # 确保类别索引在有效范围内
                cls_id = min(int(cls_id), len(class_names) - 1)
                class_name = class_names[cls_id]
                # 格式: <x1> <y1> <x2> <y2> <conf> <class_id> <class_name>
                f.write(f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {conf:.4f} {int(cls_id)} {class_name}\n")
        print(f"检测结果已保存到 {output_txt_path}")
    
    # 保存CSV结果
    if save_csv:
        import csv
        output_csv_path = os.path.join(output_dir, f"{basename}_yolo_result.csv")
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            # 写入表头
            csv_writer.writerow(["x1", "y1", "x2", "y2", "宽度", "高度", "中心X", "中心Y", 
                                 "YOLO置信度", "YOLO类别ID", "YOLO类别名称"])
            for bbox in bboxes:
                x1, y1, x2, y2, conf, cls_id = bbox
                # 计算额外信息
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # 确保类别索引在有效范围内
                cls_id = min(int(cls_id), len(class_names) - 1)
                class_name = class_names[cls_id]
                # 写入行
                csv_writer.writerow([
                    f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", 
                    f"{width:.1f}", f"{height:.1f}", f"{center_x:.1f}", f"{center_y:.1f}",
                    f"{conf:.4f}", int(cls_id), class_name
                ])
        print(f"CSV格式检测结果已保存到 {output_csv_path}")
    
    # 保存JSON结果
    if save_json:
        import json
        output_json_path = os.path.join(output_dir, f"{basename}_yolo_result.json")
        json_data = {
            "image": image_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detections": []
        }
        for bbox in bboxes:
            x1, y1, x2, y2, conf, cls_id = bbox
            # 计算额外信息
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            # 确保类别索引在有效范围内
            cls_id = min(int(cls_id), len(class_names) - 1)
            class_name = class_names[cls_id]
            
            # 添加检测结果
            json_data["detections"].append({
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "width": float(width),
                    "height": float(height),
                    "center_x": float(center_x),
                    "center_y": float(center_y)
                },
                "yolo": {
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "class_name": class_name
                }
            })
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"JSON格式检测结果已保存到 {output_json_path}")
    
    # 显示结果
    if view_img:
        cv2.imshow('YOLO Detection Result', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return bboxes


def detect_video_yolo_only(detector, video_path, output_dir, class_names=None, view_img=False, save_txt=False, save_csv=False, save_json=False):
    """
    仅使用YOLO模型检测视频
    
    参数:
        detector: YOLO检测器
        video_path: 视频路径
        output_dir: 输出目录
        class_names: 类别名称列表
        view_img: 是否显示视频
        save_txt: 是否保存结果为文本文件
        save_csv: 是否保存结果为CSV文件
        save_json: 是否保存结果为JSON文件
    """
    # 打开视频
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
        video_name = 'camera'
    else:
        cap = cv2.VideoCapture(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if not cap.isOpened():
        print(f"无法打开视频 {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建输出视频写入器
    output_video_path = os.path.join(output_dir, f"{video_name}_yolo_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 如果未指定类别名称，使用默认的COCO类别名称
    if class_names is None:
        class_names = [
            '正常', '气孔', '裂纹', '夹渣', '其他缺陷',
            '未知缺陷1', '未知缺陷2', '未知缺陷3', '未知缺陷4', '未知缺陷5'
        ]
    
    # 颜色映射
    colors = [
        (0, 255, 0),    # 正常 - 绿色
        (0, 165, 255),  # 气孔 - 橙色
        (0, 0, 255),    # 裂纹 - 红色
        (255, 0, 0),    # 夹渣 - 蓝色
        (255, 0, 255)   # 其他类型 - 紫色
    ]
    
    # 扩展颜色列表
    if len(class_names) > len(colors):
        import random
        for _ in range(len(class_names) - len(colors)):
            colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    
    # 预备文本输出文件
    txt_file = None
    csv_writer = None
    csv_file = None
    json_data = {
        "video": video_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fps": fps,
        "frames": []
    }
    
    if save_txt:
        txt_path = os.path.join(output_dir, f"{video_name}_yolo_result.txt")
        txt_file = open(txt_path, 'w', encoding='utf-8')
        txt_file.write("# 格式: 帧号 x1 y1 x2 y2 YOLO置信度 YOLO类别ID YOLO类别名称\n")
    
    if save_csv:
        import csv
        csv_path = os.path.join(output_dir, f"{video_name}_yolo_result.csv")
        csv_file = open(csv_path, 'w', encoding='utf-8', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["帧号", "x1", "y1", "x2", "y2", "宽度", "高度", "中心X", "中心Y", 
                            "YOLO置信度", "YOLO类别ID", "YOLO类别名称"])
    
    # 处理视频帧
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 执行检测
        bboxes, _ = detector.detect(frame)
        
        # 保存文本结果
        if save_txt and txt_file:
            for bbox in bboxes:
                x1, y1, x2, y2, conf, cls_id = bbox
                # 确保类别索引在有效范围内
                cls_id = min(int(cls_id), len(class_names) - 1)
                class_name = class_names[cls_id]
                # 格式: <帧号> <x1> <y1> <x2> <y2> <conf> <class_id> <class_name>
                txt_file.write(f"{frame_idx} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {conf:.4f} {int(cls_id)} {class_name}\n")
        
        # 保存CSV结果
        if save_csv and csv_writer:
            for bbox in bboxes:
                x1, y1, x2, y2, conf, cls_id = bbox
                # 计算额外信息
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # 确保类别索引在有效范围内
                cls_id = min(int(cls_id), len(class_names) - 1)
                class_name = class_names[cls_id]
                # 写入行
                csv_writer.writerow([
                    frame_idx, f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", 
                    f"{width:.1f}", f"{height:.1f}", f"{center_x:.1f}", f"{center_y:.1f}",
                    f"{conf:.4f}", int(cls_id), class_name
                ])
        
        # 保存JSON数据
        if save_json:
            frame_data = {
                "frame_idx": frame_idx,
                "detections": []
            }
            for bbox in bboxes:
                x1, y1, x2, y2, conf, cls_id = bbox
                # 计算额外信息
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # 确保类别索引在有效范围内
                cls_id = min(int(cls_id), len(class_names) - 1)
                class_name = class_names[cls_id]
                
                # 添加检测结果
                frame_data["detections"].append({
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width": float(width),
                        "height": float(height),
                        "center_x": float(center_x),
                        "center_y": float(center_y)
                    },
                    "yolo": {
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                        "class_name": class_name
                    }
                })
            json_data["frames"].append(frame_data)
        
        # 可视化结果
        vis_frame = frame.copy()
        
        # 绘制检测框
        for bbox in bboxes:
            x1, y1, x2, y2, conf, cls_id = bbox
            
            # 确保类别索引在有效范围内
            cls_id = min(int(cls_id), len(class_names) - 1)
            
            # 获取颜色和类别名称
            color = colors[cls_id % len(colors)]
            label = f"{class_names[cls_id]}: {conf:.2f}"
            
            # 绘制矩形框
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 绘制标签背景
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                vis_frame, 
                (int(x1), int(y1) - text_size[1] - 5),
                (int(x1) + text_size[0], int(y1)), 
                color, 
                -1
            )
            
            # 绘制标签
            cv2.putText(
                vis_frame, 
                label, 
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
        
        # 写入输出视频
        writer.write(vis_frame)
        
        # 显示结果
        if view_img:
            cv2.imshow('YOLO Detection Result', vis_frame)
            if cv2.waitKey(1) == ord('q'):  # 按q键退出
                break
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"处理帧 {frame_idx}")
    
    # 关闭所有文件
    if save_txt and txt_file:
        txt_file.close()
        print(f"文本检测结果已保存到 {txt_path}")
    
    if save_csv and csv_file:
        csv_file.close()
        print(f"CSV格式检测结果已保存到 {csv_path}")
    
    if save_json:
        json_path = os.path.join(output_dir, f"{video_name}_yolo_result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"JSON格式检测结果已保存到 {json_path}")
    
    # 释放资源
    cap.release()
    writer.release()
    if view_img:
        cv2.destroyAllWindows()
    
    print(f"结果视频已保存到 {output_video_path}")


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理类别名称参数
    class_names = None
    if args.class_names:
        class_names = args.class_names.split(',')
        print(f"使用自定义类别名称: {class_names}")
    
    # 判断是否使用YOLO单级检测
    if args.yolo_only:
        print("使用YOLO单级检测模式...")
        
        # 检查YOLO模型文件是否存在
        if not os.path.exists(args.yolo_model):
            print(f"错误: YOLOv8模型文件 {args.yolo_model} 不存在")
            return
        
        # 初始化YOLO检测器
        detector = YOLODetector(
            model_path=args.yolo_model,
            conf_threshold=args.conf_thres,
            iou_threshold=args.iou_thres
        )
        
        # 判断输入是图像还是视频
        source = args.source
        is_video = False
        
        if source.isdigit():  # 摄像头
            is_video = True
        elif os.path.isfile(source):
            # 判断文件类型
            ext = os.path.splitext(source)[1].lower()
            is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        else:
            print(f"错误: 输入源 {source} 不存在")
            return
        
        # 执行检测
        print(f"开始YOLO单级检测 {'视频' if is_video else '图像'}...")
        
        if is_video:
            detect_video_yolo_only(
                detector, 
                source, 
                args.output_dir,
                class_names,
                args.view_img,
                args.save_txt, 
                args.save_csv,
                args.save_json
            )
        else:
            detect_image_yolo_only(
                detector, 
                source, 
                args.output_dir,
                class_names,
                args.view_img, 
                args.save_txt, 
                args.save_csv,
                args.save_json
            )
        
        print("YOLO单级检测完成！")
        return
    
    # 级联检测模式
    # 检查CNN模型文件是否存在
    if not os.path.exists(args.cnn_model):
        print(f"错误: CNN模型文件 {args.cnn_model} 不存在")
        return
    
    # 初始化级联检测器
    print("初始化级联检测器...")
    detector = CascadeDetector(
        yolo_model_path=args.yolo_model,
        cnn_model_path=args.cnn_model,
        num_classes=args.num_classes,  # 可以为None，让检测器自动检测
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
        cnn_input_size=(224, 224),  # 固定CNN输入大小
        class_names=class_names  # 传入自定义类别名称
    )
    
    # 判断输入是图像还是视频
    source = args.source
    is_video = False
    
    if source.isdigit():  # 摄像头
        is_video = True
    elif os.path.isfile(source):
        # 判断文件类型
        ext = os.path.splitext(source)[1].lower()
        is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    else:
        print(f"错误: 输入源 {source} 不存在")
        return
    
    # 执行检测
    print(f"开始检测 {'视频' if is_video else '图像'}...")
    
    if is_video:
        detect_video(
            detector, 
            source, 
            args.output_dir, 
            args.view_img,
            args.save_txt, 
            args.save_csv,
            args.save_json
        )
    else:
        detect_image(
            detector, 
            source, 
            args.output_dir, 
            args.view_img, 
            args.save_txt, 
            args.save_csv,
            args.save_json
        )
    
    print("检测完成！")


if __name__ == '__main__':
    main() 