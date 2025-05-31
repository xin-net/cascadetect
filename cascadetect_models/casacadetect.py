#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cascadetect - 焊缝缺陷多模型级联检测系统 - 主程序
提供统一的命令行接口调用各功能模块
"""

import os
import sys
import argparse
import logging

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="焊缝缺陷多模型级联检测系统")
    subparsers = parser.add_subparsers(dest="command", help="选择操作命令")
    
    # 数据处理命令
    data_parser = subparsers.add_parser("data", help="数据处理相关命令")
    data_subparsers = data_parser.add_subparsers(dest="data_command")
    
    # 数据转换
    convert_parser = data_subparsers.add_parser("convert", help="多阶段数据集处理")
    convert_parser.add_argument("--step1", action="store_true", help="第一步: 多边形标注转矩形框")
    convert_parser.add_argument("--step2", action="store_true", help="第二步: 划分数据集")
    convert_parser.add_argument("--step3", action="store_true", help="第三步: YOLOv8标准格式转CNN格式")
    convert_parser.add_argument("--step4", action="store_true", help="第四步: 将多类别缺陷合并为单一类别")
    convert_parser.add_argument("--all", action="store_true", help="执行所有步骤")
    convert_parser.add_argument("--yolo-dir", default="datasets/yolov8", help="YOLO数据集目录")
    convert_parser.add_argument("--output-dir", default="datasets/cnn", help="输出目录")
    convert_parser.add_argument("--source1", type=str, help="第一步源目录")
    convert_parser.add_argument("--target1", type=str, help="第一步目标目录")
    convert_parser.add_argument("--source2", type=str, help="第二步源目录")
    convert_parser.add_argument("--target2", type=str, help="第二步目标目录")
    convert_parser.add_argument("--source3", type=str, help="第三步源目录")
    convert_parser.add_argument("--target3", type=str, help="第三步目标目录")
    convert_parser.add_argument("--source4", type=str, help="第四步源目录")
    convert_parser.add_argument("--target4", type=str, help="第四步目标目录")
    convert_parser.add_argument("--train-ratio", type=float, default=0.7, help="训练集比例 (默认: 0.7)")
    convert_parser.add_argument("--valid-ratio", type=float, default=0.2, help="验证集比例 (默认: 0.2)")
    convert_parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例 (默认: 0.1)")
    convert_parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    
    # 数据检查
    check_parser = data_subparsers.add_parser("check", help="检查数据集")
    check_parser.add_argument("--yolo-dir", default="yolo_v8", help="YOLO数据集目录")
    check_parser.add_argument("--cnn-dir", default="cnn", help="CNN数据集目录")
    check_parser.add_argument("--data-dir", default="datasets", help="数据根目录")
    
    # 详细数据集检查
    check_dataset_parser = data_subparsers.add_parser("check-dataset", help="详细检查数据集")
    check_dataset_parser.add_argument("--dataset-dir", type=str, help="数据集目录")
    check_dataset_parser.add_argument("--dataset-type", type=str, choices=["yolo", "cnn"], help="数据集类型 (yolo 或 cnn)")
    
    # 模型命令
    model_parser = subparsers.add_parser("model", help="模型相关命令")
    model_subparsers = model_parser.add_subparsers(dest="model_command")
    
    # 训练模型
    train_parser = model_subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--data-dir", default="datasets", help="数据集目录")
    train_parser.add_argument("--yolo-only", action="store_true", help="仅训练YOLO模型")
    train_parser.add_argument("--cnn-only", action="store_true", help="仅训练CNN模型")
    train_parser.add_argument("--yolo-dataset", type=str, help="YOLO数据集路径 (例如: yolo_v8/03_yolo_standard)")
    train_parser.add_argument("--cnn-dataset", type=str, help="CNN数据集路径 (例如: cnn/resnet50_standard)")
    train_parser.add_argument("--batch-size", type=int, default=16, help="批量大小")
    
    # 检测命令
    detect_parser = subparsers.add_parser("detect", help="检测焊缝缺陷")
    detect_parser.add_argument("--source", required=True, help="输入图像或视频")
    detect_parser.add_argument("--view-img", action="store_true", help="显示检测结果")
    detect_parser.add_argument("--num-classes", type=int, help="类别数量，如果不指定则自动从模型中检测")
    detect_parser.add_argument("--class-names", type=str, help="类别名称，用逗号分隔，例如：正常,气孔,裂纹,夹渣,其他")
    detect_parser.add_argument("--save-txt", action="store_true", help="保存检测结果为TXT文本文件")
    detect_parser.add_argument("--save-csv", action="store_true", help="保存检测结果为CSV文件")
    detect_parser.add_argument("--save-json", action="store_true", help="保存检测结果为JSON文件")
    detect_parser.add_argument("--all-formats", action="store_true", help="保存所有格式的检测结果(包括TXT、CSV和JSON)")
    detect_parser.add_argument("--yolo-only", action="store_true", help="仅使用YOLO模型进行单级检测，不使用CNN进行二级分类")
    detect_parser.add_argument("--yolo-model", type=str, help="指定要使用的YOLO模型路径，默认为runs/train/yolo_model.pt")
    detect_parser.add_argument("--cnn-model", type=str, help="指定要使用的CNN模型路径，默认为runs/train/cnn_model.pt")
    
    # 可视化命令
    vis_parser = subparsers.add_parser("visualize", help="可视化检测结果")
    vis_parser.add_argument("--image", required=True, help="输入图像")
    
    # 清理命令
    clean_parser = subparsers.add_parser("clean", help="清理项目")
    clean_parser.add_argument("--dry-run", action="store_true", help="仅显示将要删除的文件，不实际删除")
    clean_parser.add_argument("--clean-temp", action="store_true", help="删除临时文件和缓存")
    clean_parser.add_argument("--clean-results", action="store_true", help="删除训练结果和检测结果")
    clean_parser.add_argument("--clean-tests", action="store_true", help="删除测试脚本")
    clean_parser.add_argument("--all", action="store_true", help="执行所有清理操作")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    # 处理数据命令
    if args.command == "data":
        if args.data_command == "convert":
            # 判断是否为多阶段处理
            if hasattr(args, 'step1') or hasattr(args, 'step2') or hasattr(args, 'step3') or hasattr(args, 'step4') or hasattr(args, 'all'):
                # 调用多阶段数据处理
                from utils.convert_datasets import main as convert_main
                # 构建参数列表
                new_args = [sys.argv[0]]
                if args.step1:
                    new_args.append("--step1")
                if args.step2:
                    new_args.append("--step2")
                if args.step3:
                    new_args.append("--step3")
                if args.step4:
                    new_args.append("--step4")
                if args.all:
                    new_args.append("--all")
                if hasattr(args, 'source1') and args.source1:
                    new_args.append(f"--source1={args.source1}")
                if hasattr(args, 'target1') and args.target1:
                    new_args.append(f"--target1={args.target1}")
                if hasattr(args, 'source2') and args.source2:
                    new_args.append(f"--source2={args.source2}")
                if hasattr(args, 'target2') and args.target2:
                    new_args.append(f"--target2={args.target2}")
                if hasattr(args, 'source3') and args.source3:
                    new_args.append(f"--source3={args.source3}")
                if hasattr(args, 'target3') and args.target3:
                    new_args.append(f"--target3={args.target3}")
                if hasattr(args, 'source4') and args.source4:
                    new_args.append(f"--source4={args.source4}")
                if hasattr(args, 'target4') and args.target4:
                    new_args.append(f"--target4={args.target4}")
                if hasattr(args, 'train_ratio'):
                    new_args.append(f"--train-ratio={args.train_ratio}")
                if hasattr(args, 'valid_ratio'):
                    new_args.append(f"--valid-ratio={args.valid_ratio}")
                if hasattr(args, 'test_ratio'):
                    new_args.append(f"--test-ratio={args.test_ratio}")
                if hasattr(args, 'seed'):
                    new_args.append(f"--seed={args.seed}")
                
                sys.argv = new_args
                return convert_main()
            else:
                # 处理简单的YOLO转CNN格式转换（旧版本保留功能）
                # 使用convert替代原有convert功能
                logger.info(f"使用convert替代原有convert功能: {args.yolo_dir} -> {args.output_dir}")
                from utils.convert_datasets import step3_yolo_to_cnn
                
                # 创建输出目录
                os.makedirs(args.output_dir, exist_ok=True)
                
                # 执行第三步转换
                step3_yolo_to_cnn(args.yolo_dir, args.output_dir, split_data=False)
                return 0
        elif args.data_command == "check":
            # 使用convert_new中的检查函数
            from utils.convert_datasets import check_dataset
            return check_dataset(data_dir=args.data_dir, yolo_dir=args.yolo_dir, cnn_dir=args.cnn_dir)
        elif args.data_command == "check-dataset":
            # 调用数据集检查模块
            from utils.check_dataset import main as check_dataset_main
            # 构建参数列表
            check_dataset_args = [sys.argv[0]]
            if hasattr(args, 'dataset_dir') and args.dataset_dir:
                check_dataset_args.append(f"--dataset-dir={args.dataset_dir}")
            if hasattr(args, 'dataset_type') and args.dataset_type:
                check_dataset_args.append(f"--dataset-type={args.dataset_type}")
            
            sys.argv = check_dataset_args
            return check_dataset_main()
            
    # 处理模型命令
    elif args.command == "model":
        if args.model_command == "train":
            # 导入训练模块
            from train import main as train_main
            
            # 检查数据目录
            data_dir = args.data_dir
            
            # 确保数据目录存在
            if not os.path.exists(data_dir):
                logger.error(f"数据目录 {data_dir} 不存在")
                return 1
            
            # 重构命令行参数
            new_args = [sys.argv[0], f"--data-dir={data_dir}"]
            if args.yolo_only:
                new_args.append("--yolo-only")
            if args.cnn_only:
                new_args.append("--cnn-only")
            
            # 添加YOLO数据集路径参数
            if hasattr(args, 'yolo_dataset') and args.yolo_dataset:
                new_args.append(f"--yolo-dataset={args.yolo_dataset}")
            
            # 添加CNN数据集路径参数
            if hasattr(args, 'cnn_dataset') and args.cnn_dataset:
                new_args.append(f"--cnn-dataset={args.cnn_dataset}")
            
            # 添加批量大小参数
            if hasattr(args, 'batch_size'):
                new_args.append(f"--batch-size={args.batch_size}")
                
            sys.argv = new_args
            return train_main()
            
    # 处理检测命令
    elif args.command == "detect":
        # 导入检测模块
        from detect import main as detect_main
        
        # 重构命令行参数
        new_args = [sys.argv[0], f"--source={args.source}"]
        if args.view_img:
            new_args.append("--view-img")
        
        # 添加类别数量和类别名称参数
        if hasattr(args, 'num_classes') and args.num_classes is not None:
            new_args.append(f"--num-classes={args.num_classes}")
        
        if hasattr(args, 'class_names') and args.class_names:
            new_args.append(f"--class-names={args.class_names}")
        
        # 添加保存格式参数
        if hasattr(args, 'save_txt') and args.save_txt:
            new_args.append("--save-txt")
        
        if hasattr(args, 'save_csv') and args.save_csv:
            new_args.append("--save-csv")
        
        if hasattr(args, 'save_json') and args.save_json:
            new_args.append("--save-json")
        
        # 如果选择了保存所有格式，添加所有格式参数
        if hasattr(args, 'all_formats') and args.all_formats:
            new_args.append("--save-txt")
            new_args.append("--save-csv")
            new_args.append("--save-json")
        
        # 添加YOLO单级检测选项
        if hasattr(args, 'yolo_only') and args.yolo_only:
            new_args.append("--yolo-only")
        
        # 添加YOLO模型路径参数
        if hasattr(args, 'yolo_model') and args.yolo_model:
            new_args.append(f"--yolo-model={args.yolo_model}")
            
        # 添加CNN模型路径参数
        if hasattr(args, 'cnn_model') and args.cnn_model:
            new_args.append(f"--cnn-model={args.cnn_model}")
            
        sys.argv = new_args
        return detect_main()
        
    # 处理可视化命令
    elif args.command == "visualize":
        # 导入可视化模块
        from visualize import main as visualize_main
        
        # 重构命令行参数
        sys.argv = [sys.argv[0], f"--image={args.image}"]
        return visualize_main()
            
    # 处理清理命令
    elif args.command == "clean":
        # 调用清理模块
        from utils.clean_project import main as clean_main
        # 构建参数列表
        clean_args = [sys.argv[0]]
        if args.dry_run:
            clean_args.append("--dry-run")
        if args.clean_temp:
            clean_args.append("--clean-temp")
        if args.clean_results:
            clean_args.append("--clean-results")
        if args.clean_tests:
            clean_args.append("--clean-tests")
        if args.all:
            clean_args.append("--all")
        
        sys.argv = clean_args
        return clean_main()
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 