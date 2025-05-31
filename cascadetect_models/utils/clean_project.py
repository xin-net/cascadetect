#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
项目清理工具 - 删除冗余代码和临时文件
"""

import os
import sys
import shutil
import glob
import logging
import argparse
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 临时文件和目录列表
TEMP_FILES = [
    '*.pyc',
    '.DS_Store'
]

TEMP_DIRS = [
    './__pycache__',
    './models/__pycache__',
    './utils/__pycache__'
]

# 结果目录列表
RESULT_DIRS = [
    './runs',
    './runs/detect',
    './runs/train'
]

# 测试脚本列表
TEST_FILES = [
    './check_dependencies.py',
    './fix_yolo_tensorboard.py',  # 已集成到主代码
    './compare_models.py',
    './test_*.py',
    './visualize_tensorboard.py',
    './yolo11n.pt',  # 临时模型文件
    './TENSORBOARD_GUIDE.md'  # 临时文档
]

# 要删除的生成数据集目录(除了原始的01_yolo_roboflow)
GENERATED_DATASETS = [
    './datasets/yolo_v8/02_yolo_rectangle',
    './datasets/yolo_v8/03_yolo_standard',
    './datasets/yolo_v8/04_yolo_nocategory',
    './datasets/cnn',
    './datasets/yolo_defw'
]

def confirm(message):
    """用户确认函数"""
    response = input(f"{message} [y/N]: ").lower().strip()
    return response == 'y'

def delete_files(pattern_list, dry_run=False):
    """删除匹配模式的文件"""
    deleted = 0
    for pattern in pattern_list:
        for file_path in glob.glob(pattern, recursive=True):
            if os.path.isfile(file_path):
                if dry_run:
                    logger.info(f"将删除文件: {file_path}")
                else:
                    try:
                        os.remove(file_path)
                        logger.info(f"已删除文件: {file_path}")
                        deleted += 1
                    except Exception as e:
                        logger.error(f"删除文件失败 {file_path}: {e}")
    return deleted

def delete_dirs(dir_list, dry_run=False):
    """删除目录列表"""
    deleted = 0
    for dir_path in dir_list:
        if os.path.exists(dir_path):
            if dry_run:
                logger.info(f"将删除目录: {dir_path}")
            else:
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"已删除目录: {dir_path}")
                    deleted += 1
                except Exception as e:
                    logger.error(f"删除目录失败 {dir_path}: {e}")
    return deleted

def clean_temp(dry_run=False):
    """清理临时文件和缓存"""
    logger.info("清理临时文件和缓存...")
    files_deleted = delete_files(TEMP_FILES, dry_run)
    dirs_deleted = delete_dirs(TEMP_DIRS, dry_run)
    
    if dry_run:
        logger.info(f"将删除 {files_deleted} 个临时文件和 {dirs_deleted} 个缓存目录")
    else:
        logger.info(f"已删除 {files_deleted} 个临时文件和 {dirs_deleted} 个缓存目录")
    
def clean_results(dry_run=False):
    """清理训练结果和检测结果"""
    logger.info("清理训练结果和检测结果...")
    dirs_deleted = delete_dirs(RESULT_DIRS, dry_run)
    
    if dry_run:
        logger.info(f"将删除 {dirs_deleted} 个结果目录")
    else:
        logger.info(f"已删除 {dirs_deleted} 个结果目录")
    
def clean_tests(dry_run=False):
    """清理测试脚本"""
    logger.info("清理测试脚本...")
    files_deleted = delete_files(TEST_FILES, dry_run)
    
    if dry_run:
        logger.info(f"将删除 {files_deleted} 个测试脚本")
    else:
        logger.info(f"已删除 {files_deleted} 个测试脚本")

def clean_datasets(dry_run=False):
    """清理生成的数据集，保留原始数据集"""
    logger.info("清理生成的数据集(保留原始数据集)...")
    dirs_deleted = delete_dirs(GENERATED_DATASETS, dry_run)
    
    if dry_run:
        logger.info(f"将删除 {dirs_deleted} 个生成的数据集目录")
    else:
        logger.info(f"已删除 {dirs_deleted} 个生成的数据集目录")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="清理项目，删除冗余代码和临时文件")
    parser.add_argument("--dry-run", action="store_true", help="仅显示将要删除的文件，不实际删除")
    parser.add_argument("--clean-temp", action="store_true", help="删除临时文件和缓存")
    parser.add_argument("--clean-results", action="store_true", help="删除训练结果和检测结果")
    parser.add_argument("--clean-tests", action="store_true", help="删除测试脚本")
    parser.add_argument("--clean-datasets", action="store_true", help="删除生成的数据集目录(保留原始数据集)")
    parser.add_argument("--all", action="store_true", help="执行所有清理操作")
    
    args = parser.parse_args()
    
    # 如果未指定任何操作，则执行所有操作
    if not (args.clean_temp or args.clean_results or args.clean_tests or args.clean_datasets or args.all):
        args.all = True
    
    # 显示清理操作信息
    logger.info("=" * 50)
    logger.info("项目清理工具")
    logger.info("=" * 50)
    
    if args.dry_run:
        logger.info("当前为预览模式，不会实际删除文件")
    
    # 执行清理操作
    if args.all or args.clean_temp:
        clean_temp(args.dry_run)
    
    if args.all or args.clean_results:
        clean_results(args.dry_run)
    
    if args.all or args.clean_tests:
        clean_tests(args.dry_run)
    
    if args.all or args.clean_datasets:
        clean_datasets(args.dry_run)
    
    logger.info("=" * 50)
    logger.info("清理操作完成")
    logger.info("注意: 原始数据集(01_yolo_roboflow)和核心功能代码已保留")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 