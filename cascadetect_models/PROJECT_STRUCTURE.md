# 焊缝缺陷多模型级联检测系统 - 项目结构

## 目录结构

```
./
├── README.md                 # 项目说明文档
├── PROJECT_STRUCTURE.md      # 项目结构说明（本文件）
├── requirements.txt          # 项目依赖库
├── casacadetect.py           # 项目统一入口
├── train.py                  # 模型训练模块
├── detect.py                 # 图像检测模块
├── visualize.py              # 可视化模块
├── models                    # 模型目录
│   ├── yolo_model.py         # YOLOv8模型封装
│   ├── cnn_model.py          # CNN模型（ResNet50）封装
│   └── cascade_detector.py   # 级联检测器实现
├── utils                     # 工具函数目录
│   ├── __init__.py           # 初始化文件
│   ├── convert_datasets.py   # 多阶段数据处理工具
│   ├── clean_project.py      # 项目清理工具
│   ├── check_dataset.py      # 数据集检查模块
│   └── data_utils.py         # 数据处理工具函数
├── assets                    # 资源文件目录
│   └── fonts                 # 字体文件目录（用于中文显示）
├── datasets                  # 数据集目录
│   ├── yolo_v8               # YOLO数据集
│   │   └── 01_yolo_roboflow  # 原始多边形标注数据集
│   └── ...                   # 数据处理后生成的其他数据集
└── results                   # 结果输出目录
```

## 项目文件说明

1. **casacadetect.py**：项目统一入口，支持以下子命令：
   - `train`：训练模型
   - `detect`：使用模型进行检测
   - `data`：数据处理相关命令
   - `clean`：清理项目临时文件和冗余代码

2. **train.py**：模型训练模块，支持训练YOLOv8和CNN模型

3. **detect.py**：图像检测模块，支持使用YOLOv8和级联检测器进行检测，包含以下功能：
   - 级联检测模式：结合YOLOv8和CNN进行两阶段检测
   - YOLO单级检测模式：仅使用YOLOv8进行轻量级检测
   - 多种结果输出格式：TXT、CSV、JSON、图像/视频

4. **visualize.py**：数据可视化模块，用于可视化YOLO和CNN数据集

5. **models目录**：
   - `yolo_model.py`：YOLOv8模型的封装
   - `cnn_model.py`：CNN模型（ResNet50）的封装
   - `cascade_detector.py`：级联检测器的实现，支持类别数自适应和自定义类别名称

6. **utils目录**：
   - `__init__.py`：初始化文件
   - `convert_datasets.py`：多阶段数据处理工具，包括四个主要步骤：
     1. 将多边形标注转换为矩形框
     2. 划分数据集为训练集、验证集和测试集
     3. 将YOLOv8标准格式数据集转换为ResNet50所需的CNN数据集格式
     4. 将多种缺陷类别合并为单一"缺陷"类别
   - `clean_project.py`：项目清理工具，用于清理临时文件、训练结果和测试脚本
   - `check_dataset.py`：数据集检查模块，用于检查数据集的完整性
   - `data_utils.py`：数据处理工具函数

7. **assets目录**：
   - `fonts`：字体文件目录，用于渲染中文文本

8. **datasets目录**：
   - `yolo_v8`：YOLO数据集，包含原始多边形标注数据集和数据处理后生成的其他数据集

9. **results目录**：用于存储分析结果和比较报告

## 数据流转换流程

1. **数据准备**：使用`utils/convert_datasets.py`进行多阶段数据处理。
2. **模型训练**：使用`train.py`训练YOLOv8检测器和CNN分类器。
3. **缺陷检测**：使用`detect.py`进行焊缝缺陷检测，支持图像和视频输入以及YOLO单级检测。
4. **结果可视化**：使用`visualize.py`可视化检测结果和模型性能。
所有功能通过统一的命令行接口`casacadetect.py`进行访问。 