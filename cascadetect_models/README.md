# 焊缝缺陷多模型级联检测系统

焊缝缺陷多模型级联检测系统(项目英文名:cascadetect)，采用YOLO与CNN级联架构，用于焊缝缺陷的高精度检测。

## 系统架构
- **第一阶段**：YOLO快速定位焊缝区域中的潜在缺陷候选框
- **第二阶段**：CNN对YOLO输出的候选区域进行高分辨率特征提取，提升对气孔、裂纹等细微缺陷的分类精度
- **单级检测**：也支持仅使用YOLO模型进行单级检测的轻量级模式


## 环境配置
系统要求：
- Python 3.8+
- CUDA 11.0+（推荐，用于GPU加速）
- 内存 8GB+

安装依赖：
```bash
conda activate cascadetect
pip install -r requirements.txt
```

## 数据集结构

项目只包含原始数据集，其他数据集可以通过数据转换命令自动生成：

```
datasets/
└── yolo_v8/
    └── 01_yolo_roboflow/   # 原始标注数据
        ├── data.yaml        # 数据集配置文件
        ├── train/           # 训练数据
        │   ├── images/      # 图像文件
        │   └── labels/      # 标签文件 (.txt格式)
```

### 关于数据路径处理的重要说明

系统完全支持相对路径，**不会修改原始data.yaml文件**。当YOLOv8训练开始时：

1. **临时配置文件机制**：
   - 系统会创建一个临时配置文件（`data_temp.yaml`）
   - 临时文件中的路径会被转换为绝对路径，以便YOLOv8正确加载数据
   - **原始配置文件保持不变**，不会被修改
   - 训练结束后，临时文件会被自动删除

2. **支持的路径格式**：
   - **标准相对路径**：`train/images`和`valid/images`（相对于data.yaml所在目录）
   - **向上引用相对路径**：`../train/images`（相对于data.yaml所在的上一级目录）
   - **绝对路径**：可选，但不推荐

这种机制确保了配置文件的完整性，同时提高了项目在不同环境间的移植性。

## 项目使用说明

本项目提供了一个统一的入口脚本`casacadetect.py`，可以通过以下方式运行各种子命令：

```bash
python casacadetect.py <command> [args...]
```

主要支持以下几种子命令：

- **train**: 训练模型
- **detect**: 使用模型进行检测
- **data**: 数据处理相关命令
- **clean**: 清理项目临时文件和冗余代码

### 数据处理命令

```bash
# 多阶段数据处理 - 执行所有步骤
python casacadetect.py data convert --all

# 仅执行第一步：多边形标注转矩形框
python casacadetect.py data convert --step1 --source1 datasets/yolo_v8/01_yolo_roboflow --target1 datasets/yolo_v8/02_yolo_rectangle

# 仅执行第二步：划分数据集
python casacadetect.py data convert --step2 --source2 datasets/yolo_v8/02_yolo_rectangle --target2 datasets/yolo_v8/03_yolo_standard

# 仅执行第三步：YOLOv8标准格式转CNN格式（ResNet50所需）
python casacadetect.py data convert --step3 --source3 datasets/yolo_v8/03_yolo_standard --target3 datasets/cnn/resnet50_standard

# 仅执行第四步：将多类别缺陷合并为单一"缺陷"类别
python casacadetect.py data convert --step4 --source4 datasets/yolo_v8/03_yolo_standard --target4 datasets/yolo_v8/04_yolo_merged

# 检查数据集
python casacadetect.py data check --yolo-dir yolo_v8/03_yolo_standard --cnn-dir cnn/resnet50_standard --data-dir datasets
```

### 模型训练命令

```bash
# 训练完整级联系统
python casacadetect.py model train

# 仅训练YOLO模型
python casacadetect.py model train --yolo-only

# 仅训练CNN模型
python casacadetect.py model train --cnn-only

# 使用指定的数据集进行训练
python casacadetect.py model train --yolo-dataset yolo_v8/03_yolo_standard --cnn-dataset cnn/resnet50_standard
```

### 实验跟踪与可视化

项目支持使用 TensorBoard 和 MLflow 对训练过程和实验结果进行跟踪与可视化。

**TensorBoard**

TensorBoard 用于可视化训练过程中的指标，例如损失、准确率等。日志文件默认保存在 `logs/tensorboard` 目录下。

启动 TensorBoard：
```bash
tensorboard --logdir logs/tensorboard
```

**MLflow**

MLflow 用于跟踪、记录和管理机器学习实验。实验数据默认保存在 `logs/mlruns` 目录下。

启动 MLflow UI：
```bash
mlflow ui --backend-store-uri logs/mlruns
```

### 缺陷检测命令

```bash
# 使用级联模型检测图像
python casacadetect.py detect --source path/to/image.jpg

# 使用级联模型检测视频并显示结果
python casacadetect.py detect --source path/to/video.mp4 --view-img

# 仅使用YOLO单级模型检测图像
python casacadetect.py detect --source path/to/image.jpg --yolo-only

# 仅使用YOLO单级模型检测视频
python casacadetect.py detect --source path/to/video.mp4 --yolo-only --view-img

# 保存检测结果为不同文件格式
python casacadetect.py detect --source path/to/image.jpg --save-txt --save-csv --save-json

# 指定自定义类别名称
python casacadetect.py detect --source path/to/image.jpg --class-names 正常,气孔,裂纹,夹渣,其他

# 使用自定义YOLO模型路径
python casacadetect.py detect --source path/to/image.jpg --yolo-model runs/train/yolo/1/weights/best.pt

# 使用自定义CNN模型路径
python casacadetect.py detect --source path/to/image.jpg --cnn-model runs/train/cnn/2/cnn_model.pt

# 同时指定两个自定义模型路径
python casacadetect.py detect --source path/to/image.jpg --yolo-model custom_yolo.pt --cnn-model custom_cnn.pt

## 检测结果输出

系统支持多种检测结果输出格式：

1. **文本格式 (TXT)**：简洁的文本格式，包含边界框坐标、置信度和类别信息
2. **CSV格式**：表格形式，便于使用Excel等软件进行后续分析
3. **JSON格式**：结构化数据，适合程序处理和API交互
4. **图像/视频输出**：可视化结果，直观展示检测效果

使用`--save-txt`、`--save-csv`和`--save-json`参数可分别保存对应格式的结果。
```

### 清理项目命令

```bash
# 查看所有可删除的文件（不实际删除）
python casacadetect.py clean --dry-run --all

# 清理临时文件和缓存
python casacadetect.py clean --clean-temp

# 清理训练结果和检测结果
python casacadetect.py clean --clean-results

# 清理测试脚本
python casacadetect.py clean --clean-tests

# 执行所有清理操作
python casacadetect.py clean --all
```

清理功能包含三个主要部分：

1. **清理临时文件和缓存**：包括 `__pycache__` 目录、`.pyc` 文件和 `.DS_Store` 文件
2. **清理训练结果和检测结果**：包括 `runs/detect`、`runs/train` 和 `results` 目录
3. **清理测试脚本**：包括项目根目录下的 `test_*.py` 文件


## 项目结构

```
./
├── README.md                 # 项目说明文档
├── PROJECT_STRUCTURE.md      # 项目结构说明
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
└── datasets                  # 数据集目录
    └── yolo_v8               # YOLO数据集
        └── 01_yolo_roboflow  # 原始多边形标注数据集
```

