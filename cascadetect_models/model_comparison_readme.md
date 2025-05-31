# 焊缝缺陷检测模型性能比较工具

该工具用于对比YOLOv8和级联检测(YOLO+CNN)两种焊缝缺陷检测方法的性能表现。

## 功能特点

该比较工具可以评估和对比以下指标：

- **精度指标**：
  - mAP（平均精度均值）
  - AP@0.5（IoU阈值0.5时的平均精度）
  - Recall（召回率）
  - Precision（精确率）
  - F1分数
  - 误检率（False Positive Rate）
  - 漏检率（Miss Rate）
  
- **性能指标**：
  - FPS（推理速度）
  - 参数量（Parameters）
  - 计算量（FLOPs）
  - CPU内存使用
  
- **可视化功能**：
  - 精度指标对比图
  - 性能指标对比图
  - 精确率-召回率(PR)曲线
  - F1曲线
  - 标准混淆矩阵
  - 归一化混淆矩阵

## 使用需求

1. Python 3.8+
2. PyTorch
3. OpenCV
4. Ultralytics YOLOv8
5. matplotlib
6. numpy
7. seaborn (用于混淆矩阵可视化)
8. psutil (用于内存监测)
9. thop (可选，用于计算FLOPs)

可以使用以下命令安装所需依赖：

```bash
pip install torch torchvision opencv-python ultralytics matplotlib tqdm psutil seaborn thop
```

## 使用方法

### 基本使用

在项目根目录运行以下命令：

```bash
python model_comparison.py
```

默认情况下，将会：
- 使用 `datasets/yolo_v8/03_yolo_standard/test` 目录下的测试数据
- 加载默认的YOLOv8和CNN模型
- 将结果保存到 `runs/compare` 目录

### 自定义参数

可以通过命令行参数自定义评估过程：

```bash
python model_comparison.py --dataset [数据集路径] --yolo-model [YOLO模型路径] --cnn-model [CNN模型路径] --output-dir [输出目录]
```

所有可用参数：

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--dataset` | 测试数据集目录 | `datasets/yolo_v8/03_yolo_standard/test` |
| `--yolo-model` | YOLOv8模型路径 | `runs/train/yolo/exp/weights/best.pt` |
| `--cnn-model` | CNN模型路径 | `runs/train/cnn_model.pt` |
| `--conf-thres` | 置信度阈值 | `0.25` |
| `--iou-thres` | IOU阈值 | `0.45` |
| `--batch-size` | 批次大小 | `16` |
| `--img-size` | 图像大小 | `640` |
| `--output-dir` | 输出目录 | `runs/compare` |

### 示例

1. 使用自定义模型路径进行评估：

```bash
python model_comparison.py --yolo-model runs/train/yolo_custom/weights/best.pt --cnn-model runs/train/cnn_custom.pt
```

2. 更改评估阈值：

```bash
python model_comparison.py --conf-thres 0.3 --iou-thres 0.5
```

3. 指定其他测试数据集：

```bash
python model_comparison.py --dataset path/to/custom/test/dataset
```

## 输出结果

脚本运行后将生成以下输出：

1. **性能对比图**：包含主要指标对比的图表
2. **JSON结果文件**：包含所有详细的评估数据
3. **控制台输出**：输出关键性能指标摘要
4. **混淆矩阵**：标准混淆矩阵和归一化混淆矩阵

## 理解比较结果

### 精度指标解释

- **mAP (平均精度均值)**：评估对象检测模型精度的综合指标
- **AP@0.5**：IoU阈值为0.5时的平均精度
- **Precision (精确率)**：模型检测为正例中实际为正例的比例
- **Recall (召回率)**：实际为正例中被正确检测到的比例
- **F1分数**：精确率和召回率的调和平均数，综合评估模型性能
- **误检率(FPR)**：模型误将负例识别为正例的比例
- **漏检率(Miss Rate)**：模型未能识别的正例比例

### 性能指标解释

- **FPS (每秒帧数)**：模型的推理速度，越高越好
- **Parameters (参数量)**：模型的参数总数，反映模型复杂度
- **FLOPs (浮点运算次数)**：反映模型的计算复杂度
- **内存占用**：模型在CPU内存中的占用量

### 混淆矩阵解释

- **标准混淆矩阵**：展示每个类别的预测结果与真实标签的对应关系
- **归一化混淆矩阵**：将标准混淆矩阵的每一行归一化处理，更清晰地展示各类别之间的混淆情况，特别适用于类别不平衡的情况

## 注意事项

1. 确保测试数据集有正确的YOLO格式标签（位于labels目录中）
2. 比较结果受到测试设备硬件条件的影响
3. 在GPU设备上运行可获得更准确的性能评估
4. 对于中文显示，请确保assets/fonts/simhei.ttf字体文件存在
5. 对于更准确的内存使用评估，建议在测试前重启Python进程 