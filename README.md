# 人机配合的航空件缺陷检测软件

本项目是一个基于大模型的航空件缺陷检测软件，采用B/S架构，旨在帮助用户高效地识别和标注航空件图片中的缺陷。

## 项目结构

```
/Cascadetect
├── cascadetect_backend/      # 后端 (Flask)
├── cascadetect_forend/       # 前端 (Vue.js + Vite)
├── cascadetect_models/       # 缺陷检测模型和脚本
└── README.md                 # 本项目说明文档
```

## 主要功能

- **图片上传与管理**: 用户可以批量上传航空件图片。
- **自动缺陷检测**: 利用预训练的深度学习模型（YOLO + CNN级联）自动识别图片中的缺陷。
- **交互式标注**: 用户可以在前端GUI查看、修改、删除模型生成的标注，也可以手动创建新的标注。
- **三栏布局**: 
    - 左侧：图片列表，支持筛选。
    - 中间：当前选中图片的展示区域，可进行标注操作。
    - 右侧：当前图片的详细标注信息列表。
- **数据持久化**: 检测结果和标注信息保存到数据库中。

## 技术栈

- **后端**: Python, Flask, Flask-SQLAlchemy, Flask-RESTful, Flask-JWT-Extended (本项目中JWT暂未强制启用所有接口), PyMySQL (本项目使用SQLite作为示例), OpenCV, PyTorch, Ultralytics (YOLO)
- **前端**: Vue.js (Vue 3), Vite, Axios, JavaScript
- **模型**: YOLOv8, ResNet50 (或其他CNN模型)
- **数据库**: MySQL (本项目 `app.py` 中配置为SQLite，方便快速启动，可按需修改为MySQL)

## 环境配置与启动

### 1. Conda 环境 (推荐)

本项目推荐在Conda环境下运行。

```bash
# 假设您已有一个名为 fullstack 的conda环境
conda activate fullstack 
# 如果没有，请先创建并激活
# conda create -n fullstack python=3.9  (或其他兼容版本)
# conda activate fullstack
```

### 2. 安装依赖

#### a. 模型依赖

模型部分的依赖项在 `cascadetect_models/requirements.txt` 中定义。

```bash
cd cascadetect_models
# 注意：根据您的CUDA版本，可能需要安装特定版本的torch和torchvision
# 例如: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
cd ..
```

**重要**: 
- `cascadetect_models/requirements.txt` 中可能包含 `ultralytics`，它会自动安装 PyTorch。请确保安装的 PyTorch 版本与您的 CUDA 环境兼容。
- 如果遇到模型加载或运行问题，请检查 `cascadetect_models/README.md` 获取更详细的模型配置和使用说明。

#### b. 后端依赖

```bash
cd cascadetect_backend
pip install -r requirements.txt
cd ..
```

#### c. 前端依赖

需要 Node.js 和 npm (或 yarn/pnpm)。

```bash
cd cascadetect_forend
npm install 
# 或者 yarn install / pnpm install
cd ..
```

### 3. 数据库配置 (后端)

- 本项目后端 `cascadetect_backend/app.py` 默认配置使用 **SQLite** 数据库 (`backend.db`)，它会在首次运行时自动创建在 `cascadetect_backend` 目录下，无需额外配置即可运行。
- 如果您想使用 **MySQL**，请修改 `cascadetect_backend/app.py` 中的 `SQLALCHEMY_DATABASE_URI` 配置，并确保MySQL服务已启动且数据库已创建：
  ```python
  # app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://user:password@host/db_name'
  ```
  同时，您可能需要安装 `PyMySQL` (已包含在 `requirements.txt` 中)。

### 4. 模型文件准备 (重要)

- **YOLO模型**: 后端代码 (`cascadetect_backend/resources.py`) 默认会尝试加载 `cascadetect_models/runs/train/yolo/exp/weights/best.pt`。
    - 如果此路径下没有模型，它会回退到使用 `cascadetect_models/yolov8n.pt` (一个预训练的通用模型)。
    - 您可以通过训练自己的YOLO模型 (参考 `cascadetect_models/README.md` 中的训练命令) 并将生成的 `best.pt` 放到上述期望路径，或者修改 `resources.py` 中的 `yolo_model_path`。
- **CNN模型**: 类似地，后端会尝试加载 `cascadetect_models/runs/train/cnn_model/best_cnn_model.pt`。
    - 如果此路径下没有模型，`cnn_model_path` 会被设为 `None`，这意味着级联检测器可能只使用YOLO进行检测，或者其内部有进一步的处理逻辑。
    - 训练您自己的CNN模型 (参考 `cascadetect_models/README.md`) 并放置到期望路径，或修改 `resources.py` 中的 `cnn_model_path`。
- **类别名称**: 类别名称默认从 `cascadetect_models/datasets/yolo_v8/03_yolo_standard/data.yaml` 文件中加载。如果此文件不存在或格式不正确，会使用一组默认类别。请确保此文件存在且包含正确的 `names` 列表。

### 5. 启动服务

需要分别启动后端和前端服务。

#### a. 启动后端服务

打开一个新的终端：
```bash
conda activate fullstack # (如果需要)
cd cascadetect_backend
python app.py
```
后端服务默认运行在 `http://localhost:5000`。

#### b. 启动前端服务

打开另一个新的终端：
```bash
conda activate fullstack # (如果需要，主要确保Node环境可用)
cd cascadetect_forend
npm run dev
# 或者 yarn dev / pnpm dev
```
前端开发服务器通常会运行在 `http://localhost:5173` (或其他 Vite 默认端口)，并在浏览器中自动打开。

### 6. 使用软件

- 打开浏览器访问前端地址 (例如 `http://localhost:5173`)。
- 在左侧面板上传图片。
- 图片上传并自动检测后，会显示在列表中。
- 点击列表中的图片，中间会显示图片和标注框，右侧会显示详细标注信息。
- 用户可以与标注进行交互。

## 注意事项

- **JWT认证**: 后端代码中包含了 `Flask-JWT-Extended` 的设置，但在 `resources.py` 中的API端点前，`@jwt_required()` 被注释掉了。如果需要启用认证，请取消注释并实现登录逻辑。
- **文件上传大小**: Flask 和 Web 服务器 (如 Nginx/Apache，如果部署在生产环境) 可能有文件上传大小限制，如果需要上传大图片，请相应调整配置。
- **错误处理**: 本项目为基础框架，错误处理和日志记录方面可以进一步完善。
- **UI/UX**: 前端UI可以根据实际需求进一步美化和优化用户体验。
- **模型性能**: 模型的检测精度和速度取决于训练数据、模型结构和硬件配置。请参考 `cascadetect_models` 中的文档进行模型优化。

## 开发说明

- **后端API**: 
    - `POST /upload`: 上传图片并进行检测。
    - `GET /images`: 获取所有图片列表。
    - `GET /images/<image_id>`: 获取单张图片的详细信息和标注。
    - `PUT /annotations/<annotation_id>`: 更新指定标注。
    - `DELETE /annotations/<annotation_id>`: 删除指定标注。
    - `POST /images/<image_id>/annotations`: 为指定图片创建新标注。
    - `GET /uploads/<filename>`: 获取上传的图片文件。
- **前端组件**: 
    - `App.vue`: 主应用组件，组织布局。
    - `ImageListPanel.vue`: 左侧图片列表和上传区域。
    - `ImageDisplayPanel.vue`: 中间图片显示和绘制标注区域。
    - `AnnotationPanel.vue`: 右侧标注信息编辑区域。

祝您使用愉快！