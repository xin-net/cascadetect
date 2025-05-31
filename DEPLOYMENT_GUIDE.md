# 项目部署指南

本指南提供了将 CascadeDetect 项目部署到服务器上的基本步骤。项目包含三个主要部分：前端 (Vue.js)、后端 (Python) 和机器学习模型。

请注意，具体的部署步骤可能因您的服务器环境（操作系统、Web 服务器选择等）而有所不同。本指南假设使用 Linux 环境。

## 1. 服务器环境准备

确保您的服务器满足以下要求：

-   **操作系统**: 推荐使用 Linux 发行版 (如 Ubuntu, CentOS)。
-   **Python**: 安装 Python 3.8 或更高版本。
-   **Node.js**: 安装 Node.js 和 npm 或 yarn。
-   **Git**: 安装 Git 以克隆代码库。
-   **Web 服务器**: 考虑安装 Nginx 或 Apache 作为反向代理服务器（可选，但推荐用于生产环境）。
-   **数据库**: 后端使用 SQLite 数据库 (`backend.db`)，默认情况下无需额外安装数据库服务器。如果需要使用其他数据库，请修改后端配置。

## 2. 克隆代码库

在服务器上选择一个合适的目录，克隆项目代码库：

```bash
git clone <您的代码库URL>
cd cascadetect
```

## 3. 安装依赖

项目包含多个部分的依赖，需要分别安装。

### 后端依赖

进入后端目录并安装 Python 依赖：

```bash
cd cascadetect_backend
pip install -r requirements.txt
```

### 前端依赖

进入前端目录并安装 Node.js 依赖：

```bash
cd ../cascadetect_forend
npm install # 或者 yarn install
```

### 模型依赖

进入模型目录并安装 Python 依赖：

```bash
cd ../cascadetect_models
pip install -r requirements.txt
```

## 4. 构建前端应用

在前端目录 (`cascadetect_forend`) 中，构建生产环境的前端文件：

```bash
npm run build # 或者 yarn build
```

这将在 `cascadetect_forend` 目录下生成一个 `dist` 文件夹，包含所有静态文件。

## 5. 配置后端

根据您的服务器环境，您可能需要修改后端配置。主要配置文件可能是 `cascadetect_backend/app.py` 或其他相关文件。

-   **数据库路径**: 如果需要修改数据库文件位置，请更新相关配置。
-   **模型路径**: 确保后端能够正确访问 `cascadetect_models` 目录中的模型文件。
-   **上传目录**: 配置图片上传的存储目录。
-   **端口**: 配置后端服务监听的端口。

## 6. 运行后端服务

进入后端目录 (`cascadetect_backend`)，运行后端应用程序。推荐使用生产级的 WSGI 服务器（如 Gunicorn 或 uWSGI）来运行 Flask 应用。

首先，安装 Gunicorn：

```bash
pip install gunicorn
```

然后，使用 Gunicorn 运行应用（假设您的 Flask 应用实例在 `app.py` 中名为 `app`）：

```bash
gunicorn -w 4 'app:app' -b 0.0.0.0:5000 # 5000是示例端口
```

`-w 4` 表示使用4个 worker 进程，您可以根据服务器性能调整。

## 7. 部署前端静态文件

将构建好的前端静态文件 (`cascadetect_forend/dist` 目录) 部署到 Web 服务器（如 Nginx 或 Apache）或直接通过后端服务提供。

### 使用 Nginx 部署前端

安装 Nginx 并配置一个 server 块来服务前端静态文件，并将 API 请求代理到后端服务。

示例 Nginx 配置 (位于 `/etc/nginx/sites-available/your_app`):

```nginx
server {
    listen 80;
    server_name your_domain_or_ip;

    location / {
        root /path/to/cascadetect/cascadetect_forend/dist;
        try_files $uri $uri/ /index.html;
    }

    location /api/ { # 假设后端API路径以/api/开头
        proxy_pass http://127.0.0.1:5000; # 替换为后端实际监听的地址和端口
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

创建软链接到 `sites-enabled` 并重启 Nginx：

```bash
sudo ln -s /etc/nginx/sites-available/your_app /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

### 通过后端服务提供前端文件

另一种方法是配置后端服务来提供前端静态文件。这需要在后端框架中设置静态文件路由和单页面应用 (SPA) 的回退路由。

## 8. 运行为服务

为了确保应用在服务器重启后自动启动，并方便管理，建议将后端和前端（如果不是通过 Nginx 等服务）配置为系统服务。

### 使用 systemd (Linux)

创建一个 systemd service 文件 (例如 `/etc/systemd/system/cascadetect_backend.service`)：

```ini
[Unit]
Description=CascadeDetect Backend
After=network.target

[Service]
User=your_user # 替换为运行服务的用户
WorkingDirectory=/path/to/cascadetect/cascadetect_backend # 替换为后端目录的绝对路径
ExecStart=/usr/local/bin/gunicorn -w 4 'app:app' -b 0.0.0.0:5000 # 替换为您的Python环境和Gunicorn命令
Restart=always

[Install]
WantedBy=multi-user.target
```

重新加载 systemd 配置并启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl start cascadetect_backend
sudo systemctl enable cascadetect_backend # 设置开机自启
```

对前端服务（如果需要）或模型服务重复类似步骤。

## 9. 模型部署

确保 `cascadetect_models` 目录及其内容在服务器上可用，并且后端服务能够访问到模型文件 (`yolov8n.pt` 等)。如果模型文件较大，可能需要单独下载或通过其他方式传输到服务器。

## 10. 访问应用

如果通过 Nginx 部署，通过配置的域名或服务器IP访问。如果直接通过后端服务提供前端，通过后端服务的地址和端口访问。

## 故障排除

-   **检查日志**: 查看后端服务（Gunicorn/uWSGI）、Web 服务器 (Nginx/Apache) 和 systemd 的日志文件，查找错误信息。
-   **防火墙**: 确保服务器防火墙允许外部访问应用端口 (例如 80 或 443)。
-   **依赖**: 确认所有依赖都已正确安装，并且 Python 和 Node.js 环境配置正确。

这是一个基本的部署指南，您可能需要根据项目的具体需求和服务器环境进行调整。