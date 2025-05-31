import sys
import os

# Add the parent directory of cascadetect_models to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, send_from_directory
from flask_restful import Api
from flask_jwt_extended import JWTManager
from flask_cors import CORS # 导入CORS
from extensions import db
import secrets

app = Flask(__name__)
CORS(app) # 为整个应用启用CORS，允许所有来源

# 获取当前脚本所在的目录
basedir = os.path.abspath(os.path.dirname(__file__))
# 配置SQLite数据库，存储在项目根目录的 backend.db 文件中
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/cascadetect_db' # 请替换为您的MySQL连接信息
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = secrets.token_hex(32)  # 使用secrets模块生成强密钥
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'uploads') # 定义上传文件夹

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db.init_app(app)
api = Api(app)
jwt = JWTManager(app)

# 导入资源
import resources

# 添加API资源路由
api.add_resource(resources.ImageUpload, '/upload')
api.add_resource(resources.ImageList, '/images')
api.add_resource(resources.ImageDetail, '/images/<int:image_id>')
api.add_resource(resources.ImageExport, '/images/<int:image_id>/export') # 添加导出路由
api.add_resource(resources.AnnotationUpdate, '/annotations/<int:annotation_id>')
api.add_resource(resources.AnnotationCreate, '/images/<int:image_id>/annotations')

# 添加静态文件服务路由，用于前端访问上传的图片
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

first_request_done = False

@app.before_request
def before_first_request_handler():
    global first_request_done
    if not first_request_done:
        with app.app_context():
            db.create_all()
        first_request_done = True

if __name__ == '__main__':
    # 在运行应用前创建数据库表（如果它们还不存在）
    with app.app_context(): # 确保在应用上下文中执行
        db.create_all()
    app.run(debug=True, port=5000) # 指定端口，避免与前端默认端口冲突