import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import yaml
import shutil

class WeldingDefectDataset(Dataset):
    """焊缝缺陷数据集类"""
    
    def __init__(self, data_dir, transform=None, phase='train'):
        """
        初始化焊缝缺陷数据集
        
        参数:
            data_dir: 数据集根目录
            transform: 图像变换
            phase: 'train', 'valid' 或 'test'
        """
        self.data_dir = data_dir
        self.phase = phase
        self.transform = transform
        
        # 自动检测类别目录
        phase_dir = os.path.join(self.data_dir, self.phase)
        if os.path.exists(phase_dir):
            # 获取所有子目录作为类别
            self.class_names = [d for d in os.listdir(phase_dir) 
                               if os.path.isdir(os.path.join(phase_dir, d)) and not d.startswith('.')]
            if not self.class_names:
                # 如果没有找到任何类别目录，使用默认类别
                print(f"警告: 在 {phase_dir} 中没有找到任何类别目录，使用默认类别")
                self.class_names = ['CC', 'NG', 'PX', 'QP', 'SX']
        else:
            # 如果目录不存在，使用默认类别
            print(f"警告: 目录 {phase_dir} 不存在，使用默认类别")
            self.class_names = ['CC', 'NG', 'PX', 'QP', 'SX']
        
        print(f"检测到的类别: {self.class_names}")
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        # 获取图像文件路径和标签
        self.samples = self._load_samples()
        
        print(f"加载了 {len(self.samples)} 个 {phase} 样本")
    
    def _load_samples(self):
        """加载样本路径和标签"""
        samples = []
        
        # 构建相对路径
        phase_dir = os.path.join(self.data_dir, self.phase)
        
        # 遍历每个类别目录
        for class_name in self.class_names:
            class_dir = os.path.join(phase_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"警告: 类别目录 {class_dir} 不存在")
                continue
            
            # 获取此类别的所有图像
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    label = self.class_to_idx[class_name]
                    samples.append((img_path, label))
        
        return samples
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        img_path, label = self.samples[idx]
        
        # 读取图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法读取图像 {img_path}: {e}")
            # 返回一个空白图像
            image = Image.new('RGB', (224, 224), color='black')
            
        # 应用变换
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


def create_dataloaders(data_dir, batch_size=16, input_size=(224, 224), num_workers=4, val_dir='valid'):
    """
    创建数据加载器
    
    参数:
        data_dir: 数据集根目录
        batch_size: 批大小
        input_size: 输入图像尺寸 (高度, 宽度)
        num_workers: 数据加载线程数
        val_dir: 验证集目录名称，默认为'valid'
        
    返回:
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    
    # 训练集变换
    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证/测试集变换
    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = WeldingDefectDataset(data_dir, transform=train_transform, phase='train')
    valid_dataset = WeldingDefectDataset(data_dir, transform=val_transform, phase=val_dir)
    test_dataset = WeldingDefectDataset(data_dir, transform=val_transform, phase='test')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, valid_loader, test_loader


def prepare_yolo_dataset(data_dir, output_dir=None, class_names=None):
    """
    准备YOLOv8格式的数据集
    
    参数:
        data_dir: 源数据目录
        output_dir: 输出目录 (如果为None，则直接使用原始目录)
        class_names: 类别名称列表
    
    返回:
        data_yaml_path: YAML配置文件路径
    """
    # 如果output_dir为None，使用原始目录
    if output_dir is None:
        output_dir = data_dir
    
    # 检查是否需要创建新目录
    create_new_dirs = output_dir != data_dir
    
    # 只有在需要创建新目录时才创建YOLO数据集目录
    if create_new_dirs:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. 检查是否已经是YOLOv8标准格式数据集(含data.yaml)
    data_yaml_path = os.path.join(data_dir, 'data.yaml')
    if os.path.exists(data_yaml_path):
        print(f"在 {data_dir} 中发现data.yaml文件")
        # 检查yaml文件是否包含必要的配置
        try:
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
                print(f"data.yaml内容: {data_config}")
                if all(k in data_config for k in ['train', 'val']) or all(k in data_config for k in ['train', 'valid']):
                    # 处理相对路径
                    # 首先获取train相对路径
                    train_rel_path = data_config.get('train', 'train/images')
                    
                    # 处理验证集路径 - 支持'val'或'valid'键
                    val_rel_path = None
                    if 'val' in data_config:
                        val_rel_path = data_config.get('val')
                    elif 'valid' in data_config:
                        val_rel_path = data_config.get('valid')
                    else:
                        val_rel_path = 'valid/images'  # 默认值
                    
                    # 检查并处理相对路径 "../"
                    def resolve_relative_path(base_dir, rel_path):
                        """解析相对路径，支持多层'../'"""
                        if rel_path.startswith('../'):
                            # 获取yaml文件所在目录
                            current_dir = os.path.abspath(base_dir)
                            
                            # 处理多层 '../'
                            while rel_path.startswith('../'):
                                # 移动到上一级目录
                                current_dir = os.path.dirname(current_dir)
                                # 移除一个 '../' 前缀
                                rel_path = rel_path[3:]
                            
                            # 拼接剩余路径
                            resolved_path = os.path.join(current_dir, rel_path)
                            print(f"解析相对路径: '{rel_path}' -> '{resolved_path}'")
                        else:
                            # 不是以../开头的相对路径，直接拼接
                            resolved_path = os.path.join(base_dir, rel_path)
                        return resolved_path
                    
                    # 解析路径以用于检查（不保存）
                    train_images_path = resolve_relative_path(data_dir, train_rel_path)
                    val_images_path = resolve_relative_path(data_dir, val_rel_path)
                    
                    # 根据图像路径推断标签路径
                    train_labels_path = train_images_path.replace('images', 'labels')
                    val_labels_path = val_images_path.replace('images', 'labels')
                    
                    print(f"解析后的训练图像路径: {train_images_path}")
                    print(f"解析后的训练标签路径: {train_labels_path}")
                    print(f"解析后的验证图像路径: {val_images_path}")
                    print(f"解析后的验证标签路径: {val_labels_path}")
                    
                    # 检查路径是否存在
                    paths_exist = os.path.exists(train_images_path) and os.path.exists(train_labels_path)
                    val_paths_exist = os.path.exists(val_images_path) and os.path.exists(val_labels_path)

                    if paths_exist:
                        print(f"✅ 检测到训练集目录: {train_images_path}")
                        print(f"✅ 检测到训练集标签目录: {train_labels_path}")
                        
                        if val_paths_exist:
                            print(f"✅ 检测到验证集目录: {val_images_path}")
                            print(f"✅ 检测到验证集标签目录: {val_labels_path}")
                        else:
                            print(f"⚠️ 验证集目录不完整，将尝试使用训练集进行拆分")
                        
                        # 如果不需要创建新的数据集，直接返回原始的yaml路径
                        if not create_new_dirs:
                            print(f"✅ 使用原始数据集目录: {data_dir}")
                            print(f"✅ 使用原始data.yaml文件: {data_yaml_path}")
                            return data_yaml_path
                        
                        # 创建新的data.yaml文件，保留原始路径结构
                        new_data_yaml_path = os.path.join(output_dir, 'data.yaml')
                        new_data_config = data_config.copy()
                        
                        # 保留相对路径，不使用绝对路径
                        new_data_config['train'] = train_rel_path
                        
                        # 处理验证集路径
                        val_key = 'val'
                        if 'valid' in data_config and not 'val' in data_config:
                            val_key = 'valid'  # 保持与原配置一致的键名
                        
                        new_data_config[val_key] = val_rel_path
                        
                        with open(new_data_yaml_path, 'w') as f:
                            yaml.dump(new_data_config, f, default_flow_style=False)
                        
                        print(f"✅ 已创建新的data.yaml文件: {new_data_yaml_path}")
                        
                        # 复制所有文件（如果需要）
                        if create_new_dirs:
                            # 创建images和labels目录
                            for phase in ['train', 'val', 'test']:
                                os.makedirs(os.path.join(output_dir, phase, 'images'), exist_ok=True)
                                os.makedirs(os.path.join(output_dir, phase, 'labels'), exist_ok=True)
                            
                            # 构建源目录和目标目录映射
                            dir_mapping = [
                                (train_images_path, os.path.join(output_dir, 'train', 'images')),
                                (train_labels_path, os.path.join(output_dir, 'train', 'labels')),
                            ]
                            
                            # 确定验证集目录名(valid或val)和目标路径
                            val_output_dir = 'val'  # YOLOv8模型默认使用'val'
                            dir_mapping.extend([
                                (val_images_path, os.path.join(output_dir, val_output_dir, 'images')),
                                (val_labels_path, os.path.join(output_dir, val_output_dir, 'labels')),
                            ])
                            
                            # 检查测试集目录
                            test_rel_path = data_config.get('test', 'test/images')
                            test_images_path = resolve_relative_path(data_dir, test_rel_path)
                            test_labels_path = test_images_path.replace('images', 'labels')
                            
                            if os.path.exists(test_images_path) and os.path.exists(test_labels_path):
                                dir_mapping.extend([
                                    (test_images_path, os.path.join(output_dir, 'test', 'images')),
                                    (test_labels_path, os.path.join(output_dir, 'test', 'labels')),
                                ])
                            
                            # 复制所有文件
                            for src_dir, dst_dir in dir_mapping:
                                if os.path.exists(src_dir):
                                    print(f"  - 正在复制 {src_dir} 到 {dst_dir}")
                                    img_count = 0
                                    for filename in os.listdir(src_dir):
                                        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.txt')):
                                            shutil.copy(
                                                os.path.join(src_dir, filename),
                                                os.path.join(dst_dir, filename)
                                            )
                                            img_count += 1
                                    print(f"    已复制 {img_count} 个文件")
                        
                        return new_data_yaml_path if create_new_dirs else data_yaml_path
                    else:
                        print(f"❌ 部分目录不存在:")
                        if not os.path.exists(train_images_path):
                            print(f"  - 训练图像目录不存在: {train_images_path}")
                        if not os.path.exists(train_labels_path):
                            print(f"  - 训练标签目录不存在: {train_labels_path}")
                        if not os.path.exists(val_images_path):
                            print(f"  - 验证图像目录不存在: {val_images_path}")
                        if not os.path.exists(val_labels_path):
                            print(f"  - 验证标签目录不存在: {val_labels_path}")
                else:
                    print(f"❌ data.yaml缺少必要的配置项 'train'和'val'/'valid'")
        except Exception as e:
            print(f"❌ 读取data.yaml时出错: {e}")
    
    # 2. 检查是否是标准YOLOv8目录结构(无data.yaml)
    standard_structure = True
    # 同时检查train和valid/val目录
    train_images_dir = os.path.join(data_dir, 'train', 'images')
    train_labels_dir = os.path.join(data_dir, 'train', 'labels')
    
    # 先检查valid目录 (Roboflow导出的数据集通常使用valid而不是val)
    valid_images_dir = os.path.join(data_dir, 'valid', 'images')
    valid_labels_dir = os.path.join(data_dir, 'valid', 'labels')
    print(f"检查验证集目录(valid): {valid_images_dir}")

    # 如果valid不存在，检查val目录
    if not (os.path.exists(valid_images_dir) and os.path.exists(valid_labels_dir)):
        valid_images_dir = os.path.join(data_dir, 'val', 'images')
        valid_labels_dir = os.path.join(data_dir, 'val', 'labels')
        print(f"valid目录不存在，检查val目录: {valid_images_dir}")
    
    # 确定实际使用的验证集目录名
    val_dir_name = 'valid' if os.path.exists(valid_images_dir) and os.path.exists(valid_labels_dir) else 'val'

    # 检查目录是否都存在
    if not (os.path.exists(train_images_dir) and os.path.exists(train_labels_dir)):
        print(f"❌ 训练集目录不完整:\n - 图像目录: {train_images_dir} ({'存在' if os.path.exists(train_images_dir) else '不存在'})\n - 标签目录: {train_labels_dir} ({'存在' if os.path.exists(train_labels_dir) else '不存在'})")
        standard_structure = False
    elif not (os.path.exists(valid_images_dir) and os.path.exists(valid_labels_dir)):
        print(f"❌ 验证集目录不完整:\n - 图像目录: {valid_images_dir} ({'存在' if os.path.exists(valid_images_dir) else '不存在'})\n - 标签目录: {valid_labels_dir} ({'存在' if os.path.exists(valid_labels_dir) else '不存在'})")
        standard_structure = False
    else:
        print(f"✅ 检测到完整的YOLOv8目录结构")
    
    if standard_structure:
        # 如果不需要创建新目录，直接创建data.yaml
        if not create_new_dirs:
            print(f"✅ 使用原始数据集目录: {data_dir}")
            output_data_yaml_path = os.path.join(data_dir, 'data.yaml')
            
            # 检测类别 (如未提供)
            if not class_names:
                # 尝试从标签文件中推断类别数量
                labels_dir = os.path.join(data_dir, 'train', 'labels')
                max_class_id = -1
                if os.path.exists(labels_dir):
                    for label_file in os.listdir(labels_dir):
                        if label_file.endswith('.txt'):
                            with open(os.path.join(labels_dir, label_file), 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if parts:
                                        class_id = int(parts[0])
                                        max_class_id = max(max_class_id, class_id)
                
                # 如果找到类别ID，创建默认类别名称
                if max_class_id >= 0:
                    num_classes = max_class_id + 1
                    class_names = [f'class{i}' for i in range(num_classes)]
                    print(f"✅ 从标签文件推断出 {num_classes} 个类别")
            
            # 使用相对路径创建data.yaml
            data_yaml = {
                'train': 'train/images',
                'val': f'{val_dir_name}/images',
                'test': 'test/images',
                'nc': len(class_names),
                'names': class_names
            }
            
            with open(output_data_yaml_path, 'w') as f:
                yaml.dump(data_yaml, f, default_flow_style=False)
            
            print(f"✅ 已在原目录创建data.yaml文件: {output_data_yaml_path}")
            return output_data_yaml_path
        
        # 以下是需要创建新目录时的处理逻辑
        # 创建images和labels目录
        for phase in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, phase, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, phase, 'labels'), exist_ok=True)
        
        # 数据已经是标准YOLOv8格式(但没有yaml或yaml有问题)，直接复制
        print(f"✅ 检测到标准YOLOv8目录结构，正在复制数据...")
        
        # 构建源目录和目标目录映射
        dir_mapping = [
            (train_images_dir, os.path.join(output_dir, 'train', 'images')),
            (train_labels_dir, os.path.join(output_dir, 'train', 'labels')),
        ]
        
        # 确定验证集目录名(valid或val)和目标路径
        val_output_dir = 'val'  # YOLOv8模型默认使用'val'
        dir_mapping.extend([
            (valid_images_dir, os.path.join(output_dir, val_output_dir, 'images')),
            (valid_labels_dir, os.path.join(output_dir, val_output_dir, 'labels')),
        ])
        
        # 检查测试集目录
        test_images_dir = os.path.join(data_dir, 'test', 'images')
        test_labels_dir = os.path.join(data_dir, 'test', 'labels')
        if os.path.exists(test_images_dir) and os.path.exists(test_labels_dir):
            dir_mapping.extend([
                (test_images_dir, os.path.join(output_dir, 'test', 'images')),
                (test_labels_dir, os.path.join(output_dir, 'test', 'labels')),
            ])
        
        # 复制所有文件
        for src_dir, dst_dir in dir_mapping:
            if os.path.exists(src_dir):
                print(f"  - 正在复制 {src_dir} 到 {dst_dir}")
                img_count = 0
                for filename in os.listdir(src_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.txt')):
                        shutil.copy(
                            os.path.join(src_dir, filename),
                            os.path.join(dst_dir, filename)
                        )
                        img_count += 1
                print(f"    已复制 {img_count} 个文件")
        
        # 创建新的data.yaml文件
        output_data_yaml_path = os.path.join(output_dir, 'data.yaml')
        
        # 检测类别
        if not class_names:
            # 尝试从标签文件中推断类别数量
            labels_dir = os.path.join(data_dir, 'train', 'labels')
            max_class_id = -1
            if os.path.exists(labels_dir):
                for label_file in os.listdir(labels_dir):
                    if label_file.endswith('.txt'):
                        with open(os.path.join(labels_dir, label_file), 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    class_id = int(parts[0])
                                    max_class_id = max(max_class_id, class_id)
            
            # 如果找到类别ID，创建默认类别名称
            if max_class_id >= 0:
                num_classes = max_class_id + 1
                class_names = [f'class{i}' for i in range(num_classes)]
                print(f"✅ 从标签文件推断出 {num_classes} 个类别")
        
        # 使用相对路径
        data_yaml = {
            'train': 'train/images',
            'val': f'{val_output_dir}/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(output_data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        return output_data_yaml_path
    else:
        # 既不是标准YOLOv8格式，也不是按类别组织的目录
        print(f"❌ 无法识别数据集格式，请确保符合YOLOv8标准格式或按类别组织的目录结构")
        print("标准YOLOv8格式要求：data/train/images/, data/train/labels/, data/val/images/, data/val/labels/")
        print("按类别组织的目录结构要求：data/train/class1/, data/train/class2/, data/val/class1/, data/val/class2/")
        
        # 如果需要创建新目录，才创建伪数据集
        if create_new_dirs:
            # 创建一个伪数据集，避免程序崩溃
            print("⚠️ 创建一个空的伪数据集结构...")
            for phase in ['train', 'val']:
                # 创建一个1x1像素的空白图像和一个空标签
                empty_img_path = os.path.join(output_dir, phase, 'images', 'empty.jpg')
                empty_label_path = os.path.join(output_dir, phase, 'labels', 'empty.txt')
                
                # 创建空白图像
                img = np.zeros((1, 1, 3), dtype=np.uint8)
                cv2.imwrite(empty_img_path, img)
                
                # 创建空标签文件
                with open(empty_label_path, 'w') as f:
                    f.write("0 0.5 0.5 0.5 0.5\n")
            
            # 创建data.yaml文件
            output_data_yaml_path = os.path.join(output_dir, 'data.yaml')
            data_yaml = {
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'nc': len(class_names) if class_names else 1,
                'names': class_names if class_names else ['default']
            }
            
            with open(output_data_yaml_path, 'w') as f:
                yaml.dump(data_yaml, f, default_flow_style=False)
            
            return output_data_yaml_path
        else:
            return None


def download_sample_dataset(output_dir):
    """
    下载示例焊缝缺陷数据集
    
    参数:
        output_dir: 输出目录
    """
    print("注意: 这只是一个模拟函数，实际实现需要用户根据自己的数据集情况修改")
    print(f"将创建模拟数据集目录结构在: {output_dir}")
    
    # 创建目录结构
    for phase in ['train', 'val', 'test']:
        for cls in ['CC', 'NG', 'PX', 'QP', 'SX']:
            os.makedirs(os.path.join(output_dir, phase, cls), exist_ok=True)
    
    print("目录结构已创建，请将对应的焊缝缺陷图片放入相应目录中")
    print("或者访问以下推荐的公开焊缝缺陷数据集:")
    print("1. GDXray+ 数据集: https://github.com/computervision-xray-testing/GDXray")
    print("2. KolektorSDD2 数据集: https://www.vicos.si/resources/kolektorsdd2/")
    print("3. NEU表面缺陷数据集: http://faculty.neu.edu.cn/songkc/en/su.html")
    
    # 创建一个简单的README文件
    readme_path = os.path.join(output_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("焊缝缺陷数据集目录结构\n")
        f.write("------------------------\n\n")
        f.write("数据集应按以下结构组织:\n")
        f.write("data_root/\n")
        f.write("  ├── train/\n")
        f.write("  │   ├── CC/\n")
        f.write("  │   ├── NG/\n")
        f.write("  │   ├── PX/\n")
        f.write("  │   ├── QP/\n")
        f.write("  │   └── SX/\n")
        f.write("  ├── val/\n")
        f.write("  │   ├── CC/\n")
        f.write("  │   ├── NG/\n")
        f.write("  │   ├── PX/\n")
        f.write("  │   ├── QP/\n")
        f.write("  │   └── SX/\n")
        f.write("  └── test/\n")
        f.write("      ├── CC/\n")
        f.write("      ├── NG/\n")
        f.write("      ├── PX/\n")
        f.write("      ├── QP/\n")
        f.write("      └── SX/\n")
    
    print(f"已创建README文件: {readme_path}") 