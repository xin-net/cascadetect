import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import time

class CNNClassifier:
    """基于ResNet50的CNN分类器类，用于焊缝缺陷的精细分类"""
    
    def __init__(self, num_classes=4, model_path=None, input_size=(224, 224)):
        """
        初始化CNN分类器
        
        参数:
            num_classes: 缺陷类别数量
            model_path: 预训练模型路径，如果为None则从头训练
            input_size: 输入图像尺寸(高度,宽度)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        
        # 创建ResNet50模型
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 修改最后的全连接层以适应我们的分类任务
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 将模型移动到设备上
        self.model = self.model.to(self.device)
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 如果提供了模型路径，则加载预训练权重
        if model_path:
            self.load(model_path)
        
        print(f"CNN分类器已初始化，使用设备: {self.device}")
    
    def preprocess(self, image):
        """
        预处理输入图像
        
        参数:
            image: 输入图像(OpenCV格式BGR)
            
        返回:
            tensor: 预处理后的图像张量
        """
        # OpenCV格式(BGR)转换为PIL格式(RGB)
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # 应用变换
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  # 添加批次维度
        tensor = tensor.to(self.device)
        
        return tensor
    
    def classify(self, image):
        """
        对输入图像进行分类
        
        参数:
            image: 输入图像(OpenCV BGR格式或PIL格式)
            
        返回:
            class_id: 预测的类别ID
            confidence: 置信度
        """
        # 图像预处理
        tensor = self.preprocess(image)
        
        # 推理
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            confidence, predicted = torch.max(probs, 1)
            
            class_id = predicted.item()
            confidence = confidence.item()
        
        return class_id, confidence
    
    def train(self, train_loader, val_loader, epochs=50, learning_rate=0.001, log_dir='runs/train/cnn_model'):
        """
        训练CNNClassifier模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
            log_dir: TensorBoard日志目录
        """
        print(f"开始训练CNN模型，使用设备: {self.device}")
        
        # 设置优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
        
        # 设置TensorBoard
        from torch.utils.tensorboard import SummaryWriter
        import os
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard日志将保存到: {log_dir}")
        
        best_val_loss = float('inf')
        best_model_state = None
        
        # 计算总批次数，用于进度显示
        total_batches = len(train_loader)
        
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            # 训练循环开始时间
            start_time = time.time()
            
            # 显示进度的频率（每10%显示一次）
            display_freq = max(1, total_batches // 10)
            
            for i, (inputs, labels) in enumerate(train_loader):
                # 将数据移动到设备上
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 清除梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                # 累计损失
                train_loss += loss.item()
                
                # 计算准确率
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 显示批处理进度
                if (i + 1) % display_freq == 0 or (i + 1) == total_batches:
                    elapsed_time = time.time() - start_time
                    avg_time_per_batch = elapsed_time / (i + 1)
                    remain_batches = total_batches - (i + 1)
                    est_remain_time = remain_batches * avg_time_per_batch
                    
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{i+1}/{total_batches}] "
                          f"Loss: {train_loss/(i+1):.4f} Acc: {100.*correct/total:.2f}% "
                          f"已用时间: {elapsed_time:.1f}s 预计剩余: {est_remain_time:.1f}s")
            
            # 计算平均训练损失和准确率
            train_loss = train_loss / len(train_loader)
            train_acc = 100.0 * correct / total
            
            # 验证模式
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            # 计算平均验证损失和准确率
            val_loss = val_loss / len(val_loader)
            val_acc = 100.0 * correct / total
            
            # 记录到TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # 学习率调度器步进
            scheduler.step(val_loss)
            
            # 打印每个epoch的结果
            print(f"Epoch [{epoch+1}/{epochs}] 完成: "
                  f"TrainLoss: {train_loss:.4f}, TrainAcc: {train_acc:.2f}%, "
                  f"ValLoss: {val_loss:.4f}, ValAcc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                print(f"=> 保存新的最佳模型 (验证损失: {best_val_loss:.4f})")
        
        # 训练结束后，加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"训练完成，加载最佳模型 (验证损失: {best_val_loss:.4f})")
            
        # 添加模型图到TensorBoard
        try:
            # 获取一个批次的样本用于可视化模型结构
            sample_inputs, _ = next(iter(train_loader))
            sample_inputs = sample_inputs.to(self.device)
            writer.add_graph(self.model, sample_inputs)
            print("已将模型结构添加到TensorBoard")
        except Exception as e:
            print(f"添加模型图到TensorBoard时出错: {e}")
        
        # 关闭TensorBoard写入器
        writer.close()
    
    def evaluate(self, data_loader, criterion=None):
        """评估模型性能"""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        eval_loss = running_loss / len(data_loader)
        eval_acc = 100 * correct / total
        
        return eval_loss, eval_acc
    
    def save(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到 {path}")
    
    def load(self, path):
        """加载模型"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"模型已从 {path} 加载")
        return self
        
        # 自动生成cnn训练目录序号
        cnn_dirs = [d for d in os.listdir('runs/train') if d.startswith('cnn')]
        next_num = len(cnn_dirs) + 1
        save_dir = f'runs/train/cnn{next_num}'
        
        # 确保模型保存到指定目录
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))