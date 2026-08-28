# ResNet 实践教程 Jupyter Notebook 代码大纲

## 文件说明

本文件提供了配套 ResNet 教程的 Jupyter Notebook 代码大纲，包含核心代码实现和实验步骤。代码设计注重简洁性，突出重点概念，适合学生学习。

---

## 📋 Notebook 整体结构

### 第一部分：环境设置与数据准备

- 导入必要的库
- 设置随机种子保证实验可复现
- 准备数据集（CIFAR-10）
- 实现数据增强

### 第二部分：ResNet 核心组件实现

- 实现基础残差块（BasicBlock）
- 构建完整的ResNet-18网络
- 添加维度匹配机制

### 第三部分：训练与测试流程

- 定义训练循环
- 实现评估函数
- 设置训练参数

### 第四部分：实验验证

- 训练ResNet-18
- 可视化训练过程
- 性能分析

### 第五部分：迁移学习实践

- 使用预训练ResNet-50
- 微调实践
- 对比实验

---

## 📝 详细代码大纲

### 1. 环境设置与数据准备

```python
# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子保证可复现性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
```

#### 代码要点：

- 只导入核心库，避免过度依赖
- 设置随机种子是机器学习实践的基础
- 确保CUDA环境的一致性

```python
# 数据增强配置
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
```

#### 学习重点：

- 理解训练和测试时数据增强的区别
- 掌握标准化参数的含义
- 认识到数据增强对模型性能的重要性

---

### 2. ResNet 核心组件实现

#### 2.1 基础残差块（BasicBlock）

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 处理维度不匹配的情况
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)))
        out += self.shortcut(x)  # 关键：残差连接
        out = F.relu(out)
        return out
```

#### 学习重点：

- 理解残差连接的本质：`out += self.shortcut(x)`
- 掌握BatchNorm在激活函数前的位置
- 认识到维度匹配的重要性

#### 实验建议：

- 创建一个简单的输入tensor，测试BasicBlock的输出
- 对比有无残差连接的梯度流动

#### 2.2 完整的ResNet-18实现

```python
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 残差块组
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # 最终分类层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(num_blocks - 1):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
```

#### 学习重点：

- 理解ResNet的层次结构
- 掌握`_make_layer`的复用机制
- 认识到网络结构的模块化设计

---

### 3. 训练与测试流程

#### 3.1 训练函数

```python
def train_model(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc
```

#### 3.2 测试函数

```python
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc
```

#### 学习重点：

- 理解训练和测试模式的区别
- 掌握梯度计算和参数更新的流程
- 认识到模型评估的标准化方法

---

### 4. 实验验证

#### 4.1 主训练循环

```python
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128,
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100,
                            shuffle=False, num_workers=2)

    # 创建模型
    model = ResNet18(num_classes=10).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                         weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 训练循环
    num_epochs = 50
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion,
                                           optimizer, device, epoch)
        test_loss, test_acc = test_model(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        scheduler.step()

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print('-' * 50)
```

#### 学习重点：

- 理解完整的训练流程
- 掌握学习率调度的使用
- 认识到模型性能监控的重要性

#### 4.2 结果可视化

```python
def plot_results(train_losses, train_accs, test_losses, test_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()

    # 准确率曲线
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(test_accs, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()
```

---

### 5. 迁移学习实践

#### 5.1 使用预训练ResNet-50

```python
from torchvision import models

# 加载预训练模型
pretrained_model = models.resnet50(pretrained=True)

# 修改最后的全连接层
num_ftrs = pretrained_model.fc.in_features
pretrained_model.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10有10个类别

# 冻结前面的层
for param in pretrained_model.parameters():
    param.requires_grad = False

# 只训练最后的全连接层
pretrained_model.fc.requires_grad = True
pretrained_model = pretrained_model.to(device)

# 定义优化器（只优化全连接层）
optimizer = optim.Adam(pretrained_model.fc.parameters(), lr=0.001)
```

#### 学习重点：

- 理解预训练模型的优势
- 掌握迁移学习的基本流程
- 认识到冻结层的作用和意义

#### 5.2 对比实验

```python
def compare_models():
    # 比较从头训练的ResNet-18和预训练的ResNet-50
    models_to_compare = {
        'ResNet-18 (from scratch)': ResNet18(num_classes=10).to(device),
        'ResNet-50 (pretrained)': pretrained_model
    }

    results = {}

    for name, model in models_to_compare.items():
        print(f'\nTraining {name}...')
        # 训练和测试代码...
        results[name] = {
            'test_accuracy': final_test_acc,
            'training_time': training_time
        }

    # 显示对比结果
    print('\nFinal Results:')
    for name, metrics in results.items():
        print(f'{name}: {metrics["test_accuracy"]:.2f}%')
```

---

## 🎯 学习路径建议

### 阶段一：基础理解（第1-2次运行）

1. 运行完整训练流程，观察ResNet-18的训练过程
2. 理解残差块的作用，尝试修改跳跃连接
3. 对比有无残差连接的模型性能

### 阶段二：深入实践（第3-4次运行）

1. 调整学习率调度策略，观察对训练的影响
2. 尝试不同的数据增强策略
3. 分析训练过程中的梯度流动

### 阶段三：扩展应用（第5-6次运行）

1. 实施迁移学习实践
2. 对比不同模型的性能
3. 探索ResNet在不同数据集上的表现

---

## 📊 预期学习成果

完成本教程后，您将能够：

- 理解ResNet的核心思想和实现原理
- 从头实现一个完整的ResNet网络
- 掌握深度网络的训练和调试技巧
- 理解迁移学习的基本概念和应用
- 具备调试和优化深度学习模型的能力

---

## 🚀 扩展实验建议

1. **深度影响实验**：对比不同深度ResNet的性能（ResNet-18 vs ResNet-34）
2. **瓶颈结构实验**：实现Bottleneck块，观察性能差异
3. **正则化实验**：尝试不同的正则化策略
4. **可视化实验**：可视化中间特征图和梯度
5. **消融实验**：逐步移除ResNet的关键组件，观察影响

---

## ⚠️ 常见问题与解决方案

### 问题1：显存不足

**解决方案**：减小batch_size或使用梯度累积

### 问题2：训练不收敛

**解决方案**：检查学习率设置，尝试使用学习率预热

### 问题3：过拟合

**解决方案**：增加数据增强强度，使用正则化技术

### 问题4：训练速度慢

**解决方案**：使用GPU加速，优化数据加载流程

---

## 📚 参考资源

- 原始论文：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- PyTorch官方文档：[torchvision.models](https://pytorch.org/vision/stable/models.html)
- CIFAR-10数据集：[CIFAR官网](https://www.cs.toronto.edu/~kriz/cifar.html)

---

**备注**：本代码大纲配合理论教程使用，建议先理解理论概念，再进行代码实践。每个代码块都应该独立运行和测试，逐步深入理解ResNet的工作原理。
