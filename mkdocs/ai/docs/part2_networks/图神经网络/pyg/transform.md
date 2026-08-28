# PyTorch Geometric Transforms 完全教程

## 📋 目录

1. [Transform 基础概念](#transform)
2. [通用变换（General Transforms）](#general-transforms)
3. [图变换（Graph Transforms）](#graph-transforms)
4. [视觉变换（Vision Transforms）](#vision-transforms)
5. [自定义 Transform](#transform_1)
6. [最佳实践 Practice](#practice)
7. [实战案例 Example](#example)
8. [总结 Summary](#summary)

---

## 一、Transform 基础概念

### 1.1 什么是 Transform？

**Transform 是对图数据执行预处理和增强操作的可调用对象**。它接收一个 `Data` 或 `HeteroData` 对象作为输入，返回修改后的数据对象。

```python
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures

# 创建原始图数据
data = Data(x=torch.randn(10, 16), edge_index=edge_index)

# 应用变换
transform = NormalizeFeatures()
transformed_data = transform(data)
```

**核心特点**：

- **可组合**：多个变换可通过 `Compose` 组合成管道
- **惰性执行**：变换在调用时才执行
- **状态保持**：变换对象可携带参数和状态
- **链式传递**：输出可作为下一个变换的输入

### 1.2 BaseTransform 基类

所有变换继承自 `BaseTransform`，需实现 `__call__` 方法。

```python
from torch_geometric.transforms import BaseTransform

class MyTransform(BaseTransform):
    def __init__(self, param=True):
        self.param = param

    def __call__(self, data):
        """核心方法：定义变换逻辑"""
        # 修改 data 对象
        data.new_attr = some_value
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(param={self.param})'
```

### 1.3 变换的应用方式

#### 方式1：预变换（Pre-transform）

在数据集第一次处理时应用，结果会被缓存。

```python
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

# 预变换只在数据集构建时执行一次
pre_transform = T.Compose([
    T.NormalizeFeatures(),
    T.AddSelfLoops(),
])

dataset = Planetoid(
    root='data/Cora',
    name='Cora',
    pre_transform=pre_transform  # 结果会保存到 processed/
)
```

**特点**：

- 执行一次，结果缓存
- 适合计算密集型操作（如构建 k-NN 图）
- 删除 `processed/` 目录可重新执行

#### 方式2：动态变换（Transform）

每次访问数据时都会执行。

```python
# 动态变换在每次 __getitem__ 时执行
transform = T.Compose([
    T.RandomRotate(degrees=15),  # 数据增强
    T.RandomJitter(0.01),
])

dataset = Planetoid(
    root='data/Cora',
    name='Cora',
    transform=transform  # 每次加载时执行
)
```

**特点**：

- 每次访问都执行
- 适合数据增强（随机变换）
- 不会缓存结果

#### 方式3：直接调用

手动应用变换。

```python
data = dataset[0]
transform = T.NormalizeFeatures()
data = transform(data)
```

### 1.4 Compose 组合器

将多个变换按顺序组合成管道。

```python
from torch_geometric.transforms import Compose

transform = Compose([
    T.NormalizeFeatures(),      # 第1步
    T.AddSelfLoops(),           # 第2步
    T.ToDevice('cuda'),         # 第3步
])

# 等价于
data = T.NormalizeFeatures()(data)
data = T.AddSelfLoops()(data)
data = T.ToDevice('cuda')(data)
```

---

## 二、通用变换（General Transforms）

适用于所有图数据的基础变换。

### 2.1 特征归一化

#### NormalizeFeatures

将节点特征归一化，使每个节点的特征向量的 L2 范数为 1。

$$
\mathbf{x}_i \leftarrow \frac{\mathbf{x}_i}{\|\mathbf{x}_i\|_2}
$$

```python
from torch_geometric.transforms import NormalizeFeatures

transform = NormalizeFeatures()
data = transform(data)
# data.x: 每行的 L2 范数都为 1
```

**适用场景**：

- 特征尺度差异大
- 需要归一化输入的模型（如 GCN）
- 避免数值不稳定

#### NormalizeScale

将节点坐标归一化到 $[-1, 1]$ 或 $[0, 1]$ 范围。

```python
from torch_geometric.transforms import NormalizeScale

transform = NormalizeScale()
data = transform(data)
# data.pos: 坐标范围 [-1, 1]
```

**适用场景**：

- 点云数据
- 几何图神经网络
- 需要标准化空间坐标

### 2.2 拓扑结构修改

#### AddSelfLoops

为图添加自环边（节点到自身的边）。

```python
from torch_geometric.transforms import AddSelfLoops

transform = AddSelfLoops()
data = transform(data)
# 每个节点都会有一条指向自己的边
```

**原理**：

```python
# 原始边
edge_index = [[0, 1, 2], [1, 2, 0]]  # 3条边

# 添加自环后
edge_index = [[0, 1, 2, 0, 1, 2],
              [1, 2, 0, 0, 1, 2]]  # 6条边（3条原始 + 3条自环）
```

**适用场景**：

- GCN 等需要自环的模型
- 保留节点自身特征的聚合

#### RemoveIsolatedNodes

移除孤立节点（度为 0 的节点）。

```python
from torch_geometric.transforms import RemoveIsolatedNodes

transform = RemoveIsolatedNodes()
data = transform(data)
# 孤立节点会被移除，节点索引会重新编号
```

**适用场景**：

- 清理数据集
- 减少无效计算
- 图分类任务

#### ToUndirected

将有向图转换为无向图。

```python
from torch_geometric.transforms import ToUndirected

transform = ToUndirected()
data = transform(data)
# 为每条边添加反向边
```

**原理**：

```python
# 有向边
edge_index = [[0, 1, 2], [1, 2, 0]]

# 转换为无向边
edge_index = [[0, 1, 1, 2, 2, 0],
              [1, 0, 2, 1, 0, 2]]  # 每条边都有反向边
```

#### LineGraph

将原图转换为线图（边变为节点，相邻边变为边）。

```python
from torch_geometric.transforms import LineGraph

transform = LineGraph()
data = transform(data)
# 原图的边成为新图的节点
```

**适用场景**：

- 边预测任务
- 分析边之间的关系
- 道路网络分析

### 2.3 采样与子图

#### RandomNodeSplit

随机划分节点为训练集/验证集/测试集。

```python
from torch_geometric.transforms import RandomNodeSplit

transform = RandomNodeSplit(
    split='train_rest',
    num_val=0.1,
    num_test=0.2
)
data = transform(data)
# 生成 data.train_mask, data.val_mask, data.test_mask
```

**适用场景**：

- 节点分类任务
- 创建数据划分
- 交叉验证

#### RandomLinkSplit

随机划分边为训练集/验证集/测试集。

```python
from torch_geometric.transforms import RandomLinkSplit

transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.2,
    is_undirected=True,
    add_negative_train_samples=True
)
train_data, val_data, test_data = transform(data)
```

**适用场景**：

- 链接预测任务
- 边分类任务
- 推荐系统

### 2.4 设备管理

#### ToDevice

将数据转移到指定设备（CPU 或 GPU）。

```python
from torch_geometric.transforms import ToDevice

transform = ToDevice('cuda')
data = transform(data)
# 所有张量都转移到 GPU
```

**适用场景**：

- 模型训练前的数据准备
- 多 GPU 训练
- 预处理时统一设备

### 2.5 数据类型转换

#### ToSparseTensor

将 `edge_index` 转换为 `SparseTensor` 格式。

```python
from torch_geometric.transforms import ToSparseTensor

transform = ToSparseTensor()
data = transform(data)
# data.adj_t: SparseTensor 格式的邻接矩阵
```

**优势**：

- 更高效的稀疏矩阵操作
- 减少内存占用
- 支持更多矩阵运算

---

## 三、图变换（Graph Transforms）

用于构建、修改和增强图结构的变换。

### 3.1 图构建变换

#### KNNGraph

根据节点坐标构建 k 近邻图。

```python
from torch_geometric.transforms import KNNGraph

transform = KNNGraph(k=6, loop=False)
data = transform(data)
# 连接每个节点的 6 个最近邻居
```

**参数**：

- `k`：近邻数量
- `loop`：是否包含自环
- `force_undirected`：是否强制无向图

**适用场景**：

- 点云数据
- 没有预定义边的数据
- 几何深度学习

**示例**：

```python
# 输入：节点坐标
pos = [[0, 0], [1, 0], [0, 1], [2, 2]]

# k=2 的 k-NN 图
# 节点0 最近的2个邻居: 节点1, 节点2
# 节点1 最近的2个邻居: 节点0, 节点3
# ...
```

#### RadiusGraph

根据半径阈值构建图，连接距离小于 $r$ 的节点对。

```python
from torch_geometric.transforms import RadiusGraph

transform = RadiusGraph(r=0.5, loop=False)
data = transform(data)
# 连接距离 ≤ 0.5 的节点对
```

**适用场景**：

- 需要物理距离约束的问题
- 点云分割
- 分子图（化学键范围）

#### Delaunay

根据节点坐标构建 Delaunay 三角剖分图。

```python
from torch_geometric.transforms import Delaunay

transform = Delaunay()
data = transform(data)
# data.face: [3, num_faces] 三角形顶点索引
```

**特点**：

- 生成三角形面片 `face`
- 最大化最小角（避免狭长三角形）
- 需要配合 `FaceToEdge` 转换为边

**适用场景**：

- 2D 点云
- 网格生成
- 计算流体力学（CFD）

#### FaceToEdge

将三角形面片表示转换为边表示。

```python
from torch_geometric.transforms import FaceToEdge

transform = FaceToEdge(remove_faces=True)
data = transform(data)
# 将 data.face 转换为 data.edge_index
```

**原理**：

```python
# 输入：三角形 [v0, v1, v2]
face = [[0, 1, 2]]

# 输出：三条边
edge_index = [[0, 1, 1, 2, 2, 0],  # 双向边
              [1, 0, 2, 1, 0, 2]]
```

### 3.2 特征增强变换

#### Distance

计算边的欧几里得距离并添加为边特征。

```python
from torch_geometric.transforms import Distance

transform = Distance(norm=True, cat=True)
data = transform(data)
# data.edge_attr: 包含边长度
```

**参数**：

- `norm`：是否归一化到 $[0, 1]$
- `cat`：是否拼接到现有 `edge_attr`

**公式**：

$$
d_{ij} = \|\mathbf{p}_j - \mathbf{p}_i\|_2
$$

#### Cartesian

计算边的笛卡尔坐标差（相对位置向量）。

```python
from torch_geometric.transforms import Cartesian

transform = Cartesian(norm=True, cat=True)
data = transform(data)
# data.edge_attr: 包含 [Δx, Δy, Δz]
```

**公式**：

$$
\mathbf{e}_{ij} = \mathbf{p}_j - \mathbf{p}_i
$$

**适用场景**：

- 编码空间相对位置
- 点云分割
- 分子性质预测

#### Polar

计算边的极坐标表示（距离和角度）。

```python
from torch_geometric.transforms import Polar

transform = Polar(norm=True, cat=True)
data = transform(data)
# data.edge_attr: 包含 [r, θ]
```

**公式**：

$$
r_{ij} = \|\mathbf{p}_j - \mathbf{p}_i\|_2, \quad \theta_{ij} = \arctan\left(\frac{\Delta y}{\Delta x}\right)
$$

#### LocalCartesian

计算局部坐标系下的相对坐标。

```python
from torch_geometric.transforms import LocalCartesian

transform = LocalCartesian(norm=True, cat=True)
data = transform(data)
# data.edge_attr: 局部坐标系下的相对位置
```

**适用场景**：

- 旋转不变性学习
- 分子构象
- 点云对齐

### 3.3 数据增强变换

#### RandomFlip

随机翻转节点坐标。

```python
from torch_geometric.transforms import RandomFlip

transform = RandomFlip(axis=0, p=0.5)
data = transform(data)
# 以 50% 概率沿 x 轴翻转
```

**参数**：

- `axis`：翻转轴（0=x, 1=y, 2=z）
- `p`：翻转概率

#### RandomRotate

随机旋转节点坐标。

```python
from torch_geometric.transforms import RandomRotate

transform = RandomRotate(degrees=180, axis=2)
data = transform(data)
# 绕 z 轴随机旋转 [-180°, +180°]
```

**适用场景**：

- 点云分类
- 增强旋转不变性
- 几何数据增强

#### RandomScale

随机缩放节点坐标。

```python
from torch_geometric.transforms import RandomScale

transform = RandomScale(scales=[0.8, 1.2])
data = transform(data)
# 随机缩放 0.8x 到 1.2x
```

#### RandomJitter

为节点坐标添加随机高斯噪声。

```python
from torch_geometric.transforms import RandomJitter

transform = RandomJitter(translate=0.01)
data = transform(data)
# 添加标准差为 0.01 的噪声
```

**适用场景**：

- 数据增强
- 鲁棒性训练
- 模拟测量误差

#### RandomShear

随机剪切变换。

```python
from torch_geometric.transforms import RandomShear

transform = RandomShear(shear=0.2)
data = transform(data)
```

### 3.4 图采样变换

#### FixedPoints

从图中采样固定数量的节点。

```python
from torch_geometric.transforms import FixedPoints

transform = FixedPoints(num=1024, replace=False)
data = transform(data)
# 随机采样 1024 个节点及其相关边
```

**适用场景**：

- 点云下采样
- 统一输入大小
- 减少计算量

#### SamplePoints

按比例采样节点。

```python
from torch_geometric.transforms import SamplePoints

transform = SamplePoints(num=2048)
data = transform(data)
```

### 3.5 图聚合变换

#### ToDense

将批量图转换为稠密表示（固定大小的邻接矩阵）。

```python
from torch_geometric.transforms import ToDense

transform = ToDense(num_nodes=100)
data = transform(data)
# data.adj: [batch_size, num_nodes, num_nodes]
```

**适用场景**：

- 需要稠密矩阵的模型
- 图分类任务
- 小规模图

#### TwoHop

扩展图到二跳邻居。

```python
from torch_geometric.transforms import TwoHop

transform = TwoHop()
data = transform(data)
# 添加二跳邻居的边
```

**原理**：

```
原始图：A -- B -- C
添加二跳边：A -- B -- C
             \-------/
```

---

## 四、视觉变换（Vision Transforms）

专门用于处理图像和点云数据的变换。

### 4.1 点云变换

#### GridSampling

在 3D 网格中进行下采样。

```python
from torch_geometric.transforms import GridSampling

transform = GridSampling(size=0.01)
data = transform(data)
# 将点云划分为 0.01×0.01×0.01 的网格，每个网格保留一个点
```

**适用场景**：

- 大规模点云下采样
- 保持空间分布
- 减少计算量

#### Center

将点云中心移到原点。

```python
from torch_geometric.transforms import Center

transform = Center()
data = transform(data)
# data.pos -= data.pos.mean(dim=0)
```

#### PointPairFeatures

计算点对特征（法向量相关）。

```python
from torch_geometric.transforms import PointPairFeatures

transform = PointPairFeatures()
data = transform(data)
# 需要 data.pos 和 data.normal
```

**特征**：

- 点间距离
- 法向量角度
- 旋转不变性特征

### 4.2 图像到图转换

#### Grayscale

将 RGB 图像转换为灰度图。

```python
from torch_geometric.transforms import Grayscale

transform = Grayscale()
data = transform(data)
# 将 3 通道转换为 1 通道
```

---

## 五、自定义 Transform

### 5.1 基础自定义 Transform

```python
from torch_geometric.transforms import BaseTransform
import torch

class MyDistance(BaseTransform):
    """计算边长度并添加为边特征"""

    def __init__(self, norm: bool = False):
        """
        Args:
            norm: 是否归一化到 [0, 1]
        """
        self.norm = norm

    def __call__(self, data):
        """
        Args:
            data: Data 对象，必须包含 pos 和 edge_index

        Returns:
            修改后的 Data 对象
        """
        row, col = data.edge_index
        pos = data.pos

        # 计算欧几里得距离
        dist = torch.norm(pos[col] - pos[row], p=2, dim=-1, keepdim=True)

        # 归一化
        if self.norm and dist.numel() > 0:
            dist = dist / dist.max()

        # 拼接到边特征
        if data.edge_attr is not None:
            data.edge_attr = torch.cat([data.edge_attr, dist], dim=-1)
        else:
            data.edge_attr = dist

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(norm={self.norm})'
```

### 5.2 使用装饰器注册

```python
from torch_geometric.data.datapipes import functional_transform

@functional_transform('my_distance')
class MyDistance(BaseTransform):
    def __init__(self, norm: bool = False):
        self.norm = norm

    def __call__(self, data):
        # 实现逻辑
        return data

# 现在支持两种调用方式
import torch_geometric.transforms as T

# 方式1：类实例化
transform = MyDistance(norm=True)
data = transform(data)

# 方式2：函数式（装饰器提供）
data = T.my_distance(data, norm=True)
```

### 5.3 处理边界条件的 Transform

```python
class DirichletBC(BaseTransform):
    """标记 Dirichlet 边界节点"""

    def __init__(self, boundary_nodes=None):
        self.boundary_nodes = boundary_nodes

    def set_boundary(self, indices):
        """动态设置边界节点"""
        self.boundary_nodes = torch.tensor(indices, dtype=torch.long)

    def __call__(self, data):
        if self.boundary_nodes is None:
            raise ValueError("Boundary nodes not set")

        # 标记边界节点
        data.dirichlet_index = self.boundary_nodes

        # 创建边界掩码
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[self.boundary_nodes] = True
        data.dirichlet_mask = mask

        return data
```

### 5.4 组合自定义 Transform

```python
from torch_geometric.transforms import Compose

# 完整的数据处理管道
transform = Compose([
    # 1. 构建图结构
    T.Delaunay(),
    T.FaceToEdge(remove_faces=True),

    # 2. 计算几何特征
    MyDistance(norm=True),
    T.Cartesian(norm=True, cat=True),

    # 3. 标记边界
    DirichletBC(),

    # 4. 归一化
    T.NormalizeFeatures(),
])

dataset = MyDataset(root='data/', pre_transform=transform)
```

---

## 六、最佳实践 Practice

### 6.1 Transform 设计原则

#### 原则1：单一职责

每个 Transform 只做一件事。

```python
# ✅ 好的设计
class ComputeDistance(BaseTransform):
    """只计算距离"""
    pass

class NormalizeDistance(BaseTransform):
    """只归一化"""
    pass

# ❌ 不好的设计
class ComputeAndNormalizeDistance(BaseTransform):
    """做太多事情，难以复用"""
    pass
```

#### 原则2：保持幂等性

多次应用相同变换应该得到相同结果（对于非随机变换）。

```python
class AddConstant(BaseTransform):
    def __call__(self, data):
        # ❌ 每次调用都会累加
        data.x = data.x + 1
        return data

class SetConstant(BaseTransform):
    def __call__(self, data):
        # ✅ 幂等操作
        data.constant = 1
        return data
```

#### 原则3：避免隐式依赖

明确声明所需属性。

```python
class MyTransform(BaseTransform):
    def __call__(self, data):
        # ✅ 检查依赖
        if not hasattr(data, 'pos'):
            raise ValueError("Data must have 'pos' attribute")

        # 处理逻辑
        return data
```

### 6.2 性能优化技巧

#### 技巧1：使用 InMemoryDataset 缓存

```python
class MyDataset(InMemoryDataset):
    def __init__(self, root, pre_transform=None):
        super().__init__(root, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        # 预处理只执行一次，结果缓存
        data_list = [self.create_graph(i) for i in range(100)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # 保存到磁盘
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
```

#### 技巧2：批量处理

```python
class BatchTransform(BaseTransform):
    """批量处理多个图"""

    def __call__(self, data_list):
        # 并行处理
        return [self.transform_single(d) for d in data_list]
```

#### 技巧3：惰性计算

```python
class LazyTransform(BaseTransform):
    def __init__(self):
        self._cache = None

    def __call__(self, data):
        # 只在第一次调用时计算
        if self._cache is None:
            self._cache = expensive_computation(data)

        data.feature = self._cache
        return data
```

### 6.3 调试技巧

#### 技巧1：逐步验证

```python
# 单独测试每个 Transform
data = dataset.get(0)

print("原始数据:", data)

t1 = T.Delaunay()
data = t1(data)
print("Delaunay 后:", data)

t2 = T.FaceToEdge()
data = t2(data)
print("FaceToEdge 后:", data)
```

#### 技巧2：可视化中间结果

```python
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

def visualize_transform(data, title):
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(6, 6))
    pos = {i: data.pos[i].numpy() for i in range(data.num_nodes)}
    nx.draw(G, pos, node_size=50, node_color='blue')
    plt.title(title)
    plt.show()

# 应用变换前
visualize_transform(data, "Before Transform")

# 应用变换后
data = transform(data)
visualize_transform(data, "After Transform")
```

#### 技巧3：单元测试

```python
import unittest

class TestMyTransform(unittest.TestCase):
    def test_distance_calculation(self):
        # 创建测试数据
        data = Data(
            pos=torch.tensor([[0., 0.], [1., 0.], [0., 1.]]),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]])
        )

        # 应用变换
        transform = MyDistance(norm=False)
        data = transform(data)

        # 验证结果
        expected_dist = torch.tensor([[1.0], [1.0], [1.414]])
        torch.testing.assert_close(data.edge_attr, expected_dist, atol=1e-3, rtol=1e-3)

    def test_with_existing_edge_attr(self):
        # 测试拼接行为
        data = Data(
            pos=torch.tensor([[0., 0.], [1., 0.]]),
            edge_index=torch.tensor([[0], [1]]),
            edge_attr=torch.tensor([[0.5]])
        )

        transform = MyDistance(norm=False)
        data = transform(data)

        # 应该拼接到原有边特征
        self.assertEqual(data.edge_attr.shape, (1, 2))  # [原特征, 距离]
        torch.testing.assert_close(data.edge_attr[:, 0], torch.tensor([0.5]))
        torch.testing.assert_close(data.edge_attr[:, 1], torch.tensor([1.0]))
```

### 6.4 常见错误与解决方案

#### 错误1：修改了输入但未返回

```python
# ❌ 错误
class BadTransform(BaseTransform):
    def __call__(self, data):
        data.new_attr = 1
        # 忘记 return

# ✅ 正确
class GoodTransform(BaseTransform):
    def __call__(self, data):
        data.new_attr = 1
        return data  # 必须返回
```

#### 错误2：维度不匹配

```python
# ❌ 错误：边特征维度不一致
data.edge_attr = torch.randn(10, 3)  # 10条边，3维特征
new_feature = torch.randn(10)        # 缺少维度
data.edge_attr = torch.cat([data.edge_attr, new_feature], dim=-1)  # 报错

# ✅ 正确
new_feature = new_feature.view(-1, 1)  # 调整为 [10, 1]
data.edge_attr = torch.cat([data.edge_attr, new_feature], dim=-1)
```

#### 错误3：边界情况未处理

```python
# ❌ 错误：空图时崩溃
class BadNormalize(BaseTransform):
    def __call__(self, data):
        data.x = data.x / data.x.max()  # 如果 data.x 为空会报错
        return data

# ✅ 正确
class GoodNormalize(BaseTransform):
    def __call__(self, data):
        if data.x.numel() > 0:  # 检查非空
            max_val = data.x.max()
            if max_val > 0:  # 避免除零
                data.x = data.x / max_val
        return data
```

---

## 七、实战案例 Example

### 7.1 完整的点云分类管道

```python
import torch_geometric.transforms as T

# 点云数据预处理
pre_transform = T.Compose([
    T.NormalizeScale(),              # 归一化坐标
    T.KNNGraph(k=20),               # 构建 k-NN 图
    T.Distance(norm=True, cat=True), # 边长度特征
])

# 训练时数据增强
train_transform = T.Compose([
    T.RandomRotate(degrees=180, axis=2),  # 随机旋转
    T.RandomJitter(translate=0.01),       # 添加噪声
    T.RandomScale(scales=[0.9, 1.1]),     # 随机缩放
])

from torch_geometric.datasets import ModelNet
dataset = ModelNet(
    root='data/ModelNet10',
    name='10',
    pre_transform=pre_transform,
    transform=train_transform  # 只在训练时应用
)
```

### 7.2 分子性质预测管道

```python
# 分子图预处理
pre_transform = T.Compose([
    T.Distance(norm=False, cat=True),    # 原子间距离
    T.AddSelfLoops(),                    # 自环（考虑原子自身）
])

from torch_geometric.datasets import QM9
dataset = QM9(
    root='data/QM9',
    pre_transform=pre_transform
)
```

### 7.3 CFD 网格处理管道（PhyMPGN 案例）

```python
# 完整的流体力学数据处理
class FluidTransform:
    def __init__(self):
        # 边界条件标记
        self.dirichlet = Dirichlet()       # 固定值边界
        self.neumann = Neumann()           # 零梯度边界

        # 图结构构建
        self.graph = T.Compose([
            T.Delaunay(),                  # 三角剖分
            MaskFace(),                    # 移除障碍物内部
            T.FaceToEdge(),                # 面转边
            MyDistance(norm=True),         # 边长度
            MyCartesian(norm=True),        # 边方向
        ])

    def __call__(self, data):
        # 1. 标记边界
        data = self.dirichlet(data)
        data = self.neumann(data)

        # 2. 构建图
        data = self.graph(data)

        return data
```

---

## 八、总结 Summary

### 核心要点

| 分类         | 典型 Transform                        | 主要用途     |
| ------------ | ------------------------------------- | ------------ |
| **通用**     | `NormalizeFeatures`, `AddSelfLoops`   | 基础预处理   |
| **图构建**   | `KNNGraph`, `RadiusGraph`, `Delaunay` | 从坐标构建图 |
| **特征增强** | `Distance`, `Cartesian`, `Polar`      | 添加几何特征 |
| **数据增强** | `RandomRotate`, `RandomJitter`        | 训练时增强   |
| **采样**     | `FixedPoints`, `RandomNodeSplit`      | 子图采样     |
| **视觉**     | `GridSampling`, `Center`              | 点云处理     |

### 学习路径

**第1阶段（1-2天）**：

- 理解 `BaseTransform` 和 `__call__` 机制
- 使用内置 Transform：`NormalizeFeatures`, `AddSelfLoops`, `ToDevice`
- 学习 `Compose` 组合

**第2阶段（3-5天）**：

- 图构建 Transform：`KNNGraph`, `RadiusGraph`
- 特征增强：`Distance`, `Cartesian`
- 数据增强：`RandomRotate`, `RandomJitter`

**第3阶段（1周）**：

- 编写自定义 Transform
- 理解 `pre_transform` vs `transform`
- 处理边界条件（物理问题）

**第4阶段（持续）**：

- 性能优化和缓存
- 复杂管道设计
- 领域特定 Transform

### 关键设计模式

```python
# 模式1：可配置 Transform
class ConfigurableTransform(BaseTransform):
    def __init__(self, param1=True, param2=1.0):
        self.param1 = param1
        self.param2 = param2

# 模式2：状态共享 Transform
class StatefulTransform(BaseTransform):
    def __init__(self):
        self._state = None

    def compute_state(self, data):
        self._state = expensive_computation(data)

    def __call__(self, data):
        if self._state is None:
            self.compute_state(data)
        return self.apply(data)

# 模式3：条件 Transform
class ConditionalTransform(BaseTransform):
    def __call__(self, data):
        if hasattr(data, 'pos'):
            # 处理点云
            pass
        else:
            # 处理特征图
            pass
        return data
```

Transform 是 PyG 数据处理的核心抽象，掌握它们的设计和使用是构建高效 GNN 系统的关键。
