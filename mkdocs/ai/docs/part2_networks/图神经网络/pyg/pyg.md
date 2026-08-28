# PyTorch Geometric (PyG) 入门中文文档

---

## 一、torch_geometric

PyTorch Geometric 的根包，提供图神经网络开发的完整框架。主要用于处理图结构数据的深度学习任务，包括节点分类、图分类、链接预测等场景。

> 官方网址：[PyG Documentation — pytorch_geometric documentation](https://pytorch-geometric.readthedocs.io/)  
> github 项目源代码：[pyg-team/pytorch_geometric: Graph Neural Network Library for PyTorch](https://github.com/pyg-team/pytorch_geometric?tab=readme-ov-file)

---

## 二、torch_geometric.nn

图神经网络的核心模块，包含各类图卷积层、池化层、归一化层等神经网络组件。

### 2.1 卷积层（Convolutional Layers）

实现各种图卷积操作，用于聚合节点邻居信息并更新节点表示。

#### 2.1.1 GCNConv

**图卷积网络层**。基于谱图理论的经典卷积方法，计算效率高，适合半监督节点分类任务。通过归一化的邻接矩阵聚合邻居特征。

**适用场景**：引文网络分类、社交网络分析、推荐系统。

#### 2.1.2 GATConv

**图注意力网络层**。通过注意力机制自动学习邻居节点的重要性权重，可以区分不同邻居的贡献。支持多头注意力机制。

**适用场景**：需要区分邻居重要性的任务、异质图、处理噪声边。

#### 2.1.3 SAGEConv

**GraphSAGE 层**。通过采样固定数量的邻居并聚合特征进行卷积，支持归纳学习（对未见过的节点泛化）。

**适用场景**：大规模图、动态图、需要对新节点推理的场景。

#### 2.1.4 GINConv

**图同构网络层**。理论上具有最强的图表示能力，能区分不同的图结构。需要传入一个 MLP 作为更新函数。

**适用场景**：图分类、分子性质预测、需要精确捕捉图结构差异的任务。

#### 2.1.5 EdgeConv

**边卷积层**。用于点云数据，通过动态构建 k 近邻图并在边上进行卷积操作。

**适用场景**：3D 点云分类、点云分割、几何深度学习。

#### 2.1.6 MessagePassing (基类)

所有卷积层的抽象基类，定义了消息传递的标准流程：消息构造 → 消息聚合 → 节点更新。自定义图卷积层时继承此类。

**核心方法**：`message()`, `aggregate()`, `update()`

### 2.2 池化层（Pooling Layers）

将图的节点级表示聚合为图级表示，或对图进行粗化。

#### 2.2.1 全局池化

- **global_mean_pool**：对所有节点特征求平均
- **global_max_pool**：对所有节点特征取最大值
- **global_add_pool**：对所有节点特征求和

**适用场景**：图分类任务，将节点表示转换为图表示。

#### 2.2.2 TopKPooling

基于节点重要性得分保留 Top-K 个节点，同时更新图结构。

**适用场景**：需要层次化图表示的任务、图生成。

#### 2.2.3 SAGPooling

自注意力图池化，通过可学习的注意力机制选择重要节点。

**适用场景**：复杂图分类任务、需要可解释性的场景。

### 2.3 归一化层（Normalization Layers）

用于稳定训练和加速收敛。

#### 2.3.1 BatchNorm

对节点特征进行批归一化，跨批次统计均值和方差。

#### 2.3.2 LayerNorm

对单个样本的特征维度进行归一化，适合小批量或动态图。

#### 2.3.3 GraphNorm

专门为图数据设计的归一化方法，考虑图结构的不规则性。

### 2.4 模型（Models）

预定义的完整图神经网络架构。

#### 2.4.1 GCN

完整的图卷积网络模型，包含多层 GCNConv 和激活函数。

#### 2.4.2 GAT

完整的图注意力网络模型。

#### 2.4.3 GraphSAGE

完整的 GraphSAGE 模型，适合大规模图和归纳学习。

---

## 三、torch_geometric.data

图数据的存储和表示模块。

### 3.1 Data

**核心数据结构**，用于存储单个图的所有信息。

**关键属性**：

- `x`：节点特征矩阵 `[num_nodes, num_features]`
- `edge_index`：边索引 `[2, num_edges]`，COO 格式
- `edge_attr`：边特征 `[num_edges, num_edge_features]`
- `y`：标签（节点级或图级）
- `pos`：节点空间坐标（用于点云或几何图）

**动态属性**：可自由添加任意属性，如 `data.custom_attr = tensor`

### 3.2 HeteroData

**异构图数据结构**，用于存储包含多种节点类型和边类型的图。

**适用场景**：知识图谱、生物网络、多关系图。

### 3.3 Batch

**批处理类**，将多个 `Data` 对象合并为一个大图，通过 `batch` 向量区分不同图的节点。

**自动生成属性**：`batch` 向量，标记每个节点属于哪个图。

### 3.4 Dataset (抽象类)

自定义数据集的基类，定义数据加载和处理的标准接口。

**核心方法**：

- `len()`：返回数据集大小
- `get(idx)`：返回第 idx 个数据

### 3.5 InMemoryDataset

将所有数据一次性加载到内存的数据集类，适合中小规模数据。

**核心方法**：

- `raw_file_names`：原始文件名列表
- `processed_file_names`：处理后文件名列表
- `download()`：下载原始数据
- `process()`：处理数据并保存

---

## 四、torch_geometric.loader

数据加载和批处理模块。

### 4.1 DataLoader

**基础数据加载器**，自动将多个图批处理为 `Batch` 对象。

**适用场景**：图分类任务、小规模图数据集。

**关键参数**：`batch_size`, `shuffle`, `num_workers`

### 4.2 NeighborLoader

**邻居采样加载器**，用于大图的节点级任务。每次采样目标节点的 k 跳邻居子图。

**适用场景**：大规模图的节点分类、链接预测。

**关键参数**：

- `num_neighbors`：每跳采样的邻居数量列表，如 `[10, 5]` 表示 2 跳采样
- `input_nodes`：目标节点（通常是训练/测试掩码）

### 4.3 LinkNeighborLoader

**链接预测专用加载器**，采样边的邻居子图。

**适用场景**：链接预测任务。

### 4.4 ClusterLoader

**图聚类加载器**，将大图划分为多个子图簇，每次加载一个簇。

**适用场景**：超大规模图、分布式训练。

### 4.5 GraphSAINTLoader

**GraphSAINT 采样加载器**，通过节点、边或随机游走采样子图。

**适用场景**：大图的全图训练、需要保持图结构完整性的任务。

---

## 五、torch_geometric.sampler

采样策略模块，定义各种图采样方法。

### 5.1 NeighborSampler

邻居采样器，定义如何采样多跳邻居。

### 5.2 NegativeSampling

负采样器，为链接预测生成负样本边。

**适用场景**：链接预测训练时生成负样本。

---

## 六、torch_geometric.datasets

内置数据集模块，提供常用图数据集的自动下载和处理。

### 6.1 节点分类数据集

#### 6.1.1 Planetoid

包含引文网络数据集：Cora, CiteSeer, PubMed。

**任务**：半监督节点分类。

#### 6.1.2 Amazon

亚马逊商品共购网络。

#### 6.1.3 Coauthor

合著者网络（CS 和 Physics 领域）。

### 6.2 图分类数据集

#### 6.2.1 TUDataset

包含多个分子和社交网络图分类数据集，如 MUTAG, PROTEINS, ENZYMES。

**任务**：图分类。

#### 6.2.2 QM9

分子性质预测数据集，包含 13 万个分子和多个量子化学性质标签。

**任务**：回归任务、图分类。

#### 6.2.3 ZINC

大规模分子数据集，用于图生成和性质预测。

### 6.3 点云数据集

#### 6.3.1 ModelNet

3D 模型分类数据集（ModelNet10 和 ModelNet40）。

**任务**：点云分类。

#### 6.3.2 ShapeNet

大规模 3D 形状分割数据集。

**任务**：点云分割。

### 6.4 其他数据集

#### 6.4.1 KarateClub

空手道俱乐部社交网络，用于社区检测。

#### 6.4.2 Entities

知识图谱数据集。

---

## 七、torch_geometric.llm

大语言模型与图神经网络结合的模块（实验性功能）。

**功能**：将图结构和文本信息联合建模，支持图问答、图文本匹配等任务。

**适用场景**：图推理、图文多模态学习。

---

## 八、torch_geometric.transforms

数据预处理和增强模块，提供图变换操作。

### 8.1 基础变换

#### 8.1.1 NormalizeFeatures

归一化节点特征，使每个节点的特征向量 L2 范数为 1。

#### 8.1.2 ToDevice

将数据转移到指定设备（CPU 或 GPU）。

#### 8.1.3 AddSelfLoops

为图添加自环边。

**适用场景**：GCN 等需要自环的模型。

#### 8.1.4 RemoveIsolatedNodes

移除图中的孤立节点（度为 0 的节点）。

### 8.2 图构建变换

#### 8.2.1 KNNGraph

根据节点坐标构建 k 近邻图。

**适用场景**：点云数据、几何图。

#### 8.2.2 RadiusGraph

根据半径构建图，连接距离小于阈值的节点。

#### 8.2.3 Delaunay

根据节点坐标构建 Delaunay 三角剖分图。

**适用场景**：2D/3D 点云、网格数据。

#### 8.2.4 FaceToEdge

将三角面片表示转换为边表示。

### 8.3 特征增强变换

#### 8.3.1 Distance

计算边的欧几里得距离并添加为边特征。

**适用场景**：几何图、点云。

#### 8.3.2 Cartesian

计算边的笛卡尔坐标差（Δx, Δy, Δz）并添加为边特征。

#### 8.3.3 Polar

计算边的极坐标表示（距离和角度）。

#### 8.3.4 LocalCartesian

计算局部坐标系下的坐标差。

### 8.4 数据增强变换

#### 8.4.1 RandomFlip

随机翻转节点坐标。

#### 8.4.2 RandomRotate

随机旋转节点坐标。

#### 8.4.3 RandomScale

随机缩放节点坐标。

#### 8.4.4 RandomJitter

为节点坐标添加随机噪声。

### 8.5 组合变换

#### 8.5.1 Compose

将多个变换组合为一个变换管道，按顺序执行。

```python
transform = T.Compose([
    T.NormalizeFeatures(),
    T.AddSelfLoops(),
    T.ToDevice('cuda')
])
```

### 8.6 自定义变换

继承 `BaseTransform` 类实现自定义变换，重写 `__call__` 方法。

---

## 九、torch_geometric.utils

图操作的工具函数模块。

### 9.1 图结构操作

#### 9.1.1 to_undirected

将有向图转换为无向图，添加反向边。

#### 9.1.2 add_self_loops

为图添加自环边。

#### 9.1.3 remove_self_loops

移除图中的自环边。

#### 9.1.4 coalesce

合并重复边，将多条边的属性求和或取平均。

#### 9.1.5 subgraph

提取包含指定节点的子图。

#### 9.1.6 k_hop_subgraph

提取指定节点的 k 跳邻居子图。

### 9.2 图转换

#### 9.2.1 to_dense_adj

将稀疏邻接矩阵（edge_index）转换为稠密邻接矩阵。

#### 9.2.2 to_scipy_sparse_matrix

转换为 SciPy 稀疏矩阵格式。

#### 9.2.3 to_networkx

转换为 NetworkX 图对象，便于可视化和分析。

#### 9.2.4 from_networkx

从 NetworkX 图对象转换为 PyG 的 Data 对象。

### 9.3 图计算

#### 9.3.1 degree

计算节点的度（入度、出度或总度）。

#### 9.3.2 softmax

在边上计算 softmax，常用于注意力机制。

#### 9.3.3 scatter

聚合操作，将多个值按索引聚合（求和、求平均、取最大等）。

### 9.4 采样相关

#### 9.4.1 negative_sampling

为链接预测任务生成负样本边。

#### 9.4.2 structured_negative_sampling

结构化负采样，返回正样本和对应的负样本。

---

## 十、torch_geometric.explain

模型可解释性模块，解释 GNN 的预测结果。

### 10.1 Explainer

可解释性框架的统一接口，支持多种解释算法。

### 10.2 GNNExplainer

经典的 GNN 解释方法，通过学习边和节点特征的掩码来解释预测。

**适用场景**：节点分类、图分类的可解释性分析。

### 10.3 PGExplainer

参数化的 GNN 解释器，通过神经网络预测边的重要性。

### 10.4 Captum 集成

支持 Captum 库的归因方法，如积分梯度、显著性图等。

---

## 十一、torch_geometric.metrics

评估指标模块。

### 11.1 accuracy

计算分类准确率。

### 11.2 auc

计算 AUC（Area Under Curve）。

**适用场景**：链接预测、二分类任务。

### 11.3 average_precision

计算平均精度。

**适用场景**：链接预测评估。

### 11.4 mean_reciprocal_rank

计算平均倒数排名（MRR）。

**适用场景**：知识图谱补全、推荐系统。

---

## 十二、torch_geometric.distributed

分布式训练模块，支持大规模图的多机多卡训练。

### 12.1 分布式数据划分

将大图划分到多个计算节点。

### 12.2 分布式采样

在多个节点上并行采样子图。

### 12.3 分布式训练

支持数据并行和模型并行的分布式训练策略。

**适用场景**：超大规模图（数十亿节点和边）。

---

## 十三、torch_geometric.contrib

社区贡献的实验性功能模块。

**内容**：包含尚未稳定的新功能、新模型和新算法，可能在未来版本中移入正式模块或移除。

---

## 十四、torch_geometric.graphgym

自动化实验框架，通过配置文件管理 GNN 实验。

### 14.1 配置驱动

使用 YAML 配置文件定义数据集、模型、训练参数。

### 14.2 模型注册

注册自定义模型、数据集和训练流程。

### 14.3 超参数搜索

自动化超参数调优和实验管理。

**适用场景**：大量实验对比、论文复现、超参数调优。

---

## 十五、torch_geometric.profile

性能分析模块，评估模型的计算效率。

### 15.1 模型速度分析

统计模型的前向传播时间。

### 15.2 内存使用分析

分析模型的显存占用。

### 15.3 FLOPs 计算

计算模型的浮点运算量。

**适用场景**：模型优化、效率对比、资源规划。

---

## 总结：模块功能速查表

| 模块                          | 核心功能           | 典型使用场景       |
| ----------------------------- | ------------------ | ------------------ |
| `torch_geometric.nn`          | 图神经网络层和模型 | 构建 GNN 架构      |
| `torch_geometric.data`        | 图数据结构         | 存储和表示图       |
| `torch_geometric.loader`      | 数据加载和批处理   | 训练时的数据迭代   |
| `torch_geometric.sampler`     | 采样策略           | 大图训练、负采样   |
| `torch_geometric.datasets`    | 内置数据集         | 快速实验和基准测试 |
| `torch_geometric.transforms`  | 数据预处理         | 特征工程、数据增强 |
| `torch_geometric.utils`       | 工具函数           | 图操作、转换、计算 |
| `torch_geometric.explain`     | 可解释性           | 模型解释和分析     |
| `torch_geometric.metrics`     | 评估指标           | 模型性能评估       |
| `torch_geometric.distributed` | 分布式训练         | 超大规模图         |
| `torch_geometric.graphgym`    | 自动化实验         | 实验管理和调优     |
| `torch_geometric.profile`     | 性能分析           | 效率评估和优化     |

---

## 学习路径建议

**初学者（第 1-2 周）**：

1. `torch_geometric.data`：理解 Data 对象
2. `torch_geometric.datasets`：加载内置数据集
3. `torch_geometric.nn`：学习 GCNConv、GATConv 等基础层
4. `torch_geometric.loader`：掌握 DataLoader

**进阶（第 3-4 周）**：

1. `torch_geometric.transforms`：数据预处理管道
2. `torch_geometric.utils`：图操作工具
3. `torch_geometric.nn`：池化层、自定义卷积层
4. 实现完整的节点分类和图分类任务

**高级（第 5+ 周）**：

1. `torch_geometric.loader`：NeighborLoader 处理大图
2. `torch_geometric.explain`：模型可解释性
3. `torch_geometric.distributed`：分布式训练
4. 自定义 Dataset、Transform 和 MessagePassing 层
