# AI 技术学习资源中心

## 项目概述

本资源中心是课题组内部的技术学习与知识分享平台，主要面向流体力学、AI交叉领域的研究人员。基于 MkDocs + Material 主题构建的中文技术文档网站，提供从零基础入门到前沿研究的系统性学习路径。

## 核心资源

### 📚 mkdocs/ai —— 主要文档网站

基于 MkDocs Material 主题构建的中文技术文档网站，网址：http://expfluid.bond/ （仅 tju 校园网可访问）

#### 内容结构

- **零基础入门** (`part1_basics`)
  - 环境准备：JupyterHub 空间使用指南、本地环境搭建
  - Python 基础：Python 入门、NumPy 入门、面向对象编程简介
  - PyTorch 基础：PyTorch 简介、自动微分理解

- **神经网络基石** (`part2_networks`)
  - 基本数学原理
  - MLP 和 PyTorch 入门
  - RNN、CNN 介绍
  - PINN（物理信息神经网络）介绍

- **优良机制/架构解析** (`part3_advance`)
  - ResNet 机制解析
  - FiLM 机制解析
  - RevIN 机制解析
  - 门控线性单元 (GLU) 激活函数进化
  - U-Net 模型

- **文献阅读与分享** (`part4_paper_reading`)
  - 基于 PINN 的粒子图像物理场重建
  - 分布式 PINN 对稀疏数据的物理场重构
  - 基于 LSTM 的 POD 系数重构系统
  - 自适应模型预测温度控制
  - 边缘数据中心智能运维
  - 基于分数的生成模型
  - 生成式预训练模型

- **课题组特色研究** (`part5_our_research`)
  - 强化学习与流体控制
  - 基于图的生成式RNN设计

#### 技术特性

- 响应式设计，支持日间/暗黑模式切换
- 内置数学公式渲染（KaTeX）
- 支持Mermaid图表可视化
- 中文搜索优化
- 完整的导航结构

### 💻 notebooks_template —— Jupyter Notebook 模板

为文档网站配套的交互式学习模板：

- `part1_basics`: Python和NumPy基础模板
- `part2_networks`: 神经网络架构模板
- `part3_advance`: 高级模型模板

### 👥 notebooks —— 成员个人工作区

各课题组成员的个人工作目录，包含：
- chn_fluid
- gzh
- hzx
- qrc
- wqx
- test_user

### 📁 shared_files —— 共享资源库

课题组共享的工具和资源：
- JupyterHub 源代码
- 工具软件（Typora等）
- 论文推荐内容模板

## 快速开始

### 访问文档网站

直接访问：http://expfluid.bond/

### 本地运行文档网站

```bash
cd /mnt/data1/resources/mkdocs/ai

# 安装依赖
pip install mkdocs-material
pip install pymdown-extensions
pip install mkdocs-mermaid2-plugin

# 启动本地服务器
mkdocs serve

# 构建静态网站
mkdocs build
```

### 更新文档内容

1. 编辑 `mkdocs/ai/docs/` 下的 Markdown 文件
2. 运行 `mkdocs build` 重新构建
3. 或运行 `mkdocs serve` 实时预览

## 维护说明

### 文档更新流程

1. **内容创作**：在对应 `partX_xxx` 目录下创建/编辑 Markdown 文件
2. **导航配置**：在 `mkdocs.yml` 中更新导航结构
3. **本地测试**：运行 `mkdocs serve` 检查效果
4. **构建部署**：运行 `mkdocs build` 生成静态文件

### 文档规范

- 使用中文表述
- 数学公式使用 KaTeX 语法
- 图表使用 Mermaid 语法
- 遵循项目目录结构组织内容

## 项目特点

- **系统性**：从基础到前沿的完整学习路径
- **实用性**：理论与实践相结合，提供可运行的代码模板
- **专业性**：聚焦流体力学与AI交叉领域
- **协作性**：支持多成员协同贡献和维护

## 联系方式

- 项目维护者：XYChouMian
- 技术支持：通过 JupyterHub 空间或内部联系渠道

---

*本项目主要服务于课题组内部技术交流与学习，持续更新中...*