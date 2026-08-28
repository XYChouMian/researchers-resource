# 从条件 score 到边缘 score

### **核心问题：为什么要区分条件 score 和边缘 score？**

#### **两种 score 的定义**

**条件 score**：

$$\nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))$$

- 给定**特定的**初始图像 $\mathbf{x}(0)$（比如一只特定的猫）
- 问：加噪图像 $\mathbf{x}(t)$ 应该往哪个方向移动，才能回到**这只猫**？
- 这是一个"有答案"的问题：我们知道原图是什么

**边缘 score**：

$$\nabla_{\mathbf{x}(t)}\log p_t(\mathbf{x}(t))$$

- **不知道**原图 $\mathbf{x}(0)$ 是什么
- 只看到加噪图像 $\mathbf{x}(t)$（一团模糊）
- 问：这团模糊应该往哪个方向移动，才能变成**某张合理的图像**？
- 这是我们在生成时实际需要的：从纯噪声开始，不知道目标是什么

---

### **为什么训练时用条件 score，生成时需要边缘 score？**

#### **训练阶段**

我们有数据集 $\{\mathbf{x}(0)^{(i)}\}_{i=1}^N$（比如 10000 张猫的图片）：

1. 随机选一张图 $\mathbf{x}(0)$
2. 随机加噪得到 $\mathbf{x}(t)$
3. 训练网络预测 $\nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))$（条件 score）

这很简单，因为我们**知道原图是什么**，可以直接计算正确答案 $-\boldsymbol{\epsilon}/\sigma(t)$。

#### **生成阶段**

从纯噪声 $\mathbf{x}(T) \sim \mathcal{N}(0, \mathbf{I})$ 开始：

1. 我们**不知道**目标图像是什么
2. 需要预测 $\nabla_{\mathbf{x}(t)}\log p_t(\mathbf{x}(t))$（边缘 score）
3. 但网络训练的是条件 score！

**问题**：训练和推理的目标不一致，为什么能工作？

**答案**：[**Vincent 定理**](https://ieeexplore.ieee.org/abstract/document/6795935)证明了，当我们对所有可能的 $\mathbf{x}(0)$ 取平均时，条件 score 自动变成边缘 score！

---

### **详细推导**

#### **第一步：边缘分布的定义**

边缘分布 $p_t(\mathbf{x}(t))$ 是"所有可能的 $\mathbf{x}(0)$ 通过扩散过程到达 $\mathbf{x}(t)$ 的概率之和"：

$$p_t(\mathbf{x}(t)) = \int p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))p_{data}(\mathbf{x}(0))d\mathbf{x}(0)$$

**符号解释**：

- $p_{data}(\mathbf{x}(0))$：数据分布，表示原图 $\mathbf{x}(0)$ 出现的概率（训练集中猫的分布）
- $p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))$：转移核，给定原图，加噪图的分布
- 积分：对所有可能的原图求和（连续版本）

**物理直觉**：

> 看到一团模糊 $\mathbf{x}(t)$，它可能来自猫1（概率 $p_1$）、猫2（概率 $p_2$）、……，总概率是所有可能性的和。

**具体例子**：

假设数据集只有两张图：猫（概率 0.6）和狗（概率 0.4）。

在某个时刻 $t$，看到模糊图像 $\mathbf{x}(t) = [50, 100]$：

- 它可能来自猫加噪：$p_{0t}([50,100]|猫) \times 0.6$
- 也可能来自狗加噪：$p_{0t}([50,100]|狗) \times 0.4$
- 总概率：$p_t([50,100]) = p_{0t}([50,100]|猫) \times 0.6 + p_{0t}([50,100]|狗) \times 0.4$

#### **第二步：对边缘分布取对数并求梯度**

我们的目标是计算边缘 score：

$$\nabla_{\mathbf{x}(t)}\log p_t(\mathbf{x}(t)) = \nabla_{\mathbf{x}(t)}\log \int p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))p_{data}(\mathbf{x}(0))d\mathbf{x}(0)$$

**问题**：对数内部有积分，无法直接求导（$\log \int \neq \int \log$）

**解决思路**：先对对数求导，再处理积分

#### **第三步：应用链式法则**

利用 $\nabla \log f = \frac{1}{f} \nabla f$：

$$\nabla_{\mathbf{x}(t)}\log p_t(\mathbf{x}(t)) = \frac{1}{p_t(\mathbf{x}(t))} \nabla_{\mathbf{x}(t)} p_t(\mathbf{x}(t))$$

代入 $p_t(\mathbf{x}(t))$ 的积分形式：

$$= \frac{1}{p_t(\mathbf{x}(t))} \nabla_{\mathbf{x}(t)} \int p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))p_{data}(\mathbf{x}(0))d\mathbf{x}(0)$$

#### **第四步：梯度与积分交换顺序**

因为积分变量是 $\mathbf{x}(0)$，求导变量是 $\mathbf{x}(t)$，两者独立，所以梯度可以移到积分内部：

$$= \frac{1}{p_t(\mathbf{x}(t))} \int \nabla_{\mathbf{x}(t)} p_{0t}(\mathbf{x}(t)|\mathbf{x}(0)) \cdot p_{data}(\mathbf{x}(0))d\mathbf{x}(0)$$

注意：$p_{data}(\mathbf{x}(0))$ 不含 $\mathbf{x}(t)$，所以对它求导为0，它保持不变。

#### **第五步：对条件概率应用链式法则**

对 $p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))$ 应用链式法则的反向形式 $\nabla f = f \cdot \nabla \log f$：

$$\nabla_{\mathbf{x}(t)} p_{0t}(\mathbf{x}(t)|\mathbf{x}(0)) = p_{0t}(\mathbf{x}(t)|\mathbf{x}(0)) \cdot \nabla_{\mathbf{x}(t)} \log p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))$$

代入上式：

$$= \frac{1}{p_t(\mathbf{x}(t))} \int p_{0t}(\mathbf{x}(t)|\mathbf{x}(0)) \cdot \nabla_{\mathbf{x}(t)} \log p_{0t}(\mathbf{x}(t)|\mathbf{x}(0)) \cdot p_{data}(\mathbf{x}(0))d\mathbf{x}(0)$$

#### **第六步：重新排列项的顺序**

把各项重新排列（不改变数学含义，只是调整书写顺序）：

$$= \int \frac{p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))p_{data}(\mathbf{x}(0))}{p_t(\mathbf{x}(t))} \cdot \nabla_{\mathbf{x}(t)} \log p_{0t}(\mathbf{x}(t)|\mathbf{x}(0)) d\mathbf{x}(0)$$

**关键观察**：分数部分 $\frac{p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))p_{data}(\mathbf{x}(0))}{p_t(\mathbf{x}(t))}$ 有特殊的含义！

#### **第七步：识别贝叶斯公式**

根据贝叶斯定理：

$$p(\mathbf{x}(0)|\mathbf{x}(t)) = \frac{p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))p_{data}(\mathbf{x}(0))}{p_t(\mathbf{x}(t))}$$

**解释**：

- 分子：$p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))$ 是似然（前向概率），$p_{data}(\mathbf{x}(0))$ 是先验
- 分母：$p_t(\mathbf{x}(t))$ 是边缘概率（归一化常数）
- 结果：$p(\mathbf{x}(0)|\mathbf{x}(t))$ 是后验概率（从模糊推断清晰）

因此我们的积分变为：

$$\nabla_{\mathbf{x}(t)}\log p_t(\mathbf{x}(t)) = \int p(\mathbf{x}(0)|\mathbf{x}(t)) \cdot \nabla_{\mathbf{x}(t)} \log p_{0t}(\mathbf{x}(t)|\mathbf{x}(0)) d\mathbf{x}(0)$$

#### **第八步：改写为期望形式**

根据期望的定义：

$$\mathbb{E}_{\mathbf{x}(0)|\mathbf{x}(t)}[f(\mathbf{x}(0))] = \int f(\mathbf{x}(0)) \cdot p(\mathbf{x}(0)|\mathbf{x}(t)) d\mathbf{x}(0)$$

这里 $f(\mathbf{x}(0)) = \nabla_{\mathbf{x}(t)} \log p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))$，因此：

$$= \mathbb{E}_{\mathbf{x}(0)|\mathbf{x}(t)}\left[\nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))\right]$$

---

### **最终结论**

$$\boxed{\nabla_{\mathbf{x}(t)}\log p_t(\mathbf{x}(t)) = \mathbb{E}_{\mathbf{x}(0)|\mathbf{x}(t)}\left[\nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t)|\mathbf{x}(0))\right]}$$

**用大白话说**：

- **左边（边缘 score）**：看到模糊图 $\mathbf{x}(t)$，不知道原图是什么，应该往哪个方向移动使概率密度增大？

- **右边（条件 score 的期望）**：
  1. 考虑所有可能的原图 $\mathbf{x}(0)$（猫、狗、鸟……）
  2. 每个原图给出一个"去噪方向"（条件 score）
  3. 按后验概率 $p(\mathbf{x}(0)|\mathbf{x}(t))$ 加权平均（越可能是哪个原图，权重越大）

**物理意义**：

这个定理告诉我们，边缘 score（生成时需要的）等于条件 score（训练时计算的）在后验分布下的期望。因此，训练时虽然用的是条件 score，但对整个数据集取平均后，网络自动学到了边缘 score！

---

### **为什么训练时自动得到边缘 score？**

训练时的损失函数：

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}(0) \sim p_{data}} \mathbb{E}_{\mathbf{x}(t)|\mathbf{x}(0)} \left\| s_\theta(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t)|\mathbf{x}(0)) \right\|^2$$

**两层期望**：

1. **外层期望** $\mathbb{E}_{\mathbf{x}(0) \sim p_{data}}$：从数据集随机采样原图（猫1、猫2、狗1、……）
2. **内层期望** $\mathbb{E}_{\mathbf{x}(t)|\mathbf{x}(0)}$：给定原图，随机加噪

当网络优化到最优时：

$$s_\theta(\mathbf{x}(t), t) = \mathbb{E}_{\mathbf{x}(0) \sim p_{data}} \left[\nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t)|\mathbf{x}(0)) \Big| \mathbf{x}(t)\right]$$

这恰好就是 Vincent 定理的右边！

**直觉**：

- 训练时，同一个模糊图 $\mathbf{x}(t)$ 可能来自不同的原图
- 网络被迫学习所有可能性的加权平均
- 这个平均值正是边缘 score

---

### **具体例子**

假设数据集只有两张图：

- 猫：$\mathbf{x}_{\text{cat}}(0) = [200, 100]$，概率 $0.7$
- 狗：$\mathbf{x}_{\text{dog}}(0) = [100, 200]$，概率 $0.3$

在某个时刻 $t$，看到模糊图 $\mathbf{x}(t) = [150, 150]$：

**条件 score（如果知道原图）**：

- 如果原图是猫：score 指向 $[200, 100]$ 方向
- 如果原图是狗：score 指向 $[100, 200]$ 方向

**边缘 score（不知道原图）**：

需要计算后验概率：

- $p(\text{猫}|\mathbf{x}(t))$：看到 $[150,150]$，多大概率来自猫？
- $p(\text{狗}|\mathbf{x}(t))$：看到 $[150,150]$，多大概率来自狗？

假设计算得到：$p(\text{猫}|\mathbf{x}(t)) = 0.6$，$p(\text{狗}|\mathbf{x}(t)) = 0.4$

边缘 score = $0.6 \times \text{(指向猫的 score)} + 0.4 \times \text{(指向狗的 score)}$

这是两个方向的加权平均，结果是一个折中的方向。

**训练过程**：

- 随机采样猫（70%的时间）或狗（30%的时间）
- 加噪到 $[150, 150]$ 附近
- 网络看到很多次 $[150, 150]$，70%来自猫，30%来自狗
- 网络学会预测加权平均的方向

---

### **总结**

1. **条件 score**：知道原图，告诉你怎么去噪（训练时可以计算）
2. **边缘 score**：不知道原图，告诉你怎么生成（推理时需要的）
3. **Vincent 定理**：边缘 score = 条件 score 在后验分布下的期望
4. **训练的魔法**：对所有数据取期望时，自动学到边缘 score

因此，虽然训练时用的是条件 score（有"答案"的监督学习），但通过对整个数据集的平均，网络实际学到的是边缘 score（无"答案"的生成能力）！
