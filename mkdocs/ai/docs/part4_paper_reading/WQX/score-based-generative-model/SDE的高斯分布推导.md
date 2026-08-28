# 为什么方差保持 SDE 的分布是高斯分布？

对于方差保持 SDE：

$$d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}dt + \sqrt{\beta(t)}d\mathbf{w}$$

从 $\mathbf{x}(0)$ 到 $\mathbf{x}(t)$ 的分布是**高斯分布**：

$$p_{0t}(\mathbf{x}(t)|\mathbf{x}(0)) = \mathcal{N}(\mathbf{x}(t); \alpha(t)\mathbf{x}(0), \sigma^2(t)\mathbf{I})$$

#### **推导过程**

**第一步：线性 SDE 的解**

这是一个**线性 SDE**（drift 项正比于 $\mathbf{x}$，diffusion 项是常数），可以写成积分形式：

$$\mathbf{x}(t) = \mathbf{x}(0) \cdot \exp\left(-\frac{1}{2}\int_0^t \beta(s)ds\right) + \int_0^t \sqrt{\beta(s)} \exp\left(-\frac{1}{2}\int_s^t \beta(r)dr\right) d\mathbf{w}(s)$$

定义：

$$\alpha(t) = \exp\left(-\frac{1}{2}\int_0^t \beta(s)ds\right)$$

则：

$$\mathbf{x}(t) = \alpha(t)\mathbf{x}(0) + \int_0^t \frac{\sqrt{\beta(s)}}{\alpha(s)} \alpha(t) \, d\mathbf{w}(s)$$

**第二步：为什么是高斯分布？**

**关键性质**：随机积分 $\int_0^t f(s) d\mathbf{w}(s)$ 是**高斯随机变量**，因为：

- 布朗运动 $\mathbf{w}(t)$ 的每个微小增量 $d\mathbf{w}$ 都是高斯的
- 高斯随机变量的线性组合仍是高斯

因此 $\mathbf{x}(t)$ 是**两项的和**：

1. 确定性项：$\alpha(t)\mathbf{x}(0)$（均值）
2. 随机项：$\int_0^t \frac{\sqrt{\beta(s)}}{\alpha(s)} \alpha(t) \, d\mathbf{w}(s)$（零均值高斯噪声）

**均值**：

$$\mathbb{E}[\mathbf{x}(t)|\mathbf{x}(0)] = \alpha(t)\mathbf{x}(0)$$

**方差**（使用 Itô 等距）：

$$\text{Var}[\mathbf{x}(t)|\mathbf{x}(0)] = \mathbb{E}\left[\left(\int_0^t \frac{\sqrt{\beta(s)}}{\alpha(s)} \alpha(t) \, d\mathbf{w}(s)\right)^2\right]$$

$$= \alpha^2(t) \int_0^t \frac{\beta(s)}{\alpha^2(s)} ds$$

定义 $\sigma^2(t) = \alpha^2(t) \int_0^t \frac{\beta(s)}{\alpha^2(s)} ds$，得到：

$$p_{0t}(\mathbf{x}(t)|\mathbf{x}(0)) = \mathcal{N}(\alpha(t)\mathbf{x}(0), \sigma^2(t)\mathbf{I})$$

#### **具体系数推导（方差保持的特殊性质）**

对于 VP-SDE，要求**方差保持不变**：

$$\mathbb{E}[\|\mathbf{x}(t)\|^2] = \mathbb{E}[\|\mathbf{x}(0)\|^2]$$

即：

$$\alpha^2(t) \mathbb{E}[\|\mathbf{x}(0)\|^2] + \sigma^2(t) \cdot d = \mathbb{E}[\|\mathbf{x}(0)\|^2]$$

其中 $d$ 是数据维度。假设 $\mathbb{E}[\|\mathbf{x}(0)\|^2] = d$（标准化数据），则：

$$\alpha^2(t) + \sigma^2(t) = 1$$

代入 $\sigma^2(t) = \alpha^2(t) \int_0^t \frac{\beta(s)}{\alpha^2(s)} ds$，解出：

$$\alpha^2(t) = \exp\left(-\int_0^t \beta(s)ds\right)$$

$$\sigma^2(t) = 1 - \exp\left(-\int_0^t \beta(s)ds\right)$$

#### **直觉理解**

**为什么是高斯？**

- SDE 的 drift 是**线性**的（$-\frac{1}{2}\beta(t)\mathbf{x}$），不会改变分布的形状
- Diffusion 项是**高斯噪声**（$d\mathbf{w}$）
- 高斯分布在线性变换和加高斯噪声下**保持高斯性**

**类比**：

- 初始状态：$\mathbf{x}(0)$
- 每一小步：旧值乘以衰减因子（drift）+ 加一点高斯噪声（diffusion）
- 最终结果：$\mathbf{x}(t) = \text{衰减的旧值} + \text{累积的高斯噪声}$
- 两个高斯相加还是高斯 → 整体是高斯分布

#### **总结**

| 步骤             | 公式/结论                                                                     |
| ---------------- | ----------------------------------------------------------------------------- |
| **线性 SDE**     | $d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}dt + \sqrt{\beta(t)}d\mathbf{w}$ |
| **解的形式**     | $\mathbf{x}(t) = \alpha(t)\mathbf{x}(0) + \text{高斯噪声}$                    |
| **为什么是高斯** | 线性变换 + 高斯噪声 = 高斯分布                                                |
| **均值**         | $\alpha(t)\mathbf{x}(0)$                                                      |
| **方差**         | $\sigma^2(t)\mathbf{I}$，其中 $\alpha^2(t) + \sigma^2(t) = 1$（方差保持）     |

这个高斯性质是**训练 score 网络的关键**——因为高斯分布的 score 有解析解 $-\frac{\mathbf{x}(t) - \alpha(t)\mathbf{x}(0)}{\sigma^2(t)}$，可以直接用来监督神经网络！
