# Fokker-Planck推导-从SDE到ODE

---

## **为什么 ODE 和 SDE 边缘分布相同？**

这是概率流 ODE 的核心理论：虽然 SDE 的单条轨迹是随机的，ODE 的轨迹是确定的，但它们的**边缘分布**（在每个时刻 $t$ 的概率密度 $p_t(\mathbf{x})$）完全相同。这个深刻的结果来自于 Fokker-Planck 方程和连续性方程的等价性。

### **第一步：SDE 的边缘分布演化——Fokker-Planck 方程**

对于随机微分方程（SDE）：

$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)dt + g(t)d\mathbf{w}$$

其边缘概率密度 $p_t(\mathbf{x})$ 的演化由 **Fokker-Planck 方程**描述：

$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (\mathbf{f}p_t) + \frac{1}{2}g^2\nabla^2 p_t$$

**符号解释**：

- $\frac{\partial p_t}{\partial t}$：概率密度随时间的变化率
- $-\nabla \cdot (\mathbf{f}p_t)$：漂移项（确定性流动的贡献）
- $\frac{1}{2}g^2\nabla^2 p_t$：扩散项（随机布朗运动的贡献）
- $\nabla \cdot$ 是散度算子，$\nabla^2$ 是拉普拉斯算子

**物理直觉**：

- **漂移项**：概率质量沿向量场 $\mathbf{f}$ 流动
- **扩散项**：概率质量因随机噪声而扩散

### **第二步：ODE 的边缘分布演化——连续性方程**

对于常微分方程（ODE）：

$$\frac{d\mathbf{x}}{dt} = \mathbf{v}(\mathbf{x}, t)$$

其边缘概率密度 $p_t(\mathbf{x})$ 的演化由 **连续性方程**描述：

$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (\mathbf{v}p_t)$$

**物理直觉**：

连续性方程描述概率质量沿确定性向量场 $\mathbf{v}$ 流动，只有漂移项，没有扩散项（因为 ODE 没有随机性）。

> 更详细的推导参考：[第二步：ODE 的边缘分布演化——连续性方程](连续性方程.md)

### **第三步：将扩散项改写为散度形式**

关键是证明扩散项可以改写成某个散度。利用恒等式：

$$\nabla p_t = p_t \nabla \log p_t$$

对扩散项应用散度算子：

$$\frac{1}{2}g^2\nabla^2 p_t = \frac{1}{2}g^2 \nabla \cdot (\nabla p_t) = \frac{1}{2}g^2 \nabla \cdot (p_t \nabla \log p_t)$$

应用乘积法则 $\nabla \cdot (f\mathbf{v}) = f(\nabla \cdot \mathbf{v}) + \mathbf{v} \cdot \nabla f$：

$$= \frac{1}{2}g^2 [p_t \nabla^2 \log p_t + \nabla \log p_t \cdot \nabla p_t]$$

注意到 $\nabla p_t = p_t \nabla \log p_t$，所以：

$$\nabla \log p_t \cdot \nabla p_t = p_t \|\nabla \log p_t\|^2$$

但我们需要的是更直接的形式。关键观察是：

$$\frac{1}{2}g^2\nabla^2 p_t = -\nabla \cdot \left(-\frac{1}{2}g^2 \nabla p_t\right) = -\nabla \cdot \left(-\frac{1}{2}g^2 p_t \nabla \log p_t\right)$$

### **第四步：合并 Fokker-Planck 方程的两项**

将 Fokker-Planck 方程改写：

$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (\mathbf{f}p_t) - \nabla \cdot \left(-\frac{1}{2}g^2 p_t \nabla \log p_t\right)$$

由于散度算子是线性的，合并两项：

$$= -\nabla \cdot \left(\mathbf{f}p_t - \frac{1}{2}g^2 p_t \nabla \log p_t\right)$$

提取公因子 $p_t$：

$$= -\nabla \cdot \left[\left(\mathbf{f} - \frac{1}{2}g^2 \nabla \log p_t\right) p_t\right]$$

### **第五步：识别等价的速度场**

定义 ODE 的速度场：

$$\mathbf{v}(\mathbf{x}, t) = \mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$

则 SDE 的 Fokker-Planck 方程变为：

$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (\mathbf{v}p_t)$$

**这正是 ODE 的连续性方程！**

---

## **最终结论**

当 ODE 的速度场选择为：

$$\boxed{\mathbf{v} = \mathbf{f} - \frac{1}{2}g^2\nabla\log p_t}$$

时，ODE 和 SDE 满足相同的概率密度演化方程，因此它们的边缘分布 $p_t(\mathbf{x})$ 在任何时刻都相同。

---

## **物理意义**

**SDE 的速度分解**：

$$\mathbf{v} = \underbrace{\mathbf{f}}_{\text{原始漂移}} - \underbrace{\frac{1}{2}g^2 \nabla\log p_t}_{\text{扩散的净效应}}$$

**关键洞察**：

- SDE 中的随机扩散 $g(t)d\mathbf{w}$ 让粒子随机散开
- 从统计角度看，扩散的净效应是沿着 score 函数（$\nabla \log p_t$）的方向
- ODE 通过修正速度场 $-\frac{1}{2}g^2\nabla\log p_t$，用确定性流动模拟了随机扩散的平均效应
- 因此虽然单条轨迹不同（SDE 是随机的，ODE 是确定的），但整体分布完全相同

**具体例子**：

想象一群粒子从一点出发：

- **SDE**：每个粒子既被向量场 $\mathbf{f}$ 推动，又受随机扰动 $g d\mathbf{w}$
- **ODE**：每个粒子沿确定路径移动，路径由 $\mathbf{v} = \mathbf{f} - \frac{1}{2}g^2\nabla\log p_t$ 决定

> 虽然单个粒子的轨迹不同，但在任何时刻 $t$，粒子群的**空间分布**完全相同。这意味着我们可以用更快、更稳定的确定性 ODE 来替代随机 SDE 进行采样！

---

## **总结**

1. **SDE** 的边缘分布由 **Fokker-Planck 方程**描述（含漂移项和扩散项）
2. **ODE** 的边缘分布由 **连续性方程**描述（只有漂移项）
3. 通过数学变换，可以证明当 $\mathbf{v} = \mathbf{f} - \frac{1}{2}g^2\nabla\log p_t$ 时，两个方程等价
4. 因此 ODE 和 SDE 的边缘分布 $p_t(\mathbf{x})$ 在所有时刻都相同
5. 这意味着我们可以用**确定性的 ODE**（更快、更稳定）来替代**随机的 SDE**进行采样，而不改变生成的分布

这就是为什么概率流 ODE 可以作为扩散模型的确定性采样方法！

> 其他可参考链接：  
> [福克-普朗克方程_百度百科](https://baike.baidu.com/item/福克-普朗克方程/16550164)  
> [Fokker-Planck方程：从物理到机器学习-CSDN博客](https://blog.csdn.net/shizheng_Li/article/details/146988326)  
> [什么是Fokker-Planck方程？ - 知乎](https://zhuanlan.zhihu.com/p/535688931)  
