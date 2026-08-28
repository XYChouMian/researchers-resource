import numpy as np
import matplotlib.pyplot as plt
import os

# 设置全局样式
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

save_dir = r'mkdocs/ai/docs/part2_networks/2.1_neuron'
if not os.path.exists(save_dir):
    raise FileNotFoundError(f"目录 {save_dir} 不存在")


x = np.linspace(-5, 5, 200)
figsize = (5, 3)

# 1. Sigmoid
plt.figure(figsize=figsize)
y = 1 / (1 + np.exp(-x))
plt.plot(x, y, 'r-')
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-0.1, 1.1)
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'sigmoid.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# 2. Tanh
plt.figure(figsize=figsize)
y = np.tanh(x)
plt.plot(x, y, 'r-')
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-1.1, 1.1)
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'tanh.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# 3. ReLU
plt.figure(figsize=figsize)
y = np.maximum(0, x)
plt.plot(x, y, 'r-')
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-0.5, 5.5)
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'relu.png'),
            dpi=150, bbox_inches='tight')
plt.close()

print("三个图像已保存：sigmoid.png, tanh.png, relu.png")
