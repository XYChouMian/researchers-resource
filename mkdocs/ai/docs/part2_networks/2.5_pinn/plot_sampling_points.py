from pathlib import Path
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Get the current directory of the script
current_dir = Path(__file__).resolve().parent

# Create figure with high quality
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# Set background color
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# 1. PDE Collocation Points (interior domain)
n_pde = 3000
x_pde = np.random.rand(n_pde)
t_pde = np.random.rand(n_pde)
ax.scatter(x_pde, t_pde, c='#2c3e50', s=2, alpha=0.4,
           label='PDE Collocation Points', zorder=1)

# 2. Boundary Condition Points (left and right boundaries)
n_bc = 80
t_bc = np.linspace(0, 1, n_bc)

# Left boundary (x=0)
ax.scatter(np.zeros(n_bc), t_bc, c='#e74c3c', s=30,
           marker='s', edgecolors='darkred', linewidths=0.5,
           label='Boundary Conditions (x=0, x=1)', zorder=3)

# Right boundary (x=1)
ax.scatter(np.ones(n_bc), t_bc, c='#e74c3c', s=30,
           marker='s', edgecolors='darkred', linewidths=0.5, zorder=3)

# 3. Initial Condition Points (bottom boundary, t=0)
n_ic = 80
x_ic = np.linspace(0, 1, n_ic)
ax.scatter(x_ic, np.zeros(n_ic), c='#27ae60', s=30,
           marker='^', edgecolors='darkgreen', linewidths=0.5,
           label='Initial Conditions (t=0)', zorder=3)

# 4. Observation Data Points (sparse measurements)
n_data = 25
x_data = np.random.rand(n_data)
t_data = np.random.rand(n_data)
ax.scatter(x_data, t_data, c='#3498db', s=120,
           marker='*', edgecolors='#2980b9', linewidths=1.5,
           label='Observation Data (sparse)', zorder=4)

# Add domain boundary rectangle
rect = Rectangle((0, 0), 1, 1, linewidth=2.5,
                 edgecolor='#34495e', facecolor='none',
                 linestyle='-', zorder=2)
ax.add_patch(rect)

# Add annotations for domain regions
ax.text(0.5, 0.5, 'Domain $\Omega$\n$x \in [0,1], t \in [0,1]$',
        ha='center', va='center', fontsize=14,
        color='#333333', alpha=1.0, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='none', alpha=1.0))

# Add arrows and labels for boundaries
ax.annotate('', xy=(0, 0.5), xytext=(-0.15, 0.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='#e74c3c'))
ax.text(-0.18, 0.5, 'BC', ha='right', va='center',
        fontsize=11, color='#e74c3c', weight='bold')

ax.annotate('', xy=(1, 0.5), xytext=(1.15, 0.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='#e74c3c'))
ax.text(1.18, 0.5, 'BC', ha='left', va='center',
        fontsize=11, color='#e74c3c', weight='bold')

ax.annotate('', xy=(0.5, 0), xytext=(0.5, -0.15),
            arrowprops=dict(arrowstyle='->', lw=2, color='#27ae60'))
ax.text(0.5, -0.18, 'IC', ha='center', va='top',
        fontsize=11, color='#27ae60', weight='bold')

# Styling
ax.set_xlabel('Spatial Coordinate $x$', fontsize=14, weight='bold')
ax.set_ylabel('Time Coordinate $t$', fontsize=14, weight='bold')
ax.set_title('PINN Sampling Strategy in Spatio-Temporal Domain',
             fontsize=16, weight='bold', pad=20)

# Set axis limits with padding
ax.set_xlim(-0.25, 1.25)
ax.set_ylim(-0.25, 1.25)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')

# Legend
legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95,
                   edgecolor='#34495e', fancybox=True, shadow=True)
legend.get_frame().set_facecolor('white')

# Add tick labels
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(labelsize=11)

# Add statistics text box
stats_text = f'Sampling Statistics:\n' \
             f'• PDE points: {n_pde}\n' \
             f'• BC points: {2*n_bc}\n' \
             f'• IC points: {n_ic}\n' \
             f'• Data points: {n_data}'

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(current_dir / 'pinn_sampling_points.png',
            dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Figure saved as 'pinn_sampling_points.png'")
plt.show()
