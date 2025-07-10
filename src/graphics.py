import matplotlib.pyplot as plt
import numpy as np

# Data extracted from training log
epochs = [1, 3, 5, 7, 9, 11,

          13, 15, 17, 19, 21, 23, 25, 27, 29,

          31, 33, 35,

          37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65]
val_loss = [58.367, 57.264, 56.885, 56.406, 55.777, 55.307,

            54.672, 53.931, 53.343, 52.698, 52.515, 51.970, 51.497, 51.188, 50.777,

            51.431, 50.959, 50.905,

            50.527, 50.560, 50.332, 50.226, 50.047, 50.199, 49.535, 49.955, 49.656, 49.575, 49.809, 49.675, 49.687,
            48.953, 48.794]

# Set up the plot with ACM-style formatting
plt.figure(figsize=(8, 6))
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
    'legend.shadow': False
})

# Create the plot
plt.plot(epochs, val_loss, 'b-o', linewidth=2, markersize=4, markerfacecolor='blue', markeredgecolor='blue',
         label='Validation Loss')

# plt.plot([70, 71, 72, 73], [50.503, 50.409, 50.269, 50.231], 'b-o', linewidth=2, markersize=4, markerfacecolor='blue', markeredgecolor='blue')

# Customize the plot
plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Validation Loss', fontsize=14, fontweight='bold')
plt.title('Validation Loss vs Epoch', fontsize=16, fontweight='bold', pad=20)

# Add grid
plt.grid(True, alpha=0.3, linestyle='--')

# Add legend
plt.legend(loc='upper right', fontsize=12)

# Tight layout to ensure everything fits
plt.tight_layout()

# Save as high-quality PDF for ACM paper
plt.savefig('val_loss.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Also save as PNG for preview
plt.savefig('val_loss.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Show the plot
plt.show()

# Print some statistics
print("Validation Loss Statistics:")
print(f"Initial value (epoch 1): {val_loss[0]:.3f}")
print(f"Final value (epoch 65): {val_loss[-1]:.3f}")
print(f"Minimum value: {min(val_loss):.3f}")
print(f"Maximum value: {max(val_loss):.3f}")
print(f"Total reduction: {val_loss[0] - val_loss[-1]:.3f}")
print(f"Percentage reduction: {((val_loss[0] - val_loss[-1]) / val_loss[0]) * 100:.1f}%")
