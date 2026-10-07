import matplotlib.pyplot as plt
import numpy as np
import os

models = ['TripoSR', 'LGM', 'TRELLIS']
latency = [0.5, 5.0, 12.0]
vram = [6.0, 10.0, 24.0]

x = np.arange(len(models))
width = 0.35

# Make the figure much larger for readability
fig, ax1 = plt.subplots(figsize=(10, 6))

# High-contrast, printer-friendly colors
color1 = '#2c7bb6' 
color2 = '#d7191c' 

ax1.set_xlabel('Feed-Forward 3D Models', fontsize=16, fontweight='bold')
ax1.set_ylabel('Inference Latency (seconds)', color='black', fontsize=16, fontweight='bold')
# Add bold black edges and hatching (stripes) so it is readable even in black & white printing
bars1 = ax1.bar(x - width/2, latency, width, color=color1, edgecolor='black', linewidth=1.5, hatch='//', label='Latency (s)')
ax1.tick_params(axis='y', labelcolor='black', labelsize=14)
ax1.tick_params(axis='x', labelsize=16)

ax2 = ax1.twinx()  
ax2.set_ylabel('Peak VRAM Requirement (GB)', color='black', fontsize=16, fontweight='bold')  
bars2 = ax2.bar(x + width/2, vram, width, color=color2, edgecolor='black', linewidth=1.5, hatch='\\\\', label='VRAM (GB)')
ax2.tick_params(axis='y', labelcolor='black', labelsize=14)

# Make the text labels massive and black so they don't fade when printed
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f'{yval}s', ha='center', va='bottom', color='black', fontsize=14, fontweight='bold')

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval}GB', ha='center', va='bottom', color='black', fontsize=14, fontweight='bold')

plt.title('Computational Efficiency: Latency vs. Hardware Constraints', fontsize=18, fontweight='bold', pad=20)
plt.xticks(x, models, fontsize=16, fontweight='bold')

# Combine legends and make the box opaque
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=14, framealpha=1.0, edgecolor='black')

# Add a subtle grid to make reading values easier
ax1.grid(axis='y', linestyle='--', alpha=0.7)

fig.tight_layout()  

output_path = os.path.join(os.path.dirname(__file__), 'efficiency_chart.pdf')
# Save at ultra-high DPI
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"Graph successfully generated and saved to: {output_path}")
