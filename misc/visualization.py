import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Load the data
df = pd.read_csv('data/results_tokenized_65536_rwkv7_1b5_128k_complete.csv')

# Filter out Document Depth = 100
df_filtered = df[df['Document Depth'] != 100]

# Create a pivot table for the heatmap
pivot_table = pd.pivot_table(
    df_filtered, values='Score', index=['Document Depth', 'Context Length'], 
    aggfunc='mean'
).reset_index()
pivot_table = pivot_table.pivot(
    index="Document Depth", columns="Context Length", values="Score"
)

plt.figure(figsize=(3, 3), dpi=150)

cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

# Create the heatmap
ax = sns.heatmap(
    pivot_table,
    cmap=cmap,
    cbar=False,
    linewidths=0,
    linecolor='none',
)

# Format x-axis (context length)
context_lengths = sorted(df['Context Length'].unique())

desired_ticks = [0, len(context_lengths)//3, 2*len(context_lengths)//3, len(context_lengths)-1]
x_tick_positions = [pos for pos in desired_ticks]
x_labels = [f"{int(context_lengths[pos]/1000)}k" for pos in desired_ticks]

plt.xticks(
    [p for p in x_tick_positions], 
    x_labels,
    fontsize=9, 
    rotation=0
)

# Format y-axis (document depth)
depths_filtered = sorted(df_filtered['Document Depth'].unique())

# Set positions and labels for y-ticks
positions = [0, len(depths_filtered)/2, len(depths_filtered)]
labels = ["0", "50", "100"]

plt.yticks(
    positions,
    labels,
    fontsize=9,
    rotation=0
)

# Add labels with consistent font size
plt.xlabel('Context Len.', fontsize=9)
plt.ylabel('Ans. Depth (%)', fontsize=9)

# Remove the frame
for spine in ax.spines.values():
    spine.set_visible(False)

# Tight layout to eliminate unnecessary whitespace
plt.tight_layout(pad=0.5)

# Save the figure with high DPI for clarity when small
plt.savefig('data/heatmap_square_65536_rwkv7_1b5_128k.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()