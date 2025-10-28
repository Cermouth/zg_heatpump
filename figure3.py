import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '-'

# Load the LCOHP scenario comparison data
file_path = "./LCOHP/lcohp_scenario_comparison.csv"
df = pd.read_csv(file_path)

# Rename 'base' scenario if exists
df = df.rename(columns=lambda x: x.strip())
if 'Unnamed: 1' in df.columns:
    df = df.rename(columns={'Unnamed: 1': 'Base'})

# Melt data to long format
df_melted = df.melt(id_vars=['location'], var_name='Scenario', value_name='LCOHP (Euro)')

# Create the plot
# Define the color map for scenarios
color_map = {
    'Base': '#8E6713',
    'DemandMet': '#007894',
    'FastDiffusion': '#A7117A',
    'Recycling': '#627313',
    'SelfSuff40': '#215CAF',
    'Tariffs': '#B7352D'
}

# === Create and Save Plot ===
fig, ax = plt.subplots(figsize=(8, 6.5))
markers = ['_', 'o', 'x', 'v', '+', '^', '*']#'D', 'o', 's', '^', 'v', 'p', '*'

# Plot with assigned colors
for i, scenario in enumerate(df_melted['Scenario'].unique()):#['Tariffs','FastDiffusion','Recycling']
    subset = df_melted[df_melted['Scenario'] == scenario]
    color = color_map.get(scenario, 'gray')
    ax.scatter(subset['location'], subset['LCOHP (Euro)'],
               s=150, edgecolor=color, facecolor='none',#color,alpha=0.5,
               linewidth=1, marker='o')
               #marker=markers[i % len(markers)])
    plt.ylim(80,320)

# Title and axes labels
ax.set_xlabel("Region", fontsize=12)
ax.set_ylabel("LCOHP (€/kW)", fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(True, linestyle='--', alpha=0.2)

# === Replace legend with colored text ===
legend_y_start = 0.98
legend_y_spacing = 0.03

for i, scenario_name in enumerate(['Base', 'DemandMet', 'SelfSuff40', 'Tariffs', 'FastDiffusion', 'Recycling']):
    if scenario_name in color_map:
        color = color_map[scenario_name]
        y_pos = legend_y_start - i * legend_y_spacing
        ax.text(0.85, y_pos, scenario_name,
                transform=ax.transAxes,
                fontsize=12,
                color=color,
                verticalalignment='top')

plt.tight_layout()

# === Save and show ===
output_dir = Path('visualization')
output_dir.mkdir(exist_ok=True)

fig.savefig(output_dir / 'Fig3_lcohp_byregion.png', dpi=330, bbox_inches='tight')
fig.savefig(output_dir / 'Fig3_lcohp_byregion.pdf', bbox_inches='tight')
plt.show()

print(f"\nPlots saved to {output_dir}/")


# df = df.set_index('location')
#
# # Reorder scenarios as desired
# scenario_order = ['Base', 'DemandMet', 'SelfSuff40', 'Tariffs', 'FastDiffusion', 'Recycling']
# df = df[scenario_order]
#
# # Transpose so scenarios are on y-axis and regions on x-axis
# df_transposed = df.T
#
# # === Create Heatmap ===
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # Create heatmap
# im = ax.imshow(df_transposed.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=350)
#
# # Set ticks and labels
# ax.set_xticks(np.arange(len(df_transposed.columns)))
# ax.set_yticks(np.arange(len(df_transposed.index)))
# ax.set_xticklabels(df_transposed.columns)
# ax.set_yticklabels(df_transposed.index)
#
# # Rotate x-axis labels if needed
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
# # Add text annotations with values
# for i in range(len(df_transposed.index)):
#     for j in range(len(df_transposed.columns)):
#         value = df_transposed.values[i, j]
#         # Choose text color based on background intensity
#         text_color = 'white' if value > 175 else 'black'
#         text = ax.text(j, i, f'{value:.0f}',
#                       ha="center", va="center", color=text_color, fontsize=10)
#
# # Labels
# ax.set_xlabel("Region", fontsize=12)
# ax.set_ylabel("Scenario", fontsize=12)
#
# # Add colorbar
# cbar = plt.colorbar(im, ax=ax, pad=0.02)
# cbar.set_label('LCOHP (Euro)', rotation=270, labelpad=20, fontsize=12)
#
# plt.tight_layout()
#
# # === Save and show ===
# output_dir = Path('visualization')
# output_dir.mkdir(exist_ok=True)
#
# fig.savefig(output_dir / 'Fig3_lcohp_heatmap.png', dpi=330, bbox_inches='tight')
# fig.savefig(output_dir / 'Fig3_lcohp_heatmap.pdf', bbox_inches='tight')
# plt.show()
#
# print(f"\nHeatmap saved to {output_dir}/")