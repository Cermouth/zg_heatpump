import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
               s=150, edgecolor=color, facecolor=color,#alpha=0.5,
               linewidth=1, marker='x')
               #marker=markers[i % len(markers)])
    plt.ylim(100,400)

# Title and axes labels
ax.set_xlabel("Region", fontsize=12)
ax.set_ylabel("LCOHP (Euro)", fontsize=12)
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
