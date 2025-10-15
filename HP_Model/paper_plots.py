import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import os


def plot_regional_comparison(flow_file, capacity_file, demand_file, output_dir="paper"):
    """
    Create one plot for each region (Global, China, Europe, USA) showing production, capacity, and demand.

    Parameters:
    flow_file (str/Path): Path to flow conversion output data file
    capacity_file (str/Path): Path to capacity data file
    demand_file (str/Path): Path to demand data file
    output_dir (str/Path): Directory to save output plots (default: "paper")
    """
    # Set plot style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.titlesize': 15
    })

    # Define color palette for different data types
    colors = {
        'capacity_s1': "#b7afd5",  # Light purple
        'capacity_s2': "#9c91c7",  # Darker purple
        'production_s1': "#b2cfa9",  # Light green
        'production_s2': "#bfcc67",  # Darker green
        'demand': "#bc556d"  # Reddish
    }

    # Ensure files are Path objects
    flow_file = Path(flow_file)
    capacity_file = Path(capacity_file)
    demand_file = Path(demand_file)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Read data files
    flow_df = pd.read_csv(flow_file)
    capacity_df = pd.read_csv(capacity_file)

    # Process demand file (special format where each country is a column)
    demand_df = pd.read_csv(demand_file)

    # Transform demand data from wide to long format
    demand_long = pd.melt(
        demand_df,
        id_vars=['year'],
        var_name='node',
        value_name='demand'
    )

    # Convert kW to GW for demand
    demand_long['demand'] = demand_long['demand'] / 1000000

    # Filter for HP assembly technology in flow data
    flow_df = flow_df[flow_df['technology'] == 'HP_assembly']

    # Filter for HP assembly in capacity data
    capacity_df = capacity_df[capacity_df['technology'] == 'HP_assembly']

    # Define European countries and mapping for regions
    european_countries = ['CZE', 'AUT', 'ITA', 'DEU', 'ROE', 'EUR']

    # Define region mapping function
    def map_to_region(node):
        if node in european_countries:
            return 'EUR'
        elif node in ['CHN', 'USA']:
            return node
        else:
            return 'Global'

    # Apply region mapping to dataframes
    flow_df['region'] = flow_df['node'].apply(map_to_region)
    if 'location' in capacity_df.columns:
        capacity_df['region'] = capacity_df['location'].apply(map_to_region)
    else:
        capacity_df['region'] = capacity_df['node'].apply(map_to_region)

    demand_long['region'] = demand_long['node'].apply(map_to_region)

    # Get scenario columns
    flow_s1_col = [col for col in flow_df.columns if col == 'value_scenario_'][0]
    flow_s2_col = [col for col in flow_df.columns if col == 'value_scenario_S1'][0]

    capacity_s1_col = [col for col in capacity_df.columns if col == 'value_scenario_'][0]
    capacity_s2_col = [col for col in capacity_df.columns if col == 'value_scenario_S1'][0]

    # Convert time indices to actual years
    if 'time_operation' in flow_df.columns:
        flow_df['year'] = flow_df['time_operation'] + 2022

    if 'year' in capacity_df.columns and capacity_df['year'].max() < 2000:
        # If year values are small, they likely need to be offset
        capacity_df['year'] = capacity_df['year'] + 2022

    # Aggregate data by region and year
    flow_by_region = flow_df.groupby(['region', 'year']).agg({
        flow_s1_col: 'sum',
        flow_s2_col: 'sum'
    }).reset_index()

    capacity_by_region = capacity_df.groupby(['region', 'year']).agg({
        capacity_s1_col: 'sum',
        capacity_s2_col: 'sum'
    }).reset_index()

    demand_by_region = demand_long.groupby(['region', 'year']).agg({
        'demand': 'sum'
    }).reset_index()

    # Create plots for each region
    regions = ['Global', 'CHN', 'EUR', 'USA']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, region in enumerate(regions):
        ax = axes[i]

        # Plot capacity lines (scenario 1)
        region_capacity = capacity_by_region[capacity_by_region['region'] == region]
        if not region_capacity.empty:
            ax.plot(region_capacity['year'], region_capacity[capacity_s1_col],
                    marker='o', color=colors['capacity_s1'], linewidth=2,
                    label='Capacity (Base)')

            # Plot capacity lines (scenario 2)
            ax.plot(region_capacity['year'], region_capacity[capacity_s2_col],
                    marker='o', color=colors['capacity_s2'], linewidth=2,
                    linestyle='--', label='Capacity (NZE)')

        # Plot production lines (scenario 1)
        region_flow = flow_by_region[flow_by_region['region'] == region]
        if not region_flow.empty:
            ax.plot(region_flow['year'], region_flow[flow_s1_col],
                    marker='s', color=colors['production_s1'], linewidth=2,
                    label='Production (Base)')

            # Plot production lines (scenario 2)
            ax.plot(region_flow['year'], region_flow[flow_s2_col],
                    marker='s', color=colors['production_s2'], linewidth=2,
                    linestyle='--', label='Production (NZE)')

        # Plot demand line
        region_demand = demand_by_region[demand_by_region['region'] == region]
        if not region_demand.empty:
            ax.plot(region_demand['year'], region_demand['demand'],
                    marker='^', color=colors['demand'], linewidth=2,
                    label='Demand')

        # Customize plot
        ax.set_title(f"{region}", fontsize=14)
        ax.set_ylabel('Heat Pump Production (GW)', fontsize=13)
        ax.set_xlabel('Year', fontsize=13)

        # Set x-axis ticks to show years (use demand years if capacity is empty)
        x_ticks = sorted(region_capacity['year'].unique() if not region_capacity.empty
                         else region_demand['year'].unique())
        ax.set_xticks(x_ticks)
        ax.tick_params(axis='x', rotation=45)

        # Add grid
        ax.grid(True, which="both", ls="-", alpha=0.2)

    # Add a single legend to the bottom of the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               bbox_to_anchor=(0.5, 0.02),
               ncol=5, frameon=False, fontsize=13)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend

    # Save the plot
    output_path = output_dir / 'regional_comparison_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    return fig

if __name__ == "__main__":
    flow_file = "./HP_Model/parameter_results/flow_conversion_output/flow_conversion_output_scenarios.csv"
    capacity_file = "./HP_Model/parameter_results/capacity/capacity_scenarios.csv"
    demand_file = './ZEN-Model_HP/set_carriers/HP/demand_yearly_variation.csv'
    # Create the plots
    plot_regional_comparison(flow_file, capacity_file, demand_file)