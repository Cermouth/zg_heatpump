import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib
from matplotlib.gridspec import GridSpec

try:
    matplotlib.use('TkAgg')  # Try TkAgg first
except:
    try:
        matplotlib.use('Qt5Agg')  # Try Qt5Agg if TkAgg fails
    except:
        matplotlib.use('Agg')  # Fallback to Agg if all else fails
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend('Agg')  # Additional backend switch
import seaborn as sns
from matplotlib.patches import Patch
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import math
import matplotlib.ticker as ticker
import joypy

# Update global font size and styles
plt.rcParams.update({
    'font.size': 13,  # General font size
    'axes.labelsize': 13,  # Font size for axis labels
    'axes.titlesize': 13,  # Font size for subplot titles
    'xtick.labelsize': 13,  # Font size for x-axis tick labels
    'ytick.labelsize': 13,  # Font size for y-axis tick labels
    'legend.fontsize': 13,  # Font size for legend
    'figure.titlesize': 15  # Font size for figure titles
})


def plot_scenario_results(param_dir, parameter, technology, carrier):
    """
    Create stacked bar plots for flow data by scenario, mode, and region
    """
    try:
        output_dir = Path(
            "./HP_Model/Final_plots")
        # Set up Seaborn style
        sns.set_theme(style="whitegrid")

        # Color palette
        colors = ["#b7afd5", "#becbd8",
                  "#bfcc67",
                  "#b2cfa9",
                  "#9c91c7",
                  "#b2cde9",
                  "#9fc7c6",
                  "#ae879c",
                  "#bc556d",
                  "#d596bd", "#ada599",
                  "#ddd7c6"]
        palette = sns.color_palette(colors, n_colors=12)

        data_file = Path(param_dir)

        df = pd.read_csv(data_file)

        # Filter for technology/carrier
        if 'technology' in df.columns:
            mask = df['technology'].astype(str) == str(technology)
            df = df[mask]
        elif 'carrier' in df.columns:
            mask = df['carrier'].astype(str) == str(carrier)
            df = df[mask]

        if len(df) == 0:
            print(f"No data found for {'technology' if 'technology' in df.columns else 'carrier'} {technology}")
            return

        scenario_cols = [col for col in df.columns if col.startswith('value_')]

        # Set up time and category columns
        if parameter == 'capacity':
            time_col = 'year'
            category_cols = ['location', 'capacity_type']
        else:
            time_col = 'time_operation'
            category_cols = ['node']

        # Verify time column exists
        if time_col not in df.columns:
            print(f"Error: '{time_col}' column not found in DataFrame")
            return

        # Modified subplot layout: 3 rows, 2 columns
        n_rows = 2
        n_cols = 3

        # Create figure
        fig = plt.figure(figsize=(18, 10))

        # Dictionary for scenario name mapping
        scenario_names = {
            'value_scenario_': 'BAU',
            'value_scenario_DemandMet': 'DemandMet',
            'value_scenario_SelfSuff40': 'SelfSuff40',
            'value_scenario_Recycling': 'Recycling',
            'value_scenario_FastDiffusion': 'FastDiffusion',
            'value_scenario_Tariffs': 'Tariffs'
        }

        # Process each scenario
        for idx, scenario_col in enumerate(scenario_cols, 1):
            try:
                # Prepare data for stacked bar plot
                pivot_data = pd.pivot_table(
                    data=df,
                    values=scenario_col,
                    index=time_col,
                    columns=category_cols,
                    fill_value=0
                )

                # Add 2022 to x-axis values
                pivot_data.index = pivot_data.index + 2022

                # Create subplot
                ax = plt.subplot(n_rows, n_cols, idx)

                if parameter == 'capacity':
                    # Line plot for capacity
                    lines = pivot_data.plot(kind='line', marker='o', ax=ax, color=palette)
                    if idx == 1:
                        handles = lines.get_lines()
                        labels = pivot_data.columns
                else:
                    # Stacked bar plot
                    bottom = np.zeros(len(pivot_data))
                    bars = []

                    for i, column in enumerate(pivot_data.columns):
                        bar = ax.bar(pivot_data.index,
                                     pivot_data[column],
                                     bottom=bottom,
                                     width=0.8,
                                     label=column if isinstance(column, str) else ' - '.join(column),
                                     color=palette[i % len(palette)],
                                     edgecolor='none',
                                     linewidth=0)
                        bars.append(bar)
                        bottom += pivot_data[column]

                    if idx == 1:
                        handles = bars
                        labels = [column if isinstance(column, str) else ' - '.join(column)
                                  for column in pivot_data.columns]

                # Set title using the mapping
                ax.set_title(scenario_names.get(scenario_col, scenario_col), fontsize=14)
                ax.set_ylabel('Heat Pump Production (GW)', fontsize=13)

                if len(pivot_data.index) > 5:
                    plt.xticks(rotation=45)

                # Remove individual subplot legends
                if ax.get_legend():
                    ax.get_legend().remove()

            except Exception as e:
                print(f"Error creating subplot for scenario {scenario_col}: {str(e)}")
                continue

        # Add a single legend to the bottom of all subplots
        fig.legend(handles,
                   labels,
                   bbox_to_anchor=(0.5, -0.05),
                   loc='lower center',
                   frameon=False,
                   ncols=6, fontsize=13)

        # Adjust layout to prevent legend overlap
        plt.tight_layout()

        # Save individual plots
        file_prefix = technology if 'technology' in df.columns else carrier
        plot_file = output_dir / f"production_{parameter}_{file_prefix}_scenarios.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error in plot creation: {str(e)}")
        if 'plt' in globals():
            plt.close()


def plot_pie_scenarios(param_dir):
    """
    Create three separate plots comparing node distributions across scenarios,
    with Olympic rings-style layout (3 above, 2 below centered) and a single legend.
    """
    try:
        # Set style
        sns.set_theme(style="whitegrid")

        # Define simplified color scheme
        node_colors = {
            "AUS": "#b7afd5",
            "BRA": "#bfcc67",
            "CHN": "#b2cfa9",
            "EUR": "#9fc7c6",
            "JPN": "#ae879c",
            "KOR": "#bc556d",
            "ROW": "#ada599",
            "USA": "#ddd7c6"
        }

        # Read data
        data_file = Path(param_dir)
        df = pd.read_csv(data_file)

        # Combine European countries
        european_countries = ['CZE', 'AUT', 'ITA', 'DEU', 'ROE']
        df['node'] = df['node'].replace(european_countries, 'EUR')

        # Get scenario columns
        scenario_cols = [col for col in df.columns if col.startswith('value_scenario')]

        # Dictionary for scenario name mapping
        scenario_names = {
            'value_scenario_': 'BAU',
            'value_scenario_DemandMet': 'DemandMet',
            'value_scenario_SelfSuff40': 'SelfSuff40',
            'value_scenario_Recycling': 'Recycling',
            'value_scenario_FastDiffusion': 'FastDiffusion',
            'value_scenario_Tariffs': 'Tariffs'
        }

        def create_scenario_plot(data, title, filename):
            # Create figure
            fig = plt.figure(figsize=(26, 8))  # Wider, shorter figure

            # Create grid with Olympic rings layout
            gs = GridSpec(2, 6, figure=fig, height_ratios=[1, 1])

            # Create legend patches for later use
            legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in node_colors.values()]
            legend_labels = list(node_colors.keys())

            # Define axes positions for Olympic rings layout
            for idx, scenario in enumerate(scenario_cols):
                ax = fig.add_subplot(gs[0, idx])  # All plots in row 0, varying column

                node_values = data.groupby('node')[scenario].sum()
                total = node_values.sum()
                percentages = (node_values / total * 100).round(1)

                # Sort wedges for consistent order
                sorted_indices = percentages.index.sort_values()
                percentages = percentages[sorted_indices]

                colors = [node_colors[node] for node in percentages.index]

                # Create pie chart without built-in percentage labels
                wedges, _ = ax.pie(percentages,
                                   labels=None,
                                   colors=colors)

                # Add percentage labels manually
                for i, p in enumerate(percentages):
                    # Get the angle of the wedge center
                    ang = (wedges[i].theta2 + wedges[i].theta1) / 2

                    # Calculate the position for the label
                    # If percentage is small (< 5%), place label outside
                    radius = 1.2 if p < 6.5 else 0.6

                    # Convert angle to radians
                    ang_rad = np.deg2rad(ang)

                    # Calculate x and y positions
                    x = radius * np.cos(ang_rad)
                    y = radius * np.sin(ang_rad)

                    # Add percentage text
                    ax.text(x, y, f'{p:.1f}%',
                            ha='center', va='center',
                            fontsize=11)

                # Add subplot title closer to the plot
                ax.set_title(scenario_names[scenario], pad=0, fontsize=14, y=0.95)

            # Add main title
            plt.suptitle(title, y=1.02, fontsize=13)

            # Add single legend below the plots
            fig.legend(legend_patches,
                       legend_labels,
                       bbox_to_anchor=(0.5, -0.02),
                       loc='lower center',
                       borderaxespad=0,
                       frameon=False,
                       ncol=8,
                       fontsize=13)

            # Adjust spacing between subplots
            plt.subplots_adjust(wspace=-0.2, hspace=-0.1)

            # Save plot
            output_dir = Path(
                "./HP_Model/Final_plots")
            plot_file = output_dir / filename
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved plot: {filename}")

        # 1. HP Assembly at final timestep
        hp_data = df[df['technology'] == 'HP_assembly']
        final_timestep = hp_data['time_operation'].max()
        final_data = hp_data[hp_data['time_operation'] == final_timestep]
        create_scenario_plot(
            final_data,
            'HP Assembly Node Distribution - Final Timestep (2035)',
            'share_hp_assembly_final_timestep_scenarios_2035.png'
        )

        # 2. HP Assembly cumulative
        create_scenario_plot(
            hp_data,
            'HP Assembly Node Distribution - Cumulative (2022-2035)',
            'share_hp_assembly_cumulative_scenarios_cumulated.png'
        )

        # 3. Total supply chain (excluding storage and transport)
        mask = ~df['technology'].str.contains('storage|transport', case=False, na=False)
        supply_chain_data = df[mask]
        create_scenario_plot(
            supply_chain_data,
            'Total Supply Chain Node Distribution - Cumulative (2022-2035)',
            'share_total_supply_chain_scenarios_cumulated.png'
        )
        # 3. Total supply chain (excluding storage and transport) final timestep
        final_suppy_data = supply_chain_data[supply_chain_data['time_operation'] == final_timestep]
        create_scenario_plot(
            final_suppy_data,
            'Total Supply Chain Node Distribution - 2035',
            'share_total_supply_chain_scenarios_2035.png'
        )
        hp_data = df[df['technology'] == 'HP_assembly']
        final_timestep = hp_data['time_operation'].max()
        final_data = hp_data[hp_data['time_operation'] == final_timestep]
        create_scenario_plot(
            final_data,
            'HP Assembly Node Distribution - Final Timestep (2035)',
            'share_hp_assembly_final_timestep_scenarios_2035.png'
        )

    except Exception as e:
        print(f"Error in plot creation: {str(e)}")
        if 'plt' in globals():
            plt.close()
        raise


def plot_flow_conversion(input_file, scenario='value_scenario_'):
    """
    Create and save side-by-side stacked bar plots showing flow conversion outputs per node,
    separated by units (GW and kiloton/year). Saves plots in the same directory as the input file.

    Parameters:
    input_file (str/Path): Path to flow conversion data file
    scenario (str): Scenario column prefix to analyze (default: base scenario)
    """
    # Set seaborn style
    sns.set_style("whitegrid")

    # Define colors for each unit type
    colors_gw = ["#cdd8ba",  # Lavender
                 "#b4a6b9",  # Muted Sky Blue
                 "#ded3d3"]  # Yellow-Green

    colors_kt = [
        "#a0c5de",  # Aluminium Production
        "#b0dbdf",  # Bauxite Mining
        "#c9b5d9",  # Copper Mining
        "#c47c5a",  # Copper Recycling
        "#e1a382",  # Copper Recycling
        "#b4708b",  # Iron
        "#725f91",  # Nickel MInin
        "#d596bd",  # Nickel Recycling
        "#bc556d"  # Steel Production
    ]

    # Ensure input_file is a Path object
    file_path = Path(input_file)
    output_dir = Path(
        "./HP_Model/Final_plots")

    # Read the data file
    df = pd.read_csv(file_path)

    # Get the scenario column
    flow_col = [col for col in df.columns if col.startswith(scenario)][0]

    # Define technologies for each unit type
    gw_technologies = ['Compressor_manufacturing', 'HEX_manufacturing', 'HP_assembly']
    kt_technologies = [tech for tech in df['technology'].unique()
                       if tech not in gw_technologies]

    # Create separate dataframes for each unit type
    df_gw = df[df['technology'].isin(gw_technologies)]
    df_kt = df[df['technology'].isin(kt_technologies)]

    def create_stacked_plot(ax, data, unit_type, colors):
        """
        Function to create a stacked plot on given axes `ax`.
        Returns the plot data for legend handling.
        """
        # Calculate total flow per technology and location
        totals = data.groupby(['technology', 'node'])[flow_col].sum().reset_index()

        if unit_type == "Mt":
            # Convert to tons
            totals[flow_col] = totals[flow_col] / 1000

        # Pivot data for stacked bar plot
        plot_data = totals.pivot(
            index='node',
            columns='technology',
            values=flow_col
        ).fillna(0)

        # Create stacked bar plot
        plot = plot_data.plot(
            kind='bar',
            stacked=True,
            color=colors[:len(plot_data.columns)],
            width=0.8,
            edgecolor='none',
            linewidth=0,
            ax=ax,
            legend=False  # Don't create individual legends
        )

        # Customize axes
        ax.set_ylabel(f'Production ({unit_type})', fontsize=13)
        ax.set_xlabel('')

        # Add grid
        ax.grid(True, which="both", ls="-", alpha=0.2)

        return plot_data

    # Create figure with two subplots (not sharing y-axis)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False, sharex=False)

    # Create both plots and get their data
    plot_data_kt = create_stacked_plot(ax1, df_kt, 'Mt', colors_kt)
    plot_data_gw = create_stacked_plot(ax2, df_gw, 'GW', colors_gw)
    ax1.tick_params(axis='x', rotation=45, labelrotation=45)
    ax2.tick_params(axis='x', rotation=45, labelrotation=45)

    # Create custom legend labels
    custom_labels = {
        'Aluminium_production': 'Aluminium Production',
        'Bauxite_mining': 'Bauxite Mining',
        'Compressor_manufacturing': 'Compressor Manufacturing',
        'Copper_mining': 'Copper Mining',
        'Copper_recycling': 'Copper Recycling',
        'Copper_refinement': 'Copper Refinement',
        'HEX_manufacturing': 'HEX Manufacturing',
        'HP_assembly': 'HP Assembly',
        'Iron_mining': 'Iron Mining',
        'Nickel_mining': 'Nickel Mining',
        'Nickel_recycling': 'Nickel Recycling',
        'Steel_production': 'Steel Production'
    }

    # Create patches for the legend
    legend_patches = []
    legend_labels = []

    # Add kt technologies
    for tech in plot_data_kt.columns:
        color_idx = list(plot_data_kt.columns).index(tech)
        patch = plt.Rectangle((0, 0), 1, 1, fc=colors_kt[color_idx])
        legend_patches.append(patch)
        legend_labels.append(custom_labels.get(tech, tech))

    # Add GW technologies first
    for tech in plot_data_gw.columns:
        color_idx = list(plot_data_gw.columns).index(tech)
        patch = plt.Rectangle((0, 0), 1, 1, fc=colors_gw[color_idx])
        legend_patches.append(patch)
        legend_labels.append(custom_labels.get(tech, tech))

    # Add single legend below the plots
    fig.legend(legend_patches, legend_labels,
               bbox_to_anchor=(0.5, -0.08),  # Center the legend below the plots
               loc='lower center',  # Position it at the bottom center
               borderaxespad=0,
               frameon=False,
               ncol=5,
               fontsize=12)  # Adjust columns to fit legend items horizontally

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.xticks(rotation=45, ha='right')

    # Save the figure
    plot_filename = f'production_all_{scenario}.png'
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    return fig


def plot_capacity_timeseries(input_file, scenario='value_scenario_'):
    """
    Create line plots showing capacity development over time for each node,
    with separate subplots for each GW technology.

    Parameters:
    input_file (str/Path): Path to capacity data file
    scenario (str): Scenario column prefix to analyze (default: base scenario)
    """
    # Set up Seaborn style
    sns.set_theme(style="whitegrid")

    # Use colors from original function
    colors = ["#b7afd5", "#becbd8",
              "#bfcc67",
              "#b2cfa9",
              "#9c91c7",
              "#b2cde9",
              "#9fc7c6",
              "#ae879c",
              "#bc556d",
              "#d596bd", "#ada599",
              "#ddd7c6"]

    # Create year mapping (0 -> 2022, 1 -> 2023, etc.)
    year_mapping = {i: 2022 + i for i in range(14)}  # Goes up to 2035

    # Ensure input_file is a Path object
    output_dir = Path("./HP_Model/Final_plots")

    # Read the data file
    df = pd.read_csv(input_file)

    # Map the years to actual calendar years
    df['actual_year'] = df['year'].map(year_mapping)

    # Get the scenario column
    capacity_col = [col for col in df.columns if col.startswith(scenario)][0]

    # Define GW technologies
    gw_technologies = ['Compressor_manufacturing', 'HEX_manufacturing', 'HP_assembly']

    # Filter for GW technologies
    df_gw = df[df['technology'].isin(gw_technologies)]

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Plot each technology in a separate subplot
    for idx, tech in enumerate(gw_technologies):
        ax = axes[idx]
        tech_data = df_gw[df_gw['technology'] == tech]

        # Plot lines for each node
        for node_idx, node in enumerate(tech_data['location'].unique()):
            node_data = tech_data[tech_data['location'] == node]
            ax.plot(node_data['actual_year'],
                    node_data[capacity_col],
                    label=node,
                    color=colors[node_idx],
                    marker='o',
                    linewidth=2,
                    markersize=6)

        # Customize subplot
        ax.set_title(tech.replace('_', ' '), pad=20, fontsize=14)
        ax.set_ylabel('Capacity (GW)', fontsize=14)
        ax.grid(True, which="both", ls="-", alpha=0.2)

        # Set x-axis ticks to show all years
        ax.set_xticks(range(2022, 2036))
        ax.set_xticklabels(range(2022, 2036), rotation=45, ha='right')
        # Set font size for tick labels
        plt.xticks(fontsize=14)  # Set font size for x-axis ticks
        plt.yticks(fontsize=14)  # Set font size for y-axis ticks

    # Add a single legend to the right of all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               bbox_to_anchor=(0.5, -0.05),  # Center the legend below the plots
               loc='lower center',  # Position it at the bottom center
               borderaxespad=0,
               frameon=False,
               ncol=6)  # Adjust columns to fit legend items horizontally

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / 'capacity_development_by_technology.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    return plt.gcf()


def plot_cumulative_capacity_extension(input_file, scenario='value_scenario_'):
    """
    Create and save a stacked bar plot showing cumulative capacity extension
    for GW technologies (Compressor, HEX, HP) per node.

    Parameters:
    input_file (str/Path): Path to flow conversion data file
    scenario (str): Scenario column prefix to analyze (default: base scenario)
    """
    # Set seaborn style
    sns.set_style("whitegrid")

    # Colors for GW technologies
    colors_gw = [
        "#c7a3d0",  # Lavender
        "#98cce2",  # Muted Sky Blue
        "#cfe8a3",  # Yellow-Green
    ]

    # Ensure input_file is a Path object
    file_path = Path(input_file)
    output_dir = file_path.parent

    # Read the data file
    df = pd.read_csv(file_path)

    # Get the scenario column
    flow_col = [col for col in df.columns if col.startswith(scenario)][0]

    # Define GW technologies
    gw_technologies = ['Compressor_manufacturing', 'HEX_manufacturing', 'HP_assembly']

    # Filter for GW technologies
    df_gw = df[df['technology'].isin(gw_technologies)]

    # Calculate total capacity per technology and node
    totals = df_gw.groupby(['technology', 'location'])[flow_col].sum().reset_index()

    # Pivot data for stacked bar plot
    plot_data = totals.pivot(
        index='location',
        columns='technology',
        values=flow_col
    ).fillna(0)

    # Create figure
    plt.figure(figsize=(12, 6))

    # Create stacked bar plot
    ax = plot_data.plot(
        kind='bar',
        stacked=True,
        color=colors_gw,
        width=0.8,
        edgecolor='none',
        linewidth=0
    )

    # Customize plot appearance
    plt.title('Cumulative Capacity Extension by Node',
              pad=20,
              fontsize=12,
              fontweight='bold')

    plt.xlabel('Node', fontsize=10)
    plt.ylabel('Capacity (GW)', fontsize=10)

    # Custom labels for legend
    custom_labels = {
        'Compressor_manufacturing': 'Compressor Manufacturing',
        'HEX_manufacturing': 'HEX Manufacturing',
        'HP_assembly': 'HP Assembly'
    }

    # Get the current handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Replace the labels with custom ones
    custom_labels = [custom_labels.get(label, label) for label in labels]

    # Create the legend with custom labels
    plt.legend(handles, custom_labels,
               bbox_to_anchor=(0.5, -0.15),  # Center the legend below the plots
               loc='lower center',  # Position it at the bottom center
               borderaxespad=0,
               frameon=False,
               ncol=4)  # Adjust columns to fit legend items horizontally

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Format y-axis with scientific notation
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Add grid
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Adjust layout
    plt.tight_layout()

    # Add subtle spines
    sns.despine(left=False, bottom=False)

    # Save plot
    plot_path = output_dir / 'capacity_cumulated_extension_by_node.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    return plt.gcf()


def plot_yearly_capacity_extension(input_file, scenario='value_scenario_'):
    """
    Create and save a stacked bar plot showing yearly capacity extension
    summed across all nodes for GW technologies (Compressor, HEX, HP).

    Parameters:
    input_file (str/Path): Path to flow conversion data file
    scenario (str): Scenario column prefix to analyze (default: base scenario)
    """
    # Set seaborn style
    sns.set_style("whitegrid")

    # Colors for GW technologies
    colors_gw = [
        "#c7a3d0",  # Lavender
        "#98cce2",  # Muted Sky Blue
        "#cfe8a3",  # Yellow-Green
    ]

    # Ensure input_file is a Path object
    file_path = Path(input_file)
    output_dir = file_path.parent

    # Read the data file
    df = pd.read_csv(file_path)

    # Get the scenario column
    flow_col = [col for col in df.columns if col.startswith(scenario)][0]

    # Define GW technologies
    gw_technologies = ['Compressor_manufacturing', 'HEX_manufacturing', 'HP_assembly']

    # Filter for GW technologies
    df_gw = df[df['technology'].isin(gw_technologies)]

    # Calculate total capacity per technology and year (summing across all nodes)
    totals = df_gw.groupby(['technology', 'year'])[flow_col].sum().reset_index()

    # Adjust the 'year' column to start from 2022
    totals['year'] = totals['year'] + 2022
    # Pivot data for stacked bar plot
    plot_data = totals.pivot(
        index='year',
        columns='technology',
        values=flow_col
    ).fillna(0)

    # Create figure
    plt.figure(figsize=(10, 9))

    # Create stacked bar plot
    ax = plot_data.plot(
        kind='bar',
        stacked=True,
        color=colors_gw,
        width=0.8,
        edgecolor='none',
        linewidth=0
    )

    # Customize plot appearance
    plt.title('Yearly Capacity Extension Across All Nodes',
              pad=20,
              fontsize=12,
              fontweight='bold')

    plt.xlabel('Year', fontsize=10)
    plt.ylabel('Capacity (GW)', fontsize=10)

    # Custom labels for legend
    custom_labels = {
        'Compressor_manufacturing': 'Compressor Manufacturing',
        'HEX_manufacturing': 'HEX Manufacturing',
        'HP_assembly': 'HP Assembly'
    }

    # Get the current handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Replace the labels with custom ones
    custom_labels = [custom_labels.get(label, label) for label in labels]

    # Create the legend with custom labels
    plt.legend(handles, custom_labels,
               title='Technology',
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               borderaxespad=0,
               frameon=True,
               shadow=False
               )

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Format y-axis with scientific notation
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Add grid
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Adjust layout
    plt.tight_layout()

    # Add subtle spines
    sns.despine(left=False, bottom=False)

    # Save plot
    plot_path = output_dir / 'capacity_yearly_extension.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")


def plot_cumulative_costs(input_file, scenario='value_scenario_', cost_type="OPEX"):
    """
    Create and save a stacked bar plot showing cumulative technology costs per node using seaborn.
    Only includes conversion/production technologies (filters out transport and storage).
    Uses logarithmic scale for OPEX and linear scale for CAPEX.

    Parameters:
    input_file (str/Path): Path to cost data file
    scenario (str): Scenario column prefix to analyze (default: base scenario)
    cost_type (str): Type of cost data ('OPEX' or 'CAPEX'). Determines scaling.
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    # Carrier Color Palette
    carrier_colors = [
        "#d9c2bd",  # Pale Amber
        "#a3d9c5",  # Soft Teal
        "#c7a3d0",  # Lavender
        "#f7b6c2",  # Blush Pink
        "#c3bfe3",  # Soft Peach (Replaced Light Peach)
        "#98cce2",  # Muted Sky Blue (Updated Sky Blue)
        "#cfe8a3",  # Yellow-Green
        "#b4e0a8",  # Green
        "#f5a89c",  # Coral
        "#e6ce87",  # gray /beige
    ]
    # Set Carrier Color Palette in Seaborn
    colors = sns.color_palette(carrier_colors, n_colors=10)

    # Convert to Path objects
    file_path = Path(input_file)

    # Read the data files
    df = pd.read_csv(file_path)

    # Filter out transport and storage technologies
    exclude_keywords = ['transport', 'storage']
    df = df[~df['technology'].str.lower().str.contains('|'.join(exclude_keywords))]

    # Filter for base scenario
    cost_col = [col for col in df.columns if col.startswith(scenario)][0]

    # Calculate total cost per technology and location
    totals = df.groupby(['technology', 'location'])[cost_col].sum().reset_index()

    # Convert costs to Mio. euros
    totals[cost_col] = totals[cost_col] / 1000

    # Pivot data for stacked bar plot
    plot_data = totals.pivot(
        index='location',
        columns='technology',
        values=cost_col
    ).fillna(0)

    # Create figure with specified size
    plt.figure(figsize=(12, 6))

    # Create stacked bar plot
    ax = plot_data.plot(
        kind='bar',
        stacked=True,
        color=colors,
        width=0.8,
        edgecolor='none',
        linewidth=0
    )

    # Format y-axis with scientific notation for CAPEX
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Customize plot appearance
    plt.title(f'Cumulative {cost_type} Conversion Technology by Node',
              pad=20,
              fontsize=12,
              fontweight='bold')

    plt.xlabel('Node', fontsize=10)
    plt.ylabel(f'Total Cost (Mio.€)', fontsize=10)

    # Create a dictionary mapping original labels to custom labels
    custom_labels = {
        'Aluminium_production': 'Aluminium Production',
        'Bauxite_mining': 'Bauxite Mining',
        'Compressor_manufacturing': 'Compressor Manufacturing',
        'Copper_mining': 'Copper Mining',
        'Copper_refinement': 'Copper Refinement',
        'HEX_manufacturing': 'HEX Manufacturing',
        'HP_assembly': 'HP Assembly',
        'Iron_mining': 'Iron Mining',
        'Nickel_mining': 'Nickel Mining',
        'Steel_production': 'Steel Production'
    }

    # Get the current handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Replace the labels with custom ones
    custom_labels = [custom_labels.get(label, label) for label in labels]

    # Create the legend with custom labels
    plt.legend(handles, custom_labels,
               title='Technology',
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               borderaxespad=0,
               frameon=True,
               shadow=False
               )

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Add subtle spines
    sns.despine(left=False, bottom=False)

    # Save plot
    output_dir = Path("./HP_Model/Final_plots")
    plot_path = output_dir / f'cost_cumulative_conversion_{cost_type}_by_node.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    return plt.gcf()


def plot_temporal_costs(opex_file, capex_file, scenario='value_scenario_'):
    """
    Create and save a stacked bar plot showing cumulative technology costs over time periods.
    Shows how different technology costs evolve across timesteps, summed across all nodes.
    Uses a linear scale for both OPEX and CAPEX.

    Parameters:
    input_file (str/Path): Path to cost data file
    scenario (str): Scenario column prefix to analyze (default: base scenario)
    cost_type (str): Type of cost data ('OPEX' or 'CAPEX').
    """

    # Set seaborn style
    sns.set_style("whitegrid")
    # Carrier Color Palette
    carrier_colors = [
        "#a0c5de",
        "#b0dbdf",
        "#cdd8ba",
        "#c9b5d9",
        "#725f91",
        "#b4a6b9",
        "#ded3d3",
        "#b4708b",
        "#d596bd",
        "#bc556d"
    ]

    # Set Carrier Color Palette in Seaborn
    colors = sns.color_palette(carrier_colors, n_colors=10)

    # Read the data files
    df_opex = pd.read_csv(opex_file)
    df_capex = pd.read_csv(capex_file)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Process and plot each dataset
    for df, ax, cost_type in [(df_opex, ax1, 'OPEX'), (df_capex, ax2, 'CAPEX')]:
        # Filter out transport and storage technologies
        exclude_keywords = ['transport', 'storage']
        df = df[~df['technology'].str.lower().str.contains('|'.join(exclude_keywords))]

        # Filter for base scenario
        cost_col = [col for col in df.columns if col.startswith(scenario)][0]

        # Calculate total cost per technology and timestep (summing across all nodes)
        totals = df.groupby(['technology', 'year'])[cost_col].sum().reset_index()

        # Convert costs to euros
        totals[cost_col] = totals[cost_col] * 1000

        # Adjust the 'year' column to start from 2022
        totals['year'] = totals['year'] + 2022

        # Pivot data for stacked bar plot
        plot_data = totals.pivot(
            index='year',
            columns='technology',
            values=cost_col
        ).fillna(0)

        # Create stacked bar plot
        plot_data.plot(
            kind='bar',
            stacked=True,
            color=colors,
            width=0.8,
            edgecolor='none',
            linewidth=0,
            legend=False,
            ax=ax
        )

        ax.grid(True, which="both", ls="-", alpha=0.2)
        # Format y-axis for better readability
        ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(9, 9))

        ax.set_ylabel(f'Total {cost_type} Cost (€)', fontsize=13)
        ax.set_xlabel('')
    # Rotate x-axis labels
    ax1.tick_params(axis='x', rotation=45, labelrotation=45, labelsize=13)
    ax2.tick_params(axis='x', rotation=45, labelrotation=45, labelsize=13)

    # Create a dictionary mapping original labels to custom labels
    custom_labels = {
        'Aluminium_production': 'Aluminium Production',
        'Bauxite_mining': 'Bauxite Mining',
        'Compressor_manufacturing': 'Compressor Manufacturing',
        'Copper_mining': 'Copper Mining',
        'Copper_refinement': 'Copper Refinement',
        'HEX_manufacturing': 'HEX Manufacturing',
        'HP_assembly': 'HP Assembly',
        'Iron_mining': 'Iron Mining',
        'Nickel_mining': 'Nickel Mining',
        'Steel_production': 'Steel Production'
    }

    # Get the current handles and labels from the last plot
    handles, labels = ax2.get_legend_handles_labels()

    # Replace the labels with custom labels
    custom_labels = [custom_labels.get(label, label) for label in labels]

    # Add single legend below the plots
    fig.legend(handles, custom_labels,
               bbox_to_anchor=(0.5, -0.05),  # Center the legend below the plots
               loc='center',  # Position it at the bottom center
               borderaxespad=0,
               frameon=False,
               fontsize=13,
               ncol=5)  # Adjust columns to fit legend items horizontally

    plt.xticks(rotation=45, ha='right')
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot
    output_dir = Path(
        "./HP_Model/Final_plots")
    plot_path = output_dir / 'cost_temporal_combined.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")


def plot_total_costs_scenarios(opex_file, capex_file, save=True):
    """
    Create a stacked bar plot showing total cumulative OPEX and CAPEX costs for each scenario.
    Each scenario gets one bar with OPEX stacked on top of CAPEX.

    Parameters:
    opex_file (str/Path): Path to OPEX data CSV file
    capex_file (str/Path): Path to CAPEX data CSV file
    save (bool): Whether to save the plot to file
    """
    # Set seaborn style
    sns.set_style("whitegrid")

    # Read the data files
    opex_data = pd.read_csv(opex_file)
    capex_data = pd.read_csv(capex_file)

    # Get scenarios
    scenarios = [col for col in opex_data.columns if col.startswith('value_scenario')]

    # Calculate total costs for each scenario
    total_costs = {
        'scenario': [],
        'capex': [],
        'opex': []
    }
    # Dictionary for scenario name mapping
    scenario_names = {
        'value_scenario_': 'BAU',
        'value_scenario_DemandMet': 'DemandMet',
        'value_scenario_SelfSuff40': 'SelfSuff40',
        'value_scenario_Recycling': 'Recycling',
        'value_scenario_FastDiffusion': 'FastDiffusion',
        'value_scenario_Tariffs': 'Tariffs'
    }

    for scenario in scenarios:
        scenario_name = scenario_names.get(scenario, scenario)
        total_costs['scenario'].append(scenario_name)
        total_costs['capex'].append(capex_data[scenario].sum())
        total_costs['opex'].append(opex_data[scenario].sum())

    # Convert costs euros
    total_costs['capex'] = [x * 1000 for x in total_costs['capex']]
    total_costs['opex'] = [x * 1000 for x in total_costs['opex']]

    # Convert to DataFrame for plotting
    plot_data = pd.DataFrame(total_costs)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create stacked bars
    bar_width = 0.5
    x = np.arange(len(scenarios))

    # Plot CAPEX (bottom)
    plt.bar(x, plot_data['capex'], bar_width,
            label='CAPEX', color="#ada599", alpha=0.7, edgecolor='none')

    # Plot OPEX (top)
    plt.bar(x, plot_data['opex'], bar_width,
            bottom=plot_data['capex'],
            label='OPEX', color="#ddd7c6", alpha=0.7, edgecolor='none')

    # Customize the plot
    plt.ylabel('Total Cumulative Cost (€)', fontsize=12)

    # Set x-axis ticks
    plt.xticks(x, plot_data['scenario'], rotation=45, ha='right')

    # Set the y-axis to use scientific notation with a specific format
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

    ax.grid(True, which="both", ls="-", alpha=0.2)
    # Optionally, customize the tick label formatting (e.g., for different powers of 10)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(9, 9))

    # Add value labels on the bars
    for i in x:
        # Label for CAPEX
        capex_value = plot_data['capex'].iloc[i]
        capex_display = capex_value / 1e9  # Divide by 10^9
        plt.text(i, capex_value / 2,
                 f'{capex_display:.1f}',  # Display in billions with 2 decimal places
                 ha='center', va='center')

        # Label for OPEX
        opex_value = plot_data['opex'].iloc[i]
        opex_display = opex_value / 1e9  # Divide by 10^9
        total_height = capex_value + opex_value
        plt.text(i, capex_value + opex_value / 2,
                 f'{opex_display:.1f}',  # Display in billions with 2 decimal places
                 ha='center', va='center')

    # Add legend below the plot and centered
    plt.legend(
        loc='lower center',  # Center horizontally
        bbox_to_anchor=(0.5, -0.3),  # Fine-tune the position (0.5 = center horizontally, -0.15 = below the plot)
        ncol=2,  # Display legend items in 3 columns (adjust based on number of items)
        frameon=False)

    # Adjust layout to prevent text cutoff
    plt.tight_layout()

    if save:
        # Get the directory of the input file for saving
        save_dir = Path("./HP_Model/Final_plots")

        filepath = save_dir / 'cost_total_scenario_costs.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filepath}")

    return fig


def plot_country_costs_scenarios(opex_file, capex_file, save=True):
    """
    Create a stacked bar plot showing costs by country for each scenario.
    Each scenario gets one bar with countries stacked to show their contribution.

    Parameters:
    opex_file (str/Path): Path to OPEX data CSV file
    capex_file (str/Path): Path to CAPEX data CSV file
    save (bool): Whether to save the plot to file
    """
    # Set seaborn style
    sns.set_style("whitegrid")

    # Read the data files
    opex_data = pd.read_csv(opex_file)
    capex_data = pd.read_csv(capex_file)

    exclude_keywords = ['transport', 'storage']
    opex_data = opex_data[~opex_data['technology'].str.lower().str.contains('|'.join(exclude_keywords))]
    capex_data = capex_data[~capex_data['technology'].str.lower().str.contains('|'.join(exclude_keywords))]
    # Combine European countries
    european_countries = ['CZE', 'AUT', 'ITA', 'DEU', 'ROE']
    opex_data['location'] = opex_data['location'].replace(european_countries, 'EUR')
    capex_data['location'] = capex_data['location'].replace(european_countries, 'EUR')

    # Define node colors (from the provided code)
    node_colors = {
        "AUS": "#b7afd5",
        "BRA": "#bfcc67",
        "CHN": "#b2cfa9",
        "EUR": "#9fc7c6",
        "JPN": "#ae879c",
        "KOR": "#bc556d",
        "ROW": "#ada599",
        "USA": "#ddd7c6"
    }

    # Get scenarios
    scenarios = [col for col in opex_data.columns if col.startswith('value_scenario')]

    # Dictionary for scenario name mapping
    scenario_names = {
        'value_scenario_': 'BAU',
        'value_scenario_DemandMet': 'DemandMet',
        'value_scenario_SelfSuff40': 'SelfSuff40',
        'value_scenario_Recycling': 'Recycling',
        'value_scenario_FastDiffusion': 'FastDiffusion',
        'value_scenario_Tariffs': 'Tariffs'
    }

    # Create a combined dataframe for total costs (OPEX + CAPEX)
    # Group by node and sum for each scenario
    total_costs_by_country = {}

    # Process each scenario
    for scenario in scenarios:
        # Group and sum OPEX by node for this scenario
        opex_by_node = opex_data.groupby('location')[scenario].sum()

        # Group and sum CAPEX by node for this scenario
        capex_by_node = capex_data.groupby('location')[scenario].sum()

        # Combine OPEX and CAPEX for each node
        total_by_node = opex_by_node + capex_by_node

        # Convert to thousands of euros
        total_by_node = total_by_node * 1000

        # Store in dictionary
        total_costs_by_country[scenario] = total_by_node

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Setup for stacked bars
    bar_width = 0.5
    x = np.arange(len(scenarios))
    bottom = np.zeros(len(scenarios))

    # Get all unique nodes across all scenarios
    all_nodes = sorted(set().union(*[set(costs.index) for costs in total_costs_by_country.values()]))

    # Plot stacked bars for each node
    for node in all_nodes:
        node_values = []

        for scenario in scenarios:
            if node in total_costs_by_country[scenario].index:
                node_values.append(total_costs_by_country[scenario][node])
            else:
                node_values.append(0)

        # Plot this node's contribution for each scenario
        bars = ax.bar(x, node_values, bar_width, label=node,
                      bottom=bottom, color=node_colors.get(node, "#999999"), alpha=0.7, edgecolor='none')

        # Add value labels for this node's contribution
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:  # Only add label if there's a visible contribution
                # Calculate percentage of total for this scenario
                scenario = scenarios[i]
                total_for_scenario = sum(total_costs_by_country[scenario])
                percentage = (height / total_for_scenario) * 100

                # Only show label if percentage is significant enough
                if percentage > 5:
                    # Position the text in the middle of this node's segment
                    y_pos = bottom[i] + height / 2

                    # Format in billions for readability
                    value_in_billions = height / 1e9

                    ax.text(i, y_pos, f'{value_in_billions:.1f}',
                            ha='center', va='center', fontsize=9)

        # Update the bottom for the next stack
        bottom += np.array(node_values)

    # Customize the plot
    ax.set_ylabel('Total Cost (€)', fontsize=12)
    ax.set_xticks(x)

    # *** FIX IS HERE ***
    # Use .get(s, s) for a safe lookup. If the key 's' is not found, it defaults to 's'.
    ax.set_xticklabels([scenario_names.get(s, s) for s in scenarios], rotation=45, ha='right')

    # Set the y-axis to use scientific notation with a specific format
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(9, 9))

    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Add legend below the plot and centered
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, -0.3),
        ncol=8,  # len(all_nodes) // 2 + len(all_nodes) % 2,  # Distribute nodes evenly
        frameon=False
    )

    # Adjust layout to prevent text cutoff
    plt.tight_layout()

    if save:
        # Get the directory for saving
        save_dir = Path("./HP_Model/Final_plots")

        filepath = save_dir / 'cost_country_costs_by_scenario.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filepath}")

    return fig


def plot_timestep_costs_comparison(opex_file, capex_file, save=True):
    """
    Create a single stacked bar plot showing OPEX and CAPEX costs for each timestep across multiple scenarios, including the base case.
    Each scenario is represented as a group of bars for direct comparison.

    Parameters:
    opex_file (str/Path): Path to OPEX data CSV file
    capex_file (str/Path): Path to CAPEX data CSV file
    save (bool): Whether to save the plot to file
    """
    # Set seaborn style
    sns.set_style("whitegrid")

    # Read the data files
    opex_data = pd.read_csv(opex_file)
    capex_data = pd.read_csv(capex_file)

    # Filter columns that start with 'value_scenario_'
    scenarios = [col for col in opex_data.columns if col.startswith('value_scenario_')]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create stacked bars for all scenarios
    bar_width = 0.15
    x = np.arange(len(capex_data))

    for i, scenario in enumerate(scenarios):
        # Convert costs to euros
        opex_data[scenario] = opex_data[scenario] * 1000
        capex_data[scenario] = capex_data[scenario] * 1000

        # Offset for each scenario
        offset = x + i * bar_width

        # Plot CAPEX (bottom)
        plt.bar(offset, capex_data[scenario], bar_width,
                label=f'CAPEX ({scenario})', color=f'#a3d9c5', alpha=0.7)

        # Plot OPEX (top)
        plt.bar(offset, opex_data[scenario], bar_width,
                bottom=capex_data[scenario],
                label=f'OPEX ({scenario})', color=f'#c7a3d0', alpha=0.7)

    # Customize the plot
    plt.title('Comparison of OPEX and CAPEX Across Scenarios', fontsize=16)
    plt.xlabel('', fontsize=14)
    plt.ylabel('Cost (€)', fontsize=14)

    # Set x-axis ticks
    plt.xticks(x + (len(scenarios) - 1) * bar_width / 2,
               [f'{i + 2022}' for i in range(len(capex_data))], rotation=45, ha='right')

    # Format y-axis for scientific notation
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(9, 9))
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Add legend
    plt.legend(bbox_to_anchor=(0.5, -0.12), loc='lower center', fontsize=10, frameon=False, ncols=5)

    # Adjust layout to prevent text cutoff
    plt.tight_layout()

    if save:
        # Get the directory of the input file for saving
        save_dir = Path("./HP_Model/Final_plots")

        filepath = save_dir / 'cost_sceanario_comparison_timestep_costs.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filepath}")

    return fig


def plot_nodal_scenario_costs(opex_data, capex_data):
    """
    Create a stacked bar plot showing CAPEX and OPEX costs per node across different scenarios.

    Parameters:
    capex_data (DataFrame): DataFrame containing CAPEX data
    opex_data (DataFrame): DataFrame containing OPEX data
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    capex_data = pd.read_csv(capex_data)
    opex_data = pd.read_csv(opex_data)

    # Filter out transport and storage technologies if needed
    exclude_keywords = ['transport', 'storage']
    capex_data = capex_data[~capex_data['technology'].str.lower().str.contains('|'.join(exclude_keywords))]
    opex_data = opex_data[~opex_data['technology'].str.lower().str.contains('|'.join(exclude_keywords))]

    # Filter columns that start with 'value_scenario_'
    scenarios = [col for col in opex_data.columns if col.startswith('value_scenario_')]

    # Define color pairs for each scenario (CAPEX, OPEX)
    scenario_colors = {
        'value_scenario_': ("#ada599", "#ddd7c6"),
        'value_scenario_DemandMet': ("#9c91c7", "#b7afd5"),
        'value_scenario_SelfSuff40': ("#9fc7c6", "#b2cde9"),
        'value_scenario_Recycling': ("#bfcc67", "#cdd8ba"),
        'value_scenario_FastDiffusion': ("#bc556d", "#d596bd"),
        'value_scenario_Tariffs': ("#e49b3e", "#f0c996")
    }

    # Dictionary for scenario name mapping
    scenario_names = {
        'value_scenario_': 'BAU',
        'value_scenario_DemandMet': 'DemandMet',
        'value_scenario_SelfSuff40': 'SelfSuff40',
        'value_scenario_Recycling': 'Recycling',
        'value_scenario_FastDiffusion': 'FastDiffusion',
        'value_scenario_Tariffs': 'Tariffs'
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Convert costs to euros
    for scenario in scenarios:
        opex_data[scenario] = opex_data[scenario] * 1000
        capex_data[scenario] = capex_data[scenario] * 1000

    # Calculate bar positions
    bar_width = 0.15
    x = np.arange(len(capex_data['location'].unique()))

    # Process each scenario
    for i, scenario in enumerate(scenarios):
        # Calculate totals per location
        location_capex = capex_data.groupby('location')[scenario].sum()
        location_opex = opex_data.groupby('location')[scenario].sum()

        # Calculate bar positions for this scenario
        scenario_position = x + (i - len(scenarios) / 2) * bar_width

        capex_color, opex_color = scenario_colors.get(scenario, ("#808080", "#C0C0C0"))
        scenario_name = scenario_names.get(scenario, scenario)

        # Plot stacked bars
        plt.bar(scenario_position, location_capex, bar_width,
                label=f'{scenario_name} CAPEX',
                color=capex_color, alpha=0.7,
                edgecolor='none')

        plt.bar(scenario_position, location_opex, bar_width,
                bottom=location_capex,
                label=f'{scenario_name} OPEX',
                color=opex_color, alpha=0.7,
                edgecolor='none')

    # Customize plot appearance
    plt.xlabel('', fontsize=10)
    plt.ylabel('Total Cost (€)', fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Set x-axis ticks
    plt.xticks(x, capex_data['location'].unique(), rotation=45, ha='right')

    # Format y-axis with scientific notation
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(9, 9))

    # Add legend
    plt.legend(bbox_to_anchor=(0.5, -0.2),
               loc='lower center',
               borderaxespad=0,
               frameon=False,
               ncols=5)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    output_dir = Path("./HP_Model/Final_plots")
    plot_path = output_dir / 'cost_scenario_nodal_cost_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    return plt.gcf()


def plot_timestep_costs_base(opex_file, capex_file, save=True):
    """
    Create a stacked bar plot showing OPEX and CAPEX costs for each timestep in the base scenario.
    Each timestep gets one bar with OPEX stacked on top of CAPEX.

    Parameters:
    opex_file (str/Path): Path to OPEX data CSV file
    capex_file (str/Path): Path to CAPEX data CSV file
    save (bool): Whether to save the plot to file
    """
    # Set seaborn style
    sns.set_style("whitegrid")

    # Read the data files
    opex_data = pd.read_csv(opex_file)
    capex_data = pd.read_csv(capex_file)

    # Extract base scenario column
    base_scenario = 'value_scenario_'

    # Convert costs to euros
    opex_data[base_scenario] = opex_data[base_scenario] * 1000
    capex_data[base_scenario] = capex_data[base_scenario] * 1000

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create stacked bars
    bar_width = 0.5
    x = np.arange(len(capex_data))

    # Plot CAPEX (bottom)
    plt.bar(x, capex_data[base_scenario], bar_width,
            label='CAPEX', color='#a3d9c5', alpha=0.7)

    # Plot OPEX (top)
    plt.bar(x, opex_data[base_scenario], bar_width,
            bottom=capex_data[base_scenario],
            label='OPEX', color='#c7a3d0', alpha=0.7)

    # Customize the plot
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Cost (€)', fontsize=12)

    # Set x-axis ticks
    plt.xticks(x, [f' {i + 2022}' for i in range(len(capex_data))],
               rotation=45, ha='right')

    # Format y-axis for scientific notation
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(9, 9))
    ax.grid(True, which="both", ls="-", alpha=0.2)
    # # Add value labels on the bars
    for i in x:
        # Label for CAPEX
        capex_value = capex_data[base_scenario].iloc[i]
        capex_display = capex_value / 1e9
        plt.text(i, capex_value / 2,
                 f'{capex_display:,.1f}',
                 ha='center', va='center')

        # Label for OPEX
        opex_value = opex_data[base_scenario].iloc[i]
        total_height = capex_value + opex_value
        opex_display = opex_value / 1e9
        plt.text(i, capex_value + opex_value / 2,
                 f'{opex_display:,.0f}',
                 ha='center', va='center')

    # Add legend
    plt.legend(loc='upper right')

    # Add subtle spines
    sns.despine(left=False, bottom=False)

    # Adjust layout to prevent text cutoff
    plt.tight_layout()

    if save:
        # Get the directory of the input file for saving
        save_dir = Path("./HP_Model/Final_plots")

        filepath = save_dir / 'cost_base_scenario_timestep_costs.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filepath}")

    return fig


def plot_transport_flow(transport_file, parameter='transport'):
    """
    Create Sankey diagrams for transport flows with improved styling to match other plots.
    """
    print("Creating transport flow Sankey diagrams")

    # Set consistent style
    sns.set_theme(style="whitegrid")

    # Define units for different technologies
    technology_units = {
        'HP_transport': 'GW',
        'HEX_transport': 'GW',
        'Compressor_transport': 'GW',
        'Bauxite_transport': 'Mt',  # Changed from kt to Mt
        'Copper_ore_transport': 'Mt',  # Changed from kt to Mt
        'Iron_transport': 'Mt',  # Changed from kt to Mt
        'Nickel_transport': 'Mt',  # Changed from kt to Mt
        'Aluminium_transport': 'Mt',  # Changed from kt to Mt
        'Copper_transport': 'Mt',  # Changed from kt to Mt
        'Steel_transport': 'Mt',  # Changed from kt to Mt
        'Refrigerant_transport': 'Mt'  # Changed from kt to Mt
    }

    # Define technology order for comparison plot
    technology_order = [
        # Column 1
        'Bauxite_transport',
        'Copper_ore_transport',
        'Iron_transport',
        'Nickel_transport',
        # Column 2
        'Aluminium_transport',
        'Copper_transport',
        'Steel_transport',
        'Refrigerant_transport',
        # Column 3
        'HEX_transport',
        'Compressor_transport',
        'HP_transport'
    ]
    # Define which technologies need kt to Mt conversion
    kt_to_mt_technologies = [
        'Bauxite_transport', 'Copper_ore_transport', 'Iron_transport',
        'Nickel_transport', 'Aluminium_transport', 'Copper_transport',
        'Steel_transport', 'Refrigerant_transport'
    ]

    # Define fixed color mapping for countries
    country_color_mapping = {
        'AUS': "#b7afd5",
        'AUT': "#becbd8",
        'BRA': "#bfcc67",
        'CHN': "#b2cfa9",
        'CZE': "#9c91c7",
        'DEU': "#b2cde9",
        'ITA': "#9fc7c6",
        'JPN': "#ae879c",
        'KOR': "#bc556d",
        'ROE': "#d596bd",
        'ROW': "#ada599",
        'USA': "#ddd7c6"
    }

    try:
        param_dir = Path(
            "./HP_Model/Final_plots")
        df = pd.read_csv(transport_file)
        scenario_cols = [col for col in df.columns if col.startswith('value_scenario_')]
        technologies = df['technology'].unique()

        # Convert kt values to Mt for relevant technologies
        for tech in kt_to_mt_technologies:
            tech_mask = df['technology'] == tech
            scenario_cols = [col for col in df.columns if col.startswith('value_scenario_')]
            for col in scenario_cols:
                df.loc[tech_mask, col] = df.loc[tech_mask, col] / 1000  # Convert kt to Mt

        scenario_cols = [col for col in df.columns if col.startswith('value_scenario_')]
        technologies = df['technology'].unique()

        # Define separate technology groups
        gw_technologies = ['HP_transport', 'HEX_transport', 'Compressor_transport']
        mt_technologies = [tech for tech in technology_order if tech not in gw_technologies]

        # Calculate global maximum flows for both unit types
        last_timestep = df['time_operation'].max()
        last_timestep_data = df[df['time_operation'] == last_timestep]

        global_max_flow_gw = 0
        global_max_flow_mt = 0  # Changed from kt to Mt

        for tech in technology_order:
            tech_data = last_timestep_data[last_timestep_data['technology'] == tech]
            if not tech_data.empty:
                for scenario_col in scenario_cols:
                    max_flow = tech_data[scenario_col].max()
                    if tech in gw_technologies:
                        global_max_flow_gw = max(global_max_flow_gw, max_flow)
                    else:
                        global_max_flow_mt = max(global_max_flow_mt, max_flow)

        print(f"Global maximum flow (GW): {global_max_flow_gw:.0f}")
        print(f"Global maximum flow (Mt): {global_max_flow_mt:.0f}")  # Changed from kt to Mt

        def create_sankey_figure(data, value_col, title):
            """Helper function to create a single Sankey diagram with improved styling"""
            sources = []
            targets = []
            values = []
            node_labels = set()

            for _, row in data.iterrows():
                try:
                    source, target = row['edge'].split('-')
                    value = float(row[value_col])
                    if value > 0:
                        sources.append(source)
                        targets.append(target)
                        values.append(value)
                        node_labels.add(source)
                        node_labels.add(target)
                except (ValueError, TypeError, AttributeError) as e:
                    continue

            if not sources:
                return None

            node_labels = list(sorted(node_labels))
            node_indices = {node: idx for idx, node in enumerate(node_labels)}

            # Assign colors based on the fixed mapping
            node_colors = [country_color_mapping.get(label) for label in node_labels]

            return {
                'type': 'sankey',
                'node': {
                    'pad': 20,
                    'thickness': 25,
                    'line': {'color': "gray", 'width': 0.3},
                    'label': [label.replace('_', ' ') for label in node_labels],
                    'color': node_colors
                },
                'link': {
                    'source': [node_indices[s] for s in sources],
                    'target': [node_indices[t] for t in targets],
                    'value': values,
                    'color': [f"rgba({int(int(node_colors[node_indices[s]][1:], 16) / 0x1000000 * 255)}, "
                              f"{int(int(node_colors[node_indices[s]][3:5], 16) / 0x100 * 255)}, "
                              f"{int(int(node_colors[node_indices[s]][5:], 16) * 255)}, 0.3)"
                              for s in sources],
                    'hovertemplate': "From: %{source.label}<br>" +
                                     "To: %{target.label}<br>" +
                                     f"Flow: %{value:.1f} {technology_units.get(title.replace(' ', '_') + '_transport', 'kt')}<extra></extra>",
                    'hoverlabel': {'bgcolor': 'white', 'font': {'family': 'Arial'}}
                }
            }

        # 1. Create scenario comparison for each technology
        for technology in technologies:
            tech_data = df[df['technology'] == technology]
            last_timestep = tech_data['time_operation'].max()
            last_timestep_data = tech_data[tech_data['time_operation'] == last_timestep]

            n_scenarios = len(scenario_cols)
            n_rows = math.ceil(math.sqrt(n_scenarios))
            n_cols = math.ceil(n_scenarios / n_rows)

            fig = go.Figure()

            # Dictionary for scenario name mapping
            scenario_names = {
                'value_scenario_': 'BAU',
                'value_scenario_DemandMet': 'DemandMet',
                'value_scenario_SelfSuff40': 'SelfSuff40',
                'value_scenario_Recycling': 'Recycling',
                'value_scenario_FastDiffusion': 'FastDiffusion',
                'value_scenario_Tariffs': 'Tariffs'
            }

            annotations = []
            for i, scenario_col in enumerate(scenario_cols):
                scenario_name = scenario_names.get(scenario_col, scenario_col)
                row = i // n_cols
                col = i % n_cols

                x_domain = [col / n_cols + 0.05, (col + 0.95) / n_cols]
                y_domain = [(n_rows - 1 - row) / n_rows + 0.05, (n_rows - row - 0.05) / n_rows]

                # Calculate total trade volume for this scenario
                scenario_data = last_timestep_data[last_timestep_data[scenario_col] > 0]
                total_volume = scenario_data[scenario_col].sum()

                # Add units to the total
                unit = technology_units[technology]

                sankey_data = create_sankey_figure(last_timestep_data, scenario_col, f"Scenario {scenario_name}")

                if sankey_data:
                    sankey_data.update(domain=dict(x=x_domain, y=y_domain))
                    fig.add_trace(go.Sankey(**sankey_data))

                    # Add scenario name with total volume and units
                    annotations.append(dict(
                        text=f"{scenario_name}<br>(Total: {total_volume:.1f} {unit})",
                        x=(x_domain[0] + x_domain[1]) / 2,
                        y=y_domain[1] + 0.03,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=12, family='Arial', color='gray'),
                        align='center'
                    ))

            # Update layout with consistent styling
            fig.update_layout(
                width=450 * n_cols,
                height=450 * n_rows + 50,
                showlegend=False,
                paper_bgcolor='white',
                plot_bgcolor='white',
                annotations=annotations,
                margin=dict(t=100, b=50, l=50, r=50),
            )

            # Save with higher DPI
            fig.write_image(str(param_dir / f"transport_{parameter}_{technology}_scenarios.png"), scale=3)
            print(f"Created Sankey diagrams grid for {technology}")

        # 2. Create comparison plot with all technologies for base scenario
        base_scenario = 'value_scenario_' if 'value_scenario_' in scenario_cols else scenario_cols[0]
        last_timestep = df['time_operation'].max()
        last_timestep_data = df[df['time_operation'] == last_timestep]

        # 2. Create comparison plot with all technologies for base scenario
        base_scenario = 'value_scenario_' if 'value_scenario_' in scenario_cols else scenario_cols[0]

        # Calculate grid dimensions based on predefined columns
        n_cols = 3  # Fixed number of columns
        n_rows = 4  # Enough rows to fit all technologies

        # Create figure for comparison plot
        fig = go.Figure()

        # Add each technology as a separate Sankey diagram
        annotations = []
        for i, tech in enumerate(technology_order):
            # Calculate position in grid
            col = i // n_rows
            row = i % n_rows

            # Calculate domain for this subplot
            x_domain = [col / n_cols + 0.05, (col + 0.95) / n_cols]
            y_domain = [(n_rows - 1 - row) / n_rows + 0.05, (n_rows - row - 0.05) / n_rows]

            tech_data = last_timestep_data[last_timestep_data['technology'] == tech]
            if not tech_data.empty:
                tech_total = tech_data[base_scenario].sum()
                sankey_data = create_sankey_figure(tech_data, base_scenario, tech)

                # Add total volume to the tech name for reference
                # Get the unit for this technology
                unit = technology_units[tech]
                tech_display = f"{tech.replace('_transport', '').replace('_', ' ')}\n(Total: {tech_total:.1f} {unit})"

            if sankey_data:
                sankey_data.update(
                    domain=dict(x=x_domain, y=y_domain),
                )
                fig.add_trace(go.Sankey(**sankey_data))

                annotations.append(dict(
                    text=tech_display,
                    x=(x_domain[0] + x_domain[1]) / 2,
                    y=y_domain[1] + 0.03,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14, family='Arial', color='gray')
                ))

        # Just keep track of flow sizes for scaling purposes
        flow_sizes = last_timestep_data[base_scenario].dropna()
        max_flow = flow_sizes.max()

        # Update comparison plot layout
        fig.update_layout(
            width=450 * n_cols,
            height=450 * n_rows + 50,
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white',
            annotations=annotations,
            margin=dict(t=100, b=50, l=50, r=50)
        )

        # Save comparison figure
        fig.write_image(str(param_dir / f"transport_{parameter}_all_technologies_comparison.png"), scale=3)
        print("Created technology comparison Sankey diagrams grid")

    except Exception as e:
        print(f"Error creating Sankey diagrams: {str(e)}")
        raise


# uncertainty plots
def plot_uncertainty_results(param_dir, parameter, uncertainty_type, technology='HP_assembly'):
    if parameter in "cost_opex_yearly":
        print("Creating cost analysis boxplots")
        uncertainty_cost_plotting()
    if parameter in "flow_transport":
        return

    # Read data
    data_file = param_dir / f"{parameter}_MC_{uncertainty_type}.csv"
    print("plotting uncertainty for:")
    print(data_file)
    df = pd.read_csv(data_file)

    # Filter for technology
    if 'technology' in df.columns:
        df = df[df['technology'].astype(str) == str(technology)]

    if len(df) == 0:
        print(f"No data found for technology {technology}")
        return

    # Get max time_operation and filter
    time_col = 'year' if 'year' in df.columns else 'time_operation'
    max_time = df[time_col].max()
    df_filtered = df[df[time_col] == max_time]

    # Get scenario columns and prepare plot data
    scenario_cols = [col for col in df.columns if col.startswith('value_')]
    plot_data = []

    for node in df_filtered['node'].unique():
        node_data = df_filtered[df_filtered['node'] == node]
        if not node_data.empty:
            scenario_values = node_data[scenario_cols].values.flatten()
            plot_data.extend([(node, val) for val in scenario_values if pd.notna(val)])

    # Create and save boxplot
    plot_df = pd.DataFrame(plot_data, columns=['Node', 'Value'])
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Node', y='Value', data=plot_df)

    plt.title(f'Uncertainty Analysis for {technology} in {uncertainty_type} (t={max_time})')
    plt.xlabel('Node')
    plt.ylabel(parameter)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(param_dir / f"{parameter}_{uncertainty_type}_boxplot.png")
    plt.close()


def load_and_prepare_data(file_path):
    """
    Loads and prepares the data for analysis, separating storage and transport technologies.
    """
    df = pd.read_csv(file_path)

    # Convert numeric columns to float
    scenario_cols = [col for col in df.columns if col.startswith('value_scenario_')]
    for col in scenario_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure time column exists
    time_col = 'year' if 'year' in df.columns else 'time_operation'
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')

    # Separate production, storage, and transport technologies
    storage_transport_mask = df['technology'].str.contains('storage|transport', case=False, na=False)

    production_df = df[~storage_transport_mask].copy()
    storage_transport_df = df[storage_transport_mask].copy()

    return production_df, scenario_cols, time_col


def plot_node_comparison(df_opex, df_capacity, scenario_cols_opex, scenario_cols_capacity, technology, output_dir):
    """
    Creates comparative boxplots for OPEX and capacity costs across nodes.
    """
    tech_data_opex = df_opex[df_opex['technology'] == technology]
    tech_data_capacity = df_capacity[df_capacity['technology'] == technology]

    plot_data_opex = []
    plot_data_capacity = []

    # Process OPEX data
    for node in tech_data_opex['location'].unique():
        node_data = tech_data_opex[tech_data_opex['location'] == node]
        for scenario in scenario_cols_opex:
            sum_value = node_data[scenario].sum()
            if pd.notna(sum_value):
                plot_data_opex.append((node, sum_value))

    # Process Capacity data
    for node in tech_data_capacity['location'].unique():
        node_data = tech_data_capacity[tech_data_capacity['location'] == node]
        for scenario in scenario_cols_capacity:
            sum_value = node_data[scenario].sum()
            if pd.notna(sum_value):
                plot_data_capacity.append((node, sum_value))

    if not plot_data_opex or not plot_data_capacity:
        return

    plot_df_opex = pd.DataFrame(plot_data_opex, columns=['Node', 'Total Cost'])
    plot_df_capacity = pd.DataFrame(plot_data_capacity, columns=['Node', 'Total Cost'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # OPEX plot
    sns.boxplot(data=plot_df_opex, x='Node', y='Total Cost', ax=ax1)
    ax1.set_title(f'OPEX Cost Comparison Across Nodes - {technology}')
    ax1.tick_params(axis='x', rotation=45)

    # Capacity plot
    sns.boxplot(data=plot_df_capacity, x='Node', y='Total Cost', ax=ax2)
    ax2.set_title(f'Capacity Cost Comparison Across Nodes - {technology}')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / f'node_comparison_{technology}.png',
                bbox_inches='tight', dpi=300)
    plt.close()


def plot_production_uncertainty_cumulative(param_dir, technology="HP_assembly"):
    """
    Create boxplots for capacity-related parameters (CAPEX, OPEX, Capacity) for a specific technology
    All showing capacity values in GW

    Args:
        param_dir (Path): Directory containing the parameter files
        technology (str): Technology to analyze, defaults to "HP_assembly"
    """
    try:
        # Ensure param_dir is a Path object
        param_dir = Path(param_dir)

        # Set up Seaborn style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = [15, 8]
        colors = ["#bfcc67", "#ada599", "#ddd7c6", "#bc556d"]
        palette = sns.color_palette(colors)

        # Define files to read with more descriptive titles
        param_files = {
            'Production with capacity uncertainty': 'flow_conversion_output_MC_capacity.csv',
            'Production with CAPEX uncertainty': 'flow_conversion_output_MC_capex.csv',
            'Production with OPEX uncertainty ': 'flow_conversion_output_MC_opex.csv',
            'Production with combined uncertainty': 'flow_conversion_output_MC_combined.csv'
        }

        # Create figure for boxplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()  # Convert 2D array to 1D for easier indexing

        # Store all y values to set consistent limits
        all_values = []

        # First pass to collect all y values
        for filename in param_files.values():
            try:
                data_file = param_dir / filename
                if data_file.exists():
                    df = pd.read_csv(data_file)
                    if 'technology' in df.columns:
                        df = df[df['technology'] == str(technology)]
                    scenario_cols = [col for col in df.columns if col.startswith('value_')]
                    # Calculate cumulative values for each node and scenario
                    grouped_df = df.groupby('node')[scenario_cols].sum().reset_index()
                    melted_df = pd.melt(
                        grouped_df,
                        id_vars=['node'],
                        value_vars=scenario_cols,
                        var_name='scenario',
                        value_name='value'
                    )
                    # Convert to numeric, handling any non-numeric values
                    melted_df['value'] = pd.to_numeric(melted_df['value'], errors='coerce')

            except Exception as e:
                print(f"Error in first pass for {filename}: {str(e)}")
                continue

        # Calculate global y limits
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            y_range = y_max - y_min
            y_limits = [y_min - 0.1 * y_range, y_max + 0.1 * y_range]

        # Process each parameter file
        for idx, (param_name, filename) in enumerate(param_files.items()):
            try:
                # Read data file
                data_file = param_dir / filename
                print(f"Reading file: {data_file.absolute()}")

                if not data_file.exists():
                    print(f"No data file found at: {data_file.absolute()}")
                    continue

                df = pd.read_csv(data_file)

                # Filter for technology
                if 'technology' in df.columns:
                    mask = df['technology'].astype(str) == str(technology)
                    df = df[mask]

                if len(df) == 0:
                    print(f"No data found for technology {technology} in {filename}")
                    continue

                # Get scenario columns
                scenario_cols = [col for col in df.columns if col.startswith('value_')]
                # Calculate cumulative values for each node and scenario
                grouped_df = df.groupby('node')[scenario_cols].sum().reset_index()

                ## Melt the dataframe to get all scenario values in one column
                melted_df = pd.melt(
                    grouped_df,
                    id_vars=['node'],
                    value_vars=scenario_cols,
                    var_name='scenario',
                    value_name='value'
                )

                # Convert to numeric, handling any non-numeric values
                melted_df['value'] = pd.to_numeric(melted_df['value'], errors='coerce')

                # Create boxplot using seaborn in the subplot
                sns.violinplot(
                    data=melted_df,
                    x='node',
                    y='value',
                    ax=axes[idx],
                    width=0.7,
                    inner='box',  # Shows box plot inside violin
                    density_norm='width',  # Normalize the violin width
                    color=colors[idx],
                    linewidth=0.8
                )

                # Customize plot
                axes[idx].set_title(param_name, pad=20)
                axes[idx].set_xlabel('')
                axes[idx].set_ylabel('Cumulative Heat Pump Production [GW]')

                # Set consistent y limits
                if all_values:
                    axes[idx].set_ylim(y_limits)

                # Rotate x-axis labels if needed
                if len(melted_df['node'].unique()) > 5:
                    axes[idx].tick_params(axis='x', rotation=45)

                # Add grid for better readability
                axes[idx].grid(True, axis='y', alpha=0.3)

            except Exception as e:
                print(f"Error processing {param_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Adjust layout for all subplots
        plt.tight_layout()

        param_dir = Path("./HP_Model/Final_plots/uncertainty_plot")
        # Save combined plot
        plot_file = param_dir / f"cumulative_producion_uncertainty_{technology}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Created combined capacity boxplots for all parameters")

    except Exception as e:
        print(f"Error in plot creation: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'plt' in globals():
            plt.close()


def plot_production_uncertainty_combined(param_dir, technology="HP_assembly"):
    """
    Create a single violin plot combining capacity-related parameters (CAPEX, OPEX, Capacity) for a specific technology
    All showing capacity values in GW

    Args:
        param_dir (Path): Directory containing the parameter files
        technology (str): Technology to analyze, defaults to "HP_assembly"
    """
    try:
        # Ensure param_dir is a Path object
        param_dir = Path(param_dir)

        # Set up Seaborn style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = [15, 8]
        colors = ["#bfcc67", "#ada599", "#ddd7c6", "#bc556d"]

        # Define files to read with more descriptive titles
        param_files = {
            'Capacity': 'flow_conversion_output_MC_capacity.csv',
            'CAPEX': 'flow_conversion_output_MC_capex.csv',
            'OPEX': 'flow_conversion_output_MC_opex.csv',
            'Combined': 'flow_conversion_output_MC_combined.csv'
        }

        # Create figure with extra space at bottom for legend
        plt.figure(figsize=(15, 9))

        # Create a list to store all processed dataframes
        all_data = []

        # Process each parameter file
        for param_name, filename in param_files.items():
            try:
                # Read data file
                data_file = param_dir / filename
                print(f"Reading file: {data_file.absolute()}")

                if not data_file.exists():
                    print(f"No data file found at: {data_file.absolute()}")
                    continue

                df = pd.read_csv(data_file)

                # Filter for technology
                if 'technology' in df.columns:
                    mask = df['technology'].astype(str) == str(technology)
                    df = df[mask]

                if len(df) == 0:
                    print(f"No data found for technology {technology} in {filename}")
                    continue

                # Get scenario columns
                scenario_cols = [col for col in df.columns if col.startswith('value_')]
                # Calculate cumulative values for each node and scenario
                grouped_df = df.groupby('node')[scenario_cols].sum().reset_index()

                # Melt the dataframe to get all scenario values in one column
                melted_df = pd.melt(
                    grouped_df,
                    id_vars=['node'],
                    value_vars=scenario_cols,
                    var_name='scenario',
                    value_name='value'
                )

                # Convert to numeric, handling any non-numeric values
                melted_df['value'] = pd.to_numeric(melted_df['value'], errors='coerce')

                # Add uncertainty type column
                melted_df['uncertainty_type'] = param_name

                all_data.append(melted_df)

            except Exception as e:
                print(f"Error processing {param_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Create violin plot
        sns.violinplot(
            data=combined_df,
            x='node',
            y='value',
            hue='uncertainty_type',
            width=0.7,
            inner='box',  # Shows box plot inside violin
            density_norm='width',  # Normalize the violin width
            palette=colors,
            linewidth=0.8
        )

        plt.ylabel('Cumulative Heat Pump Production (GW)', size=12)
        plt.xlabel('')

        # Add grid for better readability
        plt.grid(True, axis='y', alpha=0.3)

        # Adjust legend: remove frame, remove title, place at bottom
        plt.legend(
            frameon=False,  # Remove frame
            bbox_to_anchor=(0.5, -0.15),  # Position below plot
            loc='upper center',  # Center horizontally
            ncol=4  # Arrange legend items in one row
        )

        # Adjust layout to make room for legend at bottom
        plt.tight_layout()

        # Save plot
        param_dir = Path(
            "./HP_Model/Final_plots/uncertainty_plot")
        # Save combined plot
        plot_file = param_dir / f"combined_production_uncertainty_{technology}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Created combined production uncertainty plot")

    except Exception as e:
        print(f"Error in plot creation: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'plt' in globals():
            plt.close()


def plot_production_uncertainty_2035(param_dir, technology="HP_assembly"):
    """
    Create boxplots for capacity-related parameters (CAPEX, OPEX, Capacity) for a specific technology
    Shows values at the last timestep in GW

    Args:
        param_dir (Path): Directory containing the parameter files
        technology (str): Technology to analyze, defaults to "HP_assembly"
    """
    try:
        # Ensure param_dir is a Path object
        param_dir = Path(param_dir)

        # Set up Seaborn style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = [15, 8]
        colors = ["#f3ca4f", "#6b5b95", "#d09b9a", "#93a9d2", "#9c7bbc", "#d14e3d", "#f1e8b7", "#698c5a", "#6274a6",
                  "#b5daa4", "#d46e6f", "#e7849c"]
        palette = sns.color_palette(colors, n_colors=20)

        # Define files to read with more descriptive titles
        param_files = {
            'Production with capacity uncertainty': 'flow_conversion_output_MC_capacity.csv',
            'Production with CAPEX uncertainty': 'flow_conversion_output_MC_capex.csv',
            'Production with OPEX uncertainty ': 'flow_conversion_output_MC_opex.csv',
            'Production with combined uncertainty': 'flow_conversion_output_MC_combined.csv'
        }

        # Create figure for plots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()

        # Store all y values to set consistent limits
        all_values = []

        # First pass to collect all y values
        for filename in param_files.values():
            try:
                data_file = param_dir / filename
                if data_file.exists():
                    df = pd.read_csv(data_file)
                    if 'technology' in df.columns:
                        df = df[df['technology'] == str(technology)]

                    # Get the last timestep for each node
                    last_timestep = df.groupby('node')['time_operation'].max().reset_index()
                    df = df.merge(last_timestep, on=['node', 'time_operation'])

                    scenario_cols = [col for col in df.columns if col.startswith('value_')]
                    melted_df = pd.melt(
                        df,
                        id_vars=['node'],
                        value_vars=scenario_cols,
                        var_name='scenario',
                        value_name='value'
                    )
                    melted_df['value'] = pd.to_numeric(melted_df['value'], errors='coerce')
                    all_values.extend(melted_df['value'].dropna())

            except Exception as e:
                print(f"Error in first pass for {filename}: {str(e)}")
                continue

        # Calculate global y limits
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            y_range = y_max - y_min
            y_limits = [y_min - 0.1 * y_range, y_max + 0.1 * y_range]

        # Process each parameter file
        for idx, (param_name, filename) in enumerate(param_files.items()):
            try:
                data_file = param_dir / filename
                print(f"Reading file: {data_file.absolute()}")

                if not data_file.exists():
                    print(f"No data file found at: {data_file.absolute()}")
                    continue

                df = pd.read_csv(data_file)

                if 'technology' in df.columns:
                    mask = df['technology'].astype(str) == str(technology)
                    df = df[mask]

                if len(df) == 0:
                    print(f"No data found for technology {technology} in {filename}")
                    continue

                # Get the last timestep for each node
                last_timestep = df.groupby('node')['time_operation'].max().reset_index()
                df = df.merge(last_timestep, on=['node', 'time_operation'])

                scenario_cols = [col for col in df.columns if col.startswith('value_')]
                melted_df = pd.melt(
                    df,
                    id_vars=['node'],
                    value_vars=scenario_cols,
                    var_name='scenario',
                    value_name='value'
                )

                melted_df['value'] = pd.to_numeric(melted_df['value'], errors='coerce')

                sns.violinplot(
                    data=melted_df,
                    x='node',
                    y='value',
                    ax=axes[idx],
                    width=0.7,
                    inner='box',
                    density_norm='width'
                )

                # Customize plot
                axes[idx].set_title(param_name, pad=20)
                axes[idx].set_xlabel('')
                axes[idx].set_ylabel('Heat Pump Production 2035 [GW]')

                if all_values:
                    axes[idx].set_ylim(y_limits)

                if len(melted_df['node'].unique()) > 5:
                    axes[idx].tick_params(axis='x', rotation=45)

                axes[idx].grid(True, axis='y', alpha=0.3)

            except Exception as e:
                print(f"Error processing {param_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        plt.tight_layout()

        param_dir = Path(
            "./HP_Model/Final_plots/uncertainty_plot")
        plot_file = param_dir / f"last_timestep_uncertainty_{technology}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Created plots for last timestep values")

    except Exception as e:
        print(f"Error in plot creation: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'plt' in globals():
            plt.close()


def plot_total_cost_uncertainty_violins(param_dir_opex, param_dir_capex):
    """
    Create a single combined violin plot for total cost variations across different uncertainty analyses
    showing the distribution of costs at each timestep, using scientific notation.
    """
    try:
        param_dir_opex = Path(param_dir_opex)
        param_dir_capex = Path(param_dir_capex)

        # Set up plot style
        sns.set_style("whitegrid")
        plt.figure(figsize=(15, 8))

        param_files_opex = {
            'Capacity': 'cost_opex_total_MC_capacity.csv',
            'CAPEX': 'cost_opex_total_MC_capex.csv',
            'OPEX': 'cost_opex_total_MC_opex.csv',
            'Combined': 'cost_opex_total_MC_combined.csv'
        }
        param_files_capex = {
            'Capacity': 'cost_capex_total_MC_capacity.csv',
            'CAPEX': 'cost_capex_total_MC_capex.csv',
            'OPEX': 'cost_capex_total_MC_opex.csv',
            'Combined': 'cost_capex_total_MC_combined.csv'
        }

        # Create a DataFrame to store all combined data
        all_data = []
        colors = ["#bfcc67", "#ada599", "#ddd7c6", "#bc556d"]

        for uncertainty_type, filename in param_files_opex.items():
            try:
                # Read both OPEX and CAPEX files
                opex_file = param_dir_opex / filename
                capex_file = param_dir_capex / param_files_capex[uncertainty_type]

                if not opex_file.exists() or not capex_file.exists():
                    print(f"Files not found: {opex_file} or {capex_file}")
                    continue

                df_opex = pd.read_csv(opex_file)
                df_capex = pd.read_csv(capex_file)

                # Get Monte Carlo columns
                mc_cols_opex = [col for col in df_opex.columns if 'value_scenario_MC' in col]
                mc_cols_capex = [col for col in df_capex.columns if 'value_scenario_MC' in col]

                # Combine OPEX and CAPEX data
                for i in range(len(mc_cols_opex)):
                    combined_values = df_opex[mc_cols_opex[i]] + df_capex[mc_cols_capex[i]]
                    for year, value in zip(df_opex['year'], combined_values):
                        all_data.append({
                            'year': year + 2022,
                            'cost': value * 1000,  # Convert to euros
                            'uncertainty_type': uncertainty_type
                        })

            except Exception as e:
                print(f"Error processing {uncertainty_type}: {str(e)}")
                continue

        # Create combined DataFrame
        df_combined = pd.DataFrame(all_data)

        # Create violin plot
        plt.figure(figsize=(15, 8))
        violin_plot = sns.violinplot(
            data=df_combined,
            x='year',
            y='cost',
            hue='uncertainty_type',
            palette=colors,
            split=False,
            inner='box',
            density_norm='width',
            linewidth=0.6
        )

        # Remove only the violin edges but keep the box plots
        for violin in violin_plot.collections:
            if isinstance(violin, matplotlib.collections.PolyCollection):  # This identifies the violin shapes
                violin.set_linewidth(0)  # Remove edges from violins only

        # Customize plot
        plt.xlabel('')
        plt.ylabel('Total Cost (€)', fontsize=14)

        # Format y-axis with scientific notation (10^9)
        plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(9, 9))

        # Set font size for tick labels
        plt.xticks(fontsize=14)  # Set font size for x-axis ticks
        plt.yticks(fontsize=14)  # Set font size for y-axis ticks

        # Customize grid
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.grid(True, which="major", ls="-", alpha=0.5)

        # Add legend below the plot and centered
        legend = plt.legend(
            loc='lower center',  # Center horizontally
            bbox_to_anchor=(0.5, -0.15),  # Fine-tune the position (0.5 = center horizontally, -0.15 = below the plot)
            ncol=4,
            frameon=False, fontsize=14)

        # Save plot
        param_dir = Path(
            "./HP_Model/Final_plots/uncertainty_plot")
        plot_file = param_dir / "total_combined_cost_uncertainty_comparison_violin_combined.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_total_cost_uncertainty_violin(param_dir_opex, param_dir_capex):
    """
    Create violin plots for total cost variations across different uncertainty analyses
    showing the distribution of costs at each timestep, using scientific notation.
    """
    try:
        param_dir_opex = Path(param_dir_opex)
        param_dir_capex = Path(param_dir_capex)
        # Set up plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [15, 20]

        param_files_opex = {
            'Total Cost with Capacity Uncertainty': 'cost_opex_total_MC_capacity.csv',
            'Total Cost with CAPEX Uncertainty': 'cost_opex_total_MC_capex.csv',
            'Total Cost with OPEX Uncertainty': 'cost_opex_total_MC_opex.csv'
        }
        param_files_capex = {
            'Total Cost with Capacity Uncertainty': 'cost_capex_total_MC_capacity.csv',
            'Total Cost with CAPEX Uncertainty': 'cost_capex_total_MC_capex.csv',
            'Total Cost with OPEX Uncertainty': 'cost_capex_total_MC_opex.csv'
        }
        fig, axes = plt.subplots(3, 1, figsize=(15, 20))
        fig.suptitle('Total System Cost Distribution Over Time\nUnder Different Parameter Uncertainties',
                     fontsize=16, y=0.95)

        for idx, param_name in enumerate(param_files_opex.keys()):
            try:
                # Determine file names for both OPEX and CAPEX
                opex_file = param_dir_opex / param_files_opex[param_name]
                capex_file = param_dir_capex / param_files_capex[param_name]

                if not opex_file.exists() or not capex_file.exists():
                    print(f"Files not found: {opex_file} or {capex_file}")
                    continue

                # Read both files
                df_opex = pd.read_csv(opex_file)
                df_capex = pd.read_csv(capex_file)

                # Get Monte Carlo columns
                mc_cols_opex = [col for col in df_opex.columns if 'value_scenario_MC' in col]
                mc_cols_capex = [col for col in df_capex.columns if 'value_scenario_MC' in col]

                sum_dict = {'year': df_opex['year']}
                for i in range(len(mc_cols_opex)):
                    sum_dict[f'value_scenario_MC_{i}'] = df_opex[mc_cols_opex[i]] + df_capex[mc_cols_capex[i]]

                # Create combined DataFrame all at once
                df_combined = pd.DataFrame(sum_dict)

                # Melt the combined dataframe for plotting
                melted_df = pd.melt(
                    df_combined,
                    id_vars=['year'],
                    value_vars=[col for col in df_combined.columns if 'value_scenario_MC' in col],
                    var_name='simulation',
                    value_name='cost'
                )

                # Convert to numeric and convert to euros (multiply by 1000)
                melted_df['cost'] = pd.to_numeric(melted_df['cost'], errors='coerce') * 1000

                # Create violin plot
                sns.violinplot(
                    data=melted_df,
                    x='year',
                    y='cost',
                    ax=axes[idx],
                    inner='box',  # Show box plot inside violin
                    density_norm='width'  # Scale all violins to same width
                )

                # Customize subplot
                axes[idx].set_title(param_name, pad=20)
                axes[idx].set_xlabel('Year')
                axes[idx].set_ylabel('Total Cost (€)')

                # Format y-axis with scientific notation (10^9)
                axes[idx].yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
                axes[idx].ticklabel_format(style='sci', axis='y', scilimits=(9, 9))

                # Grid customization
                axes[idx].grid(True, which="both", ls="-", alpha=0.2)
                axes[idx].grid(True, which="major", ls="-", alpha=0.5)

            except Exception as e:
                print(f"Error processing {param_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Save plot
        param_dir = Path(
            "./HP_Model/Final_plots/uncertainty_plot")
        plot_file = param_dir / "total_combined_cost_uncertainty_comparison_violin.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()