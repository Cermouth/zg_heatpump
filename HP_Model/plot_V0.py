import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

def plot_scenario_results(combined_df, param_dir, parameter, technology = "HP_assembly", carrier = "HP"):
    """
    Create stacked bar plots for flow data by scenario, mode, and region
    """
    try:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        distinct_colors = colors[::2]
        plt.style.use('default')
        sns.set_palette(distinct_colors)

        data_file = param_dir / f"{parameter}_all_scenarios.csv"
        print(f"Looking for data file at: {data_file.absolute()}")

        if not data_file.exists():
            print(f"No data file found at: {data_file.absolute()}")
            return

        df = pd.read_csv(data_file)

        if parameter == "flow_transport":
            plot_transport_flow(df, param_dir)
        return

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

        if parameter == 'capacity':
            time_col = 'year'
            category_cols = ['location', 'capacity_type']
        else:
            time_col = 'time_operation'
            category_cols = ['node']
            if 'carrier' in df.columns:
                category_cols.append('carrier')

        n_scenarios = len(scenario_cols)
        n_cols = min(3, n_scenarios)
        n_rows = (n_scenarios + n_cols - 1) // n_cols

        # Create figure with extra space on right for legend
        fig = plt.figure(figsize=(15 * n_cols / 3 + 3, 8 * n_rows / 2))

        # Store first subplot's lines/artists for legend
        first_pivot_df = None
        legend_handles = None

        for idx, scenario_col in enumerate(scenario_cols, 1):
            try:
                scenario_name = scenario_col.replace('value_', '')

                # Create pivot table
                pivot_df = pd.pivot_table(
                    data=df,
                    index=time_col,
                    columns=category_cols,
                    values=scenario_col,
                    aggfunc='sum',
                    fill_value=0
                )

                if first_pivot_df is None:
                    first_pivot_df = pivot_df

                ax = plt.subplot(n_rows, n_cols, idx)

                if parameter == 'capacity':
                    lines = pivot_df.plot(kind='line', marker='o', ax=ax, legend=False)
                else:
                    lines = pivot_df.plot(kind='bar', stacked=True, ax=ax, legend=False)

                if idx == 1:
                    legend_handles = lines.get_lines() if parameter == 'capacity' else lines.containers

                ax.set_title(f'{scenario_name}')
                ax.set_xlabel('Time' if parameter != 'capacity' else 'Year')
                ax.set_ylabel(parameter.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)

                if len(pivot_df.columns) > 5:
                    plt.xticks(rotation=45)

            except Exception as e:
                print(f"Error creating subplot for scenario {scenario_name}: {str(e)}")
                continue

        # Add single legend to the right of all subplots
        title_entity = technology if 'technology' in df.columns else carrier
        fig.suptitle(f'{parameter} - {title_entity} - All Scenarios',
                     fontsize=16, y=1.02)

        if legend_handles and first_pivot_df is not None:
            if isinstance(first_pivot_df.columns, pd.MultiIndex):
                labels = [' - '.join(col) for col in first_pivot_df.columns]
            edef plot_scenario_results(combined_df, param_dir, parameter, technology = "HP_assembly", carrier = "HP"):
    """
    Create stacked bar plots for flow data by scenario, mode, and region
    """
    try:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        distinct_colors = colors[::2]
        plt.style.use('default')
        sns.set_palette(distinct_colors)

        data_file = param_dir / f"{parameter}_all_scenarios.csv"
        print(f"Looking for data file at: {data_file.absolute()}")

        if not data_file.exists():
            print(f"No data file found at: {data_file.absolute()}")
            return

        df = pd.read_csv(data_file)

        if parameter == "flow_transport":
            plot_transport_flow(df, param_dir)
        return

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

        if parameter == 'capacity':
            time_col = 'year'
            category_cols = ['location', 'capacity_type']
        else:
            time_col = 'time_operation'
            category_cols = ['node']
            if 'carrier' in df.columns:
                category_cols.append('carrier')

        n_scenarios = len(scenario_cols)
        n_cols = min(3, n_scenarios)
        n_rows = (n_scenarios + n_cols - 1) // n_cols

        # Create figure with extra space on right for legend
        fig = plt.figure(figsize=(15 * n_cols / 3 + 3, 8 * n_rows / 2))

        # Store first subplot's lines/artists for legend
        first_pivot_df = None
        legend_handles = None

        for idx, scenario_col in enumerate(scenario_cols, 1):
            try:
                scenario_name = scenario_col.replace('value_', '')

                # Create pivot table
                pivot_df = pd.pivot_table(
                    data=df,
                    index=time_col,
                    columns=category_cols,
                    values=scenario_col,
                    aggfunc='sum',
                    fill_value=0
                )

                if first_pivot_df is None:
                    first_pivot_df = pivot_df

                ax = plt.subplot(n_rows, n_cols, idx)

                if parameter == 'capacity':
                    lines = pivot_df.plot(kind='line', marker='o', ax=ax, legend=False)
                else:
                    lines = pivot_df.plot(kind='bar', stacked=True, ax=ax, legend=False)

                if idx == 1:
                    legend_handles = lines.get_lines() if parameter == 'capacity' else lines.containers

                ax.set_title(f'{scenario_name}')
                ax.set_xlabel('Time' if parameter != 'capacity' else 'Year')
                ax.set_ylabel(parameter.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)

                if len(pivot_df.columns) > 5:
                    plt.xticks(rotation=45)

            except Exception as e:
                print(f"Error creating subplot for scenario {scenario_name}: {str(e)}")
                continue

        # Add single legend to the right of all subplots
        title_entity = technology if 'technology' in df.columns else carrier
        fig.suptitle(f'{parameter} - {title_entity} - All Scenarios',
                     fontsize=16, y=1.02)

        if legend_handles and first_pivot_df is not None:
            if isinstance(first_pivot_df.columns, pd.MultiIndex):
                labels = [' - '.join(col) for col in first_pivot_df.columns]
            else:
                labels = first_pivot_df.columns

            fig.legend(handles=legend_handles,
                       labels=labels,
                       title='-'.join(category_cols),
                       bbox_to_anchor=(1.02, 0.5),
                       loc='center left')

        plt.tight_layout()

        file_prefix = technology if 'technology' in df.columns else carrier
        plot_file = param_dir /lse:
                labels = first_pivot_df.columns

            fig.legend(handles=legend_handles,
                       labels=labels,
                       title='-'.join(category_cols),
                       bbox_to_anchor=(1.02, 0.5),
                       loc='center left')

        plt.tight_layout()

        file_prefix = technology if 'technology' in df.columns else carrier
        plot_file = param_dir / f"{parameter}_{file_prefix}_all_scenarios.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        # Create a combined plot for all scenarios
        # For scenario comparison, use a different color palette
        sns.set_palette(sns.color_palette("Set2", len(scenario_cols)))
        try:
            plt.figure(figsize=(15, 8))

            # Calculate the total per timestep for each scenario
            scenario_totals = pd.DataFrame()
            for scenario_col in scenario_cols:
                scenario_name = scenario_col.replace('value_', '')
                total = df.groupby(time_col)[scenario_col].sum()
                scenario_totals[scenario_name] = total

            # Plot the comparison using a different color palette for scenarios
            scenario_colors = sns.color_palette('Set1', len(scenario_totals.columns))
            for column, color in zip(scenario_totals.columns, scenario_colors):
                plt.plot(scenario_totals.index, scenario_totals[column],
                         marker='o', label=column, linewidth=2, markersize=6,
                         color=color)

            title_entity = technology if 'technology' in df.columns else carrier
            plt.title(f'{parameter} - {title_entity} - All Scenarios Comparison')
            plt.xlabel('Time' if parameter != 'capacity' else 'Year')
            plt.ylabel(parameter.replace('_', ' ').title())
            plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save the comparison plot
            file_prefix = technology if 'technology' in df.columns else carrier
            plot_file = param_dir / f"{parameter}_{file_prefix}_scenario_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Created scenario comparison plot for {parameter} - {title_entity}")

        except Exception as e:
            print(f"Error creating comparison plot: {str(e)}")
            plt.close()

    except Exception as e:
        print(f"Error in plot creation: {str(e)}")
        if 'plt' in globals():
            plt.close()


def plot_transport_flow(df, param_dir, parameter='transport'):
    """
    Create Sankey diagrams for transport flows:
    1. For each technology: grid of scenarios (one Sankey per scenario)
    2. Comparison plot: grid of technologies (one Sankey per technology)

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing transport flow data
    param_dir : str or Path
        Directory path where the output files should be saved
    parameter : str, optional
        Parameter name for the output file (default: 'transport')
    """
    print("Creating transport flow Sankey diagrams")
    try:
        param_dir = Path(param_dir)
        scenario_cols = [col for col in df.columns if col.startswith('value_scenario_')]
        technologies = df['technology'].unique()

        def create_sankey_figure(data, value_col, title):
            """Helper function to create a single Sankey diagram"""
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

            if not sources:  # Skip if no flows
                return None

            node_labels = list(sorted(node_labels))
            node_indices = {node: idx for idx, node in enumerate(node_labels)}

            return {
                'type': 'sankey',
                'node': {
                    'pad': 15,
                    'thickness': 20,
                    'line': {'color': "black", 'width': 0.5},
                    'label': node_labels,
                    'color': "lightblue"
                },
                'link': {
                    'source': [node_indices[s] for s in sources],
                    'target': [node_indices[t] for t in targets],
                    'value': values,
                    'color': "rgba(0,0,255,0.2)"
                }
            }

        # 1. Create plots for each technology with scenarios in grid
        for technology in technologies:
            tech_data = df[df['technology'] == technology]
            last_timestep = tech_data['time_operation'].max()
            last_timestep_data = tech_data[tech_data['time_operation'] == last_timestep]

            # Calculate grid dimensions
            n_scenarios = len(scenario_cols)
            n_rows = math.ceil(math.sqrt(n_scenarios))
            n_cols = math.ceil(n_scenarios / n_rows)

            # Create figure with grid
            fig = go.Figure()

            # Add each scenario as a separate Sankey diagram
            annotations = []
            for i, scenario_col in enumerate(scenario_cols):
                scenario_name = scenario_col.replace('value_scenario_', '') or 'Base'

                # Calculate position in grid
                row = i // n_cols
                col = i % n_cols

                # Calculate domain for this subplot
                x_domain = [col / n_cols, (col + 0.9) / n_cols]
                y_domain = [(n_rows - 1 - row) / n_rows, (n_rows - row - 0.1) / n_rows]

                sankey_data = create_sankey_figure(
                    last_timestep_data,
                    scenario_col,
                    f"Scenario {scenario_name}"
                )

                if sankey_data:
                    sankey_data.update(
                        domain=dict(x=x_domain, y=y_domain),
                    )
                    fig.add_trace(go.Sankey(**sankey_data))

                    # Add annotation for this subplot
                    annotations.append(dict(
                        text=f"Scenario {scenario_name}",
                        x=(x_domain[0] + x_domain[1]) / 2,
                        y=y_domain[1] + 0.02,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=14, weight='bold')
                    ))

            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"{technology} Flows - Last Timestep ({last_timestep})",
                    x=0.5,
                    xanchor='center',
                    font=dict(size=16)
                ),
                width=400 * n_cols,
                height=400 * n_rows + 50,  # Added extra height for titles
                showlegend=False,
                grid=dict(rows=n_rows, columns=n_cols),
                paper_bgcolor='white',
                annotations=annotations,
                margin=dict(t=80, b=20)  # Increased top margin for main title
            )

            # Save figure
            fig.write_image(str(param_dir / f"{parameter}_{technology}_all_scenarios.png"), scale=2)
            print(f"Created Sankey diagrams grid for {technology}")

        # 2. Create comparison plot with all technologies for base scenario
        base_scenario = 'value_scenario_' if 'value_scenario_' in scenario_cols else scenario_cols[0]
        last_timestep = df['time_operation'].max()
        last_timestep_data = df[df['time_operation'] == last_timestep]

        # Calculate grid dimensions for technologies
        n_tech = len(technologies)
        n_rows = math.ceil(math.sqrt(n_tech))
        n_cols = math.ceil(n_tech / n_rows)

        # Create figure with grid
        fig = go.Figure()

        # Add each technology as a separate Sankey diagram
        annotations = []
        for i, tech in enumerate(technologies):
            # Calculate position in grid
            row = i // n_cols
            col = i % n_cols

            # Calculate domain for this subplot
            x_domain = [col / n_cols, (col + 0.9) / n_cols]
            y_domain = [(n_rows - 1 - row) / n_rows, (n_rows - row - 0.1) / n_rows]

            tech_data = last_timestep_data[last_timestep_data['technology'] == tech]
            sankey_data = create_sankey_figure(tech_data, base_scenario, tech)

            if sankey_data:
                sankey_data.update(
                    domain=dict(x=x_domain, y=y_domain),
                )
                fig.add_trace(go.Sankey(**sankey_data))

                # Add annotation for this subplot
                annotations.append(dict(
                    text=tech.replace('_transport', ''),  # Remove '_transport' suffix for cleaner title
                    x=(x_domain[0] + x_domain[1]) / 2,
                    y=y_domain[1] + 0.02,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14, weight='bold')
                ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Transport Flows Comparison - Base Scenario - Last Timestep ({last_timestep})",
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            width=400 * n_cols,
            height=400 * n_rows + 50,  # Added extra height for titles
            showlegend=False,
            grid=dict(rows=n_rows, columns=n_cols),
            paper_bgcolor='white',
            annotations=annotations,
            margin=dict(t=80, b=20)  # Increased top margin for main title
        )

        # Save comparison figure
        fig.write_image(str(param_dir / f"{parameter}_all_technologies_comparison.png"), scale=2)
        print("Created technology comparison Sankey diagrams grid")

    except Exception as e:
        print(f"Error creating Sankey diagrams: {str(e)}")
        raise


def plot_uncertainty_results(param_dir, parameter, technology='HP_assembly'):

    if parameter in "cost_opex_yearly":
        print("Creating cost analysis boxplots")
        uncertainty_cost_plotting()
    if parameter in "flow_transport":
        return

    # Read data
    data_file = param_dir / f"{parameter}_MC.csv"
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
    max_time = df['time_operation'].max()
    df_filtered = df[df['time_operation'] == max_time]

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

    plt.title(f'Uncertainty Analysis for {technology} (t={max_time})')
    plt.xlabel('Node')
    plt.ylabel(parameter)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(param_dir / f"{parameter}_boxplot.png")
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

    return production_df, storage_transport_df, scenario_cols, time_col


def plot_final_timestep_comparison_by_node(df, scenario_cols, time_col, output_dir):
    """
    Creates separate boxplots for each node comparing different technologies at the final timestep,
    with adjusted y-axis scales for better visualization.
    """
    # Get final timestep data
    final_timestep = df[time_col].max()
    final_data = df[df[time_col] == final_timestep]

    # Process each node separately
    for node in final_data['location'].unique():
        node_data = final_data[final_data['location'] == node]

        # Prepare data for plotting
        plot_data = []
        for tech in node_data['technology'].unique():
            tech_data = node_data[node_data['technology'] == tech]
            for scenario in scenario_cols:
                values = tech_data[scenario].values
                plot_data.extend([(tech, val) for val in values if pd.notna(val)])

        if not plot_data:  # Skip if no valid data for this node
            continue

        plot_df = pd.DataFrame(plot_data, columns=['Technology', 'Cost'])

        # Calculate statistics to identify outliers and scale breaks
        tech_stats = plot_df.groupby('Technology')['Cost'].agg(['median', 'std']).reset_index()
        tech_stats['upper'] = tech_stats['median'] + 2 * tech_stats['std']

        # Sort technologies by median cost
        tech_order = tech_stats.sort_values('median')['Technology'].tolist()

        # Determine if we need split visualization
        high_cost_techs = tech_stats[tech_stats['upper'] > tech_stats['upper'].median() * 2]['Technology'].tolist()
        low_cost_techs = [tech for tech in tech_order if tech not in high_cost_techs]

        if high_cost_techs and low_cost_techs:
            # Create figure with two subplots for different scales
            fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[2, 1], figsize=(12, 10))

            # Plot high-cost technologies
            sns.boxplot(data=plot_df[plot_df['Technology'].isin(high_cost_techs)],
                        x='Technology', y='Cost', ax=ax1,
                        order=[t for t in tech_order if t in high_cost_techs])
            ax1.set_title(f'High-Cost Technologies - {node} (Final Timestep)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)

            # Plot low-cost technologies
            sns.boxplot(data=plot_df[plot_df['Technology'].isin(low_cost_techs)],
                        x='Technology', y='Cost', ax=ax2,
                        order=[t for t in tech_order if t in low_cost_techs])
            ax2.set_title(f'Low-Cost Technologies - {node} (Final Timestep)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)

        else:
            # Create single plot if no significant scale difference
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=plot_df, x='Technology', y='Cost', order=tech_order)
            plt.title(f'Technology Cost Comparison - {node} (Final Timestep)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'technology_comparison_{node}_final_timestep.png',
                    bbox_inches='tight', dpi=300)
        plt.close()


def plot_node_comparison(df, scenario_cols, technology, output_dir):
    """
    Creates boxplots comparing costs across different nodes for a specific technology,
    summed over all timesteps.
    """
    tech_data = df[df['technology'] == technology]

    # Calculate sums for each node and scenario
    plot_data = []
    for node in tech_data['location'].unique():
        node_data = tech_data[tech_data['location'] == node]
        for scenario in scenario_cols:
            sum_value = node_data[scenario].sum()
            if pd.notna(sum_value):
                plot_data.append((node, sum_value))

    if not plot_data:  # Skip if no valid data
        return

    plot_df = pd.DataFrame(plot_data, columns=['Node', 'Total Cost'])

    # Sort nodes by median cost
    node_order = plot_df.groupby('Node')['Total Cost'].median().sort_values().index

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=plot_df, x='Node', y='Total Cost', order=node_order)
    plt.title(f'Cost Comparison Across Nodes - {technology}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / f'node_comparison_{technology}.png', bbox_inches='tight', dpi=300)
    plt.close()


def uncertainty_cost_plotting():
    """
    Main function to run the analysis.
    """
    file_path = "./HP_Model/uncertainty_results/cost_opex_yearly/cost_opex_yearly_MC.csv"
    try:
        output_dir = Path(file_path).parent / 'plots'
        output_dir.mkdir(exist_ok=True)

        # Create subdirectories for different analysis types
        production_dir = output_dir / 'production'
        storage_transport_dir = output_dir / 'storage_transport'
        production_dir.mkdir(exist_ok=True)
        storage_transport_dir.mkdir(exist_ok=True)

        # Load and prepare data
        production_df, storage_transport_df, scenario_cols, time_col = load_and_prepare_data(file_path)

        # Process production technologies
        print("Processing production technologies...")
        plot_final_timestep_comparison_by_node(production_df, scenario_cols, time_col, production_dir)

        for tech in production_df['technology'].unique():
            plot_node_comparison(production_df, scenario_cols, tech, production_dir)

        # Save separated datasets for later use
        production_df.to_csv(output_dir / 'production_data.csv', index=False)
        storage_transport_df.to_csv(output_dir / 'storage_transport_data.csv', index=False)

        print("Analysis completed successfully!")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        if 'plt' in globals():
            plt.close()
