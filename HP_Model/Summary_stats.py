import pandas as pd
import os
from pathlib import Path

import pandas as pd
from pathlib import Path

import pandas as pd
from pathlib import Path


def analyze_european_production_share(production_path, demand_path, output_dir):
    """
    Analyze the share of European production compared to European demand across technologies.
    Creates three separate files - one for each technology containing all scenario results.

    Parameters:
    production_path (str): Path to the CSV file containing production data
    demand_path (str): Path to the CSV file containing demand data
    output_dir (Path): Directory to save output files
    """
    # Read the data
    prod_df = pd.read_csv(production_path)
    demand_df = pd.read_csv(demand_path)

    # Scale demand values (divide by 10^6)
    demand_columns = [col for col in demand_df.columns if col != 'year']
    demand_df[demand_columns] = demand_df[demand_columns] / 1_000_000

    # Define European nodes
    european_nodes = ['AUT', 'CZE', 'DEU', 'ITA', 'ROE']

    # Define technologies to analyze
    technologies = ['HP_assembly', 'Compressor_manufacturing', 'HEX_manufacturing']

    # Get scenario columns from production data
    scenario_cols = [col for col in prod_df.columns if col.startswith('value_scenario_')]

    # Calculate total European demand for each year
    european_demand = demand_df[european_nodes].sum(axis=1)

    # Initialize dictionaries to store results for each technology
    tech_results = {tech: [] for tech in technologies}

    # Process each technology
    for tech in technologies:
        # Filter for specific technology and European nodes
        euro_prod = prod_df[
            (prod_df['technology'] == tech) &
            (prod_df['node'].isin(european_nodes))
            ]

        # Process each year
        for year_idx in range(len(demand_df)):
            year = 2022 + year_idx
            demand = european_demand[year_idx]

            # Get production for each scenario
            scenario_data = {'time_operation': year, 'demand': demand}

            for scenario in scenario_cols:
                scenario_name = scenario.replace('value_scenario_', '')
                if not scenario_name:
                    scenario_name = 'base'

                # Calculate total production for this scenario
                production = euro_prod[euro_prod['time_operation'] == year_idx][scenario].sum()

                # Calculate share
                share = (production / demand * 100).round(2)

                # Add to scenario data
                scenario_data[f'production_{scenario_name}'] = production
                scenario_data[f'share_{scenario_name}'] = share

            tech_results[tech].append(scenario_data)

    # Convert results to DataFrames and save
    for tech in technologies:
        df = pd.DataFrame(tech_results[tech])

        # Reorder columns to group production and share metrics
        production_cols = [col for col in df.columns if col.startswith('production_')]
        share_cols = [col for col in df.columns if col.startswith('share_')]

        # Create final column order
        cols_order = ['time_operation', 'demand'] + production_cols + share_cols
        df = df[cols_order]

        # Save to CSV
        output_path = output_dir / f'{tech.lower()}_european_analysis.csv'
        df.to_csv(output_path, index=False)

        # Store in results dictionary
        tech_results[tech] = df

    print(f"Technology-specific analysis files saved to: {output_dir}")

    return tech_results


def analyze_scenarios(data_path, output_dir):
    """
    Analyze production for each carrier/technology across nodes, showing each node's
    contribution to total technology production.

    Parameters:
    data_path (str): Path to the CSV file containing scenario data
    output_dir (Path): Directory to save output files
    """
    # Read the data
    df = pd.read_csv(data_path)

    # Get base scenario column
    base_scenario = 'value_scenario_'

    # 1. Carrier Summary (Base Scenario Only)
    # Group by carrier and time, summing across nodes
    summary = df.groupby(['carrier', 'time_operation'])[base_scenario].sum().reset_index()

    # Pivot to get time periods as columns
    pivot_df = summary.pivot(
        index='carrier',
        columns='time_operation',
        values=base_scenario
    )

    # Rename columns to include scenario and time
    pivot_df.columns = [f'{base_scenario}_time_{col}' for col in pivot_df.columns]
    carrier_summary = pivot_df

    # Add total across all times for each carrier
    carrier_summary['total'] = carrier_summary.sum(axis=1)

    # Calculate percentage of total production and round to whole numbers
    carrier_summary['percentage_of_total'] = (carrier_summary['total'] / carrier_summary['total'].sum() * 100).round(0)

    # Round all numeric columns to whole numbers
    carrier_summary = carrier_summary.round(0)

    # Sort by total production (descending)
    carrier_summary = carrier_summary.sort_values('total', ascending=False)

    # 2. Node Technology Summary (Base Scenario Only)
    # Sum over all timesteps for each node-carrier combination
    node_tech_summary = df.groupby(['node', 'carrier'])[base_scenario].sum().reset_index()

    # Pivot to get nodes as columns and technologies as rows
    node_summary = node_tech_summary.pivot(
        index='carrier',
        columns='node',
        values=base_scenario
    ).fillna(0)

    # Calculate total production for each technology (row sum)
    node_summary['total_tech'] = node_summary.sum(axis=1)

    # Calculate percentage contribution of each node to total technology production
    percentage_summary = pd.DataFrame()
    for node in node_summary.columns:
        if node != 'total_tech':  # Skip the total column
            percentage_summary[f'{node}_percentage'] = (node_summary[node] / node_summary['total_tech'] * 100).round(0)

    # Sort by total technology production
    node_summary = node_summary.sort_values('total_tech', ascending=False)
    percentage_summary = percentage_summary.reindex(node_summary.index)

    # Round all values
    node_summary = node_summary.round(0)

    # 3. HP Assembly Analysis (All Scenarios)
    # Filter for HP_assembly technology
    hp_data = df[df['technology'] == 'HP_assembly']

    # Get all scenario columns
    scenario_cols = [col for col in df.columns if col.startswith('value_scenario_')]

    hp_summary = pd.DataFrame()

    for scenario in scenario_cols:
        # Sum over all timesteps for each node
        hp_by_node = hp_data.groupby('node')[scenario].sum()

        # Calculate percentage contribution for this scenario
        scenario_total = hp_by_node.sum()
        if scenario_total > 0:  # Avoid division by zero
            hp_percentage = (hp_by_node / scenario_total * 100).round(0)
            hp_summary[f'{scenario}_percentage'] = hp_percentage
            hp_summary[f'{scenario}_total'] = hp_by_node.round(0)

    # Sort by base scenario percentage (if it exists)
    if f'value_scenario_base_percentage' in hp_summary.columns:
        hp_summary = hp_summary.sort_values(f'value_scenario_base_percentage', ascending=False)

    # 4. HP Manufacturing Analysis Across Scenarios
    # Filter for HP_manufacturing technology
    hp_mfg_data = df[df['technology'] == 'HP_assembly'].copy()

    # 4.1 Detailed timestep analysis for all scenarios
    # Initialize list to store DataFrames for each scenario
    restructured_data = []

    # Get unique timesteps and nodes
    timesteps = hp_mfg_data['time_operation'].unique()
    nodes = hp_mfg_data['node'].unique()

    for timestep in timesteps:
        for node in nodes:
            row_data = {
                'time_operation': timestep,
                'node': node
            }

            # Add data for each scenario
            for scenario in scenario_cols:
                scenario_name = 'base' if scenario == 'value_scenario_' else scenario.replace('value_scenario_', '')

                # Get data for this specific combination
                scenario_data = hp_mfg_data[
                    (hp_mfg_data['time_operation'] == timestep) &
                    (hp_mfg_data['node'] == node)
                    ]

                if not scenario_data.empty:
                    value = scenario_data[scenario].iloc[0]
                    total_timestep = hp_mfg_data[
                        hp_mfg_data['time_operation'] == timestep
                        ][scenario].sum()

                    percentage = (value / total_timestep * 100) if total_timestep > 0 else 0

                    row_data[f'absolute_value_{scenario_name}'] = round(value, 1)
                    row_data[f'percentage_{scenario_name}'] = round(percentage, 1)
                else:
                    row_data[f'absolute_value_{scenario_name}'] = 0
                    row_data[f'percentage_{scenario_name}'] = 0

            restructured_data.append(row_data)

    # Create the restructured DataFrame
    detailed_analysis = pd.DataFrame(restructured_data)

    # Sort by timestep and node
    detailed_analysis = detailed_analysis.sort_values(['time_operation', 'node'])


    # 4.2 Scenario comparison summary
    summary_dfs = []

    for scenario in scenario_cols:
        # Calculate total production by node for this scenario
        scenario_summary = hp_mfg_data.groupby('node')[scenario].sum().reset_index()

        # Calculate total production for this scenario
        total_production = scenario_summary[scenario].sum()

        # Calculate percentage
        scenario_summary['percentage'] = (scenario_summary[scenario] / total_production * 100).round(1)

        # Add scenario name
        scenario_name = 'base' if scenario == 'value_scenario_' else scenario.replace('value_scenario_', '')
        scenario_summary['scenario'] = scenario_name

        # Rename columns
        scenario_summary.rename(columns={scenario: 'total_production'}, inplace=True)

        summary_dfs.append(scenario_summary)

    # Combine all scenario summaries
    scenario_comparison = pd.concat(summary_dfs, ignore_index=True)

    # Create two separate pivot tables for clarity
    production_pivot = pd.pivot_table(
        scenario_comparison,
        index='node',
        columns='scenario',
        values='total_production'
    ).round(1)

    percentage_pivot = pd.pivot_table(
        scenario_comparison,
        index='node',
        columns='scenario',
        values='percentage'
    ).round(1)

    # Add summary statistics to percentage pivot
    percentage_pivot['min_percentage'] = percentage_pivot.min(axis=1).round(1)
    percentage_pivot['max_percentage'] = percentage_pivot.max(axis=1).round(1)
    percentage_pivot['avg_percentage'] = percentage_pivot.mean(axis=1).round(1)

    # Save the new consolidated files
    detailed_analysis.to_csv(output_dir / 'hp_manufacturing_detailed_analysis.csv', index=False)
    production_pivot.to_csv(output_dir / 'hp_manufacturing_production_by_scenario.csv')
    percentage_pivot.to_csv(output_dir / 'hp_manufacturing_percentages_by_scenario.csv')

    carrier_summary.to_csv(output_dir / 'carrier_production_by_time.csv')
    node_summary.to_csv(output_dir / 'node_technology_production.csv')
    percentage_summary.to_csv(output_dir / 'node_technology_percentages.csv')
    hp_summary.to_csv(output_dir / 'hp_assembly_by_node.csv')


    print(f"Analysis results saved to: {output_dir}")


def analyze_capacity_stats(data_path, output_dir):
    """
    Analyze capacity statistics for specific technologies, producing a single consolidated CSV
    with CAGR, initial/final values, and absolute increase for each technology and node.

    Parameters:
    data_path (str): Path to the CSV file containing scenario data
    output_dir (Path): Directory to save output files

    Returns:
    dict: Dictionary containing the final analysis results
    """
    # Read the data
    df = pd.read_csv(data_path)
    output_dir = Path(output_dir)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter for relevant technologies and base scenario
    technologies = ["HP_assembly", "HEX_manufacturing", "Compressor_manufacturing"]
    base_scenario = 'value_scenario_'
    filtered_df = df[df['technology'].isin(technologies)]

    # Initialize list for results
    results_data = []

    # Calculate metrics for each technology and node
    for tech in technologies:
        tech_data = filtered_df[filtered_df['technology'] == tech]

        for node in tech_data['location'].unique():
            node_data = tech_data[tech_data['location'] == node]

            # Calculate initial and final values
            initial_value = node_data[node_data['year'] == 0][base_scenario].sum()
            final_value = node_data[node_data['year'] == 13][base_scenario].sum()

            # Calculate CAGR
            if initial_value > 0 and final_value > 0:
                cagr = (((final_value / initial_value) ** (1 / 13)) - 1) * 100
            else:
                cagr = 0

            # Calculate absolute increase
            absolute_increase = final_value - initial_value

            # Store all metrics
            results_data.append({
                'technology': tech,
                'node': node,
                'initial_value': round(initial_value, 2),
                'final_value': round(final_value, 2),
                'absolute_increase': round(absolute_increase, 2),
                'CAGR_percent': round(cagr, 2)
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results_data)

    # Sort by technology and absolute increase
    results_df = results_df.sort_values(['technology', 'absolute_increase'], ascending=[True, False])

    # Save single consolidated CSV file
    results_df.to_csv(output_dir / 'capacity_analysis_final.csv', index=False)

    print(f"Final capacity analysis saved to: {output_dir}/capacity_analysis_final.csv")



def analyze_costs(opex_path, capex_path, output_dir):
    """
    Analyze OPEX and CAPEX data from separate files for each process across timesteps,
    showing percentage contribution of each process and cost increases over time.

    Now includes scenario comparison analysis.

    Parameters:
    opex_path (str): Path to the CSV file containing OPEX data
    capex_path (str): Path to the CSV file containing CAPEX data
    output_dir (Path): Directory to save output files

    Returns:
    dict: Dictionary containing the cost analysis results
    """
    # Initialize results dictionary
    results = {}

    # Process both OPEX and CAPEX files
    for cost_type, file_path in [('opex', opex_path), ('capex', capex_path)]:
        # Read the data
        df = pd.read_csv(file_path)

        # Get all scenario columns (not just the base scenario)
        scenario_cols = [col for col in df.columns if col.startswith('value_scenario_')]

        # If no scenario prefix is found, assume base scenario is the only one
        if len(scenario_cols) == 0:
            scenario_cols = ['value_scenario_']

        #########################################
        # Original code for base scenario analysis
        #########################################
        # Get base scenario column
        base_scenario = 'value_scenario_'

        # 1. Process Contribution Analysis
        # Sum costs across all nodes for each process and timestep
        process_time_summary = df.groupby(['technology', 'year'])[base_scenario].sum().reset_index()

        # Calculate total cost for each timestep
        total_by_time = process_time_summary.groupby('year')[base_scenario].sum()

        # Pivot to get processes as rows and timesteps as columns
        pivot_df = process_time_summary.pivot(
            index='technology',
            columns='year',
            values=base_scenario
        ).fillna(0)

        # Calculate percentages
        percentage_df = pivot_df.copy()
        for col in percentage_df.columns:
            percentage_df[col] = (pivot_df[col] / total_by_time[col] * 100).round(1)

        # Original code continues...
        # 2. Cost Increase Analysis
        # ... (keeping this part unchanged)

        # Calculate absolute and percentage increase over time for each process
        cost_increase = pd.DataFrame()

        for process in pivot_df.index:
            process_data = pivot_df.loc[process]

            # Get first and last non-zero values
            initial_value = process_data.iloc[0]
            final_value = process_data.iloc[-1]

            # Calculate absolute increase
            absolute_increase = final_value - initial_value

            # Calculate percentage increase
            if initial_value != 0:
                percentage_increase = ((final_value / initial_value) - 1) * 100
            else:
                percentage_increase = float('inf') if final_value > 0 else 0

            # Calculate CAGR
            if initial_value > 0 and final_value > 0:
                num_periods = len(process_data) - 1
                cagr = (((final_value / initial_value) ** (1 / num_periods)) - 1) * 100
            else:
                cagr = 0

            cost_increase.loc[process, 'initial_value'] = round(initial_value, 2)
            cost_increase.loc[process, 'final_value'] = round(final_value, 2)
            cost_increase.loc[process, 'absolute_increase'] = round(absolute_increase, 2)
            cost_increase.loc[process, 'percentage_increase'] = round(percentage_increase, 1)
            cost_increase.loc[process, 'CAGR_percent'] = round(cagr, 1)

        # Sort by absolute increase
        cost_increase = cost_increase.sort_values('absolute_increase', ascending=False)

        #########################################
        # NEW CODE for scenario comparison
        #########################################

        # Create a DataFrame to store average process contributions across scenarios
        scenario_comparison = pd.DataFrame(index=df['technology'].unique())

        # Process each scenario
        for scenario in scenario_cols:
            # Extract scenario name from column name
            scenario_name = scenario.replace('value_scenario_', '')
            if scenario_name == '':
                scenario_name = 'base'

            # Sum costs across all nodes for each process and timestep
            scenario_process_summary = df.groupby(['technology', 'year'])[scenario].sum().reset_index()

            # Calculate total cost for each timestep in this scenario
            scenario_total_by_time = scenario_process_summary.groupby('year')[scenario].sum()

            # Calculate percentage contribution for each process and timestep
            scenario_percentages = []

            for process in df['technology'].unique():
                process_data = scenario_process_summary[scenario_process_summary['technology'] == process]

                # Calculate percentage for each timestep
                process_percentages = []
                for year in process_data['year'].unique():
                    year_value = process_data[process_data['year'] == year][scenario].values[0]
                    year_total = scenario_total_by_time[year]
                    if year_total > 0:
                        percentage = (year_value / year_total) * 100
                    else:
                        percentage = 0
                    process_percentages.append(percentage)

                # Calculate average percentage across all timesteps
                avg_percentage = sum(process_percentages) / len(process_percentages) if process_percentages else 0
                scenario_percentages.append((process, avg_percentage))

            # Add to comparison DataFrame
            for process, avg_pct in scenario_percentages:
                scenario_comparison.loc[process, f'avg_contribution_{scenario_name}'] = round(avg_pct, 1)

        # Sort by base scenario contribution (if available)
        if 'avg_contribution_base' in scenario_comparison.columns:
            scenario_comparison = scenario_comparison.sort_values('avg_contribution_base', ascending=False)

        # Store results
        results[f'{cost_type}_contribution'] = percentage_df
        results[f'{cost_type}_increases'] = cost_increase
        results[f'{cost_type}_scenario_comparison'] = scenario_comparison

        # Save to CSV
        percentage_df.to_csv(output_dir / f'{cost_type}_process_contribution.csv')
        cost_increase.to_csv(output_dir / f'{cost_type}_process_increases.csv')
        scenario_comparison.to_csv(output_dir / f'{cost_type}_scenario_comparison.csv')

    print(f"Cost analysis results saved to: {output_dir}")

    return results


def analyze_monte_carlo_results_production(param_dir, param_files, output_dir):
    """
    Analyze Monte Carlo simulation results for heat pump production across different nodes.

    Parameters:
    param_dir (str): Directory containing Monte Carlo simulation files
    param_files (dict): Dictionary mapping parameter types to filenames
    output_dir (Path): Directory to save output files
    """
    results_by_param = {}

    # Process each parameter file
    for param_type, filename in param_files.items():
        file_path = os.path.join(param_dir, filename)
        df = pd.read_csv(file_path)

        # Filter for HP_assembly technology
        hp_data = df[df['technology'] == 'HP_assembly'].copy()

        # Get scenario columns (assuming they start with 'value_scenario_')
        scenario_cols = [col for col in df.columns if col.startswith('value_scenario_')]

        # Group by node and sum across years for each scenario
        node_results = hp_data.groupby('node')[scenario_cols].sum()

        # Calculate statistics for each node
        stats = pd.DataFrame()
        for node in node_results.index:
            node_data = node_results.loc[node]

            stats.loc[node, f'mean_{param_type}'] = node_data.mean()
            stats.loc[node, f'std_{param_type}'] = node_data.std()
            stats.loc[node, f'cv_{param_type}'] = (node_data.std() / node_data.mean()) * 100
            stats.loc[node, f'q05_{param_type}'] = node_data.quantile(0.05)
            stats.loc[node, f'q95_{param_type}'] = node_data.quantile(0.95)
            stats.loc[node, f'min_{param_type}'] = node_data.min()
            stats.loc[node, f'max_{param_type}'] = node_data.max()
            stats.loc[node, f'range_{param_type}'] = node_data.max() - node_data.min()
            stats.loc[node, f'skewness_{param_type}'] = node_data.skew()
            stats.loc[node, f'kurtosis_{param_type}'] = node_data.kurtosis()

        results_by_param[param_type] = stats

    # Combine results from all parameter types
    final_stats = pd.concat(results_by_param.values(), axis=1)

    # Reorder columns by metric type
    metrics = ['mean', 'std', 'cv', 'q05', 'q95', 'min', 'max', 'range', 'skewness', 'kurtosis']
    params = list(param_files.keys())

    # Create new column order
    new_cols = []
    for metric in metrics:
        metric_cols = [col for col in final_stats.columns if col.startswith(f'{metric}_')]
        new_cols.extend(sorted(metric_cols))

    # Reorder columns
    final_stats = final_stats[new_cols]

    # Sort by mean production (using Combined parameter type if available)
    if 'mean_Combined' in final_stats.columns:
        final_stats = final_stats.sort_values('mean_Combined', ascending=False)

    # Save results
    output_path = output_dir / 'monte_carlo_hp_production_stats.csv'
    final_stats.to_csv(output_path)
    print(f"Monte Carlo analysis results saved to: {output_path}")

    return final_stats


def analyze_monte_carlo_results_temporal_costs(param_dir_opex, param_dir_capex, param_files_opex, param_files_capex, output_dir):
    """
    Analyze temporal statistics for combined OPEX and CAPEX from Monte Carlo simulations.
    """
    results_by_param = {}

    for param_type in param_files_opex.keys():
        # Read files
        opex_path = os.path.join(param_dir_opex, param_files_opex[param_type])
        capex_path = os.path.join(param_dir_capex, param_files_capex[param_type])

        df_opex = pd.read_csv(opex_path)
        df_capex = pd.read_csv(capex_path)

        # Get Monte Carlo scenario columns
        scenario_cols = [col for col in df_opex.columns if 'value_scenario_MC' in col]

        # Calculate statistics for each year
        stats = pd.DataFrame()
        for year in df_opex['year'].unique():
            # Combine OPEX and CAPEX for this year
            opex_data = df_opex[df_opex['year'] == year][scenario_cols].sum()
            capex_data = df_capex[df_capex['year'] == year][scenario_cols].sum()
            total_cost = opex_data + capex_data

            stats.loc[year, f'mean_{param_type}'] = total_cost.mean() * 1000
            stats.loc[year, f'std_{param_type}'] = total_cost.std() * 1000
            stats.loc[year, f'cv_{param_type}'] = (total_cost.std() / total_cost.mean()) * 100
            stats.loc[year, f'q05_{param_type}'] = total_cost.quantile(0.05) * 1000
            stats.loc[year, f'q95_{param_type}'] = total_cost.quantile(0.95) * 1000
            stats.loc[year, f'min_{param_type}'] = total_cost.min() * 1000
            stats.loc[year, f'max_{param_type}'] = total_cost.max() * 1000
            stats.loc[year, f'range_{param_type}'] = (total_cost.max() - total_cost.min()) * 1000
            stats.loc[year, f'skewness_{param_type}'] = total_cost.skew()
            stats.loc[year, f'kurtosis_{param_type}'] = total_cost.kurtosis()

        results_by_param[param_type] = stats

    # Combine results for all parameter types
    final_stats = pd.concat(results_by_param.values(), axis=1)

    # Reorder columns by metric type
    metrics = ['mean', 'std', 'cv', 'q05', 'q95', 'min', 'max', 'range', 'skewness', 'kurtosis']
    new_cols = []
    for metric in metrics:
        metric_cols = [col for col in final_stats.columns if col.startswith(f'{metric}_')]
        new_cols.extend(sorted(metric_cols))

    final_stats = final_stats[new_cols]

    # Add year as column
    final_stats['year'] = final_stats.index + 2022

    # Save results
    output_path = output_dir / 'monte_carlo_total_cost_temporal_stats.csv'
    final_stats.to_csv(output_path, index=False)
    print(f"Total cost temporal analysis saved to: {output_path}")

    return final_stats



production_MC_dir = "./HP_Model/uncertainty_results/flow_conversion_output"
opex_MC_dir = "./HP_Model/uncertainty_results/cost_opex_total"
capex_MC_dir = "./HP_Model/uncertainty_results/cost_capex_total"
prod_path = './HP_Model/parameter_results/flow_conversion_output/flow_conversion_output_scenarios.csv'
capac_path = "./HP_Model/parameter_results/capacity/capacity_scenarios.csv"
opex_yearly_file = "./HP_Model/parameter_results/cost_opex_yearly/cost_opex_yearly_scenarios.csv"
capex_yearly_file = "./HP_Model/parameter_results/cost_capex/cost_capex_scenarios.csv"
output_dir = Path(
    "./HP_Model/Summary Stats")
# Add this line after defining output_dir
output_dir.mkdir(parents=True, exist_ok=True)

analyze_scenarios(prod_path, output_dir)
results = analyze_capacity_stats(capac_path,output_dir)
results = analyze_costs(opex_yearly_file, capex_yearly_file, output_dir)

results = analyze_european_production_share(
    production_path=prod_path,
    demand_path='./ZEN-Model_HP/set_carriers/HP/demand_yearly_variation.csv',
    output_dir=output_dir
)

param_files = {
    'Capacity': 'flow_conversion_output_MC_capacity.csv',
    'CAPEX': 'flow_conversion_output_MC_capex.csv',
    'OPEX': 'flow_conversion_output_MC_opex.csv',
    'Combined': 'flow_conversion_output_MC_combined.csv'
}
#monte_carlo_stats = analyze_monte_carlo_results_production(production_MC_dir, param_files, output_dir)

# param_files_opex = {
#             'Capacity': 'cost_opex_total_MC_capacity.csv',
#             'CAPEX': 'cost_opex_total_MC_capex.csv',
#             'OPEX': 'cost_opex_total_MC_opex.csv',
#             'Combined': 'cost_opex_total_MC_combined.csv'
#         }
# param_files_capex = {
#     'Capacity': 'cost_capex_total_MC_capacity.csv',
#     'CAPEX': 'cost_capex_total_MC_capex.csv',
#     'OPEX': 'cost_capex_total_MC_opex.csv',
#     'Combined': 'cost_capex_total_MC_combined.csv'
# }
# temporal_stats = analyze_monte_carlo_results_temporal_costs(
#     opex_MC_dir,
#     capex_MC_dir,
#     param_files_opex,
#     param_files_capex,
#     output_dir
# )
