import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# Set up the plotting style
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


# ====================================
# UTILITY FUNCTIONS
# ====================================

def map_to_region(node):
    """Map individual nodes to regions following the project's logic"""
    european_countries = ['CZE', 'AUT', 'ITA', 'DEU', 'ROE', 'EUR']

    if node in european_countries:
        return 'EUR'
    elif node in ['CHN', 'USA']:
        return node
    else:
        return 'ROW'


def calculate_self_sufficiency(production, demand):
    """Calculate self-sufficiency ratio (production / demand)"""
    if demand == 0:
        return 0
    return min(production / demand, 1.0)  # Cap at 100%


# ====================================
# DATA LOADING FUNCTIONS
# ====================================

def load_production_data(flow_file):
    """Load production data from flow file - keep all technologies"""
    flow_df = pd.read_csv(flow_file)

    # Apply region mapping
    flow_df['region'] = flow_df['node'].apply(map_to_region)

    # Convert time to years
    if 'time_operation' in flow_df.columns:
        flow_df['year'] = flow_df['time_operation'] + 2022

    return flow_df


def load_demand_data(demand_file):
    """Load demand data"""
    demand_df = pd.read_csv(demand_file)

    # Process year column
    if 'year' not in demand_df.columns:
        first_col = demand_df.columns[0]
        if demand_df[first_col].dtype in ['int64', 'float64']:
            demand_df = demand_df.rename(columns={first_col: 'year'})

    if demand_df['year'].max() < 2000:
        demand_df['year'] = demand_df['year'] + 2022

    # Convert to long format
    country_cols = [col for col in demand_df.columns if col != 'year']

    demand_long = pd.melt(
        demand_df,
        id_vars=['year'],
        value_vars=country_cols,
        var_name='node',
        value_name='demand'
    )

    demand_long['demand'] = pd.to_numeric(demand_long['demand'], errors='coerce').fillna(0)
    demand_long['demand'] = demand_long['demand'] / 1000000  # Convert to GW
    demand_long['region'] = demand_long['node'].apply(map_to_region)

    return demand_long


def load_cost_data(parameter_results_dir='parameter_results'):
    """Load CAPEX and OPEX data - keep all technologies"""
    param_dir = Path(parameter_results_dir)

    # Load CAPEX
    capex_path = param_dir / 'cost_capex' / 'cost_capex_scenarios.csv'
    capex_df = pd.read_csv(capex_path) if capex_path.exists() else None

    # Load OPEX
    opex_path = param_dir / 'cost_opex_yearly' / 'cost_opex_yearly_scenarios.csv'
    opex_df = pd.read_csv(opex_path) if opex_path.exists() else None

    return capex_df, opex_df


def process_costs_by_scenario_tech_year(capex_df, opex_df, scenario_col, target_years):
    """Process costs for a specific scenario by technology, region, and year"""
    costs_by_region_tech_year = {}

    # Process CAPEX - cumulative up to each year
    if capex_df is not None:
        capex_filtered = capex_df.copy()
        if 'location' in capex_filtered.columns:
            # Filter out edges
            capex_filtered = capex_filtered[~capex_filtered['location'].str.contains('-', na=False)].copy()
            capex_filtered['region'] = capex_filtered['location'].apply(map_to_region)

            # Ensure year column exists
            if 'year' in capex_filtered.columns:
                if capex_filtered['year'].max() < 2000:
                    capex_filtered['year'] = capex_filtered['year'] + 2022

                # For each target year, sum cumulative costs up to that year
                for region in ['CHN', 'EUR', 'USA', 'ROW']:
                    region_data = capex_filtered[capex_filtered['region'] == region]

                    for tech in region_data['technology'].unique():
                        tech_data = region_data[region_data['technology'] == tech]

                        for target_year in target_years:
                            # Cumulative CAPEX up to target year
                            cumulative_capex = tech_data[tech_data['year'] <= target_year][scenario_col].sum()

                            key = (region, tech, target_year)
                            if key not in costs_by_region_tech_year:
                                costs_by_region_tech_year[key] = {'capex': 0, 'production': 0, 'trade': 0}
                            costs_by_region_tech_year[key]['capex'] = cumulative_capex

    # Process OPEX - cumulative up to each year
    if opex_df is not None:
        production_df = opex_df[~opex_df['technology'].str.endswith('_transport', na=False)].copy()
        if not production_df.empty and 'location' in production_df.columns:
            production_df = production_df[~production_df['location'].str.contains('-', na=False)].copy()
            production_df['region'] = production_df['location'].apply(map_to_region)

            # Ensure year column exists
            if 'year' in production_df.columns:
                if production_df['year'].max() < 2000:
                    production_df['year'] = production_df['year'] + 2022

                for region in ['CHN', 'EUR', 'USA', 'ROW']:
                    region_data = production_df[production_df['region'] == region]

                    for tech in region_data['technology'].unique():
                        tech_data = region_data[region_data['technology'] == tech]

                        for target_year in target_years:
                            # Cumulative OPEX up to target year
                            cumulative_opex = tech_data[tech_data['year'] <= target_year][scenario_col].sum()

                            key = (region, tech, target_year)
                            if key not in costs_by_region_tech_year:
                                costs_by_region_tech_year[key] = {'capex': 0, 'production': 0, 'trade': 0}
                            costs_by_region_tech_year[key]['production'] = cumulative_opex

        # Trade costs - CORRECTED LOGIC
        transport_to_base_tech_map = {
            'HP_transport': 'HP_assembly',
            'HEX_transport': 'HEX_manufacturing',
            'Compressor_transport': 'Compressor_manufacturing'
        }

        # Filter for *only* these specific transport technologies
        transport_df = opex_df[opex_df['technology'].isin(transport_to_base_tech_map.keys())].copy()

        if not transport_df.empty and 'location' in transport_df.columns:
            valid_transport = transport_df[transport_df['location'].str.contains('-', na=False)].copy()
            if not valid_transport.empty:
                valid_transport['to_node'] = valid_transport['location'].str.split('-').str[1]
                valid_transport['region'] = valid_transport['to_node'].apply(map_to_region)

                # Use the map to create the 'base_tech' column
                valid_transport['base_tech'] = valid_transport['technology'].map(transport_to_base_tech_map)

                if 'year' in valid_transport.columns:
                    if valid_transport['year'].max() < 2000:
                        valid_transport['year'] = valid_transport['year'] + 2022

                    for region in ['CHN', 'EUR', 'USA', 'ROW']:
                        region_data = valid_transport[valid_transport['region'] == region]

                        for tech in region_data['base_tech'].unique():
                            tech_data = region_data[region_data['base_tech'] == tech]

                            for target_year in target_years:
                                cumulative_trade = tech_data[tech_data['year'] <= target_year][scenario_col].sum()

                                key = (region, tech, target_year)
                                if key not in costs_by_region_tech_year:
                                    costs_by_region_tech_year[key] = {'capex': 0, 'production': 0, 'trade': 0}
                                costs_by_region_tech_year[key]['trade'] = cumulative_trade

    return costs_by_region_tech_year


def extract_scenario_data_multi_year(flow_df, demand_df, costs_by_region_tech_year,
                                     scenario_col, target_years=[2022, 2025, 2030, 2035]):
    """Extract self-sufficiency and cost data by technology for multiple years, focusing on EUR and USA"""
    scenario_data = []

    for target_year in target_years:
        # Get data for target year
        flow_year = flow_df[flow_df['year'] == target_year].copy()
        demand_year = demand_df[demand_df['year'] == target_year].copy()

        # Get total demand by region
        demand_by_region = demand_year.groupby('region')['demand'].sum()

        # Get production by region and technology
        production_by_region_tech = flow_year.groupby(['region', 'technology'])[scenario_col].sum()

        for region in ['EUR', 'USA']:  # Focus on Europe and USA
            demand = demand_by_region.get(region, 0)

            # Get all technologies for this region
            region_production = production_by_region_tech.get(region, pd.Series())

            for tech in region_production.index:
                production = region_production[tech]

                # Calculate self-sufficiency for this technology
                self_sufficiency = calculate_self_sufficiency(production, demand)

                # Get costs for this region-technology-year combination
                costs = costs_by_region_tech_year.get((region, tech, target_year),
                                                      {'capex': 0, 'production': 0, 'trade': 0})
                total_cost = sum(costs.values()) / 1000000  # Convert to billion â‚¬

                scenario_data.append({
                    'region': region,
                    'technology': tech,
                    'year': target_year,
                    'self_sufficiency': self_sufficiency,
                    'total_cost': total_cost,
                    'production': production,
                    'demand': demand
                })

    return pd.DataFrame(scenario_data)


# ====================================
# PLOTTING FUNCTIONS
# ====================================

def get_scenario_colors():
    """Define colors for different scenarios"""
    return {
        'Base': '#8E6713',
        'DemandMet': '#007894',
        'SelfSuff40': '#215CAF',
        'Tariffs': '#B7352D',
        'FastDiffusion': '#A7117A',
        'Recycling': '#627313'
    }


def create_technology_evolution_plot(data_dict, output_dir='visualization'):
    """Create scatter plot showing technology evolution over time in Europe and USA
    2 rows (EUR/USA) x 3 panels (one per technology), colored by scenario, with arrows showing temporal evolution"""

    scenario_colors = get_scenario_colors()

    # Focus on 3 specific technologies
    focus_technologies = ['HP_assembly', 'HEX_manufacturing', 'Compressor_manufacturing']
    tech_labels = {
        'HP_assembly': 'Heat Pump',
        'HEX_manufacturing': 'Heat Exchanger',
        'Compressor_manufacturing': 'Compressor'
    }
    regions_list = ['EUR', 'USA']

    # Create figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

    # Plot for each region and technology
    for row_idx, region in enumerate(regions_list):
        region_label = 'Europe' if region == 'EUR' else 'United States'

        for col_idx, tech in enumerate(focus_technologies):
            ax = axes[row_idx, col_idx]  # Access the subplot using [row, col]

            # Labeling for the subplot (e.g., 'a.', 'b.', 'c.', 'd.', 'e.', 'f.')
            label_index = row_idx * 3 + col_idx
            subplot_label = chr(97 + label_index)
            ax.text(0.92, 0.98, f'{subplot_label}.',
                    transform=ax.transAxes,
                    fontsize=12, fontweight='bold',
                    verticalalignment='top',
                    ha='right')

            # Column Titles (Technology Names) - only for the top row
            if row_idx == 0:
                ax.set_title(tech_labels[tech], fontsize=12)

            # Row Labels (Region Names) - only for the left column
            if col_idx == 0:
                ax.text(-0.25, 0.5, region_label,
                        transform=ax.transAxes,
                        fontsize=12,
                        va='center', rotation=90)

            # Plot each scenario
            for scenario_name in ['Base', 'DemandMet', 'SelfSuff40', 'Tariffs']:
                scenario_df = data_dict[scenario_name]

                # Filter for this technology AND region
                tech_data = scenario_df[
                    (scenario_df['technology'] == tech) &
                    (scenario_df['region'] == region)
                    ].sort_values('year')

                if not tech_data.empty:
                    color = scenario_colors.get(scenario_name, '#999999')

                    # Scatter plot with varying alpha
                    for _, row in tech_data.iterrows():
                        alpha_val = (row['year'] - 2020) / 15
                        ax.scatter(row['self_sufficiency'],
                                   row['total_cost'],
                                   color=color,
                                   s=50, alpha=alpha_val,
                                   edgecolor='none', linewidth=1,
                                   zorder=3)

                    # Draw arrows connecting consecutive years
                    for i in range(len(tech_data) - 1):
                        row_current = tech_data.iloc[i]
                        row_next = tech_data.iloc[i + 1]
                        alpha_val = (i + 1) / 4

                        arrow = FancyArrowPatch(
                            (row_current['self_sufficiency'], row_current['total_cost']),
                            (row_next['self_sufficiency'], row_next['total_cost']),
                            arrowstyle='-|>', mutation_scale=10,
                            color=color, alpha=alpha_val, linewidth=1,
                            zorder=2
                        )
                        ax.add_patch(arrow)

            # Formatting
            ax.set_xlabel('Self-Sufficiency', fontsize=12)

            # Only set Y-axis label on the left column
            if col_idx == 0:
                ax.set_ylabel('Cumulative Costs (Billion â‚¬)', fontsize=12)
            else:
                # Hide y-ticks and y-label on other columns for cleaner look
                ax.tick_params(labelleft=False)

            ax.grid(True, alpha=0.2, linestyle='-')
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(0, 75)

            # Format x-axis as percentage
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x * 100:.0f}%'))

            # Legend: only add the text-based legend once to the top-left subplot
            if row_idx == 0 and col_idx == 0:
                legend_y_start = 0.98
                legend_y_spacing = 0.05

                for i, scenario_name in enumerate(['Base', 'DemandMet', 'SelfSuff40', 'Tariffs']):
                    color = scenario_colors.get(scenario_name, '#999999')
                    y_pos = legend_y_start - i * legend_y_spacing

                    ax.text(0.02, y_pos, scenario_name,
                            transform=ax.transAxes,
                            fontsize=12,
                            color=color,
                            verticalalignment='top')

    # ====================================
    # EXPORT DATA TO CSV
    # ====================================

    print("\n" + "=" * 60)
    print("EXPORTING DATA TO CSV")
    print("=" * 60)

    # Combine all scenario data for export
    export_data = []

    for scenario_name, scenario_df in data_dict.items():
        scenario_df_copy = scenario_df.copy()
        scenario_df_copy['scenario'] = scenario_name
        export_data.append(scenario_df_copy)

    # Concatenate all scenarios
    combined_df = pd.concat(export_data, ignore_index=True)

    # Select relevant columns for export
    columns_to_export = ['scenario', 'technology', 'region', 'year',
                         'production', 'demand', 'self_sufficiency',
                         'capex', 'production_cost', 'trade_cost', 'total_cost']

    # Filter to only include columns that exist
    columns_to_export = [col for col in columns_to_export if col in combined_df.columns]

    # Export to CSV
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    combined_df[columns_to_export].to_csv(
        output_path / 'Fig2_technology_evolution_data.csv',
        index=False
    )
    print(f"Exported technology evolution data to {output_path / 'Fig2_technology_evolution_data.csv'}")
    print("CSV export complete!")
    print("=" * 60)

    # Save plots
    plt.savefig(output_path / 'Fig2_eur_usa_suff_cost_3tech.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'Fig2_eur_usa_suff_cost_3tech.pdf',
                bbox_inches='tight')

    print(f"\nPlots saved to {output_path}/")


# ====================================
# MAIN EXECUTION
# ====================================

def main():
    """Main execution function"""

    print("\n" + "=" * 60)
    print("TECHNOLOGY EVOLUTION ANALYSIS - EUROPE AND USA")
    print("=" * 60)

    # Define file paths (assuming they exist in the execution environment)
    flow_file = 'parameter_results/flow_conversion_output/flow_conversion_output_scenarios.csv'
    demand_file = 'ZEN-Model_HP/set_carriers/HP/demand_yearly_variation.csv'
    parameter_results_dir = 'parameter_results'

    # Define target years
    target_years = [2022, 2025, 2030, 2035]

    # Load data
    print("\nLoading production data...")
    try:
        flow_df = load_production_data(flow_file)
    except FileNotFoundError:
        print(f"Error: Flow file not found at {flow_file}")
        return

    print("Loading demand data...")
    try:
        demand_df = load_demand_data(demand_file)
    except FileNotFoundError:
        print(f"Error: Demand file not found at {demand_file}")
        return

    print("Loading cost data...")
    capex_df, opex_df = load_cost_data(parameter_results_dir)

    # Check if cost data was loaded successfully
    if capex_df is None or opex_df is None:
        print("Error: Required cost data could not be loaded. Cannot proceed with cost analysis.")
        return

    # Get scenario columns
    flow_scenario_cols = [col for col in flow_df.columns if 'value_scenario' in col]
    print(f"\nAvailable scenarios: {flow_scenario_cols}")

    # Process each scenario
    data_dict = {}

    for scenario_col in flow_scenario_cols:
        scenario_name = scenario_col.replace('value_scenario_', '') or 'Base'
        print(f"\nProcessing scenario: {scenario_name}")

        # Process costs by technology, region, and year
        costs_by_region_tech_year = process_costs_by_scenario_tech_year(
            capex_df, opex_df, scenario_col, target_years
        )

        # Extract scenario data for multiple years
        scenario_data = extract_scenario_data_multi_year(
            flow_df, demand_df, costs_by_region_tech_year,
            scenario_col, target_years
        )

        data_dict[scenario_name] = scenario_data

        print(f"Scenario: {scenario_name}")
        print(f"Number of data points: {len(scenario_data)}")

        # Show sample for HP_assembly in Europe
        sample_eur = scenario_data[
            (scenario_data['technology'] == 'HP_assembly') &
            (scenario_data['region'] == 'EUR')
            ]
        if not sample_eur.empty:
            print("\nHP_assembly in Europe:")
            print(sample_eur[['year', 'self_sufficiency', 'total_cost']].to_string(index=False))

        # Show sample for HP_assembly in USA
        sample_usa = scenario_data[
            (scenario_data['technology'] == 'HP_assembly') &
            (scenario_data['region'] == 'USA')
            ]
        if not sample_usa.empty:
            print("\nHP_assembly in USA:")
            print(sample_usa[['year', 'self_sufficiency', 'total_cost']].to_string(index=False))

    # Create plots
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    create_technology_evolution_plot(data_dict)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()