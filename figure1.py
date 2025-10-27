import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from pathlib import Path

# Set style
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
# DATA EXTRACTION AND REGION MAPPING
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


def explore_cost_files(parameter_results_dir='parameter_results'):
    """
    Explore available cost files to determine which to use
    """
    param_dir = Path(parameter_results_dir)

    print("\n" + "=" * 60)
    print("EXPLORING COST FILES")
    print("=" * 60)

    # List all cost-related files
    cost_files = list(param_dir.glob('*cost*.csv')) + list(param_dir.glob('*capex*.csv'))

    print(f"\nFound {len(cost_files)} cost-related files:")
    for f in cost_files:
        print(f"  - {f.name}")

    # Check each file
    cost_file_info = {}

    for cost_file in cost_files:
        try:
            df = pd.read_csv(cost_file)
            print(f"\n{cost_file.name}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")
            if 'technology' in df.columns:
                print(f"  Sample technologies: {df['technology'].unique()[:5].tolist()}")
            if 'location' in df.columns:
                print(f"  Sample locations: {df['location'].unique()[:5].tolist()}")

            cost_file_info[cost_file.name] = {
                'path': cost_file,
                'columns': df.columns.tolist(),
                'shape': df.shape
            }
        except Exception as e:
            print(f"  Error reading {cost_file.name}: {e}")

    return cost_file_info


def load_cost_data(parameter_results_dir='parameter_results'):
    """
    Load and process cost data from appropriate files

    Returns capital costs (CAPEX) and operational costs (OPEX) by region and scenario
    """
    param_dir = Path(parameter_results_dir)

    # First, explore to find the right files
    cost_file_info = explore_cost_files(parameter_results_dir)

    print("\n" + "=" * 60)
    print("LOADING COST DATA")
    print("=" * 60)

    # Try to load CAPEX data
    capex_files = [
        'cost_capex/cost_capex_scenarios.csv'
        # 'capex_yearly_scenarios.csv',
        # 'capex_approximation_scenarios.csv'
    ]

    capex_df = None
    for capex_file in capex_files:
        capex_path = param_dir / capex_file
        if capex_path.exists():
            print(f"\nTrying CAPEX file: {capex_file}")
            capex_df = pd.read_csv(capex_path)
            print(f"  Loaded successfully. Shape: {capex_df.shape}")

            # Check if it has HP_assembly data
            if 'technology' in capex_df.columns:
                hp_data = capex_df[capex_df['technology'] == 'HP_assembly']
                if not hp_data.empty:
                    print(f"  Found HP_assembly data: {len(hp_data)} rows")
                    break

    if capex_df is None:
        print("WARNING: No suitable CAPEX file found!")
        return None, None

    # Try to load OPEX data
    opex_files = [
        'cost_opex_yearly/cost_opex_yearly_scenarios.csv'
        # 'cost_opex_scenarios.csv'
    ]

    opex_df = None
    for opex_file in opex_files:
        opex_path = param_dir / opex_file
        if opex_path.exists():
            print(f"\nTrying OPEX file: {opex_file}")
            opex_df = pd.read_csv(opex_path)
            print(f"  Loaded successfully. Shape: {opex_df.shape}")

            if 'technology' in opex_df.columns:
                print(f"  Sample technologies: {opex_df['technology'].unique()[:10].tolist()}")
                # Check for HP_assembly or HP_transport
                hp_assembly = opex_df[opex_df['technology'] == 'HP_assembly']
                hp_transport = opex_df[opex_df['technology'].str.contains('HP_transport', na=False)]

                print(f"  HP_assembly rows: {len(hp_assembly)}")
                print(f"  HP_transport rows: {len(hp_transport)}")

                if not hp_assembly.empty or not hp_transport.empty:
                    break

    if opex_df is None:
        print("WARNING: No suitable OPEX file found!")

    return capex_df, opex_df


def process_capex_data(capex_df, scenario_bau, scenario_nze):
    """
    Process CAPEX data to get cumulative capital costs by region

    CAPEX represents the cost of building new capacity
    """
    if capex_df is None:
        return {}

    print("\n" + "=" * 60)
    print("PROCESSING CAPEX DATA")
    print("=" * 60)

    # Filter for HP_assembly
    capex_df = capex_df[capex_df['technology'] == 'HP_assembly'].copy()

    # CAPEX should only have node locations (no edges)
    if 'location' in capex_df.columns:
        # Check for edges (containing '-')
        has_edges = capex_df['location'].str.contains('-', na=False).any()
        if has_edges:
            print("  Warning: Found edge locations in CAPEX data, filtering to nodes only")
            capex_df = capex_df[~capex_df['location'].str.contains('-', na=False)].copy()

        capex_df['region'] = capex_df['location'].apply(map_to_region)
    else:
        print("ERROR: No location column found in CAPEX data")
        return {}

    # Get scenario columns
    scenario_cols = [col for col in capex_df.columns if 'value_scenario' in col]
    print(f"Scenario columns: {scenario_cols}")

    # Sum CAPEX by region and scenario (cumulative over all years)
    capex_by_region = {}

    for region in ['CHN', 'EUR', 'USA', 'ROW']:
        region_df = capex_df[capex_df['region'] == region]

        capex_by_region[region] = {
            'Base': region_df[scenario_bau].sum() if scenario_bau in scenario_cols else 0,
            'DemandMet': region_df[scenario_nze].sum() if scenario_nze in scenario_cols else 0
        }

        print(f"{region}: BAU={capex_by_region[region]['Base']:.2e}, NZE={capex_by_region[region]['DemandMet']:.2e}")

    return capex_by_region


def process_opex_data(opex_df, scenario_bau, scenario_nze):
    """
    Process OPEX data to separate production costs and trade costs

    Technology column determines the type:
    - 'HP_assembly' = production costs at the node location
    - Technologies ending with '_transport' (e.g., 'HP_transport', 'Steel_transport') = trade costs
      -> Location format: 'NODE1-NODE2' where NODE1 exports to NODE2
      -> Trade costs are assigned to the IMPORTER (NODE2)
    """
    if opex_df is None:
        return {}, {}

    print("\n" + "=" * 60)
    print("PROCESSING OPEX DATA")
    print("=" * 60)

    # Get scenario columns
    scenario_cols = [col for col in opex_df.columns if 'value_scenario' in col]
    print(f"Scenario columns: {scenario_cols}")

    if 'location' not in opex_df.columns or 'technology' not in opex_df.columns:
        print("ERROR: Missing required columns in OPEX data")
        return {}, {}

    # Separate production and transport technologies
    production_df = opex_df[opex_df['technology'] == 'HP_assembly'].copy()
    transport_df = opex_df[opex_df['technology'] == 'HP_transport'].copy()

    print(f"\nProduction costs (HP_assembly): {len(production_df)} rows")
    print(f"Trade costs (_transport technologies): {len(transport_df)} rows")

    if not transport_df.empty:
        print(f"  Transport technologies found: {transport_df['technology'].unique().tolist()}")
        print(f"  Sample transport locations: {transport_df['location'].unique()[:5].tolist()}")

    # Process production costs
    production_by_region = {}

    if not production_df.empty:
        # Production should have single-node locations
        production_df = production_df[~production_df['location'].str.contains('-', na=False)].copy()
        production_df['region'] = production_df['location'].apply(map_to_region)

        for region in ['CHN', 'EUR', 'USA', 'ROW']:
            region_df = production_df[production_df['region'] == region]

            production_by_region[region] = {
                'Base': region_df[scenario_bau].sum() if scenario_bau in scenario_cols else 0,
                'DemandMet': region_df[scenario_nze].sum() if scenario_nze in scenario_cols else 0
            }

            print(
                f"{region} Production: BAU={production_by_region[region]['Base']:.2e}, NZE={production_by_region[region]['DemandMet']:.2e}")
    else:
        print("No production cost data found")
        production_by_region = {region: {'Base': 0, 'DemandMet': 0} for region in ['CHN', 'EUR', 'USA', 'ROW']}

    # Process trade costs from transport technologies
    trade_by_region = {}

    if not transport_df.empty:
        # Transport locations should be in format 'NODE1-NODE2'
        transport_df['has_dash'] = transport_df['location'].str.contains('-', na=False)
        valid_transport = transport_df[transport_df['has_dash']].copy()

        if not valid_transport.empty:
            # Split location into from_node and to_node
            valid_transport['from_node'] = valid_transport['location'].str.split('-').str[0]
            valid_transport['to_node'] = valid_transport['location'].str.split('-').str[1]

            # Assign to importer (to_node)
            valid_transport['region'] = valid_transport['to_node'].apply(map_to_region)

            print(f"\nSample trade routes (assigned to importer):")
            for idx in valid_transport.head(5).index:
                print(
                    f"  {valid_transport.loc[idx, 'location']} ({valid_transport.loc[idx, 'technology']}) -> assigned to {valid_transport.loc[idx, 'region']}")

            for region in ['CHN', 'EUR', 'USA', 'ROW']:
                region_df = valid_transport[valid_transport['region'] == region]

                trade_by_region[region] = {
                    'Base': region_df[scenario_bau].sum() if scenario_bau in scenario_cols else 0,
                    'DemandMet': region_df[scenario_nze].sum() if scenario_nze in scenario_cols else 0
                }

                print(
                    f"{region} Trade (imports): BAU={trade_by_region[region]['Base']:.2e}, NZE={trade_by_region[region]['DemandMet']:.2e}")
        else:
            print("No valid transport locations found (expected format: NODE1-NODE2)")
            trade_by_region = {region: {'Base': 0, 'DemandMet': 0} for region in ['CHN', 'EUR', 'USA', 'ROW']}
    else:
        print("No trade data found")
        trade_by_region = {region: {'Base': 0, 'DemandMet': 0} for region in ['CHN', 'EUR', 'USA', 'ROW']}

    return production_by_region, trade_by_region


def combine_costs(capex_by_region, production_by_region, trade_by_region):
    """
    Combine CAPEX, production, and trade costs into final cost structure

    Returns dict with structure: {region: {scenario: [capital, production, trade]}}
    """
    combined_costs = {}

    print("\n" + "=" * 60)
    print("COMBINING COSTS")
    print("=" * 60)

    for region in ['CHN', 'EUR', 'USA', 'ROW']:
        combined_costs[region] = {
            'Base': [
                capex_by_region.get(region, {}).get('Base', 0),
                production_by_region.get(region, {}).get('Base', 0),
                trade_by_region.get(region, {}).get('Base', 0)
            ],
            'DemandMet': [
                capex_by_region.get(region, {}).get('DemandMet', 0),
                production_by_region.get(region, {}).get('DemandMet', 0),
                trade_by_region.get(region, {}).get('DemandMet', 0)
            ]
        }

        total_bau = sum(combined_costs[region]['Base'])
        total_nze = sum(combined_costs[region]['DemandMet'])

        print(f"\n{region}:")
        print(f"  BAU: Capital={combined_costs[region]['Base'][0]:.2e}, "
              f"Production={combined_costs[region]['Base'][1]:.2e}, "
              f"Trade={combined_costs[region]['Base'][2]:.2e}, "
              f"Total={total_bau:.2e}")
        print(f"  NZE: Capital={combined_costs[region]['DemandMet'][0]:.2e}, "
              f"Production={combined_costs[region]['DemandMet'][1]:.2e}, "
              f"Trade={combined_costs[region]['DemandMet'][2]:.2e}, "
              f"Total={total_nze:.2e}")

    return combined_costs


def load_and_process_data(flow_file, capacity_file, demand_file, parameter_results_dir='parameter_results'):
    """
    Load and process all data including costs
    """

    # Read production and capacity data
    flow_df = pd.read_csv(flow_file)
    capacity_df = pd.read_csv(capacity_file)
    demand_df = pd.read_csv(demand_file)

    # Process demand
    print("\n" + "=" * 60)
    print("PROCESSING DEMAND DATA")
    print("=" * 60)

    if 'year' not in demand_df.columns:
        first_col = demand_df.columns[0]
        if demand_df[first_col].dtype in ['int64', 'float64']:
            demand_df = demand_df.rename(columns={first_col: 'year'})

    if demand_df['year'].max() < 2000:
        demand_df['year'] = demand_df['year'] + 2022

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

    # Filter for HP assembly
    flow_df = flow_df[flow_df['technology'] == 'HP_assembly'].copy()
    capacity_df = capacity_df[capacity_df['technology'] == 'HP_assembly'].copy()

    # Apply region mapping
    flow_df['region'] = flow_df['node'].apply(map_to_region)

    if 'location' in capacity_df.columns:
        capacity_df['region'] = capacity_df['location'].apply(map_to_region)
    else:
        capacity_df['region'] = capacity_df['node'].apply(map_to_region)

    # Convert time to years
    if 'time_operation' in flow_df.columns:
        flow_df['year'] = flow_df['time_operation'] + 2022

    if 'year' in capacity_df.columns and capacity_df['year'].max() < 2000:
        capacity_df['year'] = capacity_df['year'] + 2022

    # Get scenario columns
    flow_scenario_cols = [col for col in flow_df.columns if 'value_scenario' in col]
    capacity_scenario_cols = [col for col in capacity_df.columns if 'value_scenario' in col]

    flow_bau_col = flow_scenario_cols[0]
    flow_nze_col = 'value_scenario_DemandMet' if 'value_scenario_DemandMet' in flow_scenario_cols else \
        flow_scenario_cols[1]

    capacity_bau_col = capacity_scenario_cols[0]
    capacity_nze_col = 'value_scenario_DemandMet' if 'value_scenario_DemandMet' in capacity_scenario_cols else \
        capacity_scenario_cols[1]

    print(f"\nFlow BAU: {flow_bau_col}, NZE: {flow_nze_col}")
    print(f"Capacity BAU: {capacity_bau_col}, NZE: {capacity_nze_col}")

    # Load and process cost data
    capex_df, opex_df = load_cost_data(parameter_results_dir)

    capex_by_region = process_capex_data(capex_df, capacity_bau_col, capacity_nze_col)
    production_by_region, trade_by_region = process_opex_data(opex_df, flow_bau_col, flow_nze_col)

    costs_by_region = combine_costs(capex_by_region, production_by_region, trade_by_region)

    return flow_df, capacity_df, demand_long, flow_bau_col, flow_nze_col, capacity_bau_col, capacity_nze_col, costs_by_region


def extract_region_scenario_data(flow_df, capacity_df, demand_df,
                                 flow_bau_col, flow_nze_col,
                                 capacity_bau_col, capacity_nze_col,
                                 costs_by_region,
                                 region, years=[2022, 2025, 2030, 2035]):
    """Extract data for a specific region and both scenarios"""

    region_data = {
        'Base': {'capacity': [], 'production': [], 'demand': []},
        'DemandMet': {'capacity': [], 'production': [], 'demand': []}
    }

    # Aggregate by region and year
    flow_by_region = flow_df[flow_df['region'] == region].groupby('year').agg({
        flow_bau_col: 'sum',
        flow_nze_col: 'sum'
    }).reset_index()

    capacity_by_region = capacity_df[capacity_df['region'] == region].groupby('year').agg({
        capacity_bau_col: 'sum',
        capacity_nze_col: 'sum'
    }).reset_index()

    demand_by_region = demand_df[demand_df['region'] == region].groupby('year').agg({
        'demand': 'sum'
    }).reset_index()

    # Extract values for specific years
    for year in years:
        cap_bau = capacity_by_region[capacity_by_region['year'] == year][capacity_bau_col]
        cap_nze = capacity_by_region[capacity_by_region['year'] == year][capacity_nze_col]
        prod_bau = flow_by_region[flow_by_region['year'] == year][flow_bau_col]
        prod_nze = flow_by_region[flow_by_region['year'] == year][flow_nze_col]
        dem = demand_by_region[demand_by_region['year'] == year]['demand']

        region_data['Base']['capacity'].append(cap_bau.values[0] if len(cap_bau) > 0 else 0)
        region_data['Base']['production'].append(prod_bau.values[0] if len(prod_bau) > 0 else 0)
        region_data['Base']['demand'].append(dem.values[0] if len(dem) > 0 else 0)

        region_data['DemandMet']['capacity'].append(cap_nze.values[0] if len(cap_nze) > 0 else 0)
        region_data['DemandMet']['production'].append(prod_nze.values[0] if len(prod_nze) > 0 else 0)
        region_data['DemandMet']['demand'].append(dem.values[0] if len(dem) > 0 else 0)

    # Use actual cost data
    region_data['costs'] = costs_by_region.get(region, {'Base': [0, 0, 0], 'DemandMet': [0, 0, 0]})

    return region_data


# ====================================
# MAIN EXECUTION
# ====================================

# Define file paths
flow_file = 'parameter_results/flow_conversion_output/flow_conversion_output_scenarios.csv'
capacity_file = 'parameter_results/capacity/capacity_scenarios.csv'
demand_file = 'ZEN-Model_HP/set_carriers/HP/demand_yearly_variation.csv'

# Load and process data
try:
    flow_df, capacity_df, demand_df, flow_bau_col, flow_nze_col, capacity_bau_col, capacity_nze_col, costs_by_region = \
        load_and_process_data(flow_file, capacity_file, demand_file)

    print("\n" + "=" * 60)
    print("DATA LOADED SUCCESSFULLY")
    print("=" * 60)

    # Define regions and years
    regions_list = ['CHN', 'EUR', 'USA', 'ROW']
    years = [2022, 2030, 2035]

    # Extract data for each region
    data = {}
    for region in regions_list:
        data[region] = extract_region_scenario_data(
            flow_df, capacity_df, demand_df,
            flow_bau_col, flow_nze_col, capacity_bau_col, capacity_nze_col,
            costs_by_region,
            region, years
        )

except Exception as e:
    print(f"Error loading data: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# ====================================
# PLOTTING
# ====================================

# Colors
bau_colors = {'capacity': '#8E6713', 'production': '#A58542', 'demand': '#365213'}
nze_colors = {'capacity': '#007894', 'production': '#3395AB', 'demand': '#365213'}

# Create figure
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(3, 4, hspace=0.25, wspace=0.3, left=0.08, right=0.98, top=0.96, bottom=0.08)

# X positions for discrete bars
x_pos = np.arange(len(years))
bar_width = 0.5

# Calculate maximum y value
all_values = []
for region in regions_list:
    all_values.extend(data[region]['Base']['capacity'])
    all_values.extend(data[region]['Base']['production'])
    all_values.extend(data[region]['Base']['demand'])
    all_values.extend(data[region]['DemandMet']['capacity'])
    all_values.extend(data[region]['DemandMet']['production'])
    all_values.extend(data[region]['DemandMet']['demand'])

maxY = max(all_values) * 1.1 if max(all_values) > 0 else 100

# Calculate max cost - convert to billions if needed
all_costs = []
for region in regions_list:
    all_costs.append(sum(data[region]['costs']['Base']))
    all_costs.append(sum(data[region]['costs']['DemandMet']))

# Check if costs are very large (likely in dollars, need to convert to billions)
max_cost_raw = max(all_costs) if all_costs else 0
if max_cost_raw > 1e5:  # unit capex opex is kiloEuro
    cost_scale = 1e6  # Convert to billions
    cost_label = 'Billion €'
else:
    cost_scale = 1
    cost_label = ''

maxCost = (max_cost_raw / cost_scale) * 1.1 if max_cost_raw > 0 else 500

print(f"\nCost scaling: max_raw={max_cost_raw:.2e}, scale={cost_scale:.2e}, max_display={maxCost:.2f}")

# LABELS A B C D
subplot_labels = [f'{chr(ord("a") + i)}.' for i in range(12)]  # 3 rows × 4 columns

# Plot for each region (now organized by columns)
for col, region in enumerate(regions_list):
    region_data = data[region]

    # Region label at the top of each column
    if region == 'CHN':
        region_label = 'China'
    elif region == 'EUR':
        region_label = 'Europe'
    elif region == 'USA':
        region_label = 'United States'
    elif region == 'ROW':
        region_label = 'Rest of World'
    else:
        region_label = region

    # BAU subplot (row 0)
    ax1 = fig.add_subplot(gs[0, col])

    label_index = col
    ax1.text(0.04, 0.96, subplot_labels[label_index], transform=ax1.transAxes,
             fontsize=12, va='top', ha='left')

    ax1.bar(x_pos, region_data['Base']['production'], width=bar_width,
            color=bau_colors['production'], alpha=0.8)
    ax1.plot(x_pos, region_data['Base']['capacity'], '+',
             color=bau_colors['capacity'], markersize=10, markerfacecolor='none',
             markeredgewidth=3)
    ax1.plot(x_pos, region_data['Base']['demand'], '_',
             color=bau_colors['demand'], markersize=10, markerfacecolor='none',
             markeredgewidth=3)

    ax1.set_ylim(0, 200)
    if col == 0:
        ax1.set_ylabel('GW', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(years)
    ax1.grid(True, alpha=0.2)
    ax1.set_title(region_label, fontsize=12)

    # NZE subplot (row 1)
    ax2 = fig.add_subplot(gs[1, col])

    label_index = col + 4
    ax2.text(0.04, 0.96, subplot_labels[label_index], transform=ax2.transAxes,
             fontsize=12, va='top', ha='left')

    ax2.bar(x_pos, region_data['DemandMet']['production'], width=bar_width,
            color=nze_colors['production'], alpha=0.8)
    ax2.plot(x_pos, region_data['DemandMet']['capacity'], '+',
             color=nze_colors['capacity'], markersize=10, markerfacecolor='none',
             markeredgewidth=3)
    ax2.plot(x_pos, region_data['DemandMet']['demand'], '_',
             color=nze_colors['demand'], markersize=10, markerfacecolor='none',
             markeredgewidth=3)

    ax2.set_ylim(0, 200)
    if col == 0:
        ax2.set_ylabel('GW', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(years)
    ax2.grid(True, alpha=0.2)

    # Cost subplot (row 2)
    ax3 = fig.add_subplot(gs[2, col])

    label_index = col + 8
    ax3.text(0.08, 0.96, subplot_labels[label_index], transform=ax3.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right')

    cost_x_pos = [0, 1]
    cost_width = 0.4

    # Scale costs
    costs_bau_scaled = [c / cost_scale for c in region_data['costs']['Base']]
    costs_nze_scaled = [c / cost_scale for c in region_data['costs']['DemandMet']]

    # BAU stacked bar
    bottom_bau = 0
    colors_bau = ['#A58542',
                  '#BBA471',
                  '#D2C2A1'
                  ]
    for i, cost in enumerate(costs_bau_scaled):
        ax3.bar(cost_x_pos[0], cost, cost_width, bottom=bottom_bau,
                color=colors_bau[i], edgecolor='white', linewidth=0.5)
        bottom_bau += cost

    # NZE stacked bar
    bottom_nze = 0
    colors_nze = ['#3395AB',
                  '#66AFC0',
                  '#99CAD5']
    for i, cost in enumerate(costs_nze_scaled):
        ax3.bar(cost_x_pos[1], cost, cost_width, bottom=bottom_nze,
                color=colors_nze[i], edgecolor='white', linewidth=0.5)
        bottom_nze += cost
        plt.ylim(0, 70)

    if col == 0:
        ax3.set_ylabel(cost_label, fontsize=12)
    ax3.set_xticks(cost_x_pos)
    ax3.set_xticklabels(['Base', 'Demand-met'])
    ax3.set_xlim(-0.5, 1.5)
    ax3.grid(True, alpha=0.2)

# Add row labels on the left side
fig.text(0.02, gs[0, 0].get_position(fig).y0 +
         (gs[0, 0].get_position(fig).y1 - gs[0, 0].get_position(fig).y0) / 2,
         'Base scenario', fontsize=12, va='center', ha='center', rotation=90)
fig.text(0.02, gs[1, 0].get_position(fig).y0 +
         (gs[1, 0].get_position(fig).y1 - gs[1, 0].get_position(fig).y0) / 2,
         'Demand-met scenario', fontsize=12, va='center', ha='center', rotation=90)
fig.text(0.02, gs[2, 0].get_position(fig).y0 +
         (gs[2, 0].get_position(fig).y1 - gs[2, 0].get_position(fig).y0) / 2,
         'Cumulative Costs', fontsize=12, va='center', ha='center', rotation=90)

# Legend - UPDATED to reflect production instead of operation
legend_elements = [
    plt.Line2D([0], [0], marker='+', color='w', markerfacecolor='none',
               markeredgecolor='#959595', markeredgewidth=3, markersize=8,
               label='Capacity'),
    mpatches.Patch(facecolor='#959595', label='Production (bars)'),
    plt.Line2D([0], [0], marker='_', color='w', markerfacecolor='none',
               markeredgecolor='#28a745', markeredgewidth=3, markersize=8,
               label='Demand'),
    mpatches.Patch(facecolor=(2 / 255, 2 / 255, 2 / 255, 1), label='Capital costs'),
    mpatches.Patch(facecolor=(2 / 255, 2 / 255, 2 / 255, 0.6), label='Production costs'),
    mpatches.Patch(facecolor=(2 / 255, 2 / 255, 2 / 255, 0.35), label='Importing costs'),
]

fig.legend(handles=legend_elements, loc='lower center',
           ncol=6, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.001))

# plt.figtext(0.5, 0.005, 'Colors: BAU (orange tones), NZE (blue tones), with different opacities for cost types.',
#             ha='center', fontsize=9, style='italic')

# Save
output_dir = Path('visualization')
output_dir.mkdir(exist_ok=True)

plt.savefig(output_dir / 'Fig1_dcp_costs_base_demandmet3.png', dpi=330, bbox_inches='tight')
plt.savefig(output_dir / 'Fig1_dcp_costs_base_demandmet3.pdf', bbox_inches='tight')
plt.show()

print(f"\nPlots saved to {output_dir}/")
print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)