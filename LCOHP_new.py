import pandas as pd
import numpy as np
from pathlib import Path


def load_data_files(flow_file, opex_file, capex_file, transport_file):
    """
    Load and prepare data from CSV files, storing in separate dataframes

    Returns:
    --------
    tuple of DataFrames:
        (opex_df, capex_df, production_output_df, transport_flow_df, scenarios)
    """
    print(f"Reading data files...")

    # Read CSV files
    flow_df = pd.read_csv(flow_file)
    opex_df = pd.read_csv(opex_file)
    capex_df = pd.read_csv(capex_file)
    transport_df = pd.read_csv(transport_file)

    # Print column names for debugging
    print("Flow file columns:", flow_df.columns.tolist())
    print("OPEX file columns:", opex_df.columns.tolist())
    print("CAPEX file columns:", capex_df.columns.tolist())
    print("Transport file columns:", transport_df.columns.tolist())

    # Look for scenario columns with format value_scenario_X
    scenarios = []

    # Check for value_scenario_ columns
    scenario_columns = [col for col in flow_df.columns if col.startswith('value_scenario_')]

    if scenario_columns:
        # Extract scenario names from column names (remove 'value_scenario_' prefix)
        scenarios = [col.replace('value_scenario_', '') for col in scenario_columns]
        print(f"Found {len(scenarios)} scenarios from columns: {scenarios}")

        # Process flow_df - use a different approach to avoid value_name conflict
        # First, create a copy of the dataframe
        flow_processed = pd.DataFrame()

        # Process each scenario separately
        for scenario_col in scenario_columns:
            scenario_name = scenario_col.replace('value_scenario_', '')
            temp_df = flow_df[['technology', 'carrier', 'node', 'time_operation', scenario_col]].copy()
            temp_df.rename(columns={scenario_col: 'flow_value'}, inplace=True)
            temp_df['scenario'] = scenario_name
            flow_processed = pd.concat([flow_processed, temp_df], ignore_index=True)

        # Replace the original dataframe
        flow_df = flow_processed

        # Process opex_df
        opex_processed = pd.DataFrame()
        for scenario_col in [col for col in opex_df.columns if col.startswith('value_scenario_')]:
            scenario_name = scenario_col.replace('value_scenario_', '')
            temp_df = opex_df[['technology', 'location', 'year', scenario_col]].copy()
            temp_df.rename(columns={scenario_col: 'opex_value'}, inplace=True)
            temp_df['scenario'] = scenario_name
            opex_processed = pd.concat([opex_processed, temp_df], ignore_index=True)
        opex_df = opex_processed

        # Process capex_df
        capex_processed = pd.DataFrame()
        for scenario_col in [col for col in capex_df.columns if col.startswith('value_scenario_')]:
            scenario_name = scenario_col.replace('value_scenario_', '')
            temp_df = capex_df[['technology', 'capacity_type', 'location', 'year', scenario_col]].copy()
            temp_df.rename(columns={scenario_col: 'capex_value'}, inplace=True)
            temp_df['scenario'] = scenario_name
            capex_processed = pd.concat([capex_processed, temp_df], ignore_index=True)
        capex_df = capex_processed

        # Process transport_df
        transport_processed = pd.DataFrame()
        for scenario_col in [col for col in transport_df.columns if col.startswith('value_scenario_')]:
            scenario_name = scenario_col.replace('value_scenario_', '')
            temp_df = transport_df[['technology', 'edge', 'time_operation', scenario_col]].copy()
            temp_df.rename(columns={scenario_col: 'flow_value'}, inplace=True)
            temp_df['scenario'] = scenario_name
            transport_processed = pd.concat([transport_processed, temp_df], ignore_index=True)
        transport_df = transport_processed

    # If no scenarios found, use 'base' as default
    if not scenarios:
        scenarios = ['base']
        print("No scenario information found, assuming base scenario only")

    # Print the new column structures after transformation
    print("Flow file columns after processing:", flow_df.columns.tolist())
    print("OPEX file columns after processing:", opex_df.columns.tolist())
    print("CAPEX file columns after processing:", capex_df.columns.tolist())
    print("Transport file columns after processing:", transport_df.columns.tolist())

    # Rename columns for clarity
    flow_df = flow_df.rename(columns={'flow_value': 'flow', 'node': 'location'})
    opex_df = opex_df.rename(columns={'opex_value': 'opex', 'year': 'time_operation'})
    capex_df = capex_df.rename(columns={'capex_value': 'capex', 'year': 'time_operation'})
    transport_df = transport_df.rename(columns={'flow_value': 'flow'})

    # Identify transport technologies - looking for "transport" in technology name
    is_transport = flow_df['technology'].str.contains('transport', case=False, na=False)

    # Split into production and transport dataframes
    production_output_df = flow_df[~is_transport].copy()

    # Extract transport flows from the transport file
    transport_flow_df = transport_df.copy()

    print(f"Production output data: {len(production_output_df)} rows")
    print(f"Transport flow data: {len(transport_flow_df)} rows")
    print(f"OPEX data: {len(opex_df)} rows")
    print(f"CAPEX data: {len(capex_df)} rows")

    # Return the dataframes and detected scenarios
    return opex_df, capex_df, production_output_df, transport_flow_df, scenarios


def calculate_production_cost_per_unit(flow_df, opex_df, capex_df):
    """Calculate production cost per unit"""
    # Merge dataframes on common columns
    merged_df = flow_df.merge(
        opex_df,
        on=['technology', 'location', 'time_operation'],
        how='left'
    ).merge(
        capex_df,
        on=['technology', 'location', 'time_operation'],
        how='left'
    )

    # Fill NaN values with 0 for cost calculations
    merged_df['opex'] = merged_df['opex'].fillna(0)
    merged_df['capex'] = merged_df['capex'].fillna(0)

    # Calculate total cost per unit
    merged_df['total_cost'] = merged_df['opex'] + merged_df['capex']

    # Also keep track of opex separately for ratio calculations
    merged_df['opex_cost'] = merged_df['opex']

    # Set cost_per_unit to NaN where flow is 0 (no production)
    merged_df['cost_per_unit'] = np.where(
        (merged_df['flow'] == 0),
        np.nan,
        merged_df['total_cost'] / merged_df['flow']
    )

    # Also calculate opex per unit
    merged_df['opex_per_unit'] = np.where(
        (merged_df['flow'] == 0),
        np.nan,
        merged_df['opex'] / merged_df['flow']
    )

    # Rename columns to match what downstream functions expect
    merged_df = merged_df.rename(columns={
        'flow': 'total_production',
        'total_cost': 'total_production_cost',
        'cost_per_unit': 'production_cost_per_unit'
    })

    # Add commodity column based on technology
    if 'carrier' in merged_df.columns:
        merged_df['commodity'] = merged_df['carrier']
    else:
        merged_df['commodity'] = merged_df['technology'].str.replace('_production', '').str.replace('_mining', '')

    print(f"Production cost calculation: {len(merged_df)} rows processed")

    return merged_df


def calculate_transport_cost(transport_flow_df, opex_df, capex_df):
    """
    Calculate transport cost for each importing country, timestep, and commodity

    Returns:
    --------
    tuple of DataFrames:
        (transport_cost_df, import_details_df, transport_cost_per_unit_df)
    """
    print("\nCalculating transport costs...")

    # Filter for transport technologies in OPEX and CAPEX
    opex_transport = opex_df[opex_df['technology'].str.contains('transport', case=False, na=False)].copy()
    capex_transport = capex_df[capex_df['technology'].str.contains('transport', case=False, na=False)].copy()

    # Ensure we have edge information in transport flows
    if 'edge' not in transport_flow_df.columns or not transport_flow_df['edge'].str.contains('-', na=False).any():
        print("ERROR: Transport flows missing edge information in format 'EXPORTER-IMPORTER'")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Extract exporter-importer pairs and commodity from transport flows
    transport_flow_df[['exporter', 'importer']] = transport_flow_df['edge'].str.split('-', expand=True)
    transport_flow_df['commodity'] = transport_flow_df['technology'].apply(
        lambda x: x.replace('_transport', '').replace('transport_', '')
    )

    # Prepare OPEX data for merging
    if 'location' in opex_transport.columns and opex_transport['location'].str.contains('-', na=False).any():
        # Location contains country pairs (exporter-importer)
        opex_transport[['exporter', 'importer']] = opex_transport['location'].str.split('-', expand=True)
        opex_transport['commodity'] = opex_transport['technology'].apply(
            lambda x: x.replace('_transport', '').replace('transport_', '')
        )

        # Merge on exporter, importer, commodity, time
        transport_costs = transport_flow_df.merge(
            opex_transport[['exporter', 'importer', 'commodity', 'time_operation', 'opex']],
            on=['exporter', 'importer', 'commodity', 'time_operation'],
            how='left'
        )
    else:
        # Location does not contain country pairs, try to match on technology and time
        opex_transport['commodity'] = opex_transport['technology'].apply(
            lambda x: x.replace('_transport', '').replace('transport_', '')
        )

        # Merge on technology and time
        transport_costs = transport_flow_df.merge(
            opex_transport[['technology', 'time_operation', 'opex']],
            on=['technology', 'time_operation'],
            how='left'
        )

    # Prepare CAPEX data for merging (similar approach as OPEX)
    if 'location' in capex_transport.columns and capex_transport['location'].str.contains('-', na=False).any():
        capex_transport[['exporter', 'importer']] = capex_transport['location'].str.split('-', expand=True)
        capex_transport['commodity'] = capex_transport['technology'].apply(
            lambda x: x.replace('_transport', '').replace('transport_', '')
        )

        # Merge with CAPEX
        transport_costs = transport_costs.merge(
            capex_transport[['exporter', 'importer', 'commodity', 'time_operation', 'capex']],
            on=['exporter', 'importer', 'commodity', 'time_operation'],
            how='left'
        )
    else:
        capex_transport['commodity'] = capex_transport['technology'].apply(
            lambda x: x.replace('_transport', '').replace('transport_', '')
        )

        # Merge on technology and time
        transport_costs = transport_costs.merge(
            capex_transport[['technology', 'time_operation', 'capex']],
            on=['technology', 'time_operation'],
            how='left'
        )

    # Fill NaN values with 0
    transport_costs['opex'] = transport_costs['opex'].fillna(0)
    transport_costs['capex'] = transport_costs['capex'].fillna(0)

    # Calculate total transport cost
    transport_costs['transport_cost'] = transport_costs['opex'] + transport_costs['capex']

    # Create a detailed import dataframe (which country imports from where)
    if 'importer' in transport_costs.columns and 'exporter' in transport_costs.columns:
        import_details = transport_costs[['importer', 'exporter', 'commodity', 'time_operation', 'flow']].copy()
        import_details = import_details.rename(columns={'flow': 'import_flow'})
    else:
        print("Warning: Missing importer/exporter columns. Creating empty import details dataframe.")
        import_details = pd.DataFrame(columns=['location', 'exporter', 'commodity', 'time_operation', 'import_flow'])

    # Sum transport costs by importing country, commodity, and timestep
    transport_by_importer = transport_costs.groupby(['importer', 'commodity', 'time_operation']).agg(
        total_transport_cost=('transport_cost', 'sum'),
        total_transport_opex=('opex', 'sum'),
        total_transport_flow=('flow', 'sum')
    ).reset_index()

    # Calculate transport cost per unit
    transport_by_importer['transport_cost_per_unit'] = np.where(
        transport_by_importer['total_transport_flow'] == 0,
        np.nan,
        transport_by_importer['total_transport_cost'] / transport_by_importer['total_transport_flow']
    )

    # Create a pivot table with locations as rows and timesteps as columns
    pivot_table = transport_by_importer.pivot_table(
        index=['importer', 'commodity'],
        columns='time_operation',
        values='transport_cost_per_unit'
    )

    # Rename columns to timestep format
    pivot_table.columns = [f'timestep_{t}' for t in pivot_table.columns]

    # Add average column
    pivot_table['average'] = pivot_table.mean(axis=1, skipna=True)

    # Reset index for better output format
    pivot_table = pivot_table.reset_index()

    # Rename importer to location for consistency with other functions
    transport_by_importer = transport_by_importer.rename(columns={'importer': 'location'})
    import_details = import_details.rename(columns={'importer': 'location'})
    pivot_table = pivot_table.rename(columns={'importer': 'location'})

    print(f"Transport cost calculation: {len(transport_by_importer)} location-commodity-timestep combinations")

    return transport_by_importer, import_details, pivot_table


def calculate_total_cost_of_goods(production_cost_df, transport_cost_df, import_details_df, output_dir=None):
    """
    Calculate the total cost of goods for each country, including domestic production and imports

    Returns:
    --------
    DataFrame: total_cost_df with weighted cost per unit for each location, commodity, and timestep
    """
    print("\nCalculating total cost of goods...")

    # 1. Start with domestic production data
    domestic_df = production_cost_df[['location', 'commodity', 'time_operation',
                                      'total_production', 'total_production_cost',
                                      'production_cost_per_unit', 'opex_cost']].copy()

    # 2. Get import data with production costs from exporting countries
    if not import_details_df.empty:
        # Merge import flows with production costs from exporting countries
        import_with_prod_cost = import_details_df.merge(
            production_cost_df[['location', 'commodity', 'time_operation', 'opex_per_unit']],
            left_on=['exporter', 'commodity', 'time_operation'],
            right_on=['location', 'commodity', 'time_operation'],
            how='left'
        )

        # Rename columns for clarity
        import_with_prod_cost = import_with_prod_cost.rename(
            columns={'location_x': 'location', 'location_y': 'exporter_location',
                     'opex_per_unit': 'exporter_production_cost_per_unit'}
        )

        # Calculate production cost for each import flow
        import_with_prod_cost['import_production_cost'] = (
                import_with_prod_cost['import_flow'] *
                import_with_prod_cost['exporter_production_cost_per_unit']
        )

        # Sum import flows and production costs by importing location
        import_summary = import_with_prod_cost.groupby(['location', 'commodity', 'time_operation']).agg(
            total_import_flow=('import_flow', 'sum'),
            total_import_production_cost=('import_production_cost', 'sum')
        ).reset_index()

        # 3. Add transport costs to imports
        if not transport_cost_df.empty:
            import_summary = import_summary.merge(
                transport_cost_df[['location', 'commodity', 'time_operation',
                                   'total_transport_cost', 'total_transport_opex']],
                on=['location', 'commodity', 'time_operation'],
                how='left'
            )

            # Fill NaN values with 0
            import_summary['total_transport_cost'] = import_summary['total_transport_cost'].fillna(0)
            import_summary['total_transport_opex'] = import_summary['total_transport_opex'].fillna(0)

            # Calculate total import cost (production + transport)
            import_summary['total_import_cost'] = (
                    import_summary['total_import_production_cost'] +
                    import_summary['total_transport_cost']
            )
        else:
            # If no transport cost data, use just the production cost
            import_summary['total_transport_cost'] = 0
            import_summary['total_transport_opex'] = 0
            import_summary['total_import_cost'] = import_summary['total_import_production_cost']
    else:
        # Create empty import summary if no import data
        import_summary = pd.DataFrame(
            columns=['location', 'commodity', 'time_operation',
                     'total_import_flow', 'total_import_cost',
                     'total_transport_cost', 'total_transport_opex']
        )

    # 4. Combine domestic and import data
    total_cost = domestic_df.merge(
        import_summary,
        on=['location', 'commodity', 'time_operation'],
        how='outer'
    )

    # Fill NaN values with 0
    fill_cols = ['total_production', 'total_production_cost', 'opex_cost',
                 'total_import_flow', 'total_import_cost',
                 'total_transport_cost', 'total_transport_opex']

    for col in fill_cols:
        if col in total_cost.columns:
            total_cost[col] = total_cost[col].fillna(0)

    # Calculate total output (domestic + import)
    total_cost['total_output'] = total_cost['total_production'] + total_cost['total_import_flow']

    # Calculate ratios
    total_cost['domestic_ratio'] = np.where(
        total_cost['total_output'] == 0,
        0,
        total_cost['total_production'] / total_cost['total_output']
    )

    total_cost['import_ratio'] = np.where(
        total_cost['total_output'] == 0,
        0,
        total_cost['total_import_flow'] / total_cost['total_output']
    )

    # Calculate weighted cost
    total_cost['weighted_cost'] = (
            (total_cost['domestic_ratio'] * total_cost['total_production_cost']) +
            (total_cost['import_ratio'] * total_cost['total_import_cost'])
    )

    # Calculate weighted cost per unit
    total_cost['weighted_cost_per_unit'] = np.where(
        total_cost['total_output'] == 0,
        np.nan,
        total_cost['weighted_cost'] / total_cost['total_output']
    )

    # Calculate ratio of transport cost to production cost
    if 'total_transport_opex' in total_cost.columns:
        total_cost['transport_to_production_ratio'] = np.where(
            total_cost['total_production_cost'] == 0,
            np.nan,
            total_cost['total_transport_opex'] / total_cost['total_production_cost']
        )

    # Create a pivot table with location-commodity as index and timesteps as columns
    pivot_table = total_cost.pivot_table(
        index=['location', 'commodity'],
        columns='time_operation',
        values='weighted_cost_per_unit'
    )

    # Rename columns to timestep format
    pivot_table.columns = [f'timestep_{t}' for t in pivot_table.columns]

    # Add average column
    pivot_table['average'] = pivot_table.mean(axis=1, skipna=True)

    # Reset index for better output format
    pivot_table = pivot_table.reset_index()

    # Create calculation summary by aggregating by location and timestep
    # This removes the commodity dimension and gives us a country-level view
    calc_summary = total_cost.groupby(['location', 'time_operation']).agg({
        'opex_cost': 'sum',  # Local production OPEX
        'total_production': 'sum',  # Local production output
        'total_transport_opex': 'sum',  # Transport OPEX
        'total_import_flow': 'sum'  # Transport output (imports)
    }).reset_index()

    # Rename columns for clarity
    calc_summary = calc_summary.rename(columns={
        'opex_cost': 'local_production_opex',
        'total_production': 'local_production_output',
        'total_transport_opex': 'transport_opex',
        'total_import_flow': 'transport_output'
    })

    # Calculate total OPEX and total output
    calc_summary['total_opex'] = calc_summary['local_production_opex'] + calc_summary['transport_opex']
    calc_summary['total_output'] = calc_summary['local_production_output'] + calc_summary['transport_output']

    # Calculate OPEX per unit
    calc_summary['total_opex_per_unit'] = np.where(
        calc_summary['total_output'] == 0,
        np.nan,
        calc_summary['total_opex'] / calc_summary['total_output']
    )

    # Calculate ratios
    calc_summary['transport_to_production_opex_ratio'] = np.where(
        calc_summary['local_production_opex'] == 0,
        np.nan,
        calc_summary['transport_opex'] / calc_summary['local_production_opex']
    )

    calc_summary['transport_to_production_output_ratio'] = np.where(
        calc_summary['local_production_output'] == 0,
        np.nan,
        calc_summary['transport_output'] / calc_summary['local_production_output']
    )

    # Create TOTAL row
    total_row = calc_summary.groupby('time_operation').agg({
        'local_production_opex': 'sum',
        'local_production_output': 'sum',
        'transport_opex': 'sum',
        'transport_output': 'sum',
        'total_opex': 'sum',
        'total_output': 'sum'
    }).reset_index()

    total_row['location'] = 'TOTAL'

    # Calculate per-unit and ratios for the TOTAL row
    total_row['total_opex_per_unit'] = np.where(
        total_row['total_output'] == 0,
        np.nan,
        total_row['total_opex'] / total_row['total_output']
    )

    total_row['transport_to_production_opex_ratio'] = np.where(
        total_row['local_production_opex'] == 0,
        np.nan,
        total_row['transport_opex'] / total_row['local_production_opex']
    )

    total_row['transport_to_production_output_ratio'] = np.where(
        total_row['local_production_output'] == 0,
        np.nan,
        total_row['transport_output'] / total_row['local_production_output']
    )

    # Combine with the main summary
    calculation_summary = pd.concat([calc_summary, total_row], ignore_index=True)

    # Create pivot tables for the ratios and per-unit values
    opex_ratio_pivot = calculation_summary.pivot_table(
        index='location',
        columns='time_operation',
        values='transport_to_production_opex_ratio'
    )

    output_ratio_pivot = calculation_summary.pivot_table(
        index='location',
        columns='time_operation',
        values='transport_to_production_output_ratio'
    )

    opex_per_unit_pivot = calculation_summary.pivot_table(
        index='location',
        columns='time_operation',
        values='total_opex_per_unit'
    )

    # If output directory is provided, save the calculation summary files
    if output_dir is not None:
        calculation_summary.to_csv(output_dir / "calculation_summary.csv", index=False)
        opex_ratio_pivot.reset_index().to_csv(output_dir / "opex_ratio_by_country.csv", index=False)
        output_ratio_pivot.reset_index().to_csv(output_dir / "output_ratio_by_country.csv", index=False)
        opex_per_unit_pivot.reset_index().to_csv(output_dir / "total_opex_per_unit_by_country.csv", index=False)

        print(f"Calculation summary files saved to {output_dir}")

    print(f"Total cost calculation: {len(total_cost)} location-commodity-timestep combinations")
    print(f"Calculation summary: {len(calculation_summary)} location-timestep combinations")

    return total_cost, pivot_table, calculation_summary, opex_ratio_pivot, output_ratio_pivot, opex_per_unit_pivot


def calculate_lcohp(total_cost_df, scenario=None):
    """
    Calculate the Levelized Cost of Heat Pump (LCOHP) by applying conversion
    factors to the weighted cost per unit of each material.

    Parameters:
    -----------
    total_cost_df : DataFrame
        DataFrame with cost data
    scenario : str, optional
        Scenario name to include in the output

    Returns:
    --------
    tuple: levelized_df, lcohp_df, pivot_table
    """
    print("\nCalculating Levelized Cost of Heat Pump (LCOHP)...")

    # Define conversion factors in kg/kW (or kW/kW for components)
    conversion_factors = {
        'Aluminium': 4.0375,  # kg/kW
        'Bauxite': 16.15,  # kg/kW
        'Compressor': 1,  # kW/kW
        'Copper': 5.51666667,  # kg/kW
        'Copper_ore': 5.51666667,  # kg/kW
        'HEX': 1,  # kW/kW
        'HP': 1,  # kW/kW
        'Iron': 29.89,  # kg/kW
        'Nickel': 0.605625,  # kg/kW
        'Steel': 18.68125,  # kg/kW
        'Refrigerant': 0.56875,  # kg/kW
    }

    # Create a copy of the input dataframe
    df = total_cost_df.copy()

    # Create a dataframe for commodity-level calculations
    commodity_costs = []

    # Process each row to calculate commodity-specific levelized costs
    for _, row in df.iterrows():
        location = row['location']
        time_operation = row['time_operation']
        commodity = row['commodity']
        weighted_cost = row['weighted_cost_per_unit']

        # Skip rows with missing or zero cost
        if pd.isna(weighted_cost) or weighted_cost == 0:
            continue

        # Check if commodity has a conversion factor
        if commodity not in conversion_factors:
            print(f"Warning: No conversion factor for {commodity}, skipping")
            continue

        # Get the conversion factor
        conversion_factor = conversion_factors[commodity]

        # Calculate levelized cost based on commodity type
        if commodity in ['HP', 'Compressor', 'HEX']:
            # Component costs are in €/GW, convert to €/kW
            # 1 GW = 1,000,000 kW, cost is in k€, so divide by 1000
            levelized_cost = weighted_cost / 1_000 * conversion_factor
        else:
            # Material costs are in k€/kiloton, convert to €/kW
            # 1 kiloton = 1,000,000 kg, cost is in k€, so divide by 1000
            levelized_cost = weighted_cost / 1_000 * conversion_factor

        # Store the calculation
        commodity_costs.append({
            'location': location,
            'time_operation': time_operation,
            'commodity': commodity,
            'weighted_cost_per_unit': weighted_cost,
            'conversion_factor': conversion_factor,
            'levelized_cost_per_kw': levelized_cost,
            'scenario': scenario  # Add scenario if provided
        })

    # Create dataframe from commodity calculations
    if commodity_costs:
        levelized_df = pd.DataFrame(commodity_costs)

        # Sum levelized costs by location and timestep
        lcohp_df = levelized_df.groupby(['location', 'time_operation']).agg(
            lcohp=('levelized_cost_per_kw', 'sum')
        ).reset_index()

        # Add scenario if provided
        if scenario is not None:
            lcohp_df['scenario'] = scenario

        # Create a pivot table with locations as rows and timesteps as columns
        pivot_table = lcohp_df.pivot(
            index='location',
            columns='time_operation',
            values='lcohp'
        )

        # Rename columns to timestep format
        pivot_table.columns = [f'timestep_{t}' for t in pivot_table.columns]

        # Add average column
        pivot_table['average'] = pivot_table.mean(axis=1, skipna=True)

        # Reset index for better output format
        pivot_table = pivot_table.reset_index()

        print(f"LCOHP calculation: {len(lcohp_df)} location-timestep combinations")

        return levelized_df, lcohp_df, pivot_table
    else:
        print("Warning: No valid commodity costs found for LCOHP calculation")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def calculate_transport_production_ratio(total_cost_df, production_cost_df, transport_cost_df):
    """
    Calculate the ratio of transport cost to production cost for each commodity
    aggregated across all locations
    """
    print("\nCalculating transport-to-production ratio for each commodity...")

    # Prepare production cost data - group by commodity and time_operation only
    prod_df = production_cost_df[['commodity', 'time_operation', 'total_production_cost']].copy()
    prod_df = prod_df.groupby(['commodity', 'time_operation']).agg(
        total_prod_cost=('total_production_cost', 'sum')
    ).reset_index()

    # Prepare transport cost data - group by commodity and time_operation only
    trans_df = transport_cost_df[['commodity', 'time_operation', 'total_transport_cost']].copy()
    trans_df = trans_df.groupby(['commodity', 'time_operation']).agg(
        total_trans_cost=('total_transport_cost', 'sum')
    ).reset_index()

    # Merge production and transport costs
    ratio_df = prod_df.merge(
        trans_df,
        on=['commodity', 'time_operation'],
        how='outer'
    )

    # Fill NaN values with 0
    ratio_df['total_prod_cost'] = ratio_df['total_prod_cost'].fillna(0)
    ratio_df['total_trans_cost'] = ratio_df['total_trans_cost'].fillna(0)

    # Calculate the ratio of transport cost to production cost
    ratio_df['transport_production_ratio'] = np.where(
        ratio_df['total_prod_cost'] == 0,
        np.nan,
        ratio_df['total_trans_cost'] / ratio_df['total_prod_cost']
    )

    # Create a pivot table with commodity as index and timesteps as columns
    pivot_table = ratio_df.pivot_table(
        index='commodity',
        columns='time_operation',
        values='transport_production_ratio'
    )

    # Rename columns to timestep format
    pivot_table.columns = [f'timestep_{t}' for t in pivot_table.columns]

    # Add average column
    pivot_table['average'] = pivot_table.mean(axis=1, skipna=True)

    # Reset index for better output format
    pivot_table = pivot_table.reset_index()

    print(f"Transport-to-production ratio calculation: {len(ratio_df)} commodity-timestep combinations")

    return pivot_table


def generate_comprehensive_summary(production_cost_df, transport_cost_df, total_cost_df, import_details_df):
    """
    Generate a comprehensive summary with all key metrics by country, commodity, and timestep.

    This includes production and import details, costs, and ratios as requested.

    Parameters:
    -----------
    production_cost_df : DataFrame
        Production cost information
    transport_cost_df : DataFrame
        Transport cost information
    total_cost_df : DataFrame
        Total cost of goods information
    import_details_df : DataFrame
        Import details information

    Returns:
    --------
    DataFrame: comprehensive_summary_df with all requested metrics
    """
    print("\nGenerating comprehensive summary...")

    # Start with the total cost dataframe as the base
    summary_df = total_cost_df[['location', 'commodity', 'time_operation',
                                'total_production', 'total_production_cost',
                                'production_cost_per_unit', 'opex_cost',
                                'total_import_flow', 'total_import_cost',
                                'weighted_cost_per_unit', 'weighted_cost',
                                'domestic_ratio', 'import_ratio']].copy()

    # Rename columns for clarity in the final summary
    summary_df = summary_df.rename(columns={
        'total_production': 'amount_produced',
        'opex_cost': 'opex_production',
        'total_import_flow': 'amount_imported',
        'weighted_cost': 'total_cost',
        'domestic_ratio': 'production_ratio',
        'import_ratio': 'import_ratio'
    })

    # Add transport OPEX from transport costs dataframe
    if not transport_cost_df.empty:
        transport_opex_df = transport_cost_df[
            ['location', 'commodity', 'time_operation', 'total_transport_opex']].copy()
        transport_opex_df = transport_opex_df.rename(columns={'total_transport_opex': 'opex_transport'})

        # Merge with the main summary
        summary_df = summary_df.merge(
            transport_opex_df,
            on=['location', 'commodity', 'time_operation'],
            how='left'
        )
    else:
        # Add empty column if no transport data
        summary_df['opex_transport'] = 0

    # Add per-unit import cost
    summary_df['per_unit_import_cost'] = np.where(
        summary_df['amount_imported'] == 0,
        np.nan,
        summary_df['total_import_cost'] / summary_df['amount_imported']
    )

    # Calculate production to import ratio
    summary_df['production_to_import_ratio'] = np.where(
        summary_df['amount_imported'] == 0,
        np.nan,
        summary_df['amount_produced'] / summary_df['amount_imported']
    )

    # Fill NaN values with 0 for numeric columns
    numeric_cols = ['amount_produced', 'opex_production', 'total_production_cost',
                    'amount_imported', 'total_import_cost', 'opex_transport',
                    'total_cost']

    for col in numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].fillna(0)

    # Sort by location, commodity, and timestep
    summary_df = summary_df.sort_values(['location', 'commodity', 'time_operation'])

    # Reorder columns to match requested order
    ordered_columns = [
        'location', 'commodity', 'time_operation',
        'opex_production', 'amount_produced', 'production_cost_per_unit',
        'opex_transport', 'amount_imported', 'per_unit_import_cost',
        'weighted_cost_per_unit', 'total_cost', 'production_to_import_ratio'
    ]

    # Ensure all columns exist
    for col in ordered_columns:
        if col not in summary_df.columns:
            summary_df[col] = np.nan

    # Select and order columns
    comprehensive_summary_df = summary_df[ordered_columns]

    print(f"Comprehensive summary generated: {len(comprehensive_summary_df)} rows")

    return comprehensive_summary_df


def compare_scenarios_lcohp(lcohp_data_by_scenario, output_dir):
    """
    Create a comparison of LCOHP values across scenarios

    Parameters:
    -----------
    lcohp_data_by_scenario : dict
        Dictionary with scenario names as keys and lcohp_df as values
    output_dir : Path
        Directory to save the comparison file
    """
    print("\nCreating LCOHP scenario comparison...")

    # Create an empty list to hold all scenario data
    all_scenarios_data = []

    # Process each scenario's LCOHP data
    for scenario, lcohp_df in lcohp_data_by_scenario.items():
        if not lcohp_df.empty:
            # Make a copy to avoid modifying the original
            scenario_data = lcohp_df.copy()

            # Add scenario column if not already present
            if 'scenario' not in scenario_data.columns:
                scenario_data['scenario'] = scenario

            # Add to the list
            all_scenarios_data.append(scenario_data)

    # If we have data, combine it all
    if all_scenarios_data:
        # Concatenate all scenario data
        comparison_df = pd.concat(all_scenarios_data, ignore_index=True)

        # Create a pivot table with scenarios as columns and locations as rows
        # Average LCOHP across timesteps for each location-scenario combination
        pivot_comparison = comparison_df.groupby(['location', 'scenario']).agg({
            'lcohp': 'mean'
        }).reset_index()

        # Reshape to have scenarios as columns
        scenario_comparison = pivot_comparison.pivot(
            index='location',
            columns='scenario',
            values='lcohp'
        ).reset_index()

        # Save the comparison file
        comparison_path = output_dir / "lcohp_scenario_comparison.csv"
        scenario_comparison.to_csv(comparison_path, index=False)
        print(f"LCOHP scenario comparison saved to {comparison_path}")

        return scenario_comparison
    else:
        print("Warning: No LCOHP data available for scenario comparison")
        return pd.DataFrame()


def save_results(output_dir, filename, df, index=False):
    """
    Save dataframe to CSV file
    """
    output_path = output_dir / filename
    df.to_csv(output_path, index=index)
    print(f"Saved: {output_path}")


def main():
    """Main function to execute the analysis"""
    # File paths
    flow_file = './parameter_results/flow_conversion_output/flow_conversion_output_scenarios.csv'
    opex_file = './parameter_results/cost_opex_yearly/cost_opex_yearly_scenarios.csv'
    capex_file = './parameter_results/cost_capex/cost_capex_scenarios.csv'
    transport_file = "./parameter_results/flow_transport/flow_transport_scenarios.csv"

    # Output directory
    output_dir = Path(
        "./LCOHP")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting cost analysis...")

    # 1. Load data files and get scenarios
    opex_df, capex_df, production_output_df, transport_flow_df, scenarios = load_data_files(
        flow_file, opex_file, capex_file, transport_file
    )

    # Dictionary to store LCOHP data by scenario
    lcohp_data_by_scenario = {}

    # Dictionary to store comprehensive summary data by scenario
    summary_data_by_scenario = {}

    # Process each scenario separately
    if len(scenarios) > 1:
        print(f"\nProcessing {len(scenarios)} scenarios: {scenarios}")

        # Process each scenario separately
        for scenario in scenarios:
            print(f"\n--- Processing scenario: {scenario} ---")

            # Filter data for this scenario
            scenario_opex = opex_df[opex_df['scenario'] == scenario].copy()
            scenario_capex = capex_df[capex_df['scenario'] == scenario].copy()
            scenario_production = production_output_df[production_output_df['scenario'] == scenario].copy()
            scenario_transport = transport_flow_df[transport_flow_df['scenario'] == scenario].copy()

            # Create scenario subfolder
            scenario_dir = output_dir / scenario
            scenario_dir.mkdir(exist_ok=True)

            # Run analysis for this scenario
            production_cost_df = calculate_production_cost_per_unit(
                scenario_production, scenario_opex, scenario_capex
            )

            transport_cost_df, import_details_df, transport_cost_pivot = calculate_transport_cost(
                scenario_transport, scenario_opex, scenario_capex
            )

            total_cost_df, total_cost_pivot, calculation_summary_df, opex_ratio_pivot, output_ratio_pivot, opex_per_unit_pivot = calculate_total_cost_of_goods(
                production_cost_df, transport_cost_df, import_details_df, scenario_dir
            )

            # Calculate LCOHP for this scenario
            levelized_details_df, lcohp_df, lcohp_pivot = calculate_lcohp(total_cost_df, scenario)

            # Store LCOHP data for comparison
            lcohp_data_by_scenario[scenario] = lcohp_df

            # Calculate transport-to-production ratio
            transport_ratio_pivot = calculate_transport_production_ratio(
                total_cost_df, production_cost_df, transport_cost_df
            )

            # Generate comprehensive summary for this scenario
            comprehensive_summary_df = generate_comprehensive_summary(
                production_cost_df, transport_cost_df, total_cost_df, import_details_df
            )

            # Store summary data for potential comparison
            summary_data_by_scenario[scenario] = comprehensive_summary_df

            # Add scenario column if not already present
            if 'scenario' not in comprehensive_summary_df.columns:
                comprehensive_summary_df['scenario'] = scenario

            # Create pivot table with the renamed columns for production cost
            production_cost_pivot = production_cost_df.pivot_table(
                index=['location', 'commodity'],
                columns='time_operation',
                values='production_cost_per_unit'
            )
            production_cost_pivot.columns = [f'timestep_{t}' for t in production_cost_pivot.columns]
            production_cost_pivot['average'] = production_cost_pivot.mean(axis=1, skipna=True)
            production_cost_pivot = production_cost_pivot.reset_index()

            # Save results for this scenario
            save_results(scenario_dir, "production_cost_details.csv", production_cost_df)
            save_results(scenario_dir, "production_cost_per_unit.csv", production_cost_pivot)
            save_results(scenario_dir, "transport_cost_details.csv", transport_cost_df)
            save_results(scenario_dir, "transport_cost_per_unit.csv", transport_cost_pivot)
            save_results(scenario_dir, "import_details.csv", import_details_df)
            save_results(scenario_dir, "total_cost_details.csv", total_cost_df)
            save_results(scenario_dir, "total_cost_per_unit.csv", total_cost_pivot)
            save_results(scenario_dir, "comprehensive_summary.csv", comprehensive_summary_df)

            if not levelized_details_df.empty:
                save_results(scenario_dir, "lcohp_commodity_details.csv", levelized_details_df)
                save_results(scenario_dir, "lcohp_by_location_timestep.csv", lcohp_df)
                save_results(scenario_dir, "lcohp_summary.csv", lcohp_pivot)

            save_results(scenario_dir, "transport_production_ratio_by_commodity.csv", transport_ratio_pivot)

            print(f"--- Completed scenario: {scenario} ---")

        # Create combined summary for all scenarios
        if summary_data_by_scenario:
            all_scenarios_summary = pd.concat(summary_data_by_scenario.values(), ignore_index=True)
            save_results(output_dir, "all_scenarios_comprehensive_summary.csv", all_scenarios_summary)
            print(
                f"Combined summary for all scenarios saved to {output_dir / 'all_scenarios_comprehensive_summary.csv'}")

    else:
        # Process without scenarios (base case)
        print("\nProcessing base case (no scenarios detected)")

        # 2. Calculate production costs
        production_cost_df = calculate_production_cost_per_unit(
            production_output_df, opex_df, capex_df
        )

        # 3. Calculate transport costs
        transport_cost_df, import_details_df, transport_cost_pivot = calculate_transport_cost(
            transport_flow_df, opex_df, capex_df
        )

        # 4. Calculate total cost of goods - now includes calculation summary
        total_cost_df, total_cost_pivot, calculation_summary_df, opex_ratio_pivot, output_ratio_pivot, opex_per_unit_pivot = calculate_total_cost_of_goods(
            production_cost_df, transport_cost_df, import_details_df, output_dir
        )

        # 5. Calculate LCOHP
        levelized_details_df, lcohp_df, lcohp_pivot = calculate_lcohp(total_cost_df, 'base')

        # Store LCOHP data for comparison (even though we only have one scenario)
        lcohp_data_by_scenario['base'] = lcohp_df

        # 6. Calculate transport-to-production ratio
        transport_ratio_pivot = calculate_transport_production_ratio(
            total_cost_df, production_cost_df, transport_cost_df
        )

        # 7. Generate comprehensive summary
        comprehensive_summary_df = generate_comprehensive_summary(
            production_cost_df, transport_cost_df, total_cost_df, import_details_df
        )

        # Create pivot table with the renamed columns
        production_cost_pivot = production_cost_df.pivot_table(
            index=['location', 'commodity'],
            columns='time_operation',
            values='production_cost_per_unit'
        )
        production_cost_pivot.columns = [f'timestep_{t}' for t in production_cost_pivot.columns]
        production_cost_pivot['average'] = production_cost_pivot.mean(axis=1, skipna=True)
        production_cost_pivot = production_cost_pivot.reset_index()

        # Save results
        print("\nSaving results...")

        # Save production costs
        save_results(output_dir, "production_cost_details.csv", production_cost_df)
        save_results(output_dir, "production_cost_per_unit.csv", production_cost_pivot)

        # Save transport costs
        save_results(output_dir, "transport_cost_details.csv", transport_cost_df)
        save_results(output_dir, "transport_cost_per_unit.csv", transport_cost_pivot)
        save_results(output_dir, "import_details.csv", import_details_df)

        # Save total costs
        save_results(output_dir, "total_cost_details.csv", total_cost_df)
        save_results(output_dir, "total_cost_per_unit.csv", total_cost_pivot)

        # Save comprehensive summary
        save_results(output_dir, "comprehensive_summary.csv", comprehensive_summary_df)

        # Save LCOHP results
        if not levelized_details_df.empty:
            save_results(output_dir, "lcohp_commodity_details.csv", levelized_details_df)
            save_results(output_dir, "lcohp_by_location_timestep.csv", lcohp_df)
            save_results(output_dir, "lcohp_summary.csv", lcohp_pivot)

        # Save transport-to-production ratio results (summary only)
        save_results(output_dir, "transport_production_ratio_by_commodity.csv", transport_ratio_pivot)

    # Create and save scenario comparison
    scenario_comparison = compare_scenarios_lcohp(lcohp_data_by_scenario, output_dir)

    print("\nAnalysis complete!")

    # Display summary statistics for the scenario comparison
    if not scenario_comparison.empty:
        print("\nLCOHP Scenario Comparison (first 5 rows):")
        print(scenario_comparison.head())


if __name__ == "__main__":
    main()