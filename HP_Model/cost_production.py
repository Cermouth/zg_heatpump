import pandas as pd
import numpy as np
from pathlib import Path


def load_and_prepare_data(flow_file, opex_file, capex_file):
    """Load and prepare data from CSV files"""
    # Read CSV files
    print(f"Reading data files...")
    flow_df = pd.read_csv(flow_file)
    opex_df = pd.read_csv(opex_file)
    capex_df = pd.read_csv(capex_file)

    # Focus on base scenario (value_scenario_)
    flow_df = flow_df[['technology', 'carrier', 'node', 'time_operation', 'value_scenario_']]
    opex_df = opex_df[['technology', 'location', 'year', 'value_scenario_']]
    capex_df = capex_df[['technology', 'location', 'year', 'value_scenario_']]

    # Rename columns for clarity
    flow_df = flow_df.rename(columns={'value_scenario_': 'flow', 'node': 'location'})
    opex_df = opex_df.rename(columns={'value_scenario_': 'opex', 'year': 'time_operation'})
    capex_df = capex_df.rename(columns={'value_scenario_': 'capex', 'year': 'time_operation'})

    # Identify transport technologies - looking for "transport" in technology name
    is_transport = flow_df['technology'].str.contains('transport', case=False, na=False)

    # Also check in OPEX file to identify any transport technologies not in flow_df
    opex_transport_techs = opex_df[opex_df['technology'].str.contains('transport', case=False, na=False)][
        'technology'].unique()

    print(f"\nTransport technologies in flow data: {sum(is_transport)} rows")
    print(f"Transport technologies in OPEX data: {len(opex_transport_techs)} unique technologies")

    if len(opex_transport_techs) > 0 and sum(is_transport) == 0:
        print("WARNING: Transport technologies found in OPEX but not in flow data!")
        print("Example transport technologies in OPEX:")
        for tech in opex_transport_techs[:5]:
            print(f"  - {tech}")

        # Check if we have location pairs in the OPEX data (e.g., AUS-JPN)
        location_pairs = opex_df[opex_df['technology'].isin(opex_transport_techs)]['location'].str.contains('-',
                                                                                                            na=False)

        if location_pairs.any():
            print("\nFound location pairs in OPEX data for transport technologies!")
            print("Example location pairs:")
            for loc in opex_df[opex_df['technology'].isin(opex_transport_techs)]['location'].unique()[:5]:
                print(f"  - {loc}")

    # Split into production and transport dataframes
    prod_flow_df = flow_df[~is_transport].copy()
    transport_flow_df = flow_df[is_transport].copy()

    return prod_flow_df, transport_flow_df, opex_df, capex_df


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

    print(f"Production cost calculation: {len(merged_df)} rows processed")

    return merged_df


def extract_commodity_from_transport(tech_name):
    """Extract commodity name from transport technology name"""
    return tech_name.replace('_transport', '').replace('transport_', '')


def calculate_transport_costs(transport_flow_df, opex_df, capex_df):
    """Calculate transport costs using OPEX and CAPEX data for transport technologies"""
    # Identify transport technologies in the OPEX data
    transport_techs = opex_df['technology'].str.contains('transport', case=False, na=False)
    transport_opex = opex_df[transport_techs].copy()

    # Identify transport technologies in the CAPEX data
    transport_techs_capex = capex_df['technology'].str.contains('transport', case=False, na=False)
    transport_capex = capex_df[transport_techs_capex].copy()

    print(
        f"\nTransport OPEX data: {len(transport_opex)} rows, {transport_opex['technology'].nunique()} unique technologies")
    print(
        f"Transport CAPEX data: {len(transport_capex)} rows, {transport_capex['technology'].nunique()} unique technologies")

    # If we have transport data in flow_df, merge it with the cost data
    if len(transport_flow_df) > 0:
        # Merge with flow data
        transport_merged = transport_flow_df.merge(
            transport_opex,
            on=['technology', 'location', 'time_operation'],
            how='left'
        ).merge(
            transport_capex,
            on=['technology', 'location', 'time_operation'],
            how='left'
        )
    else:
        # If no transport flows in flow_df, we'll use OPEX data directly
        # This assumes transport costs exist even if flows aren't explicit
        transport_merged = transport_opex.merge(
            transport_capex,
            on=['technology', 'location', 'time_operation'],
            how='left'
        )
        # Add a placeholder for flow if it doesn't exist
        if 'flow' not in transport_merged.columns:
            print("WARNING: No transport flow data found. Using placeholder values.")
            transport_merged['flow'] = 1.0  # Placeholder - adjust as needed

    # Fill NaN values with 0 for cost calculations
    transport_merged['opex'] = transport_merged['opex'].fillna(0)
    transport_merged['capex'] = transport_merged['capex'].fillna(0)

    # Calculate total transport cost
    transport_merged['transport_cost'] = transport_merged['opex'] + transport_merged['capex']

    # Also keep track of opex separately for ratio calculations
    transport_merged['transport_opex'] = transport_merged['opex']

    # Extract the commodity being transported from the technology name
    transport_merged['commodity'] = transport_merged['technology'].apply(extract_commodity_from_transport)

    # Extract source and destination from location field (assumes format like "AUS-JPN")
    if transport_merged['location'].str.contains('-').any():
        transport_merged['source'] = transport_merged['location'].str.split('-').str[0]
        transport_merged['destination'] = transport_merged['location'].str.split('-').str[1]

    # Summarize transport costs per commodity and timestep
    transport_summary = transport_merged.groupby(['commodity', 'time_operation']).agg(
        total_transport_cost=('transport_cost', 'sum'),
        total_transport_opex=('transport_opex', 'sum'),
        total_flow=('flow', 'sum')
    ).reset_index()

    # Calculate transport cost per unit where flow is not zero
    transport_summary['transport_cost_per_unit'] = np.where(
        transport_summary['total_flow'] == 0,
        np.nan,
        transport_summary['total_transport_cost'] / transport_summary['total_flow']
    )

    # Also calculate opex per unit
    transport_summary['transport_opex_per_unit'] = np.where(
        transport_summary['total_flow'] == 0,
        np.nan,
        transport_summary['total_transport_opex'] / transport_summary['total_flow']
    )

    print(f"Transport cost calculation: {len(transport_merged)} rows processed")
    print(f"Transport summary: {len(transport_summary)} commodity-timestep combinations")

    return transport_merged, transport_summary


def calculate_cost_ratio(prod_results, transport_summary):
    """Calculate ratio of transport costs to production costs"""
    # Extract the commodity from production technologies to match with transport commodities
    # This maps production technologies to their corresponding commodity
    # For example: "Aluminium_production" → "Aluminium"

    # Create a mapping for technologies to commodities
    # First, use the carrier field if available
    if 'carrier' in prod_results.columns:
        prod_results['commodity'] = prod_results['carrier']
    else:
        # Otherwise, try to extract from technology name
        # This simplistic approach just removes "_production" suffix
        prod_results['commodity'] = prod_results['technology'].str.replace('_production', '').str.replace('_mining', '')

    # Group production costs by commodity and timestep, including OPEX separately
    prod_summary = prod_results.groupby(['commodity', 'time_operation']).agg(
        total_prod_cost=('total_cost', 'sum'),
        total_prod_opex=('opex_cost', 'sum')
    ).reset_index()

    # Merge production and transport summaries on commodity and timestep
    cost_ratio_df = prod_summary.merge(
        transport_summary[['commodity', 'time_operation', 'total_transport_cost', 'total_transport_opex']],
        on=['commodity', 'time_operation'],
        how='left'
    )

    # Fill NaN values with 0 for transport costs (where there is no transport)
    cost_ratio_df['total_transport_cost'] = cost_ratio_df['total_transport_cost'].fillna(0)
    cost_ratio_df['total_transport_opex'] = cost_ratio_df['total_transport_opex'].fillna(0)

    # Calculate ratio of total transport cost to total production cost (includes both OPEX and CAPEX)
    cost_ratio_df['transport_to_production_ratio'] = np.where(
        cost_ratio_df['total_prod_cost'] == 0,
        np.nan,
        cost_ratio_df['total_transport_cost'] / cost_ratio_df['total_prod_cost']
    )

    # Calculate ratio of transport OPEX to production OPEX (OPEX only)
    cost_ratio_df['transport_opex_to_production_opex_ratio'] = np.where(
        cost_ratio_df['total_prod_opex'] == 0,
        np.nan,
        cost_ratio_df['total_transport_opex'] / cost_ratio_df['total_prod_opex']
    )

    print(f"Cost ratio calculation: {len(cost_ratio_df)} commodity-timestep combinations")

    return cost_ratio_df


def create_final_format(results_df):
    """Create final format for production costs"""
    # Create a unique identifier for technology-location combination
    results_df['tech_location'] = results_df['location'] + '-' + results_df['technology']

    # Pivot the table to get timesteps as columns
    pivot_table = results_df.pivot(
        index='tech_location',
        columns='time_operation',
        values='cost_per_unit'
    )

    # Rename columns to timestep format
    pivot_table.columns = [f'timestep_{t}' for t in pivot_table.columns]

    # Add average column, ignoring NaN values
    pivot_table['average'] = pivot_table.mean(axis=1)

    # Replace NaN with '-'
    pivot_table = pivot_table.fillna('-')

    return pivot_table


def create_transport_summary_format(transport_summary):
    """Create final format for transport costs"""
    # If transport_summary is empty, return empty dataframe
    if len(transport_summary) == 0:
        print("WARNING: No transport data available for summary format.")
        return pd.DataFrame()

    # Pivot the table to get timesteps as columns
    pivot_table = transport_summary.pivot(
        index='commodity',
        columns='time_operation',
        values='total_transport_cost'
    )

    # Rename columns to timestep format
    pivot_table.columns = [f'transport_cost_timestep_{t}' for t in pivot_table.columns]

    # Add average column, ignoring NaN values
    pivot_table['average_transport_cost'] = pivot_table.mean(axis=1)

    # Also pivot the transport cost per unit
    per_unit_pivot = transport_summary.pivot(
        index='commodity',
        columns='time_operation',
        values='transport_cost_per_unit'
    )

    # Rename columns
    per_unit_pivot.columns = [f'transport_cost_per_unit_timestep_{t}' for t in per_unit_pivot.columns]

    # Add average column
    per_unit_pivot['average_transport_cost_per_unit'] = per_unit_pivot.mean(axis=1)

    # Merge the two pivot tables
    result = pd.concat([pivot_table, per_unit_pivot], axis=1)

    # Replace NaN with '-'
    result = result.fillna('-')

    return result


def create_ratio_summary_format(ratio_df):
    """Create final format for transport-to-production cost ratios"""
    # If ratio_df is empty, return empty dataframe
    if len(ratio_df) == 0:
        print("WARNING: No ratio data available for summary format.")
        return pd.DataFrame(), pd.DataFrame()

    # 1. Create pivot table for total cost ratio (OPEX + CAPEX)
    total_pivot = ratio_df.pivot(
        index='commodity',
        columns='time_operation',
        values='transport_to_production_ratio'
    )

    # Rename columns to timestep format
    total_pivot.columns = [f'transport_ratio_timestep_{t}' for t in total_pivot.columns]

    # Add average column, ignoring NaN values
    total_pivot['average_ratio'] = total_pivot.mean(axis=1)

    # Replace NaN with '-'
    total_pivot = total_pivot.fillna('-')

    # 2. Create pivot table for OPEX-only ratio
    opex_pivot = ratio_df.pivot(
        index='commodity',
        columns='time_operation',
        values='transport_opex_to_production_opex_ratio'
    )

    # Rename columns to timestep format
    opex_pivot.columns = [f'transport_opex_ratio_timestep_{t}' for t in opex_pivot.columns]

    # Add average column, ignoring NaN values
    opex_pivot['average_opex_ratio'] = opex_pivot.mean(axis=1)

    # Replace NaN with '-'
    opex_pivot = opex_pivot.fillna('-')

    return total_pivot, opex_pivot


def process_trade_data(transport_file, prod_results):
    """
    Process trade data from transport file, calculating imports and exports by country and commodity.

    Parameters:
    -----------
    transport_file : str
        Path to the transport flow file
    prod_results : DataFrame
        Production costs per unit for each country

    Returns:
    --------
    tuple of DataFrames
        (import_flows, import_summary)
    """
    try:
        # Add commodity to prod_results if not present
        if 'commodity' not in prod_results.columns:
            if 'carrier' in prod_results.columns:
                prod_results['commodity'] = prod_results['carrier']
            else:
                prod_results['commodity'] = prod_results['technology'].str.replace('_production', '').str.replace(
                    '_mining', '')

        # Read transport data
        transport_df = pd.read_csv(transport_file)

        # Check for required columns
        required_cols = ['technology', 'edge', 'time_operation', 'value_scenario_']
        if not all(col in transport_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in transport_df.columns]
            print(f"WARNING: Missing required columns in transport file: {missing_cols}")
            return pd.DataFrame(), pd.DataFrame()

        # Filter for transport technologies
        is_transport = transport_df['technology'].str.contains('transport', case=False, na=False)
        transport_flows = transport_df[is_transport].copy()

        if transport_flows.empty:
            print("WARNING: No transport technologies found in transport file")
            return pd.DataFrame(), pd.DataFrame()

        # Split edge into exporter and importer
        if not transport_flows['edge'].str.contains('-').any():
            print("WARNING: Edge column does not contain country pairs in 'EXPORTER-IMPORTER' format")
            return pd.DataFrame(), pd.DataFrame()

        transport_flows[['exporter', 'importer']] = transport_flows['edge'].str.split('-', expand=True)

        # Extract commodity from technology
        transport_flows['commodity'] = transport_flows['technology'].apply(
            lambda x: x.replace('_transport', '').replace('transport_', '')
        )

        # Rename columns
        transport_flows = transport_flows.rename(columns={'value_scenario_': 'flow'})

        # Create production cost mapping
        prod_costs = prod_results[['location', 'commodity', 'time_operation', 'opex_per_unit']].copy()
        prod_costs = prod_costs.rename(columns={'location': 'exporter'})

        # Merge transport flows with production costs
        import_flows = transport_flows.merge(
            prod_costs,
            on=['exporter', 'commodity', 'time_operation'],
            how='left'
        )

        # Summarize imports by destination country
        import_summary = import_flows.groupby(['importer', 'commodity', 'time_operation']).agg(
            total_imported_units=('flow', 'sum'),
            avg_prod_cost_per_unit=('opex_per_unit', 'mean')
        ).reset_index()

        # Calculate import value (without transport costs for now)
        import_summary['import_production_cost'] = import_summary['total_imported_units'] * import_summary[
            'avg_prod_cost_per_unit']

        print(
            f"Processed {len(import_flows)} trade flows for {import_summary['importer'].nunique()} importing countries")

        return import_flows, import_summary

    except Exception as e:
        print(f"ERROR processing trade data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()


def calculate_transport_costs_by_country(transport_file, opex_file):
    """
    Calculate transport costs for each country pair and commodity.

    Parameters:
    -----------
    transport_file : str
        Path to the transport flow file
    opex_file : str
        Path to the OPEX file with transport costs

    Returns:
    --------
    DataFrame
        Transport costs summarized by country pair and commodity
    """
    try:
        # Read files
        transport_df = pd.read_csv(transport_file)
        opex_df = pd.read_csv(opex_file)

        # Filter for transport technologies
        trans_mask = transport_df['technology'].str.contains('transport', case=False, na=False)
        transport_flows = transport_df[trans_mask].copy()

        if transport_flows.empty:
            print("WARNING: No transport flows found")
            return pd.DataFrame()

        # Split edge into exporter and importer
        if 'edge' not in transport_flows.columns or not transport_flows['edge'].str.contains('-').any():
            print("WARNING: Edge column not found or doesn't contain country pairs")
            return pd.DataFrame()

        transport_flows[['exporter', 'importer']] = transport_flows['edge'].str.split('-', expand=True)

        # Extract commodity from technology
        transport_flows['commodity'] = transport_flows['technology'].apply(
            lambda x: x.replace('_transport', '').replace('transport_', '')
        )

        # Rename flow column
        transport_flows = transport_flows.rename(columns={'value_scenario_': 'flow'})

        # Filter for transport technologies in OPEX
        opex_trans_mask = opex_df['technology'].str.contains('transport', case=False, na=False)
        opex_transport = opex_df[opex_trans_mask].copy()

        # In OPEX data, check if location contains country pairs
        if 'location' in opex_transport.columns and opex_transport['location'].str.contains('-').any():
            # Split location into exporter-importer
            opex_transport[['exporter', 'importer']] = opex_transport['location'].str.split('-', expand=True)

            # Extract commodity from technology
            opex_transport['commodity'] = opex_transport['technology'].apply(
                lambda x: x.replace('_transport', '').replace('transport_', '')
            )

            # Rename columns
            opex_transport = opex_transport.rename(columns={'value_scenario_': 'opex', 'year': 'time_operation'})

            # Merge transport flows with costs
            # This approach assumes the OPEX data has exporter-importer pairs in location
            transport_costs = transport_flows.merge(
                opex_transport[['exporter', 'importer', 'commodity', 'time_operation', 'opex']],
                on=['exporter', 'importer', 'commodity', 'time_operation'],
                how='left'
            )
        else:
            # If OPEX doesn't have country pairs, try to match on technology and time
            opex_transport = opex_transport.rename(columns={'value_scenario_': 'opex', 'year': 'time_operation'})

            # Create a mapping from each transport technology to its commodity
            opex_transport['commodity'] = opex_transport['technology'].apply(
                lambda x: x.replace('_transport', '').replace('transport_', '')
            )

            # Merge based on technology and time
            transport_costs = transport_flows.merge(
                opex_transport[['technology', 'time_operation', 'opex']],
                on=['technology', 'time_operation'],
                how='left'
            )

        # Fill NA opex values with 0
        transport_costs['opex'] = transport_costs['opex'].fillna(0)

        # Calculate transport cost per unit
        transport_costs['transport_cost_per_unit'] = np.where(
            transport_costs['flow'] == 0,
            0,
            transport_costs['opex'] / transport_costs['flow']
        )

        # Calculate total transport cost
        transport_costs['total_transport_cost'] = transport_costs['flow'] * transport_costs['transport_cost_per_unit']

        # Summarize by country pair and commodity
        transport_summary = transport_costs.groupby(['exporter', 'importer', 'commodity', 'time_operation']).agg(
            total_flow=('flow', 'sum'),
            total_transport_cost=('total_transport_cost', 'sum')
        ).reset_index()

        # Recalculate average cost per unit
        transport_summary['avg_transport_cost_per_unit'] = np.where(
            transport_summary['total_flow'] == 0,
            0,
            transport_summary['total_transport_cost'] / transport_summary['total_flow']
        )

        print(f"Calculated transport costs for {len(transport_summary)} country-commodity pairs")

        return transport_summary

    except Exception as e:
        print(f"ERROR calculating transport costs: {str(e)}")
        return pd.DataFrame()


def calculate_total_demand_costs(prod_flow_df, import_summary, transport_costs, prod_results):
    """
    Calculate the total cost of meeting final demand for each country and commodity,
    considering both domestic production and imports with transport costs.

    Parameters:
    -----------
    prod_flow_df : DataFrame
        Production flow data
    import_summary : DataFrame
        Summary of imports by destination country
    transport_costs : DataFrame
        Transport costs by country pair
    prod_results : DataFrame
        Production costs per unit for each country

    Returns:
    --------
    DataFrame
        Total cost of meeting final demand for each country and commodity
    """
    try:
        # Ensure commodity field is available
        if 'commodity' not in prod_results.columns:
            if 'carrier' in prod_results.columns:
                prod_results['commodity'] = prod_results['carrier']
            else:
                prod_results['commodity'] = prod_results['technology'].str.replace('_production', '').str.replace(
                    '_mining', '')

        # 1. Calculate domestic production
        domestic_prod = prod_flow_df.copy()
        domestic_prod['commodity'] = domestic_prod.get('carrier', domestic_prod['technology'].str.replace('_production',
                                                                                                          '').str.replace(
            '_mining', ''))

        # Group by country, commodity, and time
        domestic_summary = domestic_prod.groupby(['location', 'commodity', 'time_operation']).agg(
            domestic_production=('flow', 'sum')
        ).reset_index()

        # 2. Get domestic production costs
        domestic_costs = prod_results[['location', 'commodity', 'time_operation', 'opex_per_unit']].copy()

        # Merge production with costs
        domestic_summary = domestic_summary.merge(
            domestic_costs,
            on=['location', 'commodity', 'time_operation'],
            how='left'
        )

        # Calculate domestic production cost
        domestic_summary['total_domestic_cost'] = domestic_summary['domestic_production'] * domestic_summary[
            'opex_per_unit']

        # 3. Process imports and transport costs
        if not import_summary.empty and not transport_costs.empty:
            # Rename columns for consistency
            import_renamed = import_summary.rename(columns={'importer': 'location'})
            transport_renamed = transport_costs.rename(columns={'importer': 'location'})

            # Add transport costs to imports
            # First, create a way to merge the data
            import_with_transport = import_renamed.merge(
                transport_renamed[
                    ['exporter', 'location', 'commodity', 'time_operation', 'avg_transport_cost_per_unit']],
                on=['location', 'commodity', 'time_operation'],
                how='left'
            )

            # Fill NA transport costs with 0
            import_with_transport['avg_transport_cost_per_unit'] = import_with_transport[
                'avg_transport_cost_per_unit'].fillna(0)

            # Calculate total cost per unit (production + transport)
            import_with_transport['total_cost_per_unit'] = import_with_transport['avg_prod_cost_per_unit'] + \
                                                           import_with_transport['avg_transport_cost_per_unit']

            # Calculate total import cost
            import_with_transport['total_import_cost'] = import_with_transport['total_imported_units'] * \
                                                         import_with_transport['total_cost_per_unit']

            # Aggregate by destination country
            import_costs = import_with_transport.groupby(['location', 'commodity', 'time_operation']).agg(
                total_imported_units=('total_imported_units', 'sum'),
                total_import_cost=('total_import_cost', 'sum')
            ).reset_index()

            # Calculate average import cost per unit
            import_costs['avg_import_cost_per_unit'] = np.where(
                import_costs['total_imported_units'] == 0,
                0,
                import_costs['total_import_cost'] / import_costs['total_imported_units']
            )
        else:
            # Create empty DataFrame if no import data
            import_costs = pd.DataFrame(columns=[
                'location', 'commodity', 'time_operation', 'total_imported_units', 'total_import_cost',
                'avg_import_cost_per_unit'
            ])

        # 4. Combine domestic and import data
        # Use outer join to ensure we include all countries and commodities
        total_demand = domestic_summary.merge(
            import_costs[['location', 'commodity', 'time_operation', 'total_imported_units', 'total_import_cost',
                          'avg_import_cost_per_unit']],
            on=['location', 'commodity', 'time_operation'],
            how='outer'
        )

        # Fill NaN values with 0
        fill_cols = ['domestic_production', 'total_domestic_cost', 'total_imported_units', 'total_import_cost']
        for col in fill_cols:
            if col in total_demand.columns:
                total_demand[col] = total_demand[col].fillna(0)

        # Calculate total units (domestic + imported)
        total_demand['total_units'] = total_demand['domestic_production'] + total_demand['total_imported_units']

        # Calculate total cost (domestic + imported)
        total_demand['total_cost'] = total_demand['total_domestic_cost'] + total_demand['total_import_cost']

        # Calculate weighted average cost per unit
        total_demand['weighted_avg_cost_per_unit'] = np.where(
            total_demand['total_units'] == 0,
            0,
            total_demand['total_cost'] / total_demand['total_units']
        )

        # Calculate domestic and import proportions
        total_demand['domestic_proportion'] = np.where(
            total_demand['total_units'] == 0,
            0,
            total_demand['domestic_production'] / total_demand['total_units']
        )

        total_demand['import_proportion'] = np.where(
            total_demand['total_units'] == 0,
            0,
            total_demand['total_imported_units'] / total_demand['total_units']
        )

        print(f"Calculated total demand costs for {len(total_demand)} country-commodity combinations")

        return total_demand

    except Exception as e:
        print(f"ERROR calculating total demand costs: {str(e)}")
    return pd.DataFrame()


def create_final_demand_cost_format(total_demand_df, scenarios=None):
    """
    Create final format for total demand costs, with columns for different scenarios

    Parameters:
    -----------
    total_demand_df : DataFrame
        Total demand costs dataframe
    scenarios : list, optional
        List of scenario names (S1, S2, etc.)

    Returns:
    --------
    DataFrame
        Formatted total demand costs
    """
    try:
        # If we have scenario information, use it
        if scenarios:
            # Assuming total_demand_df has a 'scenario' column
            if 'scenario' not in total_demand_df.columns:
                print("WARNING: No scenario column found, but scenarios were provided.")
                # Create a pivot table with countries and commodities as index
                pivot_table = total_demand_df.pivot_table(
                    index=['location', 'commodity'],
                    columns='time_operation',
                    values='weighted_avg_cost_per_unit'
                )
            else:
                # Create a pivot table with countries, scenarios, and commodities as index
                pivot_table = total_demand_df.pivot_table(
                    index=['location', 'scenario', 'commodity'],
                    columns='time_operation',
                    values='weighted_avg_cost_per_unit'
                )
        else:
            # Create a pivot table with countries and commodities as index
            pivot_table = total_demand_df.pivot_table(
                index=['location', 'commodity'],
                columns='time_operation',
                values='weighted_avg_cost_per_unit'
            )

        # Rename columns to timestep format
        pivot_table.columns = [f'timestep_{t}' for t in pivot_table.columns]

        # Add average column, ignoring NaN values
        pivot_table['average'] = pivot_table.mean(axis=1)

        # Reset index for better readability
        pivot_table = pivot_table.reset_index()

        # If we have scenario information, create a more readable index
        if scenarios and 'scenario' in pivot_table.columns:
            pivot_table['country_scenario'] = pivot_table['location'] + ' ' + pivot_table['scenario']
            pivot_table = pivot_table.set_index(['country_scenario', 'commodity'])
        else:
            pivot_table = pivot_table.set_index(['location', 'commodity'])

        return pivot_table

    except Exception as e:
        print(f"ERROR creating final demand cost format: {str(e)}")
        return pd.DataFrame()


def calculate_levelized_cost(total_demand_df, output_dir):
    """
    Calculate the levelized cost of heat pump production by applying conversion
    factors to the weighted average cost per unit of each material.

    This function converts all costs to €/kW of heat pump capacity.
    For each node and timestep, it sums the contributions of all commodities.

    Parameters:
    -----------
    total_demand_df : DataFrame
        Total demand costs with weighted_avg_cost_per_unit column
    output_dir : Path
        Directory to save output files

    Returns:
    --------
    DataFrame
        Levelized costs by country, commodity, and timestep in €/kW
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path

    print("\nCalculating levelized cost of heat pump production...")

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

    # Create a copy of the input dataframe to avoid modifying it
    df = total_demand_df.copy()

    # Create a new dataframe for individual commodity calculations
    commodity_costs = []

    # Process each row in the dataframe to calculate individual commodity costs
    for _, row in df.iterrows():
        location = row['location']
        time_operation = row['time_operation']
        commodity = row['commodity']
        weighted_cost = row['weighted_avg_cost_per_unit']

        # Skip rows with missing or zero cost
        if pd.isna(weighted_cost) or weighted_cost == 0:
            continue

        # Check if commodity matches any conversion factor key
        if commodity not in conversion_factors:
            print(f"Warning: No conversion factor for {commodity}, skipping")
            continue

        # Get the appropriate conversion factor
        conversion_factor = conversion_factors[commodity]

        # Calculate levelized cost based on the unit type
        if commodity in ['HP', 'Compressor', 'HEX']:
            # These are in GW, convert to kW
            # 1 GW = 1,000,000 kW and 1000 euro in kiloeuro
            levelized_cost = weighted_cost / 1_000 * conversion_factor
        else:
            # These are in kilotons, convert to kg, then to kW
            # 1 kiloton = 1,000,000 kg
            # Then multiply by kg/kW conversion factor
            levelized_cost = weighted_cost / 1_000 * conversion_factor

        # Store the individual commodity calculation
        commodity_costs.append({
            'location': location,
            'time_operation': time_operation,
            'commodity': commodity,
            'weighted_cost_per_unit': weighted_cost,
            'conversion_factor': conversion_factor,
            'levelized_cost_per_kw': levelized_cost
        })

    # Create dataframe from individual commodity calculations
    levelized_df = pd.DataFrame(commodity_costs)

    # Calculate total levelized cost by location and timestep (sum all commodities)
    total_by_location_timestep = levelized_df.groupby(['location', 'time_operation']).agg(
        total_levelized_cost=('levelized_cost_per_kw', 'sum')
    ).reset_index()

    # Create pivot table with locations as rows, timesteps as columns
    pivot_table = total_by_location_timestep.pivot(
        index='location',
        columns='time_operation',
        values='total_levelized_cost'
    )

    # Rename columns to timestep format
    pivot_table.columns = [f'timestep_{t}' for t in pivot_table.columns]

    # Add average column
    pivot_table['average'] = pivot_table.mean(axis=1)

    # Reset index for better readability
    pivot_table = pivot_table.reset_index()

    # Save the detailed calculations for each commodity
    levelized_df.to_csv(output_dir / 'levelized_cost_details.csv', index=False)
    print(f"Detailed levelized costs saved to: {output_dir / 'levelized_cost_details.csv'}")

    # Save the summary pivot table with total cost per location per timestep
    pivot_table.to_csv(output_dir / 'levelized_cost_summary.csv', index=False)
    print(f"Summary levelized costs saved to: {output_dir / 'levelized_cost_summary.csv'}")

    # Create material breakdown
    material_breakdown = levelized_df.groupby(['location', 'commodity']).agg(
        avg_levelized_cost=('levelized_cost_per_kw', 'mean')
    ).reset_index()

    # Calculate total cost per location for percentage
    location_totals = material_breakdown.groupby('location')['avg_levelized_cost'].sum().reset_index()
    location_totals.rename(columns={'avg_levelized_cost': 'total_cost'}, inplace=True)

    # Merge to calculate percentages
    material_breakdown = material_breakdown.merge(location_totals, on='location')
    material_breakdown['percentage'] = material_breakdown['avg_levelized_cost'] / material_breakdown['total_cost'] * 100

    # Save material breakdown
    material_breakdown.to_csv(output_dir / 'material_cost_breakdown.csv', index=False)
    print(f"Material breakdown saved to: {output_dir / 'material_cost_breakdown.csv'}")

    print("Levelized cost calculation complete.")

    # Return both the detailed calculations and the location/timestep summary
    return levelized_df, pivot_table

def main():
    """Main function to execute the analysis"""
    # File paths
    flow_file = './HP_Model/parameter_results/flow_conversion_output/flow_conversion_output_scenarios.csv'
    opex_file = './HP_Model/parameter_results/cost_opex_yearly/cost_opex_yearly_scenarios.csv'
    capex_file = './HP_Model/parameter_results/cost_capex/cost_capex_scenarios.csv'
    transport_file = "./HP_Model/parameter_results/flow_transport/flow_transport_scenarios.csv"

    print("Starting transport cost analysis...")

    # Load and process data
    prod_flow_df, transport_flow_df, opex_df, capex_df = load_and_prepare_data(flow_file, opex_file, capex_file)

    # Calculate production cost per unit
    print("\nCalculating production costs...")
    prod_results = calculate_production_cost_per_unit(prod_flow_df, opex_df, capex_df)

    # Calculate transport costs
    print("\nCalculating transport costs...")
    transport_detail, transport_summary = calculate_transport_costs(transport_flow_df, opex_df, capex_df)

    # Calculate ratio of transport to production costs
    print("\nCalculating transport to production cost ratios...")
    cost_ratio_df = calculate_cost_ratio(prod_results, transport_summary)

    # Process trade data
    print("\nProcessing trade data...")
    import_flows, import_summary = process_trade_data(transport_file, prod_results)

    # Calculate transport costs by country
    print("\nCalculating transport costs by country pair...")
    transport_costs_by_country = calculate_transport_costs_by_country(transport_file, opex_file)

    # Calculate total demand costs
    print("\nCalculating total demand costs...")
    total_demand_costs = calculate_total_demand_costs(prod_flow_df, import_summary, transport_costs_by_country,
                                                      prod_results)

    # Format the total demand costs
    print("\nFormatting total demand costs...")
    # Check if we have scenario information in the data
    scenarios = None
    if 'scenario' in prod_flow_df.columns:
        scenarios = prod_flow_df['scenario'].unique().tolist()

    final_demand_costs = create_final_demand_cost_format(total_demand_costs, scenarios)

    # Create final formats for original outputs
    print("\nFormatting results...")
    final_prod_results = create_final_format(prod_results)
    final_transport_summary = create_transport_summary_format(transport_summary)
    final_ratio_summary, final_opex_ratio_summary = create_ratio_summary_format(cost_ratio_df)

    # Save results
    output_dir = Path(
        "./HP_Model/production cost")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate levelized costs for heat pump production
    print("\nCalculating levelized costs for heat pump production...")
    levelized_details, levelized_summary = calculate_levelized_cost(total_demand_costs,output_dir)

    # Save to CSV - original outputs
    final_prod_results.to_csv(output_dir / 'conversion_costs_per_unit.csv')

    if not final_transport_summary.empty:
        final_transport_summary.to_csv(output_dir / 'transport_costs_summary.csv')
        print(f"Transport costs saved to: {output_dir / 'transport_costs_summary.csv'}")
    else:
        print("WARNING: No transport cost data to save.")

    if not final_ratio_summary.empty:
        final_ratio_summary.to_csv(output_dir / 'transport_to_production_ratio.csv')
        print(f"Transport to production ratios saved to: {output_dir / 'transport_to_production_ratio.csv'}")

        final_opex_ratio_summary.to_csv(output_dir / 'transport_opex_to_production_opex_ratio.csv')
        print(
            f"Transport OPEX to production OPEX ratios saved to: {output_dir / 'transport_opex_to_production_opex_ratio.csv'}")
    else:
        print("WARNING: No ratio data to save.")

    # Save new outputs
    if not import_flows.empty:
        import_flows.to_csv(output_dir / 'import_flows.csv')
        print(f"Import flows saved to: {output_dir / 'import_flows.csv'}")

    if not import_summary.empty:
        import_summary.to_csv(output_dir / 'import_summary.csv')
        print(f"Import summary saved to: {output_dir / 'import_summary.csv'}")

    if not transport_costs_by_country.empty:
        transport_costs_by_country.to_csv(output_dir / 'transport_costs_by_country.csv')
        print(f"Transport costs by country saved to: {output_dir / 'transport_costs_by_country.csv'}")

    if not total_demand_costs.empty:
        total_demand_costs.to_csv(output_dir / 'total_demand_costs.csv')
        print(f"Total demand costs saved to: {output_dir / 'total_demand_costs.csv'}")

    if not final_demand_costs.empty:
        final_demand_costs.to_csv(output_dir / 'final_demand_costs.csv')
        print(f"Final demand costs saved to: {output_dir / 'final_demand_costs.csv'}")

    print(f"\nAnalysis complete.")

    # Display first few rows of results
    print("\nFirst few rows of production costs:")
    print(final_prod_results.head())

    print("\nFirst few rows of total demand costs:")
    print(total_demand_costs.head())

    print("\nFirst few rows of final demand costs format:")
    print(final_demand_costs.head())


if __name__ == "__main__":
    main()