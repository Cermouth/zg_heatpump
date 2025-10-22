import pandas as pd
from pathlib import Path

# ETH Color Scheme - matching your regions
REGION_COLORS = {
    'CHN': '#8C0A59',  # ETH Purple 120%
    'JPN': '#B73B92',  # ETH Purple 80%
    'KOR': '#DC9EC9',  # ETH Purple 40%
    'ROA': '#EFD0E3',  # ETH Purple 20%
    'EUR': '#215CAF',  # ETH Blue 100%
    'DEU': '#08407E',  # ETH Blue 120%
    'ITA': '#7A9DCF',  # ETH Blue 60%
    'AUT': '#4D7DBF',  # ETH Blue 80%
    'CZE': '#D3DEEF',  # ETH Blue 20%
    'ROE': '#A6BEDF',  # ETH Blue 40%
    'USA': '#007894',  # ETH Petrol 100%
    'BRA': '#D48681',  # ETH Red 60%
    'AUS': '#8E6713',  # ETH Bronze 100%
    'ROW': '#6F6F6F',  # ETH Grey 100%
}


def load_and_process_data(production_file, transport_file, target_year=2035):
    """Load production and transport data for the target year."""
    prod_df = pd.read_csv(production_file)
    trans_df = pd.read_csv(transport_file)

    year_offset = target_year - 2022
    prod_df = prod_df[prod_df['time_operation'] == year_offset].copy()
    trans_df = trans_df[trans_df['time_operation'] == year_offset].copy()

    return prod_df, trans_df


def prepare_r_format_data(prod_df, trans_df, technology, scenario_col, scenario_name):
    """
    Prepare flow data in R format with From, To, Value columns.
    Also includes domestic use flows.

    Returns:
    --------
    DataFrame with columns: From, To, scenario_name, Product, Value
    """
    # Get transport technology name
    tech_mapping = {
        'HP_assembly': 'HP_transport',
        'HEX_manufacturing': 'HEX_transport',
        'Compressor_manufacturing': 'Compressor_transport'
    }

    if technology not in tech_mapping:
        raise ValueError(f"Unknown technology: {technology}")

    transport_tech = tech_mapping[technology]

    # Filter data
    prod_data = prod_df[prod_df['technology'] == technology].copy()
    trans_data = trans_df[trans_df['technology'] == transport_tech].copy()

    # Get production by country
    production_by_country = {}
    for _, row in prod_data.iterrows():
        if scenario_col in row and pd.notna(row[scenario_col]):
            production = float(row[scenario_col])
            if production > 0:
                location = row['node']
                production_by_country[location] = production

    # Get exports by country
    exports_by_country = {}
    for _, row in trans_data.iterrows():
        if scenario_col in row and pd.notna(row[scenario_col]) and 'edge' in row:
            flow_value = float(row[scenario_col])
            if flow_value > 0:
                try:
                    exporter, importer = row['edge'].split('-')
                    exports_by_country[exporter] = exports_by_country.get(exporter, 0) + flow_value
                except (ValueError, AttributeError):
                    continue

    # Build flow dataframe including trade and domestic use
    flows = []

    # Add trade flows
    for _, row in trans_data.iterrows():
        if scenario_col in row and pd.notna(row[scenario_col]) and 'edge' in row:
            flow_value = float(row[scenario_col])
            if flow_value > 0.01:  # Filter very small flows
                try:
                    exporter, importer = row['edge'].split('-')
                    flows.append({
                        'From': exporter,
                        'To': importer,
                        'scenario_name': scenario_name,
                        'Product': technology.replace('_', ' ').title(),
                        'Value': flow_value
                    })
                except (ValueError, AttributeError):
                    continue

    # Add domestic use flows (From = To = same country)
    for country, production in production_by_country.items():
        exports = exports_by_country.get(country, 0)
        domestic_use = production - exports
        if domestic_use > 0.01:  # Filter very small flows
            flows.append({
                'From': country,
                'To': country,
                'scenario_name': scenario_name,
                'Product': technology.replace('_', ' ').title(),
                'Value': domestic_use
            })

    if not flows:
        return None

    flow_df = pd.DataFrame(flows)

    # Reorder columns: From, To, scenario_name, Product, Value
    flow_df = flow_df[['From', 'To', 'scenario_name', 'Product', 'Value']]

    return flow_df


def prepare_combined_r_format_data(prod_df, trans_df, technologies, material_name, scenario_col, scenario_name):
    """
    Prepare flow data in R format for combined technologies (e.g., Copper recycling + refinement).
    Aggregates production from multiple technologies.

    Returns:
    --------
    DataFrame with columns: From, To, scenario_name, Product, Value
    """
    # Get transport technology name
    if technologies[0] in ['Copper_recycling', 'Copper_refinement']:
        transport_tech = 'Copper_transport'
    elif technologies[0] in ['Nickel_recycling', 'Nickel_refinement']:
        transport_tech = 'Nickel_transport'
    else:
        raise ValueError(f"Unknown technology group: {technologies}")

    # Filter production data for all technologies and aggregate
    prod_data = prod_df[prod_df['technology'].isin(technologies)].copy()
    trans_data = trans_df[trans_df['technology'] == transport_tech].copy()

    # Get production by country (aggregated across all technologies)
    production_by_country = {}
    for _, row in prod_data.iterrows():
        if scenario_col in row and pd.notna(row[scenario_col]):
            production = float(row[scenario_col])
            if production > 0:
                location = row['node']
                production_by_country[location] = production_by_country.get(location, 0) + production

    # Get exports by country
    exports_by_country = {}
    for _, row in trans_data.iterrows():
        if scenario_col in row and pd.notna(row[scenario_col]) and 'edge' in row:
            flow_value = float(row[scenario_col])
            if flow_value > 0:
                try:
                    exporter, importer = row['edge'].split('-')
                    exports_by_country[exporter] = exports_by_country.get(exporter, 0) + flow_value
                except (ValueError, AttributeError):
                    continue

    # Build flow dataframe including trade and domestic use
    flows = []

    # Add trade flows
    for _, row in trans_data.iterrows():
        if scenario_col in row and pd.notna(row[scenario_col]) and 'edge' in row:
            flow_value = float(row[scenario_col])
            if flow_value > 0.01:  # Filter very small flows
                try:
                    exporter, importer = row['edge'].split('-')
                    flows.append({
                        'From': exporter,
                        'To': importer,
                        'scenario_name': scenario_name,
                        'Product': material_name,
                        'Value': flow_value
                    })
                except (ValueError, AttributeError):
                    continue

    # Add domestic use flows (From = To = same country)
    for country, production in production_by_country.items():
        exports = exports_by_country.get(country, 0)
        domestic_use = production - exports
        if domestic_use > 0.01:  # Filter very small flows
            flows.append({
                'From': country,
                'To': country,
                'scenario_name': scenario_name,
                'Product': material_name,
                'Value': domestic_use
            })

    if not flows:
        return None

    flow_df = pd.DataFrame(flows)

    # Reorder columns: From, To, scenario_name, Product, Value
    flow_df = flow_df[['From', 'To', 'scenario_name', 'Product', 'Value']]

    return flow_df


def create_combined_csv_for_r(prod_df, trans_df, technologies, scenarios, output_dir):
    """
    Create a single CSV with all scenarios for easy loading in R.
    """
    all_data = []

    for technology in technologies:
        for scenario_col, scenario_name in scenarios.items():
            flow_df = prepare_r_format_data(prod_df, trans_df, technology, scenario_col, scenario_name)

            if flow_df is not None and not flow_df.empty:
                all_data.append(flow_df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        output_file = output_dir / "all_trade_flows_2035.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"\nCombined data saved to: {output_file}")

        # Print summary statistics
        print("\nSummary statistics:")
        trade_flows = combined_df[combined_df['From'] != combined_df['To']]
        domestic_flows = combined_df[combined_df['From'] == combined_df['To']]
        print(f"  Total flows: {len(combined_df)}")
        print(f"  Trade flows: {len(trade_flows)}")
        print(f"  Domestic flows: {len(domestic_flows)}")
        print(f"  Total value: {combined_df['Value'].sum():.2f} GW")

        return combined_df

    return None


def create_combined_csv_for_materials(prod_df, trans_df, material_technologies, scenarios, output_dir):
    """
    Create a single CSV with all material flows (combined recycling + refinement).
    """
    all_data = []

    for material_name, tech_list in material_technologies.items():
        for scenario_col, scenario_name in scenarios.items():
            flow_df = prepare_combined_r_format_data(prod_df, trans_df, tech_list, material_name, scenario_col, scenario_name)

            if flow_df is not None and not flow_df.empty:
                all_data.append(flow_df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        output_file = output_dir / "material_trade_flows_2035.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"\nMaterial flows saved to: {output_file}")

        # Print summary statistics
        print("\nMaterial flow statistics:")
        trade_flows = combined_df[combined_df['From'] != combined_df['To']]
        domestic_flows = combined_df[combined_df['From'] == combined_df['To']]
        print(f"  Total flows: {len(combined_df)}")
        print(f"  Trade flows: {len(trade_flows)}")
        print(f"  Domestic flows: {len(domestic_flows)}")
        print(f"  Total value: {combined_df['Value'].sum():.2f} GW")

        return combined_df

    return None


def main():
    """Main execution function"""

    # Define file paths
    production_file = './parameter_results/flow_conversion_output/flow_conversion_output_scenarios.csv'
    transport_file = './parameter_results/flow_transport/flow_transport_scenarios.csv'

    # Output directory
    output_dir = Path('./r_export')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data for 2035
    print("Loading data for 2035...")
    prod_df, trans_df = load_and_process_data(production_file, transport_file, target_year=2035)

    # Define scenarios
    scenarios = {
        'value_scenario_': 'Base',
        'value_scenario_DemandMet': 'DemandMet',
        'value_scenario_SelfSuff40': 'SelfSuff40',
        'value_scenario_Tariffs': 'Tariffs'
    }

    # Define technologies for heat pump components
    hp_technologies = ['HP_assembly', 'HEX_manufacturing', 'Compressor_manufacturing']

    # Define combined material technologies
    material_technologies = {
        'Copper': ['Copper_recycling', 'Copper_refinement'],
        'Nickel': ['Nickel_recycling', 'Nickel_refinement']
    }
    scenariosm = {
        'value_scenario_': 'Base',
        'value_scenario_DemandMet': 'DemandMet',
        'value_scenario_SelfSuff40': 'SelfSuff40',
        'value_scenario_Recycling': 'Recycling'
    }
    # Create combined CSV for heat pump components
    print("\n" + "=" * 60)
    print("Exporting heat pump component flows...")
    print("=" * 60)
    hp_df = create_combined_csv_for_r(prod_df, trans_df, hp_technologies, scenarios, output_dir)

    # Create combined CSV for materials
    print("\n" + "=" * 60)
    print("Exporting material flows (Copper & Nickel)...")
    print("=" * 60)
    material_df = create_combined_csv_for_materials(prod_df, trans_df, material_technologies, scenariosm, output_dir)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"\nFiles exported to: {output_dir}")

    # Show sample data
    if hp_df is not None and not hp_df.empty:
        print("\nSample of HP component data:")
        print(hp_df.head(10).to_string(index=False))

    if material_df is not None and not material_df.empty:
        print("\nSample of material data:")
        print(material_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
