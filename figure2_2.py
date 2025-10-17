import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

# ETH Color Scheme
ETH_COLORS = {
    'ETH Blue': {'100%': '#215CAF', '60%': '#7A9DCF', '40%': '#A6BEDF', '20%': '#D3DEEF'},
    'ETH Petrol': {'100%': '#007894', '60%': '#66AFC0', '40%': '#99CAD5', '20%': '#CCE4EA'},
    'ETH Green': {'100%': '#627313', '60%': '#A1AB71', '40%': '#C0C7A1', '20%': '#E0E3D0'},
    'ETH Bronze': {'100%': '#8E6713', '60%': '#BBA471', '40%': '#D2C2A1', '20%': '#E8E1D0'},
    'ETH Red': {'100%': '#B7352D', '60%': '#D48681', '40%': '#E2AEAB', '20%': '#F1D7D5'},
    'ETH Purple': {'100%': '#A7117A', '60%': '#CA6CAE', '40%': '#DC9EC9', '20%': '#EFD0E3'},
    'ETH Grey': {'100%': '#6F6F6F', '60%': '#A9A9A9', '40%': '#C5C5C5', '20%': '#E2E2E2'}
}

# Region to color mapping
REGION_COLORS = {
    'CHN': 'ETH Red',  # China - Red
    'AUS': 'ETH Bronze',  # Australia - Bronze
    'ROW': 'ETH Purple',  # Rest of World - Purple
    'JPN': 'ETH Bronze',  # Japan - Bronze
    'ROA': 'ETH Green',  # Rest of Asia - Green
    'EUR': 'ETH Blue',  # Europe - Blue
    'USA': 'ETH Petrol',  # USA - Petrol
    'BRA': 'ETH Green',  # Brazil - Green
    'KOR': 'ETH Petrol',  # Korea - Petrol
    'DEU': 'ETH Blue',  # Germany - Blue
    'ITA': 'ETH Blue',  # Italy - Blue
    'AUT': 'ETH Blue',  # Austria - Blue
    'CZE': 'ETH Blue',  # Czechia - Blue
    'ROE': 'ETH Blue',  # Rest of Europe - Blue
}


def get_node_color(node_name, intensity='100%'):
    """Get color for a node based on its region."""
    # Extract country code from node name
    country = node_name.split('_')[0] if '_' in node_name else node_name

    region_color = REGION_COLORS.get(country, 'ETH Grey')
    return ETH_COLORS[region_color][intensity]


def get_link_color_with_alpha(node_name, alpha=0.3):
    """Get color for a link with transparency."""
    base_color = get_node_color(node_name, '60%')
    # Convert hex to rgba
    hex_color = base_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'


def load_and_process_data(production_file, transport_file, target_year=2035):
    """
    Load production and transport data for the target year.

    Parameters:
    -----------
    production_file : str
        Path to flow_conversion_output CSV file
    transport_file : str
        Path to flow_transport CSV file
    target_year : int
        Target year for analysis (default: 2035)

    Returns:
    --------
    tuple : (production_df, transport_df)
    """
    # Load data
    prod_df = pd.read_csv(production_file)
    trans_df = pd.read_csv(transport_file)

    # Filter for target year (assuming time_operation is years after 2022)
    year_offset = target_year - 2022
    prod_df = prod_df[prod_df['time_operation'] == year_offset].copy()
    trans_df = trans_df[trans_df['time_operation'] == year_offset].copy()

    return prod_df, trans_df


def prepare_flows_for_sankey(prod_df, trans_df, technology, scenario_col):
    """
    Prepare flow data for Sankey diagram combining production and transport.
    Creates separate left (source) and right (destination) nodes.

    Parameters:
    -----------
    prod_df : DataFrame
        Production flow data
    trans_df : DataFrame
        Transport flow data
    technology : str
        Technology name (e.g., 'HP_assembly', 'HEX_manufacturing', 'Compressor_manufacturing')
    scenario_col : str
        Scenario column name (e.g., 'value_scenario_', 'value_scenario_DemandMet')

    Returns:
    --------
    dict : Dictionary with 'sources', 'targets', 'values', 'labels', 'x_positions', 'y_positions'
    """
    # Get transport technology name
    if technology == 'HP_assembly':
        transport_tech = 'HP_transport'
    elif technology == 'HEX_manufacturing':
        transport_tech = 'HEX_transport'
    elif technology == 'Compressor_manufacturing':
        transport_tech = 'Compressor_transport'
    else:
        raise ValueError(f"Unknown technology: {technology}")

    # Filter production data
    prod_data = prod_df[prod_df['technology'] == technology].copy()

    # Filter transport data
    trans_data = trans_df[trans_df['technology'] == transport_tech].copy()

    # Calculate exports and imports by country
    exports_by_country = {}
    imports_by_country = {}

    for _, row in trans_data.iterrows():
        if scenario_col in row and pd.notna(row[scenario_col]) and 'edge' in row:
            flow_value = float(row[scenario_col])
            if flow_value > 0:
                try:
                    exporter, importer = row['edge'].split('-')
                    exports_by_country[exporter] = exports_by_country.get(exporter, 0) + flow_value
                    imports_by_country[importer] = imports_by_country.get(importer, 0) + flow_value
                except (ValueError, AttributeError):
                    continue

    # Get all countries that have production or consumption
    production_by_country = {}
    for _, row in prod_data.iterrows():
        if scenario_col in row and pd.notna(row[scenario_col]):
            production = float(row[scenario_col])
            if production > 0:
                location = row['node']
                production_by_country[location] = production

    # Calculate domestic use for each country
    domestic_by_country = {}
    for country, production in production_by_country.items():
        exports = exports_by_country.get(country, 0)
        domestic_use = production - exports
        if domestic_use > 0:
            domestic_by_country[country] = domestic_use

    # Get all unique countries (left and right)
    all_countries = set()
    all_countries.update(production_by_country.keys())
    all_countries.update(imports_by_country.keys())
    # Group countries by region, then sort within groups
    region_groups = {
        'Asia': ['CHN', 'JPN', 'KOR', 'ROA'],
        'Europe': ['EUR', 'DEU', 'ITA', 'AUT', 'CZE', 'ROE'],
        'Americas': ['USA', 'BRA'],
        'Oceania': ['AUS'],
        'Other': ['ROW']
    }
    all_countries_ordered = []
    for region in ['Asia', 'Europe', 'Americas', 'Oceania', 'Other']:
        all_countries_ordered.extend([c for c in region_groups[region] if c in all_countries])
    all_countries = all_countries_ordered

    # Create left (source) and right (destination) nodes
    node_labels = []
    left_nodes = {}  # country -> node_index
    right_nodes = {}  # country -> node_index

    idx = 0
    for country in all_countries:
        # Left node (source/production)
        node_labels.append(country)
        left_nodes[country] = idx
        idx += 1

    for country in all_countries:
        # Right node (destination/consumption)
        node_labels.append(country)
        right_nodes[country] = idx
        idx += 1

    # Build flows
    sources = []
    targets = []
    values = []

    # Add domestic flows (left to right, same country)
    for country, domestic_use in domestic_by_country.items():
        sources.append(left_nodes[country])
        targets.append(right_nodes[country])
        values.append(domestic_use)

    # Add trade flows (left source to right destination)
    for _, row in trans_data.iterrows():
        if scenario_col in row and pd.notna(row[scenario_col]) and 'edge' in row:
            flow_value = float(row[scenario_col])
            if flow_value > 0:
                try:
                    exporter, importer = row['edge'].split('-')
                    sources.append(left_nodes[exporter])
                    targets.append(right_nodes[importer])
                    values.append(flow_value)
                except (ValueError, AttributeError, KeyError):
                    continue

    # Set x positions (left=0.05, right=0.95 for more spacing)
    n_countries = len(all_countries)
    x_positions = [0.05] * n_countries + [0.95] * n_countries
    y_positions = [i / max(n_countries - 1, 1) for i in range(n_countries)] * 2

    return {
        'sources': sources,
        'targets': targets,
        'values': values,
        'labels': node_labels,
        'x_positions': x_positions,
        'y_positions': y_positions,
        'n_countries': n_countries
    }


def create_sankey_figure(flow_data, title, unit='GW'):
    """
    Create a Plotly Sankey diagram with ETH color scheme.
    Nodes positioned on left and right sides.

    Parameters:
    -----------
    flow_data : dict
        Dictionary with flow data from prepare_flows_for_sankey
    title : str
        Title for the plot
    unit : str
        Unit for display (default: 'GW')

    Returns:
    --------
    plotly.graph_objects.Figure
    """
    if not flow_data['sources']:
        print(f"No flow data available for {title}")
        return None

    # Get country names (first half are left nodes, second half are right nodes)
    n_countries = flow_data['n_countries']

    # Assign colors to nodes based on region (both left and right use same color for same country)
    node_colors = []
    for i, label in enumerate(flow_data['labels']):
        node_colors.append(get_node_color(label, '100%'))

    # Assign colors to links based on source node
    link_colors = []
    for src_idx in flow_data['sources']:
        src_label = flow_data['labels'][src_idx]
        link_colors.append(get_link_color_with_alpha(src_label, alpha=0.4))

    # Calculate total flow
    total_flow = sum(flow_data['values'])

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=55,
            thickness=30,
            line=dict(color="white", width=2),
            label=flow_data['labels'],
            color=node_colors,
            x=flow_data['x_positions'],
            y=flow_data['y_positions'],
            customdata=[f"{label}" for label in flow_data['labels']],
            hovertemplate='%{customdata}<br>Total: %{value:.2f} ' + unit + '<extra></extra>'
        ),
        link=dict(
            source=flow_data['sources'],
            target=flow_data['targets'],
            value=flow_data['values'],
            color=link_colors,
            hovertemplate='%{source.label} → %{target.label}<br>Flow: %{value:.2f} ' + unit + '<extra></extra>'
        )
    )])

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Total Flow: {total_flow:.2f} {unit}</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=18, family='Arial', weight='bold', color='#333')
        ),
        font=dict(size=12, family='Arial'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=800,
        width=800,
        margin=dict(t=50, b=50, l=50, r=50)
    )

    return fig


def create_sankey_grid(prod_df, trans_df, technology, scenarios, output_dir):
    """
    Create individual Sankey diagrams for each scenario.

    Parameters:
    -----------
    prod_df : DataFrame
        Production flow data
    trans_df : DataFrame
        Transport flow data
    technology : str
        Technology name
    scenarios : dict
        Dictionary mapping scenario column names to display names
    output_dir : Path
        Output directory for saving plots
    """
    # Create individual plots for each scenario
    for scenario_col, scenario_name in scenarios.items():
        flow_data = prepare_flows_for_sankey(prod_df, trans_df, technology, scenario_col)

        if not flow_data['sources']:
            print(f"No data for {technology} - {scenario_name}")
            continue

        tech_display = technology.replace('_', ' ').title()
        title = f"{tech_display} - {scenario_name} (2035)"

        fig = create_sankey_figure(flow_data, title)

        if fig:
            # Save as PNG
            output_file = output_dir / f"sankey_{technology}_{scenario_name}_2035.png"
            fig.write_image(str(output_file), scale=2)
            print(f"Saved: {output_file}")

            # Save as HTML for interactivity
            output_html = output_dir / f"sankey_{technology}_{scenario_name}_2035.html"
            fig.write_html(str(output_html))
            print(f"Saved: {output_html}")


def create_combined_scenario_grid(prod_df, trans_df, technology, scenarios, output_dir):
    """
    Create a single figure with all scenarios in a grid layout.

    Parameters:
    -----------
    prod_df : DataFrame
        Production flow data
    trans_df : DataFrame
        Transport flow data
    technology : str
        Technology name
    scenarios : dict
        Dictionary mapping scenario column names to display names
    output_dir : Path
        Output directory for saving plots
    """
    n_scenarios = len(scenarios)
    n_cols = 2
    n_rows = (n_scenarios + n_cols - 1) // n_cols

    # Create figure
    fig = go.Figure()

    annotations = []

    for i, (scenario_col, scenario_name) in enumerate(scenarios.items()):
        flow_data = prepare_flows_for_sankey(prod_df, trans_df, technology, scenario_col)

        if not flow_data['sources']:
            continue

        # Calculate position in grid with more spacing
        row = i // n_cols
        col = i % n_cols

        # Increase spacing between subplots
        x_spacing = 0.1
        y_spacing = 0.15
        subplot_width = (1.0 - (n_cols + 1) * x_spacing)  / n_cols
        subplot_height = (1.0 - (n_rows + 1) * y_spacing) / n_rows

        # Shift left by reducing x values, shift up by increasing y values
        x_shift = -0.1  # Negative moves left
        y_shift = 0.1  # Positive moves up

        x_domain = [
            x_spacing + col * (subplot_width + x_spacing) + x_shift,
            x_spacing + col * (subplot_width + x_spacing) + subplot_width + x_shift
        ]
        y_domain = [
            1.0 - y_spacing - (row + 1) * (subplot_height + y_spacing) + y_shift,
            1.0 - y_spacing - row * (subplot_height + y_spacing) - y_spacing + y_shift
        ]

        # Adjust x positions to be within domain
        x_positions_adjusted = []
        for x in flow_data['x_positions']:
            x_new = x_domain[0] + x * (x_domain[1] - x_domain[0])
            x_positions_adjusted.append(x_new)

        # Adjust y positions to be within domain
        y_positions_adjusted = []
        for y in flow_data['y_positions']:
            y_new = y_domain[0] + y * (y_domain[1] - y_domain[0])
            y_positions_adjusted.append(y_new)

        # Assign colors to nodes
        node_colors = [get_node_color(label, '100%') for label in flow_data['labels']]

        # Assign colors to links
        link_colors = []
        for src_idx in flow_data['sources']:
            src_label = flow_data['labels'][src_idx]
            link_colors.append(get_link_color_with_alpha(src_label, alpha=0.4))

        # Add Sankey trace
        fig.add_trace(go.Sankey(
            domain=dict(x=x_domain, y=y_domain),
            arrangement='snap',
            node=dict(
                pad=10,
                thickness=35,
                line=dict(color="white", width=2),
                label=flow_data['labels'],
                color=node_colors,
                x=x_positions_adjusted,
                y=y_positions_adjusted
            ),
            link=dict(
                source=flow_data['sources'],
                target=flow_data['targets'],
                value=flow_data['values'],
                color=link_colors
            )
        ))

        # Add scenario title (centered above each subplot)
        total_flow = sum(flow_data['values'])
        annotations.append(dict(
            text=f"<b>{scenario_name}</b><br>Total: {total_flow:.1f} GW",
            x=(x_domain[0] + x_domain[1]) / 2,
            y=y_domain[1] + 0.05,  # Change from 0.03 to 0.05 (move title up more)
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16, family='Arial', color='#333'),
            align='center',
            xanchor='center'
        ))

    tech_display = technology.replace('_', ' ').title()

    fig.update_layout(
        title=dict(
            text=f"<b>{tech_display} - All Scenarios (2035)</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=20, family='Arial', weight='bold', color='#333')
        ),
        width=1000,
        height=1000 * n_rows,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        annotations=annotations,
        margin=dict(t=10, b=10, l=10, r=10)
    )

    # Save combined grid
    output_file = output_dir / f"sankey_{technology}_all_scenarios_2035.png"
    fig.write_image(str(output_file), scale=2)
    print(f"Saved combined grid: {output_file}")

    output_html = output_dir / f"sankey_{technology}_all_scenarios_2035.html"
    fig.write_html(str(output_html))
    print(f"Saved combined grid: {output_html}")
    print(f"Saved combined grid: {output_html}")


def main():
    """Main execution function"""

    # Define file paths
    production_file = './parameter_results/flow_conversion_output/flow_conversion_output_scenarios.csv'
    transport_file = './parameter_results/flow_transport/flow_transport_scenarios.csv'

    # Output directory
    output_dir = Path('./visualization')
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

    # Define technologies
    technologies = ['HP_assembly', 'HEX_manufacturing', 'Compressor_manufacturing']

    # Create Sankey plots for each technology
    for technology in technologies:
        print(f"\nCreating Sankey plots for {technology}...")

        # Create individual scenario plots
        create_sankey_grid(prod_df, trans_df, technology, scenarios, output_dir)

        # Create combined grid plot
        create_combined_scenario_grid(prod_df, trans_df, technology, scenarios, output_dir)

    print(f"\nAll Sankey plots saved to: {output_dir}")


if __name__ == "__main__":
    main()