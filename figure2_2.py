import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

# ETH Color Scheme
ETH_COLORS = {
    'ETH Blue': {'120%': '#08407E', '100%': '#215CAF', '80%': '#4D7DBF', '60%': '#7A9DCF', '40%': '#A6BEDF', '20%': '#D3DEEF', '10%': '#E9EFF7'},
    'ETH Petrol': {'120%': '#00596D', '100%': '#007894', '80%': '#3395AB', '60%': '#66AFC0', '40%': '#99CAD5', '20%': '#CCE4EA', '10%': '#E7F4F7'},
    'ETH Green': {'120%': '#365213', '100%': '#627313', '80%': '#818F42', '60%': '#A1AB71', '40%': '#C0C7A1', '20%': '#E0E3D0', '10%': '#EEF1E7'},
    'ETH Bronze': {'120%': '#704F12', '100%': '#8E6713', '80%': '#A58542', '60%': '#BBA471', '40%': '#D2C2A1', '20%': '#E8E1D0', '10%': '#F4F0E7'},
    'ETH Red': {'120%': '#96272D', '100%': '#B7352D', '80%': '#C55D57', '60%': '#D48681', '40%': '#E2AEAB', '20%': '#F1D7D5', '10%': '#F8EBEA'},
    'ETH Purple': {'120%': '#8C0A59', '100%': '#A7117A', '80%': '#B73B92', '60%': '#CA6CAE', '40%': '#DC9EC9', '20%': '#EFD0E3', '10%': '#F8E8F3'},
    'ETH Grey': {'120%': '#575757', '100%': '#6F6F6F', '80%': '#8C8C8C', '60%': '#A9A9A9', '40%': '#C5C5C5', '20%': '#E2E2E2', '10%': '#F1F1F1'}
}

# Region to color mapping
REGION_COLORS = {
    'CHN': ETH_COLORS['ETH Purple']['120%'],  # China - Red
    'AUS': ETH_COLORS['ETH Bronze']['100%'],  # Australia - Bronze
    'ROW': ETH_COLORS['ETH Grey']['100%'],  # Rest of World - Purple
    'JPN': ETH_COLORS['ETH Purple']['80%'],  # Japan - Bronze
    'ROA': ETH_COLORS['ETH Purple']['20%'],  # Rest of Asia - Green
    'EUR': ETH_COLORS['ETH Blue']['100%'],  # Europe - Blue
    'USA': ETH_COLORS['ETH Petrol']['100%'],  # USA - Petrol
    'BRA': ETH_COLORS['ETH Red']['60%'],  # Brazil - Green
    'KOR': ETH_COLORS['ETH Purple']['40%'],  # Korea - Petrol
    'DEU': ETH_COLORS['ETH Blue']['120%'],  # Germany - Blue
    'ITA': ETH_COLORS['ETH Blue']['60%'],  # Italy - Blue
    'AUT': ETH_COLORS['ETH Blue']['80%'],  # Austria - Blue
    'CZE': ETH_COLORS['ETH Blue']['20%'],  # Czechia - Blue
    'ROE': ETH_COLORS['ETH Blue']['40%'],  # Rest of Europe - Blue
}


def get_node_color(node_name, intensity='100%'):
    """Get color for a node based on its region."""
    # Extract country code from node name
    country = node_name.split('_')[0] if '_' in node_name else node_name

    region_color = REGION_COLORS.get(country)
    return region_color


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
                    if technology == 'Compressor_transport':
                        continue
                except (ValueError, AttributeError, KeyError):
                    continue

    # 优化节点位置计算 - 为底部留出更多空间
    n_countries = len(all_countries)
    if n_countries > 1:
        # 使用0.1到0.9的范围，为顶部和底部留出空间
        y_positions = [0.01 + (i * 0.9 / (n_countries - 1)) for i in range(n_countries)]
    else:
        y_positions = [0.5]  # 如果只有一个国家，放在中间

    # 左右节点使用相同的位置
    y_positions = y_positions * 2
    x_positions = [0.15] * n_countries + [0.85] * n_countries

    return {
        'sources': sources,
        'targets': targets,
        'values': values,
        'labels': node_labels,
        'x_positions': x_positions,
        'y_positions': y_positions,
        'n_countries': n_countries
    }


def prepare_combined_flows_for_sankey(prod_df, trans_df, technologies, scenario_col):
    """
    Prepare flow data for Sankey diagram combining multiple production technologies.
    This is useful for materials like Copper and Nickel that have both recycling and refinement.

    Parameters:
    -----------
    prod_df : DataFrame
        Production flow data
    trans_df : DataFrame
        Transport flow data
    technologies : list
        List of technology names to combine (e.g., ['Copper_recycling', 'Copper_refinement'])
    scenario_col : str
        Scenario column name (e.g., 'value_scenario_', 'value_scenario_DemandMet')

    Returns:
    --------
    dict : Dictionary with 'sources', 'targets', 'values', 'labels', 'x_positions', 'y_positions'
    """
    # Get transport technology name (should be same for all input technologies)
    if technologies[0] in ['Copper_recycling', 'Copper_refinement']:
        transport_tech = 'Copper_transport'
    elif technologies[0] in ['Nickel_recycling', 'Nickel_refinement']:
        transport_tech = 'Nickel_transport'
    else:
        raise ValueError(f"Unknown technology group: {technologies}")

    # Filter production data for all technologies
    prod_data = prod_df[prod_df['technology'].isin(technologies)].copy()

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
    # Aggregate production from all technologies
    production_by_country = {}
    for _, row in prod_data.iterrows():
        if scenario_col in row and pd.notna(row[scenario_col]):
            production = float(row[scenario_col])
            if production > 0:
                location = row['node']
                production_by_country[location] = production_by_country.get(location, 0) + production

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

    # Optimize node positions
    n_countries = len(all_countries)
    if n_countries > 1:
        y_positions = [0.01 + (i * 0.9 / (n_countries - 1)) for i in range(n_countries)]
    else:
        y_positions = [0.5]

    y_positions = y_positions * 2
    x_positions = [0.15] * n_countries + [0.85] * n_countries

    return {
        'sources': sources,
        'targets': targets,
        'values': values,
        'labels': node_labels,
        'x_positions': x_positions,
        'y_positions': y_positions,
        'n_countries': n_countries
    }


def calculate_optimal_layout(flow_data):
    """
    根据节点和流线数量计算最优的布局参数
    """
    n_nodes = len(flow_data['labels'])
    n_links = len(flow_data['sources'])

    # 基础高度
    base_height = 800
    # 每增加一个节点增加的高度
    extra_height_per_node = 40
    # 每增加一条流线增加的高度
    extra_height_per_link = 5

    optimal_height = base_height + (n_nodes * extra_height_per_node) + (n_links * extra_height_per_link)

    # 动态调整节点参数
    if n_nodes > 15:
        node_pad = 10
        node_thickness = 15
        font_size = 10
    elif n_nodes > 10:
        node_pad = 20
        node_thickness = 20
        font_size = 11
    else:
        node_pad = 30
        node_thickness = 25
        font_size = 12

    # 动态调整边距
    bottom_margin = max(100, 60 + (n_nodes * 3))

    return {
        'height': optimal_height,
        'node_pad': node_pad,
        'node_thickness': node_thickness,
        'font_size': font_size,
        'bottom_margin': bottom_margin,
        'width': max(900, 800 + (n_nodes * 10))
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

    # 计算最优布局参数
    layout_params = calculate_optimal_layout(flow_data)

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

    # Create Sankey diagram with optimized parameters
    fig = go.Figure(data=[go.Sankey(
        arrangement='freeform',  # 使用freeform获得更好的布局控制
        node=dict(
            pad=layout_params['node_pad'],  # 动态调整
            thickness=layout_params['node_thickness'],  # 动态调整
            line=dict(color="white", width=1.5),
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
            font=dict(size=16, family='Arial', weight='bold', color='#333')
        ),
        font=dict(size=layout_params['font_size'], family='Arial'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=layout_params['height'],  # 动态高度
        width=layout_params['width'],  # 动态宽度
        margin=dict(
            t=80,  # 顶部边距
            b=layout_params['bottom_margin'],  # 动态底部边距
            l=80,  # 左侧边距
            r=80  # 右侧边距
        )
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

def create_combined_scenario_grid(prod_df, trans_df, technology, scenarios, output_dir,
                                  n_cols=4, x_spacing=0.03, y_spacing=0.12):
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
    n_cols : int
        Number of columns in the grid (default: 4 for single row)
    x_spacing : float
        Horizontal spacing between subplots
    y_spacing : float
        Vertical spacing between subplots
    """
    n_scenarios = len(scenarios)
    n_rows = (n_scenarios + n_cols - 1) // n_cols

    # åŠ¨æ€è®¡ç®—æ•´ä½“é«˜åº¦ - ä¸ºåº•éƒ¨ç•™å‡ºæ›´å¤šç©ºé—´
    base_height_per_plot = 900
    extra_height_for_bottom = 300
    total_height = (base_height_per_plot * n_rows) + extra_height_for_bottom
    total_width = 2200  # å¢žåŠ å®½åº¦ä»¥å®¹çº³4ä¸ªå›¾è¡¨

    # Create figure
    fig = go.Figure()
    annotations = []

    for i, (scenario_col, scenario_name) in enumerate(scenarios.items()):
        flow_data = prepare_flows_for_sankey(prod_df, trans_df, technology, scenario_col)

        if not flow_data['sources']:
            continue

        # Calculate position in grid
        row = i // n_cols
        col = i % n_cols

        # Calculate subplot dimensions with more vertical space
        subplot_width = (1.0 - (n_cols + 1) * x_spacing) / n_cols
        subplot_height = (1.0 - (n_rows + 1) * y_spacing) / n_rows

        # Calculate domain positions - ä¸ºåº•éƒ¨ç•™å‡ºæ›´å¤šç©ºé—´
        x_domain = [
            x_spacing + col * (subplot_width + x_spacing),
            x_spacing + col * (subplot_width + x_spacing) + subplot_width
        ]

        # è°ƒæ•´yåŸŸä½ç½®ï¼Œä¸ºåº•éƒ¨ç•™å‡ºæ›´å¤šç©ºé—´
        y_domain = [
            1.0 - y_spacing - (row + 1) * (subplot_height + y_spacing) + 0.05,  # ä¸Šç§»
            1.0 - y_spacing - row * (subplot_height + y_spacing) - 0.05  # ä¸‹ç§»ï¼Œä¸ºåº•éƒ¨ç•™ç©ºé—´
        ]

        # Get colors for nodes and links
        node_colors = []
        for label in flow_data['labels']:
            node_colors.append(get_node_color(label, '100%'))

        link_colors = []
        for src_idx in flow_data['sources']:
            src_label = flow_data['labels'][src_idx]
            link_colors.append(get_link_color_with_alpha(src_label, alpha=0.4))

        # Add Sankey trace with optimized parameters for grid
        fig.add_trace(go.Sankey(
            arrangement='snap',
            domain=dict(x=x_domain, y=y_domain),
            node=dict(
                pad=5,  # å‡å°é—´è·ä»¥é€‚åº”ç½‘æ ¼
                thickness=10,  # å‡å°åŽšåº¦
                line=dict(color="white", width=1),
                label=flow_data['labels'],
                color=node_colors,
                x=flow_data['x_positions'],
                y=flow_data['y_positions']
            ),
            link=dict(
                source=flow_data['sources'],
                target=flow_data['targets'],
                value=flow_data['values'],
                color=link_colors
            )
        ))

        # Add scenario title - è°ƒæ•´ä½ç½®é¿å…é‡å
        total_flow = sum(flow_data['values'])
        annotations.append(dict(
            text=f"<b>{scenario_name}</b><br>Total: {total_flow:.1f} GW",
            x=(x_domain[0] + x_domain[1]) / 2,
            y=y_domain[1] + 0.2,  # ç¨å¾®ä¸Šç§»æ ‡é¢˜
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14, family='Arial', color='#333'),
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
        width=total_width,
        height=total_height,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        annotations=annotations,
        margin=dict(t=50, b=350, l=50, r=50)  # æ˜¾è‘—å¢žåŠ åº•éƒ¨è¾¹è·
    )

    # Save combined grid
    output_file = output_dir / f"sankey_{technology}_all_scenarios_2035.png"
    fig.write_image(str(output_file), scale=3)  # æé«˜åˆ†è¾¨çŽ‡
    print(f"Saved combined grid: {output_file}")

    output_html = output_dir / f"sankey_{technology}_all_scenarios_2035.html"
    fig.write_html(str(output_html))
    print(f"Saved combined grid: {output_html}")

def create_sankey_grid_combined(prod_df, trans_df, technologies, material_name, scenarios, output_dir):
    """
    Create individual Sankey diagrams for each scenario using combined technologies.

    Parameters:
    -----------
    prod_df : DataFrame
        Production flow data
    trans_df : DataFrame
        Transport flow data
    technologies : list
        List of technology names to combine
    material_name : str
        Material name for display (e.g., 'Copper', 'Nickel')
    scenarios : dict
        Dictionary mapping scenario column names to display names
    output_dir : Path
        Output directory for saving plots
    """
    # Create individual plots for each scenario
    for scenario_col, scenario_name in scenarios.items():
        flow_data = prepare_combined_flows_for_sankey(prod_df, trans_df, technologies, scenario_col)

        if not flow_data['sources']:
            print(f"No data for {material_name} - {scenario_name}")
            continue

        title = f"{material_name} - {scenario_name} (2035)"

        fig = create_sankey_figure(flow_data, title)

        if fig:
            # Save as PNG
            output_file = output_dir / f"sankey_{material_name}_{scenario_name}_2035.png"
            fig.write_image(str(output_file), scale=2)
            print(f"Saved: {output_file}")

            # Save as HTML for interactivity
            output_html = output_dir / f"sankey_{material_name}_{scenario_name}_2035.html"
            fig.write_html(str(output_html))
            print(f"Saved: {output_html}")

def create_combined_scenario_grid_combined(prod_df, trans_df, technologies, material_name, scenarios, output_dir,
                                          n_cols=4, x_spacing=0.03, y_spacing=0.12):
    """
    Create a single figure with all scenarios in a grid layout for combined technologies.

    Parameters:
    -----------
    prod_df : DataFrame
        Production flow data
    trans_df : DataFrame
        Transport flow data
    technologies : list
        List of technology names to combine
    material_name : str
        Material name for display (e.g., 'Copper', 'Nickel')
    scenarios : dict
        Dictionary mapping scenario column names to display names
    output_dir : Path
        Output directory for saving plots
    n_cols : int
        Number of columns in the grid (default: 4 for single row)
    x_spacing : float
        Horizontal spacing between subplots
    y_spacing : float
        Vertical spacing between subplots
    """
    n_scenarios = len(scenarios)
    n_rows = (n_scenarios + n_cols - 1) // n_cols

    # åŠ¨æ€è®¡ç®—æ•´ä½"é«˜åº¦ - ä¸ºåº•éƒ¨ç•™å‡ºæ›´å¤šç©ºé—´
    base_height_per_plot = 900
    extra_height_for_bottom = 300
    total_height = (base_height_per_plot * n_rows) + extra_height_for_bottom
    total_width = 2200  # å¢žåŠ å®½åº¦ä»¥å®¹çº³4ä¸ªå›¾è¡¨

    # Create figure
    fig = go.Figure()
    annotations = []

    for i, (scenario_col, scenario_name) in enumerate(scenarios.items()):
        flow_data = prepare_combined_flows_for_sankey(prod_df, trans_df, technologies, scenario_col)

        if not flow_data['sources']:
            continue

        # Calculate position in grid
        row = i // n_cols
        col = i % n_cols

        # Calculate subplot dimensions with more vertical space
        subplot_width = (1.0 - (n_cols + 1) * x_spacing) / n_cols
        subplot_height = (1.0 - (n_rows + 1) * y_spacing) / n_rows

        # Calculate domain positions - ä¸ºåº•éƒ¨ç•™å‡ºæ›´å¤šç©ºé—´
        x_domain = [
            x_spacing + col * (subplot_width + x_spacing),
            x_spacing + col * (subplot_width + x_spacing) + subplot_width
        ]

        # è°ƒæ•´yåŸŸä½ç½®ï¼Œä¸ºåº•éƒ¨ç•™å‡ºæ›´å¤šç©ºé—´
        y_domain = [
            1.0 - y_spacing - (row + 1) * (subplot_height + y_spacing) + 0.05,  # ä¸Šç§»
            1.0 - y_spacing - row * (subplot_height + y_spacing) - 0.05  # ä¸‹ç§»ï¼Œä¸ºåº•éƒ¨ç•™ç©ºé—´
        ]

        # Get colors for nodes and links
        node_colors = []
        for label in flow_data['labels']:
            node_colors.append(get_node_color(label, '100%'))

        link_colors = []
        for src_idx in flow_data['sources']:
            src_label = flow_data['labels'][src_idx]
            link_colors.append(get_link_color_with_alpha(src_label, alpha=0.4))

        # Add Sankey trace with optimized parameters for grid
        fig.add_trace(go.Sankey(
            arrangement='snap',
            domain=dict(x=x_domain, y=y_domain),
            node=dict(
                pad=5,  # å‡å°é—´è·ä»¥é€‚åº"ç½'æ ¼
                thickness=10,  # å‡å°åŽšåº¦
                line=dict(color="white", width=1),
                label=flow_data['labels'],
                color=node_colors,
                x=flow_data['x_positions'],
                y=flow_data['y_positions']
            ),
            link=dict(
                source=flow_data['sources'],
                target=flow_data['targets'],
                value=flow_data['values'],
                color=link_colors
            )
        ))

        # Add scenario title - è°ƒæ•´ä½ç½®é¿å…é‡å
        total_flow = sum(flow_data['values'])
        annotations.append(dict(
            text=f"<b>{scenario_name}</b><br>Total: {total_flow:.1f} GW",
            x=(x_domain[0] + x_domain[1]) / 2,
            y=y_domain[1] + 0.2,  # ç¨å¾®ä¸Šç§»æ ‡é¢˜
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14, family='Arial', color='#333'),
            align='center',
            xanchor='center'
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>{material_name} - All Scenarios (2035)</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=20, family='Arial', weight='bold', color='#333')
        ),
        width=total_width,
        height=total_height,
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        annotations=annotations,
        margin=dict(t=50, b=350, l=50, r=50)  # æ˜¾è'—å¢žåŠ åº•éƒ¨è¾¹è·
    )

    # Save combined grid
    output_file = output_dir / f"sankey_{material_name}_all_scenarios_2035.png"
    fig.write_image(str(output_file), scale=3)  # æé«˜åˆ†è¾¨çŽ‡
    print(f"Saved combined grid: {output_file}")

    output_html = output_dir / f"sankey_{material_name}_all_scenarios_2035.html"
    fig.write_html(str(output_html))
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
    scenariosm = {
        'value_scenario_': 'Base',
        'value_scenario_DemandMet': 'DemandMet',
        'value_scenario_SelfSuff40': 'SelfSuff40',
        'value_scenario_Recycling': 'Recycling'
    }
    # Define technologies
    technologies = ['HP_assembly', 'HEX_manufacturing', 'Compressor_manufacturing']
    material_technologies = {
        'Copper': ['Copper_recycling', 'Copper_refinement'],
        'Nickel': ['Nickel_recycling', 'Nickel_refinement']
    }
    # Create Sankey plots for each technology
    for technology in technologies:
        print(f"\nCreating Sankey plots for {technology}...")

        # 先检查数据量
        for scenario_col, scenario_name in scenarios.items():
            flow_data = prepare_flows_for_sankey(prod_df, trans_df, technology, scenario_col)
            n_nodes = len(flow_data['labels'])
            n_flows = len(flow_data['sources'])
            print(f"{scenario_name}: {n_nodes} nodes, {n_flows} flows")

        # Create individual scenario plots
        create_sankey_grid(prod_df, trans_df, technology, scenarios, output_dir)

        # Create combined grid plot
        create_combined_scenario_grid(prod_df, trans_df, technology, scenarios, output_dir, n_cols=4)

    for material_name, tech_list in material_technologies.items():
        for scenario_col, scenario_name in scenariosm.items():
            flow_data = prepare_combined_flows_for_sankey(prod_df, trans_df, tech_list, scenario_col)
            n_nodes = len(flow_data['labels'])
            n_flows = len(flow_data['sources'])
            print(f"{scenario_name}: {n_nodes} nodes, {n_flows} flows")

            # Create individual scenario plots
        create_sankey_grid_combined(prod_df, trans_df, tech_list, material_name, scenariosm, output_dir)

        # Create combined grid plot
        create_combined_scenario_grid_combined(prod_df, trans_df, tech_list, material_name, scenariosm, output_dir,
                                               n_cols=4)

    print(f"\nAll Sankey plots saved to: {output_dir}")


if __name__ == "__main__":
    main()