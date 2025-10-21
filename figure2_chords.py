import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
import numpy as np
from pathlib import Path

# ETH Color Scheme
ETH_COLORS = {
    'ETH Blue': {'120%': '#08407E', '100%': '#215CAF', '80%': '#4D7DBF', '60%': '#7A9DCF', '40%': '#A6BEDF',
                 '20%': '#D3DEEF', '10%': '#E9EFF7'},
    'ETH Petrol': {'120%': '#00596D', '100%': '#007894', '80%': '#3395AB', '60%': '#66AFC0', '40%': '#99CAD5',
                   '20%': '#CCE4EA', '10%': '#E7F4F7'},
    'ETH Green': {'120%': '#365213', '100%': '#627313', '80%': '#818F42', '60%': '#A1AB71', '40%': '#C0C7A1',
                  '20%': '#E0E3D0', '10%': '#EEF1E7'},
    'ETH Bronze': {'120%': '#704F12', '100%': '#8E6713', '80%': '#A58542', '60%': '#BBA471', '40%': '#D2C2A1',
                   '20%': '#E8E1D0', '10%': '#F4F0E7'},
    'ETH Red': {'120%': '#96272D', '100%': '#B7352D', '80%': '#C55D57', '60%': '#D48681', '40%': '#E2AEAB',
                '20%': '#F1D7D5', '10%': '#F8EBEA'},
    'ETH Purple': {'120%': '#8C0A59', '100%': '#A7117A', '80%': '#B73B92', '60%': '#CA6CAE', '40%': '#DC9EC9',
                   '20%': '#EFD0E3', '10%': '#F8E8F3'},
    'ETH Grey': {'120%': '#575757', '100%': '#6F6F6F', '80%': '#8C8C8C', '60%': '#A9A9A9', '40%': '#C5C5C5',
                 '20%': '#E2E2E2', '10%': '#F1F1F1'}
}

# Region to color mapping
REGION_COLORS = {
    'CHN': ETH_COLORS['ETH Purple']['120%'],
    'AUS': ETH_COLORS['ETH Bronze']['100%'],
    'ROW': ETH_COLORS['ETH Grey']['100%'],
    'JPN': ETH_COLORS['ETH Purple']['80%'],
    'ROA': ETH_COLORS['ETH Purple']['20%'],
    'EUR': ETH_COLORS['ETH Blue']['100%'],
    'USA': ETH_COLORS['ETH Petrol']['100%'],
    'BRA': ETH_COLORS['ETH Red']['60%'],
    'KOR': ETH_COLORS['ETH Purple']['40%'],
    'DEU': ETH_COLORS['ETH Blue']['120%'],
    'ITA': ETH_COLORS['ETH Blue']['60%'],
    'AUT': ETH_COLORS['ETH Blue']['80%'],
    'CZE': ETH_COLORS['ETH Blue']['20%'],
    'ROE': ETH_COLORS['ETH Blue']['40%'],
}


def load_and_process_data(production_file, transport_file, target_year=2035):
    """Load production and transport data for the target year."""
    prod_df = pd.read_csv(production_file)
    trans_df = pd.read_csv(transport_file)

    year_offset = target_year - 2022
    prod_df = prod_df[prod_df['time_operation'] == year_offset].copy()
    trans_df = trans_df[trans_df['time_operation'] == year_offset].copy()

    return prod_df, trans_df


def prepare_chord_data(prod_df, trans_df, technology, scenario_col):
    """
    Prepare flow data for chord diagram.

    Returns:
    --------
    DataFrame with columns: source, target, value
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

    # Filter transport data
    trans_data = trans_df[trans_df['technology'] == transport_tech].copy()

    # Build flow dataframe
    flows = []
    for _, row in trans_data.iterrows():
        if scenario_col in row and pd.notna(row[scenario_col]) and 'edge' in row:
            flow_value = float(row[scenario_col])
            if flow_value > 0.01:  # Filter very small flows
                try:
                    exporter, importer = row['edge'].split('-')
                    flows.append({
                        'source': exporter,
                        'target': importer,
                        'value': flow_value
                    })
                except (ValueError, AttributeError):
                    continue

    if not flows:
        return None

    flow_df = pd.DataFrame(flows)
    return flow_df


def hex_to_rgba(hex_color, alpha=0.6):
    """Convert hex color to RGBA tuple."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return rgb + (alpha,)


def draw_chord_diagram(ax, matrix, names, colors, gap=0.03):
    """
    Draw a chord diagram on given axes.

    Parameters:
    -----------
    ax : matplotlib axes
        Axes to draw on
    matrix : numpy array
        Flow matrix (n x n)
    names : list
        Node names
    colors : list
        Node colors (hex)
    gap : float
        Gap between nodes (in radians)
    """
    n = len(names)

    # Calculate total flow for each node (use max of in/out to avoid double counting)
    outflows = matrix.sum(axis=1)
    inflows = matrix.sum(axis=0)
    node_totals = np.maximum(outflows, inflows)
    total = node_totals.sum()

    if total == 0:
        return

    # Calculate angles for each node
    node_angles = []
    current_angle = np.pi / 2  # Start at top
    total_gap = gap * n

    for i in range(n):
        node_width = (2 * np.pi - total_gap) * (node_totals[i] / total)
        start_angle = current_angle
        end_angle = current_angle + node_width
        node_angles.append((start_angle, end_angle))
        current_angle = end_angle + gap

    # Draw nodes (arcs)
    radius = 1.0
    for i, (start, end) in enumerate(node_angles):
        theta = np.linspace(start, end, 100)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        ax.plot(x, y, color=colors[i], linewidth=15, solid_capstyle='butt')

        # Add labels
        mid_angle = (start + end) / 2
        label_radius = radius + 0.15
        label_x = label_radius * np.cos(mid_angle)
        label_y = label_radius * np.sin(mid_angle)

        # Adjust text rotation
        rotation = np.degrees(mid_angle) - 90
        if rotation < -90 or rotation > 90:
            rotation += 180

        ax.text(label_x, label_y, names[i],
                ha='center', va='center',
                rotation=rotation,
                fontsize=10, fontweight='bold')

    # Draw ribbons (connections)
    for i in range(n):
        # Track cumulative angles for this source node
        source_start, source_end = node_angles[i]
        current_source_angle = source_start

        for j in range(n):
            if matrix[i, j] > 0:
                # Calculate flow angles on source node (based on outflow)
                outflow_total = outflows[i]
                if outflow_total > 0:
                    flow_width_source = (source_end - source_start) * (matrix[i, j] / outflow_total)
                    flow_start_i = current_source_angle
                    flow_end_i = current_source_angle + flow_width_source
                    current_source_angle = flow_end_i
                else:
                    continue

                # Calculate flow angles on target node (based on inflow)
                target_start, target_end = node_angles[j]
                inflow_total = inflows[j]

                if inflow_total > 0:
                    # Find position based on flows from all sources to j
                    incoming_before = sum(matrix[k, j] for k in range(i))
                    flow_start_j = target_start + (target_end - target_start) * (incoming_before / inflow_total)
                    flow_end_j = flow_start_j + (target_end - target_start) * (matrix[i, j] / inflow_total)
                else:
                    continue

                # Create bezier curve for the ribbon
                p0 = (radius * np.cos(flow_start_i), radius * np.sin(flow_start_i))
                p1 = (radius * np.cos(flow_end_i), radius * np.sin(flow_end_i))
                p2 = (radius * np.cos(flow_start_j), radius * np.sin(flow_start_j))
                p3 = (radius * np.cos(flow_end_j), radius * np.sin(flow_end_j))

                # Control points (pulled toward center)
                ctrl_factor = 0.5
                c0 = (p0[0] * ctrl_factor, p0[1] * ctrl_factor)
                c1 = (p1[0] * ctrl_factor, p1[1] * ctrl_factor)
                c2 = (p2[0] * ctrl_factor, p2[1] * ctrl_factor)
                c3 = (p3[0] * ctrl_factor, p3[1] * ctrl_factor)

                # Create path
                verts = [p0, c0, c2, p2, p3, c3, c1, p1, p0]
                codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                         MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                         MplPath.CLOSEPOLY]

                path = MplPath(verts, codes)
                patch = patches.PathPatch(path, facecolor=hex_to_rgba(colors[i], 0.5),
                                          edgecolor='none', linewidth=0)
                ax.add_patch(patch)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')


def create_chord_diagram(flow_df, title, unit='GW', output_file=None):
    """
    Create a chord diagram using matplotlib.

    Parameters:
    -----------
    flow_df : DataFrame
        DataFrame with columns: source, target, value
    title : str
        Title for the plot
    unit : str
        Unit for display
    output_file : str
        Path to save the figure

    Returns:
    --------
    matplotlib figure
    """
    if flow_df is None or flow_df.empty:
        print(f"No flow data available for {title}")
        return None

    # Get unique nodes
    all_nodes = list(set(flow_df['source'].unique().tolist() + flow_df['target'].unique().tolist()))

    # Sort nodes by region
    region_groups = {
        'Asia': ['CHN', 'JPN', 'KOR', 'ROA'],
        'Europe': ['EUR', 'DEU', 'ITA', 'AUT', 'CZE', 'ROE'],
        'Americas': ['USA', 'BRA'],
        'Oceania': ['AUS'],
        'Other': ['ROW']
    }

    sorted_nodes = []
    for region in ['Asia', 'Europe', 'Americas', 'Oceania', 'Other']:
        sorted_nodes.extend([n for n in region_groups[region] if n in all_nodes])

    n = len(sorted_nodes)
    node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}

    # Create matrix
    matrix = np.zeros((n, n))
    for _, row in flow_df.iterrows():
        if row['source'] in node_to_idx and row['target'] in node_to_idx:
            i = node_to_idx[row['source']]
            j = node_to_idx[row['target']]
            matrix[i, j] = row['value']

    # Get colors
    colors = [REGION_COLORS.get(node, '#999999') for node in sorted_nodes]

    total_flow = flow_df['value'].sum()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_chord_diagram(ax, matrix, sorted_nodes, colors)

    plt.title(f"{title}\nTotal Flow: {total_flow:.2f} {unit}",
              fontsize=14, fontweight='bold', pad=20)

    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_file}")
        plt.close(fig)

    return fig


def create_chord_grid(prod_df, trans_df, technology, scenarios, output_dir):
    """Create chord diagrams for each scenario."""
    for scenario_col, scenario_name in scenarios.items():
        flow_df = prepare_chord_data(prod_df, trans_df, technology, scenario_col)

        if flow_df is None or flow_df.empty:
            print(f"No data for {technology} - {scenario_name}")
            continue

        tech_display = technology.replace('_', ' ').title()
        title = f"{tech_display} - {scenario_name} (2035)"

        output_file = output_dir / f"chord_{technology}_{scenario_name}_2035.png"
        create_chord_diagram(flow_df, title, output_file=str(output_file))

    # Create combined grid
    create_combined_grid(prod_df, trans_df, technology, scenarios, output_dir)


def create_combined_grid(prod_df, trans_df, technology, scenarios, output_dir):
    """Create a combined figure with all scenarios in a grid."""
    n_scenarios = len(scenarios)
    n_cols = 2
    n_rows = (n_scenarios + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (scenario_col, scenario_name) in enumerate(scenarios.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        flow_df = prepare_chord_data(prod_df, trans_df, technology, scenario_col)

        if flow_df is None or flow_df.empty:
            ax.axis('off')
            continue

        # Get data
        all_nodes = list(set(flow_df['source'].unique().tolist() + flow_df['target'].unique().tolist()))

        region_groups = {
            'Asia': ['CHN', 'JPN', 'KOR', 'ROA'],
            'Europe': ['EUR', 'DEU', 'ITA', 'AUT', 'CZE', 'ROE'],
            'Americas': ['USA', 'BRA'],
            'Oceania': ['AUS'],
            'Other': ['ROW']
        }

        sorted_nodes = []
        for region in ['Asia', 'Europe', 'Americas', 'Oceania', 'Other']:
            sorted_nodes.extend([n for n in region_groups[region] if n in all_nodes])

        n = len(sorted_nodes)
        node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}

        matrix = np.zeros((n, n))
        for _, row_data in flow_df.iterrows():
            if row_data['source'] in node_to_idx and row_data['target'] in node_to_idx:
                i = node_to_idx[row_data['source']]
                j = node_to_idx[row_data['target']]
                matrix[i, j] = row_data['value']

        colors = [REGION_COLORS.get(node, '#999999') for node in sorted_nodes]

        draw_chord_diagram(ax, matrix, sorted_nodes, colors)

        tech_display = technology.replace('_', ' ').title()
        total_flow = flow_df['value'].sum()
        ax.set_title(f"{tech_display} - {scenario_name} (2035)\nTotal Flow: {total_flow:.2f} GW",
                     fontsize=12, fontweight='bold', pad=10)

    # Hide unused subplots
    for idx in range(len(scenarios), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    output_file = output_dir / f"chord_{technology}_all_scenarios_2035.png"
    plt.savefig(str(output_file), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved combined: {output_file}")
    plt.close(fig)


def main():
    """Main execution function"""

    # Define file paths
    production_file = './parameter_results/flow_conversion_output/flow_conversion_output_scenarios.csv'
    transport_file = './parameter_results/flow_transport/flow_transport_scenarios.csv'

    # Output directory
    output_dir = Path('./visualization_chord')
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

    # Create chord plots for each technology
    for technology in technologies:
        print(f"\nCreating chord diagrams for {technology}...")
        create_chord_grid(prod_df, trans_df, technology, scenarios, output_dir)

    print(f"\nAll chord diagrams saved to: {output_dir}")


if __name__ == "__main__":
    main()