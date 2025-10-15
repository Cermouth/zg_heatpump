from pathlib import Path
import pandas as pd
import numpy as np
# Try different matplotlib backend configurations
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
from zen_garden.postprocess.results.results import Results


# Configure paths
RESULTS_PATH = './outputs/ZEN-Model_HP'
OUTPUT_BASE_DIR = 'parameter_results'

# List of parameters to export
parameters = [
    "capacity",
    "demand",
    "flow_conversion_output",
    "flow_import",
    "flow_transport",
    "cost_opex_total",
    "shed_demand"
]

def combine_scenario_data(df_dict):
    """
    Combines data from multiple scenarios into a single DataFrame with scenario columns
    """
    # Initialize with the first scenario to get the structure
    first_scenario = list(df_dict.keys())[0]
    first_df = df_dict[first_scenario]

    # Create a base DataFrame with index columns
    if isinstance(first_df.index, pd.MultiIndex):
        index_names = first_df.index.names
    else:
        index_names = [first_df.index.name]

    # Reset index to get all columns, including index levels
    combined_df = first_df.reset_index()

    # Rename the value column to include scenario name
    value_cols = [col for col in combined_df.columns if col not in index_names]
    for col in value_cols:
        combined_df.rename(columns={col: f"value_{first_scenario}"}, inplace=True)

    # Add data from other scenarios
    for scenario_name, df in df_dict.items():
        if scenario_name == first_scenario:
            continue

        temp_df = df.reset_index()
        # Only keep the value columns from additional scenarios
        for col in value_cols:
            combined_df[f"value_{scenario_name}"] = temp_df[col]

    return combined_df


def create_summary_stats(df):
    """
    Create summary statistics for the combined data
    """
    """
       Create summary statistics for the combined data
       """
    try:
        # Identify value columns (scenario columns)
        value_cols = [col for col in df.columns if col.startswith('value_')]

        if not value_cols:
            print("No value columns found for statistics calculation")
            return None

        # Calculate basic statistics for numeric columns only
        numeric_df = df[value_cols].apply(pd.to_numeric, errors='coerce')

        summary_dict = {
            'mean': numeric_df.mean().round(4),
            'min': numeric_df.min().round(4),
            'max': numeric_df.max().round(4),
            'median': numeric_df.median().round(4)
        }
        # Only calculate std if we have more than one value
        if len(value_cols) > 1:
            summary_dict['std'] = numeric_df.std().round(4)
            # Calculate coefficient of variation where mean is not zero
            means = summary_dict['mean']
            stds = summary_dict['std']
            cv = pd.Series(np.where(means != 0, stds / means, np.nan), index=means.index)
            summary_dict['coefficient_of_variation'] = cv.round(4)

        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_dict)

        return summary_df.T

    except Exception as e:
        print(f"Error in summary statistics calculation: {str(e)}")
        return None


def create_flow_plots(data_file, parameter: str, technology, carrier: str = "HP"):
    """
    Create stacked bar plots for flow data by scenario, mode, and region
    """
    try:
        # Set up Seaborn style
        sns.set_theme(style="whitegrid")
        output_dir = Path(
            "./HP_Model/Final_plots")

        # Define color palette
        palette = ["#b7afd5", "#becbd8", "#bfcc67", "#b2cfa9", "#9c91c7",
                   "#b2cde9", "#9fc7c6", "#ae879c", "#bc556d", "#d596bd",
                   "#ada599", "#ddd7c6"]

        # Dictionary for scenario name mapping
        scenario_names = {
            0: 'Base Case',
            1: 'Scenario 1',
            2: 'Scenario 2',
            3: 'Scenario 3',
            4: 'Scenario 4'
        }

        df = pd.read_csv(data_file)

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
            category_cols = ['location']
        else:
            time_col = 'time_operation'
            category_cols = ['node']
            if 'carrier' in df.columns:
                category_cols.append('carrier')

        # Modified subplot layout
        n_rows = 3
        n_cols = 2

        # Create figure with adjusted size
        fig = plt.figure(figsize=(15, 16))

        first_handles = []
        first_labels = []

        for idx, scenario_col in enumerate(scenario_cols, 1):
            try:
                # Prepare pivot table
                pivot_data = pd.pivot_table(
                    data=df,
                    values=scenario_col,
                    index=time_col,
                    columns=category_cols,
                    fill_value=0
                )

                # Add 2022 to x-axis values if needed
                if parameter == 'capacity':
                    pivot_data.index = pivot_data.index + 2022

                ax = plt.subplot(n_rows, n_cols, idx)

                if parameter == 'capacity':
                    lines = []
                    for col_idx, column in enumerate(pivot_data.columns):
                        line = ax.plot(pivot_data.index,
                                       pivot_data[column],
                                       label=column,
                                       color=palette[col_idx % len(palette)],
                                       marker='o',
                                       linewidth=2,
                                       markersize=6)[0]
                        if idx == 1:
                            first_handles.append(line)
                            first_labels.append(column)
                else:
                    # Stacked bar plot
                    bottom = np.zeros(len(pivot_data))

                    # Filter out columns containing 'power'
                    filtered_columns = []
                    for col in pivot_data.columns:
                        col_str = str(col) if not isinstance(col, tuple) else ' - '.join(str(x) for x in col)
                        if 'power' not in col_str.lower():
                            filtered_columns.append(col)

                    for i, column in enumerate(filtered_columns):
                        bar = ax.bar(pivot_data.index,
                                     pivot_data[column],
                                     bottom=bottom,
                                     width=0.8,
                                     label=str(column) if not isinstance(column, tuple) else column[0],
                                     color=palette[i % len(palette)],
                                     edgecolor='none',
                                     linewidth=0)
                        bottom += pivot_data[column]

                        if idx == 1:
                            first_handles.append(bar)
                            first_labels.append(str(column) if not isinstance(column, tuple) else column[0])

                # Set title using the mapping
                ax.set_title(scenario_names.get(idx - 1, f'Scenario {idx - 1}'), fontsize=14)
                ax.set_ylabel('Production (GW)', fontsize=13)
                ax.set_xlabel('')
                ax.grid(True, alpha=0.2)

                # Fix for xticks rotation and size
                if len(pivot_data.index) > 5:
                    ax.tick_params(axis='x', rotation=45)
                    for label in ax.get_xticklabels():
                        label.set_fontsize(13)

                # Remove individual subplot legends
                if ax.get_legend():
                    ax.get_legend().remove()

            except Exception as e:
                print(f"Error creating subplot for scenario {idx}: {str(e)}")
                continue

        # Add single legend at the bottom
        if first_handles and first_labels:
            fig.legend(handles=first_handles,
                       labels=first_labels,
                       bbox_to_anchor=(0.5, -0.05),
                       loc='lower center',
                       borderaxespad=0,
                       frameon=False,
                       ncol=6)

        plt.tight_layout()

        # Save the plot
        file_prefix = technology if 'technology' in df.columns else carrier
        plot_file = output_dir / f"{parameter}_{file_prefix}_all_scenarios.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error in create_flow_plots: {str(e)}")


