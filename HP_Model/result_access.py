from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import false

from zen_garden.postprocess.results.results import Results
from plotting import plot_scenario_results, plot_uncertainty_results
import os

RESULTS_PATH = '../../zg_heatpump/outputs'
OUTPUT_BASE_DIR = '../../zg_heatpump/parameter_results'
UNCERTAINTY_DIR = '../../zg_heatpump/uncertainty_results'

parameters = [
    "capacity",
    "demand",
    "flow_conversion_output",
    "flow_import",
    "flow_transport",
    "cost_opex_total",
    "cost_opex_yearly",
    "shed_demand"
]

def get_scenario_type(scenario_names):
    """Determine if scenarios are deterministic (S1, S2, etc.) or Monte Carlo (MC)"""
    scenario_types = {'scenario': [], 'uncertainty_capacity': [], 'uncertainty_capex': [], 'uncertainty_opex': [], 'uncertainty_combined': []}

    for name in scenario_names:
        # Skip scenario 172
        if 'MC172' in name:
            continue
        if 'MC' in name and 'capacity' in name:
            scenario_types['uncertainty_capacity'].append(name)
        elif 'MC' in name and 'opex' in name:
            scenario_types['uncertainty_opex'].append(name)
        elif 'MC' in name and 'capex' in name:
            scenario_types['uncertainty_capex'].append(name)
        elif 'MC' in name and 'combined' in name:
            scenario_types['uncertainty_combined'].append(name)
        # elif any(f'S{i}' in name for i in range(1, 1000)) or name.endswith('scenario_'):
        else:
            scenario_types['scenario'].append(name)

    return scenario_types

def combine_scenario_data(df_dict):
    """Combines data from scenarios using pd.concat for better performance"""
    first_scenario = list(df_dict.keys())[0]
    first_df = df_dict[first_scenario]

    index_names = first_df.index.names if isinstance(first_df.index, pd.MultiIndex) else [first_df.index.name]

    # Create list to store DataFrames for each scenario
    scenario_dfs = []

    for scenario_name, df in df_dict.items():
        temp_df = df.reset_index()
        value_cols = [col for col in temp_df.columns if col not in index_names]

        # Keep index columns and rename value column
        scenario_data = temp_df[index_names + value_cols]
        for col in value_cols:
            scenario_data = scenario_data.rename(columns={col: f"value_{scenario_name}"})

        scenario_dfs.append(scenario_data)

    # Combine all scenarios efficiently using merge
    combined_df = scenario_dfs[0]
    for df in scenario_dfs[1:]:
        combined_df = pd.merge(combined_df, df, on=index_names)

    return combined_df


def create_summary_stats(df):
    """Create summary statistics for scenario analysis"""
    try:
        value_cols = [col for col in df.columns if col.startswith('value_')]
        if not value_cols:
            return None

        numeric_df = df[value_cols].apply(pd.to_numeric, errors='coerce')
        summary_dict = {
            'mean': numeric_df.mean().round(4),
            'min': numeric_df.min().round(4),
            'max': numeric_df.max().round(4),
            'median': numeric_df.median().round(4)
        }

        if len(value_cols) > 1:
            summary_dict['std'] = numeric_df.std().round(4)
            means = summary_dict['mean']
            stds = summary_dict['std']
            cv = pd.Series(np.where(means != 0, stds / means, np.nan), index=means.index)
            summary_dict['coefficient_of_variation'] = cv.round(4)

        return pd.DataFrame(summary_dict).T

    except Exception as e:
        print(f"Error in summary statistics calculation: {str(e)}")
        return None


def main():
    base_dir = Path(OUTPUT_BASE_DIR)
    base_dir.mkdir(exist_ok=True)
    uncertainty_dir = Path(UNCERTAINTY_DIR)
    uncertainty_dir.mkdir(exist_ok=True)

    #try:
    results = Results(RESULTS_PATH)
    parameters = results.get_component_names('variable')
    # parameters = [
    #     "capacity",
    #     "demand",
    #     "flow_conversion_output",
    #     "flow_import",
    #     "flow_transport",
    #     "cost_opex_total",
    #     "cost_opex_yearly",
    #     "shed_demand"
    # ]
    parameters2 = results.get_component_names('parameter')
    set= results.get_component_names('sets')
    timesteps = results.get_total("time_steps_operation_duration")
    capex_conversion = results.get_total("capex_specific_conversion")
    construction = results.get_total("construction_time")
    knowledge_spillover = results.get_total("knowledge_spillover_rate")
    opex_fixed = results.get_total("opex_specific_fixed")
    distance =results.get_total("distance")
    opex_variable = results.get_total("opex_specific_variable")
    opex_var_unit = results.get_unit("opex_specific_variable")
    opex_fixed_unit = results.get_unit("opex_specific_fixed")
    capex_unit = results.get_unit("capex_specific_conversion")
    unit =results.get_unit("capex_specific_conversion")
    unit_distance =results.get_unit("distance")
    units_opex_yearly = results.get_unit("cost_opex_yearly")
    units_capex_yearly = results.get_unit("cost_capex")
    units_production = results.get_unit("flow_conversion_output")
    units_production.to_csv("units_output.csv")
    units_opex_yearly.to_csv("unit_opex")
    units_capex_yearly.to_csv("unit_capex")
    opex_fixed.to_csv("opex_fixed.csv")
    opex_fixed_unit.to_csv("opex_fixed_unit")
    opex_variable.to_csv("opex_variable")
    opex_var_unit.to_csv("opex_var_unit")
    a = 1
    for parameter in parameters:
        try:
            df_dict = results.get_df(parameter)
            if df_dict is None:
                continue

            scenario_types = get_scenario_type(df_dict.keys())

            param_dir = uncertainty_dir / parameter
            param_dir.mkdir(exist_ok=True)

            # # # Process capacity MC scenarios
            if scenario_types['scenario']:
                param_dir = base_dir / parameter
                param_dir.mkdir(exist_ok=True)
                mc_dict = {k: df_dict[k] for k in scenario_types['scenario']}
                combined_df = combine_scenario_data(mc_dict)
                if combined_df is not None:
                    combined_df.to_csv(param_dir / f"{parameter}_scenarios.csv", index=True)
                    print(param_dir, parameter, "scneario analysis csv", combined_df.head())

                    #plot_scenario_results(param_dir, parameter, "HP_assembly", "HP")

            # Process capacity MC scenarios
            if scenario_types['uncertainty_capacity']:
                mc_dict = {k: df_dict[k] for k in scenario_types['uncertainty_capacity']}
                combined_df = combine_scenario_data(mc_dict)
                if combined_df is not None:
                    combined_df.to_csv(param_dir / f"{parameter}_MC_capacity.csv", index=False)
                    plot_uncertainty_results(param_dir, parameter, "capacity")

            # Process opex MC scenarios
            if scenario_types['uncertainty_opex']:
                mc_dict = {k: df_dict[k] for k in scenario_types['uncertainty_opex']}
                combined_df = combine_scenario_data(mc_dict)
                if combined_df is not None:
                    combined_df.to_csv(param_dir / f"{parameter}_MC_opex.csv", index=False)
                    plot_uncertainty_results(param_dir, parameter, "opex")

            # Process opex MC scenarios
            if scenario_types['uncertainty_capex']:
                mc_dict = {k: df_dict[k] for k in scenario_types['uncertainty_capex']}
                combined_df = combine_scenario_data(mc_dict)
                if combined_df is not None:
                    combined_df.to_csv(param_dir / f"{parameter}_MC_capex.csv", index=False)
                    plot_uncertainty_results(param_dir, parameter, "capex")

            # Process opex MC scenarios
            if scenario_types['uncertainty_combined']:
                mc_dict = {k: df_dict[k] for k in scenario_types['uncertainty_combined']}
                combined_df = combine_scenario_data(mc_dict)
                if combined_df is not None:
                    combined_df.to_csv(param_dir / f"{parameter}_MC_combined.csv", index=False)
                    plot_uncertainty_results(param_dir, parameter, "combined")

        except Exception as e:
            print(f"Error processing {parameter}: {str(e)}")
            continue

    # except Exception as e:
    #     print(f"Error initializing Results: {str(e)}")
    #     return

    print("Data export and plotting completed")

if __name__ == "__main__":
    main()