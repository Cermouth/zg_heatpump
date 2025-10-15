from pathlib import Path
from plotting import *
import pandas as pd
from test2 import create_flow_plots

def run_plotting():
    """Run plotting functions independently using existing CSV files"""
    # Define your base directories
    base_dir = Path('parameter_results')
    uncertainty_dir = Path('uncertainty_results')

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

    # Example usage
    opex_yearly_file = "./parameter_results/cost_opex_yearly/cost_opex_yearly_scenarios.csv"
    capex_yearly_file = "./parameter_results/cost_capex/cost_capex_scenarios.csv"
    production_file = "./parameter_results/flow_conversion_output/flow_conversion_output_scenarios.csv"
    capacity_extension = "./parameter_results/capacity_addition/capacity_addition_scenarios.csv"
    capacity_file = "./parameter_results/capacity/capacity_scenarios.csv"
    opex_data = "./parameter_results/cost_opex_total/cost_opex_total_scenarios.csv"
    capex_data = "./parameter_results/cost_capex_total/cost_capex_total_scenarios.csv"
    transport_data = "./parameter_results/flow_transport/flow_transport_scenarios.csv"


    plot_country_costs_scenarios(opex_yearly_file, capex_yearly_file)

    # #plot uncertainty boxplot
    production_MC_dir = "./uncertainty_results/flow_conversion_output"
    opex_MC_dir = "./uncertainty_results/cost_opex_total"
    capex_MC_dir = "./uncertainty_results/cost_capex_total"
    # Run the analysis
    # plot_production_uncertainty_cumulative(production_MC_dir, "HP_assembly")
    # plot_production_uncertainty_combined(production_MC_dir, "HP_assembly")
    #plot_total_cost_uncertainty_violins(opex_MC_dir,capex_MC_dir)
    #plot_total_cost_uncertainty_errorbars(opex_MC_dir,capex_MC_dir)

    # # #plot necessary
    plot_flow_conversion(production_file)
    plot_flow_conversion(production_file, 'value_scenario_Recycling')

    #
    # #for scenario
    # #pie chart
    plot_pie_scenarios(production_file)
    # #temporal evolution
    plot_scenario_results(production_file, "flow_conversion_output", "HP_assembly", "HP")
    plot_scenario_results(production_file, "flow_conversion_output", "Compressor_manufacturing", "Compressor")
    plot_scenario_results(production_file, "flow_conversion_output", "HEX_manufacturing", "HEX")
    #
    plot_total_costs_scenarios(opex_data, capex_data)
    #
    plot_capacity_timeseries(capacity_file)
    plot_yearly_capacity_extension(capacity_extension)
    plot_temporal_costs(opex_yearly_file, capex_yearly_file)

    # #
    plot_cumulative_capacity_extension(capacity_extension)

    plot_timestep_costs_base(opex_data, capex_data)
    plot_timestep_costs_comparison(opex_data, capex_data, save=True)
    #need yearly files to get nodal resolution
    plot_nodal_scenario_costs(opex_yearly_file, capex_yearly_file)

    plot_transport_flow(transport_data, parameter="transport")


    #for appendix
    create_flow_plots(capacity_file,"capacity","HP_assembly")
    create_flow_plots(capacity_file, "capacity", "Compressor_manufacturing")
    create_flow_plots(capacity_file, "capacity", "HEX_manufacturing")

if __name__ == "__main__":
    import pandas as pd  # Import here to avoid circular imports
    run_plotting()