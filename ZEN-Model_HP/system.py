"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
==========================================================================================================================================================================="""

## System - Default dictionary
system = dict()

## System - settings update compared to default values
system['set_conversion_technologies']     = ["HP_assembly",
                                             "HEX_manufacturing",
                                             "Aluminium_production",
                                             "Steel_production",
                                             "Bauxite_mining",
                                             "Nickel_mining",
                                            "Nickel_recycling",
                                             "Copper_mining",
                                            "Copper_recycling",
                                             "Iron_mining",
                                             "Copper_refinement",
                                             "Compressor_manufacturing"]
system['set_storage_technologies']        = ["Aluminium_storage",
                                             "Bauxite_storage",
                                             "HEX_storage",
                                             "HP_storage",
                                             "Nickel_storage",
                                             "Copper_storage",
                                             "Compressor_storage",
                                             "Iron_storage",
                                             "Steel_storage"]
system['set_transport_technologies']      = ["Bauxite_transport",
                                             "Aluminium_transport",
                                             "HEX_transport",
                                             "HP_transport",
                                             "Nickel_transport",
                                             "Copper_ore_transport",
                                             "Copper_transport",
                                             "Compressor_transport",
                                             "Iron_transport",
                                             "Steel_transport",
                                             "Refrigerant_transport"]

system['set_nodes']                      = ["AUS", "AUT", "CHN", "CZE", "ITA", "DEU", "JPN", "KOR", "ROE", "ROW", "USA", "BRA"]

system['set_regions'] = {
    "europe": ["AUT", "DEU", "CZE","ITA","ROE"],
    "asia_pacific": ["AUS", "JPN", "KOR"],
    "americas": ["USA", "BRA"],
    "china": ["CHN"],
    "rest": ["ROW"]
}

# time steps
system["reference_year"]                 = 2022
system["unaggregated_time_steps_per_year"]  = 8760
system["aggregated_time_steps_per_year"]    = 1
system["conduct_time_series_aggregation"]  = True
system["conduct_scenario_analysis"] = True
system["conduct_uncertainty_capacity"] = False
system["conduct_uncertainty_opex"] = False
system["conduct_uncertainty_capex"] =False
system["conduct_uncertainty_combined"] = False

system["optimized_years"]                = 14
system["interval_between_years"]          = 1
system["use_rolling_horizon"]             = False
system["years_in_rolling_horizon"]         = 1
