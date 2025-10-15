"""
:Title:        ZEN-GARDEN
:Created:      October-2021
:Authors:      Alissa Ganter (aganter@ethz.ch),
               Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Compilation  of the optimization problem.
"""
import cProfile
import importlib.util
import logging
import os
from collections import defaultdict
import importlib

from .model.optimization_setup import OptimizationSetup
from .postprocess.postprocess import Postprocess
from .utils import setup_logger, InputDataChecks, StringUtils, ScenarioUtils, OptimizationError
from .preprocess.unit_handling import Scaling
import numpy as np
import concurrent.futures
import importlib.metadata
import logging
import os
from .model.optimization_setup import OptimizationSetup
from .utils import setup_logger, InputDataChecks, StringUtils, ScenarioUtils, OptimizationError
from .parallel_processor import ParallelProcessor
from .postprocess.postprocess import Postprocess

# we setup the logger here
setup_logger()


def main(config, dataset_path=None, job_index=None):
    """Main function for ZEN garden"""
    version = importlib.metadata.version("zen-garden")
    logging.info(f"Running ZEN-garden version: {version}")
    logging.propagate = False

    if dataset_path is not None:
        config.analysis["dataset"] = dataset_path

    config.analysis["dataset"] = os.path.abspath(config.analysis['dataset'])
    config.analysis["folder_output"] = os.path.abspath(config.analysis['folder_output'])

    input_data_checks = InputDataChecks(config=config, optimization_setup=None)
    input_data_checks.check_dataset()
    input_data_checks.read_system_file(config)
    input_data_checks.check_technology_selections()
    input_data_checks.check_year_definitions()

    scenarios, elements = ScenarioUtils.get_scenarios(config, job_index)
    model_name, out_folder = StringUtils.setup_model_folder(config.analysis, config.system)
    ScenarioUtils.clean_scenario_folder(config, out_folder)

    processor = ParallelProcessor(
        config=config,
        input_data_checks=input_data_checks,  # Added input_data_checks
        model_name=model_name  # Added model_name
    )
    results = processor.run_parallel(scenarios, elements)

    for result in results:
        logging.info(result)

    logging.info("--- Optimization finished ---")

    return "optimization_setup finished"
