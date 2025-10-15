import concurrent.futures
import torch
import logging
import os
from typing import List, Dict, Any
import json


class ParallelProcessor:
    def __init__(self, config: Dict[str, Any], input_data_checks, model_name, use_gpu: bool = False):
        self.config = config
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.input_data_checks = input_data_checks
        self.model_name = model_name

        # Ensure output directory exists
        os.makedirs(self.config.analysis['folder_output'], exist_ok=True)

    @staticmethod
    def _worker_init():
        """Initialize worker process with required imports"""
        global OptimizationSetup, StringUtils, OptimizationError, Postprocess
        from zen_garden.model.optimization_setup import OptimizationSetup
        from zen_garden.utils import StringUtils, OptimizationError
        from zen_garden.postprocess.postprocess import Postprocess

    def process_scenario(self, scenario: str, scenario_data: Dict) -> str:
        self._worker_init()  # Ensure imports are available in worker process
        try:
            optimization_setup = OptimizationSetup(
                self.config,
                scenario_dict=scenario_data,
                input_data_checks=self.input_data_checks
            )

            steps_horizon = optimization_setup.get_optimization_horizon()

            for step in steps_horizon:
                StringUtils.print_optimization_progress(scenario, steps_horizon, step, system=self.config.system)
                optimization_setup.overwrite_time_indices(step)
                optimization_setup.construct_optimization_problem()

                if self.config.solver["use_scaling"]:
                    optimization_setup.scaling.run_scaling()
                elif self.config.solver["analyze_numerics"]:
                    optimization_setup.scaling.analyze_numerics()

                optimization_setup.solve()

                if not optimization_setup.optimality:
                    optimization_setup.write_IIS()
                    raise OptimizationError(optimization_setup.model.termination_condition)

                if self.config.solver["use_scaling"]:
                    optimization_setup.scaling.re_scale()

                optimization_setup.add_results_of_optimization_step(step)

                scenario_name, subfolder, param_map = StringUtils.generate_folder_path(
                    config=self.config,
                    scenario=scenario,
                    scenario_dict=scenario_data,
                    steps_horizon=steps_horizon,
                    step=step
                )

                subfolder = os.path.join(self.config.analysis['folder_output'], os.path.basename(subfolder))
                output_path = os.path.abspath(self.config.analysis['folder_output'])
                subfolder_path = os.path.abspath(subfolder)
                if not subfolder_path.startswith(output_path):
                    raise ValueError(f"Subfolder {subfolder_path} is outside output directory {output_path}")

                os.makedirs(subfolder_path, exist_ok=True)

                # Save analysis.json
                analysis_path = os.path.join(subfolder, 'analysis.json')
                analysis_data = {
                    "scenario": scenario,
                    "step": step,
                    "param_map": param_map,
                    "config": {k: str(v) for k, v in self.config.__dict__.items() if
                               isinstance(v, (str, int, float, bool))}
                }
                with open(analysis_path, 'w') as f:
                    json.dump(analysis_data, f, indent=2)

                Postprocess(
                    optimization_setup,
                    scenarios=self.config.scenarios,
                    subfolder=subfolder,
                    model_name=self.model_name,
                    scenario_name=scenario_name,
                    param_map=param_map
                )

            return f"Scenario {scenario} completed successfully"

        except Exception as e:
            logging.error(f"Error in scenario {scenario}: {str(e)}", exc_info=True)
            return f"Error in scenario {scenario}: {str(e)}"

    def run_parallel(self, scenarios: List[str], elements) -> List[str]:
        elements_dict = dict(zip(scenarios, elements))

        max_workers = min(len(scenarios),
                          torch.cuda.device_count() if self.use_gpu else os.cpu_count())

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for scenario in scenarios:
                scenario_data = elements_dict[scenario]
                futures.append(
                    executor.submit(self.process_scenario, scenario, scenario_data)
                )

            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    logging.info(result)
                except Exception as e:
                    error_msg = f"Failed scenario: {str(e)}"
                    logging.error(error_msg)
                    results.append(error_msg)

        return results