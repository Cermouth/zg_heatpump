import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime


class Validator:
    """Validator class to check if location-specific construction time is working correctly"""

    def __init__(self, optimization_setup):
        self.optimization_setup = optimization_setup
        self.model = optimization_setup.model
        self.params = optimization_setup.parameters.dict_parameters
        self.vars = optimization_setup.variables

    def run_all_checks(self):
        """Run all validation checks and return results"""
        results = {
            'input_data': self.check_input_data(),
            'investment_timing': self.check_investment_timing(),
            'capacity_addition': self.check_capacity_addition(),
            'constraint_satisfaction': self.check_constraint_satisfaction()
        }
        return pd.DataFrame(results).T

    def check_input_data(self):
        """Verify that construction times are location-specific"""
        construction_time = self.params.construction_time

        # Check if construction time has location dimension
        has_location_dim = 'set_location' in construction_time.dims

        # Check if values vary by location
        varies_by_location = len(np.unique(construction_time.values)) > 1

        return {
            'check_passed': has_location_dim and varies_by_location,
            'has_location_dimension': has_location_dim,
            'varies_by_location': varies_by_location,
            'unique_values': np.unique(construction_time.values)
        }

    def check_investment_timing(self):
        """Verify that investments respect location-specific construction times"""
        capacity_investment = self.vars["capacity_investment"].values
        capacity_addition = self.vars["capacity_addition"].values

        # Get the time indices where investments and additions occur
        investment_times = np.where(capacity_investment > 1e-6)
        addition_times = np.where(capacity_addition > 1e-6)

        # Check if the time difference matches construction time for each location
        correct_timing = True
        mismatches = []

        for inv_idx, add_idx in zip(investment_times[0], addition_times[0]):
            tech_idx = investment_times[1]
            loc_idx = investment_times[2]
            expected_delay = self.params.construction_time.values[tech_idx, loc_idx]
            actual_delay = add_idx - inv_idx

            if abs(actual_delay - expected_delay) > 1e-6:
                correct_timing = False
                mismatches.append({
                    'tech': tech_idx,
                    'location': loc_idx,
                    'expected_delay': expected_delay,
                    'actual_delay': actual_delay
                })

        return {
            'check_passed': correct_timing,
            'mismatches': mismatches
        }

    def check_capacity_addition(self):
        """Verify that capacity additions occur at correct times based on construction delays"""
        capacity_addition = self.vars["capacity_addition"].values

        # Check if any capacity additions occur before their construction time
        invalid_additions = []
        valid_timing = True

        for tech in range(capacity_addition.shape[0]):
            for loc in range(capacity_addition.shape[1]):
                construction_time = self.params.construction_time.values[tech, loc]
                earliest_possible_addition = int(np.ceil(construction_time))

                if np.any(capacity_addition[tech, loc, :earliest_possible_addition] > 1e-6):
                    valid_timing = False
                    invalid_additions.append({
                        'tech': tech,
                        'location': loc,
                        'construction_time': construction_time,
                        'earliest_addition': np.where(capacity_addition[tech, loc, :] > 1e-6)[0][0]
                    })

        return {
            'check_passed': valid_timing,
            'invalid_additions': invalid_additions
        }

    def check_constraint_satisfaction(self):
        """Verify that all construction time constraints are satisfied"""
        constraints = self.model.constraints['constraint_technology_construction_time']

        # Check if all constraints are satisfied within tolerance
        tolerance = 1e-6
        violations = []
        constraints_satisfied = True

        # Get constraint values
        constraint_values = constraints.values

        # Check for violations
        violation_indices = np.where(np.abs(constraint_values) > tolerance)
        if len(violation_indices[0]) > 0:
            constraints_satisfied = False
            for idx in zip(*violation_indices):
                violations.append({
                    'indices': idx,
                    'violation_value': constraint_values[idx]
                })

        return {
            'check_passed': constraints_satisfied,
            'tolerance_used': tolerance,
            'constraint_violations': violations
        }


def print_validation_report(validator_results):
    """Print a formatted validation report"""
    print("=== Location-Specific Construction Time Validation Report ===")
    print(f"Generated at: {datetime.now()}\n")

    for check_name, results in validator_results.iterrows():
        print(f"\n--- {check_name} ---")
        print(f"Check Passed: {results['check_passed']}")

        # Print additional details based on check type
        if check_name == 'input_data':
            print(f"Has Location Dimension: {results['has_location_dimension']}")
            print(f"Varies by Location: {results['varies_by_location']}")
            print(f"Unique Values: {results['unique_values']}")

        elif check_name == 'investment_timing' and not results['check_passed']:
            print("\nTiming Mismatches Found:")
            for mismatch in results['mismatches']:
                print(f"Tech {mismatch['tech']}, Location {mismatch['location']}:")
                print(f"Expected delay: {mismatch['expected_delay']}")
                print(f"Actual delay: {mismatch['actual_delay']}")

        elif check_name == 'capacity_addition' and not results['check_passed']:
            print("\nInvalid Additions Found:")
            for invalid in results['invalid_additions']:
                print(f"Tech {invalid['tech']}, Location {invalid['location']}:")
                print(f"Construction Time: {invalid['construction_time']}")
                print(f"Earliest Addition: {invalid['earliest_addition']}")

        elif check_name == 'constraint_satisfaction' and not results['check_passed']:
            print(f"\nConstraint Violations (tolerance: {results['tolerance_used']}):")
            for violation in results['constraint_violations']:
                print(f"Indices: {violation['indices']}")
                print(f"Violation Value: {violation['violation_value']}")


# Example usage
def validate_construction_time(optimization_setup):
    """Run validation checks and print report"""
    validator = ConstructionTimeValidator(optimization_setup)
    results = validator.run_all_checks()
    print_validation_report(results)
    return results