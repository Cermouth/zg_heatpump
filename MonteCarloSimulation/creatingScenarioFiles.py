import os
import json
from pathlib import Path


def find_mc_files(base_path):
    opex_files = {}
    capacity_files = {}
    capex_files = {}
    tech_path = base_path / "set_technologies/set_conversion_technologies"

    # Process OPEX files
    for file in tech_path.glob("**/opex_specific_*MC*"):
        try:
            mc_num = ''.join(filter(str.isdigit, file.stem.split('MC')[1].split('_')[0]))
            file_type = "variable" if "variable" in file.name else "fixed"

            if mc_num not in opex_files:
                opex_files[mc_num] = {
                    "set_conversion_technologies": {}
                }

            opex_files[mc_num]["set_conversion_technologies"][f"opex_specific_{file_type}"] = {
                "file": file.stem
            }
        except Exception as e:
            print(f"Error processing OPEX file {file}: {e}")

    # Process capacity limit files
    for file in tech_path.glob("**/capacity_limit_*MC*"):
        try:
            mc_num = ''.join(filter(str.isdigit, file.stem.split('MC')[1].split('_')[0]))

            if mc_num not in capacity_files:
                capacity_files[mc_num] = {
                    "set_conversion_technologies": {}
                }

            capacity_files[mc_num]["set_conversion_technologies"]["capacity_limit"] = {
                "file": file.stem
            }
        except Exception as e:
            print(f"Error processing capacity limit file {file}: {e}")

    # Process capex files
    for file in tech_path.glob("**/capex_specific_conversion_*MC*"):
        try:
            mc_num = ''.join(filter(str.isdigit, file.stem.split('MC')[1].split('_')[0]))

            if mc_num not in capex_files:
                capex_files[mc_num] = {
                    "set_conversion_technologies": {}
                }

            capex_files[mc_num]["set_conversion_technologies"]["capex_specific_conversion"] = {
                "file": file.stem
            }
        except Exception as e:
            print(f"Error processing capex_specific_conversion file {file}: {e}")

    return opex_files, capacity_files, capex_files


def generate_scenario_json(mc_files, scenario_type):
    """
    Convert MC files to JSON format with scenario_type prefix
    scenario_type should be either 'opex' or 'capacity'
    """
    sorted_mc = sorted(mc_files.items(), key=lambda x: int(x[0]))
    return {f"MC{mc_num}_{scenario_type}": data for mc_num, data in sorted_mc}


def combine_uncertainties(opex_files, capacity_files, capex_files):
    """
    Combine OPEX and capacity uncertainties into a single dictionary
    """
    combined = {}
    all_mc_nums = set(opex_files.keys()) | set(capacity_files.keys())

    for mc_num in all_mc_nums:
        combined[mc_num] = {"set_conversion_technologies": {}}

        # Add OPEX data if available
        if mc_num in opex_files:
            opex_data = opex_files[mc_num]["set_conversion_technologies"]
            combined[mc_num]["set_conversion_technologies"].update(opex_data)

        # Add capacity data if available
        if mc_num in capacity_files:
            capacity_data = capacity_files[mc_num]["set_conversion_technologies"]
            combined[mc_num]["set_conversion_technologies"].update(capacity_data)

        # Add OPEX data if available
        if mc_num in capex_files:
            capex_data = capex_files[mc_num]["set_conversion_technologies"]
            combined[mc_num]["set_conversion_technologies"].update(capex_data)

    return combined

def save_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    base_path = Path(
        "./ZEN-Model_HP")

    # Define output paths for both files
    output_path_opex = base_path / "uncertainty_opex.json"
    output_path_capacity = base_path / "uncertainty_capacity.json"
    output_path_capex = base_path / "uncertainty_capex.json"
    output_path_combined = base_path / "uncertainty_combined.json"

    # Get both types of files
    opex_files, capacity_files, capex_files = find_mc_files(base_path)

    # Process and save OPEX scenarios
    if opex_files:
        scenarios_opex = generate_scenario_json(opex_files, "opex")
        save_json(scenarios_opex, output_path_opex)
        print(f"Successfully updated OPEX scenarios file at {output_path_opex}")
    else:
        print("No OPEX MC files found!")

    # Process and save Capacity scenarios
    if capacity_files:
        scenarios_capacity = generate_scenario_json(capacity_files, "capacity")
        save_json(scenarios_capacity, output_path_capacity)
        print(f"Successfully updated Capacity scenarios file at {output_path_capacity}")
    else:
        print("No Capacity MC files found!")

    # Process and save Capacity scenarios
    if capex_files:
        scenarios_capex = generate_scenario_json(capex_files, "capex")
        save_json(scenarios_capex, output_path_capex)
        print(f"Successfully updated capex scenarios file at {output_path_capex}")
    else:
        print("No capex MC files found!")

        # Process and save Combined scenarios
    if opex_files or capacity_files or capex_files:
        combined_files = combine_uncertainties(opex_files, capacity_files,capex_files)
        scenarios_combined = generate_scenario_json(combined_files, "combined")
        save_json(scenarios_combined, output_path_combined)
        print(f"Successfully updated Combined scenarios file at {output_path_combined}")
    else:
        print("No files found for combined uncertainty!")


if __name__ == "__main__":
    main()