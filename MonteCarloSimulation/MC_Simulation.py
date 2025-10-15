from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend('Agg')

def find_opex_files(root_dir):
    """
    Find base OPEX csv files (not simulation results) in the given directory.
    Only finds files named exactly 'opex_specific_variable' or 'opex_specific_fixed'
    """
    opex_files = []
    for path in Path(root_dir).rglob('*'):
        if path.is_file() and path.suffix.lower() == '.csv':
            filename = path.stem.lower()
            if filename in ['opex_specific_variable', 'opex_specific_fixed']:
                opex_files.append(str(path))
    return opex_files

def find_capex_files(root_dir):
    """
    Find base capex csv files (not simulation results) in the given directory.
    Only finds files named exactly 'opex_specific_variable' or 'opex_specific_fixed'
    """
    capex_files = []
    for path in Path(root_dir).rglob('*'):
        if path.is_file() and path.suffix.lower() == '.csv':
            filename = path.stem.lower()
            if filename == 'capex_specific_conversion':
                capex_files.append(str(path))
    return capex_files

def find_capacity_files(root_dir):
    """
    Find base capacity limit csv files (not simulation results) in the given directory.
    Only finds files named exactly 'capacity_limits'
    """
    capacity_files = []
    for path in Path(root_dir).rglob('*'):
        if path.is_file() and path.suffix.lower() == '.csv':
            filename = path.stem.lower()
            if filename == 'capacity_limit':
                capacity_files.append(str(path))
    return capacity_files


def process_opex_file(file_path, num_simulations):
    """
    Process OPEX files with row-wise random multipliers.
    """
    file_path = Path(file_path)
    output_dir = file_path.parent

    # Read file
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    simulation_data = []
    tech_name = output_dir.parent.parent.name

    for i in range(num_simulations):
        sim_df = df.copy()

        # Generate multipliers for each row (node)
        multipliers = np.random.uniform(0.5, 1.5, size=len(df))

        # Get the value column (assuming it's the second column)
        value_column = df.columns[1]

        # Apply multipliers to the values
        sim_df[value_column] = df[value_column] * multipliers

        # Store simulation results
        for idx, row in sim_df.iterrows():
            simulation_data.append({
                'Technology': tech_name,
                'Node': row.iloc[0],
                'Simulation': i + 1,
                'Value': row[value_column],
                'Type': 'Variable' if 'variable' in file_path.name.lower() else 'Fixed'
            })

        # Create output filename
        opex_type = 'variable' if 'variable' in file_path.name.lower() else 'fixed'
        output_path = output_dir / f"opex_specific_{opex_type}_MC{i + 1}{file_path.suffix}"

        # Save simulation result
        if file_path.suffix.lower() == '.csv':
            sim_df.to_csv(output_path, index=False)
        else:
            sim_df.to_excel(output_path, index=False)

    print(f"Created {num_simulations} OPEX simulations in {output_dir}")
    return pd.DataFrame(simulation_data)

def process_capex_file(file_path, num_simulations):
    """Process CAPEX files with row-wise random multipliers."""
    file_path = Path(file_path)
    output_dir = file_path.parent
    df = pd.read_csv(file_path) if file_path.suffix.lower() == '.csv' else pd.read_excel(file_path)

    simulation_data = []
    tech_name = output_dir.parent.parent.name

    for i in range(num_simulations):
        sim_df = df.copy()
        multipliers = np.random.uniform(0.5, 1.5, size=len(df))
        value_column = df.columns[1]
        sim_df[value_column] = df[value_column] * multipliers

        for idx, row in sim_df.iterrows():
            simulation_data.append({
                'Technology': tech_name,
                'Node': row.iloc[0],
                'Simulation': i + 1,
                'Value': row[value_column],
                'Type': 'Capex'
            })

        output_path = output_dir / f"capex_specific_conversion_MC{i + 1}{file_path.suffix}"
        if file_path.suffix.lower() == '.csv':
            sim_df.to_csv(output_path, index=False)
        else:
            sim_df.to_excel(output_path, index=False)

    print(f"Created {num_simulations} CAPEX simulations in {output_dir}")
    return pd.DataFrame(simulation_data)

def process_capacity_file(file_path, num_simulations):
    """
    Process capacity files with debugging focus on AUS column.
    """
    file_path = Path(file_path)
    output_dir = file_path.parent

    # Read file
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # First column is year, rest are countries
    year_col = df.columns[0]
    country_cols = df.columns[1:]  # All country columns

    for i in range(num_simulations):
        sim_df = df.copy()

        for col in country_cols[0:]:
            multiplier = np.random.uniform(0.5, 1.5)
            sim_df[col] = df[col] * multiplier

        # Create output filename
        output_path = output_dir / f"capacity_limit_MC{i + 1}{file_path.suffix}"

        # Save simulation result
        if file_path.suffix.lower() == '.csv':
            sim_df.to_csv(output_path, index=False)
        else:
            sim_df.to_excel(output_path, index=False)

    print(f"\nCreated {num_simulations} capacity simulations in {output_dir}")
    return sim_df

def process_all_technologies(root_dir, script_dir):
    """
    Process all OPEX and capacity files
    """
    # Find all relevant files
    opex_files = find_opex_files(root_dir)
    capex_files = find_capex_files(root_dir)
    capacity_files = find_capacity_files(root_dir)

    if not opex_files and not capacity_files and not capex_files:
        print("No files found to process!")
        return

    num_simulations = 1000

    # Process OPEX files
    opex_results = []
    for file_path in opex_files:
        print(f"Processing OPEX file: {file_path}")
        simulation_df = process_opex_file(file_path, num_simulations)
        opex_results.append(simulation_df)

    # Process capex files
    capex_results = []
    for file_path in capex_files:
        print(f"Processing capacity file: {file_path}")
        simulation_df = process_capex_file(file_path, num_simulations)
        capex_results.append(simulation_df)

    # Process capacity files
    capacity_results = []
    for file_path in capacity_files:
        print(f"Processing capacity file: {file_path}")
        simulation_df = process_capacity_file(file_path, num_simulations)
        capacity_results.append(simulation_df)

    # Combine results if needed
    if opex_results:
        combined_opex = pd.concat(opex_results, ignore_index=True)

    if capacity_results:
        combined_capacity = pd.concat(capacity_results, ignore_index=True)


def test_mode(root_dir):
    """
    Test mode to verify file structure and readability
    """
    print("\n=== Starting Test Mode ===")

    script_dir = Path(__file__).parent.absolute()
    print(f"Script directory: {script_dir}")

    opex_files = find_opex_files(root_dir)
    capacity_files = find_capacity_files(root_dir)

    all_files = opex_files + capacity_files

    if not all_files:
        print("❌ No files found!")
        return False

    print(f"\nFound {len(all_files)} files:")
    for file in all_files:
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            print(f"✓ {file} (Rows: {len(df)})")
        except Exception as e:
            print(f"❌ Error reading {file}: {str(e)}")
            return False

    print("\n=== All Tests Passed ===\n")
    return True


if __name__ == "__main__":
    script_dir = Path(__file__).parent.absolute()
    root_directory = "./ZEN-Model_HP"

    if test_mode(root_directory):
        print("Proceeding with processing...")
        process_all_technologies(root_directory, script_dir)
    else:
        print("Tests failed. Please check the errors above before proceeding.")