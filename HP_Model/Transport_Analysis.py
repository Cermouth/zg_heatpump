import pandas as pd
from pathlib import Path

def process_trade_data(trans_path, technology_filter):

    df = pd.read_csv(trans_path)
    mask = df['technology'] == technology_filter

    # Create new DataFrame instead of modifying a slice
    df_filtered = pd.DataFrame({
        'technology': df.loc[mask, 'technology'],
        'time_operation': df.loc[mask, 'time_operation'],
        'value_scenario_': df.loc[mask, 'value_scenario_']
    })

    if df_filtered.empty:
        print(f"No data found for technology: {technology_filter}")
        return pd.DataFrame()

    # Split edge into exporter and importer
    df_filtered[['exporter', 'importer']] = df.loc[mask, 'edge'].str.split('-', expand=True)

    # Get unique countries and timesteps
    countries = sorted(set(df_filtered['exporter'].unique()) | set(df_filtered['importer'].unique()))
    timesteps = sorted(df_filtered['time_operation'].unique())

    # Create results DataFrame
    results = []
    for time in timesteps:
        df_time = df_filtered[df_filtered['time_operation'] == time]
        row = {'technology': technology_filter, 'timestep': time}

        for country in countries:
            # Calculate exports
            exports = df_time[df_time['exporter'] == country]['value_scenario_'].astype(float).sum()
            row[f'{country}_exports'] = exports

            # Calculate imports
            imports = df_time[df_time['importer'] == country]['value_scenario_'].astype(float).sum()
            row[f'{country}_imports'] = imports

        results.append(row)

    # Save results
    results = pd.DataFrame(results)
    output_dir = Path("./HP_Model/Transport")
    output_path = output_dir / f'transport_stats_{technology_filter}.csv'
    results.to_csv(output_path, index=False)
    print(f"Total cost temporal analysis saved to: {output_path}")

    return pd.DataFrame(results)


def process_excel_trade_data(excel_path, target_countries, sheetname, years=[2022, 2023]):
    hs_codes = ['841430', '841440', '841581', '841861', '841950']
    hs_dict = {
        '841430': '841430_compressors_AC',
        '841440': '841440_compressors_air',
        '841581': '841581_AC',
        '841861': '841861_heatPumpsotherthanAC_machines',
        '841950': '841950_industrialHP'
    }
    # Read Excel file
    df = pd.read_excel(excel_path,sheet_name = sheetname)

    # Convert values to numeric, replacing non-numeric with NaN
    df['primaryValue'] = pd.to_numeric(df['primaryValue'], errors='coerce')
    df['year'] = pd.to_numeric(df['refYear'], errors='coerce')

    # Filter for specified years and drop NaN values
    df = df[df['year'].isin(years)].dropna(subset=['primaryValue'])

    for hs_code, file_name in hs_dict.items():
        hs_mask = df['cmdCode'].astype(str) == str(hs_code)
        df_hs = df[hs_mask].dropna(subset=['primaryValue'])

        results = []
        for year in years:
            year_data = df_hs[df_hs['year'] == year]
            row = {'timestep': year, 'hs_code': hs_code}

            for country in target_countries:
                exports_mask = year_data['reporterDesc'] == country
                exports = year_data[exports_mask]['primaryValue'].sum()

                imports_mask = year_data['partnerDesc'] == country
                imports = year_data[imports_mask]['primaryValue'].sum()

                row[f'{country}_exports'] = exports
                row[f'{country}_imports'] = imports

            results.append(row)

        results_df = pd.DataFrame(results)
        output_dir = Path(
            "./HP_Model/Transport")
        output_path = output_dir / f'comtrade_stats_{file_name}.csv'
        results_df.to_csv(output_path, index=False)
        print(f"Analysis saved: {output_path}")



# Target countries
countries = [
    'Australia', 'Austria', 'Brazil', 'China', "Czechia",
    'Germany', 'Italy', 'Japan',  'Rep. of Korea','USA'
]
# Comtrade data
excel_path = "./HP_Model/Transport/TradeData_Comtrade.xlsx"
trade_df = process_excel_trade_data(excel_path, countries, "Sheet1")

#Model Data
data = "./HP_Model/parameter_results/flow_transport/flow_transport_scenarios.csv"

# Process for both technologies
compressor_df = process_trade_data(data, "Compressor_transport")
hp_df = process_trade_data(data, "HP_transport")


