import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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

file_data = "./HP_Model/parameter_results/cost_opex_yearly/cost_opex_yearly_scenarios.csv"
# Read data
data = pd.read_csv(file_data)
# Filter out storage and transport technologies
filtered_data = data[~data['technology'].str.contains('storage|transport', case=False, na=False)]

# Sum over locations for each technology and timestep
summed_data = filtered_data.groupby(['year', 'technology'])['value_scenario_'].sum().reset_index()

# Pivot data for plotting
plot_data = summed_data.pivot(index='year',
                             columns='technology',
                             values='value_scenario_')

# Create stacked bar plot
ax = plot_data.plot(kind='bar', stacked=True, figsize=(12, 6))

# Calculate and display total for each timestep
totals = plot_data.sum(axis=1)
print("\nTotal OPEX per timestep:")
for time, total in totals.items():
    print(f"Timestep {time}: {total:.2f}")
# Create plot
plot_data.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('OPEX by Technology Over Time')
plt.xlabel('Time Step')
plt.ylabel('OPEX')
plt.legend(title='Technology', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('opex_plot.png')