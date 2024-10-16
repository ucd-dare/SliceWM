import os
import pandas as pd
import matplotlib.pyplot as plt

# Folder containing the CSV files
folder_path = 'process_csv_0'
output_path = '/home/ucdavis/sliceWM/dynamic_process.png'

# Get the list of CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
if 'World model.csv' in csv_files:
    csv_files.remove('World model.csv')
    csv_files.insert(0, 'World model.csv')

# Create the figure and axes for the 2x3 subplot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Define a smoothing function (rolling average)
def smooth_data(series, window=20):
    return series.rolling(window=window, min_periods=1).mean()

# Loop through each CSV file
for idx, csv_file in enumerate(csv_files):
    # Read the CSV file with error handling (skip bad lines)
    file_path = os.path.join(folder_path, csv_file)
    
    try:
        # Try to load the file, skipping problematic rows
        data = pd.read_csv(file_path, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"Error reading {csv_file}: {e}")
        continue  # Skip this file if there's an error
    
    label_name = os.path.splitext(csv_file)[0]  # Removes the .csv suffix
    
    # Check if the file is 'World model.csv' to make it red
    if csv_file == 'World model.csv':
        color = 'red' 
    elif  csv_file == 'Equal allocation.csv':
        color = 'blue'
    elif  csv_file == 'PPO.csv':
        color = 'green'
    else:
        color = None
    width = 1.5
    
    # Plot the first row: requirements_1, requirements_2, requirements_3
    if idx == 0:
        axes[0, 0].plot(data['requirements_1'], linewidth=width)
        axes[0, 1].plot(data['requirements_2'], linewidth=width)
        axes[0, 2].plot(data['requirements_3'], linewidth=width)
    
     # Plot the second row: smoothed log_delay_1, log_delay_2, log_throughput_3
    axes[1, 0].plot(smooth_data(data['log_delay_1']), label=label_name, color=color, linewidth=width)
    axes[1, 1].plot(smooth_data(data['log_delay_2']), label=label_name, color=color, linewidth=width)
    axes[1, 2].plot(smooth_data(data['log_throughput_3']), label=label_name, color=color, linewidth=width)

# Set titles for each subplot
axes[0, 0].set_title('Bandwidth requirements in slice 1 (URLLC)')
axes[0, 1].set_title('Bandwidth requirements in slice 2 (URLLC)')
axes[0, 2].set_title('Bandwidth requirements slice 3 (eMBB)')

axes[1, 0].set_title('Delay in slice 1 (URLLC)')
axes[1, 1].set_title('Delay in slice 2 (URLLC)')
axes[1, 2].set_title('Throughput in slice 3 (eMBB)')

# Set y labels for the first row to 'Bandwidth (Mbps)'
for i in range(3):
    axes[0, i].set_ylabel('Bandwidth (Mbps)')

# Set y labels for (1,0) and (1,1) to 'Delay (ms)'
axes[1, 0].set_ylabel('Delay (ms)')
axes[1, 1].set_ylabel('Delay (ms)')

# Set y label for (1,2) to 'Throughput (Mbps)'
axes[1, 2].set_ylabel('Throughput (Mbps)')

# Set x labels for all subplots to 'Step'
for i in range(3):
    axes[0, i].set_xlabel('')
    axes[1, i].set_xlabel('Step')

# Add legends
for ax in axes.flat:
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[1, 0].legend(loc='upper right')


# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure as a PNG
plt.savefig(output_path, dpi=300)

# Show the plot (optional, can be commented out if running on a non-GUI server)
plt.show()
