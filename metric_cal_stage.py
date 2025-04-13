import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

def calculate_metrics_by_stage(file_name='recorded_metrics.csv', loop_size=4000, num_stages=4):
    # Containers for metrics grouped by stage
    stage_metrics = {i: {'latency_1': [], 'latency_2': [], 'latency_3': [],
                         'throughput_1': [], 'throughput_2': [], 'throughput_3': []} for i in range(num_stages)}
    
    with open(file_name, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            step = int(row['step'])
            stage = (step - 1) // (loop_size // num_stages)  # Determine stage index
            
            # Helper function to safely convert to float
            def safe_float(val, default=0.0):
                try:
                    return float(val) if val not in [None, ''] else default
                except ValueError:
                    return default
            
            # Store values in corresponding stage
            stage_metrics[stage]['latency_1'].append(safe_float(row['latency_1']))
            stage_metrics[stage]['latency_2'].append(safe_float(row['latency_2']))
            stage_metrics[stage]['throughput_3'].append(safe_float(row['throughput_3']))
    
    # Calculate and print averages per stage
    for stage in range(num_stages):
        print(f"Stage {stage + 1}:")
        print(f"  Average Latency 1: {np.mean(stage_metrics[stage]['latency_1']):.2f} ms")
        print(f"  Average Latency 2: {np.mean(stage_metrics[stage]['latency_2']):.2f} ms")
        print(f"  Average Throughput 3: {np.mean(stage_metrics[stage]['throughput_3']):.2f} packets")
        print()

def plot_metrics_by_step(file_name='recorded_metrics.csv', smoothing_sigma=15):
    step_metrics = defaultdict(lambda: {'latency_1': [], 'latency_2': [], 'latency_3': [],
                                        'throughput_1': [], 'throughput_2': [], 'throughput_3': []})
    
    with open(file_name, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            step = int(row['step'])
            
            def safe_float(val, default=0.0):
                try:
                    return float(val) if val not in [None, ''] else default
                except ValueError:
                    return default
            
            step_metrics[step]['latency_1'].append(safe_float(row['latency_1']))
            step_metrics[step]['latency_2'].append(safe_float(row['latency_2']))
            step_metrics[step]['latency_3'].append(safe_float(row['latency_3']))
            step_metrics[step]['throughput_1'].append(safe_float(row['throughput_1']))
            step_metrics[step]['throughput_2'].append(safe_float(row['throughput_2']))
            step_metrics[step]['throughput_3'].append(safe_float(row['throughput_3']))
    
    steps = sorted(step_metrics.keys())
    metrics = {key: np.array([np.mean(step_metrics[step][key]) for step in steps]) for key in step_metrics[steps[0]].keys()}
    
    # Smooth metrics
    for key in metrics:
        metrics[key] = gaussian_filter1d(metrics[key], sigma=smoothing_sigma)

    # Define four stage boundaries
    total_steps = len(steps)
    stage_bounds = [steps[0], steps[total_steps//4], steps[total_steps//2], steps[3*total_steps//4], steps[-1]]
    
    # Define colors for each stage
    stage_colors = ['red', 'blue', 'green', 'purple']  # Different colors for each stage

    # Plot with subplots and different colors for each metric
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))  # Increase figure size
    labels = {"latency_1": "Latency 1 (ms)", "latency_2": "Latency 2 (ms)", "throughput_3": "Throughput (packets)"}

    for i, (key, ax) in enumerate(zip(['latency_1', 'latency_2', 'throughput_3'], axs)):
        for j in range(4):  # Iterate over 4 stages
            start_idx = np.searchsorted(steps, stage_bounds[j])
            end_idx = np.searchsorted(steps, stage_bounds[j+1])

            stage_steps = steps[start_idx:end_idx]
            stage_values = metrics[key][start_idx:end_idx]

            ax.fill_between(stage_steps, stage_values - np.std(stage_values),
                            stage_values + np.std(stage_values), color=stage_colors[j], alpha=0.1)
            ax.plot(stage_steps, stage_values, color=stage_colors[j], linewidth=2, label=f"{labels[key]} (Stage {j+1})")
        
        ax.set_title(f"{labels[key]} Over Steps")
        ax.set_xlabel("Step")
        ax.set_ylabel(labels[key])
        ax.legend()

    plt.tight_layout()
    plt.savefig('stage_metrics_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

# Call the functions
calculate_metrics_by_stage('recorded_metrics.csv')
plot_metrics_by_step('recorded_metrics.csv')
