import csv
import matplotlib.pyplot as plt

def safe_float(val, default=0.0, threshold=1e6):
    try:
        value = float(val) if val not in [None, ''] else 0.0
        return value if abs(value) < threshold else default
    except ValueError:
        return default

def plot_metrics(file_name='recorded metrics.csv'):
    time_steps = []
    requirements_1, requirements_2, requirements_3 = [], [], []
    throughputs_1, throughputs_2, throughputs_3 = [], [], []
    delays_1, delays_2, delays_3 = [], [], []

    # Open the CSV file and read its contents
    with open(file_name, mode='r') as file:
        reader = csv.DictReader(file)
        step = 0  # Track the time steps

        for row in reader:
            time_steps.append(step)

            requirements_1.append(safe_float(row['requirements_1'], default= 50.0, threshold=100.0))
            requirements_2.append(safe_float(row['requirements_2'], default= 50.0, threshold=100.0))
            requirements_3.append(safe_float(row['requirements_3'], default= 50.0, threshold=100.0))

            throughputs_1.append(safe_float(row['log_throughput_1'], default= 50.0, threshold=100.0))
            throughputs_2.append(safe_float(row['log_throughput_2'], default= 50.0, threshold=100.0))
            throughputs_3.append(safe_float(row['log_throughput_3'], default= 50.0, threshold=100.0))

            delays_1.append(safe_float(row['log_delay_1'], default=500.0, threshold=1000.0))
            delays_2.append(safe_float(row['log_delay_2'], default=500.0, threshold=1000.0))
            delays_3.append(safe_float(row['log_delay_3'], default=500.0, threshold=1000.0))

            step += 1

    # Set up a 3x3 grid of subplots
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle('Requirements, Throughput, and Delay over Time for Each Slice', fontsize=16)

    # Plot for Slice URLLC 1
    axs[0, 0].plot(time_steps, requirements_1, label='Requirements 1', color='blue')
    axs[0, 0].set_title('Slice URLLC 1 - Requirements')
    axs[0, 0].set_xlabel('Time Step')
    axs[0, 0].set_ylabel('Requirement')

    axs[1, 0].plot(time_steps, throughputs_1, label='Throughput 1', color='green')
    axs[1, 0].set_title('Slice URLLC 1 - Throughput')
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].set_ylabel('Throughput (Mbps)')

    axs[2, 0].plot(time_steps, delays_1, label='Delay 1', color='red')
    axs[2, 0].set_title('Slice URLLC 1 - Delay')
    axs[2, 0].set_xlabel('Time Step')
    axs[2, 0].set_ylabel('Delay (ms)')

    # Plot for Slice URLLC 2
    axs[0, 1].plot(time_steps, requirements_2, label='Requirements 2', color='blue')
    axs[0, 1].set_title('Slice URLLC 2 - Requirements')
    axs[0, 1].set_xlabel('Time Step')
    axs[0, 1].set_ylabel('Requirement')

    axs[1, 1].plot(time_steps, throughputs_2, label='Throughput 2', color='green')
    axs[1, 1].set_title('Slice URLLC 2 - Throughput')
    axs[1, 1].set_xlabel('Time Step')
    axs[1, 1].set_ylabel('Throughput (Mbps)')

    axs[2, 1].plot(time_steps, delays_2, label='Delay 2', color='red')
    axs[2, 1].set_title('Slice URLLC 2 - Delay')
    axs[2, 1].set_xlabel('Time Step')
    axs[2, 1].set_ylabel('Delay (ms)')

    # Plot for Slice eMBB 3
    axs[0, 2].plot(time_steps, requirements_3, label='Requirements 3', color='blue')
    axs[0, 2].set_title('Slice eMBB 3 - Requirements')
    axs[0, 2].set_xlabel('Time Step')
    axs[0, 2].set_ylabel('Requirement')

    axs[1, 2].plot(time_steps, throughputs_3, label='Throughput 3', color='green')
    axs[1, 2].set_title('Slice eMBB 3 - Throughput')
    axs[1, 2].set_xlabel('Time Step')
    axs[1, 2].set_ylabel('Throughput (Mbps)')

    axs[2, 2].plot(time_steps, delays_3, label='Delay 3', color='red')
    axs[2, 2].set_title('Slice eMBB 3 - Delay')
    axs[2, 2].set_xlabel('Time Step')
    axs[2, 2].set_ylabel('Delay (ms)')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for title
    plt.savefig('metrics_plot.png')
    plt.show()

# Example usage:
plot_metrics('/home/ucdavis/sliceWM/recorded metrics test.csv')
