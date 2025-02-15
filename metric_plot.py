import csv

def calculate_metrics(file_name='recorded_metrics.csv'):
    latencies_1 = []
    latencies_2 = []
    latencies_3 = []
    throughputs_1 = []
    throughputs_2 = []
    throughputs_3 = []
    
    total_steps = 0

    # Open the CSV file and read its contents
    with open(file_name, mode='r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            total_steps += 1

            # Helper function to convert values to float and handle None or empty values
            def safe_float(val, default=0.0):
                try:
                    return float(val) if val not in [None, ''] else default
                except ValueError:
                    return default

            # Collect latency values (in ms)
            latencies_1.append(safe_float(row['latency_1']))
            latencies_2.append(safe_float(row['latency_2']))
            latencies_3.append(safe_float(row['latency_3']))

            # Collect throughput values (in Mbps)
            throughputs_1.append(safe_float(row['throughput_1']))
            throughputs_2.append(safe_float(row['throughput_2']))
            throughputs_3.append(safe_float(row['throughput_3']))

    # Calculate average latency for each slice
    avg_latency_1 = sum(latencies_1) / len(latencies_1) if latencies_1 else 0.0
    avg_latency_2 = sum(latencies_2) / len(latencies_2) if latencies_2 else 0.0
    avg_latency_3 = sum(latencies_3) / len(latencies_3) if latencies_3 else 0.0

    # Calculate average throughput for each slice
    avg_throughput_1 = sum(throughputs_1) / len(throughputs_1) if throughputs_1 else 0.0
    avg_throughput_2 = sum(throughputs_2) / len(throughputs_2) if throughputs_2 else 0.0
    avg_throughput_3 = sum(throughputs_3) / len(throughputs_3) if throughputs_3 else 0.0

    # Print out the calculated metrics with grouping for each slice
    print(f"Slice URLLC 1:")
    print(f"  Average Latency: {avg_latency_1:.2f} ms")
    print(f"  Average Throughput: {avg_throughput_1:.2f} Mbps")
    
    print(f"Slice URLLC 2:")
    print(f"  Average Latency: {avg_latency_2:.2f} ms")
    print(f"  Average Throughput: {avg_throughput_2:.2f} Mbps")
    
    print(f"Slice eMBB 3:")
    print(f"  Average Latency: {avg_latency_3:.2f} ms")
    print(f"  Average Throughput: {avg_throughput_3:.2f} Mbps")

# Example usage
calculate_metrics('recorded_metrics.csv')