import csv

def calculate_metrics(file_name='recorded metrics.csv'):
    delays_1 = []
    delays_2 = []
    delays_3 = []
    throughputs_1 = []
    throughputs_2 = []
    throughputs_3 = []
    violations_1 = []
    violations_2 = []
    violations_3 = []
    
    total_steps = 0
    total_violations_1 = 0
    total_violations_2 = 0
    total_violations_3 = 0

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

            # Collect delay values (in ms)
            delays_1.append(safe_float(row['log_delay_1']))
            delays_2.append(safe_float(row['log_delay_2']))
            delays_3.append(safe_float(row['log_delay_3']))

            # Collect throughput values (in Mbps)
            throughputs_1.append(safe_float(row['log_throughput_1']))
            throughputs_2.append(safe_float(row['log_throughput_2']))
            throughputs_3.append(safe_float(row['log_throughput_3']))

            # Collect violation values (boolean if > 0.0)
            violations_1.append(safe_float(row['log_violation_1']))
            violations_2.append(safe_float(row['log_violation_2']))
            violations_3.append(safe_float(row['log_violation_3']))

            # Check for violations (count steps with violation > 0.0)
            if safe_float(row['log_violation_1']) > 0.0:
                total_violations_1 += 1
            if safe_float(row['log_violation_2']) > 0.0:
                total_violations_2 += 1
            if safe_float(row['log_violation_3']) > 0.0:
                total_violations_3 += 1

    # Calculate average delay for each slice
    avg_delay_1 = sum(delays_1) / len(delays_1) if delays_1 else 0.0
    avg_delay_2 = sum(delays_2) / len(delays_2) if delays_2 else 0.0
    avg_delay_3 = sum(delays_3) / len(delays_3) if delays_3 else 0.0

    # Calculate average throughput for each slice
    avg_throughput_1 = sum(throughputs_1) / len(throughputs_1) if throughputs_1 else 0.0
    avg_throughput_2 = sum(throughputs_2) / len(throughputs_2) if throughputs_2 else 0.0
    avg_throughput_3 = sum(throughputs_3) / len(throughputs_3) if throughputs_3 else 0.0

    # Calculate violation rate for each slice
    violation_rate_1 = total_violations_1 / total_steps if total_steps else 0.0
    violation_rate_2 = total_violations_2 / total_steps if total_steps else 0.0
    violation_rate_3 = total_violations_3 / total_steps if total_steps else 0.0

    # Print out the calculated metrics with grouping for each slice
    print(f"Slice URLLC 1:")
    print(f"  Average Delay: {avg_delay_1:.2f} ms")
    print(f"  Average Throughput: {avg_throughput_1:.2f} Mbps")
    # print(f"  Violation Rate: {violation_rate_1:.2%}")
    
    print(f"Slice URLLC 2:")
    print(f"  Average Delay: {avg_delay_2:.2f} ms")
    print(f"  Average Throughput: {avg_throughput_2:.2f} Mbps")
    # print(f"  Violation Rate: {violation_rate_2:.2%}")
    
    print(f"Slice eMBB 3:")
    print(f"  Average Delay: {avg_delay_3:.2f} ms")
    print(f"  Average Throughput: {avg_throughput_3:.2f} Mbps")
    # print(f"  Violation Rate: {violation_rate_3:.2%}")


# Example usage
calculate_metrics('recorded metrics test.csv')
