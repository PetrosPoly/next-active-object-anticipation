import multiprocessing as mp
import time

# Define a simple function that simulates a workload
def simulate_workload(duration):
    print(f"Task with duration {duration} seconds is starting...")
    time.sleep(duration)  # Simulate a task taking `duration` seconds
    print(f"Task with duration {duration} seconds is complete!")
    return f"Task with duration {duration} seconds completed"

# Main function to demonstrate parallel execution
if __name__ == "__main__":
    # List of durations to simulate different workloads
    durations = [2, 4, 3, 1]  # These are the "parameters" for the simulation

    # Get the number of available CPUs
    available_cpus = mp.cpu_count()
    print(f"Available CPUs: {available_cpus}")

    # Create a pool of workers, you can adjust the number of processes to use
    with mp.Pool(available_cpus - 1) as pool:  # Use N-1 CPUs
        # Run the simulate_workload function in parallel for each duration
        results = pool.map(simulate_workload, durations)

    # Print the results after all tasks have completed
    for result in results:
        print(result)
