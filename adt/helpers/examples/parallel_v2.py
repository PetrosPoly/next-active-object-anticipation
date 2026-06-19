import multiprocessing as mp

# ==============================================
# Parameters Setting
# ==============================================
param_combinations = [
    {'param1': 1, 'param2': 2}, 
    {'param1': 3, 'param2': 4},
    {'param1': 5, 'param2': 6}
]

# ==============================================
# Define your main function
# ==============================================
def run_simulation(parameters):
    # You need to use the 'parameters' variable here
    # For example, accessing param1 and param2 from the 'parameters' dictionary
    param1 = parameters['param1']
    param2 = parameters['param2']

    print(f"Running simulation with param1={param1}, param2={param2}")

    # Simulate some work or logic with these parameters
    result = param1 + param2  # Just an example of computation
    print(f"Result of simulation: {result}")

# ==============================================
# Run for different parameter combinations in parallel
# ==============================================
if __name__ == "__main__":
    with mp.Pool(mp.cpu_count() - 1) as pool:  # Using N-1 CPUs
        pool.map(run_simulation, param_combinations)
