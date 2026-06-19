from itertools import product

# Define the parameter ranges
time_thresholds = [1, 2, 3, 4, 5]
avg_dot_threshold_highs = [0.7]
avg_dot_threshold_lows = [0.2]
avg_distance_threshold_highs = [3]
avg_distance_threshold_lows = [1]
high_dot_thresholds = [0.9]
distance_thresholds = [2]
low_distance_thresholds = [1]
variables_window_times = [3.0]

# Define the parameter ranges
time_thresholds = [1, 2, 3, 4, 5]
avg_dot_threshold_highs = [0.7]
avg_dot_threshold_lows = [0.2]
avg_distance_threshold_highs = [3]
avg_distance_threshold_lows = [1]
high_dot_thresholds = [0.9]
distance_thresholds = [2]
low_distance_thresholds = [1]
variables_window_times = [3.0]


# Generate all combinations of the parameters
param_combinations = [
    {
        "time_threshold": t,
        "avg_dot_high": adh,
        "avg_dot_low": adl,
        "avg_distance_high": adhg,
        "avg_distance_low": adlg,
        "high_dot_threshold": hdt,
        "distance_threshold": dt,
        "low_distance_threshold": ldt,
        "window_time": w
    }
    for t, adh, adl, adhg, adlg, hdt, dt, ldt, w in product(             
        time_thresholds, avg_dot_threshold_highs, avg_dot_threshold_lows, 
        avg_distance_threshold_highs, avg_distance_threshold_lows,
        high_dot_thresholds, distance_thresholds, low_distance_thresholds, variables_window_times
    )
]

"""

Explanation of product

Given multiple input lists (or any iterables), product produces tuples that represent every possible combination of elements

Sample code: 
        import itertools

        list1 = [1, 2]
        list2 = ['a', 'b']

        combinations = list(itertools.product(list1, list2))
        print(combinations)

Output: 
     [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]

"""


# Example usage
for params in param_combinations:
    print(params)
    # Here you would pass `params` to your main logic
    # main_logic(params)
