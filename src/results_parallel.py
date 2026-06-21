import os
import json
import time
import multiprocessing as mp
from itertools import product
from utils.evaluation import LLMEvaluation
import pandas as pd
    
# Project path
project_path = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
sequences = ['Apartment_release_clean_seq150_M1292'] #, 'Apartment_release_work_seq107_M1292']

# Parameters for the language model module
# time_thresholds = [2]                     
# avg_dot_threshold_highs = [0.7]                         
# avg_dot_threshold_lows = [0.2]                          
# avg_distance_threshold_highs = [3]                       
# avg_distance_threshold_lows = [1]                       
# high_dot_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]                 
# distance_thresholds = [1.5, 2, 2.5]                        
# high_dot_counters_threshold = [15, 30, 45, 60, 75, 90]  
# distance_counters_threshold = [15, 30, 45, 60, 75, 90]  
# variables_window_times = [3.0]                          
# minimum_time_deactivated = [2.0]                        
# maximum_time_deactivated = [5.0]                        
# user_relative_movement = [2.0]                          
# object_percentage_overlap = [0.7]                                            

# Parameters for the language model module
time_thresholds = [2] 
avg_dot_threshold_highs = [0.7]
avg_dot_threshold_lows = [0.2]
avg_distance_threshold_highs = [3]
avg_distance_threshold_lows = [1]
high_dot_thresholds = [0.7]
distance_thresholds = [1.5]
high_dot_counters_threshold = [15]
distance_counters_threshold = [15]
variables_window_times = [3.0]
minimum_time_deactivated = [2.0]
maximum_time_deactivated = [5.0]
user_relative_movement = [1.5]
object_percentage_overlap = [0.7]

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
        "high_dot_counters_threshold": hdct,
        "distance_counters_threshold": dct,
        "window_time": w, 
        "minimum_time_deactivated": mintd,                
        "maximum_time_deactivated": maxtd,             
        "user_relative_movement": urm,                 
        "object_percentage_overlap" : obo
    }
    for t, adh, adl, adhg, adlg, hdt, dt, hdct, dct, w, mintd, maxtd, urm, obo in product(
        time_thresholds, avg_dot_threshold_highs, avg_dot_threshold_lows, 
        avg_distance_threshold_highs, avg_distance_threshold_lows,
        high_dot_thresholds, distance_thresholds,
        high_dot_counters_threshold, distance_counters_threshold, variables_window_times, 
        minimum_time_deactivated, maximum_time_deactivated, user_relative_movement, object_percentage_overlap   
    )
]

# Custom writing function to match the desired output format
def write_custom_json(data, file_path):
    
    with open(file_path, 'w') as f:
        
        # Start the JSON array
        f.write('[\n')
        
        for i, item in enumerate(data):
            
            # Dump each item and remove the surrounding list brackets
            json_item = json.dumps(item)
            
            # Write each item followed by a comma, except the last item
            if i < len(data) - 1:
                f.write(f'    {json_item},\n')
            else:
                f.write(f'    {json_item}\n')
        
        # End the JSON array
        f.write(']\n')

# Function to filter ground truth data
def filter_ground_truth(sequence):
    # Load ground truth data
    with open(os.path.join(project_path, 'data', 'gt', sequence, 'objects_that_moved.json'), 'r') as json_file:
        original_objects_that_moved_dict = json.load(json_file)

    with open(os.path.join(project_path, 'data', 'gt', sequence, 'user_object_movement.json'), 'r') as json_file:
        user_object_movement = json.load(json_file)

    # Filter the ground truth data based on the user's motion
    filtered_objects = {k: float(v) for k, v in user_object_movement.items() if v > 1.0}
    ground_truth = {float(k): v for k, v in original_objects_that_moved_dict.items() if v in filtered_objects}
    
    # Save filtered ground truth (optional, if needed later)
    ground_truth_folder = os.path.join(project_path, 'data', 'gt', sequence)
    os.makedirs(ground_truth_folder, exist_ok=True)
    with open(os.path.join(ground_truth_folder, 'filtered_ground_truth.json'), 'w') as file:
        json.dump(ground_truth, file, indent=4)

    return ground_truth

def run_simulation(parameters, ground_truth, sequence):  
    results = []

    # Parameters folder name
    parameter_folder_name = (
        f"time_{parameters['time_threshold']}_"
        f"highdot_{parameters['high_dot_threshold']}_"
        f"highdotcount_{parameters['high_dot_counters_threshold']}_"
        f"dist_{parameters['distance_threshold']}_"
        f"distcount_{parameters['distance_counters_threshold']}"
    )

    # Load predictions
    with open(os.path.join(project_path, 'data', 'predictions', sequence, parameter_folder_name, 'large_language_model_prediction.json')) as json_file:
        predictions_dict = json.load(json_file)

    # LLM predictions and evaluation logic
    LLM_predictions = {float(k): v for k, v in predictions_dict.items()}

    evaluation = LLMEvaluation(LLM_predictions, ground_truth)
    evaluation.calculate_FP_FN_GT_correspondances()
    evaluation.calculate_final_TP_FP()
    metrics = evaluation.calculate_metrics()

    # Store results
    result = {
        'sequence': sequence,
        'time':parameters['time_threshold'],
        'dot_value': parameters['high_dot_threshold'],
        'distance_value': parameters['distance_threshold'],
        'dot_counts': parameters['high_dot_counters_threshold'], 
        'distance_counts': parameters['distance_counters_threshold'],
        'model_overall_accuracy': metrics[0],
        'precision': metrics[1],
        'recall': metrics[2],
        'llm_activation_sensitivity': metrics[3],
        'llm_interaction_accuracy': metrics[4],
        'Tp': metrics[5],
        'Fp': metrics[6],
        'Fp_out': metrics[7],
        'Fp_in': metrics[8],
        'Fn': metrics[9],
        'Total_ground_truths': metrics[10],
        'Total_llm_predictions': metrics[11], 
        'Total_llm_activations': metrics[12],
        'Total_correspondances': metrics[13]
    }
    results.append(result)
    
    # Define the folders that we will write the results       
    result_folder = os.path.join(project_path, 'data', 'results', sequence, parameter_folder_name)
    os.makedirs(result_folder, exist_ok=True)  

    # Write the correspondances
    write_custom_json(metrics[14], os.path.join(result_folder, 'correspondances.json'))
    
    return results

if __name__ == "__main__":
    
    # Filter ground truth once per sequence
    filtered_ground_truths = {sequence: filter_ground_truth(sequence) for sequence in sequences}

    # Get available CPUs
    available_cpus = mp.cpu_count()

    # Record the start time
    start_time = time.time()

    # Create a multiprocessing pool
    with mp.Pool(available_cpus) as pool:
        all_results = []
        
        # Run the parameter combinations in parallel for each sequence
        for sequence in sequences:
            # Queue up all parameter combinations for this sequence
            sequence_results = pool.starmap(run_simulation, [(params, filtered_ground_truths[sequence], sequence) for params in param_combinations])
            all_results.extend(sequence_results)

    # Record the end time
    end_time = time.time()

    # Calculate the total time taken
    total_time = end_time - start_time

    # Print the total runtime
    print(f"Total runtime: {total_time:.2f} seconds")

    # Flatten the list of lists
    flattened_results = [item for sublist in all_results for item in sublist]

    # Save final results to JSON and CSV
    results_folder = os.path.join(project_path, 'data', 'results')
    os.makedirs(results_folder, exist_ok=True)

    with open(os.path.join(results_folder, 'final_results.json'), 'w') as file:
        json.dump(flattened_results, file, indent=4)

    # write results to a csv file
    df = pd.DataFrame(flattened_results)
    csv_path = os.path.join(results_folder, 'llm_predictions_results.csv')

    # Check if the file already exists
    if os.path.exists(csv_path):
        # Append without writing the header
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        # Write the data with the header
        df.to_csv(csv_path, index=False)