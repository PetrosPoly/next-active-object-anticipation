import numpy as np
import os                                   
import json
import matplotlib.pyplot as plt
from evaluation.metrics import LLMEvaluation
from itertools import product               # added by Petros ()
import pandas as pd



# Project path
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_SRC)
project_path = os.environ.get("PROJECT_ROOT", _SRC)
sequences = ['Apartment_release_clean_seq150_M1292'] # , 'Apartment_release_work_seq107_M1292']
# sequences = ['Apartment_release_work_seq107_M1292'] # , 'Apartment_release_clean_seq150_M1292']

# ==============================================
# Parameters Settting
# ==============================================

# Parameters for the language model module (unchanged)
time_thresholds = [2] # [1, 2, 3, 4, 5]
avg_dot_threshold_highs = [0.7]
avg_dot_threshold_lows = [0.2]
avg_distance_threshold_highs = [3]
avg_distance_threshold_lows = [1]
high_dot_thresholds = [0.7]
distance_thresholds = [1.5]
high_dot_counters_threshold = [15] # [90]  # [15, 30, 45, 60]
distance_counters_threshold = [15] # [90]  # [15, 30, 45, 60] 
variables_window_times = [3.0]

# Parameters for the LLM reactivation module (unchanged)
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
        
def main():
    
    # Initialize storage for metrics
    results = []

    for sequence in sequences:

        # ==============================================
        # Define the filenames / folder names 
        # ==============================================
        
        # Folder to load the parameters compinations 
        parameters_comb = os.path.join(project_path, 'utils', 'json', 'param_combinations.json')

        # Folder to write the ground truth data - Where the filtered ground t
        ground_truth_folder = os.path.join(_REPO, 'results', 'gt', sequence)
        os.makedirs(ground_truth_folder, exist_ok = True)

        # ==============================================
        # Load --> Parameters / Ground Truth 
        # ==============================================

        # # Load the Parameters
        # with open(parameters_comb, 'r') as file:
        #     loaded_param_combinations = json.load(file)

        # Load the Ground Truth
        with open(os.path.join(_REPO, 'results', 'gt', sequence, 'objects_that_moved.json'), 'r') as json_file:
            original_objects_that_moved_dict = json.load(json_file)

        with open(os.path.join(_REPO, 'results', 'gt', sequence, 'user_object_movement.json'), 'r') as json_file:
            user_object_movement = json.load(json_file)

        # Filter the Ground Truth Data based on the user's motion
        filtered_objects = {k: float(v) for k, v in user_object_movement.items() if v > 1.0}
        ground_truth = {float(k): v for k, v in original_objects_that_moved_dict.items() if v in filtered_objects}
        
        # Write the filtered ground truth interactions (filtered by user's motion) in a json  
        with open(os.path.join(ground_truth_folder, 'filtered_ground_truth.json'), 'w') as file:
            json.dump (ground_truth, file, indent =4)
        
        # ==============================================
        # Run through all the predictions 
        # ==============================================
        
        # Predictions
        for parameters in param_combinations:
            
            # Parameters folder name
            parameter_folder_name = (
                f"time_{parameters['time_threshold']}_"
                f"highdot_{parameters['high_dot_threshold']}_"
                f"highdotcount_{parameters['high_dot_counters_threshold']}_"
                f"dist_{parameters['distance_threshold']}_"
                f"distcount_{parameters['distance_counters_threshold']}"
            )
                
            # ==============================================
            # Load the predictions
            # ==============================================
            
            # Load the predictions from each corresponding folder 
            with open(os.path.join(_REPO, 'results', 'predictions', sequence, parameter_folder_name, 'large_language_model_prediction.json')) as json_file:
                predictions_dict = json.load(json_file)

            # LLM predictions
            LLM_predictions = {float(k): v for k, v in predictions_dict.items()}

            # ==============================================
            # Calculate the metrics 
            # ==============================================
            
            # Initialize the LLMEvaluation class
            evaluation = LLMEvaluation(LLM_predictions, ground_truth)
            evaluation.calculate_FP_FN_GT_correspondances()  # Adjust the predictions with ground truth items and calculate FP and FN
            evaluation.calculate_final_TP_FP()               # after getting the correspondances calculate the final FP, TP 
            metrics = evaluation.calculate_metrics()
            
            # Store results
            result = {
                'sequence': sequence,
                'time':parameters['time_threshold'],
                'dot_value': parameters['high_dot_counters_threshold'],
                'distance_value': parameters['high_dot_counters_threshold'],
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
            
            print(result)   

            # Define the folders that we will write the results       
            result_folder = os.path.join(_REPO, 'results', 'eval', sequence, parameter_folder_name)
            os.makedirs(result_folder, exist_ok=True)  

            # Write the correspondances
            write_custom_json(metrics[14], os.path.join(result_folder, 'correspondances.json'))
            
            # Write the results
            with open(os.path.join(result_folder,'results.json'), 'w') as file:
                json.dump (result, file, indent=4)

    # ==============================================
    # Write the results 
    # ==============================================
    
    # Define the folders that we will write the results       
    results_folder = os.path.join(_REPO, 'results', 'eval')
    os.makedirs(results_folder, exist_ok=True)  

    # Write the results
    with open(os.path.join(results_folder, 'results.json')):
        json.dump (results, file, )

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(results)

    # Export the DataFrame to a CSV file
    df.to_csv(os.path.join(results_folder, 'llm_predictions_results.csv'), index=False)

    # ==============================================
    # Plot the results 
    # ==============================================
    
    # plot_results(results)
    # plot_bar_metrics(results)
    # plot_precision_vs_recall(results)
    # plot_tp_fp_pie(results)
    # plot_combined_metrics(results)

if __name__ == "__main__":
    main()