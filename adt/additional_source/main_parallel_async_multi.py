# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import numpy as np

from itertools import product               

import os                                   
import logging

import asyncio
import multiprocessing
from multiprocessing import Process, Queue, Pool, cpu_count, Manager
from algorithm import execute_algorithm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_path", type=str, required=True, help="path to the ADT sequence")
    parser.add_argument("--device_number", type=int, default=0, help="Device_number you want to visualize, default is 0")
    parser.add_argument("--down_sampling_factor", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--jpeg_quality", type=int, default=75, help=argparse.SUPPRESS)
    parser.add_argument("--rrd_output_path", type=str, default="", help=argparse.SUPPRESS  )                                  # Me: If this path is set, we will save the rerun (.rrd) file to the given path
    parser.add_argument("--use_llm", action='store_true',help="If you include it in arguments becomes True")                              # Me: added by Petros, if there is a value that 
    parser.add_argument("--runrr", action='store_true',help="Run the the visualization part..same as above")   
    parser.add_argument("--visualize_objects", action='store_true',help="Visualize the objects in the rerun.io")   
    return parser.parse_args()

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
main_logger = logging.getLogger(__name__)

# ==============================================
# Parameters Settting
# ==============================================

# Parameters for the language model module
time_thresholds = [2] # [1, 2, 3]                       
avg_dot_threshold_highs = [0.7]                         
avg_dot_threshold_lows = [0.2]                          
avg_distance_threshold_highs = [3]                       
avg_distance_threshold_lows = [1]                       

high_dot_thresholds = [0.9] # [0.7, 0.8, 0.9]                 # [0.5, 0.6, 0.7, 0.8, 0.9]                 
distance_thresholds = [2] # [1.5, 2, 2.5]                        
high_dot_counters_threshold = [60] # [15, 30, 45, 60, 75, 90]  
distance_counters_threshold = [15, 30, 45, 60] # , 75, 90]  

variables_window_times = [3.0]                          

# Parameters for the LLM reactivation module
minimum_time_deactivated = [2.0]                        
maximum_time_deactivated = [5.0]                        
user_relative_movement = [2.0]                          
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
        "distance_counters_threshold": dct,       # Corrected to match the earlier definition
        "window_time": w, 
        "minimum_time_deactivated": mintd,                
        "maximum_time_deactivated": maxtd,             
        "user_relative_movement": urm,                 
        "object_percentage_overlap" : obo,   
    }
    for t, adh, adl, adhg, adlg, hdt, dt, hdct, dct, w, mintd, maxtd, urm, obo in product(
        time_thresholds, avg_dot_threshold_highs, avg_dot_threshold_lows, 
        avg_distance_threshold_highs, avg_distance_threshold_lows,
        high_dot_thresholds, distance_thresholds,
        high_dot_counters_threshold, distance_counters_threshold, variables_window_times, 
        minimum_time_deactivated, maximum_time_deactivated, user_relative_movement, object_percentage_overlap   
    )
]

# ==============================================
# Run through the parameter combinatons in parallel
# ==============================================

# # Get the number of available CPUs
available_cpus = multiprocessing.cpu_count()
print(f"Available CPUs: {available_cpus}")

# This function will run within each process, creating a local asyncio event loop for handling asynchronous tasks.
def process_param_combinations(params_batch, result_queue, data):
    async def run_all_tasks():
        # Create tasks for each parameter combination in the batch
        tasks = [run_algorithm_async(params, result_queue, data) for params in params_batch]
        await asyncio.gather(*tasks)

    # Run the asyncio loop in each process
    asyncio.run(run_all_tasks())

# This function runs execute_algorithm asynchronously for each parameter combination
async def run_algorithm_async(parameters, queue, data):
    await execute_algorithm(parameters, data, queue)

# Load paths, data, and initialize any other necessary components here
def load_data_and_initialize(args):
    project_path = "Documents/projectaria_sandbox/projectaria_tools/projects/AriaDigitalTwinDatasetTools/object_anticipation/adt/"
    
    sequence_path = args.sequence_path
    datasets_path = 'Documents/projectaria_tools_adt_data/'
    dataset_folder = os.path.join(datasets_path, sequence_path)
    json_folder = os.path.join(project_path,'utils','json')
    txt_folder = os.path.join(project_path, 'utils', 'txt_files')

    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)                        
    os.makedirs(txt_folder, exist_ok=True)  

    vrsfile = os.path.join(dataset_folder, "video.vrs")
    ADT_trajectory_file = os.path.join(dataset_folder, "aria_trajectory.csv")
    json_file = os.path.join(json_folder,'param_combinations.json')
    movement_time_dict = os.path.join(project_path,'data','gt',args.sequence_path,'movement_time_dict.json')
    prompt_path = os.path.join(txt_folder,'prompts.txt')

    return {
        "project_path": project_path,
        "dataset_folder": dataset_folder,
        "vrsfile": vrsfile,
        "ADT_trajectory_file": ADT_trajectory_file,
        "json_file": json_file, 
        "movement_time_dict": movement_time_dict,
        "prompt": prompt_path,
    }

def main():

    args = parse_args()  # Obtain command-line arguments
    num_processes = min(cpu_count(), 8)  # Limit the number of concurrent processes

    # Step 1: Load data once to pass to each process
    data = load_data_and_initialize(args)

    # Step 2: Split parameter combinations among processes
    params_batches = np.array_split(param_combinations, num_processes)

    # Step 3: Create a multiprocessing Queue to gather LLM results
    result_queue = Queue()

    # Step 4: Launch each process with its batch of parameters and data
    processes = [
        Process(target=process_param_combinations, args=(params_batch, result_queue, data))
        for params_batch in params_batches
    ]
    for process in processes:
        process.start()
    
    # wait for the process to finish 
    for process in processes:
        process.join()

if __name__ == "__main__":
    main()

