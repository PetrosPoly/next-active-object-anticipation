import numpy as np
from projectaria_tools.core.sophus import SE3
from collections import deque
from typing import Dict, List, Tuple      

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llamaapi import LlamaAPI

import logging
import os
import csv
import yaml
import json

# Initialize the tokenizer and model
print('The Llama Key is the following', os.environ['LLAMA_API_KEY'])
llama = LlamaAPI(os.environ['LLAMA_API_KEY'])

# Project path
# project_path = "/home/ppolydorou/Documents/projectaria_sandbox/next_active_object_anticipation_projectaria_twin_dataset/projects/AriaDigitalTwinDatasetTools/object_anticipation/adt"
project_path = "Documents/projectaria_sandbox/projectaria_tools/projects/AriaDigitalTwinDatasetTools/object_anticipation/adt/"
txt_folders = os.path.join(project_path, 'utils', 'txt_files')

# Interaction log filename
filename = 'interaction_log.txt'
filepath = os.path.join(txt_folders, filename)

# Prompt filename
prompt_name = 'prompts.txt'
prompt_path = os.path.join(txt_folders, prompt_name)

# Set up logging configuration to log to a file
logging.basicConfig(filename=filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

history_log = []

def read_prompts_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
    # Split the content based on the delimiter (e.g., '---')
    sections = content.split('---')
    prompts = {}
    for section in sections:
        if ':' in section:
            key, value = section.split(':', 1)
            prompts[key.strip()] = value.strip()
    return prompts

def append_to_history_string(time, 
                             location, 
                             filtered_names_high_dot_counts_and_distance_counts,
                             filtered_names_low_distance_counts_and_high_dot_counts, 
                             filtered_names_high_dot_counts_and_distance_values,
                             filtered_names_low_distance_counts_and_high_dot_values,
                             time_to_approach_dict, 
                             predictions_dict):
    
    log_entry = {
        'timestamp': time,
        'place': location,
        'focus_consistency_from_user_to_objects_measured_in_counts': filtered_names_high_dot_counts_and_distance_counts,
        'proximity_consistency_from_user_to_objects_measured_in_counts': filtered_names_low_distance_counts_and_high_dot_counts,
        'current_distance_from_user_to_objects_measured_in_meters': filtered_names_low_distance_counts_and_high_dot_values,
        'time_to_approach_objects_measured_in_seconds': time_to_approach_dict, 
        'past_predictions_with_timestamps': predictions_dict  
    }
    
    return log_entry

def activate_llama(log_content, parameters):

    # Read the prompts from the file
    prompts = read_prompts_from_file(prompt_path)

    # Use the prompts in your code
    prompt_instruction = prompts.get('prompt_instruction', '')
    prompt_reasoning = prompts.get('prompt_reasoning', '')
    prompt_predict = prompts.get('prompt_predict', '')
    
    # Build the messages for the API
    messages = [
        {"role": "system", "content": "You are an AI assistant that continuously predicts the objects the user might want to interact with, based on the spatial context."},
        {"role": "assistant", "content": "The user is performing a specific task and interacts with various objects sequentially to complete it."},
        {"role": "user", "content": f"Spatial context information: {log_content}"},
        {"role": "user", "content": f"Thresholds: focus = {parameters['high_dot_counters_threshold']}, distance = {parameters['distance_counters_threshold']}, time = {parameters['time_threshold']}."},
        {"role": "user", "content": f"Instructions regarding the provided context: {prompt_instruction}"},
        {"role": "user", "content": f"Rationale behind the selection: {prompt_reasoning}"},
        {"role": "user", "content": f"Prediction: {prompt_predict}"}
    ]
    
    print('Messages to LLM:', messages)
    
    # Prepare the API request
    api_request_json = {
        "model": "llama3.1-8b",  # Replace with the actual model name
        "messages": messages
    }
    
    # Make the API call
    response = llama.run(api_request_json)
    
    # Handle the response
    response_json = response.json()
    
    # Extract the assistant's reply
    if 'choices' in response_json and len(response_json['choices']) > 0:
        llm_generated_msg = response_json['choices'][0]['message']['content']
    else:
        llm_generated_msg = ''
        print('No response from LLM')
    
    print('LLM reply:', llm_generated_msg)
    return llm_generated_msg

def clean_llm_response(llm_response):
    # Strip leading and trailing whitespace and triple quotes
    cleaned_response = llm_response.strip('"""').strip()
    return cleaned_response

def process_llm_response(llm_response):
    cleaned_response = clean_llm_response(llm_response)
    # Clean and parse the YAML response
    try:
        data = yaml.safe_load(cleaned_response)
        most_likely_objects_to_interact_with = data['most_likely_objects_to_interact_with']
        rationale = data['rationale']
        predicted_interaction_objects = data['predicted_interaction_objects']
        goal = data['goal_of_the_user']
        return most_likely_objects_to_interact_with, rationale, predicted_interaction_objects, goal
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
    except KeyError as e:
        print(f"Key not found in the response: {e}")
    return [], '', [], ''

def log_to_csv(timestamp_ns, obj_id, obj_name, time, csv_file):
    write_header = not os.path.exists(csv_file)
    # Ensure the CSV header is written only once
    if write_header:
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestep', 'Object ID', 'Object Name', 'Time to Contact'])
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp_ns, obj_id, obj_name, time])
            
def setup_logger(log_filename):
    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
