import numpy as np
import random
import time

from projectaria_tools.core.sophus import SE3

from collections import deque
from typing import Dict, List, Tuple      

from openai import OpenAI, OpenAIError, APIStatusError, RateLimitError, Timeout
import tiktoken

import logging
import os
import csv
import yaml
import json

# OpenAI 
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Project path
project_path = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
txt_folders = os.path.join(project_path, 'utils', 'txt_files')

# Prompt filename
prompt_name = 'prompts.txt'
prompt_path = os.path.join(txt_folders,prompt_name)

os.makedirs("logs", exist_ok=True)                                        

# Initialize a logger for this module
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.DEBUG)
if not module_logger.handlers:
    fh = logging.FileHandler("logs/openai_models.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    module_logger.addHandler(fh)

# Read the prompt from the file
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

# Create the content
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

# Decorator to retry in case the output of the LLM is not the expected one
def retry_with_exponential_backoff(
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (
        APIStatusError,
        RateLimitError,
        OpenAIError, 
    ),
):
    def decorator(func):
        def wrapper(*args, **kwargs):
            num_retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except errors as e:
                    num_retries += 1
                    
                    if num_retries > max_retries:
                        logging.error(f"Maximum retries ({max_retries}) exceeded for function {func.__name__}.")
                        raise
                    delay = initial_delay * exponential_base ** (num_retries - 1)

                    if jitter:
                        delay *= random.uniform(0.5, 1.5) # add jitter 
                    logging.warning(f"Rate limit exceeded. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                except Exception as e:
                    logging.error(f"Unexpected error in function {func.__name__}")
                    raise e
        return wrapper
    return decorator

@retry_with_exponential_backoff()
def activate_llm(log_content, parameters, max_retries = 5):

    """
    Activates the LLM to generate predictions based on the provided log content and parameters.
    """

    # models 
    """
    1. gpt_4o_mini: $0.150/1M input tokens  ----> Affordable and intelligent small model for fast, lightweigth tasks 
    2. gpt-4o: $5/1M input tokens           ----> High Intelligence flaghship model for complex, multi-step tasks
    """
   
    models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]      # gpt_4o_mini: $0.150/1M input tokes while gpt-4o: $5/1M input tokens 
   
    # Read the prompts from the file
    prompts = read_prompts_from_file(prompt_path)
    prompt_instruction = prompts.get('prompt_instruction', '')
    prompt_reasoning = prompts.get('prompt_reasoning', '')
    prompt_predict = prompts.get('prompt_predict', '')
    full_prompt = prompt_instruction + prompt_reasoning + prompt_predict

    # try: 
    max_tokens = 30000  # Set your token limit
    
    # Check if the combined tokens are within the limit   
    within_limit, total_tokens = check_token_limit(full_prompt, log_content, max_tokens - 1000)  # Adjust for response tokens

    if not within_limit:
        module_logger.warning(f"Skipping request: Token limit exceeded ({total_tokens} > {max_tokens - 1000})")
        return None
    
    message_to_LLM = [
    {"role": "system", "content": "You are an AI assistant that continuously predicts the objects the user might want to interact with, based on the spatial context."},
    {"role": "assistant", "content": "The user is performing a specific task and interacts with various objects sequentially to complete it."},
    {"role": "user", "content": f"Spatial context information: {log_content}"},
    {"role": "user", "content": f"Thresholds: focus = {parameters['high_dot_counters_threshold']}, distance = {parameters['distance_counters_threshold']}, time = {parameters['time_threshold']}."},
    {"role": "user", "content": f"Instructions regarding the provided context: {prompt_instruction}"},
    {"role": "user", "content": f"Rationale behind the selection: {prompt_reasoning}"},
    {"role": "user", "content": f"Prediction: {prompt_predict}"}
]

    # print('Message to LLM:', message_to_LLM)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Adjust model as needed
            messages=message_to_LLM
        )
        llm_generated_msg = response.choices[0].message.content
        return llm_generated_msg
    except Exception as e:
        module_logger.error(f"Failed to activate LLM: {e}")
        raise

def get_valid_llm_response(log_content, parameters, max_retries=5):
    """
    Retrieves a valid LLM response, processing it accordingly.
    """
    for attempt in range(1, max_retries + 1):
        try:
            llm_response = activate_llm(log_content, parameters)
            if not llm_response:
                raise ValueError("LLM response is empty.")

            processed_response = process_llm_response(llm_response)
            return processed_response
        except Exception as e:
            module_logger.error(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                module_logger.error("Max retries reached. Raising exception.")
                raise
            else:
                delay = 2 ** attempt
                module_logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)

def clean_llm_response(llm_response):
    """
    Cleans the LLM response by stripping unwanted characters.
    """
    cleaned_response = llm_response.strip('"""').strip()
    return cleaned_response

def process_llm_response(llm_response):
    """
    Processes the LLM response, parsing it and validating required fields.
    """
    cleaned_response = clean_llm_response(llm_response)
    try:
        data = yaml.safe_load(cleaned_response)
        most_likely_objects_to_interact_with = data['most_likely_objects_to_interact_with']
        rationale = data['rationale']
        predicted_interaction_objects = data['predicted_interaction_objects']
        goal = data['goal_of_the_user']
        if not all([most_likely_objects_to_interact_with, rationale, predicted_interaction_objects, goal]):
            raise ValueError("One or more required fields are missing or empty in the LLM response.")
        return most_likely_objects_to_interact_with, rationale, predicted_interaction_objects, goal
    except yaml.YAMLError as e:
        module_logger.error(f"Error parsing YAML: {e}")
        raise
    except KeyError as e:
        module_logger.error(f"Key not found in the response: {e}")
        raise
    except ValueError as e:
        module_logger.error(f"Validation error: {e}")
        raise

# Initialize the tokenizer for the OpenAI GPT-3 or GPT-4 model
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(prompt):
    """
    Counts the number of tokens in a given prompt.
    """
    tokens = tokenizer.encode(prompt)
    return len(tokens)

def check_token_limit(prompt, log, max_tokens):
    """
    Checks if the combined tokens of prompt and log are within the limit.
    """
    prompt_tokens = count_tokens(prompt)
    log_tokens = count_tokens(log)
    total_tokens = prompt_tokens + log_tokens
    if total_tokens > max_tokens:
        return False, total_tokens
    return True, total_tokens

def log_to_csv(timestamp_ns, obj_id, obj_name, time, csv_file):
    """
    Logs data to a CSV file.
    """
    write_header = not os.path.exists(csv_file)
    if write_header:
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestep', 'Object ID', 'Object Name', 'Time to Contact'])
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp_ns, obj_id, obj_name, time])


def setup_logger(log_filename):
    """
    Sets up a logger for a given filename.
    """
    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger