import numpy as np

from projectaria_tools.core.sophus import SE3

from collections import deque
from typing import Dict, List, Tuple      

import asyncio
from openai import OpenAI, OpenAIError, APIStatusError, AsyncOpenAI, RateLimitError, Timeout
import tiktoken

import random
import logging
import os
import csv
import yaml
import json
import time

from functools import wraps
import time
import logging
import asyncio

# Project path
project_path = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
txt_folders = os.path.join(project_path, 'utils', 'txt_files')

# Interaction log filename
filename = 'interaction_log.txt'
filepath = os.path.join(txt_folders, filename)

# Set up logging configuration to log to a file
logging.basicConfig(filename=filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenAI 
client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Global variables
global total_api_calls 
total_api_calls = 0
total_api_calls_lock = asyncio.Lock()

# Functions
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

def async_retry_with_exponential_backoff(
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
        async def wrapper(*args, **kwargs):
            num_retries = 0
            delay = initial_delay

            while True:
                try:
                    return await func(*args, **kwargs)
                except errors as e:
                    num_retries += 1
                    if num_retries > max_retries:
                        print(f"Maximum retries ({max_retries}) exceeded.")
                        raise

                    delay_with_jitter = delay * (1 + jitter * random.random())
                    print(f"Error: {e}. Retrying in {delay_with_jitter:.2f} seconds.")
                    await asyncio.sleep(delay_with_jitter)
                    delay *= exponential_base
                except Exception as e:
                    raise e
        return wrapper
    return decorator

@async_retry_with_exponential_backoff()
async def activate_llm(log_content, parameters, rate_limiter, prompt_path, max_retries=5):
    """Activate the LLM with retry logic and rate limiting."""
    
    # Initialize retry count and API call counter
    retry_count = 0

    # Read the prompts from the file
    prompts = read_prompts_from_file(prompt_path)
    prompt_instruction = prompts.get('prompt_instruction', '')
    prompt_reasoning = prompts.get('prompt_reasoning', '')
    prompt_predict = prompts.get('prompt_predict', '')

    # Construct the prompt
    full_prompt = prompt_instruction + prompt_reasoning + prompt_predict
    max_tokens = 30000  # Set token limit

    # Check if token limit is within acceptable range
    within_limit, total_tokens = check_token_limit(full_prompt, log_content, max_tokens - 1000)
    if not within_limit:
        logging.info(f"Skipping request: Token limit exceeded ({total_tokens} > {max_tokens - 1000})")
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

    # Retry logic for calling the API
    while retry_count < max_retries:
        try:
            # Apply rate limiting before making API call
            await rate_limiter.acquire()
            total_api_calls += 1
            logging.info(f"API Call #{total_api_calls} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Make the API call
            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # Example model, adjust as needed
                messages=message_to_LLM
            )

            # Process and return the response
            llm_generated_msg = response.choices[0].message.content
            logging.info(f"LLM reply: {llm_generated_msg}")
            return llm_generated_msg

        except Exception as e:
            retry_count += 1
            logging.error(f"Error during LLM call attempt {retry_count}/{max_retries}: {e}")
            if retry_count == max_retries:
                logging.error("Max retries reached, returning None.")
                return None  # Return None after max retries


def clean_llm_response(llm_response):
    # Strip leading and trailing whitespace and triple quotes
    cleaned_response = llm_response.strip('"""').strip()
    return cleaned_response


def process_llm_response(llm_response, parameters):
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
        logging.error(f"Error parsing YAML: {e}")
        logging.error(f"Error parsing YAML: {e} | Parameters: {parameters}")
    except KeyError as e:
        logging.error(f"Key not found in the response: {e}")
        print(f"Key not found in the response: {e}| Parameters: {parameters}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.error(f"An unexpected error occurred: {e} | Parameters: {parameters}")
    return None, None, None, None

# Initialize the tokenizer for the OpenAI GPT-3 or GPT-4 model
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(prompt):
    """Count the number of tokens in a given prompt."""
    tokens = tokenizer.encode(prompt)
    return len(tokens)

def check_token_limit(prompt, log, max_tokens):
    """Check if the combined tokens of prompt and log are within the limit."""
    prompt_tokens = count_tokens(prompt)
    log_tokens = count_tokens(log)
    total_tokens = prompt_tokens + log_tokens
    if total_tokens > max_tokens:
        return False, total_tokens
    return True, total_tokens

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