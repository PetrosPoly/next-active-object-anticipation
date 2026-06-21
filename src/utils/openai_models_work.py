# Import necessary libraries 
from openai import OpenAI, OpenAIError, APIStatusError, AsyncOpenAI, RateLimitError
import tiktoken
import logging
import os
import csv
import yaml
import json
from utils.tools import load_config

# Load the configuration
config = load_config()

# OpenAI client, initialized lazily so the module can be imported (and the
# perception-only pipeline can run) without an API key. The key is only
# required when the LLM is actually queried.
_client = None
def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it to use the LLM, or run perception-only (omit --use_llm)."
            )
        _client = OpenAI(api_key=api_key)
    return _client

# Project paths
project_path = os.path.expanduser(config["project_path"])
txt_folders = os.path.join(project_path, "utils", "txt_files")
os.makedirs(txt_folders, exist_ok=True)

# Interaction log filename
filename = config["interaction_log_filename"]
filepath = os.path.join(txt_folders, filename)

# Prompt filename
prompt_name = config["prompt_filename"]
print('Prompt name:', prompt_name)
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

def activate_llm(log_content, parameters, max_retries = 5):
    delay = 1  # Initial delay in seconds

    # Ensure log_content is a string
    if not isinstance(log_content, str):
        log_content = json.dumps(log_content)  # Convert to JSON string if it's a dictionary or list

    # models 
    """
    1. gpt_4o_mini: $0.150/1M input tokens  ----> Affordable and intelligent small model for fast, lightweigth tasks 
    2. gpt-4o: $5/1M input tokens           ----> High Intelligence flaghship model for complex, multi-step tasks
    """
   
    models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]      # gpt_4o_mini: $0.150/1M input tokes while gpt-4o: $5/1M input tokens 
    # Read the prompts from the file
    prompts = read_prompts_from_file(prompt_path)

    # Use the prompts in your code
    prompt_instruction = prompts.get('prompt_instruction', '')
    prompt_reasoning = prompts.get('prompt_reasoning', '')
    prompt_predict = prompts.get('prompt_predict', '')
    
    # Manage the number of tokens 
    full_prompt = prompt_instruction + prompt_reasoning + prompt_predict

    # try: 
    max_tokens = 30000  # Set your token limit
    
    # Check if the combined tokens are within the limit   
    within_limit, total_tokens = check_token_limit(full_prompt, log_content, max_tokens - 1000)  # Adjust for response tokens

    if not within_limit:
        print(f"Skipping request: Token limit exceeded ({total_tokens} > {max_tokens - 1000})")
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

    print('Message to LLM:', message_to_LLM)
    
    response = get_client().chat.completions.create(
    model= models[0],  # Use GPT-4o mini model
    messages= message_to_LLM
    )

    # LLM reply
    llm_generated_msg = response.choices[0].message.content
    print('LLM reply:', llm_generated_msg)
    return llm_generated_msg

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
        print(f"Error parsing YAML: {e}")
        raise
    except KeyError as e:
        print(f"Key not found in the response: {e}")
        raise
    except ValueError as e:
        print(f"Validation error: {e}")
        raise

# Initialize the tokenizer for the OpenAI GPT-3 or GPT-4 model
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(prompt):
    """Count the number of tokens in a given prompt."""
    tokens = tokenizer.encode(prompt)
    return len(tokens)

def check_token_limit(prompt, log, max_tokens):
    """Check if the combined tokens of prompt and log are within the limit."""
    print('Prompt:', prompt)
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