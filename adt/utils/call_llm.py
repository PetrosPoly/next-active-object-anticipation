# Import necessary libraries
from openai import OpenAI, OpenAIError
import tiktoken
import logging
import os
import csv
import yaml
import time

# Load configuration from a YAML file
def load_config(config_path="../config.yaml"):
    """
    Load configuration from a YAML file located in a folder higher than the current script.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Initialize OpenAI client
def initialize_openai_client():
    """
    Initialize the OpenAI client using the API key from environment variables.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key)

# Set up logging
def setup_logging(log_filepath):
    """
    Set up logging to a specified file.
    """
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    logging.basicConfig(
        filename=log_filepath,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Read prompts from a file
def read_prompts_from_file(file_path):
    """
    Read and parse prompts from a file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found at {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    sections = content.split('---')
    prompts = {}
    for section in sections:
        if ':' in section:
            key, value = section.split(':', 1)
            prompts[key.strip()] = value.strip()
    return prompts

# Clean LLM response
def clean_llm_response(llm_response):
    """
    Cleans the LLM response by stripping unwanted characters.
    """
    cleaned_response = llm_response.strip('"""').strip()
    return cleaned_response

# Process LLM response
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
        logging.error(f"Error parsing YAML: {e}")
        raise
    except KeyError as e:
        logging.error(f"Key not found in the response: {e}")
        raise
    except ValueError as e:
        logging.error(f"Validation error: {e}")
        raise

# Initialize the tokenizer for the OpenAI GPT-3 or GPT-4 model
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(prompt):
    """
    Count the number of tokens in a given prompt.
    """
    tokens = tokenizer.encode(prompt)
    return len(tokens)

def check_token_limit(prompt, log, max_tokens):
    """
    Check if the combined tokens of prompt and log are within the limit.
    """
    prompt_tokens = count_tokens(prompt)
    log_tokens = count_tokens(log)
    total_tokens = prompt_tokens + log_tokens
    if total_tokens > max_tokens:
        return False, total_tokens
    return True, total_tokens

# Log data to a CSV file
def log_to_csv(timestamp_ns, obj_id, obj_name, time_to_contact, csv_file):
    """
    Log data to a CSV file.
    """
    write_header = not os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['Timestep', 'Object ID', 'Object Name', 'Time to Contact'])
        writer.writerow([timestamp_ns, obj_id, obj_name, time_to_contact])

# Activate the LLM
def activate_llm(client, log_content, parameters, prompts, max_tokens=30000, model="gpt-4o-mini"):
    """
    Activate the LLM with the given log content, parameters, and prompts.
    """
    full_prompt = prompts.get('prompt_instruction', '') + prompts.get('prompt_reasoning', '') + prompts.get('prompt_predict', '')

    within_limit, total_tokens = check_token_limit(full_prompt, log_content, max_tokens - 1000)
    if not within_limit:
        logging.warning(f"Token limit exceeded: {total_tokens} > {max_tokens - 1000}")
        return None

    message_to_llm = [
        {"role": "system", "content": "You are an AI assistant that predicts user interactions with objects based on spatial context."},
        {"role": "user", "content": f"Spatial context information: {log_content}"},
        {"role": "user", "content": f"Thresholds: {parameters}"},
        {"role": "user", "content": f"Instructions: {prompts.get('prompt_instruction', '')}"},
        {"role": "user", "content": f"Rationale: {prompts.get('prompt_reasoning', '')}"},
        {"role": "user", "content": f"Prediction: {prompts.get('prompt_predict', '')}"}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=message_to_llm
    )
    llm_generated_msg = response.choices[0].message.content
    logging.info(f"LLM Response: {llm_generated_msg}")
    return llm_generated_msg

# Main execution
if __name__ == "__main__":
    try:
        # Load configuration
        config = load_config()

        # Initialize OpenAI client
        client = initialize_openai_client()

        # Set up logging
        setup_logging(os.path.expanduser(config["project_path"] + "/utils/txt_files/" + config["interaction_log_filename"]))

        # Read prompts
        prompt_path = os.path.expanduser(config["project_path"] + "/utils/txt_files/" + config["prompt_filename"])
        prompts = read_prompts_from_file(prompt_path)

        # Example usage of activate_llm
        log_content = "Example log content"
        parameters = {
            "high_dot_counters_threshold": 0.8,
            "distance_counters_threshold": 1.5,
            "time_threshold": 5
        }
        response = activate_llm(client, log_content, parameters, prompts)
        print("LLM Response:", response)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")