import os

base_path = "/Users/petrospolydorou/ETH_Thesis/coding/Actionanticipation/projectaria_sandbox/projectaria_tools/projectaria_tools/utils"
promptname = 'prompts.txt'
prompt_path = os.path.join(base_path,promptname)

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

# Read the prompts from the file
prompts = read_prompts_from_file(prompt_path)

# Use the prompts in your code
prompt_constant = prompts.get('prompt_constant', '')
prompt_change = prompts.get('prompt_change', '')

# Now you can use prompt_constant and prompt_change in your code
full_prompt = prompt_constant + prompt_change

# Example usage
print("Prompt Constant:", prompt_constant)
print("Prompt Change:", prompt_change)