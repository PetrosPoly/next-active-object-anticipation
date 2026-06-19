import yaml

def clean_llm_response(llm_response):
    # Strip leading and trailing whitespace and triple quotes
    cleaned_response = llm_response.strip('"""').strip()
    return cleaned_response

def process_llm_response(llm_response):
    cleaned_response = clean_llm_response(llm_response)
    
    # Print the cleaned response to check its format
    print("Cleaned Response:")
    print(cleaned_response)

    # Parse the YAML response without try-except block for debugging purposes
    data = yaml.safe_load(cleaned_response)
    
    # Print the parsed data to check if it was successful
    print('\n')
    print(f"Parsed YAML Data:, {data['most_likely_objects_to_interact_with']}\n")
    print(f"Parsed YAML Data:, {data['rationale']}\n")
    print(f"Parsed YAML Data:, {data['goal_of_the_user']}\n")
    
    # Extract the required information
    predicted_interaction_objects = data['predicted_interaction_objects']
    goal = data['goal_of_the_user']
    
    return predicted_interaction_objects, goal

llm_response = """
most_likely_objects_to_interact_with:
    - object: MuffinPan
      probability: 0.35
    - object: ChoppingBoard
      probability: 0.33
    - object: BlackCeramicMug
      probability: 0.32

rationale:
    - object: MuffinPan
      reason: "The MuffinPan has the highest focus intensity (0.994) and a significant count of 55 in high focus objects, indicating strong user attention."
    - object: ChoppingBoard
      reason: "The ChoppingBoard has a high focus intensity of 0.974 and a high count of 58, showing it was a primary focus for the user."
    - object: BlackCeramicMug
      reason: "The BlackCeramicMug has a high focus intensity of 0.970 and notable counts in high focus (58) and nearby objects (4), suggesting both focus and proximity interest."

predicted_interaction_objects:
    - MuffinPan
    - ChoppingBoard
    - BlackCeramicMug

goal_of_the_user: "Based on above predictions and the past predictions from the list (previous_predictions) I assume that the goal of the user is, "The user's goal appears to be related to preparing food or drink, potentially baking or making a beverage, considering the mixing and preparation tools like the MuffinPan, ChoppingBoard, and CeramicMug."
"""

# Run the function
result = process_llm_response(llm_response)
print("Result:", result)