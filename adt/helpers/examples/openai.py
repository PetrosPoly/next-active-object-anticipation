import openai
import logging

# from openai import OpenAI                                                                  # Me: in case you use openai >= 1.0.0 
# import os                                                                                  # Me: in case you use openai >= 1.0.0 

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
You tried to access openai.ChatCompletion, but this is no longer supported in openai>=1.0.0
see the README at https://github.com/openai/openai-python for the API.
"""

def query_llm(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Use GPT-4 model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

# def query_llm_completion(prompt):
#     response = client.completion.create(model="gpt-4")  # Use GPT-4 model
#     return response.choices[0].message['content']

def main():
    # Example prompt
    prompt = "What is the capital of France?"
    response = query_llm(prompt)
    print(response)
    logging.info(f"LLM Response: {response}")

if __name__ == "__main__":
    main()
