from dotenv import load_dotenv

load_dotenv()

import aisuite as ai

# Define the client. You can use this variable inside your graded functions!
CLIENT = ai.Client()

import unittests

# GRADED FUNCTION: generate_draft

def generate_draft(topic: str, model: str = "openai:gpt-4o") -> str: 
    
    ### START CODE HERE ###

    # Define your prompt here. A multi-line f-string is typically used for this.
    prompt = "Write an essay about helipads in Bangalore as per the building code" 

    ### END CODE HERE ###
    
    # Get a response from the LLM by creating a chat with the client.
    response = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )

    return response.choices[0].message.content

# Test your code!
unittests.test_generate_draft(generate_draft)

# GRADED FUNCTION: reflect_on_draft

def reflect_on_draft(draft: str, model: str = "openai:gpt-5-mini") -> str:

    ### START CODE HERE ###

    # Define your prompt here. A multi-line f-string is typically used for this.
    prompt = "Understand the essay about helipads in Bangalore as per the building code and reflect on the same" 

    ### END CODE HERE ###

    # Get a response from the LLM by creating a chat with the client.
    response = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )

    return response.choices[0].message.content

# Test your code!
unittests.test_reflect_on_draft(reflect_on_draft)

# GRADED FUNCTION: revise_draft

def revise_draft(original_draft: str, reflection: str, model: str = "openai:gpt-4o") -> str:

    ### START CODE HERE ###

    # Define your prompt here. A multi-line f-string is typically used for this.
    prompt = "Revise the essay about helipads in Bangalore as per the building code based on reflection of draft" 

    # Get a response from the LLM by creating a chat with the client.
    response = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )

    ### END CODE HERE ###

    return response.choices[0].message.content

# Test your code!
unittests.test_revise_draft(revise_draft)

essay_prompt = "Should social media platforms be regulated by the government?"

# Agent 1 – Draft
draft = generate_draft(essay_prompt)
print("📝 Draft:\n")
print(draft)

# Agent 2 – Reflection
feedback = reflect_on_draft(draft)
print("\n🧠 Feedback:\n")
print(feedback)

# Agent 3 – Revision
revised = revise_draft(draft, feedback)
print("\n✍️ Revised:\n")
print(revised)

from utils import show_output

essay_prompt = "Should social media platforms be regulated by the government?"