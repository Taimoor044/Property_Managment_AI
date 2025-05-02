import streamlit as st
from transformers import pipeline

# Load the model
generator = pipeline("text-generation", model="distilgpt2")

# Load training data
with open("training_data.txt", "r", encoding="utf-8") as f:
    training_data = f.read()

# Streamlit app
st.title("Property Management AI")
st.write("Ask questions or request assistance with property management tasks.")

# User input
user_input = st.text_area("Enter your query", "Draft an email to a leaseholder requesting payment for overdue service charges.")

if st.button("Submit"):
    # Construct the prompt
    system_prompt = """You are a property management AI assistant. Use the following examples to guide your responses:\n\n"""
    system_prompt += training_data
    full_prompt = f"{system_prompt}\n\nUser Query: {user_input}\n\nAssistant Response: "
    
    # Generate response
    response = generator(full_prompt, max_length=600, num_return_sequences=1, truncation=True, pad_token_id=50256)
    generated_text = response[0]["generated_text"]
    
    # Extract response
    response_start = generated_text[len(full_prompt):].strip()
    if not response_start:
        response_start = "No response generated. Please try rephrasing your query."
    
    # Display response
    st.write("**Response:**")
    st.write(response_start)