import gradio as gr
from transformers import pipeline

# Load a smaller model from Hugging Face
generator = pipeline("text-generation", model="distilgpt2")

def chat_with_model(user_input):
    system_prompt = """You are a property management AI assistant. Use the following examples to guide your responses:\n\n"""
    with open("training_data.txt", "r", encoding="utf-8") as f:
        system_prompt += f.read()
    full_prompt = f"{system_prompt}\n\nUser: {user_input} ### Response:"
    
    # Generate response using the model
    response = generator(full_prompt, max_length=500, num_return_sequences=1, truncation=True)
    return response[0]["generated_text"].split("### Response:")[1].strip()

# Create the Gradio interface
iface = gr.Interface(
    fn=chat_with_model,
    inputs=gr.Textbox(label="Enter your query"),
    outputs=gr.Textbox(label="Response"),
    title="Property Management AI",
    description="Ask questions or request assistance with property management tasks."
)

iface.launch()