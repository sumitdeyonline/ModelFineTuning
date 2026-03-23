import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re

# Set page config
st.set_page_config(page_title="JSON Extraction Model", page_icon="🤖", layout="centered")

@st.cache_resource(show_spinner="Loading Fine-Tuned Model... (This may take a minute)")
def get_model():
    model_path = "./data/lora-model"
    # Load tokenizers and base model with LoRA adapters
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

# Load globally via Streamlit Cache
model, tokenizer = get_model()

def build_prompt(instruction, input_text):
    return f"""### Instruction:
{instruction} (Output strictly as JSON)

### Input:
{input_text}

### Response:
"""

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.2,
            do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_json(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            clean_str = match.group().replace("'", '"')
            return json.loads(clean_str)
    except Exception as e:
        print("Parsing error:", e)
    return None

# Streamlit Interface
st.title("🤖 JSON Extraction Fine-tuned Model")
st.markdown("Enter an instruction and an input text to test out the LoRA fine-tuned model's extraction capabilities.")

st.sidebar.header("Configuration")
st.sidebar.markdown("Model: **Qwen2.5-1.5B (LoRA Fine-tuned)**")

# UI Inputs
instruction = st.text_input("Instruction:", value="Extract name and age", placeholder="e.g. Extract product and price")
input_text = st.text_area("Input Text:", value="Alice is 30 years old", placeholder="Paste the text here...", height=100)

if st.button("Extract JSON", type="primary"):
    if not instruction or not input_text:
        st.warning("Please provide both an Instruction and Input Text.")
    else:
        with st.spinner("Generating response..."):
            prompt = build_prompt(instruction, input_text)
            raw_output = generate_response(prompt)
            parsed = extract_json(raw_output)

        st.subheader("Results")
        if parsed:
            st.success("Successfully Parsed JSON!")
            st.json(parsed)
        else:
            st.error("Failed to parse a valid JSON object.")
        
        with st.expander("Show Raw Model Output"):
            st.text(raw_output)
