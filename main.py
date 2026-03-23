from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re


def loadModel():
    print("Loading finetune data....")
    model_path = "./data/lora-model"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    return model, tokenizer

def build_prompt(instruction, input_text):
    return f"""### Instruction:
{instruction} (Output strictly as JSON)

### Input:
{input_text}

### Response:
"""

def generate_response(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.2,   # lower = more deterministic
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_json(text):
    try:
        # Extract JSON block using regex
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            clean_str = match.group().replace("'", '"')
            return json.loads(clean_str)
    except Exception as e:
        print("Parsing error:", e)

    return None

def main():
    ## Load Model
    model, tokenizer = loadModel()

    instruction = "Extract device and battery life"
    input_text = "This phone has 10 hours of battery life"

    # Generate Prompt
    prompt = build_prompt(instruction, input_text)

    # Generate Raw Output
    raw_output  = generate_response(prompt, tokenizer, model)
    print("Raw Output:\n", raw_output)

    parsed = extract_json(raw_output)
    print("Parsed JSON:\n", parsed)


if __name__ == "__main__":
    main()