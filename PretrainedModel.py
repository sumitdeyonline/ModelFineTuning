import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer
from transformers import pipeline
from loadDataset import getDataSet

load_dotenv()

# Load Dataset
dataset = getDataSet()

model_name = "Qwen/Qwen2.5-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)

print("Model loaded successfully!")

# 2. Configure LoRA
lora_config = LoraConfig(
    r=16,  # Rank (dimension of the update matrices)
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training Setup

training_args = TrainingArguments(
    output_dir="./finetuned-model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=20,
    logging_steps=10,
    save_strategy="epoch"
)

# Train the Model
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args
)

trainer.train()


# Save the Model
model.save_pretrained("data/lora-model")
tokenizer.save_pretrained("data/lora-model")


# Inference (Test Your Fine-Tuned Model)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = """### Instruction:
Extract name and age

### Input:
Alice is 30 years old

### Response:
"""

result = pipe(prompt, max_new_tokens=50)
print(result[0]["generated_text"])