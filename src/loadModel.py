from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def loadModel():
    load_dotenv()
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
    return model, tokenizer