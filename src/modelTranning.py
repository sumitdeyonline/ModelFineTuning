from transformers import TrainingArguments
from trl import SFTTrainer


def traningModel(model, tokenizer, dataset):
    # Training Setup

    training_args = TrainingArguments(
        output_dir="./data/finetuned-model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
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
    return model, tokenizer


