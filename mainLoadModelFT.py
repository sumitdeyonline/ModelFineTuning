
from src.loadDataset import getDataSet
from src.loadModel import loadModel
from src.modelTranning import traningModel
from transformers import pipeline


def main():

    #Load dataset
    dataset = getDataSet()
    #print(dataset)

    #Load model
    model, tokenizer = loadModel()

    #Train model
    model, tokenizer = traningModel(model, tokenizer, dataset)

    #Save model
    # model.save_pretrained("data/lora-model")
    # tokenizer.save_pretrained("data/lora-model")

    #Inference
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    prompt = """### Instruction:
Extract name and age

### Input:
Alice is 30 years old

### Response:
"""

    result = pipe(prompt, max_new_tokens=50)
    print(result[0]["generated_text"])

if __name__ == "__main__":
    main()