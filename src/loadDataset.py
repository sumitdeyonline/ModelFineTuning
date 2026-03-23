from datasets import load_dataset

#dataset = load_dataset("json", data_files="data/data.json")

def format_example(example):
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    }

def getDataSet():
    dataset = load_dataset("json", data_files="data/data.json")
    dataset = dataset.map(format_example)
    return dataset

# for item in dataset['train']:
#     print(item['text'])
#     print("-" * 40)

