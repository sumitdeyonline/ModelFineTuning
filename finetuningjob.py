from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Upload the training file
file_response = client.files.create(
  file=open("data/dataset.jsonl", "rb"),
  purpose="fine-tune"
)

# 2. Create the fine-tuning job
job = client.fine_tuning.jobs.create(
  training_file=file_response.id, 
  model="gpt-4o-mini-2024-07-18"
)
