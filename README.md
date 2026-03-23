# 🤖 Local LLM Fine-Tuning & JSON Extraction API

This project demonstrates how to fine-tune an open-source Large Language Model (like **Qwen2.5-1.5B**) locally on your machine using **PEFT / LoRA** to perform highly structured tasks such as extracting JSON information from text. It includes scripts for training, testing, and a fully functional Streamlit Web App interface.

## 📁 Project Structure

* **`data/data.json`**: The dataset containing `instruction`, `input`, and `output` pairs used for fine-tuning.
* **`PretrainedModel.py`**: A complete, integrated script that loads the base Hugging Face model, configures LoRA, trains the model using `SFTTrainer`, and saves the adapter weights locally to `data/lora-model`.
* **`main.py`**: A CLI script demonstrating how to load the fine-tuned adapter weights and run rapid text-generation inference to structure input data into parsed JSON.
* **`app.py`**: A beautiful Streamlit Web App that visually wraps the inference logic for an interactive, user-friendly experience using `@st.cache_resource` for optimized loading times.
* **`src/`**: Modularized scripts (`loadModel.py` and `modelTranning.py`) exploring different ways to configure the pipeline.

---

## 🚀 Setup Instructions

This project uses `uv` for lightning-fast Python package management.

**1. Install Dependencies**
```bash
uv sync
```
*(Note: If you are running this on a Mac, the `uv sync` process automatically uses CPU-compatible PyTorch and older versions of transformers specifically tailored to avoid C++ incompatibility crashes!)*

**2. Verify Environment**
If you need to verify your local python version (the standard command uses `--version`, not `-version`!):
```bash
uv run python --version
```

---

## 🧠 1. Fine-Tuning the Model

The default model used is `Qwen/Qwen2.5-1.5B`. It focuses on a small, fast structure making it perfect for laptops.

To trigger the PEFT fine-tuning pipeline on the dataset:
```bash
uv run python PretrainedModel.py
```
* **Output:** The trained LoRA adapter weights will be dumped to `./data/lora-model`
* **Underfitting Note:** If the model hasn't completely learned your custom feature extraction formats, consider increasing the dataset size in `data.json` or increasing `num_train_epochs` inside `PretrainedModel.py`!

---

## 💻 2. Testing Inference (CLI)

Once fine-tuned, you can test it directly via the terminal. The `main.py` script automatically builds an instruction prompt, generates output, and safely parses it into JSON:

```bash
uv run python main.py
```

---

## 🌐 3. Launching the Web Interace (Streamlit)

Want a visual interface instead? Launch the Streamlit application!

```bash
uv run streamlit run app.py
```
This will start a local webserver. Open http://localhost:8501 in your browser. The app caches the 1.5GB model upon first load, making all subsequent extractions instantaneous.
