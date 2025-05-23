# Prompt Injection Detection via Reward Scoring

This is a Flask web application that detects potential prompt injection by comparing original and paraphrased prompts using a generative model and a learned reward scoring model.

## 🚀 Features

- Generates paraphrased variants of input prompts using a DPO fine-tuned language model.
- Uses Google Gemini API for generating AI responses.
- Compares responses from original and paraphrased prompts using a reward model to detect prompt injection.
- Lightweight Flask web UI for prompt submission and inspection.

## 📦 Requirements
Download models and move them to the `models` directory.:
Add gemini API key to the genai.configure(api_key="")

Install dependencies via:

```bash
pip install -r requirements.txt
```

## 🛠️ Usage 
1. Run the Flask app:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://localhost:5000`.
3. Enter your prompt in the text area and click "Submit".
4. The app will generate paraphrased prompts and display the original and paraphrased responses.
5. The app will display the final decision on whether prompt injection is detected based on the reward scores.
