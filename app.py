from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from peft import PeftModel, PeftConfig
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch.nn as nn
import os

app = Flask(__name__)

genai.configure(api_key="")

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model  # Pre-trained LM
        self.scorer = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask)
        hidden_states = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.scorer(hidden_states)


# To use the model later:
tokenizer_loaded = AutoTokenizer.from_pretrained("bert-base-uncased") # Make sure tokenizer matches training
base_model_loaded = AutoModel.from_pretrained("bert-base-uncased") # Make sure base model matches training
reward_model_loaded = RewardModel(base_model_loaded)

# 2. Load the saved state dict
reward_model_loaded.load_state_dict(torch.load('models/reward_model_state_dict.pth'))
reward_model_loaded.eval() # Set to evaluation mode

# Example of using the loaded model for prediction:
def predict_reward(prompt: str, answer: str, model, tokenizer, max_length=128):
    text = prompt + " " + answer
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        prediction = model(inputs["input_ids"], inputs["attention_mask"])
    return prediction.item()


# 1. Load tokenizer & model
model_path = "models/prompt_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForCausalLM.from_pretrained(model_path)

# 2. Fix pad & chat template
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Build a pipeline that does NOT return the full text
dpo_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
)

def generate_paraphrase_with_dpo(prompt: str) -> str:
    """
    Generates just the paraphrase (no echo of the original).
    """
    # 4. Build your chat-style prompt
    full_prompt = f"""Generate 2 diverse paraphrases of this prompt that might elicit different responses.
    Focus on changing sentence structure and wording while maintaining the original intent.
    Return only the paraphrased sentences as plain text, one per line, with no numbering, no bullet points, and no additional commentary.
    Original: {prompt}

    Paraphrases:"""

    response = dpo_generator(
        full_prompt,
        max_new_tokens=50,
        temperature=0.8,
        num_return_sequences=1,
        do_sample=True
    )

    output = response[0]['generated_text'].replace(prompt, "").strip()
    variants = [v.strip() for v in output.split('\n') if v.strip()]
    return list(dict.fromkeys(variants))[0]

def getResults(qustion_txt):
  model = genai.GenerativeModel("gemini-1.5-flash")
  response = model.generate_content(qustion_txt)
  return response.text

# Route for rendering the main page
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint to handle the form submission
@app.route('/submit', methods=['POST'])
def handle_prompt():
    input_prompt = request.json['prompt']

    result=generate_paraphrase_with_dpo(input_prompt)
    altered_prompt=result.replace('"',"")

    original_result = getResults(input_prompt)
    altered_result = getResults(altered_prompt)

    reward=predict_reward(input_prompt, altered_result, reward_model_loaded, tokenizer_loaded)

    if reward < 0.5:
      promptinjection_result="no prompt injection"
    else:
      promptinjection_result="prompt injection"

    return jsonify({
        'originalResult': original_result,
        'alteredResult': altered_result,
        'promptinjection':promptinjection_result
    })

if __name__ == '__main__':
    app.run()