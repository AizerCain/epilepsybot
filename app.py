from flask import Flask, request, jsonify
import json
import pickle
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import requests

app = Flask(__name__)

# Load Model & Assets
model = tf.keras.models.load_model("epilepsy_risk_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

with open("epilepsy_faq.json") as f:
    faq_data = json.load(f)

# Risk Mapping
risk_map = {
    "symptoms": "High",
    "precaution": "Moderate",
    "lifestyle": "Low",
    "emergency": "High"
}

SAFETY_PROMPT = "You are a helpful assistant for epilepsy awareness. Never diagnose. Recommend professional help when needed."

# DeepSeek API Details
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "your_deepseek_api_key_here"

def predict_tag(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=20)
    pred = model.predict(padded)[0]
    tag = label_encoder.inverse_transform([np.argmax(pred)])[0]
    return tag

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_input = data.get("message")

    if not user_input:
        return jsonify({"error": "Missing 'message' field"}), 400

    tag = predict_tag(user_input)
    risk_level = risk_map.get(tag, "Low")

    # New prompt for a full-sentence DeepSeek response
    full_prompt = f"""
    {SAFETY_PROMPT}
    
    User Question: "{user_input}"
    Risk Level (Predicted): {risk_level}
    Topic Category: {tag}

    Respond in a natural, conversational way as if you're directly answering the user.
    """

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SAFETY_PROMPT},
            {"role": "user", "content": full_prompt}
        ]
    }

    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=body)

    if response.status_code == 200:
        reply = response.json()['choices'][0]['message']['content']
        return jsonify({"message": reply})  # ✅ Final conversational response
    else:
        return jsonify({"error": "DeepSeek API error", "details": response.text}), 500

if __name__ == '__main__':
    app.run(debug=True)
