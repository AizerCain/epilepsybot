from flask import Flask, request, jsonify
import json
import pickle
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import requests
import traceback

app = Flask(__name__)

# Load Model & Assets with Exception Handling
try:
    model = tf.keras.models.load_model("epilepsy_risk_model.h5")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    with open("epilepsy_faq.json") as f:
        faq_data = json.load(f)
except Exception as e:
    print(f"❌ Error loading model or assets: {str(e)}")
    traceback.print_exc()
    model, tokenizer, label_encoder, faq_data = None, None, None, None

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
DEEPSEEK_API_KEY = "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"  # 🔴 Replace this with your actual API key


def predict_tag(text):
    """Predicts the category tag for the given text using the trained model."""
    try:
        if not model or not tokenizer or not label_encoder:
            return None, "⚠️ Model or tokenizer not loaded correctly."

        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=20)
        pred = model.predict(padded)[0]
        tag = label_encoder.inverse_transform([np.argmax(pred)])[0]
        return tag, None  # No errors
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        traceback.print_exc()
        return None, "⚠️ Internal model error."


@app.route("/ask", methods=["POST"])
def ask():
    """Handles POST requests to the /ask endpoint."""
    try:
        data = request.json
        user_input = data.get("message")

        if not user_input:
            return jsonify({"error": "Missing 'message' field"}), 400

        tag, error = predict_tag(user_input)
        if error:
            return jsonify({"error": error}), 500

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

        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=body, timeout=10)

        if response.status_code == 200:
            try:
                deepseek_response = response.json()
                reply = deepseek_response.get("choices", [{}])[0].get("message", {}).get("content", "No response generated.")
                return jsonify({"message": reply})
            except Exception as e:
                print(f"❌ Error processing DeepSeek response: {str(e)}")
                traceback.print_exc()
                return jsonify({"error": "DeepSeek API response format error"}), 500
        else:
            print(f"❌ DeepSeek API error: {response.status_code} - {response.text}")
            return jsonify({"error": "DeepSeek API error", "details": response.text}), 500

    except Exception as e:
        print(f"❌ Internal Server Error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == '__main__':
    app.run(debug=True)
