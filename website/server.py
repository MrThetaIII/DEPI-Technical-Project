from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)
CORS(app)

# Configuration
MODELS_DIR = "models"
MODEL_FILE = "destillbert.pt"  # Or your specific checkpoint file
TOKENIZER_DIR = "distilbert-base-uncased"  # For tokenizer if not in checkpoint
MAX_LEN = 128  # Maximum sequence length for tokenizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_DIR)
model = DistilBertForSequenceClassification.from_pretrained(TOKENIZER_DIR, num_labels=2)

# Load checkpoint
model_path = os.path.join(MODELS_DIR, MODEL_FILE)
try:
    pretrain_data = torch.load(model_path, map_location=device)
    model.load_state_dict(pretrain_data['model_state_dict'])
    print(f"Loaded checkpoint from {model_path}")
except FileNotFoundError:
    print(f"Checkpoint not found at {model_path}. Using pretrained model.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    exit(1)  # Exit if checkpoint loading fails critically

model.to(device)
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        tweets = data.get('tweets', None)  # Expecting a list of tweets

        if tweets is None or not isinstance(tweets, list):
            return jsonify({'error': 'Missing or invalid "tweets" field. Must be a list.'}), 400

        results = []
        for tweet_data in tweets:
            tweet = tweet_data.get('text', None)
            if tweet is None or tweet.strip() == "":
                results.append({'error': 'Missing or empty "text" field for a tweet.'})
                continue

            inputs = tokenizer(tweet, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()

            sentiment = "positive" if pred_class == 1 else "negative"

            results.append({
                'text': tweet,         # Include original text in response
                'sentiment': sentiment,
                'confidence': confidence
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False)
