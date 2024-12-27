import os
import requests
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

# Dropbox link to the model file
model_url = "https://www.dropbox.com/scl/fi/y0gag93rcsmpx70k2j8eh/simple_rnn.pth?rlkey=w0u0w19rm3m3owr1er2yoegsw&dl=1"
model_path = "simple_rnn.pth"

# Download the model if not present
if not os.path.exists(model_path):
    print("Downloading model from Dropbox...")
    response = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    print("Model downloaded successfully!")

# Load the model
model = torch.load(model_path)
model.eval()  # Set the model to evaluation mode

@app.route('/')
def home():
    return "Your Flask app is running on Heroku!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('sequence', [])
    if not data:
        return jsonify({'error': 'No sequence provided'}), 400

    # Example logic: Replace with actual model inference
    prediction = sum(data) / len(data) if data else 0
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
