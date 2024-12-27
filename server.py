from flask import Flask, request, jsonify
import os  # Import os first

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch  # Import PyTorch after setting the environment variable

# Initialize Flask app
app = Flask(__name__)

# Define a simple route
@app.route('/')
def home():
    return "Your Flask app is running on Heroku!"

# Example route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Dummy logic for prediction (replace with your actual RNN logic)
    data = request.json.get('sequence', [])
    if not data:
        return jsonify({'error': 'No sequence provided'}), 400

    # Replace the following with your model prediction logic
    prediction = sum(data) / len(data) if data else 0
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
