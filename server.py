import os
import torch
import torch.nn as nn
import requests
from flask import Flask, request, jsonify

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define the model architecture
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)  # Initialize hidden state
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Use the last time step
        return out

# Initialize Flask app
app = Flask(__name__)

# Google Drive link to the model file
model_url = "https://drive.google.com/uc?id=19Xq4G-4mR93TWE6r7dGjM2M6qX3Dtjhc"
model_path = "simple_rnn.pth"

# Download the model if not present
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    response = requests.get(model_url, stream=True)
    with open(model_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded successfully!")

# Initialize the model with the correct architecture
model = SimpleRNN(input_size=1, hidden_size=8, output_size=1)  # Match hidden_size=8
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))  # Load weights
model.eval()  # Set model to evaluation mode

# Home route
@app.route('/')
def home():
    return "Your Flask app is running on Render!"

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse and validate input
        data = request.json.get('sequence', [])
        if not isinstance(data, list) or not data:
            return jsonify({'error': 'Input must be a non-empty list of numbers'}), 400

        # Reshape input data and make predictions
        input_tensor = torch.tensor(data, dtype=torch.float32).view(-1, 1, 1)  # Reshape for RNN
        predictions = model(input_tensor).squeeze().tolist()  # Perform inference and convert to list

        # Ensure predictions are always returned as a list
        if isinstance(predictions, float):  # Handle single scalar prediction case
            predictions = [predictions]

        return jsonify({'predictions': predictions})
    except Exception as e:
        print(f"ERROR: {str(e)}")  # Log error details
        return jsonify({'error': 'An error occurred during prediction. Please check your input.'}), 500

if __name__ == "__main__":
    # Use the PORT environment variable or default to 8080
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
