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

# Dropbox link to the model file
model_url = "https://www.dropbox.com/s/abcd1234/simple_rnn.pth?dl=1"
model_path = "simple_rnn.pth"

# Download the model if not present
if not os.path.exists(model_path):
    print("Downloading model from Dropbox...")
    response = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    print("Model downloaded successfully!")

# Load the model architecture
model = SimpleRNN(input_size=1, hidden_size=10, output_size=1)

# Load the state dictionary
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()

@app.route('/')
def home():
    return "Your Flask app is running on Render!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('sequence', [])
    if not data:
        return jsonify({'error': 'No sequence provided'}), 400

    # Example logic: Replace with your actual model inference
    input_tensor = torch.tensor(data, dtype=torch.float32).view(-1, 1, 1)  # Reshape for RNN
    prediction = model(input_tensor).item()  # Perform inference
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
