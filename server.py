{\rtf1\ansi\ansicpg1252\cocoartf2820
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from flask import Flask, request, jsonify\
import torch\
\
# Initialize Flask app\
app = Flask(__name__)\
\
# Define a simple route\
@app.route('/')\
def home():\
    return "Your Flask app is running on Heroku!"\
\
# Example route for predictions\
@app.route('/predict', methods=['POST'])\
def predict():\
    # Dummy logic for prediction (replace with your actual RNN logic)\
    data = request.json.get('sequence', [])\
    if not data:\
        return jsonify(\{'error': 'No sequence provided'\}), 400\
    \
    # Replace the following with your model prediction logic\
    prediction = sum(data) / len(data) if data else 0\
    return jsonify(\{'prediction': prediction\})\
\
if __name__ == "__main__":\
    app.run(debug=True)\
}