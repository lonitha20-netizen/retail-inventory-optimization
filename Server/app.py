from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# This part safely looks for your model file
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'clustering_model.pkl')

try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Success: Model loaded perfectly!")
    else:
        model = None
        print("Warning: model file not found at path.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

# This is the part that was causing the crash
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Prediction Error: Model is not loaded on the server. Please check logs."
    
    # ... rest of your prediction code here ...
    return "Prediction successful"

if __name__ == "__main__":
    app.run(debug=True)
