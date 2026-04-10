from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# This is the simplest way to find the file in the same folder as app.py
model_path = os.path.join(os.path.dirname(__file__), 'clustering_model.pkl')

try:
    model = joblib.load(model_path)
    print("Success: Model loaded perfectly!")
except Exception as e:
    model = None
    print(f"Error: Could not load the model. Reason: {e}")
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

       df_input = pd.DataFrame(data)

prediction = model.predict(df_input)[0]

labels = {
    0: "Standard Inventory",
    1: "High Priority",
    2: "Seasonal/Promotional"
}

result = labels.get(prediction, "Unknown")

return render_template('index.html', prediction=result)
    except Exception as e:
        return f"Prediction Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
