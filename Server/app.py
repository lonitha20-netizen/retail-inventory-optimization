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
        data = {
            'Store': [int(request.form['Store'])],
            'Dept': [int(request.form['Dept'])],
            'IsHoliday': [0], 
            'Size': [int(request.form['Size'])],
            'Temperature': [55.0],
            'Fuel_Price': [3.5],
            'CPI': [210.0],
            'Unemployment': [8.0],
            'Type': [int(request.form['Type'])],
            'Month': [int(request.form['Month'])],
            'WeekOfYear': [int(request.form['Month']) * 4]
        }
        df_input = pd.DataFrame(data)
        prediction = model.predict(df_input)[0]
        return render_template('index.html', prediction=round(prediction, 2))
    except Exception as e:
        return f"Prediction Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
