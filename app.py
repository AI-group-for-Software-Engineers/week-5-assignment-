# app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('dropout_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    return jsonify({'dropout_risk': bool(prediction)})

if __name__ == '__main__':
    app.run(debug=True)