#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import joblib
from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf

app = Flask(__name__)

scaler_path = 'scaler.save'
model_path = 'water_model.h5'

print("Current directory:", os.getcwd())
print("Files in current directory:", os.listdir(os.getcwd()))
print("Loading scaler from:", scaler_path)

try:
    scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully.")
except FileNotFoundError:
    print("File not found:", scaler_path)
except Exception as e:
    print("Error loading scaler:", str(e))

try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("File not found:", model_path)
except Exception as e:
    print("Error loading model:", str(e))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    
    if not file.filename.endswith('.xlsx'):
        return jsonify({"error": "File must be an .xlsx Excel file"}), 400

    df = pd.read_excel(file)
    
    expected_columns = ['Solar radiation', 'Relative humidity', 'Air temperature', 'Windspeed (m/s)',
                        'Water temperature (W.T.)', 'Glass temperature (G.T.)']
    if list(df.columns) != expected_columns:
        return jsonify({"error": f"Incorrect columns. Expected columns: {expected_columns}"}), 400

    input_data = df.values
    input_data_normalized = scaler.transform(input_data)
    
    predictions = model.predict(input_data_normalized)
    predictions = predictions.flatten().tolist()

    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


# In[ ]:




