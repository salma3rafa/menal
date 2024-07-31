import clf as clf
import joblib as joblib
from flask import Flask, render_template, redirect, request, session,jsonify
from flask_session import Session
from flask import Flask, render_template, request
import pickle
import pandas as pd
import json
import os
import numpy as np
app=Flask(__name__)
filename = "models/Copy_of_Mental_disorder_predictor(accuracy_94_).pk1"
os.makedirs('models', exist_ok=True)

# Assuming 'clf' is your trained machine learning model
with open('models/Copy_of_Mental_disorder_predictor(accuracy_94_).pkl', 'wb') as file:
    pickle.dump(clf, file)
model = joblib.load(open('models/Copy_of_Mental_disorder_predictor(accuracy_94_).pkl', "rb"))
@app.route('/')
def home():
    return "Depression Prediction Model API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_features = np.array(data['input']).reshape(1, -1)
    prediction = model.predict(input_features)
    return jsonify({'prediction': prediction.tolist()})
if __name__ == '__main__':
    app.run(debug=True)


