# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:00:23 2024

@author: nidhi
"""

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

# Create and configure the Flask application
app = Flask(__name__)

# Load the pre-trained model from file
with open('titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Extracting features from the form
    pclass = int(request.form['pclass'])
    age = float(request.form['age'])
    fare = float(request.form['fare'])
    sex = request.form['sex']
    embarked = request.form['embarked']
    family_size = int(request.form['familySize'])
    title = request.form['title']

    # Create a DataFrame for the new input
    data = {
        'Pclass': [pclass],
        'Age_replaced': [age],
        'Fare_replaced': [fare],
        'FamilySize': [family_size],
        'Sex_male': [1 if sex == 'male' else 0],
        'Embarked_Q': [1 if embarked == 'Q' else 0],
        'Embarked_S': [1 if embarked == 'S' else 0],
        'Title_Miss': [1 if title == 'Miss' else 0],
        'Title_Mr': [1 if title == 'Mr' else 0],
        'Title_Mrs': [1 if title == 'Mrs' else 0],
        'Title_Rare': [1 if title == 'Rare' else 0]
    }
    input_df = pd.DataFrame(data)

    # Make a prediction
    prediction = model.predict(input_df)

    # Return the prediction result
    if prediction[0] == 1:
        result = "Survived"
    else:
        result = "Not Survived"

    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)