# Titanic Survival Prediction

Welcome to the Titanic Survival Prediction repository! This project uses machine learning to predict the survival of Titanic passengers based on their characteristics. The goal is to apply data preprocessing, model training, and evaluation techniques to build a predictive model and deploy it via a web application.

## Overview
The Titanic Survival Prediction project demonstrates how to:

###### Preprocess Data: Handle missing values and encode categorical features.
###### Build and Train a Model: Use a Random Forest Classifier to predict survival.
###### Evaluate Model Performance: Assess accuracy and generate classification reports.
###### Tune Hyperparameters: Optimize model performance with GridSearchCV.
###### Deploy the Model: Create a Streamlit app for interactive predictions.

## Dataset
The dataset contains the following columns:
###### PassengerId: Unique identifier for each passenger.
###### Survived: Whether the passenger survived (0 = No, 1 = Yes).
###### Pclass: Ticket class (1, 2, or 3).
###### Name: Name of the passenger.
###### Sex: Gender of the passenger.
###### Age: Age of the passenger (86 missing values).
###### SibSp: Number of siblings/spouses aboard.
###### Parch: Number of parents/children aboard.
###### Ticket: Ticket number.
###### Fare: Ticket fare (1 missing value).
###### Cabin: Cabin number (327 missing values).
###### Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Key Files
###### titanic_model.pkl: Pickle file containing the trained Random Forest model.
###### feature_names.pkl: Pickle file with the list of feature names used in the model.
###### label_encoders.pkl: Pickle file with the label encoders for categorical features.
###### app.py: Streamlit application for making predictions via a web interface.
###### titanic_data_preprocessing.py: Script for data preprocessing, model training, and saving.
