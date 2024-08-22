import streamlit as st
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the saved model, feature names, and label encoders
with open('titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('feature_names.pkl', 'rb') as file:
    feature_names = pickle.load(file)
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Preprocessing function
def preprocess_data(df, label_encoders):
    imputer = SimpleImputer(strategy='median')
    df['Age'] = imputer.fit_transform(df[['Age']])
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Use the same label encoders from the training phase
    for col in ['Sex', 'Embarked']:
        df[col] = label_encoders[col].transform(df[col].fillna(df[col].mode()[0]))
    
    return df

# Streamlit app interface with simple HTML
st.markdown("<h1>Titanic Survival Prediction</h1>", unsafe_allow_html=True)

# Input fields for all features
pclass = st.selectbox('Pclass', [1, 2, 3])
sibsp = st.number_input('SibSp', min_value=0, step=1)
parch = st.number_input('Parch', min_value=0, step=1)
age = st.number_input('Age', min_value=0)
fare = st.number_input('Fare', min_value=0)
sex = st.selectbox('Sex', ['female', 'male'])
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

# Create DataFrame for user inputs with all necessary features
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Age': [age],
    'Fare': [fare],
    'Sex': [sex],
    'Embarked': [embarked]
})

# Preprocess input data
input_data = preprocess_data(input_data, label_encoders)

# Reorder columns to match the training data
input_data = input_data[feature_names]

# Predict survival
if st.button('Predict'):
    prediction = model.predict(input_data)
    result = 'Survived' if prediction[0] == 1 else 'Not Survived'
    st.write(f"<h2>Prediction: {result}</h2>", unsafe_allow_html=True)
