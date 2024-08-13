# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:45:05 2024

@author: nidhi
"""
# 1. Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Load the dataset

dataset=pd.read_csv(r"E:\titanic_project\tested.csv")

""" OVERVIEW OF DATASET
The dataset consists of 418 entries with the following columns:

PassengerId: Unique identifier for each passenger.
Survived: Whether the passenger survived (0 = No, 1 = Yes).
Pclass: Passenger's class (1 = 1st, 2 = 2nd, 3 = 3rd).
Name: Name of the passenger.
Sex: Gender of the passenger.
Age: Age of the passenger (86 missing values).
SibSp: Number of siblings or spouses aboard the Titanic.
Parch: Number of parents or children aboard the Titanic.
Ticket: Ticket number.
Fare: Fare paid by the passenger (1 missing value).
Cabin: Cabin number (327 missing values).
Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)."""


## 3. Understanding the data

dataset.info()  ## Check the structure and summary of the dataset
dataset.describe() ## Summary statistics
dataset.shape ## (418, 12)
dataset.columns
"""
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')"""
dataset.dtypes 

# Display the first/last 5 rows of the dataset
dataset.head(5)
dataset.tail(5)

# Summary Statistics
dataset.Age.mean() # The mean age of passengers is around 30.3 years.
dataset.Pclass.mean() # Most passengers are in 3rd class, with a mean Pclass of 2.27.
dataset.Fare.mean() # The mean fare paid is around 35.6 units of currency.

# 4. Data Cleaning

### 4.a. Data Type Conversion
dataset.dtypes
# int to float
dataset.Age = dataset.Age.astype('float32')
dataset.dtypes

dataset.Fare = dataset.Fare.astype('float32')
dataset.dtypes


### 4.b. Missing Values
dataset.isna().sum()
"""
Age: 86 missing values.
Fare: 1 missing value.
Cabin: 327 missing values (most passengers do not have cabin information)."""

# Handling Missing Values
"""
Age: We'll fill missing Age values with the median age.
Fare: We'll fill the single missing Fare value with the median fare.
Cabin: We'll drop this column due to a high percentage of missing values."""

# Fill missing Age & Fare values with the median age
dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

# Drop the Cabin column as it has too many missing values
dataset.drop('Cabin', axis=1, inplace=True)

dataset.isna().sum() # Now, there is no null value

### 4.c. Variance
dataset.var() == 0

### 4.d. Duplicates 
duplicate = dataset.duplicated()
duplicate
sum(duplicate) # there no duplicate 



# 5. Feature Engineering

"""
Creating a Family Size Feature
We can create a new feature FamilySize by combining SibSp (siblings/spouses aboard) 
and Parch (parents/children aboard)."""
# Create a new feature 'FamilySize'
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# FamilySize: Combining SibSp and Parch to create a new FamilySize feature is a good idea, as it can provide insight into whether traveling with family affects survival.


# Extract Title from Name
dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Replace rare titles with 'Rare'
dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 
                                        'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

dataset['Title'].value_counts()

# Drop unnecessary columns
dataset.drop(['Name', 'Ticket', 'PassengerId','Parch','SibSp'], axis=1, inplace=True)
dataset.info()

# 6. Visualization

# Barplot for Embarked vs. Survival
sns.barplot(x='Embarked', y='Survived', data=dataset)
plt.title('Survival Rate by Port of Embarkation')
plt.xlabel('Port of Embarkation')
plt.ylabel('Survival Rate')
plt.show()

# Pie chart for Sex distribution
sex_counts = dataset['Sex'].value_counts()
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%')
plt.title('Gender Distribution')
plt.show()

# Barplot for Title vs. Survival
sns.barplot(x='Title', y='Survived', data=dataset)
plt.title('Survival Rate by Title')
plt.xlabel('Title')
plt.ylabel('Survival Rate')
plt.xticks(rotation=45)
plt.show()


# Encode categorical variables
dataset = pd.get_dummies(dataset, columns=['Sex', 'Embarked', 'Title'], drop_first=True)


dataset.Survived.value_counts()
dataset.Pclass.value_counts()

### 4.e. Let's find outliers in Age & Fare
sns.boxplot(dataset.Age)
sns.boxplot(dataset.Fare)

# Detection of outliers (find limits for Age based on IQR)
IQR = dataset['Age'].quantile(0.75) - dataset['Age'].quantile(0.25)

lower_limit = dataset['Age'].quantile(0.25) - (IQR * 1.5)
upper_limit = dataset['Age'].quantile(0.75) + (IQR * 1.5)


# Replace the outliers by the maximum and minimum limit
dataset['Age_replaced'] = pd.DataFrame(np.where(dataset['Age'] > upper_limit, upper_limit, np.where(dataset['Age'] < lower_limit, lower_limit, dataset['Age'])))
sns.boxplot(dataset.Age_replaced)

# Detection of outliers (find limits for Fare based on IQR)
IQR = dataset['Fare'].quantile(0.75) - dataset['Fare'].quantile(0.25)

lower_limit1 = dataset['Fare'].quantile(0.25) - (IQR * 1.5)
upper_limit1 = dataset['Fare'].quantile(0.75) + (IQR * 1.5)


# Replace the outliers by the maximum and minimum limit
dataset['Fare_replaced'] = pd.DataFrame(np.where(dataset['Fare'] > upper_limit1, upper_limit1, np.where(dataset['Fare'] < lower_limit1, lower_limit1, dataset['Fare'])))
sns.boxplot(dataset.Fare_replaced)

# Drop unnecessary columns
dataset.drop(['Age', 'Fare'], axis=1, inplace=True)
dataset.info()


# Set the style for seaborn
sns.set(style="whitegrid")

# Bar Graph for Survival Rates by Passenger Class
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', data=dataset, ci=None)
plt.title('Survival Rates by Passenger Class')
plt.ylabel('Survival Rate')
plt.xlabel('Passenger Class')
plt.show()

# Count Plot for Number of Passengers Survived or Not
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', data=dataset)
plt.title('Count of Passengers Survived vs Not Survived')
plt.xlabel('Survived (1=Yes, 0=No)')
plt.ylabel('Count')
plt.show()

# Pie Chart for Survival Rates
survived_counts = dataset['Survived'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(survived_counts, labels=['Not Survived', 'Survived'], autopct='%1.1f%%', startangle=140)
plt.title('Survival Rates')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()

# Histogram for Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(dataset['Age_replaced'], bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Histogram for Fare Distribution
plt.figure(figsize=(10, 6))
sns.histplot(dataset['Fare_replaced'], bins=30, kde=True)
plt.title('Fare Distribution of Passengers')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

# Countplot for FamilySize
sns.countplot(x='FamilySize', hue='Survived', data=dataset)
plt.title('Survival Count by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Count')
plt.show()


# 7. Model Building

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Prepare features and target variable
X = dataset.drop('Survived', axis=1)
y = dataset['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 8. Model Improvement and Validation

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Define the model
rf_model = RandomForestClassifier()

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f"Accuracy of Best Model: {accuracy_score(y_test, y_pred_best):.2f}")



importances = best_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for feature importances
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.show()



import pickle

# Save the model to a file using pickle
with open('titanic_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)