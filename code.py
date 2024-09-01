import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the Titanic dataset
file_path = r'E:\1_titanic_project/tested.csv'
titanic_data = pd.read_csv(file_path)

# Preprocessing: Handle missing values and encode categorical variables
def preprocess_data(df, is_train=True, label_encoders=None):
    # Drop unnecessary columns
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
    
    # Fill missing values
    imputer = SimpleImputer(strategy='median')
    df['Age'] = imputer.fit_transform(df[['Age']])
    df['Fare'].fillna(df['Fare'].median(), inplace=True)  # Fill missing 'Fare'
    
    # Encode categorical variables
    if is_train:
        label_encoders = {}
        for col in ['Sex', 'Embarked']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna(df[col].mode()[0]))
            label_encoders[col] = le
    else:
        for col in ['Sex', 'Embarked']:
            df[col] = label_encoders[col].transform(df[col].fillna(df[col].mode()[0]))
    
    return df, label_encoders

# Preprocess data (save label encoders for future use)
titanic_data, label_encoders = preprocess_data(titanic_data, is_train=True)

# Split data
X = titanic_data.drop(columns='Survived')
y = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
print("Feature Importances:\n", feature_importance)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("Mean cross-validation score: ", cv_scores.mean())

# Save the model and necessary objects
with open('titanic_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the feature names
with open('feature_names.pkl', 'wb') as file:
    pickle.dump(X.columns.tolist(), file)

# Save the label encoders
with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)
