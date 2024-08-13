Titanic Survival Prediction
Overview
This project involves predicting the survival of passengers aboard the Titanic using various features from the dataset. The goal is to build a predictive model that can accurately determine whether a passenger survived based on their characteristics.

Dataset
The dataset contains the following columns:

PassengerId: Unique identifier for each passenger.
Survived: Whether the passenger survived (0 = No, 1 = Yes).
Pclass: Ticket class (1, 2, or 3).
Name: Name of the passenger.
Sex: Gender of the passenger.
Age: Age of the passenger (86 missing values).
SibSp: Number of siblings/spouses aboard.
Parch: Number of parents/children aboard.
Ticket: Ticket number.
Fare: Ticket fare (1 missing value).
Cabin: Cabin number (327 missing values).
Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
Project Structure
1. Import Libraries
The project uses the following libraries:

pandas for data manipulation.
numpy for numerical operations.
seaborn and matplotlib for data visualization.
scikit-learn for machine learning and model evaluation.
pickle for saving the trained model.
2. Load the Dataset
The dataset is loaded from a CSV file located at E:\titanic_project\tested.csv. The dataset consists of 418 entries with 12 columns.

3. Understanding the Data
Inspect the Dataset: Review the dataset structure, summary statistics, and the first/last few rows.
Summary Statistics: Calculate the mean age, ticket class, and fare to understand the distribution.
4. Data Cleaning
Data Type Conversion: Convert Age and Fare to float32 for consistency.
Missing Values:
Age: Fill missing values with the median age.
Fare: Fill the single missing value with the median fare.
Cabin: Drop the column due to a high percentage of missing values.
Variance: Check for columns with zero variance.
Duplicates: Verify and remove any duplicate rows.
5. Feature Engineering
Family Size: Create a new feature combining SibSp and Parch to represent the family size.
Title Extraction: Extract and standardize titles from passenger names.
Drop Unnecessary Columns: Remove columns that are not needed for the analysis.
6. Visualization
Barplot for Embarked vs. Survival: Visualize survival rates based on the port of embarkation.
Pie Chart for Sex Distribution: Show the distribution of genders among passengers.
Barplot for Title vs. Survival: Analyze survival rates based on titles.
Histograms and Boxplots: Examine distributions and detect outliers in Age and Fare.
Countplots and Pie Charts: Show survival counts and survival rates.
7. Model Building
Prepare Data: Split the data into training and testing sets.
Train Logistic Regression Model: Fit a logistic regression model to predict survival.
Evaluate Model: Use accuracy, classification report, and confusion matrix to assess model performance.
8. Model Improvement and Validation
Random Forest Classifier: Use GridSearchCV to tune hyperparameters and find the best model.
Feature Importance: Analyze and visualize the importance of features in the best model.
9. Model Saving
Save the trained model using pickle for future use.
