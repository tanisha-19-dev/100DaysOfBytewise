import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.datasets import fetch_openml

# Common Preprocessing Function
def preprocess_data(df, target_column, categorical_features, numerical_features, missing_value_strategy='mean'):
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if missing_value_strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif missing_value_strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif missing_value_strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Split into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Define the preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )
    
    return X, y, preprocessor

# 1. Classifying Loan Status Using Decision Trees
# Load dataset
loan_df = pd.read_csv('path_to_lending_club_loan_data.csv')

# Define features
categorical_features_loan = ['loan_grade', 'sub_grade']
numerical_features_loan = ['loan_amount', 'annual_income']

# Preprocess data
X_loan, y_loan, preprocessor_loan = preprocess_data(
    loan_df, 'loan_status', categorical_features_loan, numerical_features_loan, 'mean'
)

# Create pipeline
pipeline_loan = Pipeline(steps=[
    ('preprocessor', preprocessor_loan),
    ('classifier', DecisionTreeClassifier())
])

# Split data
X_train_loan, X_test_loan, y_train_loan, y_test_loan = train_test_split(X_loan, y_loan, test_size=0.2, random_state=42)

# Train model
pipeline_loan.fit(X_train_loan, y_train_loan)

# Predict
y_pred_loan = pipeline_loan.predict(X_test_loan)

# Evaluate model
accuracy_loan = accuracy_score(y_test_loan, y_pred_loan)
roc_auc_loan = roc_auc_score(y_test_loan, y_pred_loan)

print(f"Loan Status - Accuracy: {accuracy_loan}, ROC-AUC: {roc_auc_loan}")

# 2. Predicting Hospital Readmission Using Logistic Regression
# Load dataset
hospital_df = pd.read_csv('path_to_hospital_readmission_dataset.csv')

# Define features
categorical_features_hospital = ['hospital_type', 'region']
numerical_features_hospital = ['age', 'bmi', 'length_of_stay']

# Preprocess data
X_hospital, y_hospital, preprocessor_hospital = preprocess_data(
    hospital_df, 'readmission', categorical_features_hospital, numerical_features_hospital, 'mode'
)

# Create pipeline
pipeline_hospital = Pipeline(steps=[
    ('preprocessor', preprocessor_hospital),
    ('classifier', LogisticRegression())
])

# Split data
X_train_hospital, X_test_hospital, y_train_hospital, y_test_hospital = train_test_split(X_hospital, y_hospital, test_size=0.2, random_state=42)

# Train model
pipeline_hospital.fit(X_train_hospital, y_train_hospital)

# Predict
y_pred_hospital = pipeline_hospital.predict(X_test_hospital)

# Evaluate model
precision_hospital = precision_score(y_test_hospital, y_pred_hospital)
recall_hospital = recall_score(y_test_hospital, y_pred_hospital)
f1_hospital = f1_score(y_test_hospital, y_pred_hospital)

print(f"Hospital Readmission - Precision: {precision_hospital}, Recall: {recall_hospital}, F1-Score: {f1_hospital}")

# 3. Classifying Digit Images Using Decision Trees
# Load dataset
mnist = fetch_openml('mnist_784', version=1)
X_mnist, y_mnist = mnist["data"], mnist["target"]

# Normalize pixel values
X_mnist = X_mnist / 255.0

# Create pipeline
pipeline_mnist = Pipeline(steps=[
    ('classifier', DecisionTreeClassifier())
])

# Split data
X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = train_test_split(X_mnist, y_mnist, test_size=0.2, random_state=42)

# Train model
pipeline_mnist.fit(X_train_mnist, y_train_mnist)

# Predict
y_pred_mnist = pipeline_mnist.predict(X_test_mnist)

# Evaluate model
accuracy_mnist = accuracy_score(y_test_mnist, y_pred_mnist)
confusion_matrix_mnist = confusion_matrix(y_test_mnist, y_pred_mnist)

print(f"Digit Images - Accuracy: {accuracy_mnist}")
print("Confusion Matrix:\n", confusion_matrix_mnist)

# 4. Predicting Loan Approval Using Logistic Regression
# Load dataset
loan_approval_df = pd.read_csv('path_to_loan_prediction_dataset.csv')

# Define features
categorical_features_loan_approval = ['gender', 'married_status']
numerical_features_loan_approval = ['loan_amount', 'applicant_income']

# Preprocess data
X_loan_approval, y_loan_approval, preprocessor_loan_approval = preprocess_data(
    loan_approval_df, 'loan_approved', categorical_features_loan_approval, numerical_features_loan_approval, 'mode'
)

# Create pipeline
pipeline_loan_approval = Pipeline(steps=[
    ('preprocessor', preprocessor_loan_approval),
    ('classifier', LogisticRegression())
])

# Split data
X_train_loan_approval, X_test_loan_approval, y_train_loan_approval, y_test_loan_approval = train_test_split(X_loan_approval, y_loan_approval, test_size=0.2, random_state=42)

# Train model
pipeline_loan_approval.fit(X_train_loan_approval, y_train_loan_approval)

# Predict
y_pred_loan_approval = pipeline_loan_approval.predict(X_test_loan_approval)

# Evaluate model
accuracy_loan_approval = accuracy_score(y_test_loan_approval, y_pred_loan_approval)
confusion_matrix_loan_approval = confusion_matrix(y_test_loan_approval, y_pred_loan_approval)

print(f"Loan Approval - Accuracy: {accuracy_loan_approval}")
print("Confusion Matrix:\n", confusion_matrix_loan_approval)

# 5. Classifying Wine Quality Using Decision Trees
# Load dataset
wine_quality_df = pd.read_csv('path_to_wine_quality_dataset.csv')

# Define features
numerical_features_wine = wine_quality_df.columns.drop('quality')
# Assuming binary classification of quality: 0 = bad, 1 = good
wine_quality_df['quality'] = wine_quality_df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Preprocess data
X_wine, y_wine, preprocessor_wine = preprocess_data(
    wine_quality_df, 'quality', [], numerical_features_wine, 'mean'
)

# Create pipeline
pipeline_wine = Pipeline(steps=[
    ('preprocessor', preprocessor_wine),
    ('classifier', DecisionTreeClassifier())
])

# Split data
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

# Train model
pipeline_wine.fit(X_train_wine, y_train_wine)

# Predict
y_pred_wine = pipeline_wine.predict(X_test_wine)

# Evaluate model
accuracy_wine = accuracy_score(y_test_wine, y_pred_wine)
roc_auc_wine = roc_auc_score(y_test_wine, y_pred_wine)

print(f"Wine Quality - Accuracy: {accuracy_wine}, ROC-AUC: {roc_auc_wine}")
