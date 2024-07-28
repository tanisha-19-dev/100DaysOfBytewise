import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix

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

# 1. Predicting Employee Attrition Using Logistic Regression
# Load dataset
employee_df = pd.read_csv('path_to_employee_attrition_dataset.csv')

# Define features
categorical_features_employee = ['department', 'gender']
numerical_features_employee = ['age', 'salary', 'years_at_company']

# Preprocess data
X_employee, y_employee, preprocessor_employee = preprocess_data(
    employee_df, 'attrition', categorical_features_employee, numerical_features_employee, 'mean'
)

# Create pipeline
pipeline_employee = Pipeline(steps=[
    ('preprocessor', preprocessor_employee),
    ('classifier', LogisticRegression())
])

# Split data
X_train_employee, X_test_employee, y_train_employee, y_test_employee = train_test_split(X_employee, y_employee, test_size=0.2, random_state=42)

# Train model
pipeline_employee.fit(X_train_employee, y_train_employee)

# Predict
y_pred_employee = pipeline_employee.predict(X_test_employee)

# Evaluate model
precision_employee = precision_score(y_test_employee, y_pred_employee)
recall_employee = recall_score(y_test_employee, y_pred_employee)
f1_employee = f1_score(y_test_employee, y_pred_employee)

print(f"Employee Attrition - Precision: {precision_employee}, Recall: {recall_employee}, F1-Score: {f1_employee}")

# 2. Classifying Credit Card Fraud Using Decision Trees
# Load dataset
credit_card_df = pd.read_csv('path_to_credit_card_fraud_dataset.csv')

# Define features
numerical_features_credit = credit_card_df.columns.drop('fraudulent')

# Preprocess data
X_credit, y_credit, preprocessor_credit = preprocess_data(
    credit_card_df, 'fraudulent', [], numerical_features_credit, 'mean'
)

# Create pipeline
pipeline_credit = Pipeline(steps=[
    ('preprocessor', preprocessor_credit),
    ('classifier', DecisionTreeClassifier())
])

# Split data
X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)

# Train model
pipeline_credit.fit(X_train_credit, y_train_credit)

# Predict
y_pred_credit = pipeline_credit.predict(X_test_credit)

# Evaluate model
roc_auc_credit = roc_auc_score(y_test_credit, y_pred_credit)
confusion_matrix_credit = confusion_matrix(y_test_credit, y_pred_credit)

print(f"Credit Card Fraud - ROC-AUC: {roc_auc_credit}")
print("Confusion Matrix:\n", confusion_matrix_credit)

# 3. Predicting Heart Disease Using Logistic Regression
# Load dataset
heart_disease_df = pd.read_csv('path_to_heart_disease_dataset.csv')

# Define features
categorical_features_heart = ['gender', 'chest_pain_type']
numerical_features_heart = ['age', 'trestbps', 'chol', 'thalach']

# Preprocess data
X_heart, y_heart, preprocessor_heart = preprocess_data(
    heart_disease_df, 'target', categorical_features_heart, numerical_features_heart, 'mean'
)

# Create pipeline
pipeline_heart = Pipeline(steps=[
    ('preprocessor', preprocessor_heart),
    ('classifier', LogisticRegression())
])

# Split data
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

# Train model
pipeline_heart.fit(X_train_heart, y_train_heart)

# Predict
y_pred_heart = pipeline_heart.predict(X_test_heart)

# Evaluate model
accuracy_heart = accuracy_score(y_test_heart, y_pred_heart)
roc_auc_heart = roc_auc_score(y_test_heart, y_pred_heart)

print(f"Heart Disease - Accuracy: {accuracy_heart}, ROC-AUC: {roc_auc_heart}")

# 4. Classifying Emails as Spam Using Decision Trees
# Load dataset
spam_email_df = pd.read_csv('path_to_spam_email_dataset.csv')

# Define features
numerical_features_spam = spam_email_df.columns.drop('spam')

# Preprocess data
X_spam, y_spam, preprocessor_spam = preprocess_data(
    spam_email_df, 'spam', [], numerical_features_spam, 'mean'
)

# Create pipeline
pipeline_spam = Pipeline(steps=[
    ('preprocessor', preprocessor_spam),
    ('classifier', DecisionTreeClassifier())
])

# Split data
X_train_spam, X_test_spam, y_train_spam, y_test_spam = train_test_split(X_spam, y_spam, test_size=0.2, random_state=42)

# Train model
pipeline_spam.fit(X_train_spam, y_train_spam)

# Predict
y_pred_spam = pipeline_spam.predict(X_test_spam)

# Evaluate model
precision_spam = precision_score(y_test_spam, y_pred_spam)
recall_spam = recall_score(y_test_spam, y_pred_spam)
f1_spam = f1_score(y_test_spam, y_pred_spam)

print(f"Spam Email - Precision: {precision_spam}, Recall: {recall_spam}, F1-Score: {f1_spam}")

# 5. Predicting Customer Satisfaction Using Logistic Regression
# Load dataset
customer_satisfaction_df = pd.read_csv('path_to_customer_satisfaction_dataset.csv')

# Define features
categorical_features_customer = ['region']
numerical_features_customer = ['age', 'income', 'satisfaction_score']

# Preprocess data
X_customer, y_customer, preprocessor_customer = preprocess_data(
    customer_satisfaction_df, 'satisfied', categorical_features_customer, numerical_features_customer, 'median'
)

# Create pipeline
pipeline_customer = Pipeline(steps=[
    ('preprocessor', preprocessor_customer),
    ('classifier', LogisticRegression())
])

# Split data
X_train_customer, X_test_customer, y_train_customer, y_test_customer = train_test_split(X_customer, y_customer, test_size=0.2, random_state=42)

# Train model
pipeline_customer.fit(X_train_customer, y_train_customer)

# Predict
y_pred_customer = pipeline_customer.predict(X_test_customer)

# Evaluate model
accuracy_customer = accuracy_score(y_test_customer, y_pred_customer)
confusion_matrix_customer = confusion_matrix(y_test_customer, y_pred_customer)

print(f"Customer Satisfaction - Accuracy: {accuracy_customer}")
print("Confusion Matrix:\n", confusion_matrix_customer)
