import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Function to preprocess and train logistic regression for diabetes dataset
def diabetes_logistic_regression(data_path):
    diabetes_data = pd.read_csv(data_path)
    X = diabetes_data.drop('Outcome', axis=1)
    y = diabetes_data['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print('Diabetes Prediction - Logistic Regression')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print('')

# Function to preprocess and train decision tree for iris dataset
def iris_decision_tree(data_path):
    iris_data = pd.read_csv(data_path)
    X = iris_data.drop('species', axis=1)
    y = iris_data['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X_train, y_train)
    y_pred = tree_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Iris Classification - Decision Tree')
    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    print('')

# Function to preprocess and train logistic regression for titanic dataset
def titanic_logistic_regression(data_path):
    titanic_data = pd.read_csv(data_path)
    titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
    titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
    X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = titanic_data['Survived']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Age', 'SibSp', 'Parch', 'Fare']),
            ('cat', OneHotEncoder(), ['Pclass', 'Sex', 'Embarked'])
        ])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print('Titanic Survival Prediction - Logistic Regression')
    print(f'ROC-AUC: {roc_auc}')
    print('')

# Function to preprocess and train decision tree for spam email dataset
def spam_decision_tree(data_path):
    spam_data = pd.read_csv(data_path)
    X = spam_data.drop('Label', axis=1)
    y = spam_data['Label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X_train, y_train)
    y_pred = tree_clf.predict(X_test)
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    print('Spam Email Classification - Decision Tree')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')
    print('')

# Function to preprocess and train logistic regression for customer satisfaction dataset
def customer_satisfaction_logistic_regression(data_path):
    customer_data = pd.read_csv(data_path)
    customer_data.fillna(customer_data.median(), inplace=True)
    X = customer_data.drop('Satisfaction', axis=1)
    y = customer_data['Satisfaction']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Age', 'Income', 'SpendingScore']),
            ('cat', OneHotEncoder(), ['Region', 'Gender'])
        ])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Customer Satisfaction Prediction - Logistic Regression')
    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    print('')

# Main function to run all tasks
def main():
    diabetes_logistic_regression('path_to_diabetes_dataset.csv')
    iris_decision_tree('path_to_iris_dataset.csv')
    titanic_logistic_regression('path_to_titanic_dataset.csv')
    spam_decision_tree('path_to_spam_email_dataset.csv')
    customer_satisfaction_logistic_regression('path_to_customer_satisfaction_dataset.csv')

if __name__ == "__main__":
    main()
