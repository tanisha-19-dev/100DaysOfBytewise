import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, plot_roc_curve

# Load Titanic dataset
titanic_df = pd.read_csv('path_to_titanic_dataset.csv')

# Preprocessing function specific to Titanic dataset
def preprocess_titanic(df):
    # Handle missing values
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Define features and target
    X = df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
    y = df['Survived']
    
    # Define categorical and numerical features
    categorical_features = ['Sex', 'Embarked', 'Pclass']
    numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )
    
    return X, y, preprocessor

X_titanic, y_titanic, preprocessor_titanic = preprocess_titanic(titanic_df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_titanic, y_titanic, test_size=0.2, random_state=42)

# 1. Logistic Regression with Cross-Validation
logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_titanic),
    ('classifier', LogisticRegression())
])

# K-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(logreg_pipeline, X_train, y_train, cv=kfold, scoring='accuracy')

print(f"Logistic Regression Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()}")

# Single train-test split evaluation
logreg_pipeline.fit(X_train, y_train)
y_pred_logreg = logreg_pipeline.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

print(f"Logistic Regression Test Accuracy: {accuracy_logreg}")

# 2. Decision Tree Classifier - Analyzing Overfitting and Underfitting
train_accuracies = []
test_accuracies = []

for depth in range(1, 21):
    tree_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_titanic),
        ('classifier', DecisionTreeClassifier(max_depth=depth))
    ])
    
    tree_pipeline.fit(X_train, y_train)
    
    train_accuracies.append(tree_pipeline.score(X_train, y_train))
    test_accuracies.append(tree_pipeline.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), train_accuracies, label='Training Accuracy')
plt.plot(range(1, 21), test_accuracies, label='Validation Accuracy')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Decision Tree: Analyzing Overfitting and Underfitting')
plt.show()

# 3. Precision, Recall, and F1-Score for Logistic Regression
precision_logreg = precision_score(y_test, y_pred_logreg)
recall_logreg = recall_score(y_test, y_pred_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg)

print(f"Logistic Regression - Precision: {precision_logreg}, Recall: {recall_logreg}, F1-Score: {f1_logreg}")

# 4. ROC Curve for Decision Tree Classifier
tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_titanic),
    ('classifier', DecisionTreeClassifier())
])

tree_pipeline.fit(X_train, y_train)
y_pred_tree = tree_pipeline.predict(X_test)
y_pred_tree_proba = tree_pipeline.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_pred_tree_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

print(f"Decision Tree ROC AUC: {roc_auc}")

# 5. Comparing Model Performance with and without Cross-Validation
# Logistic Regression with cross-validation
cv_scores_logreg = cross_val_score(logreg_pipeline, X_train, y_train, cv=kfold, scoring='accuracy')
logreg_pipeline.fit(X_train, y_train)
y_pred_logreg = logreg_pipeline.predict(X_test)

accuracy_logreg_cv = cv_scores_logreg.mean()
precision_logreg_cv = precision_score(y_test, y_pred_logreg)
recall_logreg_cv = recall_score(y_test, y_pred_logreg)

print(f"Logistic Regression with CV - Accuracy: {accuracy_logreg_cv}, Precision: {precision_logreg_cv}, Recall: {recall_logreg_cv}")

# Decision Tree with cross-validation
tree_pipeline_cv = Pipeline(steps=[
    ('preprocessor', preprocessor_titanic),
    ('classifier', DecisionTreeClassifier())
])
cv_scores_tree = cross_val_score(tree_pipeline_cv, X_train, y_train, cv=kfold, scoring='accuracy')
tree_pipeline_cv.fit(X_train, y_train)
y_pred_tree_cv = tree_pipeline_cv.predict(X_test)

accuracy_tree_cv = cv_scores_tree.mean()
precision_tree_cv = precision_score(y_test, y_pred_tree_cv)
recall_tree_cv = recall_score(y_test, y_pred_tree_cv)

print(f"Decision Tree with CV - Accuracy: {accuracy_tree_cv}, Precision: {precision_tree_cv}, Recall: {recall_tree_cv}")
