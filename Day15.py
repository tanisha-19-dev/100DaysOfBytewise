import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, accuracy_score, plot_roc_curve

# Load Adult Income dataset
adult_df = pd.read_csv('path_to_adult_income_dataset.csv')

# Preprocessing function specific to Adult Income dataset
def preprocess_adult(df):
    # Handle missing values (replace '?' with NaN and drop them)
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Define features and target
    X = df.drop(columns=['income'])
    y = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    
    # Define categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )
    
    return X, y, preprocessor

X_adult, y_adult, preprocessor_adult = preprocess_adult(adult_df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_adult, y_adult, test_size=0.2, random_state=42)

# 1. Applying Cross-Validation to Random Forest Classifier
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_adult),
    ('classifier', RandomForestClassifier(random_state=42))
])

# K-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores_rf = cross_val_score(rf_pipeline, X_train, y_train, cv=kfold, scoring='accuracy')

print(f"Random Forest Cross-Validation Scores: {cv_scores_rf}")
print(f"Mean CV Accuracy: {cv_scores_rf.mean()}")

# 2. Investigating Overfitting and Underfitting in Gradient Boosting Machines
train_accuracies_gb = []
test_accuracies_gb = []

for n_estimators in [50, 100, 150, 200]:
    for learning_rate in [0.01, 0.05, 0.1, 0.2]:
        gb_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor_adult),
            ('classifier', GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42))
        ])
        
        gb_pipeline.fit(X_train, y_train)
        
        train_accuracies_gb.append(gb_pipeline.score(X_train, y_train))
        test_accuracies_gb.append(gb_pipeline.score(X_test, y_test))

# Plotting the results
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(train_accuracies_gb, label='Training Accuracy')
ax[1].plot(test_accuracies_gb, label='Validation Accuracy')

ax[0].set_title('Gradient Boosting: Training Accuracy')
ax[1].set_title('Gradient Boosting: Validation Accuracy')
ax[0].legend()
ax[1].legend()
plt.show()

# 3. Evaluating Precision, Recall, and F1-Score for Random Forests
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"Random Forest - Precision: {precision_rf}, Recall: {recall_rf}, F1-Score: {f1_rf}")

# 4. ROC Curve and AUC for Gradient Boosting Classifier
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_adult),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

gb_pipeline.fit(X_train, y_train)
y_pred_gb_proba = gb_pipeline.predict_proba(X_test)[:, 1]

fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_gb_proba)
roc_auc_gb = auc(fpr_gb, tpr_gb)

plt.figure(figsize=(10, 6))
plt.plot(fpr_gb, tpr_gb, label=f'ROC curve (area = {roc_auc_gb:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Gradient Boosting')
plt.legend(loc='lower right')
plt.show()

print(f"Gradient Boosting ROC AUC: {roc_auc_gb}")

# 5. Model Performance Comparison with Different Metrics
# SVM Classifier
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_adult),
    ('classifier', SVC(probability=True, random_state=42))
])

# Random Forest Classifier
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_adult),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Gradient Boosting Classifier
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_adult),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Evaluate models
models = {'SVM': svm_pipeline, 'Random Forest': rf_pipeline, 'Gradient Boosting': gb_pipeline}
metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': []}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if name != 'SVM' else model.predict_proba(X_test)[:, 1]
    
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1_score'].append(f1_score(y_test, y_pred))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    metrics['roc_auc'].append(auc(fpr, tpr))

for metric, values in metrics.items():
    print(f"{metric.capitalize()}:")
    for name, value in zip(models.keys(), values):
        print(f"{name}: {value:.4f}")
    print()
