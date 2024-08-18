import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from scipy.stats import zscore, boxcox
from sklearn.impute import KNNImputer
import miceforest as mf
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
import numpy as np

# 1. Handling Missing Data in Titanic Dataset
titanic = pd.read_csv('titanic.csv')

# Mean/Median Imputation
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)

# Mode Imputation
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)

# Dropping Rows
titanic.dropna(subset=['Cabin'], inplace=True)

# Dropping Columns
titanic.drop(columns=['Cabin'], inplace=True)

# 2. Encoding Categorical Variables in Car Evaluation Dataset
car_eval = pd.read_csv('car_evaluation.csv')

# Label Encoding
le = LabelEncoder()
car_eval['buying'] = le.fit_transform(car_eval['buying'])

# One-Hot Encoding
car_eval = pd.get_dummies(car_eval, columns=['buying'])

# 3. Scaling Features in the Wine Quality Dataset
wine = pd.read_csv('wine_quality.csv')

# Standardization
scaler = StandardScaler()
wine_standardized = scaler.fit_transform(wine)

# Normalization
normalizer = MinMaxScaler()
wine_normalized = normalizer.fit_transform(wine)

# 4. Handling Outliers in the Boston Housing Dataset
boston = pd.read_csv('boston_housing.csv')

# Z-score method
z_scores = np.abs(zscore(boston))
boston_no_outliers = boston[(z_scores < 3).all(axis=1)]

# IQR method
Q1 = boston.quantile(0.25)
Q3 = boston.quantile(0.75)
IQR = Q3 - Q1
boston_no_outliers_iqr = boston[~((boston < (Q1 - 1.5 * IQR)) | (boston > (Q3 + 1.5 * IQR))).any(axis=1)]

# 5. Data Imputation in the Retail Sales Dataset
retail_sales = pd.read_csv('retail_sales.csv')

# KNN Imputation
knn_imputer = KNNImputer(n_neighbors=5)
retail_sales_knn_imputed = knn_imputer.fit_transform(retail_sales)

# MICE Imputation
kernel = mf.ImputationKernel(retail_sales, datasets=5, save_all_iterations=True, random_state=1991)
kernel.mice(5)
retail_sales_mice_imputed = kernel.complete_data()

# 6. Feature Engineering in the Heart Disease Dataset
heart = pd.read_csv('heart_disease.csv')

# Create age groups
heart['age_group'] = pd.cut(heart['age'], bins=[29, 40, 50, 60, 70], labels=['30-40', '40-50', '50-60', '60-70'])

# Create cholesterol levels
heart['cholesterol_level'] = pd.cut(heart['chol'], bins=[0, 200, 240, 600], labels=['Normal', 'Borderline', 'High'])

# 7. Transforming Variables in the Bike Sharing Dataset
bike_sharing = pd.read_csv('bike_sharing.csv')

# Log transformation
bike_sharing['log_count'] = np.log(bike_sharing['count'] + 1)

# Square root transformation
bike_sharing['sqrt_count'] = np.sqrt(bike_sharing['count'])

# Box-Cox transformation
bike_sharing['boxcox_count'], _ = boxcox(bike_sharing['count'] + 1)

# 8. Feature Selection in the Diabetes Dataset
diabetes = pd.read_csv('diabetes.csv')

# Correlation analysis
correlation = diabetes.corr()

# Mutual Information
X = diabetes.drop('outcome', axis=1)
y = diabetes['outcome']
mutual_info = mutual_info_classif(X, y)

# Recursive Feature Elimination (RFE)
model = LogisticRegression()
rfe = RFE(model, 5)
fit = rfe.fit(X, y)

# 9. Dealing with Imbalanced Data in the Credit Card Fraud Detection Dataset
credit_card = pd.read_csv('credit_card_fraud.csv')
X = credit_card.drop('Class', axis=1)
y = credit_card['Class']

# SMOTE
smote = SMOTE()
X_res_smote, y_res_smote = smote.fit_resample(X, y)

# ADASYN
adasyn = ADASYN()
X_res_adasyn, y_res_adasyn = adasyn.fit_resample(X, y)

# Undersampling
undersample = RandomUnderSampler()
X_res_under, y_res_under = undersample.fit_resample(X, y)

# 10. Combining Multiple Datasets in the Movie Lens Dataset
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
users = pd.read_csv('users.csv')

# Merge datasets
merged_df = ratings.merge(movies, on='movieId').merge(users, on='userId')
