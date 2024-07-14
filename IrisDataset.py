import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Exercise 1: Print the first 5 rows
print("First 5 rows of the Iris dataset:")
print(df.head())

# Exercise 2: Implement a function that takes a dataset and returns the number of features and samples
def dataset_info(dataset):
    num_samples = dataset.shape[0]
    num_features = dataset.shape[1]
    return num_samples, num_features

samples, features = dataset_info(df)
print(f"\nNumber of samples: {samples}")
print(f"Number of features: {features}")

# Exercise 3: Split a dataset into training and testing sets with an 80/20 split
X_train, X_test, y_train, y_test = train_test_split(df, iris.target, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Exercise 4: Explore basic statistics of the dataset
mean = df.mean()
median = df.median()
std_dev = df.std()

print("\nBasic Statistics:")
print("Mean:\n", mean)
print("Median:\n", median)
print("Standard Deviation:\n", std_dev)

# Exercise 5: Visualize the distribution of one of the features using a histogram
plt.hist(df[df.columns[0]], bins=20, edgecolor='k')
plt.title('Distribution of ' + df.columns[0])
plt.xlabel(df.columns[0])
plt.ylabel('Frequency')
plt.show()

# Exercise 6: Create a list of 10 numbers and compute their mean
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mean_value = np.mean(numbers)
print("\nMean of the list of numbers:", mean_value)

# Exercise 7: Create a function that returns count, mean, median, and std deviation of a list
def calculate_statistics(numbers):
    stats = {
        'count': len(numbers),
        'mean': np.mean(numbers),
        'median': np.median(numbers),
        'std_dev': np.std(numbers)
    }
    return stats

numbers_stats = calculate_statistics(numbers)
print("\nStatistics of the list of numbers:", numbers_stats)

# Exercise 8: Generate a 5x5 matrix of random numbers and print it
random_matrix = np.random.rand(5, 5)
print("\n5x5 Matrix of random numbers:\n", random_matrix)

# Exercise 9: Load a CSV file into a Pandas DataFrame and print summary statistics for each column
# Note: You need to have a CSV file at the specified path for this to work
# Uncomment the following lines if you have a CSV file
# df_csv = pd.read_csv('path_to_csv_file.csv')
# print("\nSummary Statistics for CSV data:\n", df_csv.describe())

# Exercise 10: Implement a simple linear regression model and print the model coefficients
X = df
y = iris.target
model = LinearRegression()
model.fit(X, y)
print("\nLinear Regression model coefficients:\n", model.coef_)
