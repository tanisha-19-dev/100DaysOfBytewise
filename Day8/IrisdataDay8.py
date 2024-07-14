import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import ttest_ind, norm

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# Calculate mean, median, and mode of sepal lengths
sepal_lengths = iris_df['sepal length (cm)']
mean_sepal_length = sepal_lengths.mean()
median_sepal_length = sepal_lengths.median()
mode_sepal_length = sepal_lengths.mode()[0]

# Calculate variance and standard deviation of petal widths
petal_widths = iris_df['petal width (cm)']
variance_petal_width = petal_widths.var()
std_dev_petal_width = petal_widths.std()

# Create summary table for all numerical features
summary_stats = iris_df.describe().loc[['mean', '50%', 'std']]
summary_stats.rename(index={'50%': 'median', 'std': 'std_dev'}, inplace=True)
summary_stats.loc['variance'] = iris_df.var()

# Plot probability distribution of sepal lengths
plt.figure(figsize=(10, 6))
sepal_lengths.hist(bins=10, density=True, alpha=0.6, color='g')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Probability')
plt.title('Probability Distribution of Sepal Lengths')
plt.grid(True)
plt.show()

# Plot CDF for petal lengths
petal_lengths = iris_df['petal length (cm)']
sorted_petal_lengths = np.sort(petal_lengths)
cdf = np.arange(len(petal_lengths)) / float(len(petal_lengths))

plt.figure(figsize=(10, 6))
plt.plot(sorted_petal_lengths, cdf, marker='.', linestyle='none')
plt.xlabel('Petal Length (cm)')
plt.ylabel('CDF')
plt.title('Cumulative Distribution Function of Petal Lengths')
plt.grid(True)
plt.show()

# Plot PDF for sepal width
sepal_widths = iris_df['sepal width (cm)']
mean_sepal_width = sepal_widths.mean()
std_dev_sepal_width = sepal_widths.std()

x = np.linspace(sepal_widths.min(), sepal_widths.max(), 100)
pdf = norm.pdf(x, mean_sepal_width, std_dev_sepal_width)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, label='PDF')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Density')
plt.title('Probability Density Function of Sepal Width')
plt.grid(True)
plt.legend()
plt.show()

# Probability of a petal length greater than a given value (5.0 cm)
value = 5.0
probability = (petal_lengths > value).mean()

# Hypothesis test for mean petal length between two species
setosa = iris_df[iris_df['species'] == 0]['petal length (cm)']
versicolor = iris_df[iris_df['species'] == 1]['petal length (cm)']
t_stat, p_value = ttest_ind(setosa, versicolor)

# Covariance and correlation between sepal length and sepal width
sepal_length = iris_df['sepal length (cm)']
sepal_width = iris_df['sepal width (cm)']
covariance = np.cov(sepal_length, sepal_width)[0, 1]
correlation = np.corrcoef(sepal_length, sepal_width)[0, 1]

(mean_sepal_length, median_sepal_length, mode_sepal_length,
 variance_petal_width, std_dev_petal_width,
 summary_stats, probability, t_stat, p_value,
 covariance, correlation)
