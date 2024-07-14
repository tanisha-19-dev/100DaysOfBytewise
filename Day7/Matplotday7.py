Sure! Below is a combined script that includes all the exercises using Matplotlib and Seaborn:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Line Plot with Matplotlib
values = [10, 15, 13, 17, 14, 19, 21, 18]
time = list(range(1, len(values) + 1))
plt.figure(figsize=(10, 6))
plt.plot(time, values, marker='o')
plt.title('Trend of Values Over Time')
plt.xlabel('Time')
plt.ylabel('Values')
plt.grid(True)
plt.show()

# 2. Bar Chart with Matplotlib
categories = ['A', 'B', 'C', 'D', 'E']
frequencies = [5, 7, 2, 4, 6]
plt.figure(figsize=(10, 6))
plt.bar(categories, frequencies, color='skyblue')
plt.title('Frequency of Different Categories')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.show()

# 3. Scatter Plot with Matplotlib
x = [1, 2, 3, 4, 5]
y = [10, 14, 12, 15, 17]
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red')
plt.title('Scatter Plot of Two Variables')
plt.xlabel('X Variable')
plt.ylabel('Y Variable')
plt.grid(True)
plt.show()

# 4. Pairplot with Seaborn
data = sns.load_dataset('iris')
sns.pairplot(data)
plt.show()

# 5. Box Plot with Seaborn
data = sns.load_dataset('tips')
plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='total_bill', data=data)
plt.title('Box Plot of Total Bill by Day')
plt.xlabel('Day')
plt.ylabel('Total Bill')
plt.show()

# 6. Heatmap with Seaborn
correlation_matrix = data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# 7. Subplot Grid with Matplotlib
y1 = [10, 14, 12, 15, 17]
y2 = [5, 9, 7, 10, 12]
y3 = [3, 6, 4, 7, 9]
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].plot(x, y1, marker='o')
axs[0, 0].set_title('Line Plot')
axs[0, 1].bar(x, y2, color='skyblue')
axs[0, 1].set_title('Bar Chart')
axs[1, 0].scatter(x, y3, color='red')
axs[1, 0].set_title('Scatter Plot')
axs[1, 1].axis('off')
plt.tight_layout()
plt.show()

# 8. Customizing Seaborn Plot
plt.figure(figsize=(10, 6))
sns.set_palette('pastel')
sns.boxplot(x='day', y='total_bill', data=data)
plt.title('Total Bill by Day')
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill')
plt.show()

# 9. Violin Plot with Seaborn
plt.figure(figsize=(10, 6))
sns.violinplot(x='day', y='total_bill', data=data)
plt.title('Violin Plot of Total Bill by Day')
plt.xlabel('Day')
plt.ylabel('Total Bill')
plt.show()

# 10. Combining Matplotlib and Seaborn
plt.figure(figsize=(10, 6))
sns.histplot(data['total_bill'], kde=True, color='skyblue')
plt.title('Histogram and KDE of Total Bill')
plt.xlabel('Total Bill')
plt.ylabel('Frequency')
plt.show()
```

This combined script covers all the required visualizations using Matplotlib and Seaborn, and it will produce the plots sequentially.