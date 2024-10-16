## Exploratory Data Analysis with Python
Slide 1: Introduction to Exploratory Data Analysis (EDA)

Exploratory Data Analysis is a crucial step in the data science process, allowing us to understand the structure, patterns, and characteristics of our dataset before formal modeling. EDA helps us identify trends, detect outliers, and formulate hypotheses about the data.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load a sample dataset
df = pd.read_csv('sample_data.csv')

# Display basic information about the dataset
print(df.info())

# Show the first few rows
print(df.head())
```

Slide 2: Loading and Inspecting Data

The first step in EDA is loading the data and getting a quick overview. We'll use pandas to load a CSV file and display basic information about the dataset.

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('iris.csv')

# Display basic information
print(df.info())

# Show the first few rows
print(df.head())

# Display summary statistics
print(df.describe())
```

Slide 3: Handling Missing Values

Identifying and handling missing values is crucial in EDA. We'll explore ways to detect and visualize missing data.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load a dataset with missing values
df = pd.read_csv('dataset_with_missing_values.csv')

# Calculate percentage of missing values
missing_percentage = df.isnull().mean() * 100

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.show()

print("Percentage of missing values:\n", missing_percentage)
```

Slide 4: Data Distribution Analysis

Understanding the distribution of variables is essential. We'll use histograms and kernel density plots to visualize distributions.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('iris.csv')

# Create histograms for numerical columns
df.hist(figsize=(12, 8))
plt.suptitle('Histograms of Numerical Variables')
plt.tight_layout()
plt.show()

# Create kernel density plots
plt.figure(figsize=(12, 8))
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    sns.kdeplot(data=df[column], shade=True, label=column)
plt.title('Kernel Density Plots of Numerical Variables')
plt.legend()
plt.show()
```

Slide 5: Correlation Analysis

Exploring relationships between variables is crucial. We'll use correlation matrices and heatmaps to visualize these relationships.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('iris.csv')

# Calculate correlation matrix
corr_matrix = df.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap')
plt.show()

# Pairplot for visualizing relationships
sns.pairplot(df, hue='species')
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()
```

Slide 6: Outlier Detection

Identifying outliers is important for understanding data quality and potential anomalies. We'll use box plots and scatter plots for outlier detection.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('iris.csv')

# Create box plots
plt.figure(figsize=(12, 6))
df.boxplot()
plt.title('Box Plots for Numerical Variables')
plt.show()

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['sepal_length'], df['sepal_width'])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Scatter Plot: Sepal Length vs Sepal Width')
plt.show()
```

Slide 7: Categorical Data Analysis

Analyzing categorical variables is essential for understanding the composition of our dataset. We'll use bar plots and pie charts for this purpose.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('iris.csv')

# Create a bar plot
plt.figure(figsize=(10, 6))
df['species'].value_counts().plot(kind='bar')
plt.title('Distribution of Iris Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()

# Create a pie chart
plt.figure(figsize=(8, 8))
df['species'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Iris Species')
plt.ylabel('')
plt.show()
```

Slide 8: Time Series Analysis

For time-series data, we need to analyze trends, seasonality, and patterns over time. We'll use line plots and rolling statistics for this purpose.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load a time series dataset
df = pd.read_csv('time_series_data.csv', parse_dates=['date'], index_col='date')

# Plot the time series
plt.figure(figsize=(12, 6))
df['value'].plot()
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Calculate and plot rolling mean and standard deviation
rolling_mean = df['value'].rolling(window=30).mean()
rolling_std = df['value'].rolling(window=30).std()

plt.figure(figsize=(12, 6))
df['value'].plot(label='Original')
rolling_mean.plot(label='Rolling Mean', color='red')
rolling_std.plot(label='Rolling Std', color='green')
plt.title('Time Series with Rolling Statistics')
plt.legend()
plt.show()
```

Slide 9: Feature Engineering

Feature engineering is the process of creating new features from existing ones. We'll demonstrate how to create interaction terms and polynomial features.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
df = pd.read_csv('iris.csv')

# Create interaction terms
df['sepal_area'] = df['sepal_length'] * df['sepal_width']
df['petal_area'] = df['petal_length'] * df['petal_width']

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['sepal_length', 'sepal_width']])
poly_features_df = pd.DataFrame(poly_features, columns=poly.get_feature_names(['sepal_length', 'sepal_width']))

# Combine original and new features
df_engineered = pd.concat([df, poly_features_df], axis=1)

print(df_engineered.head())
```

Slide 10: Dimensionality Reduction

When dealing with high-dimensional data, dimensionality reduction techniques like PCA can be useful for visualization and feature selection.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('iris.csv')

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('species', axis=1))

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

# Plot the results
plt.figure(figsize=(10, 8))
for species in df['species'].unique():
    mask = df['species'] == species
    plt.scatter(pca_result[mask, 0], pca_result[mask, 1], label=species)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.title('PCA of Iris Dataset')
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Slide 11: Statistical Tests

Performing statistical tests can help us understand the significance of our findings. We'll demonstrate how to perform a t-test and chi-square test.

```python
import pandas as pd
import numpy as np
from scipy import stats

# Load the dataset
df = pd.read_csv('iris.csv')

# Perform t-test
setosa = df[df['species'] == 'setosa']['sepal_length']
versicolor = df[df['species'] == 'versicolor']['sepal_length']
t_stat, p_value = stats.ttest_ind(setosa, versicolor)

print("T-test results:")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Perform chi-square test
observed = pd.crosstab(df['species'], df['sepal_length'] > df['sepal_length'].mean())
chi2, p_value, dof, expected = stats.chi2_contingency(observed)

print("\nChi-square test results:")
print(f"Chi-square statistic: {chi2}")
print(f"P-value: {p_value}")
```

Slide 12: Data Visualization with Seaborn

Seaborn is a powerful library for statistical data visualization. We'll use it to create more advanced plots for our EDA.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('iris.csv')

# Create a pair plot
sns.pairplot(df, hue='species', height=2.5)
plt.suptitle('Pair Plot of Iris Dataset', y=1.02)
plt.show()

# Create a violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='species', y='sepal_length', data=df)
plt.title('Violin Plot of Sepal Length by Species')
plt.show()

# Create a joint plot
sns.jointplot(x='sepal_length', y='sepal_width', data=df, kind='kde', hue='species')
plt.suptitle('Joint Plot of Sepal Length vs Sepal Width', y=1.02)
plt.show()
```

Slide 13: Real-Life Example: Weather Data Analysis

Let's analyze a weather dataset to demonstrate EDA in a real-world scenario. We'll explore temperature trends and precipitation patterns.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load weather data
weather_df = pd.read_csv('weather_data.csv', parse_dates=['date'])

# Plot temperature trend
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='temperature', data=weather_df)
plt.title('Temperature Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.show()

# Analyze precipitation patterns
plt.figure(figsize=(12, 6))
sns.boxplot(x=weather_df['date'].dt.month, y='precipitation', data=weather_df)
plt.title('Monthly Precipitation Distribution')
plt.xlabel('Month')
plt.ylabel('Precipitation (mm)')
plt.show()

# Correlation between temperature and precipitation
plt.figure(figsize=(10, 8))
sns.scatterplot(x='temperature', y='precipitation', data=weather_df)
plt.title('Temperature vs Precipitation')
plt.xlabel('Temperature (°C)')
plt.ylabel('Precipitation (mm)')
plt.show()
```

Slide 14: Real-Life Example: Customer Churn Analysis

In this example, we'll explore a dataset related to customer churn in a telecommunications company, demonstrating how EDA can provide insights into business problems.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load customer churn data
churn_df = pd.read_csv('customer_churn_data.csv')

# Visualize churn distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Churn', data=churn_df)
plt.title('Distribution of Customer Churn')
plt.show()

# Analyze relationship between tenure and churn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='tenure', data=churn_df)
plt.title('Customer Tenure by Churn Status')
plt.show()

# Explore correlation between numerical features
correlation = churn_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
```

Slide 15: Additional Resources

For further exploration of Exploratory Data Analysis techniques and best practices, consider the following resources:

1. "Exploratory Data Analysis" by John W. Tukey (1977) - The foundational book on EDA.
2. "Python for Data Analysis" by Wes McKinney - Covers pandas and numpy for data manipulation and analysis.
3. "Data Science from Scratch" by Joel Grus - Provides a comprehensive introduction to data science concepts.
4. "Statistical Rethinking" by Richard McElreath - Offers a Bayesian perspective on data analysis.
5. ArXiv paper: "A Survey on Exploratory Data Analysis and Visualization Techniques for Big Data" ([https://arxiv.org/abs/2005.02218](https://arxiv.org/abs/2005.02218)) - Discusses modern EDA techniques for large datasets.

Remember to adapt these techniques to your specific dataset and research questions. EDA is an iterative process, and the insights gained often lead to further questions and analyses.

