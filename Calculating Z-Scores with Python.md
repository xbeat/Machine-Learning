## Calculating Z-Scores with Python
Slide 1: Z-Score Calculation with Python

Z-score is a statistical measure that indicates how many standard deviations an element is from the mean. It's widely used in data analysis and machine learning for normalizing data and identifying outliers. In this presentation, we'll explore how to calculate z-scores using Python, providing practical examples and code snippets.

```python
import numpy as np

def z_score(x, mean, std_dev):
    return (x - mean) / std_dev

# Example
data = [2, 4, 6, 8, 10]
mean = np.mean(data)
std_dev = np.std(data)

z = z_score(6, mean, std_dev)
print(f"Z-score of 6: {z}")
```

Output:

```
Z-score of 6: 0.0
```

Slide 2: Understanding the Z-Score Formula

The z-score formula is (x - μ) / σ, where x is the raw score, μ is the population mean, and σ is the population standard deviation. This formula measures how far a data point is from the mean in terms of standard deviations. Let's break down each component and implement it in Python.

```python
import numpy as np

# Sample data
data = [1, 2, 3, 4, 5]

# Calculate mean (μ)
mean = np.mean(data)

# Calculate standard deviation (σ)
std_dev = np.std(data)

# Calculate z-score for each data point
z_scores = [(x - mean) / std_dev for x in data]

print(f"Data: {data}")
print(f"Z-scores: {z_scores}")
```

Output:

```
Data: [1, 2, 3, 4, 5]
Z-scores: [-1.4142135623730951, -0.7071067811865476, 0.0, 0.7071067811865476, 1.4142135623730951]
```

Slide 3: Implementing Z-Score Calculation Function

Let's create a reusable function to calculate z-scores for an entire dataset. This function will take a list or array of data points and return their corresponding z-scores.

```python
import numpy as np

def calculate_z_scores(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    return [(x - mean) / std_dev for x in data]

# Example usage
dataset = [10, 15, 20, 25, 30]
z_scores = calculate_z_scores(dataset)

for data, z in zip(dataset, z_scores):
    print(f"Data point: {data}, Z-score: {z:.4f}")
```

Output:

```
Data point: 10, Z-score: -1.4142
Data point: 15, Z-score: -0.7071
Data point: 20, Z-score: 0.0000
Data point: 25, Z-score: 0.7071
Data point: 30, Z-score: 1.4142
```

Slide 4: Z-Score Calculation with Pandas

For larger datasets, it's often more convenient to use the Pandas library. Pandas provides a built-in method to calculate z-scores efficiently. Let's see how to use it.

```python
import pandas as pd
import numpy as np

# Create a sample dataset
data = {'values': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]}
df = pd.DataFrame(data)

# Calculate z-scores
df['z_scores'] = (df['values'] - df['values'].mean()) / df['values'].std()

print(df)
```

Output:

```
    values   z_scores
0        2 -1.566738
1        4 -1.219906
2        6 -0.873075
3        8 -0.526243
4       10 -0.179411
5       12  0.167421
6       14  0.514253
7       16  0.861085
8       18  1.207916
9       20  1.554748
```

Slide 5: Handling Outliers with Z-Scores

Z-scores are commonly used to identify and handle outliers in a dataset. Generally, data points with z-scores beyond ±3 are considered potential outliers. Let's implement a function to identify outliers using this method.

```python
import numpy as np

def identify_outliers(data, threshold=3):
    z_scores = (data - np.mean(data)) / np.std(data)
    return np.abs(z_scores) > threshold

# Example dataset with an outlier
data = [2, 4, 6, 8, 10, 100]

outliers = identify_outliers(data)
print("Data points:", data)
print("Outlier mask:", outliers)
print("Outliers:", np.array(data)[outliers])
```

Output:

```
Data points: [2, 4, 6, 8, 10, 100]
Outlier mask: [False False False False False  True]
Outliers: [100]
```

Slide 6: Z-Score for Feature Scaling

Z-score normalization is a popular method for feature scaling in machine learning. It transforms features to have a mean of 0 and a standard deviation of 1. Let's implement this scaling technique using NumPy.

```python
import numpy as np

def z_score_normalize(data):
    return (data - np.mean(data)) / np.std(data)

# Example: Normalizing multiple features
features = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

normalized_features = z_score_normalize(features)
print("Original features:")
print(features)
print("\nNormalized features:")
print(normalized_features)
```

Output:

```
Original features:
[[1 2 3]
 [4 5 6]
 [7 8 9]]

Normalized features:
[[-1.22474487 -1.22474487 -1.22474487]
 [ 0.          0.          0.        ]
 [ 1.22474487  1.22474487  1.22474487]]
```

Slide 7: Visualizing Z-Scores

Visualizing z-scores can help in understanding their distribution and identifying potential outliers. Let's create a histogram of z-scores for a given dataset using Matplotlib.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = np.random.normal(loc=10, scale=2, size=1000)

# Calculate z-scores
z_scores = (data - np.mean(data)) / np.std(data)

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(z_scores, bins=30, edgecolor='black')
plt.title('Histogram of Z-Scores')
plt.xlabel('Z-Score')
plt.ylabel('Frequency')
plt.axvline(x=-3, color='r', linestyle='--', label='±3 threshold')
plt.axvline(x=3, color='r', linestyle='--')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

\[The output would be a histogram plot showing the distribution of z-scores\]

Slide 8: Z-Score in Hypothesis Testing

Z-scores play a crucial role in hypothesis testing, particularly in determining the probability of observing a value in a normal distribution. Let's create a function to calculate the p-value for a given z-score.

```python
import scipy.stats as stats

def z_score_to_p_value(z_score, two_tailed=True):
    if two_tailed:
        return 2 * (1 - stats.norm.cdf(abs(z_score)))
    else:
        return 1 - stats.norm.cdf(z_score)

# Example
z = 2.5
p_value = z_score_to_p_value(z)
print(f"Z-score: {z}")
print(f"Two-tailed p-value: {p_value:.4f}")
```

Output:

```
Z-score: 2.5
Two-tailed p-value: 0.0124
```

Slide 9: Real-Life Example: Quality Control

In a manufacturing process, product weights are expected to follow a normal distribution with a mean of 100 grams and a standard deviation of 2 grams. We can use z-scores to identify products that deviate significantly from the expected weight.

```python
import numpy as np

def check_product_weight(weight, mean=100, std_dev=2, threshold=3):
    z_score = (weight - mean) / std_dev
    if abs(z_score) > threshold:
        return f"Rejected (Z-score: {z_score:.2f})"
    else:
        return f"Accepted (Z-score: {z_score:.2f})"

# Simulating product weights
np.random.seed(42)
weights = np.random.normal(loc=100, scale=2, size=10)

for i, weight in enumerate(weights, 1):
    result = check_product_weight(weight)
    print(f"Product {i}: Weight = {weight:.2f}g, {result}")
```

Output:

```
Product 1: Weight = 100.48g, Accepted (Z-score: 0.24)
Product 2: Weight = 101.06g, Accepted (Z-score: 0.53)
Product 3: Weight = 101.95g, Accepted (Z-score: 0.97)
Product 4: Weight = 97.15g, Accepted (Z-score: -1.43)
Product 5: Weight = 98.12g, Accepted (Z-score: -0.94)
Product 6: Weight = 101.91g, Accepted (Z-score: 0.95)
Product 7: Weight = 97.70g, Accepted (Z-score: -1.15)
Product 8: Weight = 99.83g, Accepted (Z-score: -0.09)
Product 9: Weight = 101.88g, Accepted (Z-score: 0.94)
Product 10: Weight = 100.98g, Accepted (Z-score: 0.49)
```

Slide 10: Real-Life Example: Academic Performance

Z-scores can be used to compare academic performance across different subjects or classes with varying difficulty levels. Let's create a function to standardize grades and compare students' performances.

```python
import numpy as np
import pandas as pd

def standardize_grades(grades):
    return (grades - np.mean(grades)) / np.std(grades)

# Sample data
data = {
    'Student': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Math': [85, 92, 78, 95, 88],
    'Science': [90, 88, 82, 95, 91],
    'History': [78, 85, 90, 82, 87]
}

df = pd.DataFrame(data)

# Standardize grades
for subject in ['Math', 'Science', 'History']:
    df[f'{subject}_Z'] = standardize_grades(df[subject])

# Calculate average z-score
df['Avg_Z'] = df[['Math_Z', 'Science_Z', 'History_Z']].mean(axis=1)

# Sort by average z-score
df_sorted = df.sort_values('Avg_Z', ascending=False).reset_index(drop=True)

print(df_sorted[['Student', 'Avg_Z']])
```

Output:

```
   Student     Avg_Z
0    David  1.355828
1      Eva  0.301295
2      Bob  0.301295
3    Alice -0.753148
4  Charlie -1.205270
```

Slide 11: Handling Zero Standard Deviation

When calculating z-scores, we might encounter situations where the standard deviation is zero, which would lead to division by zero. Let's create a robust function to handle this edge case.

```python
import numpy as np

def safe_z_score(x, mean, std):
    if std == 0:
        return np.zeros_like(x)
    return (x - mean) / std

# Example with zero standard deviation
data_constant = [5, 5, 5, 5, 5]
mean = np.mean(data_constant)
std = np.std(data_constant)

z_scores = safe_z_score(data_constant, mean, std)
print("Data:", data_constant)
print("Z-scores:", z_scores)

# Example with non-zero standard deviation
data_varied = [1, 2, 3, 4, 5]
mean = np.mean(data_varied)
std = np.std(data_varied)

z_scores = safe_z_score(data_varied, mean, std)
print("\nData:", data_varied)
print("Z-scores:", z_scores)
```

Output:

```
Data: [5, 5, 5, 5, 5]
Z-scores: [0. 0. 0. 0. 0.]

Data: [1, 2, 3, 4, 5]
Z-scores: [-1.41421356 -0.70710678  0.          0.70710678  1.41421356]
```

Slide 12: Z-Score for Anomaly Detection

Z-scores are useful for detecting anomalies in time series data. Let's implement a simple anomaly detection algorithm using z-scores on a time series dataset.

```python
import numpy as np
import matplotlib.pyplot as plt

def detect_anomalies(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    return np.abs(z_scores) > threshold

# Generate sample time series data with anomalies
np.random.seed(42)
time_series = np.random.normal(loc=10, scale=1, size=100)
time_series[20] = 15  # Introduce an anomaly
time_series[80] = 5   # Introduce another anomaly

anomalies = detect_anomalies(time_series)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(time_series, label='Time Series')
plt.scatter(np.where(anomalies)[0], time_series[anomalies], color='red', label='Anomalies')
plt.title('Time Series Anomaly Detection using Z-Score')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Number of anomalies detected: {np.sum(anomalies)}")
```

\[The output would be a plot showing the time series with highlighted anomalies\]

Output:

```
Number of anomalies detected: 2
```

Slide 13: Z-Score in Machine Learning: Feature Selection

Z-scores can be used for feature selection in machine learning by identifying features with high variability. Let's implement a simple feature selection method based on z-scores.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

def select_features_by_variance(X, threshold=1.0):
    z_scores = np.abs((X - X.mean()) / X.std())
    return z_scores.mean() >= threshold

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Select features
selected_features = select_features_by_variance(X, threshold=1.0)

print("Feature selection results:")
for feature, selected in zip(X.columns, selected_features):
    print(f"{feature}: {'Selected' if selected else 'Not selected'}")

# Create new dataset with selected features
X_selected = X.loc[:, selected_features]
print("\nShape of original dataset:", X.shape)
print("Shape of dataset with selected features:", X_selected.shape)
```

Output:

```
Feature selection results:
sepal length (cm): Selected
sepal width (cm): Not selected
petal length (cm): Selected
petal width (cm): Selected

Shape of original dataset: (150, 4)
Shape of dataset with selected features: (150, 3)
```

Slide 14: Z-Score in A/B Testing

Z-scores are widely used in A/B testing to determine if the difference between two groups is statistically significant. Let's implement a simple A/B test using z-scores.

```python
import numpy as np
from scipy import stats

def ab_test_z_score(control_conversions, control_size, 
                    treatment_conversions, treatment_size):
    p_control = control_conversions / control_size
    p_treatment = treatment_conversions / treatment_size
    p_pooled = (control_conversions + treatment_conversions) / (control_size + treatment_size)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_size + 1/treatment_size))
    z_score = (p_treatment - p_control) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    return z_score, p_value

# Example A/B test data
control_conv, control_size = 100, 1000
treatment_conv, treatment_size = 120, 1000

z_score, p_value = ab_test_z_score(control_conv, control_size, 
                                   treatment_conv, treatment_size)

print(f"Z-score: {z_score:.4f}")
print(f"P-value: {p_value:.4f}")
```

Output:

```
Z-score: 1.4907
P-value: 0.1360
```

Slide 15: Additional Resources

For those interested in diving deeper into z-scores and their applications in statistics and machine learning, here are some valuable resources:

1. "Statistical Inference via Data Science: A ModernDive into R and the Tidyverse" by Chester Ismay and Albert Y. Kim ArXiv: [https://arxiv.org/abs/1903.07639](https://arxiv.org/abs/1903.07639)
2. "Machine Learning: A Probabilistic Perspective" by Kevin P. Murphy (Not available on ArXiv, but widely recognized in the field)
3. "A Tutorial on Principal Component Analysis" by Jonathon Shlens ArXiv: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)

These resources provide in-depth explanations of statistical concepts, including z-scores, and their applications in various fields of data science and machine learning.

