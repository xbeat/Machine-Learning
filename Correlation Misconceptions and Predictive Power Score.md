## Correlation Misconceptions and Predictive Power Score
Slide 1: Understanding Correlation and Its Limitations

Correlation is a widely used statistical measure, but it's often misunderstood. This slideshow will explore the limitations of correlation and introduce the Predictive Power Score (PPS) as an alternative measure for assessing relationships between variables.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
x = np.linspace(0, 10, 100)
y = x + np.random.normal(0, 1, 100)

# Calculate correlation
correlation = np.corrcoef(x, y)[0, 1]

# Plot the data
plt.scatter(x, y)
plt.title(f"Correlation: {correlation:.2f}")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Slide 2: Linear Relationships in Correlation

Correlation measures how two variables vary together linearly or monotonically. This means it may not capture complex, non-linear relationships effectively.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate non-linear data
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)

# Calculate correlation
correlation = np.corrcoef(x, y)[0, 1]

# Plot the data
plt.scatter(x, y)
plt.title(f"Non-linear Relationship\nCorrelation: {correlation:.2f}")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Slide 3: Symmetry in Correlation

One key property of correlation is its symmetry: corr(A, B) = corr(B, A). However, this symmetry can be misleading when the relationship between variables is inherently asymmetric.

```python
import numpy as np

# Generate sample data
A = np.random.rand(100)
B = A ** 2 + np.random.normal(0, 0.1, 100)

# Calculate correlations
corr_AB = np.corrcoef(A, B)[0, 1]
corr_BA = np.corrcoef(B, A)[0, 1]

print(f"Correlation(A, B): {corr_AB:.4f}")
print(f"Correlation(B, A): {corr_BA:.4f}")
```

Slide 4: Real-Life Example: Date and Month

Consider the relationship between a date and its corresponding month. Given a date, it's easy to determine the month, but the reverse is not true. Correlation fails to capture this asymmetry.

```python
import pandas as pd
import numpy as np

# Generate dates for a year
dates = pd.date_range(start='2023-01-01', end='2023-12-31')
df = pd.DataFrame({'date': dates, 'day': dates.day, 'month': dates.month})

# Calculate correlations
corr_day_month = np.corrcoef(df['day'], df['month'])[0, 1]
corr_month_day = np.corrcoef(df['month'], df['day'])[0, 1]

print(f"Correlation(day, month): {corr_day_month:.4f}")
print(f"Correlation(month, day): {corr_month_day:.4f}")
```

Slide 5: Predictiveness vs. Correlation

Correlation is often misinterpreted as a measure of predictiveness. However, it doesn't necessarily indicate how well one variable can predict another, especially in non-linear relationships.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Generate non-linear data
X = np.linspace(0, 10, 1000).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, 1000)

# Calculate correlation
correlation = np.corrcoef(X.ravel(), y)[0, 1]

# Train a decision tree regressor
model = DecisionTreeRegressor(max_depth=5)
model.fit(X, y)

# Calculate MAE
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)

print(f"Correlation: {correlation:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
```

Slide 6: Limitations with Categorical Data

Correlation is primarily designed for numerical data, making it less suitable for categorical variables. This limitation can be problematic when working with diverse datasets.

```python
import pandas as pd
import numpy as np

# Create a sample dataset with categorical and numerical data
data = {
    'Category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'],
    'Value': [10, 15, 12, 18, 14, 20, 11, 16]
}
df = pd.DataFrame(data)

# Convert categorical data to numerical
df['Category_encoded'] = pd.Categorical(df['Category']).codes

# Calculate correlation
correlation = np.corrcoef(df['Category_encoded'], df['Value'])[0, 1]

print(f"Correlation between Category and Value: {correlation:.4f}")
```

Slide 7: Introducing the Predictive Power Score (PPS)

The Predictive Power Score (PPS) addresses many limitations of correlation. It measures the predictive power of a feature, considering both linear and non-linear relationships.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def calculate_pps(X, y):
    # Train a decision tree regressor
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X.reshape(-1, 1), y)
    
    # Calculate MAE
    y_pred = model.predict(X.reshape(-1, 1))
    mae_model = mean_absolute_error(y, y_pred)
    
    # Calculate baseline MAE (median prediction)
    mae_baseline = mean_absolute_error(y, np.full_like(y, np.median(y)))
    
    # Calculate PPS
    pps = 1 - mae_model / mae_baseline
    return max(0, pps)  # PPS is always between 0 and 1

# Generate sample data
X = np.linspace(0, 10, 1000)
y = np.sin(X) + np.random.normal(0, 0.1, 1000)

pps_x_y = calculate_pps(X, y)
pps_y_x = calculate_pps(y, X)

print(f"PPS(X → Y): {pps_x_y:.4f}")
print(f"PPS(Y → X): {pps_y_x:.4f}")
```

Slide 8: Asymmetry in PPS

Unlike correlation, PPS is asymmetric, meaning PPS(a → b) is not necessarily equal to PPS(b → a). This property allows PPS to capture the directional nature of predictive relationships.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def calculate_pps(X, y):
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X.reshape(-1, 1), y)
    y_pred = model.predict(X.reshape(-1, 1))
    mae_model = mean_absolute_error(y, y_pred)
    mae_baseline = mean_absolute_error(y, np.full_like(y, np.median(y)))
    return max(0, 1 - mae_model / mae_baseline)

# Generate asymmetric data
X = np.random.uniform(0, 10, 1000)
y = X ** 2 + np.random.normal(0, 5, 1000)

pps_x_y = calculate_pps(X, y)
pps_y_x = calculate_pps(y, X)

print(f"PPS(X → Y): {pps_x_y:.4f}")
print(f"PPS(Y → X): {pps_y_x:.4f}")
```

Slide 9: PPS for Categorical Data

PPS can handle both numerical and categorical data, making it more versatile than correlation. For categorical targets, it uses a decision tree classifier and compares F1 scores.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

def calculate_pps_categorical(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X.reshape(-1, 1), y_encoded)
    
    y_pred = model.predict(X.reshape(-1, 1))
    f1_model = f1_score(y_encoded, y_pred, average='weighted')
    
    # Baseline: most frequent class prediction
    y_baseline = np.full_like(y_encoded, np.argmax(np.bincount(y_encoded)))
    f1_baseline = f1_score(y_encoded, y_baseline, average='weighted')
    
    return max(0, (f1_model - f1_baseline) / (1 - f1_baseline))

# Generate sample data
X = np.random.uniform(0, 10, 1000)
y = ['Low' if x < 3 else 'Medium' if x < 7 else 'High' for x in X]

pps = calculate_pps_categorical(X, y)
print(f"PPS(X → Y): {pps:.4f}")
```

Slide 10: Non-linear and Non-monotonic Relationships

PPS excels at capturing both linear and non-linear relationships, as well as monotonic and non-monotonic relationships. This makes it more robust than correlation in complex scenarios.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def calculate_pps(X, y):
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X.reshape(-1, 1), y)
    y_pred = model.predict(X.reshape(-1, 1))
    mae_model = mean_absolute_error(y, y_pred)
    mae_baseline = mean_absolute_error(y, np.full_like(y, np.median(y)))
    return max(0, 1 - mae_model / mae_baseline)

# Generate non-monotonic data
X = np.linspace(0, 4*np.pi, 1000)
y = np.sin(X) + np.random.normal(0, 0.1, 1000)

correlation = np.corrcoef(X, y)[0, 1]
pps = calculate_pps(X, y)

plt.scatter(X, y, alpha=0.5)
plt.title(f"Non-monotonic Relationship\nCorrelation: {correlation:.4f}, PPS: {pps:.4f}")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

print(f"Correlation: {correlation:.4f}")
print(f"PPS(X → Y): {pps:.4f}")
```

Slide 11: Real-Life Example: Weather Prediction

Consider predicting temperature based on humidity. While correlation might suggest a weak linear relationship, PPS can reveal a stronger predictive power due to non-linear patterns.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def calculate_pps(X, y):
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X.reshape(-1, 1), y)
    y_pred = model.predict(X.reshape(-1, 1))
    mae_model = mean_absolute_error(y, y_pred)
    mae_baseline = mean_absolute_error(y, np.full_like(y, np.median(y)))
    return max(0, 1 - mae_model / mae_baseline)

# Simulate weather data
np.random.seed(42)
humidity = np.random.uniform(30, 100, 1000)
temperature = 20 + 0.2 * humidity + 0.01 * humidity**2 + np.random.normal(0, 2, 1000)

correlation = np.corrcoef(humidity, temperature)[0, 1]
pps_humidity_temp = calculate_pps(humidity, temperature)
pps_temp_humidity = calculate_pps(temperature, humidity)

print(f"Correlation: {correlation:.4f}")
print(f"PPS(Humidity → Temperature): {pps_humidity_temp:.4f}")
print(f"PPS(Temperature → Humidity): {pps_temp_humidity:.4f}")
```

Slide 12: When to Use Correlation vs. PPS

Choosing between correlation and PPS depends on your objective. Use correlation to understand general monotonic trends between variables. Use PPS to assess the predictive power of features.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def calculate_pps(X, y):
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X.reshape(-1, 1), y)
    y_pred = model.predict(X.reshape(-1, 1))
    mae_model = mean_absolute_error(y, y_pred)
    mae_baseline = mean_absolute_error(y, np.full_like(y, np.median(y)))
    return max(0, 1 - mae_model / mae_baseline)

# Generate data with different relationships
X = np.linspace(0, 10, 1000)
y_linear = 2 * X + np.random.normal(0, 1, 1000)
y_nonlinear = np.sin(X) + np.random.normal(0, 0.1, 1000)

# Calculate correlation and PPS for both relationships
corr_linear = np.corrcoef(X, y_linear)[0, 1]
pps_linear = calculate_pps(X, y_linear)
corr_nonlinear = np.corrcoef(X, y_nonlinear)[0, 1]
pps_nonlinear = calculate_pps(X, y_nonlinear)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X, y_linear, alpha=0.5)
ax1.set_title(f"Linear Relationship\nCorr: {corr_linear:.4f}, PPS: {pps_linear:.4f}")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

ax2.scatter(X, y_nonlinear, alpha=0.5)
ax2.set_title(f"Non-linear Relationship\nCorr: {corr_nonlinear:.4f}, PPS: {pps_nonlinear:.4f}")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")

plt.tight_layout()
plt.show()
```

Slide 13: Conclusion and Best Practices

Both correlation and PPS have their place in data analysis. Correlation is useful for quick assessments of linear relationships, while PPS provides a more comprehensive view of predictive power. Use them in combination to gain deeper insights into your data.

```python
import numpy as np
import pandas as pd

# Sample data
data = {
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'target': np.random.rand(100)
}
df = pd.DataFrame(data)

# Calculate correlation matrix
corr_matrix = df.corr()

# Calculate PPS (pseudocode)
def calculate_pps(df, target):
    pps_scores = {}
    for feature in df.columns:
        if feature != target:
            # Train decision tree and calculate PPS
            pps_scores[feature] = pps_calculation(df[feature], df[target])
    return pps_scores

pps_scores = calculate_pps(df, 'target')

print("Correlation Matrix:")
print(corr_matrix)
print("\nPPS Scores:")
print(pps_scores)
```

Slide 14: Additional Resources

For those interested in delving deeper into the concepts of correlation and Predictive Power Score, the following resources are recommended:

1. "The Predictive Power Score: A New Measure of Predictive Power" by Florian Wetschoreck, Lukas Reubelt, and Tobias Schmidt (2020). Available on arXiv: [https://arxiv.org/abs/2011.01088](https://arxiv.org/abs/2011.01088)
2. "Correlation and dependence" by Roger B. Nelsen (1999). In Encyclopedia of Statistical Sciences, Wiley Online Library.
3. "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009). This book provides a comprehensive overview of statistical learning methods, including correlation and predictive modeling.

These resources offer in-depth explanations and mathematical foundations for the concepts discussed in this presentation. They can help you further understand the nuances of correlation, predictive power, and their applications in data analysis and machine learning.

