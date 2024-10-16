## Understanding Robust Scaler with Python

Slide 1: Understanding Robust Scaler

Robust Scaler is a preprocessing technique used in machine learning to scale features that may contain outliers. It uses statistics that are robust to outliers, making it particularly useful for datasets with extreme values.

```python
from sklearn.preprocessing import RobustScaler
import numpy as np

# Sample data with outliers
data = np.array([[1], [2], [3], [100], [5]])

scaler = RobustScaler()
scaled_data = scaler.fit_transform(data)

print("Original data:", data.flatten())
print("Scaled data:", scaled_data.flatten())
```

Slide 2: How Robust Scaler Works

Robust Scaler uses the interquartile range (IQR) and median to scale the data. It subtracts the median and divides by the IQR, making it less sensitive to outliers compared to standard scaling methods.

```python
import numpy as np

def robust_scale(data):
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return (data - median) / iqr

# Example usage
data = np.array([1, 2, 3, 100, 5])
scaled_data = robust_scale(data)
print("Scaled data:", scaled_data)
```

Slide 3: Comparing Robust Scaler with Standard Scaler

Robust Scaler performs better than Standard Scaler when dealing with outliers. Let's compare their performance on a dataset with extreme values.

```python
from sklearn.preprocessing import RobustScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Generate data with outliers
data = np.concatenate([np.random.normal(0, 1, 100), np.array([10, -10, 15, -15])])

# Apply scalers
robust_scaler = RobustScaler()
standard_scaler = StandardScaler()

robust_scaled = robust_scaler.fit_transform(data.reshape(-1, 1))
standard_scaled = standard_scaler.fit_transform(data.reshape(-1, 1))

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.title("Original Data")
plt.hist(data, bins=30)
plt.subplot(132)
plt.title("Robust Scaled")
plt.hist(robust_scaled, bins=30)
plt.subplot(133)
plt.title("Standard Scaled")
plt.hist(standard_scaled, bins=30)
plt.tight_layout()
plt.show()
```

Slide 4: Implementing Robust Scaler from Scratch

Understanding the inner workings of Robust Scaler helps in grasping its concept better. Let's implement a simple version of Robust Scaler.

```python
import numpy as np

class SimpleRobustScaler:
    def __init__(self):
        self.median = None
        self.iqr = None
    
    def fit(self, X):
        self.median = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        self.iqr = q3 - q1
        return self
    
    def transform(self, X):
        return (X - self.median) / self.iqr
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# Usage
X = np.array([[1, 2], [3, 4], [5, 6], [100, 200]])
scaler = SimpleRobustScaler()
X_scaled = scaler.fit_transform(X)
print("Original data:\n", X)
print("Scaled data:\n", X_scaled)
```

Slide 5: Handling Multi-dimensional Data

Robust Scaler can handle multi-dimensional data, scaling each feature independently. This is particularly useful for datasets with multiple columns.

```python
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd

# Create a sample dataset
data = pd.DataFrame({
    'A': [1, 2, 3, 100, 5],
    'B': [10, 20, 30, 40, 1000],
    'C': [100, 200, 300, 400, 500]
})

scaler = RobustScaler()
scaled_data = scaler.fit_transform(data)

# Convert back to DataFrame for better visualization
scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

print("Original data:\n", data)
print("\nScaled data:\n", scaled_df)
```

Slide 6: Robust Scaler in Machine Learning Pipeline

Integrating Robust Scaler into a machine learning pipeline ensures consistent preprocessing of both training and test data.

```python
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, 
                           n_redundant=10, n_repeated=0, n_classes=2, 
                           n_clusters_per_class=2, weights=None, flip_y=0.01, 
                           class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, 
                           shuffle=True, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with RobustScaler and LogisticRegression
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('classifier', LogisticRegression())
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the model
score = pipeline.score(X_test, y_test)
print(f"Model accuracy: {score:.2f}")
```

Slide 7: Visualizing the Effect of Robust Scaler

Let's visualize how Robust Scaler transforms data compared to the original distribution, especially in the presence of outliers.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

# Generate data with outliers
np.random.seed(42)
data = np.concatenate([np.random.normal(0, 1, 1000), np.random.normal(0, 1, 10) * 10])

# Apply RobustScaler
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(data, bins=50, edgecolor='black')
ax1.set_title('Original Data Distribution')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')

ax2.hist(scaled_data, bins=50, edgecolor='black')
ax2.set_title('Robust Scaled Data Distribution')
ax2.set_xlabel('Scaled Value')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

Slide 8: Robust Scaler vs. Min-Max Scaler

Comparing Robust Scaler with Min-Max Scaler helps understand when to use each scaling method.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler

# Generate data with outliers
np.random.seed(42)
data = np.concatenate([np.random.normal(0, 1, 100), np.array([10, -10, 15, -15])])

# Apply scalers
robust_scaler = RobustScaler()
minmax_scaler = MinMaxScaler()

robust_scaled = robust_scaler.fit_transform(data.reshape(-1, 1)).flatten()
minmax_scaled = minmax_scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.boxplot(data)
ax1.set_title('Original Data')

ax2.boxplot(robust_scaled)
ax2.set_title('Robust Scaled Data')

ax3.boxplot(minmax_scaled)
ax3.set_title('Min-Max Scaled Data')

plt.tight_layout()
plt.show()
```

Slide 9: Handling Missing Values with Robust Scaler

Robust Scaler can handle datasets with missing values by using the `sklearn.impute` module.

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# Create a dataset with missing values
data = pd.DataFrame({
    'A': [1, 2, np.nan, 100, 5],
    'B': [10, np.nan, 30, 40, 1000],
    'C': [100, 200, 300, np.nan, 500]
})

# Create a pipeline with imputer and scaler
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

# Fit and transform the data
scaled_data = pipeline.fit_transform(data)

print("Original data with missing values:\n", data)
print("\nScaled data after imputation:\n", scaled_data)
```

Slide 10: Real-life Example: Image Processing

Robust Scaler can be useful in image processing to normalize pixel intensities while being robust to outliers caused by noise or artifacts.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from skimage import io, util

# Load a sample image and add some noise
image = io.imread('https://scikit-image.org/docs/stable/_static/img/logo.png')
noisy_image = util.random_noise(image, mode='s&p', amount=0.1)

# Apply RobustScaler to the noisy image
scaler = RobustScaler()
scaled_image = scaler.fit_transform(noisy_image.reshape(-1, 1)).reshape(noisy_image.shape)

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(image)
ax1.set_title('Original Image')
ax1.axis('off')

ax2.imshow(noisy_image)
ax2.set_title('Noisy Image')
ax2.axis('off')

ax3.imshow(scaled_image)
ax3.set_title('Robust Scaled Image')
ax3.axis('off')

plt.tight_layout()
plt.show()
```

Slide 11: Real-life Example: Sensor Data Processing

Robust Scaler is particularly useful for processing sensor data, which often contains outliers due to measurement errors or environmental factors.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

# Simulate sensor data with outliers
np.random.seed(42)
timestamps = pd.date_range(start='2023-01-01', periods=1000, freq='H')
temperature = np.random.normal(25, 5, 1000)
temperature[np.random.randint(0, 1000, 20)] = np.random.uniform(50, 100, 20)  # Add outliers

sensor_data = pd.DataFrame({'timestamp': timestamps, 'temperature': temperature})

# Apply RobustScaler
scaler = RobustScaler()
scaled_temperature = scaler.fit_transform(sensor_data[['temperature']])

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.plot(sensor_data['timestamp'], sensor_data['temperature'])
ax1.set_title('Original Temperature Data')
ax1.set_ylabel('Temperature (°C)')

ax2.plot(sensor_data['timestamp'], scaled_temperature)
ax2.set_title('Robust Scaled Temperature Data')
ax2.set_ylabel('Scaled Temperature')

plt.tight_layout()
plt.show()
```

Slide 12: Limitations of Robust Scaler

While Robust Scaler is effective for handling outliers, it's important to understand its limitations and when to use alternative methods.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler

# Generate data with a very skewed distribution
np.random.seed(42)
skewed_data = np.exp(np.random.normal(0, 1, 1000))

# Apply RobustScaler and StandardScaler
robust_scaler = RobustScaler()
standard_scaler = StandardScaler()

robust_scaled = robust_scaler.fit_transform(skewed_data.reshape(-1, 1)).flatten()
standard_scaled = standard_scaler.fit_transform(skewed_data.reshape(-1, 1)).flatten()

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.hist(skewed_data, bins=50, edgecolor='black')
ax1.set_title('Original Skewed Data')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')

ax2.hist(robust_scaled, bins=50, edgecolor='black')
ax2.set_title('Robust Scaled Data')
ax2.set_xlabel('Scaled Value')
ax2.set_ylabel('Frequency')

ax3.hist(standard_scaled, bins=50, edgecolor='black')
ax3.set_title('Standard Scaled Data')
ax3.set_xlabel('Scaled Value')
ax3.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

Slide 13: Choosing the Right Scaler

Selecting the appropriate scaling method depends on your data characteristics and the requirements of your machine learning algorithm.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Generate a dataset with outliers
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X[np.random.randint(0, 1000, 50)] = np.random.uniform(10, 20, (50, 20))

# Compare different scalers
scalers = {
    'No Scaling': None,
    'RobustScaler': RobustScaler(),
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler()
}

results = {}

for name, scaler in scalers.items():
    X_scaled = scaler.fit_transform(X) if scaler else X
    scores = cross_val_score(SVC(), X_scaled, y, cv=5)
    results[name] = scores.mean()

# Print results
for name, score in results.items():
    print(f"{name}: {score:.4f}")
```

Slide 14: Robust Scaler in Feature Engineering

Robust Scaler can be an essential tool in feature engineering, especially when dealing with features that have different scales and contain outliers.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer

# Create a sample dataset with mixed features
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.normal(35, 10, 1000),
    'income': np.random.lognormal(10, 1, 1000),
    'credit_score': np.random.uniform(300, 850, 1000)
})

# Add some outliers
data.loc[np.random.choice(data.index, 20), 'income'] *= 10

# Create a ColumnTransformer to apply RobustScaler only to numerical columns
ct = ColumnTransformer([
    ('robust', RobustScaler(), ['age', 'income', 'credit_score'])
])

# Fit and transform the data
scaled_data = ct.fit_transform(data)

# Convert back to DataFrame for better visualization
scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

print("Original data summary:")
print(data.describe())
print("\nScaled data summary:")
print(scaled_df.describe())
```

Slide 15: Additional Resources

For those interested in diving deeper into Robust Scaler and related preprocessing techniques, consider exploring these resources:

1. Scikit-learn documentation on RobustScaler: [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
2. "A Comparative Study of White Box Data Standardization Techniques in Credit Card Fraud Detection" by S. Maldonado et al. (2020): [https://arxiv.org/abs/2012.11202](https://arxiv.org/abs/2012.11202)
3. "Benchmarking and Survey of Outlier Detection Methods" by S. Kandanaarachchi and R. J. Hyndman (2020): [https://arxiv.org/abs/2003.06979](https://arxiv.org/abs/2003.06979)

These resources provide in-depth information on preprocessing techniques, their applications, and comparative studies in various domains.

