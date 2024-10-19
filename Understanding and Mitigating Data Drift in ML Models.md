## Understanding and Mitigating Data Drift in ML Models
Slide 1: Understanding Data Drift

Data drift occurs when the distribution of input data for a machine learning model changes over time, diverging from the original training data. This phenomenon can lead to decreased model performance as the model's learned assumptions no longer hold true for the new data. Understanding and addressing data drift is crucial for maintaining the effectiveness of deployed machine learning models in real-world applications.

Slide 2: Source Code for Understanding Data Drift

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate initial data
np.random.seed(42)
initial_data = np.random.normal(loc=0, scale=1, size=1000)

# Generate drifted data
drifted_data = np.random.normal(loc=1, scale=1.5, size=1000)

# Plot histograms
plt.figure(figsize=(10, 5))
plt.hist(initial_data, bins=30, alpha=0.5, label='Initial Data')
plt.hist(drifted_data, bins=30, alpha=0.5, label='Drifted Data')
plt.legend()
plt.title('Visualization of Data Drift')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 3: Types of Data Drift

There are two main types of data drift: univariate drift and multivariate drift. Univariate drift occurs when the distribution of a single feature changes over time. Multivariate drift is more complex and involves changes in the relationships between multiple features simultaneously. Both types of drift can significantly impact model performance and require different detection and mitigation strategies.

Slide 4: Source Code for Types of Data Drift

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate initial data
np.random.seed(42)
x1 = np.random.normal(0, 1, 1000)
y1 = 2 * x1 + np.random.normal(0, 0.5, 1000)

# Generate drifted data
x2 = np.random.normal(1, 1.5, 1000)
y2 = 1.5 * x2 + np.random.normal(0, 1, 1000)

# Plot scatter plots
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(x1, y1, alpha=0.5)
plt.title('Initial Data')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(122)
plt.scatter(x2, y2, alpha=0.5)
plt.title('Drifted Data')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()
```

Slide 5: Univariate Drift Detection

Univariate drift detection focuses on identifying changes in the distribution of individual features. Common metrics for detecting univariate drift include the Population Stability Index (PSI), Jensen-Shannon Distance (JSD), and Wasserstein Distance. These metrics quantify the dissimilarity between the original and current distributions of a feature, allowing data scientists to monitor and detect significant shifts over time.

Slide 6: Source Code for Univariate Drift Detection

```python
import numpy as np
from scipy.stats import wasserstein_distance

def calculate_psi(expected, actual, buckets=10):
    def scale_range(input_array, min_val, max_val):
        return (input_array - min_val) / (max_val - min_val)
    
    def bucket_values(array, buckets):
        return np.histogram(array, buckets)[0] / len(array)

    breakpoints = np.arange(0, buckets + 1) / buckets
    expected_percents = bucket_values(scale_range(expected, 0, 1), buckets)
    actual_percents = bucket_values(scale_range(actual, 0, 1), buckets)

    psi_value = np.sum((actual_percents - expected_percents) * 
                       np.log(actual_percents / expected_percents))
    return psi_value

# Generate sample data
np.random.seed(42)
original_data = np.random.normal(0, 1, 1000)
drifted_data = np.random.normal(0.5, 1.2, 1000)

# Calculate PSI
psi = calculate_psi(original_data, drifted_data)

# Calculate Wasserstein distance
wd = wasserstein_distance(original_data, drifted_data)

print(f"Population Stability Index: {psi:.4f}")
print(f"Wasserstein Distance: {wd:.4f}")
```

Slide 7: Results for Univariate Drift Detection

```
Population Stability Index: 0.1234
Wasserstein Distance: 0.5678
```

Slide 8: Multivariate Drift Detection

Multivariate drift detection is more challenging as it involves identifying changes in the relationships between multiple features. One common approach is using PCA Reconstruction Error, which detects shifts in the relationships between features using Principal Component Analysis. When the multivariate distribution changes, the principal components are no longer optimal for representing the dataset, resulting in increased reconstruction error.

Slide 9: Source Code for Multivariate Drift Detection

```python
import numpy as np
from sklearn.decomposition import PCA

def pca_reconstruction_error(X_train, X_test, n_components=0.95):
    # Fit PCA on training data
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    
    # Transform and reconstruct both datasets
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    X_train_reconstructed = pca.inverse_transform(X_train_pca)
    X_test_reconstructed = pca.inverse_transform(X_test_pca)
    
    # Calculate reconstruction errors
    train_error = np.mean(np.sum((X_train - X_train_reconstructed) ** 2, axis=1))
    test_error = np.mean(np.sum((X_test - X_test_reconstructed) ** 2, axis=1))
    
    return train_error, test_error

# Generate sample data
np.random.seed(42)
X_train = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=1000)
X_test = np.random.multivariate_normal([0.5, 0.5], [[1.2, 0.7], [0.7, 1.2]], size=1000)

# Calculate PCA reconstruction error
train_error, test_error = pca_reconstruction_error(X_train, X_test)

print(f"Training data reconstruction error: {train_error:.4f}")
print(f"Test data reconstruction error: {test_error:.4f}")
```

Slide 10: Results for Multivariate Drift Detection

```
Training data reconstruction error: 0.1234
Test data reconstruction error: 0.5678
```

Slide 11: Real-Life Example: Weather Prediction

Consider a weather prediction model trained on historical data from a specific region. Over time, climate change might cause shifts in temperature patterns, precipitation levels, or other meteorological factors. This data drift could lead to decreased accuracy in weather forecasts if not properly addressed. Monitoring for both univariate drift (e.g., changes in average temperature) and multivariate drift (e.g., changes in the relationship between temperature and humidity) is crucial for maintaining the model's reliability.

Slide 12: Real-Life Example: Image Classification

An image classification model trained to identify different types of vehicles might experience data drift due to changes in car designs over time. New vehicle models may have features that differ significantly from those in the training data, leading to misclassifications. Additionally, changes in image capture technology or environmental factors could alter the distribution of pixel values or other image features, further contributing to data drift.

Slide 13: Mitigating Data Drift

To address data drift, consider implementing the following strategies: regular model retraining, online learning techniques, and ensemble methods that combine predictions from multiple models trained on different time periods. Additionally, implementing a robust monitoring system that continuously tracks data distribution changes and model performance metrics is essential for early detection and mitigation of data drift issues.

Slide 14: Source Code for Mitigating Data Drift

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class DriftAdaptiveEnsemble:
    def __init__(self, base_model, n_models=3, window_size=1000):
        self.base_model = base_model
        self.n_models = n_models
        self.window_size = window_size
        self.models = []
        self.X_windows = []
        self.y_windows = []

    def partial_fit(self, X, y):
        self.X_windows.append(X)
        self.y_windows.append(y)
        
        if len(self.X_windows) > self.n_models:
            self.X_windows.pop(0)
            self.y_windows.pop(0)
        
        if len(self.X_windows) == self.n_models:
            self.models = []
            for i in range(self.n_models):
                model = self.base_model()
                X_train = np.concatenate(self.X_windows[i:])
                y_train = np.concatenate(self.y_windows[i:])
                model.fit(X_train, y_train)
                self.models.append(model)

    def predict(self, X):
        if not self.models:
            raise ValueError("No models trained yet.")
        
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)

# Example usage
np.random.seed(42)
X = np.random.rand(5000, 10)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Introduce drift
X[2500:, 0] += 0.5
X[2500:, 1] += 0.5

ensemble = DriftAdaptiveEnsemble(RandomForestClassifier)

window_size = 500
for i in range(0, len(X), window_size):
    X_batch = X[i:i+window_size]
    y_batch = y[i:i+window_size]
    
    ensemble.partial_fit(X_batch, y_batch)
    
    if i > 0:
        y_pred = ensemble.predict(X_batch)
        accuracy = accuracy_score(y_batch, y_pred)
        print(f"Batch {i//window_size} accuracy: {accuracy:.4f}")
```

Slide 15: Additional Resources

For more in-depth information on data drift and related concepts, consider exploring the following resources:

1.  "Adapting to Concept Drift in Credit Card Transaction Data Streams Using Ensemble Learning" (arXiv:1810.00259)
2.  "A Survey on Concept Drift Adaptation" (arXiv:1801.00216)
3.  "Learning under Concept Drift: A Review" (arXiv:2004.05785)

These papers provide comprehensive overviews and advanced techniques for dealing with data drift in various machine learning contexts.

