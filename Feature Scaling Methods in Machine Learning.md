## Feature Scaling Methods in Machine Learning
Slide 1: Understanding Feature Scaling Methods

Feature scaling is a crucial preprocessing step in machine learning that transforms numerical features to a similar scale, improving model performance and convergence. The main techniques include normalization, standardization, and robust scaling.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Sample data
X = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]])

# Different scaling methods
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

# Apply scaling
X_standard = standard_scaler.fit_transform(X)
X_minmax = minmax_scaler.fit_transform(X)
X_robust = robust_scaler.fit_transform(X)
```

Slide 2: Min-Max Normalization (Feature Scaling)

Min-Max scaling transforms features to a fixed range \[0,1\] or \[-1,1\] by scaling the minimum and maximum values. This method is sensitive to outliers but preserves zero values and is particularly useful for neural networks.

```python
# Mathematical formula for Min-Max scaling:
'''
$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
'''

def min_max_scaling(X):
    # Custom implementation of Min-Max scaling
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_normalized = (X - X_min) / (X_max - X_min)
    return X_normalized, X_min, X_max

# Example usage
X = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
X_norm, min_vals, max_vals = min_max_scaling(X)
print("Original data:\n", X)
print("\nNormalized data:\n", X_norm)
```

Slide 3: Standardization (Z-score Normalization)

Standardization transforms features to have zero mean and unit variance. This technique is less affected by outliers compared to min-max scaling and is preferred when the data follows a Gaussian distribution.

```python
# Mathematical formula for Standardization:
'''
$$X_{std} = \frac{X - \mu}{\sigma}$$
'''

def standardization(X):
    # Custom implementation of standardization
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_standardized = (X - mean) / std
    return X_standardized, mean, std

# Example usage
X = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
X_std, mean_vals, std_vals = standardization(X)
print("Original data:\n", X)
print("\nStandardized data:\n", X_std)
```

Slide 4: Robust Scaling

Robust scaling uses statistics that are robust to outliers. This method scales features using the interquartile range (IQR) and is particularly useful when data contains significant outliers or follows a non-Gaussian distribution.

```python
def robust_scaling(X):
    # Custom implementation of robust scaling
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    median = np.median(X, axis=0)
    X_robust = (X - median) / iqr
    return X_robust, median, iqr

# Example usage with outliers
X = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1], [100, -100, 200]])
X_rob, median_vals, iqr_vals = robust_scaling(X)
print("Original data:\n", X)
print("\nRobust scaled data:\n", X_rob)
```

Slide 5: L2 Normalization (Not a Feature Scaling Method)

L2 normalization, while often confused with feature scaling, is actually a vector normalization technique that scales samples individually to have unit norm. This is fundamentally different from traditional feature scaling methods.

```python
# Mathematical formula for L2 normalization:
'''
$$X_{l2} = \frac{X}{\sqrt{\sum_{i=1}^{n} x_i^2}}$$
'''

def l2_normalization(X):
    # Custom implementation of L2 normalization
    norm = np.sqrt(np.sum(X**2, axis=1))[:, np.newaxis]
    X_normalized = X / norm
    return X_normalized

# Example usage
X = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
X_l2 = l2_normalization(X)
print("Original data:\n", X)
print("\nL2 normalized data:\n", X_l2)
print("\nVerify unit norm:", np.sum(X_l2**2, axis=1))
```

Slide 6: Real-world Example: Credit Card Fraud Detection

A practical implementation showing how different scaling methods affect the performance of fraud detection models. This example demonstrates the importance of choosing the right scaling method for financial data.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

# Generate synthetic credit card transaction data
np.random.seed(42)
n_samples = 1000
transaction_amounts = np.random.lognormal(mean=5, sigma=2, size=n_samples)
transaction_times = np.random.uniform(0, 24, n_samples)
fraud_labels = np.random.binomial(n=1, p=0.1, size=n_samples)

# Create dataset
X = np.column_stack((transaction_amounts, transaction_times))
y = fraud_labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Compare different scaling methods
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

results = {}
for name, scaler in scalers.items():
    # Scale data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred)
    }
```

Slide 7: Results for Credit Card Fraud Detection

Analysis of the performance metrics for different scaling methods in the fraud detection example, showcasing how each method affects model accuracy and precision.

```python
# Display results
for scaler_name, metrics in results.items():
    print(f"\n{scaler_name} Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")

# Visualization of results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
metrics = ['accuracy', 'precision']
for metric in metrics:
    values = [results[scaler][metric] for scaler in results]
    plt.plot(list(results.keys()), values, marker='o', label=metric)

plt.title('Scaling Methods Comparison')
plt.ylabel('Score')
plt.legend()
plt.xticks(rotation=45)
```

Slide 8: Maximum Absolute Scaling

Maximum absolute scaling scales features by dividing each feature by its maximum absolute value, ensuring the feature range falls within \[-1, 1\] while preserving sparsity and zero values.

```python
def max_abs_scaling(X):
    # Mathematical formula:
    '''
    $$X_{scaled} = \frac{X}{max(|X|)}$$
    '''
    max_abs = np.max(np.abs(X), axis=0)
    X_scaled = X / max_abs
    return X_scaled, max_abs

# Example usage
X = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
X_maxabs, max_abs_vals = max_abs_scaling(X)
print("Original data:\n", X)
print("\nMax Abs scaled data:\n", X_maxabs)
print("\nMax absolute values:", max_abs_vals)
```

Slide 9: Real-world Example: Image Processing

Demonstrating how different scaling methods affect image processing tasks, particularly in preparing data for convolutional neural networks.

```python
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def process_image(image_array):
    # Original pixel values (0-255)
    print("Original pixel range:", np.min(image_array), "-", np.max(image_array))
    
    # MinMax scaling (0-1)
    minmax = MinMaxScaler()
    img_minmax = minmax.fit_transform(image_array.reshape(-1, 1)).reshape(image_array.shape)
    print("MinMax scaled range:", np.min(img_minmax), "-", np.max(img_minmax))
    
    # Standardization
    standard = StandardScaler()
    img_standard = standard.fit_transform(image_array.reshape(-1, 1)).reshape(image_array.shape)
    print("Standardized range:", np.min(img_standard), "-", np.max(img_standard))
    
    return img_minmax, img_standard

# Create synthetic image data
image_array = np.random.randint(0, 256, size=(64, 64))
img_minmax, img_standard = process_image(image_array)

# Example showing how scaling affects image intensities
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(image_array, cmap='gray')
plt.title('Original')
plt.subplot(132)
plt.imshow(img_minmax, cmap='gray')
plt.title('MinMax Scaled')
plt.subplot(133)
plt.imshow(img_standard, cmap='gray')
plt.title('Standardized')
```

Slide 10: Log Transformation

Log transformation is a powerful technique for handling skewed data and dealing with multiplicative relationships, though not strictly a scaling method. It's particularly useful for right-skewed distributions and features with exponential growth.

```python
def log_transform(X, epsilon=1e-10):
    # Mathematical formula:
    '''
    $$X_{log} = log(X + \epsilon)$$
    '''
    # Add small constant to avoid log(0)
    X_log = np.log(X + epsilon)
    
    # Example with skewed data
    np.random.seed(42)
    skewed_data = np.random.lognormal(mean=0, sigma=1, size=1000)
    
    # Apply log transformation
    transformed_data = log_transform(skewed_data)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(skewed_data, bins=50)
    ax1.set_title('Original Skewed Data')
    ax2.hist(transformed_data, bins=50)
    ax2.set_title('Log Transformed Data')
    
    return X_log

# Example usage
X = np.array([[10, 100, 1000], [1, 10, 100], [100, 1000, 10000]])
X_log = log_transform(X)
print("Original data:\n", X)
print("\nLog transformed data:\n", X_log)
```

Slide 11: Box-Cox Transformation

Box-Cox transformation is a power transformation method that helps to normalize data and stabilize variance. It's particularly useful when dealing with non-normal distributions and heteroscedasticity.

```python
from scipy import stats

def box_cox_transform(X, lambda_param=None):
    # Mathematical formula:
    '''
    $$X_{box-cox} = \begin{cases} 
    \frac{X^\lambda - 1}{\lambda} & \lambda \neq 0 \\
    \ln(X) & \lambda = 0 
    \end{cases}$$
    '''
    
    # Ensure data is positive
    X = np.array(X)
    if np.any(X <= 0):
        X = X - np.min(X) + 1  # Make all values positive
    
    if lambda_param is None:
        # Find optimal lambda parameter
        transformed_data = []
        for col in range(X.shape[1]):
            data_transformed, lambda_opt = stats.boxcox(X[:, col])
            transformed_data.append(data_transformed)
        return np.column_stack(transformed_data)
    else:
        return stats.boxcox(X, lambda_param)

# Example usage
np.random.seed(42)
X = np.abs(np.random.normal(loc=5, scale=2, size=(100, 3)))
X_boxcox = box_cox_transform(X)
print("Original data statistics:")
print("Skewness:", stats.skew(X))
print("\nTransformed data statistics:")
print("Skewness:", stats.skew(X_boxcox))
```

Slide 12: Quantile Transformation

Quantile transformation maps the original data distribution to a uniform or normal distribution, making it robust against outliers and useful for non-linear feature relationships.

```python
from sklearn.preprocessing import QuantileTransformer

def quantile_transform(X, n_quantiles=1000, output_distribution='normal'):
    # Custom implementation
    def custom_quantile_transform(x, n_quantiles):
        sorted_idx = np.argsort(x)
        ranks = np.zeros_like(sorted_idx)
        ranks[sorted_idx] = np.linspace(0, 1, len(x))
        return ranks
    
    # Compare with sklearn implementation
    qt = QuantileTransformer(n_quantiles=n_quantiles, 
                           output_distribution=output_distribution)
    
    # Apply transformations
    X_custom = np.apply_along_axis(
        lambda x: custom_quantile_transform(x, n_quantiles), 0, X)
    X_sklearn = qt.fit_transform(X)
    
    return X_custom, X_sklearn

# Example with skewed data
np.random.seed(42)
X = np.random.exponential(scale=2.0, size=(1000, 2))
X_custom, X_sklearn = quantile_transform(X)

print("Original data shape:", X.shape)
print("Transformed data shape:", X_sklearn.shape)
print("\nOriginal data statistics:")
print("Mean:", np.mean(X, axis=0))
print("Std:", np.std(X, axis=0))
print("\nTransformed data statistics:")
print("Mean:", np.mean(X_sklearn, axis=0))
print("Std:", np.std(X_sklearn, axis=0))
```

Slide 13: Additional Resources

*   Feature Scaling Methods in Machine Learning
    *   [https://arxiv.org/abs/2001.09876](https://arxiv.org/abs/2001.09876)
*   Impact of Feature Scaling on Deep Learning Models
    *   [https://arxiv.org/abs/1912.03747](https://arxiv.org/abs/1912.03747)
*   Comparative Analysis of Data Normalization Techniques
    *   Search on Google Scholar: "comparative analysis feature scaling machine learning"
*   Robust Feature Scaling for Machine Learning
    *   [https://www.sciencedirect.com/science/article/pii/S0925231219311622](https://www.sciencedirect.com/science/article/pii/S0925231219311622)

