## Lasso L1 Penalty in Machine Learning with Python
Slide 1: Introduction to Lasso L1 Penalty

Lasso (Least Absolute Shrinkage and Selection Operator) is a regularization technique in machine learning that uses L1 penalty. It's particularly useful for feature selection and preventing overfitting in linear regression models. Lasso adds the absolute value of the coefficients as a penalty term to the loss function, encouraging sparsity in the model.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 20)
y = np.dot(X[:, :5], np.random.randn(5)) + np.random.randn(100)

# Create and fit Lasso model
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# Plot coefficient values
plt.figure(figsize=(10, 6))
plt.stem(range(20), lasso.coef_)
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficients')
plt.show()
```

Slide 2: Mathematical Formulation

The Lasso regression objective function combines the ordinary least squares (OLS) with an L1 penalty term:

minimize: ∑(y\_i - β\_0 - ∑(β\_j \* x\_ij))^2 + λ \* ∑|β\_j|

Where λ is the regularization strength, and |β\_j| represents the L1 norm of the coefficients.

```python
import numpy as np

def lasso_objective(X, y, beta, lambda_):
    n_samples = X.shape[0]
    ols_term = np.sum((y - np.dot(X, beta)) ** 2) / (2 * n_samples)
    l1_penalty = lambda_ * np.sum(np.abs(beta))
    return ols_term + l1_penalty

# Example usage
X = np.random.randn(100, 5)
y = np.random.randn(100)
beta = np.random.randn(5)
lambda_ = 0.1

objective_value = lasso_objective(X, y, beta, lambda_)
print(f"Objective function value: {objective_value}")
```

Slide 3: Lasso vs. Ridge Regression

Lasso (L1) and Ridge (L2) are both regularization techniques, but they have different effects on the model. Lasso tends to produce sparse models by forcing some coefficients to zero, effectively performing feature selection. Ridge, on the other hand, shrinks all coefficients towards zero but rarely sets them exactly to zero.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 10)
y = np.dot(X[:, :3], np.random.randn(3)) + np.random.randn(100)

# Fit Lasso and Ridge models
alphas = np.logspace(-3, 1, 100)
lasso_coefs = []
ridge_coefs = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    ridge = Ridge(alpha=alpha)
    lasso.fit(X, y)
    ridge.fit(X, y)
    lasso_coefs.append(lasso.coef_)
    ridge_coefs.append(ridge.coef_)

# Plot coefficient paths
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.semilogx(alphas, np.array(lasso_coefs))
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Lasso Path')

plt.subplot(122)
plt.semilogx(alphas, np.array(ridge_coefs))
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Ridge Path')

plt.tight_layout()
plt.show()
```

Slide 4: Feature Selection with Lasso

One of the key advantages of Lasso is its ability to perform feature selection by shrinking less important feature coefficients to exactly zero. This property makes Lasso particularly useful in high-dimensional datasets where identifying relevant features is crucial.

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression

# Generate synthetic data with some irrelevant features
X, y = make_regression(n_samples=100, n_features=20, n_informative=5, noise=0.1, random_state=42)

# Fit Lasso model
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# Display selected features
selected_features = np.where(lasso.coef_ != 0)[0]
print("Selected features:", selected_features)
print("Number of selected features:", len(selected_features))

# Plot feature importance
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(range(len(lasso.coef_)), np.abs(lasso.coef_))
plt.xlabel('Feature Index')
plt.ylabel('Absolute Coefficient Value')
plt.title('Feature Importance (Lasso)')
plt.show()
```

Slide 5: Implementing Lasso from Scratch

To better understand how Lasso works, let's implement a simple version using coordinate descent optimization. This method updates each coefficient individually while holding others fixed.

```python
import numpy as np

def soft_threshold(x, lambda_):
    return np.sign(x) * max(abs(x) - lambda_, 0)

def lasso_coordinate_descent(X, y, lambda_, max_iter=1000, tol=1e-4):
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    
    for _ in range(max_iter):
        beta_old = beta.()
        
        for j in range(n_features):
            X_j = X[:, j]
            r = y - np.dot(X, beta) + beta[j] * X_j
            z_j = np.dot(X_j, r)
            beta[j] = soft_threshold(z_j, lambda_ * n_samples) / (np.dot(X_j, X_j) + 1e-8)
        
        if np.sum(np.abs(beta - beta_old)) < tol:
            break
    
    return beta

# Example usage
np.random.seed(42)
X = np.random.randn(100, 10)
y = np.dot(X[:, :3], np.random.randn(3)) + np.random.randn(100)

beta = lasso_coordinate_descent(X, y, lambda_=0.1)
print("Estimated coefficients:", beta)
```

Slide 6: Choosing the Regularization Parameter (λ)

The regularization parameter λ controls the strength of the L1 penalty. A larger λ leads to more coefficients being set to zero, resulting in a sparser model. Cross-validation is commonly used to select an optimal λ value.

```python
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(42)
X = np.random.randn(200, 20)
y = np.dot(X[:, :5], np.random.randn(5)) + np.random.randn(200)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform cross-validation to select optimal alpha
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)

print("Optimal alpha:", lasso_cv.alpha_)

# Plot MSE for different alphas
plt.figure(figsize=(10, 6))
plt.semilogx(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=-1))
plt.xlabel('Alpha')
plt.ylabel('Mean Square Error')
plt.title('Cross-validation Result')
plt.show()
```

Slide 7: Lasso for Time Series Prediction

Lasso can be effective in time series prediction by selecting relevant lagged features. This example demonstrates how to use Lasso for predicting future values based on past observations.

```python
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Generate sample time series data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
ts = pd.Series(np.cumsum(np.random.randn(len(dates))), index=dates)

# Create lagged features
def create_features(data, lags):
    df = pd.DataFrame(data)
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df.iloc[:, 0].shift(lag)
    return df.dropna()

# Prepare data
df = create_features(ts, lags=30)
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Split data and scale features
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Lasso model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

# Make predictions
y_pred = lasso.predict(X_test_scaled)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.legend()
plt.title('Time Series Prediction using Lasso')
plt.show()
```

Slide 8: Lasso for Image Denoising

Lasso can be applied to image processing tasks such as denoising. In this example, we'll use Lasso to reconstruct a noisy image by promoting sparsity in the wavelet domain.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from pywt import wavedec2, waverec2

def add_noise(image, noise_level):
    return image + noise_level * np.random.randn(*image.shape)

def lasso_denoise(noisy_image, wavelet='db1', level=2, alpha=0.1):
    coeffs = wavedec2(noisy_image, wavelet, level=level)
    
    # Flatten coefficients
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    
    # Apply Lasso
    lasso = Lasso(alpha=alpha)
    coeff_arr_denoised = lasso.fit_transform(coeff_arr.reshape(-1, 1)).reshape(coeff_arr.shape)
    
    # Reconstruct image
    coeffs_denoised = pywt.array_to_coeffs(coeff_arr_denoised, coeff_slices, output_format='wavedec2')
    return waverec2(coeffs_denoised, wavelet)

# Create sample image
x, y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
image = np.sin(5 * np.pi * x) * np.cos(5 * np.pi * y)

# Add noise and denoise
noisy_image = add_noise(image, 0.5)
denoised_image = lasso_denoise(noisy_image)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(noisy_image, cmap='gray')
ax2.set_title('Noisy Image')
ax3.imshow(denoised_image, cmap='gray')
ax3.set_title('Denoised Image (Lasso)')
plt.tight_layout()
plt.show()
```

Slide 9: Lasso for Compressed Sensing

Compressed sensing aims to reconstruct a signal from fewer samples than required by the Nyquist-Shannon sampling theorem. Lasso's sparsity-inducing property makes it suitable for this task.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def create_sparse_signal(n, k):
    signal = np.zeros(n)
    indices = np.random.choice(n, k, replace=False)
    signal[indices] = np.random.randn(k)
    return signal

def compressed_sensing(signal, m):
    n = len(signal)
    A = np.random.randn(m, n)
    y = np.dot(A, signal)
    
    lasso = Lasso(alpha=0.1)
    lasso.fit(A, y)
    
    return lasso.coef_

# Parameters
n = 1000  # Signal length
k = 10    # Number of non-zero components
m = 100   # Number of measurements

# Create and reconstruct signal
original_signal = create_sparse_signal(n, k)
reconstructed_signal = compressed_sensing(original_signal, m)

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.stem(original_signal)
plt.title('Original Sparse Signal')
plt.subplot(212)
plt.stem(reconstructed_signal)
plt.title('Reconstructed Signal (Lasso)')
plt.tight_layout()
plt.show()

# Compute reconstruction error
error = np.linalg.norm(original_signal - reconstructed_signal) / np.linalg.norm(original_signal)
print(f"Relative reconstruction error: {error:.4f}")
```

Slide 10: Elastic Net: Combining L1 and L2 Penalties

Elastic Net is a regularization technique that combines both L1 (Lasso) and L2 (Ridge) penalties. It addresses some limitations of Lasso, such as its behavior in the presence of highly correlated features.

```python
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=100, n_informative=10, noise=0.1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Elastic Net model
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = elastic_net.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(elastic_net.coef_)), abs(elastic_net.coef_))
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Magnitude')
plt.title('Feature Importance (Elastic Net)')
plt.show()
```

Slide 11: Group Lasso

Group Lasso extends the Lasso penalty to groups of features, allowing for simultaneous selection of feature groups. This is particularly useful when dealing with categorical variables or predefined feature groups.

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

def group_lasso(X, y, groups, alpha=1.0, max_iter=1000, tol=1e-4):
    n_samples, n_features = X.shape
    n_groups = len(np.unique(groups))
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize coefficients
    beta = np.zeros(n_features)
    
    for _ in range(max_iter):
        beta_old = beta.()
        
        for g in range(n_groups):
            group_indices = np.where(groups == g)[0]
            X_g = X_scaled[:, group_indices]
            
            # Compute group residuals
            r = y - np.dot(X_scaled, beta) + np.dot(X_g, beta[group_indices])
            
            # Update group coefficients
            z_g = np.dot(X_g.T, r)
            norm_z_g = np.linalg.norm(z_g)
            
            if norm_z_g > alpha:
                beta[group_indices] = (1 - alpha / norm_z_g) * z_g
            else:
                beta[group_indices] = 0
        
        if np.sum(np.abs(beta - beta_old)) < tol:
            break
    
    return beta

# Example usage
np.random.seed(42)
X = np.random.randn(100, 10)
y = np.dot(X[:, [0, 1, 5]], [1, 2, -1]) + np.random.randn(100) * 0.1
groups = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]

beta = group_lasso(X, y, groups, alpha=0.1)
print("Estimated coefficients:", beta)
```

Slide 12: Lasso for Text Classification

Lasso can be applied to text classification tasks by selecting relevant features (words) from a large vocabulary. This example demonstrates using Lasso for sentiment analysis on movie reviews.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Sample movie reviews and sentiments
reviews = [
    "This movie was amazing and entertaining",
    "Terrible plot and poor acting",
    "I loved the characters and storyline",
    "Boring and predictable, waste of time",
    "Great special effects and action scenes"
]
sentiments = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.2, random_state=42)

# Train Lasso model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Make predictions
y_pred = (lasso.predict(X_test) > 0.5).astype(int)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display important words
feature_names = vectorizer.get_feature_names_out()
important_words = sorted(zip(feature_names, lasso.coef_), key=lambda x: abs(x[1]), reverse=True)
print("Top words:", important_words[:5])
```

Slide 13: Lasso for Anomaly Detection

Lasso can be used for anomaly detection by identifying sparse representations of normal data points. Anomalies are then detected based on their reconstruction error.

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def lasso_anomaly_detection(X, alpha=0.1, threshold=0.1):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, X_scaled)
    
    reconstruction = lasso.predict(X_scaled)
    errors = np.mean((X_scaled - reconstruction) ** 2, axis=1)
    
    anomalies = errors > threshold
    return anomalies, errors

# Generate sample data with anomalies
np.random.seed(42)
X_normal = np.random.randn(100, 2)
X_anomalies = np.random.uniform(low=-4, high=4, size=(10, 2))
X = np.vstack([X_normal, X_anomalies])

# Detect anomalies
anomalies, errors = lasso_anomaly_detection(X, alpha=0.1, threshold=0.5)

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(X[~anomalies, 0], X[~anomalies, 1], label='Normal')
plt.scatter(X[anomalies, 0], X[anomalies, 1], color='red', label='Anomaly')
plt.legend()
plt.title('Lasso Anomaly Detection')
plt.show()

print(f"Number of detected anomalies: {np.sum(anomalies)}")
```

Slide 14: Additional Resources

For those interested in diving deeper into Lasso and its applications, here are some valuable resources:

1. "Regularization Paths for Generalized Linear Models via Coordinate Descent" by Friedman et al. (2010) ArXiv: [https://arxiv.org/abs/0708.1485](https://arxiv.org/abs/0708.1485)
2. "The Elements of Statistical Learning" by Hastie et al. (2009) Available online: [https://web.stanford.edu/~hastie/ElemStatLearn/](https://web.stanford.edu/~hastie/ElemStatLearn/)
3. "An Introduction to Statistical Learning" by James et al. (2013) ArXiv: [https://arxiv.org/abs/1315.2737](https://arxiv.org/abs/1315.2737)
4. "Sparse Modeling for Image and Vision Processing" by Mairal et al. (2014) ArXiv: [https://arxiv.org/abs/1411.3230](https://arxiv.org/abs/1411.3230)

These resources provide in-depth explanations of Lasso, its theoretical foundations, and various applications in machine learning and signal processing.

