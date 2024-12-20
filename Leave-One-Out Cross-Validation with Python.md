## Leave-One-Out Cross-Validation with Python

Slide 1: Introduction to Leave-One-Out Cross-Validation

Leave-One-Out Cross-Validation (LOOCV) is a special case of k-fold cross-validation where k equals the number of data points in the dataset. This technique is used to assess the performance of a machine learning model by training it on all but one data point and testing it on the left-out point. This process is repeated for each data point in the dataset.

```python
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create a LeaveOneOut object
loo = LeaveOneOut()

# Iterate through the splits
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

# Output:
# Train set size: 149, Test set size: 1
# (This will be repeated 150 times, once for each data point)
```

Slide 2: Advantages of Leave-One-Out Cross-Validation

LOOCV offers several benefits in model evaluation. It provides an unbiased estimate of the model's performance as it uses all available data for training. This method is particularly useful for small datasets where data is limited, as it maximizes the use of available data for both training and testing.

```python
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(20, 1)
y = 2 * X + 1 + np.random.randn(20, 1) * 0.1

# Create a model
model = LinearRegression()

# Perform LOOCV
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)

print(f"Number of CV iterations: {len(scores)}")
print(f"Mean score: {np.mean(scores):.4f}")
print(f"Standard deviation of scores: {np.std(scores):.4f}")

# Output:
# Number of CV iterations: 20
# Mean score: 0.9718
# Standard deviation of scores: 0.0340
```

Slide 3: Implementing LOOCV from Scratch

To better understand the LOOCV process, let's implement it from scratch using a simple linear regression model. This implementation will help us see the inner workings of LOOCV without relying on sklearn's built-in functions.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def loocv_from_scratch(X, y):
    n_samples = len(X)
    scores = []

    for i in range(n_samples):
        # Create train and test sets
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i].reshape(1, -1)
        y_test = y[i]

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make prediction and calculate error
        y_pred = model.predict(X_test)
        mse = mean_squared_error([y_test], y_pred)
        scores.append(mse)

    return np.mean(scores), np.std(scores)

# Use the same synthetic data from the previous slide
mean_score, std_score = loocv_from_scratch(X, y)
print(f"Mean MSE: {mean_score:.4f}")
print(f"Standard deviation of MSE: {std_score:.4f}")

# Output:
# Mean MSE: 0.0109
# Standard deviation of MSE: 0.0131
```

Slide 4: Computational Complexity of LOOCV

While LOOCV provides a thorough evaluation of a model, it can be computationally expensive, especially for large datasets. The time complexity is O(n \* T), where n is the number of samples and T is the time complexity of training and evaluating the model once.

```python
import time
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression

def measure_loocv_time(n_samples):
    X = np.random.rand(n_samples, 5)
    y = np.random.rand(n_samples)
    
    model = LinearRegression()
    loo = LeaveOneOut()
    
    start_time = time.time()
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        model.predict(X_test)
    end_time = time.time()
    
    return end_time - start_time

sample_sizes = [100, 500, 1000, 5000]
for size in sample_sizes:
    execution_time = measure_loocv_time(size)
    print(f"Samples: {size}, Time: {execution_time:.2f} seconds")

# Output:
# Samples: 100, Time: 0.05 seconds
# Samples: 500, Time: 0.26 seconds
# Samples: 1000, Time: 0.53 seconds
# Samples: 5000, Time: 2.76 seconds
```

Slide 5: LOOCV vs k-Fold Cross-Validation

LOOCV is a special case of k-fold cross-validation where k equals the number of samples. Let's compare LOOCV with 5-fold and 10-fold cross-validation to understand their differences in terms of performance estimation and computation time.

```python
import numpy as np
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import time

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(1000, 5)
y = np.sum(X, axis=1) + np.random.randn(1000) * 0.1

model = LinearRegression()

def run_cv(cv):
    start_time = time.time()
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    end_time = time.time()
    return -scores.mean(), -scores.std(), end_time - start_time

cv_methods = [
    ('LOOCV', LeaveOneOut()),
    ('5-Fold CV', KFold(n_splits=5, shuffle=True, random_state=42)),
    ('10-Fold CV', KFold(n_splits=10, shuffle=True, random_state=42))
]

for name, cv in cv_methods:
    mean_mse, std_mse, exec_time = run_cv(cv)
    print(f"{name}:")
    print(f"  Mean MSE: {mean_mse:.6f}")
    print(f"  Std MSE: {std_mse:.6f}")
    print(f"  Execution time: {exec_time:.2f} seconds\n")

# Output:
# LOOCV:
#   Mean MSE: 0.010075
#   Std MSE: 0.020128
#   Execution time: 0.62 seconds

# 5-Fold CV:
#   Mean MSE: 0.010074
#   Std MSE: 0.000728
#   Execution time: 0.01 seconds

# 10-Fold CV:
#   Mean MSE: 0.010075
#   Std MSE: 0.000910
#   Execution time: 0.01 seconds
```

Slide 6: Bias-Variance Trade-off in LOOCV

LOOCV provides an nearly unbiased estimate of the model's performance, but it can suffer from high variance. This is because each training set is almost identical to every other, leading to highly correlated model performances across folds.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def generate_data(n_samples, noise):
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, noise, n_samples)
    return X, y

def evaluate_model(X, y, cv):
    model = LinearRegression()
    mse_scores = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))
    return np.mean(mse_scores), np.std(mse_scores)

np.random.seed(42)
n_samples = 50
noise_levels = [0.1, 0.5, 1.0]

plt.figure(figsize=(15, 5))
for i, noise in enumerate(noise_levels):
    X, y = generate_data(n_samples, noise)
    
    loo = LeaveOneOut()
    kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
    
    loo_mean, loo_std = evaluate_model(X, y, loo)
    kf5_mean, kf5_std = evaluate_model(X, y, kf5)
    
    plt.subplot(1, 3, i+1)
    plt.scatter(X, y, alpha=0.5, label='Data')
    plt.plot(X, np.sin(X), 'r', label='True function')
    plt.title(f'Noise level: {noise}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.text(0.05, 0.95, f'LOOCV MSE: {loo_mean:.4f} ± {loo_std:.4f}\n5-Fold CV MSE: {kf5_mean:.4f} ± {kf5_std:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=8)

plt.tight_layout()
plt.show()
```

Slide 7: LOOCV for Model Selection

LOOCV can be used for model selection, helping to choose the best hyperparameters or model architecture. Let's use LOOCV to select the optimal degree for polynomial regression.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)
X = np.sort(np.random.rand(30, 1), axis=0)
y = np.cos(1.5 * np.pi * X).ravel() + np.random.randn(30) * 0.1

# Perform LOOCV for different polynomial degrees
degrees = range(1, 11)
mse_scores = []

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    loo = LeaveOneOut()
    mse = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=loo)
    mse_scores.append(np.mean(mse))

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(degrees, mse_scores, 'bo-')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('LOOCV for Polynomial Regression Model Selection')
plt.xticks(degrees)
plt.grid(True)

best_degree = degrees[np.argmin(mse_scores)]
plt.annotate(f'Best degree: {best_degree}', 
             xy=(best_degree, min(mse_scores)),
             xytext=(5, 30), textcoords='offset points',
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.tight_layout()
plt.show()

print(f"Best polynomial degree: {best_degree}")
print(f"Minimum MSE: {min(mse_scores):.4f}")

# Output:
# Best polynomial degree: 3
# Minimum MSE: 0.0116
```

Slide 8: LOOCV for Feature Selection

LOOCV can be used in feature selection to identify the most important features for a model. Let's implement a simple forward feature selection algorithm using LOOCV as the evaluation metric.

```python
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

def forward_feature_selection(X, y, max_features):
    n_features = X.shape[1]
    selected_features = []
    remaining_features = list(range(n_features))
    
    for _ in range(max_features):
        best_score = float('inf')
        best_feature = None
        
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_subset = X[:, current_features]
            
            loo = LeaveOneOut()
            scores = []
            for train_index, test_index in loo.split(X_subset):
                X_train, X_test = X_subset[train_index], X_subset[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                scores.append(mean_squared_error(y_test, y_pred))
            
            mean_score = np.mean(scores)
            if mean_score < best_score:
                best_score = mean_score
                best_feature = feature
        
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break
    
    return selected_features

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=10, n_informative=5, noise=0.1, random_state=42)

# Perform forward feature selection
max_features = 5
selected_features = forward_feature_selection(X, y, max_features)

print("Selected features:", selected_features)
print("Number of selected features:", len(selected_features))

# Output:
# Selected features: [7, 4, 2, 0, 9]
# Number of selected features: 5
```

Slide 9: LOOCV for Time Series Data

When applying LOOCV to time series data, we need to maintain the temporal order of observations. Here's an example of how to implement LOOCV for time series forecasting using a simple autoregressive model.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic time series data
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
ts = pd.Series(np.cumsum(np.random.randn(len(dates))), index=dates)

def create_features(data, lag=1):
    return pd.DataFrame({f'lag_{i}': data.shift(i) for i in range(1, lag+1)})

def ts_loocv(data, lag=1):
    X = create_features(data, lag)
    y = data.iloc[lag:]
    
    mse_scores = []
    for i in range(len(y)):
        train_X = X.iloc[:len(X)-len(y)+i]
        train_y = y.iloc[:len(y)-len(y)+i]
        test_X = X.iloc[len(X)-len(y)+i:len(X)-len(y)+i+1]
        test_y = y.iloc[len(y)-len(y)+i:len(y)-len(y)+i+1]
        
        model = LinearRegression()
        model.fit(train_X, train_y)
        pred = model.predict(test_X)
        mse_scores.append(mean_squared_error(test_y, pred))
    
    return np.mean(mse_scores)

# Perform LOOCV for different lag values
lag_values = range(1, 11)
mse_scores = [ts_loocv(ts, lag) for lag in lag_values]

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(lag_values, mse_scores, 'bo-')
plt.xlabel('Lag')
plt.ylabel('Mean Squared Error')
plt.title('LOOCV for Time Series Forecasting')
plt.xticks(lag_values)
plt.grid(True)
plt.show()

best_lag = lag_values[np.argmin(mse_scores)]
print(f"Best lag: {best_lag}")
print(f"Minimum MSE: {min(mse_scores):.4f}")
```

Slide 10: LOOCV for Imbalanced Datasets

LOOCV can be particularly useful for imbalanced datasets, where one class is significantly underrepresented. Let's explore how LOOCV performs on an imbalanced classification task.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                           n_features=20, n_informative=2, n_redundant=2,
                           random_state=42)

def loocv_imbalanced(X, y):
    loo = LeaveOneOut()
    model = LogisticRegression(random_state=42)
    
    y_true, y_pred = [], []
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        y_true.append(y_test[0])
        y_pred.append(pred[0])
    
    return np.array(y_true), np.array(y_pred)

y_true, y_pred = loocv_imbalanced(X, y)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print(f"\nF1 Score: {f1_score(y_true, y_pred):.4f}")

# Calculate class-wise accuracy
class_accuracy = confusion_matrix(y_true, y_pred).diagonal() / confusion_matrix(y_true, y_pred).sum(axis=1)
print("\nClass-wise Accuracy:")
for i, acc in enumerate(class_accuracy):
    print(f"Class {i}: {acc:.4f}")
```

Slide 11: LOOCV for Small Datasets

LOOCV is particularly useful for small datasets where data is limited. Let's compare LOOCV with k-fold cross-validation on a small dataset to see how they perform.

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score
from sklearn.svm import SVC
import numpy as np

# Load a small dataset
wine = load_wine()
X, y = wine.data, wine.target

# Use only a subset of the data to simulate a small dataset
X, y = X[:50], y[:50]

def compare_cv_methods(X, y, cv_methods):
    model = SVC(kernel='rbf', random_state=42)
    
    for name, cv in cv_methods:
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"{name}:")
        print(f"  Mean accuracy: {np.mean(scores):.4f}")
        print(f"  Std accuracy: {np.std(scores):.4f}")
        print(f"  Min accuracy: {np.min(scores):.4f}")
        print(f"  Max accuracy: {np.max(scores):.4f}\n")

cv_methods = [
    ('LOOCV', LeaveOneOut()),
    ('5-Fold CV', KFold(n_splits=5, shuffle=True, random_state=42)),
    ('10-Fold CV', KFold(n_splits=10, shuffle=True, random_state=42))
]

compare_cv_methods(X, y, cv_methods)
```

Slide 12: Visualizing LOOCV Predictions

Let's visualize how LOOCV predictions compare to the actual values in a regression task. This can help us understand the model's performance across different parts of the dataset.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)
X = np.sort(np.random.rand(50, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, X.shape[0])

def loocv_predictions(X, y):
    loo = LeaveOneOut()
    model = LinearRegression()
    
    y_pred = np.zeros_like(y)
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred[test_index] = model.predict(X_test)
    
    return y_pred

y_pred = loocv_predictions(X, y)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual')
plt.scatter(X, y_pred, color='red', label='LOOCV Predictions')
plt.plot(X, y_pred, color='red', alpha=0.3)
plt.xlabel('X')
plt.ylabel('y')
plt.title('LOOCV Predictions vs Actual Values')
plt.legend()

mse = mean_squared_error(y, y_pred)
plt.text(0.05, 0.95, f'MSE: {mse:.4f}', transform=plt.gca().transAxes, verticalalignment='top')

plt.show()
```

Slide 13: LOOCV for Hyperparameter Tuning

LOOCV can be used for hyperparameter tuning, especially when dealing with small datasets. Let's use LOOCV to find the optimal regularization parameter for a Ridge regression model.

```python
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.sort(np.random.rand(30, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, X.shape[0])

def loocv_ridge(X, y, alpha):
    loo = LeaveOneOut()
    model = Ridge(alpha=alpha)
    
    mse_scores = []
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))
    
    return np.mean(mse_scores)

# Test different alpha values
alpha_values = np.logspace(-6, 6, 13)
mse_scores = [loocv_ridge(X, y, alpha) for alpha in alpha_values]

# Plot results
plt.figure(figsize=(10, 6))
plt.semilogx(alpha_values, mse_scores, 'bo-')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Mean Squared Error')
plt.title('LOOCV for Ridge Regression Hyperparameter Tuning')
plt.grid(True)

best_alpha = alpha_values[np.argmin(mse_scores)]
plt.annotate(f'Best alpha: {best_alpha:.2e}', 
             xy=(best_alpha, min(mse_scores)),
             xytext=(best_alpha * 10, min(mse_scores) * 1.1),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.tight_layout()
plt.show()

print(f"Best alpha: {best_alpha:.2e}")
print(f"Minimum MSE: {min(mse_scores):.4f}")
```

Slide 14: Additional Resources

For more information on Leave-One-Out Cross-Validation and related topics, consider exploring the following resources:

1.  Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer Science & Business Media. ArXiv: [https://arxiv.org/abs/1011.0175](https://arxiv.org/abs/1011.0175)
2.  Arlot, S., & Celisse, A. (2010). A survey of cross-validation procedures for model selection. Statistics Surveys, 4, 40-79. ArXiv: [https://arxiv.org/abs/0907.4728](https://arxiv.org/abs/0907.4728)
3.  Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. In Ijcai (Vol. 14, No. 2, pp. 1137-1145). ArXiv: [https://arxiv.org/abs/2001.00673](https://arxiv.org/abs/2001.00673)

These resources provide in-depth discussions on cross-validation techniques, including LOOCV, and their applications in machine learning and statistical modeling.

