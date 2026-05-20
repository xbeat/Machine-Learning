## K-Fold Cross Validation with Python
Slide 1: K-Fold Cross Validation: An Introduction

K-Fold Cross Validation is a statistical method used to estimate the performance of machine learning models. It involves dividing the dataset into K subsets, or folds, and iteratively using each fold as a validation set while the remaining folds serve as the training set. This technique helps to reduce overfitting and provides a more robust estimate of model performance.

```python
import numpy as np
from sklearn.model_selection import KFold

# Create a sample dataset
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 1, 0, 1, 0])

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform K-Fold split
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}:")
    print(f"  Train indices: {train_index}")
    print(f"  Validation indices: {val_index}")
```

Slide 2: Understanding K=5 in K-Fold Cross Validation

When K=5 in K-Fold Cross Validation, it means that the dataset is divided into 5 equal-sized folds. Each fold contains approximately 20% of the original dataset. The process involves 5 iterations, where each fold serves as the validation set once while the remaining 4 folds are used for training.

```python
import numpy as np
from sklearn.model_selection import KFold

# Create a sample dataset with 25 data points
X = np.arange(25).reshape(-1, 1)
y = np.arange(25)

# Initialize KFold with K=5
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform K-Fold split and print the sizes of each fold
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}:")
    print(f"  Train set size: {len(train_index)}")
    print(f"  Validation set size: {len(val_index)}")
```

Slide 3: The K-Fold Cross Validation Process

The K-Fold Cross Validation process involves several steps:

1. Divide the dataset into K equal-sized folds.
2. For each iteration (K times): a. Use one fold as the validation set. b. Use the remaining K-1 folds as the training set. c. Train the model on the training set. d. Evaluate the model on the validation set.
3. Calculate the average performance across all K iterations.

Slide 4: The K-Fold Cross Validation Process

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a sample dataset
X = np.array([[i] for i in range(50)])
y = 2 * X + np.random.randn(50, 1)

# Initialize KFold and model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LinearRegression()

# Perform K-Fold Cross Validation
mse_scores = []
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mse_scores.append(mse)

print(f"Mean MSE: {np.mean(mse_scores)}")
print(f"Standard Deviation of MSE: {np.std(mse_scores)}")
```

Slide 5: Advantages of K-Fold Cross Validation

K-Fold Cross Validation offers several benefits:

1. Reduced bias: By using multiple train-validation splits, it provides a more accurate estimate of model performance.
2. Efficient use of data: Every data point is used for both training and validation, making it suitable for small datasets.
3. Robust performance estimation: It helps detect overfitting and provides insight into model stability.

Slide 6: Advantages of K-Fold Cross Validation

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Create a sample dataset
X = np.array([[i] for i in range(100)])
y = 2 * X + np.random.randn(100, 1)

# Initialize model
model = LinearRegression()

# Perform K-Fold Cross Validation
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Convert scores to positive MSE
mse_scores = -scores

print(f"MSE scores: {mse_scores}")
print(f"Mean MSE: {np.mean(mse_scores)}")
print(f"Standard Deviation of MSE: {np.std(mse_scores)}")
```

Slide 7: Implementing K-Fold Cross Validation from Scratch

To better understand the mechanics of K-Fold Cross Validation, let's implement it from scratch without using scikit-learn's built-in functions.

```python
import numpy as np

def k_fold_cross_validation(X, y, k, model):
    fold_size = len(X) // k
    scores = []
    
    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size
        
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        X_train = np.concatenate([X[:val_start], X[val_end:]])
        y_train = np.concatenate([y[:val_start], y[val_end:]])
        
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

# Example usage
X = np.array([[i] for i in range(100)])
y = 2 * X + np.random.randn(100, 1)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

mean_score, std_score = k_fold_cross_validation(X, y, k=5, model=model)
print(f"Mean R² score: {mean_score:.4f}")
print(f"Standard deviation of R² score: {std_score:.4f}")
```

Slide 8: Stratified K-Fold Cross Validation

Stratified K-Fold Cross Validation is a variation that ensures each fold has approximately the same proportion of samples for each class as the complete dataset. This is particularly useful for imbalanced datasets or classification problems.

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform Stratified K-Fold Cross Validation
accuracies = []
for train_index, val_index in skf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    accuracies.append(accuracy)

print(f"Mean accuracy: {np.mean(accuracies):.4f}")
print(f"Standard deviation of accuracy: {np.std(accuracies):.4f}")
```

Slide 9: Choosing the Right K Value

Selecting the appropriate K value is crucial for K-Fold Cross Validation. Common choices are 5 and 10, but the optimal value depends on the dataset size and problem complexity. Let's explore how different K values affect the results.

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Create a dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Initialize model
model = LogisticRegression(random_state=42)

# Test different K values
k_values = [3, 5, 10, 20]
for k in k_values:
    scores = cross_val_score(model, X, y, cv=k, scoring='accuracy')
    print(f"K={k}:")
    print(f"  Mean accuracy: {np.mean(scores):.4f}")
    print(f"  Standard deviation: {np.std(scores):.4f}")
```

Slide 10: Handling Time Series Data with K-Fold Cross Validation

When dealing with time series data, traditional K-Fold Cross Validation can lead to data leakage. Time Series Split is a variation that respects the temporal order of the data.

```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a simple time series dataset
np.random.seed(42)
X = np.array(range(100)).reshape(-1, 1)
y = np.sin(X * 0.1) + np.random.randn(100, 1) * 0.1

# Initialize TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Perform Time Series Cross Validation
mse_scores = []
for train_index, val_index in tscv.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mse_scores.append(mse)

print(f"Mean MSE: {np.mean(mse_scores):.4f}")
print(f"Standard Deviation of MSE: {np.std(mse_scores):.4f}")
```

Slide 11: Visualizing K-Fold Cross Validation

Visualizing the K-Fold Cross Validation process can help in understanding how the data is split and used. Let's create a simple visualization of the folds.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Create a sample dataset
X = np.arange(50).reshape(-1, 1)
y = np.arange(50)

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create a figure
plt.figure(figsize=(12, 6))

# Plot each fold
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    plt.scatter(X[train_index], [fold] * len(train_index), c='blue', alpha=0.6, s=20, label='Train' if fold == 0 else "")
    plt.scatter(X[val_index], [fold] * len(val_index), c='red', alpha=0.6, s=20, label='Validation' if fold == 0 else "")

plt.yticks(range(5), [f"Fold {i+1}" for i in range(5)])
plt.xlabel("Data points")
plt.title("K-Fold Cross Validation (K=5)")
plt.legend()
plt.tight_layout()
plt.show()
```

Slide 12: Real-Life Example: Predicting Housing Prices

Let's apply K-Fold Cross Validation to a real-world problem: predicting housing prices using the Boston Housing dataset.

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the Boston Housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the model
model = Ridge(alpha=1.0)

# Perform K-Fold Cross Validation
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')

# Convert scores to positive MSE and then to RMSE
rmse_scores = np.sqrt(-scores)

print(f"RMSE scores: {rmse_scores}")
print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
print(f"Standard Deviation of RMSE: {np.std(rmse_scores):.4f}")
```

Slide 13: Real-Life Example: Iris Flower Classification

Let's apply K-Fold Cross Validation to another real-world problem: classifying iris flowers based on their features.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the model
model = SVC(kernel='rbf', random_state=42)

# Perform K-Fold Cross Validation
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

print(f"Accuracy scores: {scores}")
print(f"Mean accuracy: {np.mean(scores):.4f}")
print(f"Standard Deviation of accuracy: {np.std(scores):.4f}")
```

Slide 14: Hyperparameter Tuning with K-Fold Cross Validation

K-Fold Cross Validation is often used in conjunction with hyperparameter tuning to find the best model configuration. Let's use GridSearchCV to demonstrate this process.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Load the Breast Cancer dataset
X, y = load_breast_cancer(return_X_y=True)

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}

# Initialize the model
svm = SVC(random_state=42)

# Perform Grid Search with K-Fold Cross Validation
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_scaled, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
```

Slide 15: Limitations and Considerations of K-Fold Cross Validation

While K-Fold Cross Validation is a powerful technique, it's important to be aware of its limitations:

1. Computational cost: It can be time-consuming for large datasets or complex models.
2. Randomness: Results can vary due to random splitting, especially with small datasets.
3. Assumption of independence: It assumes that data points are independent, which may not hold for time series or spatially correlated data.
4. Bias-variance trade-off: The choice of K affects the bias-variance trade-off of the performance estimate.

Slide 16: Limitations and Considerations of K-Fold Cross Validation

```python
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Create a dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Initialize model
model = LogisticRegression(random_state=42)

# Compare different K values
k_values = [3, 5, 10, 20]
for k in k_values:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf)
    print(f"K={k}:")
    print(f"  Mean accuracy: {np.mean(scores):.4f}")
    print(f"  Standard deviation: {np.std(scores):.4f}")
```

Slide 17: Best Practices for K-Fold Cross Validation

To maximize the benefits of K-Fold Cross Validation, consider these best practices:

1. Choose an appropriate K value based on your dataset size and computational resources.
2. Use stratified sampling for classification problems to maintain class balance across folds.
3. Shuffle the data before splitting to ensure randomness, especially for ordered datasets.
4. Combine K-Fold Cross Validation with other techniques like nested cross-validation for unbiased model selection.

Slide 18: Best Practices for K-Fold Cross Validation

```python
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris

# Load dataset
X, y = load_iris(return_X_y=True)

# Create a pipeline with preprocessing and model
pipeline = make_pipeline(StandardScaler(), SVC(random_state=42))

# Perform cross-validation with multiple metrics
cv_results = cross_validate(pipeline, X, y, cv=5, 
                            scoring=['accuracy', 'f1_weighted'],
                            return_train_score=True)

print("Test Accuracy: {:.4f} (+/- {:.4f})".format(
    cv_results['test_accuracy'].mean(), 
    cv_results['test_accuracy'].std() * 2
))
print("Test F1-score: {:.4f} (+/- {:.4f})".format(
    cv_results['test_f1_weighted'].mean(), 
    cv_results['test_f1_weighted'].std() * 2
))
```

Slide 19: Additional Resources

For further reading on K-Fold Cross Validation and related topics, consider these resources:

1. ArXiv paper: "A Survey of Cross-Validation Procedures for Model Selection" by Arlot, S. and Celisse, A. (2010) URL: [https://arxiv.org/abs/0907.4728](https://arxiv.org/abs/0907.4728)
2. ArXiv paper: "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure" by Roberts, D. R. et al. (2017) URL: [https://arxiv.org/abs/1706.07592](https://arxiv.org/abs/1706.07592)

These papers provide in-depth discussions on cross-validation techniques and their applications in various domains.

