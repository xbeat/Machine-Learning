## Cross-Validation Fundamentals with Python
Slide 1: Understanding Cross-Validation Fundamentals

Cross-validation is a statistical method used to assess model performance by partitioning data into training and testing sets. It helps evaluate how well a model generalizes to unseen data and reduces overfitting by utilizing multiple rounds of evaluation on different data subsets.

```python
import numpy as np
from sklearn.model_selection import KFold

# Generate sample data
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 1, 0, 1, 0])

# Initialize K-Fold
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Perform K-Fold splits
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}:")
    print(f"Training indices: {train_idx}")
    print(f"Testing indices: {test_idx}\n")
```

Slide 2: Implementing K-Fold Cross-Validation from Scratch

A pure Python implementation of k-fold cross-validation demonstrates the underlying mechanics without relying on external libraries. This approach splits data into k equal-sized folds, using each fold as a validation set while training on the remaining data.

```python
def custom_kfold(X, y, k=5):
    fold_size = len(X) // k
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([
            indices[:test_start],
            indices[test_end:]
        ])
        
        yield X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Example usage
X = np.array([[i] for i in range(10)])
y = np.array([i % 2 for i in range(10)])

for fold, (X_train, X_test, y_train, y_test) in enumerate(custom_kfold(X, y, k=3)):
    print(f"Fold {fold + 1}:")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
```

Slide 3: Stratified K-Fold Cross-Validation

Stratified k-fold ensures that the proportion of samples for each class is preserved in each fold, making it especially useful for imbalanced datasets. This implementation maintains class distribution across all folds.

```python
from sklearn.model_selection import StratifiedKFold
from collections import Counter

# Generate imbalanced dataset
X = np.random.randn(100, 2)
y = np.array([0] * 80 + [1] * 20)  # Imbalanced classes: 80% class 0, 20% class 1

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Fold {fold + 1}:")
    print(f"Training class distribution: {Counter(y_train)}")
    print(f"Testing class distribution: {Counter(y_test)}\n")
```

Slide 4: Time Series Cross-Validation

Time series cross-validation respects temporal ordering by using forward-chaining splits, ensuring future data isn't used to predict past events. This implementation demonstrates time series specific validation techniques.

```python
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

# Generate time series data
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
X = np.random.randn(100, 2)
y = np.random.randn(100)

# Initialize TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold + 1}:")
    print(f"Training: {dates[train_idx[0]]} to {dates[train_idx[-1]]}")
    print(f"Testing: {dates[test_idx[0]]} to {dates[test_idx[-1]]}\n")
```

Slide 5: Leave-One-Out Cross-Validation (LOOCV)

Leave-one-out cross-validation represents the extreme case where k equals the number of samples, using a single observation for validation and the rest for training. This method provides unbiased performance estimates but can be computationally expensive.

```python
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression

# Generate sample data
X = np.array([[i] for i in range(5)])
y = np.array([2*i + 1 for i in range(5)])

loo = LeaveOneOut()
predictions = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions.append((test_idx[0], pred[0], y_test[0]))

for idx, pred, actual in predictions:
    print(f"Sample {idx}: Predicted = {pred:.2f}, Actual = {actual}")
```

Slide 6: Cross-Validation Performance Metrics

Understanding how to calculate and interpret various performance metrics across cross-validation folds is crucial for model evaluation. This implementation demonstrates computation of commonly used metrics.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def calculate_cv_metrics(X, y, model, cv=5):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
        metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
        metrics['f1'].append(f1_score(y_test, y_pred, average='weighted'))
    
    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}

# Example usage
X = np.random.randn(100, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
model = RandomForestClassifier(random_state=42)

results = calculate_cv_metrics(X, y, model)
for metric, (mean, std) in results.items():
    print(f"{metric}: {mean:.3f} (±{std:.3f})")
```

Slide 7: Real-World Application - Housing Price Prediction

Cross-validation applied to a housing price prediction problem demonstrates practical implementation in regression analysis. This example uses California housing data to predict median house values based on multiple features.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_validate
import numpy as np

# Load and prepare data
housing = fetch_california_housing()
X, y = housing.data, housing.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize model with cross-validation
model = LassoCV(cv=5, random_state=42)

# Perform cross-validation with multiple metrics
cv_results = cross_validate(
    model, X_scaled, y,
    cv=5,
    scoring={
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error',
        'mse': 'neg_mean_squared_error'
    },
    return_train_score=True
)

# Print results
for metric in ['r2', 'mae', 'mse']:
    train_scores = cv_results[f'train_{metric}']
    test_scores = cv_results[f'test_{metric}']
    print(f"\n{metric.upper()} Scores:")
    print(f"Train: {np.mean(train_scores):.3f} (±{np.std(train_scores):.3f})")
    print(f"Test: {np.mean(test_scores):.3f} (±{np.std(test_scores):.3f})")
```

Slide 8: Nested Cross-Validation

Nested cross-validation provides unbiased performance estimation while performing hyperparameter tuning, using an inner loop for model selection and an outer loop for evaluation.

```python
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Generate sample dataset
X, y = make_classification(n_samples=200, random_state=42)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Initialize nested cross-validation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Perform nested cross-validation
outer_scores = []

for train_idx, test_idx in outer_cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Inner loop for model selection
    grid_search = GridSearchCV(
        SVC(), param_grid, cv=inner_cv, scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    
    # Evaluate best model on test set
    best_model = grid_search.best_estimator_
    score = best_model.score(X_test, y_test)
    outer_scores.append(score)

print(f"Nested CV Score: {np.mean(outer_scores):.3f} (±{np.std(outer_scores):.3f})")
print(f"Best parameters: {grid_search.best_params_}")
```

Slide 9: Cross-Validation for Time Series Forecasting

This implementation showcases advanced time series cross-validation techniques with expanding window and rolling forecast origin approaches for financial data prediction.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def expanding_window_cv(X, y, min_train_size, horizon=1):
    results = []
    
    for i in range(min_train_size, len(X) - horizon + 1):
        # Training data: all observations up to time i
        X_train = X[:i]
        y_train = y[:i]
        
        # Test data: next horizon observations
        X_test = X[i:i+horizon]
        y_test = y[i:i+horizon]
        
        # Fit and predict
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        results.append({
            'train_size': len(X_train),
            'test_size': len(X_test),
            'pred': pred[0],
            'actual': y_test[0]
        })
    
    return pd.DataFrame(results)

# Generate sample time series data
np.random.seed(42)
n_samples = 100
X = np.arange(n_samples).reshape(-1, 1)
y = 2 * X.ravel() + np.random.randn(n_samples) * 5

# Perform expanding window CV
min_train_size = 30
results = expanding_window_cv(X, y, min_train_size)

# Calculate rolling RMSE
results['error'] = (results['actual'] - results['pred']) ** 2
rolling_rmse = np.sqrt(results['error'].rolling(window=10).mean())

print("Rolling RMSE for last 5 predictions:")
print(rolling_rmse.tail())
```

Slide 10: Monte Carlo Cross-Validation

Monte Carlo cross-validation repeatedly samples random training and test sets, offering insights into model stability and performance variability across different random splits.

```python
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def monte_carlo_cv(X, y, n_iterations=100, test_size=0.2):
    # Initialize
    ss = ShuffleSplit(n_splits=n_iterations, test_size=test_size, random_state=42)
    model = RandomForestClassifier(random_state=42)
    scores = []
    
    # Perform Monte Carlo CV
    for i, (train_idx, test_idx) in enumerate(ss.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)
        
        if (i + 1) % 10 == 0:
            current_mean = np.mean(scores)
            current_std = np.std(scores)
            print(f"Iteration {i+1}: Mean={current_mean:.3f}, Std={current_std:.3f}")
    
    return np.array(scores)

# Generate sample dataset
X, y = make_classification(n_samples=1000, random_state=42)

# Run Monte Carlo CV
scores = monte_carlo_cv(X, y)

print("\nFinal Results:")
print(f"Mean Accuracy: {np.mean(scores):.3f}")
print(f"Std Accuracy: {np.std(scores):.3f}")
print(f"95% CI: [{np.percentile(scores, 2.5):.3f}, {np.percentile(scores, 97.5):.3f}]")
```

Slide 11: Real-World Application - Credit Risk Assessment

Cross-validation applied to credit risk assessment demonstrates practical implementation with imbalanced datasets and cost-sensitive evaluation metrics. This example uses synthesized credit data to predict default probability.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.pipeline import Pipeline
import numpy as np

# Generate synthetic credit data
np.random.seed(42)
n_samples = 1000
n_features = 5

# Create imbalanced dataset (10% defaults)
X = np.random.randn(n_samples, n_features)
y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

# Custom scorer that penalizes false negatives more heavily
beta = 2  # Higher beta means false negatives are penalized more
f_beta_scorer = make_scorer(fbeta_score, beta=beta)

# Create pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Perform stratified cross-validation with custom scoring
cv_results = cross_validate(
    pipeline, X, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring={
        'f_beta': f_beta_scorer,
        'precision': 'precision',
        'recall': 'recall'
    }
)

# Print results
for metric, scores in cv_results.items():
    if metric.startswith('test_'):
        print(f"{metric[5:]}: {scores.mean():.3f} (±{scores.std():.3f})")
```

Slide 12: Cross-Validation with Custom Splits

Implementation of custom splitting strategies for specialized validation scenarios, such as group-based splits or domain-specific validation requirements.

```python
class CustomSplitter:
    def __init__(self, n_splits=5, group_labels=None):
        self.n_splits = n_splits
        self.group_labels = group_labels

    def split(self, X, y=None):
        unique_groups = np.unique(self.group_labels)
        group_kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for group_train_idx, group_test_idx in group_kfold.split(unique_groups):
            train_groups = unique_groups[group_train_idx]
            test_groups = unique_groups[group_test_idx]
            
            train_mask = np.isin(self.group_labels, train_groups)
            test_mask = np.isin(self.group_labels, test_groups)
            
            yield np.where(train_mask)[0], np.where(test_mask)[0]

# Example usage
n_samples = 100
n_groups = 10
X = np.random.randn(n_samples, 2)
y = np.random.randint(0, 2, n_samples)
groups = np.random.randint(0, n_groups, n_samples)

splitter = CustomSplitter(n_splits=5, group_labels=groups)
for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
    print(f"Fold {fold + 1}:")
    print(f"Train groups: {np.unique(groups[train_idx])}")
    print(f"Test groups: {np.unique(groups[test_idx])}\n")
```

Slide 13: Cross-Validation Statistical Analysis

Statistical analysis of cross-validation results using confidence intervals and hypothesis testing to compare model performance across different validation schemes.

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

def analyze_cv_results(cv_scores_dict, alpha=0.05):
    results = {}
    
    for model_name, scores in cv_scores_dict.items():
        # Calculate confidence intervals
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        n = len(scores)
        ci = stats.t.interval(
            1-alpha, 
            n-1,
            loc=mean,
            scale=std/np.sqrt(n)
        )
        
        # Shapiro-Wilk test for normality
        _, normality_p = stats.shapiro(scores)
        
        results[model_name] = {
            'mean': mean,
            'std': std,
            'ci': ci,
            'normal_dist': normality_p > alpha
        }
        
    return results

# Example usage
np.random.seed(42)
cv_scores = {
    'model_a': np.random.normal(0.85, 0.02, 10),
    'model_b': np.random.normal(0.82, 0.03, 10)
}

analysis = analyze_cv_results(cv_scores)
for model, stats_dict in analysis.items():
    print(f"\n{model} Analysis:")
    print(f"Mean: {stats_dict['mean']:.3f}")
    print(f"95% CI: [{stats_dict['ci'][0]:.3f}, {stats_dict['ci'][1]:.3f}]")
    print(f"Normally distributed: {stats_dict['normal_dist']}")
```

Slide 14: Additional Resources

*   Cross-Validation Pitfalls and Improved Practices: [https://arxiv.org/abs/1909.09073](https://arxiv.org/abs/1909.09073)
*   Time Series Cross-Validation: [https://arxiv.org/abs/2104.00864](https://arxiv.org/abs/2104.00864)
*   Nested Cross-Validation Analysis: [https://arxiv.org/abs/2007.08111](https://arxiv.org/abs/2007.08111)
*   For more research papers, search on Google Scholar with keywords: "cross validation machine learning"
*   Scikit-learn Cross-Validation Documentation: [https://scikit-learn.org/stable/modules/cross\_validation.html](https://scikit-learn.org/stable/modules/cross_validation.html)

