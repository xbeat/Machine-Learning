## Bagging in Ensemble Learning
Slide 1: Understanding Bagging in Ensemble Learning

Bagging (Bootstrap Aggregating) is a fundamental ensemble learning technique that creates multiple training datasets through bootstrap sampling, training individual models on these samples, and combining their predictions to reduce overfitting and variance in the final model's predictions.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

class SimpleBagging:
    def __init__(self, base_estimator, n_estimators=10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = []
        
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]
```

Slide 2: Implementing Core Bagging Methods

The implementation focuses on creating an ensemble of decision trees, where each tree is trained on a bootstrap sample of the original dataset. This approach introduces diversity among base learners while maintaining the original distribution's properties.

```python
    def fit(self, X, y):
        self.estimators = []
        
        for _ in range(self.n_estimators):
            estimator = clone(self.base_estimator)
            X_sample, y_sample = self.bootstrap_sample(X, y)
            estimator.fit(X_sample, y_sample)
            self.estimators.append(estimator)
            
        return self
            
    def predict(self, X):
        predictions = np.array([
            estimator.predict(X) for estimator in self.estimators
        ])
        return np.mean(predictions, axis=0)
```

Slide 3: Mathematical Foundation of Bagging

The theoretical basis of bagging relies on statistical principles of variance reduction through averaging independent estimators. Understanding these fundamentals is crucial for effective implementation and optimization.

```python
# Mathematical representation of bagging prediction
'''
For regression:
$$f_{bag}(x) = \frac{1}{B} \sum_{b=1}^{B} f_b(x)$$

For classification:
$$C_{bag}(x) = \text{argmax}_y \sum_{b=1}^{B} I(f_b(x) = y)$$

Where:
- B is number of bootstrap samples
- f_b is the predictor built with bootstrap sample b
- I() is the indicator function
'''
```

Slide 4: Creating Bootstrap Samples

Implementing the bootstrap sampling mechanism forms the foundation of bagging. This process involves random sampling with replacement to create diverse training sets for individual models.

```python
def generate_bootstrap_sample(X, y, sample_size=None):
    if sample_size is None:
        sample_size = len(X)
        
    indices = np.random.randint(0, len(X), size=sample_size)
    X_bootstrap = X[indices]
    y_bootstrap = y[indices]
    
    # Calculate out-of-bag indices
    oob_indices = np.array(list(set(range(len(X))) - set(indices)))
    
    return X_bootstrap, y_bootstrap, oob_indices
```

Slide 5: Implementing Random Forest from Scratch

Random Forest extends the bagging concept by introducing feature randomization at each split, creating an even more robust ensemble learning method that combines both bagging and feature selection.

```python
class SimpleRandomForest:
    def __init__(self, n_estimators=100, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees = []
        
    def _get_max_features(self, n_features):
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                return int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                return int(np.log2(n_features))
        return self.max_features
```

Slide 6: Source Code for Random Forest Implementation

```python
    def fit(self, X, y):
        n_features = X.shape[1]
        max_features = self._get_max_features(n_features)
        
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_features=max_features,
                splitter='random'
            )
            X_sample, y_sample, _ = generate_bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
        return self
        
    def predict(self, X):
        predictions = np.array([
            tree.predict(X) for tree in self.trees
        ])
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )
```

Slide 7: Real-world Application - Credit Card Fraud Detection

Banking fraud detection represents a perfect use case for bagging techniques due to its inherent class imbalance and the need for robust prediction models that can handle noisy data.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd

# Simulate credit card transaction data
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    weights=[0.97, 0.03],  # Imbalanced classes
    random_state=42
)

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Slide 8: Source Code for Fraud Detection Implementation

```python
# Initialize and train bagging classifier
bagging_classifier = SimpleBagging(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100
)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train and evaluate
bagging_classifier.fit(X_train, y_train)
y_pred = bagging_classifier.predict(X_test)

# Print results
print(classification_report(y_test, y_pred))

# Output example:
'''
              precision    recall  f1-score   support
           0       0.98      0.99      0.98      1940
           1       0.85      0.79      0.82        60
    accuracy                           0.98      2000
'''
```

Slide 9: Results Analysis for Fraud Detection

Performance analysis of the bagging classifier reveals its effectiveness in handling imbalanced datasets and reducing false positives, which is crucial in fraud detection applications.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    return cm

# Calculate and display metrics
cm = plot_confusion_matrix(y_test, y_pred)
print(f"True Negatives: {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}")
print(f"True Positives: {cm[1][1]}")
```

Slide 10: Out-of-Bag Error Estimation

Out-of-Bag (OOB) error estimation provides an unbiased estimate of the generalization error without requiring a separate validation set, making it a valuable tool for model evaluation.

```python
class BaggingWithOOB:
    def __init__(self, base_estimator, n_estimators=10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = []
        self.oob_score_ = None
        
    def _calculate_oob_score(self, X, y):
        predictions = np.zeros((X.shape[0], self.n_estimators))
        n_predictions = np.zeros(X.shape[0])
        
        for i, estimator in enumerate(self.estimators):
            oob_idx = self.oob_indices_[i]
            predictions[oob_idx, i] = estimator.predict(X[oob_idx])
            n_predictions[oob_idx] += 1
```

Slide 11: Feature Importance in Bagging

Understanding feature importance in bagging ensembles helps identify the most relevant predictors and can guide feature selection decisions in model optimization.

```python
def calculate_feature_importance(self, X, y):
    importances = np.zeros(X.shape[1])
    
    for estimator in self.estimators:
        if hasattr(estimator, 'feature_importances_'):
            importances += estimator.feature_importances_
            
    return importances / self.n_estimators

def plot_feature_importance(importances, feature_names=None):
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]
        
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importances)
    plt.xticks(rotation=45)
    plt.title('Feature Importance')
    plt.tight_layout()
```

Slide 12: Parallel Implementation of Bagging

Modern implementations of bagging algorithms leverage parallel processing to improve computational efficiency, especially important when dealing with large datasets or many base estimators.

```python
from joblib import Parallel, delayed

class ParallelBagging:
    def __init__(self, base_estimator, n_estimators=10, n_jobs=-1):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        
    def _parallel_fit(self, X, y):
        estimator = clone(self.base_estimator)
        X_sample, y_sample = self.bootstrap_sample(X, y)
        return estimator.fit(X_sample, y_sample)
        
    def fit(self, X, y):
        self.estimators = Parallel(n_jobs=self.n_jobs)(
            delayed(self._parallel_fit)(X, y)
            for _ in range(self.n_estimators)
        )
        return self
```

Slide 13: Advanced Bagging Variations

Modern variations of bagging incorporate techniques like weighted voting, adaptive resampling, and specialized aggregation methods to improve performance on specific types of problems.

```python
class WeightedBagging:
    def __init__(self, base_estimator, n_estimators=10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = []
        self.weights = []
        
    def _calculate_weight(self, estimator, X_val, y_val):
        predictions = estimator.predict(X_val)
        return np.mean(predictions == y_val)
        
    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2
        )
        
        for _ in range(self.n_estimators):
            estimator = clone(self.base_estimator)
            X_sample, y_sample = self.bootstrap_sample(X_train, y_train)
            estimator.fit(X_sample, y_sample)
            
            weight = self._calculate_weight(estimator, X_val, y_val)
            self.estimators.append(estimator)
            self.weights.append(weight)
            
        return self
```

Slide 14: Additional Resources

*   Search terms for academic papers:
    *   "Bagging Predictors" by Leo Breiman
    *   "Random Forests" original paper
    *   "Out-of-bag estimation" methodology
*   Recommended resources:
    *   [https://scikit-learn.org/stable/modules/ensemble.html](https://scikit-learn.org/stable/modules/ensemble.html)
    *   [https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
    *   [https://www.sciencedirect.com/science/article/abs/pii/S0167947307003076](https://www.sciencedirect.com/science/article/abs/pii/S0167947307003076)

