## Evaluating Machine Learning Models with AUC-ROC
Slide 1: Area Under ROC Curve (AUC-ROC) - Core Concept

The Area Under the Receiver Operating Characteristic curve represents the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance. It's particularly valuable for imbalanced classification problems and threshold-independent evaluation.

```python
import numpy as np
from sklearn.metrics import roc_curve, auc

def calculate_auc_roc(y_true, y_pred_proba):
    # Calculate ROC curve points
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr

# Example usage
y_true = np.array([0, 0, 1, 1, 1, 0])
y_scores = np.array([0.1, 0.2, 0.8, 0.7, 0.9, 0.3])
auc_score, fpr, tpr = calculate_auc_roc(y_true, y_scores)
print(f"AUC-ROC Score: {auc_score:.3f}")
```

Slide 2: AUC-ROC Mathematical Foundation

The mathematical foundation of AUC-ROC relies on the concept of ranking and probability theory. The formal definition involves integrating the true positive rate with respect to the false positive rate across all possible classification thresholds.

```python
# Mathematical representation in LaTeX (not rendered)
"""
$$AUC = \int_{0}^{1} TPR(FPR^{-1}(t))dt$$

Where:
$$TPR = \frac{TP}{TP + FN}$$
$$FPR = \frac{FP}{FP + TN}$$
"""
```

Slide 3: Implementing AUC-ROC from Scratch

Understanding the core implementation helps grasp the underlying mechanics of AUC-ROC calculation. This implementation avoids using sklearn and demonstrates the mathematical concepts directly through Python code.

```python
def auc_roc_from_scratch(y_true, y_pred_proba):
    # Sort predictions and corresponding true values
    pairs = sorted(zip(y_pred_proba, y_true), reverse=True)
    pos = sum(y_true)  # Number of positive examples
    neg = len(y_true) - pos  # Number of negative examples
    
    fp, tp = 0, 0
    xy_coords = [(0,0)]
    
    # Calculate coordinates for ROC curve
    for pred, actual in pairs:
        if actual == 1:
            tp += 1
        else:
            fp += 1
        xy_coords.append((fp/neg, tp/pos))
    
    # Calculate area using trapezoidal rule
    auc = 0
    for i in range(1, len(xy_coords)):
        x_diff = xy_coords[i][0] - xy_coords[i-1][0]
        y_avg = (xy_coords[i][1] + xy_coords[i-1][1]) / 2
        auc += x_diff * y_avg
    
    return auc

# Example usage
y_true = [1, 0, 1, 1, 0]
y_scores = [0.9, 0.1, 0.8, 0.7, 0.3]
print(f"AUC-ROC Score: {auc_roc_from_scratch(y_true, y_scores):.3f}")
```

Slide 4: Real-world Application - Credit Card Fraud Detection

In fraud detection, AUC-ROC is crucial due to highly imbalanced class distributions. This implementation demonstrates preprocessing, model training, and evaluation using a practical example with synthetic data.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Generate synthetic fraud detection data
np.random.seed(42)
n_samples = 10000
fraud_ratio = 0.02

# Create features
X = np.random.randn(n_samples, 4)  # 4 features: amount, time, V1, V2
y = np.random.choice([0, 1], size=n_samples, p=[1-fraud_ratio, fraud_ratio])

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model and get probabilities
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train_scaled, y_train)
y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

# Calculate AUC-ROC
auc_score = calculate_auc_roc(y_test, y_pred_proba)[0]
print(f"Fraud Detection AUC-ROC: {auc_score:.3f}")
```

Slide 5: Cross-Validation with AUC-ROC

Cross-validation provides a more robust estimate of model performance by evaluating AUC-ROC across multiple data splits. This implementation demonstrates stratified k-fold cross-validation to maintain class distribution across folds.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

def cv_auc_roc(X, y, model, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        auc_scores.append(auc)
        print(f"Fold {fold+1} AUC-ROC: {auc:.3f}")
    
    return np.mean(auc_scores), np.std(auc_scores)

# Example usage
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
mean_auc, std_auc = cv_auc_roc(X, y, model)
print(f"\nMean AUC-ROC: {mean_auc:.3f} (±{std_auc:.3f})")
```

Slide 6: Comparing Multiple Models Using AUC-ROC

AUC-ROC serves as an excellent metric for comparing different models' performance, especially when dealing with imbalanced datasets. This implementation demonstrates comparison across multiple classifiers.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

def compare_models_auc(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier(),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        results[name] = auc
        print(f"{name} AUC-ROC: {auc:.3f}")
    
    return results

# Example usage with previous data
results = compare_models_auc(X_train_scaled, X_test_scaled, y_train, y_test)
```

Slide 7: Handling Multiclass AUC-ROC

When dealing with multiclass problems, AUC-ROC can be calculated using one-vs-rest or one-vs-one approaches. This implementation shows how to handle multiclass scenarios effectively.

```python
from sklearn.preprocessing import label_binarize
from itertools import combinations

def multiclass_auc_roc(y_true, y_pred_proba, average='macro'):
    """
    Calculate multiclass AUC-ROC scores
    
    Parameters:
    - y_true: True labels
    - y_pred_proba: Predicted probabilities for each class
    - average: 'macro' or 'weighted'
    """
    n_classes = y_pred_proba.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Calculate AUC-ROC for each class
    auc_scores = []
    for i in range(n_classes):
        auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
        auc_scores.append(auc)
    
    if average == 'macro':
        return np.mean(auc_scores)
    elif average == 'weighted':
        class_weights = np.sum(y_true_bin, axis=0) / len(y_true)
        return np.sum(auc_scores * class_weights)

# Example usage with multiclass data
from sklearn.datasets import make_classification
X_multi, y_multi = make_classification(n_classes=3, n_informative=3, n_redundant=0, random_state=42)
clf = RandomForestClassifier()
y_pred_proba = clf.fit(X_train, y_train).predict_proba(X_test)
print(f"Multiclass AUC-ROC: {multiclass_auc_roc(y_test, y_pred_proba):.3f}")
```

Slide 8: Real-world Application - Medical Diagnosis

Medical diagnosis often requires high-precision classification where AUC-ROC helps evaluate model reliability. This implementation uses a synthetic dataset mimicking medical diagnostic features.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Generate synthetic medical data
np.random.seed(42)
n_patients = 1000

# Create features: age, blood_pressure, glucose, cholesterol
data = {
    'age': np.random.normal(60, 10, n_patients),
    'blood_pressure': np.random.normal(130, 20, n_patients),
    'glucose': np.random.normal(100, 25, n_patients),
    'cholesterol': np.random.normal(200, 30, n_patients)
}

df = pd.DataFrame(data)
# Generate target (disease presence)
baseline_risk = 0.3
df['disease'] = (baseline_risk + 
                 0.2 * (df['age'] > 70).astype(int) +
                 0.3 * (df['blood_pressure'] > 140).astype(int) +
                 0.2 * (df['glucose'] > 120).astype(int))
df['disease'] = (df['disease'] > 0.5).astype(int)

# Prepare data
X = df.drop('disease', axis=1)
y = df['disease']

# Train and evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = GradientBoostingClassifier()
model.fit(X_train_scaled, y_train)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"Medical Diagnosis AUC-ROC: {auc_score:.3f}")
```

Slide 9: AUC-ROC with Confidence Intervals

Understanding the uncertainty in AUC-ROC estimates is crucial for model evaluation. This implementation uses bootstrap sampling to calculate confidence intervals.

```python
from scipy import stats

def auc_roc_ci(y_true, y_pred_proba, n_bootstraps=1000, confidence_level=0.95):
    n_samples = len(y_true)
    bootstrapped_scores = []
    
    for i in range(n_bootstraps):
        # Bootstrap sampling with replacement
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred_proba[indices]
        
        score = roc_auc_score(y_true_bootstrap, y_pred_bootstrap)
        bootstrapped_scores.append(score)
    
    # Calculate confidence intervals
    alpha = (1 - confidence_level) / 2
    ci_lower = np.percentile(bootstrapped_scores, alpha * 100)
    ci_upper = np.percentile(bootstrapped_scores, (1 - alpha) * 100)
    
    return np.mean(bootstrapped_scores), (ci_lower, ci_upper)

# Example usage
auc_mean, (ci_lower, ci_upper) = auc_roc_ci(y_test, y_pred_proba)
print(f"AUC-ROC: {auc_mean:.3f} ({confidence_level*100}% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
```

Slide 10: Time Series Evaluation with AUC-ROC

When evaluating time series models, special consideration must be given to temporal dependencies. This implementation shows how to properly evaluate sequential predictions.

```python
def time_series_auc_roc(X, y, model, test_size=0.2, gap=5):
    """
    Evaluate time series model with AUC-ROC using forward chaining
    
    Parameters:
    - gap: Number of time steps between train and test to avoid leakage
    """
    split_idx = int(len(X) * (1 - test_size))
    
    # Time-based split
    X_train = X[:split_idx-gap]
    y_train = y[:split_idx-gap]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    # Train model
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    return auc, y_pred_proba, y_test

# Generate time series data
np.random.seed(42)
n_samples = 1000
X_ts = np.column_stack([
    np.sin(np.linspace(0, 10*np.pi, n_samples)),
    np.random.randn(n_samples)
])
y_ts = (X_ts[:, 0] > 0).astype(int)

# Evaluate
model = RandomForestClassifier()
auc, y_pred, y_true = time_series_auc_roc(X_ts, y_ts, model)
print(f"Time Series AUC-ROC: {auc:.3f}")
```

Slide 11: Optimal Threshold Selection Using AUC-ROC

AUC-ROC curve analysis can help determine the optimal classification threshold based on specific business requirements and cost functions. This implementation demonstrates threshold optimization techniques.

```python
def find_optimal_threshold(y_true, y_pred_proba, cost_ratio=1.0):
    """
    Find optimal threshold based on cost ratio between false positives and false negatives
    cost_ratio: Cost of false negative / Cost of false positive
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Calculate costs for each threshold
    n_neg = sum(y_true == 0)
    n_pos = sum(y_true == 1)
    costs = (fpr * n_neg + cost_ratio * (1 - tpr) * n_pos)
    
    # Find threshold with minimum cost
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        'optimal_threshold': optimal_threshold,
        'optimal_fpr': fpr[optimal_idx],
        'optimal_tpr': tpr[optimal_idx],
        'cost': costs[optimal_idx]
    }

# Example usage
cost_ratios = [0.5, 1.0, 2.0]
for ratio in cost_ratios:
    results = find_optimal_threshold(y_test, y_pred_proba, cost_ratio=ratio)
    print(f"\nCost Ratio {ratio}:")
    print(f"Optimal Threshold: {results['optimal_threshold']:.3f}")
    print(f"TPR: {results['optimal_tpr']:.3f}")
    print(f"FPR: {results['optimal_fpr']:.3f}")
```

Slide 12: Handling Class Imbalance with AUC-ROC

AUC-ROC's effectiveness in imbalanced scenarios can be further improved using sampling techniques. This implementation shows how to combine sampling strategies with AUC-ROC evaluation.

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def evaluate_sampling_strategies(X, y, base_classifier):
    # Create sampling strategies
    strategies = {
        'None': None,
        'SMOTE': SMOTE(random_state=42),
        'Undersampling': RandomUnderSampler(random_state=42),
        'Combined': Pipeline([
            ('under', RandomUnderSampler(random_state=42)),
            ('over', SMOTE(random_state=42))
        ])
    }
    
    results = {}
    for name, sampler in strategies.items():
        # Create pipeline
        steps = []
        if sampler:
            steps.append(('sampler', sampler))
        steps.append(('classifier', base_classifier))
        pipeline = Pipeline(steps)
        
        # Cross-validate
        scores = cross_val_score(pipeline, X, y, 
                               scoring='roc_auc', 
                               cv=5)
        
        results[name] = {
            'mean_auc': scores.mean(),
            'std_auc': scores.std()
        }
    
    return results

# Example usage
from sklearn.ensemble import GradientBoostingClassifier
base_clf = GradientBoostingClassifier()
results = evaluate_sampling_strategies(X, y, base_clf)

for strategy, metrics in results.items():
    print(f"\n{strategy}:")
    print(f"AUC-ROC: {metrics['mean_auc']:.3f} ± {metrics['std_auc']:.3f}")
```

Slide 13: Custom Loss Function Using AUC-ROC

Implementing AUC-ROC as a custom loss function for gradient boosting machines can directly optimize for this metric during training.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class AUCOptimizedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.1, n_estimators=100):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.estimators = []
    
    def _auc_loss_gradient(self, y_true, y_pred):
        """
        Compute gradient of AUC loss with respect to predictions
        """
        pos_idx = y_true == 1
        neg_idx = y_true == 0
        
        pos_pred = y_pred[pos_idx]
        neg_pred = y_pred[neg_idx]
        
        gradient = np.zeros_like(y_pred)
        n_pos = sum(pos_idx)
        n_neg = sum(neg_idx)
        
        for i, pred in enumerate(y_pred):
            if y_true[i] == 1:
                gradient[i] = -np.sum(pred < neg_pred) / (n_pos * n_neg)
            else:
                gradient[i] = np.sum(pred > pos_pred) / (n_pos * n_neg)
                
        return gradient
    
    def fit(self, X, y):
        self.estimators = []
        y_pred = np.zeros(len(y))
        
        for _ in range(self.n_estimators):
            gradient = self._auc_loss_gradient(y, y_pred)
            estimator = DecisionTreeRegressor(max_depth=3)
            estimator.fit(X, -gradient)
            
            update = self.learning_rate * estimator.predict(X)
            y_pred += update
            self.estimators.append(estimator)
        
        return self

# Example usage
model = AUCOptimizedModel()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
print(f"AUC-optimized model score: {roc_auc_score(y_test, y_pred):.3f}")
```

Slide 14: Additional Resources

*   Robust AUC-ROC optimization: [https://arxiv.org/abs/2012.03173](https://arxiv.org/abs/2012.03173)
*   Deep learning with AUC-ROC: [https://arxiv.org/abs/1811.11247](https://arxiv.org/abs/1811.11247)
*   Analysis of AUC-ROC properties: [https://arxiv.org/abs/1708.05122](https://arxiv.org/abs/1708.05122)
*   Time-series specific AUC-ROC: [https://scholar.google.com/search?q=time+series+auc+roc](https://scholar.google.com/search?q=time+series+auc+roc)
*   Comparison of evaluation metrics: [https://scholar.google.com/search?q=comparison+classification+metrics+machine+learning](https://scholar.google.com/search?q=comparison+classification+metrics+machine+learning)

