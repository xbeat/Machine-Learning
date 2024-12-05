## Evaluating Classifier Performance with ROC Curves
Slide 1: Understanding ROC Fundamentals

In binary classification, the Receiver Operating Characteristic (ROC) curve visualizes the trade-off between sensitivity (True Positive Rate) and 1-specificity (False Positive Rate) across various classification thresholds. This fundamental tool helps assess classifier performance independent of class distribution.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Generate sample prediction scores and true labels
np.random.seed(42)
y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.6, 0.3, 0.7, 0.2, 0.9, 0.5])

# Calculate ROC curve points
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Basic ROC Curve Example')
plt.legend(loc="lower right")
plt.show()
```

Slide 2: Mathematical Foundation of ROC Metrics

The core metrics that form the ROC curve are derived from the confusion matrix elements. These fundamental calculations establish the relationship between True Positive Rate (Sensitivity) and False Positive Rate, forming the basis for ROC analysis.

```python
# Mathematical formulas for ROC metrics
"""
$$TPR = \frac{TP}{TP + FN}$$

$$FPR = \frac{FP}{FP + TN}$$

$$Specificity = \frac{TN}{TN + FP}$$

$$Sensitivity = TPR = \frac{TP}{TP + FN}$$

$$Precision = \frac{TP}{TP + FP}$$
"""

def calculate_roc_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    
    return TPR, FPR, specificity, precision
```

Slide 3: ROC Curve Implementation

The implementation of an ROC curve requires calculating performance metrics across multiple classification thresholds. This code demonstrates how to create a complete ROC curve analysis system from scratch without relying on sklearn's built-in functions.

```python
def compute_roc_curve(y_true, y_scores):
    # Sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    
    return fpr, tpr, y_scores[threshold_idxs]
```

Slide 4: Advanced ROC Analysis with Cross-Validation

Cross-validation in ROC analysis provides more robust performance estimates by evaluating the classifier across multiple data splits. This implementation demonstrates how to perform ROC analysis with k-fold cross-validation.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import numpy as np

def cv_roc_analysis(classifier, X, y, n_folds=5):
    cv = StratifiedKFold(n_splits=n_folds)
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    
    for fold, (train, test) in enumerate(cv.split(X, y)):
        model = clone(classifier).fit(X[train], y[train])
        y_scores = model.predict_proba(X[test])[:, 1]
        
        fpr, tpr, _ = roc_curve(y[test], y_scores)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))
    
    return np.mean(tprs, axis=0), np.mean(aucs)
```

Slide 5: Handling Imbalanced Datasets in ROC Analysis

When working with imbalanced datasets, traditional ROC curves might not provide complete insights. This implementation shows how to adapt ROC analysis for imbalanced data using sampling techniques and weighted metrics.

```python
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

def balanced_roc_analysis(X, y, classifier):
    # Separate majority and minority classes
    X_maj = X[y == 0]
    X_min = X[y == 1]
    y_maj = y[y == 0]
    y_min = y[y == 1]
    
    # Upsample minority class
    X_min_upsampled, y_min_upsampled = resample(
        X_min, y_min,
        replace=True,
        n_samples=len(X_maj)
    )
    
    # Combine balanced dataset
    X_balanced = np.vstack([X_maj, X_min_upsampled])
    y_balanced = np.hstack([y_maj, y_min_upsampled])
    
    # Scale features
    scaler = StandardScaler()
    X_balanced_scaled = scaler.fit_transform(X_balanced)
    
    # Calculate ROC metrics
    fpr, tpr, thresholds = roc_curve(y_balanced, 
                                    classifier.fit(X_balanced_scaled, y_balanced)
                                    .predict_proba(X_balanced_scaled)[:, 1])
    
    return fpr, tpr, thresholds
```

Slide 6: Real-world Example - Credit Card Fraud Detection

Credit card fraud detection represents a classic use case for ROC analysis, where balancing between false positives and true positives is crucial for business decisions. This implementation demonstrates a complete fraud detection system.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Simulating credit card transaction data
np.random.seed(42)
n_samples = 1000
n_features = 5

def generate_fraud_data():
    # Generate normal transactions
    X_normal = np.random.normal(0, 1, (int(n_samples * 0.99), n_features))
    y_normal = np.zeros(int(n_samples * 0.99))
    
    # Generate fraudulent transactions
    X_fraud = np.random.normal(2, 1, (int(n_samples * 0.01), n_features))
    y_fraud = np.ones(int(n_samples * 0.01))
    
    X = np.vstack([X_normal, X_fraud])
    y = np.hstack([y_normal, y_fraud])
    
    return X, y

# Generate and split data
X, y = generate_fraud_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Get predictions and ROC metrics
y_pred_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
```

Slide 7: ROC Curve Comparison

When comparing multiple classifiers, it's essential to visualize their ROC curves together. This implementation provides a framework for comparing different classification algorithms on the same dataset.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def compare_classifiers(X, y, classifiers_dict):
    plt.figure(figsize=(10, 8))
    
    for name, clf in classifiers_dict.items():
        # Train and predict
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        # Calculate ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.show()

# Example usage
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True),
    'Naive Bayes': GaussianNB()
}

compare_classifiers(X, y, classifiers)
```

Slide 8: Optimal Threshold Selection

The optimal threshold selection in ROC analysis involves finding the best operating point that balances sensitivity and specificity. This implementation demonstrates various methods for selecting the optimal classification threshold.

```python
def find_optimal_thresholds(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Youden's J statistic (maximizing sensitivity + specificity - 1)
    J = tpr - fpr
    optimal_idx_youden = np.argmax(J)
    optimal_threshold_youden = thresholds[optimal_idx_youden]
    
    # Distance to perfect classifier
    distances = np.sqrt(fpr**2 + (1-tpr)**2)
    optimal_idx_distance = np.argmin(distances)
    optimal_threshold_distance = thresholds[optimal_idx_distance]
    
    # F1 score optimization
    precision = tpr / (tpr + fpr)
    precision[np.isnan(precision)] = 0
    f1_scores = 2 * (precision * tpr) / (precision + tpr)
    optimal_idx_f1 = np.argmax(f1_scores)
    optimal_threshold_f1 = thresholds[optimal_idx_f1]
    
    return {
        'youden': optimal_threshold_youden,
        'distance': optimal_threshold_distance,
        'f1': optimal_threshold_f1
    }
```

Slide 9: Confidence Intervals for ROC Curves

Understanding the uncertainty in ROC curves is crucial for reliable model evaluation. This implementation shows how to calculate and visualize confidence intervals using bootstrap resampling.

```python
def roc_confidence_intervals(X, y, classifier, n_bootstraps=1000, confidence=0.95):
    n_samples = X.shape[0]
    rng = np.random.RandomState(42)
    
    # Storage for bootstrap curves
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for i in range(n_bootstraps):
        # Bootstrap sample indices
        indices = rng.randint(0, n_samples, n_samples)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Train classifier and get predictions
        classifier.fit(X_boot, y_boot)
        y_pred = classifier.predict_proba(X)[:, 1]
        
        # Calculate ROC and interpolate
        fpr, tpr, _ = roc_curve(y, y_pred)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs.append(auc(fpr, tpr))
    
    # Calculate confidence intervals
    tprs = np.array(tprs)
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    
    tprs_upper = np.minimum(mean_tpr + std_tpr * 1.96, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr * 1.96, 0)
    
    return mean_fpr, mean_tpr, tprs_lower, tprs_upper, np.mean(aucs), np.std(aucs)
```

Slide 10: Real-world Example - Medical Diagnosis

Medical diagnosis systems frequently use ROC analysis to evaluate diagnostic test performance. This implementation demonstrates a complete medical diagnosis classifier with ROC analysis.

```python
def medical_diagnosis_roc():
    # Simulate medical test data
    np.random.seed(42)
    
    # Generate patient data (e.g., blood test results)
    n_healthy = 800
    n_sick = 200
    
    # Healthy patients (multiple markers)
    healthy_markers = np.random.normal(loc=1.0, scale=0.5, size=(n_healthy, 4))
    healthy_labels = np.zeros(n_healthy)
    
    # Sick patients (multiple markers)
    sick_markers = np.random.normal(loc=2.0, scale=0.8, size=(n_sick, 4))
    sick_labels = np.ones(n_sick)
    
    # Combine data
    X = np.vstack([healthy_markers, sick_markers])
    y = np.hstack([healthy_labels, sick_labels])
    
    # Split data and train classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Train multiple classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'Logistic Regression': LogisticRegression()
    }
    
    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        results[name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc(fpr, tpr)
        }
    
    return results
```

Slide 11: ROC Analysis with Time-Series Cross-Validation

Time-series data requires special consideration in ROC analysis to maintain temporal order and avoid look-ahead bias. This implementation demonstrates how to properly evaluate classifiers on sequential data.

```python
from sklearn.model_selection import TimeSeriesSplit

def temporal_roc_analysis(X, y, classifier, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {
        'fpr': [],
        'tpr': [],
        'auc': []
    }
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model and get predictions
        classifier.fit(X_train, y_train)
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        
        # Calculate ROC metrics
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        results['fpr'].append(fpr)
        results['tpr'].append(tpr)
        results['auc'].append(roc_auc)
    
    return results

# Example usage with synthetic time series data
def generate_time_series_data(n_samples=1000):
    time = np.arange(n_samples)
    signal = np.sin(2 * np.pi * 0.02 * time) + np.random.normal(0, 0.1, n_samples)
    
    # Create features using time lags
    X = np.column_stack([signal[:-1], signal[1:]])
    y = (signal[2:] > signal[1:-1]).astype(int)
    
    return X[:-1], y
```

Slide 12: Multi-class ROC Analysis

Extending ROC analysis to multi-class problems requires special consideration. This implementation shows how to handle multi-class classification using one-vs-rest and one-vs-one approaches.

```python
from itertools import combinations
from sklearn.preprocessing import label_binarize

def multiclass_roc_analysis(X, y, classifier, n_classes):
    # One-vs-Rest ROC curves
    y_bin = label_binarize(y, classes=range(n_classes))
    
    # Train classifier
    classifier.fit(X, y)
    y_score = classifier.predict_proba(X)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # One-vs-Rest
    for i in range(n_classes):
        fpr[f'class_{i}'], tpr[f'class_{i}'], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[f'class_{i}'] = auc(fpr[f'class_{i}'], tpr[f'class_{i}'])
    
    # One-vs-One
    for i, j in combinations(range(n_classes), 2):
        mask = np.where((y == i) | (y == j))[0]
        y_binary = (y[mask] == i).astype(int)
        scores_binary = y_score[mask][:, i] / (y_score[mask][:, i] + y_score[mask][:, j])
        
        fpr[f'class_{i}_vs_{j}'], tpr[f'class_{i}_vs_{j}'], _ = roc_curve(y_binary, scores_binary)
        roc_auc[f'class_{i}_vs_{j}'] = auc(fpr[f'class_{i}_vs_{j}'], tpr[f'class_{i}_vs_{j}'])
    
    return fpr, tpr, roc_auc
```

Slide 13: Additional Resources

*   ArXiv paper: "Deep Learning for ROC-Based Medical Diagnosis" [https://arxiv.org/abs/2103.00208](https://arxiv.org/abs/2103.00208)
*   ArXiv paper: "Advances in ROC Analysis: Recent Developments and Applications" [https://arxiv.org/abs/2008.13749](https://arxiv.org/abs/2008.13749)
*   ArXiv paper: "Novel Approaches to ROC Optimization in Machine Learning" [https://arxiv.org/abs/1912.05979](https://arxiv.org/abs/1912.05979)
*   Recommended search terms for further reading:
    *   "ROC Analysis in Clinical Decision Making"
    *   "Machine Learning ROC Optimization Techniques"
    *   "Multi-class ROC Analysis Methods"

Note: These URLs are examples. For the most current research, please search on Google Scholar or ArXiv directly.

