## Machine Learning Classification Metrics with Python
Slide 1: Understanding Classification Metrics Fundamentals

Classification metrics form the foundation for evaluating machine learning model performance. These metrics help quantify how well a model can distinguish between different classes, measuring various aspects of predictive accuracy through statistical calculations derived from the confusion matrix.

```python
import numpy as np
from sklearn.metrics import confusion_matrix

def create_confusion_matrix(y_true, y_pred):
    """
    Creates and returns a confusion matrix with basic metrics
    
    Parameters:
    y_true: array-like of shape (n_samples,) Ground truth labels
    y_pred: array-like of shape (n_samples,) Predicted labels
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Basic metrics formula in comments
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    
    print(f"Confusion Matrix:\n{cm}")
    return cm
    
# Example usage
y_true = [0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 0, 1, 1]
result = create_confusion_matrix(y_true, y_pred)
```

Slide 2: Accuracy and Precision Metrics

These fundamental metrics provide different perspectives on model performance. Accuracy measures overall correctness, while precision focuses on the reliability of positive predictions, making them essential for different use cases and business requirements.

```python
def calculate_basic_metrics(y_true, y_pred):
    """
    Calculate accuracy and precision metrics
    
    Mathematical formulas:
    $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$
    $$Precision = \frac{TP}{TP + FP}$$
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    
    return {
        'accuracy': accuracy,
        'precision': precision
    }

# Example usage
y_true = [0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 0, 1, 1]
metrics = calculate_basic_metrics(y_true, y_pred)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
```

Slide 3: Recall and F1-Score Implementation

Recall measures the model's ability to find all relevant cases, while F1-score provides a balanced measure between precision and recall. These metrics are crucial when dealing with imbalanced datasets where accuracy alone might be misleading.

```python
def calculate_advanced_metrics(y_true, y_pred):
    """
    Calculate recall and F1-score
    
    Mathematical formulas:
    $$Recall = \frac{TP}{TP + FN}$$
    $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        'recall': recall,
        'f1_score': f1
    }

# Example usage
metrics = calculate_advanced_metrics(y_true, y_pred)
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1-Score: {metrics['f1_score']:.3f}")
```

Slide 4: ROC Curve Implementation

The Receiver Operating Characteristic curve visualizes the trade-off between true positive rate and false positive rate across various classification thresholds. This metric is essential for understanding model performance across different decision boundaries.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_prob):
    """
    Plot ROC curve from probability predictions
    
    Mathematical formula:
    $$TPR = \frac{TP}{TP + FN}$$
    $$FPR = \frac{FP}{FP + TN}$$
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
# Example usage
y_true = [0, 1, 0, 1, 1, 0, 1, 0]
y_prob = [0.1, 0.9, 0.2, 0.7, 0.8, 0.1, 0.9, 0.3]
plot_roc_curve(y_true, y_prob)
```

Slide 5: Precision-Recall Curve

The Precision-Recall curve is particularly useful for imbalanced datasets where ROC curves might present an overly optimistic view of model performance. It shows the trade-off between precision and recall at various threshold settings.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall_curve(y_true, y_prob):
    """
    Plot Precision-Recall curve
    
    Mathematical formula:
    $$AP = \sum_n (R_n - R_{n-1}) P_n$$
    Where AP is Average Precision, R is Recall, P is Precision
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

# Example usage
y_true = [0, 1, 0, 1, 1, 0, 1, 0]
y_prob = [0.1, 0.9, 0.2, 0.7, 0.8, 0.1, 0.9, 0.3]
plot_precision_recall_curve(y_true, y_prob)
```

Slide 6: Multi-Class Classification Metrics

Multi-class classification requires specialized metrics that can handle multiple categories simultaneously. These implementations focus on macro and micro averaging techniques to provide comprehensive performance evaluation across all classes.

```python
def calculate_multiclass_metrics(y_true, y_pred, num_classes):
    """
    Calculate metrics for multi-class classification
    
    Mathematical formulas:
    $$Macro-Precision = \frac{1}{n}\sum_{i=1}^{n} Precision_i$$
    $$Micro-Precision = \frac{TP_{total}}{TP_{total} + FP_{total}}$$
    """
    # Initialize arrays for per-class metrics
    precisions = np.zeros(num_classes)
    recalls = np.zeros(num_classes)
    
    # Calculate per-class metrics
    for class_idx in range(num_classes):
        true_class = (y_true == class_idx)
        pred_class = (y_pred == class_idx)
        
        tp = np.sum(true_class & pred_class)
        fp = np.sum(~true_class & pred_class)
        fn = np.sum(true_class & ~pred_class)
        
        precisions[class_idx] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recalls[class_idx] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    
    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'per_class_precision': precisions,
        'per_class_recall': recalls
    }

# Example usage
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred = [0, 1, 1, 0, 1, 2, 2, 1, 2]
metrics = calculate_multiclass_metrics(y_true, y_pred, num_classes=3)
print(f"Macro Precision: {metrics['macro_precision']:.3f}")
```

Slide 7: Cohen's Kappa Score Implementation

Cohen's Kappa Score measures inter-rater agreement for categorical items, accounting for agreement occurring by chance. This metric is particularly useful when evaluating model performance on imbalanced datasets.

```python
def cohen_kappa_score(y_true, y_pred):
    """
    Calculate Cohen's Kappa Score
    
    Mathematical formula:
    $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
    Where p_o is observed agreement and p_e is expected agreement
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    sum_0 = cm.sum(axis=0)
    sum_1 = cm.sum(axis=1)
    expected = np.outer(sum_0, sum_1) / np.sum(sum_0)
    
    w_mat = np.ones([n_classes, n_classes], dtype=np.int)
    w_mat.flat[::n_classes + 1] = 0
    
    k = np.sum(cm * w_mat)
    e = np.sum(expected * w_mat)
    
    kappa = 1 - k / e if e != 0 else 1
    
    return kappa

# Example usage
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred = [0, 1, 1, 0, 1, 2, 2, 1, 2]
kappa = cohen_kappa_score(y_true, y_pred)
print(f"Cohen's Kappa Score: {kappa:.3f}")
```

Slide 8: Balanced Accuracy and Matthews Correlation Coefficient

These metrics provide robust evaluation measures for imbalanced datasets. Balanced accuracy normalizes true positive and true negative rates, while Matthews Correlation Coefficient considers all confusion matrix elements in a balanced way.

```python
def advanced_imbalanced_metrics(y_true, y_pred):
    """
    Calculate balanced accuracy and Matthews Correlation Coefficient
    
    Mathematical formulas:
    $$Balanced Accuracy = \frac{1}{2}(\frac{TP}{TP + FN} + \frac{TN}{TN + FP})$$
    $$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Balanced Accuracy
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    
    # Matthews Correlation Coefficient
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator if denominator != 0 else 0
    
    return {
        'balanced_accuracy': balanced_acc,
        'mcc': mcc
    }

# Example usage
y_true = [0, 0, 0, 0, 1, 1]
y_pred = [0, 0, 0, 1, 1, 0]
metrics = advanced_imbalanced_metrics(y_true, y_pred)
print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
print(f"Matthews Correlation Coefficient: {metrics['mcc']:.3f}")
```

Slide 9: Cross-Validation Implementation for Classification Metrics

Cross-validation provides a robust method for assessing model performance by evaluating metrics across different data splits. This implementation focuses on stratified k-fold cross-validation to maintain class distribution across folds.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import numpy as np

def cross_validate_classifier(model, X, y, n_splits=5):
    """
    Perform stratified k-fold cross-validation with multiple metrics
    
    Mathematical formula for std error:
    $$SE = \sqrt{\frac{\sum(x - \bar{x})^2}{n-1}}$$
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Clone model for fresh instance
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_val)
        
        # Calculate metrics for this fold
        fold_metrics = calculate_basic_metrics(y_val, y_pred)
        for metric in metrics:
            metrics[metric].append(fold_metrics[metric])
    
    # Calculate mean and std for each metric
    results = {}
    for metric in metrics:
        results[f'{metric}_mean'] = np.mean(metrics[metric])
        results[f'{metric}_std'] = np.std(metrics[metric])
    
    return results

# Example usage
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

# Generate sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
model = DecisionTreeClassifier(random_state=42)

results = cross_validate_classifier(model, X, y)
for metric, value in results.items():
    print(f"{metric}: {value:.3f}")
```

Slide 10: Calibration Metrics and Reliability Diagram

Model calibration assesses how well the predicted probabilities of a classifier reflect the actual probabilities of the outcomes. This implementation includes both calibration curve plotting and Brier score calculation.

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def analyze_calibration(y_true, y_prob, n_bins=10):
    """
    Analyze classifier calibration and plot reliability diagram
    
    Mathematical formula for Brier Score:
    $$BS = \frac{1}{N}\sum_{i=1}^{N}(f_i - o_i)^2$$
    Where f_i are forecasted probabilities and o_i are actual outcomes
    """
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Calculate Brier score
    brier_score = np.mean((y_prob - y_true) ** 2)
    
    # Plot reliability diagram
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.plot(prob_pred, prob_true, 's-', label='Model')
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability')
    plt.title(f'Reliability Diagram (Brier Score: {brier_score:.3f})')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return {
        'brier_score': brier_score,
        'calibration_curve': {
            'prob_true': prob_true,
            'prob_pred': prob_pred
        }
    }

# Example usage
np.random.seed(42)
# Generate sample predictions
y_true = np.random.binomial(1, 0.3, 1000)
y_prob = np.clip(np.random.normal(y_true, 0.2), 0, 1)

results = analyze_calibration(y_true, y_prob)
print(f"Brier Score: {results['brier_score']:.3f}")
```

Slide 11: Custom Scoring Function Implementation

Developing custom scoring metrics allows for domain-specific evaluation criteria. This implementation demonstrates how to create and validate custom scoring functions that can be used with scikit-learn's cross-validation framework.

```python
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

def custom_metric(y_true, y_pred, weight_fp=2.0, weight_fn=1.0):
    """
    Create custom weighted metric for domain-specific needs
    
    Mathematical formula:
    $$Score = \frac{TP}{TP + weight_{fp}FP + weight_{fn}FN}$$
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    denominator = tp + (weight_fp * fp) + (weight_fn * fn)
    score = tp / denominator if denominator > 0 else 0
    return score

# Create scorer object
custom_scorer = make_scorer(custom_metric, 
                          weight_fp=2.0, 
                          weight_fn=1.0,
                          greater_is_better=True)

# Example usage with cross-validation
def evaluate_with_custom_metric(X, y, model, cv=5):
    scores = cross_val_score(model, 
                           X, 
                           y, 
                           cv=cv,
                           scoring=custom_scorer)
    
    return {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'all_scores': scores
    }

# Example usage
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
model = DecisionTreeClassifier(random_state=42)
results = evaluate_with_custom_metric(X, y, model)
print(f"Custom Metric - Mean: {results['mean_score']:.3f} ± {results['std_score']:.3f}")
```

Slide 12: Real-World Example - Credit Card Fraud Detection

This implementation demonstrates a complete workflow for evaluating a fraud detection model, where class imbalance and cost-sensitive errors require careful metric selection and interpretation.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def evaluate_fraud_detection(X, y):
    """
    Comprehensive evaluation for fraud detection
    
    Cost matrix formula:
    $$Cost = FN \times cost_{fn} + FP \times cost_{fp}$$
    where cost_fn = 100 (missed fraud)
    and cost_fp = 10 (false alarm)
    """
    # Prepare data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split with stratification due to imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate comprehensive metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Cost-sensitive evaluation
    cost_fn = 100  # Cost of missing fraud
    cost_fp = 10   # Cost of false alarm
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    
    metrics = {
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'cost_savings': 1 - (total_cost / (len(y_test) * cost_fn)),
        'confusion_matrix': cm,
        'total_cost': total_cost
    }
    
    return metrics, y_prob, y_test

# Example usage
# Generate imbalanced dataset
np.random.seed(42)
n_samples = 10000
fraud_ratio = 0.02

X = np.random.randn(n_samples, 10)
y = np.random.choice([0, 1], size=n_samples, p=[1-fraud_ratio, fraud_ratio])

metrics, y_prob, y_test = evaluate_fraud_detection(X, y)
for key, value in metrics.items():
    if isinstance(value, (int, float)):
        print(f"{key}: {value:.3f}")
    elif isinstance(value, np.ndarray):
        print(f"{key}:\n{value}")
```

Slide 13: Real-World Example - Medical Diagnosis Classification

This implementation showcases a medical diagnosis classifier evaluation where false negatives have serious implications and multiple metrics must be considered together.

```python
def evaluate_medical_classifier(X, y, disease_names):
    """
    Comprehensive evaluation for medical diagnosis
    
    Mathematical formulas:
    $$NPV = \frac{TN}{TN + FN}$$
    $$LR+ = \frac{TPR}{FPR}$$
    """
    # Prepare stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics_per_disease = {disease: {
        'sensitivity': [],
        'specificity': [],
        'npv': [],  # Negative Predictive Value
        'ppv': [],  # Positive Predictive Value
        'likelihood_ratio_positive': []
    } for disease in disease_names}
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train multi-label classifier
        model = OneVsRestClassifier(RandomForestClassifier(random_state=42))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics for each disease
        for i, disease in enumerate(disease_names):
            tn, fp, fn, tp = confusion_matrix(y_test[:, i], y_pred[:, i]).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            lr_positive = sensitivity / (1 - specificity) if (1 - specificity) > 0 else float('inf')
            
            metrics_per_disease[disease]['sensitivity'].append(sensitivity)
            metrics_per_disease[disease]['specificity'].append(specificity)
            metrics_per_disease[disease]['npv'].append(npv)
            metrics_per_disease[disease]['ppv'].append(ppv)
            metrics_per_disease[disease]['likelihood_ratio_positive'].append(lr_positive)
    
    # Calculate mean metrics
    final_metrics = {disease: {
        metric: np.mean(values) for metric, values in disease_metrics.items()
    } for disease, disease_metrics in metrics_per_disease.items()}
    
    return final_metrics

# Example usage
n_samples = 1000
n_diseases = 3
disease_names = [f'Disease_{i}' for i in range(n_diseases)]

# Generate multi-label dataset
X = np.random.randn(n_samples, 10)
y = np.random.randint(2, size=(n_samples, n_diseases))

results = evaluate_medical_classifier(X, y, disease_names)
for disease, metrics in results.items():
    print(f"\n{disease}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
```

Slide 14: Additional Resources

*   "A Survey on Deep Learning for Named Entity Recognition" - [https://arxiv.org/abs/1812.09449](https://arxiv.org/abs/1812.09449)
*   "Deep Neural Networks for Learning Graph Representations" - [https://arxiv.org/abs/1704.06483](https://arxiv.org/abs/1704.06483)
*   "Calibration in Modern Neural Networks" - [https://arxiv.org/abs/2106.07998](https://arxiv.org/abs/2106.07998)
*   "Why Should I Trust You?: Explaining the Predictions of Any Classifier" - [https://arxiv.org/abs/1602.04938](https://arxiv.org/abs/1602.04938)
*   "Learning Deep Features for One-Class Classification" - [https://arxiv.org/abs/1801.05365](https://arxiv.org/abs/1801.05365)

