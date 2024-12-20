## Calculating Confusion Matrix Metrics
Slide 1: Understanding Confusion Matrix

In binary classification, a confusion matrix is a 2x2 table that visualizes the performance of a model by comparing predicted values against actual values. It forms the foundation for calculating key metrics like accuracy, precision, recall, and F1-score.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_confusion_matrix(y_true, y_pred):
    # Initialize 2x2 matrix
    cm = np.zeros((2,2))
    
    # Calculate metrics
    cm[0,0] = sum((y_true == 0) & (y_pred == 0))  # TN
    cm[0,1] = sum((y_true == 0) & (y_pred == 1))  # FP
    cm[1,0] = sum((y_true == 1) & (y_pred == 0))  # FN
    cm[1,1] = sum((y_true == 1) & (y_pred == 1))  # TP
    
    return cm

# Example usage
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 1])

cm = create_confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
```

Slide 2: Computing Basic Metrics

Understanding how to extract True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN) from the confusion matrix enables calculation of fundamental classification metrics.

```python
def compute_basic_metrics(confusion_matrix):
    # Extract values
    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]
    
    metrics = {
        'True Positives (TP)': tp,
        'False Positives (FP)': fp,
        'True Negatives (TN)': tn,
        'False Negatives (FN)': fn
    }
    
    return metrics

# Using previous confusion matrix
metrics = compute_basic_metrics(cm)
for metric, value in metrics.items():
    print(f"{metric}: {value}")
```

Slide 3: Calculating Performance Metrics

From the fundamental confusion matrix components, we can derive essential performance metrics. These include accuracy, precision, recall, and F1-score, which provide different perspectives on model performance.

```python
def calculate_performance_metrics(cm):
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    # Calculate metrics using formulas
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    return metrics

# Calculate and display metrics
performance = calculate_performance_metrics(cm)
for metric, value in performance.items():
    print(f"{metric}: {value:.3f}")
```

Slide 4: Mathematical Foundations

The confusion matrix serves as the basis for deriving classification metrics. Understanding these mathematical relationships is crucial for interpreting model performance and making informed decisions.

```python
# Mathematical formulas for key metrics
formulas = """
Accuracy Formula:
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

Precision Formula:
$$Precision = \frac{TP}{TP + FP}$$

Recall Formula:
$$Recall = \frac{TP}{TP + FN}$$

F1-Score Formula:
$$F1 = 2 * \frac{Precision * Recall}{Precision + Recall}$$
"""
print(formulas)
```

Slide 5: Visualization of Confusion Matrix

Effective visualization of confusion matrix results helps in quickly identifying patterns and potential issues in model performance. This implementation uses seaborn to create an informative heatmap.

```python
def visualize_confusion_matrix(cm, labels=['Negative', 'Positive']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix Heatmap')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

# Visualize the confusion matrix
plt = visualize_confusion_matrix(cm)
plt.show()
```

Slide 6: Real-world Example - Credit Card Fraud Detection

In this practical example, we'll implement confusion matrix analysis for a credit card fraud detection system, demonstrating how to handle imbalanced datasets and interpret results in a real-world context.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Simulate credit card transaction data
np.random.seed(42)
n_samples = 10000
n_features = 10

# Generate synthetic data
X = np.random.randn(n_samples, n_features)
# Create imbalanced dataset (99% normal, 1% fraudulent)
y = np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Slide 7: Source Code for Credit Card Fraud Detection

```python
# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate confusion matrix
fraud_cm = create_confusion_matrix(y_test, y_pred)

# Calculate and display metrics
fraud_metrics = calculate_performance_metrics(fraud_cm)
print("\nFraud Detection Results:")
for metric, value in fraud_metrics.items():
    print(f"{metric}: {value:.3f}")

# Visualize results
plt = visualize_confusion_matrix(fraud_cm, ['Normal', 'Fraud'])
plt.title('Credit Card Fraud Detection Results')
plt.show()
```

Slide 8: Handling Class Imbalance

Class imbalance significantly impacts confusion matrix interpretation. We'll explore techniques to address this common challenge in real-world classification problems through weighted metrics and sampling strategies.

```python
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Apply SMOTE for balanced training
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train model on balanced data
rf_balanced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_balanced.fit(X_train_balanced, y_train_balanced)

# Predict and evaluate
y_pred_balanced = rf_balanced.predict(X_test)
balanced_cm = create_confusion_matrix(y_test, y_pred_balanced)

# Calculate metrics for balanced model
balanced_metrics = calculate_performance_metrics(balanced_cm)
```

Slide 9: Real-world Example - Medical Diagnosis

Implementing confusion matrix analysis for a medical diagnosis system, where false negatives can have serious consequences. This example demonstrates how to weight different types of errors.

```python
# Simulate medical diagnosis data
n_patients = 5000
n_medical_features = 8

# Generate synthetic patient data
X_medical = np.random.randn(n_patients, n_medical_features)
# Create realistic disease prevalence (5% positive cases)
y_medical = np.random.choice([0, 1], size=n_patients, p=[0.95, 0.05])

# Split medical data
X_med_train, X_med_test, y_med_train, y_med_test = train_test_split(
    X_medical, y_medical, test_size=0.2, random_state=42)
```

Slide 10: Source Code for Medical Diagnosis Implementation

This implementation focuses on a weighted approach to handle the critical nature of false negatives in medical diagnosis, incorporating custom scoring metrics to prioritize minimal false negatives.

```python
# Train model with class weights to penalize false negatives more heavily
class_weights = {0: 1, 1: 5}  # Higher weight for positive class
rf_medical = RandomForestClassifier(
    n_estimators=100,
    class_weight=class_weights,
    random_state=42
)
rf_medical.fit(X_med_train, y_med_train)

# Generate predictions
y_med_pred = rf_medical.predict(X_med_test)

# Calculate confusion matrix for medical diagnosis
med_cm = create_confusion_matrix(y_med_test, y_med_pred)

# Custom metric focusing on minimizing false negatives
def calculate_medical_metrics(cm):
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    # Standard metrics
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0  # Negative Predictive Value
    
    return {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'NPV': npv
    }

medical_metrics = calculate_medical_metrics(med_cm)
print("\nMedical Diagnosis Metrics:")
for metric, value in medical_metrics.items():
    print(f"{metric}: {value:.3f}")
```

Slide 11: Cross-Validation with Confusion Matrix

Implementing cross-validation to obtain more robust confusion matrix metrics, essential for reliable model evaluation in production environments.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import numpy as np

def cross_validated_confusion_matrix(X, y, model, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cms = []
    
    for train_idx, val_idx in skf.split(X, y):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        # Calculate confusion matrix
        cms.append(create_confusion_matrix(y_val, y_pred))
    
    # Average confusion matrices
    mean_cm = np.mean(cms, axis=0)
    std_cm = np.std(cms, axis=0)
    
    return mean_cm, std_cm

# Example usage with medical diagnosis data
mean_cm, std_cm = cross_validated_confusion_matrix(
    X_medical, y_medical, 
    RandomForestClassifier(random_state=42)
)
```

Slide 12: Advanced Metrics Derivation

Understanding the relationship between confusion matrix components and advanced performance metrics provides deeper insights into model behavior and aids in model selection.

```python
def calculate_advanced_metrics(cm):
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    # Matthews Correlation Coefficient
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_den if mcc_den != 0 else 0
    
    # Balanced Accuracy
    balanced_acc = ((tp / (tp + fn) if (tp + fn) != 0 else 0) + 
                   (tn / (tn + fp) if (tn + fp) != 0 else 0)) / 2
    
    # Cohen's Kappa
    po = (tp + tn) / (tp + tn + fp + fn)
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / ((tp + tn + fp + fn) ** 2)
    kappa = (po - pe) / (1 - pe) if pe != 1 else 0
    
    return {
        'MCC': mcc,
        'Balanced_Accuracy': balanced_acc,
        'Cohens_Kappa': kappa
    }

advanced_metrics = calculate_advanced_metrics(mean_cm)
print("\nAdvanced Metrics:")
for metric, value in advanced_metrics.items():
    print(f"{metric}: {value:.3f}")
```

Slide 13: Confidence Intervals for Metrics

Calculating confidence intervals for confusion matrix metrics provides statistical rigor and helps in understanding the reliability of model performance measurements.

```python
from scipy import stats

def calculate_metric_confidence_intervals(metrics_list, confidence=0.95):
    ci_metrics = {}
    for metric in metrics_list[0].keys():
        values = [m[metric] for m in metrics_list]
        mean = np.mean(values)
        ci = stats.t.interval(confidence, len(values)-1,
                            loc=mean,
                            scale=stats.sem(values))
        ci_metrics[metric] = {
            'mean': mean,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }
    return ci_metrics

# Generate bootstrap samples and calculate metrics
n_bootstrap = 1000
bootstrap_metrics = []
for _ in range(n_bootstrap):
    indices = np.random.choice(len(y_test), len(y_test), replace=True)
    cm = create_confusion_matrix(y_test[indices], y_pred[indices])
    metrics = calculate_performance_metrics(cm)
    bootstrap_metrics.append(metrics)

# Calculate confidence intervals
ci_results = calculate_metric_confidence_intervals(bootstrap_metrics)
for metric, values in ci_results.items():
    print(f"\n{metric}:")
    print(f"Mean: {values['mean']:.3f}")
    print(f"95% CI: [{values['ci_lower']:.3f}, {values['ci_upper']:.3f}]")
```

Slide 14: Additional Resources

*   "A systematic study of the class imbalance problem in convolutional neural networks"
    *   Search on ArXiv: [https://arxiv.org/abs/1710.05381](https://arxiv.org/abs/1710.05381)
*   "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets"
    *   Search on ArXiv: [https://arxiv.org/abs/1705.05391](https://arxiv.org/abs/1705.05391)
*   "Beyond Accuracy, F-score and ROC: A Family of Discriminant Measures for Performance Evaluation"
    *   Search on Google Scholar for the latest version
*   "A Survey of Cross-Validation Procedures for Model Selection"
    *   Search on ArXiv: [https://arxiv.org/abs/1907.04909](https://arxiv.org/abs/1907.04909)
*   "On the Class Imbalance Problem in Medical Diagnosis Data"
    *   Visit IEEE Xplore Digital Library for the complete paper

