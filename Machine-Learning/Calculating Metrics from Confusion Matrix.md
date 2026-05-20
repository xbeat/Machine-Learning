## Calculating Metrics from Confusion Matrix
Slide 1: Understanding Confusion Matrix Basics

A confusion matrix is a fundamental tool in machine learning that tabulates predicted versus actual values, enabling evaluation of classification model performance through various derived metrics.

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_confusion_matrix(y_true, y_pred):
    # Create basic 2x2 confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

# Example usage
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 1])
cm = create_confusion_matrix(y_true, y_pred)
```

Slide 2: Computing Recall

Recall measures the ability of a classifier to identify all relevant instances, calculated as the ratio of true positives to all actual positives in the dataset.

```python
def calculate_recall(confusion_matrix):
    # Mathematical formula in LaTeX notation:
    # $$Recall = \frac{TP}{TP + FN}$$
    tp = confusion_matrix[1,1]
    fn = confusion_matrix[1,0]
    return tp / (tp + fn) if (tp + fn) > 0 else 0

# Example usage with previous confusion matrix
recall = calculate_recall(cm)
print(f"Recall: {recall:.3f}")
```

Slide 3: Computing Precision

Precision quantifies the accuracy of positive predictions, representing the ratio of correctly identified positives to total predicted positives.

```python
def calculate_precision(confusion_matrix):
    # Mathematical formula in LaTeX notation:
    # $$Precision = \frac{TP}{TP + FP}$$
    tp = confusion_matrix[1,1]
    fp = confusion_matrix[0,1]
    return tp / (tp + fp) if (tp + fp) > 0 else 0

# Example usage with previous confusion matrix
precision = calculate_precision(cm)
print(f"Precision: {recall:.3f}")
```

Slide 4: Computing Accuracy

Accuracy represents the overall correctness of the model by measuring the ratio of correct predictions to total predictions made.

```python
def calculate_accuracy(confusion_matrix):
    # Mathematical formula in LaTeX notation:
    # $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$
    tp = confusion_matrix[1,1]
    tn = confusion_matrix[0,0]
    total = np.sum(confusion_matrix)
    return (tp + tn) / total if total > 0 else 0

# Example usage with previous confusion matrix
accuracy = calculate_accuracy(cm)
print(f"Accuracy: {accuracy:.3f}")
```

Slide 5: Real-world Example - Credit

Card Fraud Detection Implementation of confusion matrix metrics for credit card fraud detection using a synthetic dataset to demonstrate practical application.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Generate synthetic credit card transaction data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                         weights=[0.97, 0.03], random_state=42)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Slide 6: Credit Card Fraud Detection Model Implementation

```python
# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate confusion matrix
fraud_cm = create_confusion_matrix(y_test, y_pred)

# Calculate all metrics
fraud_recall = calculate_recall(fraud_cm)
fraud_precision = calculate_precision(fraud_cm)
fraud_accuracy = calculate_accuracy(fraud_cm)
```

Slide 7: Results for Credit Card Fraud Detection

```python
print("Credit Card Fraud Detection Results:")
print("-" * 40)
print(f"Confusion Matrix:\n{fraud_cm}")
print(f"Recall: {fraud_recall:.3f}")
print(f"Precision: {fraud_precision:.3f}")
print(f"Accuracy: {fraud_accuracy:.3f}")
```

Slide 8: Real-world Example - Medical Diagnosis

Implementation of confusion matrix metrics for a medical diagnosis classification problem using synthetic patient data.

```python
# Generate synthetic medical diagnosis data
X_med, y_med = make_classification(n_samples=800, n_features=15, n_classes=2,
                                 weights=[0.85, 0.15], random_state=42)

# Split dataset
X_train_med, X_test_med, y_train_med, y_test_med = train_test_split(
    X_med, y_med, test_size=0.3, random_state=42)
```

Slide 9: Medical Diagnosis Model Implementation

```python
# Train model
med_model = RandomForestClassifier(n_estimators=100, random_state=42)
med_model.fit(X_train_med, y_train_med)

# Make predictions
y_pred_med = med_model.predict(X_test_med)

# Calculate confusion matrix and metrics
med_cm = create_confusion_matrix(y_test_med, y_pred_med)
med_recall = calculate_recall(med_cm)
med_precision = calculate_precision(med_cm)
med_accuracy = calculate_accuracy(med_cm)
```

Slide 10: Results for Medical Diagnosis

```python
print("Medical Diagnosis Results:")
print("-" * 40)
print(f"Confusion Matrix:\n{med_cm}")
print(f"Recall: {med_recall:.3f}")
print(f"Precision: {med_precision:.3f}")
print(f"Accuracy: {med_accuracy:.3f}")
```

Slide 11: Confusion Matrix Visualization

Creating an informative visualization of confusion matrices using seaborn to enhance interpretation of results.

```python
def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Example usage
plot_confusion_matrix(fraud_cm, 'Fraud Detection Confusion Matrix')
plot_confusion_matrix(med_cm, 'Medical Diagnosis Confusion Matrix')
```

Slide 12: Performance Comparison

Comprehensive comparison of model performance across different metrics for both real-world examples.

```python
def compare_models(metrics_dict):
    for model, metrics in metrics_dict.items():
        print(f"\n{model} Performance:")
        print("-" * 30)
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.3f}")

metrics = {
    "Fraud Detection": {
        "Recall": fraud_recall,
        "Precision": fraud_precision,
        "Accuracy": fraud_accuracy
    },
    "Medical Diagnosis": {
        "Recall": med_recall,
        "Precision": med_precision,
        "Accuracy": med_accuracy
    }
}

compare_models(metrics)
```

Slide 13: Additional Resources 

[https://arxiv.org/abs/1904.04232](https://arxiv.org/abs/1904.04232) - "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets" [https://arxiv.org/abs/2011.09573](https://arxiv.org/abs/2011.09573) - "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList" [https://arxiv.org/abs/1906.02393](https://arxiv.org/abs/1906.02393) - "From Confusion Matrix to Confusion Coefficients: Extending Performance Metrics for Imbalanced Datasets"

