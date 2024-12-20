## Understanding Confusion Matrices in Machine Learning
Slide 1: Introduction to Confusion Matrix

A confusion matrix is a fundamental evaluation metric in machine learning that tabulates the performance of a classification model by comparing predicted labels against actual labels, enabling the calculation of key performance indicators like accuracy, precision, recall, and F1-score.

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Example predictions and actual labels
y_true = [0, 1, 0, 1, 0, 1, 0, 1]
y_pred = [0, 1, 0, 0, 0, 1, 1, 1]

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

Slide 2: Mathematical Foundation of Confusion Matrix

The confusion matrix forms the basis for calculating essential classification metrics through mathematical relationships between True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

```python
# Mathematical formulas for key metrics
"""
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

$$Precision = \frac{TP}{TP + FP}$$

$$Recall = \frac{TP}{TP + FN}$$

$$F1 Score = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$
"""

def calculate_metrics(cm):
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1
```

Slide 3: Implementing a Custom Confusion Matrix

Creating a confusion matrix implementation from scratch helps understand its internal workings and provides flexibility for custom modifications and analysis in binary classification scenarios.

```python
class CustomConfusionMatrix:
    def __init__(self):
        self.matrix = np.zeros((2, 2))
    
    def update(self, y_true, y_pred):
        for t, p in zip(y_true, y_pred):
            self.matrix[t][p] += 1
    
    def get_metrics(self):
        TP = self.matrix[1,1]
        TN = self.matrix[0,0]
        FP = self.matrix[0,1]
        FN = self.matrix[1,0]
        
        metrics = {
            'accuracy': (TP + TN) / self.matrix.sum(),
            'precision': TP / (TP + FP) if (TP + FP) > 0 else 0,
            'recall': TP / (TP + FN) if (TP + FN) > 0 else 0
        }
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
                       (metrics['precision'] + metrics['recall']) \
                       if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        return metrics
```

Slide 4: Multi-class Confusion Matrix

The multi-class confusion matrix extends binary classification concepts to handle multiple categories, requiring more sophisticated interpretation and visualization techniques for comprehensive model evaluation.

```python
from sklearn.metrics import multilabel_confusion_matrix

# Example with 3 classes
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 1, 2, 1, 1, 2]

def plot_multiclass_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Multi-class Confusion Matrix')
    plt.show()

# Example usage
classes = ['Class 0', 'Class 1', 'Class 2']
plot_multiclass_confusion_matrix(y_true, y_pred, classes)
```

Slide 5: Real-world Example - Credit Card Fraud Detection

Credit card fraud detection represents a practical application of confusion matrices, where false positives and false negatives have different business impacts and costs in financial transaction classification.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Simulated credit card transaction data
np.random.seed(42)
n_samples = 1000
transactions = pd.DataFrame({
    'amount': np.random.normal(100, 50, n_samples),
    'time': np.random.uniform(0, 24, n_samples),
    'is_fraud': np.random.choice([0, 1], n_samples, p=[0.97, 0.03])
})

# Prepare data
X = transactions[['amount', 'time']]
y = transactions['is_fraud']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)
```

Slide 6: Results for Credit Card Fraud Detection

```python
# Calculate and visualize results
cm = confusion_matrix(y_test, y_pred)

# Cost matrix (example costs in dollars)
cost_matrix = np.array([
    [0, 100],    # Cost of false positive
    [1000, 0]    # Cost of false negative
])

# Calculate total cost
total_cost = np.sum(cm * cost_matrix)

print(f"Confusion Matrix:\n{cm}")
print(f"\nTotal Cost: ${total_cost:,.2f}")

# Visualize with costs
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Fraud Detection Results\nTotal Cost: ${total_cost:,.2f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

Slide 7: Real-world Example - Medical Diagnosis

Confusion matrices in medical diagnosis help evaluate diagnostic test accuracy where false negatives can have critical consequences, making it essential to optimize the balance between sensitivity and specificity.

```python
# Simulating medical diagnosis data for a disease test
np.random.seed(42)
n_patients = 1000

# Generate synthetic patient data
patient_data = pd.DataFrame({
    'age': np.random.normal(60, 15, n_patients),
    'biomarker_1': np.random.normal(100, 20, n_patients),
    'biomarker_2': np.random.normal(5, 1, n_patients),
    'has_disease': np.random.choice([0, 1], n_patients, p=[0.85, 0.15])
})

# Feature engineering
X = patient_data[['age', 'biomarker_1', 'biomarker_2']]
y = patient_data['has_disease']

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
```

Slide 8: Medical Diagnosis Results Analysis

```python
from sklearn.metrics import classification_report, roc_curve, auc

# Calculate confusion matrix and metrics
cm = confusion_matrix(y_test, y_pred)
y_pred_proba = model.predict_proba(X_test)[:,1]

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Confusion Matrix for Medical Diagnosis')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# ROC Curve
ax2.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend()

plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

Slide 9: Confusion Matrix Normalization

Normalized confusion matrices provide relative frequencies instead of absolute counts, offering better insights into class-wise performance, especially when dealing with imbalanced datasets.

```python
def plot_normalized_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Raw Counts')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Normalized values
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2)
    ax2.set_title('Normalized Values')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    return cm_normalized

# Example usage with previous medical diagnosis data
normalized_cm = plot_normalized_confusion_matrix(y_test, y_pred, ['Negative', 'Positive'])
```

Slide 10: Advanced Metrics from Confusion Matrix

The confusion matrix enables calculation of sophisticated performance metrics beyond basic accuracy, providing deeper insights into model behavior through various derived statistics.

```python
def calculate_advanced_metrics(cm):
    """
    $$Specificity = \frac{TN}{TN + FP}$$
    $$NPV = \frac{TN}{TN + FN}$$
    $$FPR = \frac{FP}{FP + TN}$$
    $$FNR = \frac{FN}{TP + FN}$$
    """
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    
    metrics = {
        'specificity': TN / (TN + FP),
        'npv': TN / (TN + FN),
        'fpr': FP / (FP + TN),
        'fnr': FN / (TP + FN),
        'diagnostic_odds_ratio': (TP * TN) / (FP * FN) if (FP * FN) != 0 else float('inf'),
        'balanced_accuracy': ((TP/(TP + FN)) + (TN/(TN + FP))) / 2
    }
    
    return metrics

# Calculate and display advanced metrics
advanced_metrics = calculate_advanced_metrics(cm)
for metric, value in advanced_metrics.items():
    print(f"{metric}: {value:.3f}")
```

Slide 11: Cross-Validation with Confusion Matrices

Cross-validation combined with confusion matrices provides a more robust evaluation of model performance by analyzing prediction patterns across multiple data folds and aggregating results.

```python
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np

def cross_validate_confusion_matrix(X, y, model, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cms = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        # Store confusion matrix
        cms.append(confusion_matrix(y_val, y_pred))
    
    # Average confusion matrices
    avg_cm = np.mean(cms, axis=0)
    std_cm = np.std(cms, axis=0)
    
    return avg_cm, std_cm, cms

# Example usage
model = RandomForestClassifier(random_state=42)
avg_cm, std_cm, all_cms = cross_validate_confusion_matrix(X, y, model)

# Visualize results
plt.figure(figsize=(10, 6))
sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues')
plt.title('Average Confusion Matrix Across Folds')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

Slide 12: Time Series Confusion Matrix Analysis

Analyzing confusion matrices over time reveals temporal patterns in model performance and helps identify periods where the model might need retraining or adjustment.

```python
def temporal_confusion_matrix_analysis(timestamps, y_true, y_pred, window_size='1D'):
    # Create DataFrame with predictions and actual values
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'actual': y_true,
        'predicted': y_pred
    })
    
    # Resample and calculate confusion matrices for each time window
    temporal_cms = {}
    
    for name, group in results_df.groupby(pd.Grouper(key='timestamp', freq=window_size)):
        if len(group) > 0:
            cm = confusion_matrix(group['actual'], group['predicted'])
            temporal_cms[name] = cm
    
    # Calculate metrics over time
    metrics_over_time = {
        'timestamp': [],
        'accuracy': [],
        'precision': [],
        'recall': []
    }
    
    for timestamp, cm in temporal_cms.items():
        TP = cm[1,1]
        TN = cm[0,0]
        FP = cm[0,1]
        FN = cm[1,0]
        
        metrics_over_time['timestamp'].append(timestamp)
        metrics_over_time['accuracy'].append((TP + TN) / (TP + TN + FP + FN))
        metrics_over_time['precision'].append(TP / (TP + FP) if (TP + FP) > 0 else 0)
        metrics_over_time['recall'].append(TP / (TP + FN) if (TP + FN) > 0 else 0)
    
    return pd.DataFrame(metrics_over_time)

# Example usage with simulated time series data
dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
y_true = np.random.choice([0, 1], size=1000)
y_pred = np.random.choice([0, 1], size=1000)

metrics_df = temporal_confusion_matrix_analysis(dates, y_true, y_pred)

# Plot temporal metrics
plt.figure(figsize=(12, 6))
for metric in ['accuracy', 'precision', 'recall']:
    plt.plot(metrics_df['timestamp'], metrics_df[metric], label=metric)
plt.legend()
plt.title('Classification Metrics Over Time')
plt.xlabel('Time')
plt.ylabel('Score')
plt.grid(True)
plt.show()
```

Slide 13: Cost-Sensitive Confusion Matrix

In real-world applications, different types of errors often have varying associated costs, requiring a cost-sensitive approach to confusion matrix analysis and model optimization.

```python
class CostSensitiveConfusionMatrix:
    def __init__(self, cost_matrix):
        """
        cost_matrix: 2D array where cost_matrix[i,j] represents the cost
        of predicting class j when true class is i
        """
        self.cost_matrix = cost_matrix
    
    def calculate_total_cost(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        total_cost = np.sum(cm * self.cost_matrix)
        return total_cost, cm
    
    def find_optimal_threshold(self, y_true, y_pred_proba, thresholds):
        best_cost = float('inf')
        best_threshold = None
        best_cm = None
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cost, cm = self.calculate_total_cost(y_true, y_pred)
            
            if cost < best_cost:
                best_cost = cost
                best_threshold = threshold
                best_cm = cm
        
        return best_threshold, best_cost, best_cm

# Example usage
cost_matrix = np.array([
    [0, 10],    # Cost of FP = 10
    [100, 0]    # Cost of FN = 100
])

cs_cm = CostSensitiveConfusionMatrix(cost_matrix)
thresholds = np.linspace(0, 1, 100)
best_threshold, best_cost, best_cm = cs_cm.find_optimal_threshold(
    y_test, 
    model.predict_proba(X_test)[:,1],
    thresholds
)

print(f"Best threshold: {best_threshold:.3f}")
print(f"Minimum total cost: ${best_cost:,.2f}")
```

Slide 14: Additional Resources

*   The Confusion Matrix, Information Gain Ratio and Cost Curves [https://arxiv.org/abs/1509.03414](https://arxiv.org/abs/1509.03414)
*   Cost-Sensitive Learning of Deep Feature Representations from Imbalanced Data [https://arxiv.org/abs/1508.03422](https://arxiv.org/abs/1508.03422)
*   Beyond Accuracy: Behavioral Testing of NLP Models with CheckList [https://arxiv.org/abs/2005.04118](https://arxiv.org/abs/2005.04118)
*   Confusion Matrix-based Feature Selection [https://arxiv.org/abs/1808.09687](https://arxiv.org/abs/1808.09687)
*   A Novel Approach to Imbalanced Classification using Cost-Sensitive Confusion Matrix [https://arxiv.org/abs/2007.12132](https://arxiv.org/abs/2007.12132)

