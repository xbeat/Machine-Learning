## Unlocking the Secrets of Model Performance with the Confusion Matrix
Slide 1: Understanding Confusion Matrix Fundamentals

A confusion matrix is a fundamental evaluation tool in classification tasks that presents the relationships between predicted and actual class labels through a structured matrix representation, enabling detailed performance assessment.

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_confusion_matrix(y_true, y_pred):
    # Calculate confusion matrix elements
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    # Create the matrix
    cm = np.array([[TN, FP], [FN, TP]])
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')
    return cm

# Example usage
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 1])
matrix = create_confusion_matrix(y_true, y_pred)
```

Slide 2: Performance Metrics Calculation

The confusion matrix serves as the foundation for calculating essential performance metrics that provide comprehensive insights into model behavior across different aspects of classification performance.

```python
def calculate_metrics(confusion_matrix):
    # Extract values from confusion matrix
    TN, FP, FN, TP = (
        confusion_matrix[0,0], confusion_matrix[0,1],
        confusion_matrix[1,0], confusion_matrix[1,1]
    )
    
    # Calculate core metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1 Score': f1_score
    }
    
    return metrics

# Example usage with previous confusion matrix
metrics = calculate_metrics(matrix)
for metric, value in metrics.items():
    print(f"{metric}: {value:.3f}")
```

Slide 3: Mathematical Foundations of Metrics

Understanding the mathematical relationships between confusion matrix components and derived metrics is crucial for proper model evaluation and optimization strategies.

```python
# Mathematical formulas for key metrics
"""
Accuracy = $$ \frac{TP + TN}{TP + TN + FP + FN} $$

Precision = $$ \frac{TP}{TP + FP} $$

Recall = $$ \frac{TP}{TP + FN} $$

Specificity = $$ \frac{TN}{TN + FP} $$

F1 Score = $$ 2 \times \frac{Precision \times Recall}{Precision + Recall} $$

Matthews Correlation Coefficient = $$ \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}} $$
"""

def matthews_correlation_coefficient(cm):
    TP, TN = cm[1,1], cm[0,0]
    FP, FN = cm[0,1], cm[1,0]
    
    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    
    return numerator / denominator if denominator != 0 else 0
```

Slide 4: ROC Curve Implementation

The Receiver Operating Characteristic curve provides a comprehensive visualization of model performance across different classification thresholds, enabling optimal threshold selection.

```python
import numpy as np
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_prob):
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create ROC plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    return roc_auc

# Example usage
np.random.seed(42)
y_true = np.random.randint(0, 2, 100)
y_prob = np.random.random(100)
auc_score = plot_roc_curve(y_true, y_prob)
```

Slide 5: Precision-Recall Curve Analysis

The Precision-Recall curve offers crucial insights for imbalanced classification problems, highlighting the trade-off between precision and recall across different decision thresholds.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall_curve(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    
    return avg_precision

# Example usage with previous data
ap_score = plot_precision_recall_curve(y_true, y_prob)
```

Slide 6: Cross-Validation for Robust Evaluation

Cross-validation enables comprehensive model assessment by partitioning data into multiple training and testing sets, providing statistical reliability to confusion matrix metrics across different data splits.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import numpy as np

def cross_validate_confusion_matrix(X, y, model, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cms = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cms.append(cm)
        
    # Calculate average confusion matrix
    avg_cm = np.mean(cms, axis=0)
    std_cm = np.std(cms, axis=0)
    
    return avg_cm, std_cm

# Example usage with dummy data
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)

avg_cm, std_cm = cross_validate_confusion_matrix(X, y, model)
```

Slide 7: Balanced Accuracy and Cohen's Kappa

Advanced metrics for imbalanced datasets provide more nuanced evaluation capabilities by accounting for class distribution and chance agreement in classification tasks.

```python
def advanced_metrics(confusion_matrix):
    TN, FP, FN, TP = (
        confusion_matrix[0,0], confusion_matrix[0,1],
        confusion_matrix[1,0], confusion_matrix[1,1]
    )
    
    # Calculate balanced accuracy
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # Calculate Cohen's Kappa
    total = TP + TN + FP + FN
    observed_accuracy = (TP + TN) / total
    expected_pos = ((TP + FP) * (TP + FN)) / total
    expected_neg = ((TN + FP) * (TN + FN)) / total
    expected_accuracy = (expected_pos + expected_neg) / total
    
    kappa = ((observed_accuracy - expected_accuracy) / 
             (1 - expected_accuracy) if expected_accuracy != 1 else 0)
    
    return {
        'Balanced Accuracy': balanced_accuracy,
        'Cohen\'s Kappa': kappa
    }

# Example usage
metrics = advanced_metrics(avg_cm)
for metric, value in metrics.items():
    print(f"{metric}: {value:.3f}")
```

Slide 8: Real-world Application: Credit Card Fraud Detection

Implementation of a complete fraud detection system demonstrating the practical application of confusion matrix analysis in a high-stakes financial context.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def fraud_detection_system(data_path):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model with class weight adjustment
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        class_weight='balanced',
        n_estimators=100,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix and metrics
    cm = confusion_matrix(y_test, y_pred)
    metrics = calculate_metrics(cm)
    
    return model, cm, metrics

# Example usage (assuming data availability)
"""
model, cm, metrics = fraud_detection_system('credit_card_fraud.csv')
print("Fraud Detection Results:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.3f}")
"""
```

Slide 9: Visualization Enhancement with Interactive Plots

Advanced visualization techniques for confusion matrix analysis using interactive plotting libraries to enhance understanding and interpretation of model performance.

```python
import plotly.graph_objects as go
import plotly.express as px

def interactive_confusion_matrix(cm, labels=['Negative', 'Positive']):
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Blues',
    ))
    
    # Update layout
    fig.update_layout(
        title='Interactive Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=600,
        height=600,
    )
    
    # Add annotations
    annotations = []
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=str(value),
                    showarrow=False,
                    font=dict(color='white' if value > cm.max()/2 else 'black')
                )
            )
    
    fig.update_layout(annotations=annotations)
    return fig

# Example usage with previous confusion matrix
fig = interactive_confusion_matrix(avg_cm)
# fig.show()  # Uncomment to display in notebook
```

Slide 10: Time-Series Performance Analysis

Implementing a temporal evaluation framework for monitoring classification performance over time, essential for detecting model degradation and concept drift in production systems.

```python
import pandas as pd
from datetime import datetime, timedelta

def temporal_performance_analysis(y_true, y_pred, timestamps, window_size='1D'):
    # Create DataFrame with predictions and timestamps
    df = pd.DataFrame({
        'timestamp': timestamps,
        'true': y_true,
        'pred': y_pred
    })
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Calculate rolling metrics
    def rolling_metrics(group):
        cm = confusion_matrix(group['true'], group['pred'])
        metrics = calculate_metrics(cm)
        return pd.Series(metrics)
    
    # Calculate metrics for each time window
    rolling_performance = df.set_index('timestamp').rolling(window_size).apply(
        lambda x: rolling_metrics(x)
    )
    
    # Plot temporal metrics
    plt.figure(figsize=(12, 6))
    for metric in rolling_performance.columns:
        plt.plot(rolling_performance.index, 
                rolling_performance[metric], 
                label=metric)
    
    plt.title('Classification Metrics Over Time')
    plt.xlabel('Time')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    
    return rolling_performance

# Example usage
np.random.seed(42)
n_samples = 1000
timestamps = pd.date_range(
    start='2024-01-01', 
    periods=n_samples, 
    freq='H'
)
y_true = np.random.randint(0, 2, n_samples)
y_pred = np.random.randint(0, 2, n_samples)

temporal_metrics = temporal_performance_analysis(y_true, y_pred, timestamps)
```

Slide 11: Multi-Class Confusion Matrix

Extending confusion matrix analysis to multi-class classification problems with comprehensive visualization and metric calculation capabilities.

```python
def multiclass_confusion_matrix_analysis(y_true, y_pred, classes):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate per-class metrics
    n_classes = len(classes)
    per_class_metrics = {}
    
    for i in range(n_classes):
        # Create binary confusion matrix for each class
        binary_cm = np.zeros((2, 2))
        binary_cm[1, 1] = cm[i, i]  # True Positives
        binary_cm[0, 0] = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]  # True Negatives
        binary_cm[0, 1] = np.sum(cm[:, i]) - cm[i, i]  # False Positives
        binary_cm[1, 0] = np.sum(cm[i, :]) - cm[i, i]  # False Negatives
        
        # Calculate metrics for current class
        metrics = calculate_metrics(binary_cm)
        per_class_metrics[classes[i]] = metrics
    
    # Visualize multi-class confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Multi-class Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    return cm, per_class_metrics

# Example usage
classes = ['A', 'B', 'C']
y_true_multi = np.random.choice(classes, 1000)
y_pred_multi = np.random.choice(classes, 1000)

cm_multi, class_metrics = multiclass_confusion_matrix_analysis(
    y_true_multi, y_pred_multi, classes
)

# Print per-class metrics
for class_name, metrics in class_metrics.items():
    print(f"\nMetrics for class {class_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
```

Slide 12: Cost-Sensitive Evaluation

Implementing cost-sensitive evaluation metrics that incorporate different misclassification costs, crucial for business-driven model optimization.

```python
def cost_sensitive_evaluation(confusion_matrix, cost_matrix):
    """
    Calculate cost-sensitive metrics considering different misclassification costs
    
    Parameters:
    confusion_matrix: 2D numpy array of shape (n_classes, n_classes)
    cost_matrix: 2D numpy array of same shape with costs for each type of prediction
    """
    
    # Calculate total cost
    total_cost = np.sum(confusion_matrix * cost_matrix)
    
    # Calculate normalized cost (per prediction)
    n_samples = np.sum(confusion_matrix)
    normalized_cost = total_cost / n_samples
    
    # Calculate cost-sensitive accuracy
    correct_predictions = np.sum(confusion_matrix * np.eye(confusion_matrix.shape[0]))
    total_possible_cost = np.sum(np.max(cost_matrix) * confusion_matrix)
    cost_sensitive_accuracy = 1 - (total_cost / total_possible_cost)
    
    return {
        'Total Cost': total_cost,
        'Normalized Cost': normalized_cost,
        'Cost-Sensitive Accuracy': cost_sensitive_accuracy
    }

# Example usage with binary classification
cost_matrix = np.array([
    [0, 10],    # Cost of FP is 10
    [100, 0]    # Cost of FN is 100
])

# Using previous confusion matrix
cost_metrics = cost_sensitive_evaluation(avg_cm, cost_matrix)
for metric, value in cost_metrics.items():
    print(f"{metric}: {value:.3f}")
```

Slide 13: Additional Resources

*   "A Systematic Analysis of Performance Measures for Classification Tasks" - [https://arxiv.org/abs/1906.04365](https://arxiv.org/abs/1906.04365)
*   "Beyond Accuracy: Precision and Recall" - [https://arxiv.org/abs/2001.10001](https://arxiv.org/abs/2001.10001)
*   "Cost-Sensitive Learning and the Class Imbalance Problem" - [https://arxiv.org/abs/1901.04567](https://arxiv.org/abs/1901.04567)
*   Recommended search: "Confusion Matrix Analysis Advanced Techniques"
*   For practical implementations: "sklearn.metrics documentation"

