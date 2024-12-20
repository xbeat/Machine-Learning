## Precision-Recall Plot Evaluating Classifier Performance
Slide 1: Understanding Precision and Recall

Precision and recall are fundamental metrics in binary classification problems. Precision measures the accuracy of positive predictions, while recall measures the ability to identify all positive instances. These metrics form the foundation for understanding classifier performance through visualization.

```python
def calculate_precision_recall(y_true, y_pred):
    # Calculate true positives, false positives, false negatives
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall
```

Slide 2: Mathematical Foundation of Precision-Recall

The mathematical formulation of precision and recall provides the theoretical basis for the precision-recall curve. These metrics are derived from the confusion matrix and represent different aspects of model performance.

```python
# Mathematical formulas in LaTeX notation
"""
Precision formula:
$$Precision = \frac{TP}{TP + FP}$$

Recall formula:
$$Recall = \frac{TP}{TP + FN}$$

F1-Score formula:
$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$
"""
```

Slide 3: Implementing a Basic Precision-Recall Curve

The precision-recall curve visualizes the trade-off between precision and recall across different classification thresholds. This implementation demonstrates how to generate the curve using sklearn and matplotlib.

```python
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def plot_precision_recall_curve(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, 'b-', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()
    
    return precision, recall, thresholds
```

Slide 4: Random Baseline Calculation

A key component of precision-recall analysis is understanding the random baseline. This represents the performance of a random classifier and helps contextualize the actual model's performance against chance.

```python
def calculate_random_baseline(y_true):
    # Random baseline is the proportion of positive samples
    n_positive = sum(y_true == 1)
    total_samples = len(y_true)
    
    random_baseline = n_positive / total_samples
    return random_baseline
```

Slide 5: Real-World Example - Binary Classification

This example demonstrates a complete implementation of precision-recall analysis using a real-world dataset for spam classification, including data preprocessing and model training.

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load and preprocess spam dataset
def prepare_spam_dataset():
    # Example spam dataset
    data = pd.DataFrame({
        'text': ['Free money now!', 'Meeting tomorrow', 'Win prizes!!!', 'Project update'],
        'label': [1, 0, 1, 0]
    })
    
    # Vectorize text
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['text'])
    y = data['label']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
```

Slide 6: Model Training and Evaluation

Implementing the complete training pipeline and generating precision-recall curves for the spam classification model, including threshold optimization and performance metrics.

```python
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Get probability scores
    y_scores = model.predict_proba(X_test)[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    
    return precision, recall, thresholds, model
```

Slide 7: Area Under Precision-Recall Curve (AUPRC)

The Area Under the Precision-Recall Curve (AUPRC) provides a single-number summary of classifier performance. This implementation shows how to calculate and interpret AUPRC.

```python
from sklearn.metrics import auc

def calculate_auprc(precision, recall):
    auprc = auc(recall, precision)
    
    # Visualize AUPRC
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, 'b-', label=f'AUPRC = {auprc:.3f}')
    plt.fill_between(recall, precision, alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Area Under Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return auprc
```

Slide 8: Optimal Threshold Selection

Finding the optimal classification threshold is crucial for model deployment. This implementation demonstrates how to select the best threshold using the F1 score as an optimization metric.

```python
def find_optimal_threshold(precision, recall, thresholds):
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    
    # Find threshold that maximizes F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plot F1 scores vs thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores[:-1], 'r-')
    plt.axvline(optimal_threshold, color='g', linestyle='--', 
                label=f'Optimal threshold: {optimal_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Classification Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return optimal_threshold
```

Slide 9: Handling Class Imbalance

Class imbalance significantly affects precision-recall curves. This implementation shows techniques to address imbalanced datasets and their impact on precision-recall analysis.

```python
from sklearn.utils import resample

def handle_imbalance(X, y):
    # Separate majority and minority classes
    X_majority = X[y == 0]
    X_minority = X[y == 1]
    y_majority = y[y == 0]
    y_minority = y[y == 1]
    
    # Upsample minority class
    X_minority_upsampled, y_minority_upsampled = resample(
        X_minority, y_minority,
        replace=True,
        n_samples=len(X_majority),
        random_state=42
    )
    
    # Combine majority and upsampled minority
    X_balanced = np.vstack([X_majority, X_minority_upsampled])
    y_balanced = np.hstack([y_majority, y_minority_upsampled])
    
    return X_balanced, y_balanced
```

Slide 10: Real-World Example - Credit Card Fraud Detection

A comprehensive implementation of precision-recall analysis for credit card fraud detection, showcasing handling of severe class imbalance and threshold optimization.

```python
def credit_fraud_detection():
    # Simulate credit card transaction data
    np.random.seed(42)
    n_samples = 10000
    
    # Generate features (amount, time, etc.)
    X = np.random.randn(n_samples, 5)
    
    # Create imbalanced labels (0.1% fraud rate)
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, size=int(0.001*n_samples), replace=False)
    y[fraud_indices] = 1
    
    # Split and preprocess
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    return X_train, X_test, y_train, y_test
```

Slide 11: Performance Visualization

Advanced visualization techniques for precision-recall analysis, including confidence intervals and comparison with multiple models.

```python
def visualize_model_comparison(models_dict, X_test, y_test):
    plt.figure(figsize=(12, 8))
    
    for name, model in models_dict.items():
        y_scores = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        auprc = auc(recall, precision)
        
        plt.plot(recall, precision, label=f'{name} (AUPRC = {auprc:.3f})')
    
    # Add random baseline
    baseline = sum(y_test == 1) / len(y_test)
    plt.plot([0, 1], [baseline, baseline], 'r--', label='Random Baseline')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Model Comparison using Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    plt.show()
```

Slide 12: Confidence Intervals for Precision-Recall

Implementing bootstrap-based confidence intervals for precision-recall curves to assess model stability and reliability.

```python
def calculate_pr_confidence_intervals(model, X, y, n_bootstraps=1000):
    n_samples = len(y)
    precisions = []
    recalls = []
    
    for _ in range(n_bootstraps):
        # Bootstrap sampling
        indices = np.random.randint(0, n_samples, n_samples)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Calculate precision-recall
        y_scores = model.predict_proba(X_boot)[:, 1]
        precision, recall, _ = precision_recall_curve(y_boot, y_scores)
        precisions.append(precision)
        recalls.append(recall)
    
    return np.array(precisions), np.array(recalls)
```

Slide 13: Additional Resources

*   Document: "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets" [https://arxiv.org/abs/1504.06375](https://arxiv.org/abs/1504.06375)
*   Paper: "Analysis of Precision-Recall Curves Through the Lens of Active Learning" [https://arxiv.org/abs/2103.12797](https://arxiv.org/abs/2103.12797)
*   Research: "On the Relationship Between Precision-Recall and ROC Curves" [https://scholar.google.com/scholar?q=precision+recall+curves+machine+learning](https://scholar.google.com/scholar?q=precision+recall+curves+machine+learning)
*   Tutorial: "Understanding Precision-Recall Curves in Machine Learning" [https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification/](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification/)

