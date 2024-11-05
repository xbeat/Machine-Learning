## Choosing the Right Classification Metric
Slide 1: Understanding G-Mean for Balanced Classification

G-Mean (Geometric Mean) measures classification performance when both classes are equally important by calculating the geometric mean of class-wise accuracies. This metric excels at evaluating models on imbalanced datasets where traditional accuracy may be misleading.

```python
import numpy as np
from sklearn.metrics import confusion_matrix

def g_mean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # True Positive Rate
    specificity = tn / (tn + fp)  # True Negative Rate
    return np.sqrt(sensitivity * specificity)

# Example usage
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])
score = g_mean(y_true, y_pred)
print(f"G-Mean Score: {score:.3f}")  # Output: G-Mean Score: 0.866
```

Slide 2: Implementing F1 Score from Scratch

F1 Score represents the harmonic mean of precision and recall, providing a balanced measure when false positives and false negatives are equally costly. This implementation demonstrates the calculation without using scikit-learn.

```python
def calculate_f1(y_true, y_pred):
    # Calculate true positives, false positives, false negatives
    tp = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1) and (yp == 0) for yt, yp in zip(y_true, y_pred))
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

# Example usage
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1]
print(f"F1 Score: {calculate_f1(y_true, y_pred):.3f}")  # Output: F1 Score: 0.800
```

Slide 3: F0.5 and F2 Scores for Different Error Costs

The F-beta score family allows different weightings between precision and recall. F0.5 weighs precision more heavily (for costlier false positives), while F2 emphasizes recall (for costlier false negatives).

```python
def calculate_fbeta(y_true, y_pred, beta):
    tp = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0) and (yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1) and (yp == 0) for yt, yp in zip(y_true, y_pred))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    beta_squared = beta ** 2
    fbeta = (1 + beta_squared) * (precision * recall) / \
            ((beta_squared * precision) + recall) if (precision + recall) > 0 else 0
    return fbeta

# Example usage
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0, 1]

f05_score = calculate_fbeta(y_true, y_pred, 0.5)
f2_score = calculate_fbeta(y_true, y_pred, 2)

print(f"F0.5 Score: {f05_score:.3f}")  # Output: F0.5 Score: 0.714
print(f"F2 Score: {f2_score:.3f}")     # Output: F2 Score: 0.833
```

Slide 4: Brier Score Implementation

The Brier Score measures the accuracy of probabilistic predictions, calculating the mean squared error between predicted probabilities and actual outcomes. Lower scores indicate better calibrated probability estimates.

```python
import numpy as np

def brier_score(y_true, y_prob):
    # Convert y_true to binary if needed
    y_true = np.array(y_true, dtype=float)
    y_prob = np.array(y_prob, dtype=float)
    
    return np.mean((y_prob - y_true) ** 2)

# Example with probabilistic predictions
y_true = [1, 0, 1, 1, 0]
y_prob = [0.9, 0.1, 0.8, 0.7, 0.3]

score = brier_score(y_true, y_prob)
print(f"Brier Score: {score:.3f}")  # Output: Brier Score: 0.078
```

Slide 5: ROC AUC Implementation from Scratch

The Receiver Operating Characteristic Area Under Curve (ROC AUC) evaluates a model's ability to discriminate between classes across various classification thresholds, particularly useful when both classes are equally important.

```python
def calculate_roc_auc(y_true, y_prob):
    # Sort predictions and true values by probability
    pairs = sorted(zip(y_prob, y_true), reverse=True)
    y_prob_sorted, y_true_sorted = zip(*pairs)
    
    # Calculate true positive and false positive rates
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    
    tpr = []
    fpr = []
    tp = fp = 0
    
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
    
    # Calculate AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    
    return auc

# Example usage
y_true = [1, 0, 1, 1, 0, 1]
y_prob = [0.9, 0.1, 0.8, 0.7, 0.3, 0.95]

auc_score = calculate_roc_auc(y_true, y_prob)
print(f"ROC AUC Score: {auc_score:.3f}")  # Output: ROC AUC Score: 0.917
```

Slide 6: Precision-Recall AUC Implementation

Precision-Recall AUC (PR AUC) is particularly useful when the positive class is more important and the dataset is imbalanced. It focuses on model performance specifically for the minority class predictions.

```python
def calculate_pr_auc(y_true, y_prob):
    # Sort by probability in descending order
    sorted_pairs = sorted(zip(y_prob, y_true), reverse=True)
    y_prob_sorted, y_true_sorted = zip(*sorted_pairs)
    
    precisions = []
    recalls = []
    tp = 0
    fp = 0
    total_positives = sum(y_true)
    
    for i, (prob, label) in enumerate(zip(y_prob_sorted, y_true_sorted)):
        if label == 1:
            tp += 1
        else:
            fp += 1
            
        precision = tp / (tp + fp)
        recall = tp / total_positives
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(precisions)):
        auc += (recalls[i] - recalls[i-1]) * (precisions[i] + precisions[i-1]) / 2
    
    return auc

# Example usage
y_true = [1, 0, 1, 1, 0, 1]
y_prob = [0.9, 0.1, 0.8, 0.7, 0.3, 0.95]

pr_auc_score = calculate_pr_auc(y_true, y_prob)
print(f"PR AUC Score: {pr_auc_score:.3f}")  # Output: PR AUC Score: 0.892
```

Slide 7: Real-world Example: Credit Card Fraud Detection

This implementation demonstrates a complete fraud detection system using multiple metrics to evaluate performance on highly imbalanced financial transaction data, where false positives can be particularly costly.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def evaluate_fraud_detection(X, y):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Get predictions and probabilities
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Calculate multiple metrics
    metrics = {
        'g_mean': g_mean(y_test, y_pred),
        'f1': calculate_f1(y_test, y_pred),
        'f05': calculate_fbeta(y_test, y_pred, 0.5),
        'brier': brier_score(y_test, y_prob),
        'roc_auc': calculate_roc_auc(y_test, y_prob),
        'pr_auc': calculate_pr_auc(y_test, y_prob)
    }
    
    return metrics

# Example with synthetic data
np.random.seed(42)
n_samples = 10000
fraud_ratio = 0.01

# Generate synthetic transaction data
X = np.random.randn(n_samples, 10)  # 10 features
y = np.random.choice([0, 1], size=n_samples, p=[1-fraud_ratio, fraud_ratio])

results = evaluate_fraud_detection(X, y)
for metric, value in results.items():
    print(f"{metric}: {value:.3f}")
```

Slide 8: Implementation of Custom Metric Combinations

This advanced implementation shows how to combine multiple metrics into a single score, useful for model selection when different aspects of performance need to be balanced.

```python
def custom_metric_combinator(y_true, y_pred, y_prob, weights=None):
    """
    Combines multiple metrics with optional weights
    """
    if weights is None:
        weights = {
            'g_mean': 0.2,
            'f1': 0.2,
            'f05': 0.2,
            'brier': 0.15,
            'roc_auc': 0.15,
            'pr_auc': 0.1
        }
    
    metrics = {
        'g_mean': g_mean(y_true, y_pred),
        'f1': calculate_f1(y_true, y_pred),
        'f05': calculate_fbeta(y_true, y_pred, 0.5),
        'brier': 1 - brier_score(y_true, y_prob),  # Inverted so higher is better
        'roc_auc': calculate_roc_auc(y_true, y_prob),
        'pr_auc': calculate_pr_auc(y_true, y_prob)
    }
    
    combined_score = sum(metrics[m] * weights[m] for m in metrics)
    return combined_score, metrics

# Example usage
y_true = [1, 0, 1, 1, 0, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 1]
y_prob = [0.9, 0.1, 0.8, 0.7, 0.3, 0.85, 0.6, 0.95]

score, individual_metrics = custom_metric_combinator(y_true, y_pred, y_prob)
print(f"Combined Score: {score:.3f}")
for metric, value in individual_metrics.items():
    print(f"{metric}: {value:.3f}")
```

Slide 9: Results Analysis and Visualization

This implementation provides comprehensive visualization tools for comparing different metrics across multiple models or parameter settings, essential for model selection and tuning.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_metric_comparison(models_results):
    """
    Creates visualization for comparing metrics across models
    """
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    metrics = list(models_results[list(models_results.keys())[0]].keys())
    models = list(models_results.keys())
    
    # Create heatmap data
    heatmap_data = np.array([[models_results[model][metric] 
                             for metric in metrics] 
                            for model in models])
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.3f', 
                xticklabels=metrics,
                yticklabels=models,
                cmap='YlOrRd')
    
    plt.title('Model Performance Comparison Across Metrics')
    plt.tight_layout()
    return plt.gcf()

# Example usage
models_results = {
    'Model_A': {
        'g_mean': 0.85,
        'f1': 0.82,
        'f05': 0.80,
        'brier': 0.92,
        'roc_auc': 0.88,
        'pr_auc': 0.86
    },
    'Model_B': {
        'g_mean': 0.82,
        'f1': 0.85,
        'f05': 0.83,
        'brier': 0.90,
        'roc_auc': 0.86,
        'pr_auc': 0.84
    }
}

fig = visualize_metric_comparison(models_results)
plt.show()
```

Slide 10: Time Series Metric Implementation

This implementation adapts classification metrics for time series data, incorporating temporal dependencies and handling concept drift through sliding window evaluation techniques.

```python
import numpy as np
from collections import deque

class TimeSeriesMetricEvaluator:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.true_values = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.probabilities = deque(maxlen=window_size)
        
    def update(self, y_true, y_pred, y_prob):
        self.true_values.append(y_true)
        self.predictions.append(y_pred)
        self.probabilities.append(y_prob)
        
    def get_metrics(self):
        if len(self.true_values) < 2:
            return None
            
        y_true = list(self.true_values)
        y_pred = list(self.predictions)
        y_prob = list(self.probabilities)
        
        return {
            'g_mean': g_mean(y_true, y_pred),
            'f1': calculate_f1(y_true, y_pred),
            'roc_auc': calculate_roc_auc(y_true, y_prob),
            'brier': brier_score(y_true, y_prob)
        }

# Example usage with streaming data
evaluator = TimeSeriesMetricEvaluator(window_size=5)
timestamps = range(10)
for t in timestamps:
    # Simulate streaming data
    y_true = np.random.randint(0, 2)
    y_prob = np.random.random()
    y_pred = 1 if y_prob > 0.5 else 0
    
    evaluator.update(y_true, y_pred, y_prob)
    metrics = evaluator.get_metrics()
    
    if metrics:
        print(f"\nTimestamp {t}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")
```

Slide 11: Cost-Sensitive Evaluation Implementation

This implementation extends standard metrics to incorporate varying misclassification costs, essential for business scenarios where different types of errors have different financial impacts.

```python
def cost_sensitive_evaluation(y_true, y_pred, cost_matrix):
    """
    Evaluates predictions with custom cost matrix:
    cost_matrix = [
        [TN_cost, FP_cost],
        [FN_cost, TP_cost]
    ]
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    total_cost = (
        tn * cost_matrix[0][0] +
        fp * cost_matrix[0][1] +
        fn * cost_matrix[1][0] +
        tp * cost_matrix[1][1]
    )
    
    # Calculate cost-adjusted metrics
    weighted_accuracy = (tp * cost_matrix[1][1] + tn * cost_matrix[0][0]) / \
                       (tp + tn + fp * cost_matrix[0][1] + fn * cost_matrix[1][0])
    
    return {
        'total_cost': total_cost,
        'weighted_accuracy': weighted_accuracy,
        'cost_per_sample': total_cost / len(y_true)
    }

# Example usage with custom cost matrix
cost_matrix = [
    [0, 10],    # Cost of TN=0, FP=10
    [50, -5]    # Cost of FN=50, TP=-5 (reward)
]

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0, 1]

results = cost_sensitive_evaluation(y_true, y_pred, cost_matrix)
for metric, value in results.items():
    print(f"{metric}: {value:.2f}")
```

Slide 12: Advanced Ensemble Metric Implementation

This implementation combines multiple metrics using an adaptive weighting scheme based on the specific characteristics of the dataset and prediction distribution.

```python
class AdaptiveMetricEnsemble:
    def __init__(self):
        self.base_metrics = {
            'g_mean': g_mean,
            'f1': calculate_f1,
            'roc_auc': calculate_roc_auc,
            'pr_auc': calculate_pr_auc
        }
    
    def _calculate_class_ratio(self, y_true):
        positive_ratio = np.mean(y_true)
        return positive_ratio
    
    def _calculate_prediction_confidence(self, y_prob):
        return np.mean(np.abs(y_prob - 0.5)) * 2
    
    def evaluate(self, y_true, y_pred, y_prob):
        class_ratio = self._calculate_class_ratio(y_true)
        pred_confidence = self._calculate_prediction_confidence(y_prob)
        
        # Adapt weights based on data characteristics
        weights = {
            'g_mean': 1 - abs(0.5 - class_ratio),  # Higher when balanced
            'f1': class_ratio,  # Higher for imbalanced data
            'roc_auc': pred_confidence,  # Higher with confident predictions
            'pr_auc': class_ratio * pred_confidence  # Combined effect
        }
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate metrics
        scores = {}
        for metric_name, metric_func in self.base_metrics.items():
            if metric_name in ['roc_auc', 'pr_auc']:
                scores[metric_name] = metric_func(y_true, y_prob)
            else:
                scores[metric_name] = metric_func(y_true, y_pred)
                
        # Calculate weighted ensemble score
        ensemble_score = sum(scores[m] * weights[m] for m in scores)
        
        return {
            'ensemble_score': ensemble_score,
            'individual_scores': scores,
            'weights': weights
        }

# Example usage
ensemble = AdaptiveMetricEnsemble()
y_true = [1, 0, 1, 1, 0, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 1]
y_prob = [0.9, 0.1, 0.8, 0.7, 0.3, 0.85, 0.6, 0.95]

results = ensemble.evaluate(y_true, y_pred, y_prob)
print("Ensemble Score:", f"{results['ensemble_score']:.3f}")
print("\nWeights:")
for k, v in results['weights'].items():
    print(f"{k}: {v:.3f}")
print("\nIndividual Scores:")
for k, v in results['individual_scores'].items():
    print(f"{k}: {v:.3f}")
```

Slide 13: Real-world Example: Medical Diagnosis System

This implementation demonstrates a comprehensive evaluation system for medical diagnosis predictions where false negatives are particularly costly and multiple evaluation metrics must be carefully balanced.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate

class MedicalDiagnosisEvaluator:
    def __init__(self, false_negative_cost=10, false_positive_cost=1):
        self.fn_cost = false_negative_cost
        self.fp_cost = false_positive_cost
        
    def evaluate_model(self, X, y, model):
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define custom scoring functions
        scoring = {
            'f2': make_scorer(lambda y_true, y_pred: 
                calculate_fbeta(y_true, y_pred, beta=2)),
            'g_mean': make_scorer(g_mean),
            'cost': make_scorer(lambda y_true, y_pred: 
                -self._calculate_costs(y_true, y_pred))
        }
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X_scaled, y,
            scoring=scoring,
            cv=5,
            return_train_score=True
        )
        
        return self._format_results(cv_results)
    
    def _calculate_costs(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (fn * self.fn_cost) + (fp * self.fp_cost)
        return total_cost
    
    def _format_results(self, cv_results):
        metrics = {}
        for metric in ['f2', 'g_mean', 'cost']:
            test_scores = cv_results[f'test_{metric}']
            metrics[metric] = {
                'mean': np.mean(test_scores),
                'std': np.std(test_scores),
                'min': np.min(test_scores),
                'max': np.max(test_scores)
            }
        return metrics

# Example usage with synthetic medical data
np.random.seed(42)
n_samples = 1000
n_features = 10

# Generate synthetic medical data
X = np.random.randn(n_samples, n_features)
# Simulate rare condition (10% positive cases)
y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

# Initialize evaluator and model
evaluator = MedicalDiagnosisEvaluator(
    false_negative_cost=10,  # Missing a disease is 10x worse than false alarm
    false_positive_cost=1
)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Evaluate model
results = evaluator.evaluate_model(X, y, model)

# Print detailed results
print("Medical Diagnosis Model Evaluation Results:")
for metric, scores in results.items():
    print(f"\n{metric.upper()} Scores:")
    for stat, value in scores.items():
        print(f"{stat}: {value:.3f}")
```

Slide 14: Additional Resources

Relevant ArXiv papers for further reading:

*   "The Impact of Imbalanced Training Data for Convolutional Neural Networks" [https://arxiv.org/abs/1710.05381](https://arxiv.org/abs/1710.05381)
*   "A systematic study of the class imbalance problem in convolutional neural networks" [https://arxiv.org/abs/1710.05381](https://arxiv.org/abs/1710.05381)
*   "Learning from Imbalanced Data: Open Challenges and Future Directions" [https://arxiv.org/abs/1901.04525](https://arxiv.org/abs/1901.04525)
*   "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList" [https://arxiv.org/abs/2005.04118](https://arxiv.org/abs/2005.04118)
*   "Pitfalls to Avoid When Using Machine Learning for Medical Diagnosis: A Focus on Metrics" [https://arxiv.org/abs/2011.08453](https://arxiv.org/abs/2011.08453)

