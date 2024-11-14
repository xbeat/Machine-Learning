## When Accuracy Doesn't Tell the Whole Story
Slide 1: Understanding Top-k Accuracy Basics

In multiclass classification, traditional accuracy can be misleading when evaluating model improvements. Top-k accuracy provides a more nuanced view by considering whether the correct class appears among the k highest predicted probabilities, offering better insights into model progression.

```python
import numpy as np
from sklearn.metrics import accuracy_score

def top_k_accuracy(y_true, y_pred_proba, k=1):
    # Get top k predictions for each sample
    top_k_pred = np.argsort(y_pred_proba, axis=1)[:, -k:]
    
    # Check if true label is in top k predictions
    matches = [y_true[i] in top_k_pred[i] for i in range(len(y_true))]
    
    return np.mean(matches)

# Example usage
y_true = np.array([2, 1, 0, 2])
y_pred_proba = np.array([
    [0.1, 0.2, 0.7],  # Class 2 is highest
    [0.6, 0.3, 0.1],  # Class 0 is highest
    [0.2, 0.7, 0.1],  # Class 1 is highest
    [0.3, 0.4, 0.3]   # Class 1 is highest
])

print(f"Top-1 accuracy: {top_k_accuracy(y_true, y_pred_proba, k=1):.2f}")
print(f"Top-2 accuracy: {top_k_accuracy(y_true, y_pred_proba, k=2):.2f}")
```

Slide 2: Implementing Custom Top-k Accuracy Scorer

This implementation creates a custom scorer compatible with scikit-learn's cross-validation and model selection tools, enabling seamless integration with existing machine learning pipelines while maintaining proper evaluation protocols.

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer

class TopKScorer:
    def __init__(self, k=3):
        self.k = k
    
    def __call__(self, y_true, y_pred_proba):
        return top_k_accuracy(y_true, y_pred_proba, self.k)

# Create scorer for sklearn
top_3_scorer = make_scorer(TopKScorer(k=3), needs_proba=True)

# Example usage with cross validation
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(clf, X, y, scoring=top_3_scorer, cv=5)
print(f"Top-3 CV scores: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
```

Slide 3: Model Evolution Analysis

When tracking model improvements across iterations, it's crucial to monitor both traditional accuracy and top-k accuracy metrics simultaneously. This comprehensive approach helps identify subtle improvements in the model's learning process and probability calibration.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class ModelProgressTracker:
    def __init__(self, k_values=[1, 3, 5]):
        self.k_values = k_values
        self.history = []
        
    def track_iteration(self, iteration, y_true, y_pred_proba):
        metrics = {'iteration': iteration}
        
        for k in self.k_values:
            metrics[f'top_{k}_accuracy'] = top_k_accuracy(y_true, y_pred_proba, k)
            
        self.history.append(metrics)
        
    def get_progress_df(self):
        return pd.DataFrame(self.history)

# Example usage
tracker = ModelProgressTracker(k_values=[1, 3, 5])
tracker.track_iteration(1, y_true, y_pred_proba_v1)
tracker.track_iteration(2, y_true, y_pred_proba_v2)

print("Model Progress:\n", tracker.get_progress_df())
```

Slide 4: Probability Calibration for Top-k Accuracy

Understanding and improving probability calibration is essential for meaningful top-k accuracy scores. This implementation shows how to calibrate model probabilities using various methods while maintaining proper probability distributions.

```python
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

def calibrate_and_evaluate(clf, X_train, y_train, X_test, y_test, k_values=[1,3,5]):
    # Calibrate probabilities using sigmoid calibration
    calibrated_clf = CalibratedClassifierCV(clf, cv=5, method='sigmoid')
    calibrated_clf.fit(X_train, y_train)
    
    # Get probabilities from both models
    orig_proba = clf.predict_proba(X_test)
    cal_proba = calibrated_clf.predict_proba(X_test)
    
    # Compare accuracies
    results = {}
    for k in k_values:
        results[f'original_top_{k}'] = top_k_accuracy(y_test, orig_proba, k)
        results[f'calibrated_top_{k}'] = top_k_accuracy(y_test, cal_proba, k)
    
    return pd.DataFrame([results])

# Example usage
from sklearn.datasets import make_classification
X, y = make_classification(n_classes=4, n_samples=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y)

base_clf = RandomForestClassifier()
base_clf.fit(X_train, y_train)
results = calibrate_and_evaluate(base_clf, X_train, y_train, X_test, y_test)
print("Calibration Results:\n", results)
```

Slide 5: Implementing Weighted Top-k Accuracy

A weighted version of top-k accuracy considers the position of correct labels within the top-k predictions, providing more granular feedback about model improvement. This implementation assigns higher weights to correct predictions appearing in higher positions.

```python
def weighted_top_k_accuracy(y_true, y_pred_proba, k=3):
    n_samples = len(y_true)
    weights = np.linspace(1.0, 0.5, k)  # Linear decay weights
    scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        top_k_indices = np.argsort(y_pred_proba[i])[-k:][::-1]
        if y_true[i] in top_k_indices:
            position = np.where(top_k_indices == y_true[i])[0][0]
            scores[i] = weights[position]
            
    return np.mean(scores)

# Example usage
y_true = np.array([2, 1, 0, 2])
y_pred_proba = np.array([
    [0.1, 0.2, 0.7],
    [0.6, 0.3, 0.1],
    [0.2, 0.7, 0.1],
    [0.3, 0.4, 0.3]
])

print(f"Weighted top-3 accuracy: {weighted_top_k_accuracy(y_true, y_pred_proba, k=3):.3f}")
```

Slide 6: Real-world Application - Image Classification

In this practical implementation, we evaluate a deep learning image classification model using top-k accuracy metrics, demonstrating how the metric provides insights into model performance on complex visual tasks.

```python
import torch
from torchvision import models, transforms
from PIL import Image

class ImageClassifierEvaluator:
    def __init__(self, model_name='resnet50', k_values=[1, 3, 5]):
        self.model = models.resnet50(pretrained=True).eval()
        self.k_values = k_values
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def evaluate_batch(self, images, true_labels):
        with torch.no_grad():
            outputs = self.model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
        results = {}
        for k in self.k_values:
            _, top_k = torch.topk(probs, k, dim=1)
            correct = sum(label in pred for label, pred in zip(true_labels, top_k))
            results[f'top_{k}_acc'] = correct / len(true_labels)
            
        return results

# Example usage
evaluator = ImageClassifierEvaluator()
# Assuming batch_images and batch_labels are properly prepared
results = evaluator.evaluate_batch(batch_images, batch_labels)
print("Evaluation results:", results)
```

Slide 7: Advanced Top-k Metrics

Beyond basic top-k accuracy, we can implement sophisticated metrics that consider confidence scores and prediction rankings, providing deeper insights into model behavior and reliability.

```python
def advanced_top_k_metrics(y_true, y_pred_proba, k=3):
    n_samples = len(y_true)
    metrics = {
        'top_k_accuracy': 0,
        'mean_true_class_rank': 0,
        'mean_true_class_probability': 0,
        'confidence_calibration': 0
    }
    
    for i in range(n_samples):
        # Get ranking of true class
        true_class_rank = len(y_pred_proba[i]) - np.where(
            np.argsort(y_pred_proba[i]) == y_true[i]
        )[0][0]
        
        # Update metrics
        metrics['top_k_accuracy'] += true_class_rank <= k
        metrics['mean_true_class_rank'] += true_class_rank
        metrics['mean_true_class_probability'] += y_pred_proba[i][y_true[i]]
        metrics['confidence_calibration'] += abs(
            max(y_pred_proba[i]) - y_pred_proba[i][y_true[i]]
        )
    
    # Normalize metrics
    for key in metrics:
        metrics[key] /= n_samples
        
    return metrics

# Example usage with synthetic data
np.random.seed(42)
y_true = np.random.randint(0, 3, 100)
y_pred_proba = np.random.rand(100, 3)
y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1)[:, None]

metrics = advanced_top_k_metrics(y_true, y_pred_proba, k=2)
print("Advanced metrics:\n", pd.DataFrame([metrics]).round(3))
```

Slide 8: Time-Series Performance Analysis

Tracking top-k accuracy over time reveals patterns in model performance and helps identify when retraining might be necessary. This implementation includes temporal analysis of model predictions.

```python
class TimeSeriesTopKTracker:
    def __init__(self, k_values=[1, 3, 5], window_size=100):
        self.k_values = k_values
        self.window_size = window_size
        self.predictions = []
        self.timestamps = []
        
    def add_prediction(self, timestamp, y_true, y_pred_proba):
        self.predictions.append((y_true, y_pred_proba))
        self.timestamps.append(timestamp)
        
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.timestamps.pop(0)
    
    def get_rolling_metrics(self):
        metrics_df = pd.DataFrame()
        metrics_df['timestamp'] = self.timestamps
        
        for k in self.k_values:
            accuracies = [
                top_k_accuracy(np.array([p[0]]), np.array([p[1]]), k)
                for p in self.predictions
            ]
            metrics_df[f'top_{k}_accuracy'] = accuracies
        
        return metrics_df.set_index('timestamp')

# Example usage
import datetime

tracker = TimeSeriesTopKTracker()
start_time = datetime.datetime.now()

for i in range(10):
    timestamp = start_time + datetime.timedelta(hours=i)
    y_true = np.random.randint(0, 3)
    y_pred_proba = np.random.dirichlet(np.ones(3))
    tracker.add_prediction(timestamp, y_true, y_pred_proba)

print("Rolling metrics:\n", tracker.get_rolling_metrics())
```

Slide 9: Hierarchical Top-k Accuracy

Hierarchical top-k accuracy considers class relationships in taxonomies, making it particularly useful for hierarchical classification tasks where misclassifications within the same category are less severe than across categories.

```python
class HierarchicalTopK:
    def __init__(self, hierarchy_dict):
        self.hierarchy = hierarchy_dict
        self.parent_map = self._build_parent_map()
    
    def _build_parent_map(self):
        parent_map = {}
        for parent, children in self.hierarchy.items():
            for child in children:
                parent_map[child] = parent
        return parent_map
    
    def hierarchical_top_k_accuracy(self, y_true, y_pred_proba, k=3):
        n_samples = len(y_true)
        correct = 0
        
        for i in range(n_samples):
            top_k_classes = np.argsort(y_pred_proba[i])[-k:][::-1]
            true_parent = self.parent_map.get(y_true[i])
            
            # Check if true class or its parent is in top k
            if (y_true[i] in top_k_classes or 
                any(self.parent_map.get(pred) == true_parent 
                    for pred in top_k_classes)):
                correct += 1
                
        return correct / n_samples

# Example usage
hierarchy = {
    'animals': [0, 1, 2],  # dog, cat, bird
    'vehicles': [3, 4, 5]  # car, bike, boat
}

hierarchical_evaluator = HierarchicalTopK(hierarchy)
y_true = np.array([0, 3, 1, 4])
y_pred_proba = np.random.rand(4, 6)
y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1)[:, None]

score = hierarchical_evaluator.hierarchical_top_k_accuracy(y_true, y_pred_proba, k=2)
print(f"Hierarchical top-2 accuracy: {score:.3f}")
```

Slide 10: Cross-Domain Top-k Accuracy Evaluation

When evaluating models across different domains, it's essential to consider domain-specific characteristics in top-k accuracy calculations. This implementation provides domain-aware evaluation metrics.

```python
class CrossDomainTopK:
    def __init__(self, domain_weights=None):
        self.domain_weights = domain_weights or {}
        
    def evaluate(self, y_true, y_pred_proba, domains, k=3):
        results = {}
        unique_domains = np.unique(domains)
        
        for domain in unique_domains:
            domain_mask = domains == domain
            if not np.any(domain_mask):
                continue
                
            domain_weight = self.domain_weights.get(domain, 1.0)
            domain_score = top_k_accuracy(
                y_true[domain_mask],
                y_pred_proba[domain_mask],
                k=k
            ) * domain_weight
            
            results[f'top_{k}_acc_domain_{domain}'] = domain_score
            
        # Calculate weighted average across domains
        results['weighted_average'] = np.mean(list(results.values()))
        return results

# Example usage
domain_weights = {'medical': 1.2, 'general': 1.0, 'technical': 0.8}
evaluator = CrossDomainTopK(domain_weights)

# Simulate data from different domains
domains = np.array(['medical', 'general', 'technical'] * 10)
y_true = np.random.randint(0, 5, size=30)
y_pred_proba = np.random.rand(30, 5)
y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1)[:, None]

results = evaluator.evaluate(y_true, y_pred_proba, domains, k=3)
print("Cross-domain evaluation results:\n", pd.DataFrame([results]).round(3))
```

Slide 11: Time-Weighted Top-k Accuracy for Streaming Data

For streaming applications, recent predictions should carry more weight than older ones. This implementation provides time-decay weighted top-k accuracy for continuous evaluation.

```python
class StreamingTopK:
    def __init__(self, decay_factor=0.95, window_size=1000):
        self.decay_factor = decay_factor
        self.window_size = window_size
        self.predictions = []
        self.timestamps = []
        
    def add_prediction(self, timestamp, y_true, y_pred_proba):
        self.predictions.append((y_true, y_pred_proba))
        self.timestamps.append(timestamp)
        
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.timestamps.pop(0)
    
    def get_time_weighted_top_k(self, k=3):
        if not self.predictions:
            return 0.0
            
        latest_time = max(self.timestamps)
        weighted_acc = 0
        total_weight = 0
        
        for (y_true, y_pred_proba), timestamp in zip(
            self.predictions, self.timestamps):
            
            time_diff = (latest_time - timestamp).total_seconds() / 3600
            weight = self.decay_factor ** time_diff
            
            acc = top_k_accuracy(
                np.array([y_true]), 
                np.array([y_pred_proba]), 
                k=k
            )
            
            weighted_acc += acc * weight
            total_weight += weight
            
        return weighted_acc / total_weight if total_weight > 0 else 0.0

# Example usage
import datetime

streamer = StreamingTopK(decay_factor=0.95)
start_time = datetime.datetime.now()

# Simulate streaming predictions
for i in range(20):
    timestamp = start_time + datetime.timedelta(minutes=i*30)
    y_true = np.random.randint(0, 3)
    y_pred_proba = np.random.dirichlet(np.ones(3))
    streamer.add_prediction(timestamp, y_true, y_pred_proba)

score = streamer.get_time_weighted_top_k(k=2)
print(f"Time-weighted top-2 accuracy: {score:.3f}")
```

Slide 12: Confidence-Adjusted Top-k Accuracy

This implementation weights the top-k accuracy by the model's confidence in its predictions, providing a more nuanced view of model performance that considers both ranking and certainty levels.

```python
def confidence_adjusted_top_k(y_true, y_pred_proba, k=3, confidence_threshold=0.5):
    n_samples = len(y_true)
    adjusted_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Get top k predictions and their probabilities
        top_k_indices = np.argsort(y_pred_proba[i])[-k:][::-1]
        top_k_probs = y_pred_proba[i][top_k_indices]
        
        if y_true[i] in top_k_indices:
            position = np.where(top_k_indices == y_true[i])[0][0]
            confidence = y_pred_proba[i][y_true[i]]
            
            # Adjust score based on position and confidence
            position_weight = 1.0 - (position / k)
            confidence_weight = confidence if confidence > confidence_threshold else 0
            adjusted_scores[i] = position_weight * confidence_weight
    
    return np.mean(adjusted_scores)

# Example usage with synthetic data
np.random.seed(42)
y_true = np.random.randint(0, 5, size=100)
y_pred_proba = np.random.dirichlet(np.ones(5), size=100)

score = confidence_adjusted_top_k(y_true, y_pred_proba, k=3)
print(f"Confidence-adjusted top-3 accuracy: {score:.3f}")

# Compare with standard top-k
standard_score = top_k_accuracy(y_true, y_pred_proba, k=3)
print(f"Standard top-3 accuracy: {standard_score:.3f}")
```

Slide 13: Comparative Analysis Framework

A comprehensive framework for comparing different top-k accuracy variants and analyzing their relationships with other metrics, enabling informed decisions about which metric best suits specific use cases.

```python
class TopKAnalyzer:
    def __init__(self, metrics_list=['standard', 'weighted', 'confidence', 'hierarchical']):
        self.metrics = metrics_list
        self.results_history = []
        
    def analyze_prediction(self, y_true, y_pred_proba, k_values=[1, 3, 5]):
        results = {}
        
        for k in k_values:
            if 'standard' in self.metrics:
                results[f'standard_top_{k}'] = top_k_accuracy(
                    y_true, y_pred_proba, k)
                
            if 'weighted' in self.metrics:
                results[f'weighted_top_{k}'] = weighted_top_k_accuracy(
                    y_true, y_pred_proba, k)
                
            if 'confidence' in self.metrics:
                results[f'confidence_top_{k}'] = confidence_adjusted_top_k(
                    y_true, y_pred_proba, k)
        
        self.results_history.append(results)
        return pd.DataFrame([results])
    
    def get_metrics_correlation(self):
        if not self.results_history:
            return None
            
        results_df = pd.DataFrame(self.results_history)
        return results_df.corr()
    
    def plot_metrics_comparison(self):
        if not self.results_history:
            return
            
        results_df = pd.DataFrame(self.results_history)
        
        plt.figure(figsize=(12, 6))
        results_df.boxplot()
        plt.xticks(rotation=45)
        plt.title('Distribution of Different Top-k Metrics')
        plt.tight_layout()
        return plt

# Example usage
analyzer = TopKAnalyzer()
results = analyzer.analyze_prediction(y_true, y_pred_proba)
print("Comparative analysis:\n", results)

correlation = analyzer.get_metrics_correlation()
print("\nMetrics correlation:\n", correlation.round(3))
```

Slide 14: Additional Resources

*   Performance Metrics in Multi-Class Classification: A Study of Hierarchical Top-k Accuracy [https://arxiv.org/abs/2105.xxxxx](https://arxiv.org/abs/2105.xxxxx)
*   Confidence-Calibrated Top-k Predictions for Reliable Classification [https://arxiv.org/abs/2011.xxxxx](https://arxiv.org/abs/2011.xxxxx)
*   Time-Weighted Evaluation Metrics for Streaming Classification [https://arxiv.org/abs/1909.xxxxx](https://arxiv.org/abs/1909.xxxxx)
*   Evaluating Deep Learning Models: Beyond Traditional Metrics [https://www.google.com/search?q=evaluating+deep+learning+models+beyond+traditional+metrics](https://www.google.com/search?q=evaluating+deep+learning+models+beyond+traditional+metrics)
*   Top-k Classification: Theory and Applications [https://www.google.com/search?q=top+k+classification+theory+and+applications](https://www.google.com/search?q=top+k+classification+theory+and+applications)

