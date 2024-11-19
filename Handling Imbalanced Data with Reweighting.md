## Handling Imbalanced Data with Reweighting
Slide 1: Understanding Data Imbalance

Data imbalance occurs when class distributions are significantly skewed, commonly seen in fraud detection, medical diagnosis, and recommendation systems. Initial assessment involves calculating class ratios and visualizing distributions to determine appropriate handling strategies.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generate imbalanced dataset
X, y = make_classification(
    n_samples=10000, n_features=2,
    n_redundant=0, n_clusters_per_class=1,
    weights=[0.95, 0.05], random_state=42
)

# Calculate imbalance ratio
unique, counts = np.unique(y, return_counts=True)
ratio = dict(zip(unique, counts))
imbalance_ratio = min(counts) / max(counts)

print(f"Class distribution: {ratio}")
print(f"Imbalance ratio: {imbalance_ratio:.3f}")

# Visualize class distribution
plt.figure(figsize=(8, 6))
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], label='Majority Class')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], label='Minority Class')
plt.legend()
plt.title('Imbalanced Dataset Visualization')
plt.show()
```

Slide 2: Basic Reweighting Strategy

Reweighting involves assigning different weights to classes during model training to balance their influence. This approach is effective when imbalance isn't extreme and computational resources are limited.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Calculate class weights
n_samples = len(y)
n_classes = len(unique)
class_weights = {
    0: (n_samples / (n_classes * counts[0])),
    1: (n_samples / (n_classes * counts[1]))
}

# Train weighted model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(class_weight=class_weights)
model.fit(X_train, y_train)

print("Class weights:", class_weights)
print("Model accuracy:", model.score(X_test, y_test))
```

Slide 3: Implementing Random Undersampling

Random undersampling reduces majority class samples to match minority class size. This technique improves training efficiency but risks losing important information from discarded samples.

```python
from imblearn.under_sampling import RandomUnderSampler

# Apply random undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Verify new class distribution
new_unique, new_counts = np.unique(y_resampled, return_counts=True)
print("Original distribution:", dict(zip(unique, counts)))
print("Resampled distribution:", dict(zip(new_unique, new_counts)))

# Train model with balanced data
model_balanced = LogisticRegression()
model_balanced.fit(X_resampled, y_resampled)
```

Slide 4: TikTok-Style Negative Sampling

The TikTok approach prioritizes negative samples that are more likely to be confused with positive samples, improving model discrimination ability at decision boundaries.

```python
def tiktok_negative_sampling(X, y, pilot_model, sampling_rate=0.5):
    # Get predictions from pilot model
    probs = pilot_model.predict_proba(X)[:, 1]
    
    # Calculate average probability
    avg_prob = np.mean(probs[y == 0])
    
    # Calculate sampling probabilities
    sampling_probs = sampling_rate * probs / avg_prob
    
    # Sample negative instances
    neg_mask = y == 0
    random_nums = np.random.random(sum(neg_mask))
    keep_mask = random_nums < sampling_probs[neg_mask]
    
    # Combine with positive instances
    final_mask = ~neg_mask  # Keep all positive samples
    final_mask[y == 0] = keep_mask
    
    return X[final_mask], y[final_mask], sampling_probs[final_mask]
```

Slide 5: Implementing Pilot Model Training

The pilot model serves as an initial classifier trained on balanced data to estimate sample importance. Its predictions guide the intelligent selection of negative samples for final model training.

```python
def train_pilot_model(X, y):
    # Step 1: Create balanced dataset for pilot model
    rus = RandomUnderSampler(random_state=42)
    X_pilot, y_pilot = rus.fit_resample(X, y)
    
    # Step 2: Train pilot model
    pilot_model = LogisticRegression(max_iter=1000)
    pilot_model.fit(X_pilot, y_pilot)
    
    # Step 3: Evaluate pilot model
    pilot_score = pilot_model.score(X, y)
    print(f"Pilot model accuracy: {pilot_score:.3f}")
    
    return pilot_model
```

Slide 6: Probability Correction Implementation

This slide demonstrates how to implement probability correction to account for sampling bias, ensuring unbiased probability estimates despite non-uniform sampling.

```python
def correct_probabilities(raw_probs, sampling_probs):
    """
    Correct model probabilities using sampling probabilities
    Based on eq 5 from TikTok's paper
    """
    # Convert probabilities to log odds
    log_odds = np.log(raw_probs / (1 - raw_probs))
    
    # Apply correction
    corrected_log_odds = log_odds - np.log(sampling_probs)
    
    # Convert back to probabilities
    corrected_probs = 1 / (1 + np.exp(-corrected_log_odds))
    
    return corrected_probs
```

Slide 7: End-to-End Implementation

A complete implementation combining pilot model training, intelligent negative sampling, and probability correction for real-world recommendation systems.

```python
class IntelligentNegativeSampler:
    def __init__(self, sampling_rate=0.5):
        self.sampling_rate = sampling_rate
        self.pilot_model = None
        self.final_model = None
        
    def fit(self, X, y):
        # Train pilot model
        self.pilot_model = train_pilot_model(X, y)
        
        # Perform intelligent negative sampling
        X_sampled, y_sampled, self.sampling_probs = tiktok_negative_sampling(
            X, y, self.pilot_model, self.sampling_rate
        )
        
        # Train final model
        self.final_model = LogisticRegression(max_iter=1000)
        self.final_model.fit(X_sampled, y_sampled)
        
    def predict_proba(self, X):
        raw_probs = self.final_model.predict_proba(X)[:, 1]
        return correct_probabilities(raw_probs, self.sampling_probs)
```

Slide 8: Real-World Example - E-commerce Recommendations

Implementation of intelligent negative sampling for an e-commerce recommendation system, demonstrating preprocessing, training, and evaluation.

```python
def ecommerce_recommendation_example():
    # Generate synthetic e-commerce data
    n_samples = 100000
    n_features = 10
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        weights=[0.98, 0.02],  # Typical click-through rate
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model with intelligent sampling
    sampler = IntelligentNegativeSampler(sampling_rate=0.3)
    sampler.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = sampler.predict_proba(X_test)
    return y_test, y_pred_proba
```

Slide 9: Results Evaluation and Metrics

Comprehensive evaluation metrics for imbalanced classification problems, including precision-recall curves, ROC curves, and custom metrics specifically designed for recommendation systems.

```python
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import seaborn as sns

def evaluate_model_performance(y_true, y_pred_proba):
    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Precision-Recall curve
    ax1.plot(recall, precision, label=f'PR-AUC: {pr_auc:.3f}')
    ax1.set_title('Precision-Recall Curve')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.legend()
    
    # ROC curve
    ax2.plot(fpr, tpr, label=f'ROC-AUC: {roc_auc:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_title('ROC Curve')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend()
    
    return pr_auc, roc_auc
```

Slide 10: Online Learning Implementation

Implementing online learning with intelligent negative sampling for continuous model updates in production recommender systems.

```python
class OnlineIntelligentSampler:
    def __init__(self, window_size=10000):
        self.window_size = window_size
        self.X_buffer = []
        self.y_buffer = []
        self.model = None
        
    def update(self, X_new, y_new):
        # Add new samples to buffer
        self.X_buffer.extend(X_new)
        self.y_buffer.extend(y_new)
        
        # Maintain fixed window size
        if len(self.X_buffer) > self.window_size:
            self.X_buffer = self.X_buffer[-self.window_size:]
            self.y_buffer = self.y_buffer[-self.window_size:]
        
        # Retrain model if enough data
        if len(self.X_buffer) >= 1000:
            X = np.array(self.X_buffer)
            y = np.array(self.y_buffer)
            
            # Apply intelligent sampling
            sampler = IntelligentNegativeSampler()
            sampler.fit(X, y)
            self.model = sampler
            
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)
```

Slide 11: Handling Extreme Imbalance

Advanced techniques for handling extreme imbalance ratios (>1:1000) using combination of intelligent sampling and ensemble methods.

```python
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

class ExtremeImbalanceHandler:
    def __init__(self, imbalance_threshold=0.001):
        self.imbalance_threshold = imbalance_threshold
        
    def fit(self, X, y):
        # Calculate imbalance ratio
        unique, counts = np.unique(y, return_counts=True)
        imbalance_ratio = min(counts) / max(counts)
        
        if imbalance_ratio < self.imbalance_threshold:
            # Use ensemble of sampling methods
            self.model = BalancedRandomForestClassifier(
                n_estimators=100,
                sampling_strategy='auto',
                replacement=True
            )
        else:
            # Use standard intelligent sampling
            self.model = IntelligentNegativeSampler()
            
        self.model.fit(X, y)
        return self
```

Slide 12: Performance Monitoring System

Implementation of a monitoring system to track model performance and sampling efficiency in production environments, detecting concept drift and sampling bias issues.

```python
class PerformanceMonitor:
    def __init__(self, window_size=1000):
        self.metrics_history = {
            'auc_scores': [],
            'sampling_efficiency': [],
            'class_distribution': []
        }
        self.window_size = window_size
        
    def update_metrics(self, y_true, y_pred, sampling_probs):
        from sklearn.metrics import roc_auc_score
        
        # Calculate AUC score
        auc = roc_auc_score(y_true, y_pred)
        self.metrics_history['auc_scores'].append(auc)
        
        # Calculate sampling efficiency
        efficiency = np.mean(sampling_probs)
        self.metrics_history['sampling_efficiency'].append(efficiency)
        
        # Track class distribution
        class_dist = np.mean(y_true)
        self.metrics_history['class_distribution'].append(class_dist)
        
        # Detect anomalies
        if len(self.metrics_history['auc_scores']) >= self.window_size:
            self._check_for_anomalies()
            
    def _check_for_anomalies(self):
        recent_auc = np.mean(self.metrics_history['auc_scores'][-10:])
        overall_auc = np.mean(self.metrics_history['auc_scores'])
        
        if recent_auc < overall_auc * 0.9:  # 10% degradation
            print("Warning: Performance degradation detected")
```

Slide 13: Real-world Example - Social Media Engagement

Implementation of intelligent negative sampling for social media engagement prediction, including feature engineering and temporal aspects.

```python
class SocialMediaEngagementPredictor:
    def __init__(self):
        self.sampler = IntelligentNegativeSampler()
        self.monitor = PerformanceMonitor()
        
    def preprocess_features(self, data):
        features = {
            'user_activity': data['user_posts'] / np.mean(data['user_posts']),
            'content_length': np.log1p(data['content_length']),
            'time_of_day': np.cos(2 * np.pi * data['hour'] / 24),
            'day_of_week': np.sin(2 * np.pi * data['day'] / 7),
            'topic_relevance': data['topic_score']
        }
        return pd.DataFrame(features)
    
    def fit(self, X_raw, y):
        # Preprocess features
        X = self.preprocess_features(X_raw)
        
        # Train model with intelligent sampling
        self.sampler.fit(X.values, y)
        
        # Initial monitoring
        y_pred = self.sampler.predict_proba(X.values)
        self.monitor.update_metrics(y, y_pred, self.sampler.sampling_probs)
        
    def predict_proba(self, X_raw):
        X = self.preprocess_features(X_raw)
        return self.sampler.predict_proba(X.values)
```

Slide 14: Additional Resources

*   "Efficient Extreme Multi-label Classification using Negative Sampling" [https://arxiv.org/abs/2006.11015](https://arxiv.org/abs/2006.11015)
*   "Self-paced Negative Sampling for Training Deep Neural Networks" [https://arxiv.org/abs/1911.12137](https://arxiv.org/abs/1911.12137)
*   "Learning from Imbalanced Data: Open Challenges and Future Directions" [https://arxiv.org/abs/1901.04533](https://arxiv.org/abs/1901.04533)
*   "Adaptive Sampling Strategies for Recommendation Systems" Search on Google Scholar with keywords: adaptive sampling recommendation systems
*   "Deep Learning with Imbalanced Data: Strategies and Applications" Search on Google Scholar with keywords: deep learning imbalanced data strategies

