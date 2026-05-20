## Intuitive Guide to Precision and Recall
Slide 1: Understanding Precision and Recall

Precision and recall are fundamental metrics in machine learning for evaluating classification models. This implementation demonstrates how to calculate these metrics from scratch using confusion matrix components: True Positives (TP), False Positives (FP), and False Negatives (FN).

```python
def calculate_metrics(y_true, y_pred):
    # Calculate TP, FP, FN
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall

# Example usage
import numpy as np
y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 0, 1, 1, 1])
precision, recall = calculate_metrics(y_true, y_pred)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")
```

Slide 2: Book Recommendation System - Precision Focus

This implementation showcases a simple book recommendation system prioritizing precision, ensuring that recommended books are highly likely to match user preferences based on historical ratings.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class PrecisionFocusedRecommender:
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        
    def recommend_books(self, user_ratings, book_features):
        # Calculate similarity between books
        similarities = cosine_similarity(book_features)
        
        # Get user's highly rated books (rating > 4)
        liked_books = np.where(user_ratings > 4)[0]
        
        recommendations = []
        for book_idx in range(len(book_features)):
            if book_idx not in liked_books:
                # Only recommend if similarity is above threshold
                avg_similarity = np.mean([similarities[book_idx][liked] 
                                       for liked in liked_books])
                if avg_similarity > self.threshold:
                    recommendations.append(book_idx)
                    
        return recommendations

# Example usage
book_features = np.random.rand(10, 5)  # 10 books, 5 features
user_ratings = np.array([5, 2, 4, 5, 1, 3, 2, 4, 5, 1])
recommender = PrecisionFocusedRecommender()
recommendations = recommender.recommend_books(user_ratings, book_features)
```

Slide 3: Interview Shortlisting System - Recall Focus

A recall-focused candidate screening system that prioritizes not missing any potentially qualified candidates, implementing a more lenient classification threshold to maximize recall at the expense of precision.

```python
import numpy as np

class RecallFocusedScreening:
    def __init__(self, threshold=0.3):  # Lower threshold for high recall
        self.threshold = threshold
        
    def screen_candidates(self, candidates_scores, skills_match):
        # Combine multiple criteria with bias towards inclusion
        qualified = np.zeros(len(candidates_scores), dtype=bool)
        
        for i in range(len(candidates_scores)):
            # Accept if either score is good or skills partially match
            if (candidates_scores[i] >= self.threshold or 
                np.mean(skills_match[i]) >= 0.5):
                qualified[i] = True
                
        return qualified

# Example usage
candidates_scores = np.array([0.45, 0.35, 0.25, 0.55, 0.30])
skills_match = np.array([
    [1, 0, 1],  # Candidate 1 skills match
    [0, 1, 0],  # Candidate 2 skills match
    [1, 1, 0],  # etc.
    [0, 0, 1],
    [1, 0, 1]
])

screener = RecallFocusedScreening()
shortlisted = screener.screen_candidates(candidates_scores, skills_match)
print(f"Shortlisted candidates: {np.where(shortlisted)[0]}")
```

Slide 4: Performance Metrics Implementation

Deep dive into implementing comprehensive performance metrics including precision, recall, F1-score, and confusion matrix visualization for binary classification problems.

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class PerformanceMetrics:
    def __init__(self):
        self.metrics = {}
        
    def calculate_all_metrics(self, y_true, y_pred):
        # Confusion matrix elements
        self.tp = np.sum((y_true == 1) & (y_pred == 1))
        self.fp = np.sum((y_true == 0) & (y_pred == 1))
        self.fn = np.sum((y_true == 1) & (y_pred == 0))
        self.tn = np.sum((y_true == 0) & (y_pred == 0))
        
        # Calculate metrics
        self.metrics['precision'] = self.tp / (self.tp + self.fp)
        self.metrics['recall'] = self.tp / (self.tp + self.fn)
        self.metrics['f1'] = 2 * (self.metrics['precision'] * self.metrics['recall']) / \
                            (self.metrics['precision'] + self.metrics['recall'])
        
        return self.metrics
    
    def plot_confusion_matrix(self):
        cm = np.array([[self.tn, self.fp], 
                      [self.fn, self.tp]])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

# Example usage
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1])
y_pred = np.array([1, 0, 0, 1, 1, 1, 0, 1])

metrics = PerformanceMetrics()
results = metrics.calculate_all_metrics(y_true, y_pred)
print("Metrics:", results)
metrics.plot_confusion_matrix()
```

Slide 5: Cross-Validation with Precision-Recall Trade-off

Implementation of k-fold cross-validation focusing on the precision-recall trade-off, allowing model evaluation across different classification thresholds.

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve

class PrecisionRecallCV:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
    
    def evaluate_model(self, X, y, model):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        precisions = []
        recalls = []
        thresholds = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model and get probabilities
            model.fit(X_train, y_train)
            y_scores = model.predict_proba(X_val)[:, 1]
            
            # Calculate precision-recall curve
            p, r, t = precision_recall_curve(y_val, y_scores)
            precisions.append(p)
            recalls.append(r)
            thresholds.append(t)
            
        return np.mean(precisions, axis=0), np.mean(recalls, axis=0)

# Example usage
from sklearn.linear_model import LogisticRegression
X = np.random.rand(100, 5)
y = (X.sum(axis=1) > 2.5).astype(int)

evaluator = PrecisionRecallCV()
model = LogisticRegression()
mean_precision, mean_recall = evaluator.evaluate_model(X, y, model)

plt.figure(figsize=(10, 6))
plt.plot(mean_recall, mean_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()
```

Slide 6: Medical Diagnosis System - High Recall Priority

A practical implementation of a medical screening system where missing a positive case (false negative) could have serious consequences. This system prioritizes recall to ensure potentially serious conditions aren't overlooked.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

class MedicalScreeningSystem:
    def __init__(self, base_threshold=0.3, risk_weights=None):
        self.base_threshold = base_threshold
        self.risk_weights = risk_weights or {}
        self.scaler = StandardScaler()
        
    def process_patient_data(self, symptoms, risk_factors):
        # Normalize symptom indicators
        normalized_symptoms = self.scaler.fit_transform(symptoms)
        
        # Calculate risk-adjusted scores
        risk_scores = np.zeros(len(symptoms))
        for i, (symptom, risk) in enumerate(zip(normalized_symptoms, risk_factors)):
            base_score = np.mean(symptom)
            risk_multiplier = sum(self.risk_weights.get(r, 1) for r in risk[i])
            risk_scores[i] = base_score * risk_multiplier
            
        # Use lower threshold for high-risk cases
        return risk_scores > (self.base_threshold / np.sqrt(1 + risk_scores))

# Example usage
symptoms = np.array([
    [0.7, 0.3, 0.8],  # Patient 1 symptoms
    [0.2, 0.1, 0.3],  # Patient 2 symptoms
    [0.5, 0.6, 0.4]   # Patient 3 symptoms
])

risk_factors = [
    ['age_65+', 'diabetes'],
    ['smoker'],
    ['hypertension', 'obesity']
]

screener = MedicalScreeningSystem(
    risk_weights={'age_65+': 1.5, 'diabetes': 1.3, 'hypertension': 1.2}
)

results = screener.process_patient_data(symptoms, risk_factors)
print(f"Patients requiring further examination: {np.where(results)[0]}")
```

Slide 7: Metrics Visualization Dashboard

Advanced visualization system for tracking precision and recall metrics over time, including interactive components for threshold adjustment and performance monitoring.

```python
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class MetricsDashboard:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.metrics_history = pd.DataFrame(
            columns=['timestamp', 'precision', 'recall', 'threshold']
        )
        
    def update_metrics(self, y_true, y_pred, threshold):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        new_record = pd.DataFrame({
            'timestamp': [datetime.now()],
            'precision': [precision],
            'recall': [recall],
            'threshold': [threshold]
        })
        
        self.metrics_history = pd.concat(
            [self.metrics_history, new_record], 
            ignore_index=True
        )
        
    def plot_metrics_trend(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.metrics_history['precision'], 
                label='Precision', marker='o')
        plt.plot(self.metrics_history['recall'], 
                label='Recall', marker='s')
        plt.axhline(y=0.5, color='r', linestyle='--', 
                   label='Minimum Acceptable')
        
        plt.title('Precision-Recall Trends')
        plt.xlabel('Evaluation Period')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
# Example usage
dashboard = MetricsDashboard()

# Simulate metrics over time
np.random.seed(42)
for day in range(30):
    y_true = np.random.binomial(1, 0.3, 100)
    threshold = 0.5 + np.sin(day/10) * 0.1
    y_pred = np.random.binomial(1, threshold, 100)
    dashboard.update_metrics(y_true, y_pred, threshold)

dashboard.plot_metrics_trend()
```

Slide 8: Ensemble Voting with Precision-Recall Optimization

Implementation of an ensemble voting system that dynamically adjusts voting weights based on individual model precision and recall performance.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class PrecisionRecallEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, models, precision_weight=0.5):
        self.models = models
        self.precision_weight = precision_weight
        self.model_weights = None
        
    def fit(self, X, y):
        self.model_weights = np.ones(len(self.models)) / len(self.models)
        
        # Train each model and calculate weights
        for i, model in enumerate(self.models):
            model.fit(X, y)
            y_pred = model.predict(X)
            
            tp = np.sum((y == 1) & (y_pred == 1))
            fp = np.sum((y == 0) & (y_pred == 1))
            fn = np.sum((y == 1) & (y_pred == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Calculate weighted score
            score = (self.precision_weight * precision + 
                    (1 - self.precision_weight) * recall)
            self.model_weights[i] = score
            
        # Normalize weights
        self.model_weights /= np.sum(self.model_weights)
        return self
        
    def predict_proba(self, X):
        predictions = np.zeros((X.shape[0], 2))
        
        for model, weight in zip(self.models, self.model_weights):
            pred = model.predict_proba(X)
            predictions += weight * pred
            
        return predictions
        
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# Example usage
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

X = np.random.rand(100, 5)
y = (X.sum(axis=1) > 2.5).astype(int)

models = [
    DecisionTreeClassifier(random_state=42),
    SVC(probability=True, random_state=42),
    LogisticRegression(random_state=42)
]

ensemble = PrecisionRecallEnsemble(models, precision_weight=0.7)
ensemble.fit(X, y)
predictions = ensemble.predict(X)
```

Slide 9: Time Series Anomaly Detection with Precision Focus

This implementation demonstrates anomaly detection in time series data with emphasis on precision, ensuring that flagged anomalies are highly likely to be true positives using statistical methods.

```python
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

class PrecisionFocusedAnomalyDetector:
    def __init__(self, window_size=10, sigma_threshold=3):
        self.window_size = window_size
        self.sigma_threshold = sigma_threshold
        self.scaler = StandardScaler()
        
    def detect_anomalies(self, time_series):
        # Standardize the data
        scaled_data = self.scaler.fit_transform(time_series.reshape(-1, 1))
        
        anomalies = np.zeros(len(time_series), dtype=bool)
        z_scores = np.zeros(len(time_series))
        
        for i in range(self.window_size, len(time_series)):
            window = scaled_data[i-self.window_size:i]
            mean = np.mean(window)
            std = np.std(window)
            
            z_score = np.abs((scaled_data[i] - mean) / std)
            z_scores[i] = z_score
            
            # Only flag if multiple conditions are met
            if (z_score > self.sigma_threshold and 
                stats.normaltest(window)[1] < 0.05):
                anomalies[i] = True
                
        return anomalies, z_scores

# Example usage
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
# Insert some anomalies
anomaly_indices = [100, 300, 600, 800]
normal_data[anomaly_indices] = [4, -5, 6, -4]

detector = PrecisionFocusedAnomalyDetector()
anomalies, z_scores = detector.detect_anomalies(normal_data)

print(f"Number of anomalies detected: {np.sum(anomalies)}")
print(f"Anomaly indices: {np.where(anomalies)[0]}")

# Visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.plot(normal_data, label='Time Series')
plt.scatter(np.where(anomalies)[0], 
           normal_data[anomalies], 
           color='red', 
           label='Anomalies')
plt.legend()
plt.title('Time Series Anomaly Detection')
plt.show()
```

Slide 10: Recall-Focused Text Classification

Implementation of a text classification system optimized for recall, particularly useful for content moderation where missing harmful content is more costly than false positives.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class RecallFocusedTextClassifier:
    def __init__(self, base_threshold=0.3, 
                 sensitivity_words=None):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.base_threshold = base_threshold
        self.sensitivity_words = sensitivity_words or []
        self.word_weights = {}
        
    def fit(self, texts, labels):
        # Transform texts to TF-IDF features
        X = self.vectorizer.fit_transform(texts)
        
        # Calculate word importance weights
        feature_names = self.vectorizer.get_feature_names_out()
        for word in feature_names:
            if word in self.sensitivity_words:
                self.word_weights[word] = 2.0
            else:
                self.word_weights[word] = 1.0
                
        # Train simple Naive Bayes-like weights
        self.class_weights = np.zeros(len(feature_names))
        positive_docs = X[labels == 1]
        negative_docs = X[labels == 0]
        
        self.class_weights = (positive_docs.mean(axis=0) - 
                            negative_docs.mean(axis=0)).A1
        
        return self
        
    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        scores = X.dot(self.class_weights)
        
        # Apply word-specific thresholds
        predictions = np.zeros(len(texts), dtype=int)
        feature_names = self.vectorizer.get_feature_names_out()
        
        for i, text in enumerate(texts):
            words = set(text.lower().split())
            threshold = self.base_threshold
            
            # Lower threshold if sensitive words present
            for word in words.intersection(self.sensitivity_words):
                threshold *= 0.8
                
            predictions[i] = 1 if scores[i] > threshold else 0
            
        return predictions

# Example usage
texts = [
    "This is a normal message",
    "This contains sensitive content",
    "Another regular message",
    "More sensitive and harmful content",
    "Just a regular update"
]
labels = np.array([0, 1, 0, 1, 0])

classifier = RecallFocusedTextClassifier(
    sensitivity_words=['sensitive', 'harmful']
)
classifier.fit(texts, labels)

new_texts = [
    "Some sensitive information here",
    "Regular message about weather",
    "Potentially harmful content"
]
predictions = classifier.predict(new_texts)
print("Predictions:", predictions)
```

Slide 11: Real-time Monitoring System with Precision-Recall Balance

An advanced implementation of a real-time monitoring system that dynamically adjusts its precision-recall trade-off based on observed error patterns and cost metrics.

```python
import numpy as np
from collections import deque
from datetime import datetime, timedelta

class AdaptiveMonitoringSystem:
    def __init__(self, window_size=100, initial_threshold=0.5,
                 false_positive_cost=1, false_negative_cost=2):
        self.window_size = window_size
        self.threshold = initial_threshold
        self.fp_cost = false_positive_cost
        self.fn_cost = false_negative_cost
        self.history = deque(maxlen=window_size)
        self.metrics_history = []
        
    def update_threshold(self, y_true, y_pred):
        # Calculate current costs
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        total_cost = fp * self.fp_cost + fn * self.fn_cost
        
        # Adjust threshold based on cost
        if total_cost > 0:
            fp_ratio = fp / (fp + fn)
            self.threshold += 0.01 if fp_ratio > 0.5 else -0.01
            self.threshold = np.clip(self.threshold, 0.2, 0.8)
            
        # Store metrics
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'threshold': self.threshold,
            'total_cost': total_cost
        })
        
    def predict(self, features, scores):
        predictions = (scores > self.threshold).astype(int)
        return predictions
    
    def monitor(self, features, scores, true_labels=None):
        predictions = self.predict(features, scores)
        
        if true_labels is not None:
            self.update_threshold(true_labels, predictions)
            self.history.append((predictions, true_labels))
            
        return predictions, self.threshold

# Example usage
np.random.seed(42)

# Simulate real-time data
def generate_data(n_samples, drift_factor=0.1):
    time = np.linspace(0, 10, n_samples)
    features = np.column_stack([
        np.sin(time) + np.random.normal(0, 0.1, n_samples),
        np.cos(time) + np.random.normal(0, 0.1, n_samples)
    ])
    
    # Add concept drift
    drift = drift_factor * time
    true_labels = (features.sum(axis=1) + drift > 0).astype(int)
    scores = features.sum(axis=1) + np.random.normal(0, 0.2, n_samples)
    
    return features, scores, true_labels

# Run simulation
monitor = AdaptiveMonitoringSystem()
n_samples = 1000
metrics = []

for i in range(0, n_samples, 10):  # Process in batches
    features, scores, labels = generate_data(10, drift_factor=0.02)
    preds, threshold = monitor.monitor(features, scores, labels)
    
    metrics.append({
        'batch': i // 10,
        'threshold': threshold,
        'accuracy': np.mean(preds == labels)
    })

# Visualize results
import matplotlib.pyplot as plt
metrics_df = pd.DataFrame(metrics)
plt.figure(figsize=(12, 6))
plt.plot(metrics_df['threshold'], label='Threshold')
plt.plot(metrics_df['accuracy'], label='Accuracy')
plt.xlabel('Batch')
plt.ylabel('Value')
plt.title('Adaptive Monitoring System Performance')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 12: Hybrid Precision-Recall Optimization

A sophisticated implementation that combines multiple optimization strategies to achieve balanced precision-recall performance while adapting to changing data patterns.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

class HybridOptimizer(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, target_precision=0.8,
                 target_recall=0.8, adaptation_rate=0.01):
        self.base_estimator = base_estimator
        self.target_precision = target_precision
        self.target_recall = target_recall
        self.adaptation_rate = adaptation_rate
        self.scaler = StandardScaler()
        self.thresholds = {}
        
    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train base estimator
        self.base_estimator.fit(X_scaled, y)
        
        # Initialize dynamic thresholds
        proba = self.base_estimator.predict_proba(X_scaled)
        self._optimize_thresholds(proba, y)
        
        return self
    
    def _optimize_thresholds(self, probabilities, y_true):
        pos_scores = probabilities[y_true == 1, 1]
        neg_scores = probabilities[y_true == 0, 1]
        
        # Calculate initial thresholds
        precision_threshold = np.percentile(neg_scores, 
            (1 - self.target_precision) * 100)
        recall_threshold = np.percentile(pos_scores, 
            (1 - self.target_recall) * 100)
        
        self.thresholds = {
            'precision': precision_threshold,
            'recall': recall_threshold,
            'hybrid': (precision_threshold + recall_threshold) / 2
        }
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.base_estimator.predict_proba(X_scaled)
    
    def predict(self, X, optimize_for='hybrid'):
        proba = self.predict_proba(X)
        threshold = self.thresholds.get(optimize_for, 0.5)
        return (proba[:, 1] >= threshold).astype(int)
    
    def update_thresholds(self, X, y):
        proba = self.predict_proba(X)
        
        # Calculate current metrics
        preds_p = (proba[:, 1] >= self.thresholds['precision']).astype(int)
        preds_r = (proba[:, 1] >= self.thresholds['recall']).astype(int)
        
        tp_p = np.sum((y == 1) & (preds_p == 1))
        fp_p = np.sum((y == 0) & (preds_p == 1))
        fn_p = np.sum((y == 1) & (preds_p == 0))
        
        precision = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else 0
        recall = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0
        
        # Update thresholds
        if precision < self.target_precision:
            self.thresholds['precision'] += self.adaptation_rate
        if recall < self.target_recall:
            self.thresholds['recall'] -= self.adaptation_rate
            
        self.thresholds['hybrid'] = (self.thresholds['precision'] + 
                                   self.thresholds['recall']) / 2

# Example usage
from sklearn.ensemble import RandomForestClassifier

X = np.random.rand(1000, 5)
y = (X.sum(axis=1) > 2.5).astype(int)

optimizer = HybridOptimizer(
    RandomForestClassifier(n_estimators=100, random_state=42)
)
optimizer.fit(X, y)

# Test different optimization strategies
preds_precision = optimizer.predict(X, optimize_for='precision')
preds_recall = optimizer.predict(X, optimize_for='recall')
preds_hybrid = optimizer.predict(X, optimize_for='hybrid')
```

Slide 13: Online Learning with Precision-Recall Adaptation

This implementation demonstrates an online learning system that continuously adapts its precision-recall trade-off based on streaming data, suitable for real-world applications with evolving patterns.

```python
import numpy as np
from collections import deque
from sklearn.linear_model import SGDClassifier

class OnlinePROptimizer:
    def __init__(self, feature_dim, buffer_size=1000,
                 pr_weight=0.5, learning_rate=0.01):
        self.model = SGDClassifier(loss='log_loss', 
                                 learning_rate='constant',
                                 eta0=learning_rate)
        self.buffer = deque(maxlen=buffer_size)
        self.pr_weight = pr_weight
        self.thresholds = np.linspace(0, 1, 100)
        self.performance_history = []
        
    def partial_fit(self, X, y):
        # Update model
        self.model.partial_fit(X, y, classes=[0, 1])
        
        # Store in buffer for threshold optimization
        proba = self.model.predict_proba(X)
        for p, label in zip(proba, y):
            self.buffer.append((p[1], label))
            
        # Optimize threshold
        self._optimize_threshold()
        
        return self
    
    def _optimize_threshold(self):
        if len(self.buffer) < 100:
            return
            
        scores, labels = zip(*self.buffer)
        scores = np.array(scores)
        labels = np.array(labels)
        
        best_score = -np.inf
        best_threshold = 0.5
        
        for threshold in self.thresholds:
            preds = (scores >= threshold).astype(int)
            
            tp = np.sum((labels == 1) & (preds == 1))
            fp = np.sum((labels == 0) & (preds == 1))
            fn = np.sum((labels == 1) & (preds == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Combined score
            score = (self.pr_weight * precision + 
                    (1 - self.pr_weight) * recall)
                    
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
        self.current_threshold = best_threshold
        self.performance_history.append({
            'threshold': best_threshold,
            'score': best_score
        })
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.current_threshold).astype(int)

# Example usage with streaming data
np.random.seed(42)

# Generate streaming data
def generate_stream(n_samples, noise_level=0.1):
    X = np.random.randn(n_samples, 5)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    # Add noise
    flip_idx = np.random.choice(n_samples, 
                               int(n_samples * noise_level), 
                               replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    return X, y

# Simulate online learning
optimizer = OnlinePROptimizer(feature_dim=5)
batch_size = 50
metrics = []

for i in range(20):
    X_batch, y_batch = generate_stream(batch_size)
    optimizer.partial_fit(X_batch, y_batch)
    
    # Evaluate
    preds = optimizer.predict(X_batch)
    accuracy = np.mean(preds == y_batch)
    metrics.append({
        'batch': i,
        'accuracy': accuracy,
        'threshold': optimizer.current_threshold
    })

# Visualization
import matplotlib.pyplot as plt
metrics_df = pd.DataFrame(metrics)
plt.figure(figsize=(10, 5))
plt.plot(metrics_df['accuracy'], label='Accuracy')
plt.plot(metrics_df['threshold'], label='Threshold')
plt.xlabel('Batch')
plt.ylabel('Value')
plt.title('Online Learning Performance')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 14: Additional Resources

*   "Deep Learning with Precision-Recall Trade-offs" - [https://arxiv.org/abs/2012.08797](https://arxiv.org/abs/2012.08797)
*   "Online Learning for Precision-Recall Optimization" - [https://arxiv.org/abs/2103.11567](https://arxiv.org/abs/2103.11567)
*   "Adaptive Threshold Optimization for Binary Classification" - [https://arxiv.org/abs/1911.09862](https://arxiv.org/abs/1911.09862)
*   Advanced resources for real-world applications:
    *   Google Scholar: "precision recall optimization deep learning"
    *   Research papers on adaptive thresholding techniques
    *   Documentation on scikit-learn's precision\_recall\_curve implementation
*   Practical guides:
    *   [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision\_recall\_curve.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
    *   [https://pytorch.org/tutorials/beginner/blitz/neural\_networks\_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
*   Recommended textbooks:
    *   "Pattern Recognition and Machine Learning" by Christopher Bishop
    *   "Deep Learning" by Goodfellow, Bengio, and Courville

