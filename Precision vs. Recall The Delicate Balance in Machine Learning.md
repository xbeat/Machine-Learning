## Precision vs. Recall The Delicate Balance in Machine Learning
Slide 1: Understanding Precision in Classification

Precision measures the accuracy of positive predictions by calculating the ratio of true positives to all positive predictions. It's crucial in applications where false positives are costly, such as spam detection or fraud identification systems where wrongly flagging legitimate cases has significant consequences.

```python
import numpy as np
from sklearn.metrics import precision_score

def calculate_precision(y_true, y_pred):
    # True positives and false positives
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    
    # Calculate precision
    precision = true_positives / (true_positives + false_positives)
    
    # Example usage
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1, 1])
    
    print(f"Manual Precision: {precision}")
    print(f"Sklearn Precision: {precision_score(y_true, y_pred)}")
    
    return precision

# Output:
# Manual Precision: 0.75
# Sklearn Precision: 0.75
```

Slide 2: Understanding Recall in Classification

Recall quantifies the model's ability to find all relevant cases by measuring the ratio of true positives to all actual positive cases. This metric is vital in medical diagnosis or security threat detection where missing positive cases could have severe consequences.

```python
import numpy as np
from sklearn.metrics import recall_score

def calculate_recall(y_true, y_pred):
    # True positives and false negatives
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate recall
    recall = true_positives / (true_positives + false_negatives)
    
    # Example usage
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1, 1])
    
    print(f"Manual Recall: {recall}")
    print(f"Sklearn Recall: {recall_score(y_true, y_pred)}")
    
    return recall

# Output:
# Manual Recall: 0.75
# Sklearn Recall: 0.75
```

Slide 3: The Mathematics Behind Precision-Recall

Understanding the mathematical foundations of precision and recall helps in grasping their relationship and trade-offs. These metrics form the basis for more complex evaluation metrics like F1-score and PR curves.

```python
# Mathematical formulas in LaTeX notation (not rendered)
"""
Precision Formula:
$$Precision = \frac{TP}{TP + FP}$$

Recall Formula:
$$Recall = \frac{TP}{TP + FN}$$

F1 Score Formula:
$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$
"""

# Implementation of formulas
def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

# Example
tp, fp, fn = 80, 20, 10
p, r, f1 = calculate_metrics(tp, fp, fn)
print(f"Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}")

# Output:
# Precision: 0.80, Recall: 0.89, F1: 0.84
```

Slide 4: Implementing a Custom Binary Classifier

This implementation demonstrates how to build a binary classifier from scratch while monitoring precision and recall metrics during training. The example uses a simple logistic regression model with gradient descent optimization.

```python
import numpy as np
from sklearn.model_selection import train_test_split

class BinaryClassifier:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, epochs=100):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(epochs):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            # Backward pass
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X, threshold=0.5):
        z = np.dot(X, self.weights) + self.bias
        return (self.sigmoid(z) >= threshold).astype(int)
```

Slide 5: Real-world Example: Email Spam Detection

Implementing a spam detection system demonstrates the practical importance of balancing precision and recall. High precision prevents legitimate emails from being marked as spam, while good recall ensures dangerous emails are caught.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

def spam_detector():
    # Sample data (in real world, this would be your email dataset)
    emails = [
        "win money now", "meeting tomorrow", "free prize claim",
        "project deadline", "lottery winner", "team update"
    ]
    labels = [1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for non-spam
    
    # Feature extraction
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(emails)
    
    # Train classifier
    classifier = BinaryClassifier(learning_rate=0.1)
    classifier.fit(X.toarray(), np.array(labels))
    
    # Predictions
    predictions = classifier.predict(X.toarray())
    
    # Performance metrics
    print(classification_report(labels, predictions))
    
    return classifier, vectorizer

# Execute example
classifier, vectorizer = spam_detector()
```

Slide 6: Precision-Recall Trade-off Analysis

Understanding how different threshold values affect the precision-recall trade-off is crucial for model optimization. This implementation demonstrates how to analyze and visualize these relationships using scikit-learn's precision\_recall\_curve function.

```python
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def analyze_pr_tradeoff(y_true, y_prob):
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Plot the precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, 'b-', label='PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Trade-off')
    plt.grid(True)
    
    # Find optimal threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, precisions[optimal_idx], recalls[optimal_idx]

# Example usage
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1])
y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.85, 0.2, 0.95])
threshold, precision, recall = analyze_pr_tradeoff(y_true, y_prob)
print(f"Optimal threshold: {threshold:.2f}")
print(f"Resulting precision: {precision:.2f}")
print(f"Resulting recall: {recall:.2f}")
```

Slide 7: Building a Precision-Recall Monitor

This implementation creates a monitoring system that tracks precision and recall metrics during model training, helping identify potential issues like overfitting or underfitting early in the training process.

```python
class PRMonitor:
    def __init__(self, validation_frequency=10):
        self.validation_frequency = validation_frequency
        self.precision_history = []
        self.recall_history = []
        
    def monitor_metrics(self, model, X_val, y_val, epoch):
        if epoch % self.validation_frequency == 0:
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            tp = np.sum((y_val == 1) & (y_pred == 1))
            fp = np.sum((y_val == 0) & (y_pred == 1))
            fn = np.sum((y_val == 1) & (y_pred == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            self.precision_history.append(precision)
            self.recall_history.append(recall)
            
            print(f"Epoch {epoch}:")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            
    def plot_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.precision_history, label='Precision')
        plt.plot(self.recall_history, label='Recall')
        plt.xlabel('Validation Steps')
        plt.ylabel('Score')
        plt.title('Precision-Recall History')
        plt.legend()
        plt.grid(True)
        plt.show()
```

Slide 8: Real-world Example: Credit Card Fraud Detection

A practical implementation of precision-recall optimization in a fraud detection system where both false positives (legitimate transactions marked as fraud) and false negatives (missed fraud cases) have significant business impacts.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def fraud_detection_system(transactions_df):
    # Preprocess data
    scaler = StandardScaler()
    X = scaler.fit_transform(transactions_df.drop('is_fraud', axis=1))
    y = transactions_df['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train classifier
    classifier = BinaryClassifier(learning_rate=0.01)
    monitor = PRMonitor(validation_frequency=5)
    
    # Training loop with monitoring
    for epoch in range(100):
        classifier.fit(X_train, y_train)
        monitor.monitor_metrics(classifier, X_test, y_test, epoch)
    
    # Final evaluation
    y_pred = classifier.predict(X_test)
    final_precision = precision_score(y_test, y_pred)
    final_recall = recall_score(y_test, y_pred)
    
    return classifier, final_precision, final_recall

# Example usage with synthetic data
np.random.seed(42)
n_samples = 1000
transactions_df = pd.DataFrame({
    'amount': np.random.normal(100, 50, n_samples),
    'time': np.random.uniform(0, 24, n_samples),
    'n_previous_transactions': np.random.poisson(5, n_samples),
    'is_fraud': np.random.binomial(1, 0.1, n_samples)
})

classifier, precision, recall = fraud_detection_system(transactions_df)
print(f"Final Precision: {precision:.3f}")
print(f"Final Recall: {recall:.3f}")
```

Slide 9: Implementing Area Under Precision-Recall Curve (AUPRC)

The Area Under the Precision-Recall Curve provides a single score that captures the model's performance across all possible thresholds, particularly useful for imbalanced classification problems where accuracy alone might be misleading.

```python
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

def calculate_auprc(y_true, y_prob):
    # Calculate precision-recall curve points
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    
    # Calculate AUPRC using auc function
    auprc_score = auc(recalls, precisions)
    
    # Visualize the curve
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, 'b-', label=f'AUPRC = {auprc_score:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    return auprc_score, (precisions, recalls)

# Example usage
np.random.seed(42)
y_true = np.random.binomial(1, 0.3, 1000)
y_prob = y_true * np.random.normal(0.7, 0.1, 1000) + (1-y_true) * np.random.normal(0.3, 0.1, 1000)
score, (p, r) = calculate_auprc(y_true, y_prob)
print(f"AUPRC Score: {score:.3f}")
```

Slide 10: Precision at K Implementation

Precision at K is particularly important in recommendation systems and search engines where we care about the precision of top-K predictions. This implementation shows how to calculate and optimize for this metric.

```python
def precision_at_k(y_true, y_prob, k):
    # Sort predictions in descending order
    sorted_indices = np.argsort(y_prob)[::-1]
    
    # Get top k predictions
    top_k_indices = sorted_indices[:k]
    
    # Calculate precision at k
    precision_k = np.sum(y_true[top_k_indices]) / k
    
    return precision_k

class PrecisionAtKOptimizer:
    def __init__(self, k=10):
        self.k = k
        self.best_precision = 0
        self.best_threshold = 0
        
    def optimize(self, y_true, y_prob, thresholds):
        results = []
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            precision = precision_at_k(y_true, y_prob, self.k)
            results.append((threshold, precision))
            
            if precision > self.best_precision:
                self.best_precision = precision
                self.best_threshold = threshold
        
        return results

# Example usage
np.random.seed(42)
y_true = np.random.binomial(1, 0.2, 100)
y_prob = y_true * np.random.normal(0.8, 0.1, 100) + (1-y_true) * np.random.normal(0.2, 0.1, 100)

optimizer = PrecisionAtKOptimizer(k=10)
thresholds = np.linspace(0, 1, 20)
results = optimizer.optimize(y_true, y_prob, thresholds)

print(f"Best threshold: {optimizer.best_threshold:.3f}")
print(f"Best Precision@10: {optimizer.best_precision:.3f}")
```

Slide 11: Multi-class Precision and Recall

Extending precision and recall metrics to multi-class scenarios requires careful consideration of averaging methods. This implementation demonstrates different averaging strategies for multi-class classification.

```python
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support

def multiclass_precision_recall(y_true, y_pred, n_classes):
    # Calculate metrics for each class
    precisions = []
    recalls = []
    
    for class_idx in range(n_classes):
        # Convert to binary problem
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate different averaging methods
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    
    # Compare with sklearn implementation
    sk_precision, sk_recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    
    return {
        'class_precisions': precisions,
        'class_recalls': recalls,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'sklearn_precision': sk_precision,
        'sklearn_recall': sk_recall
    }

# Example usage
n_classes = 3
y_true = np.random.randint(0, n_classes, 100)
y_pred = np.random.randint(0, n_classes, 100)

results = multiclass_precision_recall(y_true, y_pred, n_classes)
for metric, value in results.items():
    print(f"{metric}: {value}")
```

Slide 12: Cost-Sensitive Precision-Recall Optimization

In real-world applications, different types of errors may have varying costs. This implementation shows how to optimize precision-recall trade-offs when false positives and false negatives have different associated costs.

```python
class CostSensitiveOptimizer:
    def __init__(self, fp_cost=1.0, fn_cost=1.0):
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost
        
    def calculate_total_cost(self, y_true, y_pred):
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        total_cost = (fp * self.fp_cost) + (fn * self.fn_cost)
        return total_cost
    
    def find_optimal_threshold(self, y_true, y_prob, thresholds):
        min_cost = float('inf')
        optimal_threshold = None
        cost_history = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            cost = self.calculate_total_cost(y_true, y_pred)
            cost_history.append((threshold, cost))
            
            if cost < min_cost:
                min_cost = cost
                optimal_threshold = threshold
        
        return optimal_threshold, min_cost, cost_history

# Example usage
np.random.seed(42)
y_true = np.random.binomial(1, 0.3, 1000)
y_prob = y_true * np.random.normal(0.7, 0.1, 1000) + \
         (1-y_true) * np.random.normal(0.3, 0.1, 1000)

# Scenario where false negatives are 5 times more costly than false positives
optimizer = CostSensitiveOptimizer(fp_cost=1.0, fn_cost=5.0)
thresholds = np.linspace(0, 1, 100)
opt_threshold, min_cost, history = optimizer.find_optimal_threshold(
    y_true, y_prob, thresholds
)

print(f"Optimal threshold: {opt_threshold:.3f}")
print(f"Minimum total cost: {min_cost:.2f}")
```

Slide 13: Balanced Precision-Recall Using SMOTE

When dealing with imbalanced datasets, SMOTE (Synthetic Minority Over-sampling Technique) can help achieve better precision-recall balance. This implementation shows how to use SMOTE effectively.

```python
from collections import Counter
import numpy as np
from sklearn.neighbors import NearestNeighbors

class SMOTEBalancer:
    def __init__(self, k_neighbors=5):
        self.k = k_neighbors
        
    def generate_synthetic_samples(self, X, n_samples):
        nn = NearestNeighbors(n_neighbors=self.k+1).fit(X)
        distances, indices = nn.kneighbors(X)
        
        synthetic_samples = []
        for i in range(n_samples):
            ref_idx = np.random.randint(len(X))
            neighbor_idx = indices[ref_idx, np.random.randint(1, self.k+1)]
            
            # Generate synthetic sample
            alpha = np.random.random()
            synthetic_sample = X[ref_idx] + alpha * (X[neighbor_idx] - X[ref_idx])
            synthetic_samples.append(synthetic_sample)
            
        return np.array(synthetic_samples)
    
    def balance_dataset(self, X, y):
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        majority_count = class_counts[majority_class]
        
        balanced_X = X.copy()
        balanced_y = y.copy()
        
        for class_label in class_counts:
            if class_label == majority_class:
                continue
                
            class_indices = np.where(y == class_label)[0]
            class_samples = X[class_indices]
            n_synthetic = majority_count - len(class_indices)
            
            synthetic_samples = self.generate_synthetic_samples(
                class_samples, n_synthetic
            )
            
            balanced_X = np.vstack([balanced_X, synthetic_samples])
            balanced_y = np.hstack([
                balanced_y, 
                np.full(n_synthetic, class_label)
            ])
            
        return balanced_X, balanced_y

# Example usage
np.random.seed(42)
n_samples = 1000
X_imbalanced = np.random.randn(n_samples, 2)
y_imbalanced = np.random.binomial(1, 0.1, n_samples)

balancer = SMOTEBalancer(k_neighbors=5)
X_balanced, y_balanced = balancer.balance_dataset(X_imbalanced, y_imbalanced)

print("Original class distribution:", Counter(y_imbalanced))
print("Balanced class distribution:", Counter(y_balanced))
```

Slide 14: Additional Resources

*   Understanding Deep Learning Requires Rethinking Generalization [https://arxiv.org/abs/1611.03530](https://arxiv.org/abs/1611.03530)
*   Deep Learning with Class-Imbalanced Data: A Comprehensive Survey [https://arxiv.org/abs/2110.05154](https://arxiv.org/abs/2110.05154)
*   Precision and Recall Are Not Enough: Using Complementary Metrics for Better Evaluation [https://arxiv.org/abs/2008.03043](https://arxiv.org/abs/2008.03043)
*   The Relationship Between Precision-Recall and ROC Curves [https://www.biostat.wisc.edu/~page/rocpr.pdf](https://www.biostat.wisc.edu/~page/rocpr.pdf)
*   A Survey on Deep Learning for Class-Imbalanced Data [https://scholar.google.com/search?q=survey+deep+learning+class+imbalanced+data](https://scholar.google.com/search?q=survey+deep+learning+class+imbalanced+data)
*   Best Practices for Precision-Recall Optimization in Production Systems [https://research.google/pubs/pub48642/](https://research.google/pubs/pub48642/)

