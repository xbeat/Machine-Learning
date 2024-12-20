## Addressing Class Imbalance in Machine Learning
Slide 1: Understanding Class Imbalance

Class imbalance occurs when dataset classes have significantly different sample sizes, affecting model performance. We'll implement a basic data analysis function to detect and quantify imbalance using standard metrics like class distribution and imbalance ratio.

```python
import numpy as np
import pandas as pd

def analyze_class_imbalance(y, labels=None):
    """
    Analyzes class imbalance in a dataset
    
    Args:
        y: array-like, target labels
        labels: optional, list of class names
    """
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(labels if labels else unique, counts))
    
    # Calculate imbalance ratio (majority/minority)
    maj_class = max(counts)
    min_class = min(counts)
    imbalance_ratio = maj_class / min_class
    
    print(f"Class Distribution:\n{distribution}")
    print(f"Imbalance Ratio: {imbalance_ratio:.2f}")
    
    return distribution, imbalance_ratio

# Example usage
y = np.array([0, 0, 0, 0, 0, 1, 1, 1])
labels = ['Normal', 'Anomaly']
analyze_class_imbalance(y, labels)
```

Slide 2: Random Undersampling Implementation

Random undersampling reduces majority class samples to match minority class size. This technique helps balance datasets but may lose valuable information. Implementation focuses on preserving data structure while maintaining randomness.

```python
import numpy as np
from sklearn.utils import shuffle

def random_undersample(X, y):
    """
    Performs random undersampling on majority class
    
    Args:
        X: feature matrix
        y: target labels
    """
    unique, counts = np.unique(y, return_counts=True)
    min_class_size = min(counts)
    
    # Get indices for each class
    maj_indices = np.where(y == unique[np.argmax(counts)])[0]
    min_indices = np.where(y == unique[np.argmin(counts)])[0]
    
    # Randomly select majority samples
    maj_indices = shuffle(maj_indices)[:min_class_size]
    selected_indices = np.concatenate([maj_indices, min_indices])
    
    return X[selected_indices], y[selected_indices]

# Example usage
X = np.random.randn(1000, 5)  # 1000 samples, 5 features
y = np.array([0]*900 + [1]*100)  # Imbalanced classes
X_balanced, y_balanced = random_undersample(X, y)
print(f"Original class distribution: {np.bincount(y)}")
print(f"Balanced class distribution: {np.bincount(y_balanced)}")
```

Slide 3: SMOTE Algorithm Implementation

SMOTE creates synthetic minority class samples by interpolating between existing minority instances. This implementation demonstrates the core SMOTE algorithm with k-nearest neighbors to generate balanced datasets.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote(X_minority, n_synthetic_samples, k_neighbors=5):
    """
    Implements SMOTE algorithm for minority class oversampling
    
    Args:
        X_minority: minority class samples
        n_synthetic_samples: number of synthetic samples to generate
        k_neighbors: number of nearest neighbors to consider
    """
    n_minority_samples, n_features = X_minority.shape
    
    # Find k nearest neighbors
    neigh = NearestNeighbors(n_neighbors=k_neighbors)
    neigh.fit(X_minority)
    nearest_neighbors = neigh.kneighbors(X_minority, return_distance=False)
    
    # Generate synthetic samples
    synthetic_samples = np.zeros((n_synthetic_samples, n_features))
    for i in range(n_synthetic_samples):
        idx = np.random.randint(0, n_minority_samples)
        nn_idx = nearest_neighbors[idx, np.random.randint(1, k_neighbors)]
        
        # Interpolation
        gap = np.random.random()
        synthetic_samples[i] = X_minority[idx] + gap * (
            X_minority[nn_idx] - X_minority[idx])
    
    return synthetic_samples

# Example usage
X_min = np.random.randn(50, 3)  # Minority class samples
synthetic_X = smote(X_min, n_synthetic_samples=100)
print(f"Original samples shape: {X_min.shape}")
print(f"Generated samples shape: {synthetic_X.shape}")
```

Slide 4: Tomek Links Detection and Removal

Tomek links are pairs of samples from different classes that are closest neighbors to each other. This implementation identifies and removes these pairs to create cleaner class boundaries.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def find_tomek_links(X, y):
    """
    Identifies Tomek links in dataset
    
    Args:
        X: feature matrix
        y: target labels
    """
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X)
    
    # Find nearest neighbors for all samples
    distances, indices = nn.kneighbors(X)
    
    tomek_links = []
    for i in range(len(y)):
        if y[i] != y[indices[i, 1]] and i == indices[indices[i, 1], 1]:
            tomek_links.append(i)
    
    return np.array(tomek_links)

def remove_tomek_links(X, y):
    """Removes samples that form Tomek links"""
    tomek_indices = find_tomek_links(X, y)
    mask = ~np.isin(np.arange(len(y)), tomek_indices)
    return X[mask], y[mask]

# Example usage
X = np.random.randn(1000, 4)
y = np.array([0]*800 + [1]*200)
X_cleaned, y_cleaned = remove_tomek_links(X, y)
print(f"Original dataset size: {len(y)}")
print(f"Cleaned dataset size: {len(y_cleaned)}")
```

Slide 5: Combined SMOTE-Tomek Algorithm

This advanced implementation combines SMOTE oversampling with Tomek links removal, creating a robust approach to handle imbalanced datasets. The process first generates synthetic samples then removes borderline cases.

```python
def smote_tomek(X, y, k_neighbors=5):
    """
    Combines SMOTE and Tomek links for balanced dataset
    
    Args:
        X: feature matrix
        y: target labels
        k_neighbors: number of neighbors for SMOTE
    """
    # Get minority and majority indices
    min_class = np.argmin(np.bincount(y))
    maj_class = np.argmax(np.bincount(y))
    min_indices = np.where(y == min_class)[0]
    maj_indices = np.where(y == maj_class)[0]
    
    # Apply SMOTE
    X_min = X[min_indices]
    n_synthetic = len(maj_indices) - len(min_indices)
    synthetic_samples = smote(X_min, n_synthetic, k_neighbors)
    
    # Combine datasets
    X_combined = np.vstack([X, synthetic_samples])
    y_combined = np.hstack([y, np.full(n_synthetic, min_class)])
    
    # Remove Tomek links
    return remove_tomek_links(X_combined, y_combined)

# Example usage with performance metrics
from sklearn.metrics import classification_report
X = np.random.randn(1000, 5)
y = np.array([0]*900 + [1]*100)
X_balanced, y_balanced = smote_tomek(X, y)
print(f"Original distribution: {np.bincount(y)}")
print(f"Balanced distribution: {np.bincount(y_balanced)}")
```

Slide 6: Real-world Application: Credit Card Fraud Detection

In this implementation, we'll process a credit card fraud dataset using our balanced learning techniques. The focus is on maintaining model performance while dealing with highly imbalanced fraud cases.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def process_fraud_detection(X, y, random_state=42):
    """
    Processes credit card fraud detection dataset
    
    Args:
        X: feature matrix
        y: fraud labels (0: normal, 1: fraud)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Balance training data
    X_train_balanced, y_train_balanced = smote_tomek(
        X_train_scaled, y_train)
    
    # Train model
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train_balanced, y_train_balanced)
    
    return clf, X_test_scaled, y_test

# Example usage
X = np.random.randn(10000, 10)  # Simulated transaction features
y = np.array([0]*9900 + [1]*100)  # 1% fraud rate
clf, X_test, y_test = process_fraud_detection(X, y)
print("Model Performance:")
print(classification_report(y_test, clf.predict(X_test)))
```

Slide 7: Real-world Application: Disease Diagnosis

This implementation focuses on handling imbalanced medical diagnosis data where positive cases (diseases) are typically rare compared to negative cases (healthy patients).

```python
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder

def process_medical_diagnosis(X, y, disease_labels):
    """
    Processes imbalanced medical diagnosis data
    
    Args:
        X: patient features
        y: diagnosis labels
        disease_labels: disease names
    """
    # Encode categorical variables
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply balanced learning
    X_balanced, y_balanced = random_undersample(
        X_train_scaled, y_train)
    
    # Train model with class weights
    clf = RandomForestClassifier(class_weight='balanced')
    clf.fit(X_balanced, y_balanced)
    
    # Evaluate
    y_pred_proba = clf.predict_proba(X_test_scaled)
    auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    print(f"AUC-ROC Score: {auc_score:.3f}")
    return clf, auc_score

# Example usage
X = np.random.randn(5000, 15)  # Patient features
y = np.array(['healthy']*4800 + ['disease']*200)
disease_labels = ['healthy', 'disease']
clf, auc_score = process_medical_diagnosis(X, y, disease_labels)
```

Slide 8: Implementation of Adaptive Synthetic Sampling (ADASYN)

ADASYN improves upon SMOTE by generating synthetic samples adaptively based on data density. This implementation focuses on creating more synthetic samples for minority class instances that are harder to learn.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def adasyn(X, y, beta=1.0, k_neighbors=5):
    """
    Implements ADASYN algorithm for adaptive synthetic sampling
    
    Args:
        X: feature matrix
        y: target labels
        beta: desired balance level (0,1]
        k_neighbors: number of nearest neighbors
    """
    minority_class = np.argmin(np.bincount(y))
    X_min = X[y == minority_class]
    X_maj = X[y != minority_class]
    
    # Calculate number of synthetic samples needed
    G = len(X_maj) - len(X_min)
    synthetic_samples = []
    
    # Find k-nearest neighbors for minority samples
    neigh = NearestNeighbors(n_neighbors=k_neighbors+1)
    neigh.fit(X)
    
    # Calculate ratio of majority samples in neighborhood
    ratio = []
    for x_min in X_min:
        indices = neigh.kneighbors([x_min], return_distance=False)[0][1:]
        ratio.append(np.sum(y[indices] != minority_class) / k_neighbors)
    
    # Normalize ratios
    if sum(ratio) > 0:
        ratio = np.array(ratio) / sum(ratio)
        n_synthetic = np.round(ratio * G * beta).astype(int)
        
        # Generate synthetic samples
        for i, x_min in enumerate(X_min):
            if n_synthetic[i] > 0:
                nn_indices = neigh.kneighbors([x_min], return_distance=False)[0][1:]
                for _ in range(n_synthetic[i]):
                    nn_idx = np.random.choice(nn_indices)
                    gap = np.random.random()
                    synthetic_sample = x_min + gap * (X[nn_idx] - x_min)
                    synthetic_samples.append(synthetic_sample)
    
    if synthetic_samples:
        X_synthetic = np.vstack(synthetic_samples)
        y_synthetic = np.full(len(synthetic_samples), minority_class)
        return (np.vstack([X, X_synthetic]), 
                np.hstack([y, y_synthetic]))
    return X, y

# Example usage
X = np.random.randn(1000, 4)
y = np.array([0]*900 + [1]*100)
X_balanced, y_balanced = adasyn(X, y)
print(f"Original distribution: {np.bincount(y)}")
print(f"Balanced distribution: {np.bincount(y_balanced)}")
```

Slide 9: Ensemble Learning with Balanced Bagging

This implementation combines bagging with balanced sampling techniques to create a robust ensemble classifier for imbalanced datasets.

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class BalancedBaggingClassifier(BaseEstimator, ClassifierMixin):
    """
    Implements balanced bagging ensemble for imbalanced datasets
    """
    def __init__(self, n_estimators=10, max_samples=1.0):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.estimators_ = []
        
    def fit(self, X, y):
        self.estimators_ = []
        n_samples = int(X.shape[0] * self.max_samples)
        
        for _ in range(self.n_estimators):
            # Create balanced bootstrap sample
            X_balanced, y_balanced = random_undersample(X, y)
            
            # Train base estimator
            estimator = DecisionTreeClassifier()
            estimator.fit(X_balanced, y_balanced)
            self.estimators_.append(estimator)
        
        return self
    
    def predict_proba(self, X):
        # Average probabilities from all estimators
        probas = np.zeros((X.shape[0], 2))
        for estimator in self.estimators_:
            probas += estimator.predict_proba(X)
        return probas / self.n_estimators
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

# Example usage
X = np.random.randn(1000, 5)
y = np.array([0]*900 + [1]*100)
clf = BalancedBaggingClassifier(n_estimators=10)
clf.fit(X, y)
print("Ensemble predictions shape:", clf.predict_proba(X).shape)
```

Slide 10: Cost-Sensitive Learning Implementation

This implementation focuses on incorporating class-specific costs into the learning process, making the model more sensitive to minority class errors.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class CostSensitiveClassifier:
    """
    Implements cost-sensitive learning for imbalanced datasets
    """
    def __init__(self, base_estimator=None, cost_matrix=None):
        self.base_estimator = base_estimator or DecisionTreeClassifier()
        self.cost_matrix = cost_matrix or np.array([[0, 1], [5, 0]])
        
    def fit(self, X, y):
        # Calculate sample weights based on costs
        weights = np.zeros(len(y))
        for i, yi in enumerate(y):
            weights[i] = sum(self.cost_matrix[yi])
            
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
        # Train base estimator with sample weights
        self.base_estimator.fit(X, y, sample_weight=weights)
        return self
    
    def predict(self, X):
        # Get probability predictions
        probas = self.base_estimator.predict_proba(X)
        
        # Calculate expected costs
        expected_costs = np.dot(probas, self.cost_matrix)
        
        # Return class with minimum expected cost
        return np.argmin(expected_costs, axis=1)

# Example usage
cost_matrix = np.array([[0, 1], [5, 0]])  # Higher cost for FN than FP
clf = CostSensitiveClassifier(cost_matrix=cost_matrix)
X = np.random.randn(1000, 4)
y = np.array([0]*900 + [1]*100)
clf.fit(X, y)
predictions = clf.predict(X)
print("Class distribution in predictions:", np.bincount(predictions))
```

Slide 11: Performance Evaluation for Imbalanced Learning

This implementation focuses on metrics specifically designed for imbalanced datasets, including precision-recall AUC, F-beta score, and balanced accuracy, providing a comprehensive evaluation framework.

```python
from sklearn.metrics import precision_recall_curve, roc_curve
import numpy as np

def evaluate_imbalanced_model(y_true, y_pred, y_prob):
    """
    Comprehensive evaluation metrics for imbalanced learning
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        y_prob: prediction probabilities for positive class
    """
    # Calculate confusion matrix elements
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f2_score = 5 * precision * recall / (4 * precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate PR and ROC curves
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = np.trapz(precision_curve, recall_curve)
    
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F2-Score': f2_score,
        'PR-AUC': pr_auc,
        'G-Mean': np.sqrt(recall * specificity)
    }
    
    return metrics

# Example usage
y_true = np.array([0]*900 + [1]*100)
y_pred = np.random.binomial(1, 0.1, 1000)
y_prob = np.random.random(1000)
metrics = evaluate_imbalanced_model(y_true, y_pred, y_prob)
for metric, value in metrics.items():
    print(f"{metric}: {value:.3f}")
```

Slide 12: Real-world Application: Anomaly Detection System

Implementation of a complete anomaly detection system using multiple balancing techniques and ensemble methods for robust performance in production environments.

```python
class AnomalyDetectionSystem:
    """
    Production-ready anomaly detection system with imbalanced learning
    """
    def __init__(self, balancing_method='smote-tomek'):
        self.balancing_method = balancing_method
        self.scaler = StandardScaler()
        self.ensemble = []
        
    def preprocess_data(self, X, y=None):
        """Preprocesses data with scaling and optional balancing"""
        if y is not None:
            X_scaled = self.scaler.fit_transform(X)
            if self.balancing_method == 'smote-tomek':
                return smote_tomek(X_scaled, y)
            elif self.balancing_method == 'adasyn':
                return adasyn(X_scaled, y)
            else:
                return random_undersample(X_scaled, y)
        return self.scaler.transform(X)
    
    def build_ensemble(self, X, y):
        """Builds an ensemble of diverse models"""
        X_balanced, y_balanced = self.preprocess_data(X, y)
        
        models = [
            RandomForestClassifier(class_weight='balanced'),
            CostSensitiveClassifier(),
            BalancedBaggingClassifier()
        ]
        
        for model in models:
            model.fit(X_balanced, y_balanced)
            self.ensemble.append(model)
    
    def predict(self, X):
        """Combines predictions from ensemble"""
        X_scaled = self.preprocess_data(X)
        predictions = np.array([
            model.predict_proba(X_scaled)[:, 1] 
            for model in self.ensemble
        ])
        return np.mean(predictions, axis=0) > 0.5

# Example usage with performance monitoring
from sklearn.model_selection import cross_val_score
X = np.random.randn(10000, 10)
y = np.array([0]*9900 + [1]*100)

system = AnomalyDetectionSystem()
system.build_ensemble(X, y)
scores = cross_val_score(system, X, y, cv=5, scoring='f1')
print(f"Cross-validation F1 scores: {scores.mean():.3f} ± {scores.std():.3f}")
```

Slide 13: Additional Resources

*   "SMOTE: Synthetic Minority Over-sampling Technique"
    *   [https://arxiv.org/abs/1106.1813](https://arxiv.org/abs/1106.1813)
*   "Learning from Imbalanced Data: A Comprehensive Review"
    *   [https://arxiv.org/abs/1505.01658](https://arxiv.org/abs/1505.01658)
*   "A Survey of Predictive Modelling under Imbalanced Distributions"
    *   [https://arxiv.org/abs/1505.01658](https://arxiv.org/abs/1505.01658)
*   "Cost-Sensitive Learning Methods for Imbalanced Data"
    *   Search on Google Scholar: "Cost-Sensitive Learning Imbalanced Data Survey"
*   "Ensemble Methods for Class Imbalance Learning"
    *   [https://www.semanticscholar.org/topic/Ensemble-Methods-for-Class-Imbalance-Learning/](https://www.semanticscholar.org/topic/Ensemble-Methods-for-Class-Imbalance-Learning/)

Note: Some URLs might require institutional access or can be found through academic databases.

