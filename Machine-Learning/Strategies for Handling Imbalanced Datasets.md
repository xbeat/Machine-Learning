## Strategies for Handling Imbalanced Datasets
Slide 1: Understanding Class Imbalance

The fundamental challenge in imbalanced datasets occurs when one class significantly outnumbers others, leading to biased model predictions. This slide demonstrates how to identify and quantify imbalance ratios in classification problems using Python's scientific computing tools.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Generate synthetic imbalanced dataset
X, y = make_classification(
    n_samples=10000, n_features=4,
    n_redundant=0, n_clusters_per_class=1,
    weights=[0.95, 0.05], random_state=42
)

# Calculate imbalance ratio
unique, counts = np.unique(y, return_counts=True)
imbalance_ratio = counts[0] / counts[1]

# Display class distribution
df = pd.DataFrame({'Class': unique, 'Count': counts})
print(f"Class Distribution:\n{df}")
print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")

# Output:
# Class Distribution:
#    Class  Count
# 0      0   9500
# 1      1    500
# 
# Imbalance Ratio: 19.00:1
```

Slide 2: SMOTE Implementation from Scratch

Understanding the core mechanics of Synthetic Minority Over-sampling Technique (SMOTE) through implementation helps grasp how synthetic samples are generated. This code demonstrates the fundamental SMOTE algorithm without external libraries.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote_from_scratch(X_minority, n_synthetic_samples, k_neighbors=5):
    # Fit nearest neighbors on minority class samples
    neigh = NearestNeighbors(n_neighbors=k_neighbors + 1)
    neigh.fit(X_minority)
    
    # Find k-nearest neighbors for each minority sample
    distances, indices = neigh.kneighbors(X_minority)
    
    # Generate synthetic samples
    synthetic_samples = []
    np.random.seed(42)
    
    for i in range(len(X_minority)):
        # Randomly select k neighbors
        nn_indices = indices[i, 1:]  # Exclude the sample itself
        for _ in range(n_synthetic_samples // len(X_minority)):
            # Select random neighbor
            nn_idx = np.random.choice(nn_indices)
            # Generate synthetic sample
            diff = X_minority[nn_idx] - X_minority[i]
            gap = np.random.random()
            synthetic_sample = X_minority[i] + gap * diff
            synthetic_samples.append(synthetic_sample)
    
    return np.array(synthetic_samples)

# Example usage
minority_samples = X[y == 1]
synthetic_minority = smote_from_scratch(minority_samples, n_synthetic_samples=9000)
print(f"Generated {len(synthetic_minority)} synthetic samples")
```

Slide 3: Advanced SMOTENC for Mixed Data Types

SMOTENC extends traditional SMOTE to handle datasets with both numerical and categorical features, a common scenario in real-world applications. This implementation demonstrates handling mixed data types effectively.

```python
def smote_nc(X, categorical_features, n_synthetic_samples, k_neighbors=5):
    # Separate numerical and categorical features
    numerical_features = [i for i in range(X.shape[1]) 
                         if i not in categorical_features]
    
    # Initialize nearest neighbors for numerical features
    neigh = NearestNeighbors(n_neighbors=k_neighbors + 1)
    neigh.fit(X[:, numerical_features])
    
    synthetic_samples = np.zeros((n_synthetic_samples, X.shape[1]))
    distances, indices = neigh.kneighbors(X)
    
    for i in range(n_synthetic_samples):
        # Select random sample and its neighbor
        ref_idx = np.random.randint(len(X))
        nn_idx = np.random.choice(indices[ref_idx, 1:])
        
        # Generate numerical features
        for feat in numerical_features:
            diff = X[nn_idx, feat] - X[ref_idx, feat]
            gap = np.random.random()
            synthetic_samples[i, feat] = X[ref_idx, feat] + gap * diff
        
        # Handle categorical features
        for feat in categorical_features:
            # Random selection between reference and neighbor
            synthetic_samples[i, feat] = np.random.choice(
                [X[ref_idx, feat], X[nn_idx, feat]]
            )
    
    return synthetic_samples

# Example usage with mixed data
X_mixed = np.column_stack([X, np.random.choice([0, 1, 2], size=len(X))])
categorical_features = [4]  # Last feature is categorical
synthetic_mixed = smote_nc(
    X_mixed[y == 1], 
    categorical_features=categorical_features,
    n_synthetic_samples=100
)
print(f"Generated samples shape: {synthetic_mixed.shape}")
```

Slide 4: Cost-Sensitive Learning Implementation

Cost-sensitive learning addresses imbalanced data by assigning different misclassification costs to different classes. This implementation shows how to modify a basic neural network to incorporate class-specific costs.

```python
import tensorflow as tf

class CostSensitiveNet(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim=64):
        super(CostSensitiveNet, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
    
    def cost_sensitive_loss(self, y_true, y_pred, cost_matrix):
        # cost_matrix[i,j]: cost of predicting j when true class is i
        losses = tf.where(
            tf.equal(y_true, 1),
            cost_matrix[1,0] * (1 - y_pred),  # False negative cost
            cost_matrix[0,1] * y_pred         # False positive cost
        )
        return tf.reduce_mean(losses)

# Example usage
cost_matrix = np.array([[1, 4],    # Cost of FP = 4
                       [10, 1]])    # Cost of FN = 10

model = CostSensitiveNet(input_dim=4)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = model.cost_sensitive_loss(y, predictions, cost_matrix)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

Slide 5: Adaptive Synthetic Sampling (ADASYN)

ADASYN improves upon SMOTE by generating synthetic samples adaptively based on the density distribution of minority class instances. This implementation focuses on generating more synthetic data for minority class examples that are harder to learn.

```python
def adasyn_sampling(X, y, beta=1.0, k_neighbors=5):
    minority_class = X[y == 1]
    majority_class = X[y == 0]
    
    # Calculate number of synthetic samples needed
    G = len(majority_class) - len(minority_class)
    synthetic_samples = []
    
    # Calculate density distribution
    neigh = NearestNeighbors(n_neighbors=k_neighbors + 1)
    neigh.fit(X)
    distances, indices = neigh.kneighbors(minority_class)
    
    # Calculate ratio of majority class neighbors
    r = np.array([
        sum(y[indices[i][1:]] == 0) / k_neighbors 
        for i in range(len(minority_class))
    ])
    
    # Normalize r
    if np.sum(r) != 0:
        r = r / np.sum(r)
    
    # Calculate number of synthetic samples for each minority instance
    n_synthetic = np.round(r * G * beta).astype(int)
    
    # Generate synthetic samples
    for i, n in enumerate(n_synthetic):
        if n == 0:
            continue
        # Get k-nearest neighbors
        nn_indices = indices[i, 1:]
        for _ in range(n):
            # Select random neighbor
            nn_idx = np.random.choice(nn_indices)
            # Generate synthetic sample
            gap = np.random.random()
            synthetic = minority_class[i] + gap * (
                minority_class[nn_idx] - minority_class[i]
            )
            synthetic_samples.append(synthetic)
    
    return np.array(synthetic_samples)

# Example usage
synthetic_samples = adasyn_sampling(X, y)
print(f"Generated {len(synthetic_samples)} synthetic samples")
```

Slide 6: Threshold Moving Strategy

Threshold moving is a post-processing technique that adjusts the decision boundary of a trained classifier to better handle class imbalance. This implementation demonstrates how to optimize the threshold using various metrics.

```python
from sklearn.metrics import precision_recall_curve, f1_score
import matplotlib.pyplot as plt

def optimize_threshold(y_true, y_prob, metric='f1'):
    # Get precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Calculate F1 score for each threshold
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plot threshold optimization curve
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.axvline(optimal_threshold, color='r', linestyle='--', 
                label=f'Optimal threshold: {optimal_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('Threshold Optimization')
    plt.legend()
    plt.grid(True)
    
    return optimal_threshold

# Example usage
from sklearn.linear_model import LogisticRegression

# Train a basic model
model = LogisticRegression()
model.fit(X, y)
y_prob = model.predict_proba(X)[:, 1]

# Find optimal threshold
optimal_threshold = optimize_threshold(y, y_prob)
print(f"Optimal threshold: {optimal_threshold:.3f}")
```

Slide 7: Ensemble-Based Balancing

This implementation combines multiple resampling techniques with ensemble learning to create a robust solution for imbalanced datasets. The approach uses bagging with different sampling strategies for each base estimator.

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

class BalancedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, base_estimator=None):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator or RandomForestClassifier()
        self.estimators_ = []
        
    def fit(self, X, y):
        minority_class = X[y == 1]
        majority_class = X[y == 0]
        
        for i in range(self.n_estimators):
            # Different sampling strategy for each estimator
            if i % 3 == 0:
                # Random undersampling
                maj_sample = resample(majority_class, 
                                   n_samples=len(minority_class),
                                   random_state=i)
                X_balanced = np.vstack([minority_class, maj_sample])
                y_balanced = np.hstack([
                    np.ones(len(minority_class)),
                    np.zeros(len(maj_sample))
                ])
            elif i % 3 == 1:
                # SMOTE
                synthetic = smote_from_scratch(
                    minority_class, 
                    n_synthetic_samples=len(majority_class) - len(minority_class)
                )
                X_balanced = np.vstack([X, synthetic])
                y_balanced = np.hstack([
                    y, 
                    np.ones(len(synthetic))
                ])
            else:
                # ADASYN
                synthetic = adasyn_sampling(X, y)
                X_balanced = np.vstack([X, synthetic])
                y_balanced = np.hstack([
                    y, 
                    np.ones(len(synthetic))
                ])
            
            # Train base estimator
            estimator = clone(self.base_estimator)
            estimator.fit(X_balanced, y_balanced)
            self.estimators_.append(estimator)
        
        return self
    
    def predict_proba(self, X):
        # Average predictions from all estimators
        probas = np.array([
            estimator.predict_proba(X) 
            for estimator in self.estimators_
        ])
        return np.mean(probas, axis=0)
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)

# Example usage
ensemble = BalancedEnsemble(n_estimators=10)
ensemble.fit(X, y)
predictions = ensemble.predict(X)
```

Slide 8: Deep Learning with Class Weights

Implementing class weights in deep learning frameworks requires careful consideration of loss function modifications and batch sampling. This implementation shows how to properly weight classes in a deep learning context.

```python
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

class WeightedDeepNetwork:
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dims[0], activation='relu', 
                                input_shape=(input_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(hidden_dims[1], activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        self.class_weights = dict(enumerate(class_weights))
        
    def weighted_binary_crossentropy(self, y_true, y_pred):
        weights = tf.where(tf.equal(y_true, 1), 
                          self.class_weights[1],
                          self.class_weights[0])
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weighted_bce = tf.multiply(weights, bce)
        return tf.reduce_mean(weighted_bce)
    
    def compile_and_fit(self, X, y, validation_split=0.2, epochs=100):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=self.weighted_binary_crossentropy,
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_auc',
                    patience=10,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )
        return history

# Example usage
model = WeightedDeepNetwork(input_dim=X.shape[1])
history = model.compile_and_fit(X, y)
```

Slide 9: Focal Loss Implementation

Focal Loss addresses class imbalance by down-weighting well-classified examples and focusing on hard, misclassified examples. This implementation provides a custom Focal Loss layer for deep learning models.

```python
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma  # focusing parameter
        self.alpha = alpha  # balancing parameter
        
    def call(self, y_true, y_pred):
        # Clip predictions to prevent numerical instability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal term
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_term = tf.pow(1 - p_t, self.gamma)
        
        # Calculate alpha term
        alpha_t = tf.where(tf.equal(y_true, 1), 
                          self.alpha, 
                          1 - self.alpha)
        
        # Combine all terms
        focal_loss = alpha_t * focal_term * cross_entropy
        
        return tf.reduce_mean(focal_loss)

def create_focal_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', 
                            input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

# Example usage
focal_model = create_focal_model(X.shape[1])
history = focal_model.fit(
    X, y,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True
        )
    ],
    verbose=1
)
```

Slide 10: Two-Phase Learning Strategy

Two-phase learning combines undersampling and ensemble learning to create a robust classifier for imbalanced data. The first phase trains on balanced subsets, while the second phase fine-tunes on the full dataset.

```python
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class TwoPhaseClassifier:
    def __init__(self, base_estimator=None, n_subsets=5):
        self.base_estimator = base_estimator or RandomForestClassifier()
        self.n_subsets = n_subsets
        self.phase1_models = []
        self.phase2_model = None
        
    def fit(self, X, y):
        minority_mask = y == 1
        X_min = X[minority_mask]
        y_min = y[minority_mask]
        X_maj = X[~minority_mask]
        y_maj = y[~minority_mask]
        
        # Phase 1: Train on balanced subsets
        for i in range(self.n_subsets):
            # Random undersample majority class
            maj_indices = np.random.choice(
                len(X_maj), 
                size=len(X_min), 
                replace=False
            )
            X_maj_subset = X_maj[maj_indices]
            y_maj_subset = y_maj[maj_indices]
            
            # Combine with minority class
            X_balanced = np.vstack([X_min, X_maj_subset])
            y_balanced = np.hstack([y_min, y_maj_subset])
            
            # Train model on balanced subset
            model = clone(self.base_estimator)
            model.fit(X_balanced, y_balanced)
            self.phase1_models.append(model)
        
        # Phase 2: Fine-tune on full dataset
        # Use predictions from phase 1 as features
        phase1_predictions = np.column_stack([
            model.predict_proba(X)[:, 1] 
            for model in self.phase1_models
        ])
        
        # Combine original features with phase 1 predictions
        X_enhanced = np.hstack([X, phase1_predictions])
        
        # Train final model
        self.phase2_model = clone(self.base_estimator)
        self.phase2_model.fit(X_enhanced, y)
        
        return self
    
    def predict_proba(self, X):
        # Get phase 1 predictions
        phase1_predictions = np.column_stack([
            model.predict_proba(X)[:, 1] 
            for model in self.phase1_models
        ])
        
        # Combine with original features
        X_enhanced = np.hstack([X, phase1_predictions])
        
        # Get final predictions
        return self.phase2_model.predict_proba(X_enhanced)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# Example usage
tpc = TwoPhaseClassifier(n_subsets=5)
tpc.fit(X, y)
predictions = tpc.predict(X)
probabilities = tpc.predict_proba(X)
```

Slide 11: Performance Evaluation Suite

A comprehensive evaluation suite for imbalanced classification problems that goes beyond accuracy to include metrics specifically designed for imbalanced datasets.

```python
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                           average_precision_score, roc_curve, auc,
                           f1_score, matthews_corrcoef)

class ImbalancedEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def evaluate(self, y_true, y_pred, y_prob=None):
        # Basic metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Sensitivity (Recall) and Specificity
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        # Precision and F1
        precision = tp / (tp + fp)
        f1 = f1_score(y_true, y_pred)
        
        # G-mean
        g_mean = np.sqrt(sensitivity * specificity)
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Store basic metrics
        self.metrics = {
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision,
            'F1-Score': f1,
            'G-Mean': g_mean,
            'MCC': mcc
        }
        
        # Additional metrics if probabilities are provided
        if y_prob is not None:
            # Precision-Recall curve
            precision_curve, recall_curve, _ = precision_recall_curve(
                y_true, y_prob
            )
            pr_auc = auc(recall_curve, precision_curve)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Average Precision
            ap = average_precision_score(y_true, y_prob)
            
            # Add to metrics
            self.metrics.update({
                'PR-AUC': pr_auc,
                'ROC-AUC': roc_auc,
                'Average Precision': ap
            })
        
        return self.metrics
    
    def plot_curves(self, y_true, y_prob):
        # Create precision-recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_true, y_prob
        )
        
        plt.figure(figsize=(12, 5))
        
        # Precision-Recall curve
        plt.subplot(1, 2, 1)
        plt.plot(recall_curve, precision_curve)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        
        # ROC curve
        plt.subplot(1, 2, 2)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        
        plt.tight_layout()
        plt.show()

# Example usage
evaluator = ImbalancedEvaluator()
metrics = evaluator.evaluate(y, predictions, probabilities[:, 1])
evaluator.plot_curves(y, probabilities[:, 1])
```

Slide 12: Online Learning for Imbalanced Streams

This implementation handles imbalanced data streams using an adaptive online learning approach that maintains class balance while processing data in real-time.

```python
class OnlineImbalancedLearner:
    def __init__(self, window_size=1000, imbalance_threshold=0.1):
        self.window_size = window_size
        self.imbalance_threshold = imbalance_threshold
        self.minority_buffer = []
        self.majority_buffer = []
        self.model = None
        self.initialize_model()
    
    def initialize_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def process_sample(self, x, y):
        # Store sample in appropriate buffer
        if y == 1:
            self.minority_buffer.append(x)
        else:
            self.majority_buffer.append(x)
        
        # Check if buffers exceed window size
        if len(self.majority_buffer) > self.window_size:
            self.majority_buffer.pop(0)
        if len(self.minority_buffer) > self.window_size:
            self.minority_buffer.pop(0)
        
        # Calculate current imbalance ratio
        total_samples = len(self.minority_buffer) + len(self.majority_buffer)
        minority_ratio = len(self.minority_buffer) / total_samples
        
        # Retrain if necessary
        if minority_ratio < self.imbalance_threshold and total_samples >= 100:
            self.rebalance_and_train()
    
    def rebalance_and_train(self):
        # Generate synthetic samples for minority class
        if len(self.minority_buffer) >= 2:
            synthetic_samples = smote_from_scratch(
                np.array(self.minority_buffer),
                n_synthetic_samples=len(self.majority_buffer) - len(self.minority_buffer)
            )
            
            # Combine real and synthetic samples
            X_balanced = np.vstack([
                self.minority_buffer,
                synthetic_samples,
                self.majority_buffer
            ])
            y_balanced = np.hstack([
                np.ones(len(self.minority_buffer)),
                np.ones(len(synthetic_samples)),
                np.zeros(len(self.majority_buffer))
            ])
            
            # Retrain model
            self.initialize_model()
            self.model.fit(
                X_balanced, y_balanced,
                epochs=5,
                batch_size=32,
                verbose=0
            )
    
    def predict(self, x):
        if self.model is None:
            return 0
        return (self.model.predict(x.reshape(1, -1)) > 0.5).astype(int)[0]

# Example usage with streaming data simulation
online_learner = OnlineImbalancedLearner()

# Simulate data stream
np.random.seed(42)
for _ in range(2000):
    # Generate random sample
    if np.random.random() > 0.1:  # 90% majority class
        x = np.random.normal(0, 1, 4)
        y = 0
    else:  # 10% minority class
        x = np.random.normal(2, 1, 4)
        y = 1
    
    # Process sample
    online_learner.process_sample(x, y)
    
    # Make prediction
    pred = online_learner.predict(x)
```

Slide 13: Real-time Evaluation Dashboard

This implementation creates a monitoring system for tracking the performance of imbalanced classification models in real-time.

```python
class RealTimeMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.predictions = []
        self.true_labels = []
        self.metrics_history = {
            'precision': [],
            'recall': [],
            'f1': [],
            'g_mean': []
        }
    
    def update(self, y_true, y_pred):
        # Add new predictions to window
        self.predictions.append(y_pred)
        self.true_labels.append(y_true)
        
        # Maintain window size
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.true_labels.pop(0)
        
        # Calculate metrics
        if len(self.predictions) >= 20:  # Minimum samples for reliable metrics
            self._calculate_metrics()
    
    def _calculate_metrics(self):
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predictions)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        g_mean = np.sqrt(recall * specificity)
        
        # Store metrics
        self.metrics_history['precision'].append(precision)
        self.metrics_history['recall'].append(recall)
        self.metrics_history['f1'].append(f1)
        self.metrics_history['g_mean'].append(g_mean)
    
    def plot_metrics(self):
        plt.figure(figsize=(15, 5))
        
        for i, (metric_name, values) in enumerate(self.metrics_history.items()):
            plt.subplot(1, 4, i+1)
            plt.plot(values)
            plt.title(f'{metric_name.capitalize()} Over Time')
            plt.xlabel('Updates')
            plt.ylabel('Value')
        
        plt.tight_layout()
        plt.show()
    
    def get_current_metrics(self):
        return {
            metric: values[-1] if values else 0 
            for metric, values in self.metrics_history.items()
        }

# Example usage
monitor = RealTimeMonitor()

# Simulate real-time predictions
np.random.seed(42)
for _ in range(500):
    y_true = np.random.randint(0, 2)
    y_pred = np.random.randint(0, 2)
    monitor.update(y_true, y_pred)

# Display results
monitor.plot_metrics()
print("Current metrics:", monitor.get_current_metrics())
```

Slide 14: Additional Resources

*   ArXiv Papers:
*   "Learning from Imbalanced Data: A Comprehensive Review" [https://arxiv.org/abs/2109.12375](https://arxiv.org/abs/2109.12375)
*   "A Systematic Study of Deep Learning Models for Imbalanced Classification" [https://arxiv.org/abs/2111.14744](https://arxiv.org/abs/2111.14744)
*   "Online Learning with Concept Drift and Imbalanced Data" [https://arxiv.org/abs/2012.05925](https://arxiv.org/abs/2012.05925)
*   Recommended Search Terms for Further Research:
*   "Deep learning imbalanced classification"
*   "Online learning class imbalance"
*   "Adaptive sampling methods machine learning"
*   "Cost-sensitive learning algorithms"
*   Suggested Tools and Libraries:
*   imbalanced-learn (Python package)
*   TensorFlow Addons (for advanced loss functions)
*   Scikit-learn (for basic implementation)
*   SMOTE-variants library

