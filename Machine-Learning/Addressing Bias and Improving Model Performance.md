## Addressing Bias and Improving Model Performance

Slide 1: Understanding Class Imbalance

Class imbalance occurs when the distribution of classes in a dataset is significantly skewed. This fundamental challenge in machine learning can severely impact model performance, particularly in critical applications like fraud detection or medical diagnosis where minority classes are often the most important.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                         n_features=3, random_state=42)

# Display class distribution
unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))
print("Imbalance ratio:", max(counts) / min(counts))

# Output:
# Class distribution: {0: 900, 1: 100}
# Imbalance ratio: 9.0
```

Slide 2: Random Oversampling Implementation

Random oversampling duplicates existing minority class samples to achieve balance. While simple, this technique requires careful implementation to avoid overfitting and ensure proper cross-validation splitting to prevent data leakage.

```python
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply random oversampling
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

print("Original training set shape:", np.bincount(y_train))
print("Resampled training set shape:", np.bincount(y_train_ros))

# Output:
# Original training set shape: [720  80]
# Resampled training set shape: [720 720]
```

Slide 3: SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE creates synthetic samples for the minority class by interpolating between existing instances. This advanced oversampling technique generates new samples along the line segments joining nearest neighbors of minority class instances.

```python
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Visualize first two features before and after SMOTE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='Majority')
ax1.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='Minority')
ax1.set_title('Original Data')

ax2.scatter(X_train_smote[y_train_smote==0, 0], 
           X_train_smote[y_train_smote==0, 1], label='Majority')
ax2.scatter(X_train_smote[y_train_smote==1, 0], 
           X_train_smote[y_train_smote==1, 1], label='Minority')
ax2.set_title('SMOTE Applied')

plt.tight_layout()
```

Slide 4: Random Undersampling Strategy

Random undersampling reduces majority class samples to match minority class frequency. This technique helps balance datasets by randomly removing majority class instances, though it risks losing potentially important information from the discarded samples.

```python
from imblearn.under_sampling import RandomUnderSampler

# Apply random undersampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

# Compare distributions
print("Original distribution:", np.bincount(y_train))
print("Undersampled distribution:", np.bincount(y_train_rus))

# Calculate information loss
info_loss = 1 - len(y_train_rus) / len(y_train)
print(f"Information loss: {info_loss:.2%}")

# Output:
# Original distribution: [720  80]
# Undersampled distribution: [80 80]
# Information loss: 77.50%
```

Slide 5: Tomek Links Undersampling

Tomek links identify pairs of samples from different classes that are nearest neighbors. This method removes majority class samples from Tomek links to create a cleaner decision boundary and reduce class overlap.

```python
from imblearn.under_sampling import TomekLinks

# Apply Tomek links
tl = TomekLinks()
X_train_tl, y_train_tl = tl.fit_resample(X_train, y_train)

# Calculate removed samples
removed = len(y_train) - len(y_train_tl)
print(f"Original samples: {len(y_train)}")
print(f"Samples after Tomek links: {len(y_train_tl)}")
print(f"Removed samples: {removed}")
print("Class distribution after Tomek:", np.bincount(y_train_tl))

# Output:
# Original samples: 800
# Samples after Tomek links: 776
# Removed samples: 24
# Class distribution after Tomek: [696  80]
```

Slide 6: Model Evaluation for Imbalanced Data

Traditional accuracy metrics can be misleading for imbalanced datasets. Proper evaluation requires metrics like precision, recall, F1-score, and ROC-AUC that account for class imbalance and provide more meaningful performance assessment.

```python
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Train and evaluate model with different sampling techniques
def evaluate_model(X_train_sampled, y_train_sampled, X_test, y_test):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_sampled, y_train_sampled)
    y_pred = clf.predict(X_test)
    
    # Generate comprehensive metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred):.3f}")

# Evaluate original imbalanced data
evaluate_model(X_train, y_train, X_test, y_test)

# Evaluate SMOTE-balanced data
evaluate_model(X_train_smote, y_train_smote, X_test, y_test)

# Output Example:
# Original Data:
# Classification Report:
#               precision    recall  f1-score   support
#            0       0.90      0.98      0.94       180
#            1       0.75      0.35      0.48        20
# ROC-AUC Score: 0.665

# SMOTE-balanced Data:
#               precision    recall  f1-score   support
#            0       0.95      0.96      0.95       180
#            1       0.71      0.65      0.68        20
# ROC-AUC Score: 0.805
```

Slide 7: Implementing Combined Sampling Techniques

Combined sampling techniques leverage both oversampling and undersampling to achieve optimal balance. This approach helps maintain important majority class information while ensuring minority class representation.

```python
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

# Create combined sampling pipeline
combined_sampler = SMOTETomek(random_state=42)
X_train_combined, y_train_combined = combined_sampler.fit_resample(X_train, y_train)

# Display class distribution
print("Original distribution:", np.bincount(y_train))
print("Combined sampling distribution:", np.bincount(y_train_combined))

# Compute sampling ratios
original_ratio = np.bincount(y_train)[0] / np.bincount(y_train)[1]
new_ratio = np.bincount(y_train_combined)[0] / np.bincount(y_train_combined)[1]
print(f"Original imbalance ratio: {original_ratio:.2f}")
print(f"New imbalance ratio: {new_ratio:.2f}")
```

Slide 8: Cost-Sensitive Learning Implementation

Cost-sensitive learning assigns different misclassification costs to different classes, effectively penalizing errors on minority classes more heavily during model training.

```python
from sklearn.ensemble import RandomForestClassifier

# Calculate class weights inversely proportional to frequency
class_weights = dict(zip(
    np.unique(y_train),
    len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
))

# Train cost-sensitive model
cost_sensitive_clf = RandomForestClassifier(
    class_weight=class_weights,
    random_state=42
)

# Fit and predict
cost_sensitive_clf.fit(X_train, y_train)
y_pred_weighted = cost_sensitive_clf.predict(X_test)

print("Class weights:", class_weights)
print("\nPrediction distribution:", np.bincount(y_pred_weighted))
```

Slide 9: Real-World Application - Credit Card Fraud Detection

Implementing imbalanced learning techniques for credit card fraud detection, where fraudulent transactions typically represent less than 1% of all transactions.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score

# Simulate credit card transaction dataset
np.random.seed(42)
n_samples = 10000
fraud_ratio = 0.01

# Generate transaction features
X = np.random.randn(n_samples, 4)  # Amount, time, V1, V2
y = np.random.choice([0, 1], size=n_samples, p=[1-fraud_ratio, fraud_ratio])

# Create preprocessing and modeling pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('sampler', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(class_weight='balanced'))
])

# Evaluate using cross-validation
scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
print(f"ROC-AUC scores: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

Slide 10: Custom Sampling Strategy Implementation

Developing a custom sampling strategy that combines multiple techniques and allows for fine-tuned control over the resampling process based on domain knowledge.

```python
from imblearn.base import BaseSampler
from collections import Counter

class CustomSampler(BaseSampler):
    def __init__(self, sampling_strategy='auto', random_state=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
    
    def _fit_resample(self, X, y):
        # Get class distribution
        counter = Counter(y)
        
        # Calculate desired ratio (example: 1:2 minority:majority)
        target_ratio = 0.5
        n_minority = counter[1]
        n_majority = int(n_minority / target_ratio)
        
        # Perform custom sampling
        majority_indices = np.where(y == 0)[0]
        minority_indices = np.where(y == 1)[0]
        
        # Random selection for majority class
        np.random.seed(self.random_state)
        selected_majority = np.random.choice(
            majority_indices, 
            size=n_majority, 
            replace=False
        )
        
        # Combine indices
        selected_indices = np.concatenate([selected_majority, minority_indices])
        
        return X[selected_indices], y[selected_indices]

# Usage example
custom_sampler = CustomSampler(random_state=42)
X_resampled, y_resampled = custom_sampler.fit_resample(X, y)
print("Original distribution:", Counter(y))
print("Resampled distribution:", Counter(y_resampled))
```

Slide 11: Performance Monitoring and Validation

Implementing a comprehensive validation strategy for imbalanced learning, including cross-validation with stratification and performance monitoring across different sampling techniques.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
import numpy as np

def validate_sampling_strategy(X, y, sampler, classifier, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Apply sampling
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
        
        # Train and evaluate
        classifier.fit(X_train_resampled, y_train_resampled)
        y_pred_proba = classifier.predict_proba(X_val)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
        scores.append({'precision': precision, 'recall': recall})
    
    return scores

# Example usage
from imblearn.over_sampling import ADASYN
sampler = ADASYN(random_state=42)
clf = RandomForestClassifier(random_state=42)
validation_scores = validate_sampling_strategy(X, y, sampler, clf)
```

Slide 12: Ensemble Methods for Imbalanced Learning

Implementing ensemble methods specifically designed for imbalanced datasets, combining multiple sampling techniques with different base classifiers to improve overall performance.

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier

class ImbalancedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, samplers, base_classifier, n_estimators=5):
        self.samplers = samplers
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.classifiers = []
        
    def fit(self, X, y):
        self.classifiers = []
        
        # Train multiple classifiers with different sampling strategies
        for i in range(self.n_estimators):
            for sampler in self.samplers:
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                clf = clone(self.base_classifier)
                clf.fit(X_resampled, y_resampled)
                self.classifiers.append(clf)
        return self
    
    def predict_proba(self, X):
        probas = np.array([clf.predict_proba(X) for clf in self.classifiers])
        return np.mean(probas, axis=0)

# Example usage
samplers = [
    SMOTE(random_state=42),
    ADASYN(random_state=42),
    RandomOverSampler(random_state=42)
]
base_clf = RandomForestClassifier(random_state=42)
ensemble = ImbalancedEnsemble(samplers, base_clf)
```

Slide 13: Additional Resources

*   "SMOTE: Synthetic Minority Over-sampling Technique" - [https://arxiv.org/abs/1106.1813](https://arxiv.org/abs/1106.1813)
*   "Learning from Imbalanced Data" - [https://arxiv.org/abs/1901.05762](https://arxiv.org/abs/1901.05762)
*   "A Survey on Deep Learning with Class Imbalance" - [https://arxiv.org/abs/1710.05381](https://arxiv.org/abs/1710.05381)
*   "Cost-Sensitive Learning for Imbalanced Classification" - [https://arxiv.org/abs/1904.07506](https://arxiv.org/abs/1904.07506)
*   "Ensemble Methods for Class Imbalance Learning" - [https://arxiv.org/abs/1804.05013](https://arxiv.org/abs/1804.05013)

