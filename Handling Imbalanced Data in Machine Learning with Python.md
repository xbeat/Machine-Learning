## Handling Imbalanced Data in Machine Learning with Python
Slide 1: Imbalanced Data in Machine Learning

Imbalanced data occurs when one class significantly outnumbers the other(s) in a dataset. This common problem can lead to biased models that perform poorly on minority classes. In this presentation, we'll explore various techniques to handle imbalanced data using Python, focusing on practical implementations and real-world applications.

```python
# Simulating an imbalanced dataset
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], 
                           n_features=20, random_state=42)

print(f"Class distribution: {np.bincount(y)}")
print(f"Class ratio: 1:{np.bincount(y)[0] / np.bincount(y)[1]:.2f}")
```

Slide 2: Oversampling: Random Over-Sampling

Random over-sampling is a simple technique that involves randomly duplicating examples from the minority class to balance the dataset. While easy to implement, it can lead to overfitting.

```python
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply random over-sampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

print(f"Original class distribution: {np.bincount(y_train)}")
print(f"Resampled class distribution: {np.bincount(y_resampled)}")
```

Slide 3: Oversampling: SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE creates synthetic examples in the feature space by interpolating between existing minority instances. This technique helps to address the overfitting problem associated with random oversampling.

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(f"Original class distribution: {np.bincount(y_train)}")
print(f"Resampled class distribution: {np.bincount(y_resampled)}")

# Visualize SMOTE (for 2D data)
import matplotlib.pyplot as plt

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.5)
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled, alpha=0.5)
plt.title("Original vs SMOTE-augmented data")
plt.show()
```

Slide 4: Undersampling: Random Under-Sampling

Random under-sampling involves randomly removing examples from the majority class to balance the dataset. While it can help with class imbalance, it may discard potentially useful information.

```python
from imblearn.under_sampling import RandomUnderSampler

# Apply random under-sampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

print(f"Original class distribution: {np.bincount(y_train)}")
print(f"Resampled class distribution: {np.bincount(y_resampled)}")
```

Slide 5: Undersampling: Tomek Links

Tomek links are pairs of instances of opposite classes that are closest neighbors. Removing the majority instance of each Tomek link can help clean the overlap between classes.

```python
from imblearn.under_sampling import TomekLinks

# Apply Tomek Links
tl = TomekLinks()
X_resampled, y_resampled = tl.fit_resample(X_train, y_train)

print(f"Original class distribution: {np.bincount(y_train)}")
print(f"Resampled class distribution: {np.bincount(y_resampled)}")

# Visualize Tomek Links (for 2D data)
removed = np.setdiff1d(X_train, X_resampled)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.5)
plt.scatter(removed[:, 0], removed[:, 1], c='red', marker='x', s=200, label='Removed')
plt.legend()
plt.title("Tomek Links Undersampling")
plt.show()
```

Slide 6: Combination: SMOTEENN

SMOTEENN combines SMOTE oversampling with Edited Nearest Neighbors (ENN) cleaning. It first applies SMOTE to oversample the minority class, then uses ENN to remove instances from both classes, potentially improving class separation.

```python
from imblearn.combine import SMOTEENN

# Apply SMOTEENN
smoteenn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)

print(f"Original class distribution: {np.bincount(y_train)}")
print(f"Resampled class distribution: {np.bincount(y_resampled)}")

# Visualize SMOTEENN (for 2D data)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.5)
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled, alpha=0.5)
plt.title("Original vs SMOTEENN-resampled data")
plt.show()
```

Slide 7: Class Weights

Instead of resampling, we can adjust the importance of each class during model training. Many sklearn classifiers accept a 'class\_weight' parameter to automatically adjust weights inversely proportional to class frequencies.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Train a decision tree with balanced class weights
clf = DecisionTreeClassifier(class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Compare with unweighted model
clf_unweighted = DecisionTreeClassifier(random_state=42)
clf_unweighted.fit(X_train, y_train)
y_pred_unweighted = clf_unweighted.predict(X_test)
print(classification_report(y_test, y_pred_unweighted))
```

Slide 8: Ensemble Methods: BalancedRandomForestClassifier

Ensemble methods like Random Forests can be adapted to handle imbalanced datasets. The BalancedRandomForestClassifier combines the ideas of random under-sampling and ensemble learning.

```python
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_val_score

# Create and evaluate a BalancedRandomForestClassifier
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(brf, X, y, cv=5, scoring='f1')

print(f"Mean F1-score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Compare with standard RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
scores_rf = cross_val_score(rf, X, y, cv=5, scoring='f1')

print(f"Standard RF Mean F1-score: {scores_rf.mean():.3f} (+/- {scores_rf.std() * 2:.3f})")
```

Slide 9: Anomaly Detection: Isolation Forest

For extreme imbalances, treating the minority class as anomalies can be effective. Isolation Forest is an unsupervised learning algorithm that isolates anomalies in the feature space.

```python
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix

# Fit Isolation Forest
clf = IsolationForest(contamination=0.1, random_state=42)
y_pred = clf.fit_predict(X)

# Convert predictions to binary classification (-1 becomes 1, 1 becomes 0)
y_pred = [1 if x == -1 else 0 for x in y_pred]

# Compute confusion matrix
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize results (for 2D data)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title("Isolation Forest Results")
plt.colorbar(label='Class')
plt.show()
```

Slide 10: Real-life Example: Fraud Detection

Fraud detection is a common application of imbalanced learning. In this example, we'll use a synthetic dataset to simulate credit card transactions, where fraudulent transactions are rare.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Create synthetic credit card transaction data
n_samples = 10000
n_features = 10
fraud_ratio = 0.01

X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                           n_classes=2, weights=[1-fraud_ratio, fraud_ratio], 
                           random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train and evaluate models
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_resampled, y_train_resampled)

print("Without SMOTE:")
print(classification_report(y_test, rf.predict(X_test)))
print("\nWith SMOTE:")
print(classification_report(y_test, rf_smote.predict(X_test)))
```

Slide 11: Real-life Example: Disease Diagnosis

In medical diagnosis, certain diseases may be rare, leading to imbalanced datasets. Let's simulate a dataset for a rare disease and apply imbalanced learning techniques.

```python
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# Create synthetic medical diagnosis data
n_samples = 5000
n_features = 20
disease_ratio = 0.05

X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                           n_classes=2, weights=[1-disease_ratio, disease_ratio], 
                           random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with random under-sampling and SVM
pipeline = Pipeline([
    ('rus', RandomUnderSampler(random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
])

# Train and evaluate the model
pipeline.fit(X_train, y_train)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")

# Compare with standard SVM
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_proba_std = svm.predict_proba(X_test)[:, 1]

print(f"Standard SVM ROC AUC Score: {roc_auc_score(y_test, y_pred_proba_std):.3f}")
```

Slide 12: Evaluation Metrics for Imbalanced Data

When dealing with imbalanced datasets, accuracy can be misleading. Other metrics like precision, recall, F1-score, and ROC AUC are more informative. Here's how to calculate and interpret these metrics.

```python
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

# Using the results from the previous slide
# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

print(f"ROC AUC: {roc_auc:.3f}")
```

Slide 13: Cross-Validation Strategies for Imbalanced Data

When performing cross-validation on imbalanced datasets, it's crucial to maintain the class distribution across folds. Stratified K-Fold is a good choice for this purpose.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

# Create a stratified k-fold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
f1_scores = []
for train_index, val_index in skf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Apply SMOTE only to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Train and evaluate the model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_resampled, y_train_resampled)
    y_pred = clf.predict(X_val)
    f1_scores.append(f1_score(y_val, y_pred))

print(f"Mean F1-score: {np.mean(f1_scores):.3f} (+/- {np.std(f1_scores) * 2:.3f})")
```

Slide 14: Hyperparameter Tuning for Imbalanced Data

When tuning hyperparameters for models on imbalanced datasets, it's important to use appropriate scoring metrics. Here's an example using GridSearchCV with F1-score as the optimization metric.

```python
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Create a pipeline
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])

# Define parameter grid
param_grid = {
    'smote__k_neighbors': [3, 5, 7],
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [5, 10, None]
}

# Create GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best F1-score:", grid_search.best_score_)

# Evaluate on test set
y_pred = grid_search.predict(X_test)
print("Test F1-score:", f1_score(y_test, y_pred))
```

Slide 15: Additional Resources

For those interested in diving deeper into imbalanced data handling techniques, here are some valuable resources:

1. "Learning from Imbalanced Data" by Haibo He and Edwardo A. Garcia ArXiv URL: [https://arxiv.org/abs/0806.1250](https://arxiv.org/abs/0806.1250)
2. "A Survey of Predictive Modelling under Imbalanced Distributions" by Paula Branco, Luís Torgo, and Rita P. Ribeiro ArXiv URL: [https://arxiv.org/abs/1505.01658](https://arxiv.org/abs/1505.01658)
3. "SMOTE: Synthetic Minority Over-sampling Technique" by Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, and W. Philip Kegelmeyer ArXiv URL: [https://arxiv.org/abs/1106.1813](https://arxiv.org/abs/1106.1813)

These papers provide in-depth discussions on various techniques and their applications in handling imbalanced datasets.

