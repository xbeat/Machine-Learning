## Handling Imbalanced Datasets with kNN in Python
Slide 1: kNN for Imbalanced Datasets

k-Nearest Neighbors (kNN) is a popular machine learning algorithm, but it can struggle with imbalanced datasets. This presentation will explore techniques to effectively use kNN on highly imbalanced data using Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

# Generate an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], 
                           n_features=2, n_informative=2, n_redundant=0, 
                           random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.title('Imbalanced Dataset')
plt.show()
```

Slide 2: Understanding Imbalanced Datasets

An imbalanced dataset is one where the classes are not represented equally. This can lead to biased models that perform poorly on minority classes. In our example, we have a binary classification problem with a 9:1 ratio between classes.

```python
from collections import Counter

class_distribution = Counter(y)
print("Class distribution:", class_distribution)

# Calculate class weights
n_samples = len(y)
class_weights = {0: n_samples / (2 * class_distribution[0]),
                 1: n_samples / (2 * class_distribution[1])}
print("Class weights:", class_weights)
```

Slide 3: Basic kNN Implementation

Let's start with a basic kNN implementation to see how it performs on our imbalanced dataset.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a basic kNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
```

Slide 4: Addressing Class Imbalance: Resampling

One approach to handle imbalanced datasets is resampling. We'll use the imbalanced-learn library to oversample the minority class using SMOTE (Synthetic Minority Over-sampling Technique).

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train kNN on resampled data
knn_resampled = KNeighborsClassifier(n_neighbors=5)
knn_resampled.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred_resampled = knn_resampled.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred_resampled))
```

Slide 5: Weighted kNN

Another approach is to use weighted kNN, where the influence of each neighbor is weighted by its distance. This can help mitigate the impact of imbalanced classes.

```python
# Train a weighted kNN model
knn_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_weighted.fit(X_train, y_train)

# Make predictions
y_pred_weighted = knn_weighted.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred_weighted))
```

Slide 6: Adjusting Decision Threshold

We can also adjust the decision threshold for kNN to favor the minority class. This involves changing the probability threshold for classification.

```python
# Get probability predictions
y_prob = knn.predict_proba(X_test)

# Adjust threshold (e.g., to 0.3 instead of default 0.5)
y_pred_adjusted = (y_prob[:, 1] >= 0.3).astype(int)

# Evaluate the model with adjusted threshold
print(classification_report(y_test, y_pred_adjusted))
```

Slide 7: Ensemble Methods: Bagging

Ensemble methods can improve kNN performance on imbalanced datasets. Let's implement a simple bagging ensemble.

```python
from sklearn.ensemble import BaggingClassifier

# Create a bagging ensemble of kNN classifiers
bagging_knn = BaggingClassifier(KNeighborsClassifier(n_neighbors=5),
                                n_estimators=10, random_state=42)
bagging_knn.fit(X_train, y_train)

# Make predictions
y_pred_bagging = bagging_knn.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred_bagging))
```

Slide 8: Feature Engineering

Feature engineering can help improve kNN performance on imbalanced datasets. Let's create a new feature based on the distance to the 5 nearest neighbors of each class.

```python
def distance_feature(X, y, n_neighbors=5):
    knn_class0 = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_class1 = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    knn_class0.fit(X[y == 0], y[y == 0])
    knn_class1.fit(X[y == 1], y[y == 1])
    
    dist_class0 = knn_class0.kneighbors(X)[0].sum(axis=1)
    dist_class1 = knn_class1.kneighbors(X)[0].sum(axis=1)
    
    return np.column_stack((X, dist_class0, dist_class1))

# Apply feature engineering
X_featured = distance_feature(X, y)
X_train_featured, X_test_featured, y_train, y_test = train_test_split(X_featured, y, test_size=0.2, random_state=42)

# Train kNN on featured data
knn_featured = KNeighborsClassifier(n_neighbors=5)
knn_featured.fit(X_train_featured, y_train)

# Make predictions
y_pred_featured = knn_featured.predict(X_test_featured)

# Evaluate the model
print(classification_report(y_test, y_pred_featured))
```

Slide 9: Cross-Validation for Imbalanced Data

When working with imbalanced datasets, it's crucial to use appropriate cross-validation techniques. We'll use stratified k-fold cross-validation to maintain class distribution across folds.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# Set up stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = []
for train_index, val_index in skf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    
    cv_scores.append(f1_score(y_val, y_pred))

print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean F1 score: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
```

Slide 10: Real-Life Example: Fault Detection in Manufacturing

In a manufacturing setting, fault detection is crucial for maintaining product quality. However, fault occurrences are typically rare, leading to an imbalanced dataset. Let's simulate this scenario and apply our kNN techniques.

```python
# Simulate manufacturing data
np.random.seed(42)
n_samples = 1000
n_features = 5

# Generate normal samples (no fault)
X_normal = np.random.normal(loc=0, scale=1, size=(int(0.99 * n_samples), n_features))
y_normal = np.zeros(int(0.99 * n_samples))

# Generate fault samples
X_fault = np.random.normal(loc=2, scale=1.5, size=(int(0.01 * n_samples), n_features))
y_fault = np.ones(int(0.01 * n_samples))

X = np.vstack((X_normal, X_fault))
y = np.hstack((y_normal, y_fault))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train kNN on resampled data
knn_resampled = KNeighborsClassifier(n_neighbors=5)
knn_resampled.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred_resampled = knn_resampled.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred_resampled))
```

Slide 11: Real-Life Example: Rare Disease Diagnosis

Medical diagnosis often involves imbalanced datasets, especially for rare diseases. Let's simulate a dataset for diagnosing a rare disease and apply our kNN techniques.

```python
# Simulate medical diagnosis data
np.random.seed(42)
n_samples = 1000
n_features = 10

# Generate healthy samples
X_healthy = np.random.normal(loc=0, scale=1, size=(int(0.98 * n_samples), n_features))
y_healthy = np.zeros(int(0.98 * n_samples))

# Generate disease samples
X_disease = np.random.normal(loc=1, scale=1.2, size=(int(0.02 * n_samples), n_features))
y_disease = np.ones(int(0.02 * n_samples))

X = np.vstack((X_healthy, X_disease))
y = np.hstack((y_healthy, y_disease))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply feature engineering
X_train_featured = distance_feature(X_train, y_train)
X_test_featured = distance_feature(X_test, y_test)

# Train weighted kNN on featured data
knn_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_weighted.fit(X_train_featured, y_train)

# Make predictions
y_pred_weighted = knn_weighted.predict(X_test_featured)

# Evaluate the model
print(classification_report(y_test, y_pred_weighted))
```

Slide 12: Hyperparameter Tuning for Imbalanced Data

When working with imbalanced datasets, it's important to tune hyperparameters using appropriate metrics. We'll use GridSearchCV with F1 score as the scoring metric.

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Set up GridSearchCV
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1', n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best F1 score:", grid_search.best_score_)

# Evaluate the best model on the test set
best_knn = grid_search.best_estimator_
y_pred_best = best_knn.predict(X_test)
print(classification_report(y_test, y_pred_best))
```

Slide 13: Combining Techniques

To achieve the best results, we often need to combine multiple techniques. Let's combine SMOTE, feature engineering, and weighted kNN.

```python
# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Apply feature engineering
X_train_featured = distance_feature(X_train_resampled, y_train_resampled)
X_test_featured = distance_feature(X_test, y_test)

# Train weighted kNN on featured and resampled data
knn_combined = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_combined.fit(X_train_featured, y_train_resampled)

# Make predictions
y_pred_combined = knn_combined.predict(X_test_featured)

# Evaluate the model
print(classification_report(y_test, y_pred_combined))
```

Slide 14: Conclusion and Best Practices

When using kNN on imbalanced datasets:

1. Start with resampling techniques like SMOTE
2. Consider weighted kNN or adjusting decision thresholds
3. Apply feature engineering to create more informative features
4. Use appropriate cross-validation techniques (e.g., stratified k-fold)
5. Choose appropriate evaluation metrics (e.g., F1 score, precision-recall AUC)
6. Combine multiple techniques for best results
7. Always validate your model on a separate test set

Remember that the best approach may vary depending on your specific dataset and problem domain.

Slide 15: Additional Resources

For more information on handling imbalanced datasets and kNN:

1. "Learning from Imbalanced Data" by Haibo He and Edwardo A. Garcia ArXiv: [https://arxiv.org/abs/0806.1250](https://arxiv.org/abs/0806.1250)
2. "A Survey of Predictive Modelling under Imbalanced Distributions" by Paula Branco, Luís Torgo, and Rita P. Ribeiro ArXiv: [https://arxiv.org/abs/1505.01658](https://arxiv.org/abs/1505.01658)
3. "Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning" by Guillaume Lemaître, Fernando Nogueira, and Christos K. Aridas ArXiv: [https://arxiv.org/abs/1609.06570](https://arxiv.org/abs/1609.06570)

These resources provide in-depth discussions on various techniques for handling imbalanced datasets, including those applicable to kNN and other machine learning algorithms.

