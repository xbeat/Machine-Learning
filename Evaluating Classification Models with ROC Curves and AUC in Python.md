## Evaluating Classification Models with ROC Curves and AUC in Python
Slide 1: Introduction to ROC Curves and AUC

ROC (Receiver Operating Characteristic) curves and AUC (Area Under the Curve) are powerful tools for evaluating and comparing classification models. They provide a visual representation of a model's performance across various classification thresholds and offer a single metric to summarize that performance.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Example data
y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_scores = np.array([0.1, 0.7, 0.8, 0.3, 0.9, 0.6, 0.2, 0.4, 0.7, 0.5])

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

Slide 2: Understanding True Positive Rate and False Positive Rate

The True Positive Rate (TPR) and False Positive Rate (FPR) are key components of ROC curves. TPR, also known as sensitivity or recall, measures the proportion of actual positive cases correctly identified. FPR represents the proportion of actual negative cases incorrectly classified as positive.

```python
def calculate_tpr_fpr(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    tpr = true_positives / (true_positives + false_negatives)
    fpr = false_positives / (false_positives + true_negatives)
    
    return tpr, fpr

# Example usage
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 1, 0, 1])

tpr, fpr = calculate_tpr_fpr(y_true, y_pred)
print(f"True Positive Rate: {tpr:.2f}")
print(f"False Positive Rate: {fpr:.2f}")
```

Slide 3: Generating ROC Curves

To create an ROC curve, we need to calculate the TPR and FPR for various classification thresholds. We'll use scikit-learn's roc\_curve function to generate the necessary data points.

```python
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_scores = np.random.rand(1000)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

Slide 4: Calculating Area Under the Curve (AUC)

The Area Under the ROC Curve (AUC) provides a single scalar value to measure the overall performance of a classifier. AUC ranges from 0 to 1, with 0.5 representing a random classifier and 1 representing a perfect classifier.

```python
from sklearn.metrics import roc_auc_score
import numpy as np

# Generate sample data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_scores = np.random.rand(1000)

# Calculate AUC
auc = roc_auc_score(y_true, y_scores)

print(f"Area Under the Curve (AUC): {auc:.3f}")

# Interpret AUC
if auc < 0.5:
    print("Poor performance (worse than random)")
elif auc < 0.7:
    print("Fair performance")
elif auc < 0.8:
    print("Good performance")
elif auc < 0.9:
    print("Very good performance")
else:
    print("Excellent performance")
```

Slide 5: Comparing Multiple Classifiers

ROC curves and AUC scores are particularly useful for comparing the performance of multiple classifiers on the same dataset. This allows us to visually and quantitatively assess which model performs better across different classification thresholds.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifiers
lr = LogisticRegression()
rf = RandomForestClassifier()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predict probabilities
lr_probs = lr.predict_proba(X_test)[:, 1]
rf_probs = rf.predict_proba(X_test)[:, 1]

# Calculate ROC curves and AUC
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

lr_auc = auc(lr_fpr, lr_tpr)
rf_auc = auc(rf_fpr, rf_tpr)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.2f})')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multiple Classifiers')
plt.legend()
plt.show()
```

Slide 6: Handling Imbalanced Datasets

When working with imbalanced datasets, ROC curves might not provide a complete picture of a model's performance. In such cases, it's useful to consider Precision-Recall curves alongside ROC curves.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Generate imbalanced dataset
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict probabilities
y_probs = clf.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall curve and Average Precision
precision, recall, _ = precision_recall_curve(y_test, y_probs)
avg_precision = average_precision_score(y_test, y_probs)

# Plot ROC curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Plot Precision-Recall curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 7: Cross-Validation for Robust AUC Estimation

To get a more reliable estimate of a model's performance, we can use cross-validation to calculate AUC scores across multiple folds of the data.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import numpy as np

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Initialize classifier and cross-validation
clf = LogisticRegression()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
auc_scores = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    clf.fit(X_train, y_train)
    y_probs = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_probs)
    auc_scores.append(auc)
    print(f"Fold {fold} AUC: {auc:.3f}")

# Calculate mean and standard deviation of AUC scores
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)
print(f"\nMean AUC: {mean_auc:.3f} (+/- {std_auc:.3f})")
```

Slide 8: Optimizing Classification Threshold

The default classification threshold is usually 0.5, but we can optimize this threshold based on the ROC curve to find the best balance between true positive rate and false positive rate.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict probabilities
y_probs = clf.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Find optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")

# Apply optimal threshold
y_pred_optimal = (y_probs >= optimal_threshold).astype(int)

# Calculate accuracy with optimal threshold
accuracy_optimal = np.mean(y_pred_optimal == y_test)
print(f"Accuracy with optimal threshold: {accuracy_optimal:.3f}")

# Compare with default threshold
y_pred_default = (y_probs >= 0.5).astype(int)
accuracy_default = np.mean(y_pred_default == y_test)
print(f"Accuracy with default threshold: {accuracy_default:.3f}")
```

Slide 9: Visualizing Decision Boundaries

To better understand how the ROC curve relates to the model's decision boundary, we can visualize the decision boundary alongside the ROC curve for a simple 2D dataset.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# Generate 2D dataset
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_redundant=0, n_informative=2, random_state=42)

# Train logistic regression
clf = LogisticRegression()
clf.fit(X, y)

# Create a mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict probabilities for the mesh grid
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Calculate ROC curve and AUC
y_pred_proba = clf.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot decision boundary and data points
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')

# Plot ROC curve
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()
```

Slide 10: ROC Curves for Multi-class Classification

While ROC curves are typically used for binary classification, they can be extended to multi-class problems using a one-vs-rest approach.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a multi-class classifier
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)

# Compute ROC curve and ROC area for each class
y_score = clf.predict_proba(X_test)
n_classes = len(np.unique(y))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves')
plt.legend(loc="lower right")
plt.show()
```

Slide 11: Confidence Intervals for AUC

To assess the reliability of our AUC score, we can compute confidence intervals using bootstrapping.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats

def bootstrap_auc(y_true, y_pred, n_bootstraps=1000, ci=0.95):
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for i in range(n_bootstraps):
        # Bootstrap by sampling with replacement
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Compute confidence interval
    confidence_lower = sorted_scores[int((1.0-ci)/2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int((1.0+ci)/2 * len(sorted_scores))]
    return np.mean(bootstrapped_scores), confidence_lower, confidence_upper

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict probabilities
y_pred = clf.predict_proba(X_test)[:, 1]

# Calculate AUC and confidence interval
auc, ci_lower, ci_upper = bootstrap_auc(y_test, y_pred)

print(f"AUC: {auc:.3f}")
print(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

Slide 12: Partial AUC

In some applications, we may be interested in only a specific region of the ROC curve. Partial AUC allows us to focus on a particular range of false positive rates.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

def partial_auc(fpr, tpr, max_fpr):
    # Find the index of the first FPR value greater than max_fpr
    cut_point = next(i for i, x in enumerate(fpr) if x > max_fpr)
    
    # Linearly interpolate the TPR at max_fpr
    slope = (tpr[cut_point] - tpr[cut_point-1]) / (fpr[cut_point] - fpr[cut_point-1])
    tpr_interp = tpr[cut_point-1] + slope * (max_fpr - fpr[cut_point-1])
    
    # Compute partial AUC
    partial_auc = np.trapz([tpr[0]] + list(tpr[:cut_point]) + [tpr_interp], [0] + list(fpr[:cut_point]) + [max_fpr])
    return partial_auc / max_fpr

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict probabilities
y_pred = clf.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)

# Calculate partial AUC
max_fpr = 0.2
pauc = partial_auc(fpr, tpr, max_fpr)

# Plot ROC curve and partial AUC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
plt.plot([0, max_fpr], [0, tpr[np.argmax(fpr > max_fpr)]], color='red', lw=2, linestyle='--', label=f'Partial AUC (FPR <= {max_fpr})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve and Partial AUC (pAUC = {pauc:.3f})')
plt.legend(loc="lower right")
plt.show()
```

Slide 13: ROC Curves for Imbalanced Datasets

When dealing with imbalanced datasets, it's important to consider alternatives to ROC curves, such as Precision-Recall curves, which can provide a more informative view of a model's performance.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Generate imbalanced dataset
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict probabilities
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall curve and Average Precision
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

# Plot ROC curve and Precision-Recall curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax1.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random classifier')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
ax1.legend(loc="lower right")

ax2.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve')
ax2.legend(loc="lower left")

plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into ROC curves, AUC, and related topics, here are some valuable resources:

1. Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861-874. ArXiv link: [https://arxiv.org/abs/cs/0303029](https://arxiv.org/abs/cs/0303029)
2. Bradley, A. P. (1997). The use of the area under the ROC curve in the evaluation of machine learning algorithms. Pattern Recognition, 30(7), 1145-1159.
3. Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. Proceedings of the 23rd International Conference on Machine Learning. ArXiv link: [https://arxiv.org/abs/cs/0606118](https://arxiv.org/abs/cs/0606118)
4. Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. Radiology, 143(1), 29-36.

These resources provide in-depth explanations and analyses of ROC curves, AUC, and their applications in machine learning and statistics.

