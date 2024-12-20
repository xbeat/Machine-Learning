## Explaining ROC Curve and AUC in Model Evaluation with Python
Slide 1: Introduction to ROC Curve and AUC

The Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) are essential tools for evaluating binary classification models. They help assess a model's ability to distinguish between classes across various classification thresholds.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Generate sample data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_scores = np.random.rand(1000)

# Compute ROC curve and AUC
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

Slide 2: Understanding True Positive Rate (TPR) and False Positive Rate (FPR)

The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various classification thresholds. TPR, also known as sensitivity or recall, measures the proportion of actual positive cases correctly identified. FPR represents the proportion of actual negative cases incorrectly classified as positive.

```python
import numpy as np

def calculate_tpr_fpr(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    tpr = true_positives / (true_positives + false_negatives)
    fpr = false_positives / (false_positives + true_negatives)
    
    return tpr, fpr

# Example usage
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 1, 1, 1, 0, 0, 1, 0])

tpr, fpr = calculate_tpr_fpr(y_true, y_pred)
print(f"True Positive Rate: {tpr:.2f}")
print(f"False Positive Rate: {fpr:.2f}")
```

Slide 3: Interpreting the ROC Curve

The ROC curve visualizes the trade-off between sensitivity (TPR) and specificity (1 - FPR) across different classification thresholds. A perfect classifier would have a point at (0, 1), representing 100% TPR and 0% FPR. The diagonal line represents random guessing, while curves above this line indicate better-than-random performance.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curves(y_true, y_scores_list, labels):
    plt.figure(figsize=(10, 8))
    
    for y_scores, label in zip(y_scores_list, labels):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Generate sample data for three classifiers
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_scores_good = np.random.beta(8, 2, 1000)
y_scores_fair = np.random.beta(2, 2, 1000)
y_scores_poor = np.random.beta(2, 8, 1000)

# Plot ROC curves
plot_roc_curves(y_true, [y_scores_good, y_scores_fair, y_scores_poor], ['Good', 'Fair', 'Poor'])
```

Slide 4: Area Under the Curve (AUC)

The Area Under the ROC Curve (AUC) summarizes the model's performance across all possible thresholds. AUC ranges from 0 to 1, with 0.5 representing random guessing and 1 indicating perfect classification. Higher AUC values suggest better model performance.

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_auc(y_true, y_scores):
    auc = roc_auc_score(y_true, y_scores)
    return auc

# Generate sample data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_scores_good = np.random.beta(8, 2, 1000)
y_scores_fair = np.random.beta(2, 2, 1000)
y_scores_poor = np.random.beta(2, 8, 1000)

# Calculate AUC for each classifier
auc_good = calculate_auc(y_true, y_scores_good)
auc_fair = calculate_auc(y_true, y_scores_fair)
auc_poor = calculate_auc(y_true, y_scores_poor)

print(f"AUC (Good): {auc_good:.3f}")
print(f"AUC (Fair): {auc_fair:.3f}")
print(f"AUC (Poor): {auc_poor:.3f}")
```

Slide 5: Calculating ROC Curve and AUC using scikit-learn

Scikit-learn provides convenient functions to calculate ROC curve coordinates and AUC. This example demonstrates how to use these functions and plot the results.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate a random binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities for the test set
y_scores = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_scores)
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

print(f"AUC: {roc_auc:.3f}")
```

Slide 6: Comparing Multiple Models Using ROC Curves

ROC curves are useful for comparing the performance of multiple models on the same dataset. This example demonstrates how to plot ROC curves for different classifiers and compare their AUC scores.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Generate a random binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train different models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(probability=True)
}

plt.figure(figsize=(10, 8))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Classifiers')
plt.legend(loc="lower right")
plt.show()
```

Slide 7: Threshold Selection Using ROC Curve

The ROC curve can help in selecting an optimal classification threshold based on the desired trade-off between true positive rate and false positive rate. This example demonstrates how to find the threshold that maximizes the difference between TPR and FPR.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate a random binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities for the test set
y_scores = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Find the optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Plot ROC curve and optimal threshold
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, label=f'Optimal threshold: {optimal_threshold:.2f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Optimal Threshold')
plt.legend(loc="lower right")
plt.show()

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"True Positive Rate: {tpr[optimal_idx]:.3f}")
print(f"False Positive Rate: {fpr[optimal_idx]:.3f}")
```

Slide 8: Handling Imbalanced Datasets

ROC curves and AUC can be misleading for imbalanced datasets. In such cases, precision-recall curves may be more informative. This example compares ROC and precision-recall curves for an imbalanced dataset.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate an imbalanced binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities for the test set
y_scores = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Calculate precision-recall curve and average precision
precision, recall, _ = precision_recall_curve(y_test, y_scores)
avg_precision = average_precision_score(y_test, y_scores)

# Plot ROC curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Plot precision-recall curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()
```

Slide 9: Cross-Validation and ROC Curves

Cross-validation helps obtain a more robust estimate of model performance. This example demonstrates how to use k-fold cross-validation to generate multiple ROC curves and calculate confidence intervals for the AUC.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Generate sample data
X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)

# Perform 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression()

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(8, 6))

for i, (train, test) in enumerate(cv.split(X, y)):
    model.fit(X[train], y[train])
    y_score = model.predict_proba(X[test])[:, 1]
    fpr, tpr, _ = roc_curve(y[test], y_score)
    roc_auc = auc(fpr, tpr)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    tprs.append(interp_tpr)
    aucs.append(roc_auc)
    ax.plot(fpr, tpr, alpha=0.3, label=f'ROC fold {i+1} (AUC = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], linestyle='--', color='r', alpha=0.8)
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Cross-Validated ROC Curves')
ax.legend(loc="lower right")
plt.show()
```

Slide 10: ROC Curve for Multi-class Classification

ROC curves can be extended to multi-class problems using one-vs-rest or one-vs-one approaches. This example demonstrates the one-vs-rest method for a three-class classification problem.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Generate sample data
X, y = make_classification(n_samples=1000, n_classes=3, n_informative=3, random_state=42)
y = label_binarize(y, classes=[0, 1, 2])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a multi-class classifier
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)
y_score = clf.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 3

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
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
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.show()
```

Slide 11: Real-life Example: Medical Diagnosis

ROC curves and AUC are widely used in medical diagnosis to evaluate the performance of diagnostic tests. This example simulates a diagnostic test for a hypothetical disease.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Simulate test results for 1000 patients (200 with the disease, 800 without)
np.random.seed(42)
n_patients = 1000
n_diseased = 200

true_condition = np.concatenate([np.ones(n_diseased), np.zeros(n_patients - n_diseased)])

# Simulate test scores (higher score indicates higher likelihood of disease)
diseased_scores = np.random.normal(loc=7, scale=2, size=n_diseased)
healthy_scores = np.random.normal(loc=4, scale=2, size=n_patients - n_diseased)
test_scores = np.concatenate([diseased_scores, healthy_scores])

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(true_condition, test_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Hypothetical Disease Diagnostic Test')
plt.legend(loc="lower right")
plt.show()

# Find optimal threshold (maximize sensitivity + specificity)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.2f}")
print(f"Sensitivity at optimal threshold: {tpr[optimal_idx]:.2f}")
print(f"Specificity at optimal threshold: {1 - fpr[optimal_idx]:.2f}")
```

Slide 12: Real-life Example: Spam Detection

ROC curves and AUC are commonly used in evaluating spam detection algorithms. This example simulates a simple spam classifier based on message length and keyword count.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# Simulate email data
np.random.seed(42)
n_emails = 1000
message_length = np.random.normal(loc=100, scale=30, size=n_emails)
spam_keywords = np.random.poisson(lam=2, size=n_emails)

# Create features and labels
X = np.column_stack((message_length, spam_keywords))
y = (np.random.rand(n_emails) < 0.3).astype(int)  # 30% spam

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Spam Detection')
plt.legend(loc="lower right")
plt.show()

# Find optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.2f}")
print(f"True Positive Rate at optimal threshold: {tpr[optimal_idx]:.2f}")
print(f"False Positive Rate at optimal threshold: {fpr[optimal_idx]:.2f}")
```

Slide 13: Limitations and Considerations

When using ROC curves and AUC for model evaluation, consider the following:

1. Class imbalance: ROC curves may be less informative for highly imbalanced datasets.
2. Threshold-dependent metrics: ROC curves don't provide information about actual predicted probabilities.
3. Equal misclassification costs: AUC assumes equal costs for false positives and false negatives.
4. Comparing models: AUC should be used cautiously when comparing models across different datasets.

Slide 14: Limitations and Considerations

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Generate imbalanced dataset
np.random.seed(42)
n_samples = 1000
n_positives = 50
y_true = np.zeros(n_samples)
y_true[:n_positives] = 1
y_scores = np.random.rand(n_samples)

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall curve and Average Precision
precision, recall, _ = precision_recall_curve(y_true, y_scores)
avg_precision = average_precision_score(y_true, y_scores)

# Plot ROC and Precision-Recall curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve (Imbalanced Dataset)')
ax1.legend(loc="lower right")

ax2.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve (Imbalanced Dataset)')
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()

print(f"Class imbalance ratio: {n_positives / n_samples:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")
print(f"Average Precision: {avg_precision:.2f}")
```

Slide 15: Additional Resources

For further reading on ROC curves, AUC, and model evaluation techniques, consider the following resources:

1. Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861-874. ArXiv: [https://arxiv.org/abs/cs/0303027](https://arxiv.org/abs/cs/0303027)
2. Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. Proceedings of the 23rd International Conference on Machine Learning. ArXiv: [https://arxiv.org/abs/cs/0606118](https://arxiv.org/abs/cs/0606118)
3. Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. Radiology, 143(1), 29-36.

