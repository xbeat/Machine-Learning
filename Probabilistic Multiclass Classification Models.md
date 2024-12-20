## Probabilistic Multiclass Classification Models
Slide 1: Understanding the Limitations of Accuracy in Multiclass Classification

Accuracy, while intuitive, can be misleading in multiclass settings. It fails to capture the nuances of model performance, especially when dealing with imbalanced datasets or when the costs of different types of errors vary. This slideshow explores why relying solely on accuracy can be problematic and introduces more robust evaluation metrics for multiclass classification.

```python
import numpy as np
from sklearn.metrics import accuracy_score

# Example of how accuracy can be misleading
y_true = np.array([0, 1, 2, 2, 2])
y_pred = np.array([0, 0, 2, 2, 1])

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")  # Output: Accuracy: 0.6

# Despite 60% accuracy, the model misclassified the only instance of class 1
```

Slide 2: The Class Imbalance Problem

In multiclass settings, class imbalance is common. Accuracy can be deceptively high if a model simply predicts the majority class for all instances. This slide demonstrates how accuracy fails to capture poor performance on minority classes.

```python
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Simulating a highly imbalanced dataset
y_true = np.array([0] * 90 + [1] * 5 + [2] * 5)
y_pred = np.array([0] * 100)  # Model always predicts the majority class

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")  # Output: Accuracy: 0.9

print("\nDetailed classification report:")
print(classification_report(y_true, y_pred))
```

Slide 3: Beyond Accuracy: Precision, Recall, and F1-Score

To address the limitations of accuracy, we introduce precision, recall, and F1-score. These metrics provide a more comprehensive view of model performance, especially for imbalanced datasets.

```python
from sklearn.metrics import precision_recall_fscore_support

# Using the same imbalanced dataset from the previous slide
y_true = np.array([0] * 90 + [1] * 5 + [2] * 5)
y_pred = np.array([0] * 100)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
```

Slide 4: Confusion Matrix: A Detailed View of Model Performance

The confusion matrix provides a comprehensive breakdown of correct and incorrect classifications for each class. It's particularly useful for identifying which classes the model struggles with most.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_true = np.array([0, 0, 1, 1, 2, 2, 2, 2])
y_pred = np.array([0, 1, 0, 1, 0, 2, 2, 2])

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

Slide 5: Cohen's Kappa: Accounting for Chance Agreement

Cohen's Kappa is a metric that takes into account the possibility of agreement occurring by chance. It's particularly useful when dealing with imbalanced datasets or when some classifications are more likely than others.

```python
from sklearn.metrics import cohen_kappa_score

y_true = np.array([0, 0, 1, 1, 2, 2, 2, 2])
y_pred = np.array([0, 1, 0, 1, 0, 2, 2, 2])

kappa = cohen_kappa_score(y_true, y_pred)
print(f"Cohen's Kappa: {kappa:.2f}")

# Interpretation:
# < 0: Poor agreement
# 0.0 - 0.20: Slight agreement
# 0.21 - 0.40: Fair agreement
# 0.41 - 0.60: Moderate agreement
# 0.61 - 0.80: Substantial agreement
# 0.81 - 1.00: Almost perfect agreement
```

Slide 6: Macro vs. Micro vs. Weighted Averaging

When dealing with multiclass problems, we often need to aggregate metrics across classes. Different averaging methods can lead to different interpretations of model performance.

```python
from sklearn.metrics import precision_recall_fscore_support

y_true = np.array([0, 0, 1, 1, 2, 2, 2, 2])
y_pred = np.array([0, 1, 0, 1, 0, 2, 2, 2])

# Macro averaging
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

# Micro averaging
micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

# Weighted averaging
weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

print(f"Macro F1: {macro_f1:.2f}")
print(f"Micro F1: {micro_f1:.2f}")
print(f"Weighted F1: {weighted_f1:.2f}")
```

Slide 7: ROC Curves and AUC for Multiclass Problems

While traditionally used for binary classification, ROC curves and AUC can be extended to multiclass problems using one-vs-rest or one-vs-one approaches.

```python
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Simulating probabilistic predictions for a 3-class problem
y_true = np.array([0, 0, 1, 1, 2, 2, 2, 2])
y_score = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.5, 0.2],
    [0.2, 0.7, 0.1],
    [0.1, 0.8, 0.1],
    [0.2, 0.2, 0.6],
    [0.1, 0.1, 0.8],
    [0.1, 0.2, 0.7],
    [0.1, 0.1, 0.8]
])

# Binarize the output
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {i}) (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

Slide 8: Probabilistic Predictions and Log Loss

Probabilistic predictions provide more information than hard classifications. Log loss (cross-entropy) is a suitable metric for evaluating probabilistic predictions in multiclass settings.

```python
import numpy as np
from sklearn.metrics import log_loss

# True labels
y_true = np.array([0, 1, 2, 2])

# Probabilistic predictions
y_pred_proba = np.array([
    [0.7, 0.2, 0.1],  # High confidence, correct
    [0.3, 0.6, 0.1],  # Moderate confidence, correct
    [0.2, 0.3, 0.5],  # Low confidence, correct
    [0.6, 0.3, 0.1]   # High confidence, incorrect
])

logloss = log_loss(y_true, y_pred_proba)
print(f"Log Loss: {logloss:.4f}")

# Lower log loss indicates better performance
# Perfect predictions would result in a log loss of 0
```

Slide 9: Brier Score for Multiclass Problems

The Brier score is another metric for evaluating probabilistic predictions. It measures the mean squared difference between the predicted probability and the actual outcome.

```python
import numpy as np
from sklearn.metrics import brier_score_loss

# True labels (one-hot encoded)
y_true = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1]
])

# Probabilistic predictions
y_pred_proba = np.array([
    [0.7, 0.2, 0.1],  # High confidence, correct
    [0.3, 0.6, 0.1],  # Moderate confidence, correct
    [0.2, 0.3, 0.5],  # Low confidence, correct
    [0.6, 0.3, 0.1]   # High confidence, incorrect
])

# Calculate Brier score for each class
brier_scores = np.mean((y_true - y_pred_proba) ** 2, axis=0)

print("Brier scores for each class:")
for i, score in enumerate(brier_scores):
    print(f"Class {i}: {score:.4f}")

# Calculate overall Brier score
overall_brier_score = np.mean(brier_scores)
print(f"\nOverall Brier Score: {overall_brier_score:.4f}")

# Lower Brier scores indicate better calibrated probabilities
```

Slide 10: Calibration Curves for Multiclass Problems

Calibration curves (reliability diagrams) help visualize how well the predicted probabilities of a classifier are calibrated. For multiclass problems, we can create calibration curves for each class.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Simulating a larger dataset
np.random.seed(42)
n_samples = 1000
n_classes = 3

# True labels
y_true = np.random.randint(0, n_classes, n_samples)

# Simulated probabilistic predictions
y_pred_proba = np.random.rand(n_samples, n_classes)
y_pred_proba /= y_pred_proba.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 6))

for i in range(n_classes):
    prob_true, prob_pred = calibration_curve(y_true == i, y_pred_proba[:, i], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=f'Class {i}')

plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve for Multiclass Classification')
plt.legend()
plt.show()
```

Slide 11: Real-Life Example: Image Classification

Consider a multiclass image classification problem where a model is trained to classify images of different animal species. Accuracy alone might not capture the nuances of model performance, especially if some species are rare or visually similar.

```python
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Simulating results for an animal classification model
species = ['lion', 'tiger', 'leopard', 'cheetah']
y_true = np.random.choice(species, 1000, p=[0.4, 0.3, 0.2, 0.1])
y_pred = np.random.choice(species, 1000, p=[0.45, 0.25, 0.2, 0.1])

# Print classification report
print(classification_report(y_true, y_pred, target_names=species))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=species)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=species, yticklabels=species)
plt.title('Confusion Matrix for Animal Classification')
plt.xlabel('Predicted Species')
plt.ylabel('True Species')
plt.show()
```

Slide 12: Real-Life Example: Medical Diagnosis

In medical diagnosis, the cost of different types of errors can vary significantly. For instance, misclassifying a malignant tumor as benign (false negative) is generally more serious than the reverse (false positive).

```python
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Simulating a medical diagnosis scenario
conditions = ['Healthy', 'Condition A', 'Condition B', 'Condition C']
y_true = np.random.choice(conditions, 1000, p=[0.7, 0.1, 0.1, 0.1])
y_pred = np.random.choice(conditions, 1000, p=[0.75, 0.08, 0.09, 0.08])

# Compute and plot confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=conditions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=conditions, yticklabels=conditions)
plt.title('Confusion Matrix for Medical Diagnosis')
plt.xlabel('Predicted Condition')
plt.ylabel('True Condition')
plt.show()

# Print classification report
print(classification_report(y_true, y_pred, target_names=conditions))

# Note: In real medical scenarios, additional metrics like sensitivity and 
# specificity for each condition would be crucial for thorough evaluation.
```

Slide 13: Choosing the Right Metric for Your Problem

Selecting appropriate evaluation metrics depends on your specific problem and goals. Consider factors such as class imbalance, the cost of different types of errors, and the need for probabilistic outputs.

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, cohen_kappa_score

# Simulating a multiclass classification scenario
y_true = np.array([0, 1, 2, 2, 1, 0, 2, 1, 0, 2])
y_pred = np.array([0, 2, 2, 2, 0, 0, 2, 1, 0, 2])
y_pred_proba = np.array([
    [0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.1, 0.2, 0.7],
    [0.0, 0.1, 0.9], [0.6, 0.3, 0.1], [0.7, 0.2, 0.1],
    [0.0, 0.1, 0.9], [0.1, 0.8, 0.1], [0.9, 0.0, 0.1],
    [0.1, 0.1, 0.8]
])

# Calculate different metrics
accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro')
logloss = log_loss(y_true, y_pred_proba)
kappa = cohen_kappa_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"Log Loss: {logloss:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
```

Slide 14: Implementing Custom Metrics

Sometimes, standard metrics may not fully capture the nuances of your problem. In such cases, implementing custom metrics can provide valuable insights.

```python
import numpy as np

def weighted_accuracy(y_true, y_pred, weights):
    """
    Calculate weighted accuracy for multiclass classification.
    
    Args:
    y_true (array-like): True labels
    y_pred (array-like): Predicted labels
    weights (dict): Weight for each class
    
    Returns:
    float: Weighted accuracy score
    """
    correct = 0
    total_weight = 0
    
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += weights.get(true, 1)
        total_weight += weights.get(true, 1)
    
    return correct / total_weight if total_weight > 0 else 0

# Example usage
y_true = np.array([0, 1, 2, 2, 1, 0, 2, 1, 0, 2])
y_pred = np.array([0, 2, 2, 2, 0, 0, 2, 1, 0, 2])
class_weights = {0: 1, 1: 2, 2: 3}  # Class 2 is three times as important as class 0

w_accuracy = weighted_accuracy(y_true, y_pred, class_weights)
print(f"Weighted Accuracy: {w_accuracy:.4f}")
```

Slide 15: Visualizing Model Performance

Visualization can provide intuitive insights into model performance across multiple classes. Here's an example of creating a radar chart to compare different metrics for each class.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def radar_chart(metrics, class_names):
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    metrics += metrics[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    ax.plot(angles, metrics)
    ax.fill(angles, metrics, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1)
    plt.title("Multiclass Classification Metrics")
    plt.show()

# Example usage
y_true = np.array([0, 1, 2, 2, 1, 0, 2, 1, 0, 2])
y_pred = np.array([0, 2, 2, 2, 0, 0, 2, 1, 0, 2])
class_names = ['Class 0', 'Class 1', 'Class 2']

precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)
f1 = f1_score(y_true, y_pred, average=None)

metrics = [precision.mean(), recall.mean(), f1.mean()]
radar_chart(metrics, ['Precision', 'Recall', 'F1 Score'])
```

Slide 16: Additional Resources

For further exploration of multiclass classification metrics and evaluation techniques, consider the following resources:

1. "A Survey of Performance Metrics for Multi-Class Classification" by Sokolova, M., & Lapalme, G. (2009) ArXiv: [https://arxiv.org/abs/2008.05756](https://arxiv.org/abs/2008.05756)
2. "On the Importance of the Kullback-Leibler Divergence Term in Variational Autoencoders for Text Generation" by Prokhorov, V., et al. (2019) ArXiv: [https://arxiv.org/abs/1909.13668](https://arxiv.org/abs/1909.13668)
3. "Evaluation Metrics for Multilabel Classification: A Case Study" by Kavitha, K. S., et al. (2020) ArXiv: [https://arxiv.org/abs/2006.06902](https://arxiv.org/abs/2006.06902)

These papers provide in-depth discussions on various aspects of multiclass classification evaluation, including theoretical foundations and practical applications.

