## Evaluation Metrics Storytelling for Your Model
Slide 1: Understanding Evaluation Metrics

Evaluation metrics are crucial tools in machine learning that help us assess model performance. They provide insights into how well our models are performing and guide us in making improvements. Let's explore some key metrics and their implications.

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Example predictions and true labels
y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

# Calculate basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
```

Slide 2: Accuracy - The Basic Metric

Accuracy is the ratio of correct predictions to total predictions. While simple, it can be misleading for imbalanced datasets.

```python
def calculate_accuracy(y_true, y_pred):
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total = len(y_true)
    return correct / total

# Example usage
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1]

accuracy = calculate_accuracy(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

Slide 3: Precision - When False Positives Matter

Precision is the ratio of true positives to all positive predictions. It's crucial when the cost of false positives is high.

```python
def calculate_precision(y_true, y_pred):
    true_positives = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    predicted_positives = sum(p == 1 for p in y_pred)
    return true_positives / predicted_positives if predicted_positives > 0 else 0

# Example usage
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 1, 1]

precision = calculate_precision(y_true, y_pred)
print(f"Precision: {precision:.2f}")
```

Slide 4: Recall - When False Negatives Matter

Recall is the ratio of true positives to all actual positive instances. It's important when missing positive instances is costly.

```python
def calculate_recall(y_true, y_pred):
    true_positives = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    actual_positives = sum(t == 1 for t in y_true)
    return true_positives / actual_positives if actual_positives > 0 else 0

# Example usage
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1]

recall = calculate_recall(y_true, y_pred)
print(f"Recall: {recall:.2f}")
```

Slide 5: F1 Score - Balancing Precision and Recall

The F1 score is the harmonic mean of precision and recall, providing a balanced measure when you need to find an optimal balance between precision and recall.

```python
def calculate_f1(y_true, y_pred):
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Example usage
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 1, 1]

f1 = calculate_f1(y_true, y_pred)
print(f"F1 Score: {f1:.2f}")
```

Slide 6: Specificity - True Negative Rate

Specificity measures the proportion of actual negatives that are correctly identified. It's particularly useful in medical diagnoses and fraud detection.

```python
def calculate_specificity(y_true, y_pred):
    true_negatives = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    actual_negatives = sum(t == 0 for t in y_true)
    return true_negatives / actual_negatives if actual_negatives > 0 else 0

# Example usage
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 1, 1]

specificity = calculate_specificity(y_true, y_pred)
print(f"Specificity: {specificity:.2f}")
```

Slide 7: ROC Curve and AUC - Trading Off True and False Positives

The Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) help visualize and quantify the trade-off between true positive rate and false positive rate.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Generate sample data
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.65, 0.2, 0.9, 0.5, 0.6, 0.7])

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

Slide 8: Confusion Matrix - A Comprehensive View

The confusion matrix provides a tabular summary of a classifier's performance, showing true positives, false positives, true negatives, and false negatives.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Example usage
y_true = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 1, 1, 0]

plot_confusion_matrix(y_true, y_pred)
```

Slide 9: Mean Squared Error (MSE) - For Regression Problems

MSE is a common metric for regression tasks, measuring the average squared difference between predicted and actual values.

```python
import numpy as np

def calculate_mse(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred))**2)

# Example usage
y_true = [3, 2, 5, 1, 7]
y_pred = [2.5, 3.0, 5.0, 2.0, 7.5]

mse = calculate_mse(y_true, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
```

Slide 10: R-squared (R²) - Goodness of Fit

R-squared indicates the proportion of variance in the dependent variable that's predictable from the independent variable(s).

```python
import numpy as np

def calculate_r_squared(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - (ss_residual / ss_total)

# Example usage
y_true = [3, 2, 5, 1, 7]
y_pred = [2.5, 3.0, 5.0, 2.0, 7.5]

r_squared = calculate_r_squared(y_true, y_pred)
print(f"R-squared: {r_squared:.2f}")
```

Slide 11: Cross-Entropy Loss - For Classification Problems

Cross-entropy loss measures the performance of a classification model whose output is a probability value between 0 and 1.

```python
import numpy as np

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example usage
y_true = [1, 0, 1, 1, 0]
y_pred = [0.9, 0.1, 0.8, 0.7, 0.2]

loss = cross_entropy_loss(y_true, y_pred)
print(f"Cross-Entropy Loss: {loss:.4f}")
```

Slide 12: Real-Life Example: Medical Diagnosis

In medical diagnosis, different metrics have varying importance. For a cancer screening test, high recall is crucial to avoid missing positive cases, while maintaining good precision to minimize unnecessary anxiety and follow-up procedures.

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Simulated results of a cancer screening test
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0])

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
```

Slide 13: Real-Life Example: Content Recommendation System

In a content recommendation system, metrics like precision at k (P@k) and mean average precision (MAP) are crucial. These metrics focus on the relevance of the top-k recommended items.

```python
import numpy as np

def precision_at_k(y_true, y_pred, k):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Sort predictions in descending order and get top k
    top_k = y_pred.argsort()[-k:][::-1]
    
    # Calculate precision
    return np.mean(y_true[top_k])

# Example: Recommending 5 items to a user
y_true = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0]  # 1 indicates relevant item
y_pred = [0.9, 0.6, 0.8, 0.3, 0.7, 0.9, 0.2, 0.6, 0.1, 0.4]  # Predicted relevance scores

p_at_5 = precision_at_k(y_true, y_pred, k=5)
print(f"Precision@5: {p_at_5:.2f}")
```

Slide 14: Choosing the Right Metric

Selecting appropriate metrics depends on your problem's nature and goals. Consider the following factors:

1. Problem type (classification, regression, ranking)
2. Dataset characteristics (balanced or imbalanced)
3. Business objectives and costs of different types of errors
4. Interpretability requirements

Remember, no single metric tells the whole story. Use a combination of metrics for a comprehensive evaluation of your model's performance.

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred, problem_type='classification'):
    if problem_type == 'classification':
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print("Classification Metrics:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
    
    elif problem_type == 'regression':
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print("Regression Metrics:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared: {r2:.2f}")
    
    else:
        print("Unsupported problem type")

# Example usage
y_true_class = [1, 0, 1, 1, 0, 1, 0, 1]
y_pred_class = [1, 0, 1, 0, 0, 1, 1, 1]

y_true_reg = [3.2, 2.1, 5.7, 1.3, 7.6]
y_pred_reg = [3.0, 2.5, 5.5, 1.8, 7.2]

evaluate_model(y_true_class, y_pred_class, 'classification')
print("\n")
evaluate_model(y_true_reg, y_pred_reg, 'regression')
```

Slide 15: Additional Resources

For further exploration of evaluation metrics and their applications, consider the following resources:

1. "A Survey of Predictive Modelling under Imbalanced Distributions" by Branco et al. (2016) ArXiv: [https://arxiv.org/abs/1505.01658](https://arxiv.org/abs/1505.01658)
2. "Beyond Accuracy: Precision and Recall" by Powers (2011) ArXiv: [https://arxiv.org/abs/2101.08344](https://arxiv.org/abs/2101.08344)
3. "The Relationship Between Precision-Recall and ROC Curves" by Davis and Goadrich (2006) ArXiv: [https://arxiv.org/abs/cs/0606118](https://arxiv.org/abs/cs/0606118)
4. "An Introduction to ROC Analysis" by Fawcett (2006) Available at: [https://people.inf.elte.hu/kiss/13dwhdm/roc.pdf](https://people.inf.elte.hu/kiss/13dwhdm/roc.pdf)
5. "A Systematic Analysis of Performance Measures for Classification Tasks" by Sokolova and Lapalme (2009) Available through most academic libraries

These resources provide in-depth discussions on various evaluation metrics, their relationships, and applications in different scenarios. They can help deepen your understanding of when and how to use different metrics effectively in your machine learning projects.


