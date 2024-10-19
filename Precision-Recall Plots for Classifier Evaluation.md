## Precision-Recall Plots for Classifier Evaluation

Slide 1: Introduction to Precision-Recall Plots

Precision-Recall plots are powerful tools for evaluating the performance of classification models. They provide insights into the trade-off between precision and recall, helping data scientists and machine learning engineers to choose the best model for their specific use case. This presentation will explore the concept, implementation, and interpretation of Precision-Recall plots using Python.

```python
import matplotlib.pyplot as plt

def plot_precision_recall_curve():
    precision = [1.0, 0.8, 0.6, 0.4, 0.2]
    recall = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

plot_precision_recall_curve()
```

Slide 2: Understanding Precision and Recall

Precision and recall are fundamental metrics in classification problems. Precision measures the accuracy of positive predictions, while recall measures the ability to find all positive instances. Let's define these metrics mathematically:

Precision = TruePositivesTruePositives+FalsePositives\\frac{True Positives}{True Positives + False Positives}TruePositives+FalsePositivesTruePositives​

Recall = TruePositivesTruePositives+FalseNegatives\\frac{True Positives}{True Positives + False Negatives}TruePositives+FalseNegativesTruePositives​

```python
def calculate_precision_recall(y_true, y_pred):
    true_positives = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    false_positives = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    false_negatives = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    return precision, recall

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 1, 1, 0, 0, 1]

precision, recall = calculate_precision_recall(y_true, y_pred)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")
```

Slide 3: Generating Precision-Recall Curves

To create a Precision-Recall curve, we need to calculate precision and recall at various classification thresholds. This process involves sorting the predicted probabilities and computing precision and recall at each point.

```python
import random

def generate_precision_recall_curve(y_true, y_scores, num_thresholds=100):
    thresholds = sorted(random.sample(y_scores, num_thresholds))
    precision_values = []
    recall_values = []
    
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        precision, recall = calculate_precision_recall(y_true, y_pred)
        precision_values.append(precision)
        recall_values.append(recall)
    
    return precision_values, recall_values

# Generate sample data
y_true = [random.choice([0, 1]) for _ in range(1000)]
y_scores = [random.random() for _ in range(1000)]

precision_values, recall_values = generate_precision_recall_curve(y_true, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall_values, precision_values)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()
```

Slide 4: Interpreting Precision-Recall Curves

Precision-Recall curves provide valuable insights into model performance. A perfect classifier would have a curve that reaches the top-right corner (precision = 1, recall = 1). The area under the curve (AUC) is a single-number metric that summarizes the curve's performance. Higher AUC values indicate better overall performance.

```python
def calculate_auc(recall_values, precision_values):
    auc = 0
    for i in range(1, len(recall_values)):
        auc += (recall_values[i] - recall_values[i-1]) * (precision_values[i] + precision_values[i-1]) / 2
    return auc

auc = calculate_auc(recall_values, precision_values)
print(f"Area Under the Curve (AUC): {auc:.3f}")

plt.figure(figsize=(8, 6))
plt.plot(recall_values, precision_values)
plt.fill_between(recall_values, precision_values, alpha=0.2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AUC: {auc:.3f})')
plt.grid(True)
plt.show()
```

Slide 5: Comparison with Random Classifier

A random classifier's performance is represented by a horizontal line on the Precision-Recall plot. The position of this line depends on the class balance in the dataset. For a balanced dataset (equal number of positive and negative samples), the random classifier line would be at y = 0.5.

```python
def plot_random_classifier(y_true):
    positive_ratio = sum(y_true) / len(y_true)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_values, precision_values, label='Model')
    plt.axhline(y=positive_ratio, color='r', linestyle='--', label='Random Classifier')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve vs Random Classifier')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_random_classifier(y_true)
```

Slide 6: Real-Life Example: Email Spam Detection

Let's consider a real-life example of using Precision-Recall curves for email spam detection. In this scenario, we want to balance between correctly identifying spam (high precision) and not missing any important emails (high recall).

```python
import random

def simulate_spam_detection(num_emails=1000):
    # Simulate email classification (0: not spam, 1: spam)
    y_true = [random.choice([0, 1]) for _ in range(num_emails)]
    
    # Simulate model scores (higher score means more likely to be spam)
    y_scores = [random.uniform(0, 1) for _ in range(num_emails)]
    
    precision_values, recall_values = generate_precision_recall_curve(y_true, y_scores)
    auc = calculate_auc(recall_values, precision_values)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_values, precision_values)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for Spam Detection (AUC: {auc:.3f})')
    plt.grid(True)
    plt.show()

simulate_spam_detection()
```

Slide 7: Choosing the Optimal Threshold

In the spam detection example, we need to choose a threshold that balances precision and recall. One common approach is to use the F1 score, which is the harmonic mean of precision and recall. Let's implement a function to find the optimal threshold based on the F1 score.

```python
def find_optimal_threshold(y_true, y_scores):
    thresholds = sorted(set(y_scores))
    best_f1 = 0
    best_threshold = 0
    
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        precision, recall = calculate_precision_recall(y_true, y_pred)
        if precision + recall == 0:
            continue
        f1 = 2 * (precision * recall) / (precision + recall)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

# Using the previously simulated spam detection data
best_threshold, best_f1 = find_optimal_threshold(y_true, y_scores)
print(f"Optimal Threshold: {best_threshold:.3f}")
print(f"Best F1 Score: {best_f1:.3f}")
```

Slide 8: Real-Life Example: Medical Diagnosis

Another important application of Precision-Recall curves is in medical diagnosis. Let's consider a scenario where we're developing a model to detect a rare disease. In this case, high recall is crucial to avoid missing any positive cases.

```python
def simulate_medical_diagnosis(num_patients=10000, disease_prevalence=0.01):
    # Simulate patient diagnosis (0: healthy, 1: disease)
    y_true = [1 if random.random() < disease_prevalence else 0 for _ in range(num_patients)]
    
    # Simulate model scores (higher score means higher likelihood of disease)
    y_scores = [random.betavariate(2, 5) if label == 1 else random.betavariate(1, 3) for label in y_true]
    
    precision_values, recall_values = generate_precision_recall_curve(y_true, y_scores)
    auc = calculate_auc(recall_values, precision_values)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_values, precision_values)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for Disease Detection (AUC: {auc:.3f})')
    plt.grid(True)
    plt.show()

simulate_medical_diagnosis()
```

Slide 9: Handling Imbalanced Datasets

In many real-world scenarios, such as fraud detection or rare disease diagnosis, we deal with imbalanced datasets where one class is much more prevalent than the other. Precision-Recall curves are particularly useful in these cases, as they focus on the performance of the positive class.

```python
def compare_balanced_imbalanced():
    # Balanced dataset
    y_true_balanced = [random.choice([0, 1]) for _ in range(1000)]
    y_scores_balanced = [random.random() for _ in range(1000)]
    
    # Imbalanced dataset (1% positive class)
    y_true_imbalanced = [1 if random.random() < 0.01 else 0 for _ in range(1000)]
    y_scores_imbalanced = [random.betavariate(2, 5) if label == 1 else random.betavariate(1, 3) for label in y_true_imbalanced]
    
    precision_balanced, recall_balanced = generate_precision_recall_curve(y_true_balanced, y_scores_balanced)
    precision_imbalanced, recall_imbalanced = generate_precision_recall_curve(y_true_imbalanced, y_scores_imbalanced)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall_balanced, precision_balanced, label='Balanced Dataset')
    plt.plot(recall_imbalanced, precision_imbalanced, label='Imbalanced Dataset')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves: Balanced vs Imbalanced Datasets')
    plt.legend()
    plt.grid(True)
    plt.show()

compare_balanced_imbalanced()
```

Slide 10: Precision-Recall vs ROC Curves

While both Precision-Recall and Receiver Operating Characteristic (ROC) curves are used to evaluate classifier performance, Precision-Recall curves are often preferred for imbalanced datasets. Let's compare these two evaluation methods.

```python
from sklearn.metrics import precision_recall_curve, roc_curve

def compare_pr_roc(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(recall, precision)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.grid(True)
    
    ax2.plot(fpr, tpr)
    ax2.plot([0, 1], [0, 1], linestyle='--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Using previously generated imbalanced dataset
compare_pr_roc(y_true_imbalanced, y_scores_imbalanced)
```

Slide 11: Implementing Precision-Recall Curve from Scratch

Let's implement a Precision-Recall curve from scratch using only built-in Python functions. This will help us understand the underlying mechanics of the curve.

```python
def precision_recall_curve_scratch(y_true, y_scores):
    # Sort scores and corresponding true values
    sorted_data = sorted(zip(y_scores, y_true), reverse=True)
    sorted_scores, sorted_true = zip(*sorted_data)
    
    precision_values = []
    recall_values = []
    true_positives = 0
    false_positives = 0
    total_positives = sum(y_true)
    
    for i, (score, label) in enumerate(sorted_data):
        if label == 1:
            true_positives += 1
        else:
            false_positives += 1
        
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / total_positives
        
        precision_values.append(precision)
        recall_values.append(recall)
    
    return precision_values, recall_values

# Generate sample data
y_true = [random.choice([0, 1]) for _ in range(1000)]
y_scores = [random.random() for _ in range(1000)]

precision_values, recall_values = precision_recall_curve_scratch(y_true, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall_values, precision_values)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (From Scratch)')
plt.grid(True)
plt.show()
```

Slide 12: Visualizing Decision Boundaries

To better understand how the Precision-Recall curve relates to the model's decision-making process, let's visualize the decision boundaries for a simple 2D dataset.

```python
import random
import matplotlib.pyplot as plt

def generate_2d_data(n_samples=1000):
    X = [(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(n_samples)]
    y = [1 if x**2 + y**2 <= 9 else 0 for x, y in X]
    return X, y

def plot_decision_boundary(X, y, threshold):
    plt.figure(figsize=(8, 6))
    plt.scatter([x for x, _ in X], [y for _, y in X], c=y, cmap='coolwarm', alpha=0.5)
    
    circle = plt.Circle((0, 0), 3, fill=False, color='black', linestyle='--')
    plt.gca().add_artist(circle)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Decision Boundary (Threshold: {threshold:.2f})')
    plt.grid(True)
    plt.show()

X, y = generate_2d_data()
plot_decision_boundary(X, y, threshold=0.5)
```

Slide 13: Impact of Threshold on Precision and Recall

Let's examine how different threshold values affect precision and recall in our 2D dataset example.

```python
def calculate_metrics(y_true, y_pred):
    true_positives = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    false_positives = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    false_negatives = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    
    return precision, recall

def evaluate_thresholds(X, y, thresholds):
    results = []
    for threshold in thresholds:
        y_pred = [1 if x**2 + y**2 <= threshold**2 * 9 else 0 for x, y in X]
        precision, recall = calculate_metrics(y, y_pred)
        results.append((threshold, precision, recall))
    return results

thresholds = [0.8, 1.0, 1.2]
results = evaluate_thresholds(X, y, thresholds)

for threshold, precision, recall in results:
    print(f"Threshold: {threshold:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
```

Slide 14: Precision-Recall Curve for 2D Dataset

Now, let's create a Precision-Recall curve for our 2D dataset to visualize the trade-off between precision and recall.

```python
def precision_recall_curve_2d(X, y, num_thresholds=100):
    thresholds = [i / num_thresholds * 2 for i in range(1, num_thresholds + 1)]
    results = evaluate_thresholds(X, y, thresholds)
    
    precision_values = [p for _, p, _ in results]
    recall_values = [r for _, _, r in results]
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_values, precision_values)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for 2D Dataset')
    plt.grid(True)
    plt.show()

precision_recall_curve_2d(X, y)
```

Slide 15: Additional Resources

For those interested in diving deeper into Precision-Recall curves and related topics, here are some valuable resources:

1.  "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets" by Saito and Rehmsmeier (2015). Available on ArXiv: [https://arxiv.org/abs/1502.05803](https://arxiv.org/abs/1502.05803)
2.  "An introduction to ROC analysis" by Fawcett (2006). This paper provides a comprehensive overview of ROC analysis and touches on Precision-Recall curves. While not available on ArXiv, it can be found in many academic databases.
3.  "Classification Evaluation Metrics" in the scikit-learn documentation: [https://scikit-learn.org/stable/modules/model\_evaluation.html#classification-metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

These resources offer in-depth explanations and advanced techniques for working with Precision-Recall curves and related evaluation metrics in machine learning.

