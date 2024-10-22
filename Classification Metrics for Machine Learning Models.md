## Classification Metrics for Machine Learning Models

Slide 1: Understanding Classification Metrics

Machine learning classification models require robust evaluation metrics to assess their performance accurately and ensure reliable predictions. These metrics help quantify how well a model can generalize its learning from training data to new, unseen examples.

```python
def calculate_basic_metrics(y_true, y_pred):
    true_pos = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    true_neg = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    false_pos = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    false_neg = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    
    return true_pos, true_neg, false_pos, false_neg
```

Slide 2: Accuracy Metric

Accuracy represents the ratio of correct predictions to the total number of cases evaluated. While straightforward, this metric can be misleading when dealing with imbalanced datasets where one class significantly outnumbers others.

```python
def calculate_accuracy(y_true, y_pred):
    tp, tn, fp, fn = calculate_basic_metrics(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy

# Example usage
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1]
print(f"Accuracy: {calculate_accuracy(y_true, y_pred):.2f}")
```

Slide 3: Results for Accuracy Metric

```python
# Output
Accuracy: 0.83
```

Slide 4: Precision Metric

Precision measures the proportion of correct positive predictions among all positive predictions made. This metric is particularly useful when the cost of false positives is high, such as in medical diagnosis or spam detection systems.

```python
def calculate_precision(y_true, y_pred):
    tp, _, fp, _ = calculate_basic_metrics(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0

# Example usage
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 1, 1]
print(f"Precision: {calculate_precision(y_true, y_pred):.2f}")
```

Slide 5: Results for Precision Metric

```python
# Output
Precision: 0.75
```

Slide 6: Recall Metric

Recall, also known as sensitivity, measures the proportion of actual positive cases that were correctly identified. This metric is crucial in scenarios where missing positive cases can have serious consequences, such as disease detection or security threat identification.

```python
def calculate_recall(y_true, y_pred):
    tp, _, _, fn = calculate_basic_metrics(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0

# Example with medical diagnosis scenario
y_true = [1, 1, 0, 1, 1, 0]  # Actual patient conditions
y_pred = [1, 0, 0, 1, 1, 0]  # Predicted diagnoses
print(f"Recall: {calculate_recall(y_true, y_pred):.2f}")
```

Slide 7: F1 Score Implementation

The F1 score provides a balanced measure between precision and recall, making it particularly useful when you need to find an optimal balance between these two metrics. It is calculated as the harmonic mean of precision and recall.

```python
def calculate_f1_score(y_true, y_pred):
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    
    if precision + recall == 0:
        return 0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
```

Slide 8: Real-Life Example - Image Classification

Consider a computer vision system for identifying safety equipment in construction sites. The system needs to accurately detect whether workers are wearing proper safety gear.

```python
# Example of safety equipment detection results
safety_actual = [1, 1, 1, 0, 1, 1, 0, 1]  # 1: wearing, 0: not wearing
safety_predicted = [1, 1, 0, 0, 1, 1, 1, 1]

results = {
    'Accuracy': calculate_accuracy(safety_actual, safety_predicted),
    'Precision': calculate_precision(safety_actual, safety_predicted),
    'Recall': calculate_recall(safety_actual, safety_predicted),
    'F1': calculate_f1_score(safety_actual, safety_predicted)
}
```

Slide 9: Confusion Matrix

A confusion matrix provides a complete picture of model performance by showing true positives, true negatives, false positives, and false negatives in a structured format.

```python
def create_confusion_matrix(y_true, y_pred):
    tp, tn, fp, fn = calculate_basic_metrics(y_true, y_pred)
    matrix = {
        'True Positives': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn
    }
    return matrix
```

Slide 10: ROC Curve Implementation

The Receiver Operating Characteristic curve visualizes the trade-off between true positive rate and false positive rate across different classification thresholds.

```python
def calculate_roc_points(y_true, y_scores):
    thresholds = sorted(set(y_scores), reverse=True)
    roc_points = []
    
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        tp, tn, fp, fn = calculate_basic_metrics(y_true, y_pred)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        roc_points.append((fpr, tpr))
    
    return roc_points
```

Slide 11: Cross-Validation

Cross-validation helps assess model performance across different data splits, providing a more robust evaluation of model generalization.

```python
def k_fold_cross_validation(X, y, k=5):
    fold_size = len(X) // k
    scores = []
    
    for i in range(k):
        start = i * fold_size
        end = start + fold_size
        
        X_test = X[start:end]
        y_test = y[start:end]
        X_train = X[:start] + X[end:]
        y_train = y[:start] + y[end:]
        
        # Train and evaluate model
        fold_score = train_and_evaluate(X_train, y_train, X_test, y_test)
        scores.append(fold_score)
    
    return sum(scores) / len(scores)
```

Slide 12: Real-Life Example - Document Classification

A system for automatically categorizing scientific papers into different research fields, demonstrating the application of multiple metrics.

```python
# Example of document classification results
papers_actual = [1, 2, 1, 3, 2, 1, 3, 2]  # Research fields
papers_predicted = [1, 2, 1, 2, 2, 1, 3, 3]

# Calculate multi-class metrics
accuracy = calculate_accuracy(papers_actual, papers_predicted)
confusion = create_confusion_matrix(papers_actual, papers_predicted)
```

Slide 13: Additional Resources

ArXiv papers for deeper understanding of classification metrics:

*   "A Survey of Performance Metrics for Classification Algorithms" - arXiv:2008.05756
*   "Beyond Accuracy: Precision and Recall" - arXiv:1905.07387
*   "ROC Analysis in Machine Learning" - arXiv:2103.04655

