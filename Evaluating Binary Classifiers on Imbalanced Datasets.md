## Evaluating Binary Classifiers on Imbalanced Datasets

Slide 1: Understanding Classifier Evaluation Metrics

Evaluating binary classifiers is crucial for assessing model performance. Common metrics include AUC-ROC (Area Under the Receiver Operating Characteristic curve) and Gini coefficient. However, these metrics can be misleading when dealing with imbalanced datasets. In such cases, Precision-Recall curves and AUC-PR (Area Under the Precision-Recall curve) often provide a more accurate reflection of a model's performance.

```python
import random

def generate_imbalanced_dataset(size, imbalance_ratio):
    positive = int(size * imbalance_ratio)
    negative = size - positive
    data = [1] * positive + [0] * negative
    random.shuffle(data)
    return data

# Generate an imbalanced dataset
dataset = generate_imbalanced_dataset(1000, 0.05)
print(f"Dataset size: {len(dataset)}")
print(f"Positive samples: {sum(dataset)}")
print(f"Negative samples: {len(dataset) - sum(dataset)}")
```

Slide 2: AUC-ROC and Gini Coefficient

The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various classification thresholds. The AUC-ROC represents the area under this curve, while the Gini coefficient is derived from it. These metrics are widely used but can be misleading for imbalanced datasets because they consider True Negatives, which can dominate in such scenarios.

```python
import random

def calculate_tpr_fpr(y_true, y_pred, threshold):
    tp = fp = tn = fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred >= threshold:
            tp += 1
        elif true == 0 and pred >= threshold:
            fp += 1
        elif true == 0 and pred < threshold:
            tn += 1
        else:
            fn += 1
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr

# Generate sample data
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [0.9, 0.1, 0.8, 0.7, 0.3, 0.6, 0.2, 0.1, 0.7, 0.4]

# Calculate TPR and FPR for different thresholds
thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
for threshold in thresholds:
    tpr, fpr = calculate_tpr_fpr(y_true, y_pred, threshold)
    print(f"Threshold: {threshold:.1f}, TPR: {tpr:.2f}, FPR: {fpr:.2f}")
```

Slide 3: Limitations of AUC-ROC for Imbalanced Data

In imbalanced datasets, where one class significantly outnumbers the other, AUC-ROC can present an overly optimistic view of a model's performance. This is because it gives equal weight to both classes, potentially masking poor performance on the minority class. A model might achieve a high AUC-ROC by performing well on the majority class while struggling with the minority class.

```python
import random

def generate_predictions(y_true, bias=0.7):
    return [random.random() * bias if y == 1 else random.random() * (1 - bias) for y in y_true]

# Generate imbalanced dataset
y_true = generate_imbalanced_dataset(1000, 0.05)

# Generate biased predictions
y_pred = generate_predictions(y_true)

# Calculate AUC-ROC (simplified version)
def calculate_auc_roc(y_true, y_pred):
    positives = [p for t, p in zip(y_true, y_pred) if t == 1]
    negatives = [p for t, p in zip(y_true, y_pred) if t == 0]
    auc = sum(p > n for p in positives for n in negatives)
    return auc / (len(positives) * len(negatives))

auc_roc = calculate_auc_roc(y_true, y_pred)
print(f"AUC-ROC: {auc_roc:.4f}")
```

Slide 4: Introduction to Precision-Recall Curves

Precision-Recall curves focus on the positive class, making them more suitable for imbalanced datasets. Precision measures the proportion of correct positive predictions, while Recall (also known as sensitivity or TPR) measures the proportion of actual positives correctly identified. These metrics provide a clearer picture of a model's performance on the minority class.

```python
def calculate_precision_recall(y_true, y_pred, threshold):
    tp = fp = fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred >= threshold:
            tp += 1
        elif true == 0 and pred >= threshold:
            fp += 1
        elif true == 1 and pred < threshold:
            fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall

# Using the same y_true and y_pred from previous slides
thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
for threshold in thresholds:
    precision, recall = calculate_precision_recall(y_true, y_pred, threshold)
    print(f"Threshold: {threshold:.1f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
```

Slide 5: AUC-PR (Area Under the Precision-Recall Curve)

AUC-PR represents the area under the Precision-Recall curve. It provides a single scalar value to compare different models, similar to AUC-ROC. However, AUC-PR is more informative for imbalanced datasets as it focuses on the positive class and is not affected by the large number of true negatives typically present in such scenarios.

```python
def calculate_auc_pr(y_true, y_pred):
    # Sort predictions in descending order
    sorted_data = sorted(zip(y_pred, y_true), reverse=True)
    y_true = [y for _, y in sorted_data]
    
    precision_values = []
    recall_values = []
    true_positives = 0
    false_positives = 0
    
    for i, y in enumerate(y_true):
        if y == 1:
            true_positives += 1
        else:
            false_positives += 1
        
        precision = true_positives / (i + 1)
        recall = true_positives / sum(y_true)
        
        precision_values.append(precision)
        recall_values.append(recall)
    
    # Calculate AUC using trapezoidal rule
    auc_pr = 0
    for i in range(1, len(recall_values)):
        auc_pr += (recall_values[i] - recall_values[i-1]) * (precision_values[i] + precision_values[i-1]) / 2
    
    return auc_pr

# Using the same y_true and y_pred from previous slides
auc_pr = calculate_auc_pr(y_true, y_pred)
print(f"AUC-PR: {auc_pr:.4f}")
```

Slide 6: Comparing AUC-ROC and AUC-PR

To illustrate the difference between AUC-ROC and AUC-PR, let's compare these metrics on a highly imbalanced dataset. We'll create a dataset with a 1:99 ratio of positive to negative samples and evaluate a simple classifier using both metrics.

```python
import random

def generate_imbalanced_dataset(size, imbalance_ratio):
    positive = int(size * imbalance_ratio)
    negative = size - positive
    data = [1] * positive + [0] * negative
    random.shuffle(data)
    return data

def simple_classifier(x, threshold):
    return 1 if x >= threshold else 0

# Generate imbalanced dataset
dataset_size = 10000
imbalance_ratio = 0.01
y_true = generate_imbalanced_dataset(dataset_size, imbalance_ratio)

# Generate predictions
y_pred = [random.random() for _ in range(dataset_size)]

# Calculate AUC-ROC and AUC-PR
auc_roc = calculate_auc_roc(y_true, y_pred)
auc_pr = calculate_auc_pr(y_true, y_pred)

print(f"Dataset size: {dataset_size}")
print(f"Positive samples: {sum(y_true)}")
print(f"Negative samples: {dataset_size - sum(y_true)}")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"AUC-PR: {auc_pr:.4f}")
```

Slide 7: Interpreting AUC-ROC vs AUC-PR Results

The results from the previous slide demonstrate why AUC-PR is more informative for imbalanced datasets. While AUC-ROC might show a relatively high value due to the large number of true negatives, AUC-PR provides a more realistic assessment of the model's performance on the minority class. This is particularly important in scenarios where correctly identifying the positive class is crucial, such as in medical diagnosis or fraud detection.

```python
def interpret_results(auc_roc, auc_pr, imbalance_ratio):
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    if auc_roc > 0.8 and auc_pr < 0.3:
        print("AUC-ROC suggests good performance, but AUC-PR reveals struggles with the minority class.")
    elif auc_roc > 0.8 and auc_pr > 0.7:
        print("Both metrics indicate good performance, even on the imbalanced dataset.")
    elif auc_roc < 0.6 and auc_pr < 0.2:
        print("Both metrics suggest poor performance, particularly on the minority class.")
    else:
        print("Results are inconclusive. Further investigation is needed.")

# Using the results from the previous slide
interpret_results(auc_roc, auc_pr, imbalance_ratio)
```

Slide 8: Real-life Example: Rare Disease Detection

Consider a scenario where we're developing a model to detect a rare disease that affects only 1% of the population. We'll compare the performance of two models using both AUC-ROC and AUC-PR to demonstrate the importance of using appropriate metrics for imbalanced datasets.

```python
import random

def generate_predictions(y_true, accuracy):
    return [random.random() if random.random() > accuracy else y for y in y_true]

# Generate dataset
dataset_size = 10000
y_true = generate_imbalanced_dataset(dataset_size, 0.01)

# Generate predictions for two models
y_pred_model1 = generate_predictions(y_true, 0.8)
y_pred_model2 = generate_predictions(y_true, 0.9)

# Calculate metrics for both models
auc_roc_model1 = calculate_auc_roc(y_true, y_pred_model1)
auc_pr_model1 = calculate_auc_pr(y_true, y_pred_model1)
auc_roc_model2 = calculate_auc_roc(y_true, y_pred_model2)
auc_pr_model2 = calculate_auc_pr(y_true, y_pred_model2)

print("Model 1:")
print(f"AUC-ROC: {auc_roc_model1:.4f}")
print(f"AUC-PR: {auc_pr_model1:.4f}")
print("\nModel 2:")
print(f"AUC-ROC: {auc_roc_model2:.4f}")
print(f"AUC-PR: {auc_pr_model2:.4f}")
```

Slide 9: Analyzing the Rare Disease Detection Results

The results from the previous slide highlight the importance of using AUC-PR for imbalanced datasets. While both models might show similar and high AUC-ROC scores due to their ability to correctly identify healthy individuals (true negatives), the AUC-PR scores reveal a significant difference in their ability to detect the rare disease (true positives). This distinction is crucial in medical scenarios where false negatives can have severe consequences.

```python
def analyze_disease_detection_results(auc_roc1, auc_pr1, auc_roc2, auc_pr2):
    print(f"Model 1 - AUC-ROC: {auc_roc1:.4f}, AUC-PR: {auc_pr1:.4f}")
    print(f"Model 2 - AUC-ROC: {auc_roc2:.4f}, AUC-PR: {auc_pr2:.4f}")
    
    roc_diff = abs(auc_roc1 - auc_roc2)
    pr_diff = abs(auc_pr1 - auc_pr2)
    
    if roc_diff < 0.05 and pr_diff > 0.1:
        print("AUC-ROC suggests similar performance, but AUC-PR reveals a significant difference in rare disease detection ability.")
    elif pr_diff > roc_diff:
        print("AUC-PR provides a more sensitive measure of performance difference for this imbalanced dataset.")
    else:
        print("Both metrics show consistent differences. Further investigation may be needed.")

# Using the results from the previous slide
analyze_disease_detection_results(auc_roc_model1, auc_pr_model1, auc_roc_model2, auc_pr_model2)
```

Slide 10: Real-life Example: Anomaly Detection in Manufacturing

Consider a manufacturing process where defects occur in only 0.5% of products. We'll simulate two anomaly detection models and evaluate their performance using both AUC-ROC and AUC-PR to further illustrate the importance of appropriate metrics for imbalanced datasets.

```python
import random

def generate_anomaly_predictions(y_true, false_positive_rate, false_negative_rate):
    return [
        random.random() > false_negative_rate if y == 1
        else random.random() < false_positive_rate
        for y in y_true
    ]

dataset_size = 20000
y_true = [1 if random.random() < 0.005 else 0 for _ in range(dataset_size)]

y_pred_model1 = generate_anomaly_predictions(y_true, 0.1, 0.2)
y_pred_model2 = generate_anomaly_predictions(y_true, 0.01, 0.3)

auc_roc_model1 = calculate_auc_roc(y_true, y_pred_model1)
auc_pr_model1 = calculate_auc_pr(y_true, y_pred_model1)
auc_roc_model2 = calculate_auc_roc(y_true, y_pred_model2)
auc_pr_model2 = calculate_auc_pr(y_true, y_pred_model2)

print(f"Model 1 - AUC-ROC: {auc_roc_model1:.4f}, AUC-PR: {auc_pr_model1:.4f}")
print(f"Model 2 - AUC-ROC: {auc_roc_model2:.4f}, AUC-PR: {auc_pr_model2:.4f}")
```

Slide 11: Analyzing Anomaly Detection Results

The results from the previous slide demonstrate the importance of using AUC-PR for imbalanced datasets in manufacturing anomaly detection. Let's analyze these results to understand the implications for model selection and optimization.

```python
def analyze_anomaly_detection_results(auc_roc1, auc_pr1, auc_roc2, auc_pr2):
    print(f"Model 1 - AUC-ROC: {auc_roc1:.4f}, AUC-PR: {auc_pr1:.4f}")
    print(f"Model 2 - AUC-ROC: {auc_roc2:.4f}, AUC-PR: {auc_pr2:.4f}")
    
    if auc_roc1 > auc_roc2 and auc_pr1 < auc_pr2:
        print("AUC-ROC and AUC-PR suggest different best models.")
        print("AUC-PR is more relevant for this imbalanced dataset.")
    elif auc_pr1 > auc_pr2:
        print("Model 1 performs better for defect detection.")
    else:
        print("Model 2 performs better for defect detection.")

analyze_anomaly_detection_results(auc_roc_model1, auc_pr_model1, auc_roc_model2, auc_pr_model2)
```

Slide 12: Implementing AUC-PR from Scratch

To better understand AUC-PR, let's implement it from scratch using only Python's built-in functions. This implementation will help illustrate the underlying concepts and calculations involved in computing the AUC-PR metric.

```python
def auc_pr_from_scratch(y_true, y_pred):
    # Sort predictions in descending order
    paired_scores = sorted(zip(y_pred, y_true), reverse=True)
    y_true_sorted = [y for _, y in paired_scores]
    
    precision_values = []
    recall_values = []
    true_positives = 0
    false_positives = 0
    
    for i, y in enumerate(y_true_sorted, 1):
        if y == 1:
            true_positives += 1
        else:
            false_positives += 1
        
        precision = true_positives / i
        recall = true_positives / sum(y_true)
        
        precision_values.append(precision)
        recall_values.append(recall)
    
    # Calculate AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(recall_values)):
        auc += (recall_values[i] - recall_values[i-1]) * (precision_values[i] + precision_values[i-1]) / 2
    
    return auc

# Test the implementation
y_true_test = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred_test = [0.9, 0.1, 0.8, 0.7, 0.3, 0.6, 0.2, 0.1, 0.7, 0.4]

auc_pr = auc_pr_from_scratch(y_true_test, y_pred_test)
print(f"AUC-PR: {auc_pr:.4f}")
```

Slide 13: Visualizing Precision-Recall Curves

To better understand the relationship between Precision and Recall, let's create a simple visualization of the Precision-Recall curve. This will help illustrate how the curve behaves for different classification thresholds and how it relates to the AUC-PR metric.

```python
def calculate_precision_recall_points(y_true, y_pred):
    thresholds = sorted(set(y_pred), reverse=True)
    points = []
    for threshold in thresholds:
        y_pred_binary = [1 if p >= threshold else 0 for p in y_pred]
        tp = sum(1 for t, p in zip(y_true, y_pred_binary) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred_binary) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred_binary) if t == 1 and p == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        points.append((recall, precision))
    
    return points

# Generate sample data
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [0.9, 0.1, 0.8, 0.7, 0.3, 0.6, 0.2, 0.1, 0.7, 0.4]

# Calculate Precision-Recall points
pr_points = calculate_precision_recall_points(y_true, y_pred)

# Print points for visualization
print("Recall, Precision")
for recall, precision in pr_points:
    print(f"{recall:.2f}, {precision:.2f}")

# Note: In a real implementation, you would use a plotting library
# to create a visual representation of these points
```

Slide 14: Choosing Between AUC-ROC and AUC-PR

When deciding between AUC-ROC and AUC-PR for evaluating binary classifiers, consider the following factors:

1.  Class imbalance: AUC-PR is generally more informative for imbalanced datasets.
2.  Importance of the positive class: If correctly identifying positive instances is crucial, AUC-PR provides a better assessment.
3.  Cost of false positives vs. false negatives: AUC-ROC might be preferred if both types of errors are equally important.

```python
def recommend_metric(positive_ratio, importance_of_positives, equal_error_cost):
    if positive_ratio < 0.1:
        if importance_of_positives == "high" or not equal_error_cost:
            return "AUC-PR"
        else:
            return "Consider both AUC-PR and AUC-ROC"
    elif positive_ratio < 0.3:
        if importance_of_positives == "high":
            return "AUC-PR"
        else:
            return "Consider both AUC-PR and AUC-ROC"
    else:
        return "AUC-ROC (but consider AUC-PR if positive class is very important)"

# Example usage
dataset_characteristics = [
    (0.05, "high", False),
    (0.2, "medium", True),
    (0.4, "low", True)
]

for pos_ratio, importance, equal_cost in dataset_characteristics:
    recommendation = recommend_metric(pos_ratio, importance, equal_cost)
    print(f"Positive ratio: {pos_ratio}, Importance: {importance}, Equal error cost: {equal_cost}")
    print(f"Recommendation: {recommendation}\n")
```

Slide 15: Additional Resources

For those interested in diving deeper into the topic of evaluating binary classifiers on imbalanced datasets, here are some valuable resources:

1.  ArXiv paper: "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets" by Saito and Rehmsmeier (2015). ArXiv:1502.05803 \[cs.LG\]
2.  ArXiv paper: "Rethinking the Relationship Between AUC-ROC and AUC-PR" by Saito and Rehmsmeier (2020). ArXiv:2005.13441 \[cs.LG\]
3.  ArXiv paper: "Pitfalls of Evaluating a Classifier's Performance in High Class Imbalance Scenarios" by Brabec et al. (2021). ArXiv:2110.03782 \[cs.LG\]

These papers provide in-depth analysis and discussion on the use of AUC-PR and AUC-ROC in various scenarios, offering valuable insights for practitioners working with imbalanced datasets.

