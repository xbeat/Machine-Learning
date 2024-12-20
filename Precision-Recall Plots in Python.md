## Precision-Recall Plots in Python
Slide 1: Precision-Recall Plot: An Introduction

Precision-Recall plots are powerful tools for evaluating binary classification models, especially when dealing with imbalanced datasets. These plots help visualize the trade-off between precision and recall, providing insights into a model's performance across different classification thresholds.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Generate sample data
y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.65, 0.2, 0.9, 0.5, 0.7, 0.3])

# Calculate precision and recall values
precision, recall, _ = precision_recall_curve(y_true, y_scores)

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```

Slide 2: Understanding Precision and Recall

Precision measures the proportion of correct positive predictions among all positive predictions, while recall measures the proportion of correct positive predictions among all actual positive instances. These metrics are crucial for understanding a model's performance, especially in scenarios where false positives or false negatives have significant consequences.

```python
from sklearn.metrics import precision_score, recall_score

# Calculate precision and recall
precision = precision_score(y_true, y_scores > 0.5)
recall = recall_score(y_true, y_scores > 0.5)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Visualize true positives, false positives, and false negatives
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_scores, c=['red' if t else 'blue' for t in y_true])
plt.axhline(y=0.5, color='r', linestyle='--')
plt.xlabel('True Label')
plt.ylabel('Predicted Score')
plt.title('Classification Results')
plt.show()
```

Slide 3: The Precision-Recall Trade-off

There's often a trade-off between precision and recall. Increasing one typically results in decreasing the other. The optimal balance depends on the specific problem and the costs associated with false positives and false negatives. The Precision-Recall curve helps visualize this trade-off across different classification thresholds.

```python
# Generate more sample data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_scores = np.random.rand(1000)

# Calculate precision and recall values
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Plot Precision-Recall curve with threshold annotations
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, marker='')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with Thresholds')

# Annotate some threshold values
for i, thresh in enumerate(thresholds):
    if i % 100 == 0:  # Annotate every 100th threshold
        plt.annotate(f'{thresh:.2f}', (recall[i], precision[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.show()
```

Slide 4: Average Precision Score

The Average Precision (AP) score summarizes the Precision-Recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight. It provides a single number to evaluate the quality of the model across all possible thresholds.

```python
from sklearn.metrics import average_precision_score

# Calculate Average Precision
ap_score = average_precision_score(y_true, y_scores)

print(f"Average Precision Score: {ap_score:.4f}")

# Plot Precision-Recall curve with AP score
precision, recall, _ = precision_recall_curve(y_true, y_scores)

plt.figure(figsize=(8, 6))
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP = {ap_score:.4f})')
plt.fill_between(recall, precision, step='post', alpha=0.2)
plt.show()
```

Slide 5: Interpreting the Precision-Recall Curve

A Precision-Recall curve that hugs the top-right corner indicates better performance. The curve starts at the top-left (high precision, low recall) and ends at the bottom-right (low precision, high recall). The area under the curve (AUC) represents the model's overall performance, with higher values indicating better performance.

```python
from sklearn.metrics import auc

# Calculate AUC
auc_score = auc(recall, precision)

print(f"Area Under the Precision-Recall Curve: {auc_score:.4f}")

# Plot Precision-Recall curve with AUC
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'AUC = {auc_score:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with AUC')
plt.legend(loc='lower left')
plt.fill_between(recall, precision, alpha=0.2)
plt.show()
```

Slide 6: Comparing Multiple Models

Precision-Recall plots are excellent for comparing the performance of multiple models on the same dataset. By plotting curves for different models on the same graph, we can easily visualize which model performs better across various threshold values.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Generate a binary classification dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.7, 0.3], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train different models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

plt.figure(figsize=(10, 6))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap_score = average_precision_score(y_test, y_scores)
    plt.plot(recall, precision, label=f'{name} (AP = {ap_score:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for Different Models')
plt.legend(loc='lower left')
plt.show()
```

Slide 7: Handling Imbalanced Datasets

Precision-Recall plots are particularly useful for imbalanced datasets where one class is much more frequent than the other. In such cases, accuracy can be misleading, and Precision-Recall curves provide a more informative view of model performance.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Generate an imbalanced dataset
X, y = make_classification(n_samples=10000, n_classes=2, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model on imbalanced data
model_imb = RandomForestClassifier(random_state=42)
model_imb.fit(X_train, y_train)
y_scores_imb = model_imb.predict_proba(X_test)[:, 1]

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train model on balanced data
model_bal = RandomForestClassifier(random_state=42)
model_bal.fit(X_train_balanced, y_train_balanced)
y_scores_bal = model_bal.predict_proba(X_test)[:, 1]

# Plot Precision-Recall curves
plt.figure(figsize=(10, 6))
for y_scores, label in [(y_scores_imb, 'Imbalanced'), (y_scores_bal, 'Balanced')]:
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap_score = average_precision_score(y_test, y_scores)
    plt.plot(recall, precision, label=f'{label} (AP = {ap_score:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves: Imbalanced vs Balanced Data')
plt.legend(loc='lower left')
plt.show()
```

Slide 8: Choosing the Right Threshold

The Precision-Recall curve helps in selecting an appropriate classification threshold based on the specific requirements of your problem. You can use it to find the threshold that gives the best balance between precision and recall for your use case.

```python
from sklearn.metrics import f1_score

# Generate sample data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_scores = np.random.rand(1000)

# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Calculate F1 score for each threshold
f1_scores = [f1_score(y_true, y_scores >= threshold) for threshold in thresholds]

# Find the threshold that gives the best F1 score
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)

print(f"Best threshold: {best_threshold:.2f}")
print(f"Best F1 score: {best_f1:.2f}")

# Plot Precision-Recall curve with best threshold
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label='Precision-Recall curve')
plt.scatter(recall[np.argmax(f1_scores)], precision[np.argmax(f1_scores)], 
            color='red', label=f'Best Threshold ({best_threshold:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with Best F1 Score Threshold')
plt.legend()
plt.show()
```

Slide 9: Real-Life Example: Spam Detection

In email spam detection, we want to classify emails as spam or not spam. Here, precision represents the proportion of correctly identified spam emails among all emails classified as spam, while recall represents the proportion of correctly identified spam emails among all actual spam emails.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Sample email data (content, label)
emails = [
    ("Get rich quick! Buy now!", 1),
    ("Meeting at 3pm tomorrow", 0),
    ("You've won a free iPhone!", 1),
    ("Project deadline extended", 0),
    ("Congrats! You're our lucky winner", 1),
    ("Please review the attached document", 0),
    # Add more examples...
]

# Separate content and labels
X, y = zip(*emails)

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Get predicted probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Plot Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_scores)
ap_score = average_precision_score(y_test, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'AP = {ap_score:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Spam Detection')
plt.legend()
plt.show()
```

Slide 10: Real-Life Example: Medical Diagnosis

In medical diagnosis, such as detecting a disease from medical images, precision represents the proportion of correctly identified positive cases among all cases classified as positive, while recall represents the proportion of correctly identified positive cases among all actual positive cases.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Generate synthetic medical data
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], 
                           n_features=20, n_informative=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get predicted probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
ap_score = average_precision_score(y_test, y_scores)

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'AP = {ap_score:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Disease Detection')
plt.legend()

# Add annotations for different thresholds
for i, thresh in enumerate(thresholds):
    if i % 100 == 0:  # Annotate every 100th threshold
        plt.annotate(f'{thresh:.2f}', (recall[i], precision[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center')

plt.show()

# Print interpretation
print("In this medical diagnosis example:")
print(f"- A high precision of {precision[0]:.2f} at a low recall of {recall[0]:.2f} means we're very sure about positive diagnoses, but we might miss some cases.")
print(f"- A high recall of {recall[-1]:.2f} at a low precision of {precision[-1]:.2f} means we catch most positive cases, but with more false positives.")
print("The choice of threshold depends on the relative costs of false positives vs. false negatives in the specific medical context.")
```

Slide 11: Limitations and Considerations

While Precision-Recall plots are powerful tools, they have limitations. They don't account for true negatives, which can be important in some scenarios. Additionally, they may not be suitable for multi-class classification problems without modification. It's crucial to consider these factors when interpreting Precision-Recall curves.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Generate sample data for two different scenarios
np.random.seed(42)
y_true1 = np.random.randint(0, 2, 1000)
y_scores1 = np.random.rand(1000)

y_true2 = np.random.randint(0, 2, 1000)
y_scores2 = np.random.rand(1000) * 0.5 + 0.25

# Calculate precision-recall curves
precision1, recall1, _ = precision_recall_curve(y_true1, y_scores1)
precision2, recall2, _ = precision_recall_curve(y_true2, y_scores2)

# Plot both curves
plt.figure(figsize=(10, 6))
plt.plot(recall1, precision1, label='Scenario 1')
plt.plot(recall2, precision2, label='Scenario 2')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves: Comparing Different Scenarios')
plt.legend()
plt.show()

# Calculate and print average precision scores
ap1 = average_precision_score(y_true1, y_scores1)
ap2 = average_precision_score(y_true2, y_scores2)
print(f"Average Precision - Scenario 1: {ap1:.3f}")
print(f"Average Precision - Scenario 2: {ap2:.3f}")
```

Slide 12: Addressing Class Imbalance

Class imbalance can significantly affect the interpretation of Precision-Recall curves. When dealing with imbalanced datasets, it's important to consider techniques such as oversampling, undersampling, or using class weights to improve model performance and the reliability of the Precision-Recall curve.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Generate imbalanced dataset
X, y = make_classification(n_samples=10000, n_classes=2, weights=[0.99, 0.01], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model on imbalanced data
model_imb = RandomForestClassifier(random_state=42)
model_imb.fit(X_train, y_train)
y_scores_imb = model_imb.predict_proba(X_test)[:, 1]

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train model on balanced data
model_bal = RandomForestClassifier(random_state=42)
model_bal.fit(X_train_balanced, y_train_balanced)
y_scores_bal = model_bal.predict_proba(X_test)[:, 1]

# Plot Precision-Recall curves
plt.figure(figsize=(10, 6))
for y_scores, label in [(y_scores_imb, 'Imbalanced'), (y_scores_bal, 'Balanced (SMOTE)')]:
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap_score = average_precision_score(y_test, y_scores)
    plt.plot(recall, precision, label=f'{label} (AP = {ap_score:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves: Imbalanced vs Balanced Data')
plt.legend()
plt.show()
```

Slide 13: Combining Precision-Recall with ROC Curves

While Precision-Recall curves are especially useful for imbalanced datasets, combining them with Receiver Operating Characteristic (ROC) curves can provide a more comprehensive view of model performance. ROC curves plot the True Positive Rate against the False Positive Rate and are less sensitive to class imbalance.

```python
from sklearn.metrics import roc_curve, auc

# Generate sample data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_scores = np.random.rand(1000)

# Calculate Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_true, y_scores)
ap_score = average_precision_score(y_true, y_scores)

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot both curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Precision-Recall curve
ax1.plot(recall, precision, label=f'AP = {ap_score:.2f}')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('Precision-Recall Curve')
ax1.legend()

# ROC curve
ax2.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
ax2.plot([0, 1], [0, 1], linestyle='--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend()

plt.tight_layout()
plt.show()
```

Slide 14: Practical Tips for Using Precision-Recall Plots

When working with Precision-Recall plots, consider the following tips:

1. Always compare your model's performance to a random classifier (represented by a horizontal line).
2. Use the Average Precision score as a single metric to summarize the curve.
3. Consider the specific needs of your problem when choosing a threshold.
4. For imbalanced datasets, prefer Precision-Recall curves over ROC curves.
5. Combine Precision-Recall analysis with other evaluation metrics for a comprehensive assessment.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Generate data for a good model and a random classifier
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_scores_good = np.random.beta(2, 5, 1000)  # Skewed towards lower values
y_scores_random = np.random.rand(1000)

# Calculate Precision-Recall curves
precision_good, recall_good, _ = precision_recall_curve(y_true, y_scores_good)
precision_random, recall_random, _ = precision_recall_curve(y_true, y_scores_random)

# Calculate Average Precision scores
ap_score_good = average_precision_score(y_true, y_scores_good)
ap_score_random = average_precision_score(y_true, y_scores_random)

# Plot Precision-Recall curves
plt.figure(figsize=(10, 6))
plt.plot(recall_good, precision_good, label=f'Good Model (AP = {ap_score_good:.2f})')
plt.plot(recall_random, precision_random, label=f'Random Classifier (AP = {ap_score_random:.2f})')

# Add a line for the random classifier baseline
no_skill = len(y_true[y_true == 1]) / len(y_true)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves: Good Model vs Random Classifier')
plt.legend()
plt.show()

print("Tips for interpreting this plot:")
print("1. The 'Good Model' curve is significantly above the random classifier.")
print("2. The Average Precision (AP) score summarizes the curve's performance.")
print("3. Choose a threshold based on the trade-off between precision and recall for your specific problem.")
print("4. The 'No Skill' line represents the performance of a random classifier.")
```

Slide 15: Additional Resources

For those interested in diving deeper into Precision-Recall plots and related concepts, here are some valuable resources:

1. Saito, T., & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE, 10(3), e0118432. ArXiv: [https://arxiv.org/abs/1502.01194](https://arxiv.org/abs/1502.01194)
2. Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. Proceedings of the 23rd International Conference on Machine Learning - ICML '06. ArXiv: [https://arxiv.org/abs/cs/0606118](https://arxiv.org/abs/cs/0606118)
3. Flach, P., & Kull, M. (2015). Precision-Recall-Gain Curves: PR Analysis Done Right. Advances in Neural Information Processing Systems, 28. ArXiv: [https://arxiv.org/abs/1509.03700](https://arxiv.org/abs/1509.03700)

These papers provide in-depth analysis and advanced techniques related to Precision-Recall plots, offering valuable insights for both researchers and practitioners in the field of machine learning and data science.

