## Model Evaluation Metrics in Python
Slide 1: Introduction to Model Evaluation Metrics

Model evaluation metrics are essential tools in data science for assessing the performance of machine learning models. This slideshow will explore the top 10 metrics mentioned in the image, providing code examples and practical applications for each.

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

# Sample data
y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 1, 1, 0])

# Print available metrics
print("Available metrics:")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
print(f"Precision: {precision_score(y_true, y_pred):.2f}")
print(f"Recall: {recall_score(y_true, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_true, y_pred):.2f}")
```

Slide 2: Accuracy

Accuracy is the ratio of correct predictions to total predictions. It's a simple and intuitive metric but can be misleading for imbalanced datasets.

```python
def calculate_accuracy(y_true, y_pred):
    correct = sum(y_true[i] == y_pred[i] for i in range(len(y_true)))
    total = len(y_true)
    return correct / total

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 1, 1, 0]

accuracy = calculate_accuracy(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

Slide 3: Precision

Precision is the ratio of true positives to all positive predictions. It's useful when the cost of false positives is high.

```python
def calculate_precision(y_true, y_pred):
    true_positives = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    predicted_positives = sum(y_pred)
    return true_positives / predicted_positives if predicted_positives > 0 else 0

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 1, 1, 0]

precision = calculate_precision(y_true, y_pred)
print(f"Precision: {precision:.2f}")
```

Slide 4: Recall

Recall is the ratio of true positives to all actual positives. It's important when the cost of false negatives is high.

```python
def calculate_recall(y_true, y_pred):
    true_positives = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    actual_positives = sum(y_true)
    return true_positives / actual_positives if actual_positives > 0 else 0

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 1, 1, 0]

recall = calculate_recall(y_true, y_pred)
print(f"Recall: {recall:.2f}")
```

Slide 5: F1 Score

The F1 score is the harmonic mean of precision and recall, providing a balanced measure of a model's performance.

```python
def calculate_f1_score(y_true, y_pred):
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 1, 1, 0]

f1 = calculate_f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.2f}")
```

Slide 6: AUC-ROC

AUC-ROC (Area Under the Curve - Receiver Operating Characteristic) measures the model's ability to distinguish between classes across various thresholds.

```python
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
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

# Example usage
y_true = np.array([0, 1, 1, 0, 1, 0])
y_pred_proba = np.array([0.1, 0.7, 0.8, 0.3, 0.9, 0.2])

plot_roc_curve(y_true, y_pred_proba)
```

Slide 7: MAE (Mean Absolute Error)

MAE measures the average absolute difference between predicted and true values, useful for regression problems.

```python
def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

y_true = [3, 2, 7, 1, 5]
y_pred = [2.5, 3.1, 6.8, 1.4, 4.9]

mae = calculate_mae(y_true, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
```

Slide 8: MSE (Mean Squared Error)

MSE calculates the average squared difference between predicted and true values, penalizing larger errors more heavily.

```python
def calculate_mse(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

y_true = [3, 2, 7, 1, 5]
y_pred = [2.5, 3.1, 6.8, 1.4, 4.9]

mse = calculate_mse(y_true, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
```

Slide 9: RMSE (Root Mean Squared Error)

RMSE is the square root of MSE, providing a metric in the same unit as the target variable.

```python
def calculate_rmse(y_true, y_pred):
    return np.sqrt(calculate_mse(y_true, y_pred))

y_true = [3, 2, 7, 1, 5]
y_pred = [2.5, 3.1, 6.8, 1.4, 4.9]

rmse = calculate_rmse(y_true, y_pred)
print(f"Root Mean Squared Error: {rmse:.2f}")
```

Slide 10: Log Loss

Log loss measures the performance of a classification model where the prediction is a probability value between 0 and 1.

```python
def calculate_log_loss(y_true, y_pred_proba):
    epsilon = 1e-15  # Small value to avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))

y_true = [1, 0, 1, 1, 0]
y_pred_proba = [0.9, 0.1, 0.8, 0.7, 0.3]

log_loss = calculate_log_loss(y_true, y_pred_proba)
print(f"Log Loss: {log_loss:.4f}")
```

Slide 11: R-squared

R-squared measures the proportion of variance in the dependent variable explained by the independent variables in a regression model.

```python
def calculate_r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1) * 0.1

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate R-squared
r_squared = calculate_r_squared(y, y_pred)
print(f"R-squared: {r_squared:.4f}")
```

Slide 12: Real-life Example - Spam Detection

In spam detection, precision and recall are crucial. High precision ensures legitimate emails aren't marked as spam, while high recall catches most spam emails.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

# Sample data (email content and labels)
emails = [
    "Get rich quick!", "Meeting at 3 PM", "Viagra for sale",
    "Project deadline reminder", "You've won the lottery!"
]
labels = [1, 0, 1, 0, 1]  # 1 for spam, 0 for not spam

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate precision and recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
```

Slide 13: Real-life Example - Image Classification

In image classification tasks, accuracy and F1 score are commonly used metrics. Let's simulate an image classification scenario using a simplified dataset.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Simulate image data (flatten 32x32 RGB images)
n_samples = 1000
n_features = 32 * 32 * 3
X = np.random.rand(n_samples, n_features)
y = np.random.randint(0, 5, n_samples)  # 5 classes

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy and F1 score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
```

Slide 14: Additional Resources

For further exploration of model evaluation metrics in data science, consider the following resources:

1. "A Survey of Evaluation Metrics Used for NLP Tasks" (arXiv:2008.12009)
2. "On the Use of Classification Performance Metrics" (arXiv:2008.05756)
3. "Metrics to Evaluate Machine Learning Models" (arXiv:2102.11447)

These papers provide in-depth discussions on various evaluation metrics and their applications in different domains of machine learning and data science.

