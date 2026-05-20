## Methods to Improve Machine Learning Accuracy Evaluation
Slide 1: Introduction to Accuracy Evaluation in Machine Learning

Accuracy evaluation is crucial in machine learning to assess model performance. It involves comparing predicted outcomes with actual results. This process helps in understanding how well a model generalizes to unseen data and guides improvements in model design and training.

```python
import numpy as np
from sklearn.metrics import accuracy_score

# Sample data
y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 1, 1, 1])

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

Slide 2: Cross-Validation: K-Fold

Cross-validation is a technique to assess model performance on different subsets of data. K-Fold cross-validation divides the dataset into K subsets, trains on K-1 folds, and tests on the remaining fold. This process is repeated K times, with each fold serving as the test set once.

```python
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Perform 5-fold cross-validation
scores = cross_val_score(clf, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.2f}")
```

Slide 3: Confusion Matrix

A confusion matrix provides a detailed breakdown of correct and incorrect classifications for each class. It helps identify which classes are being confused with each other, allowing for targeted improvements in the model or feature engineering.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
y_true = [0, 1, 2, 2, 1]
y_pred = [0, 0, 2, 2, 0]

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

Slide 4: Precision, Recall, and F1-Score

Precision measures the accuracy of positive predictions, recall measures the ability to find all positive instances, and the F1-score is the harmonic mean of precision and recall. These metrics are particularly useful for imbalanced datasets where accuracy alone might be misleading.

```python
from sklearn.metrics import precision_recall_fscore_support

# Sample data
y_true = [0, 1, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 1]

# Calculate precision, recall, and F1-score
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
```

Slide 5: ROC Curve and AUC

The Receiver Operating Characteristic (ROC) curve visualizes the trade-off between true positive rate and false positive rate at various classification thresholds. The Area Under the Curve (AUC) summarizes the model's performance in a single value, with higher values indicating better performance.

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Sample data (probabilities for positive class)
y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]

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

Slide 6: Learning Curves

Learning curves help visualize how model performance changes with increasing amounts of training data. They can reveal issues like overfitting or underfitting and guide decisions on whether to collect more data or increase model complexity.

```python
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

# Create SVM classifier
clf = SVC(kernel='rbf', random_state=42)

# Calculate learning curves
train_sizes, train_scores, test_scores = learning_curve(
    clf, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# Calculate mean and standard deviation
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.title("Learning Curves")
plt.legend(loc="best")
plt.show()
```

Slide 7: Stratified Sampling

Stratified sampling ensures that the proportion of samples for each class is roughly the same in training and testing sets. This is particularly important for imbalanced datasets to prevent bias in model evaluation.

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Perform stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Check class distribution
print("Training set class distribution:")
print(np.bincount(y_train) / len(y_train))
print("\nTesting set class distribution:")
print(np.bincount(y_test) / len(y_test))
```

Slide 8: Bootstrapping for Confidence Intervals

Bootstrapping involves repeatedly sampling with replacement from the dataset to estimate the variability of a metric. This technique helps in calculating confidence intervals for performance measures, providing a range of likely values rather than a single point estimate.

```python
import numpy as np
from sklearn.utils import resample

def bootstrap_accuracy(y_true, y_pred, n_iterations=1000, ci=0.95):
    accuracies = []
    for _ in range(n_iterations):
        # Resample with replacement
        indices = resample(range(len(y_true)), n_samples=len(y_true))
        y_true_resampled = y_true[indices]
        y_pred_resampled = y_pred[indices]
        
        # Calculate accuracy for this sample
        accuracy = np.mean(y_true_resampled == y_pred_resampled)
        accuracies.append(accuracy)
    
    # Calculate confidence interval
    lower = np.percentile(accuracies, (1 - ci) / 2 * 100)
    upper = np.percentile(accuracies, (1 + ci) / 2 * 100)
    
    return np.mean(accuracies), (lower, upper)

# Example usage
y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])

mean_accuracy, ci = bootstrap_accuracy(y_true, y_pred)
print(f"Mean Accuracy: {mean_accuracy:.2f}")
print(f"95% Confidence Interval: ({ci[0]:.2f}, {ci[1]:.2f})")
```

Slide 9: McNemar's Test for Model Comparison

McNemar's test is a statistical method to compare the performance of two models on the same dataset. It focuses on the cases where one model is correct and the other is incorrect, helping to determine if the difference in performance is statistically significant.

```python
from statsmodels.stats.contingency_tables import mcnemar

def mcnemar_test(y_true, pred_model1, pred_model2):
    # Create contingency table
    table = [[sum((pred_model1 == y_true) & (pred_model2 == y_true)),
              sum((pred_model1 == y_true) & (pred_model2 != y_true))],
             [sum((pred_model1 != y_true) & (pred_model2 == y_true)),
              sum((pred_model1 != y_true) & (pred_model2 != y_true))]]
    
    # Perform McNemar's test
    result = mcnemar(table, exact=False, correction=True)
    return result.statistic, result.pvalue

# Example usage
y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
pred_model1 = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])
pred_model2 = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 1])

statistic, p_value = mcnemar_test(y_true, pred_model1, pred_model2)
print(f"McNemar's test statistic: {statistic:.2f}")
print(f"p-value: {p_value:.4f}")
```

Slide 10: Calibration Curves

Calibration curves, also known as reliability diagrams, assess how well the predicted probabilities of a classifier correspond to the actual probabilities. A well-calibrated model should have predictions that match the true probabilities.

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_pred_proba = np.random.rand(1000)

# Calculate calibration curve
prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)

# Plot calibration curve
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.show()
```

Slide 11: Feature Importance Analysis

Feature importance analysis helps identify which features contribute most to the model's predictions. This can guide feature selection, improve model interpretability, and highlight areas for potential improvement in data collection or feature engineering.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Get feature importances
importances = clf.feature_importances_
feature_names = iris.feature_names

# Sort features by importance
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()
```

Slide 12: Real-Life Example: Image Classification

In image classification tasks, accuracy evaluation is crucial. Let's consider a model classifying images of fruits. We'll use a pre-trained ResNet model and evaluate its performance on a small dataset of apple and orange images.

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

# Define data transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test dataset (assuming you have a 'test' folder with 'apple' and 'orange' subfolders)
test_data = ImageFolder('test', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Evaluate model
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: Apple vs Orange Classification')
plt.show()

# Calculate accuracy
accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
print(f"Accuracy: {accuracy:.2f}")
```

Slide 13: Real-Life Example: Sentiment Analysis

Sentiment analysis is widely used in natural language processing. Let's evaluate a simple sentiment classifier on movie reviews using a sample dataset.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Sample movie reviews
reviews = [
    "This movie was excellent! I loved every minute of it.",
    "Terrible film, waste of time and money.",
    "Great acting, but the plot was confusing.",
    "I fell asleep halfway through, so boring.",
    "A masterpiece of cinema, highly recommended!",
    "The worst movie I've ever seen.",
    "Entertaining but forgettable.",
    "A true cinematic achievement.",
    "Disappointing and poorly executed.",
    "An absolute must-watch for film enthusiasts."
]
sentiments = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]  # 1 for positive, 0 for negative

# Split the data
X_train, X_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = clf.predict(X_test_vectorized)

# Evaluate the model
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
```

Slide 14: Handling Class Imbalance: SMOTE

Class imbalance can significantly affect model performance. Synthetic Minority Over-sampling Technique (SMOTE) is a method to address this issue by creating synthetic examples of the minority class.

```python
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train a classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
```

Slide 15: Time Series Cross-Validation

Traditional cross-validation techniques can lead to data leakage in time series problems. Time series cross-validation ensures that future data is not used to predict past events.

```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Generate sample time series data
np.random.seed(42)
X = np.array([i for i in range(100)]).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, 100)

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
mse_scores = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

print(f"Mean MSE: {np.mean(mse_scores):.4f}")
print(f"Std MSE: {np.std(mse_scores):.4f}")
```

Slide 16: Additional Resources

For further exploration of accuracy evaluation methods in machine learning, consider the following resources:

1. "A Survey of Cross-Validation Procedures for Model Selection" by Arlot, S. and Celisse, A. (2010) ArXiv: [https://arxiv.org/abs/0907.4728](https://arxiv.org/abs/0907.4728)
2. "Comparing Machine Learning Models for Ensemble-Based Forecasting" by Shaub, D. (2020) ArXiv: [https://arxiv.org/abs/2003.01926](https://arxiv.org/abs/2003.01926)
3. "Evaluation Metrics for Binary Classification" by Hossin, M. and Sulaiman, M.N. (2015) ArXiv: [https://arxiv.org/abs/1504.06792](https://arxiv.org/abs/1504.06792)

These papers provide in-depth discussions on various aspects of model evaluation and comparison in machine learning.

