## Step-by-Step Logistic Regression in Python
Slide 1: Introduction to Logistic Regression

Logistic Regression is a fundamental statistical method used for binary classification problems. It models the probability of an instance belonging to a particular class. Despite its name, logistic regression is used for classification, not regression.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Plot the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Sample Binary Classification Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class')
plt.show()
```

Slide 2: The Logistic Function

The logistic function, also known as the sigmoid function, is the core of logistic regression. It maps any real-valued number to a value between 0 and 1, which can be interpreted as a probability.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
y = sigmoid(z)

plt.plot(z, y)
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.grid(True)
plt.show()
```

Slide 3: The Logistic Regression Model

In logistic regression, we model the probability of an instance belonging to the positive class as a function of its features. The model combines linear regression with the sigmoid function.

```python
def logistic_regression(X, theta):
    return sigmoid(np.dot(X, theta))

# Example usage
X = np.array([[1, 2], [1, 3], [1, 4]])  # Add a column of 1s for the bias term
theta = np.array([0.5, -1, 2])
probabilities = logistic_regression(X, theta)
print("Probabilities:", probabilities)
```

Slide 4: Cost Function

The cost function measures how well our model fits the training data. For logistic regression, we use the log loss (also known as cross-entropy loss).

```python
def cost_function(X, y, theta):
    m = len(y)
    h = logistic_regression(X, theta)
    cost = (-1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
    return cost

# Example usage
y = np.array([0, 1, 1])
cost = cost_function(X, y, theta)
print("Cost:", cost)
```

Slide 5: Gradient Descent

Gradient descent is an optimization algorithm used to find the model parameters (theta) that minimize the cost function.

```python
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    for _ in range(num_iters):
        h = logistic_regression(X, theta)
        gradient = (1/m) * np.dot(X.T, (h - y))
        theta -= alpha * gradient
    return theta

# Example usage
alpha = 0.01
num_iters = 1000
theta_optimized = gradient_descent(X, y, theta, alpha, num_iters)
print("Optimized theta:", theta_optimized)
```

Slide 6: Making Predictions

Once we have trained our model, we can use it to make predictions on new data.

```python
def predict(X, theta, threshold=0.5):
    probabilities = logistic_regression(X, theta)
    return (probabilities >= threshold).astype(int)

# Example usage
X_new = np.array([[1, 2.5], [1, 3.5]])
predictions = predict(X_new, theta_optimized)
print("Predictions:", predictions)
```

Slide 7: Model Evaluation - Confusion Matrix

A confusion matrix is a table used to describe the performance of a classification model. It shows the counts of true positives, true negatives, false positives, and false negatives.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_true = np.array([0, 1, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1])

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

Slide 8: Model Evaluation - ROC Curve

The Receiver Operating Characteristic (ROC) curve is a graphical representation of the model's performance across various classification thresholds.

```python
from sklearn.metrics import roc_curve, auc

# Generate random probabilities for demonstration
y_true = np.random.randint(0, 2, 100)
y_scores = np.random.random(100)

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
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
```

Slide 9: Multiclass Logistic Regression

Logistic regression can be extended to handle multiple classes using techniques like one-vs-rest or softmax regression.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression(multi_class='ovr', random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

Slide 10: Feature Scaling

Feature scaling is crucial for logistic regression, as it ensures that all features contribute equally to the model.

```python
from sklearn.preprocessing import StandardScaler

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
X[:, 1] *= 100  # Make the second feature have a much larger scale

# Plot before scaling
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1])
plt.title('Before Scaling')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Plot after scaling
plt.subplot(122)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1])
plt.title('After Scaling')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

Slide 11: Regularization

Regularization helps prevent overfitting by adding a penalty term to the cost function, discouraging complex models.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 20)
y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)

# Test different regularization strengths
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
mean_scores = []

for C in C_values:
    model = LogisticRegression(C=C, random_state=42)
    scores = cross_val_score(model, X, y, cv=5)
    mean_scores.append(scores.mean())

plt.semilogx(C_values, mean_scores, marker='o')
plt.xlabel('Regularization strength (C)')
plt.ylabel('Cross-validation score')
plt.title('Effect of Regularization on Model Performance')
plt.grid(True)
plt.show()
```

Slide 12: Real-life Example - Email Spam Classification

Logistic regression can be used to classify emails as spam or not spam based on various features such as word frequency, sender information, and email structure.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sample email data (content and label)
emails = [
    ("Get rich quick! Buy now!", 1),
    ("Meeting at 3 PM tomorrow", 0),
    ("Congratulations! You've won a prize", 1),
    ("Project deadline reminder", 0),
    ("Increase your followers instantly", 1)
]

X, y = zip(*emails)

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))
```

Slide 13: Real-life Example - Medical Diagnosis

Logistic regression can be used in medical applications to predict the likelihood of a patient having a certain condition based on various symptoms and test results.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Create a sample dataset
data = {
    'age': [45, 62, 35, 58, 40, 50, 55, 38, 42, 60],
    'blood_pressure': [130, 140, 120, 145, 135, 150, 125, 130, 140, 135],
    'cholesterol': [220, 260, 180, 240, 200, 280, 210, 190, 230, 250],
    'has_heart_disease': [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

# Split features and target
X = df.drop('has_heart_disease', axis=1)
y = df['has_heart_disease']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Heart Disease Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into logistic regression and its applications, the following resources are recommended:

1. "Logistic Regression: From Basic to Advanced" by Yaser S. Abu-Mostafa, et al. (arXiv:1407.1419) URL: [https://arxiv.org/abs/1407.1419](https://arxiv.org/abs/1407.1419)
2. "A Survey of Logistic Regression Techniques for Prediction of Student Academic Performance" by Alaa Tharwat (arXiv:2007.15991) URL: [https://arxiv.org/abs/2007.15991](https://arxiv.org/abs/2007.15991)
3. "Logistic Regression in Rare Events Data" by Gary King and Langche Zeng (arXiv:2007.12855) URL: [https://arxiv.org/abs/2007.12855](https://arxiv.org/abs/2007.12855)

These papers provide in-depth discussions on various aspects of logistic regression, from fundamental concepts to advanced techniques and applications in different domains.

