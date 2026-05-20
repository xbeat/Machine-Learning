## Evaluating Model Fit with Python
Slide 1: Evaluating Model Performance in Python

Model evaluation is crucial in machine learning to assess how well our predictions match actual values. Python offers various metrics and techniques to perform this evaluation effectively.

```python
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Sample data
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# Calculate Mean Squared Error
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate R-squared
r2 = r2_score(y_true, y_pred)
print(f"R-squared: {r2}")
```

Slide 2: Mean Squared Error (MSE)

MSE measures the average squared difference between predicted and actual values. Lower MSE indicates better model performance.

```python
import numpy as np

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Example usage
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

mse = calculate_mse(y_true, y_pred)
print(f"Mean Squared Error: {mse}")
```

Slide 3: Root Mean Squared Error (RMSE)

RMSE is the square root of MSE, providing an error measure in the same units as the target variable.

```python
import numpy as np

def calculate_rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)

# Example usage
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

rmse = calculate_rmse(y_true, y_pred)
print(f"Root Mean Squared Error: {rmse}")
```

Slide 4: R-squared (Coefficient of Determination)

R-squared measures the proportion of variance in the dependent variable explained by the independent variables. It ranges from 0 to 1, with 1 indicating perfect prediction.

```python
import numpy as np

def calculate_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Example usage
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

r2 = calculate_r2(y_true, y_pred)
print(f"R-squared: {r2}")
```

Slide 5: Mean Absolute Error (MAE)

MAE calculates the average absolute difference between predicted and actual values. It's less sensitive to outliers compared to MSE.

```python
import numpy as np

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Example usage
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

mae = calculate_mae(y_true, y_pred)
print(f"Mean Absolute Error: {mae}")
```

Slide 6: Classification Accuracy

For classification tasks, accuracy measures the proportion of correct predictions among the total number of cases examined.

```python
import numpy as np

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Example usage
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1])

accuracy = calculate_accuracy(y_true, y_pred)
print(f"Accuracy: {accuracy}")
```

Slide 7: Confusion Matrix

A confusion matrix provides a detailed breakdown of correct and incorrect classifications for each class.

```python
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Example usage
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
classes = ['Class 0', 'Class 1']

plot_confusion_matrix(y_true, y_pred, classes)
```

Slide 8: Precision, Recall, and F1-Score

These metrics are particularly useful for imbalanced classification problems.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_classification_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return precision, recall, f1

# Example usage
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])

precision, recall, f1 = calculate_classification_metrics(y_true, y_pred)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
```

Slide 9: ROC Curve and AUC

The Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) are used to evaluate the performance of classification models.

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
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
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
y_scores = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.4, 0.6, 0.2])

plot_roc_curve(y_true, y_scores)
```

Slide 10: Cross-Validation

Cross-validation helps assess model performance on unseen data and detect overfitting.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
```

Slide 11: Learning Curves

Learning curves help visualize model performance as the training set size increases, revealing potential overfitting or underfitting.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.datasets import load_digits

def plot_learning_curve(estimator, X, y, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Create an SVM classifier
svm = SVC(kernel='rbf', gamma=0.001)

# Plot learning curve
plot_learning_curve(svm, X, y)
```

Slide 12: Real-Life Example: Predicting Housing Prices

Let's evaluate a linear regression model for predicting housing prices based on various features.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```

Slide 13: Real-Life Example: Customer Churn Prediction

Let's evaluate a logistic regression model for predicting customer churn in a telecommunications company.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Load the Telco Customer Churn dataset (you'll need to download this)
df = pd.read_csv('telco_customer_churn.csv')

# Preprocess the data
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
```

Slide 14: Additional Resources

For more in-depth information on model evaluation techniques and metrics:

1. Scikit-learn Documentation: [https://scikit-learn.org/stable/modules/model\_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
2. "A Survey of Predictive Modeling under Imbalanced Distributions" by Haibo He and Edwardo A. Garcia: [https://arxiv.org/abs/1505.01658](https://arxiv.org/abs/1505.01658)
3. "An Introduction to ROC Analysis" by Tom Fawcett: [https://arxiv.org/abs/math/0503119](https://arxiv.org/abs/math/0503119)
4. "Visualizing Machine Learning Model Performance" tutorial: [https://machinelearningmastery.com/visualize-machine-learning-model-performance-measurements/](https://machinelearningmastery.com/visualize-machine-learning-model-performance-measurements/)

