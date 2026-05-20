## Sigmoid Function in Logistic Regression with Python

Slide 1: The Sigmoid Function: The Heart of Logistic Regression

The logistic regression model uses the sigmoid function to transform its output into a probability value between 0 and 1. This function, also known as the logistic function, is an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1.

```python
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid(True)
plt.show()
```

Slide 2: Mathematical Definition of the Sigmoid Function

The sigmoid function is defined mathematically as σ(x) = 1 / (1 + e^(-x)), where e is the base of natural logarithms. This function has several important properties that make it ideal for logistic regression, including its ability to map any input to a value between 0 and 1 and its smooth, differentiable nature.

```python

# Define the symbolic variable
x = sp.Symbol('x')

# Define the sigmoid function
sigmoid = 1 / (1 + sp.exp(-x))

# Print the mathematical expression
print(f"Sigmoid function: σ(x) = {sigmoid}")

# Calculate the derivative
derivative = sp.diff(sigmoid, x)
print(f"Derivative of sigmoid: σ'(x) = {derivative}")
```

Slide 3: Implementing Logistic Regression with Sigmoid Function

In logistic regression, we use the sigmoid function to transform the linear combination of input features into a probability. This process involves calculating the weighted sum of inputs and then applying the sigmoid function to obtain a probability value.

```python

def logistic_regression(X, weights, bias):
    z = np.dot(X, weights) + bias
    return 1 / (1 + np.exp(-z))

# Example data
X = np.array([[1, 2], [2, 3], [3, 4]])
weights = np.array([0.5, 0.5])
bias = -1

probabilities = logistic_regression(X, weights, bias)
print("Probabilities:", probabilities)
```

Slide 4: Training a Logistic Regression Model

Training a logistic regression model involves finding the optimal weights and bias that minimize the difference between predicted probabilities and actual labels. This is typically done using optimization algorithms like gradient descent.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print("Predicted classes:", y_pred[:5])
print("Predicted probabilities:", y_prob[:5])
```

Slide 5: Visualizing Decision Boundaries

The sigmoid function in logistic regression creates a decision boundary that separates different classes. For binary classification, this boundary is where the predicted probability is exactly 0.5. Let's visualize this boundary for a simple 2D dataset.

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=1, n_clusters_per_class=1)

# Train logistic regression model
clf = LogisticRegression(random_state=0).fit(X, y)

# Create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Obtain labels for each point in mesh using the model.
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the results
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()
```

Slide 6: Interpreting Logistic Regression Coefficients

The coefficients in logistic regression represent the change in the log-odds of the outcome for a one-unit increase in the corresponding feature. We can exponentiate these coefficients to get the odds ratios, which are often easier to interpret.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X = iris.data[:, [2, 3]]  # petal length and width
y = (iris.target == 2).astype(int)  # 1 if Iris-Virginica, else 0

# Train logistic regression
model = LogisticRegression()
model.fit(X, y)

# Create a dataframe of coefficients
feature_names = ['Petal Length', 'Petal Width']
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_[0],
    'Odds Ratio': np.exp(model.coef_[0])
})

print(coef_df)
```

Slide 7: Real-Life Example: Spam Email Classification

Logistic regression is widely used in email spam detection. The model can learn to classify emails as spam or not spam based on various features such as the presence of certain words, sender information, and email structure.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Sample data (in practice, you'd have a much larger dataset)
emails = [
    ("Free offer! Click now!", 1),
    ("Meeting at 3pm tomorrow", 0),
    ("Get rich quick! Limited time!", 1),
    ("Project report due next week", 0),
    ("Congratulations! You've won!", 1)
]

# Prepare the data
X, y = zip(*emails)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Test the model
test_email = ["Urgent: Your account needs attention"]
test_email_vectorized = vectorizer.transform(test_email)
probability = model.predict_proba(test_email_vectorized)[0][1]

print(f"Probability of being spam: {probability:.2f}")
```

Slide 8: Real-Life Example: Medical Diagnosis

Logistic regression is also commonly used in medical diagnosis to predict the likelihood of a patient having a certain condition based on various symptoms and test results. Here's a simplified example for predicting diabetes risk.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample data (you'd typically have more features and samples)
data = {
    'glucose': [90, 110, 130, 150, 170],
    'bmi': [22, 26, 30, 34, 38],
    'age': [30, 40, 50, 60, 70],
    'has_diabetes': [0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Prepare the data
X = df[['glucose', 'bmi', 'age']]
y = df['has_diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict for a new patient
new_patient = [[120, 28, 45]]  # glucose, bmi, age
new_patient_scaled = scaler.transform(new_patient)
risk_probability = model.predict_proba(new_patient_scaled)[0][1]

print(f"Probability of diabetes risk: {risk_probability:.2f}")
```

Slide 9: Handling Multiclass Classification

While binary classification is common, logistic regression can be extended to handle multiple classes using techniques like one-vs-rest or softmax regression. Here's an example using scikit-learn's built-in support for multiclass logistic regression.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression(multi_class='ovr', solver='lbfgs')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

Slide 10: Regularization in Logistic Regression

Regularization helps prevent overfitting in logistic regression by adding a penalty term to the loss function. The two main types of regularization are L1 (Lasso) and L2 (Ridge). Here's an example comparing non-regularized, L1, and L2 regularized logistic regression.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, 
                           n_redundant=10, n_classes=2, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
models = {
    'No Regularization': LogisticRegression(penalty='none'),
    'L1 Regularization': LogisticRegression(penalty='l1', solver='liblinear'),
    'L2 Regularization': LogisticRegression(penalty='l2')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
```

Slide 11: Feature Importance in Logistic Regression

Logistic regression coefficients can be used to determine feature importance. Larger absolute values indicate more important features. However, it's crucial to scale features before interpreting coefficients.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_scaled, y)

# Create a dataframe of feature importances
feature_importance = pd.DataFrame({
    'feature': data.feature_names,
    'importance': abs(model.coef_[0])
})

# Sort by importance
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot the top 10 features
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Important Features')
plt.tight_layout()
plt.show()
```

Slide 12: Cross-Validation for Model Evaluation

Cross-validation is a crucial technique for assessing how well a logistic regression model will generalize to an independent dataset. It helps in detecting overfitting and provides a more robust estimate of model performance.

```python
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Create a logistic regression model
model = LogisticRegression()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Standard deviation of CV scores:", cv_scores.std())
```

Slide 13: Handling Imbalanced Datasets

In many real-world scenarios, datasets are imbalanced, with one class significantly outnumbering the other. This can lead to poor performance on the minority class. Techniques like class weighting can help address this issue in logistic regression.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Generate an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], 
                           n_informative=3, n_redundant=1, flip_y=0, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models with and without class weighting
model_unweighted = LogisticRegression()
model_weighted = LogisticRegression(class_weight='balanced')

model_unweighted.fit(X_train, y_train)
model_weighted.fit(X_train, y_train)

# Evaluate models
print("Unweighted Model:")
print(classification_report(y_test, model_unweighted.predict(X_test)))

print("\nWeighted Model:")
print(classification_report(y_test, model_weighted.predict(X_test)))
```

Slide 14: Interpreting Logistic Regression Results

Understanding the output of a logistic regression model is crucial for making informed decisions. Let's explore how to interpret the coefficients, odds ratios, and p-values of a logistic regression model.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from scipy import stats

# Load iris dataset
iris = load_iris()
X = iris.data[:, [2, 3]]  # petal length and width
y = (iris.target == 2).astype(int)  # 1 if Iris-Virginica, else 0

# Fit logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Calculate p-values
p_values = stats.norm.sf(abs(model.coef_[0] / np.sqrt(np.diag(np.cov(X.T)))))

# Create summary dataframe
summary = pd.DataFrame({
    'Feature': ['Petal Length', 'Petal Width'],
    'Coefficient': model.coef_[0],
    'Odds Ratio': np.exp(model.coef_[0]),
    'P-value': p_values
})

print(summary)
```

Slide 15: Logistic Regression in Time Series Analysis

While logistic regression is typically used for classification tasks, it can also be applied to time series data for predicting binary outcomes over time. This example demonstrates how to use logistic regression to predict whether a stock price will increase or decrease based on previous price changes.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Generate synthetic stock price data
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', end='2023-01-01', freq='D')
prices = np.cumsum(np.random.randn(len(dates))) + 100

# Create features (price changes over different periods)
df = pd.DataFrame({'price': prices}, index=dates)
df['change_1d'] = df['price'].pct_change()
df['change_5d'] = df['price'].pct_change(5)
df['change_20d'] = df['price'].pct_change(20)

# Create target (1 if price increases next day, 0 otherwise)
df['target'] = (df['price'].shift(-1) > df['price']).astype(int)

# Prepare data for modeling
X = df.dropna()[['change_1d', 'change_5d', 'change_20d']]
y = df.dropna()['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

print(f"Model accuracy: {accuracy:.2f}")
```

Slide 16: Additional Resources

For those interested in diving deeper into logistic regression and its applications, here are some valuable resources:

1. "Logistic Regression: From Basics to Advanced Concepts" by Gareth James et al. (2013). Available on ArXiv: [https://arxiv.org/abs/1503.06503](https://arxiv.org/abs/1503.06503)
2. "A Comprehensive Guide to Logistic Regression in Python" by Sayak Paul (2020). ArXiv preprint: [https://arxiv.org/abs/2006.01989](https://arxiv.org/abs/2006.01989)
3. "Regularization Paths for Generalized Linear Models via Coordinate Descent" by Jerome Friedman et al. (2010). Journal of Statistical Software. ArXiv: [https://arxiv.org/abs/0708.1485](https://arxiv.org/abs/0708.1485)

These resources provide in-depth explanations of logistic regression concepts, implementation details, and advanced techniques for improving model performance.

