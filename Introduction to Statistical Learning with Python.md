## Introduction to Statistical Learning with Python:
Slide 1: 

Introduction to Statistical Learning

Statistical learning refers to a vast set of tools for understanding data. It has led to fascinating advances in fields ranging from biology to astrophysics to marketing and beyond. In this slideshow, we will explore the fundamental concepts and techniques of statistical learning using Python.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

# Load the Boston Housing dataset
boston = load_boston()
X, y = boston.data, boston.target
```

Slide 2: 

Supervised vs. Unsupervised Learning

Statistical learning methods can be broadly divided into two categories: supervised and unsupervised learning. In supervised learning, we have a target variable that we are trying to predict, while in unsupervised learning, we are exploring the data without a specific target.

```python
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Slide 3: 

Linear Regression

Linear regression is a fundamental supervised learning technique for modeling the relationship between a dependent variable and one or more independent variables. It is widely used for prediction and forecasting tasks.

```python
from sklearn.linear_model import LinearRegression

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)
```

Slide 4: 

Evaluation Metrics

To assess the performance of a machine learning model, we use evaluation metrics. Common metrics for regression tasks include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared.

```python
from sklearn.metrics import mean_squared_error, r2_score

# Calculate MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
```

Slide 5: 

Overfitting and Underfitting

Overfitting and underfitting are common challenges in machine learning. Overfitting occurs when the model captures noise in the training data, while underfitting occurs when the model fails to capture the underlying patterns.

```python
import matplotlib.pyplot as plt

# Generate a synthetic dataset
X_synth = np.linspace(-10, 10, 100)
y_synth = np.sin(X_synth) + np.random.normal(0, 0.5, 100)

# Visualize the data
plt.scatter(X_synth, y_synth)
plt.show()
```

Slide 6: 

Regularization

Regularization is a technique used to prevent overfitting by adding a penalty term to the objective function, which can help simplify the model and improve generalization.

```python
from sklearn.linear_model import Ridge, Lasso

# Create a Ridge regression model
ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)

# Create a Lasso regression model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
```

Slide 7: 

Cross-Validation

Cross-validation is a resampling technique used to evaluate the performance of a machine learning model on unseen data and to select the best hyperparameters.

```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation on the linear regression model
scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {scores.mean():.2f}")
```

Slide 8: 

Logistic Regression

Logistic regression is a supervised learning algorithm used for binary classification problems, where the target variable can take one of two values (e.g., 0 or 1, true or false).

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Create a logistic regression model
logreg = LogisticRegression()

# Fit the model to the data
logreg.fit(X, y)
```

Slide 9: 

Classification Evaluation Metrics

For classification tasks, we use different evaluation metrics than regression, such as accuracy, precision, recall, and F1-score.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Make predictions on the test data
y_pred = logreg.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
```

Slide 10: 

Decision Trees

Decision trees are a powerful machine learning algorithm that can be used for both classification and regression tasks. They work by recursively partitioning the input space based on the features.

```python
from sklearn.tree import DecisionTreeRegressor

# Create a decision tree regressor
tree = DecisionTreeRegressor(max_depth=3)

# Fit the model to the training data
tree.fit(X_train, y_train)

# Make predictions on the test data
y_pred = tree.predict(X_test)
```

Slide 11: 

Random Forests

Random forests are an ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting.

```python
from sklearn.ensemble import RandomForestRegressor

# Create a random forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf.predict(X_test)
```

Slide 12: 

Clustering

Clustering is an unsupervised learning technique used to group similar data points together based on their features. Popular clustering algorithms include K-Means and Hierarchical Clustering.

```python
from sklearn.cluster import KMeans

# Create a KMeans clustering model
kmeans = KMeans(n_clusters=3)

# Fit the model to the data
kmeans.fit(X)

# Get the cluster labels for each data point
labels = kmeans.labels_
```

Slide 13: 

Dimensionality Reduction

Dimensionality reduction techniques, such as Principal Component Analysis (PCA) and t-SNE, are used to visualize and analyze high-dimensional data by projecting it onto a lower-dimensional space.

```python
from sklearn.decomposition import PCA

# Create a PCA model
pca = PCA(n_components=2)

# Fit and transform the data
X_pca = pca.fit_transform(X)

# Visualize the data in the reduced 2D space
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.show()
```

Slide 14: 

Additional Resources

Here are some additional resources for learning more about statistical learning and machine learning with Python:

* "An Introduction to Statistical Learning" by Gareth James et al. (Book) \[[http://faculty.marshall.usc.edu/gareth-james/ISL/](http://faculty.marshall.usc.edu/gareth-james/ISL/)\]
* "Pattern Recognition and Machine Learning" by Christopher Bishop (Book) \[[https://arxiv.org/abs/1107.0913](https://arxiv.org/abs/1107.0913)\]
* "Machine Learning: A Probabilistic Perspective" by Kevin P. Murphy (Book) \[[https://arxiv.org/abs/1211.7114](https://arxiv.org/abs/1211.7114)\]

