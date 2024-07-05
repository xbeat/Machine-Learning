## Introduction to Support Vector Machines (SVM) in Python

Slide 1: Introduction to Support Vector Machines (SVM)

Support Vector Machines (SVM) are powerful supervised machine learning algorithms used for classification and regression tasks. They work by finding the optimal hyperplane that maximally separates classes in high-dimensional feature spaces. SVMs are particularly effective for handling non-linear problems and high-dimensional data.

```python
# No code for this introductory slide
```

Slide 2: Importing Libraries

To work with SVMs in Python, we need to import the necessary libraries. The scikit-learn library provides a user-friendly implementation of SVMs.

```python
from sklearn import datasets
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
```

Slide 3: Loading Data

Let's load the iris dataset, a popular example for classification tasks. This dataset contains measurements of iris flowers from three different species.

```python
iris = datasets.load_iris()
X = iris.data[:, :2]  # We only take the first two features
y = iris.target
```

Slide 4: Creating SVM Model

We create an instance of the SVM classifier from scikit-learn. The kernel parameter specifies the type of kernel function used for mapping the data into a higher-dimensional space.

```python
clf = svm.SVC(kernel='linear')
```

Slide 5: Training the SVM Model

The SVM model is trained using the fit() method, which takes the feature matrix (X) and target labels (y) as input.

```python
clf.fit(X, y)
```

Slide 6: Visualizing Decision Boundaries

We can visualize the decision boundaries learned by the SVM model using a mesh grid and the predict() method to get the predictions for each point in the grid.

```python
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
```

Slide 7: Plotting Training Data

We can plot the training data points on top of the decision boundaries to better visualize the classification.

```python
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVM with linear kernel')
plt.show()
```

Slide 8: Using Non-linear Kernels

SVMs can handle non-linear decision boundaries by using different kernel functions. The rbf (Radial Basis Function) kernel is a popular choice for non-linear problems.

```python
clf = svm.SVC(kernel='rbf')
clf.fit(X, y)
```

Slide 9: Visualizing Non-linear Decision Boundaries

We can visualize the non-linear decision boundaries learned by the SVM model with an RBF kernel using the same visualization code from earlier slides.

```python
# Same visualization code as before
```

Slide 10: Tuning SVM Hyperparameters

SVMs have several hyperparameters that can be tuned to improve performance, such as the regularization parameter C and the kernel coefficient gamma for the RBF kernel.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, verbose=3)
grid.fit(X, y)
print("Best parameters: ", grid.best_params_)
```

Slide 11: Making Predictions

Once the SVM model is trained, we can use it to make predictions on new, unseen data using the predict() method.

```python
new_data = [[5.1, 3.5], [4.9, 3.0], [6.7, 3.1]]
predictions = clf.predict(new_data)
print("Predictions: ", predictions)
```

Slide 12: Evaluating SVM Performance

To evaluate the performance of the SVM model, we can use metrics like accuracy, precision, recall, and F1-score from scikit-learn.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = [0, 1, 2]  # True labels
y_pred = clf.predict(new_data)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)
```
