## Building Decision Trees from Scratch in Python
Slide 1: Introduction to Decision Trees

Decision trees are powerful machine learning models used for both classification and regression tasks. They work by recursively splitting the data based on features to create a tree-like structure for making predictions. In this presentation, we'll explore how to build decision trees from scratch using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([[1, 2], [2, 3], [3, 1], [4, 4], [5, 5]])
y = np.array([0, 0, 1, 1, 1])

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Sample Data for Decision Tree')
plt.show()
```

Slide 2: Node Class

We'll start by defining a Node class to represent each node in our decision tree. Each node will store information about the split condition, prediction, and child nodes.

```python
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
```

Slide 3: Decision Tree Class

Next, we'll create a DecisionTree class that will handle the tree construction and prediction processes. We'll start with the class initialization and the fit method.

```python
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        # Tree growing logic will be implemented here
        pass
```

Slide 4: Splitting Criteria

To grow our decision tree, we need a way to evaluate the quality of splits. We'll implement the Gini impurity as our splitting criteria for classification tasks.

```python
def _gini_impurity(self, y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

def _information_gain(self, parent, left_child, right_child):
    weight_left = len(left_child) / len(parent)
    weight_right = len(right_child) / len(parent)
    return (self._gini_impurity(parent) - 
            (weight_left * self._gini_impurity(left_child) + 
             weight_right * self._gini_impurity(right_child)))
```

Slide 5: Finding the Best Split

We'll implement a method to find the best split for a given dataset by iterating through all features and possible thresholds.

```python
def _best_split(self, X, y):
    best_gain = -1
    best_feature, best_threshold = None, None

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            gain = self._information_gain(y, y[left_mask], y[right_mask])
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold
```

Slide 6: Growing the Tree

Now we'll implement the \_grow\_tree method to recursively build our decision tree.

```python
def _grow_tree(self, X, y, depth=0):
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    # Stopping criteria
    if (depth == self.max_depth or n_samples < 2 or n_classes == 1):
        return Node(value=np.argmax(np.bincount(y)))

    feature, threshold = self._best_split(X, y)

    # Create child nodes
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
    right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

    return Node(feature=feature, threshold=threshold, left=left, right=right)
```

Slide 7: Making Predictions

We'll add methods to our DecisionTree class for making predictions on new data.

```python
def predict(self, X):
    return np.array([self._traverse_tree(x, self.root) for x in X])

def _traverse_tree(self, x, node):
    if node.value is not None:
        return node.value
    
    if x[node.feature] <= node.threshold:
        return self._traverse_tree(x, node.left)
    else:
        return self._traverse_tree(x, node.right)
```

Slide 8: Visualizing the Decision Tree

To better understand our decision tree, let's create a method to visualize it.

```python
def visualize_tree(self, node, depth=0):
    if node.value is not None:
        print('  ' * depth + f'Prediction: {node.value}')
    else:
        print('  ' * depth + f'Feature {node.feature} <= {node.threshold}')
        self.visualize_tree(node.left, depth + 1)
        self.visualize_tree(node.right, depth + 1)

# Usage
tree = DecisionTree(max_depth=3)
tree.fit(X, y)
tree.visualize_tree(tree.root)
```

Slide 9: Real-Life Example: Iris Dataset

Let's apply our decision tree to the famous Iris dataset for flower classification.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)

predictions = tree.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy:.2f}')

tree.visualize_tree(tree.root)
```

Slide 10: Handling Continuous and Categorical Features

In practice, we often encounter datasets with both continuous and categorical features. Let's modify our decision tree to handle both types.

```python
def _best_split(self, X, y):
    best_gain = -1
    best_feature, best_threshold = None, None

    for feature in range(X.shape[1]):
        if np.issubdtype(X[:, feature].dtype, np.number):  # Continuous feature
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        else:  # Categorical feature
            unique_values = np.unique(X[:, feature])
            for value in unique_values:
                left_mask = X[:, feature] == value
                right_mask = ~left_mask
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = value

    return best_feature, best_threshold
```

Slide 11: Pruning the Decision Tree

To prevent overfitting, we can implement post-pruning using reduced error pruning.

```python
def prune(self, X_val, y_val):
    def _prune_recursive(node):
        if node.left is None and node.right is None:
            return

        _prune_recursive(node.left)
        _prune_recursive(node.right)

        # Try merging the nodes
        original_left, original_right = node.left, node.right
        original_value = node.value
        node.left = node.right = None
        node.value = np.argmax(np.bincount(y_val))

        if self._accuracy(X_val, y_val) >= original_accuracy:
            return
        else:
            node.left, node.right = original_left, original_right
            node.value = original_value

    original_accuracy = self._accuracy(X_val, y_val)
    _prune_recursive(self.root)

def _accuracy(self, X, y):
    predictions = self.predict(X)
    return np.mean(predictions == y)
```

Slide 12: Real-Life Example: Mushroom Classification

Let's apply our improved decision tree to classify mushrooms as edible or poisonous based on their characteristics.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the mushroom dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
columns = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor"]
data = pd.read_csv(url, names=columns)

# Encode categorical variables
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])

X = data.drop("class", axis=1).values
y = data["class"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)

accuracy = tree._accuracy(X_test, y_test)
print(f"Accuracy before pruning: {accuracy:.2f}")

tree.prune(X_test, y_test)
accuracy_pruned = tree._accuracy(X_test, y_test)
print(f"Accuracy after pruning: {accuracy_pruned:.2f}")

tree.visualize_tree(tree.root)
```

Slide 13: Limitations and Future Improvements

While our decision tree implementation works well for simple datasets, there are several areas for improvement:

1. Handling missing values
2. Implementing ensemble methods like Random Forests
3. Supporting multi-output problems
4. Optimizing performance for large datasets

These enhancements would make our decision tree more robust and suitable for a wider range of real-world applications.

Slide 14: Additional Resources

For those interested in diving deeper into decision trees and machine learning algorithms, here are some valuable resources:

1. "Decision Trees and Forests: A Probabilistic Perspective" by Balaji Lakshminarayanan, Daniel M. Roy, Yee Whye Teh ([https://arxiv.org/abs/1604.07900](https://arxiv.org/abs/1604.07900))
2. "Random Forests" by Leo Breiman ([https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf))
3. "Introduction to Statistical Learning" by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani ([https://www.statlearning.com/](https://www.statlearning.com/))

These resources provide in-depth explanations of decision tree algorithms, their theoretical foundations, and advanced techniques for improving their performance.

