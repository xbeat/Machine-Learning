## Improving Tree-Based Modeling with Python

Slide 1: Introduction to Tree-Based Models

Tree-based models are powerful machine learning techniques used for both classification and regression tasks. They work by partitioning the feature space into regions and making predictions based on these partitions. In this slideshow, we'll explore how to improve your tree-based modeling skills using Python.

```python
# Simple decision tree example
class DecisionNode:
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold
        self.left = None
        self.right = None

def predict(node, sample):
    if node.left is None and node.right is None:
        return node.prediction
    if sample[node.feature] <= node.threshold:
        return predict(node.left, sample)
    else:
        return predict(node.right, sample)

# Usage
root = DecisionNode(feature=0, threshold=0.5)
root.left = DecisionNode(feature=1, threshold=0.3)
root.right = DecisionNode(feature=1, threshold=0.7)
```

Slide 2: Understanding Decision Trees

Decision trees are the foundation of tree-based models. They make decisions by asking a series of questions about the features, splitting the data into subsets. Each internal node represents a decision based on a feature, while leaf nodes represent the final prediction.

```python
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        # Implementation of tree growing algorithm
        pass

    def predict(self, X):
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x, node):
        # Implementation of tree traversal
        pass
```

Slide 3: Feature Importance in Decision Trees

Feature importance helps us understand which features contribute most to the model's decisions. In decision trees, importance is often calculated based on how much each feature decreases the weighted impurity.

```python
def calculate_feature_importance(tree):
    importance = [0] * tree.n_features
    
    def traverse(node, importance):
        if node.feature is not None:
            importance[node.feature] += node.impurity_decrease
            traverse(node.left, importance)
            traverse(node.right, importance)
    
    traverse(tree.root, importance)
    return importance / sum(importance)

# Usage
tree = DecisionTree()
tree.fit(X, y)
importances = calculate_feature_importance(tree)
```

Slide 4: Pruning Decision Trees

Pruning is a technique used to reduce the complexity of decision trees and prevent overfitting. It involves removing branches that provide little predictive power, based on a validation set or a cost-complexity criterion.

```python
def prune_tree(node, X_val, y_val):
    if node.left is None and node.right is None:
        return node

    # Recursively prune children
    if node.left:
        node.left = prune_tree(node.left, X_val, y_val)
    if node.right:
        node.right = prune_tree(node.right, X_val, y_val)

    # Check if pruning improves performance
    error_before = calculate_error(node, X_val, y_val)
    temp_prediction = majority_vote(y_val)
    error_after = calculate_error_if_leaf(node, X_val, y_val, temp_prediction)

    if error_after <= error_before:
        return DecisionNode(prediction=temp_prediction)
    else:
        return node
```

Slide 5: Random Forests

Random Forests are an ensemble method that combines multiple decision trees to create a more robust and accurate model. Each tree is trained on a random subset of the data and features, reducing overfitting and improving generalization.

```python
class RandomForest:
    def __init__(self, n_trees=100, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)
```

Slide 6: Feature Selection with Random Forests

Random Forests can be used for feature selection by ranking features based on their importance across all trees in the forest. This can help identify the most relevant features for your model.

```python
def random_forest_feature_selection(X, y, n_trees=100, max_features=0.5):
    forest = RandomForest(n_trees=n_trees)
    forest.fit(X, y)
    
    importances = np.zeros(X.shape[1])
    for tree in forest.trees:
        importances += calculate_feature_importance(tree)
    
    importances /= n_trees
    return importances

# Usage
X, y = load_data()
feature_importances = random_forest_feature_selection(X, y)
selected_features = np.argsort(feature_importances)[::-1][:10]  # Top 10 features
```

Slide 7: Gradient Boosting Trees

Gradient Boosting Trees is another ensemble method that builds trees sequentially, with each tree trying to correct the errors of the previous ones. This often results in highly accurate models.

```python
class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        residuals = y - self.initial_prediction

        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=3)  # Weak learner
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions

    def predict(self, X):
        return self.initial_prediction + sum(
            self.learning_rate * tree.predict(X) for tree in self.trees
        )
```

Slide 8: Hyperparameter Tuning

Proper hyperparameter tuning is crucial for optimizing tree-based models. Common hyperparameters include max\_depth, min\_samples\_split, and n\_estimators. We'll implement a simple grid search for tuning.

```python
def grid_search(model_class, param_grid, X, y, cv=5):
    best_score = float('-inf')
    best_params = None

    for params in itertools.product(*param_grid.values()):
        cv_scores = []
        params_dict = dict(zip(param_grid.keys(), params))
        
        for train_idx, val_idx in KFold(n_splits=cv).split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = model_class(**params_dict)
            model.fit(X_train, y_train)
            score = calculate_score(model, X_val, y_val)
            cv_scores.append(score)
        
        mean_score = np.mean(cv_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params_dict

    return best_params, best_score
```

Slide 9: Handling Imbalanced Data

Imbalanced datasets can be challenging for tree-based models. We'll implement a technique called SMOTE (Synthetic Minority Over-sampling Technique) to address this issue.

```python
def smote(X, y, k=5, ratio=1.0):
    minority_class = np.where(y == 1)[0]
    majority_class = np.where(y == 0)[0]
    
    if len(minority_class) >= len(majority_class):
        return X, y
    
    n_synthetic = int(len(majority_class) * ratio) - len(minority_class)
    synthetic_samples = []
    
    for _ in range(n_synthetic):
        idx = np.random.choice(minority_class)
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X[minority_class])
        _, neighbors = nn.kneighbors(X[idx].reshape(1, -1))
        nn_idx = minority_class[np.random.choice(neighbors[0])]
        
        dif = X[nn_idx] - X[idx]
        gap = np.random.random()
        synthetic_sample = X[idx] + gap * dif
        synthetic_samples.append(synthetic_sample)
    
    X_resampled = np.vstack([X, synthetic_samples])
    y_resampled = np.hstack([y, np.ones(n_synthetic)])
    
    return X_resampled, y_resampled
```

Slide 10: Feature Engineering for Tree-Based Models

Feature engineering can significantly improve the performance of tree-based models. We'll implement some common techniques like polynomial features and interaction terms.

```python
def create_polynomial_features(X, degree=2):
    n_samples, n_features = X.shape
    poly_features = []
    
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_features), d):
            new_feature = np.prod(X[:, combo], axis=1)
            poly_features.append(new_feature)
    
    return np.column_stack(poly_features)

def create_interaction_terms(X):
    n_samples, n_features = X.shape
    interactions = []
    
    for i, j in itertools.combinations(range(n_features), 2):
        interaction = X[:, i] * X[:, j]
        interactions.append(interaction)
    
    return np.column_stack(interactions)

# Usage
X_poly = create_polynomial_features(X, degree=3)
X_interact = create_interaction_terms(X)
X_enhanced = np.hstack([X, X_poly, X_interact])
```

Slide 11: Interpreting Tree-Based Models

Interpreting complex tree-based models can be challenging. We'll implement a simple partial dependence plot to visualize the relationship between a feature and the target variable.

```python
def partial_dependence_plot(model, X, feature_idx, n_points=100):
    feature_values = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), n_points)
    pdp_values = []
    
    for value in feature_values:
        X_modified = X.copy()
        X_modified[:, feature_idx] = value
        predictions = model.predict(X_modified)
        pdp_values.append(np.mean(predictions))
    
    return feature_values, pdp_values

# Usage
model = RandomForest()
model.fit(X, y)
feature_idx = 0  # Feature to plot
x, y = partial_dependence_plot(model, X, feature_idx)

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.xlabel(f'Feature {feature_idx}')
plt.ylabel('Partial Dependence')
plt.title(f'Partial Dependence Plot for Feature {feature_idx}')
plt.show()
```

Slide 12: Handling Missing Data

Tree-based models can handle missing data in various ways. We'll implement a simple approach using surrogate splits.

```python
class DecisionTreeWithMissing:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree_with_surrogates(X, y, depth=0)

    def _grow_tree_with_surrogates(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return DecisionNode(prediction=np.mean(y))

        best_feature, best_threshold = self._find_best_split(X, y)
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        surrogate_features = self._find_surrogate_splits(X, left_mask)

        node = DecisionNode(feature=best_feature, threshold=best_threshold)
        node.surrogates = surrogate_features
        node.left = self._grow_tree_with_surrogates(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree_with_surrogates(X[right_mask], y[right_mask], depth + 1)

        return node

    def _find_surrogate_splits(self, X, left_mask):
        surrogates = []
        for feature in range(X.shape[1]):
            if feature == self.feature:
                continue
            threshold = self._find_best_threshold(X[:, feature], left_mask)
            surrogates.append((feature, threshold))
        return sorted(surrogates, key=lambda x: self._surrogate_agreement(X, x[0], x[1], left_mask), reverse=True)

    def _surrogate_agreement(self, X, feature, threshold, left_mask):
        surrogate_mask = X[:, feature] <= threshold
        return np.sum(surrogate_mask == left_mask) / len(left_mask)

    def predict(self, X):
        return np.array([self._traverse_tree_with_surrogates(x, self.root) for x in X])

    def _traverse_tree_with_surrogates(self, x, node):
        if node.left is None and node.right is None:
            return node.prediction

        if np.isnan(x[node.feature]):
            for surrogate_feature, surrogate_threshold in node.surrogates:
                if not np.isnan(x[surrogate_feature]):
                    if x[surrogate_feature] <= surrogate_threshold:
                        return self._traverse_tree_with_surrogates(x, node.left)
                    else:
                        return self._traverse_tree_with_surrogates(x, node.right)
            # If all surrogates are missing, go left (arbitrary choice)
            return self._traverse_tree_with_surrogates(x, node.left)
        else:
            if x[node.feature] <= node.threshold:
                return self._traverse_tree_with_surrogates(x, node.left)
            else:
                return self._traverse_tree_with_surrogates(x, node.right)
```

Slide 13: Real-Life Example: Predicting Customer Churn

Let's apply our tree-based modeling skills to predict customer churn for a telecommunications company. We'll use a Random Forest model and implement feature importance analysis.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Simulated telecom customer data
np.random.seed(42)
n_samples = 1000
X = np.random.rand(n_samples, 5)  # 5 features: usage, contract_length, customer_service_calls, age, tenure
y = (X[:, 0] * 0.3 + X[:, 1] * 0.2 - X[:, 2] * 0.4 + X[:, 3] * 0.1 + X[:, 4] * 0.3 + np.random.normal(0, 0.1, n_samples)) > 0.5

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForest(n_trees=100, max_depth=5)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate feature importance
feature_importance = np.mean([tree.feature_importance() for tree in rf.trees], axis=0)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Feature Importance:")
for i, imp in enumerate(feature_importance):
    print(f"Feature {i}: {imp}")
```

Slide 14: Real-Life Example: Predicting Plant Species

In this example, we'll use a Decision Tree to classify iris flowers based on their sepal and petal measurements. This demonstrates how tree-based models can be applied to biological classification tasks.

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree
dt = DecisionTree(max_depth=5)
dt.fit(X_train, y_train)

# Predict and evaluate
y_pred = dt.predict(X_test)
accuracy = sum(y_pred == y_test) / len(y_test)

print(f"Accuracy: {accuracy}")

# Visualize the tree structure (pseudo-code)
def print_tree(node, depth=0):
    if node.is_leaf():
        print("  " * depth + f"Prediction: {node.prediction}")
    else:
        print("  " * depth + f"Feature {node.feature} <= {node.threshold}")
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)

print("Decision Tree Structure:")
print_tree(dt.root)
```

Slide 15: Additional Resources

For further exploration of tree-based modeling techniques, consider the following resources:

1.  "Random Forests" by Leo Breiman (2001): A seminal paper introducing Random Forests. Available at: [https://arxiv.org/abs/2001.03606](https://arxiv.org/abs/2001.03606)
2.  "Gradient Boosting Machines: A Tutorial" by Alexey Natekin and Alois Knoll (2013): A comprehensive overview of gradient boosting. Available at: [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
3.  "XGBoost: A Scalable Tree Boosting System" by Tianqi Chen and Carlos Guestrin (2016): Introduces the popular XGBoost algorithm. Available at: [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)

These papers provide in-depth insights into advanced tree-based modeling techniques and their applications.

