## Preventing Overfitting in Decision Trees with CCP
Slide 1: Decision Trees and Overfitting

Decision trees are powerful machine learning algorithms, but they have a tendency to overfit the training data. This means they can capture noise and create complex models that don't generalize well to new data. However, the statement that decision trees always overfit is not entirely accurate. Let's explore this topic and discuss effective techniques to prevent overfitting, including Cost-Complexity Pruning (CCP).

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a decision tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

print(f"Train accuracy: {tree.score(X_train, y_train):.3f}")
print(f"Test accuracy: {tree.score(X_test, y_test):.3f}")
```

Slide 2: Understanding Overfitting in Decision Trees

Overfitting occurs when a model learns the training data too well, including its noise and peculiarities. This results in poor generalization to new, unseen data. Decision trees are prone to overfitting because they can create very complex trees that perfectly fit the training data but fail to capture the underlying patterns.

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(tree, filled=True, feature_names=[f"feature_{i}" for i in range(20)])
plt.title("Complex Decision Tree")
plt.show()

print(f"Number of nodes: {tree.tree_.node_count}")
print(f"Maximum depth: {tree.tree_.max_depth}")
```

Slide 3: Preventing Overfitting: Basic Techniques

There are several basic techniques to prevent overfitting in decision trees:

1. Limiting the maximum depth of the tree
2. Setting a minimum number of samples required to split an internal node
3. Setting a minimum number of samples required to be at a leaf node

Let's implement these techniques and compare the results:

```python
# Create and train a decision tree with constraints
constrained_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=20, min_samples_leaf=5, random_state=42)
constrained_tree.fit(X_train, y_train)

print("Constrained Tree:")
print(f"Train accuracy: {constrained_tree.score(X_train, y_train):.3f}")
print(f"Test accuracy: {constrained_tree.score(X_test, y_test):.3f}")
print(f"Number of nodes: {constrained_tree.tree_.node_count}")
print(f"Maximum depth: {constrained_tree.tree_.max_depth}")
```

Slide 4: Introduction to Cost-Complexity Pruning (CCP)

Cost-Complexity Pruning (CCP) is an effective technique to prevent overfitting in decision trees. It involves creating a series of trees with different levels of complexity and then selecting the optimal tree based on a cost-complexity parameter. This parameter balances the trade-off between tree complexity and its accuracy on the training data.

```python
from sklearn.tree import DecisionTreeClassifier

def print_tree_stats(tree, X_train, y_train, X_test, y_test):
    print(f"Train accuracy: {tree.score(X_train, y_train):.3f}")
    print(f"Test accuracy: {tree.score(X_test, y_test):.3f}")
    print(f"Number of nodes: {tree.tree_.node_count}")
    print(f"Maximum depth: {tree.tree_.max_depth}")

# Create a full tree
full_tree = DecisionTreeClassifier(random_state=42)
path = full_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

print("Full tree statistics:")
full_tree.fit(X_train, y_train)
print_tree_stats(full_tree, X_train, y_train, X_test, y_test)
```

Slide 5: Implementing Cost-Complexity Pruning

To implement CCP, we create multiple trees with different complexity parameters (ccp\_alpha) and evaluate their performance. This allows us to find the optimal balance between model complexity and accuracy.

```python
trees = []
for ccp_alpha in ccp_alphas:
    tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    tree.fit(X_train, y_train)
    trees.append(tree)

train_scores = [tree.score(X_train, y_train) for tree in trees]
test_scores = [tree.score(X_test, y_test) for tree in trees]

plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas[:-1], train_scores[:-1], marker='o', label="train", drawstyle="steps-post")
plt.plot(ccp_alphas[:-1], test_scores[:-1], marker='o', label="test", drawstyle="steps-post")
plt.xlabel("effective alpha")
plt.ylabel("accuracy")
plt.title("Accuracy vs alpha for training and testing sets")
plt.legend()
plt.show()
```

Slide 6: Selecting the Optimal Tree

After creating trees with different complexity parameters, we need to select the optimal tree. This is typically done by choosing the tree with the highest test accuracy or using cross-validation for more robust results.

```python
import numpy as np

# Find the optimal alpha
optimal_alpha = ccp_alphas[np.argmax(test_scores)]

# Create and train the optimal tree
optimal_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=optimal_alpha)
optimal_tree.fit(X_train, y_train)

print("Optimal tree statistics:")
print_tree_stats(optimal_tree, X_train, y_train, X_test, y_test)

plt.figure(figsize=(15,7))
plot_tree(optimal_tree, filled=True, feature_names=[f"feature_{i}" for i in range(20)])
plt.title("Optimal Pruned Decision Tree")
plt.show()
```

Slide 7: Comparing Pruned and Unpruned Trees

Let's compare the performance and structure of the original unpruned tree with the optimal pruned tree obtained through CCP.

```python
print("Original tree statistics:")
print_tree_stats(tree, X_train, y_train, X_test, y_test)

print("\nOptimal pruned tree statistics:")
print_tree_stats(optimal_tree, X_train, y_train, X_test, y_test)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
plot_tree(tree, filled=True, feature_names=[f"feature_{i}" for i in range(20)], ax=ax1)
ax1.set_title("Original Unpruned Tree")
plot_tree(optimal_tree, filled=True, feature_names=[f"feature_{i}" for i in range(20)], ax=ax2)
ax2.set_title("Optimal Pruned Tree")
plt.show()
```

Slide 8: Real-life Example: Predicting Customer Churn

Let's apply CCP to a real-life example of predicting customer churn in a telecommunications company. We'll use a subset of features from the Telco Customer Churn dataset.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the data
df = pd.read_csv('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv')
df = df.drop(['customerID', 'TotalCharges'], axis=1)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

le = LabelEncoder()
for column in df.select_dtypes(include=['object']):
    df[column] = le.fit_transform(df[column])

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply CCP
tree = DecisionTreeClassifier(random_state=42)
path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

trees = []
for ccp_alpha in ccp_alphas:
    tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    tree.fit(X_train, y_train)
    trees.append(tree)

train_scores = [tree.score(X_train, y_train) for tree in trees]
test_scores = [tree.score(X_test, y_test) for tree in trees]

optimal_alpha = ccp_alphas[np.argmax(test_scores)]
optimal_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=optimal_alpha)
optimal_tree.fit(X_train, y_train)

print("Optimal tree statistics for customer churn prediction:")
print_tree_stats(optimal_tree, X_train, y_train, X_test, y_test)
```

Slide 9: Visualizing the Pruned Customer Churn Decision Tree

Let's visualize the pruned decision tree for customer churn prediction to understand the most important factors influencing customer churn.

```python
plt.figure(figsize=(20,10))
plot_tree(optimal_tree, filled=True, feature_names=X.columns, class_names=['No Churn', 'Churn'], rounded=True)
plt.title("Pruned Decision Tree for Customer Churn Prediction")
plt.show()

# Feature importance
importances = optimal_tree.feature_importances_
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.title("Top 10 Important Features for Customer Churn Prediction")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

Slide 10: Real-life Example: Credit Risk Assessment

Another common application of decision trees is in credit risk assessment. Let's use the German Credit Data dataset to predict credit risk and apply CCP.

```python
from sklearn.datasets import fetch_openml

# Load the German Credit Data
X, y = fetch_openml("german", version=1, as_frame=True, return_X_y=True)
y = y.map({'bad': 0, 'good': 1})

# Preprocess the data
for column in X.select_dtypes(include=['object']):
    X[column] = le.fit_transform(X[column])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply CCP
tree = DecisionTreeClassifier(random_state=42)
path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

trees = []
for ccp_alpha in ccp_alphas:
    tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    tree.fit(X_train, y_train)
    trees.append(tree)

train_scores = [tree.score(X_train, y_train) for tree in trees]
test_scores = [tree.score(X_test, y_test) for tree in trees]

optimal_alpha = ccp_alphas[np.argmax(test_scores)]
optimal_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=optimal_alpha)
optimal_tree.fit(X_train, y_train)

print("Optimal tree statistics for credit risk assessment:")
print_tree_stats(optimal_tree, X_train, y_train, X_test, y_test)
```

Slide 11: Analyzing the Credit Risk Assessment Model

Let's visualize the pruned decision tree for credit risk assessment and examine the most important features.

```python
plt.figure(figsize=(20,10))
plot_tree(optimal_tree, filled=True, feature_names=X.columns, class_names=['Bad Credit', 'Good Credit'], rounded=True)
plt.title("Pruned Decision Tree for Credit Risk Assessment")
plt.show()

# Feature importance
importances = optimal_tree.feature_importances_
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.title("Top 10 Important Features for Credit Risk Assessment")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

Slide 12: Limitations and Considerations of CCP

While Cost-Complexity Pruning is an effective technique for preventing overfitting in decision trees, it's important to be aware of its limitations and consider alternatives:

1. CCP may not always find the globally optimal tree.
2. It can be computationally expensive for very large datasets.
3. The optimal alpha value may vary depending on the specific problem and dataset.

Alternative approaches to consider:

* Random Forests: An ensemble method that builds multiple decision trees and averages their predictions.
* Gradient Boosting: Another ensemble method that builds trees sequentially, focusing on correcting errors of previous trees.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train, y_train)

print("Random Forest performance:")
print(f"Train accuracy: {rf_classifier.score(X_train, y_train):.3f}")
print(f"Test accuracy: {rf_classifier.score(X_test, y_test):.3f}")

print("\nGradient Boosting performance:")
print(f"Train accuracy: {gb_classifier.score(X_train, y_train):.3f}")
print(f"Test accuracy: {gb_classifier.score(X_test, y_test):.3f}")
```

Slide 13: Conclusion and Best Practices

In conclusion, while decision trees have a tendency to overfit, they don't always do so. Cost-Complexity Pruning is an effective technique to prevent overfitting, but it's not the only solution. Here are some best practices for working with decision trees:

1. Start with a simple tree and gradually increase complexity.
2. Use cross-validation to estimate the optimal complexity parameter.
3. Consider ensemble methods like Random Forests or Gradient Boosting for improved performance.
4. Always evaluate your model on a separate test set or using cross-validation.
5. Interpret the tree structure and feature importances to gain insights into your data.

```python
from sklearn.model_selection import cross_val_score

# Demonstrate cross-validation
cv_scores = cross_val_score(optimal_tree, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Feature importance comparison
dt_importance = optimal_tree.feature_importances_
rf_importance = rf_classifier.feature_importances_
gb_importance = gb_classifier.feature_importances_

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'Decision Tree': dt_importance,
    'Random Forest': rf_importance,
    'Gradient Boosting': gb_importance
})

feature_importance = feature_importance.melt(id_vars=['feature'], var_name='Model', value_name='Importance')
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='feature', y='Importance', hue='Model', data=feature_importance.head(30))
plt.title("Top 10 Important Features Across Different Models")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Model')
plt.tight_layout()
plt.show()
```

Slide 14: Future Directions and Advanced Techniques

As machine learning continues to evolve, new techniques are being developed to improve decision tree performance and interpretability. Some promising areas of research include:

1. Oblique Decision Trees: These trees use linear combinations of features for splitting, potentially capturing more complex relationships in the data.
2. Fuzzy Decision Trees: Incorporating fuzzy logic to handle uncertainty and imprecision in the decision-making process.
3. Evolutionary Algorithms: Using genetic algorithms to optimize tree structure and parameters.
4. Explainable AI techniques: Developing methods to provide more intuitive explanations for tree decisions, especially in complex ensembles.

```python
# Pseudocode for an Oblique Decision Tree split
def oblique_split(X, y, features):
    best_split = None
    best_score = float('-inf')
    
    for _ in range(num_iterations):
        coefficients = generate_random_coefficients(features)
        split_value = calculate_split_value(X, coefficients)
        left_indices, right_indices = split_data(X, coefficients, split_value)
        score = calculate_split_score(y, left_indices, right_indices)
        
        if score > best_score:
            best_score = score
            best_split = (coefficients, split_value)
    
    return best_split

# Usage in a tree building process
root = Node()
root.split = oblique_split(X, y, feature_indices)
# Continue building the tree recursively
```

Slide 15: Additional Resources

For those interested in diving deeper into decision trees, Cost-Complexity Pruning, and advanced tree-based methods, here are some valuable resources:

1. Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and Regression Trees. CRC Press.
2. Quinlan, J. R. (1986). Induction of Decision Trees. Machine Learning, 1(1), 81-106. ArXiv: [https://arxiv.org/abs/cs/9805004](https://arxiv.org/abs/cs/9805004)
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794). ArXiv: [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
4. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. In Advances in Neural Information Processing Systems (pp. 4765-4774). ArXiv: [https://arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874)

These resources provide a mix of foundational knowledge and cutting-edge research in decision tree algorithms and their applications.

