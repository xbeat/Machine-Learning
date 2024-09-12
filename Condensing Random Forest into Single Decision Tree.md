## Condensing Random Forest into Single Decision Tree
Slide 1: Condensing a Random Forest into a Single Decision Tree

Condensing a random forest model into a single decision tree is a technique used to simplify complex ensemble models while retaining their predictive power. This process involves extracting the most important features and decision rules from the forest to create a more interpretable model.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)

# Train a random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get feature importances from the random forest
feature_importances = rf_model.feature_importances_
```

Slide 2: Feature Importance Analysis

The first step in condensing a random forest is to analyze feature importances. This helps identify the most influential features in the model's decision-making process.

```python
import matplotlib.pyplot as plt

# Sort features by importance
sorted_idx = np.argsort(feature_importances)
sorted_features = [f"Feature {i}" for i in sorted_idx]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, feature_importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.show()

# Select top K features
K = 5
top_k_features = sorted_idx[-K:]
```

Slide 3: Creating a Synthetic Dataset

To condense the random forest, we'll create a synthetic dataset based on the predictions of the original model. This dataset will capture the decision boundaries of the random forest.

```python
# Generate synthetic data
n_synthetic = 10000
X_synthetic = np.random.rand(n_synthetic, X.shape[1])

# Get predictions from the random forest
y_synthetic = rf_model.predict(X_synthetic)

# Create new dataset with only top K features
X_synthetic_reduced = X_synthetic[:, top_k_features]
```

Slide 4: Training a Single Decision Tree

Using the synthetic dataset, we'll train a single decision tree that approximates the behavior of the random forest.

```python
# Train a decision tree on the synthetic data
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_synthetic_reduced, y_synthetic)

# Evaluate the condensed model
from sklearn.metrics import accuracy_score

y_pred_rf = rf_model.predict(X)
y_pred_dt = dt_model.predict(X[:, top_k_features])

print(f"Random Forest Accuracy: {accuracy_score(y, y_pred_rf):.4f}")
print(f"Condensed Decision Tree Accuracy: {accuracy_score(y, y_pred_dt):.4f}")
```

Slide 5: Visualizing the Condensed Decision Tree

To better understand the condensed model, we can visualize the decision tree structure.

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=[f"Feature {i}" for i in top_k_features], filled=True, rounded=True)
plt.title("Condensed Decision Tree")
plt.show()
```

Slide 6: Extracting Rules from the Condensed Tree

We can extract decision rules from the condensed tree to gain insights into the model's decision-making process.

```python
def extract_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]
    
    def recurse(node, depth, rules):
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_rule = f"{name} <= {threshold:.2f}"
            right_rule = f"{name} > {threshold:.2f}"
            recurse(tree_.children_left[node], depth + 1, rules + [left_rule])
            recurse(tree_.children_right[node], depth + 1, rules + [right_rule])
        else:
            print("Rule:", " AND ".join(rules), f"-> class {np.argmax(tree_.value[node])}")

    recurse(0, 1, [])

extract_rules(dt_model, [f"Feature {i}" for i in top_k_features])
```

Slide 7: Handling Categorical Features

When dealing with categorical features, we need to modify our approach slightly to ensure proper encoding and interpretation.

```python
from sklearn.preprocessing import OneHotEncoder

# Assume we have a categorical feature
X_cat = np.random.choice(['A', 'B', 'C'], size=(1000, 1))
X_num = np.random.rand(1000, 5)
X_combined = np.hstack((X_num, X_cat))

# One-hot encode the categorical feature
encoder = OneHotEncoder(sparse=False)
X_cat_encoded = encoder.fit_transform(X_cat)

# Combine numerical and encoded categorical features
X_processed = np.hstack((X_num, X_cat_encoded))

# Train the random forest and condense as before
# ...
```

Slide 8: Handling Imbalanced Datasets

When condensing a random forest trained on imbalanced data, we need to consider class weights and sampling techniques.

```python
from sklearn.utils.class_weight import compute_sample_weight

# Assume y is imbalanced
class_weights = compute_sample_weight('balanced', y)

# Train random forest with class weights
rf_model_balanced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_balanced.fit(X, y, sample_weight=class_weights)

# Generate synthetic data
y_synthetic_balanced = rf_model_balanced.predict(X_synthetic)

# Train condensed decision tree with class weights
dt_model_balanced = DecisionTreeClassifier(random_state=42)
dt_model_balanced.fit(X_synthetic_reduced, y_synthetic_balanced, 
                      sample_weight=compute_sample_weight('balanced', y_synthetic_balanced))
```

Slide 9: Evaluating Model Performance

To ensure our condensed model performs well, we should evaluate it using appropriate metrics and techniques.

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

# Perform cross-validation
cv_scores_rf = cross_val_score(rf_model, X, y, cv=5)
cv_scores_dt = cross_val_score(dt_model, X[:, top_k_features], y, cv=5)

print("Random Forest CV Scores:", cv_scores_rf)
print("Condensed Decision Tree CV Scores:", cv_scores_dt)

# Generate confusion matrix
cm_rf = confusion_matrix(y, y_pred_rf)
cm_dt = confusion_matrix(y, y_pred_dt)

# Print classification report
print("Random Forest Classification Report:")
print(classification_report(y, y_pred_rf))
print("\nCondensed Decision Tree Classification Report:")
print(classification_report(y, y_pred_dt))
```

Slide 10: Interpreting the Condensed Model

Understanding the condensed model's decision-making process is crucial for gaining insights and explaining predictions.

```python
from sklearn.inspection import PartialDependenceDisplay

# Calculate and plot partial dependence for top features
fig, ax = plt.subplots(figsize=(12, 6))
PartialDependenceDisplay.from_estimator(dt_model, X[:, top_k_features], 
                                        features=list(range(K)), ax=ax)
plt.title("Partial Dependence Plots for Condensed Decision Tree")
plt.show()

# Get a sample prediction and explain it
sample_index = 0
sample_prediction = dt_model.predict(X[sample_index:sample_index+1, top_k_features])[0]
sample_path = dt_model.decision_path(X[sample_index:sample_index+1, top_k_features])

print(f"Prediction for sample {sample_index}: Class {sample_prediction}")
print("Decision path:")
for node in sample_path.indices:
    if node == dt_model.tree_.children_left[sample_path.indices[0]]:
        print(f"Feature {dt_model.tree_.feature[node]} <= {dt_model.tree_.threshold[node]:.2f}")
    elif node == dt_model.tree_.children_right[sample_path.indices[0]]:
        print(f"Feature {dt_model.tree_.feature[node]} > {dt_model.tree_.threshold[node]:.2f}")
```

Slide 11: Real-Life Example: Image Classification

In image classification tasks, condensing a random forest can help identify key features for classification while reducing model complexity.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load digit recognition dataset
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.2, random_state=42)

# Train random forest
rf_digits = RandomForestClassifier(n_estimators=100, random_state=42)
rf_digits.fit(X_train, y_train)

# Condense the model (using previous techniques)
# ...

# Visualize important features
importances = rf_digits.feature_importances_
pixel_importances = importances.reshape(8, 8)

plt.imshow(pixel_importances, cmap='viridis')
plt.colorbar()
plt.title("Pixel Importances in Digit Recognition")
plt.show()
```

Slide 12: Real-Life Example: Customer Churn Prediction

Predicting customer churn is a common application where condensing a random forest can provide interpretable insights into key factors affecting customer retention.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Assume we have a customer churn dataset
churn_data = pd.DataFrame({
    'age': np.random.randint(18, 80, 1000),
    'tenure': np.random.randint(0, 10, 1000),
    'num_products': np.random.randint(1, 5, 1000),
    'is_active': np.random.choice([0, 1], 1000),
    'churn': np.random.choice([0, 1], 1000)
})

# Preprocess the data
X_churn = churn_data.drop('churn', axis=1)
y_churn = churn_data['churn']

scaler = StandardScaler()
X_churn_scaled = scaler.fit_transform(X_churn)

# Train random forest
rf_churn = RandomForestClassifier(n_estimators=100, random_state=42)
rf_churn.fit(X_churn_scaled, y_churn)

# Condense the model (using previous techniques)
# ...

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.bar(X_churn.columns, rf_churn.feature_importances_)
plt.title("Feature Importances in Customer Churn Prediction")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()
```

Slide 13: Limitations and Considerations

While condensing a random forest into a single decision tree can improve interpretability, it's important to consider potential limitations:

1. Loss of ensemble benefits: The condensed model may not capture all the nuances of the original random forest.
2. Reduced accuracy: The simplified model might have slightly lower predictive performance.
3. Feature interactions: Complex interactions between features may be lost in the condensation process.
4. Dataset size: The synthetic dataset size can affect the quality of the condensed model.
5. Hyperparameter sensitivity: The condensed tree's performance may be sensitive to its hyperparameters.

To mitigate these limitations, consider:

1. Experimenting with different synthetic dataset sizes
2. Tuning the hyperparameters of the condensed decision tree
3. Using ensemble methods on the condensed trees from multiple random forests
4. Comparing the condensed model's performance with other interpretable models

```python
# Example of creating multiple condensed trees and ensembling them
n_condensed_trees = 5
condensed_trees = []

for _ in range(n_condensed_trees):
    # Generate new synthetic data
    X_synthetic = np.random.rand(n_synthetic, X.shape[1])
    y_synthetic = rf_model.predict(X_synthetic)
    X_synthetic_reduced = X_synthetic[:, top_k_features]
    
    # Train a new condensed tree
    dt_model = DecisionTreeClassifier(random_state=np.random.randint(1000))
    dt_model.fit(X_synthetic_reduced, y_synthetic)
    condensed_trees.append(dt_model)

# Make predictions using the ensemble of condensed trees
y_pred_ensemble = np.mean([tree.predict(X[:, top_k_features]) for tree in condensed_trees], axis=0)
y_pred_ensemble = (y_pred_ensemble > 0.5).astype(int)

print(f"Ensemble of Condensed Trees Accuracy: {accuracy_score(y, y_pred_ensemble):.4f}")
```

Slide 14: Additional Resources

For further exploration of random forest condensation and related topics, consider the following resources:

1. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32. ArXiv: [https://arxiv.org/abs/1011.1669v1](https://arxiv.org/abs/1011.1669v1)
2. Friedman, J. H., & Popescu, B. E. (2008). Predictive learning via rule ensembles. The Annals of Applied Statistics, 2(3), 916-954. ArXiv: [https://arxiv.org/abs/0811.1679](https://arxiv.org/abs/0811.1679)
3. Molnar, C. (2019). Interpretable Machine Learning: A Guide for Making Black Box Models Explainable. Available at: [https://christophm.github.io/interpretable-ml-book/](https://christophm.github.io/interpretable-ml-book/)

These resources provide in-depth discussions on random forests, rule extraction, and model interpretability, which can help deepen your understanding of the condensation process and its applications in various domains.

