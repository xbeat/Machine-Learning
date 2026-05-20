## Visualizing Decision Trees in Scikit-learn
Slide 1: Basic Decision Tree Visualization

The sklearn.tree.plot\_tree function provides a fundamental way to visualize decision trees by creating a graphical representation of the tree structure, showing decision nodes, leaf nodes, split conditions, and class distributions in a hierarchical layout.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, 100)

# Create and train decision tree
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# Visualize the tree
plt.figure(figsize=(15,10))
plot_tree(clf, filled=True, feature_names=['feature_1', 'feature_2'])
plt.show()
```

Slide 2: Enhanced Tree Visualization with Feature Importance

Decision tree visualization becomes more informative when combined with feature importance analysis, allowing us to understand which variables have the most significant impact on the model's decisions through color-coded nodes and importance scores.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

# Load iris dataset
iris = load_iris()
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(iris.data, iris.target)

# Create visualization with feature importances
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=iris.feature_names, 
          class_names=iris.target_names,
          filled=True, impurity=True, 
          rounded=True)

# Add feature importance bar plot
importances = clf.feature_importances_
plt.figure(figsize=(10,5))
plt.bar(iris.feature_names, importances)
plt.title('Feature Importances')
plt.xticks(rotation=45)
plt.show()
```

Slide 3: Customizing Decision Tree Visualization

Advanced customization options in plot\_tree allow for detailed control over the visual representation, including node colors, text properties, and tree layout, making it possible to create more interpretable and presentation-ready visualizations.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np

# Create and train model
X = np.random.rand(100, 3)
y = np.random.randint(0, 2, 100)
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

# Custom visualization
plt.figure(figsize=(20,10))
plot_tree(clf,
          feature_names=['X1', 'X2', 'X3'],
          class_names=['Class 0', 'Class 1'],
          filled=True,
          rounded=True,
          fontsize=14,
          precision=3,
          proportion=True,
          impurity=True)
plt.savefig('custom_tree.png', dpi=300, bbox_inches='tight')
plt.show()
```

Slide 4: Implementing Tree Visualization for Large Datasets

When dealing with large datasets, decision tree visualization requires special handling to maintain readability and performance, including depth limitation, node sampling, and output size optimization techniques.

```python
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Generate large dataset
X, y = make_classification(n_samples=1000, n_features=20,
                          n_informative=15, n_redundant=5,
                          random_state=42)

# Train complex tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Optimize visualization for large tree
plt.figure(figsize=(30,15))
plot_tree(clf, 
          max_depth=4,  # Limit depth for visibility
          feature_names=[f'F{i}' for i in range(20)],
          filled=True,
          fontsize=10,
          impurity=False,  # Simplify node information
          proportion=True)
plt.savefig('large_tree.png', dpi=300, bbox_inches='tight')
```

Slide 5: Real-world Example: Credit Risk Assessment

Implementation of decision tree visualization for credit risk assessment, demonstrating practical application in financial domain with actual preprocessing steps and interpretation of results.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Prepare credit data (example dataset)
data = pd.DataFrame({
    'income': np.random.normal(50000, 20000, 1000),
    'age': np.random.randint(18, 70, 1000),
    'credit_score': np.random.randint(300, 850, 1000),
    'default': np.random.randint(0, 2, 1000)
})

# Preprocess data
X = data.drop('default', axis=1)
y = data['default']

# Train model
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)

# Visualize with domain-specific formatting
plt.figure(figsize=(20,10))
plot_tree(clf,
          feature_names=X.columns,
          class_names=['Good', 'Default'],
          filled=True,
          rounded=True,
          precision=0,
          proportion=True)
plt.title('Credit Risk Decision Tree')
plt.show()
```

Slide 6: Results Analysis for Credit Risk Assessment

The decision tree visualization for credit risk assessment reveals critical decision paths and threshold values that determine credit worthiness, with node colors indicating risk levels and probability distributions at leaf nodes.

```python
# Analyze and visualize results from previous credit risk model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Generate predictions
y_pred = clf.predict(X)

# Create confusion matrix visualization
plt.figure(figsize=(10,8))
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Credit Risk Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Display classification metrics
print("Classification Report:")
print(classification_report(y, y_pred))

# Display feature importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(importance_df)
```

Slide 7: Multi-Class Decision Tree Visualization

Visualizing multi-class decision trees requires special consideration for color schemes and node labeling to effectively represent multiple class decisions and probability distributions across different categories.

```python
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load wine dataset
wine = load_wine()
X_wine = wine.data
y_wine = wine.target

# Train multi-class model
clf_wine = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_wine.fit(X_wine, y_wine)

# Create multi-class visualization
plt.figure(figsize=(20,12))
plot_tree(clf_wine,
          feature_names=wine.feature_names,
          class_names=wine.target_names,
          filled=True,
          rounded=True,
          proportion=True)
plt.title('Multi-class Wine Classification Tree')

# Add feature importance subplot
plt.figure(figsize=(12,6))
importances = pd.DataFrame({
    'feature': wine.feature_names,
    'importance': clf_wine.feature_importances_
}).sort_values('importance', ascending=False)
plt.bar(importances['feature'], importances['importance'])
plt.xticks(rotation=45)
plt.title('Feature Importance in Wine Classification')
plt.tight_layout()
plt.show()
```

Slide 8: Interactive Tree Visualization with Graphviz

Enhanced visualization using Graphviz integration provides interactive capabilities and export options for decision trees, allowing for detailed exploration of complex tree structures and decision paths.

```python
from sklearn.tree import export_graphviz
import graphviz

# Create and train a decision tree
X = np.random.rand(100, 4)
y = np.random.randint(0, 3, 100)
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)

# Export tree to Graphviz
dot_data = export_graphviz(
    clf,
    feature_names=['Feature_'+str(i) for i in range(4)],
    class_names=['Class_'+str(i) for i in range(3)],
    filled=True,
    rounded=True,
    special_characters=True
)

# Create interactive visualization
graph = graphviz.Source(dot_data)
graph.render("decision_tree_graphviz", format="png", cleanup=True)

# Display additional statistics
print("Tree Statistics:")
print(f"Number of nodes: {clf.tree_.node_count}")
print(f"Tree depth: {clf.get_depth()}")
print(f"Number of leaves: {clf.get_n_leaves()}")
```

Slide 9: Real-world Example: Medical Diagnosis Visualization

Implementation of decision tree visualization for medical diagnosis prediction, showcasing how to handle sensitive medical data and create interpretable visualizations for healthcare professionals.

```python
# Generate synthetic medical data
np.random.seed(42)
n_samples = 1000

medical_data = pd.DataFrame({
    'age': np.random.normal(60, 15, n_samples),
    'blood_pressure': np.random.normal(130, 20, n_samples),
    'glucose': np.random.normal(100, 25, n_samples),
    'bmi': np.random.normal(25, 5, n_samples)
})

# Create diagnosis based on conditions
medical_data['diagnosis'] = ((medical_data['blood_pressure'] > 140) & 
                           (medical_data['glucose'] > 126) & 
                           (medical_data['bmi'] > 30)).astype(int)

# Train and visualize medical decision tree
X_med = medical_data.drop('diagnosis', axis=1)
y_med = medical_data['diagnosis']

med_clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50)
med_clf.fit(X_med, y_med)

plt.figure(figsize=(20,12))
plot_tree(med_clf,
          feature_names=X_med.columns,
          class_names=['Healthy', 'At Risk'],
          filled=True,
          rounded=True,
          proportion=True,
          precision=1)
plt.title('Medical Diagnosis Decision Tree')
plt.show()

# Display risk factors importance
importances = pd.DataFrame({
    'factor': X_med.columns,
    'importance': med_clf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nRisk Factor Importance:")
print(importances)
```

Slide 10: Pruning Visualization Techniques

Decision tree pruning visualization helps identify optimal tree complexity by showing the impact of different pruning techniques on tree structure and performance metrics through comparative visual analysis.

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
X = np.random.rand(1000, 5)
y = np.random.randint(0, 2, 1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create trees with different pruning parameters
depths = [2, 3, 4, 5]
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
axes = axes.ravel()

for idx, depth in enumerate(depths):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    
    plot_tree(clf, 
              feature_names=[f'F{i}' for i in range(5)],
              filled=True,
              ax=axes[idx])
    axes[idx].set_title(f'Max Depth = {depth}')

plt.tight_layout()
plt.show()

# Plot accuracy vs tree depth
depths_extended = range(1, 11)
train_scores = []
test_scores = []

for depth in depths_extended:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(depths_extended, train_scores, label='Train Score')
plt.plot(depths_extended, test_scores, label='Test Score')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Tree Depth')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 11: Ensemble Tree Visualization

Visualization techniques for ensemble methods like Random Forests, showing how individual trees contribute to the overall model and highlighting the most important decision paths across multiple trees.

```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

# Create and train random forest
X = np.random.rand(1000, 4)
y = np.random.randint(0, 2, 1000)
rf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
rf.fit(X, y)

# Visualize multiple trees
fig, axes = plt.subplots(1, 3, figsize=(25, 8))
for idx, tree in enumerate(rf.estimators_):
    plot_tree(tree,
              feature_names=[f'Feature {i}' for i in range(4)],
              filled=True,
              ax=axes[idx])
    axes[idx].set_title(f'Tree {idx+1}')

plt.tight_layout()
plt.show()

# Feature importance across ensemble
importance_df = pd.DataFrame({
    'feature': [f'Feature {i}' for i in range(4)],
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(importance_df['feature'], importance_df['importance'])
plt.title('Feature Importance Across Ensemble')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Slide 12: Time Series Decision Tree Visualization

Specialized visualization technique for decision trees applied to time series data, showing how temporal features and sequential patterns are captured in the tree structure.

```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Generate time series data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'value': np.sin(np.arange(len(dates)) * 0.1) + np.random.normal(0, 0.1, len(dates))
})

# Create features
ts_data['day_of_week'] = ts_data['date'].dt.dayofweek
ts_data['month'] = ts_data['date'].dt.month
ts_data['day'] = ts_data['date'].dt.day
ts_data['lag1'] = ts_data['value'].shift(1)
ts_data = ts_data.dropna()

# Train model
X = ts_data[['day_of_week', 'month', 'day', 'lag1']]
y = ts_data['value']
ts_tree = DecisionTreeRegressor(max_depth=4, random_state=42)
ts_tree.fit(X, y)

# Visualize time series tree
plt.figure(figsize=(20, 12))
plot_tree(ts_tree,
          feature_names=X.columns,
          filled=True,
          rounded=True,
          precision=3)
plt.title('Time Series Decision Tree')
plt.show()

# Plot actual vs predicted
plt.figure(figsize=(15, 6))
plt.plot(ts_data['date'], y, label='Actual', alpha=0.7)
plt.plot(ts_data['date'], ts_tree.predict(X), label='Predicted', alpha=0.7)
plt.title('Time Series Prediction')
plt.legend()
plt.show()
```

Slide 13: Additional Resources

*   Random Forest visualization techniques and interpretation:
    *   [https://arxiv.org/abs/1711.09784](https://arxiv.org/abs/1711.09784)
*   Interactive visualization for decision trees:
    *   [https://arxiv.org/abs/1908.03725](https://arxiv.org/abs/1908.03725)
*   Ensemble tree visualization methods:
    *   [https://arxiv.org/abs/2005.08772](https://arxiv.org/abs/2005.08772)
*   Suggested Google searches:
    *   "Decision tree visualization techniques in machine learning"
    *   "Interactive tree visualization libraries for Python"
    *   "Advanced tree pruning visualization methods"

