## Regression Decision Tree in Python
Slide 1: Introduction to Regression Decision Trees

Regression Decision Trees are a powerful machine learning technique used for predicting continuous numerical values. They combine the simplicity of decision trees with the ability to handle regression tasks, making them versatile for various real-world applications.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Generate sample data
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Create and fit the regressor
regressor = DecisionTreeRegressor(max_depth=5)
regressor.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = regressor.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_pred, color="cornflowerblue", label="prediction", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
```

Slide 2: How Regression Decision Trees Work

Regression Decision Trees work by recursively splitting the data into subsets based on feature values. At each node, the tree algorithm selects the best feature and split point that minimizes the variance in the target variable within the resulting subsets.

```python
from sklearn.tree import export_text

# Train a simple regression tree
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 5, 4, 5]
regressor = DecisionTreeRegressor(max_depth=3)
regressor.fit(X, y)

# Display the tree structure
tree_rules = export_text(regressor, feature_names=['X'])
print(tree_rules)
```

Slide 3: Splitting Criteria

The most common splitting criterion for regression trees is the Mean Squared Error (MSE). The algorithm aims to minimize the MSE at each split, which helps in reducing the overall prediction error.

```python
def calculate_mse(y, y_pred):
    return np.mean((y - y_pred)**2)

# Example data
y_true = np.array([3, 4, 5, 2, 6])
y_pred1 = np.array([2.8, 3.9, 5.1, 2.2, 5.8])
y_pred2 = np.array([2.5, 3.5, 4.5, 2.5, 5.5])

mse1 = calculate_mse(y_true, y_pred1)
mse2 = calculate_mse(y_true, y_pred2)

print(f"MSE for prediction 1: {mse1:.4f}")
print(f"MSE for prediction 2: {mse2:.4f}")
```

Slide 4: Tree Growth and Pruning

Regression trees grow by continuously splitting nodes until a stopping criterion is met. Common stopping criteria include maximum tree depth, minimum number of samples per leaf, or minimum decrease in impurity. Pruning techniques can be applied to reduce overfitting.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate sample data
X = np.random.rand(100, 1) * 10
y = 2 * X + np.random.randn(100, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models with different max_depth
depths = [2, 5, 10, None]
for depth in depths:
    regressor = DecisionTreeRegressor(max_depth=depth)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Max depth: {depth}, MSE: {mse:.4f}")
```

Slide 5: Feature Importance

Regression Decision Trees provide a measure of feature importance, indicating how much each feature contributes to the predictions. This is valuable for understanding the most influential factors in your model.

```python
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Generate a random regression problem
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Train the regressor
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)

# Get feature importances
importances = regressor.feature_importances_

# Plot feature importances
plt.bar(range(len(importances)), importances)
plt.title("Feature Importances")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()
```

Slide 6: Handling Categorical Variables

Regression Decision Trees can handle both numerical and categorical variables. For categorical variables, the tree algorithm considers all possible splits based on the unique values of the feature.

```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create a sample dataset with mixed data types
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'city': ['New York', 'London', 'Paris', 'Tokyo', 'New York'],
    'salary': [50000, 60000, 70000, 80000, 90000]
})

# Prepare the feature matrix
X = data[['age', 'city']]
y = data['salary']

# Create a preprocessor for one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['age']),
        ('cat', OneHotEncoder(drop='first'), ['city'])
    ])

# Create and train the model
model = DecisionTreeRegressor()
model.fit(preprocessor.fit_transform(X), y)

# Print feature importances
feature_names = ['age'] + list(preprocessor.named_transformers_['cat'].get_feature_names(['city']))
importances = dict(zip(feature_names, model.feature_importances_))
print("Feature Importances:")
for feature, importance in importances.items():
    print(f"{feature}: {importance:.4f}")
```

Slide 7: Advantages of Regression Decision Trees

Regression Decision Trees offer several advantages, including interpretability, handling of non-linear relationships, and minimal data preprocessing requirements. They can capture complex patterns in the data without assuming a specific functional form.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import learning_curve

# Generate sample data
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Create the regressor
regressor = DecisionTreeRegressor(max_depth=5)

# Calculate learning curve
train_sizes, train_scores, test_scores = learning_curve(
    regressor, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# Calculate mean and standard deviation
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure()
plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.show()
```

Slide 8: Limitations of Regression Decision Trees

Despite their advantages, Regression Decision Trees have limitations. They can be prone to overfitting, especially with deep trees. They may also struggle with smooth, continuous functions and can be sensitive to small changes in the training data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Generate sample data
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Train two models: one with low max_depth and one with high max_depth
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=15)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 5))
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=15", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression: Overfitting Example")
plt.legend()
plt.show()
```

Slide 9: Ensemble Methods: Random Forests

To address the limitations of individual trees, ensemble methods like Random Forests can be used. Random Forests combine multiple decision trees to create a more robust and accurate model.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate a random regression dataset
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions
y_pred = rf_regressor.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Feature importance
importances = rf_regressor.feature_importances_
for i, importance in enumerate(importances):
    print(f"Feature {i+1} importance: {importance:.4f}")
```

Slide 10: Hyperparameter Tuning

Optimizing hyperparameters is crucial for achieving the best performance with Regression Decision Trees. Common hyperparameters include max\_depth, min\_samples\_split, and min\_samples\_leaf.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression

# Generate a random regression dataset
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a decision tree regressor
regressor = DecisionTreeRegressor(random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best negative MSE:", -grid_search.best_score_)
```

Slide 11: Real-life Example: Predicting House Prices

Regression Decision Trees can be applied to predict house prices based on various features such as size, location, and number of rooms. This example demonstrates how to use a Decision Tree Regressor for this task.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Create a sample dataset
data = pd.DataFrame({
    'size': np.random.randint(1000, 5000, 100),
    'bedrooms': np.random.randint(1, 6, 100),
    'location': np.random.choice(['urban', 'suburban', 'rural'], 100),
    'age': np.random.randint(0, 100, 100),
    'price': np.random.randint(100000, 1000000, 100)
})

# Encode categorical variables
data = pd.get_dummies(data, columns=['location'], drop_first=True)

# Split features and target
X = data.drop('price', axis=1)
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Feature importance
importances = regressor.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance:.4f}")
```

Slide 12: Real-life Example: Predicting Crop Yield

Regression Decision Trees can be applied to predict crop yields based on environmental factors. This example demonstrates how to implement such a model using various features that influence crop production.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create a sample dataset
np.random.seed(42)
data = pd.DataFrame({
    'temperature': np.random.uniform(10, 35, 200),
    'rainfall': np.random.uniform(500, 2000, 200),
    'soil_quality': np.random.choice(['poor', 'average', 'good'], 200),
    'fertilizer': np.random.uniform(50, 200, 200),
    'pest_control': np.random.choice([0, 1], 200),
    'yield': np.random.uniform(1000, 5000, 200)
})

# Prepare features and target
X = data.drop('yield', axis=1)
y = data['yield']

# Create preprocessor for encoding categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['temperature', 'rainfall', 'fertilizer', 'pest_control']),
        ('cat', OneHotEncoder(drop='first'), ['soil_quality'])
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = DecisionTreeRegressor(random_state=42)
model.fit(preprocessor.fit_transform(X_train), y_train)

# Make predictions
y_pred = model.predict(preprocessor.transform(X_test))

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Feature importance
feature_names = (preprocessor.named_transformers_['num'].feature_names_ +
                 preprocessor.named_transformers_['cat'].get_feature_names(['soil_quality']).tolist())
importances = dict(zip(feature_names, model.feature_importances_))
for feature, importance in importances.items():
    print(f"{feature}: {importance:.4f}")
```

Slide 13: Visualizing the Decision Tree

Visualization can help in understanding the structure and decision-making process of a Regression Decision Tree. This example shows how to create and display a simple tree diagram.

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Create a simple dataset
X = [[1], [2], [3], [4], [5], [6], [7]]
y = [2, 4, 5, 4, 5, 6, 7]

# Train the model
regressor = DecisionTreeRegressor(max_depth=3)
regressor.fit(X, y)

# Plot the tree
plt.figure(figsize=(12, 8))
plot_tree(regressor, feature_names=['X'], filled=True, rounded=True)
plt.title("Regression Decision Tree Visualization")
plt.show()

# Print the decision path for a sample input
sample_input = [[3.5]]
decision_path = regressor.decision_path(sample_input)
print("Decision path for input 3.5:")
for node_id in decision_path.indices:
    if node_id == regressor.tree_.children_left[decision_path.indices[0]]:
        print(f"Node {node_id}: Go left")
    elif node_id == regressor.tree_.children_right[decision_path.indices[0]]:
        print(f"Node {node_id}: Go right")
    else:
        print(f"Node {node_id}: Leaf node (prediction)")
```

Slide 14: Comparison with Other Regression Techniques

Regression Decision Trees can be compared with other regression techniques to understand their strengths and weaknesses. This example compares a Decision Tree Regressor with Linear Regression and Random Forest Regression.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import numpy as np

# Generate sample data
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
dt_model = DecisionTreeRegressor(random_state=42)
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train and evaluate models
models = [dt_model, lr_model, rf_model]
model_names = ['Decision Tree', 'Linear Regression', 'Random Forest']

for name, model in zip(model_names, models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} - Mean Squared Error: {mse:.4f}")

# Compare feature importance (for tree-based models)
dt_importance = dt_model.feature_importances_
rf_importance = rf_model.feature_importances_

print("\nFeature Importance:")
for i in range(X.shape[1]):
    print(f"Feature {i+1}:")
    print(f"  Decision Tree: {dt_importance[i]:.4f}")
    print(f"  Random Forest: {rf_importance[i]:.4f}")
```

Slide 15: Additional Resources

For those interested in diving deeper into Regression Decision Trees and related topics, here are some valuable resources:

1. "Decision Trees and Forests: A Probabilistic Perspective" by Antonio Criminisi and Jamie Shotton (arXiv:1906.01827) URL: [https://arxiv.org/abs/1906.01827](https://arxiv.org/abs/1906.01827)
2. "Random Forests" by Leo Breiman (Machine Learning, 45(1), 5-32, 2001) DOI: 10.1023/A:1010933404324
3. "Scikit-learn: Machine Learning in Python" by Fabian Pedregosa et al. (Journal of Machine Learning Research, 12, 2825-2830, 2011) URL: [https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf](https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf)

These resources provide in-depth explanations of decision tree algorithms, their applications, and implementations in popular machine learning libraries.

