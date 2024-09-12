## Building a Gradient Boosting Machine from Scratch in Python
Slide 1: Introduction to Gradient Boosting Machines

Gradient Boosting Machines (GBM) are a powerful ensemble learning technique used in machine learning for both regression and classification tasks. They work by building a series of weak learners, typically decision trees, and combining them to create a strong predictive model. In this presentation, we'll explore how to build a GBM algorithm from scratch using Python.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class GradientBoostingMachine:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        # Initialize prediction with zeros
        self.initial_prediction = np.mean(y)
        F = np.full(len(y), self.initial_prediction)

        for _ in range(self.n_estimators):
            residuals = y - F
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            F += self.learning_rate * tree.predict(X)

    def predict(self, X):
        F = np.full(len(X), self.initial_prediction)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F
```

Slide 2: Understanding Decision Trees as Base Learners

Decision trees are the building blocks of GBMs. They partition the feature space into regions and make predictions based on the average target value in each region. In GBMs, we typically use shallow trees to prevent overfitting.

```python
def visualize_decision_tree(tree, feature_names, max_depth=3):
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20,10))
    plot_tree(tree, feature_names=feature_names, filled=True, rounded=True, max_depth=max_depth)
    plt.show()

# Example usage
X = np.random.rand(100, 2)
y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.normal(0, 0.1, 100)

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

visualize_decision_tree(tree, ['Feature 1', 'Feature 2'])
```

Slide 3: The Gradient Boosting Algorithm

Gradient Boosting works by iteratively improving predictions. In each iteration, it fits a new tree to the residuals (the difference between the true values and the current predictions). The predictions are then updated by adding the new tree's predictions, scaled by a learning rate.

```python
def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1, max_depth=3):
    residuals = y - current_predictions
    tree = DecisionTreeRegressor(max_depth=max_depth)
    tree.fit(X, residuals)
    return current_predictions + learning_rate * tree.predict(X), tree

# Example usage
X = np.random.rand(100, 2)
y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.normal(0, 0.1, 100)

initial_prediction = np.mean(y)
current_predictions = np.full(len(y), initial_prediction)

for i in range(5):
    current_predictions, tree = gradient_boosting_step(X, y, current_predictions)
    mse = mean_squared_error(y, current_predictions)
    print(f"Iteration {i+1}, MSE: {mse:.4f}")
```

Slide 4: Implementing the Fit Method

The fit method is where we train our GBM model. We start with an initial prediction (usually the mean of the target variable) and then iteratively add trees to improve our predictions.

```python
class SimpleGBM:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        F = np.full(len(y), self.initial_prediction)

        for _ in range(self.n_estimators):
            residuals = y - F
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            F += self.learning_rate * tree.predict(X)

        return self

# Example usage
X = np.random.rand(1000, 5)
y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 2]**2 + np.random.normal(0, 0.1, 1000)

gbm = SimpleGBM(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm.fit(X, y)
```

Slide 5: Implementing the Predict Method

The predict method applies the trained model to new data. It starts with the initial prediction and then adds the predictions from each tree, scaled by the learning rate.

```python
class SimpleGBM:
    # ... (previous code)

    def predict(self, X):
        F = np.full(len(X), self.initial_prediction)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

# Example usage
X_test = np.random.rand(100, 5)
y_test = np.sin(X_test[:, 0]) + np.cos(X_test[:, 1]) + X_test[:, 2]**2 + np.random.normal(0, 0.1, 100)

gbm = SimpleGBM(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm.fit(X, y)
predictions = gbm.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.4f}")
```

Slide 6: Handling Different Loss Functions

GBMs can use different loss functions for various tasks. For regression, we typically use mean squared error, while for classification, we might use log loss. Here's how we can modify our GBM to handle different loss functions.

```python
import numpy as np
from scipy.special import expit  # For logistic function

class FlexibleGBM:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, loss='mse'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.loss = loss
        self.trees = []

    def negative_gradient(self, y, pred):
        if self.loss == 'mse':
            return y - pred
        elif self.loss == 'log_loss':
            return y - expit(pred)
        else:
            raise ValueError("Unsupported loss function")

    def fit(self, X, y):
        if self.loss == 'mse':
            self.initial_prediction = np.mean(y)
        elif self.loss == 'log_loss':
            self.initial_prediction = np.log(np.mean(y) / (1 - np.mean(y)))

        F = np.full(len(y), self.initial_prediction)

        for _ in range(self.n_estimators):
            neg_grad = self.negative_gradient(y, F)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, neg_grad)
            self.trees.append(tree)
            F += self.learning_rate * tree.predict(X)

        return self

    def predict(self, X):
        F = np.full(len(X), self.initial_prediction)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        
        if self.loss == 'log_loss':
            return expit(F)
        return F

# Example usage
X = np.random.rand(1000, 5)
y = (np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 2]**2 > 1).astype(int)

gbm = FlexibleGBM(n_estimators=100, learning_rate=0.1, max_depth=3, loss='log_loss')
gbm.fit(X, y)

X_test = np.random.rand(100, 5)
predictions = gbm.predict(X_test)
print("First few predictions:", predictions[:5])
```

Slide 7: Feature Importance

One advantage of tree-based models is that they provide a measure of feature importance. We can calculate this by summing the improvement in the splitting criterion (e.g., MSE reduction) for each feature across all trees.

```python
class GBMWithFeatureImportance(FlexibleGBM):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, loss='mse'):
        super().__init__(n_estimators, learning_rate, max_depth, loss)
        self.feature_importances_ = None

    def fit(self, X, y):
        super().fit(X, y)
        self.feature_importances_ = np.zeros(X.shape[1])
        for tree in self.trees:
            self.feature_importances_ += tree.feature_importances_
        self.feature_importances_ /= len(self.trees)
        return self

    def plot_feature_importances(self, feature_names=None):
        import matplotlib.pyplot as plt

        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(self.feature_importances_))]

        plt.figure(figsize=(10, 6))
        plt.bar(feature_names, self.feature_importances_)
        plt.title("Feature Importances")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Example usage
X = np.random.rand(1000, 5)
y = 2*X[:, 0] + 3*X[:, 1]**2 - 0.5*X[:, 2] + np.random.normal(0, 0.1, 1000)

gbm = GBMWithFeatureImportance(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm.fit(X, y)
gbm.plot_feature_importances(["X1", "X2", "X3", "X4", "X5"])
```

Slide 8: Regularization Techniques

To prevent overfitting, we can apply regularization techniques such as early stopping, which stops training when the validation error starts to increase.

```python
from sklearn.model_selection import train_test_split

class GBMWithEarlyStopping(GBMWithFeatureImportance):
    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=10):
        if X_val is None or y_val is None:
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.2)

        self.initial_prediction = np.mean(y)
        F = np.full(len(y), self.initial_prediction)
        F_val = np.full(len(y_val), self.initial_prediction)

        best_val_loss = np.inf
        best_iteration = 0
        self.trees = []

        for i in range(self.n_estimators):
            residuals = self.negative_gradient(y, F)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            F += self.learning_rate * tree.predict(X)
            F_val += self.learning_rate * tree.predict(X_val)

            val_loss = mean_squared_error(y_val, F_val)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iteration = i
            elif i - best_iteration >= early_stopping_rounds:
                print(f"Early stopping at iteration {i}")
                self.trees = self.trees[:best_iteration+1]
                break

        return self

# Example usage
X = np.random.rand(1000, 5)
y = 2*X[:, 0] + 3*X[:, 1]**2 - 0.5*X[:, 2] + np.random.normal(0, 0.1, 1000)

gbm = GBMWithEarlyStopping(n_estimators=1000, learning_rate=0.1, max_depth=3)
gbm.fit(X, y, early_stopping_rounds=10)
```

Slide 9: Hyperparameter Tuning

Choosing the right hyperparameters is crucial for optimal performance. We can use techniques like grid search or random search to find the best combination of hyperparameters.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

class SklearnCompatibleGBM(GBMWithEarlyStopping):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, early_stopping_rounds=10):
        super().__init__(n_estimators, learning_rate, max_depth)
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X, y):
        X, X_val, y, y_val = train_test_split(X, y, test_size=0.2)
        super().fit(X, y, X_val, y_val, self.early_stopping_rounds)
        return self

    def score(self, X, y):
        return -mean_squared_error(y, self.predict(X))

# Example usage
X = np.random.rand(1000, 5)
y = 2*X[:, 0] + 3*X[:, 1]**2 - 0.5*X[:, 2] + np.random.normal(0, 0.1, 1000)

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [2, 3, 4],
    'early_stopping_rounds': [5, 10, 20]
}

gbm = SklearnCompatibleGBM()
grid_search = GridSearchCV(gbm, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best score:", -grid_search.best_score_)
```

Slide 10: Handling Categorical Variables

GBMs can handle categorical variables, but we need to encode them properly. One common approach is to use one-hot encoding for low-cardinality features and target encoding for high-cardinality features.

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

def preprocess_data(X, y, categorical_cols, high_cardinality_threshold=10):
    X = pd.DataFrame(X)
    
    # Identify high and low cardinality categorical columns
    low_card_cols = [col for col in categorical_cols if X[col].nunique() <= high_cardinality_threshold]
    high_card_cols = [col for col in categorical_cols if X[col].nunique() > high_cardinality_threshold]
    
    # One-hot encoding for low cardinality features
    onehot = OneHotEncoder(sparse=False, handle_unknown='ignore')
    onehot_encoded = onehot.fit_transform(X[low_card_cols])
    onehot_cols = onehot.get_feature_names(low_card_cols)
    
    # Target encoding for high cardinality features
    target_encoder = TargetEncoder()
    target_encoded = target_encoder.fit_transform(X[high_card_cols], y)
    
    # Combine encoded features with numerical features
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    X_processed = pd.concat([
        X[numerical_cols].reset_index(drop=True),
        pd.DataFrame(onehot_encoded, columns=onehot_cols),
        target_encoded.reset_index(drop=True)
    ], axis=1)
    
    return X_processed

# Example usage
X = pd.DataFrame({
    'num1': np.random.rand(1000),
    'cat1': np.random.choice(['A', 'B', 'C'], 1000),
    'cat2': np.random.choice(['X', 'Y', 'Z', 'W'], 1000),
    'cat3': np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 1000)
})
y = np.random.rand(1000)

X_processed = preprocess_data(X, y, categorical_cols=['cat1', 'cat2', 'cat3'])
print(X_processed.head())
```

Slide 11: Implementing Gradient Boosting for Classification

While we've focused on regression, GBMs can also be used for classification tasks. Here's how we can modify our implementation to handle binary classification using log loss.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.special import expit

class GBMClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.initial_prediction = np.log(np.mean(y) / (1 - np.mean(y)))
        F = np.full(len(y), self.initial_prediction)

        for _ in range(self.n_estimators):
            prob = expit(F)
            residuals = y - prob
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            F += self.learning_rate * tree.predict(X)

        return self

    def predict_proba(self, X):
        F = np.full(len(X), self.initial_prediction)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return expit(F)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

# Example usage
X = np.random.rand(1000, 5)
y = (X[:, 0] + X[:, 1]**2 > 1).astype(int)

gbm = GBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm.fit(X, y)

X_test = np.random.rand(100, 5)
predictions = gbm.predict(X_test)
probabilities = gbm.predict_proba(X_test)

print("First few predictions:", predictions[:5])
print("First few probabilities:", probabilities[:5])
```

Slide 12: Visualizing Decision Boundaries

For binary classification problems, it can be helpful to visualize the decision boundary of our GBM model. Here's how we can create a contour plot to visualize the decision boundary for a 2D feature space.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from scipy.special import expit

def plot_decision_boundary(X, y, model, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('GBM Decision Boundary')
    
    return ax

# Example usage
X = np.random.rand(1000, 2)
y = ((X[:, 0] - 0.5)**2 + (X[:, 1] - 0.5)**2 < 0.15).astype(int)

gbm = GBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm.fit(X, y)

plot_decision_boundary(X, y, gbm)
plt.show()
```

Slide 13: Real-Life Example: Predicting Customer Churn

Let's apply our GBM implementation to a real-world problem: predicting customer churn for a telecommunications company.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (you would need to download this dataset)
df = pd.read_csv('telecom_churn.csv')

# Preprocess the data
X = df.drop(['customerID', 'Churn'], axis=1)
y = (df['Churn'] == 'Yes').astype(int)

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
gbm = GBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=4)
gbm.fit(X_train, y_train)

# Make predictions
y_pred = gbm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot feature importances
feature_importances = np.mean([tree.feature_importances_ for tree in gbm.trees], axis=0)
feature_names = X.columns
plt.figure(figsize=(12, 6))
plt.bar(feature_names, feature_importances)
plt.title("Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```

Slide 14: Real-Life Example: Predicting Air Quality

Let's apply our GBM implementation to another real-world problem: predicting air quality index based on various environmental factors.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (you would need to download this dataset)
df = pd.read_csv('air_quality_data.csv')

# Preprocess the data
X = df.drop(['Date', 'AQI'], axis=1)
y = df['AQI']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
gbm = GradientBoostingMachine(n_estimators=200, learning_rate=0.1, max_depth=4)
gbm.fit(X_train, y_train)

# Make predictions
y_pred = gbm.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted Air Quality Index")
plt.tight_layout()
plt.show()

# Plot feature importances
feature_importances = np.mean([tree.feature_importances_ for tree in gbm.trees], axis=0)
feature_names = X.columns
plt.figure(figsize=(12, 6))
plt.bar(feature_names, feature_importances)
plt.title("Feature Importances for Air Quality Prediction")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into Gradient Boosting Machines and their implementations, here are some valuable resources:

1. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189-1232. ArXiv.org URL: [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794). ArXiv.org URL: [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
3. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In Advances in Neural Information Processing Systems (pp. 3146-3154). ArXiv.org URL: [https://arxiv.org/abs/1711.08766](https://arxiv.org/abs/1711.08766)

These papers provide in-depth explanations of the theoretical foundations and practical implementations of Gradient Boosting Machines and their variants.

