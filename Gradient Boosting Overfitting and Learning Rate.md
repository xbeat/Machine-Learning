## Gradient Boosting Overfitting and Learning Rate
Slide 1: Learning Rate Impact on Gradient Boosting A smaller learning rate in gradient boosting reduces overfitting by making conservative updates to the model in each iteration, allowing for better generalization performance at the cost of increased training time.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Compare learning rates
learning_rates = [1.0, 0.1, 0.01]
test_scores = []

for lr in learning_rates:
    gb = GradientBoostingRegressor(learning_rate=lr, n_estimators=100)
    gb.fit(X_train, y_train)
    score = gb.score(X_test, y_test)
    test_scores.append(score)
    print(f"Learning rate: {lr}, R² Score: {score:.4f}")
```

Slide 2: Tree Depth Control and Complexity

```python
from sklearn.datasets import make_classification
import numpy as np

# Generate binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Test different max_depth values
depths = [2, 4, 6, 8]
depth_scores = []

for depth in depths:
    gb = GradientBoostingRegressor(max_depth=depth, n_estimators=100)
    gb.fit(X_train, y_train)
    score = gb.score(X_test, y_test)
    depth_scores.append(score)
    print(f"Max depth: {depth}, R² Score: {score:.4f}")
```

Slide 3: Early Stopping Implementation

Early stopping prevents overfitting by monitoring validation performance and stopping training when the model's performance starts to degrade, effectively determining the optimal number of trees.

```python
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

gb = GradientBoostingRegressor(n_estimators=1000, validation_fraction=0.1,
                              n_iter_no_change=5, tol=1e-4)
gb.fit(X_train, y_train)

print(f"Optimal number of trees: {gb.n_estimators_}")
print(f"Best validation score: {gb.best_score_:.4f}")
```

Slide 4: Subsample Ratio Effects

Subsampling helps prevent overfitting by introducing randomness into the training process, where each tree sees only a portion of the training data, leading to better generalization.

```python
subsample_ratios = [0.5, 0.7, 1.0]
subsample_scores = []

for ratio in subsample_ratios:
    gb = GradientBoostingRegressor(subsample=ratio, n_estimators=100)
    gb.fit(X_train, y_train)
    score = gb.score(X_test, y_test)
    subsample_scores.append(score)
    print(f"Subsample ratio: {ratio}, R² Score: {score:.4f}")
```

Slide 5: L1/L2 Regularization Implementation

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test different alpha values (L2 regularization)
alphas = [0.0, 0.1, 0.5]
reg_scores = []

for alpha in alphas:
    gb = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100,
                                  validation_fraction=0.1,
                                  alpha=alpha)  # L2 regularization
    gb.fit(X_train_scaled, y_train)
    score = gb.score(X_test_scaled, y_test)
    reg_scores.append(score)
    print(f"Alpha (L2): {alpha}, R² Score: {score:.4f}")
```

Slide 6: Real-world Example: Housing Price Prediction

Implementation of gradient boosting for predicting house prices using the California Housing dataset, demonstrating comprehensive preprocessing and model tuning.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load and prepare data
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optimize model
gb = GradientBoostingRegressor(
    learning_rate=0.1,
    n_estimators=200,
    max_depth=4,
    subsample=0.8,
    validation_fraction=0.1,
    n_iter_no_change=5,
    random_state=42
)

gb.fit(X_train_scaled, y_train)
```

Slide 7: Results for Housing Price Prediction

```python
# Model evaluation
train_pred = gb.predict(X_train_scaled)
test_pred = gb.predict(X_test_scaled)

print("Training Results:")
print(f"MSE: {mean_squared_error(y_train, train_pred):.4f}")
print(f"R² Score: {r2_score(y_train, train_pred):.4f}")

print("\nTest Results:")
print(f"MSE: {mean_squared_error(y_test, test_pred):.4f}")
print(f"R² Score: {r2_score(y_test, test_pred):.4f}")

# Feature importance
importance = pd.DataFrame({
    'feature': housing.feature_names,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(importance)
```

Slide 8: Real-world Example: Credit Risk Assessment

```python
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, roc_auc_score

# Generate credit risk dataset
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2,
                          weights=[0.9, 0.1], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model with optimal parameters
gb_classifier = GradientBoostingClassifier(
    learning_rate=0.05,
    n_estimators=300,
    max_depth=3,
    subsample=0.8,
    validation_fraction=0.1,
    n_iter_no_change=5,
    random_state=42
)

gb_classifier.fit(X_train, y_train)
```

Slide 9: Credit Risk Assessment Results

```python
# Model evaluation
y_pred = gb_classifier.predict(X_test)
y_prob = gb_classifier.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nROC AUC Score:", roc_auc_score(y_test, y_prob))

# Plot ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

Slide 10: Minimum Loss Reduction Implementation

```python
min_loss_reductions = [0.0, 0.1, 0.5, 1.0]
loss_reduction_scores = []

for min_loss in min_loss_reductions:
    gb = GradientBoostingRegressor(
        min_impurity_decrease=min_loss,
        n_estimators=100,
        learning_rate=0.1
    )
    gb.fit(X_train, y_train)
    score = gb.score(X_test, y_test)
    loss_reduction_scores.append(score)
    print(f"Min loss reduction: {min_loss}, R² Score: {score:.4f}")
```

Slide 11: Gradient Boosting Loss Functions

Implementation of different loss functions in gradient boosting to handle various types of prediction tasks and their impact on model performance.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

loss_functions = ['ls', 'lad', 'huber']
loss_scores = []

for loss in loss_functions:
    gb = GradientBoostingRegressor(
        loss=loss,
        n_estimators=100,
        learning_rate=0.1
    )
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print(f"Loss function: {loss}")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}\n")
```

Slide 12: Cross-validation Implementation

```python
from sklearn.model_selection import cross_val_score

# Define parameter grid
params = {
    'learning_rate': 0.1,
    'n_estimators': 100,
    'max_depth': 4,
    'subsample': 0.8
}

gb = GradientBoostingRegressor(**params)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(gb, X, y, cv=5, scoring='r2')

print("Cross-validation scores:", cv_scores)
print(f"Mean CV score: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")
```

Slide 13: Additional Resources 

[https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754) - XGBoost: A Scalable Tree Boosting System [https://arxiv.org/abs/1706.09516](https://arxiv.org/abs/1706.09516) - LightGBM: A Highly Efficient Gradient Boosting Decision Tree [https://arxiv.org/abs/1810.09092](https://arxiv.org/abs/1810.09092) - CatBoost: unbiased boosting with categorical features [https://arxiv.org/abs/2002.05780](https://arxiv.org/abs/2002.05780) - NGBoost: Natural Gradient Boosting for Probabilistic Prediction

