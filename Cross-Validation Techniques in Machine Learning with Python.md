## Cross-Validation Techniques in Machine Learning with Python
Slide 1: Cross-Validation in Machine Learning

Cross-validation is a statistical method used to estimate the skill of machine learning models. It's particularly useful for assessing how the results of a statistical analysis will generalize to an independent data set. Let's explore this concept with Python examples.

```python
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Perform 5-fold cross-validation
scores = cross_val_score(clf, X, y, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {np.mean(scores):.2f} (+/- {np.std(scores) * 2:.2f})")
```

Slide 2: Why Cross-Validation?

Cross-validation helps prevent overfitting by testing the model's performance on unseen data. It provides a more robust estimate of the model's generalization ability compared to a single train-test split.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Single train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Single split accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Cross-validation mean accuracy: {np.mean(scores):.2f}")
```

Slide 3: K-Fold Cross-Validation

K-Fold is the most common type of cross-validation. The data is divided into k subsets, and the holdout method is repeated k times. Each time, one of the k subsets is used as the test set and the other k-1 subsets are put together to form a training set.

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Fold {fold} accuracy: {accuracy:.2f}")
```

Slide 4: Stratified K-Fold Cross-Validation

Stratified K-Fold is a variation of K-Fold that returns stratified folds: each set contains approximately the same percentage of samples of each target class as the complete set.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Fold {fold} accuracy: {accuracy:.2f}")
```

Slide 5: Leave-One-Out Cross-Validation (LOOCV)

LOOCV is a special case of K-Fold cross-validation where K equals the number of instances in the data. Each instance in turn is used as the validation set while the remaining instances form the training set.

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(clf, X, y, cv=loo)

print(f"Number of CV iterations: {len(scores)}")
print(f"Mean accuracy: {np.mean(scores):.2f}")
```

Slide 6: Time Series Cross-Validation

For time series data, we need to respect the temporal order of observations. Time Series Split provides train/test indices to split time series data samples that are observed at fixed time intervals.

```python
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

X = np.array([[i] for i in range(100)])
y = np.random.rand(100)

tscv = TimeSeriesSplit(n_splits=5)

plt.figure(figsize=(10, 6))
for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
    plt.scatter(test_index, [fold] * len(test_index), c='r', s=5, label='Test Set' if fold == 1 else "")
    plt.scatter(train_index, [fold] * len(train_index), c='b', s=5, label='Train Set' if fold == 1 else "")

plt.ylabel("CV iteration")
plt.xlabel("Sample index")
plt.legend()
plt.title("Time Series Cross-Validation")
plt.show()
```

Slide 7: Cross-Validation with GridSearchCV

GridSearchCV combines cross-validation with hyperparameter tuning, allowing us to find the best parameters for our model.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.2f}")
```

Slide 8: Cross-Validation for Regression

Cross-validation isn't limited to classification problems. It's equally useful for regression tasks. Let's use the Boston Housing dataset as an example.

```python
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

boston = load_boston()
X, y = boston.data, boston.target

reg = LinearRegression()
mse_scores = cross_val_score(reg, X, y, cv=5, scoring='neg_mean_squared_error')

print(f"MSE scores: {-mse_scores}")
print(f"Mean MSE: {-np.mean(mse_scores):.2f} (+/- {np.std(mse_scores) * 2:.2f})")
```

Slide 9: Visualizing Cross-Validation Results

Visualizing the results of cross-validation can provide insights into the model's performance across different folds.

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(reg, X, y, cv=5)

plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, c='b', s=40, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.title("Cross-Validation Prediction")
plt.show()
```

Slide 10: Cross-Validation with Pandas DataFrames

In real-world scenarios, we often work with pandas DataFrames. Here's how to perform cross-validation on a DataFrame.

```python
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a sample DataFrame
df = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
})

X = df[['feature1', 'feature2']]
y = df['target']

# Create a pipeline with scaling and classification
pipeline = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=42))

scores = cross_val_score(pipeline, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {np.mean(scores):.2f} (+/- {np.std(scores) * 2:.2f})")
```

Slide 11: Cross-Validation for Feature Selection

Cross-validation can be used in feature selection to identify the most important features for our model.

```python
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC

# Create the RFE object and compute a cross-validated score
svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=5, scoring='accuracy')
rfecv.fit(X, y)

print(f"Optimal number of features: {rfecv.n_features_}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score")
plt.title("Recursive Feature Elimination with Cross-Validation")
plt.show()
```

Slide 12: Real-Life Example: Predicting Customer Churn

Let's apply cross-validation to a real-world problem: predicting customer churn for a telecommunications company.

```python
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Load the dataset (you would replace this with your actual data loading)
df = pd.read_csv('telecom_churn.csv')

# Prepare features and target
X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

# Create a pipeline
pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),  # Handle missing values
    StandardScaler(),                # Scale features
    RandomForestClassifier(random_state=42)
)

# Perform cross-validation
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {np.mean(scores):.2f} (+/- {np.std(scores) * 2:.2f})")
```

Slide 13: Real-Life Example: Predicting Bike Sharing Demand

Let's use cross-validation to evaluate a model for predicting bike sharing demand based on weather conditions and other factors.

```python
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Load the dataset (you would replace this with your actual data loading)
df = pd.read_csv('bike_sharing.csv')

# Prepare features and target
X = df.drop(['count', 'datetime'], axis=1)
y = df['count']

# Create a pipeline
pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),  # Handle missing values
    StandardScaler(),                # Scale features
    GradientBoostingRegressor(random_state=42)
)

# Perform cross-validation
scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')

print(f"Cross-validation MSE scores: {-scores}")
print(f"Mean MSE: {-np.mean(scores):.2f} (+/- {np.std(scores) * 2:.2f})")
```

Slide 14: Additional Resources

For more in-depth information about cross-validation and its applications in machine learning, consider exploring the following resources:

1. "A Survey of Cross-Validation Procedures for Model Selection" by Arlot, S. and Celisse, A. (2010) ArXiv link: [https://arxiv.org/abs/0907.4728](https://arxiv.org/abs/0907.4728)
2. "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure" by Roberts et al. (2017) ArXiv link: [https://arxiv.org/abs/1706.07846](https://arxiv.org/abs/1706.07846)
3. "Pitfalls of supervised feature selection" by Smialowski et al. (2010) ArXiv link: [https://arxiv.org/abs/1008.3171](https://arxiv.org/abs/1008.3171)

These papers provide detailed insights into various aspects of cross-validation and its applications in different domains of machine learning.

