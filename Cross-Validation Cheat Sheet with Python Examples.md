## Cross-Validation Cheat Sheet with Python Examples

Slide 1: Introduction to Cross-Validation

Cross-validation is a statistical method used to evaluate machine learning models by partitioning the data into subsets. It helps assess how well a model generalizes to unseen data and prevents overfitting. This technique is crucial for model selection and hyperparameter tuning.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Perform 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {np.mean(scores):.2f} (+/- {np.std(scores) * 2:.2f})")
```

Slide 2: K-Fold Cross-Validation

K-Fold Cross-Validation divides the dataset into K equal-sized subsets or folds. The model is trained on K-1 folds and validated on the remaining fold. This process is repeated K times, with each fold serving as the validation set once. The final performance is the average of all K iterations.

```python
from sklearn.metrics import accuracy_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_scores = []
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    fold_scores.append(accuracy_score(y_val, y_pred))

print(f"Fold scores: {fold_scores}")
print(f"Mean accuracy: {np.mean(fold_scores):.2f}")
```

Slide 3: Stratified K-Fold Cross-Validation

Stratified K-Fold ensures that the proportion of samples for each class is roughly the same in each fold as in the whole dataset. This is particularly useful for imbalanced datasets or when dealing with classification problems.

```python

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

stratified_scores = []
for train_index, val_index in skf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    stratified_scores.append(accuracy_score(y_val, y_pred))

print(f"Stratified fold scores: {stratified_scores}")
print(f"Mean accuracy: {np.mean(stratified_scores):.2f}")
```

Slide 4: Leave-One-Out Cross-Validation (LOOCV)

LOOCV is an extreme form of K-Fold cross-validation where K equals the number of samples. Each sample serves as the validation set once, while the model is trained on all other samples. This method is computationally expensive but can be useful for small datasets.

```python

loo = LeaveOneOut()
loo_scores = []

for train_index, val_index in loo.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    loo_scores.append(accuracy_score(y_val, y_pred))

print(f"LOOCV mean accuracy: {np.mean(loo_scores):.2f}")
```

Slide 5: Time Series Cross-Validation

For time series data, it's crucial to maintain the temporal order of observations. Time Series Split creates training-validation pairs by gradually increasing the training set size while always using future data for validation.

```python

tscv = TimeSeriesSplit(n_splits=5)

ts_scores = []
for train_index, val_index in tscv.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    ts_scores.append(accuracy_score(y_val, y_pred))

print(f"Time Series CV scores: {ts_scores}")
print(f"Mean accuracy: {np.mean(ts_scores):.2f}")
```

Slide 6: Nested Cross-Validation

Nested Cross-Validation is used when you need to tune hyperparameters and get an unbiased estimate of model performance. It consists of an outer loop for performance estimation and an inner loop for hyperparameter tuning.

```python

param_grid = {'C': [0.1, 1, 10]}
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

nested_scores = []

for train_index, val_index in outer_cv.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    clf = GridSearchCV(LogisticRegression(), param_grid, cv=inner_cv)
    clf.fit(X_train, y_train)
    
    nested_scores.append(clf.score(X_val, y_val))

print(f"Nested CV scores: {nested_scores}")
print(f"Mean accuracy: {np.mean(nested_scores):.2f}")
```

Slide 7: Cross-Validation for Regression

Cross-validation is not limited to classification problems. For regression tasks, we use appropriate metrics like Mean Squared Error (MSE) or R-squared to evaluate model performance.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
model = LinearRegression()

cv = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scores = []
r2_scores = []

for train_index, val_index in cv.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    mse_scores.append(mean_squared_error(y_val, y_pred))
    r2_scores.append(r2_score(y_val, y_pred))

print(f"Mean MSE: {np.mean(mse_scores):.2f}")
print(f"Mean R-squared: {np.mean(r2_scores):.2f}")
```

Slide 8: Cross-Validation with Preprocessing

When applying preprocessing steps, it's crucial to perform them within the cross-validation loop to avoid data leakage. We can use Pipeline to ensure proper encapsulation of preprocessing and model fitting.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Create a pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Perform cross-validation on the pipeline
scores = cross_val_score(pipeline, X, y, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {np.mean(scores):.2f}")
```

Slide 9: Cross-Validation for Feature Selection

Cross-validation can be used in feature selection processes to ensure that the selected features generalize well. Here's an example using Recursive Feature Elimination with Cross-Validation (RFECV).

```python

# Create a pipeline with feature selection and model
pipeline = Pipeline([
    ('feature_selection', RFECV(estimator=LogisticRegression(), step=1, cv=5)),
    ('model', LogisticRegression())
])

# Perform cross-validation on the pipeline
scores = cross_val_score(pipeline, X, y, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {np.mean(scores):.2f}")

# Get selected features
selected_features = pipeline.named_steps['feature_selection'].support_
print(f"Selected features: {selected_features}")
```

Slide 10: Cross-Validation for Ensemble Methods

Ensemble methods like Random Forest already use bootstrapping, but cross-validation can still be beneficial for hyperparameter tuning and performance estimation.

```python
from sklearn.model_selection import RandomizedSearchCV

# Define hyperparameter search space
param_dist = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create RandomizedSearchCV object
rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, 
                                   n_iter=20, cv=5, random_state=42)

# Fit the random search object
random_search.fit(X, y)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.2f}")
```

Slide 11: Cross-Validation for Imbalanced Datasets

When dealing with imbalanced datasets, it's important to use stratified sampling and appropriate evaluation metrics. Here's an example using balanced accuracy and precision-recall AUC.

```python
from sklearn.metrics import balanced_accuracy_score, average_precision_score
from imblearn.over_sampling import SMOTE

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], 
                           n_informative=3, n_redundant=1, flip_y=0, 
                           n_features=20, random_state=42)

# Define custom scorer
def custom_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    y_proba = estimator.predict_proba(X)[:, 1]
    
    balanced_acc = balanced_accuracy_score(y, y_pred)
    pr_auc = average_precision_score(y, y_proba)
    
    return (balanced_acc + pr_auc) / 2

# Create pipeline with SMOTE and LogisticRegression
pipeline = Pipeline([
    ('sampler', SMOTE(random_state=42)),
    ('model', LogisticRegression())
])

# Perform cross-validation
scores = cross_val_score(pipeline, X, y, cv=5, scoring=custom_scorer)

print(f"Cross-validation scores: {scores}")
print(f"Mean score: {np.mean(scores):.2f}")
```

Slide 12: Real-Life Example: Predicting Customer Churn

In this example, we'll use cross-validation to evaluate a model for predicting customer churn in a telecommunications company.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Load the dataset (you would replace this with your actual data loading code)
data = pd.read_csv('telecom_churn.csv')

# Split features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

# Perform cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')

print(f"Cross-validation ROC AUC scores: {cv_scores}")
print(f"Mean ROC AUC: {np.mean(cv_scores):.2f}")

# Fit the model on the entire training set and evaluate on test set
clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Test set ROC AUC: {test_roc_auc:.2f}")
```

Slide 13: Real-Life Example: Predicting Energy Consumption

In this example, we'll use cross-validation to evaluate a model for predicting energy consumption in buildings.

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset (replace with actual data loading)
data = pd.read_csv('energy_consumption.csv')

X = data.drop('EnergyConsumption', axis=1)
y = data['EnergyConsumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor())
])

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
mae_scores = -cv_scores

print(f"Cross-validation MAE scores: {mae_scores}")
print(f"Mean MAE: {np.mean(mae_scores):.2f}")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print(f"Test set MAE: {test_mae:.2f}")
print(f"Test set R-squared: {test_r2:.2f}")
```

Slide 14: Cross-Validation Pitfalls and Best Practices

Cross-validation is a powerful technique, but it's important to be aware of potential pitfalls and follow best practices:

1. Data leakage: Ensure that all preprocessing steps are included within the cross-validation loop to prevent information from the validation set influencing the model.
2. Stratification: Use stratified sampling for classification problems, especially with imbalanced datasets, to maintain class proportions across folds.
3. Temporal data: For time series data, use time-based splitting instead of random splitting to respect the temporal order of observations.
4. Computational cost: Be mindful of the computational resources required, especially for large datasets or complex models. Consider using fewer folds or random subsampling for initial experiments.
5. Hyperparameter tuning: Use nested cross-validation when performing both model selection and performance estimation to avoid overfitting to the validation set.
6. Appropriate metrics: Choose evaluation metrics that align with your problem and business objectives. For example, use area under the ROC curve for binary classification or mean absolute error for regression.
7. Reporting results: Always report both the mean and standard deviation of cross-validation scores to provide a complete picture of model performance and stability.

```python
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

cv_results = cross_validate(pipeline, X, y, cv=5, 
                            scoring=['accuracy', 'roc_auc'],
                            return_train_score=True)

print("Test Accuracy: {:.2f} (+/- {:.2f})".format(
    cv_results['test_accuracy'].mean(), 
    cv_results['test_accuracy'].std() * 2
))

print("Test ROC AUC: {:.2f} (+/- {:.2f})".format(
    cv_results['test_roc_auc'].mean(), 
    cv_results['test_roc_auc'].std() * 2
))
```

Slide 15: Additional Resources

For further exploration of cross-validation techniques and best practices, consider the following resources:

1. "A Survey of Cross-Validation Procedures for Model Selection" by Arlot, S. and Celisse, A. (2010) ArXiv link: [https://arxiv.org/abs/0907.4728](https://arxiv.org/abs/0907.4728)
2. "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure" by Roberts, D. R. et al. (2017) ArXiv link: [https://arxiv.org/abs/1706.07592](https://arxiv.org/abs/1706.07592)
3. "Nested Cross-Validation When Selecting Classifiers Is Overzealous For Most Practical Applications" by Wainer, J. and Cawley, G. (2017) ArXiv link: [https://arxiv.org/abs/1809.09446](https://arxiv.org/abs/1809.09446)

These papers provide in-depth discussions on various aspects of cross-validation, including its applications in different domains and potential limitations.


