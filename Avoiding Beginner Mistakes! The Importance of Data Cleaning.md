## Avoiding Beginner Mistakes! The Importance of Data Cleaning

Slide 1: The Importance of Data Cleaning

Data cleaning is a crucial step in the data science process, often overlooked by beginners. It involves handling missing values, removing duplicates, and addressing inconsistencies. Let's explore a simple example of data cleaning using pandas:

```python
import numpy as np

# Create a sample dataset with issues
data = {
    'name': ['John', 'Jane', 'Mike', 'John', np.nan],
    'age': [25, 30, np.nan, 25, 40],
    'salary': [50000, 60000, 55000, 50000, 70000]
}
df = pd.DataFrame(data)

print("Original dataset:")
print(df)

# Clean the data
df_cleaned = df.dropna()  # Remove rows with missing values
df_cleaned = df_cleaned.drop_duplicates()  # Remove duplicate rows

print("\nCleaned dataset:")
print(df_cleaned)
```

This code demonstrates basic data cleaning techniques such as removing missing values and duplicates.

Slide 2: Avoiding Overfitting

Overfitting occurs when a model learns the training data too well, including its noise and fluctuations. This leads to poor generalization on unseen data. Let's illustrate overfitting using a polynomial regression example:

```python
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and fit the models
degrees = [1, 3, 15]
plt.figure(figsize=(14, 4))

for i, degree in enumerate(degrees):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X_train)
    
    model = LinearRegression()
    model.fit(X_poly, y_train)
    
    X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
    y_plot = model.predict(poly_features.transform(X_plot))
    
    plt.subplot(1, 3, i + 1)
    plt.scatter(X_train, y_train, color='r', s=10, label='Training data')
    plt.plot(X_plot, y_plot, color='b', label='Model')
    plt.title(f'Degree {degree}')
    plt.legend()

plt.tight_layout()
plt.show()
```

This example shows how increasing the polynomial degree can lead to overfitting.

Slide 3: The Value of Exploratory Data Analysis (EDA)

EDA helps uncover patterns, relationships, and anomalies in the data. It's a crucial step before model building. Let's perform a simple EDA on the Iris dataset:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Pairplot to visualize relationships between features
sns.pairplot(df, hue='species', height=2.5)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Iris Dataset')
plt.show()
```

This code creates a pairplot and a correlation heatmap, revealing relationships between features and species.

Slide 4: Proper Model Validation

Validation is crucial for assessing a model's performance on unseen data. Cross-validation is a powerful technique for this purpose. Let's implement k-fold cross-validation:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(clf, X, y, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Standard deviation of CV score:", cv_scores.std())
```

This example demonstrates how to use cross-validation to get a more robust estimate of model performance.

Slide 5: Beyond Accuracy: Comprehensive Model Evaluation

While accuracy is important, it's not always the best metric, especially for imbalanced datasets. Let's explore other metrics using a binary classification example:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.datasets import make_classification

# Generate an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"AUC-ROC: {auc_roc:.3f}")
```

This code calculates various metrics to provide a more comprehensive evaluation of the model's performance.

Slide 6: Starting Simple: The Power of Basic Models

While complex models are powerful, simpler models often perform well and are easier to interpret. Let's compare a simple linear regression with a more complex polynomial regression:

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Fit linear regression
lr = LinearRegression()
lr.fit(X, y)

# Fit polynomial regression
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
pr = LinearRegression()
pr.fit(X_poly, y)

# Make predictions
X_test = np.linspace(0, 5, 100).reshape(-1, 1)
y_lr = lr.predict(X_test)
y_pr = pr.predict(poly.transform(X_test))

# Calculate MSE
mse_lr = mean_squared_error(y, lr.predict(X))
mse_pr = mean_squared_error(y, pr.predict(X_poly))

# Plot results
plt.scatter(X, y, color='r', label='Data')
plt.plot(X_test, y_lr, color='b', label='Linear Regression')
plt.plot(X_test, y_pr, color='g', label='Polynomial Regression')
plt.legend()
plt.title('Linear vs Polynomial Regression')
plt.show()

print(f"MSE (Linear): {mse_lr:.4f}")
print(f"MSE (Polynomial): {mse_pr:.4f}")
```

This example compares a simple linear regression with a more complex polynomial regression, showing that sometimes simpler models can perform well.

Slide 7: Real-Life Example: Predicting House Prices

Let's apply what we've learned to a real-world scenario: predicting house prices. We'll use a simplified version of the Boston Housing dataset:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

# Load the Boston Housing dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# Select a few features for simplicity
features = ['RM', 'LSTAT', 'PTRATIO']
X = df[features]
y = df['PRICE']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Plot actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
```

This example demonstrates how to build and evaluate a simple house price prediction model using real-world data.

Slide 8: Real-Life Example: Customer Churn Prediction

Let's explore another real-world scenario: predicting customer churn for a telecommunications company. We'll use a simplified dataset and focus on data cleaning, exploratory data analysis, and model building:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create a sample dataset
np.random.seed(42)
n_samples = 1000

data = {
    'tenure': np.random.randint(1, 72, n_samples),
    'monthly_charges': np.random.uniform(20, 100, n_samples),
    'total_charges': np.random.uniform(100, 5000, n_samples),
    'contract_type': np.random.choice(['Monthly', 'One year', 'Two year'], n_samples),
    'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
}

df = pd.DataFrame(data)

# Data cleaning
df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
df.dropna(inplace=True)

# Exploratory Data Analysis
plt.figure(figsize=(12, 5))
plt.subplot(121)
sns.boxplot(x='contract_type', y='monthly_charges', data=df)
plt.title('Monthly Charges by Contract Type')

plt.subplot(122)
sns.histplot(data=df, x='tenure', hue='churn', multiple='stack', bins=20)
plt.title('Tenure Distribution by Churn Status')
plt.show()

# Prepare data for modeling
X = pd.get_dummies(df.drop('churn', axis=1), drop_first=True)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

This example covers data cleaning, exploratory data analysis, and building a churn prediction model using a Random Forest classifier.

Slide 9: Handling Imbalanced Datasets

Imbalanced datasets are common in real-world scenarios, such as fraud detection or rare disease diagnosis. Let's explore techniques to handle imbalanced data:

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Generate an imbalanced dataset
X, y = make_classification(n_samples=10000, n_classes=2, weights=[0.95, 0.05], random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define resampling strategies
over = SMOTE(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy=0.5)

# Create a pipeline with SMOTE, undersampling, and Random Forest
pipeline = Pipeline([
    ('over', over),
    ('under', under),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))
```

This example demonstrates how to use SMOTE (Synthetic Minority Over-sampling Technique) and Random Undersampling to balance the dataset before training a Random Forest classifier.

Slide 10: Feature Engineering and Selection

Feature engineering and selection are crucial steps in improving model performance. Let's explore some techniques:

```python
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Engineering: Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Feature Selection: Select top k features
selector = SelectKBest(f_regression, k=10)
X_train_selected = selector.fit_transform(X_train_poly, y_train)
X_test_selected = selector.transform(X_test_poly)

# Train a model with selected features
model = LinearRegression()
model.fit(X_train_selected, y_train)

# Make predictions
y_pred = model.predict(X_test_selected)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
```

This example demonstrates polynomial feature engineering and feature selection using SelectKBest.

Slide 11: Handling Missing Data

Missing data is a common issue in real-world datasets. Let's explore techniques to handle missing values:

```python
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Create a sample dataset with missing values
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [5, np.nan, 7, 8, np.nan],
    'C': [9, 10, 11, np.nan, 13]
}
df = pd.DataFrame(data)

print("Original dataset:")
print(df)

# Simple Imputation (mean strategy)
imp_mean = SimpleImputer(strategy='mean')
df_mean_imputed = pd.DataFrame(imp_mean.fit_transform(df), columns=df.columns)

print("\nMean Imputation:")
print(df_mean_imputed)

# KNN Imputation
imp_knn = KNNImputer(n_neighbors=2)
df_knn_imputed = pd.DataFrame(imp_knn.fit_transform(df), columns=df.columns)

print("\nKNN Imputation:")
print(df_knn_imputed)

# Multiple Imputation by Chained Equations (MICE)
imp_mice = IterativeImputer(random_state=0)
df_mice_imputed = pd.DataFrame(imp_mice.fit_transform(df), columns=df.columns)

print("\nMICE Imputation:")
print(df_mice_imputed)
```

This example demonstrates three different imputation techniques: mean imputation, K-Nearest Neighbors imputation, and Multiple Imputation by Chained Equations (MICE).

Slide 12: Model Interpretability

As models become more complex, interpretability becomes crucial. Let's explore some techniques for interpreting machine learning models:

```python
from sklearn.inspection import partial_dependence, plot_partial_dependence
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# Load the Boston Housing dataset
boston = load_boston()
X, y = boston.data, boston.target
feature_names = boston.feature_names

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Calculate feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Compute and plot partial dependence for the two most important features
fig, ax = plt.subplots(figsize=(10, 6))
plot_partial_dependence(rf_model, X, [indices[0], indices[1]], 
                        feature_names=feature_names, ax=ax)
plt.tight_layout()
plt.show()
```

This example demonstrates how to calculate and visualize feature importances and partial dependence plots for a Random Forest model.

Slide 13: Hyperparameter Tuning

Optimizing model hyperparameters is crucial for achieving the best performance. Let's explore grid search and random search for hyperparameter tuning:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters (Grid Search):", grid_search.best_params_)
print("Best score (Grid Search):", grid_search.best_score_)

# Random Search
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=20, cv=5, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

print("\nBest parameters (Random Search):", random_search.best_params_)
print("Best score (Random Search):", random_search.best_score_)
```

This example demonstrates how to use GridSearchCV and RandomizedSearchCV for hyperparameter tuning of a Random Forest classifier.

Slide 14: Additional Resources

For further learning and exploration in data science and machine learning, consider the following resources:

1. ArXiv.org: A repository of scientific papers, including many on machine learning and data science. URL: [https://arxiv.org/list/stat.ML/recent](https://arxiv.org/list/stat.ML/recent)
2. Scikit-learn Documentation: Comprehensive guide to the Scikit-learn library. URL: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
3. Towards Data Science: A Medium publication featuring articles on various data science topics. URL: [https://towardsdatascience.com/](https://towardsdatascience.com/)
4. Kaggle: A platform for data science competitions and datasets. URL: [https://www.kaggle.com/](https://www.kaggle.com/)
5. Machine Learning Mastery: A blog with practical tutorials on machine learning. URL: [https://machinelearningmastery.com/](https://machinelearningmastery.com/)

These resources offer a wealth of information for beginners and intermediate practitioners in data science and machine learning.


