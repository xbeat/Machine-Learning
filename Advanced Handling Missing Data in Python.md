## Advanced Handling Missing Data in Python
Slide 1: Introduction to Missing Data

Missing data is a common problem in real-world datasets. It can occur due to various reasons such as data collection errors, sensor malfunctions, or respondents skipping questions. Handling missing data is crucial for maintaining the integrity and reliability of our analyses. In this presentation, we'll explore different techniques to handle missing data using Python, focusing on kNN imputation, MissForest, and multiple imputation methods.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a sample dataset with missing values
np.random.seed(42)
data = pd.DataFrame({
    'A': np.random.rand(100),
    'B': np.random.rand(100),
    'C': np.random.rand(100)
})
data.loc[np.random.choice(data.index, 20), 'A'] = np.nan
data.loc[np.random.choice(data.index, 15), 'B'] = np.nan
data.loc[np.random.choice(data.index, 10), 'C'] = np.nan

# Visualize missing data
plt.figure(figsize=(10, 6))
plt.imshow(data.isnull(), cmap='binary', aspect='auto')
plt.title('Missing Data Visualization')
plt.xlabel('Features')
plt.ylabel('Samples')
plt.show()
```

Slide 2: Understanding kNN Imputation

kNN (k-Nearest Neighbors) imputation is a method that fills in missing values by finding the k most similar samples and using their values. This technique works well when there's a strong correlation between features. The 'k' parameter determines the number of neighbors to consider, balancing between local and global information.

```python
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform kNN imputation
imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(data_scaled)

# Convert back to original scale
data_imputed = scaler.inverse_transform(data_imputed)

print("Original data:\n", data.head())
print("\nImputed data:\n", pd.DataFrame(data_imputed, columns=data.columns).head())
```

Slide 3: Implementing kNN Imputation

Let's implement kNN imputation on a real-world dataset. We'll use the Boston Housing dataset, which contains information about various features of houses in Boston and their prices. Some values in this dataset might be missing, so we'll apply kNN imputation to fill them in.

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load the Boston Housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Introduce some missing values
rng = np.random.RandomState(42)
X_missing = X.()
X_missing[rng.rand(*X.shape) < 0.1] = np.nan

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_missing, y, test_size=0.2, random_state=42)

# Apply kNN imputation
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train a model and evaluate
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train_imputed, y_train)
y_pred = rf.predict(X_test_imputed)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error after kNN imputation: {mse:.4f}")
```

Slide 4: Introduction to MissForest

MissForest is an iterative imputation method that uses random forests to predict missing values. It can handle mixed-type data and capture complex interactions between features. MissForest works by initially imputing missing values with mean/mode, then iteratively improving the imputation using random forest models.

```python
from missingpy import MissForest

# Create a dataset with mixed types
data_mixed = pd.DataFrame({
    'numeric': np.random.rand(100),
    'categorical': np.random.choice(['A', 'B', 'C'], 100),
    'ordinal': np.random.randint(1, 6, 100)
})

# Introduce missing values
data_mixed.loc[np.random.choice(data_mixed.index, 20), 'numeric'] = np.nan
data_mixed.loc[np.random.choice(data_mixed.index, 15), 'categorical'] = np.nan
data_mixed.loc[np.random.choice(data_mixed.index, 10), 'ordinal'] = np.nan

# Apply MissForest imputation
imputer = MissForest()
data_imputed = imputer.fit_transform(data_mixed)

print("Original data:\n", data_mixed.head())
print("\nImputed data:\n", pd.DataFrame(data_imputed, columns=data_mixed.columns).head())
```

Slide 5: MissForest in Action

Let's apply MissForest to a real-world scenario using the Titanic dataset. This dataset contains information about passengers on the Titanic, including whether they survived or not. We'll use MissForest to impute missing values in features like Age and Fare, which can be crucial for predicting survival.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load Titanic dataset
titanic = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')

# Select features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = titanic[features]
y = titanic['Survived']

# Convert categorical variables to numeric
X['Sex'] = X['Sex'].map({'female': 0, 'male': 1})
X['Embarked'] = X['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply MissForest imputation
imputer = MissForest(random_state=42)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train a model and evaluate
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_imputed, y_train)
y_pred = rf.predict(X_test_imputed)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after MissForest imputation: {accuracy:.4f}")
```

Slide 6: Multiple Imputation: Concept and Theory

Multiple Imputation is a statistical technique that creates multiple plausible imputed datasets, analyzes each dataset separately, and then combines the results. This method accounts for the uncertainty in the missing data by creating these multiple datasets. It's particularly useful when the missing data mechanism is not completely random.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Simulate the multiple imputation process
np.random.seed(42)
true_mean = 5
true_std = 2
sample_size = 100

# Generate complete data
complete_data = np.random.normal(true_mean, true_std, sample_size)

# Introduce missing values
missing_mask = np.random.choice([True, False], size=sample_size, p=[0.2, 0.8])
incomplete_data = np.where(missing_mask, np.nan, complete_data)

# Perform multiple imputation (simplified)
num_imputations = 5
imputed_datasets = []
for _ in range(num_imputations):
    imputed_data = incomplete_data.()
    imputed_values = np.random.normal(np.nanmean(incomplete_data), np.nanstd(incomplete_data), 
                                      size=np.sum(missing_mask))
    imputed_data[missing_mask] = imputed_values
    imputed_datasets.append(imputed_data)

# Plot the results
plt.figure(figsize=(12, 6))
for i, dataset in enumerate(imputed_datasets):
    plt.hist(dataset, alpha=0.3, label=f'Imputation {i+1}')
plt.hist(complete_data, alpha=0.5, color='red', label='True Data')
plt.title('Multiple Imputation Results')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

Slide 7: Implementing Multiple Imputation

Let's implement Multiple Imputation using the `mice` (Multivariate Imputation by Chained Equations) package in Python. We'll use a real-world dataset, the Iris dataset, and introduce some missing values to demonstrate the technique.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import miceforest as mf

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# Introduce missing values
rng = np.random.RandomState(42)
X_missing = X.()
X_missing[rng.rand(*X.shape) < 0.2] = np.nan

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_missing, y, test_size=0.2, random_state=42)

# Perform multiple imputation
kernel = mf.KernelDataSet(X_train, variable_schema=feature_names)
kernel.mice(3)  # Perform 3 iterations

# Create multiple imputed datasets
imputed_datasets = kernel.complete_data(3)  # Create 3 imputed datasets

# Train models on each imputed dataset
models = []
for dataset in imputed_datasets:
    rf = RandomForestClassifier(random_state=42)
    rf.fit(dataset, y_train)
    models.append(rf)

# Predict on test set (using the first imputed dataset for simplicity)
X_test_imputed = kernel.impute_new_data(X_test)
y_pred = models[0].predict(X_test_imputed.complete_data(1)[0])
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after Multiple Imputation: {accuracy:.4f}")
```

Slide 8: Model-Based Imputation

Model-based imputation uses statistical models to predict missing values based on other variables in the dataset. This approach can capture complex relationships between variables and is particularly useful when the missing data mechanism is related to observed variables.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

# Create a sample dataset with missing values
np.random.seed(42)
X = np.random.rand(100, 5)
X[np.random.rand(*X.shape) < 0.2] = np.nan

# Perform model-based imputation using ExtraTrees
imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=42),
                           random_state=42, max_iter=10)
X_imputed = imputer.fit_transform(X)

print("Original data (first 5 rows):\n", X[:5])
print("\nImputed data (first 5 rows):\n", X_imputed[:5])

# Visualize the imputation results
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(X, aspect='auto', cmap='viridis')
plt.title('Original Data')
plt.subplot(122)
plt.imshow(X_imputed, aspect='auto', cmap='viridis')
plt.title('Imputed Data')
plt.tight_layout()
plt.show()
```

Slide 9: Comparing Imputation Methods

Let's compare the performance of different imputation methods on a real-world dataset. We'll use the California Housing dataset and introduce missing values to test kNN imputation, MissForest, and Multiple Imputation.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from missingpy import MissForest
import miceforest as mf

# Load and prepare the dataset
california = fetch_california_housing()
X, y = california.data, california.target

# Introduce missing values
rng = np.random.RandomState(42)
X_missing = X.()
X_missing[rng.rand(*X.shape) < 0.2] = np.nan

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_missing, y, test_size=0.2, random_state=42)

# Define a function to evaluate imputation methods
def evaluate_imputation(X_train_imp, X_test_imp):
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train_imp, y_train)
    y_pred = rf.predict(X_test_imp)
    return mean_squared_error(y_test, y_pred)

# kNN Imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_train_knn = knn_imputer.fit_transform(X_train)
X_test_knn = knn_imputer.transform(X_test)
mse_knn = evaluate_imputation(X_train_knn, X_test_knn)

# MissForest
mf_imputer = MissForest(random_state=42)
X_train_mf = mf_imputer.fit_transform(X_train)
X_test_mf = mf_imputer.transform(X_test)
mse_mf = evaluate_imputation(X_train_mf, X_test_mf)

# Multiple Imputation (using miceforest)
kernel = mf.KernelDataSet(X_train, save_all_iterations=True, random_state=42)
kernel.mice(3)
X_train_mi = kernel.complete_data(1)[0]
X_test_mi = kernel.impute_new_data(X_test).complete_data(1)[0]
mse_mi = evaluate_imputation(X_train_mi, X_test_mi)

print(f"MSE (kNN): {mse_knn:.4f}")
print(f"MSE (MissForest): {mse_mf:.4f}")
print(f"MSE (Multiple Imputation): {mse_mi:.4f}")
```

Slide 10: Handling Time Series Data

Imputing missing values in time series data requires special consideration due to the temporal nature of the data. We'll demonstrate how to handle missing values in a time series using forward fill and interpolation methods.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a sample time series with missing values
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.random.randn(len(dates))
ts = pd.Series(values, index=dates)
ts[ts.sample(frac=0.2).index] = np.nan

# Forward fill
ts_ffill = ts.ffill()

# Interpolation
ts_interp = ts.interpolate()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(ts.index, ts, label='Original', alpha=0.7)
plt.plot(ts_ffill.index, ts_ffill, label='Forward Fill', alpha=0.7)
plt.plot(ts_interp.index, ts_interp, label='Interpolation', alpha=0.7)
plt.title('Time Series Imputation Methods')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Calculate and print the mean absolute error for each method
mae_ffill = np.abs(ts.dropna() - ts_ffill.loc[ts.dropna().index]).mean()
mae_interp = np.abs(ts.dropna() - ts_interp.loc[ts.dropna().index]).mean()
print(f"MAE (Forward Fill): {mae_ffill:.4f}")
print(f"MAE (Interpolation): {mae_interp:.4f}")
```

Slide 11: Handling Categorical Data

Imputing missing values in categorical data presents unique challenges. We'll explore methods like mode imputation and using a separate category for missing values.

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Create a sample dataset with categorical variables
data = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', np.nan, 'Red', 'Blue', np.nan],
    'Size': ['Small', 'Medium', np.nan, 'Large', 'Small', np.nan, 'Medium']
})

# Mode imputation
mode_imputer = SimpleImputer(strategy='most_frequent')
data_mode = pd.DataFrame(mode_imputer.fit_transform(data), columns=data.columns)

# Separate category for missing values
data_missing_category = data.fillna('Missing')

# One-hot encoding with missing values as a separate category
onehot = OneHotEncoder(sparse=False, handle_unknown='ignore')
data_onehot = pd.DataFrame(onehot.fit_transform(data_missing_category),
                           columns=onehot.get_feature_names(data.columns))

print("Original data:\n", data)
print("\nMode imputation:\n", data_mode)
print("\nMissing as separate category:\n", data_missing_category)
print("\nOne-hot encoded (first few columns):\n", data_onehot.iloc[:, :5])
```

Slide 12: Evaluating Imputation Quality

Assessing the quality of imputation is crucial. We'll demonstrate methods to evaluate imputation performance, including cross-validation and comparing imputed vs. known values.

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_regression

# Create a dataset with known values
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_missing = X.()

# Introduce missing values
np.random.seed(42)
missing_rate = 0.2
mask = np.random.rand(*X.shape) < missing_rate
X_missing[mask] = np.nan

# Perform imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_missing)

# Evaluate using cross-validation
rf = RandomForestRegressor(random_state=42)
scores_original = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')
scores_imputed = cross_val_score(rf, X_imputed, y, cv=5, scoring='neg_mean_squared_error')

print("MSE with original data: {:.4f} (+/- {:.4f})".format(-scores_original.mean(), scores_original.std() * 2))
print("MSE with imputed data: {:.4f} (+/- {:.4f})".format(-scores_imputed.mean(), scores_imputed.std() * 2))

# Compare imputed vs. known values
known_values = X[~mask]
imputed_values = X_imputed[mask]
mse = np.mean((known_values - imputed_values) ** 2)
print(f"MSE between known and imputed values: {mse:.4f}")
```

Slide 13: Handling Missing Data in Production

When deploying models with imputation in production, it's important to handle new missing data consistently. Here's an example of how to save and reuse an imputer in a production setting.

```python
import joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# Create a sample dataset
X = np.random.rand(100, 5)
y = np.random.rand(100)
X[np.random.rand(*X.shape) < 0.2] = np.nan

# Create a pipeline with imputation and model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', RandomForestRegressor(random_state=42))
])

# Fit the pipeline
pipeline.fit(X, y)

# Save the fitted pipeline
joblib.dump(pipeline, 'model_with_imputer.joblib')

# In production:
# Load the pipeline
loaded_pipeline = joblib.load('model_with_imputer.joblib')

# New data with missing values
X_new = np.random.rand(10, 5)
X_new[np.random.rand(*X_new.shape) < 0.2] = np.nan

# Make predictions
predictions = loaded_pipeline.predict(X_new)
print("Predictions:", predictions)
```

Slide 14: Additional Resources

For further exploration of missing data handling techniques, consider the following resources:

1. "Missing Data Mechanisms and Bayesian Inference" by Xu et al. (2022) ArXiv: [https://arxiv.org/abs/2206.02324](https://arxiv.org/abs/2206.02324)
2. "Multiple Imputation by Chained Equations: What is it and how does it work?" by Azur et al. (2011) International Journal of Methods in Psychiatric Research DOI: 10.1002/mpr.329
3. "A Review of Missing Data Handling Methods in Education Research" by Peugh and Enders (2004) Review of Educational Research DOI: 10.3102/00346543074004525

These papers provide in-depth discussions on various aspects of missing data imputation and their applications in different fields.

