## Strategies for Handling Missing Data Types Using Python
Slide 1: Introduction to Missing Data

Missing data is a common challenge in data analysis. Understanding the types of missing data and appropriate strategies for handling them is crucial for accurate results. This presentation will cover three main types of missing data: Missing Completely at Random (MCAR), Missing at Random (MAR), and Missing Not at Random (MNAR), along with Python-based strategies to address each type.

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

# Introduce missing values
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

Slide 2: Missing Completely at Random (MCAR)

MCAR occurs when the probability of missing data is the same for all observations. In this case, the missingness is unrelated to both observed and unobserved data. For example, in a survey, some participants might randomly forget to answer certain questions.

```python
def is_mcar(data):
    # Split data into groups with and without missing values
    missing = data[data.isnull().any(axis=1)]
    complete = data.dropna()
    
    # Perform t-test for each variable
    for col in data.columns:
        t_stat, p_value = stats.ttest_ind(
            complete[col],
            missing[col].dropna()
        )
        print(f"{col}: p-value = {p_value:.4f}")
    
    # If all p-values are > 0.05, data might be MCAR
    return all(stats.ttest_ind(complete[col], missing[col].dropna())[1] > 0.05 for col in data.columns)

# Example usage
from scipy import stats
is_mcar(data)
```

Slide 3: Handling MCAR Data - Listwise Deletion

For MCAR data, listwise deletion (complete case analysis) is a simple and often effective method. It involves removing all cases with any missing values. This approach is suitable when the proportion of missing data is small and the sample size is large.

```python
# Listwise deletion
data_complete = data.dropna()

print("Original data shape:", data.shape)
print("Data shape after listwise deletion:", data_complete.shape)

# Calculate and print the percentage of data lost
percent_lost = (1 - len(data_complete) / len(data)) * 100
print(f"Percentage of data lost: {percent_lost:.2f}%")
```

Slide 4: Handling MCAR Data - Mean Imputation

Another strategy for MCAR data is mean imputation. This method replaces missing values with the mean of the observed values for that variable. While simple, it can underestimate the variance and distort relationships between variables.

```python
# Mean imputation
data_mean_imputed = data.fillna(data.mean())

# Verify imputation
print("Original data:")
print(data.isnull().sum())
print("\nData after mean imputation:")
print(data_mean_imputed.isnull().sum())

# Compare original and imputed distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
data['A'].hist(ax=ax1, bins=20)
ax1.set_title('Original Distribution (with missing values)')
data_mean_imputed['A'].hist(ax=ax2, bins=20)
ax2.set_title('Distribution after Mean Imputation')
plt.tight_layout()
plt.show()
```

Slide 5: Missing at Random (MAR)

MAR occurs when the probability of missing data depends on observed data but not on the missing data itself. For instance, in a study on depression, older participants might be less likely to report their income, but this missingness is unrelated to the actual income amount.

```python
import seaborn as sns

# Create a dataset with MAR
np.random.seed(42)
age = np.random.normal(40, 10, 1000)
income = 1000 + 50 * age + np.random.normal(0, 1000, 1000)
mar_data = pd.DataFrame({'Age': age, 'Income': income})

# Introduce MAR: older people are more likely to have missing income
mar_data.loc[mar_data['Age'] > mar_data['Age'].median(), 'Income'] = np.where(
    np.random.rand(sum(mar_data['Age'] > mar_data['Age'].median())) < 0.5,
    np.nan,
    mar_data.loc[mar_data['Age'] > mar_data['Age'].median(), 'Income']
)

# Visualize MAR pattern
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Income', data=mar_data)
plt.title('Income vs Age (MAR pattern)')
plt.show()
```

Slide 6: Handling MAR Data - Multiple Imputation

Multiple Imputation is a powerful method for handling MAR data. It creates multiple plausible imputed datasets, analyzes each separately, and then combines the results. This approach accounts for the uncertainty in the missing values.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# Multiple Imputation using IterativeImputer
imp = IterativeImputer(estimator=BayesianRidge(), n_iter=10, random_state=42)
mar_data_imputed = pd.DataFrame(imp.fit_transform(mar_data), columns=mar_data.columns)

# Visualize original and imputed data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.scatterplot(x='Age', y='Income', data=mar_data, ax=ax1)
ax1.set_title('Original Data (with MAR)')
sns.scatterplot(x='Age', y='Income', data=mar_data_imputed, ax=ax2)
ax2.set_title('Data after Multiple Imputation')
plt.tight_layout()
plt.show()
```

Slide 7: Handling MAR Data - Regression Imputation

Regression imputation is another method for handling MAR data. It uses the relationship between variables to predict missing values. While it can preserve relationships between variables, it may underestimate standard errors.

```python
from sklearn.linear_model import LinearRegression

# Separate complete cases and cases with missing Income
complete_cases = mar_data.dropna()
missing_income = mar_data[mar_data['Income'].isnull()]

# Train a linear regression model
model = LinearRegression()
model.fit(complete_cases[['Age']], complete_cases['Income'])

# Impute missing values
imputed_income = model.predict(missing_income[['Age']])
mar_data.loc[mar_data['Income'].isnull(), 'Income'] = imputed_income

# Visualize the imputed data
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Income', data=mar_data, hue=mar_data['Income'].isnull())
plt.title('Income vs Age after Regression Imputation')
plt.legend(['Original', 'Imputed'])
plt.show()
```

Slide 8: Missing Not at Random (MNAR)

MNAR occurs when the probability of missing data depends on unobserved data. For example, in a survey about income, high-income individuals might be less likely to report their income. This type of missingness is the most challenging to handle.

```python
# Create a dataset with MNAR
np.random.seed(42)
true_income = np.random.lognormal(mean=10, sigma=1, size=1000)
reported_income = np.where(true_income > np.percentile(true_income, 75),
                           np.nan,
                           true_income)

mnar_data = pd.DataFrame({'True Income': true_income, 'Reported Income': reported_income})

# Visualize MNAR pattern
plt.figure(figsize=(10, 6))
plt.hist(mnar_data['True Income'], bins=30, alpha=0.5, label='True Income')
plt.hist(mnar_data['Reported Income'].dropna(), bins=30, alpha=0.5, label='Reported Income')
plt.title('Distribution of True vs Reported Income (MNAR)')
plt.legend()
plt.show()
```

Slide 9: Handling MNAR Data - Sensitivity Analysis

For MNAR data, there's no perfect solution. Sensitivity analysis involves testing different assumptions about the missing data mechanism and comparing results. This helps understand how robust the conclusions are to different missing data scenarios.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to perform sensitivity analysis
def sensitivity_analysis(data, imputation_methods):
    results = {}
    for method, imputer in imputation_methods.items():
        imputed_data = data.()
        imputed_data['Reported Income'] = imputer(imputed_data['Reported Income'])
        results[method] = imputed_data['Reported Income'].mean()
    
    return results

# Define imputation methods
imputation_methods = {
    'Mean': lambda x: x.fillna(x.mean()),
    'Median': lambda x: x.fillna(x.median()),
    'Max': lambda x: x.fillna(x.max()),
    '90th Percentile': lambda x: x.fillna(x.quantile(0.9))
}

# Perform sensitivity analysis
results = sensitivity_analysis(mnar_data, imputation_methods)

# Visualize results
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.axhline(mnar_data['True Income'].mean(), color='r', linestyle='--', label='True Mean')
plt.title('Sensitivity Analysis for MNAR Data')
plt.ylabel('Mean Income')
plt.legend()
plt.show()
```

Slide 10: Handling MNAR Data - Pattern Mixture Models

Pattern Mixture Models are a more advanced approach for MNAR data. They model the joint distribution of the data and the missingness mechanism, allowing for different distributions for observed and missing data.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Simulate pattern mixture model
np.random.seed(42)
n = 1000
x = np.random.normal(0, 1, n)
y = 2 * x + np.random.normal(0, 1, n)
missing = np.random.binomial(1, 1 / (1 + np.exp(-y)), n)
y[missing == 1] = np.nan

# Fit separate models for observed and missing data
obs_model = stats.linregress(x[~np.isnan(y)], y[~np.isnan(y)])
missing_model = stats.linregress(x[np.isnan(y)], x[np.isnan(y)] * 3)  # Assume different relationship

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(x[~np.isnan(y)], y[~np.isnan(y)], alpha=0.5, label='Observed')
plt.scatter(x[np.isnan(y)], x[np.isnan(y)] * 3, alpha=0.5, label='Imputed')
plt.plot(x, obs_model.intercept + obs_model.slope * x, 'r', label='Observed Model')
plt.plot(x, missing_model.intercept + missing_model.slope * x, 'g', label='Missing Model')
plt.legend()
plt.title('Pattern Mixture Model for MNAR Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

Slide 11: Real-life Example 1: Customer Churn Prediction

In customer churn prediction, missing data is common. For instance, some customers might not provide their age or income. This could be MAR if older customers are less likely to provide their age, or MNAR if high-income customers are less likely to report their income.

```python
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create a sample customer dataset
np.random.seed(42)
n_customers = 1000
age = np.random.normal(40, 15, n_customers)
income = 30000 + 1000 * age + np.random.normal(0, 10000, n_customers)
churn = (0.1 * age + 0.00002 * income + np.random.normal(0, 2, n_customers) > 5).astype(int)

# Introduce missing values (MAR for age, MNAR for income)
age[np.random.rand(n_customers) < 0.2] = np.nan  # 20% missing, MAR
income[income > np.percentile(income, 80)] = np.nan  # MNAR for high incomes

data = pd.DataFrame({'Age': age, 'Income': income, 'Churn': churn})

# Impute missing values using KNN
imputer = KNNImputer(n_neighbors=5)
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Split data and train a Random Forest classifier
X = data_imputed[['Age', 'Income']]
y = data_imputed['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

Slide 12: Real-life Example 2: Medical Research

In medical research, missing data is a common challenge. For example, in a study on the effectiveness of a new drug, some patients might drop out, leading to missing follow-up data. This could be MAR if dropout is related to observed side effects, or MNAR if it's related to the unobserved effectiveness of the drug.

```python
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Create a sample medical study dataset
np.random.seed(42)
n_patients = 500
baseline_severity = np.random.normal(5, 2, n_patients)
treatment = np.random.binomial(1, 0.5, n_patients)
effectiveness = 3 * treatment - 0.5 * baseline_severity + np.random.normal(0, 1, n_patients)
dropout = np.random.binomial(1, 1 / (1 + np.exp(3 - effectiveness)), n_patients)

data = pd.DataFrame({
    'Baseline_Severity': baseline_severity,
    'Treatment': treatment,
    'Effectiveness': effectiveness
})

# Introduce missing values (MNAR)
data.loc[dropout == 1, 'Effectiveness'] = np.nan

# Multiple imputation using IterativeImputer
imp = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10), n_iter=10, random_state=42)
data_imputed = pd.DataFrame(imp.fit_transform(data), columns=data.columns)

# Compare original and imputed data
print("Original data:\n", data.describe())
print("\nImputed data:\n", data_imputed.describe())

# Analyze treatment effect
original_effect = data[data['Treatment'] == 1]['Effectiveness'].mean() - data[data['Treatment'] == 0]['Effectiveness'].mean()
imputed_effect = data_imputed[data_imputed['Treatment'] == 1]['Effectiveness'].mean() - data_imputed[data_imputed['Treatment'] == 0]['Effectiveness'].mean()

print(f"\nEstimated treatment effect (original data): {original_effect:.2f}")
print(f"Estimated treatment effect (imputed data): {imputed_effect:.2f}")
```

Slide 13: Comparing Imputation Methods

Different imputation methods can lead to varying results. It's crucial to compare multiple approaches and understand their impact on your analysis. This slide demonstrates how to compare simple imputation methods with more advanced techniques like multiple imputation.

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Create a dataset with missing values
np.random.seed(42)
n = 1000
X = np.random.randn(n, 3)
y = X[:, 0] + 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n) * 0.5
X_missing = X.()
X_missing[np.random.rand(*X.shape) < 0.2] = np.nan

data = pd.DataFrame(X_missing, columns=['X1', 'X2', 'X3'])
data['y'] = y

# Define imputation methods
imputers = {
    'Mean': SimpleImputer(strategy='mean'),
    'Median': SimpleImputer(strategy='median'),
    'KNN': KNNImputer(n_neighbors=5),
    'Multiple': IterativeImputer(estimator=RandomForestRegressor(n_estimators=10), random_state=42)
}

# Perform imputations and calculate MSE
results = {}
for name, imputer in imputers.items():
    X_imputed = imputer.fit_transform(data[['X1', 'X2', 'X3']])
    data_imputed = pd.DataFrame(X_imputed, columns=['X1', 'X2', 'X3'])
    data_imputed['y'] = data['y']
    mse = ((data_imputed[['X1', 'X2', 'X3']] - X) ** 2).mean().mean()
    results[name] = mse

# Print results
for name, mse in results.items():
    print(f"{name} Imputation MSE: {mse:.4f}")

# Visualize results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.title('Comparison of Imputation Methods')
plt.ylabel('Mean Squared Error')
plt.show()
```

Slide 14: Handling Missing Data in Time Series

Time series data presents unique challenges for handling missing values. Methods like forward fill, backward fill, or interpolation are commonly used. Here's an example of how to handle missing data in a time series using different techniques.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a sample time series with missing values
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
ts = pd.Series(values, index=dates)

# Introduce missing values
ts[ts.sample(frac=0.2).index] = np.nan

# Apply different imputation methods
ts_ffill = ts.ffill()
ts_bfill = ts.bfill()
ts_interpolate = ts.interpolate()

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(ts.index, ts, 'o', label='Original (with missing)', alpha=0.5)
plt.plot(ts_ffill.index, ts_ffill, label='Forward Fill')
plt.plot(ts_bfill.index, ts_bfill, label='Backward Fill')
plt.plot(ts_interpolate.index, ts_interpolate, label='Interpolation')
plt.title('Time Series Imputation Methods')
plt.legend()
plt.show()

# Calculate and print Mean Absolute Error for each method
original = values
mae_ffill = np.abs(original - ts_ffill).mean()
mae_bfill = np.abs(original - ts_bfill).mean()
mae_interpolate = np.abs(original - ts_interpolate).mean()

print(f"MAE Forward Fill: {mae_ffill:.4f}")
print(f"MAE Backward Fill: {mae_bfill:.4f}")
print(f"MAE Interpolation: {mae_interpolate:.4f}")
```

Slide 15: Additional Resources

For further exploration of missing data handling techniques, consider the following resources:

1. "Missing Data: Our View of the State of the Art" by Roderick J. A. Little and Donald B. Rubin (ArXiv:1910.04507) URL: [https://arxiv.org/abs/1910.04507](https://arxiv.org/abs/1910.04507)
2. "Multiple Imputation for Nonresponse in Surveys" by Donald B. Rubin (Book)
3. "Flexible Imputation of Missing Data" by Stef van Buuren (Book)
4. Scikit-learn documentation on imputation techniques: [https://scikit-learn.org/stable/modules/impute.html](https://scikit-learn.org/stable/modules/impute.html)
5. Pandas documentation on handling missing data: [https://pandas.pydata.org/pandas-docs/stable/user\_guide/missing\_data.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html)

These resources provide in-depth discussions on the theory and practical applications of missing data techniques, helping you choose the most appropriate method for your specific use case.

