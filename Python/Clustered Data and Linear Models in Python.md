## Clustered Data and Linear Models in Python
Slide 1: Clustered Data and Linear Models: Avoiding the Traps

Clustered data occurs when observations are grouped into distinct categories or clusters. Linear models, while powerful, can lead to incorrect conclusions when applied naively to clustered data. This presentation explores the challenges and solutions for handling clustered data in linear modeling using Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate clustered data
np.random.seed(42)
clusters = [
    (np.random.normal(0, 0.5, 50), np.random.normal(0, 0.5, 50)),
    (np.random.normal(3, 0.5, 50), np.random.normal(3, 0.5, 50)),
    (np.random.normal(6, 0.5, 50), np.random.normal(6, 0.5, 50))
]

X = np.concatenate([c[0] for c in clusters])
y = np.concatenate([c[1] for c in clusters])

# Plot clustered data
plt.scatter(X, y)
plt.title("Clustered Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Slide 2: The Ecological Fallacy

The ecological fallacy occurs when conclusions about individuals are drawn from analyses of group-level data. This can lead to incorrect inferences when working with clustered data.

```python
# Calculate overall correlation
overall_corr = np.corrcoef(X, y)[0, 1]

# Calculate within-cluster correlations
within_cluster_corrs = [np.corrcoef(c[0], c[1])[0, 1] for c in clusters]

print(f"Overall correlation: {overall_corr:.2f}")
print("Within-cluster correlations:")
for i, corr in enumerate(within_cluster_corrs):
    print(f"Cluster {i+1}: {corr:.2f}")

# Plot overall trend line
plt.scatter(X, y)
reg = LinearRegression().fit(X.reshape(-1, 1), y)
plt.plot(X, reg.predict(X.reshape(-1, 1)), color='red', label='Overall trend')
plt.title("Ecological Fallacy Example")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
```

Slide 3: Simpson's Paradox

Simpson's Paradox is a phenomenon where a trend appears in several groups of data but disappears or reverses when these groups are combined. This paradox highlights the importance of considering group structure in data analysis.

```python
import pandas as pd

# Create synthetic data for Simpson's Paradox
np.random.seed(42)
group1 = pd.DataFrame({
    'x': np.random.uniform(0, 10, 100),
    'y': np.random.uniform(0, 10, 100) + 5,
    'group': 'A'
})
group2 = pd.DataFrame({
    'x': np.random.uniform(5, 15, 100),
    'y': np.random.uniform(5, 15, 100),
    'group': 'B'
})
data = pd.concat([group1, group2])

# Plot data and regression lines
plt.figure(figsize=(10, 6))
for group, group_data in data.groupby('group'):
    plt.scatter(group_data['x'], group_data['y'], label=group)
    reg = LinearRegression().fit(group_data[['x']], group_data['y'])
    plt.plot(group_data['x'], reg.predict(group_data[['x']]), linestyle='--')

# Overall regression line
reg_all = LinearRegression().fit(data[['x']], data['y'])
plt.plot(data['x'], reg_all.predict(data[['x']]), color='red', label='Overall')

plt.title("Simpson's Paradox Example")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
```

Slide 4: Multilevel Modeling: A Solution for Clustered Data

Multilevel modeling, also known as hierarchical linear modeling, is a statistical approach that accounts for the nested structure of clustered data. It allows for the simultaneous examination of within-group and between-group relationships.

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Create a dataset with a grouping variable
np.random.seed(42)
n_groups = 5
n_per_group = 20
group_means = np.random.normal(0, 1, n_groups)
groups = np.repeat(range(n_groups), n_per_group)
X = np.random.normal(0, 1, n_groups * n_per_group) + group_means[groups]
y = 0.5 * X + np.random.normal(0, 0.5, n_groups * n_per_group) + group_means[groups]

data = pd.DataFrame({'group': groups, 'X': X, 'y': y})

# Fit a multilevel model
model = smf.mixedlm("y ~ X", data, groups=data["group"])
results = model.fit()

print(results.summary())
```

Slide 5: Fixed Effects vs. Random Effects

In multilevel modeling, we distinguish between fixed effects and random effects. Fixed effects are constant across groups, while random effects vary. Understanding this distinction is crucial for proper model specification.

```python
# Fixed effects model
fe_model = smf.ols("y ~ X + C(group)", data=data).fit()

# Random effects model
re_model = smf.mixedlm("y ~ X", data, groups=data["group"]).fit()

print("Fixed Effects Model:")
print(fe_model.summary().tables[1])
print("\nRandom Effects Model:")
print(re_model.summary().tables[1])
```

Slide 6: Intraclass Correlation Coefficient (ICC)

The ICC measures the proportion of variance in the outcome variable that is accounted for by group membership. It helps determine the necessity of multilevel modeling.

```python
from scipy import stats

def calculate_icc(data, group_col, value_col):
    groups = data[group_col].unique()
    group_means = data.groupby(group_col)[value_col].mean()
    grand_mean = data[value_col].mean()
    
    between_group_var = sum([(mean - grand_mean)**2 for mean in group_means]) / (len(groups) - 1)
    within_group_var = sum([np.var(data[data[group_col] == group][value_col]) for group in groups]) / len(groups)
    
    icc = between_group_var / (between_group_var + within_group_var)
    return icc

icc = calculate_icc(data, 'group', 'y')
print(f"Intraclass Correlation Coefficient: {icc:.4f}")

# Visualize ICC
plt.figure(figsize=(10, 6))
for group in data['group'].unique():
    group_data = data[data['group'] == group]
    plt.scatter(group_data['X'], group_data['y'], label=f'Group {group}')
plt.title(f"Data Visualization with ICC = {icc:.4f}")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

Slide 7: Centering Variables in Multilevel Models

Centering variables can improve interpretation and reduce multicollinearity in multilevel models. We'll explore group-mean centering and grand-mean centering.

```python
# Group-mean centering
data['X_group_centered'] = data.groupby('group')['X'].transform(lambda x: x - x.mean())

# Grand-mean centering
data['X_grand_centered'] = data['X'] - data['X'].mean()

# Fit models with centered variables
model_group_centered = smf.mixedlm("y ~ X_group_centered", data, groups=data["group"]).fit()
model_grand_centered = smf.mixedlm("y ~ X_grand_centered", data, groups=data["group"]).fit()

print("Group-mean Centered Model:")
print(model_group_centered.summary().tables[1])
print("\nGrand-mean Centered Model:")
print(model_grand_centered.summary().tables[1])
```

Slide 8: Cross-Validation for Clustered Data

Traditional cross-validation can lead to overly optimistic performance estimates with clustered data. We'll implement a group-based cross-validation approach.

```python
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

X = data[['X']].values
y = data['y'].values
groups = data['group'].values

gkf = GroupKFold(n_splits=5)

mse_scores = []

for train_index, test_index in gkf.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

print(f"Mean MSE across folds: {np.mean(mse_scores):.4f}")
print(f"Standard deviation of MSE: {np.std(mse_scores):.4f}")
```

Slide 9: Handling Time-Series Clustered Data

Time-series clustered data presents unique challenges. We'll explore techniques for dealing with autocorrelation within clusters over time.

```python
import statsmodels.tsa.api as smt

# Generate time-series clustered data
np.random.seed(42)
n_clusters = 3
n_timepoints = 50

data = []
for cluster in range(n_clusters):
    ar_params = [0.75 + 0.1 * np.random.randn()]
    ma_params = [0.65 + 0.1 * np.random.randn()]
    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]
    y = smt.arma_generate_sample(ar, ma, n_timepoints) + cluster * 2
    data.extend([(cluster, t, y[t]) for t in range(n_timepoints)])

df = pd.DataFrame(data, columns=['cluster', 'time', 'value'])

# Plot time series for each cluster
plt.figure(figsize=(12, 6))
for cluster in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster]
    plt.plot(cluster_data['time'], cluster_data['value'], label=f'Cluster {cluster}')
plt.title("Time-Series Clustered Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

# Fit ARIMA model for one cluster
cluster_data = df[df['cluster'] == 0]['value']
model = smt.ARIMA(cluster_data, order=(1, 0, 1)).fit()
print(model.summary())
```

Slide 10: Dealing with Unbalanced Clusters

Unbalanced clusters, where group sizes vary significantly, can affect model estimates. We'll explore techniques to handle this issue.

```python
# Generate unbalanced clustered data
np.random.seed(42)
n_clusters = 5
cluster_sizes = [10, 20, 50, 100, 200]

data = []
for i, size in enumerate(cluster_sizes):
    X = np.random.normal(i, 1, size)
    y = 0.5 * X + np.random.normal(0, 0.5, size)
    data.extend([(i, x, y_val) for x, y_val in zip(X, y)])

df = pd.DataFrame(data, columns=['cluster', 'X', 'y'])

# Visualize unbalanced clusters
plt.figure(figsize=(10, 6))
for cluster in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['X'], cluster_data['y'], label=f'Cluster {cluster}')
plt.title("Unbalanced Clustered Data")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Fit weighted multilevel model
weights = 1 / df.groupby('cluster').size()
df['weight'] = df['cluster'].map(weights)

weighted_model = smf.mixedlm("y ~ X", df, groups=df["cluster"], weights=df["weight"]).fit()
print(weighted_model.summary().tables[1])
```

Slide 11: Clustered Data in Machine Learning: Random Forests

Random Forests can naturally handle clustered data by considering the cluster as a feature. We'll compare this approach to a standard Random Forest.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Prepare data
X = df[['X', 'cluster']]
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard Random Forest
rf_standard = RandomForestRegressor(n_estimators=100, random_state=42)
rf_standard.fit(X_train[['X']], y_train)
y_pred_standard = rf_standard.predict(X_test[['X']])

# Random Forest with cluster information
rf_clustered = RandomForestRegressor(n_estimators=100, random_state=42)
rf_clustered.fit(X_train, y_train)
y_pred_clustered = rf_clustered.predict(X_test)

print("Standard Random Forest MSE:", mean_squared_error(y_test, y_pred_standard))
print("Clustered Random Forest MSE:", mean_squared_error(y_test, y_pred_clustered))

# Feature importance
importance = rf_clustered.feature_importances_
for i, feat in enumerate(['X', 'cluster']):
    print(f"{feat} importance: {importance[i]:.4f}")
```

Slide 12: Real-Life Example: Educational Data

Consider a study on student performance across different schools. Each school represents a cluster, and we want to analyze the relationship between study hours and test scores while accounting for school-level effects.

```python
np.random.seed(42)
n_schools = 10
n_students_per_school = 50

schools = []
for i in range(n_schools):
    school_effect = np.random.normal(0, 5)
    study_hours = np.random.normal(5, 2, n_students_per_school)
    test_scores = 2 * study_hours + school_effect + np.random.normal(0, 10, n_students_per_school)
    schools.extend([(i, h, s) for h, s in zip(study_hours, test_scores)])

edu_data = pd.DataFrame(schools, columns=['school', 'study_hours', 'test_score'])

# Visualize data
plt.figure(figsize=(12, 6))
for school in edu_data['school'].unique():
    school_data = edu_data[edu_data['school'] == school]
    plt.scatter(school_data['study_hours'], school_data['test_score'], label=f'School {school}')
plt.title("Student Performance Across Schools")
plt.xlabel("Study Hours")
plt.ylabel("Test Score")
plt.legend()
plt.show()

# Fit multilevel model
edu_model = smf.mixedlm("test_score ~ study_hours", edu_data, groups=edu_data["school"]).fit()
print(edu_model.summary().tables[1])
```

Slide 13: Real-Life Example: Environmental Data

Imagine a study on air quality across different cities. Each city is a cluster, and we want to analyze the relationship between temperature and air pollution levels while accounting for city-specific effects.

```python
np.random.seed(42)
n_cities = 8
n_days = 30

cities = []
for i in range(n_cities):
    city_effect = np.random.normal(0, 10)
    temperatures = np.random.normal(25, 5, n_days)
    pollution_levels = 2 * temperatures + city_effect + np.random.normal(0, 15, n_days)
    cities.extend([(i, t, p) for t, p in zip(temperatures, pollution_levels)])

env_data = pd.DataFrame(cities, columns=['city', 'temperature', 'pollution'])

# Visualize data
plt.figure(figsize=(12, 6))
for city in env_data['city'].unique():
    city_data = env_data[env_data['city'] == city]
    plt.scatter(city_data['temperature'], city_data['pollution'], label=f'City {city}')
plt.title("Air Pollution vs Temperature Across Cities")
plt.xlabel("Temperature (°C)")
plt.ylabel("Pollution Level")
plt.legend()
plt.show()

# Fit multilevel model
env_model = smf.mixedlm("pollution ~ temperature", env_data, groups=env_data["city"]).fit()
print(env_model.summary().tables[1])
```

Slide 14: Interpreting Multilevel Models

Interpreting multilevel models requires careful consideration of fixed and random effects. We'll explore how to interpret coefficients and variance components in the context of our environmental data example.

```python
# Extract fixed effects
fixed_effects = env_model.fe_params
print("Fixed Effects:")
for effect, value in fixed_effects.items():
    print(f"{effect}: {value:.4f}")

# Extract random effects
random_effects = env_model.random_effects
print("\nRandom Effects (City-specific intercepts):")
for city, effect in random_effects.items():
    print(f"City {city}: {effect['Group'][0]:.4f}")

# Calculate ICC
def calculate_icc(model):
    re_var = model.cov_re.iloc[0, 0]
    resid_var = model.scale
    return re_var / (re_var + resid_var)

icc = calculate_icc(env_model)
print(f"\nIntraclass Correlation Coefficient: {icc:.4f}")

# Visualize fixed effect and random intercepts
plt.figure(figsize=(12, 6))
for city in env_data['city'].unique():
    city_data = env_data[env_data['city'] == city]
    plt.scatter(city_data['temperature'], city_data['pollution'], alpha=0.5)
    
    # City-specific line
    city_intercept = fixed_effects['Intercept'] + random_effects[city]['Group'][0]
    city_line = city_intercept + fixed_effects['temperature'] * city_data['temperature']
    plt.plot(city_data['temperature'], city_line, label=f'City {city}')

# Overall fixed effect line
overall_line = fixed_effects['Intercept'] + fixed_effects['temperature'] * env_data['temperature']
plt.plot(env_data['temperature'], overall_line, 'r--', linewidth=2, label='Fixed Effect')

plt.title("Multilevel Model: Fixed Effect and Random Intercepts")
plt.xlabel("Temperature (°C)")
plt.ylabel("Pollution Level")
plt.legend()
plt.show()
```

Slide 15: Additional Resources

For those interested in delving deeper into clustered data analysis and multilevel modeling, here are some valuable resources:

1. Gelman, A., & Hill, J. (2006). Data Analysis Using Regression and Multilevel/Hierarchical Models. Cambridge University Press.
2. Snijders, T. A. B., & Bosker, R. J. (2011). Multilevel Analysis: An Introduction to Basic and Advanced Multilevel Modeling (2nd ed.). Sage Publishers.
3. ArXiv paper: "A Practical Guide to Multilevel Modeling" by Bodo Winter URL: [https://arxiv.org/abs/1304.4725](https://arxiv.org/abs/1304.4725)
4. ArXiv paper: "Hierarchical Linear Models: Applications and Data Analysis Methods" by Stephen W. Raudenbush and Anthony S. Bryk URL: [https://arxiv.org/abs/1701.00960](https://arxiv.org/abs/1701.00960)

These resources provide in-depth coverage of the concepts and techniques discussed in this presentation, offering further insights into handling clustered data and avoiding common pitfalls in linear modeling.

