## Correlation Measures and Predictive Power Score in Python
Slide 1: Introduction to Correlation Measures

Introduction to Correlation Measures

Correlation measures quantify the statistical relationship between two variables. They help us understand how changes in one variable relate to changes in another. In this presentation, we'll explore various correlation measures and their implementation in Python.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
x = np.random.rand(100)
y = 0.5 * x + 0.5 * np.random.rand(100)

# Visualize the relationship
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sample Data for Correlation Analysis')
plt.show()
```

Slide 2: Pearson Correlation Coefficient

Pearson Correlation Coefficient

The Pearson correlation coefficient measures the linear relationship between two continuous variables. It ranges from -1 to 1, where -1 indicates a perfect negative linear relationship, 0 indicates no linear relationship, and 1 indicates a perfect positive linear relationship.

```python
import numpy as np
from scipy import stats

# Generate sample data
np.random.seed(42)
x = np.random.rand(100)
y = 0.7 * x + 0.3 * np.random.rand(100)

# Calculate Pearson correlation coefficient
pearson_corr, p_value = stats.pearsonr(x, y)

print(f"Pearson correlation coefficient: {pearson_corr:.4f}")
print(f"P-value: {p_value:.4f}")
```

Slide 3: Spearman Rank Correlation

Spearman Rank Correlation

The Spearman rank correlation assesses the monotonic relationship between two variables. It is less sensitive to outliers and can capture non-linear relationships. Like Pearson, it ranges from -1 to 1, with similar interpretations.

```python
import numpy as np
from scipy import stats

# Generate sample data
np.random.seed(42)
x = np.random.rand(100)
y = np.exp(x) + 0.1 * np.random.rand(100)

# Calculate Spearman rank correlation
spearman_corr, p_value = stats.spearmanr(x, y)

print(f"Spearman rank correlation: {spearman_corr:.4f}")
print(f"P-value: {p_value:.4f}")
```

Slide 4: Kendall's Tau

Kendall's Tau

Kendall's Tau is another rank-based correlation measure that assesses the ordinal association between two variables. It is particularly useful for small sample sizes and when there are many tied ranks in the data.

```python
import numpy as np
from scipy import stats

# Generate sample data
np.random.seed(42)
x = np.random.randint(1, 6, 100)  # Discrete data
y = x + np.random.randint(-1, 2, 100)  # Add some noise

# Calculate Kendall's Tau
kendall_tau, p_value = stats.kendalltau(x, y)

print(f"Kendall's Tau: {kendall_tau:.4f}")
print(f"P-value: {p_value:.4f}")
```

Slide 5: Correlation Matrix

Correlation Matrix

A correlation matrix provides a comprehensive view of pairwise correlations between multiple variables in a dataset. It's an essential tool for exploratory data analysis and feature selection in machine learning.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    'A': np.random.rand(100),
    'B': np.random.rand(100),
    'C': np.random.rand(100),
    'D': np.random.rand(100)
})

# Calculate correlation matrix
corr_matrix = data.corr()

# Visualize correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix')
plt.show()
```

Slide 6: Distance Correlation

Distance Correlation

Distance correlation is a measure of statistical dependence between two variables that can detect non-linear relationships. Unlike Pearson correlation, it can capture more complex dependencies in the data.

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

def distance_correlation(X, Y):
    def center_distance_matrix(D):
        n = D.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H.dot(D).dot(H)
    
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    
    X_dist = squareform(pdist(X))
    Y_dist = squareform(pdist(Y))
    
    X_cent = center_distance_matrix(X_dist)
    Y_cent = center_distance_matrix(Y_dist)
    
    num = np.sum(X_cent * Y_cent)
    den = np.sqrt(np.sum(X_cent**2) * np.sum(Y_cent**2))
    
    return np.sqrt(num / den)

# Example usage
X = np.random.rand(100, 1)
Y = np.sin(2 * np.pi * X) + 0.1 * np.random.randn(100, 1)

dc = distance_correlation(X, Y)
print(f"Distance correlation: {dc:.4f}")
```

Slide 7: Point-Biserial Correlation

Point-Biserial Correlation

Point-biserial correlation measures the relationship between a continuous variable and a binary variable. It's useful in scenarios where you want to assess the association between a numeric variable and a categorical variable with two levels.

```python
import numpy as np
from scipy import stats

# Generate sample data
np.random.seed(42)
continuous_var = np.random.normal(0, 1, 100)
binary_var = np.random.choice([0, 1], 100)

# Calculate point-biserial correlation
point_biserial_corr, p_value = stats.pointbiserialr(binary_var, continuous_var)

print(f"Point-biserial correlation: {point_biserial_corr:.4f}")
print(f"P-value: {p_value:.4f}")
```

Slide 8: Introduction to Predictive Power Score (PPS)

Introduction to Predictive Power Score (PPS)

The Predictive Power Score (PPS) is a novel measure of the predictive strength between two variables. Unlike traditional correlation measures, PPS can detect non-linear relationships and works with mixed data types. It ranges from 0 to 1, where 0 indicates no predictive power and 1 indicates perfect predictive power.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

def calculate_pps(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return r2

# Example usage
np.random.seed(42)
X = np.random.rand(1000, 1)
y = np.sin(2 * np.pi * X.ravel()) + 0.1 * np.random.randn(1000)

pps = calculate_pps(X, y)
print(f"Predictive Power Score: {pps:.4f}")
```

Slide 9: Calculating PPS for Multiple Features

Calculating PPS for Multiple Features

When working with datasets containing multiple features, it's useful to calculate the PPS for each feature with respect to the target variable. This can help identify the most predictive features and guide feature selection.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def calculate_pps_multiple(X, y):
    pps_scores = {}
    
    for column in X.columns:
        X_train, X_test, y_train, y_test = train_test_split(X[[column]], y, test_size=0.3, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        pps_scores[column] = r2_score(y_test, y_pred)
    
    return pps_scores

# Example usage
np.random.seed(42)
data = pd.DataFrame({
    'A': np.random.rand(1000),
    'B': np.random.rand(1000),
    'C': np.random.rand(1000),
    'D': np.random.rand(1000)
})
target = np.sin(2 * np.pi * data['A']) + 0.5 * data['B'] + 0.1 * np.random.randn(1000)

pps_scores = calculate_pps_multiple(data, target)
for feature, score in pps_scores.items():
    print(f"PPS for {feature}: {score:.4f}")
```

Slide 10: Visualizing PPS Scores

Visualizing PPS Scores

Visualizing PPS scores can provide insights into the relative predictive power of different features. A horizontal bar plot is an effective way to display these scores, allowing for easy comparison across features.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def calculate_pps_multiple(X, y):
    # (Same function as in the previous slide)
    # ...

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    'A': np.random.rand(1000),
    'B': np.random.rand(1000),
    'C': np.random.rand(1000),
    'D': np.random.rand(1000),
    'E': np.random.rand(1000)
})
target = np.sin(2 * np.pi * data['A']) + 0.5 * data['B'] + 0.3 * data['C'] + 0.1 * np.random.randn(1000)

# Calculate PPS scores
pps_scores = calculate_pps_multiple(data, target)

# Visualize PPS scores
plt.figure(figsize=(10, 6))
plt.barh(list(pps_scores.keys()), list(pps_scores.values()))
plt.xlabel('Predictive Power Score')
plt.ylabel('Features')
plt.title('Predictive Power Scores for Different Features')
plt.xlim(0, 1)
for i, v in enumerate(pps_scores.values()):
    plt.text(v, i, f' {v:.2f}', va='center')
plt.tight_layout()
plt.show()
```

Slide 11: Comparing Correlation and PPS

Comparing Correlation and PPS

It's insightful to compare traditional correlation measures with PPS. This comparison can reveal relationships that might be missed by linear correlation methods and provide a more comprehensive understanding of feature importance.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def calculate_pps_multiple(X, y):
    # (Same function as in the previous slides)
    # ...

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    'Linear': np.random.rand(1000),
    'Quadratic': np.random.rand(1000),
    'Sinusoidal': np.random.rand(1000),
    'Random': np.random.rand(1000)
})
data['Target'] = (
    data['Linear'] +
    (data['Quadratic'] - 0.5)**2 +
    np.sin(2 * np.pi * data['Sinusoidal']) +
    0.1 * np.random.randn(1000)
)

# Calculate correlations and PPS
correlations = data.corr()['Target'].drop('Target')
pps_scores = calculate_pps_multiple(data.drop('Target', axis=1), data['Target'])

# Visualize comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(correlations))
width = 0.35

ax.bar(x - width/2, correlations.abs(), width, label='Absolute Correlation')
ax.bar(x + width/2, pps_scores.values(), width, label='PPS')

ax.set_ylabel('Score')
ax.set_title('Comparison of Absolute Correlation and PPS')
ax.set_xticks(x)
ax.set_xticklabels(correlations.index)
ax.legend()

plt.tight_layout()
plt.show()
```

Slide 12: PPS for Classification Problems

PPS for Classification Problems

PPS can also be applied to classification problems. In this case, we use classification metrics like accuracy or F1-score instead of R-squared to measure predictive power. This allows us to assess feature importance for categorical target variables.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def calculate_pps_classification(X, y):
    pps_scores = {}
    
    for column in X.columns:
        X_train, X_test, y_train, y_test = train_test_split(X[[column]], y, test_size=0.3, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        pps_scores[column] = f1_score(y_test, y_pred, average='weighted')
    
    return pps_scores

# Example usage
np.random.seed(42)
data = pd.DataFrame({
    'A': np.random.rand(1000),
    'B': np.random.rand(1000),
    'C': np.random.rand(1000),
    'D': np.random.rand(1000)
})
target = (np.sin(2 * np.pi * data['A']) + 0.5 * data['B'] + 0.1 * np.random.randn(1000) > 0).astype(int)

pps_scores = calculate_pps_classification(data, target)
for feature, score in pps_scores.items():
    print(f"PPS for {feature}: {score:.4f}")
```

Slide 13: Handling Mixed Data Types with PPS

Handling Mixed Data Types with PPS

One of the advantages of PPS is its ability to handle mixed data types. This is particularly useful when dealing with datasets that contain both numerical and categorical variables. We'll demonstrate how to calculate PPS for a dataset with mixed types.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def calculate_pps_mixed(X, y):
    pps_scores = {}
    
    for column in X.columns:
        X_train, X_test, y_train, y_test = train_test_split(X[[column]], y, test_size=0.3, random_state=42)
        
        if X[column].dtype == 'object':
            preprocessor = OneHotEncoder(sparse=False, handle_unknown='ignore')
        else:
            preprocessor = StandardScaler()
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        pps_scores[column] = r2_score(y_test, y_pred)
    
    return pps_scores

# Example usage
np.random.seed(42)
data = pd.DataFrame({
    'Numeric1': np.random.rand(1000),
    'Numeric2': np.random.rand(1000),
    'Categorical1': np.random.choice(['A', 'B', 'C'], 1000),
    'Categorical2': np.random.choice(['X', 'Y', 'Z'], 1000)
})
target = 2 * data['Numeric1'] + np.sin(2 * np.pi * data['Numeric2']) + (data['Categorical1'] == 'A').astype(int) + 0.1 * np.random.randn(1000)

pps_scores = calculate_pps_mixed(data, target)
for feature, score in pps_scores.items():
    print(f"PPS for {feature}: {score:.4f}")
```

Slide 14: PPS vs. Feature Importance

PPS vs. Feature Importance

While PPS provides insights into the predictive power of individual features, it's interesting to compare it with traditional feature importance methods, such as those derived from tree-based models. This comparison can offer a more comprehensive view of feature relevance.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

def calculate_pps_multiple(X, y):
    # (Use the function from previous slides)
    pass

# Generate sample data
np.random.seed(42)
X = pd.DataFrame({
    'A': np.random.rand(1000),
    'B': np.random.rand(1000),
    'C': np.random.rand(1000),
    'D': np.random.rand(1000)
})
y = 2 * X['A'] + np.sin(2 * np.pi * X['B']) + 0.5 * X['C'] + 0.1 * np.random.randn(1000)

# Calculate PPS
pps_scores = calculate_pps_multiple(X, y)

# Calculate feature importance using Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_

# Calculate permutation importance
perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

# Visualize comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(X.columns))
width = 0.25

ax.bar(x - width, pps_scores.values(), width, label='PPS')
ax.bar(x, importances, width, label='Random Forest Importance')
ax.bar(x + width, perm_importance.importances_mean, width, label='Permutation Importance')

ax.set_ylabel('Importance Score')
ax.set_title('Comparison of Feature Importance Measures')
ax.set_xticks(x)
ax.set_xticklabels(X.columns)
ax.legend()

plt.tight_layout()
plt.show()
```

Slide 15: Limitations and Considerations

Limitations and Considerations

While correlation measures and PPS are powerful tools for understanding relationships between variables, they have limitations. It's important to consider these when interpreting results and making decisions based on these metrics.

```python
# Pseudocode for considerations when using correlation measures and PPS

def correlation_pps_considerations():
    considerations = [
        "Sample size affects reliability of correlation measures and PPS",
        "Correlation does not imply causation",
        "PPS may be computationally intensive for large datasets",
        "Both methods can be affected by outliers",
        "Multicollinearity can impact interpretation of results",
        "Non-linear relationships may not be fully captured by simple correlation measures",
        "PPS does not account for interaction effects between features",
        "Interpretability of PPS can be challenging compared to traditional correlation"
    ]
    
    for i, consideration in enumerate(considerations, 1):
        print(f"{i}. {consideration}")

# Call the function to display considerations
correlation_pps_considerations()
```

Slide 16: Additional Resources

Additional Resources

For those interested in delving deeper into correlation measures and Predictive Power Score, here are some valuable resources:

1. "Distance Correlation: A New Tool for Detecting Association and Measuring Correlation Between Multivariate Samples" by Gábor J. Székely and Maria L. Rizzo ArXiv URL: [https://arxiv.org/abs/1401.7645](https://arxiv.org/abs/1401.7645)
2. "Measuring Dependence with Matrix-based Entropy Functional" by Satoshi Kuriki and Yoichiro Miyata ArXiv URL: [https://arxiv.org/abs/1605.08293](https://arxiv.org/abs/1605.08293)
3. "Feature Selection with the Predictive Power Score" by Florian Wetschoreck, Tobias Krabel, and Søren Welling ArXiv URL: [https://arxiv.org/abs/2011.01455](https://arxiv.org/abs/2011.01455)

These papers provide in-depth discussions on advanced correlation measures and the theoretical foundations of PPS. They can serve as excellent starting points for further exploration of these topics.

