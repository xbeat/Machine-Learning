## Variance Inflation Factor (VIF) and Regression Diagnostics in Python
Slide 1: Understanding Variance Inflation Factor (VIF) and Regression Diagnostics

VIF and regression diagnostics are essential tools in statistical analysis, helping to identify and address multicollinearity and other issues in linear regression models. This presentation will explore these concepts using Python, providing practical examples and code snippets to illustrate their application and interpretation.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Sample data
np.random.seed(42)
X = np.random.randn(100, 3)
y = 2 * X[:, 0] + 3 * X[:, 1] + 4 * X[:, 2] + np.random.randn(100)

# Create a DataFrame
df = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
df['y'] = y

# Fit the model
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()

print(model.summary())
```

Slide 2: What is Variance Inflation Factor (VIF)?

VIF is a measure of multicollinearity in regression analysis. It quantifies the extent of correlation between one predictor and the other predictors in a model. VIF provides an index that measures how much the variance of an estimated regression coefficient is increased because of collinearity.

```python
def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

X_with_const = sm.add_constant(df[['X1', 'X2', 'X3']])
vif_data = calculate_vif(X_with_const)
print(vif_data)
```

Slide 3: Interpreting VIF Values

VIF values indicate the severity of multicollinearity. A general rule of thumb is:

* VIF = 1: Variables are not correlated
* 1 < VIF < 5: Moderate correlation
* 5 ≤ VIF < 10: High correlation
* VIF ≥ 10: Severe multicollinearity

```python
def interpret_vif(vif_value):
    if vif_value == 1:
        return "No correlation"
    elif 1 < vif_value < 5:
        return "Moderate correlation"
    elif 5 <= vif_value < 10:
        return "High correlation"
    else:
        return "Severe multicollinearity"

vif_data['Interpretation'] = vif_data['VIF'].apply(interpret_vif)
print(vif_data)
```

Slide 4: Dealing with High VIF Values

When high VIF values are detected, consider:

1. Removing highly correlated predictors
2. Combining predictors
3. Using regularization techniques (e.g., Ridge or Lasso regression)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['X1', 'X2', 'X3']])

# Fit Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)

print("Ridge coefficients:")
for name, coef in zip(['X1', 'X2', 'X3'], ridge.coef_):
    print(f"{name}: {coef:.4f}")
```

Slide 5: Regression Diagnostics: Residual Analysis

Residual analysis is crucial for validating the assumptions of linear regression. Key aspects include:

* Linearity
* Homoscedasticity
* Normality of residuals
* Independence of residuals

```python
import matplotlib.pyplot as plt
import seaborn as sns

residuals = model.resid
fitted_values = model.fittedvalues

plt.figure(figsize=(10, 6))
sns.scatterplot(x=fitted_values, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()
```

Slide 6: Checking Linearity and Homoscedasticity

The residual plot helps assess linearity and homoscedasticity. Ideally, residuals should be randomly scattered around zero with no clear pattern.

```python
import scipy.stats as stats

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=ax1)
ax1.set_title("Q-Q plot")

# Residuals vs Fitted
sns.scatterplot(x=fitted_values, y=residuals, ax=ax2)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Fitted values')
ax2.set_ylabel('Residuals')
ax2.set_title('Residuals vs Fitted Values')

plt.tight_layout()
plt.show()
```

Slide 7: Detecting Influential Observations

Influential observations can significantly impact the regression model. Common measures include:

* Cook's Distance
* DFBETAS
* Leverage

```python
from statsmodels.stats.outliers_influence import OLSInfluence

influence = OLSInfluence(model)
cooks_d = influence.cooks_distance[0]

plt.figure(figsize=(10, 6))
plt.stem(range(len(cooks_d)), cooks_d, markerfmt=",")
plt.xlabel('Observation')
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance Plot")
plt.show()

print("Top 5 influential observations:")
print(pd.Series(cooks_d).nlargest(5))
```

Slide 8: Handling Influential Observations

When dealing with influential observations:

1. Investigate the cause of their influence
2. Consider removing or transforming them if they're outliers
3. Use robust regression techniques

```python
from sklearn.linear_model import HuberRegressor

# Fit Huber Regression
huber = HuberRegressor()
huber.fit(X, y)

print("Huber Regression coefficients:")
for name, coef in zip(['X1', 'X2', 'X3'], huber.coef_):
    print(f"{name}: {coef:.4f}")

# Compare with OLS
print("\nOLS coefficients:")
print(model.params[1:])  # Excluding the constant term
```

Slide 9: Multicollinearity: A Real-Life Example

Consider a house price prediction model with features: 'square\_feet', 'num\_rooms', and 'total\_area'. 'total\_area' is likely highly correlated with both 'square\_feet' and 'num\_rooms', potentially causing multicollinearity.

```python
np.random.seed(42)
square_feet = np.random.randint(1000, 3000, 100)
num_rooms = np.random.randint(2, 6, 100)
total_area = square_feet + np.random.normal(0, 100, 100)
price = 100000 + 100 * square_feet + 20000 * num_rooms + np.random.normal(0, 10000, 100)

house_df = pd.DataFrame({
    'square_feet': square_feet,
    'num_rooms': num_rooms,
    'total_area': total_area,
    'price': price
})

X_house = sm.add_constant(house_df[['square_feet', 'num_rooms', 'total_area']])
vif_house = calculate_vif(X_house)
print(vif_house)
```

Slide 10: Addressing Multicollinearity in the House Price Model

To address multicollinearity, we can remove the 'total\_area' feature or combine it with other features to create a new, less correlated variable.

```python
# Remove 'total_area'
X_house_reduced = sm.add_constant(house_df[['square_feet', 'num_rooms']])
vif_house_reduced = calculate_vif(X_house_reduced)
print("VIF after removing 'total_area':")
print(vif_house_reduced)

# Create a new feature: average area per room
house_df['avg_area_per_room'] = house_df['total_area'] / house_df['num_rooms']
X_house_new = sm.add_constant(house_df[['square_feet', 'num_rooms', 'avg_area_per_room']])
vif_house_new = calculate_vif(X_house_new)
print("\nVIF with new feature 'avg_area_per_room':")
print(vif_house_new)
```

Slide 11: Heteroscedasticity: A Real-Life Example

In a study of income vs. education level, the variance of income often increases with education level, violating the homoscedasticity assumption.

```python
np.random.seed(42)
education_years = np.random.randint(8, 22, 100)
income = 10000 + 5000 * education_years + np.random.normal(0, education_years * 1000, 100)

income_df = pd.DataFrame({
    'education_years': education_years,
    'income': income
})

X_income = sm.add_constant(income_df['education_years'])
income_model = sm.OLS(income_df['income'], X_income).fit()

plt.figure(figsize=(10, 6))
plt.scatter(income_model.fittedvalues, income_model.resid)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values (Income Model)')
plt.show()
```

Slide 12: Addressing Heteroscedasticity in the Income Model

To address heteroscedasticity, we can use robust standard errors or transform the dependent variable.

```python
# Using robust standard errors
robust_income_model = sm.OLS(income_df['income'], X_income).fit(cov_type='HC3')
print(robust_income_model.summary())

# Log transformation of the dependent variable
income_df['log_income'] = np.log(income_df['income'])
log_income_model = sm.OLS(income_df['log_income'], X_income).fit()

plt.figure(figsize=(10, 6))
plt.scatter(log_income_model.fittedvalues, log_income_model.resid)
plt.xlabel('Fitted values (log scale)')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values (Log Income Model)')
plt.show()
```

Slide 13: Conclusion and Best Practices

1. Always check for multicollinearity using VIF
2. Perform thorough residual analysis
3. Be cautious of influential observations
4. Address violations of regression assumptions
5. Use robust methods when necessary
6. Interpret results in the context of your domain knowledge

```python
def regression_diagnostics(model, X, y):
    # VIF
    vif = calculate_vif(X)
    
    # Residual analysis
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    # Influential observations
    influence = OLSInfluence(model)
    cooks_d = influence.cooks_distance[0]
    
    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Residuals vs Fitted
    axes[0, 0].scatter(fitted_values, residuals)
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q plot")
    
    # Cook's distance
    axes[1, 0].stem(range(len(cooks_d)), cooks_d, markerfmt=",")
    axes[1, 0].set_xlabel('Observation')
    axes[1, 0].set_ylabel("Cook's Distance")
    axes[1, 0].set_title("Cook's Distance Plot")
    
    # VIF barplot
    axes[1, 1].bar(vif['Variable'], vif['VIF'])
    axes[1, 1].set_xlabel('Variables')
    axes[1, 1].set_ylabel('VIF')
    axes[1, 1].set_title('Variance Inflation Factors')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return vif, cooks_d

# Example usage
diagnostics_vif, diagnostics_cooks_d = regression_diagnostics(model, X_with_const, y)
```

Slide 14: Additional Resources

For further exploration of VIF and regression diagnostics:

1. "Regression Diagnostics: Identifying Influential Data and Sources of Collinearity" by Belsley, Kuh, and Welsch (1980)
2. "Applied Linear Regression Models" by Kutner, Nachtsheim, and Neter (2004)
3. ArXiv paper: "A Survey of Regression Diagnostics" by Wang and Wen (2020) URL: [https://arxiv.org/abs/2009.02638](https://arxiv.org/abs/2009.02638)

