## Poisson Regression vs. Linear Regression in Python
Slide 1: Introduction to Regression Analysis

Regression analysis is a statistical method used to model the relationship between variables. We'll explore two popular types: Linear Regression and Poisson Regression, highlighting their differences and use cases.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data
x = np.linspace(0, 10, 100)
y_linear = 2 * x + 1 + np.random.normal(0, 1, 100)
y_poisson = np.random.poisson(x)

# Plot the data
plt.figure(figsize=(12, 6))
plt.scatter(x, y_linear, label='Linear Data')
plt.scatter(x, y_poisson, label='Poisson Data')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear vs Poisson Data')
plt.show()
```

Slide 2: Linear Regression Basics

Linear regression models the relationship between a dependent variable and one or more independent variables using a linear equation. It assumes a constant change in the dependent variable for a unit change in the independent variable.

```python
from sklearn.linear_model import LinearRegression

# Prepare data
X = x.reshape(-1, 1)
y = y_linear

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Print results
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.2f}")

# Plot the regression line
plt.scatter(x, y)
plt.plot(x, model.predict(X), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.show()
```

Slide 3: Poisson Regression Basics

Poisson regression is used when modeling count data or rates. It assumes the dependent variable follows a Poisson distribution and that its logarithm can be modeled by a linear combination of unknown parameters.

```python
from sklearn.linear_model import PoissonRegressor

# Prepare data
X = x.reshape(-1, 1)
y = y_poisson

# Create and fit the model
model = PoissonRegressor()
model.fit(X, y)

# Print results
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")

# Plot the regression curve
plt.scatter(x, y)
plt.plot(x, model.predict(X), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Poisson Regression')
plt.show()
```

Slide 4: Assumptions of Linear Regression

Linear regression makes several key assumptions: linearity, independence, homoscedasticity, and normality of residuals. Let's visualize these assumptions using a simple example.

```python
import seaborn as sns

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.ravel() + 1 + np.random.normal(0, 1, 100)

# Fit linear regression model
model = LinearRegression().fit(X, y)

# Predict and calculate residuals
y_pred = model.predict(X)
residuals = y - y_pred

# Plot assumptions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Linearity
axes[0, 0].scatter(X, y)
axes[0, 0].plot(X, y_pred, color='red')
axes[0, 0].set_title('Linearity')

# Independence (Residuals vs. Fitted)
axes[0, 1].scatter(y_pred, residuals)
axes[0, 1].axhline(y=0, color='red', linestyle='--')
axes[0, 1].set_title('Independence')

# Homoscedasticity
axes[1, 0].scatter(y_pred, np.abs(residuals))
axes[1, 0].set_title('Homoscedasticity')

# Normality of Residuals
sns.histplot(residuals, kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Normality of Residuals')

plt.tight_layout()
plt.show()
```

Slide 5: Assumptions of Poisson Regression

Poisson regression assumes that the dependent variable follows a Poisson distribution, the mean equals the variance (equidispersion), and the logarithm of the mean can be modeled by a linear combination of parameters.

```python
import statsmodels.api as sm

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 5, 100).reshape(-1, 1)
y = np.random.poisson(np.exp(1 + 0.5 * X.ravel()))

# Fit Poisson regression model
model = sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson())
results = model.fit()

# Predict and calculate residuals
y_pred = results.predict(sm.add_constant(X))
residuals = y - y_pred

# Plot assumptions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Mean-variance relationship
axes[0, 0].scatter(y_pred, residuals**2)
axes[0, 0].set_title('Mean-Variance Relationship')
axes[0, 0].set_xlabel('Predicted Mean')
axes[0, 0].set_ylabel('Squared Residuals')

# Linearity of log(mean)
axes[0, 1].scatter(X, np.log(y))
axes[0, 1].plot(X, results.params[0] + results.params[1] * X, color='red')
axes[0, 1].set_title('Linearity of log(mean)')

# Independence (Residuals vs. Fitted)
axes[1, 0].scatter(y_pred, residuals)
axes[1, 0].axhline(y=0, color='red', linestyle='--')
axes[1, 0].set_title('Independence')

# Distribution of response variable
sns.histplot(y, kde=False, ax=axes[1, 1])
axes[1, 1].set_title('Distribution of Response Variable')

plt.tight_layout()
plt.show()
```

Slide 6: Key Differences Between Linear and Poisson Regression

Linear regression is used for continuous outcomes, while Poisson regression is used for count data. Linear regression assumes a normal distribution of residuals, while Poisson regression assumes a Poisson distribution of the response variable.

```python
import pandas as pd

# Create a comparison dataframe
comparison = pd.DataFrame({
    'Aspect': ['Response Variable', 'Distribution', 'Link Function', 'Variance', 'Use Case'],
    'Linear Regression': ['Continuous', 'Normal', 'Identity', 'Constant', 'Predicting continuous outcomes'],
    'Poisson Regression': ['Count', 'Poisson', 'Log', 'Mean = Variance', 'Modeling count data or rates']
})

# Display the comparison table
print(comparison.to_string(index=False))

# Visualize the difference in distributions
x = np.arange(0, 20, 1)
normal = stats.norm.pdf(x, 10, 2)
poisson = stats.poisson.pmf(x, 10)

plt.figure(figsize=(10, 6))
plt.plot(x, normal, label='Normal (Linear)')
plt.plot(x, poisson, label='Poisson')
plt.legend()
plt.title('Normal vs Poisson Distribution')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.show()
```

Slide 7: Real-Life Example: Modeling Customer Service Calls

Suppose we want to model the number of customer service calls a company receives based on the day of the week. This is a perfect scenario for Poisson regression, as we're dealing with count data.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(42)
days = np.repeat(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 50)
calls = np.random.poisson([30, 25, 25, 20, 30, 40, 35], 350)

data = pd.DataFrame({'day': days, 'calls': calls})
data['day'] = pd.Categorical(data['day'], categories=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], ordered=True)

# Prepare data for modeling
X = pd.get_dummies(data['day'], drop_first=True)
y = data['calls']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Poisson regression model
model = PoissonRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, label='Actual')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Customer Service Calls')
plt.xlabel('Sample')
plt.ylabel('Number of Calls')
plt.show()

# Print model coefficients
coef_df = pd.DataFrame({'day': X.columns, 'coefficient': model.coef_})
print(coef_df)
```

Slide 8: Real-Life Example: Predicting Plant Growth

Let's use linear regression to model the growth of a plant based on the amount of sunlight it receives. This scenario is suitable for linear regression as we expect a linear relationship between sunlight and growth.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(42)
sunlight_hours = np.random.uniform(2, 12, 100)
growth_cm = 5 + 0.5 * sunlight_hours + np.random.normal(0, 1, 100)

data = pd.DataFrame({'sunlight_hours': sunlight_hours, 'growth_cm': growth_cm})

# Split data
X = data[['sunlight_hours']]
y = data['growth_cm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.legend()
plt.title('Plant Growth vs Sunlight Hours')
plt.xlabel('Sunlight Hours')
plt.ylabel('Growth (cm)')
plt.show()

# Print model coefficients
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")
```

Slide 9: Model Evaluation: Linear Regression

For linear regression, we commonly use metrics like R-squared, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to evaluate model performance.

```python
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R-squared: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.legend()
plt.title('Linear Regression: Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

Slide 10: Model Evaluation: Poisson Regression

For Poisson regression, we often use metrics like Deviance, AIC (Akaike Information Criterion), and Mean Absolute Error (MAE) to assess model performance.

```python
from sklearn.metrics import mean_absolute_error
from scipy.stats import poisson

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 5
y = np.random.poisson(np.exp(1 + 0.5 * X.ravel()))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = PoissonRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
deviance = 2 * sum(y_test * np.log(y_test / y_pred) - (y_test - y_pred))
aic = 2 * (len(model.coef_) + 1) - 2 * sum(poisson.logpmf(y_test, y_pred))

print(f"MAE: {mae:.4f}")
print(f"Deviance: {deviance:.4f}")
print(f"AIC: {aic:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.legend()
plt.title('Poisson Regression: Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

Slide 11: Handling Overdispersion in Poisson Regression

Overdispersion occurs when the variance of the data is larger than the mean, violating the Poisson assumption. In such cases, we might consider using Negative Binomial Regression instead.

```python
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

# Generate overdispersed data
np.random.seed(42)
X = np.random.rand(100, 1) * 5
lambda_ = np.exp(1 + 0.5 * X.ravel())
y = np.random.negative_binomial(n=5, p=5/(5+lambda_), size=100)

# Fit Poisson and Negative Binomial models
poisson_model = sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson())
nb_model = NegativeBinomial(y, sm.add_constant(X))

poisson_results = poisson_model.fit()
nb_results = nb_model.fit()

# Compare AICs
print(f"Poisson AIC: {poisson_results.aic:.2f}")
print(f"Negative Binomial AIC: {nb_results.aic:.2f}")

# Plot fitted values
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Observed')
plt.plot(X, poisson_results.predict(sm.add_constant(X)), 'r-', label='Poisson')
plt.plot(X, nb_results.predict(sm.add_constant(X)), 'g-', label='Negative Binomial')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Comparison of Poisson and Negative Binomial Regression')
plt.show()
```

Slide 12: Regularization in Linear and Poisson Regression

Regularization helps prevent overfitting by adding a penalty term to the loss function. Common methods include Ridge (L2) and Lasso (L1) regularization.

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import PoissonRegressor

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 5)
y_linear = 1 + 2*X[:, 0] + 3*X[:, 1] + np.random.normal(0, 0.1, 100)
y_poisson = np.random.poisson(np.exp(1 + 2*X[:, 0] + 3*X[:, 1]))

# Fit models
ridge = Ridge(alpha=1.0).fit(X, y_linear)
lasso = Lasso(alpha=1.0).fit(X, y_linear)
poisson_reg = PoissonRegressor(alpha=1.0).fit(X, y_poisson)

# Print coefficients
print("Ridge coefficients:", ridge.coef_)
print("Lasso coefficients:", lasso.coef_)
print("Regularized Poisson coefficients:", poisson_reg.coef_)

# Plot coefficient values
plt.figure(figsize=(10, 6))
plt.plot(ridge.coef_, 'bo-', label='Ridge')
plt.plot(lasso.coef_, 'ro-', label='Lasso')
plt.plot(poisson_reg.coef_, 'go-', label='Poisson')
plt.legend()
plt.title('Comparison of Regularized Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.show()
```

Slide 13: Dealing with Non-linearity

When relationships are non-linear, we can use polynomial features or apply transformations to the predictors.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate non-linear data
X = np.linspace(0, 5, 100).reshape(-1, 1)
y = 0.5 * X**2 + X + 2 + np.random.normal(0, 1, (100, 1))

# Create polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Fit models
linear_model = LinearRegression().fit(X, y)
poly_model = LinearRegression().fit(X_poly, y)

# Predictions
X_test = np.linspace(0, 5, 100).reshape(-1, 1)
y_linear_pred = linear_model.predict(X_test)
y_poly_pred = poly_model.predict(poly_features.transform(X_test))

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data')
plt.plot(X_test, y_linear_pred, label='Linear', color='red')
plt.plot(X_test, y_poly_pred, label='Polynomial', color='green')
plt.legend()
plt.title('Linear vs Polynomial Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

Slide 14: Diagnostics and Model Checking

Proper model diagnostics are crucial for ensuring the validity of our regression models. Key aspects include residual analysis, influence diagnostics, and multicollinearity checks.

```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 3)
y = 1 + 2*X[:, 0] + 3*X[:, 1] + np.random.normal(0, 0.1, 100)

# Fit OLS model
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Residual plot
residuals = model.resid
fitted_values = model.fittedvalues

plt.figure(figsize=(10, 6))
plt.scatter(fitted_values, residuals)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# QQ plot
from statsmodels.graphics.gofplots import qqplot
qqplot(residuals, line='s')
plt.title('Q-Q Plot')
plt.show()

# VIF for multicollinearity
vif = pd.DataFrame()
vif["features"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)
```

Slide 15: Additional Resources

For further exploration of regression techniques and their implementation in Python, consider the following resources:

1. "Generalized Linear Models with Examples in R" by Dunn and Smyth (2018) ArXiv: [https://arxiv.org/abs/1906.02327](https://arxiv.org/abs/1906.02327)
2. "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani (2013) ArXiv: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
3. "Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman (2009) ArXiv: [https://arxiv.org/abs/2001.00323](https://arxiv.org/abs/2001.00323)

These resources provide in-depth explanations of various regression techniques, their assumptions, and practical applications using Python and R.

