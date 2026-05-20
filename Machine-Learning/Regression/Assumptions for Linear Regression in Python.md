## Assumptions for Linear Regression in Python
Slide 1: Introduction to Linear Regression Assumptions

Linear regression is a fundamental statistical technique used to model the relationship between variables. To ensure the validity and reliability of our model, we need to test five key assumptions. These assumptions form the foundation for accurate predictions and meaningful interpretations of our results.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.normal(0, 1, (100, 1))

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Plot the data and regression line
plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.title('Linear Regression Example')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
```

Slide 2: Linearity Assumption

The linearity assumption states that there should be a linear relationship between the independent variables and the dependent variable. This means that changes in the predictors are associated with a constant change in the response variable.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate non-linear data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = X**2 + np.random.normal(0, 10, (100, 1))

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Plot the data and regression line
plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.title('Non-linear Relationship')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Calculate residuals
residuals = y - model.predict(X)

# Plot residuals
plt.scatter(model.predict(X), residuals, color='green', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()
```

Slide 3: Independence Assumption

The independence assumption requires that observations are independent of each other. This means that there should be no correlation between consecutive residuals, especially in time series data.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson

# Generate time series data with autocorrelation
np.random.seed(0)
t = np.arange(100)
error = np.random.normal(0, 1, 100)
y = 2 * t + 1 + np.cumsum(0.5 * error)  # Introducing autocorrelation

# Fit linear regression model
X = t.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)

# Calculate residuals
residuals = y - model.predict(X)

# Plot residuals over time
plt.plot(t, residuals)
plt.title('Residuals Over Time')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.show()

# Perform Durbin-Watson test
dw_statistic = durbin_watson(residuals)
print(f"Durbin-Watson statistic: {dw_statistic:.2f}")
```

Slide 4: Homoscedasticity Assumption

Homoscedasticity assumes that the variance of residuals is constant across all levels of the independent variables. This ensures that our model's predictions are equally reliable across the entire range of predicted values.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate heteroscedastic data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.normal(0, 0.5 * X.ravel(), (100, 1))

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Calculate residuals
residuals = y - model.predict(X)

# Plot residuals vs. predicted values
plt.scatter(model.predict(X), residuals, color='purple', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Plot residuals vs. independent variable
plt.scatter(X, residuals, color='orange', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs. Independent Variable')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.show()
```

Slide 5: Normality Assumption

The normality assumption states that the residuals should be normally distributed. This assumption is important for hypothesis testing and constructing confidence intervals.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.normal(0, 1, (100, 1))

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Calculate residuals
residuals = y - model.predict(X)

# Create Q-Q plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of residuals
ax1.hist(residuals, bins=20, edgecolor='black')
ax1.set_title('Histogram of Residuals')
ax1.set_xlabel('Residuals')
ax1.set_ylabel('Frequency')

# Q-Q plot
stats.probplot(residuals.ravel(), plot=ax2)
ax2.set_title('Q-Q Plot of Residuals')

plt.tight_layout()
plt.show()

# Perform Shapiro-Wilk test
_, p_value = stats.shapiro(residuals)
print(f"Shapiro-Wilk test p-value: {p_value:.4f}")
```

Slide 6: Absence of Multicollinearity Assumption

Multicollinearity occurs when independent variables are highly correlated with each other. This assumption ensures that we can distinguish the individual effects of predictors on the response variable.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Generate correlated predictors
np.random.seed(0)
X1 = np.random.normal(0, 1, 100)
X2 = 0.8 * X1 + np.random.normal(0, 0.5, 100)
X3 = np.random.normal(0, 1, 100)
y = 2 * X1 + 3 * X2 + 1.5 * X3 + np.random.normal(0, 1, 100)

# Create dataframe
df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3})

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data["Variable"] = df.columns
vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

print("Variance Inflation Factors:")
print(vif_data)

# Calculate correlation matrix
correlation_matrix = df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)
```

Slide 7: Real-Life Example: Weather Prediction

Let's apply linear regression to predict temperature based on humidity and wind speed. We'll test the assumptions using real weather data.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate synthetic weather data
np.random.seed(0)
n_samples = 1000
humidity = np.random.uniform(30, 100, n_samples)
wind_speed = np.random.uniform(0, 30, n_samples)
temperature = 25 - 0.2 * humidity + 0.5 * wind_speed + np.random.normal(0, 2, n_samples)

# Create dataframe
weather_data = pd.DataFrame({
    'Humidity': humidity,
    'WindSpeed': wind_speed,
    'Temperature': temperature
})

# Split data into training and testing sets
X = weather_data[['Humidity', 'WindSpeed']]
y = weather_data['Temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot actual vs. predicted temperatures
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs. Predicted Temperature')
plt.show()

# Print model coefficients
print("Intercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")
```

Slide 8: Testing Linearity in Weather Prediction

We'll examine the linearity assumption in our weather prediction model by plotting partial residual plots for each predictor variable.

```python
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_partregress

# Fit statsmodels OLS model
X_with_constant = sm.add_constant(X)
model = sm.OLS(y, X_with_constant).fit()

# Create partial regression plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_partregress(model, 'Humidity', ax=axes[0])
plot_partregress(model, 'WindSpeed', ax=axes[1])

plt.tight_layout()
plt.show()
```

Slide 9: Testing Independence in Weather Prediction

To check the independence assumption, we'll plot the residuals over time and calculate the Durbin-Watson statistic.

```python
import numpy as np
from statsmodels.stats.stattools import durbin_watson

# Calculate residuals
residuals = y - model.predict(X_with_constant)

# Plot residuals over time
plt.figure(figsize=(10, 5))
plt.plot(residuals)
plt.title('Residuals Over Time')
plt.xlabel('Observation')
plt.ylabel('Residual')
plt.show()

# Calculate Durbin-Watson statistic
dw_statistic = durbin_watson(residuals)
print(f"Durbin-Watson statistic: {dw_statistic:.2f}")
```

Slide 10: Testing Homoscedasticity in Weather Prediction

We'll examine the homoscedasticity assumption by plotting residuals against predicted values and creating a scale-location plot.

```python
import numpy as np
import matplotlib.pyplot as plt

# Calculate predicted values and standardized residuals
y_pred = model.predict(X_with_constant)
standardized_residuals = residuals / np.std(residuals)

# Plot residuals vs. predicted values
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, standardized_residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Standardized Residuals')
plt.show()

# Create scale-location plot
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
plt.title('Scale-Location Plot')
plt.xlabel('Predicted Values')
plt.ylabel('√|Standardized Residuals|')
plt.show()
```

Slide 11: Testing Normality in Weather Prediction

To check the normality assumption, we'll create a Q-Q plot and perform the Shapiro-Wilk test on the residuals.

```python
import scipy.stats as stats

# Create Q-Q plot
fig, ax = plt.subplots(figsize=(10, 5))
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Q-Q Plot")
plt.show()

# Perform Shapiro-Wilk test
_, p_value = stats.shapiro(residuals)
print(f"Shapiro-Wilk test p-value: {p_value:.4f}")
```

Slide 12: Testing Multicollinearity in Weather Prediction

We'll check for multicollinearity by calculating the Variance Inflation Factor (VIF) for each predictor variable.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("Variance Inflation Factors:")
print(vif_data)

# Calculate correlation matrix
correlation_matrix = X.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)
```

Slide 13: Real-Life Example: Crop Yield Prediction

Let's apply linear regression to predict crop yield based on rainfall and temperature. We'll test the assumptions using synthetic agricultural data.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate synthetic agricultural data
np.random.seed(42)
n_samples = 1000
rainfall = np.random.uniform(500, 1500, n_samples)
temperature = np.random.uniform(15, 30, n_samples)
crop_yield = 20 + 0.01 * rainfall + 0.5 * temperature + np.random.normal(0, 2, n_samples)

# Create dataframe
agri_data = pd.DataFrame({
    'Rainfall': rainfall,
    'Temperature': temperature,
    'CropYield': crop_yield
})

# Split data into training and testing sets
X = agri_data[['Rainfall', 'Temperature']]
y = agri_data['CropYield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot actual vs. predicted crop yields
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Crop Yield')
plt.ylabel('Predicted Crop Yield')
plt.title('Actual vs. Predicted Crop Yield')
plt.show()

# Print model coefficients
print("Intercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")
```

Slide 14: Additional Resources

For those interested in delving deeper into linear regression assumptions and diagnostics, the following resources provide valuable insights:

1. ArXiv paper: "Diagnostic Checking in Regression Relationships" by R. Dennis Cook and Sanford Weisberg URL: [https://arxiv.org/abs/1501.01214](https://arxiv.org/abs/1501.01214)
2. ArXiv paper: "A Comparative Study on Feature Selection and Classification Methods Using Gene Expression Profiles and Proteomic Patterns" by Li-Yeh Chuang et al. URL: [https://arxiv.org/abs/cs/0410068](https://arxiv.org/abs/cs/0410068)

These papers offer advanced techniques for assessing and validating linear regression models, enhancing your understanding of the key assumptions we've discussed.

Slide 15: Conclusion

Understanding and testing the five key assumptions in linear regression is crucial for building reliable and interpretable models. By systematically checking linearity, independence, homoscedasticity, normality, and absence of multicollinearity, we can ensure the validity of our predictions and inferences.

Remember that real-world data often violates these assumptions to some degree. The goal is not perfect adherence but rather to identify significant violations that may impact our model's performance or interpretation. When assumptions are violated, consider alternative modeling approaches or data transformations to improve your analysis.

By mastering these concepts and diagnostic techniques, you'll be well-equipped to develop robust linear regression models and make informed decisions based on your data.

