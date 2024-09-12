## Fundamental Assumptions of Linear Regression in Python
Slide 1: Linear Regression Assumptions

Linear regression is a fundamental statistical technique used for modeling the relationship between a dependent variable and one or more independent variables. To ensure the reliability and validity of our regression models, we must consider several key assumptions. This presentation will explore these assumptions, their importance, and how to verify them using Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate sample data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Plot the data and regression line
plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.title('Linear Regression Example')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.show()
```

Slide 2: Linearity Assumption

The linearity assumption states that there should be a linear relationship between the independent variables and the dependent variable. This means that changes in the independent variables should be associated with a constant change in the dependent variable.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate non-linear data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 3 * np.sin(X) + np.random.normal(0, 1, X.shape)

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Plot data and regression line
plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.title('Non-linear Data with Linear Regression')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.show()

# Plot residuals
residuals = y - model.predict(X)
plt.scatter(X, residuals, color='green', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Independent Variable')
plt.ylabel('Residuals')
plt.show()
```

Slide 3: Independence Assumption

The independence assumption requires that observations are independent of each other. This means that there should be no correlation between consecutive residuals, also known as autocorrelation.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson

# Generate data with autocorrelation
n = 100
X = np.linspace(0, 10, n).reshape(-1, 1)
y = 2 * X + np.random.normal(0, 1, X.shape)
y = np.cumsum(y) / 10  # Introduce autocorrelation

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Calculate residuals
residuals = y - model.predict(X)

# Durbin-Watson test
dw_statistic = durbin_watson(residuals)

# Plot residuals
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X, residuals, color='green', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Independent Variable')
plt.ylabel('Residuals')

# Plot autocorrelation
plt.subplot(122)
plt.plot(residuals[:-1], residuals[1:], 'o', alpha=0.5)
plt.title(f'Autocorrelation Plot (DW: {dw_statistic:.2f})')
plt.xlabel('Residual t')
plt.ylabel('Residual t+1')
plt.tight_layout()
plt.show()

print(f"Durbin-Watson statistic: {dw_statistic:.2f}")
```

Slide 4: Homoscedasticity Assumption

Homoscedasticity assumes that the variance of residuals is constant across all levels of the independent variables. This ensures that the model's predictive power is consistent across the range of predictor variables.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate heteroscedastic data
n = 100
X = np.linspace(0, 10, n).reshape(-1, 1)
y = 2 * X + np.random.normal(0, 0.5 * X.ravel(), X.shape)

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Calculate residuals
residuals = y - model.predict(X)

# Plot data and regression line
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.title('Heteroscedastic Data')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')

# Plot residuals
plt.subplot(122)
plt.scatter(model.predict(X), residuals, color='green', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot (Heteroscedastic)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.tight_layout()
plt.show()
```

Slide 5: Normality Assumption

The normality assumption states that the residuals should be normally distributed. This assumption is important for making valid statistical inferences about the model parameters.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

# Generate data with normally distributed errors
n = 1000
X = np.linspace(0, 10, n).reshape(-1, 1)
y = 2 * X + np.random.normal(0, 1, X.shape)

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Calculate residuals
residuals = y - model.predict(X)

# Plot histogram of residuals
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.hist(residuals, bins=30, edgecolor='black')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

# Q-Q plot
plt.subplot(122)
stats.probplot(residuals.ravel(), plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.tight_layout()
plt.show()

# Shapiro-Wilk test for normality
_, p_value = stats.shapiro(residuals)
print(f"Shapiro-Wilk test p-value: {p_value:.4f}")
```

Slide 6: No Perfect Multicollinearity

This assumption requires that there is no perfect linear relationship between two or more independent variables. Perfect multicollinearity can lead to unstable and unreliable coefficient estimates.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Generate data with multicollinearity
n = 100
X1 = np.random.rand(n)
X2 = 2 * X1 + np.random.normal(0, 0.1, n)
X3 = np.random.rand(n)
y = 3 * X1 + 2 * X2 + 4 * X3 + np.random.normal(0, 0.5, n)

# Create DataFrame
df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'y': y})

# Calculate correlation matrix
correlation_matrix = df.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Calculate Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Variable"] = df.columns[:-1]
vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1] - 1)]

print("\nVariance Inflation Factors:")
print(vif_data)
```

Slide 7: No Endogeneity

Endogeneity occurs when there is a correlation between the independent variable(s) and the error term. This can lead to biased and inconsistent coefficient estimates.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate data with endogeneity
n = 1000
z = np.random.normal(0, 1, n)
x = z + np.random.normal(0, 0.5, n)
e = z + np.random.normal(0, 0.5, n)
y = 2 * x + e

# Fit linear regression
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

# Plot data and regression line
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(x, y, alpha=0.5)
plt.plot(x, model.predict(x.reshape(-1, 1)), color='red', linewidth=2)
plt.title('Data with Endogeneity')
plt.xlabel('X')
plt.ylabel('Y')

# Plot residuals vs. X
residuals = y - model.predict(x.reshape(-1, 1))
plt.subplot(122)
plt.scatter(x, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs. X')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.tight_layout()
plt.show()

# Calculate correlation between X and residuals
correlation = np.corrcoef(x, residuals)[0, 1]
print(f"Correlation between X and residuals: {correlation:.4f}")
```

Slide 8: Sample Size Considerations

While not a formal assumption, having an adequate sample size is crucial for the reliability and stability of linear regression models. Larger sample sizes lead to more precise parameter estimates and increased statistical power.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve

# Generate data
n = 1000
X = np.random.rand(n, 1)
y = 2 * X + np.random.normal(0, 0.5, (n, 1))

# Calculate learning curve
train_sizes, train_scores, test_scores = learning_curve(
    LinearRegression(), X, y.ravel(), cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='neg_mean_squared_error'
)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, -np.mean(train_scores, axis=1), label='Training error')
plt.plot(train_sizes, -np.mean(test_scores, axis=1), label='Cross-validation error')
plt.title('Learning Curve for Linear Regression')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
```

Slide 9: Outliers and Influential Points

While not a formal assumption, the presence of outliers and influential points can significantly impact the regression results. It's important to identify and handle these points appropriately.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Generate data with outliers
n = 100
X = np.random.rand(n, 1)
y = 2 * X + np.random.normal(0, 0.3, (n, 1))

# Add outliers
X = np.vstack([X, np.array([[0.1], [0.9]])])
y = np.vstack([y, np.array([[5], [-5]])])

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Calculate Cook's distance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
influence = (y - model.predict(X))**2 / (2 * model.coef_**2)
cooks_distance = influence / (1 - influence)**2

# Plot data, regression line, and Cook's distance
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X, y, alpha=0.5)
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.title('Data with Outliers')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(122)
plt.stem(range(len(X)), cooks_distance.ravel())
plt.title("Cook's Distance")
plt.xlabel('Observation')
plt.ylabel("Cook's Distance")
plt.tight_layout()
plt.show()

# Identify influential points
threshold = 4 / len(X)
influential = cooks_distance > threshold
print(f"Number of influential points: {np.sum(influential)}")
```

Slide 10: Model Specification

Correct model specification is crucial for obtaining reliable results. This includes selecting the appropriate variables and functional form for the relationship between dependent and independent variables.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Generate non-linear data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X**2 + 3 * X + np.random.normal(0, 5, X.shape)

# Fit linear and polynomial models
linear_model = LinearRegression()
linear_model.fit(X, y)

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X, y, alpha=0.5)
plt.plot(X, linear_model.predict(X), color='red', label='Linear')
plt.plot(X, poly_model.predict(X_poly), color='green', label='Quadratic')
plt.title('Model Comparison')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Plot residuals
plt.subplot(122)
plt.scatter(X, y - linear_model.predict(X), color='red', alpha=0.5, label='Linear')
plt.scatter(X, y - poly_model.predict(X_poly), color='green', alpha=0.5, label='Quadratic')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.legend()
plt.tight_layout()
plt.show()

# Print model performance metrics
print("Linear Model:")
print(f"R-squared: {r2_score(y, linear_model.predict(X)):.4f}")
print(f"MSE: {mean_squared_error(y, linear_model.predict(X)):.4f}")

print("\nQuadratic Model:")
print(f"R-squared: {r2_score(y, poly_model.predict(X_poly)):.4f}")
print(f"MSE: {mean_squared_error(y, poly_model.predict(X_poly)):.4f}")
```

Slide 11: Real-Life Example: Housing Prices

Let's apply linear regression to predict housing prices based on various features such as square footage, number of bedrooms, and location.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample housing data
np.random.seed(42)
n_samples = 1000
square_feet = np.random.randint(1000, 4000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
location_score = np.random.uniform(0, 10, n_samples)
price = 100000 + 100 * square_feet + 20000 * bedrooms + 50000 * location_score + np.random.normal(0, 50000, n_samples)

# Create DataFrame
df = pd.DataFrame({
    'square_feet': square_feet,
    'bedrooms': bedrooms,
    'location_score': location_score,
    'price': price
})

# Split data into training and testing sets
X = df[['square_feet', 'bedrooms', 'location_score']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: ${mse:.2f}")
print(f"R-squared: {r2:.4f}")
print("\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: ${coef:.2f}")
print(f"Intercept: ${model.intercept_:.2f}")
```

Slide 12: Real-Life Example: Crop Yield Prediction

In this example, we'll use linear regression to predict crop yield based on factors such as temperature, rainfall, and soil quality.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate sample crop yield data
np.random.seed(42)
n_samples = 500
temperature = np.random.uniform(15, 35, n_samples)
rainfall = np.random.uniform(500, 1500, n_samples)
soil_quality = np.random.uniform(0, 10, n_samples)
yield_per_acre = 2000 + 50 * temperature + 0.5 * rainfall + 200 * soil_quality + np.random.normal(0, 500, n_samples)

# Create DataFrame
df = pd.DataFrame({
    'temperature': temperature,
    'rainfall': rainfall,
    'soil_quality': soil_quality,
    'yield_per_acre': yield_per_acre
})

# Split data into training and testing sets
X = df[['temperature', 'rainfall', 'soil_quality']]
y = df['yield_per_acre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.4f}")
print("\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Yield (kg/acre)")
plt.ylabel("Predicted Yield (kg/acre)")
plt.title("Actual vs Predicted Crop Yield")
plt.show()
```

Slide 13: Diagnostics and Model Improvement

After building a linear regression model, it's crucial to perform diagnostics and consider ways to improve the model's performance. Here are some techniques to assess and enhance your model:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE

# Assume we have X and y from previous examples

# 1. Cross-validation
cv_scores = cross_val_score(LinearRegression(), X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f}")

# 2. Feature importance
model = LinearRegression()
model.fit(X, y)
importance = pd.DataFrame({'feature': X.columns, 'importance': np.abs(model.coef_)})
importance = importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(importance)

# 3. Recursive Feature Elimination
rfe = RFE(estimator=LinearRegression(), n_features_to_select=2)
rfe.fit(X, y)
print("\nSelected features (RFE):")
print(X.columns[rfe.support_])

# 4. Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
print("\nPolynomial model R-squared:")
print(f"R-squared: {model_poly.score(X_poly, y):.4f}")

# 5. Residual plot
y_pred = model.predict(X)
residuals = y - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into linear regression and its assumptions, here are some valuable resources:

1. "Introduction to Linear Regression Analysis" by Montgomery, Peck, and Vining (Book)
2. "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani (Free PDF available)
3. Scikit-learn Documentation: Linear Models ([https://scikit-learn.org/stable/modules/linear\_model.html](https://scikit-learn.org/stable/modules/linear_model.html))
4. StatQuest with Josh Starmer: Linear Regression (YouTube Playlist)
5. ArXiv paper: "A Comprehensive Survey of Regression-Based Loss Functions" by Wang et al. ([https://arxiv.org/abs/2103.15331](https://arxiv.org/abs/2103.15331))

These resources provide a mix of theoretical foundations and practical applications of linear regression, helping you to further understand and apply the concepts discussed in this presentation.

