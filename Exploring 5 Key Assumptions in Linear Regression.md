## Exploring 5 Key Assumptions in Linear Regression
Slide 1: Key Assumptions in Linear Regression

Linear regression is a fundamental statistical technique used to model the relationship between variables. To ensure the validity and reliability of our model, we need to verify five key assumptions. Let's explore these assumptions and learn how to test them using Python.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate sample data
X, y = make_regression(n_samples=100, n_features=1, noise=20)
model = LinearRegression().fit(X, y)

# Plot the data and regression line
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.title('Linear Regression Example')
plt.show()
```

Slide 2: Multicollinearity

Multicollinearity occurs when predictor variables are highly correlated with each other. This can make it difficult to determine the individual effect of each predictor and can inflate standard errors. To test for multicollinearity, we use the Variance Inflation Factor (VIF).

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import make_regression

# Generate sample data with multiple features
X, y = make_regression(n_samples=100, n_features=3, noise=0.1)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = ["X1", "X2", "X3"]
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

print(vif_data)
```

Slide 3: Normality of Residuals

The residuals (errors) of the model should be normally distributed. This assumption is important because many statistical tests rely on it. We can test for normality using a Q-Q plot and the Shapiro-Wilk test.

```python
from scipy import stats

# Fit the model and calculate residuals
model = LinearRegression().fit(X, y)
residuals = y - model.predict(X)

# Q-Q plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
stats.probplot(residuals, dist="norm", plot=ax1)
ax1.set_title("Q-Q plot")

# Histogram
ax2.hist(residuals, bins=20)
ax2.set_title("Histogram of Residuals")
plt.show()

# Shapiro-Wilk test
_, p_value = stats.shapiro(residuals)
print(f"Shapiro-Wilk test p-value: {p_value:.4f}")
```

Slide 4: Linearity

The relationship between the independent and dependent variables should be linear. If this assumption is violated, our model's predictions may be inaccurate. We can check for linearity by plotting residuals against predicted values.

```python
# Plot residuals vs predicted values
predicted = model.predict(X)
plt.scatter(predicted, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()

# Calculate and print correlation between residuals and predicted values
correlation = np.corrcoef(predicted.flatten(), residuals)[0, 1]
print(f"Correlation between residuals and predicted values: {correlation:.4f}")
```

Slide 5: Homoscedasticity

Homoscedasticity means that the variance of residuals should remain constant across all levels of the independent variables. We can check for homoscedasticity visually and use statistical tests like the Breusch-Pagan test.

```python
from statsmodels.stats.diagnostic import het_breuschpagan

# Visual check
plt.scatter(predicted, np.abs(residuals))
plt.xlabel("Predicted Values")
plt.ylabel("Absolute Residuals")
plt.title("Homoscedasticity Check")
plt.show()

# Breusch-Pagan test
_, p_value, _, _ = het_breuschpagan(residuals, X)
print(f"Breusch-Pagan test p-value: {p_value:.4f}")
```

Slide 6: No Autocorrelation of Residuals

Residuals should not be correlated with each other. Autocorrelation can invalidate standard errors, leading to incorrect inferences about model coefficients. We can use the Durbin-Watson test to check for autocorrelation.

```python
from statsmodels.stats.stattools import durbin_watson

# Durbin-Watson test
dw_statistic = durbin_watson(residuals)
print(f"Durbin-Watson statistic: {dw_statistic:.4f}")

# Plot residuals over time (assuming ordered data)
plt.plot(residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Observation")
plt.ylabel("Residual")
plt.title("Residuals Over Time")
plt.show()
```

Slide 7: Real-Life Example: Housing Prices

Let's apply these concepts to a real-world scenario: predicting housing prices based on various features such as square footage, number of bedrooms, and location.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset (you would typically load your own data here)
data = pd.read_csv('housing_data.csv')

# Select features and target
X = data[['sqft', 'bedrooms', 'bathrooms', 'age']]
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the model
model = LinearRegression().fit(X_train_scaled, y_train)

# Print model coefficients
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
```

Slide 8: Checking Assumptions for Housing Price Model

Now that we have our housing price model, let's check if it satisfies the assumptions we discussed earlier.

```python
# Predict and calculate residuals
y_pred = model.predict(X_train_scaled)
residuals = y_train - y_pred

# Check for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]
print("VIF for each feature:")
print(vif_data)

# Check for normality of residuals
_, p_value = stats.shapiro(residuals)
print(f"\nShapiro-Wilk test p-value: {p_value:.4f}")

# Check for homoscedasticity
_, p_value, _, _ = het_breuschpagan(residuals, X_train_scaled)
print(f"Breusch-Pagan test p-value: {p_value:.4f}")

# Check for autocorrelation
dw_statistic = durbin_watson(residuals)
print(f"Durbin-Watson statistic: {dw_statistic:.4f}")
```

Slide 9: Visualizing Assumption Checks

Let's visualize some of the assumption checks for our housing price model.

```python
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=ax1)
ax1.set_title("Q-Q plot")

# Residuals vs Predicted
ax2.scatter(y_pred, residuals)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel("Predicted Values")
ax2.set_ylabel("Residuals")
ax2.set_title("Residuals vs Predicted Values")

# Histogram of residuals
ax3.hist(residuals, bins=20)
ax3.set_title("Histogram of Residuals")

# Residuals over time
ax4.plot(residuals)
ax4.axhline(y=0, color='r', linestyle='--')
ax4.set_xlabel("Observation")
ax4.set_ylabel("Residual")
ax4.set_title("Residuals Over Time")

plt.tight_layout()
plt.show()
```

Slide 10: Addressing Violations of Assumptions

If we find violations of these assumptions, we need to take corrective actions. Here are some common approaches:

```python
# Example: Addressing multicollinearity
from sklearn.decomposition import PCA

# Apply PCA to reduce multicollinearity
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_pca = pca.fit_transform(X_train_scaled)

# Fit a new model with PCA-transformed features
model_pca = LinearRegression().fit(X_train_pca, y_train)

# Example: Addressing non-linearity
from sklearn.preprocessing import PolynomialFeatures

# Add polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)

# Fit a new model with polynomial features
model_poly = LinearRegression().fit(X_train_poly, y_train)

# Print results
print("Original R-squared:", model.score(X_test_scaled, y_test))
print("PCA R-squared:", model_pca.score(pca.transform(X_test_scaled), y_test))
print("Polynomial R-squared:", model_poly.score(poly.transform(X_test_scaled), y_test))
```

Slide 11: Real-Life Example: Air Quality Prediction

Let's explore another real-world application: predicting air quality index based on environmental factors such as temperature, humidity, and wind speed.

```python
# Load air quality data (you would typically load your own data here)
air_quality_data = pd.read_csv('air_quality_data.csv')

# Select features and target
X = air_quality_data[['temperature', 'humidity', 'wind_speed', 'pressure']]
y = air_quality_data['air_quality_index']

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the model
air_quality_model = LinearRegression().fit(X_train_scaled, y_train)

# Print model performance
print("R-squared:", air_quality_model.score(X_test_scaled, y_test))

# Print feature importances
for feature, coef in zip(X.columns, air_quality_model.coef_):
    print(f"{feature}: {coef:.2f}")
```

Slide 12: Checking Assumptions for Air Quality Model

Now let's verify if our air quality prediction model satisfies the linear regression assumptions.

```python
# Predict and calculate residuals
y_pred = air_quality_model.predict(X_train_scaled)
residuals = y_train - y_pred

# Check for multicollinearity
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]
print("VIF for each feature:")
print(vif_data)

# Check for normality of residuals
_, p_value = stats.shapiro(residuals)
print(f"\nShapiro-Wilk test p-value: {p_value:.4f}")

# Check for homoscedasticity
_, p_value, _, _ = het_breuschpagan(residuals, X_train_scaled)
print(f"Breusch-Pagan test p-value: {p_value:.4f}")

# Check for autocorrelation
dw_statistic = durbin_watson(residuals)
print(f"Durbin-Watson statistic: {dw_statistic:.4f}")

# Plot residuals vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Air Quality Index")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values (Air Quality Model)")
plt.show()
```

Slide 13: Interpreting and Improving the Air Quality Model

Based on our assumption checks, let's interpret the results and consider ways to improve our air quality prediction model.

```python
# Interpret the results
print("Model Interpretation:")
print("1. Multicollinearity: Check VIF values. VIF > 5 indicates potential issues.")
print("2. Normality: Shapiro-Wilk p-value < 0.05 suggests non-normal residuals.")
print("3. Homoscedasticity: Breusch-Pagan p-value < 0.05 indicates heteroscedasticity.")
print("4. Autocorrelation: Durbin-Watson should be close to 2.")

# Example improvement: Feature engineering
X['temp_humidity_interaction'] = X['temperature'] * X['humidity']

# Refit the model with the new feature
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

improved_model = LinearRegression().fit(X_train_scaled, y_train)

print("\nImproved Model R-squared:", improved_model.score(X_test_scaled, y_test))

# Compare feature importances
for feature, coef in zip(X.columns, improved_model.coef_):
    print(f"{feature}: {coef:.2f}")
```

Slide 14: Additional Resources

For those interested in diving deeper into linear regression assumptions and diagnostics, here are some valuable resources:

1. "Regression Diagnostics: Identifying Influential Data and Sources of Collinearity" by Belsley, Kuh, and Welsch (Wiley Series in Probability and Statistics)
2. "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani (available at: [https://www.statlearning.com/](https://www.statlearning.com/))
3. "Linear Model Methodology" by Seber and Lee (Wiley Series in Probability and Statistics)
4. ArXiv paper: "A Comprehensive Survey of Regression Based Loss Functions" by Wang et al. ([https://arxiv.org/abs/2103.15234](https://arxiv.org/abs/2103.15234))

These resources provide in-depth discussions on regression assumptions, diagnostics, and advanced techniques for improving linear models.

