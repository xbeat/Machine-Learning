## Evaluating Regression Model Metrics in Python
Slide 1: Evaluating Regression Model Performance

Regression models are essential tools in predictive analytics. To ensure their effectiveness, we need reliable metrics to assess their performance. This presentation will explore key evaluation metrics for regression models, including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared (R²), and Adjusted R-squared. We'll demonstrate how to implement these metrics using Python, providing practical examples along the way.

```python
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Fit a linear regression model
model = LinearRegression().fit(X, y)

# Make predictions
y_pred = model.predict(X)

# We'll use this data to calculate our metrics
```

Slide 2: Mean Squared Error (MSE)

Mean Squared Error is a fundamental metric that measures the average squared difference between predicted and actual values. It penalizes larger errors more heavily due to the squaring operation. A lower MSE indicates better model performance.

```python
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

mse = calculate_mse(y, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Using sklearn
mse_sklearn = mean_squared_error(y, y_pred)
print(f"MSE (sklearn): {mse_sklearn:.4f}")
```

Slide 3: Root Mean Squared Error (RMSE)

RMSE is the square root of MSE. It provides an error metric in the same unit as the target variable, making it more interpretable. Like MSE, a lower RMSE indicates better model performance.

```python
def calculate_rmse(y_true, y_pred):
    return np.sqrt(calculate_mse(y_true, y_pred))

rmse = calculate_rmse(y, y_pred)
print(f"Root Mean Squared Error: {rmse:.4f}")

# Using sklearn
rmse_sklearn = np.sqrt(mean_squared_error(y, y_pred))
print(f"RMSE (sklearn): {rmse_sklearn:.4f}")
```

Slide 4: R-squared (R²)

R-squared, also known as the coefficient of determination, measures the proportion of variance in the dependent variable that is predictable from the independent variable(s). It ranges from 0 to 1, with 1 indicating perfect prediction and 0 indicating that the model performs no better than a horizontal line.

```python
def calculate_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - (ss_residual / ss_total)

r2 = calculate_r2(y, y_pred)
print(f"R-squared: {r2:.4f}")

# Using sklearn
r2_sklearn = r2_score(y, y_pred)
print(f"R-squared (sklearn): {r2_sklearn:.4f}")
```

Slide 5: Adjusted R-squared

Adjusted R-squared modifies the R-squared by penalizing the addition of extraneous predictors to the model. It's particularly useful when comparing models with different numbers of predictors.

```python
def calculate_adjusted_r2(y_true, y_pred, n_features):
    r2 = calculate_r2(y_true, y_pred)
    n = len(y_true)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

adj_r2 = calculate_adjusted_r2(y, y_pred, X.shape[1])
print(f"Adjusted R-squared: {adj_r2:.4f}")
```

Slide 6: Real-life Example: Housing Price Prediction

Let's apply these metrics to a real-world scenario of predicting housing prices based on various features like square footage, number of bedrooms, etc.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression().fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
adj_r2 = calculate_adjusted_r2(y_test, y_pred, X.shape[1])

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adj_r2:.4f}")
```

Slide 7: Interpreting the Results

The metrics we calculated provide insights into our model's performance. A low MSE and RMSE suggest that our predictions are close to the actual values. The R-squared value indicates how much of the variance in housing prices our model explains. The adjusted R-squared helps us understand if we're overfitting by adding too many features.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Housing Prices")
plt.tight_layout()
plt.show()
```

Slide 8: Mean Absolute Error (MAE)

Mean Absolute Error is another useful metric that measures the average magnitude of errors in a set of predictions, without considering their direction. It's less sensitive to outliers compared to MSE and RMSE.

```python
from sklearn.metrics import mean_absolute_error

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

mae = calculate_mae(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.4f}")

# Using sklearn
mae_sklearn = mean_absolute_error(y_test, y_pred)
print(f"MAE (sklearn): {mae_sklearn:.4f}")
```

Slide 9: Real-life Example: Stock Price Prediction

Let's apply our metrics to another real-world scenario: predicting stock prices based on historical data and various financial indicators.

```python
import pandas as pd
import yfinance as yf

# Download stock data (using Apple Inc. as an example)
stock_data = yf.download("AAPL", start="2020-01-01", end="2023-12-31")

# Prepare features and target
stock_data['Returns'] = stock_data['Close'].pct_change()
stock_data['MA_5'] = stock_data['Close'].rolling(window=5).mean()
stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
stock_data = stock_data.dropna()

X = stock_data[['Returns', 'MA_5', 'MA_20']]
y = stock_data['Close'].shift(-1).dropna()

# Align X and y
X = X.iloc[:-1]
y = y.iloc[:-1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model and make predictions
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R-squared: {r2:.4f}")
```

Slide 10: Cross-Validation for Model Evaluation

Cross-validation is a robust technique for assessing how the results of a statistical analysis will generalize to an independent data set. It's particularly useful when you have a limited amount of data.

```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(LinearRegression(), X, y, cv=5, 
                            scoring='neg_mean_squared_error')

# Convert MSE to RMSE
rmse_scores = np.sqrt(-cv_scores)

print("Cross-validated RMSE scores:", rmse_scores)
print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
print(f"Standard deviation of RMSE: {np.std(rmse_scores):.4f}")
```

Slide 11: Residual Analysis

Residual analysis is crucial for validating the assumptions of linear regression. It involves examining the differences between observed and predicted values.

```python
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.show()

# Q-Q plot for normality check
from scipy import stats

fig, ax = plt.subplots(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
```

Slide 12: Feature Importance

Understanding which features contribute most to your model's predictions can provide valuable insights. For linear regression, we can examine the coefficients.

```python
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
feature_importance = feature_importance.sort_values('Coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

Slide 13: Overfitting and Underfitting

Comparing training and testing errors can help detect overfitting or underfitting. If the training error is much lower than the testing error, the model might be overfitting.

```python
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training MSE: {train_mse:.4f}")
print(f"Testing MSE: {test_mse:.4f}")

# Learning curve
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    LinearRegression(), X, y, cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation error')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve')
plt.legend()
plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into regression model evaluation and related topics, here are some valuable resources:

1. "A Survey of Cross-Validation Procedures for Model Selection" by Sylvain Arlot and Alain Celisse (2010). Available at: [https://arxiv.org/abs/0907.4728](https://arxiv.org/abs/0907.4728)
2. "Regression Shrinkage and Selection via the Lasso" by Robert Tibshirani (1996). Available at: [https://arxiv.org/abs/math/9508054](https://arxiv.org/abs/math/9508054)
3. "An Introduction to Statistical Learning" by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani. This book provides an accessible overview of statistical learning methods with applications in R.

These resources offer in-depth discussions on model evaluation techniques, advanced regression methods, and statistical learning principles that can enhance your understanding of regression analysis and model performance evaluation.

