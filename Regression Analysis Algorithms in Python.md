## Regression Analysis Algorithms in Python
Slide 1: Linear Regression Fundamentals

Linear regression remains the most widely used algorithm for regression analysis, forming the foundation for many advanced techniques. It models relationships between dependent and independent variables through a linear approach.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.1, 4.2, 6.1, 8.2, 9.9])

# Initialize and fit model
model = LinearRegression()
model.fit(X, y)

# Print coefficients and intercept
print(f"Coefficient: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
```

Slide 2: Mathematical Foundation

The mathematical principles behind linear regression involve minimizing the sum of squared residuals between predicted and actual values, expressed through specific formulas and matrices.

```python
# Mathematical representation (not rendered)
$$
\hat{y} = X\beta + \epsilon
$$
$$
\beta = (X^TX)^{-1}X^Ty
$$
$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$
```

Slide 3: Implementation from Scratch

Building linear regression from scratch helps understand the underlying mathematics and optimization process while providing complete control over the implementation.

```python
class LinearRegressionScratch:
    def fit(self, X, y):
        X = np.column_stack([np.ones(len(X)), X])
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y
        
    def predict(self, X):
        X = np.column_stack([np.ones(len(X)), X])
        return X @ self.beta

# Example usage
model = LinearRegressionScratch()
X_train = np.array([[1], [2], [3], [4]])
y_train = np.array([2, 4, 6, 8])
model.fit(X_train, y_train)
```

Slide 4: Multiple Linear Regression

Multiple linear regression extends the basic concept to handle multiple independent variables, enabling more complex relationship modeling in real-world scenarios.

```python
from sklearn.datasets import make_regression

# Generate synthetic data with multiple features
X, y = make_regression(n_samples=100, n_features=3, noise=0.1)

# Fit multiple linear regression
multi_model = LinearRegression()
multi_model.fit(X, y)

print("Feature coefficients:", multi_model.coef_)
```

Slide 5: Polynomial Regression

Polynomial regression addresses non-linear relationships by transforming features into polynomial terms while maintaining the linear regression framework.

```python
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
X = np.array([[1], [2], [3], [4]])
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit polynomial regression
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
```

Slide 6: Real-world Example 1: House Price Prediction

Understanding house price prediction through regression analysis using real estate data with multiple features and polynomial transformations.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load house price dataset
df = pd.DataFrame({
    'size': [1400, 1600, 1700, 1875, 1100],
    'rooms': [3, 4, 3, 4, 2],
    'price': [245000, 312000, 279000, 308000, 199000]
})

# Prepare features and target
X = df[['size', 'rooms']]
y = df['price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Slide 7: House Price Model Implementation

```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Create and train model
house_model = LinearRegression()
house_model.fit(X_train, y_train)

# Make predictions
y_pred = house_model.predict(X_test)

# Calculate metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

Slide 8: Results for House Price Prediction

```python
print(f"Mean Squared Error: ${mse:.2f}")
print(f"R2 Score: {r2:.3f}")
print("\nCoefficients:")
for feature, coef in zip(['size', 'rooms'], house_model.coef_):
    print(f"{feature}: ${coef:.2f}")
```

Slide 9: Real-world Example 2: Stock Price Analysis

Implementing regression analysis for stock price prediction using historical market data and technical indicators.

```python
import yfinance as yf
from datetime import datetime, timedelta

# Download stock data
symbol = "AAPL"
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
df = yf.download(symbol, start=start_date, end=end_date)

# Calculate technical indicators
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['RSI'] = calculate_rsi(df['Close'], periods=14)
```

Slide 10: Stock Price Model Implementation

```python
# Prepare features
feature_columns = ['SMA_20', 'RSI', 'Volume']
X = df[feature_columns].dropna()
y = df['Close'].dropna()

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
stock_model = LinearRegression()
stock_model.fit(X_train, y_train)
```

Slide 11: Results for Stock Price Analysis

```python
# Make predictions
y_pred = stock_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: ${rmse:.2f}")
print(f"R2 Score: {r2:.3f}")
```

Slide 12: Ridge Regression Implementation

Ridge regression adds L2 regularization to prevent overfitting and handle multicollinearity in the dataset effectively.

```python
from sklearn.linear_model import Ridge

# Initialize and train Ridge model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Compare with standard linear regression
ridge_pred = ridge_model.predict(X_test)
ridge_r2 = r2_score(y_test, ridge_pred)
print(f"Ridge R2 Score: {ridge_r2:.3f}")
```

Slide 13: Lasso Regression Implementation

Lasso regression implements L1 regularization for feature selection and sparse coefficient solutions.

```python
from sklearn.linear_model import Lasso

# Initialize and train Lasso model
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)

# Evaluate performance
lasso_pred = lasso_model.predict(X_test)
lasso_r2 = r2_score(y_test, lasso_pred)
print(f"Lasso R2 Score: {lasso_r2:.3f}")
```

Slide 14: Additional Resources

1.  [https://arxiv.org/abs/1509.09169](https://arxiv.org/abs/1509.09169) - "A Comparative Study of Linear Regression Algorithms"
2.  [https://arxiv.org/abs/1803.08823](https://arxiv.org/abs/1803.08823) - "Understanding Regularized Linear Models"
3.  [https://arxiv.org/abs/1711.10561](https://arxiv.org/abs/1711.10561) - "Modern Regression Techniques in Practice"
4.  [https://arxiv.org/abs/1902.06502](https://arxiv.org/abs/1902.06502) - "Advances in Linear Regression Methods"

