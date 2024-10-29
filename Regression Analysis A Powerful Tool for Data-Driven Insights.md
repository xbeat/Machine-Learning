## Regression Analysis A Powerful Tool for Data-Driven Insights
Slide 1: Linear Regression Fundamentals

Linear regression forms the foundation of predictive modeling by establishing relationships between variables through a linear equation. It minimizes the sum of squared residuals to find the best-fitting line through data points, making it essential for forecasting and analysis.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate sample data
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)

# Create and fit model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

print(f"Coefficient: {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
```

Slide 2: Mathematical Foundation of Linear Regression

The mathematical basis of linear regression relies on specific formulas that define the relationship between variables and determine the optimal parameters for prediction accuracy.

```python
# Mathematical representation of Linear Regression
"""
$$y = \beta_0 + \beta_1x + \epsilon$$

Where:
$$\beta_0$$ = Intercept
$$\beta_1$$ = Slope
$$\epsilon$$ = Error term

Cost Function:
$$J(\beta_0, \beta_1) = \frac{1}{2n} \sum_{i=1}^n (y_i - (\beta_0 + \beta_1x_i))^2$$
"""

def manual_linear_regression(X, y):
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    beta_1 = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean) ** 2)
    beta_0 = y_mean - beta_1 * X_mean
    
    return beta_0, beta_1
```

Slide 3: Multiple Linear Regression Implementation

Multiple linear regression extends the simple linear model by incorporating multiple independent variables, enabling more complex predictions based on multiple features simultaneously.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create synthetic dataset
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 3)
y = 2 * X[:, 0] + 3.5 * X[:, 1] - 1.2 * X[:, 2] + np.random.randn(n_samples) * 0.1

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
multi_model = LinearRegression()
multi_model.fit(X_train_scaled, y_train)

print("Feature coefficients:", multi_model.coef_)
print("R² Score:", multi_model.score(X_test_scaled, y_test))
```

Slide 4: Polynomial Regression

Polynomial regression captures non-linear relationships by extending linear regression to include polynomial terms, allowing for more flexible and complex model fitting capabilities.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate non-linear data
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1) * 0.5

# Create polynomial model
degree = 2
poly_model = make_pipeline(
    PolynomialFeatures(degree),
    LinearRegression()
)
poly_model.fit(X, y)

# Generate predictions
X_test = np.linspace(-6, 6, 100).reshape(-1, 1)
y_pred = poly_model.predict(X_test)
```

Slide 5: Ridge Regression (L2 Regularization)

Ridge regression adds L2 regularization to linear regression, preventing overfitting by penalizing large coefficients. This technique is particularly useful when dealing with multicollinearity.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Generate correlated features
X = np.random.randn(1000, 5)
X[:, 3] = X[:, 0] + np.random.randn(1000) * 0.1
X[:, 4] = X[:, 1] - X[:, 2] + np.random.randn(1000) * 0.1
y = np.sum(X[:, :2], axis=1) + np.random.randn(1000) * 0.1

# Train Ridge model
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

print("Ridge coefficients:", ridge.coef_)
print("MSE:", mean_squared_error(y, ridge.predict(X)))
```

Slide 6: Lasso Regression (L1 Regularization)

Lasso regression implements L1 regularization, which can force coefficients to exactly zero, effectively performing feature selection while regularizing. This makes it particularly valuable for high-dimensional datasets.

```python
from sklearn.linear_model import Lasso
import pandas as pd

# Generate sparse data
X = np.random.randn(1000, 20)
true_coef = np.zeros(20)
true_coef[:5] = [1.5, -2, 3, -1, 0.5]
y = np.dot(X, true_coef) + np.random.randn(1000) * 0.1

# Train Lasso model
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# Display results
coef_df = pd.DataFrame({
    'Feature': [f'X{i}' for i in range(20)],
    'Coefficient': lasso.coef_
})
print(coef_df[abs(coef_df['Coefficient']) > 1e-10])
```

Slide 7: Elastic Net Regression

Elastic Net combines L1 and L2 regularization, providing a balanced approach that handles both feature selection and coefficient shrinkage while maintaining model stability.

```python
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

# Prepare data with mixed effects
X = np.random.randn(1000, 15)
beta = np.array([1, 0.5, 0.2, 0, 0, 1.5, 0, 0, 0.8, 0, 0, 0.3, 0, 0, 0])
y = np.dot(X, beta) + np.random.randn(1000) * 0.1

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Elastic Net model
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_scaled, y)

# Analyze coefficients
coef_importance = pd.DataFrame({
    'Feature': [f'Feature_{i}' for i in range(15)],
    'True_Coef': beta,
    'Estimated_Coef': elastic.coef_
})
print(coef_importance)
```

Slide 8: Cross-Validation in Regression

Cross-validation ensures robust model evaluation by splitting data into multiple folds, training and testing on different combinations to assess model stability and generalization.

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, r2_score, mean_squared_error

# Generate dataset
X = np.random.randn(500, 3)
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(500) * 0.1

# Configure cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = LinearRegression()

# Perform cross-validation with multiple metrics
r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
mse_scores = -cross_val_score(model, X, y, cv=kfold, 
                             scoring='neg_mean_squared_error')

print(f"R² scores: {r2_scores.mean():.3f} (±{r2_scores.std()*2:.3f})")
print(f"MSE scores: {mse_scores.mean():.3f} (±{mse_scores.std()*2:.3f})")
```

Slide 9: Feature Selection Using Regression

Feature selection techniques in regression help identify the most relevant predictors, improving model interpretability and reducing overfitting through systematic variable elimination.

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

# Generate data with irrelevant features
n_features = 20
n_informative = 5
X = np.random.randn(1000, n_features)
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + 0.5*X[:, 3] + X[:, 4]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform feature selection using Lasso
selector = SelectFromModel(Lasso(alpha=0.01))
selector.fit(X_scaled, y)

# Get selected features
selected_features = np.where(selector.get_support())[0]
print(f"Selected features: {selected_features}")
print(f"Number of selected features: {len(selected_features)}")
```

Slide 10: Time Series Regression

Time series regression addresses temporal dependencies in data, incorporating lagged variables and seasonal components to forecast future values based on historical patterns.

```python
import pandas as pd
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

# Generate time series data
dates = pd.date_range('2022-01-01', periods=365, freq='D')
y = np.random.normal(loc=10, scale=1, size=365) + \
    np.sin(np.linspace(0, 4*np.pi, 365)) * 3

# Create time features
dp = DeterministicProcess(
    index=dates,
    constant=True,
    order=1,
    seasonal=True,
    period=365
)
X = dp.in_sample()

# Fit model
model = LinearRegression()
model.fit(X, y)

# Make predictions
future_dates = pd.date_range('2023-01-01', periods=90, freq='D')
X_forecast = dp.out_of_sample(steps=90)
y_forecast = model.predict(X_forecast)

print(f"Model R²: {model.score(X, y):.3f}")
```

Slide 11: Regularized Time Series Regression

Regularized time series regression combines temporal modeling with regularization techniques to prevent overfitting while capturing complex seasonal and trend patterns in sequential data.

```python
from sklearn.preprocessing import SplineTransformer
import pandas as pd
import numpy as np

# Generate complex time series data
n_samples = 500
time = np.linspace(0, 10, n_samples)
seasonal = 2 * np.sin(2 * np.pi * time) + np.sin(4 * np.pi * time)
trend = 0.5 * time
noise = np.random.normal(0, 0.5, n_samples)
y = seasonal + trend + noise

# Create spline features
spline = SplineTransformer(n_knots=10, degree=3)
X_spline = spline.fit_transform(time.reshape(-1, 1))

# Apply regularized regression
from sklearn.linear_model import RidgeCV
model = RidgeCV(alphas=[0.1, 1.0, 10.0])
model.fit(X_spline, y)

# Predictions and evaluation
y_pred = model.predict(X_spline)
mse = np.mean((y - y_pred) ** 2)
print(f"MSE: {mse:.4f}")
print(f"Selected alpha: {model.alpha_:.4f}")
```

Slide 12: Robust Regression Implementation

Robust regression methods handle outliers and violations of standard assumptions by using techniques less sensitive to extreme values than ordinary least squares.

```python
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

# Generate data with outliers
np.random.seed(42)
n_samples = 200
X = np.random.normal(size=(n_samples, 2))
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(size=n_samples)

# Add outliers
outlier_indices = np.random.choice(n_samples, 20, replace=False)
y[outlier_indices] += np.random.normal(size=20) * 10

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit robust model
robust_model = HuberRegressor(epsilon=1.35)
robust_model.fit(X_scaled, y)

# Compare with standard linear regression
standard_model = LinearRegression()
standard_model.fit(X_scaled, y)

print("Robust Regression coefficients:", robust_model.coef_)
print("Standard Regression coefficients:", standard_model.coef_)
```

Slide 13: Real-world Application: Housing Price Prediction

This implementation demonstrates a complete regression pipeline for predicting housing prices, including data preprocessing, feature engineering, and model evaluation.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score
import pandas as pd

# Generate synthetic housing data
n_samples = 1000
data = {
    'size': np.random.normal(150, 30, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'location': np.random.choice(['urban', 'suburban', 'rural'], n_samples),
    'age': np.random.randint(0, 50, n_samples),
    'price': np.zeros(n_samples)
}
df = pd.DataFrame(data)
df['price'] = (2000 * df['size'] + 50000 * df['bedrooms'] - 
               1000 * df['age'] + np.random.normal(0, 50000, n_samples))

# Define feature types
numeric_features = ['size', 'bedrooms', 'age']
categorical_features = ['location']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Create model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])

# Evaluate model
scores = cross_val_score(model, 
                        df.drop('price', axis=1), 
                        df['price'],
                        cv=5,
                        scoring='r2')

print(f"Cross-validation R² scores: {scores.mean():.3f} (±{scores.std()*2:.3f})")
```

Slide 14: Additional Resources

*   [https://arxiv.org/abs/2006.12832](https://arxiv.org/abs/2006.12832) - "A Survey of Deep Learning Approaches for Linear Regression"
*   [https://arxiv.org/abs/1909.12297](https://arxiv.org/abs/1909.12297) - "Regularization Methods for High-Dimensional Regression"
*   [https://arxiv.org/abs/2103.06122](https://arxiv.org/abs/2103.06122) - "Modern Regression Techniques: A Comprehensive Review"
*   [https://arxiv.org/abs/1803.08823](https://arxiv.org/abs/1803.08823) - "Time Series Regression and Its Applications"
*   [https://arxiv.org/abs/2007.04118](https://arxiv.org/abs/2007.04118) - "Robust Regression Methods: Theory and Applications"

