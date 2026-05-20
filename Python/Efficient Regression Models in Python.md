## Efficient Regression Models in Python
Slide 1: Introduction to Regression Models in Python

Regression analysis is a fundamental statistical technique used to model the relationship between variables. In Python, several regression models are available, each with its own computational efficiency. This presentation will explore the most computationally efficient regression models and their implementation using Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Plot the results
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
```

Slide 2: Linear Regression: The Baseline for Efficiency

Linear regression is one of the most computationally efficient regression models. It assumes a linear relationship between the input features and the target variable, making it fast to compute and easy to interpret.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate sample data
X = np.random.rand(1000, 5)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] - 1.5 * X[:, 4] + np.random.normal(0, 0.1, 1000)

# Create and fit the model
model = LinearRegression()
%time model.fit(X, y)

# Make predictions
predictions = model.predict(X[:5])
print("Predictions:", predictions)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
```

Slide 3: Ordinary Least Squares (OLS) Implementation

OLS is the mathematical foundation of linear regression. It minimizes the sum of squared residuals, providing an efficient closed-form solution for the regression coefficients.

```python
import numpy as np

def ols_regression(X, y):
    # Add a column of ones for the intercept
    X = np.column_stack((np.ones(X.shape[0]), X))
    
    # Calculate coefficients
    coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
    
    return coeffs[0], coeffs[1:]  # Intercept and slope coefficients

# Example usage
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

intercept, slope = ols_regression(X, y)
print(f"Intercept: {intercept}, Slope: {slope}")
```

Slide 4: Ridge Regression: Balancing Efficiency and Regularization

Ridge regression, also known as L2 regularization, adds a penalty term to the OLS objective function. This regularization helps prevent overfitting while maintaining computational efficiency.

```python
from sklearn.linear_model import Ridge
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.rand(1000, 10)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] - 1.5 * X[:, 4] + np.random.normal(0, 0.1, 1000)

# Create and fit the Ridge model
ridge_model = Ridge(alpha=1.0)
%time ridge_model.fit(X, y)

print("Ridge Coefficients:", ridge_model.coef_)
print("Ridge Intercept:", ridge_model.intercept_)
```

Slide 5: Lasso Regression: Sparse Solutions with L1 Regularization

Lasso regression uses L1 regularization, which can lead to sparse solutions by setting some coefficients to zero. While slightly less efficient than Ridge, it's still computationally favorable for feature selection.

```python
from sklearn.linear_model import Lasso
import numpy as np

# Use the same data as in the Ridge example
np.random.seed(42)
X = np.random.rand(1000, 10)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] - 1.5 * X[:, 4] + np.random.normal(0, 0.1, 1000)

# Create and fit the Lasso model
lasso_model = Lasso(alpha=0.1)
%time lasso_model.fit(X, y)

print("Lasso Coefficients:", lasso_model.coef_)
print("Lasso Intercept:", lasso_model.intercept_)
```

Slide 6: Elastic Net: Combining L1 and L2 Regularization

Elastic Net combines the strengths of Ridge and Lasso regression, offering a balance between feature selection and regularization. It's slightly more computationally intensive but still efficient for many problems.

```python
from sklearn.linear_model import ElasticNet
import numpy as np

# Use the same data as before
np.random.seed(42)
X = np.random.rand(1000, 10)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] - 1.5 * X[:, 4] + np.random.normal(0, 0.1, 1000)

# Create and fit the Elastic Net model
elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
%time elastic_net_model.fit(X, y)

print("Elastic Net Coefficients:", elastic_net_model.coef_)
print("Elastic Net Intercept:", elastic_net_model.intercept_)
```

Slide 7: Stochastic Gradient Descent (SGD) Regression

SGD regression is highly efficient for large-scale problems. It updates the model parameters iteratively, making it suitable for online learning and big data scenarios.

```python
from sklearn.linear_model import SGDRegressor
import numpy as np

# Generate a larger dataset
np.random.seed(42)
X = np.random.rand(100000, 10)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] - 1.5 * X[:, 4] + np.random.normal(0, 0.1, 100000)

# Create and fit the SGD model
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)
%time sgd_model.fit(X, y)

print("SGD Coefficients:", sgd_model.coef_)
print("SGD Intercept:", sgd_model.intercept_)
```

Slide 8: Comparison of Model Training Times

Let's compare the training times of different regression models to highlight their computational efficiency.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
import time

# Generate a large dataset
np.random.seed(42)
X = np.random.rand(100000, 20)
y = np.sum(X[:, :5], axis=1) + np.random.normal(0, 0.1, 100000)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Elastic Net": ElasticNet(),
    "SGD": SGDRegressor(max_iter=1000, tol=1e-3)
}

for name, model in models.items():
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    print(f"{name}: {end_time - start_time:.4f} seconds")
```

Slide 9: Real-Life Example: Predicting House Prices

Let's apply our knowledge to a real-world scenario: predicting house prices based on various features such as square footage, number of bedrooms, and location.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (you would normally load this from a file)
data = pd.DataFrame({
    'area': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'bedrooms': [3, 3, 2, 4, 2, 3, 4, 4, 3, 3],
    'age': [15, 8, 10, 5, 12, 20, 2, 3, 18, 9],
    'price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
})

# Prepare the features and target
X = data[['area', 'bedrooms', 'age']]
y = data['price']

# Split the data
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
print(f"R-squared Score: {r2:.2f}")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
```

Slide 10: Real-Life Example: Predicting Crop Yield

Another practical application of efficient regression models is predicting crop yields based on environmental factors. This example demonstrates how to use Ridge regression for this purpose.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic crop yield data
np.random.seed(42)
n_samples = 1000

temperature = np.random.normal(25, 5, n_samples)
rainfall = np.random.normal(100, 30, n_samples)
soil_quality = np.random.uniform(0, 10, n_samples)
fertilizer = np.random.uniform(0, 5, n_samples)

X = np.column_stack((temperature, rainfall, soil_quality, fertilizer))
y = 20 + 0.5 * temperature + 0.2 * rainfall + 2 * soil_quality + 3 * fertilizer + np.random.normal(0, 5, n_samples)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Ridge regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Make predictions
y_pred = ridge_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
print("Coefficients:", ridge_model.coef_)
print("Intercept:", ridge_model.intercept_)

# Predict crop yield for a new sample
new_sample = np.array([[30, 120, 8, 4]])  # temperature, rainfall, soil_quality, fertilizer
predicted_yield = ridge_model.predict(new_sample)
print(f"Predicted crop yield: {predicted_yield[0]:.2f}")
```

Slide 11: Efficient Feature Selection with Lasso

Lasso regression can be used for efficient feature selection, which is crucial when dealing with high-dimensional datasets. This example demonstrates how Lasso can identify the most important features.

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate synthetic data with many features
np.random.seed(42)
n_samples, n_features = 100, 50
X = np.random.randn(n_samples, n_features)
true_coef = np.zeros(n_features)
true_coef[:5] = [1.5, -2, 0.5, -1, 1]  # Only 5 features are actually relevant
y = np.dot(X, true_coef) + np.random.normal(0, 0.1, n_samples)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Lasso model
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

# Plot coefficients
plt.figure(figsize=(10, 6))
plt.bar(range(n_features), lasso.coef_)
plt.title("Lasso Coefficients")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.tight_layout()
plt.show()

# Print non-zero coefficients
non_zero = np.abs(lasso.coef_) > 1e-5
print("Number of non-zero coefficients:", sum(non_zero))
print("Indices of non-zero coefficients:", np.where(non_zero)[0])
```

Slide 12: Online Learning with SGD Regression

SGD regression is particularly useful for online learning scenarios where data arrives in a stream. This example shows how to update the model incrementally with new data.

```python
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Initialize the model and scaler
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='constant', eta0=0.01)
scaler = StandardScaler()

# Function to generate new data
def generate_data(n_samples=100):
    X = np.random.randn(n_samples, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] - 1.5 * X[:, 4] + np.random.normal(0, 0.1, n_samples)
    return X, y

# Initial training
X, y = generate_data(1000)
X_scaled = scaler.fit_transform(X)
sgd_model.fit(X_scaled, y)

# Online learning loop
for i in range(5):
    X_new, y_new = generate_data(100)
    X_new_scaled = scaler.transform(X_new)
    
    y_pred = sgd_model.predict(X_new_scaled)
    mse = mean_squared_error(y_new, y_pred)
    print(f"Batch {i+1} - MSE before update: {mse:.4f}")
    
    sgd_model.partial_fit(X_new_scaled, y_new)
    
    y_pred_updated = sgd_model.predict(X_new_scaled)
    mse_updated = mean_squared_error(y_new, y_pred_updated)
    print(f"Batch {i+1} - MSE after update: {mse_updated:.4f}")
```

Slide 13: Efficient Model Selection with Cross-Validation

Cross-validation is crucial for selecting the best model and hyperparameters. Here's an efficient implementation using scikit-learn's built-in functions.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

# Generate a random regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# Define a range of alpha values to try
alphas = [0.1, 1.0, 10.0, 100.0]

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X, y, cv=5, scoring='neg_mean_squared_error')
    mse_scores = -scores  # Convert to positive MSE
    print(f"Alpha: {alpha}, Mean MSE: {mse_scores.mean():.4f}, Std MSE: {mse_scores.std():.4f}")

# Select the best alpha and train the final model
best_alpha = min(alphas, key=lambda a: -cross_val_score(Ridge(alpha=a), X, y, cv=5, scoring='neg_mean_squared_error').mean())
best_model = Ridge(alpha=best_alpha)
best_model.fit(X, y)

print(f"\nBest alpha: {best_alpha}")
print(f"Best model coefficients shape: {best_model.coef_.shape}")
```

Slide 14: Efficient Handling of Categorical Variables

Efficiently handling categorical variables is crucial for model performance and computational efficiency. This example demonstrates how to use one-hot encoding for categorical variables in a regression model.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Create a sample dataset with mixed numerical and categorical features
data = pd.DataFrame({
    'area': np.random.randint(50, 200, 1000),
    'rooms': np.random.randint(1, 5, 1000),
    'location': np.random.choice(['urban', 'suburban', 'rural'], 1000),
    'style': np.random.choice(['modern', 'traditional', 'contemporary'], 1000),
    'price': np.random.randint(100000, 500000, 1000)
})

# Split features and target
X = data.drop('price', axis=1)
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['area', 'rooms']),
        ('cat', OneHotEncoder(drop='first', sparse=False), ['location', 'style'])
    ])

# Create a pipeline with the preprocessor and the model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the model
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train R-squared: {train_score:.4f}")
print(f"Test R-squared: {test_score:.4f}")

# Make a prediction for a new house
new_house = pd.DataFrame({
    'area': [150],
    'rooms': [3],
    'location': ['suburban'],
    'style': ['modern']
})

predicted_price = model.predict(new_house)
print(f"Predicted price for the new house: ${predicted_price[0]:.2f}")
```

Slide 15: Additional Resources

For further exploration of efficient regression models and their implementation in Python, consider the following resources:

1. Scikit-learn Documentation: [https://scikit-learn.org/stable/modules/linear\_model.html](https://scikit-learn.org/stable/modules/linear_model.html)
2. "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani (2013): This book provides a comprehensive overview of statistical learning methods, including various regression techniques.
3. ArXiv paper: "Efficient and Robust Automated Machine Learning" by Feurer et al. (2015) ArXiv URL: [https://arxiv.org/abs/1507.05444](https://arxiv.org/abs/1507.05444)
4. ArXiv paper: "Random Features for Large-Scale Kernel Machines" by Rahimi and Recht (2007) ArXiv URL: [https://arxiv.org/abs/0702219](https://arxiv.org/abs/0702219)
5. Python Data Science Handbook by Jake VanderPlas: An excellent resource for learning about data analysis and machine learning in Python, including efficient implementations of various regression models.

These resources will help you deepen your understanding of efficient regression models and their practical applications in Python.

