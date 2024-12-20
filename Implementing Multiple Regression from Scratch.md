## Implementing Multiple Regression from Scratch
Slide 1: Multiple Regression from Scratch

This presentation explores the implementation of Multiple Regression without using Sklearn. We'll dive into the mathematics behind the Ordinary Least Squares (OLS) technique and create a custom class that performs the same task as Sklearn's built-in functionality.

```python
import numpy as np
import matplotlib.pyplot as plt

class MultipleRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        # Add a column of ones to X for the intercept term
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        
        # Calculate coefficients using OLS formula
        self.coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        
        # Extract intercept and coefficients
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        return X @ self.coefficients + self.intercept
```

Slide 2: Understanding Ordinary Least Squares (OLS)

Ordinary Least Squares is a method for estimating the parameters in a linear regression model. It minimizes the sum of the squares of the differences between the observed dependent variable and the predicted dependent variable.

```python
def visualize_ols_2d(X, y):
    plt.scatter(X, y, color='blue', label='Data points')
    
    # Calculate the OLS line
    X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
    coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    
    # Plot the OLS line
    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred = coefficients[0] + coefficients[1] * x_range
    plt.plot(x_range, y_pred, color='red', label='OLS line')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('Ordinary Least Squares Visualization')
    plt.show()

# Example usage
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)
visualize_ols_2d(X, y)
```

Slide 3: Mathematics Behind Multiple Regression

The core of multiple regression lies in the OLS formula: β = (X^T X)^(-1) X^T y

Where: β: Vector of coefficients X: Matrix of independent variables y: Vector of dependent variable

```python
def ols_formula(X, y):
    X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
    beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    return beta

# Example usage
X = np.random.rand(100, 3)  # 3 independent variables
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 1 + np.random.randn(100)
coefficients = ols_formula(X, y)
print("Intercept:", coefficients[0])
print("Coefficients:", coefficients[1:])
```

Slide 4: Implementing the Fit Method

The fit method calculates the coefficients and intercept using the OLS formula. We'll break down the implementation step by step.

```python
class MultipleRegression:
    def fit(self, X, y):
        # Step 1: Add a column of ones to X for the intercept term
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        
        # Step 2: Calculate coefficients using OLS formula
        self.coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        
        # Step 3: Extract intercept and coefficients
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

# Example usage
model = MultipleRegression()
X = np.random.rand(100, 3)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 1 + np.random.randn(100)
model.fit(X, y)
print("Intercept:", model.intercept)
print("Coefficients:", model.coefficients)
```

Slide 5: Implementing the Predict Method

The predict method uses the calculated coefficients and intercept to make predictions on new data.

```python
class MultipleRegression:
    def predict(self, X):
        return X @ self.coefficients + self.intercept

# Example usage
model = MultipleRegression()
X_train = np.random.rand(100, 3)
y_train = 2 * X_train[:, 0] + 3 * X_train[:, 1] - X_train[:, 2] + 1 + np.random.randn(100)
model.fit(X_train, y_train)

X_test = np.random.rand(20, 3)
predictions = model.predict(X_test)
print("Predictions:", predictions)
```

Slide 6: Comparing Custom Implementation with Sklearn

Let's compare our custom implementation with Sklearn's LinearRegression to validate our results.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 3)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 1 + np.random.randn(100)

# Custom implementation
custom_model = MultipleRegression()
custom_model.fit(X, y)

# Sklearn implementation
sklearn_model = LinearRegression()
sklearn_model.fit(X, y)

print("Custom Model:")
print("Intercept:", custom_model.intercept)
print("Coefficients:", custom_model.coefficients)

print("\nSklearn Model:")
print("Intercept:", sklearn_model.intercept_)
print("Coefficients:", sklearn_model.coef_)
```

Slide 7: Evaluating Model Performance

To assess the performance of our custom implementation, we'll calculate the Mean Squared Error (MSE) and R-squared (R²) score.

```python
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Generate sample data
X = np.random.rand(100, 3)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 1 + np.random.randn(100)

# Fit and predict using custom model
custom_model = MultipleRegression()
custom_model.fit(X, y)
y_pred_custom = custom_model.predict(X)

# Calculate performance metrics
mse = mean_squared_error(y, y_pred_custom)
r2 = r_squared(y, y_pred_custom)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)
```

Slide 8: Handling Multicollinearity

Multicollinearity occurs when independent variables are highly correlated. Our implementation doesn't address this issue, which can lead to unstable coefficient estimates.

```python
def detect_multicollinearity(X, threshold=10):
    # Calculate correlation matrix
    corr_matrix = np.abs(np.corrcoef(X.T))
    
    # Find highly correlated features
    high_corr = np.where(np.triu(corr_matrix, k=1) > threshold)
    
    if len(high_corr[0]) > 0:
        print("Warning: Multicollinearity detected between features:")
        for i, j in zip(high_corr[0], high_corr[1]):
            print(f"Features {i} and {j} have correlation: {corr_matrix[i, j]:.2f}")
    else:
        print("No severe multicollinearity detected.")

# Example usage
X = np.random.rand(100, 4)
X[:, 3] = X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, 100)  # Create a linearly dependent feature
detect_multicollinearity(X, threshold=0.9)
```

Slide 9: Feature Scaling

Feature scaling is important for multiple regression, especially when features have different units or ranges. Let's implement standardization.

```python
class MultipleRegression:
    def __init__(self, standardize=True):
        self.coefficients = None
        self.intercept = None
        self.standardize = standardize
        self.mean = None
        self.std = None

    def fit(self, X, y):
        if self.standardize:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X_scaled = (X - self.mean) / self.std
        else:
            X_scaled = X
        
        X_with_intercept = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))
        self.coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        if self.standardize:
            X_scaled = (X - self.mean) / self.std
        else:
            X_scaled = X
        return X_scaled @ self.coefficients + self.intercept

# Example usage
X = np.random.rand(100, 3) * 100  # Features with different scales
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 1 + np.random.randn(100)

model = MultipleRegression(standardize=True)
model.fit(X, y)
print("Coefficients:", model.coefficients)
print("Intercept:", model.intercept)
```

Slide 10: Handling Categorical Variables

Our implementation assumes all features are numerical. Let's add support for categorical variables using one-hot encoding.

```python
import pandas as pd

def one_hot_encode(X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    categorical_columns = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    return X_encoded.values

# Example usage
data = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.choice(['A', 'B', 'C'], 100),
    'feature3': np.random.randint(0, 5, 100)
})

X_encoded = one_hot_encode(data)
print("Shape before encoding:", data.shape)
print("Shape after encoding:", X_encoded.shape)
print("Encoded features:")
print(pd.DataFrame(X_encoded).head())
```

Slide 11: Real-Life Example: Predicting House Prices

Let's apply our custom Multiple Regression model to predict house prices based on various features.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Boston Housing dataset
from sklearn.datasets import load_boston
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the custom model
model = MultipleRegression(standardize=False)  # We've already scaled the data
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r_squared(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Display the top 5 most important features
feature_importance = pd.DataFrame({'feature': boston.feature_names, 'coefficient': model.coefficients})
print(feature_importance.sort_values('coefficient', key=abs, ascending=False).head())
```

Slide 12: Real-Life Example: Predicting Fuel Efficiency

Let's use our custom Multiple Regression model to predict the fuel efficiency of cars based on various characteristics.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Auto MPG dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
df = pd.read_csv(url, names=column_names, delim_whitespace=True, na_values="?")

# Drop rows with missing values and the 'car_name' column
df = df.dropna().drop("car_name", axis=1)

# Prepare the features and target
X = df.drop("mpg", axis=1)
y = df["mpg"]

# Encode categorical variables
X = pd.get_dummies(X, columns=["origin"], drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the custom model
model = MultipleRegression(standardize=False)  # We've already scaled the data
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r_squared(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Display the top 5 most important features
feature_importance = pd.DataFrame({'feature': X.columns, 'coefficient': model.coefficients})
print(feature_importance.sort_values('coefficient', key=abs, ascending=False).head())
```

Slide 13: Limitations and Considerations

While our custom implementation works well for many cases, it's important to consider its limitations:

1. It assumes a linear relationship between features and target.
2. It's sensitive to outliers.
3. It doesn't handle multicollinearity well.
4. It may not perform well with high-dimensional data.

For more complex scenarios, consider using regularization techniques like Ridge or Lasso regression, or exploring non-linear models.

Slide 14: Limitations and Considerations

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_limitations():
    # Generate non-linear data
    X = np.linspace(0, 10, 100)
    y = np.sin(X) + np.random.normal(0, 0.1, 100)
    
    # Fit linear regression
    coeffs = np.polyfit(X, y, 1)
    y_pred = np.polyval(coeffs, X)
    
    # Plot
    plt.scatter(X, y, label='Data')
    plt.plot(X, y_pred, color='red', label='Linear Fit')
    plt.title('Linear Regression on Non-Linear Data')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

plot_limitations()
```

Slide 15: Addressing Multicollinearity: Ridge Regression

Ridge regression helps mitigate multicollinearity by adding a penalty term to the OLS objective function. This regularization technique can lead to more stable coefficient estimates.

```python
class RidgeRegression(MultipleRegression):
    def __init__(self, alpha=1.0, standardize=True):
        super().__init__(standardize)
        self.alpha = alpha

    def fit(self, X, y):
        if self.standardize:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X_scaled = (X - self.mean) / self.std
        else:
            X_scaled = X
        
        X_with_intercept = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))
        identity = np.eye(X_with_intercept.shape[1])
        identity[0, 0] = 0  # Don't regularize the intercept
        
        self.coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept + self.alpha * identity) @ X_with_intercept.T @ y
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

# Example usage
X = np.random.rand(100, 3)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 1 + np.random.randn(100)

ridge_model = RidgeRegression(alpha=0.1)
ridge_model.fit(X, y)
print("Ridge Regression Coefficients:", ridge_model.coefficients)
print("Ridge Regression Intercept:", ridge_model.intercept)
```

Slide 16: Handling Non-Linearity: Polynomial Regression

When the relationship between features and target is non-linear, we can use polynomial regression to capture more complex patterns.

```python
from sklearn.preprocessing import PolynomialFeatures

def polynomial_regression(X, y, degree=2):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    
    model = MultipleRegression()
    model.fit(X_poly, y)
    
    return model, poly

# Example usage
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = 1 + 2*X + 3*X**2 + np.random.randn(100, 1)

model, poly = polynomial_regression(X, y, degree=2)

X_test = np.linspace(-6, 6, 200).reshape(-1, 1)
X_test_poly = poly.transform(X_test)
y_pred = model.predict(X_test_poly)

plt.scatter(X, y, label='Data')
plt.plot(X_test, y_pred, color='red', label='Polynomial Fit')
plt.title('Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

Slide 17: Additional Resources

For those interested in diving deeper into regression techniques and their implementations, here are some valuable resources:

1. "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman ArXiv: [https://arxiv.org/abs/1501.06996](https://arxiv.org/abs/1501.06996)
2. "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani ArXiv: [https://arxiv.org/abs/1315.2737](https://arxiv.org/abs/1315.2737)
3. "Pattern Recognition and Machine Learning" by Christopher Bishop

These resources provide in-depth explanations of various regression techniques, their mathematical foundations, and practical implementations.

