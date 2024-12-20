## Building an OLS Regression Algorithm in Python
Slide 1: Introduction to Ordinary Least Squares Regression

Ordinary Least Squares (OLS) regression is a fundamental statistical method used to model the relationship between a dependent variable and one or more independent variables. This slideshow will guide you through the process of building an OLS regression algorithm from scratch using Python, providing you with a deeper understanding of the underlying mathematics and implementation details.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1) * 0.1

# Plot the data
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Sample Data for OLS Regression')
plt.show()
```

Slide 2: The Linear Regression Model

The linear regression model assumes a linear relationship between the independent variable(s) X and the dependent variable y. For simple linear regression with one independent variable, the model is expressed as:

y = β₀ + β₁X + ε

Where β₀ is the y-intercept, β₁ is the slope, and ε is the error term.

```python
def linear_model(X, beta):
    return X.dot(beta)

# Example usage
X = np.array([[1, 2], [1, 3], [1, 4]])  # Add column of 1s for intercept
beta = np.array([1, 2])  # [intercept, slope]
y_pred = linear_model(X, beta)
print("Predicted y values:", y_pred)
```

Slide 3: The Cost Function

The cost function measures the difference between the predicted values and the actual values. In OLS regression, we use the Mean Squared Error (MSE) as the cost function:

J(β) = (1/2m) \* Σ(ŷᵢ - yᵢ)²

Where m is the number of samples, ŷᵢ is the predicted value, and yᵢ is the actual value.

```python
def cost_function(X, y, beta):
    m = len(y)
    predictions = linear_model(X, beta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Example usage
X = np.array([[1, 2], [1, 3], [1, 4]])
y = np.array([2, 4, 5])
beta = np.array([0.5, 1.5])
print("Cost:", cost_function(X, y, beta))
```

Slide 4: Gradient Descent

Gradient descent is an optimization algorithm used to find the values of β that minimize the cost function. It iteratively updates the parameters in the direction of steepest descent of the cost function.

The update rule for gradient descent is: β = β - α \* ∇J(β)

Where α is the learning rate and ∇J(β) is the gradient of the cost function with respect to β.

```python
def gradient_descent(X, y, beta, alpha, num_iterations):
    m = len(y)
    for _ in range(num_iterations):
        predictions = linear_model(X, beta)
        gradient = (1 / m) * X.T.dot(predictions - y)
        beta -= alpha * gradient
    return beta

# Example usage
X = np.array([[1, 2], [1, 3], [1, 4]])
y = np.array([2, 4, 5])
beta_initial = np.array([0, 0])
alpha = 0.01
num_iterations = 1000
beta_optimal = gradient_descent(X, y, beta_initial, alpha, num_iterations)
print("Optimal beta:", beta_optimal)
```

Slide 5: Implementing OLS Regression Class

Let's create a Python class that encapsulates the OLS regression algorithm, including methods for fitting the model and making predictions.

```python
class OLSRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta = None

    def fit(self, X, y):
        # Add column of 1s for intercept
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.beta = np.zeros(X_b.shape[1])
        self.beta = gradient_descent(X_b, y, self.beta, self.learning_rate, self.num_iterations)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return linear_model(X_b, self.beta)

# Example usage
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 5, 4])
model = OLSRegression()
model.fit(X, y)
print("Optimal beta:", model.beta)
print("Predictions:", model.predict(X))
```

Slide 6: Real-Life Example: Predicting Plant Growth

Let's apply our OLS regression model to predict plant growth based on the amount of sunlight received. We'll use a small dataset of plant height measurements and corresponding hours of sunlight exposure.

```python
# Generate sample data
np.random.seed(42)
sunlight_hours = np.random.uniform(2, 8, 50)
plant_height = 5 + 2 * sunlight_hours + np.random.normal(0, 1, 50)

# Prepare data for OLS regression
X = sunlight_hours.reshape(-1, 1)
y = plant_height

# Create and fit the model
model = OLSRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X, y)

# Make predictions
X_test = np.array([[3], [5], [7]])
predictions = model.predict(X_test)

# Plot results
plt.scatter(X, y, label='Actual data')
plt.plot(X_test, predictions, color='red', label='Predictions')
plt.xlabel('Sunlight hours')
plt.ylabel('Plant height (cm)')
plt.title('Plant Growth Prediction')
plt.legend()
plt.show()

print("Predicted plant heights:")
for hours, height in zip(X_test, predictions):
    print(f"{hours[0]} hours of sunlight: {height[0]:.2f} cm")
```

Slide 7: Evaluating Model Performance

To assess the performance of our OLS regression model, we can use metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²) score.

```python
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Calculate performance metrics
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r_squared(y, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared Score: {r2:.4f}")
```

Slide 8: Handling Multiple Features

Our OLS regression implementation can handle multiple features. Let's extend our plant growth example to include temperature as an additional feature.

```python
# Generate sample data with two features
np.random.seed(42)
sunlight_hours = np.random.uniform(2, 8, 50)
temperature = np.random.uniform(15, 30, 50)
plant_height = 2 + 1.5 * sunlight_hours + 0.5 * temperature + np.random.normal(0, 1, 50)

# Prepare data for OLS regression
X = np.column_stack((sunlight_hours, temperature))
y = plant_height

# Create and fit the model
model = OLSRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X, y)

# Make predictions
X_test = np.array([[4, 20], [6, 25], [8, 30]])
predictions = model.predict(X_test)

print("Predicted plant heights:")
for features, height in zip(X_test, predictions):
    print(f"Sunlight: {features[0]} hours, Temperature: {features[1]}°C: {height[0]:.2f} cm")

# Plot results (3D scatter plot)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c='b', marker='o', label='Actual data')
ax.scatter(X_test[:, 0], X_test[:, 1], predictions, c='r', marker='^', label='Predictions')
ax.set_xlabel('Sunlight hours')
ax.set_ylabel('Temperature (°C)')
ax.set_zlabel('Plant height (cm)')
ax.legend()
plt.title('Plant Growth Prediction (Multiple Features)')
plt.show()
```

Slide 9: Feature Scaling

When dealing with multiple features that have different scales, it's important to normalize the features to ensure that the gradient descent algorithm converges properly. Let's implement feature scaling using standardization (z-score normalization).

```python
class OLSRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta = None
        self.mean = None
        self.std = None

    def normalize_features(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / self.std

    def fit(self, X, y):
        X_normalized = self.normalize_features(X)
        X_b = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]
        self.beta = np.zeros(X_b.shape[1])
        self.beta = gradient_descent(X_b, y, self.beta, self.learning_rate, self.num_iterations)

    def predict(self, X):
        X_normalized = (X - self.mean) / self.std
        X_b = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]
        return linear_model(X_b, self.beta)

# Example usage with the plant growth dataset
model = OLSRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X, y)
predictions = model.predict(X_test)

print("Predicted plant heights (with feature scaling):")
for features, height in zip(X_test, predictions):
    print(f"Sunlight: {features[0]} hours, Temperature: {features[1]}°C: {height[0]:.2f} cm")
```

Slide 10: Regularization: Ridge Regression

To prevent overfitting, we can add regularization to our OLS regression model. Ridge regression (L2 regularization) adds a penalty term to the cost function based on the magnitude of the coefficients.

```python
class RidgeRegression(OLSRegression):
    def __init__(self, learning_rate=0.01, num_iterations=1000, alpha=1.0):
        super().__init__(learning_rate, num_iterations)
        self.alpha = alpha

    def fit(self, X, y):
        X_normalized = self.normalize_features(X)
        X_b = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]
        self.beta = np.zeros(X_b.shape[1])
        m = len(y)
        
        for _ in range(self.num_iterations):
            predictions = linear_model(X_b, self.beta)
            gradient = (1 / m) * X_b.T.dot(predictions - y) + (self.alpha / m) * self.beta
            self.beta -= self.learning_rate * gradient

# Example usage
ridge_model = RidgeRegression(learning_rate=0.01, num_iterations=1000, alpha=0.1)
ridge_model.fit(X, y)
ridge_predictions = ridge_model.predict(X_test)

print("Ridge Regression Predictions:")
for features, height in zip(X_test, ridge_predictions):
    print(f"Sunlight: {features[0]} hours, Temperature: {features[1]}°C: {height[0]:.2f} cm")
```

Slide 11: Cross-Validation

To ensure our model generalizes well to unseen data, we can implement k-fold cross-validation. This technique helps us estimate the model's performance on different subsets of the data.

```python
def cross_validation(X, y, k=5):
    fold_size = len(X) // k
    mse_scores = []

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        X_test = X[start:end]
        y_test = y[start:end]
        X_train = np.concatenate([X[:start], X[end:]])
        y_train = np.concatenate([y[:start], y[end:]])

        model = OLSRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

    return np.mean(mse_scores), np.std(mse_scores)

# Perform cross-validation
mean_mse, std_mse = cross_validation(X, y)
print(f"Cross-validation MSE: {mean_mse:.4f} (+/- {std_mse:.4f})")
```

Slide 12: Real-Life Example: Predicting Housing Prices

Let's apply our OLS regression model to predict housing prices based on various features such as square footage, number of bedrooms, and age of the house.

```python
# Generate sample data
np.random.seed(42)
num_samples = 100
square_footage = np.random.uniform(1000, 3000, num_samples)
num_bedrooms = np.random.randint(1, 6, num_samples)
house_age = np.random.uniform(0, 50, num_samples)
house_price = (
    150000 +
    100 * square_footage +
    20000 * num_bedrooms -
    1000 * house_age +
    np.random.normal(0, 25000, num_samples)
)

# Prepare data for OLS regression
X = np.column_stack((square_footage, num_bedrooms, house_age))
y = house_price

# Create and fit the model
model = OLSRegression(learning_rate=0.00001, num_iterations=10000)
model.fit(X, y)

# Make predictions
X_test = np.array([
    [2000, 3, 10],
    [2500, 4, 5],
    [1800, 2, 20]
])
predictions = model.predict(X_test)

print("Predicted house prices:")
for features, price in zip(X_test, predictions):
    print(f"Square footage: {features[0]:.0f}, Bedrooms: {features[1]}, Age: {features[2]:.0f} years: ${price[0]:,.2f}")

# Evaluate model performance
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r_squared(y, y_pred)

print(f"Mean Squared Error: ${mse:,.2f}")
print(f"R-squared Score: {r2:.4f}")

# Visualize the relationship between square footage and house price
plt.scatter(X[:, 0], y, alpha=0.5)
plt.plot(X[:, 0], y_pred, color='red', linewidth=2)
plt.xlabel('Square Footage')
plt.ylabel('House Price')
plt.title('House Price vs. Square Footage')
plt.show()
```

Slide 13: Interpreting the Coefficients

Understanding the coefficients (beta values) of our OLS regression model is crucial for interpreting the impact of each feature on the target variable.

```python
feature_names = ['Intercept', 'Square Footage', 'Number of Bedrooms', 'House Age']
coefficients = model.beta

print("Model Coefficients:")
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.2f}")

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_names[1:], np.abs(coefficients[1:]))
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Slide 14: Assumptions and Limitations of OLS Regression

OLS regression makes several assumptions about the data:

1. Linearity: The relationship between features and target is linear.
2. Independence: Observations are independent of each other.
3. Homoscedasticity: Constant variance of residuals.
4. Normality: Residuals are normally distributed.
5. No multicollinearity: Features are not highly correlated.

```python
# Check for normality of residuals
residuals = y - y_pred
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30)
plt.title('Distribution of Residuals')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()

# Check for homoscedasticity
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into OLS regression and its implementation in Python, here are some valuable resources:

1. "Introduction to Linear Regression Analysis" by Montgomery, Peck, and Vining - A comprehensive textbook on regression analysis.
2. "Pattern Recognition and Machine Learning" by Christopher Bishop - Chapter 3 covers linear regression models in depth.
3. ArXiv paper: "A Tutorial on Ridge Regression" by Jan-Willem Bikker URL: [https://arxiv.org/abs/2006.10645](https://arxiv.org/abs/2006.10645)
4. ArXiv paper: "A Comprehensive Survey of Regression Based Loss Functions" by Sébastien Loustau URL: [https://arxiv.org/abs/2103.15277](https://arxiv.org/abs/2103.15277)

These resources provide a more in-depth understanding of the mathematical foundations and advanced techniques related to OLS regression and its variants.

