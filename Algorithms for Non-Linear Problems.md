## Algorithms for Non-Linear Problems
Slide 1: Introduction to Linear Regression Underfitting

Linear regression is the most basic and commonly used algorithm that tends to underfit when dealing with non-linear problems. It assumes a linear relationship between features and target variables, making it inherently limited in capturing complex patterns in the data.

```python
# Simple linear regression implementation
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        # Normal equation implementation
        X_b = np.c_[np.ones((n_samples, 1)), X]
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.bias = theta[0]
        self.weights = theta[1:]
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

Slide 2: Demonstrating Underfitting with Non-linear Data

In this example, we'll generate non-linear data and attempt to fit it using linear regression, clearly showing how the model underfits by failing to capture the underlying curved pattern in the data.

```python
# Generate non-linear data
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X**2 + np.random.normal(0, 0.1, (100, 1))

# Fit linear regression
model = LinearRegression()
model.fit(X, y.ravel())

# Plot results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Linear Prediction')
plt.title('Linear Regression Underfitting Example')
plt.legend()
plt.show()
```

Slide 3: Understanding Model Complexity and Bias

Model complexity directly relates to underfitting, where simpler models like linear regression have high bias. The mathematical representation of linear regression's hypothesis function demonstrates its limited capacity to model non-linear relationships.

```python
# Mathematical representation of linear regression
"""
Linear Regression Formula:
$$y = wx + b$$

Loss Function (Mean Squared Error):
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - (wx_i + b))^2$$
"""

# Visualization of model complexity
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y_true = 0.5 * X**2
y_pred = model.predict(X)
bias = np.mean((y_true - y_pred)**2)
print(f"Model Bias: {bias:.4f}")
```

Slide 4: Polynomial Features Transformation

To overcome underfitting in linear regression, we can transform the input features using polynomial expansion. This allows the linear model to capture non-linear patterns while maintaining its simple learning algorithm.

```python
from sklearn.preprocessing import PolynomialFeatures

# Transform features to polynomial
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Fit linear regression with polynomial features
model_poly = LinearRegression()
model_poly.fit(X_poly, y)

# Compare predictions
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model_poly.predict(X_poly), color='green', label='Polynomial Prediction')
plt.plot(X, model.predict(X), color='red', label='Linear Prediction')
plt.legend()
plt.show()
```

Slide 5: Implementing Cross-Validation for Model Selection

Cross-validation helps identify underfitting by comparing model performance across different data splits. This systematic approach reveals whether the model's simplicity is causing poor generalization.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression as SklearnLR

def evaluate_complexity(X, y, max_degree=5):
    scores = []
    degrees = range(1, max_degree + 1)
    
    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = SklearnLR()
        cv_scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
        scores.append(-cv_scores.mean())
    
    plt.plot(degrees, scores, marker='o')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('Model Complexity vs. Error')
    plt.show()
```

Slide 6: Real-world Example: Housing Price Prediction

A common real-world scenario where underfitting occurs is in housing price prediction when using only linear features. This example demonstrates how linear regression fails to capture the complex relationships between house features and prices.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic housing data
np.random.seed(42)
n_samples = 1000
size = np.random.normal(1500, 300, n_samples)
age = np.random.uniform(0, 50, n_samples)
price = 100000 + 200 * size + 50 * age**1.5 + np.random.normal(0, 10000, n_samples)

# Create dataset
housing_data = pd.DataFrame({
    'size': size,
    'age': age,
    'price': price
})

# Prepare data
X = housing_data[['size', 'age']]
y = housing_data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train linear model
model = LinearRegression()
model.fit(X_train, y_train)
print(f"R² Score: {model.score(X_test, y_test):.4f}")
```

Slide 7: Incorporating Non-linear Features in Housing Price Model

Extending the previous example, we'll add polynomial features and interaction terms to capture non-linear relationships in the housing price prediction model, demonstrating how to overcome underfitting.

```python
# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Train model with polynomial features
model_poly = LinearRegression()
model_poly.fit(X_poly_train, y_train)

# Compare results
print("Linear Model R² Score:", model.score(X_test, y_test))
print("Polynomial Model R² Score:", model_poly.score(X_poly_test, y_test))

# Feature importance analysis
feature_names = poly.get_feature_names_out(['size', 'age'])
coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model_poly.coef_
})
print("\nTop 5 most important features:")
print(coefficients.nlargest(5, 'Coefficient'))
```

Slide 8: Time Series Prediction with Linear Models

Time series forecasting is another domain where linear models often underfit due to the inherent non-linear nature of temporal patterns. This example shows how linear regression struggles with seasonal data.

```python
# Generate seasonal time series data
t = np.linspace(0, 4*np.pi, 200)
seasonal_pattern = np.sin(t) + 0.5*np.sin(2*t)
trend = 0.03 * t
noise = np.random.normal(0, 0.1, 200)
y = seasonal_pattern + trend + noise

# Prepare features
X = t.reshape(-1, 1)
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]

# Fit linear model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X, linear_model.predict(X), color='red', label='Linear Prediction')
plt.legend()
plt.title('Time Series Prediction with Linear Regression')
plt.show()
```

Slide 9: Decision Boundaries in Classification Problems

When using linear regression for classification tasks, underfitting becomes evident through oversimplified decision boundaries that fail to separate non-linearly separable classes effectively.

```python
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression

# Generate non-linear classification data
X, y = make_moons(n_samples=200, noise=0.15)

# Train logistic regression
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Plot decision boundary
def plot_decision_boundary(model, X, y):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title('Linear Decision Boundary (Underfitting)')
    plt.show()

plot_decision_boundary(log_reg, X, y)
```

Slide 10: Feature Engineering to Combat Underfitting

Feature engineering plays a crucial role in mitigating underfitting by creating meaningful transformations that capture non-linear relationships. This example demonstrates various feature engineering techniques to improve model performance.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

class FeatureEngineer:
    def __init__(self, data):
        self.data = data
        
    def create_interaction_terms(self):
        features = self.data.copy()
        for i in range(features.shape[1]):
            for j in range(i+1, features.shape[1]):
                features = np.column_stack((
                    features, 
                    self.data[:, i] * self.data[:, j]
                ))
        return features
    
    def create_polynomial_features(self, degree=2):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        return poly.fit_transform(self.data)
    
    def create_trigonometric_features(self):
        return np.column_stack((
            self.data,
            np.sin(self.data),
            np.cos(self.data)
        ))

# Example usage
X = np.random.randn(100, 2)
engineer = FeatureEngineer(X)
X_enhanced = engineer.create_polynomial_features()
print(f"Original features: {X.shape[1]}")
print(f"Enhanced features: {X_enhanced.shape[1]}")
```

Slide 11: Quantifying Underfitting with Learning Curves

Learning curves provide a visual diagnosis of underfitting by showing how model performance changes with increasing training data. A large gap between training and validation scores indicates underfitting.

```python
from sklearn.model_selection import learning_curve

def plot_learning_curves(estimator, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, val_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Example usage
model = LinearRegression()
plot_learning_curves(model, X, y)
```

Slide 12: Real-world Example: Stock Price Prediction

Stock price prediction represents a complex non-linear problem where linear models typically underfit. This implementation shows how to enhance the model with technical indicators and non-linear features.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StockPricePredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = LinearRegression()
        
    def create_technical_indicators(self, prices):
        df = pd.DataFrame(prices, columns=['price'])
        # Moving averages
        df['MA5'] = df['price'].rolling(window=5).mean()
        df['MA20'] = df['price'].rolling(window=20).mean()
        # Relative Strength Index
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df.dropna()
    
    def fit(self, prices, target):
        features_df = self.create_technical_indicators(prices)
        X = self.scaler.fit_transform(features_df)
        self.model.fit(X, target)
        
    def predict(self, prices):
        features_df = self.create_technical_indicators(prices)
        X = self.scaler.transform(features_df)
        return self.model.predict(X)

# Example usage
prices = np.random.randn(1000).cumsum() + 100
target = np.roll(prices, -1)[:-1]  # Next day's price
predictor = StockPricePredictor()
predictor.fit(prices[:-1], target)
```

Slide 13: Bias-Variance Decomposition Analysis

Bias-variance decomposition helps quantify the extent of underfitting by measuring the model's systematic error (bias) against its sensitivity to training data variations (variance). This implementation demonstrates how to calculate these components.

```python
def bias_variance_decomposition(model, X, y, n_iterations=100):
    # Generate multiple training sets
    predictions = np.zeros((n_iterations, len(X)))
    for i in range(n_iterations):
        # Bootstrap sample
        indices = np.random.randint(0, len(X), len(X))
        X_train, y_train = X[indices], y[indices]
        
        # Train model and predict
        model.fit(X_train, y_train)
        predictions[i] = model.predict(X)
    
    # Calculate components
    expected_pred = np.mean(predictions, axis=0)
    bias = np.mean((y - expected_pred) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    total_error = bias + variance
    
    print(f"Bias: {bias:.4f}")
    print(f"Variance: {variance:.4f}")
    print(f"Total Error: {total_error:.4f}")
    
    return bias, variance, total_error

# Example usage
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X**2 + np.random.normal(0, 0.1, (100, 1)).ravel()
model = LinearRegression()
bias, variance, total_error = bias_variance_decomposition(model, X, y)
```

Slide 14: Model Complexity vs. Error Analysis

This implementation creates a framework for analyzing how model complexity affects training and validation errors, helping identify the optimal model complexity to avoid underfitting.

```python
class ModelComplexityAnalyzer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def evaluate_complexity(self, max_degree=10):
        train_errors = []
        val_errors = []
        complexities = range(1, max_degree + 1)
        
        for degree in complexities:
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(self.X)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_poly, self.y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Calculate errors
            train_error = mean_squared_error(
                y_train, model.predict(X_train)
            )
            val_error = mean_squared_error(
                y_val, model.predict(X_val)
            )
            
            train_errors.append(train_error)
            val_errors.append(val_error)
        
        return complexities, train_errors, val_errors

# Visualize results
analyzer = ModelComplexityAnalyzer(X, y)
complexities, train_errors, val_errors = analyzer.evaluate_complexity()

plt.plot(complexities, train_errors, label='Training Error')
plt.plot(complexities, val_errors, label='Validation Error')
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Model Complexity vs. Error Analysis')
plt.show()
```

Slide 15: Additional Resources

*   Understanding Underfitting and Overfitting with Bias-Variance Decomposition
    *   [https://arxiv.org/abs/1912.08265](https://arxiv.org/abs/1912.08265)
*   A Comprehensive Analysis of Linear Regression for Machine Learning
    *   [https://arxiv.org/abs/2001.07115](https://arxiv.org/abs/2001.07115)
*   Feature Engineering Strategies for Improving Linear Model Performance
    *   [https://machinelearning.org/feature-engineering-techniques](https://machinelearning.org/feature-engineering-techniques)
*   Practical Guidelines for Avoiding Underfitting in Regression Problems
    *   [https://dl.acm.org/doi/10.1145/3534678](https://dl.acm.org/doi/10.1145/3534678)
*   Modern Approaches to Model Selection and Complexity Analysis
    *   [https://www.sciencedirect.com/topics/computer-science/model-complexity](https://www.sciencedirect.com/topics/computer-science/model-complexity)

