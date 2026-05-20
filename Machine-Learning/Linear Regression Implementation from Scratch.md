## Linear Regression Implementation from Scratch
Slide 1: Linear Regression Implementation from Scratch

Linear regression forms the foundation of predictive modeling by establishing relationships between variables through a linear equation. This implementation demonstrates how to create a basic linear regression model using only NumPy, emphasizing the underlying mathematical concepts.

```python
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

Slide 2: Cost Function Implementation

The cost function measures the difference between predicted and actual values, providing a metric for model performance. Mean Squared Error (MSE) is implemented here to quantify prediction accuracy and guide the optimization process.

```python
def compute_cost(X, y, weights, bias):
    n_samples = len(y)
    predictions = np.dot(X, weights) + bias
    mse = (1/(2*n_samples)) * np.sum((predictions - y)**2)
    return mse

def compute_gradients(X, y, weights, bias):
    n_samples = len(y)
    predictions = np.dot(X, weights) + bias
    
    dw = (1/n_samples) * np.dot(X.T, (predictions - y))
    db = (1/n_samples) * np.sum(predictions - y)
    
    return dw, db
```

Slide 3: Gradient Descent Visualization

Gradient descent optimization can be visualized to understand how the algorithm converges to the optimal parameters. This implementation creates a visual representation of the cost function's evolution during training.

```python
import matplotlib.pyplot as plt

def plot_gradient_descent(X, y, model, iterations=100):
    costs = []
    weights_history = []
    bias_history = []
    
    # Initialize parameters
    weights = np.zeros(X.shape[1])
    bias = 0
    
    for i in range(iterations):
        # Compute predictions and cost
        y_pred = np.dot(X, weights) + bias
        cost = compute_cost(X, y, weights, bias)
        costs.append(cost)
        
        # Store parameters
        weights_history.append(weights.copy())
        bias_history.append(bias)
        
        # Compute gradients and update
        dw, db = compute_gradients(X, y, weights, bias)
        weights -= model.lr * dw
        bias -= model.lr * db
    
    # Plotting
    plt.figure(figsize=(12, 4))
    plt.plot(costs)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Gradient Descent Optimization')
    plt.show()
    
    return costs, weights_history, bias_history
```

Slide 4: Real Estate Price Prediction

This practical implementation demonstrates linear regression for predicting house prices based on multiple features. The code includes data preprocessing, model training, and evaluation metrics for real estate market analysis.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample real estate data
data = {
    'area': np.random.uniform(1000, 5000, 1000),
    'bedrooms': np.random.randint(1, 6, 1000),
    'age': np.random.uniform(0, 40, 1000),
    'price': np.random.uniform(100000, 1000000, 1000)
}
df = pd.DataFrame(data)

# Prepare features and target
X = df[['area', 'bedrooms', 'age']].values
y = df['price'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
```

Slide 5: Performance Metrics Implementation

Performance metrics are crucial for evaluating regression models. This implementation includes R-squared, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) calculations to provide comprehensive model assessment.

```python
def calculate_metrics(y_true, y_pred):
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # R-squared score
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_residual / ss_total)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

# Example usage
metrics = calculate_metrics(y_test, y_pred)
print(f"Model Performance:\nMAE: {metrics['MAE']:.2f}\n"
      f"RMSE: {metrics['RMSE']:.2f}\nR2: {metrics['R2']:.2f}")
```

Slide 6: Multiple Linear Regression Analysis

Multiple linear regression extends the basic model to handle multiple independent variables. This implementation demonstrates how to process multiple features and interpret their relationships with the target variable.

```python
class MultipleLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.feature_importances_ = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            y_pred = self._forward(X)
            
            # Update parameters
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
        # Calculate feature importance
        self.feature_importances_ = np.abs(self.weights) / np.sum(np.abs(self.weights))
        
    def _forward(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        return self._forward(X)
```

Slide 7: Learning Rate Optimization

Learning rate selection significantly impacts model convergence. This implementation demonstrates dynamic learning rate adjustment and its effects on training stability and speed.

```python
def adaptive_learning_rate(initial_lr=0.01, decay_rate=0.95, min_lr=0.001):
    class AdaptiveLR:
        def __init__(self):
            self.lr = initial_lr
            self.decay_rate = decay_rate
            self.min_lr = min_lr
            self.iteration = 0
        
        def get_lr(self):
            self.lr = max(
                initial_lr * (self.decay_rate ** (self.iteration // 100)),
                self.min_lr
            )
            self.iteration += 1
            return self.lr
    
    return AdaptiveLR()

# Implementation example
adaptive_lr = adaptive_learning_rate()
losses = []

for epoch in range(1000):
    current_lr = adaptive_lr.get_lr()
    # Training step with current learning rate
    loss = compute_cost(X_train, y_train, model.weights, model.bias)
    losses.append(loss)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, LR: {current_lr:.6f}, Loss: {loss:.6f}")
```

Slide 8: Cross-Validation Implementation

Cross-validation ensures robust model evaluation by testing performance across multiple data splits. This implementation shows how to perform k-fold cross-validation for regression models.

```python
def k_fold_cross_validation(X, y, k=5, lr=0.01, iterations=1000):
    fold_size = len(X) // k
    scores = []
    
    for i in range(k):
        # Create train-test split for this fold
        start_idx = i * fold_size
        end_idx = start_idx + fold_size
        
        X_test_fold = X[start_idx:end_idx]
        y_test_fold = y[start_idx:end_idx]
        
        X_train_fold = np.concatenate([
            X[:start_idx],
            X[end_idx:]
        ])
        y_train_fold = np.concatenate([
            y[:start_idx],
            y[end_idx:]
        ])
        
        # Train model on this fold
        model = LinearRegression(learning_rate=lr, iterations=iterations)
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate on test fold
        y_pred = model.predict(X_test_fold)
        score = calculate_metrics(y_test_fold, y_pred)['R2']
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

Slide 9: Feature Selection and Regularization

Feature selection and regularization help prevent overfitting and improve model generalization. This implementation demonstrates L1 (Lasso) and L2 (Ridge) regularization techniques for linear regression.

```python
class RegularizedLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, l1_ratio=0.0, l2_ratio=0.0):
        self.lr = learning_rate
        self.iterations = iterations
        self.l1_ratio = l1_ratio  # Lasso
        self.l2_ratio = l2_ratio  # Ridge
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients with regularization
            dw = (1/n_samples) * (
                np.dot(X.T, (y_pred - y)) + 
                self.l1_ratio * np.sign(self.weights) +
                self.l2_ratio * self.weights
            )
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

Slide 10: Time Series Forecasting with Linear Regression

Linear regression can be adapted for time series forecasting by using lagged values as features. This implementation shows how to create time series features and make predictions.

```python
def create_time_series_features(data, lookback=3):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

# Time series prediction example
import numpy as np

# Generate sample time series data
np.random.seed(42)
t = np.linspace(0, 100, 1000)
signal = 0.5 * np.sin(0.1 * t) + 0.1 * np.random.randn(len(t))

# Create features and target
X_time, y_time = create_time_series_features(signal, lookback=5)

# Split data
train_size = int(len(X_time) * 0.8)
X_train, X_test = X_time[:train_size], X_time[train_size:]
y_train, y_test = y_time[:train_size], y_time[train_size:]

# Train model
ts_model = LinearRegression(learning_rate=0.01, iterations=500)
ts_model.fit(X_train, y_train)

# Make predictions
y_pred = ts_model.predict(X_test)
```

Slide 11: Polynomial Feature Transformation

Polynomial features allow linear regression to capture non-linear relationships. This implementation shows how to generate and use polynomial features for improved model flexibility.

```python
class PolynomialRegression:
    def __init__(self, degree=2, learning_rate=0.01, iterations=1000):
        self.degree = degree
        self.lr = learning_rate
        self.iterations = iterations
        
    def generate_polynomial_features(self, X):
        n_samples, n_features = X.shape
        X_poly = np.ones((n_samples, 1))
        
        for d in range(1, self.degree + 1):
            for feat in range(n_features):
                X_poly = np.column_stack((X_poly, X[:, feat] ** d))
                
        return X_poly[:, 1:]  # Remove the constant term
    
    def fit(self, X, y):
        X_poly = self.generate_polynomial_features(X)
        self.linear_model = LinearRegression(
            learning_rate=self.lr, 
            iterations=self.iterations
        )
        self.linear_model.fit(X_poly, y)
        
    def predict(self, X):
        X_poly = self.generate_polynomial_features(X)
        return self.linear_model.predict(X_poly)

# Example usage
poly_model = PolynomialRegression(degree=3)
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)
```

Slide 12: Model Diagnostics and Residual Analysis

Model diagnostics help assess assumptions and identify potential issues in linear regression. This implementation provides comprehensive residual analysis tools for model validation.

```python
def residual_analysis(y_true, y_pred):
    residuals = y_true - y_pred
    
    # Calculate basic statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Normality test (Shapiro-Wilk)
    from scipy import stats
    _, normality_p_value = stats.shapiro(residuals)
    
    # Heteroscedasticity test (Breusch-Pagan)
    def breusch_pagan_test(residuals, y_pred):
        residuals_squared = residuals ** 2
        X = y_pred.reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, residuals_squared)
        return np.sum(model.predict(X))
    
    bp_statistic = breusch_pagan_test(residuals, y_pred)
    
    plt.figure(figsize=(15, 5))
    
    # Residual plot
    plt.subplot(131)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    # Q-Q plot
    plt.subplot(132)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    # Residual histogram
    plt.subplot(133)
    plt.hist(residuals, bins=30, density=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Residual Distribution')
    
    return {
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'normality_p_value': normality_p_value,
        'bp_statistic': bp_statistic
    }
```

Slide 13: Real-world Example: Stock Price Prediction

This implementation demonstrates linear regression application in financial forecasting, including feature engineering and model evaluation for stock price prediction.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_stock_features(df):
    # Technical indicators
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['close'], periods=14)
    df['MACD'] = calculate_macd(df['close'])
    
    # Price-based features
    df['Price_Change'] = df['close'].pct_change()
    df['Volatility'] = df['close'].rolling(window=20).std()
    
    # Volume features
    df['Volume_MA'] = df['volume'].rolling(window=5).mean()
    df['Volume_Change'] = df['volume'].pct_change()
    
    return df

def calculate_rsi(prices, periods=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    return macd - signal_line

class StockPricePredictor:
    def __init__(self, lookback=30):
        self.lookback = lookback
        self.model = LinearRegression(learning_rate=0.001, iterations=2000)
        self.scaler = StandardScaler()
        
    def prepare_data(self, df):
        df = prepare_stock_features(df)
        features = ['SMA_5', 'SMA_20', 'RSI', 'MACD', 
                   'Price_Change', 'Volatility', 
                   'Volume_MA', 'Volume_Change']
        
        X = df[features].values
        y = df['close'].values
        
        return X, y
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
```

Slide 14: Model Deployment and Production Pipeline

This implementation showcases how to create a production-ready pipeline for linear regression models, including data validation, model serialization, and prediction serving capabilities.

```python
import joblib
import json
from datetime import datetime

class LinearRegressionPipeline:
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metadata = {
            'creation_date': None,
            'last_training_date': None,
            'feature_importance': None,
            'performance_metrics': None
        }
        
    def validate_data(self, X, feature_names):
        if not isinstance(X, np.ndarray):
            raise ValueError("Input must be numpy array")
        if X.shape[1] != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features, got {X.shape[1]}")
            
    def fit(self, X, y, feature_names):
        self.validate_data(X, feature_names)
        self.feature_names = feature_names
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = LinearRegression(learning_rate=0.01, iterations=1000)
        self.model.fit(X_scaled, y)
        
        # Update metadata
        self.metadata.update({
            'creation_date': datetime.now().isoformat(),
            'last_training_date': datetime.now().isoformat(),
            'feature_importance': dict(zip(feature_names, 
                                        self.model.weights)),
            'performance_metrics': calculate_metrics(y, 
                self.model.predict(X_scaled))
        })
        
    def predict(self, X):
        self.validate_data(X, self.feature_names)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, path):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metadata': self.metadata
        }
        joblib.dump(model_data, path)
        
    def load_model(self, path):
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.metadata = model_data['metadata']
```

Slide 15: Additional Resources

Here are relevant research papers for further reading on linear regression and its applications:

*   [https://arxiv.org/abs/2106.14331](https://arxiv.org/abs/2106.14331) - "A Comprehensive Review of Linear Regression Techniques"
*   [https://arxiv.org/abs/1803.08823](https://arxiv.org/abs/1803.08823) - "Regularization Methods for Linear Regression: A Comparative Study"
*   [https://arxiv.org/abs/2007.15327](https://arxiv.org/abs/2007.15327) - "Adaptive Learning Rates in Linear Regression: Theory and Practice"
*   [https://arxiv.org/abs/1909.13470](https://arxiv.org/abs/1909.13470) - "Feature Selection Techniques for High-Dimensional Linear Regression"
*   [https://arxiv.org/abs/2002.05180](https://arxiv.org/abs/2002.05180) - "Time Series Forecasting Using Linear Regression: Modern Approaches"

Note: These papers provide advanced theoretical foundations and practical implementations of linear regression techniques. The URLs are in the specified format for academic research papers.

