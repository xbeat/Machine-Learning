## Bias-Variance Tradeoff Clearly Explained
Slide 1: Understanding Bias and Variance Components

The bias-variance decomposition mathematically breaks down the prediction error of a machine learning model into its fundamental components. Understanding these components is crucial for diagnosing model performance and making informed decisions about model complexity.

```python
# Mathematical representation of Bias-Variance decomposition
# Error = Bias^2 + Variance + Irreducible Error

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def calculate_bias_variance(model, X_train, y_train, X_test, y_test, n_iterations=100):
    predictions = np.zeros((n_iterations, len(X_test)))
    
    for i in range(n_iterations):
        # Bootstrap sampling
        indices = np.random.randint(0, len(X_train), len(X_train))
        X_boot, y_boot = X_train[indices], y_train[indices]
        
        # Fit model and predict
        model.fit(X_boot, y_boot)
        predictions[i, :] = model.predict(X_test)
    
    # Calculate bias and variance
    mean_predictions = np.mean(predictions, axis=0)
    bias = np.mean((mean_predictions - y_test) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    
    return bias, variance

# Example usage
X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
bias, variance = calculate_bias_variance(model, X_train, y_train, X_test, y_test)
print(f"Bias: {bias:.4f}")
print(f"Variance: {variance:.4f}")
```

Slide 2: Visualizing the Bias-Variance Tradeoff

Understanding how model complexity affects bias and variance requires visualization. This implementation creates a comprehensive plot showing how different polynomial degrees impact both components, providing insight into the optimal complexity level.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def plot_bias_variance_tradeoff():
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 1, 100).reshape(-1, 1)
    y_true = np.sin(2 * np.pi * X) + np.random.normal(0, 0.1, X.shape)
    
    degrees = range(1, 15)
    bias_scores = []
    variance_scores = []
    
    for degree in degrees:
        # Create polynomial model
        model = make_pipeline(
            PolynomialFeatures(degree),
            LinearRegression()
        )
        
        # Calculate bias and variance
        predictions = np.zeros((100, len(X)))
        for i in range(100):
            # Add noise to training data
            y_noisy = y_true + np.random.normal(0, 0.1, y_true.shape)
            model.fit(X, y_noisy)
            predictions[i, :] = model.predict(X)
        
        # Calculate metrics
        mean_pred = predictions.mean(axis=0)
        bias = np.mean((mean_pred - np.sin(2 * np.pi * X.flatten())) ** 2)
        variance = np.mean(predictions.var(axis=0))
        
        bias_scores.append(bias)
        variance_scores.append(variance)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, bias_scores, label='Bias²', color='blue')
    plt.plot(degrees, variance_scores, label='Variance', color='red')
    plt.plot(degrees, np.array(bias_scores) + np.array(variance_scores), 
             label='Total Error', color='purple', linestyle='--')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Error')
    plt.title('Bias-Variance Tradeoff')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_bias_variance_tradeoff()
```

Slide 3: Implementing Cross-Validation for Model Selection

Cross-validation provides a robust framework for assessing the bias-variance tradeoff in practice. This implementation demonstrates how to use k-fold cross-validation to select optimal model complexity while avoiding overfitting.

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

def cross_validate_complexity(X, y, max_degree=10, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    degrees = range(1, max_degree + 1)
    cv_scores = np.zeros((len(degrees), n_splits))
    
    for i, degree in enumerate(degrees):
        model = make_pipeline(
            PolynomialFeatures(degree),
            LinearRegression()
        )
        
        for j, (train_idx, val_idx) in enumerate(kf.split(X)):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            cv_scores[i, j] = mean_squared_error(y_val, y_pred)
    
    # Calculate mean and std of CV scores
    mean_scores = cv_scores.mean(axis=1)
    std_scores = cv_scores.std(axis=1)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(degrees, mean_scores, yerr=std_scores, 
                label='CV Score', capsize=5)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('Cross-Validation Scores vs Model Complexity')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return degrees[np.argmin(mean_scores)]

# Example usage
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.1, X.shape)
optimal_degree = cross_validate_complexity(X, y)
print(f"Optimal polynomial degree: {optimal_degree}")
```

Slide 4: Real-world Application - Housing Price Prediction

Implementing bias-variance analysis on the Boston Housing dataset demonstrates practical model optimization. This implementation showcases how different model complexities affect prediction accuracy in a real estate valuation context.

```python
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Load and prepare data
boston = load_boston()
X, y = boston.data, boston.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

def analyze_model_complexity(X_train, X_test, y_train, y_test):
    complexities = np.linspace(0.0001, 1, 20)  # Alpha values for Ridge regression
    train_errors = []
    test_errors = []
    
    for alpha in complexities:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_errors.append(mean_squared_error(y_train, train_pred))
        test_errors.append(mean_squared_error(y_test, test_pred))
    
    return complexities, train_errors, test_errors

# Plot results
complexities, train_errors, test_errors = analyze_model_complexity(
    X_train, X_test, y_train, y_test
)

plt.figure(figsize=(10, 6))
plt.plot(complexities, train_errors, label='Training Error')
plt.plot(complexities, test_errors, label='Test Error')
plt.xscale('log')
plt.xlabel('Model Complexity (alpha)')
plt.ylabel('Mean Squared Error')
plt.title('Error vs Model Complexity in Housing Price Prediction')
plt.legend()
plt.grid(True)
plt.show()

# Print optimal complexity
optimal_idx = np.argmin(test_errors)
print(f"Optimal complexity (alpha): {complexities[optimal_idx]:.6f}")
print(f"Minimum test error: {test_errors[optimal_idx]:.4f}")
```

Slide 5: Learning Curves Analysis

Learning curves provide crucial insights into model performance by showing how training and validation errors evolve with increasing training data size, helping identify bias and variance issues.

```python
def plot_learning_curves(X, y, model, cv=5):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    
    # Calculate mean and std
    train_mean = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = -np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Error')
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1
    )
    plt.plot(train_sizes, val_mean, label='Validation Error')
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.1
    )
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curves')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    return train_mean[-1], val_mean[-1]

# Example usage with different model complexities
models = {
    'Low Complexity': Ridge(alpha=10),
    'Medium Complexity': Ridge(alpha=1),
    'High Complexity': Ridge(alpha=0.01)
}

for name, model in models.items():
    print(f"\nAnalyzing {name}:")
    train_error, val_error = plot_learning_curves(X_scaled, y, model)
    print(f"Final training error: {train_error:.4f}")
    print(f"Final validation error: {val_error:.4f}")
```

Slide 6: Ensemble Methods for Bias-Variance Control

Ensemble methods provide powerful tools for managing the bias-variance tradeoff through combining multiple models. This implementation demonstrates how bagging and boosting affect model performance.

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def compare_ensemble_methods(X_train, X_test, y_train, y_test):
    # Initialize models
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    # Train models
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    
    # Make predictions
    rf_pred = rf.predict(X_test)
    gb_pred = gb.predict(X_test)
    
    # Calculate metrics
    results = {
        'Random Forest': {
            'MSE': mean_squared_error(y_test, rf_pred),
            'R2': r2_score(y_test, rf_pred),
            'predictions': rf_pred
        },
        'Gradient Boosting': {
            'MSE': mean_squared_error(y_test, gb_pred),
            'R2': r2_score(y_test, gb_pred),
            'predictions': gb_pred
        }
    }
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, rf_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Random Forest Predictions')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, gb_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Gradient Boosting Predictions')
    
    plt.tight_layout()
    plt.show()
    
    return results

# Run comparison
results = compare_ensemble_methods(X_train, X_test, y_train, y_test)

# Print results
for model, metrics in results.items():
    print(f"\n{model} Results:")
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"R2 Score: {metrics['R2']:.4f}")
```

Slide 7: Regularization Techniques for Variance Reduction

Regularization methods provide effective tools for controlling model variance by adding constraints to the optimization objective. This implementation compares different regularization techniques and their impact on model performance.

```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet
import numpy as np
import matplotlib.pyplot as plt

def compare_regularization_methods(X_train, X_test, y_train, y_test):
    # Initialize regularization parameters
    alphas = np.logspace(-4, 4, 100)
    
    # Dictionary to store results
    results = {
        'Ridge': [],
        'Lasso': [],
        'ElasticNet': []
    }
    
    # Train models with different alphas
    for alpha in alphas:
        # Ridge Regression
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        ridge_score = mean_squared_error(y_test, ridge.predict(X_test))
        results['Ridge'].append(ridge_score)
        
        # Lasso Regression
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)
        lasso_score = mean_squared_error(y_test, lasso.predict(X_test))
        results['Lasso'].append(lasso_score)
        
        # ElasticNet
        elastic = ElasticNet(alpha=alpha, l1_ratio=0.5)
        elastic.fit(X_train, y_train)
        elastic_score = mean_squared_error(y_test, elastic.predict(X_test))
        results['ElasticNet'].append(elastic_score)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for method, scores in results.items():
        plt.plot(alphas, scores, label=method)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Regularization Parameter (alpha)')
    plt.ylabel('Mean Squared Error')
    plt.title('Regularization Methods Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Find optimal alpha for each method
    for method, scores in results.items():
        optimal_idx = np.argmin(scores)
        print(f"\n{method}:")
        print(f"Optimal alpha: {alphas[optimal_idx]:.6f}")
        print(f"Minimum MSE: {scores[optimal_idx]:.6f}")

# Example usage
X, y = make_regression(n_samples=200, n_features=50, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

compare_regularization_methods(X_train, X_test, y_train, y_test)
```

Slide 8: Model Complexity Analysis

This implementation provides a comprehensive framework for analyzing how model complexity affects the bias-variance tradeoff through polynomial feature expansion and regularization.

```python
def analyze_model_complexity():
    # Generate synthetic data with non-linear relationship
    np.random.seed(42)
    X = np.sort(np.random.uniform(0, 1, 100)).reshape(-1, 1)
    y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.2, X.shape)
    
    degrees = range(1, 15)
    train_errors = []
    val_errors = []
    test_errors = []
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
    
    for degree in degrees:
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)
        X_test_poly = poly.transform(X_test)
        
        # Fit model
        model = Ridge(alpha=0.1)
        model.fit(X_train_poly, y_train)
        
        # Calculate errors
        train_pred = model.predict(X_train_poly)
        val_pred = model.predict(X_val_poly)
        test_pred = model.predict(X_test_poly)
        
        train_errors.append(mean_squared_error(y_train, train_pred))
        val_errors.append(mean_squared_error(y_val, val_pred))
        test_errors.append(mean_squared_error(y_test, test_pred))
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(degrees, train_errors, label='Training Error')
    plt.plot(degrees, val_errors, label='Validation Error')
    plt.plot(degrees, test_errors, label='Test Error')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('Error vs Model Complexity')
    plt.legend()
    plt.grid(True)
    
    # Plot example fits
    plt.subplot(2, 1, 2)
    degrees_to_plot = [1, 3, 10]
    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    
    for degree in degrees_to_plot:
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_plot_poly = poly.transform(X_plot)
        
        model = Ridge(alpha=0.1)
        model.fit(X_train_poly, y_train)
        y_plot = model.predict(X_plot_poly)
        
        plt.plot(X_plot, y_plot, label=f'Degree {degree}')
    
    plt.scatter(X, y, color='black', alpha=0.5, label='Data')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Model Fits of Different Complexities')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

analyze_model_complexity()
```

Slide 9: Bootstrap Analysis for Variance Estimation

Bootstrap resampling provides a powerful method for estimating model variance by creating multiple training datasets. This implementation demonstrates how to use bootstrapping to assess model stability and variance.

```python
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt

def bootstrap_analysis(X, y, n_bootstraps=100):
    n_samples = X.shape[0]
    predictions = np.zeros((n_bootstraps, n_samples))
    coefficients = np.zeros((n_bootstraps, X.shape[1]))
    
    for i in range(n_bootstraps):
        # Create bootstrap sample
        X_boot, y_boot = resample(X, y, n_samples=n_samples)
        
        # Fit model
        model = Ridge(alpha=1.0)
        model.fit(X_boot, y_boot)
        
        # Store predictions and coefficients
        predictions[i, :] = model.predict(X)
        coefficients[i, :] = model.coef_
    
    # Calculate prediction intervals
    pred_mean = predictions.mean(axis=0)
    pred_std = predictions.std(axis=0)
    
    # Calculate coefficient statistics
    coef_mean = coefficients.mean(axis=0)
    coef_std = coefficients.std(axis=0)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot prediction intervals
    ax1.fill_between(range(n_samples),
                    pred_mean - 2*pred_std,
                    pred_mean + 2*pred_std,
                    alpha=0.3, label='95% Prediction Interval')
    ax1.plot(range(n_samples), pred_mean, 'r-', label='Mean Prediction')
    ax1.scatter(range(n_samples), y, alpha=0.5, label='Actual Values')
    ax1.set_title('Bootstrap Predictions with Confidence Intervals')
    ax1.legend()
    ax1.grid(True)
    
    # Plot coefficient distributions
    ax2.bar(range(X.shape[1]), coef_mean)
    ax2.errorbar(range(X.shape[1]), coef_mean, yerr=2*coef_std,
                fmt='none', color='black', capsize=5)
    ax2.set_title('Feature Coefficients with 95% Confidence Intervals')
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Coefficient Value')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'coef_mean': coef_mean,
        'coef_std': coef_std
    }

# Example usage
X, y = make_regression(n_samples=100, n_features=5, random_state=42)
results = bootstrap_analysis(X, y)

# Print summary statistics
print("\nFeature Importance Analysis:")
for i, (mean, std) in enumerate(zip(results['coef_mean'], results['coef_std'])):
    print(f"Feature {i}: {mean:.4f} ± {2*std:.4f}")
```

Slide 10: Cross-Decomposition of Error Sources

This implementation provides a detailed breakdown of prediction error into its bias and variance components, helping identify the primary sources of model error.

```python
def error_decomposition_analysis(X, y, model_complexities):
    """
    Analyze error components across different model complexities.
    """
    n_complexities = len(model_complexities)
    bias_squared = np.zeros(n_complexities)
    variance = np.zeros(n_complexities)
    total_error = np.zeros(n_complexities)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    for i, complexity in enumerate(model_complexities):
        # Create model with current complexity
        model = make_pipeline(
            PolynomialFeatures(complexity),
            Ridge(alpha=0.1)
        )
        
        # Bootstrap iterations for variance estimation
        n_bootstrap = 100
        predictions = np.zeros((n_bootstrap, len(X_test)))
        
        for b in range(n_bootstrap):
            # Create bootstrap sample
            boot_idx = np.random.choice(len(X_train), len(X_train))
            X_boot = X_train[boot_idx]
            y_boot = y_train[boot_idx]
            
            # Fit model and predict
            model.fit(X_boot, y_boot)
            predictions[b] = model.predict(X_test)
        
        # Calculate error components
        expected_predictions = np.mean(predictions, axis=0)
        bias_squared[i] = np.mean((expected_predictions - y_test) ** 2)
        variance[i] = np.mean(np.var(predictions, axis=0))
        total_error[i] = bias_squared[i] + variance[i]
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(model_complexities, bias_squared, label='Bias²')
    plt.plot(model_complexities, variance, label='Variance')
    plt.plot(model_complexities, total_error, label='Total Error')
    plt.xlabel('Model Complexity (Polynomial Degree)')
    plt.ylabel('Error')
    plt.title('Decomposition of Prediction Error')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return bias_squared, variance, total_error

# Example usage
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = np.sin(4 * np.pi * X) + np.random.normal(0, 0.3, X.shape)
complexities = range(1, 15)

bias, var, total = error_decomposition_analysis(X, y, complexities)

# Print optimal complexity
optimal_idx = np.argmin(total)
print(f"\nOptimal model complexity: {complexities[optimal_idx]}")
print(f"Minimum total error: {total[optimal_idx]:.4f}")
print(f"At optimal complexity:")
print(f"Bias²: {bias[optimal_idx]:.4f}")
print(f"Variance: {var[optimal_idx]:.4f}")
```

Slide 11: Real-world Application - Time Series Prediction

Applying bias-variance analysis to time series forecasting demonstrates how model complexity affects prediction accuracy in sequential data, particularly important for financial and environmental modeling.

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def time_series_complexity_analysis(data, window_sizes, forecast_horizon=1):
    """
    Analyze bias-variance tradeoff in time series prediction.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    results = {
        'window_size': [],
        'train_error': [],
        'test_error': [],
        'variance': []
    }
    
    for window in window_sizes:
        # Prepare sequences
        X, y = [], []
        for i in range(len(scaled_data) - window - forecast_horizon + 1):
            X.append(scaled_data[i:i+window])
            y.append(scaled_data[i+window:i+window+forecast_horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Create and train model
        model = Sequential([
            LSTM(50, input_shape=(window, 1)),
            Dense(forecast_horizon)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Calculate metrics
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Store results
        results['window_size'].append(window)
        results['train_error'].append(mean_squared_error(y_train, train_pred))
        results['test_error'].append(mean_squared_error(y_test, test_pred))
        
        # Calculate prediction variance
        bootstrap_preds = []
        for _ in range(10):
            boot_idx = np.random.choice(len(X_train), len(X_train))
            model.fit(X_train[boot_idx], y_train[boot_idx], 
                     epochs=50, verbose=0)
            bootstrap_preds.append(model.predict(X_test))
        
        variance = np.mean([np.var(pred) for pred in zip(*bootstrap_preds)])
        results['variance'].append(variance)
    
    # Plotting results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(results['window_size'], results['train_error'], 
             label='Training Error')
    plt.plot(results['window_size'], results['test_error'], 
             label='Test Error')
    plt.xlabel('Window Size')
    plt.ylabel('Mean Squared Error')
    plt.title('Error vs Window Size')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(results['window_size'], results['variance'], 
             label='Prediction Variance')
    plt.xlabel('Window Size')
    plt.ylabel('Variance')
    plt.title('Model Variance vs Window Size')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Generate example time series data
t = np.linspace(0, 10, 1000)
data = np.sin(2*np.pi*t) + 0.5*np.sin(4*np.pi*t) + np.random.normal(0, 0.1, len(t))

# Analyze different window sizes
window_sizes = [5, 10, 20, 30, 40, 50]
results = time_series_complexity_analysis(data, window_sizes)

# Print optimal window size
optimal_idx = np.argmin(results['test_error'])
print(f"\nOptimal window size: {window_sizes[optimal_idx]}")
print(f"Minimum test error: {results['test_error'][optimal_idx]:.6f}")
print(f"Corresponding variance: {results['variance'][optimal_idx]:.6f}")
```

Slide 12: Feature Selection Impact on Bias-Variance

This implementation explores how feature selection methods affect the bias-variance tradeoff, demonstrating the relationship between feature dimensionality and model performance.

```python
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

def analyze_feature_selection_impact(X, y, max_features):
    """
    Analyze how feature selection affects bias-variance tradeoff.
    """
    n_features_range = range(1, min(max_features + 1, X.shape[1]))
    results = {
        'filter': {'bias': [], 'variance': [], 'total_error': []},
        'pca': {'bias': [], 'variance': [], 'total_error': []}
    }
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    for n_features in n_features_range:
        # Filter-based selection
        selector = SelectKBest(f_regression, k=n_features)
        X_train_filter = selector.fit_transform(X_train, y_train)
        X_test_filter = selector.transform(X_test)
        
        # PCA
        pca = PCA(n_components=n_features)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # Analyze both methods
        for method, (X_tr, X_te) in [
            ('filter', (X_train_filter, X_test_filter)),
            ('pca', (X_train_pca, X_test_pca))
        ]:
            # Bootstrap for variance estimation
            predictions = np.zeros((100, len(X_test)))
            for i in range(100):
                boot_idx = np.random.choice(len(X_tr), len(X_tr))
                model = Ridge(alpha=1.0)
                model.fit(X_tr[boot_idx], y_train[boot_idx])
                predictions[i] = model.predict(X_te)
            
            # Calculate error components
            mean_pred = predictions.mean(axis=0)
            bias = np.mean((mean_pred - y_test) ** 2)
            variance = np.mean(np.var(predictions, axis=0))
            total_error = bias + variance
            
            # Store results
            results[method]['bias'].append(bias)
            results[method]['variance'].append(variance)
            results[method]['total_error'].append(total_error)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    for method, color in [('filter', 'blue'), ('pca', 'red')]:
        plt.plot(n_features_range, results[method]['bias'], 
                f'{color}--', label=f'{method.upper()} Bias²')
        plt.plot(n_features_range, results[method]['variance'], 
                f'{color}:', label=f'{method.upper()} Variance')
        plt.plot(n_features_range, results[method]['total_error'], 
                color, label=f'{method.upper()} Total Error')
    
    plt.xlabel('Number of Features')
    plt.ylabel('Error')
    plt.title('Feature Selection Impact on Bias-Variance Tradeoff')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results

# Generate example data
X, y = make_regression(n_samples=200, n_features=20, 
                      n_informative=10, random_state=42)
results = analyze_feature_selection_impact(X, y, 15)
```

Slide 13: Model Calibration and Bias-Variance Balance

Understanding model calibration in relation to the bias-variance tradeoff is crucial for achieving reliable probability estimates. This implementation analyzes calibration curves across different model complexities.

```python
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def analyze_calibration_complexity(X, y, complexities):
    """
    Analyze how model complexity affects probability calibration.
    """
    # Prepare data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    plt.figure(figsize=(12, 8))
    
    for complexity in complexities:
        # Create polynomial features
        poly = PolynomialFeatures(degree=complexity)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # Train model
        model = LogisticRegression(C=1.0)
        model.fit(X_train_poly, y_train)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test_poly)[:, 1]
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(
            y_test, y_pred_proba, n_bins=10
        )
        
        # Plot calibration curve
        plt.plot(prob_pred, prob_true, 
                marker='o', label=f'Degree {complexity}')
        
    # Plot ideal calibration
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curves for Different Model Complexities')
    plt.legend()
    plt.grid(True)
    
    # Calculate and return calibration metrics
    results = {}
    for complexity in complexities:
        poly = PolynomialFeatures(degree=complexity)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        model = LogisticRegression(C=1.0)
        model.fit(X_train_poly, y_train)
        
        y_pred_proba = model.predict_proba(X_test_poly)[:, 1]
        
        # Calculate Brier score
        brier_score = np.mean((y_pred_proba - y_test) ** 2)
        
        # Store results
        results[complexity] = {
            'brier_score': brier_score,
            'predictions': y_pred_proba
        }
    
    plt.show()
    return results

# Generate example binary classification data
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=15, random_state=42)
complexities = [1, 2, 3, 5, 7]
calibration_results = analyze_calibration_complexity(X, y, complexities)

# Print calibration metrics
print("\nCalibration Results:")
for complexity, metrics in calibration_results.items():
    print(f"\nPolynomial Degree {complexity}:")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
```

Slide 14: Adaptive Model Selection Framework

This implementation provides a comprehensive framework for automatically selecting model complexity based on the bias-variance tradeoff through cross-validation and adaptive regularization.

```python
def adaptive_model_selection(X, y, max_degree=10):
    """
    Implement adaptive model selection based on bias-variance analysis.
    """
    # Initialize storage for metrics
    degrees = range(1, max_degree + 1)
    cv_scores = np.zeros((len(degrees), 5))
    bias_estimates = np.zeros(len(degrees))
    variance_estimates = np.zeros(len(degrees))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    for i, degree in enumerate(degrees):
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # Cross-validation for model stability
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for j, (train_idx, val_idx) in enumerate(kf.split(X_train_poly)):
            # Split data
            X_cv_train = X_train_poly[train_idx]
            X_cv_val = X_train_poly[val_idx]
            y_cv_train = y_train[train_idx]
            y_cv_val = y_train[val_idx]
            
            # Train model
            model = Ridge(alpha=0.1)
            model.fit(X_cv_train, y_cv_train)
            
            # Calculate validation score
            cv_scores[i, j] = mean_squared_error(
                y_cv_val, model.predict(X_cv_val)
            )
        
        # Bootstrap for bias-variance estimation
        predictions = np.zeros((100, len(X_test)))
        for b in range(100):
            boot_idx = np.random.choice(len(X_train), len(X_train))
            X_boot = X_train_poly[boot_idx]
            y_boot = y_train[boot_idx]
            
            model = Ridge(alpha=0.1)
            model.fit(X_boot, y_boot)
            predictions[b] = model.predict(X_test_poly)
        
        # Calculate bias and variance
        mean_pred = predictions.mean(axis=0)
        bias_estimates[i] = np.mean((mean_pred - y_test) ** 2)
        variance_estimates[i] = np.mean(np.var(predictions, axis=0))
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.errorbar(degrees, cv_scores.mean(axis=1), 
                yerr=cv_scores.std(axis=1), 
                label='Cross-validation Score')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('Cross-validation Scores vs Model Complexity')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(degrees, bias_estimates, label='Bias²')
    plt.plot(degrees, variance_estimates, label='Variance')
    plt.plot(degrees, bias_estimates + variance_estimates, 
            label='Total Error')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Error')
    plt.title('Bias-Variance Decomposition')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Select optimal complexity
    total_error = bias_estimates + variance_estimates
    optimal_degree = degrees[np.argmin(total_error)]
    
    return {
        'optimal_degree': optimal_degree,
        'cv_scores': cv_scores,
        'bias_estimates': bias_estimates,
        'variance_estimates': variance_estimates
    }

# Example usage
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = np.sin(4 * np.pi * X) + np.random.normal(0, 0.3, X.shape)

results = adaptive_model_selection(X, y)
print(f"\nOptimal polynomial degree: {results['optimal_degree']}")
print(f"Minimum total error: {(results['bias_estimates'] + results['variance_estimates'])[results['optimal_degree']-1]:.4f}")
```

Slide 15: Additional Resources

*   arXiv:1906.10742 - "Understanding the Bias-Variance Tradeoff: An Information-Theoretic Perspective" [https://arxiv.org/abs/1906.10742](https://arxiv.org/abs/1906.10742)
*   arXiv:2001.00686 - "Reconciling Modern Machine Learning Practice and the Bias-Variance Trade-Off" [https://arxiv.org/abs/2001.00686](https://arxiv.org/abs/2001.00686)
*   arXiv:1812.11118 - "A Unified View of the Bias-Variance Decomposition in Neural Networks" [https://arxiv.org/abs/1812.11118](https://arxiv.org/abs/1812.11118)
*   For more practical implementations and tutorials, search for:
    *   "Bias-Variance Tradeoff in Deep Learning"
    *   "Practical Guide to Model Selection and Validation"
    *   "Advanced Model Complexity Analysis"

