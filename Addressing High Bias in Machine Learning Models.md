## Addressing High Bias in Machine Learning Models
Slide 1: Understanding High Bias in Machine Learning

High bias in machine learning manifests as underfitting, where the model makes oversimplified assumptions about the data relationships. This results in poor performance on both training and test datasets, indicating the model lacks sufficient complexity to capture underlying patterns.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate non-linear data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

# Fit a linear model (high bias example)
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Plotting
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Linear Model (High Bias)')
plt.title('High Bias Example: Linear Model on Non-linear Data')
plt.legend()
plt.show()

# Calculate training error
mse = np.mean((y - y_pred) ** 2)
print(f"Training MSE: {mse:.4f}")
```

Slide 2: Detecting High Bias Through Learning Curves

Learning curves provide a diagnostic tool for identifying high bias by plotting training and validation errors against training set size. In high bias scenarios, both curves converge quickly to a high error value, indicating underfitting.

```python
from sklearn.model_selection import learning_curve

def plot_learning_curves(estimator, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    val_scores_mean = -np.mean(val_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, label='Training Error')
    plt.plot(train_sizes, val_scores_mean, label='Validation Error')
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curves: High Bias Model')
    plt.legend()
    plt.show()

# Using the same linear model
model = LinearRegression()
plot_learning_curves(model, X, y)
```

Slide 3: Mathematical Representation of Bias-Variance Decomposition

The mathematical foundation of bias-variance decomposition helps understand how high bias contributes to model error. This fundamental concept explains the relationship between model complexity and prediction error.

```python
# Mathematical representation of Bias-Variance decomposition
"""
Expected Test Error = Bias² + Variance + Irreducible Error

Where:
$$E[(y - \hat{f}(x))^2] = [E[\hat{f}(x)] - f(x)]^2 + E[\hat{f}(x) - E[\hat{f}(x)]]^2 + \sigma^2$$

$$Bias[\hat{f}(x)] = E[\hat{f}(x)] - f(x)$$
$$Variance[\hat{f}(x)] = E[\hat{f}(x) - E[\hat{f}(x)]]^2$$
"""

def bias_variance_demo(n_samples=100, n_experiments=1000):
    true_func = lambda x: np.sin(x)
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    predictions = np.zeros((n_experiments, n_samples))
    
    for i in range(n_experiments):
        y = true_func(X) + np.random.normal(0, 0.1, X.shape)
        model = LinearRegression()
        model.fit(X, y)
        predictions[i] = model.predict(X).ravel()
    
    mean_predictions = np.mean(predictions, axis=0)
    bias = mean_predictions - true_func(X).ravel()
    variance = np.var(predictions, axis=0)
    
    return np.mean(bias**2), np.mean(variance)

bias_squared, variance = bias_variance_demo()
print(f"Average Bias²: {bias_squared:.4f}")
print(f"Average Variance: {variance:.4f}")
```

Slide 4: Implementing Cross-Validation for Bias Detection

Cross-validation provides a robust method for detecting high bias by evaluating model performance across different data splits. Consistent high error across folds indicates high bias.

```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_squared_error

def analyze_bias_with_cv(X, y, model, cv=5):
    # Define scoring metrics
    scoring = {
        'mse': make_scorer(mean_squared_error),
        'neg_mse': 'neg_mean_squared_error'
    }
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True
    )
    
    # Calculate average scores
    train_mse = np.mean(cv_results['train_mse'])
    test_mse = np.mean(cv_results['test_mse'])
    
    print(f"Average Training MSE: {train_mse:.4f}")
    print(f"Average Test MSE: {test_mse:.4f}")
    print(f"Difference: {abs(train_mse - test_mse):.4f}")
    
    return train_mse, test_mse

# Example usage
model = LinearRegression()
train_mse, test_mse = analyze_bias_with_cv(X, y, model)
```

Slide 5: Real-world Example: House Price Prediction

This implementation demonstrates high bias in a real estate pricing model using actual housing data. The simple linear model fails to capture complex relationships between features and house prices.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate synthetic housing data
np.random.seed(42)
n_samples = 1000

# Create features
data = {
    'square_feet': np.random.normal(2000, 500, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'age': np.random.randint(0, 50, n_samples)
}

# Create non-linear price relationship
df = pd.DataFrame(data)
df['price'] = (
    100000 + 
    150 * df['square_feet'] + 
    25000 * df['bedrooms'] + 
    30000 * df['bathrooms'] -
    1000 * df['age'] +
    0.1 * df['square_feet']**2
)

# Add noise
df['price'] += np.random.normal(0, 50000, n_samples)

# Prepare data
X = df.drop('price', axis=1)
y = df['price']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train linear model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate
train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

print("Training MSE:", mean_squared_error(y_train, train_pred))
print("Test MSE:", mean_squared_error(y_test, test_pred))
```

Slide 6: Addressing High Bias Through Feature Engineering

Feature engineering can help mitigate high bias by introducing non-linear relationships and interaction terms. This transformation allows linear models to capture more complex patterns in the data.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def create_polynomial_features(X_train, X_test, degree=2):
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Transform training and test data
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Create pipeline
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    # Fit and evaluate
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training R² Score: {train_score:.4f}")
    print(f"Test R² Score: {test_score:.4f}")
    
    return model, X_train_poly, X_test_poly

# Example usage with housing data
model_poly, X_train_poly, X_test_poly = create_polynomial_features(
    X_train_scaled, X_test_scaled
)
```

Slide 7: Bias Reduction with Regularization Parameters

Understanding how regularization affects model bias is crucial. This implementation demonstrates the relationship between regularization strength and model bias using Ridge regression.

```python
from sklearn.linear_model import Ridge
import seaborn as sns

def analyze_regularization_impact(X_train, X_test, y_train, y_test):
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    train_scores = []
    test_scores = []
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, train_scores, 'b-', label='Training R²')
    plt.semilogx(alphas, test_scores, 'r-', label='Test R²')
    plt.xlabel('Regularization Strength (alpha)')
    plt.ylabel('R² Score')
    plt.title('Impact of Regularization on Model Performance')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return alphas, train_scores, test_scores

# Example usage
alphas, train_scores, test_scores = analyze_regularization_impact(
    X_train_scaled, X_test_scaled, y_train, y_test
)
```

Slide 8: Real-world Example: Credit Risk Assessment

High bias in credit risk assessment can lead to oversimplified models that fail to capture important risk factors. This implementation shows how to detect and address such issues.

```python
# Generate synthetic credit data
np.random.seed(42)
n_samples = 1000

# Create features
credit_data = {
    'income': np.random.normal(50000, 20000, n_samples),
    'debt_ratio': np.random.uniform(0, 1, n_samples),
    'credit_history': np.random.uniform(300, 850, n_samples),
    'employment_years': np.random.uniform(0, 30, n_samples)
}

# Create non-linear default probability
df_credit = pd.DataFrame(credit_data)
df_credit['default_prob'] = 1 / (1 + np.exp(-(
    -5 +
    0.00003 * df_credit['income'] -
    2 * df_credit['debt_ratio'] +
    0.01 * df_credit['credit_history'] +
    0.1 * df_credit['employment_years']**2
)))

# Convert to binary outcome
df_credit['default'] = (df_credit['default_prob'] > 0.5).astype(int)

# Prepare data
X_credit = df_credit.drop(['default_prob', 'default'], axis=1)
y_credit = df_credit['default']

# Split and scale data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_credit, y_credit, test_size=0.2
)
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

# Train and evaluate simple logistic regression
from sklearn.linear_model import LogisticRegression
model_credit = LogisticRegression()
model_credit.fit(X_train_c_scaled, y_train_c)

# Calculate metrics
from sklearn.metrics import classification_report
y_pred_c = model_credit.predict(X_test_c_scaled)
print(classification_report(y_test_c, y_pred_c))
```

Slide 9: Visualizing Decision Boundaries in High Bias Models

High bias models create oversimplified decision boundaries that fail to capture the true complexity of data relationships. This visualization demonstrates how linear boundaries misclassify non-linear patterns.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def visualize_decision_boundary(X, y, model, title):
    # Create mesh grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title(title)
    plt.show()

# Generate non-linear dataset
X_moons, y_moons = make_moons(n_samples=200, noise=0.15)

# Train linear model (high bias)
linear_model = LogisticRegression()
linear_model.fit(X_moons, y_moons)

# Visualize
visualize_decision_boundary(X_moons, y_moons, linear_model, 
                          'High Bias: Linear Decision Boundary')
```

Slide 10: Bias Analysis with Learning Curves Across Model Complexities

Comparing learning curves across different model complexities helps identify the optimal balance between bias and variance. This implementation demonstrates the progression from high to balanced bias.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def compare_model_complexities(X, y, degrees=[1, 2, 3]):
    plt.figure(figsize=(15, 5))
    
    for i, degree in enumerate(degrees, 1):
        # Create polynomial model
        model = make_pipeline(
            PolynomialFeatures(degree),
            LinearRegression()
        )
        
        # Calculate learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5,
            scoring='neg_mean_squared_error'
        )
        
        # Plot learning curves
        plt.subplot(1, 3, i)
        plt.plot(train_sizes, -np.mean(train_scores, axis=1), 
                label='Training Error')
        plt.plot(train_sizes, -np.mean(val_scores, axis=1), 
                label='Validation Error')
        plt.title(f'Degree {degree} Polynomial')
        plt.xlabel('Training Size')
        plt.ylabel('Mean Squared Error')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage with housing data
compare_model_complexities(X_train_scaled, y_train)
```

Slide 11: Cross-Model Bias Analysis Using K-Fold Validation

Systematic comparison of bias across different model types helps identify the most appropriate model complexity for a given problem domain.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def compare_model_bias(X, y, cv=5):
    # Define models with varying complexity
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Polynomial': make_pipeline(PolynomialFeatures(2), LinearRegression()),
        'SVR': SVR(kernel='linear'),
        'RandomForest': RandomForestRegressor(n_estimators=100)
    }
    
    results = {}
    
    for name, model in models.items():
        # Perform cross-validation
        scores = cross_validate(
            model, X, y,
            cv=cv,
            scoring=('neg_mean_squared_error'),
            return_train_score=True
        )
        
        # Store results
        results[name] = {
            'train_mse': -scores['train_score'].mean(),
            'test_mse': -scores['test_score'].mean(),
            'bias': abs(-scores['test_score'].mean() + 
                       scores['train_score'].mean())
        }
    
    # Display results
    results_df = pd.DataFrame(results).transpose()
    print("\nModel Comparison Results:")
    print(results_df.round(4))
    
    return results_df

# Example usage
bias_comparison = compare_model_bias(X_train_scaled, y_train)
```

Slide 12: High Bias in Time Series Prediction

Time series data often exhibits complex patterns that high-bias models fail to capture. This implementation demonstrates the limitations of linear models in forecasting non-linear time series.

```python
import pandas as pd
from datetime import datetime, timedelta

def generate_complex_timeseries():
    # Generate dates
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Generate complex pattern with multiple components
    trend = np.linspace(0, 100, n)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n)/365)
    noise = np.random.normal(0, 5, n)
    
    # Combine components
    values = trend + seasonal + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    return df

# Generate and prepare data
ts_data = generate_complex_timeseries()
X_time = np.arange(len(ts_data)).reshape(-1, 1)
y_time = ts_data['value'].values

# Fit linear model
linear_forecast = LinearRegression()
linear_forecast.fit(X_time, y_time)

# Make predictions
y_pred_time = linear_forecast.predict(X_time)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(ts_data['date'], y_time, label='Actual', alpha=0.7)
plt.plot(ts_data['date'], y_pred_time, label='Linear Prediction', color='red')
plt.title('High Bias in Time Series Forecasting')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# Calculate error metrics
mse_time = mean_squared_error(y_time, y_pred_time)
print(f"MSE for time series prediction: {mse_time:.2f}")
```

Slide 13: Bias Reduction Through Feature Selection

Strategic feature selection can help reduce model bias by focusing on the most relevant predictors while maintaining model interpretability.

```python
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score

def analyze_feature_importance(X, y, max_features=None):
    if max_features is None:
        max_features = X.shape[1]
    
    results = []
    feature_names = (X.columns if isinstance(X, pd.DataFrame) 
                    else [f'Feature_{i}' for i in range(X.shape[1])])
    
    for k in range(1, max_features + 1):
        # Select k best features
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Train model with selected features
        model = LinearRegression()
        scores = cross_validate(
            model, X_selected, y,
            cv=5,
            scoring=('r2', 'neg_mean_squared_error'),
            return_train_score=True
        )
        
        results.append({
            'n_features': k,
            'train_r2': scores['train_r2'].mean(),
            'test_r2': scores['test_r2'].mean(),
            'train_mse': -scores['train_neg_mean_squared_error'].mean(),
            'test_mse': -scores['test_neg_mean_squared_error'].mean()
        })
        
        # Get selected feature names
        mask = selector.get_support()
        selected_features = [f for f, m in zip(feature_names, mask) if m]
        print(f"\nTop {k} features: {', '.join(selected_features)}")
    
    return pd.DataFrame(results)

# Example usage with housing data
feature_analysis = analyze_feature_importance(pd.DataFrame(X_train), y_train)
print("\nFeature Selection Results:")
print(feature_analysis.round(4))
```

Slide 14: Additional Resources

*   Real-Time Detection of High Bias in ML Models: [https://arxiv.org/abs/2103.08191](https://arxiv.org/abs/2103.08191)
*   Advanced Techniques for Bias-Variance Analysis: [https://arxiv.org/abs/1910.13492](https://arxiv.org/abs/1910.13492)
*   Optimal Model Selection Under Bias Constraints: [https://arxiv.org/abs/2008.07152](https://arxiv.org/abs/2008.07152)
*   Suggested searches:
    *   "Bias reduction techniques in machine learning"
    *   "Feature engineering for reducing model bias"
    *   "Cross-validation strategies for bias detection"

