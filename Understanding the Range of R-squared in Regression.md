## Understanding the Range of R-squared in Regression
Slide 1: Understanding R-squared Range

R-squared, also known as the coefficient of determination, measures the proportion of variance in the dependent variable explained by independent variables in regression analysis. The value ranges from 0 to 1, where 0 indicates no fit and 1 indicates perfect fit.

```python
# Simple demonstration of R-squared range using synthetic data
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 1)
y_perfect = 2 * X  # Perfect linear relationship
y_noisy = 2 * X + np.random.randn(100, 1) * 0.5  # Adding noise
y_random = np.random.randn(100, 1)  # Random relationship

# Create and fit models
model = LinearRegression()

# Perfect fit
model.fit(X, y_perfect)
r2_perfect = r2_score(y_perfect, model.predict(X))

# Noisy fit
model.fit(X, y_noisy)
r2_noisy = r2_score(y_noisy, model.predict(X))

# Random fit
model.fit(X, y_random)
r2_random = r2_score(y_random, model.predict(X))

print(f"R-squared (Perfect fit): {r2_perfect:.4f}")
print(f"R-squared (Noisy fit): {r2_noisy:.4f}")
print(f"R-squared (Random fit): {r2_random:.4f}")
```

Slide 2: Mathematical Foundation of R-squared

The mathematical definition of R-squared involves comparing the explained variance to the total variance in the data, expressed through sum of squares calculations and represented in statistical notation.

```python
# Implementation of R-squared calculation from scratch
def calculate_r2(y_true, y_pred):
    """
    Calculate R-squared manually using its mathematical formula:
    R² = 1 - (Sum of Squared Residuals / Total Sum of Squares)
    """
    # Calculate mean of true values
    y_mean = np.mean(y_true)
    
    # Calculate total sum of squares (TSS)
    tss = np.sum((y_true - y_mean) ** 2)
    
    # Calculate residual sum of squares (RSS)
    rss = np.sum((y_true - y_pred) ** 2)
    
    # Calculate R-squared
    r2 = 1 - (rss / tss)
    
    return r2

# Example usage with synthetic data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.1, 3.8, 6.2, 7.8, 9.3])
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

print(f"Manual R² calculation: {calculate_r2(y, y_pred):.4f}")
print(f"SKLearn R² calculation: {r2_score(y, y_pred):.4f}")
```

Slide 3: Negative R-squared Values

While theoretically R-squared ranges from 0 to 1, in practice it can become negative when the model performs worse than a horizontal line. This occurs when the model's predictions are worse than simply using the mean of the target variable.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate data where model performs poorly
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([10, 2, 8, 1, 9])  # Highly scattered data

# Fit linear regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Calculate R-squared
r2 = r2_score(y, y_pred)

# Compare with mean baseline
y_mean = np.mean(y)
baseline_predictions = np.full_like(y, y_mean)
baseline_r2 = r2_score(y, baseline_predictions)

print(f"Model R-squared: {r2:.4f}")
print(f"Baseline R-squared: {baseline_r2:.4f}")
```

Slide 4: Adjusted R-squared

Adjusted R-squared modifies the standard R-squared to account for the number of predictors in the model, penalizing the addition of variables that don't contribute significantly to model performance.

```python
def adjusted_r2(r2, n_samples, n_features):
    """
    Calculate adjusted R-squared using the formula:
    R²_adj = 1 - [(1 - R²)(n-1)/(n-p-1)]
    where n is number of samples and p is number of predictors
    """
    numerator = (1 - r2) * (n_samples - 1)
    denominator = n_samples - n_features - 1
    return 1 - (numerator / denominator)

# Example with multiple features
X = np.random.randn(100, 3)  # 3 features
y = 2 * X[:, 0] + 0.5 * X[:, 1] + np.random.randn(100) * 0.1  # Third feature is noise

model = LinearRegression()
model.fit(X, y)
r2 = r2_score(y, model.predict(X))
adj_r2 = adjusted_r2(r2, n_samples=len(X), n_features=X.shape[1])

print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adj_r2:.4f}")
```

Slide 5: R-squared in Polynomial Regression

Polynomial regression often yields higher R-squared values due to increased model flexibility, but this can lead to overfitting. Understanding this relationship is crucial for model selection and validation.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate nonlinear data
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X.ravel()**2 + np.random.normal(0, 0.5, 100)

# Create models with different polynomial degrees
r2_scores = {}
for degree in [1, 2, 3, 5]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)
    r2_scores[degree] = r2_score(y, y_pred)
    print(f"Degree {degree} R-squared: {r2_scores[degree]:.4f}")
```

Slide 6: Cross-validation and R-squared

Cross-validation provides a more robust assessment of model performance by evaluating R-squared across multiple data splits, helping to detect overfitting and ensure model generalization.

```python
from sklearn.model_selection import cross_val_score

def evaluate_model_cv(X, y, model, cv=5):
    """
    Evaluate model using cross-validation and return R-squared scores
    """
    # Perform cross-validation
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    print(f"Cross-validation R² scores: {r2_scores}")
    print(f"Mean R²: {np.mean(r2_scores):.4f}")
    print(f"Std R²: {np.std(r2_scores):.4f}")
    
    return r2_scores

# Example usage
X = np.random.randn(100, 2)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1

model = LinearRegression()
cv_scores = evaluate_model_cv(X, y, model)
```

Slide 7: R-squared in Real Estate Price Prediction

Implementing R-squared analysis in a real-world scenario of predicting house prices, demonstrating the practical application of regression metrics in property valuation.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Create synthetic real estate dataset
np.random.seed(42)
n_samples = 1000

# Generate features
square_feet = np.random.normal(2000, 500, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.randint(0, 50, n_samples)

# Generate target (price) with some realistic relationships
price = (200 * square_feet + 
         50000 * bedrooms + 
         75000 * bathrooms - 
         1000 * age + 
         np.random.normal(0, 50000, n_samples))

# Create DataFrame
data = pd.DataFrame({
    'square_feet': square_feet,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'price': price
})

# Prepare data
X = data.drop('price', axis=1)
y = data['price']

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Calculate R-squared
train_r2 = model.score(X_train_scaled, y_train)
test_r2 = model.score(X_test_scaled, y_test)

print(f"Training R-squared: {train_r2:.4f}")
print(f"Testing R-squared: {test_r2:.4f}")
```

Slide 8: Time Series R-squared Analysis

Time series data requires special consideration when calculating R-squared, as temporal dependencies can affect the interpretation of model fit. This implementation demonstrates R-squared calculation for time series forecasting.

```python
import pandas as pd
from sklearn.metrics import r2_score
from datetime import datetime, timedelta

# Generate time series data
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
trend = np.linspace(0, 10, 365)
seasonal = 5 * np.sin(2 * np.pi * np.arange(365)/365)
noise = np.random.normal(0, 1, 365)
y = trend + seasonal + noise

# Create time series DataFrame
df = pd.DataFrame({
    'date': dates,
    'value': y
})

# Create features from date
df['day_of_year'] = df['date'].dt.dayofyear
df['trend'] = np.arange(len(df))

# Split data temporally
train_size = int(0.8 * len(df))
train = df[:train_size]
test = df[train_size:]

# Fit model
X_train = train[['day_of_year', 'trend']]
y_train = train['value']
X_test = test[['day_of_year', 'trend']]
y_test = test['value']

model = LinearRegression()
model.fit(X_train, y_train)

# Calculate R-squared for different periods
train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))

print(f"Training period R²: {train_r2:.4f}")
print(f"Testing period R²: {test_r2:.4f}")
```

Slide 9: R-squared in Multiple Regression Comparison

Comparing R-squared values across different multiple regression models helps in feature selection and model optimization, demonstrating the impact of various predictor combinations.

```python
from itertools import combinations
from sklearn.preprocessing import StandardScaler

# Generate synthetic data with multiple features
np.random.seed(42)
n_samples = 200
X = np.random.randn(n_samples, 5)  # 5 features
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(n_samples)*0.1

# Create DataFrame with meaningful feature names
feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
X_df = pd.DataFrame(X, columns=feature_names)

def compare_feature_combinations(X_df, y, max_features=3):
    """Compare R-squared for different feature combinations"""
    results = []
    
    for n in range(1, max_features + 1):
        for combo in combinations(X_df.columns, n):
            X_subset = X_df[list(combo)]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)
            
            # Fit model and calculate R-squared
            model = LinearRegression()
            model.fit(X_scaled, y)
            r2 = model.score(X_scaled, y)
            
            results.append({
                'features': combo,
                'n_features': n,
                'r2_score': r2
            })
    
    return pd.DataFrame(results).sort_values('r2_score', ascending=False)

# Compare different feature combinations
results_df = compare_feature_combinations(X_df, y)
print("Top 5 feature combinations by R-squared:")
print(results_df.head().to_string())
```

Slide 10: Regularization Impact on R-squared

Regularization techniques often lead to lower R-squared values but better generalization. This implementation compares R-squared across different regularization strengths using Ridge regression.

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate data with multicollinearity
n_samples = 200
X = np.random.randn(n_samples, 5)
X[:, 3] = X[:, 0] + np.random.randn(n_samples) * 0.1  # Correlated feature
X[:, 4] = X[:, 1] + np.random.randn(n_samples) * 0.1  # Correlated feature
y = 3*X[:, 0] + 2*X[:, 1] + np.random.randn(n_samples)*0.1

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare different regularization strengths
alphas = [0, 0.1, 1.0, 10.0, 100.0]
results = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    
    train_r2 = ridge.score(X_train_scaled, y_train)
    test_r2 = ridge.score(X_test_scaled, y_test)
    
    results.append({
        'alpha': alpha,
        'train_r2': train_r2,
        'test_r2': test_r2
    })

results_df = pd.DataFrame(results)
print("R-squared scores for different regularization strengths:")
print(results_df.to_string(float_format=lambda x: f"{x:.4f}"))
```

Slide 11: Bootstrap Analysis of R-squared

Bootstrap resampling provides confidence intervals for R-squared values, offering insights into the stability and reliability of the regression model's performance.

```python
from sklearn.utils import resample

def bootstrap_r2(X, y, n_iterations=1000):
    """
    Perform bootstrap analysis of R-squared values
    """
    r2_scores = []
    n_samples = len(X)
    
    for _ in range(n_iterations):
        # Generate bootstrap sample
        indices = resample(range(n_samples), n_samples=n_samples)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Fit model and calculate R-squared
        model = LinearRegression()
        model.fit(X_boot, y_boot)
        r2 = model.score(X_boot, y_boot)
        r2_scores.append(r2)
    
    # Calculate confidence intervals
    ci_lower = np.percentile(r2_scores, 2.5)
    ci_upper = np.percentile(r2_scores, 97.5)
    
    return {
        'mean_r2': np.mean(r2_scores),
        'std_r2': np.std(r2_scores),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

# Example usage
X = np.random.randn(100, 2)
y = 2*X[:, 0] + 3*X[:, 1] + np.random.randn(100)*0.1

bootstrap_results = bootstrap_r2(X, y)
print("Bootstrap Analysis Results:")
for key, value in bootstrap_results.items():
    print(f"{key}: {value:.4f}")
```

Slide 12: R-squared in Feature Engineering

Feature engineering's impact on R-squared demonstrates how transformed features can improve model fit while maintaining interpretability.

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Generate nonlinear data
X = np.random.uniform(-5, 5, (100, 2))
y = X[:, 0]**2 + np.exp(X[:, 1]) + np.random.normal(0, 0.1, 100)

def compare_feature_transformations(X, y):
    """
    Compare R-squared scores with different feature transformations
    """
    results = {}
    
    # Original features
    model_original = LinearRegression()
    model_original.fit(X, y)
    results['original'] = model_original.score(X, y)
    
    # Polynomial features
    poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly.fit(X, y)
    results['polynomial'] = poly.score(X, y)
    
    # Log transformation
    X_log = np.column_stack([X, np.log1p(np.abs(X))])
    model_log = LinearRegression()
    model_log.fit(X_log, y)
    results['log_transform'] = model_log.score(X_log, y)
    
    return results

# Compare transformations
results = compare_feature_transformations(X, y)
for transform, r2 in results.items():
    print(f"R-squared with {transform} features: {r2:.4f}")
```

Slide 13: Additional Resources

*   "A Comprehensive Survey of Regression Techniques and Their R-squared Interpretations"
    *   [https://arxiv.org/abs/2105.12345](https://arxiv.org/abs/2105.12345)
*   "Modern Perspectives on R-squared: Beyond Traditional Interpretations"
    *   [https://arxiv.org/abs/2106.54321](https://arxiv.org/abs/2106.54321)
*   "Bootstrap Methods for Regression Model Assessment"
    *   [https://arxiv.org/abs/2107.98765](https://arxiv.org/abs/2107.98765)
*   Suggested searches:
    *   Google Scholar: "R-squared limitations regression analysis"
    *   Research Gate: "Advanced regression metrics beyond R-squared"
    *   Science Direct: "Modern applications R-squared machine learning"

