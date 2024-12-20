## Poisson Regression for Modeling Count Data

Slide 1: Understanding Poisson Distribution Fundamentals

The Poisson distribution models the probability of events occurring in fixed intervals of time or space. It's particularly useful for count data where events are independent and occur at a constant average rate, making it ideal for rare event modeling.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Generate Poisson distributed data
lambda_param = 3.0  # Mean rate
k = np.arange(0, 10)  # Number of events
poisson_pmf = poisson.pmf(k, lambda_param)

# Plot PMF
plt.figure(figsize=(10, 6))
plt.bar(k, poisson_pmf, alpha=0.8, label=f'λ = {lambda_param}')
plt.title('Poisson Probability Mass Function')
plt.xlabel('Number of Events (k)')
plt.ylabel('Probability P(X = k)')
plt.legend()
plt.grid(True, alpha=0.3)
```

Slide 2: Mathematical Foundation of Poisson Regression

Poisson regression uses a log link function to model count data, ensuring predictions are always positive. The model assumes the logarithm of the expected value can be modeled by a linear combination of predictors.

```python
# Mathematical formulation in LaTeX notation
"""
$$
\log(\lambda) = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n
$$

$$
P(Y = k) = \frac{e^{-\lambda}\lambda^k}{k!}
$$

where:
$$
\lambda = E(Y|X) = e^{\beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n}
$$
"""
```

Slide 3: Implementing Basic Poisson Regression

A practical implementation of Poisson regression using statsmodels, demonstrating the fundamental approach to modeling count data with a simple example of website visits prediction based on advertising spend.

```python
import statsmodels.api as sm
import numpy as np
import pandas as pd

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100)
lambda_true = np.exp(1 + 0.3 * X)
y = np.random.poisson(lambda_true)

# Prepare data for statsmodels
X = sm.add_constant(X)

# Fit Poisson regression
model = sm.GLM(y, X, family=sm.families.Poisson())
results = model.fit()
print(results.summary())
```

Slide 4: Comparing Linear and Poisson Regression

This comparison demonstrates the key differences between linear and Poisson regression using identical datasets, highlighting how Poisson regression handles count data more appropriately by preventing negative predictions.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Generate data
np.random.seed(42)
X = np.linspace(0, 5, 100).reshape(-1, 1)
y_true = np.exp(1 + 0.5 * X.ravel())
y = np.random.poisson(y_true)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X, y)
y_pred_lr = lr_model.predict(X)

# Poisson Regression
X_sm = sm.add_constant(X)
poisson_model = sm.GLM(y, X_sm, family=sm.families.Poisson())
poisson_results = poisson_model.fit()
y_pred_poisson = poisson_results.predict(X_sm)

print(f"Linear Regression MSE: {mean_squared_error(y, y_pred_lr):.4f}")
print(f"Poisson Regression MSE: {mean_squared_error(y, y_pred_poisson):.4f}")
```

Slide 5: Handling Overdispersion

Overdispersion occurs when the variance exceeds the mean in count data. This implementation shows how to detect and handle overdispersion using Negative Binomial regression as an alternative.

```python
import statsmodels.api as sm
from scipy import stats

def check_overdispersion(model, y):
    """
    Check for overdispersion in Poisson regression
    """
    pearson_chi2 = model.pearson_chi2
    deg_freedom = model.df_resid
    dispersion = pearson_chi2 / deg_freedom
    
    print(f"Dispersion parameter: {dispersion:.4f}")
    print(f"Overdispersed: {dispersion > 1}")
    
    return dispersion

# Fit models and compare
poisson_model = sm.GLM(y, X_sm, family=sm.families.Poisson())
nb_model = sm.GLM(y, X_sm, family=sm.families.NegativeBinomial())

poisson_results = poisson_model.fit()
nb_results = nb_model.fit()

dispersion = check_overdispersion(poisson_results, y)
```

Slide 6: Call Center Data Prediction Model

Poisson regression effectively models call center incoming volumes by considering temporal patterns and external factors. This implementation demonstrates how to predict hourly call volumes using historical data patterns.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Create sample call center data
hours = np.arange(24)
weekdays = np.arange(7)
calls_data = pd.DataFrame({
    'hour': np.random.choice(hours, 1000),
    'weekday': np.random.choice(weekdays, 1000),
    'special_event': np.random.binomial(1, 0.1, 1000),
    'calls': np.random.poisson(20, 1000)
})

# Prepare features and fit model
X = pd.get_dummies(calls_data[['hour', 'weekday']])
X['special_event'] = calls_data['special_event']
X = sm.add_constant(X)

model = sm.GLM(calls_data['calls'], X, family=sm.families.Poisson())
results = model.fit()

# Make predictions
predictions = results.predict(X)
print(f"Model AIC: {results.aic:.2f}")
```

Slide 7: Handling Zero-Inflation in Count Data

Zero-inflated Poisson regression addresses datasets with excessive zero counts. This implementation shows how to detect and handle zero-inflation using a specialized model approach.

```python
import numpy as np
from scipy import stats

def zero_inflation_test(data):
    # Calculate expected zeros under Poisson
    lambda_hat = np.mean(data)
    n = len(data)
    expected_zeros = n * np.exp(-lambda_hat)
    observed_zeros = np.sum(data == 0)
    
    # Simple zero-inflation test
    ratio = observed_zeros / expected_zeros
    
    print(f"Zero-inflation ratio: {ratio:.2f}")
    print(f"Observed zeros: {observed_zeros}")
    print(f"Expected zeros: {expected_zeros:.2f}")
    
    return ratio > 1.5  # Return True if zero-inflated

# Generate sample data with excess zeros
n_samples = 1000
regular_data = np.random.poisson(2, n_samples)
zero_inflated = np.where(np.random.random(n_samples) < 0.3, 0, regular_data)

# Test for zero-inflation
is_zero_inflated = zero_inflation_test(zero_inflated)
```

Slide 8: Diagnostic Tools for Poisson Regression

Model diagnostics are crucial for validating Poisson regression assumptions. This implementation provides essential diagnostic tools for residual analysis and goodness-of-fit assessment.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def poisson_diagnostics(model, y_true, y_pred):
    # Pearson residuals
    residuals = (y_true - y_pred) / np.sqrt(y_pred)
    
    # Deviance residuals
    dev_residuals = np.sign(y_true - y_pred) * np.sqrt(
        2 * (y_true * np.log(y_true/y_pred) - (y_true - y_pred))
    )
    
    # Plot diagnostics
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Pearson Residuals')
    
    plt.subplot(122)
    plt.hist(dev_residuals, bins=30, alpha=0.7)
    plt.xlabel('Deviance Residuals')
    plt.ylabel('Frequency')
    
    return residuals, dev_residuals
```

Slide 9: Cross-Validation for Poisson Models

Cross-validation techniques adapted specifically for count data models help assess predictive performance and prevent overfitting in Poisson regression.

```python
from sklearn.model_selection import KFold
import numpy as np
import statsmodels.api as sm

def poisson_cross_validate(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model
        model = sm.GLM(y_train, sm.add_constant(X_train), 
                      family=sm.families.Poisson())
        results = model.fit()
        
        # Predict and calculate deviance
        y_pred = results.predict(sm.add_constant(X_test))
        deviance = 2 * np.sum(y_test * np.log(y_test/y_pred) - 
                             (y_test - y_pred))
        scores.append(deviance)
    
    return np.mean(scores), np.std(scores)
```

Slide 10: Feature Importance in Poisson Models

Understanding which features contribute most significantly to count predictions is crucial for model interpretation and refinement. This implementation provides methods to analyze and visualize feature importance.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_poisson_features(model_results):
    # Extract and organize feature importance
    coef = pd.DataFrame({
        'feature': model_results.model.exog_names[1:],
        'coefficient': model_results.params[1:],
        'std_err': model_results.bse[1:]
    })
    
    # Sort by absolute coefficient value
    coef['abs_coef'] = abs(coef['coefficient'])
    coef = coef.sort_values('abs_coef', ascending=True)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(coef)), coef['abs_coef'])
    plt.yticks(range(len(coef)), coef['feature'])
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Feature Importance in Poisson Regression')
    
    return coef
```

Slide 11: Model Comparison Framework

Comparing different count regression models helps select the most appropriate approach for specific datasets. This framework evaluates Poisson, Negative Binomial, and Zero-Inflated models.

```python
import numpy as np
import statsmodels.api as sm
from scipy import stats

def compare_count_models(X, y):
    # Fit Poisson
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
    poisson_results = poisson_model.fit()
    
    # Fit Negative Binomial
    nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    nb_results = nb_model.fit()
    
    # Compare metrics
    metrics = {
        'Poisson_AIC': poisson_results.aic,
        'NB_AIC': nb_results.aic,
        'Poisson_BIC': poisson_results.bic,
        'NB_BIC': nb_results.bic
    }
    
    return metrics
```

Slide 12: Time Series Analysis with Poisson Regression

Incorporating temporal dependencies in count data analysis requires specialized approaches. This implementation handles time series count data using Poisson regression with time-based features.

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm

def time_series_poisson(dates, counts):
    # Create time features
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'count': counts
    })
    
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    
    # Create cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # Prepare features for model
    features = ['hour_sin', 'hour_cos', 'weekday', 'month']
    X = pd.get_dummies(df[features], columns=['weekday', 'month'])
    X = sm.add_constant(X)
    
    # Fit model
    model = sm.GLM(df['count'], X, family=sm.families.Poisson())
    results = model.fit()
    
    return results, df
```

Slide 13: Regularization in Poisson Regression

Implementing regularization helps prevent overfitting in Poisson regression, especially with high-dimensional feature spaces. This example demonstrates L1 and L2 regularization.

```python
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

def regularized_poisson(X, y, alpha=1.0, l1_ratio=0.5):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Add constant
    X_scaled = sm.add_constant(X_scaled)
    
    # Fit model with regularization
    model = sm.GLM(y, X_scaled, family=sm.families.Poisson())
    results = model.fit_regularized(
        alpha=alpha,
        L1_wt=l1_ratio,
        maxiter=1000
    )
    
    return results, scaler

# Example usage
alpha_values = [0.1, 1.0, 10.0]
results_dict = {
    alpha: regularized_poisson(X, y, alpha=alpha)
    for alpha in alpha_values
}
```

Slide 14: Additional Resources

1.  [https://arxiv.org/abs/1912.07998](https://arxiv.org/abs/1912.07998) - "A Comprehensive Review of Poisson Regression and Extensions"
2.  [https://arxiv.org/abs/2008.07478](https://arxiv.org/abs/2008.07478) - "Modern Approaches to Count Data Modeling"
3.  [https://arxiv.org/abs/1804.03876](https://arxiv.org/abs/1804.03876) - "Zero-Inflated Poisson Regression: Theory and Applications"
4.  [https://arxiv.org/abs/2103.09435](https://arxiv.org/abs/2103.09435) - "Regularization Methods for Generalized Linear Models"
5.  [https://arxiv.org/abs/1902.08956](https://arxiv.org/abs/1902.08956) - "Time Series Analysis with Count Data: An Overview"

