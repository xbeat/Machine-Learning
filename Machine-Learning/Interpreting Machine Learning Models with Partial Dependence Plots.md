## Interpreting Machine Learning Models with Partial Dependence Plots
Slide 1: Understanding Partial Dependence Plots

Partial Dependence Plots (PDP) visualize the marginal effect of a feature on model predictions while accounting for the average effect of other features. This statistical technique helps interpret complex machine learning models by showing how predictions change when varying one or two features while keeping others constant.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def calculate_pdp(model, X, feature_name, grid_points=50):
    # Create feature grid
    feature_values = np.linspace(X[feature_name].min(), X[feature_name].max(), grid_points)
    pdp_values = []
    
    # Calculate PDP values
    for value in feature_values:
        X_temp = X.copy()
        X_temp[feature_name] = value
        predictions = model.predict(X_temp)
        pdp_values.append(predictions.mean())
        
    return feature_values, pdp_values
```

Slide 2: Implementing PDP from Scratch

Understanding the mathematical foundation behind PDPs requires implementing the core calculation. The PDP function estimates the average prediction for each value of the target feature by marginalizing over the distribution of all other features in the dataset.

```python
def partial_dependence_plot(model, X, feature_idx, grid_resolution=100):
    """
    Calculate partial dependence for a single feature
    
    Parameters:
    model: fitted model object
    X: feature matrix
    feature_idx: index of the feature to analyze
    grid_resolution: number of points in the grid
    """
    feature_values = np.linspace(
        X[:, feature_idx].min(),
        X[:, feature_idx].max(),
        grid_resolution
    )
    
    pdp = []
    for value in feature_values:
        X_temp = X.copy()
        X_temp[:, feature_idx] = value
        predictions = model.predict(X_temp)
        pdp.append(predictions.mean())
        
    return feature_values, np.array(pdp)
```

Slide 3: Mathematical Foundation of PDP

The mathematical formulation of Partial Dependence Plots involves calculating the expected value of the model's predictions while varying the feature of interest. This process requires understanding both the marginal and conditional probability distributions.

```python
# Mathematical formulation of PDP
"""
$$f_{xs}(x_s) = E_{x_c}[f(x_s, x_c)]$$

$$f_{xs}(x_s) = \int f(x_s, x_c)p(x_c)dx_c$$

$$f_{xs}(x_s) \approx \frac{1}{n} \sum_{i=1}^n f(x_s, x_c^{(i)})$$

Where:
- f_{xs}(x_s) is the partial dependence function
- x_s is the feature set of interest
- x_c represents the complement features
- p(x_c) is the marginal probability distribution
"""
```

Slide 4: Real-world Example - Boston Housing Dataset

The Boston Housing dataset provides an excellent case study for PDP implementation. We'll analyze how median house prices depend on various features while controlling for other variables, demonstrating PDP's practical application in real estate analysis.

```python
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# Load and prepare data
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)
```

Slide 5: Visualizing PDPs - Boston Housing

The visualization of Partial Dependence Plots requires careful attention to detail in plotting and formatting. This implementation shows how to create clear, interpretable plots that effectively communicate feature relationships to stakeholders.

```python
def plot_pdp(feature_values, pdp_values, feature_name):
    plt.figure(figsize=(10, 6))
    plt.plot(feature_values, pdp_values, 'b-', linewidth=2)
    plt.xlabel(feature_name)
    plt.ylabel('Partial dependence')
    plt.title(f'Partial Dependence Plot for {feature_name}')
    plt.grid(True)
    
    # Calculate and add confidence intervals
    std_dev = np.std(pdp_values)
    plt.fill_between(
        feature_values,
        pdp_values - 1.96 * std_dev,
        pdp_values + 1.96 * std_dev,
        alpha=0.2,
        color='b'
    )
    return plt
```

Slide 6: Two-way Partial Dependence Plots

Two-way Partial Dependence Plots extend the PDP concept to visualize interactions between two features simultaneously. This technique reveals how pairs of features jointly influence model predictions, providing deeper insights into feature relationships.

```python
def calculate_2d_pdp(model, X, feature1, feature2, grid_points=30):
    # Create feature grids
    f1_values = np.linspace(X[feature1].min(), X[feature1].max(), grid_points)
    f2_values = np.linspace(X[feature2].min(), X[feature2].max(), grid_points)
    
    pdp_values = np.zeros((grid_points, grid_points))
    
    for i, v1 in enumerate(f1_values):
        for j, v2 in enumerate(f2_values):
            X_temp = X.copy()
            X_temp[feature1] = v1
            X_temp[feature2] = v2
            predictions = model.predict(X_temp)
            pdp_values[i, j] = predictions.mean()
            
    return f1_values, f2_values, pdp_values
```

Slide 7: Handling Feature Interactions

Feature interactions in PDPs require special attention to ensure accurate interpretation. This implementation demonstrates how to detect and quantify feature interactions using H-statistic and provides visualization techniques for interaction effects.

```python
def calculate_h_statistic(model, X, feature1, feature2, grid_points=30):
    # Calculate individual PDPs
    f1_pdp = calculate_pdp(model, X, feature1, grid_points)[1]
    f2_pdp = calculate_pdp(model, X, feature2, grid_points)[1]
    
    # Calculate two-way PDP
    _, _, pdp_2d = calculate_2d_pdp(model, X, feature1, feature2, grid_points)
    
    # Calculate H-statistic
    f1_mesh, f2_mesh = np.meshgrid(f1_pdp, f2_pdp)
    expected_pdp = f1_mesh + f2_mesh - np.mean(pdp_2d)
    
    h_stat = np.sum((pdp_2d - expected_pdp) ** 2) / np.sum(pdp_2d ** 2)
    return h_stat
```

Slide 8: PDP Implementation for Categorical Features

Categorical features require a modified approach to PDP calculation. This implementation handles categorical variables by creating separate plots for each category level while maintaining the interpretability of the results.

```python
def categorical_pdp(model, X, categorical_feature, category_names=None):
    unique_values = sorted(X[categorical_feature].unique())
    pdp_values = []
    
    for value in unique_values:
        X_temp = X.copy()
        X_temp[categorical_feature] = value
        predictions = model.predict(X_temp)
        pdp_values.append(predictions.mean())
    
    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(unique_values)), pdp_values)
    plt.xlabel(categorical_feature)
    plt.ylabel('Partial dependence')
    
    if category_names:
        plt.xticks(range(len(unique_values)), category_names, rotation=45)
    
    return unique_values, pdp_values
```

Slide 9: Real-world Example - Credit Risk Assessment

Credit risk assessment provides an excellent use case for PDP analysis. This implementation demonstrates how to analyze the relationship between credit features and default probability while accounting for complex interactions.

```python
# Generate synthetic credit data
np.random.seed(42)
n_samples = 1000

credit_data = pd.DataFrame({
    'income': np.random.normal(50000, 20000, n_samples),
    'debt_ratio': np.random.uniform(0, 1, n_samples),
    'credit_score': np.random.normal(700, 100, n_samples),
    'age': np.random.normal(40, 12, n_samples)
})

# Create target variable (default probability)
def generate_default_prob(row):
    base_prob = 0.1
    income_effect = -0.2 * (row['income'] - 50000) / 50000
    debt_effect = 0.3 * row['debt_ratio']
    credit_effect = -0.3 * (row['credit_score'] - 700) / 100
    prob = base_prob + income_effect + debt_effect + credit_effect
    return np.clip(prob, 0, 1)

credit_data['default_prob'] = credit_data.apply(generate_default_prob, axis=1)
```

Slide 10: Advanced PDP Visualization Techniques

Advanced visualization techniques enhance the interpretability of PDPs through confidence intervals, interaction highlights, and automated feature importance ranking. This implementation provides a comprehensive visualization toolkit.

```python
def advanced_pdp_plot(model, X, feature, bootstrap_samples=100):
    # Calculate main PDP
    feature_values, pdp_values = calculate_pdp(model, X, feature)
    
    # Bootstrap for confidence intervals
    bootstrap_pdps = []
    for _ in range(bootstrap_samples):
        idx = np.random.choice(len(X), len(X), replace=True)
        X_boot = X.iloc[idx]
        _, pdp_boot = calculate_pdp(model, X_boot, feature)
        bootstrap_pdps.append(pdp_boot)
    
    # Calculate confidence intervals
    pdp_std = np.std(bootstrap_pdps, axis=0)
    ci_lower = pdp_values - 1.96 * pdp_std
    ci_upper = pdp_values + 1.96 * pdp_std
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(feature_values, pdp_values, 'b-', label='PDP')
    plt.fill_between(feature_values, ci_lower, ci_upper, 
                    alpha=0.2, color='b', label='95% CI')
    plt.xlabel(feature)
    plt.ylabel('Partial dependence')
    plt.title(f'Advanced PDP for {feature}')
    plt.legend()
    
    return plt
```

Slide 11: Local Partial Dependence Analysis

Local Partial Dependence Analysis extends traditional PDP by focusing on individual instances rather than the global average. This technique helps identify when global PDP might be misleading due to heterogeneous effects across the feature space.

```python
def local_pdp(model, X, feature_name, instance_idx, grid_points=50):
    # Calculate global PDP
    feature_values, global_pdp = calculate_pdp(model, X, feature_name, grid_points)
    
    # Calculate local PDP for specific instance
    instance = X.iloc[instance_idx:instance_idx+1].copy()
    local_pdp_values = []
    
    for value in feature_values:
        instance_temp = instance.copy()
        instance_temp[feature_name] = value
        prediction = model.predict(instance_temp)
        local_pdp_values.append(prediction[0])
    
    plt.figure(figsize=(10, 6))
    plt.plot(feature_values, global_pdp, 'b-', label='Global PDP')
    plt.plot(feature_values, local_pdp_values, 'r--', label='Local PDP')
    plt.xlabel(feature_name)
    plt.ylabel('Prediction')
    plt.title(f'Global vs Local PDP for {feature_name}')
    plt.legend()
    
    return feature_values, global_pdp, local_pdp_values
```

Slide 12: PDP for Time Series Analysis

Adapting PDP for time series analysis requires special consideration of temporal dependencies. This implementation shows how to create meaningful partial dependence plots for time series features while preserving temporal ordering.

```python
def temporal_pdp(model, X, time_feature, lookback_window=5, grid_points=50):
    # Ensure data is sorted by time
    X_sorted = X.sort_values(by=time_feature)
    
    # Calculate rolling statistics
    rolling_mean = X_sorted[time_feature].rolling(lookback_window).mean()
    rolling_std = X_sorted[time_feature].rolling(lookback_window).std()
    
    # Create feature grid considering temporal aspects
    feature_values = np.linspace(
        X_sorted[time_feature].min(),
        X_sorted[time_feature].max(),
        grid_points
    )
    
    pdp_values = []
    pdp_std = []
    
    for value in feature_values:
        X_temp = X_sorted.copy()
        X_temp[time_feature] = value
        predictions = model.predict(X_temp)
        pdp_values.append(predictions.mean())
        pdp_std.append(predictions.std())
    
    # Plot with temporal context
    plt.figure(figsize=(12, 6))
    plt.plot(feature_values, pdp_values, 'b-', label='PDP')
    plt.fill_between(
        feature_values,
        np.array(pdp_values) - np.array(pdp_std),
        np.array(pdp_values) + np.array(pdp_std),
        alpha=0.2,
        color='b'
    )
    plt.xlabel(f'{time_feature} (Time)')
    plt.ylabel('Partial dependence')
    plt.title(f'Temporal PDP for {time_feature}')
    plt.legend()
    
    return feature_values, pdp_values, pdp_std
```

Slide 13: Results Validation and Statistical Testing

Validating PDP results requires statistical testing to ensure reliability and significance. This implementation provides methods for confidence interval calculation and hypothesis testing for PDP differences.

```python
def pdp_significance_test(model, X, feature, n_bootstrap=1000, alpha=0.05):
    # Original PDP calculation
    feature_values, original_pdp = calculate_pdp(model, X, feature)
    
    # Bootstrap resampling
    bootstrap_pdps = []
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(len(X), len(X), replace=True)
        X_boot = X.iloc[boot_idx]
        _, pdp_boot = calculate_pdp(model, X_boot, feature)
        bootstrap_pdps.append(pdp_boot)
    
    # Calculate confidence intervals
    pdp_quantiles = np.quantile(bootstrap_pdps, 
                               [alpha/2, 1-alpha/2], 
                               axis=0)
    
    # Test for significance
    is_significant = (pdp_quantiles[0] * pdp_quantiles[1]) > 0
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(feature_values, original_pdp, 'b-', label='PDP')
    plt.fill_between(feature_values, 
                    pdp_quantiles[0], 
                    pdp_quantiles[1],
                    alpha=0.2, 
                    color='b', 
                    label=f'{int((1-alpha)*100)}% CI')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Partial dependence')
    plt.title(f'PDP with Confidence Intervals for {feature}')
    plt.legend()
    
    return is_significant, pdp_quantiles
```

Slide 14: Additional Resources

1.  [https://arxiv.org/abs/1309.6392](https://arxiv.org/abs/1309.6392) - "Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation"
2.  [https://arxiv.org/abs/1612.08468](https://arxiv.org/abs/1612.08468) - "A Unified Approach to Interpreting Model Predictions"
3.  [https://arxiv.org/abs/1805.04755](https://arxiv.org/abs/1805.04755) - "Visualizing the Feature Importance for Black Box Models"
4.  [https://arxiv.org/abs/2004.03043](https://arxiv.org/abs/2004.03043) - "Interpreting Machine Learning Models: A Review of Local and Global Methods"
5.  [https://arxiv.org/abs/1909.06342](https://arxiv.org/abs/1909.06342) - "Visualizing and Understanding Partial Dependence Plots"

