## Enhancing Regression Models with Quantile Analysis
Slide 1: Understanding Quantile Regression Fundamentals

Quantile regression extends beyond traditional mean-based regression by modeling the relationship between predictors and different quantiles of the response variable. This enables capturing the full conditional distribution rather than just the central tendency, providing richer insights into variable relationships.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=3, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Mathematical formulation for quantile regression
'''
$$
\min_{\beta} \sum_{i=1}^n \rho_\tau(y_i - x_i^T\beta)
$$

where $\rho_\tau(u) = u(\tau - I(u < 0))$ is the tilted absolute value function
'''
```

Slide 2: Implementing Basic Quantile Regression with Scikit-learn

Building a quantile regression model requires specialized implementations as it's not directly available in scikit-learn. We'll use the QuantileRegressor from sklearn.linear\_model to demonstrate the basic implementation for multiple quantiles.

```python
from sklearn.linear_model import QuantileRegressor
import matplotlib.pyplot as plt

# Initialize and train models for different quantiles
quantiles = [0.25, 0.5, 0.75]
models = {}

for q in quantiles:
    qr = QuantileRegressor(quantile=q, alpha=0)
    qr.fit(X_train, y_train)
    models[q] = qr
    
# Predict for each quantile
predictions = {q: model.predict(X_test) for q, model in models.items()}
```

Slide 3: LightGBM Quantile Regression Implementation

LightGBM provides native support for quantile regression through its objective function. This implementation demonstrates how to leverage LightGBM's capabilities for robust quantile predictions with gradient boosting.

```python
import lightgbm as lgb

# Create dataset for LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Parameters for different quantiles
params = {
    'objective': 'quantile',
    'metric': 'quantile',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'alpha': 0.5  # Corresponds to 50th percentile
}

# Train model
model = lgb.train(params, train_data, num_boost_round=100)
predictions_median = model.predict(X_test)
```

Slide 4: Multiple Quantile Predictions with LightGBM

Expanding our LightGBM implementation to predict multiple quantiles simultaneously requires training separate models for each desired quantile level, enabling comprehensive distributional insights.

```python
def train_quantile_models(X_train, y_train, quantiles):
    models = {}
    for q in quantiles:
        params = {
            'objective': 'quantile',
            'metric': 'quantile',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'alpha': q
        }
        train_data = lgb.Dataset(X_train, label=y_train)
        models[q] = lgb.train(params, train_data, num_boost_round=100)
    return models

# Train models for multiple quantiles
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
quantile_models = train_quantile_models(X_train, y_train, quantiles)
```

Slide 5: Real-world Application: Salary Prediction Model

Processing real salary data requires careful handling of categorical variables and feature engineering. This implementation demonstrates a complete pipeline for salary prediction across different quantiles.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Sample salary dataset preparation
salary_data = pd.DataFrame({
    'job_title': ['Data Scientist', 'ML Engineer', 'Data Analyst'] * 100,
    'experience': np.random.randint(1, 15, 300),
    'education': ['BS', 'MS', 'PhD'] * 100,
    'salary': np.random.normal(90000, 20000, 300)
})

# Preprocess categorical variables
le = LabelEncoder()
salary_data['job_title_encoded'] = le.fit_transform(salary_data['job_title'])
salary_data['education_encoded'] = le.fit_transform(salary_data['education'])
```

Slide 6: Source Code for Salary Prediction Model

This implementation showcases the complete modeling pipeline for salary prediction, including feature preprocessing, model training, and quantile predictions for different salary ranges.

```python
# Prepare features and target
X = salary_data[['job_title_encoded', 'education_encoded', 'experience']]
y = salary_data['salary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train quantile models
salary_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
salary_models = {}

for q in salary_quantiles:
    params = {
        'objective': 'quantile',
        'metric': 'quantile',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'alpha': q,
        'n_estimators': 200
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    salary_models[q] = lgb.train(params, train_data)

# Generate predictions for each quantile
salary_predictions = {
    q: model.predict(X_test) 
    for q, model in salary_models.items()
}

# Create prediction intervals
results_df = pd.DataFrame({
    f'q{int(q*100)}': preds 
    for q, preds in salary_predictions.items()
})
```

Slide 7: Visualizing Quantile Predictions

Understanding the distribution of predictions across different quantiles requires effective visualization. This implementation creates comprehensive plots to show prediction intervals and uncertainty.

```python
import seaborn as sns

# Create visualization of quantile predictions
plt.figure(figsize=(12, 6))

# Plot actual vs predicted for different quantiles
for q in [0.25, 0.5, 0.75]:
    plt.scatter(y_test, salary_predictions[q], 
               alpha=0.5, label=f'Q{int(q*100)}')

plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', label='Perfect Prediction')

plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Quantile Regression Predictions')
plt.legend()

# Add confidence intervals
plt.fill_between(y_test, 
                salary_predictions[0.25], 
                salary_predictions[0.75], 
                alpha=0.2, 
                label='50% Prediction Interval')

plt.show()
```

Slide 8: Model Evaluation Metrics for Quantile Regression

Traditional regression metrics aren't suitable for quantile regression. This implementation introduces specialized metrics for evaluating quantile regression performance, including pinball loss and quantile calibration.

```python
def pinball_loss(y_true, y_pred, quantile):
    """Calculate pinball loss for a specific quantile"""
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

# Calculate pinball loss for each quantile
evaluation_metrics = {}
for q in salary_quantiles:
    loss = pinball_loss(y_test, salary_predictions[q], q)
    evaluation_metrics[f'pinball_loss_q{int(q*100)}'] = loss

# Calculate quantile calibration
def quantile_calibration(y_true, y_pred, quantile):
    """Calculate proportion of true values below predicted quantile"""
    return np.mean(y_true <= y_pred)

calibration_metrics = {
    q: quantile_calibration(y_test, preds, q)
    for q, preds in salary_predictions.items()
}

# Print evaluation results
print("Pinball Loss by Quantile:")
for metric, value in evaluation_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nCalibration by Quantile:")
for q, cal in calibration_metrics.items():
    print(f"Q{int(q*100)}: {cal:.4f}")
```

Slide 9: Handling Non-linear Relationships in Quantile Regression

Non-linear relationships between features and target variables require special attention in quantile regression. This implementation demonstrates how to incorporate non-linear transformations and interaction terms.

```python
# Create polynomial features
from sklearn.preprocessing import PolynomialFeatures

# Generate non-linear features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train non-linear quantile regression model
nonlinear_models = {}
for q in [0.25, 0.5, 0.75]:
    params = {
        'objective': 'quantile',
        'metric': 'quantile',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'alpha': q,
        'feature_fraction': 0.8
    }
    
    train_data = lgb.Dataset(X_train_poly, label=y_train)
    nonlinear_models[q] = lgb.train(params, train_data)

# Generate predictions with non-linear features
nonlinear_predictions = {
    q: model.predict(X_test_poly)
    for q, model in nonlinear_models.items()
}
```

Slide 10: Cross-validation for Quantile Regression

Implementing robust cross-validation for quantile regression requires special consideration of the quantile loss function. This implementation demonstrates how to perform stratified k-fold cross-validation while maintaining quantile predictions.

```python
from sklearn.model_selection import KFold
import numpy as np

def quantile_cv(X, y, quantiles, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = {q: [] for q in quantiles}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        for q in quantiles:
            params = {
                'objective': 'quantile',
                'metric': 'quantile',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'alpha': q
            }
            
            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            model = lgb.train(params, train_data)
            
            # Calculate pinball loss for validation set
            predictions = model.predict(X_val_fold)
            loss = pinball_loss(y_val_fold, predictions, q)
            cv_scores[q].append(loss)
    
    return {q: (np.mean(scores), np.std(scores)) 
            for q, scores in cv_scores.items()}
```

Slide 11: Feature Importance Analysis for Quantile Models

Understanding feature importance across different quantiles provides insights into how variables affect different parts of the distribution. This implementation analyzes and visualizes feature importance for multiple quantiles.

```python
def analyze_feature_importance(models, feature_names, quantiles):
    importance_df = pd.DataFrame()
    
    for q in quantiles:
        # Get feature importance for current quantile
        importance = models[q].feature_importance()
        
        # Create temporary DataFrame
        temp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance,
            'Quantile': f'Q{int(q*100)}'
        })
        
        importance_df = pd.concat([importance_df, temp_df])
    
    # Create feature importance plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df, 
                x='Feature', 
                y='Importance',
                hue='Quantile')
    
    plt.xticks(rotation=45)
    plt.title('Feature Importance by Quantile')
    plt.tight_layout()
    plt.show()
    
    return importance_df

# Example usage
feature_names = ['job_title', 'education', 'experience']
importance_analysis = analyze_feature_importance(
    salary_models, 
    feature_names, 
    salary_quantiles
)
```

Slide 12: Confidence Intervals for Quantile Predictions

Implementing bootstrap-based confidence intervals for quantile regression predictions provides uncertainty estimates around quantile predictions.

```python
def bootstrap_quantile_ci(X, y, n_bootstrap=100, quantiles=[0.25, 0.5, 0.75]):
    n_samples = len(X)
    bootstrap_predictions = {q: [] for q in quantiles}
    
    for _ in range(n_bootstrap):
        # Bootstrap sampling
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot, y_boot = X[indices], y[indices]
        
        # Train models for each quantile
        for q in quantiles:
            params = {
                'objective': 'quantile',
                'metric': 'quantile',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'alpha': q
            }
            
            train_data = lgb.Dataset(X_boot, label=y_boot)
            model = lgb.train(params, train_data)
            
            # Store predictions
            predictions = model.predict(X)
            bootstrap_predictions[q].append(predictions)
    
    # Calculate confidence intervals
    ci_bounds = {}
    for q in quantiles:
        predictions_array = np.array(bootstrap_predictions[q])
        ci_bounds[q] = {
            'lower': np.percentile(predictions_array, 2.5, axis=0),
            'upper': np.percentile(predictions_array, 97.5, axis=0)
        }
    
    return ci_bounds
```

Slide 13: Dynamic Quantile Selection

This implementation introduces an adaptive approach to selecting optimal quantile levels based on the data distribution and prediction requirements.

```python
def optimize_quantile_levels(y, min_samples_per_quantile=100):
    # Calculate optimal number of quantiles based on data size
    n_samples = len(y)
    max_quantiles = n_samples // min_samples_per_quantile
    
    # Generate candidate quantile levels
    candidate_quantiles = np.linspace(0.1, 0.9, max_quantiles)
    
    # Calculate sample density around each quantile
    density_scores = []
    window_size = n_samples // (max_quantiles * 2)
    
    sorted_y = np.sort(y)
    for q in candidate_quantiles:
        q_idx = int(q * n_samples)
        window_density = len(y[
            (y >= sorted_y[max(0, q_idx - window_size)]) & 
            (y <= sorted_y[min(n_samples - 1, q_idx + window_size)])
        ])
        density_scores.append(window_density)
    
    # Select quantiles with highest density
    selected_quantiles = candidate_quantiles[
        np.argsort(density_scores)[-5:]  # Top 5 densest regions
    ]
    
    return np.sort(selected_quantiles)

# Example usage
optimal_quantiles = optimize_quantile_levels(y_train)
print(f"Optimal quantile levels: {optimal_quantiles}")
```

Slide 14: Additional Resources

 * [http://arxiv.org/abs/2011.06693](http://arxiv.org/abs/2011.06693) - "Quantile Regression Forests with Applications in High Dimensions" 
 * [http://arxiv.org/abs/1909.03123](http://arxiv.org/abs/1909.03123) - "Deep Quantile Regression" 
 * [http://arxiv.org/abs/2003.08536](http://arxiv.org/abs/2003.08536) - "Distributional Regression Forests for Probabilistic Precipitation Forecasting" 
 * [http://arxiv.org/abs/1901.09084](http://arxiv.org/abs/1901.09084) - "Statistical Inference for Quantile Regression Neural Networks" 
 * [http://arxiv.org/abs/1803.08084](http://arxiv.org/abs/1803.08084) - "Gradient-Based Quantile Optimization for Machine Learning"

