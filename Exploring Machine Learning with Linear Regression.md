## Exploring Machine Learning with Linear Regression
Slide 1: Understanding Linear Regression Fundamentals

Linear regression serves as the cornerstone of predictive modeling, establishing relationships between dependent and independent variables through a linear equation. This mathematical framework enables us to model real-world relationships and make predictions based on historical data patterns.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate synthetic data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Initialize and train the model
model = LinearRegression()
model.fit(X, y)

# Print model parameters
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Slope: {model.coef_[0][0]:.2f}")

# Visualize the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.show()
```

Slide 2: Mathematical Foundation of Linear Regression

The mathematical basis of linear regression relies on minimizing the sum of squared residuals between predicted and actual values. The optimization problem seeks to find parameters that minimize this error, leading to the best-fit line through our data points.

```python
# Mathematical representation in code block (not rendered)
$$
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n
$$

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

$$
\beta = (X^TX)^{-1}X^Ty
$$
```

Slide 3: Implementation from Scratch

Understanding the inner workings of linear regression requires implementing it from scratch. This implementation showcases the fundamental computations involved in finding the optimal parameters without relying on external libraries.

```python
class LinearRegressionScratch:
    def __init__(self):
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Add bias term
        X_b = np.c_[np.ones((n_samples, 1)), X]
        # Calculate parameters using normal equation
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.bias = theta[0]
        self.weights = theta[1:]
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage
X = np.random.randn(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1
model = LinearRegressionScratch()
model.fit(X, y)
predictions = model.predict(X)
```

Slide 4: Multiple Linear Regression with Real Estate Data

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create sample real estate dataset
data = {
    'price': np.random.normal(200000, 50000, 1000),
    'sqft': np.random.normal(2000, 500, 1000),
    'bedrooms': np.random.randint(2, 6, 1000),
    'bathrooms': np.random.randint(1, 4, 1000),
    'lot_size': np.random.normal(8000, 2000, 1000)
}

df = pd.DataFrame(data)

# Prepare features and target
X = df[['sqft', 'bedrooms', 'bathrooms', 'lot_size']]
y = df['price']

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Print coefficients and score
print("R² Score:", model.score(X_test_scaled, y_test))
for feat, coef in zip(X.columns, model.coef_):
    print(f"{feat}: {coef:.2f}")
```

Slide 5: Model Evaluation and Diagnostics

Understanding model performance requires comprehensive evaluation metrics and diagnostic plots. We analyze residuals, Q-Q plots, and leverage points to ensure our linear regression assumptions are met and identify potential issues.

```python
import statsmodels.api as sm
from scipy import stats

def regression_diagnostics(X, y, model):
    # Fit using statsmodels for detailed diagnostics
    X_with_const = sm.add_constant(X)
    model_sm = sm.OLS(y, X_with_const).fit()
    
    # Get residuals
    residuals = model_sm.resid
    fitted_values = model_sm.fittedvalues
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Residuals vs Fitted
    axes[0,0].scatter(fitted_values, residuals)
    axes[0,0].set_xlabel('Fitted values')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Fitted')
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Normal Q-Q')
    
    # Scale-Location
    axes[1,0].scatter(fitted_values, np.sqrt(np.abs(residuals)))
    axes[1,0].set_xlabel('Fitted values')
    axes[1,0].set_ylabel('√|Residuals|')
    axes[1,0].set_title('Scale-Location')
    
    # Cook's distance
    influence = model_sm.get_influence()
    cooks = influence.cooks_distance[0]
    axes[1,1].stem(range(len(cooks)), cooks)
    axes[1,1].set_title("Cook's Distance")
    
    plt.tight_layout()
    plt.show()
    
    return model_sm.summary()
```

Slide 6: Feature Engineering and Selection

Effective feature engineering transforms raw data into meaningful predictors, while feature selection identifies the most relevant variables. This process is crucial for building robust linear regression models that generalize well.

```python
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures

def engineer_and_select_features(X, y, k=5):
    # Create polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    
    # Feature selection using F-regression
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X_poly, y)
    
    # Get selected feature names
    selected_features_mask = selector.get_support()
    selected_features = feature_names[selected_features_mask]
    
    # Print feature scores
    scores = pd.DataFrame({
        'Feature': feature_names,
        'Score': selector.scores_
    }).sort_values('Score', ascending=False)
    
    return X_selected, scores, selected_features

# Example usage
X_selected, feature_scores, selected_features = engineer_and_select_features(X, y)
print("Top features and their scores:")
print(feature_scores.head())
```

Slide 7: Regularization Techniques

Regularization prevents overfitting by adding penalty terms to the loss function. We explore Ridge (L2), Lasso (L1), and Elastic Net regularization, comparing their effects on model complexity and performance.

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

def compare_regularization(X_train, X_test, y_train, y_test, alphas=[0.1, 1.0, 10.0]):
    results = []
    
    for alpha in alphas:
        # Ridge Regression
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_test)
        
        # Lasso Regression
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)
        lasso_pred = lasso.predict(X_test)
        
        # Elastic Net
        elastic = ElasticNet(alpha=alpha, l1_ratio=0.5)
        elastic.fit(X_train, y_train)
        elastic_pred = elastic.predict(X_test)
        
        # Collect results
        results.append({
            'alpha': alpha,
            'ridge_mse': mean_squared_error(y_test, ridge_pred),
            'lasso_mse': mean_squared_error(y_test, lasso_pred),
            'elastic_mse': mean_squared_error(y_test, elastic_pred),
            'ridge_r2': r2_score(y_test, ridge_pred),
            'lasso_r2': r2_score(y_test, lasso_pred),
            'elastic_r2': r2_score(y_test, elastic_pred)
        })
    
    return pd.DataFrame(results)
```

Slide 8: Cross-Validation and Model Selection

Cross-validation provides a robust method for assessing model performance and selecting optimal hyperparameters. This implementation demonstrates k-fold cross-validation with various linear regression variants.

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def cross_validate_models(X, y, n_splits=5):
    # Initialize cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Create pipelines for different models
    models = {
        'Linear': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))
        ]),
        'Lasso': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Lasso(alpha=1.0))
        ])
    }
    
    # Perform cross-validation for each model
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=kf, 
                               scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        results[name] = {
            'mean_rmse': rmse_scores.mean(),
            'std_rmse': rmse_scores.std(),
            'individual_scores': rmse_scores
        }
    
    return results

# Example usage
cv_results = cross_validate_models(X, y)
for model, scores in cv_results.items():
    print(f"\n{model} Results:")
    print(f"Mean RMSE: {scores['mean_rmse']:.4f}")
    print(f"Std RMSE: {scores['std_rmse']:.4f}")
```

Slide 9: Time Series Forecasting with Linear Regression

Linear regression can be adapted for time series analysis by incorporating temporal features. This implementation shows how to create time-based predictors and handle autocorrelation.

```python
import pandas as pd
from datetime import datetime, timedelta

def create_time_series_features(data, target_col, window_sizes=[1, 7, 30]):
    df = data.copy()
    
    # Create lag features
    for window in window_sizes:
        df[f'lag_{window}'] = df[target_col].shift(window)
    
    # Create rolling mean features
    for window in window_sizes:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(
            window=window).mean()
    
    # Create time-based features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    
    # Drop NaN values created by lagging
    df = df.dropna()
    
    return df

# Create sample time series data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
np.random.seed(42)
values = np.random.normal(100, 10, len(dates))
values += np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 20  # Add seasonality

ts_data = pd.Series(values, index=dates, name='sales')
df = pd.DataFrame(ts_data)

# Prepare features and train model
features_df = create_time_series_features(df, 'sales')
X = features_df.drop('sales', axis=1)
y = features_df['sales']

# Split data temporally
train_size = int(len(features_df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train and evaluate
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")
```

Slide 10: Handling Non-linear Relationships

When relationships between variables are non-linear, we can extend linear regression using polynomial features and splines. This implementation demonstrates how to capture complex patterns while maintaining interpretability.

```python
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import UnivariateSpline

def handle_nonlinearity(X, y, max_degree=3):
    # Polynomial features
    poly = PolynomialFeatures(degree=max_degree)
    X_poly = poly.fit_transform(X)
    
    # Fit models
    linear_model = LinearRegression().fit(X, y)
    poly_model = LinearRegression().fit(X_poly, y)
    
    # Spline regression
    spline = UnivariateSpline(X.ravel(), y, k=3)
    
    # Generate points for plotting
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    
    # Make predictions
    y_linear = linear_model.predict(X_plot)
    y_poly = poly_model.predict(X_plot_poly)
    y_spline = spline(X_plot.ravel())
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.scatter(X, y, color='blue', alpha=0.5, label='Data')
    plt.plot(X_plot, y_linear, 'r-', label='Linear')
    plt.plot(X_plot, y_poly, 'g-', label=f'Polynomial (degree={max_degree})')
    plt.plot(X_plot, y_spline, 'y-', label='Spline')
    plt.legend()
    plt.show()
    
    return linear_model, poly_model, spline

# Generate non-linear data
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = 0.5 * X**2 + X + 2 + np.random.normal(0, 1, X.shape)

# Apply non-linear transformations
linear_model, poly_model, spline_model = handle_nonlinearity(X, y)
```

Slide 11: Robust Regression Techniques

Standard linear regression can be sensitive to outliers. Robust regression methods like Huber and RANSAC provide resistance against outliers while maintaining good statistical properties.

```python
from sklearn.linear_model import HuberRegressor, RANSACRegressor

def compare_robust_methods(X, y, contamination=0.2):
    # Add outliers
    n_outliers = int(contamination * len(X))
    outlier_indices = np.random.choice(len(X), n_outliers, replace=False)
    y_corrupted = y.copy()
    y_corrupted[outlier_indices] += np.random.normal(0, 50, n_outliers)
    
    # Initialize models
    standard_model = LinearRegression()
    huber_model = HuberRegressor(epsilon=1.35)
    ransac_model = RANSACRegressor(random_state=42)
    
    # Fit models
    models = {
        'Standard': standard_model.fit(X, y_corrupted),
        'Huber': huber_model.fit(X, y_corrupted),
        'RANSAC': ransac_model.fit(X, y_corrupted)
    }
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.scatter(X, y_corrupted, c='blue', alpha=0.5, label='Data with outliers')
    
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    colors = ['red', 'green', 'orange']
    
    for (name, model), color in zip(models.items(), colors):
        y_pred = model.predict(X_plot)
        plt.plot(X_plot, y_pred, c=color, label=f'{name} Regression')
    
    plt.legend()
    plt.show()
    
    return models

# Generate data with outliers
X = np.linspace(0, 10, 200).reshape(-1, 1)
y = 2 * X + 1 + np.random.normal(0, 1, X.shape)

# Compare robust methods
robust_models = compare_robust_methods(X, y)
```

Slide 12: Interpretability and Model Insights

Understanding model decisions is crucial for real-world applications. This implementation focuses on extracting insights through feature importance, partial dependence plots, and coefficient analysis.

```python
import shap
from sklearn.inspection import partial_dependence
from sklearn.inspection import PermutationImportance

def analyze_model_insights(model, X, y, feature_names):
    # Calculate feature importances using permutation
    perm = PermutationImportance(model, random_state=42)
    perm.fit(X, y)
    
    # SHAP values
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Coefficient plot
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', ascending=True)
    
    axes[0,0].barh(coef_df['Feature'], coef_df['Coefficient'])
    axes[0,0].set_title('Feature Coefficients')
    
    # Permutation importance
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    axes[0,1].barh(importances['Feature'], importances['Importance'])
    axes[0,1].set_title('Permutation Importance')
    
    # SHAP summary plot
    shap.summary_plot(shap_values, X, feature_names=feature_names, 
                     plot_type='bar', show=False, ax=axes[1,0])
    axes[1,0].set_title('SHAP Feature Importance')
    
    # Partial dependence plot for most important feature
    top_feature = importances.iloc[-1]['Feature']
    pdp = partial_dependence(model, X, [list(feature_names).index(top_feature)])
    axes[1,1].plot(pdp[1][0], pdp[0][0])
    axes[1,1].set_title(f'Partial Dependence Plot: {top_feature}')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'coefficients': coef_df,
        'importances': importances,
        'shap_values': shap_values
    }

# Example usage with your housing dataset
feature_names = ['sqft', 'bedrooms', 'bathrooms', 'lot_size']
insights = analyze_model_insights(model, X_train_scaled, y_train, feature_names)
```

Slide 13: Production Deployment Considerations

Production deployment requires careful handling of model persistence, input validation, and monitoring. This implementation showcases best practices for deploying linear regression models.

```python
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import json

class ModelPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, scaler=None, model=None):
        self.feature_names = feature_names
        self.scaler = scaler or StandardScaler()
        self.model = model or LinearRegression()
        self.feature_ranges = {}
        
    def fit(self, X, y):
        # Calculate feature ranges for validation
        self.feature_ranges = {
            feature: {'min': X[feature].min(), 
                     'max': X[feature].max()}
            for feature in self.feature_names
        }
        
        # Fit pipeline
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def validate_input(self, X):
        if not all(feat in X.columns for feat in self.feature_names):
            raise ValueError("Missing required features")
            
        # Check feature ranges
        for feature, ranges in self.feature_ranges.items():
            if X[feature].min() < ranges['min'] * 0.5 or \
               X[feature].max() > ranges['max'] * 1.5:
                raise ValueError(f"Feature {feature} outside expected range")
    
    def predict(self, X):
        self.validate_input(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, path):
        # Save model and metadata
        pipeline_data = {
            'feature_names': self.feature_names,
            'feature_ranges': self.feature_ranges,
            'scaler': joblib.dump(self.scaler, f"{path}_scaler.joblib"),
            'model': joblib.dump(self.model, f"{path}_model.joblib")
        }
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(pipeline_data, f)
    
    @classmethod
    def load(cls, path):
        # Load model and metadata
        with open(f"{path}_metadata.json", 'r') as f:
            pipeline_data = json.load(f)
        
        instance = cls(
            feature_names=pipeline_data['feature_names'],
            scaler=joblib.load(f"{path}_scaler.joblib"),
            model=joblib.load(f"{path}_model.joblib")
        )
        instance.feature_ranges = pipeline_data['feature_ranges']
        return instance

# Example usage
pipeline = ModelPipeline(feature_names)
pipeline.fit(X_train, y_train)
pipeline.save('model/housing_model')
```

Slide 14: Additional Resources

*   "Deep Learning vs Linear Regression for Time Series" - [https://arxiv.org/abs/2008.07669](https://arxiv.org/abs/2008.07669)
*   "Robust Linear Regression: A Review and Comparison" - [https://arxiv.org/abs/1404.6274](https://arxiv.org/abs/1404.6274)
*   "On the Convergence of Linear Regression Algorithms" - [https://arxiv.org/abs/1509.09169](https://arxiv.org/abs/1509.09169)
*   "Feature Selection Methods in Linear Regression: A Survey" - [https://scholar.google.com/](https://scholar.google.com/)
*   "Interpretable Machine Learning with Linear Models" - [https://research.google/pubs/](https://research.google/pubs/)

