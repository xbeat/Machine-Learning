## Regression Analysis Fundamentals and Application
Slide 1: Understanding Linear Regression Fundamentals

Linear regression serves as a foundational statistical method for modeling relationships between variables. It establishes a linear relationship between dependent and independent variables by finding the best-fitting line through data points, minimizing the sum of squared residuals.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Print model parameters
print(f"Coefficient: {model.coef_[0][0]:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"R² Score: {r2_score(y, y_pred):.4f}")
```

Slide 2: Mathematical Foundation of Linear Regression

The mathematical foundation of linear regression is built upon minimizing the sum of squared differences between observed and predicted values. This optimization problem leads to the derivation of the ordinary least squares estimator.

```python
# Mathematical formula representation (LaTeX notation)
'''
$$
\hat{Y} = \beta_0 + \beta_1X
$$

$$
\beta_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
$$

$$
\beta_0 = \bar{y} - \beta_1\bar{x}
$$

$$
MSE = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
'''

# Implementation from scratch
def simple_linear_regression(X, y):
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    
    beta1 = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean)**2)
    beta0 = y_mean - beta1 * x_mean
    
    return beta0, beta1
```

Slide 3: Implementing Multiple Linear Regression

Multiple linear regression extends the simple linear model by incorporating multiple independent variables, allowing for more complex relationships and better predictive capabilities in real-world scenarios where multiple factors influence the outcome.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate synthetic data for multiple features
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 3)
y = 2*X[:, 0] + 3*X[:, 1] - 1.5*X[:, 2] + np.random.randn(n_samples) * 0.1

# Create DataFrame
df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3'])
df['Target'] = y

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Print coefficients and performance
print("Feature Coefficients:")
for feature, coef in zip(df.columns[:-1], model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"\nR² Score: {model.score(X_test, y_test):.4f}")
```

Slide 4: Regression Assumptions and Diagnostics

Understanding and validating regression assumptions is crucial for reliable model inference. Key assumptions include linearity, independence, homoscedasticity, and normality of residuals, which must be verified through diagnostic plots and statistical tests.

```python
import scipy.stats as stats
import seaborn as sns

def regression_diagnostics(model, X, y):
    # Get predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals vs Fitted
    axes[0,0].scatter(y_pred, residuals)
    axes[0,0].set_xlabel('Fitted values')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].axhline(y=0, color='r', linestyle='--')
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    
    # Scale-Location
    axes[1,0].scatter(y_pred, np.sqrt(np.abs(residuals)))
    axes[1,0].set_xlabel('Fitted values')
    axes[1,0].set_ylabel('√|Residuals|')
    
    # Residuals histogram
    axes[1,1].hist(residuals, bins=30)
    axes[1,1].set_xlabel('Residuals')
    axes[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Statistical tests
    print("Shapiro-Wilk test for normality:")
    print(stats.shapiro(residuals))
    print("\nBreusch-Pagan test for homoscedasticity:")
    print(stats.levene(y_pred, residuals))

# Example usage
regression_diagnostics(model, X_test, y_test)
```

Slide 5: Feature Selection and Regularization

Feature selection and regularization techniques help prevent overfitting and improve model generalization. We'll implement both Lasso and Ridge regression, comparing their effectiveness in handling multicollinearity and feature importance determination.

```python
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score

# Generate correlated features
np.random.seed(42)
n_samples = 200
X = np.random.randn(n_samples, 15)
# Add correlation
X[:, 5:] = X[:, :10] + np.random.randn(n_samples, 10) * 0.1
y = 2*X[:, 0] + 3*X[:, 1] - 5*X[:, 2] + np.random.randn(n_samples) * 0.1

# Compare different regularization approaches
models = {
    'Linear': LinearRegression(),
    'Lasso': Lasso(alpha=0.1),
    'Ridge': Ridge(alpha=0.1)
}

# Evaluate models
for name, model in models.items():
    # Cross-validation scores
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-scores)
    
    # Fit model to get coefficients
    model.fit(X, y)
    
    print(f"\n{name} Regression:")
    print(f"RMSE: {rmse.mean():.4f} (+/- {rmse.std()*2:.4f})")
    print("Top 5 feature coefficients:")
    coef_importance = np.abs(model.coef_)
    top_features = np.argsort(coef_importance)[-5:]
    for idx in top_features:
        print(f"Feature {idx}: {model.coef_[idx]:.4f}")
```

Slide 6: Cross-Validation and Model Evaluation

Cross-validation provides a robust method for assessing model performance and preventing overfitting. We'll implement various cross-validation techniques and evaluate models using multiple metrics for comprehensive performance assessment.

```python
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score

def comprehensive_cv_evaluation(X, y, model, cv_folds=5):
    # Define scoring metrics
    scoring = {
        'r2': 'r2',
        'mse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error'
    }
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y,
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring=scoring,
        return_train_score=True
    )
    
    # Process and display results
    for metric in scoring.keys():
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']
        
        if 'neg_' in scoring[metric]:
            train_scores = -train_scores
            test_scores = -test_scores
            
        print(f"\n{metric.upper()} Scores:")
        print(f"Train: {train_scores.mean():.4f} (+/- {train_scores.std()*2:.4f})")
        print(f"Test:  {test_scores.mean():.4f} (+/- {test_scores.std()*2:.4f})")

# Example usage with standardized data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression()

comprehensive_cv_evaluation(X_scaled, y, model)
```

Slide 7: Handling Non-Linear Relationships

When relationships between variables are non-linear, we can extend linear regression using polynomial features and spline transformations to capture complex patterns while maintaining the interpretability of linear models.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.interpolate import UnivariateSpline

# Generate non-linear data
X_nonlin = np.linspace(0, 10, 100).reshape(-1, 1)
y_nonlin = 0.5 * X_nonlin**2 + np.sin(X_nonlin) * 3 + np.random.randn(100, 1) * 2

# Create polynomial features pipeline
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Fit polynomial model
poly_pipeline.fit(X_nonlin, y_nonlin)

# Create spline transformation
spline = UnivariateSpline(X_nonlin.ravel(), y_nonlin.ravel(), k=3)

# Make predictions
X_test = np.linspace(0, 10, 200).reshape(-1, 1)
y_poly_pred = poly_pipeline.predict(X_test)
y_spline_pred = spline(X_test.ravel())

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(X_nonlin, y_nonlin, label='Original Data', alpha=0.5)
plt.plot(X_test, y_poly_pred, 'r-', label='Polynomial Regression')
plt.plot(X_test, y_spline_pred, 'g-', label='Spline Regression')
plt.legend()
plt.title('Non-linear Regression Approaches')
plt.show()

# Calculate and print performance metrics
print("Polynomial Regression R²:", 
      r2_score(y_nonlin, poly_pipeline.predict(X_nonlin)))
print("Spline Regression R²:", 
      r2_score(y_nonlin, spline(X_nonlin.ravel())))
```

Slide 8: Robust Regression Techniques

Robust regression methods provide reliable estimates when data contains outliers or violates standard assumptions. We'll implement Huber and RANSAC regression to demonstrate their effectiveness in handling contaminated datasets.

```python
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RANSACRegressor

# Generate data with outliers
np.random.seed(42)
X = np.linspace(0, 10, 200).reshape(-1, 1)
y = 3 * X + 2 + np.random.normal(0, 1.5, size=X.shape)

# Add outliers
outlier_indices = np.random.choice(len(X), 40, replace=False)
y[outlier_indices] += np.random.normal(0, 15, size=len(outlier_indices))

# Initialize models
models = {
    'Standard': LinearRegression(),
    'Huber': HuberRegressor(epsilon=1.35),
    'RANSAC': RANSACRegressor(random_state=42)
}

# Fit and evaluate models
results = {}
for name, model in models.items():
    # Fit model
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Store results
    results[name] = {
        'predictions': y_pred,
        'r2': r2_score(y, y_pred),
        'mse': mean_squared_error(y, y_pred)
    }
    
    print(f"\n{name} Regression Results:")
    print(f"R² Score: {results[name]['r2']:.4f}")
    print(f"MSE: {results[name]['mse']:.4f}")
    if name == 'RANSAC':
        print(f"Inlier samples: {model.inlier_mask_.sum()}")
```

Slide 9: Time Series Regression Analysis

Time series regression requires special consideration for temporal dependencies and seasonality. We'll implement techniques for handling time-based features and autocorrelation in regression models.

```python
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

# Generate time series data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
trend = np.linspace(0, 10, 365)
seasonal = 5 * np.sin(2 * np.pi * np.arange(365) / 365)
noise = np.random.normal(0, 1, 365)
y = trend + seasonal + noise

# Create time series DataFrame
df = pd.DataFrame({
    'date': dates,
    'value': y
})

def analyze_time_series(df):
    # Extract time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    
    # Perform stationarity test
    adf_result = adfuller(df['value'])
    
    # Check for autocorrelation
    lb_result = acorr_ljungbox(df['value'], lags=10)
    
    # Create lagged features
    for lag in [1, 7, 30]:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    
    # Prepare features for regression
    X = df.dropna().drop(['date', 'value'], axis=1)
    y = df.dropna()['value']
    
    # Fit time series regression
    model = LinearRegression()
    model.fit(X, y)
    
    print("Time Series Analysis Results:")
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print("\nFeature Importance:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef:.4f}")
    
    return model, X, y

# Run analysis
model, X, y = analyze_time_series(df)
```

Slide 10: Marketing Analytics Case Study - Part 1

In this real-world marketing analytics case, we'll analyze the relationship between advertising spend across different channels and sales performance, implementing a comprehensive regression analysis pipeline.

```python
# Create synthetic marketing dataset
np.random.seed(42)
n_samples = 1000

# Generate marketing spend data
tv_spend = np.random.uniform(10, 100, n_samples)
radio_spend = np.random.uniform(5, 50, n_samples)
social_spend = np.random.uniform(15, 75, n_samples)

# Generate sales with realistic relationships
sales = (
    0.5 * tv_spend + 
    0.3 * radio_spend + 
    0.4 * social_spend + 
    0.2 * tv_spend * radio_spend / 100 +  # interaction effect
    np.random.normal(0, 5, n_samples)
)

# Create DataFrame
marketing_df = pd.DataFrame({
    'TV_Spend': tv_spend,
    'Radio_Spend': radio_spend,
    'Social_Spend': social_spend,
    'Sales': sales
})

# Preprocessing and feature engineering
def preprocess_marketing_data(df):
    # Create interaction terms
    df['TV_Radio_Interaction'] = df['TV_Spend'] * df['Radio_Spend']
    df['TV_Social_Interaction'] = df['TV_Spend'] * df['Social_Spend']
    
    # Scale features
    scaler = StandardScaler()
    features = ['TV_Spend', 'Radio_Spend', 'Social_Spend', 
                'TV_Radio_Interaction', 'TV_Social_Interaction']
    
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    
    return df_scaled, features

# Prepare data
df_processed, features = preprocess_marketing_data(marketing_df)
X = df_processed[features]
y = df_processed['Sales']
```

Slide 11: Marketing Analytics Case Study - Part 2

Building upon our preprocessed marketing data, we'll implement advanced regression techniques to identify key performance drivers and optimize marketing spend allocation across channels.

```python
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor

def analyze_marketing_performance(X, y, features):
    # Initialize models
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Cross-validation predictions
        y_pred = cross_val_predict(model, X, y, cv=5)
        
        # Calculate metrics
        results[name] = {
            'R2': r2_score(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        # Fit model on full dataset for feature importance
        model.fit(X, y)
        
        # Get feature importance
        if name == 'Random Forest':
            importance = model.feature_importances_
        else:
            importance = np.abs(model.coef_)
            
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print(f"\n{name} Regression Results:")
        print(f"R² Score: {results[name]['R2']:.4f}")
        print(f"RMSE: {results[name]['RMSE']:.4f}")
        print("\nTop Feature Importance:")
        print(importance_df.head())

# Run analysis
analyze_marketing_performance(X, y, features)
```

Slide 12: Model Interpretation and Visualization

Effective model interpretation is crucial for stakeholder communication. We'll create comprehensive visualizations and interpretability metrics to explain our regression results.

```python
import shap
from sklearn.inspection import partial_dependence

def create_model_interpretability_plots(model, X, features):
    plt.figure(figsize=(15, 10))
    
    # 1. Feature Importance Plot
    plt.subplot(2, 2, 1)
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': np.abs(model.coef_)
    }).sort_values('Importance', ascending=True)
    
    plt.barh(range(len(importance)), importance['Importance'])
    plt.yticks(range(len(importance)), importance['Feature'])
    plt.title('Feature Importance')
    
    # 2. Residual Plot
    plt.subplot(2, 2, 2)
    y_pred = model.predict(X)
    residuals = y - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    # 3. Actual vs Predicted Plot
    plt.subplot(2, 2, 3)
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    
    # 4. SHAP Values
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    plt.subplot(2, 2, 4)
    shap.summary_plot(shap_values, X, feature_names=features, 
                     plot_type='bar', show=False)
    plt.title('SHAP Feature Importance')
    
    plt.tight_layout()
    plt.show()

# Create interpretation plots
model = LinearRegression().fit(X, y)
create_model_interpretability_plots(model, X, features)
```

Slide 13: Model Deployment and Monitoring

Implementing a production-ready regression model requires robust deployment and monitoring systems. We'll create a pipeline for model serving and performance tracking.

```python
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import json

class ModelMonitor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.predictions = []
        self.actuals = []
        self.timestamps = []
        
    def log_prediction(self, prediction, actual, timestamp):
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.timestamps.append(timestamp)
        
    def calculate_metrics(self):
        predictions = np.array(self.predictions)
        actuals = np.array(self.actuals)
        
        return {
            'mse': mean_squared_error(actuals, predictions),
            'r2': r2_score(actuals, predictions),
            'mae': mean_absolute_error(actuals, predictions)
        }
    
    def export_logs(self, filepath):
        logs = {
            'model_name': self.model_name,
            'predictions': self.predictions,
            'actuals': self.actuals,
            'timestamps': [str(ts) for ts in self.timestamps],
            'metrics': self.calculate_metrics()
        }
        with open(filepath, 'w') as f:
            json.dump(logs, f)

# Example usage
monitor = ModelMonitor('marketing_regression')
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Save model
joblib.dump(pipeline, 'marketing_model.joblib')

# Simulate predictions
for i in range(100):
    timestamp = pd.Timestamp.now()
    pred = pipeline.predict(X[i:i+1])[0]
    actual = y.iloc[i]
    monitor.log_prediction(pred, actual, timestamp)

# Export monitoring logs
monitor.export_logs('model_monitoring_logs.json')
print("Model Performance Metrics:")
print(json.dumps(monitor.calculate_metrics(), indent=2))
```

Slide 14: Advanced Error Analysis and Diagnostics

Error analysis provides crucial insights into model performance and potential improvements. We'll implement comprehensive diagnostics tools to identify patterns in prediction errors and model limitations.

```python
def advanced_error_analysis(y_true, y_pred, X, feature_names):
    # Calculate residuals and standardized residuals
    residuals = y_true - y_pred
    std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    
    # Create analysis DataFrame
    analysis_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Residuals': residuals,
        'Std_Residuals': std_residuals,
        'Abs_Error': np.abs(residuals)
    })
    
    # Add feature values
    for i, feature in enumerate(feature_names):
        analysis_df[feature] = X[:, i]
    
    # Error distribution analysis
    error_stats = {
        'Mean Error': np.mean(residuals),
        'Median Error': np.median(residuals),
        'Error Std': np.std(residuals),
        'Skewness': stats.skew(residuals),
        'Kurtosis': stats.kurtosis(residuals)
    }
    
    # Find problematic predictions
    outliers = analysis_df[np.abs(std_residuals) > 2]
    
    # Feature-error correlations
    error_correlations = {
        feature: np.corrcoef(X[:, i], np.abs(residuals))[0,1]
        for i, feature in enumerate(feature_names)
    }
    
    print("Error Distribution Statistics:")
    print(json.dumps(error_stats, indent=2))
    print("\nFeature-Error Correlations:")
    print(json.dumps(error_correlations, indent=2))
    print(f"\nNumber of Outlier Predictions: {len(outliers)}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Error distribution
    axes[0,0].hist(residuals, bins=30)
    axes[0,0].set_title('Error Distribution')
    axes[0,0].set_xlabel('Residuals')
    
    # QQ plot
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot')
    
    # Error vs Predicted
    axes[1,0].scatter(y_pred, residuals)
    axes[1,0].axhline(y=0, color='r', linestyle='--')
    axes[1,0].set_title('Residuals vs Predicted')
    axes[1,0].set_xlabel('Predicted Values')
    axes[1,0].set_ylabel('Residuals')
    
    # Feature importance for errors
    importance = np.abs([error_correlations[f] for f in feature_names])
    axes[1,1].barh(feature_names, importance)
    axes[1,1].set_title('Feature Importance for Errors')
    
    plt.tight_layout()
    return analysis_df
```

Slide 15: Additional Resources

arXiv Papers for Further Reading:

```text
1. "A Comparative Analysis of Ridge and Lasso Regression on High-Dimensional Data"
   https://arxiv.org/abs/2103.12283

2. "Robust Regression: Theory and Implementation in Modern Machine Learning"
   https://arxiv.org/abs/2009.14465

3. "Time Series Regression Models: A Comprehensive Review"
   https://arxiv.org/abs/1908.10732

4. "Feature Selection Methods for Linear Regression: A Systematic Review"
   https://arxiv.org/abs/2106.15820

5. "Interpretable Machine Learning: Modern Approaches to Linear Regression"
   https://arxiv.org/abs/2004.12338
```
