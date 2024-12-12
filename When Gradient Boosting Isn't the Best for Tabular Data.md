## When Gradient Boosting Isn't the Best for Tabular Data
Slide 1: Linear Relationships in Tabular Data

Understanding when linear models outperform gradient boosting requires analyzing feature-target relationships through correlation analysis and visualizations. Linear regression provides interpretable coefficients and faster training when relationships are predominantly linear.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns

# Generate synthetic linear data
np.random.seed(42)
X = np.random.randn(1000, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(1000) * 0.1

# Check linearity with correlation matrix
df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
df['target'] = y

# Fit linear model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print(f"R² Score: {r2_score(y, y_pred):.4f}")
print("\nFeature Coefficients:")
for feat, coef in zip(['feature1', 'feature2', 'feature3'], model.coef_):
    print(f"{feat}: {coef:.4f}")
```

Slide 2: Handling Noisy and Sparse Data

When dealing with noisy and sparse tabular data, simpler models often provide better generalization. This implementation demonstrates how to identify and handle sparse features while maintaining model performance.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Generate sparse data
n_samples = 1000
n_features = 50
X_sparse = np.zeros((n_samples, n_features))
X_sparse[np.random.randint(0, n_samples, 100), np.random.randint(0, n_features, 100)] = 1

# Add noise
noise = np.random.normal(0, 0.1, (n_samples, n_features))
X_noisy = X_sparse + noise

# Calculate sparsity
sparsity = 1.0 - np.count_nonzero(X_sparse) / X_sparse.size
print(f"Data sparsity: {sparsity:.2%}")

# Fit Ridge regression with regularization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_noisy)
model = Ridge(alpha=1.0)
model.fit(X_scaled, y)
```

Slide 3: Neural Networks for Extrapolation

Neural networks excel at capturing complex patterns and extrapolating beyond training data ranges. This implementation creates a custom neural network architecture specifically designed for tabular data extrapolation.

```python
import torch
import torch.nn as nn

class TabularNN(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        return self.model(x)

# Initialize model
model = TabularNN(input_size=10)

# Training loop setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Example training iteration
x = torch.randn(32, 10)  # batch_size=32, features=10
y = torch.randn(32, 1)
optimizer.zero_grad()
output = model(x)
loss = criterion(output, y)
loss.backward()
optimizer.step()
```

Slide 4: Quick Non-Linear Baseline with Random Forest

Random Forests provide an excellent alternative to gradient boosting when quick model iteration is needed. This implementation showcases automatic hyperparameter tuning and feature importance analysis for rapid model development.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Initialize RandomForestRegressor with parameter grid
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist,
    n_iter=10, cv=5, random_state=42, n_jobs=-1
)

# Fit and evaluate
X = np.random.randn(1000, 10)
y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.randn(1000) * 0.1
random_search.fit(X, y)

# Get feature importance
importances = random_search.best_estimator_.feature_importances_
print("Best parameters:", random_search.best_params_)
print("\nFeature importances:", importances)
```

Slide 5: Gaussian Process for Optimization Tasks

Gaussian Processes provide smooth, differentiable predictions ideal for optimization tasks. This implementation demonstrates GP regression with uncertainty estimation and acquisition function optimization.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Define kernel
kernel = C(1.0) * RBF([1.0])

# Initialize and fit GP
gp = GaussianProcessRegressor(kernel=kernel, random_state=42)

# Generate sample data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.normal(0, 0.1, X.shape[0])

# Fit GP
gp.fit(X, y)

# Predict with uncertainty
X_test = np.linspace(-2, 12, 200).reshape(-1, 1)
y_pred, sigma = gp.predict(X_test, return_std=True)

# Define acquisition function (Expected Improvement)
def expected_improvement(X, gp, y_best):
    mean, std = gp.predict(X.reshape(-1, 1), return_std=True)
    z = (mean - y_best) / std
    ei = (mean - y_best) * norm.cdf(z) + std * norm.pdf(z)
    return ei

# Find next point to evaluate
y_best = y.max()
ei = expected_improvement(X_test, gp, y_best)
next_point = X_test[ei.argmax()]
```

Slide 6: Implementing Splines for Smooth Interpolation

Spline regression offers smooth interpolation capabilities while maintaining interpretability. This implementation shows how to use B-splines for complex non-linear relationships.

```python
from scipy.interpolate import BSpline
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class SplineRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, degree=3, n_knots=5):
        self.degree = degree
        self.n_knots = n_knots
        
    def fit(self, X, y):
        # Generate knot sequence
        x = X.ravel()
        knots = np.linspace(x.min(), x.max(), self.n_knots)
        
        # Fit B-spline
        self.spline = BSpline.fit(x, y, self.degree, knots)
        return self
        
    def predict(self, X):
        return self.spline(X.ravel())

# Example usage
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.normal(0, 0.1, 100)

spline_reg = SplineRegressor(degree=3, n_knots=10)
spline_reg.fit(X, y)
y_pred = spline_reg.predict(X)

print(f"MSE: {np.mean((y - y_pred)**2):.4f}")
```

Slide 7: Data Preprocessing for Linear Models

Effective preprocessing is crucial when using linear models as alternatives to gradient boosting. This implementation demonstrates robust scaling, outlier handling, and feature engineering techniques optimized for linear modeling.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class RobustPreprocessor:
    def __init__(self, polynomial_degree=2, interaction_only=True):
        self.polynomial_degree = polynomial_degree
        self.interaction_only = interaction_only
        
    def create_pipeline(self, numeric_features, categorical_features):
        numeric_transformer = Pipeline(steps=[
            ('scaler', RobustScaler()),
            ('poly', PolynomialFeatures(
                degree=self.polynomial_degree, 
                interaction_only=self.interaction_only,
                include_bias=False
            ))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ]
        )
        
        return preprocessor

# Example usage
np.random.seed(42)
n_samples = 1000
X = pd.DataFrame({
    'feature1': np.random.normal(0, 1, n_samples),
    'feature2': np.random.normal(0, 1, n_samples),
    'feature3': np.random.normal(0, 1, n_samples)
})

# Add outliers
X.loc[0:10, 'feature1'] = 100

preprocessor = RobustPreprocessor(polynomial_degree=2)
pipeline = preprocessor.create_pipeline(
    numeric_features=['feature1', 'feature2', 'feature3'],
    categorical_features=[]
)

X_transformed = pipeline.fit_transform(X)
print(f"Original shape: {X.shape}, Transformed shape: {X_transformed.shape}")
```

Slide 8: Model Selection Strategy

This implementation provides a systematic approach to choosing between linear models and gradient boosting using cross-validation and statistical tests for model comparison.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats

class ModelSelector:
    def __init__(self, significance_level=0.05):
        self.significance_level = significance_level
        
    def compare_models(self, X, y, cv=5):
        # Initialize models
        linear_model = LassoCV(cv=5)
        gb_model = GradientBoostingRegressor(random_state=42)
        
        # Get cross-validation scores
        linear_scores = cross_val_score(linear_model, X, y, cv=cv)
        gb_scores = cross_val_score(gb_model, X, y, cv=cv)
        
        # Perform statistical test
        t_stat, p_value = stats.ttest_rel(linear_scores, gb_scores)
        
        results = {
            'linear_mean': linear_scores.mean(),
            'linear_std': linear_scores.std(),
            'gb_mean': gb_scores.mean(),
            'gb_std': gb_scores.std(),
            'p_value': p_value,
            'significant_difference': p_value < self.significance_level
        }
        
        return results

# Example usage
X = np.random.randn(1000, 5)
y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(1000) * 0.1

selector = ModelSelector()
results = selector.compare_models(X, y)
print("Model Comparison Results:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")
```

Slide 9: Feature Importance Analysis

Understanding feature importance helps determine when simpler models might be more appropriate. This implementation provides multiple methods for analyzing feature relationships and importance.

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

class FeatureAnalyzer:
    def __init__(self, n_repeats=10):
        self.n_repeats = n_repeats
        
    def analyze_features(self, X, y):
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Lasso coefficients
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_scaled, y)
        
        # Permutation importance
        perm_importance = permutation_importance(
            lasso, X_scaled, y, 
            n_repeats=self.n_repeats
        )
        
        # Calculate feature correlations
        correlations = np.corrcoef(X_scaled.T)
        
        return {
            'lasso_coefficients': lasso.coef_,
            'permutation_importance': perm_importance.importances_mean,
            'feature_correlations': correlations
        }

# Example usage
X = np.random.randn(1000, 5)
y = 2*X[:, 0] + 3*X[:, 1] + np.random.randn(1000) * 0.1

analyzer = FeatureAnalyzer()
results = analyzer.analyze_features(X, y)

print("Lasso Coefficients:", results['lasso_coefficients'])
print("\nPermutation Importance:", results['permutation_importance'])
```

Slide 10: Real-World Application - Credit Risk Modeling

Implementation of a credit risk prediction system demonstrating when linear models outperform gradient boosting in a highly regulated environment requiring model interpretability.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

class CreditRiskModel:
    def __init__(self, regularization='l1'):
        self.model = LogisticRegression(
            penalty=regularization,
            solver='liblinear',
            random_state=42
        )
        
    def prepare_features(self, X):
        # Calculate financial ratios
        X['debt_to_income'] = X['total_debt'] / (X['income'] + 1e-6)
        X['payment_to_income'] = X['monthly_payment'] / (X['income'] + 1e-6)
        
        # Log transform skewed features
        for col in ['income', 'total_debt']:
            X[f'{col}_log'] = np.log1p(X[col])
            
        return X
    
    def get_feature_importance(self):
        return pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0],
            'odds_ratio': np.exp(self.model.coef_[0])
        })

# Generate synthetic credit data
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'income': np.random.lognormal(10, 1, n_samples),
    'total_debt': np.random.lognormal(9, 1.5, n_samples),
    'monthly_payment': np.random.lognormal(6, 0.5, n_samples),
    'credit_score': np.random.normal(650, 100, n_samples)
})

# Create target variable
data['default'] = (data['total_debt'] / data['income'] > 0.5).astype(int)

# Train model
model = CreditRiskModel()
X = model.prepare_features(data.drop('default', axis=1))
y = data['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.feature_names = X.columns
model.model.fit(X_train, y_train)

# Evaluate
y_pred_proba = model.model.predict_proba(X_test)[:, 1]
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

Slide 11: Performance Monitoring System

This implementation creates a monitoring system to detect when model performance degrades and determines if switching from gradient boosting to simpler models is warranted.

```python
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

class ModelMonitor:
    def __init__(self, baseline_metrics, window_size=30):
        self.baseline_metrics = baseline_metrics
        self.window_size = window_size
        self.performance_history = []
        
    def add_daily_metrics(self, date, metrics):
        self.performance_history.append({
            'date': date,
            'metrics': metrics
        })
        
        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)
            
    def detect_degradation(self, threshold=0.05):
        if len(self.performance_history) < self.window_size:
            return False, None
            
        recent_metrics = [p['metrics'] for p in self.performance_history]
        
        # Perform statistical tests
        t_stat, p_value = stats.ttest_1samp(
            recent_metrics,
            self.baseline_metrics
        )
        
        degradation_detected = p_value < threshold and t_stat < 0
        
        analysis = {
            'degradation_detected': degradation_detected,
            'p_value': p_value,
            't_statistic': t_stat,
            'current_mean': np.mean(recent_metrics),
            'baseline_mean': self.baseline_metrics
        }
        
        return degradation_detected, analysis

# Example usage
monitor = ModelMonitor(baseline_metrics=0.85)

# Simulate 60 days of metrics
np.random.seed(42)
start_date = datetime(2024, 1, 1)
for i in range(60):
    date = start_date + timedelta(days=i)
    # Simulate gradual degradation
    metric = 0.85 - (i/200) + np.random.normal(0, 0.02)
    monitor.add_daily_metrics(date, metric)
    
    if (i + 1) % 30 == 0:
        degraded, analysis = monitor.detect_degradation()
        print(f"\nDay {i+1} Analysis:")
        for key, value in analysis.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
```

Slide 12: Time Series Feature Engineering

Advanced feature engineering techniques specifically designed for time series data when linear models are preferred over gradient boosting due to temporal dependencies.

```python
import pandas as pd
import numpy as np
from scipy.stats import skew
from statsmodels.tsa.seasonal import seasonal_decompose

class TimeSeriesFeatureEngineer:
    def __init__(self, window_sizes=[7, 14, 30]):
        self.window_sizes = window_sizes
        
    def create_features(self, df, target_col):
        features = pd.DataFrame(index=df.index)
        
        # Rolling statistics
        for window in self.window_sizes:
            features[f'roll_mean_{window}'] = df[target_col].rolling(window).mean()
            features[f'roll_std_{window}'] = df[target_col].rolling(window).std()
            features[f'roll_skew_{window}'] = df[target_col].rolling(window).apply(skew)
        
        # Seasonal decomposition
        decomposition = seasonal_decompose(
            df[target_col], 
            period=min(self.window_sizes),
            extrapolate_trend='freq'
        )
        
        features['trend'] = decomposition.trend
        features['seasonal'] = decomposition.seasonal
        features['residual'] = decomposition.resid
        
        # Lag features
        for lag in self.window_sizes:
            features[f'lag_{lag}'] = df[target_col].shift(lag)
            
        return features.fillna(method='bfill')

# Example usage
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
data = pd.DataFrame({
    'date': dates,
    'value': np.random.normal(0, 1, 365) + np.sin(np.linspace(0, 4*np.pi, 365))
}).set_index('date')

engineer = TimeSeriesFeatureEngineer()
features = engineer.create_features(data, 'value')

print("Generated features shape:", features.shape)
print("\nFeature columns:", features.columns.tolist())
```

Slide 13: Model Interpretability Framework

Comprehensive framework for interpreting and explaining linear model predictions with techniques specifically designed for regulatory compliance.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class InterpretableModel:
    def __init__(self):
        self.model = LogisticRegression(penalty='l2')
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit(self, X, y):
        self.feature_names = X.columns
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
        
    def get_feature_importance(self):
        coefficients = self.model.coef_[0]
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_importance': np.abs(coefficients)
        }).sort_values('abs_importance', ascending=False)
        
        return importance_df
        
    def explain_prediction(self, X_sample):
        X_scaled = self.scaler.transform(X_sample)
        contribution = X_scaled * self.model.coef_[0]
        
        explanation_df = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_sample.iloc[0],
            'scaled_value': X_scaled[0],
            'coefficient': self.model.coef_[0],
            'contribution': contribution[0]
        })
        
        explanation_df['contribution_pct'] = (
            explanation_df['contribution'].abs() / 
            explanation_df['contribution'].abs().sum() * 100
        )
        
        return explanation_df.sort_values('contribution_pct', ascending=False)

# Example usage
np.random.seed(42)
X = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(0, 1, 1000),
    'feature3': np.random.normal(0, 1, 1000)
})
y = (X['feature1'] + 2*X['feature2'] - X['feature3'] > 0).astype(int)

model = InterpretableModel()
model.fit(X, y)

print("Feature Importance:")
print(model.get_feature_importance())

print("\nSample Prediction Explanation:")
print(model.explain_prediction(X.iloc[[0]]))
```

Slide 14: Additional Resources

*   Machine Learning for Tabular Data: A Critical Analysis - [https://arxiv.org/abs/2110.01889](https://arxiv.org/abs/2110.01889)
*   Beyond Gradient Boosting: When Linear Models Win - [https://arxiv.org/abs/2012.01315](https://arxiv.org/abs/2012.01315)
*   Linear Models vs Tree-based Models: A Comprehensive Study - [https://arxiv.org/abs/2103.11869](https://arxiv.org/abs/2103.11869)
*   Search Suggestions:
    *   "Linear Models vs Gradient Boosting Performance Comparison"
    *   "When to Use Simple Models in Machine Learning"
    *   "Model Selection for Tabular Data"
*   Books:
    *   "Introduction to Statistical Learning"
    *   "Elements of Statistical Learning"

