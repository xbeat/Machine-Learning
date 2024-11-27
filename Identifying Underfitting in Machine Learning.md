## Identifying Underfitting in Machine Learning
Slide 1: Understanding Underfitting in Machine Learning

High bias or underfitting occurs when a model performs poorly on both training and test datasets, indicating the model is too simple to capture the underlying patterns in the data. This fundamental concept affects model selection and optimization strategies.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate non-linear data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

# Fit underfitting model (linear)
model = LinearRegression()
model.fit(X, y)

# Calculate errors
train_mse = mean_squared_error(y, model.predict(X))
print(f"Training MSE: {train_mse:.4f}")  # High error indicates underfitting

# Plot results
plt.scatter(X, y, label='True Data')
plt.plot(X, model.predict(X), color='red', label='Linear Model')
plt.title('Underfitting Example: Linear Model on Non-linear Data')
plt.legend()
```

Slide 2: Detecting Underfitting Through Learning Curves

Learning curves provide visual insight into model underfitting by plotting training and validation errors against training set size. Parallel high error curves indicate underfitting, showing the model lacks complexity to capture data patterns.

```python
from sklearn.model_selection import learning_curve

def plot_learning_curves(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='neg_mean_squared_error'
    )
    
    train_mean = -np.mean(train_scores, axis=1)
    val_mean = -np.mean(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Error')
    plt.plot(train_sizes, val_mean, label='Validation Error')
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curves: Underfitting Analysis')
    plt.legend()
    plt.grid(True)

# Plot learning curves for linear model
plot_learning_curves(LinearRegression(), X, y)
```

Slide 3: Quantifying Underfitting with Bias-Variance Analysis

Understanding the bias-variance decomposition helps quantify underfitting through mathematical analysis. High bias indicates systematic model error, while low variance suggests consistent but poor predictions across different training sets.

```python
def bias_variance_decomposition(model, X, y, test_size=0.2, n_bootstrap=100):
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    predictions = np.zeros((n_bootstrap, len(X_test)))
    
    for i in range(n_bootstrap):
        # Bootstrap sampling
        indices = np.random.randint(0, len(X_train), len(X_train))
        X_bootstrap = X_train[indices]
        y_bootstrap = y_train[indices]
        
        # Fit and predict
        model.fit(X_bootstrap, y_bootstrap)
        predictions[i] = model.predict(X_test)
    
    # Calculate bias and variance
    expected_predictions = np.mean(predictions, axis=0)
    bias = np.mean((y_test - expected_predictions) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    
    return bias, variance

bias, variance = bias_variance_decomposition(LinearRegression(), X, y)
print(f"Bias: {bias:.4f}, Variance: {variance:.4f}")
```

Slide 4: Comparative Analysis of Model Complexity

Understanding how different model complexities affect underfitting requires systematic comparison. This analysis demonstrates the transition from underfitting through optimal fitting using polynomial features of increasing degrees.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def compare_model_complexities(X, y, max_degree=5):
    mse_scores = []
    models = []
    
    for degree in range(1, max_degree + 1):
        model = make_pipeline(
            PolynomialFeatures(degree),
            LinearRegression()
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        mse_scores.append(mse)
        models.append(model)
        
        print(f"Degree {degree} - MSE: {mse:.4f}")
    
    return models, mse_scores

models, scores = compare_model_complexities(X, y)
plt.plot(range(1, len(scores) + 1), scores, marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Model Complexity vs. Error')
```

Slide 5: Cross-Validation for Underfitting Detection

Cross-validation provides a robust method for detecting underfitting by evaluating model performance across multiple data splits. Consistent poor performance across folds indicates systematic underfitting issues.

```python
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer

def cross_validate_underfit(X, y, model, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scorer = make_scorer(mean_squared_error)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        score = mse_scorer(model, X_val, y_val)
        fold_scores.append(score)
        print(f"Fold {fold + 1} MSE: {score:.4f}")
    
    print(f"\nMean MSE: {np.mean(fold_scores):.4f}")
    print(f"Std MSE: {np.std(fold_scores):.4f}")
    
    return fold_scores

underfit_model = LinearRegression()
cv_scores = cross_validate_underfit(X, y, underfit_model)
```

Slide 6: Real-world Example: Housing Price Prediction

Analyzing underfitting in the context of housing price prediction demonstrates the practical implications of model complexity. Simple linear models often underfit due to the inherent non-linear relationships in real estate data.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
housing = fetch_california_housing()
X_housing = housing.data
y_housing = housing.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_housing)

# Create and evaluate underfitting model
basic_model = LinearRegression()
basic_model.fit(X_scaled, y_housing)

# Calculate performance metrics
y_pred = basic_model.predict(X_scaled)
mse = mean_squared_error(y_housing, y_pred)
r2 = basic_model.score(X_scaled, y_housing)

print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")
```

Slide 7: Feature Engineering to Combat Underfitting

Feature engineering plays a crucial role in addressing underfitting by creating more informative representations of the data. This process involves creating interaction terms and polynomial features to capture complex relationships.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_enhanced_features(X, degree=2, interaction_only=False):
    # Create feature transformer
    polynomial = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=False
    )
    
    # Transform features and maintain interpretability
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_poly = polynomial.fit_transform(X)
    poly_features = polynomial.get_feature_names_out(feature_names)
    
    # Evaluate feature importance
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Print feature importance
    for feature, coef in zip(poly_features, model.coef_):
        print(f"{feature}: {coef:.4f}")
    
    return X_poly, model

X_enhanced, enhanced_model = create_enhanced_features(X_scaled)
enhanced_mse = mean_squared_error(y_housing, enhanced_model.predict(X_enhanced))
print(f"Enhanced Model MSE: {enhanced_mse:.4f}")
```

Slide 8: Regularization Impact on Underfitting

Regularization techniques can sometimes exacerbate underfitting by oversimplifying the model. Understanding this relationship helps in finding the right balance between model complexity and regularization strength.

```python
from sklearn.linear_model import Ridge, Lasso

def compare_regularization_impact(X, y, alphas=[0.01, 0.1, 1.0, 10.0]):
    results = {}
    
    for alpha in alphas:
        # Test Ridge regression
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, y)
        ridge_mse = mean_squared_error(y, ridge.predict(X))
        
        # Test Lasso regression
        lasso = Lasso(alpha=alpha)
        lasso.fit(X, y)
        lasso_mse = mean_squared_error(y, lasso.predict(X))
        
        results[alpha] = {
            'ridge_mse': ridge_mse,
            'lasso_mse': lasso_mse
        }
        
        print(f"Alpha {alpha}:")
        print(f"Ridge MSE: {ridge_mse:.4f}")
        print(f"Lasso MSE: {lasso_mse:.4f}\n")
    
    return results

regularization_results = compare_regularization_impact(X_scaled, y_housing)
```

Slide 9: Neural Network Architecture and Underfitting

Neural networks can also suffer from underfitting when their architecture is too simple. This implementation demonstrates how network depth and width affect model capacity and performance.

```python
import torch
import torch.nn as nn

class UnderfinishingNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
        return self.layers[-1](x)

# Compare different architectures
architectures = [
    [10],
    [10, 10],
    [10, 10, 10]
]

for hidden_sizes in architectures:
    model = UnderfinishingNN(X_scaled.shape[1], hidden_sizes, 1)
    print(f"Architecture {hidden_sizes}: {sum(p.numel() for p in model.parameters())} parameters")
```

Slide 10: Model Capacity Analysis with Information Criteria

Information criteria like AIC and BIC help quantify the trade-off between model complexity and underfitting by penalizing both poor fit and excessive parameters, providing objective metrics for model selection.

```python
from scipy.stats import chi2

def calculate_information_criteria(model, X, y):
    n_samples = X.shape[0]
    n_params = len(model.coef_) + 1  # Add 1 for intercept
    
    # Calculate log-likelihood
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    sigma2 = mse
    log_likelihood = -0.5 * n_samples * (np.log(2 * np.pi * sigma2) + 1)
    
    # Calculate AIC and BIC
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + np.log(n_samples) * n_params
    
    print(f"Number of parameters: {n_params}")
    print(f"AIC: {aic:.4f}")
    print(f"BIC: {bic:.4f}")
    
    return {'aic': aic, 'bic': bic}

# Compare models of different complexities
models = {
    'linear': LinearRegression(),
    'poly2': make_pipeline(PolynomialFeatures(2), LinearRegression()),
    'poly3': make_pipeline(PolynomialFeatures(3), LinearRegression())
}

for name, model in models.items():
    print(f"\nModel: {name}")
    model.fit(X_scaled, y_housing)
    criteria = calculate_information_criteria(model, X_scaled, y_housing)
```

Slide 11: Time Series Underfitting Detection

Time series data presents unique challenges for detecting underfitting, requiring specialized metrics and visualization techniques to assess model performance across different temporal patterns.

```python
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def analyze_timeseries_underfitting(data, periods=10):
    # Create time series with trend and seasonality
    dates = pd.date_range(start='2023-01-01', periods=len(data))
    ts_data = pd.Series(data.ravel(), index=dates)
    
    # Fit simple exponential smoothing (potentially underfitting)
    simple_model = ExponentialSmoothing(
        ts_data,
        seasonal_periods=periods,
        seasonal='add'
    ).fit()
    
    # Calculate residuals and perform tests
    residuals = simple_model.resid
    
    # Ljung-Box test for autocorrelation in residuals
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    
    print("Residual Analysis:")
    print(f"Mean Residual: {residuals.mean():.4f}")
    print(f"Residual Std: {residuals.std():.4f}")
    print("\nLjung-Box Test Results:")
    print(lb_test)
    
    return simple_model

# Generate sample time series data
time_data = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
model = analyze_timeseries_underfitting(time_data)
```

Slide 12: Real-world Example: Credit Risk Modeling

Credit risk assessment demonstrates how underfitting can have significant practical implications. This implementation shows how simple models may fail to capture complex relationships in financial data.

```python
from sklearn.preprocessing import LabelEncoder
import numpy as np

def create_credit_risk_model(features, target, test_size=0.2):
    # Simulate credit risk data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic credit data
    credit_data = {
        'income': np.random.normal(50000, 20000, n_samples),
        'debt_ratio': np.random.uniform(0.1, 0.6, n_samples),
        'credit_history': np.random.choice(['good', 'fair', 'poor'], n_samples),
        'employment_years': np.random.exponential(5, n_samples)
    }
    
    # Create features matrix
    X = pd.DataFrame(credit_data)
    le = LabelEncoder()
    X['credit_history'] = le.fit_transform(X['credit_history'])
    
    # Generate target (default probability)
    y = (0.3 * X['debt_ratio'] + 
         -0.4 * np.log(X['income']) + 
         0.2 * X['credit_history'] + 
         -0.1 * X['employment_years'] + 
         np.random.normal(0, 0.1, n_samples))
    y = (y > np.mean(y)).astype(int)
    
    # Train basic model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Evaluate performance
    from sklearn.metrics import classification_report
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))
    
    return model, X, y

model, X_credit, y_credit = create_credit_risk_model(['income', 'debt_ratio'], 'default')
```

Slide 13: Graphical Model Diagnostics

Visual diagnostics provide intuitive insights into model underfitting through residual analysis and prediction-vs-actual plots, helping identify patterns that simple metrics might miss.

```python
def plot_model_diagnostics(model, X, y):
    # Create predictions
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residual plot
    axes[0,0].scatter(y_pred, residuals)
    axes[0,0].axhline(y=0, color='r', linestyle='--')
    axes[0,0].set_xlabel('Predicted Values')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residual Plot')
    
    # QQ plot of residuals
    from scipy.stats import probplot
    probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Normal Q-Q Plot')
    
    # Predicted vs Actual
    axes[1,0].scatter(y_pred, y)
    axes[1,0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    axes[1,0].set_xlabel('Predicted Values')
    axes[1,0].set_ylabel('Actual Values')
    axes[1,0].set_title('Predicted vs Actual')
    
    # Residual histogram
    axes[1,1].hist(residuals, bins=30)
    axes[1,1].set_xlabel('Residual Value')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Residual Distribution')
    
    plt.tight_layout()
    return fig

diagnostic_plots = plot_model_diagnostics(model, X_credit, y_credit)
```

Slide 14: Additional Resources

*   Bias-Variance Trade-off in Machine Learning: [https://arxiv.org/abs/2001.00686](https://arxiv.org/abs/2001.00686)
*   Understanding Deep Learning Requires Rethinking Generalization: [https://arxiv.org/abs/1611.03530](https://arxiv.org/abs/1611.03530)
*   Model Selection and Assessment of Regularization: [https://arxiv.org/abs/2003.01704](https://arxiv.org/abs/2003.01704)
*   Empirical Model Building and Response Surfaces: [https://cs.stanford.edu/research/machine-learning-resources](https://cs.stanford.edu/research/machine-learning-resources)
*   Diagnostics for Statistical Models in Machine Learning: [https://www.machinelearning.org/model-diagnostics](https://www.machinelearning.org/model-diagnostics)

