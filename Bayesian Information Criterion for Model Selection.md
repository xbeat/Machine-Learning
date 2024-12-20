## Bayesian Information Criterion for Model Selection
Slide 1: Introduction to BIC Implementation

The Bayesian Information Criterion provides a mathematical framework for model selection based on the trade-off between model complexity and goodness of fit. This implementation demonstrates the fundamental BIC calculation for a simple linear regression model.

```python
import numpy as np
from scipy import stats

def calculate_bic(y, y_pred, k):
    """
    Calculate BIC for a model
    y: actual values
    y_pred: predicted values
    k: number of parameters
    """
    n = len(y)
    mse = np.sum((y - y_pred) ** 2) / n
    ll = -0.5 * n * (np.log(2 * np.pi * mse) + 1)  # Log-likelihood
    bic = -2 * ll + k * np.log(n)
    return bic

# Example usage
X = np.random.randn(100, 1)
y = 2 * X + np.random.randn(100, 1)
y_pred = 2.1 * X
bic_value = calculate_bic(y, y_pred, k=2)
print(f"BIC: {bic_value}")
```

Slide 2: Model Comparison Using BIC

BIC enables objective comparison between competing models by penalizing model complexity. This implementation compares different polynomial regression models to identify the optimal model order based on BIC scores.

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def compare_models_bic(X, y, max_degree=5):
    results = []
    for degree in range(1, max_degree + 1):
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X.reshape(-1, 1))
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        # Calculate BIC
        n = len(y)
        k = degree + 1  # number of parameters
        mse = mean_squared_error(y, y_pred)
        ll = -0.5 * n * (np.log(2 * np.pi * mse) + 1)
        bic = -2 * ll + k * np.log(n)
        
        results.append((degree, bic))
    
    return results

# Example usage
X = np.linspace(0, 1, 100)
y = 1 + 2*X + 0.5*X**2 + np.random.normal(0, 0.1, 100)
results = compare_models_bic(X, y)
for degree, bic in results:
    print(f"Polynomial degree {degree}: BIC = {bic}")
```

Slide 3: Cross-Validation with BIC

The combination of cross-validation with BIC provides a robust framework for model selection. This implementation demonstrates how to use k-fold cross-validation to compute average BIC scores across different data splits.

```python
from sklearn.model_selection import KFold
import numpy as np
from sklearn.linear_model import LinearRegression

def bic_cv(X, y, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    bic_scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model
        model = LinearRegression()
        model.fit(X_train.reshape(-1, 1), y_train)
        y_pred = model.predict(X_test.reshape(-1, 1))
        
        # Calculate BIC for this fold
        n = len(y_test)
        k = 2  # intercept and slope
        mse = np.sum((y_test - y_pred) ** 2) / n
        ll = -0.5 * n * (np.log(2 * np.pi * mse) + 1)
        bic = -2 * ll + k * np.log(n)
        bic_scores.append(bic)
    
    return np.mean(bic_scores), np.std(bic_scores)

# Example usage
X = np.random.randn(200)
y = 2*X + np.random.randn(200)*0.5
mean_bic, std_bic = bic_cv(X, y)
print(f"Mean BIC: {mean_bic:.2f} ± {std_bic:.2f}")
```

Slide 4: Multivariate Model Selection

In multivariate analysis, BIC helps select the optimal combination of features. This implementation demonstrates feature selection using BIC for multiple regression models with different variable combinations.

```python
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression

def multivariate_bic_selection(X, y, max_features=None):
    n_samples, n_features = X.shape
    if max_features is None:
        max_features = n_features
    
    results = []
    for k in range(1, max_features + 1):
        for feature_combo in combinations(range(n_features), k):
            X_subset = X[:, list(feature_combo)]
            model = LinearRegression()
            model.fit(X_subset, y)
            y_pred = model.predict(X_subset)
            
            # Calculate BIC
            mse = np.sum((y - y_pred) ** 2) / n_samples
            ll = -0.5 * n_samples * (np.log(2 * np.pi * mse) + 1)
            bic = -2 * ll + (k + 1) * np.log(n_samples)
            
            results.append((feature_combo, bic))
    
    return sorted(results, key=lambda x: x[1])

# Example usage
X = np.random.randn(100, 4)
y = 2*X[:, 0] + 3*X[:, 2] + np.random.randn(100)*0.1
best_features = multivariate_bic_selection(X, y)
for features, bic in best_features[:3]:
    print(f"Features {features}: BIC = {bic:.2f}")
```

Slide 5: Time Series Model Selection with BIC

BIC is particularly useful for selecting the optimal order of time series models. This implementation demonstrates how to use BIC for determining the best ARIMA model parameters (p,d,q) for a given time series.

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from itertools import product

def arima_bic_selection(data, p_range, d_range, q_range):
    best_bic = np.inf
    best_params = None
    results = []
    
    for p, d, q in product(p_range, d_range, q_range):
        try:
            model = ARIMA(data, order=(p, d, q))
            results_fit = model.fit()
            bic = results_fit.bic
            results.append((p, d, q, bic))
            
            if bic < best_bic:
                best_bic = bic
                best_params = (p, d, q)
        except:
            continue
    
    return sorted(results, key=lambda x: x[3]), best_params

# Example usage
np.random.seed(42)
n_points = 200
t = np.linspace(0, 20, n_points)
data = np.sin(t) + np.random.normal(0, 0.2, n_points)

p_range = range(0, 3)
d_range = range(0, 2)
q_range = range(0, 3)

results, best_params = arima_bic_selection(data, p_range, d_range, q_range)
print(f"Best ARIMA parameters (p,d,q): {best_params}")
print("\nTop 3 models by BIC:")
for p, d, q, bic in results[:3]:
    print(f"ARIMA({p},{d},{q}): BIC = {bic:.2f}")
```

Slide 6: BIC for Clustering Analysis

BIC can determine the optimal number of clusters in clustering algorithms. This implementation shows how to use BIC to select the best number of clusters for Gaussian Mixture Models.

```python
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def optimal_clusters_bic(X, max_clusters=10):
    n_components_range = range(1, max_clusters + 1)
    bic_scores = []
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for n_components in n_components_range:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=42
        )
        gmm.fit(X_scaled)
        bic_scores.append(gmm.bic(X_scaled))
    
    # Find optimal number of clusters
    optimal_clusters = n_components_range[np.argmin(bic_scores)]
    
    return optimal_clusters, bic_scores

# Example usage
# Generate synthetic clustered data
np.random.seed(42)
n_samples = 300
X = np.concatenate([
    np.random.normal(0, 1, (n_samples, 2)),
    np.random.normal(4, 1.5, (n_samples, 2)),
    np.random.normal(-4, 0.5, (n_samples, 2))
])

optimal_k, bic_values = optimal_clusters_bic(X)
print(f"Optimal number of clusters: {optimal_k}")
print("\nBIC scores:")
for k, bic in enumerate(bic_values, 1):
    print(f"k={k}: BIC={bic:.2f}")
```

Slide 7: Real-world Application: Stock Market Analysis

This implementation uses BIC to select the optimal model for predicting stock market returns using multiple features. The example includes data preprocessing, model selection, and performance evaluation.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

def stock_return_model_selection(returns, features):
    """
    Select optimal model for stock returns prediction using BIC
    """
    # Prepare data
    X = StandardScaler().fit_transform(features)
    y = returns
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit model with cross-validation
    lasso = LassoCV(cv=5)
    lasso.fit(X_train, y_train)
    
    # Get selected features
    selected_features = features.columns[np.abs(lasso.coef_) > 0]
    
    # Calculate BIC for selected model
    y_pred = lasso.predict(X_test)
    n = len(y_test)
    k = len(selected_features)
    mse = np.sum((y_test - y_pred) ** 2) / n
    ll = -0.5 * n * (np.log(2 * np.pi * mse) + 1)
    bic = -2 * ll + k * np.log(n)
    
    return selected_features, bic, lasso

# Example usage with synthetic data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
data = pd.DataFrame({
    'returns': np.random.randn(500),
    'volume': np.random.randn(500),
    'volatility': np.random.randn(500),
    'market_returns': np.random.randn(500),
    'sentiment': np.random.randn(500)
}, index=dates)

features = data.drop('returns', axis=1)
selected_features, bic, model = stock_return_model_selection(
    data['returns'], features
)

print(f"Selected features: {selected_features.tolist()}")
print(f"Model BIC: {bic:.2f}")
print(f"Model R² score: {model.score(StandardScaler().fit_transform(features), data['returns']):.4f}")
```

Slide 8: Real-world Application: Clinical Trial Analysis

BIC implementation for analyzing clinical trial data with multiple treatment groups. This example demonstrates model selection for identifying significant treatment effects while controlling for covariates.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from scipy import stats

def clinical_trial_bic(data, outcome, treatments, covariates):
    """
    Analyze clinical trial data using BIC for model selection
    """
    # Prepare data
    X_base = pd.get_dummies(data[treatments], prefix='treatment')
    X_covs = data[covariates]
    X_full = pd.concat([X_base, X_covs], axis=1)
    y = data[outcome]
    
    # Define models to compare
    models = {
        'null': [],
        'treatment_only': list(X_base.columns),
        'covariates_only': list(X_covs.columns),
        'full': list(X_full.columns)
    }
    
    results = {}
    for model_name, features in models.items():
        if not features:
            X = np.ones((len(y), 1))  # Intercept only
        else:
            X = data[features]
            X = np.column_stack([np.ones(len(X)), X])
        
        # Fit model
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate BIC
        n = len(y)
        k = X.shape[1]
        mse = np.sum((y - y_pred) ** 2) / n
        ll = -0.5 * n * (np.log(2 * np.pi * mse) + 1)
        bic = -2 * ll + k * np.log(n)
        
        results[model_name] = {
            'bic': bic,
            'parameters': k,
            'r2': model.score(X, y)
        }
    
    return results

# Example usage with synthetic clinical trial data
np.random.seed(42)
n_patients = 200

# Generate synthetic data
data = pd.DataFrame({
    'patient_id': range(n_patients),
    'treatment': np.random.choice(['A', 'B', 'C'], n_patients),
    'age': np.random.normal(50, 15, n_patients),
    'baseline_score': np.random.normal(10, 2, n_patients),
    'sex': np.random.choice(['M', 'F'], n_patients),
    'outcome': np.random.normal(0, 1, n_patients)
})

# Add treatment effect
treatment_effects = {'A': 0.5, 'B': 1.0, 'C': 0.0}
data['outcome'] += data['treatment'].map(treatment_effects)

results = clinical_trial_bic(
    data,
    outcome='outcome',
    treatments=['treatment'],
    covariates=['age', 'baseline_score', 'sex']
)

# Display results
for model, stats in results.items():
    print(f"\nModel: {model}")
    print(f"BIC: {stats['bic']:.2f}")
    print(f"Parameters: {stats['parameters']}")
    print(f"R²: {stats['r2']:.4f}")
```

Slide 9: BIC for Neural Network Architecture Selection

Implementation of BIC for selecting optimal neural network architectures, considering both network complexity and performance. This approach helps prevent overfitting by penalizing excessive model complexity.

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class NeuralNetBIC:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.model = self._build_model()
        
    def _build_model(self):
        layers = []
        prev_size = self.input_size
        
        for hidden_size in self.hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, self.output_size))
        return nn.Sequential(*layers)
    
    def calculate_bic(self, X, y, trained_model=None):
        if trained_model is not None:
            self.model = trained_model
            
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        
        # Calculate log-likelihood
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)
            mse = nn.MSELoss()(y_pred, y)
            n = len(y)
            ll = -0.5 * n * (torch.log(2 * torch.tensor(np.pi) * mse) + 1)
            
        # Calculate BIC
        bic = -2 * ll + n_params * np.log(n)
        
        return bic.item(), n_params

# Example usage
np.random.seed(42)
architectures = [
    [10],
    [20],
    [10, 5],
    [20, 10],
    [20, 10, 5]
]

# Generate synthetic data
X = torch.FloatTensor(np.random.randn(1000, 5))
y = torch.FloatTensor(np.random.randn(1000, 1))

results = []
for hidden_sizes in architectures:
    model = NeuralNetBIC(5, hidden_sizes, 1)
    bic, n_params = model.calculate_bic(X, y)
    results.append({
        'architecture': hidden_sizes,
        'bic': bic,
        'parameters': n_params
    })

# Display results
for result in results:
    print(f"\nArchitecture: {result['architecture']}")
    print(f"BIC: {result['bic']:.2f}")
    print(f"Parameters: {result['parameters']}")
```

Slide 10: Bayesian Model Averaging Using BIC

This implementation demonstrates how to perform Bayesian Model Averaging (BMA) using BIC weights to combine predictions from multiple models, providing more robust predictions than single model selection.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import softmax

class BICModelAveraging:
    def __init__(self, max_degree=5):
        self.max_degree = max_degree
        self.models = []
        self.bic_weights = None
        
    def fit(self, X, y):
        bic_scores = []
        self.models = []
        
        for degree in range(1, self.max_degree + 1):
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X.reshape(-1, 1))
            
            # Fit model
            model = LinearRegression()
            model.fit(X_poly, y)
            y_pred = model.predict(X_poly)
            
            # Calculate BIC
            n = len(y)
            k = degree + 1
            mse = np.mean((y - y_pred) ** 2)
            ll = -0.5 * n * (np.log(2 * np.pi * mse) + 1)
            bic = -2 * ll + k * np.log(n)
            
            self.models.append((poly, model))
            bic_scores.append(bic)
            
        # Calculate BIC weights
        bic_scores = np.array(bic_scores)
        self.bic_weights = softmax(-0.5 * bic_scores)
        
    def predict(self, X):
        predictions = np.zeros((len(X), len(self.models)))
        
        for i, (poly, model) in enumerate(self.models):
            X_poly = poly.transform(X.reshape(-1, 1))
            predictions[:, i] = model.predict(X_poly)
            
        # Weighted average of predictions
        return np.sum(predictions * self.bic_weights, axis=1)

# Example usage
np.random.seed(42)
X = np.linspace(-3, 3, 100)
y = 0.5 + 2*X + 0.3*X**2 + np.random.normal(0, 0.2, 100)

# Train model
bma = BICModelAveraging(max_degree=5)
bma.fit(X, y)

# Make predictions
X_test = np.linspace(-4, 4, 200)
y_pred = bma.predict(X_test)

print("Model weights based on BIC:")
for degree, weight in enumerate(bma.bic_weights, 1):
    print(f"Polynomial degree {degree}: {weight:.3f}")
```

Slide 11: Time-varying BIC for Dynamic Model Selection

Implementation of a dynamic BIC framework that adapts to temporal changes in data distributions, suitable for real-time applications and streaming data analysis.

```python
import numpy as np
from scipy.stats import norm
from collections import deque

class DynamicBIC:
    def __init__(self, window_size=100, forget_factor=0.95):
        self.window_size = window_size
        self.forget_factor = forget_factor
        self.data_window = deque(maxlen=window_size)
        self.bic_history = []
        
    def update_bic(self, new_data, models):
        """
        Update BIC scores with new data point
        models: list of tuples (model_function, n_params)
        """
        self.data_window.append(new_data)
        
        if len(self.data_window) < self.window_size:
            return None
            
        data_array = np.array(self.data_window)
        bic_scores = {}
        
        for model_name, (model_func, n_params) in models.items():
            # Calculate time-weighted log-likelihood
            weights = self.forget_factor ** np.arange(self.window_size)[::-1]
            predictions = model_func(data_array[:-1])
            errors = data_array[1:] - predictions
            
            # Weighted maximum likelihood estimation
            weighted_mse = np.average(errors**2, weights=weights[1:])
            weighted_ll = np.sum(weights[1:] * norm.logpdf(errors, scale=np.sqrt(weighted_mse)))
            
            # Calculate dynamic BIC
            effective_n = np.sum(weights)
            bic = -2 * weighted_ll + n_params * np.log(effective_n)
            bic_scores[model_name] = bic
            
        self.bic_history.append(bic_scores)
        return bic_scores

# Example usage with simple time series models
def ar1_model(data):
    return data[:-1]

def ar2_model(data):
    return 0.7 * data[1:-1] + 0.3 * data[:-2]

# Generate synthetic data with regime change
np.random.seed(42)
n_points = 500
regime1 = np.random.normal(0, 1, n_points//2)
regime2 = np.random.normal(2, 1.5, n_points//2)
data = np.concatenate([regime1, regime2])

# Initialize dynamic BIC
dynamic_bic = DynamicBIC(window_size=50)

# Define models
models = {
    'AR(1)': (ar1_model, 2),  # intercept + 1 coefficient
    'AR(2)': (ar2_model, 3)   # intercept + 2 coefficients
}

# Process data
bic_scores_over_time = []
for i in range(2, len(data)):
    scores = dynamic_bic.update_bic(data[i], models)
    if scores:
        bic_scores_over_time.append(scores)

# Print results
print("Final BIC scores:")
for model, score in bic_scores_over_time[-1].items():
    print(f"{model}: {score:.2f}")
```

Slide 12: Hierarchical Model Selection with BIC

This implementation shows how to use BIC for selecting optimal hierarchical model structures, particularly useful in mixed-effects modeling and nested data analysis.

```python
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

class HierarchicalBIC:
    def __init__(self):
        self.levels = {}
        self.bic_scores = {}
        
    def fit_hierarchical_model(self, data, group_cols, target, random_effects):
        """
        Fit hierarchical models with different random effects structures
        """
        n_total = len(data)
        results = {}
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(data[random_effects])
        y = data[target].values
        groups = data[group_cols].values
        
        # Try different random effects combinations
        for i in range(1, len(random_effects) + 1):
            for combo in self._get_combinations(random_effects, i):
                # Fit random effects
                group_means = {}
                residuals = y.copy()
                
                for group in np.unique(groups):
                    mask = (groups == group)
                    group_data = X[mask]
                    group_means[group] = np.mean(y[mask])
                    residuals[mask] -= group_means[group]
                
                # Calculate likelihood
                ll = np.sum(stats.norm.logpdf(residuals))
                
                # Calculate number of parameters
                n_params = 1 + len(np.unique(groups)) * len(combo)
                
                # Calculate BIC
                bic = -2 * ll + n_params * np.log(n_total)
                
                results[str(combo)] = {
                    'bic': bic,
                    'n_params': n_params,
                    'log_likelihood': ll
                }
        
        self.bic_scores = results
        return results
    
    def _get_combinations(self, items, n):
        from itertools import combinations
        return list(combinations(items, n))

# Example usage with synthetic hierarchical data
np.random.seed(42)

# Generate synthetic hierarchical data
n_groups = 20
n_subjects_per_group = 30

data = []
for group in range(n_groups):
    group_effect = np.random.normal(0, 1)
    for subject in range(n_subjects_per_group):
        age = np.random.normal(40, 10)
        treatment = np.random.choice([0, 1])
        
        # Generate outcome with group and subject-level effects
        outcome = (
            2 + 
            group_effect +  # Random group effect
            0.5 * age +     # Fixed age effect
            1.5 * treatment + # Fixed treatment effect
            np.random.normal(0, 0.5)  # Error
        )
        
        data.append({
            'group': group,
            'subject': f'{group}_{subject}',
            'age': age,
            'treatment': treatment,
            'outcome': outcome
        })

import pandas as pd
df = pd.DataFrame(data)

# Fit hierarchical models
hierarchical_bic = HierarchicalBIC()
results = hierarchical_bic.fit_hierarchical_model(
    df,
    group_cols=['group'],
    target='outcome',
    random_effects=['age', 'treatment']
)

# Display results
print("Hierarchical Model Selection Results:")
for model, stats in results.items():
    print(f"\nRandom effects: {model}")
    print(f"BIC: {stats['bic']:.2f}")
    print(f"Parameters: {stats['n_params']}")
    print(f"Log-likelihood: {stats['log_likelihood']:.2f}")
```

Slide 13: BIC-based Feature Selection with Stability Assessment

This implementation combines BIC with bootstrap resampling to assess the stability of feature selection, providing robust variable selection for high-dimensional data.

```python
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

class StableBICSelector:
    def __init__(self, n_bootstrap=100, threshold=0.5):
        self.n_bootstrap = n_bootstrap
        self.threshold = threshold
        self.selection_frequencies = None
        self.stable_features = None
        
    def select_features(self, X, y):
        """
        Perform stable feature selection using BIC and bootstrap
        """
        n_samples, n_features = X.shape
        selection_matrix = np.zeros((self.n_bootstrap, n_features))
        bic_scores = []
        
        for i in range(self.n_bootstrap):
            # Bootstrap resample
            X_boot, y_boot = resample(X, y)
            
            # Standardize features
            scaler = StandardScaler()
            X_boot_scaled = scaler.fit_transform(X_boot)
            
            # Fit Lasso with CV
            lasso = LassoCV(cv=5)
            lasso.fit(X_boot_scaled, y_boot)
            
            # Get selected features
            selected = np.abs(lasso.coef_) > 0
            selection_matrix[i, :] = selected
            
            # Calculate BIC
            y_pred = lasso.predict(X_boot_scaled)
            mse = np.mean((y_boot - y_pred) ** 2)
            k = np.sum(selected)
            ll = -0.5 * n_samples * (np.log(2 * np.pi * mse) + 1)
            bic = -2 * ll + k * np.log(n_samples)
            bic_scores.append(bic)
            
        # Calculate selection frequencies
        self.selection_frequencies = np.mean(selection_matrix, axis=0)
        self.stable_features = np.where(self.selection_frequencies >= self.threshold)[0]
        
        return {
            'stable_features': self.stable_features,
            'frequencies': self.selection_frequencies,
            'mean_bic': np.mean(bic_scores),
            'std_bic': np.std(bic_scores)
        }

# Example usage
np.random.seed(42)

# Generate synthetic high-dimensional data
n_samples = 200
n_features = 50
n_informative = 5

# Generate informative features
X_informative = np.random.randn(n_samples, n_informative)
beta_informative = np.random.uniform(1, 2, n_informative)
y = np.dot(X_informative, beta_informative)

# Add noise features
X_noise = np.random.randn(n_samples, n_features - n_informative)
X = np.hstack([X_informative, X_noise])

# Add noise to response
y += np.random.normal(0, 0.1, n_samples)

# Perform stable feature selection
selector = StableBICSelector(n_bootstrap=100, threshold=0.5)
results = selector.select_features(X, y)

print("Stable Feature Selection Results:")
print(f"Number of stable features: {len(results['stable_features'])}")
print("\nFeature selection frequencies:")
for i, freq in enumerate(results['frequencies']):
    if freq > 0:
        print(f"Feature {i}: {freq:.3f}")
print(f"\nMean BIC: {results['mean_bic']:.2f}")
print(f"Std BIC: {results['std_bic']:.2f}")
```

Slide 14: Additional Resources

*   "Statistical Inference Using the Bayesian Information Criterion" - [https://arxiv.org/abs/1011.2643](https://arxiv.org/abs/1011.2643)
*   "On Model Selection and the Principle of Minimum Description Length" - [https://arxiv.org/abs/0804.3665](https://arxiv.org/abs/0804.3665)
*   "Bayesian Model Selection and Model Averaging" - [https://arxiv.org/abs/1001.0995](https://arxiv.org/abs/1001.0995)
*   Suggested searches:
    *   "Bayesian Information Criterion Applications in Machine Learning"
    *   "Model Selection Techniques Comparison"
    *   "BIC vs AIC in Practice"
    *   "Modern Approaches to Model Selection"

