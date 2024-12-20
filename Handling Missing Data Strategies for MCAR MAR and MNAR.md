## Handling Missing Data Strategies for MCAR MAR and MNAR
Slide 1: Understanding Missing Data Patterns

Missing data mechanisms fundamentally shape our imputation strategy choices. We'll explore how to detect MCAR, MAR, and MNAR patterns using statistical tests and visualization techniques that guide subsequent handling approaches.

```python
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_missing_patterns(df):
    # Create missing value indicator matrix
    missing_matrix = df.isnull().astype(int)
    
    # Little's MCAR test implementation
    def littles_mcar_test(data):
        n = len(data)
        means = data.mean()
        cov = data.cov()
        
        # Calculate D2 statistic
        d2 = 0
        for i in range(n):
            row = data.iloc[i]
            diff = row - means
            d2 += np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff)
        
        # Chi-square test
        df = data.shape[1] * (data.shape[1] - 1) / 2
        p_value = 1 - stats.chi2.cdf(d2, df)
        return p_value
    
    # Perform test and visualize patterns
    p_value = littles_mcar_test(df)
    
    # Visualize missing patterns
    plt.figure(figsize=(10, 6))
    sns.heatmap(missing_matrix, cmap='viridis')
    plt.title('Missing Value Patterns')
    
    return {
        'mcar_p_value': p_value,
        'missing_counts': missing_matrix.sum(),
        'missing_correlations': missing_matrix.corr()
    }

# Example usage
np.random.seed(42)
df = pd.DataFrame({
    'A': np.random.randn(1000),
    'B': np.random.randn(1000),
    'C': np.random.randn(1000)
})

# Introduce MAR pattern
mask = df['A'] > 1
df.loc[mask, 'B'] = np.nan

results = analyze_missing_patterns(df)
print(f"MCAR test p-value: {results['mcar_p_value']:.4f}")
```

Slide 2: kNN Imputation Implementation

The k-Nearest Neighbors imputation method leverages similarity between observations to fill missing values. This implementation includes distance-weighted voting and handles both numerical and categorical features through custom distance metrics.

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances

class KNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
    
    def _get_weights(self, distances):
        if self.weights == 'uniform':
            return np.ones(distances.shape)
        else:  # 'distance' weighted
            return 1 / (distances + 1e-8)
    
    def fit(self, X):
        self.X_ = np.array(X)
        self.missing_mask_ = np.isnan(self.X_)
        return self
    
    def transform(self, X):
        X_imputed = np.array(X, copy=True)
        
        for feature_idx in range(X.shape[1]):
            missing_idx = np.where(np.isnan(X[:, feature_idx]))[0]
            
            if len(missing_idx) == 0:
                continue
                
            # Create distance matrix using non-missing features
            valid_features = ~np.isnan(X).any(axis=0)
            distances = euclidean_distances(
                X[missing_idx][:, valid_features],
                X[:, valid_features]
            )
            
            # Get k nearest neighbors
            k_nearest_idx = np.argsort(distances, axis=1)[:, 1:self.n_neighbors+1]
            
            for idx, neighbors in zip(missing_idx, k_nearest_idx):
                weights = self._get_weights(distances[idx, neighbors])
                X_imputed[idx, feature_idx] = np.average(
                    X[neighbors, feature_idx],
                    weights=weights
                )
                
        return X_imputed

# Example usage
X = np.array([
    [1, 2, np.nan],
    [3, np.nan, 2],
    [np.nan, 5, 6],
    [4, 5, 6],
    [7, 8, 9]
])

imputer = KNNImputer(n_neighbors=2)
X_imputed = imputer.fit_transform(X)
print("Original data:\n", X)
print("\nImputed data:\n", X_imputed)
```

Slide 3: MissForest Algorithm Implementation

MissForest employs an iterative imputation strategy using Random Forests as the underlying predictor. This implementation includes convergence monitoring and handles both regression and classification tasks adaptively.

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

class MissForest(BaseEstimator, TransformerMixin):
    def __init__(self, max_iter=10, n_estimators=100, tol=1e-3):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.tol = tol
        
    def _get_mask(self, X):
        return np.isnan(X)
    
    def _get_initial_imputation(self, X):
        imp = np.array(X, copy=True)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                imp[mask, j] = np.nanmean(X[:, j])
        return imp
    
    def fit_transform(self, X):
        X_curr = self._get_initial_imputation(X)
        missing_mask = self._get_mask(X)
        
        for iteration in range(self.max_iter):
            X_prev = X_curr.copy()
            
            # Sort features by missing count
            n_missing_per_feature = missing_mask.sum(axis=0)
            features_ordered = np.argsort(n_missing_per_feature)
            
            for feature in features_ordered:
                if n_missing_per_feature[feature] == 0:
                    continue
                    
                mask_feature = missing_mask[:, feature]
                
                # Prepare training data
                X_train = X_curr[~mask_feature]
                y_train = X[~mask_feature, feature]
                X_imp = X_curr[mask_feature]
                
                # Train Random Forest
                rf = RandomForestRegressor(n_estimators=self.n_estimators)
                rf.fit(X_train, y_train)
                
                # Update imputed values
                X_curr[mask_feature, feature] = rf.predict(X_imp)
            
            # Check convergence
            change = np.mean((X_curr - X_prev) ** 2)
            if change < self.tol:
                break
                
        return X_curr

# Example usage with convergence monitoring
X = np.array([
    [1, 2, np.nan, 4],
    [np.nan, 2, 3, 4],
    [1, np.nan, 3, 4],
    [1, 2, 3, 4]
])

imputer = MissForest(max_iter=5)
X_imputed = imputer.fit_transform(X)
print("Original data:\n", X)
print("\nImputed data:\n", X_imputed)
```

Slide 4: Evaluating Imputation Quality

Statistical evaluation of imputation quality requires specialized metrics beyond standard error measures. We implement normalized root mean squared error (NRMSE) and feature-wise accuracy assessment for both numerical and categorical variables.

```python
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class ImputationEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, X_true, X_imputed, X_missing_mask):
        """
        Evaluates imputation quality using multiple metrics
        """
        # Calculate NRMSE for numerical features
        nrmse = np.sqrt(
            mean_squared_error(
                X_true[X_missing_mask],
                X_imputed[X_missing_mask]
            )
        ) / np.std(X_true[~X_missing_mask])
        
        # Feature-wise evaluation
        feature_metrics = {}
        for j in range(X_true.shape[1]):
            mask_j = X_missing_mask[:, j]
            if mask_j.any():
                mse_j = mean_squared_error(
                    X_true[mask_j, j],
                    X_imputed[mask_j, j]
                )
                feature_metrics[f'feature_{j}_mse'] = mse_j
        
        # Calculate imputation bias
        bias = np.mean(X_imputed[X_missing_mask] - X_true[X_missing_mask])
        
        return {
            'nrmse': nrmse,
            'feature_metrics': feature_metrics,
            'bias': bias
        }

# Example usage with artificial missing data
np.random.seed(42)
X_complete = np.random.randn(1000, 5)

# Generate missing mask
missing_mask = np.random.rand(*X_complete.shape) < 0.2
X_missing = X_complete.copy()
X_missing[missing_mask] = np.nan

# Apply imputation (using previous MissForest implementation)
imputer = MissForest(max_iter=5)
X_imputed = imputer.fit_transform(X_missing)

# Evaluate results
evaluator = ImputationEvaluator()
results = evaluator.evaluate(X_complete, X_imputed, missing_mask)
print("NRMSE:", results['nrmse'])
print("Bias:", results['bias'])
print("Feature-wise MSE:", results['feature_metrics'])
```

Slide 5: Imputation for Time Series Data

Time series data requires specialized imputation approaches that account for temporal dependencies. This implementation combines local temporal patterns with global feature relationships using an autoregressive component.

```python
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class TimeSeriesImputer:
    def __init__(self, method='local_ar', window_size=5):
        self.method = method
        self.window_size = window_size
    
    def _local_ar_impute(self, series):
        """
        Local autoregressive imputation for time series
        """
        imputed = series.copy()
        missing_idx = np.where(pd.isna(series))[0]
        
        for idx in missing_idx:
            start = max(0, idx - self.window_size)
            end = min(len(series), idx + self.window_size)
            
            window = series[start:end]
            valid_values = window[~pd.isna(window)]
            
            if len(valid_values) > 0:
                # Fit local AR model
                coeffs = np.polyfit(
                    np.arange(len(valid_values)),
                    valid_values,
                    min(1, len(valid_values)-1)
                )
                
                # Predict missing value
                relative_idx = idx - start
                imputed[idx] = np.polyval(coeffs, relative_idx)
            
        return imputed
    
    def fit_transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        X_imputed = np.array(X, copy=True)
        
        for col in range(X.shape[1]):
            series = pd.Series(X[:, col])
            if series.isna().any():
                X_imputed[:, col] = self._local_ar_impute(series)
        
        return X_imputed

# Example with synthetic time series data
np.random.seed(42)
t = np.linspace(0, 10, 1000)
y = np.sin(t) + 0.1 * np.random.randn(1000)

# Introduce missing values
missing_mask = np.random.rand(1000) < 0.1
y_missing = y.copy()
y_missing[missing_mask] = np.nan

# Apply imputation
imputer = TimeSeriesImputer(window_size=10)
y_imputed = imputer.fit_transform(y_missing.reshape(-1, 1))

# Calculate error metrics
from sklearn.metrics import mean_squared_error
error = mean_squared_error(
    y[missing_mask],
    y_imputed[missing_mask]
)
print(f"MSE on missing values: {error:.6f}")
```

Slide 6: Multiple Imputation by Chained Equations (MICE)

MICE implements an iterative approach where each feature is imputed using all other features as predictors. This implementation includes custom prediction models for different variable types and handles convergence monitoring.

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class MICE(BaseEstimator):
    def __init__(self, max_iter=10, n_imputations=5, tol=1e-3):
        self.max_iter = max_iter
        self.n_imputations = n_imputations
        self.tol = tol
        
    def _get_predictor(self, dtype):
        if np.issubdtype(dtype, np.number):
            return RandomForestRegressor(n_estimators=100)
        return RandomForestClassifier(n_estimators=100)
    
    def fit_transform(self, X):
        X = np.array(X, copy=True)
        missing_mask = np.isnan(X)
        
        # Initialize imputations
        imputed_arrays = []
        for m in range(self.n_imputations):
            X_imp = X.copy()
            # Initial mean imputation
            for j in range(X.shape[1]):
                mask_j = missing_mask[:, j]
                if mask_j.any():
                    X_imp[mask_j, j] = np.nanmean(X[:, j])
            
            # Iterate until convergence
            for iteration in range(self.max_iter):
                X_old = X_imp.copy()
                
                # Impute each feature
                for j in range(X.shape[1]):
                    mask_j = missing_mask[:, j]
                    if not mask_j.any():
                        continue
                    
                    # Create predictor matrix
                    predictor = self._get_predictor(X.dtype)
                    observed = ~mask_j
                    
                    # Fit on observed data
                    predictor.fit(
                        X_imp[observed, :],
                        X[observed, j]
                    )
                    
                    # Impute missing
                    X_imp[mask_j, j] = predictor.predict(X_imp[mask_j, :])
                
                # Check convergence
                change = np.mean((X_imp - X_old) ** 2)
                if change < self.tol:
                    break
                    
            imputed_arrays.append(X_imp)
            
        # Combine multiple imputations
        final_imputation = np.mean(imputed_arrays, axis=0)
        
        return final_imputation, imputed_arrays

# Example usage
np.random.seed(42)
n_samples = 1000
n_features = 5

# Generate synthetic data with missing values
X = np.random.randn(n_samples, n_features)
missing_mask = np.random.rand(*X.shape) < 0.2
X[missing_mask] = np.nan

# Apply MICE
mice = MICE(max_iter=5, n_imputations=3)
X_imputed, multiple_imputations = mice.fit_transform(X)

# Calculate imputation variance
imputation_variance = np.var(multiple_imputations, axis=0)
print("Average imputation variance:", np.mean(imputation_variance))
```

Slide 7: Handling Mixed Data Types in Imputation

Mixed data types require specialized distance metrics and imputation strategies. This implementation combines numerical and categorical handling in a unified framework with adaptive feature processing.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist

class MixedTypeImputer:
    def __init__(self, categorical_features=None, n_neighbors=5):
        self.categorical_features = categorical_features
        self.n_neighbors = n_neighbors
        self.label_encoders = {}
        
    def _compute_distance_matrix(self, X, categorical_mask):
        # Split numerical and categorical
        X_num = X[:, ~categorical_mask]
        X_cat = X[:, categorical_mask]
        
        # Normalize numerical features
        X_num_normalized = (X_num - np.nanmean(X_num, axis=0)) / np.nanstd(X_num, axis=0)
        
        # Compute Gower-like distance
        num_dist = cdist(X_num_normalized, X_num_normalized, metric='euclidean')
        
        cat_dist = np.zeros((X.shape[0], X.shape[0]))
        if X_cat.shape[1] > 0:
            cat_dist = cdist(X_cat, X_cat, metric='hamming')
        
        # Combine distances
        return (num_dist + cat_dist) / 2
    
    def fit_transform(self, X):
        if isinstance(X, pd.DataFrame):
            if self.categorical_features is None:
                self.categorical_features = X.select_dtypes(
                    include=['object', 'category']
                ).columns
            
            # Convert to numpy array
            X = X.copy()
            for col in self.categorical_features:
                le = LabelEncoder()
                mask = X[col].notna()
                X.loc[mask, col] = le.fit_transform(X.loc[mask, col])
                self.label_encoders[col] = le
            X = X.values.astype(float)
        
        categorical_mask = np.zeros(X.shape[1], dtype=bool)
        if self.categorical_features is not None:
            categorical_mask[self.categorical_features] = True
        
        # Compute distance matrix
        distances = self._compute_distance_matrix(X, categorical_mask)
        
        # Impute values
        X_imputed = X.copy()
        missing_mask = np.isnan(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if missing_mask[i, j]:
                    # Find k nearest neighbors
                    neighbor_idx = np.argsort(distances[i])[1:self.n_neighbors+1]
                    neighbor_values = X[neighbor_idx, j]
                    
                    if categorical_mask[j]:
                        # Mode for categorical
                        X_imputed[i, j] = np.nanmode(neighbor_values)[0]
                    else:
                        # Mean for numerical
                        X_imputed[i, j] = np.nanmean(neighbor_values)
        
        return X_imputed

# Example usage with mixed data types
np.random.seed(42)
n_samples = 1000

# Create mixed-type dataset
df = pd.DataFrame({
    'numerical1': np.random.randn(n_samples),
    'categorical1': np.random.choice(['A', 'B', 'C'], n_samples),
    'numerical2': np.random.randn(n_samples),
    'categorical2': np.random.choice(['X', 'Y', 'Z'], n_samples)
})

# Introduce missing values
for col in df.columns:
    mask = np.random.rand(n_samples) < 0.2
    df.loc[mask, col] = np.nan

# Apply imputation
imputer = MixedTypeImputer(categorical_features=['categorical1', 'categorical2'])
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

print("Missing values before:", df.isna().sum())
print("Missing values after:", df_imputed.isna().sum())
```

Slide 8: Matrix Factorization for Imputation

Matrix factorization approaches decompose the incomplete data matrix into lower-rank approximations, effectively capturing latent patterns for imputation. This implementation includes regularization and alternating optimization.

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MatrixFactorizationImputer(BaseEstimator, TransformerMixin):
    def __init__(self, rank=10, lambda_reg=0.1, max_iter=100, tol=1e-4):
        self.rank = rank
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        
    def _initialize_factors(self, X):
        n, m = X.shape
        # Initialize with normal distribution
        U = np.random.randn(n, self.rank) * 0.1
        V = np.random.randn(self.rank, m) * 0.1
        return U, V
    
    def fit_transform(self, X):
        X = np.array(X, copy=True)
        missing_mask = np.isnan(X)
        
        # Initialize with mean
        X_filled = X.copy()
        column_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            X_filled[missing_mask[:, j], j] = column_means[j]
        
        # Initialize factors
        U, V = self._initialize_factors(X)
        
        # Alternating minimization
        for iteration in range(self.max_iter):
            U_old = U.copy()
            
            # Update U
            for i in range(X.shape[0]):
                observed = ~missing_mask[i, :]
                if observed.any():
                    V_obs = V[:, observed]
                    x_obs = X[i, observed]
                    
                    # Ridge regression update
                    A = V_obs.dot(V_obs.T) + self.lambda_reg * np.eye(self.rank)
                    b = V_obs.dot(x_obs)
                    U[i, :] = np.linalg.solve(A, b)
            
            # Update V
            for j in range(X.shape[1]):
                observed = ~missing_mask[:, j]
                if observed.any():
                    U_obs = U[observed, :]
                    x_obs = X[observed, j]
                    
                    # Ridge regression update
                    A = U_obs.T.dot(U_obs) + self.lambda_reg * np.eye(self.rank)
                    b = U_obs.T.dot(x_obs)
                    V[:, j] = np.linalg.solve(A, b)
            
            # Check convergence
            change = np.mean((U - U_old) ** 2)
            if change < self.tol:
                break
        
        # Final imputation
        X_imputed = U.dot(V)
        X[missing_mask] = X_imputed[missing_mask]
        
        return X

# Example usage
np.random.seed(42)
n_samples, n_features = 1000, 20

# Generate low-rank matrix with noise
true_rank = 5
U_true = np.random.randn(n_samples, true_rank)
V_true = np.random.randn(true_rank, n_features)
X_true = U_true.dot(V_true) + 0.1 * np.random.randn(n_samples, n_features)

# Introduce missing values
missing_mask = np.random.rand(n_samples, n_features) < 0.2
X_missing = X_true.copy()
X_missing[missing_mask] = np.nan

# Apply imputation
imputer = MatrixFactorizationImputer(rank=true_rank)
X_imputed = imputer.fit_transform(X_missing)

# Calculate error
mse = np.mean((X_true[missing_mask] - X_imputed[missing_mask]) ** 2)
print(f"MSE on missing values: {mse:.6f}")
```

Slide 9: Robust Imputation with Autoencoder

Autoencoder-based imputation leverages deep learning to capture complex non-linear relationships in the data. This implementation includes denoising and dropout for improved robustness.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ImputationAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dims=[64, 32]):
        super().__init__()
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        encoding_dims.reverse()
        prev_dim = encoding_dims[0]
        for dim in encoding_dims[1:]:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x, mask):
        # Add noise to observed values
        if self.training:
            noise = 0.1 * torch.randn_like(x)
            x = x * mask + (x + noise) * (1 - mask)
            
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderImputer:
    def __init__(self, epochs=100, batch_size=32, learning_rate=1e-3):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
    def fit_transform(self, X):
        X = np.array(X, copy=True)
        missing_mask = np.isnan(X)
        
        # Initial mean imputation
        X_filled = X.copy()
        column_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            X_filled[missing_mask[:, j], j] = column_means[j]
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_filled)
        mask_tensor = torch.FloatTensor(~missing_mask)
        
        # Create model
        model = ImputationAutoencoder(X.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        dataset = TensorDataset(X_tensor, mask_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_mask in loader:
                optimizer.zero_grad()
                output = model(batch_x, batch_mask)
                loss = criterion(output * batch_mask, batch_x * batch_mask)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.6f}")
        
        # Final imputation
        model.eval()
        with torch.no_grad():
            X_imputed = model(X_tensor, mask_tensor).numpy()
            X[missing_mask] = X_imputed[missing_mask]
            
        return X

# Example usage
np.random.seed(42)
n_samples, n_features = 1000, 20

# Generate synthetic data
X = np.random.randn(n_samples, n_features)
missing_mask = np.random.rand(n_samples, n_features) < 0.2
X_missing = X.copy()
X_missing[missing_mask] = np.nan

# Apply imputation
imputer = AutoencoderImputer(epochs=50)
X_imputed = imputer.fit_transform(X_missing)

# Calculate error
mse = np.mean((X[missing_mask] - X_imputed[missing_mask]) ** 2)
print(f"MSE on missing values: {mse:.6f}")
```

Slide 10: Streaming Data Imputation

Real-time imputation for streaming data requires efficient online algorithms that can update incrementally. This implementation uses exponential moving averages and sliding windows for adaptive imputation.

```python
import numpy as np
from collections import deque

class StreamingImputer:
    def __init__(self, window_size=100, alpha=0.1):
        self.window_size = window_size
        self.alpha = alpha
        self.windows = {}
        self.ema_values = {}
        self.var_estimates = {}
        
    def _update_statistics(self, feature_id, value):
        if feature_id not in self.windows:
            self.windows[feature_id] = deque(maxlen=self.window_size)
            self.ema_values[feature_id] = value
            self.var_estimates[feature_id] = 0
            
        window = self.windows[feature_id]
        window.append(value)
        
        # Update exponential moving average
        self.ema_values[feature_id] = (
            self.alpha * value + 
            (1 - self.alpha) * self.ema_values[feature_id]
        )
        
        # Update variance estimate
        if len(window) > 1:
            self.var_estimates[feature_id] = np.var(window)
    
    def process_sample(self, sample):
        """
        Process a single sample in real-time
        """
        sample = np.array(sample, copy=True)
        missing_mask = np.isnan(sample)
        
        # Update statistics for observed values
        for i, (value, is_missing) in enumerate(zip(sample, missing_mask)):
            if not is_missing:
                self._update_statistics(i, value)
        
        # Impute missing values
        for i, is_missing in enumerate(missing_mask):
            if is_missing:
                if i in self.ema_values:
                    # Add random noise based on variance estimate
                    noise = np.random.normal(
                        0,
                        np.sqrt(self.var_estimates[i])
                    ) if self.var_estimates[i] > 0 else 0
                    
                    sample[i] = self.ema_values[i] + noise
                else:
                    # If no history available, use 0
                    sample[i] = 0
                    
        return sample

class StreamingDataSimulator:
    def __init__(self, n_features=5, missing_prob=0.2):
        self.n_features = n_features
        self.missing_prob = missing_prob
        self.time = 0
        
    def generate_sample(self):
        """
        Generate a single sample with seasonal patterns
        """
        # Generate base signal with seasonality
        sample = np.sin(self.time / 10) + 0.1 * np.random.randn(self.n_features)
        
        # Introduce missing values
        missing_mask = np.random.rand(self.n_features) < self.missing_prob
        sample[missing_mask] = np.nan
        
        self.time += 1
        return sample

# Example usage
np.random.seed(42)

# Create simulator and imputer
simulator = StreamingDataSimulator(n_features=5)
imputer = StreamingImputer(window_size=100)

# Process streaming data
n_samples = 1000
original_data = []
imputed_data = []

for _ in range(n_samples):
    sample = simulator.generate_sample()
    original_data.append(sample)
    imputed_sample = imputer.process_sample(sample)
    imputed_data.append(imputed_sample)

# Convert to numpy arrays for analysis
original_data = np.array(original_data)
imputed_data = np.array(imputed_data)

# Calculate streaming imputation error
missing_mask = np.isnan(original_data)
mse = np.mean(
    (original_data[~missing_mask] - imputed_data[~missing_mask]) ** 2
)
print(f"Streaming MSE: {mse:.6f}")
```

Slide 11: Real-world Example: Medical Time Series Data

Implementation of a specialized imputation strategy for medical time series data, handling irregular sampling rates and physiological constraints while preserving temporal patterns.

```python
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class MedicalTimeSeriesImputer:
    def __init__(self, physio_bounds=None):
        self.physio_bounds = physio_bounds or {
            'heart_rate': (40, 200),
            'blood_pressure': (60, 200),
            'temperature': (35, 42),
            'oxygen_saturation': (80, 100)
        }
        
    def _validate_physiological(self, values, feature):
        """
        Ensure imputed values are physiologically plausible
        """
        if feature in self.physio_bounds:
            lower, upper = self.physio_bounds[feature]
            return np.clip(values, lower, upper)
        return values
    
    def _interpolate_gaps(self, times, values, max_gap=None):
        """
        Interpolate gaps with physiological constraints
        """
        if len(times) < 2:
            return values
            
        # Create interpolator for non-missing values
        valid_mask = ~np.isnan(values)
        if not valid_mask.any():
            return values
            
        f = interp1d(
            times[valid_mask],
            values[valid_mask],
            kind='cubic',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # Interpolate all timestamps
        interpolated = f(times)
        
        # Handle large gaps if specified
        if max_gap is not None:
            gaps = np.diff(times[valid_mask])
            large_gaps = gaps > max_gap
            
            if large_gaps.any():
                # Mark interpolated values in large gaps as missing
                gap_starts = times[valid_mask][:-1][large_gaps]
                gap_ends = times[valid_mask][1:][large_gaps]
                
                for start, end in zip(gap_starts, gap_ends):
                    gap_mask = (times > start) & (times < end)
                    interpolated[gap_mask] = np.nan
                    
        return interpolated
    
    def fit_transform(self, df, time_col='timestamp'):
        """
        Impute missing values in medical time series data
        """
        df = df.copy().sort_values(time_col)
        times = df[time_col].values
        
        result = pd.DataFrame(index=df.index)
        result[time_col] = times
        
        for column in df.columns:
            if column == time_col:
                continue
                
            values = df[column].values
            
            # Interpolate with physiological constraints
            imputed = self._interpolate_gaps(
                times,
                values,
                max_gap=pd.Timedelta(hours=6).total_seconds()
            )
            
            # Apply physiological bounds
            imputed = self._validate_physiological(imputed, column)
            
            result[column] = imputed
            
        return result

# Example with synthetic medical data
np.random.seed(42)

# Generate synthetic medical time series
n_samples = 1000
timestamps = pd.date_range(
    start='2024-01-01',
    periods=n_samples,
    freq='5min'
)

data = pd.DataFrame({
    'timestamp': timestamps,
    'heart_rate': 70 + 10 * np.sin(np.arange(n_samples)/50) + 
                 5 * np.random.randn(n_samples),
    'blood_pressure': 120 + 20 * np.sin(np.arange(n_samples)/100) + 
                     10 * np.random.randn(n_samples),
    'oxygen_saturation': 98 + np.random.randn(n_samples)
})

# Introduce missing values
for col in ['heart_rate', 'blood_pressure', 'oxygen_saturation']:
    mask = np.random.rand(n_samples) < 0.2
    data.loc[mask, col] = np.nan

# Apply imputation
imputer = MedicalTimeSeriesImputer()
imputed_data = imputer.fit_transform(data)

# Calculate statistics
for col in ['heart_rate', 'blood_pressure', 'oxygen_saturation']:
    missing = data[col].isna().sum()
    print(f"\n{col}:")
    print(f"Missing values: {missing}")
    print(f"Original mean: {data[col].mean():.2f}")
    print(f"Imputed mean: {imputed_data[col].mean():.2f}")
```

Slide 12: Real-world Example: Financial Market Data

Implementation of a specialized imputation strategy for financial time series, handling market hours, forward-filling for categorical data, and maintaining time-dependent relationships.

```python
import numpy as np
import pandas as pd
from datetime import time

class FinancialMarketImputer:
    def __init__(self, market_hours={'start': time(9, 30), 'end': time(16, 0)}):
        self.market_hours = market_hours
        
    def _is_market_hours(self, timestamp):
        """Check if timestamp is within market hours"""
        current_time = timestamp.time()
        return (current_time >= self.market_hours['start'] and 
                current_time < self.market_hours['end'])
    
    def _handle_price_data(self, series, timestamps):
        """Special handling for price data"""
        # Forward fill within same trading day
        filled = series.copy()
        current_day = None
        
        for i, (timestamp, value) in enumerate(zip(timestamps, series)):
            if pd.isna(value):
                if (current_day is not None and 
                    timestamp.date() == current_day and 
                    self._is_market_hours(timestamp)):
                    filled[i] = filled[i-1]
            else:
                current_day = timestamp.date()
                
        return filled
    
    def _handle_volume_data(self, series, timestamps):
        """Special handling for volume data"""
        # Use 0 for missing volume within market hours
        filled = series.copy()
        for i, (timestamp, value) in enumerate(zip(timestamps, series)):
            if pd.isna(value) and self._is_market_hours(timestamp):
                filled[i] = 0
        return filled
    
    def fit_transform(self, df):
        """
        Impute missing values in financial market data
        """
        df = df.copy()
        result = pd.DataFrame(index=df.index)
        
        # Identify column types
        price_columns = [col for col in df.columns if 'price' in col.lower()]
        volume_columns = [col for col in df.columns if 'volume' in col.lower()]
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Handle different types of data
        timestamps = df.index
        
        # Process price data
        for col in price_columns:
            result[col] = self._handle_price_data(df[col].values, timestamps)
            
        # Process volume data
        for col in volume_columns:
            result[col] = self._handle_volume_data(df[col].values, timestamps)
            
        # Forward fill categorical data
        for col in categorical_columns:
            result[col] = df[col].ffill()
            
        # Handle remaining numerical columns
        remaining_cols = (set(df.columns) - set(price_columns) - 
                        set(volume_columns) - set(categorical_columns))
        for col in remaining_cols:
            # Interpolate within market hours
            series = df[col].copy()
            market_hours_mask = [self._is_market_hours(t) for t in timestamps]
            series.loc[market_hours_mask] = series.loc[market_hours_mask].interpolate(
                method='time'
            )
            result[col] = series
            
        return result

# Example with synthetic financial data
np.random.seed(42)

# Generate synthetic trading day data
trading_dates = pd.date_range(
    start='2024-01-01 09:30:00',
    end='2024-01-31 16:00:00',
    freq='5min'
)

# Filter for market hours
trading_dates = trading_dates[
    trading_dates.map(lambda x: 9.5 <= x.hour + x.minute/60 <= 16)
]

# Generate price process with jumps
n_samples = len(trading_dates)
base_price = 100
price = base_price + np.cumsum(0.001 * np.random.randn(n_samples))
price += np.random.binomial(1, 0.01, n_samples) * np.random.randn(n_samples)

data = pd.DataFrame({
    'price': price,
    'volume': np.random.poisson(1000, n_samples),
    'bid_size': np.random.poisson(500, n_samples),
    'ask_size': np.random.poisson(500, n_samples),
    'status': np.random.choice(['TRADING', 'AUCTION'], n_samples)
}, index=trading_dates)

# Introduce missing values
for col in data.columns:
    if col != 'status':
        mask = np.random.rand(n_samples) < 0.1
        data.loc[mask, col] = np.nan

# Apply imputation
imputer = FinancialMarketImputer()
imputed_data = imputer.fit_transform(data)

# Calculate statistics
for col in data.columns:
    if col != 'status':
        missing = data[col].isna().sum()
        print(f"\n{col}:")
        print(f"Missing values: {missing}")
        print(f"Original mean: {data[col].mean():.2f}")
        print(f"Imputed mean: {imputed_data[col].mean():.2f}")
```

Slide 13: Additional Resources

*   "Missing Data Imputation Through Machine Learning Methods: A Survey" - [https://arxiv.org/abs/2106.14656](https://arxiv.org/abs/2106.14656)
*   "Deep Learning for Missing Value Imputation in Tables with Non-Numerical Data" - [https://arxiv.org/abs/1902.06398](https://arxiv.org/abs/1902.06398)
*   "MICE: Multivariate Imputation by Chained Equations in R" - [https://arxiv.org/abs/1501.02155](https://arxiv.org/abs/1501.02155)
*   "Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares" - [https://arxiv.org/abs/1410.2596](https://arxiv.org/abs/1410.2596)
*   "Imputation of Clinical Time Series with Deep Generative Models" - [https://arxiv.org/abs/2011.08858](https://arxiv.org/abs/2011.08858)

