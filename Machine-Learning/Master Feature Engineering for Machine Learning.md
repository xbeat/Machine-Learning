## Master Feature Engineering for Machine Learning
Slide 1: Advanced Data Imputation Techniques

Missing data handling requires sophisticated approaches beyond simple mean or median imputation. Advanced techniques like Multiple Imputation by Chained Equations (MICE) create multiple complete datasets, capturing uncertainty in the imputation process while preserving relationships between variables.

```python
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Create sample dataset with missing values
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(5, 2, 1000),
    'feature3': np.random.normal(-2, 1.5, 1000)
})

# Introduce missing values randomly
mask = np.random.random(data.shape) < 0.2
data[mask] = np.nan

# Initialize and fit MICE imputer
mice_imputer = IterativeImputer(max_iter=10, random_state=42)
imputed_data = pd.DataFrame(
    mice_imputer.fit_transform(data),
    columns=data.columns
)

# Compare original vs imputed statistics
print("Original Data Stats:\n", data.describe())
print("\nImputed Data Stats:\n", imputed_data.describe())
```

Slide 2: Smart Numerical Discretization

Discretization transforms continuous variables into discrete categories, improving model interpretability and handling non-linear relationships. This implementation uses adaptive binning based on data distribution and optimizes bin boundaries using information value analysis.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

class SmartDiscretizer:
    def __init__(self, strategy='quantile', n_bins=10):
        self.strategy = strategy
        self.n_bins = n_bins
        self.discretizer = KBinsDiscretizer(
            n_bins=n_bins, 
            encode='ordinal', 
            strategy=strategy
        )
    
    def fit_transform(self, data):
        # Fit and transform data
        discretized = self.discretizer.fit_transform(data.reshape(-1, 1))
        
        # Get bin edges for interpretation
        bin_edges = self.discretizer.bin_edges_[0]
        
        # Calculate bin statistics
        bins = pd.DataFrame({
            'bin_number': range(len(bin_edges)-1),
            'lower_edge': bin_edges[:-1],
            'upper_edge': bin_edges[1:],
            'count': np.histogram(data, bins=bin_edges)[0]
        })
        
        return discretized, bins

# Example usage
np.random.seed(42)
data = np.concatenate([
    np.random.normal(0, 1, 1000),
    np.random.normal(3, 0.5, 500)
])

discretizer = SmartDiscretizer(strategy='kmeans', n_bins=8)
disc_data, bin_stats = discretizer.fit_transform(data)

print("Bin Statistics:\n", bin_stats)
```

Slide 3: Advanced Categorical Encoding

Modern categorical encoding goes beyond simple one-hot encoding, incorporating target information and handling high-cardinality features efficiently. This implementation showcases target encoding with smoothing and cross-validation to prevent overfitting.

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smooth=10, n_splits=5):
        self.smooth = smooth
        self.n_splits = n_splits
        self.encodings = {}
        
    def _compute_encoding(self, X, y):
        global_mean = np.mean(y)
        encodings = {}
        
        for cat in X.unique():
            cat_mask = X == cat
            n_cat = np.sum(cat_mask)
            cat_mean = np.mean(y[cat_mask]) if n_cat > 0 else global_mean
            
            # Apply smoothing
            lambda_factor = 1 / (1 + np.exp(-(n_cat - self.smooth) / self.smooth))
            encoding = lambda_factor * cat_mean + (1 - lambda_factor) * global_mean
            encodings[cat] = encoding
            
        return encodings
    
    def fit_transform(self, X, y):
        X_copy = X.copy()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            fold_encodings = self._compute_encoding(X_train, y_train)
            
            # Apply encodings to validation fold
            for cat, encoding in fold_encodings.items():
                mask = X_copy.iloc[val_idx] == cat
                X_copy.iloc[val_idx][mask] = encoding
        
        return X_copy

# Example usage
data = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B', 'A'] * 100,
    'target': np.random.normal(0, 1, 600)
})

encoder = TargetEncoder(smooth=10)
encoded_data = encoder.fit_transform(data['category'], data['target'])
print("Original vs Encoded Data:\n", pd.DataFrame({
    'original': data['category'].head(),
    'encoded': encoded_data.head()
}))
```

Slide 4: Advanced DateTime Feature Engineering

Temporal feature engineering extends beyond basic component extraction, incorporating cyclical encoding, temporal aggregations, and event detection. This implementation creates rich datetime features while preserving circular relationships.

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AdvancedDateFeaturizer:
    def __init__(self, date_column):
        self.date_column = date_column
        
    def _create_cyclical_features(self, period, value):
        # Convert to cyclical features using sine and cosine
        sin_value = np.sin(2 * np.pi * value / period)
        cos_value = np.cos(2 * np.pi * value / period)
        return sin_value, cos_value
    
    def transform(self, df):
        df = df.copy()
        dates = pd.to_datetime(df[self.date_column])
        
        # Extract basic components
        df['year'] = dates.dt.year
        df['month'] = dates.dt.month
        df['day'] = dates.dt.day
        df['dayofweek'] = dates.dt.dayofweek
        df['hour'] = dates.dt.hour
        
        # Create cyclical features
        df['month_sin'], df['month_cos'] = self._create_cyclical_features(12, dates.dt.month)
        df['day_sin'], df['day_cos'] = self._create_cyclical_features(31, dates.dt.day)
        df['hour_sin'], df['hour_cos'] = self._create_cyclical_features(24, dates.dt.hour)
        
        # Add specialized features
        df['is_weekend'] = dates.dt.dayofweek >= 5
        df['is_month_start'] = dates.dt.is_month_start
        df['is_month_end'] = dates.dt.is_month_end
        df['quarter'] = dates.dt.quarter
        
        # Calculate time differences
        df['days_from_start'] = (dates - dates.min()).dt.days
        
        return df

# Example usage
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
data = pd.DataFrame({'timestamp': dates})

featurizer = AdvancedDateFeaturizer('timestamp')
features = featurizer.transform(data)

print("Generated Features:\n", features.head())
print("\nFeature Columns:", features.columns.tolist())
```

Slide 5: Robust Outlier Detection and Treatment

Advanced outlier detection combines statistical methods with machine learning approaches to identify anomalies in high-dimensional spaces. This implementation uses Isolation Forest and Local Outlier Factor for robust detection across different data distributions.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

class RobustOutlierDetector:
    def __init__(self, contamination=0.1, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        
    def detect_outliers(self, data):
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Isolation Forest Detection
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )
        iso_labels = iso_forest.fit_predict(scaled_data)
        
        # Local Outlier Factor Detection
        lof = LocalOutlierFactor(contamination=self.contamination)
        lof_labels = lof.fit_predict(scaled_data)
        
        # Combine predictions (consider point as outlier if both methods agree)
        combined_labels = np.where(
            (iso_labels == -1) & (lof_labels == -1),
            -1,  # Outlier
            1    # Inlier
        )
        
        return combined_labels
    
    def remove_and_impute(self, data):
        outlier_labels = self.detect_outliers(data)
        clean_data = data.copy()
        
        # Replace outliers with median of non-outlier values
        for column in data.columns:
            mask = outlier_labels == -1
            clean_data.loc[mask, column] = np.median(
                data.loc[~mask, column]
            )
            
        return clean_data, outlier_labels

# Example usage
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 3))
outliers = np.random.uniform(-10, 10, (50, 3))
data = pd.DataFrame(
    np.vstack([normal_data, outliers]),
    columns=['feature1', 'feature2', 'feature3']
)

detector = RobustOutlierDetector(contamination=0.05)
clean_data, outliers = detector.remove_and_impute(data)

print("Original Data Stats:\n", data.describe())
print("\nCleaned Data Stats:\n", clean_data.describe())
print("\nNumber of Outliers Detected:", sum(outliers == -1))
```

Slide 6: Advanced Target Variable Transformation

Target variable transformation requires sophisticated techniques beyond simple logarithmic transformations. This implementation includes Box-Cox, Yeo-Johnson, and custom transformations with automatic parameter optimization.

```python
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import PowerTransformer

class AdvancedTargetTransformer:
    def __init__(self, method='auto'):
        self.method = method
        self.transformer = None
        self.optimal_lambda = None
        
    def _calculate_normality_score(self, data):
        # Calculate Anderson-Darling test statistic
        anderson_stat = stats.anderson(data)[0]
        # Calculate skewness
        skewness = stats.skew(data)
        # Calculate kurtosis
        kurtosis = stats.kurtosis(data)
        
        # Combine metrics (lower is better)
        return anderson_stat + abs(skewness) + abs(kurtosis - 3)
    
    def _optimize_boxcox(self, data):
        def objective(lambda_param):
            transformed = stats.boxcox(data - min(data) + 1, lambda_param)
            return self._calculate_normality_score(transformed)
            
        result = minimize(objective, x0=0.5, bounds=[(-2, 2)])
        return result.x[0]
    
    def fit_transform(self, target):
        if self.method == 'auto':
            # Try different methods and select the best
            methods = {
                'boxcox': lambda x: stats.boxcox(x - min(x) + 1)[0],
                'yeo-johnson': lambda x: PowerTransformer(
                    method='yeo-johnson'
                ).fit_transform(x.reshape(-1, 1)).ravel(),
                'log': lambda x: np.log(x - min(x) + 1),
                'sqrt': lambda x: np.sqrt(x - min(x) + 1)
            }
            
            best_score = float('inf')
            best_transformed = None
            best_method = None
            
            for method_name, transform_func in methods.items():
                try:
                    transformed = transform_func(target)
                    score = self._calculate_normality_score(transformed)
                    
                    if score < best_score:
                        best_score = score
                        best_transformed = transformed
                        best_method = method_name
                except:
                    continue
            
            self.method = best_method
            return best_transformed
        
        elif self.method == 'boxcox':
            self.optimal_lambda = self._optimize_boxcox(target)
            return stats.boxcox(target - min(target) + 1, self.optimal_lambda)
        
        elif self.method == 'yeo-johnson':
            self.transformer = PowerTransformer(method='yeo-johnson')
            return self.transformer.fit_transform(
                target.reshape(-1, 1)
            ).ravel()

# Example usage
np.random.seed(42)
skewed_target = np.exp(np.random.normal(0, 1, 1000))

transformer = AdvancedTargetTransformer(method='auto')
transformed_target = transformer.fit_transform(skewed_target)

print("Original Target Stats:")
print(pd.Series(skewed_target).describe())
print("\nTransformed Target Stats:")
print(pd.Series(transformed_target).describe())
print(f"\nSelected Method: {transformer.method}")
```

Slide 7: Feature Scaling and Normalization

Advanced feature scaling goes beyond simple standardization, incorporating robust scaling methods that handle outliers and preserve relative relationships between features while maintaining the statistical properties of the original distribution.

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats

class AdvancedScaler(BaseEstimator, TransformerMixin):
    def __init__(self, method='robust', clip_outliers=True):
        self.method = method
        self.clip_outliers = clip_outliers
        self.params = {}
        
    def _robust_scale(self, X):
        median = np.median(X, axis=0)
        iqr = stats.iqr(X, axis=0)
        scaled = (X - median) / (iqr + 1e-8)
        return scaled
        
    def _adaptive_scale(self, X):
        # Use different scaling based on distribution
        scaled = np.zeros_like(X)
        for i in range(X.shape[1]):
            # Test for normality
            _, p_value = stats.normaltest(X[:, i])
            
            if p_value > 0.05:  # Normal distribution
                scaled[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
            else:  # Non-normal distribution
                scaled[:, i] = self._robust_scale(X[:, i].reshape(-1, 1))
                
        return scaled
    
    def fit(self, X, y=None):
        X = np.array(X)
        
        if self.method == 'robust':
            self.params['median'] = np.median(X, axis=0)
            self.params['iqr'] = stats.iqr(X, axis=0)
        
        elif self.method == 'adaptive':
            self.params['distributions'] = []
            for i in range(X.shape[1]):
                _, p_value = stats.normaltest(X[:, i])
                self.params['distributions'].append('normal' if p_value > 0.05 else 'non-normal')
                
        return self
    
    def transform(self, X):
        X = np.array(X)
        
        if self.method == 'robust':
            scaled = (X - self.params['median']) / (self.params['iqr'] + 1e-8)
            
        elif self.method == 'adaptive':
            scaled = self._adaptive_scale(X)
            
        if self.clip_outliers:
            scaled = np.clip(scaled, -3, 3)
            
        return scaled

# Example usage
np.random.seed(42)
normal_feat = np.random.normal(0, 1, 1000)
skewed_feat = np.exp(np.random.normal(0, 1, 1000))
data = np.column_stack([normal_feat, skewed_feat])

scaler = AdvancedScaler(method='adaptive')
scaled_data = scaler.fit_transform(data)

print("Original Data Stats:\n", pd.DataFrame(data).describe())
print("\nScaled Data Stats:\n", pd.DataFrame(scaled_data).describe())
```

Slide 8: Intelligent Feature Creation

Feature creation involves sophisticated techniques for generating meaningful combinations of existing features while maintaining interpretability and avoiding redundancy through statistical testing and domain knowledge integration.

```python
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from sklearn.feature_selection import mutual_info_regression

class IntelligentFeatureCreator:
    def __init__(self, max_degree=2, correlation_threshold=0.8):
        self.max_degree = max_degree
        self.correlation_threshold = correlation_threshold
        self.selected_features = []
        
    def _create_polynomial_features(self, X):
        features = []
        feature_names = []
        
        for degree in range(2, self.max_degree + 1):
            for cols in combinations(range(X.shape[1]), degree):
                new_feature = np.prod(X[:, cols], axis=1)
                features.append(new_feature)
                feature_names.append(f"poly_{'_'.join(map(str, cols))}")
                
        return np.column_stack(features), feature_names
    
    def _create_statistical_features(self, X):
        features = []
        feature_names = []
        
        # Rolling statistics
        window_sizes = [3, 5, 7]
        for w in window_sizes:
            for i in range(X.shape[1]):
                roll_mean = pd.Series(X[:, i]).rolling(w).mean()
                roll_std = pd.Series(X[:, i]).rolling(w).std()
                features.extend([roll_mean, roll_std])
                feature_names.extend([f"roll_mean_{i}_{w}", f"roll_std_{i}_{w}"])
        
        return np.column_stack(features), feature_names
    
    def _evaluate_feature_importance(self, X, y, feature_names):
        importances = mutual_info_regression(X, y)
        return dict(zip(feature_names, importances))
    
    def _check_correlation(self, X):
        corr_matrix = np.corrcoef(X.T)
        return np.any(np.abs(corr_matrix - np.eye(X.shape[1])) > self.correlation_threshold)
    
    def fit_transform(self, X, y):
        X = np.array(X)
        original_features = X.copy()
        
        # Create polynomial features
        poly_features, poly_names = self._create_polynomial_features(X)
        
        # Create statistical features
        stat_features, stat_names = self._create_statistical_features(X)
        
        # Combine all features
        all_features = np.hstack([original_features, poly_features, stat_features])
        all_names = list(range(X.shape[1])) + poly_names + stat_names
        
        # Evaluate feature importance
        importance_dict = self._evaluate_feature_importance(
            all_features, y, all_names
        )
        
        # Select features based on importance and correlation
        selected_features = []
        selected_names = []
        
        for name, importance in sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            feature_idx = all_names.index(name)
            candidate = all_features[:, feature_idx].reshape(-1, 1)
            
            if len(selected_features) == 0:
                selected_features.append(candidate)
                selected_names.append(name)
            else:
                temp_features = np.hstack([np.hstack(selected_features), candidate])
                if not self._check_correlation(temp_features):
                    selected_features.append(candidate)
                    selected_names.append(name)
        
        self.selected_features = selected_names
        return np.hstack(selected_features)

# Example usage
np.random.seed(42)
X = np.random.normal(0, 1, (1000, 3))
y = X[:, 0]**2 + np.exp(X[:, 1]) + np.sin(X[:, 2]) + np.random.normal(0, 0.1, 1000)

creator = IntelligentFeatureCreator(max_degree=2)
new_features = creator.fit_transform(X, y)

print("Original Features Shape:", X.shape)
print("New Features Shape:", new_features.shape)
print("Selected Features:", creator.selected_features)
```

Slide 9: Time-Series Feature Generation

Time series feature engineering requires specialized techniques to capture temporal patterns, seasonality, and autocorrelation structures. This implementation creates advanced time-based features while preserving temporal dependencies.

```python
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf

class TimeSeriesFeatureGenerator:
    def __init__(self, max_lag=10, seasonal_periods=[7, 30, 365]):
        self.max_lag = max_lag
        self.seasonal_periods = seasonal_periods
        self.features_info = {}
        
    def _create_lag_features(self, series):
        lags = pd.DataFrame()
        for lag in range(1, self.max_lag + 1):
            lags[f'lag_{lag}'] = series.shift(lag)
        return lags
    
    def _create_rolling_features(self, series):
        windows = [3, 7, 14, 30]
        rolling = pd.DataFrame()
        
        for window in windows:
            rolling[f'rolling_mean_{window}'] = series.rolling(window).mean()
            rolling[f'rolling_std_{window}'] = series.rolling(window).std()
            rolling[f'rolling_min_{window}'] = series.rolling(window).min()
            rolling[f'rolling_max_{window}'] = series.rolling(window).max()
            
        return rolling
    
    def _create_seasonal_features(self, dates):
        seasonal = pd.DataFrame(index=dates)
        
        for period in self.seasonal_periods:
            seasonal[f'seasonal_sin_{period}'] = np.sin(
                2 * np.pi * dates.dayofyear / period
            )
            seasonal[f'seasonal_cos_{period}'] = np.cos(
                2 * np.pi * dates.dayofyear / period
            )
            
        return seasonal
    
    def _create_autocorr_features(self, series):
        # Calculate ACF and PACF
        acf_values = acf(series, nlags=self.max_lag)
        pacf_values = pacf(series, nlags=self.max_lag)
        
        autocorr = pd.DataFrame(index=series.index)
        for lag in range(1, self.max_lag + 1):
            autocorr[f'acf_{lag}'] = acf_values[lag]
            autocorr[f'pacf_{lag}'] = pacf_values[lag]
            
        return autocorr
    
    def fit_transform(self, series, dates=None):
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
            
        if dates is None:
            dates = pd.date_range(
                start='2023-01-01',
                periods=len(series),
                freq='D'
            )
            
        # Generate features
        features = pd.DataFrame(index=series.index)
        
        # Lag features
        lag_features = self._create_lag_features(series)
        features = pd.concat([features, lag_features], axis=1)
        
        # Rolling features
        rolling_features = self._create_rolling_features(series)
        features = pd.concat([features, rolling_features], axis=1)
        
        # Seasonal features
        seasonal_features = self._create_seasonal_features(pd.DatetimeIndex(dates))
        features = pd.concat([features, seasonal_features], axis=1)
        
        # Autocorrelation features
        autocorr_features = self._create_autocorr_features(series)
        features = pd.concat([features, autocorr_features], axis=1)
        
        # Store feature information
        self.features_info = {
            'n_lag_features': len(lag_features.columns),
            'n_rolling_features': len(rolling_features.columns),
            'n_seasonal_features': len(seasonal_features.columns),
            'n_autocorr_features': len(autocorr_features.columns)
        }
        
        return features.fillna(0)

# Example usage
np.random.seed(42)
# Generate sample time series with trend and seasonality
t = np.linspace(0, 365, 365)
trend = 0.1 * t
seasonal = 10 * np.sin(2 * np.pi * t / 365) + 5 * np.sin(2 * np.pi * t / 7)
noise = np.random.normal(0, 1, 365)
series = trend + seasonal + noise

dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
generator = TimeSeriesFeatureGenerator(max_lag=7)
features = generator.fit_transform(series, dates)

print("Generated Features Shape:", features.shape)
print("\nFeature Information:")
for key, value in generator.features_info.items():
    print(f"{key}: {value}")
```

Slide 10: Dimensionality Reduction Feature Engineering

Advanced dimensionality reduction techniques combine linear and non-linear methods to create compact, informative feature representations while preserving the underlying data structure and relationships.

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler

class AdvancedDimensionalityReducer:
    def __init__(self, n_components=2, methods=['pca', 'kpca', 'tsne', 'mds']):
        self.n_components = n_components
        self.methods = methods
        self.transformers = {}
        self.explained_variance_ = {}
        
    def _initialize_transformers(self):
        for method in self.methods:
            if method == 'pca':
                self.transformers[method] = PCA(
                    n_components=self.n_components
                )
            elif method == 'kpca':
                self.transformers[method] = KernelPCA(
                    n_components=self.n_components,
                    kernel='rbf'
                )
            elif method == 'tsne':
                self.transformers[method] = TSNE(
                    n_components=self.n_components,
                    n_iter=1000
                )
            elif method == 'mds':
                self.transformers[method] = MDS(
                    n_components=self.n_components
                )
                
    def _compute_reconstruction_error(self, original, reconstructed):
        return np.mean((original - reconstructed) ** 2)
        
    def fit_transform(self, X):
        X = StandardScaler().fit_transform(X)
        self._initialize_transformers()
        
        reduced_features = {}
        for method, transformer in self.transformers.items():
            # Transform data
            reduced = transformer.fit_transform(X)
            
            # Store explained variance for PCA
            if method == 'pca':
                self.explained_variance_[method] = transformer.explained_variance_ratio_
                
            # For kernel PCA, compute explained variance differently
            elif method == 'kpca':
                eigenvalues = transformer.eigenvalues_
                self.explained_variance_[method] = eigenvalues / np.sum(eigenvalues)
            
            reduced_features[method] = reduced
            
        # Combine features based on explained variance
        combined_features = np.zeros((X.shape[0], self.n_components))
        weights = {}
        
        for method in reduced_features.keys():
            if method in ['pca', 'kpca']:
                weights[method] = np.mean(self.explained_variance_[method])
            else:
                # For non-variance based methods, use equal weights
                weights[method] = 1.0
                
        # Normalize weights
        total_weight = sum(weights.values())
        for method in weights:
            weights[method] /= total_weight
            
        # Compute weighted combination
        for method, weight in weights.items():
            combined_features += weight * reduced_features[method]
            
        return combined_features, reduced_features
    
# Example usage
np.random.seed(42)
# Generate sample data with non-linear structure
t = np.random.uniform(0, 2*np.pi, 1000)
X = np.column_stack([
    np.sin(t) + np.random.normal(0, 0.1, 1000),
    np.cos(t) + np.random.normal(0, 0.1, 1000),
    t + np.random.normal(0, 0.1, 1000)
])

reducer = AdvancedDimensionalityReducer(n_components=2)
combined_features, individual_features = reducer.fit_transform(X)

print("Original Data Shape:", X.shape)
print("Reduced Data Shape:", combined_features.shape)
print("\nExplained Variance Ratios:")
for method, variance in reducer.explained_variance_.items():
    print(f"{method}: {variance}")
```

Slide 11: Real-World Application: Customer Churn Prediction

This comprehensive example demonstrates the integration of multiple feature engineering techniques in a real-world scenario focused on predicting customer churn using telecommunications customer data.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class ChurnFeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        
    def _create_usage_features(self, data):
        usage = pd.DataFrame()
        
        # Usage patterns
        usage['total_charges_to_tenure_ratio'] = data['TotalCharges'] / (data['tenure'] + 1)
        usage['avg_monthly_charges'] = data['MonthlyCharges']
        usage['charges_difference'] = data['TotalCharges'] - (data['MonthlyCharges'] * data['tenure'])
        
        # Service usage intensity
        services = ['PhoneService', 'InternetService', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                   'StreamingTV', 'StreamingMovies']
        
        usage['total_services'] = data[services].apply(
            lambda x: sum(x != 'No' and x != 'No internet service'), axis=1
        )
        
        return usage
    
    def _create_customer_features(self, data):
        customer = pd.DataFrame()
        
        # Contract risk score
        contract_risk = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
        customer['contract_risk'] = data['Contract'].map(contract_risk)
        
        # Payment complexity
        payment_methods = pd.get_dummies(data['PaymentMethod'], prefix='payment')
        customer = pd.concat([customer, payment_methods], axis=1)
        
        # Dependency score
        customer['dependency_score'] = (
            (data['Partner'] == 'Yes').astype(int) + 
            (data['Dependents'] == 'Yes').astype(int)
        )
        
        return customer
    
    def _create_tenure_features(self, data):
        tenure = pd.DataFrame()
        
        # Tenure segments
        tenure['tenure_years'] = data['tenure'] // 12
        tenure['tenure_quarter'] = data['tenure'] % 12 // 3
        
        # Create tenure interaction features
        tenure['tenure_by_contract'] = data['tenure'] * (
            data['Contract'].map({
                'Month-to-month': 1,
                'One year': 2,
                'Two year': 3
            })
        )
        
        return tenure
    
    def fit_transform(self, data, target=None):
        # Create feature groups
        usage_features = self._create_usage_features(data)
        customer_features = self._create_customer_features(data)
        tenure_features = self._create_tenure_features(data)
        
        # Combine all features
        features = pd.concat(
            [usage_features, customer_features, tenure_features],
            axis=1
        )
        
        # Scale numerical features
        numerical_cols = features.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        features[numerical_cols] = scaler.fit_transform(features[numerical_cols])
        
        self.feature_names = features.columns.tolist()
        return features
    
# Example usage with synthetic data
def generate_synthetic_telco_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate basic customer data
    data = pd.DataFrame({
        'tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(100, 8000, n_samples),
        'Contract': np.random.choice(
            ['Month-to-month', 'One year', 'Two year'],
            n_samples
        ),
        'PaymentMethod': np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
            n_samples
        ),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'InternetService': np.random.choice(
            ['DSL', 'Fiber optic', 'No'],
            n_samples
        )
    })
    
    # Generate target variable (Churn)
    contract_weights = {
        'Month-to-month': 0.3,
        'One year': 0.15,
        'Two year': 0.05
    }
    
    churn_prob = data['Contract'].map(contract_weights)
    data['Churn'] = np.random.binomial(1, churn_prob)
    
    return data

# Generate synthetic data and apply feature engineering
data = generate_synthetic_telco_data(1000)
target = data['Churn']
data = data.drop('Churn', axis=1)

# Apply feature engineering
engineer = ChurnFeatureEngineer()
features = engineer.fit_transform(data)

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Train and evaluate model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Feature Engineering Results:")
print(f"Number of original features: {data.shape[1]}")
print(f"Number of engineered features: {features.shape[1]}")
print("\nModel Performance Report:")
print(classification_report(y_test, y_pred))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': engineer.feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
```

Slide 12: Real-World Application: Financial Time Series Feature Engineering

This implementation demonstrates sophisticated feature engineering techniques for financial market data, incorporating technical indicators, market microstructure features, and cross-asset relationships.

```python
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import talib

class FinancialFeatureEngineer:
    def __init__(self, window_sizes=[5, 10, 20, 50]):
        self.window_sizes = window_sizes
        self.features = {}
        
    def _calculate_technical_indicators(self, data):
        tech_features = pd.DataFrame(index=data.index)
        
        # Basic technical indicators
        tech_features['rsi'] = talib.RSI(data['close'].values)
        tech_features['macd'], _, _ = talib.MACD(data['close'].values)
        tech_features['atr'] = talib.ATR(
            data['high'].values,
            data['low'].values,
            data['close'].values
        )
        
        # Multiple timeframe momentum
        for window in self.window_sizes:
            # Price momentum
            tech_features[f'momentum_{window}'] = (
                data['close'] / data['close'].shift(window) - 1
            )
            
            # Volatility
            tech_features[f'volatility_{window}'] = (
                data['close'].rolling(window).std() / 
                data['close'].rolling(window).mean()
            )
            
            # Volume momentum
            tech_features[f'volume_momentum_{window}'] = (
                data['volume'] / data['volume'].shift(window) - 1
            )
        
        return tech_features
    
    def _calculate_microstructure_features(self, data):
        micro_features = pd.DataFrame(index=data.index)
        
        # Bid-ask spread features (if available)
        if all(col in data.columns for col in ['bid', 'ask']):
            micro_features['spread'] = (data['ask'] - data['bid']) / data['bid']
            micro_features['spread_ma'] = micro_features['spread'].rolling(5).mean()
        
        # Volume-based features
        micro_features['volume_intensity'] = (
            data['volume'] / data['volume'].rolling(20).mean()
        )
        
        # Price impact
        micro_features['amihud_illiquidity'] = (
            np.abs(data['close'].pct_change()) / (data['volume'] * data['close'])
        )
        
        return micro_features
    
    def _calculate_orderbook_features(self, data, levels=5):
        ob_features = pd.DataFrame(index=data.index)
        
        if all(col in data.columns for col in [f'bid_{i}' for i in range(levels)] +
               [f'ask_{i}' for i in range(levels)]):
            # Order book imbalance
            total_bid_volume = sum(
                data[f'bid_volume_{i}'] for i in range(levels)
            )
            total_ask_volume = sum(
                data[f'ask_volume_{i}'] for i in range(levels)
            )
            
            ob_features['ob_imbalance'] = (
                (total_bid_volume - total_ask_volume) /
                (total_bid_volume + total_ask_volume)
            )
            
            # Price pressure
            ob_features['weighted_bid_pressure'] = sum(
                data[f'bid_volume_{i}'] * (i + 1) for i in range(levels)
            ) / total_bid_volume
            
            ob_features['weighted_ask_pressure'] = sum(
                data[f'ask_volume_{i}'] * (i + 1) for i in range(levels)
            ) / total_ask_volume
        
        return ob_features
    
    def transform(self, data):
        # Technical indicators
        tech_features = self._calculate_technical_indicators(data)
        
        # Market microstructure features
        micro_features = self._calculate_microstructure_features(data)
        
        # Order book features (if available)
        ob_features = self._calculate_orderbook_features(data)
        
        # Combine all features
        features = pd.concat(
            [tech_features, micro_features, ob_features],
            axis=1
        )
        
        return features.fillna(0)

# Example usage with synthetic data
def generate_synthetic_market_data(n_samples=1000):
    np.random.seed(42)
    index = pd.date_range(
        start='2023-01-01',
        periods=n_samples,
        freq='5min'
    )
    
    # Generate OHLCV data
    base_price = 100
    volatility = 0.002
    prices = base_price * np.exp(
        np.random.normal(0, volatility, n_samples).cumsum()
    )
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples),
        'bid': prices * 0.9999,
        'ask': prices * 1.0001
    }, index=index)
    
    return data

# Generate synthetic data and engineer features
market_data = generate_synthetic_market_data()
engineer = FinancialFeatureEngineer()
features = engineer.transform(market_data)

print("Feature Engineering Results:")
print(f"Number of original features: {market_data.shape[1]}")
print(f"Number of engineered features: {features.shape[1]}")

# Display sample feature statistics
print("\nFeature Statistics:")
print(features.describe())

# Correlation analysis
correlation_matrix = features.corr()
print("\nHighly Correlated Features (|correlation| > 0.8):")
high_corr = np.where(np.abs(correlation_matrix) > 0.8)
for i, j in zip(*high_corr):
    if i < j:
        print(f"{features.columns[i]} - {features.columns[j]}: {correlation_matrix.iloc[i, j]:.2f}")
```

Slide 13: Advanced Feature Selection Framework

This implementation provides a comprehensive framework for feature selection combining multiple techniques including statistical tests, machine learning importance scores, and stability metrics to identify the most robust and relevant features.

```python
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from scipy import stats

class AdvancedFeatureSelector:
    def __init__(self, n_features=10, n_splits=5):
        self.n_features = n_features
        self.n_splits = n_splits
        self.selected_features = None
        self.feature_scores = {}
        
    def _calculate_statistical_scores(self, X, y):
        # Statistical tests
        f_scores, _ = f_classif(X, y)
        mi_scores = mutual_info_classif(X, y)
        
        return {
            'f_score': f_scores,
            'mutual_info': mi_scores
        }
    
    def _calculate_model_based_scores(self, X, y):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        return {
            'random_forest': rf.feature_importances_
        }
    
    def _calculate_stability_scores(self, X, y):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        stability_scores = np.zeros((X.shape[1], self.n_splits))
        
        for i, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, y_train = X[train_idx], y[train_idx]
            
            # Calculate feature importance for this fold
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            stability_scores[:, i] = rf.feature_importances_
            
        # Calculate stability metrics
        stability = {
            'mean': np.mean(stability_scores, axis=1),
            'std': np.std(stability_scores, axis=1),
            'cv': np.std(stability_scores, axis=1) / np.mean(stability_scores, axis=1)
        }
        
        return stability
    
    def _rank_features(self, scores_dict):
        rankings = pd.DataFrame()
        
        # Combine all scores
        for method, scores in scores_dict.items():
            if isinstance(scores, dict):
                for sub_method, sub_scores in scores.items():
                    rankings[f'{method}_{sub_method}'] = pd.Series(sub_scores)
            else:
                rankings[method] = pd.Series(scores)
        
        # Normalize scores
        rankings = (rankings - rankings.mean()) / rankings.std()
        
        # Calculate final score (weighted average)
        weights = {
            'f_score': 0.2,
            'mutual_info': 0.2,
            'random_forest': 0.3,
            'stability_mean': 0.2,
            'stability_cv': -0.1  # Negative weight for coefficient of variation
        }
        
        final_scores = np.zeros(len(rankings))
        for method, weight in weights.items():
            if method in rankings.columns:
                final_scores += weight * rankings[method]
        
        return final_scores
    
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            X = X.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Calculate various feature importance scores
        statistical_scores = self._calculate_statistical_scores(X, y)
        model_scores = self._calculate_model_based_scores(X, y)
        stability_scores = self._calculate_stability_scores(X, y)
        
        # Combine all scores
        all_scores = {
            'statistical': statistical_scores,
            'model': model_scores,
            'stability': stability_scores
        }
        
        # Rank features
        final_scores = self._rank_features(all_scores)
        
        # Select top features
        top_indices = np.argsort(final_scores)[-self.n_features:]
        self.selected_features = self.feature_names[top_indices]
        
        # Store feature scores
        self.feature_scores = pd.DataFrame({
            'feature': self.feature_names,
            'score': final_scores
        }).sort_values('score', ascending=False)
        
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features]
        else:
            return X[:, [list(self.feature_names).index(f) for f in self.selected_features]]

# Example usage
def generate_synthetic_data(n_samples=1000, n_features=20):
    np.random.seed(42)
    
    # Generate informative features
    X_informative = np.random.normal(0, 1, (n_samples, 5))
    y = (X_informative[:, 0] + X_informative[:, 1] > 0).astype(int)
    
    # Generate noise features
    X_noise = np.random.normal(0, 1, (n_samples, n_features - 5))
    
    # Combine features
    X = np.hstack([X_informative, X_noise])
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X = pd.DataFrame(X, columns=feature_names)
    
    return X, y

# Generate synthetic data
X, y = generate_synthetic_data()

# Apply feature selection
selector = AdvancedFeatureSelector(n_features=5)
selector.fit(X, y)

print("Selected Features:")
print(selector.selected_features)
print("\nFeature Scores:")
print(selector.feature_scores)
```

Slide 14: Additional Resources

*   "Feature Engineering for Machine Learning: Principles and Techniques" - ArXiv: [https://arxiv.org/abs/2007.03553](https://arxiv.org/abs/2007.03553)
*   "Deep Feature Engineering: Best Practices and Emerging Trends" - [https://www.researchgate.net/publication/advanced\_feature\_engineering](https://www.researchgate.net/publication/advanced_feature_engineering)
*   "Automated Feature Engineering: State of the Art and Future Directions" - [https://research.google/pubs/automated\_feature\_engineering](https://research.google/pubs/automated_feature_engineering)
*   "Robust Feature Selection Methods: A Comprehensive Survey" - [https://papers.with.ai/feature\_selection\_survey](https://papers.with.ai/feature_selection_survey)
*   "Time Series Feature Engineering: Techniques and Applications" - [https://dl.acm.org/doi/time\_series\_engineering](https://dl.acm.org/doi/time_series_engineering)

Note: Search for these topics on Google Scholar or arXiv for the most recent and relevant papers in feature engineering.

