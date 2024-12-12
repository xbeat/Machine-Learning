## The Importance of Data in Deep Learning
Slide 1: Data Quality Assessment

Data quality assessment is fundamental in deep learning pipelines. This implementation demonstrates how to evaluate dataset characteristics including missing values, statistical distributions, and potential biases that could impact model performance.

```python
import pandas as pd
import numpy as np
from scipy import stats

def assess_data_quality(dataset):
    # Calculate missing values percentage
    missing_percent = (dataset.isnull().sum() / len(dataset)) * 100
    
    # Check for statistical outliers using z-score
    z_scores = stats.zscore(dataset.select_dtypes(include=[np.number]))
    outliers = np.abs(z_scores) > 3
    
    # Calculate basic statistics
    statistics = dataset.describe()
    
    # Check for class imbalance if target column exists
    if 'target' in dataset.columns:
        class_distribution = dataset['target'].value_counts(normalize=True)
    
    return {
        'missing_values': missing_percent,
        'outliers_count': outliers.sum(),
        'statistics': statistics
    }

# Example usage
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(5, 2, 1000),
    'target': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
})

quality_metrics = assess_data_quality(data)
print(f"Missing values:\n{quality_metrics['missing_values']}")
print(f"\nOutliers per feature:\n{quality_metrics['outliers_count']}")
```

Slide 2: Data Preprocessing Pipeline

A robust preprocessing pipeline ensures data consistency and optimal model performance. This implementation showcases essential preprocessing steps including normalization, encoding, and handling missing values.

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class PreprocessingPipeline:
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def fit_transform(self, df):
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Handle numerical features
        df_num = pd.DataFrame(self.numerical_imputer.fit_transform(df[numerical_cols]),
                            columns=numerical_cols)
        df_num = pd.DataFrame(self.scaler.fit_transform(df_num), 
                            columns=numerical_cols)
        
        # Handle categorical features
        df_cat = df[categorical_cols].copy()
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            df_cat[col] = self.categorical_imputer.fit_transform(
                df_cat[[col]])
            df_cat[col] = self.label_encoders[col].fit_transform(
                df_cat[col].astype(str))
        
        return pd.concat([df_num, df_cat], axis=1)

# Example usage
df = pd.DataFrame({
    'age': [25, np.nan, 30, 35],
    'income': [50000, 60000, np.nan, 75000],
    'category': ['A', 'B', np.nan, 'A']
})

pipeline = PreprocessingPipeline()
processed_df = pipeline.fit_transform(df)
print("Processed dataset:\n", processed_df)
```

Slide 3: Data Augmentation Techniques

Data augmentation helps increase dataset size and diversity, crucial for model generalization. This implementation demonstrates various augmentation techniques for different data types including numerical and categorical features.

```python
import numpy as np
from scipy.interpolate import interp1d

class DataAugmenter:
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level
    
    def add_gaussian_noise(self, data):
        noise = np.random.normal(0, self.noise_level, data.shape)
        return data + noise
    
    def smote_like_synthesis(self, data, num_synthetic):
        synthetic_samples = []
        for _ in range(num_synthetic):
            idx1, idx2 = np.random.choice(len(data), 2, replace=False)
            alpha = np.random.random()
            synthetic = data[idx1] + alpha * (data[idx2] - data[idx1])
            synthetic_samples.append(synthetic)
        return np.array(synthetic_samples)
    
    def time_warping(self, sequence, num_synthetic):
        time = np.arange(len(sequence))
        warped_sequences = []
        for _ in range(num_synthetic):
            # Generate random warping
            warp = np.random.normal(1, 0.1, len(sequence))
            warped_time = np.cumsum(warp)
            # Interpolate
            f = interp1d(time, sequence)
            warped_seq = f(np.linspace(0, len(sequence)-1, len(sequence)))
            warped_sequences.append(warped_seq)
        return np.array(warped_sequences)

# Example usage
data = np.random.randn(100, 5)  # Original dataset
augmenter = DataAugmenter()

# Apply different augmentation techniques
noisy_data = augmenter.add_gaussian_noise(data)
synthetic_data = augmenter.smote_like_synthesis(data, 50)
sequence = np.sin(np.linspace(0, 10, 100))
warped_sequences = augmenter.time_warping(sequence, 5)

print("Original data shape:", data.shape)
print("Synthetic data shape:", synthetic_data.shape)
print("Warped sequences shape:", warped_sequences.shape)
```

Slide 4: Feature Engineering Framework

Feature engineering transforms raw data into meaningful representations that enhance model performance. This framework implements various feature extraction techniques including statistical measures, temporal features, and interaction terms.

```python
class FeatureEngineer:
    def __init__(self):
        self.interaction_features = []
        self.temporal_features = []
    
    def create_statistical_features(self, df, group_col, target_col):
        stats = df.groupby(group_col)[target_col].agg([
            'mean', 'std', 'min', 'max', 
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75))
        ]).reset_index()
        return stats
    
    def create_interaction_terms(self, df, feature_pairs):
        for f1, f2 in feature_pairs:
            name = f"{f1}_{f2}_interaction"
            df[name] = df[f1] * df[f2]
            self.interaction_features.append(name)
        return df
    
    def create_temporal_features(self, df, date_column):
        df[date_column] = pd.to_datetime(df[date_column])
        df['hour'] = df[date_column].dt.hour
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['month'] = df[date_column].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        self.temporal_features = ['hour', 'day_of_week', 'month', 'is_weekend']
        return df

# Example usage
data = pd.DataFrame({
    'date': pd.date_range(start='2024-01-01', periods=1000, freq='H'),
    'value1': np.random.normal(0, 1, 1000),
    'value2': np.random.normal(0, 1, 1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000)
})

engineer = FeatureEngineer()

# Create temporal features
data = engineer.create_temporal_features(data, 'date')

# Create interaction terms
data = engineer.create_interaction_terms(data, [('value1', 'value2')])

# Create statistical features
stats = engineer.create_statistical_features(data, 'category', 'value1')

print("Engineered features:\n", data.head())
print("\nStatistical features:\n", stats)
```

Slide 5: Advanced Data Validation Framework

A comprehensive data validation framework ensures data consistency and quality throughout the machine learning pipeline. This implementation includes schema validation, constraint checking, and distribution monitoring.

```python
from typing import Dict, List, Any
import numpy as np
from scipy import stats

class DataValidator:
    def __init__(self, schema: Dict[str, Dict[str, Any]]):
        self.schema = schema
        self.validation_results = {}
        
    def validate_types(self, df):
        type_errors = []
        for column, properties in self.schema.items():
            expected_type = properties['type']
            if not df[column].dtype == expected_type:
                type_errors.append(f"Column {column} has type {df[column].dtype}, expected {expected_type}")
        return type_errors
    
    def validate_ranges(self, df):
        range_errors = []
        for column, properties in self.schema.items():
            if 'range' in properties:
                min_val, max_val = properties['range']
                if df[column].min() < min_val or df[column].max() > max_val:
                    range_errors.append(f"Column {column} contains values outside range [{min_val}, {max_val}]")
        return range_errors
    
    def validate_distributions(self, df, significance_level=0.05):
        distribution_tests = {}
        for column, properties in self.schema.items():
            if 'distribution' in properties:
                expected_dist = properties['distribution']
                data = df[column].dropna()
                
                if expected_dist == 'normal':
                    _, p_value = stats.normaltest(data)
                    distribution_tests[column] = {
                        'test': 'normal',
                        'p_value': p_value,
                        'passed': p_value > significance_level
                    }
        return distribution_tests
    
    def validate_dataset(self, df):
        self.validation_results = {
            'type_validation': self.validate_types(df),
            'range_validation': self.validate_ranges(df),
            'distribution_validation': self.validate_distributions(df)
        }
        return self.validation_results

# Example usage
schema = {
    'age': {
        'type': np.int64,
        'range': (0, 120),
        'distribution': 'normal'
    },
    'income': {
        'type': np.float64,
        'range': (0, 1000000),
        'distribution': 'normal'
    }
}

data = pd.DataFrame({
    'age': np.random.normal(35, 10, 1000).astype(int),
    'income': np.random.normal(50000, 10000, 1000)
})

validator = DataValidator(schema)
validation_results = validator.validate_dataset(data)

print("Validation Results:")
for check_type, results in validation_results.items():
    print(f"\n{check_type}:")
    print(results)
```

Slide 6: Data Sampling Strategies

Implementing effective sampling strategies is crucial for handling imbalanced datasets and creating representative training sets. This implementation showcases various sampling techniques including stratified, weighted, and adaptive sampling methods.

```python
class AdvancedSampler:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def stratified_sample(self, df, strata_col, size=None, proportions=None):
        if proportions is None:
            proportions = df[strata_col].value_counts(normalize=True)
            
        sampled_data = []
        for stratum in proportions.index:
            stratum_data = df[df[strata_col] == stratum]
            stratum_size = int(size * proportions[stratum]) if size else len(stratum_data)
            sampled_stratum = stratum_data.sample(
                n=min(stratum_size, len(stratum_data)),
                random_state=self.random_state
            )
            sampled_data.append(sampled_stratum)
            
        return pd.concat(sampled_data, axis=0)
    
    def weighted_sample(self, df, weights_col, size):
        weights = df[weights_col] / df[weights_col].sum()
        return df.sample(
            n=size,
            weights=weights,
            random_state=self.random_state
        )
    
    def adaptive_sample(self, df, target_col, difficulty_func):
        # Calculate sample difficulty scores
        difficulties = df.apply(lambda x: difficulty_func(x), axis=1)
        
        # Compute adaptive weights based on difficulty
        weights = 1 + np.exp(difficulties - difficulties.mean())
        weights = weights / weights.sum()
        
        # Sample based on adaptive weights
        return self.weighted_sample(df, weights, size=len(df))

# Example usage
# Create sample dataset
data = pd.DataFrame({
    'feature': np.random.normal(0, 1, 1000),
    'target': np.random.choice(['A', 'B', 'C'], 1000, p=[0.6, 0.3, 0.1]),
    'importance': np.random.uniform(0, 1, 1000)
})

# Define a difficulty function
def sample_difficulty(row):
    return abs(row['feature'])  # Simple difficulty metric

sampler = AdvancedSampler()

# Stratified sampling
stratified = sampler.stratified_sample(
    data, 
    'target', 
    size=500
)

# Weighted sampling
weighted = sampler.weighted_sample(
    data, 
    'importance',
    size=500
)

# Adaptive sampling
adaptive = sampler.adaptive_sample(
    data,
    'target',
    sample_difficulty
)

print("Original class distribution:\n", data['target'].value_counts(normalize=True))
print("\nStratified sample distribution:\n", stratified['target'].value_counts(normalize=True))
print("\nWeighted sample distribution:\n", weighted['target'].value_counts(normalize=True))
```

Slide 7: Data Integrity Monitoring

Continuous monitoring of data integrity is essential for maintaining model performance. This implementation provides a framework for tracking data drift, feature correlations, and statistical stability over time.

```python
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

class DataIntegrityMonitor:
    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.reference_stats = self._compute_statistics(reference_data)
        
    def _compute_statistics(self, df):
        stats = {
            'means': df.mean(),
            'stds': df.std(),
            'correlations': df.corr(),
            'distributions': {col: df[col].value_counts(normalize=True)
                            for col in df.columns}
        }
        return stats
    
    def detect_drift(self, new_data, threshold=0.05):
        drift_results = {}
        
        # Distribution drift using KS-test
        for column in self.reference_data.columns:
            if new_data[column].dtype in ['int64', 'float64']:
                statistic, p_value = ks_2samp(
                    self.reference_data[column],
                    new_data[column]
                )
                drift_results[column] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'drift_detected': p_value < threshold
                }
        
        return drift_results
    
    def measure_stability(self, new_data):
        stability_metrics = {}
        new_stats = self._compute_statistics(new_data)
        
        # Compare means and standard deviations
        stability_metrics['mean_shifts'] = (
            new_stats['means'] - self.reference_stats['means']
        ) / self.reference_stats['stds']
        
        # Compare correlations
        stability_metrics['correlation_changes'] = (
            new_stats['correlations'] - self.reference_stats['correlations']
        ).abs()
        
        # Compare distributions using Jensen-Shannon divergence
        stability_metrics['distribution_divergence'] = {}
        for col in self.reference_data.columns:
            ref_dist = self.reference_stats['distributions'][col]
            new_dist = new_stats['distributions'][col]
            # Align distributions
            all_categories = ref_dist.index.union(new_dist.index)
            ref_aligned = ref_dist.reindex(all_categories, fill_value=0)
            new_aligned = new_dist.reindex(all_categories, fill_value=0)
            
            divergence = jensenshannon(ref_aligned, new_aligned)
            stability_metrics['distribution_divergence'][col] = divergence
            
        return stability_metrics

# Example usage
reference_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(5, 2, 1000)
})

# Simulate new data with drift
new_data = pd.DataFrame({
    'feature1': np.random.normal(0.5, 1.2, 1000),  # Introduced drift
    'feature2': np.random.normal(5, 2, 1000)       # Stable feature
})

monitor = DataIntegrityMonitor(reference_data)

# Detect drift
drift_results = monitor.detect_drift(new_data)
print("Drift Detection Results:")
for feature, results in drift_results.items():
    print(f"\n{feature}:")
    print(f"Drift detected: {results['drift_detected']}")
    print(f"P-value: {results['p_value']:.4f}")

# Measure stability
stability_metrics = monitor.measure_stability(new_data)
print("\nStability Metrics:")
print("Mean shifts:\n", stability_metrics['mean_shifts'])
print("\nDistribution divergence:\n", stability_metrics['distribution_divergence'])
```

Slide 8: Advanced Data Filtering Framework

Implementing sophisticated data filtering techniques ensures dataset quality by removing noise, outliers, and inconsistent samples. This framework provides methods for both statistical and model-based filtering approaches.

```python
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

class AdvancedDataFilter:
    def __init__(self):
        self.outlier_detectors = {}
        self.filter_stats = {}
        
    def statistical_filter(self, df, columns, n_std=3):
        filtered_df = df.copy()
        stats = {}
        
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - n_std * std
            upper_bound = mean + n_std * std
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            filtered_df = filtered_df[mask]
            
            stats[col] = {
                'removed_samples': (~mask).sum(),
                'bounds': (lower_bound, upper_bound)
            }
            
        self.filter_stats['statistical'] = stats
        return filtered_df
    
    def isolation_forest_filter(self, df, columns, contamination=0.1):
        X = df[columns]
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(X)
        
        self.outlier_detectors['isolation_forest'] = iso_forest
        self.filter_stats['isolation_forest'] = {
            'removed_samples': (outlier_labels == -1).sum()
        }
        
        return df[outlier_labels == 1]
    
    def robust_covariance_filter(self, df, columns, contamination=0.1):
        X = df[columns]
        envelope = EllipticEnvelope(contamination=contamination, random_state=42)
        outlier_labels = envelope.fit_predict(X)
        
        self.outlier_detectors['robust_covariance'] = envelope
        self.filter_stats['robust_covariance'] = {
            'removed_samples': (outlier_labels == -1).sum()
        }
        
        return df[outlier_labels == 1]
    
    def ensemble_filter(self, df, columns, methods=['statistical', 'isolation_forest'],
                       n_std=3, contamination=0.1):
        filtered_df = df.copy()
        
        if 'statistical' in methods:
            filtered_df = self.statistical_filter(filtered_df, columns, n_std)
            
        if 'isolation_forest' in methods:
            filtered_df = self.isolation_forest_filter(filtered_df, columns, contamination)
            
        if 'robust_covariance' in methods:
            filtered_df = self.robust_covariance_filter(filtered_df, columns, contamination)
            
        return filtered_df

# Example usage
# Create sample dataset with outliers
np.random.seed(42)
n_samples = 1000
n_outliers = 50

data = pd.DataFrame({
    'feature1': np.concatenate([
        np.random.normal(0, 1, n_samples-n_outliers),
        np.random.normal(0, 5, n_outliers)
    ]),
    'feature2': np.concatenate([
        np.random.normal(0, 1, n_samples-n_outliers),
        np.random.normal(0, 5, n_outliers)
    ])
})

filter_engine = AdvancedDataFilter()

# Apply different filtering methods
columns = ['feature1', 'feature2']
filtered_statistical = filter_engine.statistical_filter(data, columns)
filtered_iforest = filter_engine.isolation_forest_filter(data, columns)
filtered_ensemble = filter_engine.ensemble_filter(data, columns)

print("Original data shape:", data.shape)
print("After statistical filtering:", filtered_statistical.shape)
print("After Isolation Forest:", filtered_iforest.shape)
print("After ensemble filtering:", filtered_ensemble.shape)
print("\nFiltering statistics:", filter_engine.filter_stats)
```

Slide 9: Time Series Data Processing

Time series data requires specialized preprocessing techniques to capture temporal dependencies and patterns. This implementation demonstrates advanced time series processing including seasonal decomposition and feature extraction.

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fftpack import fft

class TimeSeriesProcessor:
    def __init__(self, seasonality_period=None):
        self.seasonality_period = seasonality_period
        self.decomposition_results = None
        
    def extract_temporal_features(self, df, datetime_col):
        df = df.copy()
        dt = pd.to_datetime(df[datetime_col])
        
        # Basic temporal features
        df['hour'] = dt.dt.hour
        df['day_of_week'] = dt.dt.dayofweek
        df['month'] = dt.dt.month
        df['year'] = dt.dt.year
        df['quarter'] = dt.dt.quarter
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        return df
    
    def decompose_series(self, series):
        decomposition = seasonal_decompose(
            series,
            period=self.seasonality_period,
            extrapolate_trend='freq'
        )
        
        self.decomposition_results = {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }
        
        return self.decomposition_results
    
    def extract_spectral_features(self, series, num_components=10):
        # Compute FFT
        fft_vals = fft(series.values)
        fft_abs = np.abs(fft_vals)[:num_components]
        fft_phase = np.angle(fft_vals)[:num_components]
        
        spectral_features = {
            'fft_magnitude': fft_abs,
            'fft_phase': fft_phase,
            'dominant_frequencies': np.argsort(fft_abs)[-3:]
        }
        
        return spectral_features
    
    def create_lagged_features(self, series, lags):
        df_lagged = pd.DataFrame(index=series.index)
        
        for lag in lags:
            df_lagged[f'lag_{lag}'] = series.shift(lag)
            
        # Add rolling statistics
        df_lagged['rolling_mean_7'] = series.rolling(window=7).mean()
        df_lagged['rolling_std_7'] = series.rolling(window=7).std()
        
        return df_lagged.dropna()

# Example usage
# Create sample time series data
dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
values = np.sin(np.linspace(0, 10*np.pi, 1000)) + \
         0.5 * np.sin(np.linspace(0, 50*np.pi, 1000)) + \
         np.random.normal(0, 0.2, 1000)

data = pd.DataFrame({
    'timestamp': dates,
    'value': values
})

processor = TimeSeriesProcessor(seasonality_period=24)  # 24 hours seasonality

# Extract temporal features
temporal_features = processor.extract_temporal_features(data, 'timestamp')

# Decompose series
decomposition = processor.decompose_series(data['value'])

# Extract spectral features
spectral_features = processor.extract_spectral_features(data['value'])

# Create lagged features
lagged_features = processor.create_lagged_features(data['value'], [1, 2, 3, 24])

print("Temporal features shape:", temporal_features.shape)
print("Decomposition components:", list(decomposition.keys()))
print("Spectral features:", spectral_features['dominant_frequencies'])
print("Lagged features shape:", lagged_features.shape)
```

Slide 10: Advanced Dataset Partitioning

Dataset partitioning requires sophisticated strategies beyond simple random splits to ensure representative samples across all data subsets. This implementation provides methods for temporal, stratified, and group-aware splitting.

```python
class AdvancedDataPartitioner:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.split_stats = {}
        
    def temporal_split(self, df, timestamp_col, train_ratio=0.7, val_ratio=0.15):
        df = df.sort_values(timestamp_col)
        n = len(df)
        
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        train = df.iloc[:train_idx]
        val = df.iloc[train_idx:val_idx]
        test = df.iloc[val_idx:]
        
        self.split_stats['temporal'] = {
            'train_period': (train[timestamp_col].min(), train[timestamp_col].max()),
            'val_period': (val[timestamp_col].min(), val[timestamp_col].max()),
            'test_period': (test[timestamp_col].min(), test[timestamp_col].max())
        }
        
        return train, val, test
    
    def stratified_group_split(self, df, group_col, strata_col, 
                             train_ratio=0.7, val_ratio=0.15):
        groups = df[group_col].unique()
        strata = df[strata_col].unique()
        
        # Calculate target distribution
        target_dist = df[strata_col].value_counts(normalize=True)
        
        # Split groups while maintaining stratification
        train_groups, remain_groups = [], []
        current_dist = pd.Series(0, index=strata)
        
        np.random.shuffle(groups)
        
        for group in groups:
            group_data = df[df[group_col] == group]
            group_dist = group_data[strata_col].value_counts(normalize=True)
            
            # Calculate distribution if we add this group to train
            temp_dist = (current_dist * len(train_groups) + group_dist) / (len(train_groups) + 1)
            
            if len(train_groups) < len(groups) * train_ratio and \
               np.abs(temp_dist - target_dist).mean() < 0.1:
                train_groups.append(group)
                current_dist = temp_dist
            else:
                remain_groups.append(group)
        
        # Split remaining groups into val and test
        val_size = int(len(groups) * val_ratio)
        val_groups = remain_groups[:val_size]
        test_groups = remain_groups[val_size:]
        
        train = df[df[group_col].isin(train_groups)]
        val = df[df[group_col].isin(val_groups)]
        test = df[df[group_col].isin(test_groups)]
        
        self.split_stats['stratified_group'] = {
            'train_groups': len(train_groups),
            'val_groups': len(val_groups),
            'test_groups': len(test_groups),
            'train_distribution': train[strata_col].value_counts(normalize=True),
            'val_distribution': val[strata_col].value_counts(normalize=True),
            'test_distribution': test[strata_col].value_counts(normalize=True)
        }
        
        return train, val, test
    
# Example usage
# Create sample dataset
dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
groups = np.random.choice(range(50), size=1000)
strata = np.random.choice(['A', 'B', 'C'], size=1000, p=[0.5, 0.3, 0.2])

data = pd.DataFrame({
    'timestamp': dates,
    'group': groups,
    'strata': strata,
    'value': np.random.normal(0, 1, 1000)
})

partitioner = AdvancedDataPartitioner()

# Temporal split
train_temporal, val_temporal, test_temporal = partitioner.temporal_split(
    data, 'timestamp'
)

# Stratified group split
train_strat, val_strat, test_strat = partitioner.stratified_group_split(
    data, 'group', 'strata'
)

print("Temporal Split Statistics:")
print(partitioner.split_stats['temporal'])
print("\nStratified Group Split Statistics:")
print(partitioner.split_stats['stratified_group'])
```

Slide 11: Data Version Control System

Implementing a robust data version control system is crucial for maintaining data lineage and reproducibility in machine learning pipelines. This implementation provides methods for tracking data transformations and maintaining version history.

```python
import hashlib
import json
from datetime import datetime

class DataVersionControl:
    def __init__(self, storage_path='./.dvc'):
        self.storage_path = storage_path
        self.version_history = {}
        self.current_version = None
        
    def _compute_hash(self, df):
        # Compute hash of dataframe content
        df_bytes = pd.util.hash_pandas_object(df).values.tobytes()
        return hashlib.sha256(df_bytes).hexdigest()
    
    def _create_version_metadata(self, df, description):
        return {
            'timestamp': datetime.now().isoformat(),
            'hash': self._compute_hash(df),
            'shape': df.shape,
            'columns': list(df.columns),
            'description': description,
            'parent_version': self.current_version
        }
    
    def commit(self, df, description):
        version_id = self._compute_hash(df)[:8]
        metadata = self._create_version_metadata(df, description)
        
        self.version_history[version_id] = metadata
        self.current_version = version_id
        
        return version_id
    
    def get_lineage(self, version_id):
        lineage = []
        current = version_id
        
        while current is not None:
            lineage.append({
                'version': current,
                'metadata': self.version_history[current]
            })
            current = self.version_history[current]['parent_version']
            
        return lineage
    
    def compare_versions(self, version_id1, version_id2):
        v1 = self.version_history[version_id1]
        v2 = self.version_history[version_id2]
        
        differences = {
            'columns_added': list(set(v2['columns']) - set(v1['columns'])),
            'columns_removed': list(set(v1['columns']) - set(v2['columns'])),
            'shape_change': (
                v2['shape'][0] - v1['shape'][0],
                v2['shape'][1] - v1['shape'][1]
            )
        }
        
        return differences
    
    def export_history(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.version_history, f, indent=2)

# Example usage
# Create sample datasets with modifications
initial_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(0, 1, 1000)
})

modified_data = initial_data.copy()
modified_data['feature3'] = np.random.normal(0, 1, 1000)

filtered_data = modified_data[modified_data['feature1'] > 0]

# Initialize version control
dvc = DataVersionControl()

# Track versions
v1 = dvc.commit(initial_data, "Initial dataset")
v2 = dvc.commit(modified_data, "Added feature3")
v3 = dvc.commit(filtered_data, "Filtered by feature1")

# Get lineage
lineage = dvc.get_lineage(v3)
print("Data Lineage:")
for entry in lineage:
    print(f"\nVersion: {entry['version']}")
    print(f"Description: {entry['metadata']['description']}")
    print(f"Shape: {entry['metadata']['shape']}")

# Compare versions
diff = dvc.compare_versions(v1, v2)
print("\nDifferences between v1 and v2:")
print(json.dumps(diff, indent=2))
```

Slide 12: Data Quality Metrics Implementation

This implementation provides a comprehensive framework for measuring and monitoring various aspects of data quality, including completeness, consistency, and reliability metrics using statistical methods.

```python
class DataQualityMetrics:
    def __init__(self):
        self.metrics_history = {}
        self.threshold_violations = {}
        
    def compute_completeness_metrics(self, df):
        completeness = {
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'complete_rows': (1 - df.isnull().any(axis=1).sum() / len(df)) * 100
        }
        return completeness
    
    def compute_consistency_metrics(self, df, categorical_cols):
        consistency = {}
        
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            consistency[col] = {
                'unique_values': len(value_counts),
                'entropy': stats.entropy(value_counts.values),
                'mode_frequency': value_counts.iloc[0] / len(df)
            }
            
        return consistency
    
    def compute_statistical_metrics(self, df, numerical_cols):
        statistics = {}
        
        for col in numerical_cols:
            statistics[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'iqr': df[col].quantile(0.75) - df[col].quantile(0.25)
            }
            
        return statistics
    
    def detect_anomalies(self, df, numerical_cols, n_std=3):
        anomalies = {}
        
        for col in numerical_cols:
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - n_std * std
            upper_bound = mean + n_std * std
            
            anomalies[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'n_anomalies': len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
            }
            
        return anomalies
    
    def compute_correlation_stability(self, df, numerical_cols):
        corr_matrix = df[numerical_cols].corr()
        
        stability = {
            'mean_correlation': np.abs(corr_matrix.values).mean(),
            'correlation_std': np.abs(corr_matrix.values).std(),
            'high_correlations': len(np.where(np.abs(corr_matrix.values) > 0.8)[0]) // 2
        }
        
        return stability
    
    def generate_quality_report(self, df, categorical_cols, numerical_cols):
        report = {
            'completeness': self.compute_completeness_metrics(df),
            'consistency': self.compute_consistency_metrics(df, categorical_cols),
            'statistics': self.compute_statistical_metrics(df, numerical_cols),
            'anomalies': self.detect_anomalies(df, numerical_cols),
            'correlation_stability': self.compute_correlation_stability(df, numerical_cols),
            'timestamp': datetime.now().isoformat()
        }
        
        return report

# Example usage
# Create sample dataset with quality issues
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'numeric1': np.concatenate([np.random.normal(0, 1, n_samples-10), 
                              np.random.normal(10, 1, 10)]),  # with outliers
    'numeric2': np.random.normal(5, 2, n_samples),
    'category1': np.random.choice(['A', 'B', 'C', None], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    'category2': np.random.choice(['X', 'Y', 'Z'], n_samples)
})

metrics = DataQualityMetrics()
quality_report = metrics.generate_quality_report(
    data,
    categorical_cols=['category1', 'category2'],
    numerical_cols=['numeric1', 'numeric2']
)

print("Data Quality Report:")
print(json.dumps(quality_report, indent=2))
```

Slide 13: Multi-Format Data Integration

This implementation provides a framework for integrating data from multiple sources and formats, handling various data types and ensuring consistency across merged datasets.

```python
class DataIntegrator:
    def __init__(self):
        self.integration_logs = []
        self.schema_registry = {}
        
    def read_csv_data(self, file_path, **kwargs):
        try:
            data = pd.read_csv(file_path, **kwargs)
            self.log_operation('csv_read', file_path, data.shape)
            return data
        except Exception as e:
            self.log_operation('csv_read_error', file_path, str(e))
            raise
    
    def read_json_data(self, json_str, normalize=True):
        try:
            if normalize:
                data = pd.json_normalize(json.loads(json_str))
            else:
                data = pd.DataFrame(json.loads(json_str))
            self.log_operation('json_read', 'string_input', data.shape)
            return data
        except Exception as e:
            self.log_operation('json_read_error', 'string_input', str(e))
            raise
    
    def normalize_column_names(self, df):
        df = df.copy()
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        return df
    
    def harmonize_datatypes(self, df, schema=None):
        if schema is None:
            return df
            
        for column, dtype in schema.items():
            if column in df.columns:
                try:
                    df[column] = df[column].astype(dtype)
                except:
                    self.log_operation('type_conversion_error', column, dtype)
        
        return df
    
    def merge_datasets(self, dfs, merge_keys, merge_strategy='outer'):
        result = dfs[0]
        for df in dfs[1:]:
            try:
                result = pd.merge(
                    result, 
                    df,
                    on=merge_keys,
                    how=merge_strategy,
                    validate='1:1'
                )
                self.log_operation('merge_success', 'datasets', result.shape)
            except Exception as e:
                self.log_operation('merge_error', 'datasets', str(e))
                raise
                
        return result
    
    def validate_integration(self, df, rules):
        validations = {}
        
        for rule_name, rule_func in rules.items():
            try:
                validation_result = rule_func(df)
                validations[rule_name] = validation_result
            except Exception as e:
                validations[rule_name] = f"Validation error: {str(e)}"
                
        return validations
    
    def log_operation(self, operation_type, target, result):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation_type,
            'target': target,
            'result': result
        }
        self.integration_logs.append(log_entry)

# Example usage
# Create sample data sources
csv_data = pd.DataFrame({
    'ID': range(1000),
    'Feature A': np.random.normal(0, 1, 1000),
    'Category': np.random.choice(['X', 'Y', 'Z'], 1000)
})
csv_data.to_csv('sample.csv', index=False)

json_data = json.dumps([{
    'id': i,
    'value': np.random.random()
} for i in range(1000)])

# Define schema and validation rules
schema = {
    'id': 'int64',
    'feature_a': 'float64',
    'category': 'category',
    'value': 'float64'
}

validation_rules = {
    'no_missing_ids': lambda df: df['id'].isnull().sum() == 0,
    'unique_ids': lambda df: df['id'].is_unique,
    'value_range': lambda df: (df['value'] >= 0).all() and (df['value'] <= 1).all()
}

# Integrate data
integrator = DataIntegrator()

df1 = integrator.read_csv_data('sample.csv')
df1 = integrator.normalize_column_names(df1)

df2 = integrator.read_json_data(json_data)
df2 = integrator.normalize_column_names(df2)

# Harmonize and merge
df1 = integrator.harmonize_datatypes(df1, schema)
df2 = integrator.harmonize_datatypes(df2, schema)

merged_data = integrator.merge_datasets([df1, df2], merge_keys=['id'])

# Validate results
validation_results = integrator.validate_integration(merged_data, validation_rules)

print("Integration Results:")
print(f"Final shape: {merged_data.shape}")
print("\nValidation Results:")
print(json.dumps(validation_results, indent=2))
print("\nIntegration Logs:")
print(json.dumps(integrator.integration_logs, indent=2))
```

Slide 14: Additional Resources

*   Deep Learning with Big Data: Challenges and Approaches [https://arxiv.org/abs/2110.01064](https://arxiv.org/abs/2110.01064)
*   A Survey on Data Collection for Machine Learning [https://arxiv.org/abs/1811.03402](https://arxiv.org/abs/1811.03402)
*   Systematic Data Collection for Deep Learning Applications [https://research.google/pubs/data-collection-deep-learning](https://research.google/pubs/data-collection-deep-learning)
*   Best Practices for Data Quality in Deep Learning Systems [https://dl.acm.org/doi/10.1145/3447548.3467162](https://dl.acm.org/doi/10.1145/3447548.3467162)
*   Quality Assessment Methods for Deep Learning Training Data Search: "data quality assessment machine learning arxiv"

Note: These URLs are for reference. Please verify them independently as they may have changed or been updated.

