## Importance of Data Quality in Machine Learning
Slide 1: Data Quality Assessment

Data quality assessment is a crucial first step in machine learning projects. We'll create a comprehensive data quality analyzer that checks for missing values, duplicates, statistical outliers, and data distribution characteristics across numerical and categorical features.

```python
import pandas as pd
import numpy as np
from scipy import stats

class DataQualityAnalyzer:
    def __init__(self, df):
        self.df = df
        
    def check_missing_values(self):
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        return pd.DataFrame({
            'Missing Values': missing,
            'Percentage': missing_pct
        }).sort_values('Percentage', ascending=False)
    
    def check_duplicates(self):
        duplicates = self.df.duplicated().sum()
        return f"Found {duplicates} duplicate rows ({duplicates/len(self.df):.2%})"
    
    def detect_outliers(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        outliers = np.where(z_scores > threshold)[0]
        return f"Found {len(outliers)} outliers in {column}"

# Example usage
import numpy as np
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 100],
    'B': [1, 1, 3, 4, 5]
})
analyzer = DataQualityAnalyzer(df)
print(analyzer.check_missing_values())
print(analyzer.check_duplicates())
print(analyzer.detect_outliers('B'))
```

Slide 2: Data Cleaning Pipeline

A robust data cleaning pipeline is essential for maintaining data quality. This implementation showcases a modular approach to data cleaning, including handling missing values, removing duplicates, and standardizing formats using scikit-learn's Pipeline architecture.

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_strategy='mean', categorical_strategy='mode'):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_values = {}
        
    def fit(self, X, y=None):
        for column in X.columns:
            if pd.api.types.is_numeric_dtype(X[column]):
                if self.numeric_strategy == 'mean':
                    self.fill_values[column] = X[column].mean()
            else:
                if self.categorical_strategy == 'mode':
                    self.fill_values[column] = X[column].mode()[0]
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for column, value in self.fill_values.items():
            X_copy[column] = X_copy[column].fillna(value)
        return X_copy

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        numeric_cols = X_copy.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(X_copy[col]))
            X_copy[col] = np.where(z_scores > self.threshold,
                                 X_copy[col].mean(),
                                 X_copy[col])
        return X_copy

# Create and use pipeline
pipeline = Pipeline([
    ('missing_handler', MissingValueHandler()),
    ('outlier_handler', OutlierHandler())
])

# Example usage
df = pd.DataFrame({
    'numeric': [1, 2, np.nan, 4, 100],
    'categorical': ['A', 'B', None, 'B', 'C']
})
cleaned_df = pipeline.fit_transform(df)
print("Original DataFrame:\n", df)
print("\nCleaned DataFrame:\n", cleaned_df)
```

Slide 3: Feature Quality Score

A comprehensive feature quality scoring system helps identify the most reliable features for model training. This implementation calculates a quality score based on missing values, uniqueness, and statistical properties of each feature.

```python
import pandas as pd
import numpy as np
from scipy import stats

class FeatureQualityScorer:
    def __init__(self, df):
        self.df = df
        
    def calculate_missing_score(self, column):
        missing_ratio = self.df[column].isnull().mean()
        return 1 - missing_ratio
    
    def calculate_uniqueness_score(self, column):
        unique_ratio = len(self.df[column].unique()) / len(self.df)
        return min(unique_ratio, 1.0)
    
    def calculate_distribution_score(self, column):
        if pd.api.types.is_numeric_dtype(self.df[column]):
            try:
                _, p_value = stats.normaltest(self.df[column].dropna())
                return min(p_value * 10, 1.0)  # Scale p-value
            except:
                return 0.5
        return 1.0  # For non-numeric columns
    
    def get_feature_scores(self):
        scores = {}
        for column in self.df.columns:
            missing_score = self.calculate_missing_score(column)
            uniqueness_score = self.calculate_uniqueness_score(column)
            distribution_score = self.calculate_distribution_score(column)
            
            # Calculate weighted average
            final_score = (0.4 * missing_score + 
                         0.3 * uniqueness_score + 
                         0.3 * distribution_score)
            
            scores[column] = {
                'missing_score': missing_score,
                'uniqueness_score': uniqueness_score,
                'distribution_score': distribution_score,
                'final_score': final_score
            }
        
        return pd.DataFrame(scores).T

# Example usage
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5] * 20,
    'B': np.random.normal(0, 1, 100),
    'C': ['x', 'y', 'z', 'x', 'y'] * 20
})

scorer = FeatureQualityScorer(df)
feature_scores = scorer.get_feature_scores()
print("Feature Quality Scores:\n", feature_scores)
```

Slide 4: Data Distribution Analyzer

Understanding data distributions is crucial for feature engineering and model selection. This implementation provides comprehensive distribution analysis including skewness, kurtosis, and visualization capabilities for both numerical and categorical features.

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, Any

class DistributionAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        
    def analyze_numeric_distribution(self, column: str) -> Dict[str, Any]:
        data = self.df[column].dropna()
        analysis = {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'normality_test': stats.normaltest(data)[1]
        }
        
        # Calculate percentiles
        analysis.update({
            f'percentile_{p}': np.percentile(data, p)
            for p in [25, 50, 75, 95, 99]
        })
        
        return analysis
    
    def get_distribution_metrics(self) -> pd.DataFrame:
        metrics = {}
        for col in self.numeric_cols:
            try:
                metrics[col] = self.analyze_numeric_distribution(col)
            except Exception as e:
                print(f"Error analyzing {col}: {str(e)}")
                continue
        
        return pd.DataFrame(metrics).T

# Example usage
np.random.seed(42)
df = pd.DataFrame({
    'normal': np.random.normal(0, 1, 1000),
    'skewed': np.random.exponential(2, 1000),
    'uniform': np.random.uniform(0, 1, 1000)
})

analyzer = DistributionAnalyzer(df)
distribution_metrics = analyzer.get_distribution_metrics()
print("Distribution Metrics:\n", distribution_metrics)

# Example output visualization
for col in df.columns:
    plt.figure(figsize=(10, 4))
    plt.hist(df[col], bins=50, density=True, alpha=0.7)
    plt.title(f'Distribution of {col}')
    plt.show()
```

Slide 5: Advanced Data Balancing

Data imbalance can significantly impact model performance. This implementation provides sophisticated techniques for handling imbalanced datasets using both oversampling and undersampling strategies with cross-validation support.

```python
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from collections import Counter

class AdvancedDataBalancer:
    def __init__(self, 
                 method='hybrid',
                 sampling_strategy='auto',
                 random_state=42):
        self.method = method
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        
    def balance_dataset(self, X, y):
        if self.method == 'hybrid':
            # First apply SMOTE
            smote = SMOTE(sampling_strategy=self.sampling_strategy,
                         random_state=self.random_state)
            X_smote, y_smote = smote.fit_resample(X, y)
            
            # Then clean using Tomek Links
            tomek = TomekLinks(sampling_strategy='majority')
            X_balanced, y_balanced = tomek.fit_resample(X_smote, y_smote)
            
        elif self.method == 'adasyn':
            adasyn = ADASYN(sampling_strategy=self.sampling_strategy,
                           random_state=self.random_state)
            X_balanced, y_balanced = adasyn.fit_resample(X, y)
            
        return X_balanced, y_balanced
    
    def get_balance_metrics(self, y_original, y_balanced):
        original_dist = Counter(y_original)
        balanced_dist = Counter(y_balanced)
        
        metrics = {
            'original_distribution': original_dist,
            'balanced_distribution': balanced_dist,
            'original_ratio': min(original_dist.values()) / max(original_dist.values()),
            'balanced_ratio': min(balanced_dist.values()) / max(balanced_dist.values())
        }
        
        return metrics

# Example usage
from sklearn.datasets import make_classification

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                         random_state=42)

balancer = AdvancedDataBalancer(method='hybrid')
X_balanced, y_balanced = balancer.balance_dataset(X, y)

metrics = balancer.get_balance_metrics(y, y_balanced)
print("Balance Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")

# Visualize class distribution before and after
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
pd.Series(y).value_counts().plot(kind='bar', ax=ax1, title='Original Distribution')
pd.Series(y_balanced).value_counts().plot(kind='bar', ax=ax2, title='Balanced Distribution')
plt.tight_layout()
plt.show()
```

Slide 6: Feature Correlation Analysis

Advanced correlation analysis helps identify redundant features and complex relationships between variables. This implementation includes various correlation metrics and visualization techniques for both linear and non-linear relationships.

```python
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.cluster import mutual_info_score
import seaborn as sns
import matplotlib.pyplot as plt

class FeatureCorrelationAnalyzer:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        
    def get_correlation_matrix(self, method='pearson'):
        if method == 'pearson':
            return self.df[self.numeric_cols].corr()
        elif method == 'spearman':
            return self.df[self.numeric_cols].corr(method='spearman')
        elif method == 'mutual_info':
            mi_matrix = np.zeros((len(self.numeric_cols), len(self.numeric_cols)))
            for i, col1 in enumerate(self.numeric_cols):
                for j, col2 in enumerate(self.numeric_cols):
                    mi_matrix[i, j] = mutual_info_score(
                        self.df[col1], self.df[col2]
                    )
            return pd.DataFrame(mi_matrix, 
                              index=self.numeric_cols,
                              columns=self.numeric_cols)
    
    def get_highly_correlated_features(self, threshold=0.8):
        corr_matrix = self.get_correlation_matrix()
        high_corr = np.where(np.abs(corr_matrix) > threshold)
        high_corr_pairs = [(corr_matrix.index[x],
                           corr_matrix.columns[y],
                           corr_matrix.iloc[x, y])
                          for x, y in zip(*high_corr) if x != y]
        return pd.DataFrame(high_corr_pairs,
                          columns=['Feature 1', 'Feature 2', 'Correlation'])
    
    def plot_correlation_heatmap(self, method='pearson'):
        plt.figure(figsize=(12, 8))
        corr_matrix = self.get_correlation_matrix(method=method)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title(f'{method.capitalize()} Correlation Matrix')
        plt.show()

# Example usage
np.random.seed(42)
df = pd.DataFrame({
    'A': np.random.normal(0, 1, 1000),
    'B': np.random.normal(0, 1, 1000),
    'C': np.random.normal(0, 1, 1000)
})
df['D'] = df['A'] * 0.9 + np.random.normal(0, 0.1, 1000)

analyzer = FeatureCorrelationAnalyzer(df)
print("Highly correlated features:")
print(analyzer.get_highly_correlated_features())
analyzer.plot_correlation_heatmap()
```

Slide 7: Data Drift Detection

Data drift detection is crucial for maintaining model performance over time. This implementation provides methods to detect and quantify statistical drift in both feature distributions and target variables using multiple statistical tests.

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler

class DataDriftDetector:
    def __init__(self, reference_data: pd.DataFrame, current_data: pd.DataFrame):
        self.reference_data = reference_data
        self.current_data = current_data
        self.drift_metrics = {}
        
    def detect_numerical_drift(self, column: str) -> Dict[str, float]:
        ref_data = self.reference_data[column].dropna()
        curr_data = self.current_data[column].dropna()
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(ref_data, curr_data)
        
        # Population Stability Index (PSI)
        bins = np.histogram_bin_edges(np.concatenate([ref_data, curr_data]), bins='auto')
        ref_hist, _ = np.histogram(ref_data, bins=bins, density=True)
        curr_hist, _ = np.histogram(curr_data, bins=bins, density=True)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        ref_hist = ref_hist + epsilon
        curr_hist = curr_hist + epsilon
        
        psi = np.sum((curr_hist - ref_hist) * np.log(curr_hist / ref_hist))
        
        return {
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'psi': psi
        }
    
    def detect_drift_all_features(self) -> pd.DataFrame:
        drift_results = {}
        
        for column in self.reference_data.columns:
            if pd.api.types.is_numeric_dtype(self.reference_data[column]):
                drift_results[column] = self.detect_numerical_drift(column)
                
        return pd.DataFrame.from_dict(drift_results, orient='index')
    
    def get_drifted_features(self, 
                            ks_threshold: float = 0.05,
                            psi_threshold: float = 0.2) -> List[str]:
        drift_results = self.detect_drift_all_features()
        
        drifted_features = []
        for feature in drift_results.index:
            if (drift_results.loc[feature, 'ks_pvalue'] < ks_threshold or
                drift_results.loc[feature, 'psi'] > psi_threshold):
                drifted_features.append(feature)
                
        return drifted_features

# Example usage
np.random.seed(42)

# Generate reference data
reference_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.exponential(2, 1000),
    'feature3': np.random.uniform(0, 1, 1000)
})

# Generate current data with drift in feature1
current_data = pd.DataFrame({
    'feature1': np.random.normal(0.5, 1.2, 1000),  # Introduced drift
    'feature2': np.random.exponential(2, 1000),
    'feature3': np.random.uniform(0, 1, 1000)
})

# Detect drift
detector = DataDriftDetector(reference_data, current_data)
drift_results = detector.detect_drift_all_features()
print("Drift Detection Results:\n", drift_results)

drifted_features = detector.get_drifted_features()
print("\nDrifted Features:", drifted_features)

# Visualize distributions
import matplotlib.pyplot as plt

for feature in reference_data.columns:
    plt.figure(figsize=(10, 4))
    plt.hist(reference_data[feature], bins=50, alpha=0.5, label='Reference')
    plt.hist(current_data[feature], bins=50, alpha=0.5, label='Current')
    plt.title(f'Distribution Comparison: {feature}')
    plt.legend()
    plt.show()
```

Slide 8: Advanced Data Quality Metrics

This implementation provides sophisticated metrics for assessing data quality beyond basic statistics, including entropy-based measures, correlation analysis, and completeness scores across different data dimensions.

```python
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any

class AdvancedDataQualityMetrics:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.metrics = {}
        
    def calculate_entropy_score(self, column: str) -> float:
        if pd.api.types.is_numeric_dtype(self.df[column]):
            # For numerical columns, use binned entropy
            hist, _ = np.histogram(self.df[column].dropna(), bins='auto')
            prob = hist / hist.sum()
            return entropy(prob)
        else:
            # For categorical columns, use value counts entropy
            value_counts = self.df[column].value_counts()
            prob = value_counts / value_counts.sum()
            return entropy(prob)
    
    def calculate_completeness_score(self, column: str) -> float:
        return 1 - (self.df[column].isnull().sum() / len(self.df))
    
    def calculate_consistency_score(self, column: str) -> float:
        if pd.api.types.is_numeric_dtype(self.df[column]):
            # For numerical columns, use coefficient of variation
            mean = self.df[column].mean()
            std = self.df[column].std()
            return 1 - (std / mean if mean != 0 else 0)
        else:
            # For categorical columns, use mode frequency
            mode_freq = self.df[column].value_counts().iloc[0] / len(self.df)
            return mode_freq
    
    def calculate_all_metrics(self) -> Dict[str, Dict[str, float]]:
        for column in self.df.columns:
            self.metrics[column] = {
                'completeness': self.calculate_completeness_score(column),
                'entropy': self.calculate_entropy_score(column),
                'consistency': self.calculate_consistency_score(column)
            }
            
            # Calculate overall quality score
            self.metrics[column]['overall_score'] = np.mean([
                self.metrics[column]['completeness'],
                1 - (self.metrics[column]['entropy'] / np.log(2)),  # Normalize entropy
                self.metrics[column]['consistency']
            ])
        
        return self.metrics
    
    def get_quality_summary(self) -> pd.DataFrame:
        if not self.metrics:
            self.calculate_all_metrics()
        return pd.DataFrame.from_dict(self.metrics, orient='index')

# Example usage
np.random.seed(42)

# Create sample dataset with various quality issues
df = pd.DataFrame({
    'complete_numeric': np.random.normal(0, 1, 1000),
    'missing_numeric': np.random.normal(0, 1, 1000).astype(float),
    'categorical': np.random.choice(['A', 'B', 'C'], 1000),
    'imbalanced_categorical': np.random.choice(['X', 'Y'], 1000, p=[0.9, 0.1])
})

# Add missing values
df.loc[np.random.choice(1000, 100), 'missing_numeric'] = np.nan

# Calculate quality metrics
quality_analyzer = AdvancedDataQualityMetrics(df)
quality_summary = quality_analyzer.get_quality_summary()
print("Data Quality Summary:\n", quality_summary)

# Visualize quality scores
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
quality_summary['overall_score'].plot(kind='bar')
plt.title('Overall Quality Scores by Feature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Slide 9: Time Series Data Quality

Time series data requires specialized quality checks including temporal consistency, seasonality detection, and trend analysis. This implementation provides comprehensive time series data quality assessment tools.

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Dict, Any
import warnings

class TimeSeriesQualityAnalyzer:
    def __init__(self, df: pd.DataFrame, timestamp_col: str, value_col: str):
        self.df = df.sort_values(timestamp_col).copy()
        self.timestamp_col = timestamp_col
        self.value_col = value_col
        
    def check_temporal_consistency(self) -> Dict[str, Any]:
        time_diffs = self.df[self.timestamp_col].diff()
        
        consistency_metrics = {
            'missing_timestamps': time_diffs.isnull().sum(),
            'irregular_intervals': (time_diffs != time_diffs.mode()[0]).sum(),
            'avg_interval': time_diffs.mean(),
            'max_interval': time_diffs.max(),
            'min_interval': time_diffs.min()
        }
        
        return consistency_metrics
    
    def detect_anomalies(self, window: int = 10, sigma: float = 3) -> pd.Series:
        rolling_mean = self.df[self.value_col].rolling(window=window).mean()
        rolling_std = self.df[self.value_col].rolling(window=window).std()
        
        upper_bound = rolling_mean + (sigma * rolling_std)
        lower_bound = rolling_mean - (sigma * rolling_std)
        
        anomalies = (self.df[self.value_col] > upper_bound) | \
                   (self.df[self.value_col] < lower_bound)
                   
        return anomalies
    
    def analyze_seasonality(self) -> Dict[str, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decomposition = seasonal_decompose(
                self.df[self.value_col],
                period=self._estimate_period(),
                model='additive'
            )
        
        seasonality_metrics = {
            'trend_strength': np.std(decomposition.trend.dropna()),
            'seasonal_strength': np.std(decomposition.seasonal.dropna()),
            'residual_strength': np.std(decomposition.resid.dropna())
        }
        
        return seasonality_metrics
    
    def _estimate_period(self) -> int:
        # Simple period estimation using autocorrelation
        n = len(self.df)
        if n < 4:  # Minimum required for meaningful analysis
            return 1
            
        acf = np.correlate(self.df[self.value_col] - 
                          self.df[self.value_col].mean(),
                          self.df[self.value_col] - 
                          self.df[self.value_col].mean(),
                          mode='full')[n-1:]
                          
        peaks = np.where((acf[1:] > acf[:-1]) & 
                        (acf[1:] > acf[2:]))[0] + 1
                        
        return peaks[0] if len(peaks) > 0 else 1
    
    def get_quality_report(self) -> Dict[str, Any]:
        report = {
            'temporal_consistency': self.check_temporal_consistency(),
            'seasonality_analysis': self.analyze_seasonality(),
            'anomaly_count': self.detect_anomalies().sum()
        }
        
        return report

# Example usage
np.random.seed(42)

# Create sample time series data with quality issues
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365)  # Yearly seasonality
values += np.random.normal(0, 0.1, len(dates))  # Add noise

# Add some anomalies
values[10:15] += 2
values[100:105] -= 2

df = pd.DataFrame({
    'timestamp': dates,
    'value': values
})

# Analyze time series quality
analyzer = TimeSeriesQualityAnalyzer(df, 'timestamp', 'value')
quality_report = analyzer.get_quality_report()

print("Time Series Quality Report:")
for metric_group, metrics in quality_report.items():
    print(f"\n{metric_group}:")
    if isinstance(metrics, dict):
        for name, value in metrics.items():
            print(f"  {name}: {value}")
    else:
        print(f"  {metrics}")

# Visualize anomalies
anomalies = analyzer.detect_anomalies()
plt.figure(figsize=(15, 5))
plt.plot(df['timestamp'], df['value'], label='Original')
plt.scatter(df[anomalies]['timestamp'], 
           df[anomalies]['value'], 
           color='red', 
           label='Anomalies')
plt.title('Time Series with Detected Anomalies')
plt.legend()
plt.show()
```

Slide 10: Data Quality Monitoring Dashboard

Implementation of a comprehensive data quality monitoring system that tracks quality metrics over time and generates automated alerts when quality degrades below specified thresholds.

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

class DataQualityDashboard:
    def __init__(self, alert_thresholds: Dict[str, float] = None):
        self.alert_thresholds = alert_thresholds or {
            'completeness': 0.95,
            'consistency': 0.90,
            'freshness': 24  # hours
        }
        self.metrics_history = []
        
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        metrics = {
            'timestamp': datetime.now(),
            'row_count': len(df),
            'completeness': self._calculate_completeness(df),
            'consistency': self._calculate_consistency(df),
            'freshness': self._calculate_freshness(df),
            'column_stats': self._calculate_column_stats(df)
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        return 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
    
    def _calculate_consistency(self, df: pd.DataFrame) -> float:
        consistency_scores = []
        
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                # Check for values within expected range
                z_scores = np.abs((df[column] - df[column].mean()) / 
                                df[column].std())
                consistency_scores.append(
                    (z_scores < 3).mean()  # Within 3 standard deviations
                )
            else:
                # Check for consistent categories
                value_counts = df[column].value_counts(normalize=True)
                consistency_scores.append(
                    (value_counts > 0.01).sum() / len(value_counts)
                )
                
        return np.mean(consistency_scores)
    
    def _calculate_freshness(self, df: pd.DataFrame) -> float:
        timestamp_cols = [col for col in df.columns 
                         if pd.api.types.is_datetime64_any_dtype(df[col])]
        
        if not timestamp_cols:
            return 0
            
        latest_timestamp = df[timestamp_cols[0]].max()
        hours_since_update = (datetime.now() - 
                            pd.to_datetime(latest_timestamp)).total_seconds() / 3600
                            
        return hours_since_update
    
    def _calculate_column_stats(self, df: pd.DataFrame) -> Dict[str, Dict]:
        stats = {}
        
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                stats[column] = {
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'min': df[column].min(),
                    'max': df[column].max()
                }
            else:
                stats[column] = {
                    'unique_values': df[column].nunique(),
                    'most_common': df[column].mode()[0],
                    'most_common_freq': df[column].value_counts().iloc[0] / len(df)
                }
                
        return stats
    
    def generate_alerts(self, metrics: Dict[str, Any]) -> List[str]:
        alerts = []
        
        if metrics['completeness'] < self.alert_thresholds['completeness']:
            alerts.append(
                f"Data completeness ({metrics['completeness']:.2%}) below threshold "
                f"({self.alert_thresholds['completeness']:.2%})"
            )
            
        if metrics['consistency'] < self.alert_thresholds['consistency']:
            alerts.append(
                f"Data consistency ({metrics['consistency']:.2%}) below threshold "
                f"({self.alert_thresholds['consistency']:.2%})"
            )
            
        if metrics['freshness'] > self.alert_thresholds['freshness']:
            alerts.append(
                f"Data freshness ({metrics['freshness']:.1f} hours) exceeds threshold "
                f"({self.alert_thresholds['freshness']} hours)"
            )
            
        return alerts

# Example usage
np.random.seed(42)

# Generate sample data with quality issues
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
df = pd.DataFrame({
    'timestamp': dates,
    'numeric_col': np.random.normal(0, 1, len(dates)),
    'categorical_col': np.random.choice(['A', 'B', 'C'], len(dates)),
    'value': np.random.uniform(0, 100, len(dates))
})

# Add some quality issues
df.loc[np.random.choice(len(df), 50), 'numeric_col'] = np.nan
df.loc[np.random.choice(len(df), 20), 'categorical_col'] = 'INVALID'

# Create dashboard and monitor data quality
dashboard = DataQualityDashboard()
metrics = dashboard.calculate_metrics(df)
alerts = dashboard.generate_alerts(metrics)

print("Data Quality Metrics:")
print(json.dumps(metrics, default=str, indent=2))

print("\nQuality Alerts:")
for alert in alerts:
    print(f"- {alert}")
```

Slide 11: Feature Engineering Quality Assessment

This implementation provides methods to evaluate the quality and effectiveness of engineered features, including information gain, correlation analysis, and feature importance metrics.

```python
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Tuple
from scipy.stats import spearmanr

class FeatureQualityAssessor:
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target_col = target_col
        self.feature_cols = [col for col in df.columns if col != target_col]
        
    def calculate_feature_importance(self) -> Dict[str, float]:
        X = self.df[self.feature_cols]
        y = self.df[self.target_col]
        
        # Handle non-numeric features
        X = pd.get_dummies(X)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importance_dict = {}
        for feature, importance in zip(X.columns, rf.feature_importances_):
            importance_dict[feature] = importance
            
        return dict(sorted(importance_dict.items(), 
                         key=lambda x: x[1], 
                         reverse=True))
    
    def calculate_information_gain(self) -> Dict[str, float]:
        X = self.df[self.feature_cols]
        y = self.df[self.target_col]
        
        info_gains = {}
        for col in self.feature_cols:
            if pd.api.types.is_numeric_dtype(X[col]):
                mi_score = mutual_info_regression(
                    X[[col]], y, random_state=42)[0]
                info_gains[col] = mi_score
            else:
                # Handle categorical features
                dummies = pd.get_dummies(X[col])
                mi_scores = mutual_info_regression(
                    dummies, y, random_state=42)
                info_gains[col] = np.mean(mi_scores)
                
        return dict(sorted(info_gains.items(), 
                         key=lambda x: x[1], 
                         reverse=True))
    
    def detect_multicollinearity(self, 
                               threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        numeric_cols = self.df[self.feature_cols].select_dtypes(
            include=[np.number]).columns
        
        correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                corr, _ = spearmanr(self.df[col1], self.df[col2])
                
                if abs(corr) > threshold:
                    correlations.append((col1, col2, corr))
                    
        return sorted(correlations, key=lambda x: abs(x[2]), reverse=True)
    
    def get_quality_report(self) -> Dict[str, Any]:
        report = {
            'feature_importance': self.calculate_feature_importance(),
            'information_gain': self.calculate_information_gain(),
            'multicollinearity': self.detect_multicollinearity()
        }
        
        return report

# Example usage
np.random.seed(42)

# Generate sample dataset with engineered features
n_samples = 1000
df = pd.DataFrame({
    'original_feature': np.random.normal(0, 1, n_samples),
    'target': np.random.normal(0, 1, n_samples)
})

# Add engineered features
df['squared_feature'] = df['original_feature'] ** 2
df['noisy_feature'] = df['original_feature'] + np.random.normal(0, 0.5, n_samples)
df['random_feature'] = np.random.normal(0, 1, n_samples)
df['categorical_feature'] = pd.qcut(df['original_feature'], q=5, labels=['A', 'B', 'C', 'D', 'E'])

# Update target with relationships
df['target'] = (df['original_feature'] * 0.5 + 
                df['squared_feature'] * 0.3 + 
                np.random.normal(0, 0.1, n_samples))

# Assess feature quality
assessor = FeatureQualityAssessor(df, 'target')
quality_report = assessor.get_quality_report()

# Display results
print("Feature Quality Report:\n")
print("Feature Importance:")
for feature, importance in quality_report['feature_importance'].items():
    print(f"{feature}: {importance:.4f}")

print("\nInformation Gain:")
for feature, gain in quality_report['information_gain'].items():
    print(f"{feature}: {gain:.4f}")

print("\nMulticollinearity:")
for col1, col2, corr in quality_report['multicollinearity']:
    print(f"{col1} - {col2}: {corr:.4f}")

# Visualize feature importance
plt.figure(figsize=(12, 6))
importance_df = pd.DataFrame.from_dict(
    quality_report['feature_importance'], 
    orient='index', 
    columns=['importance']
)
importance_df.plot(kind='bar')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Slide 12: Data Schema Validation

A robust implementation for validating data schema consistency, including type checking, value range validation, and custom constraint enforcement.

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional
from dataclasses import dataclass
import re

@dataclass
class ColumnSchema:
    name: str
    dtype: str
    nullable: bool = True
    unique: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    valid_values: Optional[List[Any]] = None
    regex_pattern: Optional[str] = None
    custom_validator: Optional[callable] = None

class SchemaValidator:
    def __init__(self, schema: Dict[str, ColumnSchema]):
        self.schema = schema
        
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        violations = {}
        
        # Check for missing columns
        missing_cols = set(self.schema.keys()) - set(df.columns)
        if missing_cols:
            violations['missing_columns'] = list(missing_cols)
        
        # Validate each column
        for col_name, col_schema in self.schema.items():
            if col_name not in df.columns:
                continue
                
            col_violations = self._validate_column(df[col_name], col_schema)
            if col_violations:
                violations[col_name] = col_violations
                
        return violations
    
    def _validate_column(self, 
                        series: pd.Series, 
                        schema: ColumnSchema) -> List[str]:
        violations = []
        
        # Check dtype
        if not pd.api.types.is_dtype_equal(series.dtype, schema.dtype):
            violations.append(
                f"Expected dtype {schema.dtype}, got {series.dtype}"
            )
        
        # Check nullability
        if not schema.nullable and series.isnull().any():
            violations.append("Column contains null values")
        
        # Check uniqueness
        if schema.unique and series.duplicated().any():
            violations.append("Column contains duplicate values")
        
        # Check value range for numeric columns
        if pd.api.types.is_numeric_dtype(series):
            if schema.min_value is not None and series.min() < schema.min_value:
                violations.append(
                    f"Values below minimum: {schema.min_value}"
                )
            if schema.max_value is not None and series.max() > schema.max_value:
                violations.append(
                    f"Values above maximum: {schema.max_value}"
                )
        
        # Check valid values
        if schema.valid_values is not None:
            invalid_values = set(series.dropna().unique()) - set(schema.valid_values)
            if invalid_values:
                violations.append(
                    f"Invalid values found: {invalid_values}"
                )
        
        # Check regex pattern
        if schema.regex_pattern is not None:
            pattern = re.compile(schema.regex_pattern)
            invalid_patterns = series.dropna()[
                ~series.dropna().astype(str).str.match(pattern)
            ]
            if len(invalid_patterns) > 0:
                violations.append(
                    f"Values not matching pattern: {invalid_patterns.tolist()}"
                )
        
        # Apply custom validator
        if schema.custom_validator is not None:
            try:
                if not schema.custom_validator(series):
                    violations.append("Failed custom validation")
            except Exception as e:
                violations.append(f"Custom validation error: {str(e)}")
        
        return violations

# Example usage
def custom_date_validator(series):
    return pd.to_datetime(series, errors='coerce').notnull().all()

# Define schema
schema = {
    'id': ColumnSchema(
        name='id',
        dtype='int64',
        nullable=False,
        unique=True
    ),
    'name': ColumnSchema(
        name='name',
        dtype='object',
        regex_pattern=r'^[A-Za-z\s]+$'
    ),
    'age': ColumnSchema(
        name='age',
        dtype='int64',
        min_value=0,
        max_value=120
    ),
    'category': ColumnSchema(
        name='category',
        dtype='object',
        valid_values=['A', 'B', 'C']
    ),
    'date': ColumnSchema(
        name='date',
        dtype='object',
        custom_validator=custom_date_validator
    )
}

# Create sample dataframe with violations
df = pd.DataFrame({
    'id': [1, 2, 2, 4],  # Duplicate ID
    'name': ['John Doe', 'Jane123', 'Bob', 'Alice'],  # Invalid name format
    'age': [25, 150, 30, -5],  # Age out of range
    'category': ['A', 'B', 'D', 'C'],  # Invalid category
    'date': ['2023-01-01', 'invalid_date', '2023-02-01', '2023-03-01']  # Invalid date
})

# Validate schema
validator = SchemaValidator(schema)
violations = validator.validate_dataframe(df)

print("Schema Violations:")
for column, column_violations in violations.items():
    print(f"\n{column}:")
    for violation in column_violations:
        print(f"  - {violation}")
```

Slide 13: Data Version Control and Lineage

Implementation of a system to track data versions, transformations, and lineage throughout the data processing pipeline, enabling reproducibility and quality control.

```python
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class DataTransformation:
    name: str
    description: str
    parameters: Dict[str, Any]
    timestamp: datetime
    input_hash: str
    output_hash: str

class DataVersionControl:
    def __init__(self):
        self.versions = {}
        self.transformations = []
        self.lineage_graph = {}
        
    def compute_hash(self, df: pd.DataFrame) -> str:
        """Compute a deterministic hash of the dataframe content"""
        serialized = pd.util.hash_pandas_object(df).values
        return hashlib.sha256(serialized.tobytes()).hexdigest()
    
    def save_version(self, 
                    df: pd.DataFrame, 
                    version_name: str, 
                    metadata: Dict[str, Any] = None) -> str:
        """Save a version of the dataframe with metadata"""
        version_hash = self.compute_hash(df)
        
        self.versions[version_hash] = {
            'version_name': version_name,
            'timestamp': datetime.now(),
            'shape': df.shape,
            'columns': list(df.columns),
            'metadata': metadata or {},
            'stats': self._compute_stats(df)
        }
        
        return version_hash
    
    def _compute_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute basic statistics for the dataframe"""
        stats = {
            'null_counts': df.isnull().sum().to_dict(),
            'numeric_stats': {}
        }
        
        for col in df.select_dtypes(include=[np.number]).columns:
            stats['numeric_stats'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
            
        return stats
    
    def record_transformation(self,
                            name: str,
                            description: str,
                            input_df: pd.DataFrame,
                            output_df: pd.DataFrame,
                            parameters: Dict[str, Any] = None) -> None:
        """Record a transformation between two dataframe versions"""
        input_hash = self.compute_hash(input_df)
        output_hash = self.compute_hash(output_df)
        
        transformation = DataTransformation(
            name=name,
            description=description,
            parameters=parameters or {},
            timestamp=datetime.now(),
            input_hash=input_hash,
            output_hash=output_hash
        )
        
        self.transformations.append(transformation)
        
        # Update lineage graph
        if input_hash not in self.lineage_graph:
            self.lineage_graph[input_hash] = []
        self.lineage_graph[input_hash].append(output_hash)
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """Get the complete version history"""
        history = []
        for version_hash, version_info in self.versions.items():
            transformations = [
                asdict(t) for t in self.transformations
                if t.input_hash == version_hash or t.output_hash == version_hash
            ]
            
            history.append({
                'hash': version_hash,
                'info': version_info,
                'transformations': transformations
            })
            
        return sorted(history, key=lambda x: x['info']['timestamp'])
    
    def get_lineage(self, version_hash: str) -> Dict[str, Any]:
        """Get the complete lineage for a specific version"""
        lineage = {
            'upstream': self._get_upstream_lineage(version_hash),
            'downstream': self._get_downstream_lineage(version_hash)
        }
        return lineage
    
    def _get_upstream_lineage(self, version_hash: str) -> List[str]:
        """Get all ancestor versions"""
        upstream = []
        for input_hash, output_hashes in self.lineage_graph.items():
            if version_hash in output_hashes:
                upstream.append(input_hash)
                upstream.extend(self._get_upstream_lineage(input_hash))
        return list(set(upstream))
    
    def _get_downstream_lineage(self, version_hash: str) -> List[str]:
        """Get all descendant versions"""
        downstream = []
        if version_hash in self.lineage_graph:
            downstream.extend(self.lineage_graph[version_hash])
            for output_hash in self.lineage_graph[version_hash]:
                downstream.extend(self._get_downstream_lineage(output_hash))
        return list(set(downstream))

# Example usage
np.random.seed(42)

# Create initial dataset
df_original = pd.DataFrame({
    'id': range(1000),
    'value': np.random.normal(0, 1, 1000)
})

# Initialize version control
dvc = DataVersionControl()

# Save original version
original_hash = dvc.save_version(
    df_original, 
    'original',
    metadata={'source': 'synthetic_data'}
)

# Apply transformation 1: Add noise
df_noisy = df_original.copy()
df_noisy['value'] += np.random.normal(0, 0.1, 1000)

dvc.record_transformation(
    name='add_noise',
    description='Added Gaussian noise to values',
    input_df=df_original,
    output_df=df_noisy,
    parameters={'noise_std': 0.1}
)

noisy_hash = dvc.save_version(
    df_noisy,
    'noisy',
    metadata={'transformation': 'added_noise'}
)

# Apply transformation 2: Feature engineering
df_featured = df_noisy.copy()
df_featured['value_squared'] = df_featured['value'] ** 2

dvc.record_transformation(
    name='add_features',
    description='Added squared value feature',
    input_df=df_noisy,
    output_df=df_featured,
    parameters={'features': ['value_squared']}
)

featured_hash = dvc.save_version(
    df_featured,
    'featured',
    metadata={'transformation': 'feature_engineering'}
)

# Print version history
print("Version History:")
for version in dvc.get_version_history():
    print(f"\nVersion: {version['info']['version_name']}")
    print(f"Hash: {version['hash']}")
    print(f"Timestamp: {version['info']['timestamp']}")
    print(f"Shape: {version['info']['shape']}")
    print(f"Metadata: {version['info']['metadata']}")
    
# Print lineage for featured version
print("\nLineage for featured version:")
lineage = dvc.get_lineage(featured_hash)
print(f"Upstream versions: {lineage['upstream']}")
print(f"Downstream versions: {lineage['downstream']}")
```

Slide 14: Additional Resources

*   Data Quality in Machine Learning Systems: [https://arxiv.org/abs/2108.11497](https://arxiv.org/abs/2108.11497)
*   A Survey on Data Quality for Machine Learning: [https://arxiv.org/abs/2106.05528](https://arxiv.org/abs/2106.05528)
*   Automating Large-Scale Data Quality Verification: [https://arxiv.org/abs/1904.00904](https://arxiv.org/abs/1904.00904)
*   Search "data quality machine learning" on Google Scholar for more research papers
*   MLOps and Data Quality guides on Google Cloud documentation
*   Data Version Control (DVC) documentation at [https://dvc.org/doc](https://dvc.org/doc)
*   Great Expectations documentation at [https://docs.greatexpectations.io/](https://docs.greatexpectations.io/)

