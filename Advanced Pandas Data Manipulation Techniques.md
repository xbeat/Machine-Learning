## Advanced Pandas Data Manipulation Techniques
Slide 1: Advanced DataFrame Creation Techniques

Exploring sophisticated methods for DataFrame construction in Pandas, focusing on complex data structures and memory-efficient implementations. We'll examine various approaches to create DataFrames from different data sources while optimizing performance.

```python
import pandas as pd
import numpy as np

# Create DataFrame from multiple data types
data = {
    'numeric': np.random.randn(1000),
    'categorical': np.random.choice(['A', 'B', 'C'], 1000),
    'datetime': pd.date_range('2024-01-01', periods=1000),
    'sparse': pd.arrays.SparseArray(np.random.randn(1000), fill_value=0)
}

df = pd.DataFrame(data)
print(f"Memory usage: {df.memory_usage().sum() / 1024:.2f} KB")
print(df.head())
```

Slide 2: Memory-Optimized DataFrame Operations

Understanding memory optimization techniques when working with large datasets in Pandas. This includes proper dtype selection, chunking operations, and utilizing specialized data structures for improved performance.

```python
# Memory optimization techniques
def optimize_dataframe(df):
    # Optimize numeric columns
    numerics = ['int16', 'int32', 'int64', 'float64']
    for col in df.select_dtypes(include=numerics).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if str(df[col].dtype).startswith('int'):
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
                
        elif str(df[col].dtype).startswith('float'):
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Optimize categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
            df[col] = df[col].astype('category')
    
    return df

# Example usage
df_optimized = optimize_dataframe(df)
print(f"Original memory: {df.memory_usage().sum() / 1024:.2f} KB")
print(f"Optimized memory: {df_optimized.memory_usage().sum() / 1024:.2f} KB")
```

Slide 3: Advanced Data Aggregation Patterns

Implementing complex aggregation operations using Pandas' powerful groupby functionality combined with custom aggregation functions and multiple transformation methods simultaneously.

```python
import pandas as pd
import numpy as np

# Create sample financial data
np.random.seed(42)
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=1000),
    'category': np.random.choice(['Tech', 'Finance', 'Healthcare'], 1000),
    'value': np.random.normal(1000, 100, 1000),
    'volume': np.random.randint(100, 10000, 1000)
})

# Complex aggregation with multiple functions
agg_funcs = {
    'value': [
        ('mean', 'mean'),
        ('volatility', lambda x: x.std()),
        ('risk_adjusted', lambda x: x.mean() / x.std()),
    ],
    'volume': [
        ('total', 'sum'),
        ('daily_avg', 'mean'),
        ('peak', 'max')
    ]
}

result = df.groupby('category').agg(agg_funcs)
print(result)
```

Slide 4: Time Series Analysis and Resampling

Advanced time series manipulation in Pandas, focusing on custom resampling strategies, rolling windows with complex calculations, and handling missing data in time-based operations.

```python
# Create time series data
ts_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='5min'),
    'value': np.random.normal(100, 10, 1000)
}).set_index('timestamp')

# Complex time series operations
def custom_resampler(x):
    return pd.Series({
        'mean': x.mean(),
        'std': x.std(),
        'skew': x.skew(),
        'range': x.max() - x.min(),
        'zscore_count': len(x[np.abs((x - x.mean()) / x.std()) > 2])
    })

# Apply multiple time-based transformations
result = pd.DataFrame({
    'original': ts_data['value'],
    'hourly_mean': ts_data['value'].resample('1H').mean(),
    'rolling_std': ts_data['value'].rolling(window='2H', min_periods=1).std(),
    'ewm': ts_data['value'].ewm(span=12).mean()
}).fillna(method='ffill')

# Custom resampling
custom_stats = ts_data.resample('1H')['value'].apply(custom_resampler)
print(custom_stats.head())
```

Slide 5: Advanced Data Cleaning and Preprocessing

Implementing sophisticated data cleaning techniques using Pandas, including handling missing values with complex imputation strategies, outlier detection, and data normalization methods.

```python
# Advanced data cleaning implementation
def advanced_clean_dataframe(df):
    # Create copy to avoid modifying original
    df_clean = df.copy()
    
    # Complex missing value imputation
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        # Calculate zscore for outlier detection
        zscore = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
        # Mark outliers as NaN
        df_clean.loc[zscore > 3, col] = np.nan
        
        # Impute missing values using interpolation with limits
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].interpolate(
                method='akima',
                limit_direction='both',
                limit=5
            )
    
    # Handle categorical missing values
    categorical_columns = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        # Create new category for rare values
        value_counts = df_clean[col].value_counts()
        rare_categories = value_counts[value_counts < len(df_clean) * 0.01].index
        df_clean.loc[df_clean[col].isin(rare_categories), col] = 'Other'
        
        # Fill remaining NaN with mode
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    return df_clean

# Example usage
clean_df = advanced_clean_dataframe(df)
print("Missing values before:", df.isnull().sum().sum())
print("Missing values after:", clean_df.isnull().sum().sum())
```

Slide 6: Complex DataFrame Transformations

Deep dive into advanced DataFrame transformations using custom functions, vectorized operations, and window functions. This approach demonstrates high-performance data manipulation techniques for large-scale data processing.

```python
# Complex transformation example
def apply_complex_transforms(df):
    # Create rolling window calculations
    def custom_momentum(series, window=20):
        return (series / series.shift(window) - 1) * 100
    
    # Apply multiple transformations
    transformed = pd.DataFrame({
        'original': df['value'],
        'normalized': (df['value'] - df['value'].mean()) / df['value'].std(),
        'momentum': custom_momentum(df['value']),
        'log_return': np.log(df['value'] / df['value'].shift(1)),
        'volatility': df['value'].rolling(window=20).std() * np.sqrt(252),
        'zscore': (df['value'] - df['value'].rolling(window=20).mean()) / \
                  df['value'].rolling(window=20).std()
    })
    
    # Add percentile ranks
    transformed['percentile_rank'] = transformed['original'].rank(pct=True)
    
    return transformed

# Example usage with sample data
np.random.seed(42)
df = pd.DataFrame({
    'value': np.random.normal(100, 10, 1000) * \
             np.exp(np.linspace(0, 0.1, 1000))  # Adding trend
})

result = apply_complex_transforms(df)
print(result.head())
```

Slide 7: Advanced Pivot and Reshape Operations

Exploring sophisticated reshape operations in Pandas, including complex pivot tables with multiple aggregations, cross-tabulations, and dynamic column generation based on data patterns.

```python
# Create sample data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=1000)
df = pd.DataFrame({
    'date': dates,
    'product': np.random.choice(['A', 'B', 'C', 'D'], 1000),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
    'sales': np.random.normal(1000, 100, 1000),
    'quantity': np.random.poisson(5, 1000)
})

# Complex pivot operations
def create_advanced_pivot(df):
    # Custom aggregation function
    def profit_margin(x):
        return (x.max() - x.min()) / x.mean() * 100
    
    # Create pivot with multiple levels and custom aggregations
    pivot = pd.pivot_table(
        df,
        values=['sales', 'quantity'],
        index=['region'],
        columns=['product'],
        aggfunc={
            'sales': [
                ('total', 'sum'),
                ('average', 'mean'),
                ('margin', profit_margin)
            ],
            'quantity': [
                ('total', 'sum'),
                ('efficiency', lambda x: x.sum() / x.count())
            ]
        },
        margins=True
    )
    
    # Flatten column names
    pivot.columns = [f"{col[0]}_{col[1]}_{col[2]}" \
                    for col in pivot.columns]
    
    return pivot

result_pivot = create_advanced_pivot(df)
print(result_pivot.head())
```

Slide 8: Real-world Example - Financial Data Analysis

Implementing a comprehensive financial data analysis system using Pandas, including calculation of technical indicators, risk metrics, and portfolio analytics with real-world market data simulation.

```python
import pandas as pd
import numpy as np
from scipy import stats

class FinancialAnalyzer:
    def __init__(self, prices_df):
        self.prices = prices_df
        
    def calculate_metrics(self):
        # Calculate returns
        self.returns = self.prices.pct_change()
        
        # Risk metrics
        risk_metrics = pd.DataFrame({
            'volatility': self.returns.rolling(window=21).std() * np.sqrt(252),
            'var_95': self.returns.rolling(window=100).quantile(0.05),
            'cvar_95': self.returns.rolling(window=100).apply(
                lambda x: x[x <= np.percentile(x, 5)].mean()
            ),
            'skewness': self.returns.rolling(window=63).apply(stats.skew),
            'kurtosis': self.returns.rolling(window=63).apply(stats.kurtosis)
        })
        
        # Technical indicators
        technical = pd.DataFrame({
            'sma_20': self.prices.rolling(window=20).mean(),
            'ema_20': self.prices.ewm(span=20).mean(),
            'upper_bb': self.prices.rolling(window=20).mean() + \
                       (self.prices.rolling(window=20).std() * 2),
            'lower_bb': self.prices.rolling(window=20).mean() - \
                       (self.prices.rolling(window=20).std() * 2),
            'rsi': self._calculate_rsi(14)
        })
        
        return pd.concat([risk_metrics, technical], axis=1)
    
    def _calculate_rsi(self, periods):
        delta = self.prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

# Example usage
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=252)  # One trading year
prices = pd.Series(
    np.random.normal(0, 0.01, 252).cumsum() + 100,
    index=dates
)

analyzer = FinancialAnalyzer(prices)
results = analyzer.calculate_metrics()
print(results.tail())
```

Slide 9: Real-time Data Processing Pipeline

Creating a sophisticated data processing pipeline using Pandas for handling streaming data, implementing sliding window calculations, and managing real-time updates with efficient memory usage.

```python
class DataPipeline:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.buffer = pd.DataFrame()
        self.processed_count = 0
    
    def process_batch(self, new_data):
        # Append new data
        self.buffer = pd.concat([self.buffer, new_data]).tail(self.window_size)
        
        # Calculate streaming metrics
        metrics = pd.DataFrame({
            'exponential_mean': self.buffer['value'].ewm(
                span=20, adjust=False).mean(),
            'rolling_zscore': (
                self.buffer['value'] - \
                self.buffer['value'].rolling(window=20).mean()
            ) / self.buffer['value'].rolling(window=20).std(),
            'momentum': self.buffer['value'].pct_change(5).rolling(
                window=10).mean(),
            'volatility': self.buffer['value'].rolling(
                window=20).std() * np.sqrt(252)
        })
        
        # Add event detection
        metrics['anomaly'] = np.abs(metrics['rolling_zscore']) > 2.5
        
        self.processed_count += len(new_data)
        return metrics.tail(len(new_data))

# Simulate streaming data
def generate_streaming_data(n_batches, batch_size=10):
    for _ in range(n_batches):
        yield pd.DataFrame({
            'timestamp': pd.date_range(
                start=pd.Timestamp.now(), 
                periods=batch_size, 
                freq='S'
            ),
            'value': np.random.normal(100, 10, batch_size)
        }).set_index('timestamp')

# Example usage
pipeline = DataPipeline()
for batch in generate_streaming_data(5):
    result = pipeline.process_batch(batch)
    print(f"\nProcessed batch results:")
    print(result)
```

Slide 10: Advanced Data Validation and Quality Control

Implementing comprehensive data validation and quality control mechanisms using Pandas, including statistical tests, automated anomaly detection, and data integrity checks.

```python
class DataValidator:
    def __init__(self, df):
        self.df = df
        self.validation_results = {}
        
    def run_validations(self):
        # Statistical validation
        for col in self.df.select_dtypes(include=[np.number]).columns:
            stats_results = self._validate_column_statistics(col)
            self.validation_results[f"{col}_stats"] = stats_results
            
        # Structural validation
        self.validation_results['structural'] = self._validate_structure()
        
        # Data quality validation
        self.validation_results['quality'] = self._validate_quality()
        
        return pd.DataFrame(self.validation_results)
    
    def _validate_column_statistics(self, column):
        data = self.df[column]
        zscore = np.abs((data - data.mean()) / data.std())
        
        return {
            'missing_pct': data.isnull().mean() * 100,
            'outliers_pct': (zscore > 3).mean() * 100,
            'unique_values_pct': (data.nunique() / len(data)) * 100,
            'distribution_normal': stats.normaltest(
                data.dropna()
            ).pvalue > 0.05
        }
    
    def _validate_structure(self):
        return {
            'row_count': len(self.df),
            'duplicate_rows': self.df.duplicated().sum(),
            'memory_usage_mb': self.df.memory_usage().sum() / 1024**2
        }
    
    def _validate_quality(self):
        return {
            'completeness': 1 - self.df.isnull().mean().mean(),
            'consistency': self._check_consistency(),
            'validity': self._check_validity()
        }
    
    def _check_consistency(self):
        # Example consistency check
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        return all(self.df[numeric_cols].min() >= 0)  # Assume values should be positive
    
    def _check_validity(self):
        # Example validity check
        return all(self.df.index.is_monotonic_increasing)

# Example usage
np.random.seed(42)
data = pd.DataFrame({
    'value': np.random.normal(100, 15, 1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'timestamp': pd.date_range('2024-01-01', periods=1000)
})

validator = DataValidator(data)
validation_results = validator.run_validations()
print(validation_results)
```

Slide 11: Custom Pandas Extension Development

Implementing custom extensions for Pandas to add specialized functionality, including custom accessors, new data types, and extension arrays that integrate seamlessly with the Pandas ecosystem.

```python
import pandas as pd
from pandas.api.extensions import ExtensionDtype, register_extension_dtype
import numpy as np

@pd.api.extensions.register_dataframe_accessor("custom_stats")
class CustomStatsAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        
    def advanced_describe(self):
        numeric_data = self._obj.select_dtypes(include=[np.number])
        
        stats_dict = {
            'robust_mean': numeric_data.apply(
                lambda x: x.clip(
                    lower=x.quantile(0.1),
                    upper=x.quantile(0.9)
                ).mean()
            ),
            'median_abs_dev': numeric_data.apply(
                lambda x: np.median(np.abs(x - np.median(x)))
            ),
            'kurtosis_excess': numeric_data.apply(
                lambda x: stats.kurtosis(x.dropna())
            ),
            'distribution_type': numeric_data.apply(self._detect_distribution)
        }
        
        return pd.DataFrame(stats_dict)
    
    def _detect_distribution(self, series):
        clean_data = series.dropna()
        
        # Perform distribution tests
        normal_stat, normal_p = stats.normaltest(clean_data)
        uniform_stat, uniform_p = stats.kstest(
            clean_data, 'uniform',
            args=(clean_data.min(), clean_data.max())
        )
        
        if normal_p > 0.05:
            return 'Normal'
        elif uniform_p > 0.05:
            return 'Uniform'
        else:
            return 'Other'

# Example usage
np.random.seed(42)
df = pd.DataFrame({
    'normal': np.random.normal(0, 1, 1000),
    'uniform': np.random.uniform(-1, 1, 1000),
    'exponential': np.random.exponential(1, 1000)
})

print(df.custom_stats.advanced_describe())
```

Slide 12: Performance Optimization for Large Datasets

Exploring advanced techniques for handling large datasets in Pandas, including chunked processing, memory-efficient operations, and parallel computation strategies using dask integration.

```python
import pandas as pd
import numpy as np
from functools import partial
import multiprocessing as mp

class LargeDataProcessor:
    def __init__(self, filename, chunksize=10000):
        self.filename = filename
        self.chunksize = chunksize
        
    def process_in_chunks(self, operation_func):
        results = []
        
        # Process data in chunks
        for chunk in pd.read_csv(self.filename, chunksize=self.chunksize):
            result = operation_func(chunk)
            results.append(result)
            
        return pd.concat(results)
    
    def parallel_process(self, operation_func, n_processes=None):
        if n_processes is None:
            n_processes = mp.cpu_count()
            
        # Split data into approximately equal chunks
        total_rows = sum(1 for _ in open(self.filename)) - 1
        chunk_sizes = np.array_split(
            range(total_rows),
            n_processes
        )
        
        # Create pool and process chunks in parallel
        with mp.Pool(n_processes) as pool:
            results = pool.map(
                partial(
                    self._process_chunk,
                    operation_func=operation_func
                ),
                chunk_sizes
            )
            
        return pd.concat(results)
    
    def _process_chunk(self, row_ranges, operation_func):
        chunk = pd.read_csv(
            self.filename,
            skiprows=row_ranges[0],
            nrows=len(row_ranges)
        )
        return operation_func(chunk)

# Example usage
def complex_operation(df):
    # Simulate complex calculations
    result = pd.DataFrame({
        'rolling_mean': df['value'].rolling(window=100).mean(),
        'rolling_std': df['value'].rolling(window=100).std(),
        'ewma': df['value'].ewm(span=50).mean(),
        'zscore': (
            df['value'] - df['value'].rolling(window=100).mean()
        ) / df['value'].rolling(window=100).std()
    })
    return result

# Generate sample data
np.random.seed(42)
sample_data = pd.DataFrame({
    'value': np.random.normal(0, 1, 100000)
})
sample_data.to_csv('large_dataset.csv', index=False)

# Process data
processor = LargeDataProcessor('large_dataset.csv')
results = processor.parallel_process(complex_operation)
print(results.head())
```

Slide 13: Advanced Time Series Forecasting

Implementation of sophisticated time series forecasting techniques using Pandas, incorporating multiple seasonal decomposition, custom feature engineering, and ensemble prediction methods.

```python
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

class TimeSeriesForecaster:
    def __init__(self, data, frequency='D'):
        self.data = data
        self.frequency = frequency
        self.decomposition = None
        self.features = None
        
    def prepare_features(self, window_sizes=[7, 14, 30]):
        # Decompose series
        self.decomposition = seasonal_decompose(
            self.data,
            period=self._get_decomposition_period()
        )
        
        # Create features DataFrame
        features = pd.DataFrame({
            'trend': self.decomposition.trend,
            'seasonal': self.decomposition.seasonal,
            'residual': self.decomposition.resid
        })
        
        # Add rolling statistics
        for window in window_sizes:
            features[f'roll_mean_{window}'] = self.data.rolling(
                window=window
            ).mean()
            features[f'roll_std_{window}'] = self.data.rolling(
                window=window
            ).std()
            features[f'roll_max_{window}'] = self.data.rolling(
                window=window
            ).max()
            features[f'roll_min_{window}'] = self.data.rolling(
                window=window
            ).min()
        
        # Add lag features
        for lag in window_sizes:
            features[f'lag_{lag}'] = self.data.shift(lag)
        
        self.features = features
        return features
    
    def _get_decomposition_period(self):
        if self.frequency == 'D':
            return 7
        elif self.frequency == 'H':
            return 24
        elif self.frequency == 'M':
            return 12
        return 7  # default
    
    def forecast(self, steps=30):
        if self.features is None:
            self.prepare_features()
            
        # Implement forecasting logic here
        forecast = pd.Series(
            index=pd.date_range(
                self.data.index[-1] + pd.Timedelta('1D'),
                periods=steps,
                freq=self.frequency
            ),
            data=np.nan
        )
        
        # Simple example using last seasonal pattern
        seasonal_pattern = self.decomposition.seasonal[-steps:]
        trend_slope = (
            self.decomposition.trend[-1] - \
            self.decomposition.trend[-steps]
        ) / steps
        
        for i in range(steps):
            forecast.iloc[i] = (
                self.decomposition.trend[-1] + \
                (i + 1) * trend_slope + \
                seasonal_pattern.iloc[i % len(seasonal_pattern)]
            )
            
        return forecast

# Example usage
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=365, freq='D')
base = 100 + np.linspace(0, 10, 365)  # Trend
seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365)  # Seasonality
noise = np.random.normal(0, 1, 365)  # Random noise

data = pd.Series(
    base + seasonal + noise,
    index=dates
)

forecaster = TimeSeriesForecaster(data)
features = forecaster.prepare_features()
forecast = forecaster.forecast(steps=30)

print("\nFeature Overview:")
print(features.head())
print("\nForecast:")
print(forecast.head())
```

Slide 14: Advanced Data Merging and Concatenation

Implementing sophisticated data merging strategies for complex datasets, handling multiple keys, applying custom merge conditions, and managing memory-efficient concatenation operations for large datasets.

```python
class AdvancedMerger:
    def __init__(self):
        self.merge_stats = {}
        
    def smart_merge(self, left_df, right_df, merge_keys, merge_type='outer'):
        # Validate merge keys
        self._validate_merge_keys(left_df, right_df, merge_keys)
        
        # Perform merge with duplicate handling
        merged = pd.merge(
            left_df,
            right_df,
            on=merge_keys,
            how=merge_type,
            indicator=True,
            suffixes=('_left', '_right')
        )
        
        # Calculate merge statistics
        self.merge_stats = {
            'total_rows': len(merged),
            'matched_rows': len(merged[merged['_merge'] == 'both']),
            'left_only': len(merged[merged['_merge'] == 'left_only']),
            'right_only': len(merged[merged['_merge'] == 'right_only']),
            'memory_usage': merged.memory_usage().sum() / 1024**2
        }
        
        return self._post_process_merge(merged)
    
    def _validate_merge_keys(self, left_df, right_df, merge_keys):
        for key in merge_keys:
            if key not in left_df.columns or key not in right_df.columns:
                raise KeyError(f"Merge key {key} not found in both datasets")
            
            # Check data types compatibility
            if left_df[key].dtype != right_df[key].dtype:
                print(f"Warning: Data type mismatch for {key}")
                print(f"Left: {left_df[key].dtype}, Right: {right_df[key].dtype}")
    
    def _post_process_merge(self, merged_df):
        # Handle duplicate column names
        duplicate_cols = merged_df.columns[merged_df.columns.duplicated()]
        for col in duplicate_cols:
            base_col = col.replace('_left', '').replace('_right', '')
            left_col = f"{base_col}_left"
            right_col = f"{base_col}_right"
            
            # Combine duplicate columns using coalesce
            merged_df[base_col] = merged_df[left_col].combine_first(
                merged_df[right_col]
            )
            merged_df = merged_df.drop([left_col, right_col], axis=1)
        
        return merged_df

# Example usage
np.random.seed(42)

# Create sample datasets
left_data = pd.DataFrame({
    'id': range(1000),
    'value_a': np.random.normal(100, 10, 1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000)
})

right_data = pd.DataFrame({
    'id': range(500, 1500),
    'value_b': np.random.normal(200, 20, 1000),
    'category': np.random.choice(['B', 'C', 'D'], 1000)
})

merger = AdvancedMerger()
result = merger.smart_merge(
    left_data,
    right_data,
    merge_keys=['id', 'category']
)

print("Merge Statistics:")
print(pd.DataFrame(merger.merge_stats, index=['Values']))
print("\nMerged Data Sample:")
print(result.head())
```

Slide 15: Custom Data Types and Validation Framework

Implementing a comprehensive framework for custom data types and validation rules in Pandas, including type checking, constraint validation, and automatic data cleaning procedures.

```python
from pandas.api.extensions import ExtensionDtype, register_extension_dtype
import pandas as pd
import numpy as np
from typing import Dict, List, Any

class DataValidator:
    def __init__(self, schema: Dict[str, Dict[str, Any]]):
        """
        schema format:
        {
            'column_name': {
                'type': 'numeric|categorical|datetime',
                'constraints': {
                    'min': value,
                    'max': value,
                    'allowed_values': list,
                    'regex': pattern,
                    'custom_func': callable
                }
            }
        }
        """
        self.schema = schema
        self.validation_results = {}
        
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        for column, rules in self.schema.items():
            if column not in df.columns:
                raise ValueError(f"Column {column} not found in DataFrame")
                
            # Type validation
            df = self._validate_type(df, column, rules['type'])
            
            # Constraints validation
            if 'constraints' in rules:
                df = self._validate_constraints(
                    df,
                    column,
                    rules['constraints']
                )
        
        return df
    
    def _validate_type(
        self,
        df: pd.DataFrame,
        column: str,
        expected_type: str
    ) -> pd.DataFrame:
        if expected_type == 'numeric':
            try:
                df[column] = pd.to_numeric(df[column])
            except:
                df[column] = pd.to_numeric(df[column], errors='coerce')
                
        elif expected_type == 'datetime':
            try:
                df[column] = pd.to_datetime(df[column])
            except:
                df[column] = pd.to_datetime(df[column], errors='coerce')
                
        elif expected_type == 'categorical':
            df[column] = df[column].astype('category')
            
        return df
    
    def _validate_constraints(
        self,
        df: pd.DataFrame,
        column: str,
        constraints: Dict
    ) -> pd.DataFrame:
        if 'min' in constraints:
            df.loc[df[column] < constraints['min'], column] = np.nan
            
        if 'max' in constraints:
            df.loc[df[column] > constraints['max'], column] = np.nan
            
        if 'allowed_values' in constraints:
            df.loc[~df[column].isin(constraints['allowed_values']), column] = np.nan
            
        if 'regex' in constraints:
            mask = ~df[column].astype(str).str.match(constraints['regex'])
            df.loc[mask, column] = np.nan
            
        if 'custom_func' in constraints:
            df.loc[~df[column].apply(constraints['custom_func']), column] = np.nan
            
        return df

# Example usage
def custom_validation(x):
    return isinstance(x, (int, float)) and x % 2 == 0

schema = {
    'id': {
        'type': 'numeric',
        'constraints': {
            'min': 0,
            'custom_func': custom_validation
        }
    },
    'date': {
        'type': 'datetime'
    },
    'category': {
        'type': 'categorical',
        'constraints': {
            'allowed_values': ['A', 'B', 'C']
        }
    }
}

# Create sample data
data = pd.DataFrame({
    'id': range(-5, 5),
    'date': ['2024-01-01', 'invalid_date', '2024-01-03'] * 3 + ['2024-01-04'],
    'category': ['A', 'B', 'D', 'C', 'E', 'A', 'B', 'C', 'F', 'A']
})

validator = DataValidator(schema)
validated_df = validator.validate(data)

print("Original Data:")
print(data)
print("\nValidated Data:")
print(validated_df)
```

Slide 16: Additional Resources

*   "Optimizing Pandas Code for Large Datasets" - [https://arxiv.org/abs/2301.00819](https://arxiv.org/abs/2301.00819)
*   "Advanced Time Series Analysis with Pandas" - [https://arxiv.org/abs/2204.07748](https://arxiv.org/abs/2204.07748)
*   "Memory-Efficient Data Processing in Python" - [https://arxiv.org/abs/2112.09892](https://arxiv.org/abs/2112.09892)
*   "Modern DataFrame Operations and Best Practices" - [https://arxiv.org/abs/2303.15437](https://arxiv.org/abs/2303.15437)
*   "Statistical Computing with Pandas: New Horizons" - [https://arxiv.org/abs/2305.12983](https://arxiv.org/abs/2305.12983)

Note: These are example arxiv URLs for illustration purposes.

