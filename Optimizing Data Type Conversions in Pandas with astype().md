## Optimizing Data Type Conversions in Pandas with astype()
Slide 1: Understanding Data Type Conversion in Pandas

Data type conversion is a critical operation in data preprocessing that impacts memory usage and computational efficiency. The astype() method in Pandas enables explicit conversion of column data types, ensuring data integrity and optimal performance in downstream analysis.

```python
import pandas as pd
import numpy as np

# Create sample DataFrame
df = pd.DataFrame({
    'string_col': ['1', '2', '3'],
    'float_col': ['1.1', '2.2', '3.3'],
    'bool_col': ['True', 'False', 'True']
})

# Individual conversions
df['string_col'] = df['string_col'].astype(int)
df['float_col'] = df['float_col'].astype(float)
df['bool_col'] = df['bool_col'].astype(bool)

print("Data types after conversion:")
print(df.dtypes)
```

Slide 2: Dictionary-Based Type Conversion

Using a dictionary to specify data types allows for simultaneous conversion of multiple columns, reducing code verbosity and improving maintainability. This approach is particularly valuable when dealing with large datasets containing numerous columns.

```python
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'id': ['1', '2', '3'],
    'value': ['1.1', '2.2', '3.3'],
    'status': ['True', 'False', 'True'],
    'category': ['A', 'B', 'C']
})

# Define type conversion dictionary
dtype_dict = {
    'id': int,
    'value': float,
    'status': bool,
    'category': 'category'  # Category type for memory optimization
}

# Convert multiple columns at once
df = df.astype(dtype_dict)

print("Updated data types:")
print(df.dtypes)
```

Slide 3: Memory Optimization Through Type Conversion

Memory efficiency is crucial when handling large datasets. Strategic data type conversion can significantly reduce memory usage by selecting appropriate data types that minimize storage requirements while maintaining data precision.

```python
import pandas as pd
import numpy as np

# Generate sample data
df = pd.DataFrame({
    'int_col': np.random.randint(0, 100, 10000),
    'float_col': np.random.random(10000),
    'category_col': np.random.choice(['A', 'B', 'C'], 10000)
})

# Memory usage before optimization
print("Memory usage before optimization:")
print(df.memory_usage(deep=True))

# Optimize types
dtype_optimize = {
    'int_col': 'int8',  # Small integers (-128 to 127)
    'float_col': 'float32',  # Reduced precision float
    'category_col': 'category'  # Category for repeated strings
}

df_optimized = df.astype(dtype_optimize)

print("\nMemory usage after optimization:")
print(df_optimized.memory_usage(deep=True))
```

Slide 4: Handling Mixed Data Types

Real-world datasets often contain columns with mixed data types. Understanding how to properly handle these cases is essential for maintaining data integrity while performing type conversion operations.

```python
import pandas as pd
import numpy as np

# Create DataFrame with mixed types
df = pd.DataFrame({
    'mixed_numbers': ['1', '2.5', '3', 'NA', '4.2'],
    'mixed_strings': ['True', 'yes', '1', 'false', '0']
})

# Custom conversion function
def safe_numeric_convert(series):
    try:
        return pd.to_numeric(series, errors='coerce')
    except:
        return series

# Handle mixed types with dictionary
conversion_dict = {
    'mixed_numbers': lambda x: safe_numeric_convert(x).astype('float64'),
    'mixed_strings': lambda x: x.map({'True': True, 'yes': True, '1': True,
                                     'false': False, '0': False})
}

df_converted = df.transform(conversion_dict)
print("Converted DataFrame:")
print(df_converted)
print("\nData types:")
print(df_converted.dtypes)
```

Slide 5: Real-World Example - Financial Data Processing

Processing financial data requires precise handling of numerical values and efficient type conversion for large datasets. This example demonstrates optimizing a financial dataset for analysis while maintaining numerical accuracy.

```python
import pandas as pd
import numpy as np

# Generate sample financial data
np.random.seed(42)
financial_data = pd.DataFrame({
    'transaction_id': [str(i) for i in range(1000)],
    'amount': np.random.uniform(100, 10000, 1000).astype(str),
    'currency': np.random.choice(['USD', 'EUR', 'GBP'], 1000),
    'timestamp': pd.date_range(start='2023-01-01', periods=1000).astype(str),
    'status': np.random.choice(['pending', 'completed', 'failed'], 1000)
})

# Define optimization dictionary
financial_dtypes = {
    'transaction_id': 'string',
    'amount': 'float64',
    'currency': 'category',
    'timestamp': 'datetime64[ns]',
    'status': 'category'
}

# Optimize dataset
optimized_financial = financial_data.astype(financial_dtypes)

print("Memory usage per column:")
print(optimized_financial.memory_usage(deep=True))
print("\nData types:")
print(optimized_financial.dtypes)
```

Slide 6: Advanced Type Conversion Techniques

Advanced type conversion scenarios often require combining multiple approaches and handling complex data structures. This implementation showcases nested type conversions and custom type mapping for specialized data formats.

```python
import pandas as pd
import numpy as np

# Create complex DataFrame
df = pd.DataFrame({
    'nested_lists': str([1, 2, 3]),
    'json_string': '{"value": 42}',
    'complex_date': ['2023-Q1', '2023-Q2', '2023-Q3'],
    'custom_format': ['A_123', 'B_456', 'C_789']
})

# Define custom conversion functions
def convert_list_string(x):
    return eval(x) if isinstance(x, str) else x

def parse_quarter_date(x):
    quarter = int(x[-1])
    year = int(x[:4])
    month = (quarter - 1) * 3 + 1
    return pd.to_datetime(f'{year}-{month:02d}-01')

# Complex conversion dictionary
advanced_dtypes = {
    'nested_lists': convert_list_string,
    'json_string': lambda x: pd.json_normalize(eval(x.replace('null', 'None'))),
    'complex_date': parse_quarter_date,
    'custom_format': lambda x: x.str.extract(r'([A-Z])_(\d+)').astype({'1': 'category', '2': int})
}

# Apply conversions
df_converted = df.transform(advanced_dtypes)
print("Converted complex types:")
print(df_converted.dtypes)
```

Slide 7: Performance Optimization with Batch Processing

When dealing with large-scale data type conversions, batch processing can significantly improve performance. This implementation demonstrates efficient handling of large datasets through chunked processing.

```python
import pandas as pd
import numpy as np

def batch_convert_types(filepath, dtype_dict, chunksize=10000):
    """
    Process large CSV files in chunks for memory-efficient type conversion
    """
    # Create empty list to store chunks
    chunks = []
    
    # Process file in chunks
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        # Apply type conversion to chunk
        converted_chunk = chunk.astype(dtype_dict)
        chunks.append(converted_chunk)
        
    # Concatenate all chunks
    return pd.concat(chunks, ignore_index=True)

# Generate sample large dataset
large_df = pd.DataFrame({
    'id': range(100000),
    'value': np.random.random(100000),
    'category': np.random.choice(['A', 'B', 'C'], 100000)
}).to_csv('large_dataset.csv', index=False)

# Define conversion types
conversion_types = {
    'id': 'int32',
    'value': 'float32',
    'category': 'category'
}

# Process in batches
result = batch_convert_types('large_dataset.csv', conversion_types)
print("Memory usage after batch processing:")
print(result.memory_usage(deep=True))
```

Slide 8: Error Handling in Type Conversion

Robust error handling is crucial when performing data type conversions, especially with real-world data that may contain unexpected values or formats.

```python
import pandas as pd
import numpy as np

class TypeConversionHandler:
    def __init__(self, df):
        self.df = df
        self.conversion_log = []
        
    def safe_convert(self, dtype_dict):
        result_df = self.df.copy()
        
        for column, dtype in dtype_dict.items():
            try:
                result_df[column] = result_df[column].astype(dtype)
                self.conversion_log.append(f"Successfully converted {column} to {dtype}")
            except Exception as e:
                self.conversion_log.append(f"Error converting {column}: {str(e)}")
                # Attempt fallback conversion
                result_df[column] = self._fallback_conversion(result_df[column], dtype)
        
        return result_df
    
    def _fallback_conversion(self, series, target_dtype):
        try:
            if target_dtype in ['int64', 'float64']:
                return pd.to_numeric(series, errors='coerce')
            elif target_dtype == 'datetime64[ns]':
                return pd.to_datetime(series, errors='coerce')
            else:
                return series
        except:
            return series
        
# Test with problematic data
problem_df = pd.DataFrame({
    'numbers': ['1', '2.5', 'invalid', '4'],
    'dates': ['2023-01-01', 'invalid_date', '2023-02-01'],
    'categories': ['A', 'B', None, 'D']
})

converter = TypeConversionHandler(problem_df)
conversion_dict = {
    'numbers': 'float64',
    'dates': 'datetime64[ns]',
    'categories': 'category'
}

result = converter.safe_convert(conversion_dict)
print("Conversion results:")
print(result.dtypes)
print("\nConversion log:")
for log in converter.conversion_log:
    print(log)
```

Slide 9: Type Inference and Automatic Optimization

Implementing intelligent type inference can automatically determine optimal data types based on column contents, leading to improved memory usage without manual specification.

```python
import pandas as pd
import numpy as np

class SmartTypeOptimizer:
    def __init__(self, df):
        self.df = df
        
    def infer_optimal_types(self):
        dtype_dict = {}
        
        for column in self.df.columns:
            # Sample data for type inference
            sample = self.df[column].dropna().head(1000)
            
            if self._is_categorical(sample):
                dtype_dict[column] = 'category'
            elif self._is_integer(sample):
                dtype_dict[column] = self._get_optimal_integer_type(sample)
            elif self._is_float(sample):
                dtype_dict[column] = self._get_optimal_float_type(sample)
            elif self._is_datetime(sample):
                dtype_dict[column] = 'datetime64[ns]'
            else:
                dtype_dict[column] = 'string'
                
        return dtype_dict
    
    def _is_categorical(self, series):
        unique_ratio = len(series.unique()) / len(series)
        return unique_ratio < 0.1  # Less than 10% unique values
    
    def _is_integer(self, series):
        return pd.to_numeric(series, errors='coerce').apply(float.is_integer).all()
    
    def _is_float(self, series):
        return pd.to_numeric(series, errors='coerce').notna().all()
    
    def _is_datetime(self, series):
        return pd.to_datetime(series, errors='coerce').notna().all()
    
    def _get_optimal_integer_type(self, series):
        max_val = pd.to_numeric(series).max()
        min_val = pd.to_numeric(series).min()
        
        if min_val >= 0:
            if max_val < 255: return 'uint8'
            if max_val < 65535: return 'uint16'
            return 'uint32'
        else:
            if min_val > -128 and max_val < 127: return 'int8'
            if min_val > -32768 and max_val < 32767: return 'int16'
            return 'int32'
    
    def _get_optimal_float_type(self, series):
        return 'float32' if series.abs().max() < 1e38 else 'float64'

# Test the optimizer
test_df = pd.DataFrame({
    'id': range(1000),
    'value': np.random.random(1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'timestamp': pd.date_range('2023-01-01', periods=1000)
})

optimizer = SmartTypeOptimizer(test_df)
optimal_types = optimizer.infer_optimal_types()

print("Inferred optimal types:")
for col, dtype in optimal_types.items():
    print(f"{col}: {dtype}")

# Apply optimized types
optimized_df = test_df.astype(optimal_types)
print("\nMemory usage after optimization:")
print(optimized_df.memory_usage(deep=True))
```

Slide 10: Real-World Example - Time Series Data Processing

Time series data often requires specific type conversions to handle temporal components efficiently while maintaining analytical capabilities. This implementation demonstrates optimizing financial time series data.

```python
import pandas as pd
import numpy as np
from datetime import datetime

# Generate sample time series financial data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
df_timeseries = pd.DataFrame({
    'timestamp': dates.astype(str),
    'price': np.random.uniform(100, 200, len(dates)).astype(str),
    'volume': np.random.randint(1000, 10000, len(dates)).astype(str),
    'trade_type': np.random.choice(['market', 'limit'], len(dates)),
    'trade_id': [f'TRADE_{i}' for i in range(len(dates))]
})

# Define optimized conversion dictionary
timeseries_dtypes = {
    'timestamp': 'datetime64[ns]',
    'price': 'float32',
    'volume': 'int32',
    'trade_type': 'category',
    'trade_id': 'string'
}

# Memory usage before optimization
print("Memory usage before optimization (MB):")
print(df_timeseries.memory_usage(deep=True).sum() / 1024**2)

# Apply optimized conversions with performance monitoring
import time

start_time = time.time()
df_optimized = df_timeseries.astype(timeseries_dtypes)
conversion_time = time.time() - start_time

print("\nMemory usage after optimization (MB):")
print(df_optimized.memory_usage(deep=True).sum() / 1024**2)
print(f"\nConversion time: {conversion_time:.2f} seconds")

# Verify data integrity
print("\nData sample after conversion:")
print(df_optimized.head())
print("\nData types:")
print(df_optimized.dtypes)
```

Slide 11: Handling Missing Values During Type Conversion

Missing value handling is crucial during type conversion to prevent data loss and ensure proper representation of null values across different data types.

```python
import pandas as pd
import numpy as np

class NullAwareTypeConverter:
    def __init__(self, df):
        self.df = df
        
    def convert_with_nulls(self, dtype_dict):
        result_df = self.df.copy()
        
        for column, dtype in dtype_dict.items():
            result_df[column] = self._convert_column(result_df[column], dtype)
            
        return result_df
    
    def _convert_column(self, series, dtype):
        # Handle different null representations
        null_values = ['nan', 'null', 'none', 'na', '', 'NaN', 'NULL', 'None', 'NA']
        series = series.replace(null_values, np.nan)
        
        if dtype == 'category':
            return series.astype('category')
        elif dtype in ['int8', 'int16', 'int32', 'int64']:
            return pd.to_numeric(series, errors='coerce').astype('Int64')  # Nullable integer
        elif dtype.startswith('float'):
            return pd.to_numeric(series, errors='coerce').astype(dtype)
        elif dtype == 'boolean':
            return series.map({'true': True, 'false': False, '1': True, '0': False,
                             True: True, False: False}).astype('boolean')
        else:
            return series.astype(dtype)

# Test with data containing various null representations
test_data = pd.DataFrame({
    'integers': ['1', '2', 'nan', '4', 'null', '6'],
    'floats': ['1.1', 'na', '3.3', 'NULL', '5.5', 'none'],
    'categories': ['A', 'B', '', 'D', 'NA', 'F'],
    'booleans': ['true', 'false', 'NA', '1', '0', 'null']
})

converter = NullAwareTypeConverter(test_data)
null_aware_types = {
    'integers': 'int32',
    'floats': 'float32',
    'categories': 'category',
    'booleans': 'boolean'
}

result = converter.convert_with_nulls(null_aware_types)
print("Converted data with null handling:")
print(result)
print("\nData types:")
print(result.dtypes)
print("\nNull value counts:")
print(result.isna().sum())
```

Slide 12: Performance Comparison of Type Conversion Methods

Understanding the performance implications of different type conversion approaches helps in choosing the most efficient method for specific use cases.

```python
import pandas as pd
import numpy as np
import time

def benchmark_conversions(df, iterations=5):
    results = {
        'individual': [],
        'dictionary': [],
        'batch': []
    }
    
    dtype_dict = {
        'col1': 'int32',
        'col2': 'float32',
        'col3': 'category',
        'col4': 'string'
    }
    
    for _ in range(iterations):
        # Individual conversions
        start_time = time.time()
        df_individual = df.copy()
        for col, dtype in dtype_dict.items():
            df_individual[col] = df_individual[col].astype(dtype)
        results['individual'].append(time.time() - start_time)
        
        # Dictionary conversion
        start_time = time.time()
        df_dict = df.copy().astype(dtype_dict)
        results['dictionary'].append(time.time() - start_time)
        
        # Batch conversion
        start_time = time.time()
        chunks = np.array_split(df, 5)
        df_batch = pd.concat([chunk.astype(dtype_dict) for chunk in chunks])
        results['dictionary'].append(time.time() - start_time)
    
    return {method: np.mean(times) for method, times in results.items()}

# Generate test data
np.random.seed(42)
test_size = 1000000
df_test = pd.DataFrame({
    'col1': np.random.randint(1, 100, test_size).astype(str),
    'col2': np.random.random(test_size).astype(str),
    'col3': np.random.choice(['A', 'B', 'C'], test_size),
    'col4': [f'str_{i}' for i in range(test_size)]
})

# Run benchmark
benchmark_results = benchmark_conversions(df_test)

print("Performance comparison (average seconds):")
for method, time_taken in benchmark_results.items():
    print(f"{method}: {time_taken:.4f}")
```

Slide 13: Additional Resources

*   Research Papers:
*   [https://doi.org/10.1109/TKDE.2019.2911071](https://doi.org/10.1109/TKDE.2019.2911071) - "Efficient Data Type Conversion in Modern Database Systems"
*   [https://arxiv.org/abs/2103.05561](https://arxiv.org/abs/2103.05561) - "Optimizing Data Processing Pipelines for Time Series Analysis"
*   [https://arxiv.org/abs/2008.13600](https://arxiv.org/abs/2008.13600) - "Performance Optimization Techniques for Large-Scale Data Processing"
*   Online Resources:
*   Pandas Official Documentation: [https://pandas.pydata.org/docs/user\_guide/basics.html#dtypes](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes)
*   NumPy Data Types Guide: [https://numpy.org/doc/stable/user/basics.types.html](https://numpy.org/doc/stable/user/basics.types.html)
*   Python Memory Management: [https://docs.python.org/3/c-api/memory.html](https://docs.python.org/3/c-api/memory.html)
*   Recommended Search Terms:
*   "Pandas memory optimization techniques"
*   "Data type conversion best practices Python"
*   "Time series data optimization pandas"

