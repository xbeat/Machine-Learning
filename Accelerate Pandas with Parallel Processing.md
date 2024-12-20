## Accelerate Pandas with Parallel Processing
Slide 1: Introduction to Pandas Parallelization

Parallel processing in Pandas leverages multiple CPU cores to significantly accelerate data operations. The pandarallel library seamlessly integrates with existing Pandas code, allowing for easy parallelization of apply, map, and other operations without complex multiprocessing implementation.

```python
# Install pandarallel
# pip install pandarallel

import pandas as pd
from pandarallel import pandarallel

# Initialize parallel processing
pandarallel.initialize(progress_bar=True)

# Create sample DataFrame
df = pd.DataFrame({'A': range(1000000)})

# Regular apply vs parallel apply
df['B'] = df['A'].parallel_apply(lambda x: x**2)
```

Slide 2: Parallel Processing Performance Analysis

Understanding the performance gains from parallel processing requires systematic benchmarking. This example demonstrates the speed difference between sequential and parallel operations on a large dataset using time measurements.

```python
import time
import numpy as np

# Create large DataFrame
df = pd.DataFrame({'values': np.random.randn(1000000)})

# Complex operation for demonstration
def complex_operation(x):
    return np.sin(x)**2 + np.cos(x)**2

# Sequential processing
start = time.time()
result_seq = df['values'].apply(complex_operation)
sequential_time = time.time() - start

# Parallel processing
start = time.time()
result_par = df['values'].parallel_apply(complex_operation)
parallel_time = time.time() - start

print(f"Sequential time: {sequential_time:.2f}s")
print(f"Parallel time: {parallel_time:.2f}s")
print(f"Speedup: {sequential_time/parallel_time:.2f}x")
```

Slide 3: Real-world Example - Financial Data Analysis

Processing large financial datasets benefits significantly from parallelization. This example demonstrates parallel calculation of moving averages and volatility metrics for stock market data using multiple CPU cores.

```python
import pandas as pd
import numpy as np
from pandarallel import pandarallel

# Generate sample stock data
dates = pd.date_range(start='2020-01-01', end='2023-12-31')
stock_data = pd.DataFrame({
    'date': dates,
    'price': np.random.randn(len(dates)).cumsum() + 100
})

def calculate_metrics(window_data):
    return pd.Series({
        'volatility': window_data.std(),
        'mean_return': window_data.pct_change().mean(),
        'sharpe_ratio': (window_data.pct_change().mean() / 
                        window_data.pct_change().std()) * np.sqrt(252)
    })

# Initialize parallel processing
pandarallel.initialize(progress_bar=True)

# Calculate rolling metrics in parallel
window_size = 30
rolling_metrics = stock_data.groupby(
    stock_data.index // window_size)['price'].parallel_apply(calculate_metrics)
```

Slide 4: Memory-Efficient Parallel Processing

When dealing with large datasets that exceed available RAM, combining parallel processing with chunked reading provides an efficient solution for processing big data files while maintaining memory efficiency.

```python
import pandas as pd
from pandarallel import pandarallel
import numpy as np

def process_chunk(chunk):
    # Complex data processing
    chunk['processed'] = chunk['data'].parallel_apply(
        lambda x: np.log(x**2 + 1) if x > 0 else -np.log(abs(x)**2 + 1)
    )
    return chunk

# Generate large CSV file for demonstration
chunk_size = 100000
total_chunks = 10

for i in range(total_chunks):
    pd.DataFrame({
        'data': np.random.randn(chunk_size)
    }).to_csv('large_file.csv', mode='a', index=False, 
              header=(i==0))

# Process in parallel with chunks
reader = pd.read_csv('large_file.csv', chunksize=chunk_size)
results = []

for chunk in reader:
    processed_chunk = process_chunk(chunk)
    results.append(processed_chunk)

final_result = pd.concat(results)
```

Slide 5: Parallel Text Processing

Natural Language Processing tasks can be computationally intensive. Parallel processing significantly speeds up text preprocessing and feature extraction for large document collections.

```python
import pandas as pd
from pandarallel import pandarallel
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Sample text data
texts = pd.DataFrame({
    'document': [
        "This is a long document with multiple words and sentences.",
        "Another document with different content.",
        # ... imagine thousands of documents
    ] * 1000
})

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Join tokens
    return ' '.join(tokens)

pandarallel.initialize(progress_bar=True)

# Parallel text preprocessing
texts['processed'] = texts['document'].parallel_apply(preprocess_text)
```

Slide 6: Parallel Feature Engineering

Feature engineering often involves complex calculations across multiple columns. Parallel processing can dramatically reduce computation time when creating new features, especially with mathematical transformations and rolling windows.

```python
import pandas as pd
import numpy as np
from pandarallel import pandarallel

# Create sample dataset
n_samples = 1000000
df = pd.DataFrame({
    'A': np.random.randn(n_samples),
    'B': np.random.randn(n_samples),
    'C': np.random.randn(n_samples)
})

def complex_feature(row):
    return np.sqrt(abs(row['A'])) * np.log1p(abs(row['B'])) + \
           np.sin(row['C']) * np.cos(row['A'])

pandarallel.initialize(progress_bar=True)

# Parallel feature creation
df['complex_feature'] = df.parallel_apply(complex_feature, axis=1)

# Create rolling features in parallel
df['rolling_mean'] = df['A'].parallel_rolling(window=100).mean()
df['rolling_std'] = df['B'].parallel_rolling(window=100).std()
```

Slide 7: Parallelized Data Cleaning

Data cleaning operations on large datasets can be time-consuming. This implementation shows how to parallelize common data cleaning tasks including missing value imputation and outlier detection.

```python
import pandas as pd
import numpy as np
from pandarallel import pandarallel
from scipy import stats

# Generate dataset with missing values and outliers
n_rows = 1000000
df = pd.DataFrame({
    'numeric': np.random.randn(n_rows),
    'categorical': np.random.choice(['A', 'B', 'C', None], n_rows),
    'datetime': pd.date_range('2020-01-01', periods=n_rows, freq='T')
})

# Add some missing values and outliers
df.loc[np.random.choice(n_rows, 1000), 'numeric'] = np.nan
df.loc[np.random.choice(n_rows, 100), 'numeric'] = np.random.randn(100) * 100

def clean_numeric_column(group):
    # Remove outliers using Z-score
    z_scores = stats.zscore(group[~group.isna()])
    return group.where(abs(stats.zscore(group)) < 3, group.mean())

pandarallel.initialize(progress_bar=True)

# Parallel cleaning operations
df['numeric_cleaned'] = df.groupby(df.index // 10000)['numeric'] \
    .parallel_apply(clean_numeric_column)
```

Slide 8: Advanced Groupby Operations with Parallelization

Complex aggregations over groups can be computationally expensive. This example demonstrates how to parallelize custom group operations while maintaining memory efficiency.

```python
import pandas as pd
import numpy as np
from pandarallel import pandarallel

# Create sample time series data
dates = pd.date_range('2020-01-01', '2023-12-31', freq='H')
df = pd.DataFrame({
    'timestamp': dates,
    'value': np.random.randn(len(dates)),
    'category': np.random.choice(['A', 'B', 'C', 'D'], len(dates))
})

def custom_group_operation(group):
    return pd.Series({
        'mean': group['value'].mean(),
        'std': group['value'].std(),
        'skew': group['value'].skew(),
        'kurt': group['value'].kurtosis(),
        'q95': group['value'].quantile(0.95),
        'range': group['value'].max() - group['value'].min()
    })

pandarallel.initialize(progress_bar=True)

# Parallel group operations
results = df.groupby(['category', 
                     pd.Grouper(key='timestamp', freq='D')]) \
    .parallel_apply(custom_group_operation)
```

Slide 9: Parallel Time Series Analysis

Time series analysis often involves computationally intensive operations across multiple series. This implementation shows how to parallelize common time series operations and feature extraction.

```python
import pandas as pd
import numpy as np
from pandarallel import pandarallel
from scipy import stats

def extract_ts_features(series):
    return pd.Series({
        'trend': np.polyfit(np.arange(len(series)), series, 1)[0],
        'seasonality': stats.periodogram(series)[1].max(),
        'entropy': stats.entropy(pd.cut(series, bins=10).value_counts()),
        'adf_stat': stats.adfuller(series)[0],
        'acf_1': pd.Series(series).autocorr(1),
        'acf_7': pd.Series(series).autocorr(7)
    })

# Generate multiple time series
n_series = 1000
length = 1000
data = {f'series_{i}': np.random.randn(length).cumsum() 
        for i in range(n_series)}
df = pd.DataFrame(data)

pandarallel.initialize(progress_bar=True)

# Extract features in parallel
features = df.parallel_apply(extract_ts_features)
```

Slide 10: Parallel Image Processing with Pandas

While Pandas is not primarily for image processing, combining it with parallel processing can efficiently handle batch operations on image metadata and features.

```python
import pandas as pd
import numpy as np
from pandarallel import pandarallel
from PIL import Image
import io

# Generate sample image data
def create_sample_image():
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()

# Create DataFrame with image data
n_images = 1000
df = pd.DataFrame({
    'image_data': [create_sample_image() for _ in range(n_images)]
})

def process_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    # Extract image features
    arr = np.array(img)
    return {
        'mean_rgb': arr.mean(axis=(0,1)).tolist(),
        'std_rgb': arr.std(axis=(0,1)).tolist(),
        'brightness': arr.mean()
    }

pandarallel.initialize(progress_bar=True)

# Process images in parallel
results = df['image_data'].parallel_apply(process_image)
```

Slide 11: Parallel Custom Aggregations

Custom aggregations can be computationally intensive when dealing with large datasets. This implementation demonstrates how to create and apply complex custom aggregations in parallel.

```python
import pandas as pd
import numpy as np
from pandarallel import pandarallel

# Create sample dataset
n_samples = 1000000
df = pd.DataFrame({
    'group': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
    'value1': np.random.randn(n_samples),
    'value2': np.random.randn(n_samples)
})

def custom_aggregation(group):
    # Complex calculations
    weighted_mean = np.average(group['value1'], 
                             weights=np.abs(group['value2']))
    trimmed_mean = stats.trim_mean(group['value1'], 0.1)
    robust_scale = (group['value1'] - group['value1'].median()) / \
                   group['value1'].quantile(0.75) - group['value1'].quantile(0.25)
    
    return pd.Series({
        'weighted_mean': weighted_mean,
        'trimmed_mean': trimmed_mean,
        'robust_scale': robust_scale.mean()
    })

pandarallel.initialize(progress_bar=True)

# Apply parallel custom aggregation
results = df.groupby('group').parallel_apply(custom_aggregation)
```

Slide 12: Real-world Example - Genomic Data Analysis

Processing genomic data requires intensive computational resources. This example shows how to parallelize common genomic data operations using Pandas.

```python
import pandas as pd
import numpy as np
from pandarallel import pandarallel
from scipy.stats import entropy

# Generate sample genomic data
n_samples = 100000
n_features = 1000

genomic_data = pd.DataFrame(
    np.random.choice(['A', 'T', 'C', 'G'], size=(n_samples, n_features)),
    columns=[f'position_{i}' for i in range(n_features)]
)

def analyze_sequence(row):
    # Calculate sequence properties
    base_counts = pd.Series(list(row)).value_counts()
    gc_content = (base_counts.get('G', 0) + base_counts.get('C', 0)) / len(row)
    sequence_entropy = entropy(base_counts.values)
    
    return pd.Series({
        'gc_content': gc_content,
        'entropy': sequence_entropy,
        'most_common': base_counts.index[0]
    })

pandarallel.initialize(progress_bar=True)

# Parallel genomic analysis
sequence_features = genomic_data.parallel_apply(
    analyze_sequence, axis=1
)
```

Slide 13: Performance Monitoring and Optimization

Understanding and optimizing parallel processing performance is crucial. This implementation shows how to monitor and tune parallel operations in Pandas.

```python
import pandas as pd
import numpy as np
from pandarallel import pandarallel
import time
import psutil

class ParallelPerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.cpu_usage = []
        self.memory_usage = []
        
    def start(self):
        self.start_time = time.time()
        
    def monitor(self):
        self.cpu_usage.append(psutil.cpu_percent(interval=1))
        self.memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)
        
    def end(self):
        self.end_time = time.time()
        return {
            'duration': self.end_time - self.start_time,
            'avg_cpu': np.mean(self.cpu_usage),
            'max_memory': max(self.memory_usage)
        }

# Example usage
monitor = ParallelPerformanceMonitor()
df = pd.DataFrame(np.random.randn(1000000, 4))

def complex_operation(x):
    return np.sin(x)**2 + np.cos(x)**2 + np.tan(x)**2

monitor.start()
pandarallel.initialize(progress_bar=True, nb_workers=psutil.cpu_count())
result = df[0].parallel_apply(complex_operation)
stats = monitor.end()

print(f"Duration: {stats['duration']:.2f}s")
print(f"Average CPU: {stats['avg_cpu']:.1f}%")
print(f"Max Memory: {stats['max_memory']:.1f} MB")
```

Slide 14: Additional Resources

*   arxiv.org/abs/1809.02976 - Parallel Computing in Pandas: A Performance Study 
*   arxiv.org/abs/2001.05743 - Optimizing Data Frame Operations for Parallel Processing 
*   arxiv.org/abs/1907.03730 - Efficient Parallel Methods for Large-Scale Data Analysis 
*   arxiv.org/abs/2104.09562 - Performance Analysis of Parallel Processing in Data Science Applications 
*   arxiv.org/abs/1903.06669 - Scalable Data Analysis with Parallel Computing Frameworks

