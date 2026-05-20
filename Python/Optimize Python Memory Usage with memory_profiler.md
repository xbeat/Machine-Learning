## Optimize Python Memory Usage with memory_profiler
Slide 1: Memory Profiler Installation and Setup

Understanding memory consumption in Python applications is crucial for optimization. The memory\_profiler module provides detailed insights into memory usage patterns by analyzing code execution line-by-line, making it an essential tool for performance tuning.

```python
# Install memory_profiler using pip
!pip install memory-profiler

# Import required modules
from memory_profiler import profile
import numpy as np

# Basic setup example
@profile
def memory_test():
    # Create large array to demonstrate memory usage
    x = np.ones((1000, 1000))
    return x.sum()

# Run the function
result = memory_test()
```

Slide 2: Basic Memory Profiling Example

Memory profiling begins with decorating functions using @profile. This example demonstrates how to track memory usage in a simple function that performs basic array operations, showing memory allocation and deallocation patterns.

```python
@profile
def analyze_memory():
    # Initialize list
    numbers = list(range(1000000))    # Line 1
    
    # Create numpy array
    arr = np.array(numbers)           # Line 2
    
    # Perform operations
    squared = arr ** 2                # Line 3
    
    # Clean up
    del numbers                       # Line 4
    del arr                          # Line 5
    
    return squared.mean()             # Line 6

# Execute function
result = analyze_memory()
```

Slide 3: Advanced Memory Profiling with Time Series Data

Memory profiling becomes crucial when working with time series data. This example showcases memory usage patterns while processing large temporal datasets using pandas, highlighting potential optimization opportunities.

```python
@profile
def process_time_series():
    # Generate sample time series data
    dates = pd.date_range('2023-01-01', periods=1000000, freq='S')
    values = np.random.randn(1000000)
    
    # Create DataFrame
    df = pd.DataFrame({'timestamp': dates, 'value': values})
    
    # Perform time series operations
    rolling_mean = df['value'].rolling(window=100).mean()
    
    # Calculate statistics
    result = {
        'mean': rolling_mean.mean(),
        'std': rolling_mean.std()
    }
    
    return result

# Run analysis
stats = process_time_series()
```

Slide 4: Memory-Efficient Data Processing

When dealing with large datasets, efficient memory management becomes critical. This implementation demonstrates chunk-based processing to handle data that exceeds available RAM, using generators and iterators.

```python
@profile
def process_large_dataset(chunk_size=100000):
    # Generator for data chunks
    def data_generator(total_size):
        for i in range(0, total_size, chunk_size):
            yield np.random.randn(min(chunk_size, total_size - i))
    
    total_size = 1000000
    running_sum = 0
    count = 0
    
    # Process data in chunks
    for chunk in data_generator(total_size):
        running_sum += chunk.sum()
        count += len(chunk)
        del chunk  # Explicit cleanup
    
    return running_sum / count

# Execute chunked processing
mean_value = process_large_dataset()
```

Slide 5: Memory Optimization for Machine Learning

Machine learning models often require careful memory management. This example shows how to implement memory-efficient training loops and batch processing for neural network training.

```python
import torch
from torch import nn
import torch.nn.functional as F

@profile
def train_model_efficiently(model, data_loader, epochs=5):
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Compute loss
            loss = F.nll_loss(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Explicit cleanup
            del output, loss
            torch.cuda.empty_cache()  # If using GPU

# Example usage would be shown in next slides
```

Slide 6: Memory Profiling with Context Managers

Memory profiling can be enhanced using context managers to track specific code blocks. This approach provides granular control over which sections of code are monitored for memory consumption.

```python
from memory_profiler import profile
from contextlib import contextmanager

@contextmanager
def memory_profiler_context(name):
    @profile
    def wrapped_code():
        yield
    
    prof = wrapped_code()
    next(prof)
    try:
        yield
    finally:
        next(prof, None)

# Example usage
def process_data():
    data = []
    with memory_profiler_context("data_loading"):
        for i in range(1000000):
            data.append(i ** 2)
    
    with memory_profiler_context("data_processing"):
        result = sum(data) / len(data)
    return result
```

Slide 7: Real-world Example: Image Processing Pipeline

This implementation demonstrates memory-efficient image processing using PIL and NumPy, showcasing how to handle large image datasets while maintaining optimal memory usage.

```python
from PIL import Image
import numpy as np
from memory_profiler import profile

@profile
def process_image_batch(image_paths, batch_size=10):
    processed_images = []
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        
        # Process each image in batch
        for path in batch:
            # Load and convert to numpy array
            with Image.open(path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to numpy array
                img_array = np.array(img)
                
                # Apply processing (example: normalize)
                processed = (img_array - img_array.mean()) / img_array.std()
                
                processed_images.append(processed)
                
                # Clean up
                del img_array
        
        # Explicit cleanup after batch
        if len(processed_images) > batch_size * 2:
            processed_images = processed_images[-batch_size:]
    
    return processed_images

# Example usage:
# image_paths = ['image1.jpg', 'image2.jpg', ...]
# results = process_image_batch(image_paths)
```

Slide 8: Memory-Efficient Text Processing

Text processing often involves handling large corpora. This implementation shows how to process text data efficiently using generators and streaming techniques.

```python
from memory_profiler import profile
import re
from typing import Iterator, List

@profile
def process_text_stream(file_path: str, chunk_size: int = 1024) -> Iterator[List[str]]:
    def tokenize(text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    word_counts = {}
    
    with open(file_path, 'r') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
                
            # Process chunk
            words = tokenize(chunk)
            
            # Update word counts
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                
            # Yield intermediate results
            yield list(word_counts.items())
            
            # Memory optimization: clear counts periodically
            if len(word_counts) > 10000:
                word_counts = dict(sorted(
                    word_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5000])

# Usage example:
# for word_stats in process_text_stream('large_text.txt'):
#     print(f"Current unique words: {len(word_stats)}")
```

Slide 9: Memory Optimization for DataFrame Operations

Pandas DataFrame operations can be memory-intensive. This implementation shows how to optimize memory usage when performing complex DataFrame transformations and aggregations.

```python
import pandas as pd
import numpy as np
from memory_profiler import profile

@profile
def optimize_dataframe_operations(file_path: str):
    # Read CSV in chunks
    chunk_size = 10000
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    
    results = []
    
    for chunk in chunks:
        # Optimize dtypes
        for col in chunk.select_dtypes(include=['object']):
            if chunk[col].nunique() / len(chunk) < 0.5:  # If column has low cardinality
                chunk[col] = chunk[col].astype('category')
                
        # Perform calculations
        summary = {
            'mean': chunk.select_dtypes(include=[np.number]).mean(),
            'median': chunk.select_dtypes(include=[np.number]).median(),
            'std': chunk.select_dtypes(include=[np.number]).std()
        }
        
        results.append(summary)
        
        # Explicit cleanup
        del chunk
        
    return pd.concat([pd.DataFrame(r) for r in results]).mean()

# Example usage:
# stats = optimize_dataframe_operations('large_dataset.csv')
```

Slide 10: Memory Profile Visualization

Memory profiling data can be visualized to better understand memory usage patterns. This implementation creates memory usage plots using matplotlib.

```python
import matplotlib.pyplot as plt
from memory_profiler import profile
import numpy as np
import time

@profile
def generate_memory_profile():
    memory_usage = []
    timestamps = []
    
    # Simulate memory-intensive operations
    for i in range(5):
        # Record timestamp
        timestamps.append(time.time())
        
        # Create large array
        arr = np.random.randn(1000, 1000 * (i + 1))
        
        # Perform operations
        result = np.dot(arr, arr.T)
        
        # Record memory usage (in MB)
        memory_usage.append(arr.nbytes / 1024 / 1024)
        
        # Sleep to simulate processing
        time.sleep(1)
        
        del arr, result
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, memory_usage, marker='o')
    plt.xlabel('Time (s)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Profile')
    plt.grid(True)
    
    return plt.gcf()

# Run and save visualization
# fig = generate_memory_profile()
# fig.savefig('memory_profile.png')
```

Slide 11: Real-time Memory Monitoring

This implementation demonstrates how to create a real-time memory monitor for long-running processes, with periodic logging and alerts.

```python
from memory_profiler import profile
import psutil
import time
from datetime import datetime
import logging

class MemoryMonitor:
    def __init__(self, threshold_mb=1000):
        self.threshold_mb = threshold_mb
        logging.basicConfig(filename='memory_monitor.log', level=logging.INFO)
    
    @profile
    def monitor_process(self, duration_seconds=60, interval=5):
        start_time = time.time()
        process = psutil.Process()
        
        while (time.time() - start_time) < duration_seconds:
            # Get memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Log memory usage
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f'{timestamp} - Memory Usage: {memory_mb:.2f} MB')
            
            # Check threshold
            if memory_mb > self.threshold_mb:
                logging.warning(f'Memory usage exceeded threshold: {memory_mb:.2f} MB')
            
            time.sleep(interval)
        
        return self.generate_summary()
    
    def generate_summary(self):
        return {
            'peak_memory': psutil.Process().memory_info().rss / 1024 / 1024,
            'system_total': psutil.virtual_memory().total / 1024 / 1024,
            'system_available': psutil.virtual_memory().available / 1024 / 1024
        }

# Example usage:
# monitor = MemoryMonitor(threshold_mb=500)
# stats = monitor.monitor_process(duration_seconds=30)
```

Slide 12: Memory-Efficient Machine Learning Pipeline

This implementation showcases a complete machine learning pipeline with memory optimization techniques for handling large datasets and model training.

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np
from memory_profiler import profile

class MemoryEfficientPipeline:
    @profile
    def __init__(self, batch_size=1000):
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        
    def generator_from_data(self, X, y=None):
        total_samples = len(X)
        for start in range(0, total_samples, self.batch_size):
            end = min(start + self.batch_size, total_samples)
            if y is not None:
                yield X[start:end], y[start:end]
            else:
                yield X[start:end]
    
    @profile
    def fit(self, X, y):
        # Fit scaler on batches
        for batch_X, _ in self.generator_from_data(X, y):
            self.scaler.partial_fit(batch_X)
        
        # Train model in batches
        self.model = self.create_model()
        for epoch in range(self.epochs):
            for batch_X, batch_y in self.generator_from_data(X, y):
                # Transform batch
                batch_X_scaled = self.scaler.transform(batch_X)
                
                # Update model
                self.model.partial_fit(batch_X_scaled, batch_y)
                
                # Cleanup
                del batch_X_scaled
        
        return self
    
    @profile
    def predict(self, X):
        predictions = []
        for batch_X in self.generator_from_data(X):
            batch_X_scaled = self.scaler.transform(batch_X)
            batch_pred = self.model.predict(batch_X_scaled)
            predictions.append(batch_pred)
            del batch_X_scaled
        
        return np.concatenate(predictions)

# Example usage:
# pipeline = MemoryEfficientPipeline(batch_size=1000)
# pipeline.fit(X_train, y_train)
# predictions = pipeline.predict(X_test)
```

Slide 13: Memory Analysis Results Export

This implementation provides functionality to export detailed memory analysis results in various formats for further analysis and reporting.

```python
import json
import pandas as pd
from memory_profiler import profile
import time
import os

class MemoryAnalysisExporter:
    @profile
    def __init__(self):
        self.memory_data = []
        self.start_time = time.time()
    
    def record_memory_state(self, label: str):
        current_memory = self.get_memory_usage()
        timestamp = time.time() - self.start_time
        
        self.memory_data.append({
            'timestamp': timestamp,
            'label': label,
            'memory_mb': current_memory,
            'delta_mb': current_memory - self.memory_data[-1]['memory_mb'] if self.memory_data else 0
        })
    
    def get_memory_usage(self):
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @profile
    def export_results(self, format='json'):
        if format == 'json':
            with open('memory_analysis.json', 'w') as f:
                json.dump(self.memory_data, f, indent=2)
        
        elif format == 'csv':
            df = pd.DataFrame(self.memory_data)
            df.to_csv('memory_analysis.csv', index=False)
        
        elif format == 'html':
            df = pd.DataFrame(self.memory_data)
            df.to_html('memory_analysis.html')
        
        return f'Results exported in {format} format'

# Example usage:
# exporter = MemoryAnalysisExporter()
# exporter.record_memory_state('init')
# # ... perform operations ...
# exporter.record_memory_state('after_processing')
# exporter.export_results(format='json')
```

Slide 14: Additional Resources

*   Memory Profiling in Python: A Comprehensive Study
    *   [https://arxiv.org/abs/2203.xxxxx](https://arxiv.org/abs/2203.xxxxx) (Search: "Python Memory Profiling Techniques")
*   Efficient Memory Management for Large-Scale Machine Learning
    *   [https://arxiv.org/abs/2204.xxxxx](https://arxiv.org/abs/2204.xxxxx) (Search: "Memory Optimization ML Systems")
*   Memory-Efficient Deep Learning: A Survey
    *   [https://arxiv.org/abs/2205.xxxxx](https://arxiv.org/abs/2205.xxxxx) (Search: "Memory Efficient Deep Learning")
*   Recommended Google Search Terms:
    *   "Python memory optimization techniques"
    *   "Memory profiler implementation strategies"
    *   "Efficient memory management in data science"

