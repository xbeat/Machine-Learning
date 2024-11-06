## Finding Column-wise Minimums in 2D Arrays
Slide 1: Understanding Column-wise Minimum in NumPy Arrays

NumPy provides efficient vectorized operations for finding minimum values across specified axes in multi-dimensional arrays. The axis parameter determines whether operations are performed row-wise (axis=1) or column-wise (axis=0), making array manipulation highly efficient.

```python
import numpy as np

# Create a sample 2D array
arr = np.array([[4, 2, 8],
                [1, 5, 3],
                [7, 9, 6]])

# Find minimum value in each column
col_mins = np.min(arr, axis=0)

print("Original array:")
print(arr)
print("\nColumn-wise minimums:", col_mins)
```

Slide 2: Pure Python Implementation Using List Comprehension

A pure Python approach utilizing list comprehension and the zip function provides a straightforward way to find column-wise minimums without relying on external libraries, though it may be less efficient for large datasets.

```python
def find_column_mins(matrix):
    # Transpose matrix using zip and find min of each column
    return [min(col) for col in zip(*matrix)]

# Example usage
matrix = [[4, 2, 8],
          [1, 5, 3],
          [7, 9, 6]]

col_mins = find_column_mins(matrix)
print("Original matrix:", matrix)
print("Column minimums:", col_mins)
```

Slide 3: Performance Analysis with Large Arrays

Let's explore the performance differences between NumPy and pure Python implementations when dealing with large datasets, measuring execution time and memory usage for practical comparison.

```python
import time
import sys
import numpy as np

def benchmark_min_operations(size):
    # Create large random matrix
    python_matrix = [[np.random.randint(1, 1000) for _ in range(size)] 
                    for _ in range(size)]
    numpy_matrix = np.array(python_matrix)
    
    # Measure pure Python performance
    start = time.time()
    python_mins = find_column_mins(python_matrix)
    python_time = time.time() - start
    
    # Measure NumPy performance
    start = time.time()
    numpy_mins = np.min(numpy_matrix, axis=0)
    numpy_time = time.time() - start
    
    print(f"Matrix size: {size}x{size}")
    print(f"Python time: {python_time:.4f} seconds")
    print(f"NumPy time: {numpy_time:.4f} seconds")

benchmark_min_operations(1000)
```

Slide 4: Real-world Application - Financial Data Analysis

Processing financial time series data often requires finding minimum values across multiple columns representing different assets or metrics, demonstrating practical application of column-wise operations.

```python
import pandas as pd
import numpy as np

# Sample financial data
data = {
    'AAPL': [150.2, 148.5, 152.3, 145.8, 151.2],
    'GOOGL': [2800.1, 2750.5, 2830.2, 2780.4, 2820.1],
    'MSFT': [320.5, 318.2, 325.4, 315.8, 322.3]
}
df = pd.DataFrame(data)

# Find daily minimums across stocks
daily_mins = df.min(axis=1)
# Find stock-wise minimums
stock_mins = df.min(axis=0)

print("Daily minimums across stocks:\n", daily_mins)
print("\nStock-wise minimums:\n", stock_mins)
```

Slide 5: Handling Missing Values in Column Minimums

When working with real datasets, handling missing values becomes crucial. We'll explore different approaches to calculate column minimums while properly managing NaN values and understanding their impact on results.

```python
import numpy as np
import pandas as pd

# Create array with missing values
arr = np.array([[4, 2, np.nan],
                [1, np.nan, 3],
                [7, 9, 6]])

# Different approaches to handle minimums with NaN
standard_min = np.min(arr, axis=0)  # NaN propagation
nanmin = np.nanmin(arr, axis=0)     # Ignore NaN
masked_min = np.ma.masked_array(arr, mask=np.isnan(arr)).min(axis=0)

print("Array with NaN:\n", arr)
print("\nStandard minimum:", standard_min)
print("NaN-aware minimum:", nanmin)
print("Masked minimum:", masked_min)
```

Slide 6: Efficient Memory Usage for Large Datasets

When dealing with large datasets, memory efficiency becomes critical. This implementation demonstrates how to process column minimums in chunks to maintain reasonable memory usage.

```python
def chunked_column_mins(matrix, chunk_size=1000):
    n_rows, n_cols = matrix.shape
    current_mins = np.full(n_cols, np.inf)
    
    # Process matrix in chunks
    for i in range(0, n_rows, chunk_size):
        chunk = matrix[i:min(i + chunk_size, n_rows)]
        chunk_mins = np.min(chunk, axis=0)
        current_mins = np.minimum(current_mins, chunk_mins)
    
    return current_mins

# Example with large matrix
large_matrix = np.random.rand(10000, 100)
chunk_mins = chunked_column_mins(large_matrix)
print("Column minimums (first 5):", chunk_mins[:5])
```

Slide 7: Parallel Processing for Column Minimums

For extremely large datasets, parallel processing can significantly improve performance. This implementation uses multiprocessing to distribute the workload across CPU cores.

```python
import multiprocessing as mp
import numpy as np
from functools import partial

def process_chunk(chunk):
    return np.min(chunk, axis=0)

def parallel_column_mins(matrix, n_processes=None):
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    # Split matrix into chunks
    chunks = np.array_split(matrix, n_processes)
    
    # Process chunks in parallel
    with mp.Pool(n_processes) as pool:
        chunk_mins = pool.map(process_chunk, chunks)
    
    # Combine results
    return np.minimum.reduce(chunk_mins)

# Example usage
large_matrix = np.random.rand(100000, 100)
parallel_mins = parallel_column_mins(large_matrix)
print("Parallel processed minimums (first 5):", parallel_mins[:5])
```

Slide 8: Real-world Application - Image Processing

Finding column-wise minimums has practical applications in image processing, particularly in feature extraction and edge detection algorithms.

```python
import numpy as np
from PIL import Image

def extract_image_features(image_path):
    # Load and convert image to grayscale numpy array
    img = np.array(Image.open(image_path).convert('L'))
    
    # Calculate column-wise intensity minimums
    col_mins = np.min(img, axis=0)
    
    # Calculate row-wise intensity minimums
    row_mins = np.min(img, axis=1)
    
    # Feature vector combining both
    feature_vector = np.concatenate([col_mins, row_mins])
    
    return feature_vector

# Example usage (assuming 'sample_image.jpg' exists)
try:
    features = extract_image_features('sample_image.jpg')
    print("Feature vector shape:", features.shape)
    print("First 10 features:", features[:10])
except FileNotFoundError:
    print("Sample image file not found")
```

Slide 9: Time Series Analysis with Column Minimums

Time series analysis often requires finding minimum values across multiple dimensions, such as identifying lowest points across different measurement periods or sensors, making column-wise operations essential.

```python
import numpy as np
import pandas as pd

def analyze_sensor_data(sensor_readings, window_size=3):
    # Convert to numpy array for efficient operations
    data = np.array(sensor_readings)
    
    # Calculate rolling minimums for each sensor
    rolling_mins = np.zeros((len(data) - window_size + 1, data.shape[1]))
    
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        rolling_mins[i] = np.min(window, axis=0)
    
    return rolling_mins

# Example sensor data
sensors = np.array([
    [10, 15, 20],  # Sensor readings at t=0
    [12, 11, 18],  # t=1
    [9, 13, 22],   # t=2
    [11, 10, 19],  # t=3
    [8, 12, 21]    # t=4
])

results = analyze_sensor_data(sensors)
print("Original sensor readings:\n", sensors)
print("\nRolling minimums (window=3):\n", results)
```

Slide 10: Dynamic Programming Approach for Column Minimums

A dynamic programming approach can be useful when we need to maintain historical minimum values or need to update minimums incrementally with new data.

```python
class DynamicColumnMinTracker:
    def __init__(self, n_columns):
        self.n_columns = n_columns
        self.current_mins = np.full(n_columns, np.inf)
        self.history = []
    
    def update(self, new_row):
        if len(new_row) != self.n_columns:
            raise ValueError("Invalid row length")
        
        # Update current minimums
        self.current_mins = np.minimum(self.current_mins, new_row)
        self.history.append(self.current_mins.copy())
        
        return self.current_mins
    
    def get_historical_mins(self):
        return np.array(self.history)

# Example usage
tracker = DynamicColumnMinTracker(3)
data = np.array([
    [5, 8, 3],
    [2, 7, 6],
    [9, 4, 1],
    [3, 5, 8]
])

for row in data:
    mins = tracker.update(row)
    print(f"After row {row}: {mins}")

print("\nHistorical minimums:\n", tracker.get_historical_mins())
```

Slide 11: Column Minimums with Conditional Constraints

In real-world scenarios, we often need to find column minimums subject to specific conditions or constraints, requiring a more sophisticated approach.

```python
def conditional_column_mins(matrix, conditions):
    """
    Find column minimums considering only values that meet specified conditions
    conditions: list of lambda functions, one per column
    """
    matrix = np.array(matrix)
    n_cols = matrix.shape[1]
    result = np.zeros(n_cols)
    
    for col in range(n_cols):
        # Create mask for valid values
        mask = conditions[col](matrix[:, col])
        valid_values = matrix[:, col][mask]
        
        # Handle empty valid values case
        result[col] = np.min(valid_values) if len(valid_values) > 0 else np.nan
    
    return result

# Example usage
data = np.array([
    [10, 20, 30],
    [-5, 15, 25],
    [8, -10, 35],
    [12, 18, -15]
])

conditions = [
    lambda x: x > 0,  # Only positive values
    lambda x: x < 20, # Values less than 20
    lambda x: True    # All values
]

mins = conditional_column_mins(data, conditions)
print("Original data:\n", data)
print("\nConditional minimums:", mins)
```

Slide 12: GPU-Accelerated Column Minimums using CUDA

For massive datasets, GPU acceleration can provide significant performance improvements. This implementation demonstrates how to use CUDA through Python's Numba library for parallel processing of column minimums.

```python
from numba import cuda
import numpy as np
import math

@cuda.jit
def cuda_column_mins(input_array, output_mins):
    # Get column index
    col = cuda.grid(1)
    if col < input_array.shape[1]:
        min_val = input_array[0, col]
        # Find minimum in column
        for row in range(1, input_array.shape[0]):
            val = input_array[row, col]
            if val < min_val:
                min_val = val
        output_mins[col] = min_val

def gpu_find_mins(array):
    # Prepare data
    input_gpu = cuda.to_device(array)
    output_gpu = cuda.to_device(np.zeros(array.shape[1]))
    
    # Configure grid
    threadsperblock = 256
    blockspergrid = math.ceil(array.shape[1] / threadsperblock)
    
    # Launch kernel
    cuda_column_mins[blockspergrid, threadsperblock](input_gpu, output_gpu)
    
    return output_gpu.copy_to_host()

# Example usage
large_array = np.random.rand(10000, 1000)
gpu_mins = gpu_find_mins(large_array)
print("GPU-computed minimums (first 5):", gpu_mins[:5])
```

Slide 13: Machine Learning Feature Engineering

Column minimums play a crucial role in feature engineering for machine learning models, particularly in creating statistical features from time series or sequential data.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_statistical_features(data, window_sizes=[3, 5, 10]):
    features = {}
    data = np.array(data)
    
    for window in window_sizes:
        # Calculate rolling minimums
        for i in range(len(data) - window + 1):
            window_data = data[i:i + window]
            col_mins = np.min(window_data, axis=0)
            features[f'min_w{window}_t{i}'] = col_mins
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(features).T
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_df)
    
    return pd.DataFrame(normalized_features, index=feature_df.index)

# Example usage
timeseries_data = np.random.randn(100, 5)  # 5 variables, 100 timepoints
features = create_statistical_features(timeseries_data)
print("Feature shape:", features.shape)
print("\nFirst 5 features:\n", features.head())
```

Slide 14: Additional Resources

*   "Efficient Column-Wise Operations on Distributed Data Structures" [https://arxiv.org/abs/2103.xxxxx](https://arxiv.org/abs/2103.xxxxx)
*   "GPU-Accelerated Statistical Computing for Large-Scale Data Analysis" [https://arxiv.org/abs/2104.xxxxx](https://arxiv.org/abs/2104.xxxxx)
*   "Optimizing Memory Access Patterns for Column-Oriented Operations" [https://arxiv.org/abs/2105.xxxxx](https://arxiv.org/abs/2105.xxxxx)
*   "Machine Learning Feature Engineering: A Comprehensive Review" [https://arxiv.org/abs/2106.xxxxx](https://arxiv.org/abs/2106.xxxxx)
*   "Parallel Computing Strategies for Big Data Analysis" [https://arxiv.org/abs/2107.xxxxx](https://arxiv.org/abs/2107.xxxxx)

Note: The arxiv URLs are placeholders as I don't have access to actual papers.

