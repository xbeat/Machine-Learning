## Optimizing NumPy Performance with Thread Configuration
Slide 1: Understanding NumPy Thread Configuration

NumPy's performance heavily depends on its underlying BLAS (Basic Linear Algebra Subprograms) implementation and thread configuration. The three main environment variables that control threading behavior are MKL\_NUM\_THREADS, OPENBLAS\_NUM\_THREADS, and OMP\_NUM\_THREADS, each corresponding to different BLAS backends.

```python
import os
import numpy as np

# Set environment variables for thread control
os.environ['MKL_NUM_THREADS'] = '4'  # Intel MKL
os.environ['OPENBLAS_NUM_THREADS'] = '4'  # OpenBLAS
os.environ['OMP_NUM_THREADS'] = '4'  # OpenMP

# Check NumPy configuration
np.show_config()  # Shows BLAS implementation details
```

Slide 2: Thread Configuration Impact Analysis

Understanding the impact of thread configuration requires benchmarking matrix operations under different settings. This code demonstrates how to measure performance variations with different thread counts using matrix multiplication as an example.

```python
import time
import numpy as np
from contextlib import contextmanager

@contextmanager
def thread_config(num_threads):
    # Store original settings
    original_mkl = os.environ.get('MKL_NUM_THREADS')
    original_openblas = os.environ.get('OPENBLAS_NUM_THREADS')
    original_omp = os.environ.get('OMP_NUM_THREADS')
    
    # Set new thread count
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    
    try:
        yield
    finally:
        # Restore original settings
        if original_mkl:
            os.environ['MKL_NUM_THREADS'] = original_mkl
        if original_openblas:
            os.environ['OPENBLAS_NUM_THREADS'] = original_openblas
        if original_omp:
            os.environ['OMP_NUM_THREADS'] = original_omp
```

Slide 3: Performance Benchmarking Function

A comprehensive benchmarking function helps determine optimal thread configurations for specific matrix operations. This implementation measures execution time across different thread counts and matrix sizes.

```python
def benchmark_matrix_operation(sizes, thread_counts):
    results = {}
    
    for size in sizes:
        results[size] = {}
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        
        for threads in thread_counts:
            with thread_config(threads):
                start_time = time.perf_counter()
                C = np.dot(A, B)  # Matrix multiplication
                duration = time.perf_counter() - start_time
                results[size][threads] = duration
                
    return results

# Example usage
sizes = [1000, 2000, 4000]
thread_counts = [1, 2, 4, 8]
results = benchmark_matrix_operation(sizes, thread_counts)
```

Slide 4: Visualizing Thread Performance

Creating visualizations helps understand the relationship between thread count and performance for different matrix sizes. This implementation uses matplotlib to generate comparative performance plots.

```python
import matplotlib.pyplot as plt

def plot_thread_performance(results):
    plt.figure(figsize=(12, 6))
    
    for size in results:
        threads = list(results[size].keys())
        times = list(results[size].values())
        plt.plot(threads, times, marker='o', label=f'Matrix Size: {size}x{size}')
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Matrix Multiplication Performance vs Thread Count')
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate performance visualization
plot_thread_performance(results)
```

Slide 5: Thread-Aware Context Manager

Implementing a context manager for temporary thread configuration changes allows for safe experimentation with different thread settings without affecting the global environment.

```python
class ThreadConfiguration:
    def __init__(self, num_threads):
        self.num_threads = str(num_threads)
        self.original_config = {}
    
    def __enter__(self):
        # Save current configuration
        thread_vars = ['MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']
        for var in thread_vars:
            self.original_config[var] = os.environ.get(var)
            os.environ[var] = self.num_threads
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original configuration
        for var, value in self.original_config.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
```

Slide 6: Real-world Example: Image Processing Pipeline

A practical demonstration of thread optimization in image processing tasks using NumPy. This implementation shows how different thread configurations affect the performance of common image operations.

```python
import numpy as np
from PIL import Image
import time

def process_image_batch(images, thread_counts):
    results = {}
    
    for threads in thread_counts:
        with ThreadConfiguration(threads):
            start_time = time.perf_counter()
            
            # Simulate batch processing
            processed = []
            for img in images:
                # Convert to numpy array
                img_array = np.array(img)
                
                # Apply various transformations
                filtered = np.fft.fft2(img_array)  # FFT
                inverse = np.fft.ifft2(filtered)   # Inverse FFT
                processed.append(inverse.real)
            
            duration = time.perf_counter() - start_time
            results[threads] = duration
    
    return results
```

Slide 7: Source Code for Image Processing Analysis

```python
def analyze_image_processing_performance():
    # Generate synthetic images
    images = []
    for _ in range(10):
        img_array = np.random.rand(1024, 1024)
        images.append(img_array)
    
    # Test different thread configurations
    thread_counts = [1, 2, 4, 8, 16]
    results = process_image_batch(images, thread_counts)
    
    # Print results
    print("\nImage Processing Performance Results:")
    print("-" * 40)
    for threads, duration in results.items():
        print(f"Threads: {threads:2d} | Time: {duration:.4f} seconds")
    
    return results

# Example usage
performance_results = analyze_image_processing_performance()
```

Slide 8: Memory-Bound vs CPU-Bound Operations

Understanding the distinction between memory-bound and CPU-bound operations is crucial for optimal thread configuration. Different types of operations benefit from different thread counts based on their resource requirements.

```python
def compare_operation_types(size=5000):
    operations = {
        'memory_bound': lambda: np.sum(np.random.rand(size, size)),
        'cpu_bound': lambda: np.linalg.svd(np.random.rand(size, size))
    }
    
    thread_counts = [1, 2, 4, 8]
    results = {op: {} for op in operations}
    
    for op_name, operation in operations.items():
        for threads in thread_counts:
            with ThreadConfiguration(threads):
                start_time = time.perf_counter()
                operation()
                duration = time.perf_counter() - start_time
                results[op_name][threads] = duration
    
    return results
```

Slide 9: Automatic Thread Optimization

Implementing an automatic thread optimizer that determines the optimal thread count for specific operations through iterative testing and performance measurement.

```python
class ThreadOptimizer:
    def __init__(self, max_threads=16):
        self.max_threads = max_threads
        self.thread_cache = {}
    
    def find_optimal_threads(self, operation, *args, **kwargs):
        # Create a unique key for this operation
        op_key = f"{operation.__name__}_{hash(str(args))}"
        
        if op_key in self.thread_cache:
            return self.thread_cache[op_key]
        
        best_time = float('inf')
        optimal_threads = 1
        
        for threads in range(1, self.max_threads + 1):
            with ThreadConfiguration(threads):
                times = []
                # Run multiple trials
                for _ in range(3):
                    start = time.perf_counter()
                    operation(*args, **kwargs)
                    times.append(time.perf_counter() - start)
                
                avg_time = np.mean(times)
                if avg_time < best_time:
                    best_time = avg_time
                    optimal_threads = threads
        
        self.thread_cache[op_key] = optimal_threads
        return optimal_threads
```

Slide 10: Dynamic Thread Scaling Implementation

A system for dynamically adjusting thread counts based on system load and available resources, ensuring optimal performance across varying workload conditions.

```python
import psutil
import multiprocessing

class DynamicThreadScaler:
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
    
    def get_optimal_threads(self, operation_type='default'):
        # Get current CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent
        
        # Base calculation of available threads
        available_threads = max(1, self.cpu_count - int(cpu_usage / 100 * self.cpu_count))
        
        # Adjust based on operation type
        if operation_type == 'memory_bound':
            # Reduce threads if memory usage is high
            if memory_usage > 80:
                available_threads = max(1, available_threads // 2)
        elif operation_type == 'cpu_bound':
            # Ensure minimum threads for CPU-bound operations
            available_threads = max(2, available_threads)
        
        return available_threads
```

Slide 11: Performance Monitoring and Logging

Implementing a comprehensive monitoring system to track thread performance over time helps in making informed decisions about thread configuration adjustments for different types of operations.

```python
import logging
from datetime import datetime
import json

class ThreadPerformanceMonitor:
    def __init__(self, log_file='thread_performance.log'):
        self.log_file = log_file
        self.setup_logging()
        self.performance_history = {}
    
    def setup_logging(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
    
    def record_performance(self, operation_name, matrix_size, thread_count, execution_time):
        timestamp = datetime.now().isoformat()
        entry = {
            'timestamp': timestamp,
            'operation': operation_name,
            'matrix_size': matrix_size,
            'threads': thread_count,
            'execution_time': execution_time
        }
        
        # Log the entry
        logging.info(json.dumps(entry))
        
        # Update performance history
        key = f"{operation_name}_{matrix_size}"
        if key not in self.performance_history:
            self.performance_history[key] = []
        self.performance_history[key].append(entry)
```

Slide 12: Advanced Thread Analysis Toolkit

A comprehensive toolkit for analyzing thread performance across different types of NumPy operations, providing detailed insights into optimal thread configurations.

```python
class ThreadAnalysisToolkit:
    def __init__(self):
        self.monitor = ThreadPerformanceMonitor()
        self.optimizer = ThreadOptimizer()
        self.scaler = DynamicThreadScaler()
    
    def analyze_operation(self, operation, sizes, thread_range):
        results = {}
        
        for size in sizes:
            results[size] = {}
            for threads in thread_range:
                with ThreadConfiguration(threads):
                    # Warm-up run
                    operation(size)
                    
                    # Actual measurement
                    start_time = time.perf_counter()
                    operation(size)
                    duration = time.perf_counter() - start_time
                    
                    # Record performance
                    self.monitor.record_performance(
                        operation.__name__,
                        size,
                        threads,
                        duration
                    )
                    results[size][threads] = duration
        
        return results
```

Slide 13: Results Analysis and Visualization

```python
def visualize_thread_analysis(results):
    import seaborn as sns
    
    # Prepare data for plotting
    plot_data = []
    for size in results:
        for threads, time in results[size].items():
            plot_data.append({
                'Matrix Size': size,
                'Threads': threads,
                'Execution Time': time
            })
    
    # Create heatmap
    df = pd.DataFrame(plot_data)
    pivot_table = df.pivot('Matrix Size', 'Threads', 'Execution Time')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Thread Performance Analysis')
    plt.xlabel('Number of Threads')
    plt.ylabel('Matrix Size')
    plt.show()
```

Slide 14: Additional Resources

*   Optimizing NumPy Array Operations: Search for "Optimizing NumPy: Tips and Tricks for Numerical Computing"
*   Performance Analysis of Multi-Threaded NumPy Operations: [http://www.google.com/search?q=numpy+multithreading+performance+analysis](http://www.google.com/search?q=numpy+multithreading+performance+analysis)
*   Thread Optimization Techniques in Scientific Computing:
    *   [https://scicomp.stackexchange.com/questions/tagged/numpy+parallel-computing](https://scicomp.stackexchange.com/questions/tagged/numpy+parallel-computing)
    *   [https://numpy.org/doc/stable/reference/routines.linalg.html](https://numpy.org/doc/stable/reference/routines.linalg.html)
    *   [https://scipy-lectures.org/advanced/optimizing/](https://scipy-lectures.org/advanced/optimizing/)
*   Recommended search terms for further research:
    *   "NumPy BLAS optimization techniques"
    *   "Multi-threaded linear algebra performance"
    *   "OpenBLAS vs MKL threading comparison"

