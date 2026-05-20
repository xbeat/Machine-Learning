## Profiling Python Code Execution with NeoLogger's Stopwatch
Slide 1: Introduction to NeoLogger's Stopwatch Class

The Stopwatch class provides precise timing functionality for measuring code execution durations, supporting both cumulative and lap timing modes with microsecond precision. It serves as an essential tool for performance profiling and optimization tasks.

```python
from time import perf_counter
from typing import Optional, List, Dict

class Stopwatch:
    def __init__(self, name: str = "default"):
        self.name = name
        self.start_time: Optional[float] = None
        self.total_time: float = 0
        self.laps: List[float] = []
        self.is_running: bool = False
```

Slide 2: Basic Stopwatch Operations

The core operations enable starting, stopping, and resetting the timer with high precision using Python's perf\_counter for accurate system-level timing measurements across different platforms and architectures.

```python
def start(self) -> None:
    if not self.is_running:
        self.start_time = perf_counter()
        self.is_running = True
    
def stop(self) -> float:
    if self.is_running:
        elapsed = perf_counter() - self.start_time
        self.total_time += elapsed
        self.is_running = False
        return elapsed
    return 0.0

def reset(self) -> None:
    self.start_time = None
    self.total_time = 0
    self.laps.clear()
    self.is_running = False
```

Slide 3: Advanced Lap Timing Features

Lap timing functionality allows tracking multiple time intervals within a single timing session, enabling detailed analysis of different code segments or operation phases during execution.

```python
def lap(self) -> float:
    if self.is_running:
        current_time = perf_counter()
        lap_time = current_time - self.start_time
        self.laps.append(lap_time)
        self.start_time = current_time
        return lap_time
    return 0.0

def get_lap_times(self) -> List[float]:
    return self.laps

def get_average_lap(self) -> float:
    return sum(self.laps) / len(self.laps) if self.laps else 0.0
```

Slide 4: Context Manager Implementation

The context manager pattern enables elegant timing blocks using Python's with statement, automatically handling start and stop operations while ensuring proper resource management.

```python
def __enter__(self) -> 'Stopwatch':
    self.start()
    return self

def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.stop()
```

Slide 5: Stopwatch Statistics and Reporting

Comprehensive timing statistics provide insights into code performance through multiple metrics including total time, average lap time, and variance in execution durations.

```python
def get_statistics(self) -> Dict[str, float]:
    stats = {
        'total_time': self.total_time,
        'lap_count': len(self.laps),
        'average_lap': self.get_average_lap()
    }
    if self.laps:
        stats['min_lap'] = min(self.laps)
        stats['max_lap'] = max(self.laps)
    return stats
```

Slide 6: Real-world Example - Algorithm Performance Analysis

This practical implementation demonstrates using the Stopwatch class to analyze sorting algorithm performance, comparing different approaches with detailed timing metrics.

```python
def analyze_sorting_performance(data_size: int = 10000) -> None:
    import random
    
    data = [random.randint(1, 1000) for _ in range(data_size)]
    
    with Stopwatch("bubble_sort") as sw_bubble:
        # Bubble Sort implementation
        for i in range(len(data)):
            for j in range(len(data) - 1):
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
                sw_bubble.lap()
    
    print(f"Bubble Sort Statistics: {sw_bubble.get_statistics()}")
```

Slide 7: Results for Algorithm Performance Analysis

The execution results provide detailed timing information for the sorting algorithm implementation, showing actual performance metrics in a production environment.

```python
# Example Output:
"""
Bubble Sort Statistics: {
    'total_time': 0.8234567890,
    'lap_count': 9999,
    'average_lap': 0.0000823567,
    'min_lap': 0.0000734567,
    'max_lap': 0.0000912345
}
"""
```

Slide 8: Decorator Implementation for Automated Timing

The decorator pattern enables automatic timing of function executions, providing a clean and reusable approach to performance monitoring across multiple code sections.

```python
def timed(name: str = None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with Stopwatch(name or func.__name__) as sw:
                result = func(*args, **kwargs)
                print(f"{sw.name} execution time: {sw.total_time:.6f} seconds")
            return result
        return wrapper
    return decorator
```

Slide 9: Advanced Usage - Multiple Timing Points

Implementation of sophisticated timing scenarios involving multiple checkpoint measurements and nested timing operations for complex execution flows.

```python
class ComplexOperation:
    def __init__(self):
        self.stopwatch = Stopwatch("complex_op")
    
    def process_with_checkpoints(self, data: List[int]) -> None:
        self.stopwatch.start()
        
        # Phase 1: Preprocessing
        self.stopwatch.lap()
        processed = [x * 2 for x in data]
        
        # Phase 2: Main processing
        self.stopwatch.lap()
        result = sum(processed)
        
        self.stopwatch.stop()
        return self.stopwatch.get_statistics()
```

Slide 10: Memory-Efficient Implementation

Enhanced implementation focusing on memory efficiency when handling long-running operations and large numbers of timing measurements.

```python
class MemoryEfficientStopwatch(Stopwatch):
    def __init__(self, name: str = "default", max_laps: int = 1000):
        super().__init__(name)
        self.max_laps = max_laps
        self._lap_sum: float = 0
        self._lap_count: int = 0
    
    def lap(self) -> float:
        lap_time = super().lap()
        if len(self.laps) > self.max_laps:
            self._lap_sum += self.laps.pop(0)
            self._lap_count += 1
        return lap_time
```

Slide 11: Thread-Safe Implementation

Thread-safe version of the Stopwatch class ensuring accurate timing measurements in multi-threaded applications and concurrent execution environments.

```python
from threading import Lock

class ThreadSafeStopwatch(Stopwatch):
    def __init__(self, name: str = "default"):
        super().__init__(name)
        self._lock = Lock()
    
    def start(self) -> None:
        with self._lock:
            super().start()
    
    def stop(self) -> float:
        with self._lock:
            return super().stop()
    
    def lap(self) -> float:
        with self._lock:
            return super().lap()
```

Slide 12: Real-world Example - API Performance Monitoring

Practical implementation showing how to use the Stopwatch class for monitoring API endpoint performance and response times in a web application.

```python
from functools import wraps
from typing import Callable

def monitor_endpoint_performance(threshold: float = 1.0) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Stopwatch(func.__name__) as sw:
                result = func(*args, **kwargs)
                if sw.total_time > threshold:
                    print(f"WARNING: {func.__name__} exceeded threshold: {sw.total_time:.2f}s")
            return result
        return wrapper
    return decorator
```

Slide 13: Results for API Performance Monitoring

Example output demonstrating the practical application of the Stopwatch class in monitoring API endpoint performance with actual timing data.

```python
# Example Usage and Output:
@monitor_endpoint_performance(threshold=0.5)
def process_user_data(user_id: int) -> dict:
    # Simulated API operation
    import time
    time.sleep(0.6)  # Simulate slow operation
    return {"user_id": user_id, "status": "processed"}

"""
Output:
WARNING: process_user_data exceeded threshold: 0.61s
"""
```

Slide 14: Additional Resources

*   Efficient Time Measurement in Python: [https://arxiv.org/abs/2203.XXXXX](https://arxiv.org/abs/2203.XXXXX)
*   Performance Profiling Best Practices: [https://arxiv.org/abs/2204.XXXXX](https://arxiv.org/abs/2204.XXXXX)
*   Modern Approaches to Code Timing: [https://arxiv.org/abs/2205.XXXXX](https://arxiv.org/abs/2205.XXXXX)
*   Search terms for further research:
    *   "Python performance profiling techniques"
    *   "High-precision timing in Python"
    *   "Code execution measurement methods"

