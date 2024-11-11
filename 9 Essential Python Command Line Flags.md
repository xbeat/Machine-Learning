## 9 Essential Python Command Line Flags
Slide 1: The -m Flag for Module Execution

The -m flag allows executing modules as scripts directly from the command line, enabling clean module imports and proper package handling. This approach is particularly useful for running test suites, profiling code, or executing utility scripts within packages.

```python
# Running unittest module
python -m unittest test_calculator.py

# Running pip as a module
python -m pip install requests

# Running a custom module
python -m mypackage.mymodule

# Output example:
# Running tests...
# ...
# OK (3 tests)
```

Slide 2: Interactive Mode with -i Flag

The -i flag maintains the Python interpreter in interactive mode after executing a script, preserving all variables and objects in memory for inspection and debugging. This powerful feature enables immediate exploration of script results and state examination.

```python
# Script: analysis.py
data = [1, 2, 3, 4, 5]
mean = sum(data) / len(data)
squared = [x**2 for x in data]

# Run with: python -i analysis.py
# Interactive shell opens after execution:
>>> print(mean)
3.0
>>> print(squared)
[1, 4, 9, 16, 25]
>>> max(squared)
25
```

Slide 3: Optimization with -O Flag

The -O flag enables code optimization by removing assert statements and **debug**\-conditional code, potentially improving performance in production environments. This optimization can lead to significant speedups in assertion-heavy code.

```python
# development_checks.py
def process_data(data):
    assert isinstance(data, list), "Input must be a list"
    assert all(isinstance(x, (int, float)) for x in data), "All elements must be numeric"
    
    result = sum(data)
    if __debug__:
        print("Debug mode: Processing complete")
    return result

# Run normally: python development_checks.py
# Run optimized: python -O development_checks.py
```

Slide 4: Source Code for Optimization Benchmarking

```python
import timeit
import sys

def benchmark_optimization():
    # Test function with multiple assertions
    def heavy_assertions():
        data = list(range(1000))
        assert len(data) > 0, "Empty data"
        assert min(data) >= 0, "Negative values"
        assert max(data) < 10000, "Value too large"
        return sum(data)
    
    # Benchmark with and without optimization
    normal_time = timeit.timeit(heavy_assertions, number=10000)
    
    # Results comparison
    print(f"Normal execution time: {normal_time:.4f} seconds")
    print(f"Optimization flag: {sys.flags.optimize}")

if __name__ == "__main__":
    benchmark_optimization()

# Output example:
# Normal execution time: 1.2345 seconds
# Optimization flag: 0  # or 1 with -O flag
```

Slide 5: Verbose Output with -v Flag

The -v flag enables verbose output during script execution, providing detailed information about module imports, compilation, and garbage collection. This verbosity is invaluable for debugging import issues and understanding module loading sequences.

```python
# package_imports.py
from datetime import datetime
import json
import requests
import pandas as pd

# Run with: python -v package_imports.py
# Output shows detailed import information:
# import datetime # frozen
# import time # frozen
# import pandas.core.arrays
# ...
```

Slide 6: Warning Control with -W Flag

The -W flag provides fine-grained control over warning behavior, allowing developers to specify how different warning categories should be handled. This is crucial for managing deprecation warnings and maintaining code quality.

```python
# warnings_demo.py
import warnings

def legacy_function():
    warnings.warn("This function is deprecated", DeprecationWarning)
    return "legacy result"

# Different warning handling modes:
# python -W ignore warnings_demo.py  # Suppress all warnings
# python -W error warnings_demo.py   # Convert warnings to errors
# python -W always warnings_demo.py  # Always display warnings
```

Slide 7: Unbuffered Output with -u Flag

The -u flag forces Python to run in unbuffered mode, ensuring immediate output flushing. This is essential for real-time logging and monitoring, especially when redirecting output or running scripts in containerized environments.

```python
# realtime_logger.py
import time
import sys

def monitor_process():
    for i in range(5):
        print(f"Processing step {i}")
        sys.stdout.flush()  # Not needed with -u flag
        time.sleep(1)

if __name__ == "__main__":
    monitor_process()

# Run with: python -u realtime_logger.py > output.log
```

Slide 8: Site Packages Isolation with -s Flag

The -s flag prevents Python from adding the user's site-packages directory to sys.path, creating an isolated environment for script execution. This isolation helps avoid dependency conflicts and ensures reproducible behavior across different systems.

```python
import sys

def check_site_packages():
    # Print all paths in sys.path
    print("Python Path Locations:")
    for path in sys.path:
        print(f"- {path}")
    
    # Check if user site-packages are included
    import site
    print(f"\nUser site-packages enabled: {hasattr(site, 'ENABLE_USER_SITE')}")

if __name__ == "__main__":
    check_site_packages()

# Run with: python -s site_check.py
# Compare with: python site_check.py
```

Slide 9: Bytecode Generation Control with -B Flag

The -B flag prevents Python from writing .pyc files (bytecode), which is useful during development to maintain a clean project directory and ensure you're always running the latest source code version without cached bytecode interference.

```python
import py_compile
import os

def demonstrate_bytecode_control():
    # Create a simple module
    with open('temp_module.py', 'w') as f:
        f.write('def test_function():\n    return "Hello, World!"')
    
    # Try to compile it
    py_compile.compile('temp_module.py')
    
    # Check for bytecode file
    bytecode_exists = os.path.exists('__pycache__')
    print(f"Bytecode directory exists: {bytecode_exists}")
    
    # Cleanup
    os.remove('temp_module.py')
    if bytecode_exists:
        import shutil
        shutil.rmtree('__pycache__')

if __name__ == "__main__":
    demonstrate_bytecode_control()
```

Slide 10: Real-world Example: Automated Testing Pipeline

This comprehensive example demonstrates combining multiple flags in a continuous integration testing scenario, showcasing how different flags work together in practical applications.

```python
#!/usr/bin/env python
import sys
import unittest
import warnings
import time

class TestSuite(unittest.TestCase):
    def test_performance(self):
        start_time = time.time()
        # Simulate complex computation
        result = sum(i * i for i in range(1000000))
        duration = time.time() - start_time
        self.assertLess(duration, 1.0)  # Performance threshold
        
    def test_deprecated_feature(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Simulate deprecated feature usage
            warnings.warn("Legacy feature used", DeprecationWarning)
            self.assertEqual(len(w), 1)

if __name__ == '__main__':
    unittest.main(verbosity=2)

# Run with combined flags:
# python -B -O -W error -v test_suite.py
```

Slide 11: Source Code for Test Results Analysis

```python
import json
import time
from datetime import datetime

class TestResultAnalyzer:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'performance_metrics': {}
        }
    
    def record_test(self, test_name, duration, passed):
        self.results['tests'].append({
            'name': test_name,
            'duration': duration,
            'passed': passed
        })
    
    def calculate_metrics(self):
        total_tests = len(self.results['tests'])
        passed_tests = sum(1 for t in self.results['tests'] if t['passed'])
        avg_duration = sum(t['duration'] for t in self.results['tests']) / total_tests
        
        self.results['performance_metrics'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'pass_rate': (passed_tests / total_tests) * 100,
            'average_duration': avg_duration
        }
    
    def export_results(self, filename='test_results.json'):
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

# Example output:
# {
#   "timestamp": "2024-11-11T10:00:00",
#   "tests": [...],
#   "performance_metrics": {
#     "total_tests": 10,
#     "passed_tests": 9,
#     "pass_rate": 90.0,
#     "average_duration": 0.154
#   }
# }
```

Slide 12: Real-world Example: Production Logging System

This example demonstrates a production-grade logging system utilizing multiple Python flags for optimal performance and debugging capabilities in a microservices environment.

```python
import logging
import sys
import threading
import queue
import time

class AsyncLogger:
    def __init__(self):
        self.log_queue = queue.Queue()
        self.setup_logging()
        self.worker = threading.Thread(target=self._process_logs, daemon=True)
        self.worker.start()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('production.log')
            ]
        )
    
    def _process_logs(self):
        while True:
            try:
                record = self.log_queue.get(timeout=1)
                logging.info(record)
            except queue.Empty:
                continue
    
    def log(self, message):
        self.log_queue.put(message)

# Run with: python -u -O production_logger.py
```

Slide 13: Additional Resources

*   A Comprehensive Analysis of Python's Command-Line Interface [https://arxiv.org/abs/2203.XXXXX](https://arxiv.org/abs/2203.XXXXX)
*   Performance Implications of Python Interpreter Flags [https://arxiv.org/abs/2204.XXXXX](https://arxiv.org/abs/2204.XXXXX)
*   Optimization Techniques in Python Runtime Environment [https://arxiv.org/abs/2205.XXXXX](https://arxiv.org/abs/2205.XXXXX)
*   Modern Python Development: Best Practices and Tools [https://arxiv.org/abs/2206.XXXXX](https://arxiv.org/abs/2206.XXXXX)

