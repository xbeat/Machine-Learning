## Organizing Python Code with Modules and Imports
Slide 1: Basic Module Structure

Python modules serve as the fundamental building blocks for organizing code into reusable components. They encapsulate related functions, classes, and variables while providing a namespace that prevents naming conflicts between different parts of a larger application.

```python
# module_example.py
def calculate_statistics(data):
    """Calculate basic statistical measures."""
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return {'mean': mean, 'variance': variance}

class DataProcessor:
    """Process and transform data sequences."""
    def __init__(self, data):
        self.data = data
    
    def normalize(self):
        """Normalize data to range [0,1]."""
        min_val = min(self.data)
        max_val = max(self.data)
        return [(x - min_val) / (max_val - min_val) for x in self.data]

# Example usage
if __name__ == "__main__":
    sample_data = [1, 2, 3, 4, 5]
    stats = calculate_statistics(sample_data)
    processor = DataProcessor(sample_data)
    normalized = processor.normalize()
    print(f"Statistics: {stats}")
    print(f"Normalized: {normalized}")
```

Slide 2: Import Mechanisms

Understanding Python's import system is crucial for managing dependencies effectively. Python provides multiple ways to import modules, each serving different purposes and offering varying levels of namespace control.

```python
# Direct import of specific components
from math import sqrt, pi
result = sqrt(pi)

# Import with alias for namespace management
import numpy as np
array = np.array([1, 2, 3])

# Import all symbols (generally discouraged)
from statistics import *
mean_val = mean([1, 2, 3])

# Relative imports within packages
from .utils import helper_function
from ..core import main_function

# Conditional imports with error handling
try:
    import pandas as pd
except ImportError:
    print("Pandas is required. Please install it.")
    raise
```

Slide 3: Package Organization

Creating well-structured packages involves organizing related modules into directories with proper initialization and import mechanisms. The package structure determines how modules interact and how external code accesses package functionality.

```python
# project_root/
#   setup.py
#   mypackage/
#     __init__.py
#     core.py
#     utils/
#       __init__.py
#       helpers.py
#       config.py

# __init__.py
from .core import main_function
from .utils.helpers import utility_function

__all__ = ['main_function', 'utility_function']

# Version and metadata
__version__ = '1.0.0'
__author__ = 'Developer Name'

# Lazy loading implementation
def get_optional_component():
    from .utils.config import ConfigManager
    return ConfigManager()
```

Slide 4: Circular Import Prevention

Circular imports can create complex dependencies that lead to runtime errors. Understanding how to prevent and resolve circular imports is essential for maintaining a clean and efficient codebase.

```python
# Wrong approach (circular import)
# file1.py
from file2 import ClassB
class ClassA:
    def __init__(self):
        self.b = ClassB()

# file2.py
from file1 import ClassA
class ClassB:
    def __init__(self):
        self.a = ClassA()

# Correct approach
# file1.py
class ClassA:
    def __init__(self):
        from file2 import ClassB  # Import inside method
        self.b = ClassB()

# file2.py
class ClassB:
    def __init__(self):
        from file1 import ClassA  # Import inside method
        self.a = ClassA()
```

Slide 5: Dynamic Imports

Dynamic imports provide flexibility in loading modules at runtime based on specific conditions or requirements. This approach enables plugin systems and conditional functionality loading.

```python
import importlib
import sys

def load_module(module_name):
    """Dynamically load a module and return its namespace."""
    try:
        # Import module dynamically
        module = importlib.import_module(module_name)
        
        # Reload if already imported
        if module_name in sys.modules:
            module = importlib.reload(module)
        
        return module
    except ImportError as e:
        print(f"Failed to load module {module_name}: {e}")
        return None

# Example usage
data_processor = load_module('data_processor')
if data_processor:
    processor = data_processor.DataProcessor()
    result = processor.process_data([1, 2, 3])
```

Slide 6: Module-level Singletons

Module-level singletons provide a powerful way to maintain global state while ensuring thread-safety and consistent initialization. This pattern is commonly used for configuration management and shared resources.

```python
# config_manager.py
class ConfigManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.settings = {}
            self._initialized = True
    
    def set(self, key, value):
        self.settings[key] = value
    
    def get(self, key, default=None):
        return self.settings.get(key, default)

# Usage example
config = ConfigManager()
config.set('debug', True)

# In another module
from config_manager import ConfigManager
config = ConfigManager()  # Same instance
debug_mode = config.get('debug')  # Returns True
```

Slide 7: Lazy Module Importing

Implementing lazy importing mechanisms can significantly improve application startup time by deferring module imports until they're actually needed. This approach is particularly useful for large applications.

```python
class LazyImporter:
    def __init__(self, module_name):
        self.module_name = module_name
        self._module = None
    
    def __getattr__(self, name):
        if self._module is None:
            import importlib
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, name)

# Usage example
numpy = LazyImporter('numpy')
pandas = LazyImporter('pandas')

def process_data(data):
    if len(data) > 1000:
        # numpy only imported if this condition is met
        return numpy.mean(data)
    return sum(data) / len(data)

# Example with timing
import time
start = time.time()
result = process_data([1, 2, 3])  # Fast, no numpy import
print(f"Processing time: {time.time() - start:.4f} seconds")
```

Slide 8: Module Execution Context

Understanding module execution context is crucial for writing robust initialization code and managing module-level resources. The `__name__` variable provides important context about how a module is being used.

```python
# my_module.py
def initialize_resources():
    """Initialize module resources."""
    print("Initializing resources...")
    return {'status': 'initialized'}

def cleanup_resources():
    """Cleanup module resources."""
    print("Cleaning up resources...")

# Module initialization code
if __name__ == '__main__':
    # Code that runs only when module is executed directly
    resources = initialize_resources()
    try:
        print("Running main module code...")
        # Main module logic here
    finally:
        cleanup_resources()
else:
    # Code that runs when module is imported
    print(f"Module {__name__} imported")
    resources = None  # Defer initialization until explicitly requested
```

Slide 9: Advanced Package Distribution

Creating distributable packages requires careful consideration of dependencies, version compatibility, and installation requirements. This example demonstrates a comprehensive package structure with setup tools.

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="advanced_analytics",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'pandas>=1.0.0',
        'scipy>=1.4.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'black>=20.8b1',
        ],
        'viz': [
            'matplotlib>=3.3.0',
            'seaborn>=0.11.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'analyze=advanced_analytics.cli:main',
        ],
    },
    python_requires='>=3.8',
)

# Example package structure
"""
advanced_analytics/
    __init__.py
    cli.py
    core/
        __init__.py
        analysis.py
        visualization.py
    utils/
        __init__.py
        helpers.py
    tests/
        __init__.py
        test_analysis.py
"""
```

Slide 10: Module Performance Optimization

Understanding Python's module caching mechanism and implementing optimization techniques can significantly improve import performance and memory usage in large applications. This is crucial for production environments.

```python
# performance_optimized.py
import sys
import importlib
from functools import lru_cache

class OptimizedModuleLoader:
    _cache = {}
    
    @staticmethod
    @lru_cache(maxsize=128)
    def get_module(module_name):
        """Cache-optimized module loading."""
        if module_name not in OptimizedModuleLoader._cache:
            try:
                # Check sys.modules first
                if module_name in sys.modules:
                    return sys.modules[module_name]
                
                # Load and cache module
                module = importlib.import_module(module_name)
                OptimizedModuleLoader._cache[module_name] = module
                return module
            except ImportError as e:
                print(f"Failed to load {module_name}: {e}")
                return None
        return OptimizedModuleLoader._cache[module_name]

# Usage example
import time

def measure_loading_time(module_name, iterations=1000):
    start_time = time.time()
    for _ in range(iterations):
        module = OptimizedModuleLoader.get_module(module_name)
    end_time = time.time()
    return (end_time - start_time) / iterations

# Performance comparison
regular_time = measure_loading_time('json')
print(f"Average loading time: {regular_time:.6f} seconds")
```

Slide 11: Real-world Example - Data Processing Pipeline

This example demonstrates a complete data processing pipeline using modular design principles and proper import organization for a production environment.

```python
# data_pipeline/
#   __init__.py
#   processors/
#     __init__.py
#     cleaner.py
#     transformer.py
#     validator.py

# cleaner.py
import pandas as pd
import numpy as np
from typing import Dict, List

class DataCleaner:
    def __init__(self, config: Dict):
        self.config = config
        
    def remove_outliers(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Remove statistical outliers from specified columns."""
        clean_data = data.copy()
        for col in columns:
            q1 = clean_data[col].quantile(0.25)
            q3 = clean_data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            clean_data = clean_data[
                (clean_data[col] >= lower_bound) & 
                (clean_data[col] <= upper_bound)
            ]
        return clean_data
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to configuration."""
        for col, strategy in self.config['missing_values'].items():
            if strategy == 'mean':
                data[col].fillna(data[col].mean(), inplace=True)
            elif strategy == 'median':
                data[col].fillna(data[col].median(), inplace=True)
            elif strategy == 'mode':
                data[col].fillna(data[col].mode()[0], inplace=True)
        return data

# Example usage
if __name__ == "__main__":
    config = {
        'missing_values': {
            'age': 'median',
            'income': 'mean',
            'category': 'mode'
        }
    }
    
    # Create sample data
    data = pd.DataFrame({
        'age': [25, np.nan, 30, 45, np.nan],
        'income': [50000, 60000, np.nan, 75000, 80000],
        'category': ['A', 'B', np.nan, 'A', 'C']
    })
    
    cleaner = DataCleaner(config)
    cleaned_data = cleaner.handle_missing_values(data)
    print("Cleaned Data:\n", cleaned_data)
```

Slide 12: Real-world Example - Model Training System

A comprehensive example showing how to organize machine learning model training code using proper module organization and import strategies.

```python
# ml_system/
#   __init__.py
#   models/
#     __init__.py
#     base.py
#     neural_net.py
#   training/
#     __init__.py
#     trainer.py
#   utils/
#     __init__.py
#     metrics.py

# base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseModel(ABC):
    @abstractmethod
    def train(self, X: Any, y: Any, **kwargs) -> None:
        pass
    
    @abstractmethod
    def predict(self, X: Any) -> Any:
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass

# neural_net.py
import torch
import torch.nn as nn
from .base import BaseModel

class NeuralNetwork(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(config['input_size'], config['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['hidden_size'], config['output_size'])
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate', 0.001)
        )
        
    def train(self, X: torch.Tensor, y: torch.Tensor, 
              epochs: int = 100) -> None:
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            self.optimizer.step()
            
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(X)
            
    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        
    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
```

Slide 13: Results for Data Processing Pipeline

The following code block demonstrates the execution results and performance metrics for the data processing pipeline implemented in Slide 11.

```python
import pandas as pd
import numpy as np
from time import time

# Create larger sample dataset
np.random.seed(42)
n_samples = 10000
data = pd.DataFrame({
    'age': np.random.normal(40, 15, n_samples),
    'income': np.random.normal(60000, 20000, n_samples),
    'category': np.random.choice(['A', 'B', 'C'], n_samples)
})

# Introduce missing values
data.loc[np.random.choice(n_samples, 1000), 'age'] = np.nan
data.loc[np.random.choice(n_samples, 1000), 'income'] = np.nan
data.loc[np.random.choice(n_samples, 1000), 'category'] = np.nan

# Configuration
config = {
    'missing_values': {
        'age': 'median',
        'income': 'mean',
        'category': 'mode'
    }
}

# Performance measurement
start_time = time()
cleaner = DataCleaner(config)
cleaned_data = cleaner.handle_missing_values(data)
processing_time = time() - start_time

# Results
print(f"Processing Time: {processing_time:.4f} seconds")
print("\nMissing Values Before:")
print(data.isnull().sum())
print("\nMissing Values After:")
print(cleaned_data.isnull().sum())
print("\nStatistical Summary:")
print(cleaned_data.describe())
```

Slide 14: Results for Model Training System

This slide presents the execution results and performance metrics for the neural network training system implemented in Slide 12.

```python
import torch
import numpy as np
from time import time

# Generate synthetic dataset
np.random.seed(42)
X = torch.FloatTensor(np.random.randn(1000, 10))
y = torch.FloatTensor(np.random.randn(1000, 1))

# Model configuration
config = {
    'input_size': 10,
    'hidden_size': 20,
    'output_size': 1,
    'learning_rate': 0.001
}

# Training and evaluation
def evaluate_model():
    model = NeuralNetwork(config)
    
    # Training time measurement
    start_time = time()
    model.train(X, y, epochs=100)
    training_time = time() - start_time
    
    # Predictions and metrics
    y_pred = model.predict(X)
    mse = torch.mean((y - y_pred) ** 2).item()
    mae = torch.mean(torch.abs(y - y_pred)).item()
    
    return {
        'training_time': training_time,
        'mse': mse,
        'mae': mae
    }

# Execute evaluation
results = evaluate_model()
print(f"Training Time: {results['training_time']:.4f} seconds")
print(f"Mean Squared Error: {results['mse']:.6f}")
print(f"Mean Absolute Error: {results['mae']:.6f}")
```

Slide 15: Additional Resources

*   Paper: "Dynamic Import Patterns in Python" - Search on Google Scholar with keywords: "python import patterns optimization"
*   ArXiv Paper: "Optimizing Python Module Loading in Large Scale Applications"
    *   Search: arxiv.org/search with keywords: python module organization
*   Reference Documentation and Tutorials:
    *   Python Packaging User Guide: [https://packaging.python.org](https://packaging.python.org)
    *   Real Python - Python Modules and Packages: [https://realpython.com/python-modules-packages](https://realpython.com/python-modules-packages)
    *   Python Import System Documentation: [https://docs.python.org/3/reference/import.html](https://docs.python.org/3/reference/import.html)
*   Community Resources:
    *   Python Packaging Authority (PyPA): [https://www.pypa.io](https://www.pypa.io)
    *   Python Package Index (PyPI): [https://pypi.org](https://pypi.org)
    *   Hitchhiker's Guide to Python - Structuring Your Project: [https://docs.python-guide.org/writing/structure](https://docs.python-guide.org/writing/structure)

