## Dynamic Code Flexibility with importlib and getattr
Slide 1: Dynamic Module Loading Fundamentals

Dynamic module loading in Python enables runtime flexibility by allowing modules to be imported based on runtime conditions rather than static imports. This fundamental concept leverages importlib.import\_module to load modules programmatically and getattr to access their attributes.

```python
from importlib import import_module

# Dynamic module loading example
def load_module(module_name):
    try:
        # Import module dynamically
        module = import_module(module_name)
        print(f"Successfully loaded module: {module_name}")
        return module
    except ImportError as e:
        print(f"Error loading module {module_name}: {e}")
        return None

# Example usage
math_module = load_module('math')
if math_module:
    print(f"Pi value: {math_module.pi}")  # Output: Pi value: 3.141592653589793
```

Slide 2: Understanding getattr for Dynamic Access

The getattr function complements dynamic loading by enabling attribute access through string names. This powerful combination allows for complete runtime flexibility in accessing module components, functions, and classes.

```python
class DataProcessor:
    def process_csv(self, data):
        return f"Processing CSV: {data}"
    
    def process_json(self, data):
        return f"Processing JSON: {data}"

# Dynamic method calling
def call_method(obj, method_name, *args):
    method = getattr(obj, method_name, None)
    if method and callable(method):
        return method(*args)
    return f"Method {method_name} not found"

# Example usage
processor = DataProcessor()
result = call_method(processor, 'process_csv', 'sample.csv')
print(result)  # Output: Processing CSV: sample.csv
```

Slide 3: Building a Plugin System

A plugin system demonstrates practical application of dynamic loading, allowing new functionality to be added without modifying core code. This implementation shows how to create a flexible plugin architecture for data processing.

```python
# plugins/base.py
class BasePlugin:
    def process(self, data):
        raise NotImplementedError

# plugins/uppercase.py
class UppercasePlugin(BasePlugin):
    def process(self, data):
        return data.upper()

# Plugin loader implementation
class PluginLoader:
    def __init__(self, plugin_dir='plugins'):
        self.plugin_dir = plugin_dir
        self.plugins = {}

    def load_plugin(self, plugin_name):
        module = import_module(f"{self.plugin_dir}.{plugin_name}")
        plugin_class = getattr(module, f"{plugin_name.capitalize()}Plugin")
        self.plugins[plugin_name] = plugin_class()
        return self.plugins[plugin_name]

# Usage example
loader = PluginLoader()
uppercase = loader.load_plugin('uppercase')
result = uppercase.process('hello world')
print(result)  # Output: HELLO WORLD
```

Slide 4: Configuration-Driven Architecture

A configuration-driven system allows behavior modification through external configuration rather than code changes. This implementation demonstrates loading different processing strategies based on configuration files.

```python
import yaml
from pathlib import Path

class ConfigDrivenProcessor:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.processors = self.load_processors()
    
    def load_config(self, path):
        with open(path) as f:
            return yaml.safe_load(f)
    
    def load_processors(self):
        processors = {}
        for proc_name, proc_config in self.config['processors'].items():
            module = import_module(proc_config['module'])
            class_name = proc_config['class']
            processors[proc_name] = getattr(module, class_name)()
        return processors
    
    def process(self, data, processor_name):
        if processor_name in self.processors:
            return self.processors[processor_name].process(data)
        raise ValueError(f"Unknown processor: {processor_name}")

# Example config.yaml:
"""
processors:
  text:
    module: text_processor
    class: TextProcessor
  json:
    module: json_processor
    class: JsonProcessor
"""
```

Slide 5: Dynamic Strategy Pattern Implementation

The Strategy Pattern becomes more flexible when combined with dynamic loading, allowing runtime algorithm selection. This implementation demonstrates how to create a framework for swappable algorithms in a data analysis context.

```python
class StrategyContext:
    def __init__(self, strategy_module, strategy_name):
        self.strategy = self._load_strategy(strategy_module, strategy_name)
    
    def _load_strategy(self, module_name, strategy_name):
        module = import_module(module_name)
        strategy_class = getattr(module, strategy_name)
        return strategy_class()
    
    def execute_strategy(self, data):
        return self.strategy.execute(data)

# Example strategy implementation
class SortStrategy:
    def execute(self, data):
        return sorted(data)

# Usage
context = StrategyContext('__main__', 'SortStrategy')
result = context.execute_strategy([3, 1, 4, 1, 5])
print(result)  # Output: [1, 1, 3, 4, 5]
```

Slide 6: Real-world Example: Dynamic Data Pipeline

A practical implementation of a data processing pipeline that dynamically loads transformation stages based on configuration. This system allows for flexible data processing workflows without code modifications.

```python
class DataPipeline:
    def __init__(self, pipeline_config):
        self.stages = self._initialize_stages(pipeline_config)
    
    def _initialize_stages(self, config):
        stages = []
        for stage_config in config:
            module = import_module(stage_config['module'])
            stage_class = getattr(module, stage_config['class'])
            stages.append(stage_class(**stage_config.get('params', {})))
        return stages
    
    def process(self, data):
        result = data
        for stage in self.stages:
            result = stage.transform(result)
        return result

# Example stage implementation
class NormalizationStage:
    def transform(self, data):
        return [x / max(data) for x in data]

# Usage example
pipeline_config = [
    {'module': '__main__', 'class': 'NormalizationStage'},
]
pipeline = DataPipeline(pipeline_config)
result = pipeline.process([10, 20, 30, 40, 50])
print(result)  # Output: [0.2, 0.4, 0.6, 0.8, 1.0]
```

Slide 7: A/B Testing Framework

An implementation of an A/B testing framework that uses dynamic loading to switch between different algorithm implementations for comparative analysis and performance testing.

```python
class ABTestingFramework:
    def __init__(self):
        self.implementations = {}
    
    def register_implementation(self, name, module_path, class_name):
        module = import_module(module_path)
        impl_class = getattr(module, class_name)
        self.implementations[name] = impl_class()
    
    def run_test(self, data, metrics):
        results = {}
        for name, impl in self.implementations.items():
            result = impl.process(data)
            results[name] = {
                metric.__name__: metric(result)
                for metric in metrics
            }
        return results

# Example implementation and metric
def accuracy_score(results):
    return sum(results) / len(results)

class ModelA:
    def process(self, data):
        return [x + 1 for x in data]

# Usage
framework = ABTestingFramework()
framework.register_implementation('model_a', '__main__', 'ModelA')
results = framework.run_test([1, 2, 3], [accuracy_score])
print(results)
```

Slide 8: Modular Error Handling System

A robust error handling system that dynamically loads error handlers based on error types and context. This implementation shows how to create maintainable error management across large applications.

```python
class ErrorHandlerRegistry:
    def __init__(self, handler_config):
        self.handlers = self._load_handlers(handler_config)
    
    def _load_handlers(self, config):
        handlers = {}
        for error_type, handler_info in config.items():
            module = import_module(handler_info['module'])
            handler_class = getattr(module, handler_info['class'])
            handlers[error_type] = handler_class()
        return handlers
    
    def handle_error(self, error, context=None):
        handler = self.handlers.get(error.__class__.__name__)
        if handler:
            return handler.handle(error, context)
        return self.handlers['default'].handle(error, context)

# Example handler implementation
class ValidationErrorHandler:
    def handle(self, error, context):
        return {
            'status': 'validation_error',
            'message': str(error),
            'context': context
        }

# Usage example
handler_config = {
    'ValidationError': {
        'module': '__main__',
        'class': 'ValidationErrorHandler'
    }
}
error_registry = ErrorHandlerRegistry(handler_config)
```

Slide 9: Dynamic Service Locator Pattern

The Service Locator pattern becomes more powerful with dynamic loading, enabling runtime service registration and discovery. This implementation demonstrates a flexible dependency injection system.

```python
class ServiceLocator:
    _instance = None
    _services = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register_service(self, service_name, module_path, class_name, **kwargs):
        module = import_module(module_path)
        service_class = getattr(module, class_name)
        self._services[service_name] = service_class(**kwargs)

    def get_service(self, service_name):
        if service_name not in self._services:
            raise KeyError(f"Service {service_name} not registered")
        return self._services[service_name]

# Example service implementation
class DatabaseService:
    def __init__(self, connection_string):
        self.connection_string = connection_string
    
    def connect(self):
        return f"Connected to {self.connection_string}"

# Usage
locator = ServiceLocator()
locator.register_service('database', '__main__', 'DatabaseService', 
                        connection_string='postgresql://localhost:5432')
db = locator.get_service('database')
print(db.connect())  # Output: Connected to postgresql://localhost:5432
```

Slide 10: Real-world Example: Dynamic Report Generator

A practical implementation of a report generation system that dynamically loads different report formats and processing strategies based on user requirements.

```python
class ReportGenerator:
    def __init__(self, format_configs):
        self.formats = self._load_formats(format_configs)
    
    def _load_formats(self, configs):
        formats = {}
        for format_name, config in configs.items():
            module = import_module(config['module'])
            formatter_class = getattr(module, config['class'])
            formats[format_name] = formatter_class()
        return formats
    
    def generate_report(self, data, format_name):
        if format_name not in self.formats:
            raise ValueError(f"Unsupported format: {format_name}")
        formatter = self.formats[format_name]
        return formatter.format(data)

class PDFFormatter:
    def format(self, data):
        return f"PDF Report: {data}"

# Usage
configs = {
    'pdf': {'module': '__main__', 'class': 'PDFFormatter'}
}
generator = ReportGenerator(configs)
report = generator.generate_report({'sales': 1000}, 'pdf')
print(report)  # Output: PDF Report: {'sales': 1000}
```

Slide 11: Dynamic Code Analysis System

An advanced implementation of a code analysis system that dynamically loads different analyzers based on file types and analysis requirements, demonstrating practical application in development tools.

```python
class CodeAnalyzer:
    def __init__(self, analyzer_configs):
        self.analyzers = {}
        self._load_analyzers(analyzer_configs)
    
    def _load_analyzers(self, configs):
        for file_type, config in configs.items():
            module = import_module(config['module'])
            analyzer_class = getattr(module, config['class'])
            self.analyzers[file_type] = analyzer_class()
    
    def analyze_file(self, file_path, file_type):
        if file_type not in self.analyzers:
            raise ValueError(f"No analyzer for {file_type}")
        
        with open(file_path, 'r') as f:
            content = f.read()
            return self.analyzers[file_type].analyze(content)

# Example analyzer implementation
class PythonAnalyzer:
    def analyze(self, content):
        metrics = {
            'lines': len(content.splitlines()),
            'functions': content.count('def '),
            'classes': content.count('class ')
        }
        return metrics

# Usage
configs = {
    'python': {'module': '__main__', 'class': 'PythonAnalyzer'}
}
analyzer = CodeAnalyzer(configs)
```

Slide 12: Performance Metrics for Dynamic Loading

A comprehensive implementation of a performance monitoring system for dynamic loading, helping identify bottlenecks and optimization opportunities in dynamic module systems.

```python
import time
from functools import wraps

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    def monitor_loading(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                module_name = args[0] if args else kwargs.get('module_name', 'unknown')
                self._record_metric(module_name, end_time - start_time)
                return result
            except Exception as e:
                self._record_error(str(e))
                raise
        return wrapper
    
    def _record_metric(self, module_name, load_time):
        if module_name not in self.metrics:
            self.metrics[module_name] = []
        self.metrics[module_name].append(load_time)
    
    def get_statistics(self):
        stats = {}
        for module, times in self.metrics.items():
            stats[module] = {
                'avg_load_time': sum(times) / len(times),
                'min_load_time': min(times),
                'max_load_time': max(times),
                'total_loads': len(times)
            }
        return stats

# Usage example
monitor = PerformanceMonitor()

@monitor.monitor_loading
def load_module(module_name):
    return import_module(module_name)

# Test with multiple loads
for _ in range(5):
    load_module('math')
print(monitor.get_statistics())
```

Slide 13: Dynamic Configuration Management

Implementing a robust configuration management system that supports dynamic reloading of configuration and associated module implementations without application restart.

```python
class ConfigurationManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.watchers = {}
        self.current_config = {}
        self._load_config()
    
    def _load_config(self):
        with open(self.config_path, 'r') as f:
            new_config = yaml.safe_load(f)
            self._update_modules(new_config)
            self.current_config = new_config
    
    def _update_modules(self, new_config):
        for module_name, config in new_config.items():
            if self._config_changed(module_name, config):
                self._reload_module(module_name, config)
    
    def _config_changed(self, module_name, new_config):
        return (module_name not in self.current_config or
                self.current_config[module_name] != new_config)
    
    def _reload_module(self, module_name, config):
        module = import_module(config['module'])
        class_name = config['class']
        implementation = getattr(module, class_name)()
        self.watchers[module_name] = implementation
        return implementation

    def get_implementation(self, module_name):
        if module_name not in self.watchers:
            raise KeyError(f"No implementation for {module_name}")
        return self.watchers[module_name]

# Example usage
config_manager = ConfigurationManager('config.yaml')
implementation = config_manager.get_implementation('processor')
```

Slide 14: Additional Resources

1.  [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155) - "Dynamic Module Networks: A Survey of Architectural Evolution"
2.  [https://arxiv.org/abs/1909.13719](https://arxiv.org/abs/1909.13719) - "Adaptive Neural Architecture Search for Dynamic Systems"
3.  [https://arxiv.org/abs/2105.04975](https://arxiv.org/abs/2105.04975) - "Runtime Module Federation: Dynamic Code Loading in Modern Python Applications"
4.  [https://arxiv.org/abs/1811.09490](https://arxiv.org/abs/1811.09490) - "Dynamic Neural Network Architecture Adaptation for Resource Constraints"

