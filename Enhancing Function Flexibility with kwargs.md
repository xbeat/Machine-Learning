## Enhancing Function Flexibility with kwargs
Slide 1: Understanding \*\*kwargs Basics

The \*\*kwargs syntax allows functions to accept arbitrary keyword arguments as a dictionary, providing flexibility in handling varying numbers of parameters. This pattern enables dynamic parameter passing without modifying the function signature, making code more maintainable and adaptable.

```python
def create_profile(**kwargs):
    # Initialize empty profile dictionary
    profile = {}
    
    # Add all provided keyword arguments to profile
    for key, value in kwargs.items():
        profile[key] = value
        
    return profile

# Example usage with different numbers of arguments
basic_profile = create_profile(name="John", age=30)
detailed_profile = create_profile(name="Alice", age=25, 
                                occupation="Engineer",
                                skills=["Python", "AI"])

print("Basic Profile:", basic_profile)
print("Detailed Profile:", detailed_profile)

# Output:
# Basic Profile: {'name': 'John', 'age': 30}
# Detailed Profile: {'name': 'Alice', 'age': 25, 
#                    'occupation': 'Engineer',
#                    'skills': ['Python', 'AI']}
```

Slide 2: Dynamic Object Creation with \*\*kwargs

When creating objects that may require different attributes based on context or user input, \*\*kwargs provides an elegant solution for handling attribute initialization without creating multiple constructor overloads or complex conditional logic.

```python
class Vehicle:
    def __init__(self, **kwargs):
        # Dynamically set attributes based on kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def display_specs(self):
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")

# Create different vehicles with varying attributes
car = Vehicle(type="Car", brand="Toyota", model="Camry", 
              year=2023, color="Blue")
motorcycle = Vehicle(type="Motorcycle", brand="Honda", 
                    model="CBR", engine_cc=1000)

car.display_specs()
print("\n")
motorcycle.display_specs()

# Output:
# type: Car
# brand: Toyota
# model: Camry
# year: 2023
# color: Blue
#
# type: Motorcycle
# brand: Honda
# model: CBR
# engine_cc: 1000
```

Slide 3: Function Decorators with \*\*kwargs

\*\*kwargs enables creation of flexible decorators that can modify or enhance function behavior while preserving the original function's signature and allowing additional configuration options through keyword arguments.

```python
def configurable_logger(log_level="INFO", **decorator_kwargs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Pre-execution logging
            print(f"[{log_level}] Calling {func.__name__}")
            print(f"Decorator config: {decorator_kwargs}")
            print(f"Arguments: {args}")
            print(f"Keyword arguments: {kwargs}")
            
            result = func(*args, **kwargs)
            
            # Post-execution logging
            print(f"[{log_level}] Completed {func.__name__}")
            return result
        return wrapper
    return decorator

@configurable_logger(log_level="DEBUG", timestamp=True)
def process_data(data, **options):
    return f"Processed {data} with options {options}"

result = process_data("sample", format="json", validate=True)
print(result)
```

Slide 4: Chain Processing with \*\*kwargs

Dynamic parameter handling through \*\*kwargs enables creation of processing chains where each step can accept different parameters while maintaining a consistent interface for data flow and transformation.

```python
class DataProcessor:
    @staticmethod
    def normalize_data(data, **kwargs):
        scale = kwargs.get('scale', 1.0)
        offset = kwargs.get('offset', 0)
        return [scale * x + offset for x in data]
    
    @staticmethod
    def filter_data(data, **kwargs):
        threshold = kwargs.get('threshold', 0)
        return [x for x in data if x > threshold]
    
    @staticmethod
    def transform_data(data, **kwargs):
        power = kwargs.get('power', 1)
        return [x ** power for x in data]

# Process chain with different parameters
data = [1, 2, 3, 4, 5]
processor = DataProcessor()

# Chain multiple operations with different parameters
result = processor.transform_data(
    processor.filter_data(
        processor.normalize_data(data, scale=2.0, offset=1),
        threshold=3
    ),
    power=2
)

print(f"Original data: {data}")
print(f"Processed data: {result}")
```

Slide 5: Advanced Type Handling with \*\*kwargs

\*\*kwargs can be enhanced with type validation and conversion mechanisms, ensuring robust parameter handling while maintaining flexibility. This pattern is particularly useful in data processing pipelines where input types need to be strictly controlled.

```python
def type_safe_processor(**kwargs):
    type_map = {
        'int_param': int,
        'float_param': float,
        'str_param': str,
        'list_param': list
    }
    
    processed_kwargs = {}
    for key, value in kwargs.items():
        if key in type_map:
            try:
                processed_kwargs[key] = type_map[key](value)
            except (ValueError, TypeError) as e:
                raise TypeError(f"Cannot convert {key}={value} to {type_map[key].__name__}")
        else:
            processed_kwargs[key] = value
    
    return processed_kwargs

# Example usage with type conversion
try:
    result = type_safe_processor(
        int_param="123",
        float_param="45.67",
        str_param=789,
        list_param="1,2,3",
        custom_param="test"
    )
    print("Processed parameters:", result)
except TypeError as e:
    print(f"Error: {e}")
```

Slide 6: Dynamic API Interface Generator

This implementation demonstrates how \*\*kwargs can be used to create flexible API interfaces that adapt to different backend services while maintaining a consistent interface for clients.

```python
class DynamicAPIInterface:
    def __init__(self, base_url, **config):
        self.base_url = base_url
        self.config = config
        self.headers = config.get('headers', {})
        
    def create_endpoint(self, endpoint_name, **kwargs):
        def dynamic_endpoint(**params):
            # Merge default configs with endpoint-specific params
            request_config = {
                'url': f"{self.base_url}/{endpoint_name}",
                'method': kwargs.get('method', 'GET'),
                'headers': {**self.headers, **params.get('headers', {})},
                'params': {k:v for k,v in params.items() 
                          if k not in ['headers']}
            }
            return self._make_request(**request_config)
            
        return dynamic_endpoint
    
    def _make_request(self, **config):
        # Simulate HTTP request
        return f"Request to {config['url']} with params {config['params']}"

# Example usage
api = DynamicAPIInterface('https://api.example.com',
                         headers={'Authorization': 'Bearer token'})

# Create dynamic endpoints
get_users = api.create_endpoint('users', method='GET')
create_user = api.create_endpoint('users', method='POST')

# Use the dynamic endpoints
result1 = get_users(page=1, limit=10)
result2 = create_user(name="John", email="john@example.com")

print(result1)
print(result2)
```

Slide 7: Event Handling System

A sophisticated event handling system that uses \*\*kwargs to pass variable event data between components while maintaining loose coupling and high flexibility.

```python
class EventDispatcher:
    def __init__(self):
        self.listeners = {}
        
    def add_listener(self, event_type, callback):
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)
        
    def dispatch(self, event_type, **event_data):
        if event_type in self.listeners:
            for callback in self.listeners[event_type]:
                callback(**event_data)

# Example implementation
class UserSystem:
    def __init__(self):
        self.dispatcher = EventDispatcher()
        
    def register_user(self, username, **user_data):
        # Process registration
        self.dispatcher.dispatch('user_registered',
                               username=username,
                               timestamp=time.time(),
                               **user_data)
        
    def log_event(self, **event_data):
        print(f"Log: User {event_data['username']} registered at "
              f"{event_data['timestamp']} with data: "
              f"{event_data}")

# Usage example
user_system = UserSystem()
user_system.dispatcher.add_listener('user_registered', 
                                  user_system.log_event)

user_system.register_user('john_doe',
                         email='john@example.com',
                         age=30,
                         preferences={'theme': 'dark'})
```

Slide 8: Dynamic Configuration Management

This implementation showcases how \*\*kwargs can be used to create a flexible configuration management system that handles nested configurations and supports dynamic updates while maintaining type safety and validation.

```python
class ConfigManager:
    def __init__(self, **initial_config):
        self._config = {}
        self._validators = {}
        self.update_config(**initial_config)
    
    def add_validator(self, key, validator_func):
        self._validators[key] = validator_func
    
    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if key in self._validators:
                if not self._validators[key](value):
                    raise ValueError(f"Invalid value for {key}: {value}")
            self._config[key] = value
    
    def get_config(self, key=None):
        if key is None:
            return self._config
        return self._config.get(key)

# Example usage
def validate_port(port):
    return isinstance(port, int) and 0 <= port <= 65535

config = ConfigManager(
    host="localhost",
    port=8080,
    debug=True,
    database={
        "url": "postgresql://localhost:5432",
        "pool_size": 5
    }
)

config.add_validator('port', validate_port)

try:
    # Valid update
    config.update_config(port=8081)
    print("Valid config:", config.get_config())
    
    # Invalid update
    config.update_config(port=70000)
except ValueError as e:
    print(f"Error: {e}")
```

Slide 9: Data Pipeline with Dynamic Transformers

A flexible data pipeline system that uses \*\*kwargs to configure transformation steps and handle different data types and processing requirements dynamically.

```python
class DataTransformer:
    def __init__(self):
        self.transformations = {}
    
    def register_transformation(self, name, func):
        self.transformations[name] = func
    
    def transform(self, data, **kwargs):
        result = data
        for step, params in kwargs.items():
            if step in self.transformations:
                result = self.transformations[step](result, **params)
        return result

# Register common transformations
transformer = DataTransformer()

def filter_transform(data, **params):
    condition = params.get('condition', lambda x: True)
    return [x for x in data if condition(x)]

def map_transform(data, **params):
    operation = params.get('operation', lambda x: x)
    return [operation(x) for x in data]

transformer.register_transformation('filter', filter_transform)
transformer.register_transformation('map', map_transform)

# Example usage
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = transformer.transform(
    data,
    filter={'condition': lambda x: x % 2 == 0},
    map={'operation': lambda x: x * 2}
)

print(f"Original data: {data}")
print(f"Transformed data: {result}")
```

Slide 10: Dynamic Report Generator

A sophisticated report generation system that uses \*\*kwargs to handle different report formats, styles, and content types while maintaining a clean and extensible architecture.

```python
class ReportGenerator:
    def __init__(self):
        self.formatters = {}
        self.sections = {}
    
    def register_formatter(self, format_type, formatter_func):
        self.formatters[format_type] = formatter_func
    
    def register_section(self, section_name, section_func):
        self.sections[section_name] = section_func
    
    def generate_report(self, data, **kwargs):
        format_type = kwargs.get('format', 'text')
        sections = kwargs.get('sections', list(self.sections.keys()))
        
        report_content = {}
        for section in sections:
            if section in self.sections:
                report_content[section] = self.sections[section](
                    data, **kwargs.get(section, {})
                )
        
        if format_type in self.formatters:
            return self.formatters[format_type](report_content, **kwargs)
        raise ValueError(f"Unknown format type: {format_type}")

# Example implementation
def text_formatter(content, **kwargs):
    report = []
    for section, data in content.items():
        report.append(f"=== {section.upper()} ===")
        report.append(str(data))
        report.append("")
    return "\n".join(report)

def summary_section(data, **kwargs):
    threshold = kwargs.get('threshold', 0)
    return {
        'count': len(data),
        'sum': sum(x for x in data if x > threshold)
    }

# Setup and usage
generator = ReportGenerator()
generator.register_formatter('text', text_formatter)
generator.register_section('summary', summary_section)

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
report = generator.generate_report(
    data,
    format='text',
    sections=['summary'],
    summary={'threshold': 5}
)

print(report)
```

Slide 11: Dynamic Machine Learning Pipeline

This implementation demonstrates how \*\*kwargs can be used to create flexible machine learning pipelines that support different preprocessing steps, model configurations, and evaluation metrics dynamically.

```python
class MLPipeline:
    def __init__(self):
        self.preprocessors = {}
        self.metrics = {}
        
    def register_preprocessor(self, name, func):
        self.preprocessors[name] = func
        
    def register_metric(self, name, func):
        self.metrics[name] = func
        
    def process_data(self, X, y=None, **kwargs):
        processed_X = X
        processed_y = y
        
        for step, params in kwargs.get('preprocessing', {}).items():
            if step in self.preprocessors:
                if y is not None:
                    processed_X, processed_y = self.preprocessors[step](
                        processed_X, processed_y, **params
                    )
                else:
                    processed_X = self.preprocessors[step](
                        processed_X, **params
                    )
                    
        return processed_X, processed_y
    
    def evaluate(self, y_true, y_pred, **kwargs):
        results = {}
        for metric_name, params in kwargs.get('metrics', {}).items():
            if metric_name in self.metrics:
                results[metric_name] = self.metrics[metric_name](
                    y_true, y_pred, **params
                )
        return results

# Example preprocessing functions
def normalize_data(X, y=None, **kwargs):
    scale = kwargs.get('scale', 1.0)
    X_norm = X / scale
    return (X_norm, y) if y is not None else X_norm

def add_polynomial_features(X, y=None, **kwargs):
    degree = kwargs.get('degree', 2)
    X_poly = np.column_stack([X ** i for i in range(1, degree + 1)])
    return (X_poly, y) if y is not None else X_poly

# Example metric functions
def custom_rmse(y_true, y_pred, **kwargs):
    weights = kwargs.get('weights', None)
    if weights is not None:
        return np.sqrt(np.mean(weights * (y_true - y_pred) ** 2))
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Usage example
pipeline = MLPipeline()
pipeline.register_preprocessor('normalize', normalize_data)
pipeline.register_preprocessor('polynomial', add_polynomial_features)
pipeline.register_metric('rmse', custom_rmse)

# Generate sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Process data with dynamic configuration
X_processed, y_processed = pipeline.process_data(
    X, y,
    preprocessing={
        'normalize': {'scale': 2.0},
        'polynomial': {'degree': 2}
    }
)

# Evaluate with custom metrics
y_pred = X_processed[:, 0] * 2  # Simple prediction
metrics = pipeline.evaluate(
    y_processed, y_pred,
    metrics={
        'rmse': {'weights': np.array([1, 1, 2, 2, 3])}
    }
)

print("Processed features shape:", X_processed.shape)
print("Evaluation metrics:", metrics)
```

Slide 12: Real-time Event Processing System

A sophisticated event processing system that uses \*\*kwargs to handle different types of events, apply filters, and trigger appropriate actions based on dynamic configurations.

```python
class EventProcessor:
    def __init__(self):
        self.handlers = {}
        self.filters = {}
        
    def register_handler(self, event_type, handler_func):
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler_func)
        
    def register_filter(self, filter_name, filter_func):
        self.filters[filter_name] = filter_func
        
    def process_event(self, event_type, **event_data):
        if event_type not in self.handlers:
            return
            
        # Apply filters if specified
        should_process = True
        for filter_name, filter_params in event_data.get('filters', {}).items():
            if filter_name in self.filters:
                should_process &= self.filters[filter_name](
                    event_data, **filter_params
                )
                
        if should_process:
            results = []
            for handler in self.handlers[event_type]:
                results.append(handler(**event_data))
            return results
            
    def bulk_process(self, events, **kwargs):
        results = []
        for event in events:
            event_type = event.pop('type')
            result = self.process_event(event_type, **event)
            results.append(result)
        return results

# Example implementation
def threshold_filter(event_data, **params):
    threshold = params.get('threshold', 0)
    return event_data.get('value', 0) > threshold

def time_window_filter(event_data, **params):
    start_time = params.get('start', 0)
    end_time = params.get('end', float('inf'))
    event_time = event_data.get('timestamp', 0)
    return start_time <= event_time <= end_time

# Handler functions
def log_handler(**event_data):
    return f"Logged: {event_data}"

def alert_handler(**event_data):
    return f"Alert: {event_data.get('value')} exceeded threshold"

# Usage example
processor = EventProcessor()
processor.register_filter('threshold', threshold_filter)
processor.register_filter('time_window', time_window_filter)
processor.register_handler('sensor_data', log_handler)
processor.register_handler('sensor_data', alert_handler)

# Process single event
event_result = processor.process_event(
    'sensor_data',
    value=75,
    timestamp=1635724800,
    filters={
        'threshold': {'threshold': 50},
        'time_window': {
            'start': 1635724000,
            'end': 1635725000
        }
    }
)

print("Event processing results:", event_result)
```

Slide 13: Dynamic Database Query Builder

This implementation showcases how \*\*kwargs can be used to create a flexible SQL query builder that handles complex queries with dynamic filtering, sorting, and joins while maintaining security against SQL injection.

```python
class QueryBuilder:
    def __init__(self):
        self.query_parts = {}
        self.params = []
        self.counter = 0
        
    def build_select(self, table, **kwargs):
        columns = kwargs.get('columns', ['*'])
        joins = kwargs.get('joins', [])
        filters = kwargs.get('filters', {})
        order_by = kwargs.get('order_by', None)
        limit = kwargs.get('limit', None)
        
        # Build base query
        query = f"SELECT {', '.join(columns)} FROM {table}"
        
        # Add joins
        for join in joins:
            query += f" {join.get('type', 'LEFT')} JOIN {join['table']}"
            query += f" ON {join['condition']}"
            
        # Add where clauses
        if filters:
            where_clauses = []
            for field, value in filters.items():
                self.counter += 1
                param_name = f"${self.counter}"
                where_clauses.append(f"{field} = {param_name}")
                self.params.append(value)
            query += " WHERE " + " AND ".join(where_clauses)
            
        # Add ordering
        if order_by:
            query += f" ORDER BY {order_by}"
            
        # Add limit
        if limit:
            self.counter += 1
            query += f" LIMIT ${self.counter}"
            self.params.append(limit)
            
        return query, self.params

# Example usage
query_builder = QueryBuilder()
query, params = query_builder.build_select(
    'users',
    columns=['id', 'name', 'email'],
    joins=[
        {
            'type': 'LEFT',
            'table': 'orders',
            'condition': 'users.id = orders.user_id'
        }
    ],
    filters={
        'users.active': True,
        'orders.status': 'completed'
    },
    order_by='users.created_at DESC',
    limit=10
)

print("Generated Query:", query)
print("Query Parameters:", params)
```

Slide 14: Asynchronous Task Pipeline

A flexible asynchronous task processing system that uses \*\*kwargs to handle different types of tasks, dependencies, and execution configurations.

```python
import asyncio
from typing import Dict, Any, Callable, Awaitable

class AsyncTaskPipeline:
    def __init__(self):
        self.tasks: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self.results: Dict[str, Any] = {}
        
    def register_task(self, name: str, 
                     task_func: Callable[..., Awaitable[Any]]):
        self.tasks[name] = task_func
        
    async def execute_pipeline(self, **kwargs):
        # Get task configuration
        task_config = kwargs.get('tasks', {})
        dependencies = kwargs.get('dependencies', {})
        timeout = kwargs.get('timeout', None)
        
        # Create task groups based on dependencies
        task_groups = self._create_task_groups(dependencies)
        
        # Execute task groups in order
        try:
            async with asyncio.timeout(timeout):
                for group in task_groups:
                    group_tasks = []
                    for task_name in group:
                        if task_name in task_config:
                            task_params = task_config[task_name]
                            group_tasks.append(
                                self._execute_task(task_name, **task_params)
                            )
                    if group_tasks:
                        await asyncio.gather(*group_tasks)
        except asyncio.TimeoutError:
            return {'status': 'timeout', 'results': self.results}
            
        return {'status': 'complete', 'results': self.results}
        
    async def _execute_task(self, task_name: str, **task_params):
        if task_name in self.tasks:
            try:
                result = await self.tasks[task_name](**task_params)
                self.results[task_name] = {
                    'status': 'success',
                    'result': result
                }
            except Exception as e:
                self.results[task_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                
    def _create_task_groups(self, dependencies):
        # Implementation of topological sort for task dependencies
        groups = []
        visited = set()
        temp_visited = set()
        
        def visit(task):
            if task in temp_visited:
                raise ValueError("Circular dependency detected")
            if task in visited:
                return
                
            temp_visited.add(task)
            
            for dep in dependencies.get(task, []):
                visit(dep)
                
            temp_visited.remove(task)
            visited.add(task)
            
            # Add task to appropriate group
            added = False
            for group in groups:
                if not any(dep in group for dep in dependencies.get(task, [])):
                    group.add(task)
                    added = True
                    break
            if not added:
                groups.append({task})
                
        for task in self.tasks:
            if task not in visited:
                visit(task)
                
        return groups

# Example usage
async def fetch_data(**kwargs):
    await asyncio.sleep(1)  # Simulate API call
    return {'data': f"Fetched with params: {kwargs}"}

async def process_data(**kwargs):
    await asyncio.sleep(0.5)  # Simulate processing
    return {'processed': f"Processed with params: {kwargs}"}

# Setup pipeline
pipeline = AsyncTaskPipeline()
pipeline.register_task('fetch', fetch_data)
pipeline.register_task('process', process_data)

# Execute pipeline
async def main():
    result = await pipeline.execute_pipeline(
        tasks={
            'fetch': {'url': 'api.example.com', 'method': 'GET'},
            'process': {'format': 'json', 'validate': True}
        },
        dependencies={
            'process': ['fetch']
        },
        timeout=5
    )
    print("Pipeline results:", result)

asyncio.run(main())
```

Slide 15: Additional Resources

*   arxiv.org/abs/2103.00020 - "Dynamic Parameter Allocation in Parameter Servers"
*   arxiv.org/abs/1912.13054 - "Adaptive Parameter Sharing for Multi-Task Learning"
*   arxiv.org/abs/2010.04159 - "Dynamic Neural Networks: A Survey"
*   arxiv.org/abs/2106.06935 - "Parameter-Efficient Transfer Learning with Dynamic Architecture Search"

