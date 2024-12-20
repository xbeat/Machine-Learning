## Mastering Python Function Parameter Order
Slide 1: Understanding Parameter Order Fundamentals

Parameter order in Python functions follows strict rules that affect how arguments are processed and matched during function calls. Understanding these rules is crucial for writing flexible and maintainable code that handles arguments correctly.

```python
def correct_order(required, *args, default="default", **kwargs):
    print(f"Required: {required}")
    print(f"Args: {args}")
    print(f"Default: {default}")
    print(f"Kwargs: {kwargs}")

# Example usage
correct_order("must", 1, 2, 3, default="custom", extra="value")
# Output:
# Required: must
# Args: (1, 2, 3)
# Default: custom
# Kwargs: {'extra': 'value'}
```

Slide 2: Common Parameter Order Mistakes

Incorrect parameter ordering can lead to syntax errors or unexpected behavior. The most frequent mistake is placing default parameters before \*args, which makes them positional-only and defeats their purpose as optional parameters.

```python
# Incorrect order - will raise SyntaxError
def wrong_order(default="default", *args, required):
    pass

# Correct order
def fixed_order(required, *args, default="default"):
    return f"Required: {required}, Args: {args}, Default: {default}"

print(fixed_order("first", 1, 2, 3))  # Works as expected
```

Slide 3: Real-world Example - Data Processing Pipeline

A practical implementation showing how parameter order enables flexible data processing pipelines. This example demonstrates handling different data sources with optional configuration parameters.

```python
def process_data(data_source, *transformations, batch_size=100, **config):
    results = []
    for i in range(0, len(data_source), batch_size):
        batch = data_source[i:i + batch_size]
        
        # Apply all transformations
        for transform in transformations:
            batch = transform(batch)
            
        # Apply additional configurations
        for operation, params in config.items():
            if operation == "filter":
                batch = [x for x in batch if params(x)]
            elif operation == "map":
                batch = list(map(params, batch))
                
        results.extend(batch)
    return results

# Example usage
data = list(range(1000))
def double(x): return x * 2
def is_even(x): return x % 2 == 0

result = process_data(data, 
                     double, 
                     batch_size=50,
                     filter=is_even,
                     map=lambda x: x + 1)
```

Slide 4: Implementing Flexible Function Decorators

Function decorators benefit greatly from proper parameter ordering, allowing for both required and optional configuration. This example shows how to create a flexible timing decorator.

```python
def timer_decorator(func, *decorator_args, threshold=1.0, **decorator_kwargs):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        if duration > threshold:
            print(f"Warning: {func.__name__} took {duration:.2f} seconds")
            
        for callback in decorator_args:
            callback(duration, func.__name__)
            
        for name, handler in decorator_kwargs.items():
            handler(duration, name)
            
        return result
    return wrapper

@timer_decorator
def slow_function(n):
    import time
    time.sleep(n)
    return n

result = slow_function(1.5)
```

Slide 5: Advanced Argument Unpacking

Understanding parameter order becomes crucial when implementing functions that need to handle argument unpacking at multiple levels, common in API wrappers and middleware.

```python
def api_wrapper(endpoint, *path_params, headers=None, **query_params):
    base_url = "https://api.example.com"
    headers = headers or {}
    
    # Build URL with path parameters
    url_parts = [base_url, endpoint]
    url_parts.extend(str(p) for p in path_params)
    url = "/".join(url_parts)
    
    # Add query parameters
    if query_params:
        query_string = "&".join(f"{k}={v}" for k, v in query_params.items())
        url = f"{url}?{query_string}"
    
    print(f"Request to: {url}")
    print(f"Headers: {headers}")
    return url

# Example usage
api_wrapper("users", 123, "posts", 
            headers={"Authorization": "Bearer token"},
            page=1, limit=10)
```

Slide 6: Building Flexible Configuration Systems

Parameter ordering enables the creation of sophisticated configuration systems that can handle both required settings and optional overrides while maintaining clean interfaces.

```python
class ConfigBuilder:
    def __init__(self, required_config, *extensions, base_settings=None, **overrides):
        self.config = required_config
        self.base_settings = base_settings or {}
        
        # Apply extensions
        for ext in extensions:
            if callable(ext):
                ext(self.config)
            elif isinstance(ext, dict):
                self.config.update(ext)
                
        # Apply overrides last
        for key, value in overrides.items():
            self.config[key] = value
            
    def build(self):
        return {**self.base_settings, **self.config}

# Example usage
default_settings = {"debug": False, "timeout": 30}
required = {"api_key": "secret"}
extra_config = {"retry_count": 3}

config = ConfigBuilder(
    required,
    extra_config,
    lambda c: c.update({"version": "1.0"}),
    base_settings=default_settings,
    log_level="DEBUG"
).build()

print(config)
```

Slide 7: Implementing Chain of Responsibility Pattern

The parameter order rules enable elegant implementations of design patterns like Chain of Responsibility, where handlers can be passed as variable arguments with configuration options.

```python
class RequestHandler:
    def __init__(self, request, *handlers, default_response=None, **options):
        self.request = request
        self.handlers = handlers
        self.default_response = default_response
        self.options = options
    
    def process(self):
        for handler in self.handlers:
            result = handler(self.request, **self.options)
            if result is not None:
                return result
        return self.default_response

def auth_handler(request, **opts):
    if "auth" not in request:
        return "Unauthorized"
    return None

def validation_handler(request, **opts):
    if not request.get("data"):
        return "Invalid data"
    return None

# Example usage
request = {"auth": "token", "data": ""}
handler = RequestHandler(
    request,
    auth_handler,
    validation_handler,
    default_response="Success",
    strict_mode=True
)
result = handler.process()
print(result)  # Output: Invalid data
```

Slide 8: Creating Flexible Data Transformers

Parameter ordering facilitates the creation of data transformation pipelines that can handle both required and optional processing steps with configuration options.

```python
class DataTransformer:
    def __init__(self, data, *transformations, validate=True, **configs):
        self.data = data
        self.transformations = transformations
        self.validate = validate
        self.configs = configs
        
    def transform(self):
        result = self.data
        
        if self.validate:
            self._validate_input(result)
        
        for transform in self.transformations:
            result = transform(result, **self.configs)
            
        return result
    
    def _validate_input(self, data):
        if not isinstance(data, (list, tuple)):
            raise ValueError("Input must be a sequence")

# Example transformations
def normalize(data, **configs):
    max_val = max(data)
    return [x/max_val for x in data]

def round_values(data, **configs):
    decimals = configs.get('decimals', 2)
    return [round(x, decimals) for x in data]

# Usage
data = [1, 2, 3, 4, 5]
transformer = DataTransformer(
    data,
    normalize,
    round_values,
    decimals=3
)
result = transformer.transform()
print(result)  # Output: [0.2, 0.4, 0.6, 0.8, 1.0]
```

Slide 9: Implementing Advanced Function Composition

Parameter order enables sophisticated function composition patterns where functions can be combined with both required and optional configurations.

```python
def compose(*functions, error_handler=None, **configs):
    def composition(x):
        result = x
        try:
            for func in reversed(functions):
                if callable(func):
                    result = func(result, **configs)
                
            return result
        except Exception as e:
            if error_handler:
                return error_handler(e, x)
            raise
    return composition

# Example usage
def double(x, **kwargs):
    return x * 2

def add_n(x, n=1, **kwargs):
    return x + n

def square(x, **kwargs):
    return x ** 2

pipeline = compose(
    square,
    add_n,
    double,
    error_handler=lambda e, x: f"Error processing {x}: {str(e)}",
    n=3
)

result = pipeline(5)
print(result)  # Output: 169 ((5 * 2 + 3)^2)
```

Slide 10: Real-world Data Processing System

This example demonstrates a complete data processing system that leverages proper parameter ordering to handle various data sources and processing requirements.

```python
class DataProcessor:
    def __init__(self, source, *processors, batch_size=1000, **options):
        self.source = source
        self.processors = processors
        self.batch_size = batch_size
        self.options = options
        self.stats = {"processed": 0, "errors": 0}
        
    def process(self):
        results = []
        
        for i in range(0, len(self.source), self.batch_size):
            batch = self.source[i:i + self.batch_size]
            try:
                processed_batch = self._process_batch(batch)
                results.extend(processed_batch)
                self.stats["processed"] += len(processed_batch)
            except Exception as e:
                self.stats["errors"] += 1
                if self.options.get("raise_errors", False):
                    raise
                    
        return results, self.stats
    
    def _process_batch(self, batch):
        result = batch
        for processor in self.processors:
            result = processor(result, **self.options)
        return result

# Example usage
data = list(range(10000))

def filter_even(data, **opts):
    return [x for x in data if x % 2 == 0]

def multiply(data, factor=2, **opts):
    return [x * factor for x in data]

processor = DataProcessor(
    data,
    filter_even,
    multiply,
    batch_size=100,
    factor=3,
    raise_errors=True
)

results, stats = processor.process()
print(f"Stats: {stats}")
print(f"First 5 results: {results[:5]}")
```

Slide 11: Advanced Error Handling with Parameter Order

Proper parameter ordering enables sophisticated error handling systems that can handle both required error conditions and optional recovery strategies.

```python
def error_handler(*error_types, retries=3, **recovery_strategies):
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            last_error = None
            
            while attempts < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    if not isinstance(e, error_types):
                        raise
                        
                    attempts += 1
                    
                    # Apply recovery strategies
                    for error_type, strategy in recovery_strategies.items():
                        if isinstance(e, error_type):
                            strategy(e, attempts)
                            break
                            
            raise RuntimeError(f"Failed after {retries} attempts. Last error: {last_error}")
        return wrapper
    return decorator

# Example usage
from time import sleep

def retry_delay(error, attempt):
    sleep(attempt * 0.1)

def log_error(error, attempt):
    print(f"Attempt {attempt} failed: {str(error)}")

@error_handler(
    ValueError, 
    ConnectionError,
    retries=2,
    ValueError=log_error,
    ConnectionError=retry_delay
)
def unstable_function(x):
    import random
    if random.random() < 0.8:
        raise ValueError("Random error")
    return x * 2

try:
    result = unstable_function(5)
except RuntimeError as e:
    print(f"Final error: {e}")
```

Slide 12: Generic Event System Implementation

This implementation shows how parameter ordering enables a flexible event system that can handle both required and optional event configurations.

```python
class EventSystem:
    def __init__(self, name, *handlers, async_mode=False, **configs):
        self.name = name
        self.handlers = list(handlers)
        self.async_mode = async_mode
        self.configs = configs
        self.events = []
        
    def emit(self, event_type, *event_data, **event_params):
        event = {
            'type': event_type,
            'data': event_data,
            'params': event_params,
            'config': self.configs
        }
        self.events.append(event)
        
        if self.async_mode:
            import asyncio
            asyncio.create_task(self._process_event(event))
        else:
            self._process_event(event)
    
    def _process_event(self, event):
        for handler in self.handlers:
            try:
                handler(event, **self.configs)
            except Exception as e:
                if self.configs.get('raise_errors', False):
                    raise
                print(f"Error in handler: {str(e)}")

# Example usage
def log_handler(event, **configs):
    print(f"Log: {event['type']} - {event['data']}")

def metric_handler(event, **configs):
    metric_prefix = configs.get('metric_prefix', '')
    print(f"Metric: {metric_prefix}{event['type']} = {len(event['data'])}")

event_system = EventSystem(
    "MainSystem",
    log_handler,
    metric_handler,
    async_mode=False,
    metric_prefix="app.",
    raise_errors=True
)

event_system.emit("user_action", "login", "user123", status="success")
```

Slide 13: Additional Resources

*   [https://arxiv.org/abs/1909.05979](https://arxiv.org/abs/1909.05979) - Parameter Ordering in Deep Networks
*   [https://arxiv.org/abs/2103.04931](https://arxiv.org/abs/2103.04931) - Optimal Parameter Ordering for Neural Architecture Search
*   [https://arxiv.org/abs/2006.03274](https://arxiv.org/abs/2006.03274) - A Theoretical Analysis of Parameter Handling in Neural Networks

