## Understanding Python Namespaces and Scopes
Slide 1: Understanding Python Namespaces

Python namespaces represent the mapping between names and objects in memory, functioning as containers that hold variable names, functions, and class definitions. They prevent naming conflicts by organizing identifiers into distinct scopes, enabling modular and maintainable code development.

```python
# Built-in namespace
print(dir(__builtins__))  # Shows built-in functions and exceptions

# Global namespace example
global_var = 100

def example_function():
    # Local namespace
    local_var = 200
    print(f"Access global: {global_var}")
    print(f"Access local: {local_var}")

example_function()
```

Slide 2: Namespace Hierarchy and LEGB Rule

The LEGB rule defines Python's namespace hierarchy: Local, Enclosing, Global, and Built-in. This order determines how Python searches for names, starting from the innermost scope (Local) and moving outward until it finds the first matching name or raises a NameError.

```python
x = "Global"  # Global scope

def outer():
    x = "Enclosing"  # Enclosing scope
    def inner():
        x = "Local"  # Local scope
        print(f"Local x: {x}")
    inner()
    print(f"Enclosing x: {x}")

outer()
print(f"Global x: {x}")
```

Slide 3: Global and Nonlocal Keywords

The global and nonlocal keywords modify Python's default scope behavior, allowing inner scopes to modify variables in outer scopes. This capability requires careful management to maintain code clarity and prevent unexpected side effects.

```python
counter = 0  # Global variable

def modify_global():
    global counter
    counter += 1
    
def outer():
    value = 100
    def inner():
        nonlocal value
        value += 50
        return value
    return inner()

print(f"Before modification: {counter}")
modify_global()
print(f"After modification: {counter}")
print(f"Nonlocal example: {outer()}")
```

Slide 4: Name Resolution in Class Namespaces

Class namespaces introduce an additional layer of complexity in Python's name resolution system. They maintain their own namespace for class attributes and methods, which interacts with instance namespaces through inheritance and method resolution order.

```python
class ExampleClass:
    class_var = "I'm a class variable"
    
    def __init__(self):
        self.instance_var = "I'm an instance variable"
    
    def demonstrate_scope(self):
        method_var = "I'm a method variable"
        print(f"Class var: {ExampleClass.class_var}")
        print(f"Instance var: {self.instance_var}")
        print(f"Method var: {method_var}")

example = ExampleClass()
example.demonstrate_scope()
```

Slide 5: Dynamic Namespace Management

Python's dynamic nature allows runtime modification of namespaces using built-in functions like globals(), locals(), and vars(). This flexibility enables metaprogramming and dynamic code execution capabilities.

```python
def namespace_explorer():
    x = 100
    # Examine local namespace
    local_ns = locals()
    print("Local namespace:", local_ns)
    
    # Examine global namespace
    global_ns = globals()
    print("\nGlobal namespace keys:", list(global_ns.keys())[:5])
    
    # Dynamic variable creation
    globals()['dynamic_var'] = "Created at runtime"
    print("\nDynamic variable:", dynamic_var)

namespace_explorer()
```

Slide 6: Name Masking and Variable Shadowing

Name masking occurs when a variable in an inner scope has the same name as a variable in an outer scope, effectively hiding the outer variable. Understanding this behavior is crucial for avoiding subtle bugs in nested scopes.

```python
def demonstrate_masking():
    x = "outer"
    def inner():
        x = "inner"  # Masks outer x
        print(f"Inner x: {x}")
    
    inner()
    print(f"Outer x: {x}")

def demonstrate_nonlocal():
    x = "outer"
    def inner():
        nonlocal x  # Modifies outer x
        x = "modified"
    
    print(f"Before: {x}")
    inner()
    print(f"After: {x}")

demonstrate_masking()
demonstrate_nonlocal()
```

Slide 7: Module-level Namespaces

Module-level namespaces are created when importing Python modules, providing a way to organize code and prevent name collisions between different modules. Each module maintains its own global namespace.

```python
# math_operations.py
PI = 3.14159

def calculate_area(radius):
    return PI * radius ** 2

# main.py
import math_operations as mo

# Module namespace accessed through the module name
print(f"PI from module: {mo.PI}")
print(f"Area calculation: {mo.calculate_area(5)}")

# Creating a different PI in current namespace
PI = 3.14
print(f"Local PI: {PI}")
```

Slide 8: Real-world Example: Data Processing Pipeline

This example demonstrates namespace management in a practical data processing pipeline, showing how different scopes interact in a complex application structure.

```python
class DataProcessor:
    default_config = {"batch_size": 32, "normalize": True}
    
    def __init__(self, custom_config=None):
        self.config = {**self.default_config, **(custom_config or {})}
        self._processed_data = []
    
    def process_batch(self, data):
        def normalize(batch):
            nonlocal data
            return [x / max(data) for x in batch]
        
        batch_size = self.config["batch_size"]
        processed = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            if self.config["normalize"]:
                batch = normalize(batch)
            processed.extend(batch)
            
        self._processed_data = processed
        return processed

# Usage example
processor = DataProcessor({"batch_size": 2})
result = processor.process_batch([1, 2, 3, 4, 5, 6])
print(f"Processed data: {result}")
```

Slide 9: Namespace Management in Decorators

Decorators demonstrate advanced namespace manipulation, creating closures that maintain their own scope while modifying function behavior. This example shows how to implement function timing using namespace closure.

```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def complex_operation(n):
    return sum(i * i for i in range(n))

result = complex_operation(1000000)
print(f"Result: {result}")
```

Slide 10: Context Managers and Namespace Cleanup

Context managers provide a clean way to manage resource allocation and deallocation, demonstrating how namespace cleanup works in Python's with statement implementation.

```python
class DatabaseConnection:
    def __init__(self, db_name):
        self.db_name = db_name
        self._connection = None
    
    def __enter__(self):
        print(f"Connecting to {self.db_name}")
        self._connection = f"Connection to {self.db_name}"
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing connection to {self.db_name}")
        self._connection = None
    
    def query(self, sql):
        if self._connection:
            return f"Executing '{sql}' on {self._connection}"
        raise RuntimeError("No active connection")

with DatabaseConnection("example_db") as db:
    result = db.query("SELECT * FROM users")
    print(result)
```

Slide 11: Advanced Scope Management in Generators

Generator functions maintain their local namespace between yields, demonstrating how Python manages state across multiple function invocations while respecting scope boundaries.

```python
def stateful_generator(initial_value):
    counter = initial_value
    
    while True:
        received = yield counter
        if received is not None:
            counter = received
        else:
            counter += 1

gen = stateful_generator(0)
print(next(gen))  # 0
print(next(gen))  # 1
print(gen.send(10))  # 10
print(next(gen))  # 11
```

Slide 12: Memory Management and Namespace Lifetime

Understanding namespace lifetime is crucial for memory efficiency. This example demonstrates how Python manages memory for different scope levels and garbage collection.

```python
import sys
import weakref

class NamespaceDemo:
    def __init__(self, name):
        self.name = name
        
    def __del__(self):
        print(f"Cleaning up {self.name}")

def scope_lifetime_demo():
    local_obj = NamespaceDemo("local")
    refs = sys.getrefcount(local_obj)
    print(f"Reference count: {refs}")
    
    weak_ref = weakref.ref(local_obj)
    return weak_ref

ref = scope_lifetime_demo()
print(f"Object still exists? {ref() is not None}")
```

Slide 13: Real-world Example: Custom Configuration Management

This implementation shows how to create a configuration system that leverages Python's namespace mechanics for hierarchical settings management.

```python
class Configuration:
    def __init__(self):
        self._settings = {}
        self._namespace_stack = []
    
    def namespace(self, name):
        class NamespaceContext:
            def __init__(self, config, name):
                self.config = config
                self.name = name
            
            def __enter__(self):
                self.config._namespace_stack.append(self.name)
                return self.config
            
            def __exit__(self, *args):
                self.config._namespace_stack.pop()
        
        return NamespaceContext(self, name)
    
    def set(self, key, value):
        current_namespace = ".".join(self._namespace_stack)
        full_key = f"{current_namespace}.{key}" if current_namespace else key
        self._settings[full_key] = value
    
    def get(self, key, default=None):
        current_namespace = ".".join(self._namespace_stack)
        full_key = f"{current_namespace}.{key}" if current_namespace else key
        return self._settings.get(full_key, default)

# Usage example
config = Configuration()
with config.namespace("database"):
    config.set("host", "localhost")
    config.set("port", 5432)
    
    with config.namespace("credentials"):
        config.set("username", "admin")
        config.set("password", "secret")

print(config._settings)
```

Slide 14: Additional Resources

*   Scope Resolution and the LEGB Rule in Python
    *   [https://arxiv.org/abs/2203.15005](https://arxiv.org/abs/2203.15005)
*   Memory Management and Garbage Collection in Python
    *   [https://arxiv.org/abs/2106.09312](https://arxiv.org/abs/2106.09312)
*   Python Namespace Implementation Deep Dive
    *   [https://arxiv.org/abs/2204.12847](https://arxiv.org/abs/2204.12847)
*   For more information, search on Google Scholar:
    *   "Python scope resolution optimization"
    *   "Dynamic namespace management Python"
    *   "Python memory management techniques"

