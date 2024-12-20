## Python's Interpreted Nature and Default Argument Traps
Slide 1: The Default Argument Trap

Default arguments in Python are evaluated at function definition time, not during execution. This behavior can lead to unexpected results when using mutable objects like lists or dictionaries as default arguments, potentially causing data persistence between function calls.

```python
def append_to_list(item, target_list=[]):
    target_list.append(item)
    return target_list

# First call
print(append_to_list(1))  # Output: [1]
# Second call - unexpected behavior
print(append_to_list(2))  # Output: [1, 2]
# Third call - list keeps growing
print(append_to_list(3))  # Output: [1, 2, 3]
```

Slide 2: The Proper Way to Handle Mutable Defaults

To avoid the mutable default argument trap, use None as the default value and create the mutable object inside the function. This ensures a fresh instance is created for each function call, preventing unintended data sharing between calls.

```python
def append_to_list(item, target_list=None):
    if target_list is None:
        target_list = []
    target_list.append(item)
    return target_list

# Each call now creates a new list
print(append_to_list(1))  # Output: [1]
print(append_to_list(2))  # Output: [2]
print(append_to_list(3))  # Output: [3]
```

Slide 3: Default Arguments with Timestamps

When working with timestamps as default arguments, they are evaluated only once at module load time. This can cause issues in functions that need current timestamps, as the default value remains constant throughout the program's execution.

```python
from datetime import datetime

def log_event(message, timestamp=datetime.now()):
    return f"{timestamp}: {message}"

# First call
print(log_event("Start"))  # Output: 2024-11-27 10:00:00: Start
# Wait 5 seconds...
print(log_event("End"))    # Output: 2024-11-27 10:00:00: End (same timestamp!)
```

Slide 4: Correcting Timestamp Defaults

The proper way to handle timestamp defaults is to use None and generate the timestamp inside the function. This ensures that each function call uses the current time rather than the time when the function was defined.

```python
from datetime import datetime

def log_event(message, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now()
    return f"{timestamp}: {message}"

# Each call gets current timestamp
print(log_event("Start"))  # Output: 2024-11-27 10:00:00: Start
# Wait 5 seconds...
print(log_event("End"))    # Output: 2024-11-27 10:00:05: End (different timestamp)
```

Slide 5: Default Arguments in Class Methods

Class methods with mutable default arguments can be particularly problematic as they share state across all instances of the class. This can lead to unexpected behavior when multiple instances modify the same default argument.

```python
class DataCollector:
    def __init__(self):
        self.name = "Collector"
        
    def add_data(self, value, data_list=[]):
        data_list.append(value)
        return data_list

# Creating two instances
collector1 = DataCollector()
collector2 = DataCollector()

print(collector1.add_data(1))  # Output: [1]
print(collector2.add_data(2))  # Output: [1, 2] - Unexpected!
```

Slide 6: Class Methods with Safe Defaults

To properly implement class methods with mutable defaults, use instance attributes to store mutable state. This ensures each instance maintains its own independent state without sharing default arguments.

```python
class DataCollector:
    def __init__(self):
        self.name = "Collector"
        self.data_list = []
        
    def add_data(self, value, data_list=None):
        if data_list is None:
            data_list = self.data_list
        data_list.append(value)
        return data_list

# Creating two instances
collector1 = DataCollector()
collector2 = DataCollector()

print(collector1.add_data(1))  # Output: [1]
print(collector2.add_data(2))  # Output: [2]
```

Slide 7: Default Arguments in Recursive Functions

Default arguments in recursive functions require special attention as they can maintain state between recursive calls. This can lead to accumulated state that affects subsequent function calls.

```python
def traverse_tree(node, visited=[]):
    if not node:
        return visited
    visited.append(node.value)
    traverse_tree(node.left, visited)
    traverse_tree(node.right, visited)
    return visited

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

# Creating a simple tree
root = Node(1)
root.left = Node(2)
root.right = Node(3)

print(traverse_tree(root))  # First traversal
print(traverse_tree(root))  # Second traversal includes previous results!
```

Slide 8: Safe Recursive Functions with Defaults

The correct implementation of recursive functions should avoid mutable default arguments and instead use helper functions or parameter passing to maintain state during recursion.

```python
def traverse_tree(node, visited=None):
    if visited is None:
        visited = []
    if not node:
        return visited
    
    visited.append(node.value)
    traverse_tree(node.left, visited)
    traverse_tree(node.right, visited)
    return visited

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

# Creating a simple tree
root = Node(1)
root.left = Node(2)
root.right = Node(3)

print(traverse_tree(root))  # First traversal: [1, 2, 3]
print(traverse_tree(root))  # Second traversal: [1, 2, 3] (clean state)
```

Slide 9: Real-World Example: Cache Implementation

A common pitfall occurs when implementing caching mechanisms with default arguments. This example demonstrates how default arguments can create a shared cache across all function calls, leading to potential memory leaks.

```python
def fetch_data(url, cache={}):
    if url in cache:
        return cache[url]
    # Simulate data fetching
    data = f"Data from {url}"
    cache[url] = data
    return data

# First requests
print(fetch_data("api/users"))  # Fetches and caches
print(fetch_data("api/posts"))  # Fetches and caches
# Later requests
print(fetch_data("api/users"))  # Uses shared cache
print(len(fetch_data.__defaults__[0]))  # Shows growing cache size
```

Slide 10: Improved Cache Implementation

A better approach to implementing caching involves class-based implementation or function factories. This ensures proper cache isolation and lifecycle management.

```python
class DataFetcher:
    def __init__(self):
        self.cache = {}
        
    def fetch_data(self, url):
        if url not in self.cache:
            # Simulate data fetching
            data = f"Data from {url}"
            self.cache[url] = data
        return self.cache[url]

# Create separate instances for different contexts
user_fetcher = DataFetcher()
post_fetcher = DataFetcher()

print(user_fetcher.fetch_data("api/users"))
print(post_fetcher.fetch_data("api/posts"))
```

Slide 11: Default Arguments with Datetime Objects

Default arguments with datetime objects can cause confusion in long-running applications or scheduled tasks. This problem manifests when the default timestamp becomes stale but continues to be used across function calls.

```python
from datetime import datetime, timedelta

def create_task(name, deadline=datetime.now() + timedelta(days=7)):
    return {
        'name': name,
        'created_at': datetime.now(),
        'deadline': deadline
    }

# Create tasks at different times
task1 = create_task("Project A")
# Simulate time passing
task2 = create_task("Project B")  # Same deadline as task1!
print(f"Task 1 deadline: {task1['deadline']}")
print(f"Task 2 deadline: {task2['deadline']}")
```

Slide 12: Dynamic Datetime Defaults

The correct implementation should calculate datetime values dynamically within the function to ensure accurate timestamps and deadlines for each function call.

```python
from datetime import datetime, timedelta

def create_task(name, deadline=None):
    current_time = datetime.now()
    if deadline is None:
        deadline = current_time + timedelta(days=7)
    
    return {
        'name': name,
        'created_at': current_time,
        'deadline': deadline
    }

# Create tasks at different times
task1 = create_task("Project A")
# Simulate time passing
task2 = create_task("Project B")  # Gets fresh deadline
print(f"Task 1 deadline: {task1['deadline']}")
print(f"Task 2 deadline: {task2['deadline']}")
```

Slide 13: Default Arguments in API Clients

When building API clients, default arguments for configuration can lead to shared state between different client instances, potentially causing security and functionality issues.

```python
def create_api_client(base_url, headers={}):
    headers.setdefault('Content-Type', 'application/json')
    
    def make_request(endpoint):
        return f"Requesting {base_url}/{endpoint} with headers {headers}"
    
    return make_request

# Creating clients
client1 = create_api_client('api1.example.com')
client2 = create_api_client('api2.example.com')
client1_headers = client1.__closure__[1].cell_contents
client2_headers = client2.__closure__[1].cell_contents
print(id(client1_headers) == id(client2_headers))  # True - shared headers!
```

Slide 14: Secure API Client Implementation

A secure implementation of API clients should ensure configuration isolation between instances and proper handling of default arguments.

```python
def create_api_client(base_url, headers=None):
    if headers is None:
        headers = {'Content-Type': 'application/json'}
    else:
        headers = headers.copy()  # Create a new copy for each instance
    
    def make_request(endpoint, extra_headers=None):
        request_headers = headers.copy()
        if extra_headers:
            request_headers.update(extra_headers)
        return f"Requesting {base_url}/{endpoint} with headers {request_headers}"
    
    return make_request

# Creating clients
client1 = create_api_client('api1.example.com')
client2 = create_api_client('api2.example.com')
print(client1('users'))
print(client2('products', {'Authorization': 'Bearer token'}))
```

Slide 15: Additional Resources

*   Understanding Python's Default Arguments
    *   [https://docs.python.org/3/tutorial/controlflow.html#default-argument-values](https://docs.python.org/3/tutorial/controlflow.html#default-argument-values)
    *   [https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument](https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument)
*   Best Practices for Python Function Arguments
    *   [https://google.github.io/styleguide/pyguide.html](https://google.github.io/styleguide/pyguide.html)
    *   [https://realpython.com/python-kwargs-and-args/](https://realpython.com/python-kwargs-and-args/)
*   Advanced Python Function Design
    *   [https://www.python.org/dev/peps/pep-3102/](https://www.python.org/dev/peps/pep-3102/)
    *   [https://docs.python-guide.org/writing/style/](https://docs.python-guide.org/writing/style/)

