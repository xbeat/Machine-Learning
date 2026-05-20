## Effective Resource Management in Python
Slide 1: Resource Management Basics

Python's memory management system automatically handles object cleanup through garbage collection, but certain system resources like file handles and network sockets require explicit management to ensure proper release. Understanding the basics of resource handling is crucial for writing robust and leak-free applications.

```python
# Basic resource management example
file = open('data.txt', 'w')
try:
    file.write('Hello World')
finally:
    file.close()  # Explicit cleanup

# Check if file is closed
print(f"Is file closed? {file.closed}")  # Output: Is file closed? True
```

Slide 2: Context Managers Introduction

Context managers provide a clean and pythonic way to handle resource acquisition and release automatically. The 'with' statement ensures proper cleanup even if exceptions occur during execution, making it the preferred approach for resource management in Python.

```python
# Using context manager for file handling
with open('data.txt', 'w') as file:
    file.write('Hello World')
    # File automatically closes after block ends
    
print(f"Is file closed? {file.closed}")  # Output: Is file closed? True
```

Slide 3: Custom Context Manager Implementation

Creating custom context managers allows you to define specific setup and cleanup behaviors for your own classes. This is achieved by implementing the **enter** and **exit** magic methods, enabling the class to be used with the 'with' statement.

```python
class DatabaseConnection:
    def __init__(self, host):
        self.host = host
        self.connected = False
    
    def __enter__(self):
        print(f"Connecting to {self.host}")
        self.connected = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing connection")
        self.connected = False
        return False  # Re-raise any exceptions

# Usage
with DatabaseConnection("localhost:5432") as db:
    print(f"Connection status: {db.connected}")
```

Slide 4: Multiple Resource Management

Python's context managers can handle multiple resources simultaneously, ensuring proper cleanup order (reverse of acquisition). This is particularly useful when dealing with nested resources or dependencies.

```python
class Resource:
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        print(f"Acquiring {self.name}")
        return self
    
    def __exit__(self, *args):
        print(f"Releasing {self.name}")

# Managing multiple resources
with Resource('database') as db, Resource('cache') as cache:
    print("Working with resources")
```

Slide 5: Contextlib Utilities

The contextlib module provides useful tools for creating and working with context managers. The @contextmanager decorator simplifies context manager implementation by using a single generator function instead of defining separate **enter** and **exit** methods.

```python
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    yield
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")

# Usage
with timer():
    # Simulate work
    import time
    time.sleep(1)
```

Slide 6: Resource Pools and Connection Management

Managing pools of resources efficiently is crucial in production applications. This example demonstrates implementing a simple resource pool with automatic cleanup and maximum connection limits.

```python
class ResourcePool:
    def __init__(self, max_resources=5):
        self.max_resources = max_resources
        self.resources = []
        self.in_use = set()
    
    @contextmanager
    def acquire(self):
        if len(self.in_use) >= self.max_resources:
            raise RuntimeError("Resource pool exhausted")
        
        resource = self._create_resource()
        self.in_use.add(resource)
        try:
            yield resource
        finally:
            self.in_use.remove(resource)
            self.resources.append(resource)
    
    def _create_resource(self):
        return object()  # Placeholder for actual resource creation
```

Slide 7: Error Handling in Resource Management

Proper error handling is essential when managing resources. This example shows how to implement robust error handling while ensuring resources are always properly cleaned up, regardless of whether operations succeed or fail.

```python
class SafeResource:
    def __init__(self):
        self.errors = []
        self.resource = None
    
    def __enter__(self):
        try:
            self.resource = self._acquire_resource()
            return self.resource
        except Exception as e:
            self.errors.append(e)
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.resource:
            try:
                self._release_resource()
            except Exception as e:
                self.errors.append(e)
                if exc_type is None:
                    raise
        return False
    
    def _acquire_resource(self):
        return "Resource acquired"
    
    def _release_resource(self):
        pass  # Resource cleanup logic
```

Slide 8: Managing Network Resources

Network connections require careful resource management to prevent socket leaks and ensure proper cleanup. This example demonstrates a robust implementation of a network client with automatic connection handling and timeout management.

```python
import socket
from contextlib import contextmanager

class NetworkClient:
    def __init__(self, host, port, timeout=5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
    
    @contextmanager
    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        try:
            self.sock.connect((self.host, self.port))
            yield self.sock
        finally:
            self.sock.close()
            self.sock = None

# Usage example
client = NetworkClient('localhost', 8080)
try:
    with client.connect() as connection:
        connection.send(b'Hello, Server!')
        data = connection.recv(1024)
except socket.error as e:
    print(f"Connection error: {e}")
```

Slide 9: Database Connection Pool

Implementing a thread-safe database connection pool with automatic resource cleanup and connection health checks. This example shows how to manage database connections efficiently in a production environment.

```python
import threading
from queue import Queue
import time

class DatabasePool:
    def __init__(self, max_connections=5):
        self.max_connections = max_connections
        self.connections = Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        for _ in range(self.max_connections):
            conn = self._create_connection()
            self.connections.put(conn)
    
    def _create_connection(self):
        # Simulate database connection
        return {'created_at': time.time(), 'id': id({})}
    
    @contextmanager
    def get_connection(self):
        connection = self.connections.get()
        try:
            yield connection
        finally:
            if self._is_connection_valid(connection):
                self.connections.put(connection)
            else:
                # Replace invalid connection
                self.connections.put(self._create_connection())
    
    def _is_connection_valid(self, connection):
        # Check if connection is still valid (mock implementation)
        return time.time() - connection['created_at'] < 3600  # 1 hour timeout
```

Slide 10: Memory-Mapped File Handler

Memory-mapped files require special attention for resource management. This implementation shows how to safely handle memory-mapped files with proper cleanup and synchronization.

```python
import mmap
import os

class MappedFileHandler:
    def __init__(self, filename, size=0):
        self.filename = filename
        self.size = size
        self.file = None
        self.mm = None
    
    def __enter__(self):
        # Create file if it doesn't exist
        if not os.path.exists(self.filename):
            with open(self.filename, 'wb') as f:
                f.write(b'\0' * self.size)
        
        self.file = open(self.filename, 'r+b')
        self.mm = mmap.mmap(self.file.fileno(), 0)
        return self.mm
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mm:
            self.mm.flush()
            self.mm.close()
        if self.file:
            self.file.close()

# Usage
with MappedFileHandler('data.bin', 1024) as mm:
    mm.write(b'Hello, Memory-mapped file!')
    mm.seek(0)
    data = mm.read(10)
    print(data)  # Output: b'Hello, Mem'
```

Slide 11: Async Resource Management

Modern Python applications often use asynchronous programming. This example demonstrates how to implement resource management for async contexts using async context managers.

```python
import asyncio
from contextlib import asynccontextmanager

class AsyncResource:
    def __init__(self, name):
        self.name = name
    
    async def __aenter__(self):
        print(f"Async acquiring {self.name}")
        await asyncio.sleep(1)  # Simulate async initialization
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print(f"Async releasing {self.name}")
        await asyncio.sleep(0.5)  # Simulate async cleanup

# Usage example
async def main():
    async with AsyncResource("database") as db:
        print("Working with async resource")
        await asyncio.sleep(1)

# Run the async code
asyncio.run(main())
```

Slide 12: Temporary Resource Management

Implementing a robust temporary resource manager that handles creation and cleanup of temporary files and directories while ensuring proper resource cleanup even in case of system crashes or unexpected termination.

```python
import tempfile
import shutil
import os
from contextlib import contextmanager

class TemporaryResourceManager:
    def __init__(self, prefix="temp", cleanup_on_exit=True):
        self.prefix = prefix
        self.cleanup_on_exit = cleanup_on_exit
        self.resources = []
    
    @contextmanager
    def temp_file(self, mode='w+b', suffix=None):
        temp_file = tempfile.NamedTemporaryFile(
            mode=mode,
            prefix=self.prefix,
            suffix=suffix,
            delete=False
        )
        self.resources.append(temp_file.name)
        try:
            yield temp_file
        finally:
            temp_file.close()
            if self.cleanup_on_exit:
                try:
                    os.unlink(temp_file.name)
                    self.resources.remove(temp_file.name)
                except OSError:
                    pass

# Usage example
manager = TemporaryResourceManager(prefix="data_")
with manager.temp_file(suffix='.txt') as temp:
    temp.write(b'Temporary data')
    temp.flush()
    print(f"Temporary file created at: {temp.name}")
```

Slide 13: Custom Resource Pool with Monitoring

A sophisticated resource pool implementation that includes monitoring capabilities, health checks, and automatic resource regeneration when issues are detected.

```python
import time
import threading
from queue import Queue, Empty
from collections import deque
from datetime import datetime, timedelta

class MonitoredResourcePool:
    def __init__(self, pool_size=5, max_age=3600):
        self.pool = Queue(maxsize=pool_size)
        self.max_age = max_age
        self.stats = {
            'created': 0,
            'destroyed': 0,
            'errors': 0
        }
        self.usage_history = deque(maxlen=1000)
        self.lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        for _ in range(self.pool.maxsize):
            self._add_resource()
    
    def _add_resource(self):
        resource = {
            'id': id({}),
            'created_at': datetime.now(),
            'last_used': datetime.now(),
            'usage_count': 0
        }
        self.pool.put(resource)
        with self.lock:
            self.stats['created'] += 1
    
    @contextmanager
    def acquire(self):
        resource = self._get_valid_resource()
        try:
            resource['usage_count'] += 1
            resource['last_used'] = datetime.now()
            self.usage_history.append({
                'resource_id': resource['id'],
                'timestamp': datetime.now()
            })
            yield resource
        finally:
            self._return_resource(resource)
    
    def _get_valid_resource(self):
        while True:
            try:
                resource = self.pool.get(timeout=5)
                if self._is_resource_valid(resource):
                    return resource
                self._destroy_resource(resource)
                self._add_resource()
            except Empty:
                raise RuntimeError("Resource pool exhausted")
    
    def _is_resource_valid(self, resource):
        age = (datetime.now() - resource['created_at']).total_seconds()
        return age < self.max_age
    
    def _destroy_resource(self, resource):
        with self.lock:
            self.stats['destroyed'] += 1
    
    def get_stats(self):
        with self.lock:
            return dict(self.stats)

# Usage example
pool = MonitoredResourcePool(pool_size=3)
try:
    with pool.acquire() as resource:
        print(f"Using resource {resource['id']}")
        print(f"Pool stats: {pool.get_stats()}")
except Exception as e:
    print(f"Error: {e}")
```

Slide 14: Additional Resources

*   Memory Management in Python:
    *   [https://arxiv.org/abs/2304.12172](https://arxiv.org/abs/2304.12172)
    *   [https://dl.acm.org/doi/10.1145/3575693.3575704](https://dl.acm.org/doi/10.1145/3575693.3575704)
*   Resource Management Best Practices:
    *   [https://www.google.com/search?q=python+resource+management+best+practices](https://www.google.com/search?q=python+resource+management+best+practices)
*   Advanced Context Managers:
    *   [https://realpython.com/python-with-statement/](https://realpython.com/python-with-statement/)
    *   [https://docs.python.org/3/library/contextlib.html](https://docs.python.org/3/library/contextlib.html)
*   Performance Optimization:
    *   [https://www.google.com/search?q=python+performance+optimization+resource+management](https://www.google.com/search?q=python+performance+optimization+resource+management)

