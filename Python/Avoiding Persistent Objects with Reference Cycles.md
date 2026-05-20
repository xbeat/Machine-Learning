## Avoiding Persistent Objects with Reference Cycles
Slide 1: Understanding Reference Cycles

Reference cycles occur when two or more objects contain references to each other, creating a circular dependency that prevents Python's reference counting mechanism from properly identifying when objects are no longer needed and can be garbage collected.

```python
import gc

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

def create_cycle():
    # Create two nodes that reference each other
    node1 = Node(1)
    node2 = Node(2)
    
    # Create the cycle
    node1.next = node2
    node2.next = node1
    
    # Both objects go out of scope when function returns
    # but remain in memory due to circular reference

create_cycle()
# Objects still exist in memory
print(f"Garbage collector stats before collection: {gc.get_count()}")
gc.collect()  # Force garbage collection
print(f"Garbage collector stats after collection: {gc.get_count()}")
```

Slide 2: Memory Leak Detection

Python provides tools to help identify potential memory leaks caused by reference cycles. The gc module offers functions to inspect objects tracked by the garbage collector and find reference cycles that might indicate memory leaks.

```python
import gc
import sys

class LeakExample:
    def __init__(self):
        self.ref = None

def find_cycles():
    # Create objects that reference each other
    obj1 = LeakExample()
    obj2 = LeakExample()
    obj1.ref = obj2
    obj2.ref = obj1
    
    # Get all objects tracked by gc
    gc.set_debug(gc.DEBUG_LEAK)
    print("Objects in garbage collector:")
    for obj in gc.get_objects():
        if isinstance(obj, LeakExample):
            print(f"Found LeakExample object: {sys.getrefcount(obj)} references")
    
    # Clean up
    gc.collect()

find_cycles()
```

Slide 3: Weak References

Weak references provide a way to reference an object without increasing its reference count, helping to prevent reference cycles and memory leaks while still maintaining access to the object if it exists.

```python
import weakref
import gc

class Resource:
    def __init__(self, name):
        self.name = name
    
    def __del__(self):
        print(f"Resource {self.name} destroyed")

def demonstrate_weak_reference():
    # Create a resource and a weak reference to it
    resource = Resource("example")
    weak_ref = weakref.ref(resource)
    
    # Access the object through weak reference
    print(f"Resource still exists: {weak_ref() is not None}")
    
    # Delete the strong reference
    del resource
    gc.collect()  # Force garbage collection
    
    # Weak reference now returns None
    print(f"Resource still exists: {weak_ref() is not None}")

demonstrate_weak_reference()
```

Slide 4: Context Managers for Resource Management

Context managers provide a clean way to handle resource acquisition and release, helping to prevent memory leaks by ensuring resources are properly cleaned up even if exceptions occur.

```python
from contextlib import contextmanager
import gc

class DatabaseConnection:
    def __init__(self, db_name):
        self.db_name = db_name
        print(f"Opening connection to {db_name}")
    
    def close(self):
        print(f"Closing connection to {self.db_name}")

@contextmanager
def managed_db_connection(db_name):
    connection = DatabaseConnection(db_name)
    try:
        yield connection
    finally:
        connection.close()
        gc.collect()  # Ensure cleanup

# Usage example
with managed_db_connection("example_db") as conn:
    print("Performing database operations")
# Connection automatically closed after block
```

Slide 5: Circular References in Class Hierarchies

Complex class hierarchies can inadvertently create circular references through parent-child relationships or event handling systems. Understanding these patterns helps in designing better class structures that avoid memory leaks.

```python
class Parent:
    def __init__(self):
        self.children = []
    
    def add_child(self, child):
        self.children.append(child)
        child.parent = self  # Creates circular reference

class Child:
    def __init__(self):
        self.parent = None

def demonstrate_hierarchy_leak():
    parent = Parent()
    child1 = Child()
    child2 = Child()
    
    # Create circular references
    parent.add_child(child1)
    parent.add_child(child2)
    
    print(f"Parent has {len(parent.children)} children")
    print(f"Child1's parent is same as parent: {child1.parent is parent}")
    
    return id(parent), id(child1)  # Return IDs to track objects

# Create and track objects
obj_ids = demonstrate_hierarchy_leak()
print(f"Objects still exist after function return: {any(id(obj) == obj_ids[0] for obj in gc.get_objects())}")
gc.collect()  # Force cleanup
```

Slide 6: Breaking Circular References with **del**

The **del** method can be used to properly clean up resources and break circular references when an object is about to be garbage collected, though it should be used with caution.

```python
class ResourceManager:
    def __init__(self, name):
        self.name = name
        self.related = None
        print(f"Created {self.name}")
    
    def __del__(self):
        print(f"Cleaning up {self.name}")
        # Break circular reference
        if hasattr(self, 'related'):
            self.related = None

def create_circular_reference():
    # Create objects that reference each other
    res1 = ResourceManager("Resource 1")
    res2 = ResourceManager("Resource 2")
    
    # Create circular reference
    res1.related = res2
    res2.related = res1
    
    print("Objects created and linked")
    return res1, res2

# Create and immediately discard references
r1, r2 = create_circular_reference()
del r1, r2
gc.collect()  # Force garbage collection
```

Slide 7: Memory Profiling Techniques

Understanding memory usage patterns is crucial for identifying and fixing memory leaks. Python's memory\_profiler module provides tools to analyze memory consumption of code blocks.

```python
from memory_profiler import profile
import gc

@profile
def memory_leak_example():
    # Create a large list that could be leaked
    large_data = list(range(1000000))
    
    class Container:
        def __init__(self, data):
            self.data = data
            self.self_ref = self  # Create self-reference
    
    # Create container with large data
    container = Container(large_data)
    
    # Delete reference but keep cycle
    del large_data
    
    # Force garbage collection
    gc.collect()
    
    return container

# Run profiled function
result = memory_leak_example()
del result
gc.collect()
```

Slide 8: Debugging Reference Cycles

The garbage collector module provides debugging tools to help identify objects involved in reference cycles and understand their relationships.

```python
import gc
import pprint

class CyclicObject:
    def __init__(self, name):
        self.name = name
        self.ref = None
    
    def __repr__(self):
        return f"CyclicObject({self.name})"

def debug_reference_cycles():
    # Create cyclic references
    obj1 = CyclicObject("A")
    obj2 = CyclicObject("B")
    obj1.ref = obj2
    obj2.ref = obj1
    
    # Enable garbage collector debugging
    gc.set_debug(gc.DEBUG_SAVEALL)
    
    # Collect and analyze garbage
    gc.collect()
    
    # Print objects in garbage
    print("Objects in garbage:")
    pprint.pprint(gc.garbage)
    
    # Show referrers
    print("\nReferrers for obj1:")
    pprint.pprint(gc.get_referrers(obj1))

debug_reference_cycles()
```

Slide 9: Automated Memory Leak Detection

Implementing an automated memory leak detection system helps identify potential memory issues during development and testing phases by tracking object creation and destruction patterns over time.

```python
import gc
import weakref
import time
from collections import defaultdict

class MemoryLeakDetector:
    def __init__(self):
        self.object_tracker = defaultdict(int)
        self.creation_times = {}
    
    def track_object(self, obj):
        obj_id = id(obj)
        self.object_tracker[obj.__class__.__name__] += 1
        self.creation_times[obj_id] = time.time()
        
        # Create weak reference with callback
        weakref.finalize(obj, self._object_deleted, obj_id, obj.__class__.__name__)
    
    def _object_deleted(self, obj_id, class_name):
        self.object_tracker[class_name] -= 1
        del self.creation_times[obj_id]
    
    def report(self):
        print("\nMemory Leak Report:")
        for class_name, count in self.object_tracker.items():
            print(f"{class_name}: {count} instances")
        
        # Show objects that lived too long
        current_time = time.time()
        old_objects = {obj_id: age for obj_id, creation_time 
                      in self.creation_times.items()
                      if (age := current_time - creation_time) > 5}
        
        if old_objects:
            print("\nPotential leaks (objects older than 5 seconds):")
            for obj_id, age in old_objects.items():
                print(f"Object {obj_id}: {age:.2f} seconds old")

# Example usage
detector = MemoryLeakDetector()

class LeakyClass:
    def __init__(self):
        self.data = [1] * 1000000  # Large data
        self.self_ref = self

# Create and track objects
objects = []
for _ in range(3):
    obj = LeakyClass()
    detector.track_object(obj)
    objects.append(obj)

time.sleep(6)  # Wait to simulate long-lived objects
detector.report()
```

Slide 10: Reference Counting vs. Garbage Collection

Python uses a hybrid memory management approach combining reference counting with generational garbage collection. Understanding both mechanisms is crucial for writing memory-efficient code.

```python
import sys
import gc
import time

class ReferenceExample:
    def __init__(self, name):
        self.name = name
        self.data = [0] * 100000  # Large data to make memory usage visible
    
    def __del__(self):
        print(f"Destroying {self.name}")

def demonstrate_memory_management():
    # Disable automatic garbage collection to observe behavior
    gc.disable()
    print("Garbage collection disabled")
    
    # Create objects and check reference counts
    obj1 = ReferenceExample("Object 1")
    print(f"Reference count for obj1: {sys.getrefcount(obj1) - 1}")  # -1 for getrefcount's temporary reference
    
    # Create reference cycle
    obj2 = ReferenceExample("Object 2")
    obj1.ref = obj2
    obj2.ref = obj1
    
    print("\nAfter creating reference cycle:")
    print(f"Objects tracked by GC: {len(gc.get_objects())}")
    
    # Delete references and observe behavior
    del obj1, obj2
    print("\nAfter deleting references:")
    print(f"Garbage collector generations: {gc.get_count()}")
    
    # Re-enable and run garbage collection
    gc.enable()
    collected = gc.collect()
    print(f"\nObjects collected by GC: {collected}")

demonstrate_memory_management()
```

Slide 11: Real-world Example: Memory Management in Web Scraping

Web scraping often involves handling large amounts of data and multiple resources, making proper memory management crucial for long-running scraping tasks.

```python
import gc
import weakref
from datetime import datetime
import time

class WebScraperCache:
    def __init__(self, max_size=1000):
        self._cache = {}
        self._max_size = max_size
        self._access_times = {}
        
    def __setitem__(self, key, value):
        if len(self._cache) >= self._max_size:
            self._cleanup_old_entries()
        
        # Store weak reference to value
        self._cache[key] = weakref.ref(value)
        self._access_times[key] = datetime.now()
    
    def __getitem__(self, key):
        if key in self._cache:
            value = self._cache[key]()
            if value is not None:
                self._access_times[key] = datetime.now()
                return value
            else:
                # Remove dead weak reference
                del self._cache[key]
                del self._access_times[key]
        raise KeyError(key)
    
    def _cleanup_old_entries(self):
        current_time = datetime.now()
        old_keys = [
            key for key, access_time in self._access_times.items()
            if (current_time - access_time).seconds > 3600
        ]
        
        for key in old_keys:
            del self._cache[key]
            del self._access_times[key]
        
        # Force garbage collection after cleanup
        gc.collect()

# Example usage
class WebPage:
    def __init__(self, url, content):
        self.url = url
        self.content = content

# Simulate web scraping with cache
cache = WebScraperCache(max_size=5)

# Add pages to cache
for i in range(6):
    page = WebPage(f"http://example.com/page{i}", f"Content {i}")
    cache[f"page{i}"] = page
    time.sleep(0.1)  # Simulate delay between requests

# Access some pages
try:
    print(f"Page 1 content: {cache['page1'].content}")
    print(f"Page 4 content: {cache['page4'].content}")
except KeyError as e:
    print(f"Page not found in cache: {e}")
```

Slide 12: Preventing Memory Leaks in Event Systems

Event-driven architectures often create hidden reference cycles through event handlers and callbacks. Implementing proper cleanup mechanisms is essential to prevent memory leaks in such systems.

```python
class EventEmitter:
    def __init__(self):
        self._events = {}
        self._listener_count = 0
    
    def on(self, event, callback):
        if event not in self._events:
            self._events[event] = []
        # Store callback as weak reference
        self._events[event].append(weakref.ref(callback))
        self._listener_count += 1
        
    def emit(self, event, *args, **kwargs):
        if event in self._events:
            # Filter out dead references and call remaining callbacks
            active_callbacks = []
            for cb_ref in self._events[event]:
                callback = cb_ref()
                if callback is not None:
                    active_callbacks.append(cb_ref)
                    callback(*args, **kwargs)
                else:
                    self._listener_count -= 1
            self._events[event] = active_callbacks
    
    def cleanup(self):
        for event in list(self._events.keys()):
            # Remove empty event lists
            if not any(cb_ref() for cb_ref in self._events[event]):
                del self._events[event]
        gc.collect()

# Example usage
def create_event_system():
    emitter = EventEmitter()
    
    def event_handler(data):
        print(f"Handling event with data: {data}")
    
    # Register handler
    emitter.on("test", event_handler)
    
    # Emit event
    emitter.emit("test", "sample data")
    
    # Cleanup
    emitter.cleanup()
    
    return emitter

# Test the event system
event_system = create_event_system()
```

Slide 13: Memory-Efficient Data Structures

Implementing memory-efficient data structures using **slots** and proper cleanup methods helps reduce memory usage and prevent memory leaks in large-scale applications.

```python
class MemoryEfficientNode:
    __slots__ = ['value', 'next', '__weakref__']
    
    def __init__(self, value):
        self.value = value
        self.next = None

class MemoryEfficientLinkedList:
    def __init__(self):
        self.head = None
        self._size = 0
        self._node_refs = []
    
    def append(self, value):
        new_node = MemoryEfficientNode(value)
        # Store weak reference to node
        self._node_refs.append(weakref.ref(new_node))
        
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        
        self._size += 1
    
    def cleanup_dead_refs(self):
        # Remove dead references
        self._node_refs = [ref for ref in self._node_refs if ref() is not None]
    
    def __len__(self):
        return self._size
    
    def __del__(self):
        # Break reference cycle in linked list
        current = self.head
        while current:
            next_node = current.next
            current.next = None
            current = next_node
        self.cleanup_dead_refs()

# Example usage
def test_memory_efficient_list():
    linked_list = MemoryEfficientLinkedList()
    
    # Add elements
    for i in range(1000):
        linked_list.append(i)
    
    print(f"List size: {len(linked_list)}")
    print(f"Active node references: {len(linked_list._node_refs)}")
    
    # Force cleanup
    linked_list.cleanup_dead_refs()
    gc.collect()
    
    return linked_list

test_list = test_memory_efficient_list()
```

Slide 14: Additional Resources

*   Memory Management in Python - ArXiv Paper
    *   [https://arxiv.org/abs/2304.12578](https://arxiv.org/abs/2304.12578)
*   Efficient Memory Usage in Large-Scale Python Applications
    *   [https://arxiv.org/abs/2301.09645](https://arxiv.org/abs/2301.09645)
*   Reference Counting and Garbage Collection Algorithms
    *   [https://arxiv.org/abs/2203.08398](https://arxiv.org/abs/2203.08398)

Additional search terms for Google:

*   "Python memory management best practices"
*   "Detecting and fixing memory leaks in Python"
*   "Memory profiling tools for Python applications"

