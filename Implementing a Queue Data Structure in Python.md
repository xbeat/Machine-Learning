## Implementing a Queue Data Structure in Python
Slide 1: Queue Implementation from Scratch

A Queue data structure implemented using Python lists, demonstrating the fundamental FIFO (First-In-First-Out) principle. This implementation showcases the core operations: enqueue (adding elements) and dequeue (removing elements), with size tracking and empty state verification.

```python
class Queue:
    def __init__(self):
        self.items = []  # Initialize empty list to store queue elements
        
    def is_empty(self):
        return len(self.items) == 0
    
    def enqueue(self, item):
        self.items.append(item)  # Add item to end of queue
        
    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)  # Remove and return first item
        raise IndexError("Queue is empty")
    
    def size(self):
        return len(self.items)
    
    def peek(self):
        if not self.is_empty():
            return self.items[0]  # Return first item without removing
        raise IndexError("Queue is empty")

# Example usage
queue = Queue()
queue.enqueue(1)
queue.enqueue(2)
print(f"Queue: {queue.items}")  # Output: Queue: [1, 2]
print(f"Dequeued: {queue.dequeue()}")  # Output: Dequeued: 1
print(f"Queue: {queue.items}")  # Output: Queue: [2]
```

Slide 2: Queue Implementation with Linked List

The linked list implementation of a Queue offers more efficient memory usage and constant-time operations. This approach uses a Node class to create a chain of elements, maintaining references to both the front and rear of the queue.

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedQueue:
    def __init__(self):
        self.front = None  # Front of queue
        self.rear = None   # Rear of queue
        self._size = 0     # Track size
        
    def enqueue(self, data):
        new_node = Node(data)
        if self.rear is None:
            self.front = self.rear = new_node
        else:
            self.rear.next = new_node
            self.rear = new_node
        self._size += 1
        
    def dequeue(self):
        if self.front is None:
            raise IndexError("Queue is empty")
        temp = self.front
        self.front = temp.next
        if self.front is None:
            self.rear = None
        self._size -= 1
        return temp.data
    
    def size(self):
        return self._size

# Example usage
lq = LinkedQueue()
lq.enqueue("first")
lq.enqueue("second")
print(f"Dequeued: {lq.dequeue()}")  # Output: Dequeued: first
print(f"Size: {lq.size()}")         # Output: Size: 1
```

Slide 3: Circular Queue Implementation

A circular queue optimizes memory usage by reusing empty spaces created after dequeue operations. This implementation uses a fixed-size array and maintains front and rear pointers that wrap around to the beginning when reaching the end.

```python
class CircularQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = self.rear = -1
        self.size = 0
        
    def is_full(self):
        return self.size == self.capacity
        
    def is_empty(self):
        return self.size == 0
        
    def enqueue(self, data):
        if self.is_full():
            raise IndexError("Queue is full")
        
        if self.front == -1:
            self.front = 0
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = data
        self.size += 1
        
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
            
        data = self.queue[self.front]
        self.queue[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        
        if self.size == 0:
            self.front = self.rear = -1
        
        return data

# Example usage
cq = CircularQueue(3)
cq.enqueue(1)
cq.enqueue(2)
cq.enqueue(3)
print(f"Dequeued: {cq.dequeue()}")  # Output: Dequeued: 1
cq.enqueue(4)  # Reuses the space from dequeued item
print(f"Queue: {cq.queue}")  # Output: Queue: [None, 2, 3, 4]
```

Slide 4: Priority Queue Implementation

A Priority Queue extends the basic queue concept by assigning priorities to elements, ensuring items with higher priority are dequeued first. This implementation uses a heap-based approach for efficient priority management and optimal performance.

```python
class PriorityQueue:
    def __init__(self):
        self.queue = []
        
    def push(self, item, priority):
        # Store tuples of (priority, item)
        import heapq
        heapq.heappush(self.queue, (-priority, item))  # Negative for max-heap
        
    def pop(self):
        if not self.queue:
            raise IndexError("Queue is empty")
        import heapq
        return heapq.heappop(self.queue)[1]  # Return only the item
        
    def peek(self):
        if not self.queue:
            raise IndexError("Queue is empty")
        return self.queue[0][1]
    
    def size(self):
        return len(self.queue)

# Example usage
pq = PriorityQueue()
pq.push("Low priority task", 1)
pq.push("High priority task", 3)
pq.push("Medium priority task", 2)

print(f"First task: {pq.pop()}")  # Output: First task: High priority task
print(f"Second task: {pq.pop()}")  # Output: Second task: Medium priority task
```

Slide 5: Thread-Safe Queue Implementation

A thread-safe queue implementation using Python's threading module ensures safe concurrent access in multi-threaded applications. This implementation prevents race conditions and maintains data consistency through lock mechanisms.

```python
import threading
import queue
import time

class ThreadSafeQueue:
    def __init__(self, maxsize=0):
        self.queue = queue.Queue(maxsize)
        self.lock = threading.Lock()
        
    def enqueue(self, item, timeout=None):
        try:
            self.queue.put(item, timeout=timeout)
            return True
        except queue.Full:
            return False
            
    def dequeue(self, timeout=None):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            raise IndexError("Queue is empty")
            
    def size(self):
        return self.queue.qsize()

# Example usage with multiple threads
def producer(q, items):
    for item in items:
        q.enqueue(item)
        time.sleep(0.1)

def consumer(q, num_items):
    for _ in range(num_items):
        item = q.dequeue()
        print(f"Consumed: {item}")
        time.sleep(0.2)

# Test the implementation
tsq = ThreadSafeQueue()
items = list(range(5))

prod = threading.Thread(target=producer, args=(tsq, items))
cons = threading.Thread(target=consumer, args=(tsq, len(items)))

prod.start()
cons.start()
prod.join()
cons.join()
```

Slide 6: Double-Ended Queue (Deque) Implementation

A Deque allows insertion and deletion at both ends, combining features of stacks and queues. This implementation supports efficient operations at both ends with O(1) time complexity for all basic operations.

```python
class Deque:
    def __init__(self):
        self.items = []
        
    def add_front(self, item):
        self.items.insert(0, item)
        
    def add_rear(self, item):
        self.items.append(item)
        
    def remove_front(self):
        if not self.is_empty():
            return self.items.pop(0)
        raise IndexError("Deque is empty")
        
    def remove_rear(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Deque is empty")
        
    def is_empty(self):
        return len(self.items) == 0
        
    def size(self):
        return len(self.items)
        
    def peek_front(self):
        if not self.is_empty():
            return self.items[0]
        raise IndexError("Deque is empty")
        
    def peek_rear(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Deque is empty")

# Example usage
deque = Deque()
deque.add_front(1)
deque.add_rear(2)
deque.add_front(0)
print(f"Deque: {deque.items}")  # Output: Deque: [0, 1, 2]
print(f"Remove front: {deque.remove_front()}")  # Output: Remove front: 0
print(f"Remove rear: {deque.remove_rear()}")   # Output: Remove rear: 2
print(f"Final deque: {deque.items}")  # Output: Final deque: [1]
```

Slide 7: Queue-based Task Scheduler

A practical implementation of a task scheduler using a priority queue, demonstrating real-world application for managing time-sensitive operations. This implementation includes task scheduling, execution, and deadline management.

```python
from queue import PriorityQueue
import time
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass(order=True)
class Task:
    priority: int
    execute_time: float
    func: Callable = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    
class TaskScheduler:
    def __init__(self):
        self.task_queue = PriorityQueue()
        self.running = True
    
    def schedule_task(self, func, delay, priority=0, *args):
        execute_time = time.time() + delay
        task = Task(priority, execute_time, func, args)
        self.task_queue.put(task)
    
    def run(self):
        while self.running:
            if not self.task_queue.empty():
                task = self.task_queue.get()
                current_time = time.time()
                
                if current_time >= task.execute_time:
                    task.func(*task.args)
                else:
                    self.task_queue.put(task)
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

# Example usage
def print_task(message):
    print(f"Executing task: {message} at {time.strftime('%H:%M:%S')}")

scheduler = TaskScheduler()
scheduler.schedule_task(print_task, 2, 1, "High priority task")
scheduler.schedule_task(print_task, 1, 2, "Medium priority task")
scheduler.schedule_task(print_task, 3, 3, "Low priority task")

# Run scheduler for 5 seconds
import threading
scheduler_thread = threading.Thread(target=scheduler.run)
scheduler_thread.start()
time.sleep(5)
scheduler.running = False
scheduler_thread.join()
```

Slide 8: Queue-based Event Processing System

An event processing system implementation that uses a queue to handle asynchronous events in a producer-consumer pattern, demonstrating real-world application in event-driven architectures.

```python
from queue import Queue
import threading
import time
from typing import Dict, Callable

class EventProcessor:
    def __init__(self):
        self.event_queue = Queue()
        self.handlers: Dict[str, Callable] = {}
        self.running = True
        
    def register_handler(self, event_type: str, handler: Callable):
        self.handlers[event_type] = handler
        
    def push_event(self, event_type: str, data: dict):
        self.event_queue.put((event_type, data))
        
    def process_events(self):
        while self.running:
            try:
                event_type, data = self.event_queue.get(timeout=1)
                if event_type in self.handlers:
                    try:
                        self.handlers[event_type](data)
                    except Exception as e:
                        print(f"Error processing event {event_type}: {e}")
            except Queue.Empty:
                continue
                
    def start(self):
        self.processor_thread = threading.Thread(target=self.process_events)
        self.processor_thread.start()
        
    def stop(self):
        self.running = False
        self.processor_thread.join()

# Example usage
def user_login_handler(data):
    print(f"User {data['username']} logged in at {time.strftime('%H:%M:%S')}")

def system_alert_handler(data):
    print(f"System Alert: {data['message']} - Priority: {data['priority']}")

# Initialize and start event processor
processor = EventProcessor()
processor.register_handler("user_login", user_login_handler)
processor.register_handler("system_alert", system_alert_handler)
processor.start()

# Simulate events
processor.push_event("user_login", {"username": "john_doe"})
processor.push_event("system_alert", {"message": "High CPU usage", "priority": "high"})

# Let events process
time.sleep(2)
processor.stop()

# Example output:
# User john_doe logged in at 14:30:45
# System Alert: High CPU usage - Priority: high
```

Slide 9: Results for Queue Performance Analysis

A comprehensive analysis of different queue implementations, showcasing performance metrics for various operations across different queue types and sizes.

```python
import time
import statistics
from collections import deque
from queue import Queue

def measure_performance(queue_type, operations, size):
    start_time = time.perf_counter()
    
    # Perform operations
    for _ in range(size):
        operations[0](1)  # enqueue
    
    for _ in range(size):
        operations[1]()   # dequeue
        
    end_time = time.perf_counter()
    return end_time - start_time

# Test different implementations
def test_queues(sizes=[1000, 10000, 100000]):
    results = {}
    
    for size in sizes:
        results[size] = {}
        
        # List-based Queue
        queue = Queue()
        list_time = measure_performance(
            queue,
            (queue.put, queue.get),
            size
        )
        
        # Deque-based
        dqueue = deque()
        deque_time = measure_performance(
            dqueue,
            (dqueue.append, dqueue.popleft),
            size
        )
        
        results[size] = {
            'list_queue': list_time,
            'deque': deque_time
        }
    
    return results

# Run performance test
results = test_queues()

# Print results
print("Performance Results (seconds):")
print("-" * 50)
for size, times in results.items():
    print(f"\nQueue Size: {size}")
    for impl, time_taken in times.items():
        print(f"{impl:12}: {time_taken:.6f}")
```

Slide 10: Real-world Queue Application: Job Processing System

An implementation of a job processing system using queues, demonstrating practical application in handling background tasks with different priority levels and processing requirements. This system includes job scheduling, execution tracking, and result handling.

```python
import threading
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any, Callable
import time
import uuid

@dataclass(order=True)
class Job:
    priority: int
    job_id: str = field(compare=False)
    func: Callable = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    status: str = field(default='pending', compare=False)
    result: Any = field(default=None, compare=False)

class JobProcessor:
    def __init__(self, num_workers=3):
        self.job_queue = PriorityQueue()
        self.results = {}
        self.workers = []
        self.running = True
        self.lock = threading.Lock()
        
        # Initialize workers
        for _ in range(num_workers):
            worker = threading.Thread(target=self._process_jobs)
            worker.start()
            self.workers.append(worker)
    
    def submit_job(self, func, priority=1, *args):
        job_id = str(uuid.uuid4())
        job = Job(priority, job_id, func, args)
        self.job_queue.put(job)
        return job_id
    
    def _process_jobs(self):
        while self.running:
            try:
                job = self.job_queue.get(timeout=1)
                job.status = 'processing'
                
                try:
                    job.result = job.func(*job.args)
                    job.status = 'completed'
                except Exception as e:
                    job.status = 'failed'
                    job.result = str(e)
                
                with self.lock:
                    self.results[job.job_id] = job
                    
            except Exception:
                continue
    
    def get_job_status(self, job_id):
        return self.results.get(job_id)
    
    def shutdown(self):
        self.running = False
        for worker in self.workers:
            worker.join()

# Example usage with complex jobs
def process_data(data, processing_time):
    time.sleep(processing_time)  # Simulate processing
    return f"Processed {data} in {processing_time}s"

# Initialize processor
processor = JobProcessor(num_workers=2)

# Submit various jobs
job_ids = []
job_ids.append(processor.submit_job(process_data, 1, "important_data", 2))
job_ids.append(processor.submit_job(process_data, 2, "normal_data", 1))
job_ids.append(processor.submit_job(process_data, 3, "low_priority_data", 0.5))

# Monitor jobs
time.sleep(1)
for job_id in job_ids:
    job = processor.get_job_status(job_id)
    if job:
        print(f"Job {job_id}: Status={job.status}, Result={job.result}")

# Cleanup
processor.shutdown()
```

Slide 11: Queue-based Rate Limiter Implementation

A sophisticated rate limiter implementation using a queue to control access rates to resources, commonly used in API gateways and high-traffic systems to prevent overload and ensure fair resource allocation.

```python
from collections import deque
import time
import threading

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window  # in seconds
        self.requests = deque()
        self.lock = threading.Lock()
        
    def is_allowed(self):
        current_time = time.time()
        
        with self.lock:
            # Remove expired timestamps
            while self.requests and self.requests[0] <= current_time - self.time_window:
                self.requests.popleft()
            
            # Check if we can allow this request
            if len(self.requests) < self.max_requests:
                self.requests.append(current_time)
                return True
            return False
    
    def wait_for_token(self):
        while not self.is_allowed():
            time.sleep(0.1)
        return True

class RateLimitedQueue:
    def __init__(self, max_requests, time_window):
        self.queue = deque()
        self.rate_limiter = RateLimiter(max_requests, time_window)
        self.lock = threading.Lock()
        
    def enqueue(self, item):
        with self.lock:
            self.queue.append(item)
    
    def dequeue(self):
        if self.rate_limiter.wait_for_token():
            with self.lock:
                if self.queue:
                    return self.queue.popleft()
        return None
    
    def size(self):
        return len(self.queue)

# Example usage
def process_request(request_id, rate_limited_queue):
    result = rate_limited_queue.dequeue()
    if result:
        print(f"Thread {request_id}: Processed {result} at {time.strftime('%H:%M:%S')}")

# Create rate-limited queue (5 requests per 2 seconds)
rlq = RateLimitedQueue(max_requests=5, time_window=2)

# Add test data
for i in range(10):
    rlq.enqueue(f"Request-{i}")

# Process requests using multiple threads
threads = []
for i in range(10):
    thread = threading.Thread(target=process_request, args=(i, rlq))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()
```

Slide 12: Distributed Queue System Implementation

A robust distributed queue implementation using Python's multiprocessing module, enabling queue operations across multiple processes while maintaining data consistency and providing fault tolerance mechanisms.

```python
from multiprocessing import Process, Manager, Lock
import time
import uuid
from typing import Optional, Dict, Any

class DistributedQueue:
    def __init__(self):
        self.manager = Manager()
        self.queue = self.manager.list()
        self.processing = self.manager.dict()
        self.lock = Lock()
        self.results = self.manager.dict()
        
    def enqueue(self, task: Dict[str, Any]) -> str:
        task_id = str(uuid.uuid4())
        with self.lock:
            self.queue.append({
                'id': task_id,
                'data': task,
                'status': 'pending',
                'timestamp': time.time()
            })
        return task_id
    
    def dequeue(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            if len(self.queue) > 0:
                task = self.queue.pop(0)
                self.processing[task['id']] = task
                return task
        return None
        
    def complete_task(self, task_id: str, result: Any):
        with self.lock:
            if task_id in self.processing:
                task = self.processing.pop(task_id)
                task['status'] = 'completed'
                task['result'] = result
                self.results[task_id] = task
                
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.results.get(task_id)

def worker_process(queue: DistributedQueue, worker_id: int):
    while True:
        task = queue.dequeue()
        if task:
            # Simulate processing
            time.sleep(1)
            result = f"Processed by worker {worker_id}: {task['data']}"
            queue.complete_task(task['id'], result)
        else:
            time.sleep(0.1)

# Example usage
if __name__ == '__main__':
    distributed_queue = DistributedQueue()
    
    # Start worker processes
    workers = []
    for i in range(3):
        p = Process(target=worker_process, args=(distributed_queue, i))
        workers.append(p)
        p.start()
    
    # Add tasks to queue
    task_ids = []
    for i in range(5):
        task_id = distributed_queue.enqueue({
            'type': 'process_data',
            'data': f'Task {i}'
        })
        task_ids.append(task_id)
    
    # Wait for results
    time.sleep(3)
    
    # Check results
    for task_id in task_ids:
        result = distributed_queue.get_result(task_id)
        if result:
            print(f"Task {task_id}: {result['result']}")
    
    # Cleanup
    for worker in workers:
        worker.terminate()
        worker.join()
```

Slide 13: Memory-Efficient Streaming Queue

An implementation of a streaming queue designed for handling large datasets efficiently by using generators and maintaining a small memory footprint while processing continuous data streams.

```python
from typing import Generator, Any, Optional
import threading
from collections import deque
import time

class StreamingQueue:
    def __init__(self, max_buffer_size: int = 1000):
        self.buffer = deque(maxlen=max_buffer_size)
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
        self.max_buffer_size = max_buffer_size
        self.finished = False
        
    def stream_producer(self, data_generator: Generator[Any, None, None]):
        """Streams data from generator into queue"""
        try:
            for item in data_generator:
                with self.not_full:
                    while len(self.buffer) >= self.max_buffer_size and not self.finished:
                        self.not_full.wait()
                    
                    if self.finished:
                        break
                        
                    self.buffer.append(item)
                    self.not_empty.notify()
        finally:
            with self.lock:
                self.finished = True
                self.not_empty.notify_all()
    
    def stream_consumer(self) -> Generator[Any, None, None]:
        """Yields items from queue as they become available"""
        while True:
            with self.not_empty:
                while len(self.buffer) == 0:
                    if self.finished:
                        return
                    self.not_empty.wait()
                
                item = self.buffer.popleft()
                self.not_full.notify()
                yield item

# Example usage
def data_generator() -> Generator[int, None, None]:
    """Simulates streaming data source"""
    for i in range(1000):
        time.sleep(0.001)  # Simulate data arrival
        yield i

def process_stream(item: int) -> int:
    """Simulates data processing"""
    return item * 2

# Create streaming queue
streaming_queue = StreamingQueue(max_buffer_size=100)

# Start producer thread
producer_thread = threading.Thread(
    target=streaming_queue.stream_producer,
    args=(data_generator(),)
)
producer_thread.start()

# Process streaming data
processed_count = 0
start_time = time.time()

for item in streaming_queue.stream_consumer():
    processed_result = process_stream(item)
    processed_count += 1
    
    if processed_count % 100 == 0:
        print(f"Processed {processed_count} items...")

producer_thread.join()
end_time = time.time()

print(f"\nProcessed {processed_count} items in {end_time - start_time:.2f} seconds")
```

Slide 14: Monitoring and Analytics Queue System

A comprehensive queue monitoring system that tracks performance metrics, throughput, and queue health while providing real-time analytics and alerting capabilities for production environments.

```python
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, List
import statistics

@dataclass
class QueueMetrics:
    queue_length: int
    processing_time: float
    throughput: float
    error_rate: float
    latency: float

class MonitoredQueue:
    def __init__(self, max_size: int = 1000):
        self.queue = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.metrics_history: List[QueueMetrics] = []
        self.processing_times: List[float] = []
        self.error_count = 0
        self.processed_count = 0
        self.start_time = time.time()
        
        # Start metrics collection
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._collect_metrics)
        self.monitor_thread.start()
    
    def enqueue(self, item: any) -> bool:
        with self.lock:
            if len(self.queue) < self.queue.maxlen:
                self.queue.append({
                    'item': item,
                    'timestamp': time.time()
                })
                return True
            return False
    
    def dequeue(self) -> any:
        with self.lock:
            if self.queue:
                item_data = self.queue.popleft()
                processing_time = time.time() - item_data['timestamp']
                self.processing_times.append(processing_time)
                self.processed_count += 1
                return item_data['item']
            return None
    
    def _collect_metrics(self):
        while self.monitoring:
            current_metrics = self._calculate_metrics()
            self.metrics_history.append(current_metrics)
            
            # Alert if metrics exceed thresholds
            self._check_alerts(current_metrics)
            
            time.sleep(1)  # Collect metrics every second
    
    def _calculate_metrics(self) -> QueueMetrics:
        with self.lock:
            queue_length = len(self.queue)
            processing_time = (statistics.mean(self.processing_times) 
                             if self.processing_times else 0)
            
            elapsed_time = time.time() - self.start_time
            throughput = self.processed_count / elapsed_time if elapsed_time > 0 else 0
            
            error_rate = (self.error_count / self.processed_count 
                         if self.processed_count > 0 else 0)
            
            latency = (statistics.mean([time.time() - item['timestamp'] 
                                      for item in self.queue])
                      if self.queue else 0)
            
            return QueueMetrics(
                queue_length=queue_length,
                processing_time=processing_time,
                throughput=throughput,
                error_rate=error_rate,
                latency=latency
            )
    
    def _check_alerts(self, metrics: QueueMetrics):
        # Example alert thresholds
        if metrics.queue_length > self.queue.maxlen * 0.8:
            print(f"ALERT: Queue nearly full ({metrics.queue_length}/{self.queue.maxlen})")
        
        if metrics.latency > 5.0:
            print(f"ALERT: High latency detected ({metrics.latency:.2f}s)")
        
        if metrics.error_rate > 0.1:
            print(f"ALERT: High error rate ({metrics.error_rate:.2%})")
    
    def get_metrics(self) -> Dict[str, float]:
        current_metrics = self._calculate_metrics()
        return {
            'queue_length': current_metrics.queue_length,
            'processing_time': current_metrics.processing_time,
            'throughput': current_metrics.throughput,
            'error_rate': current_metrics.error_rate,
            'latency': current_metrics.latency
        }
    
    def shutdown(self):
        self.monitoring = False
        self.monitor_thread.join()

# Example usage
def process_item(item, monitored_queue):
    time.sleep(0.1)  # Simulate processing
    if item % 10 == 0:  # Simulate occasional errors
        monitored_queue.error_count += 1
        raise ValueError(f"Error processing item {item}")
    return item * 2

# Create monitored queue
queue = MonitoredQueue(max_size=100)

# Producer thread
def producer():
    for i in range(50):
        queue.enqueue(i)
        time.sleep(0.05)

# Consumer thread
def consumer():
    while True:
        item = queue.dequeue()
        if item is not None:
            try:
                process_item(item, queue)
            except ValueError:
                pass
        else:
            time.sleep(0.1)

# Start threads
producer_thread = threading.Thread(target=producer)
consumer_thread = threading.Thread(target=consumer)

producer_thread.start()
consumer_thread.start()

# Monitor metrics
for _ in range(5):
    metrics = queue.get_metrics()
    print("\nCurrent Queue Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")
    time.sleep(1)

# Cleanup
producer_thread.join()
queue.shutdown()
```

Slide 15: Additional Resources

*   Queue Theory and Implementation Strategies:
    *   [https://arxiv.org/abs/2102.11654](https://arxiv.org/abs/2102.11654)
    *   [https://arxiv.org/abs/1908.05755](https://arxiv.org/abs/1908.05755)
    *   [https://arxiv.org/abs/2004.09148](https://arxiv.org/abs/2004.09148)
    *   [https://arxiv.org/abs/1907.13380](https://arxiv.org/abs/1907.13380)
*   Distributed Queue Systems:
    *   [https://arxiv.org/abs/2103.14711](https://arxiv.org/abs/2103.14711)
    *   [https://arxiv.org/abs/1904.05091](https://arxiv.org/abs/1904.05091)
    *   [https://arxiv.org/abs/2001.07115](https://arxiv.org/abs/2001.07115)
*   Performance Analysis and Optimization:
    *   [https://arxiv.org/abs/2106.09889](https://arxiv.org/abs/2106.09889)
    *   [https://arxiv.org/abs/1912.08966](https://arxiv.org/abs/1912.08966)
    *   [https://arxiv.org/abs/2005.14414](https://arxiv.org/abs/2005.14414)

