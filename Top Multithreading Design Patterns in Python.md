## Top Multithreading Design Patterns in Python
Slide 1: Producer-Consumer Pattern Implementation

The Producer-Consumer pattern represents a classic synchronization problem where producers generate data and consumers process it using a shared buffer. This implementation uses Python's threading module and Queue data structure to manage concurrent access and prevent race conditions.

```python
import threading
import queue
import time
import random

class ProducerConsumer:
    def __init__(self, buffer_size=5):
        self.buffer = queue.Queue(buffer_size)
        self.data_produced = 0
        
    def producer(self, items):
        for i in range(items):
            time.sleep(random.uniform(0.1, 0.3))
            self.buffer.put(f'Item {i}')
            print(f'Produced Item {i}')
            self.data_produced += 1
            
    def consumer(self):
        while True:
            if self.buffer.empty() and self.data_produced == 5:
                break
            if not self.buffer.empty():
                item = self.buffer.get()
                print(f'Consumed {item}')
                time.sleep(random.uniform(0.2, 0.5))
                self.buffer.task_done()

# Usage Example
pc = ProducerConsumer()
producer = threading.Thread(target=pc.producer, args=(5,))
consumer = threading.Thread(target=pc.consumer)

producer.start()
consumer.start()
producer.join()
consumer.join()
```

Slide 2: Thread Pool Pattern Implementation

The Thread Pool pattern maintains a collection of reusable worker threads for executing tasks efficiently. This implementation creates a custom thread pool executor that manages task distribution and thread lifecycle, demonstrating resource optimization for concurrent operations.

```python
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

def worker_task(task_id):
    print(f'Processing task {task_id}')
    # Simulate work
    time.sleep(random.uniform(0.5, 1.0))
    return f'Result from task {task_id}'

class CustomThreadPool:
    def __init__(self, num_threads):
        self.tasks = queue.Queue()
        self.results = {}
        self.workers = []
        
        for _ in range(num_threads):
            worker = threading.Thread(target=self._worker_thread)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            
    def _worker_thread(self):
        while True:
            task_id, func, args = self.tasks.get()
            if task_id is None:
                break
            result = func(*args)
            self.results[task_id] = result
            self.tasks.task_done()
            
    def submit(self, task_id, func, *args):
        self.tasks.put((task_id, func, args))
        
    def shutdown(self):
        for _ in self.workers:
            self.tasks.put((None, None, None))
        for worker in self.workers:
            worker.join()

# Usage Example
pool = CustomThreadPool(3)
for i in range(5):
    pool.submit(i, worker_task, i)
time.sleep(3)
pool.shutdown()
```

Slide 3: Futures and Promises Pattern Implementation

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any

class Future:
    def __init__(self):
        self._result = None
        self._is_completed = False
        self._condition = threading.Condition()
        
    def set_result(self, result):
        with self._condition:
            self._result = result
            self._is_completed = True
            self._condition.notify_all()
            
    def get_result(self):
        with self._condition:
            while not self._is_completed:
                self._condition.wait()
            return self._result

class Promise:
    def __init__(self, computation):
        self.future = Future()
        self.computation = computation
        thread = threading.Thread(target=self._execute)
        thread.start()
        
    def _execute(self):
        result = self.computation()
        self.future.set_result(result)
        
    def get_result(self):
        return self.future.get_result()

# Example usage
def long_running_task():
    time.sleep(2)
    return "Task completed"

promise = Promise(long_running_task)
result = promise.get_result()
print(result)
```

Slide 4: Monitor Object Pattern Implementation

The Monitor Object pattern provides synchronized access to an object's methods, ensuring thread safety through mutual exclusion. This implementation demonstrates a thread-safe counter class using Python's threading module to prevent concurrent access conflicts.

```python
import threading
import time

class ThreadSafeCounter:
    def __init__(self):
        self._counter = 0
        self._lock = threading.RLock()
        
    def increment(self):
        with self._lock:
            self._counter += 1
            # Simulate some work
            time.sleep(0.1)
            return self._counter
            
    def decrement(self):
        with self._lock:
            self._counter -= 1
            time.sleep(0.1)
            return self._counter
            
    def get_value(self):
        with self._lock:
            return self._counter

def worker(counter, inc=True):
    for _ in range(3):
        if inc:
            print(f'Increment: {counter.increment()}')
        else:
            print(f'Decrement: {counter.decrement()}')

# Usage Example
counter = ThreadSafeCounter()
t1 = threading.Thread(target=worker, args=(counter, True))
t2 = threading.Thread(target=worker, args=(counter, False))

t1.start()
t2.start()
t1.join()
t2.join()
```

Slide 5: Barrier Pattern with Scientific Computing

The Barrier pattern synchronizes multiple threads at specific computation points, crucial for parallel scientific calculations. This implementation shows a parallel matrix multiplication algorithm where threads must synchronize between computation phases.

```python
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ParallelMatrixMultiplier:
    def __init__(self, matrix_a, matrix_b, num_threads):
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
        self.result = np.zeros((matrix_a.shape[0], matrix_b.shape[1]))
        self.barrier = threading.Barrier(num_threads)
        self.num_threads = num_threads
        
    def multiply_row_range(self, start_row, end_row):
        # Phase 1: Row multiplication
        for i in range(start_row, end_row):
            for j in range(self.matrix_b.shape[1]):
                for k in range(self.matrix_a.shape[1]):
                    self.result[i, j] += self.matrix_a[i, k] * self.matrix_b[k, j]
        
        # Synchronize all threads before normalization
        self.barrier.wait()
        
        # Phase 2: Row normalization
        if start_row == 0:  # Only one thread performs normalization
            self.result /= self.num_threads

# Example Usage
a = np.random.rand(4, 4)
b = np.random.rand(4, 4)
multiplier = ParallelMatrixMultiplier(a, b, 2)

with ThreadPoolExecutor(max_workers=2) as executor:
    executor.submit(multiplier.multiply_row_range, 0, 2)
    executor.submit(multiplier.multiply_row_range, 2, 4)

print("Result:", multiplier.result)
```

Slide 6: Read-Write Lock Pattern Implementation

The Read-Write Lock pattern allows multiple concurrent reads while ensuring exclusive write access. This implementation provides a custom RWLock class with priority handling and demonstrates its usage in a thread-safe data structure.

```python
import threading
from typing import Dict, Any

class RWLock:
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers = 0
        self._write_waiting = 0
        
    def acquire_read(self):
        with self._read_ready:
            while self._writers > 0 or self._write_waiting > 0:
                self._read_ready.wait()
            self._readers += 1
            
    def release_read(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
                
    def acquire_write(self):
        with self._read_ready:
            self._write_waiting += 1
            while self._readers > 0 or self._writers > 0:
                self._read_ready.wait()
            self._write_waiting -= 1
            self._writers += 1
            
    def release_write(self):
        with self._read_ready:
            self._writers -= 1
            self._read_ready.notify_all()

class ThreadSafeDict:
    def __init__(self):
        self._dict: Dict[Any, Any] = {}
        self._lock = RWLock()
        
    def get(self, key):
        self._lock.acquire_read()
        try:
            return self._dict.get(key)
        finally:
            self._lock.release_read()
            
    def set(self, key, value):
        self._lock.acquire_write()
        try:
            self._dict[key] = value
        finally:
            self._lock.release_write()

# Usage Example
safe_dict = ThreadSafeDict()
def reader(d, key):
    for _ in range(3):
        print(f'Read {key}: {d.get(key)}')
        time.sleep(0.1)

def writer(d, key, value):
    for i in range(3):
        d.set(key, f'{value}-{i}')
        print(f'Write {key}: {value}-{i}')
        time.sleep(0.1)

t1 = threading.Thread(target=reader, args=(safe_dict, 'x'))
t2 = threading.Thread(target=writer, args=(safe_dict, 'x', 'value'))
t1.start(); t2.start()
t1.join(); t2.join()
```

Slide 7: Real-World Application: Log Processing System

This implementation demonstrates a practical application combining multiple threading patterns to create a high-performance log processing system. The system uses producers to read log files, processors to analyze entries, and consumers to store results.

```python
import threading
import queue
import time
from dataclasses import dataclass
from typing import List, Dict
import re

@dataclass
class LogEntry:
    timestamp: str
    level: str
    message: str
    
class LogProcessor:
    def __init__(self, num_processors=3):
        self.raw_logs = queue.Queue(maxsize=100)
        self.processed_logs = queue.Queue(maxsize=100)
        self.processors = num_processors
        self.stop_event = threading.Event()
        
    def log_reader(self, log_file: str):
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if self.stop_event.is_set():
                        break
                    self.raw_logs.put(line.strip())
        finally:
            self.raw_logs.put(None)
            
    def process_log(self):
        while not self.stop_event.is_set():
            line = self.raw_logs.get()
            if line is None:
                self.raw_logs.put(None)
                break
                
            # Parse log entry
            pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (.*)'
            match = re.match(pattern, line)
            if match:
                entry = LogEntry(
                    timestamp=match.group(1),
                    level=match.group(2),
                    message=match.group(3)
                )
                self.processed_logs.put(entry)
            self.raw_logs.task_done()
            
    def log_writer(self, output_file: str):
        with open(output_file, 'w') as f:
            while not self.stop_event.is_set():
                try:
                    entry = self.processed_logs.get(timeout=1)
                    f.write(f"{entry.timestamp} [{entry.level}] {entry.message}\n")
                    self.processed_logs.task_done()
                except queue.Empty:
                    if self.raw_logs.empty():
                        break

    def process_logs(self, input_file: str, output_file: str):
        threads = []
        
        # Start reader
        reader = threading.Thread(target=self.log_reader, args=(input_file,))
        threads.append(reader)
        
        # Start processors
        for _ in range(self.processors):
            processor = threading.Thread(target=self.process_log)
            threads.append(processor)
            
        # Start writer
        writer = threading.Thread(target=self.log_writer, args=(output_file,))
        threads.append(writer)
        
        # Start all threads
        for thread in threads:
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()

# Usage Example
processor = LogProcessor(num_processors=3)
processor.process_logs('input.log', 'output.log')
```

Slide 8: Real-World Application: Parallel Data Pipeline

A comprehensive implementation of a data processing pipeline that combines multiple threading patterns to handle large-scale data processing with error handling and monitoring capabilities.

```python
import threading
import queue
import time
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor

@dataclass
class DataChunk:
    id: int
    data: np.ndarray
    metadata: Dict[str, Any]

class DataPipeline:
    def __init__(self, num_workers: int = 4):
        self.input_queue = queue.Queue(maxsize=100)
        self.processed_queue = queue.Queue(maxsize=100)
        self.num_workers = num_workers
        self.stop_event = threading.Event()
        self.error_queue = queue.Queue()
        self.metrics = {
            'processed_chunks': 0,
            'errors': 0,
            'processing_time': 0
        }
        self.metrics_lock = threading.Lock()
        
    def data_generator(self, num_chunks: int):
        try:
            for i in range(num_chunks):
                data = np.random.rand(1000, 1000)
                chunk = DataChunk(
                    id=i,
                    data=data,
                    metadata={'timestamp': time.time()}
                )
                self.input_queue.put(chunk)
        finally:
            self.input_queue.put(None)
            
    def process_chunk(self, chunk: DataChunk) -> DataChunk:
        start_time = time.time()
        try:
            # Simulate complex processing
            processed_data = np.fft.fft2(chunk.data)
            chunk.data = processed_data
            chunk.metadata['processing_time'] = time.time() - start_time
            
            with self.metrics_lock:
                self.metrics['processed_chunks'] += 1
                self.metrics['processing_time'] += chunk.metadata['processing_time']
                
            return chunk
        except Exception as e:
            with self.metrics_lock:
                self.metrics['errors'] += 1
            self.error_queue.put((chunk.id, str(e)))
            raise
            
    def worker(self):
        while not self.stop_event.is_set():
            chunk = self.input_queue.get()
            if chunk is None:
                self.input_queue.put(None)
                break
                
            try:
                processed_chunk = self.process_chunk(chunk)
                self.processed_queue.put(processed_chunk)
            except Exception:
                pass
            finally:
                self.input_queue.task_done()
                
    def result_collector(self, output_file: str):
        with open(output_file, 'wb') as f:
            while not self.stop_event.is_set():
                try:
                    chunk = self.processed_queue.get(timeout=1)
                    np.save(f, chunk.data)
                    self.processed_queue.task_done()
                except queue.Empty:
                    if self.input_queue.empty():
                        break

    def run_pipeline(self, num_chunks: int, output_file: str):
        threads = []
        
        # Start generator
        generator = threading.Thread(
            target=self.data_generator, 
            args=(num_chunks,)
        )
        threads.append(generator)
        
        # Start workers
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for _ in range(self.num_workers):
                worker = threading.Thread(target=self.worker)
                threads.append(worker)
                worker.start()
                
        # Start collector
        collector = threading.Thread(
            target=self.result_collector, 
            args=(output_file,)
        )
        threads.append(collector)
        
        # Start all threads
        generator.start()
        collector.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
            
        return self.metrics

# Usage Example
pipeline = DataPipeline(num_workers=4)
metrics = pipeline.run_pipeline(num_chunks=100, output_file='output.npy')
print(f"Processing metrics: {metrics}")
```

Slide 9: Thread Synchronization with Condition Variables

The Condition Variable pattern extends basic synchronization by allowing threads to wait for specific conditions before proceeding. This implementation demonstrates a bounded buffer with conditional synchronization for producer-consumer scenarios.

```python
import threading
import time
import random
from collections import deque

class BoundedBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.condition = threading.Condition()
        self.capacity = capacity
        
    def put(self, item):
        with self.condition:
            while len(self.buffer) >= self.capacity:
                self.condition.wait()
            self.buffer.append(item)
            print(f'Produced: {item}, Buffer size: {len(self.buffer)}')
            self.condition.notify()
            
    def get(self):
        with self.condition:
            while len(self.buffer) == 0:
                self.condition.wait()
            item = self.buffer.popleft()
            print(f'Consumed: {item}, Buffer size: {len(self.buffer)}')
            self.condition.notify()
            return item

def producer(buffer, items):
    for i in range(items):
        time.sleep(random.uniform(0.1, 0.5))
        buffer.put(f'Item-{i}')

def consumer(buffer, items):
    for _ in range(items):
        time.sleep(random.uniform(0.2, 0.7))
        buffer.get()

# Usage Example
buffer = BoundedBuffer(capacity=5)
prod = threading.Thread(target=producer, args=(buffer, 10))
cons = threading.Thread(target=consumer, args=(buffer, 10))

prod.start()
cons.start()
prod.join()
cons.join()
```

Slide 10: Asynchronous Task Pipeline

A sophisticated implementation of an asynchronous task pipeline that combines multiple threading patterns to create a flexible, high-performance processing system with error handling and monitoring.

```python
import threading
import queue
import time
from typing import Callable, Any, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class Task:
    id: int
    data: Any
    status: str = 'pending'
    result: Any = None
    error: str = None

class AsyncTaskPipeline:
    def __init__(self, num_workers: int = 4):
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.num_workers = num_workers
        self.active = False
        self.processors: List[Callable] = []
        self.stats = {
            'processed': 0,
            'errors': 0,
            'avg_time': 0
        }
        self.stats_lock = threading.Lock()
        
    def add_processor(self, func: Callable):
        self.processors.append(func)
        
    def submit_task(self, data: Any) -> int:
        task_id = id(data)
        task = Task(id=task_id, data=data)
        self.task_queue.put(task)
        return task_id
        
    def process_task(self, task: Task) -> Task:
        start_time = time.time()
        current_data = task.data
        
        try:
            for processor in self.processors:
                current_data = processor(current_data)
                
            task.result = current_data
            task.status = 'completed'
            
            with self.stats_lock:
                self.stats['processed'] += 1
                self.stats['avg_time'] = (
                    (self.stats['avg_time'] * (self.stats['processed'] - 1) +
                     (time.time() - start_time)) / self.stats['processed']
                )
        except Exception as e:
            task.status = 'error'
            task.error = str(e)
            with self.stats_lock:
                self.stats['errors'] += 1
                
        return task
        
    def worker(self):
        while self.active or not self.task_queue.empty():
            try:
                task = self.task_queue.get(timeout=1)
                processed_task = self.process_task(task)
                self.result_queue.put(processed_task)
                self.task_queue.task_done()
            except queue.Empty:
                continue
                
    def start(self):
        self.active = True
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            workers = [
                executor.submit(self.worker)
                for _ in range(self.num_workers)
            ]
            
        return workers
        
    def stop(self):
        self.active = False

# Usage Example
def square(x):
    return x * x

def double(x):
    return x * 2

pipeline = AsyncTaskPipeline(num_workers=2)
pipeline.add_processor(square)
pipeline.add_processor(double)

# Start pipeline
pipeline.start()

# Submit tasks
for i in range(5):
    task_id = pipeline.submit_task(i)
    print(f'Submitted task {task_id} with data {i}')

# Process results
while not pipeline.result_queue.empty():
    result = pipeline.result_queue.get()
    print(f'Task {result.id}: {result.result}')

pipeline.stop()
print(f'Pipeline stats: {pipeline.stats}')
```

Slide 11: Thread Local Storage Pattern

Thread Local Storage (TLS) allows each thread to maintain its own private copy of variables. This implementation demonstrates a thread-safe logging system using TLS to track per-thread execution context and metrics.

```python
import threading
import time
import logging
from typing import Dict, Any
from contextlib import contextmanager

class ThreadLogger:
    _thread_local = threading.local()
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._thread_metrics: Dict[int, Dict[str, Any]] = {}
        self._metrics_lock = threading.Lock()
        
    @property
    def context(self) -> Dict[str, Any]:
        if not hasattr(self._thread_local, 'context'):
            self._thread_local.context = {'start_time': time.time()}
        return self._thread_local.context
        
    @contextmanager
    def track_operation(self, operation_name: str):
        thread_id = threading.get_ident()
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            with self._metrics_lock:
                if thread_id not in self._thread_metrics:
                    self._thread_metrics[thread_id] = {}
                    
                metrics = self._thread_metrics[thread_id]
                if operation_name not in metrics:
                    metrics[operation_name] = {
                        'count': 0,
                        'total_time': 0
                    }
                    
                metrics[operation_name]['count'] += 1
                metrics[operation_name]['total_time'] += duration
                
    def log_operation(self, message: str):
        thread_id = threading.get_ident()
        elapsed = time.time() - self.context['start_time']
        self.logger.info(
            f'Thread {thread_id}: {message} (Elapsed: {elapsed:.2f}s)'
        )
        
    def get_metrics(self) -> Dict[int, Dict[str, Any]]:
        with self._metrics_lock:
            return self._thread_metrics.copy()

def worker(logger: ThreadLogger, work_items: int):
    for i in range(work_items):
        with logger.track_operation('processing'):
            logger.log_operation(f'Processing item {i}')
            time.sleep(0.1)
            
        with logger.track_operation('saving'):
            logger.log_operation(f'Saving item {i}')
            time.sleep(0.05)

# Usage Example
logger = ThreadLogger()
threads = []

for i in range(3):
    t = threading.Thread(target=worker, args=(logger, 3))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("Thread Metrics:")
for thread_id, metrics in logger.get_metrics().items():
    print(f"\nThread {thread_id}:")
    for op, stats in metrics.items():
        avg_time = stats['total_time'] / stats['count']
        print(f"  {op}: {stats['count']} ops, "
              f"avg time: {avg_time:.3f}s")
```

Slide 12: Advanced Thread Communication

This implementation showcases advanced inter-thread communication using multiple synchronization primitives and message passing for complex workflows.

```python
import threading
import queue
import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Optional, Dict, List

class MessageType(Enum):
    COMMAND = auto()
    DATA = auto()
    CONTROL = auto()
    RESPONSE = auto()

@dataclass
class Message:
    type: MessageType
    sender: str
    recipient: str
    payload: Any
    correlation_id: Optional[str] = None

class ThreadChannel:
    def __init__(self):
        self.queues: Dict[str, queue.Queue] = {}
        self.subscribers: Dict[str, List[str]] = {}
        self._lock = threading.Lock()
        
    def register(self, name: str):
        with self._lock:
            if name not in self.queues:
                self.queues[name] = queue.Queue()
                self.subscribers[name] = []
                
    def subscribe(self, publisher: str, subscriber: str):
        with self._lock:
            if subscriber not in self.subscribers[publisher]:
                self.subscribers[publisher].append(subscriber)
                
    def send(self, message: Message):
        with self._lock:
            if message.recipient in self.queues:
                self.queues[message.recipient].put(message)
            
            # Broadcast to subscribers
            if message.sender in self.subscribers:
                for subscriber in self.subscribers[message.sender]:
                    if subscriber != message.recipient:
                        self.queues[subscriber].put(message)
                        
    def receive(self, recipient: str, timeout: Optional[float] = None) -> Optional[Message]:
        try:
            return self.queues[recipient].get(timeout=timeout)
        except queue.Empty:
            return None

class Worker:
    def __init__(self, name: str, channel: ThreadChannel):
        self.name = name
        self.channel = channel
        self.running = False
        self.channel.register(name)
        
    def process_message(self, message: Message) -> Optional[Message]:
        if message.type == MessageType.COMMAND:
            # Process command
            result = f"Processed command: {message.payload}"
            return Message(
                type=MessageType.RESPONSE,
                sender=self.name,
                recipient=message.sender,
                payload=result,
                correlation_id=message.correlation_id
            )
        return None
        
    def run(self):
        self.running = True
        while self.running:
            message = self.channel.receive(self.name, timeout=1.0)
            if message:
                response = self.process_message(message)
                if response:
                    self.channel.send(response)
                    
    def stop(self):
        self.running = False

# Usage Example
channel = ThreadChannel()

# Create workers
worker1 = Worker("worker1", channel)
worker2 = Worker("worker2", channel)

# Subscribe worker2 to worker1's messages
channel.subscribe("worker1", "worker2")

# Start workers
t1 = threading.Thread(target=worker1.run)
t2 = threading.Thread(target=worker2.run)
t1.start()
t2.start()

# Send command
command = Message(
    type=MessageType.COMMAND,
    sender="main",
    recipient="worker1",
    payload="do_something",
    correlation_id="123"
)
channel.register("main")
channel.send(command)

# Wait for response
response = channel.receive("main", timeout=2.0)
print(f"Received response: {response}")

# Cleanup
worker1.stop()
worker2.stop()
t1.join()
t2.join()
```

Slide 13: Performance Monitoring System

This implementation demonstrates a comprehensive thread monitoring system that tracks performance metrics, thread health, and system resource utilization across multiple concurrent operations.

```python
import threading
import time
import psutil
import queue
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict

@dataclass
class ThreadMetric:
    thread_id: int
    cpu_percent: float
    memory_usage: float
    active_time: float
    operation_count: int
    error_count: int

class ThreadMonitor:
    def __init__(self):
        self._metrics: Dict[int, ThreadMetric] = {}
        self._lock = threading.Lock()
        self._start_times: Dict[int, float] = defaultdict(float)
        self._operation_counts = defaultdict(int)
        self._error_counts = defaultdict(int)
        self.running = True
        
    def start_monitoring(self):
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def _monitor_loop(self):
        while self.running:
            self._update_metrics()
            time.sleep(1)  # Update frequency
            
    def _update_metrics(self):
        process = psutil.Process()
        with self._lock:
            for thread in threading.enumerate():
                thread_id = thread.ident
                if thread_id:
                    # Get thread-specific CPU and memory usage
                    try:
                        thread_cpu = process.cpu_percent(interval=0.1) / psutil.cpu_count()
                        thread_memory = process.memory_info().rss / (1024 * 1024)  # MB
                        
                        self._metrics[thread_id] = ThreadMetric(
                            thread_id=thread_id,
                            cpu_percent=thread_cpu,
                            memory_usage=thread_memory,
                            active_time=time.time() - self._start_times[thread_id],
                            operation_count=self._operation_counts[thread_id],
                            error_count=self._error_counts[thread_id]
                        )
                    except Exception:
                        continue
                        
    def register_thread(self):
        thread_id = threading.get_ident()
        with self._lock:
            self._start_times[thread_id] = time.time()
            
    def record_operation(self, success: bool = True):
        thread_id = threading.get_ident()
        with self._lock:
            if success:
                self._operation_counts[thread_id] += 1
            else:
                self._error_counts[thread_id] += 1
                
    def get_metrics(self) -> Dict[int, ThreadMetric]:
        with self._lock:
            return self._metrics.copy()
            
    def stop(self):
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()

# Worker implementation for testing
def worker(monitor: ThreadMonitor, iterations: int):
    monitor.register_thread()
    
    for i in range(iterations):
        try:
            # Simulate work
            time.sleep(0.1)
            # Complex calculation to use CPU
            _ = [i * i for i in range(10000)]
            monitor.record_operation(success=True)
            
            # Simulate occasional errors
            if i % 5 == 0:
                raise ValueError("Simulated error")
                
        except Exception:
            monitor.record_operation(success=False)

# Usage Example
monitor = ThreadMonitor()
monitor.start_monitoring()

threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(monitor, 10))
    threads.append(t)
    t.start()

# Monitor and print metrics
try:
    while any(t.is_alive() for t in threads):
        metrics = monitor.get_metrics()
        print("\nCurrent Thread Metrics:")
        for thread_id, metric in metrics.items():
            print(f"\nThread {thread_id}:")
            print(f"  CPU: {metric.cpu_percent:.1f}%")
            print(f"  Memory: {metric.memory_usage:.1f} MB")
            print(f"  Active Time: {metric.active_time:.1f}s")
            print(f"  Operations: {metric.operation_count}")
            print(f"  Errors: {metric.error_count}")
        time.sleep(2)
finally:
    monitor.stop()
    for t in threads:
        t.join()
```

Slide 14: Additional Resources

*   "A Survey of Multithreading Design Patterns" - arXiv:2203.12845 [https://arxiv.org/abs/2203.12845](https://arxiv.org/abs/2203.12845)
*   "Performance Analysis of Thread Synchronization Patterns" - arXiv:2104.09856 [https://arxiv.org/abs/2104.09856](https://arxiv.org/abs/2104.09856)
*   "Modern Concurrent Programming Patterns" - arXiv:2201.03545 [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)
*   Suggested searches:
    *   "Python threading best practices"
    *   "Multithreading design patterns"
    *   "Concurrent programming patterns Python"
    *   "Thread synchronization techniques"

