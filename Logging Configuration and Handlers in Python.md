## Logging Configuration and Handlers in Python
Slide 1: Basic Logging Configuration

The logging module in Python provides a flexible framework for generating log messages with different severity levels. Understanding the basic configuration is essential for implementing robust logging systems in production applications.

```python
import logging

# Basic configuration for logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log',
    filemode='w'
)

# Example usage
logger = logging.getLogger(__name__)
logger.debug('Debug message')
logger.info('Info message')
logger.warning('Warning message')
logger.error('Error message')
logger.critical('Critical message')

# Output in app.log:
# 2024-11-17 10:30:45,123 - __main__ - DEBUG - Debug message
# 2024-11-17 10:30:45,124 - __main__ - INFO - Info message
# 2024-11-17 10:30:45,125 - __main__ - WARNING - Warning message
# 2024-11-17 10:30:45,126 - __main__ - ERROR - Error message
# 2024-11-17 10:30:45,127 - __main__ - CRITICAL - Critical message
```

Slide 2: Custom Logging Formatters

Formatters define the structure and content of log messages. Custom formatters allow precise control over log message presentation, including timestamps, log levels, and contextual information.

```python
import logging
from datetime import datetime

class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # Custom datetime formatting with milliseconds
        created = datetime.fromtimestamp(record.created)
        if datefmt:
            return created.strftime(datefmt)
        return created.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def format(self, record):
        # Add custom fields to the log record
        record.custom_field = f"Process-{record.process}"
        return super().format(record)

# Configure logger with custom formatter
logger = logging.getLogger('custom_logger')
handler = logging.StreamHandler()
formatter = CustomFormatter(
    '%(asctime)s [%(custom_field)s] %(levelname)s: %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Example usage
logger.info('Custom formatted log message')
# Output: 2024-11-17 10:30:45.123 [Process-1234] INFO: Custom formatted log message
```

Slide 3: File Handler Configuration

File handlers enable logging to files with rotation capabilities, ensuring logs are properly maintained and archived. This implementation shows how to set up file rotation based on size and backup count.

```python
import logging
from logging.handlers import RotatingFileHandler
import os

# Configure rotating file handler
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

handler = RotatingFileHandler(
    filename=os.path.join(log_dir, 'rotating.log'),
    maxBytes=1024 * 1024,  # 1MB
    backupCount=5,
    encoding='utf-8'
)

# Configure formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)

# Setup logger
logger = logging.getLogger('rotating_logger')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Example usage
for i in range(1000):
    logger.info(f'Log entry {i}: This is a test message')
    # Files created: rotating.log, rotating.log.1, rotating.log.2, etc.
```

Slide 4: TimedRotatingFileHandler Implementation

The TimedRotatingFileHandler provides time-based log rotation, essential for maintaining organized log files in long-running applications. This handler supports rotation based on various time intervals.

```python
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

# Configure timed rotating handler
handler = TimedRotatingFileHandler(
    filename='timed_rotating.log',
    when='midnight',  # Rotate at midnight
    interval=1,       # Rotate every day
    backupCount=30,   # Keep 30 days of logs
    encoding='utf-8'
)

# Set custom formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)

# Setup logger
logger = logging.getLogger('timed_rotating_logger')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Example usage
def log_daily_metrics():
    logger.info(f'Daily metrics for {datetime.now().date()}')
    logger.info('System health: OK')
    logger.info('Active users: 1250')
    logger.info('Average response time: 150ms')

# Simulate daily logging
log_daily_metrics()
```

Slide 5: Custom Log Levels

Python's logging system allows definition of custom log levels for specific application needs. This implementation demonstrates how to create and use custom log levels with appropriate severity rankings.

```python
import logging

# Define custom log levels
TRACE_LEVEL = 5
VERBOSE_LEVEL = 15
AUDIT_LEVEL = 25

# Register custom levels
logging.addLevelName(TRACE_LEVEL, 'TRACE')
logging.addLevelName(VERBOSE_LEVEL, 'VERBOSE')
logging.addLevelName(AUDIT_LEVEL, 'AUDIT')

# Create custom logger class
class CustomLogger(logging.Logger):
    def trace(self, msg, *args, **kwargs):
        self.log(TRACE_LEVEL, msg, *args, **kwargs)
        
    def verbose(self, msg, *args, **kwargs):
        self.log(VERBOSE_LEVEL, msg, *args, **kwargs)
        
    def audit(self, msg, *args, **kwargs):
        self.log(AUDIT_LEVEL, msg, *args, **kwargs)

# Register custom logger class
logging.setLoggerClass(CustomLogger)

# Configure logger
logger = logging.getLogger('custom_levels')
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(TRACE_LEVEL)

# Example usage
logger.trace('Detailed trace message')
logger.verbose('Verbose debug information')
logger.audit('User authentication successful')
```

Slide 6: Network Handler Implementation

Network handlers enable distributed logging by sending log records over network protocols. This implementation showcases a custom handler for sending logs to a remote server using TCP/IP sockets.

```python
import logging
import pickle
import struct
import socket
import threading

class NetworkHandler(logging.Handler):
    def __init__(self, host='localhost', port=9999):
        super().__init__()
        self.host = host
        self.port = port
        self.socket = None
        self.connect()

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def emit(self, record):
        try:
            # Pickle the log record
            data = pickle.dumps(record)
            # Send data length first
            length = struct.pack('>L', len(data))
            self.socket.send(length + data)
        except Exception:
            self.handleError(record)

# Example server implementation
def start_log_server(host='localhost', port=9999):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    
    while True:
        client, addr = server.accept()
        while True:
            # Receive data length
            length_data = client.recv(4)
            if not length_data:
                break
            length = struct.unpack('>L', length_data)[0]
            # Receive log record
            data = client.recv(length)
            record = pickle.loads(data)
            print(f"Received: {record.getMessage()}")

# Usage example
if __name__ == '__main__':
    # Start server in separate thread
    server_thread = threading.Thread(target=start_log_server)
    server_thread.daemon = True
    server_thread.start()

    # Configure client logger
    logger = logging.getLogger('network_logger')
    handler = NetworkHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Send some logs
    logger.info("Test network logging")
    logger.error("Network error simulation")
```

Slide 7: Context-Based Logging

Context-based logging allows for tracking related log messages across different parts of an application. This implementation demonstrates a context manager for maintaining logging context.

```python
import logging
import threading
from contextlib import contextmanager
import uuid

class ContextLogger:
    _context = threading.local()
    
    @classmethod
    def get_context(cls):
        if not hasattr(cls._context, 'stack'):
            cls._context.stack = []
        return cls._context.stack
    
    @classmethod
    def get_current_context(cls):
        stack = cls.get_context()
        return stack[-1] if stack else {}

    @contextmanager
    def context(self, **kwargs):
        stack = self.get_context()
        context = {
            'trace_id': str(uuid.uuid4()),
            **kwargs
        }
        stack.append(context)
        try:
            yield context
        finally:
            stack.pop()

# Custom formatter that includes context
class ContextFormatter(logging.Formatter):
    def format(self, record):
        context = ContextLogger.get_current_context()
        record.trace_id = context.get('trace_id', 'no-trace')
        record.user = context.get('user', 'anonymous')
        return super().format(record)

# Configure logger
logger = logging.getLogger('context_logger')
handler = logging.StreamHandler()
formatter = ContextFormatter(
    '%(asctime)s - [%(trace_id)s] - %(user)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage example
context_logger = ContextLogger()

def process_request(user_id):
    with context_logger.context(user=user_id):
        logger.info(f"Processing request")
        perform_operation()
        logger.info(f"Request completed")

def perform_operation():
    logger.info("Performing operation")

# Example execution
process_request("user123")
```

Slide 8: Asynchronous Logging Handler

Asynchronous logging prevents I/O operations from blocking the main application thread. This implementation shows a queue-based asynchronous handler for improved performance.

```python
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

class AsyncHandler(logging.Handler):
    def __init__(self, max_queue_size=1000):
        super().__init__()
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.running = True
        self.start_consumer()

    def emit(self, record):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.handleError(record)

    def start_consumer(self):
        def consumer():
            while self.running:
                try:
                    record = self.queue.get(timeout=1)
                    message = self.format(record)
                    # Simulate I/O operation
                    time.sleep(0.1)
                    print(message)
                except queue.Empty:
                    continue

        self.executor.submit(consumer)

    def close(self):
        self.running = False
        self.executor.shutdown(wait=True)
        super().close()

# Configure logger with async handler
logger = logging.getLogger('async_logger')
handler = AsyncHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage example
def heavy_operation():
    for i in range(100):
        logger.info(f"Processing item {i}")
        # Main thread continues without blocking
        time.sleep(0.01)

# Run example
heavy_operation()
time.sleep(2)  # Wait for async handler to process queue
handler.close()
```

Slide 9: Structured Logging Implementation

Structured logging organizes log data in a machine-readable format, typically JSON, making it easier to parse and analyze logs using tools like ELK stack or Splunk. This implementation demonstrates advanced JSON-based structured logging.

```python
import logging
import json
import datetime
import socket
import traceback
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    def __init__(self, **kwargs):
        self.default_fields = {
            'hostname': socket.gethostname(),
            'app_name': 'my_application'
        }
        self.default_fields.update(kwargs)

    def format(self, record: logging.LogRecord) -> str:
        message: Dict[str, Any] = {
            'timestamp': datetime.datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'line_number': record.lineno,
            **self.default_fields
        }

        # Add exception info if present
        if record.exc_info:
            message['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'stacktrace': traceback.format_exception(*record.exc_info)
            }

        # Add extra fields from record
        if hasattr(record, 'extra_fields'):
            message.update(record.extra_fields)

        return json.dumps(message)

# Configure structured logger
logger = logging.getLogger('structured_logger')
handler = logging.StreamHandler()
formatter = JSONFormatter(environment='production')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Example usage with extra fields and exception handling
try:
    # Simulate operation with context
    extra = {'user_id': '12345', 'operation': 'data_processing'}
    logger.info("Starting data processing", extra={'extra_fields': extra})
    
    # Simulate error
    raise ValueError("Invalid data format")
except Exception as e:
    logger.error("Processing failed", 
                 extra={'extra_fields': extra},
                 exc_info=True)

# Output example:
# {"timestamp": "2024-11-17T10:30:45.123456", "level": "INFO", "logger": "structured_logger", ...}
```

Slide 10: Buffered Logging Handler

Buffered logging improves performance by collecting log records in memory and writing them in batches. This implementation includes memory management and forced flush capabilities.

```python
import logging
import threading
import time
from typing import List
from dataclasses import dataclass, field

@dataclass
class BufferedHandler(logging.Handler):
    capacity: int = 1000
    flush_interval: int = 5  # seconds
    records: List[logging.LogRecord] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_flush: float = field(default_factory=time.time)
    
    def __post_init__(self):
        super().__init__()
        self.start_flush_timer()

    def emit(self, record: logging.LogRecord) -> None:
        with self.lock:
            self.records.append(record)
            
            if len(self.records) >= self.capacity:
                self.flush()
            elif time.time() - self.last_flush >= self.flush_interval:
                self.flush()

    def flush(self) -> None:
        with self.lock:
            if not self.records:
                return
                
            # Process all buffered records
            for record in self.records:
                try:
                    message = self.format(record)
                    with open('buffered.log', 'a') as f:
                        f.write(message + '\n')
                except Exception:
                    self.handleError(record)
            
            self.records.clear()
            self.last_flush = time.time()

    def start_flush_timer(self) -> None:
        def flush_timer():
            while True:
                time.sleep(self.flush_interval)
                self.flush()

        timer_thread = threading.Thread(target=flush_timer, daemon=True)
        timer_thread.start()

# Configure buffered logger
logger = logging.getLogger('buffered_logger')
handler = BufferedHandler(capacity=100, flush_interval=3)
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Example usage
def simulate_logging():
    for i in range(1000):
        logger.info(f"Log message {i}")
        time.sleep(0.01)  # Simulate processing time

# Run simulation
simulate_logging()
handler.flush()  # Force final flush
```

Slide 11: Multi-Process Safe Logging

Implementing thread-safe and process-safe logging is crucial in multi-process applications. This implementation ensures log integrity across multiple processes using file locking mechanisms.

```python
import logging
import multiprocessing
import fcntl
import time
import os
from contextlib import contextmanager
from typing import Optional

class MultiProcessHandler(logging.FileHandler):
    def __init__(self, filename: str, mode: str = 'a', encoding: Optional[str] = None):
        super().__init__(filename, mode, encoding)
        self.stream_lock = None

    @contextmanager
    def _lock_file(self):
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with self._lock_file():
                super().emit(record)
                self.flush()
        except Exception:
            self.handleError(record)

def worker_process(pid: int) -> None:
    logger = logging.getLogger(f'process_{pid}')
    handler = MultiProcessHandler('multiprocess.log')
    formatter = logging.Formatter(
        '%(asctime)s - Process %(process)d - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    for i in range(5):
        logger.info(f'Message {i} from process {pid}')
        time.sleep(0.1)

if __name__ == '__main__':
    # Create multiple processes
    processes = []
    for i in range(4):
        p = multiprocessing.Process(target=worker_process, args=(i,))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Read and display the log file
    with open('multiprocess.log', 'r') as f:
        print(f.read())
```

Slide 12: Real-time Log Analysis Handler

This implementation provides real-time log analysis capabilities, including pattern matching and alert generation based on log content and frequency.

```python
import logging
import re
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Pattern

class LogAnalyzer:
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.log_buffer = deque()
        self.patterns: Dict[str, Pattern] = {
            'error': re.compile(r'error|exception|fail', re.IGNORECASE),
            'warning': re.compile(r'warning|warn', re.IGNORECASE),
            'critical': re.compile(r'critical|fatal', re.IGNORECASE)
        }
        self.alerts: List[str] = []

    def analyze(self, record: logging.LogRecord) -> None:
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(seconds=self.window_seconds)
        
        # Remove old entries
        while self.log_buffer and self.log_buffer[0].created < cutoff_time.timestamp():
            self.log_buffer.popleft()
        
        # Add new record
        self.log_buffer.append(record)
        
        # Analyze patterns
        self._check_patterns(record)
        self._check_frequency()

    def _check_patterns(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        for pattern_name, pattern in self.patterns.items():
            if pattern.search(message):
                alert = f"Pattern '{pattern_name}' detected: {message}"
                self.alerts.append(alert)

    def _check_frequency(self) -> None:
        if len(self.log_buffer) > 100:
            alert = f"High log frequency: {len(self.log_buffer)} messages in {self.window_seconds} seconds"
            self.alerts.append(alert)

class AnalyticsHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.analyzer = LogAnalyzer()

    def emit(self, record: logging.LogRecord) -> None:
        self.analyzer.analyze(record)
        if self.analyzer.alerts:
            for alert in self.analyzer.alerts:
                print(f"ALERT: {alert}")
            self.analyzer.alerts.clear()

# Configure logger with analytics
logger = logging.getLogger('analytics_logger')
handler = AnalyticsHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Example usage
def simulate_logs():
    logger.info("Application started")
    logger.warning("Resource usage high")
    logger.error("Database connection failed")
    for i in range(150):  # Trigger frequency alert
        logger.info(f"Processing item {i}")
    logger.critical("System shutdown required")

# Run simulation
simulate_logs()
```

Slide 13: Distributed Tracing Integration

This implementation combines logging with distributed tracing capabilities, enabling correlation of logs across different services and components in a microservices architecture.

```python
import logging
import uuid
import threading
import json
from typing import Dict, Optional
from dataclasses import dataclass, field

@dataclass
class TraceContext:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)

class TracingLogger:
    _context = threading.local()

    @classmethod
    def get_context(cls) -> TraceContext:
        if not hasattr(cls._context, 'trace_context'):
            cls._context.trace_context = TraceContext()
        return cls._context.trace_context

    @classmethod
    def set_context(cls, context: TraceContext) -> None:
        cls._context.trace_context = context

class TracingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.traces = []

    def emit(self, record: logging.LogRecord) -> None:
        context = TracingLogger.get_context()
        trace_data = {
            'timestamp': self.formatter.formatTime(record),
            'trace_id': context.trace_id,
            'span_id': context.span_id,
            'parent_span_id': context.parent_span_id,
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'baggage': context.baggage
        }
        self.traces.append(trace_data)
        
        # Format for output
        output = json.dumps(trace_data)
        print(output)

class TraceContextManager:
    def __init__(self, service_name: str, operation_name: str):
        self.service_name = service_name
        self.operation_name = operation_name
        self.parent_context = None

    def __enter__(self) -> TraceContext:
        self.parent_context = TracingLogger.get_context()
        new_context = TraceContext(
            trace_id=self.parent_context.trace_id,
            parent_span_id=self.parent_context.span_id,
            baggage={
                'service': self.service_name,
                'operation': self.operation_name
            }
        )
        TracingLogger.set_context(new_context)
        return new_context

    def __exit__(self, exc_type, exc_val, exc_tb):
        TracingLogger.set_context(self.parent_context)

# Configure tracing logger
logger = logging.getLogger('tracing_logger')
handler = TracingHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Example usage
def process_order(order_id: str):
    with TraceContextManager('order_service', 'process_order'):
        logger.info(f"Processing order {order_id}")
        
        with TraceContextManager('payment_service', 'validate_payment'):
            logger.info("Validating payment")
            process_payment(order_id)

def process_payment(order_id: str):
    with TraceContextManager('payment_service', 'process_transaction'):
        logger.info(f"Processing payment for order {order_id}")
        logger.info("Payment completed successfully")

# Run example
process_order("ORD-123")
```

Slide 14: Additional Resources

*   Distributed Logging Patterns for Microservices:
    *   [https://example.com/distributed-logging-patterns](https://example.com/distributed-logging-patterns)
    *   Search terms: "distributed logging architecture patterns"
*   Advanced Python Logging Techniques:
    *   [https://docs.python.org/3/howto/logging-cookbook.html](https://docs.python.org/3/howto/logging-cookbook.html)
    *   [https://example.com/python-logging-best-practices](https://example.com/python-logging-best-practices)
*   Performance Optimization in Python Logging:
    *   [https://example.com/python-logging-performance](https://example.com/python-logging-performance)
    *   Search terms: "python logging performance optimization techniques"
*   Structured Logging Implementation Guidelines:
    *   [https://example.com/structured-logging-python](https://example.com/structured-logging-python)
    *   Search terms: "structured logging python implementation"
*   Distributed Tracing with Python:
    *   [https://example.com/distributed-tracing-python](https://example.com/distributed-tracing-python)
    *   Search terms: "python distributed tracing implementation opentelemetry"

