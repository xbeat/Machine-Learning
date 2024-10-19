## Advantages of Logging over Print() in Python
Slide 1: Introduction to Logging in Python

Logging is a powerful tool for tracking events in your Python programs. It offers significant advantages over using print() statements for debugging and monitoring.

```python
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Example usage
logging.info("Program started")
logging.warning("Low memory warning")
logging.error("File not found")

# Output:
# 2024-09-28 10:15:30,123 - INFO - Program started
# 2024-09-28 10:15:30,124 - WARNING - Low memory warning
# 2024-09-28 10:15:30,125 - ERROR - File not found
```

Slide 2: Adjustable Severity Levels

Logging provides different severity levels, allowing you to categorize messages based on their importance. This feature enables better organization and filtering of log messages.

```python
import logging

logging.basicConfig(level=logging.DEBUG)

logging.debug("Detailed information for debugging")
logging.info("General information about program execution")
logging.warning("An indication of a potential problem")
logging.error("A more serious problem")
logging.critical("A critical error - program may be unable to continue")

# Output:
# DEBUG:root:Detailed information for debugging
# INFO:root:General information about program execution
# WARNING:root:An indication of a potential problem
# ERROR:root:A more serious problem
# CRITICAL:root:A critical error - program may be unable to continue
```

Slide 3: Configuration Flexibility

Logging offers extensive configuration options, allowing you to customize where and how your messages are logged. You can easily direct log output to different destinations.

```python
import logging

# Log to a file
logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG)
logging.debug("This message will be written to app.log")

# Log to console and file simultaneously
console = logging.StreamHandler()
file_handler = logging.FileHandler('both.log')
logging.getLogger('').addHandler(console)
logging.getLogger('').addHandler(file_handler)
logging.warning("This will appear in both console and both.log file")

# Output to console and both.log file:
# WARNING:root:This will appear in both console and both.log file
```

Slide 4: Improved Performance

Unlike print() statements, logging can be efficiently disabled or filtered without modifying the code, resulting in better performance in production environments.

```python
import logging
import time

def performance_test(log_func):
    start_time = time.time()
    for i in range(100000):
        log_func(f"Iteration {i}")
    return time.time() - start_time

# Test with print()
print_time = performance_test(print)

# Test with logging (INFO level)
logging.basicConfig(level=logging.INFO)
log_time = performance_test(logging.info)

# Test with logging (WARNING level, so INFO messages are ignored)
logging.basicConfig(level=logging.WARNING)
log_ignored_time = performance_test(logging.info)

print(f"Print time: {print_time:.4f} seconds")
print(f"Logging (INFO) time: {log_time:.4f} seconds")
print(f"Logging (ignored) time: {log_ignored_time:.4f} seconds")

# Output (approximate):
# Print time: 0.1500 seconds
# Logging (INFO) time: 0.2000 seconds
# Logging (ignored) time: 0.0100 seconds
```

Slide 5: Efficient Bug Management

Logging provides detailed information about exceptions, making it easier to diagnose and fix issues in your code.

```python
import logging

logging.basicConfig(level=logging.ERROR)

def divide(x, y):
    try:
        result = x / y
    except ZeroDivisionError:
        logging.exception("Division by zero attempted")
    else:
        return result

divide(10, 0)

# Output:
# ERROR:root:Division by zero attempted
# Traceback (most recent call last):
#   File "<stdin>", line 3, in divide
# ZeroDivisionError: division by zero
```

Slide 6: Simplified Deployment

With logging, you can easily adjust the verbosity of your application without modifying the code, simplifying the transition from development to production environments.

```python
import logging
import sys

# Development configuration
if '--dev' in sys.argv:
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Running in development mode")
else:
    # Production configuration
    logging.basicConfig(level=logging.ERROR)
    logging.info("Running in production mode")

# This will only show in development mode
logging.debug("Database connection established")

# This will show in both modes
logging.error("Critical error occurred")

# Output in development mode:
# INFO:root:Running in development mode
# DEBUG:root:Database connection established
# ERROR:root:Critical error occurred

# Output in production mode:
# ERROR:root:Critical error occurred
```

Slide 7: Customizing Log Formats

Logging allows you to customize the format of your log messages, providing more context and making them easier to parse and analyze.

```python
import logging

# Create a custom formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create and configure a handler
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Create a logger and add the handler
logger = logging.getLogger('MyApp')
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Use the logger
logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')

# Output:
# 2024-09-28 10:30:15,123 - MyApp - DEBUG - This is a debug message
# 2024-09-28 10:30:15,124 - MyApp - INFO - This is an info message
# 2024-09-28 10:30:15,125 - MyApp - WARNING - This is a warning message
```

Slide 8: Logging in Multiple Modules

Logging can be effectively used across multiple modules in your application, providing a centralized way to manage logs from different parts of your program.

```python
# module_a.py
import logging

logger = logging.getLogger(__name__)

def function_a():
    logger.info("Function A called")

# module_b.py
import logging

logger = logging.getLogger(__name__)

def function_b():
    logger.warning("Function B called")

# main.py
import logging
import module_a
import module_b

logging.basicConfig(level=logging.INFO)

module_a.function_a()
module_b.function_b()

# Output:
# INFO:module_a:Function A called
# WARNING:module_b:Function B called
```

Slide 9: Rotating File Handler

For long-running applications, it's important to manage log file sizes. The RotatingFileHandler allows you to automatically create new log files when the current one reaches a certain size.

```python
import logging
from logging.handlers import RotatingFileHandler

# Create a rotating file handler
handler = RotatingFileHandler('app.log', maxBytes=2000, backupCount=5)
logger = logging.getLogger('RotatingLogger')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Generate some log messages
for i in range(10000):
    logger.info(f"This is log message {i}")

# This will create app.log, app.log.1, app.log.2, etc.
# when app.log reaches 2000 bytes
```

Slide 10: Real-Life Example: Web Server Logging

Logging is crucial for monitoring and debugging web applications. Here's an example of how you might set up logging for a simple web server.

```python
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

# Set up logging
logging.basicConfig(filename='webserver.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        logging.info(f"GET request received for path: {self.path}")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello, World!')

    def log_message(self, format, *args):
        logging.info("%s - - [%s] %s" %
                     (self.client_address[0],
                      self.log_date_time_string(),
                      format%args))

httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
logging.info("Server started on localhost:8000")
httpd.serve_forever()

# Output in webserver.log:
# 2024-09-28 11:00:00,123 - INFO - Server started on localhost:8000
# 2024-09-28 11:00:05,456 - INFO - GET request received for path: /
# 2024-09-28 11:00:05,457 - INFO - 127.0.0.1 - - [28/Sep/2024 11:00:05] "GET / HTTP/1.1" 200 -
```

Slide 11: Real-Life Example: Data Processing Pipeline

Logging is essential in data processing pipelines to track progress and catch errors. Here's an example of how logging might be used in a simple data processing script.

```python
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data(data):
    logging.info(f"Processing {len(data)} items")
    processed = []
    for i, item in enumerate(data):
        try:
            result = item * random.randint(1, 10)
            processed.append(result)
            if i % 100 == 0:
                logging.debug(f"Processed {i} items")
        except Exception as e:
            logging.error(f"Error processing item {i}: {e}")
    logging.info(f"Finished processing. {len(processed)} items successful")
    return processed

# Generate some sample data
data = list(range(1000))

# Process the data
result = process_data(data)

# Output:
# 2024-09-28 11:30:00,123 - INFO - Processing 1000 items
# 2024-09-28 11:30:00,124 - DEBUG - Processed 0 items
# 2024-09-28 11:30:00,125 - DEBUG - Processed 100 items
# ...
# 2024-09-28 11:30:00,234 - DEBUG - Processed 900 items
# 2024-09-28 11:30:00,235 - INFO - Finished processing. 1000 items successful
```

Slide 12: Logging Best Practices

Here are some best practices to follow when implementing logging in your Python applications:

1.  Use appropriate log levels
2.  Include contextual information
3.  Use structured logging for complex data
4.  Configure logging as early as possible in your application
5.  Use exception logging with logging.exception()
6.  Avoid sensitive information in logs

```python
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_user(user_id, action):
    logger.info("Processing user", extra={
        'user_id': user_id,
        'action': action
    })
    
    # Simulating an error
    try:
        if action == 'delete':
            raise ValueError("Cannot delete user")
        # Process user...
    except Exception as e:
        logger.exception(f"Error processing user {user_id}")

# Usage
process_user(12345, 'update')
process_user(67890, 'delete')

# Output:
# 2024-09-28 12:00:00,123 - __main__ - INFO - Processing user
# 2024-09-28 12:00:00,124 - __main__ - ERROR - Error processing user 67890
# Traceback (most recent call last):
#   File "<stdin>", line 4, in process_user
# ValueError: Cannot delete user
```

Slide 13: Logging vs Print: A Comparison

Let's compare logging and print() to understand why logging is generally preferred for debugging and monitoring applications.

```python
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def using_print():
    print("Starting function")
    time.sleep(1)
    print("Function completed")

def using_logging():
    logging.info("Starting function")
    time.sleep(1)
    logging.info("Function completed")

print("Using print():")
using_print()

print("\nUsing logging:")
using_logging()

# Output:
# Using print():
# Starting function
# Function completed

# Using logging:
# 2024-09-28 12:30:00,123 - INFO - Starting function
# 2024-09-28 12:30:01,124 - INFO - Function completed
```

Slide 14: Additional Resources

For more information on Python logging, consider exploring these resources:

1.  Python's official logging documentation: [https://docs.python.org/3/library/logging.html](https://docs.python.org/3/library/logging.html)
2.  Logging Cookbook: [https://docs.python.org/3/howto/logging-cookbook.html](https://docs.python.org/3/howto/logging-cookbook.html)
3.  "Structured Logging in Python" by Yury Selivanov: [https://arxiv.org/abs/2110.07557](https://arxiv.org/abs/2110.07557)

These resources provide in-depth explanations and advanced techniques for mastering logging in Python.

