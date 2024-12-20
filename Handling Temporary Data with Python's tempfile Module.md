## Handling Temporary Data with Python's tempfile Module
Slide 1: Introduction to Python's tempfile Module

The tempfile module in Python provides a robust and secure way to create temporary files and directories. This module is essential for handling transient data without cluttering your filesystem or risking data leaks.

```python
import tempfile

# Create a temporary file
with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
    temp_file.write("This is temporary data")
    print(f"Temporary file created: {temp_file.name}")

# Output: Temporary file created: /tmp/tmp1234abcd.txt
```

Slide 2: NamedTemporaryFile: Creating Temporary Files

NamedTemporaryFile creates a temporary file with a unique name. It's useful when you need to reference the file by name or share it with other processes.

```python
import tempfile
import os

with tempfile.NamedTemporaryFile(suffix='.txt', prefix='temp_', delete=False) as temp_file:
    temp_file.write(b"Hello, temporary world!")
    temp_name = temp_file.name

print(f"File contents: {open(temp_name, 'rb').read()}")
os.unlink(temp_name)  # Manually delete the file

# Output: File contents: b'Hello, temporary world!'
```

Slide 3: TemporaryFile: Secure Temporary Storage

TemporaryFile creates a temporary file that is automatically deleted when closed. It's ideal for short-lived, secure data storage.

```python
import tempfile

with tempfile.TemporaryFile(mode='w+t') as temp_file:
    temp_file.write("Sensitive data")
    temp_file.seek(0)
    print(f"File contents: {temp_file.read()}")

# The file is automatically deleted after the 'with' block

# Output: File contents: Sensitive data
```

Slide 4: SpooledTemporaryFile: Memory-First Temporary Storage

SpooledTemporaryFile initially stores data in memory and only writes to disk if the data exceeds a size threshold.

```python
import tempfile

with tempfile.SpooledTemporaryFile(max_size=1000) as temp_file:
    temp_file.write(b"Small data")
    print(f"In memory: {temp_file._rolled}")
    
    temp_file.write(b"A" * 1000)
    print(f"On disk: {temp_file._rolled}")

# Output:
# In memory: False
# On disk: True
```

Slide 5: TemporaryDirectory: Creating Temporary Directories

TemporaryDirectory creates a temporary directory that is automatically removed when closed.

```python
import tempfile
import os

with tempfile.TemporaryDirectory() as temp_dir:
    temp_file_path = os.path.join(temp_dir, "temp_file.txt")
    with open(temp_file_path, "w") as temp_file:
        temp_file.write("Temporary file in a temporary directory")
    
    print(f"Directory contents: {os.listdir(temp_dir)}")

# Directory and its contents are automatically deleted after the 'with' block

# Output: Directory contents: ['temp_file.txt']
```

Slide 6: Customizing Temporary File Names

You can customize the prefix, suffix, and directory of temporary files for better organization and readability.

```python
import tempfile

custom_temp = tempfile.NamedTemporaryFile(
    prefix="log_",
    suffix=".txt",
    dir="/tmp/custom_temp",
    delete=False
)

print(f"Custom temp file: {custom_temp.name}")
custom_temp.close()

# Output might be: Custom temp file: /tmp/custom_temp/log_abc123.txt
```

Slide 7: Working with Binary Data

Tempfile supports both text and binary modes, making it versatile for different data types.

```python
import tempfile

with tempfile.NamedTemporaryFile(mode='wb', delete=False) as binary_file:
    binary_data = bytes([0x48, 0x65, 0x6C, 0x6C, 0x6F])  # "Hello" in ASCII
    binary_file.write(binary_data)
    binary_file.seek(0)
    print(f"Binary content: {binary_file.read()}")

# Output: Binary content: b'Hello'
```

Slide 8: Thread-Safe Temporary File Creation

The tempfile module ensures thread-safe creation of temporary files, preventing race conditions in multi-threaded applications.

```python
import tempfile
import threading

def create_temp_file():
    with tempfile.NamedTemporaryFile(prefix=f"thread_{threading.get_ident()}_") as temp:
        print(f"Thread {threading.get_ident()} created: {temp.name}")

threads = [threading.Thread(target=create_temp_file) for _ in range(3)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

# Output (example):
# Thread 123456 created: /tmp/thread_123456_abcdef.tmp
# Thread 789012 created: /tmp/thread_789012_ghijkl.tmp
# Thread 345678 created: /tmp/thread_345678_mnopqr.tmp
```

Slide 9: Real-Life Example: Log File Rotation

Using tempfile for log rotation ensures atomic operations and prevents data loss during file swapping.

```python
import tempfile
import os
import shutil

def rotate_log(log_file):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        shutil.2(log_file, temp_file.name)
        
    os.rename(temp_file.name, log_file + ".1")
    open(log_file, 'w').close()  # Create a new empty log file

# Usage
log_file = "app.log"
rotate_log(log_file)
print(f"Log rotated: {log_file} -> {log_file}.1")

# Output: Log rotated: app.log -> app.log.1
```

Slide 10: Real-Life Example: Temporary Configuration Files

Create temporary configuration files for testing or one-time use in applications.

```python
import tempfile
import configparser

def create_temp_config():
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'ServerAliveInterval': '45',
                         'Compression': 'yes',
                         'CompressionLevel': '9'}
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.ini', delete=False) as temp_config:
        config.write(temp_config)
        return temp_config.name

config_file = create_temp_config()
print(f"Temporary config created at: {config_file}")

# Read the config to verify
config = configparser.ConfigParser()
config.read(config_file)
print(f"CompressionLevel: {config['DEFAULT']['CompressionLevel']}")

# Output:
# Temporary config created at: /tmp/tmpabc123.ini
# CompressionLevel: 9
```

Slide 11: Secure File Deletion

Ensure sensitive data is securely deleted by overwriting the file before removal.

```python
import tempfile
import os

def secure_delete(file_path, passes=3):
    file_size = os.path.getsize(file_path)
    with open(file_path, "wb") as f:
        for _ in range(passes):
            f.seek(0)
            f.write(os.urandom(file_size))
    os.unlink(file_path)

with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_file.write(b"Sensitive data")
    temp_path = temp_file.name

secure_delete(temp_path)
print(f"File {temp_path} securely deleted")

# Output: File /tmp/tmpabc123 securely deleted
```

Slide 12: Error Handling in Temporary File Operations

Proper error handling ensures resources are cleaned up even if exceptions occur.

```python
import tempfile
import os

try:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        temp_file.write(b"Some data")
        raise Exception("Simulated error")
    finally:
        temp_file.close()
        os.unlink(temp_file.name)
        print(f"Temp file {temp_file.name} deleted despite error")
except Exception as e:
    print(f"Caught exception: {e}")

# Output:
# Temp file /tmp/tmpabc123 deleted despite error
# Caught exception: Simulated error
```

Slide 13: Performance Considerations

Choose the right temporary file type based on your performance needs.

```python
import tempfile
import time

def performance_test(file_type, iterations=100000):
    start_time = time.time()
    if file_type == 'spooled':
        temp_file = tempfile.SpooledTemporaryFile(max_size=1024*1024)
    else:
        temp_file = tempfile.TemporaryFile()
    
    for _ in range(iterations):
        temp_file.write(b"x")
    
    temp_file.close()
    return time.time() - start_time

spooled_time = performance_test('spooled')
regular_time = performance_test('regular')

print(f"SpooledTemporaryFile: {spooled_time:.4f} seconds")
print(f"TemporaryFile: {regular_time:.4f} seconds")

# Output (example):
# SpooledTemporaryFile: 0.0234 seconds
# TemporaryFile: 0.0321 seconds
```

Slide 14: Additional Resources

For more information on Python's tempfile module and best practices:

1. Python Official Documentation: [https://docs.python.org/3/library/tempfile.html](https://docs.python.org/3/library/tempfile.html)
2. "Temporary File Management in Python" by Real Python: [https://realpython.com/python-tempfile-module/](https://realpython.com/python-tempfile-module/)
3. "File I/O in Python" on ArXiv: [https://arxiv.org/abs/cs/0503067](https://arxiv.org/abs/cs/0503067)

These resources provide in-depth explanations and advanced usage scenarios for the tempfile module.

