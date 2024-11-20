## Python File Handling Mastery Key Methods Explained
Slide 1: Basic File Reading Operations

Reading files in Python involves understanding fundamental operations that allow us to extract content efficiently. The read() method loads the entire file content into memory as a single string, making it suitable for smaller files where memory constraints aren't a concern.

```python
# Opening and reading a file using different methods
with open('example.txt', 'r') as file:
    # Read entire file content
    content = file.read()
    print("Full content:", content)
    
    # Return to beginning of file
    file.seek(0)
    
    # Read first line
    first_line = file.readline()
    print("\nFirst line:", first_line)

# Output example:
# Full content: Hello World!
# This is a sample file.
# 
# First line: Hello World!
```

Slide 2: Reading Files Line by Line

When dealing with larger files, reading line by line becomes crucial for memory efficiency. The readlines() method returns a list containing each line as a separate string, while readline() allows controlled iteration through the file content.

```python
def process_large_file(filename):
    with open(filename, 'r') as file:
        # Method 1: Using readline()
        line = file.readline()
        while line:
            print("Processing:", line.strip())
            line = file.readline()
            
        file.seek(0)  # Reset file pointer
        
        # Method 2: Using readlines()
        all_lines = file.readlines()
        print("\nTotal lines:", len(all_lines))

# Example usage
process_large_file('data.txt')
```

Slide 3: Writing Operations

File writing in Python enables creating and modifying files through write() and writelines() methods. Understanding these operations is essential for data persistence and logging applications in production environments.

```python
def write_example():
    # Writing single strings
    with open('output.txt', 'w') as file:
        file.write("First line\n")
        file.write("Second line\n")
    
    # Writing multiple lines at once
    lines = ['Line 1\n', 'Line 2\n', 'Line 3\n']
    with open('output.txt', 'a') as file:
        file.writelines(lines)

# Create and write to file
write_example()
```

Slide 4: File Modes and Context Managers

Python's file handling implements various modes for different operations, while context managers ensure proper resource management. Understanding these concepts is crucial for writing robust file handling code.

```python
def demonstrate_file_modes():
    # Write binary data
    with open('binary.dat', 'wb') as file:
        file.write(b'Binary content')
    
    # Read and write (r+)
    with open('text.txt', 'r+') as file:
        content = file.read()
        file.seek(0)
        file.write("New content")
        
    # Append mode
    with open('log.txt', 'a') as file:
        file.write("\nNew log entry")
```

Slide 5: File Navigation with seek() and tell()

File pointer manipulation allows precise control over file reading and writing operations. The seek() method positions the file pointer, while tell() reports the current position, enabling sophisticated file processing strategies.

```python
def demonstrate_navigation():
    with open('sample.txt', 'r+') as file:
        # Get current position
        position = file.tell()
        print(f"Initial position: {position}")
        
        # Move to specific position
        file.seek(5)
        print(f"Content from position 5: {file.read(10)}")
        
        # Move relative to current position
        file.seek(0, 2)  # Seek to end
        print(f"End position: {file.tell()}")
```

Slide 6: File Buffering and Flush Operations

Understanding buffer management is crucial for optimal file I/O performance. The flush() method forces writing of buffered data to disk, essential for ensuring data persistence in critical applications and preventing data loss.

```python
def demonstrate_buffering():
    # Open file with custom buffer size
    with open('buffered.txt', 'w', buffering=8192) as file:
        file.write("Critical data\n")
        # Force write to disk
        file.flush()
        
        # Writing more data
        for i in range(1000):
            file.write(f"Line {i}\n")
            if i % 100 == 0:
                file.flush()  # Periodic flush
```

Slide 7: Error Handling in File Operations

Robust file handling requires comprehensive error management. Python provides specific exceptions for file operations that allow graceful handling of common scenarios like missing files or permission issues.

```python
def safe_file_operations(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: {filename} does not exist")
        return None
    except PermissionError:
        print(f"Error: No permission to read {filename}")
        return None
    except IOError as e:
        print(f"I/O error occurred: {e}")
        return None
    else:
        return content
```

Slide 8: Real-world Example: Log File Analysis

Implementing a log file analyzer demonstrates practical application of file operations. This example processes a log file, extracts relevant information, and generates statistical analysis of the data.

```python
def analyze_log_file(log_path):
    error_count = {}
    timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
    
    with open(log_path, 'r') as log_file:
        for line in log_file:
            if 'ERROR' in line:
                # Extract error type
                error_type = line.split('ERROR:')[1].strip()
                error_count[error_type] = error_count.get(error_type, 0) + 1
    
    # Generate report
    with open('error_report.txt', 'w') as report:
        report.write("Error Analysis Report\n")
        for error_type, count in error_count.items():
            report.write(f"{error_type}: {count} occurrences\n")

    return error_count
```

Slide 9: Binary File Operations

Binary file operations are essential for handling non-text data such as images, executables, or custom data formats. Understanding binary read/write operations enables development of sophisticated data processing applications.

```python
def process_binary_file():
    # Writing binary data
    data = bytes([0x50, 0x51, 0x52, 0x53])
    with open('binary_data.bin', 'wb') as bin_file:
        bin_file.write(data)
    
    # Reading binary data
    with open('binary_data.bin', 'rb') as bin_file:
        chunk_size = 2
        while True:
            chunk = bin_file.read(chunk_size)
            if not chunk:
                break
            print(f"Chunk: {chunk.hex()}")
```

Slide 10: Memory-Efficient File Processing

Large file processing requires memory-efficient strategies. This implementation demonstrates processing gigabyte-sized files without loading the entire content into memory, using generators and chunked reading.

```python
def process_large_file_efficiently(filepath, chunk_size=8192):
    def file_generator(file_obj):
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            yield chunk
    
    total_processed = 0
    with open(filepath, 'rb') as file:
        for chunk in file_generator(file):
            # Process each chunk
            processed_size = len(chunk)
            total_processed += processed_size
            
            # Update progress
            if total_processed % (chunk_size * 100) == 0:
                print(f"Processed: {total_processed / 1024 / 1024:.2f} MB")
```

Slide 11: Real-time File Monitoring

Implementing a file monitoring system demonstrates advanced file operations for real-time log analysis and system monitoring. This implementation uses file position tracking and periodic checking for changes.

```python
import time
import os

def monitor_file(filename, interval=1.0):
    # Get initial file size
    file_size = os.path.getsize(filename)
    
    with open(filename, 'r') as file:
        # Move to end of file
        file.seek(0, 2)
        
        while True:
            current_size = os.path.getsize(filename)
            if current_size > file_size:
                # Read new content
                new_content = file.read()
                print(f"New content: {new_content}")
                file_size = current_size
            time.sleep(interval)
```

Slide 12: CSV File Processing Implementation

CSV file handling requires specialized approaches for efficient data processing. This implementation showcases robust CSV reading and writing with error handling and data validation.

```python
def process_csv_file(input_file, output_file):
    import csv
    
    processed_rows = []
    try:
        with open(input_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Process each row
                processed_row = {
                    key: float(value) if value.replace('.','').isdigit() else value
                    for key, value in row.items()
                }
                processed_rows.append(processed_row)
        
        # Write processed data
        with open(output_file, 'w', newline='') as csvfile:
            if processed_rows:
                writer = csv.DictWriter(csvfile, fieldnames=processed_rows[0].keys())
                writer.writeheader()
                writer.writerows(processed_rows)
                
    except csv.Error as e:
        print(f"CSV processing error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
```

Slide 13: File Compression Handling

Working with compressed files requires special handling techniques. This implementation demonstrates reading and writing compressed files efficiently using Python's built-in compression libraries.

```python
import gzip
import shutil

def handle_compressed_files(input_file, compressed_file):
    # Compress file
    with open(input_file, 'rb') as f_in:
        with gzip.open(compressed_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Read compressed file
    with gzip.open(compressed_file, 'rb') as f:
        file_content = f.read().decode('utf-8')
        
    # Process compressed file line by line
    with gzip.open(compressed_file, 'rt') as f:
        for line in f:
            print(f"Processing line: {line.strip()}")
```

Slide 14: Additional Resources

*   Research on Efficient File Processing:
*   [https://arxiv.org/abs/2104.07935](https://arxiv.org/abs/2104.07935)
*   [https://arxiv.org/abs/2003.02645](https://arxiv.org/abs/2003.02645)
*   [https://arxiv.org/abs/1909.08725](https://arxiv.org/abs/1909.08725)
*   General File Operation Resources:
*   [https://docs.python.org/3/tutorial/inputoutput.html](https://docs.python.org/3/tutorial/inputoutput.html)
*   [https://realpython.com/working-with-files-in-python/](https://realpython.com/working-with-files-in-python/)
*   [https://www.geeksforgeeks.org/file-handling-python/](https://www.geeksforgeeks.org/file-handling-python/)
*   Performance Optimization Guidelines:
*   Search "Python File I/O Performance Optimization" on Google Scholar
*   Visit Python's official documentation for advanced file handling
*   Explore PyPI for specialized file handling packages

