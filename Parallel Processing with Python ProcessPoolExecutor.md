## Parallel Processing with Python ProcessPoolExecutor
Slide 1: Introduction to ProcessPoolExecutor

ProcessPoolExecutor is a powerful tool in Python's concurrent.futures module. It allows you to parallelize CPU-bound tasks by utilizing multiple processes. This class is particularly useful when you need to perform computationally intensive operations that can benefit from parallel execution across multiple CPU cores.

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def cpu_bound_task(x):
    return sum(i * i for i in range(x))

# Get the number of CPU cores
num_cores = multiprocessing.cpu_count()

# Create a ProcessPoolExecutor with the number of cores
with ProcessPoolExecutor(max_workers=num_cores) as executor:
    results = list(executor.map(cpu_bound_task, [10**6, 10**7, 10**8]))

print(f"Results: {results}")
```

Slide 2: Basic Usage of ProcessPoolExecutor

The ProcessPoolExecutor class provides a simple interface for parallel execution. You can create an instance of the executor, submit tasks, and retrieve results. The executor manages a pool of worker processes, distributing tasks among them efficiently.

```python
from concurrent.futures import ProcessPoolExecutor

def square(x):
    return x * x

with ProcessPoolExecutor() as executor:
    future = executor.submit(square, 5)
    result = future.result()

print(f"Result: {result}")  # Output: Result: 25
```

Slide 3: Submitting Multiple Tasks

You can submit multiple tasks to the ProcessPoolExecutor using the submit() method. This allows you to execute several functions concurrently and retrieve their results as they become available.

```python
from concurrent.futures import ProcessPoolExecutor
import time

def slow_task(seconds):
    time.sleep(seconds)
    return f"Task completed in {seconds} seconds"

with ProcessPoolExecutor() as executor:
    future1 = executor.submit(slow_task, 2)
    future2 = executor.submit(slow_task, 1)
    
    print(future1.result())  # Waits for 2 seconds
    print(future2.result())  # Already completed, returns immediately
```

Slide 4: Using map() for Parallel Execution

The map() method of ProcessPoolExecutor allows you to apply a function to an iterable of inputs in parallel. This is particularly useful when you have a large number of similar tasks to process.

```python
from concurrent.futures import ProcessPoolExecutor

def cube(x):
    return x ** 3

numbers = range(10)

with ProcessPoolExecutor() as executor:
    results = executor.map(cube, numbers)

print(list(results))  # Output: [0, 1, 8, 27, 64, 125, 216, 343, 512, 729]
```

Slide 5: Handling Exceptions in ProcessPoolExecutor

When using ProcessPoolExecutor, exceptions raised in worker processes are propagated to the main process. It's important to handle these exceptions to ensure robust error management in your parallel code.

```python
from concurrent.futures import ProcessPoolExecutor

def risky_function(x):
    if x == 0:
        raise ValueError("Cannot process zero")
    return 1 / x

with ProcessPoolExecutor() as executor:
    futures = [executor.submit(risky_function, i) for i in range(5)]
    
    for future in futures:
        try:
            result = future.result()
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error occurred: {str(e)}")
```

Slide 6: Cancelling Tasks

ProcessPoolExecutor allows you to cancel submitted tasks that haven't started execution yet. This feature is useful when you need to stop processing based on certain conditions or user input.

```python
from concurrent.futures import ProcessPoolExecutor
import time

def long_running_task(seconds):
    time.sleep(seconds)
    return f"Task completed after {seconds} seconds"

with ProcessPoolExecutor() as executor:
    future1 = executor.submit(long_running_task, 10)
    future2 = executor.submit(long_running_task, 5)
    
    # Cancel the first task
    cancelled = future1.cancel()
    print(f"Task 1 cancelled: {cancelled}")
    
    # Try to cancel the second task (already running)
    cancelled = future2.cancel()
    print(f"Task 2 cancelled: {cancelled}")
    
    print(future2.result())
```

Slide 7: Context Manager and Resource Management

Using ProcessPoolExecutor with a context manager (the 'with' statement) ensures proper resource management. It automatically shuts down the executor and releases system resources when the block is exited.

```python
from concurrent.futures import ProcessPoolExecutor
import os

def get_process_id():
    return os.getpid()

with ProcessPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(get_process_id) for _ in range(5)]
    
    for future in futures:
        print(f"Task executed in process: {future.result()}")

# The executor is automatically shut down here
print("Executor has been shut down")
```

Slide 8: Controlling the Number of Workers

You can control the number of worker processes in the ProcessPoolExecutor by specifying the max\_workers parameter. This allows you to optimize resource usage based on your system's capabilities and the nature of your tasks.

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time

def cpu_intensive_task(x):
    start = time.time()
    result = sum(i * i for i in range(x))
    end = time.time()
    return f"Task completed in {end - start:.2f} seconds"

num_cores = multiprocessing.cpu_count()
print(f"Number of CPU cores: {num_cores}")

with ProcessPoolExecutor(max_workers=num_cores) as executor:
    futures = [executor.submit(cpu_intensive_task, 10**7) for _ in range(num_cores * 2)]
    
    for future in futures:
        print(future.result())
```

Slide 9: Real-Life Example: Image Processing

ProcessPoolExecutor can significantly speed up image processing tasks. In this example, we'll use it to apply a simple filter to multiple images concurrently.

```python
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import os

def apply_sepia_filter(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        pixels = img.load()
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                pixels[x, y] = (min(tr, 255), min(tg, 255), min(tb, 255))
        output_path = f"sepia_{os.path.basename(image_path)}"
        img.save(output_path)
    return output_path

image_files = ["image1.jpg", "image2.jpg", "image3.jpg"]

with ProcessPoolExecutor() as executor:
    results = list(executor.map(apply_sepia_filter, image_files))

print(f"Processed images: {results}")
```

Slide 10: Real-Life Example: Web Scraping

ProcessPoolExecutor can be used to parallelize web scraping tasks, significantly reducing the time required to fetch data from multiple web pages.

```python
from concurrent.futures import ProcessPoolExecutor
import requests
from bs4 import BeautifulSoup

def fetch_and_parse(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('title').text if soup.find('title') else "No title found"
    return f"{url}: {title}"

urls = [
    "https://www.python.org",
    "https://www.github.com",
    "https://www.stackoverflow.com",
    "https://www.reddit.com"
]

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(fetch_and_parse, urls))

for result in results:
    print(result)
```

Slide 11: Combining ProcessPoolExecutor with ThreadPoolExecutor

In some scenarios, you might need to combine CPU-bound and I/O-bound tasks. You can nest ProcessPoolExecutor within ThreadPoolExecutor to achieve this, leveraging both multiprocessing and multithreading.

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

def cpu_bound_task(x):
    return sum(i * i for i in range(x))

def io_bound_task(x):
    time.sleep(1)  # Simulate I/O operation
    return f"I/O task {x} completed"

def process_task(x):
    with ThreadPoolExecutor(max_workers=2) as thread_executor:
        future1 = thread_executor.submit(cpu_bound_task, x)
        future2 = thread_executor.submit(io_bound_task, x)
        return future1.result(), future2.result()

with ProcessPoolExecutor(max_workers=2) as process_executor:
    results = list(process_executor.map(process_task, range(4)))

for result in results:
    print(result)
```

Slide 12: Performance Considerations

When using ProcessPoolExecutor, it's important to consider the overhead of process creation and inter-process communication. For small tasks, this overhead might outweigh the benefits of parallelization.

```python
from concurrent.futures import ProcessPoolExecutor
import time

def quick_task(x):
    return x * 2

def measure_time(func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()
    return result, end - start

# Sequential execution
seq_result, seq_time = measure_time(lambda: [quick_task(i) for i in range(1000)])

# Parallel execution
def parallel_execution():
    with ProcessPoolExecutor() as executor:
        return list(executor.map(quick_task, range(1000)))

par_result, par_time = measure_time(parallel_execution)

print(f"Sequential time: {seq_time:.4f} seconds")
print(f"Parallel time: {par_time:.4f} seconds")
```

Slide 13: Best Practices and Limitations

When working with ProcessPoolExecutor, it's crucial to understand its limitations and follow best practices for optimal performance and reliability.

```python
from concurrent.futures import ProcessPoolExecutor
import os

def cpu_bound_task(x):
    return sum(i * i for i in range(x))

def io_bound_task(x):
    with open(f"temp_file_{x}.txt", "w") as f:
        f.write(f"Data for task {x}")
    os.remove(f"temp_file_{x}.txt")
    return f"I/O task {x} completed"

# Good practice: Use ProcessPoolExecutor for CPU-bound tasks
with ProcessPoolExecutor() as executor:
    cpu_results = list(executor.map(cpu_bound_task, [10**6, 10**7, 10**8]))

# Not recommended: Using ProcessPoolExecutor for I/O-bound tasks
with ProcessPoolExecutor() as executor:
    io_results = list(executor.map(io_bound_task, range(5)))

print("CPU-bound results:", cpu_results)
print("I/O-bound results:", io_results)
```

Slide 14: Additional Resources

For further exploration of ProcessPoolExecutor and parallel processing in Python, consider the following resources:

1. Python's official documentation on concurrent.futures: [https://docs.python.org/3/library/concurrent.futures.html](https://docs.python.org/3/library/concurrent.futures.html)
2. "Parallel Processing in Python: A Practical Guide with Examples" by Real Python: [https://realpython.com/python-concurrency/](https://realpython.com/python-concurrency/)
3. "Multiprocessing in Python: The Complete Guide" by DataCamp: [https://www.datacamp.com/community/tutorials/multiprocessing-in-python](https://www.datacamp.com/community/tutorials/multiprocessing-in-python)
4. "Python Parallel Programming Cookbook" by Giancarlo Zaccone (Book)
5. "High Performance Python: Practical Performant Programming for Humans" by Micha Gorelick and Ian Ozsvald (Book)

These resources provide in-depth explanations, advanced techniques, and real-world applications of parallel processing in Python, including the use of ProcessPoolExecutor.

