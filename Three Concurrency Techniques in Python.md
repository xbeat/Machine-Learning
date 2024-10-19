## Three Concurrency Techniques in Python

Slide 1: Three Ways to Perform Concurrency in Python

Concurrency in Python allows multiple tasks to run simultaneously, improving efficiency and performance. This presentation explores three primary methods: Threading, Multiprocessing, and Asyncio. Each approach has its strengths and ideal use cases, which we'll examine in detail.

```python
import multiprocessing
import asyncio

# Placeholder for demonstration
def concurrent_task():
    pass

# Threading
thread = threading.Thread(target=concurrent_task)

# Multiprocessing
process = multiprocessing.Process(target=concurrent_task)

# Asyncio
async def async_task():
    await asyncio.sleep(1)

asyncio.run(async_task())
```

Slide 2: Threading: Lightweight Concurrency

Threading in Python allows multiple threads to run within a single process. It's particularly useful for I/O-bound tasks where the program spends time waiting for external operations. The threading module provides a high-level interface for creating and managing threads.

```python
import time

def worker(name):
    print(f"Worker {name} starting")
    time.sleep(2)  # Simulate I/O operation
    print(f"Worker {name} finished")

# Create and start two threads
thread1 = threading.Thread(target=worker, args=("A",))
thread2 = threading.Thread(target=worker, args=("B",))

thread1.start()
thread2.start()

thread1.join()
thread2.join()

print("All workers completed")
```

Slide 3: Threading: Real-Life Example - Web Scraping

Web scraping is an excellent use case for threading. It allows multiple web pages to be downloaded concurrently, significantly reducing the overall time for large-scale scraping tasks.

```python
import requests
import time

def fetch_url(url):
    response = requests.get(url)
    print(f"Fetched {url}, status: {response.status_code}")

urls = [
    "https://www.example.com",
    "https://www.example.org",
    "https://www.example.net"
]

threads = []
start_time = time.time()

for url in urls:
    thread = threading.Thread(target=fetch_url, args=(url,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")
```

Slide 4: Threading: Limitations and Considerations

While threading is useful for I/O-bound tasks, it has limitations due to Python's Global Interpreter Lock (GIL). The GIL ensures that only one thread executes Python bytecode at a time, which can limit performance gains for CPU-bound tasks.

```python
import time

def cpu_bound_task():
    count = 0
    for i in range(100_000_000):
        count += i
    return count

def run_tasks_sequentially():
    start = time.time()
    cpu_bound_task()
    cpu_bound_task()
    end = time.time()
    print(f"Sequential execution time: {end - start:.2f} seconds")

def run_tasks_threaded():
    start = time.time()
    t1 = threading.Thread(target=cpu_bound_task)
    t2 = threading.Thread(target=cpu_bound_task)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    end = time.time()
    print(f"Threaded execution time: {end - start:.2f} seconds")

run_tasks_sequentially()
run_tasks_threaded()
```

Slide 5: Multiprocessing: Harnessing Multiple CPU Cores

Multiprocessing in Python allows true parallelism by creating separate processes, each with its own Python interpreter and memory space. This approach bypasses the GIL and is ideal for CPU-bound tasks that require heavy computation.

```python
import time

def cpu_intensive_task(n):
    return sum(i * i for i in range(n))

if __name__ == '__main__':
    start_time = time.time()
    
    # Create a pool of worker processes
    with multiprocessing.Pool() as pool:
        results = pool.map(cpu_intensive_task, [10**7, 10**7, 10**7, 10**7])
    
    end_time = time.time()
    print(f"Results: {results}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
```

Slide 6: Multiprocessing: Process Communication

Multiprocessing requires careful handling of shared data. Python's multiprocessing module provides various mechanisms for inter-process communication, such as Queue and Pipe.

```python

def square_numbers(numbers, queue):
    for n in numbers:
        queue.put(n * n)

if __name__ == "__main__":
    numbers = range(1, 6)
    q = Queue()

    p = Process(target=square_numbers, args=(numbers, q))
    p.start()
    p.join()

    while not q.empty():
        print(q.get())
```

Slide 7: Multiprocessing: Real-Life Example - Image Processing

Image processing is a CPU-intensive task that benefits greatly from multiprocessing. Here's an example of applying a simple blur filter to multiple images concurrently.

```python
from PIL import Image, ImageFilter
import os

def apply_blur(image_path):
    with Image.open(image_path) as img:
        blurred = img.filter(ImageFilter.BLUR)
        output_path = f"blurred_{os.path.basename(image_path)}"
        blurred.save(output_path)
        return output_path

if __name__ == "__main__":
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]
    
    with multiprocessing.Pool() as pool:
        results = pool.map(apply_blur, image_paths)
    
    print("Blurred images:", results)
```

Slide 8: Asyncio: Event-Driven Programming

Asyncio is a Python library for writing concurrent code using the async/await syntax. It's based on coroutines and is particularly well-suited for I/O-bound tasks that involve waiting, such as network operations.

```python

async def fetch_data(url):
    print(f"Start fetching {url}")
    await asyncio.sleep(2)  # Simulate I/O operation
    print(f"Finished fetching {url}")
    return f"Data from {url}"

async def main():
    urls = ['http://example.com', 'http://example.org', 'http://example.net']
    tasks = [asyncio.create_task(fetch_data(url)) for url in urls]
    results = await asyncio.gather(*tasks)
    print(results)

asyncio.run(main())
```

Slide 9: Asyncio: Event Loop and Coroutines

Asyncio runs on an event loop that manages and executes asynchronous tasks. Coroutines are special functions that can be paused and resumed, allowing other tasks to run in the meantime.

```python

async def countdown(name, n):
    while n > 0:
        print(f"{name}: {n}")
        await asyncio.sleep(1)
        n -= 1

async def main():
    # Schedule two coroutines to run concurrently
    await asyncio.gather(
        countdown("Countdown A", 5),
        countdown("Countdown B", 3)
    )

asyncio.run(main())
```

Slide 10: Asyncio: Real-Life Example - Asynchronous Web Scraping

Asyncio shines in scenarios involving multiple network requests. Here's an example of asynchronous web scraping using aiohttp, a popular asyncio-compatible HTTP client library.

```python
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = [
        "https://www.example.com",
        "https://www.example.org",
        "https://www.example.net"
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    
    for url, html in zip(urls, results):
        print(f"{url}: {len(html)} characters")

asyncio.run(main())
```

Slide 11: Choosing the Right Concurrency Method

The choice between Threading, Multiprocessing, and Asyncio depends on the nature of your tasks:

1. Threading: Best for I/O-bound tasks with minimal CPU usage.
2. Multiprocessing: Ideal for CPU-bound tasks that can benefit from parallel processing.
3. Asyncio: Excellent for I/O-bound tasks, especially those involving many concurrent operations.

```python
import threading
import multiprocessing
import asyncio

def io_bound_task():
    time.sleep(1)  # Simulate I/O operation

def cpu_bound_task():
    for _ in range(10**7):
        _ = 1 + 1  # CPU-intensive operation

async def async_io_task():
    await asyncio.sleep(1)  # Simulate asynchronous I/O

def benchmark(func, n):
    start = time.time()
    func(n)
    end = time.time()
    print(f"{func.__name__}: {end - start:.2f} seconds")

def run_threading(n):
    threads = [threading.Thread(target=io_bound_task) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

def run_multiprocessing(n):
    with multiprocessing.Pool(n) as p:
        p.map(cpu_bound_task, range(n))

async def run_asyncio(n):
    await asyncio.gather(*[async_io_task() for _ in range(n)])

if __name__ == "__main__":
    n = 4
    benchmark(run_threading, n)
    benchmark(run_multiprocessing, n)
    benchmark(lambda x: asyncio.run(run_asyncio(x)), n)
```

Slide 12: Best Practices and Considerations

When working with concurrency in Python, keep these best practices in mind:

1. Choose the appropriate method based on your task type (I/O-bound vs CPU-bound).
2. Be aware of the GIL's impact on threading performance for CPU-bound tasks.
3. Use proper synchronization mechanisms (locks, semaphores) to avoid race conditions.
4. Consider the overhead of creating and managing threads or processes.
5. For asyncio, ensure all I/O operations use async-compatible libraries.

```python
import asyncio

# Example of using a lock in threading
lock = threading.Lock()

def thread_safe_function(shared_resource):
    with lock:
        # Access shared resource safely
        shared_resource.update()

# Example of using a semaphore in asyncio
sem = asyncio.Semaphore(3)  # Limit to 3 concurrent operations

async def controlled_async_task():
    async with sem:
        # Perform limited concurrent operation
        await asyncio.sleep(1)
```

Slide 13: Debugging and Profiling Concurrent Code

Debugging and profiling concurrent code can be challenging. Python provides tools to help:

1. Use logging to track execution flow across threads/processes.
2. The `multiprocessing.Manager` can help debug shared state in multiprocessing.
3. For asyncio, `asyncio.create_task()` allows naming tasks for easier debugging.
4. The `cProfile` module can profile threaded code, while `multiprocessing.Pool` has built-in profiling options.

```python
import asyncio
from multiprocessing import Pool, cpu_count
import cProfile

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(threadName)s: %(message)s')

# Asyncio task with name for debugging
async def named_task():
    task = asyncio.current_task()
    print(f"Running task: {task.get_name()}")
    await asyncio.sleep(1)

asyncio.run(named_task(), debug=True)

# Profiling a function
def cpu_bound_func():
    return sum(i * i for i in range(10**6))

cProfile.run('cpu_bound_func()')

# Profiling with multiprocessing
if __name__ == '__main__':
    with Pool(cpu_count()) as p:
        p.apply(cpu_bound_func)  # The worker process will be profiled
```

Slide 14: Additional Resources

For further exploration of concurrency in Python, consider these resources:

1. "Asyncio: Understanding Python's Asynchronous Programming" - arXiv:2012.15786 \[cs.PL\] [https://arxiv.org/abs/2012.15786](https://arxiv.org/abs/2012.15786)
2. "A Comparative Study on Concurrency and Parallelism in Python" - arXiv:2103.11627 \[cs.DC\] [https://arxiv.org/abs/2103.11627](https://arxiv.org/abs/2103.11627)

These papers provide in-depth analysis and comparisons of different concurrency methods in Python, offering valuable insights for both beginners and advanced programmers.


