## 3 Approaches to Concurrency in Python
Slide 1: Introduction to Concurrency in Python

Concurrency in Python allows multiple tasks to run seemingly simultaneously, improving performance and efficiency. This slideshow explores three main approaches to concurrency in Python: Threading, Multiprocessing, and Asyncio. Each method has its strengths and is suited for different types of tasks.

```python
import threading
import multiprocessing
import asyncio

print("Threading:", threading.__name__)
print("Multiprocessing:", multiprocessing.__name__)
print("Asyncio:", asyncio.__name__)
```

Slide 2: Threading - Basics

Threading in Python allows multiple threads of execution within a single process. It's particularly useful for I/O-bound tasks where operations spend time waiting for external resources. The `threading` module provides a high-level interface for working with threads.

```python
import threading
import time

def worker(name):
    print(f"Worker {name} starting")
    time.sleep(2)  # Simulate some work
    print(f"Worker {name} finished")

threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("All workers finished")
```

Slide 3: Threading - Real-life Example: Web Scraping

Web scraping is an excellent use case for threading. It allows multiple web pages to be fetched concurrently, significantly reducing the total time required.

```python
import threading
import requests
import time

def fetch_url(url):
    response = requests.get(url)
    print(f"Fetched {url}, status code: {response.status_code}")

urls = [
    "https://python.org",
    "https://pypi.org",
    "https://docs.python.org",
]

start_time = time.time()

threads = []
for url in urls:
    t = threading.Thread(target=fetch_url, args=(url,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")
```

Slide 4: Threading - Challenges and GIL

While threading is useful for I/O-bound tasks, it's limited by the Global Interpreter Lock (GIL) in CPython. The GIL ensures that only one thread executes Python bytecode at a time, which can limit performance for CPU-bound tasks.

```python
import threading
import time

def cpu_bound_task():
    count = 0
    for i in range(10**7):
        count += i

start_time = time.time()

threads = []
for _ in range(4):
    t = threading.Thread(target=cpu_bound_task)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

end_time = time.time()
print(f"Time taken with threading: {end_time - start_time:.2f} seconds")

# Compare with sequential execution
start_time = time.time()
for _ in range(4):
    cpu_bound_task()
end_time = time.time()
print(f"Time taken sequentially: {end_time - start_time:.2f} seconds")
```

Slide 5: Multiprocessing - Basics

Multiprocessing in Python uses separate processes instead of threads, each with its own Python interpreter and memory space. This approach bypasses the GIL, making it ideal for CPU-bound tasks. The `multiprocessing` module provides a high-level interface for working with processes.

```python
import multiprocessing
import time

def worker(name):
    print(f"Worker {name} starting")
    time.sleep(2)  # Simulate some work
    print(f"Worker {name} finished")

if __name__ == '__main__':
    processes = []
    for i in range(3):
        p = multiprocessing.Process(target=worker, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All workers finished")
```

Slide 6: Multiprocessing - Real-life Example: Image Processing

Image processing is a CPU-intensive task that benefits from multiprocessing. Here's an example of applying a simple filter to multiple images concurrently.

```python
import multiprocessing
import time
from PIL import Image, ImageFilter

def apply_filter(image_path):
    with Image.open(image_path) as img:
        filtered = img.filter(ImageFilter.BLUR)
        filtered.save(f"blurred_{image_path}")
    print(f"Processed {image_path}")

if __name__ == '__main__':
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]

    start_time = time.time()

    with multiprocessing.Pool() as pool:
        pool.map(apply_filter, image_paths)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
```

Slide 7: Multiprocessing - Shared Memory and Locks

Multiprocessing allows for shared memory between processes, but care must be taken to avoid race conditions. Here's an example using a shared value and a lock.

```python
import multiprocessing

def increment(counter, lock):
    for _ in range(10000):
        with lock:
            counter.value += 1

if __name__ == '__main__':
    counter = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()

    p1 = multiprocessing.Process(target=increment, args=(counter, lock))
    p2 = multiprocessing.Process(target=increment, args=(counter, lock))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print(f"Final counter value: {counter.value}")
```

Slide 8: Asyncio - Basics

Asyncio is a library for writing concurrent code using the async/await syntax. It's based on coroutines and an event loop, making it excellent for I/O-bound tasks without the overhead of threads or processes.

```python
import asyncio

async def say_hello(name, delay):
    await asyncio.sleep(delay)  # Non-blocking sleep
    print(f"Hello, {name}!")

async def main():
    tasks = [
        say_hello("Alice", 2),
        say_hello("Bob", 1),
        say_hello("Charlie", 3)
    ]
    await asyncio.gather(*tasks)

asyncio.run(main())
```

Slide 9: Asyncio - Coroutines and Tasks

Coroutines are the building blocks of asyncio-based programs. They can be scheduled as tasks on the event loop. Here's an example demonstrating coroutines and task creation.

```python
import asyncio

async def countdown(name, n):
    while n > 0:
        print(f"{name}: {n}")
        await asyncio.sleep(1)
        n -= 1
    print(f"{name}: Countdown finished!")

async def main():
    # Create tasks
    task1 = asyncio.create_task(countdown("Countdown 1", 3))
    task2 = asyncio.create_task(countdown("Countdown 2", 5))

    # Wait for both tasks to complete
    await asyncio.gather(task1, task2)

asyncio.run(main())
```

Slide 10: Asyncio - Real-life Example: Asynchronous Web Scraping

Asyncio shines in scenarios involving many I/O operations, such as web scraping. Here's an example of asynchronous web scraping using aiohttp.

```python
import asyncio
import aiohttp
import time

async def fetch_url(session, url):
    async with session.get(url) as response:
        print(f"Fetched {url}, status code: {response.status}")

async def main():
    urls = [
        "https://python.org",
        "https://pypi.org",
        "https://docs.python.org",
        "https://github.com",
        "https://stackoverflow.com"
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        await asyncio.gather(*tasks)

start_time = time.time()
asyncio.run(main())
end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")
```

Slide 11: Asyncio - Event Loop and Concurrency

The event loop is the core of asyncio's concurrency model. It manages and schedules asynchronous tasks. Here's an example demonstrating the event loop and task scheduling.

```python
import asyncio

async def task(name, duration):
    print(f"Task {name} starting")
    await asyncio.sleep(duration)
    print(f"Task {name} completed after {duration} seconds")

async def main():
    # Schedule tasks with different durations
    task1 = asyncio.create_task(task("A", 2))
    task2 = asyncio.create_task(task("B", 1))
    task3 = asyncio.create_task(task("C", 3))

    # Wait for all tasks to complete
    await asyncio.gather(task1, task2, task3)

asyncio.run(main())
```

Slide 12: Choosing the Right Concurrency Approach

The choice between Threading, Multiprocessing, and Asyncio depends on the nature of your tasks:

*   Use Threading for I/O-bound tasks with minimal CPU usage.
*   Use Multiprocessing for CPU-bound tasks that can benefit from parallel execution.
*   Use Asyncio for I/O-bound tasks that involve a lot of waiting and can be expressed as coroutines.

Here's a simple decision tree to help choose:

```python
def choose_concurrency_approach(task_type, io_bound, cpu_bound):
    if io_bound and not cpu_bound:
        if task_type == "network":
            return "Asyncio"
        else:
            return "Threading"
    elif cpu_bound:
        return "Multiprocessing"
    else:
        return "Sequential execution"

print(choose_concurrency_approach("network", True, False))  # Asyncio
print(choose_concurrency_approach("file_io", True, False))  # Threading
print(choose_concurrency_approach("computation", False, True))  # Multiprocessing
```

Slide 13: Concurrency Best Practices

When working with concurrency in Python, keep these best practices in mind:

1.  Use the appropriate concurrency model for your task type.
2.  Be aware of the GIL's limitations when using threading.
3.  Handle exceptions properly in concurrent code.
4.  Use synchronization primitives (locks, semaphores) to prevent race conditions.
5.  Consider using higher-level abstractions like ThreadPoolExecutor or ProcessPoolExecutor for simpler management of concurrent tasks.

```python
import concurrent.futures
import time

def task(n):
    time.sleep(n)
    return f"Task {n} completed"

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(task, i) for i in range(1, 4)]
    for future in concurrent.futures.as_completed(futures):
        print(future.result())
```

Slide 14: Additional Resources

For further exploration of concurrency in Python, consider these resources:

1.  "Asynchronous Programming in Python" by Caleb Hattingh (ArXiv:1901.03560) [https://arxiv.org/abs/1901.03560](https://arxiv.org/abs/1901.03560)
2.  "Python Concurrency with asyncio" by Matthew Fowler (Not an ArXiv source, but a comprehensive book on asyncio)
3.  Python's official documentation on concurrency: [https://docs.python.org/3/library/concurrency.html](https://docs.python.org/3/library/concurrency.html)

These resources provide in-depth coverage of the topics we've introduced, offering advanced techniques and best practices for concurrent programming in Python.

