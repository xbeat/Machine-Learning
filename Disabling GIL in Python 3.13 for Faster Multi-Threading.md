## Disabling GIL in Python 3.13 for Faster Multi-Threading
Slide 1: Python 3.13 and the Global Interpreter Lock (GIL)

Python 3.13 introduces a significant change in how it handles multi-threading. The Global Interpreter Lock (GIL), a mechanism that prevents multiple native threads from executing Python bytecodes at once, can now be disabled in certain scenarios. This change aims to improve performance in multi-threaded applications, especially those that are CPU-bound.

Slide 2: Source Code for Python 3.13 and the Global Interpreter Lock (GIL)

```python
import sys

print(f"Python version: {sys.version}")
print(f"GIL status: {'Disabled' if sys.flags.nogil else 'Enabled'}")

# Note: This code will only work in Python 3.13 or later
# The output will depend on whether the script is run with 'python3.13' or 'python3.13t'
```

Slide 3: Understanding the GIL

The Global Interpreter Lock (GIL) is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecodes at once. While it simplifies the implementation of CPython and helps in memory management, it can limit the performance of multi-threaded CPU-bound programs.

Slide 4: Source Code for Understanding the GIL

```python
import threading
import time

def cpu_bound_task(n):
    while n > 0:
        n -= 1

def run_threads(num_threads, iterations):
    threads = []
    start_time = time.time()
    
    for _ in range(num_threads):
        t = threading.Thread(target=cpu_bound_task, args=(iterations,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    end_time = time.time()
    print(f"Time taken with {num_threads} threads: {end_time - start_time:.2f} seconds")

# Run with 1 thread and 4 threads
run_threads(1, 100_000_000)
run_threads(4, 100_000_000)

# Note: With GIL enabled, the 4-thread version might not be significantly faster
```

Slide 5: Introducing python3.13t

Python 3.13 introduces a new executable called "python3.13t" that runs Python with the GIL disabled. This allows true parallel execution of Python code on multiple CPU cores, potentially leading to significant performance improvements for multi-threaded, CPU-bound tasks.

Slide 6: Source Code for Introducing python3.13t

```python
import os
import sys
import subprocess

def run_script(script_name):
    for interpreter in ['python3.13', 'python3.13t']:
        result = subprocess.run([interpreter, script_name], 
                                capture_output=True, text=True)
        print(f"Results for {interpreter}:")
        print(result.stdout)
        print("-" * 40)

# Assume we have a script called 'multi_threaded_task.py'
run_script('multi_threaded_task.py')

# Note: This code runs the same script with both python3.13 and python3.13t
# The actual performance difference will depend on the content of multi_threaded_task.py
```

Slide 7: Real-life Example: Image Processing

Let's consider a real-life example of image processing, where we apply a simple blur effect to multiple images concurrently. This CPU-bound task can benefit from true multi-threading.

Slide 8: Source Code for Real-life Example: Image Processing

```python
import threading
import time

def apply_blur(image):
    # Simulating a CPU-intensive task
    for _ in range(10_000_000):
        pass
    print(f"Blurred image: {image}")

def process_images(images):
    threads = []
    for img in images:
        t = threading.Thread(target=apply_blur, args=(img,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()

images = [f"image_{i}.jpg" for i in range(10)]

start_time = time.time()
process_images(images)
end_time = time.time()

print(f"Total time: {end_time - start_time:.2f} seconds")

# Note: Run this with python3.13 and python3.13t to see the difference
```

Slide 9: Real-life Example: Scientific Computation

Another area where disabling the GIL can be beneficial is scientific computation. Let's consider a simple example of calculating prime numbers in parallel.

Slide 10: Source Code for Real-life Example: Scientific Computation

```python
import threading
import time

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_primes(start, end):
    return [n for n in range(start, end) if is_prime(n)]

def parallel_find_primes(ranges):
    threads = []
    results = []
    
    for start, end in ranges:
        results.append([])
        t = threading.Thread(target=lambda: results[-1].extend(find_primes(start, end)))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    return [prime for sublist in results for prime in sublist]

ranges = [(1, 25000), (25001, 50000), (50001, 75000), (75001, 100000)]

start_time = time.time()
primes = parallel_find_primes(ranges)
end_time = time.time()

print(f"Found {len(primes)} primes in {end_time - start_time:.2f} seconds")

# Note: Run this with python3.13 and python3.13t to see the difference
```

Slide 11: Limitations and Considerations

While disabling the GIL can improve performance for CPU-bound tasks, it's not a silver bullet. I/O-bound tasks may not see significant improvements. Additionally, disabling the GIL can introduce new challenges, such as race conditions and the need for explicit synchronization in some cases.

Slide 12: Source Code for Limitations and Considerations

```python
import threading
import time

shared_counter = 0
lock = threading.Lock()

def increment_counter():
    global shared_counter
    for _ in range(1_000_000):
        # Uncomment the next line to fix the race condition
        # with lock:
        shared_counter += 1

threads = [threading.Thread(target=increment_counter) for _ in range(4)]

start_time = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()
end_time = time.time()

print(f"Final counter value: {shared_counter}")
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Note: Without proper synchronization, the final counter value may be incorrect
# Run this with python3.13t to see the effects of race conditions
```

Slide 13: Future of Python Threading

The ability to disable the GIL in Python 3.13 is a significant step towards improving Python's performance in multi-threaded scenarios. As this feature matures, we can expect to see more libraries and frameworks taking advantage of true parallelism in Python, potentially changing how we approach certain types of problems in the language.

Slide 14: Additional Resources

For more information on Python 3.13's GIL changes and their implications, refer to the following resources:

1.  Python Enhancement Proposal (PEP) 703: Making the Global Interpreter Lock Optional in CPython [https://peps.python.org/pep-0703/](https://peps.python.org/pep-0703/)
2.  "The Future of Python: Eliminating the Global Interpreter Lock" by Sam Gross [https://arxiv.org/abs/2205.11068](https://arxiv.org/abs/2205.11068)

These resources provide in-depth discussions on the technical aspects and potential impact of GIL-free Python execution.

