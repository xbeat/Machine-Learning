## Timing Python Code with timeit
Slide 1: Introduction to timeit

The timeit module in Python is a powerful tool for measuring the execution time of small code snippets. It provides an easy and accurate way to benchmark your code, helping you optimize performance and make informed decisions about implementation choices.

```python
import timeit

# Basic usage of timeit
execution_time = timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)
print(f"Execution time: {execution_time:.6f} seconds")
```

Slide 2: Why Use timeit?

timeit offers several advantages over simple time measurements:

1.  Accuracy: It minimizes distortions from system load or cache effects.
2.  Repeatability: It runs the code multiple times for a more accurate average.
3.  Granularity: It's optimized for timing small Python code segments.

```python
import time

# Compare timeit with simple time measurement
def simple_timing():
    start = time.time()
    "-".join(str(n) for n in range(100))
    end = time.time()
    return end - start

print(f"Simple timing: {simple_timing():.6f} seconds")
print(f"timeit: {timeit.timeit('"-".join(str(n) for n in range(100))', number=1):.6f} seconds")
```

Slide 3: Basic Syntax of timeit

The timeit module provides two main ways to measure code performance: the timeit() function and the Timer class. Here's the basic syntax for using timeit():

```python
import timeit

# Basic syntax
result = timeit.timeit(stmt='code_to_measure', setup='setup_code', number=number_of_executions)

# Example
result = timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)
print(f"Execution time: {result:.6f} seconds")
```

Slide 4: Using setup Code

The setup parameter in timeit allows you to define any necessary imports or variable initializations before running the timed code.

```python
import timeit

# Using setup code
setup_code = '''
import random
data = [random.randint(1, 100) for _ in range(1000)]
'''

stmt = 'sorted(data)'

result = timeit.timeit(stmt=stmt, setup=setup_code, number=1000)
print(f"Execution time: {result:.6f} seconds")
```

Slide 5: Timing Functions with timeit

To time a function using timeit, you can pass the function call as a string or use the globals() parameter to access the function directly.

```python
import timeit

def test_function():
    return sum(range(1000))

# Timing a function
result = timeit.timeit('test_function()', globals=globals(), number=10000)
print(f"Execution time: {result:.6f} seconds")
```

Slide 6: Using the Timer Class

The Timer class provides more flexibility and control over the timing process. You can create a Timer object and use its methods to run the timing multiple times or get more detailed statistics.

```python
import timeit

timer = timeit.Timer('"-".join(str(n) for n in range(100))')

# Run once
print(f"Single run: {timer.timeit(number=1):.6f} seconds")

# Run multiple times
print(f"Average of 3 runs: {timer.repeat(repeat=3, number=10000)}")
```

Slide 7: Command-line Interface

timeit also provides a command-line interface for quick timing of Python expressions. You can use it directly from the terminal without writing a full Python script.

```python
# This is not executable Python code, but a command-line example
# Run this in your terminal:
# python -m timeit "'-'.join(str(n) for n in range(100))"

# Example output:
# 100000 loops, best of 5: 3.39 usec per loop
```

Slide 8: Real-life Example: String Concatenation

Let's compare different methods of string concatenation using timeit to find the most efficient approach.

```python
import timeit

setup_code = '''
words = ["Python", "is", "awesome", "for", "programming"]
'''

print(timeit.timeit("''.join(words)", setup=setup_code, number=100000))
print(timeit.timeit("result = ''; [result += word for word in words]", setup=setup_code, number=100000))
print(timeit.timeit("result = ''; for word in words: result += word", setup=setup_code, number=100000))
```

Slide 9: Results for: Real-life Example: String Concatenation

```
0.004876100000000001
0.06876390000000001
0.06524890000000001
```

Slide 10: Analyzing the Results

The results show that using the join() method is significantly faster than concatenating strings using the += operator or a list comprehension. This demonstrates how timeit can help identify the most efficient implementation for a given task.

```python
import timeit
import matplotlib.pyplot as plt

setup_code = '''
words = ["Python", "is", "awesome", "for", "programming"]
'''

methods = ['join', 'list comprehension', 'for loop']
times = [
    timeit.timeit("''.join(words)", setup=setup_code, number=100000),
    timeit.timeit("result = ''; [result += word for word in words]", setup=setup_code, number=100000),
    timeit.timeit("result = ''; for word in words: result += word", setup=setup_code, number=100000)
]

plt.bar(methods, times)
plt.title('String Concatenation Performance')
plt.xlabel('Method')
plt.ylabel('Execution Time (seconds)')
plt.show()
```

Slide 11: Real-life Example: List Comprehension vs. For Loop

Let's compare the performance of list comprehension and traditional for loops for creating a list of squares.

```python
import timeit

setup_code = '''
numbers = range(1000)
'''

list_comp_time = timeit.timeit("[x**2 for x in numbers]", setup=setup_code, number=10000)
for_loop_time = timeit.timeit("result = []\nfor x in numbers:\n    result.append(x**2)", setup=setup_code, number=10000)

print(f"List comprehension: {list_comp_time:.6f} seconds")
print(f"For loop: {for_loop_time:.6f} seconds")
```

Slide 12: Results for: Real-life Example: List Comprehension vs. For Loop

```
List comprehension: 0.413222 seconds
For loop: 0.608942 seconds
```

Slide 13: Interpreting the Results

The results demonstrate that list comprehension is faster than the traditional for loop for this task. This showcases how timeit can help developers make informed decisions about coding style and performance optimization.

```python
import timeit
import matplotlib.pyplot as plt

setup_code = '''
numbers = range(1000)
'''

methods = ['List comprehension', 'For loop']
times = [
    timeit.timeit("[x**2 for x in numbers]", setup=setup_code, number=10000),
    timeit.timeit("result = []\nfor x in numbers:\n    result.append(x**2)", setup=setup_code, number=10000)
]

plt.bar(methods, times)
plt.title('List Creation Performance')
plt.xlabel('Method')
plt.ylabel('Execution Time (seconds)')
plt.show()
```

Slide 14: Best Practices and Tips

When using timeit, consider the following best practices:

1.  Use realistic data sizes and types in your benchmarks.
2.  Run multiple tests and average the results for more accurate measurements.
3.  Be aware of external factors that may affect timing, such as system load or background processes.
4.  Use the repeat() method to get a range of execution times and identify outliers.

```python
import timeit

# Example of using repeat() for multiple runs
setup_code = 'data = [i for i in range(1000)]'
stmt = 'sorted(data)'

results = timeit.repeat(stmt=stmt, setup=setup_code, repeat=5, number=1000)
print(f"Min: {min(results):.6f}, Max: {max(results):.6f}, Avg: {sum(results)/len(results):.6f}")
```

Slide 15: Additional Resources

For more information on timeit and performance optimization in Python, consider exploring the following resources:

1.  Python official documentation on timeit: [https://docs.python.org/3/library/timeit.html](https://docs.python.org/3/library/timeit.html)
2.  "Optimizing Python Code" by Jake VanderPlas (ArXiv:1509.03781): [https://arxiv.org/abs/1509.03781](https://arxiv.org/abs/1509.03781)
3.  Python Performance Analysis tools: [https://wiki.python.org/moin/PythonSpeed/PerformanceTips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)

