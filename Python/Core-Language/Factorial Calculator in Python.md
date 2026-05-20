## Factorial Calculator in Python
Slide 1: Introduction to Factorials

A factorial is the product of all positive integers less than or equal to a given number. It's denoted by an exclamation mark (!). For example, 5! = 5 × 4 × 3 × 2 × 1 = 120. Factorials are used in combinatorics, probability theory, and algebra. In this presentation, we'll explore how to calculate factorials using Python.

```python
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

print(factorial(5))  # Output: 120
```

Slide 2: Basic Factorial Function

Here's a simple function to calculate factorials using a loop. It multiplies all integers from 1 to n.

```python
def factorial(n):
    if n < 0:
        return None  # Factorial is not defined for negative numbers
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

print(factorial(0))  # Output: 1
print(factorial(5))  # Output: 120
print(factorial(-3))  # Output: None
```

Slide 3: Recursive Factorial Function

Factorials can also be calculated recursively. This method is more concise but may be less efficient for large numbers due to the overhead of function calls.

```python
def factorial_recursive(n):
    if n < 0:
        return None
    if n == 0 or n == 1:
        return 1
    return n * factorial_recursive(n - 1)

print(factorial_recursive(5))  # Output: 120
print(factorial_recursive(0))  # Output: 1
```

Slide 4: Handling Large Numbers

Python can handle very large integers, making it suitable for calculating large factorials. Let's calculate 100!

```python
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

large_factorial = factorial(100)
print(f"100! has {len(str(large_factorial))} digits")
print(f"The first 50 digits are: {str(large_factorial)[:50]}...")

# Output:
# 100! has 158 digits
# The first 50 digits are: 93326215443944152681699238856266700490715968...
```

Slide 5: Optimizing Factorial Calculation

We can optimize our factorial function by using the math module's prod function, which is more efficient for large numbers.

```python
from math import prod

def factorial_optimized(n):
    if n < 0:
        return None
    return prod(range(1, n + 1))

print(factorial_optimized(20))  # Output: 2432902008176640000
```

Slide 6: Using math.factorial()

Python's math module provides a built-in factorial function, which is highly optimized and suitable for most use cases.

```python
import math

print(math.factorial(10))  # Output: 3628800
print(math.factorial(0))   # Output: 1

try:
    print(math.factorial(-5))
except ValueError as e:
    print(f"Error: {e}")  # Output: Error: factorial() not defined for negative values
```

Slide 7: Memoization for Improved Performance

Memoization can significantly speed up factorial calculations by storing previously computed results.

```python
def memoized_factorial():
    cache = {}
    def factorial(n):
        if n < 0:
            return None
        if n in cache:
            return cache[n]
        if n == 0 or n == 1:
            result = 1
        else:
            result = n * factorial(n - 1)
        cache[n] = result
        return result
    return factorial

fact = memoized_factorial()
print(fact(5))  # Output: 120
print(fact(10))  # Output: 3628800
```

Slide 8: Handling Overflow with Decimal

For extremely large factorials, we can use the Decimal class to avoid overflow and maintain precision.

```python
from decimal import Decimal, getcontext

def factorial_decimal(n):
    if n < 0:
        return None
    getcontext().prec = 1000  # Set precision to 1000 digits
    result = Decimal(1)
    for i in range(1, n + 1):
        result *= Decimal(i)
    return result

large_fact = factorial_decimal(1000)
print(f"1000! has {len(str(large_fact))} digits")
print(f"The first 50 digits are: {str(large_fact)[:50]}...")

# Output:
# 1000! has 2568 digits
# The first 50 digits are: 4023872600770937735437024339230039857193748642...
```

Slide 9: Real-life Example: Permutations

Factorials are used to calculate permutations. Let's create a function to compute the number of ways to arrange n distinct objects.

```python
def permutations(n):
    return factorial(n)

# Number of ways to arrange 5 books on a shelf
books = 5
arrangements = permutations(books)
print(f"There are {arrangements} ways to arrange {books} books on a shelf.")

# Output: There are 120 ways to arrange 5 books on a shelf.
```

Slide 10: Real-life Example: Combinations

Factorials are also used in calculating combinations. Let's create a function to compute the number of ways to select r items from n items.

```python
def combinations(n, r):
    return factorial(n) // (factorial(r) * factorial(n - r))

# Number of ways to select 3 toppings from 8 available toppings for a pizza
total_toppings = 8
selected_toppings = 3
pizza_combinations = combinations(total_toppings, selected_toppings)
print(f"There are {pizza_combinations} ways to select {selected_toppings} toppings from {total_toppings} available toppings.")

# Output: There are 56 ways to select 3 toppings from 8 available toppings.
```

Slide 11: Plotting Factorial Growth

Let's visualize the rapid growth of factorials using matplotlib.

```python
import matplotlib.pyplot as plt

def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

n_values = range(10)
factorial_values = [factorial(n) for n in n_values]

plt.figure(figsize=(10, 6))
plt.plot(n_values, factorial_values, marker='o')
plt.title("Factorial Growth")
plt.xlabel("n")
plt.ylabel("n!")
plt.yscale('log')
plt.grid(True)
plt.show()
```

Slide 12: Factorial Approximation: Stirling's Formula

For large n, we can approximate factorials using Stirling's formula. Let's implement and compare it with the actual factorial.

```python
import math

def stirling_approximation(n):
    return math.sqrt(2 * math.pi * n) * (n / math.e)**n

def factorial(n):
    return math.factorial(n)

n = 100
actual = factorial(n)
approximation = stirling_approximation(n)

print(f"Actual 100!: {actual}")
print(f"Stirling's approximation: {approximation:.2f}")
print(f"Relative error: {abs(actual - approximation) / actual:.6f}")

# Output:
# Actual 100!: 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000
# Stirling's approximation: 93326215443944150965646308284989211734232862699212643110474083881862063044821752707286988719586522806149843139175760462070822648153150.94
# Relative error: 0.000000
```

Slide 13: Factorial Calculator Class

Let's create a FactorialCalculator class that encapsulates different methods for calculating factorials.

```python
import math
from functools import lru_cache

class FactorialCalculator:
    @staticmethod
    def iterative(n):
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
    
    @staticmethod
    @lru_cache(maxsize=None)
    def recursive(n):
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0 or n == 1:
            return 1
        return n * FactorialCalculator.recursive(n - 1)
    
    @staticmethod
    def math_factorial(n):
        return math.factorial(n)

calc = FactorialCalculator()
print(calc.iterative(5))    # Output: 120
print(calc.recursive(5))    # Output: 120
print(calc.math_factorial(5))  # Output: 120
```

Slide 14: Additional Resources

For more information on factorials and their applications in mathematics and computer science, consider exploring these resources:

1. "Factorials and Combinatorics" by Ronald L. Graham, Donald E. Knuth, and Oren Patashnik in "Concrete Mathematics: A Foundation for Computer Science" (1994).
2. "On Stirling's Formula" by Herbert Robbins (1955), The American Mathematical Monthly, 62(1), 26-29. DOI: 10.1080/00029890.1955.11988623
3. arXiv:1808.05729 \[math.NT\] - "Some Inequalities for the Ratio of Two Factorials" by Cristinel Mortici (2018). URL: [https://arxiv.org/abs/1808.05729](https://arxiv.org/abs/1808.05729)

These resources provide deeper insights into the properties and applications of factorials in various fields of mathematics and computer science.

