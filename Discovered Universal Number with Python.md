## Discovered Universal Number with Python
Slide 1: The Discovered Universal Number

The concept of a "discovered universal number" is not a well-established mathematical or scientific concept. There is no single number that has been universally recognized as having special properties across all mathematical and scientific domains. Instead, there are several numbers that have significant roles in various fields of mathematics and science. In this presentation, we'll explore some of these important numbers and their applications.

```python
import math

# Some important numbers in mathematics and science
pi = math.pi
e = math.e
phi = (1 + math.sqrt(5)) / 2

print(f"Pi: {pi}")
print(f"Euler's number: {e}")
print(f"Golden ratio: {phi}")
```

Slide 2: Pi (π): The Circular Constant

Pi is the ratio of a circle's circumference to its diameter. It appears in many mathematical formulas beyond geometry, including statistics, physics, and engineering.

```python
import math

def estimate_pi(n):
    inside_circle = 0
    total_points = n
    
    for _ in range(total_points):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside_circle += 1
    
    return 4 * inside_circle / total_points

estimated_pi = estimate_pi(1000000)
print(f"Estimated Pi: {estimated_pi}")
print(f"Math.pi: {math.pi}")
print(f"Difference: {abs(estimated_pi - math.pi)}")
```

Slide 3: Real-life Application of Pi

Pi is used in calculating the area of a circular field for irrigation systems in agriculture. Let's calculate the area of a circular field with a radius of 100 meters.

```python
import math

radius = 100  # meters
area = math.pi * radius ** 2

print(f"Area of a circular field with radius {radius}m: {area:.2f} square meters")

# Calculate water needed for 1cm of irrigation
water_depth = 0.01  # meters
water_volume = area * water_depth

print(f"Water needed for 1cm irrigation: {water_volume:.2f} cubic meters")
```

Slide 4: Euler's Number (e): The Natural Exponential Base

Euler's number is the base of natural logarithms and appears in calculations involving exponential growth and decay. It's fundamental in calculus and complex analysis.

```python
import math

def estimate_e(n):
    return (1 + 1/n) ** n

approximations = [estimate_e(n) for n in [10, 100, 1000, 10000, 100000]]

for i, approx in enumerate(approximations):
    n = 10 ** (i+1)
    print(f"e ≈ {approx} (n = {n})")

print(f"math.e = {math.e}")
```

Slide 5: Real-life Application of Euler's Number

Euler's number is used in modeling population growth. Let's model the growth of a bacteria culture over time.

```python
import math

initial_population = 1000
growth_rate = 0.5  # 50% growth per hour
time = 5  # hours

final_population = initial_population * math.exp(growth_rate * time)

print(f"Initial population: {initial_population}")
print(f"After {time} hours: {final_population:.0f}")

# Calculate doubling time
doubling_time = math.log(2) / growth_rate
print(f"Population doubling time: {doubling_time:.2f} hours")
```

Slide 6: The Golden Ratio (φ): Nature's Proportion

The golden ratio, approximately 1.618, is found in art, architecture, and nature. It's considered aesthetically pleasing and appears in the proportions of many natural objects.

```python
import math

phi = (1 + math.sqrt(5)) / 2

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

ratios = [fibonacci(n+1) / fibonacci(n) for n in range(10, 20)]

print(f"Golden Ratio: {phi}")
print("Fibonacci Ratios:")
for i, ratio in enumerate(ratios, start=10):
    print(f"F({i+1})/F({i}) = {ratio}")
```

Slide 7: Imaginary Unit (i): The Square Root of -1

The imaginary unit i is defined as the square root of -1. It's crucial in complex analysis and has applications in electrical engineering and quantum mechanics.

```python
import cmath

def complex_roots(a, b, c):
    discriminant = b**2 - 4*a*c
    root1 = (-b + cmath.sqrt(discriminant)) / (2*a)
    root2 = (-b - cmath.sqrt(discriminant)) / (2*a)
    return root1, root2

a, b, c = 1, 2, 5
roots = complex_roots(a, b, c)

print(f"Roots of {a}x^2 + {b}x + {c} = 0:")
for i, root in enumerate(roots, start=1):
    print(f"x{i} = {root}")
```

Slide 8: Real-life Application of Complex Numbers

Complex numbers are used in signal processing. Let's generate a simple signal and perform a Fourier transform to analyze its frequency components.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a signal with two frequency components
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)

# Perform Fourier transform
fft = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(t), t[1] - t[0])

plt.figure(figsize=(12, 4))
plt.plot(freqs, np.abs(fft))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of the Signal')
plt.show()
```

Slide 9: Euler's Identity: eiπ + 1 = 0

Euler's identity is often regarded as one of the most beautiful equations in mathematics, connecting five fundamental mathematical constants.

```python
import cmath

# Verify Euler's identity
result = cmath.exp(1j * cmath.pi) + 1

print(f"e^(iπ) + 1 = {result}")
print(f"Magnitude of the result: {abs(result)}")

# Visualize Euler's formula
theta = np.linspace(0, 2*np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)

plt.figure(figsize=(8, 8))
plt.plot(x, y)
plt.plot([0, 1], [0, 0], 'r', linewidth=2)
plt.plot([0], [0], 'ko')
plt.axis('equal')
plt.title("Euler's Formula: e^(iθ) = cos(θ) + i*sin(θ)")
plt.show()
```

Slide 10: The Number Zero: More Than Nothing

Zero, while seemingly simple, has profound implications in mathematics. It allows for the concept of negative numbers and is crucial in place value systems.

```python
def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    return 1 if n == 0 else n * factorial(n-1)

print("Factorial of 0:", factorial(0))
print("5 + 0:", 5 + 0)
print("5 * 0:", 5 * 0)
print("5 / 1:", 5 / 1)

try:
    result = 5 / 0
except ZeroDivisionError as e:
    print("5 / 0:", str(e))
```

Slide 11: Infinity: The Concept of Endlessness

Infinity is not a number but a concept representing endlessness. It's crucial in calculus and set theory.

```python
import math

print("Is infinity larger than any number?")
print(math.inf > 10**100)

print("\nWhat happens when we add to infinity?")
print(math.inf + 1 == math.inf)

print("\nWhat about infinity divided by infinity?")
print(math.inf / math.inf)

# Visualize the limit of 1/x as x approaches infinity
x = np.linspace(1, 100, 1000)
y = 1 / x

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("Limit of 1/x as x approaches infinity")
plt.xlabel("x")
plt.ylabel("1/x")
plt.ylim(0, 0.2)
plt.show()
```

Slide 12: Prime Numbers: The Building Blocks of Integers

Prime numbers are integers greater than 1 that are only divisible by 1 and themselves. They play a crucial role in number theory and cryptography.

```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_primes(limit):
    return [n for n in range(2, limit) if is_prime(n)]

primes = find_primes(100)
print("Prime numbers up to 100:", primes)

# Visualize prime number distribution
plt.figure(figsize=(12, 6))
plt.plot(primes, range(len(primes)), 'bo-')
plt.title("Distribution of Prime Numbers")
plt.xlabel("Prime Number")
plt.ylabel("Count")
plt.grid(True)
plt.show()
```

Slide 13: The Square Root of 2: Irrationality in Nature

The square root of 2 is the first number proven to be irrational. It appears in geometry as the length of a square's diagonal with side length 1.

```python
import math

def approximate_sqrt2(iterations):
    approximation = 1
    for _ in range(iterations):
        approximation = (approximation + 2/approximation) / 2
    return approximation

iterations = [1, 5, 10, 20]
for i in iterations:
    approx = approximate_sqrt2(i)
    print(f"After {i} iterations: {approx}")

print(f"math.sqrt(2): {math.sqrt(2)}")

# Visualize convergence
x = list(range(1, 21))
y = [approximate_sqrt2(i) for i in x]

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo-')
plt.axhline(y=math.sqrt(2), color='r', linestyle='--')
plt.title("Convergence of Square Root of 2 Approximation")
plt.xlabel("Iterations")
plt.ylabel("Approximation")
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into the world of important numbers in mathematics and their applications, here are some valuable resources:

1. "The Constants of Nature" by John D. Barrow (2002)
2. "An Introduction to the Theory of Numbers" by G.H. Hardy and E.M. Wright
3. "Prime Obsession: Bernhard Riemann and the Greatest Unsolved Problem in Mathematics" by John Derbyshire

For more technical and research-oriented material, you can explore these papers on arXiv:

1. "On the Irrationality Measure of Pi" by S. Saidak (2006) arXiv:math/0601543
2. "Euler's Constant: Euler's Work and Modern Developments" by J.C. Lagarias (2013) arXiv:1303.1856
3. "Prime Numbers: A Computational Perspective" by R. Crandall and C. Pomerance (2005) This book is not on arXiv, but it's a comprehensive resource on computational number theory.

