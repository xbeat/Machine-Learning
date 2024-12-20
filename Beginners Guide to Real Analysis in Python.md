## Beginner's Guide to Real Analysis in Python
Mathematical real analysis topics in Python at a beginner level, with each slide containing a title, brief description, and example

Slide 1: Introduction to Real Analysis in Python This slideshow covers fundamental real analysis concepts and their implementation in Python. Real analysis deals with the theoretical foundations of calculus, such as limits, continuity, differentiation, and integration of functions.

Slide 2: Limits The limit of a function describes the value the function approaches as its input approaches some value. Limits are essential for understanding continuity and differentiability.

```python
import math

def limit(f, x, x_val):
    delta = 0.001
    while True:
        x_low = x_val - delta
        x_high = x_val + delta
        if f(x_low) == f(x_high):
            break
        delta *= 0.5
    return f(x_val)

f = lambda x: (x**2 - 1)/(x - 1) 
x_val = 1
result = limit(f, 0.9999, x_val)
print(f"The limit of f(x) as x approaches {x_val} is: {result}")
```

Slide 3: Continuity A function is continuous if it has no breaks or holes. Continuity is required for many mathematical operations like differentiation.

```python
def is_continuous(f, x, x_val, delta=1e-6):
    x_low = x_val - delta
    x_high = x_val + delta
    limit_exists = True
    try:
        lim = limit(f, x, x_val)
    except:
        limit_exists = False
    return limit_exists and (f(x_low) - lim)**2 < delta and (f(x_high) - lim)**2 < delta

f = lambda x: x**2
x_val = 2
print(is_continuous(f, lambda x: x, x_val))
```

Slide 4: Differentiation The derivative measures the rate of change of a function. It is a fundamental tool in optimization, physics, and other applied math fields.

```python
def derivative(f, x, h=1e-5):
    return (f(x+h) - f(x))/h

f = lambda x: x**3
x = 2
print(f"Derivative of f(x)=x^3 at x={x} is: {derivative(f, x)}")
```

Slide 5: Derivatives of Polynomials The "power rule" provides an easy way to take derivatives of polynomial functions.

```python
from math import factorial

def poly_deriv(coeffs):
    derived_coeffs = []
    for i, c in enumerate(coeffs[1:]):
        derived_coeffs.append(c * (len(coeffs) - i - 1))
    return derived_coeffs

f = [1, 2, 1, 3, 4] # 1 + 2x + x^2 + 3x^3 + 4x^4
print(f"f'(x) coeffs: {poly_deriv(f)}")
```

Slide 6: Integration The inverse operation of differentiation. It computes the total accumulated change of a function over an interval.

```python
import numpy as np

def riemann_sum(f, a, b, n=100):
    x = np.linspace(a, b, n+1)
    dx = (b - a) / n
    area = 0
    for i in range(n):
        area += f(x[i]) * dx
    return area
    
f = lambda x: x**2
a = 0
b = 2
print(f"Integral of f(x)=x^2 from {a} to {b} is: {riemann_sum(f, a, b)}")
```

Slide 7: Numerical Integration Error The Riemann sum approximation has error that depends on n, the number of rectangles used. Higher n gives lower error.

```python
actual_int = 4/3 # Integral of x^2 from 0 to 2
n_values = [10, 100, 1000, 10000]
for n in n_values:
    approx = riemann_sum(lambda x: x**2, 0, 2, n)
    error = abs(actual_int - approx)
    print(f"For n={n}, approx={approx:.6f}, error={error:.6f}")
```

Slide 8: Sequences A sequence is an ordered list of elements that can be finite or infinite. Limits of sequences are important in analysis.

```python
def seq_limit(seq, n):
    if n >= len(seq):
        return None
    s = seq[n]
    for i in range(n+1, len(seq)):
        if abs(seq[i] - s) > 1e-6:
            return None
    return s

seq = [1/n for n in range(1, 21)]
n = 5
limit = seq_limit(seq, n)
print(f"The limit of the sequence 1/n as n->inf is: {limit}")
```

Slide 9: Infinite Series An infinite series sums the terms of an infinite sequence. Its convergence is determined by the limit of the sequence of partial sums.

```python
def sum_series(seq, n):
    total = 0
    for i in range(n):
        total += seq[i]
    return total

def series_conv(seq, abs_tol=1e-6, max_terms=100):
    prev_sum = 0
    for n in range(1, max_terms+1):
        curr_sum = sum_series(seq, n)
        if abs(curr_sum - prev_sum) < abs_tol:
            return curr_sum
        prev_sum = curr_sum
    return None
        
harmonic = [1/n for n in range(1, 51)]
print(f"Sum of harmonic series is: {series_conv(harmonic)}")
```

Slide 10: Power Series A power series is a series representation of a function as an infinite sum of terms calculated from the derivatives at a single point.

```python
from math import factorial

def power_series(f, a, n):
    coeffs = []
    for i in range(n+1):
        coeffs.append(derivative(f, a, i)/factorial(i))
    return coeffs

f = lambda x: 1/(1 - x) 
a = 0
print(f"Power Series of f(x)=1/(1-x) around x=0: {power_series(f, a, 5)}")
```

Slide 11: Taylor Series The Taylor series is a representation of a function as an infinite sum of terms calculated from its derivatives at a single point.

```python
from math import factorial, exp

def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n-1)
        
def exp_taylor(x, n):
    approx = 0
    for i in range(n+1):
        approx += x**i / factorial(i)
    return approx

x = 1
n = 10 
print(f"e^{x} approximated by the Taylor series expansion to {n} terms: {exp_taylor(x, n)}")
```

Slide 12: Taylor Polynomial Error The Taylor polynomial approximates a function by truncating its Taylor series. The approximation error depends on n and the smoothness of the function.

```python
import sympy as sp
import numpy as np

def taylor_error(f, x0, n, x):
    # Compute Taylor polynomial of degree n around x0
    f_x = sp.lambdify((), f.series(x0, n).removeO(), 'numpy')
    
    # Lambdify the exact function
    f_exact = sp.lambdify(f.atoms(sp.Symbol), f, 'numpy')
    
    # Compute maximum error over the interval
    max_err = 0
    for x_val in x:
        err = abs(f_exact(x_val) - f_x(x_val))
        if err > max_err:
            max_err = err
            
    return max_err

# Example: Approximate exp(x) around x0 = 0 on [-1, 1]
x = np.linspace(-1, 1, 101)
f = sp.exp(sp.symbols('x'))
x0 = 0
n = 5

max_err = taylor_error(f, x0, n, x)
print(f"Maximum error in approximating exp(x) by a degree {n} Taylor polynomial around x={x0} on [-1, 1] is: {max_err}")
```

Explanation:

1. We use SymPy to construct the symbolic expression for the function f.
2. We compute the Taylor polynomial of degree n around x0 using f.series(x0, n).
3. We lambdify both the Taylor polynomial and the exact function f for efficient numerical evaluation.
4. We evaluate the maximum absolute error between the Taylor polynomial and f over the given interval x.
5. We print out the maximum error as a measure of the approximation quality.

This example approximates the exponential function exp(x) around x0=0 on the interval \[-1, 1\] using a degree 5 Taylor polynomial, and computes the maximum error in that approximation.


Slide 13: Fourier Series The Fourier series represents a periodic function as an infinite sum of sines and cosines. It allows analyzing functions as superpositions of simple waves.

```python
import numpy as np

def fourier_series(f, L, n):
    coeffs = {}
    for k in range(-n, n+1):
        if k == 0:
            coeffs[k] = 1/L * scipy.integrate.quad(f, 0, L)[0]
        else:
            def int_func(x):
                return f(x) * np.cos(2*np.pi*k*x/L)
            coeffs[k] = 2/L * scipy.integrate.quad(int_func, 0, L)[0]
    return coeffs

def square_wave(x):
    return 1 if x < 0.5 else -1

L = 1
n = 10
coeffs = fourier_series(square_wave, L, n)
print(f"First {2*n+1} Fourier coeffs of square wave: {coeffs}")
```

Slide 14: Convergence of Fourier Series The Fourier series converges pointwise to the periodic function for most well-behaved functions, but may diverge at points of discontinuity.

```python
import matplotlib.pyplot as plt

def plot_fourier_approx(f, L, coeffs, x_vals):
    approx = np.zeros(len(x_vals))
    for k, a_k in coeffs.items():
        if k == 0:
            approx += a_k
        else:
            approx += a_k * np.cos(2*np.pi*k/L * x_vals)
    return approx

x_vals = np.linspace(0, 2*L, 1000)
approx = plot_fourier_approx(square_wave, L, coeffs, x_vals)
plt.plot(x_vals, [square_wave(x) for x in x_vals], label='Actual')
plt.plot(x_vals, approx, '--', label=f'Fourier Approx (n={n})')
plt.legend()
plt.show()
```

This covers many fundamental concepts in mathematical real analysis and their implementation in Python at a beginner level. The slides provide brief theoretical descriptions along with example code implementations to help build an understanding through concrete examples. The topics cover limits, continuity, differentiation, integration, sequences and series, power series, Taylor series, Fourier series, and convergence analysis.