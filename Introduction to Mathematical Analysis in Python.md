## Introduction to Mathematical Analysis in Python

Slide 1: 
Introduction to Mathematical Analysis in Python 
Mathematical analysis deals with the study of limits, continuity, differentiation, and integration. Python provides various libraries and tools to perform mathematical analysis tasks.

Slide 2: 
Limits 
The limit of a function represents the value that the function approaches as the input approaches a particular value. Calculating limits is a fundamental concept in calculus.

```python
import sympy as sp

x = sp.symbols('x')
f = (x**2 - 4) / (x - 2)
limit = sp.limit(f, x, 2)
print(f"The limit of f(x) as x approaches 2 is: {limit}")
```

Slide 3: 
Continuity 
A function is continuous if it is defined at a point, and its limit at that point exists and is equal to the function's value at that point.

```python
import numpy as np

def is_continuous(f, x0):
    x = np.linspace(x0 - 1, x0 + 1, 100)
    y = f(x)
    if np.isnan(y).any():
        return False
    return True

f = lambda x: x**2 / (x - 2)
print(is_continuous(f, 2))  # False
```

Slide 4: 
Differentiation 
Differentiation is the process of finding the rate of change of a function at a given point. It is a fundamental concept in calculus and has numerous applications.

```python
import sympy as sp

x = sp.symbols('x')
f = x**3 + 2*x**2 - 5*x + 3
derivative = sp.diff(f, x)
print(f"The derivative of f(x) is: {derivative}")
```

Slide 5: 
Higher-Order Derivatives 
Functions can be differentiated multiple times, leading to higher-order derivatives, which are useful in various applications.

```python
import sympy as sp

x = sp.symbols('x')
f = x**4 - 3*x**3 + 2*x**2 - 5*x + 7
second_derivative = sp.diff(f, x, 2)
print(f"The second derivative of f(x) is: {second_derivative}")
```

Slide 6: 
Integration 
Integration is the reverse process of differentiation and is used to find the area under a curve, among other applications.

```python
import sympy as sp

x = sp.symbols('x')
f = x**3 - 2*x**2 + 3*x - 5
integral = sp.integrate(f, x)
print(f"The indefinite integral of f(x) is: {integral}")
```

Slide 7: 
Definite Integrals 
Definite integrals are used to calculate the area under a curve between two specific points.

```python
import sympy as sp

x = sp.symbols('x')
f = x**2 + 2*x + 1
integral = sp.integrate(f, (x, 1, 3))
print(f"The definite integral of f(x) from 1 to 3 is: {integral}")
```

Slide 8: 
Sequences and Series 
Sequences and series are important concepts in mathematical analysis, with applications in various fields.

```python
import numpy as np

def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

n = 10
sequence = [fibonacci(i) for i in range(n)]
print(f"The first {n} Fibonacci numbers are: {sequence}")
```

Slide 9: 
Convergence and Divergence 
Sequences and series can either converge (approach a finite value) or diverge (become arbitrarily large or small).

```python
import numpy as np

def harmonic_series(n):
    return sum(1/i for i in range(1, n+1))

n = 10
series = harmonic_series(n)
print(f"The sum of the first {n} terms of the harmonic series is: {series}")
```

Slide 10: 
Taylor Series 
Taylor series are powerful tools for approximating functions using infinite polynomials.

```python
import sympy as sp

x = sp.symbols('x')
f = sp.exp(x)
n = 5
taylor_series = sp.series(f, x0=0, n=n+1)
print(f"The Taylor series approximation of e^x up to order {n} is: {taylor_series}")
```

Slide 11: 
Fourier Series 
Fourier series are used to represent periodic functions as the sum of sine and cosine functions.

```python
import numpy as np
import matplotlib.pyplot as plt

def square_wave(x, L):
    y = np.zeros_like(x)
    for n in range(1, 101, 2):
        y += (4/np.pi) * (1/n) * np.sin(2*np.pi*n*x/L)
    return y

x = np.linspace(-np.pi, np.pi, 1000)
L = 2*np.pi
y = square_wave(x, L)

plt.plot(x, y)
plt.show()
```

Slide 12: 
Numerical Integration 
Numerical integration techniques approximate the value of a definite integral when an analytical solution is not available or too complex.

```python
import numpy as np

def trapezoidal(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    integral = y[0] + y[-1]
    for i in range(1, n):
        integral += 2 * y[i]
    integral *= h / 2
    return integral

def f(x):
    return x**3 - 2*x**2 + 3*x - 5

a = 1
b = 3
n = 100
approximation = trapezoidal(f, a, b, n)
print(f"The approximate value of the integral from {a} to {b} is: {approximation}")
```

Slide 13: 
Differential Equations 
Differential equations relate a function and its derivatives, and they are used to model various physical and mathematical phenomena.

```python
import sympy as sp

t = sp.symbols('t')
y = sp.Function('y')

# Define the differential equation
eq = sp.Eq(y(t).diff(t, 2) + 2*y(t).diff(t) + y(t), 0)

# Solve the differential equation
solution = sp.dsolve(eq, y(t))
print(f"The solution to the differential equation is: {solution}")
```

Slide 14: Resources and Further Learning Mathematical analysis is a vast field with numerous applications. Here are some resources for further learning:

* "Introduction to Real Analysis" by Robert G. Bartle and Donald R. Sherbert
* "Calculus" by James Stewart
* Online courses (e.g., Coursera, edX, MIT OpenCourseWare)
* Python libraries: SymPy, NumPy, SciPy, Matplotlib

These slides cover various topics in mathematical (real) analysis, including limits, continuity, differentiation, integration, sequences and series, Taylor and Fourier series, numerical integration, and differential equations. Each slide provides a brief description of the topic, along with Python code examples to illustrate the concepts. The examples are designed to be actionable and suitable for beginners.

Mastering Mathematical Analysis with Python

Embark on a comprehensive journey through the fundamental concepts of mathematical analysis, including limits, continuity, differentiation, integration, sequences and series, Taylor and Fourier series, numerical methods, and differential equations. Explore these essential topics using Python's powerful libraries like SymPy, NumPy, and SciPy. This series provides a solid foundation for anyone interested in pursuing advanced studies in mathematics, physics, engineering, and related fields. #MathematicalAnalysis #Calculus #RealAnalysis #Python #SymPy #NumPy #SciPy #AcademicSeries

Hashtags: #MathematicalAnalysis #Calculus #RealAnalysis #Python #SymPy #NumPy #SciPy #AcademicSeries #HigherEducation #STEM #MathematicsEducation #ComputationalMathematics

In this title and description, the focus is on presenting the TikTok series as a comprehensive exploration of mathematical (real) analysis concepts using Python. The institutional tone is maintained by highlighting the academic nature of the series, its relevance to advanced studies, and its potential benefits for those interested in STEM fields. The hashtags cover relevant keywords related to mathematical analysis, calculus, real analysis, Python, the libraries used, academic content, higher education, STEM, and mathematics education.

