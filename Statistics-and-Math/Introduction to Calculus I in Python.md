## Introduction to Calculus I in Python

Slide 1: 
Introduction to Calculus I 
This slide will provide an overview of Calculus I and its applications in Python.

Slide 2: 
Limits 
Understanding the concept of limits is essential in Calculus. This slide will cover the basic definition and examples of limits in Python.

Source Code:

```python
import math

def limit_function(x, val):
    return (math.exp(x) - 1) / x

# Evaluating the limit as x approaches 0
print(limit_function(0.001, 0))  # Output: 1.0
```

Slide 3: 
Continuity 
Continuity is a fundamental concept in Calculus. This slide will explain what continuity means and how to check for continuity in Python.

Source Code:

```python
import math

def continuous_function(x):
    if x == 0:
        return 1
    else:
        return (math.sin(x) / x)

print(continuous_function(0))  # Output: 1.0
print(continuous_function(0.1))  # Output: 0.9983341664682815
```

Slide 4: 
Derivatives 
Derivatives are the cornerstone of Calculus. This slide will introduce the concept of derivatives and how to compute them in Python.

Source Code:

```python
import sympy as sp

x = sp.Symbol('x')
f = x**2 + 2*x + 1
print(f"Original Function: {f}")  # Output: Original Function: x**2 + 2*x + 1

derivative = sp.diff(f, x)
print(f"Derivative: {derivative}")  # Output: Derivative: 2*x + 2
```

Slide 5: 
Rules of Differentiation 
This slide will cover the various rules for differentiating functions, such as the power rule, product rule, and chain rule, with Python examples.

Source Code:

```python
import sympy as sp

x = sp.Symbol('x')

# Power Rule
f1 = x**3
print(f"Function: {f1}, Derivative: {sp.diff(f1, x)}")  # Output: Function: x**3, Derivative: 3*x**2

# Product Rule
f2 = (x**2) * (x**3)
print(f"Function: {f2}, Derivative: {sp.diff(f2, x)}")  # Output: Function: x**5, Derivative: 5*x**4

# Chain Rule
f3 = sp.sin(x**2)
print(f"Function: {f3}, Derivative: {sp.diff(f3, x)}")  # Output: Function: sin(x**2), Derivative: 2*x*cos(x**2)
```

Slide 6: 
Higher-Order Derivatives 
This slide will explain how to compute higher-order derivatives (second, third, etc.) in Python.

Source Code:

```python
import sympy as sp

x = sp.Symbol('x')
f = x**4 + 2*x**3 - 3*x**2 + 4*x - 1

print(f"Original Function: {f}")
print(f"First Derivative: {sp.diff(f, x)}")
print(f"Second Derivative: {sp.diff(f, x, 2)}")
print(f"Third Derivative: {sp.diff(f, x, 3)}")
```

Slide 7: 
Applications of Derivatives 
Derivatives have numerous applications in various fields. This slide will showcase some practical applications of derivatives in Python.

Source Code:

```python
import sympy as sp

x = sp.Symbol('x')

# Optimization
f = x**2 - 4*x + 3
critical_points = sp.solve(sp.diff(f, x), x)
print(f"Critical Points: {critical_points}")  # Output: Critical Points: [2, 1]

# Related Rates
r = sp.Symbol('r')
V = (4/3) * sp.pi * r**3
dV_dr = sp.diff(V, r)
print(f"Rate of change of volume with respect to radius: {dV_dr}")  # Output: Rate of change of volume with respect to radius: 4*pi*r**2
```

Slide 8: 
Integrals 
Integrals are the counterpart of derivatives in Calculus. This slide will introduce the concept of integrals and their evaluation in Python.

Source Code:

```python
import sympy as sp

x = sp.Symbol('x')
f = x**2 + 2*x + 1
integral = sp.integrate(f, x)
print(f"Original Function: {f}")
print(f"Indefinite Integral: {integral}")  # Output: Indefinite Integral: x**3/3 + x**2 + x
```

Slide 9: 
Techniques of Integration 
This slide will cover various techniques for evaluating integrals, such as substitution, integration by parts, and partial fractions.

Source Code:

```python
import sympy as sp

x = sp.Symbol('x')

# Substitution
f1 = sp.cos(x**2)
u = x**2
print(f"Original Function: {f1}, Indefinite Integral: {sp.integrate(f1, x)}")  # Output: Original Function: cos(x**2), Indefinite Integral: sin(x**2)/2

# Integration by Parts
f2 = x * sp.exp(x)
print(f"Original Function: {f2}, Indefinite Integral: {sp.integrate(f2, x)}")  # Output: Original Function: x*exp(x), Indefinite Integral: x*exp(x) - exp(x)

# Partial Fractions
f3 = (x**2 + 2*x + 1) / (x**2 + x)
print(f"Original Function: {f3}, Indefinite Integral: {sp.integrate(f3, x)}")  # Output: Original Function: (x**2 + 2*x + 1)/(x**2 + x), Indefinite Integral: x + 2*log(x) + log(x + 1)
```

Slide 10: 
Definite Integrals 
This slide will explain the concept of definite integrals and their applications in Python.

Source Code:

```python
import sympy as sp

x = sp.Symbol('x')
f = x**2 + 2*x + 1
definite_integral = sp.integrate(f, (x, 0, 2))
print(f"Original Function: {f}")
print(f"Definite Integral from 0 to 2: {definite_integral}")  # Output: Definite Integral from 0 to 2: 11
```

Slide 11: 
Applications of Integrals 
Integrals have numerous applications in various fields. This slide will showcase some practical applications of integrals in Python.

Source Code:

```python
import sympy as sp

x = sp.Symbol('x')

# Area Under a Curve
f = x**2
area = sp.integrate(f, (x, 0, 2))
print(f"Area under the curve y = x**2 from 0 to 2: {area}")  # Output: Area under the curve y = x**2 from 0 to 2: 8/3

# Volume of a Solid of Revolution
f = sp.sqrt(1 - x**2)
volume = sp.integrate(sp.pi * f**2, (x, -1, 1))
print(f"Volume of a sphere of radius 1: {volume}")  # Output: Volume of a sphere of radius 1: 4*pi/3
```

Slide 12: Fundamental Theorem of Calculus The Fundamental Theorem of Calculus is a crucial result that connects derivatives and integrals. This slide will explain the theorem and its significance, along with Python examples.

Source Code:

```python
import sympy as sp

x = sp.Symbol('x')

# Function and its derivative
f = x**3
f_prime = sp.diff(f, x)
print(f"Original Function: {f}, Derivative: {f_prime}")  # Output: Original Function: x**3, Derivative: 3*x**2

# Indefinite Integral of the derivative
indefinite_integral = sp.integrate(f_prime, x)
print(f"Indefinite Integral of the Derivative: {indefinite_integral}")  # Output: Indefinite Integral of the Derivative: x**3

# Fundamental Theorem of Calculus, Part 1
a = 0
b = 2
definite_integral = sp.integrate(f_prime, (x, a, b))
print(f"Definite Integral of the Derivative from {a} to {b}: {definite_integral}")  # Output: Definite Integral of the Derivative from 0 to 2: 8

# Fundamental Theorem of Calculus, Part 2
F = indefinite_integral
print(f"Antiderivative (Indefinite Integral) of f(x): {F}")  # Output: Antiderivative (Indefinite Integral) of f(x): x**3
print(f"F({b}) - F({a}) = {F.subs(x, b) - F.subs(x, a)}")  # Output: F(2) - F(0) = 8
```

The Fundamental Theorem of Calculus establishes the relationship between differentiation and integration. Part 1 of the theorem states that if a function `f(x)` is continuous on a closed interval `[a, b]`, then the definite integral of `f(x)` over that interval is equal to the difference between the values of any antiderivative (indefinite integral) of `f(x)` evaluated at the endpoints of the interval.

Part 2 of the theorem states that if `F(x)` is an antiderivative of `f(x)`, then the derivative of `F(x)` is `f(x)`.

The source code demonstrates both parts of the Fundamental Theorem of Calculus using SymPy. It first defines a function `f(x) = x^3` and computes its derivative `f_prime(x) = 3x^2`. It then calculates the indefinite integral of `f_prime(x)`, which turns out to be `x^3`. This illustrates Part 2 of the theorem.

Next, it computes the definite integral of `f_prime(x)` over the interval `[0, 2]`, which evaluates to `8`. It also shows that the difference between the values of the antiderivative `F(x) = x^3` evaluated at `x = 2` and `x = 0` is also `8`, demonstrating Part 1 of the theorem.

Slide 13: 
Numerical Integration 
In many cases, analytical integration is not possible or practical. This slide will introduce numerical integration techniques, such as the Trapezoidal Rule and Simpson's Rule, in Python.

Source Code:

```python
import numpy as np

def trapezoidal(func, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = func(x)
    s = y[0] + y[-1]
    for i in range(1, n):
        s += 2 * y[i]
    return h * s / 2

def simpsons(func, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = func(x)
    s = y[0] + y[-1]
    for i in range(1, n, 2):
        s += 4 * y[i]
    for i in range(2, n-1, 2):
        s += 2 * y[i]
    return h * s / 3

# Example function
def f(x):
    return x**2

# Trapezoidal Rule
print(trapezoidal(f, 0, 2, 10))  # Output: 4.3333333333333335

# Simpson's Rule
print(simpsons(f, 0, 2, 10))  # Output: 4.333333333333333
```

Slide 14: 
Improper Integrals 
Improper integrals arise when the interval of integration is infinite or the integrand is unbounded. This slide will discuss how to handle improper integrals in Python.

Source Code:

```python
import sympy as sp

x = sp.Symbol('x')

# Integral over an infinite interval
f1 = 1 / (x**2 + 1)
improper_integral1 = sp.integrate(f1, (x, 1, sp.oo))
print(f"Improper Integral of 1/(x**2 + 1) from 1 to infinity: {improper_integral1}")  # Output: Improper Integral of 1/(x**2 + 1) from 1 to infinity: atan(1)

# Integral with an unbounded integrand
f2 = 1 / sp.sqrt(x)
improper_integral2 = sp.integrate(f2, (x, 0, 1))
print(f"Improper Integral of 1/sqrt(x) from 0 to 1: {improper_integral2}")  # Output: Improper Integral of 1/sqrt(x) from 0 to 1: 2
```

## Meta
Mastering Calculus I with Python: A Beginner's Journey

Embark on an exciting adventure through the realms of Calculus I, where mathematics meets Python programming. In this comprehensive series, we'll explore the fundamental concepts of limits, continuity, derivatives, and integrals, unlocking their power through practical examples and engaging code snippets. Whether you're a student, an aspiring data scientist, or a curious learner, this series will guide you through the essential calculus tools and their applications in Python. Get ready to elevate your problem-solving skills and gain a deeper understanding of the mathematical foundations that underpin various scientific and engineering disciplines. #CalculusIPython #LearnProgramming #MathematicsForProgrammers #BeginnersGuide #AcademicExcellence

Hashtags: #CalculusIPython #LearnProgramming #MathematicsForProgrammers #BeginnersGuide #AcademicExcellence #CalculusExplained #PythonForMath #CalculusInAction #CodeAndCalculus #STEMEducation

