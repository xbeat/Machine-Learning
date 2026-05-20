## Simpsons Rule in Python

Slide 1: 

Introduction to Simpson's Rule

Simpson's Rule is a numerical integration technique used to approximate the definite integral of a function over a given interval. It is based on the parabolic rule, which approximates the function by fitting a parabola through three points on the curve. Simpson's Rule provides a more accurate approximation than the Trapezoidal Rule, making it a valuable tool for computing integrals numerically.

Code:

```python
import numpy as np

def simpson(f, a, b, n):
    """
    Approximate the integral of f(x) from a to b using Simpson's Rule
    with n subintervals.
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    integral = y[0] + y[-1]
    
    for i in range(1, n, 2):
        integral += 4 * y[i]
    for i in range(2, n - 1, 2):
        integral += 2 * y[i]
    
    integral *= h / 3
    return integral
```

Caption: This code defines a function `simpson` that takes a function `f`, the integration limits `a` and `b`, and the number of subintervals `n`. It calculates the approximation of the integral using Simpson's Rule.

Slide 2: 

Understanding Simpson's Rule

Simpson's Rule is a Newton-Cotes formula that approximates the integral of a function by fitting a parabola through three consecutive points on the curve. The area under the parabola is then computed and used as an approximation for the integral over that interval. Simpson's Rule is more accurate than the Trapezoidal Rule because it uses a higher-degree polynomial approximation.

Code:

```python
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return x**2

a = 0
b = 2
n = 10

x = np.linspace(a, b, n + 1)
y = f(x)

plt.plot(x, y)
plt.fill_between(x, 0, y, alpha=0.3)
plt.title("Simpson's Rule Approximation")
plt.show()
```

Caption: This code demonstrates the concept of Simpson's Rule by plotting the function `f(x) = x^2` over the interval `[0, 2]` with 10 subintervals. The shaded area under the curve represents the approximation of the integral using Simpson's Rule.

Slide 3: 

Derivation of Simpson's Rule

Simpson's Rule can be derived by integrating the Lagrange interpolating polynomial that passes through three consecutive points on the curve. The coefficients of this polynomial are determined using the values of the function and its derivatives at these points. The resulting formula for Simpson's Rule is an approximation of the integral over the interval spanned by the three points.

Code:

```python
import sympy as sp

x = sp.symbols('x')
f = x**2
a = 0
b = 2
n = 2

h = (b - a) / n
x0, x1, x2 = a, a + h, b

y0 = f.subs(x, x0)
y1 = f.subs(x, x1)
y2 = f.subs(x, x2)

integral = h/3 * (y0 + 4*y1 + y2)
print(f"Approximate integral: {integral}")
```

Caption: This code demonstrates the derivation of Simpson's Rule using SymPy, a Python library for symbolic mathematics. It calculates the approximate integral of `f(x) = x^2` over the interval `[0, 2]` with two subintervals using the Simpson's Rule formula.

Slide 4: 

Composite Simpson's Rule

For more accurate approximations, Simpson's Rule can be applied repeatedly over smaller subintervals. This process is known as Composite Simpson's Rule. By dividing the interval into an even number of subintervals and applying Simpson's Rule on each subinterval, the overall approximation becomes more accurate, especially for functions with higher-order derivatives.

Code:

```python
import numpy as np

def composite_simpson(f, a, b, n):
    """
    Approximate the integral of f(x) from a to b using Composite Simpson's Rule
    with n subintervals.
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    integral = y[0] + y[-1]
    
    for i in range(1, n, 2):
        integral += 4 * y[i]
    for i in range(2, n - 1, 2):
        integral += 2 * y[i]
    
    integral *= h / 3
    return integral
```

Caption: This code implements the Composite Simpson's Rule by dividing the interval into `n` subintervals and applying Simpson's Rule on each subinterval. The final approximation is the sum of the approximations for each subinterval.

Slide 5: 

Error Analysis

Simpson's Rule provides a more accurate approximation than the Trapezoidal Rule, but it still introduces an error due to the polynomial approximation. The error in Simpson's Rule is proportional to the fourth derivative of the function being integrated, multiplied by a constant that depends on the interval and the number of subintervals. Understanding the error behavior is crucial for determining the appropriate number of subintervals for a desired accuracy.

Code:

```python
import sympy as sp

x = sp.symbols('x')
f = x**4
a = 0
b = 1
n = 4

h = (b - a) / n
x0, x1, x2, x3, x4 = a, a + h, a + 2*h, a + 3*h, b

y0 = f.subs(x, x0)
y1 = f.subs(x, x1)
y2 = f.subs(x, x2)
y3 = f.subs(x, x3)
y4 = f.subs(x, x4)

integral_exact = f.integrate()(b) - f.integrate()(a)
integral_approx = h/3 * (y0 + 4*y1 + 2*y2 + 4*y3 + y4)
error = abs(integral_exact - integral_approx)

print(f"Exact integral: {integral_exact}")
print(f"Approximate integral: {integral_approx}")
print(f"Error: {error}")
```

Caption: This code compares the exact and approximate integrals of `f(x) = x^4` over the interval `[0, 1]` with four subintervals using Simpson's Rule. It calculates the error between the exact and approximate values, demonstrating the error behavior of Simpson's Rule.

Slide 6: 

Improving Accuracy

To improve the accuracy of Simpson's Rule, one can increase the number of subintervals or use adaptive techniques that automatically adjust the number of subintervals based on the behavior of the function. Another approach is to use higher-order Newton-Cotes formulas, which involve more points and higher-degree polynomial approximations, providing better accuracy at the cost of increased computation.

Code:

```python
import numpy as np

def adaptive_simpson(f, a, b, tol=1e-6, max_iter=100):
    """
    Approximate the integral of f(x) from a to b using Adaptive Simpson's Rule
    with a specified tolerance and maximum number of iterations.
    """
    n = 2
    integral_old = simpson(f, a, b, n)
    n *= 2
    integral_new = simpson(f, a, b, n)
    
    iter = 1
    while abs(integral_new - integral_old) > tol and iter < max_iter:
        iter += 1
        integral_old = integral_new
        n *= 2
        integral_new = simpson(f, a, b, n)
    
    return integral_new
```

Caption: This code implements an adaptive version of Simpson's Rule, where the number of subintervals is automatically adjusted based on a specified tolerance. It repeatedly doubles the number of subintervals until the difference between successive approximations falls below the tolerance or the maximum number of iterations is reached.

Slide 7: 

Applications of Simpson's Rule

Simpson's Rule finds numerous applications in various fields where numerical integration is required. It is widely used in physics, engineering, mathematics, and computer science for approximating definite integrals that cannot be evaluated analytically. Some specific applications include calculating areas and volumes, solving differential equations, computing statistical quantities, and evaluating integrals in numerical methods such as finite element analysis.

Code:

```python
import numpy as np

def volume_of_revolution(f, a, b, n):
    """
    Compute the volume of a solid of revolution formed by
    rotating the area under f(x) from a to b around the x-axis
    using Simpson's Rule with n subintervals.
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    volume = 0
    
    for i in range(n):
        x0, x1, x2 = x[i], x[i+1], x[i+2]
        y0, y1, y2 = y[i], y[i+1], y[i+2]
        volume += h/3 * (y0**2 + 4*y1**2 + y2**2)
    
    volume *= 2 * np.pi
    return volume
```

Caption: This code demonstrates an application of Simpson's Rule in computing the volume of a solid of revolution formed by rotating the area under a curve `f(x)` from `a` to `b` around the x-axis. It uses Simpson's Rule with `n` subintervals to approximate the integral and calculates the volume by summing the volumes of the individual disks.

Slide 8: 

Limitations and Alternatives

While Simpson's Rule is a powerful numerical integration technique, it has some limitations. It assumes that the function being integrated is well-behaved and has continuous derivatives up to the fourth order. For functions with discontinuities or rapidly varying behavior, Simpson's Rule may not provide accurate results. In such cases, alternative methods like Gaussian quadrature or Monte Carlo integration may be more suitable.

Code:

```python
import scipy.integrate as integrate

def f(x):
    return 1 / (1 + x**2)

a = 0
b = 1
n = 10

integral_simpson = simpson(f, a, b, n)
integral_quad = integrate.quad(f, a, b)[0]

print(f"Simpson's Rule: {integral_simpson}")
print(f"Gaussian Quadrature: {integral_quad}")
```

Caption: This code compares the results of Simpson's Rule and Gaussian Quadrature for approximating the integral of `f(x) = 1 / (1 + x^2)` over the interval `[0, 1]` with 10 subintervals. Gaussian Quadrature, which uses optimally chosen points and weights, may provide better accuracy for certain functions.

Slide 9: 

Extensions and Variations

Simpson's Rule has several extensions and variations that address different integration scenarios. For example, the Three-Eighths Rule is a variation that provides higher accuracy by using more points and a higher-degree polynomial approximation. Another extension is the Adaptive Simpson's Rule, which automatically adjusts the number of subintervals based on the desired accuracy, as shown in an earlier slide.

Code:

```python
import numpy as np

def three_eighths_rule(f, a, b, n):
    """
    Approximate the integral of f(x) from a to b using the Three-Eighths Rule
    with n subintervals.
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    integral = y[0] + y[-1]
    
    for i in range(1, n, 3):
        integral += 3 * (y[i] + y[i+2])
    for i in range(2, n - 1, 3):
        integral += 9 * y[i]
    
    integral *= 3 * h / 8
    return integral
```

Caption: This code implements the Three-Eighths Rule, a variation of Newton-Cotes formulas that provides higher accuracy than Simpson's Rule by using more points and a higher-degree polynomial approximation. It approximates the integral by fitting a fourth-degree polynomial through five points.

Slide 10: 

Comparison with Other Methods

While Simpson's Rule is widely used and provides a good balance between accuracy and computational complexity, it is essential to compare it with other numerical integration methods to understand its strengths and weaknesses. Some alternative methods include the Trapezoidal Rule, Gaussian Quadrature, Monte Carlo integration, and advanced techniques like adaptive quadrature or Clenshaw-Curtis quadrature.

Code:

```python
import scipy.integrate as integrate

def f(x):
    return np.exp(-x**2)

a = 0
b = 2
n = 10

integral_simpson = simpson(f, a, b, n)
integral_quad = integrate.quad(f, a, b)[0]
integral_monte_carlo = integrate.nquad(f, [[a, b]])[0]

print(f"Simpson's Rule: {integral_simpson}")
print(f"Gaussian Quadrature: {integral_quad}")
print(f"Monte Carlo Integration: {integral_monte_carlo}")
```

Caption: This code compares the results of Simpson's Rule, Gaussian Quadrature, and Monte Carlo integration for approximating the integral of `f(x) = exp(-x^2)` over the interval `[0, 2]` with 10 subintervals (for Simpson's Rule). It demonstrates the different approaches and their respective strengths and weaknesses.

Slide 11: 

Symbolic Integration with SymPy

While Simpson's Rule is primarily used for numerical integration, it can also be applied symbolically using computer algebra systems like SymPy. Symbolic integration allows for exact calculations and can provide insights into the behavior of the integration method, as well as facilitate error analysis and comparisons with other methods.

Code:

```python
import sympy as sp

x = sp.symbols('x')
f = x**3
a = 0
b = 1
n = 2

h = (b - a) / n
x0, x1, x2 = a, a + h, b

y0 = f.subs(x, x0)
y1 = f.subs(x, x1)
y2 = f.subs(x, x2)

integral_exact = f.integrate()(b) - f.integrate()(a)
integral_approx = h/3 * (y0 + 4*y1 + y2)
error = abs(integral_exact - integral_approx)

print(f"Exact integral: {integral_exact}")
print(f"Approximate integral: {integral_approx}")
print(f"Error: {error}")
```

Caption: This code demonstrates the use of SymPy for symbolic integration and error analysis of Simpson's Rule. It calculates the exact and approximate integrals of `f(x) = x^3` over the interval `[0, 1]` with two subintervals, and computes the error between them.

Slide 12: 

Challenges and Future Directions

While Simpson's Rule is a well-established numerical integration technique, there are still challenges and areas for improvement. These include handling functions with singularities or discontinuities, improving accuracy for highly oscillatory functions, and developing efficient algorithms for high-dimensional integration problems. Additionally, parallel computing and GPU acceleration can be explored to speed up computations for large-scale integration tasks.

Code:

```python
import numpy as np
import multiprocessing as mp

def parallel_simpson(f, a, b, n, num_cores):
    """
    Approximate the integral of f(x) from a to b using Parallel Simpson's Rule
    with n subintervals and num_cores parallel processes.
    """
    h = (b - a) / n
    pool = mp.Pool(num_cores)
    results = []
    
    for i in range(num_cores):
        start = a + i * (b - a) / num_cores
        end = a + (i + 1) * (b - a) / num_cores
        results.append(pool.apply_async(simpson, args=(f, start, end, n // num_cores)))
    
    pool.close()
    pool.join()
    
    integral = sum(result.get() for result in results)
    return integral
```

Caption: This code demonstrates a parallelized version of Simpson's Rule using the `multiprocessing` module in Python. It divides the integration interval into subintervals and distributes the computations across multiple processes, taking advantage of multiple cores or processors. This approach can significantly improve the performance for large-scale integration tasks.

Slide 13: 

Advanced Techniques and Adaptivity

To address the limitations of Simpson's Rule and improve its accuracy and efficiency, various advanced techniques have been developed. One approach is to use adaptive integration methods that automatically adjust the number and placement of subintervals based on the behavior of the function being integrated. This ensures higher accuracy in regions where the function exhibits rapid changes or singularities.

Code:

```python
import scipy.integrate as integrate

def f(x):
    return np.sin(x**2)

a = 0
b = np.pi
integral_fixed, error_estimate = integrate.quad(f, a, b)
integral_adaptive = integrate.nquad(f, [[a, b]])[0]

print(f"Fixed Gaussian Quadrature: {integral_fixed}")
print(f"Adaptive Quadrature: {integral_adaptive}")
```

Caption: This code demonstrates the use of SciPy's adaptive quadrature routines for integrating the function `f(x) = sin(x^2)` over the interval `[0, pi]`. It compares the results of fixed Gaussian quadrature with adaptive quadrature, which adjusts the subintervals based on the function's behavior, leading to improved accuracy.

Slide 14: 

Case Study: Integrating a Complex Function

To illustrate the practical application of Simpson's Rule and its variants, let's consider a case study involving the integration of a complex function that arises in a scientific or engineering context. This case study will showcase the implementation details, error analysis, and comparative performance of different integration techniques.

Code:

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(-x**2) * np.cos(x**3)

a = -2
b = 2
n = 100

x = np.linspace(a, b, 1000)
y = f(x)

plt.plot(x, y)
plt.title("Function to be Integrated")
plt.show()

integral_simpson = simpson(f, a, b, n)
integral_three_eighths = three_eighths_rule(f, a, b, n)
integral_quad = integrate.quad(f, a, b)[0]

print(f"Simpson's Rule: {integral_simpson}")
print(f"Three-Eighths Rule: {integral_three_eighths}")
print(f"Gaussian Quadrature: {integral_quad}")
```

Caption: This code presents a case study involving the integration of the complex function `f(x) = exp(-x^2) * cos(x^3)` over the interval `[-2, 2]`. It compares the results obtained using Simpson's Rule, the Three-Eighths Rule, and Gaussian Quadrature. The function is first plotted to visualize its behavior, and then the different integration techniques are applied and their results are printed.

This case study demonstrates the practical application of Simpson's Rule and its variants in a realistic scenario, showcasing their implementation details, error analysis, and comparative performance. It serves as a comprehensive example that brings together the concepts and techniques covered throughout the slideshow.

## Meta
Here is a suggested TikTok title, description, and hashtags with an institutional tone for our conversation about creating a slideshow on Simpson's Rule in Python:

Mastering Numerical Integration: Simpson's Rule in Python

Explore the powerful Simpson's Rule integration technique through an in-depth slideshow journey. Learn the theory, implementation, and applications of this essential numerical method in Python. Gain insights into accuracy, error analysis, and advanced techniques for tackling complex integrals. Perfect for students, researchers, and professionals in STEM fields. #NumericalAnalysis #IntegrationTechniques #SimpsonsRule #Python #STEM #LearningResources

Hashtags: #NumericalAnalysis #IntegrationTechniques  
#SimpsonsRule #Python #STEM #LearningResources #ComputationalMathematics #CodeTutorials #EducationalContent

The title "Mastering Numerical Integration: Simpson's Rule in Python" establishes the topic and conveys an authoritative and educational tone. The description provides an overview of the content, highlighting the key aspects covered in the slideshow while maintaining an institutional and informative style. The hashtags are relevant to the topic and cater to audiences interested in numerical analysis, integration techniques, Python programming, STEM education, and learning resources.

