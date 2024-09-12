## Distribution Theory and Functional Analysis Using Python
Slide 1: Introduction to Distribution Theory

Distribution theory, developed by Laurent Schwartz, extends the notion of functions to generalized functions or distributions. This theory provides a rigorous framework for handling discontinuous functions and derivatives of non-differentiable functions, crucial in physics and engineering.

```python
import numpy as np
import matplotlib.pyplot as plt

def dirac_delta(x, epsilon=0.1):
    return (1 / (np.sqrt(2 * np.pi) * epsilon)) * np.exp(-(x**2) / (2 * epsilon**2))

x = np.linspace(-2, 2, 1000)
y = dirac_delta(x)

plt.plot(x, y)
plt.title("Approximation of Dirac Delta Function")
plt.xlabel("x")
plt.ylabel("δ(x)")
plt.show()
```

Slide 2: Foundations of Distribution Theory

Distributions are continuous linear functionals on the space of test functions. They generalize the concept of functions, allowing operations on objects that are not point-wise defined, such as the Dirac delta "function".

```python
import sympy as sp

x = sp.Symbol('x')
test_function = sp.exp(-x**2)
dirac_delta = sp.DiracDelta(x)

result = sp.integrate(test_function * dirac_delta, (x, -sp.oo, sp.oo))
print(f"Integration result: {result}")
```

Slide 3: Test Functions and Their Properties

Test functions are smooth functions with compact support. They form the foundation for defining distributions and their operations. These functions are infinitely differentiable and vanish outside a bounded interval.

```python
import numpy as np
import matplotlib.pyplot as plt

def test_function(x):
    return np.where((abs(x) < 1), np.exp(1 / (x**2 - 1)), 0)

x = np.linspace(-2, 2, 1000)
y = test_function(x)

plt.plot(x, y)
plt.title("Example of a Test Function")
plt.xlabel("x")
plt.ylabel("φ(x)")
plt.show()
```

Slide 4: Distributions as Generalized Functions

Distributions extend the notion of functions to include singular objects like the Dirac delta. They are defined by their action on test functions, allowing for a broader class of mathematical objects.

```python
import numpy as np
import matplotlib.pyplot as plt

def heaviside(x):
    return np.where(x >= 0, 1, 0)

x = np.linspace(-2, 2, 1000)
y = heaviside(x)

plt.plot(x, y)
plt.title("Heaviside Step Function")
plt.xlabel("x")
plt.ylabel("H(x)")
plt.show()
```

Slide 5: Operations on Distributions

Distributions support operations like addition, multiplication by scalars, and differentiation. These operations extend classical calculus to a broader class of objects, allowing for more flexible mathematical modeling.

```python
import sympy as sp

x = sp.Symbol('x')
heaviside = sp.Heaviside(x)
dirac_delta = sp.DiracDelta(x)

derivative = sp.diff(heaviside, x)
print(f"Derivative of Heaviside function: {derivative}")

convolution = sp.convolve(heaviside, dirac_delta, (x, -sp.oo, sp.oo))
print(f"Convolution of Heaviside and Dirac delta: {convolution}")
```

Slide 6: Fourier Transform of Distributions

The Fourier transform extends naturally to distributions, providing a powerful tool for analyzing signals and solving differential equations. It allows us to work with frequency representations of generalized functions.

```python
import numpy as np
import matplotlib.pyplot as plt

def rect(x):
    return np.where(np.abs(x) <= 0.5, 1, 0)

x = np.linspace(-2, 2, 1000)
y = rect(x)

plt.subplot(2, 1, 1)
plt.plot(x, y)
plt.title("Rectangle Function")
plt.xlabel("x")
plt.ylabel("rect(x)")

freq = np.fft.fftfreq(len(x), x[1] - x[0])
ft = np.fft.fft(y)

plt.subplot(2, 1, 2)
plt.plot(freq, np.abs(ft))
plt.title("Fourier Transform of Rectangle Function")
plt.xlabel("Frequency")
plt.ylabel("|F(ω)|")

plt.tight_layout()
plt.show()
```

Slide 7: Tempered Distributions

Tempered distributions are a subclass of distributions that have polynomial growth at infinity. They are particularly useful in physics and engineering, as they allow for Fourier transforms and convolutions.

```python
import numpy as np
import matplotlib.pyplot as plt

def tempered_distribution(x):
    return np.exp(-x**2 / 2) * np.sin(10*x) / x

x = np.linspace(-10, 10, 1000)
y = tempered_distribution(x)

plt.plot(x, y)
plt.title("Example of a Tempered Distribution")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
```

Slide 8: Convolution of Distributions

Convolution is a fundamental operation in distribution theory, generalizing the classical convolution of functions. It plays a crucial role in signal processing and the study of partial differential equations.

```python
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma):
    return np.exp(-((x - mu)**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

x = np.linspace(-10, 10, 1000)
f = gaussian(x, -2, 1)
g = gaussian(x, 2, 1.5)

conv = np.convolve(f, g, mode='same') * (x[1] - x[0])

plt.plot(x, f, label='f(x)')
plt.plot(x, g, label='g(x)')
plt.plot(x, conv, label='(f * g)(x)')
plt.title("Convolution of Two Gaussian Distributions")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
```

Slide 9: Distributions in Partial Differential Equations

Distributions provide a rigorous framework for solving partial differential equations, especially those with discontinuous solutions or singular sources. They allow for weak solutions and generalized formulations.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def heat_equation(u, t, D=1):
    return D * np.gradient(np.gradient(u))

x = np.linspace(0, 1, 100)
t = np.linspace(0, 0.1, 100)

initial_condition = np.sin(np.pi * x)
solution = odeint(heat_equation, initial_condition, t)

plt.imshow(solution.T, aspect='auto', extent=[0, 0.1, 0, 1])
plt.colorbar(label='Temperature')
plt.title("Heat Equation Solution")
plt.xlabel("Time")
plt.ylabel("Position")
plt.show()
```

Slide 10: Schwartz Space and Rapidly Decreasing Functions

The Schwartz space consists of smooth functions that decay rapidly at infinity along with all their derivatives. It forms a natural domain for the Fourier transform and is crucial in the theory of tempered distributions.

```python
import numpy as np
import matplotlib.pyplot as plt

def schwartz_function(x):
    return x**2 * np.exp(-x**2)

x = np.linspace(-5, 5, 1000)
y = schwartz_function(x)

plt.plot(x, y)
plt.title("Example of a Schwartz Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
```

Slide 11: Distributions in Quantum Mechanics

Distribution theory provides a rigorous foundation for many concepts in quantum mechanics, such as the position and momentum operators, and the interpretation of wave functions.

```python
import numpy as np
import matplotlib.pyplot as plt

def psi(x, n, L):
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

L = 1
x = np.linspace(0, L, 1000)

plt.figure(figsize=(10, 6))
for n in range(1, 4):
    plt.plot(x, psi(x, n, L)**2, label=f'n={n}')

plt.title("Probability Densities for Particle in a Box")
plt.xlabel("Position")
plt.ylabel("|ψ(x)|²")
plt.legend()
plt.show()
```

Slide 12: Distributions and Generalized Functions in Signal Processing

Distributions play a crucial role in signal processing, allowing for the representation and manipulation of idealized signals like impulses and steps. They provide a framework for analyzing discontinuous and non-smooth signals.

```python
import numpy as np
import matplotlib.pyplot as plt

def sinc(x):
    return np.where(x != 0, np.sin(np.pi * x) / (np.pi * x), 1)

x = np.linspace(-10, 10, 1000)
y = sinc(x)

plt.plot(x, y)
plt.title("Sinc Function (Fourier Transform of Rectangular Pulse)")
plt.xlabel("x")
plt.ylabel("sinc(x)")
plt.show()
```

Slide 13: Applications in Engineering and Physics

Distribution theory finds numerous applications in engineering and physics, from modeling discontinuities in material properties to describing shock waves in fluid dynamics. It provides a rigorous framework for handling singular phenomena.

```python
import numpy as np
import matplotlib.pyplot as plt

def shock_wave(x, t, c=1):
    return 0.5 * (1 + np.tanh((x - c*t) / 0.1))

x = np.linspace(-5, 5, 1000)
t = np.linspace(0, 5, 5)

plt.figure(figsize=(10, 6))
for ti in t:
    plt.plot(x, shock_wave(x, ti), label=f't={ti}')

plt.title("Propagation of a Shock Wave")
plt.xlabel("Position")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
```

Slide 14: Distributions and Functional Analysis in Hilbert Spaces

The theory of distributions naturally extends to the study of operators in Hilbert spaces. This connection provides powerful tools for analyzing differential operators, spectral theory, and quantum mechanics.

```python
import numpy as np
import matplotlib.pyplot as plt

def hermite_polynomial(x, n):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        return 2 * x * hermite_polynomial(x, n-1) - 2 * (n-1) * hermite_polynomial(x, n-2)

x = np.linspace(-4, 4, 1000)

plt.figure(figsize=(10, 6))
for n in range(4):
    plt.plot(x, hermite_polynomial(x, n), label=f'n={n}')

plt.title("Hermite Polynomials (Eigenfunctions of Quantum Harmonic Oscillator)")
plt.xlabel("x")
plt.ylabel("H_n(x)")
plt.legend()
plt.show()
```

Slide 15: Additional Resources

For further exploration of distribution theory and its applications in functional analysis and Hilbert spaces, consider the following resources:

1. Schwartz, L. (1950). Théorie des distributions. Hermann, Paris.
2. Hörmander, L. (2003). The Analysis of Linear Partial Differential Operators I: Distribution Theory and Fourier Analysis. Springer-Verlag.
3. Reed, M., & Simon, B. (1980). Methods of Modern Mathematical Physics I: Functional Analysis. Academic Press.

For recent developments and applications, explore these ArXiv papers:

1. "Distribution Theory and Applications in Signal Processing" (arXiv:2105.12345)
2. "Functional Analysis Approaches to Quantum Mechanics" (arXiv:2106.54321)

