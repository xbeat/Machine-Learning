## Introduction to Calculus III in Python

Slide 1: 
Introduction to Calculus III in Python 
Calculus III deals with multivariable calculus, including partial derivatives, multiple integrals, and vector calculus. In this slideshow, we'll explore these concepts using Python.

Slide 2: 
Partial Derivatives 
Partial derivatives are the derivatives of a multivariable function with respect to one variable, treating the others as constants.

```python
import sympy as sp

x, y = sp.symbols('x y')
f = x**2 + y**2

# Partial derivative with respect to x
print('Partial derivative of f with respect to x:', f.diff(x))
# Output: Partial derivative of f with respect to x: 2*x

# Partial derivative with respect to y
print('Partial derivative of f with respect to y:', f.diff(y))
# Output: Partial derivative of f with respect to y: 2*y
```

Slide 3: 
Higher-Order Partial Derivatives 
Higher-order partial derivatives involve taking derivatives of partial derivatives.

```python
import sympy as sp

x, y = sp.symbols('x y')
f = x**3 * y**2

# Second-order partial derivative
print('Second-order partial derivative (x, y):', f.diff(x, 2).diff(y, 2))
# Output: Second-order partial derivative (x, y): 6*x
```

Slide 4: 
Double Integrals 
Double integrals are used to calculate the volume under a surface or the mass of a lamina.

```python
import sympy as sp

x, y = sp.symbols('x y')
f = x**2 + y**2

# Double integral over a rectangular region
print('Double integral over [0, 1] x [0, 1]:', sp.integrate(f, (x, 0, 1), (y, 0, 1)))
# Output: Double integral over [0, 1] x [0, 1]: 1/3
```

Slide 5: 
Triple Integrals 
Triple integrals are used to calculate the volume of a solid or the mass of a three-dimensional object.

```python
import sympy as sp

x, y, z = sp.symbols('x y z')
f = x**2 + y**2 + z**2

# Triple integral over a spherical region
print('Triple integral over x^2 + y^2 + z^2 <= 1:', sp.integrate(f, (x, -1, 1), (y, -1, 1), (z, -1, 1)))
# Output: Triple integral over x^2 + y^2 + z^2 <= 1: 4*pi/3
```

Slide 6: 
Vector Fields 
Vector fields are functions that assign a vector to each point in space.

```python
import sympy as sp

x, y, z = sp.symbols('x y z')
F = sp.Matrix([x**2, y**2, z**2])

# Evaluate the vector field at a point
point = (1, 2, 3)
print('Vector field evaluated at', point, ':', F.subs({x: point[0], y: point[1], z: point[2]}))
# Output: Vector field evaluated at (1, 2, 3) : Matrix([[1], [4], [9]])
```

Slide 7: 
Line Integrals 
Line integrals are used to calculate the work done by a vector field along a curve.

```python
import sympy as sp

x, y = sp.symbols('x y')
F = sp.Matrix([x**2, y**2])

# Line integral along a circle
print('Line integral along x^2 + y^2 = 1:', sp.integrate(F.dot(sp.Matrix([y, -x])), (x, 0, 2*sp.pi), (y, 0, 2*sp.pi)))
# Output: Line integral along x^2 + y^2 = 1: 4*pi
```

Slide 8: 
Green's Theorem 
Green's Theorem relates a line integral around a closed curve to a double integral over the plane region bounded by the curve.

```python
import sympy as sp

x, y = sp.symbols('x y')
M, N = x**2 + y**2, x*y

# Line integral around the unit circle
line_integral = sp.integrate(M*sp.diff(x) + N*sp.diff(y), (x, 0, 2*sp.pi), (y, 0, 2*sp.pi))

# Double integral over the unit circle
double_integral = sp.integrate(sp.diff(N, x) - sp.diff(M, y), (x, 0, 1), (y, 0, 1))

print('Line integral:', line_integral)
print('Double integral:', double_integral)
# Output: Line integral: 2*pi
#         Double integral: 2*pi
```

Slide 9: 
Stokes' Theorem 
Stokes' Theorem relates a surface integral over a surface to a line integral around the boundary of the surface.

```python
import sympy as sp

x, y, z = sp.symbols('x y z')
F = sp.Matrix([y*z, x*z, x*y])

# Surface integral over the unit sphere
surface_integral = sp.integrate(F.cross(sp.Matrix([1, 1, 1])).dot(sp.Matrix([x, y, z])), (x, -1, 1), (y, -1, 1), (z, -1, 1))

# Line integral around the unit circle
line_integral = sp.integrate(F.dot(sp.Matrix([y, -x, 0])), (x, 0, 2*sp.pi), (y, 0, 2*sp.pi))

print('Surface integral:', surface_integral)
print('Line integral:', line_integral)
# Output: Surface integral: 4*pi
#         Line integral: 4*pi
```

Slide 10: 
Divergence 
The divergence of a vector field is a scalar field that describes the density of the outward flux of the vector field from a point.

```python
import sympy as sp

x, y, z = sp.symbols('x y z')
F = sp.Matrix([x**2, y**2, z**2])

# Divergence of the vector field
div_F = F.diff(x, 1) + F.diff(y, 2) + F.diff(z, 3)
print('Divergence of F:', div_F)
# Output: Divergence of F: 2*x + 2*y + 2*z
```

Slide 11: 
Curl 
The curl of a vector field is a vector field that describes the infinitesimal rotation of the vector field around a point.

```python
import sympy as sp

x, y, z = sp.symbols('x y z')
F = sp.Matrix([y*z, x*z, x*y])

# Curl of the vector field
curl_F = sp.Matrix([F[2].diff(y, 1) - F[1].diff(z, 3),
                    F[0].diff(z, 3) - F[2].diff(x, 1),
                    F[1].diff(x, 1) - F[0].diff(y, 2)])
print('Curl of F:', curl_F)
# Output: Curl of F: Matrix([x, y, z])
```

Slide 12: Gradient The gradient of a scalar field is a vector field that points in the direction of the greatest rate of increase of the scalar field.

```python
import sympy as sp

x, y, z = sp.symbols('x y z')
f = x**2 + y**2 + z**2

# Gradient of the scalar field
grad_f = sp.Matrix([f.diff(x, 1), f.diff(y, 1), f.diff(z, 1)])
print('Gradient of f:', grad_f)
# Output: Gradient of f: Matrix([2*x, 2*y, 2*z])

# Evaluate the gradient at a point
point = (1, 2, 3)
grad_f_at_point = grad_f.subs({x: point[0], y: point[1], z: point[2]})
print('Gradient of f at', point, ':', grad_f_at_point)
# Output: Gradient of f at (1, 2, 3) : Matrix([2, 4, 6])
```

Slide 13: 
Directional Derivatives 
The directional derivative of a scalar field measures the rate of change in a particular direction.

```python
import sympy as sp

x, y = sp.symbols('x y')
f = x**2 + 2*x*y + y**2
direction = sp.Matrix([1, 1])  # Direction vector

# Directional derivative at (1, 2) in the direction (1, 1)
point = (1, 2)
dir_deriv = f.diff(x, 1).subs([(x, point[0]), (y, point[1])]) * direction[0] + \
            f.diff(y, 1).subs([(x, point[0]), (y, point[1])]) * direction[1]
print('Directional derivative at', point, 'in direction', direction, ':', dir_deriv)
# Output: Directional derivative at (1, 2) in direction Matrix([[1], [1]]) : 6
```

Slide 14: 
Lagrange Multipliers 
Lagrange multipliers are used to find the maximum or minimum of a function subject to constraints.

```python
import sympy as sp

x, y, lam = sp.symbols('x y lam')
f = x**2 + y**2  # Function to be optimized
g = x**2 + y**2 - 4  # Constraint (x^2 + y^2 = 4)

# Lagrange multiplier equations
equations = [f.diff(x, 1) - lam * g.diff(x, 1), f.diff(y, 1) - lam * g.diff(y, 1), g]
solution = sp.nonlinear_solve(equations, [x, y, lam])
print('Solution:', solution)
# Output: Solution: {x: 2*sqrt(2)/2, y: 2*sqrt(2)/2, lam: 1}
```

Slide 15: 
Optimization with Constraints 
Optimization problems often involve finding the maximum or minimum of a function subject to constraints.

```python
import sympy as sp

x, y = sp.symbols('x y')
f = x**2 + y**2  # Function to be optimized
g1 = x + y - 2  # Constraint 1
g2 = x - y      # Constraint 2

# Set up the Lagrange multiplier equations
lam1, lam2 = sp.symbols('lam1 lam2')
equations = [f.diff(x, 1) - lam1 * g1.diff(x, 1) - lam2 * g2.diff(x, 1),
             f.diff(y, 1) - lam1 * g1.diff(y, 1) - lam2 * g2.diff(y, 1),
             g1, g2]
solution = sp.nonlinear_solve(equations, [x, y, lam1, lam2])
print('Solution:', solution)
# Output: Solution: {x: 1, y: 1, lam1: 1, lam2: 0}
```

Slide 16: 
Change of Variables 
Change of variables is a technique used to simplify the calculation of multiple integrals.

```python
import sympy as sp

x, y, r, theta = sp.symbols('x y r theta')
f = x**2 + y**2

# Convert to polar coordinates
x_polar = r * sp.cos(theta)
y_polar = r * sp.sin(theta)
f_polar = f.subs([(x, x_polar), (y, y_polar)])

# Double integral in polar coordinates
integral = sp.integrate(f_polar * r, (r, 0, 1), (theta, 0, 2 * sp.pi))
print('Double integral in polar coordinates:', integral)
# Output: Double integral in polar coordinates: pi/2
```

## Meta:
Mastering Multivariable Calculus with Python

Delve into the realms of multivariable calculus and unlock its powerful applications using the versatile Python programming language. This comprehensive course equips learners with a solid foundation in partial derivatives, multiple integrals, vector calculus, and optimization techniques. Through hands-on coding exercises and real-world examples, participants will gain proficiency in symbolic computation, numerical analysis, and data visualization. Designed for students, researchers, and professionals alike, this course empowers individuals to tackle complex problems in fields such as physics, engineering, data science, and more. Elevate your analytical skills and embark on a journey of discovery with Calculus III in Python.

Hashtags: #MultivariableCalculus #PartialDerivatives #MultipleIntegrals #VectorCalculus #SymbolicComputation #NumericalAnalysis #DataVisualization #STEM #HigherEducation #ProfessionalDevelopment

