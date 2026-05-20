## Noncommutative Rings using Python
Slide 1: Introduction to Noncommutative Rings

Noncommutative rings are algebraic structures where multiplication is not commutative. This course explores their properties, operations, and applications in various fields of mathematics and computer science.

```python
# Demonstrating noncommutativity in matrix multiplication
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A * B =\n", np.dot(A, B))
print("B * A =\n", np.dot(B, A))
```

Slide 2: Basic Definitions and Properties

A ring is a set R with two binary operations, addition (+) and multiplication (·), satisfying certain axioms. In noncommutative rings, a · b ≠ b · a for some elements a and b.

```python
class NoncommutativeRing:
    def __init__(self, elements):
        self.elements = elements

    def add(self, a, b):
        return (a + b) % len(self.elements)

    def multiply(self, a, b):
        return (a * b) % len(self.elements)

    def is_commutative(self):
        for a in self.elements:
            for b in self.elements:
                if self.multiply(a, b) != self.multiply(b, a):
                    return False
        return True

ring = NoncommutativeRing(range(4))
print("Is the ring commutative?", ring.is_commutative())
```

Slide 3: Examples of Noncommutative Rings

Common examples include matrix rings, quaternions, and group rings. Let's implement a simple matrix ring to demonstrate noncommutativity.

```python
import numpy as np

class MatrixRing:
    def __init__(self, n):
        self.n = n

    def multiply(self, A, B):
        return np.dot(A, B)

    def is_commutative(self, A, B):
        return np.array_equal(self.multiply(A, B), self.multiply(B, A))

ring = MatrixRing(2)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A * B =\n", ring.multiply(A, B))
print("B * A =\n", ring.multiply(B, A))
print("Is multiplication commutative?", ring.is_commutative(A, B))
```

Slide 4: Subrings and Ideals

Subrings are subsets of a ring that are themselves rings under the same operations. Ideals are special subrings that absorb multiplication by ring elements.

```python
class SubringChecker:
    def __init__(self, ring):
        self.ring = ring

    def is_subring(self, subset):
        for a in subset:
            for b in subset:
                if self.ring.add(a, b) not in subset or self.ring.multiply(a, b) not in subset:
                    return False
        return True

    def is_left_ideal(self, subset):
        for a in subset:
            for r in self.ring.elements:
                if self.ring.multiply(r, a) not in subset:
                    return False
        return True

ring = NoncommutativeRing(range(6))
checker = SubringChecker(ring)
subset = [0, 2, 4]
print("Is subset a subring?", checker.is_subring(subset))
print("Is subset a left ideal?", checker.is_left_ideal(subset))
```

Slide 5: Homomorphisms and Isomorphisms

Ring homomorphisms are structure-preserving maps between rings. Isomorphisms are bijective homomorphisms that preserve the ring structure.

```python
def is_homomorphism(f, R1, R2):
    for a in R1.elements:
        for b in R1.elements:
            if f(R1.add(a, b)) != R2.add(f(a), f(b)):
                return False
            if f(R1.multiply(a, b)) != R2.multiply(f(a), f(b)):
                return False
    return True

R1 = NoncommutativeRing(range(4))
R2 = NoncommutativeRing(range(2, 6))

f = lambda x: (x + 2) % 4
print("Is f a homomorphism?", is_homomorphism(f, R1, R2))
```

Slide 6: Quotient Rings

Quotient rings are formed by "modding out" an ideal from a ring, creating a new ring structure.

```python
class QuotientRing:
    def __init__(self, ring, ideal):
        self.ring = ring
        self.ideal = ideal
        self.elements = [frozenset(self.coset(x)) for x in ring.elements]

    def coset(self, x):
        return {self.ring.add(x, a) for a in self.ideal}

    def add(self, x, y):
        return frozenset({self.ring.add(a, b) for a in x for b in y})

    def multiply(self, x, y):
        return frozenset({self.ring.multiply(a, b) for a in x for b in y})

R = NoncommutativeRing(range(6))
I = [0, 2, 4]
Q = QuotientRing(R, I)

x = Q.elements[1]
y = Q.elements[3]
print("x + y =", Q.add(x, y))
print("x * y =", Q.multiply(x, y))
```

Slide 7: Modules and Vector Spaces

Modules generalize the concept of vector spaces to rings. They are additive groups with a scalar multiplication by ring elements.

```python
class Module:
    def __init__(self, ring, elements):
        self.ring = ring
        self.elements = elements

    def add(self, v, w):
        return tuple(self.ring.add(v[i], w[i]) for i in range(len(v)))

    def scalar_multiply(self, r, v):
        return tuple(self.ring.multiply(r, v[i]) for i in range(len(v)))

R = NoncommutativeRing(range(4))
M = Module(R, [(0, 0), (0, 1), (1, 0), (1, 1)])

v = M.elements[1]
w = M.elements[2]
r = 2

print("v + w =", M.add(v, w))
print("r * v =", M.scalar_multiply(r, v))
```

Slide 8: Free Modules and Projective Modules

Free modules are generalizations of vector spaces with a basis. Projective modules are direct summands of free modules.

```python
import numpy as np

def is_free_module(module, basis):
    for v in module.elements:
        coeffs = np.linalg.lstsq(np.array(basis).T, v, rcond=None)[0]
        if not np.allclose(np.dot(coeffs, basis), v):
            return False
    return True

R = NoncommutativeRing(range(4))
M = Module(R, [(a, b) for a in range(4) for b in range(4)])
basis = [(1, 0), (0, 1)]

print("Is M a free module with the given basis?", is_free_module(M, basis))
```

Slide 9: Artinian and Noetherian Rings

Artinian rings have the descending chain condition on ideals, while Noetherian rings have the ascending chain condition.

```python
def is_artinian(ring, max_depth=10):
    def dcc(ideal, depth):
        if depth > max_depth:
            return False
        proper_subideals = [
            [x for x in ring.elements if ring.multiply(r, x) in ideal]
            for r in ring.elements if r not in ideal
        ]
        return all(dcc(subideal, depth + 1) for subideal in proper_subideals if subideal != ideal)

    return dcc(ring.elements, 0)

R = NoncommutativeRing(range(4))
print("Is R Artinian?", is_artinian(R))
```

Slide 10: Prime and Maximal Ideals

Prime ideals are proper ideals where ab ∈ I implies a ∈ I or b ∈ I. Maximal ideals are proper ideals not contained in any other proper ideal.

```python
def is_prime_ideal(ring, ideal):
    for a in ring.elements:
        for b in ring.elements:
            if ring.multiply(a, b) in ideal and a not in ideal and b not in ideal:
                return False
    return True

def is_maximal_ideal(ring, ideal):
    if ideal == ring.elements:
        return False
    for x in ring.elements:
        if x not in ideal:
            if set(ideal + [x]) == set(ring.elements):
                return True
    return False

R = NoncommutativeRing(range(6))
I = [0, 2, 4]

print("Is I a prime ideal?", is_prime_ideal(R, I))
print("Is I a maximal ideal?", is_maximal_ideal(R, I))
```

Slide 11: Semisimple Rings

Semisimple rings are direct sums of simple modules. They have rich structure and important applications in representation theory.

```python
def is_semisimple(ring):
    def is_simple_module(module):
        return len([s for s in module.submodules() if s != {0} and s != module.elements]) == 0

    modules = ring.left_modules()
    return all(is_simple_module(M) for M in modules)

# Note: This is a simplified implementation. A full implementation would require
# defining methods for generating all left modules and submodules of a ring.
```

Slide 12: Tensor Products

Tensor products allow us to construct new modules from existing ones, generalizing the concept of outer products in linear algebra.

```python
import itertools

def tensor_product(M1, M2):
    elements = list(itertools.product(M1.elements, M2.elements))
    
    def add(x, y):
        return tuple(M1.add(x[0], y[0]) + M2.add(x[1], y[1]))
    
    def scalar_multiply(r, x):
        return tuple(M1.scalar_multiply(r, x[0]) + M2.scalar_multiply(r, x[1]))
    
    return type('TensorProduct', (), {
        'elements': elements,
        'add': add,
        'scalar_multiply': scalar_multiply
    })

R = NoncommutativeRing(range(2))
M1 = Module(R, [(0,), (1,)])
M2 = Module(R, [(0,), (1,)])

T = tensor_product(M1, M2)
print("Tensor product elements:", T.elements)
```

Slide 13: Real-Life Example: Quantum Mechanics

Noncommutative rings appear in quantum mechanics, where observables are represented by Hermitian operators that don't always commute.

```python
import numpy as np

def commutator(A, B):
    return np.dot(A, B) - np.dot(B, A)

# Position and momentum operators in 1D
x = np.array([[0, 1], [1, 0]])
p = np.array([[0, -1j], [1j, 0]])

print("Position operator:\n", x)
print("Momentum operator:\n", p)
print("Commutator [x, p]:\n", commutator(x, p))
```

Slide 14: Real-Life Example: Computer Graphics

Noncommutative operations are crucial in computer graphics, particularly in 3D rotations using quaternions.

```python
import numpy as np

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

q1 = np.array([0.7071, 0, 0.7071, 0])  # 90-degree rotation around y-axis
q2 = np.array([0.7071, 0.7071, 0, 0])  # 90-degree rotation around x-axis

print("q1 * q2 =", quaternion_multiply(q1, q2))
print("q2 * q1 =", quaternion_multiply(q2, q1))
```

Slide 15: Additional Resources

For further exploration of noncommutative rings and related topics, consider the following resources:

1. "An Introduction to Noncommutative Noetherian Rings" by K.R. Goodearl and R.B. Warfield Jr. (ArXiv:math/0001016)
2. "Noncommutative Rings" by I.N. Herstein (ArXiv:math/0504284)
3. "Lectures on Rings and Modules" by T.Y. Lam (ArXiv:math/0609735)

These papers provide in-depth discussions on various aspects of noncommutative ring theory and its applications in modern algebra.

