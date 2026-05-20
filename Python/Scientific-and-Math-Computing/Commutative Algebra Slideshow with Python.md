## Commutative Algebra Slideshow with Python

Slide 1: Introduction to Commutative Algebra

Commutative algebra is a branch of mathematics that studies commutative rings and their ideals. It forms the foundation for algebraic geometry and has applications in various fields, including cryptography and coding theory.

```python
# Example of a commutative operation
def commutative_addition(a, b):
    return a + b

# Demonstrating commutativity
x, y = 5, 3
print(f"{x} + {y} = {commutative_addition(x, y)}")
print(f"{y} + {x} = {commutative_addition(y, x)}")
```

Slide 2: Rings and Commutative Rings

A ring is an algebraic structure with two operations: addition and multiplication. A commutative ring is a ring where the multiplication operation is commutative, meaning a \* b = b \* a for all elements a and b in the ring.

```python
class CommutativeRing:
    def __init__(self, elements):
        self.elements = set(elements)
    
    def add(self, a, b):
        return (a + b) % len(self.elements)
    
    def multiply(self, a, b):
        return (a * b) % len(self.elements)

# Example: Ring of integers modulo 5
Z5 = CommutativeRing(range(5))
print(f"2 * 3 = {Z5.multiply(2, 3)}")
print(f"3 * 2 = {Z5.multiply(3, 2)}")
```

Slide 3: Ideals

An ideal is a subset of a ring that is closed under addition and multiplication by ring elements. Ideals play a crucial role in ring theory and are used to construct quotient rings.

```python
def is_ideal(ring, subset):
    for a in subset:
        for b in subset:
            if (ring.add(a, b) not in subset):
                return False
        for r in ring.elements:
            if (ring.multiply(a, r) not in subset):
                return False
    return True

# Example: Check if {0, 2, 4} is an ideal in Z6
Z6 = CommutativeRing(range(6))
subset = {0, 2, 4}
print(f"Is {subset} an ideal in Z6? {is_ideal(Z6, subset)}")
```

Slide 4: Prime Ideals

A prime ideal is a proper ideal P of a ring R such that for any two elements a and b in R, if their product ab is in P, then either a is in P or b is in P. Prime ideals are fundamental in studying the structure of rings.

```python
def is_prime_ideal(ring, ideal):
    if not is_ideal(ring, ideal) or ideal == ring.elements:
        return False
    for a in ring.elements:
        for b in ring.elements:
            if ring.multiply(a, b) in ideal and a not in ideal and b not in ideal:
                return False
    return True

# Example: Check if {0, 2, 4} is a prime ideal in Z6
Z6 = CommutativeRing(range(6))
ideal = {0, 2, 4}
print(f"Is {ideal} a prime ideal in Z6? {is_prime_ideal(Z6, ideal)}")
```

Slide 5: Maximal Ideals

A maximal ideal is a proper ideal that is not contained in any other proper ideal of the ring. Maximal ideals are important in the study of field theory and algebraic geometry.

```python
def is_maximal_ideal(ring, ideal):
    if not is_ideal(ring, ideal) or ideal == ring.elements:
        return False
    for element in ring.elements - ideal:
        new_ideal = ideal.union({element})
        if is_ideal(ring, new_ideal) and new_ideal != ring.elements:
            return False
    return True

# Example: Check if {0, 3} is a maximal ideal in Z6
Z6 = CommutativeRing(range(6))
ideal = {0, 3}
print(f"Is {ideal} a maximal ideal in Z6? {is_maximal_ideal(Z6, ideal)}")
```

Slide 6: Quotient Rings

A quotient ring is constructed by taking a ring R and an ideal I, and creating a new ring R/I. The elements of R/I are equivalence classes of elements in R modulo the ideal I.

```python
class QuotientRing:
    def __init__(self, ring, ideal):
        self.ring = ring
        self.ideal = ideal
        self.elements = [frozenset({x for x in ring.elements if (x - e) % len(ring.elements) in ideal}) for e in ring.elements]
    
    def add(self, a, b):
        return frozenset({(x + y) % len(self.ring.elements) for x in a for y in b})
    
    def multiply(self, a, b):
        return frozenset({(x * y) % len(self.ring.elements) for x in a for y in b})

# Example: Construct Z6/{0, 3}
Z6 = CommutativeRing(range(6))
ideal = {0, 3}
Z6_mod_3 = QuotientRing(Z6, ideal)
print("Elements of Z6/{0, 3}:", Z6_mod_3.elements)
```

Slide 7: Polynomial Rings

Polynomial rings are commutative rings formed by polynomials over a given ring. They are essential in algebraic geometry and play a crucial role in solving systems of polynomial equations.

```python
from collections import defaultdict

class Polynomial:
    def __init__(self, coeffs):
        self.coeffs = defaultdict(int, coeffs)
    
    def __add__(self, other):
        result = defaultdict(int)
        for exp in set(self.coeffs.keys()) | set(other.coeffs.keys()):
            result[exp] = self.coeffs[exp] + other.coeffs[exp]
        return Polynomial(result)
    
    def __str__(self):
        terms = [f"{coeff}x^{exp}" for exp, coeff in sorted(self.coeffs.items(), reverse=True) if coeff != 0]
        return " + ".join(terms) or "0"

# Example: Add two polynomials
p1 = Polynomial({2: 1, 1: 2, 0: 1})  # x^2 + 2x + 1
p2 = Polynomial({3: 1, 1: -1, 0: 2})  # x^3 - x + 2
print(f"({p1}) + ({p2}) = {p1 + p2}")
```

Slide 8: Gröbner Bases

Gröbner bases are a powerful tool in computational algebra for solving systems of polynomial equations and performing ideal operations. They provide a systematic way to handle multivariate polynomials.

```python
from sympy import groebner, symbols, poly

def compute_groebner_basis(polynomials, variables):
    return groebner(polynomials, variables, order='lex')

# Example: Compute Gröbner basis for a system of polynomials
x, y = symbols('x y')
f1 = poly(x**2 + y**2 - 1, x, y)
f2 = poly(x**2 - y, x, y)
basis = compute_groebner_basis([f1, f2], (x, y))
print("Gröbner basis:", basis)
```

Slide 9: Primary Decomposition

Primary decomposition is the process of expressing an ideal as an intersection of primary ideals. This concept generalizes the factorization of integers into prime powers and is crucial in understanding the structure of ideals.

```python
from sympy import symbols, poly, factor_list

def primary_decomposition(polynomial):
    factors = factor_list(polynomial)[1]
    return [poly(factor**exp, polynomial.gens) for factor, exp in factors]

# Example: Perform primary decomposition of a polynomial
x = symbols('x')
f = poly(x**4 - 1, x)
decomp = primary_decomposition(f)
print("Primary decomposition of", f)
for factor in decomp:
    print(factor)
```

Slide 10: Localization

Localization is a technique in commutative algebra that allows us to focus on a specific part of a ring by inverting a set of elements. This process is crucial in algebraic geometry for studying local properties of varieties.

```python
class LocalizedRing:
    def __init__(self, ring, S):
        self.ring = ring
        self.S = S
    
    def element(self, a, s):
        return (a, s)
    
    def multiply(self, elem1, elem2):
        a1, s1 = elem1
        a2, s2 = elem2
        return self.element(self.ring.multiply(a1, a2), self.ring.multiply(s1, s2))

# Example: Localize Z at {2^n | n >= 0}
Z = CommutativeRing(range(-100, 101))  # Approximation of Z
S = {2**n for n in range(10)}  # Approximation of {2^n | n >= 0}
Z_localized = LocalizedRing(Z, S)

# Represent 3/4 in the localized ring
elem = Z_localized.element(3, 4)
print("3/4 represented as:", elem)
```

Slide 11: Tensor Products

Tensor products are a way to combine vector spaces or modules to create new, larger spaces. They are fundamental in many areas of mathematics and have applications in physics and engineering.

```python
import numpy as np

def tensor_product(A, B):
    return np.kron(A, B)

# Example: Compute tensor product of two matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [1, 0]])
C = tensor_product(A, B)
print("Tensor product of A and B:")
print(C)
```

Slide 12: Noetherian Rings

A Noetherian ring is a ring in which every ascending chain of ideals eventually stabilizes. This property ensures that certain algebraic operations terminate, making Noetherian rings particularly useful in computational algebra.

```python
def is_noetherian(ring, max_chain_length=100):
    ideal = set()
    for _ in range(max_chain_length):
        new_element = next((x for x in ring.elements if x not in ideal), None)
        if new_element is None:
            return True
        ideal.add(new_element)
        if not is_ideal(ring, ideal):
            return False
    return False  # Inconclusive, chain might be infinite

# Example: Check if Z6 is Noetherian
Z6 = CommutativeRing(range(6))
print(f"Is Z6 Noetherian? {is_noetherian(Z6)}")
```

Slide 13: Cohen-Macaulay Rings

Cohen-Macaulay rings are an important class of rings in commutative algebra and algebraic geometry. They have nice properties related to dimension theory and are characterized by the existence of regular sequences of maximal length.

```python
def depth(ring, ideal):
    # Simplified depth calculation (not accurate for all rings)
    return len(ideal)

def dim(ring):
    # Simplified dimension calculation (not accurate for all rings)
    return len(ring.elements).bit_length() - 1

def is_cohen_macaulay(ring, ideal):
    return depth(ring, ideal) == dim(ring)

# Example: Check if Z6 with ideal {0, 3} is Cohen-Macaulay
Z6 = CommutativeRing(range(6))
ideal = {0, 3}
print(f"Is Z6 with ideal {ideal} Cohen-Macaulay? {is_cohen_macaulay(Z6, ideal)}")
```

Slide 14: Real-life Example: Error-Correcting Codes

Commutative algebra plays a crucial role in coding theory, particularly in the design of error-correcting codes. Here's an example of a simple error-detecting code using polynomial rings over finite fields.

```python
from sympy import GF, Poly, symbols

def crc_remainder(message, generator):
    return Poly(message, x) % generator

# Example: Cyclic Redundancy Check (CRC)
x = symbols('x')
GF256 = GF(2**8)
message = GF256([1, 0, 1, 1, 0, 1])  # 101101 in binary
generator = Poly(x**3 + x + 1, x, domain=GF256)

remainder = crc_remainder(message, generator)
encoded_message = message + list(remainder.all_coeffs())

print("Original message:", message)
print("Encoded message:", encoded_message)
```

Slide 15: Real-life Example: Cryptography

Commutative algebra is fundamental in modern cryptography. Here's a simple implementation of the RSA algorithm, which relies on properties of prime numbers and modular arithmetic.

```python
import random

def generate_keypair(p, q):
    n = p * q
    phi = (p - 1) * (q - 1)
    e = random.randrange(1, phi)
    while gcd(e, phi) != 1:
        e = random.randrange(1, phi)
    d = mod_inverse(e, phi)
    return ((e, n), (d, n))

def encrypt(pk, plaintext):
    e, n = pk
    return [pow(ord(char), e, n) for char in plaintext]

def decrypt(pk, ciphertext):
    d, n = pk
    return ''.join([chr(pow(char, d, n)) for char in ciphertext])

def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def mod_inverse(a, m):
    for i in range(1, m):
        if (a * i) % m == 1:
            return i
    return None

# Example usage
p, q = 61, 53  # Small primes for demonstration
public, private = generate_keypair(p, q)
message = "Hello, RSA!"
encrypted = encrypt(public, message)
decrypted = decrypt(private, encrypted)

print("Original message:", message)
print("Encrypted message:", encrypted)
print("Decrypted message:", decrypted)
```

Slide 16: Additional Resources

For further study in Commutative Algebra, consider the following resources:

1. ArXiv.org: "Introduction to Commutative Algebra" by M. F. Atiyah and I. G. Macdonald URL: [https://arxiv.org/abs/alg-geom/9502002](https://arxiv.org/abs/alg-geom/9502002)
2. ArXiv.org: "Computational Commutative Algebra" by Martin Kreuzer and Lorenzo Robbiano URL: [https://arxiv.org/abs/math/0204078](https://arxiv.org/abs/math/0204078)
3. ArXiv.org: "Commutative Algebra: Constructive Methods" by Henri Lombardi and Claude Quitté URL: [https://arxiv.org/abs/1605.04832](https://arxiv.org/abs/1605.04832)

These papers provide in-depth coverage of various aspects of commutative algebra and its applications.

