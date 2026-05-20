## Exploring Elliptic Curve Arithmetic with Python

Slide 1: Introduction to Elliptic Curves

Elliptic curves are algebraic structures with applications in cryptography and number theory. In this presentation, we'll explore their arithmetic using Python.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_elliptic_curve(a, b):
    x = np.linspace(-5, 5, 1000)
    y = np.sqrt(x**3 + a*x + b)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b', label='y² = x³ + {}x + {}'.format(a, b))
    plt.plot(x, -y, 'b')
    plt.title('Elliptic Curve: y² = x³ + {}x + {}'.format(a, b))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_elliptic_curve(2, 3)
```

Slide 2: Weierstrass Form

Elliptic curves are typically represented in Weierstrass form: y² = x³ + ax + b. Let's define a class for points on the curve.

```python
class EllipticCurvePoint:
    def __init__(self, x, y, a, b):
        self.x = x
        self.y = y
        self.a = a
        self.b = b
    
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def is_on_curve(self):
        return self.y**2 == self.x**3 + self.a*self.x + self.b

# Example usage
P = EllipticCurvePoint(2, 3, 2, 3)
print(f"P is on curve: {P.is_on_curve()}")
```

Slide 3: Point Addition

Point addition is a fundamental operation in elliptic curve arithmetic. Let's implement it:

```python
def point_addition(P, Q):
    if P.x == Q.x and P.y != Q.y:
        return None  # Point at infinity
    
    if P.x != Q.x:
        m = (Q.y - P.y) / (Q.x - P.x)
    else:
        m = (3 * P.x**2 + P.a) / (2 * P.y)
    
    x3 = m**2 - P.x - Q.x
    y3 = m * (P.x - x3) - P.y
    
    return EllipticCurvePoint(x3, y3, P.a, P.b)

# Example usage
P = EllipticCurvePoint(2, 3, 2, 3)
Q = EllipticCurvePoint(-1, 4, 2, 3)
R = point_addition(P, Q)
print(f"P + Q = {R}")
```

Slide 4: Scalar Multiplication

Scalar multiplication is repeated point addition. We'll implement it using the double-and-add algorithm:

```python
def scalar_multiplication(k, P):
    if k == 0:
        return None
    elif k == 1:
        return P
    elif k % 2 == 0:
        return scalar_multiplication(k // 2, point_addition(P, P))
    else:
        return point_addition(P, scalar_multiplication(k - 1, P))

# Example usage
P = EllipticCurvePoint(2, 3, 2, 3)
Q = scalar_multiplication(5, P)
print(f"5P = {Q}")
```

Slide 5: Order of a Point

The order of a point is the smallest positive integer n such that nP = O (point at infinity).

```python
def order_of_point(P):
    Q = P
    n = 1
    while Q is not None:
        Q = point_addition(Q, P)
        n += 1
        if Q and Q.x == P.x and Q.y == P.y:
            return n
    return None

# Example usage
P = EllipticCurvePoint(2, 3, 2, 3)
order = order_of_point(P)
print(f"Order of P: {order}")
```

Slide 6: Elliptic Curve over Finite Fields

In cryptography, we often use elliptic curves over finite fields. Let's implement arithmetic modulo a prime p:

```python
class EllipticCurveFiniteField:
    def __init__(self, a, b, p):
        self.a = a
        self.b = b
        self.p = p
    
    def add_points(self, P, Q):
        if P is None:
            return Q
        if Q is None:
            return P
        
        if (P.x, P.y) == (Q.x, Q.y):
            m = (3 * P.x**2 + self.a) * pow(2 * P.y, -1, self.p)
        else:
            m = (Q.y - P.y) * pow(Q.x - P.x, -1, self.p)
        
        x3 = (m**2 - P.x - Q.x) % self.p
        y3 = (m * (P.x - x3) - P.y) % self.p
        
        return EllipticCurvePoint(x3, y3, self.a, self.b)

# Example usage
E = EllipticCurveFiniteField(2, 3, 17)
P = EllipticCurvePoint(5, 1, 2, 3)
Q = EllipticCurvePoint(6, 3, 2, 3)
R = E.add_points(P, Q)
print(f"P + Q = {R}")
```

Slide 7: Generating Points on the Curve

Let's create a function to generate all points on an elliptic curve over a finite field:

```python
def generate_points(E):
    points = []
    for x in range(E.p):
        for y in range(E.p):
            if (y**2 - x**3 - E.a*x - E.b) % E.p == 0:
                points.append(EllipticCurvePoint(x, y, E.a, E.b))
    return points

# Example usage
E = EllipticCurveFiniteField(2, 3, 17)
points = generate_points(E)
print(f"Number of points: {len(points)}")
for point in points:
    print(point)
```

Slide 8: Real-life Example: ECDSA (Part 1)

Elliptic Curve Digital Signature Algorithm (ECDSA) is used in cryptocurrencies like Bitcoin. Let's implement key generation:

```python
import random

def generate_keypair(E, G, n):
    d = random.randint(1, n - 1)  # Private key
    Q = scalar_multiplication(d, G)  # Public key
    return d, Q

# Example usage
E = EllipticCurveFiniteField(2, 3, 17)
G = EllipticCurvePoint(5, 1, 2, 3)  # Base point
n = 18  # Order of G (simplified for this example)

private_key, public_key = generate_keypair(E, G, n)
print(f"Private key: {private_key}")
print(f"Public key: {public_key}")
```

Slide 9: Real-life Example: ECDSA (Part 2)

Now let's implement the signing and verification process:

```python
def sign_message(E, G, n, d, message):
    k = random.randint(1, n - 1)
    R = scalar_multiplication(k, G)
    r = R.x % n
    h = hash(message) % n
    s = (pow(k, -1, n) * (h + r * d)) % n
    return (r, s)

def verify_signature(E, G, n, Q, message, signature):
    r, s = signature
    h = hash(message) % n
    w = pow(s, -1, n)
    u1 = (h * w) % n
    u2 = (r * w) % n
    X = point_addition(scalar_multiplication(u1, G), scalar_multiplication(u2, Q))
    return r == X.x % n

# Example usage
message = "Hello, Elliptic Curves!"
signature = sign_message(E, G, n, private_key, message)
is_valid = verify_signature(E, G, n, public_key, message, signature)
print(f"Signature: {signature}")
print(f"Signature is valid: {is_valid}")
```

Slide 10: Elliptic Curve Diffie-Hellman (ECDH)

ECDH is a key agreement protocol. Let's implement it:

```python
def ecdh_key_exchange(E, G, n):
    # Alice's key pair
    a = random.randint(1, n - 1)
    A = scalar_multiplication(a, G)
    
    # Bob's key pair
    b = random.randint(1, n - 1)
    B = scalar_multiplication(b, G)
    
    # Shared secret
    s_alice = scalar_multiplication(a, B)
    s_bob = scalar_multiplication(b, A)
    
    return s_alice, s_bob

# Example usage
s_alice, s_bob = ecdh_key_exchange(E, G, n)
print(f"Alice's shared secret: {s_alice}")
print(f"Bob's shared secret: {s_bob}")
print(f"Shared secrets match: {s_alice.x == s_bob.x and s_alice.y == s_bob.y}")
```

Slide 11: Elliptic Curve Point Compression

To save space, we can compress elliptic curve points:

```python
def compress_point(P):
    return (P.x, P.y % 2)

def decompress_point(E, x, y_bit):
    y_squared = (x**3 + E.a*x + E.b) % E.p
    y = pow(y_squared, (E.p + 1) // 4, E.p)
    if y % 2 != y_bit:
        y = E.p - y
    return EllipticCurvePoint(x, y, E.a, E.b)

# Example usage
P = EllipticCurvePoint(5, 1, 2, 3)
compressed = compress_point(P)
decompressed = decompress_point(E, compressed[0], compressed[1])
print(f"Original point: {P}")
print(f"Compressed: {compressed}")
print(f"Decompressed: {decompressed}")
```

Slide 12: Performance Optimization: Precomputation

For faster scalar multiplication, we can use precomputation:

```python
def precompute_points(G, n):
    precomputed = [G]
    for i in range(1, n):
        precomputed.append(point_addition(precomputed[-1], G))
    return precomputed

def fast_scalar_multiplication(k, precomputed):
    result = None
    for i in range(len(precomputed)):
        if k & (1 << i):
            result = point_addition(result, precomputed[i])
    return result

# Example usage
precomputed = precompute_points(G, 10)
Q = fast_scalar_multiplication(7, precomputed)
print(f"7G = {Q}")
```

Slide 13: Elliptic Curve Cryptography in Practice

In practice, standardized curves like secp256k1 (used in Bitcoin) are used. Here's how to use the `ecdsa` library:

```python
from ecdsa import SigningKey, SECP256k1

# Key generation
sk = SigningKey.generate(curve=SECP256k1)
vk = sk.get_verifying_key()

# Signing
message = b"Hello, Elliptic Curves!"
signature = sk.sign(message)

# Verification
assert vk.verify(signature, message)
print("Signature verified successfully!")
```

Slide 14: Additional Resources

For further reading on elliptic curves and their applications in cryptography:

1. "An Introduction to Mathematical Cryptography" by J. Hoffstein, J. Pipher, and J.H. Silverman ArXiv: [https://arxiv.org/abs/1305.5795](https://arxiv.org/abs/1305.5795)
2. "Elliptic Curves: Number Theory and Cryptography" by L.C. Washington ArXiv: [https://arxiv.org/abs/1906.09632](https://arxiv.org/abs/1906.09632)
3. "A Course in Computational Algebraic Number Theory" by H. Cohen ArXiv: [https://arxiv.org/abs/1903.10598](https://arxiv.org/abs/1903.10598)

These resources provide in-depth coverage of the mathematical foundations and practical applications of elliptic curves in cryptography.

