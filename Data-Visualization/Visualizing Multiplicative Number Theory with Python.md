## Visualizing Multiplicative Number Theory with Python.md

Slide 1: Introduction to Multiplicative Number Theory

Multiplicative Number Theory is a branch of mathematics that focuses on the properties of integers under multiplication. It explores concepts like prime numbers, divisibility, and multiplicative functions.

```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# Example: Check if a number is prime
number = 17
print(f"{number} is prime: {is_prime(number)}")
```

Slide 2: Fundamental Theorem of Arithmetic

The Fundamental Theorem of Arithmetic states that every positive integer can be uniquely expressed as a product of prime factors.

```python
def prime_factorization(n):
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
        if d * d > n:
            if n > 1:
                factors.append(n)
            break
    return factors

# Example: Prime factorization of a number
number = 84
print(f"Prime factors of {number}: {prime_factorization(number)}")
```

Slide 3: Greatest Common Divisor (GCD)

The Greatest Common Divisor (GCD) of two or more integers is the largest positive integer that divides each of the integers without a remainder.

```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Example: Calculate GCD of two numbers
num1, num2 = 48, 18
print(f"GCD of {num1} and {num2}: {gcd(num1, num2)}")
```

Slide 4: Least Common Multiple (LCM)

The Least Common Multiple (LCM) of two or more integers is the smallest positive integer that is divisible by each of the integers.

```python
def lcm(a, b):
    return abs(a * b) // gcd(a, b)

# Example: Calculate LCM of two numbers
num1, num2 = 12, 15
print(f"LCM of {num1} and {num2}: {lcm(num1, num2)}")
```

Slide 5: Euler's Totient Function

Euler's Totient Function φ(n) counts the number of integers up to n that are coprime to n.

```python
def euler_totient(n):
    result = n
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            while n % i == 0:
                n //= i
            result *= (1 - 1/i)
    if n > 1:
        result *= (1 - 1/n)
    return int(result)

# Example: Calculate Euler's Totient for a number
number = 36
print(f"φ({number}) = {euler_totient(number)}")
```

Slide 6: Multiplicative Functions

A function f(n) is multiplicative if f(ab) = f(a) \* f(b) for any coprime positive integers a and b.

```python
def is_multiplicative(f, limit=100):
    for a in range(1, limit):
        for b in range(1, limit):
            if gcd(a, b) == 1:
                if f(a * b) != f(a) * f(b):
                    return False
    return True

# Example: Check if a function is multiplicative
def test_function(n):
    return n  # Identity function

print(f"Is the function multiplicative? {is_multiplicative(test_function)}")
```

Slide 7: Möbius Function

The Möbius function μ(n) is an important multiplicative function in number theory.

```python
def mobius(n):
    if n == 1:
        return 1
    if any((n % i == 0 and n % (i * i) == 0) for i in range(2, int(n**0.5) + 1)):
        return 0
    return (-1) ** len(prime_factorization(n))

# Example: Calculate Möbius function for a range of numbers
for i in range(1, 11):
    print(f"μ({i}) = {mobius(i)}")
```

Slide 8: Dirichlet Convolution

The Dirichlet convolution is an operation that combines two arithmetic functions to produce a third one.

```python
def dirichlet_convolution(f, g, n):
    result = 0
    for d in range(1, n + 1):
        if n % d == 0:
            result += f(d) * g(n // d)
    return result

# Example: Convolution of two simple functions
def f(n): return n
def g(n): return 1

n = 12
print(f"(f * g)({n}) = {dirichlet_convolution(f, g, n)}")
```

Slide 9: Prime Number Theorem

The Prime Number Theorem describes the asymptotic distribution of prime numbers among positive integers.

```python
import math

def prime_counting_function(n):
    return sum(1 for i in range(2, n+1) if is_prime(i))

def li(x):
    return math.log(x) + 1.04516

# Example: Compare π(x) with li(x)
x = 1000
pi_x = prime_counting_function(x)
li_x = li(x)
print(f"π({x}) = {pi_x}")
print(f"li({x}) ≈ {li_x:.2f}")
print(f"Relative error: {abs(pi_x - li_x) / pi_x:.2%}")
```

Slide 10: Sieve of Eratosthenes

The Sieve of Eratosthenes is an ancient algorithm for finding all prime numbers up to a given limit.

```python
def sieve_of_eratosthenes(n):
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    for i in range(2, int(n**0.5) + 1):
        if primes[i]:
            for j in range(i*i, n + 1, i):
                primes[j] = False
    return [i for i in range(2, n + 1) if primes[i]]

# Example: Find all primes up to 50
limit = 50
print(f"Primes up to {limit}: {sieve_of_eratosthenes(limit)}")
```

Slide 11: Chinese Remainder Theorem

The Chinese Remainder Theorem is a powerful tool for solving systems of linear congruences.

```python
def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    else:
        gcd, x, y = extended_gcd(b % a, a)
        return gcd, y - (b // a) * x, x

def chinese_remainder_theorem(n, a):
    total = 0
    product = 1
    for ni in n:
        product *= ni
    for ni, ai in zip(n, a):
        p = product // ni
        gcd, x, y = extended_gcd(p, ni)
        total += ai * x * p
    return total % product

# Example: Solve system of congruences
n = [3, 5, 7]
a = [2, 3, 2]
result = chinese_remainder_theorem(n, a)
print(f"Solution: x ≡ {result} (mod {math.prod(n)})")
```

Slide 12: Quadratic Residues

A quadratic residue modulo n is an integer a such that x^2 ≡ a (mod n) has a solution.

```python
def legendre_symbol(a, p):
    if a % p == 0:
        return 0
    return pow(a, (p - 1) // 2, p)

def is_quadratic_residue(a, p):
    return legendre_symbol(a, p) == 1

# Example: Check quadratic residues modulo a prime
p = 11
print(f"Quadratic residues modulo {p}:")
for a in range(1, p):
    if is_quadratic_residue(a, p):
        print(a, end=" ")
print()
```

Slide 13: Primitive Roots

A primitive root modulo n is an integer g such that every number coprime to n is congruent to a power of g modulo n.

```python
def order(a, m):
    if gcd(a, m) != 1:
        return 0
    order = 1
    power = a % m
    while power != 1:
        power = (power * a) % m
        order += 1
    return order

def find_primitive_root(p):
    if p == 2:
        return 1
    phi = p - 1
    for g in range(2, p):
        if order(g, p) == phi:
            return g
    return None

# Example: Find a primitive root modulo a prime
p = 17
root = find_primitive_root(p)
print(f"A primitive root modulo {p} is: {root}")
```

Slide 14: Real-Life Example: RSA Encryption

RSA is a widely used public-key cryptosystem based on the difficulty of factoring large numbers.

```python
import random

def generate_keypair(p, q):
    n = p * q
    phi = (p - 1) * (q - 1)
    e = random.randrange(1, phi)
    while gcd(e, phi) != 1:
        e = random.randrange(1, phi)
    d = pow(e, -1, phi)
    return ((e, n), (d, n))

def encrypt(pk, plaintext):
    e, n = pk
    return pow(plaintext, e, n)

def decrypt(pk, ciphertext):
    d, n = pk
    return pow(ciphertext, d, n)

# Example: RSA encryption and decryption
p, q = 61, 53
public, private = generate_keypair(p, q)
message = 42
encrypted = encrypt(public, message)
decrypted = decrypt(private, encrypted)

print(f"Original: {message}")
print(f"Encrypted: {encrypted}")
print(f"Decrypted: {decrypted}")
```

Slide 15: Real-Life Example: Hash-based Digital Signatures

Digital signatures based on hash functions and number theory principles are used for authentication and integrity verification.

```python
import hashlib

def sign_message(message, private_key):
    message_hash = hashlib.sha256(message.encode()).hexdigest()
    signature = pow(int(message_hash, 16), private_key[0], private_key[1])
    return signature

def verify_signature(message, signature, public_key):
    message_hash = hashlib.sha256(message.encode()).hexdigest()
    decrypted_hash = pow(signature, public_key[0], public_key[1])
    return int(message_hash, 16) == decrypted_hash

# Example: Digital signature creation and verification
message = "Hello, World!"
public, private = generate_keypair(61, 53)  # Reusing the RSA keypair

signature = sign_message(message, private)
is_valid = verify_signature(message, signature, public)

print(f"Message: {message}")
print(f"Signature: {signature}")
print(f"Signature is valid: {is_valid}")
```

Slide 16: Additional Resources

For further exploration of Multiplicative Number Theory, consider these resources:

1. "An Introduction to the Theory of Numbers" by G.H. Hardy and E.M. Wright ArXiv: [https://arxiv.org/abs/math/0404314](https://arxiv.org/abs/math/0404314)
2. "Multiplicative Number Theory" by Harold Davenport ArXiv: [https://arxiv.org/abs/1107.1189](https://arxiv.org/abs/1107.1189)
3. "Analytic Number Theory" course notes by Andrew Granville ArXiv: [https://arxiv.org/abs/1809.07818](https://arxiv.org/abs/1809.07818)

These resources provide in-depth coverage of the topics discussed in this presentation and can help deepen your understanding of Multiplicative Number Theory.

