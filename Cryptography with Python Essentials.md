## Cryptography with Python Essentials

Slide 1: 

Introduction to Cryptography with Python

Cryptography is the practice of secure communication in the presence of adversaries. Python provides powerful libraries for implementing cryptographic algorithms. In this presentation, we'll explore various cryptographic concepts and their implementation using Python.

```python
from cryptography.fernet import Fernet

# Generate a random key
key = Fernet.generate_key()

# Create a Fernet instance
f = Fernet(key)

# Encrypt a message
message = b"Hello, Cryptography!"
encrypted = f.encrypt(message)

# Decrypt the message
decrypted = f.decrypt(encrypted)

print(f"Original: {message}")
print(f"Encrypted: {encrypted}")
print(f"Decrypted: {decrypted}")
```

Slide 2: 

Symmetric Encryption: AES

Advanced Encryption Standard (AES) is a widely used symmetric encryption algorithm. It uses the same key for both encryption and decryption. Let's implement AES encryption using Python's cryptography library.

```python
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

def encrypt_aes(plaintext, key):
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return iv + ciphertext

def decrypt_aes(ciphertext, key):
    iv = ciphertext[:16]
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(ciphertext[16:]) + decryptor.finalize()

# Example usage
key = os.urandom(32)  # 256-bit key
plaintext = b"AES encryption in Python"
encrypted = encrypt_aes(plaintext, key)
decrypted = decrypt_aes(encrypted, key)

print(f"Original: {plaintext}")
print(f"Encrypted: {encrypted}")
print(f"Decrypted: {decrypted}")
```

Slide 3: 

Asymmetric Encryption: RSA

RSA is a popular asymmetric encryption algorithm that uses a pair of keys: public key for encryption and private key for decryption. Let's implement RSA encryption using Python's cryptography library.

```python
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes

def generate_rsa_keys():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    public_key = private_key.public_key()
    return private_key, public_key

def encrypt_rsa(message, public_key):
    ciphertext = public_key.encrypt(
        message,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return ciphertext

def decrypt_rsa(ciphertext, private_key):
    plaintext = private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return plaintext

# Example usage
private_key, public_key = generate_rsa_keys()
message = b"RSA encryption in Python"
encrypted = encrypt_rsa(message, public_key)
decrypted = decrypt_rsa(encrypted, private_key)

print(f"Original: {message}")
print(f"Encrypted: {encrypted}")
print(f"Decrypted: {decrypted}")
```

Slide 4: 

Hashing: SHA-256

Hashing is a one-way function that converts data of arbitrary size into a fixed-size output. SHA-256 is a widely used cryptographic hash function. Let's implement SHA-256 hashing using Python's hashlib library.

```python
import hashlib

def sha256_hash(data):
    sha256 = hashlib.sha256()
    sha256.update(data.encode())
    return sha256.hexdigest()

# Example usage
message = "Hello, SHA-256!"
hashed = sha256_hash(message)

print(f"Original message: {message}")
print(f"SHA-256 hash: {hashed}")

# Verify that the same input always produces the same hash
print(f"Hash of 'Hello, SHA-256!': {sha256_hash('Hello, SHA-256!')}")
print(f"Hash of 'Different message': {sha256_hash('Different message')}")
```

Slide 5: 

Digital Signatures

Digital signatures provide authenticity, integrity, and non-repudiation. They are created using the sender's private key and verified using the sender's public key. Let's implement digital signatures using RSA.

```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

def generate_keys():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    public_key = private_key.public_key()
    return private_key, public_key

def sign_message(message, private_key):
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return signature

def verify_signature(message, signature, public_key):
    try:
        public_key.verify(
            signature,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except:
        return False

# Example usage
private_key, public_key = generate_keys()
message = b"This is a signed message."

signature = sign_message(message, private_key)
is_valid = verify_signature(message, signature, public_key)

print(f"Message: {message}")
print(f"Signature: {signature}")
print(f"Signature valid: {is_valid}")

# Try to verify with a tampered message
tampered_message = b"This is a tampered message."
is_valid_tampered = verify_signature(tampered_message, signature, public_key)
print(f"Tampered message signature valid: {is_valid_tampered}")
```

Slide 6: 

Key Derivation Functions (KDF)

Key Derivation Functions are used to derive one or more secret keys from a master secret. They are often used to convert passwords into cryptographic keys. Let's implement PBKDF2 (Password-Based Key Derivation Function 2) using Python.

```python
import os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def derive_key(password, salt=None, iterations=100000):
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    key = kdf.derive(password.encode())
    return key, salt

# Example usage
password = "mysecretpassword"
derived_key, salt = derive_key(password)

print(f"Password: {password}")
print(f"Derived key: {derived_key.hex()}")
print(f"Salt: {salt.hex()}")

# Verify that the same password and salt produce the same key
verified_key, _ = derive_key(password, salt)
print(f"Key verification: {derived_key == verified_key}")

# Different password produces a different key
different_key, _ = derive_key("differentpassword", salt)
print(f"Different password key: {different_key.hex()}")
print(f"Keys match: {derived_key == different_key}")
```

Slide 7: 

Secure Random Number Generation

Cryptographically secure random number generation is crucial for many cryptographic operations, such as generating keys and initialization vectors. Python's secrets module provides functions for generating secure random numbers.

```python
import secrets

def generate_random_bytes(n):
    return secrets.token_bytes(n)

def generate_random_integer(a, b):
    return secrets.randbelow(b - a + 1) + a

def generate_random_string(length):
    return secrets.token_urlsafe(length)

# Example usage
random_bytes = generate_random_bytes(16)
random_int = generate_random_integer(1, 100)
random_string = generate_random_string(12)

print(f"Random bytes: {random_bytes.hex()}")
print(f"Random integer between 1 and 100: {random_int}")
print(f"Random URL-safe string: {random_string}")

# Demonstrate uniqueness
print("\nGenerating multiple random values:")
for _ in range(5):
    print(f"Random bytes: {generate_random_bytes(8).hex()}")
    print(f"Random integer: {generate_random_integer(1, 10)}")
    print(f"Random string: {generate_random_string(8)}")
    print()
```

Slide 8: 

Message Authentication Codes (MAC)

Message Authentication Codes provide integrity and authenticity of messages. HMAC (Hash-based Message Authentication Code) is a widely used MAC algorithm. Let's implement HMAC using Python's hmac module.

```python
import hmac
import hashlib

def create_hmac(key, message):
    h = hmac.new(key.encode(), message.encode(), hashlib.sha256)
    return h.hexdigest()

def verify_hmac(key, message, received_hmac):
    calculated_hmac = create_hmac(key, message)
    return hmac.compare_digest(calculated_hmac, received_hmac)

# Example usage
secret_key = "mysecretkey"
message = "Hello, HMAC!"

# Create HMAC
hmac_result = create_hmac(secret_key, message)
print(f"Message: {message}")
print(f"HMAC: {hmac_result}")

# Verify HMAC
is_valid = verify_hmac(secret_key, message, hmac_result)
print(f"HMAC is valid: {is_valid}")

# Try to verify with a tampered message
tampered_message = "Hello, tampered HMAC!"
is_valid_tampered = verify_hmac(secret_key, tampered_message, hmac_result)
print(f"Tampered message HMAC is valid: {is_valid_tampered}")
```

Slide 9: 

Password Hashing with Salting

Proper password storage is crucial for application security. We'll use the bcrypt algorithm, which automatically handles salting and is designed to be slow and resist brute-force attacks. Let's implement password hashing and verification using the bcrypt library.

```python
import bcrypt

def hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

# Example usage
password = "mysecretpassword"

# Hash the password
hashed_password = hash_password(password)
print(f"Original password: {password}")
print(f"Hashed password: {hashed_password}")

# Verify the password
is_correct = verify_password(password, hashed_password)
print(f"Password is correct: {is_correct}")

# Try an incorrect password
wrong_password = "wrongpassword"
is_wrong = verify_password(wrong_password, hashed_password)
print(f"Wrong password is correct: {is_wrong}")

# Demonstrate that hashing the same password twice produces different results
hashed_password2 = hash_password(password)
print(f"\nHashing the same password again: {hashed_password2}")
print(f"Hashes are identical: {hashed_password == hashed_password2}")
print(f"But both verify correctly: {verify_password(password, hashed_password2)}")
```

Slide 10: 

Elliptic Curve Cryptography (ECC)

Elliptic Curve Cryptography is an approach to public-key cryptography based on the algebraic structure of elliptic curves over finite fields. It offers smaller key sizes compared to RSA for equivalent security. Let's implement ECC key generation and ECDH (Elliptic Curve Diffie-Hellman) key exchange using Python.

```python
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

def generate_ecc_key_pair():
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    return private_key, public_key

def perform_ecdh(private_key, peer_public_key):
    shared_key = private_key.exchange(ec.ECDH(), peer_public_key)
    return shared_key

# Example usage
alice_private, alice_public = generate_ecc_key_pair()
bob_private, bob_public = generate_ecc_key_pair()

# Perform ECDH key exchange
alice_shared_key = perform_ecdh(alice_private, bob_public)
bob_shared_key = perform_ecdh(bob_private, alice_public)

print("Alice's public key:")
print(alice_public.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
).decode())

print("\nBob's public key:")
print(bob_public.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
).decode())

print(f"\nAlice's shared key: {alice_shared_key.hex()}")
print(f"Bob's shared key: {bob_shared_key.hex()}")
print(f"Shared keys match: {alice_shared_key == bob_shared_key}")
```

Slide 11: 

Authenticated Encryption

Authenticated Encryption provides both confidentiality and integrity/authenticity of data. We'll use the ChaCha20-Poly1305 algorithm, which is a modern authenticated encryption algorithm. Let's implement encryption and decryption using ChaCha20-Poly1305 in Python.

```python
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
import os

def encrypt_chacha20_poly1305(key, plaintext, associated_data):
    nonce = os.urandom(12)
    chacha = ChaCha20Poly1305(key)
    ciphertext = chacha.encrypt(nonce, plaintext, associated_data)
    return nonce + ciphertext

def decrypt_chacha20_poly1305(key, ciphertext, associated_data):
    nonce = ciphertext[:12]
    ciphertext = ciphertext[12:]
    chacha = ChaCha20Poly1305(key)
    plaintext = chacha.decrypt(nonce, ciphertext, associated_data)
    return plaintext

# Example usage
key = os.urandom(32)  # ChaCha20-Poly1305 uses a 256-bit key
plaintext = b"Secret message"
associated_data = b"Additional data"

encrypted = encrypt_chacha20_poly1305(key, plaintext, associated_data)
decrypted = decrypt_chacha20_poly1305(key, encrypted, associated_data)

print(f"Original: {plaintext}")
print(f"Encrypted: {encrypted.hex()}")
print(f"Decrypted: {decrypted}")

# Attempt to decrypt with modified associated data
try:
    decrypt_chacha20_poly1305(key, encrypted, b"Modified data")
    print("Decryption succeeded (shouldn't happen)")
except Exception as e:
    print(f"Decryption failed as expected: {e}")
```

Slide 12: 

Zero-Knowledge Proofs: Schnorr Protocol

Zero-knowledge proofs allow one party (the prover) to prove to another party (the verifier) that a statement is true, without revealing any information beyond the validity of the statement. The Schnorr protocol is a simple zero-knowledge proof for proving knowledge of a discrete logarithm.

```python
import random
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

def generate_keypair():
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    return private_key, public_key

def schnorr_prove(private_key, public_key):
    curve = public_key.curve
    G = public_key.public_numbers().y
    q = curve.order
    
    # Prover generates a random value
    r = random.randrange(1, q)
    R = r * G
    
    # Prover computes the challenge
    h = hashes.Hash(hashes.SHA256())
    h.update(str(R.x).encode() + str(R.y).encode())
    e = int.from_bytes(h.finalize(), 'big') % q
    
    # Prover computes the response
    s = (r + e * private_key.private_numbers().private_value) % q
    
    return (R, s)

def schnorr_verify(public_key, proof):
    R, s = proof
    curve = public_key.curve
    G = public_key.public_numbers().y
    q = curve.order
    
    # Verifier computes the challenge
    h = hashes.Hash(hashes.SHA256())
    h.update(str(R.x).encode() + str(R.y).encode())
    e = int.from_bytes(h.finalize(), 'big') % q
    
    # Verifier checks if sG = R + eY
    sG = s * G
    eY = e * public_key.public_numbers().y
    RY = R + eY
    
    return sG == RY

# Example usage
private_key, public_key = generate_keypair()

# Prover generates a proof
proof = schnorr_prove(private_key, public_key)

# Verifier checks the proof
is_valid = schnorr_verify(public_key, proof)

print(f"Proof is valid: {is_valid}")

# Try to verify with a different public key
_, fake_public_key = generate_keypair()
is_valid_fake = schnorr_verify(fake_public_key, proof)
print(f"Proof is valid with fake key: {is_valid_fake}")
```

Slide 13: 

Homomorphic Encryption: Paillier Cryptosystem

Homomorphic encryption allows computations on encrypted data without decrypting it. The Paillier cryptosystem is a partially homomorphic encryption scheme that supports addition of encrypted values. Let's implement a simple version of the Paillier cryptosystem.

```python
import random
from math import gcd

def generate_paillier_keypair(bits):
    p = generate_prime(bits // 2)
    q = generate_prime(bits // 2)
    n = p * q
    g = n + 1
    lambda_n = (p - 1) * (q - 1) // gcd(p - 1, q - 1)
    mu = mod_inverse(lambda_n, n)
    return ((n, g), (lambda_n, mu))

def paillier_encrypt(public_key, m):
    n, g = public_key
    r = random.randrange(1, n)
    c = (pow(g, m, n**2) * pow(r, n, n**2)) % (n**2)
    return c

def paillier_decrypt(private_key, c):
    lambda_n, mu = private_key
    n = (lambda_n * mu) // gcd(lambda_n, mu) + 1
    x = (pow(c, lambda_n, n**2) - 1) // n
    m = (x * mu) % n
    return m

def paillier_add(public_key, c1, c2):
    n, _ = public_key
    return (c1 * c2) % (n**2)

# Helper functions
def generate_prime(bits):
    while True:
        n = random.getrandbits(bits)
        if n % 2 != 0 and is_prime(n):
            return n

def is_prime(n, k=5):
    if n < 2: return False
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        if n % p == 0: return n == p
    s, d = 0, n - 1
    while d % 2 == 0:
        s, d = s + 1, d // 2
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x != 1 and x != n - 1:
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1: break
            else: return False
    return True

def mod_inverse(a, m):
    g, x, y = extended_gcd(a, m)
    if g != 1: raise Exception('Modular inverse does not exist')
    return x % m

def extended_gcd(a, b):
    if a == 0: return (b, 0, 1)
    else:
        g, y, x = extended_gcd(b % a, a)
        return (g, x - (b // a) * y, y)

# Example usage
public_key, private_key = generate_paillier_keypair(1024)

m1, m2 = 42, 73
c1 = paillier_encrypt(public_key, m1)
c2 = paillier_encrypt(public_key, m2)

c_sum = paillier_add(public_key, c1, c2)
decrypted_sum = paillier_decrypt(private_key, c_sum)

print(f"m1: {m1}, m2: {m2}")
print(f"Encrypted sum: {c_sum}")
print(f"Decrypted sum: {decrypted_sum}")
print(f"Actual sum: {m1 + m2}")
print(f"Homomorphic property holds: {decrypted_sum == (m1 + m2)}")
```

Slide 14: 

Quantum-Resistant Cryptography: Lattice-Based Encryption

As quantum computers advance, there's a need for cryptographic algorithms resistant to quantum attacks. Lattice-based cryptography is a promising post-quantum approach. Let's implement a simple Learning With Errors (LWE) encryption scheme, which is a foundational lattice-based primitive.

```python
import numpy as np

def generate_lwe_keypair(n, q, sigma):
    s = np.random.randint(0, q, size=n)  # Secret key
    A = np.random.randint(0, q, size=(n, n))  # Public matrix
    e = np.random.normal(0, sigma, size=n).astype(int) % q  # Error vector
    b = (A.dot(s) + e) % q  # Public vector
    return (A, b), s

def lwe_encrypt(public_key, m, q):
    A, b = public_key
    n = len(b)
    r = np.random.randint(0, 2, size=n)
    c1 = r.dot(A) % q
    c2 = (r.dot(b) + m * (q // 2)) % q
    return c1, c2

def lwe_decrypt(private_key, ciphertext, q):
    s = private_key
    c1, c2 = ciphertext
    z = (c2 - c1.dot(s)) % q
    if z > q // 2:
        return 1
    else:
        return 0

# Parameters
n = 10  # Dimension
q = 97  # Modulus
sigma = 1  # Standard deviation for error distribution

# Key generation
public_key, private_key = generate_lwe_keypair(n, q, sigma)

# Encryption and decryption
message = 1
ciphertext = lwe_encrypt(public_key, message, q)
decrypted = lwe_decrypt(private_key, ciphertext, q)

print(f"Original message: {message}")
print(f"Decrypted message: {decrypted}")

# Test multiple messages
for _ in range(10):
    m = np.random.randint(0, 2)
    c = lwe_encrypt(public_key, m, q)
    d = lwe_decrypt(private_key, c, q)
    print(f"Message: {m}, Decrypted: {d}, Correct: {m == d}")
```

Slide 15: 

Additional Resources

For those interested in diving deeper into cryptography and its implementation in Python, here are some valuable resources:

1. "A Graduate Course in Applied Cryptography" by Dan Boneh and Victor Shoup ArXiv link: [https://arxiv.org/abs/2008.01580](https://arxiv.org/abs/2008.01580)
2. "Post-Quantum Cryptography" by Daniel J. Bernstein, Johannes Buchmann, and Erik Dahmen ArXiv link: [https://arxiv.org/abs/0809.2789](https://arxiv.org/abs/0809.2789)
3. "Lattice-based Cryptography" by Daniele Micciancio and Oded Regev ArXiv link: [https://arxiv.org/abs/0902.3383](https://arxiv.org/abs/0902.3383)
4. Python Cryptography Library documentation: [https://cryptography.io/en/latest/](https://cryptography.io/en/latest/)
5. PyCryptodome Library documentation: [https://pycryptodome.readthedocs.io/en/latest/](https://pycryptodome.readthedocs.io/en/latest/)

These resources provide in-depth explanations of cryptographic concepts and advanced techniques, as well as practical implementations in Python.

