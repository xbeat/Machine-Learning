## Kolmogorov Complexity and Algorithmic Randomness in Python
Slide 1: Introduction to Kolmogorov Complexity

Kolmogorov complexity is a measure of the computational resources needed to specify an object. It quantifies the minimum length of a program that produces a given output. This concept is fundamental in information theory and theoretical computer science.

```python
def kolmogorov_complexity(string):
    import zlib
    return len(zlib.compress(string.encode('utf-8')))

# Example usage
print(kolmogorov_complexity("abababababab"))  # Low complexity
print(kolmogorov_complexity("4c1j5b2p0cv4w1x8rx2y39umgw5q85s7"))  # Higher complexity
```

Slide 2: Intuition Behind Kolmogorov Complexity

The Kolmogorov complexity of a string is the length of the shortest program that produces the string as output. Intuitively, a string with more pattern and structure will have lower Kolmogorov complexity than a random string of the same length.

```python
import random
import string

def generate_string(length, pattern=None):
    if pattern:
        return pattern * (length // len(pattern)) + pattern[:length % len(pattern)]
    else:
        return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

structured = generate_string(100, "abc")
random_str = generate_string(100)

print(f"Structured string: {structured[:50]}...")
print(f"Random string: {random_str[:50]}...")
```

Slide 3: Incomputability of Kolmogorov Complexity

A crucial property of Kolmogorov complexity is that it is not computable. This means there's no algorithm that can calculate the exact Kolmogorov complexity for any given string. We can only approximate it.

```python
def approximate_kolmogorov_complexity(s):
    return len(s.encode('utf-8').strip(b'\x00'))

# Example usage
print(approximate_kolmogorov_complexity("abababab"))
print(approximate_kolmogorov_complexity("4c1j5b2p0cv4w1x8rx2y39umgw5q85s7"))
```

Slide 4: Algorithmic Randomness

Algorithmic randomness is closely related to Kolmogorov complexity. A string is considered algorithmically random if its Kolmogorov complexity is close to its length. In other words, the shortest program that produces the string is not much shorter than the string itself.

```python
import random

def generate_random_string(length):
    return ''.join(random.choice('01') for _ in range(length))

random_string = generate_random_string(100)
print(f"Random string: {random_string[:50]}...")
print(f"Approximate complexity: {approximate_kolmogorov_complexity(random_string)}")
```

Slide 5: Compression and Kolmogorov Complexity

There's a close relationship between data compression and Kolmogorov complexity. The more compressible a string is, the lower its Kolmogorov complexity. This principle is used in many practical applications, including file compression.

```python
import zlib

def compress_ratio(s):
    compressed = zlib.compress(s.encode('utf-8'))
    return len(compressed) / len(s)

print(compress_ratio("abababababababab"))  # Low complexity, high compressibility
print(compress_ratio("4c1j5b2p0cv4w1x8rx2y39umgw5q85s7"))  # Higher complexity, lower compressibility
```

Slide 6: Kolmogorov Complexity and Machine Learning

In machine learning, Kolmogorov complexity relates to the principle of Occam's Razor: simpler models are often preferred. We can think of model complexity in terms of the length of the code needed to implement the model.

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Generate some data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10])

# Linear model (simpler)
linear_model = LinearRegression().fit(X, y)

# Polynomial model (more complex)
poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression().fit(X_poly, y)

print(f"Linear model coefficients: {linear_model.coef_}")
print(f"Polynomial model coefficients: {poly_model.coef_}")
```

Slide 7: Algorithmic Probability

Algorithmic probability is the probability that a universal Turing machine will produce a given string when provided with random input. It's closely related to Kolmogorov complexity and forms the basis for algorithmic information theory.

```python
import random

def simulate_turing_machine(steps):
    tape = [0] * 100
    head = 50
    state = 0
    output = ""
    
    for _ in range(steps):
        if state == 0:
            if tape[head] == 0:
                tape[head] = 1
                head += 1
                state = 1
            else:
                tape[head] = 0
                head -= 1
        else:
            if tape[head] == 0:
                tape[head] = 1
                head -= 1
            else:
                tape[head] = 0
                head += 1
                state = 0
        
        output += str(tape[head])
    
    return output

print(simulate_turing_machine(100))
```

Slide 8: Minimum Description Length (MDL)

The Minimum Description Length principle is a formalization of Occam's Razor in terms of computational theory. It states that the best hypothesis for a given set of data is the one that leads to the best compression of the data.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import log2

def mdl_score(X, y, model):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    n = len(y)
    k = len(model.coef_) + 1  # number of parameters
    
    # Model complexity
    L_model = k * log2(n)
    
    # Data complexity given the model
    L_data = n * log2(mse)
    
    return L_model + L_data

# Generate some data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression().fit(X, y)
print(f"MDL score: {mdl_score(X, y, model)}")
```

Slide 9: Kolmogorov Complexity and Cryptography

Kolmogorov complexity has implications for cryptography. A good encryption should produce ciphertext that appears random and thus has high Kolmogorov complexity. Here's a simple (and insecure) encryption example to illustrate this concept:

```python
import random

def simple_encrypt(message, key):
    return ''.join(chr((ord(c) + key) % 256) for c in message)

def simple_decrypt(ciphertext, key):
    return ''.join(chr((ord(c) - key) % 256) for c in ciphertext)

message = "Hello, World!"
key = random.randint(1, 255)

ciphertext = simple_encrypt(message, key)
print(f"Ciphertext: {ciphertext}")
print(f"Decrypted: {simple_decrypt(ciphertext, key)}")
print(f"Apparent complexity: {approximate_kolmogorov_complexity(ciphertext)}")
```

Slide 10: Kolmogorov Complexity in Data Analysis

Kolmogorov complexity can be used to measure the information content of datasets. This has applications in anomaly detection, where events with high Kolmogorov complexity relative to the rest of the data might be considered anomalous.

```python
import numpy as np
import matplotlib.pyplot as plt

def kolmogorov_complexity_timeseries(series):
    return [approximate_kolmogorov_complexity(str(series[:i+1])) for i in range(len(series))]

# Generate a time series with an anomaly
t = np.linspace(0, 10, 1000)
y = np.sin(t)
y[500:520] += 2  # Add an anomaly

complexity = kolmogorov_complexity_timeseries(y)

plt.figure(figsize=(12, 6))
plt.plot(t, y, label='Time Series')
plt.plot(t, complexity, label='Kolmogorov Complexity')
plt.legend()
plt.title('Time Series and Its Kolmogorov Complexity')
plt.show()
```

Slide 11: Algorithmic Randomness Tests

Algorithmic randomness can be tested using various statistical tests. While no test can prove true randomness, they can help identify non-random patterns. Here's an implementation of the frequency test, one of the simplest randomness tests:

```python
import numpy as np

def frequency_test(binary_string, alpha=0.01):
    n = len(binary_string)
    ones = sum(int(bit) for bit in binary_string)
    zeros = n - ones
    s_obs = abs(ones - zeros) / np.sqrt(n)
    p_value = 2 * (1 - 0.5 * (1 + np.erf(s_obs / np.sqrt(2))))
    return p_value > alpha

# Test a supposedly random string
random_string = ''.join(random.choice('01') for _ in range(1000))
print(f"Random string passes frequency test: {frequency_test(random_string)}")

# Test a non-random string
non_random_string = '0' * 500 + '1' * 500
print(f"Non-random string passes frequency test: {frequency_test(non_random_string)}")
```

Slide 12: Lossless Compression and Kolmogorov Complexity

Lossless compression algorithms provide a practical way to approximate Kolmogorov complexity. The compressed size of data serves as an upper bound on its Kolmogorov complexity. Here's an example using different compression algorithms:

```python
import zlib
import bz2
import lzma

def compare_compression(data):
    original_size = len(data)
    zlib_size = len(zlib.compress(data))
    bz2_size = len(bz2.compress(data))
    lzma_size = len(lzma.compress(data))
    
    print(f"Original size: {original_size}")
    print(f"zlib compressed size: {zlib_size}")
    print(f"bz2 compressed size: {bz2_size}")
    print(f"lzma compressed size: {lzma_size}")

# Compare compression of structured vs random data
structured_data = b"abcabcabcabc" * 1000
random_data = bytes(random.randint(0, 255) for _ in range(12000))

print("Structured data:")
compare_compression(structured_data)
print("\nRandom data:")
compare_compression(random_data)
```

Slide 13: Real-Life Example: File Type Detection

Kolmogorov complexity can be used in file type detection. Different file types often have different complexity profiles. Here's a simple example that attempts to distinguish between text and binary files:

```python
import magic

def is_text_file(filename):
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(filename)
    return file_type.startswith('text')

def file_complexity_ratio(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    compressed = zlib.compress(data)
    return len(compressed) / len(data)

# Example usage (you'll need to create these files)
print(f"text_file.txt is text: {is_text_file('text_file.txt')}")
print(f"text_file.txt complexity ratio: {file_complexity_ratio('text_file.txt')}")

print(f"binary_file.bin is text: {is_text_file('binary_file.bin')}")
print(f"binary_file.bin complexity ratio: {file_complexity_ratio('binary_file.bin')}")
```

Slide 14: Real-Life Example: Plagiarism Detection

Kolmogorov complexity can be used in plagiarism detection. Similar documents will have similar complexity when compressed together. Here's a simple example:

```python
import zlib

def similarity(text1, text2):
    c1 = len(zlib.compress(text1.encode()))
    c2 = len(zlib.compress(text2.encode()))
    c_both = len(zlib.compress((text1 + text2).encode()))
    return (c1 + c2 - c_both) / min(c1, c2)

original = "This is a test sentence for plagiarism detection."
plagiarized = "This is a test phrase for plagiarism identification."
different = "The quick brown fox jumps over the lazy dog."

print(f"Similarity (original vs plagiarized): {similarity(original, plagiarized)}")
print(f"Similarity (original vs different): {similarity(original, different)}")
```

Slide 15: Additional Resources

For those interested in diving deeper into Kolmogorov Complexity and Algorithmic Randomness, here are some valuable resources:

1. "An Introduction to Kolmogorov Complexity and Its Applications" by Li and Vitányi (Springer)
2. "Algorithmic Randomness and Complexity" by Downey and Hirschfeldt (Springer)
3. ArXiv paper: "Kolmogorov Complexity and Its Applications" by Ming Li ([https://arxiv.org/abs/0704.2452](https://arxiv.org/abs/0704.2452))
4. ArXiv paper: "Algorithmic Information Theory" by Marcus Hutter ([https://arxiv.org/abs/0811.4888](https://arxiv.org/abs/0811.4888))

These resources provide a more rigorous treatment of the topics we've covered and explore advanced concepts in algorithmic information theory.

