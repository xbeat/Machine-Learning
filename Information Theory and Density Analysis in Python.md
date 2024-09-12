## Information Theory and Density Analysis in Python
Slide 1: Information Theory: The Basics

Information theory is a mathematical framework for quantifying, storing, and communicating information. It was developed by Claude Shannon in 1948 and has since become fundamental to many fields, including computer science, data compression, and cryptography.

```python
import math

def entropy(probabilities):
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

# Example: Calculate entropy for a fair coin toss
fair_coin = [0.5, 0.5]
print(f"Entropy of a fair coin: {entropy(fair_coin):.2f} bits")
```

Slide 2: Shannon Entropy

Shannon entropy is a measure of the average amount of information contained in a message. It quantifies the unpredictability of information content.

```python
import numpy as np

def shannon_entropy(data):
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return -np.sum(probabilities * np.log2(probabilities))

# Example: Calculate Shannon entropy for a text message
message = "hello world"
print(f"Shannon entropy of '{message}': {shannon_entropy(list(message)):.2f} bits")
```

Slide 3: Information Gain

Information gain is a measure of the reduction in entropy achieved by splitting a dataset according to a particular feature. It's commonly used in decision tree algorithms for feature selection.

```python
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Create a sample dataset
data = pd.DataFrame({
    'feature1': [1, 2, 1, 2, 1],
    'feature2': [0, 1, 1, 0, 1],
    'target': [0, 1, 1, 0, 1]
})

# Calculate information gain
X = data[['feature1', 'feature2']]
y = data['target']
info_gain = mutual_info_classif(X, y)

for feature, gain in zip(X.columns, info_gain):
    print(f"Information gain for {feature}: {gain:.4f}")
```

Slide 4: Kullback-Leibler Divergence

KL divergence measures the difference between two probability distributions. It's often used in machine learning to compare models or distributions.

```python
import numpy as np
from scipy.stats import entropy

def kl_divergence(p, q):
    return entropy(p, q)

# Example: Compare two probability distributions
p = np.array([0.2, 0.5, 0.3])
q = np.array([0.1, 0.4, 0.5])

print(f"KL divergence: {kl_divergence(p, q):.4f}")
```

Slide 5: Mutual Information

Mutual information quantifies the mutual dependence between two variables. It measures how much information one variable provides about another.

```python
import numpy as np
from sklearn.metrics import mutual_info_score

def mutual_information(x, y):
    return mutual_info_score(x, y)

# Example: Calculate mutual information between two variables
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

print(f"Mutual information: {mutual_information(x, y):.4f}")
```

Slide 6: Data Compression: Run-Length Encoding

Run-length encoding (RLE) is a simple form of data compression that replaces sequences of repeated data elements with a count and a single  of the element.

```python
def rle_encode(data):
    encoding = ''
    prev_char = data[0]
    count = 1
    for char in data[1:]:
        if char == prev_char:
            count += 1
        else:
            encoding += str(count) + prev_char
            count = 1
            prev_char = char
    encoding += str(count) + prev_char
    return encoding

# Example
original = "AABBBCCCC"
compressed = rle_encode(original)
print(f"Original: {original}")
print(f"Compressed: {compressed}")
```

Slide 7: Huffman Coding

Huffman coding is a data compression technique that assigns variable-length codes to characters based on their frequency of occurrence.

```python
import heapq
from collections import Counter

def huffman_encode(data):
    freq = Counter(data)
    heap = [[weight, [char, ""]] for char, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return dict(heap[0][1:])

# Example
text = "this is an example of huffman encoding"
codes = huffman_encode(text)
print("Huffman Codes:")
for char, code in codes.items():
    print(f"{char}: {code}")
```

Slide 8: Channel Capacity

Channel capacity is the maximum rate at which information can be reliably transmitted over a communication channel.

```python
import numpy as np

def channel_capacity(snr):
    return np.log2(1 + snr)

# Example: Calculate channel capacity for different SNR values
snr_values = [1, 10, 100]
for snr in snr_values:
    capacity = channel_capacity(snr)
    print(f"Channel capacity for SNR {snr}: {capacity:.2f} bits/s/Hz")
```

Slide 9: Error Detection: Parity Bit

A parity bit is a simple form of error detection used in digital communication. It's added to a group of bits to ensure the total number of 1s is even (even parity) or odd (odd parity).

```python
def add_parity_bit(data, even=True):
    parity = sum(int(bit) for bit in data) % 2
    if even:
        return data + str(parity)
    else:
        return data + str(1 - parity)

def check_parity(data, even=True):
    return sum(int(bit) for bit in data) % 2 == 0 if even else sum(int(bit) for bit in data) % 2 == 1

# Example
original = "1011"
with_parity = add_parity_bit(original, even=True)
print(f"Original: {original}")
print(f"With even parity: {with_parity}")
print(f"Parity check: {check_parity(with_parity, even=True)}")
```

Slide 10: Hamming Distance

Hamming distance measures the number of positions at which corresponding symbols in two strings of equal length are different. It's used in error detection and correction.

```python
def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Strings must be of equal length")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

# Example
string1 = "10101"
string2 = "11001"
distance = hamming_distance(string1, string2)
print(f"Hamming distance between {string1} and {string2}: {distance}")
```

Slide 11: Information Density in Natural Language

Information density in natural language refers to the amount of information conveyed per unit of text. We can estimate this using measures like entropy per character.

```python
from collections import Counter
import math

def text_entropy(text):
    char_counts = Counter(text.lower())
    total_chars = sum(char_counts.values())
    char_probs = {char: count / total_chars for char, count in char_counts.items()}
    return -sum(p * math.log2(p) for p in char_probs.values())

# Example: Compare information density of two texts
text1 = "The quick brown fox jumps over the lazy dog."
text2 = "AAA BBB CCC DDD EEE FFF GGG HHH III JJJ."

print(f"Entropy of text1: {text_entropy(text1):.2f} bits/char")
print(f"Entropy of text2: {text_entropy(text2):.2f} bits/char")
```

Slide 12: Kolmogorov Complexity

Kolmogorov complexity is a measure of the computational resources needed to specify an object. While it's not directly computable, we can estimate it using compression algorithms.

```python
import zlib

def estimate_kolmogorov_complexity(data):
    return len(zlib.compress(data.encode()))

# Example: Compare Kolmogorov complexity estimates
data1 = "abababababababababababababababab"
data2 = "The quick brown fox jumps over the lazy dog"

print(f"Estimated complexity of data1: {estimate_kolmogorov_complexity(data1)}")
print(f"Estimated complexity of data2: {estimate_kolmogorov_complexity(data2)}")
```

Slide 13: Cross-Entropy and Perplexity

Cross-entropy and perplexity are metrics used to evaluate language models. They measure how well a probability distribution predicts a sample.

```python
import numpy as np

def cross_entropy(true_dist, pred_dist):
    return -np.sum(true_dist * np.log2(pred_dist))

def perplexity(cross_entropy):
    return 2 ** cross_entropy

# Example: Calculate cross-entropy and perplexity for language model predictions
true_dist = np.array([0.1, 0.2, 0.3, 0.4])
pred_dist = np.array([0.15, 0.25, 0.25, 0.35])

ce = cross_entropy(true_dist, pred_dist)
ppl = perplexity(ce)

print(f"Cross-entropy: {ce:.4f}")
print(f"Perplexity: {ppl:.4f}")
```

Slide 14: Information Bottleneck Method

The Information Bottleneck method is a technique for finding the optimal tradeoff between accuracy and complexity in signal processing and machine learning.

```python
import numpy as np
from sklearn.feature_selection import mutual_info_regression

def information_bottleneck(X, y, beta):
    mi_features = mutual_info_regression(X, y)
    mi_threshold = np.percentile(mi_features, (1 - beta) * 100)
    selected_features = mi_features >= mi_threshold
    return X[:, selected_features], selected_features

# Example: Apply Information Bottleneck for feature selection
X = np.random.rand(100, 10)
y = np.random.rand(100)
X_selected, selected_features = information_bottleneck(X, y, beta=0.5)

print(f"Original features: {X.shape[1]}")
print(f"Selected features: {X_selected.shape[1]}")
print(f"Selected feature indices: {np.where(selected_features)[0]}")
```

Slide 15: Additional Resources

For those interested in diving deeper into information theory and its applications, here are some valuable resources:

1. "Elements of Information Theory" by Thomas M. Cover and Joy A. Thomas
2. "Information Theory, Inference, and Learning Algorithms" by David J.C. MacKay
3. ArXiv.org papers:
   * "A Tutorial on Information Theory and Machine Learning" (arXiv:2304.08676)
   * "Information Theory in Machine Learning" (arXiv:2103.13197)

These resources provide comprehensive coverage of information theory concepts and their applications in various fields, including machine learning and data analysis.

