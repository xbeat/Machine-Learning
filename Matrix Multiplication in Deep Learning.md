## Matrix Multiplication in Deep Learning
Slide 1: Matrix Multiplication in Deep Learning

Matrix multiplication is a fundamental operation in deep learning, crucial for processing inputs, applying weights, and propagating information through neural networks. It allows the network to combine and transform data in ways that enable pattern recognition and complex computations.

Slide 2: Source Code for Matrix Multiplication in Deep Learning

```python
def matrix_multiply(A, B):
    # Ensure matrices can be multiplied
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions don't match for multiplication")
    
    # Initialize result matrix with zeros
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    
    # Perform matrix multiplication
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

# Example usage
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
result = matrix_multiply(A, B)
print("Result of matrix multiplication:")
for row in result:
    print(row)
```

Slide 3: Results for Matrix Multiplication in Deep Learning

```python
Result of matrix multiplication:
[19, 22]
[43, 50]
```

Slide 4: Inputs and Weights in Neural Networks

In neural networks, inputs represent the data being processed, while weights determine the importance of each input. The combination of inputs and weights through matrix multiplication forms the basis of how neural networks learn and make predictions.

Slide 5: Source Code for Inputs and Weights in Neural Networks

```python
import random

def initialize_network(input_size, hidden_size, output_size):
    network = {
        'hidden': [[random.uniform(-1, 1) for _ in range(input_size + 1)] for _ in range(hidden_size)],
        'output': [[random.uniform(-1, 1) for _ in range(hidden_size + 1)] for _ in range(output_size)]
    }
    return network

# Example usage
input_size, hidden_size, output_size = 3, 4, 2
network = initialize_network(input_size, hidden_size, output_size)
print("Hidden layer weights (including bias):")
for weights in network['hidden']:
    print(weights)
print("\nOutput layer weights (including bias):")
for weights in network['output']:
    print(weights)
```

Slide 6: Learning in Neural Networks

Neural networks learn by adjusting their weights based on the error between predicted and actual outputs. This process, called backpropagation, involves calculating gradients and updating weights to minimize the error over many iterations.

Slide 7: Source Code for Learning in Neural Networks

```python
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def forward_propagate(network, inputs):
    layer = inputs
    for weights in network:
        new_layer = []
        for neuron_weights in weights:
            activation = neuron_weights[-1]  # Bias
            for i in range(len(neuron_weights) - 1):
                activation += neuron_weights[i] * layer[i]
            new_layer.append(sigmoid(activation))
        layer = new_layer
    return layer

# Example usage
inputs = [0.5, 0.3, 0.2]
output = forward_propagate([network['hidden'], network['output']], inputs)
print("Network output:", output)
```

Slide 8: Importance of Matrix Multiplication in Deep Learning

Matrix multiplication enables neural networks to efficiently combine and transform large amounts of data. This operation allows the network to capture complex patterns and relationships in the input data, leading to powerful representations and accurate predictions.

Slide 9: Source Code for Importance of Matrix Multiplication in Deep Learning

```python
import math

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def activation_function(x):
    return 1 / (1 + math.exp(-x))

def neural_network_layer(inputs, weights):
    # Calculate dot product of inputs and weights
    dot_products = [dot_product(inputs, w) for w in weights]
    
    # Apply activation function to each dot product
    activations = [activation_function(dp) for dp in dot_products]
    
    return activations

# Example usage
inputs = [0.5, 0.3, 0.2]
weights = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
output = neural_network_layer(inputs, weights)
print("Layer output:", output)
```

Slide 10: Real-Life Example: Image Recognition

Image recognition uses matrix multiplication to process pixel values. Each pixel's RGB values are multiplied with learned weights to detect features like edges, textures, and shapes, enabling the network to classify images accurately.

Slide 11: Source Code for Image Recognition Example

```python
def convolve2d(image, kernel):
    m, n = len(image), len(image[0])
    k = len(kernel)
    pad = k // 2
    output = [[0 for _ in range(n)] for _ in range(m)]
    
    for i in range(m):
        for j in range(n):
            sum = 0
            for ki in range(k):
                for kj in range(k):
                    ii, jj = i - pad + ki, j - pad + kj
                    if 0 <= ii < m and 0 <= jj < n:
                        sum += image[ii][jj] * kernel[ki][kj]
            output[i][j] = sum
    
    return output

# Example usage: Edge detection
image = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]
edge_kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
edge_detected = convolve2d(image, edge_kernel)

print("Edge detection result:")
for row in edge_detected:
    print([round(x, 2) for x in row])
```

Slide 12: Real-Life Example: Natural Language Processing

In natural language processing, matrix multiplication is used to combine word embeddings with learned weights, allowing the network to understand context and meaning in text data for tasks like sentiment analysis or language translation.

Slide 13: Source Code for Natural Language Processing Example

```python
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (magnitude1 * magnitude2)

# Simple word embeddings (normally these would be learned)
word_embeddings = {
    "cat": [0.2, 0.4, 0.3],
    "dog": [0.1, 0.5, 0.2],
    "pet": [0.3, 0.4, 0.1]
}

# Calculate similarity between words
word1, word2 = "cat", "dog"
similarity = cosine_similarity(word_embeddings[word1], word_embeddings[word2])
print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")

word1, word2 = "cat", "pet"
similarity = cosine_similarity(word_embeddings[word1], word_embeddings[word2])
print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
```

Slide 14: Additional Resources

For more in-depth information on matrix multiplication in deep learning, consider these resources:

1.  "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press)
2.  ArXiv paper: "Matrix multiplication for deep learning" by X. Zhang et al. (arXiv:2106.10860)
3.  Stanford CS231n course: "Convolutional Neural Networks for Visual Recognition"

These resources provide comprehensive coverage of the topic and its applications in various deep learning scenarios.

