## Mastering Artificial Neural Networks Feedforward and Backpropagation Fundamentals
Slide 1: Introduction to Artificial Neural Networks

Artificial Neural Networks (ANNs) are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers, capable of learning patterns from data. ANNs are fundamental to many machine learning tasks, including image recognition, natural language processing, and predictive modeling.

Slide 2: Source Code for Introduction to Artificial Neural Networks

```python
# Simple representation of a neuron
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def activate(self, inputs):
        # Weighted sum of inputs
        total = sum(w * x for w, x in zip(self.weights, inputs))
        # Add bias and apply activation function (e.g., step function)
        return 1 if total + self.bias > 0 else 0

# Example usage
neuron = Neuron([0.5, -0.5], 0.1)
output = neuron.activate([1, 0])
print(f"Neuron output: {output}")
```

Slide 3: Feedforward Neural Networks

Feedforward neural networks are the simplest form of ANNs. In these networks, information flows in one direction: from input nodes through hidden layers to output nodes. Each connection between nodes has an associated weight, which is adjusted during training to improve the network's performance.

Slide 4: Source Code for Feedforward Neural Networks

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class FeedforwardNeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        
        for i in range(1, len(layers)):
            self.weights.append([[0 for _ in range(layers[i-1])] for _ in range(layers[i])])
            self.biases.append([0 for _ in range(layers[i])])
    
    def forward(self, inputs):
        for i in range(len(self.layers) - 1):
            new_inputs = []
            for j in range(self.layers[i+1]):
                total = sum(w * x for w, x in zip(self.weights[i][j], inputs))
                new_inputs.append(sigmoid(total + self.biases[i][j]))
            inputs = new_inputs
        return inputs

# Example usage
nn = FeedforwardNeuralNetwork([2, 3, 1])
output = nn.forward([0.5, 0.8])
print(f"Network output: {output}")
```

Slide 5: Backpropagation

Backpropagation is the primary algorithm used to train neural networks. It calculates the gradient of the loss function with respect to the network's weights, allowing for iterative weight updates to minimize the error between predicted and actual outputs.

Slide 6: Source Code for Backpropagation

```python
import random

class BackpropNetwork(FeedforwardNeuralNetwork):
    def backward(self, inputs, target, learning_rate):
        # Forward pass
        activations = [inputs]
        for i in range(len(self.layers) - 1):
            new_activations = []
            for j in range(self.layers[i+1]):
                total = sum(w * x for w, x in zip(self.weights[i][j], activations[-1]))
                new_activations.append(sigmoid(total + self.biases[i][j]))
            activations.append(new_activations)
        
        # Backward pass
        deltas = [[] for _ in range(len(self.layers) - 1)]
        for i in reversed(range(len(self.layers) - 1)):
            if i == len(self.layers) - 2:
                deltas[i] = [a * (1 - a) * (a - t) for a, t in zip(activations[-1], target)]
            else:
                deltas[i] = [a * (1 - a) * sum(w * d for w, d in zip(self.weights[i+1], deltas[i+1]))
                             for a in activations[i+1]]
        
        # Update weights and biases
        for i in range(len(self.layers) - 1):
            for j in range(self.layers[i+1]):
                self.weights[i][j] = [w - learning_rate * d * a for w, a, d in 
                                      zip(self.weights[i][j], activations[i], deltas[i])]
                self.biases[i][j] -= learning_rate * deltas[i][j]

# Example usage
nn = BackpropNetwork([2, 3, 1])
nn.backward([0.5, 0.8], [1], 0.1)
```

Slide 7: Training Process

The training process involves repeatedly presenting input data to the network, comparing the network's output to the desired output, and adjusting the weights using backpropagation. This process continues until the network achieves satisfactory performance or a predetermined number of iterations is reached.

Slide 8: Source Code for Training Process

```python
def train(network, dataset, epochs, learning_rate):
    for epoch in range(epochs):
        total_loss = 0
        for inputs, target in dataset:
            output = network.forward(inputs)
            network.backward(inputs, target, learning_rate)
            total_loss += sum((o - t) ** 2 for o, t in zip(output, target))
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataset)}")

# Example dataset (XOR problem)
xor_dataset = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

nn = BackpropNetwork([2, 3, 1])
train(nn, xor_dataset, epochs=1000, learning_rate=0.1)
```

Slide 9: Impact of Changing Parameters

The performance of a neural network is heavily influenced by various parameters, including the number of layers, neurons per layer, learning rate, and activation functions. Adjusting these parameters can significantly affect the network's ability to learn and generalize from data.

Slide 10: Source Code for Impact of Changing Parameters

```python
def experiment_with_parameters():
    architectures = [[2, 2, 1], [2, 3, 1], [2, 4, 1]]
    learning_rates = [0.01, 0.1, 0.5]
    
    for arch in architectures:
        for lr in learning_rates:
            print(f"Architecture: {arch}, Learning Rate: {lr}")
            nn = BackpropNetwork(arch)
            train(nn, xor_dataset, epochs=1000, learning_rate=lr)
            
            # Test the trained network
            for inputs, target in xor_dataset:
                output = nn.forward(inputs)
                print(f"Input: {inputs}, Output: {output[0]:.4f}, Target: {target[0]}")
            print()

experiment_with_parameters()
```

Slide 11: Real-Life Example: Image Classification

Neural networks excel at image classification tasks. A common application is handwritten digit recognition, where a network learns to identify digits from 0 to 9 based on pixel intensities.

Slide 12: Source Code for Image Classification Example

```python
def create_digit_dataset():
    # Simplified 5x5 pixel representations of digits 0 and 1
    digit_0 = [
        0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 0
    ]
    digit_1 = [
        0, 0, 1, 0, 0,
        0, 1, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 1, 1, 1, 0
    ]
    return [(digit_0, [1, 0]), (digit_1, [0, 1])]

digit_dataset = create_digit_dataset()
digit_nn = BackpropNetwork([25, 10, 2])
train(digit_nn, digit_dataset, epochs=1000, learning_rate=0.1)

# Test the trained network
for inputs, target in digit_dataset:
    output = digit_nn.forward(inputs)
    predicted = 0 if output[0] > output[1] else 1
    print(f"Predicted: {predicted}, Actual: {target.index(1)}")
```

Slide 13: Real-Life Example: Natural Language Processing

Neural networks are widely used in Natural Language Processing (NLP) tasks, such as sentiment analysis. In this example, we'll create a simple sentiment classifier for short text reviews.

Slide 14: Source Code for Natural Language Processing Example

```python
def text_to_vector(text):
    # Simple bag-of-words representation
    vocab = set("positive negative good bad excellent poor".split())
    return [1 if word in text.lower() else 0 for word in vocab]

def create_sentiment_dataset():
    reviews = [
        ("This product is excellent", [1, 0]),  # Positive
        ("Poor quality, very disappointed", [0, 1]),  # Negative
        ("Good value for money", [1, 0]),  # Positive
        ("Terrible experience, avoid", [0, 1])  # Negative
    ]
    return [(text_to_vector(review), sentiment) for review, sentiment in reviews]

sentiment_dataset = create_sentiment_dataset()
sentiment_nn = BackpropNetwork([6, 4, 2])
train(sentiment_nn, sentiment_dataset, epochs=1000, learning_rate=0.1)

# Test the trained network
test_reviews = [
    "This is a good product",
    "Negative experience overall"
]
for review in test_reviews:
    vec = text_to_vector(review)
    output = sentiment_nn.forward(vec)
    sentiment = "Positive" if output[0] > output[1] else "Negative"
    print(f"Review: '{review}'\nSentiment: {sentiment}\n")
```

Slide 15: Additional Resources

For a deeper understanding of artificial neural networks and advanced techniques, consider exploring these peer-reviewed articles from arXiv.org:

1.  "Deep Learning" by LeCun, Y., Bengio, Y., & Hinton, G. (2015) arXiv:1505.00387
2.  "Neural Networks and Deep Learning: A Textbook" by Aggarwal, C. C. (2018) arXiv:1803.08533
3.  "Efficient BackProp" by LeCun, Y., Bottou, L., Orr, G. B., & Müller, K. R. (2012) arXiv:1206.5533

These resources provide comprehensive coverage of neural network architectures, training algorithms, and applications in various domains.

