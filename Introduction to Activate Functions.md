## Introduction to Activate Functions

Slide 1: Introduction to Activation Functions

Activation functions are crucial components in neural networks, introducing non-linearity to the model and enabling it to learn complex patterns. They transform the input signal of a neuron into an output signal, determining whether the neuron should be activated or not.

```python
import matplotlib.pyplot as plt

def plot_activation(func, x_range=(-10, 10)):
    x = np.linspace(x_range[0], x_range[1], 200)
    y = func(x)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title(f"{func.__name__} Activation Function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.show()

# Example usage:
# plot_activation(np.tanh)
```

Slide 2: Linear Activation Function

The linear activation function, also known as the identity function, simply returns the input value. While it doesn't introduce non-linearity, it's sometimes used in the output layer for regression tasks.

```python
    return x

plot_activation(linear)
print(f"Linear activation of 2: {linear(2)}")
print(f"Linear activation of -3: {linear(-3)}")
```

Slide 3: Sigmoid Activation Function

The sigmoid function maps input values to a range between 0 and 1, making it useful for binary classification problems. However, it can suffer from vanishing gradients for extreme input values.

```python
    return 1 / (1 + np.exp(-x))

plot_activation(sigmoid)
print(f"Sigmoid activation of 2: {sigmoid(2):.4f}")
print(f"Sigmoid activation of -3: {sigmoid(-3):.4f}")
```

Slide 4: Hyperbolic Tangent (tanh) Activation Function

The tanh function is similar to sigmoid but maps inputs to a range between -1 and 1. It's often preferred over sigmoid as it's zero-centered, which can help with faster convergence during training.

```python
    return np.tanh(x)

plot_activation(tanh)
print(f"Tanh activation of 2: {tanh(2):.4f}")
print(f"Tanh activation of -3: {tanh(-3):.4f}")
```

Slide 5: Rectified Linear Unit (ReLU) Activation Function

ReLU is one of the most popular activation functions due to its simplicity and effectiveness. It returns the input if positive, otherwise zero. ReLU helps mitigate the vanishing gradient problem and allows for sparse activation.

```python
    return np.maximum(0, x)

plot_activation(relu)
print(f"ReLU activation of 2: {relu(2)}")
print(f"ReLU activation of -3: {relu(-3)}")
```

Slide 6: Leaky ReLU Activation Function

Leaky ReLU addresses the "dying ReLU" problem by allowing a small gradient when the input is negative. This can help prevent neurons from becoming inactive during training.

```python
    return np.where(x > 0, x, alpha * x)

plot_activation(leaky_relu)
print(f"Leaky ReLU activation of 2: {leaky_relu(2):.4f}")
print(f"Leaky ReLU activation of -3: {leaky_relu(-3):.4f}")
```

Slide 7: Exponential Linear Unit (ELU) Activation Function

ELU is another alternative to ReLU that can help with the dying ReLU problem. It has a smooth curve for negative values, which can lead to faster learning in some cases.

```python
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

plot_activation(elu)
print(f"ELU activation of 2: {elu(2):.4f}")
print(f"ELU activation of -3: {elu(-3):.4f}")
```

Slide 8: Softmax Activation Function

Softmax is commonly used in the output layer for multi-class classification problems. It converts a vector of real numbers into a probability distribution.

```python
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

x = np.array([2.0, 1.0, 0.1])
result = softmax(x)
print(f"Softmax output: {result}")
print(f"Sum of probabilities: {result.sum():.4f}")
```

Slide 9: Implementing Activation Functions in a Simple Neural Network

Let's create a basic neural network with a single hidden layer to demonstrate how activation functions are used in practice.

```python

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X, activation_func):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = activation_func(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = activation_func(self.z2)
        return self.a2

# Example usage
nn = SimpleNeuralNetwork(2, 3, 1)
X = np.array([[0.5, 0.1]])
output = nn.forward(X, sigmoid)
print(f"Network output: {output[0][0]:.4f}")
```

Slide 10: Comparing Activation Functions

Different activation functions can lead to varying performance depending on the task. Let's compare how some common activation functions behave for a range of inputs.

```python

def compare_activations(x):
    activations = {
        'ReLU': relu(x),
        'Leaky ReLU': leaky_relu(x),
        'Sigmoid': sigmoid(x),
        'Tanh': tanh(x),
        'ELU': elu(x)
    }
    
    plt.figure(figsize=(12, 8))
    for name, y in activations.items():
        plt.plot(x, y, label=name)
    
    plt.title("Comparison of Activation Functions")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(True)
    plt.show()

x = np.linspace(-5, 5, 200)
compare_activations(x)
```

Slide 11: Activation Functions and Gradients

The choice of activation function affects the gradient flow during backpropagation. Let's visualize the gradients of different activation functions.

```python
    gradients = {
        'ReLU': np.where(x > 0, 1, 0),
        'Leaky ReLU': np.where(x > 0, 1, 0.01),
        'Sigmoid': sigmoid(x) * (1 - sigmoid(x)),
        'Tanh': 1 - np.tanh(x)**2,
        'ELU': np.where(x > 0, 1, elu(x) + 1)
    }
    
    plt.figure(figsize=(12, 8))
    for name, y in gradients.items():
        plt.plot(x, y, label=name)
    
    plt.title("Gradients of Activation Functions")
    plt.xlabel("Input")
    plt.ylabel("Gradient")
    plt.legend()
    plt.grid(True)
    plt.show()

x = np.linspace(-5, 5, 200)
plot_gradients(x)
```

Slide 12: Real-life Example: Image Classification

In image classification tasks, ReLU is often used in convolutional layers due to its efficiency and ability to handle sparse representations. Let's simulate a simple convolutional layer with ReLU activation.

```python

def convolve2d(image, kernel):
    output = np.zeros_like(image)
    padding = kernel.shape[0] // 2
    padded_image = np.pad(image, padding, mode='constant')
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(
                padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel
            )
    return output

# Simulating a simple edge detection
image = np.random.rand(10, 10)
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

convolved = convolve2d(image, kernel)
activated = relu(convolved)

print("Original image:")
print(image)
print("\nAfter convolution and ReLU activation:")
print(activated)
```

Slide 13: Real-life Example: Natural Language Processing

In natural language processing tasks, the softmax function is commonly used in the output layer for text classification. Let's simulate a simple sentiment analysis model.

```python

def simple_sentiment_analysis(text, word_vectors, weights):
    words = text.lower().split()
    text_vector = np.mean([word_vectors.get(word, np.zeros(5)) for word in words], axis=0)
    logits = np.dot(text_vector, weights)
    probabilities = softmax(logits)
    return probabilities

# Simulated word vectors and model weights
word_vectors = {
    'good': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    'bad': np.array([-0.1, -0.2, -0.3, -0.4, -0.5]),
    'movie': np.array([0.0, 0.1, 0.2, 0.3, 0.4])
}
weights = np.random.randn(5, 3)  # 3 classes: negative, neutral, positive

text = "good movie"
sentiment_probs = simple_sentiment_analysis(text, word_vectors, weights)
print(f"Sentiment probabilities for '{text}':")
print(f"Negative: {sentiment_probs[0]:.4f}")
print(f"Neutral: {sentiment_probs[1]:.4f}")
print(f"Positive: {sentiment_probs[2]:.4f}")
```

Slide 14: Choosing the Right Activation Function

The choice of activation function depends on various factors such as the nature of the problem, network architecture, and desired properties. Here are some general guidelines:

1. ReLU and its variants (Leaky ReLU, ELU) are often good default choices for hidden layers in many types of neural networks.
2. Sigmoid is useful for binary classification problems in the output layer.
3. Softmax is ideal for multi-class classification problems in the output layer.
4. Tanh can be a good alternative to ReLU in recurrent neural networks (RNNs).
5. Linear activation is typically used in the output layer for regression tasks.

Experiment with different activation functions and combinations to find what works best for your specific problem.

Slide 15: Additional Resources

For more in-depth information on activation functions and their applications in neural networks, consider exploring these resources:

1. "Empirical Evaluation of Rectified Activations in Convolutional Network" by Bing Xu et al. (2015) - [https://arxiv.org/abs/1505.00853](https://arxiv.org/abs/1505.00853)
2. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" by Kaiming He et al. (2015) - [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852)
3. "Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units" by Dan Hendrycks and Kevin Gimpel (2016) - [https://arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415)

These papers provide valuable insights into the development and analysis of various activation functions used in modern neural networks.


