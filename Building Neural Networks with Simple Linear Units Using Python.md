## Building Neural Networks with Simple Linear Units Using Python
Slide 1: The Linear Unit in Neural Networks

The fundamental building block of neural networks, the linear unit, is indeed as simple as high school math. It's essentially a weighted sum of inputs plus a bias term, similar to the equation of a line: y = mx + b.

```python
import numpy as np

def linear_unit(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

# Example usage
inputs = np.array([1, 2, 3])
weights = np.array([0.5, -0.2, 0.1])
bias = 1.0

output = linear_unit(inputs, weights, bias)
print(f"Linear unit output: {output}")
```

Slide 2: Understanding Inputs and Weights

Inputs represent the features of our data, while weights determine the importance of each feature. The bias allows the model to fit the data better by shifting the output.

```python
# Visualizing the effect of weights
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y1 = 0.5 * x + 1  # weight = 0.5, bias = 1
y2 = 2 * x + 1    # weight = 2, bias = 1

plt.plot(x, y1, label='Weight = 0.5')
plt.plot(x, y2, label='Weight = 2')
plt.legend()
plt.title('Effect of Weights on Linear Unit Output')
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()
```

Slide 3: The Activation Function

To introduce non-linearity and enable the network to learn complex patterns, we apply an activation function to the linear unit's output.

```python
def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Comparing ReLU and Sigmoid
x = np.linspace(-10, 10, 100)
y_relu = [relu(i) for i in x]
y_sigmoid = [sigmoid(i) for i in x]

plt.plot(x, y_relu, label='ReLU')
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.legend()
plt.title('ReLU vs Sigmoid Activation Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()
```

Slide 4: Combining Linear Units: The Neuron

A neuron is created by applying an activation function to the output of a linear unit. This forms the basic computational unit of neural networks.

```python
def neuron(inputs, weights, bias, activation_function):
    linear_output = np.dot(inputs, weights) + bias
    return activation_function(linear_output)

# Example usage
inputs = np.array([1, 2, 3])
weights = np.array([0.5, -0.2, 0.1])
bias = 1.0

output_relu = neuron(inputs, weights, bias, relu)
output_sigmoid = neuron(inputs, weights, bias, sigmoid)

print(f"Neuron output (ReLU): {output_relu}")
print(f"Neuron output (Sigmoid): {output_sigmoid}")
```

Slide 5: Building a Layer of Neurons

A layer in a neural network consists of multiple neurons operating in parallel. Each neuron in the layer processes the same input differently.

```python
def layer(inputs, weights, biases, activation_function):
    return np.array([neuron(inputs, w, b, activation_function) 
                     for w, b in zip(weights, biases)])

# Example usage
inputs = np.array([1, 2, 3])
weights = np.array([[0.5, -0.2, 0.1],
                    [-0.3, 0.4, 0.2],
                    [0.1, 0.1, 0.8]])
biases = np.array([1.0, 0.5, -0.5])

layer_output = layer(inputs, weights, biases, relu)
print(f"Layer output: {layer_output}")
```

Slide 6: Forward Propagation

Forward propagation is the process of passing input data through the network to generate predictions. It involves applying the layer function repeatedly.

```python
def forward_propagation(inputs, layers):
    for layer_weights, layer_biases in layers:
        inputs = layer(inputs, layer_weights, layer_biases, relu)
    return inputs

# Example usage
inputs = np.array([1, 2, 3])
layers = [
    (np.array([[0.5, -0.2, 0.1], [-0.3, 0.4, 0.2]]), np.array([1.0, 0.5])),
    (np.array([[0.1, 0.8], [0.7, -0.1]]), np.array([-0.5, 0.2]))
]

output = forward_propagation(inputs, layers)
print(f"Network output: {output}")
```

Slide 7: Backpropagation: The Learning Process

Backpropagation is the algorithm used to train neural networks. It calculates the gradient of the loss function with respect to the network's weights.

```python
def simple_backpropagation(inputs, target, weights, learning_rate):
    # Forward pass
    output = neuron(inputs, weights, 0, sigmoid)
    
    # Backward pass
    error = target - output
    delta = error * output * (1 - output)
    
    # Update weights
    for i in range(len(weights)):
        weights[i] += learning_rate * inputs[i] * delta
    
    return weights

# Example usage
inputs = np.array([1, 2, 3])
target = 0.7
weights = np.array([0.5, -0.2, 0.1])
learning_rate = 0.1

for _ in range(1000):
    weights = simple_backpropagation(inputs, target, weights, learning_rate)

print(f"Trained weights: {weights}")
print(f"Final output: {neuron(inputs, weights, 0, sigmoid)}")
```

Slide 8: Gradient Descent: Optimizing the Network

Gradient descent is the optimization algorithm used in backpropagation. It iteratively adjusts the weights to minimize the loss function.

```python
def gradient_descent(X, y, learning_rate, epochs):
    m, n = X.shape
    weights = np.zeros(n)
    
    for _ in range(epochs):
        y_pred = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (y_pred - y)) / m
        weights -= learning_rate * gradient
    
    return weights

# Example usage
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([0, 1, 1])
learning_rate = 0.01
epochs = 1000

trained_weights = gradient_descent(X, y, learning_rate, epochs)
print(f"Trained weights: {trained_weights}")
```

Slide 9: Loss Functions: Measuring Performance

Loss functions quantify the difference between predicted and actual outputs, guiding the learning process.

```python
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example usage
y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0.1, 0.9, 0.8, 0.2])

mse = mean_squared_error(y_true, y_pred)
bce = binary_cross_entropy(y_true, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Binary Cross-Entropy: {bce}")
```

Slide 10: Regularization: Preventing Overfitting

Regularization techniques help prevent overfitting by adding a penalty term to the loss function, discouraging complex models.

```python
def ridge_regression(X, y, alpha):
    return np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ y

# Example usage
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([3, 7, 11])
alpha = 0.1

weights = ridge_regression(X, y, alpha)
print(f"Ridge regression weights: {weights}")
```

Slide 11: Real-life Example: Handwritten Digit Recognition

Using the concepts we've learned, let's build a simple neural network to recognize handwritten digits from the MNIST dataset.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load data
digits = load_digits()
X, y = digits.data, digits.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = mlp.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

Slide 12: Real-life Example: Image Classification

Let's use a pre-trained convolutional neural network to classify images, demonstrating the power of deep learning in computer vision tasks.

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Load pre-trained model
model = MobileNetV2(weights='imagenet')

# Load and preprocess image
img_path = 'path_to_your_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make prediction
preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]

# Print results
for _, label, score in decoded_preds:
    print(f"{label}: {score:.2f}")
```

Slide 13: Conclusion and Future Directions

We've explored the fundamental concepts of neural networks, from the basic linear unit to more complex architectures. As AI continues to evolve, new techniques and architectures are constantly being developed.

Some emerging areas include:

* Attention mechanisms and transformers
* Generative models like GANs and VAEs
* Reinforcement learning
* Neuromorphic computing

The field of neural networks and deep learning is rapidly advancing, offering exciting opportunities for research and application in various domains.

Slide 14: Additional Resources

For those interested in diving deeper into neural networks and deep learning, here are some valuable resources:

1. ArXiv.org: A vast repository of research papers on neural networks and AI. URL: [https://arxiv.org/list/cs.NE/recent](https://arxiv.org/list/cs.NE/recent)
2. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville ArXiv link: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
3. "Neural Networks and Deep Learning" by Michael Nielsen (Free online book, not on ArXiv)
4. TensorFlow and PyTorch documentation for practical implementations (Official websites, not on ArXiv)

These resources provide a mix of theoretical foundations and practical implementations to further your understanding of neural networks.

