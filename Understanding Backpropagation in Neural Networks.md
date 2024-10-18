## Understanding Backpropagation in Neural Networks

Slide 1: What is Backpropagation?

Backpropagation is a fundamental algorithm in neural networks that calculates gradients of the loss function with respect to network weights. It's the key mechanism allowing neural networks to learn from errors and improve their performance over time.

```python

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Simple neural network with one hidden layer
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        
        self.weights1 += d_weights1
        self.weights2 += d_weights2
```

Slide 2: Forward Pass

The forward pass is the first step in backpropagation. During this phase, input data flows through the network, layer by layer, until it reaches the output. Each neuron applies its activation function to produce an output.

```python

def relu(x):
    return np.maximum(0, x)

def forward_pass(x, weights):
    layer1 = relu(np.dot(x, weights[0]))
    layer2 = relu(np.dot(layer1, weights[1]))
    output = np.dot(layer2, weights[2])
    return output, (layer1, layer2)

# Example usage
x = np.array([[0.5, 0.1, 0.2]])  # Input
weights = [
    np.random.randn(3, 4),  # Weights for layer 1
    np.random.randn(4, 4),  # Weights for layer 2
    np.random.randn(4, 1)   # Weights for output layer
]

output, hidden_layers = forward_pass(x, weights)
print("Output:", output)
print("Hidden layer 1:", hidden_layers[0])
print("Hidden layer 2:", hidden_layers[1])
```

Slide 3: Calculating Error

After the forward pass, we calculate the error between the network's prediction and the actual target value. This error quantifies how far off our prediction is and guides the weight updates.

```python

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Small value to avoid log(0)
    return -np.sum(y_true * np.log(y_pred + epsilon))

# Example usage
y_true = np.array([[1, 0, 0]])  # One-hot encoded true label
y_pred = np.array([[0.7, 0.2, 0.1]])  # Predicted probabilities

mse = mean_squared_error(y_true, y_pred)
ce = cross_entropy(y_true, y_pred)

print("Mean Squared Error:", mse)
print("Cross Entropy Loss:", ce)
```

Slide 4: Backward Pass

The backward pass is where the actual "backpropagation" happens. We compute gradients of the loss with respect to each weight, starting from the output layer and moving backwards through the network.

```python

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def backward_pass(x, y, output, hidden_layers, weights):
    # Compute gradients
    output_error = output - y
    output_delta = output_error

    layer2_error = np.dot(output_delta, weights[2].T)
    layer2_delta = layer2_error * relu_derivative(hidden_layers[1])

    layer1_error = np.dot(layer2_delta, weights[1].T)
    layer1_delta = layer1_error * relu_derivative(hidden_layers[0])

    # Compute weight gradients
    grad_w2 = np.dot(hidden_layers[1].T, output_delta)
    grad_w1 = np.dot(hidden_layers[0].T, layer2_delta)
    grad_w0 = np.dot(x.T, layer1_delta)

    return [grad_w0, grad_w1, grad_w2]

# Example usage (continuing from forward pass)
y = np.array([[1]])  # Target
gradients = backward_pass(x, y, output, hidden_layers, weights)

for i, grad in enumerate(gradients):
    print(f"Gradients for layer {i}:")
    print(grad)
```

Slide 5: Updating Weights

After computing gradients, we update the weights using an optimization algorithm like gradient descent. This step adjusts the network's parameters to reduce the error in future predictions.

```python

def update_weights(weights, gradients, learning_rate):
    return [w - learning_rate * g for w, g in zip(weights, gradients)]

# Example usage (continuing from backward pass)
learning_rate = 0.01
updated_weights = update_weights(weights, gradients, learning_rate)

for i, (old_w, new_w) in enumerate(zip(weights, updated_weights)):
    print(f"Layer {i} weight change:")
    print(np.mean(np.abs(new_w - old_w)))
```

Slide 6: Gradient Descent Optimization

Gradient descent is the optimization algorithm commonly used with backpropagation. It iteratively adjusts weights in the direction that reduces the loss function, helping the network converge to a good solution.

```python
import matplotlib.pyplot as plt

def gradient_descent(start, gradient, learn_rate, n_iter=50, tolerance=1e-06):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector

# Example: Find minimum of f(x) = x^2 + 5
def f(x):
    return x**2 + 5

def gradient(x):
    return 2*x

x = np.linspace(-10, 10, 100)
y = f(x)

plt.plot(x, y)
plt.title("f(x) = x^2 + 5")
plt.xlabel("x")
plt.ylabel("f(x)")

result = gradient_descent(start=10.0, gradient=gradient, learn_rate=0.1)
plt.plot(result, f(result), 'ro')
plt.show()

print(f"Minimum found at x = {result}")
```

Slide 7: Activation Functions

Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns. Common activation functions include ReLU, sigmoid, and tanh.

```python
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

x = np.linspace(-5, 5, 100)

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(x, relu(x))
plt.title("ReLU")

plt.subplot(132)
plt.plot(x, sigmoid(x))
plt.title("Sigmoid")

plt.subplot(133)
plt.plot(x, tanh(x))
plt.title("Tanh")

plt.tight_layout()
plt.show()
```

Slide 8: Learning Rate

The learning rate is a crucial hyperparameter in neural network training. It determines the step size at each iteration while moving toward a minimum of the loss function. A proper learning rate is essential for efficient convergence.

```python
import matplotlib.pyplot as plt

def quadratic(x):
    return x**2 + 2*x + 1

def gradient(x):
    return 2*x + 2

def gradient_descent(start, learn_rate, n_iter=100):
    x = start
    path = [x]
    for _ in range(n_iter):
        x = x - learn_rate * gradient(x)
        path.append(x)
    return np.array(path)

x = np.linspace(-5, 3, 100)
y = quadratic(x)

plt.figure(figsize=(12, 4))

for i, lr in enumerate([0.01, 0.1, 0.5]):
    path = gradient_descent(start=3, learn_rate=lr)
    
    plt.subplot(1, 3, i+1)
    plt.plot(x, y, 'r-')
    plt.plot(path, quadratic(path), 'bo-')
    plt.title(f"Learning rate: {lr}")
    plt.xlabel("x")
    plt.ylabel("f(x)")

plt.tight_layout()
plt.show()
```

Slide 9: Vanishing and Exploding Gradients

Vanishing and exploding gradients are common problems in deep neural networks. They occur when gradients become extremely small or large during backpropagation, making it difficult for the network to learn effectively.

```python
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

x = np.linspace(-10, 10, 1000)

plt.figure(figsize=(12, 4))

plt.subplot(121)
plt.plot(x, sigmoid_derivative(x), label='Sigmoid derivative')
plt.title("Vanishing Gradient (Sigmoid)")
plt.legend()

plt.subplot(122)
plt.plot(x, relu_derivative(x), label='ReLU derivative')
plt.title("Potential Exploding Gradient (ReLU)")
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 10: Batch Normalization

Batch Normalization is a technique to improve the stability and performance of neural networks. It normalizes the inputs of each layer, reducing internal covariate shift and allowing higher learning rates.

```python

def batch_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    return out

# Example usage
batch_size, features = 4, 3
x = np.random.randn(batch_size, features)
gamma = np.ones((features,))
beta = np.zeros((features,))

normalized = batch_norm(x, gamma, beta)

print("Original data:")
print(x)
print("\nNormalized data:")
print(normalized)
```

Slide 11: Regularization

Regularization techniques help prevent overfitting in neural networks by adding a penalty term to the loss function. Common methods include L1 and L2 regularization.

```python
import matplotlib.pyplot as plt

def model(x, w):
    return np.dot(x, w)

def loss(y, y_pred):
    return np.mean((y - y_pred)**2)

def l1_reg(w, alpha):
    return alpha * np.sum(np.abs(w))

def l2_reg(w, alpha):
    return alpha * np.sum(w**2)

# Generate synthetic data
np.random.seed(42)
x = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * x + 1 + np.random.normal(0, 1, x.shape)

# Train models with different regularization
w_no_reg = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
w_l1 = np.linalg.inv(x.T.dot(x) + 0.1 * np.eye(x.shape[1])).dot(x.T).dot(y)
w_l2 = np.linalg.inv(x.T.dot(x) + 0.1 * np.eye(x.shape[1])).dot(x.T).dot(y)

plt.scatter(x, y, alpha=0.5)
plt.plot(x, model(x, w_no_reg), label='No regularization')
plt.plot(x, model(x, w_l1), label='L1 regularization')
plt.plot(x, model(x, w_l2), label='L2 regularization')
plt.legend()
plt.show()
```

Slide 12: Dropout

Dropout is a regularization technique that randomly sets a fraction of input units to 0 at each update during training, which helps prevent overfitting. It forces the network to learn more robust features.

```python

def dropout(x, keep_prob):
    mask = np.random.binomial(1, keep_prob, size=x.shape) / keep_prob
    return x * mask

# Example usage
x = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10]])

keep_prob = 0.8
dropped = dropout(x, keep_prob)

print("Original input:")
print(x)
print("\nAfter dropout:")
print(dropped)
```

Slide 13: Real-life Example: Image Classification

Image classification is a common application of neural networks. Here's a simple example using the MNIST dataset of handwritten digits.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=128)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")
```

Slide 14: Real-life Example: Natural Language Processing

Natural Language Processing (NLP) is another area where neural networks excel. Here's a simple sentiment analysis example using a basic neural network.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Sample data
texts = [
    "I love this product",
    "This movie was terrible",
    "The weather is nice today",
    "I'm feeling neutral about this",
    "That was an amazing experience"
]
labels = [1, 0, 1, 0.5, 1]  # 1: positive, 0: negative, 0.5: neutral

# Vectorize the text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=2, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy}")

# Make predictions
new_texts = ["This is wonderful", "I don't like this at all"]
new_X = vectorizer.transform(new_texts).toarray()
predictions = model.predict(new_X)
print("Predictions:", predictions)
```

Slide 15: Additional Resources

For those interested in diving deeper into backpropagation and neural networks, here are some valuable resources:

1. "Gradient-Based Learning Applied to Document Recognition" by LeCun et al. (1998) ArXiv: [https://arxiv.org/abs/1102.0183](https://arxiv.org/abs/1102.0183)
2. "Deep Learning" by Goodfellow, Bengio, and Courville Available online: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
3. "Neural Networks and Deep Learning" by Michael Nielsen Available online: [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)

These resources provide comprehensive explanations of neural network concepts, including backpropagation, and offer both theoretical and practical insights into the field of deep learning.

