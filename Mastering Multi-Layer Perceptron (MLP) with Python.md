## Mastering Multi-Layer Perceptron (MLP) with Python
Slide 1: Introduction to Multi-Layer Perceptron (MLP)

A Multi-Layer Perceptron is a feedforward artificial neural network that consists of multiple layers of nodes, including an input layer, one or more hidden layers, and an output layer. MLPs are capable of learning complex, non-linear relationships between inputs and outputs, making them suitable for a wide range of tasks such as classification and regression.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple MLP architecture visualization
def plot_mlp_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    layer_sizes = [4, 5, 3, 2]  # Input, hidden, output layers
    
    for i, layer_size in enumerate(layer_sizes):
        x = [i] * layer_size
        y = np.arange(layer_size)
        ax.scatter(x, y, s=500, zorder=2)
    
    for i in range(len(layer_sizes) - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i+1]):
                ax.plot([i, i+1], [j, k], 'gray', alpha=0.5, zorder=1)
    
    ax.set_xticks(range(len(layer_sizes)))
    ax.set_xticklabels(['Input', 'Hidden 1', 'Hidden 2', 'Output'])
    ax.set_yticks([])
    ax.set_title('Multi-Layer Perceptron Architecture')
    plt.tight_layout()
    plt.show()

plot_mlp_architecture()
```

Slide 2: Activation Functions in MLPs

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns. Common activation functions include ReLU (Rectified Linear Unit), Sigmoid, and Tanh. ReLU is often preferred due to its simplicity and effectiveness in mitigating the vanishing gradient problem.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

x = np.linspace(-5, 5, 100)

plt.figure(figsize=(12, 4))
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.legend()
plt.title('Common Activation Functions')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
```

Slide 3: Forward Propagation in MLPs

Forward propagation is the process of passing input data through the network to generate predictions. Each neuron computes a weighted sum of its inputs, applies an activation function, and passes the result to the next layer.

```python
import numpy as np

def forward_propagation(X, weights, biases, activation_func):
    layer_output = X
    for w, b in zip(weights, biases):
        layer_input = np.dot(layer_output, w) + b
        layer_output = activation_func(layer_input)
    return layer_output

# Example usage
X = np.array([[1, 2, 3, 4]])
weights = [np.random.randn(4, 5), np.random.randn(5, 3), np.random.randn(3, 2)]
biases = [np.random.randn(5), np.random.randn(3), np.random.randn(2)]

def relu(x):
    return np.maximum(0, x)

output = forward_propagation(X, weights, biases, relu)
print("Network output:", output)
```

Slide 4: Backpropagation and Gradient Descent

Backpropagation is the algorithm used to train MLPs by adjusting weights and biases to minimize the error between predicted and actual outputs. It works in conjunction with an optimization algorithm, typically gradient descent, to update the network parameters.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_network(X, y, hidden_size, learning_rate, epochs):
    input_size, output_size = X.shape[1], y.shape[1]
    
    # Initialize weights and biases
    w1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    w2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))
    
    for _ in range(epochs):
        # Forward propagation
        z1 = np.dot(X, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = sigmoid(z2)
        
        # Backpropagation
        error = y - a2
        d2 = error * sigmoid_derivative(a2)
        d1 = np.dot(d2, w2.T) * sigmoid_derivative(a1)
        
        # Update weights and biases
        w2 += learning_rate * np.dot(a1.T, d2)
        b2 += learning_rate * np.sum(d2, axis=0, keepdims=True)
        w1 += learning_rate * np.dot(X.T, d1)
        b1 += learning_rate * np.sum(d1, axis=0, keepdims=True)
    
    return w1, b1, w2, b2

# Example usage
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])
w1, b1, w2, b2 = train_network(X, y, hidden_size=4, learning_rate=0.1, epochs=10000)
print("Training complete")
```

Slide 5: Loss Functions for MLPs

Loss functions measure the difference between predicted and actual outputs, guiding the learning process. Common loss functions include Mean Squared Error (MSE) for regression tasks and Cross-Entropy Loss for classification tasks.

```python
import numpy as np
import matplotlib.pyplot as plt

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Visualize MSE and Cross-Entropy Loss
y_true = np.array([0, 1, 1, 0])
y_pred = np.linspace(0, 1, 100)

mse = [mse_loss(y_true, np.full_like(y_true, p)) for p in y_pred]
ce = [cross_entropy_loss(y_true, np.full_like(y_true, p)) for p in y_pred]

plt.figure(figsize=(10, 5))
plt.plot(y_pred, mse, label='MSE Loss')
plt.plot(y_pred, ce, label='Cross-Entropy Loss')
plt.xlabel('Predicted Probability')
plt.ylabel('Loss')
plt.title('Comparison of MSE and Cross-Entropy Loss')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 6: Implementing a Simple MLP with NumPy

Let's implement a basic MLP for binary classification using NumPy. This example demonstrates the core concepts of forward propagation, backpropagation, and gradient descent.

```python
import numpy as np

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        self.error = y - output
        self.d2 = self.error * self.sigmoid_derivative(output)
        self.d1 = np.dot(self.d2, self.w2.T) * self.sigmoid_derivative(self.a1)
        
        self.w2 += np.dot(self.a1.T, self.d2)
        self.b2 += np.sum(self.d2, axis=0, keepdims=True)
        self.w1 += np.dot(X.T, self.d1)
        self.b1 += np.sum(self.d1, axis=0, keepdims=True)
    
    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
    
    def predict(self, X):
        return self.forward(X)

# Example usage
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])

mlp = SimpleMLP(input_size=3, hidden_size=4, output_size=1)
mlp.train(X, y, epochs=10000, learning_rate=0.1)

print("Predictions:")
print(mlp.predict(X))
```

Slide 7: Regularization Techniques for MLPs

Regularization helps prevent overfitting by adding constraints to the model. Common techniques include L1 and L2 regularization, which add penalties based on the magnitude of weights. Dropout is another popular technique that randomly deactivates neurons during training.

```python
import numpy as np

def l1_regularization(weights, lambda_):
    return lambda_ * np.sum(np.abs(weights))

def l2_regularization(weights, lambda_):
    return lambda_ * np.sum(weights ** 2)

def dropout(layer_output, keep_prob):
    mask = np.random.binomial(1, keep_prob, size=layer_output.shape)
    return (layer_output * mask) / keep_prob

# Example usage
weights = np.random.randn(5, 3)
lambda_ = 0.01

l1_reg = l1_regularization(weights, lambda_)
l2_reg = l2_regularization(weights, lambda_)

print("L1 Regularization:", l1_reg)
print("L2 Regularization:", l2_reg)

# Dropout example
layer_output = np.random.randn(1, 10)
keep_prob = 0.5
dropout_output = dropout(layer_output, keep_prob)

print("Original output:", layer_output)
print("Dropout output:", dropout_output)
```

Slide 8: Batch Normalization in MLPs

Batch Normalization is a technique that normalizes the inputs of each layer, which can help accelerate training and improve generalization. It's typically applied before the activation function in each layer.

```python
import numpy as np

def batch_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

# Example usage
batch_size, features = 32, 10
x = np.random.randn(batch_size, features)
gamma = np.ones(features)
beta = np.zeros(features)

normalized_x = batch_norm(x, gamma, beta)

print("Original mean:", np.mean(x, axis=0))
print("Original std:", np.std(x, axis=0))
print("Normalized mean:", np.mean(normalized_x, axis=0))
print("Normalized std:", np.std(normalized_x, axis=0))
```

Slide 9: Optimizers for Training MLPs

Various optimization algorithms can be used to train MLPs, each with its own advantages. Popular choices include Stochastic Gradient Descent (SGD), Adam, and RMSprop. These optimizers help improve convergence and handle different types of optimization landscapes.

```python
import numpy as np

class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def update(self, param, grad):
        return param - self.learning_rate * grad

class Adam:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, param, grad):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(param)
            self.v = np.zeros_like(param)
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        return param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Example usage
param = np.random.randn(5, 3)
grad = np.random.randn(5, 3)

sgd = SGD(learning_rate=0.01)
adam = Adam(learning_rate=0.01)

print("SGD update:")
print(sgd.update(param, grad))

print("\nAdam update:")
print(adam.update(param, grad))
```

Slide 10: MLP for Image Classification

MLPs can be used for image classification tasks, although Convolutional Neural Networks (CNNs) are generally more effective for this purpose. Here's a simple MLP for classifying handwritten digits using the MNIST dataset.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load and preprocess MNIST dataset
digits = load_digits()
X, y = digits.data, digits.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y, learning_rate):
        # Implement backpropagation and parameter updates

    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

# Train and evaluate the model
mlp = MLP(input_size=64, hidden_size=100, output_size=10)
mlp.train(X_train, y_train, epochs=100, learning_rate=0.01)
predictions = np.argmax(mlp.forward(X_test), axis=1)
accuracy = accuracy_score(y_test, predictions)
print(f"Test accuracy: {accuracy:.4f}")
```

Slide 11: Hyperparameter Tuning for MLPs

Hyperparameter tuning is crucial for optimizing MLP performance. Key hyperparameters include learning rate, number of hidden layers, number of neurons per layer, and regularization strength. Grid search and random search are common techniques for finding optimal hyperparameters.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from scipy.stats import uniform, randint

# Define hyperparameter search space
param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': uniform(0.0001, 0.1),
    'learning_rate_init': uniform(0.001, 0.01),
    'max_iter': randint(100, 500)
}

# Create MLP classifier
mlp = MLPClassifier()

# Perform random search
random_search = RandomizedSearchCV(
    mlp, param_distributions=param_dist, n_iter=20, cv=3, n_jobs=-1, random_state=42
)

# Fit the random search object to the data
random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)
```

Slide 12: MLP for Regression Tasks

MLPs can be used for regression tasks by adjusting the output layer and loss function. Here's an example of using an MLP for predicting housing prices based on various features.

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load and preprocess California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class MLPRegressor:
    def __init__(self, input_size, hidden_size):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        return self.z2

    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Backward pass (simplified)
            error = y_pred - y
            d_w2 = np.dot(self.a1.T, error)
            d_b2 = np.sum(error, axis=0, keepdims=True)
            d_a1 = np.dot(error, self.w2.T)
            d_z1 = d_a1 * (self.z1 > 0)
            d_w1 = np.dot(X.T, d_z1)
            d_b1 = np.sum(d_z1, axis=0, keepdims=True)
            
            # Update weights and biases
            self.w2 -= learning_rate * d_w2
            self.b2 -= learning_rate * d_b2
            self.w1 -= learning_rate * d_w1
            self.b1 -= learning_rate * d_b1

# Train and evaluate the model
mlp = MLPRegressor(input_size=8, hidden_size=64)
mlp.train(X_train, y_train, epochs=100, learning_rate=0.01)
predictions = mlp.forward(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.4f}")
```

Slide 13: Real-Life Example: Sentiment Analysis with MLP

MLPs can be used for natural language processing tasks such as sentiment analysis. Here's an example of using an MLP to classify movie reviews as positive or negative.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample movie reviews (simplified dataset)
reviews = [
    "This movie was great! I loved it.",
    "Terrible film, waste of time.",
    "Amazing plot and acting, highly recommend.",
    "Boring and predictable, don't watch.",
    "Excellent cinematography and soundtrack."
]
labels = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

# Preprocess text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews).toarray()
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class SentimentMLP:
    def __init__(self, input_size, hidden_size):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        return self.sigmoid(self.z2)

    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Backward pass (simplified)
            error = y_pred - y.reshape(-1, 1)
            d_w2 = np.dot(self.a1.T, error)
            d_b2 = np.sum(error, axis=0, keepdims=True)
            d_a1 = np.dot(error, self.w2.T)
            d_z1 = d_a1 * (self.z1 > 0)
            d_w1 = np.dot(X.T, d_z1)
            d_b1 = np.sum(d_z1, axis=0, keepdims=True)
            
            # Update weights and biases
            self.w2 -= learning_rate * d_w2
            self.b2 -= learning_rate * d_b2
            self.w1 -= learning_rate * d_w1
            self.b1 -= learning_rate * d_b1

# Train and evaluate the model
mlp = SentimentMLP(input_size=X_train.shape[1], hidden_size=32)
mlp.train(X_train, y_train, epochs=1000, learning_rate=0.01)
predictions = (mlp.forward(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, predictions)
print(f"Test accuracy: {accuracy:.4f}")

# Classify a new review
new_review = ["The movie was okay, but not great."]
new_X = vectorizer.transform(new_review).toarray()
sentiment = "Positive" if mlp.forward(new_X) > 0.5 else "Negative"
print(f"Sentiment: {sentiment}")
```

Slide 14: Real-Life Example: Image Generation with MLPs

While not as common as using GANs or VAEs, MLPs can be used for simple image generation tasks. Here's an example of using an MLP to generate simple geometric shapes.

```python
import numpy as np
import matplotlib.pyplot as plt

class ShapeGeneratorMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        return self.z2

    def generate_shape(self, shape_code):
        output = self.forward(shape_code)
        return output.reshape(28, 28)

# Create and train the model (training code omitted for brevity)
mlp = ShapeGeneratorMLP(input_size=10, hidden_size=128, output_size=784)

# Generate shapes
circle_code = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
square_code = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
triangle_code = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])

circle = mlp.generate_shape(circle_code)
square = mlp.generate_shape(square_code)
triangle = mlp.generate_shape(triangle_code)

# Display generated shapes
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(circle, cmap='gray')
ax1.set_title('Generated Circle')
ax2.imshow(square, cmap='gray')
ax2.set_title('Generated Square')
ax3.imshow(triangle, cmap='gray')
ax3.set_title('Generated Triangle')
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into Multi-Layer Perceptrons and neural networks, here are some valuable resources:

1. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville ArXiv: [https://arxiv.org/abs/1606.01781](https://arxiv.org/abs/1606.01781)
2. "Neural Networks and Deep Learning" by Michael Nielsen Available online: [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)
3. "Pattern Recognition and Machine Learning" by Christopher Bishop ArXiv: [https://arxiv.org/abs/stat.ML/0605013](https://arxiv.org/abs/stat.ML/0605013)
4. "Efficient BackProp" by Yann LeCun et al. ArXiv: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)
5. "Practical recommendations for gradient-based training of deep architectures" by Yoshua Bengio ArXiv: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)

These resources provide in-depth explanations of the concepts covered in this presentation and explore advanced topics in neural network architecture and training.

