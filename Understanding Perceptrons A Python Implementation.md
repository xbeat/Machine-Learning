## Understanding Perceptrons A Python Implementation

Slide 1: What is a Perceptron?

A perceptron is a simple artificial neuron that forms the building block of neural networks. It takes multiple inputs, applies weights to them, sums them up, and passes the result through an activation function to produce an output.

```python
import numpy as np

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    def activate(self, x):
        return 1 if x > 0 else 0

    def predict(self, inputs):
        sum_inputs = np.dot(inputs, self.weights) + self.bias
        return self.activate(sum_inputs)
```

Slide 2: Perceptron Architecture

The perceptron consists of input nodes, weights, a bias term, and an activation function. The inputs are multiplied by their respective weights, summed together with the bias, and then passed through the activation function to produce the output.

```python
import matplotlib.pyplot as plt

def plot_perceptron():
    plt.figure(figsize=(8, 6))
    plt.scatter([0, 1], [0, 1], c='red', s=100, label='Inputs')
    plt.plot([0, 1], [0.5, 0.5], 'b--', label='Decision Boundary')
    plt.annotate('w1', xy=(0.2, 0.1))
    plt.annotate('w2', xy=(0.8, 0.1))
    plt.annotate('Σ', xy=(0.5, 0.5), fontsize=20)
    plt.annotate('Activation', xy=(0.7, 0.6))
    plt.annotate('Output', xy=(0.9, 0.5))
    plt.title('Perceptron Architecture')
    plt.legend()
    plt.axis('off')
    plt.show()

plot_perceptron()
```

Slide 3: Activation Function

The activation function introduces non-linearity into the perceptron's output. Common activation functions include the step function, sigmoid, and ReLU. Here's an implementation of these functions:

```python
import numpy as np
import matplotlib.pyplot as plt

def step(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-5, 5, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, step(x), label='Step')
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, relu(x), label='ReLU')
plt.title('Activation Functions')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 4: Training a Perceptron

Training a perceptron involves adjusting its weights and bias to minimize the error between predicted and actual outputs. The process typically uses the perceptron learning rule or gradient descent.

```python
class TrainablePerceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def predict(self, inputs):
        return np.dot(inputs, self.weights) + self.bias > 0

    def train(self, inputs, label):
        prediction = self.predict(inputs)
        error = label - prediction
        self.weights += self.learning_rate * error * inputs
        self.bias += self.learning_rate * error

# Example usage
perceptron = TrainablePerceptron(2)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

for _ in range(100):
    for inputs, label in zip(X, y):
        perceptron.train(inputs, label)
```

Slide 5: Perceptron for Binary Classification

One of the most common applications of perceptrons is binary classification. Let's implement a perceptron to classify points above or below a line.

```python
import numpy as np
import matplotlib.pyplot as plt

class BinaryClassifier:
    def __init__(self):
        self.weights = np.random.randn(2)
        self.bias = np.random.randn()

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias > 0

    def train(self, X, y, epochs=100):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                update = self.learning_rate * (yi - self.predict(xi))
                self.weights += update * xi
                self.bias += update

# Generate data
np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train and plot
classifier = BinaryClassifier()
classifier.train(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y)
x_range = np.linspace(-3, 3, 100)
decision_boundary = -classifier.weights[0] / classifier.weights[1] * x_range - classifier.bias / classifier.weights[1]
plt.plot(x_range, decision_boundary, 'r--')
plt.title('Binary Classification with Perceptron')
plt.show()
```

Slide 6: Perceptron Limitations

While perceptrons are powerful for linearly separable problems, they have limitations. They cannot solve problems that are not linearly separable, such as the XOR problem.

```python
import numpy as np
import matplotlib.pyplot as plt

# XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('XOR Problem')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# Attempt to train a perceptron
perceptron = TrainablePerceptron(2)
for _ in range(1000):
    for inputs, label in zip(X, y):
        perceptron.train(inputs, label)

# Plot decision boundary
x_range = np.linspace(-0.5, 1.5, 100)
y_range = -(perceptron.weights[0] * x_range + perceptron.bias) / perceptron.weights[1]
plt.plot(x_range, y_range, 'r--')
plt.title('Perceptron Attempt at XOR')
plt.show()
```

Slide 7: Multi-Layer Perceptrons

To overcome the limitations of single perceptrons, we can use multi-layer perceptrons (MLPs) or neural networks. These consist of multiple layers of perceptrons, allowing them to learn non-linear decision boundaries.

```python
import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden = TrainablePerceptron(input_size)
        self.output = TrainablePerceptron(hidden_size)

    def predict(self, X):
        hidden_output = self.hidden.predict(X)
        return self.output.predict(hidden_output)

    def train(self, X, y, epochs=1000):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                hidden_output = self.hidden.predict(xi)
                self.output.train(hidden_output, yi)
                self.hidden.train(xi, hidden_output)

# Train MLP on XOR
mlp = MLP(2, 2, 1)
mlp.train(X, y)

# Visualize decision boundary
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
Z = np.array([mlp.predict([x, y]) for x, y in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('MLP Decision Boundary for XOR')
plt.show()
```

Slide 8: Real-Life Example: Iris Flower Classification

Let's use a perceptron to classify Iris flowers based on their sepal length and width. We'll focus on distinguishing Setosa from other species.

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data[:, [0, 1]]  # sepal length and width
y = (iris.target == 0).astype(int)  # 1 for Setosa, 0 for others

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train perceptron
perceptron = TrainablePerceptron(2)
for _ in range(100):
    for inputs, label in zip(X_train, y_train):
        perceptron.train(inputs, label)

# Visualize results
plt.scatter(X[:, 0], X[:, 1], c=y)
x_range = np.linspace(4, 8, 100)
decision_boundary = -perceptron.weights[0] / perceptron.weights[1] * x_range - perceptron.bias / perceptron.weights[1]
plt.plot(x_range, decision_boundary, 'r--')
plt.title('Iris Setosa Classification')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# Evaluate accuracy
accuracy = np.mean(perceptron.predict(X_test) == y_test)
print(f"Accuracy: {accuracy:.2f}")
```

Slide 9: Real-Life Example: Image Edge Detection

Perceptrons can be used for simple image processing tasks like edge detection. Let's implement a basic edge detection algorithm using a perceptron.

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color

# Load and prepare image
image = color.rgb2gray(data.camera())

class EdgeDetector:
    def __init__(self):
        self.weights = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ])

    def detect_edges(self, image):
        height, width = image.shape
        edges = np.zeros((height-2, width-2))
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                patch = image[i-1:i+2, j-1:j+2]
                edges[i-1, j-1] = np.sum(patch * self.weights)
        
        return edges

# Detect edges
detector = EdgeDetector()
edge_image = detector.detect_edges(image)

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(edge_image, cmap='gray')
ax2.set_title('Edge Detected Image')
plt.show()
```

Slide 10: Perceptron vs. Logistic Regression

While perceptrons use a step function for activation, logistic regression uses a sigmoid function. This allows logistic regression to output probabilities rather than binary classifications.

```python
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        return self.predict_proba(X) > 0.5

    def train(self, X, y, learning_rate=0.1, epochs=100):
        for _ in range(epochs):
            y_pred = self.predict_proba(X)
            error = y - y_pred
            self.weights += learning_rate * np.dot(X.T, error)
            self.bias += learning_rate * np.sum(error)

# Generate data
np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train models
perceptron = TrainablePerceptron(2)
logistic = LogisticRegression(2)

for _ in range(100):
    for xi, yi in zip(X, y):
        perceptron.train(xi, yi)
logistic.train(X, y)

# Plot decision boundaries
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z_perceptron = np.array([perceptron.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
Z_logistic = logistic.predict(np.c_[xx.ravel(), yy.ravel()])

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z_perceptron.reshape(xx.shape), alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Perceptron Decision Boundary')

plt.subplot(122)
plt.contourf(xx, yy, Z_logistic.reshape(xx.shape), alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Logistic Regression Decision Boundary')

plt.show()
```

Slide 11: Perceptron Learning Rule

The perceptron learning rule is an algorithm for training a perceptron. It iteratively adjusts the weights based on the error between predicted and actual outputs.

```python
import numpy as np
import matplotlib.pyplot as plt

class PerceptronLearningRule:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate

    def predict(self, inputs):
        return np.dot(inputs, self.weights) + self.bias > 0

    def train(self, X, y, epochs=100):
        errors = []
        for epoch in range(epochs):
            total_error = 0
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                error = yi - prediction
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error
                total_error += abs(error)
            errors.append(total_error)
        return errors

# Generate linearly separable data
np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train the perceptron
perceptron = PerceptronLearningRule(2)
training_errors = perceptron.train(X, y)

# Plot training errors
plt.plot(range(len(training_errors)), training_errors)
plt.title('Perceptron Learning Rule: Training Errors')
plt.xlabel('Epoch')
plt.ylabel('Total Error')
plt.show()

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Perceptron Decision Boundary')
plt.show()
```

Slide 12: Perceptron as a Linear Classifier

Perceptrons are linear classifiers, meaning they can only separate classes with a linear decision boundary. This characteristic is both a strength and a limitation.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_linear_data(n_samples=100):
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def generate_nonlinear_data(n_samples=100):
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int)
    return X, y

def plot_data_and_boundary(X, y, perceptron, title):
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(10, 5))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title(title)
    plt.show()

# Linear data
X_linear, y_linear = generate_linear_data()
perceptron_linear = PerceptronLearningRule(2)
perceptron_linear.train(X_linear, y_linear)
plot_data_and_boundary(X_linear, y_linear, perceptron_linear, "Linear Data Classification")

# Nonlinear data
X_nonlinear, y_nonlinear = generate_nonlinear_data()
perceptron_nonlinear = PerceptronLearningRule(2)
perceptron_nonlinear.train(X_nonlinear, y_nonlinear)
plot_data_and_boundary(X_nonlinear, y_nonlinear, perceptron_nonlinear, "Nonlinear Data Classification Attempt")
```

Slide 13: Perceptron in Neural Networks

While a single perceptron has limitations, multiple perceptrons can be combined to form neural networks capable of solving complex, nonlinear problems.

```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.w1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2)
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def train(self, X, y, learning_rate=0.1, epochs=1000):
        for _ in range(epochs):
            output = self.forward(X)
            error = y - output
            d_output = error * output * (1 - output)
            error_hidden = np.dot(d_output, self.w2.T)
            d_hidden = error_hidden * self.a1 * (1 - self.a1)
            
            self.w2 += learning_rate * np.dot(self.a1.T, d_output)
            self.w1 += learning_rate * np.dot(X.T, d_hidden)

# Generate XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train neural network
nn = SimpleNeuralNetwork(2, 4, 1)
nn.train(X, y)

# Visualize decision boundary
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
Z = nn.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y.ravel())
plt.title('Neural Network Decision Boundary for XOR')
plt.show()
```

Slide 14: Perceptron Applications

Perceptrons and their extensions find applications in various fields, including:

1. Image classification
2. Sentiment analysis
3. Spam detection
4. Medical diagnosis
5. Weather prediction

While simple perceptrons have limitations, they form the foundation for more complex neural networks used in these applications.

```python
# Pseudocode for a simple spam detection perceptron

def tokenize(email):
    # Convert email to lowercase and split into words
    return set(email.lower().split())

def train_spam_detector(spam_emails, ham_emails):
    vocabulary = set()
    for email in spam_emails + ham_emails:
        vocabulary.update(tokenize(email))
    
    weights = {word: 0 for word in vocabulary}
    bias = 0
    
    for email in spam_emails:
        words = tokenize(email)
        for word in words:
            weights[word] += 1
        bias += 1
    
    for email in ham_emails:
        words = tokenize(email)
        for word in words:
            weights[word] -= 1
        bias -= 1
    
    return weights, bias

def classify_email(email, weights, bias):
    words = tokenize(email)
    score = sum(weights.get(word, 0) for word in words) + bias
    return "spam" if score > 0 else "ham"

# Usage example
spam_emails = ["buy now", "limited offer", "click here"]
ham_emails = ["meeting tomorrow", "project update", "lunch plans"]

weights, bias = train_spam_detector(spam_emails, ham_emails)
new_email = "special discount, buy now!"
result = classify_email(new_email, weights, bias)
print(f"The email is classified as: {result}")
```

Slide 15: Additional Resources

For those interested in diving deeper into perceptrons and neural networks, here are some valuable resources:

1. "Neural Networks and Deep Learning" by Michael Nielsen ([http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/))
2. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville ([https://www.deeplearningbook.org/](https://www.deeplearningbook.org/))
3. ArXiv paper: "Perceptron Learning with Random Coordinate Descent" by Ling Li ([https://arxiv.org/abs/1811.01322](https://arxiv.org/abs/1811.01322))
4. ArXiv paper: "The Perceptron Algorithm: A Study of its Asymptotic Convergence and its Resistance to Noise" by Shai Shalev-Shwartz and Yoram Singer ([https://arxiv.org/abs/0904.3837](https://arxiv.org/abs/0904.3837))

These resources provide in-depth explanations and mathematical foundations of perceptrons and their role in modern machine learning.

