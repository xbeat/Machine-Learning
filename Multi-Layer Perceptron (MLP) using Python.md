## Multi-Layer Perceptron (MLP) using Python

Slide 1: Introduction to Multi-Layer Perceptron (MLP)

A Multi-Layer Perceptron is a type of feedforward artificial neural network. It consists of at least three layers of nodes: an input layer, one or more hidden layers, and an output layer. Each node is a neuron that uses a nonlinear activation function. MLPs are widely used for supervised learning tasks, such as classification and regression.

```python
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Create a simple MLP structure
input_layer = np.random.randn(4)
hidden_layer = sigmoid(np.dot(np.random.randn(3, 4), input_layer))
output_layer = sigmoid(np.dot(np.random.randn(2, 3), hidden_layer))

# Visualize the MLP structure
plt.figure(figsize=(10, 6))
plt.scatter([1]*4, range(4), s=100, c='r', label='Input Layer')
plt.scatter([2]*3, range(3), s=100, c='b', label='Hidden Layer')
plt.scatter([3]*2, range(2), s=100, c='g', label='Output Layer')
plt.title('Simple MLP Structure')
plt.legend()
plt.axis('off')
plt.show()
```

Slide 2: Neuron Anatomy

A neuron in an MLP is the basic unit of computation. It receives inputs, applies weights to them, sums them up, and passes the result through an activation function. The output of this process becomes the input for the next layer or the final output of the network.

```python

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
    
    def activate(self, inputs):
        return np.dot(self.weights, inputs) + self.bias

    def output(self, inputs):
        return sigmoid(self.activate(inputs))

# Example usage
neuron = Neuron(3)
inputs = np.array([0.5, 0.3, 0.2])
print(f"Neuron output: {neuron.output(inputs)}")
```

Slide 3: Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns. Common activation functions include sigmoid, tanh, and ReLU. Each has its own characteristics and use cases.

```python
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-5, 5, 100)

plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.plot(x, relu(x), label='ReLU')
plt.title('Common Activation Functions')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 4: Forward Propagation

Forward propagation is the process of passing input data through the network to generate an output. Each layer receives inputs, applies weights and biases, and passes the result through an activation function.

```python

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class MLP:
    def __init__(self, layer_sizes):
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
    
    def forward_propagation(self, x):
        for w, b in zip(self.weights, self.biases):
            x = sigmoid(np.dot(w, x) + b)
        return x

# Example usage
mlp = MLP([3, 4, 2])
input_data = np.array([[0.5], [0.3], [0.2]])
output = mlp.forward_propagation(input_data)
print(f"Network output: {output.flatten()}")
```

Slide 5: Backpropagation

Backpropagation is the algorithm used to train MLPs. It calculates the gradient of the loss function with respect to the network's weights, allowing for weight updates that minimize the loss.

```python

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class MLP:
    # ... (previous implementation) ...

    def backpropagation(self, x, y):
        # Forward pass
        activations = [x]
        for w, b in zip(self.weights, self.biases):
            x = sigmoid(np.dot(w, x) + b)
            activations.append(x)
        
        # Backward pass
        delta = (activations[-1] - y) * sigmoid_derivative(activations[-1])
        gradients = []
        for l in range(len(self.weights) - 1, -1, -1):
            gradients.append((np.dot(delta, activations[l].T), delta))
            delta = np.dot(self.weights[l].T, delta) * sigmoid_derivative(activations[l])
        
        return gradients[::-1]

# Example usage
mlp = MLP([2, 3, 1])
x = np.array([[0.5], [0.3]])
y = np.array([[0.7]])
gradients = mlp.backpropagation(x, y)
print("Gradients for each layer:", gradients)
```

Slide 6: Training Process

Training an MLP involves iteratively updating the weights and biases to minimize the loss function. This process typically includes forward propagation, loss calculation, backpropagation, and weight updates.

```python

class MLP:
    # ... (previous implementation) ...

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for x, y_true in zip(X, y):
                # Forward propagation
                y_pred = self.forward_propagation(x)
                
                # Compute loss
                loss = mse_loss(y_true, y_pred)
                total_loss += loss
                
                # Backpropagation
                gradients = self.backpropagation(x, y_true)
                
                # Update weights and biases
                for l in range(len(self.weights)):
                    self.weights[l] -= learning_rate * gradients[l][0]
                    self.biases[l] -= learning_rate * gradients[l][1]
            
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X)}")

# Example usage
X = np.array([[[0.5], [0.3]], [[0.1], [0.8]], [[0.9], [0.2]]])
y = np.array([[[0.7]], [[0.2]], [[0.8]]])
mlp = MLP([2, 3, 1])
mlp.train(X, y, epochs=100, learning_rate=0.1)
```

Slide 7: Hyperparameter Tuning

Hyperparameters are configuration settings for the MLP that are not learned during training. They include the number of hidden layers, neurons per layer, learning rate, and activation functions. Proper tuning can significantly impact model performance.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Define hyperparameter grid
param_grid = {
    'hidden_layer_sizes': [(10,), (20,), (30,), (10, 10), (20, 10)],
    'activation': ['relu', 'tanh', 'logistic'],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.01, 0.1]
}

# Create MLPClassifier
mlp = MLPClassifier(max_iter=1000)

# Perform grid search
grid_search = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

Slide 8: Regularization Techniques

Regularization helps prevent overfitting in MLPs. Common techniques include L1 and L2 regularization, dropout, and early stopping. These methods add constraints to the learning process, encouraging the model to generalize better.

```python
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# No regularization
mlp_no_reg = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp_no_reg.fit(X_train, y_train)
print("No regularization accuracy:", accuracy_score(y_test, mlp_no_reg.predict(X_test)))

# L2 regularization
mlp_l2 = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, alpha=0.01, random_state=42)
mlp_l2.fit(X_train, y_train)
print("L2 regularization accuracy:", accuracy_score(y_test, mlp_l2.predict(X_test)))

# Early stopping
mlp_early_stopping = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, early_stopping=True, random_state=42)
mlp_early_stopping.fit(X_train, y_train)
print("Early stopping accuracy:", accuracy_score(y_test, mlp_early_stopping.predict(X_test)))
```

Slide 9: Handling Multi-class Classification

MLPs can be used for multi-class classification problems. The output layer typically uses the softmax activation function to produce probability distributions over the classes.

```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the MLP
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Visualize decision boundaries
import matplotlib.pyplot as plt

def plot_decision_boundary(X, y, model, ax=None):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')

plt.figure(figsize=(12, 5))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

plot_decision_boundary(X[:, [0, 2]], y, mlp, ax=ax1)
ax1.set_title("Decision Boundary (Sepal Length vs Petal Length)")

plot_decision_boundary(X[:, [1, 3]], y, mlp, ax=ax2)
ax2.set_title("Decision Boundary (Sepal Width vs Petal Width)")

plt.tight_layout()
plt.show()
```

Slide 10: MLPs for Regression

MLPs can also be used for regression tasks. The main difference is in the output layer, which typically uses a linear activation function instead of softmax.

```python
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the MLP
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Print metrics
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print("R2 score: ", r2_score(y_test, y_pred))

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black', label='Ground truth')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predictions')
plt.title("MLP Regression: Sine Function with Noise")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

One common application of MLPs is in handwritten digit recognition. We'll use the MNIST dataset to train an MLP for this task.

```python
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # Normalize pixel values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the MLP
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=20, random_state=42)
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Print accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Visualize some predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]}, True: {y_test[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 12: Real-life Example: Customer Churn Prediction

Another practical application of MLPs is predicting customer churn in a business context. This example demonstrates how to use an MLP for binary classification.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Generate synthetic customer data
np.random.seed(42)
n_samples = 1000
age = np.random.normal(40, 10, n_samples)
tenure = np.random.poisson(5, n_samples)
num_products = np.random.randint(1, 5, n_samples)
balance = np.random.exponential(1000, n_samples)
churn = (0.3 * age + 0.5 * tenure + 0.1 * num_products - 0.001 * balance + np.random.normal(0, 5, n_samples) > 25).astype(int)

X = np.column_stack((age, tenure, num_products, balance))
y = churn

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the MLP
mlp = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Make predictions
y_pred = mlp.predict(X_test_scaled)

# Print classification report
print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))
```

Slide 13: Advantages and Limitations of MLPs

Advantages:

1. Can learn complex, non-linear relationships in data
2. Versatile, applicable to various tasks (classification, regression, etc.)
3. Can handle high-dimensional data effectively

Limitations:

1. Prone to overfitting, especially with small datasets
2. Sensitive to feature scaling
3. Requires careful hyperparameter tuning

Slide 14: Advantages and Limitations of MLPs

```python
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Generate data
X = np.linspace(-5, 5, 200).reshape(-1, 1)
y = np.sin(X).ravel()

# Create two MLPs: one with appropriate complexity, one overfitted
mlp_good = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
mlp_overfit = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit the models
mlp_good.fit(X_scaled, y)
mlp_overfit.fit(X_scaled, y)

# Plot the results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X, y, color='black', label='True function')
plt.plot(X, mlp_good.predict(X_scaled), color='blue', label='MLP prediction')
plt.title('Appropriate Complexity')
plt.legend()

plt.subplot(122)
plt.scatter(X, y, color='black', label='True function')
plt.plot(X, mlp_overfit.predict(X_scaled), color='red', label='Overfitted MLP')
plt.title('Overfitted Model')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 15: Future Directions and Advanced Concepts

While MLPs are powerful, there are more advanced neural network architectures:

1. Convolutional Neural Networks (CNNs) for image processing
2. Recurrent Neural Networks (RNNs) for sequential data
3. Transformer models for natural language processing

Ongoing research areas include:

1. Improving training efficiency
2. Developing more interpretable models
3. Exploring neural architecture search

Slide 16: Future Directions and Advanced Concepts

```python
class CNN:
    def __init__(self):
        self.conv1 = Convolution2D(filters=32, kernel_size=3, activation='relu')
        self.pool1 = MaxPooling2D(pool_size=2)
        self.conv2 = Convolution2D(filters=64, kernel_size=3, activation='relu')
        self.pool2 = MaxPooling2D(pool_size=2)
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

# Note: This is pseudocode and won't run as-is
```

Slide 17: Additional Resources

For those interested in diving deeper into MLPs and neural networks, here are some valuable resources:

1. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016) ArXiv: [https://arxiv.org/abs/1607.06952](https://arxiv.org/abs/1607.06952)
2. "Neural Networks and Deep Learning" by Michael Nielsen (2015) Available online: [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)
3. "Efficient BackProp" by Yann LeCun et al. (1998) ArXiv: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)
4. "Practical Recommendations for Gradient-Based Training of Deep Architectures" by Yoshua Bengio (2012) ArXiv: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)

These resources provide in-depth explanations of the concepts we've covered and explore advanced topics in neural network research and applications.

