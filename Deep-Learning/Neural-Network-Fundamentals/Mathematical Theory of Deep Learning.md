## Mathematical Theory of Deep Learning

Slide 1: Introduction to Mathematical Theory of Deep Learning

Deep learning has revolutionized machine learning, achieving remarkable success in various domains. This mathematical theory, developed by Philipp Petersen and Jakob Zech, provides a rigorous foundation for understanding the capabilities and limitations of deep neural networks. We'll explore key concepts, starting with the basics of feedforward neural networks and progressing to advanced topics like generalization and robustness.

Slide 2: Introduction to Mathematical Theory of Deep Learning

```python
import matplotlib.pyplot as plt

# Simple visualization of a deep neural network
def plot_neural_network():
    layers = [4, 5, 6, 5, 3]  # Number of neurons in each layer
    layer_sizes = len(layers)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    left = 0.1
    right = 0.9
    bottom = 0.1
    top = 0.9
    
    v_spacing = (top - bottom)/float(max(layers))
    h_spacing = (right - left)/float(len(layers) - 1)
    
    # Nodes
    for n, layer_size in enumerate(layers):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layers[:-1], layers[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k', alpha=0.5)
                ax.add_artist(line)
    
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.title('Deep Neural Network Architecture')
    plt.show()

plot_neural_network()
```

Slide 3: Feedforward Neural Networks

Feedforward neural networks form the backbone of deep learning. These networks consist of layers of interconnected neurons, where information flows in one direction from input to output. Each neuron applies a nonlinear activation function to a weighted sum of its inputs, allowing the network to learn complex patterns and representations.

Slide 4: Feedforward Neural Networks

```python

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
    
    def activate(self, inputs):
        return np.dot(self.weights, inputs) + self.bias

class ActivationFunction:
    @staticmethod
    def relu(x):
        return max(0, x)

class FeedforwardNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(1, len(layer_sizes)):
            layer = [Neuron(layer_sizes[i-1]) for _ in range(layer_sizes[i])]
            self.layers.append(layer)
    
    def forward(self, inputs):
        for layer in self.layers:
            next_inputs = []
            for neuron in layer:
                activation = neuron.activate(inputs)
                next_inputs.append(ActivationFunction.relu(activation))
            inputs = next_inputs
        return inputs

# Example usage
nn = FeedforwardNeuralNetwork([2, 3, 1])
input_data = [1, 2]
output = nn.forward(input_data)
print(f"Input: {input_data}")
print(f"Output: {output}")
```

Slide 5: Universal Approximation

The Universal Approximation Theorem is a fundamental result in neural network theory. It states that a feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of R^n to any desired degree of accuracy. This theorem provides theoretical justification for the expressive power of neural networks.

```python
import matplotlib.pyplot as plt

def universal_approximation_demo():
    # Generate data from a complex function
    x = np.linspace(-5, 5, 200)
    y = np.sin(x) + 0.5 * np.cos(2 * x)
    
    # Simple neural network approximation
    class SimpleNN:
        def __init__(self, num_hidden):
            self.w1 = np.random.randn(1, num_hidden)
            self.b1 = np.random.randn(num_hidden)
            self.w2 = np.random.randn(num_hidden, 1)
            self.b2 = np.random.randn(1)
        
        def forward(self, x):
            h = np.maximum(0, np.dot(x.reshape(-1, 1), self.w1) + self.b1)
            return np.dot(h, self.w2) + self.b2
    
    # Train a simple neural network
    nn = SimpleNN(20)
    learning_rate = 0.01
    for _ in range(10000):
        y_pred = nn.forward(x)
        loss = np.mean((y_pred - y.reshape(-1, 1))**2)
        grad = 2 * (y_pred - y.reshape(-1, 1)) / len(x)
        nn.w2 -= learning_rate * np.dot(nn.forward(x).T, grad).T
        nn.b2 -= learning_rate * np.sum(grad)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='True function')
    plt.plot(x, nn.forward(x), label='NN approximation')
    plt.legend()
    plt.title('Universal Approximation Demonstration')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

universal_approximation_demo()
```

Slide 6: Splines and Neural Networks

Splines are piecewise polynomial functions used for interpolation and approximation. In the context of neural networks, splines provide a useful analogy for understanding how networks can approximate complex functions. ReLU (Rectified Linear Unit) neural networks, in particular, can be seen as adaptively creating piecewise linear approximations of target functions.

```python
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def compare_spline_and_nn():
    # Generate data
    x = np.linspace(0, 10, 20)
    y = np.sin(x) + np.random.normal(0, 0.1, x.shape)
    
    # Cubic spline interpolation
    cs = CubicSpline(x, y)
    xs = np.linspace(0, 10, 200)
    ys_spline = cs(xs)
    
    # Simple neural network (piecewise linear approximation)
    class SimpleNN:
        def __init__(self, num_hidden):
            self.w1 = np.random.randn(1, num_hidden)
            self.b1 = np.random.randn(num_hidden)
            self.w2 = np.random.randn(num_hidden, 1)
            self.b2 = np.random.randn(1)
        
        def forward(self, x):
            h = np.maximum(0, np.dot(x.reshape(-1, 1), self.w1) + self.b1)
            return np.dot(h, self.w2) + self.b2
    
    nn = SimpleNN(20)
    # (Training code omitted for brevity)
    
    ys_nn = nn.forward(xs)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data points')
    plt.plot(xs, ys_spline, label='Cubic spline')
    plt.plot(xs, ys_nn, label='Neural network')
    plt.legend()
    plt.title('Comparison of Spline and Neural Network Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

compare_spline_and_nn()
```

Slide 7: ReLU Neural Networks

ReLU (Rectified Linear Unit) activation functions have become popular in deep learning due to their simplicity and effectiveness. ReLU networks create piecewise linear approximations of target functions, with each neuron contributing a "hinge" to the overall function. This property allows ReLU networks to efficiently approximate a wide range of functions.

```python
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def plot_relu_network():
    x = np.linspace(-5, 5, 1000)
    
    # Define weights and biases for a simple ReLU network
    w1 = np.array([[1.0, -1.0, 0.5]])
    b1 = np.array([0.0, 1.0, -1.0])
    w2 = np.array([[1.0], [1.0], [1.0]])
    b2 = np.array([0.0])
    
    # Compute the output of the network
    h = relu(np.dot(x.reshape(-1, 1), w1) + b1)
    y = np.dot(h, w2) + b2
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='ReLU Network Output')
    plt.plot(x, relu(x), '--', label='Single ReLU')
    plt.title('ReLU Neural Network Behavior')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_relu_network()
```

Slide 8: Affine Pieces for ReLU Neural Networks

ReLU neural networks partition the input space into regions where the network behaves as an affine function. The number and arrangement of these affine pieces contribute to the network's expressive power. Understanding this partitioning helps explain how ReLU networks can efficiently approximate complex functions.

```python
import matplotlib.pyplot as plt

def plot_relu_partitions():
    # Define a simple 2D ReLU network
    def relu_network(x, y):
        h1 = np.maximum(0, x + y - 1)
        h2 = np.maximum(0, x - y)
        return h1 + h2
    
    # Create a grid of points
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    
    # Compute the network output
    Z = relu_network(X, Y)
    
    # Plot the result
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Network Output')
    
    # Plot the partition boundaries
    plt.plot([-2, 3], [3, -2], 'r--', label='x + y = 1')
    plt.plot([-2, 2], [-2, 2], 'w--', label='x = y')
    
    plt.title('Affine Partitions in a 2D ReLU Network')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.axis('equal')
    plt.show()

plot_relu_partitions()
```

Slide 9: Deep ReLU Neural Networks

Deep ReLU networks, with multiple hidden layers, can create increasingly complex partitions of the input space. This hierarchical structure allows deep networks to efficiently represent functions with intricate geometries. The depth of the network plays a crucial role in its expressive power and ability to capture high-level abstractions.

```python
import matplotlib.pyplot as plt

def deep_relu_network(x, weights, biases):
    for w, b in zip(weights, biases):
        x = np.maximum(0, np.dot(x, w) + b)
    return x

def plot_deep_relu_partitions():
    # Define a deep ReLU network
    weights = [
        np.array([[1, -1], [1, 1]]),
        np.array([[1, -1], [1, 1]]),
        np.array([[1], [1]])
    ]
    biases = [
        np.array([0, -1]),
        np.array([0, 0]),
        np.array([0])
    ]
    
    # Create a grid of points
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    
    # Compute the network output
    inputs = np.stack([X, Y], axis=-1)
    Z = deep_relu_network(inputs, weights, biases).squeeze()
    
    # Plot the result
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Network Output')
    
    plt.title('Partitions in a Deep ReLU Network')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

plot_deep_relu_partitions()
```

Slide 10: High-Dimensional Approximation

Neural networks excel at approximating functions in high-dimensional spaces, where traditional methods often struggle due to the curse of dimensionality. The ability of neural networks to adapt their representation to the intrinsic structure of the data allows them to overcome many challenges associated with high-dimensional approximation.

```python
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

def high_dimensional_approximation():
    # Generate high-dimensional data
    dim = 50
    n_samples = 1000
    X = np.random.randn(n_samples, dim)
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1]) + 0.1 * np.sum(X[:, 2:], axis=1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a neural network
    nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    nn.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = nn.score(X_train, y_train)
    test_score = nn.score(X_test, y_test)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(nn.predict(X_test), y_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'High-Dimensional Approximation (dim={dim})\n'
              f'Train R² = {train_score:.4f}, Test R² = {test_score:.4f}')
    plt.show()

high_dimensional_approximation()
```

Slide 11: High-Dimensional Approximation

Neural networks excel at approximating functions in high-dimensional spaces, where traditional methods often struggle due to the curse of dimensionality. The ability of neural networks to adapt their representation to the intrinsic structure of the data allows them to overcome many challenges associated with high-dimensional approximation.

```python
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def high_dimensional_approximation(dim=50, n_samples=1000):
    # Generate high-dimensional data
    X = np.random.randn(n_samples, dim)
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1]) + 0.1 * np.sum(X[:, 2:], axis=1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a neural network
    nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    nn.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred_train = nn.predict(X_train)
    y_pred_test = nn.predict(X_test)
    train_score = r2_score(y_train, y_pred_train)
    test_score = r2_score(y_test, y_pred_test)
    
    print(f"Dimension: {dim}")
    print(f"Train R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")

high_dimensional_approximation()
```

Slide 12: Interpolation in Neural Networks

Interpolation is a crucial aspect of neural network learning, where the network aims to fit the training data perfectly. In the context of deep learning, interpolation often leads to good generalization, contrary to classical statistical wisdom. This phenomenon, known as "benign overfitting," is an active area of research in the mathematical theory of deep learning.

```python
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def interpolation_demo():
    # Generate noisy data
    np.random.seed(42)
    X = np.linspace(0, 10, 20).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])
    
    # Train a neural network to interpolate
    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=10000, alpha=0)
    model.fit(X, y)
    
    # Generate predictions
    X_test = np.linspace(0, 10, 200).reshape(-1, 1)
    y_pred = model.predict(X_test)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', label='Training data')
    plt.plot(X_test, y_pred, label='NN interpolation')
    plt.plot(X_test, np.sin(X_test), '--', label='True function')
    plt.title('Neural Network Interpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

interpolation_demo()
```

Slide 13: Training of Neural Networks

Training neural networks involves optimizing the network parameters to minimize a loss function. Gradient descent and its variants are commonly used optimization algorithms. The backpropagation algorithm efficiently computes gradients through the network layers. Understanding the dynamics of training is crucial for developing effective deep learning models.

```python

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1)
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2)
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * sigmoid_derivative(output)
        
        self.hidden_error = np.dot(self.output_delta, self.W2.T)
        self.hidden_delta = self.hidden_error * sigmoid_derivative(self.a1)
        
        self.W2 += np.dot(self.a1.T, self.output_delta)
        self.W1 += np.dot(X.T, self.hidden_delta)
    
    def train(self, X, y, epochs):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

# Usage example
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])
nn = SimpleNeuralNetwork(3, 4, 1)
nn.train(X, y, 10000)
print("Final output after training:")
print(nn.forward(X))
```

Slide 14: Wide Neural Networks

Wide neural networks, characterized by layers with a large number of neurons, have interesting theoretical properties. As the width approaches infinity, the behavior of these networks becomes more predictable and amenable to analysis. This regime, known as the "infinite-width limit," provides insights into neural network optimization and generalization.

```python
import matplotlib.pyplot as plt

def plot_wide_network_convergence():
    widths = [10, 100, 1000, 10000]
    n_samples = 1000
    
    plt.figure(figsize=(12, 8))
    for width in widths:
        # Initialize random weights
        W = np.random.randn(width, n_samples) / np.sqrt(width)
        
        # Compute the empirical distribution of outputs
        outputs = np.dot(W.T, np.random.randn(width))
        
        # Plot histogram
        plt.hist(outputs, bins=50, density=True, alpha=0.5, label=f'Width {width}')
    
    # Plot the standard normal distribution
    x = np.linspace(-3, 3, 100)
    plt.plot(x, np.exp(-x**2/2) / np.sqrt(2*np.pi), 'r-', lw=2, label='Standard Normal')
    
    plt.title('Convergence to Gaussian as Width Increases')
    plt.xlabel('Output')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

plot_wide_network_convergence()
```

Slide 15: Loss Landscape Analysis

The loss landscape of neural networks is a high-dimensional surface that represents the value of the loss function for different network parameters. Understanding the geometry of this landscape is crucial for developing effective optimization strategies. Recent research has revealed interesting properties of loss landscapes, such as the prevalence of saddle points and the existence of wide, flat minima that correlate with good generalization.

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loss_function(w1, w2):
    return np.sin(5 * w1) * np.cos(5 * w2) / 5 + w1**2 + w2**2

def plot_loss_landscape():
    w1 = np.linspace(-2, 2, 100)
    w2 = np.linspace(-2, 2, 100)
    W1, W2 = np.meshgrid(w1, w2)
    Z = loss_function(W1, W2)

    fig = plt.figure(figsize=(12, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(W1, W2, Z, cmap='viridis')
    ax1.set_xlabel('w1')
    ax1.set_ylabel('w2')
    ax1.set_zlabel('Loss')
    ax1.set_title('3D Loss Landscape')
    
    # 2D contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(W1, W2, Z, levels=20)
    ax2.set_xlabel('w1')
    ax2.set_ylabel('w2')
    ax2.set_title('Loss Landscape Contours')
    
    plt.colorbar(surf, ax=ax1, shrink=0.6, aspect=10)
    plt.colorbar(contour, ax=ax2)
    plt.tight_layout()
    plt.show()

plot_loss_landscape()
```

Slide 16: Shape of Neural Network Spaces

The space of neural networks with a given architecture forms a rich geometric structure. Understanding this shape is crucial for analyzing the behavior of optimization algorithms and the generalization properties of trained networks. Recent research has explored connections between neural network spaces and algebraic geometry, providing new insights into the expressiveness and trainability of deep learning models.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_network_space():
    # Generate random neural networks
    n_networks = 1000
    n_params = 50
    networks = np.random.randn(n_networks, n_params)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    networks_2d = pca.fit_transform(networks)
    
    # Visualize the network space
    plt.figure(figsize=(10, 8))
    plt.scatter(networks_2d[:, 0], networks_2d[:, 1], alpha=0.5)
    plt.title('Visualization of Neural Network Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), 
                 label='Density')
    plt.show()
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

visualize_network_space()
```

Slide 17: Generalization Properties of Deep Neural Networks

Generalization in deep learning refers to a model's ability to perform well on unseen data. Despite their high capacity, deep neural networks often generalize well, contradicting classical learning theory. This phenomenon has led to new theoretical frameworks, such as the study of overparameterization and implicit regularization, to explain the generalization capabilities of deep networks.

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPRegressor

def plot_learning_curves():
    # Generate synthetic data
    np.random.seed(0)
    X = np.sort(5 * np.random.rand(200, 1), axis=0)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

    # Create MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500)

    # Calculate learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        mlp, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    # Calculate mean and std
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

plot_learning_curves()
```

Slide 18: Generalization in the Overparameterized Regime

In the overparameterized regime, where the number of model parameters exceeds the number of training samples, traditional learning theory predicts poor generalization. However, deep neural networks often perform well in this regime. Recent work has shown that overparameterization can lead to simpler functions and better generalization through implicit regularization and the dynamics of gradient descent.

```python
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def overparameterized_demo():
    # Generate synthetic data
    np.random.seed(0)
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, X.shape[0])

    # Create overparameterized MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(1000,), max_iter=2000, alpha=0)

    # Fit the model
    mlp.fit(X, y)

    # Generate predictions
    X_test = np.linspace(0, 1, 1000).reshape(-1, 1)
    y_pred = mlp.predict(X_test)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Training data')
    plt.plot(X_test, y_pred, color='red', label='MLP prediction')
    plt.plot(X_test, np.sin(2 * np.pi * X_test), '--', color='green', label='True function')
    plt.title('Overparameterized Neural Network Generalization')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

overparameterized_demo()
```

Slide 19: Robustness and Adversarial Examples

Robustness is a critical aspect of neural network performance, especially in safety-critical applications. Adversarial examples, small perturbations to inputs that cause misclassification, have revealed vulnerabilities in deep learning models. Understanding and improving robustness is an active area of research, involving techniques such as adversarial training and certified defenses.

```python
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons

def adversarial_example_demo():
    # Generate synthetic data
    X, y = make_moons(n_samples=100, noise=0.1, random_state=0)

    # Train a simple neural network
    clf = MLPClassifier(hidden_layer_sizes=(10,), random_state=0)
    clf.fit(X, y)

    # Create a grid to visualize decision boundary
    xx, yy = np.meshgrid(np.linspace(-2, 3, 100), np.linspace(-1.5, 2, 100))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')

    # Generate an adversarial example
    epsilon = 0.2
    idx = 50  # Choose a sample to perturb
    grad = clf.loss_gradient(X[idx:idx+1], [y[idx]])
    X_adv = X[idx] + epsilon * grad[0] / np.linalg.norm(grad[0])

    # Plot original and adversarial examples
    plt.scatter(X[idx, 0], X[idx, 1], c='g', s=200, marker='*', edgecolor='k', label='Original')
    plt.scatter(X_adv[0], X_adv[1], c='r', s=200, marker='*', edgecolor='k', label='Adversarial')

    plt.title("Decision Boundary with Adversarial Example")
    plt.legend()
    plt.show()

    print(f"Original prediction: {clf.predict([X[idx]])[0]}")
    print(f"Adversarial prediction: {clf.predict([X_adv])[0]}")

adversarial_example_demo()
```

Slide 20: Additional Resources

For those interested in diving deeper into the mathematical theory of deep learning, here are some valuable resources:

1. ArXiv paper: "Mathematics of Deep Learning" by René Vidal, Joan Bruna, Raja Giryes, and Stefano Soatto ArXiv link: [https://arxiv.org/abs/1712.04741](https://arxiv.org/abs/1712.04741)
2. ArXiv paper: "Theoretical Insights into the Optimization and Generalization of Deep Neural Networks" by Zhanxing Zhu, Jingfeng Wu, Bing Yu, Lei Wu, and Jinwen Ma ArXiv link: [https://arxiv.org/abs/1807.11124](https://arxiv.org/abs/1807.11124)
3. ArXiv paper: "The Modern Mathematics of Deep Learning" by Julius Berner, Philipp Grohs, Gitta Kutyniok, and Philipp Petersen ArXiv link: [https://arxiv.org/abs/2105.04026](https://arxiv.org/abs/2105.04026)

These papers provide in-depth discussions on various aspects of the mathematical foundations of deep learning, including optimization, generalization, and the role of depth in neural networks.

