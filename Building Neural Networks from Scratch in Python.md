## Building Neural Networks from Scratch in Python

Slide 1: Introduction to Neural Networks

Neural networks are computational models inspired by the human brain. They consist of interconnected nodes (neurons) that process and transmit information. Neural networks can learn from data to perform tasks like classification and regression.

```python
# Simple representation of a neuron
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def activate(self, inputs):
        return sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
```

Slide 2: Activation Functions

Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns. Common activation functions include ReLU, sigmoid, and tanh.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)
```

Slide 3: Forward Propagation

Forward propagation is the process of passing input data through the network to generate predictions. It involves matrix multiplication and applying activation functions.

```python
import numpy as np

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2
```

Slide 4: Loss Functions

Loss functions measure the difference between predicted and actual values. They guide the network in adjusting its parameters to improve performance.

```python
import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

Slide 5: Backpropagation

Backpropagation is the algorithm used to calculate gradients of the loss function with respect to the network's parameters. It's crucial for updating weights and biases during training.

```python
def backpropagation(X, Y, Z1, A1, Z2, A2, W2):
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2
```

Slide 6: Gradient Descent

Gradient descent is an optimization algorithm used to minimize the loss function by iteratively adjusting the network's parameters in the direction of steepest descent.

```python
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2
```

Slide 7: Building a Simple Neural Network Class

Let's create a basic neural network class that encapsulates the concepts we've covered so far.

```python
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))
    
    def forward(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
```

Slide 8: Training the Neural Network

Training involves iteratively performing forward propagation, calculating loss, backpropagation, and updating parameters.

```python
def train(self, X, Y, iterations, learning_rate):
    for i in range(iterations):
        A2 = self.forward(X)
        cost = self.compute_cost(A2, Y)
        dW1, db1, dW2, db2 = self.backward(X, Y, A2)
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")

def compute_cost(self, A2, Y):
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    return cost
```

Slide 9: Data Preprocessing

Proper data preprocessing is crucial for effective training. This includes normalization, handling missing values, and encoding categorical variables.

```python
def preprocess_data(X, Y):
    # Normalize features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # One-hot encode labels
    Y_encoded = np.eye(np.max(Y) + 1)[Y].T
    
    return X, Y_encoded

# Example usage
X_raw = np.random.randn(10, 100)
Y_raw = np.random.randint(0, 3, 100)
X_processed, Y_processed = preprocess_data(X_raw, Y_raw)
```

Slide 10: Implementing Mini-batch Gradient Descent

Mini-batch gradient descent is a variation of gradient descent that processes small batches of data at a time, offering a balance between computational efficiency and convergence speed.

```python
def create_mini_batches(X, Y, batch_size):
    mini_batches = []
    data = np.hstack((X, Y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    
    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1].T
        Y_mini = mini_batch[:, -1].reshape((-1, 1)).T
        mini_batches.append((X_mini, Y_mini))
    
    return mini_batches
```

Slide 11: Regularization Techniques

Regularization helps prevent overfitting by adding a penalty term to the loss function. L2 regularization (weight decay) is a common technique.

```python
def compute_cost_with_regularization(A2, Y, parameters, lambd):
    m = Y.shape[1]
    W1, W2 = parameters['W1'], parameters['W2']
    
    cross_entropy_cost = compute_cost(A2, Y)
    L2_regularization_cost = (lambd / (2 * m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    
    return cross_entropy_cost + L2_regularization_cost

def backward_propagation_with_regularization(X, Y, cache, parameters, lambd):
    m = X.shape[1]
    W1, W2 = parameters['W1'], parameters['W2']
    A1, A2 = cache['A1'], cache['A2']
    
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T) + (lambd / m) * W2
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T) + (lambd / m) * W1
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2
```

Slide 12: Dropout Regularization

Dropout is another regularization technique that randomly "drops out" a proportion of neurons during training, which helps prevent overfitting.

```python
def forward_propagation_with_dropout(X, parameters, keep_prob):
    W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1]) < keep_prob
    A1 = A1 * D1
    A1 = A1 / keep_prob
    
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {'Z1': Z1, 'A1': A1, 'D1': D1, 'Z2': Z2, 'A2': A2}
    return A2, cache
```

Slide 13: Hyperparameter Tuning

Hyperparameters are configuration settings for the neural network that are not learned during training. Proper tuning can significantly impact performance.

```python
import itertools

def grid_search(X, Y, param_grid):
    best_accuracy = 0
    best_params = None
    
    param_combinations = list(itertools.product(*param_grid.values()))
    
    for params in param_combinations:
        hidden_size, learning_rate, num_iterations = params
        
        model = SimpleNeuralNetwork(X.shape[0], hidden_size, Y.shape[0])
        model.train(X, Y, num_iterations, learning_rate)
        
        accuracy = model.evaluate(X, Y)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
    
    return best_params, best_accuracy

# Example usage
param_grid = {
    'hidden_size': [10, 20, 30],
    'learning_rate': [0.01, 0.1, 0.3],
    'num_iterations': [1000, 2000, 3000]
}

best_params, best_accuracy = grid_search(X_train, Y_train, param_grid)
print(f"Best parameters: {best_params}, Best accuracy: {best_accuracy}")
```

Slide 14: Saving and Loading Models

Saving trained models allows you to use them later without retraining. Here's a simple way to save and load neural network parameters using Python's pickle module.

```python
import pickle

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Example usage
model = SimpleNeuralNetwork(input_size, hidden_size, output_size)
model.train(X_train, Y_train, iterations, learning_rate)

save_model(model, 'neural_network_model.pkl')

loaded_model = load_model('neural_network_model.pkl')
predictions = loaded_model.forward(X_test)
```

Slide 15: Additional Resources

For further learning on neural networks and deep learning, consider exploring these peer-reviewed papers from arXiv:

1. "Deep Learning" by Yann LeCun, Yoshua Bengio, and Geoffrey Hinton arXiv:1521.00561
2. "Understanding the difficulty of training deep feedforward neural networks" by Xavier Glorot and Yoshua Bengio arXiv:1003.0485
3. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" by Nitish Srivastava et al. arXiv:1207.0580

These papers provide in-depth insights into various aspects of neural networks and can help deepen your understanding of the field.

