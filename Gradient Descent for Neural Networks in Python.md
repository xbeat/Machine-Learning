## Gradient Descent for Neural Networks in Python

Slide 1: Introduction to Gradient Descent

Gradient descent is an optimization algorithm used to find the minimum of a function by iteratively adjusting the parameters in the direction of the negative gradient. In the context of neural networks, gradient descent is used to optimize the weights and biases of the network by minimizing the cost or loss function.

Slide 2: Understanding Gradient Descent

Gradient descent works by calculating the gradients (partial derivatives) of the cost function with respect to each weight and bias in the network. These gradients represent the rate of change of the cost function for small changes in the weights and biases. The algorithm then updates the weights and biases in the opposite direction of the gradients, effectively minimizing the cost function.

Slide 3: Gradient Descent Algorithm

The general algorithm for gradient descent can be expressed as:

```python
while not converged:
    for each weight and bias:
        compute gradient of cost function
    update weights and biases using gradients
```

Slide 4: Implementing Gradient Descent in Python

To implement gradient descent in Python for a neural network, we need to define the network architecture, compute the forward pass, compute the cost function, compute the gradients using backpropagation, and update the weights and biases based on the gradients.

```python
import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the neural network architecture
input_size = 2
hidden_size = 3
output_size = 1

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))
```

Slide 5: Forward Pass

The forward pass involves computing the activations of each layer in the neural network, given the input data.

```python
# Forward pass
def forward_pass(X):
    hidden_layer = sigmoid(np.dot(X, W1) + b1)
    output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)
    return output_layer
```

Slide 6: Cost Function

The cost function measures the difference between the predicted output and the true output. A common cost function for neural networks is the mean squared error.

```python
# Cost function
def cost_function(X, y):
    y_pred = forward_pass(X)
    return np.mean((y_pred - y) ** 2)
```

Slide 7: Backpropagation

Backpropagation is the algorithm used to compute the gradients of the cost function with respect to the weights and biases in the neural network.

```python
# Backpropagation
def backpropagation(X, y):
    # Forward pass
    hidden_layer = sigmoid(np.dot(X, W1) + b1)
    output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)

    # Compute gradients
    error = output_layer - y
    delta_output = error * output_layer * (1 - output_layer)
    delta_hidden = np.dot(delta_output, W2.T) * hidden_layer * (1 - hidden_layer)

    # Update weights and biases
    W2_grad = np.dot(hidden_layer.T, delta_output)
    b2_grad = np.sum(delta_output, axis=0, keepdims=True)
    W1_grad = np.dot(X.T, delta_hidden)
    b1_grad = np.sum(delta_hidden, axis=0, keepdims=True)

    return W1_grad, b1_grad, W2_grad, b2_grad
```

Slide 8: Gradient Descent Update

The gradients computed by backpropagation are used to update the weights and biases in the direction of minimizing the cost function.

```python
# Gradient descent update
learning_rate = 0.1
for epoch in range(num_epochs):
    W1_grad, b1_grad, W2_grad, b2_grad = backpropagation(X, y)
    W1 -= learning_rate * W1_grad
    b1 -= learning_rate * b1_grad
    W2 -= learning_rate * W2_grad
    b2 -= learning_rate * b2_grad
```

Slide 9: Stochastic Gradient Descent

Stochastic gradient descent is a variant of gradient descent where the gradients are computed and the weights are updated for each training example, rather than the entire training set. This can lead to faster convergence and better generalization.

```python
for epoch in range(num_epochs):
    for X_batch, y_batch in data_loader:
        W1_grad, b1_grad, W2_grad, b2_grad = backpropagation(X_batch, y_batch)
        W1 -= learning_rate * W1_grad
        b1 -= learning_rate * b1_grad
        W2 -= learning_rate * W2_grad
        b2 -= learning_rate * b2_grad
```

Slide 10: Learning Rate

The learning rate is a hyperparameter that controls the step size of the weight updates during gradient descent. A large learning rate may cause the algorithm to diverge, while a small learning rate may result in slow convergence.

```python
# Adjust the learning rate dynamically
learning_rate = 0.1
for epoch in range(num_epochs):
    if epoch % 100 == 0:
        learning_rate *= 0.9
    # Perform gradient descent update with adjusted learning rate
```

Slide 11: Momentum

Momentum is a technique used in gradient descent to accelerate the convergence and escape local minima by adding a fraction of the previous update to the current update.

```python
momentum = 0.9
velocity_W1 = 0
velocity_b1 = 0
velocity_W2 = 0
velocity_b2 = 0

for epoch in range(num_epochs):
    W1_grad, b1_grad, W2_grad, b2_grad = backpropagation(X, y)
    velocity_W1 = momentum * velocity_W1 + learning_rate * W1_grad
    velocity_b1 = momentum * velocity_b1 + learning_rate * b1_grad
    velocity_W2 = momentum * velocity_W2 + learning_rate * W2_grad
    velocity_b2 = momentum * velocity_b2 + learning_rate * b2_grad

    W1 -= velocity_W1
    b1 -= velocity_b1
    W2 -= velocity_W2
    b2 -= velocity_b2
```

Slide 12: Challenges and Extensions

While gradient descent is a powerful optimization algorithm, it has limitations, such as the possibility of getting stuck in local minima, the need for careful selection of hyperparameters, and the computational cost for large datasets. Advanced techniques like adaptive learning rates, regularization, and normalization techniques can help mitigate these challenges.

```python
# Pseudocode for adaptive learning rate (e.g., Adam optimizer)
initialize adam_parameters
for epoch in range(num_epochs):
    W1_grad, b1_grad, W2_grad, b2_grad = backpropagation(X, y)
    update_adam_parameters(W1_grad, b1_grad, W2_grad, b2_grad)
    update_weights_and_biases(W1, b1, W2, b2, adam_parameters)
```

Slide 13: Regularization

Regularization is a technique used to prevent overfitting by adding a penalty term to the cost function, which encourages smaller weight values and promotes generalization.

```python
# L2 regularization
lambda_reg = 0.01
regularization_term = lambda_reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
cost_function = np.mean((y_pred - y) ** 2) +
```
