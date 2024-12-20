## Adam Optimizer Exponentially Weighted Gradients and Squared Gradients
Slide 1: Introduction to Adam Optimizer

Adam (Adaptive Moment Estimation) optimizer combines the advantages of two other optimization algorithms: RMSprop and momentum, maintaining exponentially decaying average of past gradients and squared gradients to adapt learning rates for each parameter.

```python
# Mathematical formulation for Adam optimizer
"""
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t$$
"""
```

Slide 2: Basic Adam Implementation

The core implementation of Adam optimizer showcases how it maintains moving averages of both gradients and their squares, using bias correction to adjust for initialization bias during early iterations.

```python
import numpy as np

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def initialize(self, params_shape):
        self.m = np.zeros(params_shape)
        self.v = np.zeros(params_shape)
        
    def update(self, params, grads):
        if self.m is None:
            self.initialize(params.shape)
            
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1**self.t)
        # Compute bias-corrected second moment estimate
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update parameters
        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params
```

Slide 3: Linear Regression with Adam

A practical example implementing linear regression using Adam optimizer demonstrates its effectiveness in minimizing the mean squared error loss function for basic regression tasks.

```python
import numpy as np
from sklearn.datasets import make_regression

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=1, noise=0.1, random_state=42)
y = y.reshape(-1, 1)

# Initialize parameters
w = np.random.randn(1, 1)
b = np.zeros(1)

# Create Adam optimizer instances
adam_w = Adam(learning_rate=0.01)
adam_b = Adam(learning_rate=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    y_pred = np.dot(X, w) + b
    
    # Compute gradients
    dw = np.dot(X.T, (y_pred - y)) / len(X)
    db = np.sum(y_pred - y) / len(X)
    
    # Update parameters using Adam
    w = adam_w.update(w, dw)
    b = adam_b.update(b, db)
    
    if epoch % 10 == 0:
        loss = np.mean(np.square(y_pred - y))
        print(f'Epoch {epoch}, Loss: {loss:.6f}')
```

Slide 4: Neural Network with Adam

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        # Initialize Adam optimizers for each parameter
        self.adam_W1 = Adam()
        self.adam_b1 = Adam()
        self.adam_W2 = Adam()
        self.adam_b2 = Adam()
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2))  # Sigmoid activation
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update parameters using Adam
        self.W1 = self.adam_W1.update(self.W1, dW1)
        self.b1 = self.adam_b1.update(self.b1, db1)
        self.W2 = self.adam_W2.update(self.W2, dW2)
        self.b2 = self.adam_b2.update(self.b2, db2)
```

Slide 5: Training Neural Network with Adam

The training process demonstrates how Adam optimizer effectively adjusts learning rates for each parameter, leading to faster convergence compared to standard gradient descent methods.

```python
# Generate synthetic classification data
X = np.random.randn(1000, 10)
y = (np.sum(X, axis=1) > 0).reshape(-1, 1)

# Create and train neural network
nn = NeuralNetwork(input_size=10, hidden_size=5, output_size=1)

# Training loop
for epoch in range(1000):
    # Forward pass
    output = nn.forward(X)
    
    # Compute loss
    loss = -np.mean(y * np.log(output + 1e-8) + (1-y) * np.log(1-output + 1e-8))
    
    # Backward pass and parameter update
    nn.backward(X, y, output)
    
    if epoch % 100 == 0:
        accuracy = np.mean((output > 0.5) == y)
        print(f'Epoch {epoch}, Loss: {loss:.6f}, Accuracy: {accuracy:.4f}')
```

Slide 6: Adam with Weight Decay (AdamW)

AdamW modifies the original Adam optimizer by implementing weight decay as a direct multiplicative factor on the weights rather than incorporating it into the gradient computation, resulting in better generalization.

```python
class AdamW:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        
        # Weight decay term
        params = params * (1 - self.learning_rate * self.weight_decay)
        
        # Update moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update parameters
        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params
```

Slide 7: Implementing AMSGrad Variant

AMSGrad addresses the convergence issues of Adam by maintaining the maximum of all past squared gradients and using it for parameter updates instead of exponential moving averages.

```python
class AMSGrad:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.v_hat = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            self.v_hat = np.zeros_like(params)
        
        self.t += 1
        
        # Update moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)
        
        # Update maximum squared gradient
        self.v_hat = np.maximum(self.v_hat, self.v)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        
        # Update parameters using maximum
        params -= self.learning_rate * m_hat / (np.sqrt(self.v_hat) + self.epsilon)
        return params
```

Slide 8: Comparison of Optimizers

A comprehensive comparison of different Adam variants on a complex optimization problem demonstrates their relative performance characteristics and convergence behaviors.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate complex optimization landscape
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# Initialize optimizers
adam = Adam(learning_rate=0.001)
adamw = AdamW(learning_rate=0.001)
amsgrad = AMSGrad(learning_rate=0.001)

# Initial points
x = np.array([1.5])
y = np.array([1.5])

# Training history
history = {
    'adam': {'x': [], 'y': [], 'loss': []},
    'adamw': {'x': [], 'y': [], 'loss': []},
    'amsgrad': {'x': [], 'y': [], 'loss': []}
}

# Optimization loop
for step in range(1000):
    # Compute gradients
    dx = 2 * (x - 1) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    
    # Update parameters with different optimizers
    x_adam = adam.update(x.copy(), dx)
    y_adam = adam.update(y.copy(), dy)
    
    x_adamw = adamw.update(x.copy(), dx)
    y_adamw = adamw.update(y.copy(), dy)
    
    x_amsgrad = amsgrad.update(x.copy(), dx)
    y_amsgrad = amsgrad.update(y.copy(), dy)
    
    # Store history
    for opt, x_val, y_val in [('adam', x_adam, y_adam),
                             ('adamw', x_adamw, y_adamw),
                             ('amsgrad', x_amsgrad, y_amsgrad)]:
        history[opt]['x'].append(float(x_val))
        history[opt]['y'].append(float(y_val))
        history[opt]['loss'].append(float(rosenbrock(x_val, y_val)))

# Plot results
plt.figure(figsize=(12, 6))
for opt in ['adam', 'adamw', 'amsgrad']:
    plt.plot(history[opt]['loss'], label=opt.upper())
plt.yscale('log')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.title('Optimizer Comparison on Rosenbrock Function')
plt.show()
```

Slide 9: Rectified Adam (RAdam)

RAdam addresses the adaptive learning rate's variance in the early stages of training by implementing a rectification term that automatically adjusts the adaptive learning rate based on the variance of the momentum.

```python
class RAdam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        
        # Update momentum and variance
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Compute rectification term
        rho_inf = 2 / (1 - self.beta2) - 1
        rho_t = rho_inf - 2 * self.t * self.beta2**self.t / (1 - self.beta2**self.t)
        
        if rho_t > 4:
            # Reactive variance adaptation
            r_t = np.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
            params -= self.learning_rate * r_t * m_hat / (np.sqrt(v_hat) + self.epsilon)
        else:
            # Fall back to SGD
            params -= self.learning_rate * m_hat
        
        return params
```

Slide 10: Convolutional Neural Network with RAdam

This implementation showcases RAdam's effectiveness in training deep convolutional networks, particularly during the initial phase where adaptive learning rates can be unstable.

```python
import numpy as np

class ConvLayer:
    def __init__(self, input_shape, kernel_size, num_filters):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        
        # Initialize kernels and bias
        self.kernels = np.random.randn(
            num_filters, 
            input_shape[0], 
            kernel_size, 
            kernel_size) * 0.01
        self.bias = np.zeros(num_filters)
        
        # Initialize RAdam optimizer
        self.kernel_optimizer = RAdam()
        self.bias_optimizer = RAdam()
    
    def forward(self, inputs):
        self.inputs = inputs
        batch_size = inputs.shape[0]
        
        # Calculate output dimensions
        output_height = self.input_shape[1] - self.kernel_size + 1
        output_width = self.input_shape[2] - self.kernel_size + 1
        
        # Initialize output
        self.output = np.zeros((
            batch_size,
            self.num_filters,
            output_height,
            output_width
        ))
        
        # Convolution operation
        for i in range(output_height):
            for j in range(output_width):
                input_slice = inputs[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                for k in range(self.num_filters):
                    self.output[:, k, i, j] = np.sum(
                        input_slice * self.kernels[k], 
                        axis=(1,2,3)
                    ) + self.bias[k]
        
        return self.output
    
    def backward(self, grad_output):
        batch_size = grad_output.shape[0]
        grad_kernels = np.zeros_like(self.kernels)
        grad_bias = np.sum(grad_output, axis=(0,2,3))
        
        # Update parameters using RAdam
        self.kernels = self.kernel_optimizer.update(self.kernels, grad_kernels)
        self.bias = self.bias_optimizer.update(self.bias, grad_bias)
        
        return grad_input
```

Slide 11: AdaBelief Optimizer

AdaBelief adapts the step size based on the "belief" in observed gradients, offering improved training stability and generalization compared to traditional Adam variants.

```python
class AdaBelief:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.s = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.s = np.zeros_like(params)
        
        self.t += 1
        
        # Update first moment
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        
        # Update second moment based on belief
        grad_residual = grads - self.m
        self.s = self.beta2 * self.s + (1 - self.beta2) * np.square(grad_residual)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        s_hat = self.s / (1 - self.beta2**self.t)
        
        # Update parameters
        params -= self.learning_rate * m_hat / (np.sqrt(s_hat + self.epsilon))
        
        return params
```

Slide 12: Complex Optimization Example

This comprehensive example demonstrates the performance differences between various Adam-based optimizers on a challenging non-convex optimization problem using a mixture of Gaussian distributions.

```python
import numpy as np
from scipy.stats import multivariate_normal

class OptimizationProblem:
    def __init__(self):
        # Create mixture of Gaussians
        self.means = np.array([[-2, -2], [2, 2], [-2, 2], [2, -2]])
        self.covs = np.array([[[1, 0.5], [0.5, 1]]] * 4)
        self.weights = np.array([0.25, 0.25, 0.25, 0.25])
        
    def objective(self, x):
        result = 0
        for mean, cov, weight in zip(self.means, self.covs, self.weights):
            result += weight * multivariate_normal.pdf(x, mean=mean, cov=cov)
        return -result  # Negative because we want to minimize
    
    def gradient(self, x):
        grad = np.zeros_like(x)
        for mean, cov, weight in zip(self.means, self.covs, self.weights):
            diff = x - mean
            grad += weight * multivariate_normal.pdf(x, mean=mean, cov=cov) * \
                   np.linalg.solve(cov, diff)
        return grad

# Initialize optimizers
optimizers = {
    'Adam': Adam(learning_rate=0.01),
    'AdamW': AdamW(learning_rate=0.01),
    'RAdam': RAdam(learning_rate=0.01),
    'AdaBelief': AdaBelief(learning_rate=0.01)
}

# Training loop
iterations = 1000
x_init = np.array([3.0, 3.0])
results = {name: {'trajectory': [], 'loss': []} for name in optimizers}

for name, optimizer in optimizers.items():
    x = x_init.copy()
    problem = OptimizationProblem()
    
    for i in range(iterations):
        grad = problem.gradient(x)
        loss = problem.objective(x)
        
        results[name]['trajectory'].append(x.copy())
        results[name]['loss'].append(loss)
        
        x = optimizer.update(x, grad)
```

Slide 13: Visualization of Optimizer Performance

A detailed visualization comparing convergence rates, stability, and final optimization results for different Adam variants on the mixture of Gaussians problem.

```python
import matplotlib.pyplot as plt

def plot_optimization_results(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss curves
    for name, result in results.items():
        losses = np.array(result['loss'])
        ax1.plot(losses, label=name)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_title('Convergence Comparison')
    ax1.legend()
    
    # Plot trajectories
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    problem = OptimizationProblem()
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i,j] = problem.objective([X[i,j], Y[i,j]])
    
    ax2.contour(X, Y, Z, levels=20)
    
    for name, result in results.items():
        trajectory = np.array(result['trajectory'])
        ax2.plot(trajectory[:,0], trajectory[:,1], 
                label=f'{name} path', marker='.')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Optimization Trajectories')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Plot results
plot_optimization_results(results)
```

Slide 14: Additional Resources

*   "Adam: A Method for Stochastic Optimization" - [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
*   "Decoupled Weight Decay Regularization" - [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
*   "On the Variance of the Adaptive Learning Rate and Beyond" - [https://arxiv.org/abs/1908.03265](https://arxiv.org/abs/1908.03265)
*   "AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients" - [https://arxiv.org/abs/2010.07468](https://arxiv.org/abs/2010.07468)
*   "AMSGrad: On the Convergence of Adam and Beyond" - [https://openreview.net/forum?id=ryQu7f-RZ](https://openreview.net/forum?id=ryQu7f-RZ)

