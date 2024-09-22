## Mathematical Foundations for Machine Learning
Slide 1: Notation

Mathematical Notation in Linear Algebra and Machine Learning

Mathematical notation is the foundation for expressing complex ideas concisely. In machine learning, we often use vectors (boldface lowercase letters) and matrices (boldface uppercase letters). Scalars are typically represented by italic lowercase letters.

```python
import numpy as np

# Vector notation
v = np.array([1, 2, 3])  # Column vector
w = np.array([[1, 2, 3]])  # Row vector

# Matrix notation
A = np.array([[1, 2], [3, 4], [5, 6]])

print("Vector v:\n", v)
print("\nVector w:\n", w)
print("\nMatrix A:\n", A)

# Scalar multiplication
alpha = 2
scaled_v = alpha * v
print("\nScaled vector alpha * v:\n", scaled_v)
```

Slide 2: Linear Algebra

Fundamental Operations in Linear Algebra

Linear algebra forms the backbone of many machine learning algorithms. Key operations include vector addition, scalar multiplication, and matrix multiplication.

```python
import numpy as np

# Vector addition
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
v_sum = v1 + v2
print("Vector sum:", v_sum)

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)
print("\nMatrix product:\n", C)

# Transpose
A_transpose = A.T
print("\nTranspose of A:\n", A_transpose)
```

Slide 3: Calculus and Optimization

Gradients and Optimization in Machine Learning

Calculus is crucial for optimization in machine learning. The gradient represents the direction of steepest ascent for a function, and we often use it to minimize loss functions.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 2*x + 1

def df(x):
    return 2*x + 2

x = np.linspace(-3, 1, 100)
y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='f(x) = x^2 + 2x + 1')
plt.plot(x, df(x), label="f'(x) = 2x + 2")
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=-1, color='r', linestyle='--', label='Minimum at x=-1')
plt.legend()
plt.title('Function and its Derivative')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
```

Slide 4: Probability

Fundamentals of Probability Theory

Probability theory is essential in machine learning for modeling uncertainty and making predictions. It provides a framework for reasoning about random events and their likelihood.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate coin flips
num_flips = 1000
coin_flips = np.random.choice(['H', 'T'], size=num_flips)

# Calculate cumulative probability of heads
cumulative_prob = np.cumsum(coin_flips == 'H') / np.arange(1, num_flips + 1)

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_flips + 1), cumulative_prob)
plt.axhline(y=0.5, color='r', linestyle='--', label='Expected probability')
plt.title('Cumulative Probability of Heads in Coin Flips')
plt.xlabel('Number of Flips')
plt.ylabel('Probability of Heads')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 5: Random Variables and Distributions

Understanding Random Variables and Their Distributions

Random variables are fundamental in probability theory and statistics. They can be discrete or continuous, each with its own probability distribution.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate random data from different distributions
normal_data = np.random.normal(0, 1, 1000)
uniform_data = np.random.uniform(-3, 3, 1000)
exponential_data = np.random.exponential(1, 1000)

# Plot histograms
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.hist(normal_data, bins=30, density=True, alpha=0.7)
x = np.linspace(-4, 4, 100)
plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2)
plt.title('Normal Distribution')

plt.subplot(132)
plt.hist(uniform_data, bins=30, density=True, alpha=0.7)
plt.plot([-3, -3, 3, 3], [0, 1/6, 1/6, 0], 'r-', lw=2)
plt.title('Uniform Distribution')

plt.subplot(133)
plt.hist(exponential_data, bins=30, density=True, alpha=0.7)
x = np.linspace(0, 10, 100)
plt.plot(x, stats.expon.pdf(x), 'r-', lw=2)
plt.title('Exponential Distribution')

plt.tight_layout()
plt.show()
```

Slide 6: Estimation of Parameters

Parameter Estimation in Statistical Models

Parameter estimation is a crucial task in machine learning and statistics. It involves inferring the underlying parameters of a probability distribution or model from observed data.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data from a normal distribution
true_mean = 5
true_std = 2
sample_size = 100
data = np.random.normal(true_mean, true_std, sample_size)

# Estimate parameters
estimated_mean = np.mean(data)
estimated_std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation

# Plot histogram of data and estimated distribution
plt.figure(figsize=(10, 6))
plt.hist(data, bins=20, density=True, alpha=0.7, label='Sample Data')
x = np.linspace(min(data), max(data), 100)
plt.plot(x, stats.norm.pdf(x, estimated_mean, estimated_std), 'r-', lw=2, 
         label=f'Estimated N({estimated_mean:.2f}, {estimated_std:.2f})')
plt.plot(x, stats.norm.pdf(x, true_mean, true_std), 'g--', lw=2, 
         label=f'True N({true_mean}, {true_std})')
plt.title('Parameter Estimation for Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

print(f"True parameters: mean = {true_mean}, std = {true_std}")
print(f"Estimated parameters: mean = {estimated_mean:.2f}, std = {estimated_std:.2f}")
```

Slide 7: The Gaussian Distribution

Exploring the Gaussian (Normal) Distribution

The Gaussian distribution is a fundamental probability distribution in statistics and machine learning. It's characterized by its bell-shaped curve and is defined by two parameters: mean (μ) and standard deviation (σ).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))

x = np.linspace(-5, 5, 100)

plt.figure(figsize=(12, 6))

# Standard normal distribution
plt.plot(x, gaussian(x, 0, 1), label='N(0, 1)')

# Varying mean
plt.plot(x, gaussian(x, -2, 1), label='N(-2, 1)')
plt.plot(x, gaussian(x, 2, 1), label='N(2, 1)')

# Varying standard deviation
plt.plot(x, gaussian(x, 0, 0.5), label='N(0, 0.5)')
plt.plot(x, gaussian(x, 0, 2), label='N(0, 2)')

plt.title('Gaussian Distributions with Different Parameters')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

# Demonstrate properties of standard normal distribution
standard_normal = np.random.normal(0, 1, 10000)
print(f"Mean: {np.mean(standard_normal):.4f}")
print(f"Standard Deviation: {np.std(standard_normal):.4f}")
print(f"Probability within 1 sigma: {np.mean(np.abs(standard_normal) < 1):.4f}")
print(f"Probability within 2 sigma: {np.mean(np.abs(standard_normal) < 2):.4f}")
print(f"Probability within 3 sigma: {np.mean(np.abs(standard_normal) < 3):.4f}")
```

Slide 8: Maximum Likelihood Estimation

Understanding Maximum Likelihood Estimation (MLE)

Maximum Likelihood Estimation is a method of estimating the parameters of a probability distribution by maximizing a likelihood function. It's widely used in machine learning for parameter estimation.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize

# Generate sample data
true_mean = 2
true_std = 1.5
sample_size = 100
data = np.random.normal(true_mean, true_std, sample_size)

# Define negative log-likelihood function
def neg_log_likelihood(params, data):
    mean, std = params
    return -np.sum(stats.norm.logpdf(data, mean, std))

# Perform MLE
initial_guess = [0, 1]
result = optimize.minimize(neg_log_likelihood, initial_guess, args=(data,))
mle_mean, mle_std = result.x

# Plot results
x = np.linspace(min(data) - 2, max(data) + 2, 100)
plt.figure(figsize=(10, 6))
plt.hist(data, bins=20, density=True, alpha=0.7, label='Sample Data')
plt.plot(x, stats.norm.pdf(x, true_mean, true_std), 'g--', lw=2, 
         label=f'True N({true_mean}, {true_std})')
plt.plot(x, stats.norm.pdf(x, mle_mean, mle_std), 'r-', lw=2, 
         label=f'MLE N({mle_mean:.2f}, {mle_std:.2f})')
plt.title('Maximum Likelihood Estimation for Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

print(f"True parameters: mean = {true_mean}, std = {true_std}")
print(f"MLE parameters: mean = {mle_mean:.2f}, std = {mle_std:.2f}")
```

Slide 9: Maximum a Posteriori Estimation

Exploring Maximum a Posteriori (MAP) Estimation

Maximum a Posteriori estimation is a Bayesian approach to parameter estimation. Unlike MLE, MAP incorporates prior beliefs about the parameters, leading to more robust estimates, especially with limited data.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize

# Generate sample data
true_mean = 2
true_std = 1.5
sample_size = 20
data = np.random.normal(true_mean, true_std, sample_size)

# Define negative log posterior (combining likelihood and prior)
def neg_log_posterior(params, data, prior_mean, prior_std):
    mean, std = params
    if std <= 0:
        return np.inf
    log_likelihood = np.sum(stats.norm.logpdf(data, mean, std))
    log_prior = stats.norm.logpdf(mean, prior_mean, prior_std)
    return -(log_likelihood + log_prior)

# Perform MAP estimation
prior_mean = 0
prior_std = 2
initial_guess = [0, 1]
result = optimize.minimize(neg_log_posterior, initial_guess, args=(data, prior_mean, prior_std))
map_mean, map_std = result.x

# Compare with MLE
mle_result = optimize.minimize(lambda params, data: -np.sum(stats.norm.logpdf(data, params[0], params[1])), 
                               initial_guess, args=(data,))
mle_mean, mle_std = mle_result.x

# Plot results
x = np.linspace(min(data) - 2, max(data) + 2, 100)
plt.figure(figsize=(10, 6))
plt.hist(data, bins=10, density=True, alpha=0.7, label='Sample Data')
plt.plot(x, stats.norm.pdf(x, true_mean, true_std), 'g--', lw=2, 
         label=f'True N({true_mean}, {true_std})')
plt.plot(x, stats.norm.pdf(x, map_mean, map_std), 'r-', lw=2, 
         label=f'MAP N({map_mean:.2f}, {map_std:.2f})')
plt.plot(x, stats.norm.pdf(x, mle_mean, mle_std), 'b:', lw=2, 
         label=f'MLE N({mle_mean:.2f}, {mle_std:.2f})')
plt.title('MAP vs MLE Estimation for Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

print(f"True parameters: mean = {true_mean}, std = {true_std}")
print(f"MAP parameters: mean = {map_mean:.2f}, std = {map_std:.2f}")
print(f"MLE parameters: mean = {mle_mean:.2f}, std = {mle_std:.2f}")
```

Slide 10: Singular Value Decomposition (SVD)

Understanding Singular Value Decomposition

Singular Value Decomposition is a powerful technique in linear algebra with applications in dimensionality reduction, data compression, and machine learning. It decomposes a matrix into three matrices: U, Σ, and V^T.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# Perform SVD
U, s, Vt = np.linalg.svd(A)

# Reconstruct the matrix using different numbers of singular values
def reconstruct(U, s, Vt, k):
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# Plot original and reconstructed matrices
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0, 0].imshow(A, cmap='viridis')
axs[0, 0].set_title('Original Matrix')

for i, k in enumerate([1, 2, 3]):
    A_approx = reconstruct(U, s, Vt, k)
    axs[(i+1)//2, (i+1)%2].imshow(A_approx, cmap='viridis')
    axs[(i+1)//2, (i+1)%2].set_title(f'Reconstructed (k={k})')

plt.tight_layout()
plt.show()

print("Singular values:", s)
print("Frobenius norm of difference (k=1):", np.linalg.norm(A - reconstruct(U, s, Vt, 1)))
print("Frobenius norm of difference (k=2):", np.linalg.norm(A - reconstruct(U, s, Vt, 2)))
print("Frobenius norm of difference (k=3):", np.linalg.norm(A - reconstruct(U, s, Vt, 3)))
```

Slide 11: Positive Definite Matrices

Properties and Applications of Positive Definite Matrices

Positive definite matrices play a crucial role in various areas of mathematics and machine learning. They have several important properties, including having all positive eigenvalues and being invertible.

```python
import numpy as np
import matplotlib.pyplot as plt

def is_positive_definite(A):
    return np.all(np.linalg.eigvals(A) > 0)

# Create a positive definite matrix
A = np.array([[2, -1], [-1, 2]])

# Check if A is positive definite
print("A is positive definite:", is_positive_definite(A))

# Visualize the quadratic form x^T A x
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = A[0,0]*X**2 + 2*A[0,1]*X*Y + A[1,1]*Y**2

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='z = x^T A x')
plt.title('Quadratic Form of a Positive Definite Matrix')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
plt.show()

# Compute and print eigenvalues
eigenvalues = np.linalg.eigvals(A)
print("Eigenvalues of A:", eigenvalues)
```

Slide 12: Eigenvalues and Eigenvectors

Understanding Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors are fundamental concepts in linear algebra with applications in various fields, including machine learning, computer graphics, and quantum mechanics.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a 2x2 matrix
A = np.array([[3, 1], [1, 2]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print("\nEigenvalues:", eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

# Visualize the eigenvectors
plt.figure(figsize=(8, 8))
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')

# Plot original basis vectors
plt.arrow(0, 0, 1, 0, head_width=0.05, head_length=0.1, fc='b', ec='b', label='Standard Basis')
plt.arrow(0, 0, 0, 1, head_width=0.05, head_length=0.1, fc='b', ec='b')

# Plot eigenvectors
for i in range(2):
    plt.arrow(0, 0, eigenvectors[0, i], eigenvectors[1, i], head_width=0.05, head_length=0.1, 
              fc='r', ec='r', label=f'Eigenvector {i+1}')

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.title('Eigenvectors of Matrix A')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Verify Av = λv
for i in range(2):
    print(f"\nVerifying Av = λv for eigenvector {i+1}:")
    print("Av =", A @ eigenvectors[:, i])
    print("λv =", eigenvalues[i] * eigenvectors[:, i])
```

Slide 13: Convexity and Optimization

Convex Functions and Optimization in Machine Learning

Convexity is a crucial concept in optimization, particularly in machine learning. Convex functions have a single global minimum, making them ideal for many optimization problems.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 2*x + 1

def gradient_descent(f, df, x0, learning_rate, num_iterations):
    x = x0
    history = [x]
    for _ in range(num_iterations):
        x = x - learning_rate * df(x)
        history.append(x)
    return np.array(history)

# Define the derivative of f
df = lambda x: 2*x + 2

# Perform gradient descent
x0 = 2
learning_rate = 0.1
num_iterations = 20
history = gradient_descent(f, df, x0, learning_rate, num_iterations)

# Plot the function and gradient descent steps
x = np.linspace(-3, 3, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, f(x), label='f(x) = x^2 + 2x + 1')
plt.scatter(history, f(history), c='r', label='Gradient Descent Steps')
plt.title('Gradient Descent on a Convex Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

print("Optimization steps:")
for i, xi in enumerate(history):
    print(f"Step {i}: x = {xi:.4f}, f(x) = {f(xi):.4f}")
```

Slide 14: Additional Resources

For further exploration of these topics, consider the following resources:

1. "Mathematics for Machine Learning" by Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (arXiv:1803.08823)
2. "Convex Optimization" by Stephen Boyd and Lieven Vandenberghe (available online: [https://web.stanford.edu/~boyd/cvxbook/](https://web.stanford.edu/~boyd/cvxbook/))
3. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (available online: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/))
4. "Pattern Recognition and Machine Learning" by Christopher M. Bishop

These resources provide in-depth coverage of the mathematical foundations of machine learning and can help deepen your understanding of the concepts covered in this presentation.

