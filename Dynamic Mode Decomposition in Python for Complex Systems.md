## Dynamic Mode Decomposition in Python for Complex Systems
Slide 1: Introduction to Dynamic Mode Decomposition (DMD)

Dynamic Mode Decomposition is a powerful data-driven technique for analyzing complex dynamical systems. It extracts spatiotemporal coherent structures from high-dimensional data, allowing us to understand and predict system behavior. DMD is particularly useful in fluid dynamics, neuroscience, and climate science.

```python
import numpy as np
from scipy import linalg

def dmd(X, r=None):
    X1, X2 = X[:, :-1], X[:, 1:]
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    if r is not None:
        U, S, Vh = U[:, :r], S[:r], Vh[:r, :]
    A_tilde = U.conj().T @ X2 @ Vh.conj().T @ np.diag(1/S)
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
    Phi = X2 @ Vh.conj().T @ np.diag(1/S) @ eigenvectors
    return Phi, eigenvalues
```

Slide 2: Mathematical Foundation of DMD

DMD is based on the idea that complex systems can be approximated by linear dynamics in a high-dimensional space. It assumes that the state of the system at the next time step can be represented as a linear combination of the current state.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
t = np.linspace(0, 10, 1000)
x1 = np.sin(t)
x2 = np.cos(t)
X = np.vstack((x1, x2))

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(t, x1, label='x1')
plt.plot(t, x2, label='x2')
plt.legend()
plt.title('Synthetic Data for DMD')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
```

Slide 3: DMD Algorithm Step 1 - Data Matrix Construction

The first step in DMD is to construct data matrices from time-series measurements. We create two matrices: X1 containing all but the last time step, and X2 containing all but the first time step.

```python
def construct_data_matrices(X):
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    return X1, X2

# Example usage
X1, X2 = construct_data_matrices(X)
print("X1 shape:", X1.shape)
print("X2 shape:", X2.shape)
```

Slide 4: DMD Algorithm Step 2 - Singular Value Decomposition (SVD)

The next step is to perform SVD on X1. This decomposes X1 into left singular vectors (U), singular values (S), and right singular vectors (V).

```python
def compute_svd(X1, r=None):
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    if r is not None:
        U, S, Vh = U[:, :r], S[:r], Vh[:r, :]
    return U, S, Vh

# Example usage
U, S, Vh = compute_svd(X1)
print("U shape:", U.shape)
print("S shape:", S.shape)
print("Vh shape:", Vh.shape)
```

Slide 5: DMD Algorithm Step 3 - Compute DMD Matrix

We compute the DMD matrix A\_tilde, which represents the linear dynamics in the reduced space.

```python
def compute_dmd_matrix(U, S, Vh, X2):
    return U.conj().T @ X2 @ Vh.conj().T @ np.diag(1/S)

# Example usage
A_tilde = compute_dmd_matrix(U, S, Vh, X2)
print("A_tilde shape:", A_tilde.shape)
```

Slide 6: DMD Algorithm Step 4 - Eigendecomposition

We perform eigendecomposition on A\_tilde to obtain the DMD modes and eigenvalues.

```python
def compute_dmd_modes(A_tilde, X2, S, Vh):
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
    Phi = X2 @ Vh.conj().T @ np.diag(1/S) @ eigenvectors
    return Phi, eigenvalues

# Example usage
Phi, eigenvalues = compute_dmd_modes(A_tilde, X2, S, Vh)
print("Phi shape:", Phi.shape)
print("eigenvalues shape:", eigenvalues.shape)
```

Slide 7: Interpreting DMD Results

The DMD modes (columns of Phi) represent spatial patterns, while the eigenvalues represent their temporal behavior. The magnitude of an eigenvalue indicates the mode's growth or decay, and its phase represents its frequency.

```python
def plot_dmd_modes(Phi, eigenvalues):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.scatter(eigenvalues.real, eigenvalues.imag)
    plt.title('DMD Eigenvalues')
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    
    plt.subplot(122)
    plt.plot(np.abs(Phi))
    plt.title('DMD Modes')
    plt.xlabel('Component')
    plt.ylabel('Magnitude')
    plt.show()

# Example usage
plot_dmd_modes(Phi, eigenvalues)
```

Slide 8: DMD for Prediction

DMD can be used for future state prediction by evolving the initial condition using the computed modes and eigenvalues.

```python
def dmd_predict(Phi, eigenvalues, x0, t):
    omega = np.log(eigenvalues)
    return np.real(Phi @ np.diag(np.exp(omega * t)) @ np.linalg.pinv(Phi) @ x0)

# Example usage
t_pred = np.linspace(0, 20, 2000)
x0 = X[:, 0]
X_pred = np.array([dmd_predict(Phi, eigenvalues, x0, t) for t in t_pred])

plt.figure(figsize=(10, 5))
plt.plot(t, X[0, :], 'b', label='Original')
plt.plot(t_pred, X_pred[:, 0], 'r--', label='DMD Prediction')
plt.legend()
plt.title('DMD Prediction')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
```

Slide 9: DMD for Dimensionality Reduction

DMD can be used for dimensionality reduction by keeping only the most dominant modes.

```python
def dmd_reduce(Phi, eigenvalues, X, n_modes):
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    Phi_reduced = Phi[:, idx[:n_modes]]
    eigenvalues_reduced = eigenvalues[idx[:n_modes]]
    X_reduced = Phi_reduced @ np.diag(eigenvalues_reduced) @ np.linalg.pinv(Phi_reduced) @ X
    return X_reduced

# Example usage
n_modes = 5
X_reduced = dmd_reduce(Phi, eigenvalues, X, n_modes)

plt.figure(figsize=(10, 5))
plt.plot(t, X[0, :], 'b', label='Original')
plt.plot(t, X_reduced[0, :], 'r--', label='Reduced')
plt.legend()
plt.title(f'DMD Dimensionality Reduction (n_modes={n_modes})')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
```

Slide 10: DMD for Noise Reduction

DMD can be used for noise reduction by filtering out high-frequency modes.

```python
def dmd_denoise(Phi, eigenvalues, X, freq_cutoff):
    omega = np.log(eigenvalues).imag / (2 * np.pi)
    idx = np.abs(omega) < freq_cutoff
    Phi_denoised = Phi[:, idx]
    eigenvalues_denoised = eigenvalues[idx]
    X_denoised = Phi_denoised @ np.diag(eigenvalues_denoised) @ np.linalg.pinv(Phi_denoised) @ X
    return X_denoised

# Example usage
X_noisy = X + np.random.normal(0, 0.1, X.shape)
X_denoised = dmd_denoise(Phi, eigenvalues, X_noisy, freq_cutoff=0.5)

plt.figure(figsize=(10, 5))
plt.plot(t, X_noisy[0, :], 'b', label='Noisy')
plt.plot(t, X_denoised[0, :], 'r--', label='Denoised')
plt.legend()
plt.title('DMD Noise Reduction')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
```

Slide 11: Real-Life Example: Fluid Dynamics

DMD is widely used in fluid dynamics to analyze complex flow patterns. Let's simulate a simple fluid flow and apply DMD to extract coherent structures.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def fluid_flow(y, t, sigma, rho, beta):
    x, y, z = y
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

# Generate data
sigma, rho, beta = 10, 28, 8/3
y0 = [1, 1, 1]
t = np.linspace(0, 50, 5000)
solution = odeint(fluid_flow, y0, t, args=(sigma, rho, beta))

# Apply DMD
X = solution.T
Phi, eigenvalues = dmd(X)

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(t, X[0, :])
plt.title('Fluid Flow Time Series')
plt.xlabel('Time')
plt.ylabel('X component')

plt.subplot(122)
plt.scatter(eigenvalues.real, eigenvalues.imag)
plt.title('DMD Eigenvalues')
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.tight_layout()
plt.show()
```

Slide 12: Real-Life Example: Climate Data Analysis

DMD can be applied to climate data to identify patterns and trends. Let's use a simplified climate dataset and apply DMD to extract dominant modes.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic climate data
years = np.arange(1900, 2023)
temp = 15 + 0.01 * (years - 1900) + 0.5 * np.sin(2 * np.pi * years / 11) + np.random.normal(0, 0.2, len(years))

# Apply DMD
X = temp.reshape(1, -1)
Phi, eigenvalues = dmd(X)

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(years, temp)
plt.title('Global Temperature Anomaly')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')

plt.subplot(122)
plt.scatter(eigenvalues.real, eigenvalues.imag)
plt.title('DMD Eigenvalues')
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.tight_layout()
plt.show()

# Reconstruct and predict
X_dmd = dmd_predict(Phi, eigenvalues, X[:, 0], np.arange(len(years) + 20))
future_years = np.arange(1900, 2043)

plt.figure(figsize=(10, 5))
plt.plot(years, temp, label='Original Data')
plt.plot(future_years, X_dmd[0], '--', label='DMD Reconstruction and Prediction')
plt.title('Climate Data Analysis with DMD')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
```

Slide 13: Challenges and Limitations of DMD

While DMD is a powerful technique, it has some limitations:

1. Assumes linear dynamics, which may not always hold for complex systems.
2. Sensitive to noise and outliers in the data.
3. Requires careful selection of parameters, such as the number of modes to retain.
4. May struggle with systems that have multiple time scales or intermittent behavior.

To address these challenges, researchers have developed variants like Compressed DMD, Multi-resolution DMD, and Robust DMD.

```python
def robust_dmd(X, r=None, iterations=10):
    X1, X2 = X[:, :-1], X[:, 1:]
    for _ in range(iterations):
        U, S, Vh = np.linalg.svd(X1, full_matrices=False)
        if r is not None:
            U, S, Vh = U[:, :r], S[:r], Vh[:r, :]
        A_tilde = U.conj().T @ X2 @ Vh.conj().T @ np.diag(1/S)
        eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
        Phi = X2 @ Vh.conj().T @ np.diag(1/S) @ eigenvectors
        X1 = Phi @ np.diag(eigenvalues) @ np.linalg.pinv(Phi) @ X1
    return Phi, eigenvalues

# Example usage (not run due to computational intensity)
# Phi_robust, eigenvalues_robust = robust_dmd(X, r=10, iterations=5)
```

Slide 14: Future Directions and Advanced Topics

DMD continues to evolve with new research directions:

1. Integration with machine learning techniques, such as neural networks.
2. Application to non-linear systems through Koopman operator theory.
3. Extension to handle partial observations and missing data.
4. Development of real-time DMD algorithms for online analysis.

These advancements are expanding the applicability of DMD to even more complex systems and datasets.

```python
# Conceptual code for DMD with neural networks (not functional)
import tensorflow as tf

class DMDLayer(tf.keras.layers.Layer):
    def __init__(self, num_modes):
        super(DMDLayer, self).__init__()
        self.num_modes = num_modes

    def build(self, input_shape):
        self.Phi = self.add_weight("Phi", shape=[input_shape[-1], self.num_modes])
        self.eigenvalues = self.add_weight("eigenvalues", shape=[self.num_modes])

    def call(self, inputs):
        return tf.matmul(inputs, self.Phi) * self.eigenvalues

# Example usage (not run)
# model = tf.keras.Sequential([
#     DMDLayer(10),
#     tf.keras.layers.Dense(1)
# ])
```

Slide 15: Additional Resources

For further exploration of Dynamic Mode Decomposition, consider these resources:

1. Kutz, J. N., Brunton, S. L., Brunton, B. W., & Proctor, J. L. (2016). Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems. SIAM. ArXiv: [https://arxiv.org/abs/1409.6358](https://arxiv.org/abs/1409.6358)
2. Schmid, P. J. (2010). Dynamic mode decomposition of numerical and experimental data. Journal of Fluid Mechanics, 656, 5-28. ArXiv: [https://arxiv.org/abs/1312.0041](https://arxiv.org/abs/1312.0041)
3. Brunton, S. L., & Kutz, J. N. (2019). Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control. Cambridge University Press. ArXiv: [https://arxiv.org/abs/1609.00893](https://arxiv.org/abs/1609.00893)
4. Tu, J. H., Rowley, C. W., Luchtenburg, D. M., Brunton, S. L., & Kutz, J. N. (2014). On dynamic mode decomposition: Theory and applications. Journal of Computational Dynamics, 1(2), 391-421. ArXiv: [https://arxiv.org/abs/1312.0041](https://arxiv.org/abs/1312.0041)

These resources provide in-depth discussions on the theory, implementation, and applications of DMD across various fields. They offer valuable insights for both beginners and advanced practitioners looking to deepen their understanding of this powerful technique.

