## Dynamic Mode Decomposition in Python for Data Analysis
Slide 1: Introduction to Dynamic Mode Decomposition (DMD)

Dynamic Mode Decomposition is a powerful data-driven method for analyzing complex dynamical systems. It extracts spatiotemporal coherent structures from high-dimensional data, providing insights into the system's underlying dynamics.

```python
import numpy as np
from scipy import linalg

def dmd(X, r=None):
    # X: data matrix (each column is a snapshot)
    # r: rank truncation (optional)
    
    # Compute SVD of X
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    
    # Truncate to rank r if specified
    if r is not None:
        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:r, :]
    
    # Compute DMD matrix
    A_tilde = U.conj().T @ X[:, 1:] @ Vh.conj().T @ np.diag(1/S)
    
    # Eigendecomposition of A_tilde
    w, v = np.linalg.eig(A_tilde)
    
    # DMD modes
    Phi = X[:, 1:] @ Vh.conj().T @ np.diag(1/S) @ v
    
    return w, Phi
```

Slide 2: Mathematical Foundation of DMD

DMD approximates the dynamics of a system by finding a best-fit linear operator that advances the system forward in time. It decomposes the system into modes, each associated with a growth rate and frequency.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
t = np.linspace(0, 10, 1000)
x = np.sin(t) + 0.5 * np.sin(2*t) + 0.1 * np.random.randn(len(t))

# Reshape data into snapshot matrix
X = np.column_stack((x[:-1], x[1:]))

# Compute DMD
U, S, Vh = np.linalg.svd(X, full_matrices=False)
A = U.conj().T @ X[:, 1:] @ Vh.conj().T @ np.diag(1/S)
w, v = np.linalg.eig(A)

plt.figure(figsize=(10, 5))
plt.plot(t[:-1], x[:-1], label='Original')
plt.plot(t[:-1], np.real(U @ v[:, 0]), label='DMD Mode 1')
plt.plot(t[:-1], np.real(U @ v[:, 1]), label='DMD Mode 2')
plt.legend()
plt.title('Original Signal vs DMD Modes')
plt.show()
```

Slide 3: Data Preparation for DMD

Proper data preparation is crucial for effective DMD analysis. This involves organizing your data into snapshot matrices and considering appropriate preprocessing steps.

```python
import numpy as np

def prepare_data(data, time_delay=1):
    """
    Prepare data for DMD analysis
    
    Parameters:
    data (array): Time series data
    time_delay (int): Number of time steps between snapshots
    
    Returns:
    X (array): Matrix of snapshots at time t
    Y (array): Matrix of snapshots at time t+1
    """
    n = len(data)
    X = data[:-time_delay].reshape(-1, n-time_delay)
    Y = data[time_delay:].reshape(-1, n-time_delay)
    return X, Y

# Example usage
time_series = np.sin(np.linspace(0, 10, 1000))
X, Y = prepare_data(time_series)

print(f"Shape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")
```

Slide 4: Implementing DMD in Python

Here's a step-by-step implementation of the DMD algorithm using NumPy and SciPy libraries.

```python
import numpy as np
from scipy import linalg

def dmd(X, Y, r=None):
    """
    Compute Dynamic Mode Decomposition
    
    Parameters:
    X (array): Matrix of snapshots at time t
    Y (array): Matrix of snapshots at time t+1
    r (int): Rank truncation (optional)
    
    Returns:
    eigenvalues (array): DMD eigenvalues
    modes (array): DMD modes
    """
    # Compute SVD of X
    U, s, Vh = linalg.svd(X, full_matrices=False)
    
    # Truncate to rank r if specified
    if r is not None:
        U = U[:, :r]
        s = s[:r]
        Vh = Vh[:r, :]
    
    # Compute DMD matrix
    A_tilde = U.conj().T @ Y @ Vh.conj().T @ np.diag(1/s)
    
    # Eigendecomposition of A_tilde
    eigenvalues, eigenvectors = linalg.eig(A_tilde)
    
    # Compute DMD modes
    modes = Y @ Vh.conj().T @ np.diag(1/s) @ eigenvectors
    
    return eigenvalues, modes

# Example usage
X = np.random.rand(100, 1000)
Y = np.random.rand(100, 1000)
eigenvalues, modes = dmd(X, Y)

print(f"Number of DMD eigenvalues: {len(eigenvalues)}")
print(f"Shape of DMD modes: {modes.shape}")
```

Slide 5: Interpreting DMD Results

Understanding the output of DMD is crucial for extracting meaningful insights from your data. Let's explore how to interpret the eigenvalues and modes.

```python
import numpy as np
import matplotlib.pyplot as plt

# Assuming we have already computed eigenvalues and modes
eigenvalues, modes = dmd(X, Y)

# Compute growth rates and frequencies
dt = 0.1  # time step
growth_rates = np.log(np.abs(eigenvalues)) / dt
frequencies = np.angle(eigenvalues) / (2 * np.pi * dt)

# Plot eigenvalues on complex plane
plt.figure(figsize=(10, 8))
plt.scatter(eigenvalues.real, eigenvalues.imag)
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.title('DMD Eigenvalues')
plt.axis('equal')
plt.grid(True)
plt.show()

# Plot mode energies
mode_energies = np.linalg.norm(modes, axis=0)
plt.figure(figsize=(10, 6))
plt.bar(range(len(mode_energies)), mode_energies)
plt.xlabel('Mode index')
plt.ylabel('Mode energy')
plt.title('DMD Mode Energies')
plt.show()
```

Slide 6: Reconstructing and Forecasting with DMD

One of the powerful applications of DMD is its ability to reconstruct the original data and forecast future states of the system.

```python
import numpy as np
import matplotlib.pyplot as plt

def dmd_reconstruct(modes, eigenvalues, b, t):
    """
    Reconstruct data using DMD
    
    Parameters:
    modes (array): DMD modes
    eigenvalues (array): DMD eigenvalues
    b (array): Initial amplitudes of modes
    t (array): Time points for reconstruction
    
    Returns:
    X_dmd (array): Reconstructed data
    """
    omega = np.log(eigenvalues)
    X_dmd = np.zeros((modes.shape[0], len(t)), dtype=complex)
    for i, ti in enumerate(t):
        X_dmd[:, i] = modes @ (b * np.exp(omega * ti))
    return X_dmd.real

# Example usage
t = np.linspace(0, 10, 1000)
X = np.sin(t) + 0.5 * np.sin(2*t) + 0.1 * np.random.randn(len(t))
X_snapshots = X.reshape(-1, len(t))

eigenvalues, modes = dmd(X_snapshots[:, :-1], X_snapshots[:, 1:])
b = np.linalg.pinv(modes) @ X_snapshots[:, 0]

X_reconstructed = dmd_reconstruct(modes, eigenvalues, b, t)

plt.figure(figsize=(12, 6))
plt.plot(t, X, label='Original')
plt.plot(t, X_reconstructed[0], label='DMD Reconstruction')
plt.legend()
plt.title('Original vs DMD Reconstruction')
plt.show()
```

Slide 7: DMD for Dimensionality Reduction

DMD can be used as a powerful tool for dimensionality reduction, particularly for time-evolving systems.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate high-dimensional data
t = np.linspace(0, 10, 1000)
X = np.zeros((50, len(t)))
for i in range(50):
    X[i] = np.sin(t + i*0.1) + 0.1 * np.random.randn(len(t))

# Apply DMD
eigenvalues, modes = dmd(X[:, :-1], X[:, 1:], r=5)  # Truncate to 5 modes

# Reconstruct using DMD
b = np.linalg.pinv(modes) @ X[:, 0]
X_dmd = dmd_reconstruct(modes, eigenvalues, b, t)

# Compare with PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X.T).T

# Plot results
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(X, aspect='auto', cmap='viridis')
plt.title('Original Data')
plt.subplot(132)
plt.imshow(X_dmd, aspect='auto', cmap='viridis')
plt.title('DMD Reconstruction')
plt.subplot(133)
plt.imshow(X_pca, aspect='auto', cmap='viridis')
plt.title('PCA Reconstruction')
plt.tight_layout()
plt.show()
```

Slide 8: Handling Noisy Data with DMD

Real-world data often contains noise. Let's explore how DMD performs with noisy data and some techniques to improve its robustness.

```python
import numpy as np
import matplotlib.pyplot as plt

def noisy_sine(t, noise_level=0.1):
    return np.sin(t) + noise_level * np.random.randn(len(t))

t = np.linspace(0, 10, 1000)
X_clean = noisy_sine(t, noise_level=0)
X_noisy = noisy_sine(t, noise_level=0.2)

# Apply DMD to clean and noisy data
eigenvalues_clean, modes_clean = dmd(X_clean.reshape(1, -1)[:, :-1], X_clean.reshape(1, -1)[:, 1:])
eigenvalues_noisy, modes_noisy = dmd(X_noisy.reshape(1, -1)[:, :-1], X_noisy.reshape(1, -1)[:, 1:])

# Reconstruct
b_clean = np.linalg.pinv(modes_clean) @ X_clean[0]
b_noisy = np.linalg.pinv(modes_noisy) @ X_noisy[0]

X_dmd_clean = dmd_reconstruct(modes_clean, eigenvalues_clean, b_clean, t)
X_dmd_noisy = dmd_reconstruct(modes_noisy, eigenvalues_noisy, b_noisy, t)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.plot(t, X_clean, label='Clean Data')
plt.plot(t, X_dmd_clean[0], label='DMD Reconstruction')
plt.legend()
plt.title('DMD on Clean Data')
plt.subplot(212)
plt.plot(t, X_noisy, label='Noisy Data')
plt.plot(t, X_dmd_noisy[0], label='DMD Reconstruction')
plt.legend()
plt.title('DMD on Noisy Data')
plt.tight_layout()
plt.show()
```

Slide 9: DMD for Anomaly Detection

DMD can be used for anomaly detection in time series data by identifying deviations from the expected behavior predicted by the DMD model.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_data_with_anomaly(t, anomaly_start, anomaly_end):
    y = np.sin(t) + 0.5 * np.sin(2*t)
    anomaly = np.zeros_like(t)
    anomaly[(t > anomaly_start) & (t < anomaly_end)] = 2
    return y + anomaly

t = np.linspace(0, 20, 2000)
X = generate_data_with_anomaly(t, 10, 12)

# Apply DMD
eigenvalues, modes = dmd(X.reshape(1, -1)[:, :-1], X.reshape(1, -1)[:, 1:])
b = np.linalg.pinv(modes) @ X[0]
X_dmd = dmd_reconstruct(modes, eigenvalues, b, t)

# Calculate residuals
residuals = np.abs(X - X_dmd[0])

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.plot(t, X, label='Original Data')
plt.plot(t, X_dmd[0], label='DMD Reconstruction')
plt.legend()
plt.title('Original Data vs DMD Reconstruction')
plt.subplot(212)
plt.plot(t, residuals)
plt.title('Residuals (Anomaly Score)')
plt.tight_layout()
plt.show()
```

Slide 10: DMD for Video Processing

DMD can be applied to video data for background subtraction and motion detection. This technique is particularly useful in surveillance and object tracking applications.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_moving_blob(frames, size):
    video = np.zeros((frames, size, size))
    for i in range(frames):
        x, y = (i % size, i % size)
        video[i, max(0, x-2):min(size, x+2), max(0, y-2):min(size, y+2)] = 1
    return video.reshape(frames, -1).T

# Generate video data
frames, size = 100, 50
video = generate_moving_blob(frames, size)

# Apply DMD
U, s, Vh = np.linalg.svd(video[:, :-1], full_matrices=False)
r = 10  # number of modes to keep
A_tilde = U[:, :r].conj().T @ video[:, 1:] @ Vh[:r, :].conj().T @ np.diag(1/s[:r])
eigenvalues, modes = np.linalg.eig(A_tilde)
modes = video[:, 1:] @ Vh[:r, :].conj().T @ np.diag(1/s[:r]) @ modes

# Reconstruct background and foreground
background = np.real(modes[:, :5] @ np.diag(eigenvalues[:5]) @ np.linalg.pinv(modes[:, :5]) @ video)
foreground = video - background

# Plot results
plt.figure(figsize=(15, 5))
for i, (title, data) in enumerate([('Original', video), ('Background', background), ('Foreground', foreground)]):
    plt.subplot(1, 3, i+1)
    plt.imshow(data[:, 50].reshape(size, size), cmap='gray')
    plt.title(title)
plt.tight_layout()
plt.show()
```

Slide 11: DMD for Fluid Dynamics Analysis

DMD is particularly useful in fluid dynamics for identifying coherent structures and analyzing flow patterns.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_vortex_shedding(nx, ny, nt):
    x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    u = np.zeros((nt, ny, nx))
    v = np.zeros((nt, ny, nx))
    
    for t in range(nt):
        u[t] = np.sin(2*np.pi*(x + 0.01*t))
        v[t] = np.cos(2*np.pi*(y + 0.01*t))
    
    return u, v

# Simulate flow
nx, ny, nt = 50, 50, 100
u, v = simulate_vortex_shedding(nx, ny, nt)

# Prepare data for DMD
X = np.reshape(u, (nx*ny, nt-1))

# Apply DMD
U, s, Vh = np.linalg.svd(X, full_matrices=False)
r = 10
A_tilde = U[:, :r].conj().T @ X[:, 1:] @ Vh[:r, :].conj().T @ np.diag(1/s[:r])
eigenvalues, modes = np.linalg.eig(A_tilde)
modes = X[:, 1:] @ Vh[:r, :].conj().T @ np.diag(1/s[:r]) @ modes

# Plot dominant mode
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(u[0], cmap='RdBu_r')
plt.title('Original Flow')
plt.subplot(122)
plt.imshow(np.real(modes[:, 0]).reshape(nx, ny), cmap='RdBu_r')
plt.title('Dominant DMD Mode')
plt.tight_layout()
plt.show()
```

Slide 12: DMD for Climate Data Analysis

DMD can be applied to climate data to identify patterns and trends over time.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_climate_data(years, locations):
    t = np.linspace(0, years, years*12)
    data = np.zeros((locations, len(t)))
    for i in range(locations):
        trend = 0.02 * t  # global warming trend
        seasonal = 5 * np.sin(2*np.pi*t)  # seasonal variation
        noise = np.random.normal(0, 1, len(t))  # random fluctuations
        data[i] = trend + seasonal + noise + 15 + i  # base temperature varies by location
    return data

# Generate climate data
years, locations = 50, 10
climate_data = generate_climate_data(years, locations)

# Apply DMD
X = climate_data[:, :-1]
Y = climate_data[:, 1:]
U, s, Vh = np.linalg.svd(X, full_matrices=False)
r = 5
A_tilde = U[:, :r].conj().T @ Y @ Vh[:r, :].conj().T @ np.diag(1/s[:r])
eigenvalues, modes = np.linalg.eig(A_tilde)

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(climate_data.T)
plt.title('Original Climate Data')
plt.subplot(212)
plt.scatter(eigenvalues.real, eigenvalues.imag)
plt.title('DMD Eigenvalues')
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.tight_layout()
plt.show()
```

Slide 13: Challenges and Limitations of DMD

While DMD is a powerful technique, it's important to be aware of its limitations and challenges.

1. Linearity Assumption: DMD assumes the underlying system dynamics are linear, which may not always hold for complex, nonlinear systems.
2. Sensitivity to Noise: DMD can be sensitive to noise in the data, potentially leading to spurious modes.
3. Choice of Rank: Selecting the appropriate number of modes to retain can be challenging and may require domain expertise.
4. Temporal Resolution: The time step between snapshots can affect the quality of the DMD results.
5. Interpretability: Interpreting the physical meaning of DMD modes can be challenging, especially for high-dimensional systems.

```python
# Pseudocode for addressing some challenges

def robust_dmd(X, Y, noise_level):
    # Add noise to data
    X_noisy = X + noise_level * np.random.randn(*X.shape)
    Y_noisy = Y + noise_level * np.random.randn(*Y.shape)
    
    # Apply DMD with rank truncation
    U, s, Vh = np.linalg.svd(X_noisy, full_matrices=False)
    r = optimal_hard_threshold(s)  # Implement optimal hard threshold method
    A_tilde = U[:, :r].conj().T @ Y_noisy @ Vh[:r, :].conj().T @ np.diag(1/s[:r])
    eigenvalues, modes = np.linalg.eig(A_tilde)
    
    return eigenvalues, modes

def optimal_hard_threshold(singular_values):
    # Implement method to determine optimal rank
    pass

# Example usage
eigenvalues, modes = robust_dmd(X, Y, noise_level=0.1)
```

Slide 14: Future Directions and Advanced DMD Techniques

DMD continues to evolve with new variants and applications being developed. Some advanced techniques include:

1. Sparsity-Promoting DMD: Identifies a small number of dynamic modes that accurately represent the system.
2. Multi-Resolution DMD: Analyzes data across different timescales.
3. Compressed DMD: Applies DMD to compressed or subsampled data for computational efficiency.
4. Koopman DMD: Extends DMD to nonlinear systems using Koopman operator theory.
5. DMD with Control: Incorporates control inputs into the DMD framework.

```python
# Pseudocode for Sparsity-Promoting DMD

def sparsity_promoting_dmd(X, Y, gamma):
    # Perform standard DMD
    eigenvalues, modes = dmd(X, Y)
    
    # Implement sparsity-promoting optimization
    def objective(alpha):
        return np.linalg.norm(X - modes @ np.diag(alpha) @ np.vander(eigenvalues, N)) + gamma * np.sum(np.abs(alpha))
    
    # Solve optimization problem
    alpha_sparse = optimize.minimize(objective, x0=np.ones_like(eigenvalues))
    
    # Select modes with non-zero coefficients
    sparse_modes = modes[:, np.abs(alpha_sparse) > threshold]
    
    return sparse_modes

# Example usage
sparse_modes = sparsity_promoting_dmd(X, Y, gamma=0.1)
```

Slide 15: Additional Resources

For those interested in diving deeper into Dynamic Mode Decomposition, here are some valuable resources:

1. "Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems" by Kutz, Brunton, Brunton, and Proctor. ArXiv preprint: [https://arxiv.org/abs/1312.0041](https://arxiv.org/abs/1312.0041)
2. "On Dynamic Mode Decomposition: Theory and Applications" by Tu, Rowley, Luchtenburg, Brunton, and Kutz. ArXiv preprint: [https://arxiv.org/abs/1312.0041](https://arxiv.org/abs/1312.0041)
3. "A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition" by Williams, Kevrekidis, and Rowley. ArXiv preprint: [https://arxiv.org/abs/1408.4408](https://arxiv.org/abs/1408.4408)
4. "Compressed Sensing and Dynamic Mode Decomposition" by Brunton, Proctor, and Kutz. ArXiv preprint: [https://arxiv.org/abs/1312.5186](https://arxiv.org/abs/1312.5186)

These resources provide in-depth theoretical background, advanced techniques, and various applications of DMD across different fields.

