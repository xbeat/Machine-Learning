## Dynamic Mode Decomposition
Slide 1: Introduction to Dynamic Mode Decomposition (DMD)

Dynamic Mode Decomposition (DMD) is a powerful data-driven method for analyzing complex dynamical systems. It extracts spatiotemporal coherent structures from high-dimensional data, providing insights into the system's behavior and evolution over time.

```python
import numpy as np
from scipy import linalg

def simple_dmd(X, dt=1):
    # X: data matrix, columns are snapshots
    # dt: time step between snapshots
    
    # Separate the data into two matrices
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    # Compute SVD of X1
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    
    # Truncate to rank r
    r = 5  # You can adjust this based on your needs
    U_r = U[:, :r]
    S_r = np.diag(S[:r])
    V_r = Vh.conj().T[:, :r]
    
    # Compute reduced Koopman operator
    A_tilde = U_r.conj().T @ X2 @ V_r @ np.linalg.inv(S_r)
    
    # Eigendecomposition of A_tilde
    eigvals, eigvecs = np.linalg.eig(A_tilde)
    
    # DMD modes
    Phi = X2 @ V_r @ np.linalg.inv(S_r) @ eigvecs
    
    # DMD eigenvalues
    omega = np.log(eigvals) / dt
    
    return Phi, omega

# Example usage
t = np.linspace(0, 10, 100)
X = np.array([np.sin(t), np.cos(t)])
Phi, omega = simple_dmd(X)

print("DMD modes shape:", Phi.shape)
print("DMD eigenvalues:", omega)
```

Slide 2: The Mathematics Behind DMD

DMD is rooted in Koopman operator theory, which transforms nonlinear dynamical systems into linear ones in a higher-dimensional space. The core idea is to approximate the Koopman operator using finite-dimensional data.

```python
import numpy as np
import matplotlib.pyplot as plt

def koopman_modes(X, dt=1):
    # X: data matrix, columns are snapshots
    # dt: time step between snapshots
    
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    # Compute DMD modes and eigenvalues
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    A = U.conj().T @ X2 @ Vh.conj().T @ np.linalg.inv(np.diag(S))
    eigvals, eigvecs = np.linalg.eig(A)
    
    # Koopman modes
    Phi = X2 @ Vh.conj().T @ np.linalg.inv(np.diag(S)) @ eigvecs
    
    # Koopman eigenvalues
    omega = np.log(eigvals) / dt
    
    return Phi, omega

# Generate example data
t = np.linspace(0, 10, 1000)
X = np.array([np.sin(t) + 0.5*np.sin(2*t), np.cos(t) + 0.5*np.cos(2*t)])

Phi, omega = koopman_modes(X)

# Plot Koopman modes
plt.figure(figsize=(12, 4))
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.plot(np.real(Phi[:, i]))
    plt.plot(np.imag(Phi[:, i]))
    plt.title(f"Koopman Mode {i+1}")
    plt.legend(['Real', 'Imaginary'])
plt.tight_layout()
plt.show()

print("Koopman eigenvalues:", omega)
```

Slide 3: SVD and Low-Rank Approximation in DMD

Singular Value Decomposition (SVD) plays a crucial role in DMD, allowing for dimensionality reduction and noise filtering. It helps in creating a low-rank approximation of the data.

```python
import numpy as np
import matplotlib.pyplot as plt

def low_rank_dmd(X, r):
    # X: data matrix, columns are snapshots
    # r: rank of approximation
    
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    
    # Truncate to rank r
    U_r = U[:, :r]
    S_r = np.diag(S[:r])
    V_r = Vh[:r, :].conj().T
    
    # Compute reduced Koopman operator
    A_tilde = U_r.conj().T @ X2 @ V_r @ np.linalg.inv(S_r)
    
    return A_tilde, U_r, S_r, V_r

# Generate noisy data
t = np.linspace(0, 10, 1000)
clean_data = np.array([np.sin(t), np.cos(t)])
noise = np.random.normal(0, 0.1, clean_data.shape)
noisy_data = clean_data + noise

# Apply low-rank DMD
r = 2  # Rank of approximation
A_tilde, U_r, S_r, V_r = low_rank_dmd(noisy_data, r)

# Reconstruct data
X_reconstructed = U_r @ A_tilde @ S_r @ V_r.conj().T

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t, noisy_data[0])
plt.title("Noisy Data")
plt.subplot(1, 2, 2)
plt.plot(t[:-1], X_reconstructed[0])
plt.title("Reconstructed Data")
plt.tight_layout()
plt.show()

print("Rank of approximation:", r)
print("Shape of reduced Koopman operator:", A_tilde.shape)
```

Slide 4: Exact DMD Algorithm

The exact DMD algorithm computes the dynamic modes and eigenvalues directly from the data matrices. This method is suitable for datasets where the number of spatial measurements is less than the number of time snapshots.

```python
import numpy as np

def exact_dmd(X, dt=1):
    # X: data matrix, columns are snapshots
    # dt: time step between snapshots
    
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    # Compute SVD of X1
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    
    # Compute Koopman operator
    A = U.conj().T @ X2 @ Vh.conj().T @ np.linalg.inv(np.diag(S))
    
    # Eigendecomposition of A
    eigvals, eigvecs = np.linalg.eig(A)
    
    # DMD modes
    Phi = X2 @ Vh.conj().T @ np.linalg.inv(np.diag(S)) @ eigvecs
    
    # DMD eigenvalues
    omega = np.log(eigvals) / dt
    
    return Phi, omega, eigvals

# Generate example data
t = np.linspace(0, 10, 200)
X = np.array([np.sin(t) + 0.5*np.sin(2*t), 
              np.cos(t) + 0.5*np.cos(2*t), 
              0.5*np.sin(3*t)])

Phi, omega, eigvals = exact_dmd(X)

print("Number of DMD modes:", Phi.shape[1])
print("DMD eigenvalues:", eigvals)
print("Continuous-time eigenvalues:", omega)

# Plot the magnitude of DMD modes
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
for i in range(Phi.shape[1]):
    plt.plot(np.abs(Phi[:, i]), label=f'Mode {i+1}')
plt.title("Magnitude of DMD Modes")
plt.legend()
plt.show()
```

Slide 5: DMD for Forecasting

One of the powerful applications of DMD is forecasting future states of a dynamical system. By extracting the underlying dynamics, DMD can predict the system's behavior beyond the original data.

```python
import numpy as np
import matplotlib.pyplot as plt

def dmd_forecast(X, num_modes, dt, forecast_steps):
    # Perform DMD
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    A = U.conj().T @ X2 @ Vh.conj().T @ np.linalg.inv(np.diag(S))
    eigvals, eigvecs = np.linalg.eig(A)
    
    # Sort eigenvalues and vectors
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[idx][:num_modes]
    eigvecs = eigvecs[:, idx][:, :num_modes]
    
    # Compute DMD modes
    Phi = X2 @ Vh.conj().T @ np.linalg.inv(np.diag(S)) @ eigvecs
    
    # Initial amplitudes
    b = np.linalg.lstsq(Phi, X[:, 0], rcond=None)[0]
    
    # Forecast
    t_forecast = np.arange(0, (X.shape[1] + forecast_steps) * dt, dt)
    dynamics = np.exp(np.outer(np.log(eigvals) / dt, t_forecast))
    X_dmd = Phi @ (dynamics * b[:, None])
    
    return X_dmd, t_forecast

# Generate example data
t = np.linspace(0, 10, 100)
X = np.array([np.sin(t) + 0.5*np.sin(2*t), 
              np.cos(t) + 0.5*np.cos(2*t)])

# Perform DMD forecast
num_modes = 2
dt = t[1] - t[0]
forecast_steps = 50
X_dmd, t_forecast = dmd_forecast(X, num_modes, dt, forecast_steps)

# Plot results
plt.figure(figsize=(12, 4))
for i in range(X.shape[0]):
    plt.plot(t, X[i], label=f'Original Data {i+1}')
    plt.plot(t_forecast, np.real(X_dmd[i]), '--', label=f'DMD Forecast {i+1}')
plt.axvline(x=t[-1], color='r', linestyle='--', label='Forecast Start')
plt.legend()
plt.title("DMD Forecasting")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

print("Forecast steps:", forecast_steps)
print("Number of modes used:", num_modes)
```

Slide 6: Companion Matrix DMD

Companion matrix DMD is an alternative formulation of the DMD algorithm that can be more computationally efficient for certain types of data, especially when the number of spatial measurements is much larger than the number of time snapshots.

```python
import numpy as np
import matplotlib.pyplot as plt

def companion_dmd(X, r):
    # X: data matrix, columns are snapshots
    # r: truncation rank
    
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    # Compute SVD and truncate
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    U_r = U[:, :r]
    S_r = np.diag(S[:r])
    V_r = Vh[:r, :].conj().T
    
    # Compute companion matrix
    C = X2.T @ U_r @ np.linalg.inv(S_r)
    
    # Eigendecomposition of companion matrix
    eigvals, eigvecs = np.linalg.eig(C)
    
    # DMD modes
    Phi = X2 @ eigvecs
    
    return Phi, eigvals

# Generate example data
t = np.linspace(0, 10, 200)
X = np.array([np.sin(t) + 0.5*np.sin(2*t), 
              np.cos(t) + 0.5*np.cos(2*t), 
              0.5*np.sin(3*t)])

# Apply companion matrix DMD
r = 3  # Truncation rank
Phi, eigvals = companion_dmd(X, r)

# Plot DMD modes
plt.figure(figsize=(12, 4))
for i in range(r):
    plt.subplot(1, r, i+1)
    plt.plot(np.real(Phi[:, i]))
    plt.plot(np.imag(Phi[:, i]))
    plt.title(f"DMD Mode {i+1}")
    plt.legend(['Real', 'Imaginary'])
plt.tight_layout()
plt.show()

print("Number of DMD modes:", Phi.shape[1])
print("DMD eigenvalues:", eigvals)
```

Slide 7: DMD with Control

DMD with control extends the standard DMD algorithm to systems with external inputs or control. This allows for modeling and analysis of systems where external factors influence the dynamics.

```python
import numpy as np
import matplotlib.pyplot as plt

def dmd_with_control(X, U, dt=1):
    X1, X2 = X[:, :-1], X[:, 1:]
    Omega = np.vstack([X1, U[:, :-1]])
    
    A, B = np.linalg.lstsq(Omega.T, X2.T, rcond=None)[0].T
    A_c = (A - np.eye(A.shape[0])) / dt
    B_c = B / dt
    
    return A_c, B_c

# Example data
t = np.linspace(0, 10, 100)
X = np.array([np.sin(t) + 0.5*np.sin(2*t), np.cos(t) + 0.5*np.cos(2*t)])
U = np.array([np.sin(3*t)])

A_c, B_c = dmd_with_control(X, U)

# Simulate system
def simulate_system(A_c, B_c, u, t, x0):
    x = [x0]
    for i in range(1, len(t)):
        dx = A_c @ x[-1] + B_c @ u[:, i-1]
        x.append(x[-1] + dx * (t[i] - t[i-1]))
    return np.array(x).T

t_sim = np.linspace(0, 15, 150)
u_sim = np.sin(3*t_sim).reshape(1, -1)
y_sim = simulate_system(A_c, B_c, u_sim, t_sim, X[:, 0])

plt.figure(figsize=(12, 4))
plt.plot(t_sim, y_sim.T)
plt.title("Simulated System Response")
plt.xlabel("Time")
plt.ylabel("State")
plt.legend(['State 1', 'State 2'])
plt.show()

print("A_c shape:", A_c.shape)
print("B_c shape:", B_c.shape)
```

Slide 8: Sparsity-Promoting DMD

Sparsity-promoting DMD aims to identify a small number of dynamic modes that accurately represent the system's behavior. This approach is particularly useful for complex systems with many degrees of freedom.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.optimize import minimize

def sparsity_promoting_dmd(X, gamma, max_iter=100):
    X1, X2 = X[:, :-1], X[:, 1:]
    U, S, Vt = svd(X1, full_matrices=False)
    A_tilde = U.T @ X2 @ Vt.T @ np.diag(1/S)
    evals, evecs = np.linalg.eig(A_tilde)
    Phi = X2 @ Vt.T @ np.diag(1/S) @ evecs
    
    def objective(alpha):
        return np.linalg.norm(X2 - Phi @ np.diag(alpha) @ np.diag(evals) @ evecs.T @ Vt, 'fro')**2 + gamma * np.sum(np.abs(alpha))
    
    alpha0 = np.ones(len(evals))
    res = minimize(objective, alpha0, method='L-BFGS-B', options={'maxiter': max_iter})
    
    return Phi, evals, res.x

# Generate example data
t = np.linspace(0, 10, 200)
X = np.array([np.sin(t) + 0.5*np.sin(2*t), np.cos(t) + 0.5*np.cos(2*t), 0.3*np.sin(3*t)])

# Apply sparsity-promoting DMD
gamma = 0.1
Phi, evals, alpha = sparsity_promoting_dmd(X, gamma)

# Plot mode amplitudes
plt.figure(figsize=(10, 4))
plt.stem(np.abs(alpha))
plt.title("Mode Amplitudes")
plt.xlabel("Mode Index")
plt.ylabel("Amplitude")
plt.show()

print("Number of modes:", len(alpha))
print("Number of non-zero modes:", np.sum(np.abs(alpha) > 1e-6))
```

Slide 9: Multiresolution DMD

Multiresolution DMD decomposes the dynamics of a system at different time scales, allowing for the analysis of complex systems with multiple characteristic frequencies.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

def multiresolution_dmd(X, levels):
    def dmd(X):
        X1, X2 = X[:, :-1], X[:, 1:]
        U, S, Vt = svd(X1, full_matrices=False)
        A = U.T @ X2 @ Vt.T @ np.diag(1/S)
        evals, evecs = np.linalg.eig(A)
        Phi = X2 @ Vt.T @ np.diag(1/S) @ evecs
        return Phi, evals

    results = []
    residual = X.()

    for _ in range(levels):
        Phi, evals = dmd(residual)
        b = np.linalg.lstsq(Phi, residual[:, 0], rcond=None)[0]
        X_dmd = np.real(Phi @ np.diag(b) @ np.vander(evals, N=X.shape[1]))
        residual -= X_dmd
        results.append((Phi, evals, b))

    return results, residual

# Generate example data
t = np.linspace(0, 10, 1000)
X = np.array([np.sin(t) + 0.5*np.sin(5*t) + 0.1*np.sin(20*t)])

# Apply multiresolution DMD
levels = 3
results, residual = multiresolution_dmd(X, levels)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(levels+2, 1, 1)
plt.plot(t, X[0])
plt.title("Original Signal")

for i, (Phi, evals, b) in enumerate(results):
    X_dmd = np.real(Phi @ np.diag(b) @ np.vander(evals, N=X.shape[1]))
    plt.subplot(levels+2, 1, i+2)
    plt.plot(t, X_dmd[0])
    plt.title(f"Level {i+1}")

plt.subplot(levels+2, 1, levels+2)
plt.plot(t, residual[0])
plt.title("Residual")

plt.tight_layout()
plt.show()

for i, (Phi, evals, b) in enumerate(results):
    print(f"Level {i+1} frequencies:", np.abs(np.log(evals)))
```

Slide 10: DMD for Streaming Data

DMD can be adapted for streaming data, allowing for real-time analysis of dynamical systems. This is particularly useful for online monitoring and control applications.

```python
import numpy as np
import matplotlib.pyplot as plt

class StreamingDMD:
    def __init__(self, n, r):
        self.n = n  # State dimension
        self.r = r  # Rank of approximation
        self.U = np.zeros((n, r))
        self.S = np.zeros(r)
        self.V = np.zeros((r, r))
        self.count = 0

    def update(self, x_new):
        self.count += 1
        if self.count == 1:
            self.x_prev = x_new
            return

        # Compute new column of Omega matrix
        omega_col = np.concatenate([self.x_prev, x_new])

        # Update SVD
        m = 2 * self.n
        Q = np.eye(m) - self.U @ self.U.T
        P = Q @ omega_col
        p_norm = np.linalg.norm(P)

        if p_norm > 1e-10:
            P /= p_norm
            self.U = np.column_stack([self.U, P[:self.n]])
            self.S = np.diag(np.concatenate([self.S, [p_norm]]))
            self.V = np.block([[self.V, np.zeros((self.r, 1))],
                               [np.zeros((1, self.r)), np.array([[1]])]])
            self.r += 1

        # Truncate to rank r
        if self.r > self.r:
            self.U = self.U[:, :self.r]
            self.S = self.S[:self.r, :self.r]
            self.V = self.V[:self.r, :self.r]

        self.x_prev = x_new

    def get_dmd_modes(self):
        A_tilde = self.U.T @ np.column_stack([self.x_prev, self.U @ self.S @ self.V.T[1:, :]]) @ \
                  np.linalg.pinv(np.column_stack([self.U @ self.S @ self.V.T[:-1, :], self.x_prev]))
        evals, evecs = np.linalg.eig(A_tilde)
        Phi = self.U @ evecs
        return Phi, evals

# Example usage
n = 3  # State dimension
r = 2  # Rank of approximation
streaming_dmd = StreamingDMD(n, r)

# Generate streaming data
t = np.linspace(0, 10, 1000)
X = np.array([np.sin(t), np.cos(t), 0.5*np.sin(2*t)])

modes = []
for x in X.T:
    streaming_dmd.update(x)
    if streaming_dmd.count > 1:
        Phi, evals = streaming_dmd.get_dmd_modes()
        modes.append(np.abs(evals))

plt.figure(figsize=(10, 4))
plt.plot(modes)
plt.title("DMD Eigenvalue Magnitudes Over Time")
plt.xlabel("Time Step")
plt.ylabel("Eigenvalue Magnitude")
plt.show()
```

Slide 11: DMD for Nonlinear Systems

While DMD is inherently linear, it can be extended to analyze nonlinear systems through various techniques such as extended DMD or kernel DMD.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler

def kernel_dmd(X, n_components=100, gamma=1.0):
    # Apply random Fourier features
    rbf_feature = RBFSampler(n_components=n_components, gamma=gamma)
    X_features = rbf_feature.fit_transform(X.T)
    
    # Perform DMD on the feature space
    X1 = X_features[:-1]
    X2 = X_features[1:]
    U, S, Vt = np.linalg.svd(X1, full_matrices=False)
    A_tilde = U.T @ X2 @ Vt.T @ np.diag(1/S)
    evals, evecs = np.linalg.eig(A_tilde)
    
    return evals, evecs, rbf_feature

# Generate nonlinear system data
def nonlinear_system(x, t):
    return np.array([
        -x[1] - x[2],
        x[0] + 0.1 * x[0] * x[2],
        0.1 * x[0] * x[1] + 0.4 - 0.1 * x[2]
    ])

t = np.linspace(0, 20, 1000)
X0 = np.array([0.1, 0, 0])
X = np.zeros((3, len(t)))
X[:, 0] = X0

for i in range(1, len(t)):
    dt = t[i] - t[i-1]
    X[:, i] = X[:, i-1] + nonlinear_system(X[:, i-1], t[i-1]) * dt

# Apply kernel DMD
evals, evecs, rbf_feature = kernel_dmd(X)

# Reconstruct and forecast
n_forecast = 500
t_forecast = np.linspace(20, 30, n_forecast)
X_reconstructed = np.zeros((3, len(t) + n_forecast))
X_features_reconstructed = rbf_feature.transform(X.T)

for i in range(len(t) + n_forecast):
    if i < len(t):
        X_reconstructed[:, i] = X[:, i]
    else:
        x_feature = rbf_feature.transform(X_reconstructed[:, i-1].reshape(1, -1))
        x_feature_next = x_feature @ evecs @ np.diag(evals) @ np.linalg.inv(evecs)
        X_reconstructed[:, i] = rbf_feature.inverse_transform(x_feature_next)

plt.figure(figsize=(12, 4))
for i in range(3):
    plt.plot(np.concatenate([t, t_forecast]), X_reconstructed[i], label=f'State {i+1}')
plt.axvline(x=20, color='r', linestyle='--', label='Forecast Start')
plt.legend()
plt.title("Nonlinear System: Original and Forecasted States")
plt.xlabel("Time")
plt.ylabel("State")
plt.show()
```

Slide 12: DMD for Parameter Estimation

DMD can be used for parameter estimation in dynamical systems, helping to identify underlying physical parameters from data.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def lorenz_system(X, t, sigma, rho, beta):
    x, y, z = X
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

def generate_data(sigma, rho, beta, X0, t):
    return odeint(lorenz_system, X0, t, args=(sigma, rho, beta))

def dmd_parameter_estimation(X, dt):
    X1, X2 = X[:-1].T, X[1:].T
    U, S, Vt = np.linalg.svd(X1, full_matrices=False)
    A = U.T @ X2 @ Vt.T @ np.diag(1/S)
    evals, evecs = np.linalg.eig(A)
    omega = np.log(evals) / dt
    return np.real(omega)

# Generate Lorenz system data
sigma, rho, beta = 10, 28, 8/3
X0 = [1, 1, 1]
t = np.linspace(0, 50, 5000)
X = generate_data(sigma, rho, beta, X0, t)

# Estimate parameters using DMD
dt = t[1] - t[0]
omega_estimated = dmd_parameter_estimation(X, dt)

# Sort eigenvalues by magnitude
idx = np.argsort(np.abs(omega_estimated))[::-1]
omega_estimated = omega_estimated[idx]

print("Estimated eigenvalues:", omega_estimated[:3])
print("True eigenvalues:", [0, -beta, -(sigma + 1)])

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, X)
plt.title("Lorenz System Trajectory")
plt.xlabel("Time")
plt.ylabel("State")
plt.legend(['x', 'y', 'z'])

plt.subplot(2, 1, 2)
plt.scatter(np.real(omega_estimated), np.imag(omega_estimated), c='r', label='Estimated')
plt.scatter([0, -beta, -(sigma + 1)], [0, 0, 0], c='b', label='True')
plt.title("Eigenvalue Comparison")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.legend()
plt.tight_layout()
plt.show()
```

Slide 13: DMD for Dimensionality Reduction

DMD can be used as a powerful tool for dimensionality reduction, capturing the most important dynamic features of high-dimensional systems.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

def dmd_dimensionality_reduction(X, r):
    X1, X2 = X[:, :-1], X[:, 1:]
    U, S, Vt = np.linalg.svd(X1, full_matrices=False)
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vr = Vt[:r, :].T
    
    Atilde = Ur.T @ X2 @ Vr @ np.linalg.inv(Sr)
    evals, evecs = np.linalg.eig(Atilde)
    
    Phi = X2 @ Vr @ np.linalg.inv(Sr) @ evecs
    return Phi, evals

# Generate high-dimensional data
X, color = make_swiss_roll(n_samples=2000, noise=0.1)
t = np.linspace(0, 1, X.shape[0])
X_time = np.column_stack((X.T, np.sin(2*np.pi*t), np.cos(2*np.pi*t)))

# Apply DMD for dimensionality reduction
r = 3  # Number of modes to retain
Phi, evals = dmd_dimensionality_reduction(X_time, r)

# Project data onto DMD modes
X_dmd = Phi @ np.linalg.pinv(Phi) @ X_time

# Plot results
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.viridis)
ax1.set_title("Original Data")

ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(X_dmd[0], X_dmd[1], X_dmd[2], c=color, cmap=plt.cm.viridis)
ax2.set_title("DMD Reduced Data")

ax3 = fig.add_subplot(133)
ax3.scatter(np.real(evals), np.imag(evals), c='r')
ax3.set_title("DMD Eigenvalues")
ax3.set_xlabel("Real Part")
ax3.set_ylabel("Imaginary Part")
ax3.axhline(y=0, color='k', linestyle='--')
ax3.axvline(x=0, color='k', linestyle='--')

plt.tight_layout()
plt.show()

print(f"Original data shape: {X_time.shape}")
print(f"Reduced data shape: {X_dmd.shape}")
```

Slide 14: DMD for Anomaly Detection

DMD can be applied to detect anomalies in time series data by identifying deviations from the expected dynamic behavior.

```python
import numpy as np
import matplotlib.pyplot as plt

def dmd_anomaly_detection(X, r, threshold):
    X1, X2 = X[:, :-1], X[:, 1:]
    U, S, Vt = np.linalg.svd(X1, full_matrices=False)
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vr = Vt[:r, :].T
    
    Atilde = Ur.T @ X2 @ Vr @ np.linalg.inv(Sr)
    evals, evecs = np.linalg.eig(Atilde)
    
    Phi = X2 @ Vr @ np.linalg.inv(Sr) @ evecs
    b = np.linalg.pinv(Phi) @ X[:, 0]
    
    X_dmd = np.zeros_like(X)
    for i in range(X.shape[1]):
        X_dmd[:, i] = np.real(Phi @ (b * evals**i))
    
    reconstruction_error = np.linalg.norm(X - X_dmd, axis=0)
    anomalies = reconstruction_error > threshold
    
    return X_dmd, reconstruction_error, anomalies

# Generate example data with anomalies
t = np.linspace(0, 10, 1000)
X = np.array([np.sin(t), np.cos(t)])
anomaly_indices = [200, 400, 600, 800]
X[:, anomaly_indices] += np.random.randn(2, len(anomaly_indices)) * 2

# Apply DMD for anomaly detection
r = 2  # Number of modes to retain
threshold = 0.5  # Anomaly threshold
X_dmd, reconstruction_error, anomalies = dmd_anomaly_detection(X, r, threshold)

# Plot results
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(t, X.T)
plt.title("Original Data")
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(3, 1, 2)
plt.plot(t, X_dmd.T)
plt.title("DMD Reconstruction")
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(3, 1, 3)
plt.plot(t, reconstruction_error)
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(t[anomalies], reconstruction_error[anomalies], color='r', label='Anomalies')
plt.title("Reconstruction Error and Detected Anomalies")
plt.xlabel("Time")
plt.ylabel("Error")
plt.legend()

plt.tight_layout()
plt.show()

print(f"Number of detected anomalies: {np.sum(anomalies)}")
print(f"Anomaly indices: {np.where(anomalies)[0]}")
```

Slide 15: Additional Resources

For those interested in diving deeper into Dynamic Mode Decomposition, here are some valuable resources:

1. "Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems" by J. Nathan Kutz, Steven L. Brunton, Bingni W. Brunton, and Joshua L. Proctor (2016) ArXiv: [https://arxiv.org/abs/1409.6358](https://arxiv.org/abs/1409.6358)
2. "On Dynamic Mode Decomposition: Theory and Applications" by Jonathan H. Tu, Clarence W. Rowley, Dirk M. Luchtenburg, Steven L. Brunton, and J. Nathan Kutz (2014) ArXiv: [https://arxiv.org/abs/1312.0041](https://arxiv.org/abs/1312.0041)
3. "A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition" by Matthew O. Williams, Ioannis G. Kevrekidis, and Clarence W. Rowley (2015) ArXiv: [https://arxiv.org/abs/1408.4408](https://arxiv.org/abs/1408.4408)
4. "Compressed Sensing and Dynamic Mode Decomposition" by Maziar S. Hemati, Clarence W. Rowley, Eric A. Deem, and Louis N. Cattafesta (2017) ArXiv: [https://arxiv.org/abs/1401.7664](https://arxiv.org/abs/1401.7664)
5. "Exact Dynamic Mode Decomposition with Nonlinear Observables" by Steven L. Brunton, Bingni W. Brunton, Joshua L. Proctor, and J. Nathan Kutz (2016) ArXiv: [https://arxiv.org/abs/1607.01067](https://arxiv.org/abs/1607.01067)

These resources provide in-depth explanations of DMD theory, applications, and extensions, suitable for readers looking to enhance their understanding of this powerful technique.

