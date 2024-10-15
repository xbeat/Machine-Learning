## Challenges and Applications of Dynamic Mode Decomposition
Slide 1: Understanding Dynamic Mode Decomposition (DMD)

Dynamic Mode Decomposition (DMD) is a powerful data-driven method for analyzing complex dynamical systems. It extracts spatiotemporal coherent structures from high-dimensional data, providing insights into the system's behavior and underlying dynamics.

```python
import numpy as np
from scipy import linalg

def dmd(X, r=None):
    # Perform DMD on data matrix X
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    U, S, Vh = linalg.svd(X1, full_matrices=False)
    
    if r is not None:
        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:r, :]
    
    A_tilde = U.conj().T @ X2 @ Vh.conj().T @ np.diag(1/S)
    
    eigenvalues, eigenvectors = linalg.eig(A_tilde)
    modes = X2 @ Vh.conj().T @ np.diag(1/S) @ eigenvectors
    
    return modes, eigenvalues
```

Slide 2: Challenges with Translational Invariance

Translational invariance poses a challenge for DMD when analyzing systems with moving patterns. DMD struggles to capture the essence of translating structures, often resulting in mode shapes that don't accurately represent the system's behavior.

```python
import numpy as np
import matplotlib.pyplot as plt

def create_translating_wave(x, t, k, omega):
    return np.sin(k * x - omega * t)

x = np.linspace(0, 10, 100)
t = np.linspace(0, 5, 50)
k, omega = 2*np.pi, 2*np.pi

data = np.array([create_translating_wave(x, ti, k, omega) for ti in t]).T

plt.figure(figsize=(10, 4))
plt.imshow(data, aspect='auto', extent=[t.min(), t.max(), x.min(), x.max()])
plt.colorbar(label='Amplitude')
plt.title('Translating Wave')
plt.xlabel('Time')
plt.ylabel('Space')
plt.show()
```

Slide 3: Rotational Invariance Issues

Similar to translational invariance, rotational invariance presents difficulties for DMD. Systems with rotating structures or periodic motions can lead to inaccurate mode decompositions, as DMD struggles to capture the rotational nature of the dynamics.

```python
import numpy as np
import matplotlib.pyplot as plt

def create_rotating_pattern(x, y, t, omega):
    return np.sin(x * np.cos(omega * t) + y * np.sin(omega * t))

x, y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
t = np.linspace(0, 2*np.pi, 50)
omega = 1

data = np.array([create_rotating_pattern(x, y, ti, omega).flatten() for ti in t]).T

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(data[:, 0].reshape(50, 50), cmap='viridis')
axes[0].set_title('Initial State')
axes[1].imshow(data[:, -1].reshape(50, 50), cmap='viridis')
axes[1].set_title('Final State')
plt.tight_layout()
plt.show()
```

Slide 4: Challenges with Transient Phenomena

DMD often struggles to capture transient phenomena accurately. These short-lived events or rapid changes in system behavior can be overshadowed by more dominant, long-term dynamics in the DMD analysis.

```python
import numpy as np
import matplotlib.pyplot as plt

def create_transient_signal(t):
    return np.exp(-t) * np.sin(2*np.pi*t) + 0.5*np.sin(0.5*np.pi*t)

t = np.linspace(0, 10, 1000)
signal = create_transient_signal(t)

plt.figure(figsize=(10, 4))
plt.plot(t, signal)
plt.title('Transient Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
```

Slide 5: Importance of Understanding DMD Limitations

Recognizing the limitations of DMD is crucial for accurate data analysis. By understanding these constraints, researchers can make informed decisions about when to use DMD and when to explore alternative methods.

```python
def dmd_analysis_workflow(data, known_limitations):
    if 'translational_invariance' in known_limitations:
        data = preprocess_for_translation(data)
    if 'rotational_invariance' in known_limitations:
        data = preprocess_for_rotation(data)
    if 'transient_phenomena' in known_limitations:
        data = isolate_transients(data)
    
    modes, eigenvalues = dmd(data)
    return analyze_results(modes, eigenvalues, known_limitations)

def preprocess_for_translation(data):
    # Implement translation-aware preprocessing
    pass

def preprocess_for_rotation(data):
    # Implement rotation-aware preprocessing
    pass

def isolate_transients(data):
    # Implement transient isolation techniques
    pass

def analyze_results(modes, eigenvalues, known_limitations):
    # Analyze DMD results considering known limitations
    pass
```

Slide 6: Overcoming Translational Invariance

To address translational invariance, researchers can employ techniques such as shifting reference frames or using relative coordinates. These approaches help DMD capture the underlying dynamics more accurately.

```python
import numpy as np

def shift_reference_frame(data, shift_vector):
    shifted_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        shifted_data[:, i] = np.roll(data[:, i], shift_vector[i], axis=0)
    return shifted_data

# Example usage
data = np.random.rand(100, 10)  # 100 spatial points, 10 time steps
shift_vector = np.arange(10)  # Shift increases with each time step
shifted_data = shift_reference_frame(data, shift_vector)

print("Original data shape:", data.shape)
print("Shifted data shape:", shifted_data.shape)
```

Slide 7: Addressing Rotational Invariance

To handle rotational invariance, techniques like polar coordinate transformation or proper orthogonal decomposition (POD) can be used in conjunction with DMD. These methods help preserve the rotational information in the system.

```python
import numpy as np

def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def transform_to_polar(data, x, y):
    r, theta = cartesian_to_polar(x, y)
    polar_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        polar_data[:, i] = np.interp(np.linspace(0, 2*np.pi, len(r)), theta, data[:, i])
    return polar_data

# Example usage
x, y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
data = np.random.rand(2500, 10)  # 50x50 spatial points, 10 time steps
polar_data = transform_to_polar(data, x.flatten(), y.flatten())

print("Original data shape:", data.shape)
print("Polar data shape:", polar_data.shape)
```

Slide 8: Capturing Transient Phenomena

To better capture transient phenomena, researchers can employ windowed DMD or multi-resolution DMD techniques. These approaches allow for a more detailed analysis of short-lived events within the system.

```python
import numpy as np

def windowed_dmd(data, window_size, overlap):
    n_samples, n_features = data.shape
    step = window_size - overlap
    n_windows = (n_samples - window_size) // step + 1
    
    modes_list = []
    eigenvalues_list = []
    
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        window_data = data[start:end, :]
        
        modes, eigenvalues = dmd(window_data)
        modes_list.append(modes)
        eigenvalues_list.append(eigenvalues)
    
    return modes_list, eigenvalues_list

# Example usage
data = np.random.rand(1000, 50)  # 1000 time steps, 50 features
window_size = 100
overlap = 50

windowed_modes, windowed_eigenvalues = windowed_dmd(data, window_size, overlap)

print("Number of windows:", len(windowed_modes))
print("Modes shape for each window:", windowed_modes[0].shape)
print("Eigenvalues shape for each window:", windowed_eigenvalues[0].shape)
```

Slide 9: Alternative Methods to DMD

When DMD's limitations are significant, alternative methods can be explored. Techniques such as Koopman Mode Decomposition (KMD) or Time-Delay Embedding can provide complementary insights into complex dynamical systems.

```python
import numpy as np
from scipy import linalg

def koopman_mode_decomposition(X, Y, r=None):
    U, S, Vh = linalg.svd(X, full_matrices=False)
    
    if r is not None:
        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:r, :]
    
    K = Y @ Vh.conj().T @ np.diag(1/S) @ U.conj().T
    
    eigenvalues, eigenvectors = linalg.eig(K)
    modes = Y @ Vh.conj().T @ np.diag(1/S) @ eigenvectors
    
    return modes, eigenvalues

# Example usage
X = np.random.rand(100, 50)  # 100 features, 50 time steps
Y = np.random.rand(100, 50)  # 100 features, 50 time steps (shifted by one step)

kmd_modes, kmd_eigenvalues = koopman_mode_decomposition(X, Y, r=10)

print("KMD Modes shape:", kmd_modes.shape)
print("KMD Eigenvalues shape:", kmd_eigenvalues.shape)
```

Slide 10: Real-Life Example: Fluid Dynamics

DMD has been successfully applied in fluid dynamics to analyze complex flow patterns. However, when dealing with rotating or translating flow structures, researchers must be cautious and consider the method's limitations.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_vortex_street(nx, ny, nt, Re):
    # Simplified vortex street simulation
    x = np.linspace(0, 10, nx)
    y = np.linspace(-2, 2, ny)
    t = np.linspace(0, 5, nt)
    
    X, Y, T = np.meshgrid(x, y, t)
    
    u = np.sin(X - 0.1*T) * np.exp(-Y**2/Re)
    v = 0.1 * np.cos(X - 0.1*T) * Y * np.exp(-Y**2/Re)
    
    return u, v

nx, ny, nt = 50, 30, 100
Re = 0.5
u, v = simulate_vortex_street(nx, ny, nt, Re)

plt.figure(figsize=(10, 5))
plt.quiver(u[:,:,0], v[:,:,0])
plt.title('Vortex Street Velocity Field (t=0)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Perform DMD on the velocity field
velocity_field = u.reshape(nx*ny, nt) + 1j*v.reshape(nx*ny, nt)
dmd_modes, dmd_eigenvalues = dmd(velocity_field, r=10)

print("Number of DMD modes:", dmd_modes.shape[1])
print("Number of DMD eigenvalues:", len(dmd_eigenvalues))
```

Slide 11: Real-Life Example: Climate Data Analysis

Climate data often exhibits complex spatiotemporal patterns, making it an interesting application for DMD. However, the presence of multiple timescales and potential transient events requires careful consideration of DMD's limitations.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_climate_data(nx, ny, nt):
    x = np.linspace(0, 100, nx)
    y = np.linspace(0, 50, ny)
    t = np.linspace(0, 10, nt)
    
    X, Y, T = np.meshgrid(x, y, t)
    
    # Simulated temperature data with seasonal and long-term trends
    temp = 15 + 5*np.sin(2*np.pi*T) + 0.1*T + 2*np.sin(0.1*X) * np.cos(0.1*Y)
    
    # Add some noise and transient events
    temp += np.random.normal(0, 0.5, temp.shape)
    temp[nx//4:nx//2, ny//4:ny//2, nt//2:nt//2+10] += 3  # Simulated heatwave
    
    return temp

nx, ny, nt = 50, 25, 365
climate_data = generate_climate_data(nx, ny, nt)

plt.figure(figsize=(10, 5))
plt.imshow(climate_data[:,:,0], aspect='auto', cmap='viridis')
plt.title('Simulated Climate Data (t=0)')
plt.colorbar(label='Temperature (°C)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Perform DMD on the climate data
reshaped_data = climate_data.reshape(nx*ny, nt)
dmd_modes, dmd_eigenvalues = dmd(reshaped_data, r=10)

print("Number of DMD modes:", dmd_modes.shape[1])
print("Number of DMD eigenvalues:", len(dmd_eigenvalues))
```

Slide 12: Interpreting DMD Results

When interpreting DMD results, it's crucial to consider the method's limitations. Analyze the modes and eigenvalues in the context of the system's known behavior and be cautious of overinterpreting results, especially for systems with known translational or rotational invariances.

```python
import numpy as np
import matplotlib.pyplot as plt

def interpret_dmd_results(modes, eigenvalues, dt):
    growth_rates = np.log(np.abs(eigenvalues)) / dt
    frequencies = np.angle(eigenvalues) / (2 * np.pi * dt)
    
    sorted_indices = np.argsort(np.abs(growth_rates))[::-1]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.stem(growth_rates[sorted_indices])
    plt.title('DMD Mode Growth Rates')
    plt.xlabel('Mode Index')
    plt.ylabel('Growth Rate')
    
    plt.subplot(122)
    plt.stem(frequencies[sorted_indices])
    plt.title('DMD Mode Frequencies')
    plt.xlabel('Mode Index')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return growth_rates, frequencies

# Example usage
dt = 0.1
growth_rates, frequencies = interpret_dmd_results(dmd_modes, dmd_eigenvalues, dt)

print("Top 3 growing modes:")
for i in range(3):
    print(f"Mode {i+1}: Growth rate = {growth_rates[i]:.4f}, Frequency = {frequencies[i]:.4f}")
```

Slide 13: Enhancing DMD with Machine Learning

To address some of DMD's limitations, researchers are exploring ways to enhance it with machine learning techniques. These hybrid approaches can help capture more complex dynamics and improve the method's performance on challenging datasets.

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

def nonlinear_dmd(X, degree=2):
    # Create a nonlinear feature mapping
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X.T)
    
    # Perform DMD on the nonlinear features
    X1 = X_poly[:-1, :]
    X2 = X_poly[1:, :]
    
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    A_tilde = U.conj().T @ X2 @ Vh.conj().T @ np.diag(1/S)
    
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
    modes = X2 @ Vh.conj().T @ np.diag(1/S) @ eigenvectors
    
    return modes, eigenvalues, poly

# Example usage
X = np.random.rand(100, 50)  # 100 features, 50 time steps
nl_modes, nl_eigenvalues, poly = nonlinear_dmd(X, degree=2)

print("Nonlinear DMD Modes shape:", nl_modes.shape)
print("Nonlinear DMD Eigenvalues shape:", nl_eigenvalues.shape)
```

Slide 14: Future Directions and Ongoing Research

Researchers continue to explore new ways to overcome DMD's limitations and extend its capabilities. Some promising directions include developing adaptive DMD algorithms, incorporating physics-informed constraints, and combining DMD with other data-driven methods.

```python
def adaptive_dmd(X, max_rank, tolerance):
    r = 1
    error = float('inf')
    
    while r <= max_rank and error > tolerance:
        modes, eigenvalues = dmd(X, r=r)
        reconstructed = reconstruct_data(modes, eigenvalues, X.shape[1])
        error = np.linalg.norm(X - reconstructed) / np.linalg.norm(X)
        
        if error <= tolerance:
            break
        
        r += 1
    
    return modes, eigenvalues, r, error

def reconstruct_data(modes, eigenvalues, n_samples):
    time_dynamics = np.vander(eigenvalues, n_samples, increasing=True)
    return np.real(modes @ time_dynamics)

# Example usage
X = np.random.rand(100, 50)  # 100 features, 50 time steps
max_rank = 20
tolerance = 1e-3

modes, eigenvalues, rank, error = adaptive_dmd(X, max_rank, tolerance)

print(f"Optimal rank: {rank}")
print(f"Reconstruction error: {error:.6f}")
```

Slide 15: Additional Resources

For those interested in diving deeper into DMD and its applications, here are some valuable resources:

1. Kutz, J. N., Brunton, S. L., Brunton, B. W., & Proctor, J. L. (2016). Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems. SIAM. ArXiv: [https://arxiv.org/abs/1312.0041](https://arxiv.org/abs/1312.0041)
2. Schmid, P. J. (2010). Dynamic mode decomposition of numerical and experimental data. Journal of Fluid Mechanics, 656, 5-28. ArXiv: [https://arxiv.org/abs/1010.3658](https://arxiv.org/abs/1010.3658)
3. Brunton, S. L., Kutz, J. N. (2019). Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control. Cambridge University Press. ArXiv: [https://arxiv.org/abs/1609.00377](https://arxiv.org/abs/1609.00377)

These resources provide in-depth explanations of DMD theory, applications, and recent advancements in the field.

