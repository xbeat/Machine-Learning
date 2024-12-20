## Dynamic Mode Decomposition with OpenFOAM and Python
Slide 1: Introduction to Dynamic Mode Decomposition (DMD) with OpenFOAM and Python

Dynamic Mode Decomposition is a powerful data-driven technique for analyzing complex fluid dynamics systems. It extracts dominant modes from time-series data, revealing underlying patterns and coherent structures. This presentation explores DMD implementation using OpenFOAM for CFD simulations and Python for data processing and visualization.

```python
import numpy as np
from scipy import linalg

def compute_dmd(X, dt, rank=None):
    # Compute DMD of data matrix X
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    U, S, Vh = linalg.svd(X1, full_matrices=False)
    
    if rank is not None:
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
    
    Atilde = U.conj().T @ X2 @ Vh.conj().T @ np.diag(1/S)
    eigenvalues, eigenvectors = linalg.eig(Atilde)
    
    omega = np.log(eigenvalues) / dt
    modes = X2 @ Vh.conj().T @ np.diag(1/S) @ eigenvectors
    
    return omega, modes
```

Slide 2: Setting up OpenFOAM for DMD Analysis

OpenFOAM provides a robust framework for CFD simulations. To prepare for DMD analysis, we need to set up a case directory with proper boundary conditions, mesh generation, and solver settings. Here's a basic structure for an incompressible flow case:

```bash
mkdir -p DMD_case/{0,constant,system}
cd DMD_case
cp -r $FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily/* .
```

Now, let's modify the controlDict file to output data for DMD analysis:

```c
// In system/controlDict
writeControl    timeStep;
writeInterval   1;
writeFormat     binary;
writePrecision  8;
writeCompression off;
```

Slide 3: Running OpenFOAM Simulation

With the case set up, we can run the OpenFOAM simulation to generate the data for DMD analysis. This process involves mesh generation, initialization, and solving the flow equations.

```bash
blockMesh
simpleFoam
```

The simulation results will be stored in time directories, which we'll use for DMD analysis.

Slide 4: Data Extraction from OpenFOAM Results

To perform DMD on OpenFOAM results, we need to extract the velocity field data. We'll use Python with the OpenFOAM Python API to read the data.

```python
import oftest

# Load OpenFOAM case
case = oftest.Case("path/to/DMD_case")

# Get time directories
times = case.times()

# Extract velocity field data
U_data = []
for time in times:
    U = case.getField("U", time)
    U_data.append(U.internalField().reshape(-1))

# Convert to numpy array
X = np.array(U_data).T
```

Slide 5: Preprocessing Data for DMD

Before applying DMD, we need to preprocess the data. This includes centering the data and possibly reducing its dimensionality using Principal Component Analysis (PCA).

```python
import numpy as np
from sklearn.decomposition import PCA

# Center the data
X_mean = np.mean(X, axis=1, keepdims=True)
X_centered = X - X_mean

# Apply PCA for dimensionality reduction
pca = PCA(n_components=100)
X_reduced = pca.fit_transform(X_centered.T).T

print(f"Original data shape: {X.shape}")
print(f"Reduced data shape: {X_reduced.shape}")
```

Slide 6: Implementing DMD Algorithm

Now we'll implement the core DMD algorithm using the reduced data. This involves computing the dynamic modes and their corresponding eigenvalues.

```python
def dmd(X, r=10):
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    U, S, Vt = np.linalg.svd(X1, full_matrices=False)
    U_r = U[:, :r]
    S_r = np.diag(S[:r])
    Vt_r = Vt[:r, :]
    
    A_tilde = U_r.T @ X2 @ Vt_r.T @ np.linalg.inv(S_r)
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
    
    Phi = X2 @ Vt_r.T @ np.linalg.inv(S_r) @ eigenvectors
    
    return Phi, eigenvalues

Phi, lambda_vals = dmd(X_reduced, r=10)
```

Slide 7: Analyzing DMD Results

After computing the DMD modes and eigenvalues, we need to analyze their significance. This involves examining the mode amplitudes and frequencies.

```python
import matplotlib.pyplot as plt

# Compute mode amplitudes and frequencies
omega = np.log(lambda_vals) / dt
frequencies = np.imag(omega) / (2 * np.pi)
amplitudes = np.abs(Phi).sum(axis=0)

# Plot mode amplitudes vs frequencies
plt.figure(figsize=(10, 6))
plt.scatter(frequencies, amplitudes, c='r', alpha=0.5)
plt.xlabel('Frequency')
plt.ylabel('Mode Amplitude')
plt.title('DMD Mode Spectrum')
plt.show()
```

Slide 8: Reconstructing Flow Field

Using the computed DMD modes and eigenvalues, we can reconstruct the flow field and compare it with the original data.

```python
def dmd_reconstruction(Phi, lambda_vals, t):
    return np.real(Phi @ np.diag(np.power(lambda_vals, t)))

# Reconstruct flow field
t = np.arange(X.shape[1])
X_dmd = dmd_reconstruction(Phi, lambda_vals, t)

# Compare original and reconstructed data
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.imshow(X_reduced.real, aspect='auto', interpolation='nearest')
plt.title('Original Data')
plt.subplot(212)
plt.imshow(X_dmd.real, aspect='auto', interpolation='nearest')
plt.title('DMD Reconstruction')
plt.tight_layout()
plt.show()
```

Slide 9: Real-life Example: Turbulent Wake Analysis

DMD can be applied to analyze the turbulent wake behind a bluff body. Let's set up an OpenFOAM case for a cylinder in crossflow and perform DMD analysis on the resulting velocity field.

```bash
# OpenFOAM case setup
mkdir -p cylinderWake/{0,constant,system}
cd cylinderWake
cp -r $FOAM_TUTORIALS/incompressible/pimpleFoam/LES/periodicHill/constant .
cp -r $FOAM_TUTORIALS/incompressible/pimpleFoam/LES/periodicHill/system .

# Modify mesh and boundary conditions
sed -i 's/periodicHill/cylinderWake/g' system/controlDict
# Additional modifications to setup cylinder geometry
```

Slide 10: Cylinder Wake DMD Analysis

After running the OpenFOAM simulation, we can apply DMD to the velocity field data to identify dominant modes in the wake.

```python
# Extract velocity data from OpenFOAM results
# ... (similar to previous data extraction code)

# Perform DMD
Phi_wake, lambda_wake = dmd(X_wake, r=20)

# Analyze wake modes
frequencies_wake = np.imag(np.log(lambda_wake)) / (2 * np.pi * dt)
amplitudes_wake = np.abs(Phi_wake).sum(axis=0)

plt.figure(figsize=(10, 6))
plt.scatter(frequencies_wake, amplitudes_wake, c='b', alpha=0.5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Mode Amplitude')
plt.title('DMD Spectrum of Cylinder Wake')
plt.show()
```

Slide 11: Visualizing DMD Modes for Cylinder Wake

To better understand the physical meaning of DMD modes, we can visualize them in the context of the original flow field.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_dmd_mode(mode, mesh):
    x, y, z = mesh
    mode_reshaped = mode.reshape(x.shape)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(x, y, np.real(mode_reshaped), cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Mode Amplitude')
    plt.colorbar(surf)
    plt.title('DMD Mode Visualization')
    plt.show()

# Assuming we have the mesh information
x, y, z = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100), np.linspace(0, 1, 1))

# Plot the first DMD mode
plot_dmd_mode(Phi_wake[:, 0], (x, y, z))
```

Slide 12: Real-life Example: Mixing Layer Analysis

DMD can also be applied to analyze mixing layers in fluid flows. Let's set up an OpenFOAM case for a planar mixing layer and perform DMD analysis on the resulting scalar concentration field.

```bash
# OpenFOAM case setup for mixing layer
mkdir -p mixingLayer/{0,constant,system}
cd mixingLayer
cp -r $FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily/* .

# Modify boundary conditions and initial fields
sed -i 's/pitzDaily/mixingLayer/g' system/controlDict
# Additional modifications for mixing layer setup
```

Slide 13: Mixing Layer DMD Analysis

After running the OpenFOAM simulation for the mixing layer, we can apply DMD to the scalar concentration field to identify dominant mixing modes.

```python
# Extract scalar concentration data from OpenFOAM results
# ... (similar to previous data extraction code)

# Perform DMD on scalar concentration field
Phi_mixing, lambda_mixing = dmd(X_mixing, r=15)

# Analyze mixing modes
frequencies_mixing = np.imag(np.log(lambda_mixing)) / (2 * np.pi * dt)
amplitudes_mixing = np.abs(Phi_mixing).sum(axis=0)

plt.figure(figsize=(10, 6))
plt.scatter(frequencies_mixing, amplitudes_mixing, c='g', alpha=0.5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Mode Amplitude')
plt.title('DMD Spectrum of Mixing Layer')
plt.show()
```

Slide 14: Comparing DMD Results: Cylinder Wake vs. Mixing Layer

To gain insights into different flow phenomena, we can compare the DMD results from the cylinder wake and mixing layer analyses.

```python
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.scatter(frequencies_wake, amplitudes_wake, c='b', alpha=0.5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Mode Amplitude')
plt.title('Cylinder Wake DMD Spectrum')

plt.subplot(122)
plt.scatter(frequencies_mixing, amplitudes_mixing, c='g', alpha=0.5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Mode Amplitude')
plt.title('Mixing Layer DMD Spectrum')

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For further exploration of Dynamic Mode Decomposition and its applications in fluid dynamics:

1. Kutz, J. N., Brunton, S. L., Brunton, B. W., & Proctor, J. L. (2016). Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems. SIAM. ArXiv: [https://arxiv.org/abs/1409.6358](https://arxiv.org/abs/1409.6358)
2. Schmid, P. J. (2010). Dynamic mode decomposition of numerical and experimental data. Journal of Fluid Mechanics, 656, 5-28. ArXiv: [https://arxiv.org/abs/1312.0041](https://arxiv.org/abs/1312.0041)
3. Taira, K., Brunton, S. L., Dawson, S. T., Rowley, C. W., Colonius, T., McKeon, B. J., ... & Ukeiley, L. S. (2017). Modal analysis of fluid flows: An overview. AIAA Journal, 55(12), 4013-4041. ArXiv: [https://arxiv.org/abs/1702.01453](https://arxiv.org/abs/1702.01453)

