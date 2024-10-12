## Physics-Informed Neural Networks for Quantum Wave Functions
Slide 1: Introduction to Physics-Informed Neural Networks (PINNs)

Physics-Informed Neural Networks (PINNs) are a powerful tool that combines the flexibility of neural networks with the constraints of physical laws. They are designed to solve complex problems in physics and engineering by incorporating domain knowledge into the learning process.

```python
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# Create a PINN model
model = PINN()
print(model)
```

Slide 2: Quantum Wave Functions: The Basics

Quantum wave functions describe the state of a quantum system. They are complex-valued functions that contain information about the probability of finding a particle in a particular state or position.

```python
import numpy as np
import matplotlib.pyplot as plt

def psi(x, n, L):
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

L = 1  # Length of the box
n = 3  # Quantum number
x = np.linspace(0, L, 1000)
y = psi(x, n, L)

plt.plot(x, y)
plt.title(f"Wave Function for n={n}")
plt.xlabel("Position")
plt.ylabel("Amplitude")
plt.show()
```

Slide 3: Schrödinger Equation: The Heart of Quantum Mechanics

The Schrödinger equation is a fundamental equation in quantum mechanics that describes how the quantum state of a physical system changes over time.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def schrodinger(psi, x, t, V, m, hbar):
    # Finite difference method for second derivative
    d2psi = np.gradient(np.gradient(psi, x), x)
    return 1j * hbar / (2 * m) * d2psi - 1j / hbar * V(x) * psi

# Parameters
m = 1  # Mass
hbar = 1  # Reduced Planck's constant
L = 10  # Length of the box
N = 1000  # Number of spatial points
T = 5  # Total time
Nt = 100  # Number of time points

x = np.linspace(0, L, N)
t = np.linspace(0, T, Nt)
dx = x[1] - x[0]

# Initial wave function (Gaussian wave packet)
psi0 = np.exp(-(x - L/2)**2)
psi0 = psi0 / np.sqrt(np.sum(np.abs(psi0)**2) * dx)

# Potential (free particle)
V = lambda x: np.zeros_like(x)

# Solve the Schrödinger equation
solution = odeint(schrodinger, psi0, t, args=(x, V, m, hbar))

# Plot the results
plt.imshow(np.abs(solution)**2, extent=[0, L, T, 0], aspect='auto')
plt.colorbar(label='Probability density')
plt.xlabel('Position')
plt.ylabel('Time')
plt.title('Time evolution of wave function')
plt.show()
```

Slide 4: Building a PINN for the Schrödinger Equation

We can use a PINN to solve the Schrödinger equation by incorporating the physics into the loss function.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class SchrodingerPINN(nn.Module):
    def __init__(self):
        super(SchrodingerPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# Define the physics-informed loss function
def pinn_loss(model, x, t):
    psi = model(x, t)
    psi_t = torch.autograd.grad(psi[:, 0], t, create_graph=True)[0]
    psi_xx = torch.autograd.grad(torch.autograd.grad(psi[:, 0], x, create_graph=True)[0], x, create_graph=True)[0]
    
    # Schrödinger equation: i * psi_t = -0.5 * psi_xx + V * psi
    eq = psi_t + 0.5j * psi_xx
    return torch.mean(torch.abs(eq)**2)

# Training loop
model = SchrodingerPINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    optimizer.zero_grad()
    x = torch.rand(100, 1, requires_grad=True)
    t = torch.rand(100, 1, requires_grad=True)
    loss = pinn_loss(model, x, t)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Plot the results
x = torch.linspace(0, 1, 100).view(-1, 1)
t = torch.linspace(0, 1, 100).view(-1, 1)
X, T = torch.meshgrid(x.squeeze(), t.squeeze())
XT = torch.stack([X.flatten(), T.flatten()], dim=1)

with torch.no_grad():
    psi = model(XT[:, 0].view(-1, 1), XT[:, 1].view(-1, 1))

plt.figure(figsize=(10, 8))
plt.contourf(X, T, psi[:, 0].view(100, 100).abs(), levels=20)
plt.colorbar(label='|ψ|')
plt.xlabel('x')
plt.ylabel('t')
plt.title('PINN Solution of Schrödinger Equation')
plt.show()
```

Slide 5: Real-Life Example: Quantum Harmonic Oscillator

The quantum harmonic oscillator is a fundamental model in quantum mechanics, describing systems like vibrating molecules or electromagnetic fields in a cavity.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite

def psi_n(x, n, m=1, omega=1, hbar=1):
    # Quantum harmonic oscillator wave function
    xi = np.sqrt(m * omega / hbar) * x
    N = 1 / np.sqrt(2**n * np.math.factorial(n)) * (m * omega / (np.pi * hbar))**(1/4)
    return N * np.exp(-xi**2 / 2) * hermite(n)(xi)

x = np.linspace(-10, 10, 1000)
plt.figure(figsize=(12, 8))

for n in range(4):
    y = psi_n(x, n)
    plt.plot(x, y + n, label=f'n = {n}')

plt.title('Quantum Harmonic Oscillator Wave Functions')
plt.xlabel('Position')
plt.ylabel('Wave Function (shifted for clarity)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 6: Implementing a PINN for the Quantum Harmonic Oscillator

We can use a PINN to solve the time-independent Schrödinger equation for the quantum harmonic oscillator.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class HarmonicOscillatorPINN(nn.Module):
    def __init__(self):
        super(HarmonicOscillatorPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x)

def pinn_loss(model, x):
    psi = model(x)
    psi_xx = torch.autograd.grad(torch.autograd.grad(psi, x, create_graph=True)[0], x, create_graph=True)[0]
    
    # Time-independent Schrödinger equation for harmonic oscillator: -0.5 * psi_xx + 0.5 * x^2 * psi = E * psi
    V = 0.5 * x**2
    eq = -0.5 * psi_xx + V * psi - psi  # Assuming E = 1 for ground state
    return torch.mean(eq**2)

# Training loop
model = HarmonicOscillatorPINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5000):
    optimizer.zero_grad()
    x = torch.linspace(-5, 5, 100, requires_grad=True).view(-1, 1)
    loss = pinn_loss(model, x)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Plot the results
x = torch.linspace(-5, 5, 1000).view(-1, 1)
with torch.no_grad():
    psi = model(x)

plt.figure(figsize=(10, 6))
plt.plot(x, psi.numpy())
plt.title('PINN Solution for Quantum Harmonic Oscillator Ground State')
plt.xlabel('Position')
plt.ylabel('Wave Function')
plt.grid(True)
plt.show()
```

Slide 7: Real-Life Example: Particle in a Box

The particle in a box is another fundamental quantum mechanical system, often used to model electrons in metals or semiconductor quantum wells.

```python
import numpy as np
import matplotlib.pyplot as plt

def psi_n(x, n, L):
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

def E_n(n, L, m=1, hbar=1):
    return (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)

L = 1  # Length of the box
x = np.linspace(0, L, 1000)
plt.figure(figsize=(12, 8))

for n in range(1, 5):
    y = psi_n(x, n, L)
    E = E_n(n, L)
    plt.plot(x, y + n - 1, label=f'n = {n}, E = {E:.2f}')

plt.title('Particle in a Box Wave Functions')
plt.xlabel('Position')
plt.ylabel('Wave Function (shifted for clarity)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 8: PINN for Particle in a Box

Let's implement a PINN to solve the time-independent Schrödinger equation for a particle in a box.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class ParticleInBoxPINN(nn.Module):
    def __init__(self):
        super(ParticleInBoxPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x) * x * (1 - x)  # Enforce boundary conditions

def pinn_loss(model, x):
    psi = model(x)
    psi_xx = torch.autograd.grad(torch.autograd.grad(psi, x, create_graph=True)[0], x, create_graph=True)[0]
    
    # Time-independent Schrödinger equation: -0.5 * psi_xx = E * psi
    eq = -0.5 * psi_xx - psi  # Assuming E = 1 for simplicity
    return torch.mean(eq**2)

# Training loop
model = ParticleInBoxPINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5000):
    optimizer.zero_grad()
    x = torch.linspace(0, 1, 100, requires_grad=True).view(-1, 1)
    loss = pinn_loss(model, x)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Plot the results
x = torch.linspace(0, 1, 1000).view(-1, 1)
with torch.no_grad():
    psi = model(x)

plt.figure(figsize=(10, 6))
plt.plot(x, psi.numpy())
plt.title('PINN Solution for Particle in a Box (Ground State)')
plt.xlabel('Position')
plt.ylabel('Wave Function')
plt.grid(True)
plt.show()
```

Slide 9: Handling Complex Wave Functions

Quantum wave functions are generally complex-valued. Let's modify our PINN to handle complex wave functions.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class ComplexPINN(nn.Module):
    def __init__(self):
        super(ComplexPINN, self).__init__()
        self.real_net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
        self.imag_net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        real = self.real_net(x)
        imag = self.imag_net(x)
        return torch.complex(real, imag)

def pinn_loss(model, x):
    psi = model(x)
    psi_xx = torch.autograd.grad(torch.autograd.grad(psi.real, x, create_graph=True)[0], x, create_graph=True)[0] + \
             1j * torch.autograd.grad(torch.autograd.grad(psi.imag, x, create_graph=True)[0], x, create_graph=True)[0]
    
    V = 0.5 * x**2  # Harmonic oscillator potential
    eq = -0.5 * psi_xx + V * psi - psi  # Assuming E = 1 for ground state
    return torch.mean(torch.abs(eq)**2)

model = ComplexPINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5000):
    optimizer.zero_grad()
    x = torch.linspace(-5, 5, 100, requires_grad=True).view(-1, 1)
    loss = pinn_loss(model, x)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

x = torch.linspace(-5, 5, 1000).view(-1, 1)
with torch.no_grad():
    psi = model(x)

plt.figure(figsize=(10, 6))
plt.plot(x, psi.abs().numpy())
plt.title('Complex PINN Solution for Quantum Harmonic Oscillator')
plt.xlabel('Position')
plt.ylabel('|ψ|')
plt.grid(True)
plt.show()
```

Slide 10: Time-Dependent Schrödinger Equation

Let's extend our PINN to solve the time-dependent Schrödinger equation, which describes how quantum states evolve over time.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class TimeDependentPINN(nn.Module):
    def __init__(self):
        super(TimeDependentPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        outputs = self.net(inputs)
        return torch.complex(outputs[:, 0:1], outputs[:, 1:2])

def pinn_loss(model, x, t):
    psi = model(x, t)
    
    psi_t = torch.autograd.grad(psi, t, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    psi_xx = torch.autograd.grad(torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0], x, create_graph=True)[0]
    
    V = 0.5 * x**2  # Harmonic oscillator potential
    eq = 1j * psi_t + 0.5 * psi_xx - V * psi
    return torch.mean(torch.abs(eq)**2)

model = TimeDependentPINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5000):
    optimizer.zero_grad()
    x = torch.linspace(-5, 5, 20, requires_grad=True).view(-1, 1)
    t = torch.linspace(0, 1, 20, requires_grad=True).view(-1, 1)
    x, t = torch.meshgrid(x.squeeze(), t.squeeze())
    x = x.reshape(-1, 1)
    t = t.reshape(-1, 1)
    
    loss = pinn_loss(model, x, t)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

x = torch.linspace(-5, 5, 100).view(-1, 1)
t = torch.linspace(0, 1, 100).view(-1, 1)
x, t = torch.meshgrid(x.squeeze(), t.squeeze())
x = x.reshape(-1, 1)
t = t.reshape(-1, 1)

with torch.no_grad():
    psi = model(x, t)

plt.figure(figsize=(10, 8))
plt.contourf(x.reshape(100, 100), t.reshape(100, 100), psi.abs().reshape(100, 100), levels=20)
plt.colorbar(label='|ψ|')
plt.title('Time-Dependent PINN Solution')
plt.xlabel('Position')
plt.ylabel('Time')
plt.show()
```

Slide 11: Incorporating Boundary Conditions

Boundary conditions are crucial in quantum mechanics. Let's modify our PINN to enforce specific boundary conditions.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class BoundaryPINN(nn.Module):
    def __init__(self):
        super(BoundaryPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x) * x * (1 - x)  # Enforce ψ(0) = ψ(1) = 0

def pinn_loss(model, x):
    psi = model(x)
    psi_xx = torch.autograd.grad(torch.autograd.grad(psi, x, create_graph=True)[0], x, create_graph=True)[0]
    
    eq = -0.5 * psi_xx - psi  # Assuming E = 1 for simplicity
    return torch.mean(eq**2)

model = BoundaryPINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5000):
    optimizer.zero_grad()
    x = torch.linspace(0, 1, 100, requires_grad=True).view(-1, 1)
    loss = pinn_loss(model, x)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

x = torch.linspace(0, 1, 1000).view(-1, 1)
with torch.no_grad():
    psi = model(x)

plt.figure(figsize=(10, 6))
plt.plot(x, psi.numpy())
plt.title('PINN Solution with Boundary Conditions')
plt.xlabel('Position')
plt.ylabel('Wave Function')
plt.grid(True)
plt.show()
```

Slide 12: Eigenvalue Problems in Quantum Mechanics

Many quantum mechanical problems involve finding eigenvalues and eigenfunctions. Let's use a PINN to solve an eigenvalue problem.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class EigenPINN(nn.Module):
    def __init__(self):
        super(EigenPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
        self.E = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.net(x)

def pinn_loss(model, x):
    psi = model(x)
    psi_xx = torch.autograd.grad(torch.autograd.grad(psi, x, create_graph=True)[0], x, create_graph=True)[0]
    
    V = 0.5 * x**2  # Harmonic oscillator potential
    eq = -0.5 * psi_xx + V * psi - model.E * psi
    return torch.mean(eq**2) + torch.abs(torch.sum(psi**2) - 1)  # Normalization constraint

model = EigenPINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10000):
    optimizer.zero_grad()
    x = torch.linspace(-5, 5, 100, requires_grad=True).view(-1, 1)
    loss = pinn_loss(model, x)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, E: {model.E.item()}')

x = torch.linspace(-5, 5, 1000).view(-1, 1)
with torch.no_grad():
    psi = model(x)

plt.figure(figsize=(10, 6))
plt.plot(x, psi.numpy())
plt.title(f'PINN Eigenfunction (E = {model.E.item():.2f})')
plt.xlabel('Position')
plt.ylabel('Wave Function')
plt.grid(True)
plt.show()
```

Slide 13: Coupling PINNs with Quantum Chemistry

PINNs can be used to solve complex quantum chemistry problems. Here's a simplified example of using a PINN to approximate the electronic structure of a hydrogen atom.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class HydrogenPINN(nn.Module):
    def __init__(self):
        super(HydrogenPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
        self.E = nn.Parameter(torch.randn(1))

    def forward(self, x, y, z):
        r = torch.sqrt(x**2 + y**2 + z**2)
        return self.net(torch.cat([x, y, z], dim=1)) * torch.exp(-r)

def pinn_loss(model, x, y, z):
    psi = model(x, y, z)
    r = torch.sqrt(x**2 + y**2 + z**2)
    
    grad_x = torch.autograd.grad(psi, x, create_graph=True)[0]
    grad_y = torch.autograd.grad(psi, y, create_graph=True)[0]
    grad_z = torch.autograd.grad(psi, z, create_graph=True)[0]
    
    laplacian = torch.autograd.grad(grad_x, x, create_graph=True)[0] + \
                torch.autograd.grad(grad_y, y, create_graph=True)[0] + \
                torch.autograd.grad(grad_z, z, create_graph=True)[0]
    
    eq = -0.5 * laplacian - 1/r * psi - model.E * psi
    return torch.mean(eq**2) + torch.abs(torch.sum(psi**2) - 1)

model = HydrogenPINN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10000):
    optimizer.zero_grad()
    x = torch.linspace(-5, 5, 20, requires_grad=True).view(-1, 1)
    y = torch.linspace(-5, 5, 20, requires_grad=True).view(-1, 1)
    z = torch.linspace(-5, 5, 20, requires_grad=True).view(-1, 1)
    x, y, z = torch.meshgrid(x.squeeze(), y.squeeze(), z.squeeze())
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    
    loss = pinn_loss(model, x, y, z)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, E: {model.E.item()}')

x = torch.linspace(-5, 5, 100).view(-1, 1)
y = torch.zeros_like(x)
z = torch.zeros_like(x)

with torch.no_grad():
    psi = model(x, y, z)

plt.figure(figsize=(10, 6))
plt.plot(x, psi.numpy())
plt.title(f'PINN Hydrogen Atom Ground State (E = {model.E.item():.2f})')
plt.xlabel('x')
plt.ylabel('Wave Function')
plt.grid(True)
plt.show()
```

Slide 14: Additional Resources

For further exploration of Physics-Informed Neural Networks and their applications in quantum mechanics, consider the following resources:

1. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations" by M. Raissi, P. Perdikaris, and G.E. Karniadakis (2019). ArXiv: [https://arxiv.org/abs/1711.10561](https://arxiv.org/abs/1711.10561)
2. "Solving the electronic Schrödinger equation for multiple nuclear geometries with weight-sharing deep neural networks" by J. Hermann, Z. Schätzle, and F. Noé (2020). ArXiv: [https://arxiv.org/abs/1909.08423](https://arxiv.org/abs/1909.08423)
3. "Quantum chemistry with neural networks" by J.S. Smith, O. Isayev, and A.E. Roitberg (2017). ArXiv: [https://arxiv.org/abs/1701.06715](https://arxiv.org/abs/1701.06715)

These papers provide in-depth discussions on the application of neural networks to quantum mechanical problems and offer insights into more advanced techniques and considerations.

