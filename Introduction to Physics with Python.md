## Introduction to Physics with Python

Slide 1: Introduction to Physics with Python "Exploring Physics with Python" This slide deck will cover fundamental physics concepts and demonstrate how to simulate them using Python.

Slide 2: Kinematics "Kinematics: The Study of Motion" Learn how to calculate position, velocity, and acceleration using Python. Code Example:

```python
# Calculate distance traveled
initial_velocity = 10  # m/s
acceleration = 2  # m/s^2
time = 5  # s
distance = initial_velocity * time + 0.5 * acceleration * time ** 2
print(f"Distance traveled: {distance} meters")
```

Slide 3: Newton's Laws of Motion "Newton's Laws: The Fundamentals" Explore Newton's three laws of motion and how to apply them in Python. Code Example:

```python
# Calculate force using Newton's Second Law
mass = 2  # kg
acceleration = 3  # m/s^2
force = mass * acceleration
print(f"Force required: {force} Newtons")
```

Slide 4: Work and Energy "Work and Energy: The Dynamics of Motion" Understand the relationship between work, kinetic energy, and potential energy. Code Example:

```python
# Calculate kinetic energy
mass = 5  # kg
velocity = 10  # m/s
kinetic_energy = 0.5 * mass * velocity ** 2
print(f"Kinetic energy: {kinetic_energy} Joules")
```

Slide 5: Momentum and Collisions "Momentum and Collisions: The Principles of Impact" Learn about momentum and how to simulate collisions using Python. Code Example:

```python
# Calculate momentum after an elastic collision
m1, m2 = 2, 3  # kg
v1, v2 = 5, -2  # m/s
momentum_before = m1 * v1 + m2 * v2
v1_after = (m1 - m2) / (m1 + m2) * v1 + (2 * m2) / (m1 + m2) * v2
v2_after = (2 * m1) / (m1 + m2) * v1 - (m1 - m2) / (m1 + m2) * v2
momentum_after = m1 * v1_after + m2 * v2_after
print(f"Momentum before: {momentum_before}, Momentum after: {momentum_after}")
```

Slide 6: Projectile Motion "Projectile Motion: The Art of Trajectory" Explore the principles of projectile motion and how to simulate it using Python. Code Example:

```python
# Calculate the maximum height of a projectile
initial_velocity = 30  # m/s
angle = 45  # degrees
gravity = 9.8  # m/s^2
max_height = (initial_velocity * numpy.sin(numpy.radians(angle))) ** 2 / (2 * gravity)
print(f"Maximum height: {max_height} meters")
```

Slide 7: Circular Motion "Circular Motion: The Path of Planets" Understand the mechanics of circular motion and how to model it with Python. Code Example:

```python
# Calculate the centripetal force for a planet orbiting the sun
mass = 5.97e24  # kg (Earth's mass)
velocity = 29780  # m/s (Earth's orbital velocity)
radius = 1.496e11  # m (Earth's orbital radius)
centripetal_force = mass * velocity ** 2 / radius
print(f"Centripetal force: {centripetal_force} Newtons")
```

Slide 8: Rotational Motion "Rotational Motion: The Spin of Things" Explore the principles of rotational motion and how to simulate it using Python. Code Example:

```python
# Calculate the angular momentum of a spinning object
moment_of_inertia = 0.5  # kg*m^2
angular_velocity = 10  # rad/s
angular_momentum = moment_of_inertia * angular_velocity
print(f"Angular momentum: {angular_momentum} kg*m^2/s")
```

Slide 9: Waves and Oscillations "Waves and Oscillations: The Dance of Disturbances" Understand the behavior of waves and oscillations using Python simulations. Code Example:

```python
# Simulate a simple harmonic oscillator
mass = 1  # kg
spring_constant = 10  # N/m
amplitude = 0.5  # m
time = numpy.linspace(0, 10, 1000)  # s
angular_frequency = numpy.sqrt(spring_constant / mass)
position = amplitude * numpy.cos(angular_frequency * time)
```

Slide 10: Thermodynamics "Thermodynamics: The Study of Heat and Energy" Explore the laws of thermodynamics and how to apply them using Python. Code Example:

```python
# Calculate the change in entropy for an irreversible process
initial_entropy = 10  # J/K
final_entropy = 20  # J/K
change_in_entropy = final_entropy - initial_entropy
print(f"Change in entropy: {change_in_entropy} J/K")
```

Note: This is a basic outline, and you can expand on each topic or modify the code examples as needed. Additionally, you may want to include visual aids, such as diagrams or animations, to better illustrate the concepts.

Slide 11: Numerical Methods for Solving Differential Equations "Solving Differential Equations with Python" Learn how to use numerical methods like the Euler method and Runge-Kutta methods to solve ordinary differential equations (ODEs) in Python, which are crucial for modeling many physical systems. Code Example:

```python
import numpy as np

def euler_method(f, x0, y0, xn, n):
    # f is the derivative function
    # x0, y0 are the initial conditions
    # xn is the value of x at which we want to find y
    # n is the number of steps

    x = np.linspace(x0, xn, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    h = (xn - x0) / n

    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i], y[i])

    return x, y

def f(x, y):
    # Example ODE: dy/dx = x + y
    return x + y

x0, y0, xn, n = 0, 1, 2, 10
x, y = euler_method(f, x0, y0, xn, n)
print(y[-1])  # Final value of y
```

Slide 12: Machine Learning for Particle Physics "Using ML for Particle Physics" Explore how machine learning techniques like deep neural networks can be used in particle physics for tasks like particle identification, event reconstruction, and data analysis. Code Example:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Example deep neural network for particle identification
class ParticleNet(nn.Module):
    def __init__(self):
        super(ParticleNet, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training loop
model = ParticleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    # Load training data
    inputs, labels = load_data()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

Slide 13: Molecular Dynamics Simulations "Simulating Molecular Dynamics with Python" Learn how to use Python libraries like PyMolDyn or PyRosetta to perform molecular dynamics simulations, which are essential for studying the behavior of molecules, proteins, and other biological systems. Code Example:

```python
import pymoldyn

# Create a system
system = pymoldyn.System()

# Add particles
particle_1 = pymoldyn.Particle(mass=1.0, position=[0.0, 0.0, 0.0])
particle_2 = pymoldyn.Particle(mass=1.0, position=[1.0, 0.0, 0.0])
system.add_particles([particle_1, particle_2])

# Add a bond between the particles
bond = pymoldyn.Bond(particle_1, particle_2, k=100.0, r0=1.0)
system.add_bond(bond)

# Run the simulation
integrator = pymoldyn.Integrators.Verlet()
trajectory = integrator.run(system, timestep=0.001, total_time=10.0)

# Analyze the trajectory
positions = trajectory.get_positions()
```

