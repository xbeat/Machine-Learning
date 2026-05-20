## Physics-Informed Neural Networks for Damped Oscillations
Slide 1: Introduction to Physics-Informed Neural Networks (PINNs)

Physics-Informed Neural Networks are a novel approach that combines the power of neural networks with physical laws. They are particularly useful for solving differential equations and modeling complex physical systems. In this presentation, we'll focus on applying PINNs to simple harmonic damped oscillations.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, t):
        return self.net(t)

# Create a simple PINN model
model = PINN()
print(model)
```

Slide 2: Simple Harmonic Damped Oscillations

Simple harmonic damped oscillations are a fundamental concept in physics, describing the motion of objects that experience a restoring force proportional to displacement and a damping force. The governing equation is:

d²x/dt² + 2βdx/dt + ω²x = 0

where x is displacement, t is time, β is the damping coefficient, and ω is the angular frequency.

```python
def damped_oscillator(t, x, dx):
    beta = 0.1  # Damping coefficient
    omega = 1.0  # Angular frequency
    d2x = -2 * beta * dx - omega**2 * x
    return d2x

# Generate data points
t = np.linspace(0, 10, 100)
x0, v0 = 1.0, 0.0  # Initial conditions
solution = scipy.integrate.odeint(lambda y, t: [y[1], damped_oscillator(t, y[0], y[1])], [x0, v0], t)

plt.plot(t, solution[:, 0])
plt.title("Damped Oscillation")
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.show()
```

Slide 3: PINN Architecture for Damped Oscillations

To model damped oscillations using PINNs, we design a neural network that takes time as input and predicts displacement. The network is trained to satisfy both the initial conditions and the governing differential equation.

```python
class DampedOscillatorPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.beta = nn.Parameter(torch.tensor([0.1]))
        self.omega = nn.Parameter(torch.tensor([1.0]))

    def forward(self, t):
        return self.net(t)

    def loss(self, t, x_true=None):
        t.requires_grad = True
        x = self(t)
        
        # Compute derivatives
        dx = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        d2x = torch.autograd.grad(dx, t, grad_outputs=torch.ones_like(dx), create_graph=True)[0]
        
        # Physics-informed loss
        physics_loss = torch.mean((d2x + 2*self.beta*dx + self.omega**2*x)**2)
        
        # Data loss (if true values are provided)
        data_loss = torch.mean((x - x_true)**2) if x_true is not None else 0
        
        return physics_loss + data_loss

model = DampedOscillatorPINN()
print(model)
```

Slide 4: Training the PINN

Training a PINN involves minimizing the combined loss from the physics-informed component and the data-driven component. We use gradient descent to optimize the network parameters.

```python
model = DampedOscillatorPINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

t_train = torch.linspace(0, 10, 100).unsqueeze(1)
x_train = torch.tensor(solution[:, 0]).unsqueeze(1).float()

for epoch in range(1000):
    optimizer.zero_grad()
    loss = model.loss(t_train, x_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Plot results
t_test = torch.linspace(0, 10, 200).unsqueeze(1)
x_pred = model(t_test).detach().numpy()

plt.plot(t_test, x_pred, label='PINN Prediction')
plt.plot(t_train, x_train, label='True Solution')
plt.legend()
plt.title("PINN vs True Solution")
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.show()
```

Slide 5: Interpreting PINN Results

After training, we can interpret the results by comparing the PINN predictions with the true solution. We can also extract the learned parameters (β and ω) from the model.

```python
print(f"Learned damping coefficient (β): {model.beta.item():.4f}")
print(f"Learned angular frequency (ω): {model.omega.item():.4f}")

# Compute relative error
relative_error = np.mean(np.abs(x_pred - solution[:, 0]) / np.abs(solution[:, 0]))
print(f"Relative error: {relative_error:.4f}")

# Plot error distribution
plt.hist(np.abs(x_pred - solution[:, 0]), bins=20)
plt.title("Error Distribution")
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.show()
```

Slide 6: Advantages of PINNs for Damped Oscillations

PINNs offer several advantages for modeling damped oscillations:

1. They can handle complex, nonlinear systems.
2. They require fewer data points than traditional machine learning approaches.
3. They can extrapolate beyond the training data by leveraging physical laws.
4. They can estimate system parameters (like damping coefficient) directly from data.

```python
# Demonstrate extrapolation
t_extrap = torch.linspace(10, 20, 100).unsqueeze(1)
x_extrap = model(t_extrap).detach().numpy()

plt.plot(t_test, x_pred, label='PINN Prediction (Training Range)')
plt.plot(t_extrap, x_extrap, label='PINN Extrapolation', linestyle='--')
plt.legend()
plt.title("PINN Extrapolation")
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.show()
```

Slide 7: Real-Life Example: Suspension System

A car's suspension system can be modeled as a damped oscillator. We can use PINNs to predict the car's vertical displacement over time after hitting a bump.

```python
class CarSuspensionPINN(DampedOscillatorPINN):
    def __init__(self):
        super().__init__()
        self.mass = nn.Parameter(torch.tensor([1000.0]))  # Car mass in kg

    def loss(self, t, x_true=None):
        t.requires_grad = True
        x = self(t)
        
        dx = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        d2x = torch.autograd.grad(dx, t, grad_outputs=torch.ones_like(dx), create_graph=True)[0]
        
        # F = ma
        physics_loss = torch.mean((self.mass * d2x + 2*self.beta*dx + self.omega**2*x)**2)
        data_loss = torch.mean((x - x_true)**2) if x_true is not None else 0
        
        return physics_loss + data_loss

car_model = CarSuspensionPINN()
# Training code would be similar to the previous example
```

Slide 8: Real-Life Example: Seismic Wave Propagation

PINNs can be used to model seismic wave propagation in the Earth's crust, which is crucial for earthquake prediction and analysis.

```python
class SeismicWavePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),  # 3 inputs: x, y, t
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.velocity = nn.Parameter(torch.tensor([5000.0]))  # Wave velocity in m/s

    def forward(self, x, y, t):
        return self.net(torch.cat([x, y, t], dim=1))

    def loss(self, x, y, t):
        u = self.forward(x, y, t)
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
        
        # Wave equation: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)
        physics_loss = torch.mean((u_tt - self.velocity**2 * (u_xx + u_yy))**2)
        
        return physics_loss

seismic_model = SeismicWavePINN()
# Training code would involve generating a grid of x, y, t values and optimizing the loss
```

Slide 9: Handling Boundary Conditions

In many physical systems, boundary conditions play a crucial role. PINNs can incorporate these conditions directly into the loss function.

```python
class BoundaryAwarePINN(DampedOscillatorPINN):
    def loss(self, t, x_true=None):
        physics_loss = super().loss(t, x_true)
        
        # Initial condition: x(0) = 1, dx/dt(0) = 0
        t_initial = torch.tensor([[0.0]])
        x_initial = self(t_initial)
        dx_initial = torch.autograd.grad(x_initial, t_initial, create_graph=True)[0]
        
        initial_loss = (x_initial - 1)**2 + dx_initial**2
        
        return physics_loss + initial_loss

boundary_model = BoundaryAwarePINN()
# Training would proceed as before, but now incorporating boundary conditions
```

Slide 10: Handling Noisy Data

In real-world scenarios, data is often noisy. PINNs can handle noisy data by adjusting the balance between the physics-informed loss and the data-driven loss.

```python
def add_noise(x, noise_level=0.1):
    return x + noise_level * torch.randn_like(x)

noisy_x_train = add_noise(x_train)

class NoisyDataPINN(DampedOscillatorPINN):
    def loss(self, t, x_true):
        physics_loss = super().loss(t)
        
        x_pred = self(t)
        data_loss = torch.mean((x_pred - x_true)**2)
        
        # Adjust the balance between physics and data loss
        return 0.8 * physics_loss + 0.2 * data_loss

noisy_model = NoisyDataPINN()

# Training with noisy data
optimizer = torch.optim.Adam(noisy_model.parameters(), lr=0.01)
for epoch in range(1000):
    optimizer.zero_grad()
    loss = noisy_model.loss(t_train, noisy_x_train)
    loss.backward()
    optimizer.step()

# Plot results
plt.scatter(t_train, noisy_x_train, label='Noisy Data', alpha=0.5)
plt.plot(t_test, noisy_model(t_test).detach().numpy(), label='PINN Prediction')
plt.legend()
plt.title("PINN with Noisy Data")
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.show()
```

Slide 11: Extending to Higher Dimensions

While we've focused on 1D damped oscillations, PINNs can be extended to higher dimensions for more complex systems.

```python
class MultiDimensionalPINN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, *args):
        return self.net(torch.cat(args, dim=1))

    def loss(self, *args):
        # Implement physics-informed loss for the specific problem
        pass

# Example: 2D wave equation
class WaveEquationPINN(MultiDimensionalPINN):
    def __init__(self):
        super().__init__(input_dim=3)  # x, y, t
        self.c = nn.Parameter(torch.tensor([1.0]))  # Wave speed

    def loss(self, x, y, t):
        u = self.forward(x, y, t)
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        
        return torch.mean((u_tt - self.c**2 * (u_xx + u_yy))**2)

wave_model = WaveEquationPINN()
# Training would involve generating a grid of x, y, t values and optimizing the loss
```

Slide 12: Optimizing PINN Performance

To improve PINN performance, consider adjusting the network architecture, using adaptive learning rates, implementing domain decomposition for complex geometries, and leveraging transfer learning for similar physical systems. Here's an example of implementing an adaptive learning rate:

```python
class AdaptivePINN(DampedOscillatorPINN):
    def __init__(self):
        super().__init__()
        self.lr = 0.01

    def adjust_learning_rate(self, optimizer, loss):
        if loss < 1e-3:
            self.lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr

model = AdaptivePINN()
optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)

for epoch in range(1000):
    optimizer.zero_grad()
    loss = model.loss(t_train, x_train)
    loss.backward()
    optimizer.step()
    model.adjust_learning_rate(optimizer, loss)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, LR: {model.lr:.6f}")
```

Slide 13: Challenges and Limitations of PINNs

While PINNs are powerful, they face some challenges:

1. Difficulty in handling stiff equations
2. Sensitivity to hyperparameters
3. Computational cost for large-scale problems
4. Potential overfitting to noise in the data

To address these issues, researchers are exploring techniques such as curriculum learning, where the complexity of the physics is gradually increased during training.

```python
class CurriculumPINN(DampedOscillatorPINN):
    def __init__(self):
        super().__init__()
        self.curriculum_step = 0

    def loss(self, t, x_true=None):
        physics_loss = super().loss(t, x_true)
        
        # Gradually increase the weight of the physics loss
        physics_weight = min(1.0, self.curriculum_step / 500)
        data_weight = 1 - physics_weight
        
        total_loss = physics_weight * physics_loss + data_weight * torch.mean((self(t) - x_true)**2)
        
        self.curriculum_step += 1
        return total_loss

model = CurriculumPINN()
# Training would proceed as before, but with the curriculum-based loss
```

Slide 14: Future Directions and Research Opportunities

The field of Physics-Informed Neural Networks is rapidly evolving. Some promising research directions include:

1. Combining PINNs with other machine learning techniques like reinforcement learning
2. Applying PINNs to inverse problems in physics
3. Developing PINNs for multi-scale and multi-physics problems
4. Exploring the theoretical foundations of PINNs

Here's a conceptual example of using PINNs for an inverse problem:

```python
class InversePINN(DampedOscillatorPINN):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.1]))
        self.omega = nn.Parameter(torch.tensor([1.0]))

    def loss(self, t, x_true):
        x_pred = self(t)
        mse_loss = torch.mean((x_pred - x_true)**2)
        
        # The physics parameters (beta and omega) are learned from data
        return mse_loss

inverse_model = InversePINN()
# Training would focus on learning the physical parameters from data
```

Slide 15: Additional Resources

For those interested in delving deeper into Physics-Informed Neural Networks, here are some valuable resources:

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707. ArXiv: [https://arxiv.org/abs/1711.10561](https://arxiv.org/abs/1711.10561)
2. Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. SIAM Review, 63(1), 208-228. ArXiv: [https://arxiv.org/abs/1907.04502](https://arxiv.org/abs/1907.04502)
3. Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. Nature Reviews Physics, 3(6), 422-440. ArXiv: [https://arxiv.org/abs/2003.04919](https://arxiv.org/abs/2003.04919)

These papers provide a comprehensive overview of PINNs, their applications, and current research directions in the field.

