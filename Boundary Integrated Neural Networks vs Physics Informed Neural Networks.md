## Boundary Integrated Neural Networks vs Physics Informed Neural Networks
Slide 1: Introduction to BINNs and PINNs

Boundary Integrated Neural Networks (BINNs) and Physics Informed Neural Networks (PINNs) are advanced machine learning techniques used to solve differential equations and model complex physical systems. These approaches combine the power of neural networks with domain-specific knowledge to improve accuracy and efficiency in scientific computing.

Slide 2: Introduction to BINNs and PINNs Code

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class BINN(nn.Module):
    def __init__(self):
        super(BINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)

# Visualize the architecture
x = torch.linspace(0, 1, 100).unsqueeze(1)
binn = BINN()
pinn = PINN()

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title("BINN Output")
plt.plot(x.numpy(), binn(x).detach().numpy())
plt.subplot(122)
plt.title("PINN Output")
plt.plot(x.numpy(), pinn(x).detach().numpy())
plt.tight_layout()
plt.show()
```

Slide 3: Key Differences Between BINNs and PINNs

BINNs focus on integrating boundary conditions directly into the neural network architecture, while PINNs incorporate physical laws and constraints into the loss function. BINNs are particularly effective for problems with complex boundary conditions, whereas PINNs excel in scenarios where the underlying physical principles are well-understood.

Slide 4: Key Differences Between BINNs and PINNs Code

```python
import torch
import torch.nn as nn

class BINN(nn.Module):
    def __init__(self):
        super(BINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
        self.boundary_layer = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.net(x) + self.boundary_layer(x)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)

    def physics_loss(self, x, y):
        # Example: Enforce conservation of energy
        dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                    create_graph=True)[0]
        return torch.mean((dy_dx ** 2 - y ** 2) ** 2)

# Demonstrate the difference in forward pass
x = torch.linspace(0, 1, 100).unsqueeze(1)
binn = BINN()
pinn = PINN()

y_binn = binn(x)
y_pinn = pinn(x)
physics_loss = pinn.physics_loss(x, y_pinn)

print(f"BINN output shape: {y_binn.shape}")
print(f"PINN output shape: {y_pinn.shape}")
print(f"PINN physics loss: {physics_loss.item()}")
```

Slide 5: BINN Architecture and Boundary Condition Integration

BINNs incorporate boundary conditions directly into the network architecture. This is typically achieved by adding a separate layer or component that explicitly satisfies the boundary conditions. The main network then learns the interior solution, while the boundary component ensures that the overall solution meets the specified conditions at the domain boundaries.

Slide 6: BINN Architecture and Boundary Condition Integration Code

```python
import torch
import torch.nn as nn

class BINN(nn.Module):
    def __init__(self):
        super(BINN, self).__init__()
        self.interior_net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
        self.boundary_net = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        interior_solution = self.interior_net(x)
        boundary_solution = self.boundary_net(x)
        return interior_solution + x * (1 - x) * boundary_solution

# Demonstrate how BINN satisfies boundary conditions
binn = BINN()
x = torch.tensor([[0.0], [1.0]])  # Domain boundaries
y = binn(x)

print("BINN output at boundaries:")
print(f"x = 0: {y[0].item():.6f}")
print(f"x = 1: {y[1].item():.6f}")
```

Slide 7: PINN Architecture and Physics Integration

PINNs incorporate physical laws and constraints into the loss function rather than the network architecture. The network learns to satisfy these constraints during training, effectively embedding the physics into the learned solution. This approach allows PINNs to generalize well to unseen scenarios that follow the same physical principles.

Slide 8: PINN Architecture and Physics Integration Code

```python
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)

    def physics_loss(self, x, y):
        # Example: Enforcing the heat equation
        dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                    create_graph=True)[0]
        d2y_dx2 = torch.autograd.grad(dy_dx, x, grad_outputs=torch.ones_like(dy_dx),
                                      create_graph=True)[0]
        return torch.mean((d2y_dx2 - y) ** 2)

# Demonstrate PINN training with physics loss
pinn = PINN()
optimizer = torch.optim.Adam(pinn.parameters(), lr=0.01)
x = torch.linspace(0, 1, 100, requires_grad=True).unsqueeze(1)

for _ in range(100):
    optimizer.zero_grad()
    y = pinn(x)
    loss = pinn.physics_loss(x, y)
    loss.backward()
    optimizer.step()

print(f"Final physics loss: {loss.item():.6f}")
```

Slide 9: Training Process for BINNs

The training process for BINNs focuses on minimizing the loss function while ensuring that the boundary conditions are satisfied. This is typically achieved by using a combination of data-driven loss and a penalty term for boundary condition violations. The network architecture itself helps in satisfying the boundary conditions, reducing the need for explicit constraints in the loss function.

Slide 10: Training Process for BINNs Code

```python
import torch
import torch.nn as nn
import torch.optim as optim

class BINN(nn.Module):
    def __init__(self):
        super(BINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
        self.boundary_layer = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.net(x) + x * (1 - x) * self.boundary_layer(x)

# Training loop
binn = BINN()
optimizer = optim.Adam(binn.parameters(), lr=0.01)
mse_loss = nn.MSELoss()

x_train = torch.linspace(0, 1, 100).unsqueeze(1)
y_train = torch.sin(np.pi * x_train)  # Example target function

for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = binn(x_train)
    loss = mse_loss(y_pred, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}")

# Verify boundary conditions
x_boundary = torch.tensor([[0.0], [1.0]])
y_boundary = binn(x_boundary)
print("\nBoundary values:")
print(f"y(0) = {y_boundary[0].item():.6f}")
print(f"y(1) = {y_boundary[1].item():.6f}")
```

Slide 11: Training Process for PINNs

PINNs are trained by minimizing a composite loss function that includes both data-driven loss and physics-informed loss. The physics-informed component enforces the governing equations and boundary conditions, allowing the network to learn solutions that are consistent with the underlying physical principles. This approach often leads to better generalization and more physically meaningful results.

Slide 12: Training Process for PINNs Code

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)

    def physics_loss(self, x, y):
        dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                    create_graph=True)[0]
        d2y_dx2 = torch.autograd.grad(dy_dx, x, grad_outputs=torch.ones_like(dy_dx),
                                      create_graph=True)[0]
        return torch.mean((d2y_dx2 + y) ** 2)  # Example: wave equation

# Training loop
pinn = PINN()
optimizer = optim.Adam(pinn.parameters(), lr=0.01)
mse_loss = nn.MSELoss()

x_train = torch.linspace(0, 1, 100, requires_grad=True).unsqueeze(1)
y_train = torch.sin(np.pi * x_train)  # Example target function

for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = pinn(x_train)
    data_loss = mse_loss(y_pred, y_train)
    physics_loss = pinn.physics_loss(x_train, y_pred)
    total_loss = data_loss + 0.1 * physics_loss
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/1000], Total Loss: {total_loss.item():.4f}, "
              f"Data Loss: {data_loss.item():.4f}, Physics Loss: {physics_loss.item():.4f}")
```

Slide 13: Advantages of BINNs

BINNs excel in problems with complex boundary conditions or geometries. By integrating boundary conditions directly into the network architecture, they ensure that the solution always satisfies these conditions. This can lead to more stable and accurate solutions, especially in scenarios where traditional numerical methods struggle.

Slide 14: Advantages of BINNs Code

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class ComplexBoundaryBINN(nn.Module):
    def __init__(self):
        super(ComplexBoundaryBINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
        self.boundary_net = nn.Linear(2, 1, bias=False)

    def forward(self, x, y):
        interior = self.net(torch.cat([x, y], dim=1))
        boundary = self.boundary_net(torch.cat([x, y], dim=1))
        return interior + (x**2 + y**2 - 1) * boundary

# Demonstrate BINN on a circular domain
binn = ComplexBoundaryBINN()
x = torch.linspace(-1, 1, 100)
y = torch.linspace(-1, 1, 100)
X, Y = torch.meshgrid(x, y)
XY = torch.stack([X.flatten(), Y.flatten()], dim=1)

with torch.no_grad():
    Z = binn(XY[:, 0].unsqueeze(1), XY[:, 1].unsqueeze(1)).reshape(100, 100)

plt.figure(figsize=(10, 8))
plt.contourf(X.numpy(), Y.numpy(), Z.numpy(), levels=20)
plt.colorbar(label='BINN Output')
plt.title('BINN Solution on a Circular Domain')
circle = plt.Circle((0, 0), 1, fill=False, color='r')
plt.gca().add_artist(circle)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

Slide 15: Advantages of PINNs

PINNs excel in problems where the underlying physical laws are well-understood but traditional numerical methods are computationally expensive. By incorporating these laws into the learning process, PINNs can often achieve accurate results with fewer data points and generalize better to unseen scenarios. This makes them particularly useful for complex physical systems and partial differential equations.

Slide 16: Advantages of PINNs Code

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class WaveEquationPINN(nn.Module):
    def __init__(self):
        super(WaveEquationPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

    def physics_loss(self, x, t):
        y = self.forward(x, t)
        dy_dt = torch.autograd.grad(y, t, create_graph=True)[0]
        d2y_dt2 = torch.autograd.grad(dy_dt, t, create_graph=True)[0]
        dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
        d2y_dx2 = torch.autograd.grad(dy_dx, x, create_graph=True)[0]
        return torch.mean((d2y_dt2 - d2y_dx2)**2)

# Example usage
pinn = WaveEquationPINN()
x = torch.linspace(0, 1, 50).unsqueeze(1).requires_grad_(True)
t = torch.linspace(0, 1, 50).unsqueeze(1).requires_grad_(True)
loss = pinn.physics_loss(x, t)
print(f"Physics loss: {loss.item():.6f}")
```

Slide 17: Real-life Example: Heat Transfer Simulation

BINNs and PINNs can be applied to simulate heat transfer in materials, a common problem in engineering and materials science. This example demonstrates how these neural network approaches can model temperature distribution in a 2D plate with specific boundary conditions.

Slide 18: Real-life Example: Heat Transfer Simulation Code

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class HeatTransferNN(nn.Module):
    def __init__(self, is_binn=True):
        super(HeatTransferNN, self).__init__()
        self.is_binn = is_binn
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
        if is_binn:
            self.boundary_net = nn.Linear(2, 1, bias=False)

    def forward(self, x, y):
        if self.is_binn:
            interior = self.net(torch.cat([x, y], dim=1))
            boundary = self.boundary_net(torch.cat([x, y], dim=1))
            return interior + x * (1-x) * y * (1-y) * boundary
        else:
            return self.net(torch.cat([x, y], dim=1))

# Create and visualize heat distribution
binn = HeatTransferNN(is_binn=True)
pinn = HeatTransferNN(is_binn=False)

x = torch.linspace(0, 1, 50)
y = torch.linspace(0, 1, 50)
X, Y = torch.meshgrid(x, y)

with torch.no_grad():
    Z_binn = binn(X.unsqueeze(-1), Y.unsqueeze(-1)).squeeze()
    Z_pinn = pinn(X.unsqueeze(-1), Y.unsqueeze(-1)).squeeze()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
im1 = ax1.contourf(X, Y, Z_binn, levels=20, cmap='hot')
ax1.set_title('BINN Heat Distribution')
fig.colorbar(im1, ax=ax1)
im2 = ax2.contourf(X, Y, Z_pinn, levels=20, cmap='hot')
ax2.set_title('PINN Heat Distribution')
fig.colorbar(im2, ax=ax2)
plt.tight_layout()
plt.show()
```

Slide 19: Real-life Example: Fluid Dynamics Simulation

Another application of BINNs and PINNs is in fluid dynamics, where they can be used to simulate complex fluid flows. This example demonstrates how these neural networks can be applied to solve the Navier-Stokes equations for incompressible fluid flow.

Slide 20: Real-life Example: Fluid Dynamics Simulation Code

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class FluidFlowNN(nn.Module):
    def __init__(self, is_binn=True):
        super(FluidFlowNN, self).__init__()
        self.is_binn = is_binn
        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 4)
        )
        if is_binn:
            self.boundary_net = nn.Linear(3, 4, bias=False)

    def forward(self, x, y, t):
        if self.is_binn:
            interior = self.net(torch.cat([x, y, t], dim=1))
            boundary = self.boundary_net(torch.cat([x, y, t], dim=1))
            return interior + x * (1-x) * y * (1-y) * boundary
        else:
            return self.net(torch.cat([x, y, t], dim=1))

# Create and visualize fluid flow
binn = FluidFlowNN(is_binn=True)
pinn = FluidFlowNN(is_binn=False)

x = torch.linspace(0, 1, 20)
y = torch.linspace(0, 1, 20)
t = torch.tensor([0.5])
X, Y = torch.meshgrid(x, y)

with torch.no_grad():
    flow_binn = binn(X.unsqueeze(-1), Y.unsqueeze(-1), t.expand_as(X.unsqueeze(-1)))
    flow_pinn = pinn(X.unsqueeze(-1), Y.unsqueeze(-1), t.expand_as(X.unsqueeze(-1)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.quiver(X, Y, flow_binn[:,:,0], flow_binn[:,:,1])
ax1.set_title('BINN Fluid Flow')
ax2.quiver(X, Y, flow_pinn[:,:,0], flow_pinn[:,:,1])
ax2.set_title('PINN Fluid Flow')
plt.tight_layout()
plt.show()
```

Slide 21: Challenges and Limitations of BINNs

While BINNs are powerful for problems with complex boundary conditions, they may struggle with very high-dimensional problems or when the boundary conditions are difficult to express mathematically. Additionally, the explicit boundary integration can sometimes lead to overfitting near the boundaries.

Slide 22: Challenges and Limitations of BINNs

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class HighDimensionalBINN(nn.Module):
    def __init__(self, input_dim):
        super(HighDimensionalBINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
        self.boundary_net = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        interior = self.net(x)
        boundary = self.boundary_net(x)
        return interior + torch.prod(x * (1-x), dim=1, keepdim=True) * boundary

# Demonstrate the challenge with high-dimensional input
input_dims = [2, 5, 10, 20]
x = torch.rand(1000, max(input_dims))

plt.figure(figsize=(12, 4))
for i, dim in enumerate(input_dims):
    binn = HighDimensionalBINN(dim)
    with torch.no_grad():
        y = binn(x[:, :dim])
    plt.subplot(1, 4, i+1)
    plt.hist(y.numpy(), bins=30)
    plt.title(f"{dim}D BINN Output")
plt.tight_layout()
plt.show()
```

Slide 23: Challenges and Limitations of PINNs

PINNs may face difficulties when dealing with very stiff or highly nonlinear differential equations. The physics-informed loss term can sometimes dominate the training process, leading to slow convergence or suboptimal solutions. Additionally, PINNs may struggle with discontinuities or shocks in the solution.

Slide 24: Challenges and Limitations of PINNs Code

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class NonlinearPINN(nn.Module):
    def __init__(self):
        super(NonlinearPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x)

    def physics_loss(self, x):
        y = self.forward(x)
        dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        d2y_dx2 = torch.autograd.grad(dy_dx, x, grad_outputs=torch.ones_like(dy_dx), create_graph=True)[0]
        return torch.mean((d2y_dx2 + torch.sin(y))**2)  # Nonlinear ODE: y'' + sin(y) = 0

# Demonstrate the challenge with a nonlinear ODE
pinn = NonlinearPINN()
optimizer = torch.optim.Adam(pinn.parameters(), lr=0.001)

x = torch.linspace(0, 10, 100, requires_grad=True).unsqueeze(1)
losses = []

for epoch in range(1000):
    optimizer.zero_grad()
    loss = pinn.physics_loss(x)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(losses)
plt.title('PINN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Physics Loss')
plt.subplot(122)
with torch.no_grad():
    y = pinn(x)
plt.plot(x.detach().numpy(), y.detach().numpy())
plt.title('PINN Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
```

Slide 25: Future Directions and Hybrid Approaches

Researchers are exploring hybrid approaches that combine the strengths of BINNs and PINNs. These methods aim to leverage the boundary condition integration of BINNs with the physics-informed learning of PINNs. Additionally, there's ongoing work on improving the training stability and convergence of both approaches for more complex systems.

Slide 26: Future Directions and Hybrid Approache Code

```python
import torch
import torch.nn as nn

class HybridNN(nn.Module):
    def __init__(self):
        super(HybridNN, self).__init__()
        self.interior_net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
        self.boundary_net = nn.Linear(2, 1, bias=False)

    def forward(self, x, y):
        interior = self.interior_net(torch.cat([x, y], dim=1))
        boundary = self.boundary_net(torch.cat([x, y], dim=1))
        return interior + x * (1-x) * y * (1-y) * boundary

    def physics_loss(self, x, y):
        u = self.forward(x, y)
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
        d2u_dy2 = torch.autograd.grad(du_dy, y, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0]
        return torch.mean((d2u_dx2 + d2u_dy2)**2)  # Laplace equation: ∇²u = 0

# Example usage of the hybrid approach
hybrid_nn = HybridNN()
x = torch.linspace(0, 1, 10).unsqueeze(1).requires_grad_(True)
y = torch.linspace(0, 1, 10).unsqueeze(1).requires_grad_(True)
u = hybrid_nn(x, y)
physics_loss = hybrid_nn.physics_loss(x, y)
print(f"Hybrid NN output shape: {u.shape}")
print(f"Physics loss: {physics_loss.item():.6f}")
```

Slide 27: Additional Resources

For those interested in diving deeper into BINNs and PINNs, here are some valuable resources:

1. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations" by M. Raissi, P. Perdikaris, and G.E. Karniadakis (2019). Available at: [https://arxiv.org/abs/1711.10561](https://arxiv.org/abs/1711.10561)
2. "Boundary-informed neural networks for PDEs" by Y. Wang, S. Cai, and Z. Xu (2021). Available at: [https://arxiv.org/abs/2102.06573](https://arxiv.org/abs/2102.06573)
3. "Physics-Informed Machine Learning" by G.E. Karniadakis, I.G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, and L. Yang (2021). Available at: [https://arxiv.org/abs/2107.10483](https://arxiv.org/abs/2107.10483)

These papers provide in-depth explanations of the theories behind BINNs and PINNs, as well as various applications and case studies.

