## Fluid Dynamics with Navier-Stokes Equations in Python

Slide 1: Introduction to Fluid Dynamics and Navier-Stokes Equations

Fluid dynamics is the study of fluid motion, including liquids and gases. The Navier-Stokes equations are fundamental to fluid dynamics, describing the motion of viscous fluid substances. These equations are based on the principles of conservation of mass, momentum, and energy.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_fluid_flow(grid_size, time_steps, viscosity, density):
    u = np.zeros((grid_size, grid_size))
    v = np.zeros((grid_size, grid_size))
    
    for _ in range(time_steps):
        # Simplified 2D Navier-Stokes solver
        laplacian_u = np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) + \
                      np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
        laplacian_v = np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) + \
                      np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4 * v
        
        u += viscosity * laplacian_u / density
        v += viscosity * laplacian_v / density
    
    return u, v

# Example usage
u, v = simulate_fluid_flow(50, 100, 0.1, 1.0)
plt.quiver(u, v)
plt.title("Simulated Fluid Flow")
plt.show()
```

Slide 2: Conservation of Mass - The Continuity Equation

The continuity equation represents the conservation of mass in fluid dynamics. It states that the rate of mass entering a system is equal to the rate of mass leaving the system plus the accumulation of mass within the system.

```python
import numpy as np

def continuity_equation(density, velocity, x, t):
    # Partial derivative of density with respect to time
    density_t = np.gradient(density, t, axis=1)
    
    # Divergence of density * velocity
    div_density_velocity = np.zeros_like(density)
    for i in range(3):  # Assuming 3D
        div_density_velocity += np.gradient(density * velocity[i], x[i], axis=i)
    
    # Check if continuity equation is satisfied
    residual = density_t + div_density_velocity
    return np.allclose(residual, 0, atol=1e-6)

# Example usage
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
z = np.linspace(0, 1, 10)
t = np.linspace(0, 1, 5)

X, Y, Z, T = np.meshgrid(x, y, z, t, indexing='ij')

density = np.exp(-(X**2 + Y**2 + Z**2))
velocity = [np.sin(X) * np.cos(T), np.sin(Y) * np.cos(T), np.sin(Z) * np.cos(T)]

is_satisfied = continuity_equation(density, velocity, [x, y, z], t)
print(f"Continuity equation is satisfied: {is_satisfied}")
```

Slide 3: Conservation of Momentum - The Momentum Equation

The momentum equation, derived from Newton's second law, describes the conservation of momentum in a fluid. It relates the forces acting on a fluid element to its acceleration and accounts for pressure gradients, viscous forces, and external forces.

```python
import numpy as np

def momentum_equation(velocity, pressure, viscosity, density, x, t):
    # Acceleration term (left-hand side of the equation)
    acceleration = np.gradient(velocity, t, axis=1)
    
    # Convective term
    convective = np.zeros_like(velocity)
    for i in range(3):  # Assuming 3D
        for j in range(3):
            convective[i] += velocity[j] * np.gradient(velocity[i], x[j], axis=j)
    
    # Pressure gradient term
    pressure_grad = np.gradient(pressure, x[0], axis=0), np.gradient(pressure, x[1], axis=1), np.gradient(pressure, x[2], axis=2)
    
    # Viscous term
    viscous = np.zeros_like(velocity)
    for i in range(3):
        for j in range(3):
            viscous[i] += np.gradient(np.gradient(velocity[i], x[j], axis=j), x[j], axis=j)
    
    # Check if momentum equation is satisfied
    residual = acceleration + convective + np.array(pressure_grad) / density - viscosity * viscous / density
    return np.allclose(residual, 0, atol=1e-6)

# Example usage
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
z = np.linspace(0, 1, 10)
t = np.linspace(0, 1, 5)

X, Y, Z, T = np.meshgrid(x, y, z, t, indexing='ij')

velocity = [np.sin(X) * np.cos(T), np.sin(Y) * np.cos(T), np.sin(Z) * np.cos(T)]
pressure = np.cos(X + Y + Z) * np.sin(T)
viscosity = 0.1
density = 1.0

is_satisfied = momentum_equation(velocity, pressure, viscosity, density, [x, y, z], t)
print(f"Momentum equation is satisfied: {is_satisfied}")
```

Slide 4: Incompressible Navier-Stokes Equations

For incompressible fluids, where density remains constant, the Navier-Stokes equations simplify. This form is widely used in many practical applications, such as modeling airflow around vehicles or water flow in pipes.

```python
import numpy as np

def incompressible_navier_stokes(u, v, dx, dy, dt, viscosity):
    # Compute spatial derivatives
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    
    # Compute Laplacians
    laplacian_u = np.gradient(du_dx, dx, axis=1) + np.gradient(du_dy, dy, axis=0)
    laplacian_v = np.gradient(dv_dx, dx, axis=1) + np.gradient(dv_dy, dy, axis=0)
    
    # Update velocities
    u_new = u - dt * (u * du_dx + v * du_dy) + dt * viscosity * laplacian_u
    v_new = v - dt * (u * dv_dx + v * dv_dy) + dt * viscosity * laplacian_v
    
    return u_new, v_new

# Example usage
nx, ny = 50, 50
dx, dy = 1.0 / nx, 1.0 / ny
x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))

u = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
v = -np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)

dt = 0.001
viscosity = 0.1

for _ in range(100):
    u, v = incompressible_navier_stokes(u, v, dx, dy, dt, viscosity)

import matplotlib.pyplot as plt
plt.streamplot(x, y, u, v)
plt.title("Incompressible Flow Field")
plt.show()
```

Slide 5: Boundary Conditions in Fluid Dynamics

Boundary conditions are crucial in fluid dynamics simulations. They define how the fluid interacts with its surroundings, such as solid walls, inlets, or outlets. Common types include no-slip, free-slip, and periodic boundary conditions.

```python
import numpy as np
import matplotlib.pyplot as plt

def apply_boundary_conditions(u, v, boundary_type):
    if boundary_type == "no_slip":
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0
        v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 0
    elif boundary_type == "free_slip":
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        v[:, 0] = v[:, 1]
        v[:, -1] = v[:, -2]
    elif boundary_type == "periodic":
        u[0, :] = u[-2, :]
        u[-1, :] = u[1, :]
        v[:, 0] = v[:, -2]
        v[:, -1] = v[:, 1]
    return u, v

# Example usage
nx, ny = 50, 50
u = np.random.rand(nx, ny)
v = np.random.rand(nx, ny)

boundary_types = ["no_slip", "free_slip", "periodic"]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, boundary_type in enumerate(boundary_types):
    u_bc, v_bc = apply_boundary_conditions(u.(), v.(), boundary_type)
    axs[i].streamplot(np.arange(nx), np.arange(ny), u_bc, v_bc)
    axs[i].set_title(f"{boundary_type.capitalize()} Boundary")
    axs[i].set_aspect('equal')

plt.tight_layout()
plt.show()
```

Slide 6: Vorticity and Circulation

Vorticity is a measure of local rotation in a fluid flow, while circulation quantifies the overall rotation around a closed path. These concepts are essential for understanding complex fluid behaviors like turbulence and vortex formation.

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_vorticity(u, v, dx, dy):
    dudy = np.gradient(u, dy, axis=0)
    dvdx = np.gradient(v, dx, axis=1)
    return dvdx - dudy

def compute_circulation(u, v, dx, dy):
    vorticity = compute_vorticity(u, v, dx, dy)
    return np.sum(vorticity) * dx * dy

# Example usage
nx, ny = 100, 100
x = np.linspace(-5, 5, nx)
y = np.linspace(-5, 5, ny)
X, Y = np.meshgrid(x, y)

# Create a vortex-like flow field
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)
u = -np.sin(theta) * (1 - np.exp(-r/2))
v = np.cos(theta) * (1 - np.exp(-r/2))

vorticity = compute_vorticity(u, v, x[1]-x[0], y[1]-y[0])
circulation = compute_circulation(u, v, x[1]-x[0], y[1]-y[0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.streamplot(X, Y, u, v, density=1, color='k', linewidth=1)
ax1.set_title("Velocity Field")
im = ax2.imshow(vorticity, extent=[x.min(), x.max(), y.min(), y.max()], cmap='RdBu_r')
ax2.set_title("Vorticity")
plt.colorbar(im, ax=ax2)
plt.tight_layout()
plt.show()

print(f"Total circulation: {circulation:.4f}")
```

Slide 7: Reynolds Number and Flow Regimes

The Reynolds number is a dimensionless quantity that helps predict flow patterns in different fluid flow situations. It is the ratio of inertial forces to viscous forces within a fluid, which is used to determine whether the flow will be laminar or turbulent.

```python
import numpy as np
import matplotlib.pyplot as plt

def reynolds_number(velocity, characteristic_length, kinematic_viscosity):
    return velocity * characteristic_length / kinematic_viscosity

def plot_flow_regime(Re):
    plt.figure(figsize=(10, 5))
    plt.semilogx(Re, np.ones_like(Re), 'bo')
    plt.axvline(x=2300, color='r', linestyle='--', label='Transition')
    plt.text(1e2, 1.05, 'Laminar', horizontalalignment='center')
    plt.text(1e4, 1.05, 'Turbulent', horizontalalignment='center')
    plt.xlim(1e1, 1e5)
    plt.ylim(0.9, 1.1)
    plt.xlabel('Reynolds Number')
    plt.title('Flow Regime Based on Reynolds Number')
    plt.legend()
    plt.yticks([])
    plt.show()

# Example: Flow in a pipe
pipe_diameter = 0.05  # 5 cm
kinematic_viscosity = 1e-6  # water at 20°C
velocities = np.logspace(-1, 2, 20)  # Range of velocities

Re = reynolds_number(velocities, pipe_diameter, kinematic_viscosity)
plot_flow_regime(Re)

print("Reynolds numbers:", Re)
print("Flow regime:")
for r in Re:
    if r < 2300:
        print(f"Re = {r:.2f}: Laminar")
    elif 2300 <= r < 4000:
        print(f"Re = {r:.2f}: Transitional")
    else:
        print(f"Re = {r:.2f}: Turbulent")
```

Slide 8: Bernoulli's Principle

Bernoulli's principle relates pressure, velocity, and elevation in a moving fluid. It states that an increase in the speed of a fluid occurs simultaneously with a decrease in pressure or a decrease in the fluid's potential energy.

```python
import numpy as np
import matplotlib.pyplot as plt

def bernoulli_equation(v1, h1, p1, v2, h2, density, g=9.81):
    p2 = p1 + 0.5 * density * (v1**2 - v2**2) + density * g * (h1 - h2)
    return p2

def plot_pressure_velocity_relation(v, p):
    plt.figure(figsize=(10, 6))
    plt.plot(v, p, 'b-')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Pressure (Pa)')
    plt.title("Pressure vs Velocity (Bernoulli's Principle)")
    plt.grid(True)
    plt.show()

# Example: Flow in a constricted pipe
density = 1000  # kg/m^3 (water)
v1, h1, p1 = 2, 10, 101325  # Initial conditions
h2 = 10  # Same height

velocities = np.linspace(v1, 10, 100)
pressures = [bernoulli_equation(v1, h1, p1, v, h2, density) for v in velocities]

plot_pressure_velocity_relation(velocities, pressures)

# Calculate pressure at a specific point
v2 = 5  # m/s
p2 = bernoulli_equation(v1, h1, p1, v2, h2, density)
print(f"Pressure at v2 = {v2} m/s: {p2:.2f} Pa")
```

Slide 9: Computational Fluid Dynamics (CFD) Basics

Computational Fluid Dynamics (CFD) is a branch of fluid mechanics that uses numerical analysis and data structures to analyze and solve problems involving fluid flows. CFD is based on the Navier-Stokes equations and involves discretizing the fluid domain into small cells.

```python
import numpy as np
import matplotlib.pyplot as plt

def simple_2d_cfd(nx, ny, nt, nit, c, dx, dy, dt):
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))
    
    for n in range(nt):
        un = u.()
        vn = v.()
        
        for q in range(nit):
            pn = p.()
            b[1:-1,1:-1] = (1/dt)*((u[1:-1,2:]-u[1:-1,0:-2])/(2*dx)+(v[2:,1:-1]-v[0:-2,1:-1])/(2*dy))
            
            p[1:-1,1:-1] = ((pn[1:-1,2:]+pn[1:-1,0:-2])*dy**2+(pn[2:,1:-1]+pn[0:-2,1:-1])*dx**2)/(2*(dx**2+dy**2)) \
                           - dx**2*dy**2/(2*(dx**2+dy**2))*b[1:-1,1:-1]
            
            p[:,-1] = p[:,-2]  # dp/dx = 0 at x = 2
            p[0,:] = p[1,:]    # dp/dy = 0 at y = 0
            p[-1,:] = 0        # p = 0 at y = 2
            
        u[1:-1,1:-1] = un[1:-1,1:-1] - un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[1:-1,0:-2]) \
                       - vn[1:-1,1:-1]*dt/dy*(un[1:-1,1:-1]-un[0:-2,1:-1]) \
                       - dt/(2*c*dx)*(p[1:-1,2:]-p[1:-1,0:-2]) + c*(dt/dx**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2]) \
                       + dt/dy**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1]))
        
        v[1:-1,1:-1] = vn[1:-1,1:-1] - un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[1:-1,0:-2]) \
                       - vn[1:-1,1:-1]*dt/dy*(vn[1:-1,1:-1]-vn[0:-2,1:-1]) \
                       - dt/(2*c*dy)*(p[2:,1:-1]-p[0:-2,1:-1]) + c*(dt/dx**2*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,0:-2]) \
                       + dt/dy**2*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[0:-2,1:-1]))
        
        u[0,:] = 0
        u[:,0] = 0
        u[:,-1] = 0
        v[0,:] = 0
        v[-1,:] = 0
        v[:,0] = 0
        v[:,-1] = 0
        u[-1,:] = 1  # set velocity on cavity lid equal to 1
    
    return u, v, p

nx, ny = 41, 41
nt, nit = 500, 50
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = .001

u, v, p = simple_2d_cfd(nx, ny, nt, nit, c, dx, dy, dt)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.contourf(u, levels=np.linspace(-0.1, 1, 20), cmap='RdBu_r')
plt.colorbar()
plt.title('u-velocity')
plt.subplot(122)
plt.contourf(v, levels=np.linspace(-0.5, 0.5, 20), cmap='RdBu_r')
plt.colorbar()
plt.title('v-velocity')
plt.tight_layout()
plt.show()
```

Slide 10: Turbulence Modeling

Turbulence is a complex phenomenon characterized by chaotic changes in pressure and flow velocity. Modeling turbulence is crucial for many engineering applications. One common approach is the k-ε model, which introduces two additional transport equations.

```python
import numpy as np

def k_epsilon_model(u, v, k, epsilon, nu, Cmu, C1, C2, sigma_k, sigma_epsilon, dx, dy, dt):
    # Compute velocity gradients
    du_dx, du_dy = np.gradient(u, dx, dy)
    dv_dx, dv_dy = np.gradient(v, dx, dy)
    
    # Compute turbulent viscosity
    nu_t = Cmu * k**2 / epsilon
    
    # Compute production term
    P = nu_t * (2*du_dx**2 + 2*dv_dy**2 + (du_dy + dv_dx)**2)
    
    # Solve k equation
    d2k_dx2, d2k_dy2 = np.gradient(np.gradient(k, dx, axis=1), dx, axis=1), np.gradient(np.gradient(k, dy, axis=0), dy, axis=0)
    dk_dt = P - epsilon + np.gradient((nu + nu_t/sigma_k) * np.gradient(k, dx), dx) + \
            np.gradient((nu + nu_t/sigma_k) * np.gradient(k, dy), dy)
    k += dk_dt * dt
    
    # Solve epsilon equation
    d2e_dx2, d2e_dy2 = np.gradient(np.gradient(epsilon, dx, axis=1), dx, axis=1), np.gradient(np.gradient(epsilon, dy, axis=0), dy, axis=0)
    de_dt = C1 * epsilon/k * P - C2 * epsilon**2/k + \
            np.gradient((nu + nu_t/sigma_epsilon) * np.gradient(epsilon, dx), dx) + \
            np.gradient((nu + nu_t/sigma_epsilon) * np.gradient(epsilon, dy), dy)
    epsilon += de_dt * dt
    
    return k, epsilon, nu_t

# Example usage (simplified)
nx, ny = 50, 50
u = np.random.rand(nx, ny)
v = np.random.rand(nx, ny)
k = np.ones((nx, ny)) * 0.1
epsilon = np.ones((nx, ny)) * 0.01

nu = 1e-5
Cmu, C1, C2 = 0.09, 1.44, 1.92
sigma_k, sigma_epsilon = 1.0, 1.3
dx, dy, dt = 0.1, 0.1, 0.01

for _ in range(100):
    k, epsilon, nu_t = k_epsilon_model(u, v, k, epsilon, nu, Cmu, C1, C2, sigma_k, sigma_epsilon, dx, dy, dt)

print("Final average turbulent viscosity:", np.mean(nu_t))
```

Slide 11: Aerodynamics and Lift Generation

Aerodynamics is the study of how air interacts with solid objects, such as aircraft wings. Lift is generated due to the pressure difference between the upper and lower surfaces of a wing, which is related to the wing's shape and angle of attack.

```python
import numpy as np
import matplotlib.pyplot as plt

def joukowski_airfoil(zeta, a, alpha):
    z = zeta + a**2 / zeta
    z *= np.exp(-1j * alpha)
    return z

def plot_airfoil_flow(a, alpha, n_points=1000, n_streamlines=20):
    theta = np.linspace(0, 2*np.pi, n_points)
    zeta = a * np.exp(1j * theta)
    z = joukowski_airfoil(zeta, a, alpha)
    
    # Generate flow field
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-2, 2, 80)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    W = Z + a**2 / Z
    U = np.real(np.exp(-1j * alpha) * (1 - a**2 / Z**2))
    V = -np.imag(np.exp(-1j * alpha) * (1 - a**2 / Z**2))
    
    plt.figure(figsize=(12, 8))
    plt.streamplot(X, Y, U, V, density=1, color='lightblue')
    plt.plot(z.real, z.imag, 'k-', linewidth=2)
    plt.title(f"Airfoil at {alpha*180/np.pi:.1f}° Angle of Attack")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Example usage
a = 0.2
alpha = 5 * np.pi / 180  # 5 degrees angle of attack
plot_airfoil_flow(a, alpha)
```

Slide 12: Heat Transfer in Fluids

Heat transfer in fluids involves complex interactions between fluid motion and temperature gradients. The energy equation, coupled with the Navier-Stokes equations, describes how heat is transported in a fluid system.

```python
import numpy as np
import matplotlib.pyplot as plt

def heat_transfer_simulation(nx, ny, nt, alpha, dx, dy, dt):
    # Initialize temperature field
    T = np.zeros((ny, nx))
    T[0, :] = 100  # Hot bottom wall
    T[-1, :] = 0   # Cold top wall
    
    # Simulation loop
    for n in range(nt):
        Tn = T.()
        T[1:-1, 1:-1] = Tn[1:-1, 1:-1] + alpha * dt * (
            (Tn[1:-1, 2:] - 2*Tn[1:-1, 1:-1] + Tn[1:-1, :-2])/dx**2 +
            (Tn[2:, 1:-1] - 2*Tn[1:-1, 1:-1] + Tn[:-2, 1:-1])/dy**2
        )
        
        # Boundary conditions
        T[:, 0] = T[:, 1]    # Insulated left wall
        T[:, -1] = T[:, -2]  # Insulated right wall
    
    return T

# Simulation parameters
nx, ny = 50, 50
nt = 500
alpha = 0.01  # Thermal diffusivity
dx = dy = 0.01
dt = 0.01

# Run simulation
T = heat_transfer_simulation(nx, ny, nt, alpha, dx, dy, dt)

# Plot results
plt.figure(figsize=(10, 8))
plt.imshow(T, cmap='hot', interpolation='nearest')
plt.colorbar(label='Temperature')
plt.title('Heat Transfer in a 2D Fluid')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

Slide 13: Multiphase Flows

Multiphase flows involve the simultaneous flow of materials with different states or phases. Examples include gas-liquid flows in pipelines or solid particles suspended in a fluid. Modeling these flows requires considering interface dynamics and phase interactions.

```python
import numpy as np
import matplotlib.pyplot as plt

def two_phase_flow_simulation(nx, ny, nt):
    # Initialize phase field (phi)
    phi = np.ones((ny, nx))
    phi[ny//2:, :] = -1  # Bottom half is phase 2
    
    # Initialize velocity field
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    
    for _ in range(nt):
        # Compute interface curvature
        grad_phi_x, grad_phi_y = np.gradient(phi)
        grad_phi_norm = np.sqrt(grad_phi_x**2 + grad_phi_y**2)
        kappa = np.div(grad_phi_x/grad_phi_norm, grad_phi_y/grad_phi_norm)
        
        # Update phase field (simplified)
        phi += 0.1 * kappa
        
        # Update velocity field (simplified)
        u[1:-1, 1:-1] += 0.1 * (phi[1:-1, 2:] - phi[1:-1, :-2])
        v[1:-1, 1:-1] += 0.1 * (phi[2:, 1:-1] - phi[:-2, 1:-1])
        
        # Apply boundary conditions
        phi[:, 0] = phi[:, 1]
        phi[:, -1] = phi[:, -2]
        phi[0, :] = phi[1, :]
        phi[-1, :] = phi[-2, :]
    
    return phi, u, v

# Simulation parameters
nx, ny = 100, 100
nt = 100

# Run simulation
phi, u, v = two_phase_flow_simulation(nx, ny, nt)

# Visualize results
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(phi, cmap='bwr')
plt.title('Phase Field')
plt.subplot(132)
plt.imshow(u, cmap='RdBu')
plt.title('U Velocity')
plt.subplot(133)
plt.imshow(v, cmap='RdBu')
plt.title('V Velocity')
plt.tight_layout()
plt.show()
```

Slide 14: Fluid-Structure Interaction (FSI)

Fluid-Structure Interaction (FSI) involves the mutual interaction between a deformable structure and a surrounding or internal fluid flow. FSI is crucial in many engineering applications, such as the design of aircraft wings, blood flow in arteries, or the behavior of offshore structures.

```python
import numpy as np
import matplotlib.pyplot as plt

def simple_fsi_simulation(n_points, n_steps, fluid_density, structure_stiffness):
    # Initialize structure position and velocity
    y = np.zeros(n_points)
    v = np.zeros(n_points)
    
    # Initialize fluid velocity
    u_fluid = np.ones(n_points)
    
    time = np.linspace(0, 10, n_steps)
    y_history = []
    
    for _ in range(n_steps):
        # Compute fluid force on structure
        fluid_force = 0.5 * fluid_density * (u_fluid - v)**2
        
        # Update structure position and velocity
        a = (fluid_force - structure_stiffness * y) / fluid_density
        v += a * 0.1  # Time step of 0.1
        y += v * 0.1
        
        # Update fluid velocity (simplified)
        u_fluid = 1 + 0.1 * y
        
        y_history.append(y.())
    
    return np.array(y_history), time

# Simulation parameters
n_points = 50
n_steps = 200
fluid_density = 1.0
structure_stiffness = 10.0

# Run simulation
y_history, time = simple_fsi_simulation(n_points, n_steps, fluid_density, structure_stiffness)

# Visualize results
plt.figure(figsize=(10, 6))
plt.imshow(y_history.T, aspect='auto', cmap='viridis', extent=[0, time[-1], 0, n_points])
plt.colorbar(label='Displacement')
plt.title('Fluid-Structure Interaction')
plt.xlabel('Time')
plt.ylabel('Position along structure')
plt.show()
```

Slide 15: Additional Resources

For those interested in delving deeper into Fluid Dynamics and the Navier-Stokes equations, here are some valuable resources:

1. ArXiv.org papers:
   * "Numerical Methods for the Navier-Stokes Equations" (arXiv:1901.04943)
   * "Machine Learning for Fluid Dynamics: A Review" (arXiv:2002.00021)
2. Textbooks:
   * "Fluid Mechanics" by Frank M. White
   * "An Introduction to Computational Fluid Dynamics" by H. K. Versteeg and W. Malalasekera
3. Online courses:
   * MIT OpenCourseWare: "Computational Fluid Dynamics"
   * Coursera: "Fluid Mechanics: Advanced Topics and Applications"
4. Software and tools:
   * OpenFOAM: Open-source CFD software
   * ANSYS Fluent: Commercial CFD package
   * FEniCS: Open-source computing platform for solving PDEs

Remember to verify the accuracy and relevance of these resources, as the field of fluid dynamics is continuously evolving.

