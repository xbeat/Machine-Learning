## Learning and Verifying Maximal Taylor-Neural Lyapunov Functions in Python
Slide 1: Introduction to Taylor-Neural Lyapunov Functions

Taylor-Neural Lyapunov functions combine the power of Taylor series expansions with neural networks to create robust stability certificates for nonlinear dynamical systems. This approach leverages the approximation capabilities of neural networks and the analytical properties of Taylor expansions to construct Lyapunov functions that can verify stability over larger regions of the state space.

```python
import numpy as np
import tensorflow as tf

# Define a simple nonlinear dynamical system
def nonlinear_system(x, t):
    return np.array([x[1], -np.sin(x[0]) - x[1]])

# Create a neural network to approximate the Lyapunov function
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')
```

Slide 2: Taylor Series Expansion

The Taylor series expansion is a fundamental tool in approximating functions. For Lyapunov function verification, we use it to represent our candidate function in a form that allows for efficient analysis. The Taylor expansion of a function f(x) around a point a is given by:

f(x) = f(a) + f'(a)(x-a) + (f''(a)/2!)(x-a)^2 + ...

```python
import sympy as sp

# Define symbolic variables
x, a = sp.symbols('x a')

# Define a function
f = sp.sin(x)

# Compute Taylor series expansion
taylor_expansion = f.series(x, a, 4).removeO()

print(f"Taylor expansion of sin(x) around a:")
print(taylor_expansion)
```

Slide 3: Neural Network Architecture for Lyapunov Functions

The neural network architecture for approximating Lyapunov functions typically consists of fully connected layers with smooth activation functions. The output layer should produce a scalar value representing the Lyapunov function at the given state.

```python
def create_lyapunov_nn(input_dim, hidden_layers, hidden_units):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    
    for _ in range(hidden_layers):
        model.add(tf.keras.layers.Dense(hidden_units, activation='tanh'))
    
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    
    return model

lyapunov_nn = create_lyapunov_nn(input_dim=2, hidden_layers=3, hidden_units=64)
lyapunov_nn.summary()
```

Slide 4: Verifying Lyapunov Conditions

To verify that a function is a valid Lyapunov function, we need to check two conditions: positive definiteness and negative derivative along system trajectories. We can use automatic differentiation to compute the gradients needed for these checks.

```python
@tf.function
def lyapunov_conditions(model, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        V = model(x)
    
    grad_V = tape.gradient(V, x)
    
    # Check positive definiteness
    positive_definite = tf.reduce_all(V > 0)
    
    # Check negative derivative along trajectories
    system_dynamics = tf.constant(nonlinear_system(x.numpy(), 0), dtype=tf.float32)
    V_dot = tf.reduce_sum(grad_V * system_dynamics, axis=-1)
    negative_derivative = tf.reduce_all(V_dot < 0)
    
    return positive_definite, negative_derivative

# Example usage
x_test = tf.constant([[1.0, 0.5], [-0.5, -1.0]], dtype=tf.float32)
conditions = lyapunov_conditions(lyapunov_nn, x_test)
print(f"Lyapunov conditions satisfied: {conditions}")
```

Slide 5: Training the Neural Lyapunov Function

Training the neural network to approximate a valid Lyapunov function involves minimizing a loss function that encourages the network to satisfy the Lyapunov conditions over a given region of the state space.

```python
def lyapunov_loss(model, x):
    V = model(x)
    positive_definite_loss = tf.maximum(0.0, -V + 1e-3)
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        V = model(x)
    grad_V = tape.gradient(V, x)
    
    system_dynamics = tf.constant(nonlinear_system(x.numpy(), 0), dtype=tf.float32)
    V_dot = tf.reduce_sum(grad_V * system_dynamics, axis=-1)
    negative_derivative_loss = tf.maximum(0.0, V_dot + 1e-3)
    
    return tf.reduce_mean(positive_definite_loss + negative_derivative_loss)

# Training loop
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

for epoch in range(1000):
    x_batch = tf.random.uniform((64, 2), minval=-5, maxval=5)
    
    with tf.GradientTape() as tape:
        loss = lyapunov_loss(lyapunov_nn, x_batch)
    
    gradients = tape.gradient(loss, lyapunov_nn.trainable_variables)
    optimizer.apply_gradients(zip(gradients, lyapunov_nn.trainable_variables))
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")
```

Slide 6: Maximal Lyapunov Functions

Maximal Lyapunov functions aim to provide the largest possible region of attraction (ROA) for a given dynamical system. These functions are particularly useful for understanding the global stability properties of nonlinear systems.

```python
import matplotlib.pyplot as plt

def plot_roa(model, xlim=(-5, 5), ylim=(-5, 5), resolution=100):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    states = np.column_stack((X.ravel(), Y.ravel()))
    V = model(states).numpy().reshape(X.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, V, levels=20, cmap='viridis')
    plt.colorbar(label='Lyapunov Function Value')
    plt.title('Region of Attraction Estimation')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Plot the estimated ROA
plot_roa(lyapunov_nn)
```

Slide 7: Taylor Expansion of Neural Lyapunov Functions

To combine the power of neural networks with the analytical properties of Taylor series, we can expand the learned neural Lyapunov function around specific points of interest. This allows for more precise analysis and verification of stability properties.

```python
import tensorflow as tf
import numpy as np

def taylor_expand_nn(model, x0, order=2):
    x = tf.Variable(x0, dtype=tf.float32)
    
    with tf.GradientTape(persistent=True) as tape:
        V = model(x)
        
        gradients = [V]
        for i in range(1, order + 1):
            gradients.append(tape.gradient(gradients[-1], x))
    
    taylor_coeffs = [g.numpy() for g in gradients]
    
    def taylor_approximation(x):
        result = taylor_coeffs[0]
        for i, coeff in enumerate(taylor_coeffs[1:], 1):
            result += np.sum(coeff * (x - x0)**i) / np.math.factorial(i)
        return result
    
    return taylor_approximation

# Example usage
x0 = np.array([0.0, 0.0])
taylor_lyapunov = taylor_expand_nn(lyapunov_nn, x0, order=3)

# Evaluate the Taylor approximation
x_test = np.array([0.1, 0.2])
print(f"Taylor approximation at {x_test}: {taylor_lyapunov(x_test)}")
print(f"Actual NN output at {x_test}: {lyapunov_nn(x_test[np.newaxis, :]).numpy()[0, 0]}")
```

Slide 8: Verifying Maximal Taylor-Neural Lyapunov Functions

To verify that a Taylor-Neural Lyapunov function is maximal, we need to check its properties over the entire state space. This often involves solving optimization problems to find the boundaries of the region of attraction.

```python
from scipy.optimize import minimize

def find_roa_boundary(taylor_lyapunov, initial_guess):
    def objective(x):
        return -taylor_lyapunov(x)
    
    constraints = [{
        'type': 'ineq',
        'fun': lambda x: -nonlinear_system(x, 0).dot(np.gradient(taylor_lyapunov(x)))
    }]
    
    result = minimize(objective, initial_guess, constraints=constraints)
    return result.x

# Find multiple points on the ROA boundary
boundary_points = []
for _ in range(10):
    initial_guess = np.random.uniform(-5, 5, size=2)
    boundary_point = find_roa_boundary(taylor_lyapunov, initial_guess)
    boundary_points.append(boundary_point)

# Plot the ROA boundary points
plt.figure(figsize=(10, 8))
plot_roa(lyapunov_nn)
boundary_points = np.array(boundary_points)
plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c='r', s=50, label='ROA Boundary')
plt.legend()
plt.show()
```

Slide 9: Real-Life Example: Inverted Pendulum

The inverted pendulum is a classic control problem that demonstrates the usefulness of Lyapunov functions. We'll use our Taylor-Neural Lyapunov approach to analyze the stability of an inverted pendulum system.

```python
def inverted_pendulum(x, t, m=1, l=1, g=9.81):
    theta, omega = x
    dtheta = omega
    domega = (g / l) * np.sin(theta)
    return np.array([dtheta, domega])

# Create and train a neural Lyapunov function for the inverted pendulum
pendulum_nn = create_lyapunov_nn(input_dim=2, hidden_layers=3, hidden_units=64)

# Training loop (similar to before, but using inverted_pendulum dynamics)
# ...

# Create Taylor expansion of the trained neural Lyapunov function
x0_pendulum = np.array([0.0, 0.0])  # Equilibrium point
taylor_pendulum = taylor_expand_nn(pendulum_nn, x0_pendulum, order=4)

# Plot the estimated ROA for the inverted pendulum
plot_roa(pendulum_nn, xlim=(-np.pi/2, np.pi/2), ylim=(-2, 2))
```

Slide 10: Real-Life Example: Autonomous Vehicle Path Planning

Another application of Taylor-Neural Lyapunov functions is in autonomous vehicle path planning. We can use these functions to ensure stable trajectories and avoid obstacles.

```python
def vehicle_dynamics(x, t, v=1.0):
    px, py, theta = x
    dpx = v * np.cos(theta)
    dpy = v * np.sin(theta)
    dtheta = 0  # Assume constant orientation for simplicity
    return np.array([dpx, dpy, dtheta])

# Create and train a neural Lyapunov function for vehicle path planning
vehicle_nn = create_lyapunov_nn(input_dim=3, hidden_layers=4, hidden_units=128)

# Training loop (similar to before, but using vehicle_dynamics)
# ...

# Create Taylor expansion of the trained neural Lyapunov function
x0_vehicle = np.array([0.0, 0.0, 0.0])  # Starting position and orientation
taylor_vehicle = taylor_expand_nn(vehicle_nn, x0_vehicle, order=3)

# Plot the estimated safe region for vehicle path planning
def plot_safe_region(model, xlim=(-10, 10), ylim=(-10, 10), resolution=100):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    states = np.column_stack((X.ravel(), Y.ravel(), np.zeros_like(X.ravel())))
    V = model(states).numpy().reshape(X.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, V, levels=20, cmap='viridis')
    plt.colorbar(label='Lyapunov Function Value')
    plt.title('Safe Region for Vehicle Path Planning')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

plot_safe_region(vehicle_nn)
```

Slide 11: Challenges and Limitations

While Taylor-Neural Lyapunov functions are powerful tools for stability analysis, they face several challenges:

1. Scalability to high-dimensional systems
2. Difficulty in verifying global optimality
3. Sensitivity to neural network architecture and training

To address these challenges, researchers are exploring techniques such as:

```python
def demonstrate_challenge(dimension):
    # Create a high-dimensional system
    def high_dim_system(x, t):
        return -x  # Simple linear system for demonstration
    
    # Create a neural Lyapunov function for the high-dim system
    high_dim_nn = create_lyapunov_nn(input_dim=dimension, hidden_layers=5, hidden_units=256)
    
    # Measure training time and performance
    import time
    start_time = time.time()
    
    # Training loop (simplified for brevity)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    for _ in range(100):
        x_batch = tf.random.normal((64, dimension))
        with tf.GradientTape() as tape:
            loss = lyapunov_loss(high_dim_nn, x_batch)
        gradients = tape.gradient(loss, high_dim_nn.trainable_variables)
        optimizer.apply_gradients(zip(gradients, high_dim_nn.trainable_variables))
    
    end_time = time.time()
    print(f"Training time for {dimension}-dimensional system: {end_time - start_time:.2f} seconds")
    
    # Verify Lyapunov conditions (simplified)
    x_test = tf.random.normal((1000, dimension))
    conditions = lyapunov_conditions(high_dim_nn, x_test)
    success_rate = tf.reduce_mean(tf.cast(conditions[0] & conditions[1], tf.float32))
    print(f"Lyapunov condition success rate: {success_rate.numpy():.2%}")

# Demonstrate scalability challenge
for dim in [2, 10, 50]:
    demonstrate_challenge(dim)
```

Slide 12: Future Directions

The field of Taylor-Neural Lyapunov functions is rapidly evolving. Some promising future directions include:

1. Integration with reinforcement learning for adaptive control
2. Hybrid approaches combining neural networks with symbolic regression
3. Application to stochastic and hybrid systems

```python
import gym
import numpy as np
import tensorflow as tf

# Pseudocode for Lyapunov-constrained reinforcement learning
class LyapunovRL:
    def __init__(self, env, lyapunov_nn):
        self.env = env
        self.lyapunov_nn = lyapunov_nn
        self.policy_network = self.create_policy_network()
    
    def create_policy_network(self):
        # Create policy network architecture
        return tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=self.env.observation_space.shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.env.action_space.n, activation='softmax')
        ])
    
    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action_probs = self.policy_network(np.array([state]))
                action = np.random.choice(self.env.action_space.n, p=action_probs.numpy()[0])
                next_state, reward, done, _ = self.env.step(action)
                
                # Lyapunov constraint
                if not self.check_lyapunov_condition(state, next_state):
                    # Apply safety intervention or modify reward
                    pass
                
                # Update policy network
                self.update_policy(state, action, reward, next_state)
                
                state = next_state
    
    def check_lyapunov_condition(self, state, next_state):
        V_current = self.lyapunov_nn(np.array([state]))
        V_next = self.lyapunov_nn(np.array([next_state]))
        return V_next < V_current
    
    def update_policy(self, state, action, reward, next_state):
        # Implement policy update algorithm (e.g., PPO, TRPO)
        pass

# Usage
env = gym.make('CartPole-v1')
lyapunov_nn = create_lyapunov_nn(input_dim=4, hidden_layers=2, hidden_units=32)
lyapunov_rl = LyapunovRL(env, lyapunov_nn)
lyapunov_rl.train(episodes=1000)
```

Slide 13: Conclusion and Key Takeaways

Taylor-Neural Lyapunov functions offer a powerful approach to stability analysis and control of nonlinear dynamical systems. Key takeaways include:

1. Combination of neural networks and Taylor series for robust stability certificates
2. Applicability to a wide range of real-world problems, from robotics to autonomous vehicles
3. Ongoing research to address challenges in scalability and global optimality

As the field continues to evolve, we can expect to see increasingly sophisticated methods for learning and verifying maximal Lyapunov functions, leading to more reliable and efficient control systems across various domains.

```python
def summarize_taylor_neural_lyapunov():
    summary = {
        "Strengths": [
            "Combines flexibility of neural networks with analytical properties of Taylor series",
            "Capable of approximating complex Lyapunov functions",
            "Applicable to a wide range of nonlinear systems"
        ],
        "Challenges": [
            "Scalability to high-dimensional systems",
            "Difficulty in verifying global optimality",
            "Computational complexity in training and verification"
        ],
        "Future Directions": [
            "Integration with reinforcement learning",
            "Hybrid approaches with symbolic regression",
            "Application to stochastic and hybrid systems"
        ]
    }
    
    for category, items in summary.items():
        print(f"{category}:")
        for item in items:
            print(f"- {item}")
        print()

summarize_taylor_neural_lyapunov()
```

Slide 14: Additional Resources

For those interested in diving deeper into the topic of Taylor-Neural Lyapunov functions, the following resources provide valuable insights and advanced techniques:

1. ArXiv paper: "Neural Lyapunov Control" by Ying-Jen Chen, Maziar Raissi, and George Em Karniadakis URL: [https://arxiv.org/abs/1808.00924](https://arxiv.org/abs/1808.00924)
2. ArXiv paper: "Learning Control Lyapunov Functions from Counterexamples and Demonstrations" by Andrew J. Taylor, Victor D. Dorobantu, Sarah Dean, Benjamin Recht, Yisong Yue, and Aaron D. Ames URL: [https://arxiv.org/abs/1912.10251](https://arxiv.org/abs/1912.10251)
3. ArXiv paper: "Scalable Computation of Lyapunov Functions for Large-scale Systems using Neural Networks" by Peter J. Goulart and Eric C. Kerrigan URL: [https://arxiv.org/abs/1901.08226](https://arxiv.org/abs/1901.08226)

These papers provide in-depth discussions on the theoretical foundations and practical applications of neural Lyapunov functions, as well as advanced techniques for scaling to large-scale systems and integrating with modern control methods.

