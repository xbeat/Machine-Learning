## Physics Informed Neural Networks with Python
Slide 1: Introduction to Physics Informed Neural Networks (PINNs)

Physics Informed Neural Networks (PINNs) are a novel approach that combines the power of neural networks with the fundamental laws of physics. They aim to solve complex physical problems by incorporating physical constraints into the learning process, resulting in more accurate and physically consistent predictions.

```python
import tensorflow as tf
import numpy as np

class PINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(50, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(50, activation='tanh')
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

# Create a simple PINN model
model = PINN()
```

Slide 2: Key Components of PINNs

PINNs consist of three main components: the neural network architecture, the physics-based loss function, and the training process. The neural network learns to approximate the solution to a physical problem, while the physics-based loss function ensures that the predictions satisfy the underlying physical laws.

```python
def pde_loss(model, x, t):
    # Compute gradients
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)
        u = model(tf.stack([x, t], axis=1))
        u_x = tape.gradient(u, x)
        u_xx = tape.gradient(u_x, x)
    u_t = tape.gradient(u, t)
    
    # PDE: u_t + u * u_x - (0.01 / pi) * u_xx = 0
    pde = u_t + u * u_x - (0.01 / np.pi) * u_xx
    return tf.reduce_mean(tf.square(pde))

# Define loss function
def loss(model, x, t, u):
    u_pred = model(tf.stack([x, t], axis=1))
    mse_loss = tf.reduce_mean(tf.square(u - u_pred))
    physics_loss = pde_loss(model, x, t)
    return mse_loss + physics_loss
```

Slide 3: Training a PINN

Training a PINN involves minimizing both the data-driven loss and the physics-based loss. This process ensures that the model not only fits the available data but also respects the underlying physical laws governing the system.

```python
@tf.function
def train_step(model, optimizer, x, t, u):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, t, u)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value

# Training loop
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
epochs = 10000

for epoch in range(epochs):
    loss_value = train_step(model, optimizer, x_train, t_train, u_train)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value.numpy()}")
```

Slide 4: Advantages of PINNs

PINNs offer several advantages over traditional numerical methods. They can handle complex geometries, require fewer data points, and provide smooth solutions. Additionally, PINNs can incorporate multiple physical constraints simultaneously, making them suitable for multiphysics problems.

```python
# Example: Solving heat equation with complex geometry
def complex_geometry(x, y):
    return tf.cast(((x - 0.5)**2 + (y - 0.5)**2 <= 0.16) & 
                   ((x - 0.5)**2 + (y - 0.5)**2 >= 0.09), tf.float32)

def heat_equation_pinn(model, x, y, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, y, t])
        u = model(tf.stack([x, y, t], axis=1))
        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
    
    u_xx = tape.gradient(u_x, x)
    u_yy = tape.gradient(u_y, y)
    u_t = tape.gradient(u, t)
    
    pde = u_t - (u_xx + u_yy)
    return tf.reduce_mean(tf.square(pde * complex_geometry(x, y)))
```

Slide 5: Real-Life Example: Fluid Dynamics

PINNs can be applied to solve complex fluid dynamics problems, such as predicting the flow around an airfoil. This example demonstrates how PINNs can handle the Navier-Stokes equations, which govern fluid motion.

```python
def navier_stokes_2d(model, x, y, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, y, t])
        predictions = model(tf.stack([x, y, t], axis=1))
        u, v, p = tf.split(predictions, 3, axis=1)
        
        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
        u_t = tape.gradient(u, t)
        v_x = tape.gradient(v, x)
        v_y = tape.gradient(v, y)
        v_t = tape.gradient(v, t)
        p_x = tape.gradient(p, x)
        p_y = tape.gradient(p, y)
    
    u_xx = tape.gradient(u_x, x)
    u_yy = tape.gradient(u_y, y)
    v_xx = tape.gradient(v_x, x)
    v_yy = tape.gradient(v_y, y)
    
    continuity = u_x + v_y
    x_momentum = u_t + u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy)
    y_momentum = v_t + u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy)
    
    return tf.reduce_mean(tf.square(continuity) + tf.square(x_momentum) + tf.square(y_momentum))

# Re: Reynolds number
Re = 100
```

Slide 6: Implementing Boundary Conditions in PINNs

Boundary conditions are crucial in physical problems. PINNs can incorporate various types of boundary conditions, such as Dirichlet, Neumann, or mixed conditions, directly into the loss function.

```python
def dirichlet_bc(model, x_bc, t_bc, u_bc):
    u_pred = model(tf.stack([x_bc, t_bc], axis=1))
    return tf.reduce_mean(tf.square(u_pred - u_bc))

def neumann_bc(model, x_bc, t_bc, du_dx_bc):
    with tf.GradientTape() as tape:
        tape.watch(x_bc)
        u_pred = model(tf.stack([x_bc, t_bc], axis=1))
    du_dx_pred = tape.gradient(u_pred, x_bc)
    return tf.reduce_mean(tf.square(du_dx_pred - du_dx_bc))

def total_loss(model, x, t, u, x_bc, t_bc, u_bc, x_neumann, t_neumann, du_dx_bc):
    mse_loss = tf.reduce_mean(tf.square(model(tf.stack([x, t], axis=1)) - u))
    physics_loss = pde_loss(model, x, t)
    bc_loss = dirichlet_bc(model, x_bc, t_bc, u_bc) + neumann_bc(model, x_neumann, t_neumann, du_dx_bc)
    return mse_loss + physics_loss + bc_loss
```

Slide 7: Handling Inverse Problems with PINNs

PINNs excel at solving inverse problems, where the goal is to infer system parameters or initial conditions from observed data. This capability is particularly useful in fields like geophysics and material science.

```python
class InversePINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(50, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(50, activation='tanh')
        self.output_layer = tf.keras.layers.Dense(1)
        self.param = tf.Variable(initial_value=tf.random.uniform([1]), trainable=True)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

def inverse_problem_loss(model, x, t, u_obs):
    u_pred = model(tf.stack([x, t], axis=1))
    mse_loss = tf.reduce_mean(tf.square(u_pred - u_obs))
    physics_loss = pde_loss(model, x, t)
    return mse_loss + physics_loss

# Training loop for inverse problem
inverse_model = InversePINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss_value = inverse_problem_loss(inverse_model, x_obs, t_obs, u_obs)
    grads = tape.gradient(loss_value, inverse_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, inverse_model.trainable_variables))
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value.numpy()}, Inferred parameter: {inverse_model.param.numpy()[0]}")
```

Slide 8: Multi-Physics Problems and PINNs

PINNs can handle multi-physics problems by incorporating multiple governing equations into the loss function. This approach is particularly useful for complex systems involving interactions between different physical phenomena.

```python
def multi_physics_pinn(model, x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        predictions = model(tf.stack([x, t], axis=1))
        u, v = tf.split(predictions, 2, axis=1)
        
        u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
        v_x = tape.gradient(v, x)
        v_t = tape.gradient(v, t)
    
    u_xx = tape.gradient(u_x, x)
    v_xx = tape.gradient(v_x, x)
    
    # Coupled PDEs
    pde1 = u_t - 0.1 * u_xx + u * v
    pde2 = v_t - 0.01 * v_xx - u * v
    
    return tf.reduce_mean(tf.square(pde1) + tf.square(pde2))

# Multi-physics PINN model
multi_physics_model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='tanh', input_shape=(2,)),
    tf.keras.layers.Dense(50, activation='tanh'),
    tf.keras.layers.Dense(2)
])
```

Slide 9: Real-Life Example: Heat Transfer in a Composite Material

PINNs can model heat transfer in complex materials with varying thermal properties. This example demonstrates how PINNs handle spatially varying coefficients in a heat equation for a composite material.

```python
def composite_heat_transfer(model, x, y, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, y, t])
        u = model(tf.stack([x, y, t], axis=1))
        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
    
    u_xx = tape.gradient(u_x, x)
    u_yy = tape.gradient(u_y, y)
    u_t = tape.gradient(u, t)
    
    # Spatially varying thermal diffusivity
    alpha = tf.where(x < 0.5, 0.1, 0.5)
    
    pde = u_t - alpha * (u_xx + u_yy)
    return tf.reduce_mean(tf.square(pde))

# Composite material PINN model
composite_model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='tanh', input_shape=(3,)),
    tf.keras.layers.Dense(50, activation='tanh'),
    tf.keras.layers.Dense(1)
])

# Generate sample data
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 50)
X, Y, T = np.meshgrid(x, y, t)

# Train the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
for epoch in range(10000):
    with tf.GradientTape() as tape:
        loss = composite_heat_transfer(composite_model, X.flatten(), Y.flatten(), T.flatten())
    gradients = tape.gradient(loss, composite_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, composite_model.trainable_variables))
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

# Visualize results
import matplotlib.pyplot as plt

u_pred = composite_model(np.stack([X[:,:,0], Y[:,:,0], T[:,:,0]], axis=-1)).numpy().reshape(100, 100)

plt.figure(figsize=(10, 8))
plt.contourf(X[:,:,0], Y[:,:,0], u_pred, levels=20, cmap='viridis')
plt.colorbar(label='Temperature')
plt.title('Predicted Temperature Distribution in Composite Material')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

Slide 10: Uncertainty Quantification in PINNs

PINNs can be extended to provide uncertainty estimates for their predictions. This is crucial in many scientific and engineering applications where understanding the confidence in model outputs is essential.

```python
import tensorflow_probability as tfp

class BayesianPINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tfp.layers.DenseVariational(50, activation='tanh')
        self.dense2 = tfp.layers.DenseVariational(50, activation='tanh')
        self.output_layer = tfp.layers.DenseVariational(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

# Create Bayesian PINN
bayesian_model = BayesianPINN()

# Training loop (pseudocode)
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        predictions = bayesian_model(inputs)
        loss = compute_loss(predictions, targets)
    gradients = tape.gradient(loss, bayesian_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, bayesian_model.trainable_variables))

# Prediction with uncertainty
num_samples = 100
predictions = [bayesian_model(test_inputs) for _ in range(num_samples)]
mean_prediction = tf.reduce_mean(predictions, axis=0)
std_prediction = tf.math.reduce_std(predictions, axis=0)
```

Slide 11: Transfer Learning with PINNs

Transfer learning in PINNs allows knowledge from one physical system to be applied to another related system. This approach can significantly reduce the amount of data and training time required for new problems.

```python
# Pre-trained PINN for heat equation
pretrained_model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='tanh', input_shape=(2,)),
    tf.keras.layers.Dense(50, activation='tanh'),
    tf.keras.layers.Dense(1)
])

# Load pre-trained weights
pretrained_model.load_weights('pretrained_heat_equation_weights.h5')

# New model for advection-diffusion equation
new_model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='tanh', input_shape=(2,)),
    tf.keras.layers.Dense(50, activation='tanh'),
    tf.keras.layers.Dense(1)
])

# Transfer weights from pre-trained model
new_model.layers[0].set_weights(pretrained_model.layers[0].get_weights())
new_model.layers[1].set_weights(pretrained_model.layers[1].get_weights())

# Fine-tune the new model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
new_model.compile(optimizer=optimizer, loss='mse')
new_model.fit(new_data_x, new_data_y, epochs=100, batch_size=32)
```

Slide 12: Handling Discontinuities in PINNs

PINNs can be adapted to handle problems with discontinuities, which are common in many physical systems. This approach involves modifying the network architecture and loss function to account for sharp transitions.

```python
class DiscontinuousPINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.branch1 = tf.keras.Sequential([
            tf.keras.layers.Dense(50, activation='tanh'),
            tf.keras.layers.Dense(50, activation='tanh'),
            tf.keras.layers.Dense(1)
        ])
        self.branch2 = tf.keras.Sequential([
            tf.keras.layers.Dense(50, activation='tanh'),
            tf.keras.layers.Dense(50, activation='tanh'),
            tf.keras.layers.Dense(1)
        ])
        self.switch = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        s = self.switch(inputs)
        y1 = self.branch1(inputs)
        y2 = self.branch2(inputs)
        return s * y1 + (1 - s) * y2

# Custom loss function for discontinuous problems
def discontinuous_loss(model, x, t, u_true):
    u_pred = model(tf.stack([x, t], axis=1))
    mse_loss = tf.reduce_mean(tf.square(u_pred - u_true))
    physics_loss = pde_loss(model, x, t)
    return mse_loss + physics_loss

# Train the discontinuous PINN
disc_model = DiscontinuousPINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        loss = discontinuous_loss(disc_model, x_train, t_train, u_train)
    gradients = tape.gradient(loss, disc_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, disc_model.trainable_variables))
```

Slide 13: Scaling PINNs for Large-Scale Problems

For large-scale problems, traditional PINNs may face computational challenges. Techniques such as domain decomposition and parallel training can be employed to scale PINNs to handle complex, high-dimensional systems.

```python
import tensorflow as tf

def domain_decomposition(x, y, num_subdomains):
    x_split = tf.split(x, num_subdomains)
    y_split = tf.split(y, num_subdomains)
    return list(zip(x_split, y_split))

class SubdomainPINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(50, activation='tanh', input_shape=(2,)),
            tf.keras.layers.Dense(50, activation='tanh'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        return self.model(inputs)

# Create subdomains and models
num_subdomains = 4
subdomains = domain_decomposition(x, y, num_subdomains)
subdomain_models = [SubdomainPINN() for _ in range(num_subdomains)]

# Parallel training (pseudocode)
@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        predictions = model(tf.stack([x, y], axis=1))
        loss = compute_loss(predictions, targets)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

for epoch in range(num_epochs):
    losses = []
    for i, (subdomain_x, subdomain_y) in enumerate(subdomains):
        loss = train_step(subdomain_models[i], subdomain_x, subdomain_y)
        losses.append(loss)
    avg_loss = tf.reduce_mean(losses)
    print(f"Epoch {epoch}, Average Loss: {avg_loss.numpy()}")
```

Slide 14: Future Directions and Challenges in PINNs

As PINNs continue to evolve, several challenges and opportunities emerge:

1. Improving training stability and convergence for complex systems.
2. Developing adaptive sampling techniques for more efficient training.
3. Incorporating physical symmetries and conservation laws into network architectures.
4. Extending PINNs to handle stochastic differential equations and uncertainty quantification.
5. Integrating PINNs with other machine learning techniques like reinforcement learning for optimal control problems.

```python
# Conceptual code for adaptive sampling in PINNs
def adaptive_sampling(model, x_range, y_range, num_samples):
    x = tf.random.uniform((num_samples,), *x_range)
    y = tf.random.uniform((num_samples,), *y_range)
    
    with tf.GradientTape() as tape:
        tape.watch([x, y])
        u = model(tf.stack([x, y], axis=1))
    
    gradients = tape.gradient(u, [x, y])
    gradient_magnitude = tf.sqrt(tf.square(gradients[0]) + tf.square(gradients[1]))
    
    # Select points with highest gradient magnitude
    _, indices = tf.nn.top_k(gradient_magnitude, k=num_samples // 2)
    x_selected = tf.gather(x, indices)
    y_selected = tf.gather(y, indices)
    
    return x_selected, y_selected

# Use adaptive sampling in training loop
for epoch in range(num_epochs):
    x_sampled, y_sampled = adaptive_sampling(model, x_range, y_range, 1000)
    train_step(model, x_sampled, y_sampled)
```

Slide 15: Additional Resources

For further exploration of Physics Informed Neural Networks, consider the following resources:

1. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations" by M. Raissi, P. Perdikaris, and G.E. Karniadakis (2019). ArXiv: [https://arxiv.org/abs/1711.10561](https://arxiv.org/abs/1711.10561)
2. "Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations" by M. Raissi, A. Yazdani, and G.E. Karniadakis (2020). ArXiv: [https://arxiv.org/abs/1808.04327](https://arxiv.org/abs/1808.04327)
3. "Physics-informed neural networks for high-speed flows" by X. Meng, Z. Li, D. Zhang, and G.E. Karniadakis (2020). ArXiv: [https://arxiv.org/abs/2003.12397](https://arxiv.org/abs/2003.12397)
4. "Solving forward and inverse problems in cardiovascular mathematics using physics-informed neural networks" by T. Chiesa, Z. Li, A. Kissas, and G.E. Karniadakis (2021). ArXiv: [https://arxiv.org/abs/2102.01880](https://arxiv.org/abs/2102.01880)

These papers provide in-depth discussions on the theory and applications of PINNs in various scientific domains.

