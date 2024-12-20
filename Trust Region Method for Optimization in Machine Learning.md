## Trust Region Method for Optimization in Machine Learning
Slide 1: Introduction to Trust Region Method (TRM)

The Trust Region Method is an optimization algorithm used in machine learning and artificial intelligence to find the minimum of a function. It's particularly useful for non-linear optimization problems. TRM works by defining a region around the current point within which it trusts the approximation to be accurate.

```python
import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return x**2 + 4*np.sin(x)

x = np.linspace(-10, 10, 1000)
y = objective_function(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Objective Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
```

Slide 2: Basic Concepts of TRM

TRM iteratively improves the solution by solving a simpler subproblem within a trust region. The algorithm defines a quadratic approximation of the objective function and a trust region where this approximation is considered reliable. The size of this region is adjusted based on the agreement between the approximation and the true function.

```python
def quadratic_approximation(x, x0, f0, grad, hess):
    return f0 + grad * (x - x0) + 0.5 * hess * (x - x0)**2

x0 = 0
f0 = objective_function(x0)
grad = 2 * x0 + 4 * np.cos(x0)
hess = 2 - 4 * np.sin(x0)

x_approx = np.linspace(-2, 2, 100)
y_approx = quadratic_approximation(x_approx, x0, f0, grad, hess)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='True function')
plt.plot(x_approx, y_approx, label='Quadratic approximation')
plt.title('Objective Function and Its Quadratic Approximation')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 3: Trust Region Definition

The trust region is typically defined as a ball around the current point, with a radius that determines the maximum step size. The algorithm solves the subproblem of minimizing the quadratic approximation within this region.

```python
def trust_region(x0, radius):
    theta = np.linspace(0, 2*np.pi, 100)
    x = x0 + radius * np.cos(theta)
    y = objective_function(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Objective function')
    plt.scatter(x0, objective_function(x0), color='red', s=100, label='Current point')
    plt.title(f'Trust Region (radius = {radius})')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

trust_region(0, 1)  # Example with x0 = 0 and radius = 1
```

Slide 4: Subproblem Solution

Within the trust region, TRM solves a subproblem to find the next point. This subproblem involves minimizing the quadratic approximation subject to the trust region constraint.

```python
def solve_subproblem(x0, grad, hess, radius):
    # Simplified subproblem solution for 1D case
    step = -grad / hess
    if np.abs(step) > radius:
        step = np.sign(step) * radius
    return x0 + step

x0 = 0
grad = 2 * x0 + 4 * np.cos(x0)
hess = 2 - 4 * np.sin(x0)
radius = 1

new_x = solve_subproblem(x0, grad, hess, radius)

print(f"Current point: {x0}")
print(f"New point: {new_x}")
```

Slide 5: Trust Region Update

After solving the subproblem, TRM compares the actual improvement in the objective function with the predicted improvement from the quadratic approximation. Based on this comparison, it updates the trust region size.

```python
def update_trust_region(actual_reduction, predicted_reduction, radius):
    ratio = actual_reduction / predicted_reduction
    if ratio < 0.25:
        return radius * 0.5
    elif ratio > 0.75 and abs(radius - abs(step)) < 1e-8:
        return min(2 * radius, 10)
    else:
        return radius

actual_reduction = objective_function(x0) - objective_function(new_x)
predicted_reduction = quadratic_approximation(x0, x0, f0, grad, hess) - quadratic_approximation(new_x, x0, f0, grad, hess)

new_radius = update_trust_region(actual_reduction, predicted_reduction, radius)

print(f"Old radius: {radius}")
print(f"New radius: {new_radius}")
```

Slide 6: TRM Algorithm Implementation

Let's implement a basic version of the Trust Region Method algorithm for a one-dimensional optimization problem.

```python
def trust_region_method(obj_func, x0, max_iter=100, tol=1e-6):
    x = x0
    radius = 1.0
    
    for i in range(max_iter):
        f = obj_func(x)
        grad = 2 * x + 4 * np.cos(x)  # Gradient of our objective function
        hess = 2 - 4 * np.sin(x)  # Hessian of our objective function
        
        step = solve_subproblem(x, grad, hess, radius)
        new_x = x + step
        
        actual_reduction = f - obj_func(new_x)
        predicted_reduction = f - quadratic_approximation(new_x, x, f, grad, hess)
        
        ratio = actual_reduction / predicted_reduction if predicted_reduction != 0 else 0
        
        if ratio > 0.75 and abs(step) == radius:
            radius = min(2 * radius, 10)
        elif ratio < 0.25:
            radius *= 0.25
        
        if ratio > 0:
            x = new_x
        
        if abs(grad) < tol:
            break
    
    return x, obj_func(x), i+1

result, min_value, iterations = trust_region_method(objective_function, 1.0)
print(f"Minimum found at x = {result}")
print(f"Minimum value = {min_value}")
print(f"Number of iterations: {iterations}")
```

Slide 7: Visualizing TRM Iterations

To better understand how TRM works, let's visualize its iterations on our objective function.

```python
def visualize_trm_iterations(obj_func, x0, max_iter=10):
    x = x0
    radius = 1.0
    
    plt.figure(figsize=(12, 8))
    x_range = np.linspace(-5, 5, 1000)
    plt.plot(x_range, obj_func(x_range), label='Objective function')
    
    for i in range(max_iter):
        f = obj_func(x)
        grad = 2 * x + 4 * np.cos(x)
        hess = 2 - 4 * np.sin(x)
        
        step = solve_subproblem(x, grad, hess, radius)
        new_x = x + step
        
        plt.scatter(x, f, color='red', s=100)
        plt.annotate(f'Iter {i+1}', (x, f), xytext=(5, 5), textcoords='offset points')
        
        x = new_x
        
        if abs(grad) < 1e-6:
            break
    
    plt.title('Trust Region Method Iterations')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

visualize_trm_iterations(objective_function, 1.0)
```

Slide 8: Advantages of TRM

TRM offers several advantages over other optimization methods:

1. Robustness: It can handle non-convex functions and is less sensitive to the initial point.
2. Global convergence: Under certain conditions, it converges to a local minimum from any starting point.
3. Fast convergence: It can achieve quadratic convergence rate near the solution.

```python
def compare_convergence(obj_func, x0, methods):
    plt.figure(figsize=(12, 8))
    x_range = np.linspace(-5, 5, 1000)
    plt.plot(x_range, obj_func(x_range), label='Objective function')
    
    for method, color in methods:
        x, _, iterations = method(obj_func, x0)
        plt.scatter(x, obj_func(x), color=color, s=100, label=f'{method.__name__} ({iterations} iter)')
    
    plt.title('Convergence Comparison')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Dummy gradient descent method for comparison
def gradient_descent(obj_func, x0, max_iter=100, learning_rate=0.1):
    x = x0
    for i in range(max_iter):
        grad = 2 * x + 4 * np.cos(x)
        x = x - learning_rate * grad
        if abs(grad) < 1e-6:
            break
    return x, obj_func(x), i+1

compare_convergence(objective_function, 1.0, [(trust_region_method, 'red'), (gradient_descent, 'blue')])
```

Slide 9: TRM in Higher Dimensions

While our examples have been in one dimension, TRM is particularly powerful in higher-dimensional optimization problems. Let's look at how it can be extended to 2D.

```python
def objective_function_2d(x):
    return x[0]**2 + 2*x[1]**2 + 2*np.sin(x[0]) + 2*np.sin(x[1])

def visualize_2d_function(obj_func):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = obj_func(np.array([X, Y]))
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('2D Objective Function')
    fig.colorbar(surf)
    plt.show()

visualize_2d_function(objective_function_2d)
```

Slide 10: Implementing TRM in 2D

Let's implement a basic version of TRM for our 2D objective function.

```python
def trust_region_method_2d(obj_func, x0, max_iter=100, tol=1e-6):
    x = np.array(x0)
    radius = 1.0
    
    def gradient(x):
        return np.array([2*x[0] + 2*np.cos(x[0]), 4*x[1] + 2*np.cos(x[1])])
    
    def hessian(x):
        return np.array([[2 - 2*np.sin(x[0]), 0],
                         [0, 4 - 2*np.sin(x[1])]])
    
    for i in range(max_iter):
        f = obj_func(x)
        grad = gradient(x)
        hess = hessian(x)
        
        # Solve the subproblem (simplified for this example)
        eigenvalues, eigenvectors = np.linalg.eigh(hess)
        min_eigenvalue = np.min(eigenvalues)
        
        if min_eigenvalue > 0:
            step = -np.linalg.solve(hess, grad)
            if np.linalg.norm(step) > radius:
                step = radius * step / np.linalg.norm(step)
        else:
            step = -radius * grad / np.linalg.norm(grad)
        
        new_x = x + step
        
        actual_reduction = f - obj_func(new_x)
        predicted_reduction = -np.dot(grad, step) - 0.5 * np.dot(step, np.dot(hess, step))
        
        ratio = actual_reduction / predicted_reduction if predicted_reduction != 0 else 0
        
        if ratio > 0.75 and np.linalg.norm(step) == radius:
            radius = min(2 * radius, 10)
        elif ratio < 0.25:
            radius *= 0.25
        
        if ratio > 0:
            x = new_x
        
        if np.linalg.norm(grad) < tol:
            break
    
    return x, obj_func(x), i+1

result, min_value, iterations = trust_region_method_2d(objective_function_2d, [1.0, 1.0])
print(f"Minimum found at x = {result}")
print(f"Minimum value = {min_value}")
print(f"Number of iterations: {iterations}")
```

Slide 11: Visualizing 2D TRM Iterations

To better understand how TRM works in 2D, let's visualize its iterations on our 2D objective function.

```python
def visualize_2d_trm_iterations(obj_func, x0, max_iter=10):
    x = np.array(x0)
    radius = 1.0
    
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = obj_func(np.array([X, Y]))
    
    plt.figure(figsize=(12, 8))
    plt.contour(X, Y, Z, levels=20)
    
    for i in range(max_iter):
        plt.scatter(x[0], x[1], color='red', s=100)
        plt.annotate(f'Iter {i+1}', (x[0], x[1]), xytext=(5, 5), textcoords='offset points')
        
        grad = np.array([2*x[0] + 2*np.cos(x[0]), 4*x[1] + 2*np.cos(x[1])])
        hess = np.array([[2 - 2*np.sin(x[0]), 0],
                         [0, 4 - 2*np.sin(x[1])]])
        
        step = solve_trm_subproblem(grad, hess, radius)
        x = x + step
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    plt.title('Trust Region Method Iterations (2D)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='f(x, y)')
    plt.show()

def solve_trm_subproblem(grad, hess, radius):
    # Simplified subproblem solution
    eigenvalues, eigenvectors = np.linalg.eigh(hess)
    min_eigenvalue = np.min(eigenvalues)
    
    if min_eigenvalue > 0:
        step = -np.linalg.solve(hess, grad)
        if np.linalg.norm(step) > radius:
            step = radius * step / np.linalg.norm(step)
    else:
        step = -radius * grad / np.linalg.norm(grad)
    
    return step

visualize_2d_trm_iterations(objective_function_2d, [1.0, 1.0])
```

Slide 12: TRM in Machine Learning

Trust Region Methods are widely used in machine learning, particularly for training neural networks. They help overcome challenges like vanishing gradients and provide more stable convergence.

```python
import tensorflow as tf

class TRMOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01, radius=1.0, name="TRMOptimizer", **kwargs):
        super(TRMOptimizer, self).__init__(name, **kwargs)
        self._lr = learning_rate
        self._radius = radius

    def _resource_apply_dense(self, grad, var):
        # Simplified TRM update
        step = -self._lr * grad
        if tf.norm(step) > self._radius:
            step = self._radius * step / tf.norm(step)
        var.assign_add(step)
        return var

    def get_config(self):
        config = super(TRMOptimizer, self).get_config()
        config.update({"learning_rate": self._lr, "radius": self._radius})
        return config

# Example usage
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])

optimizer = TRMOptimizer()
model.compile(optimizer=optimizer, loss='mse')

# Train the model (assuming x_train and y_train are defined)
# model.fit(x_train, y_train, epochs=10, batch_size=32)
```

Slide 13: Real-Life Example: Image Classification

Let's consider an image classification task using a convolutional neural network (CNN) optimized with a Trust Region Method.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

model = create_cnn_model()
optimizer = TRMOptimizer(learning_rate=0.001, radius=0.1)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
# history = model.fit(train_images, train_labels, epochs=5, 
#                     validation_data=(test_images, test_labels))

# Evaluate the model
# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
# print(f"Test accuracy: {test_acc}")
```

Slide 14: Real-Life Example: Robotics Path Planning

Trust Region Methods are also used in robotics for path planning and control optimization. Here's a simplified example of using TRM for 2D robot path planning.

```python
import numpy as np
import matplotlib.pyplot as plt

def obstacle_cost(x, y, obstacles):
    cost = 0
    for ox, oy, radius in obstacles:
        dist = np.sqrt((x - ox)**2 + (y - oy)**2)
        if dist < radius:
            cost += (radius - dist)**2
    return cost

def path_cost(path, obstacles):
    total_cost = 0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_cost += segment_length + obstacle_cost(x2, y2, obstacles)
    return total_cost

def optimize_path(initial_path, obstacles, max_iter=100):
    path = initial_path.()
    for _ in range(max_iter):
        for i in range(1, len(path) - 1):
            current_cost = path_cost(path, obstacles)
            
            # Try moving the point in different directions
            for dx, dy in [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]:
                new_path = path.()
                new_path[i] = (path[i][0] + dx, path[i][1] + dy)
                new_cost = path_cost(new_path, obstacles)
                
                if new_cost < current_cost:
                    path = new_path
                    break
    
    return path

# Example usage
start = (0, 0)
goal = (10, 10)
obstacles = [(3, 3, 1), (7, 7, 1.5)]  # (x, y, radius)

initial_path = [start, (5, 5), goal]
optimized_path = optimize_path(initial_path, obstacles)

# Visualize the result
plt.figure(figsize=(10, 10))
plt.plot([p[0] for p in initial_path], [p[1] for p in initial_path], 'b--', label='Initial Path')
plt.plot([p[0] for p in optimized_path], [p[1] for p in optimized_path], 'r-', label='Optimized Path')

for ox, oy, radius in obstacles:
    circle = plt.Circle((ox, oy), radius, fill=False)
    plt.gca().add_artist(circle)

plt.scatter(*zip(start, goal), c=['g', 'r'], s=100)
plt.legend()
plt.title('Robot Path Planning with TRM')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.show()
```

Slide 15: Additional Resources

For those interested in delving deeper into Trust Region Methods and their applications in machine learning and artificial intelligence, here are some valuable resources:

1. "Trust Region Methods" by Conn, Gould, and Toint (2000) - A comprehensive book on the subject.
2. "Trust Region Methods for Machine Learning" by Lin and Mor� (ArXiv:1303.7309) - Explores the application of TRM in machine learning contexts.
3. "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers" by Boyd et al. (2011) - Discusses optimization methods including trust region approaches.
4. "On the Use of Stochastic Hessian Information in Optimization Methods for Machine Learning" by Byrd et al. (ArXiv:1107.5586) - Investigates the use of Hessian information in machine learning optimization.
5. "Trust-Region Methods on Riemannian Manifolds" by Absil et al. (2007) - Extends trust region methods to optimization on manifolds, relevant for certain ML problems.

These resources provide a mix of theoretical foundations and practical applications of Trust Region Methods in various domains of machine learning and AI.

