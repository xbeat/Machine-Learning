## Accelerating Python with Numba
Slide 1: Introduction to Numba

Numba is a just-in-time compiler for Python that can significantly speed up numerical and scientific Python code. It works by translating Python functions to optimized machine code at runtime using the industry-standard LLVM compiler library.

```python
import numba
import numpy as np

@numba.jit
def sum_array(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total

# Example usage
data = np.random.rand(1000000)
result = sum_array(data)
print(f"Sum of array: {result}")
```

Slide 2: How Numba Works

Numba works by analyzing your Python code and generating optimized machine code tailored to your CPU. It can automatically parallelize and vectorize your code, often achieving performance similar to hand-written C code.

```python
import numba
import time
import numpy as np

@numba.jit(nopython=True, parallel=True)
def monte_carlo_pi(nsamples):
    acc = 0
    for i in numba.prange(nsamples):
        x = np.random.random()
        y = np.random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

nsamples = 10_000_000
start = time.time()
pi_estimate = monte_carlo_pi(nsamples)
end = time.time()

print(f"Pi estimate: {pi_estimate}")
print(f"Time taken: {end - start:.2f} seconds")
```

Slide 3: Basic Numba Usage: The @jit Decorator

The @jit decorator is the simplest way to use Numba. It tells Numba to compile your function. The first time the function is called, it will be compiled to machine code. Subsequent calls will use the compiled version.

```python
import numba
import numpy as np
import time

@numba.jit
def sum_squares(arr):
    sum = 0.0
    for i in range(arr.shape[0]):
        sum += arr[i] * arr[i]
    return sum

# Compare performance
data = np.random.rand(10_000_000)

start = time.time()
result_numpy = np.sum(np.square(data))
end = time.time()
print(f"NumPy time: {end - start:.4f} seconds")

start = time.time()
result_numba = sum_squares(data)
end = time.time()
print(f"Numba time: {end - start:.4f} seconds")

print(f"Results match: {np.allclose(result_numpy, result_numba)}")
```

Slide 4: Numba and NumPy: A Powerful Combination

Numba works particularly well with NumPy arrays. It can optimize operations on NumPy arrays to be as fast as hand-written C code. This makes it an excellent tool for scientific computing and data analysis.

```python
import numba
import numpy as np
import time

@numba.jit(nopython=True)
def convolve2d(image, kernel):
    M, N = image.shape
    K, L = kernel.shape
    result = np.zeros((M-K+1, N-L+1))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.sum(image[i:i+K, j:j+L] * kernel)
    return result

# Example usage
image = np.random.rand(1000, 1000)
kernel = np.random.rand(3, 3)

start = time.time()
result = convolve2d(image, kernel)
end = time.time()

print(f"Convolution time: {end - start:.4f} seconds")
print(f"Result shape: {result.shape}")
```

Slide 5: Numba's nopython Mode

Numba's nopython mode ensures that the entire function is compiled without any calls to the Python interpreter. This often results in the best performance but can be more restrictive in terms of what Python features are allowed.

```python
import numba
import numpy as np
import time

@numba.jit(nopython=True)
def mandelbrot(h, w, max_iter):
    y, x = np.mgrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
    c = x + y*1j
    z = c
    divtime = max_iter + np.zeros(z.shape, dtype=int)
    
    for i in range(max_iter):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2
        div_now = diverge & (divtime == max_iter)
        divtime[div_now] = i
        z[diverge] = 2
    
    return divtime

h, w = 1000, 1500
max_iter = 100

start = time.time()
mandelbrot_set = mandelbrot(h, w, max_iter)
end = time.time()

print(f"Mandelbrot set calculation time: {end - start:.4f} seconds")
print(f"Output shape: {mandelbrot_set.shape}")
```

Slide 6: Parallelization with Numba

Numba can automatically parallelize your code using multiple CPU cores. This is done using the parallel=True option in the @jit decorator and the prange function for parallel loops.

```python
import numba
import numpy as np
import time

@numba.jit(nopython=True, parallel=True)
def parallel_sum(arr):
    sum = 0.0
    for i in numba.prange(arr.shape[0]):
        sum += arr[i]
    return sum

# Compare parallel vs non-parallel
data = np.random.rand(100_000_000)

start = time.time()
result_serial = np.sum(data)
end = time.time()
print(f"Serial time: {end - start:.4f} seconds")

start = time.time()
result_parallel = parallel_sum(data)
end = time.time()
print(f"Parallel time: {end - start:.4f} seconds")

print(f"Results match: {np.allclose(result_serial, result_parallel)}")
```

Slide 7: Numba and Custom Data Types

Numba supports custom data types through its @jitclass decorator. This allows you to create high-performance classes that can be used in Numba-compiled functions.

```python
import numba
import numpy as np

spec = [
    ('position', numba.float64[:]),
    ('velocity', numba.float64[:]),
    ('mass', numba.float64)
]

@numba.jitclass(spec)
class Particle:
    def __init__(self, position, velocity, mass):
        self.position = position
        self.velocity = velocity
        self.mass = mass
    
    def kinetic_energy(self):
        return 0.5 * self.mass * np.sum(self.velocity**2)

# Example usage
position = np.array([1.0, 2.0, 3.0])
velocity = np.array([0.1, 0.2, 0.3])
mass = 1.5

particle = Particle(position, velocity, mass)
ke = particle.kinetic_energy()
print(f"Particle kinetic energy: {ke}")
```

Slide 8: Numba and GPU Acceleration

Numba can also compile Python code to run on NVIDIA GPUs using CUDA. This can lead to massive speedups for certain types of computations, especially those involving large arrays.

```python
import numba
from numba import cuda
import numpy as np
import time

@cuda.jit
def vector_add(a, b, result):
    i = cuda.grid(1)
    if i < result.shape[0]:
        result[i] = a[i] + b[i]

# Example usage
n = 10_000_000
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)
result = np.zeros_like(a)

threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

start = time.time()
vector_add[blocks_per_grid, threads_per_block](a, b, result)
cuda.synchronize()
end = time.time()

print(f"GPU vector addition time: {end - start:.4f} seconds")
print(f"Result (first 5 elements): {result[:5]}")
```

Slide 9: Debugging Numba Code

Debugging Numba-compiled code can be challenging. Numba provides several tools to help, including the option to disable compilation for easier debugging and the ability to print debug information.

```python
import numba

@numba.jit(debug=True)
def buggy_function(x):
    # This function has a bug (division by zero when x = 0)
    return 1 / x

# Try to run the function
try:
    result = buggy_function(0)
except Exception as e:
    print(f"Caught exception: {e}")

# Disable JIT compilation for debugging
buggy_function.disable_jit()
try:
    result = buggy_function(0)
except Exception as e:
    print(f"Caught exception in Python mode: {e}")

# Print Numba debug info
print(buggy_function.inspect_types())
```

Slide 10: Numba Performance Tips

To get the best performance from Numba, follow these tips: use nopython mode when possible, avoid Python objects in compiled functions, use NumPy arrays instead of lists, and vectorize operations where applicable.

```python
import numba
import numpy as np
import time

@numba.vectorize
def fast_exp(x):
    return np.exp(x)

# Compare performance
data = np.random.rand(10_000_000)

start = time.time()
result_numpy = np.exp(data)
end = time.time()
print(f"NumPy exp time: {end - start:.4f} seconds")

start = time.time()
result_numba = fast_exp(data)
end = time.time()
print(f"Numba vectorized exp time: {end - start:.4f} seconds")

print(f"Results match: {np.allclose(result_numpy, result_numba)}")
```

Slide 11: Real-Life Example: Image Processing

Let's use Numba to speed up a common image processing task: applying a Gaussian blur to an image. This example demonstrates how Numba can accelerate computationally intensive tasks in real-world scenarios.

```python
import numba
import numpy as np
from scipy import signal
import time
from PIL import Image

@numba.jit(nopython=True)
def gaussian_kernel(size, sigma=1.0):
    k = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            x, y = i - size//2, j - size//2
            k[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return k / np.sum(k)

@numba.jit(nopython=True)
def apply_kernel(image, kernel):
    h, w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    result = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            result[i, j] = np.sum(padded[i:i+k_h, j:j+k_w] * kernel)
    return result

# Load image and convert to grayscale
img = Image.open('your_image.jpg').convert('L')
img_array = np.array(img, dtype=np.float32) / 255.0

# Apply Gaussian blur
kernel = gaussian_kernel(15, sigma=2.0)

start = time.time()
blurred = apply_kernel(img_array, kernel)
end = time.time()

print(f"Numba Gaussian blur time: {end - start:.4f} seconds")

# Convert back to image and save
blurred_img = Image.fromarray((blurred * 255).astype(np.uint8))
blurred_img.save('blurred_image.jpg')
```

Slide 12: Real-Life Example: N-body Simulation

N-body simulations are computationally intensive and commonly used in physics and astronomy. Numba can significantly speed up these simulations, making it possible to run larger and more complex models.

```python
import numba
import numpy as np
import matplotlib.pyplot as plt
import time

@numba.jit(nopython=True)
def compute_acceleration(pos, mass, G, softening):
    n = pos.shape[0]
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = pos[j] - pos[i]
                r = np.sqrt(np.sum(dx**2) + softening**2)
                acc[i] += G * mass[j] * dx / r**3
    return acc

@numba.jit(nopython=True)
def update_pos_vel(pos, vel, acc, dt):
    vel += acc * dt
    pos += vel * dt
    return pos, vel

@numba.jit(nopython=True)
def n_body_simulation(n_bodies, n_steps, dt, G, softening):
    pos = np.random.randn(n_bodies, 3)
    vel = np.random.randn(n_bodies, 3) * 0.1
    mass = np.random.rand(n_bodies) * 0.1 + 0.9
    
    for _ in range(n_steps):
        acc = compute_acceleration(pos, mass, G, softening)
        pos, vel = update_pos_vel(pos, vel, acc, dt)
    
    return pos

# Run simulation
n_bodies, n_steps = 1000, 100
G, softening, dt = 1.0, 0.1, 0.01

start = time.time()
final_pos = n_body_simulation(n_bodies, n_steps, dt, G, softening)
end = time.time()

print(f"N-body simulation time: {end - start:.4f} seconds")

# Plot final positions
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(final_pos[:, 0], final_pos[:, 1], final_pos[:, 2])
plt.savefig('n_body_simulation.png')
plt.close()
```

Slide 13: Limitations and Considerations

While Numba is powerful, it has limitations. Not all Python code can be compiled, especially code that relies heavily on Python objects or dynamic features. Additionally, compilation time can be significant for complex functions, which may not be worth it for functions that are only called a few times.

```python
import numba
import time

@numba.jit
def function_with_python_objects(x):
    # This function uses a Python dictionary, which Numba can't optimize
    d = {1: 'one', 2: 'two', 3: 'three'}
    return d.get(x, 'unknown')

# Measure compilation time
start = time.time()
result = function_with_python_objects(2)
end = time.time()
print(f"First call (compilation) time: {end - start:.4f} seconds")

# Measure subsequent call time
start = time.time()
result = function_with_python_objects(2)
end = time.time()
print(f"Second call time: {end - start:.4f} seconds")

# Function that Numba can optimize
@numba.jit(nopython=True)
def optimizable_function(x):
    return x * x + 2 * x + 1

start = time.time()
result = optimizable_function(1000000)
end = time.time()
print(f"Optimizable function time: {end - start:.8f} seconds")
```

Slide 14: Best Practices for Using Numba

To get the most out of Numba, follow these best practices: use nopython mode whenever possible, work with NumPy arrays instead of Python lists, avoid complex Python objects in compiled functions, and use Numba's built-in functions like prange for parallelization.

```python
import numba
import numpy as np

@numba.jit(nopython=True, parallel=True)
def numba_optimized_function(arr):
    result = np.zeros_like(arr)
    for i in numba.prange(arr.shape[0]):
        result[i] = np.sin(arr[i]) + np.cos(arr[i])
    return result

# Example usage
data = np.random.rand(10_000_000)
result = numba_optimized_function(data)
print(f"Result shape: {result.shape}")
print(f"First few values: {result[:5]}")
```

Slide 15: Additional Resources

For more information on Numba and its advanced features, consider exploring these resources:

1. Numba official documentation: [https://numba.pydata.org/](https://numba.pydata.org/)
2. "Accelerating Scientific Workloads with Numba" (arXiv:1801.03103): [https://arxiv.org/abs/1801.03103](https://arxiv.org/abs/1801.03103)
3. "Performance Comparison of Julia, Python, R, and Numba for Machine Learning" (arXiv:2206.10778): [https://arxiv.org/abs/2206.10778](https://arxiv.org/abs/2206.10778)
4. Numba GitHub repository: [https://github.com/numba/numba](https://github.com/numba/numba)

These resources provide in-depth information on Numba's capabilities, performance comparisons, and ongoing development.

