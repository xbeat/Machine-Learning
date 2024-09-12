## CUDA Programming with Python Slideshow
Slide 1: Introduction to CUDA Programming with Python

CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). This slideshow will introduce you to CUDA programming using Python, focusing on practical examples and actionable code.

```python
import numpy as np
import cupy as cp

# Compare CPU and GPU array creation
cpu_array = np.arange(1000000)
gpu_array = cp.arange(1000000)

print(f"CPU array: {cpu_array[:5]}...")
print(f"GPU array: {gpu_array[:5]}...")
```

Slide 2: Setting Up the CUDA Environment

Before diving into CUDA programming, we need to set up our environment. This involves installing the necessary libraries and verifying that our system can access the GPU.

```python
import cupy as cp

# Check CUDA availability
print(f"CUDA available: {cp.cuda.is_available()}")
print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
print(f"Number of GPUs: {cp.cuda.runtime.getDeviceCount()}")

# Get information about the first GPU
device = cp.cuda.Device(0)
print(f"GPU name: {device.name}")
print(f"GPU memory: {device.mem_info[1] / (1024**3):.2f} GB")
```

Slide 3: Basic Array Operations with CuPy

CuPy is a NumPy-like library for GPU-accelerated computing with Python. Let's explore some basic array operations using CuPy.

```python
import cupy as cp
import time

# Create a large array on GPU
start_time = time.time()
gpu_array = cp.arange(10**7)
gpu_time = time.time() - start_time

# Perform operations on GPU
gpu_result = cp.sum(gpu_array ** 2)

print(f"GPU array creation time: {gpu_time:.4f} seconds")
print(f"Sum of squares: {gpu_result}")
```

Slide 4: Memory Transfer between CPU and GPU

Understanding how to transfer data between CPU and GPU memory is crucial for effective CUDA programming.

```python
import numpy as np
import cupy as cp
import time

# Create a large array on CPU
cpu_array = np.random.rand(10**7)

# Transfer to GPU
start_time = time.time()
gpu_array = cp.asarray(cpu_array)
transfer_time = time.time() - start_time

# Perform operation on GPU
gpu_result = cp.mean(gpu_array)

# Transfer result back to CPU
cpu_result = gpu_result.get()

print(f"Transfer time to GPU: {transfer_time:.4f} seconds")
print(f"Mean value: {cpu_result}")
```

Slide 5: CUDA Kernels with Numba

Numba is a Just-In-Time (JIT) compiler that can compile Python functions to run on the GPU. Let's create a simple CUDA kernel using Numba.

```python
from numba import cuda
import numpy as np

@cuda.jit
def add_arrays(a, b, result):
    i = cuda.grid(1)
    if i < result.shape[0]:
        result[i] = a[i] + b[i]

# Prepare data
n = 10**6
a = np.arange(n).astype(np.float32)
b = np.arange(n).astype(np.float32)
result = np.zeros_like(a)

# Configure the blocks
threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

# Run the kernel
add_arrays[blocks_per_grid, threads_per_block](a, b, result)

print(f"First 5 elements of the result: {result[:5]}")
```

Slide 6: Matrix Multiplication with CuPy

Matrix multiplication is a common operation in many fields. Let's compare CPU and GPU performance for matrix multiplication.

```python
import numpy as np
import cupy as cp
import time

# Define matrix size
n = 1000

# CPU matrix multiplication
a_cpu = np.random.rand(n, n).astype(np.float32)
b_cpu = np.random.rand(n, n).astype(np.float32)

start_time = time.time()
c_cpu = np.dot(a_cpu, b_cpu)
cpu_time = time.time() - start_time

# GPU matrix multiplication
a_gpu = cp.asarray(a_cpu)
b_gpu = cp.asarray(b_cpu)

start_time = time.time()
c_gpu = cp.dot(a_gpu, b_gpu)
gpu_time = time.time() - start_time

print(f"CPU time: {cpu_time:.4f} seconds")
print(f"GPU time: {gpu_time:.4f} seconds")
print(f"Speedup: {cpu_time / gpu_time:.2f}x")
```

Slide 7: Parallel Reduction with CUDA

Parallel reduction is a technique used to perform operations like sum or max across an array efficiently on a GPU.

```python
from numba import cuda
import numpy as np

@cuda.reduce
def sum_reduce(a, b):
    return a + b

# Prepare data
n = 10**7
data = np.random.rand(n)

# Perform reduction on GPU
gpu_sum = sum_reduce(data)

# Compare with CPU sum
cpu_sum = np.sum(data)

print(f"GPU sum: {gpu_sum}")
print(f"CPU sum: {cpu_sum}")
print(f"Difference: {abs(gpu_sum - cpu_sum)}")
```

Slide 8: Real-life Example: Image Processing

Let's use CUDA to perform image processing tasks, such as applying a blur filter to an image.

```python
import cupy as cp
import numpy as np
from scipy import signal
import time

def gaussian_kernel(size, sigma=1):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

# Create a sample image (random noise)
image = np.random.rand(1000, 1000).astype(np.float32)

# Create blur kernel
kernel = gaussian_kernel(5, sigma=1).astype(np.float32)

# CPU convolution
start_time = time.time()
cpu_result = signal.convolve2d(image, kernel, mode='same', boundary='wrap')
cpu_time = time.time() - start_time

# GPU convolution
gpu_image = cp.asarray(image)
gpu_kernel = cp.asarray(kernel)

start_time = time.time()
gpu_result = cp.asnumpy(cp.signal.convolve2d(gpu_image, gpu_kernel, mode='same', boundary='wrap'))
gpu_time = time.time() - start_time

print(f"CPU time: {cpu_time:.4f} seconds")
print(f"GPU time: {gpu_time:.4f} seconds")
print(f"Speedup: {cpu_time / gpu_time:.2f}x")
```

Slide 9: Memory Management in CUDA

Proper memory management is crucial for efficient CUDA programming. Let's explore different types of GPU memory and how to use them.

```python
import cupy as cp

# Allocate memory on GPU
x_gpu = cp.zeros(10**6, dtype=cp.float32)

# Pinned memory for faster transfers
with cp.cuda.pinned_memory():
    x_pinned = cp.zeros(10**6, dtype=cp.float32)

# Unified memory (accessible from both CPU and GPU)
x_unified = cp.cuda.unified_memory.malloc(10**6 * 4)  # 4 bytes per float32

# Stream-ordered memory pool
pool = cp.cuda.MemoryPool()
with pool:
    x_pooled = cp.zeros(10**6, dtype=cp.float32)

print("Memory allocated successfully")
```

Slide 10: CUDA Streams

CUDA streams allow for concurrent execution of kernels and memory transfers, improving overall performance.

```python
import cupy as cp
import time

def process_data(data):
    return cp.sin(data) + cp.cos(data)

# Create two streams
stream1 = cp.cuda.Stream()
stream2 = cp.cuda.Stream()

# Prepare data
data1 = cp.random.rand(10**7)
data2 = cp.random.rand(10**7)

start_time = time.time()

# Process data in parallel using streams
with stream1:
    result1 = process_data(data1)

with stream2:
    result2 = process_data(data2)

# Synchronize streams
stream1.synchronize()
stream2.synchronize()

total_time = time.time() - start_time

print(f"Total processing time: {total_time:.4f} seconds")
print(f"Result 1 mean: {cp.mean(result1)}")
print(f"Result 2 mean: {cp.mean(result2)}")
```

Slide 11: Profiling CUDA Code

Profiling is essential for optimizing CUDA code performance. Let's use NVIDIA's nvprof tool to profile a simple CUDA kernel.

```python
from numba import cuda
import numpy as np
import os

@cuda.jit
def vector_add(a, b, c):
    i = cuda.grid(1)
    if i < c.shape[0]:
        c[i] = a[i] + b[i]

# Prepare data
n = 10**7
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)
c = np.zeros_like(a)

# Configure the blocks
threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

# Run the kernel with profiling
os.environ['CUDA_PROFILE'] = '1'
os.environ['CUDA_PROFILE_CSV'] = '1'
vector_add[blocks_per_grid, threads_per_block](a, b, c)
cuda.synchronize()
os.environ['CUDA_PROFILE'] = '0'

print("Profiling data saved. Use nvprof to analyze the results.")
```

Slide 12: Real-life Example: N-body Simulation

N-body simulations are used in various scientific fields. Let's implement a simple 2D N-body simulation using CUDA.

```python
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

@cp.cuda.jit
def update_positions(pos, vel, acc, dt):
    i = cp.cuda.grid(1)
    if i < pos.shape[0]:
        pos[i] += vel[i] * dt + 0.5 * acc[i] * dt**2
        vel[i] += acc[i] * dt

@cp.cuda.jit
def compute_acceleration(pos, mass, acc):
    i = cp.cuda.grid(1)
    if i < pos.shape[0]:
        ax, ay = 0.0, 0.0
        for j in range(pos.shape[0]):
            if i != j:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                inv_r3 = (dx*dx + dy*dy + 1e-6)**(-1.5)
                ax += mass[j] * dx * inv_r3
                ay += mass[j] * dy * inv_r3
        acc[i, 0] = ax
        acc[i, 1] = ay

# Initialize simulation parameters
n_bodies = 1000
dt = 0.01
steps = 100

# Initialize positions, velocities, and masses
pos = cp.random.uniform(-1, 1, (n_bodies, 2)).astype(cp.float32)
vel = cp.zeros((n_bodies, 2), dtype=cp.float32)
mass = cp.ones(n_bodies, dtype=cp.float32)
acc = cp.zeros_like(pos)

# Run simulation
for _ in range(steps):
    compute_acceleration[(n_bodies + 255) // 256, 256](pos, mass, acc)
    update_positions[(n_bodies + 255) // 256, 256](pos, vel, acc, dt)

# Plot final positions
plt.scatter(cp.asnumpy(pos[:, 0]), cp.asnumpy(pos[:, 1]), s=1)
plt.title("2D N-body Simulation")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.show()
```

Slide 13: Best Practices and Optimization Techniques

To get the most out of CUDA programming, it's important to follow best practices and apply optimization techniques.

```python
import cupy as cp
import time

# Use appropriate data types
float_array = cp.random.rand(10**6).astype(cp.float32)  # Use float32 instead of float64 when possible

# Minimize data transfer between CPU and GPU
def process_on_gpu(data):
    return cp.sum(cp.sin(data) + cp.cos(data))

start_time = time.time()
result = process_on_gpu(float_array)
gpu_time = time.time() - start_time

print(f"GPU processing time: {gpu_time:.4f} seconds")
print(f"Result: {result}")

# Use asynchronous operations when possible
stream = cp.cuda.Stream(non_blocking=True)
with stream:
    async_result = cp.sum(cp.sin(float_array) + cp.cos(float_array))

stream.synchronize()
print(f"Asynchronous result: {async_result}")
```

Slide 14: Debugging CUDA Code

Debugging CUDA code can be challenging. Here are some techniques to help identify and fix issues in your CUDA programs.

```python
import cupy as cp
from numba import cuda
import numpy as np

@cuda.jit
def buggy_kernel(arr):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        # Intentional bug: accessing out-of-bounds memory
        arr[i] = arr[i + 1]

# Enable CUDA debug mode
cp.cuda.runtime.setDevice(0)
cp.cuda.runtime.deviceSynchronize()

# Prepare data
data = cp.arange(1000, dtype=cp.float32)

# Run the kernel
threads_per_block = 256
blocks_per_grid = (data.size + threads_per_block - 1) // threads_per_block

try:
    buggy_kernel[blocks_per_grid, threads_per_block](data)
    cp.cuda.runtime.deviceSynchronize()
except cp.cuda.runtime.CUDARuntimeError as e:
    print(f"CUDA error detected: {e}")

# Use assert statements for debugging
@cuda.jit
def debug_kernel(arr):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        cuda.atomic.add(arr, 0, 1)
        cuda.syncthreads()
        if i == 0:
            assert arr[0] == arr.shape[0], "Incorrect atomic operation result"

debug_data = cp.zeros(1000, dtype=cp.int32)
debug_kernel[blocks_per_grid, threads_per_block](debug_data)
cp.cuda.runtime.deviceSynchronize()

print("Debugging complete")
```

Slide 15: Additional Resources

To further your knowledge of CUDA programming with Python, consider exploring these resources:

1. NVIDIA CUDA Documentation: [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)
2. CuPy Documentation: [https://docs.cupy.dev/](https://docs.cupy.dev/)
3. Numba CUDA Documentation: [https://numba.pydata.org/numba-doc/latest/cuda/index.html](https://numba.pydata.org/numba-doc/latest/cuda/index.html)
4. "GPU Programming in Python" by Andreas Klö

