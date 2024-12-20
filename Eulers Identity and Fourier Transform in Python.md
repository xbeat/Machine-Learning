## Euler's Identity and Fourier Transform in Python
Slide 1: Introduction to Euler's Identity

Euler's Identity is a mathematical equation that connects five fundamental constants in a single, elegant expression. It's often considered one of the most beautiful equations in mathematics due to its simplicity and profound implications.

```python
import math

e = math.e
pi = math.pi
i = complex(0, 1)

result = e**(i * pi) + 1

print(f"e^(iπ) + 1 = {result}")
# Output: e^(iπ) + 1 = (1.2246467991473532e-16+0j)
# The small deviation from 0 is due to floating-point arithmetic limitations
```

Slide 2: Breaking Down Euler's Identity

Let's examine each component of Euler's Identity: e^(iπ) + 1 = 0. We'll explore the mathematical constants e, π, and i, and how they come together in this equation.

```python
import math

e = math.e
pi = math.pi
i = complex(0, 1)

print(f"e ≈ {e:.6f}")
print(f"π ≈ {pi:.6f}")
print(f"i = {i}")
```

Slide 3: Euler's Formula

Euler's Formula, e^(ix) = cos(x) + i\*sin(x), is the foundation of Euler's Identity. It relates complex exponentials to trigonometric functions.

```python
import math
import cmath

def euler_formula(x):
    return cmath.exp(1j * x)

x = math.pi / 4
result = euler_formula(x)
print(f"e^(i*π/4) = {result}")
print(f"cos(π/4) + i*sin(π/4) = {math.cos(x) + 1j*math.sin(x)}")
```

Slide 4: Visualizing Euler's Formula

Euler's Formula can be visualized as a point moving around the complex plane. As x increases, the point traces out a circle with radius 1 centered at the origin.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)
y = np.exp(1j * x)

plt.figure(figsize=(8, 8))
plt.plot(y.real, y.imag)
plt.title("Euler's Formula: e^(ix) on the Complex Plane")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid(True)
plt.axis('equal')
plt.show()
```

Slide 5: Deriving Euler's Identity

Euler's Identity is a special case of Euler's Formula when x = π. Let's see how it leads to the famous equation.

```python
import math
import cmath

pi = math.pi
result = cmath.exp(1j * pi)

print(f"e^(iπ) = {result}")
print(f"e^(iπ) + 1 = {result + 1}")
```

Slide 6: Applications in Signal Processing

Euler's Formula is fundamental in signal processing, particularly in the analysis of periodic signals. It forms the basis for Fourier analysis.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_signal(t, frequencies, amplitudes):
    return sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(frequencies, amplitudes))

t = np.linspace(0, 1, 1000)
frequencies = [3, 7, 11]
amplitudes = [1, 0.5, 0.3]

signal = generate_signal(t, frequencies, amplitudes)

plt.figure(figsize=(10, 6))
plt.plot(t, signal)
plt.title("Complex Periodic Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
```

Slide 7: Introduction to Fourier Transform

The Fourier Transform decomposes a signal into its constituent frequencies. It's a powerful tool in signal processing and has deep connections to Euler's Formula.

```python
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 1000)
signal = generate_signal(t, frequencies, amplitudes)

fft = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(t), t[1] - t[0])

plt.figure(figsize=(10, 6))
plt.plot(freqs, np.abs(fft))
plt.title("Frequency Spectrum")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.xlim(0, 20)
plt.grid(True)
plt.show()
```

Slide 8: Discrete Fourier Transform (DFT)

The Discrete Fourier Transform is the practical implementation of the Fourier Transform for digital signals. It's the foundation for many signal processing algorithms.

```python
import numpy as np

def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, x)

# Example usage
x = np.array([1, 2, 3, 4])
X = dft(x)
print("DFT result:", X)
```

Slide 9: Fast Fourier Transform (FFT)

The Fast Fourier Transform is an efficient algorithm for computing the DFT. It reduces the complexity from O(N^2) to O(N log N), making it practical for large datasets.

```python
import numpy as np
import time

N = 1024
x = np.random.random(N)

start = time.time()
X_dft = dft(x)
end = time.time()
print(f"DFT time: {end - start:.6f} seconds")

start = time.time()
X_fft = np.fft.fft(x)
end = time.time()
print(f"FFT time: {end - start:.6f} seconds")

print(f"Max difference: {np.max(np.abs(X_dft - X_fft))}")
```

Slide 10: Fourier Transform in Image Processing

The Fourier Transform can be applied to images, revealing information about their frequency content. This is useful in various image processing tasks.

```python
import numpy as np
import matplotlib.pyplot as plt

def create_image():
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) + np.cos(Y)
    return Z

image = create_image()
fft_image = np.fft.fft2(image)
fft_shift = np.fft.fftshift(fft_image)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.subplot(122)
plt.imshow(np.log(1 + np.abs(fft_shift)), cmap='gray')
plt.title("Fourier Transform")
plt.show()
```

Slide 11: Euler's Identity in Machine Learning

Euler's Identity and the concepts it embodies play a role in various machine learning techniques, particularly in areas involving complex numbers or periodic patterns.

```python
import numpy as np
import matplotlib.pyplot as plt

def complex_rbf(X, Y, sigma=1.0):
    dist = np.abs(X[:, np.newaxis] - Y)
    return np.exp(-0.5 * (dist / sigma) ** 2) * np.exp(1j * dist)

X = np.linspace(0, 10, 100)
Y = np.linspace(0, 10, 100)
K = complex_rbf(X, Y)

plt.figure(figsize=(10, 8))
plt.imshow(np.abs(K), cmap='viridis')
plt.title("Magnitude of Complex RBF Kernel")
plt.colorbar()
plt.show()
```

Slide 12: Complex-valued Neural Networks

Complex-valued neural networks use complex numbers in their weights and activations. They can be particularly useful for processing signals or data with phase information.

```python
import numpy as np

def complex_relu(z):
    return np.maximum(0, z.real) + 1j * np.maximum(0, z.imag)

class ComplexLayer:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) + 1j * np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size) + 1j * np.random.randn(output_size)
    
    def forward(self, x):
        return complex_relu(np.dot(x, self.W) + self.b)

# Example usage
layer = ComplexLayer(5, 3)
x = np.random.randn(10, 5) + 1j * np.random.randn(10, 5)
output = layer.forward(x)
print("Output shape:", output.shape)
print("Output dtype:", output.dtype)
```

Slide 13: Fourier Neural Operator

The Fourier Neural Operator is a type of neural network that leverages the Fourier Transform to learn mappings between function spaces. It's particularly useful for solving partial differential equations.

```python
import torch
import torch.nn as nn

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

# Example usage
conv = SpectralConv2d(3, 4, 5, 5)
x = torch.randn(2, 3, 32, 32)
output = conv(x)
print("Output shape:", output.shape)
```

Slide 14: Additional Resources

For those interested in diving deeper into these topics, here are some recommended resources from arXiv.org:

1. "A Survey on Complex-Valued Neural Networks" - [https://arxiv.org/abs/2101.12249](https://arxiv.org/abs/2101.12249)
2. "Fourier Neural Operator for Parametric Partial Differential Equations" - [https://arxiv.org/abs/2010.08895](https://arxiv.org/abs/2010.08895)
3. "On the Spectral Bias of Neural Networks" - [https://arxiv.org/abs/1806.08734](https://arxiv.org/abs/1806.08734)

These papers provide in-depth discussions on complex-valued neural networks, Fourier Neural Operators, and the relationship between neural networks and frequency domains.

