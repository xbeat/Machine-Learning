## Scalar, Vector, Matrix, and Tensor The Foundations of Data Science in Python
Slide 1: The Quartet of Data Science: Scalar, Vector, Matrix, and Tensor

These fundamental mathematical structures form the backbone of modern data science, enabling complex computations and representations in Python. We'll explore each concept, their relationships, and practical applications in data analysis and machine learning.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple visualization of the quartet
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].text(0.5, 0.5, 'Scalar', ha='center', va='center', fontsize=20)
axs[0, 1].plot([1, 2, 3], [4, 5, 6])
axs[0, 1].set_title('Vector')
axs[1, 0].imshow(np.random.rand(5, 5), cmap='viridis')
axs[1, 0].set_title('Matrix')
axs[1, 1].text(0.5, 0.5, 'Tensor', ha='center', va='center', fontsize=20)
plt.tight_layout()
plt.show()
```

Slide 2: Scalar: The Building Block

A scalar is a single numerical value, representing a magnitude without direction. In Python, scalars are typically represented by simple numeric types like integers or floats. They form the foundation for more complex data structures.

```python
# Scalar examples
temperature = 25.5  # Temperature in Celsius
count = 100  # Number of items

# Basic operations with scalars
fahrenheit = (temperature * 9/5) + 32
double_count = count * 2

print(f"Temperature: {temperature}°C = {fahrenheit}°F")
print(f"Count: {count}, Doubled: {double_count}")
```

Slide 3: Vector: One-Dimensional Arrays

Vectors are one-dimensional arrays of scalars, representing quantities with both magnitude and direction. In Python, we often use NumPy arrays to work with vectors efficiently.

```python
import numpy as np

# Create a vector
v = np.array([1, 2, 3, 4, 5])

# Basic vector operations
magnitude = np.linalg.norm(v)
normalized = v / magnitude

print(f"Vector: {v}")
print(f"Magnitude: {magnitude:.2f}")
print(f"Normalized: {normalized}")

# Dot product of two vectors
u = np.array([2, 3, 4, 5, 6])
dot_product = np.dot(v, u)
print(f"Dot product of {v} and {u}: {dot_product}")
```

Slide 4: Matrix: Two-Dimensional Arrays

Matrices are two-dimensional arrays of scalars, organized in rows and columns. They are fundamental in linear algebra and form the basis for many data science algorithms.

```python
import numpy as np

# Create a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Matrix operations
B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

# Matrix addition
C = A + B

# Matrix multiplication
D = np.dot(A, B)

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)
print("\nA + B:")
print(C)
print("\nA · B:")
print(D)
```

Slide 5: Tensor: Multi-Dimensional Arrays

Tensors are generalizations of vectors and matrices to higher dimensions. They are crucial in deep learning and complex data representation. In Python, we can use NumPy or specialized libraries like TensorFlow or PyTorch to work with tensors.

```python
import numpy as np

# Create a 3D tensor (3x3x3)
T = np.array([[[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]],
              [[10, 11, 12],
               [13, 14, 15],
               [16, 17, 18]],
              [[19, 20, 21],
               [22, 23, 24],
               [25, 26, 27]]])

print("3D Tensor shape:", T.shape)
print("First 2D slice of the tensor:")
print(T[0])

# Tensor operations
sum_along_axis0 = np.sum(T, axis=0)
print("\nSum along axis 0:")
print(sum_along_axis0)
```

Slide 6: Scalar Operations: Beyond Basic Arithmetic

Scalars in Python can be used in various mathematical operations, including trigonometry, exponentiation, and logarithms. These operations are essential in many scientific and engineering applications.

```python
import math

angle_degrees = 45
angle_radians = math.radians(angle_degrees)

# Trigonometric functions
sin_value = math.sin(angle_radians)
cos_value = math.cos(angle_radians)

# Exponentiation and logarithms
base = 2
exponent = 3
power_result = math.pow(base, exponent)
log_result = math.log(power_result, base)

print(f"Sin({angle_degrees}°) = {sin_value:.4f}")
print(f"Cos({angle_degrees}°) = {cos_value:.4f}")
print(f"{base}^{exponent} = {power_result}")
print(f"log_{base}({power_result}) = {log_result}")
```

Slide 7: Vector Operations: Geometric Transformations

Vectors are powerful tools for representing and manipulating geometric data. We can use them to perform translations, rotations, and scaling operations in 2D and 3D space.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a 2D vector
v = np.array([3, 2])

# Translation
translation = np.array([2, 1])
v_translated = v + translation

# Rotation (45 degrees counterclockwise)
theta = np.radians(45)
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
v_rotated = np.dot(rotation_matrix, v)

# Scaling
scale_factor = 2
v_scaled = v * scale_factor

# Plotting
plt.figure(figsize=(10, 10))
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label='Original')
plt.quiver(0, 0, v_translated[0], v_translated[1], angles='xy', scale_units='xy', scale=1, color='g', label='Translated')
plt.quiver(0, 0, v_rotated[0], v_rotated[1], angles='xy', scale_units='xy', scale=1, color='b', label='Rotated')
plt.quiver(0, 0, v_scaled[0], v_scaled[1], angles='xy', scale_units='xy', scale=1, color='m', label='Scaled')
plt.xlim(-1, 8)
plt.ylim(-1, 8)
plt.legend()
plt.grid(True)
plt.show()
```

Slide 8: Matrix Applications: Image Processing

Matrices are extensively used in image processing. We can represent images as 2D matrices and apply various transformations to manipulate them.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 5x5 grayscale image
image = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                  [0.2, 0.3, 0.4, 0.5, 0.6],
                  [0.3, 0.4, 0.5, 0.6, 0.7],
                  [0.4, 0.5, 0.6, 0.7, 0.8],
                  [0.5, 0.6, 0.7, 0.8, 0.9]])

# Define a blur kernel
blur_kernel = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]]) / 16

# Apply convolution for blurring
blurred_image = np.zeros_like(image)
for i in range(1, image.shape[0]-1):
    for j in range(1, image.shape[1]-1):
        blurred_image[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * blur_kernel)

# Plot original and blurred images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(blurred_image, cmap='gray')
ax2.set_title('Blurred Image')
plt.show()
```

Slide 9: Tensor Operations: Color Image Processing

Tensors allow us to work with multi-dimensional data, such as color images. We can use 3D tensors to represent and manipulate RGB images.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 5x5x3 RGB image
rgb_image = np.zeros((5, 5, 3))
rgb_image[:, :, 0] = np.linspace(0, 1, 25).reshape(5, 5)  # Red channel
rgb_image[:, :, 1] = np.linspace(0, 1, 25).reshape(5, 5)[::-1]  # Green channel
rgb_image[:, :, 2] = np.linspace(0, 1, 25).reshape(5, 5).T  # Blue channel

# Increase brightness
brightened_image = np.clip(rgb_image * 1.5, 0, 1)

# Convert to grayscale
grayscale_image = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])

# Plot images
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(rgb_image)
ax1.set_title('Original RGB Image')
ax2.imshow(brightened_image)
ax2.set_title('Brightened Image')
ax3.imshow(grayscale_image, cmap='gray')
ax3.set_title('Grayscale Image')
plt.show()
```

Slide 10: Real-Life Example: Weather Data Analysis

Scalars, vectors, and matrices can be used to analyze weather data. We'll demonstrate how to work with temperature data for multiple cities over time.

```python
import numpy as np
import matplotlib.pyplot as plt

# Weather data: temperatures for 5 cities over 7 days
weather_data = np.array([
    [20, 22, 23, 19, 21, 24, 22],  # City 1
    [18, 20, 19, 21, 23, 24, 22],  # City 2
    [22, 21, 23, 24, 25, 23, 21],  # City 3
    [17, 18, 20, 21, 22, 20, 19],  # City 4
    [19, 20, 22, 23, 21, 20, 18]   # City 5
])

# Calculate average temperature for each city
avg_temp = np.mean(weather_data, axis=1)

# Find the hottest day for each city
hottest_day = np.argmax(weather_data, axis=1)

# Plot the data
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(weather_data[i], label=f'City {i+1}')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.title('Weekly Temperature Data for 5 Cities')
plt.legend()
plt.grid(True)
plt.show()

print("Average temperatures:")
for i, temp in enumerate(avg_temp):
    print(f"City {i+1}: {temp:.1f}°C")

print("\nHottest day for each city:")
for i, day in enumerate(hottest_day):
    print(f"City {i+1}: Day {day+1}")
```

Slide 11: Real-Life Example: Image Compression using SVD

Singular Value Decomposition (SVD) is a matrix factorization technique that can be used for image compression. We'll demonstrate how to use SVD to compress a grayscale image.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

# Load a sample image and convert to grayscale
image = load_sample_image("china.jpg")
gray_image = np.mean(image, axis=2).astype(np.float64)

# Perform SVD
U, s, Vt = np.linalg.svd(gray_image, full_matrices=False)

# Function to reconstruct image with k singular values
def reconstruct_image(U, s, Vt, k):
    return np.matrix(U[:, :k]) * np.diag(s[:k]) * np.matrix(Vt[:k, :])

# Reconstruct images with different numbers of singular values
k_values = [5, 20, 50, 200]
reconstructed_images = [reconstruct_image(U, s, Vt, k) for k in k_values]

# Plot original and reconstructed images
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(gray_image, cmap='gray')
axes[0, 0].set_title("Original Image")

for i, (k, img) in enumerate(zip(k_values, reconstructed_images), 1):
    ax = axes[i // 3, i % 3]
    ax.imshow(img, cmap='gray')
    ax.set_title(f"k = {k}")
    compression_ratio = (k * (U.shape[0] + Vt.shape[1] + 1)) / (U.shape[0] * Vt.shape[1])
    ax.set_xlabel(f"Compression Ratio: {compression_ratio:.2%}")

plt.tight_layout()
plt.show()
```

Slide 12: Tensors in Machine Learning: Neural Networks

Tensors are fundamental in deep learning, particularly in neural networks. We'll create a simple feedforward neural network to demonstrate how tensors are used in this context.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feedforward(input_data, weights, biases):
    # Input layer to hidden layer
    hidden = sigmoid(np.dot(input_data, weights[0]) + biases[0])
    # Hidden layer to output layer
    output = sigmoid(np.dot(hidden, weights[1]) + biases[1])
    return output

# Define network architecture
input_size = 3
hidden_size = 4
output_size = 2

# Initialize random weights and biases
np.random.seed(42)
weights = [
    np.random.randn(input_size, hidden_size),
    np.random.randn(hidden_size, output_size)
]
biases = [
    np.random.randn(hidden_size),
    np.random.randn(output_size)
]

# Create a batch of input data
batch_size = 5
input_data = np.random.randn(batch_size, input_size)

# Perform feedforward pass
output = feedforward(input_data, weights, biases)

print("Input shape:", input_data.shape)
print("Output shape:", output.shape)
print("\nSample input:")
print(input_data[0])
print("\nCorresponding output:")
print(output[0])
```

Slide 13: The Power of the Quartet in Data Science

The interplay between scalars, vectors, matrices, and tensors forms the foundation of numerous data science algorithms. This synergy enables complex computations and representations crucial for advanced analytics and machine learning.

```python
import numpy as np

# Scalar: Simple statistic
data = np.array([1, 2, 3, 4, 5])
mean = np.mean(data)

# Vector: Feature representation
features = np.array([height, weight, age])

# Matrix: Dataset of features
dataset = np.array([
    [170, 70, 30],
    [165, 65, 25],
    [180, 80, 35]
])

# Tensor: Time series of image data
image_series = np.random.rand(10, 64, 64, 3)  # 10 RGB images of 64x64 pixels

print(f"Scalar (mean): {mean}")
print(f"Vector (features): {features}")
print("Matrix (dataset):")
print(dataset)
print(f"Tensor (image series) shape: {image_series.shape}")
```

Slide 14: Practical Application: Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique that leverages the power of matrices and eigenvalue decomposition. It's widely used in data preprocessing and feature extraction.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate sample data
np.random.seed(42)
n_samples = 300
X = np.dot(np.random.randn(n_samples, 2), [[2, 1], [1, 3]])

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot original and transformed data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], alpha=0.7)
ax1.set_title('Original Data')

ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
ax2.set_title('PCA Transformed Data')

plt.tight_layout()
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Slide 15: Future Directions and Advanced Topics

The quartet of scalar, vector, matrix, and tensor computations continues to evolve, driving innovations in data science and machine learning. Some advanced topics include:

1. Tensor Networks: Used in quantum computing and complex system modeling
2. Differential Geometry: Applying tensor calculus to machine learning
3. Quantum Tensors: Representing quantum states and operations
4. Tensor Decompositions: Advanced techniques for multi-dimensional data analysis

These topics showcase the ongoing research and development in leveraging these mathematical structures for cutting-edge applications in data science and beyond.

Slide 16: Additional Resources

For those interested in diving deeper into these topics, here are some valuable resources:

1. ArXiv paper on Tensor Networks: "Tensor Networks for Big Data Analytics and Large-Scale Optimization Problems" (arXiv:1407.3124)
2. ArXiv paper on Differential Geometry in Machine Learning: "Riemannian Geometry in Machine Learning" (arXiv:2011.01538)
3. ArXiv paper on Quantum Tensors: "Quantum Tensor Networks: A Pathway to Machine Learning" (arXiv:1803.11537)
4. ArXiv paper on Tensor Decompositions: "Tensor Decompositions and Applications" (arXiv:0905.0454)

These papers provide in-depth discussions on advanced applications of tensors and related concepts in various fields of data science and quantum computing.

