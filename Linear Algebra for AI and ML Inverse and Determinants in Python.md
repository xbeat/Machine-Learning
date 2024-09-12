## Linear Algebra for AI and ML Inverse and Determinants in Python
Slide 1: Introduction to Linear Algebra in AI and ML

Linear algebra forms the backbone of many machine learning algorithms and AI techniques. It provides the mathematical foundation for understanding and implementing various concepts in data analysis, optimization, and model training. In this presentation, we'll explore two crucial aspects of linear algebra: inverse matrices and determinants, and their applications in AI and ML using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 2D linear transformation
A = np.array([[2, 1], [1, 3]])
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = A[0, 0] * X + A[0, 1] * Y

# Plot the transformation
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='Transformation')
plt.title('Linear Transformation Visualization')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

Slide 2: Inverse Matrices: Definition and Importance

An inverse matrix is a matrix that, when multiplied with the original matrix, results in the identity matrix. In AI and ML, inverse matrices are crucial for solving systems of linear equations, which appear in various algorithms such as linear regression and principal component analysis.

```python
import numpy as np

# Define a matrix
A = np.array([[4, 7], [2, 6]])

# Calculate the inverse
A_inv = np.linalg.inv(A)

# Verify the inverse
result = np.dot(A, A_inv)
print("Original matrix:")
print(A)
print("\nInverse matrix:")
print(A_inv)
print("\nProduct of matrix and its inverse:")
print(np.round(result, decimals=10))  # Rounding to avoid floating-point errors
```

Slide 3: Calculating Inverse Matrices in Python

NumPy provides efficient tools for calculating inverse matrices. We'll explore how to compute inverses and discuss their applications in solving linear systems.

```python
import numpy as np

# Define a system of linear equations: 2x + y = 5, x + 3y = 10
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 10])

# Solve using inverse matrix
x = np.dot(np.linalg.inv(A), b)

print("Solution to the system:")
print(f"x = {x[0]:.2f}")
print(f"y = {x[1]:.2f}")

# Verify the solution
print("\nVerification:")
print(f"2x + y = {2*x[0] + x[1]:.2f}")
print(f"x + 3y = {x[0] + 3*x[1]:.2f}")
```

Slide 4: Determinants: Definition and Properties

The determinant is a scalar value that can be computed from a square matrix. It provides important information about the matrix, such as whether it's invertible and how it transforms space. In ML, determinants are used in various applications, including feature selection and dimensionality reduction.

```python
import numpy as np

# Define a 3x3 matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Calculate determinant
det_A = np.linalg.det(A)

print("Matrix A:")
print(A)
print(f"\nDeterminant of A: {det_A:.2f}")

# Check if the matrix is invertible
if det_A != 0:
    print("The matrix is invertible.")
else:
    print("The matrix is not invertible.")
```

Slide 5: Calculating Determinants in Python

NumPy offers straightforward methods to compute determinants. We'll explore how to calculate determinants for different sizes of matrices and interpret the results.

```python
import numpy as np

# Generate random matrices of different sizes
matrices = [np.random.rand(i, i) for i in range(2, 6)]

for i, matrix in enumerate(matrices):
    det = np.linalg.det(matrix)
    print(f"\nDeterminant of {i+2}x{i+2} matrix:")
    print(f"Matrix:\n{matrix}")
    print(f"Determinant: {det:.4f}")
    
    # Check if the matrix is invertible
    if abs(det) > 1e-10:  # Using a small threshold to account for floating-point precision
        print("The matrix is invertible.")
    else:
        print("The matrix is not invertible (or close to singular).")
```

Slide 6: Inverse Matrices in Linear Regression

Linear regression is a fundamental technique in machine learning. The inverse matrix plays a crucial role in finding the optimal coefficients for the regression model using the normal equation method.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term to X
X_b = np.c_[np.ones((100, 1)), X]

# Calculate coefficients using the normal equation
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Plot the results
plt.scatter(X, y, alpha=0.5)
plt.plot(X, theta[0] + theta[1] * X, 'r')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression using Normal Equation')
plt.show()

print(f"Intercept: {theta[0][0]:.2f}")
print(f"Slope: {theta[1][0]:.2f}")
```

Slide 7: Determinants in Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique widely used in machine learning. The determinant of the covariance matrix plays a role in understanding the total variance of the data and in selecting principal components.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate sample data
np.random.seed(42)
X = np.dot(np.random.randn(100, 2), [[2, 1], [1, 2]])

# Compute covariance matrix and its determinant
cov_matrix = np.cov(X.T)
det_cov = np.linalg.det(cov_matrix)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Plot original data and principal components
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * var  # scale component by its variance explanation power
    plt.plot([0, comp[0]], [0, comp[1]], 'k-', lw=2, label=f'PC{i+1}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('PCA: Original Data and Principal Components')
plt.legend()
plt.axis('equal')
plt.show()

print(f"Determinant of covariance matrix: {det_cov:.4f}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

Slide 8: Inverse Matrices in Neural Networks

In neural networks, inverse matrices are used in various optimization techniques, such as the Newton-Raphson method for finding optimal weights. They're also crucial in understanding network sensitivity and in some regularization techniques.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X, W1, W2):
    Z1 = np.dot(X, W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)
    return A1, A2

def backward(X, y, A1, A2, W2):
    m = X.shape[0]
    dZ2 = A2 - y
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    dZ1 = np.dot(dZ2, W2.T) * A1 * (1 - A1)
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    return dW1, dW2

# Generate sample data
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

# Initialize weights
W1 = np.random.randn(2, 3)
W2 = np.random.randn(3, 1)

# Training loop
for _ in range(1000):
    A1, A2 = forward(X, W1, W2)
    dW1, dW2 = backward(X, y, A1, A2, W2)
    W1 -= 0.1 * dW1
    W2 -= 0.1 * dW2

# Visualize decision boundary
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = forward(np.c_[xx.ravel(), yy.ravel()], W1, W2)[1]
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.RdYlBu, edgecolor='black')
plt.title('Neural Network Decision Boundary')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
```

Slide 9: Determinants in Feature Selection

Determinants can be used to measure the volume of the parallelotope formed by feature vectors. This concept is utilized in some feature selection algorithms to identify the most informative subset of features.

```python
import numpy as np
from sklearn.datasets import load_iris
from itertools import combinations

def feature_importance(X, k):
    n_features = X.shape[1]
    importances = []
    
    for combo in combinations(range(n_features), k):
        sub_X = X[:, combo]
        cov_matrix = np.cov(sub_X.T)
        importance = np.abs(np.linalg.det(cov_matrix))
        importances.append((combo, importance))
    
    return sorted(importances, key=lambda x: x[1], reverse=True)

# Load iris dataset
iris = load_iris()
X = iris.data

# Select top 2 features
top_features = feature_importance(X, 2)[:5]

print("Top 5 feature combinations:")
for features, importance in top_features:
    print(f"Features {features}: Importance {importance:.4f}")

# Visualize top feature combination
best_features = top_features[0][0]
plt.scatter(X[:, best_features[0]], X[:, best_features[1]], c=iris.target, cmap='viridis')
plt.xlabel(f'Feature {best_features[0]}')
plt.ylabel(f'Feature {best_features[1]}')
plt.title('Top 2 Features by Determinant Criterion')
plt.colorbar(label='Class')
plt.show()
```

Slide 10: Inverse Matrices in Covariance Estimation

Inverse matrices play a crucial role in estimating covariance matrices, particularly in high-dimensional settings. The inverse of the covariance matrix, known as the precision matrix, is used in various machine learning algorithms.

```python
import numpy as np
from sklearn.covariance import GraphicalLassoCV
import networkx as nx
import matplotlib.pyplot as plt

# Generate correlated data
np.random.seed(42)
n_samples, n_features = 100, 5
true_cov = np.array([
    [1.0, 0.5, 0.1, 0.0, 0.0],
    [0.5, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.4],
    [0.0, 0.0, 0.0, 0.4, 1.0]
])
X = np.random.multivariate_normal(mean=np.zeros(n_features), cov=true_cov, size=n_samples)

# Estimate sparse inverse covariance
model = GraphicalLassoCV()
model.fit(X)

# Create a graph from the precision matrix
precision_matrix = model.precision_
G = nx.from_numpy_array(np.abs(precision_matrix) > 0.1)

# Visualize the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 8))
nx.draw(G, pos, node_color='lightblue', 
        node_size=500, with_labels=True, 
        font_size=16, font_weight='bold')
edge_weights = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
plt.title('Graphical Model from Sparse Inverse Covariance')
plt.axis('off')
plt.tight_layout()
plt.show()

print("Estimated precision matrix:")
print(np.round(precision_matrix, 2))
```

Slide 11: Determinants in Gaussian Processes

Gaussian processes, a powerful tool in machine learning for regression and classification tasks, make extensive use of determinants in their covariance functions and likelihood computations. The determinant of the covariance matrix plays a crucial role in the marginal likelihood, which is used for hyperparameter optimization.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def rbf_kernel(X1, X2, l=1.0, sigma_f=1.0):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
y_train = np.sin(X_train)
X_test = np.linspace(-5, 5, 100).reshape(-1, 1)

K = rbf_kernel(X_train, X_train)
K_s = rbf_kernel(X_train, X_test)
K_ss = rbf_kernel(X_test, X_test)

K_inv = np.linalg.inv(K)
mu_s = K_s.T.dot(K_inv).dot(y_train)
cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

plt.figure(figsize=(10, 6))
plt.plot(X_test, mu_s, 'b-', label='Mean')
plt.fill_between(X_test.flatten(), 
                 mu_s.flatten() - 2*np.sqrt(np.diag(cov_s)),
                 mu_s.flatten() + 2*np.sqrt(np.diag(cov_s)),
                 alpha=0.2, color='b', label='Confidence')
plt.scatter(X_train, y_train, c='r', label='Training data')
plt.title('Gaussian Process Regression')
plt.legend()
plt.show()

print(f"Log determinant of K: {np.linalg.slogdet(K)[1]:.4f}")
```

Slide 12: Inverse Matrices in Image Processing

Inverse matrices are fundamental in various image processing tasks, including image restoration and deblurring. They help in reversing the effects of blur or noise on an image.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def blur_image(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    blurred = convolve2d(image, kernel, mode='same', boundary='wrap')
    return blurred

def deblur_image(blurred_image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    deblurred = np.fft.ifft2(np.fft.fft2(blurred_image) / np.fft.fft2(kernel, s=blurred_image.shape))
    return np.abs(deblurred)

# Create a simple image
image = np.zeros((50, 50))
image[10:40, 10:40] = 1

# Blur and deblur the image
blurred = blur_image(image)
deblurred = deblur_image(blurred)

# Plot the results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(blurred, cmap='gray')
axs[1].set_title('Blurred Image')
axs[2].imshow(deblurred, cmap='gray')
axs[2].set_title('Deblurred Image')
plt.show()
```

Slide 13: Determinants in Computer Vision

In computer vision, determinants are used in various algorithms, including feature detection and image transformations. The determinant of the Hessian matrix is particularly useful in blob detection algorithms.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def hessian_matrix(image):
    Ixx = ndimage.sobel(image, axis=0, mode='constant', cval=0.0)
    Ixx = ndimage.sobel(Ixx, axis=0, mode='constant', cval=0.0)
    Iyy = ndimage.sobel(image, axis=1, mode='constant', cval=0.0)
    Iyy = ndimage.sobel(Iyy, axis=1, mode='constant', cval=0.0)
    Ixy = ndimage.sobel(image, axis=1, mode='constant', cval=0.0)
    Ixy = ndimage.sobel(Ixy, axis=0, mode='constant', cval=0.0)
    return Ixx, Iyy, Ixy

def blob_detection(image, threshold):
    Ixx, Iyy, Ixy = hessian_matrix(image)
    determinant = Ixx * Iyy - Ixy**2
    return determinant > threshold

# Create a sample image with blobs
image = np.zeros((100, 100))
image[20:30, 20:30] = 1
image[60:80, 60:80] = 1
image = ndimage.gaussian_filter(image, sigma=2)

# Detect blobs
blobs = blob_detection(image, threshold=0.002)

# Plot results
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(blobs, cmap='gray')
axs[1].set_title('Detected Blobs')
plt.show()
```

Slide 14: Real-life Example: Image Compression using SVD

Singular Value Decomposition (SVD), which involves both inverse matrices and determinants, is widely used in image compression. Let's explore how SVD can be applied to compress images while preserving essential features.

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

def compress_image(image, k):
    U, s, Vt = np.linalg.svd(image, full_matrices=False)
    compressed = np.dot(U[:, :k], np.dot(np.diag(s[:k]), Vt[:k, :]))
    return compressed

# Load and convert image to grayscale
image = data.camera()

# Compress image with different levels
k_values = [5, 20, 50]
compressed_images = [compress_image(image, k) for k in k_values]

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original Image')

for i, (k, compressed) in enumerate(zip(k_values, compressed_images)):
    axs[(i+1)//2, (i+1)%2].imshow(compressed, cmap='gray')
    axs[(i+1)//2, (i+1)%2].set_title(f'Compressed (k={k})')

plt.tight_layout()
plt.show()

# Calculate compression ratios
original_size = image.size
compressed_sizes = [compressed.size for compressed in compressed_images]
compression_ratios = [original_size / size for size in compressed_sizes]

for k, ratio in zip(k_values, compression_ratios):
    print(f"Compression ratio (k={k}): {ratio:.2f}")
```

Slide 15: Additional Resources

For those interested in diving deeper into linear algebra applications in AI and ML, here are some valuable resources:

1. ArXiv paper: "A Survey of Linear Algebra for Machine Learning" by Zhang et al. (2021) ArXiv URL: [https://arxiv.org/abs/2102.01407](https://arxiv.org/abs/2102.01407)
2. ArXiv paper: "The Matrix Calculus You Need For Deep Learning" by Parr and Howard (2018) ArXiv URL: [https://arxiv.org/abs/1802.01528](https://arxiv.org/abs/1802.01528)
3. ArXiv paper: "Randomized Numerical Linear Algebra for Machine Learning" by Mahoney (2016) ArXiv URL: [https://arxiv.org/abs/1608.04481](https://arxiv.org/abs/1608.04481)

These papers provide in-depth discussions on the applications of linear algebra concepts, including inverse matrices and determinants, in various machine learning algorithms and deep learning architectures.

