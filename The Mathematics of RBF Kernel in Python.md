## The Mathematics of RBF Kernel in Python
Slide 1: Introduction to RBF Kernel

The Radial Basis Function (RBF) kernel is a popular kernel function used in various machine learning algorithms, particularly in Support Vector Machines (SVM) for classification and regression tasks. It measures the similarity between two points in a high-dimensional space.

```python
import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

# Example usage
x1 = np.array([1, 2])
x2 = np.array([3, 4])
gamma = 0.5

similarity = rbf_kernel(x1, x2, gamma)
print(f"Similarity between {x1} and {x2}: {similarity}")

# Visualize RBF kernel
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = rbf_kernel(np.array([X, Y]), np.array([0, 0]), gamma=0.1)

plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='Similarity')
plt.title('RBF Kernel Visualization')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
```

Slide 2: Mathematical Formula of RBF Kernel

The RBF kernel, also known as the Gaussian kernel, is defined by the following formula:

K(x, y) = exp(-γ ||x - y||^2)

Where x and y are two input vectors, γ (gamma) is a parameter that determines the kernel's width, and ||x - y|| is the Euclidean distance between x and y.

```python
import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(x, y, gamma):
    return np.exp(-gamma * np.linalg.norm(x - y)**2)

# Generate sample points
x = np.linspace(-5, 5, 100)
y = np.zeros_like(x)

# Calculate kernel values for different gamma
gammas = [0.1, 0.5, 1, 2]
for gamma in gammas:
    z = [rbf_kernel(np.array([xi, 0]), np.array([0, 0]), gamma) for xi in x]
    plt.plot(x, z, label=f'γ = {gamma}')

plt.title('RBF Kernel for Different γ Values')
plt.xlabel('x')
plt.ylabel('K(x, 0)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 3: Properties of RBF Kernel

The RBF kernel has several important properties that make it useful in machine learning:

1. It is symmetric: K(x, y) = K(y, x)
2. It is always positive: K(x, y) > 0
3. It decreases monotonically with distance
4. It has a maximum value of 1 when x = y

```python
import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(x, y, gamma):
    return np.exp(-gamma * np.linalg.norm(x - y)**2)

# Generate sample points
x = np.linspace(-5, 5, 100)
y = np.zeros_like(x)

# Calculate kernel values
gamma = 0.5
z = [rbf_kernel(np.array([xi, 0]), np.array([0, 0]), gamma) for xi in x]

# Plot kernel values
plt.figure(figsize=(10, 6))
plt.plot(x, z)
plt.title('RBF Kernel Properties')
plt.xlabel('Distance from origin')
plt.ylabel('Kernel value')
plt.axhline(y=1, color='r', linestyle='--', label='Maximum value')
plt.axvline(x=0, color='g', linestyle='--', label='x = y')
plt.legend()
plt.grid(True)
plt.show()

# Demonstrate symmetry
x1, x2 = np.array([1, 2]), np.array([3, 4])
print(f"K(x1, x2) = {rbf_kernel(x1, x2, gamma):.6f}")
print(f"K(x2, x1) = {rbf_kernel(x2, x1, gamma):.6f}")
```

Slide 4: Interpreting Gamma in RBF Kernel

The gamma parameter in the RBF kernel controls the influence of a single training example. Low gamma values mean far away points have a high influence, while high gamma values mean only nearby points have a high influence.

```python
import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(x, y, gamma):
    return np.exp(-gamma * np.linalg.norm(x - y)**2)

# Generate sample points
x = np.linspace(-5, 5, 100)
y = np.zeros_like(x)

# Calculate kernel values for different gamma
gammas = [0.1, 0.5, 1, 2]
plt.figure(figsize=(10, 6))
for gamma in gammas:
    z = [rbf_kernel(np.array([xi, 0]), np.array([0, 0]), gamma) for xi in x]
    plt.plot(x, z, label=f'γ = {gamma}')

plt.title('Effect of Gamma on RBF Kernel')
plt.xlabel('Distance from origin')
plt.ylabel('Kernel value')
plt.legend()
plt.grid(True)
plt.show()

# Print influence at specific distances
distances = [0.5, 1, 2]
for gamma in gammas:
    print(f"\nGamma = {gamma}")
    for d in distances:
        influence = rbf_kernel(np.array([d, 0]), np.array([0, 0]), gamma)
        print(f"  Influence at distance {d}: {influence:.4f}")
```

Slide 5: RBF Kernel in Feature Space

The RBF kernel implicitly maps input data into an infinite-dimensional feature space. This allows it to handle non-linear relationships in the original input space.

```python
import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel_matrix(X, gamma):
    sq_dists = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    return np.exp(-gamma * sq_dists)

# Generate sample data
np.random.seed(42)
X = np.random.rand(50, 2) * 4 - 2

# Compute kernel matrix
gamma = 0.5
K = rbf_kernel_matrix(X, gamma)

# Visualize original data and kernel matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1])
ax1.set_title('Original Data')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')

im = ax2.imshow(K, cmap='viridis')
ax2.set_title('RBF Kernel Matrix')
plt.colorbar(im, ax=ax2)

plt.tight_layout()
plt.show()

# Print dimensions
print(f"Original data shape: {X.shape}")
print(f"Kernel matrix shape: {K.shape}")
```

Slide 6: RBF Kernel in Support Vector Machines

The RBF kernel is widely used in Support Vector Machines (SVM) for non-linear classification. It allows the SVM to create complex decision boundaries in the original input space.

```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Generate non-linear data
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM with RBF kernel
svm = SVC(kernel='rbf', gamma=1, C=1)
svm.fit(X_train, y_train)

# Create a mesh to plot in
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict the function value for the whole grid
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the results
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM with RBF Kernel')
plt.show()

# Print accuracy
print(f"Accuracy on test set: {svm.score(X_test, y_test):.2f}")
```

Slide 7: Choosing Gamma in RBF Kernel

Selecting the right gamma value is crucial for the performance of RBF kernel-based models. Too small gamma can lead to underfitting, while too large gamma can cause overfitting.

```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Generate non-linear data
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM with different gamma values
gammas = [0.01, 0.1, 1, 10]
plt.figure(figsize=(15, 10))

for i, gamma in enumerate(gammas, 1):
    svm = SVC(kernel='rbf', gamma=gamma, C=1)
    svm.fit(X_train, y_train)
    
    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict the function value for the whole grid
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the results
    plt.subplot(2, 2, i)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.title(f'Gamma = {gamma}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
plt.tight_layout()
plt.show()

# Print accuracies
for gamma in gammas:
    svm = SVC(kernel='rbf', gamma=gamma, C=1)
    svm.fit(X_train, y_train)
    print(f"Gamma = {gamma}, Accuracy = {svm.score(X_test, y_test):.2f}")
```

Slide 8: RBF Kernel in Gaussian Process Regression

The RBF kernel is also used in Gaussian Process Regression as a covariance function. It defines how the function values at different points are related.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Generate sample data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Define and fit the Gaussian Process model
kernel = RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
gp.fit(X, y)

# Make predictions
X_pred = np.linspace(-1, 11, 100).reshape(-1, 1)
y_pred, sigma = gp.predict(X_pred, return_std=True)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(X_pred, y_pred, 'b-', label='Prediction')
plt.fill_between(X_pred.ravel(), y_pred - 1.96*sigma, y_pred + 1.96*sigma, 
                 alpha=0.2, color='b', label='95% confidence interval')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gaussian Process Regression with RBF Kernel')
plt.legend()
plt.show()

# Print kernel parameters
print(f"Optimized kernel parameters: {gp.kernel_}")
```

Slide 9: RBF Kernel in Spectral Clustering

The RBF kernel is used in spectral clustering to compute the similarity matrix between data points. This allows for clustering of non-linearly separable data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons

# Generate sample data
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# Perform spectral clustering
sc = SpectralClustering(n_clusters=2, affinity='rbf', gamma=10, random_state=42)
y_pred = sc.fit_predict(X)

# Plot the results
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title('Spectral Clustering Result')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Print cluster sizes
unique, counts = np.unique(y_pred, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"Cluster {cluster}: {count} points")
```

Slide 10: RBF Kernel in Dimensionality Reduction

The RBF kernel can be used in kernel PCA for non-linear dimensionality reduction. It helps to capture complex relationships in high-dimensional data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA

# Generate Swiss Roll dataset
X, color = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)

# Apply Kernel PCA with RBF kernel
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_kpca = kpca.fit_transform(X)

# Plot original data and KPCA result
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.viridis)
ax1.set_title("Original 3D Swiss Roll")

ax2 = fig.add_subplot(122)
ax2.scatter(X_kpca[:, 0], X_kpca[:, 1], c=color, cmap=plt.cm.viridis)
ax2.set_title("2D Projection using Kernel PCA with RBF Kernel")

plt.tight_layout()
plt.show()

# Calculate and print explained variance ratio
explained_variance = np.var(X_kpca, axis=0)
explained_variance_ratio = explained_variance / np.sum(explained_variance)
print(f"Explained variance ratio: {explained_variance_ratio}")
```

Slide 11: RBF Kernel in Image Processing

The RBF kernel is used in various image processing tasks, such as image denoising and edge detection. Here's an example of using the RBF kernel for image sharpening:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import data

def rbf_kernel_2d(size, sigma):
    x, y = np.meshgrid(np.arange(-size//2+1, size//2+1), np.arange(-size//2+1, size//2+1))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()

# Load sample image
image = data.camera()

# Create RBF kernel
kernel_size = 5
sigma = 1.0
kernel = rbf_kernel_2d(kernel_size, sigma)

# Apply sharpening
blurred = ndimage.convolve(image, kernel, mode='reflect')
sharpened = image + (image - blurred)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')

ax2.imshow(blurred, cmap='gray')
ax2.set_title('Blurred Image')
ax2.axis('off')

ax3.imshow(sharpened, cmap='gray')
ax3.set_title('Sharpened Image')
ax3.axis('off')

plt.tight_layout()
plt.show()

# Print kernel statistics
print(f"Kernel shape: {kernel.shape}")
print(f"Kernel sum: {kernel.sum():.6f}")
print(f"Kernel max value: {kernel.max():.6f}")
```

Slide 12: RBF Kernel in Time Series Analysis

The RBF kernel is useful in time series analysis, particularly for tasks like smoothing and forecasting. Here's an example of using the RBF kernel for time series smoothing:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Generate noisy time series data
np.random.seed(42)
t = np.linspace(0, 10, 100)
y = np.sin(t) + 0.1 * np.random.randn(100)

# Define and fit the Gaussian Process model with RBF kernel
kernel = RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
gp.fit(t.reshape(-1, 1), y)

# Make predictions for smoothing
t_pred = np.linspace(0, 10, 200).reshape(-1, 1)
y_pred, sigma = gp.predict(t_pred, return_std=True)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(t, y, 'r.', markersize=10, label='Original data')
plt.plot(t_pred, y_pred, 'b-', label='Smoothed data')
plt.fill_between(t_pred.ravel(), y_pred - 1.96*sigma, y_pred + 1.96*sigma, 
                 alpha=0.2, color='b', label='95% confidence interval')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Smoothing with RBF Kernel')
plt.legend()
plt.show()

# Print kernel parameters
print(f"Optimized kernel parameters: {gp.kernel_}")
```

Slide 13: RBF Kernel in Anomaly Detection

The RBF kernel is effective in anomaly detection tasks, particularly in One-Class SVM. It helps to identify data points that are significantly different from the majority.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs

# Generate sample data with outliers
X, _ = make_blobs(n_samples=200, centers=1, cluster_std=1, random_state=42)
X = np.r_[X, np.random.uniform(low=-6, high=6, size=(20, 2))]  # Add outliers

# Fit One-Class SVM with RBF kernel
svm = OneClassSVM(kernel='rbf', nu=0.1, gamma=0.1)
svm.fit(X)

# Predict anomalies
y_pred = svm.predict(X)

# Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='blue', label='Normal')
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='red', label='Anomaly')
plt.title('Anomaly Detection with One-Class SVM using RBF Kernel')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(-8, 8, 200), np.linspace(-8, 8, 200))
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='green')

plt.show()

# Print statistics
print(f"Number of normal points: {sum(y_pred == 1)}")
print(f"Number of anomalies: {sum(y_pred == -1)}")
```

Slide 14: Real-life Example: Handwritten Digit Recognition

The RBF kernel is widely used in handwritten digit recognition tasks. Here's an example using the MNIST dataset:

```python
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load MNIST dataset
digits = datasets.load_digits()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Train SVM with RBF kernel
clf = svm.SVC(kernel='rbf', gamma='scale')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Display some predictions
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax, image, prediction in zip(axes.ravel(), X_test, y_pred):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f"Prediction: {prediction}")

plt.tight_layout()
plt.show()

# Print classification report
print("Classification Report:")
print(metrics.classification_report(y_test, y_pred))
```

Slide 15: Real-life Example: Sentiment Analysis

The RBF kernel can be applied in text classification tasks like sentiment analysis. Here's a simplified example:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (you would typically use a larger dataset)
texts = [
    "I love this product, it's amazing!",
    "This is terrible, worst purchase ever.",
    "Neutral opinion, neither good nor bad.",
    "Absolutely fantastic experience!",
    "Disappointed with the quality."
]
labels = [1, 0, 2, 1, 0]  # 1: positive, 0: negative, 2: neutral

# Split the data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train SVM with RBF kernel
clf = SVC(kernel='rbf', gamma='scale')
clf.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = clf.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive', 'Neutral']))

# Test with new sentences
new_texts = [
    "I really enjoyed using this product.",
    "The service was awful and unprofessional.",
    "It's an okay product, nothing special."
]
new_texts_vectorized = vectorizer.transform(new_texts)
predictions = clf.predict(new_texts_vectorized)

print("\nNew Text Predictions:")
for text, prediction in zip(new_texts, predictions):
    sentiment = ['Negative', 'Positive', 'Neutral'][prediction]
    print(f"Text: '{text}'\nPredicted sentiment: {sentiment}\n")
```

Slide 16: Additional Resources

For those interested in diving deeper into the mathematics and applications of the RBF kernel, here are some valuable resources:

1. Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press. Available at: [http://www.gaussianprocess.org/gpml/](http://www.gaussianprocess.org/gpml/)
2. Schölkopf, B., & Smola, A. J. (2002). Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond. MIT Press.
3. Hofmann, T., Schölkopf, B., & Smola, A. J. (2008). Kernel methods in machine learning. The Annals of Statistics, 36(3), 1171-1220. ArXiv: [https://arxiv.org/abs/math/0701907](https://arxiv.org/abs/math/0701907)
4. Genton, M. G. (2001). Classes of kernels for machine learning: a statistics perspective. Journal of Machine Learning Research, 2(Dec), 299-312.

These resources provide in-depth explanations of kernel methods, including the RBF kernel, and their applications in various machine learning tasks.

