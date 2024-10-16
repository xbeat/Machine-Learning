## Unsupervised Learning Concepts 101! Real-World Examples

Slide 1: Introduction to Unsupervised Learning

Unsupervised learning is a branch of machine learning where algorithms find patterns in data without explicit labels. It's like being a detective, uncovering hidden structures in complex datasets. Unlike supervised learning, where we train models on labeled data, unsupervised learning works with unlabeled data, making it powerful for exploratory data analysis and pattern discovery.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate random data
np.random.seed(42)
X = np.random.randn(300, 2)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('K-means Clustering Example')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

Slide 2: The Expectation-Maximization (EM) Algorithm

The EM algorithm is an iterative method for finding maximum likelihood estimates of parameters in statistical models. It's particularly useful when dealing with latent variables or missing data. The algorithm alternates between two steps: the Expectation step (E-step) and the Maximization step (M-step).

```python
import numpy as np
from scipy.stats import norm

def em_algorithm(data, n_components, n_iterations):
    # Initialize parameters
    weights = np.ones(n_components) / n_components
    means = np.random.choice(data, n_components)
    variances = np.ones(n_components)

    for _ in range(n_iterations):
        # E-step: Compute responsibilities
        responsibilities = np.array([w * norm.pdf(data, m, np.sqrt(v))
                                     for w, m, v in zip(weights, means, variances)])
        responsibilities /= responsibilities.sum(axis=0)

        # M-step: Update parameters
        weights = responsibilities.sum(axis=1) / len(data)
        means = np.sum(responsibilities * data, axis=1) / responsibilities.sum(axis=1)
        variances = np.sum(responsibilities * (data - means[:, np.newaxis])**2, axis=1) / responsibilities.sum(axis=1)

    return weights, means, variances

# Example usage
data = np.concatenate([np.random.normal(0, 1, 300), np.random.normal(5, 1.5, 200)])
weights, means, variances = em_algorithm(data, n_components=2, n_iterations=100)
print(f"Estimated means: {means}")
print(f"Estimated variances: {variances}")
print(f"Estimated weights: {weights}")
```

Slide 3: Hierarchical Clustering

Hierarchical clustering builds a tree-like structure of clusters, allowing for different levels of granularity in data analysis. It can be either agglomerative (bottom-up) or divisive (top-down). This method is particularly useful when the number of clusters is unknown or when you want to explore the hierarchical structure of your data.

```python
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(50, 2)

# Perform hierarchical clustering
Z = linkage(X, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

Slide 4: Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique that identifies the principal components (directions of maximum variance) in high-dimensional data. It's widely used for feature extraction, data compression, and visualization of high-dimensional datasets. PCA helps in reducing the complexity of data while retaining most of its important information.

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(200, 3)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Plot original and reduced data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], c=X[:, 2], cmap='viridis')
ax1.set_title('Original 3D Data (2D Projection)')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], c=X[:, 2], cmap='viridis')
ax2.set_title('PCA Reduced 2D Data')
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')

plt.tight_layout()
plt.show()

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

Slide 5: Independent Component Analysis (ICA)

ICA is a computational method for separating a multivariate signal into additive subcomponents, assuming the mutual statistical independence of the non-Gaussian source signals. It's particularly useful in signal processing and data analysis where mixed signals need to be separated into their original sources.

```python
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# Generate mixed signals
np.random.seed(42)
t = np.linspace(0, 10, 1000)
s1 = np.sin(2 * t)
s2 = np.sign(np.sin(3 * t))
S = np.c_[s1, s2]
A = np.array([[1, 0.5], [0.5, 1]])
X = np.dot(S, A.T)

# Apply ICA
ica = FastICA(n_components=2)
S_ = ica.fit_transform(X)

# Plot original and separated signals
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
axs[0].plot(X)
axs[0].set_title('Mixed Signals')
axs[1].plot(S)
axs[1].set_title('Original Signals')
axs[2].plot(S_)
axs[2].set_title('ICA Recovered Signals')

for ax in axs:
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

plt.tight_layout()
plt.show()
```

Slide 6: Real-Life Example: Image Compression with PCA

PCA can be used for image compression by reducing the dimensionality of image data. This technique is particularly useful for large datasets of images, such as in satellite imagery or medical imaging, where storage and transmission of high-resolution images can be challenging.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage import data

# Load sample image
image = data.camera()

# Reshape image to 2D array
X = image.reshape(-1, image.shape[1])

# Apply PCA with different numbers of components
n_components = [10, 50, 100, 200]
reconstructed_images = []

for n in n_components:
    pca = PCA(n_components=n)
    X_reduced = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_reduced)
    reconstructed_images.append(X_reconstructed.reshape(image.shape))

# Plot original and reconstructed images
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original Image')

for i, (n, img) in enumerate(zip(n_components, reconstructed_images)):
    row, col = divmod(i + 1, 3)
    axs[row, col].imshow(img, cmap='gray')
    axs[row, col].set_title(f'{n} Components')

for ax in axs.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()
```

Slide 7: Real-Life Example: Customer Segmentation with K-means

K-means clustering is widely used in customer segmentation, helping businesses understand their customer base and tailor their marketing strategies. This example demonstrates how to segment customers based on their annual income and spending score.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate sample customer data
np.random.seed(42)
n_customers = 200
annual_income = np.random.normal(50000, 15000, n_customers)
spending_score = np.random.normal(50, 25, n_customers)

# Combine features and scale
X = np.column_stack((annual_income, spending_score))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Plot the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(annual_income, spending_score, c=clusters, cmap='viridis')
plt.colorbar(scatter)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation using K-means Clustering')
plt.show()

# Print cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i+1} center: Annual Income: ${center[0]:.2f}, Spending Score: {center[1]:.2f}")
```

Slide 8: Gaussian Mixture Models (GMM)

Gaussian Mixture Models are probabilistic models that assume data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. GMMs are more flexible than K-means as they allow for clusters of different sizes and shapes.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Generate sample data
np.random.seed(42)
n_samples = 300
X = np.concatenate([
    np.random.normal(0, 1, (n_samples, 2)),
    np.random.normal(4, 1.5, (n_samples, 2)),
    np.random.normal(-4, 0.5, (n_samples, 2))
])

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Generate mesh grid
x, y = np.meshgrid(np.linspace(-8, 8, 200), np.linspace(-8, 8, 200))
XX = np.array([x.ravel(), y.ravel()]).T

# Predict probabilities on mesh grid
Z = -gmm.score_samples(XX)
Z = Z.reshape(x.shape)

# Plot results
plt.figure(figsize=(10, 8))
plt.contourf(x, y, Z, levels=20, cmap='viridis', alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c='white', edgecolor='black', alpha=0.5)
plt.title('Gaussian Mixture Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Negative log-likelihood')
plt.show()

print(f"Means of the Gaussian components:\n{gmm.means_}")
print(f"Covariances of the Gaussian components:\n{gmm.covariances_}")
```

Slide 9: t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is a nonlinear dimensionality reduction technique particularly well-suited for visualizing high-dimensional data. It excels at preserving local structures in the data, making it useful for exploring complex datasets and identifying clusters or patterns that might not be apparent in other visualization methods.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
plt.colorbar(scatter)
plt.title('t-SNE visualization of the digits dataset')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()

print(f"Original data shape: {X.shape}")
print(f"t-SNE reduced data shape: {X_tsne.shape}")
```

Slide 10: DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN is a density-based clustering algorithm that groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions. It's particularly useful for datasets with clusters of arbitrary shape and for identifying noise points.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Generate sample data
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(X)

# Plot the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.colorbar(scatter)
plt.title('DBSCAN Clustering on Moons Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

print(f"Number of clusters found: {len(set(clusters)) - (1 if -1 in clusters else 0)}")
print(f"Number of noise points: {list(clusters).count(-1)}")
```

Slide 11: Self-Organizing Maps (SOM)

Self-Organizing Maps are a type of artificial neural network used for dimensionality reduction and visualization of high-dimensional data. SOMs create a low-dimensional representation of the input space, preserving topological properties of the data.

```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleSOM:
    def __init__(self, m, n, dim, learning_rate=0.5):
        self.m, self.n, self.dim = m, n, dim
        self.weights = np.random.rand(m, n, dim)
        self.learning_rate = learning_rate

    def find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def train(self, data, epochs):
        for _ in range(epochs):
            for x in data:
                bmu = self.find_bmu(x)
                self.weights[bmu] += self.learning_rate * (x - self.weights[bmu])

# Generate sample data and train SOM
np.random.seed(42)
data = np.random.rand(1000, 3)
som = SimpleSOM(10, 10, 3)
som.train(data, 100)

# Visualize SOM
plt.figure(figsize=(10, 10))
for i in range(som.m):
    for j in range(som.n):
        plt.scatter(i, j, c=som.weights[i, j], s=50)
plt.title('Self-Organizing Map')
plt.show()
```

Slide 12: Autoencoders for Dimensionality Reduction

Autoencoders are neural networks designed to learn efficient data representations (encoding) typically for dimensionality reduction. They consist of an encoder that compresses the input and a decoder that attempts to reconstruct the input from the compressed representation.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = np.random.rand(1000, 10)

# Define autoencoder model
input_dim = 10
encoding_dim = 2

input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = models.Model(input_layer, decoded)
encoder = models.Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(data, data, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Encode the data
encoded_data = encoder.predict(data)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(encoded_data[:, 0], encoded_data[:, 1])
plt.title('Encoded Data')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

plt.tight_layout()
plt.show()
```

Slide 13: One-Class SVM for Anomaly Detection

One-Class SVM is an unsupervised algorithm that learns a decision function for novelty detection: classifying new data as similar or different to the training set. It's particularly useful for anomaly detection in scenarios where normal behavior is well-defined but anomalies are diverse and difficult to characterize.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs

# Generate normal data and anomalies
np.random.seed(42)
normal_data, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.5)
anomalies = np.random.uniform(low=-4, high=4, size=(30, 2))

# Combine normal data and anomalies
X = np.vstack([normal_data, anomalies])

# Fit One-Class SVM
svm = OneClassSVM(kernel='rbf', nu=0.1)
svm.fit(normal_data)

# Predict on all data
y_pred = svm.predict(X)

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='blue', label='Normal')
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='red', label='Anomaly')
plt.title('One-Class SVM for Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

print(f"Number of detected anomalies: {sum(y_pred == -1)}")
```

Slide 14: Additional Resources

For those interested in diving deeper into unsupervised learning concepts, here are some valuable resources:

1. "Unsupervised Learning: Foundations of Neural Computation" by Geoffrey Hinton and Terrence J. Sejnowski ArXiv: [https://arxiv.org/abs/cs/9805016](https://arxiv.org/abs/cs/9805016)
2. "A Tutorial on Spectral Clustering" by Ulrike von Luxburg ArXiv: [https://arxiv.org/abs/0711.0189](https://arxiv.org/abs/0711.0189)
3. "Reducing the Dimensionality of Data with Neural Networks" by G. E. Hinton and R. R. Salakhutdinov ArXiv: [https://arxiv.org/abs/1601.00670](https://arxiv.org/abs/1601.00670)

These papers provide in-depth explanations and mathematical foundations of various unsupervised learning techniques. They are excellent starting points for understanding the theoretical aspects behind the algorithms we've discussed in this presentation.

