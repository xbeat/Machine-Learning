## Unsupervised Learning Techniques for Data Exploration
Slide 1: K-Means Clustering Implementation from Scratch

K-means clustering partitions data points into distinct groups by iteratively assigning points to the nearest centroid and updating centroids based on mean cluster values. This fundamental algorithm forms the basis for many advanced clustering techniques.

```python
import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        
    def fit(self, X):
        # Randomly initialize centroids
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[self.labels == k].mean(axis=0) 
                                    for k in range(self.k)])
            
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids
            
        return self.labels
    
# Generate sample data
np.random.seed(42)
X = np.concatenate([
    np.random.normal(0, 1, (100, 2)),
    np.random.normal(4, 1, (100, 2)),
    np.random.normal(8, 1, (100, 2))
])

# Fit and plot
kmeans = KMeansClustering(k=3)
labels = kmeans.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
           marker='x', s=200, linewidths=3, color='r')
plt.title('K-Means Clustering Results')
plt.show()
```

Slide 2: Principal Component Analysis Implementation

Principal Component Analysis (PCA) reduces data dimensionality by projecting it onto principal components that capture maximum variance. This implementation demonstrates the mathematical foundations using eigendecomposition of the covariance matrix.

```python
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvectors & eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, idx[:self.n_components]]
        
        # Calculate explained variance ratio
        self.explained_variance_ratio_ = eigenvalues[idx][:self.n_components] / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

# Example usage
X = np.random.randn(100, 5)
pca = PCA(n_components=2)
X_transformed = pca.fit(X).transform(X)

print("Original shape:", X.shape)
print("Transformed shape:", X_transformed.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Slide 3: DBSCAN Clustering Implementation

DBSCAN identifies clusters based on density, capable of discovering clusters of arbitrary shapes and automatically detecting noise points. This implementation showcases the core algorithm logic including neighborhood computation and cluster expansion.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        
    def fit_predict(self, X):
        # Find neighbors for all points
        neighbors = NearestNeighbors(radius=self.eps).fit(X)
        distances, indices = neighbors.radius_neighbors(X)
        
        labels = np.full(len(X), -1)  # Initialize all points as noise
        current_cluster = 0
        
        # Iterate through each point
        for i in range(len(X)):
            if labels[i] != -1:  # Skip if already processed
                continue
                
            if len(indices[i]) >= self.min_samples:  # Check core point
                labels[i] = current_cluster
                # Expand cluster
                stack = list(indices[i])
                while stack:
                    neighbor = stack.pop()
                    if labels[neighbor] == -1:
                        labels[neighbor] = current_cluster
                        if len(indices[neighbor]) >= self.min_samples:
                            stack.extend(indices[neighbor])
                current_cluster += 1
                
        return labels

# Example usage with synthetic data
X = np.concatenate([
    np.random.normal(0, 0.5, (100, 2)),
    np.random.normal(4, 0.5, (100, 2))
])

dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title('DBSCAN Clustering Results')
plt.show()
```

Slide 4: t-SNE Implementation Fundamentals

t-SNE (t-Distributed Stochastic Neighbor Embedding) reduces dimensionality while preserving local structure, making it particularly effective for visualization. This implementation covers the core concepts of probability distribution computation and gradient descent.

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        
    def _compute_pairwise_dist(self, X):
        return squareform(pdist(X, 'euclidean'))
    
    def _compute_joint_probabilities(self, distances):
        # Compute conditional probabilities
        P = np.zeros((distances.shape[0], distances.shape[0]))
        beta = np.ones(distances.shape[0])
        
        # Binary search for sigma (beta)
        target = np.log(self.perplexity)
        for i in range(distances.shape[0]):
            beta[i] = self._binary_search(distances[i], target)
            P[i] = np.exp(-distances[i] * beta[i])
            P[i, i] = 0
        
        # Symmetrize and normalize
        P = (P + P.T) / (2 * distances.shape[0])
        P = np.maximum(P, 1e-12)
        
        return P
    
    def _binary_search(self, dist, target):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0
        max_iter = 50
        
        for _ in range(max_iter):
            sum_P = np.sum(np.exp(-dist * beta))
            H = np.log(sum_P) + beta * np.sum(dist * np.exp(-dist * beta)) / sum_P
            if abs(H - target) < 1e-5:
                break
            if H > target:
                beta_min = beta
                beta = (beta + beta_max) / 2.0 if beta_max != np.inf else beta * 2
            else:
                beta_max = beta
                beta = (beta + beta_min) / 2.0 if beta_min != -np.inf else beta / 2
                
        return beta

    def fit_transform(self, X):
        # Initialize solution
        np.random.seed(42)
        Y = np.random.randn(X.shape[0], self.n_components) * 1e-4
        
        # Compute pairwise affinities
        distances = self._compute_pairwise_dist(X)
        P = self._compute_joint_probabilities(distances)
        
        # Gradient descent
        for iter in range(self.n_iter):
            # Compute Q distribution
            dist_Y = self._compute_pairwise_dist(Y)
            Q = 1 / (1 + dist_Y)
            np.fill_diagonal(Q, 0)
            Q = Q / np.sum(Q)
            Q = np.maximum(Q, 1e-12)
            
            # Compute gradients
            PQ = P - Q
            grad = np.zeros_like(Y)
            for i in range(Y.shape[0]):
                grad[i] = 4 * np.sum(np.tile(PQ[:, i] * Q[:, i], (self.n_components, 1)).T * 
                                   (Y[i] - Y), axis=0)
            
            # Update Y
            Y = Y - self.learning_rate * grad
            
            if iter % 100 == 0:
                C = np.sum(P * np.log(P / Q))
                print(f"Iteration {iter}: KL divergence = {C}")
                
        return Y

# Example usage
X = np.random.randn(200, 50)
tsne = TSNE()
Y = tsne.fit_transform(X)

plt.scatter(Y[:, 0], Y[:, 1])
plt.title('t-SNE Visualization')
plt.show()
```

Slide 5: Gaussian Mixture Models Implementation

Gaussian Mixture Models represent complex probability distributions as a weighted sum of Gaussian components, enabling soft clustering and density estimation. This implementation demonstrates the EM algorithm for parameter estimation.

```python
import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixture:
    def __init__(self, n_components=3, max_iters=100, tol=1e-4):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covs_ = [np.eye(n_features) for _ in range(self.n_components)]
        
        # EM Algorithm
        log_likelihood = -np.inf
        
        for _ in range(self.max_iters):
            # E-step: Compute responsibilities
            resp = self._e_step(X)
            
            # M-step: Update parameters
            prev_log_likelihood = log_likelihood
            log_likelihood = self._m_step(X, resp)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
                
        return self
    
    def _e_step(self, X):
        resp = np.zeros((X.shape[0], self.n_components))
        
        for k in range(self.n_components):
            resp[:, k] = self.weights_[k] * multivariate_normal.pdf(
                X, mean=self.means_[k], cov=self.covs_[k]
            )
            
        # Normalize responsibilities
        resp /= resp.sum(axis=1, keepdims=True)
        return resp
    
    def _m_step(self, X, resp):
        N = resp.sum(axis=0)
        
        # Update weights
        self.weights_ = N / X.shape[0]
        
        # Update means
        self.means_ = np.dot(resp.T, X) / N[:, np.newaxis]
        
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covs_[k] = np.dot(resp[:, k] * diff.T, diff) / N[k]
            
        # Compute log likelihood
        return self._compute_log_likelihood(X)
    
    def _compute_log_likelihood(self, X):
        log_likelihood = 0
        for k in range(self.n_components):
            log_likelihood += self.weights_[k] * multivariate_normal.pdf(
                X, mean=self.means_[k], cov=self.covs_[k]
            )
        return np.sum(np.log(log_likelihood))
    
    def predict(self, X):
        resp = self._e_step(X)
        return np.argmax(resp, axis=1)

# Example usage
np.random.seed(42)
X = np.concatenate([
    np.random.normal(0, 1, (200, 2)),
    np.random.normal(4, 1.5, (200, 2)),
    np.random.normal(8, 0.5, (200, 2))
])

gmm = GaussianMixture(n_components=3)
gmm.fit(X)
labels = gmm.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], 
           marker='x', s=200, linewidths=3, color='r')
plt.title('Gaussian Mixture Model Clustering')
plt.show()
```

Slide 6: Hierarchical Agglomerative Clustering

Hierarchical clustering builds a tree of nested clusters by iteratively merging the closest pairs of clusters. This implementation shows the bottom-up approach using different linkage criteria.

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        
    def fit_predict(self, X):
        n_samples = X.shape[0]
        
        # Initialize distances and clusters
        distances = squareform(pdist(X))
        current_clusters = [[i] for i in range(n_samples)]
        labels = np.arange(n_samples)
        
        while len(current_clusters) > self.n_clusters:
            # Find closest clusters
            min_dist = np.inf
            merge_i, merge_j = 0, 0
            
            for i in range(len(current_clusters)):
                for j in range(i + 1, len(current_clusters)):
                    dist = self._compute_distance(
                        current_clusters[i], 
                        current_clusters[j], 
                        distances
                    )
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # Merge clusters
            new_cluster = current_clusters[merge_i] + current_clusters[merge_j]
            current_clusters = (
                current_clusters[:merge_i] + 
                current_clusters[merge_i+1:merge_j] + 
                current_clusters[merge_j+1:] + 
                [new_cluster]
            )
            
            # Update labels
            new_label = len(current_clusters) - 1
            for idx in new_cluster:
                labels[idx] = new_label
                
        return labels
    
    def _compute_distance(self, cluster1, cluster2, distances):
        if self.linkage == 'single':
            return np.min([distances[i, j] for i in cluster1 for j in cluster2])
        elif self.linkage == 'complete':
            return np.max([distances[i, j] for i in cluster1 for j in cluster2])
        else:  # average linkage
            return np.mean([distances[i, j] for i in cluster1 for j in cluster2])

# Example usage
np.random.seed(42)
X = np.concatenate([
    np.random.normal(0, 1, (50, 2)),
    np.random.normal(5, 1, (50, 2))
])

hac = AgglomerativeClustering(n_clusters=2, linkage='average')
labels = hac.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title('Hierarchical Agglomerative Clustering')
plt.show()
```

Slide 7: Real-world Application: Customer Segmentation

A practical implementation of clustering techniques for customer segmentation using real-world-like data, demonstrating preprocessing, model selection, and evaluation metrics.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Generate synthetic customer data
np.random.seed(42)
n_customers = 1000

# Create customer features
data = {
    'recency': np.random.exponential(50, n_customers),
    'frequency': np.random.poisson(10, n_customers),
    'monetary': np.random.lognormal(4, 1, n_customers),
    'avg_purchase': np.random.normal(100, 30, n_customers),
    'items_bought': np.random.poisson(5, n_customers)
}

df = pd.DataFrame(data)

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Dimensionality reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Find optimal number of clusters
silhouette_scores = []
K = range(2, 8)

for k in K:
    kmeans = KMeansClustering(k=k)
    labels = kmeans.fit(X_reduced)
    score = silhouette_score(X_reduced, labels)
    silhouette_scores.append(score)

optimal_k = K[np.argmax(silhouette_scores)]

# Final clustering
kmeans = KMeansClustering(k=optimal_k)
labels = kmeans.fit(X_reduced)

# Analyze segments
segments = pd.DataFrame(X_scaled, columns=df.columns)
segments['Cluster'] = labels

# Calculate segment profiles
profiles = segments.groupby('Cluster').mean()
print("\nCustomer Segment Profiles:")
print(profiles)

# Visualize results
plt.figure(figsize=(12, 5))

# Plot clusters
plt.subplot(1, 2, 1)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels)
plt.title('Customer Segments')

# Plot silhouette scores
plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Optimal Cluster Selection')
plt.tight_layout()
plt.show()
```

Slide 8: Autoencoder Implementation for Dimensionality Reduction

Autoencoders provide a powerful neural network-based approach to dimensionality reduction, learning a compressed representation of the input data through an encoding-decoding process.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

# Training function
def train_autoencoder(model, data_loader, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data_loader):.4f}')

# Example usage
input_dim = 784  # e.g., for MNIST
encoding_dim = 32
batch_size = 128

# Create synthetic data
X = torch.randn(1000, input_dim)
dataset = torch.utils.data.TensorDataset(X)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize and train model
model = Autoencoder(input_dim, encoding_dim)
train_autoencoder(model, data_loader)

# Get encoded representations
encoded_data = model.encode(X)
print("Original dimension:", X.shape)
print("Encoded dimension:", encoded_data.shape)
```

Slide 9: Spectral Clustering Implementation

Spectral clustering leverages the eigendecomposition of the graph Laplacian matrix to perform dimensionality reduction before clustering, making it effective for complex non-linear structures.

```python
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph

class SpectralClustering:
    def __init__(self, n_clusters=2, n_neighbors=10):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        
    def fit_predict(self, X):
        # Construct similarity graph
        connectivity = kneighbors_graph(
            X, n_neighbors=self.n_neighbors, mode='distance'
        )
        adjacency = 0.5 * (connectivity + connectivity.T)
        
        # Compute normalized Laplacian
        degree = np.array(adjacency.sum(axis=1)).flatten()
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        L = np.eye(len(X)) - D_inv_sqrt @ adjacency @ D_inv_sqrt
        
        # Compute eigenvectors
        eigenvalues, eigenvectors = eigsh(
            L, k=self.n_clusters, which='SM'
        )
        
        # Apply k-means to eigenvectors
        kmeans = KMeansClustering(k=self.n_clusters)
        labels = kmeans.fit(eigenvectors)
        
        return labels

# Example usage with non-linear structures
def generate_circles():
    t = np.linspace(0, 2*np.pi, 200)
    inner_x = 3 * np.cos(t)
    inner_y = 3 * np.sin(t)
    outer_x = 6 * np.cos(t)
    outer_y = 6 * np.sin(t)
    
    X = np.vstack([
        np.column_stack([inner_x, inner_y]),
        np.column_stack([outer_x, outer_y])
    ])
    X += np.random.normal(0, 0.3, X.shape)
    return X

# Generate and cluster data
X = generate_circles()
spectral = SpectralClustering(n_clusters=2, n_neighbors=5)
labels = spectral.fit_predict(X)

# Visualize results
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c='b', alpha=0.5)
plt.title('Original Data')

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title('Spectral Clustering Results')
plt.show()
```

Slide 10: Real-world Application: Image Compression using PCA

This implementation demonstrates the practical application of PCA for image compression, showing the trade-off between compression ratio and image quality.

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

def compress_image(image_path, n_components):
    # Load and prepare image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    compressed = pca.fit_transform(img_array)
    reconstructed = pca.inverse_transform(compressed)
    
    # Calculate compression ratio and error
    original_size = img_array.shape[0] * img_array.shape[1]
    compressed_size = compressed.shape[0] * compressed.shape[1]
    compression_ratio = original_size / compressed_size
    
    mse = np.mean((img_array - reconstructed) ** 2)
    psnr = 10 * np.log10(255**2 / mse)
    
    return reconstructed, compression_ratio, psnr

# Example usage with synthetic image
def create_synthetic_image():
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2)) + np.random.normal(0, 0.1, X.shape)
    return (Z - Z.min()) / (Z.max() - Z.min()) * 255

# Generate and save synthetic image
synthetic_img = create_synthetic_image()
plt.imsave('synthetic.png', synthetic_img, cmap='gray')

# Test different compression levels
n_components_list = [5, 10, 20, 50]
fig, axes = plt.subplots(1, len(n_components_list) + 1, figsize=(15, 3))

# Original image
axes[0].imshow(synthetic_img, cmap='gray')
axes[0].set_title('Original')

# Compressed versions
for i, n in enumerate(n_components_list):
    reconstructed, ratio, psnr = compress_image('synthetic.png', n)
    axes[i+1].imshow(reconstructed, cmap='gray')
    axes[i+1].set_title(f'n={n}\nRatio={ratio:.1f}\nPSNR={psnr:.1f}')

plt.tight_layout()
plt.show()
```

Slide 11: UMAP Implementation

UMAP (Uniform Manifold Approximation and Projection) provides state-of-the-art dimensionality reduction while preserving both local and global structure. This implementation shows the core algorithm components.

```python
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

class UMAP:
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        
    def fit_transform(self, X):
        # Compute nearest neighbors
        knn = NearestNeighbors(n_neighbors=self.n_neighbors)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        
        # Compute fuzzy simplicial set
        rows, cols, vals = self._compute_fuzzy_simplicial_set(distances, indices)
        graph = csr_matrix((vals, (rows, cols)))
        
        # Initialize low-dimensional embedding
        embedding = self._initialize_embedding(X)
        
        # Optimize embedding
        embedding = self._optimize_embedding(embedding, graph)
        
        return embedding
    
    def _compute_fuzzy_simplicial_set(self, distances, indices):
        rows = []
        cols = []
        vals = []
        
        for i in range(len(indices)):
            for j, d in zip(indices[i], distances[i]):
                if i != j:
                    # Compute membership strength
                    val = np.exp(-d / distances[i].mean())
                    rows.append(i)
                    cols.append(j)
                    vals.append(val)
                    
        return np.array(rows), np.array(cols), np.array(vals)
    
    def _initialize_embedding(self, X):
        return np.random.normal(0, 1e-4, size=(X.shape[0], self.n_components))
    
    def _optimize_embedding(self, embedding, graph, n_epochs=200):
        learning_rate = 1.0
        
        for epoch in range(n_epochs):
            # Compute attractive and repulsive forces
            attractive_force = self._compute_attractive_force(embedding, graph)
            repulsive_force = self._compute_repulsive_force(embedding)
            
            # Update embedding
            embedding = embedding + learning_rate * (attractive_force - repulsive_force)
            
            # Update learning rate
            learning_rate = 1.0 * (1.0 - (epoch / n_epochs))
            
        return embedding
    
    def _compute_attractive_force(self, embedding, graph):
        # Simplified attractive force computation
        dist_matrix = self._compute_distance_matrix(embedding)
        attractive = graph.multiply(1.0 / (1.0 + dist_matrix))
        return attractive.dot(embedding)
    
    def _compute_repulsive_force(self, embedding):
        # Simplified repulsive force computation
        dist_matrix = self._compute_distance_matrix(embedding)
        repulsive = 1.0 / (1.0 + dist_matrix) ** 2
        return repulsive.dot(embedding)
    
    def _compute_distance_matrix(self, embedding):
        sum_squares = np.sum(embedding ** 2, axis=1)
        return np.sqrt(sum_squares[:, np.newaxis] + sum_squares - 2 * embedding.dot(embedding.T))

# Example usage
np.random.seed(42)
# Generate Swiss roll dataset
t = np.random.uniform(0, 4 * np.pi, 1000)
x = t * np.cos(t)
y = t * np.sin(t)
z = np.random.uniform(0, 2, 1000)
X = np.column_stack((x, y, z))

# Apply UMAP
umap = UMAP(n_components=2)
X_reduced = umap.fit_transform(X)

# Visualize results
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=t)
plt.title('Original Data (First 2 Dimensions)')

plt.subplot(122)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t)
plt.title('UMAP Embedding')
plt.show()
```

Slide 12: Anomaly Detection using Isolation Forest

Isolation Forest provides an efficient approach to anomaly detection by isolating outliers through random partitioning of the feature space.

```python
import numpy as np
from numpy.random import choice

class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.size = 0
        self.height = 0
        
    def fit(self, X, height=0):
        self.size = X.shape[0]
        self.height = height
        
        if height >= self.height_limit or self.size <= 1:
            return
            
        # Select random feature and split value
        feature_idx = choice(X.shape[1])
        feature_max = X[:, feature_idx].max()
        feature_min = X[:, feature_idx].min()
        
        if feature_max == feature_min:
            return
            
        split_value = np.random.uniform(feature_min, feature_max)
        
        self.split_feature = feature_idx
        self.split_value = split_value
        
        # Split data
        left_mask = X[:, feature_idx] < split_value
        X_left = X[left_mask]
        X_right = X[~left_mask]
        
        # Create child trees
        if len(X_left) > 0:
            self.left = IsolationTree(self.height_limit)
            self.left.fit(X_left, height + 1)
            
        if len(X_right) > 0:
            self.right = IsolationTree(self.height_limit)
            self.right.fit(X_right, height + 1)
    
    def path_length(self, x):
        if self.split_feature is None:
            return self.height
            
        if x[self.split_feature] < self.split_value:
            if self.left is None:
                return self.height
            return self.left.path_length(x)
        else:
            if self.right is None:
                return self.height
            return self.right.path_length(x)

class IsolationForest:
    def __init__(self, n_trees=100, sample_size=256):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.trees = []
        
    def fit(self, X):
        self.trees = []
        height_limit = int(np.ceil(np.log2(self.sample_size)))
        
        for _ in range(self.n_trees):
            idx = choice(X.shape[0], size=min(self.sample_size, X.shape[0]), replace=False)
            tree = IsolationTree(height_limit)
            tree.fit(X[idx])
            self.trees.append(tree)
            
    def anomaly_score(self, x):
        paths = [tree.path_length(x) for tree in self.trees]
        mean_path = np.mean(paths)
        
        # Normalize score
        n = self.sample_size
        c = 2 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)
        score = 2 ** (-mean_path / c)
        
        return score

# Example usage
np.random.seed(42)

# Generate normal data
n_samples = 1000
n_outliers = 50
n_features = 2

X = np.random.normal(0, 1, (n_samples, n_features))

# Add outliers
X_outliers = np.random.uniform(-4, 4, (n_outliers, n_features))
X = np.vstack([X, X_outliers])

# Fit Isolation Forest
clf = IsolationForest()
clf.fit(X)

# Compute anomaly scores
scores = np.array([clf.anomaly_score(x) for x in X])

# Visualize results
plt.figure(figsize=(10, 5))
plt.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c=scores[:-n_outliers],
           cmap='viridis', label='Normal points')
plt.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='red',
           label='Outliers')
plt.colorbar(label='Anomaly Score')
plt.legend()
plt.title('Isolation Forest Anomaly Detection')
plt.show()
```

Slide 13: Online K-Means Clustering Implementation

This implementation demonstrates an online learning approach to K-means clustering, allowing for real-time updates as new data points arrive, making it suitable for streaming data applications.

```python
import numpy as np
from collections import deque

class OnlineKMeans:
    def __init__(self, n_clusters=3, window_size=1000, learning_rate=0.1):
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.centroids = None
        self.data_window = deque(maxlen=window_size)
        self.n_samples_seen = 0
        
    def partial_fit(self, X):
        # Initialize centroids if needed
        if self.centroids is None:
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            self.centroids = X[indices].copy()
            
        # Update data window
        for x in X:
            self.data_window.append(x)
            
        # Update centroids
        for x in X:
            self.n_samples_seen += 1
            closest_centroid = self._get_closest_centroid(x)
            lr = self.learning_rate * (1.0 / (1.0 + self.n_samples_seen))
            self.centroids[closest_centroid] += lr * (x - self.centroids[closest_centroid])
            
        return self
    
    def predict(self, X):
        return np.array([self._get_closest_centroid(x) for x in X])
    
    def _get_closest_centroid(self, x):
        distances = np.sum((self.centroids - x) ** 2, axis=1)
        return np.argmin(distances)
    
    def get_stream_statistics(self):
        if len(self.data_window) < 2:
            return {}
            
        labels = self.predict(np.array(self.data_window))
        stats = {
            'n_samples_seen': self.n_samples_seen,
            'window_cluster_sizes': [np.sum(labels == i) for i in range(self.n_clusters)],
            'window_cluster_means': [np.mean(np.array(self.data_window)[labels == i], axis=0) 
                                   for i in range(self.n_clusters)]
        }
        return stats

# Example usage with streaming data
np.random.seed(42)

# Generate streaming data
def generate_stream(n_points=1000):
    centers = [(0, 0), (5, 5), (-5, 5)]
    while True:
        center = centers[np.random.randint(len(centers))]
        yield np.random.normal(center, 1, 2)

# Initialize online k-means
online_kmeans = OnlineKMeans(n_clusters=3)

# Process streaming data
stream = generate_stream()
n_batches = 10
batch_size = 100

plt.figure(figsize=(15, 5))
for i in range(n_batches):
    # Get batch of data
    batch = np.array([next(stream) for _ in range(batch_size)])
    
    # Update model
    online_kmeans.partial_fit(batch)
    
    # Plot current state
    plt.subplot(1, n_batches, i+1)
    labels = online_kmeans.predict(batch)
    plt.scatter(batch[:, 0], batch[:, 1], c=labels, alpha=0.5)
    plt.scatter(online_kmeans.centroids[:, 0], 
               online_kmeans.centroids[:, 1], 
               marker='x', s=200, linewidths=3, 
               color='r', label='Centroids')
    plt.title(f'Batch {i+1}')
    
    # Print statistics
    stats = online_kmeans.get_stream_statistics()
    print(f"\nBatch {i+1} Statistics:")
    print(f"Samples seen: {stats['n_samples_seen']}")
    print(f"Cluster sizes: {stats['window_cluster_sizes']}")

plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

*   ArXiv Papers on Clustering and Dimensionality Reduction:
    *   "A Tutorial on Spectral Clustering" - [https://arxiv.org/abs/0711.0189](https://arxiv.org/abs/0711.0189)
    *   "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction" - [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
    *   "Understanding the t-SNE Algorithm for Dimensionality Reduction" - [https://arxiv.org/abs/2002.12016](https://arxiv.org/abs/2002.12016)
    *   "A Survey of Recent Advances in Hierarchical Clustering Algorithms" - [https://arxiv.org/abs/1908.08604](https://arxiv.org/abs/1908.08604)
    *   "Deep Learning for Unsupervised Clustering: A Survey" - [https://arxiv.org/abs/2010.01587](https://arxiv.org/abs/2010.01587)
*   Additional Resources for Advanced Topics:
    *   Google Scholar: "Advances in Neural Information Processing Systems (NeurIPS)" proceedings
    *   "Pattern Recognition and Machine Learning" by Christopher Bishop
    *   Journal of Machine Learning Research (JMLR) Special Issues on Clustering
    *   International Conference on Machine Learning (ICML) workshops on unsupervised learning
*   Online Learning Resources:
    *   Scikit-learn Documentation: [https://scikit-learn.org/stable/modules/clustering.html](https://scikit-learn.org/stable/modules/clustering.html)
    *   Stanford CS229 Course Notes on Unsupervised Learning
    *   Deep Learning Specialization on Coursera (Course 4: Unsupervised Learning)
    *   KDnuggets tutorials on clustering and dimensionality reduction

