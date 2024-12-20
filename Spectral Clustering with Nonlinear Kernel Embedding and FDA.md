## Spectral Clustering with Nonlinear Kernel Embedding and FDA
Slide 1: Introduction to Spectral Clustering and FDA

Spectral clustering is a powerful technique for unsupervised learning that leverages the eigenvalues of similarity matrices to perform dimensionality reduction before clustering. Functional Data Analysis (FDA) extends traditional data analysis methods to handle functions as data points. Combining these approaches with nonlinear kernel embedding allows for robust clustering of complex, high-dimensional data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons

# Generate sample data
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Perform spectral clustering
sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
labels = sc.fit_predict(X)

# Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Spectral Clustering on Moons Dataset')
plt.show()
```

Slide 2: Nonlinear Kernel Embedding

Nonlinear kernel embedding maps data into a high-dimensional feature space where linear separation becomes possible. This is crucial for spectral clustering as it allows the algorithm to capture complex, nonlinear relationships in the data.

```python
from sklearn.kernel_approximation import RBFSampler

# Generate sample data
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Apply RBF kernel approximation
rbf_feature = RBFSampler(gamma=1, n_components=100, random_state=42)
X_embedded = rbf_feature.fit_transform(X)

# Visualize the first two dimensions of the embedded space
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis')
plt.title('First Two Dimensions of RBF Kernel Embedding')
plt.show()
```

Slide 3: Similarity Graph Construction

The first step in spectral clustering is constructing a similarity graph. This graph represents the pairwise similarities between data points, which can be computed using various kernel functions.

```python
from sklearn.metrics.pairwise import rbf_kernel

def construct_similarity_graph(X, gamma=1.0):
    # Compute pairwise similarities using RBF kernel
    S = rbf_kernel(X, gamma=gamma)
    
    # Set diagonal elements to 0 to avoid self-loops
    np.fill_diagonal(S, 0)
    
    return S

# Construct similarity graph
S = construct_similarity_graph(X)

# Visualize the similarity matrix
plt.imshow(S, cmap='viridis')
plt.colorbar()
plt.title('Similarity Matrix')
plt.show()
```

Slide 4: Laplacian Matrix Computation

The Laplacian matrix is a key component in spectral clustering. It encodes the structure of the similarity graph and is used to compute the spectral embedding.

```python
def compute_laplacian(S):
    # Compute degree matrix
    D = np.diag(np.sum(S, axis=1))
    
    # Compute unnormalized Laplacian
    L = D - S
    
    # Compute normalized Laplacian
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    L_norm = np.eye(S.shape[0]) - D_inv_sqrt @ S @ D_inv_sqrt
    
    return L_norm

# Compute normalized Laplacian
L_norm = compute_laplacian(S)

# Visualize the Laplacian matrix
plt.imshow(L_norm, cmap='viridis')
plt.colorbar()
plt.title('Normalized Laplacian Matrix')
plt.show()
```

Slide 5: Eigendecomposition and Spectral Embedding

Spectral embedding is achieved by computing the eigenvectors of the Laplacian matrix. These eigenvectors form a low-dimensional representation of the data that preserves its cluster structure.

```python
def spectral_embedding(L_norm, n_components=2):
    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
    
    # Sort eigenvectors by eigenvalues in ascending order
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    
    # Select the first n_components eigenvectors (excluding the first one)
    embedding = eigenvectors[:, 1:n_components+1]
    
    return embedding

# Compute spectral embedding
embedding = spectral_embedding(L_norm, n_components=2)

# Visualize the spectral embedding
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis')
plt.title('Spectral Embedding')
plt.show()
```

Slide 6: Clustering in the Embedded Space

After obtaining the spectral embedding, we can apply traditional clustering algorithms like K-means to group the data points.

```python
from sklearn.cluster import KMeans

def spectral_clustering(X, n_clusters=2):
    S = construct_similarity_graph(X)
    L_norm = compute_laplacian(S)
    embedding = spectral_embedding(L_norm, n_components=n_clusters)
    
    # Apply K-means clustering to the embedding
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embedding)
    
    return labels

# Perform spectral clustering
sc_labels = spectral_clustering(X, n_clusters=2)

# Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=sc_labels, cmap='viridis')
plt.title('Custom Spectral Clustering Results')
plt.show()
```

Slide 7: Introduction to Functional Data Analysis (FDA)

Functional Data Analysis deals with data that are functions rather than discrete values. In FDA, each data point is a function defined on some continuous domain, such as time or space.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Generate functional data
n_samples = 10
t = np.linspace(0, 1, 100)

def generate_function(t):
    return np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, t.shape)

functions = [generate_function(t) for _ in range(n_samples)]

# Plot functional data
plt.figure(figsize=(10, 6))
for f in functions:
    plt.plot(t, f, alpha=0.7)
plt.title('Functional Data Examples')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
```

Slide 8: Functional Principal Component Analysis (FPCA)

FPCA is an extension of PCA to functional data. It helps identify the main modes of variation in a set of functions and can be used for dimensionality reduction.

```python
from sklearn.decomposition import PCA

# Perform FPCA
n_components = 3
pca = PCA(n_components=n_components)
fpca_scores = pca.fit_transform(np.array(functions))

# Plot FPCA components
plt.figure(figsize=(12, 4))
for i in range(n_components):
    plt.subplot(1, 3, i+1)
    plt.plot(t, pca.components_[i])
    plt.title(f'FPCA Component {i+1}')
    plt.xlabel('Time')
plt.tight_layout()
plt.show()

# Reconstruct functions using FPCA
reconstructed = pca.inverse_transform(fpca_scores)

# Plot original and reconstructed functions
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for f in functions:
    plt.plot(t, f, alpha=0.7)
plt.title('Original Functions')
plt.subplot(1, 2, 2)
for f in reconstructed:
    plt.plot(t, f, alpha=0.7)
plt.title('Reconstructed Functions')
plt.tight_layout()
plt.show()
```

Slide 9: Combining Spectral Clustering and FDA

We can apply spectral clustering to functional data by using a suitable kernel function that measures similarity between functions.

```python
from scipy.integrate import simps

def l2_distance(f1, f2):
    return simps((f1 - f2)**2, t)

def rbf_kernel_functional(f1, f2, gamma=1.0):
    return np.exp(-gamma * l2_distance(f1, f2))

# Compute similarity matrix for functional data
n = len(functions)
S_functional = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        S_functional[i, j] = rbf_kernel_functional(functions[i], functions[j])

# Apply spectral clustering to functional data
sc_functional = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
labels_functional = sc_functional.fit_predict(S_functional)

# Visualize clustering results
plt.figure(figsize=(10, 6))
for f, label in zip(functions, labels_functional):
    plt.plot(t, f, c='r' if label == 0 else 'b', alpha=0.7)
plt.title('Spectral Clustering of Functional Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
```

Slide 10: Real-Life Example: Climate Data Analysis

Spectral clustering with FDA can be applied to analyze climate data, such as temperature patterns across different locations.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and preprocess climate data (replace with actual data loading)
data = pd.DataFrame({
    'time': pd.date_range(start='2020-01-01', end='2020-12-31', freq='D'),
    'temp_city1': np.random.normal(20, 5, 366) + 10 * np.sin(np.linspace(0, 2*np.pi, 366)),
    'temp_city2': np.random.normal(25, 3, 366) + 5 * np.sin(np.linspace(0, 2*np.pi, 366)),
    'temp_city3': np.random.normal(15, 7, 366) + 15 * np.sin(np.linspace(0, 2*np.pi, 366))
})

# Normalize temperature data
scaler = StandardScaler()
temp_normalized = scaler.fit_transform(data[['temp_city1', 'temp_city2', 'temp_city3']])

# Perform spectral clustering
sc_climate = SpectralClustering(n_clusters=2, random_state=42)
labels_climate = sc_climate.fit_predict(temp_normalized.T)

# Visualize results
plt.figure(figsize=(12, 6))
for i, city in enumerate(['City 1', 'City 2', 'City 3']):
    plt.plot(data['time'], temp_normalized[:, i], label=f'{city} (Cluster {labels_climate[i]})')
plt.legend()
plt.title('Spectral Clustering of Temperature Patterns')
plt.xlabel('Time')
plt.ylabel('Normalized Temperature')
plt.show()
```

Slide 11: Real-Life Example: Gesture Recognition

Spectral clustering with FDA can be used for gesture recognition by analyzing time series data from motion sensors.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering

# Simulate gesture data (replace with actual sensor data)
def generate_gesture(n_points=100):
    t = np.linspace(0, 1, n_points)
    x = np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, n_points)
    y = np.cos(2 * np.pi * t) + np.random.normal(0, 0.1, n_points)
    z = t + np.random.normal(0, 0.1, n_points)
    return np.column_stack((x, y, z))

n_gestures = 50
gestures = [generate_gesture() for _ in range(n_gestures)]

# Flatten and normalize gesture data
X = np.array([g.flatten() for g in gestures])
X_normalized = StandardScaler().fit_transform(X)

# Perform spectral clustering
sc_gestures = SpectralClustering(n_clusters=2, random_state=42)
labels_gestures = sc_gestures.fit_predict(X_normalized)

# Visualize results
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
for gesture, label in zip(gestures, labels_gestures):
    ax.plot(gesture[:, 0], gesture[:, 1], gesture[:, 2], c='r' if label == 0 else 'b', alpha=0.7)
ax.set_title('Spectral Clustering of Gesture Data')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
```

Slide 12: Challenges and Considerations

When applying spectral clustering with nonlinear kernel embedding and FDA, several challenges may arise:

1. Choosing appropriate kernel functions and parameters
2. Determining the optimal number of clusters
3. Handling high-dimensional functional data
4. Computational complexity for large datasets

To address these challenges, consider using techniques such as:

Slide 14: Challenges and Considerations

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

# Example: Grid search for optimal parameters
param_grid = {
    'n_clusters': [2, 3, 4, 5],
    'gamma': [0.1, 1, 10]
}

def spectral_clustering_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    return silhouette_score(X, labels)

grid_search = GridSearchCV(
    SpectralClustering(random_state=42),
    param_grid,
    scoring=spectral_clustering_scorer,
    cv=5
)

# Fit the grid search (Note: This can be computationally expensive)
# grid_search.fit(X_normalized)

# Print best parameters
# print("Best parameters:", grid_search.best_params_)
```

Slide 15: Future Directions and Advanced Topics

1. Multi-view spectral clustering for heterogeneous data sources
2. Online spectral clustering for streaming functional data
3. Deep spectral clustering using neural networks
4. Integration with other dimensionality reduction techniques

Slide 16: Future Directions and Advanced Topics

```python
import torch
import torch.nn as nn

# Example: Simple deep spectral clustering model
class DeepSpectralClustering(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_clusters):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_clusters)
        )
        
    def forward(self, x):
        return self.encoder(x)

# Initialize model (replace with actual implementation)
# model = DeepSpectralClustering(input_dim=X.shape[1], hidden_dim=64, n_clusters=2)
# optimizer = torch.optim.Adam(model.parameters())

# Training loop (pseudo-code)
# for epoch in range(n_epochs):
#     optimizer.zero_grad()
#     embeddings = model(X_tensor)
#     loss = spectral_clustering_loss(embeddings)
#     loss.backward()
#     optimizer.step()
```

Slide 17: Additional Resources

For further exploration of spectral clustering, nonlinear kernel embedding, and functional data analysis, consider the following resources:

1. Ng, A. Y., Jordan, M. I., & Weiss, Y. (2002). On spectral clustering: Analysis and an algorithm. Advances in neural information processing systems, 14. ArXiv URL: [https://arxiv.org/abs/1206.5830](https://arxiv.org/abs/1206.5830)
2. Ramsay, J. O., & Silverman, B. W. (2005). Functional Data Analysis. Springer Series in Statistics.
3. Wang, X., & Davidson, I. (2010). Flexible constrained spectral clustering. Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining. ArXiv URL: [https://arxiv.org/abs/1002.4837](https://arxiv.org/abs/1002.4837)
4. Ferraty, F., & Vieu, P. (2006). Nonparametric Functional Data Analysis: Theory and Practice. Springer Series in Statistics.
5. Von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics and computing, 17(4), 395-416. ArXiv URL: [https://arxiv.org/abs/0711.0189](https://arxiv.org/abs/0711.0189)

These resources provide in-depth discussions on the theoretical foundations and practical applications of the techniques covered in this presentation.

