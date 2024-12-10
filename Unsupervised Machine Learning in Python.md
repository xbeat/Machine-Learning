## Unsupervised Machine Learning in Python
Slide 1: Introduction to Unsupervised Learning Fundamentals

Unsupervised learning algorithms discover hidden patterns in unlabeled data by identifying inherent structures and relationships. These algorithms operate without explicit target variables, making them essential for exploratory data analysis and feature learning in real-world applications where labeled data is scarce.

```python
# Basic example of data preparation for unsupervised learning
import numpy as np
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 5)  # 1000 samples, 5 features

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Original data shape:", X.shape)
print("Scaled data statistics:")
print("Mean:", X_scaled.mean(axis=0))
print("Std:", X_scaled.std(axis=0))
```

Slide 2: K-Means Clustering Implementation

K-means clustering partitions n observations into k clusters by iteratively assigning points to the nearest centroid and updating centroid positions. This implementation demonstrates the algorithm's core mechanics without using external libraries, showcasing the fundamental concepts.

```python
def kmeans_from_scratch(X, k, max_iters=100):
    # Randomly initialize centroids
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids

# Example usage
X = np.random.randn(300, 2)  # 300 samples, 2 features
labels, centroids = kmeans_from_scratch(X, k=3)
```

Slide 3: Principal Component Analysis Theory

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving maximum variance. The mathematical foundation involves eigendecomposition of the covariance matrix.

```python
# Mathematical representation of PCA
"""
Covariance Matrix:
$$C = \frac{1}{n-1}X^TX$$

Eigendecomposition:
$$C = V\Lambda V^T$$

Projection Matrix:
$$P = XV$$

Where:
$$X$$ is the centered data matrix
$$V$$ contains eigenvectors
$$\Lambda$$ contains eigenvalues
"""
```

Slide 4: PCA Implementation from Scratch

```python
def pca_from_scratch(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components eigenvectors
    components = eigenvectors[:, :n_components]
    
    # Project data
    X_transformed = np.dot(X_centered, components)
    
    return X_transformed, components

# Example usage
X = np.random.randn(200, 5)
X_transformed, components = pca_from_scratch(X, n_components=2)
print("Original shape:", X.shape)
print("Transformed shape:", X_transformed.shape)
```

Slide 5: Hierarchical Clustering Implementation

Hierarchical clustering builds a tree of clusters by progressively merging or splitting groups based on distance metrics. This implementation demonstrates agglomerative clustering using single linkage strategy.

```python
def hierarchical_clustering(X, n_clusters):
    n_samples = X.shape[0]
    
    # Initialize clusters (each point is a cluster)
    clusters = [[i] for i in range(n_samples)]
    
    # Calculate initial distances
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distances[i, j] = np.sqrt(np.sum((X[i] - X[j])**2))
            distances[j, i] = distances[i, j]
    
    while len(clusters) > n_clusters:
        # Find closest clusters
        min_dist = float('inf')
        merge_idx = None
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = min([distances[x, y] for x in clusters[i] for y in clusters[j]])
                if dist < min_dist:
                    min_dist = dist
                    merge_idx = (i, j)
        
        # Merge clusters
        i, j = merge_idx
        clusters[i].extend(clusters[j])
        clusters.pop(j)
    
    return clusters

# Example usage
X = np.random.randn(50, 2)
clusters = hierarchical_clustering(X, n_clusters=3)
```

Slide 6: DBSCAN Algorithm Implementation

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies clusters based on density, distinguishing core points, border points, and noise points. This algorithm excels at discovering clusters of arbitrary shapes and handling noise effectively.

```python
def dbscan_from_scratch(X, eps, min_samples):
    n_samples = X.shape[0]
    labels = np.zeros(n_samples, dtype=int) - 1  # -1 represents unvisited
    cluster_label = 0
    
    def get_neighbors(point_idx):
        distances = np.sqrt(np.sum((X - X[point_idx])**2, axis=1))
        return np.where(distances <= eps)[0]
    
    def expand_cluster(point_idx, neighbors, cluster_label):
        labels[point_idx] = cluster_label
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_label
                new_neighbors = get_neighbors(neighbor_idx)
                if len(new_neighbors) >= min_samples:
                    neighbors = np.append(neighbors, new_neighbors)
            i += 1
    
    for i in range(n_samples):
        if labels[i] != -1:
            continue
            
        neighbors = get_neighbors(i)
        if len(neighbors) < min_samples:
            labels[i] = 0  # Mark as noise
        else:
            cluster_label += 1
            expand_cluster(i, neighbors, cluster_label)
    
    return labels

# Example usage
X = np.random.randn(100, 2) * 2
labels = dbscan_from_scratch(X, eps=0.5, min_samples=5)
```

Slide 7: Gaussian Mixture Models Theory

Gaussian Mixture Models represent complex probability distributions as a weighted sum of Gaussian components, using the Expectation-Maximization algorithm for parameter estimation. Each component is defined by its mean, covariance, and mixing coefficient.

```python
"""
GMM Probability Density Function:
$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k,\Sigma_k)$$

Log-likelihood:
$$\ln p(X|\pi,\mu,\Sigma) = \sum_{n=1}^N \ln\left(\sum_{k=1}^K \pi_k \mathcal{N}(x_n|\mu_k,\Sigma_k)\right)$$

Where:
$$\pi_k$$ are mixing coefficients
$$\mu_k$$ are means
$$\Sigma_k$$ are covariance matrices
"""
```

Slide 8: GMM Implementation

```python
def gmm_from_scratch(X, n_components, max_iters=100):
    n_samples, n_features = X.shape
    
    # Initialize parameters
    weights = np.ones(n_components) / n_components
    means = X[np.random.choice(n_samples, n_components, replace=False)]
    covs = [np.eye(n_features) for _ in range(n_components)]
    
    def gaussian_pdf(X, mean, cov):
        n = X.shape[1]
        diff = (X - mean).T
        return np.exp(-0.5 * np.sum(diff * np.linalg.solve(cov, diff), axis=0)) / \
               np.sqrt((2 * np.pi) ** n * np.linalg.det(cov))
    
    for _ in range(max_iters):
        # E-step: Compute responsibilities
        resp = np.zeros((n_samples, n_components))
        for k in range(n_components):
            resp[:, k] = weights[k] * gaussian_pdf(X.T, means[k], covs[k])
        resp /= resp.sum(axis=1, keepdims=True)
        
        # M-step: Update parameters
        Nk = resp.sum(axis=0)
        weights = Nk / n_samples
        
        for k in range(n_components):
            means[k] = np.sum(resp[:, k:k+1] * X, axis=0) / Nk[k]
            diff = X - means[k]
            covs[k] = np.dot(resp[:, k] * diff.T, diff) / Nk[k]
    
    return weights, means, covs

# Example usage
X = np.concatenate([np.random.randn(100, 2) + [2, 2],
                   np.random.randn(100, 2) + [-2, -2]])
weights, means, covs = gmm_from_scratch(X, n_components=2)
```

Slide 9: Real-world Application: Customer Segmentation

This implementation demonstrates customer segmentation using purchase history data, combining preprocessing, feature engineering, and clustering to identify distinct customer groups for targeted marketing strategies.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Generate synthetic customer data
np.random.seed(42)
n_customers = 1000

customer_data = pd.DataFrame({
    'recency': np.random.exponential(30, n_customers),
    'frequency': np.random.poisson(5, n_customers),
    'monetary': np.random.lognormal(4, 1, n_customers),
    'avg_basket': np.random.normal(50, 15, n_customers)
})

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_data)

# Apply clustering
def optimize_kmeans(X, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        labels, _ = kmeans_from_scratch(X, k)
        inertia = sum(np.min(np.sum((X - c)**2, axis=1)) 
                     for c in [X[labels == i].mean(axis=0) 
                     for i in range(k)])
        inertias.append(inertia)
    return inertias

# Find optimal number of clusters
inertias = optimize_kmeans(X_scaled)
optimal_k = 4  # Determined from elbow method

# Final clustering
labels, centroids = kmeans_from_scratch(X_scaled, optimal_k)
customer_data['Segment'] = labels
```

Slide 10: Results for Customer Segmentation Analysis

The customer segmentation analysis reveals distinct behavioral patterns across different customer groups, showing clear separation in purchasing habits and value contribution to the business.

```python
# Analyze segments
segment_analysis = customer_data.groupby('Segment').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'mean',
    'avg_basket': 'mean'
}).round(2)

print("Segment Characteristics:")
print(segment_analysis)

# Calculate segment sizes
segment_sizes = customer_data['Segment'].value_counts()
print("\nSegment Sizes:")
print(segment_sizes)

# Example output:
"""
Segment Characteristics:
         recency  frequency  monetary  avg_basket
Segment                                         
0         15.23      7.82    89.45      62.34
1         45.67      2.31    45.23      38.91
2         28.91      4.56    67.89      51.23
3         35.78      3.45    56.78      45.67

Segment Sizes:
Segment
2    312
0    289
1    256
3    143
"""
```

Slide 11: Autoencoder Neural Network Implementation

Autoencoders learn efficient data representations through dimensionality reduction by training the network to reconstruct its input. This implementation shows a simple autoencoder using numpy for unsupervised feature learning.

```python
class Autoencoder:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Initialize weights
        self.W1 = np.random.randn(input_dim, encoding_dim) * 0.01
        self.b1 = np.zeros(encoding_dim)
        self.W2 = np.random.randn(encoding_dim, input_dim) * 0.01
        self.b2 = np.zeros(input_dim)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # Encoder
        self.hidden = self.sigmoid(np.dot(X, self.W1) + self.b1)
        # Decoder
        self.output = self.sigmoid(np.dot(self.hidden, self.W2) + self.b2)
        return self.output
    
    def backward(self, X, learning_rate=0.01):
        m = X.shape[0]
        
        # Compute gradients
        error = self.output - X
        d_output = error * self.sigmoid_derivative(self.output)
        
        dW2 = np.dot(self.hidden.T, d_output)
        db2 = np.sum(d_output, axis=0)
        
        d_hidden = np.dot(d_output, self.W2.T) * self.sigmoid_derivative(self.hidden)
        dW1 = np.dot(X.T, d_hidden)
        db1 = np.sum(d_hidden, axis=0)
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
        return np.mean(error ** 2)

# Example usage
X = np.random.rand(1000, 10)
autoencoder = Autoencoder(input_dim=10, encoding_dim=5)

# Training
for epoch in range(100):
    output = autoencoder.forward(X)
    loss = autoencoder.backward(X)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

Slide 12: Anomaly Detection with Isolation Forest

Isolation Forest detects anomalies by isolating observations through recursive partitioning, assigning anomaly scores based on the average path length to isolation.

```python
class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.size = 0
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        
    def fit(self, X, current_height=0):
        self.size = len(X)
        
        if current_height >= self.height_limit or len(X) <= 1:
            return
        
        n_features = X.shape[1]
        self.split_feature = np.random.randint(n_features)
        min_val = X[:, self.split_feature].min()
        max_val = X[:, self.split_feature].max()
        
        if min_val == max_val:
            return
        
        self.split_value = np.random.uniform(min_val, max_val)
        
        left_indices = X[:, self.split_feature] < self.split_value
        self.left = IsolationTree(self.height_limit)
        self.right = IsolationTree(self.height_limit)
        
        self.left.fit(X[left_indices], current_height + 1)
        self.right.fit(X[~left_indices], current_height + 1)
        
    def path_length(self, x, current_height=0):
        if self.split_feature is None:
            return current_height
        
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x, current_height + 1)
        else:
            return self.right.path_length(x, current_height + 1)

# Example usage
X = np.concatenate([
    np.random.randn(95, 2),  # Normal points
    np.random.randn(5, 2) * 4  # Anomalies
])

# Create and train forest
n_trees = 100
height_limit = int(np.ceil(np.log2(len(X))))
forest = [IsolationTree(height_limit) for _ in range(n_trees)]

for tree in forest:
    indices = np.random.randint(0, len(X), size=len(X))
    tree.fit(X[indices])

# Calculate anomaly scores
def anomaly_score(x, forest):
    avg_path_length = np.mean([tree.path_length(x) for tree in forest])
    n = len(X)
    c = 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
    return 2 ** (-avg_path_length / c)

scores = np.array([anomaly_score(x, forest) for x in X])
anomalies = scores > np.percentile(scores, 95)  # Top 5% as anomalies
```

Slide 13: Word Embeddings with Word2Vec Implementation

Word2Vec implements unsupervised learning of word embeddings using neural networks, creating vector representations that capture semantic relationships between words through context prediction in large text corpora.

```python
import numpy as np
from collections import defaultdict, Counter

class Word2Vec:
    def __init__(self, dimensions=100, window_size=2, learning_rate=0.01):
        self.dimensions = dimensions
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.word_to_index = {}
        self.embeddings = None
        
    def build_vocabulary(self, sentences):
        word_counts = Counter([word for sentence in sentences for word in sentence])
        vocabulary = sorted(word_counts.keys())
        self.word_to_index = {word: idx for idx, word in enumerate(vocabulary)}
        vocab_size = len(vocabulary)
        
        # Initialize embeddings randomly
        self.embeddings = np.random.randn(vocab_size, self.dimensions) * 0.01
        self.context_weights = np.random.randn(self.dimensions, vocab_size) * 0.01
        
    def train_pair(self, target_idx, context_idx):
        # Forward pass
        hidden = self.embeddings[target_idx]
        output = np.dot(hidden, self.context_weights)
        prob = self._softmax(output)
        
        # Backward pass
        error = prob.copy()
        error[context_idx] -= 1
        
        # Update weights
        grad_context = np.outer(hidden, error)
        grad_hidden = np.dot(error, self.context_weights.T)
        
        self.context_weights -= self.learning_rate * grad_context
        self.embeddings[target_idx] -= self.learning_rate * grad_hidden
        
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def train(self, sentences, epochs=5):
        for epoch in range(epochs):
            loss = 0
            pairs = 0
            for sentence in sentences:
                indices = [self.word_to_index[word] for word in sentence]
                
                for i, target_idx in enumerate(indices):
                    context_indices = indices[max(0, i-self.window_size):i] + \
                                    indices[i+1:i+1+self.window_size]
                    
                    for context_idx in context_indices:
                        self.train_pair(target_idx, context_idx)
                        pairs += 1
            
            if epoch % 1 == 0:
                print(f"Epoch {epoch+1}, Processed {pairs} pairs")

# Example usage
sentences = [
    "the quick brown fox jumps over the lazy dog".split(),
    "the fox is quick and brown".split(),
    "the dog is lazy".split()
]

model = Word2Vec(dimensions=50, window_size=2)
model.build_vocabulary(sentences)
model.train(sentences, epochs=100)
```

Slide 14: Deep Clustering Neural Network

Deep clustering combines representation learning with clustering objectives, simultaneously learning feature representations and cluster assignments through an end-to-end trainable architecture.

```python
def deep_clustering_network(X, n_clusters, encoding_dim=32, epochs=100):
    n_samples, n_features = X.shape
    
    # Initialize autoencoder weights
    W_encoder = np.random.randn(n_features, encoding_dim) * 0.01
    b_encoder = np.zeros(encoding_dim)
    W_decoder = np.random.randn(encoding_dim, n_features) * 0.01
    b_decoder = np.zeros(n_features)
    
    # Initialize clustering layer
    cluster_centers = np.random.randn(n_clusters, encoding_dim) * 0.01
    
    def encode(X):
        return np.tanh(np.dot(X, W_encoder) + b_encoder)
    
    def decode(H):
        return np.tanh(np.dot(H, W_decoder) + b_decoder)
    
    def soft_assignment(H):
        # Student's t-distribution kernel
        alpha = 1.0  # degrees of freedom
        distances = np.sum((H[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :]) ** 2, axis=2)
        q = 1.0 / (1.0 + distances / alpha)
        q = q ** ((alpha + 1.0) / 2.0)
        q = q / q.sum(axis=1, keepdims=True)
        return q
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        H = encode(X)
        X_recon = decode(H)
        Q = soft_assignment(H)
        
        # Update cluster centers
        numerator = np.dot(Q.T, H)
        denominator = Q.sum(axis=0, keepdims=True).T
        cluster_centers = numerator / denominator
        
        # Compute reconstruction loss
        recon_loss = np.mean((X - X_recon) ** 2)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Reconstruction Loss: {recon_loss:.4f}")
    
    # Final cluster assignments
    H = encode(X)
    Q = soft_assignment(H)
    clusters = np.argmax(Q, axis=1)
    
    return clusters, H, cluster_centers

# Example usage
X = np.random.randn(500, 20)  # 500 samples, 20 features
clusters, embeddings, centers = deep_clustering_network(X, n_clusters=5)
```

Slide 15: Additional Resources

*   Variational Autoencoders: [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
*   Deep Clustering: [https://arxiv.org/abs/1511.06335](https://arxiv.org/abs/1511.06335)
*   Density-Based Clustering: [https://arxiv.org/abs/1608.03165](https://arxiv.org/abs/1608.03165)
*   Modern Word Embeddings: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
*   Unsupervised Learning Survey: [https://www.cs.cornell.edu/courses/cs6785/papers/unsupervised-learning-survey.pdf](https://www.cs.cornell.edu/courses/cs6785/papers/unsupervised-learning-survey.pdf)

