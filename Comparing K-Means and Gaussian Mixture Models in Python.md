## Comparing K-Means and Gaussian Mixture Models in Python
Slide 1: Introduction to Clustering

Clustering is an unsupervised learning technique used to group similar data points together. Two popular clustering algorithms are K-Means and Gaussian Mixture Models (GMM). This presentation will explore these methods, their implementation in Python, and their practical applications.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Generate sample data
np.random.seed(42)
X = np.concatenate([
    np.random.normal(0, 1, (100, 2)),
    np.random.normal(3, 1.5, (100, 2))
])

plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.title("Sample Data for Clustering")
plt.show()
```

Slide 2: K-Means Algorithm

K-Means is a centroid-based algorithm that aims to partition n observations into k clusters. It iteratively assigns data points to the nearest centroid and updates the centroid positions until convergence.

```python
# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='r')
plt.title("K-Means Clustering Result")
plt.show()
```

Slide 3: K-Means Implementation

Let's implement a simple version of K-Means from scratch to understand its core principles.

```python
def kmeans_simple(X, k, max_iters=100):
    # Randomly initialize centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids

# Apply our simple K-Means
simple_labels, simple_centroids = kmeans_simple(X, k=2)

plt.scatter(X[:, 0], X[:, 1], c=simple_labels, cmap='viridis', alpha=0.7)
plt.scatter(simple_centroids[:, 0], simple_centroids[:, 1], 
            marker='x', s=200, linewidths=3, color='r')
plt.title("Simple K-Means Implementation Result")
plt.show()
```

Slide 4: Gaussian Mixture Models (GMM)

GMM is a probabilistic model that assumes data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. It's more flexible than K-Means as it allows for clusters of different shapes and sizes.

```python
# Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=42)
gmm_labels = gmm.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', alpha=0.7)
plt.title("Gaussian Mixture Model Clustering Result")
plt.show()
```

Slide 5: GMM Implementation

Let's implement a simplified version of GMM to understand its core concepts.

```python
from scipy.stats import multivariate_normal

def gmm_simple(X, k, max_iters=100):
    n, d = X.shape
    
    # Initialize parameters
    weights = np.ones(k) / k
    means = X[np.random.choice(n, k, replace=False)]
    covs = [np.eye(d) for _ in range(k)]
    
    for _ in range(max_iters):
        # E-step: Compute responsibilities
        resp = np.zeros((n, k))
        for j in range(k):
            resp[:, j] = weights[j] * multivariate_normal.pdf(X, means[j], covs[j])
        resp /= resp.sum(axis=1, keepdims=True)
        
        # M-step: Update parameters
        Nk = resp.sum(axis=0)
        weights = Nk / n
        means = np.dot(resp.T, X) / Nk[:, np.newaxis]
        for j in range(k):
            diff = X - means[j]
            covs[j] = np.dot(resp[:, j] * diff.T, diff) / Nk[j]
    
    labels = resp.argmax(axis=1)
    return labels, means, covs

# Apply our simple GMM
simple_gmm_labels, simple_gmm_means, _ = gmm_simple(X, k=2)

plt.scatter(X[:, 0], X[:, 1], c=simple_gmm_labels, cmap='viridis', alpha=0.7)
plt.scatter(simple_gmm_means[:, 0], simple_gmm_means[:, 1], 
            marker='x', s=200, linewidths=3, color='r')
plt.title("Simple GMM Implementation Result")
plt.show()
```

Slide 6: Comparing K-Means and GMM

K-Means and GMM have different strengths and weaknesses. K-Means assumes spherical clusters of similar size, while GMM can model elliptical clusters of varying sizes and orientations.

```python
# Generate data with different cluster shapes
np.random.seed(42)
X1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 3]], 200)
X2 = np.random.multivariate_normal([4, 4], [[1.5, 0], [0, 0.5]], 200)
X = np.vstack((X1, X2))

# Apply K-Means and GMM
kmeans = KMeans(n_clusters=2, random_state=42)
gmm = GaussianMixture(n_components=2, random_state=42)

kmeans_labels = kmeans.fit_predict(X)
gmm_labels = gmm.fit_predict(X)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
ax1.set_title("K-Means Clustering")

ax2.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', alpha=0.7)
ax2.set_title("GMM Clustering")

plt.tight_layout()
plt.show()
```

Slide 7: Choosing the Number of Clusters

Determining the optimal number of clusters is a common challenge in clustering. The elbow method for K-Means and the Bayesian Information Criterion (BIC) for GMM are popular techniques for this purpose.

```python
from sklearn.metrics import silhouette_score

def plot_elbow_method(X, max_clusters):
    inertias = []
    silhouette_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(range(2, max_clusters + 1), inertias, marker='o')
    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method for K-Means")
    
    ax2.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    ax2.set_xlabel("Number of clusters")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    
    plt.tight_layout()
    plt.show()

plot_elbow_method(X, 10)
```

Slide 8: Bayesian Information Criterion (BIC) for GMM

The BIC is a criterion for model selection that balances the model's likelihood with its complexity. Lower BIC values indicate better models.

```python
def plot_bic(X, max_components):
    n_components_range = range(1, max_components + 1)
    bic = []
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(X)
        bic.append(gmm.bic(X))
    
    plt.plot(n_components_range, bic, marker='o')
    plt.xlabel("Number of components")
    plt.ylabel("BIC")
    plt.title("BIC Score vs. Number of GMM Components")
    plt.show()

plot_bic(X, 10)
```

Slide 9: Handling High-Dimensional Data

Both K-Means and GMM can struggle with high-dimensional data due to the curse of dimensionality. Dimensionality reduction techniques like PCA can be helpful in such cases.

```python
from sklearn.decomposition import PCA

# Generate high-dimensional data
np.random.seed(42)
X_high_dim = np.random.randn(500, 50)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_high_dim)

# Cluster the reduced data
kmeans = KMeans(n_clusters=3, random_state=42)
gmm = GaussianMixture(n_components=3, random_state=42)

kmeans_labels = kmeans.fit_predict(X_reduced)
gmm_labels = gmm.fit_predict(X_reduced)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
ax1.set_title("K-Means on PCA-reduced Data")

ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], c=gmm_labels, cmap='viridis', alpha=0.7)
ax2.set_title("GMM on PCA-reduced Data")

plt.tight_layout()
plt.show()
```

Slide 10: Dealing with Outliers

K-Means is sensitive to outliers, while GMM can be more robust. Let's compare their performance on data with outliers.

```python
# Generate data with outliers
np.random.seed(42)
X_clean = np.random.randn(200, 2)
X_outliers = np.random.uniform(low=-5, high=5, size=(20, 2))
X_with_outliers = np.vstack((X_clean, X_outliers))

# Apply K-Means and GMM
kmeans = KMeans(n_clusters=2, random_state=42)
gmm = GaussianMixture(n_components=2, random_state=42)

kmeans_labels = kmeans.fit_predict(X_with_outliers)
gmm_labels = gmm.fit_predict(X_with_outliers)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
ax1.set_title("K-Means with Outliers")

ax2.scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], c=gmm_labels, cmap='viridis', alpha=0.7)
ax2.set_title("GMM with Outliers")

plt.tight_layout()
plt.show()
```

Slide 11: Real-Life Example: Image Segmentation

Image segmentation is a common application of clustering algorithms. Let's use K-Means to segment an image based on color.

```python
from sklearn.cluster import KMeans
from PIL import Image

# Load and preprocess the image
image = Image.open("sample_image.jpg")
image_array = np.array(image)
pixels = image_array.reshape(-1, 3)

# Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(pixels)

# Create segmented image
segmented_pixels = kmeans.cluster_centers_[labels]
segmented_image = segmented_pixels.reshape(image_array.shape).astype(np.uint8)

# Display results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image)
ax1.set_title("Original Image")
ax1.axis('off')

ax2.imshow(segmented_image)
ax2.set_title("Segmented Image (K-Means)")
ax2.axis('off')

plt.tight_layout()
plt.show()
```

Slide 12: Real-Life Example: Customer Segmentation

Customer segmentation is another common application of clustering algorithms. Let's use GMM to segment customers based on their purchasing behavior.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Generate sample customer data
np.random.seed(42)
n_customers = 1000

data = pd.DataFrame({
    'Age': np.random.normal(40, 15, n_customers),
    'Income': np.random.lognormal(10, 1, n_customers),
    'SpendingScore': np.random.normal(50, 25, n_customers)
})

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Apply GMM
gmm = GaussianMixture(n_components=4, random_state=42)
labels = gmm.fit_predict(X_scaled)

# Visualize results
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(data['Age'], data['Income'], data['SpendingScore'], 
                     c=labels, cmap='viridis', alpha=0.7)

ax.set_xlabel('Age')
ax.set_ylabel('Income')
ax.set_zlabel('Spending Score')
ax.set_title('Customer Segmentation using GMM')

plt.colorbar(scatter)
plt.show()
```

Slide 13: Comparing Computational Complexity

K-Means and GMM have different computational complexities, which can affect their performance on large datasets. Let's compare their runtime on datasets of increasing size.

```python
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def compare_runtime(n_samples_list, n_clusters=3, n_features=2):
    kmeans_times = []
    gmm_times = []
    
    for n_samples in n_samples_list:
        X = np.random.randn(n_samples, n_features)
        
        # K-Means
        start = time.time()
        KMeans(n_clusters=n_clusters).fit(X)
        kmeans_times.append(time.time() - start)
        
        # GMM
        start = time.time()
        GaussianMixture(n_components=n_clusters).fit(X)
        gmm_times.append(time.time() - start)
    
    return kmeans_times, gmm_times

n_samples_list = [1000, 5000, 10000, 50000, 100000]
kmeans_times, gmm_times = compare_runtime(n_samples_list)

plt.figure(figsize=(10, 6))
plt.plot(n_samples_list, kmeans_times, marker='o', label='K-Means')
plt.plot(n_samples_list, gmm_times, marker='s', label='GMM')
plt.xlabel('Number of Samples')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime Comparison: K-Means vs GMM')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.show()
```

Slide 14: Strengths and Weaknesses

K-Means and GMM each have their own strengths and weaknesses, making them suitable for different scenarios.

K-Means:

* Strengths: Simple, fast, and memory-efficient
* Weaknesses: Assumes spherical clusters, sensitive to initialization and outliers

GMM:

* Strengths: Flexible cluster shapes, provides probability of cluster membership
* Weaknesses: More complex, slower, and sensitive to initialization

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Generate non-spherical data
np.random.seed(42)
X1 = np.random.multivariate_normal([0, 0], [[2, 0], [0, 0.5]], 200)
X2 = np.random.multivariate_normal([3, 3], [[1, -0.5], [-0.5, 1]], 200)
X = np.vstack((X1, X2))

# Apply K-Means and GMM
kmeans = KMeans(n_clusters=2, random_state=42)
gmm = GaussianMixture(n_components=2, random_state=42)

kmeans_labels = kmeans.fit_predict(X)
gmm_labels = gmm.fit_predict(X)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
ax1.set_title("K-Means Clustering")

ax2.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', alpha=0.7)
ax2.set_title("GMM Clustering")

plt.tight_layout()
plt.show()
```

Slide 15: Conclusion and Best Practices

When choosing between K-Means and GMM, consider:

1. Dataset characteristics (cluster shapes, sizes)
2. Computational resources available
3. Need for probabilistic assignments
4. Domain knowledge about the data

Best practices:

* Preprocess and normalize data
* Try multiple initializations
* Use appropriate techniques to choose the number of clusters
* Validate results using domain expertise

```python
# Pseudocode for a general clustering approach
def cluster_data(X, method='kmeans', n_clusters=3):
    # Preprocess data
    X_scaled = preprocess_and_normalize(X)
    
    # Choose clustering method
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters)
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters)
    
    # Fit model with multiple initializations
    best_model = fit_with_multiple_inits(model, X_scaled)
    
    # Get cluster assignments
    labels = best_model.predict(X_scaled)
    
    # Validate results
    validate_clustering(X, labels)
    
    return labels, best_model
```

Slide 16: Additional Resources

For more in-depth information on clustering algorithms and their applications, consider exploring these resources:

1. ArXiv paper on K-Means variations: "k-means++: The Advantages of Careful Seeding" by Arthur and Vassilvitskii (2007) URL: [https://arxiv.org/abs/0711.4246](https://arxiv.org/abs/0711.4246)
2. ArXiv paper on Gaussian Mixture Models: "Variational Inference for Gaussian Mixture Models" by Bishop (2002) URL: [https://arxiv.org/abs/1206.5295](https://arxiv.org/abs/1206.5295)
3. Scikit-learn documentation on clustering algorithms: [https://scikit-learn.org/stable/modules/clustering.html](https://scikit-learn.org/stable/modules/clustering.html)
4. Book: "Pattern Recognition and Machine Learning" by Christopher M. Bishop, which covers both K-Means and GMM in detail.

