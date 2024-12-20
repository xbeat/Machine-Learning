## Visualizing K-Means Clustering with Animation
Slide 1: Introduction to K-Means Clustering

K-Means is a popular unsupervised machine learning algorithm used for clustering data points into groups based on similarity. This animation explores the step-by-step process of K-Means, providing an intuitive understanding of how it works.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 2)

# Initialize and fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title('K-Means Clustering')
plt.show()
```

Slide 2: Data Preparation

Before applying K-Means, we need to prepare our data. This involves selecting relevant features and scaling them appropriately to ensure all dimensions contribute equally to the clustering process.

```python
from sklearn.preprocessing import StandardScaler

# Sample data
data = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

print("Original data:\n", data)
print("\nScaled data:\n", scaled_data)
```

Slide 3: Choosing the Number of Clusters

Determining the optimal number of clusters is crucial. The elbow method is a common approach, where we plot the within-cluster sum of squares (WCSS) against the number of clusters and look for an "elbow" in the curve.

```python
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```

Slide 4: Initializing Centroids

K-Means begins by randomly initializing K centroids, where K is the chosen number of clusters. The 'k-means++' initialization method is often preferred as it helps in choosing centroids that are far apart from each other.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
X = np.random.rand(100, 2)

# Initialize centroids using k-means++
k = 3
centroids = [X[np.random.choice(X.shape[0])]]

for _ in range(1, k):
    dist = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in X])
    probs = dist/dist.sum()
    cumprobs = probs.cumsum()
    r = np.random.rand()
    centroids.append(X[np.argmax(cumprobs > r)])

# Plot data and centroids
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.scatter([c[0] for c in centroids], [c[1] for c in centroids], c='red', s=100, marker='x')
plt.title('K-Means++ Initialization')
plt.show()
```

Slide 5: Assigning Points to Clusters

After initializing centroids, each data point is assigned to the nearest centroid based on Euclidean distance. This step creates initial clusters.

```python
import numpy as np

def assign_clusters(X, centroids):
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

# Assume X and centroids are defined from previous slide
clusters = assign_clusters(X, np.array(centroids))

# Visualize the assignments
colors = ['r', 'g', 'b']
for i in range(k):
    cluster_points = X[clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], alpha=0.5)
plt.scatter([c[0] for c in centroids], [c[1] for c in centroids], c='black', s=100, marker='x')
plt.title('Initial Cluster Assignments')
plt.show()
```

Slide 6: Updating Centroids

After assigning points to clusters, the centroids are updated by calculating the mean of all points in each cluster. This process moves the centroids to the center of their respective clusters.

```python
def update_centroids(X, clusters, k):
    return np.array([X[clusters == i].mean(axis=0) for i in range(k)])

# Update centroids
new_centroids = update_centroids(X, clusters, k)

# Visualize the update
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.scatter([c[0] for c in centroids], [c[1] for c in centroids], c='red', s=100, marker='x', label='Old Centroids')
plt.scatter(new_centroids[:, 0], new_centroids[:, 1], c='black', s=100, marker='x', label='New Centroids')
plt.legend()
plt.title('Centroid Update')
plt.show()
```

Slide 7: Iterative Process

K-Means iterates between assigning points to clusters and updating centroids until convergence. Convergence occurs when centroids no longer move significantly or a maximum number of iterations is reached.

```python
def kmeans(X, k, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iters):
        old_centroids = centroids.copy()
        clusters = assign_clusters(X, centroids)
        centroids = update_centroids(X, clusters, k)
        if np.all(old_centroids == centroids):
            break
    return clusters, centroids

# Run K-Means
final_clusters, final_centroids = kmeans(X, k)

# Visualize final result
plt.scatter(X[:, 0], X[:, 1], c=final_clusters, cmap='viridis', alpha=0.5)
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', s=100, marker='x')
plt.title('Final K-Means Clustering')
plt.show()
```

Slide 8: Handling Outliers

K-Means can be sensitive to outliers. One way to mitigate this is by using the K-Medoids algorithm, which uses actual data points as cluster centers instead of mean values.

```python
from sklearn_extra.cluster import KMedoids

# Generate data with outliers
X = np.concatenate([np.random.randn(95, 2), np.random.randn(5, 2) * 5 + 10])

# Apply K-Medoids
kmedoids = KMedoids(n_clusters=3, random_state=42)
clusters = kmedoids.fit_predict(X)

# Visualize results
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], c='red', s=100, marker='x')
plt.title('K-Medoids Clustering with Outliers')
plt.show()
```

Slide 9: Evaluating Cluster Quality

Silhouette score is a metric used to evaluate the quality of clusters. It measures how similar an object is to its own cluster compared to other clusters. Values range from -1 to 1, with higher values indicating better-defined clusters.

```python
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Calculate silhouette scores for different numbers of clusters
silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X)
    score = silhouette_score(X, clusters)
    silhouette_scores.append(score)

# Plot silhouette scores
plt.plot(range(2, 10), silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for K-Means Clustering')
plt.show()
```

Slide 10: Real-Life Example: Image Compression

K-Means can be used for image compression by reducing the number of colors in an image. Each pixel is treated as a data point in 3D space (RGB values), and K-Means clusters these points into a smaller set of colors.

```python
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np

# Load image
image = Image.open('sample_image.jpg')
image_array = np.array(image)

# Reshape the image
width, height, channels = image_array.shape
pixels = image_array.reshape(-1, channels)

# Apply K-Means
n_colors = 16
kmeans = KMeans(n_clusters=n_colors, random_state=42)
labels = kmeans.fit_predict(pixels)

# Create new image with reduced colors
new_colors = kmeans.cluster_centers_.astype('uint8')
compressed_image = new_colors[labels].reshape(width, height, channels)

# Display original and compressed images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image)
ax1.set_title('Original Image')
ax2.imshow(Image.fromarray(compressed_image))
ax2.set_title(f'Compressed Image ({n_colors} colors)')
plt.show()
```

Slide 11: Real-Life Example: Customer Segmentation

K-Means is widely used in marketing for customer segmentation. By clustering customers based on various attributes, businesses can tailor their strategies to different customer groups.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load customer data (assume we have a CSV file)
data = pd.read_csv('customer_data.csv')

# Select relevant features
features = ['Age', 'Annual Income', 'Spending Score']
X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize results (3D scatter plot)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X['Age'], X['Annual Income'], X['Spending Score'], c=data['Cluster'], cmap='viridis')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income')
ax.set_zlabel('Spending Score')
plt.title('Customer Segments')
plt.colorbar(scatter)
plt.show()
```

Slide 12: Limitations of K-Means

While K-Means is powerful, it has limitations. It assumes clusters are spherical and equally sized, which isn't always true in real-world data. It's also sensitive to the initial placement of centroids.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans

# Generate non-spherical data
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Apply K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)

# Visualize results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title('K-Means Limitation: Non-Spherical Clusters')
plt.show()
```

Slide 13: Variations and Extensions

Several variations of K-Means address its limitations. Mini-Batch K-Means is faster for large datasets, while Gaussian Mixture Models can handle elliptical clusters. DBSCAN is better for arbitrary-shaped clusters and can detect outliers.

```python
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.mixture import GaussianMixture

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=[1.0, 2.5, 0.5, 1.5], random_state=42)

# Apply different clustering algorithms
algorithms = [
    ('K-Means', KMeans(n_clusters=4)),
    ('Mini-Batch K-Means', MiniBatchKMeans(n_clusters=4)),
    ('Gaussian Mixture Model', GaussianMixture(n_components=4)),
    ('DBSCAN', DBSCAN(eps=0.5, min_samples=5))
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for (name, algorithm), ax in zip(algorithms, axes.ravel()):
    clusters = algorithm.fit_predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
    ax.set_title(name)

plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into K-Means and clustering algorithms, here are some valuable resources:

1.  "K-Means Clustering via Principal Component Analysis" by Chris Ding and Xiaofeng He (2004). Available at: [https://arxiv.org/abs/cs/0408031](https://arxiv.org/abs/cs/0408031)
2.  "A Tutorial on Spectral Clustering" by Ulrike von Luxburg (2007). Available at: [https://arxiv.org/abs/0711.0189](https://arxiv.org/abs/0711.0189)
3.  "Mini-Batch K-Means Clustering" by David Sculley (2010). Available at: [https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf](https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf)

These papers provide in-depth discussions on various aspects of clustering algorithms, including improvements and extensions to the basic K-Means algorithm.

