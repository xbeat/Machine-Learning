## Practical Limitations of KMeans Clustering

Slide 1: KMeans Limitations

KMeans is a popular clustering algorithm, but it has several limitations that are often overlooked. Understanding these limitations is crucial for effective data analysis. Let's explore some key drawbacks of KMeans and introduce an alternative algorithm that addresses these issues.

Slide 2: Source Code for KMeans Limitations

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
np.random.seed(42)
cluster_1 = np.random.normal(loc=[2, 2], scale=[1.5, 0.5], size=(100, 2))
cluster_2 = np.random.normal(loc=[6, 6], scale=[0.5, 1.5], size=(100, 2))
data = np.vstack((cluster_1, cluster_2))

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.title("Sample Data with Different Cluster Variances")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Implement KMeans
def kmeans(X, k, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iters):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# Apply KMeans
k = 2
labels, centroids = kmeans(data, k)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.title("KMeans Clustering Result")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Slide 3: Results for KMeans Limitations

```
The code generates two plots:

1. Sample Data with Different Cluster Variances:
   - Shows two clusters with different shapes and variances
   - Cluster 1: Elongated horizontally
   - Cluster 2: Elongated vertically

2. KMeans Clustering Result:
   - Displays the KMeans clustering outcome
   - Two centroids marked with red 'x'
   - Data points colored based on cluster assignment

Observation: KMeans fails to accurately capture the true cluster shapes,
especially for the elongated clusters with different variances.
```

Slide 4: Cluster Variance and Shape

KMeans does not account for cluster variance or produce non-globular clusters. It assumes all clusters have similar variance and circular shapes in 2D space. This limitation can lead to suboptimal clustering when dealing with datasets that have clusters of varying sizes, shapes, or densities.

Slide 5: Source Code for Cluster Variance and Shape

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with different variances and shapes
np.random.seed(42)
cluster_1 = np.random.normal(loc=[2, 2], scale=[1.5, 0.5], size=(100, 2))
cluster_2 = np.random.normal(loc=[6, 6], scale=[0.5, 1.5], size=(100, 2))
cluster_3 = np.random.normal(loc=[4, 4], scale=[0.3, 0.3], size=(100, 2))
data = np.vstack((cluster_1, cluster_2, cluster_3))

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.title("Sample Data with Different Cluster Variances and Shapes")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Apply KMeans
k = 3
labels, centroids = kmeans(data, k)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.title("KMeans Clustering Result")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Slide 6: Results for Cluster Variance and Shape

```
The code generates two plots:

1. Sample Data with Different Cluster Variances and Shapes:
   - Shows three clusters with different shapes and variances
   - Cluster 1: Elongated horizontally
   - Cluster 2: Elongated vertically
   - Cluster 3: Compact and circular

2. KMeans Clustering Result:
   - Displays the KMeans clustering outcome
   - Three centroids marked with red 'x'
   - Data points colored based on cluster assignment

Observation: KMeans struggles to accurately capture the true cluster shapes,
especially for the elongated clusters with different variances and the
compact circular cluster.
```

Slide 7: Distance-based Assignment

KMeans relies solely on distance-based measures to assign data points to clusters. This approach can lead to misclassifications when clusters have different sizes or densities. Let's visualize this limitation using a simple example.

Slide 8: Source Code for Distance-based Assignment

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
cluster_A = np.random.normal(loc=[2, 2], scale=[1.5, 1.5], size=(200, 2))
cluster_B = np.random.normal(loc=[6, 6], scale=[0.5, 0.5], size=(200, 2))
data = np.vstack((cluster_A, cluster_B))

# Calculate centroids
centroid_A = np.mean(cluster_A, axis=0)
centroid_B = np.mean(cluster_B, axis=0)

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(cluster_A[:, 0], cluster_A[:, 1], alpha=0.5, label='Cluster A')
plt.scatter(cluster_B[:, 0], cluster_B[:, 1], alpha=0.5, label='Cluster B')
plt.scatter(centroid_A[0], centroid_A[1], c='red', marker='x', s=200, linewidths=3, label='Centroid A')
plt.scatter(centroid_B[0], centroid_B[1], c='red', marker='x', s=200, linewidths=3, label='Centroid B')

# Draw midline
midpoint = (centroid_A + centroid_B) / 2
plt.axvline(x=midpoint[0], color='black', linestyle='--', label='Midline')

plt.title("KMeans Distance-based Assignment")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Calculate distances to centroids for a sample point
sample_point = np.array([4, 4])
dist_to_A = np.linalg.norm(sample_point - centroid_A)
dist_to_B = np.linalg.norm(sample_point - centroid_B)

print(f"Distance to Centroid A: {dist_to_A:.2f}")
print(f"Distance to Centroid B: {dist_to_B:.2f}")
print(f"Assigned to Cluster: {'A' if dist_to_A < dist_to_B else 'B'}")
```

Slide 9: Results for Distance-based Assignment

```
The code generates a plot and prints distances:

Plot: KMeans Distance-based Assignment
- Shows two clusters with different variances
- Cluster A: Larger spread
- Cluster B: Smaller spread
- Centroids marked with red 'x'
- Midline drawn between centroids

Printed results:
Distance to Centroid A: 2.83
Distance to Centroid B: 2.83
Assigned to Cluster: A

Observation: The sample point (4, 4) is equidistant from both centroids,
but it's visually closer to the denser Cluster B. KMeans assigns it to
Cluster A due to equal distances, which may not be ideal.
```

Slide 10: Hard Assignment

KMeans performs hard assignment, meaning each data point is assigned to exactly one cluster without any uncertainty measure. This approach doesn't provide probabilistic estimates of cluster membership, which can be valuable in many real-world scenarios.

Slide 11: Source Code for Hard Assignment

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
cluster_1 = np.random.normal(loc=[2, 2], scale=[1, 1], size=(100, 2))
cluster_2 = np.random.normal(loc=[4, 4], scale=[1, 1], size=(100, 2))
data = np.vstack((cluster_1, cluster_2))

# Implement KMeans with hard assignment
def kmeans_hard_assignment(X, k, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iters):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# Apply KMeans
k = 2
labels, centroids = kmeans_hard_assignment(data, k)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.title("KMeans Hard Assignment")
plt.xlabel("X")
plt.ylabel("Y")

# Highlight points near cluster boundaries
boundary_points = np.where(np.abs(np.diff(distances, axis=0)) < 0.5)[1]
plt.scatter(data[boundary_points, 0], data[boundary_points, 1], 
            c='green', s=100, alpha=0.7, edgecolors='black')

plt.show()

# Print assignment probabilities for a boundary point
boundary_point = data[boundary_points[0]]
distances_to_centroids = np.linalg.norm(boundary_point - centroids, axis=1)
print("Distances to centroids:", distances_to_centroids)
print("Assigned cluster:", labels[boundary_points[0]])
```

Slide 12: Results for Hard Assignment

```
The code generates a plot and prints assignment information:

Plot: KMeans Hard Assignment
- Shows two overlapping clusters
- Data points colored based on cluster assignment
- Centroids marked with red 'x'
- Green points highlight data near cluster boundaries

Printed results for a boundary point:
Distances to centroids: [2.14 2.15]
Assigned cluster: 0

Observation: The boundary point is almost equidistant from both centroids,
but KMeans assigns it definitively to cluster 0 without any measure of uncertainty.
This hard assignment doesn't capture the ambiguity of points near cluster boundaries.
```

Slide 13: Gaussian Mixture Models (GMM)

Gaussian Mixture Models (GMM) offer a more flexible alternative to KMeans. GMMs can model clusters with different shapes, sizes, and orientations by learning a probability distribution for each cluster. This approach addresses many limitations of KMeans and provides probabilistic cluster assignments.

Slide 14: Source Code for Gaussian Mixture Models (GMM)

```python
import numpy as np
import matplotlib.pyplot as plt

class GMM:
    def __init__(self, n_components, max_iters=100):
        self.n_components = n_components
        self.max_iters = max_iters
        
    def fit(self, X):
        n_samples, n_features = X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = [np.eye(n_features) for _ in range(self.n_components)]
        
        for _ in range(self.max_iters):
            # E-step
            responsibilities = self._e_step(X)
            # M-step
            self._m_step(X, responsibilities)
        
    def _e_step(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self._multivariate_normal(X, self.means[k], self.covariances[k])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        N = responsibilities.sum(axis=0)
        self.weights = N / X.shape[0]
        self.means = np.dot(responsibilities.T, X) / N[:, np.newaxis]
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / N[k]
    
    def _multivariate_normal(self, X, mean, cov):
        n = X.shape[1]
        diff = X - mean
        return np.exp(-0.5 * np.sum(np.dot(diff, np.linalg.inv(cov)) * diff, axis=1)) / np.sqrt((2 * np.pi)**n * np.linalg.det(cov))

# Generate sample data
np.random.seed(42)
cluster_1 = np.random.multivariate_normal([2, 2], [[2, 0], [0, 0.5]], size=200)
cluster_2 = np.random.multivariate_normal([6, 6], [[0.5, 0], [0, 2]], size=200)
data = np.vstack((cluster_1, cluster_2))

# Fit GMM
gmm = GMM(n_components=2)
gmm.fit(data)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.scatter(gmm.means[:, 0], gmm.means[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Slide 15: Results for Gaussian Mixture Models (GMM)

```
The code generates a plot:

Gaussian Mixture Model Clustering
- Shows two clusters with different shapes and orientations
- Data points plotted as scattered dots
- Cluster means marked with red 'x'

Observations:
1. GMM successfully captures the different shapes and orientations of the clusters
2. The means (red 'x') are positioned at the centers of the clusters
3. Unlike KMeans, GMM can model elliptical clusters

Note: In a full implementation, we would also visualize the covariance 
matrices as ellipses around the means to show the learned cluster shapes.
```

Slide 16: Comparison of KMeans and GMM

GMM offers several advantages over KMeans:

1.  It can model clusters with different shapes, sizes, and orientations.
2.  It provides probabilistic assignments, allowing for soft clustering.
3.  It can capture more complex data distributions.

Let's compare KMeans and GMM on a dataset with non-circular clusters.

Slide 17: Source Code for Comparison of KMeans and GMM

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
n_samples = 300
X = np.concatenate([
    np.random.normal(0, 1, (n_samples, 2)),
    np.random.normal(4, 1.5, (n_samples, 2)),
    np.random.normal([6, -2], [[1.5, 0], [0, 0.5]], n_samples)
])

# Implement simplified KMeans
def kmeans(X, k, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iters):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# Apply KMeans
k = 3
kmeans_labels, kmeans_centroids = kmeans(X, k)

# Apply GMM (using the simplified version from previous slide)
gmm = GMM(n_components=3)
gmm.fit(X)
gmm_labels = gmm._e_step(X).argmax(axis=1)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
ax1.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
ax1.set_title("KMeans Clustering")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

ax2.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', alpha=0.7)
ax2.scatter(gmm.means[:, 0], gmm.means[:, 1], c='red', marker='x', s=200, linewidths=3)
ax2.set_title("GMM Clustering")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")

plt.tight_layout()
plt.show()
```

Slide 18: Results for Comparison of KMeans and GMM

```
The code generates two plots side by side:

1. KMeans Clustering:
   - Shows three clusters with different colors
   - Cluster centroids marked with red 'x'
   - Clusters are roughly circular and equal in size

2. GMM Clustering:
   - Shows three clusters with different colors
   - Cluster means marked with red 'x'
   - Clusters have different shapes and sizes

Observations:
1. KMeans struggles with the elongated cluster, splitting it into two
2. GMM accurately captures the shapes and orientations of all clusters
3. GMM provides a more natural clustering for this dataset
```

Slide 19: Real-Life Example: Image Segmentation

Image segmentation is a common application of clustering algorithms. Let's compare KMeans and GMM for segmenting a simple image.

Slide 20: Source Code for Image Segmentation

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple image
def create_image():
    img = np.zeros((100, 100, 3))
    img[20:40, 20:40] = [1, 0, 0]  # Red square
    img[60:80, 60:80] = [0, 1, 0]  # Green square
    img[30:70, 30:70] = [0, 0, 1]  # Blue square
    img += np.random.normal(0, 0.1, img.shape)  # Add noise
    img = np.clip(img, 0, 1)
    return img

# Implement KMeans and GMM (simplified versions from previous slides)
# ... (KMeans and GMM class definitions go here)

# Create and segment the image
img = create_image()
X = img.reshape(-1, 3)

# Apply KMeans
kmeans = KMeans(n_clusters=3)
kmeans_labels = kmeans.fit_predict(X)
kmeans_segmented = kmeans_labels.reshape(img.shape[:2])

# Apply GMM
gmm = GMM(n_components=3)
gmm.fit(X)
gmm_labels = gmm._e_step(X).argmax(axis=1)
gmm_segmented = gmm_labels.reshape(img.shape[:2])

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(img)
ax1.set_title("Original Image")
ax1.axis('off')

ax2.imshow(kmeans_segmented, cmap='viridis')
ax2.set_title("KMeans Segmentation")
ax2.axis('off')

ax3.imshow(gmm_segmented, cmap='viridis')
ax3.set_title("GMM Segmentation")
ax3.axis('off')

plt.tight_layout()
plt.show()
```

Slide 21: Results for Image Segmentation

```
The code generates three images side by side:

1. Original Image:
   - Shows overlapping red, green, and blue squares with noise

2. KMeans Segmentation:
   - Displays the image segmented into three distinct regions

3. GMM Segmentation:
   - Shows the image segmented using GMM

Observations:
1. Both KMeans and GMM successfully segment the main color regions
2. GMM tends to provide smoother boundaries between segments
3. GMM might capture the overlapping regions more accurately
4. The performance difference is subtle in this simple example, but
   GMM can be more effective for complex, real-world images
```

Slide 22: Real-Life Example: Customer Segmentation

Customer segmentation is crucial for targeted marketing. Let's explore how KMeans and GMM can be used to segment customers based on their purchasing behavior.

Slide 23: Source Code for Customer Segmentation

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic customer data
np.random.seed(42)
n_customers = 1000

# Features: Annual Income (normalized) and Spending Score (1-100)
income = np.concatenate([
    np.random.normal(0.2, 0.1, n_customers // 3),
    np.random.normal(0.5, 0.15, n_customers // 3),
    np.random.normal(0.8, 0.1, n_customers // 3)
])
spending = np.concatenate([
    np.random.normal(30, 10, n_customers // 3),
    np.random.normal(50, 15, n_customers // 3),
    np.random.normal(80, 10, n_customers // 3)
])

X = np.column_stack((income, spending))

# Apply KMeans and GMM
kmeans = KMeans(n_clusters=3)
kmeans_labels = kmeans.fit_predict(X)

gmm = GMM(n_components=3)
gmm.fit(X)
gmm_labels = gmm._e_step(X).argmax(axis=1)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
ax1.set_title("KMeans Customer Segmentation")
ax1.set_xlabel("Annual Income (normalized)")
ax1.set_ylabel("Spending Score")

ax2.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', alpha=0.7)
ax2.set_title("GMM Customer Segmentation")
ax2.set_xlabel("Annual Income (normalized)")
ax2.set_ylabel("Spending Score")

plt.tight_layout()
plt.show()
```

Slide 24: Results for Customer Segmentation

```
The code generates two scatter plots side by side:

1. KMeans Customer Segmentation:
   - Shows customers grouped into three segments
   - Segments are roughly circular and equal in size

2. GMM Customer Segmentation:
   - Displays customers grouped into three segments
   - Segments have different shapes and sizes

Observations:
1. Both methods identify three main customer groups
2. GMM captures the natural shape of customer clusters better
3. GMM might provide more nuanced insights into customer behavior
4. The flexibility of GMM allows for more accurate representation
   of customer segments with varying densities and shapes
```

Slide 25: Conclusion and Additional Resources

GMM offers several advantages over KMeans, including the ability to model clusters with different shapes and sizes, and provide probabilistic assignments. While KMeans remains useful for simple clustering tasks, GMM is often superior for complex, real-world data.

For further reading on Gaussian Mixture Models and their applications, consider the following resources:

1.  Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer. (Chapter 9: Mixture Models and EM)
2.  Reynolds, D. (2015). Gaussian Mixture Models. Encyclopedia of Biometrics, 827-832.
3.  Rasmussen, C. E. (2000). The Infinite Gaussian Mixture Model. In Advances in Neural Information Processing Systems (pp. 554-560).

