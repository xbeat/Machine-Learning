## Time Complexity of K-Means Clustering in Python
Slide 1: K-means Clustering: Time Complexity Analysis

K-means clustering is a popular unsupervised machine learning algorithm used for partitioning a dataset into K distinct, non-overlapping subgroups or clusters. Understanding its time complexity is crucial for efficient implementation and scalability. Let's explore the algorithm's time complexity using Python examples.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 2)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title('K-means Clustering Example')
plt.show()
```

Slide 2: K-means Algorithm Overview

The K-means algorithm iteratively assigns data points to clusters and updates cluster centroids. The main steps include initialization, assignment, and update. Let's implement a simple version of K-means to understand its core components.

```python
def kmeans(X, k, max_iters=100):
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

# Example usage
X = np.random.rand(100, 2)
labels, centroids = kmeans(X, k=3)
```

Slide 3: Time Complexity: Initialization

The initialization step involves selecting K random points as initial centroids. This process has a time complexity of O(K), where K is the number of clusters.

```python
def initialize_centroids(X, k):
    n_samples = X.shape[0]
    centroid_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[centroid_indices]
    return centroids

# Example usage
X = np.random.rand(1000, 2)
k = 5
initial_centroids = initialize_centroids(X, k)
print(f"Shape of initial centroids: {initial_centroids.shape}")
```

Slide 4: Time Complexity: Assignment Step

The assignment step calculates the distance between each data point and all centroids, then assigns each point to the nearest centroid. This step has a time complexity of O(n \* K \* d), where n is the number of data points, K is the number of clusters, and d is the number of dimensions.

```python
def assign_clusters(X, centroids):
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)
    return labels

# Example usage
X = np.random.rand(1000, 2)
centroids = np.random.rand(5, 2)
labels = assign_clusters(X, centroids)
print(f"Number of points in each cluster: {np.bincount(labels)}")
```

Slide 5: Time Complexity: Update Step

The update step recalculates the centroids based on the mean of all points assigned to each cluster. This step has a time complexity of O(n \* d), where n is the number of data points and d is the number of dimensions.

```python
def update_centroids(X, labels, k):
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return centroids

# Example usage
X = np.random.rand(1000, 2)
labels = np.random.randint(0, 5, 1000)
k = 5
new_centroids = update_centroids(X, labels, k)
print(f"Shape of updated centroids: {new_centroids.shape}")
```

Slide 6: Overall Time Complexity

The overall time complexity of K-means is O(n \* K \* d \* I), where:

* n: number of data points
* K: number of clusters
* d: number of dimensions
* I: number of iterations

This complexity arises from repeating the assignment and update steps for I iterations. Let's visualize how the execution time changes with different parameters.

```python
import time

def measure_kmeans_time(n, k, d, max_iters):
    X = np.random.rand(n, d)
    start_time = time.time()
    kmeans(X, k, max_iters)
    end_time = time.time()
    return end_time - start_time

n_values = [1000, 2000, 4000, 8000]
times = [measure_kmeans_time(n, k=5, d=2, max_iters=100) for n in n_values]

plt.plot(n_values, times, marker='o')
plt.xlabel('Number of data points (n)')
plt.ylabel('Execution time (seconds)')
plt.title('K-means Execution Time vs. Number of Data Points')
plt.show()
```

Slide 7: Impact of Number of Clusters (K)

The number of clusters (K) significantly affects the time complexity. Let's examine how increasing K impacts the execution time while keeping other parameters constant.

```python
k_values = [2, 4, 8, 16, 32]
times = [measure_kmeans_time(n=5000, k=k, d=2, max_iters=100) for k in k_values]

plt.plot(k_values, times, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Execution time (seconds)')
plt.title('K-means Execution Time vs. Number of Clusters')
plt.show()
```

Slide 8: Impact of Number of Dimensions (d)

The number of dimensions (d) also affects the time complexity. Let's visualize how increasing d impacts the execution time while keeping other parameters constant.

```python
d_values = [2, 4, 8, 16, 32]
times = [measure_kmeans_time(n=5000, k=5, d=d, max_iters=100) for d in d_values]

plt.plot(d_values, times, marker='o')
plt.xlabel('Number of dimensions (d)')
plt.ylabel('Execution time (seconds)')
plt.title('K-means Execution Time vs. Number of Dimensions')
plt.show()
```

Slide 9: Optimizing K-means: The Elkan Algorithm

The Elkan algorithm is an optimized version of K-means that reduces the number of distance calculations, potentially improving time complexity. It uses triangle inequality to avoid unnecessary distance computations.

```python
from sklearn.cluster import KMeans

def elkan_kmeans(X, k, max_iters=100):
    kmeans = KMeans(n_clusters=k, algorithm='elkan', max_iter=max_iters, n_init=1)
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_

# Compare standard K-means with Elkan K-means
X = np.random.rand(10000, 10)
k = 5

start_time = time.time()
kmeans(X, k, max_iters=100)
standard_time = time.time() - start_time

start_time = time.time()
elkan_kmeans(X, k, max_iters=100)
elkan_time = time.time() - start_time

print(f"Standard K-means time: {standard_time:.4f} seconds")
print(f"Elkan K-means time: {elkan_time:.4f} seconds")
print(f"Speedup: {standard_time / elkan_time:.2f}x")
```

Slide 10: Real-Life Example: Image Compression

K-means clustering can be used for image compression by reducing the number of colors in an image. Let's implement a simple image compressor using K-means.

```python
from PIL import Image

def compress_image(image_path, k):
    # Load image and convert to numpy array
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Reshape the image to 2D array of pixels
    pixels = img_array.reshape(-1, 3)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    
    # Replace each pixel with its nearest centroid
    compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
    
    # Reshape back to original image shape
    compressed_img_array = compressed_pixels.reshape(img_array.shape)
    
    # Convert to uint8 and create new image
    compressed_img = Image.fromarray(compressed_img_array.astype('uint8'))
    return compressed_img

# Example usage
original_image_path = 'path_to_your_image.jpg'
compressed_image = compress_image(original_image_path, k=16)
compressed_image.save('compressed_image.jpg')

# Display original and compressed images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(Image.open(original_image_path))
ax1.set_title('Original Image')
ax2.imshow(compressed_image)
ax2.set_title('Compressed Image (16 colors)')
plt.show()
```

Slide 11: Real-Life Example: Customer Segmentation

K-means clustering is widely used in customer segmentation to group customers based on their behavior or characteristics. Let's implement a simple customer segmentation example.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Generate sample customer data
np.random.seed(42)
n_customers = 1000
age = np.random.randint(18, 70, n_customers)
income = np.random.randint(20000, 200000, n_customers)
spending_score = np.random.randint(1, 100, n_customers)

df = pd.DataFrame({
    'Age': age,
    'Income': income,
    'SpendingScore': spending_score
})

# Normalize the features
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df)

# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_normalized)

# Visualize the results
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['Age'], df['Income'], df['SpendingScore'], c=df['Cluster'], cmap='viridis')
ax.set_xlabel('Age')
ax.set_ylabel('Income')
ax.set_zlabel('Spending Score')
plt.title('Customer Segmentation using K-means')
plt.colorbar(scatter)
plt.show()

# Print cluster statistics
print(df.groupby('Cluster').mean())
```

Slide 12: Challenges and Limitations

While K-means is widely used, it has some limitations:

1. Sensitivity to initial centroids: The algorithm may converge to local optima.
2. Predefined number of clusters: Determining the optimal K can be challenging.
3. Assumes spherical clusters: K-means may perform poorly on non-spherical or uneven cluster sizes.

Let's visualize these limitations using a simple example.

```python
from sklearn.datasets import make_blobs, make_moons

# Generate datasets
n_samples = 1000
blob_centers = [(0, 0), (5, 5), (0, 5)]
X_blobs, _ = make_blobs(n_samples=n_samples, centers=blob_centers, cluster_std=0.7)
X_moons, _ = make_moons(n_samples=n_samples, noise=0.1)

# Perform K-means clustering
kmeans_blobs = KMeans(n_clusters=3, random_state=42)
kmeans_moons = KMeans(n_clusters=2, random_state=42)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_blobs[:, 0], X_blobs[:, 1], c=kmeans_blobs.fit_predict(X_blobs), cmap='viridis')
ax1.set_title('K-means on Blob Dataset')

ax2.scatter(X_moons[:, 0], X_moons[:, 1], c=kmeans_moons.fit_predict(X_moons), cmap='viridis')
ax2.set_title('K-means on Moon Dataset')

plt.show()
```

Slide 13: Improving K-means: The K-means++ Initialization

K-means++ is an initialization method that aims to choose better initial centroids, potentially leading to faster convergence and better clustering results. Let's compare standard K-means with K-means++.

```python
from sklearn.cluster import KMeans

# Generate sample data
X, _ = make_blobs(n_samples=1000, centers=5, random_state=42)

# Standard K-means
kmeans_standard = KMeans(n_clusters=5, init='random', n_init=10, random_state=42)
kmeans_standard.fit(X)

# K-means++
kmeans_plus_plus = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
kmeans_plus_plus.fit(X)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], c=kmeans_standard.labels_, cmap='viridis')
ax1.scatter(kmeans_standard.cluster_centers_[:, 0], kmeans_standard.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
ax1.set_title('Standard K-means')

ax2.scatter(X[:, 0], X[:, 1], c=kmeans_plus_plus.labels_, cmap='viridis')
ax2.scatter(kmeans_plus_plus.cluster_centers_[:, 0], kmeans_plus_plus.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
ax2.set_title('K-means++')

plt.show()

print(f"Standard K-means inertia: {kmeans_standard.inertia_:.2f}")
print(f"K-means++ inertia: {kmeans_plus_plus.inertia_:.2f}")
```

Slide 14: Conclusion and Best Practices

To optimize K-means clustering:

1. Use K-means++ initialization for better initial centroids.
2. Normalize features to ensure equal weight.
3. Run multiple initializations to avoid local optima.
4. Use the elbow method or silhouette analysis to determine the optimal K.
5. Consider using mini-batch K-means for large datasets.

Here's an example implementing these best practices:

Slide 15: Conclusion and Best Practices

```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Generate sample data
X, _ = make_blobs(n_samples=1000, centers=5, random_state=42)

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Find optimal K using silhouette analysis
silhouette_scores = []
K_range = range(2, 11)

for K in K_range:
    kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_normalized)
    silhouette_scores.append(silhouette_score(X_normalized, kmeans.labels_))

optimal_K = K_range[silhouette_scores.index(max(silhouette_scores))]

# Fit final model with optimal K
final_kmeans = KMeans(n_clusters=optimal_K, init='k-means++', n_init=10, random_state=42)
final_kmeans.fit(X_normalized)

print(f"Optimal number of clusters: {optimal_K}")
print(f"Final model inertia: {final_kmeans.inertia_:.2f}")
```

Slide 16: Additional Resources

For further exploration of K-means clustering and its time complexity:

1. ArXiv paper on K-means++: "k-means++: The Advantages of Careful Seeding" URL: [https://arxiv.org/abs/0609164](https://arxiv.org/abs/0609164)
2. ArXiv paper on Mini-batch K-means: "Web-Scale K-Means Clustering" URL: [https://arxiv.org/abs/1006.4757](https://arxiv.org/abs/1006.4757)
3. ArXiv paper on Elkan's algorithm: "Using the Triangle Inequality to Accelerate k-Means" URL: [https://arxiv.org/abs/1203.1898](https://arxiv.org/abs/1203.1898)

These resources provide in-depth analysis and improvements to the K-means algorithm, focusing on time complexity and performance optimization.

