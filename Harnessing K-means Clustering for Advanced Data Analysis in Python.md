## Harnessing K-means Clustering for Advanced Data Analysis in Python
Slide 1: Introduction to K-means Clustering

K-means clustering is an unsupervised machine learning algorithm used for partitioning a dataset into K distinct, non-overlapping subgroups (clusters) of observations. Each observation belongs to the cluster with the nearest mean (centroid). This technique is widely used in data analysis, pattern recognition, and image segmentation.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title('K-means Clustering Example')
plt.show()
```

Slide 2: The K-means Algorithm

The K-means algorithm iteratively assigns data points to clusters and updates cluster centroids. It aims to minimize the within-cluster sum of squares (WCSS) or inertia. The algorithm steps include initialization, assignment, and update, repeated until convergence or a maximum number of iterations is reached.

```python
def kmeans(X, k, max_iters=100):
    # Initialize centroids randomly
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
X = np.random.randn(100, 2)
labels, centroids = kmeans(X, k=3)
print("Cluster labels:", labels)
print("Centroids:", centroids)
```

Slide 3: Choosing the Optimal Number of Clusters

Determining the optimal number of clusters is crucial for effective K-means clustering. The elbow method is a popular technique for this purpose. It involves plotting the WCSS against the number of clusters and identifying the "elbow" point where the rate of decrease sharply shifts.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def plot_elbow_curve(X, max_k):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, max_k + 1), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.show()

# Example usage
X = np.random.randn(200, 2)
plot_elbow_curve(X, max_k=10)
```

Slide 4: Feature Scaling for K-means

Feature scaling is essential for K-means clustering when dealing with features of different scales. Standardization (z-score normalization) is a common technique that transforms features to have zero mean and unit variance, ensuring that all features contribute equally to the clustering process.

```python
from sklearn.preprocessing import StandardScaler

# Generate sample data with different scales
np.random.seed(42)
X = np.random.randn(200, 2)
X[:, 1] *= 10  # Scale the second feature

# Perform K-means without scaling
kmeans_unscaled = KMeans(n_clusters=3, random_state=42)
labels_unscaled = kmeans_unscaled.fit_predict(X)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-means with scaling
kmeans_scaled = KMeans(n_clusters=3, random_state=42)
labels_scaled = kmeans_scaled.fit_predict(X_scaled)

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], c=labels_unscaled, cmap='viridis')
ax1.set_title('K-means without Scaling')

ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_scaled, cmap='viridis')
ax2.set_title('K-means with Scaling')

plt.show()
```

Slide 5: Handling Categorical Data

K-means clustering typically works with numerical data. To include categorical features, we need to transform them into numerical representations. One-hot encoding is a common technique for this purpose, creating binary columns for each category.

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Sample data with categorical features
data = pd.DataFrame({
    'Feature1': [1, 2, 3, 4, 5],
    'Category': ['A', 'B', 'A', 'C', 'B']
})

# One-hot encode the categorical feature
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(data[['Category']])

# Create new dataframe with encoded features
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names(['Category']))
result = pd.concat([data['Feature1'], encoded_df], axis=1)

print(result)
```

Slide 6: Dealing with High-Dimensional Data

K-means clustering can be challenging with high-dimensional data due to the curse of dimensionality. Dimensionality reduction techniques like Principal Component Analysis (PCA) can help visualize and improve clustering performance in such cases.

```python
from sklearn.decomposition import PCA

# Generate high-dimensional data
np.random.seed(42)
X_high_dim = np.random.randn(200, 50)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_high_dim)

# Perform K-means on reduced data
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_reduced)

# Plot the results
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis')
plt.title('K-means Clustering on PCA-reduced Data')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
```

Slide 7: Evaluating Cluster Quality

Assessing the quality of K-means clustering results is crucial. The silhouette score is a metric that measures how similar an object is to its own cluster compared to other clusters. It ranges from -1 to 1, where higher values indicate better-defined clusters.

```python
from sklearn.metrics import silhouette_score

def evaluate_kmeans(X, max_k):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    plt.plot(range(2, max_k + 1), silhouette_scores)
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.show()

# Example usage
X = np.random.randn(200, 2)
evaluate_kmeans(X, max_k=10)
```

Slide 8: Handling Outliers

Outliers can significantly impact K-means clustering results. One approach to mitigate their effect is to use the K-means++ initialization method, which chooses initial centroids to be distant from each other, reducing the impact of outliers on the final clustering.

```python
from sklearn.cluster import KMeans

# Generate sample data with outliers
np.random.seed(42)
X = np.random.randn(200, 2)
X = np.vstack([X, np.array([10, 10], [10, -10], [-10, 10], [-10, -10])])  # Add outliers

# Perform K-means with random initialization
kmeans_random = KMeans(n_clusters=4, init='random', n_init=10, random_state=42)
labels_random = kmeans_random.fit_predict(X)

# Perform K-means with K-means++ initialization
kmeans_plusplus = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels_plusplus = kmeans_plusplus.fit_predict(X)

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], c=labels_random, cmap='viridis')
ax1.set_title('K-means with Random Initialization')

ax2.scatter(X[:, 0], X[:, 1], c=labels_plusplus, cmap='viridis')
ax2.set_title('K-means with K-means++ Initialization')

plt.show()
```

Slide 9: Real-Life Example: Customer Segmentation

K-means clustering can be applied to customer segmentation, helping businesses understand their customer base and tailor marketing strategies. In this example, we'll cluster customers based on their annual income and spending score.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load sample customer data
data = pd.DataFrame({
    'Annual Income': [50, 75, 100, 25, 40, 60, 80, 120, 30, 70],
    'Spending Score': [30, 60, 80, 20, 40, 50, 70, 90, 25, 65]
})

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot the results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data['Annual Income'], data['Spending Score'], c=data['Cluster'], cmap='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.colorbar(scatter)
plt.show()

print(data)
```

Slide 10: Real-Life Example: Image Compression

K-means clustering can be used for image compression by reducing the number of colors in an image. This technique is particularly useful for images with a limited color palette.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io

# Load and reshape the image
image = io.imread('sample_image.jpg')
pixels = image.reshape(-1, 3)

# Perform K-means clustering
k = 16  # Number of colors
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(pixels)

# Create the compressed image
compressed_palette = kmeans.cluster_centers_.astype(int)
compressed_image = compressed_palette[labels].reshape(image.shape)

# Display original and compressed images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image)
ax1.set_title('Original Image')
ax1.axis('off')

ax2.imshow(compressed_image.astype(np.uint8))
ax2.set_title(f'Compressed Image ({k} colors)')
ax2.axis('off')

plt.show()
```

Slide 11: Limitations and Considerations

While K-means clustering is powerful, it has limitations. It assumes spherical clusters of similar size and density, which may not always hold true for real-world data. The algorithm is also sensitive to the initial placement of centroids and may converge to local optima.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans

# Generate non-spherical data
X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Perform K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_moons)

# Plot the results
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels, cmap='viridis')
plt.title('K-means Clustering on Non-Spherical Data')
plt.show()

# This example demonstrates how K-means can struggle with non-spherical clusters
```

Slide 12: Advanced Techniques: Mini-Batch K-means

For large datasets, the standard K-means algorithm can be computationally expensive. Mini-Batch K-means is a variant that uses mini-batches to reduce computation time while still attempting to optimize the same objective function.

```python
from sklearn.cluster import MiniBatchKMeans
import time

# Generate a large dataset
np.random.seed(42)
X_large = np.random.randn(100000, 2)

# Compare standard K-means with Mini-Batch K-means
start_time = time.time()
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_large)
kmeans_time = time.time() - start_time

start_time = time.time()
mbkmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=1000)
mbkmeans.fit(X_large)
mbkmeans_time = time.time() - start_time

print(f"K-means time: {kmeans_time:.2f} seconds")
print(f"Mini-Batch K-means time: {mbkmeans_time:.2f} seconds")
```

Slide 13: Integrating K-means with Other Techniques

K-means can be combined with other machine learning techniques for more advanced analysis. For example, using K-means as a preprocessing step for classification tasks or combining it with dimensionality reduction techniques for better visualization.

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_pca)

# Plot the results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.title('K-means Clustering on PCA-reduced Iris Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
```

Slide 14: Additional Resources

For those interested in delving deeper into K-means clustering and its applications, the following resources provide valuable insights and advanced techniques:

1. "A Comprehensive Survey of Clustering Algorithms" by Xu and Tian (2015) ArXiv URL: [https://arxiv.org/abs/1506.04337](https://arxiv.org/abs/1506.04337) This survey paper offers an extensive overview of various clustering algorithms, including K-means and its variants. It provides a comprehensive understanding of the field and discusses the strengths and weaknesses of different clustering approaches.
2. "k-means++: The Advantages of Careful Seeding" by Arthur and Vassilvitskii (2007) ArXiv URL: [https://arxiv.org/abs/0701164](https://arxiv.org/abs/0701164) This seminal paper introduces the k-means++ algorithm, which improves upon the standard K-means by proposing a smarter initialization technique. It offers both theoretical guarantees and practical improvements in clustering quality.
3. "Scalable K-Means++" by Bahmani et al. (2012) ArXiv URL: [https://arxiv.org/abs/1203.6402](https://arxiv.org/abs/1203.6402) This paper presents a scalable version of k-means++ that is suitable for large-scale distributed clustering tasks. It's particularly useful for those working with big data or distributed computing environments.

These resources provide a solid foundation for understanding K-means clustering in greater depth and exploring its advanced applications and variations.

