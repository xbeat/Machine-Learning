## Mastering K-Means Clustering with Python
Slide 1: Introduction to K-Means Clustering

K-Means is an unsupervised machine learning algorithm used for clustering data points into groups based on similarity. It's widely used in data analysis, pattern recognition, and image segmentation. This algorithm aims to partition n observations into k clusters, where each observation belongs to the cluster with the nearest mean.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 2)

# Create K-Means instance
kmeans = KMeans(n_clusters=3)

# Fit the model
kmeans.fit(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title('K-Means Clustering Example')
plt.show()
```

Slide 2: The K-Means Algorithm

The K-Means algorithm works by iteratively assigning data points to the nearest cluster center and then updating the cluster centers based on the mean of the assigned points. This process continues until convergence or a maximum number of iterations is reached.

```python
def kmeans(X, k, max_iters=100):
    # Randomly initialize cluster centers
    centers = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to nearest center
        distances = np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centers
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence
        if np.all(centers == new_centers):
            break
        
        centers = new_centers
    
    return labels, centers

# Example usage
X = np.random.rand(100, 2)
labels, centers = kmeans(X, k=3)
```

Slide 3: Choosing the Right Number of Clusters

Determining the optimal number of clusters is crucial for effective K-Means clustering. The elbow method is a popular technique for this purpose. It involves running K-Means with different values of k and plotting the sum of squared distances against k.

```python
from sklearn.cluster import KMeans

def elbow_method(X, max_k):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow Method for Optimal k')
    plt.show()

# Example usage
X = np.random.rand(200, 2)
elbow_method(X, max_k=10)
```

Slide 4: Preprocessing Data for K-Means

Before applying K-Means, it's important to preprocess the data to ensure optimal clustering results. This often includes scaling features to a similar range and handling missing values.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(X):
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled

# Example usage
X = np.random.rand(100, 3)
X[np.random.choice(X.shape[0], 10), :] = np.nan  # Add some missing values
X_preprocessed = preprocess_data(X)
print("Original data shape:", X.shape)
print("Preprocessed data shape:", X_preprocessed.shape)
```

Slide 5: Visualizing K-Means Results

Visualization is key to understanding the results of K-Means clustering. For 2D data, we can easily plot the clusters. For higher-dimensional data, we can use dimensionality reduction techniques like PCA before visualization.

```python
from sklearn.decomposition import PCA

def visualize_kmeans(X, labels, centers):
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        centers_2d = pca.transform(centers)
    else:
        X_2d = X
        centers_2d = centers
    
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis')
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], marker='x', s=200, linewidths=3, color='r')
    plt.title('K-Means Clustering Results')
    plt.show()

# Example usage
X = np.random.rand(200, 4)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
visualize_kmeans(X, labels, kmeans.cluster_centers_)
```

Slide 6: Handling Non-Globular Clusters

K-Means assumes that clusters are globular and equally sized, which isn't always the case in real-world data. When dealing with non-globular clusters, consider using alternative clustering algorithms like DBSCAN or Gaussian Mixture Models.

```python
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

# Generate non-globular data
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
ax1.set_title('K-Means Clustering')
ax2.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
ax2.set_title('DBSCAN Clustering')
plt.show()
```

Slide 7: Evaluating Clustering Performance

Evaluating the performance of K-Means can be challenging since it's an unsupervised learning method. However, we can use metrics like silhouette score or calinski-harabasz index to assess cluster quality.

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def evaluate_clustering(X, labels):
    silhouette = silhouette_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")

# Example usage
X = np.random.rand(200, 2)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
evaluate_clustering(X, labels)
```

Slide 8: K-Means++: Improving Initialization

K-Means++ is an improvement over the standard K-Means algorithm that provides a smarter initialization of cluster centers. This often leads to better and more consistent results.

```python
from sklearn.cluster import KMeans

def compare_kmeans_init(X, k):
    kmeans_random = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
    kmeans_plus = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    
    labels_random = kmeans_random.fit_predict(X)
    labels_plus = kmeans_plus.fit_predict(X)
    
    print("K-Means with random initialization:")
    evaluate_clustering(X, labels_random)
    print("\nK-Means++ initialization:")
    evaluate_clustering(X, labels_plus)

# Example usage
X = np.random.rand(500, 2)
compare_kmeans_init(X, k=5)
```

Slide 9: Handling Categorical Data

K-Means typically works with numerical data, but we often encounter categorical variables in real-world datasets. One approach is to use one-hot encoding to convert categorical data into a numerical format suitable for K-Means.

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocess_mixed_data(X):
    # Assume X is a pandas DataFrame with mixed data types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    # One-hot encode categorical features
    onehot = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_features = onehot.fit_transform(X[categorical_features])
    encoded_feature_names = onehot.get_feature_names(categorical_features)
    
    # Combine numeric and encoded categorical features
    X_numeric = X[numeric_features].values
    X_preprocessed = np.hstack((X_numeric, encoded_features))
    
    return X_preprocessed

# Example usage
data = {
    'age': [25, 30, 35, 40],
    'income': [50000, 60000, 75000, 90000],
    'education': ['High School', 'Bachelor', 'Master', 'PhD']
}
df = pd.DataFrame(data)
X_preprocessed = preprocess_mixed_data(df)
print("Preprocessed data shape:", X_preprocessed.shape)
```

Slide 10: Real-Life Example: Customer Segmentation

K-Means clustering can be used for customer segmentation in marketing. This example demonstrates how to cluster customers based on their purchasing behavior and demographics.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Sample customer data (replace with your own dataset)
data = {
    'customer_id': range(1, 101),
    'age': np.random.randint(18, 70, 100),
    'annual_income': np.random.randint(20000, 100000, 100),
    'spending_score': np.random.randint(1, 100, 100)
}
df = pd.DataFrame(data)

# Preprocess the data
X = df[['age', 'annual_income', 'spending_score']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the results
plt.scatter(df['annual_income'], df['spending_score'], c=df['cluster'], cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segments')
plt.show()

print(df.groupby('cluster').mean())
```

Slide 11: Real-Life Example: Image Compression

K-Means can be used for image compression by reducing the number of colors in an image. This example demonstrates how to compress an image using K-Means clustering.

```python
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np

def compress_image(image_path, k):
    # Load the image
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # Reshape the image to a 2D array of pixels
    pixels = image_array.reshape(-1, 3)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    
    # Replace each pixel with its nearest cluster center
    compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
    
    # Reshape back to the original image shape
    compressed_image = compressed_pixels.reshape(image_array.shape).astype(np.uint8)
    
    return Image.fromarray(compressed_image)

# Example usage (you'll need to provide your own image file)
original_image = Image.open('your_image.jpg')
compressed_image = compress_image('your_image.jpg', k=16)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(original_image)
ax1.set_title('Original Image')
ax2.imshow(compressed_image)
ax2.set_title('Compressed Image (16 colors)')
plt.show()
```

Slide 12: Challenges and Limitations of K-Means

While K-Means is a powerful and widely used clustering algorithm, it has some limitations. These include sensitivity to initial centroids, the assumption of spherical clusters, and difficulty with high-dimensional data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate non-spherical data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=[1.0, 2.5, 0.5, 3.0], random_state=42)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title('K-Means with Non-Spherical Clusters')
plt.show()

# Calculate the percentage of points in each cluster
unique, counts = np.unique(labels, return_counts=True)
percentages = counts / len(labels) * 100
for cluster, percentage in zip(unique, percentages):
    print(f"Cluster {cluster}: {percentage:.2f}%")
```

Slide 13: Advanced K-Means Techniques

Several advanced techniques can improve K-Means performance and address some of its limitations. These include Mini-Batch K-Means for large datasets, Gaussian Mixture Models for probabilistic clustering, and ensemble methods.

```python
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
import time

def compare_clustering_methods(X, k):
    methods = {
        'K-Means': KMeans(n_clusters=k, random_state=42),
        'Mini-Batch K-Means': MiniBatchKMeans(n_clusters=k, random_state=42),
        'Gaussian Mixture Model': GaussianMixture(n_components=k, random_state=42)
    }
    
    results = {}
    for name, method in methods.items():
        start_time = time.time()
        labels = method.fit_predict(X)
        execution_time = time.time() - start_time
        results[name] = {'labels': labels, 'time': execution_time}
    
    return results

# Example usage
X = np.random.rand(10000, 2)
results = compare_clustering_methods(X, k=5)

for name, result in results.items():
    print(f"{name}: {result['time']:.4f} seconds")
```

Slide 14: Implementing K-Means from Scratch

Understanding the inner workings of K-Means is crucial. Here's a basic implementation of the K-Means algorithm from scratch using NumPy.

```python
import numpy as np

def kmeans_scratch(X, k, max_iters=100):
    n_samples, n_features = X.shape
    
    # Randomly initialize centroids
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iters):
        # Assign each point to the nearest centroid
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
X = np.random.rand(1000, 2)
labels, centroids = kmeans_scratch(X, k=3)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title('K-Means from Scratch')
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into K-Means clustering and its applications, here are some valuable resources:

1. ArXiv paper: "k-means++: The Advantages of Careful Seeding" by Arthur and Vassilvitskii (2007) ArXiv URL: [https://arxiv.org/abs/0604034](https://arxiv.org/abs/0604034)
2. ArXiv paper: "Scalable K-Means++" by Bahmani et al. (2012) ArXiv URL: [https://arxiv.org/abs/1203.6402](https://arxiv.org/abs/1203.6402)
3. ArXiv paper: "Mini-Batch K-Means Clustering on GPU" by Gao et al. (2018) ArXiv URL: [https://arxiv.org/abs/1812.03635](https://arxiv.org/abs/1812.03635)

These papers provide in-depth analysis and improvements to the K-Means algorithm, offering valuable insights for both theoretical understanding and practical implementation.

