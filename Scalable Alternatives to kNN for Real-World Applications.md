## Scalable Alternatives to kNN for Real-World Applications
Slide 1: The Scalability Challenge of kNN

kNN (k-Nearest Neighbors) is a simple and intuitive algorithm, but it faces significant scalability issues when dealing with large datasets. As the dataset grows, the time complexity of kNN increases linearly, making it impractical for real-world applications that require near real-time responses. This limitation has led to the development of more efficient alternatives, such as Approximate Nearest Neighbor (ANN) search algorithms.

```python
import numpy as np
import time

def knn_search(data, query, k):
    distances = np.linalg.norm(data - query, axis=1)
    return np.argsort(distances)[:k]

# Generate random dataset
n_samples = 1000000
n_features = 100
data = np.random.rand(n_samples, n_features)
query = np.random.rand(n_features)

# Measure search time
start_time = time.time()
neighbors = knn_search(data, query, k=5)
end_time = time.time()

print(f"Time taken for {n_samples} samples: {end_time - start_time:.4f} seconds")
```

Slide 2: The Need for Approximate Nearest Neighbor (ANN)

Approximate Nearest Neighbor (ANN) algorithms have gained popularity due to their ability to provide faster search times while maintaining acceptable accuracy. These algorithms sacrifice some precision in exchange for significantly improved performance, making them suitable for large-scale applications where exact results are not critical.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Generate random dataset
n_samples = 1000000
n_features = 100
data = np.random.rand(n_samples, n_features)
query = np.random.rand(n_features)

# Exact kNN
exact_nn = NearestNeighbors(n_neighbors=5, algorithm='brute')
exact_nn.fit(data)

# Approximate kNN (using ball tree)
approx_nn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
approx_nn.fit(data)

# Compare search times
exact_time = time.time()
exact_neighbors = exact_nn.kneighbors([query], return_distance=False)
exact_time = time.time() - exact_time

approx_time = time.time()
approx_neighbors = approx_nn.kneighbors([query], return_distance=False)
approx_time = time.time() - approx_time

print(f"Exact kNN time: {exact_time:.4f} seconds")
print(f"Approximate kNN time: {approx_time:.4f} seconds")
print(f"Speedup: {exact_time / approx_time:.2f}x")
```

Slide 3: Inverted File Index: A Simple ANN Technique

The Inverted File Index is a straightforward ANN technique that can significantly improve search efficiency. This method involves partitioning the data using clustering algorithms and maintaining a mapping between centroids and data points. By reducing the search space, we can achieve faster query times compared to exhaustive search methods.

```python
import numpy as np
from sklearn.cluster import KMeans

class InvertedFileIndex:
    def __init__(self, data, n_clusters):
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.labels = self.kmeans.fit_predict(data)
        self.inverted_index = {i: np.where(self.labels == i)[0] for i in range(n_clusters)}
        self.data = data

    def search(self, query, k):
        centroid = self.kmeans.predict([query])[0]
        candidates = self.data[self.inverted_index[centroid]]
        distances = np.linalg.norm(candidates - query, axis=1)
        return self.inverted_index[centroid][np.argsort(distances)[:k]]

# Usage example
n_samples, n_features, n_clusters = 100000, 50, 100
data = np.random.rand(n_samples, n_features)
query = np.random.rand(n_features)

ifi = InvertedFileIndex(data, n_clusters)
result = ifi.search(query, k=5)
print(f"Nearest neighbors indices: {result}")
```

Slide 4: Step 1: Data Partitioning

The first step in the Inverted File Index method is to partition the data using a clustering algorithm such as KMeans. This process divides the dataset into 'k' clusters, each represented by a centroid. By grouping similar data points together, we create a more efficient search structure.

```python
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate 2D data for visualization
n_samples = 1000
data = np.random.rand(n_samples, 2)

# Perform KMeans clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters)
labels = kmeans.fit_predict(data)

# Visualize the clusters
plt.figure(figsize=(10, 8))
for i in range(n_clusters):
    cluster_points = data[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='X', label='Centroids')
plt.title('KMeans Clustering for Inverted File Index')
plt.legend()
plt.show()
```

Slide 5: Step 2: Building the Inverted Index

After partitioning the data, we create an inverted index that maps each centroid to its corresponding data points. This index allows us to quickly access the relevant subset of data during the search process, significantly reducing the number of comparisons needed.

```python
import numpy as np
from collections import defaultdict

def build_inverted_index(labels, n_clusters):
    inverted_index = defaultdict(list)
    for i, label in enumerate(labels):
        inverted_index[label].append(i)
    return dict(inverted_index)

# Example usage
n_samples, n_clusters = 1000, 5
labels = np.random.randint(0, n_clusters, n_samples)
inverted_index = build_inverted_index(labels, n_clusters)

print("Inverted Index Structure:")
for cluster, points in inverted_index.items():
    print(f"Cluster {cluster}: {len(points)} points")
    if len(points) > 5:
        print(f"  First 5 points: {points[:5]}")
    else:
        print(f"  Points: {points}")
```

Slide 6: Step 3: Finding the Closest Centroid

During the search process, the first step is to find the closest centroid to the query point. This operation is typically much faster than searching through the entire dataset, as the number of centroids is significantly smaller than the total number of data points.

```python
import numpy as np
from sklearn.cluster import KMeans

def find_closest_centroid(query, kmeans):
    distances = np.linalg.norm(kmeans.cluster_centers_ - query, axis=1)
    return np.argmin(distances)

# Example usage
n_samples, n_features, n_clusters = 10000, 50, 100
data = np.random.rand(n_samples, n_features)
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(data)

query = np.random.rand(n_features)
closest_centroid = find_closest_centroid(query, kmeans)

print(f"Query point: {query[:5]}...")
print(f"Closest centroid index: {closest_centroid}")
print(f"Closest centroid: {kmeans.cluster_centers_[closest_centroid][:5]}...")
```

Slide 7: Step 4: Searching Within the Partition

Once we've identified the closest centroid, we narrow our search to only the data points within that centroid's partition. This approach significantly reduces the search space, resulting in faster query times compared to exhaustive search methods.

```python
import numpy as np
from sklearn.cluster import KMeans

def search_partition(query, data, inverted_index, closest_centroid, k):
    partition = data[inverted_index[closest_centroid]]
    distances = np.linalg.norm(partition - query, axis=1)
    nearest_indices = np.argsort(distances)[:k]
    return [inverted_index[closest_centroid][i] for i in nearest_indices]

# Example usage
n_samples, n_features, n_clusters, k = 10000, 50, 100, 5
data = np.random.rand(n_samples, n_features)
kmeans = KMeans(n_clusters=n_clusters)
labels = kmeans.fit_predict(data)

inverted_index = {i: np.where(labels == i)[0] for i in range(n_clusters)}
query = np.random.rand(n_features)
closest_centroid = np.argmin(np.linalg.norm(kmeans.cluster_centers_ - query, axis=1))

nearest_neighbors = search_partition(query, data, inverted_index, closest_centroid, k)
print(f"Indices of {k} nearest neighbors: {nearest_neighbors}")
```

Slide 8: Performance Comparison: kNN vs. Inverted File Index

To demonstrate the efficiency gains of the Inverted File Index method, let's compare its performance against traditional kNN on a large dataset. We'll measure the search time for both methods and observe the speedup achieved by the approximate approach.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

def inverted_file_index_search(data, query, k, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters).fit(data)
    labels = kmeans.labels_
    inverted_index = {i: np.where(labels == i)[0] for i in range(n_clusters)}
    
    closest_centroid = np.argmin(np.linalg.norm(kmeans.cluster_centers_ - query, axis=1))
    partition = data[inverted_index[closest_centroid]]
    distances = np.linalg.norm(partition - query, axis=1)
    nearest_indices = np.argsort(distances)[:k]
    return [inverted_index[closest_centroid][i] for i in nearest_indices]

# Generate dataset
n_samples, n_features, n_clusters, k = 1000000, 50, 100, 5
data = np.random.rand(n_samples, n_features)
query = np.random.rand(n_features)

# Traditional kNN
start_time = time.time()
nn = NearestNeighbors(n_neighbors=k, algorithm='brute')
nn.fit(data)
knn_indices = nn.kneighbors([query], return_distance=False)[0]
knn_time = time.time() - start_time

# Inverted File Index
start_time = time.time()
ifi_indices = inverted_file_index_search(data, query, k, n_clusters)
ifi_time = time.time() - start_time

print(f"kNN search time: {knn_time:.4f} seconds")
print(f"Inverted File Index search time: {ifi_time:.4f} seconds")
print(f"Speedup: {knn_time / ifi_time:.2f}x")
```

Slide 9: Tuning the Number of Partitions

The choice of 'k' (number of partitions) in the Inverted File Index method is crucial for balancing search speed and accuracy. Too few partitions may result in insufficient speedup, while too many can lead to increased centroid search time. Let's explore how different 'k' values affect performance.

```python
import numpy as np
from sklearn.cluster import KMeans
import time

def measure_search_time(data, query, k, n_clusters):
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters).fit(data)
    labels = kmeans.labels_
    inverted_index = {i: np.where(labels == i)[0] for i in range(n_clusters)}
    
    closest_centroid = np.argmin(np.linalg.norm(kmeans.cluster_centers_ - query, axis=1))
    partition = data[inverted_index[closest_centroid]]
    distances = np.linalg.norm(partition - query, axis=1)
    nearest_indices = np.argsort(distances)[:k]
    end_time = time.time()
    return end_time - start_time

# Generate dataset
n_samples, n_features, k = 100000, 50, 5
data = np.random.rand(n_samples, n_features)
query = np.random.rand(n_features)

# Test different numbers of clusters
cluster_sizes = [10, 50, 100, 500, 1000]
search_times = []

for n_clusters in cluster_sizes:
    search_time = measure_search_time(data, query, k, n_clusters)
    search_times.append(search_time)
    print(f"Number of clusters: {n_clusters}, Search time: {search_time:.4f} seconds")

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(cluster_sizes, search_times, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Search Time (seconds)')
plt.title('Search Time vs. Number of Clusters')
plt.xscale('log')
plt.grid(True)
plt.show()
```

Slide 10: Real-Life Example: Image Similarity Search

Image similarity search is a common application of Approximate Nearest Neighbor algorithms. These systems quickly find visually similar images in large databases, useful for content-based image retrieval and recommendation systems.

```python
import numpy as np
from sklearn.cluster import KMeans

# Simulated image features database
n_images = 10000
feature_dim = 2048  # e.g., features from a pre-trained CNN
image_features = np.random.rand(n_images, feature_dim)

# Build Inverted File Index
n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters)
labels = kmeans.fit_predict(image_features)
inverted_index = {i: np.where(labels == i)[0] for i in range(n_clusters)}

# Simulate a query image
query_image = np.random.rand(feature_dim)

# Find similar images
closest_centroid = kmeans.predict([query_image])[0]
candidates = image_features[inverted_index[closest_centroid]]
distances = np.linalg.norm(candidates - query_image, axis=1)
similar_indices = inverted_index[closest_centroid][np.argsort(distances)[:5]]

print("Indices of similar images:", similar_indices)
```

Slide 11: Real-Life Example: Recommendation Systems

Recommendation systems often use Approximate Nearest Neighbor techniques to suggest items to users based on similarity. This approach is particularly useful for large-scale systems with millions of items and users.

```python
import numpy as np
from sklearn.cluster import KMeans

# Simulated user-item interaction matrix
n_users = 10000
n_items = 50000
user_item_matrix = np.random.rand(n_users, n_items)

# Build Inverted File Index for items
n_clusters = 500
kmeans = KMeans(n_clusters=n_clusters)
item_labels = kmeans.fit_predict(user_item_matrix.T)  # Transpose for item features
item_index = {i: np.where(item_labels == i)[0] for i in range(n_clusters)}

# Function to get recommendations for a user
def get_recommendations(user_id, top_k=10):
    user_profile = user_item_matrix[user_id]
    closest_centroid = kmeans.predict([user_profile])[0]
    candidate_items = item_index[closest_centroid]
    
    # Calculate similarity with candidate items
    similarities = np.dot(user_profile, user_item_matrix[:, candidate_items].T)
    recommended_items = candidate_items[np.argsort(similarities)[-top_k:][::-1]]
    
    return recommended_items

# Example usage
user_id = 42
recommendations = get_recommendations(user_id)
print(f"Top recommendations for user {user_id}:", recommendations)
```

Slide 12: Limitations and Considerations

While Approximate Nearest Neighbor methods like Inverted File Index offer significant performance improvements, they come with trade-offs:

1. Accuracy: ANN methods may miss some true nearest neighbors, potentially affecting the quality of results.
2. Parameter Tuning: The number of clusters (k) needs careful tuning to balance speed and accuracy.
3. Data Distribution: Performance can vary depending on the distribution of data points.

Slide 13: Limitations and Considerations

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

def compare_accuracy(data, query, k, n_clusters):
    # Exact kNN
    exact_nn = NearestNeighbors(n_neighbors=k, algorithm='brute')
    exact_nn.fit(data)
    exact_neighbors = exact_nn.kneighbors([query], return_distance=False)[0]
    
    # Approximate NN (Inverted File Index)
    kmeans = KMeans(n_clusters=n_clusters).fit(data)
    labels = kmeans.labels_
    inverted_index = {i: np.where(labels == i)[0] for i in range(n_clusters)}
    closest_centroid = kmeans.predict([query])[0]
    candidates = data[inverted_index[closest_centroid]]
    distances = np.linalg.norm(candidates - query, axis=1)
    approx_neighbors = inverted_index[closest_centroid][np.argsort(distances)[:k]]
    
    # Calculate accuracy
    accuracy = len(set(exact_neighbors) & set(approx_neighbors)) / k
    return accuracy

# Example usage
n_samples, n_features, k, n_clusters = 10000, 50, 10, 100
data = np.random.rand(n_samples, n_features)
query = np.random.rand(n_features)

accuracy = compare_accuracy(data, query, k, n_clusters)
print(f"Accuracy of Approximate NN: {accuracy:.2f}")
```

Slide 14: Future Directions and Advanced Techniques

The field of Approximate Nearest Neighbor search is continuously evolving. Some advanced techniques and future directions include:

1. Graph-based methods (e.g., HNSW - Hierarchical Navigable Small World)
2. Locality-Sensitive Hashing (LSH) for high-dimensional data
3. Hybrid approaches combining multiple indexing strategies
4. Machine learning-based indexing structures

Slide 15: Future Directions and Advanced Techniques

These advanced methods aim to further improve search speed and accuracy, especially for high-dimensional and large-scale datasets.

```python
# Pseudocode for HNSW (Hierarchical Navigable Small World)

class HNSW:
    def __init__(self, max_elements, M, ef_construction):
        self.max_elements = max_elements
        self.M = M  # Max number of connections per element
        self.ef_construction = ef_construction  # Size of dynamic candidate list
        self.layers = []  # Multi-layer graph structure

    def add_element(self, element):
        # Select insertion point and layer
        entry_point = self.select_entry_point()
        max_layer = self.select_max_layer()

        # Insert element in each layer
        for layer in range(max_layer, -1, -1):
            neighbors = self.search_layer(element, entry_point, self.ef_construction, layer)
            self.connect_neighbors(element, neighbors, layer)
            entry_point = element

    def search(self, query, k):
        # Start from the top layer
        current_layer = len(self.layers) - 1
        entry_point = self.layers[current_layer][0]

        # Traverse down the layers
        while current_layer >= 0:
            entry_point = self.search_layer(query, entry_point, self.ef_construction, current_layer)
            current_layer -= 1

        # Final search in the bottom layer
        return self.search_layer(query, entry_point, k, 0)

    # Other helper methods: search_layer, connect_neighbors, etc.
```

Slide 16: Additional Resources

For those interested in diving deeper into Approximate Nearest Neighbor algorithms and their applications, here are some valuable resources:

1. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" by Yu. A. Malkov and D. A. Yashunin (2018) ArXiv: [https://arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320)
2. "Billion-scale similarity search with GPUs" by Johnson et al. (2017) ArXiv: [https://arxiv.org/abs/1702.08734](https://arxiv.org/abs/1702.08734)
3. "A Survey of Product Quantization" by Jegou et al. (2020) ArXiv: [https://arxiv.org/abs/1910.11797](https://arxiv.org/abs/1910.11797)

These papers provide in-depth discussions on state-of-the-art ANN techniques and their practical implementations in large-scale systems.

