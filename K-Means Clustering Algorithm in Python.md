## K-Means Clustering Algorithm in Python
Slide 1: Introduction to Clustering Algorithms

Clustering is an unsupervised machine learning technique used to group similar data points together. While there are various clustering algorithms, this presentation will focus on the K-means algorithm, which is one of the most commonly used clustering methods due to its simplicity and effectiveness.

Slide 2: K-means Algorithm Overview

K-means is an iterative algorithm that partitions a dataset into K distinct, non-overlapping clusters. It works by assigning data points to the nearest cluster centroid and then updating the centroids based on the mean of the assigned points. This process continues until convergence or a maximum number of iterations is reached.

Slide 3: Source Code for K-means Algorithm Overview

```python
import random

def kmeans(data, k, max_iterations=100):
    # Initialize centroids randomly
    centroids = random.sample(data, k)
    
    for _ in range(max_iterations):
        # Assign points to nearest centroid
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid = distances.index(min(distances))
            clusters[closest_centroid].append(point)
        
        # Update centroids
        new_centroids = [calculate_centroid(cluster) for cluster in clusters]
        
        # Check for convergence
        if new_centroids == centroids:
            break
        
        centroids = new_centroids
    
    return clusters, centroids

def euclidean_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

def calculate_centroid(cluster):
    return tuple(sum(coord) / len(cluster) for coord in zip(*cluster))
```

Slide 4: Results for K-means Algorithm Overview

```python
# Example usage
data = [(1, 2), (2, 1), (4, 3), (5, 4)]
k = 2
clusters, centroids = kmeans(data, k)

print("Clusters:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {cluster}")

print("\nCentroids:")
for i, centroid in enumerate(centroids):
    print(f"Centroid {i + 1}: {centroid}")

# Output:
# Clusters:
# Cluster 1: [(1, 2), (2, 1)]
# Cluster 2: [(4, 3), (5, 4)]
#
# Centroids:
# Centroid 1: (1.5, 1.5)
# Centroid 2: (4.5, 3.5)
```

Slide 5: Initialization Methods

The initial placement of centroids can significantly impact the final clustering results. Common initialization methods include random initialization, the K-means++ algorithm, and the Forgy method. K-means++ aims to choose initial centroids that are well-spread across the dataset, potentially leading to better convergence.

Slide 6: Source Code for Initialization Methods

```python
import random

def kmeans_plus_plus(data, k):
    centroids = [random.choice(data)]
    
    for _ in range(1, k):
        distances = [min(euclidean_distance(point, centroid) ** 2 
                         for centroid in centroids) 
                     for point in data]
        total_distance = sum(distances)
        probabilities = [d / total_distance for d in distances]
        
        cumulative_prob = 0
        r = random.random()
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if cumulative_prob > r:
                centroids.append(data[i])
                break
    
    return centroids

# Use kmeans_plus_plus(data, k) instead of random.sample(data, k) in the kmeans function
```

Slide 7: Elbow Method for Choosing K

The elbow method is a heuristic used to determine the optimal number of clusters (K) for K-means. It involves running K-means with different K values and plotting the sum of squared distances between data points and their assigned cluster centroids. The "elbow" in the resulting curve suggests an appropriate K value.

Slide 8: Source Code for Elbow Method

```python
import matplotlib.pyplot as plt

def elbow_method(data, max_k):
    inertias = []
    
    for k in range(1, max_k + 1):
        clusters, centroids = kmeans(data, k)
        inertia = sum(min(euclidean_distance(point, centroid) ** 2 
                          for centroid in centroids) 
                      for point in data)
        inertias.append(inertia)
    
    plt.plot(range(1, max_k + 1), inertias, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.show()

# Example usage
data = [(1, 2), (2, 1), (4, 3), (5, 4), (1, 1), (2, 2), (4, 4), (5, 5)]
elbow_method(data, 10)
```

Slide 9: Silhouette Analysis

Silhouette analysis is another method to evaluate the quality of clustering results. It measures how similar an object is to its own cluster compared to other clusters. The silhouette score ranges from -1 to 1, where a high value indicates that the object is well-matched to its own cluster and poorly-matched to neighboring clusters.

Slide 10: Source Code for Silhouette Analysis

```python
def silhouette_score(data, clusters):
    silhouette_values = []
    
    for i, point in enumerate(data):
        a = intra_cluster_distance(point, clusters[i])
        b = min(inter_cluster_distance(point, cluster) 
                for j, cluster in enumerate(clusters) if j != i)
        
        silhouette = (b - a) / max(a, b)
        silhouette_values.append(silhouette)
    
    return sum(silhouette_values) / len(silhouette_values)

def intra_cluster_distance(point, cluster):
    return sum(euclidean_distance(point, other) for other in cluster) / len(cluster)

def inter_cluster_distance(point, cluster):
    return sum(euclidean_distance(point, other) for other in cluster) / len(cluster)

# Example usage
clusters, _ = kmeans(data, 2)
score = silhouette_score(data, clusters)
print(f"Silhouette Score: {score}")
```

Slide 11: Real-Life Example: Image Compression

K-means clustering can be used for image compression by reducing the number of colors in an image. Each pixel is treated as a data point in the RGB color space, and K-means is applied to find K representative colors. The original image is then recreated using only these K colors.

Slide 12: Source Code for Image Compression

```python
from PIL import Image
import numpy as np

def compress_image(image_path, k):
    # Open image and convert to numpy array
    img = Image.open(image_path)
    pixels = np.array(img.getdata()).astype(float)
    
    # Apply K-means clustering
    clusters, centroids = kmeans(pixels.tolist(), k)
    
    # Replace each pixel with its cluster centroid
    compressed_pixels = np.array([centroids[pixels.tolist().index(pixel)] for pixel in pixels.tolist()])
    
    # Reshape and save the compressed image
    compressed_img = Image.fromarray(compressed_pixels.astype(np.uint8).reshape(img.size[1], img.size[0], 3))
    compressed_img.save(f"compressed_{k}_colors.png")

# Example usage
compress_image("original_image.png", 16)
```

Slide 13: Real-Life Example: Customer Segmentation

K-means clustering is widely used in marketing for customer segmentation. By grouping customers based on features such as purchasing behavior, demographics, and engagement metrics, businesses can tailor their marketing strategies to specific customer segments.

Slide 14: Source Code for Customer Segmentation

```python
import random

def generate_customer_data(n_customers):
    return [(random.randint(18, 80),  # Age
             random.randint(0, 100000),  # Annual Income
             random.randint(0, 100))  # Loyalty Score
            for _ in range(n_customers)]

def segment_customers(data, k):
    clusters, centroids = kmeans(data, k)
    
    for i, cluster in enumerate(clusters):
        print(f"Segment {i + 1}:")
        print(f"  Average Age: {sum(c[0] for c in cluster) / len(cluster):.2f}")
        print(f"  Average Income: ${sum(c[1] for c in cluster) / len(cluster):.2f}")
        print(f"  Average Loyalty Score: {sum(c[2] for c in cluster) / len(cluster):.2f}")
        print()

# Example usage
customer_data = generate_customer_data(1000)
segment_customers(customer_data, 3)
```

Slide 15: Additional Resources

For more in-depth information on clustering algorithms and their applications, consider exploring the following resources:

1.  "A Survey of Clustering Data Mining Techniques" by Pavel Berkhin (ArXiv:cs/0604008) URL: [https://arxiv.org/abs/cs/0604008](https://arxiv.org/abs/cs/0604008)
2.  "Clustering by fast search and find of density peaks" by Rodriguez and Laio (ArXiv:1608.03402) URL: [https://arxiv.org/abs/1608.03402](https://arxiv.org/abs/1608.03402)
3.  "A Tutorial on Spectral Clustering" by Ulrike von Luxburg (ArXiv:0711.0189) URL: [https://arxiv.org/abs/0711.0189](https://arxiv.org/abs/0711.0189)

These papers provide comprehensive overviews of various clustering techniques, including K-means and other advanced methods.

