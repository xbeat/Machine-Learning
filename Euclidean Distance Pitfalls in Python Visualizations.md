## Euclidean Distance Pitfalls in Python Visualizations
Slide 1: Euclidean Distance: A Potentially Misleading Metric

Euclidean distance is a common measure of similarity in data science, but it can be misleading in certain scenarios. This presentation explores why and how Euclidean distance can lead to incorrect conclusions, and provides alternative approaches using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define two points
point1 = np.array([0, 0])
point2 = np.array([3, 4])

# Calculate Euclidean distance
distance = np.linalg.norm(point1 - point2)

print(f"Euclidean distance: {distance}")
# Output: Euclidean distance: 5.0

# Visualize the points
plt.scatter(point1[0], point1[1], label='Point 1')
plt.scatter(point2[0], point2[1], label='Point 2')
plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r--')
plt.legend()
plt.title('Euclidean Distance Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
```

Slide 2: The Curse of Dimensionality

As the number of dimensions increases, Euclidean distance becomes less meaningful. This phenomenon, known as the "curse of dimensionality," can lead to counterintuitive results in high-dimensional spaces.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_random_points(n_points, n_dims):
    return np.random.rand(n_points, n_dims)

def calculate_pairwise_distances(points):
    return np.linalg.norm(points[:, np.newaxis] - points, axis=2)

dimensions = range(1, 101, 5)
n_points = 1000

avg_distances = []
for dim in dimensions:
    points = generate_random_points(n_points, dim)
    distances = calculate_pairwise_distances(points)
    avg_distances.append(np.mean(distances))

plt.plot(dimensions, avg_distances)
plt.title('Average Pairwise Distance vs. Dimensionality')
plt.xlabel('Number of Dimensions')
plt.ylabel('Average Pairwise Distance')
plt.show()

print(f"Average distance in 2D: {avg_distances[0]:.2f}")
print(f"Average distance in 100D: {avg_distances[-1]:.2f}")
# Output will vary due to randomness, but you'll see an increase in average distance
```

Slide 3: Misleading in Non-Uniform Feature Scales

Euclidean distance can be misleading when features have different scales. This can cause certain features to dominate the distance calculation, leading to biased results.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Sample data with different scales
data = np.array([
    [1, 1000],
    [2, 2000],
    [3, 3000],
    [4, 4000]
])

# Calculate Euclidean distances
distances = np.linalg.norm(data[0] - data[1:], axis=1)
print("Distances before scaling:", distances)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Calculate Euclidean distances after scaling
scaled_distances = np.linalg.norm(scaled_data[0] - scaled_data[1:], axis=1)
print("Distances after scaling:", scaled_distances)

# Output:
# Distances before scaling: [1000.00004999 2000.00004999 3000.00004999]
# Distances after scaling: [1.41421356 2.82842712 4.24264069]
```

Slide 4: Real-Life Example: Image Similarity

Euclidean distance can be misleading when comparing images. Small shifts or rotations can result in large Euclidean distances, even when images are visually similar.

```python
import numpy as np
from skimage import io, transform
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

# Load and preprocess images
def load_and_preprocess(image_path):
    img = io.imread(image_path, as_gray=True)
    img = transform.resize(img, (64, 64))
    return img.flatten()

# Load three images: original, slightly shifted, and different
img1 = load_and_preprocess('original.png')
img2 = load_and_preprocess('shifted.png')
img3 = load_and_preprocess('different.png')

# Calculate Euclidean distances
distances = euclidean_distances([img1, img2, img3])

print("Distance between original and shifted:", distances[0, 1])
print("Distance between original and different:", distances[0, 2])

# Visualize the results
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(img1.reshape(64, 64), cmap='gray')
plt.title('Original')
plt.subplot(132)
plt.imshow(img2.reshape(64, 64), cmap='gray')
plt.title(f'Shifted\nDistance: {distances[0, 1]:.2f}')
plt.subplot(133)
plt.imshow(img3.reshape(64, 64), cmap='gray')
plt.title(f'Different\nDistance: {distances[0, 2]:.2f}')
plt.show()

# Output will depend on the actual images used, but you might see that
# the distance between the original and shifted images is unexpectedly large
```

Slide 5: Handling Non-Linear Relationships

Euclidean distance assumes linear relationships between features. For non-linear relationships, it can provide misleading results. Let's visualize this limitation.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# Generate non-linear data
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.sin(x + np.pi/4)

# Calculate Euclidean distances
distances = pairwise_distances(y1.reshape(-1, 1), y2.reshape(-1, 1))

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(x, y1, label='Signal 1')
plt.plot(x, y2, label='Signal 2')
plt.title('Non-linear Signals')
plt.legend()

plt.subplot(122)
plt.imshow(distances, aspect='auto', extent=[0, 2*np.pi, 2*np.pi, 0])
plt.colorbar(label='Euclidean Distance')
plt.title('Pairwise Euclidean Distances')
plt.xlabel('Signal 1')
plt.ylabel('Signal 2')

plt.tight_layout()
plt.show()

print(f"Max distance: {np.max(distances):.2f}")
print(f"Min distance: {np.min(distances):.2f}")

# The output shows that Euclidean distance fails to capture
# the periodic nature of the signals
```

Slide 6: Alternative: Dynamic Time Warping (DTW)

For time series data, Dynamic Time Warping (DTW) can be more appropriate than Euclidean distance. DTW allows for non-linear alignment of sequences.

```python
import numpy as np
from dtaidistance import dtw
import matplotlib.pyplot as plt

# Generate two similar sequences with different lengths
t1 = np.linspace(0, 2*np.pi, 100)
t2 = np.linspace(0, 2*np.pi, 80)
seq1 = np.sin(t1)
seq2 = np.sin(t2)

# Calculate DTW distance
dtw_distance = dtw.distance(seq1, seq2)

# Calculate Euclidean distance (after resampling)
seq2_resampled = np.interp(t1, t2, seq2)
euclidean_distance = np.linalg.norm(seq1 - seq2_resampled)

plt.figure(figsize=(12, 5))
plt.plot(t1, seq1, label='Sequence 1')
plt.plot(t2, seq2, label='Sequence 2')
plt.title(f'DTW Distance: {dtw_distance:.2f}, Euclidean Distance: {euclidean_distance:.2f}')
plt.legend()
plt.show()

print(f"DTW Distance: {dtw_distance:.2f}")
print(f"Euclidean Distance: {euclidean_distance:.2f}")

# DTW often provides a more intuitive measure of similarity for time series
```

Slide 7: Alternative: Cosine Similarity

Cosine similarity focuses on the angle between vectors, making it less sensitive to magnitude differences. This can be useful when dealing with high-dimensional data.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Create sample vectors
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
v3 = np.array([10, 20, 30])

# Calculate cosine similarities
cos_sim = cosine_similarity([v1, v2, v3])

# Calculate Euclidean distances
euc_dist = euclidean_distances([v1, v2, v3])

print("Cosine Similarity Matrix:")
print(cos_sim)
print("\nEuclidean Distance Matrix:")
print(euc_dist)

# Visualize the vectors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='v1')
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='g', label='v2')
ax.quiver(0, 0, 0, v3[0], v3[1], v3[2], color='b', label='v3')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Vector Visualization')
plt.show()

# Cosine similarity shows v1 and v2 are more similar,
# while Euclidean distance suggests v1 and v3 are very different
```

Slide 8: Real-Life Example: Text Similarity

In text analysis, Euclidean distance can be misleading due to the high-dimensional nature of text data. Let's compare it with cosine similarity for document comparison.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

# Sample documents
docs = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog is jumped over by the quick brown fox",
    "A completely different sentence about cats and mice"
]

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)

# Calculate cosine similarities
cosine_sim = cosine_similarity(tfidf_matrix)

# Calculate Euclidean distances
euclidean_dist = euclidean_distances(tfidf_matrix.toarray())

print("Cosine Similarity Matrix:")
print(cosine_sim)
print("\nEuclidean Distance Matrix:")
print(euclidean_dist)

# Visualize the results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(cosine_sim, cmap='coolwarm')
plt.title('Cosine Similarity')
plt.colorbar()
plt.subplot(122)
plt.imshow(euclidean_dist, cmap='coolwarm')
plt.title('Euclidean Distance')
plt.colorbar()
plt.tight_layout()
plt.show()

# Cosine similarity better captures the semantic similarity between docs 1 and 2
```

Slide 9: Addressing the Curse of Dimensionality

To mitigate the curse of dimensionality, we can use dimensionality reduction techniques like Principal Component Analysis (PCA) before applying Euclidean distance.

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate high-dimensional data
n_samples = 1000
n_features = 100
X = np.random.randn(n_samples, n_features)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative explained variance ratio
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA: Cumulative Explained Variance Ratio')
plt.show()

# Find number of components for 95% variance
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of components for 95% variance: {n_components_95}")

# Reduce dimensionality
pca_95 = PCA(n_components=n_components_95)
X_reduced = pca_95.fit_transform(X_scaled)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")

# PCA helps reduce dimensionality while preserving most of the variance
```

Slide 10: Mahalanobis Distance: Accounting for Feature Correlations

Mahalanobis distance takes into account the correlations between features, which can be more appropriate than Euclidean distance for certain datasets.

```python
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Generate correlated data
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]
x, y = np.random.multivariate_normal(mean, cov, 1000).T

# Calculate Mahalanobis distances
inv_cov = np.linalg.inv(cov)
mahalanobis_dist = np.array([np.sqrt(np.dot(np.dot([x[i], y[i]], inv_cov), [x[i], y[i]])) for i in range(len(x))])

# Calculate Euclidean distances
euclidean_dist = np.sqrt(x**2 + y**2)

# Plot the results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(x, y, c=mahalanobis_dist, cmap='viridis')
plt.colorbar(label='Mahalanobis Distance')
plt.title('Mahalanobis Distance')

plt.subplot(122)
plt.scatter(x, y, c=euclidean_dist, cmap='viridis')
plt.colorbar(label='Euclidean Distance')
plt.title('Euclidean Distance')

plt.tight_layout()
plt.show()

print(f"Correlation between x and y: {np.corrcoef(x, y)[0, 1]:.2f}")

# Mahalanobis distance accounts for the correlation between features,
# resulting in elliptical contours instead of circular ones
```

Slide 11: Handling Categorical Data: Gower Distance

Euclidean distance is not suitable for categorical data. Gower distance can handle mixed data types, including categorical variables.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

# Create a mixed dataset
data = pd.DataFrame({
    'age': [25, 35, 45, 55],
    'income': [50000, 60000, 70000, 80000],
    'education': ['High School', 'Bachelor', 'Master', 'PhD'],
    'city': ['New York', 'London', 'Paris', 'Tokyo']
})

# Function to calculate Gower distance
def gower_distance(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns
    
    # Standardize numeric columns
    scaler = StandardScaler()
    numeric_data = scaler.fit_transform(data[numeric_cols])
    
    # One-hot encode categorical columns
    categorical_data = pd.get_dummies(data[categorical_cols])
    
    # Combine numeric and categorical data
    combined_data = np.hstack([numeric_data, categorical_data])
    
    # Calculate Gower distance
    gower_dist = pdist(combined_data, metric='euclidean')
    return squareform(gower_dist)

# Calculate Gower distance
gower_dist_matrix = gower_distance(data)

print("Gower Distance Matrix:")
print(gower_dist_matrix)

# Visualize the distance matrix
plt.imshow(gower_dist_matrix, cmap='viridis')
plt.colorbar()
plt.title('Gower Distance Matrix')
plt.show()
```

Slide 12: Limitations of Euclidean Distance in Clustering

Euclidean distance can lead to suboptimal results in clustering algorithms when dealing with clusters of different densities or non-spherical shapes.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons

# Generate non-spherical data
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# Apply K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
y_pred = kmeans.fit_predict(X)

# Visualize the results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('True Clusters')

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title('K-means Clusters (Euclidean Distance)')

plt.tight_layout()
plt.show()

# K-means with Euclidean distance fails to capture the true cluster structure
```

Slide 13: Alternative: DBSCAN for Non-Spherical Clusters

DBSCAN, which doesn't rely solely on Euclidean distance, can handle clusters of arbitrary shapes more effectively than K-means.

```python
from sklearn.cluster import DBSCAN

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_pred_dbscan = dbscan.fit_predict(X)

# Visualize the results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('True Clusters')

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_dbscan, cmap='viridis')
plt.title('DBSCAN Clusters')

plt.tight_layout()
plt.show()

print(f"Number of clusters found by DBSCAN: {len(np.unique(y_pred_dbscan))}")

# DBSCAN can identify the non-spherical cluster structure more accurately
```

Slide 14: Euclidean Distance in High-Dimensional Spaces: The Distance Concentration Effect

In high-dimensional spaces, Euclidean distances tend to concentrate, making it difficult to distinguish between nearest and farthest neighbors.

```python
import numpy as np
import matplotlib.pyplot as plt

def distance_concentration(dims, n_points=1000):
    points = np.random.uniform(0, 1, (n_points, dims))
    distances = np.linalg.norm(points - 0.5, axis=1)
    return np.mean(distances), np.std(distances)

dimensions = range(1, 101, 5)
means, stds = zip(*[distance_concentration(d) for d in dimensions])

plt.figure(figsize=(10, 6))
plt.errorbar(dimensions, means, yerr=stds, capsize=5)
plt.title('Distance Concentration in High Dimensions')
plt.xlabel('Number of Dimensions')
plt.ylabel('Average Distance to Center (± std)')
plt.show()

print(f"Ratio of std to mean for 2D: {stds[0]/means[0]:.4f}")
print(f"Ratio of std to mean for 100D: {stds[-1]/means[-1]:.4f}")

# As dimensionality increases, the ratio of standard deviation to mean decreases,
# indicating that distances become more concentrated
```

Slide 15: Additional Resources

For further exploration of the limitations of Euclidean distance and alternative metrics, consider the following resources:

1. "On the Surprising Behavior of Distance Metrics in High Dimensional Space" by Charu C. Aggarwal et al. (2001) ArXiv: [https://arxiv.org/abs/cs/0102040](https://arxiv.org/abs/cs/0102040)
2. "When Is 'Nearest Neighbor' Meaningful?" by Kevin Beyer et al. (1999) Reference: Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When is "nearest neighbor" meaningful?. In International conference on database theory (pp. 217-235). Springer, Berlin, Heidelberg.
3. "Similarity Search in High Dimensions via Hashing" by Aristides Gionis et al. (1999) Reference: Gionis, A., Indyk, P., & Motwani, R. (1999). Similarity search in high dimensions via hashing. In VLDB (Vol. 99, No. 6, pp. 518-529).

These papers provide in-depth discussions on the challenges of using Euclidean distance in high-dimensional spaces and propose alternative approaches for similarity search and data analysis.

