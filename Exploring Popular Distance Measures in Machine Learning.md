## Exploring Popular Distance Measures in Machine Learning
Slide 1: Understanding Euclidean Distance

The Euclidean distance represents the shortest straight-line path between two points in a multi-dimensional space. It's calculated as the square root of the sum of squared differences between corresponding coordinates, making it particularly effective for continuous numerical data in machine learning applications.

```python
import numpy as np

def euclidean_distance(x1, x2):
    # Convert inputs to numpy arrays for vectorized operations
    x1, x2 = np.array(x1), np.array(x2)
    
    # Calculate squared differences and sum them
    squared_diff = np.sum((x1 - x2) ** 2)
    
    # Return square root of sum
    return np.sqrt(squared_diff)

# Example usage
point1 = [1, 2, 3]
point2 = [4, 5, 6]
distance = euclidean_distance(point1, point2)
print(f"Euclidean distance: {distance:.2f}")  # Output: Euclidean distance: 5.20
```

Slide 2: Implementing Cosine Similarity

Cosine similarity measures the cosine of the angle between two non-zero vectors, providing a similarity score between -1 and 1. This metric is particularly useful in text analysis and recommendation systems where the magnitude of vectors is less important than their directional relationships.

```python
def cosine_similarity(v1, v2):
    # Convert inputs to numpy arrays
    v1, v2 = np.array(v1), np.array(v2)
    
    # Calculate dot product
    dot_product = np.dot(v1, v2)
    
    # Calculate magnitudes
    magnitude1 = np.sqrt(np.sum(v1**2))
    magnitude2 = np.sqrt(np.sum(v2**2))
    
    # Return cosine similarity
    return dot_product / (magnitude1 * magnitude2)

# Example usage
vector1 = [1, 2, 3]
vector2 = [4, 5, 6]
similarity = cosine_similarity(vector1, vector2)
print(f"Cosine similarity: {similarity:.4f}")  # Output: Cosine similarity: 0.9746
```

Slide 3: Hamming Distance Implementation

The Hamming distance quantifies the number of positions at which two sequences differ, making it invaluable for error detection in communication systems and binary string comparison in computational biology and information theory.

```python
def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of equal length")
    
    # Count positions where characters differ
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

# Example with binary strings
binary1 = "1010101"
binary2 = "1110001"
distance = hamming_distance(binary1, binary2)
print(f"Hamming distance: {distance}")  # Output: Hamming distance: 2

# Example with DNA sequences
dna1 = "ATCGAT"
dna2 = "ATCTAT"
distance = hamming_distance(dna1, dna2)
print(f"DNA sequence distance: {distance}")  # Output: DNA sequence distance: 1
```

Slide 4: Manhattan Distance Calculator

Manhattan distance, also known as L1 distance or city block distance, calculates the sum of absolute differences between coordinates. This metric is particularly useful in grid-based pathfinding and when dealing with discrete movement constraints.

```python
def manhattan_distance(p1, p2):
    # Convert inputs to numpy arrays
    p1, p2 = np.array(p1), np.array(p2)
    
    # Calculate absolute differences and sum
    return np.sum(np.abs(p1 - p2))

# Example in 2D space
point1 = [1, 1]
point2 = [4, 5]
distance = manhattan_distance(point1, point2)
print(f"Manhattan distance: {distance}")  # Output: Manhattan distance: 7

# Example in 3D space
point3d1 = [1, 2, 3]
point3d2 = [4, 6, 8]
distance_3d = manhattan_distance(point3d1, point3d2)
print(f"3D Manhattan distance: {distance_3d}")  # Output: 3D Manhattan distance: 12
```

Slide 5: Minkowski Distance Generalization

The Minkowski distance represents a generalization of Euclidean and Manhattan distances through a parameter p, offering flexibility in distance measurement. When p=1, it becomes Manhattan distance; when p=2, it becomes Euclidean distance; as p approaches infinity, it converges to Chebyshev distance.

```python
def minkowski_distance(x1, x2, p):
    # Convert inputs to numpy arrays
    x1, x2 = np.array(x1), np.array(x2)
    
    # Calculate Minkowski distance
    return np.power(np.sum(np.power(np.abs(x1 - x2), p)), 1/p)

# Example with different p values
point1 = [1, 2, 3]
point2 = [4, 5, 6]

# Compare different p values
for p in [1, 2, 3]:
    dist = minkowski_distance(point1, point2, p)
    print(f"Minkowski distance (p={p}): {dist:.4f}")
```

Slide 6: Chebyshev Distance Implementation

Chebyshev distance, also known as L∞ norm, measures the maximum absolute difference between corresponding dimensions of two points. It's particularly useful in chess move calculations and applications where the maximum difference in any dimension is critical.

```python
def chebyshev_distance(x1, x2):
    # Convert inputs to numpy arrays
    x1, x2 = np.array(x1), np.array(x2)
    
    # Return maximum absolute difference
    return np.max(np.abs(x1 - x2))

# Example usage
point1 = [1, 2, 3]
point2 = [4, 6, 5]
distance = chebyshev_distance(point1, point2)
print(f"Chebyshev distance: {distance}")  # Output: 4

# Chess move example
king_pos = [4, 4]  # e4
target_pos = [6, 5]  # f6
moves_needed = chebyshev_distance(king_pos, target_pos)
print(f"Minimum king moves needed: {moves_needed}")  # Output: 2
```

Slide 7: Jaccard Distance Calculator

The Jaccard distance measures dissimilarity between finite sample sets by comparing the ratio of intersection to union. This metric is particularly valuable in document similarity analysis, clustering, and biological taxonomy applications.

```python
def jaccard_distance(set1, set2):
    # Convert inputs to sets if they aren't already
    set1, set2 = set(set1), set(set2)
    
    # Calculate intersection and union
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    # Return Jaccard distance
    return 1 - (intersection / union)

# Example with text analysis
text1 = set("machine learning".split())
text2 = set("deep learning".split())
distance = jaccard_distance(text1, text2)
print(f"Jaccard distance: {distance:.4f}")  # Output: 0.6667

# Example with gene sets
genes1 = {"BRCA1", "TP53", "KRAS", "PTEN"}
genes2 = {"BRCA1", "EGFR", "KRAS", "BRAF"}
gene_distance = jaccard_distance(genes1, genes2)
print(f"Gene set distance: {gene_distance:.4f}")  # Output: 0.6667
```

Slide 8: Haversine Distance Implementation

The Haversine formula calculates great-circle distances between points on a sphere using their latitude and longitude coordinates. Essential for geographical applications, navigation systems, and location-based services.

```python
import math

def haversine_distance(coord1, coord2, radius=6371):
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return radius * c

# Example: Distance between New York and London
new_york = (40.7128, -74.0060)
london = (51.5074, -0.1278)
distance = haversine_distance(new_york, london)
print(f"Distance between New York and London: {distance:.2f} km")
```

Slide 9: Sørensen-Dice Coefficient Implementation

The Sørensen-Dice coefficient measures similarity between two samples by comparing their overlap, commonly used in image segmentation and ecological sampling. It emphasizes common elements between sets while accounting for their sizes, making it robust for varying sample sizes.

```python
def sorensen_dice_coefficient(set1, set2):
    # Convert inputs to sets
    set1, set2 = set(set1), set(set2)
    
    # Calculate intersection and individual set sizes
    intersection = len(set1.intersection(set2))
    
    # Calculate coefficient
    return 2 * intersection / (len(set1) + len(set2))

def sorensen_dice_distance(set1, set2):
    return 1 - sorensen_dice_coefficient(set1, set2)

# Example with species presence/absence data
location1 = {'wolf', 'bear', 'deer', 'fox', 'rabbit'}
location2 = {'bear', 'deer', 'rabbit', 'raccoon', 'squirrel'}
similarity = sorensen_dice_coefficient(location1, location2)
distance = sorensen_dice_distance(location1, location2)

print(f"Similarity: {similarity:.4f}")  # Output: 0.6000
print(f"Distance: {distance:.4f}")     # Output: 0.4000
```

Slide 10: Real-world Application - Text Document Similarity

Implementation of multiple distance metrics for comparing text documents, demonstrating preprocessing, vectorization, and similarity computation in a practical document comparison scenario.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine, euclidean

def compare_documents(doc1, doc2):
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit and transform documents
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    
    # Convert to dense arrays
    vec1, vec2 = tfidf_matrix.toarray()[0], tfidf_matrix.toarray()[1]
    
    # Calculate different similarity metrics
    cosine_sim = 1 - cosine(vec1, vec2)
    euclidean_dist = euclidean(vec1, vec2)
    
    return {
        'cosine_similarity': cosine_sim,
        'euclidean_distance': euclidean_dist
    }

# Example usage
doc1 = """Machine learning is a field of artificial intelligence
          that uses statistical techniques to give computer systems
          the ability to learn from data."""

doc2 = """Deep learning is a subset of machine learning that 
          uses neural networks to learn from large amounts of data."""

results = compare_documents(doc1, doc2)
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
```

Slide 11: Real-world Application - Clustering Gene Expression Data

Implementation of distance-based clustering for gene expression analysis, demonstrating practical application in bioinformatics with multiple distance metrics.

```python
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

def analyze_gene_expression(expression_data, n_clusters=3):
    # Normalize expression data
    normalized_data = (expression_data - np.mean(expression_data, axis=0)) / \
                     np.std(expression_data, axis=0)
    
    # Perform hierarchical clustering with different metrics
    metrics = ['euclidean', 'manhattan', 'cosine']
    results = {}
    
    for metric in metrics:
        # Create linkage matrix
        linkage_matrix = linkage(normalized_data, method='ward', metric=metric)
        
        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric,
            linkage='ward'
        ).fit(normalized_data)
        
        results[metric] = clustering.labels_
    
    return results

# Example usage with synthetic data
np.random.seed(42)
n_samples = 100
n_features = 10
expression_data = np.random.rand(n_samples, n_features)

results = analyze_gene_expression(expression_data)
for metric, labels in results.items():
    unique_clusters = len(np.unique(labels))
    print(f"{metric} clustering found {unique_clusters} clusters")
```

Slide 12: Performance Analysis of Distance Metrics

An analysis framework for comparing different distance metrics on classification tasks, including cross-validation and performance metrics. This implementation helps in selecting the most appropriate distance measure for specific datasets.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

def compare_distance_metrics(X, y, metrics=['euclidean', 'manhattan', 'chebyshev']):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    for metric in metrics:
        # Initialize classifier
        clf = KNeighborsClassifier(n_neighbors=5, metric=metric)
        
        # Perform cross-validation
        scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
        
        results[metric] = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std()
        }
    
    return results

# Example usage with synthetic dataset
np.random.seed(42)
X = np.random.randn(200, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

results = compare_distance_metrics(X, y)
for metric, scores in results.items():
    print(f"\n{metric.capitalize()} Distance:")
    print(f"Mean Accuracy: {scores['mean_accuracy']:.4f}")
    print(f"Std Accuracy: {scores['std_accuracy']:.4f}")
```

Slide 13: Advanced Visualization of Distance Relationships

Implementation of a comprehensive visualization system for understanding relationships between different distance metrics and their impact on data clustering.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def visualize_distance_relationships(X, metrics=['euclidean', 'cityblock', 'cosine']):
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, n_metrics, figsize=(12, 12))
    
    distance_matrices = {
        metric: squareform(pdist(X, metric=metric))
        for metric in metrics
    }
    
    for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
            if i != j:
                # Scatter plot of distances
                axes[i, j].scatter(
                    distance_matrices[metric1].flatten(),
                    distance_matrices[metric2].flatten(),
                    alpha=0.1
                )
                axes[i, j].set_xlabel(f'{metric1} distance')
                axes[i, j].set_ylabel(f'{metric2} distance')
            else:
                # Histogram on diagonal
                axes[i, i].hist(distance_matrices[metric1].flatten(), bins=50)
                axes[i, i].set_title(f'{metric1} distribution')
    
    plt.tight_layout()
    return fig

# Example usage
np.random.seed(42)
X = np.random.randn(100, 5)
fig = visualize_distance_relationships(X)
plt.show()
```

Slide 14: Additional Resources

*   A Survey of Distance and Similarity Measures for Structured Data [https://arxiv.org/abs/2009.09972](https://arxiv.org/abs/2009.09972)
*   Comprehensive Analysis of Distance Metrics in Machine Learning [https://arxiv.org/abs/2006.13418](https://arxiv.org/abs/2006.13418)
*   Distance Metric Learning: A Comprehensive Survey [https://arxiv.org/abs/1907.08292](https://arxiv.org/abs/1907.08292)
*   Applications of Distance Metrics in Deep Learning [https://www.sciencedirect.com/science/article/pii/S0031320321002028](https://www.sciencedirect.com/science/article/pii/S0031320321002028)
*   Interactive tutorials and implementations:
    *   [https://scikit-learn.org/stable/modules/metrics.html](https://scikit-learn.org/stable/modules/metrics.html)
    *   [https://pytorch.org/docs/stable/nn.html#distance-functions](https://pytorch.org/docs/stable/nn.html#distance-functions)

