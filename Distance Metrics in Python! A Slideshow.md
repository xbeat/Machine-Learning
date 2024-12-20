## Distance Metrics in Python! A Slideshow
Slide 1: Introduction to Distance Metrics

Distance metrics are mathematical functions used to measure the similarity or dissimilarity between two points in a given space. They play a crucial role in various fields, including machine learning, data analysis, and information retrieval. This slideshow will demystify distance metrics and demonstrate their implementation using Python.

```python
import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Example usage
p1 = np.array([1, 2, 3])
p2 = np.array([4, 5, 6])
distance = euclidean_distance(p1, p2)
print(f"Euclidean distance between {p1} and {p2}: {distance:.2f}")
```

Slide 2: Euclidean Distance

Euclidean distance is the most common distance metric, representing the straight-line distance between two points in Euclidean space. It is calculated as the square root of the sum of squared differences between corresponding coordinates.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_euclidean_distance(p1, p2):
    plt.figure(figsize=(8, 6))
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'ro-')
    plt.plot([p1[0], p2[0]], [p1[1], p1[1]], 'b--')
    plt.plot([p2[0], p2[0]], [p1[1], p2[1]], 'b--')
    plt.text((p1[0] + p2[0])/2, p1[1] - 0.1, f'{abs(p2[0] - p1[0]):.2f}', ha='center')
    plt.text(p2[0] + 0.1, (p1[1] + p2[1])/2, f'{abs(p2[1] - p1[1]):.2f}', va='center')
    plt.text((p1[0] + p2[0])/2, (p1[1] + p2[1])/2, f'{euclidean_distance(p1, p2):.2f}', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    plt.xlim(min(p1[0], p2[0]) - 1, max(p1[0], p2[0]) + 1)
    plt.ylim(min(p1[1], p2[1]) - 1, max(p1[1], p2[1]) + 1)
    plt.grid(True)
    plt.title('Euclidean Distance Visualization')
    plt.show()

p1 = np.array([1, 1])
p2 = np.array([4, 5])
plot_euclidean_distance(p1, p2)
```

Slide 3: Manhattan Distance

Manhattan distance, also known as L1 distance or city block distance, measures the sum of absolute differences between coordinates. It represents the distance a taxi would drive in a city laid out in a grid-like pattern.

```python
def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

p1 = np.array([1, 2, 3])
p2 = np.array([4, 5, 6])
distance = manhattan_distance(p1, p2)
print(f"Manhattan distance between {p1} and {p2}: {distance}")

# Visualizing Manhattan vs Euclidean distance
def plot_manhattan_vs_euclidean(p1, p2):
    plt.figure(figsize=(8, 6))
    plt.plot([p1[0], p2[0]], [p1[1], p1[1]], 'r-', label='Manhattan')
    plt.plot([p2[0], p2[0]], [p1[1], p2[1]], 'r-')
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b--', label='Euclidean')
    plt.plot(p1[0], p1[1], 'go', label='Point 1')
    plt.plot(p2[0], p2[1], 'go', label='Point 2')
    plt.legend()
    plt.grid(True)
    plt.title('Manhattan vs Euclidean Distance')
    plt.show()

plot_manhattan_vs_euclidean(np.array([1, 1]), np.array([4, 5]))
```

Slide 4: Minkowski Distance

Minkowski distance is a generalization of Euclidean and Manhattan distances. It is parameterized by a value p, where p=1 gives Manhattan distance, p=2 gives Euclidean distance, and p→∞ gives Chebyshev distance.

```python
def minkowski_distance(point1, point2, p):
    return np.power(np.sum(np.abs(point1 - point2)**p), 1/p)

p1 = np.array([1, 2, 3])
p2 = np.array([4, 5, 6])

for p in [1, 2, 3, np.inf]:
    distance = minkowski_distance(p1, p2, p)
    print(f"Minkowski distance (p={p}) between {p1} and {p2}: {distance:.2f}")
```

Slide 5: Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors, indicating their directional similarity regardless of magnitude. It's particularly useful in text analysis and recommendation systems.

```python
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
similarity = cosine_similarity(v1, v2)
print(f"Cosine similarity between {v1} and {v2}: {similarity:.4f}")

# Visualizing cosine similarity
def plot_cosine_similarity(v1, v2):
    plt.figure(figsize=(8, 6))
    plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector 1')
    plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector 2')
    plt.xlim(-1, max(v1[0], v2[0]) + 1)
    plt.ylim(-1, max(v1[1], v2[1]) + 1)
    plt.legend()
    plt.grid(True)
    plt.title(f'Cosine Similarity: {cosine_similarity(v1, v2):.4f}')
    plt.show()

plot_cosine_similarity(np.array([1, 2]), np.array([2, 3]))
```

Slide 6: Hamming Distance

Hamming distance measures the number of positions at which corresponding symbols in two equal-length strings or vectors are different. It's commonly used in information theory and error detection.

```python
def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Strings must have equal length")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

string1 = "karolin"
string2 = "kathrin"
distance = hamming_distance(string1, string2)
print(f"Hamming distance between '{string1}' and '{string2}': {distance}")

# Visualizing Hamming distance
def visualize_hamming_distance(s1, s2):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    for i, (c1, c2) in enumerate(zip(s1, s2)):
        color = 'red' if c1 != c2 else 'green'
        ax.text(i, 0.6, c1, ha='center', va='center', color=color, fontsize=20)
        ax.text(i, 0.4, c2, ha='center', va='center', color=color, fontsize=20)
    plt.title(f"Hamming Distance: {hamming_distance(s1, s2)}")
    plt.tight_layout()
    plt.show()

visualize_hamming_distance(string1, string2)
```

Slide 7: Levenshtein Distance

Levenshtein distance, also known as edit distance, measures the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one word into another. It's widely used in natural language processing and bioinformatics.

```python
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

word1 = "kitten"
word2 = "sitting"
distance = levenshtein_distance(word1, word2)
print(f"Levenshtein distance between '{word1}' and '{word2}': {distance}")
```

Slide 8: Jaccard Similarity

Jaccard similarity measures the similarity between finite sample sets by comparing the size of their intersection to the size of their union. It's particularly useful for comparing text documents or gene sequences.

```python
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

text1 = "The quick brown fox"
text2 = "The fast brown dog"
set1 = set(text1.lower().split())
set2 = set(text2.lower().split())
similarity = jaccard_similarity(set1, set2)
print(f"Jaccard similarity between '{text1}' and '{text2}': {similarity:.4f}")

# Visualizing Jaccard similarity
def plot_jaccard_similarity(set1, set2):
    plt.figure(figsize=(8, 6))
    venn2([set1, set2], ('Set 1', 'Set 2'))
    plt.title(f'Jaccard Similarity: {jaccard_similarity(set1, set2):.4f}')
    plt.show()

plot_jaccard_similarity(set1, set2)
```

Slide 9: Real-Life Example: Image Similarity

Distance metrics can be used to compare images by treating them as high-dimensional vectors. Here's an example using Mean Squared Error (MSE) to compare two images:

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def mse_distance(img1, img2):
    return np.mean((img1 - img2) ** 2)

# Load and preprocess images
def load_image(path):
    return np.array(Image.open(path).convert('L').resize((100, 100)))

img1 = load_image('image1.jpg')
img2 = load_image('image2.jpg')

distance = mse_distance(img1, img2)
print(f"MSE distance between images: {distance:.2f}")

# Visualize images and their difference
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(img1, cmap='gray')
ax1.set_title('Image 1')
ax2.imshow(img2, cmap='gray')
ax2.set_title('Image 2')
ax3.imshow(np.abs(img1 - img2), cmap='hot')
ax3.set_title('Absolute Difference')
plt.show()
```

Slide 10: Real-Life Example: Text Document Similarity

Distance metrics are crucial in natural language processing for tasks like document clustering and information retrieval. Here's an example using cosine similarity to compare text documents:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "The quick brown fox jumps over the lazy dog",
    "A fast brown dog leaps over the sleepy cat",
    "Python is a versatile programming language",
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

similarity_matrix = cosine_similarity(tfidf_matrix)

print("Document Similarity Matrix:")
print(similarity_matrix)

# Visualize similarity matrix
plt.figure(figsize=(8, 6))
plt.imshow(similarity_matrix, cmap='YlOrRd')
plt.colorbar()
plt.title('Document Similarity Matrix')
plt.xlabel('Document Index')
plt.ylabel('Document Index')
plt.show()
```

Slide 11: Choosing the Right Distance Metric

The choice of distance metric depends on the nature of your data and the problem you're trying to solve. Consider these factors:

1. Data type: Continuous, categorical, or mixed
2. Dimensionality: Low or high-dimensional data
3. Scale sensitivity: Whether differences in scale matter
4. Interpretability: How easily the metric can be understood
5. Computational efficiency: Important for large datasets

Example: For text data, cosine similarity is often preferred as it focuses on the direction of vectors rather than their magnitude.

```python
# Comparison of different distance metrics
from scipy.spatial.distance import euclidean, cityblock, cosine

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"Euclidean distance: {euclidean(v1, v2):.4f}")
print(f"Manhattan distance: {cityblock(v1, v2):.4f}")
print(f"Cosine distance: {cosine(v1, v2):.4f}")
```

Slide 12: Implementing Custom Distance Metrics

Sometimes, you may need to implement a custom distance metric tailored to your specific problem. Here's an example of a weighted Euclidean distance:

```python
import numpy as np
import matplotlib.pyplot as plt

def weighted_euclidean_distance(point1, point2, weights):
    return np.sqrt(np.sum(weights * (point1 - point2)**2))

p1 = np.array([1, 2, 3])
p2 = np.array([4, 5, 6])
weights = np.array([0.5, 1, 2])  # More weight on the third dimension

distance = weighted_euclidean_distance(p1, p2, weights)
print(f"Weighted Euclidean distance: {distance:.4f}")

# Visualizing the effect of weights
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.bar(range(3), np.abs(p1 - p2), label='Unweighted')
ax1.bar(range(3), weights * np.abs(p1 - p2), alpha=0.5, label='Weighted')
ax1.set_title('Dimension-wise Differences')
ax1.legend()

ax2.scatter(p1[0], p1[1], s=100, label='Point 1')
ax2.scatter(p2[0], p2[1], s=100, label='Point 2')
ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--')
ax2.set_title('2D Projection')
ax2.legend()

plt.tight_layout()
plt.show()
```

Slide 13: Optimizing Distance Computations

When working with large datasets, efficient distance computation becomes crucial. Here are some strategies to optimize distance calculations:

```python
import numpy as np
from scipy.spatial.distance import cdist

# Generate random data
n_samples = 1000
n_features = 10
X = np.random.rand(n_samples, n_features)
Y = np.random.rand(n_samples, n_features)

# Efficient pairwise distance computation
distances = cdist(X, Y, metric='euclidean')

print(f"Shape of distance matrix: {distances.shape}")
print(f"Mean distance: {np.mean(distances):.4f}")

# K-nearest neighbors using efficient algorithms
from sklearn.neighbors import NearestNeighbors

k = 5
nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
nn.fit(X)
distances, indices = nn.kneighbors(Y)

print(f"Shape of k-NN distances: {distances.shape}")
print(f"Mean k-NN distance: {np.mean(distances):.4f}")
```

Slide 14: Dimensionality Reduction and Distance Metrics

High-dimensional data can suffer from the "curse of dimensionality," affecting distance metrics. Dimensionality reduction techniques can help mitigate this issue:

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Generate high-dimensional data
n_samples = 1000
n_features = 50
X = np.random.rand(n_samples, n_features)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
ax1.set_title('PCA')
ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
ax2.set_title('t-SNE')
plt.tight_layout()
plt.show()

print(f"Explained variance ratio (PCA): {pca.explained_variance_ratio_}")
```

Slide 15: Additional Resources

For further exploration of distance metrics and their applications, consider these resources:

1. "A Survey of Binary Similarity and Distance Measures" by Seung-Seok Choi et al. (2010) ArXiv: [https://arxiv.org/abs/1002.0184](https://arxiv.org/abs/1002.0184)
2. "Similarity Measures for Text Document Clustering" by Anna Huang (2008) Available at: [http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.332.4480](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.332.4480)
3. "An Introduction to Information Retrieval" by Christopher D. Manning et al. Available at: [https://nlp.stanford.edu/IR-book/](https://nlp.stanford.edu/IR-book/)

These resources provide in-depth discussions on various distance metrics, their properties, and applications in different domains such as information retrieval, natural language processing, and data mining.

