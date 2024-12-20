## The Hidden Pitfalls of Cosine Similarity Loss in Python
Slide 1: Introduction to Cosine Similarity Loss

Cosine Similarity Loss is a popular metric in machine learning, particularly in natural language processing and recommendation systems. It measures the cosine of the angle between two non-zero vectors, providing a value between -1 and 1. While widely used, it has several hidden pitfalls that can lead to unexpected results.

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

similarity = cosine_similarity(v1, v2)
print(f"Cosine similarity: {similarity:.4f}")
```

Slide 2: The Magnitude Blindness Issue

One of the main pitfalls of cosine similarity is its blindness to vector magnitudes. It only considers the angle between vectors, potentially leading to misleading results when the magnitude difference is significant.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_vectors(v1, v2):
    plt.figure(figsize=(8, 6))
    plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
    plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

v1 = np.array([1, 1])
v2 = np.array([5, 5])

print(f"Cosine similarity: {cosine_similarity(v1, v2):.4f}")
plot_vectors(v1, v2)
```

Slide 3: Negative Values and Their Impact

Cosine similarity can handle negative values, but this can lead to counterintuitive results. Negative similarities might be difficult to interpret in certain contexts, especially when working with non-negative data.

```python
v1 = np.array([1, -2, 3])
v2 = np.array([-1, 2, -3])

similarity = cosine_similarity(v1, v2)
print(f"Cosine similarity: {similarity:.4f}")

# Visualizing the angle between vectors
angle = np.arccos(similarity)
print(f"Angle between vectors: {np.degrees(angle):.2f} degrees")
```

Slide 4: The Curse of Dimensionality

As the number of dimensions increases, cosine similarity tends to approach 0 for most pairs of vectors. This phenomenon, known as the curse of dimensionality, can make it challenging to distinguish between similar and dissimilar items in high-dimensional spaces.

```python
import numpy as np

def random_high_dim_vectors(dim, num_vectors):
    return np.random.randn(num_vectors, dim)

dimensions = [10, 100, 1000, 10000]
num_vectors = 1000

for dim in dimensions:
    vectors = random_high_dim_vectors(dim, num_vectors)
    similarities = []
    
    for i in range(num_vectors):
        for j in range(i+1, num_vectors):
            similarities.append(cosine_similarity(vectors[i], vectors[j]))
    
    print(f"Dimension: {dim}")
    print(f"Average similarity: {np.mean(similarities):.4f}")
    print(f"Standard deviation: {np.std(similarities):.4f}\n")
```

Slide 5: Sensitivity to Small Changes

Cosine similarity can be sensitive to small changes in vector components, especially when dealing with sparse data. This sensitivity can lead to unexpected fluctuations in similarity scores.

```python
import numpy as np

def sparse_vector(size, non_zero_elements):
    v = np.zeros(size)
    v[np.random.choice(size, non_zero_elements, replace=False)] = np.random.rand(non_zero_elements)
    return v

size = 1000
non_zero = 10

v1 = sparse_vector(size, non_zero)
v2 = v1.()

# Introduce a small change
v2[np.random.randint(size)] += 0.01

similarity_before = cosine_similarity(v1, v1)
similarity_after = cosine_similarity(v1, v2)

print(f"Similarity before change: {similarity_before:.6f}")
print(f"Similarity after small change: {similarity_after:.6f}")
print(f"Difference: {similarity_before - similarity_after:.6f}")
```

Slide 6: Real-life Example: Document Similarity

In document similarity tasks, cosine similarity is often used to compare the content of two documents. However, it can lead to misleading results when documents have different lengths or varying vocabulary richness.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog sleeps all day",
    "The quick brown fox is agile and swift"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

for i, doc1 in enumerate(documents):
    for j, doc2 in enumerate(documents):
        if i < j:
            similarity = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
            print(f"Similarity between doc{i+1} and doc{j+1}: {similarity:.4f}")
```

Slide 7: Addressing Magnitude Blindness

To address the magnitude blindness issue, we can combine cosine similarity with Euclidean distance. This approach provides a more comprehensive similarity measure that considers both angle and magnitude.

```python
def combined_similarity(a, b, alpha=0.5):
    cos_sim = cosine_similarity(a, b)
    euclidean_dist = np.linalg.norm(a - b)
    max_dist = np.linalg.norm(a) + np.linalg.norm(b)
    normalized_dist = 1 - (euclidean_dist / max_dist)
    return alpha * cos_sim + (1 - alpha) * normalized_dist

v1 = np.array([1, 1])
v2 = np.array([5, 5])

cos_sim = cosine_similarity(v1, v2)
combined_sim = combined_similarity(v1, v2)

print(f"Cosine similarity: {cos_sim:.4f}")
print(f"Combined similarity: {combined_sim:.4f}")
```

Slide 8: Handling Negative Values

When working with data that should not have negative similarities, we can use the angular similarity instead of cosine similarity. This ensures that the similarity values are always between 0 and 1.

```python
def angular_similarity(a, b):
    cos_sim = cosine_similarity(a, b)
    return 1 - np.arccos(cos_sim) / np.pi

v1 = np.array([1, -2, 3])
v2 = np.array([-1, 2, -3])

cos_sim = cosine_similarity(v1, v2)
ang_sim = angular_similarity(v1, v2)

print(f"Cosine similarity: {cos_sim:.4f}")
print(f"Angular similarity: {ang_sim:.4f}")
```

Slide 9: Mitigating the Curse of Dimensionality

To address the curse of dimensionality, we can use dimensionality reduction techniques like Principal Component Analysis (PCA) before applying cosine similarity. This helps preserve the most important features while reducing the impact of high dimensionality.

```python
from sklearn.decomposition import PCA

def cosine_similarity_with_pca(X, n_components=0.95):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    
    similarities = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(i, X.shape[0]):
            similarities[i, j] = similarities[j, i] = cosine_similarity(X_reduced[i].reshape(1, -1), X_reduced[j].reshape(1, -1))[0][0]
    
    return similarities

# Generate high-dimensional data
X = np.random.randn(100, 1000)

# Compare similarities with and without PCA
similarities_original = cosine_similarity(X)
similarities_pca = cosine_similarity_with_pca(X)

print("Original similarities:")
print(f"Mean: {np.mean(similarities_original):.4f}")
print(f"Std: {np.std(similarities_original):.4f}")

print("\nSimilarities after PCA:")
print(f"Mean: {np.mean(similarities_pca):.4f}")
print(f"Std: {np.std(similarities_pca):.4f}")
```

Slide 10: Robust Similarity Measures

To address the sensitivity to small changes, we can use more robust similarity measures such as the Soft Cosine Similarity. This measure takes into account the relationships between different features.

```python
import numpy as np
from scipy.spatial.distance import cdist

def soft_cosine_similarity(a, b, similarity_matrix):
    numerator = a @ similarity_matrix @ b.T
    denominator = np.sqrt((a @ similarity_matrix @ a.T) * (b @ similarity_matrix @ b.T))
    return numerator / denominator

# Create a simple similarity matrix based on feature correlations
features = ['apple', 'orange', 'banana', 'grape']
vectors = np.random.rand(len(features), 10)
similarity_matrix = 1 - cdist(vectors, vectors, metric='correlation')

# Example vectors
v1 = np.array([0.8, 0.2, 0.0, 0.0])
v2 = np.array([0.9, 0.0, 0.1, 0.0])

cos_sim = cosine_similarity(v1, v2)
soft_cos_sim = soft_cosine_similarity(v1, v2, similarity_matrix)

print(f"Cosine similarity: {cos_sim:.4f}")
print(f"Soft cosine similarity: {soft_cos_sim:.4f}")
```

Slide 11: Real-life Example: Image Similarity

In image similarity tasks, cosine similarity can be applied to feature vectors extracted from images. However, it may not capture all aspects of visual similarity, such as color distribution or spatial relationships.

```python
import numpy as np
from skimage import io, color, feature

def image_similarity(img1_path, img2_path):
    # Load and preprocess images
    img1 = io.imread(img1_path)
    img2 = io.imread(img2_path)
    
    # Convert to grayscale
    img1_gray = color.rgb2gray(img1)
    img2_gray = color.rgb2gray(img2)
    
    # Extract HOG features
    hog1 = feature.hog(img1_gray)
    hog2 = feature.hog(img2_gray)
    
    # Calculate cosine similarity
    return cosine_similarity(hog1.reshape(1, -1), hog2.reshape(1, -1))[0][0]

# Example usage (replace with actual image paths)
img1_path = "path/to/image1.jpg"
img2_path = "path/to/image2.jpg"

similarity = image_similarity(img1_path, img2_path)
print(f"Image similarity: {similarity:.4f}")
```

Slide 12: Alternative Similarity Measures

While cosine similarity is widely used, there are alternative measures that can be more appropriate in certain scenarios. Here are a few examples:

```python
def jaccard_similarity(a, b):
    intersection = np.minimum(a, b).sum()
    union = np.maximum(a, b).sum()
    return intersection / union

def pearson_correlation(a, b):
    return np.corrcoef(a, b)[0, 1]

def euclidean_similarity(a, b):
    return 1 / (1 + np.linalg.norm(a - b))

# Example vectors
v1 = np.array([1, 2, 3, 4, 5])
v2 = np.array([2, 3, 4, 5, 6])

print(f"Cosine similarity: {cosine_similarity(v1, v2):.4f}")
print(f"Jaccard similarity: {jaccard_similarity(v1, v2):.4f}")
print(f"Pearson correlation: {pearson_correlation(v1, v2):.4f}")
print(f"Euclidean similarity: {euclidean_similarity(v1, v2):.4f}")
```

Slide 13: Choosing the Right Similarity Measure

Selecting the appropriate similarity measure depends on your specific use case and data characteristics. Consider these factors when choosing:

1. Data type (binary, continuous, categorical)
2. Importance of magnitude vs. direction
3. Presence of negative values
4. Dimensionality of the data
5. Sensitivity to outliers or noise

```python
def choose_similarity_measure(data_type, magnitude_important, has_negatives, high_dimensional, sensitive_to_outliers):
    if data_type == 'binary':
        return "Jaccard similarity"
    elif has_negatives and not magnitude_important:
        return "Cosine similarity"
    elif magnitude_important and not has_negatives:
        return "Euclidean similarity"
    elif high_dimensional and not sensitive_to_outliers:
        return "Cosine similarity with PCA"
    elif sensitive_to_outliers:
        return "Spearman rank correlation"
    else:
        return "Pearson correlation"

# Example usage
measure = choose_similarity_measure(
    data_type='continuous',
    magnitude_important=True,
    has_negatives=False,
    high_dimensional=True,
    sensitive_to_outliers=False
)

print(f"Recommended similarity measure: {measure}")
```

Slide 14: Additional Resources

For more information on cosine similarity and its alternatives, consider exploring the following resources:

1. "Understanding the Limitations of Cosine Similarity for Word Embeddings" (ArXiv:1908.09023) URL: [https://arxiv.org/abs/1908.09023](https://arxiv.org/abs/1908.09023)
2. "Soft Cosine Measure: Similarity of Features in Vector Space Model" (ArXiv:1808.10574) URL: [https://arxiv.org/abs/1808.10574](https://arxiv.org/abs/1808.10574)
3. "A Survey of Text Similarity Approaches" (ArXiv:1905.08523) URL: [https://arxiv.org/abs/1905.08523](https://arxiv.org/abs/1905.08523)

These papers provide in-depth analyses of similarity measures and their applications in various domains.

