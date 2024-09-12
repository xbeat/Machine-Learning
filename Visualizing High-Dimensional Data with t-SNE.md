## Visualizing High-Dimensional Data with t-SNE
Slide 1: Introduction to t-SNE

t-Distributed Stochastic Neighbor Embedding (t-SNE) is a machine learning algorithm for visualization of high-dimensional data. It reduces dimensionality while preserving local structure, making it easier to identify patterns and clusters in complex datasets.

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Generate sample high-dimensional data
n_samples = 1000
n_features = 50
X = np.random.randn(n_samples, n_features)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize the result
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
plt.title('t-SNE visualization of high-dimensional data')
plt.show()
```

Slide 2: The Curse of Dimensionality

The curse of dimensionality refers to various phenomena that arise when analyzing data in high-dimensional spaces. t-SNE helps overcome this challenge by reducing the number of dimensions while maintaining important relationships between data points.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_hypersphere_points(n_points, n_dims):
    points = np.random.randn(n_points, n_dims)
    return points / np.linalg.norm(points, axis=1)[:, np.newaxis]

dims = [2, 3, 5, 10, 20, 50, 100]
n_points = 1000
distances = []

for dim in dims:
    points = generate_hypersphere_points(n_points, dim)
    dist = np.linalg.norm(points[0] - points[1:], axis=1)
    distances.append(dist)

plt.figure(figsize=(10, 6))
plt.boxplot(distances, labels=dims)
plt.title('Distance distribution in high-dimensional spaces')
plt.xlabel('Number of dimensions')
plt.ylabel('Distance')
plt.show()
```

Slide 3: t-SNE Algorithm Overview

t-SNE works by converting similarities between data points to joint probabilities and minimizes the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data.

```python
import numpy as np

def compute_pairwise_affinities(X, perplexity=30.0, tol=1e-5, max_iter=50):
    n = X.shape[0]
    distances = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
    P = np.zeros((n, n))
    
    for i in range(n):
        beta = 1.0
        diff = 1.0
        for _ in range(max_iter):
            sum_Pi = np.sum(np.exp(-distances[i] * beta))
            if np.abs(np.log(sum_Pi) + beta * distances[i].dot(P[i])) < tol:
                break
            beta *= 2.0 if diff > 0 else 0.5
            diff = np.log(sum_Pi) + beta * distances[i].dot(P[i]) - np.log(perplexity)
        
        P[i] = np.exp(-distances[i] * beta)
        P[i, i] = 0.0
    
    return (P + P.T) / (2 * n)

# Example usage
X = np.random.randn(100, 50)
P = compute_pairwise_affinities(X)
print("Pairwise affinities shape:", P.shape)
```

Slide 4: Perplexity in t-SNE

Perplexity is a hyperparameter in t-SNE that balances attention between local and global aspects of the data. It can be interpreted as a smooth measure of the effective number of neighbors.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=500, centers=5, n_features=50, random_state=42)

# Function to run t-SNE and plot results
def plot_tsne(X, perplexity):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
    plt.title(f't-SNE with perplexity = {perplexity}')
    plt.colorbar()
    plt.show()

# Try different perplexity values
perplexities = [5, 30, 50, 100]
for perp in perplexities:
    plot_tsne(X, perp)
```

Slide 5: t-Distribution vs. Gaussian Distribution

t-SNE uses a t-distribution in the low-dimensional space, which helps alleviate the "crowding problem" and allows for better separation of clusters.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# Generate x values
x = np.linspace(-5, 5, 1000)

# Calculate PDF for Gaussian and t-distributions
gauss_pdf = norm.pdf(x, 0, 1)
t_pdf = t.pdf(x, df=1)  # t-distribution with 1 degree of freedom (Cauchy distribution)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, gauss_pdf, label='Gaussian distribution')
plt.plot(x, t_pdf, label='t-distribution (df=1)')
plt.title('Gaussian vs. t-distribution')
plt.xlabel('x')
plt.ylabel('Probability density')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 6: Optimization in t-SNE

t-SNE uses gradient descent to optimize the low-dimensional embeddings. The gradient is computed efficiently using the Barnes-Hut approximation for larger datasets.

```python
import numpy as np

def tsne_grad(Y, P, alpha=1):
    n = Y.shape[0]
    dim = Y.shape[1]
    
    sum_Y = np.sum(np.square(Y), 1)
    num = 1 / (1 + sum_Y.reshape((-1, 1)) + sum_Y.reshape((1, -1)) - 2 * np.dot(Y, Y.T))
    num[range(n), range(n)] = 0
    Q = num / np.sum(num)
    Q = np.maximum(Q, 1e-12)
    
    L = (P - Q) * num
    grad = 4 * (np.diag(np.sum(L, 0)) - L).dot(Y)
    
    return grad

# Example usage
n_samples = 100
n_components = 2
Y = np.random.randn(n_samples, n_components)
P = np.random.rand(n_samples, n_samples)
P = (P + P.T) / 2
P /= np.sum(P)

grad = tsne_grad(Y, P)
print("Gradient shape:", grad.shape)
```

Slide 7: Implementing t-SNE from Scratch

Here's a simplified implementation of t-SNE to understand its core principles:

```python
import numpy as np

def tsne(X, n_components=2, perplexity=30.0, n_iter=1000, learning_rate=200.0):
    n_samples = X.shape[0]
    
    # Compute pairwise distances and convert to probabilities
    distances = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
    P = np.exp(-distances / (2 * perplexity**2))
    np.fill_diagonal(P, 0)
    P = (P + P.T) / (2 * n_samples)
    P = np.maximum(P, 1e-12)
    
    # Initialize low-dimensional embeddings
    Y = np.random.randn(n_samples, n_components)
    
    for i in range(n_iter):
        # Compute Q-distribution
        sum_Y = np.sum(np.square(Y), 1)
        num = 1 / (1 + sum_Y.reshape((-1, 1)) + sum_Y.reshape((1, -1)) - 2 * np.dot(Y, Y.T))
        np.fill_diagonal(num, 0)
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        
        # Compute gradient
        PQ_diff = P - Q
        grad = 4 * (np.diag(np.sum(PQ_diff, 0)) - PQ_diff).dot(Y) * num.T
        
        # Update Y
        Y -= learning_rate * grad
    
    return Y

# Example usage
X = np.random.randn(100, 50)
Y = tsne(X)
print("Low-dimensional embedding shape:", Y.shape)
```

Slide 8: Visualizing t-SNE Progress

Let's visualize how t-SNE embeddings evolve during the optimization process:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE

# Generate sample data
X, y = make_blobs(n_samples=500, centers=5, n_features=50, random_state=42)

# Create a custom t-SNE object
tsne = TSNE(n_components=2, random_state=42)

# Fit the model and transform the data
X_iter = []
n_iters = [10, 100, 250, 500, 1000]

for n_iter in n_iters:
    tsne.n_iter = n_iter
    X_iter.append(tsne.fit_transform(X))

# Visualize the results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, (X_i, n_iter) in enumerate(zip(X_iter, n_iters)):
    axes[i].scatter(X_i[:, 0], X_i[:, 1], c=y, cmap='viridis')
    axes[i].set_title(f'Iteration {n_iter}')

plt.tight_layout()
plt.show()
```

Slide 9: Handling Large Datasets with t-SNE

For large datasets, we can use the Barnes-Hut approximation implemented in scikit-learn's TSNE class with the 'method' parameter set to 'barnes\_hut':

```python
import numpy as np
import time
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs

# Generate a large dataset
n_samples = 10000
n_features = 100
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=10, random_state=42)

# Standard t-SNE
start_time = time.time()
tsne_standard = TSNE(n_components=2, method='exact', random_state=42)
X_tsne_standard = tsne_standard.fit_transform(X)
standard_time = time.time() - start_time

# Barnes-Hut t-SNE
start_time = time.time()
tsne_bh = TSNE(n_components=2, method='barnes_hut', random_state=42)
X_tsne_bh = tsne_bh.fit_transform(X)
bh_time = time.time() - start_time

print(f"Standard t-SNE time: {standard_time:.2f} seconds")
print(f"Barnes-Hut t-SNE time: {bh_time:.2f} seconds")
print(f"Speed-up factor: {standard_time / bh_time:.2f}x")
```

Slide 10: t-SNE for Text Data

t-SNE can be used to visualize high-dimensional text data, such as word embeddings or document vectors:

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Sample text data
texts = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be, that is the question",
    "I think, therefore I am",
    "Life is like a box of chocolates",
    "May the Force be with you",
    "Elementary, my dear Watson",
    "Houston, we have a problem",
    "E.T. phone home",
    "Here's looking at you, kid"
]

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X.toarray())

# Visualize the result
plt.figure(figsize=(12, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

for i, text in enumerate(texts):
    plt.annotate(text[:20] + "...", (X_tsne[i, 0], X_tsne[i, 1]))

plt.title('t-SNE visualization of text data')
plt.tight_layout()
plt.show()
```

Slide 11: t-SNE for Image Data

t-SNE can be used to visualize high-dimensional image data, such as flattened pixel values:

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize the result
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.colorbar(scatter)
plt.title('t-SNE visualization of MNIST digits')

# Add some digit images to the plot
for i in range(10):
    idx = np.where(y == i)[0][0]
    plt.imshow(digits.images[idx], cmap='gray', interpolation='nearest',
               extent=(X_tsne[idx, 0], X_tsne[idx, 0]+1, X_tsne[idx, 1], X_tsne[idx, 1]+1))

plt.tight_layout()
plt.show()
```

Slide 12: Comparing t-SNE with PCA

Let's compare t-SNE with Principal Component Analysis (PCA), another popular dimensionality reduction technique:

```python
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Generate Swiss Roll dataset
X, color = make_swiss_roll(n_samples=1000, random_state=42)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.scatter(X[:, 0], X[:, 2], c=color, cmap='viridis')
ax1.set_title('Original Swiss Roll')

ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap='viridis')
ax2.set_title('PCA')

ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color, cmap='viridis')
ax3.set_title('t-SNE')

plt.tight_layout()
plt.show()
```

Slide 13: Real-life Example: Gene Expression Analysis

t-SNE is widely used in bioinformatics for visualizing high-dimensional gene expression data. Here's a simplified example:

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Simulate gene expression data
n_samples = 1000
n_genes = 100
n_cell_types = 5

X, y = make_blobs(n_samples=n_samples, n_features=n_genes, centers=n_cell_types, random_state=42)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize the result
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.colorbar(scatter)
plt.title('t-SNE visualization of simulated gene expression data')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()

# Analyze cluster separation
for cell_type in range(n_cell_types):
    cell_type_data = X_tsne[y == cell_type]
    centroid = np.mean(cell_type_data, axis=0)
    print(f"Cell type {cell_type} centroid: {centroid}")
```

Slide 14: Real-life Example: Image Similarity

t-SNE can be used to visualize similarities between images, which is useful in computer vision tasks:

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize the result
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
plt.colorbar(scatter)
plt.title('t-SNE visualization of handwritten digits')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

# Add some digit images to the plot
for i in range(10):
    idx = np.where(y == i)[0][0]
    plt.imshow(digits.images[idx], cmap='binary', interpolation='nearest',
               extent=(X_tsne[idx, 0], X_tsne[idx, 0]+2, X_tsne[idx, 1], X_tsne[idx, 1]+2))

plt.tight_layout()
plt.show()

# Analyze cluster separation
for digit in range(10):
    digit_data = X_tsne[y == digit]
    centroid = np.mean(digit_data, axis=0)
    print(f"Digit {digit} centroid: {centroid}")
```

Slide 15: Additional Resources

For those interested in diving deeper into t-SNE, here are some valuable resources:

1. Original t-SNE paper: "Visualizing Data using t-SNE" by Laurens van der Maaten and Geoffrey Hinton (2008) ArXiv: [https://arxiv.org/abs/1307.1662](https://arxiv.org/abs/1307.1662)
2. "How to Use t-SNE Effectively" by Martin Wattenberg, Fernanda Viégas, and Ian Johnson Distill.pub: [https://distill.pub/2016/misread-tsne/](https://distill.pub/2016/misread-tsne/)
3. "Accelerating t-SNE using Tree-Based Algorithms" by Laurens van der Maaten (2014) ArXiv: [https://arxiv.org/abs/1301.3342](https://arxiv.org/abs/1301.3342)
4. "Visualizing Large-scale and High-dimensional Data" by Jian Tang, Jingzhou Liu, Ming Zhang, and Qiaozhu Mei (2016) ArXiv: [https://arxiv.org/abs/1602.00370](https://arxiv.org/abs/1602.00370)

These resources provide in-depth explanations of t-SNE's algorithm, implementation details, and best practices for its application in various domains.

