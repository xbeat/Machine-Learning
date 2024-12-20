## Why t-SNE Uses t-Distribution Instead of Gaussian

Slide 1: Introduction to t-SNE

t-SNE (t-distributed Stochastic Neighbor Embedding) is a popular dimensionality reduction technique used for visualizing high-dimensional data. It's an improvement over the original SNE algorithm, with the key difference being the use of a t-distribution instead of a Gaussian distribution. This change addresses some limitations of SNE and provides better visualization results.

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Generate sample high-dimensional data
np.random.seed(42)
data = np.random.randn(1000, 50)  # 1000 samples, 50 dimensions

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(data)

# Visualize the result
plt.figure(figsize=(10, 8))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
plt.title('t-SNE Visualization of High-Dimensional Data')
plt.show()
```

Slide 2: Understanding SNE (Stochastic Neighbor Embedding)

SNE, the predecessor to t-SNE, uses Gaussian distributions to model the similarity between points in both high-dimensional and low-dimensional spaces. It aims to preserve the neighborhood structure of the data when reducing dimensionality.

```python
    return np.exp(-np.sum((x - y)**2) / (2 * sigma**2))

# Example: Calculate Gaussian similarity between two points
point1 = np.array([1, 2, 3])
point2 = np.array([2, 3, 4])
similarity = gaussian_similarity(point1, point2)
print(f"Gaussian similarity: {similarity}")
```

Slide 3: Limitations of SNE

SNE faces a problem known as the "crowding problem." In high-dimensional spaces, the volume of a sphere increases exponentially with its radius, which can lead to most points being equidistant. When projected to lower dimensions, this can result in points crowding in the center of the visualization.

```python

# Generate high-dimensional data
n_points = 1000
n_dims = [2, 10, 50, 100]

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i, dim in enumerate(n_dims):
    data = np.random.randn(n_points, dim)
    distances = np.linalg.norm(data[:100] - data[0], axis=1)
    
    sns.histplot(distances, kde=True, ax=axes[i])
    axes[i].set_title(f'{dim} dimensions')
    axes[i].set_xlabel('Distance from first point')

plt.tight_layout()
plt.show()
```

Slide 4: Introduction to t-Distribution

The t-distribution, also known as Student's t-distribution, is a probability distribution that arises when estimating the mean of a normally distributed population in situations where the sample size is small and the population standard deviation is unknown.

```python

# Generate t-distribution
df = 1  # Degrees of freedom
x = np.linspace(-10, 10, 1000)
y = stats.t.pdf(x, df)

# Plot t-distribution
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=f't-distribution (df={df})')
plt.plot(x, stats.norm.pdf(x), label='Normal distribution')
plt.title('t-distribution vs Normal distribution')
plt.legend()
plt.show()
```

Slide 5: Why t-SNE Uses t-Distribution

t-SNE replaces the Gaussian distribution in the low-dimensional space with a t-distribution. The t-distribution has heavier tails compared to the Gaussian, which helps alleviate the crowding problem. It allows moderately distant points in the high-dimensional space to be modeled by larger distances in the low-dimensional space.

```python
    return (1 + np.sum((x - y)**2) / df) ** (-(df + 1) / 2)

# Compare Gaussian and t-distribution similarities
distances = np.linspace(0, 5, 100)
gaussian_sim = [gaussian_similarity(np.array([0]), np.array([d])) for d in distances]
t_sim = [t_similarity(np.array([0]), np.array([d])) for d in distances]

plt.figure(figsize=(10, 6))
plt.plot(distances, gaussian_sim, label='Gaussian')
plt.plot(distances, t_sim, label='t-distribution')
plt.title('Similarity vs Distance: Gaussian vs t-distribution')
plt.legend()
plt.xlabel('Distance')
plt.ylabel('Similarity')
plt.show()
```

Slide 6: Mathematical Formulation of t-SNE

t-SNE defines the similarity of datapoint $x\_j$ to $x\_i$ using a Gaussian distribution in the high-dimensional space:

$p\_{j|i} = \\frac{\\exp(-||x\_i - x\_j||^2 / 2\\sigma\_i^2)}{\\sum\_{k \\neq i} \\exp(-||x\_i - x\_k||^2 / 2\\sigma\_i^2)}$

In the low-dimensional space, it uses a t-distribution with one degree of freedom:

$q\_{ij} = \\frac{(1 + ||y\_i - y\_j||^2)^{-1}}{\\sum\_{k \\neq l} (1 + ||y\_k - y\_l||^2)^{-1}}$

```python
    diff = X[i] - X[j]
    return np.exp(-np.dot(diff, diff) / (2 * sigma**2))

def low_dim_similarity(Y, i, j):
    diff = Y[i] - Y[j]
    return 1 / (1 + np.dot(diff, diff))

# Example usage
X = np.random.randn(100, 50)  # High-dimensional data
Y = np.random.randn(100, 2)   # Low-dimensional embedding
i, j = 0, 1
sigma = 1.0

p_ij = high_dim_similarity(X, i, j, sigma)
q_ij = low_dim_similarity(Y, i, j)

print(f"High-dimensional similarity: {p_ij}")
print(f"Low-dimensional similarity: {q_ij}")
```

Slide 7: Advantages of t-Distribution in t-SNE

The t-distribution's heavier tails allow for a more faithful representation of distances between moderately distant points in the high-dimensional space. This helps to separate clusters more effectively and reduces the tendency of points to crowd in the center of the visualization.

```python

# Generate sample data with clusters
np.random.seed(42)
n_samples = 300
X = np.concatenate([
    np.random.randn(n_samples, 50) + np.array([2] * 50),
    np.random.randn(n_samples, 50) + np.array([-2] * 50),
    np.random.randn(n_samples, 50)
])

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(X)

# Visualize the result
plt.figure(figsize=(10, 8))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=np.repeat(['A', 'B', 'C'], n_samples))
plt.title('t-SNE Visualization of Clustered Data')
plt.show()
```

Slide 8: Gradient Computation in t-SNE

The gradient of the Kullback-Leibler divergence between P and Q distributions drives the optimization process in t-SNE. The use of t-distribution simplifies this gradient computation:

$\\frac{\\partial C}{\\partial y\_i} = 4 \\sum\_j (p\_{ij} - q\_{ij})(y\_i - y\_j)(1 + ||y\_i - y\_j||^2)^{-1}$

```python
    n = Y.shape[0]
    dY = np.zeros_like(Y)
    
    for i in range(n):
        diff = Y[i] - Y
        dist_sq = np.sum(diff**2, axis=1)
        q = (1 + dist_sq)**-1
        q[i] = 0
        
        dY[i] = 4 * np.sum((P[i] - Q[i])[:, np.newaxis] * diff * q[:, np.newaxis], axis=0)
    
    return dY

# Example usage (simplified)
n, d = 100, 2
Y = np.random.randn(n, d)
P = np.random.rand(n, n)
Q = np.random.rand(n, n)

gradient = tsne_gradient(Y, P, Q)
print("Gradient shape:", gradient.shape)
```

Slide 9: Perplexity in t-SNE

Perplexity is a hyperparameter in t-SNE that balances attention between local and global aspects of the data. It's related to the number of nearest neighbors each point effectively considers. The perplexity value typically ranges from 5 to 50.

```python

def compute_perplexity(distances, sigmas):
    P = np.exp(-distances / (2 * sigmas**2))
    sumP = np.sum(P, axis=1)
    H = np.log2(sumP) + np.sum(P * np.log2(P), axis=1) / sumP
    return 2**H

# Generate sample data
X = np.random.randn(500, 50)

# Compute distances
nbrs = NearestNeighbors(n_neighbors=50, metric='euclidean').fit(X)
distances, _ = nbrs.kneighbors(X)

# Compute perplexity for different sigma values
sigmas = np.logspace(-1, 1, 20)
perplexities = [np.mean(compute_perplexity(distances, sigma)) for sigma in sigmas]

plt.figure(figsize=(10, 6))
plt.semilogx(sigmas, perplexities)
plt.title('Average Perplexity vs Sigma')
plt.xlabel('Sigma')
plt.ylabel('Perplexity')
plt.show()
```

Slide 10: Early Exaggeration in t-SNE

Early exaggeration is a technique used in t-SNE to create better global structure. It involves multiplying the early iterations' high-dimensional probabilities by a factor (typically 4-12) to encourage the formation of widely separated clusters.

```python
    P_exaggerated = P.copy()
    for i in range(n_iter):
        P_exaggerated *= exaggeration_factor
        # Perform t-SNE iteration here
        # ...
    return P_exaggerated

# Example usage
P = np.random.rand(100, 100)
P_exaggerated = early_exaggeration(P)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(P, cmap='viridis')
plt.title('Original P')
plt.subplot(122)
plt.imshow(P_exaggerated, cmap='viridis')
plt.title('Exaggerated P')
plt.tight_layout()
plt.show()
```

Slide 11: Real-life Example: Handwritten Digit Recognition

t-SNE is often used for visualizing high-dimensional datasets, such as images. Let's apply t-SNE to the MNIST dataset of handwritten digits.

```python
from sklearn.manifold import TSNE
import seaborn as sns

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize the result
plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='deep')
plt.title('t-SNE Visualization of MNIST Digits')
plt.legend(title='Digit')
plt.show()
```

Slide 12: Real-life Example: Gene Expression Analysis

t-SNE is widely used in bioinformatics for visualizing gene expression data. Here's a simplified example using synthetic gene expression data.

```python

# Generate synthetic gene expression data
n_samples = 1000
n_genes = 50
n_conditions = 3

data = np.random.randn(n_samples, n_genes)
conditions = np.random.choice(['Control', 'Treatment A', 'Treatment B'], n_samples)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(data)

# Visualize the result
plt.figure(figsize=(12, 8))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=conditions, palette='deep')
plt.title('t-SNE Visualization of Gene Expression Data')
plt.legend(title='Condition')
plt.show()
```

Slide 13: Limitations and Considerations of t-SNE

While t-SNE is powerful, it has limitations. It can be computationally expensive for large datasets, may produce different results on multiple runs due to its stochastic nature, and can sometimes create misleading visualizations if not used carefully.

```python

def compare_tsne_runtime(n_samples_list, n_features=50):
    runtimes = []
    for n_samples in n_samples_list:
        data = np.random.randn(n_samples, n_features)
        start_time = time.time()
        TSNE(n_components=2).fit_transform(data)
        end_time = time.time()
        runtimes.append(end_time - start_time)
    return runtimes

n_samples_list = [100, 500, 1000, 5000]
runtimes = compare_tsne_runtime(n_samples_list)

plt.figure(figsize=(10, 6))
plt.plot(n_samples_list, runtimes, marker='o')
plt.title('t-SNE Runtime vs Dataset Size')
plt.xlabel('Number of Samples')
plt.ylabel('Runtime (seconds)')
plt.show()
```

Slide 14: Conclusion and Best Practices

t-SNE's use of the t-distribution instead of a Gaussian distribution in the low-dimensional space addresses the crowding problem and provides better visualizations of high-dimensional data. When using t-SNE, consider experimenting with different perplexity values, running multiple times to ensure stability, and being cautious about interpreting distances between well-separated clusters.

```python
    results = {}
    for perplexity in perplexities:
        results[perplexity] = []
        for _ in range(n_runs):
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=None)
            result = tsne.fit_transform(X)
            results[perplexity].append(result)
    return results

# Example usage
X = np.random.randn(500, 50)
best_practices_results = tsne_best_practices(X)

# Visualize results for different perplexities
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, (perplexity, runs) in enumerate(best_practices_results.items()):
    for run in runs:
        axes[idx].scatter(run[:, 0], run[:, 1], alpha=0.5)
    axes[idx].set_title(f'Perplexity: {perplexity}')
plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into t-SNE and its applications, here are some valuable resources:

1. Original t-SNE paper: "Visualizing Data using t-SNE" by Laurens van der Maaten and Geoffrey Hinton (2008) ArXiv URL: [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
2. "How to Use t-SNE Effectively" by Martin Wattenberg, Fernanda Viégas, and Ian Johnson Available at: [https://distill.pub/2016/misread-tsne/](https://distill.pub/2016/misread-tsne/)
3. "Accelerating t-SNE using Tree-Based Algorithms" by Laurens van der Maaten (2014) ArXiv URL: [https://arxiv.org/abs/1301.3342](https://arxiv.org/abs/1301.3342)

These resources provide in-depth explanations of t-SNE's theory, implementation details, and best practices for its effective use in various data analysis scenarios.


