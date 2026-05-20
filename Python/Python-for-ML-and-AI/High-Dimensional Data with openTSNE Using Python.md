## High-Dimensional Data with openTSNE Using Python
Slide 1: Introduction to openTSNE

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a popular dimensionality reduction technique for visualizing high-dimensional data. openTSNE is a Python library that provides a fast and extensible implementation of t-SNE. Let's start by importing the necessary libraries and creating a simple dataset.

```python
import numpy as np
from openTSNE import TSNE
from sklearn.datasets import make_blobs

# Create a simple dataset
X, y = make_blobs(n_samples=1000, n_features=50, centers=5, random_state=42)
```

Slide 2: Installing openTSNE

Before we dive into using openTSNE, let's make sure it's installed. You can install openTSNE using pip, the Python package manager. Here's how to install it and verify the installation:

```python
# Install openTSNE
!pip install openTSNE

# Verify installation
import openTSNE
print(f"openTSNE version: {openTSNE.__version__}")
```

Slide 3: Creating a TSNE Object

To use openTSNE, we first need to create a TSNE object. This object allows us to set various parameters that control the t-SNE algorithm. Let's create a basic TSNE object with default parameters:

```python
# Create a TSNE object
tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
)
```

Slide 4: Fitting the TSNE Model

Once we have created our TSNE object, we can fit it to our data. This step performs the actual dimensionality reduction. Here's how to fit the model and transform the data:

```python
# Fit the TSNE model and transform the data
embeddings = tsne.fit(X)

print(f"Original shape: {X.shape}")
print(f"Embedded shape: {embeddings.shape}")
```

Slide 5: Visualizing the Results

After fitting the model, we can visualize the results. Let's use matplotlib to create a scatter plot of the embedded data:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y, cmap="viridis")
plt.colorbar()
plt.title("t-SNE Visualization of Blob Dataset")
plt.show()
```

Slide 6: Controlling Perplexity

Perplexity is a key parameter in t-SNE that balances local and global aspects of the data. Let's explore how different perplexity values affect the embedding:

```python
perplexities = [5, 30, 50, 100]
fig, axes = plt.subplots(2, 2, figsize=(15, 15))

for ax, perplexity in zip(axes.flat, perplexities):
    tsne = TSNE(perplexity=perplexity, random_state=42)
    embeddings = tsne.fit(X)
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=y, cmap="viridis")
    ax.set_title(f"Perplexity: {perplexity}")

plt.tight_layout()
plt.show()
```

Slide 7: Using Different Metrics

openTSNE supports various distance metrics. Let's compare the results using Euclidean and Cosine distances:

```python
metrics = ["euclidean", "cosine"]
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

for ax, metric in zip(axes, metrics):
    tsne = TSNE(metric=metric, random_state=42)
    embeddings = tsne.fit(X)
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=y, cmap="viridis")
    ax.set_title(f"Metric: {metric}")

plt.tight_layout()
plt.show()
```

Slide 8: Controlling the Learning Rate

The learning rate affects how quickly the t-SNE algorithm converges. Let's experiment with different learning rates:

```python
learning_rates = [10, 200, 1000]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, learning_rate in zip(axes, learning_rates):
    tsne = TSNE(learning_rate=learning_rate, random_state=42)
    embeddings = tsne.fit(X)
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=y, cmap="viridis")
    ax.set_title(f"Learning Rate: {learning_rate}")

plt.tight_layout()
plt.show()
```

Slide 9: Early Exaggeration

Early exaggeration is a technique used to form tight clusters in the early stages of optimization. Let's see how different exaggeration factors affect the result:

```python
exaggeration_factors = [4, 12, 20]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, factor in zip(axes, exaggeration_factors):
    tsne = TSNE(early_exaggeration_iter=250, early_exaggeration=factor, random_state=42)
    embeddings = tsne.fit(X)
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=y, cmap="viridis")
    ax.set_title(f"Exaggeration Factor: {factor}")

plt.tight_layout()
plt.show()
```

Slide 10: Initialization Methods

openTSNE offers different initialization methods. Let's compare PCA and random initialization:

```python
init_methods = ["pca", "random"]
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

for ax, init in zip(axes, init_methods):
    tsne = TSNE(initialization=init, random_state=42)
    embeddings = tsne.fit(X)
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=y, cmap="viridis")
    ax.set_title(f"Initialization: {init}")

plt.tight_layout()
plt.show()
```

Slide 11: Handling Large Datasets

For large datasets, openTSNE provides an efficient implementation using approximate nearest neighbors. Let's simulate a larger dataset and use this feature:

```python
# Create a larger dataset
X_large, y_large = make_blobs(n_samples=10000, n_features=50, centers=10, random_state=42)

# Use approximate nearest neighbors
tsne = TSNE(n_jobs=8, method='approximated', n_components=2, random_state=42)
embeddings_large = tsne.fit(X_large)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_large[:, 0], embeddings_large[:, 1], c=y_large, cmap="viridis", alpha=0.5)
plt.colorbar()
plt.title("t-SNE Visualization of Large Dataset")
plt.show()
```

Slide 12: Incremental Learning

openTSNE supports incremental learning, allowing you to add new points to an existing embedding. This is useful for updating visualizations with new data:

```python
# Initial embedding
tsne = TSNE(n_jobs=8, random_state=42)
initial_embedding = tsne.fit(X[:800])

# Add new points
new_embedding = tsne.transform(X[800:])

plt.figure(figsize=(10, 8))
plt.scatter(initial_embedding[:, 0], initial_embedding[:, 1], c=y[:800], cmap="viridis", label="Initial")
plt.scatter(new_embedding[:, 0], new_embedding[:, 1], c=y[800:], cmap="viridis", marker="x", s=100, label="New")
plt.legend()
plt.title("Incremental t-SNE")
plt.show()
```

Slide 13: Customizing the Optimization

openTSNE allows fine-tuning of the optimization process. Let's create a custom optimization schedule:

```python
from openTSNE.callbacks import ErrorLogger

# Create a custom optimization schedule
optimization = [
    (50, 12, 0.5),   # (iterations, exaggeration, learning_rate)
    (100, 4, 0.25),
    (250, 1, 0.1),
]

tsne = TSNE(n_jobs=8, random_state=42)
embeddings = tsne.fit(
    X,
    optimization=optimization,
    callbacks=ErrorLogger(),
)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y, cmap="viridis")
plt.title("t-SNE with Custom Optimization")
plt.show()
```

Slide 14: Additional Resources

For more information on t-SNE and openTSNE, consider exploring these resources:

1. Original t-SNE paper: "Visualizing Data using t-SNE" by Laurens van der Maaten and Geoffrey Hinton (2008) ArXiv link: [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
2. openTSNE GitHub repository: [https://github.com/pavlin-policar/openTSNE](https://github.com/pavlin-policar/openTSNE)
3. "How to Use t-SNE Effectively" by Martin Wattenberg, Fernanda Viégas, and Ian Johnson ArXiv link: [https://arxiv.org/abs/1610.02831](https://arxiv.org/abs/1610.02831)

