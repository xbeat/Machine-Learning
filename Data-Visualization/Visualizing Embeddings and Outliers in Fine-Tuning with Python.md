## Visualizing Embeddings and Outliers in Fine-Tuning with Python
Slide 1: Introduction to Embeddings and Outliers in Fine-Tuning

Embeddings are dense vector representations of data points in a high-dimensional space. During the fine-tuning process, visualizing these embeddings can provide valuable insights into the model's understanding of the data. Outliers, on the other hand, are data points that significantly differ from other observations. Identifying and visualizing outliers is crucial for improving model performance and understanding data distribution.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Generate sample embeddings
embeddings = np.random.rand(100, 50)

# Reduce dimensionality for visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot the embeddings
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
plt.title("2D Visualization of Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
```

Slide 2: t-SNE for Embedding Visualization

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a popular technique for visualizing high-dimensional data in 2D or 3D space. It preserves local relationships between data points, making it ideal for visualizing embeddings. This technique helps in identifying clusters and patterns in the data.

```python
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# Generate sample embeddings with labels
num_samples = 1000
embeddings = np.random.randn(num_samples, 100)
labels = np.random.randint(0, 5, num_samples)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot the results
plt.figure(figsize=(12, 10))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
plt.colorbar(scatter)
plt.title("t-SNE Visualization of Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
```

Slide 3: UMAP for Embedding Visualization

UMAP (Uniform Manifold Approximation and Projection) is another dimensionality reduction technique that can be used to visualize embeddings. It often preserves both local and global structure better than t-SNE and is computationally more efficient for larger datasets.

```python
import umap
import numpy as np
import matplotlib.pyplot as plt

# Generate sample embeddings
num_samples = 1000
embeddings = np.random.randn(num_samples, 100)
labels = np.random.randint(0, 5, num_samples)

# Apply UMAP
reducer = umap.UMAP(random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

# Plot the results
plt.figure(figsize=(12, 10))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
plt.colorbar(scatter)
plt.title("UMAP Visualization of Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
```

Slide 4: Comparing t-SNE and UMAP

Both t-SNE and UMAP have their strengths and weaknesses. t-SNE is good at preserving local structure but may struggle with global structure. UMAP aims to preserve both local and global structure. Let's compare them side by side.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

# Generate sample data
num_samples = 1000
embeddings = np.random.randn(num_samples, 100)
labels = np.random.randint(0, 5, num_samples)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(embeddings)

# Apply UMAP
umap_reducer = umap.UMAP(random_state=42)
umap_result = umap_reducer.fit_transform(embeddings)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
ax1.set_title("t-SNE")
ax1.set_xlabel("Dimension 1")
ax1.set_ylabel("Dimension 2")

ax2.scatter(umap_result[:, 0], umap_result[:, 1], c=labels, cmap='viridis')
ax2.set_title("UMAP")
ax2.set_xlabel("Dimension 1")
ax2.set_ylabel("Dimension 2")

plt.tight_layout()
plt.show()
```

Slide 5: Identifying Outliers using Isolation Forest

Isolation Forest is an unsupervised learning algorithm for detecting outliers. It works by isolating anomalies in the data rather than profiling normal points. This makes it particularly effective for high-dimensional data like embeddings.

```python
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt

# Generate sample embeddings
num_samples = 1000
embeddings = np.random.randn(num_samples, 100)

# Add some outliers
outliers = np.random.uniform(low=-4, high=4, size=(10, 100))
embeddings = np.vstack([embeddings, outliers])

# Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.01, random_state=42)
outlier_labels = iso_forest.fit_predict(embeddings)

# Reduce dimensionality for visualization
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot results
plt.figure(figsize=(12, 10))
plt.scatter(embeddings_2d[outlier_labels == 1, 0], embeddings_2d[outlier_labels == 1, 1], c='blue', label='Normal')
plt.scatter(embeddings_2d[outlier_labels == -1, 0], embeddings_2d[outlier_labels == -1, 1], c='red', label='Outlier')
plt.title("Outlier Detection using Isolation Forest")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.show()
```

Slide 6: Local Outlier Factor (LOF) for Outlier Detection

Local Outlier Factor is another popular method for detecting outliers. It measures the local deviation of density of a given sample with respect to its neighbors. Points that have a substantially lower density than their neighbors are considered outliers.

```python
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import matplotlib.pyplot as plt

# Generate sample embeddings
num_samples = 1000
embeddings = np.random.randn(num_samples, 100)

# Add some outliers
outliers = np.random.uniform(low=-4, high=4, size=(10, 100))
embeddings = np.vstack([embeddings, outliers])

# Fit Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
outlier_labels = lof.fit_predict(embeddings)

# Reduce dimensionality for visualization
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot results
plt.figure(figsize=(12, 10))
plt.scatter(embeddings_2d[outlier_labels == 1, 0], embeddings_2d[outlier_labels == 1, 1], c='blue', label='Normal')
plt.scatter(embeddings_2d[outlier_labels == -1, 0], embeddings_2d[outlier_labels == -1, 1], c='red', label='Outlier')
plt.title("Outlier Detection using Local Outlier Factor")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.show()
```

Slide 7: Visualizing Embedding Changes During Fine-Tuning

Tracking how embeddings change during the fine-tuning process can provide insights into the model's learning progress. We can visualize this by plotting embeddings at different stages of training.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Simulate embeddings at different training stages
num_samples = 500
num_stages = 3
embeddings = [np.random.randn(num_samples, 100) for _ in range(num_stages)]

# Apply t-SNE to each stage
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = [tsne.fit_transform(emb) for emb in embeddings]

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
stages = ['Initial', 'Mid-training', 'Final']

for i, (emb_2d, ax) in enumerate(zip(embeddings_2d, axes)):
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=np.arange(num_samples), cmap='viridis')
    ax.set_title(f"{stages[i]} Embeddings")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()
```

Slide 8: Visualizing Embedding Clusters

Clustering embeddings can reveal groups of similar data points. We can use algorithms like K-means to cluster the embeddings and visualize the results.

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Generate sample embeddings
num_samples = 1000
embeddings = np.random.randn(num_samples, 100)

# Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot results
plt.figure(figsize=(12, 10))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='viridis')
plt.colorbar(scatter)
plt.title("Embedding Clusters Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
```

Slide 9: Visualizing Embedding Quality

We can assess the quality of embeddings by visualizing how well they separate different classes or categories. This can be done by color-coding the embeddings based on their true labels and observing the separation in the visualization.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Generate sample embeddings with labels
num_samples = 1000
num_classes = 5
embeddings = np.random.randn(num_samples, 100)
labels = np.random.randint(0, num_classes, num_samples)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot results
plt.figure(figsize=(12, 10))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
plt.colorbar(scatter)
plt.title("Embedding Quality Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
```

Slide 10: Interactive Embedding Visualization

Interactive visualizations can provide a more engaging way to explore embeddings. We can use libraries like Plotly to create interactive scatter plots of embeddings.

```python
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE

# Generate sample embeddings with labels
num_samples = 1000
num_classes = 5
embeddings = np.random.randn(num_samples, 100)
labels = np.random.randint(0, num_classes, num_samples)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Create interactive plot
fig = go.Figure(data=go.Scatter(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    mode='markers',
    marker=dict(
        size=8,
        color=labels,
        colorscale='Viridis',
        showscale=True
    ),
    text=[f'Sample {i}, Class {label}' for i, label in enumerate(labels)],
    hoverinfo='text'
))

fig.update_layout(
    title="Interactive Embedding Visualization",
    xaxis_title="Dimension 1",
    yaxis_title="Dimension 2",
)

fig.show()
```

Slide 11: Visualizing Embedding Similarity

We can visualize the similarity between embeddings using a heatmap. This can help identify patterns and relationships between different data points or classes.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Generate sample embeddings with labels
num_samples = 50
num_classes = 5
embeddings = np.random.randn(num_samples, 100)
labels = np.random.randint(0, num_classes, num_samples)

# Compute similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Plot heatmap
plt.figure(figsize=(12, 10))
heatmap = plt.imshow(similarity_matrix, cmap='viridis')
plt.colorbar(heatmap)
plt.title("Embedding Similarity Heatmap")
plt.xlabel("Sample Index")
plt.ylabel("Sample Index")

# Add class separators
for i in range(1, num_classes):
    plt.axhline(y=(labels == i).sum() * i - 0.5, color='red', linestyle='--')
    plt.axvline(x=(labels == i).sum() * i - 0.5, color='red', linestyle='--')

plt.show()
```

Slide 12: Real-Life Example: Document Embeddings

Document embeddings are used in various natural language processing tasks. Let's visualize embeddings of news articles from different categories.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# Sample news articles (simplified for illustration)
articles = [
    "The new smartphone features a high-resolution camera.",
    "Scientists discover a new species of deep-sea fish.",
    "Local team wins the championship in overtime.",
    "Researchers develop a more efficient solar panel.",
    "Upcoming election sparks debate on key issues.",
    "New restaurant opens featuring fusion cuisine.",
    "Space agency announces plans for Mars mission.",
    "Health experts recommend new guidelines for exercise.",
    "Tech company unveils latest virtual reality headset."
]

categories = ['Technology', 'Science', 'Sports', 'Science', 
              'Politics', 'Food', 'Science', 'Health', 'Technology']

# Create TF-IDF embeddings
vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(articles)

# Reduce dimensionality for visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings.toarray())

# Plot results
plt.figure(figsize=(12, 10))
for i, category in enumerate(set(categories)):
    mask = np.array(categories) == category
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=category)

plt.title("Document Embeddings Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.show()
```

Slide 13: Real-Life Example: Image Embeddings

Image embeddings are crucial in computer vision tasks. Let's visualize embeddings of images from different categories using a pre-trained model.

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.manifold import TSNE

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to load and preprocess image
def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Load sample images (replace with actual image paths)
image_paths = [
    'cat1.jpg', 'cat2.jpg', 'dog1.jpg', 'dog2.jpg',
    'car1.jpg', 'car2.jpg', 'house1.jpg', 'house2.jpg'
]
labels = ['Cat', 'Cat', 'Dog', 'Dog', 'Car', 'Car', 'House', 'House']

# Generate embeddings
embeddings = []
for img_path in image_paths:
    img = load_image(img_path)
    embedding = model.predict(img)
    embeddings.append(embedding.flatten())

embeddings = np.array(embeddings)

# Reduce dimensionality for visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot results
plt.figure(figsize=(12, 10))
for label in set(labels):
    mask = np.array(labels) == label
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=label)

plt.title("Image Embeddings Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.show()
```

Slide 14: Visualizing Embedding Evolution

Tracking how embeddings evolve during fine-tuning can provide insights into the learning process. Let's simulate this evolution and visualize it.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Simulate embedding evolution
num_samples = 100
num_epochs = 5
embedding_dim = 50

# Initial random embeddings
initial_embeddings = np.random.randn(num_samples, embedding_dim)

# Simulate evolution
evolved_embeddings = [initial_embeddings]
for _ in range(num_epochs):
    new_embedding = evolved_embeddings[-1] + np.random.normal(0, 0.1, (num_samples, embedding_dim))
    evolved_embeddings.append(new_embedding)

# Reduce dimensionality for visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = [tsne.fit_transform(emb) for emb in evolved_embeddings]

# Plot evolution
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, emb_2d in enumerate(embeddings_2d):
    axes[i].scatter(emb_2d[:, 0], emb_2d[:, 1], c=np.arange(num_samples), cmap='viridis')
    axes[i].set_title(f"Epoch {i}")
    axes[i].set_xlabel("Dimension 1")
    axes[i].set_ylabel("Dimension 2")

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into the topic of embeddings and their visualizations, here are some valuable resources:

1. "Visualizing Data using t-SNE" by Laurens van der Maaten and Geoffrey Hinton ArXiv link: [https://arxiv.org/abs/1301.3342](https://arxiv.org/abs/1301.3342)
2. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction" by Leland McInnes, John Healy, and James Melville ArXiv link: [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
3. "Efficient Estimation of Word Representations in Vector Space" by Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean ArXiv link: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)

These papers provide in-depth explanations of various embedding techniques and visualization methods, offering a solid foundation for further exploration in this field.

