## Uniform Manifold Approximation and Projection (UMAP)
##interrupted Slide 15

Slide 1: Introduction to UMAP

Uniform Manifold Approximation and Projection (UMAP) is a dimensionality reduction technique used for visualizing high-dimensional data in a lower-dimensional space. It is particularly useful for exploring and understanding complex datasets in machine learning and data analysis.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random high-dimensional data
data = np.random.rand(1000, 50)

# Create UMAP object and fit the data
reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(data)

# Plot the results
plt.scatter(embedding[:, 0], embedding[:, 1], s=5)
plt.title("UMAP Projection of Random 50D Data")
plt.show()
```

Slide 2: UMAP Algorithm Overview

UMAP works by constructing a high-dimensional graph representation of the data and then finding a low-dimensional embedding that preserves the graph structure. It balances local and global structure preservation, resulting in meaningful visualizations.

```python
from sklearn.datasets import load_digits
from umap import UMAP

# Load the digits dataset
digits = load_digits()

# Create and fit UMAP model
umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = umap_model.fit_transform(digits.data)

print(f"Original shape: {digits.data.shape}")
print(f"Embedded shape: {embedding.shape}")
```

Slide 3: UMAP Parameters

Key parameters in UMAP include n\_neighbors, min\_dist, and n\_components. These parameters control the balance between preserving local and global structure, the compactness of the embedding, and the dimensionality of the output.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
data = np.random.rand(1000, 20)

# Define different parameter sets
params = [
    {"n_neighbors": 5, "min_dist": 0.1},
    {"n_neighbors": 15, "min_dist": 0.5},
    {"n_neighbors": 50, "min_dist": 0.1}
]

# Plot UMAP embeddings with different parameters
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, param in enumerate(params):
    reducer = umap.UMAP(**param, n_components=2)
    embedding = reducer.fit_transform(data)
    axs[i].scatter(embedding[:, 0], embedding[:, 1], s=5)
    axs[i].set_title(f"n_neighbors={param['n_neighbors']}, min_dist={param['min_dist']}")

plt.tight_layout()
plt.show()
```

Slide 4: Preparing Data for UMAP

Before applying UMAP, it's crucial to preprocess the data. This often involves scaling, handling missing values, and encoding categorical variables.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample data
data = pd.DataFrame({
    'numeric1': [1, 2, np.nan, 4],
    'numeric2': [5, 6, 7, 8],
    'categorical': ['A', 'B', 'A', 'C']
})

# Define preprocessing steps
numeric_features = ['numeric1', 'numeric2']
categorical_features = ['categorical']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', pd.get_dummies)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit and transform the data
processed_data = preprocessor.fit_transform(data)
print(processed_data)
```

Slide 5: UMAP for Dimensionality Reduction

UMAP is often used to reduce high-dimensional data to a lower-dimensional representation, typically 2D or 3D for visualization purposes.

```python
import umap
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()

# Create and fit UMAP model
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(digits.data)

# Plot the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
plt.colorbar(scatter)
plt.title('UMAP projection of the Digits dataset')
plt.show()
```

Slide 6: UMAP vs. t-SNE

UMAP often provides similar visualization quality to t-SNE but with faster computation times and better preservation of global structure.

```python
from sklearn.datasets import load_digits
import umap
import time
import matplotlib.pyplot as plt

# Load data
digits = load_digits()

# UMAP
start_time = time.time()
umap_embedding = umap.UMAP().fit_transform(digits.data)
umap_time = time.time() - start_time

# t-SNE
start_time = time.time()
tsne_embedding = TSNE().fit_transform(digits.data)
tsne_time = time.time() - start_time

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
ax1.set_title(f'UMAP (Time: {umap_time:.2f}s)')

ax2.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
ax2.set_title(f't-SNE (Time: {tsne_time:.2f}s)')

plt.show()
```

Slide 7: UMAP for Clustering

UMAP can be used as a preprocessing step for clustering algorithms, potentially improving their performance on high-dimensional data.

```python
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt

# Load data
iris = load_iris()

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(iris.data)

# Perform clustering on the embedding
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embedding)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='viridis', s=50)
plt.colorbar(scatter)
plt.title('UMAP + K-means clustering of Iris dataset')
plt.show()
```

Slide 8: UMAP for Anomaly Detection

UMAP can help visualize anomalies in high-dimensional data by projecting them into a lower-dimensional space where they may be more easily identified.

```python
from sklearn.datasets import make_blobs
import umap
import matplotlib.pyplot as plt

# Generate normal data
X, _ = make_blobs(n_samples=1000, centers=3, n_features=10, random_state=42)

# Generate anomalies
anomalies = np.random.uniform(low=-10, high=10, size=(50, 10))

# Combine normal data and anomalies
X_with_anomalies = np.vstack([X, anomalies])

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(X_with_anomalies)

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:-50, 0], embedding[:-50, 1], c='blue', s=5, label='Normal')
plt.scatter(embedding[-50:, 0], embedding[-50:, 1], c='red', s=20, label='Anomaly')
plt.legend()
plt.title('UMAP projection for Anomaly Detection')
plt.show()
```

Slide 9: Supervised UMAP

UMAP can incorporate label information to create more informative embeddings for supervised learning tasks.

```python
import umap
import matplotlib.pyplot as plt

# Load data
iris = load_iris()

# Apply supervised UMAP
supervised_reducer = umap.UMAP(n_components=2, random_state=42, target_metric='l2')
supervised_embedding = supervised_reducer.fit_transform(iris.data, y=iris.target)

# Apply unsupervised UMAP
unsupervised_reducer = umap.UMAP(n_components=2, random_state=42)
unsupervised_embedding = unsupervised_reducer.fit_transform(iris.data)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.scatter(supervised_embedding[:, 0], supervised_embedding[:, 1], c=iris.target, cmap='Spectral', s=5)
ax1.set_title('Supervised UMAP')

ax2.scatter(unsupervised_embedding[:, 0], unsupervised_embedding[:, 1], c=iris.target, cmap='Spectral', s=5)
ax2.set_title('Unsupervised UMAP')

plt.show()
```

Slide 10: UMAP for Feature Selection

UMAP can be used to identify important features by examining the contribution of each feature to the low-dimensional embedding.

```python
import umap
import pandas as pd
import matplotlib.pyplot as plt

# Load data
cancer = load_breast_cancer()

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(cancer.data)

# Get feature importances
feature_importances = pd.Series(reducer.feature_importances_, index=cancer.feature_names)

# Plot top 10 important features
plt.figure(figsize=(12, 6))
feature_importances.nlargest(10).plot(kind='bar')
plt.title('Top 10 Important Features in UMAP Embedding')
plt.tight_layout()
plt.show()
```

Slide 11: UMAP for Text Data

UMAP can be applied to text data after converting text to numerical representations, such as TF-IDF vectors.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import matplotlib.pyplot as plt

# Load data
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(newsgroups.data)

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(tfidf_matrix)

# Plot results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=newsgroups.target, cmap='Spectral', s=5)
plt.colorbar(scatter)
plt.title('UMAP projection of 20 Newsgroups dataset')
plt.show()
```

Slide 12: UMAP for Image Data

UMAP can be applied to image data to visualize similarities and differences between images in a dataset.

```python
import umap
import matplotlib.pyplot as plt

# Load data
digits = load_digits()

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(digits.data)

# Plot results
plt.figure(figsize=(12, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
plt.colorbar()
plt.title('UMAP projection of the Digits dataset')

# Plot some example digits
for i in range(10):
    plt.annotate(str(i), xy=(embedding[digits.target == i, 0].mean(), 
                              embedding[digits.target == i, 1].mean()),
                 xytext=(0, 0), textcoords="offset points",
                 ha='center', va='center',
                 bbox=dict(boxstyle="round", fc="w"),
                 arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.show()
```

Slide 13: Real-Life Example: Genome Sequencing

UMAP can be used in genomics to visualize and analyze high-dimensional genetic data, helping researchers identify patterns and relationships between different genetic profiles.

```python
import umap
import matplotlib.pyplot as plt

# Simulate genetic data (SNPs)
n_samples = 1000
n_snps = 10000
genetic_data = np.random.randint(0, 3, size=(n_samples, n_snps))

# Simulate population labels (e.g., different ethnic groups)
populations = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(genetic_data)

# Plot results
plt.figure(figsize=(12, 10))
for pop in np.unique(populations):
    mask = populations == pop
    plt.scatter(embedding[mask, 0], embedding[mask, 1], label=pop, s=5)

plt.legend()
plt.title('UMAP projection of simulated genetic data')
plt.show()
```

Slide 14: Real-Life Example: Customer Segmentation

UMAP can be used in marketing to segment customers based on their behavior, helping businesses tailor their strategies to different customer groups.

```python
import pandas as pd
import umap
import matplotlib.pyplot as plt

# Simulate customer data
n_customers = 1000
customer_data = pd.DataFrame({
    'age': np.random.normal(40, 15, n_customers),
    'income': np.random.lognormal(10, 1, n_customers),
    'spending': np.random.lognormal(5, 1, n_customers),
    'frequency': np.random.poisson(10, n_customers),
    'loyalty_years': np.random.gamma(2, 2, n_customers)
})

# Normalize the data
normalized_data = (customer_data - customer_data.mean()) / customer_data.std()

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(normalized_data)

# Apply simple clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(embedding)

# Plot results
plt.figure(figsize=(12, 10))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='viridis', s=5)
plt.colorbar(scatter)
plt.title('UMAP projection of customer segments')
plt.show()
```

Slide 15: Additional Resources

For those interested in delving deeper into UMAP, here are some valuable resources:

1. Original UMAP paper: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. ArXiv:1802.03426. URL: [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
2. UMAP documentation: [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)
3. Comparison of Dimensionality Reduction Techniques: Espadoto, M., Martins, R. M., Kerren, A., Hirata, N. S. T., & Telea, A. C. (2019). Toward a Quantitative Survey of Dimension Reduction Techniques. IEEE Transactions on Visualization and Computer Graphics. URL: [https://arxiv.org/abs/1904.08566](https://arxiv.org/abs/1904.08566)

These resources provide a comprehensive understanding of UMAP's theoretical foundations, practical applications, and comparisons with other dimensionality reduction techniques.


