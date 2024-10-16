## Clustering Algorithms for Unlabeled Data Analysis in Python
Slide 1: Introduction to Clustering Algorithms

Clustering algorithms are unsupervised machine learning techniques used to group similar data points together without prior labeling. These algorithms find patterns and structures in unlabeled data, making them valuable for exploratory data analysis, customer segmentation, and anomaly detection.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate random data points
np.random.seed(42)
X = np.random.randn(300, 2)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('K-means Clustering Example')
plt.show()
```

Slide 2: K-means Clustering

K-means is a popular clustering algorithm that partitions data into K clusters based on the distance between data points and cluster centroids. It iteratively assigns points to the nearest centroid and updates the centroids until convergence.

```python
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply K-means clustering
kmeans = KMeans(n_clusters=4, random_state=0)
labels = kmeans.fit_predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='r')
plt.title('K-means Clustering with Centroids')
plt.show()
```

Slide 3: Hierarchical Clustering

Hierarchical clustering creates a tree-like structure of clusters, allowing for different levels of granularity. This method can be agglomerative (bottom-up) or divisive (top-down) and doesn't require specifying the number of clusters in advance.

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.rand(20, 2)

# Perform hierarchical clustering
Z = linkage(X, 'ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

Slide 4: DBSCAN: Density-Based Spatial Clustering

DBSCAN groups together points that are closely packed together, marking points in low-density regions as outliers. It's particularly useful for datasets with irregularly shaped clusters and can automatically determine the number of clusters.

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Generate sample data
X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.show()
```

Slide 5: Gaussian Mixture Models (GMM)

GMMs assume that the data points are generated from a mixture of a finite number of Gaussian distributions. This probabilistic approach allows for soft clustering, where each point has a probability of belonging to each cluster.

```python
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.concatenate([np.random.normal(0, 1, (100, 2)),
                    np.random.normal(3, 1.5, (100, 2))])

# Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=42)
labels = gmm.fit_predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Gaussian Mixture Model Clustering')
plt.show()
```

Slide 6: Evaluating Clustering Performance: Silhouette Score

The Silhouette Score measures how similar an object is to its own cluster compared to other clusters. It ranges from -1 to 1, where higher values indicate better-defined clusters.

```python
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Calculate silhouette scores for different numbers of clusters
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Plot the results
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.show()
```

Slide 7: Dimensionality Reduction: PCA for Visualization

Principal Component Analysis (PCA) can be used to reduce the dimensionality of high-dimensional data, making it easier to visualize and analyze clusters.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_pca)

# Plot the results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.title('K-means Clustering on PCA-reduced Iris Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
```

Slide 8: Real-Life Example: Customer Segmentation

Clustering algorithms can be used to segment customers based on their purchasing behavior, helping businesses tailor their marketing strategies and improve customer satisfaction.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Sample customer data (replace with your own dataset)
data = {
    'Customer_ID': range(1, 101),
    'Age': np.random.randint(18, 70, 100),
    'Annual_Income': np.random.randint(20000, 100000, 100),
    'Spending_Score': np.random.randint(1, 100, 100)
}
df = pd.DataFrame(data)

# Preprocessing
X = df[['Age', 'Annual_Income', 'Spending_Score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize results
plt.scatter(df['Annual_Income'], df['Spending_Score'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.show()

print(df.groupby('Cluster').mean())
```

Slide 9: Real-Life Example: Image Segmentation

Clustering algorithms can be applied to image segmentation tasks, such as separating different regions in satellite imagery or medical images.

```python
from sklearn.cluster import KMeans
from skimage import io
import matplotlib.pyplot as plt

# Load a sample image (replace with your own image)
image = io.imread('sample_image.jpg')
image_array = image.reshape(-1, 3)

# Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(image_array)

# Recreate the image with cluster colors
segmented_image = kmeans.cluster_centers_[labels].reshape(image.shape)

# Display original and segmented images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image)
ax1.set_title('Original Image')
ax2.imshow(segmented_image.astype('uint8'))
ax2.set_title('Segmented Image')
plt.show()
```

Slide 10: Handling Mixed Data Types

Real-world datasets often contain a mix of numerical and categorical features. We can use the Gower distance metric for clustering mixed data types.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

# Sample mixed data
data = {
    'Age': [25, 30, 35, 40, 45],
    'Income': [50000, 60000, 75000, 90000, 100000],
    'Education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
    'MaritalStatus': ['Single', 'Married', 'Married', 'Divorced', 'Single']
}
df = pd.DataFrame(data)

# Preprocess data
numeric_features = ['Age', 'Income']
categorical_features = ['Education', 'MaritalStatus']

scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=categorical_features)

# Calculate Gower distance
def gower_distance(X):
    individual_variable_distances = []
    for i in range(X.shape[1]):
        if X.dtypes[i] == np.int64 or X.dtypes[i] == np.float64:
            individual_variable_distances.append((X.iloc[:, i].values[:, None] - X.iloc[:, i].values) ** 2)
        else:
            individual_variable_distances.append(X.iloc[:, i].values[:, None] != X.iloc[:, i].values)
    return np.sqrt(np.array(individual_variable_distances).mean(axis=0))

gower_dist = gower_distance(df_encoded)

# Apply hierarchical clustering
clustering = AgglomerativeClustering(n_clusters=3, affinity='precomputed', linkage='complete')
clustering.fit(gower_dist)

df['Cluster'] = clustering.labels_
print(df)
```

Slide 11: Clustering Time Series Data

Clustering algorithms can be applied to time series data to identify patterns and group similar temporal behaviors.

```python
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets

# Load and preprocess the dataset
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
X = np.concatenate((X_train, X_test))
np.random.shuffle(X)
X = X[:50]  # Select a subset of time series for demonstration

# Apply Time Series K-means
n_clusters = 3
km = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
labels = km.fit_predict(X)

# Plot the results
plt.figure(figsize=(15, 10))
for yi in range(n_clusters):
    plt.subplot(n_clusters, 1, yi + 1)
    for xx in X[labels == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.title(f"Cluster {yi + 1}")
plt.tight_layout()
plt.show()
```

Slide 12: Clustering for Anomaly Detection

Clustering algorithms can be used to identify outliers or anomalies in datasets, which is useful for fraud detection, network security, and quality control.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Generate sample data with outliers
np.random.seed(42)
X = np.random.randn(1000, 2)
X = np.r_[X, np.random.randn(100, 2) + [4, 4]]  # Add some outliers

# Standardize the data
X = StandardScaler().fit_transform(X)

# Apply DBSCAN
db = DBSCAN(eps=0.3, min_samples=10)
labels = db.fit_predict(X)

# Plot the results
plt.figure(figsize=(10, 8))
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'  # Black for noise points
    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

plt.title('DBSCAN Clustering for Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

print(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"Number of noise points: {list(labels).count(-1)}")
```

Slide 13: Clustering for Document Similarity

Clustering can be applied to text data to group similar documents together, which is useful for organizing large collections of text, topic modeling, and content recommendation.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning is a type of machine learning",
    "Neural networks are used in deep learning",
    "Clustering is an unsupervised learning technique",
    "K-means is a popular clustering algorithm",
    "Python is widely used for data science and machine learning"
]

# Create TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Apply K-means clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Print clusters
for i in range(n_clusters):
    print(f"\nCluster {i + 1}:")
    cluster_docs = np.array(documents)[kmeans.labels_ == i]
    for doc in cluster_docs:
        print(f"- {doc}")

# Find the most representative terms for each cluster
feature_names = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

print("\nTop terms per cluster:")
for i in range(n_clusters):
    print(f"\nCluster {i + 1}:")
    for ind in order_centroids[i, :5]:
        print(f"- {feature_names[ind]}")
```

Slide 14: Challenges and Limitations of Clustering Algorithms

Clustering algorithms face several challenges and limitations that users should be aware of. These include sensitivity to initial conditions, difficulty in determining the optimal number of clusters, and the curse of dimensionality. Understanding these limitations is crucial for effectively applying clustering techniques to real-world problems.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Function to run K-means with different initializations
def run_kmeans(X, n_clusters, n_init):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
    labels = kmeans.fit_predict(X)
    return kmeans.inertia_

# Compare K-means with different initializations
n_init_values = [1, 5, 10, 20]
inertias = [run_kmeans(X, 4, n) for n in n_init_values]

plt.plot(n_init_values, inertias, marker='o')
plt.xlabel('Number of Initializations')
plt.ylabel('Inertia')
plt.title('K-means Sensitivity to Initialization')
plt.show()
```

Slide 15: Future Trends in Clustering Algorithms

The field of clustering algorithms continues to evolve, with emerging trends such as deep clustering, online clustering for streaming data, and the integration of domain knowledge into clustering processes. These advancements aim to address current limitations and expand the applicability of clustering techniques to more complex datasets and scenarios.

```python
# Pseudocode for a simple online clustering algorithm

class OnlineKMeans:
    def __init__(self, n_clusters, learning_rate):
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate
        self.centroids = None

    def initialize(self, first_batch):
        # Randomly initialize centroids from the first batch
        self.centroids = first_batch[np.random.choice(len(first_batch), self.n_clusters, replace=False)]

    def update(self, data_point):
        # Find the nearest centroid
        distances = np.linalg.norm(self.centroids - data_point, axis=1)
        nearest_centroid_idx = np.argmin(distances)

        # Update the nearest centroid
        self.centroids[nearest_centroid_idx] += self.learning_rate * (data_point - self.centroids[nearest_centroid_idx])

    def fit(self, data_stream):
        self.initialize(next(data_stream))
        for batch in data_stream:
            for point in batch:
                self.update(point)

# Usage:
# online_kmeans = OnlineKMeans(n_clusters=3, learning_rate=0.1)
# online_kmeans.fit(data_stream_generator())
```

Slide 16: Additional Resources

For those interested in delving deeper into clustering algorithms and their applications, the following resources provide valuable insights:

1. ArXiv paper: "A Survey of Clustering With Deep Learning: From the Perspective of Network Architecture" ([https://arxiv.org/abs/1801.07648](https://arxiv.org/abs/1801.07648))
2. ArXiv paper: "Clustering: A Data Recovery Approach" ([https://arxiv.org/abs/1802.06915](https://arxiv.org/abs/1802.06915))
3. ArXiv paper: "A Comparative Study of Efficient Initialization Methods for the K-Means Clustering Algorithm" ([https://arxiv.org/abs/1209.1960](https://arxiv.org/abs/1209.1960))

These papers offer in-depth discussions on advanced clustering techniques, theoretical foundations, and practical applications in various domains.

