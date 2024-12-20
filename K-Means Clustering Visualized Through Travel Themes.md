## K-Means Clustering Visualized Through Travel Themes
Slide 1: Introduction to K-Means Clustering

K-means clustering is an unsupervised machine learning algorithm that partitions n observations into k clusters, where each observation belongs to the cluster with the nearest mean. The algorithm iteratively assigns points to centroids and updates centroid positions until convergence.

```python
import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k, max_iters=100):
    # Randomly initialize k centroids
    n_samples = X.shape[0]
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) 
                                for i in range(k)])
        
        # Check convergence
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
    
    return labels, centroids
```

Slide 2: Data Generation for Travel Clustering

To demonstrate K-means clustering with travel data, we'll generate synthetic 2D coordinates for our travel-themed objects. Each point represents characteristics like cost and duration, creating meaningful clusters for analysis.

```python
def generate_travel_data(n_samples=200):
    np.random.seed(42)
    
    # Generate clusters for different travel modes
    airplane = np.random.normal(loc=[8, 7], scale=0.5, size=(n_samples//4, 2))
    train = np.random.normal(loc=[4, 4], scale=0.3, size=(n_samples//4, 2))
    car = np.random.normal(loc=[2, 3], scale=0.4, size=(n_samples//4, 2))
    ship = np.random.normal(loc=[6, 2], scale=0.6, size=(n_samples//4, 2))
    
    # Combine all data points
    X = np.vstack([airplane, train, car, ship])
    np.random.shuffle(X)
    
    return X
```

Slide 3: Visualization Setup

Creating a robust visualization framework for our K-means clustering implementation requires careful consideration of plot aesthetics and color schemes to represent different travel clusters effectively.

```python
import seaborn as sns

def setup_visualization():
    # Set style and color palette
    plt.style.use('seaborn')
    colors = sns.color_palette("husl", n_colors=8)
    
    # Create figure with specific size
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Customize plot appearance
    ax.set_title('Travel Clusters', fontsize=14, pad=20)
    ax.set_xlabel('Feature 1 (Cost)', fontsize=12)
    ax.set_ylabel('Feature 2 (Duration)', fontsize=12)
    
    return fig, ax, colors
```

Slide 4: Implementing Animation Framework

Animation of the K-means clustering process requires capturing intermediate states of centroid positions and cluster assignments. We'll use matplotlib's animation functionality to create smooth transitions.

```python
from matplotlib.animation import FuncAnimation

class KMeansAnimation:
    def __init__(self, X, k):
        self.X = X
        self.k = k
        self.fig, self.ax, self.colors = setup_visualization()
        self.centroids = X[np.random.choice(len(X), k, replace=False)]
        self.anim = None
    
    def update(self, frame):
        self.ax.clear()
        
        # Calculate distances and labels
        distances = np.sqrt(((self.X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Plot current state
        for i in range(self.k):
            mask = labels == i
            self.ax.scatter(self.X[mask, 0], self.X[mask, 1], 
                          c=[self.colors[i]], alpha=0.6)
            self.ax.scatter(self.centroids[i, 0], self.centroids[i, 1], 
                          c=[self.colors[i]], marker='*', s=200)
        
        # Update centroids
        new_centroids = np.array([self.X[labels == i].mean(axis=0) 
                                for i in range(self.k)])
        self.centroids = new_centroids
```

Slide 5: Mathematical Foundation of K-Means

The K-means algorithm optimizes the within-cluster sum of squares (WCSS) objective function. Understanding the mathematical principles helps in implementing and optimizing the algorithm effectively.

```python
# Mathematical formulation of K-means
"""
Objective Function (WCSS):
$$J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

Where:
$$\mu_i$$ is the centroid of cluster $$C_i$$
$$||x - \mu_i||^2$$ is the Euclidean distance squared

Update Rule for Centroids:
$$\mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x$$
"""

def calculate_wcss(X, labels, centroids):
    wcss = 0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        wcss += np.sum((cluster_points - centroids[i])**2)
    return wcss
```

Slide 6: Data Preprocessing

Preparing travel-related data for K-means clustering involves scaling features and handling outliers to ensure the algorithm performs optimally. We'll implement robust preprocessing techniques for our travel dataset.

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_travel_data(X):
    # Create scaler object
    scaler = StandardScaler()
    
    # Convert to DataFrame for better handling
    df = pd.DataFrame(X, columns=['cost', 'duration'])
    
    # Remove outliers using IQR method
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df[~((df < (Q1 - 1.5 * IQR)) | 
                     (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    # Scale the cleaned data
    X_scaled = scaler.fit_transform(df_cleaned)
    
    return X_scaled, scaler
```

Slide 7: Optimal Cluster Selection

Determining the optimal number of clusters using the Elbow Method helps identify the most appropriate value of k for our travel data clustering problem.

```python
def elbow_method(X, max_k=10):
    wcss = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        labels, centroids = kmeans(X, k)
        wcss_value = calculate_wcss(X, labels, centroids)
        wcss.append(wcss_value)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    return wcss
```

Slide 8: Travel Emoji Mapping

Creating a mapping system to associate clustering results with travel emojis enhances visualization interpretability and adds a thematic element to our analysis.

```python
class TravelEmojiMapper:
    def __init__(self):
        self.emoji_map = {
            0: '✈️',  # airplane
            1: '🚂',  # train
            2: '🚗',  # car
            3: '🛳️',  # ship
            4: '🎒',  # backpack
            5: '🗺️',  # map
            6: '🏔️',  # mountain
            7: '📸'   # camera
        }
        
    def plot_with_emojis(self, X, labels, centroids):
        plt.figure(figsize=(12, 8))
        for i in range(len(np.unique(labels))):
            mask = labels == i
            plt.scatter(X[mask, 0], X[mask, 1], 
                       label=f'Cluster {self.emoji_map[i]}',
                       alpha=0.6)
            plt.scatter(centroids[i, 0], centroids[i, 1],
                       marker='${}$'.format(self.emoji_map[i]),
                       s=300, c='black')
        plt.legend()
        plt.title('Travel Clusters with Emoji Representation')
```

Slide 9: Silhouette Analysis

Implementing silhouette analysis to evaluate clustering quality and validate the separation between travel-themed clusters. This metric helps assess how well-defined our travel groups are.

```python
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

def silhouette_visualization(X, labels, k):
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    
    # Create silhouette plot
    plt.figure(figsize=(10, 6))
    y_lower = 10
    
    for i in range(k):
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = len(cluster_silhouette_values)
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / k)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_values,
                         facecolor=color, alpha=0.7)
        
        y_lower = y_upper + 10
        
    plt.title(f'Silhouette Analysis (avg score: {silhouette_avg:.3f})')
    plt.xlabel('Silhouette Coefficient')
    plt.ylabel('Cluster')
```

Slide 10: Real-time Cluster Updates

Implementing real-time updates for travel clusters requires efficient computation of centroids and handling dynamic data points. This implementation focuses on optimizing update operations.

```python
class RealTimeKMeans:
    def __init__(self, k, window_size=1000):
        self.k = k
        self.window_size = window_size
        self.centroids = None
        self.buffer = []
        
    def update(self, new_point):
        # Add new point to buffer
        self.buffer.append(new_point)
        
        # Initialize centroids if needed
        if self.centroids is None and len(self.buffer) >= self.k:
            X = np.array(self.buffer)
            self.centroids = X[np.random.choice(len(X), self.k, replace=False)]
        
        # Update clusters when buffer is full
        if len(self.buffer) >= self.window_size:
            X = np.array(self.buffer)
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            for i in range(self.k):
                if np.sum(labels == i) > 0:
                    self.centroids[i] = X[labels == i].mean(axis=0)
            
            # Clear buffer keeping only recent points
            self.buffer = self.buffer[-self.window_size//2:]
        
        return self.centroids
```

Slide 11: Performance Metrics Implementation

Creating comprehensive metrics to evaluate clustering quality specific to travel data patterns, including custom metrics for travel-specific characteristics.

```python
class ClusteringMetrics:
    def __init__(self, X, labels, centroids):
        self.X = X
        self.labels = labels
        self.centroids = centroids
        
    def calculate_metrics(self):
        # Calculate basic metrics
        self.inertia = calculate_wcss(self.X, self.labels, self.centroids)
        self.silhouette = silhouette_score(self.X, self.labels)
        
        # Calculate cluster densities
        self.densities = []
        for i in range(len(self.centroids)):
            cluster_points = self.X[self.labels == i]
            if len(cluster_points) > 0:
                density = len(cluster_points) / np.sum(
                    np.linalg.norm(cluster_points - self.centroids[i], axis=1))
                self.densities.append(density)
        
        return {
            'inertia': self.inertia,
            'silhouette_score': self.silhouette,
            'cluster_densities': self.densities,
            'cluster_sizes': [np.sum(self.labels == i) 
                            for i in range(len(self.centroids))]
        }
```

Slide 12: Hierarchical Travel Clustering

Implementing hierarchical clustering to identify nested relationships within travel data, allowing for multi-level analysis of travel patterns.

```python
from scipy.cluster.hierarchy import linkage, dendrogram

def hierarchical_travel_clustering(X):
    # Compute linkage matrix
    linkage_matrix = linkage(X, method='ward')
    
    # Create dendrogram visualization
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering of Travel Data')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    
    # Cut dendrogram at different heights
    def get_clusters(h):
        from scipy.cluster.hierarchy import fcluster
        return fcluster(linkage_matrix, h, criterion='distance')
    
    # Return function for flexible clustering
    return get_clusters
```

Slide 13: Additional Resources

*   "Revisiting k-means: New Algorithms via Bayesian Nonparametrics" [https://arxiv.org/abs/1512.07687](https://arxiv.org/abs/1512.07687)
*   "Mini-batch k-means clustering of streaming and evolving data" [https://arxiv.org/abs/1902.10000](https://arxiv.org/abs/1902.10000)
*   "K-Means++: The Advantages of Careful Seeding" [https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
*   For more resources on advanced clustering techniques:
    *   Google Scholar: "k-means clustering optimization"
    *   Research Gate: "travel data clustering methods"
    *   ACM Digital Library: "spatiotemporal clustering"

