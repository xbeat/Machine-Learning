## Scatter Plot for PCA Visualization
Slide 1: Introduction to PCA Visualization

Principal Component Analysis (PCA) visualization commonly employs scatter plots to represent relationships between the first two principal components. These plots reveal clustering patterns, outliers, and the overall structure of high-dimensional data projected onto a 2D space.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generate sample data
np.random.seed(42)
n_samples = 300
X = np.random.randn(n_samples, 4)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Create scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Scatter Plot')
plt.grid(True)
plt.show()
```

Slide 2: Loading Plot Visualization

Loading plots display the contribution of original features to principal components, represented as vectors in a 2D coordinate system. The length and direction of vectors indicate the strength and relationship of features to principal components.

```python
def plot_loadings(pca, feature_names):
    loadings = pca.components_.T
    
    plt.figure(figsize=(10, 8))
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                 head_width=0.05, head_length=0.05)
        plt.text(loadings[i, 0]* 1.15, loadings[i, 1] * 1.15, feature)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Loading Plot')
    plt.grid(True)
    plt.axis('equal')
    
    # Add a circle for scale
    circle = plt.Circle((0,0), 1, fill=False, linestyle='--')
    plt.gca().add_artist(circle)
    plt.axis([-1.5, 1.5, -1.5, 1.5])
```

Slide 3: Biplot Implementation

A biplot combines both the scatter plot of samples and the loading vectors in a single visualization, providing a comprehensive view of the relationship between samples and original features in the PCA space.

```python
def create_biplot(X_pca, loadings, features, scale=1):
    plt.figure(figsize=(12, 8))
    
    # Plot samples
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
    
    # Plot feature vectors
    for i, feature in enumerate(features):
        plt.arrow(0, 0, 
                 loadings[i, 0] * scale, 
                 loadings[i, 1] * scale,
                 color='r', alpha=0.5)
        plt.text(loadings[i, 0] * scale * 1.15,
                loadings[i, 1] * scale * 1.15,
                feature)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Biplot')
    plt.grid(True)
```

Slide 4: Explained Variance Plot

The explained variance plot visualizes the cumulative proportion of variance explained by each principal component, helping determine the optimal number of components to retain in the analysis.

```python
def plot_explained_variance(pca):
    plt.figure(figsize=(10, 6))
    
    # Calculate cumulative explained variance ratio
    cum_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    # Create bar plot
    plt.bar(range(1, len(cum_var_ratio) + 1), 
            pca.explained_variance_ratio_,
            alpha=0.5, label='Individual')
    
    # Add line plot for cumulative variance
    plt.step(range(1, len(cum_var_ratio) + 1), 
             cum_var_ratio,
             where='mid',
             label='Cumulative')
    
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.legend()
    plt.grid(True)
```

Slide 5: Real-World Example - Iris Dataset

The iris dataset serves as a classic example for PCA visualization, containing measurements of iris flowers. We'll implement a complete analysis pipeline including data preprocessing, PCA transformation, and various visualization techniques.

```python
from sklearn.datasets import load_iris

# Load and prepare data
iris = load_iris()
X = iris.data
feature_names = iris.feature_names
y = iris.target

# Standardize data
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Create main scatter plot with color coding
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=y, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Iris Dataset - PCA Visualization')
plt.colorbar(scatter)
plt.show()
```

Slide 6: 3D PCA Visualization

Three-dimensional PCA plots utilize the first three principal components to provide additional insight into data structure. This visualization technique is particularly useful when two components alone don't explain sufficient variance.

```python
from mpl_toolkits.mplot3d import Axes3D

def plot_pca_3d(X_pca, labels=None, title="3D PCA Visualization"):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                        c=labels if labels is not None else 'b',
                        cmap='viridis')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(title)
    
    if labels is not None:
        plt.colorbar(scatter)
    
    plt.tight_layout()
    return fig
```

Slide 7: Interactive PCA Visualization with Plotly

Modern PCA visualization benefits from interactive plotting libraries like Plotly, enabling users to zoom, rotate, and hover over data points for additional information.

```python
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_interactive_pca_plot(X_pca, labels=None, feature_names=None):
    # Create DataFrame for plotting
    df = pd.DataFrame(data=X_pca[:, :3], 
                     columns=['PC1', 'PC2', 'PC3'])
    
    if labels is not None:
        df['Label'] = labels
    
    # Create interactive scatter plot
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3',
                        color='Label' if labels is not None else None,
                        title='Interactive PCA Visualization')
    
    fig.update_layout(scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    ))
    
    return fig
```

Slide 8: Real-World Example - Wine Quality Dataset

The Wine Quality dataset demonstrates PCA visualization for complex chemical compositions, revealing underlying patterns in wine characteristics and their relationships.

```python
from sklearn.datasets import load_wine

def analyze_wine_dataset():
    # Load wine dataset
    wine = load_wine()
    X = wine.data
    y = wine.target
    
    # Standardize and apply PCA
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=y, cmap='viridis')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('Wine Quality - PCA Scatter')
    
    # Explained variance
    explained_var = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, len(explained_var) + 1), 
             explained_var, 'bo-')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Explained Variance Ratio')
    
    plt.tight_layout()
    return fig
```

Slide 9: Hierarchical Clustering with PCA

Combining hierarchical clustering with PCA visualization reveals both cluster structure and dimensional relationships, providing insights into natural groupings within the data.

```python
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

def plot_hierarchical_pca(X_pca, n_clusters=3):
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clustering.fit_predict(X_pca)
    
    # Create linkage matrix
    linkage_matrix = linkage(X_pca, method='ward')
    
    # Create subplot layout
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2)
    
    # Plot dendrogram
    ax1 = fig.add_subplot(gs[0, 0])
    dendrogram(linkage_matrix, ax=ax1)
    ax1.set_title('Hierarchical Clustering Dendrogram')
    
    # Plot PCA with cluster colors
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=cluster_labels, cmap='viridis')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('PCA with Cluster Labels')
    
    plt.colorbar(scatter)
    plt.tight_layout()
    return fig
```

Slide 10: Contribution Plot Analysis

Contribution plots reveal the relative importance of each feature to the principal components, helping identify which variables drive the most variation in the transformed space.

```python
def plot_feature_contributions(pca, feature_names):
    # Calculate absolute contributions
    contributions = np.abs(pca.components_)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot PC1 contributions
    ax1.bar(feature_names, contributions[0])
    ax1.set_title('Feature Contributions to PC1')
    ax1.set_xticklabels(feature_names, rotation=45)
    
    # Plot PC2 contributions
    ax2.bar(feature_names, contributions[1])
    ax2.set_title('Feature Contributions to PC2')
    ax2.set_xticklabels(feature_names, rotation=45)
    
    plt.tight_layout()
    return fig
```

Slide 11: Confidence Ellipses in PCA

Confidence ellipses provide statistical boundaries for clustered data in PCA space, helping visualize the uncertainty and overlap between different groups.

```python
from matplotlib.patches import Ellipse
import scipy.stats as stats

def plot_confidence_ellipses(X_pca, labels, confidence=0.95):
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        x = X_pca[mask, 0]
        y = X_pca[mask, 1]
        
        # Calculate mean and covariance
        mean = np.mean(X_pca[mask, :2], axis=0)
        cov = np.cov(X_pca[mask, :2].T)
        
        # Calculate eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        
        # Create confidence ellipse
        chi2_val = stats.chi2.ppf(confidence, df=2)
        sqrt_chi2_val = np.sqrt(chi2_val)
        width, height = 2 * sqrt_chi2_val * np.sqrt(eigvals)
        
        ellipse = Ellipse(xy=mean, width=width, height=height,
                         angle=angle, color=color, alpha=0.3)
        
        plt.scatter(x, y, c=[color], label=f'Class {label}')
        plt.gca().add_patch(ellipse)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA with Confidence Ellipses')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()
```

Slide 12: Time Series PCA Trajectory

Visualizing PCA trajectories for time series data reveals temporal patterns and cyclic behavior in the principal component space.

```python
def plot_pca_trajectory(X_pca, time_points=None):
    if time_points is None:
        time_points = np.arange(len(X_pca))
    
    plt.figure(figsize=(12, 8))
    
    # Plot scatter points
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=time_points, cmap='viridis')
    
    # Connect points with lines to show trajectory
    plt.plot(X_pca[:, 0], X_pca[:, 1], 
             'k-', alpha=0.3, linewidth=0.5)
    
    # Add arrows to show direction
    for i in range(0, len(X_pca)-1, max(1, len(X_pca)//20)):
        plt.arrow(X_pca[i, 0], X_pca[i, 1],
                 X_pca[i+1, 0] - X_pca[i, 0],
                 X_pca[i+1, 1] - X_pca[i, 1],
                 head_width=0.1, head_length=0.1,
                 fc='r', ec='r', alpha=0.5)
    
    plt.colorbar(scatter, label='Time Point')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Trajectory Analysis')
    plt.grid(True)
    
    return plt.gcf()
```

Slide 13: Advanced Visualization Metrics

Advanced PCA visualization includes quality metrics such as reconstruction error and local structure preservation, helping assess the reliability of the dimensionality reduction.

```python
def calculate_visualization_metrics(X, X_pca, pca):
    # Calculate reconstruction error
    X_reconstructed = pca.inverse_transform(X_pca)
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Reconstruction Error
    plt.subplot(131)
    plt.hist(np.sum((X - X_reconstructed) ** 2, axis=1),
             bins=30, alpha=0.5)
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    
    # Plot 2: Cumulative Variance
    plt.subplot(132)
    cum_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cum_var_ratio) + 1), 
             cum_var_ratio, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Ratio')
    
    # Plot 3: Component Correlations
    plt.subplot(133)
    corr = np.corrcoef(X_pca.T)
    plt.imshow(corr, cmap='coolwarm')
    plt.colorbar()
    plt.title('PC Correlation Matrix')
    
    plt.tight_layout()
    return plt.gcf(), reconstruction_error
```

Slide 14: Additional Resources

*   "A Tutorial on Principal Component Analysis" - [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
*   "Visualizing Data using t-SNE" - [https://arxiv.org/abs/1808.01120](https://arxiv.org/abs/1808.01120)
*   "Understanding the Role of Individual Units in a Deep Neural Network" - [https://arxiv.org/abs/2009.05041](https://arxiv.org/abs/2009.05041)
*   "Dimensionality Reduction: A Comparative Review" - [https://arxiv.org/abs/0904.1796](https://arxiv.org/abs/0904.1796)
*   "Visual Analytics of High-Dimensional Data" - [https://arxiv.org/abs/1909.04729](https://arxiv.org/abs/1909.04729)

