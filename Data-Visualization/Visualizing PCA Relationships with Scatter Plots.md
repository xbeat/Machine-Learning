## Visualizing PCA Relationships with Scatter Plots
Slide 1: Fundamentals of PCA Visualization

Principal Component Analysis (PCA) dimensionality reduction commonly uses scatter plots to visualize relationships between principal components. These plots reveal patterns, clusters, and variance explained by the first two or three components, which typically capture the most significant data variation.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generate sample data
np.random.seed(42)
n_samples = 300
X = np.random.randn(n_samples, 4)

# Standardize the features
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
plt.title('PCA Components Visualization')
plt.grid(True)
plt.show()
```

Slide 2: Mathematical Foundation of PCA Components

The mathematical basis for PCA visualization stems from eigendecomposition of the covariance matrix. The eigenvectors determine the direction of maximum variance, while eigenvalues quantify the variance explained by each component.

```python
# Mathematical representation in code block (LaTeX format)
$$
\Sigma = \frac{1}{n-1}X^TX
$$
$$
\Sigma v = \lambda v
$$
```

Slide 3: Loading Plot Implementation

Loading plots visualize the contribution of original features to principal components, helping interpret the transformed space. These plots show correlation coefficients between original variables and principal components as vectors in a 2D space.

```python
def create_loading_plot(pca, feature_names):
    loadings = pca.components_.T
    
    plt.figure(figsize=(10, 8))
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                 color='r', alpha=0.5, head_width=0.05)
        plt.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, 
                feature, color='g')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.grid(True)
    plt.axis('equal')
    return plt
```

Slide 4: Biplot Visualization

A biplot combines both the score plot (transformed data points) and loading plot (feature contributions) in a single visualization, providing a comprehensive view of the PCA results and their relationship to original variables.

```python
import seaborn as sns
from sklearn.datasets import load_iris

# Load iris dataset for demonstration
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

# Standardize and apply PCA
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create biplot
def create_biplot(score, coeff, labels=None):
    plt.figure(figsize=(12, 8))
    
    # Plot scores
    plt.scatter(score[:, 0], score[:, 1], alpha=0.5)
    
    # Plot loadings
    for i, (x, y) in enumerate(zip(coeff[:, 0], coeff[:, 1])):
        plt.arrow(0, 0, x*5, y*5, color='r', alpha=0.5)
        if labels is not None:
            plt.text(x*5.15, y*5.15, labels[i], color='r')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.grid(True)
    
create_biplot(X_pca, pca.components_.T, feature_names)
plt.show()
```

Slide 5: Interactive PCA Visualization

Modern PCA visualization often requires interactive elements to explore high-dimensional relationships. This implementation uses Plotly to create an interactive scatter plot with hoverable data points and customizable views.

```python
import plotly.express as px
import pandas as pd

# Prepare data
pca_df = pd.DataFrame(
    X_pca, 
    columns=['PC1', 'PC2']
)
pca_df['Species'] = iris.target

# Create interactive plot
fig = px.scatter(
    pca_df,
    x='PC1',
    y='PC2',
    color='Species',
    hover_data=['PC1', 'PC2'],
    title='Interactive PCA Components Visualization'
)
fig.show()
```

Slide 6: Variance Explained Plot

The scree plot or variance explained plot visualizes the cumulative and individual contribution of each principal component, helping determine the optimal number of components to retain for analysis and visualization purposes.

```python
def plot_variance_explained(pca):
    # Calculate variance ratios
    var_ratio = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(var_ratio)
    
    # Create figure with secondary y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Plot individual variance explained
    ax1.bar(range(1, len(var_ratio) + 1), var_ratio, 
            alpha=0.5, color='b', label='Individual')
    ax1.set_ylabel('Individual Variance Ratio')
    
    # Plot cumulative variance explained
    ax2.plot(range(1, len(cum_var_ratio) + 1), cum_var_ratio, 
             'r-', label='Cumulative')
    ax2.set_ylabel('Cumulative Variance Ratio')
    
    plt.title('Variance Explained by Principal Components')
    plt.xlabel('Principal Component')
    plt.show()
```

Slide 7: 3D PCA Visualization

When three principal components are needed to capture sufficient variance, a 3D scatter plot provides additional insights into data structure and clustering patterns that might not be visible in 2D representations.

```python
from mpl_toolkits.mplot3d import Axes3D

# Compute 3 components
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Create 3D visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                    c=iris.target, cmap='viridis')

ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')

plt.colorbar(scatter)
plt.show()
```

Slide 8: Real-world Example - Gene Expression Analysis

Gene expression data analysis frequently employs PCA visualization to identify patterns in high-dimensional genomic data. This implementation demonstrates preprocessing and visualization of gene expression profiles.

```python
# Generate synthetic gene expression data
n_genes = 1000
n_samples = 100
gene_expr = np.random.lognormal(mean=0, sigma=1, size=(n_samples, n_genes))

# Log transform and standardize
gene_expr_log = np.log2(gene_expr + 1)
gene_expr_scaled = StandardScaler().fit_transform(gene_expr_log)

# Apply PCA
pca_genes = PCA(n_components=2)
gene_expr_pca = pca_genes.fit_transform(gene_expr_scaled)

# Create visualization with sample conditions
conditions = np.array(['control'] * 50 + ['treated'] * 50)
plt.figure(figsize=(10, 8))
for condition in np.unique(conditions):
    mask = conditions == condition
    plt.scatter(gene_expr_pca[mask, 0], gene_expr_pca[mask, 1],
               label=condition, alpha=0.6)

plt.xlabel(f'PC1 ({pca_genes.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca_genes.explained_variance_ratio_[1]:.2%})')
plt.legend()
plt.title('Gene Expression PCA')
plt.show()
```

Slide 9: Confidence Ellipses in PCA Plots

Confidence ellipses enhance PCA scatter plots by visualizing the statistical uncertainty and distribution of different groups in the transformed space, providing insights into cluster separation and overlap.

```python
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=3.0, **kwargs):
    # Calculate the eigenvectors and eigenvalues
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Calculate the ellipse radius on x and y
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    # Create the ellipse object
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      **kwargs)
    
    # Move the ellipse to the mean of the points
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    transform = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(np.std(x) * n_std, np.std(y) * n_std) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transform + ax.transData)
    return ax.add_patch(ellipse)

# Example usage
plt.figure(figsize=(10, 8))
for i, label in enumerate(np.unique(iris.target)):
    mask = iris.target == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Class {i}')
    confidence_ellipse(X_pca[mask, 0], X_pca[mask, 1], plt.gca(),
                      alpha=0.1, facecolor='none', edgecolor='black')

plt.legend()
plt.show()
```

Slide 10: Advanced Projection Techniques

Principal component visualization can be enhanced using advanced projection techniques that preserve local structure. This implementation combines PCA with t-SNE for better visualization of complex, non-linear relationships.

```python
from sklearn.manifold import TSNE
import time

def advanced_projection_plot(X, labels, perplexity=30):
    # First reduce dimensionality with PCA
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)
    
    # Apply t-SNE on PCA results
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    t0 = time.time()
    X_tsne = tsne.fit_transform(X_pca)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                         c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'PCA + t-SNE Projection (perplexity={perplexity})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    print(f"t-SNE done! Time elapsed: {time.time()-t0:.2f} seconds")
    plt.show()
```

Slide 11: Quality Assessment Visualization

The quality of PCA projections can be assessed through various metrics visualization, including reconstruction error and local structure preservation scores across different numbers of components.

```python
def plot_quality_metrics(X, max_components=10):
    # Calculate reconstruction error and local structure preservation
    reconstruction_errors = []
    explained_variances = []
    
    for n in range(1, max_components + 1):
        pca = PCA(n_components=n)
        X_transformed = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        # Reconstruction error
        mse = np.mean((X - X_reconstructed) ** 2)
        reconstruction_errors.append(mse)
        
        # Explained variance
        explained_variances.append(np.sum(pca.explained_variance_ratio_))
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot reconstruction error
    ax1.plot(range(1, max_components + 1), reconstruction_errors, 'b-o')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Reconstruction Error (MSE)')
    ax1.set_title('Reconstruction Error vs Components')
    
    # Plot explained variance
    ax2.plot(range(1, max_components + 1), explained_variances, 'r-o')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Explained Variance vs Components')
    
    plt.tight_layout()
    plt.show()
```

Slide 12: Real-world Example - Image Data Visualization

PCA visualization applied to image data reveals patterns in pixel space and can be used for facial recognition and image compression applications. This implementation shows how to visualize image dataset projections.

```python
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.gridspec as gridspec

# Load face dataset
faces = fetch_olivetti_faces()
X_faces = faces.data
y_faces = faces.target

# Apply PCA
pca_faces = PCA(n_components=2)
X_faces_pca = pca_faces.fit_transform(X_faces)

# Create visualization
fig = plt.figure(figsize=(15, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

# Scatter plot of face projections
ax1 = plt.subplot(gs[0, :])
scatter = ax1.scatter(X_faces_pca[:, 0], X_faces_pca[:, 1],
                     c=y_faces, cmap='viridis')
ax1.set_title('Face Images PCA Projection')

# Show example faces
for i, ax in enumerate([plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])]):
    ax.imshow(X_faces[i].reshape(64, 64), cmap='gray')
    ax.set_title(f'Face {i+1}')
    ax.axis('off')

plt.colorbar(scatter)
plt.tight_layout()
plt.show()
```

Slide 13: Additional Resources

*   "Understanding Principal Component Analysis Through Visual Examples"
*   [https://arxiv.org/abs/2010.09113](https://arxiv.org/abs/2010.09113)
*   "Visualizing Data using t-SNE combined with PCA"
*   [https://arxiv.org/abs/1901.01902](https://arxiv.org/abs/1901.01902)
*   "A Survey on Visualization Techniques for High-Dimensional Data Analysis"
*   [https://arxiv.org/abs/2009.14393](https://arxiv.org/abs/2009.14393)
*   "Modern Approaches to PCA-based Anomaly Detection in High Dimensional Data"
*   [https://arxiv.org/abs/2104.12392](https://arxiv.org/abs/2104.12392)
*   "Interactive Visual Analysis of High-Dimensional Data Using PCA"
*   [https://arxiv.org/abs/1907.05234](https://arxiv.org/abs/1907.05234)

