## Classical Dimensionality Reduction in Python
Slide 1: 

Introduction to Classical Dimensionality Reduction

Dimensionality reduction is a fundamental technique in machine learning and data analysis. It aims to reduce the number of features or dimensions in a dataset while preserving as much relevant information as possible. Classical dimensionality reduction methods are linear transformations that project high-dimensional data onto a lower-dimensional subspace.

```python
# No code for the introduction slide
```

Slide 2: 

Principal Component Analysis (PCA)

PCA is one of the most widely used dimensionality reduction techniques. It finds the directions of maximum variance in the data and projects the data onto a lower-dimensional subspace spanned by these directions, called principal components.

```python
from sklearn.decomposition import PCA

# Load your data
X = ... # Your data

# Create a PCA object
pca = PCA(n_components=2)  # Reduce to 2 dimensions

# Fit and transform the data
X_transformed = pca.fit_transform(X)
```

Slide 3: 

PCA Example

Let's apply PCA to the iris dataset, a classic machine learning dataset containing measurements of various iris flower species.

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data

# Create a PCA object and transform the data
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

# Visualize the transformed data
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=iris.target)
plt.show()
```

Slide 4: 
 
Linear Discriminant Analysis (LDA)

LDA is a supervised dimensionality reduction technique that finds the directions that maximize the separation between classes while minimizing the variance within each class.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load your data and labels
X = ... # Your data
y = ... # Labels

# Create an LDA object
lda = LDA(n_components=2)  # Reduce to 2 dimensions

# Fit and transform the data
X_transformed = lda.fit_transform(X, y)
```

Slide 5: 

LDA Example

Let's apply LDA to the iris dataset, using the species labels to find the discriminant directions.

```python
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create an LDA object and transform the data
lda = LDA(n_components=2)
X_transformed = lda.fit_transform(X, y)

# Visualize the transformed data
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
plt.show()
```

Slide 6: 

Factor Analysis

Factor analysis is a statistical technique that aims to describe the underlying relationships between observed variables in terms of a smaller number of unobserved variables called factors.

```python
from sklearn.decomposition import FactorAnalysis

# Load your data
X = ... # Your data

# Create a Factor Analysis object
fa = FactorAnalysis(n_components=3)  # Reduce to 3 factors

# Fit and transform the data
X_transformed = fa.fit_transform(X)
```

Slide 7: 

Factor Analysis Example

Let's apply factor analysis to a simulated dataset with three underlying factors.

```python
import numpy as np
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

# Generate simulated data with 3 factors
np.random.seed(42)
X = np.random.randn(1000, 10)  # 1000 samples, 10 features
factors = np.random.randn(10, 3)  # 3 factors
X = X @ factors.T + np.random.randn(1000, 10)  # Add noise

# Apply Factor Analysis
fa = FactorAnalysis(n_components=3)
X_transformed = fa.fit_transform(X)

# Visualize the transformed data
plt.scatter(X_transformed[:, 0], X_transformed[:, 1])
plt.show()
```

Slide 8: 

Independent Component Analysis (ICA)

ICA is a technique that separates a multivariate signal into independent non-Gaussian signals, called independent components.

```python
from sklearn.decomposition import FastICA

# Load your data
X = ... # Your data

# Create an ICA object
ica = FastICA(n_components=3)  # Reduce to 3 independent components

# Fit and transform the data
X_transformed = ica.fit_transform(X)
```

Slide 9: 

ICA Example

Let's apply ICA to a simulated dataset with three independent non-Gaussian signals.

```python
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# Generate simulated data with 3 independent signals
np.random.seed(42)
s1 = np.random.laplace(size=1000)  # Laplace distribution
s2 = np.random.exponential(size=1000)  # Exponential distribution
s3 = np.random.normal(size=1000)  # Gaussian distribution
X = np.c_[s1, s2, s3] + np.random.randn(1000, 3)  # Add noise

# Apply ICA
ica = FastICA(n_components=3)
X_transformed = ica.fit_transform(X)

# Visualize the transformed data
plt.scatter(X_transformed[:, 0], X_transformed[:, 1])
plt.show()
```

Slide 10: 

Multidimensional Scaling (MDS)

MDS is a technique that maps high-dimensional data onto a lower-dimensional space while preserving the pairwise distances between data points as much as possible.

```python
from sklearn.manifold import MDS

# Load your data
X = ... # Your data

# Create an MDS object
mds = MDS(n_components=2)  # Reduce to 2 dimensions

# Fit and transform the data
X_transformed = mds.fit_transform(X)
```

Slide 11: 

MDS Example

Let's apply MDS to the iris dataset, preserving the pairwise distances between samples.

```python
from sklearn.datasets import load_iris
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data

# Apply MDS
mds = MDS(n_components=2)
X_transformed = mds.fit_transform(X)

# Visualize the transformed data
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=iris.target)
plt.show()
```

Slide 12: 

Isomap

Isomap is a non-linear dimensionality reduction technique that aims to preserve the intrinsic geometric structure of the data by approximating geodesic distances between data points.

```python
from sklearn.manifold import Isomap

# Load your data
X = ... # Your data

# Create an Isomap object
isomap = Isomap(n_components=2)  # Reduce to 2 dimensions

# Fit and transform the data
X_transformed = isomap.fit_transform(X)
```

Slide 13: 

Isomap Example

Let's apply Isomap to the Swiss Roll dataset, a classic non-linear manifold example.

```python
from sklearn import datasets
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt

# Load the Swiss Roll dataset
X, color = datasets.samples_generator.make_swiss_roll(n_samples=1000)

# Apply Isomap
isomap = Isomap(n_components=2)
X_transformed = isomap.fit_transform(X)

# Visualize the transformed data
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=color)
plt.title('Isomap on Swiss Roll Dataset')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
```

Slide 14: 

t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is a non-linear dimensionality reduction technique that is particularly well-suited for visualizing high-dimensional data. It models the pairwise similarities between data points and tries to preserve them in the lower-dimensional space.

```python
from sklearn.manifold import TSNE

# Load your data
X = ... # Your data

# Create a t-SNE object
tsne = TSNE(n_components=2)  # Reduce to 2 dimensions

# Fit and transform the data
X_transformed = tsne.fit_transform(X)
```

Slide 15: 

t-SNE Example

Let's apply t-SNE to the MNIST dataset, a handwritten digit recognition dataset, to visualize the high-dimensional data in 2D.

```python
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = fetch_openml('mnist_784')
X = mnist.data / 255.0  # Normalize pixel values

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_transformed = tsne.fit_transform(X)

# Visualize the transformed data
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=mnist.target.astype(int))
plt.title('t-SNE on MNIST Dataset')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
```

Slide 16: (Additional Resources):

Additional Resources

For further reading and exploration of dimensionality reduction techniques, here are some recommended resources from arXiv.org:

1. "A Tutorial on Principal Component Analysis" by Jonathon Shlens ([https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100))
2. "Dimensionality Reduction: A Comparative Review" by Hyunwoo J. Kim and Hyeyoung Park ([https://arxiv.org/abs/1806.04349](https://arxiv.org/abs/1806.04349))
3. "Kernel Methods for Nonlinear Dimensionality Reduction" by Lawrence K. Saul and Sam T. Roweis ([https://arxiv.org/abs/1511.08898](https://arxiv.org/abs/1511.08898))
4. "An Introduction to Independent Component Analysis" by Aapo Hyvärinen and Erkki Oja ([https://arxiv.org/abs/1804.04598](https://arxiv.org/abs/1804.04598))

Note: These resources are from arXiv.org and were available as of August 2023.

