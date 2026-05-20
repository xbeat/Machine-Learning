## Unsupervised Anomaly Detection Methods
Slide 1: Isolation Forest Algorithm

Isolation Forest is a powerful unsupervised anomaly detection method based on the principle that anomalies are easier to isolate than normal data points. It randomly selects a feature and splits data between minimum and maximum values, creating isolation trees where anomalies require fewer splits.

```python
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Generate sample data with anomalies
np.random.seed(42)
X_normal = np.random.normal(0, 0.5, (100, 2))
X_anomaly = np.random.uniform(-4, 4, (10, 2))
X = np.concatenate([X_normal, X_anomaly])

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
yhat = iso_forest.fit_predict(X)

# Plot results
plt.scatter(X[yhat==1, 0], X[yhat==1, 1], c='blue', label='Normal')
plt.scatter(X[yhat==-1, 0], X[yhat==-1, 1], c='red', label='Anomaly')
plt.legend()
plt.show()
```

Slide 2: Local Outlier Factor (LOF)

LOF identifies anomalies by measuring the local deviation of a data point with respect to its neighbors. It calculates a score reflecting the degree of abnormality based on local density, where points with substantially lower density than their neighbors are considered outliers.

```python
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
rng = np.random.RandomState(42)
X_inliers = 0.3 * rng.randn(100, 2)
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X_inliers, X_outliers]

# Fit LOF detector
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
scores = clf.negative_outlier_factor_

# Plot results
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='blue', label='Normal')
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='red', label='Anomaly')
plt.legend()
plt.show()
```

Slide 3: One-Class SVM

One-Class SVM learns a decision boundary that encompasses the normal data points in feature space. It maps input data into a high dimensional feature space and iteratively finds the maximal margin hyperplane that separates the dataset from the origin.

```python
from sklearn.svm import OneClassSVM
import numpy as np
import matplotlib.pyplot as plt

# Generate normal and anomaly data
X_train = np.random.normal(0, 0.5, (200, 2))
X_test = np.r_[np.random.normal(0, 0.5, (100, 2)),
               np.random.normal(3, 0.5, (50, 2))]

# Train One-Class SVM
clf = OneClassSVM(nu=0.1, kernel="rbf", gamma='scale')
clf.fit(X_train)

# Predict anomalies
y_pred_test = clf.predict(X_test)

# Plot results
plt.scatter(X_test[y_pred_test == 1, 0], X_test[y_pred_test == 1, 1], 
           c='blue', label='Normal')
plt.scatter(X_test[y_pred_test == -1, 0], X_test[y_pred_test == -1, 1], 
           c='red', label='Anomaly')
plt.legend()
plt.show()
```

Slide 4: Autoencoder-based Anomaly Detection

Autoencoders learn to compress normal data into a lower-dimensional latent space and reconstruct it. When anomalies are presented, the reconstruction error is typically higher than for normal data, making it an effective unsupervised detection method.

```python
import tensorflow as tf
import numpy as np

# Create autoencoder model
def build_autoencoder():
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu')
    ])
    
    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(100, activation='sigmoid')
    ])
    
    autoencoder = tf.keras.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Generate sample data
normal_data = np.random.normal(0, 1, (1000, 100))
anomaly_data = np.random.normal(2, 1, (100, 100))

# Train autoencoder
model = build_autoencoder()
model.fit(normal_data, normal_data, epochs=50, batch_size=32, verbose=0)

# Detect anomalies
threshold = np.mean(model.predict(normal_data))
```

Slide 5: DBSCAN Clustering for Anomaly Detection

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies clusters of normal points based on density connectivity and marks sparse points as anomalies. It's particularly effective for datasets with non-spherical clusters and varying densities.

```python
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.normal(0, 1, (300, 2))
outliers = np.random.uniform(low=-4, high=4, size=(15, 2))
X = np.vstack([X, outliers])

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)

# Plot results
plt.scatter(X[labels != -1, 0], X[labels != -1, 1], 
           c='blue', label='Normal')
plt.scatter(X[labels == -1, 0], X[labels == -1, 1], 
           c='red', label='Anomaly')
plt.legend()
plt.show()
```

Slide 6: Gaussian Mixture Models (GMM)

GMM learns the probability distribution of normal data using a mixture of Gaussian distributions. Points with low probability under the learned distribution are classified as anomalies. This method is particularly effective for multimodal data distributions.

```python
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# Generate multimodal normal data
X_normal = np.concatenate([
    np.random.normal(0, 0.5, (200, 2)),
    np.random.normal(3, 0.5, (200, 2))
])
X_anomaly = np.random.uniform(-2, 5, (50, 2))
X = np.vstack([X_normal, X_anomaly])

# Fit GMM
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_normal)

# Predict anomalies
scores = -gmm.score_samples(X)
threshold = np.percentile(scores, 90)
labels = (scores > threshold).astype(int)

# Plot results
plt.scatter(X[labels == 0, 0], X[labels == 0, 1], c='blue', label='Normal')
plt.scatter(X[labels == 1, 0], X[labels == 1, 1], c='red', label='Anomaly')
plt.legend()
plt.show()
```

Slide 7: Robust Random Cut Forest

Robust Random Cut Forest is a tree-based algorithm that creates a forest of random cut trees, where each tree partitions the space recursively. The algorithm assigns anomaly scores based on the average path length required to isolate each point.

```python
import numpy as np
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.preprocessing import StandardScaler

class RobustRandomCutForest:
    def __init__(self, n_estimators=100, max_depth=10):
        self.embedding = RandomTreesEmbedding(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def fit_predict(self, X):
        # Transform data using random trees
        X_transformed = self.embedding.fit_transform(X).toarray()
        
        # Calculate average path length for each point
        path_lengths = np.sum(X_transformed, axis=1)
        
        # Scale scores
        scores = self.scaler.fit_transform(path_lengths.reshape(-1, 1))
        
        # Classify points with scores > 2 as anomalies
        return (scores > 2).ravel().astype(int)

# Example usage
X = np.concatenate([
    np.random.normal(0, 1, (100, 2)),
    np.random.normal(5, 0.5, (10, 2))
])

rrcf = RobustRandomCutForest()
labels = rrcf.fit_predict(X)
```

Slide 8: Mahalanobis Distance-Based Detection

The Mahalanobis distance measures the distance between a point and the distribution of the data, taking into account the covariance structure. This method is particularly effective for detecting outliers in multivariate normal distributions.

```python
import numpy as np
from scipy.stats import chi2

class MahalanobisDetector:
    def __init__(self, threshold_percentile=97.5):
        self.threshold_percentile = threshold_percentile
        
    def fit_predict(self, X):
        # Calculate mean and covariance
        mean = np.mean(X, axis=0)
        covariance = np.cov(X, rowvar=False)
        
        # Calculate Mahalanobis distances
        inv_covmat = np.linalg.inv(covariance)
        diff = X - mean
        mahal_dist = np.sqrt(np.sum(np.dot(diff, inv_covmat) * diff, axis=1))
        
        # Set threshold based on chi-square distribution
        threshold = chi2.ppf(self.threshold_percentile/100, df=X.shape[1])
        
        return (mahal_dist > np.sqrt(threshold)).astype(int)

# Example usage
np.random.seed(42)
X_normal = np.random.multivariate_normal(
    mean=[0, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=200
)
X_anomaly = np.random.multivariate_normal(
    mean=[3, 3],
    cov=[[1, 0.5], [0.5, 1]],
    size=20
)
X = np.vstack([X_normal, X_anomaly])

detector = MahalanobisDetector()
anomalies = detector.fit_predict(X)
```

Slide 9: Principal Component Analysis (PCA) for Anomaly Detection

PCA-based anomaly detection projects data onto principal components and identifies anomalies based on reconstruction error or deviation in the transformed space. This method is particularly effective for high-dimensional data with linear correlations.

```python
from sklearn.decomposition import PCA
import numpy as np

class PCADetector:
    def __init__(self, n_components=2, threshold_factor=3):
        self.pca = PCA(n_components=n_components)
        self.threshold_factor = threshold_factor
        
    def fit_predict(self, X):
        # Fit PCA
        X_transformed = self.pca.fit_transform(X)
        
        # Project back to original space
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        
        # Calculate reconstruction error
        reconstruction_error = np.sum((X - X_reconstructed) ** 2, axis=1)
        
        # Set threshold based on error distribution
        threshold = (np.mean(reconstruction_error) + 
                    self.threshold_factor * np.std(reconstruction_error))
        
        return (reconstruction_error > threshold).astype(int)

# Generate example data
np.random.seed(42)
X_normal = np.random.multivariate_normal(
    mean=[0, 0, 0],
    cov=[[1, 0.5, 0.3], 
         [0.5, 1, 0.2], 
         [0.3, 0.2, 1]],
    size=300
)
X_anomaly = np.random.multivariate_normal(
    mean=[3, 3, 3],
    cov=[[1, 0.5, 0.3], 
         [0.5, 1, 0.2], 
         [0.3, 0.2, 1]],
    size=30
)
X = np.vstack([X_normal, X_anomaly])

detector = PCADetector()
anomalies = detector.fit_predict(X)
```

Slide 10: Kernel Density Estimation (KDE)

KDE estimates the probability density function of the normal data distribution using kernel functions. Points in regions of low density are classified as anomalies. This non-parametric method can capture complex data distributions.

```python
from sklearn.neighbors import KernelDensity
import numpy as np

class KDEAnomalyDetector:
    def __init__(self, bandwidth=0.5, threshold_percentile=1):
        self.kde = KernelDensity(bandwidth=bandwidth)
        self.threshold_percentile = threshold_percentile
        
    def fit_predict(self, X):
        # Fit KDE to the data
        self.kde.fit(X)
        
        # Calculate log probability density for each point
        log_density = self.kde.score_samples(X)
        
        # Set threshold based on percentile
        threshold = np.percentile(log_density, self.threshold_percentile)
        
        return (log_density < threshold).astype(int)

# Example usage with multimodal data
np.random.seed(42)
X_normal = np.concatenate([
    np.random.normal(-2, 0.5, (150, 2)),
    np.random.normal(2, 0.5, (150, 2))
])
X_anomaly = np.random.uniform(-4, 4, (30, 2))
X = np.vstack([X_normal, X_anomaly])

detector = KDEAnomalyDetector()
anomalies = detector.fit_predict(X)
```

Slide 11: Time Series Anomaly Detection using LSTM Autoencoders

LSTM Autoencoders are particularly effective for detecting anomalies in sequential data by learning temporal dependencies. The model learns to reconstruct normal time series patterns and identifies sequences with high reconstruction error as anomalies.

```python
import tensorflow as tf
import numpy as np

class LSTMAutoencoder:
    def __init__(self, sequence_length, n_features, latent_dim=32):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model(latent_dim)
        
    def _build_model(self, latent_dim):
        # Encoder
        encoder_inputs = tf.keras.layers.Input(shape=(self.sequence_length, self.n_features))
        encoder = tf.keras.layers.LSTM(latent_dim, return_sequences=False)(encoder_inputs)
        
        # Decoder
        decoder = tf.keras.layers.RepeatVector(self.sequence_length)(encoder)
        decoder = tf.keras.layers.LSTM(self.n_features, return_sequences=True)(decoder)
        
        # Autoencoder model
        autoencoder = tf.keras.Model(encoder_inputs, decoder)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder
    
    def fit(self, X, epochs=50, batch_size=32):
        self.model.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0)
    
    def detect_anomalies(self, X, threshold_sigma=3):
        # Get reconstruction error
        reconstructed = self.model.predict(X)
        mse = np.mean(np.square(X - reconstructed), axis=(1,2))
        
        # Set threshold based on standard deviation
        threshold = np.mean(mse) + threshold_sigma * np.std(mse)
        return (mse > threshold).astype(int)

# Generate sample time series data
np.random.seed(42)
n_samples = 1000
sequence_length = 50
n_features = 1

# Normal sequences
normal_data = np.array([
    np.sin(np.linspace(0, 10, sequence_length)) + np.random.normal(0, 0.1, sequence_length)
    for _ in range(n_samples)
])
normal_data = normal_data.reshape(-1, sequence_length, n_features)

# Anomalous sequences
anomaly_data = np.random.normal(0, 1, (100, sequence_length, n_features))

# Combine data
X = np.vstack([normal_data, anomaly_data])

# Detect anomalies
detector = LSTMAutoencoder(sequence_length, n_features)
detector.fit(X)
anomalies = detector.detect_anomalies(X)
```

Slide 12: Histogram-Based Outlier Score (HBOS)

HBOS is a fast unsupervised method that calculates anomaly scores using histogram density estimation. It assumes feature independence and combines per-feature histograms to compute final anomaly scores.

```python
import numpy as np
from scipy.stats import norm

class HBOS:
    def __init__(self, n_bins=10, alpha=0.1):
        self.n_bins = n_bins
        self.alpha = alpha
        self.histograms = []
        self.bin_edges = []
        
    def fit_predict(self, X):
        n_features = X.shape[1]
        scores = np.zeros(X.shape[0])
        
        for feature in range(n_features):
            # Calculate histogram for each feature
            hist, edges = np.histogram(X[:, feature], bins=self.n_bins, density=True)
            self.histograms.append(hist)
            self.bin_edges.append(edges)
            
            # Calculate bin indices for each sample
            bin_indices = np.digitize(X[:, feature], edges[:-1])
            
            # Get density scores (add small constant to avoid log(0))
            densities = hist[np.minimum(bin_indices-1, len(hist)-1)] + self.alpha
            scores += -np.log(densities)
        
        # Normalize scores
        scores = (scores - np.mean(scores)) / np.std(scores)
        
        # Classify points with high scores as anomalies
        threshold = norm.ppf(0.95)  # 95th percentile
        return (scores > threshold).astype(int)

# Example usage
np.random.seed(42)
X_normal = np.random.normal(0, 1, (500, 3))
X_anomaly = np.random.normal(3, 1, (50, 3))
X = np.vstack([X_normal, X_anomaly])

detector = HBOS()
anomalies = detector.fit_predict(X)
```

Slide 13: t-SNE Based Anomaly Detection

This method combines t-SNE dimensionality reduction with density-based anomaly detection to identify outliers in high-dimensional data by analyzing their local structure in the reduced space.

```python
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import numpy as np

class TSNEAnomalyDetector:
    def __init__(self, n_neighbors=5, threshold_factor=2):
        self.n_neighbors = n_neighbors
        self.threshold_factor = threshold_factor
        self.tsne = TSNE(n_components=2, random_state=42)
        
    def fit_predict(self, X):
        # Reduce dimensionality using t-SNE
        X_embedded = self.tsne.fit_transform(X)
        
        # Calculate local density using k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors)
        nbrs.fit(X_embedded)
        distances, _ = nbrs.kneighbors(X_embedded)
        
        # Average distance to k nearest neighbors
        avg_distances = np.mean(distances, axis=1)
        
        # Set threshold based on distance distribution
        threshold = (np.mean(avg_distances) + 
                    self.threshold_factor * np.std(avg_distances))
        
        return (avg_distances > threshold).astype(int)

# Generate example high-dimensional data
np.random.seed(42)
X_normal = np.random.normal(0, 1, (300, 20))
X_anomaly = np.random.normal(3, 1, (30, 20))
X = np.vstack([X_normal, X_anomaly])

# Detect anomalies
detector = TSNEAnomalyDetector()
anomalies = detector.fit_predict(X)
```

Slide 14: Additional Resources

*   Isolation Forest Paper: [https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
*   Deep Learning for Anomaly Detection Survey: [https://arxiv.org/abs/2007.02500](https://arxiv.org/abs/2007.02500)
*   LSTM-based Anomaly Detection: [https://arxiv.org/abs/1607.00148](https://arxiv.org/abs/1607.00148)
*   Comprehensive Survey on Anomaly Detection: [https://arxiv.org/abs/2103.02514](https://arxiv.org/abs/2103.02514)
*   Robust Random Cut Forest Paper: [https://www.kdd.org/kdd2016/papers/files/rpp0868-guhaA.pdf](https://www.kdd.org/kdd2016/papers/files/rpp0868-guhaA.pdf)
*   Advances in Outlier Detection Methods: [https://www.sciencedirect.com/science/article/pii/S0031320319303516](https://www.sciencedirect.com/science/article/pii/S0031320319303516)

