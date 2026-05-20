## PCA Basics in Machine Learning with Python
Slide 1: Introduction to PCA in Machine Learning

Principal Component Analysis (PCA) is a fundamental technique in machine learning for dimensionality reduction and data visualization. It helps identify patterns in high-dimensional data by transforming it into a new coordinate system of uncorrelated variables called principal components.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
X[:, 1] = 3 * X[:, 0] + np.random.randn(100) * 0.5

# Plot original data
plt.scatter(X[:, 0], X[:, 1])
plt.title('Original Data')
plt.show()

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Plot transformed data
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('PCA Transformed Data')
plt.show()
```

Slide 2: The Concept of Principal Components

Principal components are the directions in the feature space along which the data varies the most. They are orthogonal to each other and ordered by the amount of variance they explain in the data.

```python
# Compute and plot principal components
pca = PCA()
pca.fit(X)

plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * var  # scale component by its variance explanation
    plt.arrow(pca.mean_[0], pca.mean_[1], comp[0], comp[1], 
              color=f'C{i+2}', width=0.05, head_width=0.2)
    plt.text(pca.mean_[0] + comp[0], pca.mean_[1] + comp[1], f'PC{i+1}')

plt.title('Principal Components')
plt.axis('equal')
plt.show()
```

Slide 3: Variance Explained by Principal Components

The variance explained by each principal component indicates its importance in representing the original data. This information helps in deciding how many components to retain.

```python
# Calculate and plot explained variance ratio
pca = PCA()
pca.fit(X)

plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
        pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Component')
plt.show()

print("Cumulative explained variance ratio:")
print(np.cumsum(pca.explained_variance_ratio_))
```

Slide 4: Dimensionality Reduction with PCA

PCA can be used to reduce the dimensionality of data by selecting a subset of principal components that explain most of the variance in the data.

```python
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot reduced data
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.colorbar(scatter)
plt.title('Digits Dataset Reduced to 2 Dimensions')
plt.show()

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_pca.shape}")
```

Slide 5: Choosing the Number of Components

Selecting the right number of components is crucial. A common approach is to choose enough components to explain a certain percentage of the total variance.

```python
# Compute cumulative explained variance ratio
pca = PCA()
pca.fit(X)
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative explained variance ratio
plt.plot(range(1, len(cumulative_variance_ratio) + 1), 
         cumulative_variance_ratio, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance vs. Number of Components')
plt.show()

# Find number of components for 95% variance explained
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of components for 95% variance explained: {n_components}")
```

Slide 6: PCA for Data Visualization

PCA is often used to visualize high-dimensional data in 2D or 3D space, making it easier to identify patterns and clusters.

```python
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis')
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
ax.set_zlabel('Third Principal Component')
plt.colorbar(scatter)
plt.title('Iris Dataset in 3D PCA Space')
plt.show()
```

Slide 7: PCA for Noise Reduction

PCA can be used to reduce noise in data by reconstructing it using only the top principal components, effectively filtering out less important variations.

```python
# Generate noisy sine wave
t = np.linspace(0, 10, 1000)
x = np.sin(t)
x_noisy = x + 0.5 * np.random.randn(1000)

# Apply PCA for denoising
X = x_noisy.reshape(-1, 1)
pca = PCA(n_components=1)
X_denoised = pca.inverse_transform(pca.fit_transform(X))

# Plot results
plt.figure(figsize=(12, 4))
plt.plot(t, x, label='Original')
plt.plot(t, x_noisy, label='Noisy')
plt.plot(t, X_denoised, label='Denoised')
plt.legend()
plt.title('PCA for Noise Reduction')
plt.show()
```

Slide 8: PCA for Feature Extraction

PCA can be used to extract new features that capture the most important aspects of the data. These new features can then be used for further analysis or as input to other machine learning algorithms.

```python
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load face dataset
faces = fetch_olivetti_faces()
X, y = faces.data, faces.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PCA
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train SVM classifier
svm = SVC()
svm.fit(X_train_pca, y_train)

# Predict and calculate accuracy
y_pred = svm.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy using PCA features: {accuracy:.2f}")
```

Slide 9: PCA and Correlation

PCA works by finding directions of maximum variance, which often correspond to directions of high correlation in the original features. Understanding this relationship can help interpret PCA results.

```python
import seaborn as sns

# Generate correlated data
np.random.seed(42)
x = np.random.randn(100)
y = 2*x + np.random.randn(100)*0.5
z = 3*x - 2*y + np.random.randn(100)*0.1
data = np.column_stack((x, y, z))

# Compute correlation matrix
corr_matrix = np.corrcoef(data.T)

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Apply PCA
pca = PCA()
pca.fit(data)

# Print explained variance ratio
print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
```

Slide 10: PCA and Standardization

Standardization of features is often necessary before applying PCA, especially when features are on different scales. This ensures that PCA captures true variance rather than artificial variance due to different scales.

```python
from sklearn.preprocessing import StandardScaler

# Generate data with different scales
np.random.seed(42)
X = np.column_stack((np.random.randn(100)*10, np.random.randn(100)*0.1))

# Apply PCA without standardization
pca_no_scale = PCA()
X_pca_no_scale = pca_no_scale.fit_transform(X)

# Apply PCA with standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca_scale = PCA()
X_pca_scale = pca_scale.fit_transform(X_scaled)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(X_pca_no_scale[:, 0], X_pca_no_scale[:, 1])
ax1.set_title('PCA without Standardization')
ax2.scatter(X_pca_scale[:, 0], X_pca_scale[:, 1])
ax2.set_title('PCA with Standardization')
plt.show()

print("Explained variance ratio without standardization:")
print(pca_no_scale.explained_variance_ratio_)
print("\nExplained variance ratio with standardization:")
print(pca_scale.explained_variance_ratio_)
```

Slide 11: Incremental PCA

For large datasets that don't fit in memory, Scikit-learn provides Incremental PCA, which can process data in batches.

```python
from sklearn.decomposition import IncrementalPCA

# Generate large dataset
np.random.seed(42)
X_large = np.random.randn(10000, 100)

# Apply Incremental PCA
batch_size = 500
ipca = IncrementalPCA(n_components=10, batch_size=batch_size)

# Process data in batches
for i in range(0, X_large.shape[0], batch_size):
    ipca.partial_fit(X_large[i:i+batch_size])

# Transform data
X_ipca = ipca.transform(X_large)

print(f"Original shape: {X_large.shape}")
print(f"Reduced shape: {X_ipca.shape}")
print("\nExplained variance ratio:")
print(ipca.explained_variance_ratio_)
```

Slide 12: Real-life Example: Image Compression

PCA can be used for image compression by reducing the dimensionality of image data. This example demonstrates how PCA can compress and reconstruct a grayscale image.

```python
from sklearn.datasets import load_sample_image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load and convert image to grayscale
image = load_sample_image("china.jpg")
gray_image = np.mean(image, axis=2).astype(np.float64)

# Reshape image to 2D array
X = gray_image.reshape(-1, gray_image.shape[1])

# Apply PCA with different number of components
n_components_list = [5, 20, 50, 100]
fig, axes = plt.subplots(1, len(n_components_list) + 1, figsize=(20, 4))

axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

for i, n_components in enumerate(n_components_list, 1):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)
    
    image_reconstructed = X_reconstructed.reshape(gray_image.shape)
    
    axes[i].imshow(image_reconstructed, cmap='gray')
    axes[i].set_title(f'{n_components} components')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

Slide 13: Real-life Example: Anomaly Detection

PCA can be used for anomaly detection by identifying data points that don't fit well in the lower-dimensional PCA space. This example demonstrates how to use PCA for detecting anomalies in a dataset of sensor readings.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generate normal data and anomalies
np.random.seed(42)
n_samples = 1000
n_features = 10

# Normal data
X_normal = np.random.randn(n_samples, n_features)

# Anomalies
n_anomalies = 50
X_anomalies = np.random.randn(n_anomalies, n_features) * 2 + 5

# Combine normal data and anomalies
X = np.vstack((X_normal, X_anomalies))

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Calculate reconstruction error
X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)

# Set threshold for anomaly detection (e.g., 95th percentile)
threshold = np.percentile(reconstruction_error, 95)

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(X_pca[:n_samples, 0], X_pca[:n_samples, 1], c='blue', label='Normal')
plt.scatter(X_pca[n_samples:, 0], X_pca[n_samples:, 1], c='red', label='Anomaly')
plt.scatter(X_pca[reconstruction_error > threshold, 0], 
            X_pca[reconstruction_error > threshold, 1], 
            c='green', s=100, alpha=0.5, label='Detected Anomaly')
plt.legend()
plt.title('PCA for Anomaly Detection')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

print(f"Number of detected anomalies: {np.sum(reconstruction_error > threshold)}")
```

Slide 14: Additional Resources

For a deeper understanding of PCA and its applications in machine learning, consider exploring these resources:

1. "A Tutorial on Principal Component Analysis" by Jonathon Shlens ArXiv URL: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100) This paper provides a comprehensive introduction to PCA, covering its mathematical foundations and practical applications.
2. "Principal Component Analysis: A Review and Recent Developments" by Ian T. Jolliffe and Jorge Cadima ArXiv URL: [https://arxiv.org/abs/1511.03677](https://arxiv.org/abs/1511.03677) This review article discusses recent advancements in PCA, including robust and sparse variations of the technique.
3. Scikit-learn Documentation on PCA This official documentation offers practical examples and detailed explanations of PCA implementation in scikit-learn.
4. "Pattern Recognition and Machine Learning" by Christopher M. Bishop This textbook includes a thorough treatment of PCA in the context of machine learning and statistical pattern recognition.
5. "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman This comprehensive book covers PCA and its relationships to other statistical learning techniques.

These resources range from introductory tutorials to advanced discussions, providing a well-rounded understanding of PCA and its role in machine learning and data analysis.

