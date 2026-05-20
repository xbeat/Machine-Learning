## Visualizing Hyperplanes in Support Vector Machines
Slide 1: Understanding 1D Hyperplanes

In one-dimensional space, a hyperplane is simply a point that divides the line into two regions. This fundamental concept forms the basis for understanding higher-dimensional hyperplanes in Support Vector Machines, where the goal is to find the optimal separating point between classes.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate 1D sample data
class_1 = np.random.normal(loc=2, scale=0.5, size=10)
class_2 = np.random.normal(loc=4, scale=0.5, size=10)

# Find optimal hyperplane (point) between classes
hyperplane = (np.mean(class_1) + np.mean(class_2)) / 2

# Visualize
plt.figure(figsize=(10, 2))
plt.scatter(class_1, np.zeros_like(class_1), c='blue', label='Class 1')
plt.scatter(class_2, np.zeros_like(class_2), c='red', label='Class 2')
plt.axvline(x=hyperplane, color='green', linestyle='--', label='Hyperplane')
plt.legend()
plt.show()
```

Slide 2: 2D Hyperplane Mathematics

The mathematical foundation of 2D hyperplanes involves finding a line that maximally separates two classes. The hyperplane is defined by the equation wTx+b\=0w^Tx + b = 0wTx+b\=0, where w is the normal vector to the hyperplane and b is the bias term.

```python
# Mathematical representation of 2D hyperplane
def hyperplane_2d(x, w, b):
    """
    x: input vector [x1, x2]
    w: weight vector [w1, w2]
    b: bias term
    Returns: w1*x1 + w2*x2 + b
    """
    return np.dot(w, x) + b

# Example weights and bias
w = np.array([1, -1])
b = 0

# Test point
x = np.array([2, 2])
decision = hyperplane_2d(x, w, b)
print(f"Point {x} is classified as: {np.sign(decision)}")
```

Slide 3: Implementing 2D SVM Hyperplane Visualization

Support Vector Machines in 2D space require visualization of the decision boundary and support vectors. This implementation demonstrates how to create a complete visualization system for 2D hyperplanes with margin boundaries.

```python
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# Generate 2D dataset
np.random.seed(42)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
y = np.array([0] * 20 + [1] * 20)

# Train SVM
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

# Create grid for visualization
def plot_hyperplane():
    plt.figure(figsize=(10, 8))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=100, linewidth=1, facecolors='none', edgecolors='k')
    plt.show()

plot_hyperplane()
```

Slide 4: Hyperplane Margin Optimization

The concept of margin optimization is crucial in SVM hyperplane selection. The margin is the distance between the hyperplane and the nearest data point from either class, and maximizing this margin leads to better generalization performance.

```python
def calculate_margin(X, y, w, b):
    """
    Calculate margin for given hyperplane parameters
    """
    # Normalize weight vector
    w_norm = np.linalg.norm(w)
    w_normalized = w / w_norm
    
    # Calculate distances to hyperplane
    distances = []
    for i in range(len(X)):
        point = X[i]
        # Distance formula: |wx + b| / ||w||
        distance = abs(np.dot(w, point) + b) / w_norm
        distances.append(distance)
    
    return min(distances)

# Example usage
w = np.array([1, 1])
b = -4
X = np.array([[1, 1], [2, 2], [-1, -1]])
y = np.array([1, 1, -1])

margin = calculate_margin(X, y, w, b)
print(f"Margin size: {margin:.4f}")
```

Slide 5: 3D Hyperplane Implementation

In three-dimensional space, hyperplanes become actual planes separating data points. This implementation shows how to create and visualize 3D hyperplanes using mathematical principles and modern visualization tools.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_3d_hyperplane():
    # Generate 3D points
    np.random.seed(42)
    class1 = np.random.normal(loc=[1, 1, 1], scale=0.2, size=(20, 3))
    class2 = np.random.normal(loc=[-1, -1, -1], scale=0.2, size=(20, 3))
    
    # Create hyperplane
    xx, yy = np.meshgrid(range(-2, 3), range(-2, 3))
    z = lambda x, y: (-w[0] * x - w[1] * y - b) / w[2]
    
    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(class1[:, 0], class1[:, 1], class1[:, 2], c='b', label='Class 1')
    ax.scatter(class2[:, 0], class2[:, 1], class2[:, 2], c='r', label='Class 2')
    ax.plot_surface(xx, yy, z(xx, yy), alpha=0.3)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.legend()
    plt.show()

# Hyperplane parameters
w = np.array([1, 1, 1])
b = 0
create_3d_hyperplane()
```

Slide 6: Kernel Trick Implementation

The kernel trick allows SVM to handle non-linearly separable data by transforming the input space into a higher-dimensional feature space where a linear hyperplane can separate the classes effectively. This implementation demonstrates the RBF kernel transformation.

```python
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def rbf_kernel(X1, X2, gamma=1):
    """
    Implementation of RBF kernel transformation
    K(x,y) = exp(-gamma ||x-y||^2)
    """
    dist_matrix = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dist_matrix)

# Generate non-linearly separable data
np.random.seed(42)
X = np.r_[np.random.randn(20, 2) * 0.8 + [2, 2],
          np.random.randn(20, 2) * 0.8 + [-2, -2],
          np.random.randn(20, 2) * 0.8 + [2, -2]]
y = np.array([0] * 20 + [1] * 20 + [2] * 20)

# Apply kernel and train SVM
clf = SVC(kernel='rbf', gamma=0.5)
clf.fit(X, y)

# Visualize decision boundary
def plot_decision_boundary():
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.show()

plot_decision_boundary()
```

Slide 7: Support Vector Selection Algorithm

The process of selecting support vectors is crucial for defining the optimal hyperplane. This implementation demonstrates how to identify and visualize support vectors using both hard and soft margin approaches.

```python
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

def identify_support_vectors(X, y, C=1.0, tolerance=1e-4):
    """
    Custom implementation of support vector identification
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train linear SVM
    svm = LinearSVC(C=C, dual=True)
    svm.fit(X_scaled, y)
    
    # Calculate decision function values
    decision_vals = np.abs(svm.decision_function(X_scaled))
    
    # Identify support vectors (points close to the margin)
    support_vector_indices = np.where(decision_vals <= 1 + tolerance)[0]
    
    return support_vector_indices, svm.coef_[0], svm.intercept_[0]

# Generate sample data
np.random.seed(42)
X = np.r_[np.random.randn(50, 2) - [2, 2], np.random.randn(50, 2) + [2, 2]]
y = np.array([-1] * 50 + [1] * 50)

# Find support vectors
sv_indices, weights, bias = identify_support_vectors(X, y)

# Visualize
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6)
plt.scatter(X[sv_indices, 0], X[sv_indices, 1], 
           facecolors='none', edgecolors='r', s=100, 
           label='Support Vectors')
plt.legend()
plt.show()

print(f"Number of support vectors: {len(sv_indices)}")
print(f"Weight vector: {weights}")
print(f"Bias term: {bias}")
```

Slide 8: Hyperplane Optimization with SMO Algorithm

Sequential Minimal Optimization (SMO) is the key algorithm for finding the optimal hyperplane in SVMs. This implementation shows the core components of SMO, including the selection of Lagrange multipliers and updating of decision boundaries.

```python
class SMOOptimizer:
    def __init__(self, X, y, C=1.0, tol=1e-3, max_passes=5):
        self.X = X
        self.y = y
        self.C = C
        self.tol = tol
        self.m = X.shape[0]
        self.alphas = np.zeros(self.m)
        self.b = 0
        self.max_passes = max_passes
        
    def compute_kernel(self, x1, x2):
        return np.dot(x1, x2)
    
    def optimize(self):
        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(self.m):
                Ei = self._compute_error(i)
                if ((self.y[i] * Ei < -self.tol and self.alphas[i] < self.C) or
                    (self.y[i] * Ei > self.tol and self.alphas[i] > 0)):
                    j = np.random.randint(0, self.m)
                    while j == i:
                        j = np.random.randint(0, self.m)
                    
                    Ej = self._compute_error(j)
                    
                    # Store old alphas
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    # Compute bounds
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = (2 * self.compute_kernel(self.X[i], self.X[j]) -
                          self.compute_kernel(self.X[i], self.X[i]) -
                          self.compute_kernel(self.X[j], self.X[j]))
                    
                    if eta >= 0:
                        continue
                    
                    # Update alpha j
                    self.alphas[j] = alpha_j_old - (self.y[j] * (Ei - Ej)) / eta
                    self.alphas[j] = min(H, max(L, self.alphas[j]))
                    
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha i
                    self.alphas[i] = alpha_i_old + self.y[i] * self.y[j] * \
                                   (alpha_j_old - self.alphas[j])
                    
                    # Update threshold
                    b1 = self.b - Ei - self.y[i] * (self.alphas[i] - alpha_i_old) * \
                         self.compute_kernel(self.X[i], self.X[i]) - \
                         self.y[j] * (self.alphas[j] - alpha_j_old) * \
                         self.compute_kernel(self.X[i], self.X[j])
                    
                    b2 = self.b - Ej - self.y[i] * (self.alphas[i] - alpha_i_old) * \
                         self.compute_kernel(self.X[i], self.X[j]) - \
                         self.y[j] * (self.alphas[j] - alpha_j_old) * \
                         self.compute_kernel(self.X[j], self.X[j])
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        
        return self.alphas, self.b
    
    def _compute_error(self, i):
        return self._decision_function(self.X[i]) - self.y[i]
    
    def _decision_function(self, x):
        return np.sum(self.alphas * self.y * 
                     np.array([self.compute_kernel(x_i, x) 
                              for x_i in self.X])) + self.b

# Example usage
X_sample = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
y_sample = np.array([1, -1, -1, 1])

optimizer = SMOOptimizer(X_sample, y_sample)
alphas, b = optimizer.optimize()
print("Optimal alphas:", alphas)
print("Optimal b:", b)
```

Slide 9: Real-world Application - Text Classification with Hyperplanes

Text classification represents a high-dimensional hyperplane problem where documents are separated into categories. This implementation demonstrates document classification using TF-IDF features and linear SVM hyperplanes.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Sample dataset
documents = [
    "machine learning algorithms optimize performance",
    "deep neural networks process data",
    "stock market analysis prediction",
    "financial trading strategies",
    "gradient descent optimization technique",
    "market prediction using neural nets"
]

labels = [0, 0, 1, 1, 0, 1]  # 0: Technical, 1: Financial

# Transform text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Train SVM classifier
clf = LinearSVC(dual=False)
clf.fit(X, labels)

# Analysis of feature importance
feature_weights = pd.DataFrame(
    clf.coef_[0],
    index=vectorizer.get_feature_names_out(),
    columns=['weight']
).sort_values('weight', ascending=False)

print("Most important terms for classification:")
print(feature_weights.head(10))

# Predict new document
new_doc = ["neural network stock prediction algorithm"]
new_doc_features = vectorizer.transform(new_doc)
prediction = clf.predict(new_doc_features)
print(f"\nPrediction for new document: {'Financial' if prediction[0] == 1 else 'Technical'}")
```

Slide 10: Hyperplane Margin Visualization in High Dimensions

Visualizing margins in high-dimensional spaces requires dimensionality reduction techniques. This implementation uses PCA to project high-dimensional hyperplanes onto 2D space while preserving margin relationships.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def visualize_high_dim_margins():
    # Generate high-dimensional data
    np.random.seed(42)
    n_features = 10
    n_samples = 200
    
    X = np.random.randn(n_samples, n_features)
    # Create two clusters in high-dimensional space
    X[:100] = X[:100] + 2
    y = np.array([1]*100 + [-1]*100)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train SVM
    svm = LinearSVC(dual=False)
    svm.fit(X_scaled, y)
    
    # Project to 2D using PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    
    # Project hyperplane normal vector
    w = svm.coef_[0]
    w_2d = pca.transform(w.reshape(1, -1))[0]
    
    # Visualize
    plt.figure(figsize=(12, 8))
    plt.scatter(X_2d[y==1, 0], X_2d[y==1, 1], c='b', label='Class 1')
    plt.scatter(X_2d[y==-1, 0], X_2d[y==-1, 1], c='r', label='Class 2')
    
    # Plot decision boundary
    plt.quiver(0, 0, w_2d[0], w_2d[1], angles='xy', scale_units='xy', scale=0.5)
    
    plt.title('2D Projection of High-Dimensional Hyperplane')
    plt.legend()
    plt.show()
    
    return pca.explained_variance_ratio_

variance_explained = visualize_high_dim_margins()
print(f"Variance explained by first two components: {variance_explained.sum():.2%}")
```

Slide 11: Implementing Multi-class Hyperplanes

Multi-class classification requires multiple hyperplanes using either one-vs-rest or one-vs-one strategies. This implementation demonstrates both approaches with performance comparison.

```python
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score
import numpy as np

class MulticlassHyperplaneComparison:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def train_and_compare(self):
        # One-vs-Rest Strategy
        ovr_classifier = OneVsRestClassifier(SVC(kernel='linear'))
        ovr_classifier.fit(self.X, self.y)
        
        # One-vs-One Strategy
        ovo_classifier = OneVsOneClassifier(SVC(kernel='linear'))
        ovo_classifier.fit(self.X, self.y)
        
        # Generate predictions
        ovr_pred = ovr_classifier.predict(self.X)
        ovo_pred = ovo_classifier.predict(self.X)
        
        # Compare results
        results = {
            'ovr_accuracy': accuracy_score(self.y, ovr_pred),
            'ovo_accuracy': accuracy_score(self.y, ovo_pred),
            'ovr_n_classifiers': len(ovr_classifier.estimators_),
            'ovo_n_classifiers': len(ovo_classifier.estimators_)
        }
        
        return results, ovr_classifier, ovo_classifier

# Generate multi-class data
np.random.seed(42)
X = np.r_[np.random.randn(50, 2) + [2, 2],
          np.random.randn(50, 2) + [-2, -2],
          np.random.randn(50, 2) + [2, -2]]
y = np.array([0] * 50 + [1] * 50 + [2] * 50)

# Compare strategies
comparison = MulticlassHyperplaneComparison(X, y)
results, ovr_clf, ovo_clf = comparison.train_and_compare()

print("Performance Comparison:")
print(f"One-vs-Rest Accuracy: {results['ovr_accuracy']:.4f}")
print(f"One-vs-One Accuracy: {results['ovo_accuracy']:.4f}")
print(f"Number of OvR classifiers: {results['ovr_n_classifiers']}")
print(f"Number of OvO classifiers: {results['ovo_n_classifiers']}")
```

Slide 12: Real-world Application - Image Classification Hyperplanes

Image classification using SVM hyperplanes requires effective feature extraction and dimensionality handling. This implementation demonstrates a complete pipeline for image classification using HOG features and linear SVM.

```python
from sklearn.svm import LinearSVC
from skimage.feature import hog
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

class ImageSVMClassifier:
    def __init__(self, image_size=(64, 64)):
        self.image_size = image_size
        self.clf = LinearSVC(dual=False)
        
    def extract_features(self, image):
        """Extract HOG features from image"""
        # Resize image
        resized = resize(image, self.image_size)
        # Extract HOG features
        features = hog(resized, orientations=9,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2))
        return features
    
    def prepare_dataset(self, images, labels):
        """Extract features from multiple images"""
        features_list = []
        for image in images:
            features = self.extract_features(image)
            features_list.append(features)
        return np.array(features_list), labels
    
    def train(self, X_train, y_train):
        """Train SVM classifier"""
        self.clf.fit(X_train, y_train)
        
    def predict(self, X):
        """Predict class labels"""
        return self.clf.predict(X)

# Example usage with sample data
def generate_sample_images(n_samples=100):
    """Generate synthetic image data for demonstration"""
    images = []
    labels = []
    
    for i in range(n_samples):
        # Generate simple patterns
        if i < n_samples/2:
            # Class 0: Diagonal pattern
            img = np.eye(64)
            labels.append(0)
        else:
            # Class 1: Reverse diagonal pattern
            img = np.fliplr(np.eye(64))
            labels.append(1)
        images.append(img)
    
    return np.array(images), np.array(labels)

# Create and train classifier
images, labels = generate_sample_images()
classifier = ImageSVMClassifier()

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels)

# Prepare features and train
X_train_features, y_train = classifier.prepare_dataset(X_train, y_train)
X_test_features, y_test = classifier.prepare_dataset(X_test, y_test)
classifier.train(X_train_features, y_train)

# Evaluate
predictions = classifier.predict(X_test_features)
accuracy = np.mean(predictions == y_test)
print(f"Classification accuracy: {accuracy:.4f}")

# Visualize decision boundary weights
decision_weights = classifier.clf.coef_[0]
weight_img = decision_weights.reshape((8, 8))  # Reshape to visualize HOG features
plt.imshow(weight_img, cmap='seismic')
plt.colorbar()
plt.title('SVM Decision Weights')
plt.show()
```

Slide 13: Implementing Custom Kernels for Non-linear Hyperplanes

Creating custom kernels allows for specialized feature transformations tailored to specific problem domains. This implementation shows how to define and use custom kernels in SVM classification.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist

class CustomKernelSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel_func=None, C=1.0):
        self.kernel_func = kernel_func if kernel_func else self.default_kernel
        self.C = C
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = 0
        
    def default_kernel(self, X1, X2):
        """Custom polynomial mixture kernel"""
        linear = np.dot(X1, X2.T)
        poly = (1 + linear) ** 2
        rbf = np.exp(-np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2) / 2)
        return 0.5 * poly + 0.5 * rbf
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        # Compute kernel matrix
        K = self.kernel_func(X, X)
        
        # Solve quadratic programming problem
        from scipy.optimize import minimize
        
        def objective(alphas):
            return 0.5 * np.dot(alphas, np.dot(K * (y[:, None] * y[None, :]), alphas)) - np.sum(alphas)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.dot(x, y)},
            {'type': 'ineq', 'fun': lambda x: x},
            {'type': 'ineq', 'fun': lambda x: self.C - x}
        ]
        
        solution = minimize(objective, np.zeros(n_samples), constraints=constraints)
        self.alphas = solution.x
        
        # Find support vectors
        sv_mask = self.alphas > 1e-5
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]
        self.alphas = self.alphas[sv_mask]
        
        # Calculate bias term
        self.b = np.mean(y[sv_mask] - np.sum(self.alphas * self.support_vector_labels *
                        self.kernel_func(self.support_vectors, X[sv_mask]), axis=0))
        
        return self
    
    def decision_function(self, X):
        return np.sum(self.alphas * self.support_vector_labels *
                     self.kernel_func(self.support_vectors, X), axis=0) + self.b
    
    def predict(self, X):
        return np.sign(self.decision_function(X))

# Example usage with custom kernel
def custom_spectral_kernel(X1, X2, gamma=0.1):
    """Spectral mixture kernel"""
    dist_matrix = cdist(X1, X2, metric='euclidean')
    return np.cos(2 * np.pi * dist_matrix) * np.exp(-gamma * dist_matrix**2)

# Generate spiral dataset
def generate_spiral_data(n_samples=100):
    theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    X = np.vstack((data_a, data_b))
    y = np.array([1] * n_samples + [-1] * n_samples)
    return X, y

# Train and evaluate
X, y = generate_spiral_data()
svm = CustomKernelSVM(kernel_func=custom_spectral_kernel)
svm.fit(X, y)

# Visualize results
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                     np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.title('Custom Kernel SVM Decision Boundary')
plt.show()
```

Slide 14: Additional Resources

*   "Large Margin Classification Using the Perceptron Algorithm" - [https://arxiv.org/abs/cs/9809027](https://arxiv.org/abs/cs/9809027)
*   "Training Support Vector Machines: an Application to Face Detection" - [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640)
*   "Making Large-Scale SVM Learning Practical" - [https://arxiv.org/abs/1011.1669v3](https://arxiv.org/abs/1011.1669v3)
*   "A Tutorial on Support Vector Machines for Pattern Recognition" - [https://arxiv.org/abs/1012.3335](https://arxiv.org/abs/1012.3335)
*   "Optimal Hyperplane Algorithm for Pattern Separation by Linear Decision Functions" - [https://arxiv.org/abs/1302.5345](https://arxiv.org/abs/1302.5345)

Slide 15: Performance Optimization for High-Dimensional Hyperplanes

Understanding and implementing performance optimizations for high-dimensional hyperplane calculations is crucial for real-world applications. This implementation demonstrates advanced techniques for efficient hyperplane computation.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from numba import jit
import time

class OptimizedHighDimSVM:
    def __init__(self, max_iter=1000, batch_size=128):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.w = None
        self.b = 0
        
    @staticmethod
    @jit(nopython=True)
    def _compute_gradient(X_batch, y_batch, w, b, C):
        """Optimized gradient computation using Numba"""
        m = X_batch.shape[0]
        dw = np.zeros_like(w)
        db = 0
        
        for i in range(m):
            margin = y_batch[i] * (np.dot(X_batch[i], w) + b)
            if margin < 1:
                dw += -C * y_batch[i] * X_batch[i]
                db += -C * y_batch[i]
                
        dw = dw/m + w  # Add regularization term
        db = db/m
        return dw, db
    
    def fit(self, X, y, C=1.0, learning_rate=0.001):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Training loop with mini-batches
        for epoch in range(self.max_iter):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            
            for start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # Compute gradients
                dw, db = self._compute_gradient(X_batch, y_batch, self.w, self.b, C)
                
                # Update parameters
                self.w -= learning_rate * dw
                self.b -= learning_rate * db
                
                # Compute loss for monitoring
                margins = y_batch * (np.dot(X_batch, self.w) + self.b)
                batch_loss = np.mean(np.maximum(0, 1 - margins)) + 0.5 * np.sum(self.w ** 2)
                epoch_loss += batch_loss
                
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

# Performance comparison
def benchmark_performance():
    # Generate high-dimensional data
    n_samples = 10000
    n_features = 1000
    np.random.seed(42)
    
    X = np.random.randn(n_samples, n_features)
    y = np.sign(np.sum(X[:, :10], axis=1))  # Only first 10 features are relevant
    
    # Train optimized SVM
    start_time = time.time()
    opt_svm = OptimizedHighDimSVM(max_iter=500)
    opt_svm.fit(X, y)
    opt_time = time.time() - start_time
    
    # Train standard SVM
    from sklearn.svm import LinearSVC
    start_time = time.time()
    std_svm = LinearSVC(dual=False)
    std_svm.fit(X, y)
    std_time = time.time() - start_time
    
    return {
        'optimized_time': opt_time,
        'standard_time': std_time,
        'opt_accuracy': np.mean(opt_svm.predict(X) == y),
        'std_accuracy': np.mean(std_svm.predict(X) == y)
    }

results = benchmark_performance()
print("\nPerformance Comparison:")
print(f"Optimized SVM Time: {results['optimized_time']:.2f} seconds")
print(f"Standard SVM Time: {results['standard_time']:.2f} seconds")
print(f"Optimized SVM Accuracy: {results['opt_accuracy']:.4f}")
print(f"Standard SVM Accuracy: {results['std_accuracy']:.4f}")
```

Slide 16: Results Analysis and Visualization

In this final implementation, we analyze the performance metrics and create comprehensive visualizations of hyperplane decisions across different dimensionalities and kernel choices.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score

class HyperplaneAnalyzer:
    def __init__(self):
        self.performance_metrics = {}
        
    def analyze_hyperplane_performance(self, X, y, classifiers):
        """Analyze performance of different hyperplane configurations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        for name, clf in classifiers.items():
            # Cross-validation scores
            cv_scores = cross_val_score(clf, X, y, cv=5)
            self.performance_metrics[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Fit classifier for additional metrics
            clf.fit(X, y)
            y_pred = clf.predict(X)
            y_score = clf.decision_function(X)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            
            # Plot ROC curve
            axes[0, 0].plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1])
            axes[0, 1].set_title(f'Confusion Matrix - {name}')
            
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_title('ROC Curves')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].legend()
        
        # Plot cross-validation scores
        cv_means = [metrics['cv_mean'] for metrics in self.performance_metrics.values()]
        cv_stds = [metrics['cv_std'] for metrics in self.performance_metrics.values()]
        
        axes[1, 0].bar(self.performance_metrics.keys(), cv_means, yerr=cv_stds)
        axes[1, 0].set_title('Cross-validation Scores')
        axes[1, 0].set_ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return self.performance_metrics

# Example usage
from sklearn.svm import SVC

# Generate dataset
X, y = np.random.randn(1000, 2), np.random.randint(0, 2, 1000)

# Define different classifiers
classifiers = {
    'Linear SVM': LinearSVC(dual=False),
    'RBF SVM': SVC(kernel='rbf'),
    'Polynomial SVM': SVC(kernel='poly', degree=3),
    'Custom Kernel': CustomKernelSVM()
}

# Analyze performance
analyzer = HyperplaneAnalyzer()
metrics = analyzer.analyze_hyperplane_performance(X, y, classifiers)

# Print detailed metrics
for name, perf in metrics.items():
    print(f"\n{name} Performance:")
    print(f"Mean CV Score: {perf['cv_mean']:.4f} ± {perf['cv_std']:.4f}")

plt.show()
```

That concludes the complete presentation on hyperplanes in SVM. The implementations cover the fundamental concepts, advanced optimizations, and practical applications while maintaining a focus on both theoretical understanding and practical implementation.

