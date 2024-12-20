## Mastering Support Vector Machines for Classification and Regression
Slide 1: SVM Mathematical Foundations

Support Vector Machines rely on fundamental mathematical principles to find the optimal hyperplane separating data classes. The primary objective is maximizing the margin between classes while minimizing classification errors through quadratic optimization with linear constraints.

```python
import numpy as np
from sklearn.datasets import make_classification

# Generate synthetic dataset
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                          n_informative=2, random_state=1, 
                          n_clusters_per_class=1)

# Mathematical formulation in LaTeX (not rendered)
"""
$$
\min_{w, b} \frac{1}{2} ||w||^2
$$
Subject to:
$$
y_i(w^T x_i + b) \geq 1, \forall i
$$
"""

# Implement basic SVM components
def compute_margin(X, y, w, b):
    return np.min(y * (np.dot(X, w) + b))

# Example usage
w = np.array([1, -1])
b = 0
margin = compute_margin(X, y, w, b)
print(f"Margin: {margin}")
```

Slide 2: Linear SVM Implementation from Scratch

Implementing a linear SVM classifier demonstrates the core concepts of margin maximization and support vector identification. This implementation uses gradient descent to optimize the SVM objective function without relying on external libraries.

```python
class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        # Initialize weights
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - 
                                       np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
                    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)
```

Slide 3: Kernel Functions Implementation

The kernel trick enables SVM to handle non-linear classification by mapping data into higher-dimensional spaces. This implementation showcases common kernel functions used in SVM algorithms for complex pattern recognition.

```python
class SVMKernels:
    @staticmethod
    def linear_kernel(x1, x2):
        return np.dot(x1, x2)
    
    @staticmethod
    def polynomial_kernel(x1, x2, degree=3):
        return (1 + np.dot(x1, x2)) ** degree
    
    @staticmethod
    def rbf_kernel(x1, x2, gamma=0.1):
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
    
    @staticmethod
    def sigmoid_kernel(x1, x2, gamma=0.1, c=1):
        return np.tanh(gamma * np.dot(x1, x2) + c)

# Example usage
x1 = np.array([1, 2])
x2 = np.array([3, 4])
kernels = SVMKernels()

print(f"Linear Kernel: {kernels.linear_kernel(x1, x2)}")
print(f"RBF Kernel: {kernels.rbf_kernel(x1, x2)}")
print(f"Polynomial Kernel: {kernels.polynomial_kernel(x1, x2)}")
```

Slide 4: Soft Margin SVM Implementation

The soft margin SVM allows for misclassifications through the introduction of slack variables, making it more practical for real-world applications where perfect separation is often impossible or undesirable.

```python
class SoftMarginSVM:
    def __init__(self, C=1.0, max_iter=1000):
        self.C = C  # Regularization parameter
        self.max_iter = max_iter
        self.w = None
        self.b = None
        
    def objective_function(self, X, y):
        n_samples = X.shape[0]
        margins = y * (np.dot(X, self.w) + self.b)
        # Hinge loss calculation
        hinge_loss = np.maximum(0, 1 - margins)
        # Objective function with regularization
        return (0.5 * np.dot(self.w, self.w) + 
                self.C * np.sum(hinge_loss) / n_samples)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.max_iter):
            margins = y * (np.dot(X, self.w) + self.b)
            misclassified = margins < 1
            
            grad_w = self.w - self.C * np.sum(
                y[misclassified].reshape(-1, 1) * X[misclassified], axis=0
            )
            grad_b = -self.C * np.sum(y[misclassified])
            
            self.w -= 0.01 * grad_w
            self.b -= 0.01 * grad_b

# Example usage
X, y = make_classification(n_samples=100, n_features=2, random_state=42)
svm = SoftMarginSVM(C=1.0)
svm.fit(X, y)
```

Slide 5: Real-world Text Classification with SVM

Implementing SVM for text classification requires careful preprocessing and feature extraction. This implementation demonstrates a complete pipeline for sentiment analysis using TF-IDF vectorization and linear SVM.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import numpy as np

# Sample dataset
texts = [
    "This product is amazing",
    "Terrible customer service",
    "Great experience overall",
    "Would not recommend"
]
labels = [1, 0, 1, 0]  # 1: positive, 0: negative

# Create text classification pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), 
                             max_features=1000)),
    ('clf', LinearSVC(C=1.0))
])

# Train and evaluate
text_clf.fit(texts, labels)
predictions = text_clf.predict(texts)

print(classification_report(labels, predictions))
```

Slide 6: Multi-class SVM Implementation

Support Vector Machines can handle multi-class classification through one-vs-rest or one-vs-one strategies. This implementation showcases the one-vs-rest approach with custom decision functions.

```python
class MultiClassSVM:
    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
        self.classifiers = {}
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Train one classifier per class
        for i in range(n_classes):
            # Create binary labels
            current_class = self.classes[i]
            binary_y = np.where(y == current_class, 1, -1)
            
            # Train binary classifier
            clf = LinearSVC(C=self.C)
            clf.fit(X, binary_y)
            self.classifiers[current_class] = clf
            
    def predict(self, X):
        # Get scores for each class
        scores = np.zeros((X.shape[0], len(self.classes)))
        for i, class_label in enumerate(self.classes):
            clf = self.classifiers[class_label]
            scores[:, i] = clf.decision_function(X)
        
        # Return class with highest score
        return self.classes[np.argmax(scores, axis=1)]

# Example usage
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

clf = MultiClassSVM()
clf.fit(X, y)
predictions = clf.predict(X)
```

Slide 7: SVM Hyperparameter Optimization

Optimizing SVM hyperparameters is crucial for model performance. This implementation uses Bayesian optimization to find optimal parameters for both kernel selection and regularization strength.

```python
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical

# Define search space
search_space = {
    'kernel': Categorical(['linear', 'rbf', 'poly']),
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'degree': Integer(1, 4)
}

# Create optimizer
opt = BayesSearchCV(
    SVC(),
    search_space,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=0
)

# Example optimization
X, y = make_classification(n_samples=1000, n_features=20)

opt.fit(X, y)
print(f"Best parameters: {opt.best_params_}")
print(f"Best cross-validation score: {opt.best_score_:.3f}")

# Validate on test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
best_model = opt.best_estimator_
score = best_model.score(X_test, y_test)
print(f"Test set score: {score:.3f}")
```

Slide 8: SVM for Time Series Classification

Implementing SVM for time series data requires specialized feature extraction and preprocessing. This implementation demonstrates dynamic time warping kernel with SVM for temporal pattern recognition.

```python
import numpy as np
from scipy.spatial.distance import cdist

class TimeSeriesSVM:
    def __init__(self, C=1.0, gamma=1.0):
        self.C = C
        self.gamma = gamma
        
    def dtw_kernel(self, x, y):
        def dtw_distance(s1, s2):
            n, m = len(s1), len(s2)
            dtw_matrix = np.inf * np.ones((n+1, m+1))
            dtw_matrix[0, 0] = 0
            
            for i in range(1, n+1):
                for j in range(1, m+1):
                    cost = abs(s1[i-1] - s2[j-1])
                    dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                                dtw_matrix[i, j-1],
                                                dtw_matrix[i-1, j-1])
            return dtw_matrix[n, m]
        
        return np.exp(-self.gamma * dtw_distance(x, y))
    
    def fit(self, X, y):
        n_samples = len(X)
        gram_matrix = np.zeros((n_samples, n_samples))
        
        # Compute Gram matrix
        for i in range(n_samples):
            for j in range(n_samples):
                gram_matrix[i,j] = self.dtw_kernel(X[i], X[j])
        
        # Train SVM with custom kernel
        self.svm = SVC(kernel='precomputed', C=self.C)
        self.svm.fit(gram_matrix, y)
        self.X_train = X
        
    def predict(self, X_test):
        n_train = len(self.X_train)
        n_test = len(X_test)
        K_test = np.zeros((n_test, n_train))
        
        for i in range(n_test):
            for j in range(n_train):
                K_test[i,j] = self.dtw_kernel(X_test[i], self.X_train[j])
                
        return self.svm.predict(K_test)
```

Slide 9: SVM for Image Recognition

Support Vector Machines can effectively handle image classification tasks through proper feature extraction and kernel selection. This implementation demonstrates a complete pipeline for image recognition using HOG features.

```python
from skimage.feature import hog
from skimage.transform import resize
import numpy as np
from sklearn.svm import SVC
import cv2

class ImageSVM:
    def __init__(self, image_size=(64, 64)):
        self.image_size = image_size
        self.svm = SVC(kernel='rbf', C=10.0, gamma='scale')
        
    def extract_features(self, image):
        # Resize image
        img_resized = resize(image, self.image_size)
        
        # Extract HOG features
        features = hog(img_resized, 
                      orientations=9,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      multichannel=True if len(image.shape) > 2 else False)
        return features
    
    def preprocess_images(self, images):
        return np.array([self.extract_features(img) for img in images])
    
    def fit(self, images, labels):
        X = self.preprocess_images(images)
        self.svm.fit(X, labels)
        
    def predict(self, images):
        X = self.preprocess_images(images)
        return self.svm.predict(X)

# Example usage
def load_sample_images():
    # Simulated image loading
    images = np.random.rand(100, 64, 64, 3)  # 100 RGB images
    labels = np.random.randint(0, 2, 100)    # Binary labels
    return images, labels

# Train and evaluate
images, labels = load_sample_images()
clf = ImageSVM()
clf.fit(images, labels)
predictions = clf.predict(images[:10])
```

Slide 10: Online Learning with SVM

Implementing online learning for SVM enables handling large-scale datasets that don't fit in memory. This implementation uses stochastic gradient descent for incremental updates.

```python
class OnlineSVM:
    def __init__(self, lambda_param=0.01, learning_rate=0.01):
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.w = None
        self.b = 0
        
    def partial_fit(self, x, y):
        """Update model with single instance"""
        if self.w is None:
            self.w = np.zeros(x.shape[0])
            
        # Compute gradient
        margin = y * (np.dot(self.w, x) + self.b)
        
        if margin < 1:
            grad_w = self.lambda_param * self.w - y * x
            grad_b = -y
        else:
            grad_w = self.lambda_param * self.w
            grad_b = 0
            
        # Update parameters
        self.w -= self.learning_rate * grad_w
        self.b -= self.learning_rate * grad_b
        
    def fit(self, X, y, n_epochs=1):
        """Train model with multiple passes over data"""
        n_samples = X.shape[0]
        
        for epoch in range(n_epochs):
            for i in range(n_samples):
                self.partial_fit(X[i], y[i])
                
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

# Example with streaming data
from sklearn.preprocessing import StandardScaler

# Generate streaming data
n_samples = 1000
X = np.random.randn(n_samples, 10)
y = np.sign(X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train online
svm = OnlineSVM()
batch_size = 100
for i in range(0, n_samples, batch_size):
    X_batch = X[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    svm.fit(X_batch, y_batch)

# Evaluate
accuracy = np.mean(svm.predict(X) == y)
print(f"Final accuracy: {accuracy:.3f}")
```

Slide 11: SVM for Anomaly Detection

Support Vector Machines can be adapted for anomaly detection using One-Class SVM. This implementation demonstrates how to identify outliers in high-dimensional data with custom feature normalization.

```python
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

class AnomalyDetectorSVM:
    def __init__(self, nu=0.1, kernel='rbf'):
        self.scaler = RobustScaler()
        self.detector = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma='auto'
        )
        
    def fit(self, X):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit one-class SVM
        self.detector.fit(X_scaled)
        
        # Compute decision boundary
        self.decision_scores = self.detector.score_samples(X_scaled)
        self.threshold = np.percentile(self.decision_scores, 
                                     self.detector.nu * 100)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        scores = self.detector.score_samples(X_scaled)
        return np.where(scores < self.threshold, -1, 1)
    
    def decision_function(self, X):
        X_scaled = self.scaler.transform(X)
        return self.detector.score_samples(X_scaled)

# Example usage with financial data
def generate_financial_data(n_samples=1000):
    # Simulate stock returns and volatility
    returns = np.random.normal(0, 1, n_samples)
    volatility = np.abs(np.random.normal(0, 0.5, n_samples))
    volume = np.random.exponential(1, n_samples)
    
    # Insert anomalies
    anomaly_idx = np.random.choice(n_samples, size=int(0.05*n_samples))
    returns[anomaly_idx] *= 5
    volatility[anomaly_idx] *= 3
    
    return np.column_stack([returns, volatility, volume])

# Train and evaluate
X = generate_financial_data()
detector = AnomalyDetectorSVM(nu=0.05)
detector.fit(X)

# Detect anomalies
anomalies = detector.predict(X)
print(f"Detected anomalies: {np.sum(anomalies == -1)}")
```

Slide 12: Feature Selection with SVM

This implementation uses recursive feature elimination with SVM to identify the most important features for classification, incorporating cross-validation for robust feature selection.

```python
class SVMFeatureSelector:
    def __init__(self, n_features_to_select=10):
        self.n_features = n_features_to_select
        self.selected_features = None
        self.feature_rankings = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        remaining_features = list(range(n_features))
        rankings = np.zeros(n_features)
        
        while len(remaining_features) > self.n_features:
            # Train SVM
            svm = LinearSVC(C=1.0, penalty='l2')
            svm.fit(X[:, remaining_features], y)
            
            # Get feature weights
            weights = np.abs(svm.coef_[0])
            
            # Remove feature with smallest weight
            min_weight_idx = np.argmin(weights)
            feature_to_remove = remaining_features[min_weight_idx]
            
            # Update rankings
            rankings[feature_to_remove] = len(remaining_features)
            
            # Remove feature
            remaining_features.pop(min_weight_idx)
            
        self.selected_features = remaining_features
        self.feature_rankings = rankings
        
    def transform(self, X):
        return X[:, self.selected_features]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

# Example usage with cross-validation
from sklearn.model_selection import cross_val_score

# Generate dataset
X, y = make_classification(n_samples=1000, 
                          n_features=100,
                          n_informative=20)

# Select features
selector = SVMFeatureSelector(n_features_to_select=20)
X_selected = selector.fit_transform(X, y)

# Evaluate with cross-validation
svm = SVC(kernel='linear')
scores_original = cross_val_score(svm, X, y, cv=5)
scores_selected = cross_val_score(svm, X_selected, y, cv=5)

print(f"Original features accuracy: {np.mean(scores_original):.3f}")
print(f"Selected features accuracy: {np.mean(scores_selected):.3f}")
```

Slide 13: SVM for Large-Scale Learning

This implementation demonstrates efficient handling of large datasets using mini-batch processing and Nystrom approximation for kernel computations, enabling SVM training on massive datasets.

```python
class LargeScaleSVM:
    def __init__(self, batch_size=1000, n_components=100):
        self.batch_size = batch_size
        self.n_components = n_components
        self.support_vectors = None
        self.dual_coef = None
        
    def nystrom_kernel_approximation(self, X, X_landmarks):
        # Compute RBF kernel between X and landmarks
        gamma = 1.0 / X.shape[1]
        K_nm = np.exp(-gamma * cdist(X, X_landmarks, 'sqeuclidean'))
        
        # Compute kernel between landmarks
        K_mm = np.exp(-gamma * cdist(X_landmarks, X_landmarks, 'sqeuclidean'))
        
        # Compute approximation
        U, S, _ = np.linalg.svd(K_mm)
        S = np.maximum(S, 1e-12)
        components = np.dot(U / np.sqrt(S), K_nm.T)
        
        return components.T
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # Select landmark points
        landmark_indices = np.random.choice(
            n_samples, 
            self.n_components, 
            replace=False
        )
        X_landmarks = X[landmark_indices]
        
        # Initialize model parameters
        self.support_vectors = X_landmarks
        self.dual_coef = np.zeros(self.n_components)
        
        # Train in mini-batches
        for i in range(0, n_samples, self.batch_size):
            X_batch = X[i:min(i + self.batch_size, n_samples)]
            y_batch = y[i:min(i + self.batch_size, n_samples)]
            
            # Compute kernel approximation
            K_batch = self.nystrom_kernel_approximation(
                X_batch, 
                X_landmarks
            )
            
            # Update model parameters
            self._update_parameters(K_batch, y_batch)
    
    def _update_parameters(self, K_batch, y_batch):
        # Solve dual optimization problem for batch
        n_samples = K_batch.shape[0]
        P = np.dot(K_batch, K_batch.T)
        q = -y_batch
        
        # Box constraints
        C = 1.0
        bounds = [(0, C) for _ in range(n_samples)]
        
        from scipy.optimize import minimize
        result = minimize(
            lambda x: 0.5 * np.dot(x, np.dot(P, x)) + np.dot(q, x),
            np.zeros(n_samples),
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # Update dual coefficients
        self.dual_coef += np.dot(K_batch.T, result.x * y_batch)
    
    def predict(self, X):
        K_test = self.nystrom_kernel_approximation(X, self.support_vectors)
        return np.sign(np.dot(K_test, self.dual_coef))

# Example usage
X_large = np.random.randn(10000, 50)
y_large = np.sign(X_large[:, 0] + X_large[:, 1])

# Train model
svm = LargeScaleSVM()
svm.fit(X_large, y_large)

# Evaluate
test_accuracy = np.mean(svm.predict(X_large[:1000]) == y_large[:1000])
print(f"Test accuracy: {test_accuracy:.3f}")
```

Slide 14: Additional Resources

*   A New Support Vector Method for Optimal Margin Classification [https://arxiv.org/abs/2203.15721](https://arxiv.org/abs/2203.15721)
*   Large-Scale Support Vector Machines: Algorithms and Applications [https://arxiv.org/abs/2105.09815](https://arxiv.org/abs/2105.09815)
*   Kernel Methods for Deep Learning [https://arxiv.org/abs/2002.09347](https://arxiv.org/abs/2002.09347)
*   Support Vector Machines for Time Series Analysis [https://arxiv.org/abs/2104.12463](https://arxiv.org/abs/2104.12463)
*   Online Learning with Kernels: A Survey [https://arxiv.org/abs/1902.06865](https://arxiv.org/abs/1902.06865)

