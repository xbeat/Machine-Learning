## Advantages of Support Vector Machines for Robust Classification
Slide 1: Maximum Margin Classification

Support Vector Machines (SVM) establish optimal decision boundaries by maximizing the margin between classes, creating a robust separator that enhances generalization. The margin represents the distance between the hyperplane and the nearest data points from each class, called support vectors.

```python
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

# Create and train SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# Plot decision boundary
w = clf.coef_[0]
b = clf.intercept_[0]
x_points = np.linspace(-3, 3)
y_points = -(w[0] * x_points + b) / w[1]

plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')
plt.plot(x_points, y_points, 'k-')
plt.legend()
plt.show()
```

Slide 2: Mathematical Foundation of SVM

The SVM optimization problem aims to find the hyperplane that maximizes the geometric margin while minimizing classification errors. This involves solving a quadratic programming problem with linear constraints.

```python
# Mathematical formulation in LaTeX notation:
"""
$$
\begin{aligned}
\text{minimize} \quad & \frac{1}{2}\|w\|^2 \\
\text{subject to} \quad & y_i(w^Tx_i + b) \geq 1, \quad i=1,\ldots,n
\end{aligned}
$$

For soft margin SVM:
$$
\begin{aligned}
\text{minimize} \quad & \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i \\
\text{subject to} \quad & y_i(w^Tx_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad i=1,\ldots,n
\end{aligned}
$$
"""
```

Slide 3: Implementing SVM from Scratch

The implementation demonstrates the core concepts of SVM using gradient descent optimization to find the optimal hyperplane parameters w and b that maximize the margin between classes.

```python
class SimpleSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - 
                                       np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]
                    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
```

Slide 4: Kernel Trick Implementation

The kernel trick allows SVM to handle non-linearly separable data by mapping features into a higher-dimensional space where linear separation becomes possible, without explicitly computing the transformation.

```python
def gaussian_kernel(x1, x2, sigma=1.0):
    return np.exp(-np.linalg.norm(x1 - x2, axis=1)**2 / (2 * (sigma ** 2)))

class KernelSVM:
    def __init__(self, kernel=gaussian_kernel, C=1.0):
        self.kernel = kernel
        self.C = C
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        # Compute the kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            K[i,:] = self.kernel(X[i], X)
            
        # Solve the dual optimization problem
        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        A = y.reshape(1, -1)
        b = np.zeros(1)
        
        from cvxopt import matrix, solvers
        solution = solvers.qp(matrix(P), matrix(q), matrix(-np.eye(n_samples)),
                            matrix(np.zeros(n_samples)), matrix(A), matrix(b))
        
        # Extract support vectors
        self.alpha = np.array(solution['x']).flatten()
        sv = self.alpha > 1e-5
        self.support_vectors = X[sv]
        self.support_vector_labels = y[sv]
        self.alpha = self.alpha[sv]
```

Slide 5: Real-world Application - Text Classification

SVMs excel in text classification tasks due to their ability to handle high-dimensional sparse data effectively. This implementation demonstrates document classification using TF-IDF features.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pandas as pd

# Sample text data
documents = [
    "machine learning algorithms optimize performance",
    "deep neural networks process complex patterns",
    "stock market analysis predicts trends",
    "financial forecasting uses historical data"
]
labels = [0, 0, 1, 1]  # 0: Tech, 1: Finance

# Create pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LinearSVC())
])

# Train classifier
text_clf.fit(documents, labels)

# Predict new documents
new_docs = ["artificial intelligence improves automation",
            "market volatility affects investments"]
predictions = text_clf.predict(new_docs)
print(f"Predictions: {predictions}")  # Output: [0, 1]
```

Slide 6: SVM Hyperparameter Tuning

Optimizing SVM performance requires careful tuning of hyperparameters like C (regularization) and kernel parameters. This implementation demonstrates systematic hyperparameter optimization using grid search with cross-validation.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create pipeline with preprocessing and SVM
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', svm.SVC())
])

# Define parameter grid
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['rbf', 'poly'],
    'svm__gamma': ['scale', 'auto', 0.1, 1],
    'svm__degree': [2, 3, 4]  # Only for poly kernel
}

# Perform grid search
grid_search = GridSearchCV(
    svm_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# Fit and get best parameters
grid_search.fit(X, y)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
```

Slide 7: Multi-class SVM Classification

SVMs extend to multi-class problems using one-vs-one or one-vs-rest strategies, enabling classification across multiple categories while maintaining their maximum margin properties.

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

class MultiClassSVM:
    def __init__(self, kernel='rbf', C=1.0):
        self.encoder = LabelEncoder()
        self.classifier = OneVsRestClassifier(SVC(kernel=kernel, C=C))
        
    def fit(self, X, y):
        # Encode labels
        y_encoded = self.encoder.fit_transform(y)
        # Train classifier
        self.classifier.fit(X, y_encoded)
        
    def predict(self, X):
        # Predict and decode labels
        y_pred = self.classifier.predict(X)
        return self.encoder.inverse_transform(y_pred)

# Example usage
X = np.random.randn(300, 2)
y = np.array(['A', 'B', 'C'] * 100)

clf = MultiClassSVM(kernel='rbf', C=1.0)
clf.fit(X, y)
predictions = clf.predict(X[:5])
print(f"Sample predictions: {predictions}")
```

Slide 8: Implementing Custom Kernels

Custom kernel functions enable SVMs to capture domain-specific similarity measures between data points, enhancing their flexibility for specialized applications.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomKernelSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel_func, C=1.0):
        self.kernel_func = kernel_func
        self.C = C
        
    def spectrum_kernel(self, s1, s2, k=3):
        """Custom string kernel for sequence data"""
        def get_kmers(s):
            return set(s[i:i+k] for i in range(len(s)-k+1))
        
        s1_kmers = get_kmers(s1)
        s2_kmers = get_kmers(s2)
        return len(s1_kmers.intersection(s2_kmers))
    
    def matrix_kernel(self, X1, X2):
        """Compute kernel matrix"""
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i,j] = self.kernel_func(X1[i], X2[j])
        return K
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.K = self.matrix_kernel(X, X)
        # Implement QP solver here for weight calculation
        return self
    
    def predict(self, X):
        K_pred = self.matrix_kernel(X, self.X_train)
        # Implement prediction using kernel matrix
        return np.sign(K_pred.dot(self.alpha * self.y_train) + self.b)

# Example usage with custom string kernel
def string_kernel(s1, s2, k=3):
    return CustomKernelSVM.spectrum_kernel(None, s1, s2, k)

svm = CustomKernelSVM(kernel_func=string_kernel)
```

Slide 9: SVM for Anomaly Detection

SVMs can be adapted for anomaly detection by learning the boundary that encloses normal data points, making them effective for identifying outliers and unusual patterns.

```python
from sklearn.svm import OneClassSVM
import numpy as np
import matplotlib.pyplot as plt

class AnomalyDetectorSVM:
    def __init__(self, nu=0.1, kernel='rbf'):
        self.detector = OneClassSVM(nu=nu, kernel=kernel)
        
    def fit_detect(self, X, plot=True):
        # Fit the model
        self.detector.fit(X)
        
        # Get predictions
        y_pred = self.detector.predict(X)
        
        if plot:
            # Create mesh grid
            xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5,
                                           X[:, 0].max()+0.5, 100),
                                np.linspace(X[:, 1].min()-0.5,
                                           X[:, 1].max()+0.5, 100))
            
            # Get predictions on mesh grid
            Z = self.detector.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot results
            plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
            plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=plt.cm.Paired)
            plt.title('SVM Anomaly Detection')
            plt.show()
            
        return y_pred

# Generate sample data with anomalies
X_normal = np.random.randn(100, 2)
X_anomalies = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.vstack([X_normal, X_anomalies])

# Detect anomalies
detector = AnomalyDetectorSVM(nu=0.1)
predictions = detector.fit_detect(X)
print(f"Number of anomalies detected: {sum(predictions == -1)}")
```

Slide 10: Online Learning with SVM

Online learning enables SVMs to adapt to streaming data by updating the model incrementally. This implementation demonstrates how to handle large-scale datasets that don't fit in memory.

```python
class OnlineSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.0001):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.w = None
        self.b = 0
        
    def partial_fit(self, x, y):
        if self.w is None:
            self.w = np.zeros(x.shape[0])
            
        # Compute prediction
        prediction = np.dot(self.w, x) + self.b
        
        # Update if prediction is wrong
        if y * prediction < 1:
            self.w = (1 - self.lr * self.lambda_param) * self.w + \
                    self.lr * y * x
            self.b += self.lr * y
        else:
            self.w = (1 - self.lr * self.lambda_param) * self.w
            
    def predict(self, x):
        return np.sign(np.dot(self.w, x) + self.b)

# Example usage with streaming data
online_svm = OnlineSVM()
for _ in range(1000):
    # Simulate streaming data
    x = np.random.randn(10)
    y = np.sign(x[0] + x[1])
    
    # Update model
    online_svm.partial_fit(x, y)
    
    # Optional: evaluate performance periodically
    if _ % 100 == 0:
        correct = 0
        total = 100
        for i in range(total):
            x_test = np.random.randn(10)
            y_test = np.sign(x_test[0] + x_test[1])
            correct += (online_svm.predict(x_test) == y_test)
        print(f"Accuracy at iteration {_}: {correct/total:.2f}")
```

Slide 11: SVM for Regression (SVR)

Support Vector Regression extends SVM principles to continuous output variables by introducing an ε-insensitive loss function that creates a tube around the regression line.

```python
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

class SVRegressor:
    def __init__(self, kernel='rbf', epsilon=0.1, C=1.0):
        self.model = SVR(kernel=kernel, epsilon=epsilon, C=C)
        
    def fit_and_visualize(self, X, y):
        # Fit the model
        self.model.fit(X.reshape(-1, 1), y)
        
        # Create prediction line
        X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = self.model.predict(X_test)
        
        # Plot results
        plt.scatter(X, y, color='blue', label='Data points')
        plt.plot(X_test, y_pred, color='red', label='SVR prediction')
        plt.plot(X_test, y_pred + self.model.epsilon, 'k--', 
                label='ε-tube boundary')
        plt.plot(X_test, y_pred - self.model.epsilon, 'k--')
        plt.legend()
        plt.show()
        
        # Return support vectors
        return self.model.support_vectors_

# Generate sample regression data
np.random.seed(42)
X = np.sort(5 * np.random.rand(100))
y = np.sin(X) + np.random.normal(0, 0.1, 100)

# Create and train SVR model
svr = SVRegressor(epsilon=0.1, C=1.0)
support_vectors = svr.fit_and_visualize(X, y)
print(f"Number of support vectors: {len(support_vectors)}")
```

Slide 12: Feature Selection with SVM

SVMs can be used for feature selection by analyzing the weights assigned to different features, helping identify the most relevant variables for classification.

```python
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

class SVMFeatureSelector:
    def __init__(self, C=1.0, threshold='mean'):
        self.svm = LinearSVC(C=C, penalty='l1', dual=False)
        self.selector = SelectFromModel(self.svm, prefit=False, 
                                      threshold=threshold)
        self.scaler = StandardScaler()
        
    def fit_transform(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit selector
        self.selector.fit(X_scaled, y)
        
        # Get selected features
        selected_features = self.selector.get_support()
        feature_importance = np.abs(self.selector.estimator_.coef_).reshape(-1)
        
        # Sort features by importance
        feature_ranks = np.argsort(feature_importance)[::-1]
        
        # Transform data
        X_selected = self.selector.transform(X_scaled)
        
        return X_selected, selected_features, feature_ranks

# Example usage
X = np.random.randn(200, 20)  # 20 features
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Only first 2 features are relevant

selector = SVMFeatureSelector(C=0.1)
X_selected, selected_features, feature_ranks = selector.fit_transform(X, y)

print(f"Original features: {X.shape[1]}")
print(f"Selected features: {X_selected.shape[1]}")
print(f"Top 5 feature indices: {feature_ranks[:5]}")
```

Slide 13: Additional Resources

*   ArXiv paper: "Large-Scale Training of SVMs with Stochastic Gradient Descent" - [https://arxiv.org/abs/1202.6547](https://arxiv.org/abs/1202.6547)
*   ArXiv paper: "Multiple Kernel Learning for SVM-based Image Classification" - [https://arxiv.org/abs/1902.00415](https://arxiv.org/abs/1902.00415)
*   ArXiv paper: "Online Support Vector Machine for Large-Scale Data" - [https://arxiv.org/abs/1803.02346](https://arxiv.org/abs/1803.02346)
*   Suggested search terms for Google Scholar:
    *   "Support Vector Machines optimization techniques"
    *   "SVM kernel selection methods"
    *   "Online SVM implementations"
    *   "Feature selection with SVM"

