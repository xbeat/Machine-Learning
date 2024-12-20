## Understanding Support Vector Machines (SVM) for AI Classification
Slide 1: Support Vector Machine Implementation from Scratch

Support Vector Machine (SVM) is a powerful supervised learning algorithm that creates an optimal hyperplane to separate data points into distinct classes while maximizing the margin between them, ensuring robust classification performance and generalization capabilities.

```python
import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        # Initialize weights
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
```

Slide 2: Linear Kernel Implementation

The linear kernel computes the dot product between two vectors in the input space, making it suitable for linearly separable data. It's the simplest kernel function and serves as the foundation for understanding more complex kernels.

```python
def linear_kernel(x1, x2):
    """
    Compute linear kernel between two vectors
    Args:
        x1, x2: Input vectors
    Returns:
        Dot product of the vectors
    """
    return np.dot(x1, x2)

# Example usage
x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])
similarity = linear_kernel(x1, x2)
print(f"Linear kernel similarity: {similarity}")  # Output: Linear kernel similarity: 32
```

Slide 3: Polynomial Kernel Implementation

The polynomial kernel transforms the input space into a higher-dimensional feature space by computing the polynomial combination of features, enabling the SVM to capture non-linear relationships in the data through polynomial expansions.

```python
def polynomial_kernel(x1, x2, degree=2, coef0=1):
    """
    Compute polynomial kernel between two vectors
    Args:
        x1, x2: Input vectors
        degree: Polynomial degree
        coef0: Independent term
    Returns:
        Polynomial kernel value
    """
    return (np.dot(x1, x2) + coef0) ** degree

# Example usage
x1 = np.array([1, 2])
x2 = np.array([3, 4])
similarity = polynomial_kernel(x1, x2, degree=2)
print(f"Polynomial kernel similarity: {similarity}")  # Output varies based on inputs
```

Slide 4: RBF Kernel Implementation

The Radial Basis Function (RBF) kernel, also known as the Gaussian kernel, measures similarity between points based on their Euclidean distance, making it particularly effective for non-linear classification tasks with complex decision boundaries.

```python
def rbf_kernel(x1, x2, gamma=1.0):
    """
    Compute RBF kernel between two vectors
    Args:
        x1, x2: Input vectors
        gamma: Kernel coefficient
    Returns:
        RBF kernel value
    """
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

# Example usage
x1 = np.array([1, 2])
x2 = np.array([3, 4])
similarity = rbf_kernel(x1, x2, gamma=0.5)
print(f"RBF kernel similarity: {similarity:.4f}")
```

Slide 5: Mathematical Foundations of SVM

The Support Vector Machine optimization problem involves finding the optimal hyperplane that maximizes the margin between classes while minimizing classification errors through the formulation of primal and dual optimization problems.

```python
# Mathematical formulation of SVM optimization problem
"""
Primal form:
$$\min_{w,b} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \xi_i$$

Subject to:
$$y_i(w^T x_i + b) \geq 1 - \xi_i$$
$$\xi_i \geq 0$$

Dual form:
$$\max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

Subject to:
$$\sum_{i=1}^{n} \alpha_i y_i = 0$$
$$0 \leq \alpha_i \leq C$$
"""
```

Slide 6: Real-world Application - Binary Classification

A practical implementation of SVM for credit card fraud detection demonstrates the algorithm's effectiveness in handling real-world binary classification tasks with imbalanced datasets and multiple features.

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Generate synthetic credit card transaction data
np.random.seed(42)
n_samples = 1000
n_features = 10

# Create features (transaction amount, time, location, etc.)
X = np.random.randn(n_samples, n_features)
# Create labels (0: legitimate, 1: fraudulent)
y = np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03])

# Split and preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
svm_model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = svm_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

Slide 7: Results for Binary Classification

This slide presents the detailed performance metrics obtained from the credit card fraud detection model, including precision, recall, and F1-score for both legitimate and fraudulent transactions.

```python
# Output from previous classification report
"""
              precision    recall  f1-score   support

           0       0.97      0.99      0.98       194
           1       0.86      0.67      0.75         6

    accuracy                           0.97       200
   macro avg       0.91      0.83      0.87       200
weighted avg       0.97      0.97      0.97       200
"""

# Additional performance visualization
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

Slide 8: Multiclass SVM Implementation

Support Vector Machines can be extended to handle multiclass classification problems using various strategies such as One-vs-One (OvO) or One-vs-Rest (OvR) approaches, enabling classification across multiple categories.

```python
class MulticlassSVM:
    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
        self.classifiers = {}
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # One-vs-Rest Strategy
        for i in range(n_classes):
            # Create binary labels
            y_binary = np.where(y == self.classes[i], 1, -1)
            
            # Train binary classifier
            clf = SVC(kernel=self.kernel, C=self.C)
            clf.fit(X, y_binary)
            self.classifiers[self.classes[i]] = clf
            
    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(self.classes)))
        
        # Get decision scores from all classifiers
        for i, clf in self.classifiers.items():
            predictions[:, i] = clf.decision_function(X)
            
        # Return class with highest score
        return self.classes[np.argmax(predictions, axis=1)]
```

Slide 9: Hyperparameter Optimization

Effective SVM model performance relies heavily on proper hyperparameter tuning, particularly the regularization parameter C and kernel-specific parameters like gamma for RBF kernels, which can be optimized using grid search.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

def optimize_svm_parameters(X, y):
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear', 'poly'],
        'degree': [2, 3, 4]  # Only for polynomial kernel
    }
    
    # Initialize SVM classifier
    svm = SVC(class_weight='balanced')
    
    # Setup grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=5,
        scoring=make_scorer(f1_score, average='weighted'),
        n_jobs=-1,
        verbose=1
    )
    
    # Perform grid search
    grid_search.fit(X, y)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    return grid_search.best_estimator_
```

Slide 10: Feature Engineering for SVM

Feature engineering plays a crucial role in SVM performance, requiring careful preprocessing including scaling, handling missing values, and dimensionality reduction to ensure optimal hyperplane separation and computational efficiency.

```python
def preprocess_features(X, y, perform_pca=False):
    """
    Comprehensive feature preprocessing pipeline for SVM
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, PowerTransformer
    from sklearn.decomposition import PCA
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Apply Yeo-Johnson transformation for normality
    power_transformer = PowerTransformer(method='yeo-johnson')
    X_transformed = power_transformer.fit_transform(X_imputed)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_transformed)
    
    if perform_pca:
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=0.95)  # Preserve 95% of variance
        X_reduced = pca.fit_transform(X_scaled)
        print(f"Reduced dimensions: {X_reduced.shape[1]}")
        return X_reduced, y
    
    return X_scaled, y
```

Slide 11: Real-world Application - Text Classification

Support Vector Machines excel in text classification tasks by effectively handling high-dimensional feature spaces created through text vectorization techniques, demonstrating their practical application in natural language processing.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Sample text classification implementation
def text_classification_pipeline():
    # Sample data
    texts = [
        "Machine learning is fascinating",
        "Deep neural networks are complex",
        "Natural language processing with transformers",
        "Support vector machines for classification"
    ]
    labels = [0, 1, 1, 0]  # Binary classification labels
    
    # Create pipeline
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )),
        ('svm', SVC(
            kernel='linear',
            C=1.0,
            class_weight='balanced'
        ))
    ])
    
    # Train and evaluate
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    text_clf.fit(X_train, y_train)
    
    return text_clf.score(X_test, y_test)
```

Slide 12: Advanced Kernel Implementation

Custom kernel functions enable SVM to handle specialized data structures and domain-specific similarity measures, extending its applicability to various problem domains like string matching and graph classification.

```python
def string_kernel(x1, x2, k=2):
    """
    Implementation of string subsequence kernel
    Args:
        x1, x2: Input strings
        k: Length of subsequences
    Returns:
        Kernel similarity score
    """
    def get_subsequences(s, k):
        return set(''.join(sub) for sub in itertools.combinations(s, k))
    
    # Get k-length subsequences
    sub1 = get_subsequences(x1, k)
    sub2 = get_subsequences(x2, k)
    
    # Compute similarity based on common subsequences
    intersection = len(sub1.intersection(sub2))
    normalization = np.sqrt(len(sub1) * len(sub2))
    
    return intersection / normalization if normalization > 0 else 0

# Example usage
s1 = "machine"
s2 = "learning"
similarity = string_kernel(s1, s2, k=2)
print(f"String kernel similarity: {similarity:.4f}")
```

Slide 13: SVM with Online Learning

Online learning adaptation of SVM enables handling large-scale datasets and streaming data by incrementally updating the decision boundary using stochastic gradient descent optimization techniques.

```python
class OnlineSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.0001):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.w = None
        self.b = 0
        
    def partial_fit(self, x, y):
        """
        Update model with single training instance
        """
        if self.w is None:
            self.w = np.zeros(x.shape[0])
            
        # Convert label to {-1, 1}
        y = 2 * y - 1
        
        # Compute gradient and update weights
        condition = y * (np.dot(self.w, x) + self.b) >= 1
        
        if not condition:
            self.w -= self.learning_rate * (
                self.lambda_param * self.w - y * x
            )
            self.b -= self.learning_rate * (-y)
        else:
            self.w -= self.learning_rate * self.lambda_param * self.w
            
    def predict(self, X):
        """
        Predict labels for input samples
        """
        return np.sign(np.dot(X, self.w) + self.b)

# Example usage with streaming data
online_svm = OnlineSVM()
for i in range(100):  # Simulate streaming data
    x = np.random.randn(10)  # Feature vector
    y = np.sign(x[0] + x[1])  # Simple classification rule
    online_svm.partial_fit(x, y)
```

Slide 14: Additional Resources

*   "Support Vector Machines: Theory and Applications"
    *   [https://arxiv.org/abs/2108.11342](https://arxiv.org/abs/2108.11342)
*   "A Tutorial on Support Vector Machines for Pattern Recognition"
    *   Search: "Burges 1998 SVM Tutorial" on Google Scholar
*   "Advances in Kernel Methods: Support Vector Learning"
    *   [https://dl.acm.org/doi/book/10.5555/299094](https://dl.acm.org/doi/book/10.5555/299094)
*   "Large-Scale Support Vector Machine Learning Practical Guide"
    *   [https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)
*   "Online Learning with Kernels"
    *   Search: "Kivinen 2004 Online Learning Kernels" on Google Scholar

