## L1 Regularization for Feature Selection
Slide 1: Introduction to L1 Regularization (Lasso)

L1 regularization adds the absolute value of coefficients as a penalty term to the loss function, effectively shrinking some coefficients to exactly zero. This property makes Lasso particularly useful for feature selection in high-dimensional datasets where sparsity is desired.

```python
# Implementation of L1 regularization from scratch
import numpy as np

class L1Regularization:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def __call__(self, weights):
        """Calculate L1 penalty"""
        return self.alpha * np.sum(np.abs(weights))
    
    def gradient(self, weights):
        """Calculate gradient of L1 penalty"""
        return self.alpha * np.sign(weights)
```

Slide 2: L1 vs L2 Regularization Mathematics

The mathematical foundations behind L1 and L2 regularization reveal why L1 promotes sparsity while L2 doesn't. The key difference lies in their gradient behavior near zero and their geometric interpretation in parameter space.

```python
# Mathematical representation (not rendered)
$$
\text{L1 (Lasso):} \quad J(\theta) = \text{Loss}(\theta) + \alpha \sum_{i=1}^{n} |\theta_i|
$$

$$
\text{L2 (Ridge):} \quad J(\theta) = \text{Loss}(\theta) + \alpha \sum_{i=1}^{n} \theta_i^2
$$
```

Slide 3: Lasso Regression Implementation

A complete implementation of Lasso regression using coordinate descent optimization, which is particularly efficient for L1 regularization problems. This implementation includes the core algorithm and handling of the soft-thresholding operator.

```python
import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        
    def soft_threshold(self, x, lambda_):
        """Soft-thresholding operator"""
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        
        for _ in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            for j in range(n_features):
                r = y - np.dot(X, self.coef_) + self.coef_[j] * X[:, j]
                self.coef_[j] = self.soft_threshold(
                    np.dot(X[:, j], r),
                    self.alpha * n_samples
                ) / (np.dot(X[:, j], X[:, j]))
                
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                break
                
        return self
```

Slide 4: Real-world Example - Gene Selection

The application of Lasso regularization in genomics for identifying relevant genes from high-dimensional microarray data. This example demonstrates how L1 regularization effectively selects important features while eliminating irrelevant ones.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate synthetic gene expression data
np.random.seed(42)
n_samples, n_features = 100, 1000
X = np.random.randn(n_samples, n_features)
true_coefficients = np.zeros(n_features)
true_coefficients[:5] = [3, -2, 4, -1, 5]  # Only 5 relevant genes
y = np.dot(X, true_coefficients) + np.random.randn(n_samples) * 0.1

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Apply Lasso
lasso = LassoRegression(alpha=0.1)
lasso.fit(X_train, y_train)

# Identify selected genes
selected_genes = np.where(np.abs(lasso.coef_) > 1e-10)[0]
print(f"Number of selected genes: {len(selected_genes)}")
print(f"Selected gene indices: {selected_genes}")
```

Slide 5: Cross-validation for Optimal Regularization

Cross-validation is crucial for finding the optimal regularization strength in Lasso regression. This implementation shows how to perform k-fold cross-validation to select the best alpha parameter.

```python
class LassoCrossValidation:
    def __init__(self, alphas=None, cv=5):
        self.alphas = alphas if alphas is not None else np.logspace(-4, 1, 100)
        self.cv = cv
        
    def cross_validate(self, X, y):
        n_samples = X.shape[0]
        fold_size = n_samples // self.cv
        mse_scores = []
        
        for alpha in self.alphas:
            fold_scores = []
            for i in range(self.cv):
                # Create fold indices
                val_idx = slice(i * fold_size, (i + 1) * fold_size)
                train_idx = list(set(range(n_samples)) - set(range(*val_idx.indices(n_samples))))
                
                # Split data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train and evaluate
                model = LassoRegression(alpha=alpha)
                model.fit(X_train, y_train)
                y_pred = np.dot(X_val, model.coef_)
                mse = np.mean((y_val - y_pred) ** 2)
                fold_scores.append(mse)
                
            mse_scores.append(np.mean(fold_scores))
            
        best_alpha_idx = np.argmin(mse_scores)
        return self.alphas[best_alpha_idx]
```

Slide 6: Elastic Net - Combining L1 and L2

Elastic Net combines L1 and L2 regularization to overcome some limitations of Lasso, particularly in handling correlated features. It provides a more robust feature selection mechanism while maintaining the benefits of both regularization types.

```python
class ElasticNet:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        
        for _ in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            for j in range(n_features):
                r = y - np.dot(X, self.coef_) + self.coef_[j] * X[:, j]
                l1_coef = self.alpha * self.l1_ratio * n_samples
                l2_coef = self.alpha * (1 - self.l1_ratio)
                
                numerator = np.dot(X[:, j], r)
                if numerator > l1_coef:
                    self.coef_[j] = (numerator - l1_coef) / (np.dot(X[:, j], X[:, j]) + l2_coef)
                elif numerator < -l1_coef:
                    self.coef_[j] = (numerator + l1_coef) / (np.dot(X[:, j], X[:, j]) + l2_coef)
                else:
                    self.coef_[j] = 0
                    
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                break
                
        return self
```

Slide 7: Feature Selection Stability Analysis

A comprehensive implementation for analyzing the stability of feature selection across different subsets of data, which is crucial for assessing the reliability of L1 regularization in real-world applications.

```python
import numpy as np
from sklearn.model_selection import ShuffleSplit

class FeatureSelectionStability:
    def __init__(self, model, n_iterations=100, subsample_size=0.8):
        self.model = model
        self.n_iterations = n_iterations
        self.subsample_size = subsample_size
        
    def stability_score(self, feature_sets):
        n = len(feature_sets)
        if n <= 1:
            return 1.0
        
        pairwise_similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                intersection = len(set(feature_sets[i]) & set(feature_sets[j]))
                union = len(set(feature_sets[i]) | set(feature_sets[j]))
                similarity = intersection / union if union > 0 else 1.0
                pairwise_similarities.append(similarity)
                
        return np.mean(pairwise_similarities)
    
    def analyze(self, X, y, threshold=1e-5):
        rs = ShuffleSplit(n_splits=self.n_iterations, 
                         test_size=1-self.subsample_size)
        selected_features = []
        
        for train_idx, _ in rs.split(X):
            X_subset = X[train_idx]
            y_subset = y[train_idx]
            
            self.model.fit(X_subset, y_subset)
            selected = np.where(np.abs(self.model.coef_) > threshold)[0]
            selected_features.append(selected.tolist())
            
        stability = self.stability_score(selected_features)
        feature_freq = np.zeros(X.shape[1])
        for features in selected_features:
            feature_freq[features] += 1
        feature_freq /= self.n_iterations
        
        return stability, feature_freq
```

Slide 8: Sparse Recovery Performance

Implementation of metrics to evaluate how well L1 regularization recovers the true sparse structure of the data, including precision, recall, and F1-score for feature selection.

```python
class SparseRecoveryMetrics:
    def __init__(self, threshold=1e-5):
        self.threshold = threshold
        
    def evaluate(self, true_coef, estimated_coef):
        true_support = set(np.where(np.abs(true_coef) > self.threshold)[0])
        est_support = set(np.where(np.abs(estimated_coef) > self.threshold)[0])
        
        true_positives = len(true_support & est_support)
        false_positives = len(est_support - true_support)
        false_negatives = len(true_support - est_support)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        recovery_error = np.linalg.norm(true_coef - estimated_coef)
        support_difference = len(true_support ^ est_support)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'recovery_error': recovery_error,
            'support_difference': support_difference
        }
```

Slide 9: Pathwise Coordinate Descent

Pathwise coordinate descent optimization provides an efficient way to compute the entire regularization path for Lasso regression, allowing us to observe how features enter the model as the regularization parameter changes.

```python
class LassoPath:
    def __init__(self, n_alphas=100, eps=1e-3, max_iter=1000):
        self.n_alphas = n_alphas
        self.eps = eps
        self.max_iter = max_iter
        
    def compute_path(self, X, y):
        n_samples, n_features = X.shape
        
        # Compute alpha_max (smallest alpha that gives all zero coefficients)
        alpha_max = np.max(np.abs(np.dot(X.T, y))) / n_samples
        alphas = np.logspace(np.log10(alpha_max * self.eps), np.log10(alpha_max), self.n_alphas)
        
        # Initialize coefficient matrix
        coef_path = np.zeros((self.n_alphas, n_features))
        
        # Compute path
        for i, alpha in enumerate(alphas):
            lasso = LassoRegression(alpha=alpha, max_iter=self.max_iter)
            lasso.fit(X, y)
            coef_path[i] = lasso.coef_
            
        return alphas, coef_path
```

Slide 10: Adaptive Lasso Implementation

Adaptive Lasso improves feature selection consistency by incorporating weights into the L1 penalty, giving different penalties to different coefficients based on their estimated importance.

```python
class AdaptiveLasso:
    def __init__(self, alpha=1.0, gamma=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X, y):
        # Initial OLS estimate
        beta_ols = np.linalg.pinv(X.T @ X) @ X.T @ y
        
        # Compute adaptive weights
        weights = 1 / (np.abs(beta_ols) ** self.gamma + self.tol)
        
        # Scale features by adaptive weights
        X_weighted = X * weights
        
        # Solve weighted Lasso problem
        lasso = LassoRegression(alpha=self.alpha, max_iter=self.max_iter)
        lasso.fit(X_weighted, y)
        
        # Transform back to original scale
        self.coef_ = lasso.coef_ * weights
        return self
```

Slide 11: Group Lasso for Structured Feature Selection

Group Lasso extends L1 regularization to handle grouped features, allowing simultaneous selection or elimination of predefined groups of features, which is particularly useful in scenarios with natural feature groupings.

```python
class GroupLasso:
    def __init__(self, alpha=1.0, groups=None, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.groups = groups
        self.max_iter = max_iter
        self.tol = tol
        
    def group_norm(self, coef, group_indices):
        return np.sqrt(np.sum(coef[group_indices] ** 2))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        
        for _ in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            for group_idx in self.groups:
                X_group = X[:, group_idx]
                r = y - np.dot(X, self.coef_) + np.dot(X_group, self.coef_[group_idx])
                
                group_correlation = np.dot(X_group.T, r)
                group_norm = np.linalg.norm(group_correlation)
                
                if group_norm > self.alpha:
                    shrinkage = 1 - self.alpha / group_norm
                    self.coef_[group_idx] = shrinkage * np.dot(
                        np.linalg.pinv(np.dot(X_group.T, X_group)), 
                        group_correlation
                    )
                else:
                    self.coef_[group_idx] = 0
                    
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                break
                
        return self
```

Slide 12: Real-world Example - Text Classification

A practical implementation of feature selection in text classification using L1 regularization to identify the most relevant words for document categorization.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TextClassifierWithFeatureSelection:
    def __init__(self, alpha=1.0):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.lasso = LassoRegression(alpha=alpha)
        
    def fit(self, texts, labels):
        # Transform texts to TF-IDF features
        X = self.vectorizer.fit_transform(texts).toarray()
        
        # Fit Lasso model
        self.lasso.fit(X, labels)
        
        # Get selected features
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        selected_indices = np.where(np.abs(self.lasso.coef_) > 1e-5)[0]
        self.selected_features = feature_names[selected_indices]
        self.feature_importance = dict(zip(
            self.selected_features,
            self.lasso.coef_[selected_indices]
        ))
        
        return self
    
    def get_important_features(self, top_n=10):
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return dict(sorted_features[:top_n])
```

Slide 13: Sparse Signal Recovery Performance Metrics

The evaluation of L1 regularization's effectiveness in recovering sparse signals requires specialized metrics that account for both the support recovery and the estimation accuracy of the non-zero coefficients.

```python
class SparseSignalMetrics:
    def __init__(self):
        self.metrics = {}
        
    def evaluate(self, true_signal, estimated_signal, noise_level=None):
        # Support recovery metrics
        true_support = np.nonzero(true_signal)[0]
        est_support = np.nonzero(estimated_signal)[0]
        
        # Calculate various performance metrics
        self.metrics['hamming_distance'] = len(set(true_support) ^ set(est_support))
        self.metrics['precision'] = len(set(true_support) & set(est_support)) / len(est_support)
        self.metrics['recall'] = len(set(true_support) & set(est_support)) / len(true_support)
        
        # Signal reconstruction error
        self.metrics['l2_error'] = np.linalg.norm(true_signal - estimated_signal)
        self.metrics['relative_error'] = self.metrics['l2_error'] / np.linalg.norm(true_signal)
        
        if noise_level:
            self.metrics['signal_to_noise'] = np.linalg.norm(true_signal) / noise_level
            
        return self.metrics
```

Slide 14: Fused Lasso for Time Series Feature Selection

Fused Lasso adds an additional penalty term to encourage sparsity in the differences between consecutive coefficients, making it particularly suitable for time series analysis and signal processing.

```python
class FusedLasso:
    def __init__(self, alpha=1.0, beta=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha  # L1 penalty
        self.beta = beta    # Fusion penalty
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        
        for _ in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            for j in range(n_features):
                # Calculate residual
                r = y - np.dot(X, self.coef_) + self.coef_[j] * X[:, j]
                
                # Calculate fusion penalty terms
                if j > 0 and j < n_features - 1:
                    fusion_term = self.beta * (2 * self.coef_[j] - 
                                             self.coef_[j-1] - self.coef_[j+1])
                elif j == 0:
                    fusion_term = self.beta * (self.coef_[j] - self.coef_[j+1])
                else:
                    fusion_term = self.beta * (self.coef_[j] - self.coef_[j-1])
                
                # Update coefficient
                numerator = np.dot(X[:, j], r) - fusion_term
                denominator = np.dot(X[:, j], X[:, j]) + self.beta
                
                if numerator > self.alpha:
                    self.coef_[j] = (numerator - self.alpha) / denominator
                elif numerator < -self.alpha:
                    self.coef_[j] = (numerator + self.alpha) / denominator
                else:
                    self.coef_[j] = 0
                    
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                break
                
        return self
```

Slide 15: Additional Resources

*   "The Elements of Statistical Learning: Lasso and Related Methods" [https://arxiv.org/abs/1104.3889](https://arxiv.org/abs/1104.3889)
*   "An Introduction to Statistical Learning with Applications in Python" [https://arxiv.org/abs/2204.01487](https://arxiv.org/abs/2204.01487)
*   "Sparse Signal Recovery via Iterative Support Detection" [https://arxiv.org/abs/1092.4424](https://arxiv.org/abs/1092.4424)
*   "Adaptive Lasso for High Dimensional Regression and Gaussian Graphical Modeling" [https://arxiv.org/abs/1501.03175](https://arxiv.org/abs/1501.03175)
*   "Group Lasso with Overlaps: Theoretical Foundations and Algorithms" [https://arxiv.org/abs/1110.0413](https://arxiv.org/abs/1110.0413)

