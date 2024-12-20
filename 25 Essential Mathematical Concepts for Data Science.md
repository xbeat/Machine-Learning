## 25 Essential Mathematical Concepts for Data Science
Slide 1: Maximum Likelihood Estimation (MLE)

Maximum Likelihood Estimation is a fundamental statistical method used to estimate model parameters by maximizing the likelihood function of observed data. In data science, MLE serves as the theoretical foundation for many machine learning algorithms, particularly in parametric modeling and probabilistic approaches.

```python
import numpy as np
from scipy.optimize import minimize

def mle_estimation(data):
    # Function to calculate negative log likelihood
    def neg_log_likelihood(params):
        mu, sigma = params
        return -np.sum(norm.logpdf(data, mu, sigma))
    
    # Initial parameter guess
    initial_guess = [np.mean(data), np.std(data)]
    
    # Minimize negative log likelihood
    result = minimize(neg_log_likelihood, initial_guess)
    return result.x

# Example usage
np.random.seed(42)
data = np.random.normal(loc=5, scale=2, size=1000)
mu_mle, sigma_mle = mle_estimation(data)
print(f"MLE estimates - Mean: {mu_mle:.2f}, Std: {sigma_mle:.2f}")
```

Slide 2: Gradient Descent Implementation

Gradient Descent is an iterative optimization algorithm that finds the minimum of a function by following the direction of steepest descent. This implementation demonstrates batch gradient descent for linear regression, showing how parameters are updated using partial derivatives.

```python
import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for _ in range(self.iterations):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage
X = np.random.randn(100, 2)
y = 3*X[:, 0] + 2*X[:, 1] + 1 + np.random.randn(100)*0.1
gd = GradientDescent().fit(X, y)
print(f"Learned weights: {gd.weights}, bias: {gd.bias}")
```

Slide 3: Normal Distribution Analysis

The Normal Distribution is a cornerstone probability distribution in statistics and machine learning. This implementation provides tools for analyzing and visualizing normal distributions, including probability density calculation and statistical tests.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_normal_distribution(data):
    # Calculate basic statistics
    mean = np.mean(data)
    std = np.std(data)
    
    # Perform normality test
    statistic, p_value = stats.normaltest(data)
    
    # Generate points for theoretical normal distribution
    x = np.linspace(mean - 4*std, mean + 4*std, 100)
    y = stats.norm.pdf(x, mean, std)
    
    # Plot histogram with theoretical distribution
    plt.hist(data, density=True, alpha=0.7, bins=30)
    plt.plot(x, y, 'r-', lw=2, label='Theoretical PDF')
    plt.title('Normal Distribution Analysis')
    plt.legend()
    
    return {
        'mean': mean,
        'std': std,
        'normality_test_statistic': statistic,
        'p_value': p_value
    }

# Example usage
data = np.random.normal(loc=0, scale=1, size=1000)
results = analyze_normal_distribution(data)
print(f"Analysis results: {results}")
plt.show()
```

Slide 4: Sigmoid Function Implementation

The sigmoid function is a crucial activation function in neural networks, mapping any input to a value between 0 and 1. It's particularly important in logistic regression and binary classification problems, serving as a probability mapper.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    # Numerically stable sigmoid implementation
    return np.where(x >= 0, 
                   1 / (1 + np.exp(-x)),
                   np.exp(x) / (1 + np.exp(x)))

def plot_sigmoid():
    x = np.linspace(-10, 10, 1000)
    y = sigmoid(x)
    
    # Derivative of sigmoid
    y_prime = y * (1 - y)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Sigmoid')
    plt.plot(x, y_prime, label='Derivative')
    plt.grid(True)
    plt.legend()
    plt.title('Sigmoid Function and its Derivative')
    return x, y

# Example usage
x_vals, y_vals = plot_sigmoid()
print(f"Sigmoid at x=0: {sigmoid(0)}")
print(f"Sigmoid at x=±∞: {sigmoid(100)}, {sigmoid(-100)}")
plt.show()
```

Slide 5: Correlation Analysis Tools

Correlation analysis is essential for understanding relationships between variables in datasets. This implementation provides comprehensive correlation analysis tools including Pearson, Spearman, and visualization capabilities.

```python
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

class CorrelationAnalyzer:
    def __init__(self, data):
        self.data = data
        
    def compute_all_correlations(self):
        pearson_corr = self.data.corr(method='pearson')
        spearman_corr = self.data.corr(method='spearman')
        
        # Compute p-values matrix
        p_values = pd.DataFrame(np.zeros_like(pearson_corr),
                              columns=pearson_corr.columns,
                              index=pearson_corr.index)
        
        for i in pearson_corr.columns:
            for j in pearson_corr.index:
                _, p_value = stats.pearsonr(self.data[i], self.data[j])
                p_values.loc[i,j] = p_value
                
        return {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'p_values': p_values
        }
    
    def plot_correlation_matrix(self, method='pearson'):
        plt.figure(figsize=(10, 8))
        corr = self.data.corr(method=method)
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'{method.capitalize()} Correlation Matrix')
        plt.show()

# Example usage
np.random.seed(42)
df = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randn(100),
    'C': np.random.randn(100)
})
analyzer = CorrelationAnalyzer(df)
results = analyzer.compute_all_correlations()
analyzer.plot_correlation_matrix()
```

Slide 6: Cosine Similarity Implementation

Cosine similarity measures the cosine of the angle between two non-zero vectors, widely used in recommendation systems and document similarity analysis. This implementation includes optimized vector operations and practical examples.

```python
import numpy as np
from sklearn.preprocessing import normalize

class CosineSimilarity:
    def __init__(self, vectors):
        self.vectors = normalize(vectors)
    
    def compute_similarity_matrix(self):
        # Efficient computation using matrix multiplication
        return np.dot(self.vectors, self.vectors.T)
    
    def get_most_similar(self, vector_idx, n=5):
        if not hasattr(self, 'similarity_matrix'):
            self.similarity_matrix = self.compute_similarity_matrix()
        
        similarities = self.similarity_matrix[vector_idx]
        most_similar = np.argsort(similarities)[-n-1:-1][::-1]
        return [(idx, similarities[idx]) for idx in most_similar]

# Example usage
vectors = np.random.randn(100, 50)  # 100 vectors of dimension 50
cosine_sim = CosineSimilarity(vectors)
sim_matrix = cosine_sim.compute_similarity_matrix()
similar_vectors = cosine_sim.get_most_similar(0)
print(f"Most similar vectors to vector 0: {similar_vectors}")

# Visualize similarity distribution
plt.hist(sim_matrix.flatten(), bins=50)
plt.title('Distribution of Cosine Similarities')
plt.show()
```

Slide 7: Naive Bayes Classifier Implementation

Naive Bayes implements probabilistic classification based on Bayes' theorem with strong independence assumptions. This implementation shows both Gaussian and Multinomial variants with practical examples in text classification and continuous data.

```python
import numpy as np
from scipy.stats import norm

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.parameters = {}
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        
        # Calculate parameters for each class
        for c in self.classes:
            X_c = X[y == c]
            self.parameters[c] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0),
                'prior': len(X_c) / len(X)
            }
    
    def _calculate_likelihood(self, x, mean, var):
        return np.sum(norm.logpdf(x, mean, np.sqrt(var)))
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.parameters[c]['prior'])
                likelihood = self._calculate_likelihood(
                    x, 
                    self.parameters[c]['mean'],
                    self.parameters[c]['var']
                )
                posterior = prior + likelihood
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)

# Example usage
X = np.random.randn(1000, 4)  # 1000 samples, 4 features
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Binary classification

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and evaluate
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.3f}")
```

Slide 8: F1 Score Implementation

F1 Score is a harmonic mean of precision and recall, providing a balanced metric for classification performance. This implementation includes weighted variants and multi-class support with visualization capabilities.

```python
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

class F1ScoreCalculator:
    def __init__(self, average='binary'):
        self.average = average
        
    def calculate_metrics(self, y_true, y_pred):
        if self.average == 'binary':
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            }
        else:
            # Multi-class implementation
            classes = np.unique(y_true)
            f1_scores = []
            
            for cls in classes:
                binary_true = (y_true == cls).astype(int)
                binary_pred = (y_pred == cls).astype(int)
                metrics = self.calculate_metrics(binary_true, binary_pred)
                f1_scores.append(metrics['f1'])
            
            return {
                'f1_per_class': dict(zip(classes, f1_scores)),
                'macro_f1': np.mean(f1_scores),
                'confusion_matrix': confusion_matrix(y_true, y_pred)
            }
    
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

# Example usage
y_true = np.random.randint(0, 2, 1000)
y_pred = np.random.randint(0, 2, 1000)

calculator = F1ScoreCalculator()
metrics = calculator.calculate_metrics(y_true, y_pred)
calculator.plot_confusion_matrix(y_true, y_pred)
print(f"F1 Score: {metrics['f1']:.3f}")
```

Slide 9: ReLU (Rectified Linear Unit) Implementation

ReLU is a widely used activation function in deep learning that outputs the input directly if positive, else outputs zero. This implementation includes the function, its derivative, and variants like Leaky ReLU with visualization and performance analysis.

```python
import numpy as np
import matplotlib.pyplot as plt

class ReLUActivation:
    @staticmethod
    def relu(x, alpha=0.0):
        """Computes ReLU or Leaky ReLU based on alpha parameter"""
        return np.maximum(alpha * x, x)
    
    @staticmethod
    def relu_derivative(x, alpha=0.0):
        """Computes derivative of ReLU or Leaky ReLU"""
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx
    
    def visualize_relu(self, alpha=0.0):
        x = np.linspace(-5, 5, 1000)
        y_relu = self.relu(x, alpha)
        y_derivative = self.relu_derivative(x, alpha)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(x, y_relu, label=f'ReLU (α={alpha})')
        plt.grid(True)
        plt.legend()
        plt.title('ReLU Function')
        
        plt.subplot(1, 2, 2)
        plt.plot(x, y_derivative, label=f'Derivative (α={alpha})')
        plt.grid(True)
        plt.legend()
        plt.title('ReLU Derivative')
        
        plt.tight_layout()
        plt.show()

# Example usage with performance measurement
relu = ReLUActivation()

# Generate random input
x = np.random.randn(1000000)

# Measure performance
import time
start_time = time.time()
y_relu = relu.relu(x)
y_leaky = relu.relu(x, alpha=0.01)
end_time = time.time()

print(f"Processing time: {(end_time - start_time)*1000:.2f}ms")

# Visualize both standard and leaky ReLU
relu.visualize_relu(alpha=0.0)  # Standard ReLU
relu.visualize_relu(alpha=0.01)  # Leaky ReLU
```

Slide 10: Softmax Function Implementation

Softmax transforms a vector of real numbers into a probability distribution, commonly used in multi-class classification. This implementation includes numerical stability optimizations and gradient computation.

```python
import numpy as np

class SoftmaxLayer:
    def __init__(self):
        self.output = None
        
    def forward(self, x):
        """
        Numerically stable softmax implementation
        """
        # Shift input for numerical stability
        shifted_input = x - np.max(x, axis=1, keepdims=True)
        exp_values = np.exp(shifted_input)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output
    
    def backward(self, gradient):
        """
        Compute gradient of softmax
        """
        n_samples = gradient.shape[0]
        jacobian = np.zeros((n_samples, self.output.shape[1], self.output.shape[1]))
        
        for i in range(n_samples):
            current_output = self.output[i].reshape(-1, 1)
            jacobian[i] = np.diagflat(current_output) - np.dot(current_output, current_output.T)
            
        return np.einsum('ijk,ik->ij', jacobian, gradient)

# Example usage
softmax = SoftmaxLayer()

# Generate random logits
batch_size = 3
n_classes = 4
logits = np.random.randn(batch_size, n_classes)

# Forward pass
probabilities = softmax.forward(logits)

# Test properties
print("Probabilities sum to 1:", np.allclose(np.sum(probabilities, axis=1), 1))
print("Shape:", probabilities.shape)
print("\nExample probabilities:\n", probabilities)

# Compute gradients
gradient = np.random.randn(batch_size, n_classes)
gradients = softmax.backward(gradient)
print("\nGradient shape:", gradients.shape)
```

Slide 11: Mean Squared Error (MSE) with L2 Regularization

Mean Squared Error with L2 regularization combines the standard MSE loss function with a penalty term to prevent overfitting. This implementation includes both the loss calculation and gradient computation for optimization.

```python
import numpy as np

class MSEWithRegularization:
    def __init__(self, lambda_reg=0.01):
        self.lambda_reg = lambda_reg
        
    def compute_loss(self, y_true, y_pred, weights):
        """
        Compute MSE loss with L2 regularization
        """
        n_samples = len(y_true)
        mse = np.mean((y_true - y_pred) ** 2)
        l2_reg = self.lambda_reg * np.sum(weights ** 2)
        return mse + l2_reg
    
    def compute_gradients(self, X, y_true, y_pred, weights):
        """
        Compute gradients for both MSE and L2 regularization term
        """
        n_samples = len(y_true)
        # MSE gradient
        mse_grad = -2/n_samples * X.T.dot(y_true - y_pred)
        # L2 regularization gradient
        l2_grad = 2 * self.lambda_reg * weights
        return mse_grad + l2_grad

# Example usage
np.random.seed(42)

# Generate synthetic data
n_samples = 100
n_features = 5
X = np.random.randn(n_samples, n_features)
true_weights = np.random.randn(n_features)
y_true = X.dot(true_weights) + np.random.randn(n_samples) * 0.1

# Initialize weights and loss function
weights = np.random.randn(n_features)
mse_reg = MSEWithRegularization(lambda_reg=0.01)

# Training loop
learning_rate = 0.01
n_epochs = 100
losses = []

for epoch in range(n_epochs):
    # Forward pass
    y_pred = X.dot(weights)
    loss = mse_reg.compute_loss(y_true, y_pred, weights)
    losses.append(loss)
    
    # Backward pass
    gradients = mse_reg.compute_gradients(X, y_true, y_pred, weights)
    weights -= learning_rate * gradients

print(f"Final loss: {losses[-1]:.6f}")
```

Slide 12: K-Means Clustering Implementation

K-means clustering partitions data into k clusters based on feature similarity. This implementation includes the complete algorithm with initialization strategies and convergence monitoring.

```python
import numpy as np
from scipy.spatial.distance import cdist

class KMeansClustering:
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        
    def initialize_centroids(self, X):
        """Initialize centroids using k-means++ algorithm"""
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # Choose first centroid randomly
        centroids = [X[np.random.choice(n_samples)]]
        
        # Choose remaining centroids
        for _ in range(self.n_clusters - 1):
            distances = cdist(X, np.array(centroids))
            min_distances = np.min(distances, axis=1)
            probabilities = min_distances / np.sum(min_distances)
            next_centroid = X[np.random.choice(n_samples, p=probabilities)]
            centroids.append(next_centroid)
            
        return np.array(centroids)
    
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        prev_centroids = None
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            distances = cdist(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            prev_centroids = self.centroids.copy()
            for i in range(self.n_clusters):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = np.mean(cluster_points, axis=0)
                    
            # Check convergence
            if np.allclose(prev_centroids, self.centroids):
                break
                
        # Calculate inertia (within-cluster sum of squares)
        self.inertia_ = np.sum([
            np.sum((X[self.labels == i] - self.centroids[i]) ** 2)
            for i in range(self.n_clusters)
        ])
        
        return self

# Example usage
# Generate synthetic clustered data
np.random.seed(42)
n_samples = 300
X = np.concatenate([
    np.random.normal(0, 1, (n_samples, 2)),
    np.random.normal(4, 1, (n_samples, 2)),
    np.random.normal(-4, 1, (n_samples, 2))
])

# Fit K-means
kmeans = KMeansClustering(n_clusters=3, random_state=42)
kmeans.fit(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
           marker='x', s=200, linewidths=3, color='r')
plt.title('K-means Clustering Results')
plt.show()
```

Slide 13: Linear Regression with Statistical Analysis

Linear regression implementation with comprehensive statistical analysis including confidence intervals, p-values, and R-squared calculation. This version includes robust error handling and diagnostic plots.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class StatisticalLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.p_values = None
        self.std_errors = None
        self.r_squared = None
        
    def fit(self, X, y):
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Compute coefficients using normal equation
        XtX = X.T.dot(X)
        Xty = X.T.dot(y)
        self.coef_ = np.linalg.solve(XtX, Xty)
        
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        
        # Calculate statistical measures
        y_pred = self.predict(X[:, 1:] if self.fit_intercept else X)
        n = X.shape[0]
        p = X.shape[1] - (1 if self.fit_intercept else 0)
        
        # Residuals and R-squared
        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot)
        
        # Standard errors and p-values
        mse = ss_res / (n - p - 1)
        var_coef = mse * np.linalg.inv(XtX)
        self.std_errors = np.sqrt(np.diag(var_coef))
        
        t_stats = self.coef_ / self.std_errors[1:] if self.fit_intercept else self.coef_ / self.std_errors
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
        
        return self
    
    def predict(self, X):
        if self.fit_intercept:
            return self.intercept_ + X.dot(self.coef_)
        return X.dot(self.coef_)
    
    def plot_diagnostics(self, X, y):
        y_pred = self.predict(X)
        residuals = y - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Fitted
        axes[0, 0].scatter(y_pred, residuals)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Fitted values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot')
        
        # Scale-Location
        axes[1, 0].scatter(y_pred, np.sqrt(np.abs(residuals)))
        axes[1, 0].set_xlabel('Fitted values')
        axes[1, 0].set_ylabel('√|Residuals|')
        axes[1, 0].set_title('Scale-Location')
        
        # Actual vs Predicted
        axes[1, 1].scatter(y, y_pred)
        min_val = min(np.min(y), np.min(y_pred))
        max_val = max(np.max(y), np.max(y_pred))
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[1, 1].set_xlabel('Actual values')
        axes[1, 1].set_ylabel('Predicted values')
        axes[1, 1].set_title('Actual vs Predicted')
        
        plt.tight_layout()
        plt.show()

# Example usage
np.random.seed(42)
X = np.random.randn(100, 3)
true_coef = [2, -1, 0.5]
y = X.dot(true_coef) + np.random.randn(100) * 0.1

model = StatisticalLinearRegression()
model.fit(X, y)

print(f"Coefficients: {model.coef_}")
print(f"R-squared: {model.r_squared:.4f}")
print(f"P-values: {model.p_values}")

model.plot_diagnostics(X, y)
```

Slide 14: Support Vector Machine (SVM) Implementation

A custom implementation of Support Vector Machine using Sequential Minimal Optimization (SMO) algorithm for binary classification. This version includes kernel functions and soft margin optimization.

```python
import numpy as np
from numpy.random import rand

class SVM:
    def __init__(self, C=1.0, kernel='linear', max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter
        self.alphas = None
        self.b = None
        self.support_vectors = None
        
    def kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            gamma = 1.0
            return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("Unsupported kernel")
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.alphas = np.zeros(n_samples)
        self.b = 0
        
        # Compute kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel_function(X[i], X[j])
        
        # SMO algorithm
        iter_count = 0
        while iter_count < self.max_iter:
            alpha_pairs_changed = 0
            
            for i in range(n_samples):
                Ei = self.decision_function(X[i]) - y[i]
                
                if ((y[i] * Ei < -0.01 and self.alphas[i] < self.C) or
                    (y[i] * Ei > 0.01 and self.alphas[i] > 0)):
                    
                    # Select second alpha randomly
                    j = i
                    while j == i:
                        j = int(rand() * n_samples)
                    
                    Ej = self.decision_function(X[j]) - y[j]
                    
                    # Save old alphas
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    # Compute bounds
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * K[i,j] - K[i,i] - K[j,j]
                    if eta >= 0:
                        continue
                    
                    # Update alphas
                    self.alphas[j] -= y[j] * (Ei - Ej) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    
                    # Update threshold
                    b1 = self.b - Ei - y[i] * (self.alphas[i] - alpha_i_old) * K[i,i] \
                         - y[j] * (self.alphas[j] - alpha_j_old) * K[i,j]
                    b2 = self.b - Ej - y[i] * (self.alphas[i] - alpha_i_old) * K[i,j] \
                         - y[j] * (self.alphas[j] - alpha_j_old) * K[j,j]
                    
                    self.b = (b1 + b2) / 2
                    alpha_pairs_changed += 1
            
            if alpha_pairs_changed == 0:
                iter_count += 1
            else:
                iter_count = 0
        
        # Store support vectors
        sv_indices = self.alphas > 1e-5
        self.support_vectors = X[sv_indices]
        return self
    
    def decision_function(self, x):
        if self.alphas is None:
            return 0
        result = self.b
        for i, sv in enumerate(self.support_vectors):
            result += self.alphas[i] * self.kernel_function(x, sv)
        return result
    
    def predict(self, X):
        return np.sign(np.array([self.decision_function(x) for x in X]))

# Example usage
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=2, random_state=42)
y = np.where(y == 0, -1, 1)

svm = SVM(C=1.0, kernel='linear')
svm.fit(X, y)

# Plot decision boundary
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='b', label='Class 1')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='r', label='Class -1')
plt.legend()
plt.title('SVM Classification Results')
plt.show()
```

Slide 15: Additional Resources

*   "Deep Learning Book" by Ian Goodfellow et al.: [http://www.deeplearningbook.org](http://www.deeplearningbook.org)
*   "Pattern Recognition and Machine Learning" by Christopher Bishop: Search on Google Scholar
*   "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman: [https://web.stanford.edu/~hastie/ElemStatLearn/](https://web.stanford.edu/~hastie/ElemStatLearn/)
*   "Mathematical Foundations of Machine Learning" on arXiv: [https://arxiv.org/abs/2108.13315](https://arxiv.org/abs/2108.13315)
*   "A Tutorial on Support Vector Machines for Pattern Recognition" by Christopher Burges: Search on Google Scholar

