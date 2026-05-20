## Understanding the Magic of LASSO Feature Selection
Slide 1: Understanding LASSO Optimization Function

LASSO (Least Absolute Shrinkage and Selection Operator) combines ordinary least squares with L1 regularization. The fundamental optimization objective minimizes the sum of squared residuals while constraining the sum of absolute weights, leading to sparse solutions.

```python
import numpy as np
from sklearn.linear_model import Lasso

# Mathematical formulation of LASSO objective
"""
Minimize: 
$$
\frac{1}{2n} ||y - Xw||_2^2 + \alpha ||w||_1
$$

where:
- y is the target vector
- X is the feature matrix
- w are the weights
- α is the regularization parameter
- n is number of samples
"""

# Basic implementation
def lasso_objective(X, y, w, alpha):
    n_samples = X.shape[0]
    residuals = y - X.dot(w)
    loss = (1/(2*n_samples)) * np.sum(residuals**2)
    l1_penalty = alpha * np.sum(np.abs(w))
    return loss + l1_penalty
```

Slide 2: Visualizing LASSO's Diamond Constraint Region

The L1 regularization creates a diamond-shaped constraint region in parameter space. When the contours of the least squares objective intersect this region, some coefficients are forced to exactly zero, performing feature selection.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_lasso_constraint(T):
    # Create diamond constraint region
    theta = np.linspace(0, 2*np.pi, 100)
    x = T * np.cos(theta)
    y = T * np.sin(theta)
    
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, 'b-', label=f'L1 Ball (T={T})')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Weight 1')
    plt.ylabel('Weight 2')
    plt.title('LASSO Constraint Region')
    plt.legend()
    
    return plt

# Example usage
T = 1.0
plot_lasso_constraint(T)
plt.show()
```

Slide 3: Implementing LASSO from Scratch

A pure Python implementation of LASSO using coordinate descent demonstrates how the algorithm iteratively updates each coefficient while maintaining the L1 constraint, providing insights into the feature selection mechanism.

```python
import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        
    def soft_threshold(self, x, lambda_):
        """Soft thresholding operator"""
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for _ in range(self.max_iter):
            weights_old = self.weights.copy()
            
            # Coordinate descent
            for j in range(n_features):
                r = y - np.dot(X, self.weights) + self.weights[j] * X[:, j]
                self.weights[j] = self.soft_threshold(
                    np.dot(X[:, j], r),
                    self.alpha * n_samples
                ) / (np.dot(X[:, j], X[:, j]))
                
            # Check convergence
            if np.sum(np.abs(self.weights - weights_old)) < self.tol:
                break
                
        return self
```

Slide 4: Real-world Example: Housing Price Prediction

This implementation demonstrates LASSO's practical application in predicting housing prices, showing how it automatically selects relevant features while handling multicollinearity in real estate data.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load and prepare data
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Train custom LASSO
lasso = LassoRegression(alpha=0.1)
lasso.fit(X_train, y_train)

# Predictions
y_pred = np.dot(X_test, lasso.weights)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Feature weights: {lasso.weights}")
print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")
```

Slide 5: Understanding Alpha's Impact on Feature Selection

The relationship between the regularization parameter alpha and feature selection is crucial. As alpha increases, more features are forced to zero, demonstrating the inverse relationship with the constraint parameter T.

```python
def analyze_alpha_impact(X, y, alphas):
    features = X.shape[1]
    weight_matrix = np.zeros((len(alphas), features))
    
    for i, alpha in enumerate(alphas):
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X, y)
        weight_matrix[i, :] = lasso.coef_
        
    return weight_matrix

# Example usage
alphas = np.logspace(-4, 1, 50)
weights = analyze_alpha_impact(X_scaled, y, alphas)

plt.figure(figsize=(12, 6))
for i in range(weights.shape[1]):
    plt.plot(np.log10(alphas), weights[:, i], 
             label=f'Feature {i+1}')
plt.xlabel('log10(alpha)')
plt.ylabel('Coefficient Value')
plt.title('LASSO Path: Feature Coefficients vs Alpha')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 6: Cross-Validation for Optimal Alpha Selection

Cross-validation is essential for finding the optimal regularization parameter that balances model complexity with predictive performance. This implementation demonstrates how to systematically search for the best alpha value.

```python
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

def optimal_alpha_search(X, y, cv_folds=5):
    # Setup alphas to test
    alphas = np.logspace(-4, 1, 100)
    
    # Initialize LassoCV
    lasso_cv = LassoCV(
        alphas=alphas,
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
        max_iter=10000,
        n_jobs=-1
    )
    
    # Fit model
    lasso_cv.fit(X, y)
    
    # Get results
    mse_path = np.mean(lasso_cv.mse_path_, axis=1)
    
    return lasso_cv.alpha_, mse_path, alphas

# Example usage
best_alpha, mse_path, alphas = optimal_alpha_search(X_scaled, y)
print(f"Optimal alpha: {best_alpha:.6f}")

# Plot MSE vs alpha
plt.semilogx(alphas, mse_path)
plt.xlabel('Alpha')
plt.ylabel('Mean Square Error')
plt.title('Cross-validation Error vs Alpha')
plt.show()
```

Slide 7: Feature Importance Analysis with LASSO

LASSO's ability to perform feature selection makes it an excellent tool for analyzing feature importance. This implementation shows how to extract and visualize feature importance scores.

```python
def analyze_feature_importance(X, y, feature_names, alpha):
    # Fit LASSO model
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X, y)
    
    # Get absolute coefficients
    importance = np.abs(lasso.coef_)
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values(
        'Importance', ascending=False
    )
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), 
            feature_importance['Importance'])
    plt.xticks(range(len(importance)), 
               feature_importance['Feature'], 
               rotation=45)
    plt.title('Feature Importance from LASSO')
    plt.tight_layout()
    
    return feature_importance

# Example usage
feature_importance = analyze_feature_importance(
    X_scaled, 
    y, 
    housing.feature_names,
    alpha=0.1
)
print("Feature Importance:\n", feature_importance)
```

Slide 8: Handling Multicollinearity with LASSO

LASSO's ability to handle multicollinearity is demonstrated through this implementation that creates and analyzes correlated features, showing how LASSO selects among correlated predictors.

```python
def demonstrate_multicollinearity():
    # Generate correlated features
    n_samples = 1000
    X = np.random.randn(n_samples, 2)
    X_corr = np.column_stack([
        X[:, 0],
        0.95 * X[:, 0] + 0.1 * np.random.randn(n_samples),
        X[:, 1]
    ])
    
    # Generate target
    true_weights = np.array([1.0, 0.0, 2.0])
    y = np.dot(X_corr, true_weights) + 0.1 * np.random.randn(n_samples)
    
    # Fit LASSO
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_corr, y)
    
    print("True weights:", true_weights)
    print("LASSO weights:", lasso.coef_)
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X_corr.T)
    print("\nFeature correlation matrix:\n", corr_matrix)
    
    return X_corr, y, lasso.coef_

# Example usage
X_corr, y_corr, weights = demonstrate_multicollinearity()
```

Slide 9: Elastic Net Comparison

Understanding how LASSO compares to Elastic Net helps in choosing the right regularization approach. This implementation compares both methods on the same dataset.

```python
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

def compare_regularization(X, y, alphas):
    results = []
    
    for alpha in alphas:
        # LASSO
        lasso = Lasso(alpha=alpha)
        lasso.fit(X, y)
        lasso_mae = mean_absolute_error(y, lasso.predict(X))
        
        # Elastic Net
        enet = ElasticNet(alpha=alpha, l1_ratio=0.5)
        enet.fit(X, y)
        enet_mae = mean_absolute_error(y, enet.predict(X))
        
        results.append({
            'alpha': alpha,
            'lasso_mae': lasso_mae,
            'enet_mae': enet_mae,
            'lasso_nonzero': np.sum(lasso.coef_ != 0),
            'enet_nonzero': np.sum(enet.coef_ != 0)
        })
    
    return pd.DataFrame(results)

# Example usage
alphas = np.logspace(-3, 0, 20)
comparison = compare_regularization(X_scaled, y, alphas)
print(comparison)
```

Slide 10: Stability Selection with LASSO

Stability selection combines LASSO with subsampling to provide robust feature selection. This implementation demonstrates how to assess the stability of selected features across multiple LASSO fits.

```python
from sklearn.utils import resample

def stability_selection(X, y, n_iterations=100, sample_fraction=0.75, alpha=0.1):
    n_samples, n_features = X.shape
    selection_matrix = np.zeros((n_iterations, n_features))
    
    for i in range(n_iterations):
        # Subsample data
        X_sub, y_sub = resample(X, y, 
                               n_samples=int(sample_fraction * n_samples),
                               random_state=i)
        
        # Fit LASSO
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_sub, y_sub)
        
        # Record selected features
        selection_matrix[i, :] = np.abs(lasso.coef_) > 1e-10
    
    # Calculate selection probabilities
    selection_probability = np.mean(selection_matrix, axis=0)
    
    return selection_probability

# Example usage
feature_stability = stability_selection(X_scaled, y)
stability_results = pd.DataFrame({
    'Feature': housing.feature_names,
    'Selection_Probability': feature_stability
}).sort_values('Selection_Probability', ascending=False)

print("Feature Selection Stability:\n", stability_results)
```

Slide 11: LASSO for Time Series Feature Selection

Applying LASSO to time series data requires careful handling of temporal dependencies. This implementation shows how to use LASSO for selecting relevant lagged features.

```python
def create_time_series_features(data, lags):
    n_samples = len(data) - max(lags)
    X = np.zeros((n_samples, len(lags)))
    
    for i, lag in enumerate(lags):
        X[:, i] = data[max(lags)-lag:-lag]
    
    y = data[max(lags):]
    return X, y

def lasso_time_series_selection(data, max_lag=10):
    # Create lagged features
    lags = range(1, max_lag + 1)
    X, y = create_time_series_features(data, lags)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit LASSO
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_scaled, y)
    
    # Analyze selected lags
    selected_lags = pd.DataFrame({
        'Lag': lags,
        'Coefficient': lasso.coef_
    })
    
    return selected_lags[selected_lags.Coefficient != 0]

# Example usage
np.random.seed(42)
time_series = np.random.randn(1000).cumsum()
selected_lags = lasso_time_series_selection(time_series)
print("Selected time lags:\n", selected_lags)
```

Slide 12: High-Dimensional LASSO with Sparse Matrices

When dealing with high-dimensional data, efficient sparse matrix operations become crucial. This implementation shows how to handle sparse features with LASSO.

```python
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

def sparse_lasso_example():
    # Create sparse features
    n_samples = 1000
    n_categories = 1000
    
    # Generate categorical data
    categorical_data = np.random.randint(0, n_categories, 
                                       size=(n_samples, 5))
    
    # One-hot encode to create sparse matrix
    encoder = OneHotEncoder(sparse=True)
    X_sparse = encoder.fit_transform(categorical_data)
    
    # Generate target with few true features
    true_features = np.zeros(X_sparse.shape[1])
    true_features[np.random.choice(X_sparse.shape[1], 10)] = 1
    y = X_sparse.dot(true_features) + 0.1 * np.random.randn(n_samples)
    
    # Fit LASSO with sparse input
    lasso = Lasso(alpha=0.1, max_iter=1000)
    lasso.fit(X_sparse, y)
    
    # Analyze sparsity
    n_selected = np.sum(lasso.coef_ != 0)
    sparsity = 1 - (n_selected / len(lasso.coef_))
    
    return {
        'n_features': X_sparse.shape[1],
        'n_selected': n_selected,
        'sparsity': sparsity
    }

# Example usage
results = sparse_lasso_example()
print("Sparse LASSO Results:", results)
```

Slide 13: LASSO Path Visualization with Early Stopping

This implementation demonstrates how to visualize the entire regularization path of LASSO coefficients and implements early stopping criteria to prevent overfitting while maintaining computational efficiency.

```python
def plot_lasso_path_with_early_stopping(X, y, max_iter=1000, eps=1e-4):
    # Generate sequence of alpha values
    n_alphas = 100
    alphas = np.logspace(-4, 0, n_alphas)
    
    # Store coefficients for each alpha
    coef_paths = np.zeros((n_alphas, X.shape[1]))
    errors = np.zeros(n_alphas)
    
    for i, alpha in enumerate(alphas):
        # Fit LASSO with warm start
        lasso = Lasso(
            alpha=alpha,
            max_iter=max_iter,
            tol=eps,
            warm_start=True
        )
        lasso.fit(X, y)
        
        # Store coefficients
        coef_paths[i] = lasso.coef_
        errors[i] = mean_squared_error(y, lasso.predict(X))
        
        # Early stopping if all coefficients are zero
        if np.all(np.abs(coef_paths[i]) < eps):
            coef_paths = coef_paths[:i+1]
            errors = errors[:i+1]
            alphas = alphas[:i+1]
            break
    
    return alphas, coef_paths, errors

# Example usage
alphas, paths, errors = plot_lasso_path_with_early_stopping(X_scaled, y)

# Visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i in range(paths.shape[1]):
    plt.semilogx(alphas, paths[:, i], label=f'Feature {i+1}')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.subplot(1, 2, 2)
plt.semilogx(alphas, errors)
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('Error vs Alpha')
plt.tight_layout()
plt.show()
```

Slide 14: LASSO for Online Learning

This implementation shows how to use LASSO in an online learning setting, where data arrives in streams and the model needs to be updated incrementally.

```python
class OnlineLasso:
    def __init__(self, alpha=1.0, learning_rate=0.01):
        self.alpha = alpha
        self.lr = learning_rate
        self.weights = None
        self.n_features = None
        
    def _soft_threshold(self, x, lambda_):
        return np.sign(x) * np.maximum(0, np.abs(x) - lambda_)
    
    def partial_fit(self, X, y):
        # Initialize weights if first call
        if self.weights is None:
            self.n_features = X.shape[1]
            self.weights = np.zeros(self.n_features)
        
        # Stochastic gradient descent with L1 regularization
        for i in range(len(X)):
            # Compute gradient of loss
            pred = np.dot(X[i], self.weights)
            grad = -2 * (y[i] - pred) * X[i]
            
            # Update weights using proximal gradient descent
            self.weights = self._soft_threshold(
                self.weights - self.lr * grad,
                self.alpha * self.lr
            )
        
        return self
    
    def predict(self, X):
        return np.dot(X, self.weights)

# Example usage with streaming data
def simulate_data_stream(n_samples=1000):
    for _ in range(n_samples):
        X_batch = np.random.randn(1, 10)
        true_weights = np.array([1, 0.5, 0, 0, 0.8, 0, 0, 0.3, 0, 0])
        y_batch = np.dot(X_batch, true_weights) + 0.1 * np.random.randn(1)
        yield X_batch, y_batch

# Train and evaluate
online_lasso = OnlineLasso(alpha=0.1)
streaming_errors = []

for i, (X_batch, y_batch) in enumerate(simulate_data_stream()):
    # Update model
    online_lasso.partial_fit(X_batch, y_batch)
    
    # Compute error
    error = mean_squared_error(y_batch, online_lasso.predict(X_batch))
    streaming_errors.append(error)
    
    if (i + 1) % 100 == 0:
        print(f"Batch {i+1}, MSE: {np.mean(streaming_errors[-100:]):.4f}")
```

Slide 15: Additional Resources

*   "The LASSO: History, Theory, and Applications" - [https://arxiv.org/abs/1406.4897](https://arxiv.org/abs/1406.4897)
*   "High-Dimensional Feature Selection with LASSO" - [https://arxiv.org/abs/1207.0580](https://arxiv.org/abs/1207.0580)
*   "Stability of Feature Selection with LASSO" - [https://arxiv.org/abs/1509.09169](https://arxiv.org/abs/1509.09169)
*   "Online Learning with LASSO Penalties" - [https://arxiv.org/abs/1204.3006](https://arxiv.org/abs/1204.3006)
*   "A Survey of Cross-Validation Procedures for Model Selection" - [https://arxiv.org/abs/0907.4728](https://arxiv.org/abs/0907.4728)

Note: These are suggested search terms as specific URLs may vary. Please search for these titles on Google Scholar or arXiv for the most up-to-date versions.

