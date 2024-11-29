## Benefits of L1 Regularization
Slide 1: L1 Regularization Fundamentals

L1 regularization, also known as Lasso regularization, adds the absolute value of model parameters to the loss function, promoting sparsity by driving some coefficients exactly to zero, effectively performing feature selection during model training.

```python
# Mathematical representation of L1 regularization
'''
Cost Function with L1:
$$J(\theta) = \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\sum_{j=1}^{n}|\theta_j|$$

Where:
- m: number of training examples
- n: number of features
- λ: regularization parameter
- θ: model parameters
'''
```

Slide 2: Implementing L1 Regularization from Scratch

A fundamental implementation of L1 regularization demonstrates how it modifies the basic linear regression algorithm by incorporating the absolute value penalty term during optimization.

```python
import numpy as np

class L1RegularizedRegression:
    def __init__(self, lambda_param=0.1):
        self.lambda_param = lambda_param
        self.weights = None
        
    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for _ in range(epochs):
            # Compute predictions
            y_pred = np.dot(X, self.weights)
            
            # Compute gradients with L1 penalty
            gradients = (1/n_samples) * X.T.dot(y_pred - y)
            l1_penalty = self.lambda_param * np.sign(self.weights)
            
            # Update weights
            self.weights -= learning_rate * (gradients + l1_penalty)
            
    def predict(self, X):
        return np.dot(X, self.weights)
```

Slide 3: Real-world Example - Feature Selection in High-dimensional Data

L1 regularization excels in scenarios with high-dimensional data where feature selection is crucial. This implementation demonstrates its application to a synthetic dataset with intentionally redundant features.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate synthetic data with redundant features
X, y = make_regression(n_samples=1000, n_features=100, 
                      n_informative=10, noise=0.1, random_state=42)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Train model
model = L1RegularizedRegression(lambda_param=0.1)
model.fit(X_train, y_train)

# Analyze feature importance
important_features = np.where(np.abs(model.weights) > 0.1)[0]
print(f"Selected features: {len(important_features)} out of {len(model.weights)}")
```

Slide 4: Visualizing L1 vs L2 Regularization Effects

Understanding the geometric interpretation of L1 regularization helps explain why it promotes sparsity compared to L2 regularization, which tends to distribute weights more evenly.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_regularization_contours():
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # L1 contour
    L1 = np.abs(X) + np.abs(Y)
    
    # L2 contour
    L2 = np.sqrt(X**2 + Y**2)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.contour(X, Y, L1, levels=[1])
    plt.title('L1 Regularization\nContour')
    plt.grid(True)
    
    plt.subplot(122)
    plt.contour(X, Y, L2, levels=[1])
    plt.title('L2 Regularization\nContour')
    plt.grid(True)
    
    plt.show()

plot_regularization_contours()
```

Slide 5: Comparison with Different Regularization Parameters

The strength of L1 regularization can be controlled through the lambda parameter, demonstrating how different values affect feature selection and model performance.

```python
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def compare_lambda_effects(X_train, X_test, y_train, y_test):
    lambdas = [0.001, 0.01, 0.1, 1.0, 10.0]
    non_zero_features = []
    test_errors = []
    
    for lambda_param in lambdas:
        model = L1RegularizedRegression(lambda_param=lambda_param)
        model.fit(X_train, y_train)
        
        # Count non-zero features
        non_zero = np.sum(np.abs(model.weights) > 1e-10)
        non_zero_features.append(non_zero)
        
        # Calculate test error
        y_pred = model.predict(X_test)
        test_errors.append(mean_squared_error(y_test, y_pred))
    
    plt.figure(figsize=(10, 5))
    plt.plot(lambdas, non_zero_features, 'b-', label='Non-zero features')
    plt.plot(lambdas, test_errors, 'r--', label='Test MSE')
    plt.xscale('log')
    plt.legend()
    plt.xlabel('Lambda')
    plt.show()
```

Slide 6: Cross-Validation for Optimal Lambda Selection

Cross-validation is essential for finding the optimal L1 regularization strength that balances between underfitting and overfitting, ensuring the model generalizes well to unseen data.

```python
from sklearn.model_selection import KFold
import numpy as np

def l1_cross_validation(X, y, lambda_range, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv_scores = {lambda_val: [] for lambda_val in lambda_range}
    
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        for lambda_val in lambda_range:
            model = L1RegularizedRegression(lambda_param=lambda_val)
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            mse = np.mean((y_val_fold - y_pred) ** 2)
            cv_scores[lambda_val].append(mse)
    
    # Calculate mean scores
    mean_scores = {k: np.mean(v) for k, v in cv_scores.items()}
    return mean_scores
```

Slide 7: Handling Multicollinearity with L1 Regularization

L1 regularization effectively addresses multicollinearity by selecting one representative feature from groups of highly correlated predictors, improving model interpretability and stability.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def analyze_multicollinearity():
    # Generate correlated features
    n_samples = 1000
    X1 = np.random.normal(0, 1, n_samples)
    X2 = X1 + np.random.normal(0, 0.1, n_samples)  # Highly correlated with X1
    X3 = np.random.normal(0, 1, n_samples)
    
    # Create target variable
    y = 2*X1 + 0.5*X3 + np.random.normal(0, 0.1, n_samples)
    
    # Combine features
    X = np.column_stack([X1, X2, X3])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit model with L1 regularization
    model = L1RegularizedRegression(lambda_param=0.1)
    model.fit(X_scaled, y)
    
    print("Feature weights:", model.weights)
    return model.weights
```

Slide 8: Elastic Net: Combining L1 and L2 Regularization

The Elastic Net combines L1 and L2 regularization to leverage the benefits of both approaches, particularly useful when dealing with grouped features or when pure L1 might be too aggressive.

```python
class ElasticNetRegression:
    def __init__(self, l1_ratio=0.5, alpha=1.0):
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.weights = None
    
    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for _ in range(epochs):
            y_pred = np.dot(X, self.weights)
            
            # Combined L1 and L2 gradients
            l1_term = self.alpha * self.l1_ratio * np.sign(self.weights)
            l2_term = self.alpha * (1 - self.l1_ratio) * self.weights
            
            gradients = (1/n_samples) * X.T.dot(y_pred - y)
            self.weights -= learning_rate * (gradients + l1_term + l2_term)
    
    def predict(self, X):
        return np.dot(X, self.weights)
```

Slide 9: Sparse Recovery with L1 Regularization

L1 regularization's ability to recover sparse signals makes it particularly valuable in compressed sensing and signal processing applications where the true underlying representation is known to be sparse.

```python
def sparse_signal_recovery():
    # Generate sparse signal
    n_features = 100
    true_weights = np.zeros(n_features)
    true_weights[np.random.choice(n_features, 5, replace=False)] = np.random.randn(5)
    
    # Generate observations with noise
    X = np.random.randn(200, n_features)
    y = np.dot(X, true_weights) + np.random.normal(0, 0.1, 200)
    
    # Recover sparse signal using L1
    model = L1RegularizedRegression(lambda_param=0.1)
    model.fit(X, y)
    
    # Compare recovery
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.stem(true_weights, label='True')
    plt.title('True Sparse Signal')
    plt.subplot(122)
    plt.stem(model.weights, label='Recovered')
    plt.title('L1 Recovered Signal')
    plt.tight_layout()
    plt.show()
```

Slide 10: Gradient Computation with L1 Regularization

L1 regularization requires special attention during gradient computation due to the non-differentiability of the absolute value function at zero, necessitating subgradient methods for optimization.

```python
import numpy as np

def l1_gradient_computation():
    # Example of subgradient computation for L1 regularization
    def compute_subgradient(weights, lambda_param):
        subgradients = np.zeros_like(weights)
        
        for i, w in enumerate(weights):
            if w > 0:
                subgradients[i] = lambda_param
            elif w < 0:
                subgradients[i] = -lambda_param
            else:
                # At w = 0, subgradient is in [-lambda, lambda]
                subgradients[i] = lambda_param * np.random.uniform(-1, 1)
        
        return subgradients
    
    # Example usage
    weights = np.array([-1.0, 0.0, 2.0])
    lambda_param = 0.1
    subgrads = compute_subgradient(weights, lambda_param)
    return subgrads
```

Slide 11: Feature Importance Analysis with L1 Regularization

L1 regularization provides a natural way to assess feature importance by examining the magnitude of the learned coefficients, offering insights into which features are most relevant for prediction.

```python
def analyze_feature_importance(model, feature_names):
    # Sort features by absolute weight magnitude
    feature_weights = list(zip(feature_names, model.weights))
    sorted_weights = sorted(feature_weights, 
                          key=lambda x: abs(x[1]), 
                          reverse=True)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    features, weights = zip(*sorted_weights)
    plt.bar(features, np.abs(weights))
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importance from L1 Regularization')
    plt.tight_layout()
    
    # Print top features
    print("\nTop 5 most important features:")
    for feature, weight in sorted_weights[:5]:
        print(f"{feature}: {weight:.4f}")
```

Slide 12: Comparison with Other Feature Selection Methods

L1 regularization offers an embedded feature selection approach that can be compared with other methods like filter-based or wrapper methods in terms of computational efficiency and effectiveness.

```python
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score
import time

def compare_feature_selection_methods(X, y, k=10):
    results = {}
    
    # L1 Regularization
    start_time = time.time()
    l1_model = L1RegularizedRegression(lambda_param=0.1)
    l1_model.fit(X, y)
    l1_selected = np.argsort(np.abs(l1_model.weights))[-k:]
    results['L1'] = {
        'time': time.time() - start_time,
        'score': r2_score(y, l1_model.predict(X)),
        'features': l1_selected
    }
    
    # Filter method (F-regression)
    start_time = time.time()
    selector = SelectKBest(f_regression, k=k)
    selector.fit(X, y)
    results['Filter'] = {
        'time': time.time() - start_time,
        'score': r2_score(y, selector.inverse_transform(selector.transform(X))),
        'features': selector.get_support(indices=True)
    }
    
    return results
```

Slide 13: Additional Resources

*   "The Lasso Method for Variable Selection in the Cox Model" - [https://arxiv.org/abs/1805.10549](https://arxiv.org/abs/1805.10549)
*   "An Introduction to Statistical Learning with Applications in R" - Search on Google Scholar for comprehensive L1 regularization coverage
*   "Least Angle Regression" - [https://arxiv.org/abs/math/0406456](https://arxiv.org/abs/math/0406456)
*   "Strong Rules for Discarding Predictors in Lasso-type Problems" - [https://arxiv.org/abs/1011.2234](https://arxiv.org/abs/1011.2234)
*   "Regularization Paths for Cox's Proportional Hazards Model via Coordinate Descent" - Search on Google for implementation details

