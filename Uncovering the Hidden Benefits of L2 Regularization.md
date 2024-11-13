## Uncovering the Hidden Benefits of L2 Regularization
Slide 1: Understanding L2 Regularization Fundamentals

L2 regularization, also known as Ridge regularization, adds a penalty term to the loss function proportional to the square of the model parameters. This modification helps control parameter magnitudes and addresses both overfitting and multicollinearity issues in machine learning models.

```python
import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        
    def loss_function(self, X, y, weights):
        # Compute MSE loss with L2 penalty
        predictions = X.dot(weights)
        mse = np.mean((y - predictions) ** 2)
        l2_penalty = self.alpha * np.sum(weights ** 2)
        return mse + l2_penalty
```

Slide 2: Visualizing the Effect of L2 Regularization

L2 regularization transforms the optimization landscape by adding a quadratic penalty term. This visualization demonstrates how different alpha values affect the parameter space and create a unique global minimum, eliminating the ridge-like structure in the loss surface.

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_loss_surface(X, y, alpha_values=[0, 1]):
    theta1 = np.linspace(-2, 2, 100)
    theta2 = np.linspace(-2, 2, 100)
    T1, T2 = np.meshgrid(theta1, theta2)
    
    fig = plt.figure(figsize=(15, 5))
    
    for i, alpha in enumerate(alpha_values):
        Z = np.zeros_like(T1)
        for i in range(len(theta1)):
            for j in range(len(theta2)):
                weights = np.array([T1[i,j], T2[i,j]])
                Z[i,j] = RidgeRegression(alpha).loss_function(X, y, weights)
                
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        ax.plot_surface(T1, T2, Z, cmap='viridis')
        ax.set_title(f'Loss Surface (α={alpha})')
```

Slide 3: Implementing Ridge Regression from Scratch

A complete implementation of Ridge Regression using gradient descent optimization. The code includes parameter initialization, gradient computation with L2 penalty, and iterative weight updates to find the optimal parameters that minimize the regularized loss.

```python
class RidgeRegression:
    def __init__(self, alpha=1.0, learning_rate=0.01, iterations=1000):
        self.alpha = alpha
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.loss_history = []

    def fit(self, X, y):
        # Initialize weights
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        
        for _ in range(self.iterations):
            # Compute predictions
            y_pred = X.dot(self.weights)
            
            # Compute gradients with L2 penalty
            gradients = (-2/len(X)) * X.T.dot(y - y_pred) + \
                       2 * self.alpha * self.weights
            
            # Update weights
            self.weights -= self.lr * gradients
            
            # Store loss
            current_loss = self.loss_function(X, y, self.weights)
            self.loss_history.append(current_loss)
```

Slide 4: Handling Multicollinearity Detection

Before applying L2 regularization, it's crucial to detect multicollinearity in the dataset. This implementation shows how to compute and visualize correlation matrices, variance inflation factors (VIF), and condition numbers to identify collinear features.

```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def detect_multicollinearity(X, threshold=5.0):
    # Compute correlation matrix
    corr_matrix = pd.DataFrame(X).corr()
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = range(X.shape[1])
    vif_data["VIF"] = [variance_inflation_factor(X, i) 
                       for i in range(X.shape[1])]
    
    # Compute condition number
    eigenvals = np.linalg.eigvals(X.T.dot(X))
    condition_number = np.sqrt(np.max(eigenvals) / np.min(eigenvals))
    
    return corr_matrix, vif_data, condition_number
```

Slide 5: Real-world Example - Housing Price Prediction

A practical implementation of Ridge Regression for predicting housing prices, demonstrating how L2 regularization handles multicollinearity among features like square footage, number of rooms, and location-based variables.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and preprocess housing data
def prepare_housing_data():
    # Sample housing data
    data = pd.DataFrame({
        'price': [300k, 400k, ...],
        'sqft': [1500, 2000, ...],
        'rooms': [3, 4, ...],
        'location_score': [8, 7, ...]
    })
    
    X = data.drop('price', axis=1)
    y = data['price']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2)
```

Slide 6: Source Code for Housing Price Prediction Model

Here we implement the complete Ridge Regression model for the housing price prediction, including cross-validation for optimal alpha selection and model evaluation metrics to assess performance.

```python
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def train_housing_model(X_train, X_test, y_train, y_test):
    # Initialize alphas for cross-validation
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    best_alpha = None
    best_score = float('inf')
    
    # Cross-validation for alpha selection
    for alpha in alphas:
        model = RidgeRegression(alpha=alpha)
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        if mse < best_score:
            best_score = mse
            best_alpha = alpha
    
    # Train final model with best alpha
    final_model = RidgeRegression(alpha=best_alpha)
    final_model.fit(X_train, y_train)
    
    return final_model, best_alpha
```

Slide 7: Understanding Parameter Shrinkage

L2 regularization's effect on parameter shrinkage is crucial for model interpretation. This visualization demonstrates how increasing alpha values progressively shrink coefficients towards zero without reaching exact zero, unlike L1 regularization.

```python
def visualize_parameter_shrinkage(X, y, alphas=[0.001, 0.01, 0.1, 1.0, 10.0]):
    plt.figure(figsize=(12, 6))
    coef_paths = []
    
    for alpha in alphas:
        model = RidgeRegression(alpha=alpha)
        model.fit(X, y)
        coef_paths.append(model.weights)
    
    coef_paths = np.array(coef_paths)
    
    for i in range(coef_paths.shape[1]):
        plt.plot(np.log10(alphas), coef_paths[:, i], 
                label=f'Feature {i+1}')
    
    plt.xlabel('log(alpha)')
    plt.ylabel('Coefficient Value')
    plt.title('Ridge Coefficient Paths')
    plt.legend()
    plt.grid(True)
    return plt
```

Slide 8: Geometric Interpretation of L2 Regularization

The geometric interpretation helps understand how L2 regularization constrains the parameter space. This implementation visualizes the interaction between the loss contours and the L2 constraint region, showing the resulting optimal solution.

```python
def plot_geometric_interpretation(X, y, alpha=1.0):
    def loss_contours(theta1, theta2):
        return np.array([[np.sum((y - X.dot(np.array([t1, t2])))**2) 
                         for t1 in theta1] for t2 in theta2])
    
    theta1 = np.linspace(-2, 2, 100)
    theta2 = np.linspace(-2, 2, 100)
    T1, T2 = np.meshgrid(theta1, theta2)
    
    # Plot loss contours
    Z = loss_contours(T1, T2)
    plt.figure(figsize=(10, 10))
    plt.contour(T1, T2, Z, levels=20)
    
    # Plot L2 constraint region
    circle = plt.Circle((0, 0), 1/np.sqrt(alpha), 
                       fill=False, color='red', label='L2 constraint')
    plt.gca().add_artist(circle)
    
    plt.xlabel('θ₁')
    plt.ylabel('θ₂')
    plt.title('Loss Contours with L2 Constraint')
    plt.legend()
    return plt
```

Slide 9: Numerical Stability Improvements

L2 regularization significantly improves numerical stability when dealing with ill-conditioned matrices. This implementation demonstrates how Ridge Regression handles cases where ordinary least squares would fail due to matrix singularity.

```python
def demonstrate_numerical_stability():
    # Create an ill-conditioned matrix
    X = np.array([[1, 1], [1, 1.000001]])
    y = np.array([1, 2])
    
    def condition_number(X, alpha=0):
        # Add L2 regularization to the matrix
        XtX = X.T.dot(X) + alpha * np.eye(X.shape[1])
        eigenvals = np.linalg.eigvals(XtX)
        return np.sqrt(np.max(np.abs(eigenvals)) / 
                      np.min(np.abs(eigenvals)))
    
    alphas = [0, 0.001, 0.01, 0.1, 1.0]
    for alpha in alphas:
        print(f"Condition number (α={alpha}): "
              f"{condition_number(X, alpha):.2e}")
```

Slide 10: Cross-Validation and Model Selection

A comprehensive implementation of k-fold cross-validation for Ridge Regression, including automated alpha selection and stability analysis of the regularization parameter across folds.

```python
from sklearn.model_selection import KFold

class RidgeRegressionCV:
    def __init__(self, alphas=[0.1, 1.0, 10.0], n_folds=5):
        self.alphas = alphas
        self.n_folds = n_folds
        self.best_alpha = None
        self.cv_scores = None
        
    def fit(self, X, y):
        kf = KFold(n_splits=self.n_folds, shuffle=True)
        cv_scores = np.zeros((len(self.alphas), self.n_folds))
        
        for i, alpha in enumerate(self.alphas):
            for j, (train_idx, val_idx) in enumerate(kf.split(X)):
                # Train model
                model = RidgeRegression(alpha=alpha)
                model.fit(X[train_idx], y[train_idx])
                
                # Evaluate
                y_pred = model.predict(X[val_idx])
                cv_scores[i, j] = mean_squared_error(y[val_idx], y_pred)
        
        # Select best alpha
        mean_scores = np.mean(cv_scores, axis=1)
        self.best_alpha = self.alphas[np.argmin(mean_scores)]
        self.cv_scores = cv_scores
        
        return self
```

Slide 11: Comparison with Other Regularization Techniques

A comprehensive comparison between L2, L1, and Elastic Net regularization, demonstrating their effects on parameter estimation and model performance using synthetic data with known multicollinearity patterns.

```python
class RegularizationComparison:
    def __init__(self, n_samples=100, n_features=10):
        self.n_samples = n_samples
        self.n_features = n_features
        
    def generate_multicollinear_data(self):
        # Generate correlated features
        X = np.random.randn(self.n_samples, self.n_features)
        X[:, 1] = X[:, 0] + np.random.normal(0, 0.1, self.n_samples)
        
        # True coefficients
        beta = np.array([1, 0.5] + [0.1] * (self.n_features-2))
        
        # Generate target
        y = X.dot(beta) + np.random.normal(0, 0.1, self.n_samples)
        return X, y, beta
    
    def compare_methods(self, X, y, true_beta):
        models = {
            'Ridge': RidgeRegression(alpha=1.0),
            'Lasso': LassoRegression(alpha=1.0),
            'ElasticNet': ElasticNetRegression(alpha=1.0, l1_ratio=0.5)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X, y)
            results[name] = {
                'coefficients': model.weights,
                'mse': mean_squared_error(y, model.predict(X)),
                'coef_error': np.linalg.norm(model.weights - true_beta)
            }
        
        return results
```

Slide 12: Practical Implementation for High-dimensional Data

Implementation of Ridge Regression optimized for high-dimensional datasets, incorporating efficient matrix operations and memory management techniques for handling large-scale problems.

```python
class ScalableRidgeRegression:
    def __init__(self, alpha=1.0, chunk_size=1000):
        self.alpha = alpha
        self.chunk_size = chunk_size
        self.weights = None
        
    def fit_large_scale(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        
        # Initialize matrices for accumulation
        XtX = np.zeros((n_features, n_features))
        Xty = np.zeros(n_features)
        
        # Process data in chunks
        for i in range(0, len(X), self.chunk_size):
            X_chunk = X[i:i+self.chunk_size]
            y_chunk = y[i:i+self.chunk_size]
            
            # Accumulate gram matrix and cross-product
            XtX += X_chunk.T.dot(X_chunk)
            Xty += X_chunk.T.dot(y_chunk)
        
        # Add regularization term
        XtX += self.alpha * np.eye(n_features)
        
        # Solve using Cholesky decomposition
        L = np.linalg.cholesky(XtX)
        self.weights = np.linalg.solve(L.T, np.linalg.solve(L, Xty))
        
        return self
```

Slide 13: Results Analysis and Metrics

A comprehensive suite of evaluation metrics and visualization tools for assessing Ridge Regression performance, including stability analysis and confidence intervals for coefficient estimates.

```python
class RidgeRegressionAnalyzer:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        
    def compute_metrics(self):
        y_pred = self.model.predict(self.X)
        metrics = {
            'mse': mean_squared_error(self.y, y_pred),
            'r2': r2_score(self.y, y_pred),
            'condition_number': np.linalg.cond(self.X.T.dot(self.X) + 
                                             self.model.alpha * np.eye(self.X.shape[1]))
        }
        return metrics
    
    def coefficient_stability(self, n_bootstrap=1000):
        coef_samples = np.zeros((n_bootstrap, len(self.model.weights)))
        
        for i in range(n_bootstrap):
            # Bootstrap sampling
            indices = np.random.choice(len(self.X), len(self.X))
            X_boot = self.X[indices]
            y_boot = self.y[indices]
            
            # Fit model
            model_boot = RidgeRegression(alpha=self.model.alpha)
            model_boot.fit(X_boot, y_boot)
            coef_samples[i] = model_boot.weights
        
        # Compute confidence intervals
        ci_lower = np.percentile(coef_samples, 2.5, axis=0)
        ci_upper = np.percentile(coef_samples, 97.5, axis=0)
        
        return {'ci_lower': ci_lower, 'ci_upper': ci_upper}
```

Slide 14: Additional Resources

*   "Understanding the Role of L2 Regularization in Neural Networks"
    *   [https://arxiv.org/abs/1811.11124](https://arxiv.org/abs/1811.11124)
*   "A Comprehensive Analysis of Deep Learning Regularization Techniques"
    *   [https://arxiv.org/abs/1901.10867](https://arxiv.org/abs/1901.10867)
*   "Ridge Regression: Biased Estimation for Nonorthogonal Problems"
    *   Search on Google Scholar: "Hoerl Kennard Ridge Regression"
*   "The Geometry of Regularized Linear Models"
    *   Search on Google Scholar: "Geometry Regularization Machine Learning"

Slide 15: Eigenvalue Analysis for L2 Regularization

An implementation demonstrating how L2 regularization affects the eigenvalue spectrum of the feature matrix, providing insights into the stabilization of parameter estimation in the presence of multicollinearity.

```python
def analyze_eigenvalue_spectrum(X, alpha_range=[0, 0.1, 1.0, 10.0]):
    def compute_eigenspectrum(X, alpha):
        # Compute covariance matrix with regularization
        n_features = X.shape[1]
        cov_matrix = X.T.dot(X) + alpha * np.eye(n_features)
        return np.linalg.eigvals(cov_matrix)
    
    plt.figure(figsize=(12, 6))
    for alpha in alpha_range:
        eigenvals = compute_eigenspectrum(X, alpha)
        plt.plot(range(1, len(eigenvals) + 1), 
                np.sort(eigenvals)[::-1], 
                marker='o', 
                label=f'α={alpha}')
    
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue Magnitude')
    plt.title('Eigenvalue Spectrum with Different L2 Penalties')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    return plt
```

Slide 16: Advanced Cross-validation Techniques

Implementation of sophisticated cross-validation methods specifically designed for Ridge Regression, including stratified k-fold and time series cross-validation with proper handling of temporal dependencies.

```python
class AdvancedRidgeCV:
    def __init__(self, alphas=np.logspace(-3, 3, 7)):
        self.alphas = alphas
        self.best_alpha_ = None
        self.best_score_ = None
        
    def time_series_cv(self, X, y, n_splits=5):
        # Time series split
        scores = np.zeros((len(self.alphas), n_splits))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for i, alpha in enumerate(self.alphas):
            for j, (train_idx, val_idx) in enumerate(tscv.split(X)):
                # Ensure temporal order is preserved
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = RidgeRegression(alpha=alpha)
                model.fit(X_train, y_train)
                
                # Compute validation score
                y_pred = model.predict(X_val)
                scores[i, j] = mean_squared_error(y_val, y_pred)
        
        # Select best alpha considering temporal structure
        mean_scores = np.mean(scores, axis=1)
        self.best_alpha_ = self.alphas[np.argmin(mean_scores)]
        self.best_score_ = np.min(mean_scores)
        
        return scores, mean_scores
```

Slide 17: Regularization Path Analysis

A detailed implementation for analyzing and visualizing the regularization path of Ridge Regression, showing how coefficients evolve across different regularization strengths and their statistical significance.

```python
class RegularizationPathAnalyzer:
    def __init__(self, X, y, alphas=np.logspace(-3, 3, 100)):
        self.X = X
        self.y = y
        self.alphas = alphas
        self.coef_paths = None
        self.std_errors = None
        
    def compute_paths(self):
        n_features = self.X.shape[1]
        self.coef_paths = np.zeros((len(self.alphas), n_features))
        self.std_errors = np.zeros_like(self.coef_paths)
        
        for i, alpha in enumerate(self.alphas):
            # Fit model
            model = RidgeRegression(alpha=alpha)
            model.fit(self.X, self.y)
            
            # Store coefficients
            self.coef_paths[i] = model.weights
            
            # Compute standard errors
            sigma2 = np.mean((self.y - model.predict(self.X))**2)
            covar_matrix = np.linalg.inv(self.X.T.dot(self.X) + 
                                       alpha * np.eye(n_features))
            self.std_errors[i] = np.sqrt(np.diag(covar_matrix) * sigma2)
            
        return self.coef_paths, self.std_errors
    
    def plot_paths(self):
        plt.figure(figsize=(12, 6))
        for j in range(self.coef_paths.shape[1]):
            plt.plot(np.log10(self.alphas), 
                    self.coef_paths[:, j], 
                    label=f'Feature {j+1}')
            
        plt.xlabel('log(α)')
        plt.ylabel('Coefficient Value')
        plt.title('Regularization Paths')
        plt.legend()
        plt.grid(True)
        return plt
```

Slide 18: Efficient Implementation for Sparse Data

An optimized implementation of Ridge Regression for sparse matrices, utilizing scipy's sparse matrix operations and specialized solvers for improved computational efficiency with high-dimensional sparse datasets.

```python
from scipy import sparse
from scipy.sparse import linalg

class SparseRidgeRegression:
    def __init__(self, alpha=1.0, solver='cg', max_iter=1000):
        self.alpha = alpha
        self.solver = solver
        self.max_iter = max_iter
        self.weights = None
        
    def fit(self, X, y):
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)
            
        n_features = X.shape[1]
        
        # Construct system matrix A = (X^T X + αI)
        A = X.T.dot(X) + self.alpha * sparse.eye(n_features)
        b = X.T.dot(y)
        
        if self.solver == 'cg':
            # Use conjugate gradient solver
            self.weights, info = linalg.cg(A, b, 
                                         maxiter=self.max_iter)
        else:
            # Use direct solver for smaller problems
            self.weights = linalg.spsolve(A, b)
            
        return self
    
    def predict(self, X):
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)
        return X.dot(self.weights)
```

Slide 19: Model Diagnostics and Validation

Implementation of comprehensive diagnostic tools for Ridge Regression, including influence analysis, residual plots, and validation metrics with confidence intervals.

```python
class RidgeDiagnostics:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.residuals = None
        self.influence = None
        
    def compute_diagnostics(self):
        # Compute predictions and residuals
        y_pred = self.model.predict(self.X)
        self.residuals = self.y - y_pred
        
        # Compute hat matrix diagonals (leverage)
        H = self.X.dot(np.linalg.inv(
            self.X.T.dot(self.X) + 
            self.model.alpha * np.eye(self.X.shape[1])
        )).dot(self.X.T)
        self.influence = np.diag(H)
        
        # Compute standardized residuals
        sigma = np.sqrt(np.mean(self.residuals**2))
        std_residuals = self.residuals / (sigma * np.sqrt(1 - self.influence))
        
        return {
            'residuals': self.residuals,
            'std_residuals': std_residuals,
            'influence': self.influence,
            'cook_distance': self._compute_cooks_distance()
        }
    
    def _compute_cooks_distance(self):
        p = self.X.shape[1]
        std_residuals = self.residuals / np.sqrt(np.mean(self.residuals**2))
        return (std_residuals**2 * self.influence) / (p * (1 - self.influence))
    
    def plot_diagnostics(self):
        diagnostics = self.compute_diagnostics()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Residual plot
        axes[0,0].scatter(self.model.predict(self.X), diagnostics['residuals'])
        axes[0,0].set_xlabel('Predicted Values')
        axes[0,0].set_ylabel('Residuals')
        axes[0,0].set_title('Residual Plot')
        
        # QQ plot
        from scipy import stats
        stats.probplot(diagnostics['std_residuals'], dist="norm", plot=axes[0,1])
        axes[0,1].set_title('Normal Q-Q Plot')
        
        # Leverage plot
        axes[1,0].scatter(range(len(diagnostics['influence'])), 
                         diagnostics['influence'])
        axes[1,0].set_xlabel('Observation Index')
        axes[1,0].set_ylabel('Leverage')
        axes[1,0].set_title('Leverage Plot')
        
        # Cook's distance plot
        axes[1,1].scatter(range(len(diagnostics['cook_distance'])), 
                         diagnostics['cook_distance'])
        axes[1,1].set_xlabel('Observation Index')
        axes[1,1].set_ylabel("Cook's Distance")
        axes[1,1].set_title("Cook's Distance Plot")
        
        plt.tight_layout()
        return plt
```

Slide 20: Feature Selection with Ridge Regression

Implementation of a hybrid approach combining Ridge Regression with feature selection techniques, demonstrating how to identify and select the most important features while maintaining regularization benefits.

```python
class RidgeFeatureSelector:
    def __init__(self, alpha=1.0, threshold=0.1):
        self.alpha = alpha
        self.threshold = threshold
        self.selected_features = None
        self.importance_scores = None
        
    def select_features(self, X, y):
        # Fit Ridge model
        model = RidgeRegression(alpha=self.alpha)
        model.fit(X, y)
        
        # Compute standardized coefficients
        std_coef = model.weights * np.std(X, axis=0)
        
        # Calculate importance scores
        self.importance_scores = np.abs(std_coef)
        
        # Select features above threshold
        self.selected_features = np.where(
            self.importance_scores > self.threshold)[0]
        
        # Create feature importance summary
        feature_importance = pd.DataFrame({
            'Feature': range(X.shape[1]),
            'Importance': self.importance_scores
        }).sort_values('Importance', ascending=False)
        
        return self.selected_features, feature_importance
    
    def transform(self, X):
        if self.selected_features is None:
            raise ValueError("Must call select_features before transform")
        return X[:, self.selected_features]
```

Slide 21: Real-world Application - Financial Time Series

Implementation of Ridge Regression for financial time series prediction, handling temporal dependencies and incorporating multiple technical indicators while addressing multicollinearity among financial features.

```python
class FinancialRidgeRegression:
    def __init__(self, alpha=1.0, lookback_period=30):
        self.alpha = alpha
        self.lookback = lookback_period
        self.model = None
        self.scaler = None
        
    def create_features(self, prices):
        # Technical indicators
        features = pd.DataFrame()
        
        # Moving averages
        features['MA5'] = prices.rolling(5).mean()
        features['MA20'] = prices.rolling(20).mean()
        
        # Relative strength index
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        features['RSI'] = 100 - (100 / (1 + gain/loss))
        
        # Volatility
        features['Volatility'] = prices.rolling(20).std()
        
        features = features.dropna()
        return features
    
    def prepare_data(self, features, target, train_size=0.8):
        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(features)):
            X.append(features[i-self.lookback:i].values.flatten())
            y.append(target[i])
            
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        split = int(train_size * len(X))
        return (X[:split], X[split:], 
                y[:split], y[split:])
    
    def fit_predict(self, prices):
        # Create features
        features = self.create_features(prices)
        returns = prices.pct_change().shift(-1)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            features, returns)
        
        # Train model
        self.model = RidgeRegression(alpha=self.alpha)
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        return {
            'predictions': y_pred,
            'actual': y_test,
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
```

Slide 22: Distributed Implementation of Ridge Regression

A scalable implementation of Ridge Regression designed for distributed computing environments, utilizing parallel processing for large-scale datasets while maintaining numerical stability.

```python
class DistributedRidgeRegression:
    def __init__(self, alpha=1.0, n_partitions=4):
        self.alpha = alpha
        self.n_partitions = n_partitions
        self.weights = None
        
    def _process_partition(self, X_chunk, y_chunk):
        # Compute local sufficient statistics
        XtX = X_chunk.T.dot(X_chunk)
        Xty = X_chunk.T.dot(y_chunk)
        return XtX, Xty
    
    def fit(self, X, y):
        n_features = X.shape[1]
        chunk_size = len(X) // self.n_partitions
        
        # Initialize accumulators
        global_XtX = np.zeros((n_features, n_features))
        global_Xty = np.zeros(n_features)
        
        # Process data in parallel
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            futures = []
            
            for i in range(self.n_partitions):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < self.n_partitions-1 \
                         else len(X)
                
                X_chunk = X[start_idx:end_idx]
                y_chunk = y[start_idx:end_idx]
                
                futures.append(
                    executor.submit(self._process_partition, 
                                  X_chunk, y_chunk)
                )
            
            # Aggregate results
            for future in futures:
                XtX_chunk, Xty_chunk = future.result()
                global_XtX += XtX_chunk
                global_Xty += Xty_chunk
        
        # Add regularization
        global_XtX += self.alpha * np.eye(n_features)
        
        # Solve system
        self.weights = np.linalg.solve(global_XtX, global_Xty)
        
        return self
```

Slide 23: Additional Resources

*   "On Cross-Validation and Ridge Regression under High Dimensionality"
    *   [https://arxiv.org/abs/1902.10416](https://arxiv.org/abs/1902.10416)
*   "Distributed Ridge Regression in High Dimensions"
    *   Search on Google Scholar: "Distributed Ridge Regression High Dimensions"
*   "A Comparison of Regularization Methods in Deep Learning"
    *   [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
*   "Statistical Properties of Ridge Estimators"
    *   Search on Google Scholar: "Ridge Regression Asymptotic Properties"
*   "Modern Applications of Ridge Regression in Machine Learning"
    *   Search on Google Scholar: "Ridge Regression Deep Learning Applications"

