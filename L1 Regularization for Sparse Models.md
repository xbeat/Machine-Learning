## L1 Regularization for Sparse Models
Slide 1: L1 Regularization Overview

L1 regularization, also known as Lasso (Least Absolute Shrinkage and Selection Operator), adds a penalty term proportional to the absolute values of model coefficients. This technique promotes sparsity by driving some coefficients exactly to zero, effectively performing feature selection during model training.

```python
# Mathematical representation of L1 regularization cost function
'''
Cost Function with L1:
$$J(\theta) = \frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\sum_{j=1}^n|\theta_j|$$
Where:
- m is number of samples
- n is number of features
- λ is regularization strength
'''
```

Slide 2: Basic L1 Implementation

This implementation demonstrates how to apply L1 regularization to a linear regression model using scikit-learn. The code showcases the fundamental usage pattern and parameter tuning for controlling feature selection behavior.

```python
from sklearn.linear_model import Lasso
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 20)  # 100 samples, 20 features
y = 3*X[:, 0] + 2*X[:, 1] + np.random.randn(100)*0.1  # Only 2 relevant features

# Initialize and train Lasso model
lasso = Lasso(alpha=0.1)  # alpha is the L1 penalty coefficient
lasso.fit(X, y)

# Print non-zero coefficients
print("Non-zero coefficients:")
for idx, coef in enumerate(lasso.coef_):
    if abs(coef) > 1e-10:
        print(f"Feature {idx}: {coef:.4f}")
```

Slide 3: Custom L1 Regularizer Implementation

Understanding the mechanics of L1 regularization requires implementing it from scratch. This implementation shows how the absolute value penalty affects gradient calculations and weight updates during optimization.

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
            
            # Gradient of MSE
            grad_mse = -2/n_samples * X.T.dot(y - y_pred)
            
            # L1 gradient (subgradient)
            grad_l1 = self.lambda_param * np.sign(self.weights)
            
            # Update weights
            self.weights -= learning_rate * (grad_mse + grad_l1)
            
    def predict(self, X):
        return np.dot(X, self.weights)
```

Slide 4: Comparing L1 with Other Regularizers

L1 regularization differs fundamentally from L2 and ElasticNet in its ability to produce exact zero coefficients. This implementation compares the sparsity patterns and prediction performance of different regularization techniques.

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate sparse data
np.random.seed(42)
n_samples, n_features = 100, 50
X = np.random.randn(n_samples, n_features)
true_weights = np.zeros(n_features)
true_weights[:5] = [1.5, -2, 3, -4, 2.5]
y = np.dot(X, true_weights) + np.random.randn(n_samples) * 0.1

# Compare different models
models = {
    'No Regularization': LinearRegression(),
    'L1 (Lasso)': Lasso(alpha=0.1),
    'L2 (Ridge)': Ridge(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# Fit and collect coefficients
coefficients = {}
for name, model in models.items():
    model.fit(X, y)
    coefficients[name] = model.coef_
```

Slide 5: Real-world Example - Gene Selection

L1 regularization excels in high-dimensional biological data analysis where identifying relevant genes is crucial. This implementation demonstrates feature selection in gene expression data for cancer classification.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulate gene expression data
n_samples, n_genes = 200, 1000
X_genes = np.random.randn(n_samples, n_genes)
# Only 10 genes are actually relevant
relevant_genes = np.random.choice(n_genes, 10, replace=False)
y_cancer = np.dot(X_genes[:, relevant_genes], np.random.randn(10)) > 0

# Preprocess and split data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_genes)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cancer, test_size=0.2, random_state=42
)

# Train Lasso classifier
from sklearn.linear_model import LogisticRegression
lasso_classifier = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
lasso_classifier.fit(X_train, y_train)

# Identify selected genes
selected_genes = np.where(abs(lasso_classifier.coef_[0]) > 1e-10)[0]
print(f"Number of selected genes: {len(selected_genes)}")
print(f"Test accuracy: {accuracy_score(y_test, lasso_classifier.predict(X_test)):.4f}")
```

Slide 6: Cross-Validation for L1 Parameter Tuning

Cross-validation is essential for finding the optimal L1 regularization strength (alpha/lambda). This implementation demonstrates how to systematically search for the best regularization parameter while avoiding overfitting.

```python
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

# Setup cross-validation framework
def find_optimal_alpha(X, y, alphas=None):
    if alphas is None:
        alphas = np.logspace(-4, 1, 50)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mean_scores = []
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha)
        scores = cross_val_score(lasso, X, y, cv=cv, scoring='neg_mean_squared_error')
        mean_scores.append(-scores.mean())
    
    best_alpha = alphas[np.argmin(mean_scores)]
    return best_alpha, mean_scores

# Example usage
best_alpha, cv_scores = find_optimal_alpha(X, y)
print(f"Optimal alpha: {best_alpha:.6f}")
```

Slide 7: Handling Multicollinearity with L1

L1 regularization effectively addresses multicollinearity by selecting one feature from each group of correlated predictors. This implementation demonstrates how L1 handles correlated features in high-dimensional data.

```python
# Generate correlated features
n_samples = 200
X_base = np.random.randn(n_samples, 3)
X_correlated = np.zeros((n_samples, 9))

# Create groups of correlated features
X_correlated[:, 0:3] = X_base + np.random.randn(n_samples, 3) * 0.1
X_correlated[:, 3:6] = X_base + np.random.randn(n_samples, 3) * 0.1
X_correlated[:, 6:9] = X_base + np.random.randn(n_samples, 3) * 0.1

# True relationship uses only one feature from each group
y = 2 * X_correlated[:, 0] - 1 * X_correlated[:, 3] + 3 * X_correlated[:, 6] + np.random.randn(n_samples) * 0.1

# Apply Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_correlated, y)

# Show which features were selected
for i, coef in enumerate(lasso.coef_):
    if abs(coef) > 1e-10:
        print(f"Feature {i} (Group {i//3}): {coef:.4f}")
```

Slide 8: L1 for Time Series Feature Selection

Time series data often contains redundant or irrelevant features. This implementation shows how L1 regularization can identify the most important lagged variables and seasonal components.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_time_features(data, lags=5):
    df = pd.DataFrame(data)
    # Add lagged features
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df[0].shift(i)
    
    # Add rolling statistics
    df['rolling_mean'] = df[0].rolling(window=3).mean()
    df['rolling_std'] = df[0].rolling(window=3).std()
    
    # Remove NaN rows
    df = df.dropna()
    
    return df

# Generate sample time series
np.random.seed(42)
t = np.linspace(0, 100, 1000)
y = 3 * np.sin(0.1 * t) + 2 * np.cos(0.05 * t) + np.random.randn(1000) * 0.2

# Create features and prepare data
X_time = create_time_features(y)
y_time = X_time[0].values
X_time = X_time.drop(columns=[0])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_time)

# Apply Lasso
lasso_time = Lasso(alpha=0.01)
lasso_time.fit(X_scaled, y_time)

# Show important features
for i, coef in enumerate(lasso_time.coef_):
    if abs(coef) > 1e-10:
        print(f"Feature {X_time.columns[i]}: {coef:.4f}")
```

Slide 9: Sparse Recovery with L1 Regularization

L1 regularization can recover sparse signals from noisy measurements. This implementation demonstrates the effectiveness of L1 in signal processing and compressed sensing applications.

```python
def generate_sparse_signal(n_features, n_nonzero):
    signal = np.zeros(n_features)
    indices = np.random.choice(n_features, n_nonzero, replace=False)
    signal[indices] = np.random.randn(n_nonzero)
    return signal

# Generate sparse signal
n_features = 100
true_signal = generate_sparse_signal(n_features, 5)

# Create measurement matrix
n_measurements = 50
A = np.random.randn(n_measurements, n_features)
A = A / np.sqrt(np.sum(A**2, axis=1, keepdims=True))

# Generate noisy measurements
measurements = np.dot(A, true_signal) + np.random.randn(n_measurements) * 0.1

# Recover signal using Lasso
lasso_recovery = Lasso(alpha=0.1, max_iter=10000)
lasso_recovery.fit(A, measurements)

# Compare recovery with true signal
recovered_signal = lasso_recovery.coef_
recovery_error = np.mean((true_signal - recovered_signal)**2)
print(f"Recovery MSE: {recovery_error:.6f}")
```

Slide 10: Elastic Path and Solution Stability

The solution path of L1 regularization shows how coefficients evolve with varying regularization strength. This implementation visualizes the path and demonstrates the stability characteristics of L1 regularization solutions.

```python
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt

# Generate example data
n_samples, n_features = 100, 20
X = np.random.randn(n_samples, n_features)
true_coef = np.zeros(n_features)
true_coef[:5] = [1.5, -2, 3, -4, 2.5]
y = np.dot(X, true_coef) + np.random.randn(n_samples) * 0.1

# Compute regularization path
alphas, coefs, _ = lasso_path(X, y, alphas=np.logspace(-4, 1, 100))

# Plot regularization path
def plot_regularization_path(alphas, coefs):
    plt.figure(figsize=(10, 6))
    for coef_path in coefs:
        plt.plot(-np.log10(alphas), coef_path)
    plt.xlabel('-log(alpha)')
    plt.ylabel('Coefficients')
    plt.title('Lasso Path')
    return plt

# Save visualization code
path_viz = '''
plt = plot_regularization_path(alphas, coefs)
plt.show()
'''
```

Slide 11: Feature Importance Analysis with L1

L1 regularization provides natural feature importance rankings through coefficient magnitudes. This implementation shows how to analyze and visualize feature importance in a robust way.

```python
class L1FeatureImportance:
    def __init__(self, n_bootstrap=100):
        self.n_bootstrap = n_bootstrap
        
    def analyze(self, X, y, alpha=0.1):
        n_samples, n_features = X.shape
        importance_scores = np.zeros((self.n_bootstrap, n_features))
        
        for i in range(self.n_bootstrap):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            # Fit Lasso
            lasso = Lasso(alpha=alpha)
            lasso.fit(X_boot, y_boot)
            
            importance_scores[i] = np.abs(lasso.coef_)
            
        # Calculate statistics
        mean_importance = np.mean(importance_scores, axis=0)
        std_importance = np.std(importance_scores, axis=0)
        
        return mean_importance, std_importance

# Example usage
analyzer = L1FeatureImportance()
mean_imp, std_imp = analyzer.analyze(X, y)

# Display top features
feature_ranking = np.argsort(mean_imp)[::-1]
for rank, idx in enumerate(feature_ranking[:5]):
    print(f"Rank {rank+1}: Feature {idx} (Importance: {mean_imp[idx]:.4f} ± {std_imp[idx]:.4f})")
```

Slide 12: Handling Missing Data with L1

L1 regularization can be combined with missing data imputation techniques. This implementation shows how to handle missing values while maintaining the sparsity-inducing properties of L1.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np

class L1RegularizedImputation:
    def __init__(self, alpha=0.1, max_iter=100):
        self.alpha = alpha
        self.max_iter = max_iter
        
    def fit_transform(self, X):
        # Initialize imputer with L1 regularization
        estimator = Lasso(alpha=self.alpha)
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=self.max_iter,
            random_state=42
        )
        
        # Fit and transform
        X_imputed = imputer.fit_transform(X)
        
        # Fit final Lasso on imputed data
        self.final_model = Lasso(alpha=self.alpha)
        self.final_model.fit(X_imputed, y)
        
        return X_imputed, self.final_model.coef_

# Example with missing data
X_missing = X.copy()
mask = np.random.random(X.shape) < 0.1
X_missing[mask] = np.nan

imputer = L1RegularizedImputation()
X_imputed, coefficients = imputer.fit_transform(X_missing)
```

Slide 13: Additional Resources

*   "The Elements of Statistical Learning" - ArXiv equivalent: [https://web.stanford.edu/~hastie/Papers/ESLII.pdf](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)
*   "Compressed Sensing and L1 Minimization" - [https://arxiv.org/abs/0705.1303](https://arxiv.org/abs/0705.1303)
*   "Least Angle Regression" - [https://arxiv.org/abs/math/0406456](https://arxiv.org/abs/math/0406456)
*   "Random Projections and L1 Regularization" - [https://www.jmlr.org/papers/volume5/wainwright04a/wainwright04a.pdf](https://www.jmlr.org/papers/volume5/wainwright04a/wainwright04a.pdf)
*   "Feature Selection with L1 Regularization" - For latest research, search "L1 regularization feature selection" on Google Scholar

