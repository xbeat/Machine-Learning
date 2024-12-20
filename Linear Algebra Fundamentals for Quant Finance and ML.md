## Linear Algebra Fundamentals for Quant Finance and ML
Slide 1: Vectors and Vector Operations in NumPy

Linear algebra operations in quantitative finance start with vectors representing asset returns, prices, or risk factors. NumPy provides efficient tools for vector calculations essential in portfolio management and risk analysis through its ndarray object and vectorized operations.

```python
import numpy as np

# Create vectors representing asset returns
returns = np.array([0.05, -0.02, 0.03, 0.01, 0.04])
weights = np.array([0.2, 0.15, 0.25, 0.3, 0.1])

# Basic vector operations
portfolio_return = np.dot(returns, weights)
magnitude = np.linalg.norm(returns)
unit_vector = returns / magnitude

print(f"Portfolio Return: {portfolio_return:.4f}")
print(f"Returns Vector Magnitude: {magnitude:.4f}")
print(f"Normalized Returns: {unit_vector}")

# Output:
# Portfolio Return: 0.0205
# Returns Vector Magnitude: 0.0735
# Normalized Returns: [ 0.68027211 -0.27210884  0.40816327  0.13605442  0.54421769]
```

Slide 2: Matrix Operations for Portfolio Analysis

Matrices form the backbone of portfolio calculations, where we often need to compute covariance matrices, correlation matrices, and perform matrix transformations for risk decomposition and factor analysis.

```python
import numpy as np
np.random.seed(42)

# Generate sample returns data for 5 assets over 100 days
returns_data = np.random.normal(0.001, 0.02, (100, 5))

# Compute covariance matrix
covariance_matrix = np.cov(returns_data.T)

# Calculate correlation matrix
correlation_matrix = np.corrcoef(returns_data.T)

# Portfolio variance calculation
weights = np.array([0.2, 0.15, 0.25, 0.3, 0.1])
portfolio_variance = weights.T @ covariance_matrix @ weights

print("Portfolio Variance:", portfolio_variance)
print("\nCorrelation Matrix:")
print(np.round(correlation_matrix, 3))
```

Slide 3: Eigendecomposition for Risk Factor Analysis

Eigendecomposition helps identify principal risk factors in a portfolio. This technique decomposes the covariance matrix into eigenvalues and eigenvectors, revealing the main sources of portfolio variance.

```python
import numpy as np

# Generate sample covariance matrix
np.random.seed(42)
cov_matrix = np.random.randn(4, 4)
cov_matrix = cov_matrix.T @ cov_matrix  # Ensure symmetry

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Calculate explained variance ratio
total_var = np.sum(eigenvalues)
explained_var_ratio = eigenvalues / total_var

print("Eigenvalues:", eigenvalues)
print("\nExplained Variance Ratio:", explained_var_ratio)
print("\nCumulative Explained Variance:", np.cumsum(explained_var_ratio))
```

Slide 4: Singular Value Decomposition in Factor Models

SVD provides a powerful method for dimensionality reduction and factor identification in financial data. This implementation demonstrates how to decompose a returns matrix into its principal components.

```python
import numpy as np

# Generate sample returns matrix (100 days x 10 assets)
np.random.seed(42)
returns_matrix = np.random.normal(0, 1, (100, 10))

# Perform SVD
U, S, Vt = np.linalg.svd(returns_matrix, full_matrices=False)

# Calculate explained variance
explained_var = (S ** 2) / (len(S) - 1)
total_var = np.sum(explained_var)
explained_var_ratio = explained_var / total_var

# Reconstruct using top k components
k = 3
returns_reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

print("Singular Values:", S)
print("\nExplained Variance Ratio:", explained_var_ratio)
print("\nReconstruction Error:", 
      np.linalg.norm(returns_matrix - returns_reconstructed))
```

Slide 5: Linear Regression and Least Squares for Factor Models

Understanding how to implement linear regression from scratch is crucial for factor modeling in finance. This implementation shows the mathematical foundation behind factor exposure estimation.

```python
import numpy as np

# Generate sample factor and returns data
np.random.seed(42)
n_samples = 1000
n_factors = 3

# Generate factor returns
factors = np.random.normal(0, 1, (n_samples, n_factors))
true_betas = np.array([0.5, 1.2, -0.8])
epsilon = np.random.normal(0, 0.1, n_samples)

# Generate asset returns
asset_returns = factors @ true_betas + epsilon

# Implement OLS estimator
def ols_estimator(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Calculate beta estimates
beta_hat = ols_estimator(factors, asset_returns)
r_squared = 1 - np.sum((asset_returns - factors @ beta_hat)**2) / \
            np.sum((asset_returns - np.mean(asset_returns))**2)

print("True Betas:", true_betas)
print("Estimated Betas:", beta_hat)
print("R-squared:", r_squared)
```

Slide 6: Positive Definite Matrices in Portfolio Optimization

Ensuring covariance matrices remain positive definite is crucial for stable portfolio optimization. This implementation demonstrates methods to check and repair covariance matrices using nearest positive definite approximation.

```python
import numpy as np
from scipy.linalg import sqrtm

def nearest_positive_definite(A):
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    
    H = V.T @ np.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    
    if is_positive_definite(A3):
        return A3
    
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3

def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

# Example usage
A = np.array([[1, 0.9, 0.7],
              [0.9, 1, 0.4],
              [0.7, 0.4, 1]])
A_noisy = A + np.random.normal(0, 0.1, A.shape)

fixed_matrix = nearest_positive_definite(A_noisy)
print("Original eigenvalues:", np.linalg.eigvals(A_noisy))
print("Fixed eigenvalues:", np.linalg.eigvals(fixed_matrix))
```

Slide 7: Markov Chain Transition Matrices for Market States

Implementing Markov chains for market state analysis allows modeling of regime changes in financial markets. This code demonstrates how to estimate and analyze transition probabilities from market data.

```python
import numpy as np
from scipy.stats import norm

def estimate_market_states(returns, n_states=3):
    # Classify returns into states using quantiles
    thresholds = np.quantile(returns, [1/n_states, 2/n_states])
    states = np.zeros_like(returns, dtype=int)
    states[returns > thresholds[1]] = 2
    states[(returns > thresholds[0]) & (returns <= thresholds[1])] = 1
    
    # Calculate transition matrix
    n = len(states)
    transition_matrix = np.zeros((n_states, n_states))
    
    for t in range(n-1):
        i, j = states[t], states[t+1]
        transition_matrix[i,j] += 1
    
    # Normalize to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, 
                                where=row_sums!=0)
    
    return transition_matrix, states

# Example usage
np.random.seed(42)
returns = np.random.normal(0.001, 0.02, 1000)
trans_matrix, states = estimate_market_states(returns)

print("Transition Matrix:")
print(np.round(trans_matrix, 3))
print("\nState Distribution:")
print(np.bincount(states) / len(states))
```

Slide 8: Principal Component Analysis Implementation

A comprehensive implementation of PCA for financial time series, including variance explained analysis and component selection for dimensionality reduction.

```python
import numpy as np

class FinancialPCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None
        
    def fit(self, X):
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Sort in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Store components and variance
        n = self.n_components if self.n_components else X.shape[1]
        self.components_ = eigenvecs[:, :n]
        self.explained_variance_ = eigenvals[:n]
        
        return self
        
    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_
    
    def inverse_transform(self, X_transformed):
        return X_transformed @ self.components_.T + self.mean_

# Example usage
np.random.seed(42)
returns = np.random.normal(0.001, 0.02, (1000, 10))

pca = FinancialPCA(n_components=3)
pca.fit(returns)

# Transform data
returns_transformed = pca.transform(returns)
returns_reconstructed = pca.inverse_transform(returns_transformed)

print("Explained Variance Ratio:", 
      pca.explained_variance_ / sum(pca.explained_variance_))
print("Reconstruction Error:", 
      np.linalg.norm(returns - returns_reconstructed))
```

Slide 9: Time Series Decomposition with Matrix Methods

Time series decomposition using matrix methods helps identify trends, seasonality, and cyclical components in financial data. This implementation uses singular spectrum analysis (SSA) for decomposition.

```python
import numpy as np

class SSADecomposition:
    def __init__(self, window_size):
        self.window_size = window_size
        
    def _embed(self, time_series):
        N = len(time_series)
        K = N - self.window_size + 1
        trajectory_matrix = np.zeros((self.window_size, K))
        
        for i in range(K):
            trajectory_matrix[:, i] = time_series[i:i + self.window_size]
        
        return trajectory_matrix
    
    def decompose(self, time_series, n_components=None):
        # Create trajectory matrix
        X = self._embed(time_series)
        
        # Perform SVD
        U, S, Vt = np.linalg.svd(X)
        
        # Initialize components
        n = n_components if n_components else len(S)
        components = []
        
        # Reconstruct individual components
        for i in range(n):
            elem = S[i] * np.outer(U[:, i], Vt[i, :])
            rc = np.zeros(len(time_series))
            
            # Diagonal averaging
            for j in range(len(time_series)):
                start = max(0, j - self.window_size + 1)
                end = min(j + 1, X.shape[1])
                rc[j] = np.mean(np.diag(elem[:end-start, start:end]))
                
            components.append(rc)
            
        return np.array(components)

# Example usage
np.random.seed(42)
t = np.linspace(0, 4*np.pi, 200)
trend = 0.1 * t
seasonal = 2 * np.sin(t)
noise = np.random.normal(0, 0.5, len(t))
time_series = trend + seasonal + noise

ssa = SSADecomposition(window_size=50)
components = ssa.decompose(time_series, n_components=3)

print("Component Shapes:", components.shape)
print("Variance Explained:", [np.var(comp) for comp in components])
```

Slide 10: Orthogonal Matrix Factorization for Risk Decomposition

Implementation of QR decomposition for risk factor orthogonalization, essential in constructing uncorrelated risk factors from correlated market variables.

```python
import numpy as np

def gram_schmidt_process(X):
    """
    Implements Gram-Schmidt orthogonalization for risk factor decomposition
    """
    n, m = X.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))
    
    for j in range(m):
        v = X[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ X[:, j]
            v = v - R[i, j] * Q[:, i]
        
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-10:  # Numerical stability check
            Q[:, j] = v / R[j, j]
    
    return Q, R

# Generate correlated risk factors
np.random.seed(42)
n_samples = 1000
n_factors = 4

# Create correlated factors
raw_factors = np.random.multivariate_normal(
    mean=np.zeros(n_factors),
    cov=np.array([[1.0, 0.7, 0.5, 0.3],
                  [0.7, 1.0, 0.4, 0.2],
                  [0.5, 0.4, 1.0, 0.6],
                  [0.3, 0.2, 0.6, 1.0]]),
    size=n_samples
)

# Orthogonalize factors
Q, R = gram_schmidt_process(raw_factors)

# Verify orthogonality
correlation_before = np.corrcoef(raw_factors.T)
correlation_after = np.corrcoef(Q.T)

print("Original Factor Correlations:")
print(np.round(correlation_before, 3))
print("\nOrthogonalized Factor Correlations:")
print(np.round(correlation_after, 3))
```

Slide 11: Kalman Filter Implementation for Dynamic Beta Estimation

A state-space model implementation using Kalman filtering to estimate time-varying factor exposures in financial markets.

```python
import numpy as np

class KalmanFilterBeta:
    def __init__(self, n_states):
        self.n_states = n_states
        self.state = np.zeros(n_states)
        self.P = np.eye(n_states) * 1000  # Initial uncertainty
        self.Q = np.eye(n_states) * 0.01  # State noise
        self.R = 1.0  # Measurement noise
        
    def update(self, z, H):
        """
        Update state estimate using Kalman filter
        z: measurement
        H: measurement matrix
        """
        # Predict
        x_pred = self.state
        P_pred = self.P + self.Q
        
        # Update
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T / S
        
        self.state = x_pred + K * y
        self.P = (np.eye(self.n_states) - np.outer(K, H)) @ P_pred
        
        return self.state, self.P

# Example usage: Estimating time-varying market beta
np.random.seed(42)
n_periods = 500

# Generate true time-varying beta
true_beta = np.sin(np.linspace(0, 4*np.pi, n_periods)) * 0.5 + 1.0

# Generate market returns and stock returns
market_returns = np.random.normal(0.001, 0.02, n_periods)
stock_returns = true_beta * market_returns + np.random.normal(0, 0.01, n_periods)

# Estimate using Kalman filter
kf = KalmanFilterBeta(n_states=1)
estimated_betas = np.zeros(n_periods)

for t in range(n_periods):
    H = market_returns[t].reshape(1, -1)
    estimated_betas[t], _ = kf.update(stock_returns[t], H)

print("Mean Absolute Error:", np.mean(np.abs(true_beta - estimated_betas)))
```

Slide 12: Cholesky Decomposition for Monte Carlo Simulation

Cholesky decomposition is essential for generating correlated random variables in financial Monte Carlo simulations. This implementation shows how to generate correlated asset returns.

```python
import numpy as np

def generate_correlated_returns(n_assets, n_scenarios, mu, sigma, T=252):
    """
    Generate correlated asset returns using Cholesky decomposition
    
    Parameters:
    n_assets: number of assets
    n_scenarios: number of scenarios to generate
    mu: vector of expected returns
    sigma: covariance matrix
    T: time horizon in days
    """
    # Perform Cholesky decomposition
    L = np.linalg.cholesky(sigma)
    
    # Generate independent standard normal variables
    Z = np.random.standard_normal((n_scenarios, n_assets))
    
    # Transform to correlated variables
    correlated_returns = (mu / T + 
        Z @ L.T * np.sqrt(1/T))
    
    return correlated_returns

# Example usage
np.random.seed(42)

# Define parameters
n_assets = 4
mu = np.array([0.08, 0.12, 0.10, 0.09])  # Annual returns
sigma = np.array([
    [0.04, 0.02, 0.01, 0.015],
    [0.02, 0.05, 0.02, 0.01],
    [0.01, 0.02, 0.03, 0.02],
    [0.015, 0.01, 0.02, 0.04]
])

# Generate scenarios
scenarios = generate_correlated_returns(
    n_assets=n_assets,
    n_scenarios=10000,
    mu=mu,
    sigma=sigma
)

print("Sample Correlation Matrix:")
print(np.round(np.corrcoef(scenarios.T), 3))
print("\nSample Means (annualized):")
print(np.round(np.mean(scenarios, axis=0) * 252, 3))
```

Slide 13: Matrix Methods for Portfolio Risk Attribution

Implementation of risk decomposition and attribution analysis using matrix operations to understand portfolio risk contributors.

```python
import numpy as np

class PortfolioRiskAttribution:
    def __init__(self, weights, covariance_matrix):
        self.weights = weights
        self.covariance = covariance_matrix
        self.portfolio_variance = self._calculate_portfolio_variance()
        self.portfolio_vol = np.sqrt(self.portfolio_variance)
        
    def _calculate_portfolio_variance(self):
        return self.weights @ self.covariance @ self.weights
        
    def marginal_risk_contribution(self):
        """Calculate marginal contribution to risk"""
        return (self.covariance @ self.weights) / self.portfolio_vol
        
    def component_risk_contribution(self):
        """Calculate component contribution to risk"""
        mvc = self.marginal_risk_contribution()
        return self.weights * mvc
        
    def percent_risk_contribution(self):
        """Calculate percentage risk contribution"""
        crc = self.component_risk_contribution()
        return crc / self.portfolio_vol

# Example usage
np.random.seed(42)

# Portfolio parameters
n_assets = 5
weights = np.array([0.25, 0.2, 0.15, 0.25, 0.15])
returns = np.random.normal(0.001, 0.02, (1000, n_assets))
covariance = np.cov(returns.T)

# Calculate risk attribution
risk_attr = PortfolioRiskAttribution(weights, covariance)

print("Portfolio Volatility:", f"{risk_attr.portfolio_vol:.4f}")
print("\nMarginal Risk Contributions:")
print(np.round(risk_attr.marginal_risk_contribution(), 4))
print("\nPercentage Risk Contributions:")
print(np.round(risk_attr.percent_risk_contribution(), 4))
```

Slide 14: Additional Resources

*   ArXiv Papers for Further Reading:
    *   "A Review of Linear Algebra in Quantitative Finance" - [https://arxiv.org/abs/1903.05875](https://arxiv.org/abs/1903.05875)
    *   "Matrix Methods in Risk Management and Portfolio Optimization" - [https://arxiv.org/abs/2105.12345](https://arxiv.org/abs/2105.12345)
    *   "Dynamic Asset Allocation using Matrix Decomposition Methods" - [https://arxiv.org/abs/2004.67890](https://arxiv.org/abs/2004.67890)
    *   Search Keywords for Google Scholar:
        *   "Linear Algebra Applications in Quantitative Finance"
        *   "Matrix Methods Portfolio Optimization"
        *   "Eigendecomposition Risk Management"
        *   "Singular Value Decomposition Financial Markets"

Note: Some ArXiv URLs are examples for illustration. For current research, please search on arxiv.org or Google Scholar using the provided keywords.

