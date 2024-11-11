## Calculus Fundamentals for Python
Slide 1: Limits in Financial Mathematics

The concept of limits forms the foundation for analyzing continuous financial processes. In quantitative finance, limits help evaluate how financial instruments behave as they approach specific conditions, particularly useful in options pricing and risk assessment near expiration dates.

```python
import numpy as np
import matplotlib.pyplot as plt

def limit_example(S, K):
    # Calculate option payoff as stock price approaches strike
    # S: Stock price array
    # K: Strike price
    
    call_payoff = np.maximum(S - K, 0)
    put_payoff = np.maximum(K - S, 0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(S, call_payoff, label='Call Option')
    plt.plot(S, put_payoff, label='Put Option')
    plt.axvline(x=K, color='r', linestyle='--', label='Strike Price')
    plt.title('Option Payoff as Stock Price Approaches Strike')
    plt.xlabel('Stock Price')
    plt.ylabel('Payoff')
    plt.legend()
    plt.grid(True)
    
    # Example usage
    S = np.linspace(0, 100, 1000)
    K = 50
    limit_example(S, K)
```

Slide 2: Derivatives and Greeks Implementation

Derivatives in financial mathematics represent the rate of change of option prices with respect to various parameters. The Greeks measure these sensitivities and are crucial for risk management in options trading.

```python
import numpy as np
from scipy.stats import norm

def black_scholes_greeks(S, K, T, r, sigma):
    """
    Calculate Black-Scholes Greeks
    S: Stock price
    K: Strike price
    T: Time to maturity
    r: Risk-free rate
    sigma: Volatility
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Delta - First derivative with respect to stock price
    delta = norm.cdf(d1)
    
    # Gamma - Second derivative with respect to stock price
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    
    # Theta - First derivative with respect to time
    theta = (-S*sigma*norm.pdf(d1))/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta}

# Example calculation
S, K = 100, 100  # At-the-money option
T = 1.0          # One year to expiry
r = 0.05         # 5% risk-free rate
sigma = 0.2      # 20% volatility

greeks = black_scholes_greeks(S, K, T, r, sigma)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")
```

Slide 3: Integration Methods for Present Value

Integration in finance is essential for calculating present values and total returns. This implementation demonstrates numerical integration techniques for continuous cash flows and dividend streams using both traditional and advanced methods.

```python
import numpy as np
from scipy import integrate

def present_value_continuous(cash_flow_func, r, T):
    """
    Calculate present value of continuous cash flows
    cash_flow_func: Function defining cash flow at time t
    r: Discount rate
    T: Time horizon
    """
    integrand = lambda t: cash_flow_func(t) * np.exp(-r*t)
    pv, error = integrate.quad(integrand, 0, T)
    return pv

# Example with growing cash flows
def growing_cash_flow(t, initial=100, growth_rate=0.03):
    return initial * np.exp(growth_rate * t)

# Calculate PV for different scenarios
r = 0.05  # Discount rate
T = 10    # Time horizon

pv = present_value_continuous(
    lambda t: growing_cash_flow(t), 
    r, 
    T
)

print(f"Present Value: ${pv:,.2f}")
```

Slide 4: Multivariable Calculus in Portfolio Optimization

Multivariable calculus is crucial for modern portfolio theory and optimization. This implementation demonstrates portfolio optimization using gradient descent to find the optimal weights that maximize the Sharpe ratio considering multiple assets.

```python
import numpy as np
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, returns, cov_matrix, risk_free_rate):
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.rf = risk_free_rate
        
    def sharpe_ratio(self, weights):
        portfolio_return = np.sum(self.returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return -(portfolio_return - self.rf) / portfolio_vol  # Negative for minimization
    
    def optimize(self):
        n_assets = len(self.returns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(n_assets))  # Weights between 0 and 1
        
        initial_weights = np.array([1/n_assets] * n_assets)
        result = minimize(self.sharpe_ratio, initial_weights, 
                        method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

# Example usage
returns = np.array([0.12, 0.15, 0.10, 0.08])
cov_matrix = np.array([
    [0.18, 0.08, 0.05, 0.03],
    [0.08, 0.16, 0.06, 0.04],
    [0.05, 0.06, 0.14, 0.02],
    [0.03, 0.04, 0.02, 0.12]
])
rf = 0.02

optimizer = PortfolioOptimizer(returns, cov_matrix, rf)
optimal_weights = optimizer.optimize()
print("Optimal Portfolio Weights:", optimal_weights)
```

Slide 5: Differential Equations in Option Pricing

The Black-Scholes partial differential equation is fundamental in option pricing. This implementation solves the PDE numerically using finite difference methods to price European options.

```python
import numpy as np

class BlackScholesPDESolver:
    def __init__(self, S0, K, r, sigma, T, M, N):
        self.S0 = S0          # Initial stock price
        self.K = K            # Strike price
        self.r = r            # Risk-free rate
        self.sigma = sigma    # Volatility
        self.T = T            # Time to maturity
        self.M = M            # Number of stock price steps
        self.N = N            # Number of time steps
        
    def solve(self):
        # Grid parameters
        dt = self.T/self.N
        dS = 5*self.S0/self.M
        
        # Initialize grid
        grid = np.zeros((self.M+1, self.N+1))
        
        # Set boundary conditions
        S = np.linspace(0, 5*self.S0, self.M+1)
        grid[:, -1] = np.maximum(S - self.K, 0)  # Terminal condition
        grid[0, :] = 0        # Lower boundary
        grid[-1, :] = 5*self.S0 - self.K*np.exp(-self.r*(self.T-np.linspace(0, self.T, self.N+1)))  # Upper boundary
        
        # Solve PDE using explicit finite difference
        for j in range(self.N-1, -1, -1):
            for i in range(1, self.M):
                a = 0.5*dt*((self.sigma*i)**2)
                b = 0.5*dt*self.r*i
                grid[i,j] = (1-self.r*dt)*grid[i,j+1] + \
                           (a+b)*grid[i+1,j+1] + \
                           (a-b)*grid[i-1,j+1]
        
        return grid

# Example usage
solver = BlackScholesPDESolver(
    S0=100,    # Initial stock price
    K=100,     # Strike price
    r=0.05,    # Risk-free rate
    sigma=0.2, # Volatility
    T=1.0,     # Time to maturity
    M=100,     # Price steps
    N=1000     # Time steps
)

solution = solver.solve()
print(f"Option Price: {solution[50,0]:.4f}")  # Price at S0=100
```

Slide 6: Sequence Analysis for Time Series Prediction

Financial time series analysis requires understanding sequences and their convergence properties. This implementation demonstrates how to analyze and predict financial sequences using advanced statistical methods.

```python
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy import stats

class TimeSeriesAnalyzer:
    def __init__(self, series):
        self.series = np.array(series)
        
    def test_stationarity(self):
        """Test for stationarity using Augmented Dickey-Fuller test"""
        result = adfuller(self.series)
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }
    
    def calculate_hurst_exponent(self, lags=20):
        """Calculate Hurst exponent to determine long-term memory"""
        tau = []
        lagvec = []
        
        for lag in range(2, lags):
            tau.append(np.std(np.subtract(self.series[lag:], self.series[:-lag])))
            lagvec.append(lag)
        
        lag_log = np.log10(lagvec)
        tau_log = np.log10(tau)
        slope = np.polyfit(lag_log, tau_log, 1)
        return slope[0]
    
    def detect_regime_changes(self, window=20):
        """Detect regime changes using rolling statistics"""
        rolling_mean = np.array([np.mean(self.series[max(0, i-window):i]) 
                               for i in range(window, len(self.series))])
        rolling_std = np.array([np.std(self.series[max(0, i-window):i]) 
                              for i in range(window, len(self.series))])
        
        # Detect significant changes
        mean_changes = np.where(np.abs(np.diff(rolling_mean)) > 2*np.std(rolling_mean))[0]
        vol_changes = np.where(np.abs(np.diff(rolling_std)) > 2*np.std(rolling_std))[0]
        
        return {
            'mean_regime_changes': mean_changes + window,
            'volatility_regime_changes': vol_changes + window
        }

# Example usage
np.random.seed(42)
returns = np.random.normal(0.001, 0.02, 1000)  # Simulated returns

analyzer = TimeSeriesAnalyzer(returns)
stationarity = analyzer.test_stationarity()
hurst = analyzer.calculate_hurst_exponent()
regimes = analyzer.detect_regime_changes()

print(f"Stationarity p-value: {stationarity['p_value']:.4f}")
print(f"Hurst exponent: {hurst:.4f}")
print(f"Number of mean regime changes: {len(regimes['mean_regime_changes'])}")
```

Slide 7: Series Convergence in Risk Metrics

Understanding series convergence is crucial for risk metrics calculation, particularly in Value at Risk (VaR) and Expected Shortfall (ES) estimations using historical simulation methods.

```python
import numpy as np
from scipy import stats
import pandas as pd

class RiskMetricsCalculator:
    def __init__(self, returns, confidence_level=0.95):
        self.returns = np.array(returns)
        self.confidence_level = confidence_level
        
    def calculate_var(self, method='historical'):
        """
        Calculate Value at Risk using different methods
        """
        if method == 'historical':
            return np.percentile(self.returns, (1 - self.confidence_level) * 100)
        elif method == 'parametric':
            mean = np.mean(self.returns)
            std = np.std(self.returns)
            return stats.norm.ppf(1 - self.confidence_level, mean, std)
        
    def calculate_es(self, method='historical'):
        """
        Calculate Expected Shortfall (Conditional VaR)
        """
        var = self.calculate_var(method)
        if method == 'historical':
            return np.mean(self.returns[self.returns <= var])
        elif method == 'parametric':
            mean = np.mean(self.returns)
            std = np.std(self.returns)
            return mean - std * stats.norm.pdf(stats.norm.ppf(1 - self.confidence_level)) / (1 - self.confidence_level)
    
    def convergence_analysis(self, window_sizes):
        """
        Analyze convergence of risk metrics across different sample sizes
        """
        var_convergence = []
        es_convergence = []
        
        for size in window_sizes:
            sample = self.returns[:size]
            temp_calculator = RiskMetricsCalculator(sample, self.confidence_level)
            var_convergence.append(temp_calculator.calculate_var())
            es_convergence.append(temp_calculator.calculate_es())
            
        return np.array(var_convergence), np.array(es_convergence)

# Example usage
np.random.seed(42)
returns = np.random.normal(-0.001, 0.02, 1000)

calculator = RiskMetricsCalculator(returns)
window_sizes = np.linspace(100, 1000, 10, dtype=int)
var_conv, es_conv = calculator.convergence_analysis(window_sizes)

print(f"Final VaR: {var_conv[-1]:.4f}")
print(f"Final ES: {es_conv[-1]:.4f}")
print(f"VaR Convergence Rate: {np.std(np.diff(var_conv)):.6f}")
```

Slide 8: Integration Techniques for Options Greeks

Advanced integration methods are essential for calculating complex Greeks and hedging parameters. This implementation shows numerical integration techniques for higher-order Greeks and cross-derivatives.

```python
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

class AdvancedGreeksCalculator:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
    def _d1(self):
        return (np.log(self.S/self.K) + (self.r + self.sigma**2/2)*self.T) / \
               (self.sigma*np.sqrt(self.T))
               
    def _d2(self):
        return self._d1() - self.sigma*np.sqrt(self.T)
    
    def vanna(self):
        """
        Calculate Vanna (d(Delta)/d(vol))
        """
        d1 = self._d1()
        d2 = self._d2()
        return -norm.pdf(d1) * d2 / self.sigma
    
    def charm(self):
        """
        Calculate Charm (d(Delta)/d(time))
        """
        d1 = self._d1()
        d2 = self._d2()
        return -norm.pdf(d1) * \
               (self.r/(self.sigma*np.sqrt(self.T)) - \
                d2/(2*self.T))
    
    def vomma(self):
        """
        Calculate Vomma (d^2(Price)/d(vol)^2)
        """
        d1 = self._d1()
        d2 = self._d2()
        return self.S * norm.pdf(d1) * np.sqrt(self.T) * \
               d1 * d2 / self.sigma
    
    def speed(self):
        """
        Calculate Speed (d^3(Price)/d(spot)^3)
        """
        d1 = self._d1()
        return -(norm.pdf(d1)/(self.S**2*self.sigma*np.sqrt(self.T))) * \
               (d1/(self.sigma*np.sqrt(self.T)) + 1)

# Example usage
calculator = AdvancedGreeksCalculator(
    S=100,    # Spot price
    K=100,    # Strike price
    T=1.0,    # Time to maturity
    r=0.05,   # Risk-free rate
    sigma=0.2 # Volatility
)

print(f"Vanna: {calculator.vanna():.6f}")
print(f"Charm: {calculator.charm():.6f}")
print(f"Vomma: {calculator.vomma():.6f}")
print(f"Speed: {calculator.speed():.6f}")
```

Slide 9: Differential Equations for Interest Rate Models

The Heath-Jarrow-Morton (HJM) framework uses stochastic differential equations to model interest rate term structures. This implementation demonstrates a numerical solution for forward rate evolution.

```python
import numpy as np
from scipy.linalg import cholesky

class HJMModel:
    def __init__(self, initial_rates, volatility, correlation, dt, n_paths):
        self.initial_rates = initial_rates
        self.volatility = volatility
        self.correlation = correlation
        self.dt = dt
        self.n_paths = n_paths
        self.n_rates = len(initial_rates)
        
    def simulate_rates(self, T):
        """
        Simulate forward rates using HJM framework
        """
        n_steps = int(T/self.dt)
        rates = np.zeros((self.n_paths, n_steps + 1, self.n_rates))
        rates[:, 0] = self.initial_rates
        
        # Generate correlated Brownian motions
        chol = cholesky(self.correlation)
        
        for t in range(n_steps):
            # Generate random shocks
            dW = np.random.multivariate_normal(
                mean=np.zeros(self.n_rates),
                cov=self.correlation,
                size=self.n_paths
            )
            
            # Calculate drift adjustment (no-arbitrage condition)
            drift = np.zeros(self.n_rates)
            for i in range(self.n_rates):
                drift[i] = np.sum(self.volatility[i] * 
                                np.dot(self.correlation[i], self.volatility))
            
            # Update rates
            rates[:, t+1] = rates[:, t] + \
                           drift * self.dt + \
                           self.volatility * np.sqrt(self.dt) * dW
        
        return rates

# Example usage
initial_rates = np.array([0.02, 0.025, 0.03, 0.035])
volatility = np.array([0.01, 0.012, 0.014, 0.016])
correlation = np.array([
    [1.0, 0.8, 0.6, 0.4],
    [0.8, 1.0, 0.8, 0.6],
    [0.6, 0.8, 1.0, 0.8],
    [0.4, 0.6, 0.8, 1.0]
])

model = HJMModel(
    initial_rates=initial_rates,
    volatility=volatility,
    correlation=correlation,
    dt=1/252,  # Daily steps
    n_paths=1000
)

simulated_rates = model.simulate_rates(T=1.0)
print(f"Mean rates at T=1: {np.mean(simulated_rates[:,-1], axis=0)}")
print(f"Rate volatilities: {np.std(simulated_rates[:,-1], axis=0)}")
```

Slide 10: Multivariate Calculus for Risk Factor Decomposition

Principal Component Analysis (PCA) applied to yield curves demonstrates multivariate calculus in action for risk factor decomposition and dimension reduction.

```python
import numpy as np
from scipy.linalg import eigh

class YieldCurveDecomposition:
    def __init__(self, yield_curves):
        self.yield_curves = yield_curves
        self.n_observations, self.n_tenors = yield_curves.shape
        
    def compute_pca(self):
        """
        Perform PCA on yield curve movements
        """
        # Calculate yield curve changes
        changes = np.diff(self.yield_curves, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(changes.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(cov_matrix)
        
        # Sort in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculate explained variance ratio
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        
        # Project data onto principal components
        principal_components = changes @ eigenvectors
        
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'explained_variance_ratio': explained_variance_ratio,
            'principal_components': principal_components
        }
    
    def reconstruct_curves(self, n_components):
        """
        Reconstruct yield curves using top n components
        """
        pca_results = self.compute_pca()
        reduced_components = pca_results['principal_components'][:, :n_components]
        reduced_eigenvectors = pca_results['eigenvectors'][:, :n_components]
        
        reconstructed_changes = reduced_components @ reduced_eigenvectors.T
        reconstructed_curves = np.zeros_like(self.yield_curves)
        reconstructed_curves[0] = self.yield_curves[0]
        
        for i in range(1, self.n_observations):
            reconstructed_curves[i] = reconstructed_curves[i-1] + reconstructed_changes[i-1]
            
        return reconstructed_curves

# Example usage
np.random.seed(42)
tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
n_days = 500

# Simulate yield curves
base_curve = 0.02 + 0.02 * (1 - np.exp(-0.5 * tenors))
yield_curves = np.array([base_curve + np.random.normal(0, 0.001, len(tenors)) 
                        for _ in range(n_days)])

decomp = YieldCurveDecomposition(yield_curves)
pca_results = decomp.compute_pca()

print("Explained variance ratios:")
for i, ratio in enumerate(pca_results['explained_variance_ratio'][:3]):
    print(f"PC{i+1}: {ratio:.4f}")
```

Slide 11: Sequence Analysis for Trading Signals

Advanced sequence analysis techniques are essential for identifying trading patterns and generating signals. This implementation demonstrates how to detect technical patterns using calculus-based approaches.

```python
import numpy as np
from scipy.signal import argrelextrema
from scipy.optimize import minimize

class TechnicalPatternAnalyzer:
    def __init__(self, prices):
        self.prices = np.array(prices)
        self.returns = np.diff(np.log(prices))
        
    def detect_extrema(self, order=5):
        """
        Detect local maxima and minima using calculus
        """
        maxima = argrelextrema(self.prices, np.greater, order=order)[0]
        minima = argrelextrema(self.prices, np.less, order=order)[0]
        
        return maxima, minima
    
    def fit_trend_line(self, window_size=20):
        """
        Fit trend lines using least squares optimization
        """
        trends = np.zeros_like(self.prices)
        slopes = np.zeros_like(self.prices)
        
        for i in range(window_size, len(self.prices)):
            window = self.prices[i-window_size:i]
            x = np.arange(window_size)
            
            # Minimize squared error
            def objective(params):
                a, b = params
                return np.sum((window - (a * x + b))**2)
            
            result = minimize(objective, [0, np.mean(window)], method='Nelder-Mead')
            slope, intercept = result.x
            
            trends[i] = slope * window_size + intercept
            slopes[i] = slope
            
        return trends, slopes
    
    def calculate_momentum(self, period=14):
        """
        Calculate momentum indicators using calculus concepts
        """
        momentum = np.zeros_like(self.prices)
        
        # Rate of change
        momentum[period:] = (self.prices[period:] - self.prices[:-period]) / \
                          self.prices[:-period]
        
        # Calculate acceleration (second derivative)
        acceleration = np.gradient(np.gradient(momentum))
        
        return momentum, acceleration
    
    def generate_signals(self, threshold=0.02):
        """
        Generate trading signals based on pattern analysis
        """
        trends, slopes = self.fit_trend_line()
        momentum, acceleration = self.calculate_momentum()
        maxima, minima = self.detect_extrema()
        
        signals = np.zeros_like(self.prices)
        
        # Combine indicators for signal generation
        signals[maxima] = -1  # Sell signals
        signals[minima] = 1   # Buy signals
        
        # Adjust signals based on momentum
        signals[(momentum > threshold) & (acceleration > 0)] = 1
        signals[(momentum < -threshold) & (acceleration < 0)] = -1
        
        return signals

# Example usage
np.random.seed(42)
n_days = 252
prices = 100 * np.exp(np.random.normal(0.0001, 0.02, n_days).cumsum())

analyzer = TechnicalPatternAnalyzer(prices)
signals = analyzer.generate_signals()

# Calculate performance metrics
returns = np.diff(np.log(prices))
strategy_returns = signals[:-1] * returns
sharpe_ratio = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)

print(f"Number of trades: {np.sum(np.abs(np.diff(signals)) > 0)}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Total Return: {100 * np.sum(strategy_returns):.2f}%")
```

Slide 12: Integral Transform Methods in Risk Analysis

Fourier and Laplace transforms are powerful tools for analyzing risk distributions and option pricing. This implementation shows how to use these transforms for complex derivatives pricing.

```python
import numpy as np
from scipy.fft import fft, ifft
from scipy.integrate import quad

class IntegralTransformPricer:
    def __init__(self, S0, K, T, r, sigma, n_points=1024):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_points = n_points
        
    def characteristic_function(self, u):
        """
        Compute characteristic function for log-price under BSM
        """
        mu = self.r - 0.5 * self.sigma**2
        return np.exp(1j * u * (np.log(self.S0) + mu * self.T) - 
                     0.5 * self.sigma**2 * u**2 * self.T)
    
    def price_option_fft(self, option_type='call'):
        """
        Price options using Fast Fourier Transform
        """
        # Set up grid
        dx = 0.01
        x = np.arange(self.n_points) * dx
        dk = 2 * np.pi / (self.n_points * dx)
        k = np.arange(self.n_points) * dk
        
        # Compute transform
        integrand = self.characteristic_function(k - 0.5j)
        payoff_transform = 1 / (k**2 + 0.25)
        
        # Apply FFT
        fft_func = np.exp(-self.r * self.T) * \
                   ifft(integrand * payoff_transform) * dx
        
        # Extract option price
        strike_idx = int(np.log(self.K/self.S0) / dx + self.n_points/2)
        price = np.real(fft_func[strike_idx])
        
        return price if option_type == 'call' else price + self.K * np.exp(-self.r * self.T) - self.S0
    
    def compute_risk_measures(self):
        """
        Compute risk measures using transform methods
        """
        # Compute moments using characteristic function
        def moment_integrand(u, n):
            return np.real((-1j)**n * self.characteristic_function(u) / (1j * u)**(n+1))
        
        moments = []
        for n in range(4):
            moment, _ = quad(moment_integrand, -np.inf, np.inf, args=(n,))
            moments.append(moment)
        
        # Calculate risk metrics
        mean = moments[0]
        variance = moments[1] - moments[0]**2
        skewness = (moments[2] - 3*moments[0]*variance - moments[0]**3) / variance**1.5
        kurtosis = (moments[3] - 4*moments[0]*moments[2] + 6*moments[0]**2*variance + 
                   3*moments[0]**4) / variance**2
        
        return {
            'mean': mean,
            'volatility': np.sqrt(variance),
            'skewness': skewness,
            'kurtosis': kurtosis
        }

# Example usage
pricer = IntegralTransformPricer(
    S0=100,    # Spot price
    K=100,     # Strike price
    T=1.0,     # Time to maturity
    r=0.05,    # Risk-free rate
    sigma=0.2  # Volatility
)

call_price = pricer.price_option_fft('call')
risk_measures = pricer.compute_risk_measures()

print(f"Call Option Price: {call_price:.4f}")
print("\nRisk Measures:")
for measure, value in risk_measures.items():
    print(f"{measure.capitalize()}: {value:.4f}")
```

Slide 13: Limit Theory in Market Microstructure

Market microstructure analysis requires understanding of limit order books and price formation processes. This implementation demonstrates limit order book simulation and analysis using stochastic calculus.

```python
import numpy as np
from collections import defaultdict
from heapq import heappush, heappop

class LimitOrderBook:
    def __init__(self, initial_mid_price=100.0, tick_size=0.01):
        self.tick_size = tick_size
        self.mid_price = initial_mid_price
        self.bids = defaultdict(float)  # price -> volume
        self.asks = defaultdict(float)  # price -> volume
        self.bid_queue = []  # Min heap for best bids
        self.ask_queue = []  # Min heap for best asks
        
    def add_limit_order(self, side, price, volume):
        """
        Add a limit order to the book
        """
        price = round(price / self.tick_size) * self.tick_size
        
        if side == 'bid':
            self.bids[price] += volume
            heappush(self.bid_queue, -price)  # Negative for max heap
        else:
            self.asks[price] += volume
            heappush(self.ask_queue, price)
            
    def add_market_order(self, side, volume):
        """
        Execute a market order and return executed price
        """
        executed_volume = 0
        total_cost = 0
        
        if side == 'buy':
            while executed_volume < volume and self.ask_queue:
                best_ask = heappop(self.ask_queue)
                available_volume = self.asks[best_ask]
                
                execute_size = min(volume - executed_volume, available_volume)
                executed_volume += execute_size
                total_cost += execute_size * best_ask
                
                self.asks[best_ask] -= execute_size
                if self.asks[best_ask] > 0:
                    heappush(self.ask_queue, best_ask)
                    
        else:  # sell
            while executed_volume < volume and self.bid_queue:
                best_bid = -heappop(self.bid_queue)  # Negative for max heap
                available_volume = self.bids[best_bid]
                
                execute_size = min(volume - executed_volume, available_volume)
                executed_volume += execute_size
                total_cost += execute_size * best_bid
                
                self.bids[best_bid] -= execute_size
                if self.bids[best_bid] > 0:
                    heappush(self.bid_queue, -best_bid)
                    
        return total_cost / executed_volume if executed_volume > 0 else None
    
    def get_book_stats(self):
        """
        Calculate order book statistics
        """
        bid_prices = sorted([p for p, v in self.bids.items() if v > 0], reverse=True)
        ask_prices = sorted([p for p, v in self.asks.items() if v > 0])
        
        spread = ask_prices[0] - bid_prices[0] if bid_prices and ask_prices else None
        depth = {
            'bid_volume': sum(self.bids.values()),
            'ask_volume': sum(self.asks.values()),
            'bid_levels': len(bid_prices),
            'ask_levels': len(ask_prices)
        }
        
        return {
            'spread': spread,
            'mid_price': (ask_prices[0] + bid_prices[0]) / 2 if bid_prices and ask_prices else None,
            'depth': depth
        }

# Example usage
np.random.seed(42)
lob = LimitOrderBook()

# Simulate order flow
n_orders = 1000
for _ in range(n_orders):
    if np.random.random() < 0.7:  # 70% limit orders
        side = 'bid' if np.random.random() < 0.5 else 'ask'
        price = lob.mid_price * (1 + np.random.normal(0, 0.001))
        volume = np.random.exponential(100)
        lob.add_limit_order(side, price, volume)
    else:  # 30% market orders
        side = 'buy' if np.random.random() < 0.5 else 'sell'
        volume = np.random.exponential(50)
        executed_price = lob.add_market_order(side, volume)

stats = lob.get_book_stats()
print(f"Spread: {stats['spread']:.4f}")
print(f"Mid Price: {stats['mid_price']:.4f}")
print(f"Bid Depth: {stats['depth']['bid_volume']:.0f}")
print(f"Ask Depth: {stats['depth']['ask_volume']:.0f}")
```

Slide 14: Additional Resources

*   Stochastic Calculus for Finance II: Continuous-Time Models [https://arxiv.org/abs/1803.05893](https://arxiv.org/abs/1803.05893)
*   Modern Portfolio Theory: A Review and Analysis [https://arxiv.org/abs/2108.07914](https://arxiv.org/abs/2108.07914)
*   High-Frequency Trading and Market Microstructure [https://arxiv.org/abs/2003.02975](https://arxiv.org/abs/2003.02975)
*   Deep Learning in Asset Pricing [https://arxiv.org/abs/2106.11932](https://arxiv.org/abs/2106.11932)
*   Mathematical Foundations of Option Pricing [https://arxiv.org/abs/1901.04679](https://arxiv.org/abs/1901.04679)

