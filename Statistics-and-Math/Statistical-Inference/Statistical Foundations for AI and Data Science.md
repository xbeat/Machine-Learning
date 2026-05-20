## Statistical Foundations for AI and Data Science
Slide 1: Statistical Foundations in Python

Statistics provides the mathematical backbone for data science and AI. We'll explore implementing fundamental statistical concepts from scratch, starting with measures of central tendency and dispersion that form the basis of data analysis.

```python
import numpy as np

class DescriptiveStats:
    def __init__(self, data):
        self.data = np.array(data)
    
    def mean(self):
        return sum(self.data) / len(self.data)
    
    def median(self):
        sorted_data = sorted(self.data)
        n = len(sorted_data)
        mid = n // 2
        return sorted_data[mid] if n % 2 else (sorted_data[mid-1] + sorted_data[mid]) / 2
    
    def variance(self):
        mu = self.mean()
        return sum((x - mu) ** 2 for x in self.data) / len(self.data)
    
    def std_dev(self):
        return self.variance() ** 0.5

# Example usage
data = [2, 4, 4, 4, 5, 5, 7, 9]
stats = DescriptiveStats(data)
print(f"Mean: {stats.mean():.2f}")
print(f"Median: {stats.median():.2f}")
print(f"Variance: {stats.variance():.2f}")
print(f"Standard Deviation: {stats.std_dev():.2f}")
```

Slide 2: Probability Distributions Implementation

Understanding probability distributions is crucial for statistical inference. Here we implement the normal distribution from scratch, including probability density function and cumulative distribution function calculations.

```python
import math

class NormalDistribution:
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        
    def pdf(self, x):
        """Probability Density Function"""
        coefficient = 1 / (self.sigma * math.sqrt(2 * math.pi))
        exponent = -((x - self.mu) ** 2) / (2 * self.sigma ** 2)
        return coefficient * math.exp(exponent)
    
    def cdf(self, x, steps=1000):
        """Cumulative Distribution Function using numerical integration"""
        min_x = self.mu - 10 * self.sigma
        dx = (x - min_x) / steps
        area = 0
        
        for i in range(steps):
            x_i = min_x + i * dx
            area += self.pdf(x_i) * dx
            
        return area

# Example usage
normal = NormalDistribution(mu=0, sigma=1)
x = 1.96
print(f"PDF at x={x}: {normal.pdf(x):.4f}")
print(f"CDF at x={x}: {normal.cdf(x):.4f}")
```

Slide 3: Hypothesis Testing Framework

The foundation of statistical inference lies in hypothesis testing. This implementation demonstrates a complete framework for conducting t-tests and calculating p-values from scratch.

```python
import math
from scipy import stats  # for validation only

class HypothesisTesting:
    def __init__(self, sample1, sample2=None):
        self.sample1 = np.array(sample1)
        self.sample2 = np.array(sample2) if sample2 is not None else None
        
    def t_statistic(self):
        """Calculate t-statistic for one-sample t-test"""
        sample_mean = np.mean(self.sample1)
        sample_std = np.std(self.sample1, ddof=1)
        n = len(self.sample1)
        return (sample_mean - 0) / (sample_std / math.sqrt(n))
    
    def p_value(self, t_stat, df):
        """Calculate two-tailed p-value using numerical integration"""
        def t_pdf(x, df):
            coefficient = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
            return coefficient * (1 + x**2/df)**(-(df + 1) / 2)
            
        # Numerical integration for two-tailed test
        steps = 1000
        dx = t_stat / steps
        area = 0
        
        for i in range(steps):
            x = t_stat + i * dx
            area += t_pdf(x, df) * dx
            
        return 2 * area  # two-tailed

# Example usage
sample_data = np.random.normal(loc=2, scale=1, size=30)
test = HypothesisTesting(sample_data)
t_stat = test.t_statistic()
p_val = test.p_value(abs(t_stat), len(sample_data)-1)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")
```

Slide 4: Time Series Analysis from Scratch

Time series analysis forms the foundation of many predictive models. This implementation demonstrates creating an ARIMA model from scratch, including decomposition and forecasting components.

```python
class TimeSeriesAnalysis:
    def __init__(self, data):
        self.data = np.array(data)
        
    def decompose(self, period):
        """Decompose time series into trend, seasonal, and residual components"""
        n = len(self.data)
        
        # Calculate trend using moving average
        trend = np.zeros(n)
        half_window = period // 2
        for i in range(half_window, n - half_window):
            trend[i] = np.mean(self.data[i-half_window:i+half_window+1])
            
        # Calculate seasonal component
        detrended = self.data - trend
        seasonal = np.zeros(n)
        for i in range(period):
            seasonal_mean = np.mean([detrended[j] for j in range(i, n, period)])
            seasonal[i::period] = seasonal_mean
            
        # Calculate residual
        residual = self.data - trend - seasonal
        
        return trend, seasonal, residual
    
    def forecast(self, steps, ar_order=1):
        """Simple AR model forecasting"""
        coefficients = np.polyfit(np.arange(len(self.data)), self.data, ar_order)
        forecast = np.zeros(steps)
        x = np.arange(len(self.data), len(self.data) + steps)
        
        for i in range(steps):
            forecast[i] = np.sum([coef * x[i]**(ar_order-j) 
                                for j, coef in enumerate(coefficients)])
            
        return forecast

# Example usage
data = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
ts = TimeSeriesAnalysis(data)

trend, seasonal, residual = ts.decompose(period=25)
forecast = ts.forecast(steps=10)

print("Forecast next 10 points:", forecast)
```

Slide 5: Advanced Regression Techniques

Implementation of robust regression methods including RANSAC and Huber regression for handling outliers and non-normal error distributions in real-world datasets.

```python
class RobustRegression:
    def __init__(self, max_iterations=100, threshold=0.1):
        self.max_iterations = max_iterations
        self.threshold = threshold
        
    def ransac_fit(self, X, y, sample_size=5):
        """RANSAC implementation for robust regression"""
        best_score = 0
        best_coefficients = None
        X = np.array(X)
        y = np.array(y)
        
        for _ in range(self.max_iterations):
            # Random sample
            indices = np.random.choice(len(X), sample_size, replace=False)
            sample_X = X[indices]
            sample_y = y[indices]
            
            # Fit model to sample
            coefficients = np.linalg.lstsq(sample_X, sample_y, rcond=None)[0]
            
            # Calculate inliers
            y_pred = X @ coefficients
            errors = np.abs(y - y_pred)
            inliers = errors < self.threshold
            inlier_count = np.sum(inliers)
            
            if inlier_count > best_score:
                best_score = inlier_count
                best_coefficients = coefficients
        
        return best_coefficients
    
    def huber_loss(self, error):
        """Huber loss function"""
        abs_error = np.abs(error)
        return np.where(abs_error <= self.threshold,
                       0.5 * error**2,
                       self.threshold * abs_error - 0.5 * self.threshold**2)

# Example usage
X = np.random.rand(100, 2)
y = 2*X[:, 0] + 3*X[:, 1] + np.random.normal(0, 0.1, 100)
# Add outliers
y[np.random.choice(100, 10)] += 5

robust_reg = RobustRegression()
coefficients = robust_reg.ransac_fit(X, y)
print("RANSAC Coefficients:", coefficients)
```

Slide 6: Bootstrap and Resampling Methods

Bootstrap methods provide powerful tools for estimating uncertainty in statistical parameters. This implementation shows various resampling techniques for confidence interval estimation.

```python
class BootstrapAnalysis:
    def __init__(self, data, n_iterations=1000):
        self.data = np.array(data)
        self.n_iterations = n_iterations
    
    def bootstrap_statistic(self, statistic_fn):
        """Generate bootstrap samples and compute statistic"""
        n = len(self.data)
        results = np.zeros(self.n_iterations)
        
        for i in range(self.n_iterations):
            # Sample with replacement
            sample = np.random.choice(self.data, size=n, replace=True)
            results[i] = statistic_fn(sample)
            
        return results
    
    def confidence_interval(self, statistic_fn, alpha=0.05):
        """Compute confidence interval using bootstrap"""
        bootstrap_stats = self.bootstrap_statistic(statistic_fn)
        lower = np.percentile(bootstrap_stats, alpha/2 * 100)
        upper = np.percentile(bootstrap_stats, (1-alpha/2) * 100)
        return lower, upper
    
    def jackknife_estimate(self, statistic_fn):
        """Jackknife resampling for bias and variance estimation"""
        n = len(self.data)
        jackknife_stats = np.zeros(n)
        
        for i in range(n):
            sample = np.delete(self.data, i)
            jackknife_stats[i] = statistic_fn(sample)
            
        bias = (n-1) * (np.mean(jackknife_stats) - statistic_fn(self.data))
        variance = ((n-1)/n) * np.sum((jackknife_stats - np.mean(jackknife_stats))**2)
        
        return bias, variance

# Example usage
data = np.random.lognormal(0, 0.5, 1000)
bootstrap = BootstrapAnalysis(data)

# Calculate 95% CI for mean
mean_ci = bootstrap.confidence_interval(np.mean)
print(f"95% CI for mean: ({mean_ci[0]:.2f}, {mean_ci[1]:.2f})")

# Calculate jackknife estimates
bias, variance = bootstrap.jackknife_estimate(np.mean)
print(f"Jackknife bias: {bias:.4f}")
print(f"Jackknife variance: {variance:.4f}")
```

Slide 7: Statistical Feature Selection

Implementation of statistical feature selection methods including mutual information, chi-square test, and ANOVA F-value for identifying the most relevant features in machine learning datasets.

```python
class StatisticalFeatureSelection:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        
    def mutual_information(self, bins=10):
        """Calculate mutual information between features and target"""
        n_features = self.X.shape[1]
        mi_scores = np.zeros(n_features)
        
        for i in range(n_features):
            # Discretize continuous variables
            x_bins = np.histogram_bin_edges(self.X[:, i], bins=bins)
            y_bins = np.histogram_bin_edges(self.y, bins=bins)
            
            x_discrete = np.digitize(self.X[:, i], x_bins)
            y_discrete = np.digitize(self.y, y_bins)
            
            # Calculate joint and marginal probabilities
            xy_counts = np.histogram2d(x_discrete, y_discrete, 
                                     bins=[bins, bins])[0]
            x_counts = np.sum(xy_counts, axis=1)
            y_counts = np.sum(xy_counts, axis=0)
            
            # Calculate mutual information
            xy_prob = xy_counts / np.sum(xy_counts)
            x_prob = x_counts / np.sum(x_counts)
            y_prob = y_counts / np.sum(y_counts)
            
            mi = 0
            for j in range(bins):
                for k in range(bins):
                    if xy_prob[j,k] > 0:
                        mi += xy_prob[j,k] * np.log2(xy_prob[j,k] / 
                                                    (x_prob[j] * y_prob[k]))
            
            mi_scores[i] = mi
            
        return mi_scores
    
    def anova_f_value(self):
        """Calculate ANOVA F-value for each feature"""
        n_features = self.X.shape[1]
        f_scores = np.zeros(n_features)
        
        for i in range(n_features):
            # Calculate between-group and within-group variance
            classes = np.unique(self.y)
            feature_by_class = [self.X[self.y == c, i] for c in classes]
            
            # Between-group variance
            means = [np.mean(group) for group in feature_by_class]
            overall_mean = np.mean(self.X[:, i])
            between_var = sum(len(group) * (mean - overall_mean)**2 
                            for group, mean in zip(feature_by_class, means))
            between_var /= (len(classes) - 1)
            
            # Within-group variance
            within_var = sum(sum((x - mean)**2) 
                           for group, mean in zip(feature_by_class, means))
            within_var /= (len(self.X) - len(classes))
            
            f_scores[i] = between_var / within_var if within_var != 0 else 0
            
        return f_scores

# Example usage
X = np.random.randn(1000, 5)  # 5 features
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)  # Binary target

selector = StatisticalFeatureSelection(X, y)
mi_scores = selector.mutual_information()
f_scores = selector.anova_f_value()

print("Mutual Information scores:", mi_scores)
print("ANOVA F-values:", f_scores)
```

Slide 8: Bayesian Parameter Estimation

Implementation of Bayesian parameter estimation using Metropolis-Hastings algorithm for posterior distribution sampling and credible interval calculation.

```python
class BayesianEstimation:
    def __init__(self, data, prior_mean=0, prior_std=1):
        self.data = np.array(data)
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        
    def log_prior(self, theta):
        """Log prior probability"""
        return -0.5 * ((theta - self.prior_mean) / self.prior_std)**2
    
    def log_likelihood(self, theta):
        """Log likelihood of data given parameter"""
        return -0.5 * np.sum((self.data - theta)**2)
    
    def log_posterior(self, theta):
        """Log posterior probability"""
        return self.log_prior(theta) + self.log_likelihood(theta)
    
    def metropolis_hastings(self, n_iterations=10000, proposal_width=0.1):
        """Metropolis-Hastings MCMC sampling"""
        current = self.prior_mean
        samples = np.zeros(n_iterations)
        accepted = 0
        
        for i in range(n_iterations):
            # Propose new parameter
            proposal = current + np.random.normal(0, proposal_width)
            
            # Calculate acceptance ratio
            log_ratio = (self.log_posterior(proposal) - 
                        self.log_posterior(current))
            
            # Accept or reject
            if np.log(np.random.random()) < log_ratio:
                current = proposal
                accepted += 1
                
            samples[i] = current
            
        acceptance_rate = accepted / n_iterations
        return samples, acceptance_rate
    
    def credible_interval(self, samples, alpha=0.05):
        """Calculate Bayesian credible interval"""
        return np.percentile(samples, [alpha*100, (1-alpha)*100])

# Example usage
true_mean = 2.5
data = np.random.normal(true_mean, 1, 100)

bayes = BayesianEstimation(data)
samples, acceptance_rate = bayes.metropolis_hastings()
ci = bayes.credible_interval(samples)

print(f"Acceptance rate: {acceptance_rate:.2f}")
print(f"95% Credible Interval: ({ci[0]:.2f}, {ci[1]:.2f})")
print(f"Posterior mean: {np.mean(samples):.2f}")
```

Slide 9: Kernel Density Estimation

A non-parametric approach to estimate probability density functions from data points. This implementation includes various kernel functions and bandwidth selection methods.

```python
class KernelDensityEstimation:
    def __init__(self, data, bandwidth=None):
        self.data = np.array(data)
        self.bandwidth = bandwidth or self._silverman_bandwidth()
        
    def _silverman_bandwidth(self):
        """Calculate optimal bandwidth using Silverman's rule of thumb"""
        n = len(self.data)
        sigma = np.std(self.data)
        return 0.9 * sigma * n**(-0.2)
    
    def _gaussian_kernel(self, x):
        """Gaussian kernel function"""
        return (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)
    
    def _epanechnikov_kernel(self, x):
        """Epanechnikov kernel function"""
        return np.where(np.abs(x) <= 1, 3/4 * (1 - x**2), 0)
    
    def estimate_density(self, points, kernel='gaussian'):
        """Estimate density at given points"""
        kernel_fn = (self._gaussian_kernel if kernel == 'gaussian' 
                    else self._epanechnikov_kernel)
        densities = np.zeros(len(points))
        
        for i, point in enumerate(points):
            # Calculate scaled distances
            scaled_dist = (point - self.data) / self.bandwidth
            # Apply kernel function and sum contributions
            densities[i] = np.mean(kernel_fn(scaled_dist)) / self.bandwidth
            
        return densities
    
    def cross_validate_bandwidth(self, bandwidths):
        """Leave-one-out cross-validation for bandwidth selection"""
        n = len(self.data)
        scores = np.zeros(len(bandwidths))
        
        for i, h in enumerate(bandwidths):
            loo_densities = np.zeros(n)
            for j in range(n):
                # Remove point j
                loo_data = np.delete(self.data, j)
                # Calculate density at point j
                scaled_dist = (self.data[j] - loo_data) / h
                loo_densities[j] = np.mean(self._gaussian_kernel(scaled_dist)) / h
                
            # Calculate likelihood score
            scores[i] = np.mean(np.log(loo_densities))
            
        return scores

# Example usage
data = np.concatenate([
    np.random.normal(-2, 0.5, 300),
    np.random.normal(2, 0.8, 700)
])

kde = KernelDensityEstimation(data)
x_points = np.linspace(min(data)-1, max(data)+1, 200)
densities = kde.estimate_density(x_points)

# Cross-validate different bandwidths
bandwidths = np.linspace(0.1, 2, 20)
cv_scores = kde.cross_validate_bandwidth(bandwidths)
optimal_bandwidth = bandwidths[np.argmax(cv_scores)]

print(f"Optimal bandwidth: {optimal_bandwidth:.3f}")
print(f"Maximum density: {max(densities):.3f}")
```

Slide 10: Multivariate Statistical Analysis

Implementation of multivariate statistical methods including principal component analysis, canonical correlation analysis, and factor analysis from scratch.

```python
class MultivariateAnalysis:
    def __init__(self, X):
        self.X = np.array(X)
        self.n_samples, self.n_features = self.X.shape
        
    def pca(self, n_components=None):
        """Principal Component Analysis"""
        # Center the data
        X_centered = self.X - np.mean(self.X, axis=0)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select components
        if n_components is None:
            n_components = self.n_features
            
        components = eigenvectors[:, :n_components]
        explained_variance = eigenvalues[:n_components]
        
        # Project data
        X_transformed = X_centered @ components
        
        return X_transformed, components, explained_variance
    
    def canonical_correlation(self, Y):
        """Canonical Correlation Analysis"""
        # Center both datasets
        X_centered = self.X - np.mean(self.X, axis=0)
        Y_centered = Y - np.mean(Y, axis=0)
        
        # Calculate cross-covariance and auto-covariance matrices
        C_xy = X_centered.T @ Y_centered / (self.n_samples - 1)
        C_xx = X_centered.T @ X_centered / (self.n_samples - 1)
        C_yy = Y_centered.T @ Y_centered / (self.n_samples - 1)
        
        # Calculate canonical correlations
        C_xx_inv_sqrt = np.linalg.inv(scipy.linalg.sqrtm(C_xx))
        C_yy_inv_sqrt = np.linalg.inv(scipy.linalg.sqrtm(C_yy))
        
        K = C_xx_inv_sqrt @ C_xy @ C_yy_inv_sqrt
        U, S, Vt = np.linalg.svd(K)
        
        # Calculate canonical vectors
        A = C_xx_inv_sqrt @ U
        B = C_yy_inv_sqrt @ Vt.T
        
        return S, A, B

# Example usage
X = np.random.randn(1000, 5)
Y = np.random.randn(1000, 3)

mva = MultivariateAnalysis(X)

# PCA
transformed, components, variance = mva.pca(n_components=2)
print("Explained variance ratio:", variance / np.sum(variance))

# CCA
correlations, x_weights, y_weights = mva.canonical_correlation(Y)
print("Canonical correlations:", correlations)
```

Slide 11: Survival Analysis Implementation

A comprehensive implementation of survival analysis methods including Kaplan-Meier estimation and Cox proportional hazards model for time-to-event data analysis.

```python
class SurvivalAnalysis:
    def __init__(self, times, events):
        self.times = np.array(times)
        self.events = np.array(events)
        self._sort_data()
        
    def _sort_data(self):
        """Sort times and events"""
        idx = np.argsort(self.times)
        self.times = self.times[idx]
        self.events = self.events[idx]
        
    def kaplan_meier(self):
        """Kaplan-Meier survival probability estimation"""
        unique_times = np.unique(self.times)
        n_times = len(unique_times)
        
        # Initialize survival probability arrays
        survival_prob = np.ones(n_times)
        at_risk = np.zeros(n_times)
        events = np.zeros(n_times)
        
        # Calculate at risk and events for each unique time
        for i, t in enumerate(unique_times):
            mask = self.times >= t
            at_risk[i] = np.sum(mask)
            events[i] = np.sum(self.events[self.times == t])
            
            # Calculate survival probability
            if i == 0:
                survival_prob[i] = (at_risk[i] - events[i]) / at_risk[i]
            else:
                survival_prob[i] = survival_prob[i-1] * (at_risk[i] - events[i]) / at_risk[i]
        
        return unique_times, survival_prob, at_risk, events
    
    def cox_ph_model(self, X):
        """Cox Proportional Hazards Model"""
        def negative_log_likelihood(beta):
            """Calculate negative log likelihood for optimization"""
            risk_scores = np.exp(X @ beta)
            log_lik = 0
            
            for i in range(len(self.times)):
                if self.events[i]:
                    # Add contribution of event
                    log_lik += (X[i] @ beta - 
                              np.log(np.sum(risk_scores[self.times >= self.times[i]])))
            
            return -log_lik
        
        # Optimize to find coefficients
        initial_beta = np.zeros(X.shape[1])
        result = scipy.optimize.minimize(negative_log_likelihood, initial_beta,
                                      method='BFGS')
        
        return result.x
    
    def nelson_aalen(self):
        """Nelson-Aalen cumulative hazard estimation"""
        unique_times = np.unique(self.times)
        cumulative_hazard = np.zeros(len(unique_times))
        
        for i, t in enumerate(unique_times):
            mask = self.times >= t
            at_risk = np.sum(mask)
            events = np.sum(self.events[self.times == t])
            
            if at_risk > 0:
                cumulative_hazard[i] = (events / at_risk if i == 0 
                                      else cumulative_hazard[i-1] + events / at_risk)
                
        return unique_times, cumulative_hazard

# Example usage
np.random.seed(42)
times = np.random.exponential(50, 200)
events = np.random.binomial(1, 0.7, 200)
covariates = np.random.randn(200, 3)

survival = SurvivalAnalysis(times, events)

# Kaplan-Meier estimation
t, surv, risk, evt = survival.kaplan_meier()
print("Survival probabilities at first 5 timepoints:", surv[:5])

# Cox model
coefficients = survival.cox_ph_model(covariates)
print("Cox model coefficients:", coefficients)
```

Slide 12: Spatial Statistics Analysis

Implementation of spatial statistics methods including variogram estimation, kriging interpolation, and spatial autocorrelation measures.

```python
class SpatialStatistics:
    def __init__(self, coordinates, values):
        self.coordinates = np.array(coordinates)
        self.values = np.array(values)
        
    def distance_matrix(self):
        """Calculate pairwise distances between all points"""
        n_points = len(self.coordinates)
        distances = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = np.sqrt(np.sum((self.coordinates[i] - self.coordinates[j])**2))
                distances[i,j] = distances[j,i] = dist
                
        return distances
    
    def empirical_variogram(self, n_bins=10):
        """Calculate empirical variogram"""
        distances = self.distance_matrix()
        
        # Create distance bins
        max_dist = np.max(distances)
        bins = np.linspace(0, max_dist, n_bins + 1)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        
        # Calculate semivariance for each bin
        semivariance = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        
        for i in range(len(self.values)):
            for j in range(i+1, len(self.values)):
                dist = distances[i,j]
                bin_idx = np.digitize(dist, bins) - 1
                
                if 0 <= bin_idx < n_bins:
                    semivar = 0.5 * (self.values[i] - self.values[j])**2
                    semivariance[bin_idx] += semivar
                    counts[bin_idx] += 1
                    
        # Average semivariance in each bin
        mask = counts > 0
        semivariance[mask] /= counts[mask]
        
        return bin_centers, semivariance
    
    def moran_i(self):
        """Calculate Moran's I spatial autocorrelation"""
        distances = self.distance_matrix()
        weights = 1 / (distances + np.eye(len(distances)))  # Add eye to avoid division by zero
        weights = weights / np.sum(weights)
        
        # Standardize values
        z = (self.values - np.mean(self.values)) / np.std(self.values)
        
        # Calculate Moran's I
        n = len(self.values)
        numerator = n * np.sum(weights * np.outer(z, z))
        denominator = np.sum(weights) * np.sum(z**2)
        
        return numerator / denominator

# Example usage
n_points = 100
coordinates = np.random.rand(n_points, 2) * 10
values = np.sin(coordinates[:,0]) * np.cos(coordinates[:,1]) + np.random.normal(0, 0.1, n_points)

spatial = SpatialStatistics(coordinates, values)

# Calculate empirical variogram
distances, semivariance = spatial.empirical_variogram()
print("First 5 variogram values:", semivariance[:5])

# Calculate Moran's I
moran_i = spatial.moran_i()
print("Moran's I:", moran_i)
```

Slide 13: Model Diagnostics and Validation

Implementation of comprehensive statistical diagnostics tools including residual analysis, influence measures, and cross-validation techniques for model validation.

```python
class ModelDiagnostics:
    def __init__(self, y_true, y_pred, X=None):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.X = np.array(X) if X is not None else None
        self.residuals = self.y_true - self.y_pred
        
    def residual_analysis(self):
        """Comprehensive residual analysis"""
        results = {
            'standardized_residuals': self._standardized_residuals(),
            'normality_test': self._shapiro_wilk_test(),
            'heteroscedasticity': self._breusch_pagan_test(),
            'autocorrelation': self._durbin_watson()
        }
        return results
    
    def _standardized_residuals(self):
        """Calculate standardized residuals"""
        std_residuals = (self.residuals - np.mean(self.residuals)) / np.std(self.residuals)
        return std_residuals
    
    def _shapiro_wilk_test(self):
        """Test residuals for normality"""
        std_residuals = self._standardized_residuals()
        W = np.zeros(len(std_residuals))
        sorted_res = np.sort(std_residuals)
        
        # Simplified Shapiro-Wilk implementation
        n = len(sorted_res)
        a = np.zeros(n // 2)
        for i in range(n // 2):
            a[i] = (sorted_res[n-1-i] - sorted_res[i]) / np.sqrt(np.sum(sorted_res**2))
        
        W = np.sum(a**2) / np.sum((sorted_res - np.mean(sorted_res))**2)
        return W
    
    def _breusch_pagan_test(self):
        """Test for heteroscedasticity"""
        if self.X is None:
            return None
            
        # Fit auxiliary regression
        squared_residuals = self.residuals**2
        aux_model = np.linalg.lstsq(self.X, squared_residuals, rcond=None)[0]
        fitted_values = self.X @ aux_model
        
        # Calculate test statistic
        n = len(self.residuals)
        r_squared = 1 - np.sum((squared_residuals - fitted_values)**2) / np.sum((squared_residuals - np.mean(squared_residuals))**2)
        bp_stat = n * r_squared
        
        return bp_stat
    
    def _durbin_watson(self):
        """Calculate Durbin-Watson statistic"""
        diff_residuals = np.diff(self.residuals)
        dw_stat = np.sum(diff_residuals**2) / np.sum(self.residuals**2)
        return dw_stat
    
    def cross_validation_metrics(self, cv_folds=5):
        """K-fold cross-validation metrics"""
        n = len(self.y_true)
        fold_size = n // cv_folds
        metrics = {
            'mse': np.zeros(cv_folds),
            'mae': np.zeros(cv_folds),
            'r2': np.zeros(cv_folds)
        }
        
        for i in range(cv_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < cv_folds-1 else n
            
            # Calculate metrics for fold
            fold_true = self.y_true[start_idx:end_idx]
            fold_pred = self.y_pred[start_idx:end_idx]
            
            metrics['mse'][i] = np.mean((fold_true - fold_pred)**2)
            metrics['mae'][i] = np.mean(np.abs(fold_true - fold_pred))
            metrics['r2'][i] = 1 - np.sum((fold_true - fold_pred)**2) / np.sum((fold_true - np.mean(fold_true))**2)
            
        return metrics

# Example usage
np.random.seed(42)
X = np.random.randn(200, 3)
true_coef = [1, -0.5, 2]
y_true = X @ true_coef + np.random.normal(0, 0.1, 200)
y_pred = X @ true_coef + np.random.normal(0, 0.2, 200)

diagnostics = ModelDiagnostics(y_true, y_pred, X)

# Get residual analysis
residual_results = diagnostics.residual_analysis()
print("Durbin-Watson statistic:", residual_results['autocorrelation'])

# Get cross-validation metrics
cv_results = diagnostics.cross_validation_metrics()
print("Mean R-squared across folds:", np.mean(cv_results['r2']))
```

Slide 14: Additional Resources

1.  "Statistical Learning with Sparsity: The Lasso and Generalizations" [https://arxiv.org/abs/1412.4280](https://arxiv.org/abs/1412.4280)
2.  "A Tutorial on Principal Component Analysis" [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
3.  "Bayesian Methods for Hackers: Probabilistic Programming" [https://arxiv.org/abs/1207.1346](https://arxiv.org/abs/1207.1346)
4.  "Modern Statistical Methods for Spatial Data Analysis" [https://arxiv.org/abs/1901.09649](https://arxiv.org/abs/1901.09649)
5.  "A Survey of Cross-validation Procedures for Model Selection" [https://arxiv.org/abs/0907.4728](https://arxiv.org/abs/0907.4728)

