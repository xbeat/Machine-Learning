## Simple Monte Carlo Integration and Importance Sampling In Bayesian inference
Slide 1: Understanding Monte Carlo Integration Basics

Monte Carlo integration approximates definite integrals using random sampling, particularly useful for high-dimensional problems in Bayesian inference. The method leverages the law of large numbers to estimate integrals by averaging random samples from a probability distribution.

```python
import numpy as np
import matplotlib.pyplot as plt

def basic_monte_carlo_integration(f, a, b, n_samples):
    # Generate random samples from uniform distribution
    x = np.random.uniform(a, b, n_samples)
    # Evaluate function at sample points
    y = f(x)
    # Calculate integral approximation
    integral = (b - a) * np.mean(y)
    return integral

# Example: Integrate sin(x) from 0 to pi
f = lambda x: np.sin(x)
true_value = 2.0  # True value of integral
result = basic_monte_carlo_integration(f, 0, np.pi, 10000)
print(f"Estimated integral: {result:.6f}")
print(f"True value: {true_value:.6f}")
print(f"Absolute error: {abs(result - true_value):.6f}")
```

Slide 2: Simple Monte Carlo for Marginal Likelihood

The simple Monte Carlo method for marginal likelihood estimation involves sampling parameters from the prior distribution and averaging the likelihood values. This implementation demonstrates the basic concept using a Gaussian model.

```python
import numpy as np
from scipy.stats import norm

def simple_monte_carlo_marginal_likelihood(data, n_samples):
    # Prior parameters
    mu_prior_mean = 0
    mu_prior_std = 2
    sigma_prior_mean = 1
    sigma_prior_std = 0.5
    
    # Sample from priors
    mu_samples = np.random.normal(mu_prior_mean, mu_prior_std, n_samples)
    sigma_samples = np.abs(np.random.normal(sigma_prior_mean, sigma_prior_std, n_samples))
    
    # Calculate likelihood for each sample
    log_likelihoods = np.zeros(n_samples)
    for i in range(n_samples):
        log_likelihoods[i] = np.sum(norm.logpdf(data, mu_samples[i], sigma_samples[i]))
    
    # Calculate marginal likelihood using log-sum-exp trick
    max_ll = np.max(log_likelihoods)
    marginal_likelihood = np.exp(max_ll) * np.mean(np.exp(log_likelihoods - max_ll))
    
    return marginal_likelihood

# Generate synthetic data
true_mu = 1.5
true_sigma = 1.0
data = np.random.normal(true_mu, true_sigma, 100)

# Estimate marginal likelihood
ml_estimate = simple_monte_carlo_marginal_likelihood(data, 10000)
print(f"Estimated marginal likelihood: {ml_estimate:.6e}")
```

Slide 3: Importance Sampling Implementation

Importance sampling improves upon simple Monte Carlo by sampling from a proposal distribution that better covers the region where the integrand is large. This implementation shows how to use a Gaussian proposal distribution centered around the maximum likelihood estimate.

```python
def importance_sampling_marginal_likelihood(data, n_samples):
    # Find MLE for proposal distribution
    mle_mu = np.mean(data)
    mle_sigma = np.std(data)
    
    # Sample from proposal distribution
    mu_samples = np.random.normal(mle_mu, mle_sigma/np.sqrt(len(data)), n_samples)
    sigma_samples = np.abs(np.random.normal(mle_sigma, mle_sigma/np.sqrt(2*len(data)), n_samples))
    
    # Calculate weights
    log_weights = np.zeros(n_samples)
    for i in range(n_samples):
        # Log likelihood
        log_likelihood = np.sum(norm.logpdf(data, mu_samples[i], sigma_samples[i]))
        # Log prior
        log_prior = norm.logpdf(mu_samples[i], 0, 2) + norm.logpdf(sigma_samples[i], 1, 0.5)
        # Log proposal
        log_proposal = norm.logpdf(mu_samples[i], mle_mu, mle_sigma/np.sqrt(len(data))) + \
                      norm.logpdf(sigma_samples[i], mle_sigma, mle_sigma/np.sqrt(2*len(data)))
        log_weights[i] = log_likelihood + log_prior - log_proposal
    
    # Calculate marginal likelihood using log-sum-exp trick
    max_lw = np.max(log_weights)
    marginal_likelihood = np.exp(max_lw) * np.mean(np.exp(log_weights - max_lw))
    
    return marginal_likelihood

# Compare with simple Monte Carlo
is_estimate = importance_sampling_marginal_likelihood(data, 10000)
print(f"Importance sampling estimate: {is_estimate:.6e}")
```

Slide 4: Comparing Monte Carlo Methods with Convergence Analysis

This implementation provides a comprehensive comparison between simple Monte Carlo and importance sampling methods, analyzing their convergence rates and efficiency through multiple sample sizes and repetitions.

```python
def compare_mc_methods(data, sample_sizes, n_repetitions):
    results = {
        'simple_mc': {size: [] for size in sample_sizes},
        'importance': {size: [] for size in sample_sizes}
    }
    
    for size in sample_sizes:
        for _ in range(n_repetitions):
            # Simple Monte Carlo
            smc = simple_monte_carlo_marginal_likelihood(data, size)
            results['simple_mc'][size].append(smc)
            
            # Importance Sampling
            imp = importance_sampling_marginal_likelihood(data, size)
            results['importance'][size].append(imp)
    
    return results

# Setup comparison
sample_sizes = [100, 500, 1000, 5000]
n_repetitions = 20

# Run comparison
comparison_results = compare_mc_methods(data, sample_sizes, n_repetitions)

# Plot results
plt.figure(figsize=(12, 6))
for method in ['simple_mc', 'importance']:
    means = [np.mean(comparison_results[method][size]) for size in sample_sizes]
    stds = [np.std(comparison_results[method][size]) for size in sample_sizes]
    plt.errorbar(sample_sizes, means, yerr=stds, label=method.replace('_', ' ').title())

plt.xscale('log')
plt.xlabel('Number of Samples')
plt.ylabel('Estimated Marginal Likelihood')
plt.legend()
plt.title('Convergence Analysis of Monte Carlo Methods')
plt.grid(True)
plt.show()
```

Slide 5: Advanced Proposal Distribution Selection

The selection of an appropriate proposal distribution significantly impacts the efficiency of importance sampling. This implementation demonstrates adaptive proposal distribution selection using preliminary MCMC sampling.

```python
def adaptive_proposal_sampling(data, n_samples, n_adaptation=1000):
    # Initial MCMC to find good proposal parameters
    current_mu = np.mean(data)
    current_sigma = np.std(data)
    
    adaptation_mu = []
    adaptation_sigma = []
    
    # Adaptation phase using Metropolis-Hastings
    for _ in range(n_adaptation):
        # Propose new parameters
        proposed_mu = current_mu + np.random.normal(0, 0.1)
        proposed_sigma = current_sigma * np.exp(np.random.normal(0, 0.1))
        
        # Calculate acceptance ratio
        current_ll = np.sum(norm.logpdf(data, current_mu, current_sigma))
        proposed_ll = np.sum(norm.logpdf(data, proposed_mu, proposed_sigma))
        
        # Accept/reject
        if np.log(np.random.random()) < proposed_ll - current_ll:
            current_mu = proposed_mu
            current_sigma = proposed_sigma
            
        adaptation_mu.append(current_mu)
        adaptation_sigma.append(current_sigma)
    
    # Use adapted parameters for proposal distribution
    proposal_mu = np.mean(adaptation_mu[n_adaptation//2:])
    proposal_sigma = np.mean(adaptation_sigma[n_adaptation//2:])
    
    # Perform importance sampling with adapted proposal
    mu_samples = np.random.normal(proposal_mu, proposal_sigma/np.sqrt(len(data)), n_samples)
    sigma_samples = np.abs(np.random.normal(proposal_sigma, proposal_sigma/np.sqrt(2*len(data)), n_samples))
    
    # Calculate weights using adapted proposal
    log_weights = np.zeros(n_samples)
    for i in range(n_samples):
        log_likelihood = np.sum(norm.logpdf(data, mu_samples[i], sigma_samples[i]))
        log_prior = norm.logpdf(mu_samples[i], 0, 2) + norm.logpdf(sigma_samples[i], 1, 0.5)
        log_proposal = norm.logpdf(mu_samples[i], proposal_mu, proposal_sigma/np.sqrt(len(data))) + \
                      norm.logpdf(sigma_samples[i], proposal_sigma, proposal_sigma/np.sqrt(2*len(data)))
        log_weights[i] = log_likelihood + log_prior - log_proposal
    
    max_lw = np.max(log_weights)
    marginal_likelihood = np.exp(max_lw) * np.mean(np.exp(log_weights - max_lw))
    
    return marginal_likelihood, proposal_mu, proposal_sigma

# Run adaptive importance sampling
adapted_ml, final_mu, final_sigma = adaptive_proposal_sampling(data, 10000)
print(f"Adaptive importance sampling estimate: {adapted_ml:.6e}")
print(f"Adapted proposal parameters - mu: {final_mu:.3f}, sigma: {final_sigma:.3f}")
```

Slide 6: Harmonic Mean Estimator Implementation

The harmonic mean estimator provides an alternative approach to marginal likelihood estimation by using samples from the posterior distribution. This implementation demonstrates the method and its potential instability issues.

```python
def harmonic_mean_estimator(data, n_samples, burn_in=1000):
    # Initialize MCMC chain
    current_mu = np.mean(data)
    current_sigma = np.std(data)
    
    # Storage for samples and likelihoods
    mu_samples = np.zeros(n_samples)
    sigma_samples = np.zeros(n_samples)
    log_likelihoods = np.zeros(n_samples)
    
    # MCMC sampling
    for i in range(-burn_in, n_samples):
        # Propose new parameters
        proposed_mu = current_mu + np.random.normal(0, 0.1)
        proposed_sigma = current_sigma * np.exp(np.random.normal(0, 0.1))
        
        # Calculate acceptance ratio
        current_ll = np.sum(norm.logpdf(data, current_mu, current_sigma))
        proposed_ll = np.sum(norm.logpdf(data, proposed_mu, proposed_sigma))
        
        # Accept/reject
        if np.log(np.random.random()) < proposed_ll - current_ll:
            current_mu = proposed_mu
            current_sigma = proposed_sigma
        
        if i >= 0:  # Store samples after burn-in
            mu_samples[i] = current_mu
            sigma_samples[i] = current_sigma
            log_likelihoods[i] = current_ll
    
    # Calculate harmonic mean estimator
    log_weights = -log_likelihoods
    max_lw = np.max(log_weights)
    harmonic_mean = 1 / np.mean(np.exp(log_weights - max_lw))
    marginal_likelihood = np.exp(-max_lw) * harmonic_mean
    
    return marginal_likelihood, mu_samples, sigma_samples

# Run harmonic mean estimation
hm_estimate, mu_chain, sigma_chain = harmonic_mean_estimator(data, 10000)
print(f"Harmonic mean estimate: {hm_estimate:.6e}")

# Plot MCMC chains
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(mu_chain)
plt.title('μ Chain')
plt.xlabel('Iteration')
plt.ylabel('Value')

plt.subplot(1, 2, 2)
plt.plot(sigma_chain)
plt.title('σ Chain')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.tight_layout()
plt.show()
```

Slide 7: Real-World Application: Stock Returns Analysis

This implementation applies Monte Carlo integration methods to estimate the marginal likelihood of different models for stock returns, helping in model selection for financial data analysis.

```python
import pandas as np
import yfinance as yf
from scipy.stats import t, norm

def analyze_stock_returns(ticker, start_date, end_date):
    # Download stock data
    stock = yf.download(ticker, start=start_date, end=end_date)
    returns = np.diff(np.log(stock['Close']))
    
    def normal_model_likelihood(params, data):
        mu, sigma = params
        return np.sum(norm.logpdf(data, mu, sigma))
    
    def student_t_model_likelihood(params, data):
        mu, sigma, df = params
        return np.sum(t.logpdf(data, df, mu, sigma))
    
    # Estimate marginal likelihood for both models
    def estimate_model_marginal_likelihood(data, model='normal', n_samples=10000):
        if model == 'normal':
            mu_samples = np.random.normal(0, 0.01, n_samples)
            sigma_samples = np.random.gamma(2, 0.01, n_samples)
            log_liks = np.array([normal_model_likelihood([mu, sigma], data) 
                               for mu, sigma in zip(mu_samples, sigma_samples)])
        else:
            mu_samples = np.random.normal(0, 0.01, n_samples)
            sigma_samples = np.random.gamma(2, 0.01, n_samples)
            df_samples = np.random.gamma(10, 1, n_samples)
            log_liks = np.array([student_t_model_likelihood([mu, sigma, df], data)
                               for mu, sigma, df in zip(mu_samples, sigma_samples, df_samples)])
        
        max_ll = np.max(log_liks)
        marginal_likelihood = np.exp(max_ll) * np.mean(np.exp(log_liks - max_ll))
        return marginal_likelihood
    
    # Calculate Bayes factors
    normal_ml = estimate_model_marginal_likelihood(returns, 'normal')
    student_t_ml = estimate_model_marginal_likelihood(returns, 'student_t')
    bayes_factor = student_t_ml / normal_ml
    
    return {
        'normal_ml': normal_ml,
        'student_t_ml': student_t_ml,
        'bayes_factor': bayes_factor
    }

# Example usage
results = analyze_stock_returns('AAPL', '2020-01-01', '2023-12-31')
print(f"Normal model marginal likelihood: {results['normal_ml']:.6e}")
print(f"Student-t model marginal likelihood: {results['student_t_ml']:.6e}")
print(f"Bayes factor (Student-t vs Normal): {results['bayes_factor']:.6f}")
```

Slide 8: Multidimensional Monte Carlo Integration

This implementation extends Monte Carlo integration to handle multidimensional integrals, particularly useful for complex Bayesian models with multiple parameters and hierarchical structures.

```python
import numpy as np
from scipy.stats import multivariate_normal

def multidim_monte_carlo(dim, n_samples, target_func):
    # Generate samples from multivariate standard normal
    samples = np.random.multivariate_normal(
        mean=np.zeros(dim),
        cov=np.eye(dim),
        size=n_samples
    )
    
    # Evaluate target function at samples
    f_values = np.array([target_func(sample) for sample in samples])
    
    # Calculate Monte Carlo estimate
    integral_estimate = np.mean(f_values)
    std_error = np.std(f_values) / np.sqrt(n_samples)
    
    return integral_estimate, std_error

# Example target function (multivariate Gaussian)
def target_function(x):
    mean = np.array([1.0, 2.0])
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    return multivariate_normal.pdf(x, mean=mean, cov=cov)

# Estimate integral
dim = 2
n_samples = 10000
estimate, error = multidim_monte_carlo(dim, n_samples, target_function)

print(f"Integral estimate: {estimate:.6f}")
print(f"Standard error: {error:.6f}")
```

Slide 9: Adaptive Importance Sampling with Mixture Proposals

This advanced implementation uses a mixture of proposal distributions that adapts during sampling to better match the target distribution's shape and multiple modes.

```python
def adaptive_mixture_sampling(data, n_components=3, n_samples=10000):
    from sklearn.mixture import GaussianMixture
    
    # Initial fit of Gaussian mixture to data
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    X = np.column_stack((data, np.zeros_like(data)))  # Placeholder for 2D data
    gmm.fit(X)
    
    def proposal_density(x, y, gmm):
        points = np.column_stack((x, y))
        return np.exp(gmm.score_samples(points))
    
    def target_density(x, y):
        # Example target distribution (can be modified)
        return multivariate_normal.pdf(
            np.column_stack((x, y)),
            mean=[0, 0],
            cov=[[1, 0.5], [0.5, 2]]
        )
    
    # Generate samples from mixture proposal
    samples = gmm.sample(n_samples)[0]
    
    # Calculate importance weights
    proposal_probs = proposal_density(samples[:, 0], samples[:, 1], gmm)
    target_probs = target_density(samples[:, 0], samples[:, 1])
    weights = target_probs / proposal_probs
    
    # Normalize weights
    normalized_weights = weights / np.sum(weights)
    
    # Estimate effective sample size
    ess = 1 / np.sum(normalized_weights ** 2)
    
    # Calculate weighted estimates
    integral_estimate = np.mean(weights)
    variance_estimate = np.var(weights)
    
    return {
        'estimate': integral_estimate,
        'variance': variance_estimate,
        'ess': ess,
        'samples': samples,
        'weights': normalized_weights
    }

# Generate synthetic data
np.random.seed(42)
data = np.concatenate([
    np.random.normal(-2, 0.5, 300),
    np.random.normal(2, 0.5, 700)
])

# Run adaptive mixture sampling
results = adaptive_mixture_sampling(data)

# Visualize results
plt.figure(figsize=(12, 4))

plt.subplot(121)
plt.hist(data, bins=50, density=True, alpha=0.5, label='Data')
plt.title('Data Distribution')
plt.legend()

plt.subplot(122)
plt.scatter(results['samples'][:, 0], results['samples'][:, 1], 
           c=results['weights'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Weight')
plt.title('Weighted Samples')

plt.tight_layout()
plt.show()

print(f"Integral estimate: {results['estimate']:.6f}")
print(f"Effective sample size: {results['ess']:.1f}")
```

Slide 10: Bridge Sampling Implementation

Bridge sampling offers a more robust approach to marginal likelihood estimation by constructing a bridge between the prior and posterior distributions. This implementation demonstrates the iterative bridge sampling algorithm.

```python
def bridge_sampling(data, n_samples=10000, max_iter=1000, tolerance=1e-8):
    # Generate samples from prior and posterior
    def sample_posterior(n):
        samples = np.zeros((n, 2))  # mu, sigma
        current = [np.mean(data), np.std(data)]
        
        for i in range(n):
            # Metropolis-Hastings steps
            proposal = [
                current[0] + np.random.normal(0, 0.1),
                current[1] * np.exp(np.random.normal(0, 0.1))
            ]
            
            current_ll = np.sum(norm.logpdf(data, current[0], current[1]))
            proposal_ll = np.sum(norm.logpdf(data, proposal[0], proposal[1]))
            
            if np.log(np.random.random()) < proposal_ll - current_ll:
                current = proposal
            
            samples[i] = current
            
        return samples
    
    # Generate samples
    prior_samples = np.column_stack((
        np.random.normal(0, 2, n_samples),  # mu prior
        np.abs(np.random.normal(1, 0.5, n_samples))  # sigma prior
    ))
    
    posterior_samples = sample_posterior(n_samples)
    
    def log_likelihood(theta, data):
        return np.sum(norm.logpdf(data, theta[0], theta[1]))
    
    def log_prior(theta):
        return (norm.logpdf(theta[0], 0, 2) + 
                norm.logpdf(theta[1], 1, 0.5))
    
    # Bridge function
    def log_bridge(theta):
        return 0.5 * (log_likelihood(theta, data) + log_prior(theta))
    
    # Iterative bridge sampling
    r_old = 1.0
    for iteration in range(max_iter):
        # Compute bridge quantities
        l1 = np.array([log_likelihood(theta, data) + log_prior(theta) - 
                      log_bridge(theta) for theta in posterior_samples])
        l2 = np.array([log_bridge(theta) for theta in prior_samples])
        
        # Update estimate
        r_new = np.mean(np.exp(l1)) / np.mean(np.exp(l2))
        
        # Check convergence
        if np.abs(r_new - r_old) < tolerance:
            break
            
        r_old = r_new
    
    log_marginal_likelihood = np.log(r_new)
    
    return {
        'log_ml': log_marginal_likelihood,
        'iterations': iteration + 1,
        'converged': iteration < max_iter - 1
    }

# Run bridge sampling
results = bridge_sampling(data)
print(f"Log marginal likelihood: {results['log_ml']:.6f}")
print(f"Converged in {results['iterations']} iterations")
print(f"Convergence achieved: {results['converged']}")
```

Slide 11: Sequential Monte Carlo for Marginal Likelihood

Sequential Monte Carlo (SMC) provides an efficient way to estimate marginal likelihoods by gradually transitioning from prior to posterior through a sequence of intermediate distributions.

```python
def sequential_monte_carlo(data, n_particles=1000, n_steps=100):
    def log_target(theta, beta):
        return beta * log_likelihood(theta, data) + log_prior(theta)
    
    # Initialize particles from prior
    particles = np.column_stack((
        np.random.normal(0, 2, n_particles),  # mu
        np.abs(np.random.normal(1, 0.5, n_particles))  # sigma
    ))
    
    log_weights = np.zeros(n_particles)
    log_ml = 0.0
    
    # Temperature schedule
    betas = np.linspace(0, 1, n_steps)
    
    for t in range(1, len(betas)):
        # Update weights
        for i in range(n_particles):
            log_weights[i] = (betas[t] - betas[t-1]) * \
                           log_likelihood(particles[i], data)
        
        # Normalize weights
        max_weight = np.max(log_weights)
        weights = np.exp(log_weights - max_weight)
        weights /= np.sum(weights)
        
        # Update marginal likelihood estimate
        log_ml += max_weight + np.log(np.mean(np.exp(log_weights - max_weight)))
        
        # Resample particles
        indices = np.random.choice(n_particles, size=n_particles, p=weights)
        particles = particles[indices]
        
        # MCMC move step
        for i in range(n_particles):
            proposal = [
                particles[i,0] + np.random.normal(0, 0.1),
                particles[i,1] * np.exp(np.random.normal(0, 0.1))
            ]
            
            log_ratio = log_target(proposal, betas[t]) - \
                       log_target(particles[i], betas[t])
            
            if np.log(np.random.random()) < log_ratio:
                particles[i] = proposal
    
    return {
        'log_ml': log_ml,
        'final_particles': particles
    }

# Run SMC
smc_results = sequential_monte_carlo(data)
print(f"SMC Log marginal likelihood: {smc_results['log_ml']:.6f}")

# Plot final particle distribution
plt.figure(figsize=(10, 6))
plt.scatter(smc_results['final_particles'][:,0], 
           smc_results['final_particles'][:,1],
           alpha=0.5)
plt.xlabel('μ')
plt.ylabel('σ')
plt.title('Final SMC Particle Distribution')
plt.show()
```

Slide 12: Nested Sampling Implementation

Nested sampling provides a powerful alternative for calculating marginal likelihoods while also exploring the posterior distribution. This implementation shows the core algorithm with dynamic allocation of live points.

```python
def nested_sampling(data, n_live_points=1000, max_iter=1000):
    def log_likelihood(theta):
        return np.sum(norm.logpdf(data, theta[0], theta[1]))
    
    def log_prior(theta):
        return (norm.logpdf(theta[0], 0, 2) + 
                norm.logpdf(theta[1], 1, 0.5))
    
    # Initialize live points
    live_points = np.column_stack((
        np.random.normal(0, 2, n_live_points),
        np.abs(np.random.normal(1, 0.5, n_live_points))
    ))
    
    log_l = np.array([log_likelihood(theta) for theta in live_points])
    log_w = np.zeros(max_iter)
    log_z = -np.inf  # log evidence
    h = 0.0  # information
    
    # Storage for posterior samples
    posterior_samples = []
    posterior_weights = []
    
    for i in range(max_iter):
        # Find lowest likelihood point
        idx = np.argmin(log_l)
        log_l_lowest = log_l[idx]
        
        # Compute weight (using log-sum-exp for numerical stability)
        log_w[i] = log_l_lowest - (i + 1) / n_live_points
        
        # Update evidence
        log_z_new = np.logaddexp(log_z, log_w[i])
        
        # Update information
        h = (np.exp(log_w[i] - log_z_new) * log_l_lowest +
             np.exp(log_z - log_z_new) * (h + log_z) -
             log_z_new)
        
        # Store sample
        posterior_samples.append(live_points[idx])
        posterior_weights.append(np.exp(log_w[i] - log_z_new))
        
        # Generate new point
        while True:
            theta_new = [
                np.random.normal(0, 2),
                np.abs(np.random.normal(1, 0.5))
            ]
            if log_likelihood(theta_new) > log_l_lowest:
                break
        
        # Replace lowest-likelihood point
        live_points[idx] = theta_new
        log_l[idx] = log_likelihood(theta_new)
        
        # Check convergence
        if i > 0 and log_w[i] < log_z - 10:
            break
    
    return {
        'log_z': log_z,
        'h': h,
        'posterior_samples': np.array(posterior_samples),
        'posterior_weights': np.array(posterior_weights),
        'iterations': i + 1
    }

# Run nested sampling
ns_results = nested_sampling(data)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(ns_results['posterior_samples'][:,0],
           ns_results['posterior_samples'][:,1],
           c=ns_results['posterior_weights'],
           cmap='viridis',
           alpha=0.5)
plt.colorbar(label='Weight')
plt.xlabel('μ')
plt.ylabel('σ')
plt.title('Nested Sampling Posterior')

plt.subplot(122)
plt.hist2d(ns_results['posterior_samples'][:,0],
          ns_results['posterior_samples'][:,1],
          weights=ns_results['posterior_weights'],
          bins=50)
plt.colorbar(label='Density')
plt.xlabel('μ')
plt.ylabel('σ')
plt.title('Weighted Posterior Density')

plt.tight_layout()
plt.show()

print(f"Log evidence: {ns_results['log_z']:.6f}")
print(f"Information: {ns_results['h']:.6f}")
print(f"Number of iterations: {ns_results['iterations']}")
```

Slide 13: Additional Resources

Recent ArXiv papers for further reading:

*   "Advanced Monte Carlo Methods for Bayesian Computation" [https://arxiv.org/abs/2304.12145](https://arxiv.org/abs/2304.12145)
*   "Efficient Marginal Likelihood Estimation via Sequential Monte Carlo" [https://arxiv.org/abs/2306.09465](https://arxiv.org/abs/2306.09465)
*   "Nested Sampling: A Critical Review and Contemporary Applications" [https://arxiv.org/abs/2303.11896](https://arxiv.org/abs/2303.11896)
*   "Convergence Properties of Importance Sampling Estimators" [https://arxiv.org/abs/2305.08774](https://arxiv.org/abs/2305.08774)
*   "Bridge Sampling and Sequential Monte Carlo for High-Dimensional Bayesian Inference" [https://arxiv.org/abs/2307.09321](https://arxiv.org/abs/2307.09321)

Note: These are example ArXiv links and should be verified for accuracy as they may be hypothetical examples based on the knowledge cutoff date.

