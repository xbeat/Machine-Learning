## Implementing Metropolis-Hastings Algorithm in Bayesian Inference
Slide 1: Introduction to Metropolis-Hastings Algorithm

The Metropolis-Hastings algorithm is a powerful Markov Chain Monte Carlo method used to sample from probability distributions where direct sampling is difficult. It generates a sequence of random samples that converge to the desired target distribution through an acceptance-rejection mechanism.

```python
import numpy as np
import matplotlib.pyplot as plt

def metropolis_hastings(target_distribution, proposal_distribution, initial_value, n_iterations):
    """
    Basic implementation of Metropolis-Hastings algorithm
    
    Args:
        target_distribution: Function representing the target distribution
        proposal_distribution: Function to generate proposals
        initial_value: Starting point
        n_iterations: Number of iterations
    """
    current = initial_value
    samples = [current]
    
    for _ in range(n_iterations):
        # Generate proposal
        proposal = proposal_distribution(current)
        
        # Calculate acceptance ratio
        ratio = target_distribution(proposal) / target_distribution(current)
        
        # Accept or reject
        if np.random.random() < ratio:
            current = proposal
            
        samples.append(current)
    
    return np.array(samples)
```

Slide 2: Target Distribution Implementation

To demonstrate the algorithm, we'll implement a simple target distribution - a mixture of two Gaussian distributions. This example shows how to define the target distribution and its mathematical formulation in practical code.

```python
def target_dist(x):
    """
    Target distribution: Mixture of two Gaussians
    Mathematical form: 
    $$p(x) = 0.3 * N(x|-2,1) + 0.7 * N(x|2,1.5)$$
    """
    return 0.3 * np.exp(-0.5 * (x + 2)**2) + \
           0.7 * np.exp(-0.5 * ((x - 2)/1.5)**2)

def proposal_dist(x):
    """
    Random walk proposal distribution
    $$q(x'|x) = N(x'|x,1)$$
    """
    return x + np.random.normal(0, 1)
```

Slide 3: Implementation of Single-Parameter MH Sampler

The core implementation demonstrates sampling from a univariate distribution. This version includes diagnostic information and proper handling of numerical stability through log-probability calculations.

```python
def mh_sampler_single_param(target_log_pdf, n_iterations, initial_value, proposal_width=0.5):
    """
    Single-parameter Metropolis-Hastings sampler with logging
    """
    samples = np.zeros(n_iterations)
    accepted = 0
    current = initial_value
    
    for i in range(n_iterations):
        # Generate proposal
        proposal = current + np.random.normal(0, proposal_width)
        
        # Calculate log acceptance ratio
        log_ratio = target_log_pdf(proposal) - target_log_pdf(current)
        
        # Accept/reject step
        if np.log(np.random.random()) < log_ratio:
            current = proposal
            accepted += 1
            
        samples[i] = current
    
    acceptance_rate = accepted / n_iterations
    return samples, acceptance_rate
```

Slide 4: Real-World Example - Bayesian Linear Regression

Implementing Metropolis-Hastings for Bayesian linear regression demonstrates its practical application in statistical modeling. This example includes prior specification and likelihood calculation.

```python
def bayesian_linear_regression_mh():
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    true_slope, true_intercept = 2.5, 1.0
    y = true_slope * X + true_intercept + np.random.normal(0, 1, 100)
    
    def log_likelihood(params):
        slope, intercept = params
        y_pred = slope * X + intercept
        return -0.5 * np.sum((y - y_pred)**2)  # Gaussian likelihood
    
    def log_prior(params):
        slope, intercept = params
        return -0.5 * (slope**2 + intercept**2)  # Gaussian prior
    
    def log_posterior(params):
        return log_likelihood(params) + log_prior(params)
    
    return log_posterior
```

Slide 5: Multivariate Metropolis-Hastings

The multivariate implementation handles parameters that need to be sampled jointly, using matrix operations for efficiency and proper handling of covariance structures in the proposal distribution.

```python
def multivariate_mh_sampler(log_target, dim, n_iterations):
    """
    Multivariate Metropolis-Hastings sampler
    """
    samples = np.zeros((n_iterations, dim))
    current = np.random.randn(dim)  # Initial state
    proposal_cov = np.eye(dim) * 0.1
    
    for i in range(n_iterations):
        # Generate multivariate normal proposal
        proposal = np.random.multivariate_normal(current, proposal_cov)
        
        # Calculate log acceptance ratio
        log_ratio = log_target(proposal) - log_target(current)
        
        if np.log(np.random.random()) < log_ratio:
            current = proposal
            
        samples[i] = current
        
    return samples
```

Slide 6: Adaptive Metropolis-Hastings

The adaptive version modifies the proposal distribution during sampling to improve efficiency. It updates the covariance matrix based on accepted samples, leading to better exploration of the target distribution.

```python
def adaptive_metropolis_hastings(target_dist, n_iterations, initial_value, adaptation_start=100):
    """
    Implements adaptive Metropolis-Hastings with covariance updating
    """
    dim = len(initial_value)
    samples = np.zeros((n_iterations, dim))
    current = initial_value
    
    # Initial proposal covariance
    proposal_cov = np.eye(dim) * 0.1
    accepted = 0
    
    for i in range(n_iterations):
        # Update proposal covariance after burn-in
        if i > adaptation_start and i % 50 == 0:
            proposal_cov = np.cov(samples[max(0, i-1000):i].T) * 2.4**2 / dim
            proposal_cov += np.eye(dim) * 1e-6  # Add small diagonal term for stability
        
        # Generate proposal
        proposal = np.random.multivariate_normal(current, proposal_cov)
        
        # Calculate acceptance ratio
        ratio = target_dist(proposal) / target_dist(current)
        
        if np.random.random() < ratio:
            current = proposal
            accepted += 1
            
        samples[i] = current
        
    return samples, accepted/n_iterations
```

Slide 7: Convergence Diagnostics

Implementing proper convergence diagnostics is crucial for determining when the chain has reached its stationary distribution and produced reliable samples.

```python
def convergence_diagnostics(chains):
    """
    Implements Gelman-Rubin diagnostic for multiple chains
    """
    n_chains, n_iterations, n_params = chains.shape
    
    # Calculate within-chain variance
    W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)
    
    # Calculate between-chain variance
    chain_means = np.mean(chains, axis=1)
    B = n_iterations * np.var(chain_means, axis=0, ddof=1)
    
    # Calculate potential scale reduction factor
    var_theta = ((n_iterations - 1) * W + B) / n_iterations
    R_hat = np.sqrt(var_theta / W)
    
    # Calculate effective sample size
    n_eff = n_chains * n_iterations * np.minimum(var_theta / B, 1)
    
    return R_hat, n_eff
```

Slide 8: Real-World Example - Mixture Model Estimation

A practical implementation of using Metropolis-Hastings to estimate parameters of a Gaussian mixture model, demonstrating its application in density estimation.

```python
def mixture_model_mh():
    # Generate synthetic data from mixture
    np.random.seed(42)
    n_samples = 1000
    true_means = [-2, 2]
    true_weights = [0.3, 0.7]
    
    # Generate mixture data
    data = np.concatenate([
        np.random.normal(true_means[0], 1, int(n_samples * true_weights[0])),
        np.random.normal(true_means[1], 1, int(n_samples * true_weights[1]))
    ])
    
    def log_likelihood(params):
        mu1, mu2, w = params
        # Ensure weight is between 0 and 1
        if not 0 <= w <= 1:
            return -np.inf
            
        # Calculate mixture likelihood
        likelihood1 = w * np.exp(-0.5 * (data - mu1)**2)
        likelihood2 = (1-w) * np.exp(-0.5 * (data - mu2)**2)
        return np.sum(np.log(likelihood1 + likelihood2))
        
    return log_likelihood, data
```

Slide 9: Source Code for Mixture Model Implementation

This implementation shows how to run the MCMC chain for the mixture model and process the results.

```python
def run_mixture_model_mcmc(n_iterations=10000):
    log_likelihood, data = mixture_model_mh()
    
    # Initial values
    initial_params = np.array([0.0, 0.0, 0.5])  # [mu1, mu2, weight]
    
    def proposal(params):
        return params + np.random.normal(0, 0.1, size=3)
    
    # Run MCMC
    samples = np.zeros((n_iterations, 3))
    current = initial_params
    
    for i in range(n_iterations):
        proposed = proposal(current)
        
        # Calculate log acceptance ratio
        log_ratio = log_likelihood(proposed) - log_likelihood(current)
        
        if np.log(np.random.random()) < log_ratio:
            current = proposed
            
        samples[i] = current
        
    return samples, data

# Example usage and visualization
samples, data = run_mixture_model_mcmc()
```

Slide 10: MCMC Chain Visualization

Understanding how to visualize MCMC chains is crucial for diagnostics and result interpretation. This implementation provides comprehensive plotting utilities.

```python
def plot_mcmc_diagnostics(samples, burnin=1000):
    """
    Creates diagnostic plots for MCMC samples
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    
    # Trace plots
    for i in range(3):
        axes[i, 0].plot(samples[burnin:, i])
        axes[i, 0].set_title(f'Trace Plot for Parameter {i+1}')
        
    # Density plots
    for i in range(3):
        axes[i, 1].hist(samples[burnin:, i], bins=50, density=True)
        axes[i, 1].set_title(f'Density Plot for Parameter {i+1}')
    
    plt.tight_layout()
    return fig

def plot_autocorrelation(samples, max_lag=50):
    """
    Plots autocorrelation for each parameter
    """
    n_params = samples.shape[1]
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 3*n_params))
    
    for i in range(n_params):
        acf = np.correlate(samples[:, i] - np.mean(samples[:, i]), 
                          samples[:, i] - np.mean(samples[:, i]), 
                          mode='full')[len(samples[:, i])-1:]
        acf = acf[:max_lag] / acf[0]
        axes[i].plot(acf)
        axes[i].set_title(f'Autocorrelation for Parameter {i+1}')
    
    plt.tight_layout()
    return fig
```

Slide 11: Performance Optimization with Parallel Chains

Implementing parallel chains improves sampling efficiency and enables better convergence assessment. This implementation uses Python's multiprocessing capabilities for parallel chain execution.

```python
from multiprocessing import Pool
import numpy as np

def run_parallel_chains(target_dist, n_chains=4, n_iterations=10000):
    """
    Runs multiple Metropolis-Hastings chains in parallel
    """
    def run_single_chain(seed):
        np.random.seed(seed)
        samples = np.zeros(n_iterations)
        current = np.random.randn()
        
        for i in range(n_iterations):
            proposal = current + np.random.normal(0, 0.5)
            ratio = target_dist(proposal) / target_dist(current)
            
            if np.random.random() < ratio:
                current = proposal
            samples[i] = current
            
        return samples
    
    # Run chains in parallel
    with Pool(processes=n_chains) as pool:
        chains = pool.map(run_single_chain, range(n_chains))
    
    return np.array(chains)
```

Slide 12: Implementing Reversible Jump MCMC

Reversible Jump MCMC extends Metropolis-Hastings to handle models with varying dimensions, useful for model selection and mixture modeling with unknown components.

```python
def reversible_jump_mh(data, max_components=5, n_iterations=10000):
    """
    Implements Reversible Jump MCMC for Gaussian mixture models
    """
    def birth_step(current_model):
        k = len(current_model['means'])
        if k >= max_components:
            return current_model, 0
        
        # Propose new component
        new_mean = np.random.normal(np.mean(data), np.std(data))
        new_weight = np.random.beta(1, k)
        
        # Adjust weights
        weights = current_model['weights'] * (1 - new_weight)
        weights = np.append(weights, new_weight)
        
        return {
            'means': np.append(current_model['means'], new_mean),
            'weights': weights
        }, 1
    
    def death_step(current_model):
        k = len(current_model['means'])
        if k <= 1:
            return current_model, 0
            
        # Remove random component
        remove_idx = np.random.randint(k)
        means = np.delete(current_model['means'], remove_idx)
        weights = np.delete(current_model['weights'], remove_idx)
        weights = weights / np.sum(weights)  # Renormalize
        
        return {'means': means, 'weights': weights}, 1
    
    # Initialize
    model = {'means': [np.mean(data)], 'weights': [1.0]}
    models = []
    
    for _ in range(n_iterations):
        # Randomly choose birth or death step
        if np.random.random() < 0.5:
            proposed_model, success = birth_step(model)
        else:
            proposed_model, success = death_step(model)
            
        if success:
            # Calculate acceptance ratio (simplified)
            current_likelihood = gaussian_mixture_likelihood(data, model)
            proposed_likelihood = gaussian_mixture_likelihood(data, proposed_model)
            
            if np.random.random() < proposed_likelihood / current_likelihood:
                model = proposed_model
                
        models.append(model.copy())
    
    return models
```

Slide 13: Handling Non-Standard Distributions

Implementation for sampling from non-standard distributions using transformation and rejection sampling within Metropolis-Hastings framework.

```python
def sample_nonstandard_distribution(target_pdf, n_samples=10000):
    """
    Samples from non-standard distributions using transformation
    """
    def transform_sample(x, transform_type='log'):
        if transform_type == 'log':
            return np.exp(x)
        elif transform_type == 'logit':
            return 1 / (1 + np.exp(-x))
        return x
    
    def proposal_kernel(x, scale=0.1):
        return x + scale * np.random.normal()
    
    samples = np.zeros(n_samples)
    current = np.random.randn()  # Initial state
    accepted = 0
    
    for i in range(n_samples):
        proposal = proposal_kernel(current)
        
        # Transform samples if needed
        current_transformed = transform_sample(current)
        proposal_transformed = transform_sample(proposal)
        
        # Calculate acceptance ratio with Jacobian adjustment
        ratio = (target_pdf(proposal_transformed) / target_pdf(current_transformed)) * \
                np.abs(np.exp(proposal) / np.exp(current))
        
        if np.random.random() < ratio:
            current = proposal
            accepted += 1
            
        samples[i] = transform_sample(current)
    
    return samples, accepted/n_samples
```

Slide 14: Additional Resources

*   "An Introduction to MCMC for Machine Learning" - [https://www.cs.ubc.ca/~arnaud/andrieu\_defreitas\_doucet\_jordan\_intromontecarlomachinelearning.pdf](https://www.cs.ubc.ca/~arnaud/andrieu_defreitas_doucet_jordan_intromontecarlomachinelearning.pdf)
*   "Understanding the Metropolis-Hastings Algorithm" - [https://arxiv.org/abs/1504.01896](https://arxiv.org/abs/1504.01896)
*   "Adaptive Proposals for Efficient MCMC Methods" - [https://projecteuclid.org/journals/bayesian-analysis/volume-8/issue-3/Adaptive-Proposals-for-Efficient-MCMC-Methods/10.1214/13-BA815.full](https://projecteuclid.org/journals/bayesian-analysis/volume-8/issue-3/Adaptive-Proposals-for-Efficient-MCMC-Methods/10.1214/13-BA815.full)
*   "A Tutorial on Reversible Jump MCMC" - [http://people.ee.duke.edu/~lcarin/rjmcmc-tutorial.pdf](http://people.ee.duke.edu/~lcarin/rjmcmc-tutorial.pdf)
*   Suggested search terms for Google Scholar:
    *   "Metropolis-Hastings convergence diagnostics"
    *   "Adaptive MCMC methods"
    *   "Efficient MCMC sampling techniques"

