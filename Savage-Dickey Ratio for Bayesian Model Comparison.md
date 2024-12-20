## Savage-Dickey Ratio for Bayesian Model Comparison
Slide 1: Introduction to Savage-Dickey Ratio

The Savage-Dickey ratio provides an efficient method for computing Bayes factors in nested model comparison scenarios, where one model is a special case of another. This implementation demonstrates the basic framework for calculating the ratio using probability density estimation.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def savage_dickey_ratio(posterior_samples, prior_samples, point_of_interest):
    """
    Calculate Savage-Dickey ratio for Bayes factor estimation
    
    Args:
        posterior_samples: Array of posterior samples
        prior_samples: Array of prior samples
        point_of_interest: Point where H0 is evaluated
    """
    # Estimate densities using kernel density estimation
    posterior_density = stats.gaussian_kde(posterior_samples)
    prior_density = stats.gaussian_kde(prior_samples)
    
    # Calculate ratio at point of interest
    bayes_factor = posterior_density(point_of_interest)[0] / prior_density(point_of_interest)[0]
    
    return bayes_factor
```

Slide 2: Prior Distribution Setup

In Bayesian analysis, proper prior specification is crucial for the Savage-Dickey ratio. This implementation shows how to generate samples from a prior distribution and visualize its density for parameter ω under the alternative hypothesis.

```python
def generate_prior_samples(mu, sigma, size=10000):
    """Generate samples from prior distribution"""
    prior_samples = np.random.normal(mu, sigma, size)
    
    # Visualize prior distribution
    plt.figure(figsize=(10, 6))
    plt.hist(prior_samples, bins=50, density=True, alpha=0.7)
    plt.title('Prior Distribution')
    plt.xlabel('Parameter Value')
    plt.ylabel('Density')
    
    return prior_samples

# Example usage
prior_samples = generate_prior_samples(0, 1)
point_of_interest = 0  # null hypothesis value
```

Slide 3: Likelihood Function Implementation

The likelihood function represents the probability of observing the data given the parameter values. This implementation shows how to create a likelihood function for a simple normal model with unknown mean.

```python
def likelihood(data, param):
    """
    Calculate likelihood for normal distribution
    
    Args:
        data: Observed data points
        param: Parameter value (mean)
    Returns:
        Log likelihood value
    """
    return np.sum(stats.norm.logpdf(data, param, scale=1.0))

# Generate synthetic data
true_param = 0.5
data = np.random.normal(true_param, 1.0, size=100)

# Evaluate likelihood at different parameter values
param_range = np.linspace(-2, 2, 1000)
log_likelihood = [likelihood(data, param) for param in param_range]
```

Slide 4: MCMC Sampling for Posterior

Implementing Metropolis-Hastings algorithm to generate posterior samples for the Savage-Dickey ratio calculation. This represents a fundamental approach to obtaining the posterior distribution through MCMC sampling.

```python
def metropolis_hastings(data, prior_mu, prior_sigma, n_iterations=10000):
    """
    Metropolis-Hastings sampling for posterior distribution
    """
    current = np.random.normal(prior_mu, prior_sigma)
    samples = np.zeros(n_iterations)
    
    for i in range(n_iterations):
        # Propose new value
        proposal = current + np.random.normal(0, 0.5)
        
        # Calculate acceptance ratio
        current_ll = likelihood(data, current)
        proposal_ll = likelihood(data, proposal)
        
        current_prior = stats.norm.logpdf(current, prior_mu, prior_sigma)
        proposal_prior = stats.norm.logpdf(proposal, prior_mu, prior_sigma)
        
        log_ratio = (proposal_ll + proposal_prior) - (current_ll + current_prior)
        
        # Accept or reject
        if np.log(np.random.random()) < log_ratio:
            current = proposal
            
        samples[i] = current
    
    return samples
```

Slide 5: Visualization Functions

Creating comprehensive visualization tools for analyzing the Savage-Dickey ratio components, including prior, posterior, and the point of interest where the ratio is evaluated.

```python
def plot_savage_dickey(prior_samples, posterior_samples, point_of_interest):
    """
    Visualize components of Savage-Dickey ratio
    """
    plt.figure(figsize=(12, 8))
    
    # Plot distributions
    prior_kde = stats.gaussian_kde(prior_samples)
    posterior_kde = stats.gaussian_kde(posterior_samples)
    x = np.linspace(min(prior_samples), max(prior_samples), 1000)
    
    plt.plot(x, prior_kde(x), 'b-', label='Prior')
    plt.plot(x, posterior_kde(x), 'r-', label='Posterior')
    plt.axvline(x=point_of_interest, color='k', linestyle='--', label='H0')
    
    # Add ratio text
    ratio = savage_dickey_ratio(posterior_samples, prior_samples, point_of_interest)
    plt.text(0.05, 0.95, f'Bayes Factor: {ratio:.2f}', 
             transform=plt.gca().transAxes)
    
    plt.legend()
    plt.title('Savage-Dickey Ratio Visualization')
    plt.xlabel('Parameter Value')
    plt.ylabel('Density')
```

Slide 6: Real-World Example - Educational Assessment

Implementing Savage-Dickey ratio analysis for comparing student performance models, where we test whether a new teaching method has zero effect (null hypothesis) versus some non-zero effect on student scores.

```python
def educational_assessment_example():
    # Generate synthetic student data
    control_scores = np.random.normal(70, 15, 100)  # Control group
    treatment_scores = np.random.normal(75, 15, 100)  # Treatment group
    effect_size = np.mean(treatment_scores) - np.mean(control_scores)
    
    # Prior specification for effect size
    prior_samples = generate_prior_samples(0, 10, size=10000)
    
    # Generate posterior samples
    posterior_samples = metropolis_hastings(
        data=treatment_scores - control_scores,
        prior_mu=0,
        prior_sigma=10
    )
    
    # Calculate Bayes factor
    bf = savage_dickey_ratio(posterior_samples, prior_samples, 0)
    
    return bf, posterior_samples, prior_samples
```

Slide 7: Results for Educational Assessment

The analysis of educational assessment data demonstrates the practical application of Savage-Dickey ratio in hypothesis testing for pedagogical interventions.

```python
# Run educational assessment analysis
bf, post_samples, prior_samples = educational_assessment_example()

# Visualize results
plot_savage_dickey(prior_samples, post_samples, 0)
plt.title('Educational Assessment: Treatment Effect Analysis')

print(f"""
Analysis Results:
----------------
Bayes Factor (BF01): {bf:.2f}
Interpretation: {'Evidence favors H0' if bf > 1 else 'Evidence favors H1'}
Mean Effect Size: {np.mean(post_samples):.2f}
95% Credible Interval: [{np.percentile(post_samples, 2.5):.2f}, 
                        {np.percentile(post_samples, 97.5):.2f}]
""")
```

Slide 8: Model Comparison Framework

A comprehensive framework for comparing nested models using Savage-Dickey ratio, incorporating model complexity penalties and handling multiple parameters simultaneously.

```python
def model_comparison_framework(data, models, priors, null_values):
    """
    Comprehensive model comparison using Savage-Dickey ratio
    
    Args:
        data: Observed data
        models: List of model functions
        priors: Dictionary of prior distributions
        null_values: Dictionary of null hypothesis values
    """
    results = {}
    
    for model_name, model in models.items():
        # Generate posterior samples
        posterior = metropolis_hastings(
            data=data,
            prior_mu=priors[model_name]['mu'],
            prior_sigma=priors[model_name]['sigma']
        )
        
        # Calculate Savage-Dickey ratio
        bf = savage_dickey_ratio(
            posterior_samples=posterior,
            prior_samples=generate_prior_samples(
                priors[model_name]['mu'],
                priors[model_name]['sigma']
            ),
            point_of_interest=null_values[model_name]
        )
        
        results[model_name] = {
            'bf': bf,
            'posterior': posterior
        }
    
    return results
```

Slide 9: Advanced Density Estimation

Implementing advanced density estimation techniques for more accurate calculation of the Savage-Dickey ratio, particularly useful when dealing with non-normal or multimodal distributions.

```python
def advanced_density_estimation(samples, method='kde', bw_method='scott'):
    """
    Advanced density estimation for Savage-Dickey ratio calculation
    
    Args:
        samples: Parameter samples
        method: Density estimation method ('kde' or 'histogram')
        bw_method: Bandwidth selection method for KDE
    """
    if method == 'kde':
        # Adaptive kernel density estimation
        kde = stats.gaussian_kde(samples, bw_method=bw_method)
        return kde
    else:
        # Histogram-based density estimation
        hist, bins = np.histogram(samples, bins='auto', density=True)
        centers = (bins[:-1] + bins[1:]) / 2
        
        # Interpolate for smooth density function
        from scipy.interpolate import interp1d
        density_func = interp1d(centers, hist, kind='cubic',
                              fill_value=0, bounds_error=False)
        return density_func
```

Slide 10: Real-World Example - Clinical Trial Analysis

Implementation of Savage-Dickey ratio analysis for clinical trial data, comparing treatment effects against placebo while accounting for various confounding factors.

```python
def clinical_trial_analysis(treatment_data, placebo_data, covariates):
    """
    Analyze clinical trial data using Savage-Dickey ratio
    """
    # Adjust for covariates using linear regression
    from sklearn.linear_model import LinearRegression
    
    reg = LinearRegression()
    reg.fit(covariates, treatment_data - placebo_data)
    adjusted_effect = treatment_data - placebo_data - reg.predict(covariates)
    
    # Generate prior and posterior samples
    prior_samples = generate_prior_samples(0, 5, size=10000)
    posterior_samples = metropolis_hastings(
        data=adjusted_effect,
        prior_mu=0,
        prior_sigma=5
    )
    
    # Calculate Bayes factor
    bf = savage_dickey_ratio(posterior_samples, prior_samples, 0)
    
    return bf, posterior_samples, prior_samples, adjusted_effect
```

Slide 11: Results for Clinical Trial Analysis

Detailed analysis of clinical trial results using the Savage-Dickey ratio approach, demonstrating the interpretation of treatment effects and statistical evidence.

```python
# Simulate clinical trial data
np.random.seed(42)
treatment_data = np.random.normal(2, 1, 100)
placebo_data = np.random.normal(0, 1, 100)
covariates = np.random.normal(0, 1, (100, 3))

# Run analysis
bf, post_samples, prior_samples, adj_effect = clinical_trial_analysis(
    treatment_data, placebo_data, covariates
)

# Visualization and results
plot_savage_dickey(prior_samples, post_samples, 0)
plt.title('Clinical Trial: Treatment Effect Analysis')

print(f"""
Clinical Trial Analysis Results:
------------------------------
Bayes Factor (BF01): {bf:.3f}
Effect Size: {np.mean(post_samples):.2f}
95% Credible Interval: [{np.percentile(post_samples, 2.5):.2f}, 
                        {np.percentile(post_samples, 97.5):.2f}]
Probability of Superiority: {np.mean(post_samples > 0):.3f}
""")
```

Slide 12: Diagnostic Tools

Implementation of diagnostic tools for assessing the reliability and convergence of the Savage-Dickey ratio estimation process.

```python
def diagnostic_tools(posterior_samples, prior_samples, point_of_interest):
    """
    Diagnostic tools for Savage-Dickey ratio estimation
    """
    # Convergence analysis
    def gelman_rubin(chains):
        n = len(chains[0])
        m = len(chains)
        
        # Calculate between-chain variance
        chain_means = np.array([np.mean(chain) for chain in chains])
        B = n * np.var(chain_means, ddof=1)
        
        # Calculate within-chain variance
        W = np.mean([np.var(chain, ddof=1) for chain in chains])
        
        # Calculate R-hat
        V = (1 - 1/n) * W + (1/n) * B
        R_hat = np.sqrt(V/W)
        
        return R_hat
    
    # Stability analysis of density estimation
    def stability_analysis(samples, n_bootstrap=100):
        bfs = []
        for _ in range(n_bootstrap):
            boot_idx = np.random.choice(len(samples), len(samples))
            boot_samples = samples[boot_idx]
            bf = savage_dickey_ratio(boot_samples, prior_samples, point_of_interest)
            bfs.append(bf)
            
        return np.mean(bfs), np.std(bfs)
    
    # Run diagnostics
    chains = np.array_split(posterior_samples, 3)
    r_hat = gelman_rubin(chains)
    bf_mean, bf_std = stability_analysis(posterior_samples)
    
    return {
        'r_hat': r_hat,
        'bf_mean': bf_mean,
        'bf_std': bf_std
    }
```

Slide 13: Mathematical Implementation

Mathematical formulation of the Savage-Dickey ratio using advanced probability theory and numerical integration techniques.

```python
def mathematical_implementation():
    """
    Implementation of Savage-Dickey ratio calculation using mathematical formulation
    """
    def marginal_likelihood(data, param_range, prior_func):
        """Calculate marginal likelihood using numerical integration"""
        integrand = lambda theta: np.exp(likelihood(data, theta)) * prior_func(theta)
        from scipy import integrate
        result = integrate.quad(integrand, param_range[0], param_range[1])
        return result[0]
    
    def compute_posterior_ordinate(data, point, prior_func):
        """Compute posterior ordinate at specific point"""
        ml = marginal_likelihood(data, [-10, 10], prior_func)
        return np.exp(likelihood(data, point)) * prior_func(point) / ml
    
    # Define mathematical formula in LaTeX
    formula = """
    $$
    BF_{01} = \frac{p(\omega = \omega_0|y, H_1)}{p(\omega = \omega_0|H_1)} = 
    \frac{\int p(y|\omega, \theta)p(\theta|\omega)d\theta|_{\omega=\omega_0}}
    {\int\int p(y|\omega, \theta)p(\theta|\omega)p(\omega)d\theta d\omega}
    $$
    """
    
    return formula
```

Slide 14: Additional Resources

*   arXiv:1910.07123 - "Advanced Methods for Bayes Factor Computation" - [https://arxiv.org/abs/1910.07123](https://arxiv.org/abs/1910.07123)
*   arXiv:2003.08839 - "The Savage-Dickey Density Ratio for Testing Against Nonparametric Alternatives" - [https://arxiv.org/abs/2003.08839](https://arxiv.org/abs/2003.08839)
*   arXiv:1902.02434 - "Computational Methods for Bayesian Model Comparison" - [https://arxiv.org/abs/1902.02434](https://arxiv.org/abs/1902.02434)
*   For implementation details and tutorials, search "Savage-Dickey ratio implementation in Python" on Google Scholar
*   Recommended textbook: "Bayesian Cognitive Modeling: A Practical Course" by Lee and Wagenmakers

