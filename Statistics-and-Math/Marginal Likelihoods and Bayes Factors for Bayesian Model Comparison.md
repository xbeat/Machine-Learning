## Marginal Likelihoods and Bayes Factors for Bayesian Model Comparison

Slide 1: Marginal Likelihood Fundamentals

The marginal likelihood represents the probability of observing data under a specific model by integrating over all possible parameter values. This fundamental concept forms the backbone of Bayesian model comparison and selection methods in statistical inference.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def marginal_likelihood_normal(data, mu_prior, sigma_prior, sigma_likelihood):
    n = len(data)
    sample_mean = np.mean(data)
    
    # Calculate posterior parameters
    sigma_posterior = 1 / (1/sigma_prior**2 + n/sigma_likelihood**2)
    mu_posterior = sigma_posterior * (mu_prior/sigma_prior**2 + 
                                    n*sample_mean/sigma_likelihood**2)
    
    # Calculate marginal likelihood
    ml = stats.norm.pdf(data, loc=mu_prior, scale=np.sqrt(sigma_prior**2 + 
                                                         sigma_likelihood**2))
    return np.prod(ml)

# Example usage
data = np.random.normal(2, 1, 100)
ml = marginal_likelihood_normal(data, mu_prior=0, sigma_prior=2, sigma_likelihood=1)
print(f"Marginal Likelihood: {ml:.10f}")
```

Slide 2: Bayes Factor Implementation

Bayes factors provide a quantitative measure for comparing two competing models by taking the ratio of their respective marginal likelihoods, offering a natural Bayesian approach to hypothesis testing and model selection.

```python
def bayes_factor(data, model1_params, model2_params):
    # Calculate marginal likelihoods for both models
    ml1 = marginal_likelihood_normal(data, **model1_params)
    ml2 = marginal_likelihood_normal(data, **model2_params)
    
    # Computing Bayes Factor
    bf = ml1 / ml2
    
    # Interpret Bayes Factor
    if bf > 100:
        interpretation = "Decisive evidence for Model 1"
    elif bf > 10:
        interpretation = "Strong evidence for Model 1"
    elif bf > 3.2:
        interpretation = "Substantial evidence for Model 1"
    elif bf > 1:
        interpretation = "Weak evidence for Model 1"
    else:
        interpretation = f"Evidence supports Model 2 (BF = {1/bf:.2f})"
    
    return bf, interpretation

# Example usage
model1 = {'mu_prior': 2, 'sigma_prior': 1, 'sigma_likelihood': 1}
model2 = {'mu_prior': 0, 'sigma_prior': 1, 'sigma_likelihood': 1}

bf, interp = bayes_factor(data, model1, model2)
print(f"Bayes Factor: {bf:.2f}")
print(f"Interpretation: {interp}")
```

Slide 3: Prior Distribution Implementation

The prior distribution encapsulates our beliefs about parameter values before observing data. This implementation demonstrates how to create and visualize different prior distributions for Bayesian model comparison.

```python
def create_prior_distribution(prior_type, params, n_samples=10000):
    if prior_type == 'normal':
        samples = np.random.normal(params['mu'], params['sigma'], n_samples)
    elif prior_type == 'uniform':
        samples = np.random.uniform(params['low'], params['high'], n_samples)
    elif prior_type == 'beta':
        samples = np.random.beta(params['a'], params['b'], n_samples)
    
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=50, density=True, alpha=0.7)
    plt.title(f'{prior_type.capitalize()} Prior Distribution')
    plt.xlabel('Parameter Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return samples

# Example usage
normal_params = {'mu': 0, 'sigma': 1}
uniform_params = {'low': -3, 'high': 3}
beta_params = {'a': 2, 'b': 5}

normal_samples = create_prior_distribution('normal', normal_params)
uniform_samples = create_prior_distribution('uniform', uniform_params)
beta_samples = create_prior_distribution('beta', beta_params)
```

Slide 4: Model Evidence Calculation

Calculating model evidence involves integrating the likelihood function over all possible parameter values weighted by the prior distribution. This implementation uses numerical integration to compute model evidence.

```python
def compute_model_evidence(data, prior_samples, likelihood_func):
    n_samples = len(prior_samples)
    evidences = np.zeros(n_samples)
    
    for i, theta in enumerate(prior_samples):
        # Calculate likelihood for each parameter value
        likelihood = likelihood_func(data, theta)
        evidences[i] = likelihood
    
    # Monte Carlo integration
    model_evidence = np.mean(evidences)
    
    return model_evidence

def gaussian_likelihood(data, theta):
    return np.prod(stats.norm.pdf(data, loc=theta, scale=1))

# Example usage
data = np.random.normal(2, 1, 100)
prior_samples = np.random.normal(0, 2, 1000)

evidence = compute_model_evidence(data, prior_samples, gaussian_likelihood)
print(f"Model Evidence: {evidence:.10f}")
```

Slide 5: Multi-Model Bayesian Comparison Framework

A comprehensive framework for comparing multiple statistical models using Bayesian inference requires calculating evidence ratios and posterior probabilities across all model combinations while accounting for model complexity and fit.

```python
import numpy as np
from scipy import stats

class BayesianModelComparison:
    def __init__(self, models, data):
        self.models = models
        self.data = data
        self.n_models = len(models)
        self.evidences = np.zeros(self.n_models)
        self.bayes_factors = np.zeros((self.n_models, self.n_models))
        
    def compute_evidence(self, model_idx):
        model = self.models[model_idx]
        likelihood = stats.norm.pdf(self.data, loc=model['mean'], 
                                  scale=model['std']).prod()
        prior = stats.norm.pdf(model['mean'], loc=0, scale=1)
        return likelihood * prior
    
    def compute_bayes_factors(self):
        for i in range(self.n_models):
            self.evidences[i] = self.compute_evidence(i)
            
        for i in range(self.n_models):
            for j in range(self.n_models):
                self.bayes_factors[i,j] = self.evidences[i] / self.evidences[j]
        return self.bayes_factors

# Example usage
data = np.random.normal(2, 1, 100)
models = [
    {'mean': 0, 'std': 1},
    {'mean': 2, 'std': 1},
    {'mean': -1, 'std': 2}
]

comparison = BayesianModelComparison(models, data)
bf_matrix = comparison.compute_bayes_factors()
print("Bayes Factors Matrix:\n", bf_matrix)
```

Slide 6: Marginal Likelihood Estimation

Implementation of Monte Carlo integration methods for estimating marginal likelihoods when analytical solutions are intractable, using importance sampling to improve estimation accuracy.

```python
def estimate_marginal_likelihood(data, n_samples=10000):
    # Parameter space sampling
    theta_samples = np.random.normal(0, 2, n_samples)
    
    # Likelihood calculation
    likelihoods = np.zeros(n_samples)
    for i, theta in enumerate(theta_samples):
        likelihoods[i] = np.sum(stats.norm.logpdf(data, theta, 1))
    
    # Log-sum-exp trick for numerical stability
    max_likelihood = np.max(likelihoods)
    marginal = np.log(np.mean(np.exp(likelihoods - max_likelihood))) + max_likelihood
    
    return np.exp(marginal)

# Example usage
data = np.random.normal(1.5, 1, 50)
ml_estimate = estimate_marginal_likelihood(data)
print(f"Estimated Marginal Likelihood: {ml_estimate:.6f}")
```

Slide 7: Numerical Integration for Evidence

Advanced numerical integration techniques for computing model evidence using adaptive quadrature methods, providing more accurate estimates for complex posterior distributions.

```python
def adaptive_quadrature_evidence(data, bounds, n_points=100):
    # Grid points for integration
    theta_grid = np.linspace(bounds[0], bounds[1], n_points)
    
    # Calculate posterior at each point
    def integrand(theta):
        likelihood = np.prod(stats.norm.pdf(data, theta, 1))
        prior = stats.norm.pdf(theta, 0, 2)
        return likelihood * prior
    
    # Composite Simpson's rule
    posterior_values = np.array([integrand(theta) for theta in theta_grid])
    h = (bounds[1] - bounds[0]) / (n_points - 1)
    
    evidence = h/3 * (posterior_values[0] + posterior_values[-1] + 
                      4*np.sum(posterior_values[1:-1:2]) +
                      2*np.sum(posterior_values[2:-1:2]))
    
    return evidence

# Example usage
data = np.random.normal(0.5, 1, 30)
bounds = [-5, 5]
evidence = adaptive_quadrature_evidence(data, bounds)
print(f"Model Evidence: {evidence:.8f}")
```

Slide 8: Implementation of Jeffreys' Scale

Practical implementation of Jeffreys' scale for interpreting Bayes factors, including uncertainty quantification and visualization of evidence strengths.

```python
def interpret_bayes_factor(bf, uncertainty=0.1):
    # Add random noise to simulate uncertainty
    bf_with_uncertainty = bf * (1 + np.random.normal(0, uncertainty))
    
    interpretation = {
        'strength': '',
        'support': 0,
        'uncertainty': uncertainty * bf
    }
    
    if bf_with_uncertainty >= 100:
        interpretation['strength'] = 'Decisive'
        interpretation['support'] = 4
    elif bf_with_uncertainty >= 10:
        interpretation['strength'] = 'Strong'
        interpretation['support'] = 3
    elif bf_with_uncertainty >= 3.2:
        interpretation['strength'] = 'Substantial'
        interpretation['support'] = 2
    elif bf_with_uncertainty >= 1:
        interpretation['strength'] = 'Weak'
        interpretation['support'] = 1
    else:
        interpretation['strength'] = 'Negative'
        interpretation['support'] = 0
        
    return interpretation

# Example usage
test_bfs = [1.5, 5.0, 15.0, 150.0]
for bf in test_bfs:
    result = interpret_bayes_factor(bf)
    print(f"BF = {bf:.1f}: {result['strength']} evidence "
          f"(support level: {result['support']})")
```

Slide 9: Prior Sensitivity Analysis

Implementation of sensitivity analysis to assess how different prior distributions affect the marginal likelihood and Bayes factor calculations.

```python
def sensitivity_analysis(data, prior_params_range):
    results = []
    for prior_std in prior_params_range:
        # Calculate marginal likelihood with different priors
        prior_samples = np.random.normal(0, prior_std, 1000)
        evidence = compute_model_evidence(data, prior_samples, gaussian_likelihood)
        
        # Store results
        results.append({
            'prior_std': prior_std,
            'evidence': evidence,
            'log_evidence': np.log(evidence)
        })
    
    return results

# Example usage
data = np.random.normal(1, 1, 50)
prior_stds = np.linspace(0.1, 5, 20)
sensitivity_results = sensitivity_analysis(data, prior_stds)

for result in sensitivity_results[:5]:  # Show first 5 results
    print(f"Prior std: {result['prior_std']:.2f}, "
          f"Log Evidence: {result['log_evidence']:.4f}")
```

Slide 10: Bayesian Model Averaging

Implementation of Bayesian Model Averaging (BMA) to combine predictions from multiple models weighted by their posterior probabilities.

```python
def bayesian_model_averaging(models, data, new_x):
    # Calculate model weights (posterior probabilities)
    evidences = np.array([compute_model_evidence(data, m['params'], 
                         m['likelihood']) for m in models])
    weights = evidences / np.sum(evidences)
    
    # Make predictions
    predictions = np.zeros_like(new_x)
    for i, model in enumerate(models):
        pred = model['predict'](new_x, model['params'])
        predictions += weights[i] * pred
        
    return predictions, weights

# Example prediction function
def predict_linear(x, params):
    return params[0] + params[1] * x

# Example usage
x_new = np.linspace(-5, 5, 100)
models = [
    {'params': [0, 1], 'likelihood': gaussian_likelihood, 
     'predict': predict_linear},
    {'params': [1, 2], 'likelihood': gaussian_likelihood, 
     'predict': predict_linear}
]

predictions, model_weights = bayesian_model_averaging(models, data, x_new)
```

Slide 11: Cross-Validation for Model Comparison

Implementation of cross-validated Bayes factors to provide more robust model comparison when dealing with limited data.

```python
def cross_validated_bayes_factors(data, models, k_folds=5):
    n_samples = len(data)
    fold_size = n_samples // k_folds
    cv_evidences = np.zeros((len(models), k_folds))
    
    for fold in range(k_folds):
        # Split data
        test_idx = slice(fold*fold_size, (fold+1)*fold_size)
        train_idx = list(set(range(n_samples)) - set(range(*test_idx.indices(n_samples))))
        
        train_data = data[train_idx]
        test_data = data[test_idx]
        
        # Calculate evidence for each model
        for i, model in enumerate(models):
            cv_evidences[i, fold] = compute_model_evidence(test_data, 
                                   model['prior_samples'], model['likelihood_func'])
    
    # Average across folds
    mean_evidences = np.mean(cv_evidences, axis=1)
    cv_bayes_factors = mean_evidences[:, None] / mean_evidences
    
    return cv_bayes_factors

# Example usage
cv_bf = cross_validated_bayes_factors(data, models)
print("Cross-validated Bayes Factors:\n", cv_bf)
```

Slide 12: Visualization of Model Evidence

Implementation of visualization tools for comparing model evidences and Bayes factors across different models and parameters.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_model_comparison(models, evidences, bayes_factors):
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Model Evidences
    plt.subplot(1, 2, 1)
    plt.bar(range(len(models)), evidences)
    plt.title('Model Evidences')
    plt.xlabel('Model')
    plt.ylabel('Log Evidence')
    
    # Plot 2: Bayes Factors Heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(np.log(bayes_factors), annot=True, cmap='RdYlBu')
    plt.title('Log Bayes Factors')
    plt.xlabel('Model j')
    plt.ylabel('Model i')
    
    plt.tight_layout()
    plt.show()

# Example usage
n_models = 3
evidences = np.random.uniform(1, 10, n_models)
bayes_factors = evidences[:, None] / evidences
visualize_model_comparison(range(n_models), np.log(evidences), bayes_factors)
```

Slide 13: Additional Resources

1.  [https://arxiv.org/abs/1503.08755](https://arxiv.org/abs/1503.08755) - "Computing Bayes Factors Using a Generalization of the Savage-Dickey Density Ratio"
2.  [https://arxiv.org/abs/1101.0955](https://arxiv.org/abs/1101.0955) - "Bayesian Model Selection and Model Averaging"
3.  [https://arxiv.org/abs/1911.11876](https://arxiv.org/abs/1911.11876) - "A Tutorial on Bridge Sampling"
4.  [https://arxiv.org/abs/1804.03610](https://arxiv.org/abs/1804.03610) - "Practical Bayesian Model Evaluation Using Leave-One-Out Cross-Validation"
5.  [https://arxiv.org/abs/1601.00850](https://arxiv.org/abs/1601.00850) - "Computing Bayes Factors for Evidence-Based Decision Making"

