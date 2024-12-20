## Metropolis-Hastings Algorithm in Python
Slide 1: Introduction to Metropolis-Hastings Algorithm

The Metropolis-Hastings algorithm is a Markov Chain Monte Carlo (MCMC) method used for sampling from complex probability distributions. It's particularly useful when direct sampling is difficult or impossible. This algorithm is widely applied in Bayesian inference, statistical physics, and machine learning.

```python
import numpy as np
import matplotlib.pyplot as plt

def target_distribution(x):
    return np.exp(-0.5 * x**2)  # Example: standard normal distribution

x = np.linspace(-5, 5, 1000)
plt.plot(x, target_distribution(x))
plt.title("Target Distribution")
plt.show()
```

Slide 2: The Concept of MCMC

MCMC methods generate samples from a target distribution by constructing a Markov chain. The Metropolis-Hastings algorithm is one such method that allows sampling from distributions where only the unnormalized probability densities are known.

```python
def metropolis_hastings(target, proposal, initial_state, num_samples):
    current_state = initial_state
    samples = [current_state]
    
    for _ in range(num_samples - 1):
        proposed_state = proposal(current_state)
        acceptance_ratio = min(1, target(proposed_state) / target(current_state))
        
        if np.random.random() < acceptance_ratio:
            current_state = proposed_state
        
        samples.append(current_state)
    
    return samples

# Example usage will be shown in the next slides
```

Slide 3: The Proposal Distribution

The proposal distribution is used to generate candidate samples. It should be easy to sample from and symmetric (q(x|y) = q(y|x)). A common choice is a Gaussian distribution centered at the current state.

```python
def gaussian_proposal(current_state, proposal_width=0.5):
    return np.random.normal(current_state, proposal_width)

# Visualize the proposal distribution
current_state = 0
x = np.linspace(-3, 3, 1000)
proposal_pdf = [np.exp(-0.5 * ((xi - current_state) / 0.5)**2) for xi in x]

plt.plot(x, proposal_pdf)
plt.title("Gaussian Proposal Distribution")
plt.axvline(current_state, color='r', linestyle='--', label='Current State')
plt.legend()
plt.show()
```

Slide 4: Acceptance-Rejection Step

The key to the Metropolis-Hastings algorithm is the acceptance-rejection step. This step ensures that the samples converge to the target distribution.

```python
def acceptance_probability(current_state, proposed_state, target):
    return min(1, target(proposed_state) / target(current_state))

# Example
current_state = 0
proposed_state = 1
target = lambda x: np.exp(-0.5 * x**2)  # Standard normal distribution

prob = acceptance_probability(current_state, proposed_state, target)
print(f"Acceptance probability: {prob:.4f}")
```

Slide 5: Implementing Metropolis-Hastings

Let's implement the Metropolis-Hastings algorithm to sample from a standard normal distribution.

```python
import numpy as np
import matplotlib.pyplot as plt

def target(x):
    return np.exp(-0.5 * x**2)

def proposal(x):
    return np.random.normal(x, 0.5)

def metropolis_hastings(num_samples):
    samples = np.zeros(num_samples)
    current = 0  # Start at x=0
    
    for i in range(num_samples):
        proposed = proposal(current)
        if np.random.random() < min(1, target(proposed) / target(current)):
            current = proposed
        samples[i] = current
    
    return samples

samples = metropolis_hastings(10000)
plt.hist(samples, bins=50, density=True, alpha=0.7)
plt.title("Samples from Metropolis-Hastings")
plt.show()
```

Slide 6: Convergence and Burn-in

The initial samples may not accurately represent the target distribution. We often discard these initial samples, a process called "burn-in".

```python
def plot_trace(samples):
    plt.figure(figsize=(12, 4))
    plt.plot(samples)
    plt.title("Trace Plot")
    plt.xlabel("Iteration")
    plt.ylabel("Sample Value")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()

samples = metropolis_hastings(5000)
plot_trace(samples)

# Discard first 1000 samples as burn-in
plt.hist(samples[1000:], bins=50, density=True, alpha=0.7)
plt.title("Samples after Burn-in")
plt.show()
```

Slide 7: Autocorrelation

Samples from MCMC methods are often autocorrelated. We can assess this using an autocorrelation plot.

```python
from statsmodels.graphics.tsaplots import plot_acf

def plot_autocorrelation(samples):
    plot_acf(samples, lags=100)
    plt.title("Autocorrelation Plot")
    plt.show()

samples = metropolis_hastings(10000)
plot_autocorrelation(samples)
```

Slide 8: Tuning the Proposal Distribution

The efficiency of the Metropolis-Hastings algorithm depends on the choice of the proposal distribution. Let's compare different proposal widths.

```python
def metropolis_hastings_with_width(num_samples, proposal_width):
    samples = np.zeros(num_samples)
    current = 0
    
    for i in range(num_samples):
        proposed = np.random.normal(current, proposal_width)
        if np.random.random() < min(1, target(proposed) / target(current)):
            current = proposed
        samples[i] = current
    
    return samples

widths = [0.1, 1.0, 5.0]
plt.figure(figsize=(15, 5))

for i, width in enumerate(widths):
    samples = metropolis_hastings_with_width(10000, width)
    plt.subplot(1, 3, i+1)
    plt.hist(samples, bins=50, density=True, alpha=0.7)
    plt.title(f"Proposal Width: {width}")

plt.tight_layout()
plt.show()
```

Slide 9: Acceptance Rate

The acceptance rate is the proportion of proposed moves that are accepted. A good rule of thumb is to aim for an acceptance rate between 20% and 50%.

```python
def metropolis_hastings_with_acceptance(num_samples, proposal_width):
    samples = np.zeros(num_samples)
    current = 0
    accepted = 0
    
    for i in range(num_samples):
        proposed = np.random.normal(current, proposal_width)
        if np.random.random() < min(1, target(proposed) / target(current)):
            current = proposed
            accepted += 1
        samples[i] = current
    
    return samples, accepted / num_samples

widths = [0.1, 1.0, 5.0]
for width in widths:
    samples, rate = metropolis_hastings_with_acceptance(10000, width)
    print(f"Proposal width: {width}, Acceptance rate: {rate:.2f}")
```

Slide 10: Multivariate Metropolis-Hastings

The Metropolis-Hastings algorithm can be extended to sample from multivariate distributions.

```python
import numpy as np
import matplotlib.pyplot as plt

def multivariate_target(x):
    return np.exp(-0.5 * (x[0]**2 + x[1]**2 - 2*0.5*x[0]*x[1]) / (1 - 0.5**2))

def multivariate_proposal(x, cov):
    return np.random.multivariate_normal(x, cov)

def multivariate_metropolis_hastings(num_samples, cov):
    samples = np.zeros((num_samples, 2))
    current = np.zeros(2)
    
    for i in range(num_samples):
        proposed = multivariate_proposal(current, cov)
        if np.random.random() < min(1, multivariate_target(proposed) / multivariate_target(current)):
            current = proposed
        samples[i] = current
    
    return samples

cov = np.array([[0.1, 0], [0, 0.1]])
samples = multivariate_metropolis_hastings(10000, cov)

plt.figure(figsize=(10, 10))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1)
plt.title("Samples from 2D Gaussian")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

Slide 11: Real-life Example: Estimating Parameters of a Mixture Model

Let's use Metropolis-Hastings to estimate the parameters of a mixture of two Gaussian distributions.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data from a mixture of two Gaussians
np.random.seed(42)
n_samples = 1000
true_means = [-2, 2]
true_stds = [1, 0.5]
true_weights = [0.3, 0.7]

data = np.concatenate([
    np.random.normal(true_means[0], true_stds[0], int(n_samples * true_weights[0])),
    np.random.normal(true_means[1], true_stds[1], int(n_samples * true_weights[1]))
])

# Define the log-likelihood function
def log_likelihood(params, data):
    mean1, std1, mean2, std2, weight = params
    
    # Ensure parameters are within valid ranges
    if std1 <= 0 or std2 <= 0 or weight <= 0 or weight >= 1:
        return -np.inf
    
    mix1 = weight * np.exp(-0.5 * ((data - mean1) / std1)**2) / (std1 * np.sqrt(2 * np.pi))
    mix2 = (1 - weight) * np.exp(-0.5 * ((data - mean2) / std2)**2) / (std2 * np.sqrt(2 * np.pi))
    
    return np.sum(np.log(mix1 + mix2))

# Metropolis-Hastings algorithm
def metropolis_hastings(data, num_iterations):
    current_params = np.array([0, 1, 0, 1, 0.5])  # Initial guess
    current_ll = log_likelihood(current_params, data)
    samples = [current_params]
    
    for _ in range(num_iterations):
        proposed_params = current_params + np.random.normal(0, 0.1, 5)
        proposed_ll = log_likelihood(proposed_params, data)
        
        if np.random.random() < np.exp(proposed_ll - current_ll):
            current_params = proposed_params
            current_ll = proposed_ll
        
        samples.append(current_params)
    
    return np.array(samples)

# Run the algorithm
samples = metropolis_hastings(data, 10000)

# Plot results
plt.figure(figsize=(15, 5))
param_names = ['Mean 1', 'Std 1', 'Mean 2', 'Std 2', 'Weight']
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.hist(samples[5000:, i], bins=30)
    plt.title(param_names[i])
    plt.axvline(samples[5000:, i].mean(), color='r', linestyle='--')
plt.tight_layout()
plt.show()

print("Estimated parameters:")
print(samples[5000:].mean(axis=0))
print("True parameters:")
print(true_means + true_stds + [true_weights[0]])
```

Slide 12: Real-life Example: Image Denoising

Let's use Metropolis-Hastings for image denoising, assuming a simple prior distribution on pixel intensities.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 10x10 image
true_image = np.zeros((10, 10))
true_image[2:8, 2:8] = 1

# Add noise
noisy_image = true_image + np.random.normal(0, 0.3, true_image.shape)

# Define the log-posterior (combining likelihood and prior)
def log_posterior(image, noisy_image, beta):
    likelihood = -0.5 * np.sum((image - noisy_image)**2)
    prior = -beta * np.sum(np.abs(np.diff(image, axis=0)) + np.abs(np.diff(image, axis=1)))
    return likelihood + prior

# Metropolis-Hastings for image denoising
def denoise_image(noisy_image, num_iterations, beta):
    current_image = np.(noisy_image)
    current_log_post = log_posterior(current_image, noisy_image, beta)
    
    for _ in range(num_iterations):
        i, j = np.random.randint(0, noisy_image.shape[0]), np.random.randint(0, noisy_image.shape[1])
        proposed_image = np.(current_image)
        proposed_image[i, j] += np.random.normal(0, 0.1)
        
        proposed_log_post = log_posterior(proposed_image, noisy_image, beta)
        
        if np.random.random() < np.exp(proposed_log_post - current_log_post):
            current_image = proposed_image
            current_log_post = proposed_log_post
    
    return current_image

# Run the denoising algorithm
denoised_image = denoise_image(noisy_image, 10000, beta=1.0)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(true_image, cmap='gray')
axes[0].set_title("True Image")
axes[1].imshow(noisy_image, cmap='gray')
axes[1].set_title("Noisy Image")
axes[2].imshow(denoised_image, cmap='gray')
axes[2].set_title("Denoised Image")
plt.tight_layout()
plt.show()
```

Slide 13: Advantages and Limitations

Advantages of Metropolis-Hastings:

* Can sample from complex, high-dimensional distributions
* Only requires knowing the target distribution up to a constant
* Relatively easy to implement

Limitations:

* Samples are correlated, reducing effective sample size
* May have slow convergence for some distributions
* Requires careful tuning of the proposal distribution

```python
# Visualization of correlated samples vs independent samples
np.random.seed(42)
correlated_samples = metropolis_hastings(5000)
independent_samples = np.random.normal(0, 1, 5000)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(correlated_samples[:100])
axes[0].set_title("Correlated Samples (Metropolis-Hastings)")
axes[1].plot(independent_samples[:100])
axes[1].set_title("Independent Samples")
plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

For further exploration of the Metropolis-Hastings algorithm and MCMC methods, consider the following resources:

1. "Markov Chain Monte Carlo in Practice: A Roundtable Discussion" by Kass et al. (1998) ArXiv: [https://arxiv.org/abs/math/9705234](https://arxiv.org/abs/math/9705234)
2. "An Introduction to MCMC for Machine Learning" by Andrieu et al. (2003) ArXiv: [https://arxiv.org/abs/1109.4435](https://arxiv.org/abs/1109.4435)
3. "Handbook of Markov Chain Monte Carlo" edited by Brooks et al. (2011) Available at: [https://www.mcmchandbook.net/](https://www.mcmchandbook.net/)
4. "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo" by Hoffman and Gelman (2014) ArXiv: [https://arxiv.org/abs/1111.4246](https://arxiv.org/abs/1111.4246)

These resources provide a deeper understanding of MCMC methods, including the Metropolis-Hastings algorithm, and their applications in various fields such as statistics, machine learning, and physics.

