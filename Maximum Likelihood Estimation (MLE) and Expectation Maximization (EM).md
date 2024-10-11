## Maximum Likelihood Estimation (MLE) and Expectation Maximization (EM)
Slide 1: Introduction to Maximum Likelihood Estimation (MLE) and Expectation Maximization (EM)

Maximum Likelihood Estimation (MLE) and Expectation Maximization (EM) are powerful statistical techniques used to estimate parameters in probabilistic models. MLE finds the parameters that maximize the likelihood of observing the given data, while EM is an iterative algorithm used when dealing with incomplete or hidden data.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate data from a mixture of two Gaussian distributions
np.random.seed(42)
n_samples = 1000
weights = [0.3, 0.7]
means = [0, 5]
stds = [1, 1.5]

data = np.concatenate([
    np.random.normal(loc=means[0], scale=stds[0], size=int(weights[0] * n_samples)),
    np.random.normal(loc=means[1], scale=stds[1], size=int(weights[1] * n_samples))
])

plt.hist(data, bins=50, density=True)
plt.title("Mixture of Two Gaussian Distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()
```

Slide 2: Maximum Likelihood Estimation (MLE) - Concept

MLE aims to find the parameter values that maximize the likelihood function, which represents the probability of observing the data given the parameters. In other words, it seeks the parameters that make the observed data most probable.

```python
import scipy.stats as stats

# Define the log-likelihood function for a single Gaussian distribution
def log_likelihood(params, data):
    mu, sigma = params
    return np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))

# Generate some sample data
np.random.seed(42)
true_mu, true_sigma = 5, 2
sample_data = np.random.normal(true_mu, true_sigma, 1000)

# Perform MLE using scipy's optimize function
from scipy.optimize import minimize

initial_guess = [0, 1]
result = minimize(lambda params: -log_likelihood(params, sample_data), initial_guess)

estimated_mu, estimated_sigma = result.x
print(f"True parameters: mu={true_mu}, sigma={true_sigma}")
print(f"Estimated parameters: mu={estimated_mu:.2f}, sigma={estimated_sigma:.2f}")
```

Slide 3: MLE - Example with Bernoulli Distribution

Let's apply MLE to estimate the probability of success in a Bernoulli distribution, which models binary outcomes like coin flips.

```python
# Simulate coin flips (1 for heads, 0 for tails)
np.random.seed(42)
n_flips = 1000
true_p = 0.7
coin_flips = np.random.binomial(1, true_p, n_flips)

# MLE for Bernoulli distribution
estimated_p = np.mean(coin_flips)

print(f"True probability of heads: {true_p}")
print(f"Estimated probability of heads: {estimated_p:.4f}")

# Visualize the likelihood function
p_values = np.linspace(0, 1, 1000)
likelihood = np.prod(p_values**coin_flips * (1-p_values)**(1-coin_flips))

plt.plot(p_values, likelihood)
plt.axvline(estimated_p, color='r', linestyle='--', label='MLE estimate')
plt.xlabel('p (probability of heads)')
plt.ylabel('Likelihood')
plt.title('Likelihood Function for Bernoulli Distribution')
plt.legend()
plt.show()
```

Slide 4: Expectation Maximization (EM) - Concept

EM is an iterative algorithm used when dealing with incomplete or hidden data. It alternates between two steps: the Expectation (E) step, which estimates the missing data, and the Maximization (M) step, which maximizes the likelihood using the estimated data.

```python
import numpy as np
from scipy.stats import norm

def em_gaussian_mixture(data, k, max_iter=100, tol=1e-6):
    # Initialize parameters
    n = len(data)
    weights = np.ones(k) / k
    means = np.random.choice(data, k)
    variances = np.random.rand(k)
    
    for _ in range(max_iter):
        # E-step: Compute responsibilities
        responsibilities = np.zeros((n, k))
        for j in range(k):
            responsibilities[:, j] = weights[j] * norm.pdf(data, means[j], np.sqrt(variances[j]))
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        
        # M-step: Update parameters
        Nk = responsibilities.sum(axis=0)
        weights_new = Nk / n
        means_new = np.dot(responsibilities.T, data) / Nk[:, np.newaxis]
        variances_new = np.dot(responsibilities.T, (data - means_new[:, np.newaxis])**2) / Nk[:, np.newaxis]
        
        # Check for convergence
        if np.all(np.abs(means_new - means) < tol):
            break
        
        weights, means, variances = weights_new, means_new, variances_new
    
    return weights, means, variances

# Example usage (continued in next slide)
```

Slide 5: EM - Gaussian Mixture Model Example

Let's apply the EM algorithm to fit a Gaussian Mixture Model (GMM) to our simulated data from Slide 1.

```python
# Use the data generated in Slide 1
k = 2  # Number of components
weights_est, means_est, variances_est = em_gaussian_mixture(data, k)

print("Estimated parameters:")
for i in range(k):
    print(f"Component {i+1}:")
    print(f"  Weight: {weights_est[i]:.2f}")
    print(f"  Mean: {means_est[i]:.2f}")
    print(f"  Variance: {variances_est[i]:.2f}")

# Plot the results
x = np.linspace(min(data), max(data), 1000)
plt.hist(data, bins=50, density=True, alpha=0.5)
for i in range(k):
    plt.plot(x, weights_est[i] * norm.pdf(x, means_est[i], np.sqrt(variances_est[i])),
             label=f'Component {i+1}')
plt.plot(x, np.sum([weights_est[i] * norm.pdf(x, means_est[i], np.sqrt(variances_est[i])) for i in range(k)], axis=0),
         label='Mixture', linewidth=2)
plt.title("Gaussian Mixture Model Fitted with EM")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()
```

Slide 6: MLE vs. EM - When to Use Which?

MLE is suitable for complete data with known distributions, while EM is used for incomplete data or mixture models. MLE finds a single set of parameters, whereas EM iteratively estimates both parameters and hidden variables.

```python
import pandas as pd

comparison = pd.DataFrame({
    'Characteristic': ['Data Completeness', 'Model Complexity', 'Iteration', 'Convergence', 'Computational Cost'],
    'MLE': ['Complete data', 'Simple models', 'Non-iterative', 'Guaranteed global optimum', 'Lower'],
    'EM': ['Incomplete data', 'Complex models (e.g., mixtures)', 'Iterative', 'Local optimum', 'Higher']
})

print(comparison.to_string(index=False))
```

Slide 7: MLE - Advantages and Challenges

Advantages of MLE include its consistency, efficiency, and invariance under parameterization. However, it can be sensitive to outliers and may overfit with small sample sizes.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data with an outlier
np.random.seed(42)
data = np.random.normal(0, 1, 100)
data = np.append(data, 10)  # Add an outlier

# MLE estimation
mle_mean = np.mean(data)
mle_std = np.std(data)

# Robust estimation (median and median absolute deviation)
robust_mean = np.median(data)
robust_std = np.median(np.abs(data - robust_mean)) * 1.4826  # Scale factor for normal distribution

# Plot results
x = np.linspace(-5, 15, 1000)
plt.hist(data, bins=30, density=True, alpha=0.5, label='Data')
plt.plot(x, norm.pdf(x, mle_mean, mle_std), label='MLE fit')
plt.plot(x, norm.pdf(x, robust_mean, robust_std), label='Robust fit')
plt.title("MLE vs Robust Estimation with Outlier")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

print(f"MLE estimates: mean={mle_mean:.2f}, std={mle_std:.2f}")
print(f"Robust estimates: mean={robust_mean:.2f}, std={robust_std:.2f}")
```

Slide 8: EM - Advantages and Challenges

EM is versatile for handling missing data and fitting mixture models. It guarantees improvement in each iteration but may converge to local optima and can be slow for large datasets.

```python
import numpy as np
import matplotlib.pyplot as plt

def em_convergence(data, k, max_iter=100):
    n = len(data)
    log_likelihoods = []
    
    # Initialize parameters (simplified for demonstration)
    weights = np.ones(k) / k
    means = np.random.choice(data, k)
    variances = np.ones(k)
    
    for _ in range(max_iter):
        # E-step (simplified)
        responsibilities = np.zeros((n, k))
        for j in range(k):
            responsibilities[:, j] = weights[j] * norm.pdf(data, means[j], np.sqrt(variances[j]))
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        
        # M-step (simplified)
        weights = responsibilities.mean(axis=0)
        means = np.dot(responsibilities.T, data) / responsibilities.sum(axis=0)
        
        # Compute log-likelihood
        log_likelihood = np.sum(np.log(np.sum(weights[j] * norm.pdf(data, means[j], np.sqrt(variances[j])) for j in range(k))))
        log_likelihoods.append(log_likelihood)
    
    return log_likelihoods

# Run EM and plot convergence
log_likelihoods = em_convergence(data, k=2)

plt.plot(log_likelihoods)
plt.title("EM Algorithm Convergence")
plt.xlabel("Iteration")
plt.ylabel("Log-likelihood")
plt.show()
```

Slide 9: Real-life Example: Image Segmentation with EM

EM can be used for image segmentation, where pixels are clustered based on their color intensities. This technique is useful in medical imaging, object recognition, and computer vision.

```python
from sklearn.mixture import GaussianMixture
from skimage import io, color
import matplotlib.pyplot as plt

# Load and preprocess the image
image = io.imread('https://upload.wikimedia.org/wikipedia/commons/8/8c/STS-116_spacewalk_1.jpg')
image_gray = color.rgb2gray(image)
image_shaped = image_gray.reshape(-1, 1)

# Apply EM for image segmentation
n_components = 3
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(image_shaped)
labels = gmm.predict(image_shaped)

# Reshape labels and display segmented image
segmented_image = labels.reshape(image_gray.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image_gray, cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')
ax2.imshow(segmented_image, cmap='viridis')
ax2.set_title('Segmented Image')
ax2.axis('off')
plt.tight_layout()
plt.show()
```

Slide 10: Real-life Example: Natural Language Processing with MLE

MLE is commonly used in Natural Language Processing (NLP) for tasks such as language modeling and part-of-speech tagging. Let's implement a simple bigram language model using MLE.

```python
import numpy as np
from collections import defaultdict

# Sample text
text = "the cat sat on the mat the dog chased the cat the cat ran away"
words = text.split()

# Create bigram counts
bigram_counts = defaultdict(lambda: defaultdict(int))
for i in range(len(words) - 1):
    bigram_counts[words[i]][words[i+1]] += 1

# Compute MLE probabilities
bigram_probs = defaultdict(lambda: defaultdict(float))
for word1, next_words in bigram_counts.items():
    total = sum(next_words.values())
    for word2, count in next_words.items():
        bigram_probs[word1][word2] = count / total

# Function to generate text
def generate_text(start_word, length=10):
    current_word = start_word
    generated_text = [current_word]
    for _ in range(length - 1):
        next_word_probs = bigram_probs[current_word]
        if not next_word_probs:
            break
        next_word = max(next_word_probs, key=next_word_probs.get)
        generated_text.append(next_word)
        current_word = next_word
    return ' '.join(generated_text)

# Generate text
print(generate_text("the", 15))

# Print some bigram probabilities
print("\nSome bigram probabilities:")
for word1 in ["the", "cat", "dog"]:
    print(f"\nAfter '{word1}':")
    for word2, prob in sorted(bigram_probs[word1].items(), key=lambda x: x[1], reverse=True):
        print(f"  '{word2}': {prob:.2f}")
```

Slide 11: Regularization in MLE and EM

Regularization helps prevent overfitting in both MLE and EM. For MLE, we can add a penalty term to the likelihood function. In EM, regularization can be applied during the M-step.

```python
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Generate some data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 2, 100)

# Define the negative log-likelihood function with L2 regularization
def neg_log_likelihood(params, X, y, lambda_):
    m, b = params
    y_pred = m * X + b
    mse = np.mean((y - y_pred)**2)
    reg_term = lambda_ * (m**2 + b**2)
    return mse + reg_term

# Perform MLE with different regularization strengths
lambda_values = [0, 0.1, 1, 10]
results = []

for lambda_ in lambda_values:
    result = minimize(neg_log_likelihood, [0, 0], args=(X, y, lambda_))
    results.append(result.x)

# Plot the results
plt.figure(figsize=(12, 8))
plt.scatter(X, y, alpha=0.5, label='Data')

for i, (m, b) in enumerate(results):
    y_pred = m * X + b
    plt.plot(X, y_pred, label=f'λ = {lambda_values[i]}')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Different Regularization Strengths')
plt.legend()
plt.show()

# Print the results
for i, (m, b) in enumerate(results):
    print(f"λ = {lambda_values[i]}: m = {m:.4f}, b = {b:.4f}")
```

Slide 12: Handling Missing Data with EM

EM is particularly useful for dealing with missing data. Let's demonstrate how EM can be used to estimate parameters of a Gaussian distribution with missing values.

```python
import numpy as np
from scipy.stats import norm

def em_gaussian_missing(data, max_iter=100, tol=1e-6):
    # Initialize parameters
    mu = np.nanmean(data)
    sigma = np.nanstd(data)
    
    for _ in range(max_iter):
        # E-step: Estimate missing values
        data_filled = np.where(np.isnan(data), mu, data)
        
        # M-step: Update parameters
        mu_new = np.mean(data_filled)
        sigma_new = np.sqrt(np.mean((data_filled - mu_new)**2))
        
        # Check for convergence
        if abs(mu - mu_new) < tol and abs(sigma - sigma_new) < tol:
            break
        
        mu, sigma = mu_new, sigma_new
    
    return mu, sigma

# Generate data with missing values
np.random.seed(42)
true_mu, true_sigma = 5, 2
data = np.random.normal(true_mu, true_sigma, 1000)
missing_mask = np.random.random(1000) < 0.2
data[missing_mask] = np.nan

# Apply EM algorithm
estimated_mu, estimated_sigma = em_gaussian_missing(data)

print(f"True parameters: mu={true_mu}, sigma={true_sigma}")
print(f"Estimated parameters: mu={estimated_mu:.2f}, sigma={estimated_sigma:.2f}")

# Plot results
x = np.linspace(0, 10, 100)
plt.hist(data[~np.isnan(data)], bins=30, density=True, alpha=0.5, label='Observed Data')
plt.plot(x, norm.pdf(x, estimated_mu, estimated_sigma), label='Estimated Distribution')
plt.plot(x, norm.pdf(x, true_mu, true_sigma), label='True Distribution')
plt.title("EM for Gaussian with Missing Data")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()
```

Slide 13: MLE and EM in Machine Learning Frameworks

Many machine learning frameworks provide implementations of MLE and EM algorithms. Let's look at an example using scikit-learn for Gaussian Mixture Models.

```python
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# Generate data from a mixture of two Gaussians
np.random.seed(42)
n_samples = 1000
weights = [0.3, 0.7]
means = [[0, 0], [5, 5]]
covs = [[[1, 0], [0, 1]], [[1.5, 0], [0, 1.5]]]

X = np.vstack([
    np.random.multivariate_normal(means[0], covs[0], int(weights[0] * n_samples)),
    np.random.multivariate_normal(means[1], covs[1], int(weights[1] * n_samples))
])

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X)

# Plot results
x, y = np.meshgrid(np.linspace(-3, 8, 200), np.linspace(-3, 8, 200))
XX = np.array([x.ravel(), y.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(x.shape)

plt.figure(figsize=(10, 8))
plt.contourf(x, y, Z, levels=20, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c='white', s=10, alpha=0.5)
plt.title("Gaussian Mixture Model Fitted with EM")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label='Negative log-likelihood')
plt.show()

print("Estimated means:")
print(gmm.means_)
print("\nEstimated covariances:")
print(gmm.covariances_)
print("\nEstimated weights:")
print(gmm.weights_)
```

Slide 14: Additional Resources

For those interested in diving deeper into Maximum Likelihood Estimation and Expectation Maximization, here are some valuable resources:

1. "The EM Algorithm and Extensions" by McLachlan and Krishnan (2007) ArXiv: [https://arxiv.org/abs/0712.2886](https://arxiv.org/abs/0712.2886)
2. "A Gentle Tutorial of the EM Algorithm and its Application to Parameter Estimation for Gaussian Mixture and Hidden Markov Models" by Bilmes (1998) ArXiv: [https://arxiv.org/abs/1105.1476](https://arxiv.org/abs/1105.1476)
3. "Maximum Likelihood Estimation and Inference" by Pawitan (2001) Book ISBN: 978-0470094822
4. "Pattern Recognition and Machine Learning" by Bishop (2006) Book ISBN: 978-0387310732

These resources provide in-depth explanations, mathematical derivations, and advanced applications of MLE and EM algorithms in various fields of statistics and machine learning.

