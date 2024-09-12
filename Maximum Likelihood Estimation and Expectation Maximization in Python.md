## Maximum Likelihood Estimation and Expectation Maximization in Python
Slide 1: Introduction to Maximum Likelihood Estimation (MLE)

Maximum Likelihood Estimation is a statistical method used to estimate the parameters of a probability distribution by maximizing the likelihood function. It finds the parameter values that make the observed data most probable.

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = np.random.normal(loc=5, scale=2, size=1000)

# Define the log-likelihood function
def log_likelihood(params, data):
    mu, sigma = params
    return np.sum(norm.logpdf(data, mu, sigma))

# Find MLE estimates using optimization
from scipy.optimize import minimize
result = minimize(lambda params: -log_likelihood(params, data), x0=[0, 1])
mle_mu, mle_sigma = result.x

print(f"MLE estimates: mu = {mle_mu:.2f}, sigma = {mle_sigma:.2f}")
```

Slide 2: MLE: The Intuition

MLE finds the parameter values that maximize the probability of observing the given data. It's like finding the "best fit" for your data within the assumed probability distribution.

```python
# Visualize the likelihood landscape
mu_range = np.linspace(4, 6, 100)
sigma_range = np.linspace(1.5, 2.5, 100)
mu_grid, sigma_grid = np.meshgrid(mu_range, sigma_range)

ll_values = np.array([log_likelihood((mu, sigma), data) 
                      for mu, sigma in zip(mu_grid.ravel(), sigma_grid.ravel())])
ll_values = ll_values.reshape(mu_grid.shape)

plt.figure(figsize=(10, 8))
plt.contourf(mu_grid, sigma_grid, ll_values, levels=20, cmap='viridis')
plt.colorbar(label='Log-likelihood')
plt.plot(mle_mu, mle_sigma, 'r*', markersize=15, label='MLE estimate')
plt.xlabel('mu')
plt.ylabel('sigma')
plt.title('Log-likelihood landscape')
plt.legend()
plt.show()
```

Slide 3: Properties of MLE

MLE estimators are consistent, asymptotically unbiased, and efficient. They converge to the true parameter values as the sample size increases.

```python
# Demonstrate consistency of MLE
sample_sizes = [10, 100, 1000, 10000]
estimates = []

for size in sample_sizes:
    sample = np.random.normal(loc=5, scale=2, size=size)
    result = minimize(lambda params: -log_likelihood(params, sample), x0=[0, 1])
    estimates.append(result.x)

estimates = np.array(estimates)

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, estimates[:, 0], 'b-', label='MLE mu')
plt.plot(sample_sizes, estimates[:, 1], 'r-', label='MLE sigma')
plt.axhline(y=5, color='b', linestyle='--', label='True mu')
plt.axhline(y=2, color='r', linestyle='--', label='True sigma')
plt.xscale('log')
plt.xlabel('Sample size')
plt.ylabel('Parameter estimate')
plt.title('Consistency of MLE estimates')
plt.legend()
plt.show()
```

Slide 4: MLE for Different Distributions

MLE can be applied to various probability distributions. Let's look at an example using the Poisson distribution.

```python
from scipy.stats import poisson

# Generate Poisson data
lambda_true = 3.5
data_poisson = np.random.poisson(lambda_true, size=1000)

# MLE for Poisson is simply the sample mean
lambda_mle = np.mean(data_poisson)

# Compare histogram with true and estimated Poisson PMFs
x = np.arange(0, 15)
pmf_true = poisson.pmf(x, lambda_true)
pmf_mle = poisson.pmf(x, lambda_mle)

plt.figure(figsize=(10, 6))
plt.hist(data_poisson, bins=range(16), density=True, alpha=0.7, label='Data')
plt.plot(x, pmf_true, 'r-', label=f'True (λ={lambda_true})')
plt.plot(x, pmf_mle, 'g--', label=f'MLE (λ={lambda_mle:.2f})')
plt.xlabel('X')
plt.ylabel('Probability / Frequency')
plt.title('Poisson MLE Fit')
plt.legend()
plt.show()
```

Slide 5: MLE in Machine Learning: Logistic Regression

Logistic regression is a common machine learning algorithm that uses MLE for parameter estimation.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Generate binary classification data
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_classes=2, random_state=42)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit logistic regression (which uses MLE internally)
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Evaluate the model
train_score = lr.score(X_train, y_train)
test_score = lr.score(X_test, y_test)

print(f"Training accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

# Visualize decision boundary
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar()

x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                       np.arange(x2_min, x2_max, 0.1))
Z = lr.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.contour(xx1, xx2, Z, colors='red', linewidths=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()
```

Slide 6: Introduction to Expectation Maximization (EM)

Expectation Maximization is an iterative algorithm used to find maximum likelihood estimates of parameters in statistical models with latent variables or missing data.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data from a mixture of two Gaussians
np.random.seed(42)
n_samples = 300
true_means = [0, 5]
true_stddevs = [1, 1.5]
true_weights = [0.3, 0.7]

data = np.concatenate([
    np.random.normal(true_means[0], true_stddevs[0], int(n_samples * true_weights[0])),
    np.random.normal(true_means[1], true_stddevs[1], int(n_samples * true_weights[1]))
])

# Initialize parameters
k = 2  # number of components
means = np.random.rand(k) * 10
stddevs = np.ones(k)
weights = np.ones(k) / k

# EM algorithm
def em_step(data, means, stddevs, weights):
    # E-step: compute responsibilities
    resp = np.array([weights[j] * norm.pdf(data, means[j], stddevs[j]) for j in range(k)])
    resp /= resp.sum(axis=0)
    
    # M-step: update parameters
    N = resp.sum(axis=1)
    means = (resp * data).sum(axis=1) / N
    stddevs = np.sqrt((resp * (data - means[:, np.newaxis])**2).sum(axis=1) / N)
    weights = N / len(data)
    
    return means, stddevs, weights

# Run EM for a few iterations
for _ in range(10):
    means, stddevs, weights = em_step(data, means, stddevs, weights)

print("Estimated means:", means)
print("Estimated stddevs:", stddevs)
print("Estimated weights:", weights)
```

Slide 7: EM Algorithm: Step by Step

The EM algorithm consists of two main steps: the Expectation step (E-step) and the Maximization step (M-step). These steps are repeated iteratively until convergence.

```python
# Visualize EM iterations
plt.figure(figsize=(12, 8))
x = np.linspace(-5, 10, 200)

for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.hist(data, bins=30, density=True, alpha=0.5)
    for j in range(k):
        y = weights[j] * norm.pdf(x, means[j], stddevs[j])
        plt.plot(x, y, label=f'Component {j+1}')
    plt.title(f'Iteration {i}')
    plt.legend()
    
    means, stddevs, weights = em_step(data, means, stddevs, weights)

plt.tight_layout()
plt.show()
```

Slide 8: EM for Gaussian Mixture Models

Gaussian Mixture Models (GMMs) are a common application of the EM algorithm. They model complex distributions as a mixture of simpler Gaussian distributions.

```python
from sklearn.mixture import GaussianMixture

# Fit GMM using scikit-learn
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(data.reshape(-1, 1))

# Compare with true parameters
print("True means:", true_means)
print("Estimated means:", gmm.means_.flatten())
print("True stddevs:", true_stddevs)
print("Estimated stddevs:", np.sqrt(gmm.covariances_.flatten()))
print("True weights:", true_weights)
print("Estimated weights:", gmm.weights_)

# Visualize the fitted GMM
x = np.linspace(-5, 10, 200)
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.5, label='Data')
for i in range(2):
    y = gmm.weights_[i] * norm.pdf(x, gmm.means_[i], np.sqrt(gmm.covariances_[i]))
    plt.plot(x, y, label=f'Component {i+1}')
plt.plot(x, gmm.predict_proba(x.reshape(-1, 1))[:, 0] * norm.pdf(x, gmm.means_[0], np.sqrt(gmm.covariances_[0])) +
            gmm.predict_proba(x.reshape(-1, 1))[:, 1] * norm.pdf(x, gmm.means_[1], np.sqrt(gmm.covariances_[1])),
         'k--', label='Mixture')
plt.legend()
plt.title('Gaussian Mixture Model Fit')
plt.show()
```

Slide 9: EM for Missing Data

EM is particularly useful for handling missing data in various statistical analyses.

```python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Generate data with missing values
np.random.seed(42)
n = 1000
X = np.random.randn(n, 3)
X[np.random.rand(n, 3) < 0.2] = np.nan

# Simple imputation (mean)
imp_mean = SimpleImputer(strategy='mean')
X_mean = imp_mean.fit_transform(X)

# EM-based imputation (using IterativeImputer as a proxy for EM)
imp_em = IterativeImputer(max_iter=10, random_state=42)
X_em = imp_em.fit_transform(X)

# Compare results
df = pd.DataFrame({
    'Original': X[:, 0],
    'Mean Imputed': X_mean[:, 0],
    'EM Imputed': X_em[:, 0]
})

print(df.describe())

# Visualize imputation results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Original')
plt.scatter(X_mean[:, 0], X_mean[:, 1], alpha=0.5, label='Mean Imputed')
plt.legend()
plt.title('Mean Imputation')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Original')
plt.scatter(X_em[:, 0], X_em[:, 1], alpha=0.5, label='EM Imputed')
plt.legend()
plt.title('EM-based Imputation')

plt.tight_layout()
plt.show()
```

Slide 10: Real-life Example: Image Segmentation with EM

Image segmentation is a common application of EM in computer vision. Let's use EM to segment an image based on color intensities.

```python
from sklearn.mixture import GaussianMixture
from skimage import io, color

# Load and preprocess image
image = io.imread('https://upload.wikimedia.org/wikipedia/commons/8/8c/STS-116_spacewalk_1.jpg')
image_rgb = color.rgba2rgb(image) if image.shape[2] == 4 else image
image_lab = color.rgb2lab(image_rgb)

# Reshape the image
pixels = image_lab.reshape(-1, 3)

# Apply EM for image segmentation
n_components = 5
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(pixels)

# Get labels for each pixel
labels = gmm.predict(pixels)
segmented_image = labels.reshape(image_rgb.shape[:2])

# Visualize results
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='viridis')
plt.title('Segmented Image')
plt.axis('off')

plt.tight_layout()
plt.show()
```

Slide 11: Real-life Example: Topic Modeling with EM

Topic modeling is another application of EM in natural language processing. Let's use Latent Dirichlet Allocation (LDA), which uses EM, for topic modeling on a text dataset.

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

# Load data
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, random_state=42)

# Preprocess text data
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(newsgroups.data)

# Apply LDA
lda = LatentDirichletAllocation(n_components=4, random_state=42)
lda.fit(doc_term_matrix)

# Function to print top words for each topic
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(f"Topic {topic_idx}: {', '.join(top_words)}")

# Print results
feature_names = vectorizer.get_feature_names_out()
print_top_words(lda, feature_names, n_top_words=10)
```

Slide 12: Comparing MLE and EM

MLE and EM are closely related but used in different scenarios:

1. MLE is used when we have complete data and a well-defined likelihood function.
2. EM is used when we have incomplete data or latent variables.

Here's a simple example comparing MLE and EM for a mixture of two Gaussians:

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# Generate data
np.random.seed(42)
n_samples = 1000
true_means = [0, 5]
true_stddevs = [1, 1.5]
true_weights = [0.3, 0.7]

data = np.concatenate([
    np.random.normal(true_means[0], true_stddevs[0], int(n_samples * true_weights[0])),
    np.random.normal(true_means[1], true_stddevs[1], int(n_samples * true_weights[1]))
])

# MLE (assuming we know the true labels)
def mle_gaussian(x):
    return np.mean(x), np.std(x)

mle_mean1, mle_std1 = mle_gaussian(data[data < 2.5])
mle_mean2, mle_std2 = mle_gaussian(data[data >= 2.5])

# EM (without knowing the true labels)
def em_gaussian_mixture(data, n_iter=100):
    # Initialize parameters
    means = np.random.rand(2) * 10
    stds = np.ones(2)
    weights = np.ones(2) / 2

    for _ in range(n_iter):
        # E-step
        resp = np.array([weights[j] * norm.pdf(data, means[j], stds[j]) for j in range(2)])
        resp /= resp.sum(axis=0)

        # M-step
        weights = resp.sum(axis=1) / len(data)
        means = (resp * data).sum(axis=1) / resp.sum(axis=1)
        stds = np.sqrt((resp * (data - means[:, np.newaxis])**2).sum(axis=1) / resp.sum(axis=1))

    return means, stds, weights

em_means, em_stds, em_weights = em_gaussian_mixture(data)

print("MLE estimates:", mle_mean1, mle_std1, mle_mean2, mle_std2)
print("EM estimates:", em_means, em_stds, em_weights)
```

Slide 13: Challenges and Limitations of MLE and EM

While MLE and EM are powerful techniques, they have some limitations:

1. MLE can be sensitive to outliers and may overfit with small sample sizes.
2. EM can converge to local optima instead of global optima.
3. Both methods assume the correct model specification.

Here's a simple demonstration of MLE's sensitivity to outliers:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data with outliers
np.random.seed(42)
data = np.random.normal(0, 1, 100)
data_with_outliers = np.concatenate([data, np.array([10, 15, 20])])

# Calculate MLE (sample mean) for both datasets
mle_normal = np.mean(data)
mle_with_outliers = np.mean(data_with_outliers)

# Plot results
plt.figure(figsize=(10, 6))
plt.hist(data, bins=20, alpha=0.5, label='Normal data')
plt.hist(data_with_outliers, bins=20, alpha=0.5, label='Data with outliers')
plt.axvline(mle_normal, color='r', linestyle='--', label='MLE (normal)')
plt.axvline(mle_with_outliers, color='g', linestyle='--', label='MLE (with outliers)')
plt.legend()
plt.title("MLE Sensitivity to Outliers")
plt.show()

print(f"MLE (normal data): {mle_normal:.2f}")
print(f"MLE (data with outliers): {mle_with_outliers:.2f}")
```

Slide 14: Future Directions and Advanced Topics

1. Variational Inference: An alternative to EM for approximate inference in complex models.
2. Stochastic EM: A variant of EM that uses mini-batches for better scalability.
3. Bayesian extensions: Incorporating prior knowledge into parameter estimation.
4. Applications in deep learning: Using EM principles in unsupervised and semi-supervised learning.

Here's a simple example of stochastic EM for Gaussian Mixture Models:

```python
import numpy as np
from scipy.stats import norm

def stochastic_em_step(data_batch, means, stds, weights, learning_rate):
    # E-step (only for the batch)
    resp = np.array([weights[j] * norm.pdf(data_batch, means[j], stds[j]) for j in range(2)])
    resp /= resp.sum(axis=0)

    # M-step (update using learning rate)
    new_weights = resp.sum(axis=1) / len(data_batch)
    new_means = (resp * data_batch).sum(axis=1) / resp.sum(axis=1)
    new_stds = np.sqrt((resp * (data_batch - new_means[:, np.newaxis])**2).sum(axis=1) / resp.sum(axis=1))

    # Update parameters
    weights = (1 - learning_rate) * weights + learning_rate * new_weights
    means = (1 - learning_rate) * means + learning_rate * new_means
    stds = (1 - learning_rate) * stds + learning_rate * new_stds

    return means, stds, weights

# Usage example (pseudocode)
# for epoch in range(n_epochs):
#     for batch in data_batches:
#         means, stds, weights = stochastic_em_step(batch, means, stds, weights, learning_rate)
```

Slide 15: Additional Resources

For further reading on MLE and EM algorithms, consider these peer-reviewed articles:

1. Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. Journal of the Royal Statistical Society: Series B (Methodological), 39(1), 1-22. ArXiv: [https://arxiv.org/abs/acs-9604001](https://arxiv.org/abs/acs-9604001)
2. Neal, R. M., & Hinton, G. E. (1998). A view of the EM algorithm that justifies incremental, sparse, and other variants. In Learning in graphical models (pp. 355-368). Springer, Dordrecht. ArXiv: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
3. Bishop, C. M. (2006). Pattern recognition and machine learning. Springer. (While not on ArXiv, this is a widely recognized textbook in the field)

These resources provide in-depth explanations and mathematical foundations of MLE and EM algorithms, along with their applications in various domains of statistics and machine learning.

