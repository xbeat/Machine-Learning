## Bayesian Statistics with Python

Slide 1: 
Introduction to Bayesian Statistics
Bayesian statistics is a powerful approach that allows us to incorporate prior knowledge and update our beliefs as new data becomes available. It provides a way to quantify uncertainty and make informed decisions based on probabilities.

Slide 2: 
Bayes' Theorem Bayes' Theorem is the fundamental principle behind Bayesian statistics. It relates the conditional probabilities of events and allows us to update our beliefs based on new evidence. Code Example:

```python
from scipy.stats import beta, norm

# Prior distribution
prior = beta(alpha=2, beta=5)

# Likelihood function
likelihood = norm.pdf(data, loc=mu, scale=sigma)

# Marginal likelihood
marginal_likelihood = prior.pdf(mu) * likelihood

# Posterior distribution
posterior = (prior.pdf(mu) * likelihood) / marginal_likelihood
```

Slide 3: 
Prior Distribution
The prior distribution represents our initial belief or knowledge about a parameter before observing any data. It can be based on previous studies, expert opinions, or subjective assessments.

Slide 4: 
Likelihood Function
The likelihood function quantifies the probability of observing the data given a particular parameter value. It provides information about how well the parameter explains the observed data. Code Example:

```python
from scipy.stats import norm
likelihood = norm.pdf(data, loc=mu, scale=sigma)
```

Slide 5: 
Posterior Distribution
The posterior distribution is the updated belief about the parameter after considering the observed data. It combines the prior distribution and the likelihood function using Bayes' Theorem. Code Example:

```python
import pymc3 as pm

with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=data)

    trace = pm.sample(1000, chains=2)
    posterior_mu = trace['mu']
    posterior_sigma = trace['sigma']
```

Slide 6: 
Markov Chain Monte Carlo (MCMC)
MCMC methods are computational techniques used to approximate the posterior distribution when it cannot be calculated analytically. They generate samples from the posterior distribution, allowing for parameter estimation and uncertainty quantification.

Slide 7: 
Pymc3 Library PyMC3 is a popular Python library for Bayesian statistical modeling and Probabilistic Machine Learning. It provides a user-friendly interface for building and analyzing Bayesian models using MCMC sampling. Code Example:

```python
import pymc3 as pm
import numpy as np

data = np.random.normal(0, 1, size=100)

with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=data)

    trace = pm.sample(1000, chains=2)
    pm.plot_posterior(trace, var_names=['mu', 'sigma'])
```

Slide 8: 
Bayesian Linear Regression
Bayesian linear regression is a powerful technique for modeling linear relationships between variables while accounting for uncertainty in the parameters and making probabilistic predictions. Code Example:

```python
import pymc3 as pm
import numpy as np

X = np.random.randn(100, 2)
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 1, size=100)

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal('sigma', sigma=1)
    mu = alpha + pm.math.dot(X, beta)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(1000, chains=2)
    pm.plot_posterior(trace, var_names=['alpha', 'beta', 'sigma'])
```

Slide 9: 
Bayesian Classification
Bayesian classification techniques, such as Naive Bayes classifiers, are widely used in machine learning for tasks like spam filtering, sentiment analysis, and document classification. Code Example:

```python
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
import numpy as np

X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=42)

clf = GaussianNB()
clf.fit(X, y)

print(f"Class priors: {clf.class_prior_}")
print(f"Class means: {clf.theta_}")
print(f"Class variances: {np.sqrt(clf.sigma_)}")
```

Slide 10: 
Bayesian Time Series Analysis
Bayesian methods can be applied to time series analysis, allowing for accurate forecasting and uncertainty quantification while accounting for temporal dependencies and non-stationarities. Code Example:

```python
import pymc3 as pm
model = pm.Model()
alpha = pm.Normal('alpha', mu=0, sigma=1)
sigma = pm.HalfNormal('sigma', sigma=1)
```

Slide 11: 
Bayesian A/B Testing
Bayesian A/B testing is a powerful technique for evaluating the effectiveness of different variants or treatments, while accounting for uncertainty and making probabilistic decisions. Code Example:

```python
import pymc3 as pm
import numpy as np

control_data = np.random.binomial(1, 0.2, size=1000)
variant_data = np.random.binomial(1, 0.25, size=1000)

with pm.Model() as model:
    p_control = pm.Beta('p_control', alpha=1, beta=1)
    p_variant = pm.Beta('p_variant', alpha=1, beta=1)

    control_obs = pm.Binomial('control_obs', p=p_control, observed=control_data.sum())
    variant_obs = pm.Binomial('variant_obs', p=p_variant, observed=variant_data.sum())

    trace = pm.sample(1000, chains=2)
    pm.plot_posterior(trace, var_names=['p_control', 'p_variant'])
```

Slide 12: 
Bayesian Optimization
Bayesian optimization is a powerful technique for global optimization of black-box functions, particularly useful in situations where evaluating the objective function is computationally expensive or time-consuming. Code Example:

```python
from bayes_opt import BayesianOptimization
import numpy as np

def black_box_function(x, y):
    return -np.sin(x) * np.cos(y) + 0.5 * x**2 + 0.5 * y**2

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds={'x': (-2, 2), 'y': (-2, 2)},
    random_state=42,
)

optimizer.maximize(init_points=5, n_iter=25)
print(f"Maximum value found: {optimizer.max['max_val']}")
print(f"Maximum parameters: {optimizer.max['params']}")
```

Slide 13: 
Bayesian Networks
Bayesian networks are graphical models that represent probabilistic relationships among variables, allowing for reasoning under uncertainty and making predictions based on observed evidence. Code Example:

```python
import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

data = pd.DataFrame(np.random.randint(0, 2, size=(1000, 3)), columns=['A', 'B', 'C'])
data['D'] = (data['A'] & data['B']) | (data['B'] & data['C'])

model = BayesianModel([('A', 'D'), ('B', 'D'), ('C', 'D')])
model.fit(data, estimator=MaximumLikelihoodEstimator)

print(model.get_cpds('D'))
```

Slide 14: 
Bayesian Hierarchical Models
Bayesian hierarchical models are useful for modeling data with grouped or nested structures, allowing for the sharing of information across groups and accounting for group-level and individual-level variability. Code Example:

```python
import pymc3 as pm
import numpy as np
import pandas as pd

data = pd.DataFrame({
    'group': np.repeat([1, 2, 3], 100),
    'x': np.random.normal(0, 1, size=300),
    'y': np.random.normal(0, 1, size=300)
})

with pm.Model() as hierarchical_model:
    group_means = pm.Normal('group_means', mu=0, sigma=10, shape=3)
    group_sds = pm.HalfNormal('group_sds', sigma=1, shape=3)

    mu = group_means[data.group - 1]
    sigma = group_sds[data.group - 1]

    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=data.y)

    trace = pm.sample(1000, chains=2)
    pm.plot_posterior(trace, var_names=['group_means', 'group_sds'])
```

Slide 15: 
Bayesian Mixture Models
Bayesian mixture models are used to model data that comes from a mixture of different underlying distributions. They are useful for clustering, density estimation, and identifying heterogeneous subpopulations. Code Example:

```python
import pymc3 as pm
import numpy as np
from scipy.stats import norm

data = np.concatenate([norm.rvs(0, 1, size=500), norm.rvs(5, 1, size=300)])

with pm.Model() as mixture_model:
    p = pm.Uniform('p', 0, 1)
    mu1 = pm.Normal('mu1', mu=0, sigma=10)
    mu2 = pm.Normal('mu2', mu=5, sigma=10)
    sigma1 = pm.HalfNormal('sigma1', sigma=1)
    sigma2 = pm.HalfNormal('sigma2', sigma=1)

    category = pm.Categorical('category', p=[p, 1-p], shape=data.shape)
    y_obs = pm.NormalMixture('y_obs', w=category, mu=pm.math.stack([mu1, mu2]),
                             sigma=pm.math.stack([sigma1, sigma2]), observed=data)

    trace = pm.sample(1000, chains=2)
    pm.plot_posterior(trace, var_names=['p', 'mu1', 'mu2', 'sigma1', 'sigma2'])
```

Slide 16: 
Bayesian Non-parametric Models
Bayesian non-parametric models are flexible models that can adapt their complexity to the data, without making strong assumptions about the underlying distribution or the number of parameters. Code Example:

```python
import pymc3 as pm
import numpy as np
from scipy.stats import norm

data = np.concatenate([norm.rvs(0, 1, size=500), norm.rvs(5, 1, size=300)])

with pm.Model() as nonparametric_model:
    mu = pm.Normal('mu', mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal('sigma', sigma=1, shape=2)
    category = pm.Categorical('category', p=[0.5, 0.5], shape=data.shape)

    y_obs = pm.NormalMixture('y_obs', w=category, mu=mu, sigma=sigma, observed=data)

    trace = pm.sample(1000, chains=2)
    pm.plot_posterior(trace, var_names=['mu', 'sigma'])
```

Slide 17: 
Conclusion
Bayesian statistics provides a powerful framework for reasoning under uncertainty, incorporating prior knowledge, and making informed decisions based on data. Python libraries like PyMC3 and other tools make it easier to apply Bayesian methods to various domains, from regression and classification to time series analysis and optimization.

This completes the presentation on Bayesian Statistics in Python, covering various topics and code examples.

## Meta
Here's a title, description, and hashtags for this conversation on Bayesian Statistics in Python, with an institutional tone:

"Mastering Bayesian Statistics with Python"

In this comprehensive presentation, we delve into the powerful world of Bayesian statistics and its implementation in Python. Explore the fundamental principles of Bayes' Theorem, prior distributions, likelihood functions, and posterior distributions. Discover how Markov Chain Monte Carlo (MCMC) methods enable approximating complex posterior distributions. Gain insights into the PyMC3 library, a versatile tool for Bayesian statistical modeling and Probabilistic Machine Learning. Dive into practical applications, including Bayesian linear regression, classification, time series analysis, A/B testing, optimization, and more. Understand the advantages of hierarchical models, mixture models, and non-parametric approaches. This presentation equips you with the knowledge and code examples necessary to harness the power of Bayesian statistics in your data analysis and decision-making processes.

Hashtags: #BayesianStatistics #PyMC3 #ProbabilisticProgramming #UncertaintyQuantification #MachineLearning #DataScience #PythonProgramming #StatisticalModeling #DecisionMaking #DataAnalysis

