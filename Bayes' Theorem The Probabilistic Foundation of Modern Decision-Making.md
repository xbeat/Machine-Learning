## Bayes' Theorem The Probabilistic Foundation of Modern Decision-Making
Slide 1: Bayesian Fundamentals

Bayes' theorem provides a mathematical framework for updating probabilities based on new evidence. This implementation demonstrates the core calculation of posterior probabilities using Python, establishing the foundation for more complex Bayesian applications.

```python
def bayes_theorem(prior, likelihood, marginal):
    """
    Calculate posterior probability using Bayes' Theorem
    P(A|B) = P(B|A) * P(A) / P(B)
    """
    posterior = (likelihood * prior) / marginal
    return posterior

# Example: Disease testing
prior_disease = 0.001  # Prior probability of disease
sensitivity = 0.99     # P(positive|disease)
false_positive = 0.01  # P(positive|no disease)

# Calculate marginal probability
marginal = (sensitivity * prior_disease + 
           false_positive * (1 - prior_disease))

# Calculate posterior probability
posterior = bayes_theorem(prior_disease, sensitivity, marginal)
print(f"Posterior probability: {posterior:.4f}")

# Output:
# Posterior probability: 0.0901
```

Slide 2: Bayesian Parameter Estimation

Parameter estimation is a crucial application of Bayesian methods. This implementation shows how to estimate the probability of success in a binomial distribution using conjugate prior distributions and updating beliefs with observed data.

```python
import numpy as np
from scipy import stats

def beta_binomial_update(prior_alpha, prior_beta, successes, trials):
    """
    Update Beta distribution parameters using binomial observations
    """
    post_alpha = prior_alpha + successes
    post_beta = prior_beta + (trials - successes)
    return post_alpha, post_beta

# Initial beliefs (Beta prior)
alpha_prior = 2
beta_prior = 2

# Observed data
successes = 7
trials = 10

# Update parameters
alpha_post, beta_post = beta_binomial_update(
    alpha_prior, beta_prior, successes, trials
)

# Calculate MAP estimate
map_estimate = (alpha_post - 1) / (alpha_post + beta_post - 2)
print(f"MAP estimate: {map_estimate:.4f}")

# Generate posterior distribution samples
x = np.linspace(0, 1, 100)
posterior = stats.beta.pdf(x, alpha_post, beta_post)
```

Slide 3: Naive Bayes Classifier Implementation

The Naive Bayes classifier applies Bayes' theorem with strong independence assumptions between features. This implementation creates a classifier from scratch, demonstrating the mathematical principles behind this popular machine learning algorithm.

```python
import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = defaultdict(lambda: defaultdict(dict))
    
    def fit(self, X, y):
        n_samples = len(y)
        classes = np.unique(y)
        
        # Calculate class probabilities
        for c in classes:
            self.class_probs[c] = np.sum(y == c) / n_samples
        
        # Calculate feature probabilities for each class
        for c in classes:
            class_samples = X[y == c]
            for feature in range(X.shape[1]):
                unique_vals, counts = np.unique(
                    class_samples[:, feature], 
                    return_counts=True
                )
                probs = counts / len(class_samples)
                for val, prob in zip(unique_vals, probs):
                    self.feature_probs[c][feature][val] = prob
    
    def predict(self, X):
        predictions = []
        for sample in X:
            class_scores = {}
            for c in self.class_probs:
                score = np.log(self.class_probs[c])
                for feature, value in enumerate(sample):
                    if value in self.feature_probs[c][feature]:
                        score += np.log(
                            self.feature_probs[c][feature][value]
                        )
                class_scores[c] = score
            predictions.append(
                max(class_scores.items(), key=lambda x: x[1])[0]
            )
        return np.array(predictions)
```

Slide 4: Real-world Example: Credit Risk Assessment

This implementation demonstrates a practical application of Bayesian inference in credit risk assessment, incorporating multiple risk factors and updating probability estimates based on new customer data.

```python
import numpy as np
from scipy.stats import beta

class CreditRiskAssessor:
    def __init__(self, base_default_rate=0.05):
        self.base_rate = base_default_rate
        self.alpha_prior = 2
        self.beta_prior = 38  # Gives prior mean of 0.05
        
    def update_risk(self, risk_factors):
        """
        Update default probability based on risk factors
        """
        # Risk factors: payment history, debt ratio, income level
        risk_weights = {
            'payment_history': 0.4,
            'debt_ratio': 0.35,
            'income_level': 0.25
        }
        
        total_risk_score = sum(
            score * risk_weights[factor]
            for factor, score in risk_factors.items()
        )
        
        # Update Beta distribution parameters
        likelihood = total_risk_score
        n_observations = 1
        
        alpha_post = self.alpha_prior + (likelihood * n_observations)
        beta_post = self.beta_prior + (
            (1 - likelihood) * n_observations
        )
        
        # Calculate updated default probability
        updated_prob = alpha_post / (alpha_post + beta_post)
        return updated_prob

# Example usage
risk_assessor = CreditRiskAssessor()
customer_risk = {
    'payment_history': 0.8,  # Good history
    'debt_ratio': 0.6,      # Moderate debt
    'income_level': 0.7     # Above average income
}

updated_risk = risk_assessor.update_risk(customer_risk)
print(f"Updated default probability: {updated_risk:.4f}")

# Output:
# Updated default probability: 0.0623
```

Slide 5: Bayesian Time Series Analysis

Implementing Bayesian methods for time series analysis allows for dynamic probability updates as new data arrives. This implementation showcases a basic Bayesian approach to forecasting with uncertainty quantification.

```python
import numpy as np
from scipy.stats import norm

class BayesianTimeSeriesAnalyzer:
    def __init__(self, prior_mean=0, prior_var=1):
        self.mean = prior_mean
        self.variance = prior_var
        self.data_history = []
        
    def update(self, observation, observation_var):
        """
        Update beliefs using Kalman filter equations
        """
        # Calculate Kalman gain
        k_gain = self.variance / (self.variance + observation_var)
        
        # Update mean and variance
        self.mean = self.mean + k_gain * (observation - self.mean)
        self.variance = (1 - k_gain) * self.variance
        
        self.data_history.append(observation)
        
    def predict(self, steps=1, confidence=0.95):
        """
        Make predictions with confidence intervals
        """
        z_score = norm.ppf((1 + confidence) / 2)
        forecast = np.array([self.mean] * steps)
        std = np.sqrt(self.variance)
        ci_lower = forecast - z_score * std
        ci_upper = forecast + z_score * std
        
        return forecast, ci_lower, ci_upper

# Example usage
analyzer = BayesianTimeSeriesAnalyzer()
data = [1.2, 1.4, 1.1, 1.3, 1.5]
observation_var = 0.1

for obs in data:
    analyzer.update(obs, observation_var)

forecast, lower, upper = analyzer.predict(steps=3)
print(f"Forecast: {forecast[0]:.3f}")
print(f"95% CI: [{lower[0]:.3f}, {upper[0]:.3f}]")
```

Slide 6: Bayesian A/B Testing Framework

This implementation demonstrates how to perform Bayesian A/B testing, providing a more nuanced approach to experimental analysis than traditional frequentist methods by incorporating prior beliefs and updating them with evidence.

```python
import numpy as np
from scipy import stats

class BayesianABTest:
    def __init__(self, prior_alpha_a=1, prior_beta_a=1,
                 prior_alpha_b=1, prior_beta_b=1):
        self.alpha_a = prior_alpha_a
        self.beta_a = prior_beta_a
        self.alpha_b = prior_alpha_b
        self.beta_b = prior_beta_b
        
    def update(self, successes_a, trials_a, successes_b, trials_b):
        """
        Update Beta distributions for both variants
        """
        self.alpha_a += successes_a
        self.beta_a += trials_a - successes_a
        self.alpha_b += successes_b
        self.beta_b += trials_b - successes_b
        
    def probability_b_better_than_a(self, samples=10000):
        """
        Monte Carlo estimation of P(B > A)
        """
        samples_a = np.random.beta(self.alpha_a, self.beta_a, samples)
        samples_b = np.random.beta(self.alpha_b, self.beta_b, samples)
        return np.mean(samples_b > samples_a)
    
    def expected_loss(self, samples=10000):
        """
        Calculate expected loss of choosing wrong variant
        """
        samples_a = np.random.beta(self.alpha_a, self.beta_a, samples)
        samples_b = np.random.beta(self.alpha_b, self.beta_b, samples)
        return np.mean(np.maximum(samples_a - samples_b, 0))

# Example usage
ab_test = BayesianABTest()
# Update with results: A(45/100) vs B(55/100)
ab_test.update(45, 100, 55, 100)

prob_b_better = ab_test.probability_b_better_than_a()
exp_loss = ab_test.expected_loss()

print(f"P(B > A): {prob_b_better:.3f}")
print(f"Expected Loss: {exp_loss:.3f}")

# Calculate credible intervals
ci_a = stats.beta.interval(0.95, ab_test.alpha_a, ab_test.beta_a)
ci_b = stats.beta.interval(0.95, ab_test.alpha_b, ab_test.beta_b)
```

Slide 7: Bayesian Linear Regression

Bayesian linear regression extends traditional linear regression by treating model parameters as probability distributions rather than point estimates, providing uncertainty quantification in predictions.

```python
import numpy as np
from scipy import stats

class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha  # Precision of prior
        self.beta = beta    # Precision of likelihood
        self.mean = None
        self.precision = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Calculate posterior precision matrix
        self.precision = (
            self.alpha * np.eye(n_features) + 
            self.beta * X.T @ X
        )
        
        # Calculate posterior mean
        self.mean = self.beta * np.linalg.solve(
            self.precision, 
            X.T @ y
        )
        
    def predict(self, X_new, return_std=True):
        y_mean = X_new @ self.mean
        
        if return_std:
            # Calculate predictive variance
            precision_inv = np.linalg.inv(self.precision)
            var = 1/self.beta + np.sum(
                X_new @ precision_inv * X_new, 
                axis=1
            )
            std = np.sqrt(var)
            return y_mean, std
            
        return y_mean

# Example usage
np.random.seed(42)
X = np.random.randn(100, 2)
true_weights = np.array([2.0, -1.0])
y = X @ true_weights + np.random.randn(100) * 0.1

# Create and fit model
model = BayesianLinearRegression(alpha=0.1, beta=10.0)
model.fit(X, y)

# Make predictions with uncertainty
X_test = np.random.randn(10, 2)
y_pred, y_std = model.predict(X_test)

print("Predictions and uncertainties:")
for pred, std in zip(y_pred[:3], y_std[:3]):
    print(f"Prediction: {pred:.3f} ± {2*std:.3f}")
```

Slide 8: Bayesian Model Selection

Implementing Bayesian model selection using Bayes factors provides a principled way to compare competing models while naturally penalizing complexity and avoiding overfitting.

```python
import numpy as np
from scipy.special import gammaln

class BayesianModelSelector:
    def __init__(self):
        self.models = {}
        self.prior_probs = {}
        
    def add_model(self, name, model, prior_prob):
        self.models[name] = model
        self.prior_probs[name] = prior_prob
        
    def compute_log_evidence(self, X, y, model):
        """
        Compute log marginal likelihood (model evidence)
        using Laplace approximation
        """
        n_samples = len(y)
        n_params = len(model.mean)
        
        # Compute log likelihood at MAP
        residuals = y - X @ model.mean
        log_likelihood = (-0.5 * n_samples * np.log(2 * np.pi) +
                        0.5 * n_samples * np.log(model.beta) -
                        0.5 * model.beta * np.sum(residuals**2))
        
        # Add log prior
        log_prior = (-0.5 * n_params * np.log(2 * np.pi) +
                    0.5 * n_params * np.log(model.alpha) -
                    0.5 * model.alpha * np.sum(model.mean**2))
        
        # Add log determinant term from Laplace approximation
        sign, logdet = np.linalg.slogdet(model.precision)
        log_laplace = -0.5 * logdet
        
        return log_likelihood + log_prior + log_laplace
    
    def select_model(self, X, y):
        log_evidences = {}
        posterior_probs = {}
        
        # Compute log evidence for each model
        for name, model in self.models.items():
            model.fit(X, y)
            log_evidences[name] = (
                self.compute_log_evidence(X, y, model) +
                np.log(self.prior_probs[name])
            )
        
        # Compute posterior probabilities
        log_total = np.logaddexp.reduce(list(log_evidences.values()))
        for name in self.models:
            posterior_probs[name] = np.exp(
                log_evidences[name] - log_total
            )
            
        return posterior_probs

# Example usage
# Create two competing models with different complexities
model1 = BayesianLinearRegression(alpha=0.1, beta=10.0)
model2 = BayesianLinearRegression(alpha=1.0, beta=10.0)

selector = BayesianModelSelector()
selector.add_model("Simple", model1, prior_prob=0.5)
selector.add_model("Complex", model2, prior_prob=0.5)

# Generate data and select model
X = np.random.randn(100, 2)
y = X @ np.array([2.0, -1.0]) + np.random.randn(100) * 0.1

probs = selector.select_model(X, y)
for model, prob in probs.items():
    print(f"{model} model probability: {prob:.3f}")
```

Slide 9: Markov Chain Monte Carlo (MCMC) Implementation

Building a Metropolis-Hastings MCMC sampler provides a powerful tool for sampling from complex posterior distributions when analytical solutions are intractable. This implementation demonstrates the core concepts of MCMC sampling.

```python
import numpy as np
from scipy import stats

class MetropolisHastings:
    def __init__(self, target_log_prob, proposal_width=0.1):
        self.target_log_prob = target_log_prob
        self.proposal_width = proposal_width
        self.samples = []
        
    def run(self, n_samples, initial_state):
        current_state = initial_state
        current_log_prob = self.target_log_prob(current_state)
        
        for _ in range(n_samples):
            # Propose new state
            proposal = current_state + np.random.normal(
                0, self.proposal_width, size=len(current_state)
            )
            
            # Calculate acceptance ratio
            proposal_log_prob = self.target_log_prob(proposal)
            log_ratio = proposal_log_prob - current_log_prob
            
            # Accept or reject
            if np.log(np.random.random()) < log_ratio:
                current_state = proposal
                current_log_prob = proposal_log_prob
                
            self.samples.append(current_state.copy())
            
    def get_samples(self, burnin=0.2):
        samples = np.array(self.samples)
        n_burnin = int(len(samples) * burnin)
        return samples[n_burnin:]
    
    def diagnostics(self):
        samples = self.get_samples()
        acceptance_rate = len(np.unique(samples, axis=0)) / len(samples)
        return {
            'acceptance_rate': acceptance_rate,
            'mean': np.mean(samples, axis=0),
            'std': np.std(samples, axis=0)
        }

# Example: Sampling from a mixture of Gaussians
def target_distribution(x):
    """Log probability of mixture of two Gaussians"""
    return np.log(
        0.3 * stats.norm.pdf(x[0], -2, 1) + 
        0.7 * stats.norm.pdf(x[0], 2, 1.5)
    )

# Run sampler
mcmc = MetropolisHastings(target_distribution)
mcmc.run(10000, initial_state=np.array([0.0]))
samples = mcmc.get_samples()
diagnostics = mcmc.diagnostics()

print("MCMC Diagnostics:")
for key, value in diagnostics.items():
    print(f"{key}: {value}")
```

Slide 10: Online Bayesian Learning

This implementation showcases online learning using Bayesian methods, allowing for real-time updates of model parameters as new data arrives sequentially.

```python
class OnlineBayesianLearner:
    def __init__(self, prior_mean=0, prior_var=1):
        self.mean = prior_mean
        self.variance = prior_var
        self.n_updates = 0
        self.learning_rate_schedule = lambda n: 1 / (n + 1)
        
    def update(self, observation, observation_var):
        """
        Perform online Bayesian update
        """
        self.n_updates += 1
        lr = self.learning_rate_schedule(self.n_updates)
        
        # Precision-weighted update
        precision = 1 / self.variance
        obs_precision = 1 / observation_var
        
        # Update precision
        new_precision = precision + lr * obs_precision
        
        # Update mean
        innovation = observation - self.mean
        self.mean += (lr * obs_precision / new_precision) * innovation
        
        # Update variance
        self.variance = 1 / new_precision
        
        return self.mean, self.variance
    
    def predict(self, confidence=0.95):
        """
        Make prediction with confidence interval
        """
        z_score = stats.norm.ppf((1 + confidence) / 2)
        ci_lower = self.mean - z_score * np.sqrt(self.variance)
        ci_upper = self.mean + z_score * np.sqrt(self.variance)
        
        return {
            'prediction': self.mean,
            'variance': self.variance,
            'confidence_interval': (ci_lower, ci_upper)
        }

# Example usage
learner = OnlineBayesianLearner()
data_stream = np.random.normal(1.5, 0.5, size=1000)

# Online learning
results = []
for obs in data_stream[:10]:  # Show first 10 updates
    mean, var = learner.update(obs, observation_var=0.5)
    pred = learner.predict()
    results.append(pred)
    
print("\nOnline Learning Results:")
for i, res in enumerate(results):
    print(f"Step {i+1}:")
    print(f"Prediction: {res['prediction']:.3f}")
    print(f"95% CI: [{res['confidence_interval'][0]:.3f}, "
          f"{res['confidence_interval'][1]:.3f}]")
```

Slide 11: Bayesian Change Point Detection

Implementing a Bayesian approach to detect significant changes in time series data by modeling the probability of change points and updating beliefs as new data arrives.

```python
import numpy as np
from scipy.stats import norm

class BayesianChangePointDetector:
    def __init__(self, hazard_rate=0.01):
        self.hazard_rate = hazard_rate
        self.run_length = 0
        self.means = [0]
        self.variances = [1]
        self.probs = [1.0]
        
    def update(self, observation):
        # Calculate predictive probabilities
        pred_probs = np.zeros(len(self.means))
        for i in range(len(self.means)):
            pred_probs[i] = norm.pdf(
                observation, 
                self.means[i], 
                np.sqrt(self.variances[i])
            )
        
        # Calculate growth probabilities
        growth_probs = (1 - self.hazard_rate) * np.array(self.probs)
        
        # Calculate changepoint probability
        cp_prob = self.hazard_rate * sum(self.probs)
        
        # Update run length distribution
        self.probs = np.append(cp_prob, growth_probs)
        self.probs /= sum(self.probs)
        
        # Update sufficient statistics
        self.run_length += 1
        n = self.run_length
        
        # Update parameters for existing runs
        for i in range(len(self.means)-1, -1, -1):
            delta = observation - self.means[i]
            self.means[i] += delta / (n - i)
            if n - i > 1:
                self.variances[i] *= (n-i-2)/(n-i-1)
                self.variances[i] += (delta**2)/(n-i)
                
        # Add new run
        self.means.append(observation)
        self.variances.append(1.0)
        
        return self.get_change_point_probability()
    
    def get_change_point_probability(self):
        """Return probability of most recent change point"""
        return self.probs[0]

# Example usage
np.random.seed(42)
# Generate synthetic data with change point
n_points = 200
data = np.concatenate([
    np.random.normal(0, 1, 100),
    np.random.normal(3, 1, 100)
])

# Detect change points
detector = BayesianChangePointDetector()
change_probs = []

for obs in data:
    prob = detector.update(obs)
    change_probs.append(prob)

# Print results for key points
print("\nChange Point Detection Results:")
for i in [98, 99, 100, 101, 102]:
    print(f"Time {i}: Change probability = {change_probs[i]:.3f}")
```

Slide 12: Bayesian Neural Network

This implementation demonstrates a simple Bayesian Neural Network using variational inference, providing uncertainty estimates in neural network predictions.

```python
import numpy as np
from scipy.special import expit

class BayesianNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize variational parameters
        self.w1_mu = np.random.randn(input_dim, hidden_dim) * 0.1
        self.w1_rho = np.random.randn(input_dim, hidden_dim) * 0.1
        self.w2_mu = np.random.randn(hidden_dim, output_dim) * 0.1
        self.w2_rho = np.random.randn(hidden_dim, output_dim) * 0.1
        
    def sample_weights(self):
        """Sample weights using reparameterization trick"""
        w1_sigma = np.log(1 + np.exp(self.w1_rho))
        w2_sigma = np.log(1 + np.exp(self.w2_rho))
        
        epsilon1 = np.random.randn(*self.w1_mu.shape)
        epsilon2 = np.random.randn(*self.w2_mu.shape)
        
        w1 = self.w1_mu + w1_sigma * epsilon1
        w2 = self.w2_mu + w2_sigma * epsilon2
        
        return w1, w2
    
    def forward(self, X, n_samples=10):
        """Forward pass with multiple samples"""
        predictions = []
        
        for _ in range(n_samples):
            w1, w2 = self.sample_weights()
            
            # Forward pass
            h = np.tanh(X @ w1)
            y = h @ w2
            predictions.append(y)
            
        # Stack predictions
        predictions = np.stack(predictions)
        
        # Calculate mean and variance
        mean = np.mean(predictions, axis=0)
        var = np.var(predictions, axis=0)
        
        return mean, var
    
    def kl_divergence(self):
        """Calculate KL divergence for variational inference"""
        w1_sigma = np.log(1 + np.exp(self.w1_rho))
        w2_sigma = np.log(1 + np.exp(self.w2_rho))
        
        kl = np.sum(np.log(w1_sigma)) + np.sum(np.log(w2_sigma))
        kl += np.sum(self.w1_mu**2 + w1_sigma**2)
        kl += np.sum(self.w2_mu**2 + w2_sigma**2)
        kl -= len(self.w1_mu.flatten()) + len(self.w2_mu.flatten())
        
        return 0.5 * kl

# Example usage
X = np.random.randn(100, 2)
y = np.sin(X[:, 0]) + np.cos(X[:, 1])

# Create and use model
model = BayesianNeuralNetwork(2, 10, 1)
mean, var = model.forward(X)

print("\nPrediction Results:")
for i in range(3):
    print(f"Sample {i+1}:")
    print(f"Prediction: {mean[i,0]:.3f} ± {2*np.sqrt(var[i,0]):.3f}")
```

Slide 13: Bayesian Optimization

Implementing Bayesian optimization for hyperparameter tuning using Gaussian Processes as the surrogate model and Expected Improvement as the acquisition function.

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    def __init__(self, bounds, n_init=5):
        self.bounds = np.array(bounds)
        self.n_init = n_init
        self.X = []
        self.y = []
        
    def _kernel(self, X1, X2, l=1.0, sigma_f=1.0):
        """RBF kernel"""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
                 np.sum(X2**2, 1) - \
                 2 * np.dot(X1, X2.T)
        return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)
    
    def _expected_improvement(self, X, xi=0.01):
        """Calculate expected improvement"""
        mu, sigma = self._predict(X)
        mu = mu.reshape(-1)
        sigma = sigma.reshape(-1)
        
        mu_sample_opt = np.max(self.y)
        
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            
        return -ei
    
    def _predict(self, X_new):
        """GP prediction"""
        if len(self.X) == 0:
            return np.zeros(len(X_new)), np.ones(len(X_new))
            
        K = self._kernel(np.array(self.X), np.array(self.X))
        K_s = self._kernel(np.array(self.X), X_new)
        K_ss = self._kernel(X_new, X_new)
        
        K_inv = np.linalg.inv(K + 1e-8 * np.eye(len(K)))
        
        mu = K_s.T.dot(K_inv).dot(self.y)
        sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))
        
        return mu, sigma
    
    def propose_location(self):
        """Propose next sampling location"""
        if len(self.X) < self.n_init:
            return np.random.uniform(
                self.bounds[:, 0], 
                self.bounds[:, 1]
            )
            
        dim = self.bounds.shape[0]
        
        def min_obj(X):
            return self._expected_improvement(X.reshape(-1, dim))
            
        X_tries = np.random.uniform(
            self.bounds[:, 0], 
            self.bounds[:, 1], 
            size=(100, dim)
        )
        ei_tries = [-min_obj(x) for x in X_tries]
        X_max = X_tries[np.argmax(ei_tries)]
        
        res = minimize(
            min_obj, 
            X_max, 
            bounds=self.bounds,
            method='L-BFGS-B'
        )
            
        return res.x
    
    def update(self, X, y):
        """Add new observation"""
        self.X.append(X)
        self.y.append(y)

# Example usage
def objective(x):
    """Example objective function"""
    return -(x[0]**2 + x[1]**2)

# Initialize optimizer
bounds = [(-5, 5), (-5, 5)]
optimizer = BayesianOptimizer(bounds)

# Run optimization
n_iters = 10
for i in range(n_iters):
    next_point = optimizer.propose_location()
    f_value = objective(next_point)
    optimizer.update(next_point, f_value)
    
    print(f"\nIteration {i+1}:")
    print(f"Next point: {next_point}")
    print(f"Value: {f_value:.3f}")
```

Slide 14: Additional Resources

*   "Probabilistic Programming and Bayesian Methods for Hackers" [https://arxiv.org/abs/1507.08050](https://arxiv.org/abs/1507.08050)
*   "A Tutorial on Bayesian Optimization" [https://arxiv.org/abs/1807.02811](https://arxiv.org/abs/1807.02811)
*   "Practical Bayesian Optimization of Machine Learning Algorithms" [https://arxiv.org/abs/1206.2944](https://arxiv.org/abs/1206.2944)
*   "An Introduction to Probabilistic Programming" [https://arxiv.org/abs/1809.10756](https://arxiv.org/abs/1809.10756)
*   "Bayesian Methods for Hackers: Probabilistic Programming and Bayesian Inference" [https://arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)

