## Bayesian vs Frequentist Dart Game
Slide 1: Bayesian vs Frequentist Fundamentals

In statistical inference, understanding the core differences between Bayesian and Frequentist approaches is crucial. This implementation demonstrates basic probability calculations using both methodologies through a coin flip experiment simulation.

```python
import numpy as np
from scipy import stats

class CoinFlipExperiment:
    def __init__(self, true_probability=0.5):
        self.true_prob = true_probability
        self.data = []
        
    def flip_coin(self, n_flips):
        return np.random.binomial(1, self.true_prob, n_flips)
    
    def frequentist_estimate(self, n_flips):
        flips = self.flip_coin(n_flips)
        self.data.extend(flips)
        return np.mean(self.data)
    
    def bayesian_estimate(self, prior_a=1, prior_b=1):
        # Using Beta distribution as conjugate prior
        successes = sum(self.data)
        failures = len(self.data) - successes
        posterior_a = prior_a + successes
        posterior_b = prior_b + failures
        return posterior_a / (posterior_a + posterior_b)

# Example usage
experiment = CoinFlipExperiment(true_probability=0.7)
freq_estimate = experiment.frequentist_estimate(100)
bayes_estimate = experiment.bayesian_estimate()

print(f"True probability: {experiment.true_prob}")
print(f"Frequentist estimate: {freq_estimate:.3f}")
print(f"Bayesian estimate: {bayes_estimate:.3f}")
```

Slide 2: Maximum Likelihood Estimation Implementation

Maximum Likelihood Estimation (MLE) represents a cornerstone of frequentist statistics. This implementation demonstrates MLE for estimating parameters of a normal distribution using numerical optimization.

```python
import numpy as np
from scipy.optimize import minimize

def normal_negative_log_likelihood(params, data):
    mu, sigma = params
    return -np.sum(stats.norm.logpdf(data, mu, sigma))

# Generate sample data
true_mu, true_sigma = 2.5, 1.5
np.random.seed(42)
sample_data = np.random.normal(true_mu, true_sigma, 1000)

# MLE estimation
initial_guess = [0, 1]
result = minimize(normal_negative_log_likelihood, 
                 initial_guess,
                 args=(sample_data,),
                 method='Nelder-Mead')

estimated_mu, estimated_sigma = result.x
print(f"True parameters: μ={true_mu}, σ={true_sigma}")
print(f"MLE estimates: μ={estimated_mu:.3f}, σ={estimated_sigma:.3f}")
```

Slide 3: Bayesian Parameter Estimation

This implementation showcases Bayesian parameter estimation using Markov Chain Monte Carlo (MCMC) with PyMC3, demonstrating how to incorporate prior beliefs and update them with observed data.

```python
import pymc3 as pm
import matplotlib.pyplot as plt

def bayesian_parameter_estimation(data, n_samples=2000):
    with pm.Model() as model:
        # Prior distributions
        mu = pm.Normal('mu', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=10)
        
        # Likelihood
        observations = pm.Normal('obs', 
                               mu=mu, 
                               sigma=sigma, 
                               observed=data)
        
        # Inference
        trace = pm.sample(n_samples, 
                         tune=1000, 
                         return_inferencedata=False)
    
    return trace

# Generate and analyze data
np.random.seed(42)
data = np.random.normal(2.5, 1.5, 100)
trace = bayesian_parameter_estimation(data)

print("Posterior estimates:")
print(f"μ: {trace['mu'].mean():.3f} ± {trace['mu'].std():.3f}")
print(f"σ: {trace['sigma'].mean():.3f} ± {trace['sigma'].std():.3f}")
```

Slide 4: Frequentist A/B Testing Implementation

A practical implementation of frequentist A/B testing using hypothesis testing and confidence intervals, demonstrating how to analyze conversion rates between two different webpage versions.

```python
import scipy.stats as stats
import numpy as np

class FrequentistABTest:
    def __init__(self, control_data, treatment_data):
        self.control = control_data
        self.treatment = treatment_data
        
    def calculate_statistics(self):
        n1, n2 = len(self.control), len(self.treatment)
        p1, p2 = np.mean(self.control), np.mean(self.treatment)
        
        # Calculate pooled standard error
        p_pooled = (sum(self.control) + sum(self.treatment)) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        # Calculate z-score
        z_score = (p2 - p1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            'control_conv': p1,
            'treatment_conv': p2,
            'difference': p2 - p1,
            'z_score': z_score,
            'p_value': p_value
        }

# Example usage
np.random.seed(42)
control = np.random.binomial(1, 0.10, 1000)    # 10% conversion
treatment = np.random.binomial(1, 0.12, 1000)  # 12% conversion

ab_test = FrequentistABTest(control, treatment)
results = ab_test.calculate_statistics()

for key, value in results.items():
    print(f"{key}: {value:.4f}")
```

Slide 5: Bayesian A/B Testing

Implementing Bayesian A/B testing using Beta distributions as conjugate priors for conversion rates, providing a more nuanced view of uncertainty in the testing process.

```python
class BayesianABTest:
    def __init__(self, alpha_prior=1, beta_prior=1):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
    def analyze(self, control_successes, control_trials,
               treatment_successes, treatment_trials,
               n_samples=100000):
        
        # Posterior distributions
        control_post = stats.beta(
            self.alpha_prior + control_successes,
            self.beta_prior + control_trials - control_successes
        )
        
        treatment_post = stats.beta(
            self.alpha_prior + treatment_successes,
            self.beta_prior + treatment_trials - treatment_successes
        )
        
        # Generate samples
        control_samples = control_post.rvs(n_samples)
        treatment_samples = treatment_post.rvs(n_samples)
        
        # Calculate probability of improvement
        prob_improvement = np.mean(treatment_samples > control_samples)
        
        # Expected lift
        expected_lift = np.mean((treatment_samples - control_samples) / 
                              control_samples)
        
        return {
            'prob_improvement': prob_improvement,
            'expected_lift': expected_lift,
            'control_mean': np.mean(control_samples),
            'treatment_mean': np.mean(treatment_samples)
        }

# Example usage
bayes_test = BayesianABTest(alpha_prior=1, beta_prior=1)
results = bayes_test.analyze(
    control_successes=100,
    control_trials=1000,
    treatment_successes=120,
    treatment_trials=1000
)

for key, value in results.items():
    print(f"{key}: {value:.4f}")
```

Slide 6: Bayesian Linear Regression

A comprehensive implementation of Bayesian Linear Regression using probabilistic programming, demonstrating parameter uncertainty estimation and posterior predictive distributions.

```python
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

class BayesianLinearRegression:
    def __init__(self):
        self.trace = None
        self.model = None
    
    def fit(self, X, y, n_samples=2000):
        with pm.Model() as self.model:
            # Priors for unknown model parameters
            alpha = pm.Normal('alpha', mu=0, sd=10)
            beta = pm.Normal('beta', mu=0, sd=10)
            sigma = pm.HalfNormal('sigma', sd=1)
            
            # Expected value of outcome
            mu = alpha + beta * X
            
            # Likelihood (sampling distribution) of observations
            y_obs = pm.Normal('y_obs', mu=mu, sd=sigma, observed=y)
            
            # Inference
            self.trace = pm.sample(n_samples, 
                                 tune=1000, 
                                 return_inferencedata=False)
        
        return self
    
    def predict(self, X_new):
        with self.model:
            pm.set_data({'X': X_new})
            posterior_pred = pm.sample_posterior_predictive(
                self.trace, 
                vars=[self.model.alpha, self.model.beta]
            )
        return posterior_pred

# Example usage
np.random.seed(42)
X = np.linspace(0, 10, 100)
true_alpha, true_beta = 2, 0.5
y = true_alpha + true_beta * X + np.random.normal(0, 0.5, size=100)

model = BayesianLinearRegression()
model.fit(X, y)

# Print parameter estimates
print(f"True alpha: {true_alpha}, Estimated: {model.trace['alpha'].mean():.3f}")
print(f"True beta: {true_beta}, Estimated: {model.trace['beta'].mean():.3f}")
```

Slide 7: Frequentist Time Series Analysis

Implementation of classical time series decomposition and forecasting using frequentist methods, demonstrating seasonal decomposition and ARIMA modeling for prediction.

```python
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd

class FrequentistTimeSeriesAnalysis:
    def __init__(self, data, frequency=12):
        self.data = data
        self.frequency = frequency
        
    def decompose(self):
        result = seasonal_decompose(self.data, 
                                  period=self.frequency,
                                  model='multiplicative')
        return {
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid
        }
    
    def fit_arima(self, order=(1,1,1)):
        model = ARIMA(self.data, order=order)
        fitted = model.fit()
        return fitted
    
    def forecast(self, model, steps=12):
        forecast = model.forecast(steps=steps)
        conf_int = model.get_forecast(steps=steps).conf_int()
        return forecast, conf_int

# Example usage
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=100, freq='M')
trend = np.linspace(0, 10, 100)
seasonal = np.sin(np.linspace(0, 8*np.pi, 100))
noise = np.random.normal(0, 0.5, 100)
data = trend + seasonal + noise

ts_analysis = FrequentistTimeSeriesAnalysis(data)
decomposition = ts_analysis.decompose()
model = ts_analysis.fit_arima()
forecast, conf_intervals = ts_analysis.forecast(model)

print(f"AIC: {model.aic:.2f}")
print(f"Forecast next 3 steps: {forecast[:3]}")
```

Slide 8: Bayesian Time Series Modeling

Implementation of a Bayesian structural time series model using PyMC3, incorporating prior knowledge and uncertainty in trend and seasonal components.

```python
import pymc3 as pm
import numpy as np

class BayesianTimeSeriesModel:
    def __init__(self, data, seasonal_periods=12):
        self.data = data
        self.seasonal_periods = seasonal_periods
        self.trace = None
        
    def build_and_fit(self, samples=2000):
        n = len(self.data)
        
        with pm.Model() as model:
            # Level and trend components
            level_sd = pm.HalfNormal('level_sd', sd=0.5)
            trend_sd = pm.HalfNormal('trend_sd', sd=0.1)
            
            level = pm.GaussianRandomWalk('level', 
                                        sd=level_sd,
                                        shape=n)
            trend = pm.GaussianRandomWalk('trend',
                                        sd=trend_sd,
                                        shape=n)
            
            # Seasonal component
            season_sd = pm.HalfNormal('season_sd', sd=0.1)
            season_raw = pm.Normal('season_raw', 
                                 mu=0,
                                 sd=season_sd,
                                 shape=self.seasonal_periods)
            
            season = pm.Deterministic('season',
                                    season_raw - pm.math.mean(season_raw))
            
            # Final model
            sigma = pm.HalfNormal('sigma', sd=1)
            mu = level + trend + season[np.arange(n) % self.seasonal_periods]
            
            y = pm.Normal('y', mu=mu, sd=sigma, observed=self.data)
            
            # Inference
            self.trace = pm.sample(samples, 
                                 tune=1000,
                                 return_inferencedata=False)
        
        return self.trace

# Example usage
np.random.seed(42)
n_points = 100
trend = np.linspace(0, 5, n_points)
seasonal = np.tile(np.sin(np.linspace(0, 2*np.pi, 12)), n_points//12 + 1)[:n_points]
data = trend + seasonal + np.random.normal(0, 0.2, n_points)

model = BayesianTimeSeriesModel(data)
trace = model.build_and_fit()

print("Posterior estimates:")
print(f"Noise σ: {trace['sigma'].mean():.3f} ± {trace['sigma'].std():.3f}")
print(f"Level σ: {trace['level_sd'].mean():.3f} ± {trace['level_sd'].std():.3f}")
```

Slide 9: Bayesian Neural Networks

Implementation of a Bayesian Neural Network using PyMC3, demonstrating how to incorporate uncertainty in weight parameters and make probabilistic predictions.

```python
import pymc3 as pm
import numpy as np
import theano.tensor as tt

class BayesianNeuralNetwork:
    def __init__(self, input_dim, hidden_dim=50):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.trace = None
        
    def build_and_fit(self, X, y, samples=2000):
        with pm.Model() as model:
            # Prior distributions for weights
            w1 = pm.Normal('w1', 
                         mu=0, 
                         sd=1, 
                         shape=(self.input_dim, self.hidden_dim))
            w2 = pm.Normal('w2', 
                         mu=0, 
                         sd=1, 
                         shape=(self.hidden_dim, 1))
            
            # Prior for bias
            b1 = pm.Normal('b1', mu=0, sd=1, shape=(self.hidden_dim,))
            b2 = pm.Normal('b2', mu=0, sd=1, shape=(1,))
            
            # Neural network architecture
            act_1 = tt.tanh(tt.dot(X, w1) + b1)
            mu = tt.dot(act_1, w2) + b2
            
            # Likelihood
            sigma = pm.HalfNormal('sigma', sd=1)
            y_obs = pm.Normal('y_obs', mu=mu, sd=sigma, observed=y)
            
            # Inference
            self.trace = pm.sample(samples, 
                                 tune=1000,
                                 return_inferencedata=False)
        
        return self.trace
    
    def predict(self, X_new, samples=1000):
        preds = np.zeros((samples, len(X_new)))
        for i in range(samples):
            idx = np.random.randint(0, len(self.trace['w1']))
            w1_samp = self.trace['w1'][idx]
            w2_samp = self.trace['w2'][idx]
            b1_samp = self.trace['b1'][idx]
            b2_samp = self.trace['b2'][idx]
            
            # Forward pass
            act_1 = np.tanh(np.dot(X_new, w1_samp) + b1_samp)
            preds[i] = np.dot(act_1, w2_samp) + b2_samp.reshape(-1)
            
        return preds.mean(axis=0), preds.std(axis=0)

# Example usage
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.1, size=X.shape)

model = BayesianNeuralNetwork(input_dim=1)
trace = model.build_and_fit(X, y)
y_pred_mean, y_pred_std = model.predict(X)

print("Model trained. Example predictions:")
print(f"X: {X[0]}, Pred: {y_pred_mean[0]:.3f} ± {y_pred_std[0]:.3f}")
```

Slide 10: Frequentist vs Bayesian Model Selection

Implementation comparing frequentist AIC/BIC criteria with Bayesian model selection using Bayes factors for polynomial regression models.

```python
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

class ModelSelection:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n = len(y)
        
    def frequentist_selection(self, max_degree=5):
        results = {}
        for degree in range(1, max_degree + 1):
            X_poly = np.vander(self.X, degree + 1)
            model = LinearRegression()
            model.fit(X_poly, self.y)
            
            # Calculate AIC and BIC
            y_pred = model.predict(X_poly)
            mse = np.mean((self.y - y_pred) ** 2)
            k = degree + 1
            
            aic = self.n * np.log(mse) + 2 * k
            bic = self.n * np.log(mse) + np.log(self.n) * k
            
            results[degree] = {
                'aic': aic,
                'bic': bic,
                'r2': model.score(X_poly, self.y)
            }
        return results
    
    def bayesian_selection(self, max_degree=5):
        results = {}
        for degree in range(1, max_degree + 1):
            X_poly = np.vander(self.X, degree + 1)
            
            # Calculate marginal likelihood using BIC approximation
            model = LinearRegression()
            model.fit(X_poly, self.y)
            
            y_pred = model.predict(X_poly)
            mse = np.mean((self.y - y_pred) ** 2)
            k = degree + 1
            
            # Log marginal likelihood approximation
            log_ml = -0.5 * (self.n * np.log(2 * np.pi * mse) + 
                           k * np.log(self.n))
            
            results[degree] = {
                'log_marginal_likelihood': log_ml,
                'bayes_factor': np.exp(log_ml) if degree > 1 
                               else None
            }
        return results

# Example usage
np.random.seed(42)
X = np.linspace(-3, 3, 100)
y = 1 + 2*X + 0.5*X**2 + np.random.normal(0, 0.1, size=X.shape)

selector = ModelSelection(X, y)
freq_results = selector.frequentist_selection()
bayes_results = selector.bayesian_selection()

print("Frequentist Selection Results:")
for degree, metrics in freq_results.items():
    print(f"Degree {degree}: AIC={metrics['aic']:.2f}, BIC={metrics['bic']:.2f}")

print("\nBayesian Selection Results:")
for degree, metrics in bayes_results.items():
    print(f"Degree {degree}: Log ML={metrics['log_marginal_likelihood']:.2f}")
```

Slide 11: Implementing Bayesian Optimization

A practical implementation of Bayesian optimization using Gaussian Processes for hyperparameter tuning, demonstrating the balance between exploration and exploitation.

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
        
    def acquisition_function(self, X, model):
        mu, sigma = model.predict(X.reshape(-1, 1))
        sigma = np.maximum(sigma, 1e-4)
        
        # Expected Improvement
        imp = mu - np.max(self.y) - 0.01
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return -ei
    
    def optimize(self, objective_func, n_iter=20):
        # Initial random sampling
        X_init = np.random.uniform(
            self.bounds[0], 
            self.bounds[1], 
            size=self.n_init
        )
        y_init = [objective_func(x) for x in X_init]
        
        self.X = list(X_init)
        self.y = list(y_init)
        
        for i in range(n_iter):
            # Fit GP model
            kernel = RBF(length_scale=1.0)
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5
            )
            model.fit(np.array(self.X).reshape(-1, 1), self.y)
            
            # Find next point to evaluate
            result = minimize(
                lambda x: self.acquisition_function(x, model),
                x0=np.random.uniform(self.bounds[0], self.bounds[1]),
                bounds=[self.bounds],
                method='L-BFGS-B'
            )
            
            next_x = result.x[0]
            next_y = objective_func(next_x)
            
            self.X.append(next_x)
            self.y.append(next_y)
            
        best_idx = np.argmax(self.y)
        return self.X[best_idx], self.y[best_idx]

# Example usage
def objective(x):
    return -(x - 2)**2 + 10

optimizer = BayesianOptimizer(bounds=[-5, 5])
best_x, best_y = optimizer.optimize(objective)

print(f"Best found value: {best_y:.3f} at x = {best_x:.3f}")
```

Slide 12: Bayesian Change Point Detection

Implementation of a Bayesian approach to detecting changes in time series data using conjugate priors and MCMC sampling.

```python
import numpy as np
import pymc3 as pm

class BayesianChangePointDetector:
    def __init__(self, data):
        self.data = np.array(data)
        self.n = len(data)
        self.trace = None
        
    def fit(self, max_changepoints=2, samples=2000):
        with pm.Model() as model:
            # Prior on changepoint locations
            changepoint_locs = pm.DiscreteUniform(
                'changepoint_locs',
                lower=1,
                upper=self.n-2,
                shape=max_changepoints
            )
            
            # Prior on segment parameters
            sigma = pm.HalfNormal('sigma', sd=1)
            means = pm.Normal('means', 
                            mu=np.mean(self.data), 
                            sd=np.std(self.data),
                            shape=max_changepoints+1)
            
            # Generate segments
            idx = tt.arange(self.n)
            segment_means = means[0] * tt.ones_like(idx)
            
            for i in range(max_changepoints):
                segment_means = tt.switch(
                    idx >= changepoint_locs[i],
                    means[i+1],
                    segment_means
                )
            
            # Likelihood
            y = pm.Normal('y',
                         mu=segment_means,
                         sd=sigma,
                         observed=self.data)
            
            # Inference
            self.trace = pm.sample(samples,
                                 tune=1000,
                                 return_inferencedata=False)
            
        return self.trace
    
    def get_changepoints(self):
        cp_samples = self.trace['changepoint_locs']
        mean_cps = np.mean(cp_samples, axis=0)
        std_cps = np.std(cp_samples, axis=0)
        return mean_cps, std_cps

# Example usage
np.random.seed(42)
n_points = 200
data = np.concatenate([
    np.random.normal(0, 1, 80),
    np.random.normal(3, 1, 70),
    np.random.normal(-1, 1, 50)
])

detector = BayesianChangePointDetector(data)
trace = detector.fit()
mean_cps, std_cps = detector.get_changepoints()

print("Detected changepoints:")
for i, (mean, std) in enumerate(zip(mean_cps, std_cps)):
    print(f"Changepoint {i+1}: {mean:.1f} ± {std:.1f}")
```

Slide 13: Frequentist vs Bayesian Clustering

A comparative implementation of clustering methods using both frequentist (K-means) and Bayesian (Dirichlet Process Mixture Model) approaches, showing key differences in uncertainty handling.

```python
import numpy as np
from sklearn.cluster import KMeans
import pymc3 as pm

class ClusteringComparison:
    def __init__(self, data):
        self.data = data
        self.n_samples = len(data)
        
    def frequentist_clustering(self, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(self.data)
        centers = kmeans.cluster_centers_
        
        return {
            'labels': labels,
            'centers': centers,
            'inertia': kmeans.inertia_
        }
    
    def bayesian_clustering(self, max_clusters=10, samples=2000):
        with pm.Model() as model:
            # Concentration parameter
            alpha = pm.Gamma('alpha', 2, 2)
            
            # Stick-breaking weights
            w = pm.Dirichlet('w', 
                            a=tt.ones(max_clusters)*alpha/max_clusters,
                            shape=max_clusters)
            
            # Cluster parameters
            mu = pm.Normal('mu', 
                          mu=0,
                          sd=10,
                          shape=(max_clusters, self.data.shape[1]))
            sigma = pm.HalfNormal('sigma', 
                                sd=1,
                                shape=(max_clusters, self.data.shape[1]))
            
            # Mixture likelihood
            y = pm.Mixture('y',
                          w=w,
                          comp_dists=[pm.Normal.dist(mu=mu[i],
                                                   sd=sigma[i])
                                    for i in range(max_clusters)],
                          observed=self.data)
            
            # Inference
            self.trace = pm.sample(samples,
                                 tune=1000,
                                 return_inferencedata=False)
        
        return self.trace

# Example usage
np.random.seed(42)
n_points = 300
data = np.vstack([
    np.random.normal(0, 1, (100, 2)),
    np.random.normal(5, 1, (100, 2)),
    np.random.normal([2, 5], 1, (100, 2))
])

clustering = ClusteringComparison(data)
freq_results = clustering.frequentist_clustering()
bayes_results = clustering.bayesian_clustering()

print("Frequentist Results:")
print(f"Inertia: {freq_results['inertia']:.2f}")
print("Cluster centers:")
print(freq_results['centers'])

print("\nBayesian Results:")
print(f"Estimated number of active clusters: "
      f"{(bayes_results['w'].mean(axis=0) > 0.05).sum()}")
```

Slide 14: Additional Resources

*   "Probabilistic Programming and Bayesian Methods for Hackers" - [https://arxiv.org/abs/2002.06700](https://arxiv.org/abs/2002.06700)
*   "A Tutorial on Bayesian Optimization" - [https://arxiv.org/abs/1807.02811](https://arxiv.org/abs/1807.02811)
*   "The Bayesian Choice: From Decision-Theoretic Foundations to Computational Implementation" - Search on Google Scholar
*   Suggested search terms for implementation details:
    *   "Practical Bayesian Optimization of Machine Learning Algorithms"
    *   "Probabilistic Programming in Python using PyMC3"
    *   "Frequentist vs Bayesian Analysis in Python"
*   Recommended frameworks and tools:
    *   PyMC3: [https://docs.pymc.io/](https://docs.pymc.io/)
    *   Stan: [https://mc-stan.org/](https://mc-stan.org/)
    *   scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)

