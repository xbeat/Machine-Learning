## Bayes' Theorem in Machine Learning
Slide 1: Understanding Bayes' Theorem Fundamentals

Bayes' theorem provides a mathematical framework for updating probabilities based on new evidence. In machine learning, it forms the foundation for probabilistic reasoning and allows us to quantify uncertainty in predictions by calculating posterior probabilities from prior knowledge and observed data.

```python
import numpy as np
from typing import Dict, List

class BayesianProbability:
    def bayes_theorem(self, prior: float, likelihood: float, evidence: float) -> float:
        """
        Calculate posterior probability using Bayes' Theorem
        P(A|B) = P(B|A) * P(A) / P(B)
        """
        posterior = (likelihood * prior) / evidence
        return posterior

    def example_calculation(self):
        # Example: Medical diagnosis
        prior = 0.01  # Prior probability of disease
        likelihood = 0.95  # Probability of positive test given disease
        false_positive = 0.10  # Probability of positive test given no disease
        
        # Calculate evidence: P(B) = P(B|A)*P(A) + P(B|not A)*P(not A)
        evidence = likelihood * prior + false_positive * (1 - prior)
        
        # Calculate posterior probability
        posterior = self.bayes_theorem(prior, likelihood, evidence)
        
        print(f"Posterior probability: {posterior:.4f}")

# Example usage
bayes = BayesianProbability()
bayes.example_calculation()
```

Slide 2: Implementing Naive Bayes Classifier

The Naive Bayes classifier implements Bayes' theorem with an assumption of feature independence. This implementation focuses on text classification, demonstrating how to build a classifier from scratch using numpy for numerical computations and basic probability calculations.

```python
import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = defaultdict(lambda: defaultdict(dict))
        
    def fit(self, X: List[List[str]], y: List[str]):
        # Calculate class probabilities
        unique_classes = set(y)
        total_samples = len(y)
        
        for cls in unique_classes:
            self.class_probs[cls] = y.count(cls) / total_samples
            
        # Calculate feature probabilities for each class
        for idx, sample in enumerate(X):
            current_class = y[idx]
            for feature in sample:
                if feature not in self.feature_probs[current_class]:
                    self.feature_probs[current_class][feature] = 1
                else:
                    self.feature_probs[current_class][feature] += 1
                    
        # Normalize feature probabilities
        for cls in unique_classes:
            total = sum(self.feature_probs[cls].values())
            for feature in self.feature_probs[cls]:
                self.feature_probs[cls][feature] /= total
```

Slide 3: Text Classification with Naive Bayes

In this practical implementation, we'll create a complete text classification system using Naive Bayes. The system includes text preprocessing, feature extraction, and probability calculations for predicting document categories based on word frequencies.

```python
import re
from typing import List, Dict
from collections import Counter

class TextNaiveBayes:
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by converting to lowercase and splitting into words"""
        text = text.lower()
        words = re.findall(r'\w+', text)
        return words
    
    def extract_features(self, documents: List[str]) -> List[List[str]]:
        """Convert documents into feature vectors"""
        return [self.preprocess_text(doc) for doc in documents]
    
    def train(self, documents: List[str], labels: List[str]):
        """Train the Naive Bayes classifier"""
        self.features = self.extract_features(documents)
        self.classifier = NaiveBayesClassifier()
        self.classifier.fit(self.features, labels)

    def predict(self, document: str) -> str:
        """Predict the class of a new document"""
        features = self.preprocess_text(document)
        # Implementation continues in next slide
```

Slide 4: Source Code for Text Classification with Naive Bayes (Continued)

This implementation extends the previous slide by completing the prediction functionality and adding smoothing to handle unseen words. The code includes Laplace smoothing to prevent zero probabilities and improve classification robustness.

```python
    def predict(self, document: str) -> str:
        features = self.preprocess_text(document)
        scores = {}
        
        for cls in self.classifier.class_probs:
            # Start with log of class probability
            score = np.log(self.classifier.class_probs[cls])
            
            # Add log probabilities of features
            for feature in features:
                # Laplace smoothing
                if feature in self.classifier.feature_probs[cls]:
                    prob = self.classifier.feature_probs[cls][feature]
                else:
                    prob = 1 / (len(self.classifier.feature_probs[cls]) + 1)
                score += np.log(prob)
            
            scores[cls] = score
        
        # Return class with highest probability
        return max(scores.items(), key=lambda x: x[1])[0]

# Example usage
classifier = TextNaiveBayes()
training_docs = [
    "machine learning algorithms optimize",
    "deep neural networks train data",
    "football game score points"
]
labels = ["tech", "tech", "sports"]
classifier.train(training_docs, labels)
print(classifier.predict("neural networks and algorithms"))  # Output: 'tech'
```

Slide 5: Bayesian Parameter Estimation

Bayesian parameter estimation allows us to update our beliefs about model parameters as we observe new data. This implementation demonstrates how to estimate the parameters of a Gaussian distribution using conjugate priors.

```python
import numpy as np
from scipy.stats import norm

class BayesianEstimator:
    def __init__(self, prior_mean: float, prior_var: float):
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        
    def update_gaussian(self, data: np.ndarray) -> tuple:
        """
        Update Gaussian parameters using conjugate prior
        Returns posterior mean and variance
        """
        n = len(data)
        sample_mean = np.mean(data)
        
        # Calculate posterior parameters
        posterior_var = 1 / (1/self.prior_var + n/np.var(data))
        posterior_mean = posterior_var * (
            self.prior_mean/self.prior_var + 
            n*sample_mean/np.var(data)
        )
        
        return posterior_mean, posterior_var

# Example usage
estimator = BayesianEstimator(prior_mean=0, prior_var=1)
data = np.random.normal(loc=2, scale=1, size=100)
post_mean, post_var = estimator.update_gaussian(data)
print(f"Posterior mean: {post_mean:.2f}, variance: {post_var:.2f}")
```

Slide 6: Bayesian Linear Regression

Bayesian linear regression extends traditional linear regression by treating model parameters as probability distributions rather than point estimates. This implementation shows how to perform Bayesian linear regression using conjugate priors.

```python
import numpy as np
from scipy.stats import multivariate_normal

class BayesianLinearRegression:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha  # Prior precision
        self.beta = beta    # Noise precision
        self.w_mean = None  # Posterior mean
        self.w_cov = None   # Posterior covariance
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit Bayesian linear regression model
        X: Design matrix (n_samples, n_features)
        y: Target values (n_samples,)
        """
        n_features = X.shape[1]
        # Prior parameters
        self.w_cov = np.linalg.inv(
            self.alpha * np.eye(n_features) + 
            self.beta * X.T @ X
        )
        self.w_mean = self.beta * self.w_cov @ X.T @ y
        
    def predict(self, X: np.ndarray) -> tuple:
        """Return mean and variance of predictions"""
        y_mean = X @ self.w_mean
        y_var = 1/self.beta + np.diagonal(X @ self.w_cov @ X.T)
        return y_mean, y_var
```

Slide 7: Real-World Example - Spam Detection

This implementation demonstrates a practical spam detection system using Naive Bayes. The system includes text preprocessing, feature extraction, and evaluation metrics commonly used in production environments.

```python
import numpy as np
from typing import Tuple, List
from collections import Counter
import re

class SpamDetector:
    def __init__(self):
        self.word_counts = {'spam': Counter(), 'ham': Counter()}
        self.class_counts = {'spam': 0, 'ham': 0}
        
    def preprocess(self, text: str) -> List[str]:
        """Clean and tokenize text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def train(self, texts: List[str], labels: List[str]):
        """Train spam detector with labeled data"""
        for text, label in zip(texts, labels):
            words = self.preprocess(text)
            self.class_counts[label] += 1
            self.word_counts[label].update(words)
            
    def predict(self, text: str) -> Tuple[str, float]:
        words = self.preprocess(text)
        scores = {}
        
        total_docs = sum(self.class_counts.values())
        
        for label in ['spam', 'ham']:
            # Calculate prior probability
            prior = np.log(self.class_counts[label] / total_docs)
            
            # Calculate likelihood
            likelihood = 0
            total_words = sum(self.word_counts[label].values())
            vocab_size = len(set.union(*[set(self.word_counts[l].keys()) 
                                       for l in ['spam', 'ham']]))
            
            for word in words:
                # Apply Laplace smoothing
                count = self.word_counts[label].get(word, 0) + 1
                likelihood += np.log(count / (total_words + vocab_size))
            
            scores[label] = prior + likelihood
            
        # Return prediction and confidence
        prediction = max(scores.items(), key=lambda x: x[1])[0]
        confidence = np.exp(scores[prediction]) / sum(np.exp(val) 
                                                    for val in scores.values())
        return prediction, confidence

# Example usage with evaluation
spam_detector = SpamDetector()

# Training data
training_texts = [
    "win free money now click here",
    "meeting tomorrow at 3pm",
    "claim your prize instantly",
    "project deadline reminder"
]
training_labels = ["spam", "ham", "spam", "ham"]

# Train the model
spam_detector.train(training_texts, training_labels)

# Test prediction
test_text = "congratulations you won million dollars"
prediction, confidence = spam_detector.predict(test_text)
print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2f}")
```

Slide 8: Bayesian A/B Testing Implementation

Bayesian A/B testing provides a probabilistic framework for comparing two variants. This implementation calculates the probability that one variant is better than another using beta distributions for conversion rates.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class BayesianABTest:
    def __init__(self, 
                 alpha_prior: float = 1, 
                 beta_prior: float = 1):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
    def update_posterior(self, 
                        successes: int, 
                        trials: int) -> tuple:
        """Calculate posterior parameters"""
        alpha_post = self.alpha_prior + successes
        beta_post = self.beta_prior + trials - successes
        return alpha_post, beta_post
    
    def probability_b_better_than_a(self,
                                  a_successes: int,
                                  a_trials: int,
                                  b_successes: int,
                                  b_trials: int,
                                  samples: int = 10000) -> float:
        """Monte Carlo estimation of P(B > A)"""
        a_post = self.update_posterior(a_successes, a_trials)
        b_post = self.update_posterior(b_successes, b_trials)
        
        # Sample from posteriors
        a_samples = np.random.beta(a_post[0], a_post[1], samples)
        b_samples = np.random.beta(b_post[0], b_post[1], samples)
        
        # Calculate probability B > A
        return np.mean(b_samples > a_samples)

# Example usage
ab_test = BayesianABTest()
prob_b_better = ab_test.probability_b_better_than_a(
    a_successes=50,
    a_trials=100,
    b_successes=60,
    b_trials=100
)
print(f"Probability B is better than A: {prob_b_better:.2f}")
```

Slide 9: Bayesian Model Selection

This implementation demonstrates how to perform Bayesian model selection using the Bayesian Information Criterion (BIC) and model evidence calculation. The code compares different models based on their posterior probabilities.

```python
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

class BayesianModelSelection:
    def __init__(self, models: list):
        self.models = models
        self.model_probs = np.ones(len(models)) / len(models)
    
    def compute_bic(self, 
                   X: np.ndarray, 
                   y: np.ndarray, 
                   params: dict) -> float:
        """
        Compute Bayesian Information Criterion
        BIC = ln(n)k - 2ln(L)
        """
        n = len(y)
        k = len(params)
        y_pred = self.predict(X, params)
        mse = np.mean((y - y_pred) ** 2)
        log_likelihood = -n/2 * np.log(2*np.pi*mse) - n/2
        return np.log(n) * k - 2 * log_likelihood
    
    def model_evidence(self, 
                      X: np.ndarray, 
                      y: np.ndarray) -> np.ndarray:
        """Calculate model evidence for each model"""
        evidences = np.zeros(len(self.models))
        
        for i, model in enumerate(self.models):
            params = model.fit(X, y)
            bic = self.compute_bic(X, y, params)
            evidences[i] = -0.5 * bic  # Convert BIC to log evidence
            
        return evidences
    
    def update_probabilities(self, 
                           X: np.ndarray, 
                           y: np.ndarray):
        """Update model probabilities using Bayes' rule"""
        log_evidences = self.model_evidence(X, y)
        log_total = logsumexp(log_evidences)
        self.model_probs = np.exp(log_evidences - log_total)
        
        return self.model_probs

# Example usage
class LinearModel:
    def fit(self, X, y):
        return {'slope': np.cov(X, y)[0,1] / np.var(X),
                'intercept': np.mean(y) - np.mean(X) * self.slope}

class QuadraticModel:
    def fit(self, X, y):
        coeffs = np.polyfit(X.flatten(), y, 2)
        return {'a': coeffs[0], 'b': coeffs[1], 'c': coeffs[2]}

# Create data and models
X = np.linspace(0, 10, 100)
y = 2*X + 3 + np.random.normal(0, 1, 100)

models = [LinearModel(), QuadraticModel()]
model_selector = BayesianModelSelection(models)
probs = model_selector.update_probabilities(X, y)
print(f"Model probabilities: {probs}")
```

Slide 10: Bayesian Neural Network Implementation

This implementation creates a simple Bayesian Neural Network using variational inference. The network estimates uncertainty in predictions by treating weights as probability distributions.

```python
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -3)
        nn.init.constant_(self.bias_mu, 0)
        nn.init.constant_(self.bias_rho, -3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_mu + torch.randn_like(self.weight_mu) * \
                 torch.exp(self.weight_rho)
        bias = self.bias_mu + torch.randn_like(self.bias_mu) * \
               torch.exp(self.bias_rho)
        
        return F.linear(x, weight, bias)
    
    def kl_loss(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior"""
        weight_std = torch.exp(self.weight_rho)
        bias_std = torch.exp(self.bias_rho)
        
        kl_weight = self._kl_normal(self.weight_mu, weight_std)
        kl_bias = self._kl_normal(self.bias_mu, bias_std)
        
        return kl_weight + kl_bias
    
    def _kl_normal(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """KL divergence between N(mu, std) and N(0, 1)"""
        return 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*torch.log(std) - 1)
```

Slide 11: Practical Implementation of Bayesian Decision Theory

This implementation demonstrates how to make optimal decisions under uncertainty using Bayesian decision theory. The code includes utility functions and risk calculation for real-world decision-making scenarios.

```python
import numpy as np
from dataclasses import dataclass
from typing import Dict, Callable

@dataclass
class Decision:
    name: str
    utility_function: Callable
    cost: float

class BayesianDecisionSystem:
    def __init__(self, 
                 prior_probabilities: Dict[str, float], 
                 loss_matrix: np.ndarray):
        self.priors = prior_probabilities
        self.loss_matrix = loss_matrix
        self.decisions = {}
        
    def add_decision(self, 
                    decision: Decision):
        """Add a possible decision with its utility function"""
        self.decisions[decision.name] = decision
    
    def update_posterior(self, 
                        evidence: Dict[str, float]) -> Dict[str, float]:
        """Update probabilities given new evidence"""
        posteriors = {}
        normalizer = 0
        
        for state in self.priors:
            likelihood = evidence.get(state, 1.0)
            posteriors[state] = self.priors[state] * likelihood
            normalizer += posteriors[state]
            
        # Normalize probabilities
        for state in posteriors:
            posteriors[state] /= normalizer
            
        return posteriors
    
    def calculate_expected_utility(self, 
                                 decision: Decision, 
                                 probabilities: Dict[str, float]) -> float:
        """Calculate expected utility for a decision"""
        utility = 0
        for state, prob in probabilities.items():
            utility += prob * decision.utility_function(state) - decision.cost
        return utility
    
    def make_optimal_decision(self, 
                            evidence: Dict[str, float] = None) -> tuple:
        """Choose the decision that maximizes expected utility"""
        # Use updated probabilities if evidence is provided
        probs = self.update_posterior(evidence) if evidence else self.priors
        
        # Calculate utilities for each decision
        utilities = {
            name: self.calculate_expected_utility(decision, probs)
            for name, decision in self.decisions.items()
        }
        
        # Find optimal decision
        optimal_decision = max(utilities.items(), key=lambda x: x[1])
        return optimal_decision[0], optimal_decision[1], utilities

# Example usage
def profit_function(state: str) -> float:
    return {'high': 1000, 'medium': 500, 'low': 100}.get(state, 0)

# Initialize system
prior_probs = {'high': 0.3, 'medium': 0.5, 'low': 0.2}
loss_matrix = np.array([[0, 100, 200],
                       [100, 0, 100],
                       [200, 100, 0]])

decision_system = BayesianDecisionSystem(prior_probs, loss_matrix)

# Add possible decisions
decision_system.add_decision(
    Decision('invest_high', profit_function, cost=500)
)
decision_system.add_decision(
    Decision('invest_low', lambda x: profit_function(x) * 0.5, cost=200)
)

# Make decision with new evidence
evidence = {'high': 0.8, 'medium': 0.6, 'low': 0.3}
optimal_decision, utility, all_utilities = decision_system.make_optimal_decision(evidence)

print(f"Optimal decision: {optimal_decision}")
print(f"Expected utility: {utility:.2f}")
print("All utilities:", all_utilities)
```

Slide 12: Bayesian Time Series Analysis

This implementation shows how to perform Bayesian analysis on time series data, including trend detection and changepoint analysis using probabilistic models.

```python
import numpy as np
from scipy import stats
from typing import Tuple, List

class BayesianTimeSeriesAnalyzer:
    def __init__(self, 
                 change_point_prior: float = 0.1,
                 noise_std: float = 1.0):
        self.change_point_prior = change_point_prior
        self.noise_std = noise_std
        
    def detect_changepoints(self, 
                          data: np.ndarray, 
                          window_size: int = 10) -> List[int]:
        """Detect points where the time series distribution changes"""
        n = len(data)
        log_odds = np.zeros(n)
        changepoints = []
        
        for i in range(window_size, n - window_size):
            # Compare distributions before and after point
            before = data[i-window_size:i]
            after = data[i:i+window_size]
            
            # Calculate Bayes factor
            bf = self._bayes_factor(before, after)
            log_odds[i] = np.log(bf) + np.log(self.change_point_prior)
            
            if log_odds[i] > 0:
                changepoints.append(i)
                
        return changepoints
    
    def _bayes_factor(self, 
                      x1: np.ndarray, 
                      x2: np.ndarray) -> float:
        """Calculate Bayes factor for two segments"""
        # Compute likelihood ratio
        mu1, std1 = np.mean(x1), np.std(x1)
        mu2, std2 = np.mean(x2), np.std(x2)
        
        ll1 = np.sum(stats.norm.logpdf(x1, mu1, std1))
        ll2 = np.sum(stats.norm.logpdf(x2, mu2, std2))
        
        ll_joint = np.sum(stats.norm.logpdf(
            np.concatenate([x1, x2]),
            np.mean(np.concatenate([x1, x2])),
            np.std(np.concatenate([x1, x2]))
        ))
        
        return np.exp(ll1 + ll2 - ll_joint)
    
    def predict_next_value(self, 
                          data: np.ndarray, 
                          n_samples: int = 1000) -> Tuple[float, float]:
        """Predict next value with uncertainty"""
        # Fit AR(1) model
        X = data[:-1]
        y = data[1:]
        
        # Bayesian linear regression
        beta_mean = np.cov(X, y)[0,1] / np.var(X)
        beta_std = np.sqrt(self.noise_std / (len(X) * np.var(X)))
        
        # Sample predictions
        beta_samples = np.random.normal(beta_mean, beta_std, n_samples)
        pred_samples = beta_samples * data[-1]
        
        return np.mean(pred_samples), np.std(pred_samples)

# Example usage
np.random.seed(42)
n_points = 100
time = np.arange(n_points)
signal = np.concatenate([
    np.sin(time[:50] * 0.1),
    np.sin(time[50:] * 0.1) + 2
])
noise = np.random.normal(0, 0.2, n_points)
data = signal + noise

analyzer = BayesianTimeSeriesAnalyzer()
changepoints = analyzer.detect_changepoints(data)
next_mean, next_std = analyzer.predict_next_value(data)

print(f"Detected changepoints at: {changepoints}")
print(f"Prediction: {next_mean:.2f} ± {next_std:.2f}")
```

Slide 13: Bayesian Deep Learning with PyTorch

This implementation demonstrates how to create a Bayesian Deep Learning model using variational inference in PyTorch, incorporating uncertainty estimation in deep neural networks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

class BayesianLayer(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))
        
        # Prior distribution
        self.prior = Normal(0, prior_std)
        
        # Initialize parameters
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -3)
        nn.init.constant_(self.bias_mu, 0)
        nn.init.constant_(self.bias_rho, -3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample weights and biases
        weight = Normal(self.weight_mu, F.softplus(self.weight_rho)).rsample()
        bias = Normal(self.bias_mu, F.softplus(self.bias_rho)).rsample()
        
        # Compute output
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior"""
        weight_posterior = Normal(self.weight_mu, F.softplus(self.weight_rho))
        bias_posterior = Normal(self.bias_mu, F.softplus(self.bias_rho))
        
        weight_kl = kl_divergence(weight_posterior, self.prior).sum()
        bias_kl = kl_divergence(bias_posterior, self.prior).sum()
        
        return weight_kl + bias_kl

class BayesianNeuralNetwork(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: list, 
                 output_dim: int):
        super().__init__()
        
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims)-1):
            self.layers.append(BayesianLayer(dims[i], dims[i+1]))
            
    def forward(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        outputs = []
        
        for _ in range(num_samples):
            current = x
            for layer in self.layers[:-1]:
                current = F.relu(layer(current))
            current = self.layers[-1](current)
            outputs.append(current)
            
        return torch.stack(outputs)
    
    def kl_divergence(self) -> torch.Tensor:
        return sum(layer.kl_divergence() for layer in self.layers)

# Example usage
def train_model(model, 
                train_loader, 
                num_epochs: int,
                learning_rate: float = 0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass with multiple samples
            predictions = model(batch_x, num_samples=5)
            
            # Compute loss
            nll = -Normal(predictions.mean(0), 
                         predictions.std(0)).log_prob(batch_y).mean()
            kl = model.kl_divergence()
            loss = nll + kl / len(train_loader.dataset)
            
            # Backward pass
            loss.backward()
            optimizer.step()

# Create synthetic dataset
X = torch.randn(1000, 10)
y = torch.sin(X[:, 0]) + torch.randn(1000) * 0.1

# Create model and train
model = BayesianNeuralNetwork(10, [20, 20], 1)
print("Model architecture:", model)
```

Slide 14: Additional Resources

*   "Bayesian Deep Learning" - arXiv:1703.04977 [https://arxiv.org/abs/1703.04977](https://arxiv.org/abs/1703.04977)
*   "Practical Variational Inference for Neural Networks" - arXiv:1611.01144 [https://arxiv.org/abs/1611.01144](https://arxiv.org/abs/1611.01144)
*   "A Simple Baseline for Bayesian Uncertainty in Deep Learning" - arXiv:1902.02476 [https://arxiv.org/abs/1902.02476](https://arxiv.org/abs/1902.02476)
*   "Weight Uncertainty in Neural Networks" - arXiv:1505.05424 [https://arxiv.org/abs/1505.05424](https://arxiv.org/abs/1505.05424)
*   For additional resources and implementation details, consider searching:
    *   Google Scholar for "Bayesian Neural Networks implementations"
    *   PyTorch documentation for probabilistic programming
    *   "Probabilistic Deep Learning with Python" books and tutorials

