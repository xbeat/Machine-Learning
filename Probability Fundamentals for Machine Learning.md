## Probability Fundamentals for Machine Learning
Slide 1: Understanding Conditional Probability

Conditional probability forms the foundation of many machine learning algorithms, particularly in classification tasks. It represents the probability of an event occurring given that another event has already occurred, mathematically expressed as P(A|B).

```python
import numpy as np

# Generate sample email data
def generate_spam_data(n_samples=1000):
    # Simulate words appearing in emails
    contains_money = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    is_spam = np.zeros(n_samples)
    
    # Create relationship: 80% of emails with 'money' are spam
    for i in range(n_samples):
        if contains_money[i]:
            is_spam[i] = np.random.choice([0, 1], p=[0.2, 0.8])
        else:
            is_spam[i] = np.random.choice([0, 1], p=[0.9, 0.1])
    
    return contains_money, is_spam

# Calculate conditional probability
def calculate_conditional_prob(contains_money, is_spam):
    total_money = np.sum(contains_money)
    spam_given_money = np.sum((contains_money == 1) & (is_spam == 1)) / total_money
    return spam_given_money

# Example usage
money, spam = generate_spam_data()
p_spam_given_money = calculate_conditional_prob(money, spam)
print(f"P(Spam|Contains 'money') = {p_spam_given_money:.2f}")
```

Slide 2: Bayes' Theorem Implementation

Bayes' theorem allows us to update probabilities as new evidence becomes available. This implementation demonstrates how to calculate the probability of an email being spam given specific word occurrences.

```python
class BayesianSpamFilter:
    def __init__(self):
        self.word_spam_count = {}
        self.word_ham_count = {}
        self.spam_count = 0
        self.ham_count = 0
    
    def train(self, text, is_spam):
        words = text.lower().split()
        if is_spam:
            self.spam_count += 1
            for word in set(words):
                self.word_spam_count[word] = self.word_spam_count.get(word, 0) + 1
        else:
            self.ham_count += 1
            for word in set(words):
                self.word_ham_count[word] = self.word_ham_count.get(word, 0) + 1
    
    def calculate_probability(self, text):
        words = text.lower().split()
        p_spam = self.spam_count / (self.spam_count + self.ham_count)
        p_ham = 1 - p_spam
        
        p_word_given_spam = 1
        p_word_given_ham = 1
        
        for word in words:
            if word in self.word_spam_count:
                p_word_given_spam *= (self.word_spam_count[word] / self.spam_count)
            if word in self.word_ham_count:
                p_word_given_ham *= (self.word_ham_count[word] / self.ham_count)
        
        numerator = p_word_given_spam * p_spam
        denominator = numerator + (p_word_given_ham * p_ham)
        
        return numerator / denominator if denominator != 0 else 0

# Example usage
spam_filter = BayesianSpamFilter()
spam_filter.train("money transfer urgent", True)
spam_filter.train("meeting tomorrow morning", False)
print(f"Probability of spam: {spam_filter.calculate_probability('urgent money'):.2f}")
```

Slide 3: The Addition Rule of Probability

The addition rule calculates the probability of either event A or event B occurring. For independent events, it's P(A ∪ B) = P(A) + P(B) - P(A ∩ B). This implementation demonstrates the concept using dice rolls.

```python
import numpy as np

class ProbabilityAdditionRule:
    def __init__(self, trials=10000):
        self.trials = trials
        
    def simulate_dice_rolls(self):
        rolls = np.random.randint(1, 7, size=self.trials)
        
        # Event A: Rolling an even number
        event_a = np.sum(rolls % 2 == 0)
        
        # Event B: Rolling a number greater than 4
        event_b = np.sum(rolls > 4)
        
        # Intersection: Rolling an even number greater than 4
        intersection = np.sum((rolls % 2 == 0) & (rolls > 4))
        
        # Calculate probabilities
        p_a = event_a / self.trials
        p_b = event_b / self.trials
        p_intersection = intersection / self.trials
        p_union = p_a + p_b - p_intersection
        
        return {
            'P(A)': p_a,
            'P(B)': p_b,
            'P(A∩B)': p_intersection,
            'P(A∪B)': p_union
        }

# Example usage
simulator = ProbabilityAdditionRule()
results = simulator.simulate_dice_rolls()
for key, value in results.items():
    print(f"{key} = {value:.3f}")
```

Slide 4: The Multiplication Rule in Practice

The multiplication rule calculates the probability of two events occurring together. This implementation demonstrates calculating probabilities of drawing specific cards consecutively from a deck, incorporating conditional probability.

```python
import random

class DeckProbability:
    def __init__(self, trials=100000):
        self.trials = trials
        self.suits = ['hearts', 'diamonds', 'clubs', 'spades']
        self.ranks = list(range(2, 11)) + ['J', 'Q', 'K', 'A']
        
    def create_deck(self):
        return [(rank, suit) for suit in self.suits for rank in self.ranks]
    
    def simulate_consecutive_draws(self, target_rank1, target_rank2):
        successful_draws = 0
        
        for _ in range(self.trials):
            deck = self.create_deck()
            random.shuffle(deck)
            
            # Draw first card
            first_card = deck.pop()
            if first_card[0] == target_rank1:
                # Draw second card
                second_card = deck.pop()
                if second_card[0] == target_rank2:
                    successful_draws += 1
        
        probability = successful_draws / self.trials
        return probability

# Example usage
simulator = DeckProbability()
prob_aces = simulator.simulate_consecutive_draws('A', 'A')
print(f"Probability of drawing two aces consecutively: {prob_aces:.4f}")
```

Slide 5: Probability Distributions in Machine Learning

Understanding probability distributions is crucial for machine learning models. This implementation creates a custom normal distribution class that can generate samples and calculate probabilities, essential for many ML algorithms.

```python
import numpy as np
import matplotlib.pyplot as plt

class NormalDistribution:
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
    
    def pdf(self, x):
        """Probability density function"""
        coefficient = 1 / (self.sigma * np.sqrt(2 * np.pi))
        exponent = -((x - self.mu) ** 2) / (2 * self.sigma ** 2)
        return coefficient * np.exp(exponent)
    
    def sample(self, size=1000):
        """Generate samples from the distribution"""
        return np.random.normal(self.mu, self.sigma, size)
    
    def plot_distribution(self, samples=None):
        if samples is None:
            samples = self.sample()
            
        x = np.linspace(min(samples), max(samples), 100)
        y = self.pdf(x)
        
        plt.figure(figsize=(10, 6))
        plt.hist(samples, bins=30, density=True, alpha=0.7, label='Samples')
        plt.plot(x, y, 'r-', label='PDF')
        plt.title(f'Normal Distribution (μ={self.mu}, σ={self.sigma})')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        return plt

# Example usage
dist = NormalDistribution(mu=0, sigma=2)
samples = dist.sample(10000)
print(f"Sample mean: {np.mean(samples):.2f}")
print(f"Sample std: {np.std(samples):.2f}")
```

Slide 6: Maximum Likelihood Estimation

Maximum Likelihood Estimation is a fundamental method for parameter estimation in probabilistic models. This implementation demonstrates MLE for fitting a normal distribution to observed data.

```python
import numpy as np
from scipy.optimize import minimize

class MaximumLikelihoodEstimator:
    def __init__(self, data):
        self.data = np.array(data)
    
    def negative_log_likelihood(self, params):
        """Calculate negative log-likelihood for normal distribution"""
        mu, sigma = params
        n = len(self.data)
        log_likelihood = -n * np.log(sigma * np.sqrt(2 * np.pi)) - \
                        np.sum((self.data - mu)**2) / (2 * sigma**2)
        return -log_likelihood
    
    def fit(self):
        """Find MLE estimates for mu and sigma"""
        initial_guess = [np.mean(self.data), np.std(self.data)]
        result = minimize(self.negative_log_likelihood, 
                        initial_guess, 
                        method='Nelder-Mead')
        return result.x

# Example usage
np.random.seed(42)
true_mu, true_sigma = 2, 1.5
data = np.random.normal(true_mu, true_sigma, 1000)

mle = MaximumLikelihoodEstimator(data)
estimated_mu, estimated_sigma = mle.fit()

print(f"True parameters: μ={true_mu}, σ={true_sigma}")
print(f"MLE estimates: μ={estimated_mu:.2f}, σ={estimated_sigma:.2f}")
```

Slide 7: Bayesian Networks for Email Classification

Implementing a simple Bayesian Network for email classification demonstrates how probability theory underlies machine learning algorithms. This implementation shows how multiple features can be combined using conditional probabilities.

```python
import numpy as np
from collections import defaultdict

class BayesianNetwork:
    def __init__(self):
        self.feature_probs = defaultdict(lambda: defaultdict(float))
        self.class_probs = defaultdict(float)
        self.features = set()
        
    def train(self, X, y):
        """Train the Bayesian Network with features X and labels y"""
        n_samples = len(y)
        
        # Calculate class probabilities
        unique_classes, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            self.class_probs[cls] = count / n_samples
        
        # Calculate conditional probabilities for each feature
        for feature_idx in range(X.shape[1]):
            self.features.add(feature_idx)
            for cls in unique_classes:
                class_mask = (y == cls)
                feature_given_class = np.sum(X[class_mask, feature_idx]) / np.sum(class_mask)
                self.feature_probs[feature_idx][cls] = feature_given_class
    
    def predict_proba(self, x):
        """Calculate probability distribution over classes"""
        probs = {}
        for cls in self.class_probs:
            prob = self.class_probs[cls]
            for feature_idx in self.features:
                if x[feature_idx]:
                    prob *= self.feature_probs[feature_idx][cls]
                else:
                    prob *= (1 - self.feature_probs[feature_idx][cls])
            probs[cls] = prob
        
        # Normalize probabilities
        total = sum(probs.values())
        return {cls: p/total for cls, p in probs.items()}

# Example usage with email features
features = np.array([
    [1, 0, 1],  # contains_money, urgent, poor_grammar
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 1]
])
labels = np.array([1, 0, 1, 0])  # 1: spam, 0: ham

bn = BayesianNetwork()
bn.train(features, labels)

# Predict for new email
new_email = np.array([1, 1, 0])
probs = bn.predict_proba(new_email)
print(f"Probability of spam: {probs[1]:.3f}")
print(f"Probability of ham: {probs[0]:.3f}")
```

Slide 8: Monte Carlo Methods for Probability Estimation

Monte Carlo methods provide powerful tools for estimating probabilities through simulation. This implementation demonstrates how to estimate complex probabilities using random sampling techniques.

```python
import numpy as np
from typing import Callable

class MonteCarloEstimator:
    def __init__(self, n_samples: int = 100000):
        self.n_samples = n_samples
    
    def estimate_probability(self, event_function: Callable[[np.ndarray], np.ndarray],
                           sample_space: tuple) -> float:
        """
        Estimate probability using Monte Carlo simulation
        
        Parameters:
        - event_function: Function that returns True for points satisfying the event
        - sample_space: Tuple of (min, max) for each dimension
        """
        # Generate random samples
        dims = len(sample_space)
        samples = np.random.uniform(
            low=[s[0] for s in sample_space],
            high=[s[1] for s in sample_space],
            size=(self.n_samples, dims)
        )
        
        # Calculate probability
        successful_events = event_function(samples)
        probability = np.mean(successful_events)
        
        # Calculate standard error
        std_error = np.sqrt(probability * (1 - probability) / self.n_samples)
        
        return probability, std_error

# Example: Estimate probability of point lying inside a unit circle
def inside_circle(points):
    return np.sum(points**2, axis=1) <= 1

# Define sample space for unit square
sample_space = [(-1, 1), (-1, 1)]

# Estimate pi/4 (area of quarter circle)
mc = MonteCarloEstimator(n_samples=1000000)
prob, error = mc.estimate_probability(inside_circle, sample_space)

print(f"Estimated π/4: {prob:.6f} ± {error:.6f}")
print(f"Actual π/4: {np.pi/4:.6f}")
```

Slide 9: Probabilistic Feature Selection

Feature selection using probabilistic measures helps identify the most informative features for machine learning models. This implementation uses mutual information to rank features by their predictive power.

```python
import numpy as np
from scipy.stats import entropy

class ProbabilisticFeatureSelector:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        
    def mutual_information(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate mutual information between features and target"""
        n_features = X.shape[1]
        mi_scores = np.zeros(n_features)
        
        for i in range(n_features):
            # Discretize continuous feature
            x_bins = np.histogram_bin_edges(X[:, i], bins=self.n_bins)
            x_discrete = np.digitize(X[:, i], x_bins) - 1
            
            # Calculate mutual information
            mi_scores[i] = self._mutual_information_score(x_discrete, y)
            
        return mi_scores
    
    def _mutual_information_score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information between two variables"""
        # Joint probability
        xy_hist = np.histogram2d(x, y, bins=[self.n_bins, len(np.unique(y))])[0]
        xy_prob = xy_hist / xy_hist.sum()
        
        # Marginal probabilities
        x_prob = xy_prob.sum(axis=1)
        y_prob = xy_prob.sum(axis=0)
        
        # Calculate mutual information
        mi = 0
        for i in range(len(x_prob)):
            for j in range(len(y_prob)):
                if xy_prob[i,j] > 0:
                    mi += xy_prob[i,j] * np.log2(xy_prob[i,j] / (x_prob[i] * y_prob[j]))
                    
        return mi

# Example usage
np.random.seed(42)
X = np.random.randn(1000, 5)  # 5 features
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)  # First two features are relevant

selector = ProbabilisticFeatureSelector()
mi_scores = selector.mutual_information(X, y)

for i, score in enumerate(mi_scores):
    print(f"Feature {i} MI Score: {score:.4f}")
```

Slide 10: Joint Probability Distributions

Joint probability distributions are essential for understanding relationships between multiple random variables in machine learning. This implementation demonstrates how to work with multivariate normal distributions.

```python
import numpy as np
from scipy.stats import multivariate_normal

class JointProbabilityDistribution:
    def __init__(self, mean, covariance):
        self.mean = np.array(mean)
        self.covariance = np.array(covariance)
        self.distribution = multivariate_normal(mean=self.mean, cov=self.covariance)
        
    def sample(self, n_samples=1000):
        """Generate samples from the joint distribution"""
        return self.distribution.rvs(size=n_samples)
    
    def pdf(self, x):
        """Calculate probability density at point x"""
        return self.distribution.pdf(x)
    
    def conditional_distribution(self, given_idx, given_value):
        """Calculate conditional distribution parameters"""
        n = len(self.mean)
        idx_rest = [i for i in range(n) if i != given_idx]
        
        # Partition mean and covariance
        mu1 = self.mean[idx_rest]
        mu2 = self.mean[given_idx]
        sigma11 = self.covariance[np.ix_(idx_rest, idx_rest)]
        sigma12 = self.covariance[np.ix_(idx_rest, [given_idx])]
        sigma21 = self.covariance[np.ix_([given_idx], idx_rest)]
        sigma22 = self.covariance[given_idx, given_idx]
        
        # Calculate conditional parameters
        cond_mean = mu1 + sigma12.dot(1/sigma22 * (given_value - mu2))
        cond_cov = sigma11 - sigma12.dot(1/sigma22).dot(sigma21)
        
        return cond_mean, cond_cov

# Example usage
mean = [0, 1]
covariance = [[1, 0.5],
              [0.5, 2]]

jpd = JointProbabilityDistribution(mean, covariance)

# Generate samples
samples = jpd.sample(1000)

# Calculate conditional distribution given X1 = 1
cond_mean, cond_cov = jpd.conditional_distribution(given_idx=0, given_value=1)

print(f"Joint Distribution Mean: {jpd.mean}")
print(f"Conditional Mean given X1=1: {cond_mean}")
print(f"Conditional Covariance: {cond_cov}")
```

Slide 11: Probability Calibration for ML Models

Probability calibration ensures that model predictions accurately reflect true probabilities. This implementation shows how to calibrate classifier probabilities using isotonic regression.

```python
import numpy as np
from sklearn.isotonic import IsotonicRegression

class ProbabilityCalibrator:
    def __init__(self, method='isotonic'):
        self.method = method
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        
    def fit(self, pred_probs, true_labels):
        """Fit calibration model"""
        self.calibrator.fit(pred_probs, true_labels)
        return self
        
    def calibrate(self, pred_probs):
        """Apply calibration to predicted probabilities"""
        return self.calibrator.predict(pred_probs)
    
    def evaluate_calibration(self, pred_probs, true_labels, n_bins=10):
        """Evaluate calibration using reliability diagram metrics"""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(pred_probs, bins) - 1
        
        bin_sums = np.bincount(bin_indices, minlength=n_bins)
        bin_true = np.bincount(bin_indices, weights=true_labels, minlength=n_bins)
        bin_pred = np.bincount(bin_indices, weights=pred_probs, minlength=n_bins)
        
        # Calculate mean predicted and true probabilities in each bin
        with np.errstate(divide='ignore', invalid='ignore'):
            bin_true_probs = np.where(bin_sums > 0, bin_true / bin_sums, 0)
            bin_pred_probs = np.where(bin_sums > 0, bin_pred / bin_sums, 0)
            
        return bins[:-1], bin_true_probs, bin_pred_probs

# Example usage
np.random.seed(42)

# Generate synthetic uncalibrated probabilities
n_samples = 1000
true_probs = np.random.beta(2, 5, n_samples)
pred_probs = true_probs + 0.2 * np.random.randn(n_samples)
pred_probs = np.clip(pred_probs, 0, 1)
true_labels = np.random.binomial(1, true_probs)

# Calibrate probabilities
calibrator = ProbabilityCalibrator()
calibrator.fit(pred_probs, true_labels)
calibrated_probs = calibrator.calibrate(pred_probs)

# Evaluate calibration
bins, true_p, pred_p = calibrator.evaluate_calibration(calibrated_probs, true_labels)
print("Calibration Evaluation:")
for i in range(len(bins)):
    print(f"Bin {i}: True prob = {true_p[i]:.3f}, Predicted prob = {pred_p[i]:.3f}")
```

Slide 12: Probabilistic Cross-Validation

This implementation demonstrates a probabilistic approach to cross-validation that accounts for uncertainty in performance estimates using Bayesian methods.

```python
import numpy as np
from scipy import stats

class ProbabilisticCrossValidator:
    def __init__(self, n_splits=5, confidence_level=0.95):
        self.n_splits = n_splits
        self.confidence_level = confidence_level
        
    def split_data(self, X, y):
        """Generate probabilistic cross-validation splits"""
        n_samples = len(y)
        fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
            
            yield train_indices, test_indices
    
    def evaluate(self, model, X, y, scoring_func):
        """Perform probabilistic cross-validation"""
        scores = []
        
        for train_idx, test_idx in self.split_data(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            score = scoring_func(y_test, model.predict_proba(X_test)[:, 1])
            scores.append(score)
            
        scores = np.array(scores)
        
        # Calculate confidence intervals using t-distribution
        mean_score = np.mean(scores)
        std_error = stats.sem(scores)
        ci = stats.t.interval(self.confidence_level, 
                            len(scores)-1, 
                            loc=mean_score, 
                            scale=std_error)
        
        return {
            'mean_score': mean_score,
            'std_error': std_error,
            'confidence_interval': ci,
            'individual_scores': scores
        }

# Example usage with a simple scoring function
def log_loss(y_true, y_pred):
    """Simplified log loss implementation"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Create synthetic data and model
np.random.seed(42)
X = np.random.randn(1000, 5)
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Perform probabilistic cross-validation
pcv = ProbabilisticCrossValidator()
results = pcv.evaluate(model, X, y, log_loss)

print(f"Mean Score: {results['mean_score']:.4f}")
print(f"Standard Error: {results['std_error']:.4f}")
print(f"95% Confidence Interval: ({results['confidence_interval'][0]:.4f}, {results['confidence_interval'][1]:.4f})")
```

Slide 13: Probabilistic Early Stopping

This implementation shows how to use probabilistic criteria for early stopping in model training, using Bayesian change point detection.

```python
import numpy as np
from scipy.stats import norm

class ProbabilisticEarlyStopping:
    def __init__(self, patience=10, min_delta=0.01, confidence=0.95):
        self.patience = patience
        self.min_delta = min_delta
        self.confidence = confidence
        self.best_score = -np.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.history = []
        
    def detect_change_point(self, scores):
        """Detect if there's a significant change in the learning curve"""
        if len(scores) < 3:
            return False
            
        recent_scores = np.array(scores[-self.patience:])
        
        # Calculate rolling statistics
        window_size = min(5, len(recent_scores))
        means = np.convolve(recent_scores, 
                           np.ones(window_size)/window_size, 
                           mode='valid')
        
        # Compute probability of improvement
        if len(means) >= 2:
            diff = means[-1] - means[-2]
            std_err = np.std(recent_scores) / np.sqrt(window_size)
            z_score = diff / (std_err + 1e-10)
            prob_improvement = 1 - norm.cdf(z_score)
            
            return prob_improvement < (1 - self.confidence)
        
        return False
    
    def should_stop(self, score):
        """Determine if training should stop"""
        self.history.append(score)
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            
        # Check for plateauing using change point detection
        if self.wait >= self.patience and self.detect_change_point(self.history):
            self.stopped_epoch = len(self.history)
            return True
            
        return False
    
    def get_stopping_metrics(self):
        """Return metrics about the stopping decision"""
        return {
            'best_score': self.best_score,
            'stopped_epoch': self.stopped_epoch,
            'final_wait': self.wait,
            'improvement_probability': 1 - norm.cdf(
                (self.history[-1] - self.best_score) / 
                (np.std(self.history[-self.patience:]) + 1e-10)
            )
        }

# Example usage
early_stopping = ProbabilisticEarlyStopping()

# Simulate training scores
np.random.seed(42)
scores = []
for epoch in range(100):
    if epoch < 30:
        score = 0.5 + 0.01 * epoch + 0.02 * np.random.randn()
    else:
        score = 0.8 + 0.001 * epoch + 0.02 * np.random.randn()
    
    scores.append(score)
    if early_stopping.should_stop(score):
        print(f"Training stopped at epoch {epoch}")
        break
        
metrics = early_stopping.get_stopping_metrics()
print(f"\nBest score: {metrics['best_score']:.4f}")
print(f"Probability of improvement: {metrics['improvement_probability']:.4f}")
```

Slide 14: Additional Resources

*   A Comprehensive Survey of Probabilistic Machine Learning Models
    *   [https://arxiv.org/abs/2008.03132](https://arxiv.org/abs/2008.03132)
*   Bayesian Methods for Machine Learning: A Review
    *   [https://arxiv.org/abs/1906.02912](https://arxiv.org/abs/1906.02912)
*   Modern Approaches to Probabilistic Cross-Validation
    *   [https://arxiv.org/abs/1811.08545](https://arxiv.org/abs/1811.08545)
*   Probabilistic Neural Networks: Theory and Applications
    *   Search on Google Scholar for: "Probabilistic Neural Networks Specht"
*   Recent Advances in Probabilistic Deep Learning
    *   Visit the JMLR website and search for recent publications on probabilistic deep learning

