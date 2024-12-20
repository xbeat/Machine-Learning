## Bayes' Theorem Explained with Python
Slide 1: Introduction to Bayes' Theorem

Bayes' Theorem is a fundamental concept in probability theory and statistics. It provides a way to update our beliefs about the probability of an event based on new evidence. This powerful tool has applications in various fields, including machine learning, data analysis, and decision-making under uncertainty.

```python
import numpy as np
import matplotlib.pyplot as plt

def bayes_theorem(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

# Example probabilities
prior = 0.3
likelihood = 0.8
evidence = 0.5

posterior = bayes_theorem(prior, likelihood, evidence)
print(f"Posterior probability: {posterior:.2f}")
```

Slide 2: The Formula

Bayes' Theorem is expressed mathematically as P(A|B) = (P(B|A) \* P(A)) / P(B), where P(A|B) is the posterior probability, P(B|A) is the likelihood, P(A) is the prior probability, and P(B) is the evidence.

```python
def bayes_formula(p_a, p_b_given_a, p_b):
    return (p_b_given_a * p_a) / p_b

# Example calculation
p_a = 0.1  # Prior probability of event A
p_b_given_a = 0.7  # Likelihood of B given A
p_b = 0.3  # Evidence (probability of B)

p_a_given_b = bayes_formula(p_a, p_b_given_a, p_b)
print(f"P(A|B) = {p_a_given_b:.4f}")
```

Slide 3: Components of Bayes' Theorem

Let's break down the components of Bayes' Theorem: Prior probability (initial belief), Likelihood (probability of evidence given the hypothesis), and Evidence (total probability of observing the evidence).

```python
class BayesComponents:
    def __init__(self, prior, likelihood, evidence):
        self.prior = prior
        self.likelihood = likelihood
        self.evidence = evidence
    
    def calculate_posterior(self):
        return (self.likelihood * self.prior) / self.evidence

# Example usage
components = BayesComponents(0.2, 0.8, 0.4)
posterior = components.calculate_posterior()
print(f"Posterior probability: {posterior:.2f}")
```

Slide 4: Updating Beliefs

Bayes' Theorem allows us to update our beliefs as we gather new evidence. This process of iterative updating is central to Bayesian inference and learning.

```python
def update_belief(prior, likelihood, evidence):
    posterior = (likelihood * prior) / evidence
    return posterior

initial_belief = 0.3
new_evidence_likelihood = 0.7
total_probability = 0.5

updated_belief = update_belief(initial_belief, new_evidence_likelihood, total_probability)
print(f"Updated belief: {updated_belief:.2f}")

# Iterative updating
for i in range(3):
    updated_belief = update_belief(updated_belief, new_evidence_likelihood, total_probability)
    print(f"Iteration {i+1}: {updated_belief:.2f}")
```

Slide 5: Visualizing Bayes' Theorem

Let's create a simple visualization to understand how the posterior probability changes with different prior probabilities and likelihoods.

```python
import matplotlib.pyplot as plt
import numpy as np

def bayes_theorem(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

priors = np.linspace(0, 1, 100)
likelihoods = [0.2, 0.5, 0.8]
evidence = 0.5

plt.figure(figsize=(10, 6))
for likelihood in likelihoods:
    posteriors = [bayes_theorem(prior, likelihood, evidence) for prior in priors]
    plt.plot(priors, posteriors, label=f'Likelihood = {likelihood}')

plt.xlabel('Prior Probability')
plt.ylabel('Posterior Probability')
plt.title('Bayes\' Theorem: Posterior vs Prior')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 6: Real-Life Example: Medical Diagnosis

Consider a medical test for a rare disease. The test has a 99% true positive rate and a 1% false positive rate. The disease occurs in 0.1% of the population. What's the probability that a person who tests positive actually has the disease?

```python
def medical_diagnosis(prevalence, sensitivity, specificity):
    # prevalence: probability of having the disease
    # sensitivity: true positive rate
    # specificity: true negative rate
    
    p_positive_given_disease = sensitivity
    p_disease = prevalence
    p_positive = (p_positive_given_disease * p_disease) + 
                 ((1 - specificity) * (1 - p_disease))
    
    p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
    return p_disease_given_positive

prevalence = 0.001  # 0.1% of population has the disease
sensitivity = 0.99  # 99% true positive rate
specificity = 0.99  # 99% true negative rate (1% false positive rate)

probability = medical_diagnosis(prevalence, sensitivity, specificity)
print(f"Probability of having the disease given a positive test: {probability:.2%}")
```

Slide 7: Naive Bayes Classifier

Naive Bayes is a simple yet powerful classification algorithm based on Bayes' Theorem. It assumes that features are independent, which simplifies calculations but may not always hold true in reality.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Generate sample data
X = np.random.randn(1000, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Naive Bayes Classifier Accuracy: {accuracy:.2f}")
```

Slide 8: Bayesian Parameter Estimation

Bayesian parameter estimation uses Bayes' Theorem to update our beliefs about model parameters as we observe data. Let's estimate the probability of success in a binomial distribution.

```python
import numpy as np
from scipy import stats

def bayesian_binomial_estimation(prior_alpha, prior_beta, data):
    successes = np.sum(data)
    trials = len(data)
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + trials - successes
    return posterior_alpha, posterior_beta

# Generate some sample data (coin flips)
np.random.seed(42)
data = np.random.binomial(1, 0.7, size=100)

# Prior (Beta distribution parameters)
prior_alpha, prior_beta = 1, 1

# Estimate parameters
post_alpha, post_beta = bayesian_binomial_estimation(prior_alpha, prior_beta, data)

# Plot prior and posterior distributions
x = np.linspace(0, 1, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, stats.beta.pdf(x, prior_alpha, prior_beta), label='Prior')
plt.plot(x, stats.beta.pdf(x, post_alpha, post_beta), label='Posterior')
plt.title('Bayesian Parameter Estimation: Beta-Binomial Model')
plt.xlabel('Probability of Success')
plt.ylabel('Density')
plt.legend()
plt.show()

print(f"Posterior mean: {post_alpha / (post_alpha + post_beta):.3f}")
```

Slide 9: Bayesian A/B Testing

Bayesian A/B testing uses Bayes' Theorem to compare two or more variants and calculate the probability that one variant is better than another.

```python
import numpy as np
from scipy import stats

def bayesian_ab_test(data_a, data_b, n_simulations=100000):
    alpha_a, beta_a = 1 + np.sum(data_a), 1 + len(data_a) - np.sum(data_a)
    alpha_b, beta_b = 1 + np.sum(data_b), 1 + len(data_b) - np.sum(data_b)
    
    samples_a = np.random.beta(alpha_a, beta_a, n_simulations)
    samples_b = np.random.beta(alpha_b, beta_b, n_simulations)
    
    prob_b_better = np.mean(samples_b > samples_a)
    return prob_b_better

# Example data (conversions for variants A and B)
data_a = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 1])
data_b = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 1])

prob_b_better = bayesian_ab_test(data_a, data_b)
print(f"Probability that B is better than A: {prob_b_better:.2%}")
```

Slide 10: Real-Life Example: Spam Filter

Let's implement a simple Bayesian spam filter using the Naive Bayes assumption. We'll classify emails as spam or not spam based on the presence of certain words.

```python
import numpy as np

class NaiveBayesSpamFilter:
    def __init__(self):
        self.word_counts = {'spam': {}, 'ham': {}}
        self.class_counts = {'spam': 0, 'ham': 0}
    
    def train(self, emails, labels):
        for email, label in zip(emails, labels):
            self.class_counts[label] += 1
            for word in email.split():
                self.word_counts[label][word] = self.word_counts[label].get(word, 0) + 1
    
    def classify(self, email):
        words = email.split()
        spam_prob = np.log(self.class_counts['spam'] / sum(self.class_counts.values()))
        ham_prob = np.log(self.class_counts['ham'] / sum(self.class_counts.values()))
        
        for word in words:
            if word in self.word_counts['spam']:
                spam_prob += np.log((self.word_counts['spam'][word] + 1) / 
                                    (self.class_counts['spam'] + len(self.word_counts['spam'])))
            if word in self.word_counts['ham']:
                ham_prob += np.log((self.word_counts['ham'][word] + 1) / 
                                   (self.class_counts['ham'] + len(self.word_counts['ham'])))
        
        return 'spam' if spam_prob > ham_prob else 'ham'

# Example usage
spam_filter = NaiveBayesSpamFilter()
emails = [
    "win free money now",
    "meeting at 3pm tomorrow",
    "buy cheap watches online",
    "project deadline next week"
]
labels = ['spam', 'ham', 'spam', 'ham']

spam_filter.train(emails, labels)

test_email = "free money guaranteed"
result = spam_filter.classify(test_email)
print(f"Classification result: {result}")
```

Slide 11: Bayesian Networks

Bayesian networks are probabilistic graphical models that represent relationships between variables using directed acyclic graphs. They use Bayes' Theorem to perform inference.

```python
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Create a simple Bayesian network
model = BayesianNetwork([('Rain', 'Sprinkler'), ('Rain', 'Wet_Grass'), ('Sprinkler', 'Wet_Grass')])

# Define conditional probability distributions
cpd_rain = TabularCPD(variable='Rain', variable_card=2, values=[[0.8], [0.2]])
cpd_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
                           values=[[0.6, 0.99],
                                   [0.4, 0.01]],
                           evidence=['Rain'], evidence_card=[2])
cpd_wet_grass = TabularCPD(variable='Wet_Grass', variable_card=2,
                           values=[[1.0, 0.1, 0.1, 0.01],
                                   [0.0, 0.9, 0.9, 0.99]],
                           evidence=['Sprinkler', 'Rain'], evidence_card=[2, 2])

# Add CPDs to the model
model.add_cpds(cpd_rain, cpd_sprinkler, cpd_wet_grass)

# Perform inference
inference = VariableElimination(model)
result = inference.query(variables=['Wet_Grass'], evidence={'Rain': 1})
print("Probability of wet grass given that it's raining:")
print(result)
```

Slide 12: Markov Chain Monte Carlo (MCMC)

MCMC is a class of algorithms for sampling from probability distributions based on constructing a Markov chain. It's often used in Bayesian inference for complex models.

```python
import numpy as np
import matplotlib.pyplot as plt

def metropolis_hastings(target_distribution, proposal_distribution, initial_state, n_samples):
    current_state = initial_state
    samples = [current_state]
    
    for _ in range(n_samples - 1):
        proposed_state = proposal_distribution(current_state)
        
        acceptance_ratio = min(1, target_distribution(proposed_state) / target_distribution(current_state))
        
        if np.random.random() < acceptance_ratio:
            current_state = proposed_state
        
        samples.append(current_state)
    
    return np.array(samples)

# Example: sampling from a mixture of Gaussians
def target_dist(x):
    return 0.3 * np.exp(-0.2 * (x - 2)**2) + 0.7 * np.exp(-0.2 * (x + 2)**2)

def proposal_dist(x):
    return x + np.random.normal(0, 0.5)

samples = metropolis_hastings(target_dist, proposal_dist, 0, 10000)

plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.7)
x = np.linspace(-6, 6, 1000)
plt.plot(x, target_dist(x), 'r-', lw=2)
plt.title('MCMC Sampling from Mixture of Gaussians')
plt.xlabel('x')
plt.ylabel('Density')
plt.show()
```

Slide 13: Bayesian Optimization

Bayesian optimization is a technique for global optimization of black-box functions. It uses Bayes' Theorem to update beliefs about the function and guide the search for the optimum.

```python
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

def objective_function(x):
    return -(x**2 * np.sin(5 * np.pi * x)**6)

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X.reshape(-1, 1), return_std=True)
    mu_sample = gpr.predict(X_sample.reshape(-1, 1))
    
    sigma = sigma.reshape(-1, 1)
    mu_sample_opt = np.max(mu_sample)
    
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    
    return ei

def bayesian_optimization(n_iterations, n_initial_points=5):
    X_init = np.random.uniform(-1, 2, n_initial_points).reshape(-1, 1)
    Y_init = objective_function(X_init)
    
    X_sample = X_init
    Y_sample = Y_init
    
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25)
    
    for _ in range(n_iterations):
        gpr.fit(X_sample, Y_sample)
        
        X = np.linspace(-1, 2, 1000).reshape(-1, 1)
        ei = expected_improvement(X, X_sample, Y_sample, gpr)
        
        X_next = X[np.argmax(ei)]
        Y_next = objective_function(X_next)
        
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))
    
    return X_sample, Y_sample

X_sample, Y_sample = bayesian_optimization(15)
print(f"Optimal X: {X_sample[np.argmax(Y_sample)]}")
print(f"Optimal Y: {np.max(Y_sample)}")
```

Slide 14: Bayesian Deep Learning

Bayesian deep learning combines deep neural networks with Bayesian inference, allowing for uncertainty quantification in predictions. Here's a simple example using PyTorch.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        
    def forward(self, input, sample=False):
        if self.training or sample:
            weight = self.weight_mu + torch.log(1 + torch.exp(self.weight_rho)) * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + torch.log(1 + torch.exp(self.bias_rho)) * torch.randn_like(self.bias_mu)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return nn.functional.linear(input, weight, bias)

# Example usage
model = nn.Sequential(
    BayesianLinear(10, 20),
    nn.ReLU(),
    BayesianLinear(20, 1)
)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop (pseudo-code)
for epoch in range(num_epochs):
    for batch in data_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Prediction with uncertainty
def predict_with_uncertainty(model, input, num_samples=100):
    model.eval()
    predictions = []
    for _ in range(num_samples):
        predictions.append(model(input, sample=True))
    return torch.stack(predictions)

# Use predict_with_uncertainty to get mean and standard deviation of predictions
```

Slide 15: Additional Resources

For those interested in diving deeper into Bayesian methods and their applications, here are some valuable resources:

1. "Probabilistic Programming & Bayesian Methods for Hackers" by Cameron Davidson-Pilon An open-source book that introduces Bayesian methods through probabilistic programming.
2. "Bayesian Data Analysis" by Andrew Gelman, John Carlin, Hal Stern, David Dunson, Aki Vehtari, and Donald Rubin A comprehensive textbook on Bayesian statistical methods and their applications.
3. "Machine Learning: A Probabilistic Perspective" by Kevin Murphy A thorough introduction to machine learning with a strong focus on probabilistic approaches.
4. ArXiv paper: "A Survey of Deep Learning Techniques for Neural Machine Translation" by Shuohang Wang and Jing Jiang ArXiv link: [https://arxiv.org/abs/1511.04534](https://arxiv.org/abs/1511.04534)
5. ArXiv paper: "Practical Bayesian Optimization of Machine Learning Algorithms" by Jasper Snoek, Hugo Larochelle, and Ryan P. Adams ArXiv link: [https://arxiv.org/abs/1206.2944](https://arxiv.org/abs/1206.2944)

These resources provide a mix of practical implementations and theoretical foundations to further your understanding of Bayesian methods and their applications in various fields.

