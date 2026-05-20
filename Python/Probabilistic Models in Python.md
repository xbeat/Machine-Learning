## Probabilistic Models in Python

Slide 1: Introduction to Probabilistic Models

Probabilistic models are mathematical frameworks that use probability theory to represent and analyze uncertainty in complex systems. These models are essential in various fields, including machine learning, statistics, and decision-making under uncertainty. They allow us to quantify and reason about the likelihood of different outcomes based on available information.

```python
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
data = np.random.normal(loc=0, scale=1, size=1000)

# Plot histogram
plt.hist(data, bins=30, density=True, alpha=0.7)
plt.title("Normal Distribution: A Common Probabilistic Model")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.show()
```

Slide 2: Fundamentals of Probability

Probability is the foundation of probabilistic models. It quantifies the likelihood of an event occurring, ranging from 0 (impossible) to 1 (certain). The sum of probabilities for all possible outcomes in a sample space must equal 1.

```python

def coin_flip():
    return random.choice(['Heads', 'Tails'])

# Simulate 10000 coin flips
flips = [coin_flip() for _ in range(10000)]

# Calculate probabilities
p_heads = flips.count('Heads') / len(flips)
p_tails = flips.count('Tails') / len(flips)

print(f"P(Heads) = {p_heads:.4f}")
print(f"P(Tails) = {p_tails:.4f}")
print(f"Sum of probabilities: {p_heads + p_tails:.4f}")
```

Slide 3: Random Variables

Random variables are variables whose values are determined by the outcome of a random process. They can be discrete (taking on specific values) or continuous (taking on any value within a range).

```python
import matplotlib.pyplot as plt

# Discrete random variable: Roll of a fair die
outcomes = np.arange(1, 7)
probabilities = np.ones(6) / 6

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(outcomes, probabilities)
plt.title("Discrete: Die Roll")
plt.xlabel("Outcome")
plt.ylabel("Probability")

# Continuous random variable: Normal distribution
x = np.linspace(-4, 4, 100)
y = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

plt.subplot(1, 2, 2)
plt.plot(x, y)
plt.title("Continuous: Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Probability Density")

plt.tight_layout()
plt.show()
```

Slide 4: Probability Distributions

Probability distributions describe the likelihood of different outcomes for a random variable. Common distributions include uniform, normal (Gaussian), binomial, and Poisson distributions.

```python
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(-4, 4, 100)

# Normal distribution
normal = stats.norm.pdf(x, loc=0, scale=1)

# Uniform distribution
uniform = stats.uniform.pdf(x, loc=-2, scale=4)

# Exponential distribution
exponential = stats.expon.pdf(x, loc=0, scale=1)

plt.figure(figsize=(10, 6))
plt.plot(x, normal, label='Normal')
plt.plot(x, uniform, label='Uniform')
plt.plot(x, exponential, label='Exponential')
plt.title("Common Probability Distributions")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
```

Slide 5: Bayes' Theorem

Bayes' Theorem is a fundamental concept in probabilistic modeling, allowing us to update our beliefs based on new evidence. It relates the conditional and marginal probabilities of events.

```python
    """
    Calculate P(A|B) using Bayes' Theorem
    P(A|B) = P(B|A) * P(A) / P(B)
    """
    p_a_given_b = (p_b_given_a * p_a) / p_b
    return p_a_given_b

# Example: Medical test
p_disease = 0.01  # Prior probability of having the disease
p_positive_given_disease = 0.95  # Sensitivity of the test
p_positive_given_no_disease = 0.05  # False positive rate
p_positive = p_positive_given_disease * p_disease + p_positive_given_no_disease * (1 - p_disease)

p_disease_given_positive = bayes_theorem(p_disease, p_positive_given_disease, p_positive)

print(f"Probability of having the disease given a positive test: {p_disease_given_positive:.4f}")
```

Slide 6: Maximum Likelihood Estimation

Maximum Likelihood Estimation (MLE) is a method for estimating the parameters of a probabilistic model by maximizing the likelihood function. It finds the parameter values that make the observed data most probable.

```python
from scipy.optimize import minimize_scalar

# Generate sample data
np.random.seed(42)
true_mu = 5
true_sigma = 2
data = np.random.normal(true_mu, true_sigma, 1000)

# Define negative log-likelihood function
def neg_log_likelihood(mu, data):
    return -np.sum(np.log(np.exp(-(data - mu)**2 / (2 * true_sigma**2)) / np.sqrt(2 * np.pi * true_sigma**2)))

# Perform MLE
result = minimize_scalar(neg_log_likelihood, args=(data,))
mle_mu = result.x

print(f"True mean: {true_mu}")
print(f"MLE estimate of mean: {mle_mu:.4f}")
```

Slide 7: Markov Chains

Markov Chains are probabilistic models that represent a sequence of events where the probability of each event depends only on the state of the previous event. They are widely used in modeling systems with state transitions.

```python
import matplotlib.pyplot as plt

# Define transition matrix
P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.5, 0.2],
              [0.2, 0.3, 0.5]])

# Simulate Markov Chain
def simulate_markov_chain(P, initial_state, n_steps):
    states = [initial_state]
    for _ in range(n_steps - 1):
        next_state = np.random.choice(3, p=P[states[-1]])
        states.append(next_state)
    return states

# Run simulation
states = simulate_markov_chain(P, 0, 100)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(states)
plt.title("Markov Chain Simulation")
plt.xlabel("Time Step")
plt.ylabel("State")
plt.yticks([0, 1, 2])
plt.show()
```

Slide 8: Hidden Markov Models

Hidden Markov Models (HMMs) extend Markov Chains by introducing hidden states that are not directly observable. They are useful for modeling systems where the underlying state is unknown but produces observable outputs.

```python
from hmmlearn import hmm

# Define HMM parameters
n_states = 2
n_observations = 3

# Create HMM model
model = hmm.GaussianHMM(n_components=n_states, covariance_type="full")

# Set model parameters
model.startprob_ = np.array([0.6, 0.4])
model.transmat_ = np.array([[0.7, 0.3],
                            [0.4, 0.6]])
model.means_ = np.array([[0.0, 0.0, 0.0],
                         [3.0, 3.0, 3.0]])
model.covars_ = np.tile(np.identity(n_observations), (n_states, 1, 1))

# Generate sample data
X, Z = model.sample(100)

# Estimate hidden states
hidden_states = model.predict(X)

print("First 10 true hidden states:", Z[:10])
print("First 10 estimated hidden states:", hidden_states[:10])
```

Slide 9: Bayesian Networks

Bayesian Networks are graphical models that represent probabilistic relationships among a set of variables. They use directed acyclic graphs (DAGs) to encode conditional dependencies and independencies.

```python
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure of the Bayesian Network
model = BayesianNetwork([('Rain', 'Sprinkler'), ('Rain', 'Wet_Grass'), ('Sprinkler', 'Wet_Grass')])

# Define the conditional probability distributions
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

Slide 10: Gaussian Mixture Models

Gaussian Mixture Models (GMMs) are probabilistic models that represent a distribution as a weighted sum of Gaussian distributions. They are useful for clustering and density estimation tasks.

```python
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Generate sample data
np.random.seed(42)
X = np.concatenate([
    np.random.normal(0, 1, (300, 2)),
    np.random.normal(4, 1.5, (700, 2))
])

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X)

# Plot results
x = np.linspace(-5, 10, 100)
y = np.linspace(-5, 10, 100)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(X.shape)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c='white', alpha=0.5, s=5)
plt.title("Gaussian Mixture Model")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Negative log-likelihood")
plt.show()
```

Slide 11: Monte Carlo Methods

Monte Carlo methods are a class of computational algorithms that rely on repeated random sampling to obtain numerical results. They are widely used in probabilistic modeling for approximating complex probabilities and integrals.

```python
import matplotlib.pyplot as plt

def monte_carlo_pi(n_points):
    points = np.random.rand(n_points, 2)
    inside_circle = np.sum(np.sum(points**2, axis=1) <= 1)
    pi_estimate = 4 * inside_circle / n_points
    return pi_estimate

# Estimate pi using different numbers of points
n_points_list = [100, 1000, 10000, 100000, 1000000]
pi_estimates = [monte_carlo_pi(n) for n in n_points_list]

plt.figure(figsize=(10, 6))
plt.semilogx(n_points_list, pi_estimates, 'o-')
plt.axhline(y=np.pi, color='r', linestyle='--', label='True value of π')
plt.title("Monte Carlo Estimation of π")
plt.xlabel("Number of points")
plt.ylabel("Estimated value of π")
plt.legend()
plt.grid(True)
plt.show()

print("Estimated π:", pi_estimates[-1])
print("True π:", np.pi)
```

Slide 12: Real-life Example: Weather Prediction

Weather prediction is a common application of probabilistic models. Meteorologists use complex models that incorporate various factors like temperature, pressure, and humidity to forecast weather conditions.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic weather data
np.random.seed(42)
n_samples = 1000
temperature = np.random.normal(20, 5, n_samples)
humidity = np.random.normal(60, 10, n_samples)
pressure = np.random.normal(1013, 5, n_samples)

# Create target variable (0: No rain, 1: Rain)
rain = (0.3 * temperature + 0.4 * humidity - 0.2 * pressure + np.random.normal(0, 5, n_samples)) > 25

# Prepare data for logistic regression
X = np.column_stack((temperature, humidity, pressure))
y = rain.astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Weather prediction accuracy: {accuracy:.2f}")

# Predict probability of rain for a specific day
new_day = np.array([[25, 70, 1010]])
rain_probability = model.predict_proba(new_day)[0, 1]
print(f"Probability of rain: {rain_probability:.2f}")
```

Slide 13: Real-life Example: Natural Language Processing

Probabilistic models are extensively used in Natural Language Processing (NLP) for tasks such as text classification, sentiment analysis, and language generation. Here's a simple example of sentiment analysis using a Naive Bayes classifier.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
texts = [
    "I love this product", "Great service", "Terrible experience",
    "Awful customer support", "Highly recommended", "Waste of money",
    "Amazing quality", "Poor performance", "Outstanding results",
    "Disappointing purchase"
]
labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # 1: Positive, 0: Negative

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a bag of words representation
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Make predictions
y_pred = clf.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Predict sentiment for a new review
new_review = ["This product exceeded my expectations"]
new_review_vec = vectorizer.transform(new_review)
prediction = clf.predict(new_review_vec)
print(f"\nPredicted sentiment: {'Positive' if prediction[0] == 1 else 'Negative'}")
```

Slide 14: Probabilistic Graphical Models

Probabilistic Graphical Models (PGMs) combine graph theory and probability theory to represent complex dependencies among random variables. They provide a compact representation of joint probability distributions and enable efficient inference and learning.

```python
import matplotlib.pyplot as plt

# Create a simple Bayesian Network
G = nx.DiGraph()
G.add_edges_from([
    ('Cloudy', 'Rain'),
    ('Cloudy', 'Sprinkler'),
    ('Rain', 'Wet Grass'),
    ('Sprinkler', 'Wet Grass')
])

# Plot the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=3000, font_size=12, font_weight='bold', 
        arrows=True, edge_color='gray')
plt.title("Bayesian Network: Wet Grass Example")
plt.axis('off')
plt.show()

# Conditional Probability Tables (simplified example)
P_Cloudy = {'True': 0.4, 'False': 0.6}
P_Rain_given_Cloudy = {
    ('True', 'True'): 0.8,
    ('True', 'False'): 0.2,
    ('False', 'True'): 0.2,
    ('False', 'False'): 0.8
}

print("P(Cloudy):", P_Cloudy)
print("P(Rain|Cloudy):", P_Rain_given_Cloudy)
```

Slide 15: Probabilistic Programming

Probabilistic programming languages provide a high-level interface for defining and working with probabilistic models. They allow users to express complex probabilistic models using programming constructs and automatically perform inference.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + 1 + np.random.randn(100) * 0.5

# Define the probabilistic model
with pm.Model() as model:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)
    
    # Linear model
    mu = alpha + beta * x
    
    # Likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=y)
    
    # Inference
    trace = pm.sample(1000, return_inferencedata=False)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6)
plt.plot(x, trace['alpha'].mean() + trace['beta'].mean() * x, 'r', label='Fitted line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Probabilistic Linear Regression')
plt.legend()
plt.show()

print("Estimated alpha:", trace['alpha'].mean())
print("Estimated beta:", trace['beta'].mean())
```

Slide 16: Additional Resources

For those interested in delving deeper into probabilistic models, here are some valuable resources:

1. "Probabilistic Graphical Models: Principles and Techniques" by Daphne Koller and Nir Friedman ArXiv link: [https://arxiv.org/abs/1301.6725](https://arxiv.org/abs/1301.6725)
2. "Machine Learning: A Probabilistic Perspective" by Kevin P. Murphy ArXiv link: [https://arxiv.org/abs/1206.5528](https://arxiv.org/abs/1206.5528)
3. "Bayesian Reasoning and Machine Learning" by David Barber ArXiv link: [https://arxiv.org/abs/1504.07641](https://arxiv.org/abs/1504.07641)
4. "Introduction to Probabilistic Topic Models" by David M. Blei ArXiv link: [https://arxiv.org/abs/1108.0545](https://arxiv.org/abs/1108.0545)

These resources provide in-depth coverage of various aspects of probabilistic modeling and their applications in machine learning and artificial intelligence.


