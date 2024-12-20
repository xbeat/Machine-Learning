## Probabilistic Graphical Models in Python
Slide 1: Introduction to Probabilistic Graphical Models

Probabilistic graphical models (PGMs) are powerful tools for representing and reasoning about complex systems with uncertainty. They combine probability theory and graph theory to model relationships between variables. In this presentation, we'll explore various types of inference available for PGMs using Python, focusing on practical implementations and real-world applications.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple Bayesian network
G = nx.DiGraph()
G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, arrowsize=20)
plt.title("Simple Bayesian Network")
plt.show()
```

Slide 2: Exact Inference: Variable Elimination

Variable elimination is an exact inference algorithm for PGMs. It computes marginal probabilities by eliminating variables one by one. This method is efficient for small to medium-sized networks but can become computationally expensive for large, complex models.

```python
import numpy as np

def variable_elimination(factors, query_var, evidence):
    for var in factors:
        if var not in query_var and var not in evidence:
            factors = sum_out_variable(factors, var)
    result = multiply_factors(factors)
    return normalize(result)

def sum_out_variable(factors, var):
    # Implementation details omitted for brevity
    pass

def multiply_factors(factors):
    # Implementation details omitted for brevity
    pass

def normalize(factor):
    return factor / np.sum(factor)

# Example usage
factors = {...}  # Define factors
query_var = 'A'
evidence = {'B': True, 'C': False}
result = variable_elimination(factors, query_var, evidence)
print(f"P({query_var} | evidence) = {result}")
```

Slide 3: Exact Inference: Junction Tree Algorithm

The Junction Tree algorithm is another exact inference method for PGMs. It transforms the original graph into a tree of cliques, allowing efficient belief propagation. This algorithm is particularly useful for networks with loops.

```python
import networkx as nx

def junction_tree_algorithm(graph):
    # Step 1: Moralization
    moral_graph = nx.moral_graph(graph)
    
    # Step 2: Triangulation
    triangulated_graph = nx.triangulate(moral_graph)
    
    # Step 3: Find maximal cliques
    cliques = list(nx.find_cliques(triangulated_graph))
    
    # Step 4: Build junction tree
    junction_tree = nx.junction_tree(triangulated_graph, cliques)
    
    return junction_tree

# Example usage
G = nx.DiGraph()
G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
jt = junction_tree_algorithm(G)

# Visualize the junction tree
pos = nx.spring_layout(jt)
nx.draw(jt, pos, with_labels=True, node_color='lightgreen', node_size=1000, font_size=8)
plt.title("Junction Tree")
plt.show()
```

Slide 4: Approximate Inference: Gibbs Sampling

Gibbs sampling is a Markov Chain Monte Carlo (MCMC) method for approximate inference in PGMs. It generates samples from the joint distribution by iteratively sampling each variable conditioned on the others. This technique is useful for high-dimensional models where exact inference is intractable.

```python
import numpy as np

def gibbs_sampling(num_samples, burn_in, initial_state, conditional_distributions):
    current_state = initial_state.()
    samples = []

    for i in range(num_samples + burn_in):
        for var in current_state:
            current_state[var] = conditional_distributions[var](current_state)
        
        if i >= burn_in:
            samples.append(current_state.())

    return samples

# Example: Bivariate normal distribution
def conditional_x(state):
    return np.random.normal(0.5 * state['y'], np.sqrt(1 - 0.5**2))

def conditional_y(state):
    return np.random.normal(0.5 * state['x'], np.sqrt(1 - 0.5**2))

conditional_distributions = {'x': conditional_x, 'y': conditional_y}
initial_state = {'x': 0, 'y': 0}

samples = gibbs_sampling(1000, 100, initial_state, conditional_distributions)

# Plot results
x_samples, y_samples = zip(*[(s['x'], s['y']) for s in samples])
plt.scatter(x_samples, y_samples, alpha=0.1)
plt.title("Gibbs Sampling: Bivariate Normal Distribution")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Slide 5: Approximate Inference: Particle Filtering

Particle filtering, also known as Sequential Monte Carlo, is an approximate inference method for dynamic systems. It's particularly useful for tracking and state estimation in time-series data. The algorithm maintains a set of weighted particles representing the belief state and updates them as new observations arrive.

```python
import numpy as np
import matplotlib.pyplot as plt

def particle_filter(num_particles, observations, motion_model, observation_model):
    particles = np.random.uniform(0, 100, num_particles)
    weights = np.ones(num_particles) / num_particles
    estimated_states = []

    for obs in observations:
        # Predict
        particles = motion_model(particles)
        
        # Update
        weights *= observation_model(particles, obs)
        weights /= np.sum(weights)
        
        # Resample
        if 1.0 / np.sum(weights**2) < num_particles / 2:
            indices = np.random.choice(num_particles, num_particles, p=weights)
            particles = particles[indices]
            weights = np.ones(num_particles) / num_particles
        
        estimated_states.append(np.average(particles, weights=weights))

    return np.array(estimated_states)

# Example: 1D robot localization
def motion_model(particles):
    return particles + np.random.normal(0, 2, len(particles))

def observation_model(particles, obs):
    return np.exp(-0.5 * ((particles - obs) / 5)**2) / (5 * np.sqrt(2 * np.pi))

true_states = np.cumsum(np.random.normal(0, 1, 100))
observations = true_states + np.random.normal(0, 5, 100)

estimated_states = particle_filter(1000, observations, motion_model, observation_model)

plt.plot(true_states, label='True State')
plt.plot(observations, 'r.', label='Observations')
plt.plot(estimated_states, 'g-', label='Estimated State')
plt.legend()
plt.title("Particle Filtering: 1D Robot Localization")
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.show()
```

Slide 6: Variational Inference: Mean Field Approximation

Variational inference is an optimization-based approach to approximate inference in PGMs. The mean field approximation is a popular variational method that assumes independence between variables in the approximate posterior distribution. This technique is particularly useful for large-scale models.

```python
import numpy as np
import matplotlib.pyplot as plt

def mean_field_vi(data, num_components, num_iterations):
    N, D = data.shape
    
    # Initialize variational parameters
    alpha = np.random.gamma(1, 1, num_components)
    beta = np.random.gamma(1, 1, (num_components, D))
    phi = np.random.dirichlet(np.ones(num_components), N)
    
    for _ in range(num_iterations):
        # Update phi
        log_phi = np.dot(data, np.log(beta).T) + np.digamma(alpha) - np.sum(np.digamma(alpha))
        log_phi -= np.max(log_phi, axis=1, keepdims=True)
        phi = np.exp(log_phi)
        phi /= np.sum(phi, axis=1, keepdims=True)
        
        # Update alpha and beta
        alpha = 1 + np.sum(phi, axis=0)
        beta = 1 + np.dot(phi.T, data)
    
    return alpha, beta, phi

# Generate synthetic data
np.random.seed(42)
N, D, K = 1000, 2, 3
true_means = np.array([[0, 0], [5, 5], [-5, 5]])
data = np.vstack([np.random.multivariate_normal(mean, np.eye(2), N//K) for mean in true_means])

# Run mean field variational inference
alpha, beta, phi = mean_field_vi(data, K, 100)

# Plot results
plt.scatter(data[:, 0], data[:, 1], c=np.argmax(phi, axis=1), alpha=0.5)
plt.scatter(beta[:, 0] / alpha, beta[:, 1] / alpha, c='r', s=200, marker='*')
plt.title("Mean Field VI: Gaussian Mixture Model")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Slide 7: Expectation Propagation

Expectation Propagation (EP) is another approximate inference method for PGMs. It approximates the posterior distribution by iteratively refining local approximations. EP is particularly effective for models with non-Gaussian likelihoods and can often provide more accurate approximations than variational methods.

```python
import numpy as np
from scipy.stats import norm

def expectation_propagation(data, prior_mean, prior_var, num_iterations):
    N = len(data)
    posterior_mean = prior_mean
    posterior_var = prior_var
    site_means = np.zeros(N)
    site_vars = np.inf * np.ones(N)

    for _ in range(num_iterations):
        for i in range(N):
            # Remove site i from posterior
            cavity_var = 1 / (1/posterior_var - 1/site_vars[i])
            cavity_mean = cavity_var * (posterior_mean/posterior_var - site_means[i]/site_vars[i])

            # Moment matching
            z = (data[i] - cavity_mean) / np.sqrt(cavity_var + 1)
            alpha = norm.pdf(z) / norm.cdf(z)
            beta = alpha * (alpha + z)

            new_var = 1 / (1/cavity_var + beta / (cavity_var + 1))
            new_mean = new_var * (cavity_mean/cavity_var + alpha / np.sqrt(cavity_var + 1))

            # Update site parameters
            site_vars[i] = 1 / (1/new_var - 1/cavity_var)
            site_means[i] = site_vars[i] * (new_mean/new_var - cavity_mean/cavity_var)

            # Update posterior
            posterior_var = 1 / (1/prior_var + np.sum(1/site_vars))
            posterior_mean = posterior_var * (prior_mean/prior_var + np.sum(site_means/site_vars))

    return posterior_mean, posterior_var

# Example: Probit regression
np.random.seed(42)
X = np.random.randn(100)
y = (X > 0).astype(int)

prior_mean, prior_var = 0, 10
post_mean, post_var = expectation_propagation(y, prior_mean, prior_var, 10)

# Plot results
x_range = np.linspace(-3, 3, 100)
plt.plot(x_range, norm.pdf(x_range, post_mean, np.sqrt(post_var)), label='EP Posterior')
plt.scatter(X, y, c='r', alpha=0.5, label='Data')
plt.title("Expectation Propagation: Probit Regression")
plt.xlabel("X")
plt.ylabel("Probability")
plt.legend()
plt.show()
```

Slide 8: Loopy Belief Propagation

Loopy Belief Propagation (LBP) is an approximate inference method for PGMs with cycles. It applies the belief propagation algorithm to graphs with loops, iteratively passing messages between nodes until convergence. While not guaranteed to converge, LBP often produces good approximations in practice.

```python
import networkx as nx
import numpy as np

def loopy_belief_propagation(graph, potentials, max_iterations=100, tolerance=1e-5):
    messages = {edge: np.ones(2) for edge in graph.edges()}
    beliefs = {node: np.ones(2) for node in graph.nodes()}

    for _ in range(max_iterations):
        old_beliefs = beliefs.()

        for node in graph.nodes():
            for neighbor in graph.neighbors(node):
                incoming_messages = [messages[(other, node)] for other in graph.neighbors(node) if other != neighbor]
                message = np.einsum('i,i->i', potentials[node], np.prod(incoming_messages, axis=0))
                messages[(node, neighbor)] = message / np.sum(message)

        for node in graph.nodes():
            incoming_messages = [messages[(neighbor, node)] for neighbor in graph.neighbors(node)]
            beliefs[node] = np.einsum('i,i->i', potentials[node], np.prod(incoming_messages, axis=0))
            beliefs[node] /= np.sum(beliefs[node])

        if all(np.allclose(old_beliefs[node], beliefs[node], atol=tolerance) for node in graph.nodes()):
            break

    return beliefs

# Example: Simple 4-node graph
G = nx.Graph()
G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])

potentials = {
    'A': np.array([0.6, 0.4]),
    'B': np.array([0.7, 0.3]),
    'C': np.array([0.4, 0.6]),
    'D': np.array([0.5, 0.5])
}

beliefs = loopy_belief_propagation(G, potentials)

for node, belief in beliefs.items():
    print(f"Node {node}: P(0) = {belief[0]:.3f}, P(1) = {belief[1]:.3f}")

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
plt.title("Loopy Belief Propagation Graph")
plt.show()
```

Slide 9: Real-Life Example: Image Denoising

Probabilistic graphical models can be used for image denoising. In this example, we'll use a simple Markov Random Field (MRF) to remove noise from a binary image. The MRF represents pixel dependencies, and we'll use Gibbs sampling for inference.

```python
import numpy as np
import matplotlib.pyplot as plt

def add_noise(image, noise_level):
    return np.where(np.random.rand(*image.shape) < noise_level, 1 - image, image)

def gibbs_sampling_denoising(noisy_image, num_iterations, beta):
    height, width = noisy_image.shape
    denoised = noisy_image.()

    for _ in range(num_iterations):
        for i in range(height):
            for j in range(width):
                neighbors = [
                    denoised[max(i-1, 0), j],
                    denoised[min(i+1, height-1), j],
                    denoised[i, max(j-1, 0)],
                    denoised[i, min(j+1, width-1)]
                ]
                energy_0 = sum([beta * (0 != n) for n in neighbors])
                energy_1 = sum([beta * (1 != n) for n in neighbors])
                energy_0 += (0 != noisy_image[i, j])
                energy_1 += (1 != noisy_image[i, j])
                p1 = 1 / (1 + np.exp(energy_1 - energy_0))
                denoised[i, j] = np.random.choice([0, 1], p=[1-p1, p1])
    
    return denoised

# Create a simple binary image
image = np.zeros((50, 50))
image[10:40, 10:40] = 1

# Add noise and denoise
noisy = add_noise(image, 0.2)
denoised = gibbs_sampling_denoising(noisy, 50, 0.8)

# Display results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(image, cmap='binary')
ax1.set_title("Original")
ax2.imshow(noisy, cmap='binary')
ax2.set_title("Noisy")
ax3.imshow(denoised, cmap='binary')
ax3.set_title("Denoised")
plt.show()
```

Slide 10: Real-Life Example: Naive Bayes for Text Classification

Naive Bayes is a simple yet effective probabilistic model for text classification. It assumes independence between features (words), making it a special case of Bayesian networks. Let's implement a Naive Bayes classifier for sentiment analysis.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_probs = {c: np.mean(y == c) for c in self.classes}
        self.word_counts = {c: X[y == c].sum(axis=0) for c in self.classes}
        self.word_probs = {c: (self.word_counts[c] + 1) / (self.word_counts[c].sum() + X.shape[1])
                           for c in self.classes}
    
    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])
    
    def predict_single(self, x):
        probs = {c: np.log(self.class_probs[c]) + np.sum(x * np.log(self.word_probs[c]))
                 for c in self.classes}
        return max(probs, key=probs.get)

# Example usage
texts = [
    "I love this movie", "Great film, highly recommended",
    "Terrible movie, waste of time", "Awful acting, poor plot"
]
labels = ["positive", "positive", "negative", "negative"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()

nb = NaiveBayes()
nb.fit(X, labels)

# Test prediction
test_text = "Interesting movie with good acting"
test_vector = vectorizer.transform([test_text]).toarray()
prediction = nb.predict(test_vector)

print(f"Text: '{test_text}'")
print(f"Predicted sentiment: {prediction[0]}")
```

Slide 11: Inference in Hidden Markov Models

Hidden Markov Models (HMMs) are probabilistic graphical models used for sequential data. They consist of hidden states and observed emissions. The Forward-Backward algorithm is used for inference in HMMs.

```python
import numpy as np

class HMM:
    def __init__(self, A, B, pi):
        self.A = A  # Transition probabilities
        self.B = B  # Emission probabilities
        self.pi = pi  # Initial state distribution
    
    def forward(self, observations):
        N = len(self.pi)
        T = len(observations)
        alpha = np.zeros((N, T))
        
        # Initialization
        alpha[:, 0] = self.pi * self.B[:, observations[0]]
        
        # Forward pass
        for t in range(1, T):
            for j in range(N):
                alpha[j, t] = self.B[j, observations[t]] * np.sum(alpha[:, t-1] * self.A[:, j])
        
        return alpha
    
    def backward(self, observations):
        N = len(self.pi)
        T = len(observations)
        beta = np.zeros((N, T))
        
        # Initialization
        beta[:, -1] = 1
        
        # Backward pass
        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[i, t] = np.sum(self.A[i, :] * self.B[:, observations[t+1]] * beta[:, t+1])
        
        return beta
    
    def viterbi(self, observations):
        N = len(self.pi)
        T = len(observations)
        delta = np.zeros((N, T))
        psi = np.zeros((N, T), dtype=int)
        
        # Initialization
        delta[:, 0] = self.pi * self.B[:, observations[0]]
        
        # Recursion
        for t in range(1, T):
            for j in range(N):
                delta[j, t] = np.max(delta[:, t-1] * self.A[:, j]) * self.B[j, observations[t]]
                psi[j, t] = np.argmax(delta[:, t-1] * self.A[:, j])
        
        # Backtracking
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[:, -1])
        for t in range(T-2, -1, -1):
            states[t] = psi[states[t+1], t+1]
        
        return states

# Example usage
A = np.array([[0.7, 0.3], [0.4, 0.6]])
B = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
pi = np.array([0.6, 0.4])

hmm = HMM(A, B, pi)
observations = [0, 1, 2, 1]

alpha = hmm.forward(observations)
beta = hmm.backward(observations)
states = hmm.viterbi(observations)

print("Forward probabilities:")
print(alpha)
print("\nBackward probabilities:")
print(beta)
print("\nMost likely state sequence:")
print(states)
```

Slide 12: Inference in Conditional Random Fields

Conditional Random Fields (CRFs) are discriminative probabilistic graphical models often used for sequence labeling tasks. They model the conditional probability of the labels given the observations. Let's implement a linear-chain CRF for named entity recognition.

```python
import numpy as np
from scipy.optimize import minimize

class LinearChainCRF:
    def __init__(self, num_labels, num_features):
        self.num_labels = num_labels
        self.num_features = num_features
        self.weights = np.random.randn(num_labels * num_features + num_labels * num_labels)
    
    def potential(self, x, y, t):
        feature_weights = self.weights[:self.num_labels * self.num_features].reshape(self.num_labels, self.num_features)
        transition_weights = self.weights[self.num_labels * self.num_features:].reshape(self.num_labels, self.num_labels)
        
        emission = np.dot(feature_weights[y[t]], x[t])
        if t == 0:
            transition = 0
        else:
            transition = transition_weights[y[t-1], y[t]]
        return emission + transition
    
    def forward(self, x):
        T = len(x)
        alpha = np.zeros((T, self.num_labels))
        
        for y in range(self.num_labels):
            alpha[0, y] = self.potential(x, [y], 0)
        
        for t in range(1, T):
            for y in range(self.num_labels):
                alpha[t, y] = max(alpha[t-1] + self.potential(x, [y_prev, y], t) for y_prev in range(self.num_labels))
        
        return alpha
    
    def viterbi(self, x):
        T = len(x)
        delta = np.zeros((T, self.num_labels))
        backpointers = np.zeros((T, self.num_labels), dtype=int)
        
        for y in range(self.num_labels):
            delta[0, y] = self.potential(x, [y], 0)
        
        for t in range(1, T):
            for y in range(self.num_labels):
                scores = delta[t-1] + [self.potential(x, [y_prev, y], t) for y_prev in range(self.num_labels)]
                delta[t, y] = max(scores)
                backpointers[t, y] = np.argmax(scores)
        
        # Backtracking
        y = np.zeros(T, dtype=int)
        y[-1] = np.argmax(delta[-1])
        for t in range(T-2, -1, -1):
            y[t] = backpointers[t+1, y[t+1]]
        
        return y
    
    def neg_log_likelihood(self, X, Y):
        total_nll = 0
        for x, y in zip(X, Y):
            alpha = self.forward(x)
            Z = np.logaddexp.reduce(alpha[-1])
            sequence_potential = sum(self.potential(x, y, t) for t in range(len(x)))
            total_nll -= sequence_potential - Z
        return total_nll
    
    def fit(self, X, Y, num_iterations=100):
        def objective(weights):
            self.weights = weights
            return self.neg_log_likelihood(X, Y)
        
        result = minimize(objective, self.weights, method='L-BFGS-B', options={'maxiter': num_iterations})
        self.weights = result.x

# Example usage
X = [np.random.randn(10, 5) for _ in range(3)]  # 3 sequences, each with 10 time steps and 5 features
Y = [np.random.randint(0, 3, 10) for _ in range(3)]  # 3 label sequences, each with 10 time steps and 3 possible labels

crf = LinearChainCRF(num_labels=3, num_features=5)
crf.fit(X, Y)

# Predict on a new sequence
new_x = np.random.randn(10, 5)
predicted_y = crf.viterbi(new_x)
print("Predicted labels:", predicted_y)
```

Slide 13: Inference in Factor Graphs

Factor graphs are a unified representation for various probabilistic graphical models, including Bayesian networks and Markov random fields. They consist of variable nodes and factor nodes. The sum-product algorithm is used for inference in factor graphs.

```python
import numpy as np

class FactorNode:
    def __init__(self, function, variables):
        self.function = function
        self.variables = variables
        self.messages = {v: np.ones_like(v.domain) for v in variables}

class VariableNode:
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain
        self.factors = []
        self.messages = {}

def sum_product(variable_nodes, factor_nodes, max_iterations=10):
    for _ in range(max_iterations):
        # Variable to factor messages
        for v in variable_nodes:
            for f in v.factors:
                message = np.ones_like(v.domain)
                for other_f in v.factors:
                    if other_f != f:
                        message *= v.messages[other_f]
                f.messages[v] = message
        
        # Factor to variable messages
        for f in factor_nodes:
            for v in f.variables:
                message = f.function
                for other_v in f.variables:
                    if other_v != v:
                        message = np.tensordot(message, f.messages[other_v], axes=0)
                axes = tuple(range(len(f.variables) - 1))
                v.messages[f] = np.sum(message, axis=axes)
    
    # Compute marginals
    marginals = {}
    for v in variable_nodes:
        marginal = np.ones_like(v.domain)
        for f in v.factors:
            marginal *= v.messages[f]
        marginals[v.name] = marginal / np.sum(marginal)
    
    return marginals

# Example usage
def factor_function(x, y):
    return np.array([[0.3, 0.7], [0.6, 0.4]])

x = VariableNode("X", np.array([0, 1]))
y = VariableNode("Y", np.array([0, 1]))

f = FactorNode(factor_function, [x, y])

x.factors.append(f)
y.factors.append(f)

marginals = sum_product([x, y], [f])

print("Marginal for X:", marginals["X"])
print("Marginal for Y:", marginals["Y"])
```

Slide 14: Additional Resources

For those interested in diving deeper into probabilistic graphical models and inference techniques, here are some valuable resources:

1. "Probabilistic Graphical Models: Principles and Techniques" by Daphne Koller and Nir Friedman ArXiv: [https://arxiv.org/abs/1301.0608](https://arxiv.org/abs/1301.0608)
2. "Pattern Recognition and Machine Learning" by Christopher Bishop ArXiv: [https://arxiv.org/abs/2103.11695](https://arxiv.org/abs/2103.11695)
3. "An Introduction to Probabilistic Graphical Models" by Michael I. Jordan ArXiv: [https://arxiv.org/abs/1302.6808](https://arxiv.org/abs/1302.6808)
4. "Graphical Models, Exponential Families, and Variational Inference" by Martin J. Wainwright and Michael I. Jordan ArXiv: [https://arxiv.org/abs/0812.4774](https://arxiv.org/abs/0812.4774)
5. "A Tutorial on Energy-Based Learning" by Yann LeCun et al. ArXiv: [https://arxiv.org/abs/2101.05879](https://arxiv.org/abs/2101.05879)

These resources provide in-depth explanations of the concepts covered in this presentation and introduce advanced topics in probabilistic graphical models.

