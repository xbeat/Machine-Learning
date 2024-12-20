## Building Hidden Markov Models from Scratch in Python
Slide 1: Introduction to Hidden Markov Models

Hidden Markov Models (HMMs) are powerful statistical tools used for modeling sequential data. They're particularly useful in speech recognition, natural language processing, and bioinformatics. This presentation will guide you through building HMMs from scratch using Python.

```python
import numpy as np

class HiddenMarkovModel:
    def __init__(self, states, observations):
        self.states = states
        self.observations = observations
        self.n_states = len(states)
        self.n_observations = len(observations)
```

Slide 2: Components of an HMM

An HMM consists of three main components: the initial state probabilities, transition probabilities, and emission probabilities. These components define the behavior of the model and how it generates sequences of observations.

```python
class HiddenMarkovModel:
    def __init__(self, states, observations):
        # ... (previous code) ...
        self.initial_probs = np.zeros(self.n_states)
        self.transition_probs = np.zeros((self.n_states, self.n_states))
        self.emission_probs = np.zeros((self.n_states, self.n_observations))
```

Slide 3: Initializing HMM Parameters

To start building our HMM, we need to initialize its parameters. We'll use random initialization for this example, but in practice, these values are often estimated from training data or set based on domain knowledge.

```python
class HiddenMarkovModel:
    def __init__(self, states, observations):
        # ... (previous code) ...
        self.initialize_parameters()

    def initialize_parameters(self):
        self.initial_probs = np.random.dirichlet(np.ones(self.n_states))
        self.transition_probs = np.random.dirichlet(np.ones(self.n_states), size=self.n_states)
        self.emission_probs = np.random.dirichlet(np.ones(self.n_observations), size=self.n_states)
```

Slide 4: Forward Algorithm

The forward algorithm is used to compute the probability of an observation sequence given the model. It efficiently calculates this probability by using dynamic programming.

```python
def forward(self, observations):
    T = len(observations)
    alpha = np.zeros((T, self.n_states))
    
    # Initialize
    alpha[0] = self.initial_probs * self.emission_probs[:, observations[0]]
    
    # Iterate
    for t in range(1, T):
        for j in range(self.n_states):
            alpha[t, j] = np.sum(alpha[t-1] * self.transition_probs[:, j]) * self.emission_probs[j, observations[t]]
    
    return alpha, np.sum(alpha[-1])
```

Slide 5: Backward Algorithm

The backward algorithm is similar to the forward algorithm but works in reverse. It's used in conjunction with the forward algorithm for various HMM tasks, such as parameter estimation and decoding.

```python
def backward(self, observations):
    T = len(observations)
    beta = np.zeros((T, self.n_states))
    
    # Initialize
    beta[-1] = 1
    
    # Iterate
    for t in range(T-2, -1, -1):
        for i in range(self.n_states):
            beta[t, i] = np.sum(self.transition_probs[i] * self.emission_probs[:, observations[t+1]] * beta[t+1])
    
    return beta
```

Slide 6: Viterbi Algorithm

The Viterbi algorithm is used to find the most likely sequence of hidden states given a sequence of observations. This is often referred to as decoding the hidden state sequence.

```python
def viterbi(self, observations):
    T = len(observations)
    viterbi = np.zeros((T, self.n_states))
    backpointer = np.zeros((T, self.n_states), dtype=int)
    
    # Initialize
    viterbi[0] = self.initial_probs * self.emission_probs[:, observations[0]]
    
    # Iterate
    for t in range(1, T):
        for j in range(self.n_states):
            prob = viterbi[t-1] * self.transition_probs[:, j] * self.emission_probs[j, observations[t]]
            viterbi[t, j] = np.max(prob)
            backpointer[t, j] = np.argmax(prob)
    
    # Backtrack
    path = [np.argmax(viterbi[-1])]
    for t in range(T-1, 0, -1):
        path.append(backpointer[t, path[-1]])
    
    return list(reversed(path))
```

Slide 7: Baum-Welch Algorithm (Part 1: E-step)

The Baum-Welch algorithm is used for training HMMs. It's an iterative method that uses the Expectation-Maximization (EM) algorithm to estimate the model parameters. This slide covers the Expectation step.

```python
def expectation_step(self, observations):
    T = len(observations)
    alpha, _ = self.forward(observations)
    beta = self.backward(observations)
    
    gamma = alpha * beta / np.sum(alpha * beta, axis=1, keepdims=True)
    
    xi = np.zeros((T-1, self.n_states, self.n_states))
    for t in range(T-1):
        denominator = np.sum(alpha[t] * self.transition_probs * self.emission_probs[:, observations[t+1]] * beta[t+1])
        for i in range(self.n_states):
            numerator = alpha[t, i] * self.transition_probs[i] * self.emission_probs[:, observations[t+1]] * beta[t+1]
            xi[t, i] = numerator / denominator
    
    return gamma, xi
```

Slide 8: Baum-Welch Algorithm (Part 2: M-step)

The Maximization step of the Baum-Welch algorithm updates the model parameters based on the expectations computed in the E-step.

```python
def maximization_step(self, observations, gamma, xi):
    T = len(observations)
    
    # Update initial probabilities
    self.initial_probs = gamma[0]
    
    # Update transition probabilities
    self.transition_probs = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, np.newaxis]
    
    # Update emission probabilities
    for j in range(self.n_states):
        for k in range(self.n_observations):
            self.emission_probs[j, k] = np.sum(gamma[observations == k, j]) / np.sum(gamma[:, j])
```

Slide 9: Training the HMM

Now that we have implemented the Baum-Welch algorithm, we can use it to train our HMM on a sequence of observations.

```python
def train(self, observations, n_iterations=100):
    for _ in range(n_iterations):
        gamma, xi = self.expectation_step(observations)
        self.maximization_step(observations, gamma, xi)

# Example usage
states = ['Sunny', 'Rainy']
observations = ['Dry', 'Wet']
hmm = HiddenMarkovModel(states, observations)

# Convert observations to numerical form
obs_sequence = [0, 1, 1, 0, 0, 1]  # 0: Dry, 1: Wet

hmm.train(obs_sequence)
```

Slide 10: Generating Sequences

Once we have a trained HMM, we can use it to generate new sequences of observations. This can be useful for simulation or prediction tasks.

```python
def generate_sequence(self, length):
    current_state = np.random.choice(self.n_states, p=self.initial_probs)
    sequence = []
    
    for _ in range(length):
        observation = np.random.choice(self.n_observations, p=self.emission_probs[current_state])
        sequence.append(observation)
        current_state = np.random.choice(self.n_states, p=self.transition_probs[current_state])
    
    return sequence

# Generate a sequence of 10 observations
generated_sequence = hmm.generate_sequence(10)
print("Generated sequence:", generated_sequence)
```

Slide 11: Real-Life Example: Weather Prediction

HMMs can be used for weather prediction. In this example, we'll use a simple model with two hidden states (Sunny and Rainy) and two observations (Dry and Wet).

```python
# Define states and observations
states = ['Sunny', 'Rainy']
observations = ['Dry', 'Wet']

# Create and train the HMM
weather_hmm = HiddenMarkovModel(states, observations)

# Historical weather data (0: Dry, 1: Wet)
historical_data = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

weather_hmm.train(historical_data)

# Predict the weather for the next 5 days
predicted_weather = weather_hmm.generate_sequence(5)
print("Predicted weather for the next 5 days:", [observations[i] for i in predicted_weather])
```

Slide 12: Real-Life Example: Part-of-Speech Tagging

HMMs are widely used in natural language processing for tasks like part-of-speech tagging. Here's a simple example:

```python
# Define states (parts of speech) and observations (words)
states = ['Noun', 'Verb', 'Adjective']
observations = ['dog', 'cat', 'run', 'jump', 'big', 'small']

pos_hmm = HiddenMarkovModel(states, observations)

# Training data (simplified)
training_data = [0, 2, 1, 3, 4, 0]  # Corresponding to: dog big run jump small cat

pos_hmm.train(training_data)

# Tag a new sentence
sentence = [0, 1, 4]  # dog run big
tagged_sentence = pos_hmm.viterbi(sentence)
print("Tagged sentence:", [states[i] for i in tagged_sentence])
```

Slide 13: Evaluating HMM Performance

To assess how well our HMM is performing, we can use metrics such as log-likelihood or accuracy (for tasks like prediction or classification).

```python
def log_likelihood(self, observations):
    _, likelihood = self.forward(observations)
    return np.log(likelihood)

# Example usage
test_sequence = [0, 1, 0, 1, 1]
log_likelihood = hmm.log_likelihood(test_sequence)
print(f"Log-likelihood of test sequence: {log_likelihood}")

# For classification tasks, you can compare predicted vs. true labels
true_labels = [0, 1, 1, 0, 1]
predicted_labels = hmm.viterbi(test_sequence)
accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels)
print(f"Accuracy: {accuracy}")
```

Slide 14: Challenges and Considerations

When working with HMMs, it's important to consider:

1. Initialization: Random initialization may lead to suboptimal results. Consider using domain knowledge or multiple random starts.
2. Model selection: Choosing the right number of hidden states is crucial and often requires experimentation.
3. Computational complexity: HMMs can be computationally expensive for large state spaces or long sequences.
4. Local optima: The Baum-Welch algorithm may converge to local optima. Multiple runs with different initializations can help.

```python
def train_with_multiple_starts(self, observations, n_starts=5, n_iterations=100):
    best_likelihood = float('-inf')
    best_params = None
    
    for _ in range(n_starts):
        self.initialize_parameters()
        self.train(observations, n_iterations)
        likelihood = self.log_likelihood(observations)
        
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_params = (self.initial_probs.(), self.transition_probs.(), self.emission_probs.())
    
    self.initial_probs, self.transition_probs, self.emission_probs = best_params
```

Slide 15: Additional Resources

For further exploration of Hidden Markov Models and their applications, consider the following resources:

1. "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition" by Lawrence R. Rabiner (1989) ArXiv: [https://arxiv.org/abs/cs/0306033](https://arxiv.org/abs/cs/0306033)
2. "An Introduction to Hidden Markov Models" by L.R. Rabiner and B.H. Juang (1986) IEEE ASSP Magazine
3. "Biological Sequence Analysis: Probabilistic Models of Proteins and Nucleic Acids" by R. Durbin, S. Eddy, A. Krogh, and G. Mitchison (1998) Cambridge University Press

These resources provide in-depth explanations of HMM theory and applications, which can help you further develop your understanding and implementation skills.

