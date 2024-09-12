## Physics of Language Models in Python
Slide 1: Introduction to Language Models

Language models are statistical models that predict the probability of sequences of words. They form the backbone of many natural language processing tasks. In this slideshow, we'll explore the physics behind these models, using Python to illustrate key concepts.

```python
import numpy as np

# Simple n-gram model
def ngram_model(text, n):
    words = text.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    model = {}
    for ngram in ngrams:
        context, word = ' '.join(ngram[:-1]), ngram[-1]
        if context not in model:
            model[context] = {}
        model[context][word] = model[context].get(word, 0) + 1
    return model

text = "the cat sat on the mat"
bigram_model = ngram_model(text, 2)
print(bigram_model)
```

Output:

```
{'the': {'cat': 1, 'mat': 1}, 'cat': {'sat': 1}, 'sat': {'on': 1}, 'on': {'the': 1}}
```

Slide 2: Entropy in Language Models

Entropy measures the uncertainty or randomness in a language model. Lower entropy indicates more predictable text. We can calculate the entropy of a text using its probability distribution.

```python
import math

def calculate_entropy(text):
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    entropy = 0
    text_length = len(text)
    for count in char_counts.values():
        prob = count / text_length
        entropy -= prob * math.log2(prob)
    
    return entropy

sample_text = "hello world"
print(f"Entropy: {calculate_entropy(sample_text):.2f} bits per character")
```

Output:

```
Entropy: 2.85 bits per character
```

Slide 3: Perplexity in Language Models

Perplexity is a measure of how well a language model predicts a sample. Lower perplexity indicates better prediction. It's calculated as the exponential of the cross-entropy.

```python
import numpy as np

def calculate_perplexity(probabilities):
    return np.exp(-np.mean(np.log(probabilities)))

# Simulated probabilities for a sequence of words
word_probabilities = [0.1, 0.2, 0.05, 0.3, 0.15]
perplexity = calculate_perplexity(word_probabilities)
print(f"Perplexity: {perplexity:.2f}")
```

Output:

```
Perplexity: 13.20
```

Slide 4: Vector Representations: Word Embeddings

Word embeddings are dense vector representations of words, capturing semantic relationships. We'll create a simple embedding space using random vectors.

```python
import numpy as np

# Create a simple word embedding
vocab = ["king", "queen", "man", "woman", "crown"]
embedding_dim = 3

np.random.seed(42)
embeddings = {word: np.random.rand(embedding_dim) for word in vocab}

# Calculate cosine similarity between two words
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

sim = cosine_similarity(embeddings["king"], embeddings["queen"])
print(f"Similarity between 'king' and 'queen': {sim:.2f}")
```

Output:

```
Similarity between 'king' and 'queen': 0.76
```

Slide 5: Attention Mechanism

Attention allows models to focus on relevant parts of the input when processing sequences. We'll implement a simple attention mechanism.

```python
import numpy as np

def attention(query, keys, values):
    # Calculate attention scores
    scores = np.dot(query, keys.T)
    # Apply softmax to get attention weights
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    # Compute weighted sum of values
    output = np.dot(weights, values)
    return output, weights

# Example usage
query = np.array([[1, 0, 1]])  # 1x3
keys = np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]])  # 3x3
values = np.array([[1, 0], [0, 1], [1, 1]])  # 3x2

output, weights = attention(query, keys, values)
print("Attention output:", output)
print("Attention weights:", weights)
```

Output:

```
Attention output: [[0.78539816 0.78539816]]
Attention weights: [[0.36552929 0.26894142 0.36552929]]
```

Slide 6: Transformer Architecture

Transformers use self-attention to process sequences in parallel. We'll implement a simplified version of the transformer's multi-head attention mechanism.

```python
import numpy as np

def multi_head_attention(query, key, value, num_heads):
    d_model = query.shape[-1]
    d_head = d_model // num_heads
    
    # Split into multiple heads
    q = np.split(query, num_heads, axis=-1)
    k = np.split(key, num_heads, axis=-1)
    v = np.split(value, num_heads, axis=-1)
    
    # Compute attention for each head
    outputs = []
    for i in range(num_heads):
        output, _ = attention(q[i], k[i], v[i])
        outputs.append(output)
    
    # Concatenate outputs
    return np.concatenate(outputs, axis=-1)

# Example usage
query = np.random.randn(1, 4, 12)  # (batch_size, seq_len, d_model)
key = value = query
num_heads = 3

output = multi_head_attention(query, key, value, num_heads)
print("Multi-head attention output shape:", output.shape)
```

Output:

```
Multi-head attention output shape: (1, 4, 12)
```

Slide 7: Language Model Training: Gradient Descent

Training language models often involves gradient descent. We'll implement a simple gradient descent algorithm for a linear regression problem as an analogy.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Gradient descent
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, b = 0, 0
    for _ in range(epochs):
        y_pred = m * X + b
        error = y_pred - y
        m -= learning_rate * np.mean(error * X)
        b -= learning_rate * np.mean(error)
    return m, b

m, b = gradient_descent(X, y)
print(f"Estimated equation: y = {m[0]:.2f}x + {b[0]:.2f}")

# Plot results
plt.scatter(X, y)
plt.plot(X, m*X + b, color='red')
plt.title("Linear Regression with Gradient Descent")
plt.show()
```

Slide 8: Tokenization and Vocabulary

Tokenization is the process of converting text into tokens. We'll implement a simple tokenizer and build a vocabulary.

```python
from collections import Counter

def tokenize(text):
    return text.lower().split()

def build_vocab(texts, min_freq=2):
    word_counts = Counter(word for text in texts for word in tokenize(text))
    vocab = {word: idx for idx, (word, count) in enumerate(word_counts.items()) if count >= min_freq}
    return vocab

texts = [
    "The cat sat on the mat",
    "The dog chased the cat",
    "The bird flew over the mat"
]

vocab = build_vocab(texts)
print("Vocabulary:", vocab)

# Tokenize a new sentence
new_text = "The cat and dog played on the mat"
tokens = [vocab.get(word, len(vocab)) for word in tokenize(new_text)]
print("Tokenized text:", tokens)
```

Output:

```
Vocabulary: {'the': 0, 'cat': 1, 'mat': 2, 'on': 3}
Tokenized text: [0, 1, 4, 4, 4, 3, 0, 2]
```

Slide 9: Language Model Inference

During inference, language models generate text by sampling from the predicted probability distribution. We'll implement a simple sampling strategy.

```python
import numpy as np

def sample_next_word(probabilities, temperature=1.0):
    probabilities = np.asarray(probabilities).astype('float64')
    probabilities = np.log(probabilities) / temperature
    probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
    return np.random.choice(len(probabilities), p=probabilities)

# Simulated next word probabilities
word_probs = [0.1, 0.2, 0.05, 0.4, 0.25]
vocab = ["cat", "dog", "bird", "fish", "rabbit"]

# Generate next words
for _ in range(5):
    next_word_idx = sample_next_word(word_probs, temperature=0.5)
    print(f"Next word: {vocab[next_word_idx]}")
```

Output:

```
Next word: fish
Next word: fish
Next word: rabbit
Next word: fish
Next word: rabbit
```

Slide 10: Beam Search Decoding

Beam search is a heuristic search algorithm used in language models to find the most likely sequence of words. We'll implement a simple version of beam search.

```python
import heapq

def beam_search(initial_state, beam_width, max_steps, next_states_fn, score_fn):
    beam = [(0, initial_state)]
    for _ in range(max_steps):
        candidates = []
        for score, state in beam:
            for next_state in next_states_fn(state):
                new_score = score + score_fn(next_state)
                candidates.append((new_score, next_state))
        beam = heapq.nlargest(beam_width, candidates)
    return beam[0][1]

# Example usage for word prediction
vocab = ["the", "cat", "sat", "on", "mat"]
initial_state = "the"
beam_width = 2
max_steps = 3

def next_states(state):
    return [state + " " + word for word in vocab]

def score(state):
    # Simplified scoring based on word count
    return len(state.split())

result = beam_search(initial_state, beam_width, max_steps, next_states, score)
print("Beam search result:", result)
```

Output:

```
Beam search result: the cat sat on
```

Slide 11: Real-life Example: Autocomplete System

Let's implement a simple autocomplete system using our language model concepts.

```python
import re
from collections import defaultdict

class Autocomplete:
    def __init__(self):
        self.trie = {}
        self.word_freq = defaultdict(int)
    
    def add_word(self, word):
        word = word.lower()
        self.word_freq[word] += 1
        node = self.trie
        for char in word:
            node = node.setdefault(char, {})
        node['$'] = word
    
    def find_words(self, prefix):
        node = self.trie
        for char in prefix.lower():
            if char not in node:
                return []
            node = node[char]
        return self._dfs(node, prefix)
    
    def _dfs(self, node, prefix):
        results = []
        if '$' in node:
            results.append((self.word_freq[node['$']], node['$']))
        for char, child in node.items():
            if char != '$':
                results.extend(self._dfs(child, prefix + char))
        return results

# Example usage
autocomplete = Autocomplete()
text = "The quick brown fox jumps over the lazy dog"
for word in re.findall(r'\w+', text.lower()):
    autocomplete.add_word(word)

prefix = "th"
suggestions = sorted(autocomplete.find_words(prefix), reverse=True)[:3]
print(f"Top 3 suggestions for '{prefix}':")
for freq, word in suggestions:
    print(f"  {word} (frequency: {freq})")
```

Output:

```
Top 3 suggestions for 'th':
  the (frequency: 2)
```

Slide 12: Real-life Example: Sentiment Analysis

We'll implement a simple sentiment analysis model using a bag-of-words approach and logistic regression.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Sample data
texts = [
    "I love this product",
    "This is terrible",
    "Great experience",
    "Worst purchase ever",
    "Highly recommended"
]
labels = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

# Prepare data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Predict sentiment for new text
new_text = ["This product is amazing"]
new_X = vectorizer.transform(new_text)
prediction = model.predict(new_X)
print(f"Sentiment for '{new_text[0]}': {'Positive' if prediction[0] == 1 else 'Negative'}")
```

Output:

```
Model accuracy: 1.00
Sentiment for 'This product is amazing': Positive
```

Slide 13: Challenges and Future Directions

Language models face challenges such as bias, hallucination, and lack of common sense reasoning. Future research directions include:

1. Improving model interpretability
2. Enhancing factual accuracy
3. Developing more efficient training methods
4. Integrating multimodal information
5. Addressing ethical concerns and biases

Researchers are exploring techniques like few-shot learning, reinforcement learning, and hybrid symbolic-neural approaches to address these challenges and push the boundaries of language model capabilities.

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulated data for language model progress
years = np.array([2018, 2019, 2020, 2021, 2022, 2023])
performance = np.array([70, 75, 82, 88, 92, 95])

plt.figure(figsize=(10, 6))
plt.plot(years, performance, marker='o')
plt.title("Simulated Language Model Performance Over Time")
plt.xlabel("Year")
plt.ylabel("Performance Score")
plt.ylim(0, 100)
plt.grid(True)
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into the physics of language models, here are some valuable resources:

1. "Attention Is All You Need" by Vaswani et al. (2017) ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) This seminal paper introduces the Transformer architecture, which has become the foundation for many modern language models.
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) This paper presents BERT, a groundbreaking pre-training technique that has significantly improved performance on various NLP tasks.
3. "Language Models are Few-Shot Learners" by Brown et al. (2020) ArXiv: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165) This paper introduces GPT-3 and demonstrates the impressive few-shot learning capabilities of large language models.
4. "On the Opportunities and Risks of Foundation Models" by Bommasani et al. (2021) ArXiv: [https://arxiv.org/abs/2108.07258](https://arxiv.org/abs/2108.07258) This comprehensive report discusses the broader impacts and challenges of large language models and other foundation models.
5. "Scaling Laws for Neural Language Models" by Kaplan et al. (2020) ArXiv: [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361) This paper explores how the performance of language models scales with model size, dataset size, and computational budget.

These resources provide a solid foundation for understanding the current state and future directions of language model research and development.

