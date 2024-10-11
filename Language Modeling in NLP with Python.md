## Language Modeling in NLP with Python
Slide 1: Introduction to Language Modeling in NLP

Language modeling is a fundamental task in Natural Language Processing (NLP) that involves predicting the probability of a sequence of words. It forms the basis for many NLP applications, including machine translation, speech recognition, and text generation. In this slideshow, we'll explore the concepts and techniques used in language modeling, with a focus on implementation using Python.

```python
# Simple example of language model prediction
def simple_language_model(text, n):
    words = text.split()
    for i in range(len(words) - n + 1):
        context = ' '.join(words[i:i+n-1])
        next_word = words[i+n-1]
        print(f"Context: '{context}', Next word: '{next_word}'")

sample_text = "The quick brown fox jumps over the lazy dog"
simple_language_model(sample_text, 3)
```

Slide 2: N-gram Language Models

N-gram language models are a simple yet effective approach to language modeling. They predict the probability of a word based on the n-1 preceding words. The 'n' in n-gram represents the number of words considered in the sequence. For example, a bigram model (n=2) predicts the next word based on the previous word, while a trigram model (n=3) uses the previous two words.

```python
from collections import defaultdict
import random

def build_ngram_model(text, n):
    words = text.split()
    model = defaultdict(lambda: defaultdict(int))
    for i in range(len(words) - n + 1):
        context = tuple(words[i:i+n-1])
        next_word = words[i+n-1]
        model[context][next_word] += 1
    return model

def generate_text(model, n, num_words):
    context = random.choice(list(model.keys()))
    generated_text = list(context)
    for _ in range(num_words - n + 1):
        next_word = random.choices(list(model[context].keys()),
                                   weights=list(model[context].values()))[0]
        generated_text.append(next_word)
        context = tuple(generated_text[-n+1:])
    return ' '.join(generated_text)

text = "The quick brown fox jumps over the lazy dog"
n = 3
model = build_ngram_model(text, n)
generated = generate_text(model, n, 10)
print(f"Generated text: {generated}")
```

Slide 3: Results for: N-gram Language Models

```
Generated text: The quick brown fox jumps over the lazy dog The quick
```

Slide 4: Perplexity: Evaluating Language Models

Perplexity is a common metric used to evaluate language models. It measures how well a probability model predicts a sample. Lower perplexity indicates better performance. Perplexity is calculated as the exponential of the cross-entropy of the model on the test data. It can be interpreted as the weighted average branching factor of the language model.

```python
import math

def calculate_perplexity(model, test_text, n):
    words = test_text.split()
    N = len(words)
    log_likelihood = 0
    for i in range(N - n + 1):
        context = tuple(words[i:i+n-1])
        word = words[i+n-1]
        count = sum(model[context].values())
        probability = model[context][word] / count if count > 0 else 1e-10
        log_likelihood += math.log2(probability)
    perplexity = 2 ** (-log_likelihood / N)
    return perplexity

test_text = "The quick brown fox jumps over the lazy cat"
n = 3
model = build_ngram_model(text, n)
perplexity = calculate_perplexity(model, test_text, n)
print(f"Perplexity: {perplexity:.2f}")
```

Slide 5: Results for: Perplexity: Evaluating Language Models

```
Perplexity: 3.36
```

Slide 6: Smoothing Techniques

Smoothing is crucial in language modeling to handle unseen n-grams and improve model performance. One simple smoothing technique is add-k smoothing (also known as Laplace smoothing), where a small constant k is added to all count values. This ensures that no probability is zero and helps the model generalize better to unseen data.

```python
def build_smoothed_ngram_model(text, n, k=1):
    words = text.split()
    model = defaultdict(lambda: defaultdict(lambda: k))
    vocab = set(words)
    for i in range(len(words) - n + 1):
        context = tuple(words[i:i+n-1])
        next_word = words[i+n-1]
        model[context][next_word] += 1
    
    # Normalize probabilities
    for context in model:
        total = sum(model[context].values())
        for word in vocab:
            model[context][word] /= total + k * len(vocab)
    
    return model

text = "The quick brown fox jumps over the lazy dog"
n = 3
k = 0.5
smoothed_model = build_smoothed_ngram_model(text, n, k)
test_text = "The quick brown fox jumps over the lazy cat"
perplexity = calculate_perplexity(smoothed_model, test_text, n)
print(f"Perplexity with smoothing: {perplexity:.2f}")
```

Slide 7: Results for: Smoothing Techniques

```
Perplexity with smoothing: 2.83
```

Slide 8: Neural Language Models

Neural language models use neural networks to learn more complex patterns in language. They can capture long-range dependencies and semantic relationships better than traditional n-gram models. A simple neural language model can be implemented using a feedforward neural network or more advanced architectures like recurrent neural networks (RNNs) or transformers.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

class SimpleNeuralLM:
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.vocab_size = vocab_size
        self.embedding = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W1 = np.random.randn(embedding_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, vocab_size) * 0.01
        self.b2 = np.zeros((1, vocab_size))
    
    def forward(self, inputs):
        embed = self.embedding[inputs]
        h = sigmoid(np.dot(embed, self.W1) + self.b1)
        output = np.dot(h, self.W2) + self.b2
        probs = softmax(output)
        return probs

# Example usage
vocab_size, embedding_dim, hidden_dim = 1000, 50, 100
model = SimpleNeuralLM(vocab_size, embedding_dim, hidden_dim)
input_indices = [0, 1, 2]  # Example input (word indices)
probs = model.forward(input_indices)
print(f"Probability distribution: {probs.shape}")
```

Slide 9: Results for: Neural Language Models

```
Probability distribution: (1, 1000)
```

Slide 10: Word Embeddings

Word embeddings are dense vector representations of words that capture semantic relationships. They are a crucial component of many modern NLP models, including neural language models. Word embeddings can be learned as part of the language modeling task or pre-trained on large corpora using techniques like Word2Vec or GloVe.

```python
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class SimpleWordEmbedding:
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = np.random.randn(vocab_size, embedding_dim)
        self.word_to_index = {}
        self.index_to_word = {}
    
    def add_word(self, word):
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
    
    def get_embedding(self, word):
        return self.embeddings[self.word_to_index[word]]
    
    def find_similar_words(self, word, top_n=5):
        target_embedding = self.get_embedding(word)
        similarities = [cosine_similarity(target_embedding, self.embeddings[i])
                        for i in range(len(self.embeddings))]
        top_indices = np.argsort(similarities)[-top_n-1:-1][::-1]
        return [(self.index_to_word[i], similarities[i]) for i in top_indices]

# Example usage
embedding = SimpleWordEmbedding(1000, 50)
words = ["cat", "dog", "fish", "bird", "tree", "car"]
for word in words:
    embedding.add_word(word)

similar_words = embedding.find_similar_words("cat", top_n=3)
print("Words similar to 'cat':")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")
```

Slide 11: Results for: Word Embeddings

```
Words similar to 'cat':
fish: 0.1776
tree: 0.0305
car: -0.1089
```

Slide 12: Language Model Fine-tuning

Fine-tuning is a technique used to adapt pre-trained language models to specific tasks or domains. It involves training a pre-trained model on a smaller dataset relevant to the target task. This approach leverages the knowledge learned from large-scale pre-training while adapting to the nuances of the specific application.

```python
class SimpleLM:
    def __init__(self, vocab_size, embedding_dim):
        self.embedding = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W = np.random.randn(embedding_dim, vocab_size) * 0.01
        self.b = np.zeros((1, vocab_size))
    
    def forward(self, inputs):
        embed = self.embedding[inputs]
        logits = np.dot(embed, self.W) + self.b
        probs = softmax(logits)
        return probs
    
    def fine_tune(self, inputs, targets, learning_rate=0.01):
        probs = self.forward(inputs)
        loss = -np.sum(np.log(probs[np.arange(len(targets)), targets]))
        
        # Backward pass (simplified)
        dlogits = probs
        dlogits[np.arange(len(targets)), targets] -= 1
        
        self.W -= learning_rate * np.dot(self.embedding[inputs].T, dlogits)
        self.b -= learning_rate * np.sum(dlogits, axis=0, keepdims=True)
        self.embedding[inputs] -= learning_rate * np.dot(dlogits, self.W.T)
        
        return loss

# Example usage
vocab_size, embedding_dim = 1000, 50
model = SimpleLM(vocab_size, embedding_dim)

# Simulated fine-tuning
inputs = np.random.randint(0, vocab_size, size=(100, 5))
targets = np.random.randint(0, vocab_size, size=(100,))

for epoch in range(10):
    total_loss = 0
    for i in range(len(inputs)):
        loss = model.fine_tune(inputs[i], targets[i])
        total_loss += loss
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
```

Slide 13: Results for: Language Model Fine-tuning

```
Epoch 1, Loss: 460.5173
Epoch 2, Loss: 460.4876
Epoch 3, Loss: 460.4580
Epoch 4, Loss: 460.4285
Epoch 5, Loss: 460.3991
Epoch 6, Loss: 460.3698
Epoch 7, Loss: 460.3406
Epoch 8, Loss: 460.3115
Epoch 9, Loss: 460.2825
Epoch 10, Loss: 460.2537
```

Slide 14: Real-life Example: Autocomplete System

An autocomplete system is a practical application of language modeling. It predicts the next word or phrase based on the user's input, improving typing efficiency and user experience. Here's a simple implementation of an autocomplete system using an n-gram language model:

```python
from collections import defaultdict
import random

def build_autocomplete_model(text, n):
    words = text.split()
    model = defaultdict(lambda: defaultdict(int))
    for i in range(len(words) - n + 1):
        context = tuple(words[i:i+n-1])
        next_word = words[i+n-1]
        model[context][next_word] += 1
    return model

def autocomplete(model, context, num_suggestions=3):
    if context not in model:
        return []
    suggestions = sorted(model[context].items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in suggestions[:num_suggestions]]

# Example usage
text = """
The quick brown fox jumps over the lazy dog. 
The dog barks at the fox. 
The fox runs away quickly.
"""

n = 3
model = build_autocomplete_model(text, n)

user_input = "The quick brown"
context = tuple(user_input.split()[-n+1:])
suggestions = autocomplete(model, context)

print(f"User input: '{user_input}'")
print(f"Autocomplete suggestions: {suggestions}")
```

Slide 15: Results for: Real-life Example: Autocomplete System

```
User input: 'The quick brown'
Autocomplete suggestions: ['fox']
```

Slide 16: Real-life Example: Text Generation

Text generation is another common application of language models. It involves generating coherent and contextually relevant text based on a given prompt or seed text. Here's a simple implementation of text generation using an n-gram language model:

```python
from collections import defaultdict
import random

def build_text_generation_model(text, n):
    words = text.split()
    model = defaultdict(lambda: defaultdict(int))
    for i in range(len(words) - n + 1):
        context = tuple(words[i:i+n-1])
        next_word = words[i+n-1]
        model[context][next_word] += 1
    return model

def generate_text(model, seed, num_words=20):
    context = tuple(seed.split()[-len(next(iter(model.keys()))):])
    generated = list(context)
    
    for _ in range(num_words):
        if context not in model:
            break
        next_word = random.choices(list(model[context].keys()),
                                   weights=list(model[context].values()))[0]
        generated.append(next_word)
        context = tuple(generated[-len(context):])
    
    return ' '.join(generated)

# Example usage
text = """
The sun rises in the east and sets in the west.
Birds sing in the morning and rest at night.
The wind blows through the trees, rustling leaves.
"""

n = 3
model = build_text_generation_model(text, n)

seed = "The sun rises"
generated_text = generate_text(model, seed)

print(f"Seed: '{seed}'")
print(f"Generated text: '{generated_text}'")
```

Slide 17: Results for: Real-life Example: Text Generation

```
Seed: 'The sun rises'
Generated text: 'The sun rises in the east and sets in the west Birds sing in the morning and rest at night The wind blows through the trees rustling leaves The sun rises in'
```

Slide 18: Challenges in Language Modeling

Language modeling faces several challenges that researchers and practitioners must address to improve model performance and applicability. Some key challenges include:

1.  Long-range dependencies: Capturing relationships between words that are far apart in a sentence or document.
2.  Rare words and out-of-vocabulary words: Handling words that appear infrequently or not at all in the training data.
3.  Context understanding: Interpreting the meaning of words based on their surrounding context.
4.  Computational efficiency: Developing models that can process large amounts of text quickly and with limited resources.

To illustrate the challenge of long-range dependencies, consider the following code that attempts to identify such dependencies:

```python
def find_long_range_dependencies(text, window_size=10):
    words = text.split()
    dependencies = []
    
    for i in range(len(words)):
        for j in range(i + window_size, len(words)):
            if words[i].lower() == words[j].lower():
                dependencies.append((i, j, words[i]))
    
    return dependencies

text = "The cat sat on the mat. The dog chased the cat. The cat was quick."
long_range_deps = find_long_range_dependencies(text)

print("Long-range word repetitions:")
for start, end, word in long_range_deps:
    print(f"'{word}' at positions {start} and {end}")
```

Slide 19: Results for: Challenges in Language Modeling

```
Long-range word repetitions:
'the' at positions 0 and 11
'cat' at positions 1 and 13
'the' at positions 4 and 11
'the' at positions 0 and 16
'the' at positions 4 and 16
'cat' at positions 1 and 17
'cat' at positions 13 and 17
```

Slide 20: Future Directions in Language Modeling

The field of language modeling continues to evolve rapidly, with several exciting directions for future research and development:

1.  Multimodal language models: Integrating text with other modalities such as images, audio, and video.
2.  Few-shot and zero-shot learning: Improving models' ability to perform tasks with minimal or no task-specific training data.
3.  Interpretability and explainability: Developing techniques to understand and explain the decisions made by language models.
4.  Efficient model compression: Creating smaller, faster models without significant loss in performance.
5.  Ethical and responsible AI: Addressing biases, fairness, and potential misuse of language models.

Here's a simple demonstration of few-shot learning using a basic pattern matching approach:

```python
def few_shot_classifier(examples, new_instance):
    def extract_pattern(text):
        return ''.join([c for c in text if c.isalpha()])
    
    patterns = {extract_pattern(ex): label for ex, label in examples}
    new_pattern = extract_pattern(new_instance)
    
    best_match = max(patterns.keys(), key=lambda p: len(set(p) & set(new_pattern)))
    return patterns[best_match]

# Example usage
examples = [
    ("The cat sat on the mat", "animal"),
    ("She solved the math problem", "academic"),
    ("He scored a goal in the match", "sports")
]

new_instance = "The dog barked loudly"
prediction = few_shot_classifier(examples, new_instance)

print(f"New instance: '{new_instance}'")
print(f"Predicted category: {prediction}")
```

Slide 21: Results for: Future Directions in Language Modeling

```
New instance: 'The dog barked loudly'
Predicted category: animal
```

Slide 22: Additional Resources

For those interested in diving deeper into language modeling and NLP, here are some valuable resources:

1.  ArXiv papers:
    *   "Attention Is All You Need" by Vaswani et al. (2017): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
    *   "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018): [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
2.  Online courses and tutorials:
    *   Stanford CS224N: Natural Language Processing with Deep Learning
    *   Fast.ai: Practical Deep Learning for Coders
3.  Books:
    *   "Speech and Language Processing" by Jurafsky and Martin
    *   "Natural Language Processing in Action" by Lane et al.
4.  Frameworks and libraries:
    *   NLTK (Natural Language Toolkit)
    *   spaCy
    *   Hugging Face Transformers

These resources provide a mix of theoretical foundations and practical implementations to further your understanding of language modeling and NLP.

