## Context-Independent Word Embeddings in Python
Slide 1:

Introduction to Context-Independent Word Embeddings

Context-independent word embeddings are fixed vector representations of words, capturing semantic meaning regardless of surrounding context. These models transform words into dense numerical vectors, enabling machines to process and understand language more effectively.

```python
import numpy as np

# Example of a simple word embedding
word_embedding = {
    "cat": np.array([0.2, -0.4, 0.7]),
    "dog": np.array([0.1, -0.3, 0.8]),
    "pet": np.array([0.15, -0.35, 0.75])
}

# Calculate similarity between words
similarity = np.dot(word_embedding["cat"], word_embedding["dog"])
print(f"Similarity between 'cat' and 'dog': {similarity}")
```

Slide 2:

One-Hot Encoding: The Baseline

One-hot encoding is the simplest form of word representation. It creates a binary vector for each word, with a 1 in the position corresponding to the word and 0s elsewhere. This method is inefficient for large vocabularies and doesn't capture semantic relationships.

```python
def one_hot_encode(word, vocabulary):
    vector = [0] * len(vocabulary)
    vector[vocabulary.index(word)] = 1
    return vector

vocabulary = ["cat", "dog", "fish", "bird"]
encoded_word = one_hot_encode("dog", vocabulary)
print(f"One-hot encoding of 'dog': {encoded_word}")
```

Slide 3:

Count-Based Methods: Co-occurrence Matrix

Count-based methods create word embeddings by analyzing how often words co-occur in a large corpus. A co-occurrence matrix is built, where each cell represents the frequency of two words appearing together within a defined context window.

```python
from collections import defaultdict

def build_co_occurrence_matrix(corpus, window_size=2):
    vocab = set(corpus)
    co_occurrence = defaultdict(lambda: defaultdict(int))
    
    for i, word in enumerate(corpus):
        for j in range(max(0, i - window_size), min(len(corpus), i + window_size + 1)):
            if i != j:
                co_occurrence[word][corpus[j]] += 1
    
    return co_occurrence

corpus = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
co_occurrence_matrix = build_co_occurrence_matrix(corpus)
print(co_occurrence_matrix["the"])
```

Slide 4:

TF-IDF: Term Frequency-Inverse Document Frequency

TF-IDF is a statistical measure used to evaluate the importance of a word in a document within a collection. It combines the frequency of a term in a document with its inverse frequency across the entire corpus, highlighting words that are characteristic of a particular document.

```python
import math
from collections import Counter

def compute_tf_idf(documents):
    word_doc_freq = Counter()
    doc_word_count = []
    
    for doc in documents:
        words = doc.split()
        word_doc_freq.update(set(words))
        doc_word_count.append(Counter(words))
    
    tf_idf = {}
    num_docs = len(documents)
    
    for doc_id, word_counts in enumerate(doc_word_count):
        tf_idf[doc_id] = {}
        for word, count in word_counts.items():
            tf = count / len(word_counts)
            idf = math.log(num_docs / word_doc_freq[word])
            tf_idf[doc_id][word] = tf * idf
    
    return tf_idf

documents = [
    "The cat sits on the mat",
    "The dog runs in the park",
    "Cats and dogs are pets"
]

tf_idf_scores = compute_tf_idf(documents)
print(f"TF-IDF scores for document 0: {tf_idf_scores[0]}")
```

Slide 5:

Word2Vec: Skip-gram Model

Word2Vec is a popular word embedding model that learns vector representations of words by predicting context words given a target word (skip-gram) or vice versa. The skip-gram model is particularly effective for rare words and small datasets.

```python
import numpy as np
from sklearn.preprocessing import normalize

def skipgram_model(target_word, context_words, word_to_index, embed_dim):
    target_index = word_to_index[target_word]
    context_indices = [word_to_index[word] for word in context_words]
    
    # Initialize embeddings randomly
    word_embeddings = np.random.rand(len(word_to_index), embed_dim)
    context_embeddings = np.random.rand(len(word_to_index), embed_dim)
    
    # Simple training loop (in practice, use negative sampling and many iterations)
    learning_rate = 0.01
    for context_index in context_indices:
        pred = np.dot(word_embeddings[target_index], context_embeddings[context_index])
        error = pred - 1  # Simplified error calculation
        
        # Update embeddings
        word_embeddings[target_index] -= learning_rate * error * context_embeddings[context_index]
        context_embeddings[context_index] -= learning_rate * error * word_embeddings[target_index]
    
    # Normalize embeddings
    word_embeddings = normalize(word_embeddings)
    
    return word_embeddings

# Example usage
vocab = ["cat", "dog", "pet", "animal", "cute"]
word_to_index = {word: i for i, word in enumerate(vocab)}
target_word = "cat"
context_words = ["pet", "animal", "cute"]

embeddings = skipgram_model(target_word, context_words, word_to_index, embed_dim=3)
print(f"Learned embedding for 'cat': {embeddings[word_to_index['cat']]}")
```

Slide 6:

Word2Vec: Continuous Bag of Words (CBOW)

The Continuous Bag of Words (CBOW) model is another architecture of Word2Vec. Unlike skip-gram, CBOW predicts a target word given its context words. This model is faster to train and performs well for frequent words.

```python
import numpy as np
from sklearn.preprocessing import normalize

def cbow_model(context_words, target_word, word_to_index, embed_dim):
    context_indices = [word_to_index[word] for word in context_words]
    target_index = word_to_index[target_word]
    
    # Initialize embeddings randomly
    word_embeddings = np.random.rand(len(word_to_index), embed_dim)
    output_weights = np.random.rand(len(word_to_index), embed_dim)
    
    # Simple training loop
    learning_rate = 0.01
    context_vector = np.mean([word_embeddings[i] for i in context_indices], axis=0)
    
    pred = np.dot(context_vector, output_weights[target_index])
    error = pred - 1  # Simplified error calculation
    
    # Update embeddings
    for context_index in context_indices:
        word_embeddings[context_index] -= learning_rate * error * output_weights[target_index] / len(context_indices)
    output_weights[target_index] -= learning_rate * error * context_vector
    
    # Normalize embeddings
    word_embeddings = normalize(word_embeddings)
    
    return word_embeddings

# Example usage
vocab = ["cat", "dog", "pet", "animal", "cute"]
word_to_index = {word: i for i, word in enumerate(vocab)}
context_words = ["pet", "animal", "cute"]
target_word = "cat"

embeddings = cbow_model(context_words, target_word, word_to_index, embed_dim=3)
print(f"Learned embedding for 'cat': {embeddings[word_to_index['cat']]}")
```

Slide 7:

GloVe: Global Vectors for Word Representation

GloVe combines the advantages of count-based methods (like co-occurrence matrices) and predictive methods (like Word2Vec). It creates word embeddings by factorizing the logarithm of the word co-occurrence matrix, capturing both local and global statistical information.

```python
import numpy as np
from sklearn.preprocessing import normalize

def simplified_glove(co_occurrence_matrix, vocab, embed_dim, num_iterations=50):
    vocab_size = len(vocab)
    word_vectors = np.random.rand(vocab_size, embed_dim)
    context_vectors = np.random.rand(vocab_size, embed_dim)
    
    learning_rate = 0.05
    
    for _ in range(num_iterations):
        for i in range(vocab_size):
            for j in range(vocab_size):
                co_occurrence = co_occurrence_matrix[i][j]
                if co_occurrence > 0:
                    weight = (co_occurrence / 100) ** 0.75 if co_occurrence < 100 else 1
                    diff = np.dot(word_vectors[i], context_vectors[j]) - np.log(co_occurrence)
                    
                    grad = diff * weight
                    word_vectors[i] -= learning_rate * grad * context_vectors[j]
                    context_vectors[j] -= learning_rate * grad * word_vectors[i]
    
    return normalize(word_vectors + context_vectors)

# Example usage
vocab = ["cat", "dog", "pet", "animal", "cute"]
co_occurrence_matrix = np.array([
    [0, 2, 5, 3, 1],
    [2, 0, 4, 3, 1],
    [5, 4, 0, 6, 2],
    [3, 3, 6, 0, 1],
    [1, 1, 2, 1, 0]
])

embeddings = simplified_glove(co_occurrence_matrix, vocab, embed_dim=3)
print(f"GloVe embedding for 'cat': {embeddings[0]}")
```

Slide 8:

FastText: Subword Information

FastText extends the Word2Vec model by incorporating subword information. It represents each word as a bag of character n-grams, allowing the model to generate embeddings for out-of-vocabulary words and capture morphological information.

```python
import numpy as np
from sklearn.preprocessing import normalize

def get_ngrams(word, n_min=3, n_max=6):
    ngrams = []
    word = "<" + word + ">"
    for n in range(n_min, min(len(word), n_max) + 1):
        for i in range(len(word) - n + 1):
            ngrams.append(word[i:i+n])
    return ngrams

def fasttext_model(word, context_words, word_to_index, ngram_to_index, embed_dim):
    word_ngrams = get_ngrams(word)
    context_ngrams = [ngram for context_word in context_words for ngram in get_ngrams(context_word)]
    
    # Initialize embeddings randomly
    ngram_embeddings = np.random.rand(len(ngram_to_index), embed_dim)
    
    # Simple training loop
    learning_rate = 0.01
    word_vector = np.mean([ngram_embeddings[ngram_to_index[ng]] for ng in word_ngrams], axis=0)
    
    for context_ngram in context_ngrams:
        context_vector = ngram_embeddings[ngram_to_index[context_ngram]]
        pred = np.dot(word_vector, context_vector)
        error = pred - 1  # Simplified error calculation
        
        # Update embeddings
        for ng in word_ngrams:
            ngram_embeddings[ngram_to_index[ng]] -= learning_rate * error * context_vector / len(word_ngrams)
        ngram_embeddings[ngram_to_index[context_ngram]] -= learning_rate * error * word_vector
    
    # Normalize embeddings
    ngram_embeddings = normalize(ngram_embeddings)
    
    return ngram_embeddings

# Example usage
vocab = ["cat", "dog", "pet", "animal", "cute"]
word_to_index = {word: i for i, word in enumerate(vocab)}
all_ngrams = set(ng for word in vocab for ng in get_ngrams(word))
ngram_to_index = {ng: i for i, ng in enumerate(all_ngrams)}

word = "cat"
context_words = ["pet", "animal", "cute"]

embeddings = fasttext_model(word, context_words, word_to_index, ngram_to_index, embed_dim=3)
print(f"FastText ngram embeddings shape: {embeddings.shape}")
```

Slide 9:

ELMo: Embeddings from Language Models

ELMo (Embeddings from Language Models) generates contextual word embeddings using a deep bidirectional language model. Unlike previous models, ELMo produces dynamic embeddings that change based on the context in which a word appears.

```python
import numpy as np
import torch
import torch.nn as nn

class SimplifiedELMo(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm_forward = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.lstm_backward = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        embedded = self.embedding(x)
        forward_out, _ = self.lstm_forward(embedded)
        backward_out, _ = self.lstm_backward(torch.flip(embedded, [1]))
        backward_out = torch.flip(backward_out, [1])
        return torch.cat((forward_out, backward_out), dim=-1)

# Example usage
vocab = ["the", "cat", "sat", "on", "mat"]
word_to_index = {word: i for i, word in enumerate(vocab)}
sentence = torch.tensor([word_to_index[w] for w in ["the", "cat", "sat"]])

model = SimplifiedELMo(len(vocab), embed_dim=10, hidden_dim=20)
elmo_embeddings = model(sentence.unsqueeze(0))

print(f"ELMo embeddings shape: {elmo_embeddings.shape}")
print(f"Contextual embedding for 'cat': {elmo_embeddings[0, 1]}")
```

Slide 10:

Real-Life Example: Sentiment Analysis

Word embeddings are crucial for sentiment analysis tasks. This example demonstrates how to use pre-trained word embeddings for classifying movie reviews as positive or negative.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Simulated pre-trained word embeddings
word_embeddings = {
    "great": np.array([0.2, 0.3, 0.1]),
    "awesome": np.array([0.3, 0.4, 0.2]),
    "terrible": np.array([-0.2, -0.3, -0.1]),
    "boring": np.array([-0.1, -0.2, -0.3]),
    "movie": np.array([0.0, 0.1, 0.0])
}

# Simulated dataset
reviews = [
    "great awesome movie",
    "terrible boring movie",
    "awesome movie",
    "boring terrible movie"
]
labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

def text_to_vector(text):
    words = text.split()
    return np.mean([word_embeddings.get(word, np.zeros(3)) for word in words], axis=0)

X = np.array([text_to_vector(review) for review in reviews])
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Sentiment analysis accuracy: {accuracy:.2f}")
```

Slide 11:

Real-Life Example: Text Similarity

Word embeddings enable efficient computation of text similarity. This example shows how to use embeddings to find similar documents in a collection.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Simulated word embeddings
word_embeddings = {
    "python": np.array([0.2, 0.3, 0.1]),
    "programming": np.array([0.3, 0.2, 0.1]),
    "data": np.array([0.1, 0.4, 0.2]),
    "science": np.array([0.2, 0.3, 0.3]),
    "machine": np.array([0.3, 0.1, 0.2]),
    "learning": np.array([0.2, 0.2, 0.3])
}

documents = [
    "python programming",
    "data science",
    "machine learning",
    "python data science"
]

def document_to_vector(doc):
    words = doc.lower().split()
    return np.mean([word_embeddings.get(word, np.zeros(3)) for word in words], axis=0)

doc_vectors = np.array([document_to_vector(doc) for doc in documents])

similarity_matrix = cosine_similarity(doc_vectors)

query = "python machine learning"
query_vector = document_to_vector(query)

similarities = cosine_similarity([query_vector], doc_vectors)[0]
most_similar_index = np.argmax(similarities)

print(f"Query: {query}")
print(f"Most similar document: {documents[most_similar_index]}")
print(f"Similarity score: {similarities[most_similar_index]:.2f}")
```

Slide 12:

Challenges and Limitations

Context-independent word embeddings face several challenges:

1. Ambiguity: They struggle with polysemous words, assigning a single vector to words with multiple meanings.
2. Out-of-vocabulary words: Most models can't handle words not seen during training, limiting their applicability to new or rare words.
3. Static nature: These embeddings are fixed once trained, unable to adapt to evolving language use or domain-specific contexts.
4. Bias: Embeddings can reflect and amplify societal biases present in the training data, potentially leading to unfair or discriminatory outcomes in downstream applications.

```python
# Demonstrating ambiguity with a simple example
ambiguous_word = "bank"
context1 = "river bank"
context2 = "bank account"

def simple_context_embedding(word, context):
    # This is a placeholder function to illustrate the concept
    if "river" in context:
        return np.array([0.1, 0.2, 0.3])  # River bank embedding
    elif "account" in context:
        return np.array([0.3, 0.2, 0.1])  # Financial bank embedding
    else:
        return np.array([0.2, 0.2, 0.2])  # Default embedding

print(f"'bank' in '{context1}': {simple_context_embedding(ambiguous_word, context1)}")
print(f"'bank' in '{context2}': {simple_context_embedding(ambiguous_word, context2)}")
```

Slide 13:

Future Directions

The field of word embeddings continues to evolve, addressing limitations and exploring new approaches:

1. Contextualized embeddings: Models like BERT and GPT produce dynamic embeddings based on the surrounding context, better capturing word meaning in different situations.
2. Multilingual embeddings: Researchers are developing models that can represent words across multiple languages in a shared embedding space.
3. Bias mitigation: Techniques are being developed to reduce or eliminate biases in word embeddings, making them more fair and representative.
4. Domain-specific embeddings: Tailoring embeddings to specific domains or tasks can improve performance in specialized applications.

```python
# Pseudocode for a simple contextualized embedding model
class ContextualizedEmbedding:
    def __init__(self, vocab_size, embed_dim, context_size):
        self.word_embeddings = initialize_embeddings(vocab_size, embed_dim)
        self.context_transformer = initialize_transformer(embed_dim, context_size)
    
    def get_embedding(self, word, context):
        word_vector = self.word_embeddings[word]
        context_vectors = [self.word_embeddings[w] for w in context]
        return self.context_transformer(word_vector, context_vectors)

# Usage example
model = ContextualizedEmbedding(vocab_size=10000, embed_dim=300, context_size=5)
word = "bank"
context = ["the", "river", "flows", "by", "the"]
contextualized_embedding = model.get_embedding(word, context)
```

Slide 14:

Additional Resources

For those interested in diving deeper into context-independent word embeddings, here are some valuable resources:

1. "Efficient Estimation of Word Representations in Vector Space" by Mikolov et al. (2013) ArXiv: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
2. "GloVe: Global Vectors for Word Representation" by Pennington et al. (2014) ArXiv: [https://arxiv.org/abs/1405.3531](https://arxiv.org/abs/1405.3531)
3. "Enriching Word Vectors with Subword Information" by Bojanowski et al. (2017) ArXiv: [https://arxiv.org/abs/1607.04606](https://arxiv.org/abs/1607.04606)
4. "Deep contextualized word representations" by Peters et al. (2018) ArXiv: [https://arxiv.org/abs/1802.05365](https://arxiv.org/abs/1802.05365)

These papers provide in-depth explanations of various word embedding techniques and their applications in natural language processing tasks.

