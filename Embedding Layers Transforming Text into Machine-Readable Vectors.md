## Embedding Layers Transforming Text into Machine-Readable Vectors

Slide 1: Word to Vector Basics

An embedding layer transforms discrete word tokens into continuous vector spaces, enabling mathematical operations on text data. The fundamental process involves mapping each word to a unique index and then to a dense vector representation.

```python
import numpy as np

class BasicEmbedding:
    def __init__(self, vocab_size, embedding_dim):
        # Initialize embedding matrix with random values
        self.embedding_matrix = np.random.randn(vocab_size, embedding_dim)
        
    def get_vector(self, word_idx):
        return self.embedding_matrix[word_idx]

# Example usage
vocab_size, embedding_dim = 1000, 50
embedder = BasicEmbedding(vocab_size, embedding_dim)
word_vector = embedder.get_vector(42)  # Get vector for word index 42
print(f"Shape of word vector: {word_vector.shape}")
```

Slide 2: Creating a Word-to-Index Dictionary

Before embedding words, we need to create a mapping between words and their numerical indices. This process involves building a vocabulary from the corpus and assigning unique indices to each word.

```python
class Vocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_count = {}
        self.next_idx = 2

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.next_idx
            self.idx2word[self.next_idx] = word
            self.next_idx += 1
        self.word_count[word] = self.word_count.get(word, 0) + 1

# Example usage
vocab = Vocabulary()
text = "The quick brown fox jumps over the lazy dog"
for word in text.lower().split():
    vocab.add_word(word)

print(f"Vocabulary size: {len(vocab.word2idx)}")
print(f"Word 'the' has index: {vocab.word2idx['the']}")
```

Slide 3: Mathematical Foundation of Embeddings

The embedding layer represents words as vectors in a continuous space where semantic relationships can be captured through vector operations. The mathematical basis involves projecting discrete tokens into a lower-dimensional space.

```python
def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors:
    $$cos(θ) = \frac{vec1 \cdot vec2}{||vec1|| \cdot ||vec2||}$$
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Example vectors
vec1 = np.random.randn(50)  # Embedding for "king"
vec2 = np.random.randn(50)  # Embedding for "queen"
similarity = cosine_similarity(vec1, vec2)
print(f"Cosine similarity: {similarity:.4f}")
```

Slide 4: Implementing Word2Vec Skip-gram Model

The Skip-gram model predicts context words given a target word. This implementation shows the core architecture of the Word2Vec model using numpy for computational efficiency.

```python
class SkipGram:
    def __init__(self, vocab_size, embedding_dim):
        self.w1 = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.w2 = np.random.randn(embedding_dim, vocab_size) * 0.01
    
    def forward(self, x):
        self.h = self.w1[x]  # Input word embedding
        u = np.dot(self.h, self.w2)  # Score for each word
        y_pred = self.softmax(u)
        return y_pred
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def get_embedding(self, word_idx):
        return self.w1[word_idx]

# Example usage
model = SkipGram(vocab_size=1000, embedding_dim=50)
word_idx = 42
context_probs = model.forward(word_idx)
print(f"Probability distribution shape: {context_probs.shape}")
```

Slide 5: Neural Network Implementation of Embedding Layer

A neural network-based embedding layer transforms sparse one-hot encoded vectors into dense representations through matrix multiplication. This implementation demonstrates the forward and backward pass mechanics of an embedding layer.

```python
import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Initialize weights using Xavier initialization
        self.weights = np.random.randn(vocab_size, embedding_dim) * np.sqrt(2.0 / (vocab_size + embedding_dim))
        self.gradients = np.zeros_like(self.weights)
        
    def forward(self, input_indices):
        self.input_indices = input_indices
        return self.weights[input_indices]
    
    def backward(self, grad_output):
        self.gradients.fill(0)
        np.add.at(self.gradients, self.input_indices, grad_output)
        return self.gradients

# Example usage
embedding = EmbeddingLayer(vocab_size=1000, embedding_dim=50)
indices = np.array([1, 4, 7])
embeddings = embedding.forward(indices)
print(f"Embeddings shape: {embeddings.shape}")
```

Slide 6: Custom Dataset Preprocessing

Efficient preprocessing of text data is crucial for training embedding models. This implementation shows how to create a custom dataset with sliding window contexts for training word embeddings.

```python
class TextDataset:
    def __init__(self, text, window_size=2):
        self.vocab = Vocabulary()
        self.window_size = window_size
        self.process_text(text)
        
    def process_text(self, text):
        # Tokenize and build vocabulary
        words = text.lower().split()
        self.word_indices = []
        for word in words:
            self.vocab.add_word(word)
            self.word_indices.append(self.vocab.word2idx[word])
            
    def generate_training_pairs(self):
        pairs = []
        for i, target in enumerate(self.word_indices):
            start = max(0, i - self.window_size)
            end = min(len(self.word_indices), i + self.window_size + 1)
            
            context = (self.word_indices[start:i] + 
                      self.word_indices[i+1:end])
            pairs.extend([(target, ctx) for ctx in context])
        return pairs

# Example usage
text = "the quick brown fox jumps over the lazy dog"
dataset = TextDataset(text, window_size=2)
pairs = dataset.generate_training_pairs()
print(f"Number of training pairs: {len(pairs)}")
print(f"First few pairs: {pairs[:5]}")
```

Slide 7: Training Loop Implementation

The training process for embedding models requires careful batching and optimization. This implementation shows a complete training loop with negative sampling and stochastic gradient descent.

```python
class EmbeddingTrainer:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.embedding = EmbeddingLayer(vocab_size, embedding_dim)
        self.learning_rate = learning_rate
        
    def train_step(self, target_idx, context_idx, negative_samples):
        # Forward pass
        target_embedding = self.embedding.forward(np.array([target_idx]))
        context_embedding = self.embedding.forward(context_idx)
        neg_embedding = self.embedding.forward(negative_samples)
        
        # Compute loss and gradients
        pos_loss = -np.log(self.sigmoid(np.dot(target_embedding, 
                                              context_embedding.T)))
        neg_loss = -np.sum(np.log(self.sigmoid(-np.dot(target_embedding, 
                                                      neg_embedding.T))))
        
        # Update embeddings
        self.embedding.weights -= self.learning_rate * self.embedding.gradients
        
        return pos_loss + neg_loss
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Example usage
trainer = EmbeddingTrainer(vocab_size=1000, embedding_dim=50)
target_idx = 5
context_idx = np.array([2, 7])
negative_samples = np.array([1, 3, 4])
loss = trainer.train_step(target_idx, context_idx, negative_samples)
print(f"Training loss: {loss:.4f}")
```

Slide 8: Subword Tokenization

Subword tokenization helps handle out-of-vocabulary words by breaking them into meaningful subunits. This implementation demonstrates a simple byte-pair encoding (BPE) tokenizer.

```python
class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = set()
        
    def get_stats(self, words):
        pairs = {}
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pair = (symbols[i], symbols[i+1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs
    
    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in v_in:
            w_out = word.replace(bigram, replacement)
            v_out[w_out] = v_in[word]
        return v_out

# Example usage
tokenizer = BPETokenizer(vocab_size=100)
words = {"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6}
pairs = tokenizer.get_stats(words)
print(f"Most frequent pairs: {sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:3]}")
```

Slide 9: Performance Evaluation Metrics

Evaluating embedding quality requires multiple metrics including cosine similarity, analogy tasks, and clustering analysis. This implementation provides a comprehensive evaluation suite.

```python
class EmbeddingEvaluator:
    def __init__(self, embedding_matrix, word2idx):
        self.embedding_matrix = embedding_matrix
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in word2idx.items()}
        
    def word_similarity(self, word1, word2):
        idx1, idx2 = self.word2idx[word1], self.word2idx[word2]
        vec1, vec2 = self.embedding_matrix[idx1], self.embedding_matrix[idx2]
        return cosine_similarity(vec1, vec2)
    
    def analogy(self, a, b, c):
        """Solves a:b :: c:?"""
        a_vec = self.embedding_matrix[self.word2idx[a]]
        b_vec = self.embedding_matrix[self.word2idx[b]]
        c_vec = self.embedding_matrix[self.word2idx[c]]
        
        result_vec = b_vec - a_vec + c_vec
        similarities = np.dot(self.embedding_matrix, result_vec)
        most_similar = np.argmax(similarities)
        return self.idx2word[most_similar]

# Example usage
embedding_matrix = np.random.randn(1000, 50)
word2idx = {"king": 0, "queen": 1, "man": 2, "woman": 3}
evaluator = EmbeddingEvaluator(embedding_matrix, word2idx)
similarity = evaluator.word_similarity("king", "queen")
analogy = evaluator.analogy("king", "queen", "man")
print(f"Similarity between king and queen: {similarity:.4f}")
print(f"man is to woman as king is to: {analogy}")
```

Slide 10: Contextual Embeddings

Modern contextual embeddings like BERT consider the entire sentence context. This implementation shows how to create position-aware embeddings with sinusoidal positional encoding.

```python
def positional_encoding(seq_len, d_model):
    """
    Generate positional encodings using the formula:
    $$PE_{(pos,2i)} = sin(pos/10000^{2i/d_{model}})$$
    $$PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_{model}})$$
    """
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return pos_encoding

class ContextualEmbedding:
    def __init__(self, vocab_size, embedding_dim, max_seq_len=512):
        self.word_embedding = EmbeddingLayer(vocab_size, embedding_dim)
        self.positional_encoding = positional_encoding(max_seq_len, embedding_dim)
        
    def forward(self, input_indices):
        word_embeddings = self.word_embedding.forward(input_indices)
        seq_len = len(input_indices)
        return word_embeddings + self.positional_encoding[:seq_len]

# Example usage
contextual_emb = ContextualEmbedding(vocab_size=1000, embedding_dim=50)
input_sequence = np.array([1, 4, 7, 2])
output = contextual_emb.forward(input_sequence)
print(f"Contextual embedding shape: {output.shape}")
```

Slide 11: Real-world Application: Sentiment Analysis

This implementation demonstrates how to use word embeddings for sentiment analysis of movie reviews, including data preprocessing and model training.

```python
class SentimentClassifier:
    def __init__(self, embedding_layer, hidden_dim=64):
        self.embedding = embedding_layer
        self.W = np.random.randn(embedding_layer.embedding_dim, hidden_dim) * 0.01
        self.U = np.random.randn(hidden_dim, 1) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.b2 = np.zeros(1)
        
    def forward(self, sentence_indices):
        # Get embeddings for all words in sentence
        self.embedded = self.embedding.forward(sentence_indices)
        # Average word embeddings
        self.sentence_vector = np.mean(self.embedded, axis=0)
        # Feed-forward
        self.hidden = np.tanh(np.dot(self.sentence_vector, self.W) + self.b1)
        output = self.sigmoid(np.dot(self.hidden, self.U) + self.b2)
        return output
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Example usage with movie review data
sentences = ["this movie was great", "terrible waste of time"]
labels = [1, 0]  # 1 for positive, 0 for negative

# Preprocess and train
vocab = Vocabulary()
for sentence in sentences:
    for word in sentence.split():
        vocab.add_word(word)

embedding_layer = EmbeddingLayer(len(vocab.word2idx), embedding_dim=50)
classifier = SentimentClassifier(embedding_layer)

# Process one example
sentence = sentences[0]
indices = [vocab.word2idx.get(word, vocab.word2idx['<UNK>']) 
           for word in sentence.split()]
prediction = classifier.forward(np.array(indices))
print(f"Sentiment prediction: {prediction[0]:.4f}")
```

Slide 12: Real-world Application: Document Clustering

Implementation of document clustering using learned embeddings, demonstrating how to represent and cluster documents in the embedding space.

```python
from sklearn.cluster import KMeans
import numpy as np

class DocumentClustering:
    def __init__(self, embedding_layer, n_clusters=3):
        self.embedding = embedding_layer
        self.kmeans = KMeans(n_clusters=n_clusters)
        
    def get_document_vector(self, document):
        # Convert document to indices
        words = document.lower().split()
        indices = [self.embedding.vocab.word2idx.get(word, 
                  self.embedding.vocab.word2idx['<UNK>']) for word in words]
        
        # Get embeddings and average them
        word_embeddings = self.embedding.forward(np.array(indices))
        return np.mean(word_embeddings, axis=0)
    
    def cluster_documents(self, documents):
        # Convert all documents to vectors
        doc_vectors = np.array([self.get_document_vector(doc) 
                              for doc in documents])
        
        # Perform clustering
        clusters = self.kmeans.fit_predict(doc_vectors)
        return clusters

# Example usage
documents = [
    "machine learning algorithms perform computation",
    "deep neural networks process data",
    "shakespeare wrote many famous plays",
    "renaissance artists painted masterpieces"
]

embedding_layer = EmbeddingLayer(vocab_size=1000, embedding_dim=50)
clusterer = DocumentClustering(embedding_layer, n_clusters=2)
clusters = clusterer.cluster_documents(documents)
print(f"Document clusters: {clusters}")
```

Slide 13: Additional Resources

*   [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781) - "Efficient Estimation of Word Representations in Vector Space"
*   [https://arxiv.org/abs/1803.11175](https://arxiv.org/abs/1803.11175) - "Deep contextualized word representations"
*   [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
*   [https://arxiv.org/abs/2103.15049](https://arxiv.org/abs/2103.15049) - "Word2Vec: Optimal Hyper-Parameters and Their Impact on NLP Downstream Tasks"

