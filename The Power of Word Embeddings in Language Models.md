## The Power of Word Embeddings in Language Models
Slide 1: Understanding Token Embeddings

Token embeddings form the foundation of modern language models by transforming discrete tokens into continuous vector representations. These dense vectors capture semantic relationships between words in a high-dimensional space, enabling mathematical operations on textual data.

```python
import numpy as np
from collections import defaultdict

class TokenEmbedding:
    def __init__(self, dim=100):
        self.embedding_dim = dim
        self.word_to_idx = {}
        self.embeddings = []
        
    def fit(self, corpus):
        # Create vocabulary
        vocab = set(word for sentence in corpus for word in sentence.split())
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        
        # Initialize random embeddings
        self.embeddings = np.random.normal(
            scale=0.1, 
            size=(len(self.word_to_idx), self.embedding_dim)
        )

    def get_vector(self, word):
        idx = self.word_to_idx.get(word)
        return self.embeddings[idx] if idx is not None else None

# Example usage
corpus = ["natural language processing", "deep learning models"]
embedder = TokenEmbedding(dim=3)
embedder.fit(corpus)
print(f"Vector for 'language': {embedder.get_vector('language')}")
```

Slide 2: Context Window and Positional Encoding

Positional encoding adds crucial information about token position in a sequence, preventing the model from treating text as an unordered bag of words. This implementation demonstrates a basic sinusoidal positional encoding scheme similar to the one used in the Transformer architecture.

```python
def positional_encoding(seq_len, d_model):
    # Create position matrix
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Calculate encodings
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return pos_encoding

# Example
seq_length, dimension = 10, 8
encodings = positional_encoding(seq_length, dimension)
print("Position encodings shape:", encodings.shape)
print("First position vector:", encodings[0])
```

Slide 3: Word2Vec Implementation

Word2Vec revolutionized word embeddings by introducing the skip-gram architecture, which predicts context words given a target word. This implementation showcases the core mechanism of the skip-gram model with negative sampling.

```python
import numpy as np
from scipy.special import expit

class SkipGram:
    def __init__(self, vocab_size, embedding_dim):
        self.W = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
        self.W_context = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
        
    def forward(self, target_idx, context_idx):
        target_vector = self.W[target_idx]
        context_vector = self.W_context[context_idx]
        score = np.dot(target_vector, context_vector)
        probability = expit(score)
        return probability
    
    def negative_sampling_loss(self, target_idx, context_idx, negative_samples):
        positive_score = self.forward(target_idx, context_idx)
        negative_scores = [self.forward(target_idx, neg_idx) 
                         for neg_idx in negative_samples]
        
        loss = -np.log(positive_score) - sum(np.log(1 - ns) 
                                           for ns in negative_scores)
        return loss

# Example usage
model = SkipGram(vocab_size=5000, embedding_dim=100)
loss = model.negative_sampling_loss(
    target_idx=1, 
    context_idx=2, 
    negative_samples=[10, 20, 30]
)
print(f"Training loss: {loss:.4f}")
```

Slide 4: Subword Tokenization

Subword tokenization bridges the gap between character-level and word-level tokenization, enabling models to handle out-of-vocabulary words effectively. This implementation demonstrates a basic Byte-Pair Encoding (BPE) algorithm.

```python
class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.merges = {}
        
    def get_stats(self, words):
        pairs = defaultdict(int)
        for word in words:
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += 1
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
tokenizer = BPETokenizer(vocab_size=1000)
corpus = ["low lower lowest", "newer newest new"]
print("Initial corpus:", corpus)
```

Slide 5: Embedding Layer Implementation

The embedding layer serves as the interface between discrete tokens and their continuous vector representations. This implementation shows how to create and optimize embeddings using gradient descent with support for pretrained vectors.

```python
import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.embeddings = np.random.randn(vocab_size, embedding_dim) / np.sqrt(embedding_dim)
        self.lr = learning_rate
        
    def forward(self, token_ids):
        # Get embeddings for input tokens
        return self.embeddings[token_ids]
    
    def backward(self, token_ids, gradients):
        # Update embeddings using gradients
        self.embeddings[token_ids] -= self.lr * gradients
        
    def load_pretrained(self, pretrained_vectors):
        # Load pre-trained word vectors
        assert pretrained_vectors.shape == self.embeddings.shape
        self.embeddings = pretrained_vectors.copy()

# Example usage
embedding = EmbeddingLayer(vocab_size=1000, embedding_dim=50)
tokens = np.array([1, 4, 10])
vectors = embedding.forward(tokens)
print(f"Shape of retrieved embeddings: {vectors.shape}")
```

Slide 6: Cosine Similarity and Word Relationships

Word embeddings capture semantic relationships that can be measured through cosine similarity. This implementation demonstrates how to compute word similarities and analyze semantic relationships between words in the embedding space.

```python
class WordSimilarity:
    def __init__(self, embeddings, word_to_idx):
        self.embeddings = embeddings
        self.word_to_idx = word_to_idx
        self.idx_to_word = {v: k for k, v in word_to_idx.items()}
    
    def cosine_similarity(self, v1, v2):
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.dot(v1, v2) / norm if norm != 0 else 0
    
    def most_similar(self, word, n=5):
        if word not in self.word_to_idx:
            return []
        
        word_idx = self.word_to_idx[word]
        word_vec = self.embeddings[word_idx]
        
        similarities = []
        for idx, vec in enumerate(self.embeddings):
            if idx != word_idx:
                sim = self.cosine_similarity(word_vec, vec)
                similarities.append((self.idx_to_word[idx], sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

# Example usage
word_sim = WordSimilarity(embedding.embeddings, {"king": 0, "queen": 1, "man": 2})
print(word_sim.most_similar("king"))
```

Slide 7: Contextual Embeddings Implementation

Unlike static embeddings, contextual embeddings generate different vectors for the same word based on its context. This implementation shows a simplified version of context-aware embeddings using a bidirectional LSTM.

```python
import numpy as np

class ContextualEmbedding:
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.embedding = EmbeddingLayer(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        
        # LSTM parameters
        self.Wf = np.random.randn(hidden_dim, embedding_dim + hidden_dim)
        self.Wi = np.random.randn(hidden_dim, embedding_dim + hidden_dim)
        self.Wc = np.random.randn(hidden_dim, embedding_dim + hidden_dim)
        self.Wo = np.random.randn(hidden_dim, embedding_dim + hidden_dim)
        
    def forward(self, sentence_ids):
        embeddings = self.embedding.forward(sentence_ids)
        
        # Initialize hidden state and cell state
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        
        contextual_embeddings = []
        
        # Process each token in context
        for emb in embeddings:
            x = np.concatenate([emb, h])
            
            # LSTM gates
            f = np.sigmoid(self.Wf.dot(x))
            i = np.sigmoid(self.Wi.dot(x))
            c_tilde = np.tanh(self.Wc.dot(x))
            o = np.sigmoid(self.Wo.dot(x))
            
            # Update states
            c = f * c + i * c_tilde
            h = o * np.tanh(c)
            
            contextual_embeddings.append(h)
            
        return np.array(contextual_embeddings)

# Example usage
contextual_emb = ContextualEmbedding(vocab_size=1000, embedding_dim=50, hidden_dim=100)
sentence = np.array([1, 4, 10])  # Token IDs
context_vectors = contextual_emb.forward(sentence)
print(f"Shape of contextual embeddings: {context_vectors.shape}")
```

Slide 8: Attention Mechanism for Embeddings

The attention mechanism allows models to dynamically focus on different parts of the input sequence. This implementation demonstrates self-attention for creating context-aware embeddings.

```python
class SelfAttention:
    def __init__(self, embedding_dim, num_heads=1):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Initialize projection matrices
        self.W_q = np.random.randn(embedding_dim, embedding_dim)
        self.W_k = np.random.randn(embedding_dim, embedding_dim)
        self.W_v = np.random.randn(embedding_dim, embedding_dim)
        
    def attention(self, Q, K, V, mask=None):
        # Scaled dot-product attention
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = np.softmax(scores, axis=-1)
        return np.matmul(attention_weights, V), attention_weights
    
    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        
        # Linear projections
        Q = np.matmul(X, self.W_q)
        K = np.matmul(X, self.W_k)
        V = np.matmul(X, self.W_v)
        
        # Multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        attended_values, attention_weights = self.attention(Q, K, V)
        return attended_values.reshape(batch_size, seq_len, self.embedding_dim)

# Example usage
attention = SelfAttention(embedding_dim=64, num_heads=4)
sequence = np.random.randn(2, 10, 64)  # (batch_size, seq_len, embedding_dim)
attended = attention.forward(sequence)
print(f"Shape of attention output: {attended.shape}")
```

Slide 9: Embedding Visualization Techniques

This implementation demonstrates dimensionality reduction techniques to visualize high-dimensional embeddings in 2D/3D space using t-SNE and UMAP, essential for understanding the geometric relationships between word vectors.

```python
import numpy as np
from sklearn.manifold import TSNE
import umap

class EmbeddingVisualizer:
    def __init__(self, embeddings, word_to_idx):
        self.embeddings = embeddings
        self.word_to_idx = word_to_idx
        self.idx_to_word = {v: k for k, v in word_to_idx.items()}
        
    def reduce_dimensions(self, method='tsne', n_components=2):
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            
        reduced_embeddings = reducer.fit_transform(self.embeddings)
        
        visualization_data = []
        for idx, coords in enumerate(reduced_embeddings):
            word = self.idx_to_word[idx]
            visualization_data.append({
                'word': word,
                'coordinates': coords,
            })
            
        return visualization_data

# Example usage
embeddings = np.random.randn(1000, 100)  # 1000 words, 100 dimensions
word_to_idx = {f"word_{i}": i for i in range(1000)}

visualizer = EmbeddingVisualizer(embeddings, word_to_idx)
viz_data = visualizer.reduce_dimensions(method='tsne')
print(f"First word coordinates: {viz_data[0]}")
```

Slide 10: Embedding Training with Negative Sampling

This implementation shows how to train word embeddings using negative sampling, which approximates the full softmax computation and makes training more efficient for large vocabularies.

```python
class NegativeSamplingTrainer:
    def __init__(self, vocab_size, embedding_dim, num_negative=5):
        self.embedding_dim = embedding_dim
        self.num_negative = num_negative
        
        # Initialize word embeddings and context embeddings
        self.word_embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.context_embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        
        # Sampling table for negative sampling
        self.sampling_table = self._create_sampling_table(vocab_size)
        
    def _create_sampling_table(self, vocab_size, power=0.75):
        # Create sampling table based on word frequency
        sampling_table = np.zeros(vocab_size)
        for i in range(vocab_size):
            sampling_table[i] = i ** (-power)
        sampling_table = sampling_table / np.sum(sampling_table)
        return sampling_table
        
    def train_pair(self, word_idx, context_idx, learning_rate=0.025):
        # Positive sample
        word_vec = self.word_embeddings[word_idx]
        context_vec = self.context_embeddings[context_idx]
        
        # Calculate positive score
        pos_score = np.dot(word_vec, context_vec)
        pos_sigmoid = 1 / (1 + np.exp(-pos_score))
        
        # Negative samples
        neg_indices = np.random.choice(
            len(self.sampling_table),
            size=self.num_negative,
            p=self.sampling_table
        )
        
        # Calculate loss and gradients
        total_loss = -np.log(pos_sigmoid)
        
        # Update embeddings
        grad_word = (pos_sigmoid - 1) * context_vec
        grad_context = (pos_sigmoid - 1) * word_vec
        
        for neg_idx in neg_indices:
            neg_vec = self.context_embeddings[neg_idx]
            neg_score = np.dot(word_vec, neg_vec)
            neg_sigmoid = 1 / (1 + np.exp(-neg_score))
            
            total_loss -= np.log(1 - neg_sigmoid)
            grad_word += neg_sigmoid * neg_vec
            self.context_embeddings[neg_idx] -= learning_rate * (neg_sigmoid * word_vec)
            
        # Update vectors
        self.word_embeddings[word_idx] -= learning_rate * grad_word
        self.context_embeddings[context_idx] -= learning_rate * grad_context
        
        return total_loss

# Example usage
trainer = NegativeSamplingTrainer(vocab_size=10000, embedding_dim=100)
loss = trainer.train_pair(word_idx=1, context_idx=5)
print(f"Training loss: {loss:.4f}")
```

Slide 11: Real-world Application - Sentiment Analysis with Embeddings

This implementation demonstrates how to use word embeddings for sentiment analysis, including data preprocessing and model training with a practical example.

```python
class SentimentClassifier:
    def __init__(self, embedding_layer, hidden_dim=64):
        self.embedding_layer = embedding_layer
        self.hidden_dim = hidden_dim
        
        # Initialize weights
        emb_dim = embedding_layer.embeddings.shape[1]
        self.W1 = np.random.randn(hidden_dim, emb_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(1, hidden_dim) * 0.1
        self.b2 = np.zeros(1)
        
    def forward(self, token_ids):
        # Get embeddings
        embeddings = self.embedding_layer.forward(token_ids)
        
        # Average pooling over sequence
        sentence_embedding = np.mean(embeddings, axis=0)
        
        # Feed forward
        hidden = np.tanh(np.dot(self.W1, sentence_embedding) + self.b1)
        output = np.sigmoid(np.dot(self.W2, hidden) + self.b2)
        
        return output

# Example usage with preprocessing
def preprocess_text(text, word_to_idx, max_length=100):
    # Tokenize and convert to indices
    tokens = text.lower().split()
    token_ids = [word_to_idx.get(word, word_to_idx['<UNK>']) 
                for word in tokens[:max_length]]
    return np.array(token_ids)

# Create sample data
embedding_layer = EmbeddingLayer(vocab_size=10000, embedding_dim=100)
classifier = SentimentClassifier(embedding_layer)

# Example text
text = "this movie was absolutely fantastic"
word_to_idx = {'this': 0, 'movie': 1, 'was': 2, 'absolutely': 3, 'fantastic': 4, '<UNK>': 5}
token_ids = preprocess_text(text, word_to_idx)
sentiment_score = classifier.forward(token_ids)
print(f"Sentiment score: {sentiment_score[0]:.4f}")
```

Slide 12: Cross-lingual Embeddings

Cross-lingual embeddings enable semantic mapping between different languages by aligning word vectors from multiple languages into a shared vector space. This implementation demonstrates a basic alignment technique using parallel vocabularies.

```python
class CrossLingualEmbeddings:
    def __init__(self, source_embeddings, target_embeddings):
        self.source_emb = source_embeddings
        self.target_emb = target_embeddings
        self.alignment_matrix = None
        
    def learn_alignment(self, parallel_vocab):
        """Learn alignment matrix using Procrustes solution"""
        # Extract parallel embeddings
        source_vectors = np.array([self.source_emb[word] 
                                 for word in parallel_vocab['source']])
        target_vectors = np.array([self.target_emb[word] 
                                 for word in parallel_vocab['target']])
        
        # Compute SVD
        U, _, Vt = np.linalg.svd(target_vectors.T.dot(source_vectors))
        self.alignment_matrix = U.dot(Vt)
        
    def translate(self, source_word, n_candidates=5):
        """Find closest words in target language"""
        if source_word not in self.source_emb:
            return []
            
        # Get aligned vector
        source_vector = self.source_emb[source_word]
        aligned_vector = source_vector.dot(self.alignment_matrix)
        
        # Find nearest neighbors
        similarities = {word: np.dot(aligned_vector, vec) 
                      for word, vec in self.target_emb.items()}
        
        return sorted(similarities.items(), 
                     key=lambda x: x[1], 
                     reverse=True)[:n_candidates]

# Example usage
source_emb = {"dog": np.random.randn(100), "cat": np.random.randn(100)}
target_emb = {"perro": np.random.randn(100), "gato": np.random.randn(100)}
parallel_vocab = {
    "source": ["dog", "cat"],
    "target": ["perro", "gato"]
}

cross_lingual = CrossLingualEmbeddings(source_emb, target_emb)
cross_lingual.learn_alignment(parallel_vocab)
translations = cross_lingual.translate("dog")
print(f"Translations: {translations}")
```

Slide 13: Real-world Application - Document Classification

This implementation shows how to use document-level embeddings for multi-class document classification, incorporating attention mechanisms for better representation learning.

```python
class DocumentClassifier:
    def __init__(self, embedding_layer, num_classes, hidden_dim=128):
        self.embedding_layer = embedding_layer
        self.attention = SelfAttention(embedding_layer.embeddings.shape[1])
        
        # Model parameters
        self.hidden_dim = hidden_dim
        emb_dim = embedding_layer.embeddings.shape[1]
        
        # Initialize weights
        self.W1 = np.random.randn(hidden_dim, emb_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(num_classes, hidden_dim) * 0.1
        self.b2 = np.zeros(num_classes)
        
    def forward(self, document_tokens):
        # Get embeddings for all tokens
        embeddings = self.embedding_layer.forward(document_tokens)
        
        # Apply self-attention
        attended = self.attention.forward(
            embeddings.reshape(1, len(document_tokens), -1)
        )[0]
        
        # Document representation
        doc_embedding = np.mean(attended, axis=0)
        
        # Classification
        hidden = np.tanh(np.dot(self.W1, doc_embedding) + self.b1)
        logits = np.dot(self.W2, hidden) + self.b2
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        return probabilities

# Example usage
embedding_dim = 100
vocab_size = 10000
num_classes = 4

embeddings = EmbeddingLayer(vocab_size, embedding_dim)
classifier = DocumentClassifier(embeddings, num_classes)

# Sample document
document = np.array([1, 4, 10, 20, 5])  # Token IDs
class_probs = classifier.forward(document)
print(f"Class probabilities: {class_probs}")
```

Slide 14: Additional Resources

*   ArXiv Papers:
*   "Efficient Estimation of Word Representations in Vector Space" - [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
*   "GloVe: Global Vectors for Word Representation" - [https://arxiv.org/abs/1504.06652](https://arxiv.org/abs/1504.06652)
*   "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" - [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
*   "Advances in Pre-Training Distributed Word Representations" - [https://arxiv.org/abs/1712.09405](https://arxiv.org/abs/1712.09405)
*   "Cross-lingual Word Embedding Models" - [https://arxiv.org/abs/1710.04087](https://arxiv.org/abs/1710.04087)
*   Additional searches recommended:
*   Google Scholar: "word embeddings survey"
*   Google Scholar: "contextual embeddings comparison"
*   ACL Anthology: "embedding evaluation metrics"

