## Understanding Word Embeddings in Machine Learning
Slide 1: One-Hot Encoding Implementation

Word embeddings begin with one-hot encoding, a fundamental technique where each word is represented as a binary vector. This implementation demonstrates how to transform a vocabulary into sparse vectors where only one position contains 1 and all others are 0.

```python
from typing import List, Dict
import numpy as np

def create_one_hot_encoding(vocabulary: List[str]) -> Dict[str, np.ndarray]:
    vocab_size = len(vocabulary)
    word_to_index = {word: idx for idx, word in enumerate(vocabulary)}
    
    encodings = {}
    for word in vocabulary:
        vector = np.zeros(vocab_size)
        vector[word_to_index[word]] = 1
        encodings[word] = vector
    
    return encodings

# Example usage
vocab = ["cat", "dog", "mouse", "house"]
encodings = create_one_hot_encoding(vocab)

print("One-hot encoding for 'cat':", encodings["cat"])
# Output: [1. 0. 0. 0.]
```

Slide 2: Word2Vec Skip-gram Architecture

The Skip-gram model predicts context words given a target word. This implementation focuses on the core architecture, using a neural network to learn word representations that capture semantic relationships between words in the vocabulary.

```python
import numpy as np

class SkipGram:
    def __init__(self, vocab_size: int, embedding_dim: int):
        # Initialize embedding matrices
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_context = np.random.randn(embedding_dim, vocab_size) * 0.01
    
    def forward(self, word_idx: int) -> np.ndarray:
        # Get word embedding
        hidden = self.W[word_idx]
        # Compute context probabilities
        scores = np.dot(hidden, self.W_context)
        probs = self._softmax(scores)
        return probs
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

# Example
model = SkipGram(vocab_size=5000, embedding_dim=100)
probabilities = model.forward(word_idx=42)
```

Slide 3: GloVe Loss Function Implementation

GloVe's distinctive feature is its loss function that combines global matrix factorization with local context window methods. This implementation shows how to compute the weighted least squares regression model used in GloVe.

```python
def glove_loss(
    w_i: np.ndarray,
    w_j: np.ndarray,
    b_i: float,
    b_j: float,
    X_ij: float,
    f_X_ij: float
) -> float:
    """
    w_i: vector for target word
    w_j: vector for context word
    b_i, b_j: bias terms
    X_ij: co-occurrence count
    f_X_ij: weighting function value
    """
    prediction = np.dot(w_i, w_j) + b_i + b_j - np.log(X_ij + 1e-8)
    return 0.5 * f_X_ij * (prediction ** 2)

def weighting_function(x: float, x_max: float = 100, alpha: float = 0.75) -> float:
    if x < x_max:
        return (x / x_max) ** alpha
    return 1.0

# Example usage
w_i = np.random.randn(100)
w_j = np.random.randn(100)
loss = glove_loss(w_i, w_j, 0.1, 0.2, 30, weighting_function(30))
print(f"Loss: {loss:.4f}")
```

Slide 4: Contextual Word Embeddings

Contextual embeddings revolutionized NLP by generating different vectors for the same word based on context. This implementation shows how to create a simple contextual embedding using a bidirectional LSTM architecture.

```python
import torch
import torch.nn as nn

class ContextualEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded)
        return outputs

# Example
model = ContextualEmbedding(vocab_size=5000, embedding_dim=100, hidden_dim=128)
sample_input = torch.randint(0, 5000, (32, 20))  # batch_size=32, seq_len=20
contextual_embeddings = model(sample_input)
print(f"Output shape: {contextual_embeddings.shape}")
```

Slide 5: Word Co-occurrence Matrix

The foundation of many word embedding techniques relies on building a co-occurrence matrix. This implementation demonstrates how to construct and normalize a word co-occurrence matrix from a text corpus using efficient sparse matrix operations.

```python
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict

def build_cooccurrence_matrix(
    corpus: List[List[str]], 
    window_size: int = 2
) -> Tuple[csr_matrix, Dict[str, int]]:
    # Build vocabulary
    word_to_id = {}
    cooccurrences = defaultdict(float)
    
    # First pass: build vocabulary
    for sentence in corpus:
        for word in sentence:
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)
    
    # Second pass: count co-occurrences
    for sentence in corpus:
        word_ids = [word_to_id[w] for w in sentence]
        for i, center_word_id in enumerate(word_ids):
            for j in range(max(0, i - window_size), min(len(word_ids), i + window_size + 1)):
                if i != j:
                    cooccurrences[(center_word_id, word_ids[j])] += 1.0/abs(i-j)
    
    # Build sparse matrix
    vocab_size = len(word_to_id)
    rows, cols, data = [], [], []
    for (i, j), count in cooccurrences.items():
        rows.append(i)
        cols.append(j)
        data.append(count)
    
    return csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size)), word_to_id

# Example usage
corpus = [
    ["the", "quick", "brown", "fox"],
    ["the", "lazy", "brown", "dog"]
]
matrix, vocab = build_cooccurrence_matrix(corpus)
print(f"Vocabulary size: {len(vocab)}")
print(f"Matrix shape: {matrix.shape}")
```

Slide 6: Custom Word2Vec Training Loop

A detailed implementation of the Word2Vec training process showing how to implement negative sampling and optimize word vectors using stochastic gradient descent.

```python
import numpy as np
from typing import List, Tuple

class Word2VecTrainer:
    def __init__(self, vocab_size: int, embedding_dim: int, learning_rate: float = 0.01):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.lr = learning_rate
        
        # Initialize embeddings
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.C = np.random.randn(vocab_size, embedding_dim) * 0.01
    
    def negative_sampling(self, n_samples: int) -> np.ndarray:
        # Simple uniform sampling for demonstration
        return np.random.randint(0, self.vocab_size, size=n_samples)
    
    def train_pair(
        self, 
        target_idx: int, 
        context_idx: int, 
        n_negative: int = 5
    ) -> float:
        # Positive sample
        h = self.W[target_idx]
        context_vec = self.C[context_idx]
        pos_score = self._sigmoid(np.dot(h, context_vec))
        
        # Negative samples
        neg_indices = self.negative_sampling(n_negative)
        neg_vecs = self.C[neg_indices]
        neg_scores = self._sigmoid(-np.dot(h, neg_vecs.T))
        
        # Compute gradients
        pos_grad = (pos_score - 1) * context_vec
        neg_grad = np.sum((1 - neg_scores)[:, np.newaxis] * neg_vecs, axis=0)
        
        # Update embeddings
        self.W[target_idx] -= self.lr * (pos_grad + neg_grad)
        self.C[context_idx] -= self.lr * ((pos_score - 1) * h)
        
        # Return loss
        return -(np.log(pos_score) + np.sum(np.log(neg_scores)))
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

# Example usage
trainer = Word2VecTrainer(vocab_size=5000, embedding_dim=100)
loss = trainer.train_pair(target_idx=42, context_idx=128)
print(f"Training loss: {loss:.4f}")
```

Slide 7: Subword Tokenization with BPE

Modern embeddings often use subword tokenization. This implementation shows how to create a Byte-Pair Encoding tokenizer from scratch, which helps handle out-of-vocabulary words and rare tokens.

```python
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab: Set[str] = set()
    
    def learn_bpe(self, texts: List[str], min_freq: int = 2):
        # Initialize with characters
        word_freqs = Counter()
        for text in texts:
            for word in text.split():
                word = ' '.join(list(word)) + ' </w>'
                word_freqs[word] += 1
        
        self.vocab = set(char for word in word_freqs for char in word.split())
        
        while len(self.vocab) < self.vocab_size:
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break
                
            most_freq = max(pairs, key=pairs.get)
            self.merges[most_freq] = ''.join(most_freq)
            self.vocab.add(self.merges[most_freq])
            
            # Apply merge
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = self._apply_merge(word, most_freq)
                new_word_freqs[new_word] += freq
            word_freqs = new_word_freqs
    
    def _get_pairs(self, word_freqs: Counter) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs
    
    def _apply_merge(self, word: str, pair: Tuple[str, str]) -> str:
        new_word = []
        symbols = word.split()
        i = 0
        while i < len(symbols):
            if i < len(symbols)-1 and tuple(symbols[i:i+2]) == pair:
                new_word.append(self.merges[pair])
                i += 2
            else:
                new_word.append(symbols[i])
                i += 1
        return ' '.join(new_word)

# Example usage
texts = [
    "the quick brown fox",
    "the lazy brown dog"
]
tokenizer = BPETokenizer(vocab_size=100)
tokenizer.learn_bpe(texts)
print(f"Vocabulary size: {len(tokenizer.vocab)}")
print(f"Sample merges: {list(tokenizer.merges.items())[:3]}")
```

Slide 8: FastText-Style Subword Embeddings

FastText extends Word2Vec by incorporating subword information. This implementation demonstrates how to generate and combine subword embeddings for better representation of rare and out-of-vocabulary words.

```python
class FastTextModel:
    def __init__(self, vocab_size: int, embedding_dim: int, min_ngram: int = 3, max_ngram: int = 6):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        
        # Initialize embeddings for words and character n-grams
        self.word_embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.ngram_embeddings = {}
        
    def get_subwords(self, word: str) -> List[str]:
        # Add special boundary symbols
        word = f"<{word}>"
        subwords = []
        
        # Generate character n-grams
        for n in range(self.min_ngram, self.max_ngram + 1):
            for i in range(len(word) - n + 1):
                subwords.append(word[i:i+n])
        return subwords
    
    def get_word_vector(self, word: str) -> np.ndarray:
        # Get word embedding if in vocabulary
        word_vec = self.word_embeddings[self.word_to_id.get(word, 0)]
        
        # Get subword embeddings
        subwords = self.get_subwords(word)
        subword_vecs = []
        
        for subword in subwords:
            if subword in self.ngram_embeddings:
                subword_vecs.append(self.ngram_embeddings[subword])
        
        if subword_vecs:
            subword_vec = np.mean(subword_vecs, axis=0)
            return word_vec + subword_vec
        return word_vec
    
    def train_subword(self, word: str, context: str, learning_rate: float = 0.01):
        subwords = self.get_subwords(word)
        word_vec = self.get_word_vector(word)
        
        # Simple negative sampling context prediction
        context_vec = self.word_embeddings[self.word_to_id[context]]
        score = np.dot(word_vec, context_vec)
        grad = (self._sigmoid(score) - 1) * context_vec
        
        # Update embeddings
        self.word_embeddings[self.word_to_id.get(word, 0)] -= learning_rate * grad
        
        # Update subword embeddings
        for subword in subwords:
            if subword in self.ngram_embeddings:
                self.ngram_embeddings[subword] -= learning_rate * grad / len(subwords)
    
    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))

# Example usage
import numpy as np

model = FastTextModel(vocab_size=5000, embedding_dim=100)
word = "unhappy"
subwords = model.get_subwords(word)
print(f"Subwords for '{word}': {subwords}")
```

Slide 9: Positional Embeddings for Transformers

Positional embeddings are crucial for transformer models to understand word order. This implementation shows how to create both fixed and learned positional embeddings used in modern architectures.

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Sinusoidal position encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, embedding_dim]
        return x + self.pe[:, :x.size(1)]

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(
            x.size(1), dtype=torch.long, device=x.device
        ).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        return x + position_embeddings

# Example usage
seq_len, batch_size, d_model = 20, 32, 512
x = torch.randn(batch_size, seq_len, d_model)

# Fixed positional encoding
fixed_pos = PositionalEncoding(d_model)
output_fixed = fixed_pos(x)

# Learned positional encoding
learned_pos = LearnedPositionalEncoding(d_model)
output_learned = learned_pos(x)

print(f"Input shape: {x.shape}")
print(f"Output shape (fixed): {output_fixed.shape}")
print(f"Output shape (learned): {output_learned.shape}")
```

Slide 10: Word Similarity Metrics

Implementation of various metrics to evaluate word embedding quality through similarity measurements, including cosine similarity and Euclidean distance with efficient vectorized operations.

```python
import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple

class WordSimilarityEvaluator:
    def __init__(self, embeddings: Dict[str, np.ndarray]):
        self.embeddings = embeddings
        self.words = list(embeddings.keys())
        self.vectors = np.stack([embeddings[w] for w in self.words])
    
    def cosine_similarity(self, word1: str, word2: str) -> float:
        if word1 not in self.embeddings or word2 not in self.embeddings:
            return 0.0
        vec1 = self.embeddings[word1]
        vec2 = self.embeddings[word2]
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def find_nearest_neighbors(
        self, 
        word: str, 
        k: int = 5, 
        metric: str = 'cosine'
    ) -> List[Tuple[str, float]]:
        if word not in self.embeddings:
            return []
        
        query_vec = self.embeddings[word].reshape(1, -1)
        distances = cdist(query_vec, self.vectors, metric=metric)[0]
        
        # Get top-k closest words (excluding the query word)
        nearest_indices = np.argsort(distances)[1:k+1]
        return [
            (self.words[idx], distances[idx]) 
            for idx in nearest_indices
        ]
    
    def analogy(self, a: str, b: str, c: str, n: int = 5) -> List[Tuple[str, float]]:
        """Find words that satisfy the analogy a:b :: c:d"""
        if not all(w in self.embeddings for w in [a, b, c]):
            return []
        
        # Compute target vector: vec(b) - vec(a) + vec(c)
        target = self.embeddings[b] - self.embeddings[a] + self.embeddings[c]
        target = target.reshape(1, -1)
        
        # Find nearest neighbors to target vector
        distances = cdist(target, self.vectors, metric='cosine')[0]
        nearest_indices = np.argsort(distances)[:n]
        
        return [
            (self.words[idx], distances[idx]) 
            for idx in nearest_indices
            if self.words[idx] not in [a, b, c]
        ]

# Example usage
embeddings = {
    "king": np.random.randn(100),
    "queen": np.random.randn(100),
    "man": np.random.randn(100),
    "woman": np.random.randn(100)
}

evaluator = WordSimilarityEvaluator(embeddings)
similarity = evaluator.cosine_similarity("king", "queen")
analogies = evaluator.analogy("king", "queen", "man")

print(f"Similarity between 'king' and 'queen': {similarity:.4f}")
print(f"Analogy results (king:queen :: man:?): {analogies}")
```

Slide 11: Building a Custom Embedding Layer

This implementation shows how to create a custom embedding layer with weight tying and custom initialization, essential for training language models with shared embeddings between encoder and decoder.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CustomEmbeddingLayer(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int, 
        padding_idx: int = None,
        tie_weights: bool = False,
        init_scale: float = 0.02
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Create embedding matrix
        self.weight = nn.Parameter(torch.empty(vocab_size, embedding_dim))
        self.reset_parameters(init_scale)
        
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)
        
        self.tied_decoder = None
        if tie_weights:
            self.tied_decoder = nn.Linear(embedding_dim, vocab_size, bias=False)
            self.tied_decoder.weight = self.weight
    
    def reset_parameters(self, init_scale: float):
        # Xavier/Glorot initialization
        nn.init.normal_(self.weight, mean=0.0, std=init_scale)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass for embedding
        embedded = F.embedding(
            x, 
            self.weight,
            scale_grad_by_freq=True
        )
        return embedded
    
    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Decode back to vocabulary space if weight tying is enabled
        if self.tied_decoder is not None:
            return self.tied_decoder(hidden_states)
        return torch.matmul(hidden_states, self.weight.t())

# Example usage
vocab_size, embedding_dim = 5000, 256
batch_size, seq_length = 32, 20

embedding_layer = CustomEmbeddingLayer(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    tie_weights=True
)

# Forward pass
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
embeddings = embedding_layer(input_ids)

# Decoding
hidden_states = torch.randn(batch_size, seq_length, embedding_dim)
logits = embedding_layer.decode(hidden_states)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Logits shape: {logits.shape}")
```

Slide 12: Implementing Cross-Lingual Word Embeddings

This implementation demonstrates how to align word embeddings from different languages into a shared vector space using orthogonal Procrustes alignment, enabling cross-lingual transfer.

```python
import numpy as np
from scipy.linalg import orthogonal_procrustes
from typing import Dict, List, Tuple

class CrossLingualEmbeddings:
    def __init__(
        self,
        src_embeddings: Dict[str, np.ndarray],
        tgt_embeddings: Dict[str, np.ndarray]
    ):
        self.src_embeddings = src_embeddings
        self.tgt_embeddings = tgt_embeddings
        self.alignment_matrix = None
    
    def align(self, dictionary: List[Tuple[str, str]]) -> np.ndarray:
        """Align embeddings using a seed dictionary"""
        # Extract matched vectors
        src_vectors = []
        tgt_vectors = []
        
        for src_word, tgt_word in dictionary:
            if src_word in self.src_embeddings and tgt_word in self.tgt_embeddings:
                src_vectors.append(self.src_embeddings[src_word])
                tgt_vectors.append(self.tgt_embeddings[tgt_word])
        
        src_matrix = np.stack(src_vectors)
        tgt_matrix = np.stack(tgt_vectors)
        
        # Compute alignment using orthogonal Procrustes
        self.alignment_matrix, _ = orthogonal_procrustes(src_matrix, tgt_matrix)
        return self.alignment_matrix
    
    def translate(
        self, 
        word: str, 
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find k nearest neighbors in target language"""
        if word not in self.src_embeddings or self.alignment_matrix is None:
            return []
        
        # Project source word to target space
        query_vec = np.dot(
            self.src_embeddings[word],
            self.alignment_matrix
        )
        
        # Compute similarities with all target words
        similarities = []
        for tgt_word, tgt_vec in self.tgt_embeddings.items():
            sim = self.cosine_similarity(query_vec, tgt_vec)
            similarities.append((tgt_word, sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
    
    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Example usage
en_embeddings = {
    "cat": np.random.randn(100),
    "dog": np.random.randn(100)
}
es_embeddings = {
    "gato": np.random.randn(100),
    "perro": np.random.randn(100)
}

dictionary = [("cat", "gato"), ("dog", "perro")]

aligner = CrossLingualEmbeddings(en_embeddings, es_embeddings)
alignment_matrix = aligner.align(dictionary)
translations = aligner.translate("cat", k=1)

print(f"Translation for 'cat': {translations}")
```

Slide 13: Additional Resources

*   Learning Word Embeddings From Scratch
    *   [https://arxiv.org/abs/1411.2738](https://arxiv.org/abs/1411.2738)
*   Cross-lingual Word Embedding Evaluation
    *   [https://arxiv.org/abs/1902.00508](https://arxiv.org/abs/1902.00508)
*   Contextual Word Representations: A Contextual Introduction
    *   [https://arxiv.org/abs/2108.05542](https://arxiv.org/abs/2108.05542)
*   Advances in Pre-trained Word Embeddings
    *   [https://arxiv.org/abs/2001.12871](https://arxiv.org/abs/2001.12871)
*   Subword-level Word Vector Representations
    *   [https://arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226)

Note: For the most up-to-date research and implementations, you can search for "word embeddings" or "neural language representations" on Google Scholar or arXiv.

