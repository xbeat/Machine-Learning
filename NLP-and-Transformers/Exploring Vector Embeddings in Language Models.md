## Exploring Vector Embeddings in Language Models
Slide 1: Vector Embedding Fundamentals

Vector embeddings form the foundation of modern language models by transforming discrete tokens into continuous vector spaces. These dense representations capture semantic relationships through learned weights in neural networks, enabling mathematical operations on language.

```python
import numpy as np
from sklearn.preprocessing import normalize

class WordEmbedding:
    def __init__(self, vocab_size, embedding_dim):
        # Initialize random embeddings matrix
        self.embeddings = normalize(
            np.random.randn(vocab_size, embedding_dim)
        )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
    def get_embedding(self, word_idx):
        return self.embeddings[word_idx]
    
    def cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Example usage
embedder = WordEmbedding(vocab_size=10000, embedding_dim=300)
word_vector = embedder.get_embedding(42)
print(f"Shape of word vector: {word_vector.shape}")
```

Slide 2: Mathematical Operations with Word Vectors

Word vectors enable arithmetic operations that preserve semantic relationships. The classic example of king - man + woman = queen demonstrates how embeddings capture analogical reasoning through vector arithmetic in the embedding space.

```python
import numpy as np
from scipy.spatial.distance import cosine

def vector_analogy(king, man, woman):
    """
    Implements: king - man + woman = queen
    Returns the result vector of the analogy operation
    """
    # Normalize vectors
    king = king / np.linalg.norm(king)
    man = man / np.linalg.norm(man)
    woman = woman / np.linalg.norm(woman)
    
    # Perform vector arithmetic
    result = king - man + woman
    
    # Normalize result
    return result / np.linalg.norm(result)

# Example with random vectors
dim = 300
king_vec = np.random.randn(dim)
man_vec = np.random.randn(dim)
woman_vec = np.random.randn(dim)

result_vec = vector_analogy(king_vec, man_vec, woman_vec)
print(f"Resulting vector shape: {result_vec.shape}")
```

Slide 3: Embedding Space Visualization

Understanding the high-dimensional embedding space requires dimensionality reduction techniques. This implementation uses t-SNE to project word vectors into 2D space while preserving relative distances and relationships between words.

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, words, perplexity=30):
    # Reduce dimensionality to 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_vecs = tsne.fit_transform(embeddings)
    
    # Plot vectors
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_vecs[:, 0], reduced_vecs[:, 1], alpha=0.6)
    
    # Add word labels
    for i, word in enumerate(words):
        plt.annotate(word, xy=(reduced_vecs[i, 0], reduced_vecs[i, 1]))
    
    plt.title('Word Embeddings Visualization')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    return plt

# Example usage with random embeddings
words = ['king', 'queen', 'man', 'woman', 'prince', 'princess']
embeddings = np.random.randn(len(words), 300)
visualize_embeddings(embeddings, words)
```

Slide 4: Context Window Implementation

The context window is crucial for training word embeddings, as it defines the local scope for learning word relationships. This implementation creates context-target pairs for training embedding models.

```python
class ContextWindowGenerator:
    def __init__(self, window_size=2):
        self.window_size = window_size
    
    def generate_contexts(self, sentence):
        words = sentence.split()
        contexts = []
        
        for i, target in enumerate(words):
            # Define context range
            start = max(0, i - self.window_size)
            end = min(len(words), i + self.window_size + 1)
            
            # Get context words
            context = (
                words[start:i] + 
                words[i+1:end]
            )
            contexts.append((target, context))
            
        return contexts

# Example usage
generator = ContextWindowGenerator(window_size=2)
sentence = "the quick brown fox jumps over"
contexts = generator.generate_contexts(sentence)

for target, context in contexts:
    print(f"Target: {target:<10} Context: {context}")
```

Slide 5: Word2Vec Skip-gram Implementation

The Skip-gram model predicts context words given a target word. This implementation shows the core architecture of the Skip-gram model, including the forward pass and negative sampling for efficient training.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        # Get embeddings for input words
        embeds = self.embeddings(x)
        # Project to vocabulary space
        output = self.output_layer(embeds)
        return output

    def get_word_embedding(self, word_idx):
        return self.embeddings.weight[word_idx].detach()

# Training setup
vocab_size = 5000
embedding_dim = 300
model = SkipGram(vocab_size, embedding_dim)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Example forward pass
input_idx = torch.tensor([1, 2, 3])
output = model(input_idx)
print(f"Output shape: {output.shape}")
```

Slide 6: Negative Sampling Implementation

Negative sampling optimizes training by contrasting target context pairs with randomly sampled negative examples. This implementation shows how to generate and use negative samples effectively.

```python
import numpy as np
import torch
import torch.nn.functional as F

class NegativeSampling:
    def __init__(self, word_freqs, n_samples=5):
        """
        word_freqs: array of word frequencies
        n_samples: number of negative samples per positive
        """
        self.n_samples = n_samples
        # Create distribution for negative sampling
        freq_sum = np.sum(word_freqs ** 0.75)
        self.sample_probs = (word_freqs ** 0.75) / freq_sum
        
    def get_negative_samples(self, positive_idx, batch_size):
        # Generate negative samples
        neg_samples = np.random.choice(
            len(self.sample_probs),
            size=(batch_size, self.n_samples),
            p=self.sample_probs
        )
        # Ensure negative samples != positive
        mask = neg_samples == positive_idx
        neg_samples[mask] = np.random.randint(0, len(self.sample_probs))
        return torch.LongTensor(neg_samples)

# Example usage
word_freqs = np.random.rand(vocab_size)
sampler = NegativeSampling(word_freqs)
neg_samples = sampler.get_negative_samples(42, batch_size=16)
print(f"Negative samples shape: {neg_samples.shape}")
```

Slide 7: Contextual Embeddings with Attention

Contextual embeddings improve upon static embeddings by incorporating surrounding context. This implementation demonstrates a simple attention mechanism for context-aware representations.

```python
import torch
import torch.nn as nn

class ContextualEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            batch_first=True
        )
        self.pos_encoding = self.create_positional_encoding(
            max_seq_length, 
            embedding_dim
        )
        
    def create_positional_encoding(self, max_len, d_model):
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                           (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe
        
    def forward(self, x):
        # Get embeddings and add positional encoding
        embeds = self.embedding(x) + self.pos_encoding[:x.size(1)]
        # Apply self-attention
        context, _ = self.attention(embeds, embeds, embeds)
        return context

# Example usage
model = ContextualEmbedding(vocab_size=5000, 
                           embedding_dim=256, 
                           max_seq_length=512)
input_ids = torch.randint(0, 5000, (8, 32))  # batch_size=8, seq_len=32
outputs = model(input_ids)
print(f"Contextual embeddings shape: {outputs.shape}")
```

Slide 8: Embedding Space Metrics

Quantitative evaluation of embedding quality through various distance metrics helps assess the semantic coherence of the vector space representation.

```python
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingMetrics:
    @staticmethod
    def cosine_similarity(v1, v2):
        return 1 - cosine(v1, v2)
    
    @staticmethod
    def euclidean_distance(v1, v2):
        return euclidean(v1, v2)
    
    @staticmethod
    def neighborhood_similarity(emb1, emb2, k=5):
        """
        Compare k-nearest neighbors for two embeddings
        """
        sim1 = cosine_similarity(emb1.reshape(1, -1), emb1)
        sim2 = cosine_similarity(emb2.reshape(1, -1), emb2)
        
        # Get top k neighbors
        neighbors1 = np.argsort(sim1[0])[-k:]
        neighbors2 = np.argsort(sim2[0])[-k:]
        
        # Calculate Jaccard similarity
        intersection = len(set(neighbors1) & set(neighbors2))
        union = len(set(neighbors1) | set(neighbors2))
        return intersection / union

# Example usage
metrics = EmbeddingMetrics()
v1 = np.random.randn(300)
v2 = np.random.randn(300)

print(f"Cosine similarity: {metrics.cosine_similarity(v1, v2):.4f}")
print(f"Euclidean distance: {metrics.euclidean_distance(v1, v2):.4f}")
```

Slide 9: Real-world Application - Sentiment Analysis with Embeddings

This implementation demonstrates how word embeddings can be used for sentiment analysis of product reviews, including data preprocessing and a neural classifier built on top of the embeddings.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_embeddings, hidden_dim=128):
        super().__init__()
        vocab_size, embedding_dim = pretrained_embeddings.shape
        
        # Freeze pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_embeddings),
            freeze=True
        )
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, lengths):
        # Get embeddings
        embedded = self.embedding(x)
        
        # Pack sequence for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        
        # Process through LSTM
        output, (hidden, _) = self.lstm(packed)
        
        # Concatenate bidirectional hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Classify
        return self.classifier(hidden)

# Example usage
vocab_size = 10000
embedding_dim = 300
pretrained_embeddings = np.random.randn(vocab_size, embedding_dim)

model = SentimentClassifier(pretrained_embeddings)
batch_size = 32
seq_length = 50
x = torch.randint(0, vocab_size, (batch_size, seq_length))
lengths = torch.randint(1, seq_length, (batch_size,))
output = model(x, lengths)
print(f"Output shape: {output.shape}")
```

Slide 10: Advanced Word Vector Operations

Implementation of more sophisticated vector operations for exploring semantic relationships and analogies in the embedding space, with mathematical justification.

```python
import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple

class WordVectorOperations:
    def __init__(self, word_vectors: np.ndarray, vocabulary: List[str]):
        self.word_vectors = word_vectors
        self.vocabulary = {word: idx for idx, word in enumerate(vocabulary)}
        self.inverse_vocab = {idx: word for word, idx in self.vocabulary.items()}
        
    def find_analogies(self, 
                      word1: str, 
                      word2: str, 
                      word3: str, 
                      n_results: int = 5) -> List[Tuple[str, float]]:
        """
        Solves analogies of the form: word1 : word2 :: word3 : ?
        Example: king : man :: queen : woman
        """
        # Get word vectors
        v1 = self.word_vectors[self.vocabulary[word1]]
        v2 = self.word_vectors[self.vocabulary[word2]]
        v3 = self.word_vectors[self.vocabulary[word3]]
        
        # Compute target vector
        target_vector = v2 - v1 + v3
        
        # Calculate cosine distances to all words
        distances = cdist(
            target_vector.reshape(1, -1),
            self.word_vectors,
            metric='cosine'
        )[0]
        
        # Get top n_results (excluding input words)
        exclude_idxs = {self.vocabulary[w] for w in [word1, word2, word3]}
        sorted_idxs = np.argsort(distances)
        
        results = []
        for idx in sorted_idxs:
            if idx not in exclude_idxs:
                results.append(
                    (self.inverse_vocab[idx], float(distances[idx]))
                )
                if len(results) == n_results:
                    break
                    
        return results
    
    def semantic_direction(self, 
                         positive: List[str], 
                         negative: List[str]) -> np.ndarray:
        """
        Computes semantic direction from positive and negative examples
        Example: direction = avg(positive_vectors) - avg(negative_vectors)
        """
        pos_vecs = [self.word_vectors[self.vocabulary[w]] for w in positive]
        neg_vecs = [self.word_vectors[self.vocabulary[w]] for w in negative]
        
        direction = (np.mean(pos_vecs, axis=0) - 
                    np.mean(neg_vecs, axis=0))
        return direction / np.linalg.norm(direction)

# Example usage
dim = 300
vocab_size = 1000
word_vectors = np.random.randn(vocab_size, dim)
vocabulary = [f"word_{i}" for i in range(vocab_size)]

ops = WordVectorOperations(word_vectors, vocabulary)
analogies = ops.find_analogies("word_1", "word_2", "word_3")
print(f"Top analogies: {analogies}")
```

Slide 11: Subword Tokenization and Embeddings

Subword tokenization helps handle out-of-vocabulary words by breaking them into meaningful subunits. This implementation shows how to create and use subword embeddings with byte-pair encoding (BPE).

```python
from collections import defaultdict
import re
from typing import Dict, List, Tuple

class BPETokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.char_to_idx = {}
        self.merges = {}
        self.vocab = set()
        
    def train(self, texts: List[str]) -> None:
        # Initialize character vocabulary
        words = ' '.join(texts).split()
        char_freqs = defaultdict(int)
        
        for word in words:
            chars = ' '.join(list(word))
            for char in chars.split():
                char_freqs[char] += 1
                
        # Initialize merge vocabulary with characters
        self.vocab = set(char_freqs.keys())
        
        while len(self.vocab) < self.vocab_size:
            # Find most frequent pair
            pairs = self.get_stats(words)
            if not pairs:
                break
                
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            self.merges[best_pair] = ''.join(best_pair)
            self.vocab.add(self.merges[best_pair])
            
            # Update corpus with merged pair
            new_words = []
            for word in words:
                new_word = self.merge_word(word, best_pair)
                new_words.append(new_word)
            words = new_words
            
    def get_stats(self, words: List[str]) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for word in words:
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += 1
        return pairs
        
    def merge_word(self, 
                   word: str, 
                   pair: Tuple[str, str]) -> str:
        parts = word.split()
        i = 0
        while i < len(parts) - 1:
            if (parts[i], parts[i+1]) == pair:
                parts[i] = self.merges[pair]
                parts.pop(i+1)
            else:
                i += 1
        return ' '.join(parts)
        
    def tokenize(self, text: str) -> List[str]:
        words = text.split()
        tokens = []
        
        for word in words:
            word = ' '.join(list(word))
            while True:
                min_pair = None
                min_idx = float('inf')
                
                for pair, merge in self.merges.items():
                    try:
                        idx = word.index(' '.join(pair))
                        if idx < min_idx:
                            min_idx = idx
                            min_pair = pair
                    except ValueError:
                        continue
                        
                if min_pair is None:
                    break
                    
                word = self.merge_word(word, min_pair)
            
            tokens.extend(word.split())
            
        return tokens

# Example usage
texts = [
    "hello world",
    "world peace",
    "hello peace"
]

tokenizer = BPETokenizer(vocab_size=100)
tokenizer.train(texts)

text = "hello peaceful world"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
```

Slide 12: Real-world Application - Document Similarity

This implementation shows how to use document embeddings for finding similar documents, implementing both TF-IDF weighted embeddings and document-level attention.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple

class DocumentSimilarity:
    def __init__(self, 
                 word_embeddings: np.ndarray,
                 vocabulary: List[str]):
        self.word_embeddings = word_embeddings
        self.vocabulary = {word: idx for idx, word in enumerate(vocabulary)}
        self.tfidf = TfidfVectorizer(vocabulary=vocabulary)
        
    def get_document_embedding(self, 
                             doc: str, 
                             method: str = 'tfidf') -> np.ndarray:
        """
        Convert document to embedding using specified method
        """
        if method == 'tfidf':
            # Get TF-IDF weights
            tfidf_matrix = self.tfidf.fit_transform([doc])
            weights = np.array(tfidf_matrix.todense())[0]
            
            # Weight embeddings
            doc_embedding = np.zeros(self.word_embeddings.shape[1])
            for word in doc.split():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    weight = weights[idx]
                    doc_embedding += weight * self.word_embeddings[idx]
                    
            # Normalize
            norm = np.linalg.norm(doc_embedding)
            if norm > 0:
                doc_embedding /= norm
                
        elif method == 'attention':
            # Implement self-attention over word embeddings
            words = [w for w in doc.split() if w in self.vocabulary]
            if not words:
                return np.zeros(self.word_embeddings.shape[1])
                
            # Get word embeddings for document
            word_vectors = np.array([
                self.word_embeddings[self.vocabulary[w]] 
                for w in words
            ])
            
            # Calculate attention scores
            scores = np.dot(word_vectors, word_vectors.T)
            scores = np.exp(scores)
            scores /= scores.sum(axis=1, keepdims=True)
            
            # Apply attention
            doc_embedding = np.dot(scores, word_vectors).mean(axis=0)
            
        return doc_embedding
        
    def find_similar_documents(self,
                             query_doc: str,
                             corpus: List[str],
                             n_results: int = 5,
                             method: str = 'tfidf') -> List[Tuple[int, float]]:
        """
        Find most similar documents to query document
        """
        # Get query embedding
        query_embedding = self.get_document_embedding(query_doc, method)
        
        # Get corpus embeddings
        corpus_embeddings = np.array([
            self.get_document_embedding(doc, method)
            for doc in corpus
        ])
        
        # Calculate similarities
        similarities = np.dot(corpus_embeddings, query_embedding)
        
        # Get top results
        top_idxs = np.argsort(similarities)[-n_results:][::-1]
        results = [(idx, similarities[idx]) for idx in top_idxs]
        
        return results

# Example usage
dim = 300
vocab_size = 1000
word_embeddings = np.random.randn(vocab_size, dim)
vocabulary = [f"word_{i}" for i in range(vocab_size)]

doc_sim = DocumentSimilarity(word_embeddings, vocabulary)
query = "word_1 word_2 word_3"
corpus = ["word_1 word_4", "word_2 word_3", "word_5 word_6"]
results = doc_sim.find_similar_documents(query, corpus)
print(f"Similar documents: {results}")
```

Slide 13: Embedding Compression and Quantization

Implementing efficient storage and retrieval of embeddings through dimensionality reduction and quantization techniques, crucial for deploying embedding models in production environments.

```python
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple

class EmbeddingCompressor:
    def __init__(self, n_components: int = 100, n_bits: int = 8):
        self.n_components = n_components
        self.n_bits = n_bits
        self.pca = PCA(n_components=n_components)
        self.min_vals = None
        self.max_vals = None
        
    def compress(self, embeddings: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Compress embeddings using PCA and quantization
        """
        # Apply PCA
        reduced_embeddings = self.pca.fit_transform(embeddings)
        
        # Prepare for quantization
        self.min_vals = reduced_embeddings.min(axis=0)
        self.max_vals = reduced_embeddings.max(axis=0)
        
        # Quantize to n_bits
        quantized = self._quantize(reduced_embeddings)
        
        metadata = {
            'explained_variance': self.pca.explained_variance_ratio_.sum(),
            'components': self.pca.components_,
            'min_vals': self.min_vals,
            'max_vals': self.max_vals
        }
        
        return quantized, metadata
    
    def decompress(self, 
                  quantized: np.ndarray, 
                  metadata: dict) -> np.ndarray:
        """
        Decompress quantized embeddings back to original space
        """
        # Dequantize
        dequantized = self._dequantize(
            quantized, 
            metadata['min_vals'], 
            metadata['max_vals']
        )
        
        # Transform back to original space
        reconstructed = np.dot(dequantized, metadata['components'])
        
        return reconstructed
        
    def _quantize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Quantize floating point values to integers
        """
        scale = (2 ** self.n_bits) - 1
        normalized = (embeddings - self.min_vals) / (
            self.max_vals - self.min_vals
        )
        quantized = np.round(normalized * scale).astype(np.uint8)
        return quantized
        
    def _dequantize(self, 
                    quantized: np.ndarray,
                    min_vals: np.ndarray,
                    max_vals: np.ndarray) -> np.ndarray:
        """
        Dequantize integers back to floating point values
        """
        scale = (2 ** self.n_bits) - 1
        normalized = quantized.astype(np.float32) / scale
        dequantized = normalized * (max_vals - min_vals) + min_vals
        return dequantized

# Example usage
embeddings = np.random.randn(1000, 300)
compressor = EmbeddingCompressor(n_components=100, n_bits=8)

# Compress
quantized, metadata = compressor.compress(embeddings)
print(f"Original size: {embeddings.nbytes/1024:.2f}KB")
print(f"Compressed size: {quantized.nbytes/1024:.2f}KB")

# Decompress
reconstructed = compressor.decompress(quantized, metadata)
error = np.mean((embeddings - reconstructed) ** 2)
print(f"Mean squared error: {error:.6f}")
```

Slide 14: Evaluation Metrics for Word Embeddings

A comprehensive suite of evaluation metrics for word embeddings, including both intrinsic and extrinsic evaluation methods.

```python
import numpy as np
from scipy.stats import spearmanr
from typing import List, Dict, Tuple

class EmbeddingEvaluator:
    def __init__(self, 
                 word_vectors: np.ndarray,
                 vocabulary: Dict[str, int]):
        self.word_vectors = word_vectors
        self.vocabulary = vocabulary
        
    def evaluate_word_similarity(self, 
                               word_pairs: List[Tuple[str, str]], 
                               human_scores: List[float]) -> float:
        """
        Evaluate using word similarity benchmarks (e.g., WordSim353)
        """
        model_scores = []
        valid_pairs = []
        
        for (w1, w2), human_score in zip(word_pairs, human_scores):
            if w1 in self.vocabulary and w2 in self.vocabulary:
                v1 = self.word_vectors[self.vocabulary[w1]]
                v2 = self.word_vectors[self.vocabulary[w2]]
                
                # Calculate cosine similarity
                sim = np.dot(v1, v2) / (
                    np.linalg.norm(v1) * np.linalg.norm(v2)
                )
                
                model_scores.append(sim)
                valid_pairs.append(human_score)
        
        # Calculate Spearman correlation
        correlation, _ = spearmanr(model_scores, valid_pairs)
        return correlation
    
    def evaluate_analogy(self, 
                        analogies: List[Tuple[str, str, str, str]]) -> float:
        """
        Evaluate using analogy tasks (e.g., king:man::queen:woman)
        """
        correct = 0
        total = 0
        
        for a, b, c, d in analogies:
            if all(w in self.vocabulary for w in [a, b, c]):
                # Get vectors
                va = self.word_vectors[self.vocabulary[a]]
                vb = self.word_vectors[self.vocabulary[b]]
                vc = self.word_vectors[self.vocabulary[c]]
                
                # Calculate target vector
                target = vb - va + vc
                
                # Find nearest neighbor (excluding a, b, c)
                similarities = np.dot(self.word_vectors, target) / (
                    np.linalg.norm(self.word_vectors, axis=1) * 
                    np.linalg.norm(target)
                )
                
                # Exclude input words
                exclude_idxs = {
                    self.vocabulary[w] for w in [a, b, c]
                }
                for idx in exclude_idxs:
                    similarities[idx] = -np.inf
                
                pred_idx = np.argmax(similarities)
                pred_word = list(self.vocabulary.keys())[
                    list(self.vocabulary.values()).index(pred_idx)
                ]
                
                if pred_word == d:
                    correct += 1
                total += 1
                
        return correct / total if total > 0 else 0.0

# Example usage
dim = 300
vocab_size = 1000
word_vectors = np.random.randn(vocab_size, dim)
vocabulary = {f"word_{i}": i for i in range(vocab_size)}

evaluator = EmbeddingEvaluator(word_vectors, vocabulary)

# Word similarity evaluation
word_pairs = [("word_1", "word_2"), ("word_3", "word_4")]
human_scores = [0.8, 0.3]
sim_score = evaluator.evaluate_word_similarity(word_pairs, human_scores)
print(f"Word similarity correlation: {sim_score:.4f}")

# Analogy evaluation
analogies = [
    ("word_1", "word_2", "word_3", "word_4"),
    ("word_5", "word_6", "word_7", "word_8")
]
analogy_score = evaluator.evaluate_analogy(analogies)
print(f"Analogy accuracy: {analogy_score:.4f}")
```

Slide 15: Additional Resources

*   ArXiv Papers:
    *   "Efficient Estimation of Word Representations in Vector Space" - [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
    *   "GloVe: Global Vectors for Word Representation" - [https://arxiv.org/abs/1405.4053](https://arxiv.org/abs/1405.4053)
    *   "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" - [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
    *   "Improving Word Representations via Global Context and Multiple Word Prototypes" - [https://arxiv.org/abs/1510.04009](https://arxiv.org/abs/1510.04009)
    *   "Language Models are Few-Shot Learners" - [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

For additional implementation details and research:

*   Stanford CS224N Natural Language Processing Course
*   Papers with Code - Word Embeddings section
*   Hugging Face Transformers Documentation

