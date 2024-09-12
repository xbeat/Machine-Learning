## Decoding Embedding in LLMs with Python
Slide 1: Introduction to Embeddings in LLMs

Embeddings are dense vector representations of words or tokens in a continuous vector space. They capture semantic relationships between words, allowing Large Language Models (LLMs) to understand and process natural language more effectively. In this presentation, we'll explore how to decode embeddings using Python, providing practical examples along the way.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Example word embeddings
word_embeddings = {
    "king": np.array([0.1, 0.2, 0.3]),
    "queen": np.array([0.15, 0.25, 0.35]),
    "man": np.array([0.05, 0.1, 0.15]),
    "woman": np.array([0.08, 0.13, 0.18])
}

# Calculate cosine similarity between 'king' and 'queen'
similarity = cosine_similarity([word_embeddings["king"]], [word_embeddings["queen"]])[0][0]
print(f"Similarity between 'king' and 'queen': {similarity:.4f}")
```

Slide 2: Loading Pre-trained Embeddings

Pre-trained embeddings, such as Word2Vec or GloVe, can be loaded and used in your projects. These embeddings are trained on large corpora and capture rich semantic information. Let's see how to load GloVe embeddings using Python.

```python
import numpy as np

def load_glove_embeddings(file_path, dimension=100):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if len(vector) == dimension:
                embeddings[word] = vector
    return embeddings

# Load GloVe embeddings
glove_path = 'path/to/glove.6B.100d.txt'
embeddings = load_glove_embeddings(glove_path)

print(f"Loaded {len(embeddings)} word vectors.")
```

Slide 3: Visualizing Word Embeddings

Visualizing high-dimensional embeddings can help us understand the relationships between words. We'll use t-SNE (t-Distributed Stochastic Neighbor Embedding) to reduce the dimensionality of our embeddings and plot them in 2D space.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, words):
    vectors = np.array([embeddings[word] for word in words])
    tsne = TSNE(n_components=2, random_state=42)
    vectors_2d = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], marker='o')
    
    for i, word in enumerate(words):
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]))
    
    plt.title("Word Embeddings Visualization")
    plt.show()

# Visualize a subset of words
words_to_visualize = ["king", "queen", "man", "woman", "prince", "princess", "boy", "girl"]
visualize_embeddings(embeddings, words_to_visualize)
```

Slide 4: Word Analogies with Embeddings

One of the fascinating properties of word embeddings is their ability to capture semantic relationships. We can use vector arithmetic to solve word analogies. Let's implement a function to find the closest word to a given analogy.

```python
def word_analogy(embeddings, word1, word2, word3):
    if word1 not in embeddings or word2 not in embeddings or word3 not in embeddings:
        return "One or more words not found in the embeddings."
    
    target_vector = embeddings[word2] - embeddings[word1] + embeddings[word3]
    
    max_similarity = -1
    most_similar_word = None
    
    for word, vector in embeddings.items():
        if word in [word1, word2, word3]:
            continue
        
        similarity = cosine_similarity([target_vector], [vector])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_word = word
    
    return most_similar_word

# Example: king - man + woman = ?
result = word_analogy(embeddings, "king", "man", "woman")
print(f"king - man + woman = {result}")
```

Slide 5: Sentence Embeddings

While individual word embeddings are useful, we often need to work with entire sentences. One simple approach to create sentence embeddings is to average the embeddings of all words in the sentence. Let's implement this method.

```python
def sentence_embedding(embeddings, sentence):
    words = sentence.lower().split()
    word_vectors = [embeddings[word] for word in words if word in embeddings]
    
    if not word_vectors:
        return None
    
    return np.mean(word_vectors, axis=0)

# Example sentences
sentence1 = "The quick brown fox jumps over the lazy dog"
sentence2 = "A fast auburn canine leaps above the indolent hound"

# Calculate sentence embeddings
emb1 = sentence_embedding(embeddings, sentence1)
emb2 = sentence_embedding(embeddings, sentence2)

# Calculate similarity between sentences
similarity = cosine_similarity([emb1], [emb2])[0][0]
print(f"Similarity between the two sentences: {similarity:.4f}")
```

Slide 6: Fine-tuning Embeddings

Pre-trained embeddings can be fine-tuned for specific tasks or domains. Let's implement a simple fine-tuning process using a basic neural network and backpropagation.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, inputs):
        return self.embeddings(inputs)

# Fine-tuning process
def fine_tune_embeddings(model, data, epochs=10, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0
        for input_word, target_word in data:
            optimizer.zero_grad()
            output = model(input_word)
            loss = criterion(output, target_word)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Example usage (assuming we have prepared data)
vocab_size = 10000
embedding_dim = 100
model = EmbeddingModel(vocab_size, embedding_dim)
fine_tune_embeddings(model, data)
```

Slide 7: Contextual Embeddings with BERT

Unlike static embeddings, contextual embeddings like BERT generate different vectors for the same word based on its context. Let's use the transformers library to obtain BERT embeddings for a sentence.

```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(sentence):
    # Tokenize the sentence and convert to tensor
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the [CLS] token embedding as the sentence embedding
    sentence_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return sentence_embedding

# Example usage
sentence = "The quick brown fox jumps over the lazy dog."
embedding = get_bert_embedding(sentence)
print(f"BERT embedding shape: {embedding.shape}")
```

Slide 8: Embedding Pooling Strategies

When working with embeddings for longer texts or documents, we need strategies to combine word or token embeddings. Let's implement different pooling strategies for BERT embeddings.

```python
import torch
import numpy as np

def bert_embeddings_with_pooling(sentence, model, tokenizer, pooling_strategy='mean'):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get all token embeddings
    token_embeddings = outputs.last_hidden_state[0]
    
    if pooling_strategy == 'mean':
        return torch.mean(token_embeddings, dim=0).numpy()
    elif pooling_strategy == 'max':
        return torch.max(token_embeddings, dim=0)[0].numpy()
    elif pooling_strategy == 'cls':
        return token_embeddings[0].numpy()
    else:
        raise ValueError("Invalid pooling strategy")

# Example usage
sentence = "The quick brown fox jumps over the lazy dog."
mean_pooled = bert_embeddings_with_pooling(sentence, model, tokenizer, 'mean')
max_pooled = bert_embeddings_with_pooling(sentence, model, tokenizer, 'max')
cls_pooled = bert_embeddings_with_pooling(sentence, model, tokenizer, 'cls')

print(f"Mean pooled shape: {mean_pooled.shape}")
print(f"Max pooled shape: {max_pooled.shape}")
print(f"CLS token shape: {cls_pooled.shape}")
```

Slide 9: Embedding-based Text Classification

Embeddings can be used for various downstream tasks, including text classification. Let's implement a simple text classifier using pre-trained embeddings and a neural network.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TextDataset(Dataset):
    def __init__(self, texts, labels, embeddings):
        self.texts = texts
        self.labels = labels
        self.embeddings = embeddings
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text_embedding = sentence_embedding(self.embeddings, self.texts[idx])
        return torch.tensor(text_embedding, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Example usage (assuming we have prepared data and embeddings)
input_dim = 100  # Embedding dimension
hidden_dim = 64
output_dim = 2  # Number of classes

model = TextClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

dataset = TextDataset(texts, labels, embeddings)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(10):
    for batch_embeddings, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_embeddings)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

Slide 10: Embedding-based Semantic Search

Embeddings enable efficient semantic search by comparing the similarity between query and document embeddings. Let's implement a simple semantic search system using cosine similarity.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearch:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings
        self.doc_embeddings = [sentence_embedding(embeddings, doc) for doc in documents]
    
    def search(self, query, top_k=5):
        query_embedding = sentence_embedding(self.embeddings, query)
        similarities = cosine_similarity([query_embedding], self.doc_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'similarity': similarities[idx]
            })
        
        return results

# Example usage
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn canine leaps above the indolent hound.",
    "The lazy cat sleeps all day long.",
    "An energetic puppy plays with a tennis ball."
]

search_engine = SemanticSearch(documents, embeddings)
query = "A rapid fox jumps"
results = search_engine.search(query)

for i, result in enumerate(results, 1):
    print(f"{i}. {result['document']} (Similarity: {result['similarity']:.4f})")
```

Slide 11: Embedding Visualization with t-SNE

To gain insights into the structure of our embedding space, we can use dimensionality reduction techniques like t-SNE. Let's create a visualization of word embeddings using t-SNE and matplotlib.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(words, embeddings, perplexity=30, n_iter=1000):
    # Extract embeddings for the given words
    word_vectors = np.array([embeddings[word] for word in words if word in embeddings])
    words = [word for word in words if word in embeddings]
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reduced_vectors = tsne.fit_transform(word_vectors)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], marker='o')
    
    for i, word in enumerate(words):
        plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]))
    
    plt.title("Word Embeddings Visualization (t-SNE)")
    plt.axis('off')
    plt.show()

# Example usage
words_to_visualize = ["king", "queen", "man", "woman", "prince", "princess", 
                      "dog", "cat", "animal", "pet", "computer", "technology",
                      "book", "read", "write", "author"]

visualize_embeddings(words_to_visualize, embeddings)
```

Slide 12: Embedding Arithmetic for Concept Manipulation

Word embeddings allow for interesting arithmetic operations that can reveal semantic relationships. Let's explore how to perform and visualize these operations.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def embedding_arithmetic(embeddings, positive_words, negative_words, top_n=5):
    result_vector = np.zeros(len(next(iter(embeddings.values()))))
    
    for word in positive_words:
        if word in embeddings:
            result_vector += embeddings[word]
    
    for word in negative_words:
        if word in embeddings:
            result_vector -= embeddings[word]
    
    similarities = []
    for word, vector in embeddings.items():
        if word not in positive_words and word not in negative_words:
            similarity = cosine_similarity([result_vector], [vector])[0][0]
            similarities.append((word, similarity))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

# Example usage
positive_words = ["king", "woman"]
negative_words = ["man"]
result = embedding_arithmetic(embeddings, positive_words, negative_words)

print("King - Man + Woman =")
for word, similarity in result:
    print(f"{word}: {similarity:.4f}")
```

Slide 13: Embedding-based Text Summarization

Embeddings can be used to create extractive text summarizations by selecting the most representative sentences. Let's implement a simple summarization algorithm using sentence embeddings.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def summarize_text(text, embeddings, num_sentences=3):
    # Split text into sentences
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Calculate sentence embeddings
    sentence_embeddings = [sentence_embedding(embeddings, s) for s in sentences]
    
    # Calculate the mean embedding
    mean_embedding = np.mean(sentence_embeddings, axis=0)
    
    # Calculate similarities between each sentence and the mean
    similarities = cosine_similarity(sentence_embeddings, [mean_embedding])
    
    # Get indices of top similar sentences
    top_indices = similarities.argsort(axis=0)[-num_sentences:][::-1]
    
    # Return the summary
    summary = [sentences[i[0]] for i in top_indices]
    return ' '.join(summary)

# Example usage
text = """Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves."""

summary = summarize_text(text, embeddings)
print("Summary:")
print(summary)
```

Slide 14: Cross-lingual Embeddings

Cross-lingual embeddings allow us to represent words from different languages in the same vector space, enabling multilingual applications. Let's explore how to use and visualize cross-lingual embeddings.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_cross_lingual_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def visualize_cross_lingual(embeddings, words_lang1, words_lang2):
    vectors = [embeddings[w] for w in words_lang1 + words_lang2 if w in embeddings]
    labels = [w for w in words_lang1 + words_lang2 if w in embeddings]
    
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_vectors[:len(words_lang1), 0], reduced_vectors[:len(words_lang1), 1], c='blue', label='Language 1')
    plt.scatter(reduced_vectors[len(words_lang1):, 0], reduced_vectors[len(words_lang1):, 1], c='red', label='Language 2')
    
    for i, word in enumerate(labels):
        plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]))
    
    plt.legend()
    plt.title("Cross-lingual Word Embeddings")
    plt.show()

# Example usage (assuming we have cross-lingual embeddings)
cross_lingual_embeddings = load_cross_lingual_embeddings('path/to/cross_lingual_embeddings.txt')

words_english = ['dog', 'cat', 'house', 'car']
words_spanish = ['perro', 'gato', 'casa', 'coche']

visualize_cross_lingual(cross_lingual_embeddings, words_english, words_spanish)
```

Slide 15: Additional Resources

For those interested in diving deeper into the world of embeddings and their applications in LLMs, here are some valuable resources:

1. "Efficient Estimation of Word Representations in Vector Space" by Mikolov et al. (2013) ArXiv: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
2. "GloVe: Global Vectors for Word Representation" by Pennington et al. (2014) ArXiv: [https://arxiv.org/abs/1405.4053](https://arxiv.org/abs/1405.4053)
3. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
4. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" by Reimers and Gurevych (2019) ArXiv: [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)

These papers provide foundational knowledge and advanced techniques in the field of word and sentence embeddings, offering insights into their creation, usage, and impact on natural language processing tasks.

