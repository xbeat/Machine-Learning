## Building an Embeddings Model from Scratch in Python
Slide 1: Introduction to Embeddings

Embeddings are dense vector representations of discrete entities, such as words or sentences, in a continuous vector space. They capture semantic relationships and similarities between entities. In this presentation, we'll explore how to build an embeddings model from scratch using Python.

```python
import numpy as np

# Create a simple word embedding
word_to_index = {"cat": 0, "dog": 1, "fish": 2}
embedding_dim = 3
embedding_matrix = np.random.rand(len(word_to_index), embedding_dim)

print(embedding_matrix)
```

Slide 2: Data Preparation

Before building an embeddings model, we need to prepare our data. This involves tokenizing text, creating a vocabulary, and encoding words as indices.

```python
import re
from collections import Counter

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

text = "The quick brown fox jumps over the lazy dog"
tokens = tokenize(text)
vocab = Counter(tokens)
word_to_index = {word: i for i, (word, _) in enumerate(vocab.most_common())}

print(word_to_index)
```

Slide 3: Building the Training Dataset

We'll create a dataset of word pairs for training our embeddings model. We'll use a simple skip-gram approach, where we predict context words given a target word.

```python
import random

def create_training_data(tokens, window_size=2):
    data = []
    for i, target in enumerate(tokens):
        context = tokens[max(0, i-window_size):i] + tokens[i+1:min(len(tokens), i+window_size+1)]
        data.extend([(target, ctx) for ctx in context])
    return data

training_data = create_training_data(tokens)
print(training_data[:5])
```

Slide 4: Implementing the Embeddings Model

We'll implement a simple embeddings model using PyTorch. Our model will have two embedding layers: one for the target words and one for the context words.

```python
import torch
import torch.nn as nn

class EmbeddingsModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingsModel, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, target, context):
        target_emb = self.target_embeddings(target)
        context_emb = self.context_embeddings(context)
        return torch.sum(target_emb * context_emb, dim=1)

vocab_size = len(word_to_index)
embedding_dim = 50
model = EmbeddingsModel(vocab_size, embedding_dim)
print(model)
```

Slide 5: Preparing Training Data for PyTorch

We need to convert our training data into PyTorch tensors and create a DataLoader for efficient batch processing during training.

```python
from torch.utils.data import Dataset, DataLoader

class WordEmbeddingDataset(Dataset):
    def __init__(self, data, word_to_index):
        self.data = data
        self.word_to_index = word_to_index
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        target, context = self.data[idx]
        return (torch.tensor(self.word_to_index[target]),
                torch.tensor(self.word_to_index[context]))

dataset = WordEmbeddingDataset(training_data, word_to_index)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    print(batch)
    break
```

Slide 6: Training the Embeddings Model

We'll define the loss function and optimizer, then train our model using the prepared data.

```python
import torch.optim as optim

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for target, context in dataloader:
        optimizer.zero_grad()
        output = model(target, context)
        loss = criterion(output, torch.ones_like(output))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
```

Slide 7: Extracting and Visualizing Embeddings

After training, we can extract the learned embeddings and visualize them using techniques like t-SNE for dimensionality reduction.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_embeddings(embeddings, words):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    for i, word in enumerate(words):
        x, y = embeddings_2d[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y))
    plt.show()

embeddings = model.target_embeddings.weight.detach().numpy()
words = list(word_to_index.keys())
plot_embeddings(embeddings, words)
```

Slide 8: Real-Life Example: Semantic Similarity

One common application of word embeddings is measuring semantic similarity between words. Let's implement a function to find the most similar words to a given word.

```python
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def find_similar_words(word, word_to_index, embeddings, top_n=5):
    if word not in word_to_index:
        return []
    
    word_idx = word_to_index[word]
    word_embedding = embeddings[word_idx]
    
    similarities = [(w, cosine_similarity(word_embedding, embeddings[i]))
                    for w, i in word_to_index.items()]
    return sorted(similarities, key=lambda x: x[1], reverse=True)[1:top_n+1]

similar_words = find_similar_words("dog", word_to_index, embeddings)
print(f"Words similar to 'dog': {similar_words}")
```

Slide 9: Real-Life Example: Text Classification

Another common application of embeddings is text classification. Let's build a simple sentiment classifier using our pre-trained word embeddings.

```python
class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_embeddings, hidden_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self.fc = nn.Linear(pretrained_embeddings.shape[1], hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        hidden = torch.relu(self.fc(pooled))
        return self.output(hidden)

pretrained_embeddings = model.target_embeddings.weight.detach()
sentiment_model = SentimentClassifier(pretrained_embeddings, hidden_dim=64, output_dim=1)
print(sentiment_model)
```

Slide 10: Handling Out-of-Vocabulary Words

In real-world applications, we often encounter words that weren't in our original training vocabulary. Let's implement a strategy to handle these out-of-vocabulary (OOV) words.

```python
import numpy as np

def get_word_vector(word, word_to_index, embeddings, oov_vector=None):
    if word in word_to_index:
        return embeddings[word_to_index[word]]
    elif oov_vector is not None:
        return oov_vector
    else:
        # Generate a random vector for OOV words
        return np.random.randn(embeddings.shape[1])

# Create an average OOV vector
oov_vector = np.mean(embeddings, axis=0)

# Test the function
test_words = ["cat", "dog", "unknownword"]
for word in test_words:
    vector = get_word_vector(word, word_to_index, embeddings, oov_vector)
    print(f"Vector for '{word}': {vector[:5]}...")  # Print first 5 elements
```

Slide 11: Improving Embeddings with Subword Information

To handle OOV words better and capture morphological information, we can incorporate subword information into our embeddings model. Let's implement a simple character n-gram based approach.

```python
def get_char_ngrams(word, n=3):
    return ['#' + word[:i] for i in range(1, min(n, len(word)))] + \
           [word[i:i+n] for i in range(len(word)-n+1)] + \
           [word[-i:] + '#' for i in range(1, min(n, len(word)))]

class SubwordEmbeddingsModel(nn.Module):
    def __init__(self, vocab_size, subword_vocab_size, embedding_dim):
        super(SubwordEmbeddingsModel, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.subword_embeddings = nn.Embedding(subword_vocab_size, embedding_dim)
    
    def forward(self, word_ids, subword_ids):
        word_emb = self.word_embeddings(word_ids)
        subword_emb = self.subword_embeddings(subword_ids).mean(dim=1)
        return word_emb + subword_emb

# Example usage
word = "cat"
char_ngrams = get_char_ngrams(word)
print(f"Character n-grams for '{word}': {char_ngrams}")
```

Slide 12: Evaluating Embeddings: Word Analogy Task

A common way to evaluate word embeddings is through analogy tasks. Let's implement a function to solve word analogies using our trained embeddings.

```python
def word_analogy(word1, word2, word3, word_to_index, embeddings, top_n=5):
    if not all(word in word_to_index for word in [word1, word2, word3]):
        return []
    
    vector = (embeddings[word_to_index[word2]] - 
              embeddings[word_to_index[word1]] + 
              embeddings[word_to_index[word3]])
    
    similarities = [(w, cosine_similarity(vector, embeddings[i]))
                    for w, i in word_to_index.items()
                    if w not in [word1, word2, word3]]
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

# Example: man is to king as woman is to ?
result = word_analogy("man", "king", "woman", word_to_index, embeddings)
print(f"man : king :: woman : {result}")
```

Slide 13: Continuous Bag of Words (CBOW) Model

In addition to the skip-gram model, another popular approach for learning word embeddings is the Continuous Bag of Words (CBOW) model. Let's implement a simple CBOW model.

```python
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, contexts):
        embedded = self.embeddings(contexts)
        combined = torch.mean(embedded, dim=1)
        return self.linear(combined)

# Example usage
vocab_size = len(word_to_index)
embedding_dim = 50
cbow_model = CBOWModel(vocab_size, embedding_dim)
print(cbow_model)

# Generate CBOW training data
def create_cbow_data(tokens, window_size=2):
    data = []
    for i in range(len(tokens)):
        context = tokens[max(0, i-window_size):i] + tokens[i+1:min(len(tokens), i+window_size+1)]
        if len(context) == 2 * window_size:
            data.append((context, tokens[i]))
    return data

cbow_data = create_cbow_data(tokens)
print(f"CBOW data sample: {cbow_data[:3]}")
```

Slide 14: Fine-tuning Pre-trained Embeddings

In many cases, it's beneficial to start with pre-trained embeddings and fine-tune them for a specific task. Let's implement a simple fine-tuning process for sentiment analysis.

```python
class SentimentClassifierFineTuned(nn.Module):
    def __init__(self, pretrained_embeddings, hidden_dim, output_dim, fine_tune=True):
        super(SentimentClassifierFineTuned, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=not fine_tune)
        self.lstm = nn.LSTM(pretrained_embeddings.shape[1], hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))

# Example usage
pretrained_embeddings = model.target_embeddings.weight.detach()
sentiment_model_finetuned = SentimentClassifierFineTuned(pretrained_embeddings, hidden_dim=64, output_dim=1)
print(sentiment_model_finetuned)

# Fine-tuning process (pseudo-code)
"""
for epoch in range(num_epochs):
    for batch in dataloader:
        texts, labels = batch
        outputs = sentiment_model_finetuned(texts)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""
```

Slide 15: Additional Resources

For those interested in diving deeper into word embeddings and natural language processing, here are some valuable resources:

1. "Efficient Estimation of Word Representations in Vector Space" by Mikolov et al. (2013) ArXiv: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
2. "GloVe: Global Vectors for Word Representation" by Pennington et al. (2014) ArXiv: [https://arxiv.org/abs/1405.3531](https://arxiv.org/abs/1405.3531)
3. "Enriching Word Vectors with Subword Information" by Bojanowski et al. (2016) ArXiv: [https://arxiv.org/abs/1607.04606](https://arxiv.org/abs/1607.04606)
4. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

These papers provide in-depth explanations of various embedding techniques and their applications in natural language processing tasks.

