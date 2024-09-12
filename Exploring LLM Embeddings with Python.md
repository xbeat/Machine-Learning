## Exploring LLM Embeddings with Python
Slide 1: What are Embeddings?

Embeddings are dense vector representations of data that capture semantic meaning and relationships. They map high-dimensional discrete data (like words or images) into continuous low-dimensional vector spaces where similar items are closer together.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Sample sentences
sentences = ["I love machine learning", "I enjoy natural language processing"]

# Create embeddings using CountVectorizer
vectorizer = CountVectorizer()
embeddings = vectorizer.fit_transform(sentences).toarray()

print("Embeddings shape:", embeddings.shape)
print("Embeddings:\n", embeddings)
```

Slide 2: Word Embeddings

Word embeddings represent words as dense vectors in a continuous vector space. They capture semantic relationships between words, allowing for meaningful operations like finding similar words or analogies.

```python
from gensim.models import Word2Vec

# Sample corpus
corpus = [["cat", "dog", "pet"], ["machine", "learning", "AI"]]

# Train Word2Vec model
model = Word2Vec(sentences=corpus, vector_size=10, window=2, min_count=1, workers=4)

# Get vector for a word
cat_vector = model.wv["cat"]
print("Vector for 'cat':", cat_vector)

# Find similar words
similar_words = model.wv.most_similar("dog", topn=2)
print("Words similar to 'dog':", similar_words)
```

Slide 3: Document Embeddings

Document embeddings represent entire documents as dense vectors, capturing their overall meaning and content. They are useful for tasks like document classification and similarity comparison.

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Sample documents
documents = ["I love machine learning", "Natural language processing is fascinating"]
tagged_docs = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(documents)]

# Train Doc2Vec model
model = Doc2Vec(vector_size=10, min_count=1, epochs=20)
model.build_vocab(tagged_docs)
model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

# Get vector for a document
doc_vector = model.infer_vector("I enjoy deep learning".split())
print("Document vector:", doc_vector)
```

Slide 4: Image Embeddings

Image embeddings represent images as dense vectors, capturing visual features and semantic content. They are commonly used in computer vision tasks like image classification and similarity search.

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess image
image = Image.open("example_image.jpg")
input_tensor = transform(image).unsqueeze(0)

# Generate embedding
with torch.no_grad():
    embedding = model(input_tensor)

print("Image embedding shape:", embedding.shape)
```

Slide 5: Creating Embeddings with TensorFlow

TensorFlow provides tools for creating and working with embeddings. Here's an example of creating word embeddings using TensorFlow's Embedding layer.

```python
import tensorflow as tf

# Sample vocabulary
vocab = ["cat", "dog", "bird", "fish"]
vocab_size = len(vocab)
embedding_dim = 5

# Create embedding layer
embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim)

# Create input indices
input_indices = tf.constant([0, 2, 1, 3])  # Indices of words in vocab

# Get embeddings
embeddings = embedding_layer(input_indices)

print("Embeddings shape:", embeddings.shape)
print("Embeddings:\n", embeddings.numpy())
```

Slide 6: Cosine Similarity in Embeddings

Cosine similarity is a common metric used to measure the similarity between embeddings. It calculates the cosine of the angle between two vectors, with values closer to 1 indicating higher similarity.

```python
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Sample embeddings
embedding1 = np.array([0.2, 0.5, 0.1, 0.8])
embedding2 = np.array([0.3, 0.4, 0.2, 0.7])
embedding3 = np.array([0.9, 0.1, 0.7, 0.2])

# Calculate similarities
sim_1_2 = cosine_similarity(embedding1, embedding2)
sim_1_3 = cosine_similarity(embedding1, embedding3)

print("Similarity between embedding1 and embedding2:", sim_1_2)
print("Similarity between embedding1 and embedding3:", sim_1_3)
```

Slide 7: Visualizing Embeddings with t-SNE

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a technique for visualizing high-dimensional embeddings in 2D or 3D space, preserving local relationships between data points.

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Generate sample embeddings
num_embeddings = 100
embedding_dim = 50
embeddings = np.random.randn(num_embeddings, embedding_dim)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Visualize
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.title("t-SNE Visualization of Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
```

Slide 8: Word Analogies with Embeddings

Word embeddings can be used to perform word analogies, leveraging the semantic relationships captured in the vector space.

```python
import gensim.downloader as api

# Load pre-trained word vectors
word_vectors = api.load("glove-wiki-gigaword-100")

# Perform word analogy
result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])

print("Word analogy: woman is to man as king is to:")
for word, score in result:
    print(f"{word}: {score:.4f}")
```

Slide 9: Sentence Embeddings with BERT

BERT (Bidirectional Encoder Representations from Transformers) can be used to generate contextualized embeddings for sentences, capturing complex semantic relationships.

```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Prepare input
text = "Machine learning is fascinating."
inputs = tokenizer(text, return_tensors="pt")

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Get sentence embedding (using mean pooling)
sentence_embedding = outputs.last_hidden_state.mean(dim=1)

print("Sentence embedding shape:", sentence_embedding.shape)
```

Slide 10: Fine-tuning Embeddings for Specific Tasks

Pre-trained embeddings can be fine-tuned for specific tasks to improve performance. Here's an example of fine-tuning word embeddings for sentiment analysis.

```python
import torch
import torch.nn as nn
from torchtext.vocab import GloVe

# Load pre-trained GloVe embeddings
glove = GloVe(name='6B', dim=100)

# Define a simple sentiment analysis model
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(glove.vectors)
        self.fc = nn.Linear(embedding_dim, 2)  # Binary classification

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        return self.fc(pooled)

# Create model
model = SentimentModel(len(glove.vectors), 100)

# Fine-tuning loop (simplified)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    # Training code here
    pass

print("Fine-tuning complete")
```

Slide 11: Real-life Example: Content-based Recommendation System

Embeddings can be used to build content-based recommendation systems. Here's a simple example using movie descriptions to recommend similar movies.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie descriptions
movies = {
    "Movie A": "A thrilling action adventure with stunning visual effects",
    "Movie B": "A heartwarming romantic comedy set in Paris",
    "Movie C": "An epic science fiction journey through space and time",
    "Movie D": "A gripping thriller with unexpected plot twists",
}

# Create TF-IDF embeddings
vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(movies.values())

# Function to get similar movies
def get_similar_movies(movie_title, top_n=2):
    movie_index = list(movies.keys()).index(movie_title)
    similarities = cosine_similarity(embeddings[movie_index], embeddings).flatten()
    similar_indices = similarities.argsort()[::-1][1:top_n+1]
    return [list(movies.keys())[i] for i in similar_indices]

# Get recommendations
recommendations = get_similar_movies("Movie A")
print("Movies similar to 'Movie A':", recommendations)
```

Slide 12: Real-life Example: Semantic Search Engine

Embeddings enable semantic search, allowing users to find relevant documents based on meaning rather than exact keyword matches.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Sample documents
documents = [
    "The Eiffel Tower is an iconic landmark in Paris, France.",
    "The Great Wall of China stretches over 13,000 miles.",
    "The Statue of Liberty stands tall in New York Harbor.",
    "The Taj Mahal is a beautiful mausoleum in Agra, India.",
]

# Load pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Create embeddings for documents
doc_embeddings = model.encode(documents)

# Function for semantic search
def semantic_search(query, top_n=2):
    query_embedding = model.encode([query])
    similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    return [documents[i] for i in top_indices]

# Perform semantic search
results = semantic_search("Famous structures in Asia")
print("Search results:")
for result in results:
    print("-", result)
```

Slide 13: Challenges and Limitations of Embeddings

Embeddings, while powerful, have limitations. They may struggle with rare words, out-of-vocabulary terms, and capturing complex context. Bias in training data can also lead to biased embeddings.

```python
import gensim.downloader as api

# Load pre-trained word vectors
word_vectors = api.load("glove-wiki-gigaword-100")

# Example of limitation: out-of-vocabulary word
try:
    vector = word_vectors["supercalifragilisticexpialidocious"]
except KeyError:
    print("Word 'supercalifragilisticexpialidocious' not in vocabulary")

# Example of potential bias
male_words = ["he", "man", "boy"]
female_words = ["she", "woman", "girl"]

for male, female in zip(male_words, female_words):
    similarity = word_vectors.similarity(male, "strong")
    print(f"Similarity between '{male}' and 'strong': {similarity:.4f}")
    similarity = word_vectors.similarity(female, "strong")
    print(f"Similarity between '{female}' and 'strong': {similarity:.4f}")
    print()
```

Slide 14: Future Directions in Embedding Technology

Embedding technology continues to evolve. Current research focuses on multimodal embeddings, contextual embeddings, and more efficient training methods. These advancements aim to capture richer semantic information and improve performance across various NLP tasks.

```python
import torch
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model (multimodal embedding model)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Example input
image_url = "https://example.com/image.jpg"
text = "A cute cat playing with a ball"

# Process inputs
inputs = processor(text=[text], images=[image_url], return_tensors="pt", padding=True)

# Generate multimodal embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Get image and text embeddings
image_embeds = outputs.image_embeds
text_embeds = outputs.text_embeds

print("Image embedding shape:", image_embeds.shape)
print("Text embedding shape:", text_embeds.shape)
```

Slide 15: Additional Resources

For more in-depth information on embeddings and their applications in natural language processing and machine learning, consider exploring the following resources:

1. "Efficient Estimation of Word Representations in Vector Space" by Mikolov et al. (2013) ArXiv: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
2. "GloVe: Global Vectors for Word Representation" by Pennington et al. (2014) ArXiv: [https://arxiv.org/abs/1405.4053](https://arxiv.org/abs/1405.4053)
3. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
4. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" by Reimers and Gurevych (2019) ArXiv: [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)

These papers provide foundational knowledge and advanced techniques in the field of embeddings and their applications in natural language processing.

