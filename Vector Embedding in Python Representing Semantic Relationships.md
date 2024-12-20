## Vector Embedding in Python Representing Semantic Relationships
Slide 1: Introduction to Vector Embedding

Vector embedding is a technique used to represent discrete objects as continuous vectors in a high-dimensional space. This method allows us to capture semantic relationships and similarities between objects, making it easier for machine learning models to process and analyze data.

```python
import numpy as np

# Example: Word embedding
word_embedding = {
    "cat": np.array([0.2, 0.5, -0.3]),
    "dog": np.array([0.1, 0.4, -0.2]),
    "fish": np.array([-0.3, 0.1, 0.6])
}

# Calculate cosine similarity between "cat" and "dog"
cat_dog_similarity = np.dot(word_embedding["cat"], word_embedding["dog"]) / (
    np.linalg.norm(word_embedding["cat"]) * np.linalg.norm(word_embedding["dog"])
)

print(f"Similarity between 'cat' and 'dog': {cat_dog_similarity:.4f}")
```

Slide 2: Word2Vec: A Popular Vector Embedding Technique

Word2Vec is a widely used method for creating word embeddings. It learns vector representations of words by predicting surrounding words in a context window. There are two main architectures: Continuous Bag of Words (CBOW) and Skip-gram.

```python
from gensim.models import Word2Vec

# Sample corpus
corpus = [
    ["I", "love", "machine", "learning"],
    ["Vector", "embeddings", "are", "useful"],
    ["Natural", "language", "processing", "is", "fascinating"]
]

# Train Word2Vec model
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

# Get vector for a word
vector = model.wv["learning"]
print(f"Vector for 'learning': {vector[:5]}...")  # Showing first 5 dimensions

# Find similar words
similar_words = model.wv.most_similar("machine", topn=3)
print("Words similar to 'machine':", similar_words)
```

Slide 3: TF-IDF Vectorization

Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic that reflects the importance of a word in a document within a collection. It's commonly used for text classification and information retrieval.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "The cat sat on the mat",
    "The dog chased the cat",
    "The bird flew over the mat"
]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform documents to TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Print TF-IDF scores for the first document
first_document_vector = tfidf_matrix[0]
for idx, score in zip(first_document_vector.indices, first_document_vector.data):
    print(f"{feature_names[idx]}: {score:.4f}")
```

Slide 4: Doc2Vec: Document Embedding

Doc2Vec extends the Word2Vec model to learn vector representations for entire documents. It's useful for tasks like document classification and clustering.

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "The cat sat on the mat",
    "The bird flew over the rainbow"
]

# Prepare tagged documents
tagged_docs = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(documents)]

# Train Doc2Vec model
model = Doc2Vec(vector_size=50, min_count=1, epochs=40)
model.build_vocab(tagged_docs)
model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

# Infer vector for a new document
new_doc = "The quick brown fox"
inferred_vector = model.infer_vector(new_doc.split())
print(f"Inferred vector for '{new_doc}': {inferred_vector[:5]}...")  # Showing first 5 dimensions

# Find similar documents
similar_docs = model.dv.most_similar([inferred_vector])
print("Similar documents:", similar_docs)
```

Slide 5: BERT Embeddings

BERT (Bidirectional Encoder Representations from Transformers) provides contextualized word embeddings, capturing the context-dependent meaning of words in sentences.

```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Prepare input
text = "Vector embeddings are powerful for NLP tasks."
inputs = tokenizer(text, return_tensors="pt")

# Generate BERT embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Get the embeddings of the last layer
last_hidden_states = outputs.last_hidden_state
print(f"Shape of BERT embeddings: {last_hidden_states.shape}")
print(f"Embedding for the first token: {last_hidden_states[0][0][:5]}...")  # Showing first 5 dimensions
```

Slide 6: Visualization of Vector Embeddings

Visualizing high-dimensional embeddings can help understand the relationships between objects. t-SNE (t-Distributed Stochastic Neighbor Embedding) is a popular technique for this purpose.

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Generate sample word embeddings
words = ["cat", "dog", "fish", "bird", "lion", "tiger", "shark", "whale"]
embeddings = np.random.rand(len(words), 50)  # 50-dimensional embeddings

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings)

# Plot the reduced embeddings
plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    x, y = reduced_embeddings[i]
    plt.scatter(x, y)
    plt.annotate(word, (x, y))

plt.title("t-SNE visualization of word embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
```

Slide 7: Sentence Embeddings with Transformers

Sentence embeddings represent entire sentences as vectors, capturing their semantic meaning. We can use pre-trained models like Sentence-BERT for this purpose.

```python
from sentence_transformers import SentenceTransformer

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Sample sentences
sentences = [
    "The cat sat on the mat",
    "A feline rested on the rug",
    "The dog chased the ball"
]

# Generate sentence embeddings
embeddings = model.encode(sentences)

print(f"Shape of sentence embeddings: {embeddings.shape}")
print(f"Embedding for the first sentence: {embeddings[0][:5]}...")  # Showing first 5 dimensions

# Calculate cosine similarity between sentences
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(embeddings)
print("Similarity matrix:")
print(similarity_matrix)
```

Slide 8: Custom Embedding Layer in Neural Networks

In deep learning models, we often use custom embedding layers to learn task-specific embeddings for discrete objects like words or categories.

```python
import torch
import torch.nn as nn

class SimpleEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SimpleEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, 1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        return self.linear(embedded)

# Example usage
vocab_size = 1000
embedding_dim = 50
model = SimpleEmbeddingModel(vocab_size, embedding_dim)

# Sample input (batch of word indices)
input_tensor = torch.LongTensor([[1, 4, 2], [5, 3, 7]])

# Forward pass
output = model(input_tensor)
print(f"Model output shape: {output.shape}")

# Access learned embeddings
learned_embeddings = model.embedding.weight.data
print(f"Learned embedding for word 0: {learned_embeddings[0][:5]}...")  # Showing first 5 dimensions
```

Slide 9: Word Analogies with Vector Embeddings

Vector embeddings can capture semantic relationships, allowing us to perform word analogies. For example, "king" - "man" + "woman" ≈ "queen".

```python
import gensim.downloader as api

# Load pre-trained Word2Vec embeddings
word2vec_model = api.load('word2vec-google-news-300')

# Perform word analogy
result = word2vec_model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(f"king - man + woman ≈ {result[0][0]} (similarity: {result[0][1]:.4f})")

# Find words similar to "computer"
similar_words = word2vec_model.most_similar("computer", topn=5)
print("Words similar to 'computer':")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")
```

Slide 10: Image Embeddings with Convolutional Neural Networks

Convolutional Neural Networks (CNNs) can be used to generate embeddings for images, capturing their visual features in a compact vector representation.

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load pre-trained ResNet model
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Remove the last fully connected layer
embedding_model = torch.nn.Sequential(*list(resnet.children())[:-1])

# Prepare image transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess an image
image = Image.open("sample_image.jpg")
input_tensor = transform(image).unsqueeze(0)

# Generate image embedding
with torch.no_grad():
    embedding = embedding_model(input_tensor)

print(f"Image embedding shape: {embedding.shape}")
print(f"Image embedding (first 5 values): {embedding.squeeze()[:5]}")
```

Slide 11: Real-life Example: Sentiment Analysis with Vector Embeddings

Vector embeddings are crucial for many natural language processing tasks, including sentiment analysis. Let's use pre-trained GloVe embeddings for a simple sentiment classifier.

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
texts = ["I love this product", "This is terrible", "Not bad, but could be better"]
labels = [1, 0, 0.5]  # 1: positive, 0: negative, 0.5: neutral

# Tokenize texts
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=20)

# Load pre-trained GloVe embeddings
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create embedding matrix
embedding_matrix = np.zeros((10000, 100))
for word, i in tokenizer.word_index.items():
    if i < 10000:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Build model
model = Sequential([
    Embedding(10000, 100, weights=[embedding_matrix], input_length=20, trainable=False),
    GlobalAveragePooling1D(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(padded_sequences, np.array(labels), epochs=10, verbose=0)

# Make predictions
new_texts = ["This is amazing", "I hate this"]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=20)
predictions = model.predict(new_padded)

for text, pred in zip(new_texts, predictions):
    print(f"Text: '{text}', Sentiment: {pred[0]:.2f}")
```

Slide 12: Real-life Example: Image Search with Vector Embeddings

Vector embeddings can be used to build efficient image search systems. Here's a simple example using pre-trained ResNet features.

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained ResNet model
resnet = models.resnet50(pretrained=True)
embedding_model = torch.nn.Sequential(*list(resnet.children())[:-1])
embedding_model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to get image embedding
def get_image_embedding(image_path):
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = embedding_model(input_tensor)
    return embedding.squeeze().numpy()

# Create a database of image embeddings
image_database = {}
for image_file in os.listdir("image_database"):
    if image_file.endswith((".jpg", ".png")):
        image_path = os.path.join("image_database", image_file)
        image_database[image_file] = get_image_embedding(image_path)

# Search for similar images
query_image = "query_image.jpg"
query_embedding = get_image_embedding(query_image)

similarities = {}
for image_name, embedding in image_database.items():
    similarity = cosine_similarity(query_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
    similarities[image_name] = similarity

# Get top 5 similar images
top_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]

print("Top 5 similar images:")
for image_name, similarity in top_similar:
    print(f"{image_name}: Similarity = {similarity:.4f}")
```

Slide 13: Challenges and Future Directions in Vector Embedding

Vector embedding techniques continue to evolve, addressing challenges such as:

1. Contextual embeddings: Capturing word meaning in different contexts.
2. Multimodal embeddings: Representing different types of data (text, images, audio) in a shared embedding space.
3. Efficient storage and retrieval: Handling large-scale embedding databases.
4. Interpretability: Understanding the semantic meaning of embedding dimensions.
5. Bias mitigation: Reducing unwanted biases in learned embeddings.

Future directions include developing more sophisticated models for context-aware embeddings and exploring unsupervised learning techniques for domain-specific embeddings.

Slide 14: Challenges and Future Directions in Vector Embedding

```python
import numpy as np

# Simulating a contextual embedding model
def contextual_embedding(word, context):
    # This is a simplified representation of how contextual embeddings might work
    base_embedding = np.random.rand(100)  # Base 100-dimensional embedding
    context_influence = sum(hash(c) for c in context) % 100  # Context-based modification
    return base_embedding + np.roll(base_embedding, context_influence) * 0.1

# Example usage
word = "bank"
context1 = "I went to the bank to deposit money"
context2 = "The river bank was covered in flowers"

embedding1 = contextual_embedding(word, context1)
embedding2 = contextual_embedding(word, context2)

print(f"Cosine similarity: {np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)):.4f}")
```

Slide 15: Ethical Considerations in Vector Embeddings

As vector embeddings become more prevalent in AI systems, it's crucial to consider their ethical implications:

1. Privacy: Embeddings can potentially encode sensitive information about individuals.
2. Fairness: Embeddings may perpetuate or amplify societal biases present in training data.
3. Transparency: The black-box nature of embeddings can make it difficult to explain model decisions.
4. Misuse: Embeddings could be used for surveillance or other potentially harmful applications.

Researchers and practitioners must address these concerns to ensure responsible development and deployment of embedding-based systems.

Slide 16: Ethical Considerations in Vector Embeddings

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Simulating word embeddings
embeddings = {
    "man": np.array([0.2, 0.3, 0.1]),
    "woman": np.array([0.2, 0.3, -0.1]),
    "doctor": np.array([0.4, 0.2, 0.1]),
    "nurse": np.array([0.4, 0.2, -0.1])
}

# Check for gender bias
def check_occupation_bias(occupation1, occupation2):
    man_occ1 = cosine_similarity([embeddings["man"] + embeddings[occupation1]], [embeddings[occupation1]])[0][0]
    woman_occ1 = cosine_similarity([embeddings["woman"] + embeddings[occupation1]], [embeddings[occupation1]])[0][0]
    man_occ2 = cosine_similarity([embeddings["man"] + embeddings[occupation2]], [embeddings[occupation2]])[0][0]
    woman_occ2 = cosine_similarity([embeddings["woman"] + embeddings[occupation2]], [embeddings[occupation2]])[0][0]
    
    print(f"{occupation1.capitalize()} bias: {man_occ1 - woman_occ1:.4f}")
    print(f"{occupation2.capitalize()} bias: {man_occ2 - woman_occ2:.4f}")

check_occupation_bias("doctor", "nurse")
```

Slide 17: Additional Resources

For those interested in diving deeper into vector embeddings, here are some valuable resources:

1. "Efficient Estimation of Word Representations in Vector Space" by Mikolov et al. (2013) ArXiv: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
2. "GloVe: Global Vectors for Word Representation" by Pennington et al. (2014) ArXiv: [https://arxiv.org/abs/1405.4053](https://arxiv.org/abs/1405.4053)
3. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
4. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" by Reimers and Gurevych (2019) ArXiv: [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)

These papers provide in-depth explanations of various embedding techniques and their applications in natural language processing and machine learning.

