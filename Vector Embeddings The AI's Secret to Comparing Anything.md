## Vector Embeddings The AI's Secret to Comparing Anything

Slide 1: Vector Embeddings: The Magic Behind AI Comparisons

Vector embeddings are numerical representations of real-world objects or concepts. They allow AI systems to compare and analyze diverse items, from fruits to documents to songs, by converting them into a format that machines can easily process and understand.

```python

# Example vector embeddings
strawberry = np.array([4, 0, 1])
blueberry = np.array([3, 0, 1])

print(f"Strawberry vector: {strawberry}")
print(f"Blueberry vector: {blueberry}")
```

Slide 2: Understanding Vector Representations

Vector embeddings capture essential features of an object in a list of numbers. Each dimension in the vector represents a specific attribute or characteristic. For instance, a fruit's vector might include dimensions for sweetness, acidity, and size.

```python
def create_fruit_vector(sweetness, acidity, size):
    return np.array([sweetness, acidity, size])

apple = create_fruit_vector(7, 5, 6)
lemon = create_fruit_vector(2, 9, 4)

print(f"Apple vector: {apple}")
print(f"Lemon vector: {lemon}")
```

Slide 3: The Power of Vector Comparisons

Vector embeddings enable AI to quantify similarities and differences between objects. By comparing vectors, we can determine how closely related items are, which is crucial for tasks like recommendation systems, image classification, and natural language processing.

```python
    similarity = np.dot(fruit1, fruit2) / (np.linalg.norm(fruit1) * np.linalg.norm(fruit2))
    return similarity

apple = create_fruit_vector(7, 5, 6)
pear = create_fruit_vector(6, 4, 5)

similarity = compare_fruits(apple, pear)
print(f"Similarity between apple and pear: {similarity:.4f}")
```

Slide 4: Cosine Similarity: Measuring Vector Angles

Cosine similarity is a popular metric for comparing vectors. It measures the cosine of the angle between two vectors, providing a value between -1 and 1. A value closer to 1 indicates higher similarity, while values closer to -1 indicate dissimilarity.

```python
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

strawberry = np.array([4, 0, 1])
blueberry = np.array([3, 0, 1])

similarity = cosine_similarity(strawberry, blueberry)
print(f"Cosine similarity between strawberry and blueberry: {similarity:.4f}")
```

Slide 5: Dot Product Similarity: Direction Matters

The dot product is another way to measure vector similarity. It multiplies corresponding elements of two vectors and sums the results. A higher dot product indicates that vectors are pointing in similar directions, suggesting greater similarity.

```python
    return np.dot(v1, v2)

strawberry = np.array([4, 0, 1])
blueberry = np.array([3, 0, 1])

similarity = dot_product_similarity(strawberry, blueberry)
print(f"Dot product similarity between strawberry and blueberry: {similarity}")
```

Slide 6: Euclidean Distance: Measuring Vector Separation

Euclidean distance calculates the straight-line distance between two vectors in space. It's useful when the magnitude of vectors is important, such as comparing word counts in documents or physical measurements of objects.

```python
    return np.linalg.norm(v1 - v2)

strawberry = np.array([4, 0, 1])
blueberry = np.array([3, 0, 1])

distance = euclidean_distance(strawberry, blueberry)
print(f"Euclidean distance between strawberry and blueberry: {distance}")
```

Slide 7: Choosing the Right Similarity Metric

The choice of similarity metric depends on the specific task and the nature of the data. Cosine similarity is often used for text data, where the direction of vectors is more important than their magnitude. Euclidean distance is preferred when the absolute values of vector components matter.

```python
    cosine = cosine_similarity(v1, v2)
    dot_product = dot_product_similarity(v1, v2)
    euclidean = euclidean_distance(v1, v2)
    
    print(f"Cosine Similarity: {cosine:.4f}")
    print(f"Dot Product Similarity: {dot_product}")
    print(f"Euclidean Distance: {euclidean:.4f}")

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

compare_metrics(v1, v2)
```

Slide 8: Real-Life Example: Document Similarity

Vector embeddings are widely used in natural language processing to compare document similarity. By representing documents as vectors of word frequencies, we can easily find related content or identify plagiarism.

```python
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "The quick brown fox jumps over the lazy dog",
    "A fast auburn fox leaps above a sluggish canine",
    "Python is a versatile programming language"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

doc_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print(f"Similarity between doc 1 and doc 2: {doc_similarity[0][0]:.4f}")
```

Slide 9: Real-Life Example: Image Classification

In computer vision, vector embeddings are used to represent images. Convolutional Neural Networks (CNNs) can extract features from images and create embeddings, which can then be compared to classify new images or find similar ones.

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

def get_image_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    embedding = model.predict(x)
    return embedding.flatten()

# Example usage (assuming you have image files)
# embedding1 = get_image_embedding('cat.jpg')
# embedding2 = get_image_embedding('dog.jpg')
# similarity = cosine_similarity([embedding1], [embedding2])
# print(f"Image similarity: {similarity[0][0]:.4f}")
```

Slide 10: Dimensionality Reduction: Visualizing High-Dimensional Embeddings

Often, vector embeddings have many dimensions, making them difficult to visualize. Techniques like t-SNE (t-Distributed Stochastic Neighbor Embedding) can reduce the dimensionality of embeddings while preserving their relative relationships, allowing us to visualize complex data in 2D or 3D space.

```python
import matplotlib.pyplot as plt

# Assuming we have high-dimensional embeddings
high_dim_embeddings = np.random.rand(100, 50)  # 100 samples, 50 dimensions

tsne = TSNE(n_components=2, random_state=42)
low_dim_embeddings = tsne.fit_transform(high_dim_embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(low_dim_embeddings[:, 0], low_dim_embeddings[:, 1])
plt.title("t-SNE visualization of high-dimensional embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
```

Slide 11: Handling Categorical Data: One-Hot Encoding

Not all data is numerical. Categorical data, such as colors or types, can be converted into vector embeddings using techniques like one-hot encoding. This allows categorical features to be included in vector comparisons and machine learning models.

```python

data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['small', 'medium', 'large', 'medium', 'small']
})

one_hot_encoded = pd.get_dummies(data)
print(one_hot_encoded)
```

Slide 12: Word Embeddings: Capturing Semantic Relationships

Word embeddings are a special type of vector embedding used in natural language processing. They capture semantic relationships between words, allowing AI models to understand context and meaning in text data.

```python

# Sample sentences
sentences = [['cat', 'sat', 'on', 'mat'],
             ['dog', 'barked', 'at', 'mailman'],
             ['bird', 'flew', 'over', 'tree']]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Find similar words
similar_words = model.wv.most_similar('cat', topn=3)
print("Words similar to 'cat':")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")
```

Slide 13: Challenges and Limitations of Vector Embeddings

While vector embeddings are powerful, they have limitations. They can struggle with capturing complex, non-linear relationships and may require large amounts of data to create meaningful representations. Additionally, biases in training data can lead to biased embeddings, potentially perpetuating unfair stereotypes in AI systems.

```python

# Simplified example of bias in word embeddings
word_embeddings = {
    'man': np.array([0.2, 0.3, 0.1]),
    'woman': np.array([0.2, 0.3, -0.1]),
    'king': np.array([0.5, 0.6, 0.1]),
    'queen': np.array([0.5, 0.6, -0.1])
}

# Demonstrating gender bias
king_man = word_embeddings['king'] - word_embeddings['man']
queen_woman = word_embeddings['queen'] - word_embeddings['woman']

print("Difference between 'king - man' and 'queen - woman':")
print(np.allclose(king_man, queen_woman, atol=1e-6))
```

Slide 14: Future Directions: Contextual and Dynamic Embeddings

Research in vector embeddings is moving towards more context-aware and dynamic representations. Models like BERT and GPT use attention mechanisms to create embeddings that change based on the surrounding context, leading to more nuanced and accurate representations of language and concepts.

```python
import torch

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example sentences
sentences = ["The bank is by the river.", "I need to bank my check."]

# Get contextual embeddings
for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    print(f"Shape of embeddings for '{sentence}': {embeddings.shape}")
```

Slide 15: Additional Resources

For those interested in diving deeper into vector embeddings and their applications in AI, here are some valuable resources:

1. "Efficient Estimation of Word Representations in Vector Space" by Mikolov et al. (2013) ArXiv: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
2. "GloVe: Global Vectors for Word Representation" by Pennington et al. (2014) ArXiv: [https://arxiv.org/abs/1405.4053](https://arxiv.org/abs/1405.4053)
3. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

These papers provide foundational knowledge and advanced techniques in the field of vector embeddings and their applications in natural language processing and machine learning.


