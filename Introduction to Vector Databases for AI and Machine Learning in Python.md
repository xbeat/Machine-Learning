## Introduction to Vector Databases for AI and Machine Learning in Python

Slide 1: Introduction to Vector Databases for AI and Machine Learning

Vector databases are a type of database designed to store and manipulate high-dimensional vectors efficiently. They are particularly useful in AI and Machine Learning applications, where data is often represented as dense numerical vectors. Vector databases enable efficient similarity search, clustering, and other vector operations that are essential for tasks such as recommender systems, natural language processing, and image recognition.

Slide 2: Word2Vec

Word2Vec is a popular technique for learning word embeddings, which represent words as high-dimensional vectors. The vectors capture semantic relationships between words, allowing for efficient computation of word similarities. Word2Vec is often used as a pre-processing step for natural language processing tasks.

```python
import gensim.downloader as api
from gensim.models import Word2Vec

# Load pre-trained Word2Vec model
word2vec_model = api.load('word2vec-google-news-300')

# Get vector for a word
word_vector = word2vec_model.wv['python']

# Find most similar words
similar_words = word2vec_model.wv.most_similar(positive=['python', 'programming'], topn=5)
print(similar_words)
```

Slide 3: GloVe

GloVe (Global Vectors for Word Representation) is another popular technique for learning word embeddings. It is based on the co-occurrence statistics of words in a corpus and is particularly effective at capturing semantic relationships between words.

```python
import gensim.downloader as api
from gensim.models import KeyedVectors

# Load pre-trained GloVe model
glove_model = api.load('glove-wiki-gigaword-100')

# Get vector for a word
word_vector = glove_model.wv['computer']

# Find most similar words
similar_words = glove_model.wv.most_similar(positive=['computer', 'technology'], topn=5)
print(similar_words)
```

Slide 4: BERT

BERT (Bidirectional Encoder Representations from Transformers) is a powerful language model that has revolutionized many natural language processing tasks. It uses a transformer architecture to learn context-aware word representations, which can be used as input for downstream tasks.

```python
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize input text
text = "I love learning about machine learning!"
inputs = tokenizer(text, return_tensors='pt')

# Get BERT embeddings
outputs = model(**inputs)
embeddings = outputs.last_hidden_state
```

Slide 5: Vector Database Setup

To work with vector databases in Python, we need to install a vector database library. One popular choice is Weaviate, an open-source vector database with a user-friendly API.

```python
import weaviate

# Connect to Weaviate instance
client = weaviate.Client("http://localhost:8080")

# Create a new schema
schema = {
    "classes": [
        {
            "class": "Document",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"]
                }
            ]
        }
    ]
}

# Create the schema in Weaviate
client.schema.create(schema)
```

Slide 6: Storing and Retrieving Vectors

Once the vector database is set up, we can store and retrieve vectors efficiently. Here's an example of storing text data as vectors and performing similarity search.

```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Store text data as vectors
texts = ["I love machine learning!", "Python is great for AI.", "Data science is fascinating."]
for text in texts:
    client.data_object.create(
        data_object={
            "class": "Document",
            "properties": {
                "content": text
            }
        },
        vector=client.embedding.generate_vector(text, "text2vec-transformers")
    )

# Perform similarity search
query = "I want to learn about artificial intelligence."
results = client.query.get(
    class_name="Document",
    properties=["content"],
    query=query,
    vector=client.embedding.generate_vector(query, "text2vec-transformers")
).get("data", {}).get("Get", {}).get("Document", [])

for result in results:
    print(result.get("properties", {}).get("content"))
```

Slide 7: Vector Operations

Vector databases also enable efficient vector operations, such as computing vector similarities, clustering, and dimensionality reduction. These operations are essential for many AI and Machine Learning tasks.

```python
import numpy as np
from scipy.spatial.distance import cosine

# Example vectors
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

# Compute cosine similarity
similarity = 1 - cosine(vector1, vector2)
print(f"Similarity: {similarity}")

# Perform k-means clustering
from sklearn.cluster import KMeans

# Example data
data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [9, 3], [8, 1]])

# Fit k-means model
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)

# Get cluster labels
labels = kmeans.labels_
print(f"Cluster labels: {labels}")
```

Slide 8: Dimensionality Reduction

Dimensionality reduction techniques are often used in conjunction with vector databases to reduce the computational complexity and improve the performance of vector operations. Popular techniques include Principal Component Analysis (PCA) and t-SNE.

```python
import numpy as np
from sklearn.decomposition import PCA

# Example high-dimensional data
data = np.random.rand(1000, 100)

# Perform PCA
pca = PCA(n_components=10)
data_reduced = pca.fit_transform(data)

print(f"Original shape: {data.shape}")
print(f"Reduced shape: {data_reduced.shape}")
```

Slide 9: Vector Databases in Recommender Systems

Vector databases are widely used in recommender systems, where items (e.g., movies, products) are represented as vectors, and user preferences are modeled based on vector similarities.

```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Store item vectors
items = [
    {"name": "Movie A", "genre": "Action", "vector": [1, 2, 3]},
    {"name": "Movie B", "genre": "Comedy", "vector": [4, 5, 6]},
    {"name": "Movie C", "genre": "Action", "vector": [7, 8, 9]}
]

for item in items:
    client.data_object.create(
        data_object={
            "class": "Item",
            "properties": {
                "name": item["name"],
                "genre": item["genre"]
            }
        },
        vector=item["vector"]
    )

# Recommend items based on user preferences
user_preferences = [5, 6, 7]
results = client.query.get(
    class_name="Item",
    properties=["name", "genre"],
    vector=user_preferences
).get("data", {}).get("Get", {}).get("Item", [])

for result in results:
    print(f"Recommendation: {result.get('properties', {}).get('name')} ({result.get('properties', {}).get('genre')})")
```

Slide 10: Vector Databases in Natural Language Processing

Vector databases are essential in natural language processing tasks, such as text classification, sentiment analysis, and question answering. Word embeddings and contextualized representations like BERT are often stored and retrieved from vector databases.

```python
import weaviate
from transformers import BertTokenizer, BertModel

client = weaviate.Client("http://localhost:8080")

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize and encode input text
text = "I love learning about machine learning!"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1)

# Store text embeddings in the vector database
client.data_object.create(
    data_object={
        "class": "Document",
        "properties": {
            "content": text
        }
    },
    vector=embeddings.detach().numpy().squeeze()
)

# Perform semantic search
query = "What are some applications of artificial intelligence?"
query_inputs = tokenizer(query, return_tensors='pt')
query_embeddings = model(**query_inputs).last_hidden_state.mean(dim=1)

results = client.query.get(
    class_name="Document",
    properties=["content"],
    vector=query_embeddings.detach().numpy().squeeze()
).get("data", {}).get("Get", {}).get("Document", [])

for result in results:
    print(f"Relevant document: {result.get('properties', {}).get('content')}")
```

Slide 11: Vector Databases in Image Recognition

Vector databases can also be used in image recognition tasks, where images are represented as high-dimensional vectors. These vectors can be generated using techniques like convolutional neural networks (CNNs) or pre-trained models like ResNet or VGGNet.

```python
import weaviate
from PIL import Image
import torchvision.transforms as transforms

client = weaviate.Client("http://localhost:8080")

# Load pre-trained ResNet model
resnet = torchvision.models.resnet18(pretrained=True)

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Store image vectors
images = [
    {"path": "image1.jpg", "label": "cat"},
    {"path": "image2.jpg", "label": "dog"},
    {"path": "image3.jpg", "label": "bird"}
]

for image_data in images:
    img = Image.open(image_data["path"])
    img_tensor = preprocess(img)
    img_vector = resnet(img_tensor.unsqueeze(0)).flatten().detach().numpy()

    client.data_object.create(
        data_object={
            "class": "Image",
            "properties": {
                "label": image_data["label"]
            }
        },
        vector=img_vector
    )

# Perform image similarity search
query_img = Image.open("query_image.jpg")
query_tensor = preprocess(query_img)
query_vector = resnet(query_tensor.unsqueeze(0)).flatten().detach().numpy()

results = client.query.get(
    class_name="Image",
    properties=["label"],
    vector=query_vector
).get("data", {}).get("Get", {}).get("Image", [])

for result in results:
    print(f"Similar image: {result.get('properties', {}).get('label')}")
```

Slide 12: Vector Databases in Genomics

Vector databases are also finding applications in genomics and bioinformatics, where genetic sequences can be represented as high-dimensional vectors. This allows for efficient similarity search and pattern discovery in large genomic datasets.

```python
import weaviate
import numpy as np

client = weaviate.Client("http://localhost:8080")

# Example genetic sequence vectors
sequences = [
    {"id": "seq1", "vector": np.random.rand(100)},
    {"id": "seq2", "vector": np.random.rand(100)},
    {"id": "seq3", "vector": np.random.rand(100)}
]

for seq in sequences:
    client.data_object.create(
        data_object={
            "class": "Sequence",
            "properties": {
                "id": seq["id"]
            }
        },
        vector=seq["vector"]
    )

# Perform similarity search
query_vector = np.random.rand(100)
results = client.query.get(
    class_name="Sequence",
    properties=["id"],
    vector=query_vector
).get("data", {}).get("Get", {}).get("Sequence", [])

for result in results:
    print(f"Similar sequence: {result.get('properties', {}).get('id')}")
```

Slide 13: Hybrid Vector-Scalar Databases

Some applications may require combining vector data with traditional scalar data (e.g., structured tabular data). In such cases, hybrid vector-scalar databases can be used, which allow for efficient storage and querying of both vector and scalar data.

```python
import weaviate
import pandas as pd

client = weaviate.Client("http://localhost:8080")

# Define schema for hybrid data
schema = {
    "classes": [
        {
            "class": "Customer",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {
                    "name": "name",
                    "dataType": ["text"]
                },
                {
                    "name": "age",
                    "dataType": ["int"]
                },
                {
                    "name": "description",
                    "dataType": ["text"]
                }
            ]
        }
    ]
}

client.schema.create(schema)

# Store hybrid data
data = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "description": ["I love reading books.", "I enjoy hiking in nature.", "I'm a passionate programmer."]
})

for _, row in data.iterrows():
    client.data_object.create(
        data_object={
            "class": "Customer",
            "properties": {
                "name": row["name"],
                "age": row["age"],
                "description": row["description"]
            }
        },
        vector=client.embedding.generate_vector(row["description"], "text2vec-transformers")
    )

# Query hybrid data
query = "I'm looking for a customer who enjoys outdoor activities and is in their 30s."
query_vector = client.embedding.generate_vector(query, "text2vec-transformers")
results = client.query.get(
    class_name="Customer",
    properties=["name", "age", "description"],
    vector=query_vector,
    additional_properties=["age >= 30", "age < 40"]
).get("data", {}).get("Get", {}).get("Customer", [])

for result in results:
    print(f"Name: {result.get('properties', {}).get('name')}")
    print(f"Age: {result.get('properties', {}).get('age')}")
    print(f"{result.get('properties', {}).get('description')}")
    print("---")
```

Slide 14 (Additional Resources): Additional Resources

Here are some additional resources for further learning about vector databases and their applications in AI and Machine Learning:

* ArXiv Paper: "Efficient Vector Database for Large-Scale AI Applications" ([https://arxiv.org/abs/2101.12345](https://arxiv.org/abs/2101.12345))
* ArXiv Paper: "Vector Databases for Scalable Similarity Search in High-Dimensional Data" ([https://arxiv.org/abs/2202.67890](https://arxiv.org/abs/2202.67890))
* Book: "Vector Databases for AI and Machine Learning" by John Doe (Publisher, Year)
* Online Course: "Introduction to Vector Databases" by Jane Smith (Coursera/Udemy/edX)

Note: The specific resources listed here are fictional examples. However, ArXiv ([https://arxiv.org/](https://arxiv.org/)) is a reputable repository for scientific papers, and many universities and online platforms offer courses related to vector databases and their applications.

In this example, we use the pre-trained BERT model to generate contextualized embeddings for text data. These embeddings are then stored in a vector database (Weaviate). When we need to perform semantic search, we generate the query embeddings using the same BERT model and use them to retrieve the most relevant documents from the vector database based on vector similarity.

