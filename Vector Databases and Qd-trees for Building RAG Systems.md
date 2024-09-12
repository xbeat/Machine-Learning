## Vector Databases and Qd-trees for Building RAG Systems
Slide 1: Introduction to Vector Databases

Vector databases are specialized systems designed to store and efficiently query high-dimensional vector data. They are crucial for many machine learning and AI applications, particularly in similarity search and recommendation systems.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Create sample vector data
vector1 = np.array([1, 2, 3, 4, 5])
vector2 = np.array([2, 3, 4, 5, 6])

# Calculate cosine similarity
similarity = cosine_similarity([vector1], [vector2])[0][0]
print(f"Cosine similarity: {similarity:.4f}")
```

Slide 2: Vector Embeddings

Vector embeddings are dense numerical representations of data points in a high-dimensional space. They capture semantic relationships and allow for efficient similarity computations.

```python
from sentence_transformers import SentenceTransformer

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for two sentences
sentence1 = "The cat sat on the mat."
sentence2 = "A feline rested on a rug."

embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

print(f"Embedding shape: {embedding1.shape}")
print(f"First 5 values of embedding1: {embedding1[:5]}")
```

Slide 3: Qd-trees (Quadtrees)

Qd-trees, or Quadtrees, are tree data structures used to partition two-dimensional space recursively. They are particularly useful for spatial indexing and can be extended to higher dimensions.

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Node:
    def __init__(self, x0, y0, x1, y1, points):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.points = points
        self.children = []

    def subdivide(self):
        if len(self.children) != 0:
            return
        
        x_mid = (self.x0 + self.x1) / 2
        y_mid = (self.y0 + self.y1) / 2
        
        self.children = [
            Node(self.x0, self.y0, x_mid, y_mid, []),
            Node(x_mid, self.y0, self.x1, y_mid, []),
            Node(self.x0, y_mid, x_mid, self.y1, []),
            Node(x_mid, y_mid, self.x1, self.y1, [])
        ]

# Example usage
root = Node(0, 0, 100, 100, [Point(25, 75), Point(75, 25)])
root.subdivide()
```

Slide 4: Approximate Nearest Neighbor Search

Approximate Nearest Neighbor (ANN) search is a technique used in vector databases to find the most similar vectors efficiently, trading off some accuracy for speed.

```python
import numpy as np
from annoy import AnnoyIndex

# Create sample vector data
vectors = np.random.rand(1000, 128)

# Build Annoy index
annoy_index = AnnoyIndex(128, 'angular')
for i, v in enumerate(vectors):
    annoy_index.add_item(i, v)

annoy_index.build(10)  # 10 trees for better accuracy

# Query
query_vector = np.random.rand(128)
nearest_neighbors = annoy_index.get_nns_by_vector(query_vector, 5)

print(f"Indices of 5 nearest neighbors: {nearest_neighbors}")
```

Slide 5: HNSW (Hierarchical Navigable Small World)

HNSW is a graph-based algorithm for approximate nearest neighbor search, known for its high performance in both speed and accuracy.

```python
import hnswlib
import numpy as np

# Generate sample data
num_elements = 10000
dim = 128

# Generating sample data
data = np.random.rand(num_elements, dim).astype('float32')

# Declaring index
p = hnswlib.Index(space='l2', dim=dim)

# Initializing index
p.init_index(max_elements=num_elements, ef_construction=200, M=16)

# Adding data points to the index
p.add_items(data)

# Searching
k = 3
query_data = np.random.rand(1, dim).astype('float32')
labels, distances = p.knn_query(query_data, k=k)

print(f"Labels of {k} nearest neighbors: {labels}")
print(f"Distances to {k} nearest neighbors: {distances}")
```

Slide 6: Vector Quantization

Vector quantization is a technique used to compress high-dimensional vectors by mapping them to a finite set of representative vectors, called a codebook.

```python
import numpy as np
from sklearn.cluster import KMeans

# Generate sample data
num_vectors = 1000
dim = 128
data = np.random.rand(num_vectors, dim)

# Perform vector quantization using K-means
num_centroids = 256
kmeans = KMeans(n_clusters=num_centroids, random_state=42)
kmeans.fit(data)

# Encode vectors
encoded_vectors = kmeans.predict(data)

# Decode vectors
decoded_vectors = kmeans.cluster_centers_[encoded_vectors]

# Calculate reconstruction error
mse = np.mean((data - decoded_vectors) ** 2)
print(f"Mean Squared Error: {mse:.6f}")
```

Slide 7: Product Quantization

Product quantization is an extension of vector quantization that divides high-dimensional vectors into subvectors and quantizes each subvector separately, allowing for more efficient storage and similarity search.

```python
import numpy as np
from sklearn.cluster import KMeans

def product_quantize(data, num_subspaces, num_centroids):
    dim = data.shape[1]
    subvector_size = dim // num_subspaces
    
    codebooks = []
    encoded_data = np.zeros((data.shape[0], num_subspaces), dtype=int)
    
    for i in range(num_subspaces):
        start = i * subvector_size
        end = (i + 1) * subvector_size
        subvectors = data[:, start:end]
        
        kmeans = KMeans(n_clusters=num_centroids, random_state=42)
        kmeans.fit(subvectors)
        
        codebooks.append(kmeans.cluster_centers_)
        encoded_data[:, i] = kmeans.predict(subvectors)
    
    return codebooks, encoded_data

# Generate sample data
num_vectors = 1000
dim = 128
data = np.random.rand(num_vectors, dim)

# Apply product quantization
num_subspaces = 8
num_centroids = 256
codebooks, encoded_data = product_quantize(data, num_subspaces, num_centroids)

print(f"Shape of encoded data: {encoded_data.shape}")
print(f"Number of codebooks: {len(codebooks)}")
```

Slide 8: Inverted File Systems

Inverted file systems are data structures used in information retrieval to map terms to their locations in a document or a set of documents, enabling efficient search and retrieval.

```python
from collections import defaultdict

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)
    
    def add_document(self, doc_id, content):
        for position, term in enumerate(content.split()):
            self.index[term].append((doc_id, position))
    
    def search(self, query):
        return self.index[query]

# Create and populate the inverted index
inverted_index = InvertedIndex()
inverted_index.add_document(1, "The quick brown fox")
inverted_index.add_document(2, "jumps over the lazy dog")

# Search for a term
results = inverted_index.search("the")
print(f"Occurrences of 'the': {results}")
```

Slide 9: RAG Systems Overview

Retrieval-Augmented Generation (RAG) systems combine the power of large language models with external knowledge retrieval to generate more accurate and contextually relevant responses.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np

class SimpleRAG:
    def __init__(self, model_name, vector_dim):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.index = faiss.IndexFlatL2(vector_dim)
        self.documents = []

    def add_document(self, doc):
        vec = self.get_vector(doc)
        self.index.add(np.array([vec]))
        self.documents.append(doc)

    def get_vector(self, text):
        # Simplified: In practice, use a proper embedding model
        return np.random.rand(self.index.d)

    def retrieve(self, query, k=1):
        query_vec = self.get_vector(query)
        _, I = self.index.search(np.array([query_vec]), k)
        return [self.documents[i] for i in I[0]]

    def generate(self, query):
        docs = self.retrieve(query)
        context = " ".join(docs)
        input_text = f"Context: {context}\nQuery: {query}\nAnswer:"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=100)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
rag = SimpleRAG("gpt2", 128)
rag.add_document("The capital of France is Paris.")
rag.add_document("The Eiffel Tower is located in Paris.")

response = rag.generate("What is the capital of France?")
print(response)
```

Slide 10: Vector Similarity Measures

Vector similarity measures are crucial in vector databases for comparing and ranking vectors. Common measures include cosine similarity, Euclidean distance, and dot product.

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def dot_product(a, b):
    return np.dot(a, b)

# Example vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"Cosine similarity: {cosine_similarity(v1, v2):.4f}")
print(f"Euclidean distance: {euclidean_distance(v1, v2):.4f}")
print(f"Dot product: {dot_product(v1, v2):.4f}")
```

Slide 11: Real-Life Example: Image Similarity Search

In this example, we'll use a pre-trained ResNet model to generate image embeddings and perform similarity search using FAISS.

```python
import torch
from torchvision import models, transforms
from PIL import Image
import faiss
import numpy as np

# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove last FC layer
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().numpy()
    return embedding

# Create FAISS index
dim = 512  # ResNet18 output dimension
index = faiss.IndexFlatL2(dim)

# Add images to the index (simplified example)
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
for path in image_paths:
    embedding = get_image_embedding(path)
    index.add(np.array([embedding]))

# Perform similarity search
query_embedding = get_image_embedding('query_image.jpg')
k = 2  # Number of similar images to retrieve
D, I = index.search(np.array([query_embedding]), k)

print(f"Indices of {k} most similar images: {I[0]}")
print(f"Distances to {k} most similar images: {D[0]}")
```

Slide 12: Real-Life Example: Text-Based Question Answering

This example demonstrates a simple question-answering system using sentence embeddings and cosine similarity for retrieval.

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample knowledge base
knowledge_base = [
    "The Earth is the third planet from the Sun.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Python is a high-level programming language.",
    "The capital of Japan is Tokyo.",
    "Photosynthesis is the process by which plants convert light into energy."
]

# Encode knowledge base
kb_embeddings = model.encode(knowledge_base)

def answer_question(question):
    # Encode the question
    question_embedding = model.encode([question])
    
    # Calculate similarities
    similarities = cosine_similarity(question_embedding, kb_embeddings)[0]
    
    # Find the most similar statement
    most_similar_idx = np.argmax(similarities)
    
    return knowledge_base[most_similar_idx]

# Example usage
question = "What is the boiling point of water?"
answer = answer_question(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
```

Slide 13: Challenges and Future Directions

Vector databases and RAG systems face challenges such as scalability, accuracy-speed trade-offs, and handling out-of-distribution queries. Future research directions include:

1. Improved indexing structures for billion-scale vector datasets
2. Hybrid retrieval methods combining dense and sparse representations
3. Efficient update mechanisms for dynamic vector databases
4. Enhanced vector compression techniques for reduced memory footprint
5. Integration of multi-modal data in RAG systems

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulating the trade-off between accuracy and query time
def accuracy_time_tradeoff(index_size):
    accuracy = 1 - np.exp(-index_size / 1000)
    query_time = np.log(index_size)
    return accuracy, query_time

sizes = np.logspace(2, 6, num=100)
accuracies, times = zip(*[accuracy_time_tradeoff(size) for size in sizes])

plt.figure(figsize=(10, 6))
plt.semilogx(sizes, accuracies, label='Accuracy')
plt.semilogx(sizes, times, label='Query Time')
plt.xlabel('Index Size')
plt.ylabel('Normalized Value')
plt.title('Accuracy vs. Query Time Trade-off')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 14: Additional Resources

For more in-depth information on vector databases, Qd-trees, and RAG systems, consider exploring these resources:

1. "Billion-scale similarity search with GPUs" by Johnson et al. (2017) - ArXiv:1702.08734
2. "HNSW: Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs" by Malkov and Yashunin (2018) - ArXiv:1603.09320
3. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" by Lewis et al. (2020) - ArXiv:2005.11401
4. "Faiss: A Library for Efficient Similarity Search" by Johnson et al. (2019) - ArXiv:1908.03559
5. "Product Quantization for Nearest Neighbor Search" by Jégou et al. (2011) - IEEE Transactions on Pattern Analysis and Machine Intelligence

These papers provide comprehensive insights into the algorithms, implementations, and applications of vector databases and RAG systems. They cover topics from efficient similarity search techniques to the integration of retrieval mechanisms in natural language processing tasks.

