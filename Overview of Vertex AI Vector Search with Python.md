## Overview of Vertex AI Vector Search with Python
Slide 1: Introduction to Vertex AI Vector Search

Vertex AI Vector Search is a powerful tool for similarity search and recommendation systems. It allows developers to efficiently search through large datasets of high-dimensional vectors, making it ideal for applications in natural language processing, computer vision, and recommendation engines.

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import VectorSearchIndex

# Initialize the Vertex AI client
aiplatform.init(project="your-project-id")

# Create a Vector Search Index
index = VectorSearchIndex.create(
    display_name="my-vector-index",
    description="An example vector search index",
    dimension=128,  # Dimension of your vectors
    metric="cosine_distance"  # Similarity metric
)
```

Slide 2: Data Preparation for Vector Search

Before using Vertex AI Vector Search, you need to prepare your data by converting it into numerical vectors. This process, known as embedding, transforms raw data into dense vector representations that capture semantic meaning.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example text data
texts = ["Hello, world!", "Vector search is amazing", "AI is transforming industries"]

# Generate embeddings
embeddings = model.encode(texts)

print(f"Shape of embeddings: {embeddings.shape}")
print(f"First embedding: {embeddings[0]}")
```

Slide 3: Indexing Vectors in Vertex AI

Once you have your vector embeddings, you can index them in Vertex AI Vector Search. This process organizes the vectors for efficient similarity search later on.

```python
# Assuming 'index' is your Vector Search Index from Slide 1
# and 'embeddings' are your vector embeddings from Slide 2

# Generate unique IDs for each vector
ids = [f"id_{i}" for i in range(len(embeddings))]

# Index the vectors
index.upsert_vectors(
    vectors=embeddings.tolist(),
    ids=ids
)

print(f"Indexed {len(embeddings)} vectors")
```

Slide 4: Performing Vector Similarity Search

With indexed vectors, you can now perform similarity searches. This allows you to find the most similar vectors to a given query vector, which is useful for recommendation systems and information retrieval.

```python
# Assuming 'index' is your Vector Search Index
# Create a query vector (for example, using the same model as before)
query_text = "What is machine learning?"
query_vector = model.encode([query_text])[0]

# Perform a similarity search
results = index.search(
    query_vector=query_vector.tolist(),
    num_neighbors=5  # Number of similar vectors to return
)

for result in results:
    print(f"ID: {result.id}, Distance: {result.distance}")
```

Slide 5: Updating and Deleting Vectors

Vertex AI Vector Search allows you to update or delete vectors in your index, keeping your search results up-to-date with your latest data.

```python
# Update a vector
updated_vector = np.random.rand(128).tolist()  # Generate a new random vector
index.upsert_vectors(vectors=[updated_vector], ids=["id_0"])

# Delete a vector
index.delete_vectors(ids=["id_1"])

print("Vector updated and deleted successfully")
```

Slide 6: Filtering Search Results

You can apply filters to your vector search to narrow down results based on metadata associated with your vectors.

```python
# Assuming you've added metadata to your vectors during indexing
metadata = [
    {"category": "tech", "popularity": 0.8},
    {"category": "science", "popularity": 0.6},
    {"category": "tech", "popularity": 0.9}
]

# Index vectors with metadata
index.upsert_vectors(
    vectors=embeddings.tolist(),
    ids=ids,
    metadata=metadata
)

# Perform a filtered search
filtered_results = index.search(
    query_vector=query_vector.tolist(),
    num_neighbors=5,
    filter="category = 'tech' AND popularity > 0.7"
)

for result in filtered_results:
    print(f"ID: {result.id}, Distance: {result.distance}")
```

Slide 7: Batch Vector Operations

For large-scale applications, Vertex AI Vector Search supports batch operations, allowing you to index or search multiple vectors efficiently.

```python
# Batch indexing
batch_vectors = np.random.rand(1000, 128).tolist()
batch_ids = [f"batch_id_{i}" for i in range(1000)]

index.upsert_vectors(vectors=batch_vectors, ids=batch_ids)

# Batch searching
query_vectors = np.random.rand(10, 128).tolist()
batch_results = index.batch_search(
    query_vectors=query_vectors,
    num_neighbors=5
)

for i, results in enumerate(batch_results):
    print(f"Results for query {i}:")
    for result in results:
        print(f"  ID: {result.id}, Distance: {result.distance}")
```

Slide 8: Monitoring and Metrics

Vertex AI provides monitoring capabilities to track the performance and usage of your Vector Search index.

```python
from google.cloud import monitoring_v3

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/your-project-id"

# Define the metric
metric_type = "custom.googleapis.com/vertex_ai/vector_search/query_latency"

# Create a time series request
interval = monitoring_v3.TimeInterval()
now = time.time()
interval.end_time.seconds = int(now)
interval.end_time.nanos = int((now - interval.end_time.seconds) * 10**9)
interval.start_time.seconds = int(now - 3600)  # Last hour

request = monitoring_v3.ListTimeSeriesRequest(
    name=project_name,
    filter=f'metric.type = "{metric_type}"',
    interval=interval,
    view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
)

# List the time series
results = client.list_time_series(request=request)

for result in results:
    print(f"Metric: {result.metric}")
    for point in result.points:
        print(f"  Value: {point.value.double_value}, Time: {point.interval.end_time}")
```

Slide 9: Error Handling and Retries

Robust error handling and retry mechanisms are crucial when working with cloud services like Vertex AI Vector Search.

```python
from google.api_core import retry
from google.api_core import exceptions

@retry.Retry(predicate=retry.if_exception_type(
    exceptions.DeadlineExceeded,
    exceptions.ServiceUnavailable
))
def search_with_retry(index, query_vector, num_neighbors):
    try:
        results = index.search(
            query_vector=query_vector,
            num_neighbors=num_neighbors
        )
        return results
    except exceptions.GoogleAPICallError as e:
        print(f"API call failed: {e}")
        raise

# Usage
try:
    results = search_with_retry(index, query_vector.tolist(), 5)
    for result in results:
        print(f"ID: {result.id}, Distance: {result.distance}")
except Exception as e:
    print(f"Search failed after multiple retries: {e}")
```

Slide 10: Real-Life Example: Content Recommendation System

Vertex AI Vector Search can be used to build a content recommendation system for a news website, suggesting articles similar to what a user is currently reading.

```python
# Assume we have a function to get article content and generate embeddings
def get_article_embedding(article_id):
    # Fetch article content (placeholder)
    content = f"Content of article {article_id}"
    # Generate embedding (using the model from Slide 2)
    return model.encode([content])[0]

# Index all articles
all_article_ids = range(1000)  # Assume 1000 articles
for article_id in all_article_ids:
    embedding = get_article_embedding(article_id)
    index.upsert_vectors(
        vectors=[embedding.tolist()],
        ids=[f"article_{article_id}"]
    )

# Recommend similar articles
def recommend_articles(current_article_id, num_recommendations=5):
    query_vector = get_article_embedding(current_article_id)
    results = index.search(
        query_vector=query_vector.tolist(),
        num_neighbors=num_recommendations + 1  # +1 to exclude the current article
    )
    # Filter out the current article and return recommendations
    return [result.id for result in results if result.id != f"article_{current_article_id}"]

# Usage
current_article = 42
recommendations = recommend_articles(current_article)
print(f"Recommended articles for article {current_article}: {recommendations}")
```

Slide 11: Real-Life Example: Image Similarity Search

Vertex AI Vector Search can be used for image similarity search, helping users find visually similar images in a large dataset.

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load a pre-trained ResNet model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the last FC layer
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    
    with torch.no_grad():
        embedding = model(batch_t)
    
    return embedding.numpy().flatten()

# Index images (assuming we have a list of image paths)
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
for i, path in enumerate(image_paths):
    embedding = get_image_embedding(path)
    index.upsert_vectors(
        vectors=[embedding.tolist()],
        ids=[f"image_{i}"]
    )

# Find similar images
def find_similar_images(query_image_path, num_results=5):
    query_embedding = get_image_embedding(query_image_path)
    results = index.search(
        query_vector=query_embedding.tolist(),
        num_neighbors=num_results
    )
    return [result.id for result in results]

# Usage
similar_images = find_similar_images("query_image.jpg")
print(f"Similar images: {similar_images}")
```

Slide 12: Performance Optimization

Optimizing the performance of your Vertex AI Vector Search implementation is crucial for large-scale applications.

```python
import concurrent.futures

def batch_index_vectors(vectors, batch_size=100):
    total_vectors = len(vectors)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, total_vectors, batch_size):
            batch = vectors[i:i+batch_size]
            batch_ids = [f"id_{j}" for j in range(i, i+len(batch))]
            future = executor.submit(index.upsert_vectors, vectors=batch, ids=batch_ids)
            futures.append(future)
        
        # Wait for all futures to complete
        concurrent.futures.wait(futures)

# Generate a large number of random vectors
large_vector_set = np.random.rand(10000, 128).tolist()

# Batch index the vectors
batch_index_vectors(large_vector_set)

print("Batch indexing completed")

# Optimize search with pre-filtering
def optimized_search(query_vector, filter_condition, num_neighbors=5):
    results = index.search(
        query_vector=query_vector,
        num_neighbors=num_neighbors,
        filter=filter_condition
    )
    return results

# Usage
query = np.random.rand(128).tolist()
filter_condition = "metadata.category = 'tech' AND metadata.date > '2023-01-01'"
optimized_results = optimized_search(query, filter_condition)

for result in optimized_results:
    print(f"ID: {result.id}, Distance: {result.distance}")
```

Slide 13: Best Practices and Limitations

When working with Vertex AI Vector Search, it's important to be aware of best practices and limitations to ensure optimal performance and reliability.

```python
# Best Practice: Use appropriate vector dimensions
RECOMMENDED_MIN_DIMENSION = 2
RECOMMENDED_MAX_DIMENSION = 1024

def validate_vector_dimension(vector):
    dimension = len(vector)
    if RECOMMENDED_MIN_DIMENSION <= dimension <= RECOMMENDED_MAX_DIMENSION:
        return True
    else:
        print(f"Warning: Vector dimension {dimension} is outside the recommended range.")
        return False

# Best Practice: Handle rate limiting
from google.api_core import retry
from google.api_core import exceptions

@retry.Retry(predicate=retry.if_exception_type(exceptions.ResourceExhausted))
def rate_limited_search(index, query_vector, num_neighbors):
    return index.search(query_vector=query_vector, num_neighbors=num_neighbors)

# Best Practice: Monitor index size
def check_index_size(index):
    metadata = index.describe()
    current_size = metadata.vector_count
    max_size = metadata.max_vector_count
    utilization = current_size / max_size
    
    print(f"Current index size: {current_size}")
    print(f"Maximum index size: {max_size}")
    print(f"Index utilization: {utilization:.2%}")
    
    if utilization > 0.8:
        print("Warning: Index is nearing capacity. Consider scaling or optimizing.")

# Usage
test_vector = np.random.rand(128).tolist()
if validate_vector_dimension(test_vector):
    results = rate_limited_search(index, test_vector, 5)
    for result in results:
        print(f"ID: {result.id}, Distance: {result.distance}")

check_index_size(index)
```

Slide 14: Additional Resources

For further exploration of Vertex AI Vector Search and related topics, consider the following resources:

1. Google Cloud Vertex AI Documentation [https://cloud.google.com/vertex-ai/docs](https://cloud.google.com/vertex-ai/docs)
2. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" by Yu. A. Malkov and D. A. Yashunin ArXiv: [https://arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320)
3. "ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms" by Martin Aumüller, Erik Bernhardsson, and Alexander Faithfull ArXiv: [https://arxiv.org/abs/1807.05614](https://arxiv.org/abs/1807.05614)
4. Google Cloud Community Tutorials [https://cloud.google.com/community/tutorials](https://cloud.google.com/community/tutorials)
5. Vertex AI Samples Repository [https://github.com/GoogleCloudPlatform/vertex-ai-samples](https://github.com/GoogleCloudPlatform/vertex-ai-samples)

