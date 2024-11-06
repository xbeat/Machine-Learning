## Accelerating KMeans Clustering with Approximate Nearest Neighbors
Slide 1: Understanding Traditional KMeans Implementation

The K-means clustering algorithm partitions n observations into k clusters by iteratively assigning points to their nearest centroid and updating centroids based on mean positions. This vanilla implementation showcases the core algorithm's inefficiencies in nearest neighbor search.

```python
import numpy as np

def kmeans_vanilla(X, k, max_iters=100):
    # Randomly initialize k centroids
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iters):
        # Calculate distances - O(n*k) operation
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        # Assign points to nearest centroid
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids

# Example usage
X = np.random.randn(1000, 2)  # 1000 points in 2D
labels, centroids = kmeans_vanilla(X, k=3)
```

Slide 2: Computational Complexity Analysis

Traditional KMeans implementation suffers from quadratic time complexity in the assignment step, making it inefficient for large datasets. The distance calculation between each point and centroid becomes a significant bottleneck as dataset size increases.

```python
def calculate_complexity(n_samples, n_clusters, n_features, n_iterations):
    # Time complexity analysis
    assignment_complexity = n_samples * n_clusters * n_features  # Distance calculations
    update_complexity = n_samples * n_features  # Centroid updates
    total_complexity = n_iterations * (assignment_complexity + update_complexity)
    
    print(f"Complexity Analysis for n={n_samples}, k={n_clusters}, d={n_features}:")
    print(f"Assignment Step: O({assignment_complexity:,} operations)")
    print(f"Update Step: O({update_complexity:,} operations)")
    print(f"Total: O({total_complexity:,} operations)")

# Example analysis
calculate_complexity(n_samples=100000, n_clusters=100, n_features=128, n_iterations=50)
```

Slide 3: Introduction to Approximate Nearest Neighbors

Approximate Nearest Neighbors (ANN) algorithms trade perfect accuracy for dramatic speed improvements by using specialized data structures and approximation techniques. This fundamental shift enables efficient similarity searches in high-dimensional spaces.

```python
import numpy as np
import faiss

def create_ann_index(vectors, dimension):
    # Create a flat index for exact search (baseline)
    flat_index = faiss.IndexFlatL2(dimension)
    
    # Create an IVF index with a coarse quantizer
    nlist = int(np.sqrt(len(vectors)))  # Number of clusters for coarse quantizer
    quantizer = faiss.IndexFlatL2(dimension)
    ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    # Train and add vectors
    ivf_index.train(vectors)
    ivf_index.add(vectors)
    
    return flat_index, ivf_index

# Example setup
dim = 128
num_vectors = 10000
vectors = np.random.random((num_vectors, dim)).astype('float32')
flat_index, ivf_index = create_ann_index(vectors, dim)
```

Slide 4: Implementing Faiss-based KMeans

Faiss provides highly optimized implementations of KMeans that leverage GPU acceleration and efficient indexing structures. This implementation demonstrates how to use Faiss for clustering large-scale datasets with significant performance improvements.

```python
import faiss
import numpy as np
import time

def faiss_kmeans(X, k, niter=50):
    # Convert data to float32
    X = np.ascontiguousarray(X.astype('float32'))
    n, d = X.shape
    
    # Initialize Faiss KMeans
    kmeans = faiss.Kmeans(d, k, niter=niter, gpu=True if faiss.get_num_gpus() > 0 else False)
    
    # Train KMeans
    start_time = time.time()
    kmeans.train(X)
    end_time = time.time()
    
    # Get cluster assignments
    distances, labels = kmeans.index.search(X, 1)
    
    return labels.ravel(), kmeans.centroids, end_time - start_time

# Example usage
X = np.random.randn(100000, 128).astype('float32')
labels, centroids, runtime = faiss_kmeans(X, k=100)
print(f"Clustering completed in {runtime:.2f} seconds")
```

Slide 5: Building an Efficient Index Structure

Faiss uses specialized index structures to partition the search space efficiently. The IVF (Inverted File) index divides vectors into Voronoi cells, significantly reducing the search space for nearest neighbor queries.

```python
def build_ivf_index(vectors, n_lists):
    dimension = vectors.shape[1]
    
    # Create quantizer
    quantizer = faiss.IndexFlatL2(dimension)
    
    # Create IVF index
    index = faiss.IndexIVFFlat(quantizer, dimension, n_lists)
    
    # Train the index
    index.train(vectors)
    index.add(vectors)
    
    # Set number of probes (affects accuracy vs. speed trade-off)
    index.nprobe = 4
    
    return index

# Example usage
dim = 128
num_vectors = 100000
vectors = np.random.random((num_vectors, dim)).astype('float32')
n_lists = int(4 * np.sqrt(num_vectors))  # Rule of thumb for number of cells
index = build_ivf_index(vectors, n_lists)
```

Slide 6: Performance Comparison Framework

A systematic comparison between traditional KMeans and Faiss-based implementation reveals significant performance differences. This framework measures execution time, memory usage, and clustering quality across different dataset sizes.

```python
import numpy as np
import time
import memory_profiler
from sklearn.metrics import silhouette_score
import faiss

def benchmark_clustering(X, k, methods=['vanilla', 'faiss']):
    results = {}
    
    if 'vanilla' in methods:
        # Benchmark vanilla KMeans
        start_time = time.time()
        labels_vanilla, _ = kmeans_vanilla(X, k)
        vanilla_time = time.time() - start_time
        vanilla_score = silhouette_score(X, labels_vanilla)
        results['vanilla'] = {'time': vanilla_time, 'score': vanilla_score}
    
    if 'faiss' in methods:
        # Benchmark Faiss KMeans
        start_time = time.time()
        labels_faiss, _, _ = faiss_kmeans(X, k)
        faiss_time = time.time() - start_time
        faiss_score = silhouette_score(X, labels_faiss)
        results['faiss'] = {'time': faiss_time, 'score': faiss_score}
    
    return results

# Example benchmarking
sizes = [1000, 10000, 100000]
for size in sizes:
    X = np.random.randn(size, 128).astype('float32')
    results = benchmark_clustering(X, k=10)
    print(f"\nDataset size: {size}")
    for method, metrics in results.items():
        print(f"{method}: Time={metrics['time']:.2f}s, Score={metrics['score']:.3f}")
```

Slide 7: Optimizing Search Parameters

The trade-off between search accuracy and speed in Faiss can be fine-tuned through parameters like nprobe and number of lists. This implementation demonstrates how to optimize these parameters for your specific use case.

```python
def optimize_search_parameters(vectors, queries, n_lists_range, nprobe_range):
    dimension = vectors.shape[1]
    results = {}
    
    # Ground truth using exact search
    index_exact = faiss.IndexFlatL2(dimension)
    index_exact.add(vectors)
    D_gt, I_gt = index_exact.search(queries, k=1)
    
    for n_lists in n_lists_range:
        # Create and train index
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, n_lists)
        index.train(vectors)
        index.add(vectors)
        
        for nprobe in nprobe_range:
            index.nprobe = nprobe
            
            # Measure search time and accuracy
            start_time = time.time()
            D, I = index.search(queries, k=1)
            search_time = time.time() - start_time
            
            # Calculate recall
            recall = (I == I_gt).mean()
            
            results[(n_lists, nprobe)] = {
                'time': search_time,
                'recall': recall
            }
    
    return results

# Example optimization
vectors = np.random.random((50000, 128)).astype('float32')
queries = np.random.random((1000, 128)).astype('float32')
n_lists_range = [4, 16, 64, 256]
nprobe_range = [1, 4, 16, 32]

results = optimize_search_parameters(vectors, queries, n_lists_range, nprobe_range)
for params, metrics in results.items():
    print(f"n_lists={params[0]}, nprobe={params[1]}: "
          f"Time={metrics['time']:.3f}s, Recall={metrics['recall']:.3f}")
```

Slide 8: Real-world Application - Image Clustering

This implementation demonstrates clustering of high-dimensional image features extracted from a convolutional neural network, showing how Faiss KMeans can efficiently handle large-scale image datasets.

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

def extract_image_features(image_paths, batch_size=32):
    # Load pre-trained ResNet
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    features = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = torch.stack([
            transform(Image.open(path).convert('RGB'))
            for path in batch_paths
        ])
        
        with torch.no_grad():
            batch_features = model(batch_images).squeeze()
            features.append(batch_features.numpy())
    
    return np.vstack(features)

# Example usage
image_paths = ['path/to/images/img1.jpg', 'path/to/images/img2.jpg']  # Add your image paths
features = extract_image_features(image_paths)
labels, centroids, runtime = faiss_kmeans(features, k=10)
```

Slide 9: Real-world Application - Text Embedding Clustering

Processing large-scale text embeddings for document clustering demonstrates the practical application of Faiss KMeans in natural language processing tasks, where traditional methods would be computationally prohibitive.

```python
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

def cluster_text_embeddings(texts, k=5, batch_size=32):
    # Initialize BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    
    embeddings = []
    
    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, 
                          max_length=512, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**encoded)
            # Use [CLS] token embeddings
            batch_embeddings = outputs.last_hidden_state[:, 0].numpy()
            embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings).astype('float32')
    
    # Cluster using Faiss
    labels, centroids, runtime = faiss_kmeans(embeddings, k)
    
    return labels, centroids, runtime

# Example usage
texts = [
    "Machine learning is fascinating",
    "Deep learning revolutionizes AI",
    "Natural language processing with transformers",
    # Add more texts...
]

labels, centroids, runtime = cluster_text_embeddings(texts, k=3)
print(f"Clustering completed in {runtime:.2f} seconds")
```

Slide 10: Memory-Efficient Implementation

Large-scale clustering requires careful memory management. This implementation uses memory-efficient techniques to handle datasets that exceed available RAM by processing data in chunks.

```python
def memory_efficient_kmeans(data_generator, k, chunk_size=10000):
    first_chunk = next(data_generator)
    dimension = first_chunk.shape[1]
    
    # Initialize Faiss index
    kmeans = faiss.Kmeans(dimension, k, gpu=True if faiss.get_num_gpus() > 0 else False)
    
    # Progressive training
    for i, chunk in enumerate(data_generator):
        if i == 0:
            chunk = first_chunk
        
        # Convert chunk to float32
        chunk = chunk.astype('float32')
        
        # Partial training
        kmeans.train(chunk, init_centroids=(i==0))
    
    # Final assignments
    all_labels = []
    for chunk in data_generator:
        chunk = chunk.astype('float32')
        distances, labels = kmeans.index.search(chunk, 1)
        all_labels.extend(labels.ravel())
    
    return np.array(all_labels), kmeans.centroids

def data_generator(total_size, chunk_size, dimension):
    for i in range(0, total_size, chunk_size):
        yield np.random.randn(min(chunk_size, total_size - i), dimension)

# Example usage
total_size = 1000000
dimension = 128
chunk_size = 10000
generator = data_generator(total_size, chunk_size, dimension)
labels, centroids = memory_efficient_kmeans(generator, k=100)
```

Slide 11: Optimizing Multi-GPU Performance

Leveraging multiple GPUs can significantly accelerate clustering of large datasets. This implementation demonstrates how to distribute computation across multiple GPUs using Faiss's built-in multi-GPU support.

```python
def multi_gpu_kmeans(X, k, ngpus=None):
    if ngpus is None:
        ngpus = faiss.get_num_gpus()
    
    if ngpus == 0:
        raise RuntimeError("No GPU detected")
    
    # Convert data to float32
    X = np.ascontiguousarray(X.astype('float32'))
    n, d = X.shape
    
    # Create resources
    gpu_resources = []
    for i in range(ngpus):
        res = faiss.StandardGpuResources()
        gpu_resources.append(res)
    
    # Configure parameters
    cfg = faiss.GpuMultipleClonerOptions()
    cfg.shard = True  # Distribute data across GPUs
    
    # Create Faiss index
    kmeans = faiss.Kmeans(d, k, niter=50, verbose=True, gpu=True)
    
    # Train on multiple GPUs
    start_time = time.time()
    kmeans.train(X)
    end_time = time.time()
    
    # Get cluster assignments
    distances, labels = kmeans.index.search(X, 1)
    
    return labels.ravel(), kmeans.centroids, end_time - start_time

# Example usage
X = np.random.randn(500000, 128).astype('float32')
labels, centroids, runtime = multi_gpu_kmeans(X, k=1000, ngpus=2)
print(f"Multi-GPU clustering completed in {runtime:.2f} seconds")
```

Slide 12: Performance Metrics and Visualization

Comprehensive evaluation of clustering quality through multiple metrics helps understand the trade-offs between speed and accuracy. This implementation provides visualization tools for analyzing clustering results.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import seaborn as sns

def evaluate_clustering(X, labels, centroids, runtime):
    # Calculate clustering metrics
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    
    # Calculate cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Cluster sizes
    plt.subplot(131)
    plt.bar(cluster_sizes.keys(), cluster_sizes.values())
    plt.title('Cluster Size Distribution')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Points')
    
    # Plot 2: 2D projection if high-dimensional
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        centroids_2d = pca.transform(centroids)
    else:
        X_2d = X
        centroids_2d = centroids
    
    plt.subplot(132)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=200)
    plt.title('Cluster Visualization (PCA)')
    
    # Plot 3: Performance metrics
    plt.subplot(133)
    metrics = {
        'Silhouette': silhouette,
        'Calinski-Harabasz': calinski/1000,  # Scaled for visualization
        'Runtime (s)': runtime
    }
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Performance Metrics')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
        'runtime': runtime,
        'cluster_sizes': cluster_sizes
    }

# Example usage
X = np.random.randn(10000, 128).astype('float32')
labels, centroids, runtime = faiss_kmeans(X, k=10)
metrics = evaluate_clustering(X, labels, centroids, runtime)
```

Slide 13: Handling Out-of-Distribution Data

Real-world applications often encounter out-of-distribution data points that can affect clustering quality. This implementation includes robust preprocessing and outlier detection mechanisms.

```python
def robust_clustering(X, k, contamination=0.1):
    from sklearn.preprocessing import RobustScaler
    from sklearn.covariance import EllipticEnvelope
    
    # Robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Outlier detection
    outlier_detector = EllipticEnvelope(contamination=contamination,
                                      random_state=42)
    inlier_mask = outlier_detector.fit_predict(X_scaled) == 1
    
    # Cluster inliers
    X_clean = X_scaled[inlier_mask].astype('float32')
    labels_clean, centroids, runtime = faiss_kmeans(X_clean, k)
    
    # Assign outliers to nearest cluster
    labels = np.zeros(len(X), dtype=int) - 1  # -1 for outliers
    labels[inlier_mask] = labels_clean
    
    # Process outliers
    outlier_indices = np.where(~inlier_mask)[0]
    if len(outlier_indices) > 0:
        outlier_data = X_scaled[outlier_indices].astype('float32')
        index = faiss.IndexFlatL2(centroids.shape[1])
        index.add(centroids)
        _, outlier_labels = index.search(outlier_data, 1)
        labels[outlier_indices] = outlier_labels.ravel()
    
    return labels, centroids, runtime, inlier_mask

# Example usage with synthetic outliers
X = np.vstack([
    np.random.randn(9500, 128),  # Normal data
    np.random.randn(500, 128) * 5  # Outliers
]).astype('float32')

labels, centroids, runtime, inlier_mask = robust_clustering(X, k=10)
print(f"Identified {(~inlier_mask).sum()} outliers")
```

Slide 14: Additional Resources

*   "Billion-scale similarity search with GPUs" - [https://arxiv.org/abs/1702.08734](https://arxiv.org/abs/1702.08734)
*   "Fast and Accurate k-means++ on GPUs" - [https://arxiv.org/abs/1907.07675](https://arxiv.org/abs/1907.07675)
*   "Scaling k-means to Billion-sized Datasets" - [https://arxiv.org/abs/2010.13634](https://arxiv.org/abs/2010.13634)
*   "Approximate Nearest Neighbor Search in High Dimensions" - [https://arxiv.org/abs/1806.09823](https://arxiv.org/abs/1806.09823)
*   "A Survey of Product Quantization" - [https://arxiv.org/abs/1910.03558](https://arxiv.org/abs/1910.03558)

