## Importance of Big O Notation for Data Scientists
Slide 1: Time Complexity Fundamentals in Data Science

Time complexity analysis is fundamental to writing efficient data science code. Understanding Big O notation helps data scientists evaluate algorithmic efficiency, particularly crucial when dealing with large datasets where performance impacts are magnified exponentially.

```python
def measure_complexity(func, input_sizes):
    import time
    import matplotlib.pyplot as plt
    
    times = []
    for n in input_sizes:
        data = list(range(n))
        start = time.time()
        func(data)
        end = time.time()
        times.append(end - start)
    
    return times

# Example function with O(n^2) complexity
def quadratic_algorithm(arr):
    n = len(arr)
    result = []
    for i in range(n):
        for j in range(n):
            result.append(arr[i] * arr[j])
    return result

# Test with increasing input sizes
sizes = [100, 500, 1000, 2000, 3000]
execution_times = measure_complexity(quadratic_algorithm, sizes)
```

Slide 2: Linear vs Quadratic Complexity in Feature Engineering

Understanding the difference between linear and quadratic time complexity is crucial when designing feature engineering pipelines. This comparison demonstrates how algorithmic choices impact processing time as dataset size increases.

```python
import numpy as np
import time

def linear_feature(X):
    # O(n) complexity
    return np.mean(X, axis=1)

def quadratic_feature(X):
    # O(n^2) complexity
    n = X.shape[0]
    result = np.zeros(n)
    for i in range(n):
        for j in range(n):
            result[i] += np.sum(X[i] * X[j])
    return result

# Generate sample data
n_samples = 1000
n_features = 10
X = np.random.randn(n_samples, n_features)

# Compare execution times
start = time.time()
linear_result = linear_feature(X)
linear_time = time.time() - start

start = time.time()
quadratic_result = quadratic_feature(X)
quadratic_time = time.time() - start

print(f"Linear time: {linear_time:.4f}s")
print(f"Quadratic time: {quadratic_time:.4f}s")
```

Slide 3: Space-Time Tradeoffs in Feature Caching

When developing machine learning pipelines, understanding space-time tradeoffs becomes crucial. This implementation demonstrates how caching computed features can significantly improve performance despite increased memory usage.

```python
class FeatureCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        
    def compute_expensive_feature(self, data):
        # Simulate expensive computation
        return np.sum([x ** 2 for x in data])
    
    def get_feature(self, data_key, data):
        if data_key in self.cache:
            return self.cache[data_key]
        
        if len(self.cache) >= self.max_size:
            # Implement LRU eviction
            self.cache.pop(next(iter(self.cache)))
            
        result = self.compute_expensive_feature(data)
        self.cache[data_key] = result
        return result

# Usage example
cache = FeatureCache()
data = [1, 2, 3, 4, 5]
result1 = cache.get_feature("key1", data)  # Computed
result2 = cache.get_feature("key1", data)  # Retrieved from cache
```

Slide 4: Optimizing DataFrame Operations

Data scientists frequently work with pandas DataFrames, where inefficient operations can lead to significant performance bottlenecks. Understanding vectorization and proper indexing is crucial for maintaining O(n) complexity.

```python
import pandas as pd
import numpy as np

# Generate sample data
n_rows = 100000
df = pd.DataFrame({
    'A': np.random.randn(n_rows),
    'B': np.random.randn(n_rows),
    'C': np.random.choice(['X', 'Y', 'Z'], n_rows)
})

# Bad practice - O(n) with high constant factor
def inefficient_transform(df):
    result = []
    for idx in range(len(df)):
        if df.iloc[idx]['C'] == 'X':
            result.append(df.iloc[idx]['A'] * df.iloc[idx]['B'])
        else:
            result.append(df.iloc[idx]['A'] + df.iloc[idx]['B'])
    return result

# Efficient practice - Vectorized operations
def efficient_transform(df):
    mask = df['C'] == 'X'
    result = np.zeros(len(df))
    result[mask] = df.loc[mask, 'A'] * df.loc[mask, 'B']
    result[~mask] = df.loc[~mask, 'A'] + df.loc[~mask, 'B']
    return result
```

Slide 5: Hash-Based Feature Selection

Hash-based feature selection provides an efficient O(n) approach for dimensionality reduction in high-dimensional datasets, particularly useful in text processing and feature engineering for large-scale machine learning.

```python
import hashlib
from collections import defaultdict

class HashFeatureSelector:
    def __init__(self, n_features=1000):
        self.n_features = n_features
        self.feature_map = defaultdict(int)
    
    def hash_feature(self, feature_name):
        # Consistent hashing for feature mapping
        hash_value = int(hashlib.md5(feature_name.encode()).hexdigest(), 16)
        return hash_value % self.n_features
    
    def transform_features(self, feature_dict):
        transformed = np.zeros(self.n_features)
        for feature, value in feature_dict.items():
            idx = self.hash_feature(feature)
            transformed[idx] += value
        return transformed

# Example usage
features = {f"feature_{i}": np.random.random() for i in range(10000)}
selector = HashFeatureSelector(n_features=1000)
reduced_features = selector.transform_features(features)
```

Slide 6: Online Learning with Memory Constraints

In real-world applications, data scientists must handle streaming data efficiently while maintaining memory constraints. This implementation demonstrates an online learning approach with constant memory complexity O(1) per update.

```python
class OnlineStatistics:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # For calculating variance
        
    def update(self, new_value):
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
        delta2 = new_value - self.mean
        self.M2 += delta * delta2
    
    def get_variance(self):
        return self.M2 / self.count if self.count > 0 else 0.0
    
    def get_std(self):
        return np.sqrt(self.get_variance())

# Usage example
online_stats = OnlineStatistics()
data_stream = np.random.randn(1000000)
for value in data_stream:
    online_stats.update(value)

print(f"Mean: {online_stats.mean:.4f}")
print(f"Std: {online_stats.get_std():.4f}")
```

Slide 7: Efficient Distance Computations

Distance computations are fundamental in machine learning algorithms like k-NN and clustering. This implementation shows how to optimize pairwise distance calculations using vectorization and efficient memory management.

```python
def efficient_pairwise_distances(X, Y=None, metric='euclidean'):
    if Y is None:
        Y = X
    
    if metric == 'euclidean':
        # Efficient computation using matrix operations
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        X_norm = (X ** 2).sum(axis=1)
        if Y is X:
            Y_norm = X_norm
        else:
            Y_norm = (Y ** 2).sum(axis=1)
            
        distances = np.sqrt(
            X_norm[:, np.newaxis] + Y_norm[np.newaxis, :] - 
            2 * np.dot(X, Y.T)
        )
        return distances

# Example usage
X = np.random.randn(1000, 50)
Y = np.random.randn(500, 50)
distances = efficient_pairwise_distances(X, Y)
```

Slide 8: Optimized Text Preprocessing

Text preprocessing is a common bottleneck in NLP tasks. This implementation shows how to achieve O(n) complexity while efficiently handling large text corpora using generators and minimal memory footprint.

```python
from collections import defaultdict
import re

class EfficientTextPreprocessor:
    def __init__(self):
        self.vocab = defaultdict(int)
        self.word_pattern = re.compile(r'\b\w+\b')
    
    def preprocess_stream(self, text_stream):
        """Process text stream with constant memory usage"""
        for text in text_stream:
            # Lowercase and split in one pass
            for match in self.word_pattern.finditer(text.lower()):
                word = match.group()
                yield word
                
    def build_vocab(self, text_stream, max_vocab_size=10000):
        """Build vocabulary with frequency threshold"""
        for word in self.preprocess_stream(text_stream):
            self.vocab[word] += 1
            
        # Keep only top K words
        sorted_vocab = sorted(
            self.vocab.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:max_vocab_size]
        
        return dict(sorted_vocab)

# Example usage
texts = [
    "This is a sample text",
    "Another example text",
    # ... more texts
]
preprocessor = EfficientTextPreprocessor()
vocab = preprocessor.build_vocab(texts)
```

Slide 9: Memory-Efficient Feature Selection

Feature selection in high-dimensional spaces requires careful memory management. This implementation uses a streaming approach to compute feature importance scores with O(n) complexity and constant memory usage.

```python
class StreamingFeatureSelector:
    def __init__(self, n_features):
        self.n_features = n_features
        self.feature_scores = np.zeros(n_features)
        self.seen_samples = 0
        
    def _mutual_information(self, x, y):
        # Simplified MI calculation for streaming data
        return np.abs(np.corrcoef(x, y)[0, 1])
    
    def update(self, X_batch, y_batch):
        """Update feature scores with new batch of data"""
        batch_size = len(y_batch)
        self.seen_samples += batch_size
        
        # Update feature scores incrementally
        for j in range(self.n_features):
            mi_score = self._mutual_information(X_batch[:, j], y_batch)
            # Exponential moving average
            alpha = batch_size / self.seen_samples
            self.feature_scores[j] = (1 - alpha) * self.feature_scores[j] + alpha * mi_score
    
    def get_top_features(self, k):
        return np.argsort(self.feature_scores)[-k:]

# Example usage
selector = StreamingFeatureSelector(n_features=1000)
X = np.random.randn(10000, 1000)
y = np.random.randint(0, 2, 10000)

# Process in batches
batch_size = 100
for i in range(0, len(X), batch_size):
    X_batch = X[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    selector.update(X_batch, y_batch)
```

Slide 10: Efficient Mini-batch Processing

Mini-batch processing is crucial for handling large datasets in deep learning. This implementation demonstrates how to maintain O(m) memory complexity where m is the batch size, regardless of the total dataset size.

```python
class MiniBatchProcessor:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.running_stats = {
            'mean': 0,
            'variance': 0,
            'n_samples': 0
        }

    def generate_batches(self, X, y=None, shuffle=True):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, n_samples, self.batch_size):
            batch_idx = indices[start_idx:start_idx + self.batch_size]
            if y is not None:
                yield X[batch_idx], y[batch_idx]
            else:
                yield X[batch_idx]
    
    def update_running_stats(self, batch):
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_size = len(batch)
        
        # Welford's online algorithm for running statistics
        curr_n = self.running_stats['n_samples']
        new_n = curr_n + batch_size
        
        delta = batch_mean - self.running_stats['mean']
        self.running_stats['mean'] += (batch_size * delta) / new_n
        
        # Update variance
        m_a = self.running_stats['variance'] * curr_n
        m_b = batch_var * batch_size
        M2 = m_a + m_b + np.square(delta) * curr_n * batch_size / new_n
        self.running_stats['variance'] = M2 / new_n
        
        self.running_stats['n_samples'] = new_n

# Example usage
X = np.random.randn(10000, 50)
y = np.random.randint(0, 2, 10000)

processor = MiniBatchProcessor(batch_size=128)
for batch_X, batch_y in processor.generate_batches(X, y):
    processor.update_running_stats(batch_X)
```

Slide 11: Real-Time Feature Processing Pipeline

Real-time feature processing requires careful optimization to maintain constant time complexity. This implementation shows how to process streaming data with O(1) complexity per sample.

```python
class RealTimeFeatureProcessor:
    def __init__(self, feature_config):
        self.feature_config = feature_config
        self.window_size = feature_config.get('window_size', 100)
        self.buffer = collections.deque(maxlen=self.window_size)
        self.last_features = None
        
    def compute_rolling_features(self, new_data):
        self.buffer.append(new_data)
        
        if len(self.buffer) < self.window_size:
            return None
            
        features = {
            'mean': np.mean(self.buffer),
            'std': np.std(self.buffer),
            'min': np.min(self.buffer),
            'max': np.max(self.buffer),
            'median': np.median(self.buffer)
        }
        
        if self.last_features is not None:
            features['delta'] = features['mean'] - self.last_features['mean']
            features['acceleration'] = features['delta'] - self.last_features.get('delta', 0)
        
        self.last_features = features
        return features

# Example usage
processor = RealTimeFeatureProcessor({
    'window_size': 100
})

# Simulate streaming data
for _ in range(1000):
    new_data_point = np.random.randn()
    features = processor.compute_rolling_features(new_data_point)
    if features:
        # Process features in real-time
        pass
```

Slide 12: Efficient Graph-Based Feature Engineering

Graph-based features require careful implementation to avoid exponential complexity. This implementation demonstrates efficient graph traversal with O(V + E) complexity for feature extraction.

```python
from collections import defaultdict

class GraphFeatureExtractor:
    def __init__(self):
        self.graph = defaultdict(list)
        self.node_features = {}
        
    def add_edge(self, source, target, weight=1):
        self.graph[source].append((target, weight))
        
    def compute_node_centrality(self, max_depth=2):
        centrality = defaultdict(float)
        
        for node in self.graph:
            visited = set()
            queue = [(node, 0, 1.0)]
            
            while queue:
                current, depth, impact = queue.pop(0)
                if depth > max_depth:
                    continue
                    
                if current not in visited:
                    visited.add(current)
                    centrality[current] += impact
                    
                    # Add neighbors with decreased impact
                    for neighbor, weight in self.graph[current]:
                        if neighbor not in visited:
                            queue.append(
                                (neighbor, depth + 1, impact * weight * 0.5)
                            )
        
        return dict(centrality)

# Example usage
extractor = GraphFeatureExtractor()

# Build sample graph
edges = [
    (1, 2, 0.5), (2, 3, 0.7), (3, 4, 0.3),
    (2, 4, 0.6), (4, 5, 0.9), (1, 5, 0.4)
]

for source, target, weight in edges:
    extractor.add_edge(source, target, weight)

centrality_scores = extractor.compute_node_centrality()
```

Slide 13: Time-Series Feature Extraction with Linear Complexity

Efficient time-series feature extraction requires maintaining linear complexity while handling streaming data. This implementation demonstrates how to compute multiple time-series features in a single pass with O(n) complexity.

```python
class TimeSeriesFeatureExtractor:
    def __init__(self):
        self.reset_state()
        
    def reset_state(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0  # For variance
        self.M3 = 0  # For skewness
        self.M4 = 0  # For kurtosis
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.last_value = None
        self.crossings = 0
        
    def update(self, x):
        self.n += 1
        
        # Update min/max
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)
        
        # Zero crossings
        if self.last_value is not None:
            if (self.last_value < 0 and x >= 0) or (self.last_value >= 0 and x < 0):
                self.crossings += 1
        self.last_value = x
        
        # Update moments (using Welford's online algorithm)
        delta = x - self.mean
        delta_n = delta / self.n
        term1 = delta * delta_n * (self.n - 1)
        
        self.mean += delta_n
        self.M4 += term1 * delta_n * delta_n * (self.n * self.n - 3 * self.n + 3) + \
                   6 * delta_n * delta_n * self.M2 - 4 * delta_n * self.M3
        self.M3 += term1 * delta_n * (self.n - 2) - 3 * delta_n * self.M2
        self.M2 += term1
        
    def get_features(self):
        variance = self.M2 / self.n
        std_dev = np.sqrt(variance)
        
        return {
            'mean': self.mean,
            'std': std_dev,
            'skewness': (self.M3 / self.n) / (std_dev ** 3) if std_dev > 0 else 0,
            'kurtosis': (self.M4 / self.n) / (variance * variance) if variance > 0 else 0,
            'range': self.max_val - self.min_val,
            'zero_crossings': self.crossings
        }

# Example usage
extractor = TimeSeriesFeatureExtractor()
data = np.random.randn(10000)

for value in data:
    extractor.update(value)

features = extractor.get_features()
print("Time series features:", features)
```

Slide 14: Memory-Efficient Matrix Operations

Implementing efficient matrix operations is crucial for machine learning algorithms. This implementation shows how to perform matrix operations with minimal memory overhead using iterative methods.

```python
class EfficiencyMatrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = np.zeros((rows, cols))
        
    def power_iteration(self, num_iterations=100, tolerance=1e-10):
        """
        Compute largest eigenvalue and eigenvector using power iteration
        with O(n^2) complexity instead of O(n^3)
        """
        # Initialize random vector
        b_k = np.random.randn(self.cols)
        b_k = b_k / np.linalg.norm(b_k)
        
        for _ in range(num_iterations):
            # Power iteration step
            b_k1 = np.dot(self.data, b_k)
            b_k1_norm = np.linalg.norm(b_k1)
            b_k1 = b_k1 / b_k1_norm
            
            # Check convergence
            if np.allclose(b_k1, b_k, rtol=tolerance):
                break
                
            b_k = b_k1
            
        eigenvalue = np.dot(np.dot(b_k, self.data), b_k)
        return eigenvalue, b_k
        
    def efficient_multiply(self, other_matrix, chunk_size=1000):
        """Matrix multiplication with chunking for memory efficiency"""
        result = np.zeros((self.rows, other_matrix.shape[1]))
        
        for i in range(0, self.rows, chunk_size):
            chunk_end = min(i + chunk_size, self.rows)
            result[i:chunk_end] = np.dot(
                self.data[i:chunk_end], 
                other_matrix
            )
            
        return result

# Example usage
n = 5000
matrix = EfficiencyMatrix(n, n)
matrix.data = np.random.randn(n, n)

# Compute largest eigenvalue
eigenvalue, eigenvector = matrix.power_iteration()
print(f"Largest eigenvalue: {eigenvalue:.4f}")
```

Slide 15: Additional Resources

*   Efficient Machine Learning Algorithms: [https://arxiv.org/abs/2106.04279](https://arxiv.org/abs/2106.04279)
*   Optimizing Data Science Workflows: [https://arxiv.org/abs/2003.00527](https://arxiv.org/abs/2003.00527)
*   Memory-Efficient Deep Learning: [https://arxiv.org/abs/1907.10585](https://arxiv.org/abs/1907.10585)
*   Resource-Constrained Machine Learning: [https://scholar.google.com/](https://scholar.google.com/)
*   Algorithmic Efficiency in Data Science: [https://dl.acm.org/doi/proceedings/](https://dl.acm.org/doi/proceedings/)

