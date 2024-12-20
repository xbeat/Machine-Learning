## Efficient Pair Generation with itertools.combinations
Slide 1: Introduction to itertools.combinations

The itertools.combinations function provides an efficient way to generate all possible combinations of a specified length from an iterable. This powerful tool eliminates the need for complex nested loops and manual index tracking, making code more readable and maintainable.

```python
from itertools import combinations

# Basic usage of combinations
items = ['A', 'B', 'C', 'D']
pairs = list(combinations(items, 2))
print(f"All possible pairs: {pairs}")
# Output: [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')]

# Mathematical representation of combinations
# $$C(n,r) = \frac{n!}{r!(n-r)!}$$
```

Slide 2: Efficiency Comparison

Understanding the performance benefits of itertools.combinations versus manual nested loops is crucial for optimization. The itertools implementation uses optimized C code internally, making it significantly faster than equivalent Python loops.

```python
import time
from itertools import combinations

def manual_combinations(items, r):
    result = []
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            result.append((items[i], items[j]))
    return result

# Performance comparison
items = list(range(1000))
start = time.time()
list(combinations(items, 2))
itertools_time = time.time() - start

start = time.time()
manual_combinations(items, 2)
manual_time = time.time() - start

print(f"Itertools: {itertools_time:.4f}s")
print(f"Manual: {manual_time:.4f}s")
```

Slide 3: Memory Efficiency with Generators

The combinations function returns an iterator, allowing for memory-efficient processing of large datasets. This approach prevents memory overflow when working with extensive collections by generating combinations on-demand.

```python
from itertools import combinations
import sys

# Memory efficient processing
large_dataset = range(10000)
combos = combinations(large_dataset, 2)

# Process combinations one at a time
for i, combo in enumerate(combos):
    if i < 5:  # Show first 5 combinations
        print(combo)
    else:
        break

# Compare memory usage
list_size = sys.getsizeof(list(combinations(range(1000), 2)))
iterator_size = sys.getsizeof(combinations(range(1000), 2))
print(f"List size: {list_size} bytes")
print(f"Iterator size: {iterator_size} bytes")
```

Slide 4: Working with Custom Objects

Working with combinations extends beyond simple types to custom objects and complex data structures. This functionality enables sophisticated data processing and analysis in real-world applications.

```python
from dataclasses import dataclass
from itertools import combinations

@dataclass
class Student:
    name: str
    skill: int

# Create student pairs for project assignments
students = [
    Student("Alice", 95),
    Student("Bob", 88),
    Student("Charlie", 92),
    Student("Diana", 90)
]

# Generate balanced pairs based on skill levels
for pair in combinations(students, 2):
    avg_skill = (pair[0].skill + pair[1].skill) / 2
    print(f"Pair: {pair[0].name}-{pair[1].name}, Avg Skill: {avg_skill}")
```

Slide 5: Filtering Combinations

Combining itertools.combinations with filter functions allows for sophisticated selection of specific combination patterns. This approach maintains code clarity while implementing complex filtering logic.

```python
from itertools import combinations

def is_valid_combination(combo):
    return sum(combo) <= 10

# Generate filtered combinations
numbers = [2, 4, 6, 8]
valid_combos = filter(is_valid_combination, combinations(numbers, 2))

print("Valid combinations (sum <= 10):")
for combo in valid_combos:
    print(f"{combo}: sum = {sum(combo)}")

# Advanced filtering with comprehension
filtered_combos = [
    combo for combo in combinations(numbers, 2)
    if abs(combo[0] - combo[1]) < 3
]
print("\nCombinations with difference < 3:", filtered_combos)
```

Slide 6: Real-World Application - Portfolio Analysis

Implementing portfolio optimization using combinations to analyze different asset allocations. This example demonstrates how combinations can be used in financial analysis to evaluate multiple investment scenarios efficiently.

```python
from itertools import combinations
import numpy as np

class PortfolioAnalyzer:
    def __init__(self, assets, returns):
        self.assets = assets
        self.returns = returns
    
    def analyze_portfolios(self, portfolio_size):
        portfolios = []
        for combo in combinations(range(len(self.assets)), portfolio_size):
            selected_assets = [self.assets[i] for i in combo]
            selected_returns = [self.returns[i] for i in combo]
            avg_return = np.mean(selected_returns)
            risk = np.std(selected_returns)
            portfolios.append({
                'assets': selected_assets,
                'return': avg_return,
                'risk': risk
            })
        return portfolios

# Example usage
assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
returns = [0.15, 0.12, 0.14, 0.18, 0.10]

analyzer = PortfolioAnalyzer(assets, returns)
results = analyzer.analyze_portfolios(3)

for portfolio in results[:5]:
    print(f"Portfolio: {portfolio['assets']}")
    print(f"Expected Return: {portfolio['return']:.2f}")
    print(f"Risk: {portfolio['risk']:.2f}\n")
```

Slide 7: Handling Large-Scale Combinations with Generators

When dealing with large datasets, memory management becomes crucial. This implementation shows how to process combinations in chunks while maintaining memory efficiency.

```python
from itertools import combinations, islice
import sys

def chunk_combinations(iterable, r, chunk_size=1000):
    combo_iterator = combinations(iterable, r)
    while True:
        chunk = list(islice(combo_iterator, chunk_size))
        if not chunk:
            break
        yield chunk

# Example with large dataset
large_dataset = range(1000)
total_processed = 0
memory_usage = []

for chunk_idx, chunk in enumerate(chunk_combinations(large_dataset, 2, 100)):
    # Process chunk
    total_processed += len(chunk)
    current_memory = sys.getsizeof(chunk)
    memory_usage.append(current_memory)
    
    if chunk_idx < 2:  # Show first two chunks
        print(f"Chunk {chunk_idx}: First combination = {chunk[0]}")
        print(f"Memory usage: {current_memory} bytes\n")

print(f"Total combinations processed: {total_processed}")
print(f"Average memory usage per chunk: {sum(memory_usage)/len(memory_usage):.2f} bytes")
```

Slide 8: Graph Theory Application

Combinations are essential in graph theory for analyzing network connections and relationships. This implementation demonstrates using combinations for finding all possible edges in an undirected graph.

```python
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt

def create_graph_from_nodes(nodes):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    
    # Generate all possible edges using combinations
    edges = list(combinations(nodes, 2))
    
    # Add edges with weights based on node values
    for edge in edges:
        weight = abs(edge[0] - edge[1])
        G.add_edge(*edge, weight=weight)
    
    return G, edges

# Example usage
nodes = [1, 2, 3, 4, 5]
G, edges = create_graph_from_nodes(nodes)

print("Graph edges with weights:")
for edge in G.edges(data=True):
    print(f"Edge {edge[0]}-{edge[1]}: Weight = {edge[2]['weight']}")

# Calculate basic graph metrics
print(f"\nNumber of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Graph density: {nx.density(G):.2f}")
```

Slide 9: Database Query Optimization

Using combinations for efficient database query pattern analysis. This example shows how to optimize complex join operations by analyzing different table combination patterns.

```python
from itertools import combinations
from typing import List, Dict
import time

class QueryOptimizer:
    def __init__(self, tables: List[str], table_sizes: Dict[str, int]):
        self.tables = tables
        self.table_sizes = table_sizes
    
    def analyze_join_patterns(self):
        join_patterns = []
        for r in range(2, len(self.tables) + 1):
            for combo in combinations(self.tables, r):
                total_size = sum(self.table_sizes[table] for table in combo)
                join_cost = self._calculate_join_cost(combo, total_size)
                join_patterns.append({
                    'tables': combo,
                    'total_size': total_size,
                    'join_cost': join_cost
                })
        return sorted(join_patterns, key=lambda x: x['join_cost'])
    
    def _calculate_join_cost(self, tables, total_size):
        return total_size * len(tables) * np.log2(len(tables))

# Example usage
tables = ['users', 'orders', 'products', 'categories']
sizes = {'users': 1000, 'orders': 5000, 'products': 200, 'categories': 50}

optimizer = QueryOptimizer(tables, sizes)
patterns = optimizer.analyze_join_patterns()

print("Optimal join patterns:")
for pattern in patterns[:3]:
    print(f"Tables: {pattern['tables']}")
    print(f"Total size: {pattern['total_size']}")
    print(f"Join cost: {pattern['join_cost']:.2f}\n")
```

Slide 10: Text Analysis Pattern Matching

Implementing pattern matching for text analysis using combinations to identify recurring patterns in text data. This approach enables efficient text mining and pattern recognition across large documents.

```python
from itertools import combinations
from collections import Counter
import re

class TextPatternAnalyzer:
    def __init__(self, text):
        self.text = text
        self.words = re.findall(r'\w+', text.lower())
    
    def find_word_patterns(self, pattern_length):
        patterns = []
        for combo in combinations(self.words, pattern_length):
            if self._is_sequential(combo):
                patterns.append(' '.join(combo))
        return Counter(patterns)
    
    def _is_sequential(self, word_combo):
        text_str = ' '.join(self.words)
        pattern = ' '.join(word_combo)
        return pattern in text_str

# Example usage
text = """The quick brown fox jumps over the lazy dog.
          The quick brown fox runs fast. The lazy dog sleeps."""

analyzer = TextPatternAnalyzer(text)
patterns = analyzer.find_word_patterns(3)

print("Most common 3-word patterns:")
for pattern, count in patterns.most_common(5):
    print(f"Pattern: '{pattern}' - Occurrences: {count}")
```

Slide 11: Machine Learning Feature Selection

Using combinations for automated feature selection in machine learning models. This implementation demonstrates how to efficiently evaluate different feature combinations for optimal model performance.

```python
from itertools import combinations
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class FeatureSelector:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.features = list(range(X.shape[1]))
    
    def select_best_features(self, max_features):
        best_score = -np.inf
        best_combo = None
        
        for n in range(2, max_features + 1):
            for combo in combinations(self.features, n):
                X_subset = self.X[:, list(combo)]
                score = np.mean(cross_val_score(
                    RandomForestClassifier(),
                    X_subset, self.y,
                    cv=5
                ))
                
                if score > best_score:
                    best_score = score
                    best_combo = combo
        
        return best_combo, best_score

# Generate sample dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Feature selection
selector = FeatureSelector(X, y)
best_features, score = selector.select_best_features(5)

print(f"Best feature combination: {best_features}")
print(f"Cross-validation score: {score:.4f}")
```

Slide 12: Time Series Analysis

Implementing sliding window analysis using combinations for time series data. This approach enables pattern detection and anomaly identification in sequential data.

```python
from itertools import combinations
import pandas as pd
import numpy as np

class TimeSeriesAnalyzer:
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
    
    def analyze_patterns(self):
        patterns = []
        indices = range(len(self.data) - self.window_size + 1)
        
        for combo in combinations(indices, 2):
            window1 = self.data[combo[0]:combo[0]+self.window_size]
            window2 = self.data[combo[1]:combo[1]+self.window_size]
            
            similarity = self._calculate_similarity(window1, window2)
            patterns.append({
                'windows': combo,
                'similarity': similarity
            })
        
        return sorted(patterns, key=lambda x: x['similarity'], reverse=True)
    
    def _calculate_similarity(self, w1, w2):
        return 1 / (1 + np.mean(np.abs(w1 - w2)))

# Example usage
np.random.seed(42)
time_series = pd.Series(np.random.randn(100))

analyzer = TimeSeriesAnalyzer(time_series, window_size=5)
similar_patterns = analyzer.analyze_patterns()

print("Most similar window pairs:")
for pattern in similar_patterns[:3]:
    print(f"Windows: {pattern['windows']}")
    print(f"Similarity: {pattern['similarity']:.4f}\n")
```

Slide 13: Additional Resources

*   Combinatorial Optimization in Python: [https://arxiv.org/abs/2006.12456](https://arxiv.org/abs/2006.12456)
*   Efficient Algorithms for Feature Selection: [https://www.sciencedirect.com/science/article/pii/S0031320318302711](https://www.sciencedirect.com/science/article/pii/S0031320318302711)
*   Performance Analysis of Python Itertools: [https://dl.acm.org/doi/10.1145/3299869.3300075](https://dl.acm.org/doi/10.1145/3299869.3300075)
*   Time Series Pattern Mining using Combinations: [https://www.researchgate.net/publication/339876054](https://www.researchgate.net/publication/339876054)
*   Advanced Graph Theory Applications: [https://networkx.org/documentation/stable/reference/algorithms/index.html](https://networkx.org/documentation/stable/reference/algorithms/index.html)

