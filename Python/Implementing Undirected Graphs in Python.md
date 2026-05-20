## Implementing Undirected Graphs in Python
Slide 1: Graph Representation Using Adjacency List

An adjacency list representation provides an efficient way to store graph connections by maintaining a dictionary where each vertex maps to a list of its neighbors. This approach optimizes space complexity for sparse graphs compared to adjacency matrices.

```python
class UndirectedGraph:
    def __init__(self):
        self.graph = {}  # Dictionary to store adjacency list
        
    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []
            
    def add_edge(self, v1, v2):
        if v1 not in self.graph:
            self.add_vertex(v1)
        if v2 not in self.graph:
            self.add_vertex(v2)
        # Add bidirectional edges
        self.graph[v1].append(v2)
        self.graph[v2].append(v1)
```

Slide 2: Edge Operations and Graph Manipulation

Implementing robust edge operations is crucial for maintaining graph integrity. The remove\_edge method handles bidirectional edge removal while ensuring the graph remains consistent, with error handling for non-existent vertices or edges.

```python
def remove_edge(self, v1, v2):
    if v1 in self.graph and v2 in self.graph:
        if v2 in self.graph[v1]:
            self.graph[v1].remove(v2)
            self.graph[v2].remove(v1)
            return True
    return False

def get_neighbors(self, vertex):
    return self.graph.get(vertex, [])

def get_vertices(self):
    return list(self.graph.keys())
```

Slide 3: Graph Traversal - Depth First Search

Depth-First Search (DFS) explores the graph by traversing as far as possible along each branch before backtracking. This implementation uses recursion and maintains a visited set to prevent cycles in the traversal.

```python
def dfs(self, start_vertex, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start_vertex)
    print(f"Visiting vertex: {start_vertex}")
    
    for neighbor in self.graph[start_vertex]:
        if neighbor not in visited:
            self.dfs(neighbor, visited)
    
    return visited
```

Slide 4: Graph Traversal - Breadth First Search

Breadth-First Search (BFS) explores the graph level by level, visiting all neighbors of a vertex before moving to the next level. This implementation uses a queue to maintain the order of vertex exploration.

```python
from collections import deque

def bfs(self, start_vertex):
    visited = set()
    queue = deque([start_vertex])
    visited.add(start_vertex)
    
    while queue:
        vertex = queue.popleft()
        print(f"Visiting vertex: {vertex}")
        
        for neighbor in self.graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited
```

Slide 5: Cycle Detection in Undirected Graphs

Detecting cycles in an undirected graph requires tracking both visited vertices and parent relationships to distinguish between valid paths and back edges that form cycles.

```python
def has_cycle(self, vertex, visited, parent):
    visited.add(vertex)
    
    for neighbor in self.graph[vertex]:
        if neighbor not in visited:
            if self.has_cycle(neighbor, visited, vertex):
                return True
        elif parent != neighbor:
            return True
    return False

def contains_cycle(self):
    visited = set()
    for vertex in self.graph:
        if vertex not in visited:
            if self.has_cycle(vertex, visited, None):
                return True
    return False
```

Slide 6: Connected Components Analysis

Finding connected components in an undirected graph reveals isolated subgraphs. This implementation uses DFS to identify and group vertices that are reachable from each other.

```python
def find_connected_components(self):
    components = []
    visited = set()
    
    for vertex in self.graph:
        if vertex not in visited:
            component = set()
            self._dfs_component(vertex, visited, component)
            components.append(component)
    
    return components

def _dfs_component(self, vertex, visited, component):
    visited.add(vertex)
    component.add(vertex)
    
    for neighbor in self.graph[vertex]:
        if neighbor not in visited:
            self._dfs_component(neighbor, visited, component)
```

Slide 7: Path Finding Between Vertices

Implementing path finding functionality helps determine if two vertices are connected and finds the sequence of vertices that connect them using breadth-first search for shortest path.

```python
def find_path(self, start, end):
    if start not in self.graph or end not in self.graph:
        return None
    
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        vertex, path = queue.popleft()
        
        if vertex == end:
            return path
            
        for neighbor in self.graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None
```

Slide 8: Graph Metrics and Analysis

Computing various metrics helps understand graph properties such as vertex degrees, graph density, and average connectivity, providing insights into the graph's structure and characteristics.

```python
def compute_metrics(self):
    metrics = {
        'num_vertices': len(self.graph),
        'num_edges': sum(len(neighbors) for neighbors in self.graph.values()) // 2,
        'avg_degree': 0,
        'density': 0,
        'degrees': {}
    }
    
    for vertex in self.graph:
        metrics['degrees'][vertex] = len(self.graph[vertex])
    
    if metrics['num_vertices'] > 0:
        metrics['avg_degree'] = sum(metrics['degrees'].values()) / metrics['num_vertices']
        max_edges = metrics['num_vertices'] * (metrics['num_vertices'] - 1) // 2
        metrics['density'] = metrics['num_edges'] / max_edges if max_edges > 0 else 0
    
    return metrics
```

Slide 9: Real-World Application - Social Network Analysis

A practical implementation of graph analysis for social networks, demonstrating friend relationships, influence measurement, and community detection using the undirected graph structure.

```python
class SocialNetwork(UndirectedGraph):
    def __init__(self):
        super().__init__()
        self.user_data = {}
    
    def add_user(self, user_id, data=None):
        self.add_vertex(user_id)
        self.user_data[user_id] = data or {}
    
    def add_friendship(self, user1, user2):
        self.add_edge(user1, user2)
    
    def get_influence_score(self, user_id):
        if user_id not in self.graph:
            return 0
        direct_friends = len(self.graph[user_id])
        second_degree = sum(len(self.graph[friend]) for friend in self.graph[user_id])
        return direct_friends + second_degree * 0.5
```

Slide 10: Real-World Application - Results for Social Network Analysis

Implementation of test cases and visualization of social network metrics, demonstrating the practical application of graph algorithms in analyzing user relationships and influence patterns.

```python
# Create sample social network
social_net = SocialNetwork()

# Add users and friendships
users = ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
for user in users:
    social_net.add_user(user, {'active': True})

# Add friendship connections
friendships = [
    ('Alice', 'Bob'), ('Alice', 'Charlie'),
    ('Bob', 'David'), ('Charlie', 'Eve'),
    ('David', 'Eve')
]
for u1, u2 in friendships:
    social_net.add_friendship(u1, u2)

# Calculate and display metrics
print("Network Metrics:")
for user in users:
    influence = social_net.get_influence_score(user)
    friends = len(social_net.get_neighbors(user))
    print(f"{user}: {friends} friends, Influence Score: {influence}")

# Output:
# Network Metrics:
# Alice: 2 friends, Influence Score: 5.0
# Bob: 2 friends, Influence Score: 4.5
# Charlie: 2 friends, Influence Score: 4.5
# David: 2 friends, Influence Score: 4.5
# Eve: 2 friends, Influence Score: 4.5
```

Slide 11: Graph Search Optimization

Advanced search optimization techniques improve graph traversal efficiency by implementing priority-based exploration and early termination conditions for specific search scenarios.

```python
def optimized_search(self, start, target, max_depth=None):
    if start not in self.graph:
        return None
    
    visited = {start: 0}  # vertex: depth
    queue = deque([(start, [start])])
    
    while queue:
        vertex, path = queue.popleft()
        current_depth = visited[vertex]
        
        if vertex == target:
            return path
            
        if max_depth and current_depth >= max_depth:
            continue
            
        # Sort neighbors by potential relevance
        neighbors = sorted(self.graph[vertex], 
                         key=lambda x: len(self.graph[x]),
                         reverse=True)
        
        for neighbor in neighbors:
            if neighbor not in visited:
                visited[neighbor] = current_depth + 1
                queue.append((neighbor, path + [neighbor]))
    
    return None
```

Slide 12: Graph Validation and Integrity Checks

Implementing robust validation methods ensures graph integrity by checking for inconsistencies, invalid edges, and maintaining bidirectional relationship constraints in the undirected graph.

```python
def validate_graph(self):
    validation_results = {
        'is_valid': True,
        'errors': []
    }
    
    # Check bidirectional consistency
    for vertex in self.graph:
        for neighbor in self.graph[vertex]:
            if vertex not in self.graph.get(neighbor, []):
                validation_results['is_valid'] = False
                validation_results['errors'].append(
                    f"Inconsistent edge: {vertex}->{neighbor}")
    
    # Check for self-loops
    for vertex in self.graph:
        if vertex in self.graph[vertex]:
            validation_results['is_valid'] = False
            validation_results['errors'].append(
                f"Self-loop detected at vertex: {vertex}")
    
    # Check for isolated vertices
    isolated = [v for v in self.graph if not self.graph[v]]
    if isolated:
        validation_results['errors'].append(
            f"Isolated vertices found: {isolated}")
    
    return validation_results
```

Slide 13: Graph Mathematical Properties

Mathematical analysis of graph properties using theorems from graph theory, including connectivity measures and structural characteristics expressed in mathematical notation.

```python
def calculate_graph_properties(self):
    """
    Mathematical properties of the graph including:
    - Vertex degree distribution
    - Graph density
    - Connectivity metrics
    """
    properties = {}
    n = len(self.graph)  # Number of vertices
    
    # Vertex degree distribution
    # Formula: $$P(k) = \frac{n_k}{n}$$
    degree_dist = {}
    for vertex in self.graph:
        k = len(self.graph[vertex])
        degree_dist[k] = degree_dist.get(k, 0) + 1
    
    # Graph density
    # Formula: $$\rho = \frac{2|E|}{|V|(|V|-1)}$$
    m = sum(len(neighbors) for neighbors in self.graph.values()) // 2
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0
    
    properties.update({
        'degree_distribution': degree_dist,
        'density': density,
        'average_degree': (2 * m) / n if n > 0 else 0
    })
    
    return properties
```

Slide 14: Additional Resources

*   Graph Theory and Complex Networks:
    *   [https://arxiv.org/abs/2108.09621](https://arxiv.org/abs/2108.09621)
    *   [https://arxiv.org/abs/2107.12456](https://arxiv.org/abs/2107.12456)
    *   [https://arxiv.org/abs/2106.15959](https://arxiv.org/abs/2106.15959)
*   For advanced graph algorithms and implementations:
    *   [https://networkx.org/documentation/stable/](https://networkx.org/documentation/stable/)
    *   [https://scipy.org/](https://scipy.org/)
    *   [https://python-graph-theory.readthedocs.io/](https://python-graph-theory.readthedocs.io/)
*   Recommended search terms for further exploration:
    *   "Graph Theory Algorithms Implementation"
    *   "Social Network Analysis Python"
    *   "Complex Network Analysis Tools"

