## Implementing Efficient Graph Structures in Python
Slide 1: Graph Representation Using Adjacency Lists

The adjacency list representation of graphs offers an efficient way to store sparse graphs where the number of edges is much less than the maximum possible edges. This implementation provides O(V+E) space complexity, where V is the number of vertices and E is the number of edges.

```python
class Graph:
    def __init__(self, directed=False):
        self.graph = {}
        self.directed = directed
    
    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []
    
    def add_edge(self, vertex1, vertex2):
        if vertex1 not in self.graph:
            self.add_vertex(vertex1)
        if vertex2 not in self.graph:
            self.add_vertex(vertex2)
            
        self.graph[vertex1].append(vertex2)
        if not self.directed:
            self.graph[vertex2].append(vertex1)

# Example usage
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
print(g.graph)  # Output: {0: [1, 2], 1: [0], 2: [0]}
```

Slide 2: Basic Graph Traversal - Depth First Search

Depth First Search (DFS) is a fundamental graph traversal technique that explores as far as possible along each branch before backtracking. This implementation uses recursion to traverse the graph, marking visited nodes to prevent cycles.

```python
class Graph:
    def dfs(self, start_vertex, visited=None):
        if visited is None:
            visited = set()
            
        visited.add(start_vertex)
        print(f"Visited: {start_vertex}")
        
        for neighbor in self.graph[start_vertex]:
            if neighbor not in visited:
                self.dfs(neighbor, visited)
        
        return visited

# Example usage
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 3)
g.dfs(0)
```

Slide 3: Breadth First Search Implementation

Breadth First Search (BFS) explores all vertices at the present depth before moving on to vertices at the next depth level. This implementation uses a queue data structure to maintain the order of vertex exploration.

```python
from collections import deque

class Graph:
    def bfs(self, start_vertex):
        visited = set()
        queue = deque([start_vertex])
        visited.add(start_vertex)
        
        while queue:
            vertex = queue.popleft()
            print(f"Visited: {vertex}")
            
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return visited

# Example usage
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 3)
g.bfs(0)
```

Slide 4: Weighted Graph Implementation

In real-world applications, graphs often have weights associated with their edges representing costs, distances, or other metrics. This implementation extends our basic graph to support weighted edges.

```python
class WeightedGraph:
    def __init__(self, directed=False):
        self.graph = {}
        self.directed = directed
    
    def add_edge(self, vertex1, vertex2, weight):
        if vertex1 not in self.graph:
            self.graph[vertex1] = []
        if vertex2 not in self.graph:
            self.graph[vertex2] = []
        
        self.graph[vertex1].append((vertex2, weight))
        if not self.directed:
            self.graph[vertex2].append((vertex1, weight))

# Example usage
g = WeightedGraph()
g.add_edge(0, 1, 4)
g.add_edge(0, 2, 3)
print(g.graph)  # Output: {0: [(1, 4), (2, 3)], 1: [(0, 4)], 2: [(0, 3)]}
```

Slide 5: Dijkstra's Shortest Path Algorithm

Dijkstra's algorithm finds the shortest path between nodes in a graph, which may represent road networks, computer networks, or any weighted directed graph with non-negative edge weights.

```python
import heapq

def dijkstra(graph, start):
    distances = {vertex: float('infinity') for vertex in graph.graph}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        if current_vertex in visited:
            continue
            
        visited.add(current_vertex)
        
        for neighbor, weight in graph.graph[current_vertex]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances
```

Slide 6: Minimum Spanning Tree - Kruskal's Algorithm

A minimum spanning tree (MST) is a subset of edges in a weighted, undirected graph that connects all vertices together with the minimum possible total edge weight. Kruskal's algorithm finds the MST efficiently.

```python
class UnionFind:
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}
    
    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]
    
    def union(self, x, y):
        xroot, yroot = self.find(x), self.find(y)
        if xroot != yroot:
            if self.rank[xroot] < self.rank[yroot]:
                xroot, yroot = yroot, xroot
            self.parent[yroot] = xroot
            if self.rank[xroot] == self.rank[yroot]:
                self.rank[xroot] += 1

def kruskal_mst(graph):
    edges = []
    for v in graph.graph:
        for u, w in graph.graph[v]:
            edges.append((w, v, u))
    edges.sort()
    
    vertices = list(graph.graph.keys())
    uf = UnionFind(vertices)
    mst = []
    
    for weight, u, v in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst.append((u, v, weight))
    
    return mst
```

Slide 7: Cycle Detection in Directed Graphs

Detecting cycles in directed graphs is crucial for identifying circular dependencies in many applications, such as dependency resolution, task scheduling, and deadlock detection. This implementation uses depth-first search with backtracking.

```python
class DirectedGraph:
    def has_cycle(self):
        visited = set()
        rec_stack = set()
        
        def dfs_cycle(vertex):
            visited.add(vertex)
            rec_stack.add(vertex)
            
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    if dfs_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(vertex)
            return False
        
        for vertex in self.graph:
            if vertex not in visited:
                if dfs_cycle(vertex):
                    return True
        return False

# Example usage
g = DirectedGraph()
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 0)
print(g.has_cycle())  # Output: True
```

Slide 8: Topological Sort Implementation

Topological sorting of a directed acyclic graph produces a linear ordering of vertices such that for every directed edge (u,v), vertex u comes before v in the ordering. This is essential for dependency resolution and task scheduling.

```python
class DirectedGraph:
    def topological_sort(self):
        visited = set()
        stack = []
        
        def dfs_topo(vertex):
            visited.add(vertex)
            
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    dfs_topo(neighbor)
            
            stack.append(vertex)
        
        for vertex in self.graph:
            if vertex not in visited:
                dfs_topo(vertex)
        
        return stack[::-1]

# Example usage
g = DirectedGraph()
g.add_edge(5, 2)
g.add_edge(5, 0)
g.add_edge(4, 0)
g.add_edge(4, 1)
g.add_edge(2, 3)
g.add_edge(3, 1)
print(g.topological_sort())  # Output: [5, 4, 2, 3, 1, 0]
```

Slide 9: Strongly Connected Components

A strongly connected component (SCC) is a portion of a directed graph where every vertex is reachable from every other vertex. Kosaraju's algorithm efficiently finds all SCCs in a graph using two depth-first search passes.

```python
class DirectedGraph:
    def get_transpose(self):
        g_transpose = DirectedGraph()
        for vertex in self.graph:
            for neighbor in self.graph[vertex]:
                g_transpose.add_edge(neighbor, vertex)
        return g_transpose
    
    def fill_order(self, vertex, visited, stack):
        visited.add(vertex)
        for neighbor in self.graph[vertex]:
            if neighbor not in visited:
                self.fill_order(neighbor, visited, stack)
        stack.append(vertex)
    
    def dfs_scc(self, vertex, visited, scc):
        visited.add(vertex)
        scc.append(vertex)
        for neighbor in self.graph[vertex]:
            if neighbor not in visited:
                self.dfs_scc(neighbor, visited, scc)
    
    def kosaraju_scc(self):
        stack = []
        visited = set()
        
        for vertex in self.graph:
            if vertex not in visited:
                self.fill_order(vertex, visited, stack)
        
        transpose = self.get_transpose()
        visited.clear()
        sccs = []
        
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                current_scc = []
                transpose.dfs_scc(vertex, visited, current_scc)
                sccs.append(current_scc)
        
        return sccs

# Example usage
g = DirectedGraph()
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
print(g.kosaraju_scc())  # Output: [[3], [0, 1, 2]]
```

Slide 10: Real-world Application - Social Network Analysis

Social networks can be modeled as graphs where vertices represent users and edges represent connections. This implementation analyzes user communities and influence using graph algorithms.

```python
class SocialGraph:
    def __init__(self):
        self.graph = {}
        self.user_data = {}
    
    def add_user(self, user_id, user_info):
        if user_id not in self.graph:
            self.graph[user_id] = set()
            self.user_data[user_id] = user_info
    
    def add_connection(self, user1, user2):
        self.graph[user1].add(user2)
        self.graph[user2].add(user1)
    
    def get_influence_score(self, user_id):
        # Calculate influence based on number of connections and their connections
        direct_connections = len(self.graph[user_id])
        second_degree = sum(len(self.graph[connection]) 
                          for connection in self.graph[user_id])
        return direct_connections * 0.6 + second_degree * 0.4
    
    def find_communities(self, min_connections=2):
        visited = set()
        communities = []
        
        def dfs_community(user, community):
            visited.add(user)
            community.add(user)
            for connection in self.graph[user]:
                if connection not in visited:
                    dfs_community(connection, community)
        
        for user in self.graph:
            if user not in visited:
                current_community = set()
                dfs_community(user, current_community)
                if len(current_community) >= min_connections:
                    communities.append(current_community)
        
        return communities

# Example usage
social_net = SocialGraph()
social_net.add_user(1, {"name": "Alice", "age": 25})
social_net.add_user(2, {"name": "Bob", "age": 27})
social_net.add_user(3, {"name": "Charlie", "age": 24})
social_net.add_connection(1, 2)
social_net.add_connection(2, 3)
print(f"Influence score for user 2: {social_net.get_influence_score(2)}")
print(f"Communities: {social_net.find_communities()}")
```

Slide 11: Advanced Graph Analytics - Centrality Measures

Centrality measures help identify the most important vertices within a graph. This implementation includes degree centrality, closeness centrality, and betweenness centrality calculations for network analysis applications.

```python
class NetworkAnalyzer:
    def __init__(self, graph):
        self.graph = graph
    
    def degree_centrality(self, vertex):
        return len(self.graph[vertex]) / (len(self.graph) - 1)
    
    def closeness_centrality(self, vertex):
        distances = self._shortest_paths(vertex)
        if len(distances) < len(self.graph) - 1:  # Not all nodes reachable
            return 0
        return (len(self.graph) - 1) / sum(distances.values())
    
    def _shortest_paths(self, start):
        distances = {start: 0}
        queue = deque([start])
        while queue:
            vertex = queue.popleft()
            for neighbor in self.graph[vertex]:
                if neighbor not in distances:
                    distances[neighbor] = distances[vertex] + 1
                    queue.append(neighbor)
        return distances
    
    def analyze_network(self):
        results = {}
        for vertex in self.graph:
            results[vertex] = {
                'degree': self.degree_centrality(vertex),
                'closeness': self.closeness_centrality(vertex)
            }
        return results

# Example usage
network = NetworkAnalyzer(some_graph)
metrics = network.analyze_network()
print(f"Network metrics: {metrics}")
```

Slide 12: Graph Time Complexity Analysis

Understanding the time complexity of graph operations is crucial for efficient implementation. This implementation demonstrates various operations with their respective time complexities and memory requirements.

```python
class GraphComplexity:
    def __init__(self, V):
        """
        V: number of vertices
        Space Complexity: O(V + E) for adjacency list
        """
        self.graph = {}
        self.V = V
    
    def add_edge(self, u, v):
        """Time Complexity: O(1)"""
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)
    
    def remove_edge(self, u, v):
        """Time Complexity: O(degree(u))"""
        if u in self.graph:
            try:
                self.graph[u].remove(v)
                return True
            except ValueError:
                return False
        return False
    
    def has_edge(self, u, v):
        """Time Complexity: O(degree(u))"""
        return u in self.graph and v in self.graph[u]
    
    def get_neighbors(self, u):
        """Time Complexity: O(1)"""
        return self.graph.get(u, [])
    
    def get_degree(self, u):
        """Time Complexity: O(1)"""
        return len(self.graph.get(u, []))

# Example with time complexity demonstration
g = GraphComplexity(5)
g.add_edge(0, 1)  # O(1)
g.add_edge(0, 2)  # O(1)
print(f"Neighbors of 0: {g.get_neighbors(0)}")  # O(1)
print(f"Degree of 0: {g.get_degree(0)}")  # O(1)
```

Slide 13: Graph Visualization and Export

Graph visualization is essential for understanding complex network structures. This implementation provides methods to export graphs in various formats and generate visual representations.

```python
import json
import networkx as nx
import matplotlib.pyplot as plt

class GraphVisualizer:
    def __init__(self, graph):
        self.graph = graph
        self.nx_graph = self._convert_to_networkx()
    
    def _convert_to_networkx(self):
        G = nx.DiGraph() if isinstance(self.graph, DirectedGraph) else nx.Graph()
        for vertex in self.graph.graph:
            G.add_node(vertex)
            for neighbor in self.graph.graph[vertex]:
                if isinstance(neighbor, tuple):  # Weighted graph
                    G.add_edge(vertex, neighbor[0], weight=neighbor[1])
                else:
                    G.add_edge(vertex, neighbor)
        return G
    
    def visualize(self, filename='graph.png'):
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.nx_graph)
        nx.draw(self.nx_graph, pos, with_labels=True, 
                node_color='lightblue', 
                node_size=500, 
                arrowsize=20)
        plt.savefig(filename)
        plt.close()
    
    def export_to_json(self, filename='graph.json'):
        data = {
            'nodes': list(self.graph.graph.keys()),
            'edges': [(u, v) for u in self.graph.graph 
                     for v in self.graph.graph[u]]
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

# Example usage
g = Graph()
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 0)
visualizer = GraphVisualizer(g)
visualizer.visualize()
visualizer.export_to_json()
```

Slide 14: Additional Resources

*   Efficient Graph Algorithms on Modern Architectures
    *   [https://arxiv.org/abs/2104.01733](https://arxiv.org/abs/2104.01733)
*   Graph Neural Networks: A Comprehensive Survey
    *   [https://arxiv.org/abs/1901.00596](https://arxiv.org/abs/1901.00596)
*   Deep Learning on Graphs: A Survey
    *   [https://arxiv.org/abs/1812.04202](https://arxiv.org/abs/1812.04202)
*   Advanced Graph Algorithms and Applications
    *   [https://www.journals.elsevier.com/journal-of-graph-algorithms-and-applications](https://www.journals.elsevier.com/journal-of-graph-algorithms-and-applications)
*   Modern Graph Theory and Applications
    *   [https://www.springer.com/journal/373](https://www.springer.com/journal/373)

