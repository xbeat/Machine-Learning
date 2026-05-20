## Exploring Graph Theory Concepts with Python

Slide 1: Introduction to Graph Theory

Graph Theory is a branch of mathematics that studies the relationships between objects. It has numerous applications in computer science, networking, and social sciences.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple graph
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])

# Draw the graph
nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=16)
plt.title("A Simple Graph")
plt.show()
```

Slide 2: Graph Representation

Graphs can be represented in various ways, including adjacency matrices and adjacency lists. Here's an example of both representations:

```python
import numpy as np

# Adjacency Matrix
adj_matrix = np.array([
    [0, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 0]
])

# Adjacency List
adj_list = {
    0: [1, 2, 3],
    1: [0, 2],
    2: [0, 1, 3],
    3: [0, 2]
}

print("Adjacency Matrix:\n", adj_matrix)
print("\nAdjacency List:", adj_list)
```

Slide 3: Graph Traversal - Depth-First Search (DFS)

DFS explores a graph by going as deep as possible along each branch before backtracking.

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=' ')
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Example usage
graph = {0: [1, 2], 1: [2], 2: [3], 3: [1, 2]}
print("DFS traversal:")
dfs(graph, 0)
```

Slide 4: Graph Traversal - Breadth-First Search (BFS)

BFS explores a graph by visiting all neighbors at the present depth before moving to nodes at the next depth level.

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        vertex = queue.popleft()
        print(vertex, end=' ')
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Example usage
graph = {0: [1, 2], 1: [2], 2: [3], 3: [1, 2]}
print("BFS traversal:")
bfs(graph, 0)
```

Slide 5: Shortest Path - Dijkstra's Algorithm

Dijkstra's algorithm finds the shortest path between nodes in a graph with non-negative edge weights.

```python
import heapq

def dijkstra(graph, start):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example usage
graph = {
    'A': {'B': 4, 'C': 2},
    'B': {'D': 3, 'E': 1},
    'C': {'B': 1, 'D': 5},
    'D': {'E': 2},
    'E': {}
}
print(dijkstra(graph, 'A'))
```

Slide 6: Minimum Spanning Tree - Kruskal's Algorithm

Kruskal's algorithm finds a minimum spanning tree for a connected weighted graph.

```python
def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def kruskal(graph):
    result = []
    i, e = 0, 0
    graph = sorted(graph, key=lambda item: item[2])
    parent = []
    rank = []
    
    for node in range(len(set(sum([(edge[0], edge[1]) for edge in graph], ())))):
        parent.append(node)
        rank.append(0)
    
    while e < len(set(sum([(edge[0], edge[1]) for edge in graph], ()))) - 1:
        u, v, w = graph[i]
        i = i + 1
        x = find(parent, u)
        y = find(parent, v)
        
        if x != y:
            e = e + 1
            result.append([u, v, w])
            union(parent, rank, x, y)
    
    return result

# Example usage
graph = [
    [0, 1, 10], [0, 2, 6], [0, 3, 5],
    [1, 3, 15], [2, 3, 4]
]
print("Minimum Spanning Tree:")
print(kruskal(graph))
```

Slide 7: Graph Coloring

Graph coloring is the assignment of labels (colors) to elements of a graph subject to certain constraints.

```python
def graph_coloring(graph):
    color_map = {}
    colors = set(range(len(graph)))
    
    for node in graph:
        used_colors = set(color_map.get(neighbor) for neighbor in graph[node] if neighbor in color_map)
        available_colors = colors - used_colors
        color_map[node] = min(available_colors)
    
    return color_map

# Example usage
graph = {
    0: [1, 2, 3],
    1: [0, 2],
    2: [0, 1, 3],
    3: [0, 2]
}
print("Graph Coloring:")
print(graph_coloring(graph))
```

Slide 8: Cycle Detection

Detecting cycles in a graph is crucial for many applications. Here's an implementation using DFS:

```python
def has_cycle(graph):
    visited = set()
    rec_stack = set()
    
    def dfs(v):
        visited.add(v)
        rec_stack.add(v)
        
        for neighbor in graph[v]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(v)
        return False
    
    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False

# Example usage
graph = {
    0: [1, 2],
    1: [2],
    2: [3],
    3: [1]
}
print("Graph has cycle:", has_cycle(graph))
```

Slide 9: Topological Sorting

Topological sorting is used for scheduling tasks with dependencies. It's applicable only to Directed Acyclic Graphs (DAGs).

```python
from collections import defaultdict

def topological_sort(graph):
    def dfs(v):
        visited.add(v)
        for neighbor in graph[v]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(v)
    
    visited = set()
    stack = []
    
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return stack[::-1]

# Example usage
graph = defaultdict(list)
graph[5].append(2)
graph[5].append(0)
graph[4].append(0)
graph[4].append(1)
graph[2].append(3)
graph[3].append(1)

print("Topological Sort:")
print(topological_sort(graph))
```

Slide 10: Strongly Connected Components

Strongly connected components are maximal subgraphs where every vertex is reachable from every other vertex.

```python
def strongly_connected_components(graph):
    def dfs(node, stack):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, stack)
        stack.append(node)
    
    def reverse_graph():
        r_graph = {node: [] for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                r_graph[neighbor].append(node)
        return r_graph
    
    visited = set()
    stack = []
    for node in graph:
        if node not in visited:
            dfs(node, stack)
    
    r_graph = reverse_graph()
    visited.clear()
    components = []
    
    while stack:
        node = stack.pop()
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)
    
    return components

# Example usage
graph = {
    0: [1, 3],
    1: [2],
    2: [0],
    3: [4],
    4: []
}
print("Strongly Connected Components:")
print(strongly_connected_components(graph))
```

Slide 11: Maximum Flow - Ford-Fulkerson Algorithm

The Ford-Fulkerson algorithm computes the maximum flow in a flow network.

```python
def ford_fulkerson(graph, source, sink):
    def bfs(graph, s, t, parent):
        visited = set()
        queue = [s]
        visited.add(s)
        while queue:
            u = queue.pop(0)
            for v in graph[u]:
                if v not in visited and graph[u][v] > 0:
                    queue.append(v)
                    visited.add(v)
                    parent[v] = u
                    if v == t:
                        return True
        return False
    
    parent = [-1] * len(graph)
    max_flow = 0
    
    while bfs(graph, source, sink, parent):
        path_flow = float("Inf")
        s = sink
        while s != source:
            path_flow = min(path_flow, graph[parent[s]][s])
            s = parent[s]
        
        max_flow += path_flow
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = parent[v]
    
    return max_flow

# Example usage
graph = [
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0]
]
source = 0
sink = 5
print("Maximum Flow:", ford_fulkerson(graph, source, sink))
```

Slide 12: Graph Centrality - Betweenness Centrality

Betweenness centrality measures the extent to which a vertex lies on paths between other vertices.

```python
import networkx as nx

def betweenness_centrality(G):
    bc = nx.betweenness_centrality(G)
    return bc

# Example usage
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)])

print("Betweenness Centrality:")
print(betweenness_centrality(G))
```

Slide 13: Real-Life Example 1: Social Network Analysis

Graph theory is extensively used in social network analysis. Let's create a simple social network and analyze it:

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a social network
G = nx.Graph()
G.add_edges_from([
    ('Alice', 'Bob'), ('Alice', 'Charlie'), ('Bob', 'David'),
    ('Charlie', 'David'), ('Charlie', 'Eve'), ('David', 'Eve'),
    ('Eve', 'Frank'), ('Frank', 'George')
])

# Visualize the network
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
plt.title("Social Network")
plt.axis('off')
plt.show()

# Analyze the network
print("Number of people:", G.number_of_nodes())
print("Number of connections:", G.number_of_edges())
print("Most connected person:", max(dict(G.degree()).items(), key=lambda x: x[1])[0])
print("Average connections per person:", sum(dict(G.degree()).values()) / G.number_of_nodes())
```

Slide 14: Real-Life Example 2: Routing in Computer Networks

Graph theory is crucial in computer networking for finding optimal routes. Let's simulate a simple network routing problem:

```python
import networkx as nx

# Create a network topology
G = nx.Graph()
G.add_weighted_edges_from([
    ('Router1', 'Router2', 5),
    ('Router1', 'Router3', 3),
    ('Router2', 'Router4', 2),
    ('Router3', 'Router4', 6),
    ('Router3', 'Router5', 4),
    ('Router4', 'Router5', 1)
])

# Find the shortest path between two routers
source = 'Router1'
destination = 'Router5'
shortest_path = nx.shortest_path(G, source, destination, weight='weight')
path_length = nx.shortest_path_length(G, source, destination, weight='weight')

print(f"Shortest path from {source} to {destination}:")
print(" -> ".join(shortest_path))
print(f"Total distance: {path_length}")

# Find all paths between two routers
all_paths = list(nx.all_simple_paths(G, source, destination))
print(f"\nAll possible paths from {source} to {destination}:")
for path in all_paths:
    print(" -> ".join(path))
```

Slide 15: Additional Resources

For those interested in diving deeper into Graph Theory and its applications, here are some valuable resources:

1. ArXiv.org papers:
   * "A Survey of Graph Neural Networks for Recommender Systems: Challenges, Methods, and Directions" (arXiv:2109.12843)
   * "Graph Neural Networks: A Review of Methods and Applications" (arXiv:1812.08434)
2. Online courses:
   * Coursera: "Algorithms on Graphs" by University of California San Diego
   * edX: "Graph Algorithms" by University of California San Diego
3. Books:
   * "Introduction to Graph Theory" by Richard J. Trudeau
   * "Graph Theory and Its Applications" by Jonathan L. Gross and Jay Yellen

These resources provide a mix of theoretical foundations and practical applications of Graph Theory in various fields.

