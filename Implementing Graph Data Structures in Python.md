## Implementing Graph Data Structures in Python
Slide 1: Graph Data Structure Fundamentals

A graph is a non-linear data structure consisting of vertices (nodes) and edges connecting these vertices. In Python, we can implement a graph using adjacency lists, where each vertex maintains a list of its adjacent vertices, providing an efficient representation for sparse graphs.

```python
class Graph:
    def __init__(self):
        # Initialize an empty dictionary to store vertices and their adjacency lists
        self.graph = {}
        
    def add_vertex(self, vertex):
        """Add a new vertex to the graph if it doesn't exist"""
        if vertex not in self.graph:
            self.graph[vertex] = []
            
    def display(self):
        """Display the graph structure"""
        for vertex in self.graph:
            print(f"{vertex}: {self.graph[vertex]}")

# Example usage
g = Graph()
g.add_vertex('A')
g.add_vertex('B')
g.display()
```

Slide 2: Adding Edges to the Graph

The process of adding edges involves connecting two vertices in the graph. For an undirected graph, we need to add the connection in both directions. This implementation includes error checking to ensure vertices exist before adding edges.

```python
def add_edge(self, vertex1, vertex2):
    """Add an edge between two vertices"""
    if vertex1 in self.graph and vertex2 in self.graph:
        # Add edge in both directions for undirected graph
        if vertex2 not in self.graph[vertex1]:
            self.graph[vertex1].append(vertex2)
        if vertex1 not in self.graph[vertex2]:
            self.graph[vertex2].append(vertex1)
    else:
        raise ValueError("One or both vertices do not exist in the graph")

# Example usage
g = Graph()
g.add_vertex('A')
g.add_vertex('B')
g.add_edge('A', 'B')
g.display()
```

Slide 3: Removing Edges

Edge removal is a critical operation in graph manipulation. This implementation carefully handles the removal of connections between vertices while maintaining graph integrity and checking for invalid operations.

```python
def remove_edge(self, vertex1, vertex2):
    """Remove an edge between two vertices"""
    if vertex1 in self.graph and vertex2 in self.graph:
        # Remove edge from both adjacency lists
        if vertex2 in self.graph[vertex1]:
            self.graph[vertex1].remove(vertex2)
        if vertex1 in self.graph[vertex2]:
            self.graph[vertex2].remove(vertex1)
    else:
        raise ValueError("One or both vertices do not exist in the graph")

# Example usage
g = Graph()
g.add_vertex('A')
g.add_vertex('B')
g.add_edge('A', 'B')
g.remove_edge('A', 'B')
g.display()
```

Slide 4: Vertex Removal Implementation

Removing a vertex requires careful handling of all its connections. This implementation ensures proper cleanup by removing all edges connected to the vertex before removing the vertex itself from the graph structure.

```python
def remove_vertex(self, vertex):
    """Remove a vertex and all its edges from the graph"""
    if vertex in self.graph:
        # Remove all edges containing this vertex
        for adj_vertex in self.graph[vertex]:
            self.graph[adj_vertex].remove(vertex)
        # Remove the vertex and its adjacency list
        del self.graph[vertex]
    else:
        raise ValueError("Vertex does not exist in the graph")

# Example usage
g = Graph()
g.add_vertex('A')
g.add_vertex('B')
g.add_vertex('C')
g.add_edge('A', 'B')
g.add_edge('B', 'C')
g.remove_vertex('B')
g.display()
```

Slide 5: Graph Traversal - Depth First Search

Depth First Search (DFS) is a fundamental graph traversal algorithm that explores as far as possible along each branch before backtracking. This implementation uses recursion to traverse the graph systematically.

```python
def dfs(self, start_vertex, visited=None):
    """Perform Depth First Search traversal"""
    if visited is None:
        visited = set()
    
    visited.add(start_vertex)
    print(start_vertex, end=' ')
    
    for neighbor in self.graph[start_vertex]:
        if neighbor not in visited:
            self.dfs(neighbor, visited)
    
    return visited

# Example usage
g = Graph()
for vertex in ['A', 'B', 'C', 'D']:
    g.add_vertex(vertex)
g.add_edge('A', 'B')
g.add_edge('B', 'C')
g.add_edge('C', 'D')
print("DFS traversal starting from vertex A:")
g.dfs('A')
```

Slide 6: Graph Traversal - Breadth First Search

Breadth First Search (BFS) explores all vertices at the present depth before moving to vertices at the next depth level. This implementation uses a queue to maintain the order of vertex exploration.

```python
from collections import deque

def bfs(self, start_vertex):
    """Perform Breadth First Search traversal"""
    visited = set()
    queue = deque([start_vertex])
    visited.add(start_vertex)
    
    while queue:
        vertex = queue.popleft()
        print(vertex, end=' ')
        
        for neighbor in self.graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited

# Example usage
g = Graph()
for vertex in ['A', 'B', 'C', 'D']:
    g.add_vertex(vertex)
g.add_edge('A', 'B')
g.add_edge('B', 'C')
g.add_edge('C', 'D')
print("BFS traversal starting from vertex A:")
g.bfs('A')
```

Slide 7: Cycle Detection in Graphs

Detecting cycles in a graph is crucial for many applications. This implementation uses a recursive DFS-based approach to identify cycles in the graph, maintaining a set of vertices in the current recursion stack.

```python
def has_cycle(self, vertex, visited=None, rec_stack=None):
    """Detect if graph contains a cycle using DFS"""
    if visited is None:
        visited = set()
    if rec_stack is None:
        rec_stack = set()
    
    visited.add(vertex)
    rec_stack.add(vertex)
    
    for neighbor in self.graph[vertex]:
        if neighbor not in visited:
            if self.has_cycle(neighbor, visited, rec_stack):
                return True
        elif neighbor in rec_stack:
            return True
    
    rec_stack.remove(vertex)
    return False

# Example usage
g = Graph()
for v in ['A', 'B', 'C']:
    g.add_vertex(v)
g.add_edge('A', 'B')
g.add_edge('B', 'C')
g.add_edge('C', 'A')
print(f"Graph has cycle: {g.has_cycle('A')}")
```

Slide 8: Shortest Path Implementation

Implementing Dijkstra's algorithm for finding the shortest path between vertices in a weighted graph. This implementation uses a priority queue to efficiently select the next vertex to process.

```python
from heapq import heappush, heappop
import math

def shortest_path(self, start, end):
    """Find shortest path using Dijkstra's algorithm"""
    distances = {vertex: math.inf for vertex in self.graph}
    distances[start] = 0
    pq = [(0, start)]
    previous = {vertex: None for vertex in self.graph}
    
    while pq:
        current_distance, current_vertex = heappop(pq)
        
        if current_vertex == end:
            path = []
            while current_vertex:
                path.append(current_vertex)
                current_vertex = previous[current_vertex]
            return path[::-1], distances[end]
            
        if current_distance > distances[current_vertex]:
            continue
            
        for neighbor in self.graph[current_vertex]:
            distance = current_distance + 1  # Assuming unweighted graph
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_vertex
                heappush(pq, (distance, neighbor))
                
    return None, math.inf

# Example usage
g = Graph()
vertices = ['A', 'B', 'C', 'D']
for v in vertices:
    g.add_vertex(v)
g.add_edge('A', 'B')
g.add_edge('B', 'C')
g.add_edge('C', 'D')
path, distance = g.shortest_path('A', 'D')
print(f"Shortest path: {path}, Distance: {distance}")
```

Slide 9: Graph Connectivity Analysis

Implementing methods to analyze graph connectivity including finding connected components and checking if the graph is fully connected. This is essential for understanding graph structure and properties.

```python
def find_connected_components(self):
    """Find all connected components in the graph"""
    visited = set()
    components = []
    
    for vertex in self.graph:
        if vertex not in visited:
            component = set()
            self._dfs_component(vertex, visited, component)
            components.append(component)
            
    return components

def _dfs_component(self, vertex, visited, component):
    """Helper function for DFS traversal in component finding"""
    visited.add(vertex)
    component.add(vertex)
    
    for neighbor in self.graph[vertex]:
        if neighbor not in visited:
            self._dfs_component(neighbor, visited, component)

def is_connected(self):
    """Check if the graph is fully connected"""
    if not self.graph:
        return True
    
    start_vertex = next(iter(self.graph))
    visited = self.dfs(start_vertex)
    
    return len(visited) == len(self.graph)

# Example usage
g = Graph()
vertices = ['A', 'B', 'C', 'D', 'E']
for v in vertices:
    g.add_vertex(v)
g.add_edge('A', 'B')
g.add_edge('C', 'D')
components = g.find_connected_components()
print(f"Connected components: {components}")
print(f"Graph is connected: {g.is_connected()}")
```

Slide 10: Real-world Application - Social Network Analysis

Implementing a practical example of using the graph data structure to analyze social networks, including finding influential nodes and community detection.

```python
class SocialNetwork(Graph):
    def find_influencers(self, threshold):
        """Find nodes with high degree centrality"""
        influencers = []
        for vertex in self.graph:
            if len(self.graph[vertex]) >= threshold:
                influencers.append((vertex, len(self.graph[vertex])))
        return sorted(influencers, key=lambda x: x[1], reverse=True)
    
    def calculate_clustering_coefficient(self, vertex):
        """Calculate local clustering coefficient for a vertex"""
        neighbors = self.graph[vertex]
        if len(neighbors) < 2:
            return 0.0
        
        possible_connections = len(neighbors) * (len(neighbors) - 1) / 2
        actual_connections = 0
        
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in self.graph[neighbors[i]]:
                    actual_connections += 1
                    
        return actual_connections / possible_connections

# Example usage
sn = SocialNetwork()
users = ['User1', 'User2', 'User3', 'User4', 'User5']
for user in users:
    sn.add_vertex(user)
sn.add_edge('User1', 'User2')
sn.add_edge('User2', 'User3')
sn.add_edge('User1', 'User3')

print("Influencers:", sn.find_influencers(2))
print("Clustering coefficient for User1:", 
      sn.calculate_clustering_coefficient('User1'))
```

Slide 11: Graph Metrics and Analytics

Advanced graph metrics implementation for analyzing graph properties including density, diameter, and centrality measures. These metrics provide crucial insights into the structure and characteristics of the graph.

```python
import math

class GraphAnalytics(Graph):
    def calculate_density(self):
        """Calculate graph density: ratio of actual edges to possible edges"""
        vertices = len(self.graph)
        if vertices <= 1:
            return 0.0
        
        edges = sum(len(adj) for adj in self.graph.values()) / 2
        possible_edges = vertices * (vertices - 1) / 2
        return edges / possible_edges
    
    def calculate_eccentricity(self, vertex):
        """Calculate eccentricity: maximum shortest path length from vertex"""
        distances = self._bfs_distances(vertex)
        return max(d for d in distances.values() if d != math.inf)
    
    def _bfs_distances(self, start):
        """Helper function to calculate distances using BFS"""
        distances = {v: math.inf for v in self.graph}
        distances[start] = 0
        queue = deque([(start, 0)])
        
        while queue:
            vertex, dist = queue.popleft()
            for neighbor in self.graph[vertex]:
                if distances[neighbor] == math.inf:
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
                    
        return distances

# Example usage
ga = GraphAnalytics()
vertices = ['A', 'B', 'C', 'D']
for v in vertices:
    ga.add_vertex(v)
ga.add_edge('A', 'B')
ga.add_edge('B', 'C')
ga.add_edge('C', 'D')

print(f"Graph density: {ga.calculate_density():.2f}")
print(f"Eccentricity of vertex A: {ga.calculate_eccentricity('A')}")
```

Slide 12: Real-world Application - Route Planning System

Implementation of a route planning system using the graph structure, incorporating weighted edges for distances and additional attributes for road conditions and traffic.

```python
class RoutePlanner(Graph):
    def __init__(self):
        super().__init__()
        self.edge_weights = {}  # Store distances
        self.road_conditions = {}  # Store road condition scores
        
    def add_route(self, start, end, distance, condition):
        """Add a route with distance and road condition"""
        self.add_vertex(start)
        self.add_vertex(end)
        self.add_edge(start, end)
        self.edge_weights[(start, end)] = distance
        self.edge_weights[(end, start)] = distance
        self.road_conditions[(start, end)] = condition
        self.road_conditions[(end, start)] = condition
        
    def find_optimal_route(self, start, end, weight_factor=0.7):
        """Find optimal route considering both distance and road condition"""
        distances = {vertex: math.inf for vertex in self.graph}
        distances[start] = 0
        previous = {vertex: None for vertex in self.graph}
        pq = [(0, start)]
        
        while pq:
            current_cost, current = heappop(pq)
            
            if current == end:
                break
                
            for neighbor in self.graph[current]:
                distance = self.edge_weights[(current, neighbor)]
                condition = self.road_conditions[(current, neighbor)]
                
                # Combined cost considering both distance and road condition
                cost = weight_factor * distance + (1 - weight_factor) * condition
                total_cost = current_cost + cost
                
                if total_cost < distances[neighbor]:
                    distances[neighbor] = total_cost
                    previous[neighbor] = current
                    heappush(pq, (total_cost, neighbor))
                    
        # Reconstruct path
        path = []
        current = end
        while current:
            path.append(current)
            current = previous[current]
        return path[::-1], distances[end]

# Example usage
rp = RoutePlanner()
rp.add_route("CityA", "CityB", 100, 1)  # 100km, perfect condition
rp.add_route("CityB", "CityC", 50, 3)   # 50km, moderate condition
optimal_path, total_cost = rp.find_optimal_route("CityA", "CityC")
print(f"Optimal route: {optimal_path}")
print(f"Total cost: {total_cost:.2f}")
```

Slide 13: Additional Resources

*   Graph Theory Fundamentals and Applications:
    *   [https://arxiv.org/abs/2110.14803](https://arxiv.org/abs/2110.14803)
    *   [https://arxiv.org/abs/2012.14994](https://arxiv.org/abs/2012.14994)
    *   [https://arxiv.org/abs/2107.12741](https://arxiv.org/abs/2107.12741)
*   Useful resources for further learning:
    *   [https://www.geeksforgeeks.org/graph-data-structure-and-algorithms](https://www.geeksforgeeks.org/graph-data-structure-and-algorithms)
    *   [https://www.programiz.com/dsa/graph](https://www.programiz.com/dsa/graph)
    *   [https://algorithms.wtf/graph-algorithms](https://algorithms.wtf/graph-algorithms)
*   Recommended books:
    *   "Algorithm Design" by Kleinberg and Tardos
    *   "Introduction to Algorithms" by CLRS
    *   "Graph Theory and Its Applications" by Jonathan L. Gross

