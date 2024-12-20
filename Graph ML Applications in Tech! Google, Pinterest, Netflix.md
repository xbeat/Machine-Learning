## Graph ML Applications in Tech! Google, Pinterest, Netflix
Slide 1: Graph Machine Learning in Tech Industry

Graph Machine Learning (ML) has become a critical tool for many major tech companies, offering powerful solutions for complex problems. This presentation explores how different companies leverage graph ML techniques to enhance their products and services.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a sample graph
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)])

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=16)
plt.title("Sample Graph Structure")
plt.show()
```

Slide 2: Introduction to Graph Machine Learning

Graph ML is a subset of machine learning that focuses on analyzing and learning from graph-structured data. It combines traditional ML techniques with graph theory to extract insights from interconnected data points.

```python
import torch
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Load a sample dataset (Cora)
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Define a simple Graph Convolutional Network
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Initialize the model
model = GCN()
print(model)
```

Slide 3: Google Maps: ETA Prediction with Graph ML

Google Maps uses graph ML for Estimated Time of Arrival (ETA) prediction. The road network is represented as a graph, where intersections are nodes and roads are edges. Graph neural networks are employed to process this structure and predict travel times accurately.

```python
import networkx as nx
import random

# Create a simple road network graph
G = nx.Graph()
G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'D'), ('B', 'D')])

# Assign random travel times to edges
for (u, v) in G.edges():
    G[u][v]['time'] = random.randint(5, 30)

# Function to calculate ETA
def calculate_eta(start, end):
    path = nx.shortest_path(G, start, end, weight='time')
    eta = sum(G[path[i]][path[i+1]]['time'] for i in range(len(path)-1))
    return path, eta

# Example usage
start, end = 'A', 'D'
path, eta = calculate_eta(start, end)
print(f"Estimated route: {' -> '.join(path)}")
print(f"ETA: {eta} minutes")
```

Slide 4: Pinterest: PinSage for Recommendations

Pinterest uses a graph ML model called PinSage for content recommendations. It treats pins, boards, and users as nodes in a graph, with edges representing interactions. This approach allows for more contextual and personalized recommendations.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Simulating PinSage embeddings for pins
num_pins = 1000
embedding_dim = 128
pin_embeddings = np.random.rand(num_pins, embedding_dim)

# Function to find similar pins
def find_similar_pins(pin_id, top_k=5):
    query_embedding = pin_embeddings[pin_id].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, pin_embeddings)[0]
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
    return [(idx, similarities[idx]) for idx in similar_indices]

# Example usage
pin_id = 42
similar_pins = find_similar_pins(pin_id)
print(f"Pins similar to Pin {pin_id}:")
for idx, similarity in similar_pins:
    print(f"Pin {idx}: Similarity = {similarity:.4f}")
```

Slide 5: Netflix: SemanticGNN for Content Recommendations

Netflix employs SemanticGNN, a graph neural network-based model, for content recommendations. This approach considers the complex relationships between users, movies, and various metadata to provide more accurate and diverse recommendations.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SemanticGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(SemanticGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Example usage (assuming data is prepared)
num_features = 100
hidden_dim = 64
num_classes = 10
model = SemanticGNN(num_features, hidden_dim, num_classes)
print(model)
```

Slide 6: Spotify: HGNNs for Audiobook Recommendations

Spotify utilizes Heterogeneous Graph Neural Networks (HGNNs) for audiobook recommendations. This approach allows them to model complex relationships between users, audiobooks, authors, narrators, and genres in a single graph structure.

```python
import torch
from torch_geometric.nn import HeteroConv, SAGEConv, to_hetero

class HGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HeteroConv({
            ('user', 'listens', 'audiobook'): SAGEConv((-1, -1), hidden_channels),
            ('audiobook', 'written_by', 'author'): SAGEConv((-1, -1), hidden_channels),
            ('audiobook', 'narrated_by', 'narrator'): SAGEConv((-1, -1), hidden_channels),
        })
        self.conv2 = HeteroConv({
            ('user', 'listens', 'audiobook'): SAGEConv((-1, -1), out_channels),
            ('audiobook', 'written_by', 'author'): SAGEConv((-1, -1), out_channels),
            ('audiobook', 'narrated_by', 'narrator'): SAGEConv((-1, -1), out_channels),
        })

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

# Example usage (assuming data is prepared)
model = HGNN(hidden_channels=64, out_channels=32)
print(model)
```

Slide 7: Uber Eats: GraphSAGE Variant for Dish and Restaurant Suggestions

Uber Eats employs a variant of GraphSAGE, a graph neural network architecture, to suggest dishes and restaurants. This model leverages the interconnected nature of users, restaurants, dishes, and cuisines to provide personalized recommendations.

```python
import torch
from torch_geometric.nn import SAGEConv

class UberEatsGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(UberEatsGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x

# Example usage
in_channels = 64  # Number of input features
hidden_channels = 32
out_channels = 16  # Number of output features
model = UberEatsGraphSAGE(in_channels, hidden_channels, out_channels)

# Simulate input data
num_nodes = 1000
x = torch.randn(num_nodes, in_channels)
edge_index = torch.randint(0, num_nodes, (2, 5000))

# Forward pass
output = model(x, edge_index)
print(f"Output shape: {output.shape}")
```

Slide 8: Graph ML vs. Traditional Deep Learning

While traditional deep learning has been dominant, graph ML is becoming increasingly important due to its ability to model complex relationships. Graph ML can capture structural information that traditional methods might miss, making it particularly useful for networked data.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Traditional MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Graph Convolutional Network
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# Example usage
input_dim, hidden_dim, output_dim = 64, 32, 10
mlp = MLP(input_dim, hidden_dim, output_dim)
gcn = GCN(input_dim, hidden_dim, output_dim)

print("MLP:", mlp)
print("\nGCN:", gcn)
```

Slide 9: Advantages of Graph ML

Graph ML offers several advantages over traditional ML methods, including the ability to capture relational information, handle heterogeneous data types, and scale to large datasets. These properties make it particularly suited for complex real-world problems.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a heterogeneous graph
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4], bipartite=0)  # Users
G.add_nodes_from(['A', 'B', 'C'], bipartite=1)  # Items

G.add_edges_from([(1, 'A'), (1, 'B'), (2, 'B'), (3, 'C'), (4, 'A'), (4, 'C')])

# Visualize the graph
pos = nx.spring_layout(G)
color_map = ['lightblue' if node in [1, 2, 3, 4] else 'lightgreen' for node in G.nodes()]
nx.draw(G, pos, node_color=color_map, with_labels=True, font_weight='bold')
plt.title("Heterogeneous Graph: Users and Items")
plt.show()

# Calculate some graph properties
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average clustering coefficient: {nx.average_clustering(G):.4f}")
```

Slide 10: Real-Life Example: Social Network Analysis

Graph ML is extensively used in social network analysis to understand user behavior, detect communities, and identify influential users. This example demonstrates a simple community detection algorithm using graph clustering.

```python
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt

# Create a sample social network
G = nx.karate_club_graph()

# Detect communities
partition = community_louvain.best_partition(G)

# Visualize the network with communities
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_size=700, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos)
plt.title("Social Network Community Detection")
plt.axis('off')
plt.show()

# Print some statistics
print(f"Number of communities: {len(set(partition.values()))}")
print(f"Modularity: {community_louvain.modularity(partition, G):.4f}")
```

Slide 11: Real-Life Example: Fraud Detection

Graph ML is highly effective in fraud detection by analyzing relationships between entities and identifying suspicious patterns. This example shows a simple anomaly detection technique using graph centrality measures.

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Create a sample transaction network
G = nx.random_geometric_graph(100, 0.125)

# Calculate betweenness centrality
centrality = nx.betweenness_centrality(G)

# Identify potential fraudulent nodes (high centrality)
threshold = np.percentile(list(centrality.values()), 95)
fraudulent_nodes = [node for node, cent in centrality.items() if cent > threshold]

# Visualize the network
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue')
nx.draw_networkx_nodes(G, pos, nodelist=fraudulent_nodes, node_color='red', node_size=300)
nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.title("Fraud Detection in Transaction Network")
plt.axis('off')
plt.show()

print(f"Number of potential fraudulent nodes: {len(fraudulent_nodes)}")
```

Slide 12: Challenges in Graph ML

Despite its advantages, graph ML faces several challenges, including scalability issues with large graphs, handling dynamic graphs, and interpreting complex models. Ongoing research aims to address these limitations and improve the applicability of graph ML techniques.

```python
import networkx as nx
import time
import matplotlib.pyplot as plt

def measure_time(func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()
    return end - start

# Generate graphs of increasing size
sizes = [100, 1000, 10000, 100000]
times = []

for size in sizes:
    G = nx.erdos_renyi_graph(size, 0.01)
    t = measure_time(nx.pagerank, G)
    times.append(t)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sizes, times, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Nodes')
plt.ylabel('Computation Time (seconds)')
plt.title('Scalability Challenge: PageRank Computation Time')
plt.grid(True)
plt.show()

print(f"Time for 100,000 nodes: {times[-1]:.2f} seconds")
```

Slide 13: Future Directions in Graph ML

The field of graph ML is rapidly evolving, with new techniques and applications emerging. Future directions include improving model interpretability, developing more efficient algorithms for large-scale graphs, and exploring novel applications in areas such as drug discovery and climate science.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a graph representing potential future directions
G = nx.Graph()
G.add_edges_from([
    ('Graph ML', 'Interpretability'),
    ('Graph ML', 'Scalability'),
    ('Graph ML', 'Dynamic Graphs'),
    ('Graph ML', 'Novel Applications'),
    ('Novel Applications', 'Drug Discovery'),
    ('Novel Applications', 'Climate Science'),
    ('Novel Applications', 'Quantum Computing'),
])

# Visualize the graph
pos = nx.spring_layout(G, k=0.5, iterations=50)
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=3000, font_size=10, font_weight='bold')
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title("Future Directions in Graph ML", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()
```

Slide 14: Conclusion: The Growing Importance of Graph ML

As we've seen, graph ML is becoming increasingly critical in various domains, often outperforming traditional deep learning approaches for certain tasks. Its ability to capture complex relationships and structural information makes it invaluable for many real-world applications.

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulated data for illustration
years = np.arange(2015, 2026)
traditional_ml = 10 + 2 * (years - 2015)
graph_ml = 2 + 3 * (years - 2015)**1.5

plt.figure(figsize=(12, 6))
plt.plot(years, traditional_ml, label='Traditional ML', marker='o')
plt.plot(years, graph_ml, label='Graph ML', marker='s')
plt.xlabel('Year')
plt.ylabel('Impact Score (arbitrary units)')
plt.title('Projected Impact: Traditional ML vs Graph ML')
plt.legend()
plt.grid(True)
plt.show()

print(f"Projected Graph ML impact in 2025: {graph_ml[-1]:.2f}")
print(f"Projected Traditional ML impact in 2025: {traditional_ml[-1]:.2f}")
```

Slide 15: Additional Resources

For those interested in diving deeper into graph machine learning, here are some valuable resources:

1. "Graph Representation Learning" by William L. Hamilton (2020) ArXiv: [https://arxiv.org/abs/1709.05584](https://arxiv.org/abs/1709.05584)
2. "Graph Neural Networks: A Review of Methods and Applications" by Jie Zhou et al. (2018) ArXiv: [https://arxiv.org/abs/1812.08434](https://arxiv.org/abs/1812.08434)
3. "A Comprehensive Survey on Graph Neural Networks" by Zonghan Wu et al. (2019) ArXiv: [https://arxiv.org/abs/1901.00596](https://arxiv.org/abs/1901.00596)
4. PyTorch Geometric documentation: [https://pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/)
5. NetworkX documentation: [https://networkx.org/documentation/stable/](https://networkx.org/documentation/stable/)

These resources provide in-depth coverage of graph ML techniques, algorithms, and applications, suitable for both beginners and advanced practitioners in the field.

