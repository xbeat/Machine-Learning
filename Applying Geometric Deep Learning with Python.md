## Applying Geometric Deep Learning with Python
Slide 1: Understanding Geometric Deep Learning

Geometric Deep Learning (GDL) is a rapidly evolving field that applies deep learning techniques to non-Euclidean data structures such as graphs and manifolds. The biggest challenge in applying GDL using Python lies in effectively representing and processing complex geometric structures while leveraging the power of neural networks.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple graph
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=16)
plt.title("A Simple Graph Structure")
plt.show()
```

Slide 2: Graph Representation

One of the primary challenges in GDL is representing graph structures in a format suitable for neural networks. We often use adjacency matrices or edge lists to encode graph topology.

```python
import numpy as np

# Adjacency matrix representation
adj_matrix = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 0]
])

# Edge list representation
edge_list = [(0, 1), (0, 2), (1, 2), (2, 3)]

print("Adjacency Matrix:")
print(adj_matrix)
print("\nEdge List:")
print(edge_list)
```

Slide 3: Node Feature Encoding

Another challenge is encoding node features effectively. We need to represent node attributes in a way that preserves their meaning and allows for meaningful computations.

```python
import torch

# Example node features (4 nodes, 3 feature dimensions)
node_features = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [1.0, 1.1, 1.2]
], dtype=torch.float32)

print("Node Features:")
print(node_features)
```

Slide 4: Graph Convolution Operations

Implementing graph convolution operations is a key challenge in GDL. These operations need to aggregate information from neighboring nodes while respecting the graph structure.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.matmul(adj, support)
        return F.relu(output)

# Example usage
conv = GraphConvLayer(3, 5)
x = node_features
adj = torch.tensor(adj_matrix, dtype=torch.float32)
output = conv(x, adj)

print("Graph Convolution Output:")
print(output)
```

Slide 5: Pooling on Graphs

Pooling operations, which are straightforward in traditional neural networks, become challenging on graph-structured data. We need to devise methods to aggregate node information hierarchically.

```python
def graph_max_pool(x, cluster):
    # x: node features, cluster: cluster assignments
    max_vals = torch.zeros(cluster.max() + 1, x.size(1))
    for i in range(cluster.max() + 1):
        mask = (cluster == i)
        if mask.sum() > 0:
            max_vals[i] = torch.max(x[mask], dim=0)[0]
    return max_vals

# Example usage
cluster = torch.tensor([0, 0, 1, 1])
pooled = graph_max_pool(node_features, cluster)

print("Pooled Features:")
print(pooled)
```

Slide 6: Message Passing Framework

Implementing an efficient message passing framework is crucial for GDL. This involves propagating information along edges of the graph.

```python
def message_passing(x, edge_index):
    # x: node features, edge_index: list of edges
    src, dst = edge_index
    messages = x[src]
    aggregated = torch.zeros_like(x)
    for i in range(len(dst)):
        aggregated[dst[i]] += messages[i]
    return aggregated

# Example usage
edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]])
messages = message_passing(node_features, edge_index)

print("Aggregated Messages:")
print(messages)
```

Slide 7: Handling Dynamic Graphs

Many real-world applications involve graphs that change over time. Adapting GDL models to handle dynamic graphs is a significant challenge.

```python
class DynamicGraphNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(DynamicGraphNN, self).__init__()
        self.conv1 = GraphConvLayer(in_features, hidden_features)
        self.conv2 = GraphConvLayer(hidden_features, out_features)
    
    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = self.conv2(x, adj)
        return x

# Simulate a dynamic graph
def update_graph(adj, prob=0.1):
    mask = torch.rand(adj.shape) < prob
    return adj.clone().masked_fill_(mask, 1 - adj)

model = DynamicGraphNN(3, 10, 5)
adj = torch.tensor(adj_matrix, dtype=torch.float32)

for t in range(3):
    adj = update_graph(adj)
    output = model(node_features, adj)
    print(f"Time step {t}, Output:")
    print(output)
```

Slide 8: Scalability Issues

As graphs grow larger, scalability becomes a major challenge. Efficient implementations are crucial for handling large-scale graph data.

```python
import time

def benchmark_gcn(num_nodes, num_features):
    x = torch.randn(num_nodes, num_features)
    adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    model = GraphConvLayer(num_features, num_features)
    
    start_time = time.time()
    output = model(x, adj)
    end_time = time.time()
    
    return end_time - start_time

sizes = [100, 1000, 10000]
times = [benchmark_gcn(size, 64) for size in sizes]

print("Execution times for different graph sizes:")
for size, t in zip(sizes, times):
    print(f"{size} nodes: {t:.4f} seconds")
```

Slide 9: Incorporating Edge Features

Many graph problems require considering edge features alongside node features. Integrating edge information into GDL models presents additional complexity.

```python
class EdgeFeatureGCN(nn.Module):
    def __init__(self, node_features, edge_features, out_features):
        super(EdgeFeatureGCN, self).__init__()
        self.node_transform = nn.Linear(node_features, out_features)
        self.edge_transform = nn.Linear(edge_features, out_features)
    
    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        node_out = self.node_transform(x)
        edge_out = self.edge_transform(edge_attr)
        
        messages = node_out[src] * edge_out
        aggregated = torch.zeros_like(node_out)
        for i in range(len(dst)):
            aggregated[dst[i]] += messages[i]
        
        return F.relu(aggregated)

# Example usage
node_features = torch.randn(4, 3)
edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]])
edge_attr = torch.randn(4, 2)

model = EdgeFeatureGCN(3, 2, 5)
output = model(node_features, edge_index, edge_attr)

print("Output with edge features:")
print(output)
```

Slide 10: Heterogeneous Graphs

Real-world graphs often contain multiple types of nodes and edges. Handling heterogeneous graphs adds another layer of complexity to GDL models.

```python
class HeteroGNN(nn.Module):
    def __init__(self, node_types, edge_types, in_features, out_features):
        super(HeteroGNN, self).__init__()
        self.convs = nn.ModuleDict({
            f"{src}_{rel}_{dst}": GraphConvLayer(in_features, out_features)
            for (src, rel, dst) in edge_types
        })
    
    def forward(self, x_dict, edge_index_dict):
        out_dict = {}
        for node_type, x in x_dict.items():
            out = torch.zeros(x.size(0), self.out_features)
            for (src, rel, dst), conv in self.convs.items():
                if dst == node_type:
                    edge_index = edge_index_dict[(src, rel, dst)]
                    adj = to_dense_adj(edge_index, max_num_nodes=x.size(0))
                    out += conv(x_dict[src], adj)
            out_dict[node_type] = F.relu(out)
        return out_dict

# Example usage (simplified)
node_types = ['user', 'item']
edge_types = [('user', 'rates', 'item'), ('item', 'rated_by', 'user')]
x_dict = {
    'user': torch.randn(3, 5),
    'item': torch.randn(4, 5)
}
edge_index_dict = {
    ('user', 'rates', 'item'): torch.tensor([[0, 1, 2], [0, 1, 2]]),
    ('item', 'rated_by', 'user'): torch.tensor([[0, 1, 2], [0, 1, 2]])
}

model = HeteroGNN(node_types, edge_types, 5, 10)
output = model(x_dict, edge_index_dict)

print("Heterogeneous Graph Output:")
for node_type, out in output.items():
    print(f"{node_type}:", out)
```

Slide 11: Handling Graph Isomorphism

Distinguishing between non-isomorphic graphs is a fundamental challenge in GDL. Developing models that are sensitive to structural differences is crucial.

```python
import torch.nn.functional as F

def graph_fingerprint(adj):
    eigenvalues, _ = torch.linalg.eig(adj)
    return torch.sort(eigenvalues.real)[0]

def are_graphs_isomorphic(adj1, adj2, tolerance=1e-6):
    fp1 = graph_fingerprint(adj1)
    fp2 = graph_fingerprint(adj2)
    return torch.allclose(fp1, fp2, atol=tolerance)

# Example usage
adj1 = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.float32)
adj2 = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.float32)
adj3 = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)

print("Graph 1 and 2 isomorphic:", are_graphs_isomorphic(adj1, adj2))
print("Graph 1 and 3 isomorphic:", are_graphs_isomorphic(adj1, adj3))
```

Slide 12: Real-life Example: Social Network Analysis

GDL can be applied to analyze social networks, identifying influential users and community structures. This example demonstrates a simple implementation of node importance calculation.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a social network graph
G = nx.karate_club_graph()

# Calculate node importance (using degree centrality)
centrality = nx.degree_centrality(G)

# Visualize the graph with node sizes proportional to importance
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color='lightblue', 
        node_size=[v * 3000 for v in centrality.values()],
        with_labels=True, font_size=10)
plt.title("Social Network Analysis: Node Importance")
plt.show()

# Print top 5 most important nodes
top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:5]
print("Top 5 most important nodes:", top_nodes)
```

Slide 13: Real-life Example: Molecular Property Prediction

GDL is widely used in chemistry for predicting molecular properties. This example shows how to represent a molecule as a graph and perform a simple property prediction.

```python
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

class MoleculeGCN(torch.nn.Module):
    def __init__(self):
        super(MoleculeGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x.mean(dim=0)

# Convert molecule to graph
def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)
    
    # Node features (atomic number)
    x = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=torch.float)
    
    # Edge indices
    edge_index = torch.tensor([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] 
                               for bond in mol.GetBonds()]).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

# Example usage
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
data = mol_to_graph(smiles)

num_node_features = data.num_node_features
model = MoleculeGCN()
output = model(data)

print(f"Predicted property for {smiles}: {output.item():.4f}")
```

Slide 14: Additional Resources

For those interested in diving deeper into Geometric Deep Learning, here are some valuable resources:

1. "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges" by Michael M. Bronstein et al. (2021) ArXiv: [https://arxiv.org/abs/2104.13478](https://arxiv.org/abs/2104.13478)
2. "Graph Representation Learning" by William L. Hamilton (2020) Book website: [https://www.cs.mcgill.ca/~wlh/grl\_book/](https://www.cs.mcgill.ca/~wlh/grl_book/)
3. "A Comprehensive Survey on Graph Neural Networks" by Zonghan Wu et al. (2020) ArXiv: [https://arxiv.org/abs/1901.00596](https://arxiv.org/abs/1901.00596)

These resources provide in-depth coverage of GDL concepts, algorithms, and applications, offering valuable insights for both beginners and advanced practitioners in the field.

