## Graph Neural Networks in Python
Slide 1: Introduction to Graph Neural Networks (GNNs)

Graph Neural Networks are a class of deep learning models designed to work with graph-structured data. They can capture complex relationships between entities in a graph, making them ideal for tasks like social network analysis, recommendation systems, and molecular property prediction.

```python
import networkx as nx
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Create a simple graph
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

# Convert to PyTorch Geometric data
edge_index = torch.tensor(list(G.edges())).t().contiguous()
x = torch.randn(G.number_of_nodes(), 10)  # Node features
data = Data(x=x, edge_index=edge_index)

# Define a simple GCN layer
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(10, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN()
output = model(data)
print(output.shape)  # Shape: [num_nodes, 2]
```

Slide 2: Graph Representation

Graphs consist of nodes (vertices) and edges (connections between nodes). In GNNs, we represent graphs using adjacency matrices and node feature matrices. The adjacency matrix describes the graph structure, while the node feature matrix contains information about each node.

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple graph
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

# Generate adjacency matrix
adj_matrix = nx.adjacency_matrix(G).toarray()

# Generate node feature matrix (random features for illustration)
node_features = np.random.rand(G.number_of_nodes(), 3)

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=16)
plt.title("Example Graph")
plt.show()

print("Adjacency Matrix:")
print(adj_matrix)
print("\nNode Feature Matrix:")
print(node_features)
```

Slide 3: Message Passing in GNNs

Message passing is the core operation in GNNs. It involves aggregating information from neighboring nodes to update the representation of each node. This process allows the model to capture both local and global graph structures.

```python
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class SimpleGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SimpleGNNLayer, self).__init__(aggr='add')  # "Add" aggregation
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Add self-loops to the edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Start propagating messages
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # Transform node features
        x_j = self.linear(x_j)
        # Normalize node features
        return norm.view(-1, 1) * x_j

# Usage
layer = SimpleGNNLayer(10, 16)
x = torch.randn(4, 10)  # 4 nodes, 10 features each
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
output = layer(x, edge_index)
print(output.shape)  # Shape: [4, 16]
```

Slide 4: Graph Convolutional Networks (GCNs)

Graph Convolutional Networks are a popular type of GNN that generalizes the convolution operation to graph-structured data. GCNs update node representations by aggregating information from neighboring nodes, effectively capturing local graph structure.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Load a dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
```

Slide 5: Graph Attention Networks (GATs)

Graph Attention Networks introduce attention mechanisms to GNNs, allowing the model to assign different importance to different neighbors when aggregating information. This enables GATs to capture more complex relationships in the graph.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dataset.num_node_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GAT()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
```

Slide 6: Node Embedding and Classification

One common task in graph learning is node classification, where we predict labels for nodes in a graph. GNNs can learn node embeddings that capture both node features and graph structure, which can then be used for classification.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

class NodeClassifier(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(NodeClassifier, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = NodeClassifier(dataset.num_node_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

for epoch in range(200):
    loss = train()
    if epoch % 10 == 0:
        test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}')
```

Slide 7: Graph Pooling and Readout

Graph pooling and readout operations are crucial for tasks that require graph-level representations, such as graph classification. These operations aggregate node features to create a fixed-size representation of the entire graph.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class GraphClassifier(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GraphClassifier, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

model = GraphClassifier(dataset.num_features, 64, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(dataset)

for epoch in range(50):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
```

Slide 8: Graph Generation with GNNs

Graph generation is an emerging application of GNNs, where the goal is to create new graphs that share properties with a given set of graphs. This can be useful in drug discovery, molecule design, and synthetic data generation.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.data import Data, Batch

class GraphVAE(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim):
        super(GraphVAE, self).__init__()
        self.encoder = torch.nn.Sequential(
            GCNConv(num_features, hidden_dim),
            torch.nn.ReLU(),
            GCNConv(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar = torch.nn.Linear(hidden_dim, latent_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_features)
        )

    def encode(self, x, edge_index):
        h = self.encoder(x, edge_index)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Example usage
num_nodes = 10
num_features = 3
model = GraphVAE(num_features, 16, 2)
x = torch.randn(num_nodes, num_features)
edge_index = torch.randint(0, num_nodes, (2, 20))
data = Data(x=x, edge_index=edge_index)

reconstructed, mu, logvar = model(data)
print(f"Original shape: {x.shape}, Reconstructed shape: {reconstructed.shape}")
```

Slide 9: Graph Neural Networks for Recommender Systems

GNNs can be applied to recommender systems by modeling user-item interactions as a bipartite graph. This approach captures complex relationships between users and items, leading to more accurate recommendations.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import HeteroData

class GNNRecommender(torch.nn.Module):
    def __init__(self, num_users, num_items, hidden_channels):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(num_users, hidden_channels)
        self.item_embedding = torch.nn.Embedding(num_items, hidden_channels)
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        x_user = self.user_embedding(x_dict['user'])
        x_item = self.item_embedding(x_dict['item'])
        
        x_user = self.conv1((x_user, x_item), edge_index_dict[('user', 'to', 'item')])
        x_item = self.conv1((x_item, x_user), edge_index_dict[('item', 'to', 'user')])
        
        x_user = F.relu(x_user)
        x_item = F.relu(x_item)
        
        x_user = self.conv2((x_user, x_item), edge_index_dict[('user', 'to', 'item')])
        x_item = self.conv2((x_item, x_user), edge_index_dict[('item', 'to', 'user')])
        
        return x_user, x_item

# Example usage
num_users, num_items = 1000, 5000
model = GNNRecommender(num_users, num_items, hidden_channels=64)

# Create a heterogeneous graph
data = HeteroData()
data['user'].x = torch.arange(num_users)
data['item'].x = torch.arange(num_items)
data['user', 'to', 'item'].edge_index = torch.randint(0, (num_users, num_items), (2, 10000))
data['item', 'to', 'user'].edge_index = data['user', 'to', 'item'].edge_index[[1, 0]]

# Forward pass
user_emb, item_emb = model(data.x_dict, data.edge_index_dict)
print(f"User embeddings shape: {user_emb.shape}")
print(f"Item embeddings shape: {item_emb.shape}")
```

Slide 10: Temporal Graph Neural Networks

Temporal Graph Neural Networks (TGNNs) extend GNNs to handle dynamic graphs that change over time. These models can capture both spatial and temporal dependencies in evolving graph structures.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TemporalGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lstm = torch.nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        self.linear = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch_size, seq_len):
        # Apply GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Reshape for LSTM
        x = x.view(batch_size, seq_len, -1)
        
        # Apply LSTM layer
        x, _ = self.lstm(x)
        
        # Take the last time step
        x = x[:, -1, :]
        
        # Final classification
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

# Example usage
num_features, hidden_channels, num_classes = 10, 64, 5
model = TemporalGNN(num_features, hidden_channels, num_classes)

# Simulated data
batch_size, seq_len, num_nodes = 32, 10, 100
x = torch.randn(batch_size * seq_len * num_nodes, num_features)
edge_index = torch.randint(0, num_nodes, (2, 5000))

output = model(x, edge_index, batch_size, seq_len)
print(f"Output shape: {output.shape}")
```

Slide 11: Graph Neural Networks for Natural Language Processing

GNNs can be applied to various NLP tasks by representing text as graph structures. This approach can capture complex linguistic relationships and improve performance on tasks like text classification and machine translation.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class TextGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)  # Graph-level pooling
        x = self.linear(x)
        return F.log_softmax(x, dim=0)

# Example usage for text classification
num_features, hidden_channels, num_classes = 300, 64, 5
model = TextGNN(num_features, hidden_channels, num_classes)

# Simulated text data (word embeddings and word co-occurrence graph)
num_words = 50
x = torch.randn(num_words, num_features)  # Word embeddings
edge_index = torch.randint(0, num_words, (2, 200))  # Word co-occurrence

data = Data(x=x, edge_index=edge_index)
output = model(data.x, data.edge_index)
print(f"Output shape: {output.shape}")
```

Slide 12: Graph Neural Networks for Computer Vision

GNNs can enhance computer vision tasks by modeling images as graphs. This approach can capture spatial relationships between image regions and improve performance on tasks like image classification and object detection.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class ImageGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # Pooling
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

# Example usage for image classification
num_features, hidden_channels, num_classes = 512, 64, 10
model = ImageGNN(num_features, hidden_channels, num_classes)

# Simulated image data (superpixel features and adjacency)
num_superpixels = 100
batch_size = 32
x = torch.randn(num_superpixels * batch_size, num_features)
edge_index = torch.randint(0, num_superpixels, (2, 500 * batch_size))
batch = torch.repeat_interleave(torch.arange(batch_size), num_superpixels)

output = model(x, edge_index, batch)
print(f"Output shape: {output.shape}")
```

Slide 13: Challenges and Future Directions in Graph Neural Networks

Graph Neural Networks face several challenges and opportunities for future research:

1. Scalability: Developing methods to efficiently process large-scale graphs with billions of nodes and edges.
2. Expressiveness: Designing more powerful GNN architectures that can capture complex graph structures and long-range dependencies.
3. Heterogeneous graphs: Improving techniques for handling graphs with multiple node and edge types.
4. Dynamic graphs: Enhancing models to better handle temporal and evolving graph structures.
5. Interpretability: Developing methods to explain GNN predictions and understand their decision-making process.
6. Generalization: Improving the ability of GNNs to generalize to unseen graph structures and sizes.
7. Robustness: Enhancing the resilience of GNNs against adversarial attacks and noisy graph data.
8. Theoretical foundations: Strengthening the theoretical understanding of GNNs and their capabilities.
9. Applications: Exploring new domains and use cases for GNNs, such as drug discovery, social network analysis, and financial fraud detection.
10. Integration with other AI techniques: Combining GNNs with other machine learning approaches like reinforcement learning and natural language processing.

```python
# Pseudocode for a future GNN architecture addressing some challenges

class AdvancedGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_layer = GraphAttentionLayer()
        self.temporal_layer = TemporalConvLayer()
        self.heterogeneous_layer = HeterogeneousMessagePassing()
        self.interpretable_layer = InterpretableGNNLayer()

    def forward(self, x, edge_index, edge_type, time_data):
        # Handle heterogeneous data
        x = self.heterogeneous_layer(x, edge_index, edge_type)
        
        # Incorporate temporal information
        x = self.temporal_layer(x, time_data)
        
        # Apply attention mechanism for better expressiveness
        x = self.attention_layer(x, edge_index)
        
        # Generate interpretable output
        x, explanations = self.interpretable_layer(x, edge_index)
        
        return x, explanations
```

Slide 14: Additional Resources

For those interested in delving deeper into Graph Neural Networks, here are some valuable resources:

1. "Graph Representation Learning" by William L. Hamilton ArXiv: [https://arxiv.org/abs/1709.05584](https://arxiv.org/abs/1709.05584)
2. "A Comprehensive Survey on Graph Neural Networks" by Z. Wu et al. ArXiv: [https://arxiv.org/abs/1901.00596](https://arxiv.org/abs/1901.00596)
3. "Graph Neural Networks: A Review of Methods and Applications" by J. Zhou et al. ArXiv: [https://arxiv.org/abs/1812.08434](https://arxiv.org/abs/1812.08434)
4. PyTorch Geometric Documentation [https://pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/)
5. Deep Graph Library (DGL) [https://www.dgl.ai/](https://www.dgl.ai/)

These resources provide in-depth explanations, theoretical foundations, and practical implementations of various GNN architectures and applications.

