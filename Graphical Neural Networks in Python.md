## Graphical Neural Networks in Python:
Slide 1: Introduction to Graphical Neural Networks

Graphical Neural Networks (GNNs) are a powerful class of deep learning models designed to work with graph-structured data. They leverage the relationships between entities in a graph to perform tasks such as node classification, link prediction, and graph classification. GNNs have applications in various fields, including social network analysis, recommendation systems, and molecular property prediction.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple graph
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])

# Visualize the graph
nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=16)
plt.title("A Simple Graph")
plt.show()
```

Slide 2: Graph Representation in Python

To work with GNNs, we first need to represent graphs in Python. The NetworkX library provides efficient tools for graph manipulation and analysis. Here's an example of creating a graph and adding nodes and edges:

```python
import networkx as nx

# Create an undirected graph
G = nx.Graph()

# Add nodes
G.add_node(1, features=[0.1, 0.2, 0.3])
G.add_node(2, features=[0.2, 0.3, 0.4])
G.add_node(3, features=[0.3, 0.4, 0.5])

# Add edges
G.add_edge(1, 2, weight=0.5)
G.add_edge(1, 3, weight=0.7)
G.add_edge(2, 3, weight=0.9)

# Access node features and edge weights
print(G.nodes[1]['features'])
print(G[1][2]['weight'])
```

Slide 3: Node Feature Extraction

In GNNs, each node is associated with a feature vector. These features can be extracted from the graph structure or provided as input. Here's an example of extracting degree centrality as a node feature:

```python
import networkx as nx
import numpy as np

def extract_node_features(G):
    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(G)
    
    # Create feature matrix
    num_nodes = G.number_of_nodes()
    feature_matrix = np.zeros((num_nodes, 1))
    
    for node, centrality in degree_centrality.items():
        feature_matrix[node] = centrality
    
    return feature_matrix

# Create a sample graph
G = nx.karate_club_graph()

# Extract node features
node_features = extract_node_features(G)
print("Node features shape:", node_features.shape)
print("First 5 node features:", node_features[:5].flatten())
```

Slide 4: Graph Convolutional Layer

The core component of many GNNs is the graph convolutional layer. It aggregates information from neighboring nodes to update the representation of each node. Here's a simple implementation of a graph convolutional layer:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        # x: Node feature matrix (N x F)
        # adj: Adjacency matrix (N x N)
        
        # Perform graph convolution
        support = self.linear(x)
        output = torch.matmul(adj, support)
        
        return F.relu(output)

# Example usage
in_features, out_features, num_nodes = 10, 16, 5
gcn_layer = GraphConvLayer(in_features, out_features)
x = torch.randn(num_nodes, in_features)
adj = torch.randn(num_nodes, num_nodes)

output = gcn_layer(x, adj)
print("Output shape:", output.shape)
```

Slide 5: Message Passing in GNNs

Message passing is a fundamental concept in GNNs. Nodes exchange information with their neighbors to update their representations. Here's a simple implementation of message passing:

```python
import torch
import torch.nn as nn

class MessagePassing(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MessagePassing, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index):
        # x: Node features (N x F)
        # edge_index: Graph connectivity (2 x E)
        
        row, col = edge_index
        out = torch.cat([x[row], x[col]], dim=1)
        return self.mlp(out)

# Example usage
in_channels, out_channels, num_nodes = 16, 32, 5
mp_layer = MessagePassing(in_channels, out_channels)
x = torch.randn(num_nodes, in_channels)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                           [1, 0, 2, 1, 3, 2, 4, 3]])

output = mp_layer(x, edge_index)
print("Output shape:", output.shape)
```

Slide 6: Graph Attention Mechanism

Graph Attention Networks (GATs) introduce attention mechanisms to GNNs, allowing nodes to weigh the importance of their neighbors' features. Here's a simplified implementation of a graph attention layer:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

# Example usage
in_features, out_features, num_nodes = 16, 32, 5
gat_layer = GraphAttentionLayer(in_features, out_features, dropout=0.6, alpha=0.2)
h = torch.randn(num_nodes, in_features)
adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()

output = gat_layer(h, adj)
print("Output shape:", output.shape)
```

Slide 7: Graph Pooling

Graph pooling is used to reduce the size of graph representations and capture hierarchical structure. Here's an implementation of a simple global mean pooling layer:

```python
import torch
import torch.nn as nn

class GlobalMeanPooling(nn.Module):
    def forward(self, x, batch):
        # x: Node features (N x F)
        # batch: Batch assignment for each node (N)
        
        batch_size = batch.max().item() + 1
        output = torch.zeros(batch_size, x.size(1), device=x.device)
        
        for i in range(batch_size):
            mask = (batch == i)
            output[i] = x[mask].mean(dim=0)
        
        return output

# Example usage
num_nodes, num_features, batch_size = 12, 16, 3
x = torch.randn(num_nodes, num_features)
batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

pooling = GlobalMeanPooling()
output = pooling(x, batch)
print("Output shape:", output.shape)
```

Slide 8: Building a Complete GNN Model

Now, let's combine the components we've learned to build a complete GNN model for node classification:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphConvLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GraphConvLayer(hidden_dim, hidden_dim))
        self.convs.append(GraphConvLayer(hidden_dim, output_dim))
    
    def forward(self, x, adj):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, adj))
        x = self.convs[-1](x, adj)
        return F.log_softmax(x, dim=1)

# Example usage
input_dim, hidden_dim, output_dim, num_layers = 16, 32, 7, 3
num_nodes = 100

model = GNN(input_dim, hidden_dim, output_dim, num_layers)
x = torch.randn(num_nodes, input_dim)
adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()

output = model(x, adj)
print("Output shape:", output.shape)
```

Slide 9: Training a GNN

To train a GNN, we need to define a loss function and an optimization algorithm. Here's an example of training a GNN for node classification:

```python
import torch
import torch.nn.functional as F
import torch.optim as optim

def train_gnn(model, x, adj, labels, mask, epochs=200, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x, adj)
        loss = F.nll_loss(output[mask], labels[mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Example usage
input_dim, hidden_dim, output_dim, num_layers = 16, 32, 7, 3
num_nodes = 100

model = GNN(input_dim, hidden_dim, output_dim, num_layers)
x = torch.randn(num_nodes, input_dim)
adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
labels = torch.randint(0, output_dim, (num_nodes,))
mask = torch.randint(0, 2, (num_nodes,)).bool()

train_gnn(model, x, adj, labels, mask)
```

Slide 10: Evaluating a GNN

After training, we need to evaluate the performance of our GNN. Here's an example of how to compute accuracy on a test set:

```python
import torch

def evaluate_gnn(model, x, adj, labels, mask):
    model.eval()
    with torch.no_grad():
        output = model(x, adj)
        _, predicted = output.max(dim=1)
        correct = predicted[mask].eq(labels[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total
    return accuracy

# Example usage
model.eval()
test_mask = torch.randint(0, 2, (num_nodes,)).bool()
accuracy = evaluate_gnn(model, x, adj, labels, test_mask)
print(f"Test Accuracy: {accuracy:.4f}")
```

Slide 11: Real-Life Example: Social Network Analysis

GNNs can be applied to social network analysis for tasks such as friend recommendation or community detection. Here's a simple example of using a GNN to predict user interests based on their social connections:

```python
import networkx as nx
import torch
import torch.nn.functional as F

# Create a social network graph
G = nx.karate_club_graph()
adj = nx.to_numpy_array(G)
adj_tensor = torch.tensor(adj, dtype=torch.float32)

# Generate random user interests (0: Technology, 1: Sports, 2: Music)
num_nodes = G.number_of_nodes()
interests = torch.randint(0, 3, (num_nodes,))

# Create node features (one-hot encoding of interests)
x = F.one_hot(interests, num_classes=3).float()

# Define and train the GNN model
input_dim, hidden_dim, output_dim, num_layers = 3, 16, 3, 2
model = GNN(input_dim, hidden_dim, output_dim, num_layers)

# Train the model (simplified for demonstration)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    optimizer.zero_grad()
    output = model(x, adj_tensor)
    loss = F.nll_loss(output, interests)
    loss.backward()
    optimizer.step()

# Predict interests for a new user
new_user_connections = torch.zeros(num_nodes)
new_user_connections[0] = 1  # Connected to node 0
new_user_connections[5] = 1  # Connected to node 5
new_adj = torch.cat([adj_tensor, new_user_connections.unsqueeze(0)], dim=0)
new_adj = torch.cat([new_adj, torch.cat([new_user_connections, torch.tensor([0.0])]).unsqueeze(1)], dim=1)

new_x = torch.cat([x, torch.zeros(1, 3)], dim=0)

model.eval()
with torch.no_grad():
    output = model(new_x, new_adj)
    predicted_interest = output[-1].argmax().item()

interest_map = {0: "Technology", 1: "Sports", 2: "Music"}
print(f"Predicted interest for the new user: {interest_map[predicted_interest]}")
```

Slide 12: Real-Life Example: Molecular Property Prediction

GNNs are widely used in cheminformatics for predicting molecular properties. Here's a simplified example of using a GNN to predict the solubility of molecules:

```python
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    # Node features: atomic number
    node_features = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    
    # Edge indices
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
    
    return torch.tensor(node_features, dtype=torch.float), torch.tensor(edges).t().contiguous()

# Example GNN for molecular property prediction
class MoleculeGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MoleculeGNN, self).__init__()
        self.conv1 = GraphConvLayer(input_dim, hidden_dim)
        self.conv2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)  # Global pooling
        return self.fc(x)

# Usage example
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
x, edge_index = mol_to_graph(smiles)

model = MoleculeGNN(input_dim=1, hidden_dim=64, output_dim=1)
solubility = model(x, edge_index)
print(f"Predicted solubility: {solubility.item():.2f}")
```

Slide 13: Challenges and Future Directions in GNNs

Graphical Neural Networks have shown great promise, but there are still challenges to overcome:

1. Scalability: Many GNN architectures struggle with large-scale graphs. Developing more efficient algorithms for processing large graphs is an active area of research.
2. Depth limitation: GNNs often suffer from the over-smoothing problem when stacking many layers, limiting their depth. Addressing this issue could lead to more expressive models.
3. Heterogeneous graphs: Real-world graphs often contain different types of nodes and edges. Designing GNNs that can effectively handle heterogeneous graphs is crucial for many applications.
4. Dynamic graphs: Many real-world graphs change over time. Developing GNNs that can adapt to evolving graph structures is an important direction for future research.
5. Interpretability: As with many deep learning models, interpreting the decisions made by GNNs can be challenging. Improving the interpretability of GNNs is essential for their adoption in critical applications.

Researchers are actively working on these challenges, and we can expect significant advancements in GNN architectures and applications in the coming years.

Slide 14: Additional Resources

For those interested in diving deeper into Graphical Neural Networks, here are some valuable resources:

1. "Graph Representation Learning" by William L. Hamilton ArXiv: [https://arxiv.org/abs/1709.05584](https://arxiv.org/abs/1709.05584)
2. "A Comprehensive Survey on Graph Neural Networks" by Wu et al. ArXiv: [https://arxiv.org/abs/1901.00596](https://arxiv.org/abs/1901.00596)
3. "Graph Neural Networks: A Review of Methods and Applications" by Zhou et al. ArXiv: [https://arxiv.org/abs/1812.08434](https://arxiv.org/abs/1812.08434)
4. PyTorch Geometric Documentation [https://pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/)
5. Deep Graph Library (DGL) [https://www.dgl.ai/](https://www.dgl.ai/)

These resources provide in-depth explanations of GNN concepts, architectures, and applications, as well as practical tools for implementing GNNs in your projects.


