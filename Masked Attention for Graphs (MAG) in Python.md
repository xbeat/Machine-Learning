## Masked Attention for Graphs (MAG) in Python
Slide 1: Introduction to Masked Attention for Graphs (MAG)

Masked Attention for Graphs (MAG) is a technique used in graph neural networks to improve the efficiency and effectiveness of attention mechanisms when processing large-scale graphs. It selectively focuses on important parts of the graph, reducing computational complexity and improving performance.

```python
import networkx as nx
import numpy as np

def create_sample_graph():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
    return G

graph = create_sample_graph()
print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")
```

Slide 2: Graph Representation

In MAG, graphs are typically represented as adjacency matrices or edge lists. We'll use NetworkX to create and manipulate graphs, which provides flexible representations.

```python
import networkx as nx
import numpy as np

G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])

# Adjacency matrix representation
adj_matrix = nx.to_numpy_array(G)
print("Adjacency Matrix:")
print(adj_matrix)

# Edge list representation
edge_list = list(G.edges())
print("\nEdge List:")
print(edge_list)
```

Slide 3: Node Features and Embeddings

Node features are crucial in graph learning tasks. In MAG, we often work with node embeddings, which are low-dimensional vector representations of nodes.

```python
import torch
import torch.nn as nn

class NodeEmbedding(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(NodeEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
    
    def forward(self, node_ids):
        return self.embedding(node_ids)

num_nodes = 4
embedding_dim = 8
node_embedding = NodeEmbedding(num_nodes, embedding_dim)

# Generate embeddings for all nodes
all_nodes = torch.arange(num_nodes)
embeddings = node_embedding(all_nodes)

print("Node embeddings:")
print(embeddings)
```

Slide 4: Attention Mechanism Basics

Attention mechanisms allow the model to focus on relevant parts of the input. In graphs, attention helps nodes aggregate information from their neighbors more effectively.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        # x shape: (num_nodes, num_neighbors, input_dim)
        attn_scores = self.attention(x).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=-1)
        weighted_sum = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        return weighted_sum

# Example usage
input_dim = 8
num_nodes = 3
num_neighbors = 4
x = torch.randn(num_nodes, num_neighbors, input_dim)

attention_layer = SimpleAttention(input_dim)
output = attention_layer(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
```

Slide 5: Masked Attention Concept

Masked attention in graphs involves selectively attending to a subset of neighbors for each node. This is achieved by applying a mask to the attention scores before normalization.

```python
import torch
import torch.nn.functional as F

def masked_attention(query, key, value, mask):
    # query, key, value shapes: (num_nodes, num_neighbors, dim)
    # mask shape: (num_nodes, num_neighbors)
    
    attention_scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Apply mask (set masked positions to -inf)
    attention_scores = attention_scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
    
    attention_weights = F.softmax(attention_scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output

# Example usage
num_nodes = 3
num_neighbors = 4
dim = 8

query = torch.randn(num_nodes, 1, dim)
key = torch.randn(num_nodes, num_neighbors, dim)
value = torch.randn(num_nodes, num_neighbors, dim)
mask = torch.randint(0, 2, (num_nodes, num_neighbors)).bool()

output = masked_attention(query, key, value, mask)
print("Output shape:", output.shape)
```

Slide 6: Implementing MAG Layer

Let's implement a basic MAG layer that combines node features with masked attention.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAGLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MAGLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.attention = nn.Linear(output_dim, 1)
    
    def forward(self, x, adj_matrix, mask):
        # x shape: (num_nodes, input_dim)
        # adj_matrix shape: (num_nodes, num_nodes)
        # mask shape: (num_nodes, num_nodes)
        
        h = self.linear(x)
        
        # Compute attention scores
        attn_scores = self.attention(h)
        attn_scores = attn_scores + attn_scores.transpose(0, 1)
        
        # Apply mask and adjacency matrix
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_scores = attn_scores.masked_fill(adj_matrix == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=1)
        output = torch.matmul(attn_weights, h)
        
        return output

# Example usage
num_nodes = 5
input_dim = 8
output_dim = 16

x = torch.randn(num_nodes, input_dim)
adj_matrix = torch.randint(0, 2, (num_nodes, num_nodes))
mask = torch.randint(0, 2, (num_nodes, num_nodes))

mag_layer = MAGLayer(input_dim, output_dim)
output = mag_layer(x, adj_matrix, mask)

print("Output shape:", output.shape)
```

Slide 7: Multi-head Attention in MAG

Multi-head attention allows the model to capture different types of relationships between nodes. Let's implement a multi-head MAG layer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadMAGLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MultiHeadMAGLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.attention = nn.Linear(self.head_dim, 1, bias=False)
        self.output_linear = nn.Linear(output_dim, output_dim)
    
    def forward(self, x, adj_matrix, mask):
        batch_size = x.size(0)
        h = self.linear(x).view(batch_size, -1, self.num_heads, self.head_dim)
        
        attn_scores = self.attention(h).squeeze(-1)
        attn_scores = attn_scores + attn_scores.transpose(1, 2)
        
        # Apply mask and adjacency matrix
        mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
        adj_matrix = adj_matrix.unsqueeze(1).repeat(1, self.num_heads, 1)
        
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_scores = attn_scores.masked_fill(adj_matrix == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        output = torch.matmul(attn_weights.unsqueeze(-2), h).squeeze(-2)
        output = output.view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.output_linear(output)
        
        return output

# Example usage
num_nodes = 5
input_dim = 8
output_dim = 16
num_heads = 4

x = torch.randn(num_nodes, input_dim)
adj_matrix = torch.randint(0, 2, (num_nodes, num_nodes))
mask = torch.randint(0, 2, (num_nodes, num_nodes))

multi_head_mag_layer = MultiHeadMAGLayer(input_dim, output_dim, num_heads)
output = multi_head_mag_layer(x, adj_matrix, mask)

print("Output shape:", output.shape)
```

Slide 8: Positional Encodings in MAG

Positional encodings can be used to incorporate structural information about the graph into the model. Let's implement a simple positional encoding scheme based on node degrees.

```python
import torch
import networkx as nx

def degree_positional_encoding(graph, max_degree, embedding_dim):
    degrees = dict(graph.degree())
    num_nodes = len(degrees)
    
    pos_encoding = torch.zeros(num_nodes, embedding_dim)
    for node, degree in degrees.items():
        for i in range(embedding_dim):
            if i % 2 == 0:
                pos_encoding[node, i] = torch.sin(degree * torch.pi / max_degree * (2 ** (i // 2)))
            else:
                pos_encoding[node, i] = torch.cos(degree * torch.pi / max_degree * (2 ** ((i - 1) // 2)))
    
    return pos_encoding

# Example usage
G = nx.karate_club_graph()
max_degree = max(dict(G.degree()).values())
embedding_dim = 16

pos_encodings = degree_positional_encoding(G, max_degree, embedding_dim)

print("Positional encodings shape:", pos_encodings.shape)
print("First node positional encoding:")
print(pos_encodings[0])
```

Slide 9: Combining MAG with Graph Convolutional Networks (GCN)

We can combine MAG with traditional graph convolutional layers to create a hybrid model that leverages both approaches.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAGGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MAGGCNLayer, self).__init__()
        self.mag_layer = MultiHeadMAGLayer(input_dim, output_dim, num_heads)
        self.gcn_layer = nn.Linear(input_dim, output_dim)
        self.combine = nn.Linear(output_dim * 2, output_dim)
    
    def forward(self, x, adj_matrix, mask):
        mag_output = self.mag_layer(x, adj_matrix, mask)
        
        # GCN layer
        adj_norm = F.normalize(adj_matrix, p=1, dim=1)
        gcn_output = torch.matmul(adj_norm, self.gcn_layer(x))
        
        # Combine MAG and GCN outputs
        combined = torch.cat([mag_output, gcn_output], dim=-1)
        output = self.combine(combined)
        
        return output

# Example usage
num_nodes = 5
input_dim = 8
output_dim = 16
num_heads = 4

x = torch.randn(num_nodes, input_dim)
adj_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()
mask = torch.randint(0, 2, (num_nodes, num_nodes))

mag_gcn_layer = MAGGCNLayer(input_dim, output_dim, num_heads)
output = mag_gcn_layer(x, adj_matrix, mask)

print("Output shape:", output.shape)
```

Slide 10: Real-Life Example: Social Network Analysis

Let's use MAG for analyzing a social network to identify influential users based on their connections and activities.

```python
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

class SocialNetworkMAG(nn.Module):
    def __init__(self, num_users, embedding_dim, hidden_dim):
        super(SocialNetworkMAG, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mag_layer = MAGLayer(embedding_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, adj_matrix, mask):
        user_ids = torch.arange(adj_matrix.size(0))
        x = self.user_embedding(user_ids)
        h = self.mag_layer(x, adj_matrix, mask)
        scores = self.output_layer(h).squeeze(-1)
        return F.softmax(scores, dim=0)

# Create a sample social network
G = nx.karate_club_graph()
adj_matrix = torch.tensor(nx.to_numpy_array(G)).float()
mask = torch.ones_like(adj_matrix)

num_users = G.number_of_nodes()
embedding_dim = 16
hidden_dim = 32

model = SocialNetworkMAG(num_users, embedding_dim, hidden_dim)
influence_scores = model(adj_matrix, mask)

print("Top 5 influential users:")
top_users = torch.topk(influence_scores, k=5)
for i, (user, score) in enumerate(zip(top_users.indices, top_users.values)):
    print(f"{i+1}. User {user.item()}: {score.item():.4f}")
```

Slide 11: Real-Life Example: Recommendation System

Let's implement a simple recommendation system using MAG to suggest items to users based on their interactions and item similarities.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAGRecommendationSystem(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim):
        super(MAGRecommendationSystem, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mag_layer = MAGLayer(embedding_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, user_ids, item_ids, interaction_matrix, mask):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        x = user_emb + item_emb
        h = self.mag_layer(x, interaction_matrix, mask)
        scores = self.output_layer(h).squeeze(-1)
        return torch.sigmoid(scores)

# Sample data
num_users, num_items = 100, 50
embedding_dim, hidden_dim = 16, 32

interaction_matrix = torch.randint(0, 2, (num_users, num_items)).float()
mask = torch.ones_like(interaction_matrix)

model = MAGRecommendationSystem(num_users, num_items, embedding_dim, hidden_dim)

# Generate recommendations for a sample user
user_id = torch.tensor([0])
item_ids = torch.arange(num_items)
predictions = model(user_id.expand(num_items), item_ids, interaction_matrix, mask)

top_recommendations = torch.topk(predictions, k=5)
print("Top 5 recommended items for user 0:")
for i, (item, score) in enumerate(zip(top_recommendations.indices, top_recommendations.values)):
    print(f"{i+1}. Item {item.item()}: {score.item():.4f}")
```

Slide 12: Handling Large-Scale Graphs

For large-scale graphs, we need to implement efficient strategies to handle memory constraints and computational complexity.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LargeScaleMAGLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_neighbors):
        super(LargeScaleMAGLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.attention = nn.Linear(output_dim, 1)
        self.num_neighbors = num_neighbors
    
    def forward(self, x, edge_index):
        h = self.linear(x)
        
        row, col = edge_index
        
        # Sample neighbors
        if col.size(0) > self.num_neighbors:
            perm = torch.randperm(col.size(0))[:self.num_neighbors]
            row, col = row[perm], col[perm]
        
        # Compute attention scores
        attn_scores = self.attention(h[row] + h[col])
        attn_weights = F.softmax(attn_scores, dim=0)
        
        # Aggregate neighbors
        out = torch.zeros_like(h)
        out.index_add_(0, row, attn_weights * h[col])
        
        return out

# Example usage
num_nodes = 1000000
input_dim = 64
output_dim = 32
num_neighbors = 10

x = torch.randn(num_nodes, input_dim)
edge_index = torch.randint(0, num_nodes, (2, num_nodes * 5))

large_scale_mag = LargeScaleMAGLayer(input_dim, output_dim, num_neighbors)
output = large_scale_mag(x, edge_index)

print("Output shape:", output.shape)
```

Slide 13: Evaluating MAG Performance

To assess the effectiveness of MAG, we need to evaluate its performance on various graph-based tasks. Let's implement a simple evaluation function for node classification.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

def evaluate_node_classification(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        output = model(features, graph.edge_index)
        pred = output.argmax(dim=1)
        acc = accuracy_score(labels[mask].cpu(), pred[mask].cpu())
        f1 = f1_score(labels[mask].cpu(), pred[mask].cpu(), average='weighted')
    return acc, f1

class SimpleMAGModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMAGModel, self).__init__()
        self.mag_layer = MAGLayer(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        h = self.mag_layer(x, edge_index)
        return self.output_layer(h)

# Example usage (pseudo-code)
# graph = load_graph()
# features = load_node_features()
# labels = load_node_labels()
# train_mask, val_mask, test_mask = generate_masks()

# model = SimpleMAGModel(input_dim, hidden_dim, num_classes)
# optimizer = optim.Adam(model.parameters())
# criterion = nn.CrossEntropyLoss()

# Train the model
# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#     output = model(features, graph.edge_index)
#     loss = criterion(output[train_mask], labels[train_mask])
#     loss.backward()
#     optimizer.step()
    
#     if epoch % 10 == 0:
#         val_acc, val_f1 = evaluate_node_classification(model, graph, features, labels, val_mask)
#         print(f"Epoch {epoch}, Validation Accuracy: {val_acc:.4f}, F1-score: {val_f1:.4f}")

# Test the model
# test_acc, test_f1 = evaluate_node_classification(model, graph, features, labels, test_mask)
# print(f"Test Accuracy: {test_acc:.4f}, F1-score: {test_f1:.4f}")
```

Slide 14: Challenges and Future Directions

Masked Attention for Graphs (MAG) has shown promising results, but there are still challenges to address and opportunities for future research:

1. Scalability: Developing more efficient algorithms for handling extremely large graphs with billions of nodes and edges.
2. Dynamic graphs: Adapting MAG to graphs that change over time, incorporating temporal information.
3. Explainability: Improving the interpretability of MAG models to understand which graph structures are most important for predictions.
4. Heterogeneous graphs: Extending MAG to handle graphs with multiple node and edge types.
5. Combining with other techniques: Integrating MAG with other graph neural network architectures and learning paradigms.

```python
import torch
import torch.nn as nn

class FutureMAGLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FutureMAGLayer, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads=8)
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x, edge_index, edge_attr=None, time_attr=None):
        # Placeholder for future improvements
        # 1. Efficient sparse attention
        # 2. Incorporate temporal information
        # 3. Handle heterogeneous graph data
        # 4. Implement explainable attention mechanisms
        
        h = self.attention(x, x, x)[0]
        h = self.linear(h)
        return self.norm(h + x)

# Example usage (pseudo-code)
# model = FutureMAGLayer(input_dim, output_dim)
# output = model(x, edge_index, edge_attr, time_attr)
```

Slide 15: Additional Resources

For more information on Masked Attention for Graphs and related topics, consider exploring the following resources:

1. "Graph Attention Networks" by Veličković et al. (2017) - ArXiv:1710.10903 [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
2. "Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification" by Shi et al. (2021) - ArXiv:2009.03509 [https://arxiv.org/abs/2009.03509](https://arxiv.org/abs/2009.03509)
3. "A Generalization of Transformer Networks to Graphs" by Dwivedi and Bresson (2021) - ArXiv:2012.09699 [https://arxiv.org/abs/2012.09699](https://arxiv.org/abs/2012.09699)
4. "Benchmarking Graph Neural Networks" by Dwivedi et al. (2020) - ArXiv:2003.00982 [https://arxiv.org/abs/2003.00982](https://arxiv.org/abs/2003.00982)

These papers provide in-depth discussions on graph attention mechanisms, masked label prediction, and the application of transformer-like architectures to graphs. They offer valuable insights into the development and evaluation of graph neural networks, including masked attention techniques.

