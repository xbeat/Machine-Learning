## Self-Attention as a Directed Graph in Python
Slide 1: Self-attention as a Directed Graph

Self-attention is a fundamental mechanism in modern deep learning architectures, particularly in transformer models. By representing self-attention as a directed graph, we can gain insights into its inner workings and visualize the relationships between different elements in a sequence.

```python
import networkx as nx
import matplotlib.pyplot as plt

def create_attention_graph(attention_weights):
    G = nx.DiGraph()
    n = len(attention_weights)
    
    for i in range(n):
        for j in range(n):
            G.add_edge(i, j, weight=attention_weights[i][j])
    
    return G

# Example attention weights
attention_weights = [
    [0.7, 0.2, 0.1],
    [0.3, 0.6, 0.1],
    [0.1, 0.2, 0.7]
]

G = create_attention_graph(attention_weights)
nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, arrows=True)
plt.title("Self-attention as a Directed Graph")
plt.show()
```

Slide 2: Graph Representation of Self-attention

In this representation, nodes correspond to elements in the input sequence, and edges represent the attention weights between these elements. The direction of the edges indicates the flow of information, while the weight of each edge represents the strength of the attention.

```python
def visualize_attention_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=700, arrows=True)
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title("Weighted Self-attention Graph")
    plt.axis('off')
    plt.show()

visualize_attention_graph(G)
```

Slide 3: Calculating Attention Weights

Attention weights are computed using queries, keys, and values. The dot product of queries and keys determines the attention scores, which are then normalized using softmax to obtain the final attention weights.

```python
import numpy as np

def attention_weights(query, key, value):
    scores = np.dot(query, key.T)
    attention = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    output = np.dot(attention, value)
    return attention, output

# Example inputs
query = np.array([[1, 0], [0, 1]])
key = np.array([[1, 1], [0, 1]])
value = np.array([[1, 0], [1, 1]])

attention, output = attention_weights(query, key, value)
print("Attention weights:\n", attention)
print("Output:\n", output)
```

Slide 4: Multi-head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces. It involves applying the attention mechanism multiple times in parallel and concatenating the results.

```python
def multi_head_attention(query, key, value, num_heads):
    d_model = query.shape[-1]
    d_head = d_model // num_heads
    
    # Split into multiple heads
    query_heads = np.split(query, num_heads, axis=-1)
    key_heads = np.split(key, num_heads, axis=-1)
    value_heads = np.split(value, num_heads, axis=-1)
    
    # Calculate attention for each head
    head_outputs = []
    for q, k, v in zip(query_heads, key_heads, value_heads):
        _, output = attention_weights(q, k, v)
        head_outputs.append(output)
    
    # Concatenate head outputs
    return np.concatenate(head_outputs, axis=-1)

# Example usage
num_heads = 2
multi_head_output = multi_head_attention(query, key, value, num_heads)
print("Multi-head attention output:\n", multi_head_output)
```

Slide 5: Directed Graph Properties

The directed graph representation of self-attention allows us to analyze various graph-theoretic properties, such as centrality measures, which can provide insights into the importance of different elements in the sequence.

```python
def analyze_graph_properties(G):
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    pagerank = nx.pagerank(G)
    
    print("In-degree centrality:", in_degree)
    print("Out-degree centrality:", out_degree)
    print("PageRank centrality:", pagerank)

analyze_graph_properties(G)
```

Slide 6: Attention Visualization

Visualizing attention weights can help in understanding how the model attends to different parts of the input. We can create heatmaps to represent the attention distribution across the sequence.

```python
import seaborn as sns

def visualize_attention_heatmap(attention_weights):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, annot=True, cmap='YlGnBu')
    plt.title("Attention Weights Heatmap")
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.show()

visualize_attention_heatmap(attention)
```

Slide 7: Self-attention in Natural Language Processing

In NLP tasks, self-attention allows the model to weigh the importance of different words in a sentence. This is particularly useful for tasks like machine translation and sentiment analysis.

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out

# Example usage
embed_size = 256
heads = 8
model = SelfAttention(embed_size, heads)
x = torch.randn(32, 10, embed_size)  # (batch_size, seq_len, embed_size)
output = model(x, x, x)
print(output.shape)  # Should be (32, 10, 256)
```

Slide 8: Self-attention in Computer Vision

Self-attention can also be applied to computer vision tasks, allowing models to focus on relevant parts of an image. This is particularly useful in tasks like object detection and image captioning.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention2D(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention2D, self).__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        proj_query = self.query(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value(x).view(batch_size, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x
        return out

# Example usage
in_channels = 64
model = SelfAttention2D(in_channels)
x = torch.randn(1, in_channels, 32, 32)  # (batch_size, channels, height, width)
output = model(x)
print(output.shape)  # Should be (1, 64, 32, 32)
```

Slide 9: Graph Attention Networks (GAT)

Graph Attention Networks extend the concept of self-attention to graph-structured data. In GATs, each node attends over its neighbors, enabling the model to assign different importances to different nodes in a neighborhood.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

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

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

# Example usage
in_features = 16
out_features = 8
num_nodes = 10
model = GraphAttentionLayer(in_features, out_features, dropout=0.6, alpha=0.2)
h = torch.randn(num_nodes, in_features)
adj = torch.randint(0, 2, (num_nodes, num_nodes))
output = model(h, adj)
print(output.shape)  # Should be (10, 8)
```

Slide 10: Attention Flow in Directed Graphs

The flow of attention in a directed graph can be analyzed using techniques from network flow theory. This analysis can reveal important patterns in how information propagates through the attention mechanism.

```python
import networkx as nx

def analyze_attention_flow(G):
    # Calculate maximum flow
    source = 0  # Assuming node 0 is the source
    sink = len(G) - 1  # Assuming the last node is the sink
    max_flow_value, flow_dict = nx.maximum_flow(G, source, sink)
    
    # Calculate betweenness centrality
    betweenness = nx.betweenness_centrality(G, weight='weight')
    
    print("Maximum flow value:", max_flow_value)
    print("Flow dictionary:", flow_dict)
    print("Betweenness centrality:", betweenness)

# Using the graph created earlier
analyze_attention_flow(G)
```

Slide 11: Attention Rollout

Attention rollout is a technique used to visualize the flow of attention through multiple layers of a transformer model. It involves recursively multiplying attention matrices to obtain the cumulative attention.

```python
import numpy as np

def attention_rollout(attention_matrices):
    num_layers = len(attention_matrices)
    attention_rollout = np.eye(attention_matrices[0].shape[1])
    
    for i in range(num_layers):
        attention_rollout = np.matmul(attention_matrices[i], attention_rollout)
        attention_rollout = attention_rollout / attention_rollout.sum(axis=-1, keepdims=True)
    
    return attention_rollout

# Example usage
num_layers = 3
seq_len = 5
attention_matrices = [np.random.rand(seq_len, seq_len) for _ in range(num_layers)]
rollout = attention_rollout(attention_matrices)

plt.figure(figsize=(10, 8))
sns.heatmap(rollout, annot=True, cmap='YlGnBu')
plt.title("Attention Rollout")
plt.xlabel("Token")
plt.ylabel("Token")
plt.show()
```

Slide 12: Sparse Attention Graphs

In practice, attention mechanisms often produce sparse graphs, where many attention weights are close to zero. Techniques like pruning and sparse attention can be used to exploit this sparsity for computational efficiency.

```python
import torch
import torch.nn.functional as F

def sparse_attention(query, key, value, sparsity_threshold=0.1):
    attention_weights = torch.matmul(query, key.transpose(-2, -1))
    attention_weights = F.softmax(attention_weights, dim=-1)
    
    sparse_mask = (attention_weights > sparsity_threshold).float()
    sparse_attention_weights = attention_weights * sparse_mask
    sparse_attention_weights /= sparse_attention_weights.sum(dim=-1, keepdim=True)
    
    output = torch.matmul(sparse_attention_weights, value)
    return output, sparse_attention_weights

# Example usage
seq_len, d_model = 10, 64
query = torch.randn(1, seq_len, d_model)
key = torch.randn(1, seq_len, d_model)
value = torch.randn(1, seq_len, d_model)

output, sparse_weights = sparse_attention(query, key, value)
print("Output shape:", output.shape)
print("Sparse weights shape:", sparse_weights.shape)
print("Sparsity: {:.2f}%".format(100 * (sparse_weights == 0).float().mean()))
```

Slide 13: Attention in Graph Neural Networks

Graph Neural Networks (GNNs) can incorporate attention mechanisms to weigh the importance of different neighboring nodes when aggregating information. This allows the model to focus on the most relevant parts of the graph structure.

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
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

# Example usage
in_features, out_features, num_nodes = 16, 8, 10
model = GraphAttentionLayer(in_features, out_features, dropout=0.6, alpha=0.2)
h = torch.randn(num_nodes, in_features)
adj = torch.randint(0, 2, (num_nodes, num_nodes))
output = model(h, adj)
print("Output shape:", output.shape)
```

Slide 14: Attention-based Graph Pooling

Attention mechanisms can be used for graph pooling, allowing the model to learn which nodes are most important when creating a graph-level representation. This is particularly useful for tasks like graph classification.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, x, mask=None):
        weights = self.project(x).squeeze(-1)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = F.softmax(weights, dim=-1)
        x = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return x, weights

# Example usage
in_features, hidden_dim, num_nodes = 64, 32, 10
model = AttentionPooling(in_features, hidden_dim)
x = torch.randn(1, num_nodes, in_features)
mask = torch.ones(1, num_nodes)
pooled, attention_weights = model(x, mask)
print("Pooled shape:", pooled.shape)
print("Attention weights shape:", attention_weights.shape)
```

Slide 15: Additional Resources

For those interested in diving deeper into the topic of self-attention as a directed graph, the following resources provide valuable insights and advanced techniques:

1. "Attention Is All You Need" by Vaswani et al. (2017) ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "Graph Attention Networks" by Veličković et al. (2017) ArXiv: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
3. "Are Transformers Universal Approximators of Sequence-to-Sequence Functions?" by Yun et al. (2019) ArXiv: [https://arxiv.org/abs/1912.10077](https://arxiv.org/abs/1912.10077)

These papers provide the theoretical foundations and practical applications of self-attention mechanisms in various contexts, including natural language processing and graph-structured data.

