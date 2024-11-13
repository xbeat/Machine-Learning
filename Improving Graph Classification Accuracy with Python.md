## Improving Graph Classification Accuracy with Python
Slide 1: Graph Data Preprocessing

Understanding graph data requires proper preprocessing techniques to handle node features, edge attributes, and structural information. The preprocessing phase significantly impacts model performance by ensuring consistent data formats and meaningful feature representations.

```python
import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_graph(adj_matrix, node_features):
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    
    # Normalize node features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(node_features)
    
    # Add node features to graph
    for idx, features in enumerate(normalized_features):
        G.nodes[idx]['features'] = features
    
    # Calculate basic graph metrics
    degrees = dict(G.degree())
    clustering_coef = nx.clustering(G)
    
    return G, degrees, clustering_coef

# Example usage
adj_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
features = np.array([[1, 2], [3, 4], [5, 6]])
G, degrees, clustering = preprocess_graph(adj_mat, features)
```

Slide 2: Feature Engineering for Graphs

Graph-level features require careful engineering to capture both local and global structural properties. We'll implement key graph metrics that serve as discriminative features for classification tasks.

```python
def extract_graph_features(G):
    features = {}
    
    # Structural features
    features['num_nodes'] = G.number_of_nodes()
    features['num_edges'] = G.number_of_edges()
    features['density'] = nx.density(G)
    
    # Spectral features
    laplacian = nx.normalized_laplacian_matrix(G)
    eigvals = np.linalg.eigvals(laplacian.toarray())
    features['spectral_radius'] = max(abs(eigvals))
    
    # Centrality measures
    features['avg_betweenness'] = np.mean(list(nx.betweenness_centrality(G).values()))
    features['avg_closeness'] = np.mean(list(nx.closeness_centrality(G).values()))
    
    return features

# Example usage
G = nx.erdos_renyi_graph(10, 0.3)
graph_features = extract_graph_features(G)
```

Slide 3: Graph Neural Network Architecture

The foundation of modern graph classification lies in Graph Neural Networks (GNNs). This implementation shows a basic GNN layer using PyTorch Geometric, incorporating message passing and node feature aggregation.

```python
import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GraphClassifier, self).__init__()
        
        self.conv1 = gnn.GCNConv(input_dim, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.pool = gnn.global_mean_pool
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, edge_index, batch):
        # Message passing layers
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        
        # Graph-level pooling
        x = self.pool(x, batch)
        
        # Classification
        return self.fc(x)

# Model initialization
model = GraphClassifier(input_dim=10, hidden_dim=64, num_classes=2)
```

Slide 4: Attention Mechanisms for Graph Classification

Attention mechanisms enhance graph classification by learning to focus on important nodes and edges. This implementation demonstrates a graph attention layer with multi-head attention.

```python
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.6):
        super(GraphAttentionLayer, self).__init__()
        
        self.num_heads = num_heads
        self.attentions = nn.ModuleList([
            gnn.GATConv(
                in_features, 
                out_features // num_heads, 
                heads=1, 
                dropout=dropout
            ) for _ in range(num_heads)
        ])
    
    def forward(self, x, edge_index):
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        return torch.relu(x)

# Example usage
attention_layer = GraphAttentionLayer(64, 32)
```

Slide 5: Data Augmentation for Graphs

Graph data augmentation improves model generalization by creating meaningful variations of the input graphs. This implementation includes edge perturbation, node feature masking, and subgraph sampling techniques.

```python
def augment_graph(G, p_edge=0.1, p_feat=0.1, num_augmentations=1):
    augmented_graphs = []
    
    for _ in range(num_augmentations):
        G_aug = G.copy()
        
        # Edge perturbation
        edges = list(G_aug.edges())
        for edge in edges:
            if np.random.random() < p_edge:
                if G_aug.has_edge(*edge):
                    G_aug.remove_edge(*edge)
                else:
                    G_aug.add_edge(*edge)
        
        # Node feature masking
        for node in G_aug.nodes():
            if np.random.random() < p_feat:
                features = G_aug.nodes[node]['features']
                mask = np.random.random(features.shape) > p_feat
                G_aug.nodes[node]['features'] = features * mask
        
        augmented_graphs.append(G_aug)
    
    return augmented_graphs

# Example usage
G = nx.karate_club_graph()
augmented = augment_graph(G, p_edge=0.1, p_feat=0.1, num_augmentations=2)
```

Slide 6: Graph Pooling Strategies

Effective graph pooling mechanisms are crucial for learning hierarchical representations. This implementation showcases different pooling strategies including TopK and attention-based pooling.

```python
class HierarchicalPooling(nn.Module):
    def __init__(self, in_channels, ratio=0.5, min_score=None):
        super(HierarchicalPooling, self).__init__()
        
        self.topk_pool = gnn.TopKPooling(
            in_channels, 
            ratio=ratio,
            min_score=min_score
        )
        self.sag_pool = gnn.SAGPooling(
            in_channels,
            ratio=ratio,
            min_score=min_score
        )
        
    def forward(self, x, edge_index, batch):
        # TopK pooling
        x1, edge_index1, _, batch1, _, _ = self.topk_pool(
            x, edge_index, None, batch
        )
        
        # SAG pooling
        x2, edge_index2, _, batch2, _, _ = self.sag_pool(
            x, edge_index, None, batch
        )
        
        # Combine pooling results
        x_combined = torch.cat([x1, x2], dim=-1)
        return x_combined, batch1

# Example usage
pool_layer = HierarchicalPooling(in_channels=64)
```

Slide 7: Loss Functions for Graph Classification

Specialized loss functions can significantly improve graph classification performance. This implementation includes contrastive loss and focal loss adaptations for graph-level tasks.

```python
class GraphClassificationLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, temperature=0.07):
        super(GraphClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
    def focal_loss(self, pred, target):
        ce_loss = self.ce(pred, target)
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
    
    def contrastive_loss(self, features, labels):
        normalized = nn.functional.normalize(features, dim=1)
        similarity = torch.matmul(normalized, normalized.t()) / self.temperature
        
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        negative_mask = ~positive_mask
        
        numerator = torch.exp(similarity[positive_mask])
        denominator = torch.sum(torch.exp(similarity[negative_mask]), dim=1)
        
        return -torch.log(numerator / denominator).mean()
    
    def forward(self, pred, features, target):
        focal = self.focal_loss(pred, target)
        contrastive = self.contrastive_loss(features, target)
        return focal + contrastive

# Example usage
criterion = GraphClassificationLoss()
```

Slide 8: Graph Regularization Techniques

Advanced regularization methods specifically designed for graph neural networks help prevent overfitting and improve model generalization. Implementations include edge dropout and feature regularization.

```python
class GraphRegularization(nn.Module):
    def __init__(self, dropout_edge=0.1, l1_lambda=1e-5):
        super(GraphRegularization, self).__init__()
        self.dropout_edge = dropout_edge
        self.l1_lambda = l1_lambda
        
    def edge_dropout(self, edge_index, p=None):
        if p is None:
            p = self.dropout_edge
            
        mask = torch.rand(edge_index.size(1)) > p
        return edge_index[:, mask]
    
    def feature_regularization(self, model):
        l1_reg = 0
        for param in model.parameters():
            l1_reg += torch.norm(param, p=1)
        return self.l1_lambda * l1_reg
    
    def apply(self, model, edge_index, training=True):
        if training:
            edge_index = self.edge_dropout(edge_index)
        reg_loss = self.feature_regularization(model)
        return edge_index, reg_loss

# Example usage
regularizer = GraphRegularization()
```

Slide 9: Advanced Message Passing Implementation

Message passing is the core operation in graph neural networks. This implementation demonstrates sophisticated message aggregation schemes including attention-based and gated mechanisms for improved information flow.

```python
class AdvancedMessagePassing(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdvancedMessagePassing, self).__init__()
        
        self.message_nn = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        self.update_nn = nn.GRUCell(out_channels, out_channels)
        self.attention = nn.Linear(2 * out_channels, 1)
        
    def message_function(self, x_i, x_j):
        # Concatenate source and target node features
        msg_features = torch.cat([x_i, x_j], dim=1)
        return self.message_nn(msg_features)
    
    def attention_weights(self, x_i, x_j):
        # Compute attention scores
        attention_input = torch.cat([x_i, x_j], dim=1)
        return torch.sigmoid(self.attention(attention_input))
    
    def forward(self, x, edge_index):
        row, col = edge_index
        
        # Compute messages
        msg = self.message_function(x[row], x[col])
        
        # Apply attention
        alpha = self.attention_weights(x[row], x[col])
        msg = msg * alpha
        
        # Aggregate messages
        aggr_msg = scatter_add(msg, row, dim=0, dim_size=x.size(0))
        
        # Update node features
        x_new = self.update_nn(aggr_msg, x)
        
        return x_new

# Example usage
message_passing = AdvancedMessagePassing(64, 64)
```

Slide 10: Ensemble Methods for Graph Classification

Combining multiple graph neural networks with different architectures can significantly improve classification accuracy. This implementation shows a sophisticated ensemble approach with weighted voting.

```python
class GraphEnsemble(nn.Module):
    def __init__(self, input_dim, num_classes, num_models=3):
        super(GraphEnsemble, self).__init__()
        
        self.models = nn.ModuleList([
            GraphClassifier(input_dim, 64, num_classes),
            GraphClassifier(input_dim, 128, num_classes),
            GraphClassifier(input_dim, 256, num_classes)
        ])
        
        # Learnable weights for ensemble
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)
        
    def forward(self, x, edge_index, batch):
        predictions = []
        
        for model in self.models:
            pred = model(x, edge_index, batch)
            predictions.append(pred.unsqueeze(0))
        
        # Stack predictions and apply weighted average
        stacked_preds = torch.cat(predictions, dim=0)
        weighted_preds = torch.sum(
            stacked_preds * self.weights.view(-1, 1, 1), 
            dim=0
        )
        
        return weighted_preds
    
    def get_model_weights(self):
        return torch.softmax(self.weights, dim=0)

# Example usage
ensemble = GraphEnsemble(input_dim=10, num_classes=2)
```

Slide 11: Graph Isomorphism Testing

Graph isomorphism testing is crucial for verifying model invariance to graph permutations. This implementation provides a practical approach to test graph isomorphism and generate isomorphic variants.

```python
def test_graph_isomorphism(model, G1, G2):
    def get_graph_embedding(G):
        # Convert graph to required format
        x = torch.FloatTensor(
            [G.nodes[n].get('features', [0]) for n in G.nodes()]
        )
        edge_index = torch.LongTensor(
            [[e[0], e[1]] for e in G.edges()]
        ).t()
        
        # Get embedding through model
        with torch.no_grad():
            embedding = model.get_graph_embedding(x, edge_index)
        return embedding
    
    # Get embeddings
    emb1 = get_graph_embedding(G1)
    emb2 = get_graph_embedding(G2)
    
    # Compare embeddings
    distance = torch.norm(emb1 - emb2)
    is_isomorphic = distance < 1e-6
    
    return is_isomorphic, distance.item()

def generate_isomorphic_graph(G):
    # Create random node permutation
    perm = np.random.permutation(G.number_of_nodes())
    mapping = dict(zip(G.nodes(), perm))
    
    # Create isomorphic graph
    G_iso = nx.relabel_nodes(G, mapping)
    
    # Transfer node features if they exist
    for n in G_iso.nodes():
        if 'features' in G.nodes[mapping[n]]:
            G_iso.nodes[n]['features'] = G.nodes[mapping[n]]['features']
    
    return G_iso, mapping

# Example usage
G = nx.erdos_renyi_graph(10, 0.3)
G_iso, mapping = generate_isomorphic_graph(G)
is_iso, dist = test_graph_isomorphism(model, G, G_iso)
```

Slide 12: Graph Explainability Methods

Implementing explainability techniques helps understand model decisions by highlighting important subgraphs and node features that contribute most to the classification outcome.

```python
class GraphExplainer:
    def __init__(self, model, num_epochs=100, lr=0.01):
        self.model = model
        self.num_epochs = num_epochs
        self.lr = lr
        
    def explain_prediction(self, x, edge_index, target):
        # Initialize edge mask
        edge_mask = torch.ones(edge_index.size(1), requires_grad=True)
        optimizer = torch.optim.Adam([edge_mask], lr=self.lr)
        
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            
            # Apply mask to edges
            masked_edge_index = edge_index * edge_mask.unsqueeze(0)
            
            # Get prediction
            out = self.model(x, masked_edge_index)
            
            # Compute loss: prediction loss + sparsity regularization
            pred_loss = F.cross_entropy(out, target)
            mask_loss = torch.sum(torch.abs(edge_mask)) * 0.1
            loss = pred_loss + mask_loss
            
            loss.backward()
            optimizer.step()
            
            # Project masks to [0, 1]
            with torch.no_grad():
                edge_mask.data.clamp_(0, 1)
        
        return edge_mask.detach()
    
    def get_important_subgraph(self, G, edge_mask, threshold=0.5):
        important_edges = torch.where(edge_mask > threshold)[0]
        subgraph_edges = G.edges()[important_edges]
        
        subgraph = G.edge_subgraph(subgraph_edges)
        return subgraph

# Example usage
explainer = GraphExplainer(model)
edge_importance = explainer.explain_prediction(x, edge_index, target)
```

Slide 13: Performance Optimization Techniques

Implementing efficient batching and memory management strategies significantly improves training speed and model scalability for large graph datasets.

```python
class OptimizedGraphTrainer:
    def __init__(self, model, device='cuda', batch_size=32):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        
    def create_batches(self, dataset):
        random_idx = torch.randperm(len(dataset))
        batches = []
        
        for i in range(0, len(dataset), self.batch_size):
            batch_idx = random_idx[i:i + self.batch_size]
            batch = self.collate_graphs([dataset[j] for j in batch_idx])
            batches.append(batch)
        
        return batches
    
    @staticmethod
    def collate_graphs(graphs):
        batch_x = []
        batch_edge_index = []
        batch_assignment = []
        
        node_offset = 0
        for idx, graph in enumerate(graphs):
            num_nodes = graph.x.size(0)
            
            batch_x.append(graph.x)
            batch_edge_index.append(graph.edge_index + node_offset)
            batch_assignment.extend([idx] * num_nodes)
            
            node_offset += num_nodes
        
        return {
            'x': torch.cat(batch_x, dim=0),
            'edge_index': torch.cat(batch_edge_index, dim=1),
            'batch': torch.tensor(batch_assignment)
        }
    
    @torch.cuda.amp.autocast()
    def train_step(self, batch):
        self.model.train()
        
        batch = {k: v.to(self.device) for k, v in batch.items()}
        pred = self.model(batch['x'], batch['edge_index'], batch['batch'])
        
        return pred

# Example usage
trainer = OptimizedGraphTrainer(model)
batches = trainer.create_batches(dataset)
```

Slide 14: Additional Resources

*   Paper: "A Comprehensive Survey on Graph Neural Networks" - [https://arxiv.org/abs/1901.00596](https://arxiv.org/abs/1901.00596)
*   Paper: "How Powerful are Graph Neural Networks?" - [https://arxiv.org/abs/1810.00826](https://arxiv.org/abs/1810.00826)
*   Paper: "Graph Neural Networks: A Review of Methods and Applications" - [https://arxiv.org/abs/1812.08434](https://arxiv.org/abs/1812.08434)
*   Paper: "Benchmarking Graph Neural Networks" - [https://arxiv.org/abs/2003.00982](https://arxiv.org/abs/2003.00982)
*   Search Keywords for Further Research:
    *   "Graph attention networks"
    *   "Graph isomorphism networks"
    *   "Graph classification benchmarks"
    *   "Graph neural network architectures"

