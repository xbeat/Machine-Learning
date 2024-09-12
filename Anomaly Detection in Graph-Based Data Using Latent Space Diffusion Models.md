## Anomaly Detection in Graph-Based Data Using Latent Space Diffusion Models
Slide 1: Introduction to Anomaly Detection in Graph-Based Data

Anomaly detection in graph-based data is a crucial task in various domains, including social network analysis, fraud detection, and cybersecurity. This slideshow explores the use of latent space diffusion models for identifying anomalies in graph structures.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a sample graph
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)])

# Add an anomalous node
G.add_edge(7, 1)

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=16)
plt.title("Sample Graph with Anomaly")
plt.show()
```

Slide 2: Understanding Graph-Based Data

Graph-based data represents relationships between entities as nodes and edges. In this context, anomalies are unusual patterns or structures that deviate from the expected behavior of the graph.

```python
import networkx as nx

# Create a simple graph
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])

# Print basic graph properties
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes()}")
```

Slide 3: Latent Space Representation

Latent space models project graph data into a lower-dimensional space, capturing essential structural information. This representation facilitates anomaly detection by making patterns more apparent.

```python
import numpy as np
from sklearn.manifold import TSNE
import networkx as nx
import matplotlib.pyplot as plt

# Generate a random graph
G = nx.erdos_renyi_graph(50, 0.1)

# Get adjacency matrix
A = nx.adjacency_matrix(G).todense()

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
latent_space = tsne.fit_transform(A)

# Visualize latent space
plt.scatter(latent_space[:, 0], latent_space[:, 1])
plt.title("Latent Space Representation of Graph")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
```

Slide 4: Introduction to Diffusion Models

Diffusion models are generative models that learn to reverse a gradual noising process. In the context of graphs, they can capture the underlying distribution of normal graph structures.

```python
import torch
import torch.nn as nn

class SimpleDiffusionModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleDiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x, t):
        # t is the diffusion step
        return self.net(x)

# Initialize the model
input_dim = 50  # Example dimension
model = SimpleDiffusionModel(input_dim)
print(model)
```

Slide 5: Graph Diffusion Process

The graph diffusion process gradually adds noise to the graph structure, transforming it from its original state to a completely noisy state. This process is reversed during generation or anomaly detection.

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def add_noise_to_graph(G, noise_level):
    A = nx.adjacency_matrix(G).todense()
    noise = np.random.rand(*A.shape) < noise_level
    noisy_A = np.logical_xor(A, noise).astype(int)
    return nx.from_numpy_matrix(noisy_A)

# Create original graph
G = nx.erdos_renyi_graph(20, 0.2)

# Add noise
noise_levels = [0, 0.1, 0.3, 0.5]
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for i, noise in enumerate(noise_levels):
    noisy_G = add_noise_to_graph(G, noise)
    nx.draw(noisy_G, ax=axes[i], node_size=100)
    axes[i].set_title(f"Noise Level: {noise}")

plt.tight_layout()
plt.show()
```

Slide 6: Training a Graph Diffusion Model

Training involves learning to reverse the diffusion process. The model learns to predict the graph structure at each step of the denoising process.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class GraphDiffusionModel(nn.Module):
    def __init__(self, num_nodes):
        super(GraphDiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_nodes * num_nodes, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_nodes * num_nodes)
        )

    def forward(self, x, t):
        # x: flattened adjacency matrix, t: diffusion step
        return self.net(x)

# Example usage
num_nodes = 20
model = GraphDiffusionModel(num_nodes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop (simplified)
for epoch in range(100):
    optimizer.zero_grad()
    x = torch.randn(1, num_nodes * num_nodes)  # Random graph
    t = torch.randint(0, 1000, (1,))  # Random diffusion step
    output = model(x, t)
    loss = criterion(output, x)  # Simplified loss
    loss.backward()
    optimizer.step()

print("Training complete")
```

Slide 7: Anomaly Detection using Diffusion Models

Anomaly detection is performed by comparing the reconstruction quality of the diffusion model on test graphs. Graphs that are poorly reconstructed are likely to be anomalous.

```python
import torch
import networkx as nx
import numpy as np

def detect_anomalies(model, graphs, threshold):
    anomalies = []
    for i, G in enumerate(graphs):
        A = nx.adjacency_matrix(G).todense()
        x = torch.tensor(A.flatten(), dtype=torch.float32).unsqueeze(0)
        t = torch.zeros(1)  # Start from noise
        
        with torch.no_grad():
            reconstruction = model(x, t)
        
        error = torch.mean((x - reconstruction) ** 2)
        if error > threshold:
            anomalies.append(i)
    
    return anomalies

# Example usage
threshold = 0.5  # Set based on validation data
test_graphs = [nx.erdos_renyi_graph(20, 0.2) for _ in range(10)]
anomalous_indices = detect_anomalies(model, test_graphs, threshold)
print(f"Anomalous graphs: {anomalous_indices}")
```

Slide 8: Real-life Example: Social Network Analysis

In social networks, anomalies might represent fake accounts or unusual interaction patterns. Let's simulate a social network and detect anomalies.

```python
import networkx as nx
import random

def create_social_network(num_users, num_connections):
    G = nx.Graph()
    G.add_nodes_from(range(num_users))
    for _ in range(num_connections):
        u, v = random.sample(range(num_users), 2)
        G.add_edge(u, v)
    return G

def add_anomaly(G):
    # Add a node with unusually high connectivity
    anomaly_node = G.number_of_nodes()
    G.add_node(anomaly_node)
    for _ in range(int(G.number_of_nodes() * 0.5)):  # Connect to 50% of nodes
        neighbor = random.choice(list(G.nodes))
        G.add_edge(anomaly_node, neighbor)
    return G

# Create a social network
social_network = create_social_network(100, 300)
social_network = add_anomaly(social_network)

# Visualize
pos = nx.spring_layout(social_network)
nx.draw(social_network, pos, node_size=20, node_color='lightblue')
nx.draw_networkx_nodes(social_network, pos, nodelist=[100], node_color='red', node_size=50)
plt.title("Social Network with Anomaly")
plt.show()
```

Slide 9: Feature Extraction for Graph Diffusion Models

To improve the performance of diffusion models on graphs, we can extract meaningful features that capture graph properties.

```python
import networkx as nx
import numpy as np

def extract_graph_features(G):
    features = []
    for node in G.nodes():
        degree = G.degree(node)
        clustering = nx.clustering(G, node)
        centrality = nx.betweenness_centrality(G)[node]
        features.append([degree, clustering, centrality])
    return np.array(features)

# Example usage
G = nx.erdos_renyi_graph(50, 0.1)
features = extract_graph_features(G)
print("Graph features shape:", features.shape)
print("Sample features for node 0:", features[0])
```

Slide 10: Implementing the Diffusion Process

Let's implement a simplified version of the diffusion process for graph-based data.

```python
import torch
import numpy as np

def diffusion_process(x, num_steps, beta_schedule):
    x_seq = [x]
    for t in range(1, num_steps + 1):
        beta_t = beta_schedule[t-1]
        noise = torch.randn_like(x)
        x_t = torch.sqrt(1 - beta_t) * x_seq[-1] + torch.sqrt(beta_t) * noise
        x_seq.append(x_t)
    return x_seq

# Example usage
num_nodes = 20
x = torch.rand(num_nodes, num_nodes)  # Initial graph representation
num_steps = 1000
beta_schedule = np.linspace(0.0001, 0.02, num_steps)

diffused_seq = diffusion_process(x, num_steps, beta_schedule)
print(f"Number of diffusion steps: {len(diffused_seq)}")
print(f"Shape of final diffused graph: {diffused_seq[-1].shape}")
```

Slide 11: Reverse Diffusion for Anomaly Detection

The reverse diffusion process is key to anomaly detection. We'll implement a simplified version of this process.

```python
import torch
import torch.nn as nn

class ReverseGraphDiffusion(nn.Module):
    def __init__(self, num_nodes):
        super(ReverseGraphDiffusion, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_nodes * num_nodes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_nodes * num_nodes)
        )

    def forward(self, x, t):
        t_emb = torch.FloatTensor([t]).unsqueeze(0)
        x_t = torch.cat([x.flatten(), t_emb], dim=1)
        return self.net(x_t).view(x.shape)

def reverse_diffusion(model, x_T, num_steps):
    x_t = x_T
    for t in reversed(range(num_steps)):
        z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        x_t = model(x_t, t) + z
    return x_t

# Example usage
num_nodes = 20
model = ReverseGraphDiffusion(num_nodes)
x_T = torch.randn(num_nodes, num_nodes)  # Fully noised graph
num_steps = 1000

reconstructed_x = reverse_diffusion(model, x_T, num_steps)
print(f"Shape of reconstructed graph: {reconstructed_x.shape}")
```

Slide 12: Evaluating Anomaly Detection Performance

To assess the effectiveness of our anomaly detection method, we need to evaluate its performance using appropriate metrics.

```python
from sklearn.metrics import roc_auc_score, precision_recall_curve
import numpy as np

def evaluate_anomaly_detection(true_labels, anomaly_scores):
    # Calculate ROC AUC
    roc_auc = roc_auc_score(true_labels, anomaly_scores)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(true_labels, anomaly_scores)
    
    # Calculate average precision
    avg_precision = np.mean(precision)
    
    return roc_auc, avg_precision, precision, recall

# Example usage
true_labels = np.array([0, 0, 1, 0, 1, 0, 0, 1])  # 0: normal, 1: anomaly
anomaly_scores = np.array([0.1, 0.2, 0.8, 0.3, 0.7, 0.4, 0.2, 0.9])

roc_auc, avg_precision, precision, recall = evaluate_anomaly_detection(true_labels, anomaly_scores)

print(f"ROC AUC: {roc_auc:.3f}")
print(f"Average Precision: {avg_precision:.3f}")

# Plot precision-recall curve
plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```

Slide 13: Real-life Example: Fraud Detection in Financial Networks

Let's simulate a financial transaction network and use our graph diffusion model to detect potentially fraudulent activities.

```python
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

def create_financial_network(num_accounts, num_transactions):
    G = nx.Graph()
    G.add_nodes_from(range(num_accounts))
    for _ in range(num_transactions):
        sender, receiver = random.sample(range(num_accounts), 2)
        amount = random.uniform(10, 1000)
        G.add_edge(sender, receiver, amount=amount)
    return G

def add_fraudulent_activity(G):
    fraud_node = G.number_of_nodes()
    G.add_node(fraud_node)
    for _ in range(20):
        receiver = random.choice(list(G.nodes))
        amount = random.uniform(1, 10)
        G.add_edge(fraud_node, receiver, amount=amount)
    return G

# Create and visualize financial network
financial_network = create_financial_network(100, 500)
financial_network = add_fraudulent_activity(financial_network)

pos = nx.spring_layout(financial_network)
nx.draw(financial_network, pos, node_size=20, node_color='lightgreen')
nx.draw_networkx_nodes(financial_network, pos, nodelist=[100], node_color='red', node_size=50)
plt.title("Financial Transaction Network with Potential Fraud")
plt.show()

# Prepare data for anomaly detection
adj_matrix = nx.adjacency_matrix(financial_network).todense()
print(f"Adjacency matrix shape: {adj_matrix.shape}")
```

Slide 14: Implementing Anomaly Detection for Financial Networks

Now, let's implement a simple anomaly detection method for our financial network using the graph structure and transaction amounts.

```python
import numpy as np
from scipy.stats import zscore

def detect_anomalies(G, threshold=2.5):
    anomalies = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) == 0:
            continue
        
        transactions = [G[node][neighbor]['amount'] for neighbor in neighbors]
        avg_transaction = np.mean(transactions)
        num_transactions = len(transactions)
        
        # Calculate z-score for average transaction and number of transactions
        z_avg = zscore(avg_transaction)
        z_num = zscore(num_transactions)
        
        # If either z-score exceeds threshold, flag as anomaly
        if abs(z_avg) > threshold or abs(z_num) > threshold:
            anomalies.append(node)
    
    return anomalies

# Detect anomalies in the financial network
anomalous_nodes = detect_anomalies(financial_network)
print(f"Detected anomalies: {anomalous_nodes}")

# Visualize results
pos = nx.spring_layout(financial_network)
nx.draw(financial_network, pos, node_size=20, node_color='lightgreen')
nx.draw_networkx_nodes(financial_network, pos, nodelist=anomalous_nodes, node_color='red', node_size=50)
plt.title("Financial Network with Detected Anomalies")
plt.show()
```

Slide 15: Additional Resources

For further exploration of anomaly detection in graph-based data using latent space diffusion models, consider the following resources:

1. "Graph Neural Networks for Anomaly Detection in Industrial Internet of Things" (arXiv:2101.04064) URL: [https://arxiv.org/abs/2101.04064](https://arxiv.org/abs/2101.04064)
2. "Diffusion Models for Graphs: A Survey" (arXiv:2308.14209) URL: [https://arxiv.org/abs/2308.14209](https://arxiv.org/abs/2308.14209)
3. "A Survey on Graph Neural Networks and Graph Transformers in Computer Vision: A Task-Oriented Perspective" (arXiv:2209.13232) URL: [https://arxiv.org/abs/2209.13232](https://arxiv.org/abs/2209.13232)

These papers provide in-depth discussions on graph-based anomaly detection techniques and the application of diffusion models to graph data.

