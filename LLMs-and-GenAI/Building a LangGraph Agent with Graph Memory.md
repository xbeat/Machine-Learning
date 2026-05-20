## Building a LangGraph Agent with Graph Memory
Slide 1: LangGraph Agent Architecture Overview

A LangGraph Agent combines natural language processing with graph neural networks to create an intelligent system capable of maintaining contextual memory through graph structures. The core architecture consists of an encoder, graph memory module, and decoder working in concert.

```python
class LangGraphAgent:
    def __init__(self, encoder_model, gnn_model, decoder_model):
        self.encoder = encoder_model  # Transformer-based encoder
        self.graph_memory = gnn_model # Graph neural network
        self.decoder = decoder_model  # Language generation decoder
        
    def forward(self, text_input, graph_state):
        # Encode text input
        encoded = self.encoder(text_input)
        # Update graph memory
        graph_state = self.graph_memory(encoded, graph_state)
        # Generate response
        output = self.decoder(encoded, graph_state)
        return output, graph_state
```

Slide 2: Graph Memory Implementation

The graph memory module utilizes a Graph Convolutional Network (GCN) to process and store information in a structured format. This implementation allows for efficient message passing between nodes while maintaining temporal relationships.

```python
import torch
import torch.nn as nn
import torch_geometric

class GraphMemory(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gcn = torch_geometric.nn.GCNConv(input_dim, hidden_dim)
        self.update_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x, edge_index):
        # Apply graph convolution
        h = self.gcn(x, edge_index)
        # Memory update mechanism
        h_new = torch.sigmoid(self.update_gate(torch.cat([h, x], dim=-1)))
        return h_new * h + (1 - h_new) * x
```

Slide 3: Knowledge Graph Construction

The foundation of graph memory relies on proper knowledge graph construction from text input. This implementation demonstrates how to extract entities and relationships to build a structured graph representation.

```python
from spacy import displacy
import networkx as nx
import spacy

class KnowledgeGraphBuilder:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.graph = nx.DiGraph()
        
    def extract_entities_relations(self, text):
        doc = self.nlp(text)
        for token in doc:
            if token.dep_ in ('nsubj', 'dobj'):
                subject = token.text
                verb = token.head.text
                object_ = [c.text for c in token.head.children 
                          if c.dep_ == 'dobj']
                if object_:
                    self.graph.add_edge(subject, object_[0], 
                                      relation=verb)
        return self.graph
```

Slide 4: Encoder Architecture with BERT

The encoder transforms input text into contextualized embeddings using BERT. This implementation shows how to create a custom encoder that preserves both semantic and structural information.

```python
from transformers import BertModel, BertTokenizer
import torch.nn as nn

class GraphBERTEncoder(nn.Module):
    def __init__(self, bert_model='bert-base-uncased'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert = BertModel.from_pretrained(bert_model)
        self.projection = nn.Linear(768, 256)  # Reduce dimension
        
    def forward(self, text):
        # Tokenize and encode text
        inputs = self.tokenizer(text, return_tensors='pt', 
                              padding=True, truncation=True)
        outputs = self.bert(**inputs)
        # Project to graph space
        graph_embeddings = self.projection(outputs.last_hidden_state)
        return graph_embeddings
```

Slide 5: Graph Attention Implementation

Graph attention mechanisms allow the model to weigh different parts of the graph differently when processing information. This implementation shows a multi-head graph attention layer.

```python
import torch
import torch.nn.functional as F

class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.query = nn.Linear(in_features, out_features * n_heads)
        self.key = nn.Linear(in_features, out_features * n_heads)
        self.value = nn.Linear(in_features, out_features * n_heads)
        
    def forward(self, x, adj_matrix):
        Q = self.query(x).view(-1, self.n_heads, x.size(-1))
        K = self.key(x).view(-1, self.n_heads, x.size(-1))
        V = self.value(x).view(-1, self.n_heads, x.size(-1))
        
        attention = torch.matmul(Q, K.transpose(-2, -1))
        attention = F.softmax(attention / (x.size(-1) ** 0.5), dim=-1)
        attention = attention * adj_matrix.unsqueeze(1)
        
        return torch.matmul(attention, V)
```

Slide 6: Decoder Implementation with Attention

The decoder generates responses by attending to both the encoded input and graph memory state. This implementation uses a transformer-based architecture with cross-attention mechanisms to integrate both information sources.

```python
class GraphAwareDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead), 
            num_layers=6
        )
        self.graph_attention = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, tgt, memory, graph_state):
        # Process target sequence
        tgt = self.embedding(tgt)
        # Attend to encoder memory
        decoder_output = self.transformer_decoder(tgt, memory)
        # Attend to graph state
        graph_context, _ = self.graph_attention(
            decoder_output, graph_state, graph_state
        )
        # Combine and generate output
        output = self.output_layer(decoder_output + graph_context)
        return output
```

Slide 7: Training Loop Implementation

The training process involves optimizing both the language modeling and graph representation objectives simultaneously. This implementation shows how to handle the dual nature of the learning process.

```python
def train_langgraph_agent(agent, train_loader, epochs, device):
    optimizer = torch.optim.Adam(agent.parameters())
    text_criterion = nn.CrossEntropyLoss()
    graph_criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            text_input, graph_input, targets = batch
            text_input = text_input.to(device)
            graph_input = graph_input.to(device)
            
            # Forward pass
            text_output, graph_state = agent(text_input, graph_input)
            
            # Calculate losses
            text_loss = text_criterion(text_output, targets)
            graph_loss = graph_criterion(
                graph_state, 
                agent.graph_memory.get_target_state(graph_input)
            )
            loss = text_loss + 0.5 * graph_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch}: Loss = {total_loss/len(train_loader)}")
```

Slide 8: Graph State Management

Efficient management of graph states is crucial for maintaining memory across interactions. This implementation provides mechanisms for updating and pruning the graph structure.

```python
class GraphStateManager:
    def __init__(self, max_nodes=1000, relevance_threshold=0.5):
        self.graph_state = nx.DiGraph()
        self.max_nodes = max_nodes
        self.threshold = relevance_threshold
        
    def update_state(self, new_nodes, new_edges, relevance_scores):
        # Add new information
        for node, score in zip(new_nodes, relevance_scores):
            if score > self.threshold:
                self.graph_state.add_node(
                    node, 
                    relevance=score,
                    timestamp=time.time()
                )
                
        # Add new edges
        for edge in new_edges:
            source, target, weight = edge
            if (source in self.graph_state and 
                target in self.graph_state):
                self.graph_state.add_edge(
                    source, target, weight=weight
                )
        
        # Prune graph if necessary
        if len(self.graph_state) > self.max_nodes:
            self._prune_graph()
    
    def _prune_graph(self):
        nodes = sorted(
            self.graph_state.nodes(data=True),
            key=lambda x: (x[1]['relevance'], -x[1]['timestamp'])
        )
        nodes_to_remove = nodes[:(len(nodes) - self.max_nodes)]
        self.graph_state.remove_nodes_from(
            [node[0] for node in nodes_to_remove]
        )
```

Slide 9: Real-world Example - Question Answering System

Implementation of a question answering system leveraging graph memory for enhanced context awareness and answer generation across multiple queries.

```python
class GraphQASystem:
    def __init__(self, agent, graph_manager):
        self.agent = agent
        self.graph_manager = graph_manager
        self.context_history = []
        
    def answer_question(self, question, context=None):
        # Update context history
        if context:
            self.context_history.append(context)
            # Extract entities and update graph
            entities = self._extract_entities(context)
            self.graph_manager.update_state(
                entities,
                self._extract_relations(context),
                self._calculate_relevance(entities, question)
            )
        
        # Generate answer using graph-enhanced context
        graph_state = self.graph_manager.get_current_state()
        answer, new_state = self.agent(
            question,
            graph_state
        )
        
        # Update graph with new information
        self.graph_manager.update_state_from_response(
            answer, new_state
        )
        
        return answer
    
    def _extract_entities(self, text):
        # Entity extraction implementation
        pass
        
    def _extract_relations(self, text):
        # Relation extraction implementation
        pass
        
    def _calculate_relevance(self, entities, question):
        # Relevance scoring implementation
        pass
```

Slide 10: Mathematical Foundations of Graph Neural Networks

The theoretical foundations of graph neural networks in LangGraph agents involve message passing and node updating mechanisms. These equations define the core operations.

```python
# Mathematical foundations of GNNs
"""
Message Passing Equation:
$$m_i^{(l+1)} = \sum_{j \in \mathcal{N}(i)} M_l(h_i^l, h_j^l, e_{ij})$$

Node Update Equation:
$$h_i^{(l+1)} = U_l(h_i^l, m_i^{(l+1)})$$

Attention Mechanism:
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

Graph State Update:
$$G_{t+1} = F(G_t, X_t, A_t)$$
"""
```

Slide 11: Real-world Example - Recommendation System

A practical implementation of a recommendation system using LangGraph Agent to provide personalized suggestions based on user interactions and item relationships stored in the graph memory.

```python
class GraphRecommender:
    def __init__(self, agent, num_items, embedding_dim=128):
        self.agent = agent
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.graph_memory = nx.DiGraph()
        
    def update_user_preferences(self, user_id, item_id, rating):
        # Update user-item interaction graph
        user_embed = self.user_embeddings(torch.tensor([user_id]))
        item_embed = self.item_embeddings(torch.tensor([item_id]))
        
        # Add interaction to graph memory
        self.graph_memory.add_edge(
            f"user_{user_id}",
            f"item_{item_id}",
            weight=rating,
            embedding=(user_embed + item_embed) / 2
        )
        
        # Update item relationships
        self._update_item_relationships(item_id)
        
    def get_recommendations(self, user_id, n_recommendations=5):
        user_node = f"user_{user_id}"
        if user_node not in self.graph_memory:
            return self._cold_start_recommendations(n_recommendations)
            
        # Get user's interaction subgraph
        user_items = list(self.graph_memory.neighbors(user_node))
        
        # Use graph memory to find similar items
        recommendations = []
        for item in user_items:
            similar_items = self._find_similar_items(item)
            recommendations.extend(similar_items)
            
        # Sort and filter recommendations
        return sorted(
            set(recommendations),
            key=lambda x: self._calculate_score(user_id, x)
        )[:n_recommendations]
        
    def _update_item_relationships(self, item_id):
        item_node = f"item_{item_id}"
        item_users = list(self.graph_memory.predecessors(item_node))
        
        for user in item_users:
            similar_items = list(self.graph_memory.neighbors(user))
            for similar_item in similar_items:
                if similar_item != item_node:
                    self._update_item_similarity(item_id, similar_item)
```

Slide 12: Performance Metrics Implementation

Comprehensive evaluation metrics for LangGraph Agents, including both language modeling and graph-based performance measures.

```python
class LangGraphMetrics:
    def __init__(self):
        self.text_metrics = {
            'perplexity': [],
            'bleu_score': [],
            'rouge_score': []
        }
        self.graph_metrics = {
            'node_coverage': [],
            'edge_precision': [],
            'memory_utilization': []
        }
        
    def calculate_metrics(self, agent, test_data):
        text_results = self._evaluate_text_performance(
            agent, test_data.text_samples
        )
        graph_results = self._evaluate_graph_performance(
            agent, test_data.graph_samples
        )
        
        return {
            'text_metrics': {
                metric: np.mean(scores) 
                for metric, scores in self.text_metrics.items()
            },
            'graph_metrics': {
                metric: np.mean(scores) 
                for metric, scores in self.graph_metrics.items()
            }
        }
        
    def _evaluate_text_performance(self, agent, samples):
        for sample in samples:
            pred, _ = agent(sample.input, sample.graph)
            self.text_metrics['perplexity'].append(
                torch.exp(self._calculate_loss(pred, sample.target))
            )
            # Additional metrics calculation...
        
    def _evaluate_graph_performance(self, agent, samples):
        for sample in samples:
            _, graph_state = agent(sample.input, sample.initial_graph)
            self.graph_metrics['node_coverage'].append(
                self._calculate_node_coverage(
                    graph_state, sample.target_graph
                )
            )
            # Additional metrics calculation...
```

Slide 13: Graph Memory Optimization

Advanced techniques for optimizing graph memory usage and retrieval efficiency in LangGraph Agents through pruning and indexing strategies.

```python
class OptimizedGraphMemory:
    def __init__(self, max_nodes=5000):
        self.memory_index = faiss.IndexFlatL2(256)  # Vector similarity index
        self.node_embeddings = {}
        self.temporal_index = {}
        self.max_nodes = max_nodes
        
    def insert_node(self, node_id, embedding, timestamp):
        if len(self.node_embeddings) >= self.max_nodes:
            self._prune_memory()
            
        self.node_embeddings[node_id] = embedding
        self.temporal_index[node_id] = timestamp
        self.memory_index.add(
            embedding.reshape(1, -1).numpy()
        )
        
    def query_similar_nodes(self, query_embedding, k=5):
        D, I = self.memory_index.search(
            query_embedding.reshape(1, -1).numpy(), k
        )
        return [
            (list(self.node_embeddings.keys())[idx], dist) 
            for dist, idx in zip(D[0], I[0])
        ]
        
    def _prune_memory(self):
        # Remove oldest nodes when memory limit is reached
        sorted_nodes = sorted(
            self.temporal_index.items(),
            key=lambda x: x[1]
        )
        nodes_to_remove = sorted_nodes[:len(sorted_nodes)//4]
        
        for node_id, _ in nodes_to_remove:
            del self.node_embeddings[node_id]
            del self.temporal_index[node_id]
            
        # Rebuild index
        self._rebuild_index()
```

Slide 14: Additional Resources

1.  [https://arxiv.org/abs/2306.02257](https://arxiv.org/abs/2306.02257) - "Graph Neural Networks for Natural Language Processing: A Survey"
2.  [https://arxiv.org/abs/2308.07134](https://arxiv.org/abs/2308.07134) - "Memory-Augmented Graph Neural Networks for Sequential Recommendation"
3.  [https://arxiv.org/abs/2305.15317](https://arxiv.org/abs/2305.15317) - "Large Language Models Meet Graph Neural Networks: A Survey"
4.  [https://arxiv.org/abs/2309.00343](https://arxiv.org/abs/2309.00343) - "Graph of Thoughts: Solving Elaborate Problems with Large Language Models"
5.  [https://arxiv.org/abs/2310.01260](https://arxiv.org/abs/2310.01260) - "Knowledge Graphs and Graph Neural Networks for Large Language Models: A Survey"

