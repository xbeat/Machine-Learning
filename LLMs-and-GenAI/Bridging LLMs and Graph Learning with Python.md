## Bridging LLMs and Graph Learning with Python
Slide 1: Introduction to Graph Learning and LLMs

Graph learning and Large Language Models (LLMs) are two powerful paradigms in machine learning. Graph learning focuses on extracting insights from relational data structures, while LLMs excel at processing and generating human-like text. Bridging these two domains can lead to powerful applications that leverage both structured data and natural language understanding.

```python
import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModel

# Create a simple graph
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])

# Initialize a pre-trained language model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example of combining graph and text data
text = "Node 1 is connected to Node 2"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# This is a simplified example of bridging graphs and LLMs
```

Slide 2: Graph Representation Learning

Graph representation learning aims to encode nodes, edges, or entire graphs into low-dimensional vectors while preserving structural information. These embeddings can then be used as input for various downstream tasks, including node classification, link prediction, and graph classification.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Load a dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Define a simple Graph Convolutional Network
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize the model
model = GCN()
print(model)
```

Slide 3: LLMs for Graph Understanding

LLMs can be leveraged to understand and generate natural language descriptions of graph structures. This capability allows for more intuitive interaction with graph data and can help bridge the gap between structured data and human understanding.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Example graph description
graph_description = "The graph has 4 nodes. Node 1 is connected to Node 2 and Node 4. Node 2 is connected to Node 3. Node 3 is connected to Node 4."

# Generate a summary of the graph
input_ids = tokenizer.encode(graph_description + " In summary, this graph", return_tensors="pt")
output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)

summary = tokenizer.decode(output[0], skip_special_tokens=True)
print(summary)
```

Slide 4: Graph-to-Text Generation

Graph-to-text generation involves creating natural language descriptions or summaries of graph structures. This task combines graph representation learning with sequence-to-sequence models to produce coherent textual output based on graph input.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from transformers import BertTokenizer, BertForSequenceClassification

class GraphToTextModel(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes):
        super(GraphToTextModel, self).__init__()
        self.gcn = GCNConv(num_node_features, hidden_dim)
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
        
    def forward(self, x, edge_index, text_input_ids, text_attention_mask):
        # Graph encoding
        graph_embedding = self.gcn(x, edge_index)
        graph_embedding = torch.mean(graph_embedding, dim=0, keepdim=True)
        
        # Combine graph embedding with text input
        outputs = self.bert(input_ids=text_input_ids,
                            attention_mask=text_attention_mask,
                            inputs_embeds=graph_embedding)
        
        return outputs.logits

# Usage example (pseudo-code)
# model = GraphToTextModel(num_node_features, hidden_dim, num_classes)
# output = model(x, edge_index, text_input_ids, text_attention_mask)
```

Slide 5: Text-to-Graph Generation

Text-to-graph generation involves creating graph structures from natural language descriptions. This task requires natural language understanding capabilities to extract entities and relationships from text and construct a corresponding graph representation.

```python
import spacy
import networkx as nx
import matplotlib.pyplot as plt

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def text_to_graph(text):
    doc = nlp(text)
    G = nx.Graph()
    
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ("nsubj", "dobj", "pobj"):
                G.add_edge(token.head.text, token.text, relation=token.dep_)
    
    return G

# Example usage
text = "Alice likes Bob. Bob works with Charlie. Charlie admires Alice."
graph = text_to_graph(text)

# Visualize the graph
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
edge_labels = nx.get_edge_attributes(graph, 'relation')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

plt.title("Generated Graph from Text")
plt.axis('off')
plt.tight_layout()
plt.show()
```

Slide 6: Graph Attention Networks for LLM Enhancement

Graph Attention Networks (GATs) can be used to enhance LLMs by incorporating structural information from graphs. This approach allows the model to attend to relevant parts of the graph while processing text, potentially improving performance on tasks that require both linguistic and structural understanding.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from transformers import BertModel, BertTokenizer

class GraphEnhancedBERT(nn.Module):
    def __init__(self, num_node_features, hidden_dim):
        super(GraphEnhancedBERT, self).__init__()
        self.gat = GATConv(num_node_features, hidden_dim)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fusion_layer = nn.Linear(hidden_dim + 768, 768)  # 768 is BERT's hidden size
        
    def forward(self, x, edge_index, input_ids, attention_mask):
        # Graph attention
        graph_embedding = self.gat(x, edge_index)
        graph_embedding = torch.mean(graph_embedding, dim=0, keepdim=True)
        
        # BERT encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embedding = bert_output.last_hidden_state[:, 0, :]  # CLS token
        
        # Fusion
        fused_embedding = torch.cat([graph_embedding, bert_embedding], dim=-1)
        output = self.fusion_layer(fused_embedding)
        
        return output

# Usage example (pseudo-code)
# model = GraphEnhancedBERT(num_node_features, hidden_dim)
# output = model(x, edge_index, input_ids, attention_mask)
```

Slide 7: Knowledge Graph Enrichment with LLMs

LLMs can be used to enrich knowledge graphs by generating new relationships or entities based on existing graph structures and textual information. This process can help in expanding and refining knowledge graphs with high-quality, contextually relevant information.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import networkx as nx

def enrich_knowledge_graph(G, node, context_size=2):
    # Get context from the graph
    context_nodes = list(nx.ego_graph(G, node, radius=context_size))
    context = f"Given the following context about {node}: "
    context += ", ".join([f"{n} is related to {node}" for n in context_nodes if n != node])
    
    # Generate new relationship using GPT-2
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    prompt = context + f". Suggest a new relationship for {node}: {node} is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    
    new_relation = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the generated relationship (this is a simplified extraction)
    new_relation = new_relation.split(f"{node} is")[-1].strip()
    
    return new_relation

# Example usage
G = nx.Graph()
G.add_edges_from([('Apple', 'Fruit'), ('Banana', 'Fruit'), ('Orange', 'Fruit')])

new_relation = enrich_knowledge_graph(G, 'Apple')
print(f"Generated new relation for Apple: {new_relation}")
```

Slide 8: Graph-based Question Answering with LLMs

Combining graph structures with LLMs can enhance question answering systems by leveraging both structured knowledge and natural language understanding. This approach allows for more comprehensive and accurate answers by considering both the graph context and the textual information.

```python
import networkx as nx
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def graph_based_qa(G, question, context):
    # Extract relevant subgraph based on the question
    relevant_nodes = [node for node in G.nodes() if node.lower() in question.lower()]
    subgraph = G.subgraph(relevant_nodes)
    
    # Enhance context with graph information
    graph_context = ", ".join([f"{u} is connected to {v}" for u, v in subgraph.edges()])
    enhanced_context = f"{context} Additional information from the graph: {graph_context}"
    
    # Use BERT for question answering
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    inputs = tokenizer(question, enhanced_context, return_tensors="pt")
    outputs = model(**inputs)
    
    answer_start = outputs.start_logits.argmax()
    answer_end = outputs.end_logits.argmax()
    answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end+1])
    
    return answer

# Example usage
G = nx.Graph()
G.add_edges_from([('Paris', 'France'), ('Berlin', 'Germany'), ('London', 'UK')])

question = "What is the capital of France?"
context = "France is a country in Western Europe."

answer = graph_based_qa(G, question, context)
print(f"Question: {question}")
print(f"Answer: {answer}")
```

Slide 9: Graph-based Text Classification

Graph-based text classification combines graph neural networks with text representations to improve classification accuracy. This approach can capture both the textual content and the relational information between documents or words.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import BertTokenizer, BertModel

class GraphTextClassifier(nn.Module):
    def __init__(self, num_classes, hidden_dim):
        super(GraphTextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.gcn = GCNConv(768, hidden_dim)  # 768 is BERT's hidden size
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask, edge_index):
        # BERT encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = bert_output.last_hidden_state[:, 0, :]  # CLS token
        
        # Graph convolution
        graph_embeddings = self.gcn(text_embeddings, edge_index)
        
        # Classification
        logits = self.classifier(graph_embeddings)
        return F.log_softmax(logits, dim=1)

# Example usage (pseudo-code)
# model = GraphTextClassifier(num_classes, hidden_dim)
# output = model(input_ids, attention_mask, edge_index)
```

Slide 10: Graph-based Named Entity Recognition

Graph-based Named Entity Recognition (NER) leverages graph structures to improve entity detection and classification in text. By incorporating contextual information from knowledge graphs or document co-occurrence networks, this approach can enhance traditional NER models.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from transformers import BertTokenizer, BertForTokenClassification

class GraphNER(nn.Module):
    def __init__(self, num_labels):
        super(GraphNER, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.gcn = GCNConv(768, 768)  # 768 is BERT's hidden size
        
    def forward(self, input_ids, attention_mask, edge_index):
        # BERT token classification
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Graph convolution
        graph_enhanced = self.gcn(sequence_output.view(-1, 768), edge_index)
        graph_enhanced = graph_enhanced.view(sequence_output.shape)
        
        # Combine BERT and graph outputs
        logits = self.bert.classifier(graph_enhanced)
        return logits

# Example usage (pseudo-code)
# model = GraphNER(num_labels)
# output = model(input_ids, attention_mask, edge_index)

# Tokenize example text
text = "John works at Microsoft in Seattle."
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
print(f"Tokens: {tokens}")
```

Slide 11: Graph-based Sentiment Analysis

Graph-based sentiment analysis incorporates relational information between words or documents to improve sentiment classification. This approach can capture context and semantic relationships that might be missed by traditional bag-of-words models.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import BertTokenizer, BertModel

class GraphSentimentAnalyzer(nn.Module):
    def __init__(self, hidden_dim):
        super(GraphSentimentAnalyzer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.gcn = GCNConv(768, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 3)  # 3 classes: negative, neutral, positive
        
    def forward(self, input_ids, attention_mask, edge_index):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = bert_output.last_hidden_state[:, 0, :]
        graph_embeddings = self.gcn(text_embeddings, edge_index)
        logits = self.classifier(graph_embeddings)
        return F.log_softmax(logits, dim=1)

# Example usage (pseudo-code)
# model = GraphSentimentAnalyzer(hidden_dim=128)
# output = model(input_ids, attention_mask, edge_index)
```

Slide 12: Graph-based Text Summarization

Graph-based text summarization uses graph structures to represent relationships between sentences or concepts in a document. This approach can help identify key information and generate more coherent and informative summaries.

```python
import networkx as nx
from transformers import pipeline

def graph_based_summarization(text, num_sentences=3):
    # Create a graph of sentences
    sentences = text.split('.')
    G = nx.Graph()
    
    for i, sent1 in enumerate(sentences):
        for j, sent2 in enumerate(sentences):
            if i != j:
                similarity = len(set(sent1.split()) & set(sent2.split())) / len(set(sent1.split()) | set(sent2.split()))
                G.add_edge(i, j, weight=similarity)
    
    # Calculate centrality
    centrality = nx.pagerank(G)
    
    # Select top sentences
    top_sentences = sorted(centrality, key=centrality.get, reverse=True)[:num_sentences]
    summary = ' '.join([sentences[i].strip() for i in sorted(top_sentences)])
    
    # Refine summary using a pre-trained model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    refined_summary = summarizer(summary, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    
    return refined_summary

# Example usage
text = "Graph-based summarization is a powerful technique. It uses graph structures to represent relationships between sentences. This approach can capture the overall structure of the document."
summary = graph_based_summarization(text)
print(f"Summary: {summary}")
```

Slide 13: Graph Neural Networks for Language Understanding

Graph Neural Networks (GNNs) can be applied to various natural language processing tasks by representing linguistic structures as graphs. This approach allows for the incorporation of syntactic and semantic relationships into language understanding models.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from transformers import BertTokenizer, BertModel

class GNNLanguageUnderstanding(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(GNNLanguageUnderstanding, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.gcn1 = GCNConv(768, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, input_ids, attention_mask, edge_index):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_output.last_hidden_state
        
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        
        # Global pooling
        x = torch.mean(x, dim=1)
        
        return self.classifier(x)

# Example usage (pseudo-code)
# model = GNNLanguageUnderstanding(hidden_dim=128, num_classes=10)
# output = model(input_ids, attention_mask, edge_index)
```

Slide 14: Real-life Example: Social Network Analysis

Social network analysis is a common application that bridges graph learning and natural language processing. By combining graph structures with text data from social media posts, we can gain insights into user behavior, community detection, and information spread.

```python
import networkx as nx
from transformers import pipeline

def analyze_social_network(graph, posts):
    # Create a sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis")
    
    # Analyze sentiment of posts
    sentiments = sentiment_analyzer(posts)
    
    # Add sentiment as node attribute
    for i, (node, sentiment) in enumerate(zip(graph.nodes(), sentiments)):
        graph.nodes[node]['sentiment'] = sentiment['label']
    
    # Identify influential users using PageRank
    pagerank = nx.pagerank(graph)
    influential_users = sorted(pagerank, key=pagerank.get, reverse=True)[:5]
    
    # Detect communities
    communities = list(nx.community.greedy_modularity_communities(graph))
    
    return influential_users, communities

# Example usage (pseudo-code)
# G = nx.karate_club_graph()  # Example social network
# posts = ["I love this product!", "Great service!", "Not satisfied with the quality."]
# influential_users, communities = analyze_social_network(G, posts)
```

Slide 15: Real-life Example: Recommendation Systems

Recommendation systems can benefit from the integration of graph learning and language models. By combining user interaction graphs with textual content, we can create more accurate and context-aware recommendations.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from transformers import BertTokenizer, BertModel

class GraphRecommender(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim):
        super(GraphRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.sage = SAGEConv(hidden_dim, hidden_dim)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, user_ids, item_ids, edge_index, item_descriptions):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Graph convolution
        user_emb = self.sage(user_emb, edge_index)
        item_emb = self.sage(item_emb, edge_index)
        
        # Process item descriptions
        bert_output = self.bert(**item_descriptions)
        text_emb = bert_output.last_hidden_state[:, 0, :]
        
        # Combine graph and text embeddings
        combined_emb = torch.cat([item_emb, text_emb], dim=1)
        
        # Predict ratings
        ratings = self.classifier(combined_emb)
        return ratings.squeeze()

# Example usage (pseudo-code)
# model = GraphRecommender(num_users=1000, num_items=5000, hidden_dim=128)
# ratings = model(user_ids, item_ids, edge_index, item_descriptions)
```

Slide 16: Additional Resources

For further exploration of bridging LLMs and graph learning, consider the following resources:

1. "Graph Neural Networks for Natural Language Processing" by Shikhar Vashishth et al. (2021) - ArXiv: [https://arxiv.org/abs/2106.06090](https://arxiv.org/abs/2106.06090)
2. "A Survey on Graph Neural Networks for Natural Language Processing" by Lingfei Wu et al. (2021) - ArXiv: [https://arxiv.org/abs/2106.04829](https://arxiv.org/abs/2106.04829)
3. "Graph-based Deep Learning for Communication Networks: A Survey" by Xu Wang et al. (2021) - ArXiv: [https://arxiv.org/abs/2106.02533](https://arxiv.org/abs/2106.02533)

These papers provide comprehensive overviews of the intersection between graph learning and natural language processing, offering insights into current research directions and potential applications.

