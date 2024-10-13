## Graph-Based Zero-Shot Learning with Contrastive Learning
Slide 1: Graph-Based Zero-Shot Learning and Contrastive Learning

Graph-based zero-shot learning and contrastive learning are two powerful techniques in machine learning. This presentation explores their intersection and demonstrates how they can be combined using Python to create more robust and versatile models.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple graph
G = nx.Graph()
G.add_edges_from([('Zero-Shot Learning', 'Graph-Based'), ('Contrastive Learning', 'Graph-Based')])

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')
plt.title("Intersection of Zero-Shot Learning and Contrastive Learning")
plt.show()
```

Slide 2: Understanding Graph-Based Zero-Shot Learning

Graph-based zero-shot learning leverages graph structures to represent relationships between seen and unseen classes. This approach allows models to make predictions on classes they haven't been explicitly trained on by exploiting the graph's connectivity.

```python
import networkx as nx

# Create a knowledge graph
G = nx.Graph()
G.add_edges_from([
    ('animal', 'mammal'), ('animal', 'bird'),
    ('mammal', 'dog'), ('mammal', 'cat'),
    ('bird', 'eagle'), ('bird', 'penguin')
])

# Function to find path between two nodes
def find_path(graph, start, end):
    return nx.shortest_path(graph, start, end)

# Example: Finding path from 'animal' to 'dog'
path = find_path(G, 'animal', 'dog')
print(f"Path from 'animal' to 'dog': {' -> '.join(path)}")
```

Slide 3: Contrastive Learning Basics

Contrastive learning is a self-supervised learning technique that learns representations by contrasting similar and dissimilar samples. It aims to push similar samples closer together in the embedding space while pushing dissimilar samples apart.

```python
import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# Example usage
embedding1 = torch.randn(10, 128)  # 10 samples, 128-dim embedding
embedding2 = torch.randn(10, 128)
labels = torch.randint(0, 2, (10,))  # 0: similar, 1: dissimilar

criterion = ContrastiveLoss()
loss = criterion(embedding1, embedding2, labels)
print(f"Contrastive Loss: {loss.item()}")
```

Slide 4: Combining Graph-Based Zero-Shot Learning and Contrastive Learning

By integrating graph-based zero-shot learning with contrastive learning, we can create more powerful models that can generalize to unseen classes while learning robust representations. This combination allows for better knowledge transfer and improved performance on both seen and unseen classes.

```python
import torch
import torch.nn as nn
import networkx as nx

class GraphContrastiveModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, graph):
        super(GraphContrastiveModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        self.graph = graph

    def forward(self, x):
        return self.encoder(x)

    def graph_similarity(self, class1, class2):
        return 1 / (nx.shortest_path_length(self.graph, class1, class2) + 1)

# Create a simple knowledge graph
G = nx.Graph()
G.add_edges_from([('animal', 'mammal'), ('animal', 'bird'), ('mammal', 'dog'), ('bird', 'eagle')])

# Initialize the model
model = GraphContrastiveModel(input_dim=100, embedding_dim=64, graph=G)

# Example usage
x1 = torch.randn(1, 100)
x2 = torch.randn(1, 100)
emb1 = model(x1)
emb2 = model(x2)

# Calculate graph-based similarity
sim = model.graph_similarity('dog', 'eagle')
print(f"Graph-based similarity between 'dog' and 'eagle': {sim}")
```

Slide 5: Feature Extraction for Graph-Based Zero-Shot Learning

In graph-based zero-shot learning, feature extraction plays a crucial role in representing classes and their attributes. We can use pre-trained models or custom architectures to extract meaningful features from input data.

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load a pre-trained ResNet model
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Remove the last fully connected layer
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features from an image
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(image)
    return features.squeeze()

# Example usage
image_path = "path/to/your/image.jpg"
features = extract_features(image_path)
print(f"Extracted feature shape: {features.shape}")
```

Slide 6: Building a Graph-Based Knowledge Base

To implement graph-based zero-shot learning, we need to create a knowledge base that represents relationships between classes and their attributes. This graph structure will guide the learning process and enable inference on unseen classes.

```python
import networkx as nx
import matplotlib.pyplot as plt

class KnowledgeGraph:
    def __init__(self):
        self.G = nx.Graph()

    def add_class(self, class_name, attributes):
        self.G.add_node(class_name, type='class')
        for attr in attributes:
            self.G.add_node(attr, type='attribute')
            self.G.add_edge(class_name, attr)

    def visualize(self):
        pos = nx.spring_layout(self.G)
        class_nodes = [node for node, data in self.G.nodes(data=True) if data['type'] == 'class']
        attr_nodes = [node for node, data in self.G.nodes(data=True) if data['type'] == 'attribute']
        
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(self.G, pos, nodelist=class_nodes, node_color='lightblue', node_size=500)
        nx.draw_networkx_nodes(self.G, pos, nodelist=attr_nodes, node_color='lightgreen', node_size=300)
        nx.draw_networkx_edges(self.G, pos)
        nx.draw_networkx_labels(self.G, pos)
        plt.title("Knowledge Graph for Zero-Shot Learning")
        plt.axis('off')
        plt.show()

# Example usage
kg = KnowledgeGraph()
kg.add_class('dog', ['furry', 'barks', 'loyal'])
kg.add_class('cat', ['furry', 'meows', 'independent'])
kg.add_class('bird', ['flies', 'has_feathers', 'lays_eggs'])

kg.visualize()
```

Slide 7: Implementing Contrastive Learning for Feature Embeddings

Contrastive learning can be used to improve the quality of feature embeddings in our graph-based zero-shot learning model. By learning to distinguish between similar and dissimilar samples, we can create more discriminative representations.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ContrastiveEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(ContrastiveEmbedding, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)

def contrastive_loss(embedding1, embedding2, label, margin=1.0):
    euclidean_distance = nn.functional.pairwise_distance(embedding1, embedding2)
    loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                      (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss

# Example usage
input_dim = 100
embedding_dim = 64
model = ContrastiveEmbedding(input_dim, embedding_dim)
optimizer = optim.Adam(model.parameters())

# Simulated data
x1 = torch.randn(32, input_dim)
x2 = torch.randn(32, input_dim)
labels = torch.randint(0, 2, (32,))  # 0: similar, 1: dissimilar

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    emb1 = model(x1)
    emb2 = model(x2)
    loss = contrastive_loss(emb1, emb2, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

Slide 8: Graph Neural Networks for Zero-Shot Learning

Graph Neural Networks (GNNs) can be powerful tools for graph-based zero-shot learning. They can propagate information through the knowledge graph, allowing for effective knowledge transfer between seen and unseen classes.

```python
import torch
import torch.nn as nn
import torch_geometric.nn as geo_nn
from torch_geometric.data import Data

class GNNZeroShot(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GNNZeroShot, self).__init__()
        self.conv1 = geo_nn.GCNConv(num_features, hidden_dim)
        self.conv2 = geo_nn.GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Example usage
num_nodes = 5
num_features = 10
hidden_dim = 16
num_classes = 3

# Create a random graph
edge_index = torch.randint(0, num_nodes, (2, 10))
x = torch.randn(num_nodes, num_features)

# Create a PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index)

# Initialize and use the model
model = GNNZeroShot(num_features, hidden_dim, num_classes)
output = model(data)
print(f"Output shape: {output.shape}")
```

Slide 9: Combining Graph-Based and Visual Features

In many zero-shot learning scenarios, we need to combine graph-based knowledge with visual features. This fusion allows the model to leverage both semantic relationships and visual similarities for more accurate predictions on unseen classes.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class VisualGraphFusion(nn.Module):
    def __init__(self, num_classes, graph_dim):
        super(VisualGraphFusion, self).__init__()
        # Visual feature extractor (pre-trained ResNet)
        resnet = models.resnet50(pretrained=True)
        self.visual_features = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_projection = nn.Linear(2048, 512)
        
        # Graph feature processor
        self.graph_projection = nn.Linear(graph_dim, 512)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, graph_features):
        # Extract visual features
        visual_features = self.visual_features(image).squeeze()
        visual_features = self.visual_projection(visual_features)
        
        # Process graph features
        graph_features = self.graph_projection(graph_features)
        
        # Concatenate and fuse features
        fused_features = torch.cat((visual_features, graph_features), dim=1)
        output = self.fusion(fused_features)
        return output

# Example usage
num_classes = 10
graph_dim = 50
model = VisualGraphFusion(num_classes, graph_dim)

# Simulated inputs
batch_size = 4
image = torch.randn(batch_size, 3, 224, 224)
graph_features = torch.randn(batch_size, graph_dim)

output = model(image, graph_features)
print(f"Output shape: {output.shape}")
```

Slide 10: Training a Graph-Based Zero-Shot Model with Contrastive Loss

This slide demonstrates how to train a graph-based zero-shot learning model using contrastive loss. The approach combines graph structure with contrastive learning to improve feature representations for both seen and unseen classes.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class GraphContrastiveModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_classes):
        super(GraphContrastiveModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding, self.classifier(embedding)

def graph_contrastive_loss(embeddings, labels, adj_matrix, margin=1.0, lambda_contrast=0.5):
    distances = torch.cdist(embeddings, embeddings)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    neg_mask = 1 - pos_mask
    contrastive_loss = (pos_mask * distances.pow(2) + 
                        neg_mask * torch.clamp(margin - distances, min=0).pow(2)).mean()
    graph_loss = torch.mean(adj_matrix * distances)
    return lambda_contrast * contrastive_loss + (1 - lambda_contrast) * graph_loss

# Training loop
model = GraphContrastiveModel(input_dim=100, embedding_dim=64, num_classes=10)
optimizer = optim.Adam(model.parameters())
adj_matrix = torch.rand(32, 32)  # Simulated adjacency matrix

for epoch in range(10):
    x = torch.randn(32, 100)  # Simulated input data
    labels = torch.randint(0, 10, (32,))  # Simulated labels
    
    optimizer.zero_grad()
    embeddings, _ = model(x)
    loss = graph_contrastive_loss(embeddings, labels, adj_matrix)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

Slide 11: Evaluating Zero-Shot Learning Performance

Evaluating zero-shot learning models requires careful consideration of both seen and unseen classes. This slide demonstrates how to assess the model's performance on unseen classes using a simple evaluation metric.

```python
import numpy as np
from sklearn.metrics import accuracy_score

def evaluate_zero_shot(model, test_data, test_labels, unseen_classes):
    model.eval()
    with torch.no_grad():
        embeddings, predictions = model(test_data)
        
    # Separate seen and unseen class samples
    unseen_mask = np.isin(test_labels.numpy(), unseen_classes)
    seen_mask = ~unseen_mask
    
    # Calculate accuracy for seen and unseen classes
    seen_acc = accuracy_score(test_labels[seen_mask], predictions.argmax(dim=1)[seen_mask])
    unseen_acc = accuracy_score(test_labels[unseen_mask], predictions.argmax(dim=1)[unseen_mask])
    
    harmony_mean = 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc)
    
    return {
        "Seen Accuracy": seen_acc,
        "Unseen Accuracy": unseen_acc,
        "Harmony Mean": harmony_mean
    }

# Example usage
test_data = torch.randn(100, 100)  # Simulated test data
test_labels = torch.randint(0, 15, (100,))  # Simulated test labels
unseen_classes = [10, 11, 12, 13, 14]  # Classes not seen during training

results = evaluate_zero_shot(model, test_data, test_labels, unseen_classes)
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
```

Slide 12: Real-Life Example: Image Classification with Zero-Shot Learning

This example demonstrates how graph-based zero-shot learning can be applied to image classification tasks, allowing the model to classify images of unseen objects based on their relationships to known classes.

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

class ImageZeroShotClassifier:
    def __init__(self, model, class_attributes, transform):
        self.model = model
        self.class_attributes = class_attributes
        self.transform = transform

    def classify(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            image_embedding, _ = self.model(image_tensor)
        
        similarities = torch.mm(image_embedding, self.class_attributes.t())
        predicted_class = similarities.argmax(dim=1).item()
        return predicted_class

# Example usage (assuming model and class_attributes are pre-trained)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classifier = ImageZeroShotClassifier(model, class_attributes, transform)
image_path = "path/to/unseen_object_image.jpg"
predicted_class = classifier.classify(image_path)
print(f"Predicted class for the unseen object: {predicted_class}")
```

Slide 13: Real-Life Example: Text Classification with Zero-Shot Learning

This example shows how graph-based zero-shot learning can be applied to text classification tasks, enabling the model to classify documents into unseen categories based on their semantic relationships.

```python
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class TextZeroShotClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TextZeroShotClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

def classify_text(model, tokenizer, text, class_names):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return class_names[predicted_class]

# Example usage
model = TextZeroShotClassifier(num_classes=5)  # Assume 5 classes for this example
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
class_names = ['Technology', 'Sports', 'Politics', 'Entertainment', 'Science']

text = "The latest smartphone features an advanced AI chip for improved performance."
predicted_class = classify_text(model, tokenizer, text, class_names)
print(f"Predicted class for the text: {predicted_class}")
```

Slide 14: Additional Resources

For those interested in delving deeper into graph-based zero-shot learning and contrastive learning, here are some valuable resources:

1. "Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly" by Y. Xian et al. (2018) ArXiv link: [https://arxiv.org/abs/1707.00600](https://arxiv.org/abs/1707.00600)
2. "A Survey of Zero-Shot Learning: Settings, Methods, and Applications" by W. Wang et al. (2019) ArXiv link: [https://arxiv.org/abs/1707.00600](https://arxiv.org/abs/1707.00600)
3. "Contrastive Learning for Unpaired Image-to-Image Translation" by T. Park et al. (2020) ArXiv link: [https://arxiv.org/abs/2007.15651](https://arxiv.org/abs/2007.15651)
4. "A Simple Framework for Contrastive Learning of Visual Representations" by T. Chen et al. (2020) ArXiv link: [https://arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709)

These papers provide in-depth explanations and methodologies related to the topics covered in this presentation.

