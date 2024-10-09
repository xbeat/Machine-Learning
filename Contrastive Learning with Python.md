## Contrastive Learning with Python
Slide 1: Introduction to Contrastive Learning

Contrastive Learning is a machine learning technique that aims to learn representations by comparing similar and dissimilar samples. It's particularly useful in self-supervised learning scenarios where labeled data is scarce. This approach encourages the model to bring similar samples closer in the embedding space while pushing dissimilar samples apart.

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
```

Slide 2: Key Concepts in Contrastive Learning

Contrastive Learning revolves around the idea of learning by comparison. It involves creating positive pairs (similar samples) and negative pairs (dissimilar samples). The model learns to minimize the distance between positive pairs while maximizing the distance between negative pairs in the embedding space.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
positive_pairs = np.random.randn(100, 2)
negative_pairs = np.random.randn(100, 2) + np.array([4, 4])

plt.scatter(positive_pairs[:, 0], positive_pairs[:, 1], c='blue', label='Positive Pairs')
plt.scatter(negative_pairs[:, 0], negative_pairs[:, 1], c='red', label='Negative Pairs')
plt.legend()
plt.title('Visualization of Positive and Negative Pairs')
plt.show()
```

Slide 3: Siamese Networks

Siamese Networks are a common architecture used in Contrastive Learning. They consist of two identical neural networks that share weights. These networks process a pair of inputs and produce embeddings that can be compared using a distance metric.

```python
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 28x28x1 -> 19x19x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 19x19x64 -> 9x9x64
            nn.Conv2d(64, 128, 7),  # 9x9x64 -> 3x3x128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 3x3x128 -> 1x1x128
        )
        self.fc = nn.Linear(128, 64)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2
```

Slide 4: Data Augmentation in Contrastive Learning

Data augmentation plays a crucial role in Contrastive Learning. It helps create diverse positive pairs by applying various transformations to the same input. This process enhances the model's ability to learn robust representations.

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_positive_pair(image):
    return transform(image), transform(image)

# Usage
# positive_pair = get_positive_pair(original_image)
```

Slide 5: Contrastive Predictive Coding (CPC)

Contrastive Predictive Coding is a self-supervised learning technique that uses contrastive learning to predict future representations in a latent space. It's particularly useful for sequential data like audio or text.

```python
import torch
import torch.nn as nn

class CPCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, context_dim, num_steps):
        super(CPCModel, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.context = nn.GRU(hidden_dim, context_dim)
        self.predictors = nn.ModuleList([nn.Linear(context_dim, hidden_dim) for _ in range(num_steps)])
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        z = self.encoder(x)
        c, _ = self.context(z)
        preds = [pred(c[:, -1]) for pred in self.predictors]
        return z, torch.stack(preds, dim=1)

# Usage
# model = CPCModel(input_dim=10, hidden_dim=64, context_dim=128, num_steps=5)
# z, preds = model(torch.randn(32, 20, 10))  # batch_size=32, seq_len=20, input_dim=10
```

Slide 6: SimCLR: A Simple Framework for Contrastive Learning

SimCLR (Simple Framework for Contrastive Learning of Visual Representations) is a popular contrastive learning method for visual representations. It uses data augmentation, a neural network encoder, and a small neural network projection head to learn representations.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class SimCLR(nn.Module):
    def __init__(self, base_model='resnet18', proj_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = models.__dict__[base_model](pretrained=False)
        dim_mlp = self.encoder.fc.in_features
        self.encoder.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, proj_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# Create SimCLR model
model = SimCLR('resnet18', proj_dim=128)

# Generate random input
x = torch.randn(32, 3, 224, 224)

# Forward pass
output = model(x)
print(f"Output shape: {output.shape}")  # Expected: torch.Size([32, 128])
```

Slide 7: NT-Xent Loss Function

The NT-Xent (Normalized Temperature-scaled Cross Entropy) loss is commonly used in contrastive learning, particularly in SimCLR. It encourages the model to maximize agreement between differently augmented views of the same image.

```python
import torch
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    sim_i_j = torch.diag(sim, N)
    sim_j_i = torch.diag(sim, -N)
    positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(N, 2)
    negative_samples = sim[torch.logical_not(torch.eye(2*N, dtype=bool))].reshape(2*N, -1)
    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat([positive_samples, negative_samples], dim=1)
    loss = F.cross_entropy(logits, labels)
    return loss

# Usage
z1 = torch.randn(128, 64)  # 128 samples, 64-dim embeddings
z2 = torch.randn(128, 64)  # Augmented views of the same samples
loss = nt_xent_loss(z1, z2)
print(f"NT-Xent Loss: {loss.item()}")
```

Slide 8: Momentum Contrast (MoCo)

Momentum Contrast is another contrastive learning framework that maintains a dynamic dictionary with a queue and a moving-averaged encoder. This approach allows for a large and consistent dictionary on-the-fly.

```python
import torch
import torch.nn as nn

class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data._(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

# Usage
# model = MoCo(base_encoder=models.resnet18)
# im_q = torch.randn(32, 3, 224, 224)
# im_k = torch.randn(32, 3, 224, 224)
# logits, labels = model(im_q, im_k)
```

Slide 9: Contrastive Learning for Text Data

Contrastive Learning can also be applied to text data. Here's an example of how to create positive pairs for text using simple augmentation techniques.

```python
import random
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def augment_text(text, p=0.1):
    words = text.split()
    augmented_words = []
    for word in words:
        if random.random() < p:
            synonyms = get_synonyms(word)
            if synonyms:
                augmented_words.append(random.choice(synonyms))
            else:
                augmented_words.append(word)
        else:
            augmented_words.append(word)
    return ' '.join(augmented_words)

# Example usage
original_text = "The quick brown fox jumps over the lazy dog"
augmented_text = augment_text(original_text)

print(f"Original: {original_text}")
print(f"Augmented: {augmented_text}")
```

Slide 10: Contrastive Learning for Audio Data

Contrastive Learning can be applied to audio data as well. Here's an example of how to create positive pairs for audio using time shifting and pitch shifting.

```python
import numpy as np
import librosa

def augment_audio(audio, sr, time_shift_max=0.1, pitch_shift_max=2):
    # Time shifting
    shift_amount = int(sr * np.random.uniform(-time_shift_max, time_shift_max))
    augmented_audio = np.roll(audio, shift_amount)
    
    # Pitch shifting
    pitch_shift = np.random.uniform(-pitch_shift_max, pitch_shift_max)
    augmented_audio = librosa.effects.pitch_shift(augmented_audio, sr=sr, n_steps=pitch_shift)
    
    return augmented_audio

# Load an audio file
audio, sr = librosa.load('path_to_audio_file.wav')

# Create a positive pair
augmented_audio = augment_audio(audio, sr)

# Plot the waveforms
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(2, 1, 1)
librosa.display.waveshow(audio, sr=sr)
plt.title('Original Audio')
plt.subplot(2, 1, 2)
librosa.display.waveshow(augmented_audio, sr=sr)
plt.title('Augmented Audio')
plt.tight_layout()
plt.show()
```

Slide 11: Contrastive Learning for Graph Data

Contrastive Learning can also be applied to graph-structured data. Here's an example of how to create positive pairs for graph data using node dropping and edge perturbation.

```python
import networkx as nx
import random

def augment_graph(G, node_drop_rate=0.1, edge_perturb_rate=0.1):
    G_aug = G.()
    
    # Node dropping
    nodes = list(G_aug.nodes())
    nodes_to_remove = random.sample(nodes, int(node_drop_rate * len(nodes)))
    G_aug.remove_nodes_from(nodes_to_remove)
    
    # Edge perturbation
    edges = list(G_aug.edges())
    edges_to_remove = random.sample(edges, int(edge_perturb_rate * len(edges)))
    G_aug.remove_edges_from(edges_to_remove)
    
    # Add new random edges
    non_edges = list(nx.non_edges(G_aug))
    edges_to_add = random.sample(non_edges, len(edges_to_remove))
    G_aug.add_edges_from(edges_to_add)
    
    return G_aug

# Example usage
G = nx.erdos_renyi_graph(n=20, p=0.2)
G_augmented = augment_graph(G)

print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Augmented graph: {G_augmented.number_of_nodes()} nodes, {G_augmented.number_of_edges()} edges")
```

Slide 12: Contrastive Learning in Unsupervised Feature Learning

Contrastive Learning has shown remarkable success in unsupervised feature learning. This approach allows models to learn meaningful representations without labeled data, which is particularly useful when labeled data is scarce or expensive to obtain.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class ContrastiveModel(nn.Module):
    def __init__(self, base_model='resnet18', feature_dim=128):
        super(ContrastiveModel, self).__init__()
        self.encoder = models.__dict__[base_model](pretrained=False)
        self.encoder.fc = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

# Data augmentation for creating positive pairs
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create model and optimizer
model = ContrastiveModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# Training loop (pseudo-code)
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         x1, x2 = transform(batch), transform(batch)
#         z1, z2 = model(x1), model(x2)
#         loss = nt_xent_loss(z1, z2)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
```

Slide 13: Evaluation of Contrastive Learning Models

Evaluating contrastive learning models often involves downstream tasks. One common approach is to use the learned representations for transfer learning on a supervised task.

```python
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def extract_features(model, dataloader):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch, label in dataloader:
            feature = model(batch)
            features.append(feature.cpu().numpy())
            labels.append(label.cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)

# Assume 'model' is our pretrained contrastive learning model
# and 'train_loader' and 'test_loader' are our data loaders

# Extract features
train_features, train_labels = extract_features(model, train_loader)
test_features, test_labels = extract_features(model, test_loader)

# Train a linear classifier
classifier = LogisticRegression(random_state=0, max_iter=1000)
classifier.fit(train_features, train_labels)

# Evaluate
predictions = classifier.predict(test_features)
accuracy = accuracy_score(test_labels, predictions)
print(f"Linear evaluation accuracy: {accuracy:.4f}")
```

Slide 14: Real-life Example: Image Similarity Search

Contrastive Learning can be applied to create an image similarity search system. This is useful in various applications such as content-based image retrieval or duplicate image detection.

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Identity()  # Remove the final fully connected layer
        
    def forward(self, x):
        return self.model(x)

# Load and preprocess image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Encode images
encoder = ImageEncoder()
encoder.eval()

def encode_image(image_path):
    with torch.no_grad():
        image = preprocess_image(image_path)
        return encoder(image).squeeze().numpy()

# Compute similarity
def compute_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Example usage
image1_embedding = encode_image('path_to_image1.jpg')
image2_embedding = encode_image('path_to_image2.jpg')
similarity = compute_similarity(image1_embedding, image2_embedding)
print(f"Similarity between images: {similarity:.4f}")
```

Slide 15: Real-life Example: Anomaly Detection

Contrastive Learning can be used for anomaly detection in various domains. Here's a simple example of how it could be applied to detect anomalies in time series data.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn

class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TimeSeriesEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))

# Generate synthetic data (replace with real data in practice)
normal_data = np.random.randn(1000, 50, 1)  # 1000 normal sequences
anomaly_data = np.random.randn(100, 50, 1) + 2  # 100 anomaly sequences

# Preprocess data
scaler = StandardScaler()
normal_data_scaled = scaler.fit_transform(normal_data.reshape(-1, 1)).reshape(1000, 50, 1)
anomaly_data_scaled = scaler.transform(anomaly_data.reshape(-1, 1)).reshape(100, 50, 1)

# Split data
train_data, test_data = train_test_split(normal_data_scaled, test_size=0.2, random_state=42)

# Train encoder (pseudo-code)
encoder = TimeSeriesEncoder(input_dim=1, hidden_dim=64, output_dim=32)
# ... train encoder using contrastive learning ...

# Encode data
train_encodings = encoder(torch.tensor(train_data, dtype=torch.float32)).detach().numpy()
test_encodings = encoder(torch.tensor(test_data, dtype=torch.float32)).detach().numpy()
anomaly_encodings = encoder(torch.tensor(anomaly_data_scaled, dtype=torch.float32)).detach().numpy()

# Compute anomaly scores
train_distances = np.linalg.norm(train_encodings - train_encodings.mean(axis=0), axis=1)
threshold = np.percentile(train_distances, 95)  # 95th percentile as threshold

test_distances = np.linalg.norm(test_encodings - train_encodings.mean(axis=0), axis=1)
anomaly_distances = np.linalg.norm(anomaly_encodings - train_encodings.mean(axis=0), axis=1)

# Evaluate
y_true = np.concatenate([np.zeros(len(test_distances)), np.ones(len(anomaly_distances))])
y_scores = np.concatenate([test_distances, anomaly_distances])
auc_score = roc_auc_score(y_true, y_scores)
print(f"AUC score: {auc_score:.4f}")
```

Slide 16: Additional Resources

For those interested in diving deeper into Contrastive Learning, here are some valuable resources:

1. "A Simple Framework for Contrastive Learning of Visual Representations" by Chen et al. (2020) ArXiv: [https://arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709)
2. "Momentum Contrast for Unsupervised Visual Representation Learning" by He et al. (2020) ArXiv: [https://arxiv.org/abs/1911.05722](https://arxiv.org/abs/1911.05722)
3. "Supervised Contrastive Learning" by Khosla et al. (2020) ArXiv: [https://arxiv.org/abs/2004.11362](https://arxiv.org/abs/2004.11362)
4. "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere" by Wang and Isola (2020) ArXiv: [https://arxiv.org/abs/2005.10242](https://arxiv.org/abs/2005.10242)
5. "What Makes for Good Views for Contrastive Learning" by Tian et al. (2020) ArXiv: [https://arxiv.org/abs/2005.10243](https://arxiv.org/abs/2005.10243)

These papers provide in-depth explanations of various contrastive learning techniques and their theoretical foundations. They offer valuable insights for both beginners and advanced practitioners in the field of machine learning and computer vision.

