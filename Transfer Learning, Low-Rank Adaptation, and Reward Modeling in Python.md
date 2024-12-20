## Transfer Learning, Low-Rank Adaptation, and Reward Modeling in Python
Slide 1: Transfer Learning: Leveraging Pre-trained Models

Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a second task. This approach is particularly useful when you have limited data for your target task but can leverage knowledge from a related task with abundant data.

```python
import torch
import torchvision.models as models
from torch import nn

# Load a pre-trained ResNet model
resnet = models.resnet50(pretrained=True)

# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace the last fully connected layer
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)  # 10 is the number of classes in our new task

# Now only the last layer will be trained
for param in resnet.fc.parameters():
    param.requires_grad = True

# The model is ready for fine-tuning on your new task
```

Slide 2: Fine-tuning: Adapting Pre-trained Models

Fine-tuning involves taking a pre-trained model and further training it on a new, related task. This process allows the model to adapt its learned features to the specific nuances of the new task while retaining the general knowledge acquired from the original task.

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Assume 'resnet' is our pre-trained model from the previous slide

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your dataset
train_dataset = datasets.ImageFolder('path/to/your/dataset', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)

# Fine-tuning loop
for epoch in range(5):  # 5 epochs as an example
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Fine-tuning completed")
```

Slide 3: Feature Extraction: Utilizing Pre-trained Models as Feature Extractors

Feature extraction is another transfer learning technique where we use a pre-trained model to extract meaningful features from our data, then train a new classifier on these features. This is useful when we believe the pre-trained model has learned general features that are also relevant to our new task.

```python
import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained ResNet model
resnet = models.resnet50(pretrained=True)

# Remove the last fully connected layer
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

# Freeze all parameters
for param in feature_extractor.parameters():
    param.requires_grad = False

# Function to extract features
def extract_features(input_batch):
    with torch.no_grad():
        features = feature_extractor(input_batch)
    return features.view(features.size(0), -1)

# Now you can use this function to extract features from your data
# and train a new classifier on these features
```

Slide 4: Low-Rank Adaptation (LoRA): Efficient Fine-tuning

Low-Rank Adaptation (LoRA) is a technique that enables efficient fine-tuning of large pre-trained models. It works by adding trainable low-rank matrices to the existing weights, significantly reducing the number of trainable parameters while maintaining performance.

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = 0.01

    def forward(self, x):
        return self.scaling * (x @ self.lora_A.T @ self.lora_B.T)

# Example usage
original_layer = nn.Linear(768, 768)
lora_layer = LoRALayer(768, 768)

def forward(x):
    return original_layer(x) + lora_layer(x)

# Only train LoRA parameters
for param in original_layer.parameters():
    param.requires_grad = False
```

Slide 5: Implementing LoRA in a Transformer Model

Let's see how we can apply LoRA to a transformer model, specifically to its attention layers. This example demonstrates how to modify a simple self-attention mechanism with LoRA.

```python
import torch
import torch.nn as nn

class SelfAttentionWithLoRA(nn.Module):
    def __init__(self, embed_dim, num_heads, rank=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.lora_q = LoRALayer(embed_dim, embed_dim, rank)
        self.lora_k = LoRALayer(embed_dim, embed_dim, rank)
        self.lora_v = LoRALayer(embed_dim, embed_dim, rank)

    def forward(self, x):
        q = self.attention.in_proj_weight[:self.attention.embed_dim] @ x.T + self.lora_q(x.T)
        k = self.attention.in_proj_weight[self.attention.embed_dim:2*self.attention.embed_dim] @ x.T + self.lora_k(x.T)
        v = self.attention.in_proj_weight[2*self.attention.embed_dim:] @ x.T + self.lora_v(x.T)
        
        return self.attention(q.T, k.T, v.T)[0]

# Usage
embed_dim, num_heads = 256, 8
model = SelfAttentionWithLoRA(embed_dim, num_heads)
x = torch.randn(32, 10, embed_dim)  # (batch_size, seq_len, embed_dim)
output = model(x)
print(output.shape)  # Should be (32, 10, 256)
```

Slide 6: Reward Modeling: Defining Preferences

Reward modeling is a crucial component in reinforcement learning, particularly in scenarios where the reward function is not explicitly defined. It involves learning a reward function from human preferences or demonstrations.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

class RewardModel:
    def __init__(self, feature_dim):
        self.model = LogisticRegression()
        
    def fit(self, trajectories, preferences):
        X = np.array([self._featurize(t) for t in trajectories])
        y = np.array(preferences)
        self.model.fit(X, y)
        
    def predict(self, trajectory):
        X = self._featurize(trajectory).reshape(1, -1)
        return self.model.predict_proba(X)[0, 1]  # Probability of being preferred
    
    def _featurize(self, trajectory):
        # Simple featurization: sum of states
        return np.sum(trajectory, axis=0)

# Example usage
reward_model = RewardModel(feature_dim=5)
trajectories = [np.random.rand(10, 5) for _ in range(100)]  # 100 trajectories
preferences = np.random.randint(0, 2, 100)  # Binary preferences
reward_model.fit(trajectories, preferences)

new_trajectory = np.random.rand(10, 5)
predicted_reward = reward_model.predict(new_trajectory)
print(f"Predicted reward: {predicted_reward}")
```

Slide 7: Implementing a Simple Reward Model

Let's implement a simple neural network-based reward model. This model will learn to predict rewards based on state-action pairs.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralRewardModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)

# Example usage
state_dim, action_dim = 10, 5
model = NeuralRewardModel(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop (pseudo-code)
for epoch in range(100):
    state = torch.randn(32, state_dim)
    action = torch.randn(32, action_dim)
    true_reward = torch.randn(32, 1)  # In practice, this would come from human feedback
    
    predicted_reward = model(state, action)
    loss = criterion(predicted_reward, true_reward)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training completed")
```

Slide 8: Combining Transfer Learning and Reward Modeling

We can combine transfer learning and reward modeling to create more efficient and effective reinforcement learning systems. Here's an example of how we might use a pre-trained vision model to extract features for a reward model in a visual task.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class VisualRewardModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load pre-trained ResNet
        self.feature_extractor = models.resnet18(pretrained=True)
        # Remove the last fully connected layer
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        # Freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # Add a new fully connected layer for reward prediction
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)

# Example usage
model = VisualRewardModel(num_classes=1)  # 1 for regression, 2+ for classification
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop (pseudo-code)
for epoch in range(100):
    inputs = torch.randn(32, 3, 224, 224)  # Dummy input
    true_rewards = torch.randn(32, 1)  # In practice, this would come from human feedback
    
    predicted_rewards = model(inputs)
    loss = criterion(predicted_rewards, true_rewards)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training completed")
```

Slide 9: Real-life Example: Sentiment Analysis using Transfer Learning

Let's explore a practical application of transfer learning in natural language processing: sentiment analysis. We'll use a pre-trained BERT model and fine-tune it for sentiment classification.

```python
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch

# Load pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare your data (example)
texts = ["I love this product!", "This movie was terrible.", "Neutral opinion here."]
labels = [1, 0, 1]  # 1 for positive, 0 for negative

# Tokenize and encode the texts
encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
input_ids = encoded['input_ids']
attention_mask = encoded['attention_mask']

# Create DataLoader
dataset = TensorDataset(input_ids, attention_mask, torch.tensor(labels))
dataloader = DataLoader(dataset, batch_size=2)

# Fine-tuning
optimizer = AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(3):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

print("Fine-tuning completed")

# Inference
model.eval()
test_text = "I'm feeling great today!"
test_encoding = tokenizer(test_text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**test_encoding)
    prediction = torch.argmax(outputs.logits, dim=1)
print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
```

Slide 10: Real-life Example: Image Classification with LoRA

Let's apply Low-Rank Adaptation (LoRA) to an image classification task using a pre-trained ResNet model. This example demonstrates how LoRA can be used to efficiently fine-tune a large model for a new task.

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = 0.01

    def forward(self, x):
        return self.scaling * (x @ self.lora_A.T @ self.lora_B.T)

# Load pre-trained ResNet and add LoRA
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Sequential(
    resnet.fc,
    LoRALayer(1000, 1000)
)

# Freeze all parameters except LoRA
for param in resnet.parameters():
    param.requires_grad = False
for param in resnet.fc[-1].parameters():
    param.requires_grad = True

# Prepare dummy data and train
num_samples, num_classes = 1000, 10
dummy_data = torch.randn(num_samples, 3, 224, 224)
dummy_labels = torch.randint(0, num_classes, (num_samples,))
dataloader = DataLoader(TensorDataset(dummy_data, dummy_labels), batch_size=32)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.fc[-1].parameters(), lr=0.001)

for epoch in range(5):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        loss = criterion(resnet(inputs), labels)
        loss.backward()
        optimizer.step()

print("Fine-tuning with LoRA completed")
```

Slide 11: Reward Modeling for Content Recommendation

In this example, we'll implement a simple reward model for a content recommendation system. The model learns to predict user engagement (e.g., likes or clicks) based on content features and user preferences.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ContentRewardModel(nn.Module):
    def __init__(self, content_features, user_features):
        super().__init__()
        self.content_encoder = nn.Sequential(
            nn.Linear(content_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.user_encoder = nn.Sequential(
            nn.Linear(user_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.predictor = nn.Linear(64, 1)
    
    def forward(self, content, user):
        content_embedding = self.content_encoder(content)
        user_embedding = self.user_encoder(user)
        combined = torch.cat([content_embedding, user_embedding], dim=1)
        return self.predictor(combined)

# Example usage
content_features, user_features = 50, 20
model = ContentRewardModel(content_features, user_features)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop (with dummy data)
for epoch in range(100):
    content = torch.randn(32, content_features)
    user = torch.randn(32, user_features)
    engagement = torch.randn(32, 1)  # Simulated engagement data
    
    predicted_engagement = model(content, user)
    loss = criterion(predicted_engagement, engagement)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Reward model training completed")
```

Slide 12: Combining Transfer Learning and LoRA for Efficient Fine-tuning

This example demonstrates how to combine transfer learning with Low-Rank Adaptation (LoRA) for efficient fine-tuning of a pre-trained language model on a text classification task.

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = 0.01

    def forward(self, x):
        return self.scaling * (x @ self.lora_A.T @ self.lora_B.T)

class BertWithLoRA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, num_classes)
        self.lora = LoRALayer(768, 768)
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        lora_output = self.lora(pooled_output)
        return self.classifier(pooled_output + lora_output)

# Example usage
model = BertWithLoRA(num_classes=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input (example)
text = "This is an example sentence."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Forward pass
outputs = model(**inputs)
print(outputs.shape)  # Shape: (1, 2) for binary classification
```

Slide 13: Transfer Learning in Reinforcement Learning

Transfer learning can also be applied in reinforcement learning scenarios. This example shows how to use a pre-trained convolutional neural network (CNN) as a feature extractor in a deep Q-network (DQN) for visual-based reinforcement learning tasks.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class DQNWithTransfer(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        # Load pre-trained ResNet and remove the last layer
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze the feature extractor
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Q-value predictor
        self.q_values = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.q_values(features)

# Example usage
num_actions = 4  # Number of possible actions in the environment
model = DQNWithTransfer(num_actions)

# Simulated input (batch of images)
batch_size = 32
dummy_input = torch.randn(batch_size, 3, 224, 224)

# Forward pass
q_values = model(dummy_input)
print(f"Q-values shape: {q_values.shape}")  # Shape: (32, 4)

# In a real RL scenario, you would use these Q-values to select actions
actions = torch.argmax(q_values, dim=1)
print(f"Selected actions: {actions}")
```

Slide 14: Additional Resources

For those interested in diving deeper into the topics of Transfer Learning, Low-Rank Adaptation, and Reward Modeling, here are some valuable resources:

1. "How transferable are features in deep neural networks?" by Yosinski et al. (2014) ArXiv: [https://arxiv.org/abs/1411.1792](https://arxiv.org/abs/1411.1792)
2. "LoRA: Low-Rank Adaptation of Large Language Models" by Hu et al. (2021) ArXiv: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
3. "Deep Reinforcement Learning from Human Preferences" by Christiano et al. (2017) ArXiv: [https://arxiv.org/abs/1706.03741](https://arxiv.org/abs/1706.03741)
4. "A Survey of Deep Learning Techniques for Neural Machine Translation" by Stahlberg (2020) ArXiv: [https://arxiv.org/abs/1912.02047](https://arxiv.org/abs/1912.02047)

These papers provide in-depth discussions and innovative approaches in their respective fields, offering valuable insights for both beginners and advanced practitioners.

