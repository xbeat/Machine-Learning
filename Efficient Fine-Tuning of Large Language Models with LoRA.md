## Efficient Fine-Tuning of Large Language Models with LoRA
Slide 1: Introduction to Low-Rank Adaptation (LoRA)

Low-Rank Adaptation (LoRA) is an efficient method for fine-tuning large language models (LLMs). It addresses the computational challenges of traditional fine-tuning by updating only small, low-rank matrices instead of the entire model. This approach significantly reduces resource requirements while maintaining performance.

```python
import numpy as np

def lora_update(W, A, B):
    """
    Perform LoRA update on weight matrix W
    W: original weight matrix
    A, B: low-rank matrices
    """
    return W + np.dot(A, B.T)

# Example usage
W = np.random.randn(1000, 1000)  # Large weight matrix
A = np.random.randn(1000, 10)    # Low-rank matrix A
B = np.random.randn(1000, 10)    # Low-rank matrix B

W_updated = lora_update(W, A, B)
print(f"Original shape: {W.shape}, Updated shape: {W_updated.shape}")
```

Slide 2: Comparing LoRA with Singular Value Decomposition (SVD)

While both LoRA and SVD are low-rank approximation methods, LoRA is better suited for fine-tuning LLMs. SVD's linear nature struggles with the non-linear relationships in transformer architectures, and it's not easily updatable during training. LoRA, on the other hand, provides dynamic, step-by-step updates and focuses on small, low-rank matrices, making it more efficient and scalable.

```python
import numpy as np

def svd_approximation(W, k):
    """
    Perform SVD approximation on matrix W
    W: input matrix
    k: number of singular values to keep
    """
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    return np.dot(U[:, :k] * S[:k], Vt[:k, :])

# Example usage
W = np.random.randn(1000, 1000)
k = 10

W_svd = svd_approximation(W, k)
print(f"Original shape: {W.shape}, SVD approximation shape: {W_svd.shape}")
```

Slide 3: LoRA's Mechanism

LoRA works by freezing the original weight matrix W and introducing two small, trainable low-rank matrices A and B. The updated weight matrix W' is computed as W' = W + AB. This allows fine-tuning by training only the smaller matrices A and B, while W remains unchanged, significantly reducing the computational load.

```python
import torch

class LoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        self.A = torch.nn.Parameter(torch.randn(out_features, rank))
        self.B = torch.nn.Parameter(torch.randn(rank, in_features))
    
    def forward(self, x):
        return torch.matmul(self.W + torch.matmul(self.A, self.B), x)

# Example usage
layer = LoRALayer(100, 50, rank=10)
input_tensor = torch.randn(32, 100)
output = layer(input_tensor)
print(f"Input shape: {input_tensor.shape}, Output shape: {output.shape}")
```

Slide 4: Advantages of LoRA

LoRA offers several benefits for fine-tuning LLMs:

1. Reduced memory usage and computational requirements
2. Faster training and inference times
3. Preservation of pre-trained model knowledge
4. Flexibility to adapt to different tasks without full retraining

```python
import torch
import time

def compare_training_time(original_model, lora_model, input_data, epochs=10):
    original_start = time.time()
    for _ in range(epochs):
        original_model(input_data)
    original_time = time.time() - original_start

    lora_start = time.time()
    for _ in range(epochs):
        lora_model(input_data)
    lora_time = time.time() - lora_start

    print(f"Original model training time: {original_time:.2f}s")
    print(f"LoRA model training time: {lora_time:.2f}s")
    print(f"Speed-up: {original_time / lora_time:.2f}x")

# Example usage (pseudo-code, as actual implementation depends on specific models)
# original_model = FullyTrainableModel()
# lora_model = LoRAModel()
# input_data = torch.randn(1000, 100)
# compare_training_time(original_model, lora_model, input_data)
```

Slide 5: Implementing LoRA in PyTorch

Here's a simple implementation of LoRA in PyTorch, demonstrating how to create a LoRA layer and integrate it into a neural network:

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, rank))
        self.scale = 0.01

    def forward(self, x):
        base_output = self.linear(x)
        lora_output = (self.lora_B @ self.lora_A) @ x.T
        return base_output + self.scale * lora_output.T

# Example usage
lora_layer = LoRALayer(100, 50, rank=10)
input_tensor = torch.randn(32, 100)
output = lora_layer(input_tensor)
print(f"Input shape: {input_tensor.shape}, Output shape: {output.shape}")
```

Slide 6: Fine-tuning with LoRA

To fine-tune a model using LoRA, we need to replace the target layers with LoRA layers and only train the LoRA parameters. Here's an example of how to modify a pre-trained model:

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class LoRAModel(nn.Module):
    def __init__(self, base_model, lora_rank=4):
        super().__init__()
        self.base_model = base_model
        
        # Replace linear layers with LoRA layers
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = self.base_model.get_submodule(parent_name)
                setattr(parent, child_name, LoRALayer(module.in_features, module.out_features, lora_rank))

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

# Load pre-trained model
base_model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create LoRA model
lora_model = LoRAModel(base_model)

# Freeze base model parameters
for param in lora_model.base_model.parameters():
    param.requires_grad = False

# Only train LoRA parameters
lora_params = [p for n, p in lora_model.named_parameters() if 'lora_' in n]
optimizer = torch.optim.AdamW(lora_params, lr=1e-3)

# Example fine-tuning loop (pseudo-code)
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         optimizer.zero_grad()
#         outputs = lora_model(**batch)
#         loss = compute_loss(outputs, batch['labels'])
#         loss.backward()
#         optimizer.step()
```

Slide 7: Visualizing LoRA Updates

To better understand how LoRA updates work, let's create a visualization of the weight matrix changes:

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_lora_update(W, A, B, num_steps=5):
    fig, axes = plt.subplots(1, num_steps + 1, figsize=(20, 4))
    
    axes[0].imshow(W, cmap='coolwarm')
    axes[0].set_title('Original W')
    
    for i in range(1, num_steps + 1):
        alpha = i / num_steps
        W_updated = W + alpha * np.dot(A, B.T)
        axes[i].imshow(W_updated, cmap='coolwarm')
        axes[i].set_title(f'W + {alpha:.2f}AB')
    
    plt.tight_layout()
    plt.show()

# Example usage
W = np.random.randn(50, 50)
A = np.random.randn(50, 5)
B = np.random.randn(50, 5)

visualize_lora_update(W, A, B)
```

Slide 8: LoRA for Different Model Architectures

LoRA can be applied to various model architectures, not just transformers. Here's an example of how to implement LoRA for a convolutional neural network (CNN):

```python
import torch
import torch.nn as nn

class LoRAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank=4, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.lora_A = nn.Parameter(torch.randn(rank, in_channels, 1, 1))
        self.lora_B = nn.Parameter(torch.randn(out_channels, rank, 1, 1))
        self.scale = 0.01

    def forward(self, x):
        base_output = self.conv(x)
        lora_output = nn.functional.conv2d(x, self.scale * (self.lora_B @ self.lora_A).squeeze(0))
        return base_output + lora_output

# Example usage in a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = LoRAConv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = LoRAConv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleCNN()
input_tensor = torch.randn(1, 3, 32, 32)
output = model(input_tensor)
print(f"Input shape: {input_tensor.shape}, Output shape: {output.shape}")
```

Slide 9: Hyperparameter Tuning for LoRA

When using LoRA, it's important to tune hyperparameters for optimal performance. Here's a script to demonstrate hyperparameter tuning using a simple grid search:

```python
import torch
import torch.nn as nn
from itertools import product

def train_and_evaluate(model, train_data, val_data, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for _ in range(epochs):
        model.train()
        for batch in train_data:
            optimizer.zero_grad()
            outputs = model(batch['input'])
            loss = criterion(outputs, batch['target'])
            loss.backward()
            optimizer.step()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_data:
            outputs = model(batch['input'])
            val_loss += criterion(outputs, batch['target']).item()
    
    return val_loss / len(val_data)

def hyperparameter_tuning(model_class, train_data, val_data):
    rank_values = [4, 8, 16]
    lr_values = [1e-3, 1e-4, 1e-5]
    epoch_values = [5, 10, 20]
    
    best_params = None
    best_val_loss = float('inf')
    
    for rank, lr, epochs in product(rank_values, lr_values, epoch_values):
        model = model_class(rank=rank)
        val_loss = train_and_evaluate(model, train_data, val_data, epochs, lr)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {'rank': rank, 'lr': lr, 'epochs': epochs}
    
    return best_params

# Example usage (pseudo-code)
# best_params = hyperparameter_tuning(LoRAModel, train_dataloader, val_dataloader)
# print(f"Best hyperparameters: {best_params}")
```

Slide 10: Real-life Example: Text Classification

Let's apply LoRA to a text classification task using a pre-trained BERT model:

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class LoRABertForClassification(nn.Module):
    def __init__(self, num_classes, lora_rank=4):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = LoRALayer(self.bert.config.hidden_size, num_classes, rank=lora_rank)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits

# Example usage
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = LoRABertForClassification(num_classes=2, lora_rank=8)

# Prepare input
text = "This is a sample sentence for classification."
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Forward pass
with torch.no_grad():
    logits = model(**inputs)

print(f"Input text: {text}")
print(f"Output logits: {logits}")
```

Slide 11: Real-life Example: Image Style Transfer

Let's apply LoRA to a style transfer task using a pre-trained VGG19 model:

```python
import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.transforms import ToPILImage

class LoRAStyleTransfer(nn.Module):
    def __init__(self, lora_rank=4):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = self.add_lora_to_sequential(vgg[:4], lora_rank)
        self.slice2 = self.add_lora_to_sequential(vgg[4:9], lora_rank)
        self.slice3 = self.add_lora_to_sequential(vgg[9:18], lora_rank)
        self.slice4 = self.add_lora_to_sequential(vgg[18:27], lora_rank)

    def add_lora_to_sequential(self, sequential, rank):
        for i, layer in enumerate(sequential):
            if isinstance(layer, nn.Conv2d):
                sequential[i] = LoRAConv2d(layer.in_channels, layer.out_channels, 
                                           layer.kernel_size, rank=rank)
        return sequential

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4

# Example usage (pseudo-code)
# model = LoRAStyleTransfer()
# content_image = load_image("content.jpg")
# style_image = load_image("style.jpg")
# output = model(content_image)
# Apply style transfer algorithm using the LoRA-modified features
```

Slide 12: LoRA for Natural Language Processing Tasks

LoRA can be particularly effective for fine-tuning large language models on specific NLP tasks. Here's an example of applying LoRA to a sentiment analysis task:

```python
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

class LoRASentimentClassifier(nn.Module):
    def __init__(self, lora_rank=4):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.sentiment_classifier = LoRALayer(self.roberta.config.hidden_size, 2, rank=lora_rank)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.sentiment_classifier(outputs.last_hidden_state[:, 0, :])
        return logits

# Example usage
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = LoRASentimentClassifier()

text = "I absolutely loved this movie! The acting was superb."
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

with torch.no_grad():
    logits = model(**inputs)

print(f"Input text: {text}")
print(f"Sentiment logits: {logits}")
```

Slide 13: LoRA for Reinforcement Learning

LoRA can also be applied to reinforcement learning models to fine-tune them for specific environments. Here's a simple example using a LoRA-modified Q-network:

```python
import torch
import torch.nn as nn
import gym

class LoRAQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, lora_rank=4):
        super().__init__()
        self.fc1 = LoRALayer(input_dim, hidden_dim, rank=lora_rank)
        self.fc2 = LoRALayer(hidden_dim, hidden_dim, rank=lora_rank)
        self.fc3 = LoRALayer(hidden_dim, output_dim, rank=lora_rank)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Example usage (pseudo-code)
# env = gym.make('CartPole-v1')
# model = LoRAQNetwork(env.observation_space.shape[0], env.action_space.n)
# optimizer = torch.optim.Adam(model.parameters())

# Training loop
# for episode in range(num_episodes):
#     state = env.reset()
#     done = False
#     while not done:
#         action = model(torch.FloatTensor(state)).argmax().item()
#         next_state, reward, done, _ = env.step(action)
#         # Update Q-values using LoRA-modified network
#         # ...
#         state = next_state
```

Slide 14: Comparing LoRA Performance

To demonstrate the effectiveness of LoRA, let's compare the performance of a standard fine-tuned model with a LoRA-fine-tuned model:

```python
import torch
import torch.nn as nn
import time

def compare_models(standard_model, lora_model, input_data, num_epochs=10):
    criterion = nn.MSELoss()
    standard_optimizer = torch.optim.Adam(standard_model.parameters())
    lora_optimizer = torch.optim.Adam(lora_model.parameters())

    standard_time = 0
    lora_time = 0

    for _ in range(num_epochs):
        # Train standard model
        start_time = time.time()
        standard_optimizer.zero_grad()
        standard_output = standard_model(input_data)
        standard_loss = criterion(standard_output, input_data)
        standard_loss.backward()
        standard_optimizer.step()
        standard_time += time.time() - start_time

        # Train LoRA model
        start_time = time.time()
        lora_optimizer.zero_grad()
        lora_output = lora_model(input_data)
        lora_loss = criterion(lora_output, input_data)
        lora_loss.backward()
        lora_optimizer.step()
        lora_time += time.time() - start_time

    print(f"Standard model training time: {standard_time:.4f}s")
    print(f"LoRA model training time: {lora_time:.4f}s")
    print(f"Speed-up: {standard_time / lora_time:.2f}x")

# Example usage (pseudo-code)
# standard_model = StandardModel()
# lora_model = LoRAModel()
# input_data = torch.randn(100, 1000)
# compare_models(standard_model, lora_model, input_data)
```

Slide 15: Additional Resources

For more information on Low-Rank Adaptation (LoRA) and its applications in fine-tuning large language models, consider exploring the following resources:

1. Original LoRA paper: "LoRA: Low-Rank Adaptation of Large Language Models" by Hu et al. (2021) ArXiv link: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
2. "Parameter-Efficient Transfer Learning for NLP" by Houlsby et al. (2019) ArXiv link: [https://arxiv.org/abs/1902.00751](https://arxiv.org/abs/1902.00751)
3. "Exploiting Shared Representations for Personalized Federated Learning" by Collins et al. (2021) ArXiv link: [https://arxiv.org/abs/2102.07078](https://arxiv.org/abs/2102.07078)

These papers provide in-depth discussions on the theory and applications of LoRA and related techniques in various machine learning domains.

