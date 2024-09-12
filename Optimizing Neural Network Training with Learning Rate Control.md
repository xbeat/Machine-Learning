## Optimizing Neural Network Training with Learning Rate Control
Slide 1: Learning Rate Control in Neural Networks

Learning rate control is crucial for optimizing neural network performance. It involves adjusting the step size during gradient descent to find the right balance between convergence speed and accuracy. This slide introduces various techniques for managing learning rates effectively.

```python
import torch
import torch.optim as optim

# Define a simple neural network
model = torch.nn.Sequential(
    torch.nn.Linear(10, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 1)
)

# Initialize optimizer with a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Example of manual learning rate adjustment
for epoch in range(100):
    if epoch == 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01  # Reduce learning rate at epoch 50
```

Slide 2: Learning Rate Scheduling

Learning rate scheduling involves adjusting the learning rate during training according to a predefined strategy. This technique can help overcome plateaus and improve model convergence.

```python
from torch.optim.lr_scheduler import StepLR

# Create optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Create scheduler
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Training loop
for epoch in range(100):
    # Training code here
    
    optimizer.step()
    scheduler.step()

    print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()}")
```

Slide 3: Cosine Annealing Learning Rate Scheduler

Cosine annealing is a popular learning rate scheduling technique that smoothly decreases the learning rate following a cosine curve. This method can help the model explore different regions of the loss landscape.

```python
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

lrs = []
for epoch in range(100):
    optimizer.step()
    scheduler.step()
    lrs.append(scheduler.get_last_lr()[0])

plt.plot(lrs)
plt.title("Cosine Annealing Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.show()
```

Slide 4: Adaptive Learning Rate Methods: Adam Optimizer

Adaptive learning rate methods automatically adjust the learning rate for each parameter. Adam (Adaptive Moment Estimation) is a popular optimizer that combines ideas from RMSprop and momentum optimization.

```python
import torch.nn as nn

# Define a simple model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    output = model(input_data)
    loss = loss_function(output, target)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training completed")
```

Slide 5: Learning Rate Warmup

Learning rate warmup gradually increases the learning rate from a small value to the initial learning rate over a number of training steps. This technique can help stabilize training in the early stages.

```python
class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, initial_lr):
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.initial_lr * (self.last_epoch + 1) / self.warmup_steps for _ in self.base_lrs]
        return self.base_lrs

# Usage
optimizer = optim.Adam(model.parameters(), lr=0.001)
warmup_scheduler = WarmupLR(optimizer, warmup_steps=1000, initial_lr=1e-6)

for epoch in range(100):
    for batch in dataloader:
        optimizer.step()
        warmup_scheduler.step()
```

Slide 6: Gradient Clipping

Gradient clipping is a technique used to prevent exploding gradients by limiting the maximum norm of the gradient vector. This method is particularly useful in recurrent neural networks.

```python
import torch.nn as nn

model = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
max_grad_norm = 1.0

for epoch in range(100):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        
        # Clip gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()

print("Training completed with gradient clipping")
```

Slide 7: Monitoring Learning Progress

Monitoring the learning progress is essential for identifying issues and making informed decisions about hyperparameter tuning. This slide demonstrates how to track and visualize key metrics during training.

```python
import matplotlib.pyplot as plt

train_losses = []
val_losses = []

for epoch in range(100):
    # Training
    model.train()
    train_loss = train_one_epoch(model, train_loader, optimizer)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = validate(model, val_loader)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Plot losses
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

Slide 8: Learning Rate Finder

The learning rate finder is a technique to determine an optimal learning rate range for your model. It involves training the model with exponentially increasing learning rates and observing the loss curve.

```python
import numpy as np
import matplotlib.pyplot as plt

def find_lr(model, train_loader, optimizer, criterion, start_lr=1e-7, end_lr=10, num_iter=100):
    lrs, losses = [], []
    model.train()
    for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
    
    for i in range(num_iter):
        inputs, targets = next(iter(train_loader))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        # Update learning rate
        lr = np.exp(np.log(start_lr) + (np.log(end_lr) - np.log(start_lr)) * i / num_iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.show()

# Usage
find_lr(model, train_loader, optimizer, nn.CrossEntropyLoss())
```

Slide 9: Cyclical Learning Rates

Cyclical learning rates involve cycling the learning rate between a lower and upper bound. This technique can help the model escape local minima and find better optima.

```python
import numpy as np
import matplotlib.pyplot as plt

class CyclicLR:
    def __init__(self, optimizer, base_lr, max_lr, step_size):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.cycle = 0
        self.step_in_cycle = 0
    
    def step(self):
        cycle = np.floor(1 + self.step_in_cycle / (2 * self.step_size))
        x = np.abs(self.step_in_cycle / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.step_in_cycle += 1

# Usage
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size=1000)

lrs = []
for _ in range(10000):
    scheduler.step()
    lrs.append(optimizer.param_groups[0]['lr'])

plt.plot(lrs)
plt.title("Cyclical Learning Rate")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.show()
```

Slide 10: Layer-wise Adaptive Learning Rates

Different layers in a neural network may require different learning rates. This slide demonstrates how to set different learning rates for different layers of a model.

```python
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc = nn.Linear(32 * 6 * 6, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 6 * 6)
        x = self.fc(x)
        return x

model = CustomModel()

# Set different learning rates for different layers
optimizer = optim.SGD([
    {'params': model.conv1.parameters(), 'lr': 0.01},
    {'params': model.conv2.parameters(), 'lr': 0.001},
    {'params': model.fc.parameters(), 'lr': 0.0001}
], momentum=0.9)

print("Learning rates:")
for i, param_group in enumerate(optimizer.param_groups):
    print(f"Layer {i}: {param_group['lr']}")
```

Slide 11: Real-life Example: Image Classification

This slide demonstrates how to apply learning rate techniques in a real-world image classification task using the CIFAR-10 dataset.

```python
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Define model, loss, and optimizer
model = torchvision.models.resnet18(pretrained=False, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# Training loop
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(trainloader)
    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    scheduler.step(avg_loss)

print("Training completed")
```

Slide 12: Real-life Example: Natural Language Processing

This slide showcases the application of learning rate techniques in a sentiment analysis task using a recurrent neural network.

```python
import torch.nn as nn
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Load IMDB dataset
train_iter = IMDB(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Define model
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))

model = LSTM(len(vocab), embedding_dim=100, hidden_dim=256)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=1000)

# Training loop (simplified)
for step in range(1000):
    # ... training code ...
    optimizer.step()
    scheduler.step()

print("Training completed")
```

Slide 13: Advanced Techniques: Layer-wise Adaptive Rates for Training (LARS)

LARS is an optimization technique that adapts the learning rate for each layer based on the ratio of the L2 norm of the weights to the L2 norm of the gradients. This method is particularly useful for training very large batch sizes.

```python
import torch

class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0001, trust_coefficient=0.001):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, trust_coefficient=trust_coefficient)
        super(LARS, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            trust_coefficient = group['trust_coefficient']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(p.grad.data)
                
                if param_norm != 0 and grad_norm != 0:
                    local_lr = trust_coefficient * (param_norm / grad_norm)
                else:
                    local_lr = 1.0
                
                p.data = p.data - local_lr * group['lr'] * (p.grad.data + weight_decay * p.data)

# Usage
model = YourModel()
optimizer = LARS(model.parameters(), lr=0.1)

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), targets)
        loss.backward()
        optimizer.step()
```

Slide 14: Monitoring and Visualization with TensorBoard

TensorBoard is a powerful tool for visualizing various aspects of the training process, including learning rates, loss curves, and model architectures. This slide demonstrates how to use TensorBoard with PyTorch.

```python
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim

# Initialize TensorBoard writer
writer = SummaryWriter('runs/experiment_1')

# Define a simple model
model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Log the running loss
    writer.add_scalar('training loss', running_loss / len(trainloader), epoch)
    writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)

writer.close()
print("Training completed. Run 'tensorboard --logdir=runs' to view results.")
```

Slide 15: Combining Multiple Techniques

In practice, it's common to combine multiple learning rate control techniques. This slide demonstrates how to use learning rate warmup, cosine annealing, and gradient clipping together.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

class WarmupCosineSchedule(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_steps) / (self.t_total - self.warmup_steps)
        return [base_lr * (0.5 * (1 + torch.cos(math.pi * self.cycles * 2 * progress))) for base_lr in self.base_lrs]

# Setup
model = YourModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = WarmupCosineSchedule(optimizer, warmup_steps=1000, t_total=10000)
max_grad_norm = 1.0

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), targets)
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        scheduler.step()

print("Training completed with combined techniques")
```

Slide 16: Additional Resources

For further exploration of learning rate control techniques and optimization in deep learning, consider the following resources:

1. "Cyclical Learning Rates for Training Neural Networks" by Leslie N. Smith ArXiv: [https://arxiv.org/abs/1506.01186](https://arxiv.org/abs/1506.01186)
2. "Adaptive Methods for Stochastic Optimization" by Diederik P. Kingma and Jimmy Ba ArXiv: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
3. "SGDR: Stochastic Gradient Descent with Warm Restarts" by Ilya Loshchilov and Frank Hutter ArXiv: [https://arxiv.org/abs/1608.03983](https://arxiv.org/abs/1608.03983)
4. "An Overview of Gradient Descent Optimization Algorithms" by Sebastian Ruder ArXiv: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)

These papers provide in-depth discussions on various learning rate control techniques and their applications in deep learning.

