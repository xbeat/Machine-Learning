## PyTorch Research Reproducibility 
Slide 1: Introduction to PyTorch Research Reproducibility

Research reproducibility is a crucial aspect of scientific progress, especially in the field of machine learning. PyTorch, a popular deep learning framework, provides tools and best practices to ensure that research conducted using it can be easily reproduced and verified by others. This slideshow will explore various techniques and strategies for enhancing reproducibility in PyTorch-based research.

Slide 2: Setting Random Seeds

To ensure reproducibility, it's essential to control randomness in your experiments. PyTorch provides functions to set random seeds for various components.

```python
import torch
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Set a fixed seed for reproducibility
```

Slide 3: Documenting Hardware and Software Environment

Reproducibility often depends on the specific hardware and software used. It's crucial to document these details in your research.

```python
import torch
import platform
import psutil

def get_system_info():
    return {
        "OS": platform.system(),
        "Python Version": platform.python_version(),
        "PyTorch Version": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
        "GPU Name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "CPU": platform.processor(),
        "RAM": f"{psutil.virtual_memory().total / (1024**3):.2f} GB"
    }

system_info = get_system_info()
print(system_info)
```

Slide 4: Source Code for Documenting Hardware and Software Environment

```python
# Results
{
    'OS': 'Linux',
    'Python Version': '3.8.10',
    'PyTorch Version': '1.9.0',
    'CUDA Available': True,
    'GPU Name': 'NVIDIA GeForce RTX 3080',
    'CPU': 'x86_64',
    'RAM': '32.00 GB'
}
```

Slide 5: Version Control and Code Management

Using version control systems like Git is essential for tracking changes and collaborating with others. Here's a simple example of how to use GitPython to manage your repository:

```python
from git import Repo
import os

def initialize_git_repo(path):
    if not os.path.exists(path):
        os.makedirs(path)
    repo = Repo.init(path)
    return repo

def commit_changes(repo, message):
    repo.git.add(A=True)
    repo.index.commit(message)

# Usage
repo_path = "./my_research_project"
repo = initialize_git_repo(repo_path)
commit_changes(repo, "Initial commit with experiment setup")
```

Slide 6: Saving and Loading Model Checkpoints

Proper management of model checkpoints is crucial for reproducibility. PyTorch provides easy-to-use functions for saving and loading model states.

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def save_checkpoint(model, optimizer, epoch, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

# Usage
model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters())
save_checkpoint(model, optimizer, 10, 'checkpoint.pth')
model, optimizer, epoch = load_checkpoint(model, optimizer, 'checkpoint.pth')
```

Slide 7: Logging Experiment Results

Proper logging of experiment results is essential for tracking progress and ensuring reproducibility. Here's an example using the built-in logging module:

```python
import logging
import json
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def log_experiment(logger, experiment_name, parameters, results):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "parameters": parameters,
        "results": results
    }
    logger.info(json.dumps(log_entry))

# Usage
logger = setup_logger('experiment_logger', 'experiments.log')
log_experiment(logger, 'CNN_Classification', 
               {"learning_rate": 0.001, "batch_size": 32},
               {"accuracy": 0.95, "loss": 0.1})
```

Slide 8: Deterministic Data Loading

Ensuring deterministic data loading is crucial for reproducibility. PyTorch's DataLoader can be configured for deterministic behavior:

```python
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_deterministic_dataloader(dataset, batch_size, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, 
                      shuffle=True, num_workers=0, 
                      generator=generator)

# Usage
dataset = SimpleDataset(1000)
dataloader = create_deterministic_dataloader(dataset, batch_size=32, seed=42)
```

Slide 9: Hyperparameter Management

Managing hyperparameters is essential for reproducibility. Here's a simple approach using a configuration file:

```python
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config, config_path):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

# Example configuration
config = {
    "model": {
        "type": "CNN",
        "num_layers": 3,
        "activation": "ReLU"
    },
    "training": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    }
}

# Save and load configuration
save_config(config, 'experiment_config.json')
loaded_config = load_config('experiment_config.json')
print(loaded_config)
```

Slide 10: Reproducible Data Augmentation

Data augmentation can introduce randomness. Here's how to make it reproducible:

```python
import torch
import torchvision.transforms as transforms

def get_reproducible_transforms(seed):
    torch.manual_seed(seed)
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Usage
seed = 42
transform = get_reproducible_transforms(seed)

# Apply transform to an image
image = torch.randn(3, 224, 224)  # Dummy image
augmented_image = transform(image)
```

Slide 11: Documenting Random States

Documenting the state of random number generators is crucial for reproducing exact results:

```python
import torch
import random
import numpy as np

def get_random_states():
    return {
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate()
    }

def set_random_states(states):
    torch.set_rng_state(states["torch_rng_state"])
    torch.cuda.set_rng_state_all(states["torch_cuda_rng_state"])
    np.random.set_state(states["numpy_rng_state"])
    random.setstate(states["python_rng_state"])

# Usage
initial_states = get_random_states()
# ... perform some operations ...
set_random_states(initial_states)  # Reset to initial state
```

Slide 12: Real-Life Example: Image Classification

Let's consider a real-life example of a reproducible image classification experiment using PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1)
    
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(1, 3):
        train(model, device, train_loader, optimizer, epoch)
    
    torch.save(model.state_dict(), "mnist_cnn.pt")
    print("Finished Training")

if __name__ == '__main__':
    main()
```

Slide 13: Real-Life Example: Natural Language Processing

Here's another real-life example demonstrating reproducibility in a simple NLP task using PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, hidden = self.rnn(x)
        output = self.fc(hidden.squeeze(0))
        return output

def generate_data(num_samples, seq_length, input_size):
    X = torch.randn(num_samples, seq_length, input_size)
    y = torch.randint(0, 2, (num_samples,))
    return X, y

def train(model, X, y, optimizer, criterion, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = 10
    hidden_size = 20
    output_size = 2
    seq_length = 5
    num_samples = 1000
    epochs = 500

    X, y = generate_data(num_samples, seq_length, input_size)
    X, y = X.to(device), y.to(device)

    model = SimpleRNN(input_size, hidden_size, output_size).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train(model, X, y, optimizer, criterion, epochs)

    torch.save(model.state_dict(), "simple_rnn.pt")
    print("Finished Training")

if __name__ == '__main__':
    main()
```

Slide 14: Additional Resources

For more information on PyTorch research reproducibility, consider exploring the following resources:

1.  PyTorch Documentation: [https://pytorch.org/docs/stable/notes/randomness.html](https://pytorch.org/docs/stable/notes/randomness.html)
2.  "Reproducible Machine Learning with PyTorch and Hydra" by Lorenz Kuhn et al. (2021): [https://arxiv.org/abs/2101.04818](https://arxiv.org/abs/2101.04818)
3.  "A Step Toward Quantifying Independently Reproducible Machine Learning Research" by Edward Raff (2019): [https://arxiv.org/abs/1909.06674](https://arxiv.org/abs/1909.06674)

These resources provide in-depth discussions on best practices, tools, and techniques

