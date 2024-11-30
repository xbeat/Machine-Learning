## Batch Processing Efficiency in Neural Networks
Slide 1: Batch Processing Fundamentals

Neural networks process data in batches to optimize computational efficiency and improve model convergence. Batch processing allows parallel computation of gradients across multiple samples simultaneously, leveraging modern hardware capabilities like GPUs and reducing memory overhead compared to processing individual samples.

```python
import numpy as np

class BatchProcessor:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.num_batches = len(data) // batch_size
        
    def generate_batches(self):
        # Shuffle data for each epoch
        indices = np.random.permutation(len(self.data))
        for i in range(self.num_batches):
            batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
            yield self.data[batch_indices]

# Example usage
data = np.array(range(100))
processor = BatchProcessor(data, batch_size=16)
batches = list(processor.generate_batches())
print(f"Number of batches: {len(batches)}")
print(f"Batch shape: {batches[0].shape}")
```

Slide 2: Mini-batch Gradient Descent Implementation

Mini-batch gradient descent combines the benefits of both stochastic and batch gradient descent, offering a balance between computation speed and convergence stability. This implementation demonstrates the core concepts of mini-batch processing in neural network training.

```python
import numpy as np

def mini_batch_gradient_descent(X, y, learning_rate=0.01, batch_size=32, epochs=100):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Compute predictions
            predictions = np.dot(X_batch, weights)
            
            # Compute gradients
            gradients = 2/batch_size * X_batch.T.dot(predictions - y_batch)
            
            # Update weights
            weights -= learning_rate * gradients
            
    return weights

# Example usage
X = np.random.randn(1000, 5)
y = np.random.randn(1000)
weights = mini_batch_gradient_descent(X, y)
print("Trained weights:", weights)
```

Slide 3: Data Batching with PyTorch

PyTorch provides sophisticated tools for efficient batch processing through its DataLoader class. This implementation showcases how to properly structure data loading and batching for deep learning models while maintaining memory efficiency.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create sample data
X = np.random.randn(1000, 10)
y = np.random.randn(1000)

# Create dataset and dataloader
dataset = CustomDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example iteration
for batch_idx, (features, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}: Features shape: {features.shape}, Labels shape: {labels.shape}")
    if batch_idx == 2: break  # Show first 3 batches
```

Slide 4: Batch Normalization Theory

Batch normalization is a crucial technique that normalizes the inputs of each layer, making neural networks more stable during training. The mathematical foundation involves normalizing each feature across the mini-batch dimension using mean and variance calculations.

```python
# Mathematical formulation for batch normalization
"""
$$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i$$
$$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$
"""

import torch
import torch.nn as nn

class SimpleBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

Slide 5: Implementing Forward and Backward Passes with Batches

Understanding how gradients flow through batched operations is essential for implementing neural networks. This implementation demonstrates the mathematics and code for computing forward and backward passes on batched inputs.

```python
import numpy as np

class BatchedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        
    def forward(self, X):
        # X shape: (batch_size, input_size)
        self.input = X
        self.output = np.dot(X, self.weights) + self.bias
        return self.output
        
    def backward(self, grad_output):
        # grad_output shape: (batch_size, output_size)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        
        self.weights -= 0.01 * grad_weights
        self.bias -= 0.01 * grad_bias
        return grad_input

# Example usage
batch_size, input_size, output_size = 32, 10, 5
layer = BatchedLayer(input_size, output_size)
X = np.random.randn(batch_size, input_size)
output = layer.forward(X)
grad = np.random.randn(*output.shape)
grad_input = layer.backward(grad)
```

Slide 6: Optimizing Memory Usage in Batch Processing

Memory management becomes critical when processing large batches of data through deep neural networks. This implementation demonstrates efficient memory handling techniques using memory-mapped arrays and proper batch size selection based on available GPU memory.

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class MemoryEfficientDataset(Dataset):
    def __init__(self, data_path, batch_size):
        # Memory-mapped array for efficient data loading
        self.data = np.memmap(data_path, dtype='float32', mode='r', shape=(1000000, 100))
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Load only required data into memory
        return torch.from_numpy(self.data[idx]).float()

def calculate_optimal_batch_size(sample_size, model):
    # Estimate memory per sample
    dummy_input = torch.randn(1, *sample_size)
    torch.cuda.empty_cache()
    
    # Get available GPU memory
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    model_mem = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Calculate optimal batch size
    sample_mem = dummy_input.element_size() * dummy_input.numel()
    optimal_batch = (gpu_mem - model_mem) // (sample_mem * 2)  # Factor of 2 for gradient storage
    
    return min(optimal_batch, 256)  # Cap at 256 for stability

# Example usage
sample_size = (3, 224, 224)
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, 3),
    torch.nn.ReLU()
).cuda()

optimal_batch = calculate_optimal_batch_size(sample_size, model)
print(f"Optimal batch size: {optimal_batch}")
```

Slide 7: Advanced Batch Sampling Strategies

Implementing sophisticated batch sampling strategies can significantly impact model training effectiveness. This implementation showcases various sampling techniques including weighted sampling, hard negative mining, and curriculum learning approaches.

```python
import numpy as np
from torch.utils.data import WeightedRandomSampler

class AdvancedBatchSampler:
    def __init__(self, dataset_size, batch_size):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.sample_weights = np.ones(dataset_size)
        self.difficulty_scores = np.zeros(dataset_size)
        
    def update_sample_weights(self, indices, loss_values):
        # Update weights based on loss values
        self.sample_weights[indices] = np.exp(loss_values)
        self.sample_weights /= self.sample_weights.sum()
        
    def get_curriculum_batch(self, current_epoch, max_epochs):
        # Implement curriculum learning
        difficulty_threshold = current_epoch / max_epochs
        valid_samples = self.difficulty_scores <= difficulty_threshold
        
        # Sample based on weights and difficulty
        valid_indices = np.where(valid_samples)[0]
        valid_weights = self.sample_weights[valid_indices]
        
        sampled_indices = np.random.choice(
            valid_indices,
            size=self.batch_size,
            p=valid_weights/valid_weights.sum()
        )
        
        return sampled_indices

# Example usage
sampler = AdvancedBatchSampler(dataset_size=10000, batch_size=32)

# Simulate training loop
for epoch in range(10):
    batch_indices = sampler.get_curriculum_batch(epoch, max_epochs=10)
    # Simulate loss computation
    fake_losses = np.random.random(len(batch_indices))
    sampler.update_sample_weights(batch_indices, fake_losses)
    
    print(f"Epoch {epoch}: Mean batch weight = {sampler.sample_weights[batch_indices].mean():.4f}")
```

Slide 8: Distributed Batch Processing

Distributed computing enables processing larger batch sizes across multiple GPUs or machines. This implementation demonstrates how to distribute batch processing using PyTorch's DistributedDataParallel functionality.

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

class DistributedBatchProcessor:
    def __init__(self, model, rank, world_size):
        self.model = DDP(model.to(rank), device_ids=[rank])
        self.rank = rank
        self.world_size = world_size
        
    def process_batch(self, batch):
        # Split batch across GPUs
        local_batch_size = len(batch) // self.world_size
        start_idx = self.rank * local_batch_size
        end_idx = start_idx + local_batch_size
        local_batch = batch[start_idx:end_idx]
        
        # Process local batch
        output = self.model(local_batch)
        
        # Gather results from all processes
        gathered_outputs = [torch.zeros_like(output) for _ in range(self.world_size)]
        dist.all_gather(gathered_outputs, output)
        
        return torch.cat(gathered_outputs)

def run_distributed(rank, world_size):
    setup(rank, world_size)
    model = torch.nn.Linear(20, 10)
    processor = DistributedBatchProcessor(model, rank, world_size)
    
    # Example batch processing
    batch = torch.randn(32, 20)
    output = processor.process_batch(batch)
    
    if rank == 0:
        print(f"Final output shape: {output.shape}")

# Example usage
world_size = torch.cuda.device_count()
mp.spawn(run_distributed,
         args=(world_size,),
         nprocs=world_size,
         join=True)
```

Slide 9: Asynchronous Batch Processing

Asynchronous batch processing allows for improved throughput by overlapping computation and data loading. This implementation demonstrates how to implement asynchronous data loading and preprocessing while maintaining training efficiency.

```python
import torch
import threading
from queue import Queue
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor

class AsyncBatchProcessor:
    def __init__(self, dataset, batch_size, num_workers=4):
        self.batch_queue = Queue(maxsize=2 * num_workers)
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
    def preprocess_batch(self, batch):
        # Simulate complex preprocessing
        processed = torch.nn.functional.normalize(batch, dim=1)
        return processed.pin_memory()
        
    def start_async_loading(self):
        def load_batches():
            for batch in self.dataloader:
                processed = self.preprocess_batch(batch)
                self.batch_queue.put(processed)
            self.batch_queue.put(None)  # Signal end of data
            
        self.load_thread = threading.Thread(target=load_batches)
        self.load_thread.start()
        
    def get_next_batch(self):
        batch = self.batch_queue.get()
        if batch is None:
            return None
        return batch.cuda(non_blocking=True)

# Example usage
dataset = torch.randn(1000, 10)
processor = AsyncBatchProcessor(dataset, batch_size=32)
processor.start_async_loading()

# Training loop
while True:
    batch = processor.get_next_batch()
    if batch is None:
        break
    # Process batch
    print(f"Processed batch shape: {batch.shape}")
```

Slide 10: Dynamic Batch Sizing

Dynamic batch sizing adapts the batch size during training based on various metrics such as gradient noise scale and memory constraints. This implementation shows how to dynamically adjust batch sizes for optimal training performance.

```python
import torch
import numpy as np
from collections import deque

class DynamicBatchSizer:
    def __init__(self, init_batch_size, min_batch_size=16, max_batch_size=512):
        self.current_batch_size = init_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.grad_history = deque(maxlen=50)
        
    def compute_gradient_noise_scale(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_history.append(total_norm)
        
        if len(self.grad_history) < self.grad_history.maxlen:
            return None
            
        grad_variance = np.var(list(self.grad_history))
        grad_mean = np.mean(list(self.grad_history))
        return grad_variance / (grad_mean ** 2)
        
    def adjust_batch_size(self, model, loss):
        noise_scale = self.compute_gradient_noise_scale(model)
        
        if noise_scale is not None:
            # Increase batch size if gradient noise is high
            if noise_scale > 0.1:
                self.current_batch_size = min(
                    self.current_batch_size * 2,
                    self.max_batch_size
                )
            # Decrease batch size if gradient noise is low
            elif noise_scale < 0.01:
                self.current_batch_size = max(
                    self.current_batch_size // 2,
                    self.min_batch_size
                )
                
        return self.current_batch_size

# Example usage
model = torch.nn.Linear(10, 1)
batch_sizer = DynamicBatchSizer(init_batch_size=64)

# Training loop simulation
for epoch in range(5):
    X = torch.randn(batch_sizer.current_batch_size, 10)
    y = torch.randn(batch_sizer.current_batch_size, 1)
    
    output = model(X)
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()
    
    new_batch_size = batch_sizer.adjust_batch_size(model, loss)
    print(f"Epoch {epoch}: New batch size = {new_batch_size}")
```

Slide 11: Batch Size Impact on Model Convergence

Understanding the relationship between batch size and model convergence is crucial for efficient training. This implementation provides tools for analyzing and visualizing the impact of different batch sizes on training dynamics.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class BatchSizeAnalyzer:
    def __init__(self, model, criterion, batch_sizes=[16, 32, 64, 128]):
        self.model = model
        self.criterion = criterion
        self.batch_sizes = batch_sizes
        self.convergence_metrics = defaultdict(list)
        
    def analyze_convergence(self, X, y, epochs=10):
        for batch_size in self.batch_sizes:
            # Reset model
            self.model.apply(lambda m: m.reset_parameters() 
                           if hasattr(m, 'reset_parameters') else None)
            
            losses = []
            for epoch in range(epochs):
                # Split data into batches
                num_batches = len(X) // batch_size
                epoch_losses = []
                
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size
                    
                    batch_X = X[start_idx:end_idx]
                    batch_y = y[start_idx:end_idx]
                    
                    output = self.model(batch_X)
                    loss = self.criterion(output, batch_y)
                    loss.backward()
                    
                    epoch_losses.append(loss.item())
                
                losses.append(np.mean(epoch_losses))
                self.convergence_metrics[batch_size].append(losses[-1])
        
        return self.convergence_metrics
    
    def plot_convergence(self):
        plt.figure(figsize=(10, 6))
        for batch_size, losses in self.convergence_metrics.items():
            plt.plot(losses, label=f'Batch Size {batch_size}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Convergence Analysis for Different Batch Sizes')
        plt.legend()
        plt.grid(True)
        
        # Convert plot to ASCII art for text output
        print("Convergence Plot (ASCII representation):")
        print("-" * 50)
        for batch_size, losses in self.convergence_metrics.items():
            print(f"Batch Size {batch_size}:", end=" ")
            for loss in losses:
                print("*" if loss > np.mean(losses) else ".", end="")
            print()
        print("-" * 50)

# Example usage
model = torch.nn.Sequential(
    torch.nn.Linear(10, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 1)
)
analyzer = BatchSizeAnalyzer(model, torch.nn.MSELoss())

X = torch.randn(1000, 10)
y = torch.randn(1000, 1)

metrics = analyzer.analyze_convergence(X, y)
analyzer.plot_convergence()
```

Slide 12: Real-world Application: Image Classification with Batch Processing

This implementation demonstrates a complete image classification pipeline using batch processing, including data loading, augmentation, and training with proper batch handling for real-world datasets.

```python
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

class ImageClassifier:
    def __init__(self, num_classes, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)
        
        self.batch_size = batch_size
        self.scaler = GradScaler()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                              [0.229, 0.224, 0.225])
        ])
        
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(images)
                loss = criterion(outputs, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Batch: {batch_idx}, Loss: {loss.item():.3f}, '
                      f'Acc: {100.*correct/total:.2f}%')
                
        return total_loss / len(dataloader), 100.*correct/total

# Example usage
dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True,
    transform=transforms.ToTensor()
)

classifier = ImageClassifier(num_classes=10)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
optimizer = torch.optim.Adam(classifier.model.parameters())
criterion = nn.CrossEntropyLoss()

loss, acc = classifier.train_epoch(dataloader, optimizer, criterion)
print(f"Final Training Loss: {loss:.3f}, Accuracy: {acc:.2f}%")
```

Slide 13: Real-world Application: Natural Language Processing with Dynamic Batching

This implementation shows how to handle variable-length sequences in NLP tasks using dynamic batching and attention masking, demonstrating efficient batch processing for text data.

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from transformers import BertTokenizer

class DynamicBatchNLP:
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)  # Binary classification
        
    def prepare_batch(self, texts):
        # Tokenize and create attention masks
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        return input_ids, attention_mask
        
    def forward(self, input_ids, attention_mask):
        # Get embeddings
        embedded = self.embedding(input_ids)
        
        # Pack padded sequence
        lengths = attention_mask.sum(dim=1)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Process through LSTM
        output, (hidden, _) = self.lstm(packed)
        
        # Get final hidden state and classify
        logits = self.classifier(hidden[-1])
        return logits

# Example usage
model = DynamicBatchNLP(vocab_size=30522).to('cuda')  # BERT vocab size

# Sample texts of different lengths
texts = [
    "This is a short sentence.",
    "This is a much longer sentence with more words to process.",
    "Medium length sentence here."
]

input_ids, attention_mask = model.prepare_batch(texts)
logits = model.forward(input_ids, attention_mask)
print(f"Output logits shape: {logits.shape}")
print(f"Predictions: {torch.softmax(logits, dim=1)}")
```

Slide 14: Additional Resources

*   ArXiv Paper: "Large Batch Training of Convolutional Networks"
    *   [https://arxiv.org/abs/1708.03888](https://arxiv.org/abs/1708.03888)
*   ArXiv Paper: "Don't Decay the Learning Rate, Increase the Batch Size"
    *   [https://arxiv.org/abs/1711.00489](https://arxiv.org/abs/1711.00489)
*   ArXiv Paper: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
    *   [https://arxiv.org/abs/1706.02677](https://arxiv.org/abs/1706.02677)
*   Research Article: "Batch Size in Deep Learning"
    *   [https://www.deeplearning.ai/batch-size-in-deep-learning](https://www.deeplearning.ai/batch-size-in-deep-learning)
*   Technical Guide: "Effective Batch Processing in Neural Networks"
    *   [https://developers.google.com/machine-learning/guides/batch-processing](https://developers.google.com/machine-learning/guides/batch-processing)

