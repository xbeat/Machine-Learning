## Mixed Precision Training Fundamentals
Slide 1: Mixed Precision Training Fundamentals

Mixed precision training combines different numerical precisions during model training to optimize memory usage and computational speed while maintaining model accuracy. This approach primarily uses float16 (FP16) for computations and float32 (FP32) for critical operations.

```python
import torch
import torch.cuda.amp as amp

# Initialize model, optimizer, and scaler
model = torch.nn.Linear(10, 1).cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = amp.GradScaler()  # Handles FP16 gradient scaling

# Training loop with mixed precision
with amp.autocast():
    output = model(input)  # Automatic FP16 computation
    loss = criterion(output, target)

# Scale loss and backpropagate
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Slide 2: Memory Management in Mixed Precision

Understanding memory allocation patterns in mixed precision training requires analyzing how different data types occupy GPU memory. FP32 uses 4 bytes per number, while FP16 uses 2 bytes, effectively halving memory requirements for applicable operations.

```python
import numpy as np
import torch

# Compare memory usage
def compare_memory_usage(size=(1000, 1000)):
    # FP32 tensor
    fp32_tensor = torch.randn(size, dtype=torch.float32)
    fp32_memory = fp32_tensor.element_size() * fp32_tensor.nelement()
    
    # FP16 tensor
    fp16_tensor = torch.randn(size, dtype=torch.float16)
    fp16_memory = fp16_tensor.element_size() * fp16_tensor.nelement()
    
    print(f"FP32 Memory: {fp32_memory/1e6:.2f} MB")
    print(f"FP16 Memory: {fp16_memory/1e6:.2f} MB")
    print(f"Memory Reduction: {(1 - fp16_memory/fp32_memory)*100:.1f}%")

compare_memory_usage()
```

Slide 3: Loss Scaling Implementation

Loss scaling prevents underflow in FP16 computations by multiplying the loss by a large factor before backpropagation and dividing gradients by the same factor afterward. This maintains small gradient values that might otherwise become zero in FP16.

```python
class CustomMixedPrecisionTrainer:
    def __init__(self, model, optimizer, scale_factor=2**15):
        self.model = model
        self.optimizer = optimizer
        self.scale_factor = scale_factor
        
    def training_step(self, inputs, targets):
        # Forward pass in FP16
        with torch.cuda.amp.autocast():
            outputs = self.model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)
            
        # Scale loss and backpropagate
        scaled_loss = loss * self.scale_factor
        scaled_loss.backward()
        
        # Unscale gradients and optimize
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data = param.grad.data / self.scale_factor
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
```

Slide 4: Numerical Stability in Mixed Precision

Managing numerical stability in mixed precision training requires careful consideration of operations that may cause overflow or underflow. Critical operations like batch normalization statistics are kept in FP32 for stability.

```python
import torch.nn as nn

class MixedPrecisionBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)
        
    def forward(self, x):
        # Cast input to FP32 for stable statistics computation
        if x.dtype == torch.float16:
            x_32 = x.float()
            out = self.bn(x_32)
            return out.half()  # Return to FP16
        return self.bn(x)
```

Slide 5: Dynamic Loss Scaling

Dynamic loss scaling automatically adjusts the scaling factor based on gradient behavior, preventing both overflow and underflow during training. The scaling factor increases when no overflow occurs and decreases after detecting infinite or NaN gradients.

```python
class DynamicLossScaler:
    def __init__(self, init_scale=2**15, scale_growth=2., backoff=0.5, growth_interval=100):
        self.current_scale = init_scale
        self.scale_growth = scale_growth
        self.backoff = backoff
        self.growth_interval = growth_interval
        self.steps = 0
        self.inf_checks = 0
        
    def scale_loss(self, loss):
        return loss * self.current_scale
        
    def unscale_gradients(self, optimizer):
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data /= self.current_scale
                    
    def update_scale(self, has_inf_grad):
        if has_inf_grad:
            self.current_scale *= self.backoff
            self.steps = 0
        else:
            self.inf_checks += 1
            if self.inf_checks >= self.growth_interval:
                self.current_scale *= self.scale_growth
                self.inf_checks = 0
```

Slide 6: Implementing Custom Mixed Precision Training Loop

A complete implementation showcasing mixed precision training with dynamic loss scaling, gradient clipping, and performance monitoring. This code demonstrates the integration of all core mixed precision concepts.

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time

def train_epoch(model, loader, optimizer, criterion, scaler, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            output = model(data)
            loss = criterion(output, target)
            
        # Scale loss and backward pass
        scaler.scale(loss).backward()
        
        # Unscale gradients and clip
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Update weights
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(loader)
    throughput = len(loader.dataset) / (time.time() - start_time)
    
    return avg_loss, throughput
```

Slide 7: Real-world Example - ResNet Training with Mixed Precision

Implementation of mixed precision training for ResNet architecture on ImageNet dataset, demonstrating significant speedup while maintaining accuracy. This example includes data preprocessing and training pipeline.

```python
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

# Data preprocessing
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Model setup
model = torchvision.models.resnet50().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scaler = GradScaler()

def train_resnet():
    for epoch in range(90):
        adjust_learning_rate(optimizer, epoch)
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler)
        
        if epoch % 5 == 0:
            # Validation in FP32
            val_loss, val_acc = validate(model, val_loader, criterion)
            print(f'Epoch {epoch}: Train Loss {train_loss:.3f}, '
                  f'Val Loss {val_loss:.3f}, Val Acc {val_acc:.2%}')
```

Slide 8: Performance Monitoring and Metrics

Performance monitoring in mixed precision training requires tracking both computational efficiency and numerical stability. This implementation provides comprehensive metrics for throughput, memory usage, and gradient behavior.

```python
class PerformanceMonitor:
    def __init__(self):
        self.grad_stats = []
        self.throughput = []
        self.memory_stats = []
        
    def log_iteration(self, model, batch_size, iteration_time):
        # Track gradient statistics
        grad_norm = torch.stack([
            p.grad.norm() for p in model.parameters() if p.grad is not None
        ]).mean()
        self.grad_stats.append(grad_norm.item())
        
        # Calculate throughput
        throughput = batch_size / iteration_time
        self.throughput.append(throughput)
        
        # Memory tracking
        memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
        self.memory_stats.append(memory_allocated)
        
    def get_summary(self):
        return {
            'avg_throughput': np.mean(self.throughput),
            'avg_grad_norm': np.mean(self.grad_stats),
            'peak_memory': max(self.memory_stats)
        }
```

Slide 9: Handling Edge Cases in Mixed Precision

Edge cases in mixed precision training require special attention to prevent numerical instability. This implementation demonstrates handling of gradient overflow, underflow, and NaN values.

```python
class EdgeCaseHandler:
    def __init__(self, model):
        self.model = model
        self.nan_counter = 0
        self.inf_counter = 0
        
    def check_gradients(self):
        for param in self.model.parameters():
            if param.grad is not None:
                # Check for NaN
                if torch.isnan(param.grad).any():
                    self.nan_counter += 1
                    param.grad.zero_()
                    
                # Check for Inf
                if torch.isinf(param.grad).any():
                    self.inf_counter += 1
                    param.grad.zero_()
                    
                # Check for underflow
                grad_abs = param.grad.abs()
                if (grad_abs > 0).any() and (grad_abs < 1e-7).any():
                    param.grad.mul_(1e7)
                    
    def get_statistics(self):
        return {
            'nan_occurrences': self.nan_counter,
            'inf_occurrences': self.inf_counter
        }
```

Slide 10: Mathematical Foundations of Mixed Precision

```python
# Mathematical representation of mixed precision operations
# Note: LaTeX formulas are shown as strings for text export

"""
Weight Update Rule:
$$w_{t+1} = w_t - \eta \cdot \frac{\text{scale}(\nabla L)}{\text{scale}}$$

Loss Scaling:
$$L_{\text{scaled}} = \alpha \cdot L_{\text{original}}$$

Gradient Scaling:
$$\nabla L_{\text{scaled}} = \alpha \cdot \nabla L_{\text{original}}$$

Dynamic Scale Update:
$$\alpha_{t+1} = \begin{cases} 
\alpha_t \cdot \beta_{\text{increase}} & \text{if no overflow} \\
\alpha_t \cdot \beta_{\text{decrease}} & \text{if overflow}
\end{cases}$$
"""
```

Slide 11: Memory-Efficient Implementation Patterns

This slide demonstrates practical patterns for optimizing memory usage in mixed precision training, including gradient accumulation and efficient buffer management for large models and datasets.

```python
class MemoryEfficientTrainer:
    def __init__(self, model, accumulation_steps=4):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.scaler = GradScaler()
        
    def train_step(self, dataloader, optimizer):
        optimizer.zero_grad()
        accumulated_loss = 0
        
        for i, (data, target) in enumerate(dataloader):
            with autocast():
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                # Normalize loss to account for accumulation
                loss = loss / self.accumulation_steps
            
            # Scale and accumulate gradients
            self.scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            
            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
                
        return accumulated_loss * self.accumulation_steps
```

Slide 12: Real-world Example - BERT Fine-tuning

Implementation of mixed precision training for BERT fine-tuning on a classification task, showcasing memory optimization techniques and performance improvements.

```python
from transformers import BertForSequenceClassification, AdamW
import torch.distributed as dist

class BERTMixedPrecisionTrainer:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels).cuda()
        self.scaler = GradScaler()
        
    def train(self, train_dataloader, epochs=3, lr=2e-5):
        optimizer = AdamW(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_dataloader:
                # Move batch to GPU
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['labels'].cuda()
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                with autocast():
                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                
                # Scale loss and backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
```

Slide 13: Additional Resources

*   Mixed Precision Training of Deep Neural Networks: [https://arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740)
*   NVIDIA Apex: Mixed Precision Training Techniques: [https://developer.nvidia.com/blog/apex-training-techniques](https://developer.nvidia.com/blog/apex-training-techniques)
*   Mixed Precision Training for NLP Transformers: [https://arxiv.org/abs/1909.11556](https://arxiv.org/abs/1909.11556)
*   Automatic Mixed Precision in PyTorch: [https://pytorch.org/docs/stable/amp](https://pytorch.org/docs/stable/amp)
*   Mixed Precision Training: Theory and Practice: [https://arxiv.org/abs/1803.08383](https://arxiv.org/abs/1803.08383)

