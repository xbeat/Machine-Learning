## Techniques for Compressing Large Language Models in Python
Slide 1: Introduction to LLM Model Compression

Model compression is crucial for deploying large language models (LLMs) in resource-constrained environments. It aims to reduce model size and computational requirements while maintaining performance. This slideshow explores common techniques for LLM model compression using Python.

```python
import torch
import transformers

# Load a pre-trained model
model = transformers.AutoModel.from_pretrained("bert-base-uncased")

# Print model size
print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
```

Slide 2: Pruning - Removing Unimportant Weights

Pruning involves removing less important weights from the model. This technique can significantly reduce model size with minimal impact on performance. We'll demonstrate magnitude-based pruning.

```python
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    
    return model

pruned_model = prune_model(model)
print(f"Pruned model size: {sum(p.numel() for p in pruned_model.parameters() if p.requires_grad) / 1e6:.2f}M parameters")
```

Slide 3: Quantization - Reducing Numerical Precision

Quantization reduces the numerical precision of model weights and activations. This technique can significantly decrease model size and inference time. We'll demonstrate post-training static quantization.

```python
import torch.quantization

def quantize_model(model):
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

quantized_model = quantize_model(model)
print(f"Quantized model size: {sum(p.numel() for p in quantized_model.parameters()) / 1e6:.2f}M parameters")
```

Slide 4: Knowledge Distillation - Teacher-Student Learning

Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model. This technique can create compact models that retain much of the original model's performance.

```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    distillation_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    student_loss = F.cross_entropy(student_logits, labels)
    return alpha * distillation_loss + (1 - alpha) * student_loss

# Usage in training loop
student_logits = student_model(inputs)
teacher_logits = teacher_model(inputs)
loss = distillation_loss(student_logits, teacher_logits, labels)
```

Slide 5: Low-Rank Factorization

Low-rank factorization decomposes weight matrices into products of smaller matrices, reducing the number of parameters. This technique is particularly effective for compressing large linear layers.

```python
import torch.nn as nn

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.u = nn.Parameter(torch.randn(in_features, rank))
        self.v = nn.Parameter(torch.randn(rank, out_features))
    
    def forward(self, x):
        return x @ self.u @ self.v

# Replace a linear layer with a low-rank approximation
original_layer = nn.Linear(1000, 1000)
low_rank_layer = LowRankLinear(1000, 1000, rank=100)

print(f"Original parameters: {original_layer.weight.numel():,}")
print(f"Low-rank parameters: {low_rank_layer.u.numel() + low_rank_layer.v.numel():,}")
```

Slide 6: Weight Sharing

Weight sharing reduces model size by using the same weights for multiple parts of the network. This technique is often used in conjunction with quantization or pruning.

```python
import torch.nn as nn

class SharedWeightLinear(nn.Module):
    def __init__(self, in_features, out_features, num_blocks):
        super().__init__()
        self.shared_weight = nn.Parameter(torch.randn(in_features, out_features))
        self.biases = nn.Parameter(torch.randn(num_blocks, out_features))
        self.num_blocks = num_blocks
    
    def forward(self, x):
        # x shape: (batch_size, num_blocks, in_features)
        return torch.einsum('bni,io->bno', x, self.shared_weight) + self.biases.unsqueeze(0)

shared_layer = SharedWeightLinear(1000, 1000, num_blocks=10)
print(f"Parameters: {sum(p.numel() for p in shared_layer.parameters()):,}")
```

Slide 7: Pruning with Gradual Magnitude Increase

This advanced pruning technique gradually increases the pruning threshold during training, allowing the model to adapt to sparsity over time.

```python
import torch.nn.utils.prune as prune

def gradual_pruning(model, initial_sparsity, final_sparsity, epochs):
    for epoch in range(epochs):
        sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * (epoch / epochs)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
        
        # Train the model for one epoch
        # ...

    return model

pruned_model = gradual_pruning(model, initial_sparsity=0.1, final_sparsity=0.5, epochs=10)
```

Slide 8: Mixed-Precision Training

Mixed-precision training uses lower precision (e.g., float16) for most operations while keeping critical computations in higher precision (float32). This reduces memory usage and can speed up training on compatible hardware.

```python
import torch.cuda.amp as amp

# Initialize model and optimizer
model = YourModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

# Create a GradScaler for mixed precision training
scaler = amp.GradScaler()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Runs the forward pass with autocasting
        with amp.autocast():
            outputs = model(batch)
            loss = loss_function(outputs, targets)
        
        # Scales loss and calls backward() to create scaled gradients
        scaler.scale(loss).backward()
        
        # Unscales gradients and calls or skips optimizer.step()
        scaler.step(optimizer)
        
        # Updates the scale for next iteration
        scaler.update()
```

Slide 9: Real-Life Example: Compressing BERT for Mobile Devices

In this example, we'll compress a BERT model for deployment on mobile devices using a combination of pruning and quantization.

```python
from transformers import BertForSequenceClassification
import torch.nn.utils.prune as prune

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Apply pruning
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)

# Apply quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the compressed model
torch.save(quantized_model.state_dict(), 'compressed_bert.pth')

print(f"Original size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
print(f"Compressed size: {sum(p.numel() for p in quantized_model.parameters()) / 1e6:.2f}M parameters")
```

Slide 10: Real-Life Example: Distilling GPT-2 for Text Generation

In this example, we'll create a smaller version of GPT-2 using knowledge distillation for efficient text generation.

```python
from transformers import GPT2LMHeadModel, GPT2Config
import torch.nn.functional as F

# Load the teacher model (GPT-2 small)
teacher_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Create a smaller student model
student_config = GPT2Config(n_layer=6, n_head=8, n_embd=512)
student_model = GPT2LMHeadModel(student_config)

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0):
    distill_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    nll_loss = F.cross_entropy(student_logits, labels)
    return 0.5 * distill_loss + 0.5 * nll_loss

# Training loop (simplified)
for batch in dataloader:
    teacher_logits = teacher_model(batch).logits
    student_logits = student_model(batch).logits
    loss = distillation_loss(student_logits, teacher_logits, batch['labels'])
    # Backward pass and optimization steps

print(f"Teacher model size: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.2f}M parameters")
print(f"Student model size: {sum(p.numel() for p in student_model.parameters()) / 1e6:.2f}M parameters")
```

Slide 11: Structured Pruning with Channel-wise Pruning

Structured pruning removes entire channels or neurons, which can lead to better hardware utilization compared to unstructured pruning. Here's an example of channel-wise pruning for convolutional layers.

```python
import torch
import torch.nn as nn

def channel_pruning(model, prune_ratio=0.5):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Compute L1-norm of each filter
            l1_norm = torch.sum(torch.abs(module.weight.data), dim=(1, 2, 3))
            num_channels = int(module.out_channels * (1 - prune_ratio))
            
            # Select top-k channels
            top_k_channels = torch.argsort(l1_norm, descending=True)[:num_channels]
            
            # Create a mask for selected channels
            mask = torch.zeros_like(module.weight.data)
            mask[top_k_channels, :, :, :] = 1
            
            # Apply the mask
            module.weight.data *= mask
            if module.bias is not None:
                module.bias.data[top_k_channels] *= 1
                module.bias.data[top_k_channels.bitwise_not()] *= 0

    return model

# Usage
pruned_model = channel_pruning(model, prune_ratio=0.3)
```

Slide 12: Adaptive Quantization

Adaptive quantization adjusts the quantization parameters based on the distribution of weights and activations. This can lead to better performance compared to uniform quantization.

```python
import torch
import torch.nn as nn

class AdaptiveQuantizer(nn.Module):
    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits
    
    def forward(self, x):
        min_val, max_val = x.min(), x.max()
        scale = (max_val - min_val) / (2**self.bits - 1)
        zero_point = (-min_val / scale).round()
        
        x_quant = torch.clamp(x / scale + zero_point, 0, 2**self.bits - 1).round()
        x_dequant = (x_quant - zero_point) * scale
        
        return x_dequant

# Usage
model = nn.Sequential(
    nn.Linear(784, 256),
    AdaptiveQuantizer(bits=8),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Forward pass
x = torch.randn(32, 784)
output = model(x)
```

Slide 13: Model Pruning with Lottery Ticket Hypothesis

The Lottery Ticket Hypothesis suggests that dense networks contain sparse subnetworks that can be trained to similar accuracy. This technique involves iterative pruning and retraining to find these "winning tickets".

```python
import 
import torch.nn.utils.prune as prune

def find_lottery_ticket(model, prune_amount=0.2, num_iterations=5):
    original_weights = .deep(model.state_dict())
    
    for iteration in range(num_iterations):
        # Prune the model
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=prune_amount)
        
        # Train the pruned model
        # ...
        
        # Reset weights to their original values, keeping only the pruned structure
        for name, param in model.named_parameters():
            if 'weight_mask' in name:
                continue
            param.data = original_weights[name] * param.data.bool().float()
    
    return model

# Usage
lottery_ticket_model = find_lottery_ticket(model)
```

Slide 14: Additional Resources

For more in-depth information on LLM model compression techniques, consider exploring these resources:

1. "To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression" (ArXiv:1710.01878)
2. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter" (ArXiv:1910.01108)
3. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" (ArXiv:1803.03635)
4. "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (ArXiv:1909.11942)

These papers provide detailed insights into various compression techniques and their applications in natural language processing.

