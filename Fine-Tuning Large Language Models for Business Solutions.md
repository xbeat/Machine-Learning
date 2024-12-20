## Fine-Tuning Large Language Models for Business Solutions
Slide 1: Mixed Precision Training Fundamentals

Mixed precision training combines different numerical precisions to optimize memory usage and computational speed while maintaining model accuracy. This technique primarily uses FP16 (half-precision) for most operations while keeping critical aggregations in FP32.

```python
import torch
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
    
    def training_step(self, batch):
        with autocast():
            loss = self.model(batch)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

# Usage Example
model = torch.nn.Linear(10, 2).cuda()
optimizer = torch.optim.Adam(model.parameters())
trainer = MixedPrecisionTrainer(model, optimizer)
```

Slide 2: Parameter-Efficient Fine-Tuning Implementation

Parameter-Efficient Fine-Tuning (PEFT) enables adaptation of large language models by updating only a small subset of parameters. This implementation demonstrates LoRA (Low-Rank Adaptation), a popular PEFT technique.

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = 1.0
        
    def forward(self, x):
        return x @ (self.lora_A @ self.lora_B) * self.scaling

    def merge_weights(self, base_layer):
        base_layer.weight.data += (self.lora_A @ self.lora_B).T * self.scaling
```

Slide 3: Quantization Techniques

Quantization reduces model size and inference time by converting floating-point weights to lower precision formats. This implementation shows both dynamic and static quantization approaches for PyTorch models.

```python
import torch.quantization

def quantize_model(model, calibration_data=None):
    # Configure model for static quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model)
    
    # Calibrate with sample data if provided
    if calibration_data is not None:
        with torch.no_grad():
            for data in calibration_data:
                model_prepared(data)
    
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)
    return model_quantized
```

Slide 4: Gradient Checkpointing Implementation

Gradient checkpointing trades computation for memory by storing selective activations during forward pass and recomputing others during backpropagation. This implementation shows how to apply checkpointing to a transformer model.

```python
import torch
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, x):
        # Use checkpointing for attention computation
        attn_output = checkpoint(self.self_attn, x, x, x)
        return checkpoint(self.feed_forward, attn_output)
```

Slide 5: Gradient Accumulation Strategy

Gradient accumulation enables training with larger effective batch sizes by accumulating gradients over multiple forward and backward passes before updating model parameters. This is crucial for training with limited GPU memory.

```python
def train_with_accumulation(model, optimizer, data_loader, accumulation_steps=4):
    model.zero_grad()
    
    for i, batch in enumerate(data_loader):
        loss = model(batch) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
            
        return loss.item() * accumulation_steps
```

Slide 6: Weight Pruning Techniques

Weight pruning reduces model size by removing less important weights based on magnitude or other importance metrics. This implementation demonstrates structured and unstructured pruning approaches for neural networks.

```python
def prune_weights(model, pruning_ratio=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Calculate weight importance
            importance = torch.abs(module.weight.data)
            threshold = torch.quantile(importance, pruning_ratio)
            
            # Create pruning mask
            mask = importance > threshold
            module.weight.data *= mask
            
            # Track sparsity
            sparsity = (mask == 0).float().mean().item()
            print(f"Layer {name} sparsity: {sparsity:.2%}")
```

Slide 7: Knowledge Distillation Framework

Knowledge distillation transfers knowledge from a larger teacher model to a smaller student model. This implementation shows temperature-based distillation with custom loss functions.

```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, labels):
        soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        
        distillation_loss = self.kl_div(soft_prob, soft_targets) * (self.temperature ** 2)
        student_loss = nn.functional.cross_entropy(student_logits, labels)
        
        return self.alpha * student_loss + (1 - self.alpha) * distillation_loss
```

Slide 8: Implementing Attention Mechanisms

Attention mechanisms enable models to focus on relevant parts of input sequences. This implementation shows scaled dot-product attention with mask support.

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, value), attention_weights

# Usage example
q = torch.randn(32, 8, 64)  # (batch_size, seq_len, d_model)
k = torch.randn(32, 8, 64)
v = torch.randn(32, 8, 64)
output, weights = scaled_dot_product_attention(q, k, v)
```

Slide 9: Optimization Techniques

Advanced optimization techniques combine learning rate scheduling, gradient clipping, and weight decay for stable training. This implementation showcases a comprehensive optimization setup.

```python
class OptimizationHandler:
    def __init__(self, model, lr=1e-4, warmup_steps=1000):
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=warmup_steps
        )
        self.max_grad_norm = 1.0
    
    def step(self, loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters=self.model.parameters(),
            max_norm=self.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
```

Slide 10: Real-world Example - Text Classification

Implementation of a complete text classification pipeline using PEFT and quantization techniques on a transformer model.

```python
import transformers
from datasets import load_dataset

class EfficientClassifier:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        
        # Apply LoRA
        self.lora_config = LoRAConfig(
            r=8, 
            target_modules=["query", "value"]
        )
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Quantize model
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
    
    def train(self, train_dataset, eval_dataset):
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            gradient_accumulation_steps=4,
            fp16=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer.train()
```

Slide 11: Performance Monitoring Implementation

Real-time monitoring of model training metrics including loss curves, gradient norms, and memory usage. Implements custom callbacks for detailed performance tracking.

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        
    def log_metrics(self, phase, loss, grad_norm=None):
        self.metrics[f'{phase}_loss'].append(loss)
        if grad_norm:
            self.metrics['gradient_norm'].append(grad_norm)
        
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        self.metrics['gpu_memory'].append(memory_used)
        
    def plot_metrics(self):
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(self.metrics['train_loss'], label='Training Loss')
        ax1.plot(self.metrics['val_loss'], label='Validation Loss')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.plot(self.metrics['gradient_norm'], label='Gradient Norm')
        ax2.set_ylabel('Norm')
        ax2.legend()
```

Slide 12: Model Compression Pipeline

End-to-end pipeline combining quantization, pruning, and knowledge distillation for optimal model compression while maintaining performance.

```python
class ModelCompressor:
    def __init__(self, teacher_model, compression_ratio=0.3):
        self.teacher = teacher_model
        self.student = self._create_student_model()
        self.compression_ratio = compression_ratio
        
    def _create_student_model(self):
        config = self.teacher.config
        config.hidden_size = config.hidden_size // 2
        config.num_attention_heads = config.num_attention_heads // 2
        return AutoModelForSequenceClassification.from_config(config)
    
    def compress(self):
        # Apply pruning
        prune_weights(self.student, self.compression_ratio)
        
        # Setup distillation
        distill_loss = DistillationLoss(temperature=2.0)
        
        # Quantize student model
        self.student = quantize_model(self.student)
        
        return self.student
```

Slide 13: Efficient Fine-tuning Results Visualization

Implementation of visualization tools for comparing different fine-tuning approaches and their impact on model performance.

```python
def visualize_finetuning_results(baseline_metrics, peft_metrics, quantized_metrics):
    import seaborn as sns
    
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy comparison
    sns.lineplot(data={
        'Baseline': baseline_metrics['accuracy'],
        'PEFT': peft_metrics['accuracy'],
        'Quantized': quantized_metrics['accuracy']
    })
    
    # Add memory usage annotations
    memory_usage = {
        'Baseline': torch.cuda.max_memory_allocated() / 1e9,
        'PEFT': peft_metrics['memory_usage'],
        'Quantized': quantized_metrics['memory_usage']
    }
    
    plt.title('Model Performance Comparison')
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
```

Slide 14: Additional Resources

*   ArXiv Paper: "LoRA: Low-Rank Adaptation of Large Language Models"
    *   [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
*   ArXiv Paper: "QLoRA: Efficient Finetuning of Quantized LLMs"
    *   [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
*   ArXiv Paper: "The Power of Scale for Parameter-Efficient Prompt Tuning"
    *   [https://arxiv.org/abs/2104.08691](https://arxiv.org/abs/2104.08691)
*   General Resources:
    *   Hugging Face Documentation: [https://huggingface.co/docs](https://huggingface.co/docs)
    *   PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
    *   Microsoft DeepSpeed: [https://www.deepspeed.ai/getting-started/](https://www.deepspeed.ai/getting-started/)

