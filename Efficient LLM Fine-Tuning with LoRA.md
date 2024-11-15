## Efficient LLM Fine-Tuning with LoRA
Slide 1: LoRA Implementation Fundamentals

Low-Rank Adaptation fundamentally works by decomposing weight updates into smaller matrices through SVD-like decomposition, enabling efficient fine-tuning while maintaining model quality. The implementation starts with basic matrix operations to demonstrate the core concept.

```python
import numpy as np

class LoRALayer:
    def __init__(self, in_features, out_features, rank=4):
        self.rank = rank
        # Initialize matrices A and B for low-rank decomposition
        self.A = np.random.randn(in_features, rank) / np.sqrt(rank)
        self.B = np.random.randn(rank, out_features) / np.sqrt(rank)
        
    def forward(self, x):
        # Compute low-rank update
        return x @ (self.A @ self.B)

# Example usage
layer = LoRALayer(768, 768, rank=8)
input_tensor = np.random.randn(32, 768)
output = layer.forward(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print(f"Parameter reduction: {(768*8 + 8*768)/(768*768)*100:.2f}%")
```

Slide 2: QLoRA Implementation with 4-bit Quantization

Quantized LoRA reduces memory usage through 4-bit quantization while maintaining model performance. This implementation demonstrates the basic quantization process and dequantization during forward pass.

```python
import numpy as np

class QuantizedLoRALayer:
    def __init__(self, in_features, out_features, rank=4, bits=4):
        self.rank = rank
        self.bits = bits
        self.scale = (2**(bits-1)) - 1
        
        # Initialize and quantize matrices
        self.A = self._quantize(np.random.randn(in_features, rank) / np.sqrt(rank))
        self.B = self._quantize(np.random.randn(rank, out_features) / np.sqrt(rank))
        
    def _quantize(self, x):
        # Quantize to n-bit integers
        return np.round(np.clip(x * self.scale, -self.scale, self.scale))
    
    def _dequantize(self, x):
        # Convert back to floating point
        return x / self.scale
    
    def forward(self, x):
        # Dequantize during forward pass
        A_dequant = self._dequantize(self.A)
        B_dequant = self._dequantize(self.B)
        return x @ (A_dequant @ B_dequant)

# Example usage
layer = QuantizedLoRALayer(768, 768, rank=8, bits=4)
input_tensor = np.random.randn(32, 768)
output = layer.forward(input_tensor)
print(f"Memory savings: {(32/4)*100:.2f}%")
```

Slide 3: Dynamic Rank Adjustment in DyLoRA

DyLoRA introduces adaptive rank selection during training, optimizing the trade-off between model capacity and computational efficiency. This implementation shows the dynamic rank adjustment mechanism based on loss gradients.

```python
import numpy as np
from typing import List

class DyLoRALayer:
    def __init__(self, in_features, out_features, ranks: List[int] = [4, 8, 16]):
        self.ranks = ranks
        self.current_rank_idx = 0
        self.matrices = {
            rank: (np.random.randn(in_features, rank) / np.sqrt(rank),
                  np.random.randn(rank, out_features) / np.sqrt(rank))
            for rank in ranks
        }
    
    def adjust_rank(self, loss_gradient):
        # Simulate rank adjustment based on loss gradient
        gradient_norm = np.linalg.norm(loss_gradient)
        if gradient_norm > 1.0 and self.current_rank_idx < len(self.ranks) - 1:
            self.current_rank_idx += 1
        elif gradient_norm < 0.1 and self.current_rank_idx > 0:
            self.current_rank_idx -= 1
            
    def forward(self, x):
        current_rank = self.ranks[self.current_rank_idx]
        A, B = self.matrices[current_rank]
        return x @ (A @ B)

# Example usage
layer = DyLoRALayer(768, 768)
input_tensor = np.random.randn(32, 768)
fake_gradient = np.random.randn(32, 768)

for _ in range(3):
    output = layer.forward(input_tensor)
    layer.adjust_rank(fake_gradient)
    print(f"Current rank: {layer.ranks[layer.current_rank_idx]}")
```

Slide 4: LoRA-Drop Implementation

LoRA-Drop enhances training efficiency by selectively dropping certain adaptation matrices during training, similar to dropout but specifically designed for low-rank adaptations. This implementation demonstrates the core dropping mechanism.

```python
import numpy as np

class LoRADropLayer:
    def __init__(self, in_features, out_features, rank=4, drop_prob=0.3):
        self.rank = rank
        self.drop_prob = drop_prob
        self.A = np.random.randn(in_features, rank) / np.sqrt(rank)
        self.B = np.random.randn(rank, out_features) / np.sqrt(rank)
        self.drop_mask = None
        
    def create_drop_mask(self, batch_size):
        # Create binary mask for dropping
        return np.random.binomial(1, 1-self.drop_prob, 
                                size=(batch_size, self.rank))
        
    def forward(self, x, training=True):
        if training:
            self.drop_mask = self.create_drop_mask(x.shape[0])
            # Apply mask and scale
            dropped_output = x @ (self.A @ self.B)
            return dropped_output * self.drop_mask[:, :, np.newaxis]
        return x @ (self.A @ self.B)

# Example usage
layer = LoRADropLayer(768, 768, rank=8)
input_tensor = np.random.randn(32, 768)

# Training mode
train_output = layer.forward(input_tensor, training=True)
print(f"Training output shape: {train_output.shape}")
print(f"Active adaptations: {np.mean(layer.drop_mask) * 100:.2f}%")

# Inference mode
eval_output = layer.forward(input_tensor, training=False)
print(f"Inference output shape: {eval_output.shape}")
```

Slide 5: DoRA Implementation with Weight Decomposition

DoRA separates pretrained weights into magnitude and direction components, allowing for more stable updates during fine-tuning while maintaining the original model's knowledge. The implementation shows this decomposition process.

```python
import numpy as np

class DoRALayer:
    def __init__(self, in_features, out_features, rank=4):
        # Initialize pretrained weights
        self.pretrained_weights = np.random.randn(in_features, out_features)
        
        # Decompose into magnitude and direction
        self.magnitude = np.linalg.norm(self.pretrained_weights, axis=1, keepdims=True)
        self.direction = self.pretrained_weights / (self.magnitude + 1e-6)
        
        # LoRA matrices for updates
        self.A = np.random.randn(in_features, rank) / np.sqrt(rank)
        self.B = np.random.randn(rank, out_features) / np.sqrt(rank)
        
    def forward(self, x, alpha=0.1):
        # Combine original decomposed weights with LoRA updates
        lora_update = self.A @ self.B
        updated_direction = self.direction + alpha * lora_update
        updated_direction = updated_direction / (np.linalg.norm(updated_direction, 
                                              axis=1, keepdims=True) + 1e-6)
        return x @ (self.magnitude * updated_direction)

# Example usage
layer = DoRALayer(512, 512, rank=8)
input_tensor = np.random.randn(16, 512)
output = layer.forward(input_tensor)
print(f"Output shape: {output.shape}")
print(f"Direction matrix norm: {np.mean(np.linalg.norm(layer.direction, axis=1)):.4f}")
```

Slide 6: Efficient Parameter Merging

Implementation of efficient parameter merging for LoRA adaptation, enabling seamless integration of multiple fine-tuned adaptations into a single model while maintaining performance characteristics.

```python
import numpy as np
from typing import List, Tuple

class LoRAMerger:
    def __init__(self, base_model_dim: int, rank: int):
        self.base_dim = base_model_dim
        self.rank = rank
        self.adaptations: List[Tuple[np.ndarray, np.ndarray]] = []
        
    def add_adaptation(self, A: np.ndarray, B: np.ndarray, weight: float = 1.0):
        """Add a new LoRA adaptation with specified weight."""
        self.adaptations.append((A * weight, B))
        
    def merge(self) -> np.ndarray:
        """Merge all adaptations into a single weight update matrix."""
        merged_weights = np.zeros((self.base_dim, self.base_dim))
        
        for A, B in self.adaptations:
            merged_weights += A @ B
            
        return merged_weights

# Example usage
merger = LoRAMerger(base_model_dim=768, rank=8)

# Add multiple adaptations
for i in range(3):
    A = np.random.randn(768, 8) / np.sqrt(8)
    B = np.random.randn(8, 768) / np.sqrt(8)
    merger.add_adaptation(A, B, weight=0.3)

final_weights = merger.merge()
print(f"Merged weights shape: {final_weights.shape}")
print(f"Parameter compression ratio: {(768*8*2*3)/(768*768)*100:.2f}%")
```

Slide 7: Training Loop Implementation

A complete training loop implementation for LoRA fine-tuning, demonstrating gradient computation, weight updates, and adaptation scheduling during the training process.

```python
import numpy as np
from typing import Optional

class LoRATrainer:
    def __init__(self, model_dim: int, rank: int, learning_rate: float = 0.001):
        self.model_dim = model_dim
        self.rank = rank
        self.lr = learning_rate
        
        # Initialize LoRA matrices
        self.A = np.random.randn(model_dim, rank) / np.sqrt(rank)
        self.B = np.random.randn(rank, model_dim) / np.sqrt(rank)
        
        # Gradient accumulators
        self.A_grad = np.zeros_like(self.A)
        self.B_grad = np.zeros_like(self.B)
        
    def compute_gradients(self, input_data: np.ndarray, 
                         target: np.ndarray, 
                         base_output: Optional[np.ndarray] = None):
        # Simplified gradient computation
        lora_output = input_data @ (self.A @ self.B)
        if base_output is not None:
            total_output = base_output + lora_output
        else:
            total_output = lora_output
            
        # Compute error
        error = total_output - target
        
        # Compute gradients
        self.A_grad = input_data.T @ (error @ self.B.T)
        self.B_grad = self.A.T @ (input_data.T @ error)
        
        return np.mean(error ** 2)  # Return MSE loss
        
    def update_weights(self):
        # Apply gradients
        self.A -= self.lr * self.A_grad
        self.B -= self.lr * self.B_grad
        
        # Reset gradients
        self.A_grad.fill(0)
        self.B_grad.fill(0)

# Example training loop
trainer = LoRATrainer(model_dim=512, rank=8, learning_rate=0.001)
batch_size = 32

# Simulate training data
input_data = np.random.randn(batch_size, 512)
target_data = np.random.randn(batch_size, 512)

# Training loop
for epoch in range(5):
    loss = trainer.compute_gradients(input_data, target_data)
    trainer.update_weights()
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
```

Slide 8: Memory-Efficient Backpropagation

Implementation of memory-efficient backpropagation for LoRA, utilizing gradient checkpointing and selective computation to reduce memory requirements during training.

```python
import numpy as np
from typing import List, Optional

class MemoryEfficientLoRA:
    def __init__(self, 
                 layer_dims: List[int], 
                 rank: int,
                 checkpoint_freq: int = 2):
        self.dims = layer_dims
        self.rank = rank
        self.checkpoint_freq = checkpoint_freq
        
        # Initialize layers
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append({
                'A': np.random.randn(layer_dims[i], rank) / np.sqrt(rank),
                'B': np.random.randn(rank, layer_dims[i+1]) / np.sqrt(rank)
            })
            
        self.checkpoints: List[Optional[np.ndarray]] = []
        
    def forward(self, x: np.ndarray, store_checkpoints: bool = True) -> np.ndarray:
        self.checkpoints = []
        current = x
        
        for i, layer in enumerate(self.layers):
            if store_checkpoints and i % self.checkpoint_freq == 0:
                self.checkpoints.append(current.copy())
            else:
                self.checkpoints.append(None)
                
            current = current @ (layer['A'] @ layer['B'])
            
        return current
    
    def backward(self, grad_output: np.ndarray, learning_rate: float = 0.001):
        current_grad = grad_output
        
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            
            # Recompute forward pass if needed
            if self.checkpoints[i] is None:
                current = self.checkpoints[max(0, i - self.checkpoint_freq)]
                for j in range(max(0, i - self.checkpoint_freq), i):
                    current = current @ (self.layers[j]['A'] @ self.layers[j]['B'])
            else:
                current = self.checkpoints[i]
                
            # Compute gradients
            A_grad = current.T @ (current_grad @ layer['B'].T)
            B_grad = layer['A'].T @ (current.T @ current_grad)
            
            # Update weights
            layer['A'] -= learning_rate * A_grad
            layer['B'] -= learning_rate * B_grad
            
            # Update gradient for next layer
            current_grad = current_grad @ layer['B'].T @ layer['A'].T

# Example usage
model = MemoryEfficientLoRA(layer_dims=[768, 512, 256], rank=8)
input_data = np.random.randn(32, 768)
output = model.forward(input_data)
grad_output = np.random.randn(*output.shape)
model.backward(grad_output)

print(f"Peak memory checkpoints: {len(model.checkpoints)}")
print(f"Memory savings: {(1 - len([c for c in model.checkpoints if c is not None])/len(model.checkpoints))*100:.1f}%")
```

Slide 9: Adaptive Rank Selection for DyLoRA

Implementation of an adaptive rank selection mechanism that automatically adjusts the LoRA rank based on task complexity and performance metrics during training, optimizing computational resources.

```python
import numpy as np
from typing import List, Dict

class AdaptiveRankLoRA:
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 rank_options: List[int] = [4, 8, 16, 32],
                 eval_interval: int = 100):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank_options = rank_options
        self.eval_interval = eval_interval
        self.step_counter = 0
        self.performance_history: Dict[int, List[float]] = {r: [] for r in rank_options}
        
        # Initialize multiple ranks
        self.adaptations = {
            rank: self._init_matrices(rank) for rank in rank_options
        }
        self.current_rank = rank_options[0]
        
    def _init_matrices(self, rank: int) -> tuple:
        A = np.random.randn(self.input_dim, rank) / np.sqrt(rank)
        B = np.random.randn(rank, self.output_dim) / np.sqrt(rank)
        return (A, B)
    
    def evaluate_rank_performance(self, loss: float):
        self.performance_history[self.current_rank].append(loss)
        
        if self.step_counter % self.eval_interval == 0:
            # Calculate moving average of performance
            recent_performance = np.mean(self.performance_history[self.current_rank][-10:])
            
            # Adjust rank based on performance
            current_idx = self.rank_options.index(self.current_rank)
            if recent_performance > 0.1 and current_idx < len(self.rank_options) - 1:
                self.current_rank = self.rank_options[current_idx + 1]
            elif recent_performance < 0.05 and current_idx > 0:
                self.current_rank = self.rank_options[current_idx - 1]
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        A, B = self.adaptations[self.current_rank]
        self.step_counter += 1
        return x @ (A @ B)

# Example usage
model = AdaptiveRankLoRA(input_dim=512, output_dim=512)
for step in range(300):
    input_data = np.random.randn(16, 512)
    output = model.forward(input_data)
    
    # Simulate loss computation
    fake_loss = 0.1 * np.exp(-step/100)
    model.evaluate_rank_performance(fake_loss)
    
    if step % 100 == 0:
        print(f"Step {step}, Current Rank: {model.current_rank}")
        print(f"Loss: {fake_loss:.4f}")
```

Slide 10: LoRA Cross-Attention Implementation

An implementation of LoRA for transformer cross-attention layers, demonstrating how to apply low-rank adaptation to attention mechanisms while maintaining computational efficiency.

```python
import numpy as np

class LoRACrossAttention:
    def __init__(self, 
                 hidden_size: int,
                 num_heads: int,
                 lora_rank: int = 8,
                 lora_alpha: float = 16):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.lora_rank = lora_rank
        self.scaling = lora_alpha / lora_rank
        
        # Initialize LoRA matrices for Q, K, V
        self.query_lora = self._init_lora_pair()
        self.key_lora = self._init_lora_pair()
        self.value_lora = self._init_lora_pair()
        
    def _init_lora_pair(self) -> tuple:
        A = np.random.randn(self.hidden_size, self.lora_rank) / np.sqrt(self.lora_rank)
        B = np.random.randn(self.lora_rank, self.hidden_size) / np.sqrt(self.lora_rank)
        return (A, B)
    
    def _apply_lora(self, x: np.ndarray, lora_pair: tuple) -> np.ndarray:
        A, B = lora_pair
        return x @ (A @ B) * self.scaling
    
    def forward(self, 
                query: np.ndarray,
                key: np.ndarray,
                value: np.ndarray,
                mask: np.ndarray = None) -> np.ndarray:
        batch_size = query.shape[0]
        
        # Apply LoRA to Q, K, V
        q = query + self._apply_lora(query, self.query_lora)
        k = key + self._apply_lora(key, self.key_lora)
        v = value + self._apply_lora(value, self.value_lora)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        attention_weights = np.softmax(scores, axis=-1)
        
        # Apply attention to values
        output = np.matmul(attention_weights, v)
        output = output.reshape(batch_size, -1, self.hidden_size)
        
        return output

# Example usage
attention = LoRACrossAttention(hidden_size=512, num_heads=8)
batch_size, seq_length = 4, 32

# Generate sample inputs
query = np.random.randn(batch_size, seq_length, 512)
key = np.random.randn(batch_size, seq_length, 512)
value = np.random.randn(batch_size, seq_length, 512)

# Create attention mask (optional)
mask = np.triu(np.ones((seq_length, seq_length)) * -1e9, k=1)

output = attention.forward(query, key, value, mask)
print(f"Output shape: {output.shape}")
print(f"Memory efficiency: {(3*512*8 + 8*512)/(512*512*3)*100:.2f}% of original parameters")
```

Slide 11: LoRA with Gradient Checkpointing

Advanced implementation of LoRA utilizing gradient checkpointing for handling larger models while maintaining memory efficiency during training phases.

```python
import numpy as np
from typing import Optional, List, Tuple

class CheckpointedLoRA:
    def __init__(self, 
                 layer_sizes: List[int],
                 lora_rank: int,
                 checkpoint_granularity: int = 2):
        self.layer_sizes = layer_sizes
        self.rank = lora_rank
        self.granularity = checkpoint_granularity
        
        # Initialize layers
        self.layers = []
        for i in range(len(layer_sizes)-1):
            self.layers.append({
                'A': np.random.randn(layer_sizes[i], lora_rank) / np.sqrt(lora_rank),
                'B': np.random.randn(lora_rank, layer_sizes[i+1]) / np.sqrt(lora_rank),
                'activation': np.zeros((0,))  # Placeholder for checkpointing
            })
            
    def _forward_segment(self, 
                        x: np.ndarray, 
                        start_idx: int, 
                        end_idx: int) -> np.ndarray:
        current = x
        for i in range(start_idx, end_idx):
            current = current @ (self.layers[i]['A'] @ self.layers[i]['B'])
            current = np.maximum(current, 0)  # ReLU activation
        return current
    
    def forward(self, x: np.ndarray, save_memory: bool = True) -> np.ndarray:
        if not save_memory:
            return self._forward_segment(x, 0, len(self.layers))
            
        # Save intermediate activations at checkpoints
        checkpoints: List[Tuple[int, np.ndarray]] = [(0, x)]
        current = x
        
        for i in range(0, len(self.layers), self.granularity):
            end_idx = min(i + self.granularity, len(self.layers))
            current = self._forward_segment(current, i, end_idx)
            if end_idx < len(self.layers):
                checkpoints.append((end_idx, current.copy()))
                
        return current, checkpoints
    
    def backward(self, 
                grad_output: np.ndarray,
                checkpoints: List[Tuple[int, np.ndarray]],
                learning_rate: float = 0.001) -> None:
        current_grad = grad_output
        
        for checkpoint_idx in range(len(checkpoints)-1, -1, -1):
            start_idx, activation = checkpoints[checkpoint_idx]
            end_idx = min(start_idx + self.granularity, len(self.layers))
            
            # Recompute forward pass for segment
            segment_input = activation
            intermediate_values = []
            current = segment_input
            
            for i in range(start_idx, end_idx):
                intermediate_values.append(current)
                current = current @ (self.layers[i]['A'] @ self.layers[i]['B'])
                current = np.maximum(current, 0)  # ReLU
                
            # Backward pass for segment
            for i in range(end_idx-1, start_idx-1, -1):
                # Gradient through ReLU
                current_grad = current_grad * (intermediate_values[i-start_idx] > 0)
                
                # Compute gradients for LoRA matrices
                A_grad = intermediate_values[i-start_idx].T @ (current_grad @ self.layers[i]['B'].T)
                B_grad = self.layers[i]['A'].T @ (intermediate_values[i-start_idx].T @ current_grad)
                
                # Update weights
                self.layers[i]['A'] -= learning_rate * A_grad
                self.layers[i]['B'] -= learning_rate * B_grad
                
                # Propagate gradient
                if i > start_idx:
                    current_grad = current_grad @ self.layers[i]['B'].T @ self.layers[i]['A'].T

# Example usage
model = CheckpointedLoRA(layer_sizes=[512, 256, 128, 64], lora_rank=8)
batch_size = 32
input_data = np.random.randn(batch_size, 512)

# Forward pass with checkpointing
output, checkpoints = model.forward(input_data)
grad_output = np.random.randn(*output.shape)

# Backward pass using checkpoints
model.backward(grad_output, checkpoints)

print(f"Memory saved: {(1 - len(checkpoints)/len(model.layers))*100:.1f}%")
print(f"Output shape: {output.shape}")
```

Slide 12: Results Analysis and Visualization

Implementation of comprehensive metrics and visualization tools for analyzing LoRA adaptation performance and memory efficiency across different configurations.

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

class LoRAAnalyzer:
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            'memory_usage': [],
            'training_loss': [],
            'adaptation_efficiency': [],
            'rank_distribution': []
        }
        
    def compute_memory_usage(self, 
                           input_dim: int,
                           output_dim: int,
                           rank: int) -> float:
        full_params = input_dim * output_dim
        lora_params = rank * (input_dim + output_dim)
        return lora_params / full_params
    
    def track_metrics(self,
                     loss: float,
                     rank: int,
                     input_dim: int,
                     output_dim: int):
        memory = self.compute_memory_usage(input_dim, output_dim, rank)
        efficiency = 1.0 - loss * (memory ** 0.5)  # Simplified efficiency metric
        
        self.metrics['memory_usage'].append(memory)
        self.metrics['training_loss'].append(loss)
        self.metrics['adaptation_efficiency'].append(efficiency)
        self.metrics['rank_distribution'].append(rank)
    
    def plot_metrics(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot memory usage
        axes[0, 0].plot(self.metrics['memory_usage'])
        axes[0, 0].set_title('Memory Usage Over Time')
        axes[0, 0].set_ylabel('Memory Usage Ratio')
        
        # Plot training loss
        axes[0, 1].plot(self.metrics['training_loss'])
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_ylabel('Loss')
        
        # Plot efficiency
        axes[1, 0].plot(self.metrics['adaptation_efficiency'])
        axes[1, 0].set_title('Adaptation Efficiency')
        axes[1, 0].set_ylabel('Efficiency Score')
        
        # Plot rank distribution
        axes[1, 1].hist(self.metrics['rank_distribution'], bins=10)
        axes[1, 1].set_title('Rank Distribution')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig

# Example usage
analyzer = LoRAAnalyzer()

# Simulate training progression
input_dim, output_dim = 768, 768
ranks = [4, 8, 16, 32]
num_steps = 100

for step in range(num_steps):
    # Simulate training metrics
    rank = np.random.choice(ranks)
    loss = 1.0 * np.exp(-step/50)  # Simulated decreasing loss
    
    analyzer.track_metrics(loss, rank, input_dim, output_dim)

# Generate visualization
fig = analyzer.plot_metrics()

print("Final metrics:")
print(f"Average memory usage: {np.mean(analyzer.metrics['memory_usage']):.2%}")
print(f"Final loss: {analyzer.metrics['training_loss'][-1]:.4f}")
print(f"Average efficiency: {np.mean(analyzer.metrics['adaptation_efficiency']):.4f}")
```

Slide 13: Hybrid LoRA with Pruning

Implementation of a hybrid approach combining LoRA with weight pruning techniques to further optimize memory usage and computational efficiency during fine-tuning.

```python
import numpy as np
from typing import Dict, Tuple

class HybridLoRAPruning:
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 lora_rank: int = 8,
                 pruning_threshold: float = 0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = lora_rank
        self.threshold = pruning_threshold
        
        # Initialize LoRA matrices
        self.A = np.random.randn(input_dim, lora_rank) / np.sqrt(lora_rank)
        self.B = np.random.randn(lora_rank, output_dim) / np.sqrt(lora_rank)
        
        # Pruning masks
        self.A_mask = np.ones_like(self.A)
        self.B_mask = np.ones_like(self.B)
        
        # Importance scores
        self.importance_scores = np.zeros((lora_rank,))
        
    def _update_importance_scores(self):
        """Update importance scores based on parameter magnitudes"""
        a_scores = np.sum(np.abs(self.A), axis=0)
        b_scores = np.sum(np.abs(self.B), axis=1)
        self.importance_scores = a_scores * b_scores
        
    def prune_ranks(self):
        """Prune less important ranks based on threshold"""
        self._update_importance_scores()
        mask = self.importance_scores > (np.max(self.importance_scores) * self.threshold)
        
        # Apply masks
        self.A_mask = self.A_mask * mask[np.newaxis, :]
        self.B_mask = self.B_mask * mask[:, np.newaxis]
        
        # Apply pruning
        self.A = self.A * self.A_mask
        self.B = self.B * self.B_mask
        
        return np.sum(mask) / len(mask)  # Return pruning ratio
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with pruned LoRA matrices"""
        return x @ (self.A @ self.B)
    
    def update(self, 
               gradients: Dict[str, np.ndarray], 
               learning_rate: float = 0.001):
        """Update weights while maintaining pruning"""
        self.A -= learning_rate * gradients['A'] * self.A_mask
        self.B -= learning_rate * gradients['B'] * self.B_mask

# Example usage and evaluation
model = HybridLoRAPruning(input_dim=512, output_dim=512)
batch_size = 16

# Training simulation
for epoch in range(5):
    # Generate sample data
    x = np.random.randn(batch_size, 512)
    target = np.random.randn(batch_size, 512)
    
    # Forward pass
    output = model.forward(x)
    
    # Simulate gradients
    grad_output = output - target
    grad_A = x.T @ (grad_output @ model.B.T)
    grad_B = model.A.T @ (x.T @ grad_output)
    
    # Update weights
    model.update({'A': grad_A, 'B': grad_B})
    
    # Periodic pruning
    if epoch % 2 == 0:
        retention_ratio = model.prune_ranks()
        print(f"Epoch {epoch}, Retained ranks: {retention_ratio:.2%}")
        
        # Calculate sparsity
        sparsity_A = np.mean(model.A_mask == 0)
        sparsity_B = np.mean(model.B_mask == 0)
        print(f"Sparsity - A: {sparsity_A:.2%}, B: {sparsity_B:.2%}")

# Final compression statistics
total_params = model.input_dim * model.output_dim
lora_params = np.sum(model.A_mask) + np.sum(model.B_mask)
compression_ratio = lora_params / total_params
print(f"\nFinal compression ratio: {compression_ratio:.2%}")
print(f"Effective rank: {np.sum(model.importance_scores > 0)}")
```

Slide 14: Additional Resources

*   "LoRA: Low-Rank Adaptation of Large Language Models"
    *   [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
*   "QLoRA: Efficient Finetuning of Quantized LLMs"
    *   [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
*   "DyLoRA: Dynamic LoRA for Efficient Large Language Model Fine-tuning"
    *   [https://arxiv.org/abs/2309.03905](https://arxiv.org/abs/2309.03905)
*   "DoRA: Weight-Decomposed Low-Rank Adaptation"
    *   [https://arxiv.org/abs/2402.09353](https://arxiv.org/abs/2402.09353)

Additional reading recommendations:

*   Google Scholar search terms: "LoRA variants", "efficient fine-tuning LLMs"
*   Hugging Face documentation on LoRA implementations
*   Papers With Code: [https://paperswithcode.com/method/lora](https://paperswithcode.com/method/lora)

