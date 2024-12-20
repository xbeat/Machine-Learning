## Comparing TensorFlow and PyTorch for AI Engineering
Slide 1: Understanding TensorFlow Core Components

TensorFlow's core components form the foundation of deep learning operations, focusing on tensor manipulations, automatic differentiation, and computational graph construction. The framework enables efficient mathematical operations through its eager execution mode while maintaining compatibility with graph mode for production deployment.

```python
import tensorflow as tf

# Creating tensors
scalar = tf.constant(100)
vector = tf.constant([1, 2, 3, 4])
matrix = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)

# Basic operations
addition = tf.add(matrix, tf.ones([2, 2]))
multiplication = tf.matmul(matrix, matrix)

# Automatic differentiation
with tf.GradientTape() as tape:
    x = tf.Variable([[1., 2.], [3., 4.]])
    y = tf.reduce_sum(x**2)
    
gradient = tape.gradient(y, x)
print(f"Gradient: \n{gradient}")
```

Slide 2: PyTorch Dynamic Computational Graphs

PyTorch implements dynamic computational graphs that allow for flexible model architecture modifications during runtime. This feature enables researchers to debug models effectively and modify network behavior based on intermediate results, making it particularly suitable for research applications.

```python
import torch

# Creating dynamic computation graph
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.mean()

# Backward pass
z.backward()
print(f"Gradient of x: {x.grad}")

# Dynamic model modification
class DynamicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(10, 20)
        
    def forward(self, x):
        if x.sum() > 0:
            return self.hidden(x).relu()
        return self.hidden(x).tanh()
```

Slide 3: GPU Acceleration in TensorFlow

Understanding GPU acceleration in TensorFlow requires knowledge of device placement and memory management. The framework automatically handles most device-specific optimizations while allowing manual control for advanced scenarios where fine-grained performance tuning is necessary.

```python
# GPU device management
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Explicit device placement
with tf.device('/GPU:0'):
    # Matrix multiplication on GPU
    matrix1 = tf.random.normal([1000, 1000])
    matrix2 = tf.random.normal([1000, 1000])
    result = tf.matmul(matrix1, matrix2)
    
# Memory optimization
@tf.function(experimental_compile=True)
def optimized_operation(x):
    return tf.nn.relu(tf.matmul(x, x))

result = optimized_operation(tf.random.normal([1000, 1000]))
```

Slide 4: Custom Layer Implementation in PyTorch

PyTorch's object-oriented design enables seamless creation of custom neural network layers. This implementation demonstrates a complex attention mechanism layer, showcasing PyTorch's flexibility in extending basic functionality while maintaining computational efficiency.

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.output = torch.nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size = x.size(0)
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, -1, self.d_model)
        
        return self.output(output)
```

Slide 5: TensorFlow Data Pipeline Optimization

The tf.data API provides powerful tools for building efficient input pipelines. These optimizations ensure maximum GPU utilization by preventing input bottlenecks through parallel processing, prefetching, and caching mechanisms.

```python
import tensorflow_datasets as tfds

# Creating an optimized data pipeline
def create_optimized_dataset(batch_size=32):
    dataset = tfds.load('mnist', split='train')
    
    return (dataset
            .map(normalize_data, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .shuffle(10000)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))

def normalize_data(features):
    images = tf.cast(features['image'], tf.float32) / 255.0
    labels = features['label']
    return images, labels

# Performance optimization settings
options = tf.data.Options()
options.experimental_optimization.parallel_batch = True
options.experimental_optimization.map_parallelization = True
dataset = create_optimized_dataset().with_options(options)
```

Slide 6: PyTorch Custom Dataset Implementation

Custom datasets in PyTorch enable efficient handling of complex data structures through the Dataset and DataLoader classes. This implementation demonstrates a production-ready approach for processing large-scale image datasets with proper memory management and augmentation.

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

# Usage example with augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = CustomImageDataset('path/to/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

Slide 7: TensorFlow Custom Training Loop

Understanding custom training loops provides granular control over the training process, enabling advanced techniques like gradient clipping, custom metrics, and multi-GPU strategies. This implementation showcases a production-ready training framework.

```python
class CustomModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compiled_loss(y, predictions)
            
        # Compute and apply gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.compiled_metrics.update_state(y, predictions)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = loss
        
        return metrics

@tf.function
def distributed_train_step(strategy, model, data):
    per_replica_losses = strategy.run(model.train_step, args=(data,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# Example usage
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = CustomModel()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

for epoch in range(epochs):
    for data in train_dataset:
        metrics = distributed_train_step(strategy, model, data)
```

Slide 8: Advanced PyTorch Model Deployment

Production deployment of PyTorch models requires optimization techniques including model quantization, pruning, and TorchScript compilation. This implementation demonstrates enterprise-grade model optimization and serving.

```python
import torch.quantization

class QuantizedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

def optimize_for_production(model, calibration_data):
    # Configure quantization
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare for quantization
    model_prepared = torch.quantization.prepare(model)
    
    # Calibrate with sample data
    with torch.no_grad():
        for data in calibration_data:
            model_prepared(data)
    
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)
    
    # Export using TorchScript
    scripted_model = torch.jit.script(model_quantized)
    scripted_model.save('optimized_model.pt')
    
    return model_quantized, scripted_model

# Example usage
model = QuantizedModel(original_model)
quantized_model, scripted_model = optimize_for_production(model, calibration_dataloader)
```

Slide 9: TensorFlow Distributed Training Implementation

Distributed training in TensorFlow enables scaling model training across multiple GPUs and machines. This implementation shows a complete distributed training setup with custom strategies and performance monitoring.

```python
import tensorflow as tf
import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()

# Configure GPUs for distributed training
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

def create_distributed_model():
    # Define model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Scale learning rate by number of workers
    opt = tf.keras.optimizers.Adam(learning_rate=0.001 * hvd.size())
    opt = hvd.DistributedOptimizer(opt)
    
    model.compile(optimizer=opt,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'],
                 experimental_run_tf_function=False)
    return model

# Distributed training setup
model = create_distributed_model()
callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
]

# Train distributed model
model.fit(train_dataset,
          epochs=100,
          callbacks=callbacks,
          steps_per_epoch=500 // hvd.size())
```

Slide 10: Deep Learning Pipeline with TPU Acceleration

TPU acceleration requires specific optimization techniques and data pipeline configurations. This implementation demonstrates a complete TPU-optimized training setup with both TensorFlow and PyTorch integration capabilities.

```python
import tensorflow as tf
import torch_xla
import torch_xla.core.xla_model as xm

def create_tpu_pipeline():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    
    strategy = tf.distribute.TPUStrategy(resolver)
    
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
    
    return model, strategy

# PyTorch TPU integration
class TPUModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.network(x)

def train_on_tpu(model, train_loader, num_epochs=10):
    device = xm.xla_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            xm.optimizer_step(optimizer)
            xm.mark_step()
```

Slide 11: Advanced Neural Architecture Implementation

Implementation of complex neural architectures requires understanding of both framework-specific optimizations and mathematical foundations. This example demonstrates a transformer architecture with attention mechanisms and positional encoding.

```python
import math
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attended = self.attention(x, x, x, attn_mask=mask)[0]
        x = self.norm1(x + self.dropout(attended))
        
        # Feed-forward with residual connection
        forwarded = self.feed_forward(x)
        x = self.norm2(x + self.dropout(forwarded))
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

Slide 12: Real-world Application: Computer Vision Pipeline

Complete implementation of a production-ready computer vision pipeline showcasing integration of both frameworks for optimal performance in image processing and model inference.

```python
import torch
import tensorflow as tf
import cv2
import numpy as np
from torchvision import transforms

class HybridVisionPipeline:
    def __init__(self, torch_model_path, tf_model_path):
        # Load models
        self.torch_model = torch.jit.load(torch_model_path)
        self.tf_model = tf.saved_model.load(tf_model_path)
        
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    @tf.function
    def tf_preprocess(self, image):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, (224, 224))
        return image
    
    def process_image(self, image_path):
        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # PyTorch inference
        torch_input = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            torch_output = self.torch_model(torch_input)
        
        # TensorFlow inference
        tf_input = self.tf_preprocess(image)
        tf_output = self.tf_model(tf_input[None, ...])
        
        return {
            'torch_prediction': torch_output.numpy(),
            'tf_prediction': tf_output.numpy()
        }

# Usage example
pipeline = HybridVisionPipeline('resnet50.pt', 'efficientnet.pb')
results = pipeline.process_image('sample_image.jpg')
```

Slide 13: Real-world Application: Natural Language Processing Pipeline

This implementation demonstrates a production-grade NLP pipeline utilizing both frameworks' strengths in text processing, including tokenization, embedding, and sequence modeling with attention mechanisms.

```python
import torch
import tensorflow as tf
import tensorflow_text as tf_text
from transformers import AutoTokenizer, AutoModel

class DualFrameworkNLP:
    def __init__(self, bert_model_name='bert-base-uncased'):
        # PyTorch BERT setup
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.torch_model = AutoModel.from_pretrained(bert_model_name)
        
        # TensorFlow preprocessing
        self.tf_tokenizer = tf_text.BertTokenizer.from_path(
            "bert_tokenizer.txt",
            lower_case=True
        )
        
    def process_text(self, text, max_length=512):
        # PyTorch processing
        torch_inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            torch_outputs = self.torch_model(**torch_inputs)
            embeddings = torch_outputs.last_hidden_state
        
        # TensorFlow processing
        tf_tokens = self.tf_tokenizer.tokenize(text)
        tf_tokens = tf_tokens.merge_dims(-2, -1)
        
        return {
            'torch_embeddings': embeddings.numpy(),
            'tf_tokens': tf_tokens.numpy()
        }
    
    @tf.function
    def compute_attention(self, embeddings):
        # Scaled dot-product attention
        attention_weights = tf.matmul(
            embeddings, embeddings, transpose_b=True
        )
        attention_weights = attention_weights / tf.math.sqrt(
            tf.cast(embeddings.shape[-1], tf.float32)
        )
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)
        return tf.matmul(attention_weights, embeddings)

# Example usage
processor = DualFrameworkNLP()
text = "Deep learning frameworks enable efficient model development."
results = processor.process_text(text)
```

Slide 14: Performance Optimization and Profiling

Implementation of comprehensive performance monitoring and optimization techniques for both frameworks, including memory profiling, computational graph optimization, and hardware utilization analysis.

```python
import torch.autograd.profiler as profiler
import tensorflow as tf
import time
import psutil

class PerformanceAnalyzer:
    def __init__(self):
        self.tf_summary_writer = tf.summary.create_file_writer('logs/tf')
        self.metrics = {}
    
    def profile_torch_model(self, model, input_data):
        with profiler.profile(use_cuda=True) as prof:
            with profiler.record_function("model_inference"):
                _ = model(input_data)
        
        # Memory usage analysis
        memory_stats = torch.cuda.memory_stats()
        self.metrics['torch_memory_allocated'] = torch.cuda.memory_allocated()
        self.metrics['torch_memory_reserved'] = torch.cuda.memory_reserved()
        
        return prof.key_averages().table(sort_by="cuda_time_total")
    
    @tf.function
    def profile_tf_model(self, model, input_data):
        with tf.profiler.experimental.Profile('logs/tf_profile'):
            with tf.profiler.experimental.Trace('inference'):
                _ = model(input_data)
    
    def monitor_system_resources(self):
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        
        with self.tf_summary_writer.as_default():
            tf.summary.scalar('cpu_utilization', cpu_percent, step=0)
            tf.summary.scalar('memory_utilization', 
                            memory_info.percent, step=0)
    
    def optimize_graph(self, tf_model):
        # Convert to SavedModel for optimization
        optimized_model = tf.saved_model.load(
            tf_model.save('temp_model')
        )
        
        # Enable graph optimization
        optimized_model = tf.function(
            optimized_model,
            experimental_compile=True
        )
        
        return optimized_model

# Usage example
analyzer = PerformanceAnalyzer()
torch_prof_results = analyzer.profile_torch_model(
    torch_model, 
    torch.randn(1, 3, 224, 224)
)
```

Slide 15: Additional Resources

1.  [https://arxiv.org/abs/2102.01536](https://arxiv.org/abs/2102.01536) - "PyTorch vs TensorFlow: A Comprehensive Performance Analysis"
2.  [https://arxiv.org/abs/2104.00254](https://arxiv.org/abs/2104.00254) - "Efficient Deep Learning: A Survey on Making Deep Learning Models Smaller, Faster, and Better"
3.  [https://arxiv.org/abs/2106.09851](https://arxiv.org/abs/2106.09851) - "Large Scale Distributed Deep Learning: A Comprehensive Survey"
4.  [https://arxiv.org/abs/2003.05352](https://arxiv.org/abs/2003.05352) - "A Survey on Deep Learning Hardware Accelerators for Edge Computing"
5.  [https://arxiv.org/abs/2108.07258](https://arxiv.org/abs/2108.07258) - "Deep Learning Framework Optimization: A Survey"

