## Mini-Batching with In-Memory Datasets in Python
Slide 1: Introduction to Mini-batching with In-Memory Datasets

Mini-batching is a technique used in machine learning to process subsets of data during training. It allows for efficient computation and improved generalization. This presentation will cover the implementation of mini-batching using Python for in-memory datasets.

```python
import numpy as np

# Create a sample dataset
X = np.random.randn(1000, 10)  # 1000 samples, 10 features
y = np.random.randint(0, 2, 1000)  # Binary labels

# Define batch size
batch_size = 32

# Calculate number of batches
num_batches = len(X) // batch_size

print(f"Total samples: {len(X)}")
print(f"Batch size: {batch_size}")
print(f"Number of batches: {num_batches}")
```

Slide 2: Understanding Mini-batches

Mini-batches are small subsets of the training data used to update model parameters. They strike a balance between the computational efficiency of stochastic gradient descent and the stability of batch gradient descent. Mini-batches reduce memory usage and allow for more frequent model updates.

```python
import numpy as np

def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y.reshape(len(y), 1)))
    np.random.shuffle(data)
    n_minibatches = len(data) // batch_size
    
    for i in range(n_minibatches):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, y_mini))
    
    return mini_batches

# Example usage
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)
mini_batches = create_mini_batches(X, y, batch_size=32)

print(f"Number of mini-batches: {len(mini_batches)}")
print(f"Shape of first mini-batch X: {mini_batches[0][0].shape}")
print(f"Shape of first mini-batch y: {mini_batches[0][1].shape}")
```

Slide 3: Implementing a Simple Dataset Class

Creating a custom Dataset class helps in organizing and managing the data. This class will store the features and labels, and provide methods to access them.

```python
import numpy as np

class SimpleDataset:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create a sample dataset
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

dataset = SimpleDataset(X, y)

print(f"Dataset length: {len(dataset)}")
print(f"First item: {dataset[0]}")
```

Slide 4: Implementing a DataLoader

A DataLoader is responsible for creating mini-batches from the dataset. It shuffles the data and yields batches of the specified size.

```python
import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start in range(0, len(self.dataset), self.batch_size):
            end = min(start + self.batch_size, len(self.dataset))
            batch_indices = self.indices[start:end]
            yield self._get_batch(batch_indices)
    
    def _get_batch(self, indices):
        features = np.array([self.dataset[i][0] for i in indices])
        labels = np.array([self.dataset[i][1] for i in indices])
        return features, labels

# Using the SimpleDataset from the previous slide
dataloader = DataLoader(dataset, batch_size=32)

for batch_features, batch_labels in dataloader:
    print(f"Batch features shape: {batch_features.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    break
```

Slide 5: Training Loop with Mini-batches

Incorporating mini-batches into the training loop allows for efficient updates of model parameters. This example demonstrates a simple training loop using mini-batches.

```python
import numpy as np

def train_model(model, dataloader, num_epochs, learning_rate):
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_features, batch_labels in dataloader:
            # Forward pass
            predictions = model.forward(batch_features)
            
            # Compute loss
            loss = compute_loss(predictions, batch_labels)
            total_loss += loss
            
            # Backward pass
            gradients = compute_gradients(loss, model.parameters)
            
            # Update parameters
            for param, grad in zip(model.parameters, gradients):
                param -= learning_rate * grad
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Assuming we have a model, dataset, and dataloader defined
num_epochs = 10
learning_rate = 0.01

train_model(model, dataloader, num_epochs, learning_rate)
```

Slide 6: Handling Uneven Batch Sizes

When the dataset size is not divisible by the batch size, the last batch may have a different size. It's important to handle this case in your code.

```python
import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
    
    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, len(self.dataset), self.batch_size):
            end = min(start + self.batch_size, len(self.dataset))
            if self.drop_last and end - start < self.batch_size:
                break
            batch_indices = indices[start:end]
            yield self._get_batch(batch_indices)
    
    def _get_batch(self, indices):
        features = np.array([self.dataset[i][0] for i in indices])
        labels = np.array([self.dataset[i][1] for i in indices])
        return features, labels

# Example usage
dataset_size = 1005
batch_size = 32
dataset = SimpleDataset(np.random.randn(dataset_size, 10), np.random.randint(0, 2, dataset_size))

dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False)

for i, (batch_features, batch_labels) in enumerate(dataloader):
    print(f"Batch {i + 1}: features shape {batch_features.shape}, labels shape {batch_labels.shape}")

print(f"Total number of batches: {i + 1}")
```

Slide 7: Memory Management for Large Datasets

When working with large datasets that don't fit entirely in memory, we can use generators to load data in chunks. This approach allows processing of large datasets without exhausting system memory.

```python
import numpy as np

def data_generator(num_samples, batch_size, feature_dim):
    for _ in range(0, num_samples, batch_size):
        batch_size = min(batch_size, num_samples - _)
        yield (np.random.randn(batch_size, feature_dim), 
               np.random.randint(0, 2, batch_size))

# Simulating a large dataset
num_samples = 1_000_000
feature_dim = 100
batch_size = 128

data_gen = data_generator(num_samples, batch_size, feature_dim)

# Process batches
for i, (batch_features, batch_labels) in enumerate(data_gen):
    # Simulate processing
    if i % 1000 == 0:
        print(f"Processed batch {i}: features shape {batch_features.shape}, labels shape {batch_labels.shape}")

print("Finished processing all batches")
```

Slide 8: Implementing Shuffle Buffer

A shuffle buffer allows for efficient shuffling of data without loading the entire dataset into memory. This is particularly useful for large datasets.

```python
import numpy as np
from collections import deque

class ShuffleBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
    
    def add(self, item):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(item)
        else:
            if np.random.random() < self.buffer_size / (self.buffer_size + 1):
                idx = np.random.randint(0, self.buffer_size)
                self.buffer[idx] = item
    
    def get(self):
        if self.buffer:
            idx = np.random.randint(0, len(self.buffer))
            return self.buffer[idx]
        return None

# Example usage
buffer = ShuffleBuffer(1000)

# Simulating data stream
for i in range(10000):
    item = np.random.randn(10)  # Example data point
    buffer.add(item)

    if i % 1000 == 0:
        sample = buffer.get()
        print(f"Sample at step {i}: {sample}")
```

Slide 9: Real-Life Example: Image Classification

Mini-batching is commonly used in image classification tasks. Here's an example using a simple convolutional neural network with mini-batches for classifying images.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Simulating an image dataset
num_samples = 10000
img_height, img_width = 28, 28
num_classes = 10

# Generate random image data and labels
X = np.random.rand(num_samples, img_height, img_width, 1)
y = np.random.randint(0, num_classes, num_samples)

# Create the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using mini-batches
batch_size = 32
history = model.fit(X, y, batch_size=batch_size, epochs=5, validation_split=0.2)

print("Training completed")
```

Slide 10: Real-Life Example: Text Classification

Another common application of mini-batching is in natural language processing tasks like text classification. Here's an example using a simple recurrent neural network with mini-batches for sentiment analysis.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Simulating a text dataset
num_samples = 10000
max_words = 10000
max_length = 100

# Generate random text data and labels
texts = [''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz '), 50)) for _ in range(num_samples)]
labels = np.random.randint(0, 2, num_samples)  # Binary classification

# Tokenize the texts
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_length)

# Create the model
model = models.Sequential([
    layers.Embedding(max_words, 32, input_length=max_length),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using mini-batches
batch_size = 64
history = model.fit(X, labels, batch_size=batch_size, epochs=5, validation_split=0.2)

print("Training completed")
```

Slide 11: Optimizing Mini-batch Processing

To maximize efficiency, we can use techniques like prefetching and parallel data loading. Here's an example using TensorFlow's tf.data API for optimized mini-batch processing.

```python
import tensorflow as tf
import numpy as np

# Simulating a large dataset
num_samples = 1_000_000
feature_dim = 100

def data_generator():
    while True:
        yield (np.random.randn(feature_dim), np.random.randint(0, 2))

# Create a tf.data.Dataset
dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_types=(tf.float32, tf.int32),
    output_shapes=((feature_dim,), ()))

# Apply optimizations
batch_size = 128
dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Simple model for demonstration
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(feature_dim,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(dataset, steps_per_epoch=num_samples // batch_size, epochs=5)

print("Training completed")
```

Slide 12: Monitoring and Debugging Mini-batch Processing

It's crucial to monitor the mini-batch processing to ensure everything is working correctly. Here's an example of how to add monitoring and debugging capabilities to your mini-batch training loop.

```python
import numpy as np
import time

class DebugDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
    
    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i:i+self.batch_size]
            yield batch

def train_with_monitoring(model, dataloader, num_epochs):
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            batch_start_time = time.time()
            
            # Simulate forward pass and loss computation
            loss = np.random.rand()
            total_loss += loss
            num_batches += 1
            
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            
            print(f"Epoch {epoch+1}, Batch {num_batches}: Loss = {loss:.4f}, Duration = {batch_duration:.4f}s")
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_loss = total_loss / num_batches
        
        print(f"Epoch {epoch+1} completed: Avg Loss = {avg_loss:.4f}, Duration = {epoch_duration:.2f}s")

# Simulate a dataset and model
dataset = np.random.randn(1000, 10)
model = None  # Placeholder for a real model
batch_size = 32

dataloader = DebugDataLoader(dataset, batch_size)
train_with_monitoring(model, dataloader, num_epochs=2)
```

Slide 13: Handling Imbalanced Datasets with Mini-batches

When dealing with imbalanced datasets, it's important to ensure that mini-batches maintain a representative distribution of classes. Here's an approach to create balanced mini-batches.

```python
import numpy as np
from collections import defaultdict

def create_balanced_batches(X, y, batch_size):
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)
    
    num_classes = len(class_indices)
    samples_per_class = batch_size // num_classes
    
    batches = []
    while all(len(indices) >= samples_per_class for indices in class_indices.values()):
        batch_indices = []
        for class_label, indices in class_indices.items():
            batch_indices.extend(np.random.choice(indices, samples_per_class, replace=False))
            class_indices[class_label] = [idx for idx in indices if idx not in batch_indices]
        
        np.random.shuffle(batch_indices)
        batches.append((X[batch_indices], y[batch_indices]))
    
    return batches

# Example usage
X = np.random.randn(1000, 10)
y = np.random.choice([0, 1, 2], 1000, p=[0.6, 0.3, 0.1])  # Imbalanced dataset

balanced_batches = create_balanced_batches(X, y, batch_size=32)

print(f"Number of balanced batches: {len(balanced_batches)}")
print(f"First batch shape: X = {balanced_batches[0][0].shape}, y = {balanced_batches[0][1].shape}")
print(f"Class distribution in first batch: {np.bincount(balanced_batches[0][1])}")
```

Slide 14: Mini-batch Gradient Descent Variants

There are several variants of mini-batch gradient descent that can improve convergence and generalization. Here's an implementation of mini-batch gradient descent with momentum.

```python
import numpy as np

def sgd_momentum(params, grads, velocity, learning_rate, momentum):
    for param, grad, vel in zip(params, grads, velocity):
        vel = momentum * vel - learning_rate * grad
        param += vel
    return params, velocity

def train_with_momentum(X, y, model, loss_fn, learning_rate, momentum, batch_size, num_epochs):
    params = model.get_params()
    velocity = [np.zeros_like(param) for param in params]
    
    for epoch in range(num_epochs):
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            # Forward pass
            predictions = model.forward(batch_X)
            loss = loss_fn(predictions, batch_y)
            
            # Backward pass
            grads = model.backward(loss)
            
            # Update parameters
            params, velocity = sgd_momentum(params, grads, velocity, learning_rate, momentum)
            model.set_params(params)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

# Example usage (assuming we have a model, loss function, and data)
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)
model = None  # Placeholder for a real model
loss_fn = lambda y_pred, y_true: np.mean((y_pred - y_true)**2)  # Simple MSE loss

train_with_momentum(X, y, model, loss_fn, learning_rate=0.01, momentum=0.9, batch_size=32, num_epochs=10)
```

Slide 15: Additional Resources

For further exploration of mini-batching techniques and related topics, consider the following resources:

1. "Optimization Methods for Large-Scale Machine Learning" by Léon Bottou, Frank E. Curtis, and Jorge Nocedal (2018) ArXiv: [https://arxiv.org/abs/1606.04838](https://arxiv.org/abs/1606.04838)
2. "Efficient BackProp" by Yann LeCun, Léon Bottou, Genevieve B. Orr, and Klaus-Robert Müller (2012) Available in the book "Neural Networks: Tricks of the Trade"
3. "A Theoretical Analysis of Learning with Mini-batch Gradients" by Yossi Arjevani and Ohad Shamir (2017) ArXiv: [https://arxiv.org/abs/1705.07193](https://arxiv.org/abs/1705.07193)

These resources provide in-depth theoretical and practical insights into mini-batching and its applications in machine learning.

