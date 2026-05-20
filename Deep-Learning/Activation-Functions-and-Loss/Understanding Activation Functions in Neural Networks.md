## Understanding Activation Functions in Neural Networks
Slide 1: Understanding Basic Activation Functions

Neural networks require non-linear activation functions to model complex patterns. We'll implement the classical sigmoid and tanh functions from scratch, examining their mathematical properties and behavior across different input ranges.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    # Sigmoid function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def tanh(x):
    # Hyperbolic tangent: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    return np.tanh(x)

# Generate input values
x = np.linspace(-10, 10, 100)

# Plot both functions
plt.figure(figsize=(12, 6))
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.grid(True)
plt.legend()
plt.title('Sigmoid vs Tanh Activation Functions')
plt.show()

# Mathematical formulas (not rendered):
# $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
# $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
```

Slide 2: Activation Function Derivatives

Understanding derivatives is crucial for backpropagation. The gradient flow through activation functions determines how well the network learns. We'll implement derivatives for sigmoid and tanh to visualize their behavior.

```python
def sigmoid_derivative(x):
    # Derivative of sigmoid: f'(x) = f(x)(1 - f(x))
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh_derivative(x):
    # Derivative of tanh: f'(x) = 1 - tanh²(x)
    return 1 - np.power(tanh(x), 2)

x = np.linspace(-10, 10, 100)

plt.figure(figsize=(12, 6))
plt.plot(x, sigmoid_derivative(x), label='Sigmoid Derivative')
plt.plot(x, tanh_derivative(x), label='Tanh Derivative')
plt.grid(True)
plt.legend()
plt.title('Derivatives of Activation Functions')
plt.show()

# Mathematical formulas (not rendered):
# $$\frac{d}{dx}\sigma(x) = \sigma(x)(1 - \sigma(x))$$
# $$\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$$
```

Slide 3: ReLU and Variants Implementation

The Rectified Linear Unit (ReLU) revolutionized deep learning by solving the vanishing gradient problem. We'll implement ReLU and its popular variants including Leaky ReLU and Parametric ReLU.

```python
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def prelu(x, alpha):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

x = np.linspace(-10, 10, 100)
plt.figure(figsize=(12, 6))
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, leaky_relu(x), label='Leaky ReLU')
plt.plot(x, prelu(x, 0.05), label='PReLU')
plt.plot(x, elu(x), label='ELU')
plt.grid(True)
plt.legend()
plt.title('ReLU Family Activation Functions')
plt.show()

# Mathematical formulas (not rendered):
# $$\text{ReLU}(x) = \max(0, x)$$
# $$\text{LeakyReLU}(x) = \max(\alpha x, x)$$
```

Slide 4: Implementing Softmax Activation

Softmax activation converts raw scores into probability distributions, making it essential for classification tasks. We'll implement a numerically stable version of softmax with temperature scaling.

```python
def softmax(x, temperature=1.0):
    # Numerically stable softmax implementation
    x = x / temperature
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Example usage
scores = np.array([[1.0, 2.0, 3.0, 4.0, 1.0],
                   [2.0, 1.0, 0.1, 3.0, 2.0]])

# Compare different temperatures
temperatures = [0.5, 1.0, 2.0]
for temp in temperatures:
    probs = softmax(scores, temperature=temp)
    print(f"\nTemperature {temp}:")
    print(f"Sum of probabilities: {np.sum(probs, axis=1)}")
    print(f"Probabilities:\n{probs}")

# Mathematical formula (not rendered):
# $$\text{softmax}(x_i) = \frac{e^{x_i/T}}{\sum_j e^{x_j/T}}$$
```

Slide 5: Advanced Activation Functions - Swish and GELU

Modern neural architectures employ sophisticated activation functions like Swish and GELU, which combine the benefits of traditional functions while mitigating their drawbacks. These functions have shown superior performance in deep networks.

```python
def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

def gelu(x):
    # Gaussian Error Linear Unit
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

x = np.linspace(-10, 10, 200)
plt.figure(figsize=(12, 6))
plt.plot(x, swish(x), label='Swish')
plt.plot(x, swish(x, beta=2.0), label='Swish (β=2)')
plt.plot(x, gelu(x), label='GELU')
plt.grid(True)
plt.legend()
plt.title('Modern Activation Functions')
plt.show()

# Mathematical formulas (not rendered):
# $$\text{Swish}(x) = x \cdot \sigma(\beta x)$$
# $$\text{GELU}(x) = 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$$
```

Slide 6: Real-world Classification Example

Implementing a neural network with various activation functions for MNIST digit classification. This example demonstrates the impact of activation function choice on model performance.

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import time

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model(activation):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=activation),
        Dense(64, activation=activation),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Test different activations
activations = ['relu', 'tanh', 'sigmoid']
results = {}

for activation in activations:
    print(f"\nTraining with {activation}")
    model = create_model(activation)
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=5, 
                       validation_data=(x_test, y_test),
                       verbose=0)
    training_time = time.time() - start_time
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results[activation] = {
        'accuracy': test_acc,
        'training_time': training_time,
        'history': history.history
    }
    
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
```

Slide 7: Visualizing Activation Patterns

Understanding how different layers activate helps in choosing appropriate activation functions. We'll create a visualization tool to analyze activation patterns across network layers.

```python
def visualize_activations(model, layer_name, input_image):
    # Create a model that will output layer activations
    layer_model = tf.keras.Model(inputs=model.input,
                               outputs=model.get_layer(layer_name).output)
    
    # Get layer activations for single image
    activations = layer_model.predict(input_image[np.newaxis, ...])
    
    # Plot activation patterns
    plt.figure(figsize=(15, 5))
    n_features = min(16, activations.shape[-1])  # Show up to 16 features
    
    for i in range(n_features):
        plt.subplot(2, 8, i+1)
        plt.imshow(activations[0, :, :, i], cmap='viridis')
        plt.axis('off')
        plt.title(f'Filter {i}')
    
    plt.suptitle(f'Activation Patterns in {layer_name}')
    plt.tight_layout()
    plt.show()

# Example usage
sample_image = x_test[0]
visualize_activations(model, 'dense_1', sample_image)
```

Slide 8: Custom Activation Function Implementation

Creating custom activation functions allows for task-specific optimization. We'll implement a custom activation function with learnable parameters using TensorFlow's framework.

```python
class CustomActivation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomActivation, self).__init__(**kwargs)
        self.alpha = tf.Variable(initial_value=0.5,
                               dtype=tf.float32,
                               trainable=True)
        self.beta = tf.Variable(initial_value=1.0,
                              dtype=tf.float32,
                              trainable=True)

    def call(self, inputs):
        # Custom activation: alpha * x * sigmoid(beta * x)
        return self.alpha * inputs * tf.sigmoid(self.beta * inputs)

    def get_config(self):
        config = super(CustomActivation, self).get_config()
        config.update({'alpha': self.alpha.numpy(),
                      'beta': self.beta.numpy()})
        return config

# Create model with custom activation
model_custom = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128),
    CustomActivation(),
    Dense(64),
    CustomActivation(),
    Dense(10, activation='softmax')
])

model_custom.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Train model
history_custom = model_custom.fit(x_train, y_train,
                                epochs=5,
                                validation_data=(x_test, y_test))
```

Slide 9: Gradient Flow Analysis

Understanding how gradients flow through different activation functions is crucial for deep network training. We'll implement tools to visualize and analyze gradient behavior during backpropagation.

```python
def analyze_gradients(activation_fn, x_range=(-5, 5), samples=1000):
    x = np.linspace(x_range[0], x_range[1], samples)
    
    # Convert to TensorFlow tensors
    x_tf = tf.constant(x, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        x_tf = tf.Variable(x_tf)
        y = activation_fn(x_tf)
        
    gradients = tape.gradient(y, x_tf)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, activation_fn(x), label='Function')
    plt.grid(True)
    plt.title('Activation Function')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x, gradients, label='Gradient', color='red')
    plt.grid(True)
    plt.title('Gradient Flow')
    plt.legend()
    
    return np.mean(np.abs(gradients)), np.std(gradients)

# Analyze different activation functions
activations = {
    'ReLU': tf.nn.relu,
    'Sigmoid': tf.nn.sigmoid,
    'Tanh': tf.nn.tanh,
    'ELU': tf.nn.elu
}

for name, fn in activations.items():
    mean_grad, std_grad = analyze_gradients(fn)
    print(f"\n{name} Statistics:")
    print(f"Mean Gradient Magnitude: {mean_grad:.4f}")
    print(f"Gradient Standard Deviation: {std_grad:.4f}")
```

Slide 10: Vanishing Gradient Analysis

A practical implementation to detect and visualize the vanishing gradient problem across different activation functions in deep networks.

```python
def create_deep_network(activation, depth=20):
    model = Sequential()
    model.add(Dense(64, input_shape=(100,)))
    
    for _ in range(depth):
        model.add(Dense(64, activation=activation))
    
    model.add(Dense(1, activation='sigmoid'))
    return model

def analyze_gradient_flow(model, input_data):
    with tf.GradientTape() as tape:
        inputs = tf.Variable(input_data)
        outputs = model(inputs)
        loss = tf.reduce_mean(outputs)
    
    # Get gradients for all layers
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Calculate gradient norms per layer
    gradient_norms = [tf.norm(grad).numpy() for grad in gradients]
    
    plt.figure(figsize=(12, 6))
    plt.plot(gradient_norms, 'bo-')
    plt.yscale('log')
    plt.xlabel('Layer Index')
    plt.ylabel('Gradient Norm (log scale)')
    plt.title(f'Gradient Flow Analysis')
    plt.grid(True)
    
    return gradient_norms

# Test different activations
activations = ['relu', 'tanh', 'sigmoid']
input_data = np.random.normal(size=(32, 100))

for activation in activations:
    model = create_deep_network(activation)
    gradient_norms = analyze_gradient_flow(model, input_data)
    
    print(f"\n{activation.upper()} Statistics:")
    print(f"Max/Min Gradient Ratio: {max(gradient_norms)/min(gradient_norms):.2e}")
```

Slide 11: Activation Function Benchmark Suite

A comprehensive benchmark system to evaluate activation functions across different architectures and datasets, measuring both performance and computational efficiency.

```python
class ActivationBenchmark:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.results = {}
        
    def create_model(self, activation):
        return Sequential([
            Dense(256, activation=activation, input_shape=self.input_shape),
            Dense(128, activation=activation),
            Dense(64, activation=activation),
            Dense(self.num_classes, activation='softmax')
        ])
    
    def benchmark_activation(self, activation, x_train, y_train, x_test, y_test):
        model = self.create_model(activation)
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Training time measurement
        start_time = time.time()
        history = model.fit(x_train, y_train,
                          validation_data=(x_test, y_test),
                          epochs=10, batch_size=32, verbose=0)
        training_time = time.time() - start_time
        
        # Inference time measurement
        start_time = time.time()
        predictions = model.predict(x_test, verbose=0)
        inference_time = (time.time() - start_time) / len(x_test)
        
        return {
            'training_time': training_time,
            'inference_time': inference_time,
            'final_accuracy': history.history['val_accuracy'][-1],
            'convergence_rate': np.mean(np.diff(history.history['accuracy']))
        }

# Usage example
benchmark = ActivationBenchmark((784,), 10)
activations = ['relu', 'elu', 'selu', 'tanh']

for activation in activations:
    results = benchmark.benchmark_activation(
        activation, x_train, y_train, x_test, y_test)
    print(f"\n{activation.upper()} Benchmark Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
```

Slide 12: Advanced Activation Function Training Dynamics

Implementing a system to analyze how different activation functions affect model convergence and learning dynamics during training, with real-time visualization capabilities.

```python
class ActivationTrainingAnalyzer:
    def __init__(self, model, activation_name):
        self.model = model
        self.activation_name = activation_name
        self.activation_outputs = []
        self.gradient_history = []
        
    def activation_callback(self, epoch, logs):
        # Get activation values for a specific layer
        activation_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('dense_1').output
        )
        activations = activation_model.predict(x_test[:100], verbose=0)
        self.activation_outputs.append(activations)
        
        # Calculate gradients
        with tf.GradientTape() as tape:
            outputs = self.model(x_test[:100])
            loss = tf.keras.losses.categorical_crossentropy(
                y_test[:100], outputs
            )
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.gradient_history.append([tf.norm(g).numpy() for g in grads])

    def analyze_training_dynamics(self):
        plt.figure(figsize=(15, 5))
        
        # Plot activation distribution changes
        plt.subplot(1, 3, 1)
        activation_means = [np.mean(act) for act in self.activation_outputs]
        activation_stds = [np.std(act) for act in self.activation_outputs]
        epochs = range(len(activation_means))
        plt.errorbar(epochs, activation_means, yerr=activation_stds, 
                    label='Activation Stats')
        plt.title(f'{self.activation_name} Distribution')
        plt.xlabel('Epoch')
        plt.ylabel('Activation Value')
        
        # Plot gradient norms
        plt.subplot(1, 3, 2)
        gradient_norms = np.array(self.gradient_history)
        plt.plot(gradient_norms)
        plt.title('Gradient Norms')
        plt.xlabel('Epoch')
        plt.ylabel('Norm')
        
        # Plot activation sparsity
        plt.subplot(1, 3, 3)
        sparsity = [np.mean(act == 0) for act in self.activation_outputs]
        plt.plot(sparsity)
        plt.title('Activation Sparsity')
        plt.xlabel('Epoch')
        plt.ylabel('Sparsity Ratio')
        
        plt.tight_layout()
        return {
            'final_sparsity': sparsity[-1],
            'mean_gradient_norm': np.mean(gradient_norms),
            'activation_stability': np.std(activation_means)
        }
```

Slide 13: Practical Implementation in Computer Vision

A real-world example implementing different activation functions in a convolutional neural network for image classification, with performance comparisons.

```python
class ActivationCNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def build_model(self, activation):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation=activation,
                                 input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation=activation),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation=activation),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation=activation),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def train_and_evaluate(self, x_train, y_train, x_test, y_test,
                          activation='relu', epochs=10):
        model = self.build_model(activation)
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Add learning rate scheduling
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3)
        
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=[lr_scheduler],
            verbose=0
        )
        
        # Evaluate final performance
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        return {
            'test_accuracy': test_accuracy,
            'training_history': history.history,
            'final_loss': test_loss
        }

# Example usage with CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

cnn_analyzer = ActivationCNN((32, 32, 3), 10)
activations = ['relu', 'elu', 'selu', 'swish']

results = {}
for activation in activations:
    print(f"\nTraining with {activation}")
    results[activation] = cnn_analyzer.train_and_evaluate(
        x_train, y_train, x_test, y_test, activation)
```

Slide 14: Additional Resources

*   "Activation Functions in Deep Learning: A Comprehensive Survey" [https://arxiv.org/abs/2011.08698](https://arxiv.org/abs/2011.08698)
*   "On the Importance of Single Directions for Generalization" [https://arxiv.org/abs/1803.06959](https://arxiv.org/abs/1803.06959)
*   "Searching for Activation Functions" [https://arxiv.org/abs/1710.05941](https://arxiv.org/abs/1710.05941)
*   "Understanding and Improving Layer Normalization" [https://arxiv.org/abs/1911.07013](https://arxiv.org/abs/1911.07013)
*   "Mish: A Self Regularized Non-Monotonic Activation Function" [https://arxiv.org/abs/1908.08681](https://arxiv.org/abs/1908.08681)

