## Exploring Neural Network Fundamentals in Python
Slide 1: Neural Network Fundamentals

In neural networks, data flows through layers of interconnected nodes, each applying weights and biases to transform inputs into outputs. The provided code demonstrates a simple linear regression using a neural network with one dense layer, implementing a straightforward relationship Y = 2X.

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Input and output data representing Y = 2X relationship
X = np.array([0, 1, 2, 3, 4, 5], dtype=float)
Y = np.array([0, 2, 4, 6, 8, 10], dtype=float)

# Building the sequential model
model = Sequential([
    Input(shape=(1,)),  # Input layer with 1 neuron
    Dense(units=1)      # Output layer with 1 neuron
])
```

Slide 2: Loss Functions and Optimization

The neural network learning process requires defining how to measure prediction errors and adjust weights. Mean Squared Error (MSE) calculates the average squared difference between predictions and actual values, while Stochastic Gradient Descent (SGD) iteratively updates weights to minimize this error.

```python
# Mathematical representation of Mean Squared Error
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

# Implementing loss function and optimization
model.compile(
    optimizer='sgd',               # Stochastic Gradient Descent
    loss='mean_squared_error'      # MSE loss function
)

# Training process
history = model.fit(
    X, Y,                         # Training data
    epochs=500,                   # Number of training iterations
    verbose=0                     # Suppress training output
)
```

Slide 3: Forward Propagation Mathematics

Forward propagation involves calculating the network's output by propagating input values through layers. Each neuron computes a weighted sum of inputs plus bias, then applies an activation function to produce its output.

```python
# Mathematical representation of forward propagation
$$z = wx + b$$
$$a = f(z)$$

def forward_propagation(x, w, b):
    # Weighted sum
    z = np.dot(w, x) + b
    
    # Linear activation (no transformation)
    a = z
    
    return a

# Example usage
w = np.array([[2.0]])  # Weight matrix
b = np.array([0.0])    # Bias term
x = np.array([3.0])    # Input value

output = forward_propagation(x, w, b)
print(f"Output: {output}")  # Expected: ~6.0
```

Slide 4: Backpropagation Implementation

Understanding backpropagation reveals how neural networks learn. This fundamental algorithm calculates gradients of the loss function with respect to weights and biases, enabling the network to adjust its parameters and improve predictions.

```python
def backpropagation(x, y, w, b, learning_rate=0.01):
    # Forward pass
    z = np.dot(w, x) + b
    y_pred = z
    
    # Calculate gradients
    dL_dw = x * (y_pred - y)  # Gradient with respect to weights
    dL_db = y_pred - y        # Gradient with respect to bias
    
    # Update parameters
    w_new = w - learning_rate * dL_dw
    b_new = b - learning_rate * dL_db
    
    return w_new, b_new

# Example usage
x, y = 2.0, 4.0  # Training example
w, b = 1.0, 0.0  # Initial parameters

for epoch in range(100):
    w, b = backpropagation(x, y, w, b)

print(f"Final weight: {w}, Final bias: {b}")
```

Slide 5: Gradient Descent Visualization

The optimization process in neural networks can be visualized as descending a loss landscape. The gradient descent algorithm iteratively updates parameters to find the minimum of the loss function, where predictions are most accurate.

```python
import matplotlib.pyplot as plt

def plot_gradient_descent():
    # Generate loss landscape
    w = np.linspace(-2, 4, 100)
    loss = np.array([(2*w_i - 4)**2 for w_i in w])
    
    plt.figure(figsize=(10, 6))
    plt.plot(w, loss)
    plt.xlabel('Weight')
    plt.ylabel('Loss')
    plt.title('Gradient Descent Optimization')
    plt.grid(True)
    return plt

plot_gradient_descent()
```

Slide 6: Implementing Activation Functions

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Common functions include ReLU, sigmoid, and tanh, each serving different purposes in network architecture and affecting gradient flow during training.

```python
def activation_functions():
    # Mathematical representations
    """
    ReLU: $$f(x) = max(0, x)$$
    Sigmoid: $$f(x) = \frac{1}{1 + e^{-x}}$$
    Tanh: $$f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
    """
    
    def relu(x):
        return np.maximum(0, x)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(x):
        return np.tanh(x)
    
    # Test inputs
    x = np.linspace(-5, 5, 100)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(x, relu(x), label='ReLU')
    plt.plot(x, sigmoid(x), label='Sigmoid')
    plt.plot(x, tanh(x), label='Tanh')
    plt.legend()
    plt.grid(True)
    plt.title('Activation Functions')
    return plt
```

Slide 7: Multi-Layer Neural Network Implementation

A deeper neural network with multiple layers can capture more complex relationships in data. This implementation shows how to stack layers and propagate information through the network while maintaining proper dimensional alignment.

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a multi-layer network
def create_deep_network():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(1,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    return model

# Generate synthetic data
X = np.linspace(-10, 10, 1000).reshape(-1, 1)
Y = 0.5 * X**2 + np.random.normal(0, 1, X.shape)

# Train model
model = create_deep_network()
history = model.fit(X, Y, epochs=50, validation_split=0.2, verbose=0)
```

Slide 8: Batch Processing and Mini-batch Gradient Descent

Training neural networks efficiently requires processing data in batches. Mini-batch gradient descent strikes a balance between computational efficiency and update frequency, enabling faster convergence while maintaining stability.

```python
def create_batches(X, Y, batch_size=32):
    # Mathematical representation of mini-batch updates:
    $$\theta_{t+1} = \theta_t - \eta \frac{1}{|B|} \sum_{i \in B} \nabla_\theta L_i(\theta_t)$$
    
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    
    for i in range(0, len(X), batch_size):
        yield (X_shuffled[i:i+batch_size],
               Y_shuffled[i:i+batch_size])

# Example usage
X = np.random.randn(1000, 1)
Y = 2 * X + 1 + np.random.randn(1000, 1) * 0.1

model = Sequential([
    Dense(1, input_shape=(1,))
])
model.compile(optimizer='sgd', loss='mse')

# Manual training with batches
for epoch in range(10):
    for batch_x, batch_y in create_batches(X, Y):
        model.train_on_batch(batch_x, batch_y)
```

Slide 9: Regularization Techniques

Regularization helps prevent overfitting by adding constraints to the network's parameters. L1 and L2 regularization, along with dropout, are common techniques that improve model generalization by controlling parameter magnitudes.

```python
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l1, l2

def create_regularized_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(1,),
              kernel_regularizer=l2(0.01)),  # L2 regularization
        Dropout(0.3),  # Dropout layer
        Dense(32, activation='relu',
              kernel_regularizer=l1(0.01)),  # L1 regularization
        Dropout(0.2),
        Dense(1)
    ])
    
    # Mathematical representation of regularized loss:
    $$L_{total} = L_{original} + \lambda_1 \sum|w| + \lambda_2 \sum w^2$$
    
    model.compile(optimizer='adam', loss='mse')
    return model
```

Slide 10: Real-world Application - Time Series Prediction

Neural networks excel at time series forecasting by learning patterns in sequential data. This implementation demonstrates how to preprocess time series data and create a model for predicting future values based on historical patterns.

```python
def create_time_series_dataset(data, lookback=5):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

# Generate synthetic time series data
np.random.seed(42)
t = np.linspace(0, 100, 1000)
data = np.sin(0.1 * t) + np.random.normal(0, 0.1, len(t))

# Prepare data
X, y = create_time_series_dataset(data)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create and train model
model = Sequential([
    Dense(32, activation='relu', input_shape=(5,)),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
```

Slide 11: Model Evaluation and Metrics

Understanding model performance requires comprehensive evaluation using multiple metrics. This implementation shows how to calculate and visualize various performance metrics for neural network predictions.

```python
def evaluate_model_performance(y_true, y_pred):
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared Score
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 
             'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Prediction vs True Values')
    return plt

# Example usage
predictions = model.predict(X_test)
evaluate_model_performance(y_test, predictions)
```

Slide 12: Early Stopping and Model Checkpointing

Preventing overfitting requires monitoring training progress and stopping when performance degrades. Early stopping and model checkpointing help save the best model version during training.

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_with_early_stopping(model, X_train, y_train, X_val, y_val):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1000,
        callbacks=[early_stopping, checkpoint],
        verbose=0
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    return plt
```

Slide 13: Custom Training Loop Implementation

A custom training loop provides granular control over the training process, allowing detailed monitoring and modification of gradient updates. This implementation demonstrates how to build a training loop from scratch using TensorFlow's GradientTape.

```python
import tensorflow as tf

def custom_training_loop(model, X, y, epochs=100, learning_rate=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_history = []
    
    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = tf.keras.losses.MSE(y_batch, predictions)
        
        # Calculate gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    # Training loop
    for epoch in range(epochs):
        loss = train_step(X, y)
        loss_history.append(float(loss))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return loss_history

# Example usage
X = np.random.randn(1000, 1)
y = 3 * X + 2 + np.random.randn(1000, 1) * 0.1

simple_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

loss_history = custom_training_loop(simple_model, X, y)
```

Slide 14: Transfer Learning and Fine-tuning

Transfer learning leverages pre-trained models to improve performance on new tasks. This implementation demonstrates how to adapt a pre-trained network for a new dataset while preserving learned features.

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D

def create_transfer_learning_model(num_classes):
    # Load pre-trained model
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add new classification layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def fine_tune_model(model, num_layers_to_unfreeze=10):
    # Unfreeze last n layers
    for layer in model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
```

Slide 15: Additional Resources

*   Deep Learning Book by Ian Goodfellow: [https://www.deeplearningbook.org](https://www.deeplearningbook.org)
*   Neural Networks and Deep Learning by Michael Nielsen: [http://neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com)
*   ArXiv Paper - "A Survey of Deep Learning Techniques": [https://arxiv.org/abs/2004.13376](https://arxiv.org/abs/2004.13376)
*   ArXiv Paper - "Deep Learning in Neural Networks: An Overview": [https://arxiv.org/abs/1404.7828](https://arxiv.org/abs/1404.7828)
*   PyTorch Documentation and Tutorials: [https://pytorch.org/tutorials](https://pytorch.org/tutorials)
*   TensorFlow Documentation and Guides: [https://www.tensorflow.org/guide](https://www.tensorflow.org/guide)

