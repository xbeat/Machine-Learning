## Boosted Hierarchical Adaptive Activation Network (BHAAN) in Python
Slide 1: Introduction to Boosted Hierarchical Adaptive Activation Network (BHAAN)

Boosted Hierarchical Adaptive Activation Network (BHAAN) is an innovative approach to neural network architecture that combines the concepts of boosting, hierarchical structures, and adaptive activation functions. This advanced model aims to enhance the performance and flexibility of deep learning systems by leveraging these key components.

```python
import tensorflow as tf

class BHAAN(tf.keras.Model):
    def __init__(self, num_layers, units_per_layer):
        super(BHAAN, self).__init__()
        self.layers_list = [tf.keras.layers.Dense(units_per_layer) for _ in range(num_layers)]
        self.adaptive_activation = AdaptiveActivation()

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
            x = self.adaptive_activation(x)
        return x

class AdaptiveActivation(tf.keras.layers.Layer):
    def __init__(self):
        super(AdaptiveActivation, self).__init__()
        self.alpha = tf.Variable(initial_value=0.1, trainable=True)

    def call(self, inputs):
        return tf.nn.leaky_relu(inputs, alpha=self.alpha)

# Create a BHAAN model
model = BHAAN(num_layers=5, units_per_layer=64)
```

Slide 2: Hierarchical Structure in BHAAN

The hierarchical structure in BHAAN allows the network to learn and represent complex patterns at different levels of abstraction. This architecture organizes neurons into layers, with each layer capturing increasingly abstract features of the input data.

```python
import tensorflow as tf

class HierarchicalLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_sublayers):
        super(HierarchicalLayer, self).__init__()
        self.sublayers = [tf.keras.layers.Dense(units) for _ in range(num_sublayers)]

    def call(self, inputs):
        x = inputs
        for sublayer in self.sublayers:
            x = sublayer(x)
        return x

# Create a hierarchical model
model = tf.keras.Sequential([
    HierarchicalLayer(units=64, num_sublayers=3),
    HierarchicalLayer(units=32, num_sublayers=2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

Slide 3: Adaptive Activation Functions

Adaptive activation functions in BHAAN allow the network to adjust its non-linearity during training, optimizing performance for different types of data and tasks. This flexibility enables the model to capture complex relationships more effectively.

```python
import tensorflow as tf

class AdaptiveReLU(tf.keras.layers.Layer):
    def __init__(self):
        super(AdaptiveReLU, self).__init__()
        self.alpha = tf.Variable(initial_value=0.1, trainable=True)

    def call(self, inputs):
        return tf.maximum(self.alpha * inputs, inputs)

# Create a model with adaptive ReLU
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    AdaptiveReLU(),
    tf.keras.layers.Dense(32),
    AdaptiveReLU(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

Slide 4: Boosting in BHAAN

Boosting in BHAAN involves training multiple weak learners sequentially, with each new learner focusing on the mistakes of the previous ones. This technique improves the overall performance of the model by combining the strengths of multiple learners.

```python
import tensorflow as tf

class BoostedBHAAN:
    def __init__(self, num_models, num_layers, units_per_layer):
        self.models = [BHAAN(num_layers, units_per_layer) for _ in range(num_models)]
        self.weights = tf.Variable(initial_value=tf.ones(num_models) / num_models, trainable=True)

    def call(self, inputs):
        predictions = [model(inputs) for model in self.models]
        weighted_sum = tf.reduce_sum([w * p for w, p in zip(self.weights, predictions)], axis=0)
        return weighted_sum

# Create a boosted BHAAN model
boosted_model = BoostedBHAAN(num_models=3, num_layers=5, units_per_layer=64)

# Example usage
inputs = tf.random.normal((32, 100))
outputs = boosted_model.call(inputs)
print(f"Output shape: {outputs.shape}")
```

Slide 5: Training Process for BHAAN

The training process for BHAAN involves optimizing the network parameters, including the weights of the hierarchical layers, the adaptive activation function parameters, and the boosting weights. This process typically uses gradient descent-based optimization algorithms.

```python
import tensorflow as tf

# Assuming we have a BoostedBHAAN model and training data
boosted_model = BoostedBHAAN(num_models=3, num_layers=5, units_per_layer=64)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = boosted_model.call(inputs)
        loss = tf.keras.losses.categorical_crossentropy(targets, predictions)
    
    gradients = tape.gradient(loss, boosted_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, boosted_model.trainable_variables))
    return loss

# Training loop
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    for batch in range(0, len(train_data), batch_size):
        inputs = train_data[batch:batch+batch_size]
        targets = train_labels[batch:batch+batch_size]
        loss = train_step(inputs, targets)
    
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")
```

Slide 6: Feature Extraction in BHAAN

BHAAN's hierarchical structure allows for effective feature extraction at different levels of abstraction. Lower layers capture low-level features, while higher layers combine these to form more complex representations.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class FeatureExtractor(tf.keras.Model):
    def __init__(self, num_layers, units_per_layer):
        super(FeatureExtractor, self).__init__()
        self.layers_list = [tf.keras.layers.Dense(units_per_layer, activation='relu') 
                            for _ in range(num_layers)]

    def call(self, inputs):
        features = []
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
            features.append(x)
        return features

# Create a feature extractor
extractor = FeatureExtractor(num_layers=3, units_per_layer=64)

# Generate sample data
sample_data = np.random.rand(100, 10)

# Extract features
features = extractor(sample_data)

# Visualize features
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, feature in enumerate(features):
    axes[i].imshow(feature.numpy(), cmap='viridis')
    axes[i].set_title(f"Layer {i+1} Features")
    axes[i].axis('off')
plt.tight_layout()
plt.show()
```

Slide 7: Adaptive Learning Rate in BHAAN

BHAAN can incorporate adaptive learning rate techniques to optimize the training process. This allows the model to adjust its learning rate dynamically based on the training progress, leading to faster convergence and better performance.

```python
import tensorflow as tf

class AdaptiveLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr=0.001, patience=3, factor=0.5, min_lr=1e-6):
        super(AdaptiveLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.wait = 0
        self.best_loss = float('inf')

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.initial_lr)

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
                new_lr = max(current_lr * self.factor, self.min_lr)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                print(f"\nEpoch {epoch + 1}: reducing learning rate to {new_lr}")
                self.wait = 0

# Usage
model = BHAAN(num_layers=5, units_per_layer=64)
model.compile(optimizer='adam', loss='mse')
lr_scheduler = AdaptiveLearningRateScheduler()

# Train the model with the adaptive learning rate scheduler
history = model.fit(x_train, y_train, epochs=50, callbacks=[lr_scheduler])

# Plot the learning rate over time
plt.plot(history.history['lr'])
plt.title('Learning Rate over Time')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.show()
```

Slide 8: Regularization Techniques in BHAAN

Regularization is crucial in BHAAN to prevent overfitting and improve generalization. Techniques such as L1/L2 regularization and dropout can be applied to the model's layers.

```python
import tensorflow as tf

class RegularizedBHAAN(tf.keras.Model):
    def __init__(self, num_layers, units_per_layer, l2_reg=0.01, dropout_rate=0.2):
        super(RegularizedBHAAN, self).__init__()
        self.layers_list = [
            tf.keras.layers.Dense(
                units_per_layer,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            ) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
            if training:
                x = self.dropout(x)
        return x

# Create a regularized BHAAN model
model = RegularizedBHAAN(num_layers=5, units_per_layer=64)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

Slide 9: Ensemble Methods in BHAAN

Ensemble methods can be incorporated into BHAAN to further improve its performance. By combining multiple BHAAN models, we can create a more robust and accurate prediction system.

```python
import tensorflow as tf
import numpy as np

class BHAANEnsemble:
    def __init__(self, num_models, num_layers, units_per_layer):
        self.models = [BHAAN(num_layers, units_per_layer) for _ in range(num_models)]

    def compile(self, optimizer, loss):
        for model in self.models:
            model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y, epochs, verbose=1):
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}")
            model.fit(x, y, epochs=epochs, verbose=verbose)

    def predict(self, x):
        predictions = [model.predict(x) for model in self.models]
        return np.mean(predictions, axis=0)

# Create an ensemble of BHAAN models
ensemble = BHAANEnsemble(num_models=5, num_layers=3, units_per_layer=64)

# Compile and train the ensemble
ensemble.compile(optimizer='adam', loss='mse')
ensemble.fit(x_train, y_train, epochs=50)

# Make predictions using the ensemble
y_pred = ensemble.predict(x_test)

# Evaluate the ensemble
mse = np.mean((y_test - y_pred)**2)
print(f"Mean Squared Error: {mse}")
```

Slide 10: Visualization of BHAAN Architecture

Visualizing the BHAAN architecture can help in understanding its structure and how different components interact. We can use Python to create a simple visualization of the network.

```python
import networkx as nx
import matplotlib.pyplot as plt

def visualize_bhaan(num_layers, units_per_layer):
    G = nx.DiGraph()
    pos = {}
    
    # Add input layer
    G.add_node("Input")
    pos["Input"] = (0, 0.5)
    
    # Add hidden layers
    for layer in range(num_layers):
        for unit in range(units_per_layer):
            node_name = f"L{layer+1}U{unit+1}"
            G.add_node(node_name)
            pos[node_name] = (layer + 1, unit / units_per_layer)
            
            if layer == 0:
                G.add_edge("Input", node_name)
            else:
                for prev_unit in range(units_per_layer):
                    G.add_edge(f"L{layer}U{prev_unit+1}", node_name)
    
    # Add output layer
    G.add_node("Output")
    pos["Output"] = (num_layers + 1, 0.5)
    for unit in range(units_per_layer):
        G.add_edge(f"L{num_layers}U{unit+1}", "Output")
    
    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=8, arrows=True)
    plt.title("BHAAN Architecture")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualize a BHAAN with 3 layers and 4 units per layer
visualize_bhaan(num_layers=3, units_per_layer=4)
```

Slide 11: Handling Imbalanced Data with BHAAN

BHAAN can be adapted to handle imbalanced datasets by incorporating techniques such as class weighting or oversampling. This ensures that the model doesn't bias towards the majority class and maintains good performance across all classes.

```python
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight

# Assume we have imbalanced dataset (x_train, y_train)
class_weights = class_weight.compute_class_weight('balanced', 
                                                  classes=np.unique(y_train), 
                                                  y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Create and compile a BHAAN model
model = BHAAN(num_layers=5, units_per_layer=64)
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model with class weights
history = model.fit(x_train, y_train, 
                    epochs=50, 
                    class_weight=class_weight_dict,
                    validation_split=0.2)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy with Class Weighting')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

Slide 12: BHAAN for Multi-task Learning

BHAAN can be extended for multi-task learning, where a single model is trained to perform multiple related tasks simultaneously. This approach can lead to improved generalization and efficiency.

```python
import tensorflow as tf

class MultitaskBHAAN(tf.keras.Model):
    def __init__(self, num_shared_layers, units_per_layer, num_tasks):
        super(MultitaskBHAAN, self).__init__()
        self.shared_layers = [tf.keras.layers.Dense(units_per_layer, activation='relu') 
                              for _ in range(num_shared_layers)]
        self.task_specific_layers = [tf.keras.layers.Dense(1) for _ in range(num_tasks)]

    def call(self, inputs):
        x = inputs
        for layer in self.shared_layers:
            x = layer(x)
        outputs = [task_layer(x) for task_layer in self.task_specific_layers]
        return outputs

# Create a multi-task BHAAN model
model = MultitaskBHAAN(num_shared_layers=3, units_per_layer=64, num_tasks=3)

# Compile the model
model.compile(optimizer='adam',
              loss=['mse', 'mse', 'mse'],
              loss_weights=[1.0, 0.8, 0.5])

# Generate dummy data for demonstration
import numpy as np
x_train = np.random.rand(1000, 10)
y_train = [np.random.rand(1000, 1) for _ in range(3)]

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# Plot training history
plt.plot(history.history['loss'], label='Total Loss')
plt.title('Multi-task Learning Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

Slide 13: BHAAN for Transfer Learning

BHAAN can be effectively used for transfer learning, where knowledge gained from one task is applied to a different but related task. This approach is particularly useful when dealing with limited data in the target domain.

```python
import tensorflow as tf

# Pretrained BHAAN model
pretrained_model = BHAAN(num_layers=5, units_per_layer=64)
pretrained_model.compile(optimizer='adam', loss='mse')
pretrained_model.fit(x_train_source, y_train_source, epochs=50)

# Create a new model for transfer learning
transfer_model = tf.keras.Sequential([
    pretrained_model,
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Freeze the pretrained layers
for layer in pretrained_model.layers:
    layer.trainable = False

# Compile and train the transfer model
transfer_model.compile(optimizer='adam', loss='mse')
history = transfer_model.fit(x_train_target, y_train_target, epochs=20, validation_split=0.2)

# Plot transfer learning performance
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Transfer Learning Performance')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

Slide 14: Real-life Example: Image Classification with BHAAN

In this example, we'll use BHAAN for image classification on a dataset of handwritten digits. This demonstrates how BHAAN can be applied to real-world computer vision tasks.

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

# Create BHAAN model for image classification
model = tf.keras.Sequential([
    BHAAN(num_layers=3, units_per_layer=128),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy on MNIST Dataset')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

Slide 15: Real-life Example: Time Series Forecasting with BHAAN

In this example, we'll use BHAAN for time series forecasting, demonstrating its application in predicting future values based on historical data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic time series data
np.random.seed(42)
time = np.arange(1000)
series = np.sin(0.02 * time) + np.random.randn(1000) * 0.1

# Normalize the data
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1))

# Prepare data for BHAAN
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 10
X, Y = create_dataset(series_scaled, look_back)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Reshape input for BHAAN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Create and train BHAAN model
model = tf.keras.Sequential([
    BHAAN(num_layers=2, units_per_layer=64),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, Y_train, epochs=50, validation_split=0.2, verbose=0)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time[look_back:], series[look_back:], label='Actual')
plt.plot(time[look_back:train_size], train_predict, label='Train Predict')
plt.plot(time[train_size+look_back:], test_predict, label='Test Predict')
plt.title('Time Series Forecasting with BHAAN')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
```

Slide 16: Additional Resources

For more information on advanced neural network architectures and techniques related to BHAAN, consider exploring the following resources:

1. "Attention Is All You Need" by Vaswani et al. (2017) - Introduces the Transformer architecture, which has similarities to hierarchical networks. Available at: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "Deep Residual Learning for Image Recognition" by He et al. (2015) - Discusses ResNet, which shares concepts with hierarchical networks. Available at: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
3. "Densely Connected Convolutional Networks" by Huang et al. (2016) - Presents DenseNet, another hierarchical approach to network design. Available at: [https://arxiv.org/abs/1608.06993](https://arxiv.org/abs/1608.06993)
4. "Neural Architecture Search with Reinforcement Learning" by Zoph and Le (2016) - Explores automated methods for designing neural network architectures. Available at: [https://arxiv.org/abs/1611.01578](https://arxiv.org/abs/1611.01578)

These papers provide valuable insights into advanced neural network architectures and can help deepen your understanding of concepts related to BHAAN.

