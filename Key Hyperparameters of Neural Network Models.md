## Key Hyperparameters of Neural Network Models
Slide 1: Key Hyperparameters of Neural Network Models

Neural network models are powerful tools in machine learning, capable of learning complex patterns from data. Their performance is significantly influenced by various hyperparameters. This presentation explores the key hyperparameters that shape neural network behavior and performance.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
    
    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.a = self.sigmoid(self.z)
        self.output = np.dot(self.a, self.W2)
        return self.output
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Example usage
nn = NeuralNetwork(2, 3, 1)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output = nn.forward(X)
print("Output:", output)
```

Slide 2: Learning Rate

The learning rate is a crucial hyperparameter that determines the step size at each iteration while moving toward a minimum of the loss function. It influences how quickly or slowly a neural network learns.

```python
import tensorflow as tf

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with different learning rates
learning_rates = [0.1, 0.01, 0.001]
for lr in learning_rates:
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(f"Model compiled with learning rate: {lr}")

# The actual training would depend on your specific dataset
```

Slide 3: Batch Size

Batch size determines the number of samples processed before the model is updated. It affects both the quality of the model's convergence and the training time.

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a simple model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with different batch sizes
batch_sizes = [32, 64, 128]
for batch_size in batch_sizes:
    print(f"Training with batch size: {batch_size}")
    model.fit(x_train, y_train, epochs=1, batch_size=batch_size, validation_split=0.2)
```

Slide 4: Number of Hidden Layers

The number of hidden layers in a neural network affects its capacity to learn complex patterns. Deeper networks can potentially learn more intricate representations but may be more difficult to train.

```python
import tensorflow as tf

def create_model(num_hidden_layers):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)))
    
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(64, activation='relu'))
    
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

# Create models with different numbers of hidden layers
models = {f"{i}_hidden_layers": create_model(i) for i in range(1, 6)}

for name, model in models.items():
    print(f"Model with {name}:")
    model.summary()
    print("\n")
```

Slide 5: Number of Neurons per Hidden Layer

The number of neurons in each hidden layer determines the model's capacity to represent complex functions. More neurons can capture more intricate patterns but may lead to overfitting.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_and_train_model(neurons):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(neurons, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Generate synthetic data
    X = np.linspace(-1, 1, 200).reshape(-1, 1)
    y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.1, X.shape)
    
    model.fit(X, y, epochs=100, verbose=0)
    return model, X, y

neurons_list = [1, 5, 20, 100]
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for i, neurons in enumerate(neurons_list):
    model, X, y = create_and_train_model(neurons)
    predictions = model.predict(X)
    
    ax = axs[i // 2, i % 2]
    ax.scatter(X, y, label='Data')
    ax.plot(X, predictions, color='r', label='Prediction')
    ax.set_title(f'{neurons} Neuron(s)')
    ax.legend()

plt.tight_layout()
plt.show()
```

Slide 6: Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns. Common choices include ReLU, sigmoid, and tanh.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

x = np.linspace(-5, 5, 100)
functions = {'ReLU': relu, 'Sigmoid': sigmoid, 'Tanh': tanh}

fig, ax = plt.subplots(figsize=(10, 6))

for name, func in functions.items():
    ax.plot(x, func(x), label=name)

ax.legend()
ax.set_title('Common Activation Functions')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.grid(True)
plt.show()
```

Slide 7: Weight Initialization

Proper weight initialization is crucial for effective training. It can affect the speed of convergence and the quality of the final model.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_model(init):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', kernel_initializer=init, input_shape=(1,)),
        tf.keras.layers.Dense(1)
    ])

initializers = {
    'Zeros': 'zeros',
    'Random Normal': 'random_normal',
    'He Normal': 'he_normal',
    'Xavier': 'glorot_normal'
}

X = np.linspace(-1, 1, 1000).reshape(-1, 1)
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.1, X.shape)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.ravel()

for i, (name, init) in enumerate(initializers.items()):
    model = create_model(init)
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X, y, epochs=50, verbose=0)
    
    axs[i].plot(history.history['loss'])
    axs[i].set_title(f'{name} Initialization')
    axs[i].set_xlabel('Epoch')
    axs[i].set_ylabel('Loss')

plt.tight_layout()
plt.show()
```

Slide 8: Dropout Rate

Dropout is a regularization technique that helps prevent overfitting by randomly setting a fraction of input units to 0 during training.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_model(dropout_rate):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((60000, 784))
x_test = x_test.reshape((10000, 784))

dropout_rates = [0.0, 0.2, 0.5, 0.8]
histories = []

for rate in dropout_rates:
    model = create_model(rate)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=0)
    histories.append(history)

plt.figure(figsize=(10, 6))
for i, history in enumerate(histories):
    plt.plot(history.history['val_accuracy'], label=f'Dropout {dropout_rates[i]}')

plt.title('Validation Accuracy for Different Dropout Rates')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
```

Slide 9: L1 and L2 Regularization

L1 and L2 regularization are techniques to prevent overfitting by adding a penalty term to the loss function based on the model's weights.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_model(regularizer):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizer, input_shape=(100,)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizer),
        tf.keras.layers.Dense(1)
    ])

# Generate synthetic data
np.random.seed(0)
X = np.random.randn(1000, 100)
y = np.sum(X[:, :10], axis=1) + np.random.randn(1000) * 0.1

regularizers = {
    'No Regularization': None,
    'L1': tf.keras.regularizers.l1(0.01),
    'L2': tf.keras.regularizers.l2(0.01),
    'L1L2': tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
}

histories = {}

for name, regularizer in regularizers.items():
    model = create_model(regularizer)
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X, y, epochs=100, validation_split=0.2, verbose=0)
    histories[name] = history

plt.figure(figsize=(10, 6))
for name, history in histories.items():
    plt.plot(history.history['val_loss'], label=name)

plt.title('Validation Loss with Different Regularization')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()
```

Slide 10: Optimizer Choice

The choice of optimizer can significantly impact the training process and the final performance of the model. Common choices include SGD, Adam, and RMSprop.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

# Generate synthetic data
np.random.seed(0)
X = np.random.randn(1000, 20)
y = np.sum(X[:, :5], axis=1) + np.random.randn(1000) * 0.1

optimizers = {
    'SGD': tf.keras.optimizers.SGD(learning_rate=0.01),
    'Adam': tf.keras.optimizers.Adam(learning_rate=0.01),
    'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.01)
}

histories = {}

for name, optimizer in optimizers.items():
    model = create_model()
    model.compile(optimizer=optimizer, loss='mse')
    history = model.fit(X, y, epochs=100, validation_split=0.2, verbose=0)
    histories[name] = history

plt.figure(figsize=(10, 6))
for name, history in histories.items():
    plt.plot(history.history['loss'], label=name)

plt.title('Training Loss with Different Optimizers')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
plt.show()
```

Slide 11: Learning Rate Decay

Learning rate decay gradually reduces the learning rate during training, which can help fine-tune the model and potentially improve performance.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_model(lr_schedule):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Generate synthetic data
np.random.seed(0)
X = np.random.randn(1000, 20)
y = np.sum(X[:, :5], axis=1) + np.random.randn(1000) * 0.1

# Define learning rate schedules
constant_lr = 0.01
step_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01, decay_steps=20, decay_rate=0.9)
linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.01, decay_steps=100, end_learning_rate=0.001)

schedules = {
    'Constant': constant_lr,
    'Step Decay': step_decay,
    'Linear Decay': linear_decay
}

histories = {}

for name, schedule in schedules.items():
    model = create_model(schedule)
    history = model.fit(X, y, epochs=100, validation_split=0.2, verbose=0)
    histories[name] = history

plt.figure(figsize=(10, 6))
for name, history in histories.items():
    plt.plot(history.history['loss'], label=name)

plt.title('Training Loss with Different Learning Rate Schedules')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
plt.show()
```

Slide 12: Real-life Example: Image Classification

Image classification is a common application of neural networks. Here, we'll create a simple convolutional neural network (CNN) for classifying images from the CIFAR-10 dataset.

```python
import tensorflow as tf

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create a CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
```

Slide 13: Real-life Example: Natural Language Processing

Natural Language Processing (NLP) is another area where neural networks excel. Let's create a simple sentiment analysis model using a recurrent neural network (RNN).

```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load the IMDB dataset
max_features = 10000
maxlen = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Create an RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_features, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

# Train the model
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

# Evaluate the model
score = model.evaluate(x_test, y_test,
                       batch_size=128, verbose=0)
print('Test accuracy:', score[1])
```

Slide 14: Additional Resources

For those interested in diving deeper into neural network hyperparameters and architectures, here are some valuable resources:

1. "Neural Networks and Deep Learning" by Michael Nielsen Available online at: [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)
2. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville ArXiv link: [https://arxiv.org/abs/1601.06615](https://arxiv.org/abs/1601.06615)
3. "Efficient BackProp" by Yann LeCun et al. ArXiv link: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)
4. "Practical recommendations for gradient-based training of deep architectures" by Yoshua Bengio ArXiv link: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)

These resources provide in-depth explanations and practical advice on optimizing neural network performance through careful hyperparameter tuning and architecture design.

