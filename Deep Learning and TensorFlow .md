## Deep Learning and TensorFlow 

Slide 1: Introduction to Deep Learning

Deep Learning is a subset of Machine Learning that uses neural networks with multiple layers to learn from data. It has revolutionized various fields, including computer vision, natural language processing, and game playing. This presentation will cover key concepts and models in Deep Learning, focusing on practical implementations using Python.

```python
# Simple neural network implementation
import random

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)

    def activate(self, inputs):
        return sum(w * i for w, i in zip(self.weights, inputs)) + self.bias > 0

class SimpleNN:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.hidden_layer = [Neuron(num_inputs) for _ in range(num_hidden)]
        self.output_layer = [Neuron(num_hidden) for _ in range(num_outputs)]

    def predict(self, inputs):
        hidden_outputs = [neuron.activate(inputs) for neuron in self.hidden_layer]
        return [neuron.activate(hidden_outputs) for neuron in self.output_layer]

# Example usage
nn = SimpleNN(2, 3, 1)
print(nn.predict([1, 0]))  # Example input
```

Slide 2: LeNet Architecture

LeNet, introduced by Yann LeCun in 1998, is one of the earliest convolutional neural networks. It was primarily used for digit recognition tasks and laid the foundation for many modern CNN architectures. LeNet consists of two sets of convolutional and average pooling layers, followed by a flattening convolutional layer, then two fully-connected layers and finally a softmax classifier.

```python
# LeNet-5 architecture implementation
class LeNet5:
    def __init__(self):
        self.conv1 = ConvLayer(1, 6, 5)  # 1 input channel, 6 filters, 5x5 kernel
        self.pool1 = AvgPoolLayer(2, 2)  # 2x2 pooling
        self.conv2 = ConvLayer(6, 16, 5)
        self.pool2 = AvgPoolLayer(2, 2)
        self.conv3 = ConvLayer(16, 120, 5)
        self.fc1 = FCLayer(120, 84)
        self.fc2 = FCLayer(84, 10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.pool2.forward(x)
        x = self.conv3.forward(x)
        x = x.flatten()
        x = self.fc1.forward(x)
        x = self.fc2.forward(x)
        return softmax(x)

# Note: ConvLayer, AvgPoolLayer, FCLayer, and softmax functions are not implemented here
```

Slide 3: AlexNet Architecture

AlexNet, developed by Alex Krizhevsky in 2012, marked a significant breakthrough in the field of computer vision. It won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012, demonstrating the power of deep convolutional neural networks. AlexNet consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final softmax output.

```python
# AlexNet architecture implementation
class AlexNet:
    def __init__(self, num_classes=1000):
        self.features = [
            ConvLayer(3, 96, 11, stride=4),  # Conv1
            ReLU(),
            MaxPoolLayer(3, stride=2),
            ConvLayer(96, 256, 5, padding=2),  # Conv2
            ReLU(),
            MaxPoolLayer(3, stride=2),
            ConvLayer(256, 384, 3, padding=1),  # Conv3
            ReLU(),
            ConvLayer(384, 384, 3, padding=1),  # Conv4
            ReLU(),
            ConvLayer(384, 256, 3, padding=1),  # Conv5
            ReLU(),
            MaxPoolLayer(3, stride=2),
        ]
        self.classifier = [
            FCLayer(6*6*256, 4096),
            ReLU(),
            Dropout(0.5),
            FCLayer(4096, 4096),
            ReLU(),
            Dropout(0.5),
            FCLayer(4096, num_classes),
        ]

    def forward(self, x):
        for layer in self.features:
            x = layer.forward(x)
        x = x.flatten()
        for layer in self.classifier:
            x = layer.forward(x)
        return softmax(x)

# Note: ConvLayer, ReLU, MaxPoolLayer, FCLayer, Dropout, and softmax functions are not implemented here
```

Slide 4: ResNet Architecture

ResNet (Residual Network), introduced by Kaiming He et al. in 2015, addressed the problem of training very deep neural networks. It introduced skip connections, allowing the network to learn residual functions with reference to the layer inputs. This innovation enabled the training of networks with hundreds or even thousands of layers.

```python
# Basic ResNet block implementation
class ResidualBlock:
    def __init__(self, in_channels, out_channels, stride=1):
        self.conv1 = ConvLayer(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = BatchNorm(out_channels)
        self.conv2 = ConvLayer(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = BatchNorm(out_channels)
        
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                ConvLayer(in_channels, out_channels, 1, stride=stride),
                BatchNorm(out_channels)
            )

    def forward(self, x):
        residual = x
        
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = relu(out)
        
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        
        if self.shortcut:
            residual = self.shortcut.forward(x)
        
        out += residual
        out = relu(out)
        
        return out

# Note: ConvLayer, BatchNorm, and relu functions are not implemented here
```

Slide 5: Generative Adversarial Networks (GANs)

Generative Adversarial Networks, introduced by Ian Goodfellow et al. in 2014, are a class of AI algorithms used in unsupervised machine learning. GANs consist of two neural networks, a generator and a discriminator, which are trained simultaneously through adversarial training. The generator creates synthetic data samples, while the discriminator attempts to distinguish between real and generated samples.

```python
# Simple GAN implementation
class Generator:
    def __init__(self, latent_dim, output_dim):
        self.model = [
            FCLayer(latent_dim, 128),
            LeakyReLU(0.2),
            FCLayer(128, 256),
            LeakyReLU(0.2),
            FCLayer(256, output_dim),
            Tanh()
        ]

    def forward(self, z):
        for layer in self.model:
            z = layer.forward(z)
        return z

class Discriminator:
    def __init__(self, input_dim):
        self.model = [
            FCLayer(input_dim, 256),
            LeakyReLU(0.2),
            FCLayer(256, 128),
            LeakyReLU(0.2),
            FCLayer(128, 1),
            Sigmoid()
        ]

    def forward(self, x):
        for layer in self.model:
            x = layer.forward(x)
        return x

# Training loop (simplified)
def train_gan(generator, discriminator, real_data, num_epochs):
    for epoch in range(num_epochs):
        # Train discriminator
        real_output = discriminator.forward(real_data)
        z = generate_noise(batch_size, latent_dim)
        fake_data = generator.forward(z)
        fake_output = discriminator.forward(fake_data)
        
        d_loss = -torch.mean(torch.log(real_output) + torch.log(1 - fake_output))
        update_discriminator(d_loss)
        
        # Train generator
        z = generate_noise(batch_size, latent_dim)
        fake_data = generator.forward(z)
        fake_output = discriminator.forward(fake_data)
        
        g_loss = -torch.mean(torch.log(fake_output))
        update_generator(g_loss)

# Note: FCLayer, LeakyReLU, Tanh, Sigmoid, generate_noise, update_discriminator, and update_generator functions are not implemented here
```

Slide 6: AlphaGo

AlphaGo, developed by DeepMind, is an AI program that plays the board game Go. It was the first computer program to defeat a professional human Go player, a feat previously thought to be at least a decade away. AlphaGo combines advanced tree search with deep neural networks, using a Monte Carlo tree search algorithm guided by two deep neural networks: a policy network and a value network.

```python
# Simplified AlphaGo-style neural network
class GoNeuralNetwork:
    def __init__(self, board_size):
        self.board_size = board_size
        self.conv1 = ConvLayer(1, 32, 3, padding=1)
        self.conv2 = ConvLayer(32, 64, 3, padding=1)
        self.conv3 = ConvLayer(64, 128, 3, padding=1)
        
        # Policy head
        self.policy_conv = ConvLayer(128, 2, 1)
        self.policy_fc = FCLayer(2 * board_size * board_size, board_size * board_size + 1)
        
        # Value head
        self.value_conv = ConvLayer(128, 1, 1)
        self.value_fc1 = FCLayer(board_size * board_size, 256)
        self.value_fc2 = FCLayer(256, 1)

    def forward(self, x):
        x = relu(self.conv1.forward(x))
        x = relu(self.conv2.forward(x))
        x = relu(self.conv3.forward(x))
        
        # Policy output
        policy = relu(self.policy_conv.forward(x))
        policy = policy.flatten()
        policy = self.policy_fc.forward(policy)
        policy = softmax(policy)
        
        # Value output
        value = relu(self.value_conv.forward(x))
        value = value.flatten()
        value = relu(self.value_fc1.forward(value))
        value = tanh(self.value_fc2.forward(value))
        
        return policy, value

# Note: ConvLayer, FCLayer, relu, softmax, and tanh functions are not implemented here
```

Slide 7: Introduction to TensorFlow

TensorFlow is an open-source machine learning framework developed by Google. It provides a flexible ecosystem of tools, libraries, and community resources for building and deploying machine learning models. TensorFlow operates on tensors, which are multi-dimensional arrays of data. Let's explore some basic TensorFlow operations.

```python
import tensorflow as tf

# Creating tensors
scalar = tf.constant(42)
vector = tf.constant([1, 2, 3])
matrix = tf.constant([[1, 2], [3, 4]])

# Basic operations
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

addition = tf.add(a, b)
multiplication = tf.matmul(a, b)

# Using with statement to create a session and run operations
with tf.Session() as sess:
    print("Addition result:", sess.run(addition))
    print("Multiplication result:", sess.run(multiplication))
```

Slide 8: Tensors, Scalars, Vectors, and Matrices

In TensorFlow, data is represented as tensors. A tensor is a generalization of vectors and matrices to potentially higher dimensions. Let's explore the different types of tensors and how to manipulate them.

```python
import tensorflow as tf

# Scalar (0-D tensor)
scalar = tf.constant(42)

# Vector (1-D tensor)
vector = tf.constant([1, 2, 3, 4])

# Matrix (2-D tensor)
matrix = tf.constant([[1, 2], [3, 4], [5, 6]])

# 3-D tensor
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Tensor operations
reshaped_matrix = tf.reshape(matrix, [2, 3])
transposed_matrix = tf.transpose(matrix)

with tf.Session() as sess:
    print("Scalar:", sess.run(scalar))
    print("Vector:", sess.run(vector))
    print("Matrix:", sess.run(matrix))
    print("3D Tensor:", sess.run(tensor_3d))
    print("Reshaped Matrix:", sess.run(reshaped_matrix))
    print("Transposed Matrix:", sess.run(transposed_matrix))
```

Slide 9: Recurrent Neural Networks (RNNs)

Recurrent Neural Networks are a class of neural networks designed to work with sequential data. They maintain an internal state (memory) that allows them to process sequences of inputs. RNNs are particularly useful for tasks like natural language processing and time series analysis.

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        outputs = []
        for x in inputs:
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            outputs.append(y)
        return outputs, h

# Example usage
rnn = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
inputs = [np.random.randn(10, 1) for _ in range(5)]  # Sequence of 5 input vectors
outputs, final_hidden = rnn.forward(inputs)
print(f"Output sequence length: {len(outputs)}")
print(f"Shape of last output: {outputs[-1].shape}")
```

Slide 10: Long Short-Term Memory (LSTM) Networks

Long Short-Term Memory networks are a special kind of RNN capable of learning long-term dependencies. LSTMs have a more complex structure with gates that regulate the flow of information, allowing them to remember or forget information over long sequences.

```python
import numpy as np

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, prev_h, prev_c):
        # Concatenate input and previous hidden state
        combined = np.vstack((x, prev_h))
        
        # Compute gate activations
        f = self.sigmoid(np.dot(self.Wf, combined) + self.bf)
        i = self.sigmoid(np.dot(self.Wi, combined) + self.bi)
        c_tilde = np.tanh(np.dot(self.Wc, combined) + self.bc)
        o = self.sigmoid(np.dot(self.Wo, combined) + self.bo)
        
        # Update cell state and hidden state
        c = f * prev_c + i * c_tilde
        h = o * np.tanh(c)
        
        return h, c

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Example usage
lstm_cell = LSTMCell(input_size=10, hidden_size=20)
x = np.random.randn(10, 1)
prev_h = np.zeros((20, 1))
prev_c = np.zeros((20, 1))
h, c = lstm_cell.forward(x, prev_h, prev_c)
print(f"Hidden state shape: {h.shape}")
print(f"Cell state shape: {c.shape}")
```

Slide 11: Neural Turing Machines

Neural Turing Machines (NTMs) are a class of neural network models that couple a neural network with external memory resources, which they can interact with by attentional processes. This architecture allows the network to read from and write to the memory, similar to how a Turing machine operates.

```python
import numpy as np

class NeuralTuringMachine:
    def __init__(self, input_size, output_size, memory_size, memory_vector_dim):
        self.controller = FeedForwardNetwork(input_size, output_size)
        self.memory = np.zeros((memory_size, memory_vector_dim))
        self.read_head = ReadHead(memory_vector_dim)
        self.write_head = WriteHead(memory_vector_dim)

    def forward(self, x):
        controller_output = self.controller.forward(x)
        read_vector = self.read_head.read(self.memory)
        self.write_head.write(self.memory, controller_output)
        output = np.concatenate([controller_output, read_vector])
        return output

class ReadHead:
    def read(self, memory):
        # Simplified read operation
        return np.mean(memory, axis=0)

class WriteHead:
    def write(self, memory, write_vector):
        # Simplified write operation
        memory += np.outer(np.ones(memory.shape[0]), write_vector)

class FeedForwardNetwork:
    def forward(self, x):
        # Simplified feedforward network
        return np.tanh(x)

# Example usage
ntm = NeuralTuringMachine(input_size=10, output_size=5, memory_size=100, memory_vector_dim=20)
input_vector = np.random.randn(10)
output = ntm.forward(input_vector)
print(f"Output shape: {output.shape}")
```

Slide 12: Tensor Manipulation and Broadcasting

Tensor manipulation and broadcasting are fundamental concepts in working with multi-dimensional arrays. Broadcasting allows operations between arrays of different shapes, automatically expanding smaller arrays to match the shape of larger ones.

```python
import numpy as np

# Create tensors
a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])

# Broadcasting addition
c = a + b
print("Broadcasting addition result:")
print(c)

# Reshaping tensors
d = a.reshape(3, 2)
print("\nReshaped tensor:")
print(d)

# Transposing tensors
e = a.T
print("\nTransposed tensor:")
print(e)

# Element-wise operations
f = np.square(a)
print("\nElement-wise squaring:")
print(f)

# Matrix multiplication
g = np.dot(a, b)
print("\nMatrix multiplication result:")
print(g)

# Aggregation operations
print("\nSum of all elements:", np.sum(a))
print("Mean of all elements:", np.mean(a))
print("Max element:", np.max(a))
print("Min element:", np.min(a))
```

Slide 13: Graph Operations in TensorFlow

TensorFlow uses a computational graph model where operations are represented as nodes in a graph, and the edges represent the flow of data between operations. This allows for efficient computation and automatic differentiation.

```python
import tensorflow as tf

# Create a computational graph
a = tf.constant(3.0, name='a')
b = tf.constant(4.0, name='b')
c = tf.add(a, b, name='add')
d = tf.multiply(a, b, name='multiply')
e = tf.pow(c, d, name='power')

# Create a session and run the graph
with tf.Session() as sess:
    result = sess.run(e)
    print("Result:", result)

    # Get the default graph and print operations
    graph = tf.get_default_graph()
    operations = graph.get_operations()
    print("\nOperations in the graph:")
    for op in operations:
        print(op.name, op.type)

    # Visualize the graph (requires graphviz)
    # tf.summary.FileWriter('logs', graph)
    # print("Graph visualization saved in 'logs' directory")
```

Slide 14: Real-life Example: Image Classification

Let's implement a simple Convolutional Neural Network (CNN) for image classification using TensorFlow. This example demonstrates how to build and train a model for recognizing handwritten digits from the MNIST dataset.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')
```

Slide 15: Real-life Example: Natural Language Processing

In this example, we'll implement a simple sentiment analysis model using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) cells. We'll use the IMDB movie review dataset for binary sentiment classification.

```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load the IMDB dataset
max_features = 10000
maxlen = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to ensure uniform length
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_features, 128),
    tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=5,
                    validation_split=0.2)

# Evaluate the model
score, acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {acc}')
```

Slide 16: Additional Resources

For those interested in diving deeper into Deep Learning and TensorFlow, here are some valuable resources:

1.  ArXiv.org: A repository of research papers in various fields, including machine learning and artificial intelligence.
    *   Example: "Deep Learning" by Yann LeCun, Yoshua Bengio, and Geoffrey Hinton ([https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597))
2.  TensorFlow Documentation: Official documentation and tutorials for TensorFlow ([https://www.tensorflow.org/learn](https://www.tensorflow.org/learn))
3.  Deep Learning Book by Ian Goodfellow, Yoshua Bengio, and Aaron Courville ([https://www.deeplearningbook.org/](https://www.deeplearningbook.org/))
4.  Coursera: Deep Learning Specialization by Andrew Ng ([https://www.coursera.org/specializations/deep-learning](https://www.coursera.org/specializations/deep-learning))
5.  Fast.ai: Practical Deep Learning for Coders ([https://course.fast.ai/](https://course.fast.ai/))

These resources provide a mix of theoretical foundations and practical implementations to help you expand your knowledge and skills in Deep Learning and TensorFlow.

