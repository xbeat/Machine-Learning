## Top TensorFlow Functions for Data Science
Slide 1: TensorFlow Functions for Data Scientists

TensorFlow is a powerful open-source library for machine learning and deep learning. This presentation covers key TensorFlow functions that data scientists frequently use in their work, providing code examples and practical applications.

```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
```

Slide 2: tf.constant()

Creates a constant tensor from a tensor-like object. This function is fundamental for defining fixed values in TensorFlow computations.

```python
# Creating a constant tensor
constant_tensor = tf.constant([1, 2, 3, 4, 5])
print(constant_tensor)

# Creating a 2D constant tensor
matrix = tf.constant([[1, 2], [3, 4]])
print(matrix)
```

Slide 3: tf.Variable()

Creates a new variable with the specified initial value. Variables are essential for storing and updating model parameters during training.

```python
# Creating a variable
initial_value = tf.constant([1.0, 2.0, 3.0])
variable = tf.Variable(initial_value)
print(variable)

# Updating a variable
variable.assign([4.0, 5.0, 6.0])
print(variable)
```

Slide 4: tf.GradientTape()

Records operations for automatic differentiation. This is crucial for implementing custom training loops and advanced optimization techniques.

```python
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x**2

dy_dx = tape.gradient(y, x)
print(f"dy/dx at x = 3: {dy_dx.numpy()}")
```

Slide 5: tf.keras.Sequential()

Creates a linear stack of layers for building neural networks. This high-level API simplifies the process of constructing complex models.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
```

Slide 6: tf.data.Dataset.from\_tensor\_slices()

Creates a dataset from tensor slices. This function is essential for efficiently loading and preprocessing large datasets.

```python
# Creating a dataset from numpy arrays
import numpy as np

x = np.arange(10)
y = x * 2

dataset = tf.data.Dataset.from_tensor_slices((x, y))

for element in dataset.take(5):
    print(f"x: {element[0].numpy()}, y: {element[1].numpy()}")
```

Slide 7: tf.keras.layers.Dense()

Implements a densely-connected neural network layer. This is a fundamental building block for creating various types of neural networks.

```python
# Creating a dense layer
dense_layer = tf.keras.layers.Dense(units=64, activation='relu')

# Apply the layer to an input
input_data = tf.random.normal([1, 10])
output = dense_layer(input_data)

print(f"Input shape: {input_data.shape}")
print(f"Output shape: {output.shape}")
```

Slide 8: tf.nn.softmax()

Computes softmax activations. This function is commonly used in the output layer of classification models to convert logits into probabilities.

```python
# Computing softmax
logits = tf.constant([[2.0, 1.0, 0.1]])
probabilities = tf.nn.softmax(logits)

print(f"Logits: {logits.numpy()}")
print(f"Probabilities: {probabilities.numpy()}")
```

Slide 9: tf.train.AdamOptimizer()

Implements the Adam optimization algorithm. This optimizer is widely used for training deep learning models due to its efficiency and adaptive learning rates.

```python
# Creating an Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Using the optimizer in a simple gradient descent loop
x = tf.Variable(0.0)

for _ in range(5):
    with tf.GradientTape() as tape:
        y = x**2
    
    grads = tape.gradient(y, x)
    optimizer.apply_gradients([(grads, x)])
    print(f"Step: {_}, x: {x.numpy()}, y: {y.numpy()}")
```

Slide 10: tf.keras.layers.Conv2D()

Implements a 2D convolution layer. This layer is crucial for image processing tasks and convolutional neural networks.

```python
# Creating a Conv2D layer
conv_layer = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')

# Apply the layer to an input
input_image = tf.random.normal([1, 28, 28, 1])
output = conv_layer(input_image)

print(f"Input shape: {input_image.shape}")
print(f"Output shape: {output.shape}")
```

Slide 11: tf.image.resize()

Resizes images to a specified size using different methods. This function is essential for preprocessing image data and ensuring consistent input sizes for models.

```python
# Resizing an image
original_image = tf.random.normal([1, 100, 100, 3])
resized_image = tf.image.resize(original_image, [224, 224])

print(f"Original shape: {original_image.shape}")
print(f"Resized shape: {resized_image.shape}")
```

Slide 12: Real-Life Example: Image Classification

Let's use some of the functions we've learned to create a simple image classification model for the MNIST dataset.

```python
# Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")
```

Slide 13: Real-Life Example: Time Series Forecasting

Now, let's use TensorFlow to create a simple time series forecasting model for temperature prediction.

```python
import numpy as np

# Generate synthetic temperature data
time = np.arange(365)
temp = 20 + 10 * np.sin(2 * np.pi * time / 365) + np.random.randn(365) * 3

# Prepare data for the model
def create_time_series(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

time_step = 7
X, y = create_time_series(temp, time_step)

# Create and train the model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(time_step, 1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X.reshape(-1, time_step, 1), y, epochs=50, verbose=0)

# Make predictions
last_7_days = temp[-7:].reshape(1, 7, 1)
next_day_temp = model.predict(last_7_days)
print(f"Predicted temperature for the next day: {next_day_temp[0][0]:.2f}°C")
```

Slide 14: Additional Resources

For more in-depth information on TensorFlow and its applications in data science, consider exploring these resources:

1. TensorFlow Official Documentation: [https://www.tensorflow.org/api\_docs](https://www.tensorflow.org/api_docs)
2. "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron
3. ArXiv paper: "TensorFlow: A System for Large-Scale Machine Learning" (Dean et al., 2016) - [https://arxiv.org/abs/1605.08695](https://arxiv.org/abs/1605.08695)

These resources provide comprehensive guides, tutorials, and research papers to further enhance your understanding of TensorFlow and its applications in data science.

