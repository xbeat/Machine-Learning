## Introduction to Convolutional Neural Networks in Python
Slide 1: Introduction to Convolutional Neural Networks

Convolutional Neural Networks (CNNs) are a class of deep learning models primarily used for processing grid-like data, such as images. They are designed to automatically and adaptively learn spatial hierarchies of features from input data.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Creating a simple CNN model
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

model.summary()
```

Slide 2: Core Components of CNNs

The key components of CNNs are convolutional layers, pooling layers, and fully connected layers. Convolutional layers apply filters to detect features, pooling layers reduce spatial dimensions, and fully connected layers perform classification.

```python
# Convolutional Layer
conv_layer = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))

# Pooling Layer
pool_layer = layers.MaxPooling2D((2, 2))

# Fully Connected Layer
fc_layer = layers.Dense(64, activation='relu')

# Visualizing the output shape of each layer
input_shape = (28, 28, 1)
print(f"Input shape: {input_shape}")
print(f"Conv2D output shape: {conv_layer(tf.zeros(input_shape)).shape}")
print(f"MaxPooling2D output shape: {pool_layer(conv_layer(tf.zeros(input_shape))).shape}")
```

Slide 3: Convolutional Layers

Convolutional layers are the core building blocks of CNNs. They use filters to detect features in the input data, such as edges, textures, and patterns. The filters slide over the input, performing element-wise multiplication and summation.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 5x5 image
image = np.array([
    [0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 0]
])

# Define a 3x3 filter for edge detection
filter = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Perform convolution
output = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        output[i, j] = np.sum(image[i:i+3, j:j+3] * filter)

# Visualize the input, filter, and output
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Input Image')
ax2.imshow(filter, cmap='gray')
ax2.set_title('Filter')
ax3.imshow(output, cmap='gray')
ax3.set_title('Output')
plt.show()
```

Slide 4: Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns. Common activation functions include ReLU, sigmoid, and tanh. ReLU is widely used due to its simplicity and effectiveness in mitigating the vanishing gradient problem.

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

plt.figure(figsize=(12, 4))
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.legend()
plt.title('Activation Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
```

Slide 5: Pooling Layers

Pooling layers reduce the spatial dimensions of the feature maps, decreasing the computational load and helping to achieve spatial invariance. Common pooling operations include max pooling and average pooling.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a 4x4 input
input_data = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# Max pooling
def max_pool(input_data, pool_size):
    output_shape = input_data.shape[0] // pool_size
    output = np.zeros((output_shape, output_shape))
    for i in range(output_shape):
        for j in range(output_shape):
            output[i, j] = np.max(input_data[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size])
    return output

# Apply max pooling
max_pooled = max_pool(input_data, 2)

# Visualize input and output
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(input_data, cmap='viridis')
ax1.set_title('Input')
ax2.imshow(max_pooled, cmap='viridis')
ax2.set_title('Max Pooled Output')
plt.show()
```

Slide 6: Fully Connected Layers

Fully connected layers are typically used at the end of the CNN architecture for classification tasks. They take the flattened output of the convolutional and pooling layers and produce the final output predictions.

```python
import numpy as np

# Simulating the output of convolutional and pooling layers
flattened_input = np.random.rand(1, 64)  # 64 features

# Weights and biases for a fully connected layer
weights = np.random.rand(64, 10)  # 10 output classes
biases = np.random.rand(10)

# Forward pass through the fully connected layer
output = np.dot(flattened_input, weights) + biases

# Apply softmax activation for classification
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

probabilities = softmax(output)

print("Output probabilities:")
print(probabilities)
print("\nPredicted class:", np.argmax(probabilities))
```

Slide 7: Training CNNs

Training CNNs involves forward propagation, loss calculation, backpropagation, and parameter updates. The process aims to minimize the difference between predicted and actual outputs.

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Create the CNN model
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

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, validation_split=0.2)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

Slide 8: CNN Architectures

Various CNN architectures have been developed over time, each with its unique characteristics. Some popular architectures include LeNet, AlexNet, VGGNet, and ResNet.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_lenet():
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)),
        layers.AveragePooling2D((2, 2)),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.AveragePooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

lenet = create_lenet()
lenet.summary()
```

Slide 9: Transfer Learning

Transfer learning allows us to leverage pre-trained models on large datasets and fine-tune them for specific tasks. This approach is particularly useful when working with limited data.

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom layers for fine-tuning
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

Slide 10: Data Augmentation

Data augmentation is a technique used to artificially increase the size of the training dataset by applying various transformations to existing images. This helps improve model generalization and reduces overfitting.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Create an instance of ImageDataGenerator with augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Load a sample image
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
image = train_images[0].reshape((1, 28, 28, 1)).astype('float32') / 255

# Generate augmented images
aug_iter = datagen.flow(image, batch_size=1)

# Display original and augmented images
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
axs[0].imshow(image[0, :, :, 0], cmap='gray')
axs[0].set_title('Original')
for i in range(4):
    aug_image = next(aug_iter)[0, :, :, 0]
    axs[i+1].imshow(aug_image, cmap='gray')
    axs[i+1].set_title(f'Augmented {i+1}')
plt.show()
```

Slide 11: Real-Life Example: Image Classification

CNNs are widely used in image classification tasks. Let's demonstrate this with a simple example of classifying handwritten digits using the MNIST dataset.

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Create and train the model
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

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Make predictions
predictions = model.predict(test_images[:5])
print("\nPredictions:")
print(predictions)

# Display some test images and their predicted labels
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    axs[i].imshow(test_images[i, :, :, 0], cmap='gray')
    axs[i].set_title(f'Predicted: {predictions[i].argmax()}')
    axs[i].axis('off')
plt.show()
```

Slide 12: Real-Life Example: Object Detection

CNNs are also used in object detection tasks, where the goal is to identify and locate objects within an image. Here's a simplified example using a pre-trained model for object detection.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub

# Load a pre-trained object detection model
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Function to load and preprocess an image
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, scores, classes, threshold=0.5):
    for i in range(len(boxes)):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                              fill=False, edgecolor='red', linewidth=2))
            plt.text(xmin, ymin, f'Class {classes[i]}: {scores[i]:.2f}',
                     bbox=dict(facecolor='red', alpha=0.5))

# Load and preprocess a sample image
image_path = tf.keras.utils.get_file("example_image.jpg", "https://example.com/image.jpg")
input_tensor = load_image(image_path)

# Perform object detection
output = model(input_tensor)

# Process the output
boxes = output["detection_boxes"][0].numpy()
scores = output["detection_scores"][0].numpy()
classes = output["detection_classes"][0].numpy().astype(int)

# Visualize the results
plt.figure(figsize=(12, 8))
plt.imshow(input_tensor[0])
draw_boxes(input_tensor[0], boxes, scores, classes)
plt.axis('off')
plt.show()
```

Slide 13: CNN Applications in Medical Imaging

CNNs have found significant applications in medical imaging, particularly in the analysis of X-rays, MRIs, and CT scans. They can assist in detecting abnormalities, classifying diseases, and segmenting organs or tumors.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define a simple CNN for medical image classification
def create_medical_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Example usage
input_shape = (256, 256, 1)  # Grayscale medical images
num_classes = 2  # Binary classification (e.g., normal vs. abnormal)

model = create_medical_cnn(input_shape, num_classes)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
```

Slide 14: CNN Applications in Natural Language Processing

While CNNs are primarily associated with image processing, they have also been applied to natural language processing tasks. In NLP, CNNs can be used for text classification, sentiment analysis, and even machine translation.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define a simple CNN for text classification
def create_text_cnn(max_words, embedding_dim, max_length, num_classes):
    model = models.Sequential([
        layers.Embedding(max_words, embedding_dim, input_length=max_length),
        layers.Conv1D(128, 5, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Example usage
max_words = 10000  # Vocabulary size
embedding_dim = 100  # Embedding dimension
max_length = 100  # Maximum sequence length
num_classes = 3  # Number of classes for classification

model = create_text_cnn(max_words, embedding_dim, max_length, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
```

Slide 15: Additional Resources

For those interested in delving deeper into Convolutional Neural Networks, here are some valuable resources:

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444. ArXiv: [https://arxiv.org/abs/1807.07987](https://arxiv.org/abs/1807.07987)
2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in neural information processing systems, 25. ArXiv: [https://arxiv.org/abs/1404.5997](https://arxiv.org/abs/1404.5997)
3. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. ArXiv: [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778). ArXiv: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

These papers provide foundational knowledge and advanced concepts in CNN architectures and applications.

