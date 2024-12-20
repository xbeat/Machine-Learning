## Introduction to Padding, Strides, and Pooling in Deep Learning with Python
Slide 1: Introduction to Padding, Strides, and Pooling in Deep Learning

Padding, strides, and pooling are fundamental concepts in deep learning, particularly in convolutional neural networks (CNNs). These techniques help manage the spatial dimensions of data as it passes through the network, allowing for more effective feature extraction and computational efficiency.

```python
import numpy as np
import matplotlib.pyplot as plt

# Creating a simple 5x5 input
input_data = np.random.rand(5, 5)

plt.imshow(input_data, cmap='viridis')
plt.title('Input Data')
plt.colorbar()
plt.show()
```

Slide 2: Padding: Preserving Spatial Dimensions

Padding involves adding extra elements around the edges of an input, typically zeros. This technique helps preserve the spatial dimensions of the input after convolution operations, allowing for better retention of information at the borders.

```python
def add_padding(input_data, pad_width):
    return np.pad(input_data, pad_width, mode='constant', constant_values=0)

padded_input = add_padding(input_data, pad_width=1)

plt.imshow(padded_input, cmap='viridis')
plt.title('Padded Input (pad_width=1)')
plt.colorbar()
plt.show()
```

Slide 3: Types of Padding

There are various types of padding, including zero padding (most common), reflection padding, and replication padding. Zero padding adds zeros around the input, while reflection and replication padding use existing values from the input to fill the padded area.

```python
def compare_padding_types(input_data):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    paddings = [
        ('constant', 0),
        ('reflect', None),
        ('edge', None)
    ]
    
    for i, (mode, value) in enumerate(paddings):
        padded = np.pad(input_data, 1, mode=mode, constant_values=value)
        axs[i].imshow(padded, cmap='viridis')
        axs[i].set_title(f'{mode.capitalize()} Padding')
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()

compare_padding_types(input_data)
```

Slide 4: Strides: Controlling Feature Map Size

Strides determine the step size of the convolution operation. By adjusting the stride, we can control the size of the output feature map. A larger stride results in a smaller output, which can be useful for reducing computational complexity.

```python
def apply_stride(input_data, stride):
    return input_data[::stride, ::stride]

strided_input = apply_stride(input_data, stride=2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(input_data, cmap='viridis')
ax1.set_title('Original Input')
ax2.imshow(strided_input, cmap='viridis')
ax2.set_title('Strided Input (stride=2)')
plt.tight_layout()
plt.show()
```

Slide 5: Impact of Strides on Feature Maps

Strides affect the spatial dimensions of the output feature map. The relationship between input size, kernel size, stride, and output size can be expressed as:

output\_size = (input\_size - kernel\_size) / stride + 1

```python
def calculate_output_size(input_size, kernel_size, stride):
    return (input_size - kernel_size) // stride + 1

input_sizes = range(5, 21, 5)
kernel_size = 3
strides = [1, 2, 3]

for stride in strides:
    output_sizes = [calculate_output_size(size, kernel_size, stride) for size in input_sizes]
    plt.plot(input_sizes, output_sizes, marker='o', label=f'Stride {stride}')

plt.xlabel('Input Size')
plt.ylabel('Output Size')
plt.title('Impact of Strides on Output Size')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 6: Pooling: Downsampling Feature Maps

Pooling is a technique used to reduce the spatial dimensions of feature maps. It helps in reducing computational complexity and providing a form of translational invariance. Common pooling operations include max pooling and average pooling.

```python
def pooling(input_data, pool_size, mode='max'):
    output = np.zeros((input_data.shape[0] // pool_size, input_data.shape[1] // pool_size))
    for i in range(0, input_data.shape[0], pool_size):
        for j in range(0, input_data.shape[1], pool_size):
            if mode == 'max':
                output[i//pool_size, j//pool_size] = np.max(input_data[i:i+pool_size, j:j+pool_size])
            elif mode == 'avg':
                output[i//pool_size, j//pool_size] = np.mean(input_data[i:i+pool_size, j:j+pool_size])
    return output

max_pooled = pooling(input_data, pool_size=2, mode='max')
avg_pooled = pooling(input_data, pool_size=2, mode='avg')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(input_data, cmap='viridis')
ax1.set_title('Original Input')
ax2.imshow(max_pooled, cmap='viridis')
ax2.set_title('Max Pooling (2x2)')
ax3.imshow(avg_pooled, cmap='viridis')
ax3.set_title('Average Pooling (2x2)')
plt.tight_layout()
plt.show()
```

Slide 7: Max Pooling vs. Average Pooling

Max pooling selects the maximum value in each pooling window, while average pooling computes the average of all values in the window. Max pooling is often preferred as it tends to capture the most prominent features.

```python
def compare_pooling(input_data, pool_size):
    max_pooled = pooling(input_data, pool_size, mode='max')
    avg_pooled = pooling(input_data, pool_size, mode='avg')
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(input_data, cmap='viridis')
    axs[0].set_title('Original Input')
    axs[1].imshow(max_pooled, cmap='viridis')
    axs[1].set_title(f'Max Pooling ({pool_size}x{pool_size})')
    axs[2].imshow(avg_pooled, cmap='viridis')
    axs[2].set_title(f'Average Pooling ({pool_size}x{pool_size})')
    
    plt.tight_layout()
    plt.show()

compare_pooling(input_data, pool_size=2)
```

Slide 8: Real-Life Example: Image Classification

In image classification tasks, CNNs use padding, strides, and pooling to process input images efficiently. Let's consider a simple example of classifying handwritten digits using the MNIST dataset.

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

Slide 9: Training and Evaluating the MNIST Model

Let's train the model on the MNIST dataset and evaluate its performance.

```python
# Train the model
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1, batch_size=128, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 10: Visualizing Convolution and Pooling Operations

To better understand how convolution and pooling work in practice, let's visualize the intermediate feature maps produced by our trained model.

```python
import tensorflow as tf

# Get a sample image
sample_image = x_test[0:1]

# Create a model that outputs all layers
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

# Get the feature maps
activations = activation_model.predict(sample_image)

# Plot the feature maps
def plot_feature_maps(activations, layer_names):
    for layer_name, layer_activation in zip(layer_names, activations):
        if len(layer_activation.shape) == 4:
            n_features = layer_activation.shape[-1]
            size = layer_activation.shape[1]
            n_cols = min(n_features, 8)
            n_rows = (n_features - 1) // n_cols + 1
            fig = plt.figure(figsize=(n_cols * 2, n_rows * 2))
            for i in range(n_features):
                plt.subplot(n_rows, n_cols, i + 1)
                plt.imshow(layer_activation[0, :, :, i], cmap='viridis')
                plt.axis('off')
            plt.suptitle(f"{layer_name} - Feature Maps")
            plt.show()

layer_names = [layer.name for layer in model.layers if isinstance(layer, (Conv2D, MaxPooling2D))]
plot_feature_maps(activations[:4], layer_names)  # Plot only conv and pooling layers
```

Slide 11: Real-Life Example: Image Segmentation

Image segmentation is another task that heavily relies on padding, strides, and pooling. Let's implement a simple U-Net architecture for image segmentation.

```python
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    
    # Encoder (Downsampling)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bridge
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    
    # Decoder (Upsampling)
    up4 = UpSampling2D(size=(2, 2))(conv3)
    up4 = concatenate([up4, conv2])
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(up4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
    
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = concatenate([up5, conv1])
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

unet_model = unet()
unet_model.summary()
```

Slide 12: Implementing Custom Padding and Pooling Layers

While deep learning frameworks provide built-in padding and pooling layers, understanding how to implement them from scratch can be beneficial. Let's create custom padding and pooling layers using TensorFlow.

```python
import tensorflow as tf

class CustomPadding(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), mode='CONSTANT', constant_values=0, **kwargs):
        super(CustomPadding, self).__init__(**kwargs)
        self.padding = padding
        self.mode = mode
        self.constant_values = constant_values

    def call(self, inputs):
        return tf.pad(inputs,
                      [[0, 0], [self.padding[0], self.padding[0]], [self.padding[1], self.padding[1]], [0, 0]],
                      mode=self.mode,
                      constant_values=self.constant_values)

class CustomPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='VALID', pool_mode='MAX', **kwargs):
        super(CustomPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.padding = padding
        self.pool_mode = pool_mode

    def call(self, inputs):
        if self.pool_mode == 'MAX':
            return tf.nn.max_pool2d(inputs, self.pool_size, self.strides, self.padding)
        elif self.pool_mode == 'AVG':
            return tf.nn.avg_pool2d(inputs, self.pool_size, self.strides, self.padding)

# Example usage
input_tensor = tf.random.normal([1, 28, 28, 1])
padded = CustomPadding(padding=(2, 2))(input_tensor)
pooled = CustomPooling(pool_size=(2, 2), pool_mode='MAX')(padded)

print(f"Input shape: {input_tensor.shape}")
print(f"Padded shape: {padded.shape}")
print(f"Pooled shape: {pooled.shape}")
```

Slide 13: Exploring the Effects of Different Padding and Pooling Configurations

Let's examine how different combinations of padding and pooling affect the output of a convolutional layer. We'll create a simple function to apply convolution with various padding and pooling settings.

```python
import tensorflow as tf
import matplotlib.pyplot as plt

def apply_conv_and_pool(input_tensor, padding='SAME', pool_size=(2, 2), pool_mode='MAX'):
    conv = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding=padding)(input_tensor)
    if pool_mode == 'MAX':
        pooled = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(conv)
    else:
        pooled = tf.keras.layers.AveragePooling2D(pool_size=pool_size)(conv)
    return conv, pooled

# Generate a sample input
input_tensor = tf.random.normal([1, 28, 28, 1])

# Apply different configurations
configs = [
    ('SAME', (2, 2), 'MAX'),
    ('VALID', (2, 2), 'MAX'),
    ('SAME', (3, 3), 'AVG'),
    ('VALID', (3, 3), 'AVG')
]

fig, axs = plt.subplots(len(configs), 3, figsize=(15, 5*len(configs)))

for i, (padding, pool_size, pool_mode) in enumerate(configs):
    conv, pooled = apply_conv_and_pool(input_tensor, padding, pool_size, pool_mode)
    
    axs[i, 0].imshow(tf.squeeze(input_tensor), cmap='viridis')
    axs[i, 0].set_title('Input')
    axs[i, 1].imshow(tf.squeeze(conv[:, :, :, 0]), cmap='viridis')
    axs[i, 1].set_title(f'Conv ({padding} padding)')
    axs[i, 2].imshow(tf.squeeze(pooled[:, :, :, 0]), cmap='viridis')
    axs[i, 2].set_title(f'Pooled ({pool_mode}, {pool_size})')

    for ax in axs[i]:
        ax.axis('off')

plt.tight_layout()
plt.show()
```

Slide 14: Implementing Dilated Convolutions

Dilated convolutions, also known as atrous convolutions, introduce another parameter called the dilation rate. This allows the kernel to cover a larger receptive field without increasing the number of parameters.

```python
def dilated_conv2d(input_tensor, filters, kernel_size, dilation_rate):
    return tf.keras.layers.Conv2D(
        filters, kernel_size, padding='same', dilation_rate=dilation_rate,
        activation='relu')(input_tensor)

# Create a sample input
input_tensor = tf.random.normal([1, 28, 28, 1])

# Apply dilated convolutions with different dilation rates
dilation_rates = [1, 2, 4]
outputs = [input_tensor] + [dilated_conv2d(input_tensor, 32, (3, 3), rate) 
                            for rate in dilation_rates]

fig, axs = plt.subplots(1, len(outputs), figsize=(20, 4))
for i, output in enumerate(outputs):
    axs[i].imshow(tf.squeeze(output[:, :, :, 0]), cmap='viridis')
    axs[i].set_title(f'Dilation Rate: {dilation_rates[i-1] if i > 0 else "Input"}')
    axs[i].axis('off')

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into the topics of padding, strides, and pooling in deep learning, here are some valuable resources:

1. "A guide to convolution arithmetic for deep learning" by Vincent Dumoulin and Francesco Visin ArXiv URL: [https://arxiv.org/abs/1603.07285](https://arxiv.org/abs/1603.07285)
2. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" by Kaiming He et al. ArXiv URL: [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852)
3. "Rethinking Atrous Convolution for Semantic Image Segmentation" by Liang-Chieh Chen et al. ArXiv URL: [https://arxiv.org/abs/1706.05587](https://arxiv.org/abs/1706.05587)

These papers provide in-depth discussions on convolution operations, activation functions, and advanced convolution techniques that build upon the concepts covered in this presentation.

