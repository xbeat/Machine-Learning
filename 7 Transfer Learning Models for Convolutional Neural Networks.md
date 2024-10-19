## 7 Transfer Learning Models for Convolutional Neural Networks

Slide 1: Introduction to Transfer Learning in CNNs

Transfer learning is a powerful technique in deep learning where knowledge gained from training on one task is applied to a different but related task. In the context of Convolutional Neural Networks (CNNs), transfer learning allows us to leverage pre-trained models on large datasets to improve performance on smaller, specific tasks. This approach is particularly useful when we have limited data or computational resources.

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16, MobileNet, DenseNet121, InceptionV3, ResNet50, EfficientNetB0

# Load pre-trained models
vgg16 = VGG16(weights='imagenet', include_top=False)
mobilenet = MobileNet(weights='imagenet', include_top=False)
densenet = DenseNet121(weights='imagenet', include_top=False)
inception = InceptionV3(weights='imagenet', include_top=False)
resnet = ResNet50(weights='imagenet', include_top=False)
efficientnet = EfficientNetB0(weights='imagenet', include_top=False)

# Freeze the pre-trained layers
for model in [vgg16, mobilenet, densenet, inception, resnet, efficientnet]:
    for layer in model.layers:
        layer.trainable = False

# Add custom layers for transfer learning
def add_custom_layers(base_model):
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    return tf.keras.Model(inputs=base_model.input, outputs=outputs)

# Create transfer learning models
transfer_models = [add_custom_layers(model) for model in [vgg16, mobilenet, densenet, inception, resnet, efficientnet]]
```

Slide 2: VGG16 Architecture

VGG16 is a deep CNN architecture developed by the Visual Geometry Group at Oxford. It consists of 16 layers, including 13 convolutional layers and 3 fully connected layers. The model is known for its simplicity and effectiveness in image classification tasks.

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for transfer learning
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

# Create the transfer learning model
transfer_model = Model(inputs=base_model.input, outputs=output)

# Compile the model
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
transfer_model.summary()
```

Slide 3: MobileNet Architecture

MobileNet is a lightweight CNN architecture designed for mobile and embedded vision applications. It uses depthwise separable convolutions to reduce the number of parameters and computational cost while maintaining good performance.

```python
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained MobileNet model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for transfer learning
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

# Create the transfer learning model
transfer_model = Model(inputs=base_model.input, outputs=output)

# Compile the model
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
transfer_model.summary()
```

Slide 4: DenseNet Architecture

DenseNet is a CNN architecture that introduces dense connections between layers. Each layer receives inputs from all preceding layers, promoting feature reuse and improving gradient flow. This design allows for deeper networks with fewer parameters.

```python
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained DenseNet121 model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for transfer learning
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

# Create the transfer learning model
transfer_model = Model(inputs=base_model.input, outputs=output)

# Compile the model
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
transfer_model.summary()
```

Slide 5: Inception Architecture

The Inception architecture, also known as GoogLeNet, introduces the concept of inception modules. These modules use multiple convolution filters of different sizes in parallel, allowing the network to capture features at various scales simultaneously.

```python
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for transfer learning
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

# Create the transfer learning model
transfer_model = Model(inputs=base_model.input, outputs=output)

# Compile the model
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
transfer_model.summary()
```

Slide 6: ResNet Architecture

ResNet (Residual Network) introduces skip connections or shortcut connections that allow the gradient to flow directly through the network. This design helps mitigate the vanishing gradient problem in very deep networks, enabling the training of networks with hundreds of layers.

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for transfer learning
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

# Create the transfer learning model
transfer_model = Model(inputs=base_model.input, outputs=output)

# Compile the model
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
transfer_model.summary()
```

Slide 7: EfficientNet Architecture

EfficientNet is a family of models that achieve state-of-the-art accuracy with fewer parameters and FLOPS. It uses a compound scaling method that uniformly scales network width, depth, and resolution with a fixed set of scaling coefficients.

```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for transfer learning
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

# Create the transfer learning model
transfer_model = Model(inputs=base_model.input, outputs=output)

# Compile the model
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
transfer_model.summary()
```

Slide 8: NASNet Architecture

NASNet (Neural Architecture Search Network) is an architecture discovered through automated machine learning techniques. It uses a search space of modular components to find optimal network architectures for specific tasks.

```python
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained NASNetMobile model
base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for transfer learning
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

# Create the transfer learning model
transfer_model = Model(inputs=base_model.input, outputs=output)

# Compile the model
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
transfer_model.summary()
```

Slide 9: Transfer Learning Process

The transfer learning process involves several steps: loading a pre-trained model, freezing its layers, adding custom layers for the new task, and fine-tuning the model on the new dataset. This approach leverages the feature extraction capabilities of the pre-trained model while adapting it to the specific requirements of the new task.

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Step 1: Load pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Step 2: Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Step 3: Add custom layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

# Step 4: Create new model
transfer_model = Model(inputs=base_model.input, outputs=output)

# Step 5: Compile the model
transfer_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train the model (assuming you have your dataset ready)
# history = transfer_model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Step 7: Fine-tuning (optional)
# Unfreeze some layers of the base model
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile the model with a lower learning rate
transfer_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training
# history = transfer_model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
```

Slide 10: Real-Life Example: Plant Disease Classification

Transfer learning can be applied to various real-world problems. One practical application is plant disease classification, where a model trained on a large dataset of plant images can be fine-tuned to identify specific diseases in a particular crop.

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(5, activation='softmax')(x)  # Assuming 5 classes of plant diseases

# Create new model
plant_disease_model = Model(inputs=base_model.input, outputs=output)

# Compile the model
plant_disease_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Load and preprocess the dataset
train_generator = train_datagen.flow_from_directory(
    'path/to/train/data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Train the model
history = plant_disease_model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50)

# Evaluate the model
test_loss, test_accuracy = plant_disease_model.evaluate(test_generator, steps=50)
print(f"Test accuracy: {test_accuracy:.2f}")
```

Slide 11: Real-Life Example: Wildlife Species Identification

Another practical application of transfer learning in CNNs is wildlife species identification. This can be used in conservation efforts, ecological research, or even in applications for nature enthusiasts.

```python
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(50, activation='softmax')(x)  # Assuming 50 different wildlife species

# Create new model
wildlife_model = Model(inputs=base_model.input, outputs=output)

# Compile the model
wildlife_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Load and preprocess the dataset
train_generator = train_datagen.flow_from_directory(
    'path/to/wildlife/train/data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Train the model
history = wildlife_model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50)

# Evaluate the model
test_loss, test_accuracy = wildlife_model.evaluate(test_generator, steps=50)
print(f"Test accuracy: {test_accuracy:.2f}")
```

Slide 12: Fine-tuning Transfer Learning Models

Fine-tuning involves unfreezing some layers of the pre-trained model and training them on the new dataset. This process allows the model to adapt its learned features to the specific characteristics of the new task.

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Initially freeze all layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

# Create new model
model = Model(inputs=base_model.input, outputs=output)

# Initial training
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

# Fine-tuning: unfreeze the last few layers
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training
# model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
```

Slide 13: Comparing Transfer Learning Models

When applying transfer learning, it's important to compare different pre-trained models to find the best fit for your specific task. This code demonstrates how to create and evaluate multiple transfer learning models.

```python
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_transfer_model(base_model, num_classes):
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=output)

# Load pre-trained models
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
inception_v3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Create transfer learning models
models = {
    'VGG16': create_transfer_model(vgg16, 10),
    'ResNet50': create_transfer_model(resnet50, 10),
    'InceptionV3': create_transfer_model(inception_v3, 10)
}

# Compile and train each model
for name, model in models.items():
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    # Evaluate model
    # test_loss, test_accuracy = model.evaluate(x_test, y_test)
    # print(f"{name} Test accuracy: {test_accuracy:.2f}")
```

Slide 14: Visualizing Feature Maps

Understanding how transfer learning models process images can be enhanced by visualizing the feature maps at different layers. This technique helps in interpreting what the model has learned and how it represents features.

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Load and preprocess an image
img_path = 'path/to/your/image.jpg'
img = load_img(img_path, target_size=(224, 224))
img_tensor = img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

# Get feature maps for first convolutional layer
layer_name = 'block1_conv1'
layer_output = model.get_layer(layer_name).output
activation_model = Model(inputs=model.input, outputs=layer_output)
activations = activation_model.predict(img_tensor)

# Plot feature maps
fig, axes = plt.subplots(8, 8, figsize=(16, 16))
for i, ax in enumerate(axes.flat):
    if i < activations.shape[-1]:
        ax.imshow(activations[0, :, :, i], cmap='viridis')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into transfer learning and CNNs, here are some valuable resources:

1.  "A Survey on Transfer Learning" by Sinno Jialin Pan and Qiang Yang. Available at: [https://arxiv.org/abs/1911.02685](https://arxiv.org/abs/1911.02685)
2.  "Deep Transfer Learning for Image Classification" by Maxime Oquab et al. Available at: [https://arxiv.org/abs/1403.6382](https://arxiv.org/abs/1403.6382)
3.  "How transferable are features in deep neural networks?" by Jason Yosinski et al. Available at: [https://arxiv.org/abs/1411.1792](https://arxiv.org/abs/1411.1792)
4.  "Visualizing and Understanding Convolutional Networks" by Matthew D. Zeiler and Rob Fergus. Available at: [https://arxiv.org/abs/1311.2901](https://arxiv.org/abs/1311.2901)

These papers provide in-depth insights into transfer learning techniques, their applications in image classification, and methods for visualizing and interpreting CNNs.

