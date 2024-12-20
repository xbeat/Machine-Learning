## Deep Learning Essentials with Python
Slide 1: Introduction to Deep Learning

Deep Learning is a subset of machine learning that uses artificial neural networks to model and solve complex problems. It has revolutionized various fields, from computer vision to natural language processing.

```python
import tensorflow as tf

# Create a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
```

Slide 2: The Power of Representation Learning

Deep Learning excels at automatically learning hierarchical representations from raw data, eliminating the need for manual feature engineering.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate non-linear data
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a deep learning model
model = Sequential([
    Dense(16, activation='relu', input_shape=(2,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)

# Visualize the decision boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 100),
                     np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
plt.title("Decision Boundary of Deep Learning Model")
plt.show()
```

Slide 3: Convolutional Neural Networks (CNNs)

CNNs have revolutionized computer vision tasks by efficiently processing grid-like data such as images. They use convolutional layers to automatically learn spatial hierarchies of features.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Create a simple CNN for image classification
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

# Print model summary
model.summary()
```

Slide 4: Recurrent Neural Networks (RNNs)

RNNs are designed to work with sequential data, making them ideal for tasks like natural language processing and time series analysis. They can capture long-term dependencies in data.

```python
import tensorflow as tf
from tensorflow.keras import layers

# Create a simple RNN for sequence classification
model = tf.keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=32),
    layers.LSTM(32),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()
```

Slide 5: Transfer Learning

Transfer learning allows us to leverage pre-trained models on large datasets and fine-tune them for specific tasks, significantly reducing training time and data requirements.

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add custom layers for fine-tuning
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()
```

Slide 6: Generative Adversarial Networks (GANs)

GANs consist of two neural networks, a generator and a discriminator, that compete against each other. This architecture has led to remarkable advancements in generating realistic images, text, and other types of data.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the generator
def make_generator_model():
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

# Define the discriminator
def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# Create the GAN
generator = make_generator_model()
discriminator = make_discriminator_model()

# Print model summaries
print("Generator Summary:")
generator.summary()
print("\nDiscriminator Summary:")
discriminator.summary()
```

Slide 7: Attention Mechanisms and Transformers

Attention mechanisms have revolutionized sequence-to-sequence models, leading to the development of Transformers. These models have achieved state-of-the-art results in various natural language processing tasks.

```python
import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output, attention_weights

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights

# Example usage
d_model = 512
num_heads = 8

mha = MultiHeadAttention(d_model, num_heads)
y = tf.random.uniform((1, 60, d_model))
output, _ = mha(y, k=y, q=y, mask=None)
print(output.shape)
```

Slide 8: Reinforcement Learning with Deep Neural Networks

Deep Reinforcement Learning combines deep neural networks with reinforcement learning algorithms, enabling agents to learn complex behaviors in high-dimensional state spaces.

```python
import tensorflow as tf
import numpy as np

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Example usage
state_size = 4
action_size = 2
agent = DQNAgent(state_size, action_size)
print(agent.model.summary())
```

Slide 9: Autoencoders for Dimensionality Reduction

Autoencoders are neural networks that learn to compress data into a lower-dimensional representation and then reconstruct it. They are useful for dimensionality reduction, denoising, and feature learning.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the encoder
encoder = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, 3, activation='relu', padding='same', strides=2),
    layers.Conv2D(8, 3, activation='relu', padding='same', strides=2),
    layers.Flatten(),
    layers.Dense(10)
])

# Define the decoder
decoder = models.Sequential([
    layers.Input(shape=(10,)),
    layers.Dense(7 * 7 * 8),
    layers.Reshape((7, 7, 8)),
    layers.Conv2DTranspose(8, 3, activation='relu', padding='same', strides=2),
    layers.Conv2DTranspose(16, 3, activation='relu', padding='same', strides=2),
    layers.Conv2D(1, 3, activation='sigmoid', padding='same')
])

# Create the autoencoder
autoencoder = models.Sequential([encoder, decoder])

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Print model summary
autoencoder.summary()
```

Slide 10: One-Shot and Few-Shot Learning

One-shot and few-shot learning techniques aim to learn from very few examples, mimicking human-like learning abilities. These approaches are crucial when labeled data is scarce or expensive to obtain.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def create_prototypical_network(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(64, 3, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
    ])
    return model

def euclidean_distance(x, y):
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=-1))

def proto_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)

# Example usage
input_shape = (28, 28, 1)  # For MNIST dataset
num_classes = 10
proto_net = create_prototypical_network(input_shape, num_classes)
proto_net.compile(optimizer='adam', loss=proto_loss)
print(proto_net.summary())
```

Slide 11: Explainable AI in Deep Learning

As deep learning models become more complex, understanding their decision-making process becomes crucial. Explainable AI techniques help interpret model predictions and build trust in AI systems.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def grad_cam(model, img_array, layer_name, class_index):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Assume we have a pre-trained model and an input image
model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
img_path = 'path/to/your/image.jpg'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

# Generate class activation map
heatmap = grad_cam(model, img_array, 'block5_conv3', 285)  # 285 is the index for 'Egyptian cat' in ImageNet

# Display the results
plt.matshow(heatmap)
plt.show()
```

Slide 12: Deep Learning for Natural Language Processing

Deep learning has revolutionized natural language processing tasks such as machine translation, sentiment analysis, and text generation. Models like BERT and GPT have achieved remarkable performance across various language tasks.

```python
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name)

# Example text for sentiment analysis
text = "I love this movie! It's amazing."

# Tokenize and encode the text
inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=128)

# Make prediction
outputs = model(inputs)
logits = outputs.logits
predicted_class = tf.argmax(logits, axis=1).numpy()[0]

print(f"Predicted class: {predicted_class}")
print(f"Logits: {logits.numpy()}")
```

Slide 13: Deep Learning in Computer Vision

Deep learning has transformed computer vision tasks, enabling applications like object detection, image segmentation, and facial recognition with unprecedented accuracy.

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load and preprocess an image
img_path = 'path/to/your/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# Make prediction
preds = model.predict(x)
decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]

# Print results
for i, (imagenet_id, label, score) in enumerate(decoded_preds):
    print(f"{i + 1}: {label} ({score:.2f})")
```

Slide 14: Real-Life Example: Medical Image Analysis

Deep learning has significantly improved medical image analysis, aiding in the early detection and diagnosis of diseases. Here's a simplified example of using a CNN for chest X-ray classification.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_chest_xray_model(input_shape, num_classes):
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
input_shape = (256, 256, 1)  # Grayscale chest X-ray images
num_classes = 3  # Normal, Pneumonia, COVID-19
model = create_chest_xray_model(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
```

Slide 15: Real-Life Example: Autonomous Driving

Deep learning plays a crucial role in autonomous driving systems, handling tasks such as object detection, lane detection, and path planning. Here's a simplified example of a lane detection model.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_lane_detection_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(4)  # Output: [x1, y1, x2, y2] for lane line endpoints
    ])
    return model

# Example usage
input_shape = (720, 1280, 3)  # HD camera input
model = create_lane_detection_model(input_shape)
model.compile(optimizer='adam', loss='mse')

print(model.summary())
```

Slide 16: Additional Resources

For further exploration of deep learning concepts and applications, consider the following resources:

1. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press, 2016)
2. "Attention Is All You Need" by Vaswani et al. (2017) - ArXiv:1706.03762
3. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) - ArXiv:1810.04805
4. "Deep Residual Learning for Image Recognition" by He et al. (2015) - ArXiv:1512.03385
5. TensorFlow and PyTorch documentation for practical implementation guides

These resources provide in-depth explanations and cutting-edge research in the field of deep learning.

