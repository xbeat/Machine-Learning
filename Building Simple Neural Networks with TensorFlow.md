## Building Simple Neural Networks with TensorFlow
Slide 1: Building Simple Neural Networks with TensorFlow

TensorFlow is a powerful open-source library for machine learning and deep learning. This slideshow will guide you through the process of creating basic neural networks using TensorFlow and Python, providing practical examples and code snippets along the way.

```python
import tensorflow as tf
import numpy as np

print(f"TensorFlow version: {tf.__version__}")
```

Slide 2: Understanding Neural Networks

Neural networks are computational models inspired by the human brain. They consist of interconnected nodes (neurons) organized in layers. These networks learn from data to perform tasks like classification and regression.

```python
# Simple neuron representation
def neuron(inputs, weights, bias):
    return tf.nn.relu(tf.reduce_sum(inputs * weights) + bias)

inputs = tf.constant([1.0, 2.0, 3.0])
weights = tf.Variable([0.1, 0.2, 0.3])
bias = tf.Variable(0.1)

output = neuron(inputs, weights, bias)
print(f"Neuron output: {output.numpy()}")
```

Slide 3: Setting Up the Environment

Before building neural networks, we need to prepare our development environment. This includes importing necessary libraries and setting up our data.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Generate sample data
X = np.random.rand(1000, 10)
y = np.sum(X, axis=1) > 5
```

Slide 4: Creating a Simple Neural Network

Let's build a basic neural network with one hidden layer to classify our sample data.

```python
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
```

Slide 5: Training the Neural Network

Now that we have created our model, let's train it on our sample data.

```python
history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

Slide 6: Evaluating the Model

After training, we need to evaluate our model's performance on unseen data.

```python
# Generate test data
X_test = np.random.rand(200, 10)
y_test = np.sum(X_test, axis=1) > 5

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Make predictions
predictions = model.predict(X_test)
print(f"First 5 predictions: {predictions[:5].flatten()}")
```

Slide 7: Real-Life Example: Image Classification

Let's use a pre-trained model to classify images, a common application of neural networks.

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load and preprocess an image
img_path = 'path_to_your_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make prediction
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
```

Slide 8: Convolutional Neural Networks (CNNs)

CNNs are specialized neural networks designed for processing grid-like data, such as images. They use convolutional layers to automatically learn spatial hierarchies of features.

```python
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

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
```

Slide 9: Transfer Learning

Transfer learning allows us to leverage pre-trained models for new tasks, saving time and computational resources.

```python
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
```

Slide 10: Real-Life Example: Sentiment Analysis

Neural networks can also be applied to natural language processing tasks like sentiment analysis.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
texts = ['I love this movie', 'This film is terrible', 'Great acting and plot']
labels = [1, 0, 1]  # 1 for positive, 0 for negative

# Tokenize the texts
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')

# Create and train the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=10),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded, labels, epochs=50, verbose=0)

# Test the model
test_text = ['This movie is amazing']
test_seq = tokenizer.texts_to_sequences(test_text)
test_padded = pad_sequences(test_seq, maxlen=10, padding='post', truncating='post')
prediction = model.predict(test_padded)
print(f"Sentiment prediction: {'Positive' if prediction > 0.5 else 'Negative'}")
```

Slide 11: Regularization Techniques

Regularization helps prevent overfitting in neural networks. Common techniques include L1/L2 regularization and dropout.

```python
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,), 
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu', 
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

Slide 12: Hyperparameter Tuning

Optimizing hyperparameters is crucial for achieving the best performance from your neural network.

```python
import keras_tuner as kt

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(layers.Dense(hp.Int('units', min_value=32, max_value=512, step=32), 
                           activation='relu', input_shape=(10,)))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model

tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=10, factor=3, directory='my_dir', project_name='intro_to_kt')

tuner.search(X, y, epochs=50, validation_split=0.2)
best_model = tuner.get_best_models(num_models=1)[0]
```

Slide 13: Saving and Loading Models

Once you've trained a successful model, it's important to save it for future use.

```python
# Save the model
model.save('my_model.h5')

# Load the saved model
loaded_model = tf.keras.models.load_model('my_model.h5')

# Make predictions with the loaded model
new_data = np.random.rand(5, 10)
predictions = loaded_model.predict(new_data)
print(f"Predictions: {predictions.flatten()}")
```

Slide 14: Visualizing Neural Networks

Visualizing the structure and activations of neural networks can provide insights into their behavior.

```python
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# Plot model architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

# Visualize layer activations
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(X[:1])
for i, activation in enumerate(activations):
    plt.subplot(1, len(activations), i+1)
    plt.title(f"Layer {i+1}")
    plt.imshow(activation[0, :, :, 0], cmap='viridis')
plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For further learning about neural networks and TensorFlow, consider exploring these resources:

1. TensorFlow documentation: [https://www.tensorflow.org/guide](https://www.tensorflow.org/guide)
2. "Neural Networks and Deep Learning" by Michael Nielsen: [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)
3. ArXiv paper on deep learning: [https://arxiv.org/abs/1404.7828](https://arxiv.org/abs/1404.7828) (LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.)
4. CS231n: Convolutional Neural Networks for Visual Recognition: [http://cs231n.stanford.edu/](http://cs231n.stanford.edu/)
5. Fast.ai Practical Deep Learning for Coders: [https://course.fast.ai/](https://course.fast.ai/)

Remember to stay updated with the latest developments in the field, as neural network techniques and best practices are continually evolving.

