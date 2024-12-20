## Model Capacity in Machine Learning with Python
Slide 1: Understanding Model Capacity in Machine Learning

Model capacity refers to a model's ability to capture complex patterns in data. It's influenced by factors like the number of parameters, model architecture, and training data complexity. Let's explore this concept with practical examples.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and plot models with different capacities
degrees = [1, 3, 15]
plt.figure(figsize=(14, 4))

for i, degree in enumerate(degrees):
    ax = plt.subplot(1, 3, i + 1)
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X_train)
    
    model = LinearRegression()
    model.fit(X_poly, y_train)
    
    X_plot = np.linspace(0, 5, 100)[:, np.newaxis]
    X_plot_poly = poly_features.transform(X_plot)
    y_plot = model.predict(X_plot_poly)
    
    plt.scatter(X_train, y_train, color='r', s=10, alpha=0.7)
    plt.plot(X_plot, y_plot, color='b')
    plt.ylim(-1.5, 1.5)
    plt.title(f"Degree {degree}")

plt.tight_layout()
plt.show()
```

This code demonstrates how model capacity affects fitting to data using polynomial regression of different degrees.

Slide 2: Low Capacity Models

Low capacity models have fewer parameters and simpler architectures. They are less flexible and may underfit complex data patterns. These models are suitable for simple, linear relationships.

```python
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.normal(0, 1, (100, 1))

# Create and fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Plot the results
plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.title('Linear Regression (Low Capacity Model)')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

print(f"Model coefficients: {model.coef_[0][0]:.2f}")
print(f"Model intercept: {model.intercept_[0]:.2f}")
```

This example shows a linear regression model, which has low capacity and works well for simple linear relationships.

Slide 3: High Capacity Models

High capacity models have more parameters and complex architectures. They can capture intricate patterns but may overfit noise in the training data. Neural networks are examples of high capacity models.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.1, (100, 1))

# Create a high-capacity neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X, y, epochs=1000, verbose=0)

# Plot the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.title('Neural Network (High Capacity Model)')
plt.xlabel('X')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.yscale('log')
plt.show()
```

This example demonstrates a high-capacity neural network that can capture complex, non-linear patterns in the data.

Slide 4: Balancing Model Capacity

Finding the right balance in model capacity is crucial. Too low capacity leads to underfitting, while too high capacity can cause overfitting. Cross-validation helps in selecting the appropriate model capacity.

```python
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
X = np.sort(np.random.rand(40, 1) * 5)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Test different polynomial degrees
degrees = range(1, 20)
cv_scores = []

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(degrees, cv_scores, marker='o')
plt.title('Cross-Validation Scores for Different Model Capacities')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.show()

best_degree = degrees[np.argmin(cv_scores)]
print(f"Best polynomial degree: {best_degree}")
```

This code demonstrates how to use cross-validation to select the optimal model capacity for polynomial regression.

Slide 5: Real-Life Example: Image Classification

In image classification tasks, model capacity plays a crucial role. Let's compare a simple CNN with a more complex one for MNIST digit classification.

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# Simple CNN (lower capacity)
simple_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(10, activation='softmax')
])

# Complex CNN (higher capacity)
complex_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train both models
simple_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
complex_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

simple_history = simple_model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=0)
complex_history = complex_model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=0)

# Evaluate models
simple_test_loss, simple_test_acc = simple_model.evaluate(x_test, y_test, verbose=0)
complex_test_loss, complex_test_acc = complex_model.evaluate(x_test, y_test, verbose=0)

print(f"Simple model test accuracy: {simple_test_acc:.4f}")
print(f"Complex model test accuracy: {complex_test_acc:.4f}")
```

This example compares a simple CNN with a more complex one, demonstrating how increased model capacity can improve performance on complex tasks like image classification.

Slide 6: Regularization Techniques

Regularization helps control model capacity by adding constraints to prevent overfitting. Common techniques include L1, L2 regularization, and dropout.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.1, (200, 1))

# Create models with different regularization techniques
models = {
    'No Regularization': Sequential([
        Dense(64, activation='relu', input_shape=(1,)),
        Dense(64, activation='relu'),
        Dense(1)
    ]),
    'L2 Regularization': Sequential([
        Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_shape=(1,)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1)
    ]),
    'Dropout': Sequential([
        Dense(64, activation='relu', input_shape=(1,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])
}

# Train and plot results
plt.figure(figsize=(15, 5))
for i, (name, model) in enumerate(models.items()):
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=500, verbose=0)
    
    plt.subplot(1, 3, i+1)
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X, model.predict(X), color='red')
    plt.title(name)
    plt.xlabel('X')
    plt.ylabel('y')

plt.tight_layout()
plt.show()
```

This code demonstrates how different regularization techniques affect model capacity and prevent overfitting.

Slide 7: Transfer Learning and Model Capacity

Transfer learning allows us to leverage pre-trained models, effectively increasing model capacity without training from scratch. This is particularly useful for tasks with limited data.

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Create new model on top
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

print("\nTotal trainable parameters:", sum(p.numel() for p in model.trainable_weights))
print("Total non-trainable parameters:", sum(p.numel() for p in model.non_trainable_weights))
```

This example shows how to use a pre-trained VGG16 model for transfer learning, effectively increasing model capacity for a new task.

Slide 8: Model Pruning

Model pruning is a technique to reduce model capacity by removing unnecessary weights or neurons. This can lead to smaller, more efficient models without significant loss in performance.

```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.1, (200, 1))

# Create a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Train the model
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

# Apply pruning
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=0.5,
        begin_step=0, end_step=1000
    )
}

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
pruned_model.compile(optimizer='adam', loss='mse')
pruned_model.fit(X, y, epochs=100, callbacks=[tfmot.sparsity.keras.UpdatePruningStep()], verbose=0)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.5)
plt.plot(X, model.predict(X), color='red', label='Original')
plt.plot(X, pruned_model.predict(X), color='green', label='Pruned')
plt.title('Model Predictions')
plt.legend()

plt.subplot(1, 2, 2)
original_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
pruned_params = np.sum([np.count_nonzero(v) for v in pruned_model.get_weights()])
plt.bar(['Original', 'Pruned'], [original_params, pruned_params])
plt.title('Number of Parameters')
plt.ylabel('Parameter Count')

plt.tight_layout()
plt.show()

print(f"Original model parameters: {original_params}")
print(f"Pruned model parameters: {pruned_params}")
print(f"Reduction: {(1 - pruned_params / original_params) * 100:.2f}%")
```

This example demonstrates how to apply pruning to a neural network, reducing its capacity while maintaining performance.

Slide 9: Model Distillation

Model distillation transfers knowledge from a large, high-capacity model (teacher) to a smaller, low-capacity model (student). This creates more efficient models while retaining performance.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.1, (200, 1))

# Create and train teacher model
teacher = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
teacher.compile(optimizer='adam', loss='mse')
teacher.fit(X, y, epochs=200, verbose=0)

# Create student model
student = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

# Distillation
def distillation_loss(y_true, y_pred):
    return tf.keras.losses.MSE(teacher.predict(X), y_pred)

student.compile(optimizer='adam', loss=distillation_loss)
student.fit(X, y, epochs=200, verbose=0)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, teacher.predict(X), color='red', label='Teacher')
plt.plot(X, student.predict(X), color='green', label='Student')
plt.legend()
plt.title('Model Distillation')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

print(f"Teacher parameters: {teacher.count_params()}")
print(f"Student parameters: {student.count_params()}")
```

This example demonstrates how to perform model distillation, transferring knowledge from a larger teacher model to a smaller student model.

Slide 10: Dynamic Model Capacity

Dynamic model capacity allows models to adapt their complexity based on the input. This approach can optimize performance and computational resources.

```python
import tensorflow as tf
import numpy as np

class DynamicCapacityLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = tf.keras.layers.Dense(units)

    def call(self, inputs):
        # Compute complexity score
        complexity = tf.reduce_mean(tf.abs(inputs))
        
        # Adjust capacity based on complexity
        adjusted_units = tf.cast(tf.maximum(1, tf.minimum(self.units, tf.cast(complexity * self.units, tf.int32))), tf.int32)
        
        # Apply dense layer with adjusted capacity
        outputs = self.dense(inputs)
        return outputs[:, :adjusted_units]

# Create a model with dynamic capacity
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    DynamicCapacityLayer(64),
    tf.keras.layers.Dense(1)
])

# Example usage
x = np.random.randn(100, 10)
y = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
```

This example shows a custom layer that dynamically adjusts its capacity based on input complexity, demonstrating the concept of dynamic model capacity.

Slide 11: Real-Life Example: Natural Language Processing

In NLP tasks, model capacity significantly impacts performance. Let's compare simple and complex models for sentiment analysis.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
texts = [
    "I love this movie", "This film is terrible",
    "Great acting and plot", "Boring and predictable"
]
labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# Tokenize texts
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=10)

# Simple model (low capacity)
simple_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=10),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Complex model (high capacity)
complex_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 32, input_length=10),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile and train models
simple_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
complex_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

simple_history = simple_model.fit(padded_sequences, labels, epochs=50, verbose=0)
complex_history = complex_model.fit(padded_sequences, labels, epochs=50, verbose=0)

print("Simple model accuracy:", simple_history.history['accuracy'][-1])
print("Complex model accuracy:", complex_history.history['accuracy'][-1])
```

This example compares a simple and complex model for sentiment analysis, illustrating how increased capacity can potentially improve performance in NLP tasks.

Slide 12: Hyperparameter Tuning and Model Capacity

Hyperparameter tuning is crucial for optimizing model capacity. Let's explore how different hyperparameters affect model performance.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Define hyperparameter grid
param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (100,), (10, 10), (50, 50)],
    'alpha': [0.0001, 0.001, 0.01, 0.1]
}

# Create and train model
model = MLPRegressor(max_iter=1000)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Plot results
plt.figure(figsize=(12, 6))
results = -grid_search.cv_results_['mean_test_score'].reshape(len(param_grid['hidden_layer_sizes']), len(param_grid['alpha']))

plt.imshow(results, interpolation='nearest', cmap='viridis')
plt.colorbar()
plt.xticks(np.arange(len(param_grid['alpha'])), param_grid['alpha'])
plt.yticks(np.arange(len(param_grid['hidden_layer_sizes'])), [str(hl) for hl in param_grid['hidden_layer_sizes']])
plt.xlabel('Alpha (regularization)')
plt.ylabel('Hidden Layer Sizes')
plt.title('Model Performance (MSE) for Different Hyperparameters')

for i in range(results.shape[0]):
    for j in range(results.shape[1]):
        plt.text(j, i, f"{results[i, j]:.3f}", ha="center", va="center", color="w")

plt.tight_layout()
plt.show()

print("Best parameters:", grid_search.best_params_)
print("Best score:", -grid_search.best_score_)
```

This example demonstrates how hyperparameter tuning can help find the optimal model capacity for a given task.

Slide 13: Model Compression Techniques

Model compression reduces model size and computational requirements while maintaining performance. Techniques include quantization, pruning, and low-rank factorization.

```python
import tensorflow as tf
import numpy as np

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Generate sample data
x_train = np.random.random((1000, 10))
y_train = np.random.random((1000, 1))

# Train the model
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=5, verbose=0)

# Function to print model size
def get_model_size(model):
    return sum(tf.keras.backend.count_params(w) for w in model.trainable_weights) * 4 / 1024  # Size in KB

print("Original model size:", get_model_size(model), "KB")

# Quantization-aware training
quantize_model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(10,)),
    tf.quantization.quantize_layer(model.layers[0]),
    tf.quantization.quantize_layer(model.layers[1]),
    tf.quantization.quantize_layer(model.layers[2])
])

quantize_model.compile(optimizer='adam', loss='mse')
quantize_model.fit(x_train, y_train, epochs=5, verbose=0)

print("Quantized model size:", get_model_size(quantize_model), "KB")

# Pruning
pruning_params = {
    'pruning_schedule': tf.keras.experimental.PruningSchedule(
        pruning_rate=0.2,
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        frequency=1
    )
}

pruned_model = tf.keras.models.clone_model(
    model,
    clone_function=lambda layer: tf.keras.layers.experimental.PruneLowMagnitude(
        layer, **pruning_params
    ) if isinstance(layer, tf.keras.layers.Dense) else layer
)

pruned_model.compile(optimizer='adam', loss='mse')
pruned_model.fit(x_train, y_train, epochs=5, callbacks=[tf.keras.callbacks.UpdatePruningStep()], verbose=0)

print("Pruned model size:", get_model_size(pruned_model), "KB")
```

This example demonstrates quantization-aware training and pruning as model compression techniques to reduce model size while maintaining performance.

Slide 14: Additional Resources

For more in-depth information on model capacity and related topics in machine learning, consider exploring the following resources:

1. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press, 2016) Available online: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
2. "Neural Networks and Deep Learning" by Michael Nielsen Available online: [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)
3. ArXiv papers:
   * "Rethinking Model Size for Efficient Training and Inference of Transformers" (2020) ArXiv: [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
   * "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" (2019) ArXiv: [https://arxiv.org/abs/1803.03635](https://arxiv.org/abs/1803.03635)
4. TensorFlow Model Optimization Toolkit documentation: [https://www.tensorflow.org/model\_optimization](https://www.tensorflow.org/model_optimization)
5. PyTorch Model Compression documentation: [https://pytorch.org/tutorials/intermediate/pruning\_tutorial.html](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)

These resources provide a mix of theoretical foundations and practical implementations to deepen your understanding of model capacity and related concepts in machine learning.

