## Iterating Neural Networks with Python
Slide 1: What is an Iteration in Neural Networks?

An iteration in neural networks refers to one complete pass through the entire training dataset, including both forward and backward propagation. It's a fundamental concept in the training process, where the network learns from data by repeatedly adjusting its weights and biases.

```python
import numpy as np

# Simple neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
    
    def forward(self, X):
        self.hidden = np.dot(X, self.weights1)
        self.output = np.dot(self.hidden, self.weights2)
        return self.output

# Initialize network
nn = NeuralNetwork(2, 3, 1)

# One iteration (forward pass only)
X = np.array([[0, 1]])
output = nn.forward(X)
print(f"Output after one iteration: {output}")
```

Slide 2: Components of an Iteration

An iteration consists of several key components: forward propagation, loss calculation, backward propagation, and parameter updates. These steps work together to gradually improve the network's performance on the given task.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2*(self.y - self.output) * sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

# Example usage
X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(X, y)

# One complete iteration
nn.feedforward()
nn.backprop()
print(f"Output after one iteration:\n{nn.output}")
```

Slide 3: Forward Propagation

Forward propagation is the process of passing input data through the network to generate predictions. It involves matrix multiplications and activation functions at each layer.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # Linear activation for simplicity
        return self.a2

# Example usage
nn = SimpleNN(3, 4, 2)
X = np.array([[1, 2, 3]])
output = nn.forward(X)
print(f"Forward propagation output: {output}")
```

Slide 4: Loss Calculation

After forward propagation, we calculate the loss to measure how far our predictions are from the true values. Common loss functions include Mean Squared Error (MSE) for regression and Cross-Entropy for classification.

```python
import numpy as np

def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # Small value to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example data
y_true = np.array([[1, 0], [0, 1], [1, 1]])
y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6]])

# Calculate losses
mse = mse_loss(y_true, y_pred)
ce = cross_entropy_loss(y_true, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Cross-Entropy Loss: {ce}")
```

Slide 5: Backward Propagation

Backward propagation, or backpropagation, is the process of calculating gradients of the loss with respect to the network's parameters. These gradients indicate how to adjust the weights and biases to minimize the loss.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.y = y
        self.output = np.zeros(self.y.shape)
        
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

# Example usage
X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(X, y)

nn.feedforward()
print("Before backpropagation:")
print(nn.output)

nn.backprop()
nn.feedforward()
print("\nAfter backpropagation:")
print(nn.output)
```

Slide 6: Parameter Updates

After calculating gradients, we update the network's parameters (weights and biases) using an optimization algorithm. The most basic approach is Gradient Descent, where we move in the opposite direction of the gradient.

```python
import numpy as np

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # Linear activation for simplicity
        return self.a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        
        dZ2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, self.W2.T) * (self.z1 > 0)  # ReLU derivative
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update parameters
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

# Example usage
nn = SimpleNN(2, 3, 1)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print("Before update:")
print(nn.forward(X))

nn.backward(X, y, learning_rate=0.1)

print("\nAfter update:")
print(nn.forward(X))
```

Slide 7: Mini-batch Gradient Descent

In practice, we often use mini-batch gradient descent, where we update parameters after processing a small batch of examples. This approach balances computational efficiency and parameter update frequency.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        
        dZ2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, self.W2.T) * (self.a1 * (1 - self.a1))
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

# Generate example data
np.random.seed(0)
X = np.random.randn(1000, 2)
y = np.random.randint(2, size=(1000, 1))

# Initialize network and training parameters
nn = NeuralNetwork(2, 4, 1)
batch_size = 32
epochs = 100
learning_rate = 0.1

# Mini-batch gradient descent
for epoch in range(epochs):
    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        
        # Forward and backward pass
        nn.forward(X_batch)
        nn.backward(X_batch, y_batch, learning_rate)
    
    if epoch % 10 == 0:
        loss = np.mean(-y * np.log(nn.forward(X)) - (1-y) * np.log(1-nn.forward(X)))
        print(f"Epoch {epoch}, Loss: {loss}")

print("Final prediction samples:")
print(nn.forward(X[:5]))
```

Slide 8: Epochs and Iterations

An epoch is a complete pass through the entire dataset, while an iteration is a pass through a single batch. The number of iterations per epoch depends on the batch size and dataset size.

```python
import numpy as np

# Generate example data
np.random.seed(0)
X = np.random.randn(1000, 2)
y = np.random.randint(2, size=(1000, 1))

# Set hyperparameters
batch_size = 32
epochs = 5
dataset_size = X.shape[0]

# Calculate iterations per epoch
iterations_per_epoch = dataset_size // batch_size

print(f"Dataset size: {dataset_size}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {epochs}")
print(f"Iterations per epoch: {iterations_per_epoch}")
print(f"Total iterations: {iterations_per_epoch * epochs}")

# Simulate epochs and iterations
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}")
    for iteration in range(iterations_per_epoch):
        start_idx = iteration * batch_size
        end_idx = start_idx + batch_size
        
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        
        # Simulating forward and backward pass
        print(f"  Iteration {iteration + 1}: Processing batch {start_idx} to {end_idx}")

print("\nTraining complete!")
```

Slide 9: Learning Rate and its Impact

The learning rate determines the step size during parameter updates. A proper learning rate is crucial for convergence and training stability.

```python
import numpy as np
import matplotlib.pyplot as plt

def quadratic_function(x):
    return x**2

def gradient(x):
    return 2*x

def gradient_descent(start, learning_rate, num_iterations):
    x = start
    x_history = [x]
    for _ in range(num_iterations):
        x = x - learning_rate * gradient(x)
        x_history.append(x)
    return x_history

learning_rates = [0.01, 0.1, 0.5]
start = 5
num_iterations = 20

for lr in learning_rates:
    x_history = gradient_descent(start, lr, num_iterations)
    print(f"Learning rate: {lr}")
    print(f"Final x: {x_history[-1]}")
    print(f"Final loss: {quadratic_function(x_history[-1])}\n")

# Plotting code is omitted for simplicity
```

Slide 10: Overfitting and Underfitting

Overfitting occurs when a model learns the training data too well, including noise, while underfitting happens when the model is too simple to capture the underlying patterns in the data.

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Create models with different complexities
degrees = [1, 3, 15]  # Underfitting, good fit, overfitting
X_test = np.arange(0, 5, 0.01)[:, np.newaxis]

for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    X_test_poly = poly_features.transform(X_test)
    y_test_pred = model.predict(X_test_poly)
    
    train_rmse = np.sqrt(mean_squared_error(y, model.predict(X_poly)))
    print(f"Degree {degree} - Training RMSE: {train_rmse:.4f}")

# Plotting code is omitted for simplicity
```

Slide 11: Regularization

Regularization techniques help prevent overfitting by adding a penalty term to the loss function, discouraging complex models.

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(0)
X = np.random.randn(100, 20)
y = np.sum(X[:, :5], axis=1) + np.random.randn(100) * 0.1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

ridge.fit(X_train_scaled, y_train)
lasso.fit(X_train_scaled, y_train)

# Evaluate models
print("Ridge coefficients:", ridge.coef_)
print("Lasso coefficients:", lasso.coef_)

print("Ridge R^2 score:", ridge.score(X_test_scaled, y_test))
print("Lasso R^2 score:", lasso.score(X_test_scaled, y_test))
```

Slide 12: Dropout

Dropout is a regularization technique that randomly "drops out" a portion of neurons during training, reducing overfitting and improving generalization.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Create a simple model with dropout
model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Generate dummy data
import numpy as np
X = np.random.randn(1000, 20)
y = np.sum(X[:, :5], axis=1) + np.random.randn(1000) * 0.1

# Train the model
history = model.fit(X, y, epochs=100, validation_split=0.2, verbose=0)

print("Final training loss:", history.history['loss'][-1])
print("Final validation loss:", history.history['val_loss'][-1])
```

Slide 13: Early Stopping

Early stopping is a technique to prevent overfitting by monitoring the model's performance on a validation set and stopping training when the performance starts to degrade.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Generate synthetic data
X = np.random.randn(1000, 20)
y = np.sum(X[:, :5], axis=1) + np.random.randn(1000) * 0.1

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a model
model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_val, y_val), 
                    callbacks=[early_stopping], verbose=0)

print(f"Training stopped after {len(history.history['loss'])} epochs")
print(f"Best validation loss: {min(history.history['val_loss'])}")
```

Slide 14: Real-life Example: Image Classification

Neural networks are widely used in image classification tasks, such as identifying objects in photographs or medical imaging.

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load and preprocess an image (replace with your own image path)
img_path = 'path/to/your/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make prediction
preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]

print("Top 3 predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_preds):
    print(f"{i + 1}: {label} ({score:.2f})")
```

Slide 15: Real-life Example: Natural Language Processing

Neural networks are crucial in various NLP tasks, including sentiment analysis, machine translation, and text generation.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample data
texts = [
    "I love this movie!",
    "This film is terrible.",
    "Great acting and storyline.",
    "Waste of time and money."
]
labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# Tokenize the text
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
padded = pad_sequences(sequences, maxlen=20, padding='post', truncating='post')

# Create the model
model = Sequential([
    Embedding(1000, 16, input_length=20),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded, labels, epochs=10, verbose=0)

# Test the model
test_text = ["This movie is amazing!"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_padded = pad_sequences(test_seq, maxlen=20, padding='post', truncating='post')

prediction = model.predict(test_padded)
print(f"Sentiment prediction: {'Positive' if prediction[0][0] > 0.5 else 'Negative'}")
```

Slide 16: Additional Resources

For further exploration of neural network iterations and related topics, consider the following resources:

1. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press, 2016)
2. "Neural Networks and Deep Learning" by Michael Nielsen (Online book: [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/))
3. ArXiv paper: "An Overview of Gradient Descent Optimization Algorithms" by Sebastian Ruder ([https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747))
4. ArXiv paper: "Efficient BackProp" by Yann LeCun et al. ([https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533))
5. TensorFlow tutorials: [https://www.tensorflow.org/tutorials](https://www.tensorflow.org/tutorials)
6. PyTorch tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

These resources provide in-depth explanations, mathematical foundations, and practical implementations of neural network concepts and training algorithms.

