## Key Concepts in Deep Learning Gradients, Derivatives, and Loss Functions
Slide 1: Introduction to Gradients

Gradients are fundamental to deep learning, representing the direction of steepest increase in a function. They are crucial for optimizing neural networks through backpropagation.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 + y**2

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contour(X, Y, Z, levels=20)
plt.colorbar(label='f(x, y)')
plt.title('Contour plot of f(x, y) = x^2 + y^2')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# The gradient at point (2, 3)
gradient = np.array([2*2, 2*3])
print(f"Gradient at (2, 3): {gradient}")
```

Slide 2: Partial Derivatives

Partial derivatives measure the rate of change of a function with respect to one variable while holding others constant. They form the components of the gradient vector.

```python
def partial_derivative(f, var, point, h=1e-5):
    point_high = list(point)
    point_low = list(point)
    point_high[var] += h
    point_low[var] -= h
    return (f(*point_high) - f(*point_low)) / (2*h)

def f(x, y):
    return x**2 + y**2

point = (2, 3)
dx = partial_derivative(f, 0, point)
dy = partial_derivative(f, 1, point)

print(f"∂f/∂x at (2, 3): {dx}")
print(f"∂f/∂y at (2, 3): {dy}")
```

Slide 3: Gradient Descent

Gradient descent is an optimization algorithm that iteratively adjusts parameters in the direction opposite to the gradient to minimize a loss function.

```python
def gradient_descent(f, initial_point, learning_rate=0.1, num_iterations=100):
    point = np.array(initial_point, dtype=float)
    
    for _ in range(num_iterations):
        grad = np.array([partial_derivative(f, i, point) for i in range(len(point))])
        point -= learning_rate * grad
    
    return point

initial_point = (4, 4)
minimum = gradient_descent(f, initial_point)
print(f"Minimum found at: {minimum}")
print(f"Function value at minimum: {f(*minimum)}")
```

Slide 4: Learning Rate and Convergence

The learning rate controls the step size in gradient descent. Too large, and the algorithm may overshoot; too small, and convergence is slow.

```python
def plot_gradient_descent(f, initial_point, learning_rates):
    fig, axs = plt.subplots(1, len(learning_rates), figsize=(15, 5))
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    for i, lr in enumerate(learning_rates):
        point = np.array(initial_point, dtype=float)
        trajectory = [point]
        
        for _ in range(20):
            grad = np.array([partial_derivative(f, j, point) for j in range(len(point))])
            point -= lr * grad
            trajectory.append(point)
        
        axs[i].contour(X, Y, Z, levels=20)
        axs[i].plot(*zip(*trajectory), 'ro-')
        axs[i].set_title(f'Learning rate: {lr}')
    
    plt.tight_layout()
    plt.show()

plot_gradient_descent(f, (4, 4), [0.01, 0.1, 0.5])
```

Slide 5: Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) approximates the gradient using a small subset of data, making it more efficient for large datasets.

```python
import numpy as np

def sgd(X, y, learning_rate=0.01, epochs=1000, batch_size=32):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            gradient = 2/batch_size * X_batch.T.dot(X_batch.dot(theta) - y_batch)
            theta -= learning_rate * gradient
    
    return theta

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 5)
true_theta = np.array([1, 2, 3, 4, 5])
y = X.dot(true_theta) + np.random.randn(1000) * 0.1

# Run SGD
theta_sgd = sgd(X, y)
print("Estimated parameters:", theta_sgd)
print("True parameters:", true_theta)
```

Slide 6: Adam Optimizer

Adam (Adaptive Moment Estimation) is an advanced optimization algorithm that adapts the learning rate for each parameter.

```python
def adam(grad, x, step_size, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    
    while True:
        t += 1
        g = grad(x)
        if np.all(np.abs(g) < 1e-6):
            break
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= step_size * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return x

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    return np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
        200 * (x[1] - x[0]**2)
    ])

x0 = np.array([-1.0, 2.0])
solution = adam(rosenbrock_grad, x0, step_size=0.1)
print("Optimum found:", solution)
print("Rosenbrock value at optimum:", rosenbrock(solution))
```

Slide 7: Loss Functions

Loss functions measure the difference between predicted and actual values, guiding the learning process in neural networks.

```python
import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.linspace(0, 1, 100)

mse = [mean_squared_error(y_true, np.full_like(y_true, p)) for p in y_pred]
bce = [binary_cross_entropy(y_true, np.full_like(y_true, p)) for p in y_pred]

plt.figure(figsize=(10, 5))
plt.plot(y_pred, mse, label='Mean Squared Error')
plt.plot(y_pred, bce, label='Binary Cross-Entropy')
plt.xlabel('Predicted Value')
plt.ylabel('Loss')
plt.title('Comparison of Loss Functions')
plt.legend()
plt.show()
```

Slide 8: Backpropagation

Backpropagation efficiently computes gradients in neural networks by applying the chain rule of calculus.

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
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(X, y)

for i in range(1500):
    nn.feedforward()
    nn.backprop()

print(nn.output)
```

Slide 9: Vanishing and Exploding Gradients

Vanishing and exploding gradients are common problems in deep networks, where gradients become extremely small or large during backpropagation.

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
plt.title('Activation Functions')
plt.legend()
plt.grid(True)
plt.show()

# Simulate gradient flow in a deep network
def gradient_flow(activation, depth):
    gradient = 1
    for _ in range(depth):
        gradient *= np.random.choice([-1, 1]) * activation(np.random.randn())
    return gradient

depths = range(1, 51)
relu_gradients = [np.mean([abs(gradient_flow(relu, d)) for _ in range(1000)]) for d in depths]
sigmoid_gradients = [np.mean([abs(gradient_flow(sigmoid, d)) for _ in range(1000)]) for d in depths]
tanh_gradients = [np.mean([abs(gradient_flow(tanh, d)) for _ in range(1000)]) for d in depths]

plt.figure(figsize=(12, 4))
plt.plot(depths, relu_gradients, label='ReLU')
plt.plot(depths, sigmoid_gradients, label='Sigmoid')
plt.plot(depths, tanh_gradients, label='Tanh')
plt.title('Gradient Magnitude vs Network Depth')
plt.xlabel('Network Depth')
plt.ylabel('Average Gradient Magnitude')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()
```

Slide 10: Batch Normalization

Batch Normalization is a technique to stabilize the distribution of layer inputs, mitigating the vanishing/exploding gradient problem and accelerating training.

```python
import numpy as np

def batch_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    return out, mean, var, x_norm

# Generate sample data
np.random.seed(42)
x = np.random.randn(100, 3)  # 100 samples, 3 features
gamma = np.ones(3)
beta = np.zeros(3)

# Apply batch normalization
normalized_x, mean, var, x_norm = batch_norm(x, gamma, beta)

print("Original data mean:", np.mean(x, axis=0))
print("Original data variance:", np.var(x, axis=0))
print("Normalized data mean:", np.mean(normalized_x, axis=0))
print("Normalized data variance:", np.var(normalized_x, axis=0))

# Visualize the effect of batch normalization
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(x[:, 0], bins=30, alpha=0.5, label='Original')
plt.hist(normalized_x[:, 0], bins=30, alpha=0.5, label='Normalized')
plt.title('Distribution of First Feature')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(x[:, 0], x[:, 1], alpha=0.5, label='Original')
plt.scatter(normalized_x[:, 0], normalized_x[:, 1], alpha=0.5, label='Normalized')
plt.title('First Two Features')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 11: Regularization Techniques

Regularization helps prevent overfitting by adding a penalty term to the loss function, encouraging simpler models.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(0)
X = np.sort(np.random.rand(20, 1), axis=0)
y = np.cos(1.5 * np.pi * X).ravel() + np.random.randn(20) * 0.1

X_test = np.linspace(0, 1, 100).reshape(-1, 1)

models = [
    make_pipeline(PolynomialFeatures(degree=15), LinearRegression()),
    make_pipeline(PolynomialFeatures(degree=15), Ridge(alpha=0.1)),
    make_pipeline(PolynomialFeatures(degree=15), Lasso(alpha=0.01))
]

model_names = ['Unregularized', 'Ridge', 'Lasso']

plt.figure(figsize=(15, 5))
for i, (name, model) in enumerate(zip(model_names, models)):
    model.fit(X, y)
    y_pred = model.predict(X_test)
    
    plt.subplot(1, 3, i+1)
    plt.scatter(X, y, color='red', label='Data')
    plt.plot(X_test, y_pred, label='Prediction')
    plt.title(name)
    plt.legend()

plt.tight_layout()
plt.show()
```

Slide 12: Dropout

Dropout is a regularization technique that randomly deactivates neurons during training, reducing overfitting and improving generalization.

```python
import numpy as np

def dropout(X, drop_prob=0.5):
    mask = np.random.binomial(1, 1-drop_prob, size=X.shape) / (1-drop_prob)
    return X * mask

# Example usage
X = np.random.randn(5, 10)
X_dropout = dropout(X)

print("Original input:")
print(X)
print("\nInput after dropout:")
print(X_dropout)

# Visualize dropout effect
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(X, cmap='viridis')
plt.title('Original Input')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(X_dropout, cmap='viridis')
plt.title('Input after Dropout')
plt.colorbar()

plt.tight_layout()
plt.show()
```

Slide 13: Transfer Learning

Transfer learning leverages knowledge from pre-trained models to improve performance on new, related tasks with limited data.

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# Create new model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
```

Slide 14: Hyperparameter Tuning

Hyperparameter tuning is crucial for optimizing model performance. Grid search and random search are common techniques for finding the best hyperparameters.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# Define model and parameter grid
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Plot results
import matplotlib.pyplot as plt

results = grid_search.cv_results_
plt.figure(figsize=(12, 6))
plt.plot(results['param_n_estimators'], results['mean_test_score'], 'o-')
plt.xlabel('Number of Estimators')
plt.ylabel('Mean Test Score')
plt.title('Grid Search Results: Number of Estimators')
plt.show()
```

Slide 15: Real-life Example: Image Classification

Image classification is a common application of deep learning, used in various fields such as medical diagnosis and autonomous vehicles.

```python
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load and preprocess an image
img_path = 'path_to_your_image.jpg'  # Replace with actual image path
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make prediction
preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]

# Display results
plt.imshow(img)
plt.axis('off')
plt.title('Input Image')
plt.show()

for i, (imagenet_id, label, score) in enumerate(decoded_preds):
    print(f"{i+1}: {label} ({score:.2f})")
```

Slide 16: Real-life Example: Natural Language Processing

Natural Language Processing (NLP) is another important application of deep learning, used in tasks like sentiment analysis and language translation.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample data
texts = ["I love this movie", "This film is terrible", "Great acting and plot"]
labels = [1, 0, 1]  # 1 for positive, 0 for negative

# Tokenize text
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
padded = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')

# Create model
model = Sequential([
    Embedding(1000, 16, input_length=10),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(padded, labels, epochs=10, verbose=0)

# Make prediction
new_text = ["This movie is amazing"]
new_seq = tokenizer.texts_to_sequences(new_text)
new_padded = pad_sequences(new_seq, maxlen=10, padding='post', truncating='post')
prediction = model.predict(new_padded)

print(f"Sentiment prediction for '{new_text[0]}': {prediction[0][0]:.2f}")
```

Slide 17: Additional Resources

For further exploration of deep learning concepts:

1. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press)
2. "Neural Networks and Deep Learning" by Michael Nielsen (online book)
3. Stanford CS231n: Convolutional Neural Networks for Visual Recognition (course materials available online)
4. ArXiv.org for latest research papers in machine learning and deep learning Example: "Attention Is All You Need" by Vaswani et al. (2017) - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

Remember to verify the availability and accuracy of these resources, as they may change over time.

