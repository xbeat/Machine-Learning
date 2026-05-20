## Parameters vs Hyperparameters in Machine Learning
Slide 1: Parameters vs Hyperparameters in Machine Learning

Parameters and hyperparameters are fundamental concepts in machine learning, but they are often confused. This presentation will clarify the differences between these two important elements and demonstrate their roles using Python examples.

```python
import numpy as np
import matplotlib.pyplot as plt

# This code will be used to visualize the difference
# between parameters and hyperparameters throughout the presentation
```

Slide 2: What are Parameters?

Parameters are internal variables of a model that are learned from the training data. They are adjusted during the training process to minimize the loss function and improve the model's performance.

```python
import numpy as np

# Simple linear regression model
class LinearRegression:
    def __init__(self):
        self.slope = None  # Parameter
        self.intercept = None  # Parameter

    def fit(self, X, y):
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean)**2)
        
        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * X_mean

# Example usage
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, y)

print(f"Learned parameters: slope = {model.slope:.2f}, intercept = {model.intercept:.2f}")
```

Slide 3: Characteristics of Parameters

Parameters are internal to the model and are learned during training. They define the model's structure and are essential for making predictions. Parameters are typically not set manually but are optimized by the learning algorithm.

```python
import tensorflow as tf

# Creating a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Print model summary to see parameters
model.summary()
```

Slide 4: What are Hyperparameters?

Hyperparameters are external configuration settings that are not learned from the data. They are set before the learning process begins and control various aspects of the model's behavior and training process.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, 
                           n_redundant=0, n_classes=2, random_state=42)

# Define the model and hyperparameters to search
model = SVC()
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}

# Perform grid search
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)

print("Best hyperparameters:", grid_search.best_params_)
```

Slide 5: Characteristics of Hyperparameters

Hyperparameters are set before training and remain constant during the learning process. They often require manual tuning or automated optimization techniques to find the best values for a given problem.

```python
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                           n_redundant=5, n_classes=2, random_state=42)

# Define the model and hyperparameters to search
model = RandomForestClassifier()
param_dist = {
    'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Perform randomized search
random_search = RandomizedSearchCV(model, param_distributions=param_dist, 
                                   n_iter=100, cv=3, random_state=42)
random_search.fit(X, y)

print("Best hyperparameters:", random_search.best_params_)
```

Slide 6: Key Differences: Parameters vs Hyperparameters

Parameters are learned from data, while hyperparameters are set manually. Parameters define the model's structure, while hyperparameters control the learning process. Let's visualize this difference using a simple example.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate sample data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X**2 + 2 * X + 1 + np.random.randn(100, 1) * 10

# Create and fit models with different degrees (hyperparameter)
degrees = [1, 3, 10]
plt.figure(figsize=(15, 5))

for i, degree in enumerate(degrees):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    y_pred = model.predict(X_poly)
    
    plt.subplot(1, 3, i+1)
    plt.scatter(X, y, color='blue', alpha=0.7, label='Data')
    plt.plot(X, y_pred, color='red', label='Model')
    plt.title(f'Degree (Hyperparameter): {degree}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()

plt.tight_layout()
plt.show()
```

Slide 7: Real-life Example: Image Classification

In image classification tasks, convolutional neural networks (CNNs) are commonly used. Let's explore the parameters and hyperparameters in a CNN model for classifying handwritten digits.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the CNN model
def create_cnn_model(num_filters, kernel_size, dropout_rate):
    model = models.Sequential([
        layers.Conv2D(num_filters, kernel_size, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Hyperparameters
num_filters = 32
kernel_size = (3, 3)
dropout_rate = 0.5

# Create and compile the model
model = create_cnn_model(num_filters, kernel_size, dropout_rate)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary to see parameters
model.summary()

print(f"Hyperparameters used:")
print(f"Number of filters: {num_filters}")
print(f"Kernel size: {kernel_size}")
print(f"Dropout rate: {dropout_rate}")
```

Slide 8: Hyperparameter Tuning

Hyperparameter tuning is the process of finding the best combination of hyperparameters for a given model and dataset. Let's demonstrate this using a simple example with scikit-learn.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define the model and hyperparameter space
rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': randint(10, 200),
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 11)
}

# Perform randomized search
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, 
                                   n_iter=100, cv=5, random_state=42)
random_search.fit(X, y)

print("Best hyperparameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)
```

Slide 9: Impact of Hyperparameters on Model Performance

Let's visualize how changing a hyperparameter can affect model performance using a simple example of regularization in linear regression.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate sample data
np.random.seed(42)
X = np.sort(np.random.rand(20, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + np.random.randn(20) * 0.1

# Create and plot models with different alpha values (hyperparameter)
alphas = [0, 0.1, 1, 10]
degrees = 10
X_plot = np.linspace(0, 1, 100)[:, np.newaxis]

plt.figure(figsize=(12, 8))
for i, alpha in enumerate(alphas):
    model = make_pipeline(PolynomialFeatures(degrees), Ridge(alpha=alpha))
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    
    plt.subplot(2, 2, i+1)
    plt.scatter(X, y, color='red', label='data')
    plt.plot(X_plot, y_plot, label=f'alpha={alpha}')
    plt.ylim(-2, 2)
    plt.legend()
    plt.title(f'Ridge Regression (alpha={alpha})')

plt.tight_layout()
plt.show()
```

Slide 10: Parameters in Deep Learning

In deep learning models, parameters typically include weights and biases. Let's create a simple neural network and examine its parameters.

```python
import tensorflow as tf

# Create a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Print model summary
model.summary()

# Access and print the weights of the first layer
first_layer_weights = model.layers[0].get_weights()[0]
print("First layer weights shape:", first_layer_weights.shape)
print("First layer weights:")
print(first_layer_weights)
```

Slide 11: Hyperparameters in Deep Learning

Hyperparameters in deep learning include network architecture, learning rate, batch size, and more. Let's demonstrate how to set and use these hyperparameters.

```python
import tensorflow as tf

# Hyperparameters
learning_rate = 0.001
batch_size = 32
epochs = 10
hidden_units = [64, 32]

# Create a model with hyperparameters
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_units[0], activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(hidden_units[1], activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model with learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Generate dummy data
X_dummy = tf.random.normal((1000, 784))
y_dummy = tf.random.uniform((1000,), minval=0, maxval=10, dtype=tf.int32)

# Train the model
history = model.fit(X_dummy, y_dummy, batch_size=batch_size, epochs=epochs, validation_split=0.2)

print(f"Hyperparameters used:")
print(f"Learning rate: {learning_rate}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {epochs}")
print(f"Hidden units: {hidden_units}")
```

Slide 12: Real-life Example: Natural Language Processing

In natural language processing tasks, we often use recurrent neural networks (RNNs) or transformers. Let's examine the parameters and hyperparameters in a simple RNN model for text classification.

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.models import Sequential

# Hyperparameters
vocab_size = 10000
embedding_dim = 100
max_length = 100
rnn_units = 64

# Create the model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    SimpleRNN(rnn_units),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

print(f"Hyperparameters used:")
print(f"Vocabulary size: {vocab_size}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Maximum sequence length: {max_length}")
print(f"RNN units: {rnn_units}")
```

Slide 13: Balancing Parameters and Hyperparameters

Finding the right balance between model complexity (number of parameters) and hyperparameter settings is crucial for achieving good performance. Let's visualize this balance using a learning curve.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.datasets import load_digits

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Define the model with different complexity (C hyperparameter)
C_values = [0.1, 1, 10]

plt.figure(figsize=(15, 5))

for i, C in enumerate(C_values):
    model = SVC(kernel='rbf', C=C, random_state=42)
    
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)
    
    train_mean = np.mean(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    
    plt.subplot(1, 3, i+1)
    plt.plot(train_sizes, train_mean, 'o-', label="Training score")
    plt.plot(train_sizes, valid_mean, 'o-', label="Cross-validation score")
    plt.title(f"Learning Curve (C={C})")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")

plt.tight_layout()
plt.show()
```

Slide 14: Conclusion: Parameters vs Hyperparameters

Parameters and hyperparameters play distinct roles in machine learning models. Parameters are learned from data and define the model's structure, while hyperparameters control the learning process and model complexity. Balancing both is key to creating effective models.

```python
# Pseudocode for a typical machine learning workflow
def train_model(X, y, hyperparameters):
    model = create_model(hyperparameters)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    performance = calculate_performance(predictions, y_test)
    return performance

def optimize_hyperparameters(X, y, hyperparameter_space):
    best_hyperparameters = None
    best_performance = float('-inf')
    
    for hyperparameters in hyperparameter_space:
        model = train_model(X, y, hyperparameters)
        performance = evaluate_model(model, X_test, y_test)
        
        if performance > best_performance:
            best_performance = performance
            best_hyperparameters = hyperparameters
    
    return best_hyperparameters

# Main workflow
X, y = load_data()
best_hyperparameters = optimize_hyperparameters(X, y, hyperparameter_space)
final_model = train_model(X, y, best_hyperparameters)
```

Slide 15: Additional Resources

For more in-depth information on parameters and hyperparameters in machine learning, consider exploring these resources:

1. "Hyperparameter Optimization in Machine Learning Models" (arXiv:2003.12407) URL: [https://arxiv.org/abs/2003.12407](https://arxiv.org/abs/2003.12407)
2. "Neural Architecture Search: A Survey" (arXiv:1808.05377) URL: [https://arxiv.org/abs/1808.05377](https://arxiv.org/abs/1808.05377)
3. "Random Search for Hyper-Parameter Optimization" (arXiv:1703.01780) URL: [https://arxiv.org/abs/1703.01780](https://arxiv.org/abs/1703.01780)

These papers provide comprehensive overviews and advanced techniques for working with parameters and hyperparameters in various machine learning contexts.

