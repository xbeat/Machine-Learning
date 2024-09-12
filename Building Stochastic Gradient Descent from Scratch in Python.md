## Building Stochastic Gradient Descent from Scratch in Python
Slide 1: Introduction to Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent is a fundamental optimization algorithm used in machine learning to minimize the loss function. It's an iterative method that updates model parameters based on the gradient of the loss function with respect to those parameters. Unlike traditional gradient descent, SGD uses only a subset of the data (mini-batch) in each iteration, making it more efficient for large datasets.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple linear regression model
def predict(X, w, b):
    return X * w + b

# Mean Squared Error loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

plt.scatter(X, y)
plt.title("Sample Data for Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

Slide 2: Gradient Calculation

The gradient is the partial derivative of the loss function with respect to each parameter. For our simple linear regression model, we need to calculate the gradients for the weight (w) and bias (b).

```python
def calculate_gradients(X, y, y_pred, w, b):
    m = len(y)
    dw = (2/m) * np.sum(X * (y_pred - y))
    db = (2/m) * np.sum(y_pred - y)
    return dw, db

# Test gradient calculation
w, b = 0, 0
y_pred = predict(X, w, b)
dw, db = calculate_gradients(X, y, y_pred, w, b)
print(f"Initial gradients: dw = {dw:.4f}, db = {db:.4f}")
```

Slide 3: SGD Update Rule

The SGD update rule adjusts the parameters in the opposite direction of the gradient, scaled by the learning rate. This process is repeated for a fixed number of iterations or until convergence.

```python
def sgd_update(w, b, dw, db, learning_rate):
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

# Example update
learning_rate = 0.1
w, b = sgd_update(w, b, dw, db, learning_rate)
print(f"Updated parameters: w = {w:.4f}, b = {b:.4f}")
```

Slide 4: Mini-batch Selection

In SGD, we use mini-batches to estimate the gradient. This involves randomly selecting a subset of the data for each iteration, which helps reduce computational cost and adds noise to the optimization process, potentially helping to escape local minima.

```python
def get_mini_batch(X, y, batch_size):
    indices = np.random.randint(0, len(X), batch_size)
    return X[indices], y[indices]

# Example mini-batch selection
batch_size = 32
X_batch, y_batch = get_mini_batch(X, y, batch_size)
print(f"Mini-batch shapes: X = {X_batch.shape}, y = {y_batch.shape}")
```

Slide 5: Implementing SGD Training Loop

Now we'll implement the main SGD training loop, which combines all the previous components. We'll iterate through a specified number of epochs, selecting mini-batches, calculating gradients, and updating parameters.

```python
def train_sgd(X, y, learning_rate, batch_size, epochs):
    w, b = 0, 0
    losses = []
    
    for epoch in range(epochs):
        for _ in range(len(X) // batch_size):
            X_batch, y_batch = get_mini_batch(X, y, batch_size)
            y_pred = predict(X_batch, w, b)
            dw, db = calculate_gradients(X_batch, y_batch, y_pred, w, b)
            w, b = sgd_update(w, b, dw, db, learning_rate)
        
        # Calculate and store loss for the entire dataset
        y_pred = predict(X, w, b)
        loss = mse_loss(y, y_pred)
        losses.append(loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return w, b, losses

# Train the model
w, b, losses = train_sgd(X, y, learning_rate=0.1, batch_size=32, epochs=100)
print(f"Final parameters: w = {w:.4f}, b = {b:.4f}")
```

Slide 6: Visualizing Training Progress

To understand how our SGD algorithm is performing, we can visualize the loss over time and the final regression line.

```python
# Plot loss over time
plt.plot(losses)
plt.title("Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.show()

# Plot final regression line
plt.scatter(X, y)
plt.plot(X, predict(X, w, b), color='red')
plt.title("Final Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

Slide 7: Hyperparameter Tuning

SGD's performance depends on several hyperparameters, including learning rate and batch size. Let's experiment with different values to see their impact on the training process.

```python
learning_rates = [0.01, 0.1, 1.0]
batch_sizes = [8, 32, 128]

for lr in learning_rates:
    for bs in batch_sizes:
        w, b, losses = train_sgd(X, y, learning_rate=lr, batch_size=bs, epochs=100)
        plt.plot(losses, label=f"LR={lr}, BS={bs}")

plt.title("Loss vs. Epoch for Different Hyperparameters")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.show()
```

Slide 8: Learning Rate Scheduling

To improve convergence, we can implement learning rate scheduling, which reduces the learning rate over time. This allows for larger updates at the beginning of training and finer adjustments towards the end.

```python
def exponential_decay(initial_lr, decay_rate, epoch):
    return initial_lr * (decay_rate ** epoch)

def train_sgd_with_lr_decay(X, y, initial_lr, decay_rate, batch_size, epochs):
    w, b = 0, 0
    losses = []
    
    for epoch in range(epochs):
        lr = exponential_decay(initial_lr, decay_rate, epoch)
        
        for _ in range(len(X) // batch_size):
            X_batch, y_batch = get_mini_batch(X, y, batch_size)
            y_pred = predict(X_batch, w, b)
            dw, db = calculate_gradients(X_batch, y_batch, y_pred, w, b)
            w, b = sgd_update(w, b, dw, db, lr)
        
        y_pred = predict(X, w, b)
        loss = mse_loss(y, y_pred)
        losses.append(loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, LR: {lr:.4f}, Loss: {loss:.4f}")
    
    return w, b, losses

w, b, losses = train_sgd_with_lr_decay(X, y, initial_lr=0.1, decay_rate=0.99, batch_size=32, epochs=100)
plt.plot(losses)
plt.title("Loss vs. Epoch with Learning Rate Decay")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.show()
```

Slide 9: Momentum

Momentum is a technique that helps accelerate SGD in the relevant direction and dampens oscillations. It does this by adding a fraction of the update vector of the past time step to the current update vector.

```python
def train_sgd_with_momentum(X, y, learning_rate, momentum, batch_size, epochs):
    w, b = 0, 0
    v_w, v_b = 0, 0
    losses = []
    
    for epoch in range(epochs):
        for _ in range(len(X) // batch_size):
            X_batch, y_batch = get_mini_batch(X, y, batch_size)
            y_pred = predict(X_batch, w, b)
            dw, db = calculate_gradients(X_batch, y_batch, y_pred, w, b)
            
            v_w = momentum * v_w - learning_rate * dw
            v_b = momentum * v_b - learning_rate * db
            
            w += v_w
            b += v_b
        
        y_pred = predict(X, w, b)
        loss = mse_loss(y, y_pred)
        losses.append(loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return w, b, losses

w, b, losses = train_sgd_with_momentum(X, y, learning_rate=0.01, momentum=0.9, batch_size=32, epochs=100)
plt.plot(losses)
plt.title("Loss vs. Epoch with Momentum")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.show()
```

Slide 10: Real-Life Example: Image Classification

SGD is widely used in training neural networks for image classification. Let's create a simple example using the MNIST dataset.

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train MLP classifier using SGD
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=0.1)

mlp.fit(X_train, y_train)
print(f"Training set score: {mlp.score(X_train, y_train):.4f}")
print(f"Test set score: {mlp.score(X_test, y_test):.4f}")
```

Slide 11: Real-Life Example: Natural Language Processing

SGD is also commonly used in training word embeddings for natural language processing tasks. Here's a simple example using the Word2Vec model.

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# Sample sentences
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Natural language processing is an important field in AI"
]

# Tokenize sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Train Word2Vec model using SGD
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# Find similar words
similar_words = model.wv.most_similar("learning", topn=3)
print("Words similar to 'learning':")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")
```

Slide 12: Challenges and Considerations

When implementing SGD, there are several challenges to consider:

1. Choosing appropriate hyperparameters (learning rate, batch size, etc.) can be difficult and may require extensive tuning.
2. SGD can be sensitive to feature scaling, so preprocessing the data is often necessary.
3. The stochastic nature of SGD can make it difficult to reproduce results exactly.
4. SGD may struggle with saddle points in high-dimensional optimization problems.

To address these challenges, consider using adaptive learning rate methods like Adam or RMSprop, implementing proper initialization techniques, and using regularization to prevent overfitting.

```python
# Example of feature scaling and regularization
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sgd_reg = SGDRegressor(loss='squared_error', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3)
sgd_reg.fit(X_scaled, y.ravel())

print(f"Coefficients: {sgd_reg.coef_}")
print(f"Intercept: {sgd_reg.intercept_}")
```

Slide 13: Conclusion and Future Directions

Stochastic Gradient Descent is a powerful optimization algorithm that forms the backbone of many machine learning models. Its efficiency and ability to handle large datasets make it particularly suitable for deep learning applications. As you continue to explore SGD, consider investigating more advanced techniques such as:

1. Adaptive learning rate methods (Adam, RMSprop, Adagrad)
2. Batch normalization
3. Gradient clipping
4. Second-order optimization methods

By mastering SGD and its variants, you'll be well-equipped to tackle a wide range of machine learning problems and contribute to the ongoing advancements in the field.

```python
# Visualize the optimization landscape
from mpl_toolkits.mplot3d import Axes3D

def loss_surface(w, b):
    return np.mean((y - (X * w + b)) ** 2)

w_range = np.linspace(-1, 4, 100)
b_range = np.linspace(-1, 4, 100)
W, B = np.meshgrid(w_range, b_range)
Z = np.array([loss_surface(w, b) for w, b in zip(np.ravel(W), np.ravel(B))]).reshape(W.shape)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, Z, cmap='viridis')
ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('Loss')
ax.set_title('Loss Surface for Linear Regression')
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into Stochastic Gradient Descent and its applications, here are some valuable resources:

1. "Optimization Methods for Large-Scale Machine Learning" by Léon Bottou, Frank E. Curtis, and Jorge Nocedal (2018). Available at: [https://arxiv.org/abs/1606.04838](https://arxiv.org/abs/1606.04838)
2. "Adam: A Method for Stochastic Optimization" by Diederik P. Kingma and Jimmy Ba (2014). Available at: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
3. "An Overview of Gradient Descent Optimization Algorithms" by Sebastian Ruder (2016). Available at: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)

These papers provide in-depth analyses of SGD and its variants, offering valuable insights into the theoretical foundations and practical applications of these optimization techniques.

