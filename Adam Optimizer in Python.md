## Adam Optimizer in Python
Slide 1: Introduction to Adam Optimizer

Adam (Adaptive Moment Estimation) is an optimization algorithm used in training deep learning models. It combines the benefits of two other extensions of stochastic gradient descent: Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp). Adam is designed to efficiently handle sparse gradients and noisy problems in machine learning.

```python
import tensorflow as tf

# Creating an Adam optimizer instance
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
```

Slide 2: Key Features of Adam

Adam adapts the learning rate for each parameter individually. It maintains a running average of both the gradients and the squared gradients. This approach allows it to handle a wide range of problems effectively, making it a popular choice for deep learning practitioners.

```python
# Example of using Adam in a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

Slide 3: Adam's Parameters

Adam has several important parameters: learning rate (α), beta\_1 (β₁), beta\_2 (β₂), and epsilon (ε). The learning rate determines the step size at each iteration, while beta\_1 and beta\_2 control the decay rates of the moving averages. Epsilon is a small constant added for numerical stability.

```python
# Customizing Adam's parameters
custom_adam = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False
)
```

Slide 4: The Math Behind Adam

Adam updates weights using the following formulas: m\_t = β₁ \* m\_t-1 + (1 - β₁) \* g\_t v\_t = β₂ \* v\_t-1 + (1 - β₂) \* g\_t² m̂\_t = m\_t / (1 - β₁^t) v̂\_t = v\_t / (1 - β₂^t) θ\_t = θ\_t-1 - α \* m̂\_t / (√v̂\_t + ε)

Where g\_t is the gradient at time t, m\_t and v\_t are the first and second moment estimates, and θ\_t represents the parameters.

```python
# Simplified implementation of Adam update step
def adam_update(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v
```

Slide 5: Bias Correction in Adam

Adam incorporates bias correction to account for the initialization of the first and second moment estimates. This correction helps in the early stages of training when the moving averages are biased towards zero.

```python
# Bias correction demonstration
t = 1
m = 0
v = 0
grad = 0.1
beta1 = 0.9
beta2 = 0.999

m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * (grad ** 2)

m_corrected = m / (1 - beta1 ** t)
v_corrected = v / (1 - beta2 ** t)

print(f"Uncorrected m: {m}, Corrected m: {m_corrected}")
print(f"Uncorrected v: {v}, Corrected v: {v_corrected}")
```

Slide 6: Adam vs. Other Optimizers

Adam combines the advantages of AdaGrad and RMSProp. It adapts the learning rate for each parameter like AdaGrad and uses moving averages of squared gradients like RMSProp. This makes Adam suitable for a wide range of problems and often performs better than basic SGD.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Function to optimize
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# Gradient function
def rosenbrock_grad(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# Optimization loop
def optimize(optimizer, initial_position, steps):
    x, y = initial_position
    path = [initial_position]
    
    for _ in range(steps):
        grads = rosenbrock_grad(x, y)
        optimizer.apply_gradients(zip([tf.constant(grads)], [tf.Variable([x, y])]))
        x, y = optimizer.variables()[0].numpy()
        path.append([x, y])
    
    return np.array(path)

# Compare optimizers
optimizers = {
    'SGD': tf.optimizers.SGD(learning_rate=0.001),
    'Adam': tf.optimizers.Adam(learning_rate=0.001),
    'RMSprop': tf.optimizers.RMSprop(learning_rate=0.001)
}

initial_position = [-1.5, 2.5]
steps = 1000

for name, opt in optimizers.items():
    path = optimize(opt, initial_position, steps)
    plt.plot(path[:, 0], path[:, 1], label=name)

plt.legend()
plt.title('Optimizer Comparison on Rosenbrock Function')
plt.show()
```

Slide 7: Implementing Adam from Scratch

To truly understand Adam, let's implement it from scratch using NumPy. This implementation follows the original paper's algorithm and can be used as a standalone optimizer.

```python
import numpy as np

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params

# Usage example
params = np.array([1.0, 2.0])
grads = np.array([0.1, 0.2])
optimizer = Adam()

for _ in range(10):
    params = optimizer.update(params, grads)
    print(params)
```

Slide 8: Adam in Practice: Image Classification

Let's use Adam to train a convolutional neural network (CNN) for image classification on the CIFAR-10 dataset. This example demonstrates how Adam performs in a real-world scenario.

```python
import tensorflow as tf

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model with Adam optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")
```

Slide 9: Visualizing Adam's Learning Process

To better understand how Adam works, let's visualize its optimization process on a simple 2D function. We'll use matplotlib to create an animation of the optimization path.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def himmelblau_grad(x, y):
    dx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
    dy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
    return np.array([dx, dy])

class AdamOptimizer:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params

# Set up the plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = himmelblau(X, Y)

fig, ax = plt.subplots()
contour = ax.contour(X, Y, Z, levels=np.logspace(0, 3, 20))
ax.clabel(contour, inline=1, fontsize=10)
line, = ax.plot([], [], 'ro-', lw=2)

# Initialize optimizer and starting point
adam = AdamOptimizer()
params = np.array([-4, 4])
path = [params]

def update(frame):
    global params
    grads = himmelblau_grad(*params)
    params = adam.update(params, grads)
    path.append(params)
    line.set_data(*zip(*path))
    return line,

ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.title("Adam Optimization on Himmelblau's Function")
plt.show()
```

Slide 10: Adam's Adaptive Learning Rates

One of Adam's key features is its ability to adapt the learning rate for each parameter. Let's visualize how these learning rates change during training for different parameters in a simple neural network.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate some dummy data
X = np.random.randn(1000, 10)
y = np.random.randn(1000, 1)

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Custom callback to track learning rates
class LearningRateTracker(tf.keras.callbacks.Callback):
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.lr_history = []

    def on_train_batch_end(self, batch, logs=None):
        lr = self.optimizer.learning_rate
        if hasattr(lr, 'numpy'):
            lr = lr.numpy()
        self.lr_history.append(lr)

# Compile the model with Adam
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse')

# Train the model and track learning rates
lr_tracker = LearningRateTracker(optimizer)
history = model.fit(X, y, epochs=50, batch_size=32, callbacks=[lr_tracker], verbose=0)

# Plot the learning rate history
plt.figure(figsize=(10, 6))
plt.plot(lr_tracker.lr_history)
plt.title('Adam Learning Rate During Training')
plt.xlabel('Training Step')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.show()
```

Slide 11: Handling Sparse Gradients with Adam

Adam is particularly effective in handling sparse gradients, which are common in natural language processing tasks. Let's demonstrate this with a simple text classification example using a sparse bag-of-words representation.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Sample text data
texts = [
    "I love this movie", "This movie is great", "Awesome film",
    "Terrible movie", "I hate this film", "Worst movie ever",
    "Neutral opinion", "Average film", "It was okay"
]
labels = [1, 1, 1, 0, 0, 0, 0.5, 0.5, 0.5]

# Create sparse features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=100, batch_size=2, 
                    validation_data=(X_test, y_test), verbose=0)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss with Adam Optimizer')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

Slide 12: Adam with Learning Rate Decay

Adam's performance can sometimes be improved by implementing a learning rate decay schedule. This can help fine-tune the optimization process as training progresses.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate dummy data
X = np.random.randn(1000, 10)
y = np.random.randn(1000, 1)

# Learning rate schedule
def lr_schedule(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Create and compile model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse')

# Train model with lr schedule
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
history = model.fit(X, y, epochs=100, batch_size=32, callbacks=[lr_scheduler], verbose=0)

# Plot learning rate
plt.figure(figsize=(10, 6))
plt.plot(history.history['lr'])
plt.title('Learning Rate Decay')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.show()
```

Slide 13: Hyperparameter Tuning for Adam

Choosing the right hyperparameters for Adam can significantly impact model performance. Let's explore a simple grid search to find optimal Adam parameters.

```python
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
import numpy as np

# Generate dummy data
X = np.random.randn(1000, 10)
y = np.random.randn(1000, 1)

# Define parameter grid
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'beta_1': [0.9, 0.95],
    'beta_2': [0.999, 0.9999]
}

# Grid search function
def grid_search(param_grid, X, y):
    best_loss = float('inf')
    best_params = None
    
    for params in ParameterGrid(param_grid):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(1)
        ])
        optimizer = tf.keras.optimizers.Adam(**params)
        model.compile(optimizer=optimizer, loss='mse')
        
        history = model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        final_loss = history.history['loss'][-1]
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params
    
    return best_params, best_loss

# Perform grid search
best_params, best_loss = grid_search(param_grid, X, y)
print(f"Best parameters: {best_params}")
print(f"Best loss: {best_loss}")
```

Slide 14: Real-world Application: Image Segmentation with Adam

Let's apply Adam to a more complex task: image segmentation using a U-Net architecture. This example demonstrates Adam's effectiveness in training deep convolutional networks.

```python
import tensorflow as tf
import numpy as np

# Define U-Net architecture (simplified)
def unet(input_size=(128, 128, 1)):
    inputs = tf.keras.layers.Input(input_size)
    
    # Encoder
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bridge
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    
    # Decoder
    up4 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv3)
    up4 = tf.keras.layers.concatenate([up4, conv2])
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up4)
    
    up5 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = tf.keras.layers.concatenate([up5, conv1])
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up5)
    
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv5)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile model
model = unet()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
X = np.random.rand(100, 128, 128, 1)
y = np.random.randint(2, size=(100, 128, 128, 1))

# Train model
history = model.fit(X, y, batch_size=16, epochs=10, validation_split=0.2)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('U-Net Training with Adam')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into Adam and optimization algorithms, here are some valuable resources:

1. Original Adam paper: "Adam: A Method for Stochastic Optimization" by Kingma and Ba (2014). Available at: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
2. "An overview of gradient descent optimization algorithms" by Sebastian Ruder (2016). A comprehensive review of various optimization algorithms, including Adam. Available at: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
3. "Decoupled Weight Decay Regularization" by Loshchilov and Hutter (2017). Introduces AdamW, an improvement over Adam. Available at: [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
4. TensorFlow documentation on Adam optimizer: [https://www.tensorflow.org/api\_docs/python/tf/keras/optimizers/Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)
5. PyTorch documentation on Adam optimizer: [https://pytorch.org/docs/stable/generated/torch.optim.Adam.html](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)

These resources provide a mix of theoretical background and practical implementations to further your understanding of Adam and related optimization techniques.

