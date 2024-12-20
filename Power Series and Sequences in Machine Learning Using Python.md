## Power Series and Sequences in Machine Learning Using Python
Slide 1: Introduction to Power Series in Machine Learning

Power series are mathematical tools that represent functions as infinite sums of terms. In machine learning, they're used to approximate complex functions, enabling models to learn and represent intricate patterns in data. This slide introduces the concept and its relevance to AI and ML applications.

```python
import numpy as np
import matplotlib.pyplot as plt

def power_series(x, coefficients):
    return sum(coef * x**i for i, coef in enumerate(coefficients))

x = np.linspace(-1, 1, 100)
coefficients = [1, 1, 0.5, 1/6]  # Example coefficients
y = power_series(x, coefficients)

plt.plot(x, y)
plt.title("Power Series Example")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
```

Slide 2: Taylor Series Expansion

Taylor series is a specific type of power series that approximates functions around a point. In machine learning, Taylor series are used for function approximation and optimization techniques, such as gradient descent.

```python
import sympy as sp

x = sp.Symbol('x')
f = sp.exp(x)  # Example function: e^x
x0 = 0  # Point of expansion
n = 5  # Number of terms

taylor_series = sp.series(f, x, x0, n).removeO()
print(f"Taylor series of e^x around x=0: {taylor_series}")
```

Slide 3: Fourier Series in Signal Processing

Fourier series decompose periodic functions into sums of simple sinusoidal components. In machine learning, they're used for feature extraction, signal processing, and time series analysis.

```python
import numpy as np
import matplotlib.pyplot as plt

def fourier_series(x, n_terms):
    result = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        result += (4 / (n * np.pi)) * np.sin(n * x)
    return result

x = np.linspace(0, 2*np.pi, 1000)
y = fourier_series(x, 10)

plt.plot(x, y)
plt.title("Fourier Series Approximation of Square Wave")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
```

Slide 4: Geometric Series in Neural Networks

Geometric series appear in various aspects of neural networks, such as in the analysis of recurrent neural networks and in understanding the behavior of certain activation functions.

```python
import numpy as np
import matplotlib.pyplot as plt

def geometric_series(r, n):
    return np.sum([r**i for i in range(n)])

r_values = np.linspace(0, 0.99, 100)
n_terms = [10, 50, 100]

for n in n_terms:
    plt.plot(r_values, [geometric_series(r, n) for r in r_values], label=f'n={n}')

plt.title("Geometric Series Sum for Different n")
plt.xlabel("Common Ratio (r)")
plt.ylabel("Sum")
plt.legend()
plt.show()
```

Slide 5: Power Series in Optimization Algorithms

Power series expansions are used in optimization algorithms like Newton's method, which is the basis for many machine learning optimization techniques.

```python
import numpy as np

def newton_method(f, f_prime, x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        x = x - fx / f_prime(x)
    return x

# Example: Finding the square root of 2
f = lambda x: x**2 - 2
f_prime = lambda x: 2*x

root = newton_method(f, f_prime, 1.0)
print(f"Square root of 2: {root}")
```

Slide 6: Sequences in Time Series Forecasting

Sequences play a crucial role in time series forecasting, where models learn to predict future values based on historical sequences of data.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate sample time series data
t = np.linspace(0, 100, 1000)
y = np.sin(0.1 * t) + np.random.normal(0, 0.1, 1000)

# Prepare sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

seq_length = 50
X = create_sequences(y[:-1], seq_length)
y = y[seq_length:]

# Split data and create model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
```

Slide 7: Convergence of Series in Deep Learning

Understanding the convergence of series is crucial in deep learning, especially for analyzing the behavior of neural networks during training and ensuring stable learning.

```python
import numpy as np
import matplotlib.pyplot as plt

def series_convergence(terms, n_iterations):
    partial_sums = np.cumsum(terms)
    return partial_sums[:n_iterations]

# Example: Convergence of 1/n^2
n = np.arange(1, 1001)
terms = 1 / (n**2)
convergence = series_convergence(terms, 1000)

plt.plot(n, convergence)
plt.title("Convergence of Series 1/n^2")
plt.xlabel("Number of Terms")
plt.ylabel("Partial Sum")
plt.xscale('log')
plt.show()

print(f"Final sum: {convergence[-1]}")
print(f"Analytical limit (π^2/6): {np.pi**2/6}")
```

Slide 8: Power Series in Activation Functions

Power series expansions are used to approximate and analyze activation functions in neural networks, helping to understand their behavior and choose appropriate functions for specific tasks.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_taylor(x, n_terms):
    coeffs = [0.5, 0.25, 0, -1/48, 0, 1/480, 0, -17/80640, 0, 31/1451520]
    return sum(coef * x**i for i, coef in enumerate(coeffs[:n_terms]))

x = np.linspace(-5, 5, 100)
y_true = sigmoid(x)
y_taylor_3 = sigmoid_taylor(x, 3)
y_taylor_5 = sigmoid_taylor(x, 5)

plt.plot(x, y_true, label='True Sigmoid')
plt.plot(x, y_taylor_3, label='Taylor (3 terms)')
plt.plot(x, y_taylor_5, label='Taylor (5 terms)')
plt.title("Sigmoid Function and Its Taylor Approximations")
plt.legend()
plt.show()
```

Slide 9: Recurrent Sequences in RNNs

Recurrent Neural Networks (RNNs) process sequences of data by maintaining an internal state, which can be viewed as a form of recurrent sequence.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.models import Sequential

# Generate a simple sequence
sequence = np.array([i/10 for i in range(100)])
X = sequence.reshape(-1, 1, 1)

# Create and train an RNN model
model = Sequential([
    SimpleRNN(10, input_shape=(1, 1), activation='tanh'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X[:-1], X[1:], epochs=100, verbose=0)

# Generate predictions
start = X[-1]
generated = [start]
for _ in range(10):
    next_value = model.predict(generated[-1].reshape(1, 1, 1))
    generated.append(next_value)

print("Generated sequence:")
print([x[0][0] for x in generated])
```

Slide 10: Power Series in Kernel Methods

Kernel methods in machine learning often involve power series expansions of kernel functions, enabling the transformation of data into higher-dimensional spaces for improved separability.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate non-linearly separable data
np.random.seed(0)
X = np.random.randn(200, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# Create and train polynomial kernel SVM
poly_svm = make_pipeline(PolynomialFeatures(degree=3), SVC(kernel='linear'))
poly_svm.fit(X, y)

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
Z = poly_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
plt.title("SVM with Polynomial Kernel")
plt.show()
```

Slide 11: Convergence Analysis in Gradient Descent

Understanding the convergence of optimization algorithms like gradient descent is crucial for training machine learning models effectively.

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, df, x0, learning_rate, n_iterations):
    x = x0
    trajectory = [x]
    for _ in range(n_iterations):
        x = x - learning_rate * df(x)
        trajectory.append(x)
    return np.array(trajectory)

# Example function and its derivative
f = lambda x: x**2
df = lambda x: 2*x

x0 = 5.0
learning_rate = 0.1
n_iterations = 50

trajectory = gradient_descent(f, df, x0, learning_rate, n_iterations)

plt.plot(range(n_iterations + 1), trajectory, 'bo-')
plt.title("Convergence of Gradient Descent")
plt.xlabel("Iteration")
plt.ylabel("x")
plt.show()

print(f"Final value: {trajectory[-1]}")
```

Slide 12: Sequence-to-Sequence Models

Sequence-to-sequence models, based on recurrent or transformer architectures, process input sequences to generate output sequences, crucial for tasks like machine translation.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

# Define model architecture
encoder_inputs = Input(shape=(None, 100))  # 100-dim input sequences
encoder = LSTM(64, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, 100))
decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(100, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

print(model.summary())
```

Slide 13: Power Series in Attention Mechanisms

Attention mechanisms, which have revolutionized natural language processing and other sequence-based tasks, can be viewed as a form of learnable power series expansion.

```python
import tensorflow as tf
import numpy as np

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
temp_k = tf.constant([[10,0,0],
                      [0,10,0],
                      [0,0,10],
                      [0,0,10]], dtype=tf.float32)  # (4, 3)

temp_v = tf.constant([[1,0],
                      [10,0],
                      [100,5],
                      [1000,6]], dtype=tf.float32)  # (4, 2)

temp_q = tf.constant([[0,0,10]], dtype=tf.float32)  # (1, 3)

output, attention_weights = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
print("Attention output shape:", output.shape)
print("Attention weights:", attention_weights.numpy())
```

Slide 14: Series Expansion in Model Interpretability

Series expansions can be used to approximate complex machine learning models, aiding in their interpretation and analysis.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Generate sample data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel()

# Train a decision tree
tree = DecisionTreeRegressor(max_depth=5)
tree.fit(X, y)

# Generate predictions
y_pred = tree.predict(X)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='b', label='True function')
plt.plot(X, y_pred, c='r', label='Decision tree approximation')
plt.title("Decision Tree Approximation of Sine Function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Analyze feature importances
importances = tree.feature_importances_
print("Feature importances:", importances)
```

Slide 15: Additional Resources

For further exploration of power series and sequences in machine learning and AI, consider the following resources:

1. "On the Power of Neural Networks to Approximate Functions" by Leshno et al. (1993) arXiv: [https://arxiv.org/abs/cs/9901012](https://arxiv.org/abs/cs/9901012)
2. "Universal Approximation Bounds for Superpositions of a Sigmoidal Function" by Barron (1993) IEEE Transactions on Information Theory
3. "Deep Learning" by Goodfellow, Bengio, and Courville (2016) Book: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
4. "Attention Is All You Need" by Vaswani et al. (2017) arXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

These resources provide in-depth discussions on the theoretical foundations and practical applications of series and sequences in modern machine learning and AI techniques.

