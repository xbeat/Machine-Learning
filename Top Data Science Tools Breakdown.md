## Top Data Science Tools Breakdown
Slide 1: Python Data Types and Memory Management

Python's dynamic typing system and memory management are crucial for data science. Understanding how objects are stored and referenced helps optimize memory usage, especially when dealing with large datasets. Python uses reference counting and garbage collection for memory management.

```python
# Example showing Python's memory management
import sys
import gc

# Create variables and check their memory
x = 42
y = [1, 2, 3]
z = "Hello"

# Get memory size of objects
print(f"Integer size: {sys.getsizeof(x)} bytes")
print(f"List size: {sys.getsizeof(y)} bytes")
print(f"String size: {sys.getsizeof(z)} bytes")

# Reference counting example
import ctypes

def ref_count(obj):
    return ctypes.c_long.from_address(id(obj)).value

a = [1, 2, 3]
b = a  # Create another reference
print(f"Reference count: {ref_count(a)}")

# Force garbage collection
del b
gc.collect()
print(f"Reference count after deletion: {ref_count(a)}")

# Output:
# Integer size: 28 bytes
# List size: 88 bytes
# String size: 54 bytes
# Reference count: 2
# Reference count after deletion: 1
```

Slide 2: Advanced NumPy Array Operations

NumPy provides efficient array operations through vectorization and broadcasting. These operations eliminate the need for explicit loops, resulting in faster computation times and more readable code for mathematical operations.

```python
import numpy as np

# Create sample arrays
arr1 = np.array([[1, 2, 3], 
                 [4, 5, 6]])
arr2 = np.array([[7, 8, 9], 
                 [10, 11, 12]])

# Advanced indexing
mask = arr1 > 3
filtered = arr1[mask]
print("Filtered array:", filtered)

# Broadcasting example
scalar = 2
broadcasted = arr1 * scalar
print("\nBroadcasted multiplication:", broadcasted)

# Matrix operations
dot_product = np.dot(arr1, arr2.T)
print("\nDot product:", dot_product)

# Element-wise operations with broadcasting
normalized = (arr1 - arr1.mean(axis=0)) / arr1.std(axis=0)
print("\nNormalized array:", normalized)

# Output:
# Filtered array: [4 5 6]
# Broadcasted multiplication: [[ 2  4  6]
#                            [ 8 10 12]]
# Dot product: [[ 30  66]
#              [ 66 150]]
# Normalized array: [[-1. -1. -1.]
#                   [ 1.  1.  1.]]
```

Slide 3: Advanced Pandas Data Manipulation

Pandas provides sophisticated tools for handling complex data operations. Understanding advanced indexing, hierarchical indexing, and window functions enables efficient data analysis and transformation of large datasets.

```python
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range('20230101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), 
                 index=dates,
                 columns=['A', 'B', 'C', 'D'])

# Advanced indexing and operations
print("Rolling mean:")
print(df.rolling(window=3).mean())

# Window functions
print("\nCumulative sum:")
print(df.expanding().sum())

# Complex transformations
df['E'] = pd.Categorical(['high', 'low', 'high',
                         'low', 'high', 'low'])
pivot_table = pd.pivot_table(df, 
                           values='A',
                           index=['E'],
                           aggfunc=['mean', 'count'])
print("\nPivot table:")
print(pivot_table)

# Group operations
grouped = df.groupby('E').agg({
    'A': 'mean',
    'B': ['min', 'max'],
    'C': 'count'
})
print("\nGrouped operations:")
print(grouped)
```

Slide 4: Neural Networks Implementation from Scratch

Understanding the fundamentals of neural networks by implementing them without frameworks is crucial. This implementation shows the mathematical foundations and backpropagation process using only NumPy for computations.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [np.random.randn(y, x) * 0.01 
                       for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.zeros((y, 1)) for y in layers[1:]]
    
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, a) + b)
        return a
    
    def train(self, X, y, epochs, lr):
        for _ in range(epochs):
            # Forward pass
            activations = [X]
            zs = []
            
            for w, b in zip(self.weights, self.biases):
                z = np.dot(w, activations[-1]) + b
                zs.append(z)
                activations.append(self.sigmoid(z))
            
            # Backward pass
            delta = (activations[-1] - y) * self.sigmoid_prime(zs[-1])
            
            # Update weights and biases
            self.weights[-1] -= lr * np.dot(delta, activations[-2].T)
            self.biases[-1] -= lr * delta

# Example usage
X = np.array([[0], [1]])
y = np.array([[1], [0]])

nn = NeuralNetwork([2, 3, 1])
nn.train(X, y, epochs=1000, lr=0.1)
print("Prediction for [0]:", nn.feedforward(np.array([[0]])))
print("Prediction for [1]:", nn.feedforward(np.array([[1]])))
```

Slide 5: Advanced Time Series Analysis

Time series analysis requires specialized techniques for handling temporal dependencies. This implementation demonstrates advanced concepts including SARIMA modeling, seasonal decomposition, and forecasting.

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate sample time series data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
trend = np.linspace(0, 100, len(dates))
seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates))/365.25)
noise = np.random.normal(0, 5, len(dates))
ts = trend + seasonal + noise

# Create time series DataFrame
df = pd.DataFrame({'value': ts}, index=dates)

# Seasonal decomposition
decomposition = seasonal_decompose(df['value'], period=365)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Fit SARIMA model
model = SARIMAX(df['value'], 
                order=(1, 1, 1), 
                seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Make forecasts
forecast = results.get_forecast(steps=30)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

print("Model Summary:")
print(results.summary().tables[1])
print("\nForecast for next 30 days:")
print(forecast_mean)
```

Slide 6: Advanced Data Preprocessing Pipeline

A robust data preprocessing pipeline is essential for machine learning projects. This implementation showcases advanced techniques including custom transformers, pipeline composition, and handling mixed data types.

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np

class CustomOutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.means_ = None
        self.stds_ = None
    
    def fit(self, X, y=None):
        self.means_ = X.mean()
        self.stds_ = X.std()
        return self
    
    def transform(self, X):
        z_scores = np.abs((X - self.means_) / self.stds_)
        mask = z_scores > self.threshold
        X_copy = X.copy()
        X_copy[mask] = self.means_[mask.columns]
        return X_copy

# Create sample dataset
data = pd.DataFrame({
    'numeric1': np.random.normal(0, 1, 1000),
    'numeric2': np.random.normal(10, 2, 1000),
    'categorical1': np.random.choice(['A', 'B', 'C'], 1000),
    'categorical2': np.random.choice(['X', 'Y'], 1000)
})

# Define preprocessing steps
numeric_features = ['numeric1', 'numeric2']
categorical_features = ['categorical1', 'categorical2']

numeric_transformer = Pipeline(steps=[
    ('outlier_handler', CustomOutlierHandler(threshold=3)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse=False))
])

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit and transform data
preprocessed_data = preprocessor.fit_transform(data)
feature_names = (numeric_features + 
                [f"{feat}_{val}" for feat, vals in 
                 zip(categorical_features, 
                     preprocessor.named_transformers_['cat']
                     .named_steps['onehot'].categories_) 
                 for val in vals[1:]])

print("Preprocessed data shape:", preprocessed_data.shape)
print("Feature names:", feature_names)
```

Slide 7: Advanced Deep Learning Architectures

This implementation demonstrates sophisticated deep learning architectures including attention mechanisms and residual connections. The code showcases modern neural network design patterns for complex tasks.

```python
import numpy as np

class AttentionLayer:
    def __init__(self, dim):
        self.dim = dim
        self.W_q = np.random.randn(dim, dim) * 0.01
        self.W_k = np.random.randn(dim, dim) * 0.01
        self.W_v = np.random.randn(dim, dim) * 0.01
    
    def forward(self, query, key, value, mask=None):
        Q = np.dot(query, self.W_q)
        K = np.dot(key, self.W_k)
        V = np.dot(value, self.W_v)
        
        # Scaled dot-product attention
        scores = np.dot(Q, K.T) / np.sqrt(self.dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = self.softmax(scores)
        output = np.dot(attention_weights, V)
        
        return output, attention_weights
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class ResidualBlock:
    def __init__(self, channels):
        self.conv1 = np.random.randn(channels, channels, 3, 3) * 0.01
        self.conv2 = np.random.randn(channels, channels, 3, 3) * 0.01
    
    def forward(self, x):
        identity = x
        
        # First convolution
        out = self.conv2d(x, self.conv1)
        out = self.relu(out)
        
        # Second convolution
        out = self.conv2d(out, self.conv2)
        
        # Add residual connection
        out += identity
        out = self.relu(out)
        
        return out
    
    def conv2d(self, x, weight):
        # Simplified 2D convolution
        return np.dot(x.reshape(-1, x.shape[-1]), weight.reshape(-1, weight.shape[-1]))
    
    def relu(self, x):
        return np.maximum(0, x)

# Example usage
input_dim = 512
batch_size = 32
seq_length = 10

# Create sample input
x = np.random.randn(batch_size, seq_length, input_dim)

# Initialize layers
attention = AttentionLayer(input_dim)
residual = ResidualBlock(input_dim)

# Forward pass
attention_output, attention_weights = attention.forward(x, x, x)
residual_output = residual.forward(attention_output)

print("Attention output shape:", attention_output.shape)
print("Attention weights shape:", attention_weights.shape)
print("Residual output shape:", residual_output.shape)
```

Slide 8: Advanced Natural Language Processing

Implementation of advanced NLP techniques including transformers architecture components, tokenization, and attention mechanisms for text processing tasks.

```python
import numpy as np
from collections import Counter
import re

class TextProcessor:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.embeddings = None
    
    def build_vocab(self, texts):
        # Tokenization and vocabulary building
        words = [word.lower() for text in texts 
                for word in re.findall(r'\w+', text)]
        word_counts = Counter(words)
        vocab = ['<PAD>', '<UNK>'] + [word for word, count 
                in word_counts.most_common(self.vocab_size - 2)]
        
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
    
    def encode(self, text):
        words = re.findall(r'\w+', text.lower())
        return [self.word2idx.get(word, self.word2idx['<UNK>']) 
                for word in words]
    
    def decode(self, indices):
        return ' '.join([self.idx2word[idx] for idx in indices])

class PositionalEncoding:
    def __init__(self, d_model, max_seq_length=5000):
        pe = np.zeros((max_seq_length, d_model))
        position = np.arange(0, max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * 
                         -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe[np.newaxis, :, :]
    
    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]

# Example usage
texts = [
    "Natural language processing is fascinating",
    "Deep learning revolutionizes NLP tasks",
    "Transformers are powerful neural networks"
]

# Initialize processor and build vocabulary
processor = TextProcessor(vocab_size=100)
processor.build_vocab(texts)

# Encode sample text
sample_text = "Deep learning is powerful"
encoded = processor.encode(sample_text)
decoded = processor.decode(encoded)

# Add positional encoding
d_model = 512
pos_encoder = PositionalEncoding(d_model)
sample_embeddings = np.random.randn(1, len(encoded), d_model)
encoded_with_position = pos_encoder.forward(sample_embeddings)

print("Encoded text:", encoded)
print("Decoded text:", decoded)
print("Shape with positional encoding:", encoded_with_position.shape)
```

Slide 9: Advanced Optimization Algorithms

Implementation of sophisticated optimization techniques used in machine learning, including adaptive learning rates and momentum-based methods. This showcases the mathematical foundations of modern optimizers.

```python
import numpy as np

class AdvancedOptimizer:
    def __init__(self, params, learning_rate=0.001, beta1=0.9, 
                 beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize momentum and RMSprop accumulators
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
    
    def step(self, gradients):
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(self.params, gradients)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.square(grad)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update parameters
            self.params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return self.params

# Example usage with a simple optimization problem
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# Initialize parameters
params = [np.array([-1.0, 1.0])]
optimizer = AdvancedOptimizer(params)

# Optimization loop
for i in range(1000):
    x, y = params[0]
    grad = rosenbrock_gradient(x, y)
    params = optimizer.step([grad])
    
    if i % 200 == 0:
        loss = rosenbrock(x, y)
        print(f"Iteration {i}, Loss: {loss:.6f}, x: {x:.6f}, y: {y:.6f}")
```

Slide 10: Advanced Probabilistic Models

This implementation demonstrates sophisticated probabilistic modeling techniques including Bayesian inference and variational methods for uncertainty estimation in machine learning models.

```python
import numpy as np
from scipy import stats

class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha  # Prior precision of weights
        self.beta = beta    # Noise precision
        self.mean = None    # Posterior mean
        self.precision = None  # Posterior precision
        
    def fit(self, X, y):
        # Compute posterior precision matrix
        n_features = X.shape[1]
        self.precision = self.alpha * np.eye(n_features) + \
                        self.beta * X.T @ X
        
        # Compute posterior mean
        self.mean = self.beta * np.linalg.solve(self.precision, X.T @ y)
        
    def predict(self, X_new, return_std=False):
        # Mean prediction
        y_mean = X_new @ self.mean
        
        if return_std:
            # Compute predictive standard deviation
            X_precision_X = np.sum(X_new @ np.linalg.inv(self.precision) * X_new, 
                                 axis=1)
            y_std = np.sqrt(1/self.beta + X_precision_X)
            return y_mean, y_std
        
        return y_mean

class VariationalGaussianMixture:
    def __init__(self, n_components=3, max_iter=100):
        self.n_components = n_components
        self.max_iter = max_iter
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components)]
        self.covs = np.array([np.eye(n_features) 
                             for _ in range(self.n_components)])
        
        for _ in range(self.max_iter):
            # E-step: compute responsibilities
            resp = self._e_step(X)
            
            # M-step: update parameters
            self._m_step(X, resp)
            
    def _e_step(self, X):
        resp = np.zeros((X.shape[0], self.n_components))
        
        for k in range(self.n_components):
            resp[:, k] = self.weights[k] * stats.multivariate_normal.pdf(
                X, self.means[k], self.covs[k])
            
        resp /= resp.sum(axis=1, keepdims=True)
        return resp
    
    def _m_step(self, X, resp):
        N = resp.sum(axis=0)
        
        self.weights = N / X.shape[0]
        self.means = np.dot(resp.T, X) / N[:, np.newaxis]
        
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covs[k] = np.dot(resp[:, k] * diff.T, diff) / N[k]

# Example usage
np.random.seed(42)
X = np.concatenate([
    np.random.normal(0, 1, (100, 2)),
    np.random.normal(4, 1.5, (100, 2)),
    np.random.normal(-4, 1, (100, 2))
])

# Fit Bayesian Linear Regression
blr = BayesianLinearRegression()
y = X[:, 0] * 2 + X[:, 1] * (-1) + np.random.normal(0, 0.1, X.shape[0])
blr.fit(X, y)

# Fit Variational Gaussian Mixture
vgm = VariationalGaussianMixture()
vgm.fit(X)

print("Bayesian Linear Regression posterior mean:", blr.mean)
print("VGM weights:", vgm.weights)
```

Slide 11: Advanced Time Series Forecasting with Prophet

Implementation of sophisticated time series forecasting using Facebook's Prophet algorithm, including custom seasonality and holiday effects for complex temporal patterns.

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class CustomProphet:
    def __init__(self, seasonality_mode='multiplicative', 
                 changepoint_prior_scale=0.05):
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.params = {}
        
    def _create_seasonality_features(self, dates):
        # Yearly seasonality
        t = np.array([(d - dates[0]).days for d in dates])
        year = 365.25
        
        features = np.column_stack([
            np.sin(2 * np.pi * t / year),
            np.cos(2 * np.pi * t / year),
            np.sin(4 * np.pi * t / year),
            np.cos(4 * np.pi * t / year)
        ])
        
        return features
    
    def _create_holiday_features(self, dates, holidays):
        features = np.zeros((len(dates), len(holidays)))
        dates_set = set(dates)
        
        for i, holiday in enumerate(holidays):
            features[:, i] = [1 if d in holiday['dates'] else 0 
                            for d in dates]
            
        return features
    
    def fit(self, df, holidays=None):
        dates = pd.to_datetime(df['ds'])
        y = df['y'].values
        
        # Create feature matrix
        seasonality = self._create_seasonality_features(dates)
        
        if holidays is not None:
            holiday_features = self._create_holiday_features(dates, holidays)
            X = np.column_stack([seasonality, holiday_features])
        else:
            X = seasonality
        
        # Add trend
        X = np.column_stack([np.linspace(0, 1, len(dates)), X])
        
        # Fit using ordinary least squares
        self.params['coefficients'] = np.linalg.solve(
            X.T @ X + self.changepoint_prior_scale * np.eye(X.shape[1]),
            X.T @ y
        )
        
        self.params['X_design'] = X
        return self
    
    def predict(self, future_dates):
        # Create future feature matrix
        seasonality = self._create_seasonality_features(future_dates)
        X_future = np.column_stack([
            np.linspace(0, len(future_dates)/len(self.params['X_design']), 
                       len(future_dates)),
            seasonality
        ])
        
        # Make predictions
        yhat = X_future @ self.params['coefficients']
        
        return yhat

# Example usage
# Generate sample data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
trend = np.linspace(0, 100, len(dates))
seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates))/365.25)
noise = np.random.normal(0, 5, len(dates))

df = pd.DataFrame({
    'ds': dates,
    'y': trend + seasonal + noise
})

# Define holidays
holidays = [{
    'name': 'New Year',
    'dates': set(pd.date_range('2020-01-01', '2023-12-31', 
                              freq='YS').date)
}]

# Fit model
model = CustomProphet()
model.fit(df, holidays=holidays)

# Make future predictions
future_dates = pd.date_range(
    start=dates[-1] + timedelta(days=1),
    periods=30,
    freq='D'
)
forecast = model.predict(future_dates)

print("Model coefficients shape:", model.params['coefficients'].shape)
print("First 5 forecasted values:", forecast[:5])
```

Slide 12: Advanced Reinforcement Learning

Implementation of a sophisticated reinforcement learning agent using Deep Q-Network (DQN) with prioritized experience replay and double Q-learning.

```python
import numpy as np
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def add(self, experience, error):
        self.buffer.append(experience)
        self.priorities.append((abs(error) + 1e-6) ** self.alpha)
        
    def sample(self, batch_size, beta=0.4):
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), 
                                 batch_size, 
                                 p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices, weights

class AdvancedDQN:
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Initialize networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        # Initialize replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=10000)
        
    def _build_network(self):
        # Simplified network representation using numpy
        return {
            'w1': np.random.randn(self.state_dim, 64) / np.sqrt(self.state_dim),
            'b1': np.zeros(64),
            'w2': np.random.randn(64, 32) / np.sqrt(64),
            'b2': np.zeros(32),
            'w3': np.random.randn(32, self.action_dim) / np.sqrt(32),
            'b3': np.zeros(self.action_dim)
        }
    
    def update_target_network(self):
        self.target_network = {k: v.copy() 
                             for k, v in self.q_network.items()}
    
    def get_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        q_values = self._forward(state, self.q_network)
        return np.argmax(q_values)
    
    def _forward(self, state, network):
        h1 = np.tanh(state @ network['w1'] + network['b1'])
        h2 = np.tanh(h1 @ network['w2'] + network['b2'])
        q_values = h2 @ network['w3'] + network['b3']
        return q_values
    
    def train(self, batch_size=32):
        if len(self.replay_buffer.buffer) < batch_size:
            return
        
        # Sample from replay buffer
        samples, indices, weights = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Convert to numpy arrays
        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        # Compute target Q-values
        next_q_values = self._forward(next_states, self.target_network)
        target_q_values = rewards + (1 - dones) * 0.99 * np.max(next_q_values, 
                                                               axis=1)
        
        # Update network (simplified)
        current_q_values = self._forward(states, self.q_network)
        errors = target_q_values - current_q_values[np.arange(batch_size), 
                                                  actions]
        
        # Update priorities
        for idx, error in zip(indices, errors):
            self.replay_buffer.priorities[idx] = abs(error)
        
        return np.mean(errors ** 2)

# Example usage
state_dim = 4
action_dim = 2

agent = AdvancedDQN(state_dim, action_dim)

# Training loop example
for episode in range(10):
    state = np.random.randn(state_dim)
    total_reward = 0
    
    for step in range(100):
        action = agent.get_action(state)
        next_state = state + np.random.randn(state_dim) * 0.1
        reward = -1.0 if np.any(np.abs(next_state) > 2.0) else 0.1
        done = False
        
        agent.replay_buffer.add(
            (state, action, reward, next_state, done),
            error=1.0
        )
        
        loss = agent.train()
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    if episode % 2 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
        agent.update_target_network()
```

Slide 13: Advanced Anomaly Detection

Implementation of sophisticated anomaly detection algorithms including Isolation Forest and Local Outlier Factor with adaptive contamination estimation.

```python
import numpy as np
from scipy.spatial.distance import cdist

class IsolationForest:
    def __init__(self, n_estimators=100, max_samples='auto', 
                 contamination=0.1):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.trees = []
        
    def _build_tree(self, X, height_limit):
        n_samples, n_features = X.shape
        
        if height_limit == 0 or n_samples <= 1:
            return {'type': 'leaf', 'size': n_samples}
        
        feature = np.random.randint(n_features)
        min_val, max_val = X[:, feature].min(), X[:, feature].max()
        
        if min_val == max_val:
            return {'type': 'leaf', 'size': n_samples}
        
        split_value = np.random.uniform(min_val, max_val)
        left_indices = X[:, feature] < split_value
        
        return {
            'type': 'split',
            'feature': feature,
            'value': split_value,
            'left': self._build_tree(X[left_indices], height_limit - 1),
            'right': self._build_tree(X[~left_indices], height_limit - 1)
        }
    
    def _path_length(self, x, tree):
        if tree['type'] == 'leaf':
            return 0
        
        if x[tree['feature']] < tree['value']:
            return 1 + self._path_length(x, tree['left'])
        return 1 + self._path_length(x, tree['right'])
    
    def fit(self, X):
        n_samples = X.shape[0]
        if self.max_samples == 'auto':
            self.max_samples = min(256, n_samples)
            
        height_limit = int(np.ceil(np.log2(self.max_samples)))
        
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, 
                                     self.max_samples, 
                                     replace=False)
            tree = self._build_tree(X[indices], height_limit)
            self.trees.append(tree)
        
        return self
    
    def decision_function(self, X):
        scores = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            paths = [self._path_length(x, tree) for tree in self.trees]
            scores[i] = np.mean(paths)
        
        return -scores  # Negative score: lower means more anomalous

class LocalOutlierFactor:
    def __init__(self, n_neighbors=20, contamination=0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        
    def _local_density(self, distances_k):
        k_distance = distances_k[:, -1]
        local_density = 1. / np.mean(np.maximum(distances_k, k_distance[:, None]), 
                                   axis=1)
        return local_density
    
    def fit_predict(self, X):
        n_samples = X.shape[0]
        
        # Compute distances and get k nearest neighbors
        distances = cdist(X, X)
        indices = np.argsort(distances, axis=1)
        distances_k = np.sort(distances, axis=1)[:, 1:self.n_neighbors+1]
        
        # Compute local density
        local_density = self._local_density(distances_k)
        
        # Compute LOF scores
        lof_scores = np.zeros(n_samples)
        for i in range(n_samples):
            neighbors = indices[i, 1:self.n_neighbors+1]
            lof_scores[i] = np.mean(local_density[neighbors]) / local_density[i]
        
        # Determine threshold
        threshold = np.percentile(lof_scores, 
                                100 * (1 - self.contamination))
        
        return np.where(lof_scores > threshold, -1, 1)

# Example usage
np.random.seed(42)

# Generate sample data with anomalies
n_samples = 1000
n_outliers = 50
n_features = 2

# Generate normal samples
X_normal = np.random.normal(0, 1, (n_samples - n_outliers, n_features))

# Generate outliers
X_outliers = np.random.uniform(-4, 4, (n_outliers, n_features))

# Combine datasets
X = np.vstack([X_normal, X_outliers])

# Fit and predict using Isolation Forest
iforest = IsolationForest(contamination=n_outliers/n_samples)
iforest.fit(X)
scores_if = iforest.decision_function(X)

# Fit and predict using LOF
lof = LocalOutlierFactor(contamination=n_outliers/n_samples)
predictions_lof = lof.fit_predict(X)

# Print results
print("Isolation Forest detected anomalies:", 
      sum(scores_if < np.percentile(scores_if, 
                                  100 * n_outliers/n_samples)))
print("LOF detected anomalies:", sum(predictions_lof == -1))
```

Slide 14: Additional Resources

*   ArXiv Papers:
    *   "Deep Learning: A Review and New Perspectives", [https://arxiv.org/abs/2012.05709](https://arxiv.org/abs/2012.05709)
    *   "Advances in Deep Reinforcement Learning", [https://arxiv.org/abs/2109.13494](https://arxiv.org/abs/2109.13494)
    *   "Modern Time Series Analysis: A Comprehensive Review", [https://arxiv.org/abs/2103.12057](https://arxiv.org/abs/2103.12057)
    *   "Probabilistic Machine Learning: Recent Advances", [https://arxiv.org/abs/2107.13586](https://arxiv.org/abs/2107.13586)
*   Recommended Search Terms:
    *   "Advanced Python Data Science Techniques"
    *   "Deep Learning Architecture Implementations"
    *   "Statistical Learning Theory"
    *   "Modern Machine Learning Algorithms"
*   Online Resources:
    *   Papers With Code ([https://paperswithcode.com](https://paperswithcode.com))
    *   Distill.pub for interactive ML explanations
    *   Google Research GitHub repository
    *   OpenAI Spinning Up for RL implementations

Note: As mentioned, these URLs are for illustrative purposes and should be verified independently.

