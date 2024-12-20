## Exponentially Weighted Moving Averages in Deep Learning with Python
Slide 1: Introduction to Exponentially Weighted Moving Averages (EWMA) in Deep Learning

Exponentially Weighted Moving Averages (EWMA) are a powerful technique used in deep learning for various purposes, including optimization, parameter updates, and noise reduction. This presentation will explore the concept, implementation, and applications of EWMA in deep learning using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
data = np.random.randn(1000)

# Calculate EWMA
def ewma(data, alpha):
    ewma = np.zeros_like(data)
    ewma[0] = data[0]
    for i in range(1, len(data)):
        ewma[i] = alpha * data[i] + (1 - alpha) * ewma[i-1]
    return ewma

# Plot original data and EWMA
plt.figure(figsize=(12, 6))
plt.plot(data, label='Original Data')
plt.plot(ewma(data, 0.1), label='EWMA (α=0.1)')
plt.legend()
plt.title('EWMA Example')
plt.show()
```

Slide 2: Understanding EWMA

EWMA is a statistical technique that assigns exponentially decreasing weights to older observations. In deep learning, it's used to compute moving averages of parameters, gradients, or other quantities. The key feature of EWMA is its ability to give more importance to recent data while still considering historical information.

```python
def ewma_formula(current_value, previous_ewma, alpha):
    return alpha * current_value + (1 - alpha) * previous_ewma

# Example usage
current_value = 10
previous_ewma = 8
alpha = 0.2

new_ewma = ewma_formula(current_value, previous_ewma, alpha)
print(f"New EWMA: {new_ewma}")
```

Slide 3: EWMA in Gradient Descent

In deep learning, EWMA is often used in optimization algorithms like Adam and RMSprop. It helps to smooth out noisy gradients and accelerate convergence. Here's a simple implementation of gradient descent with EWMA:

```python
def gradient_descent_with_ewma(X, y, learning_rate, num_iterations, alpha):
    m, n = X.shape
    theta = np.zeros(n)
    v = np.zeros(n)  # EWMA of gradients
    
    for _ in range(num_iterations):
        h = X.dot(theta)
        gradient = X.T.dot(h - y) / m
        v = alpha * gradient + (1 - alpha) * v
        theta -= learning_rate * v
    
    return theta

# Example usage
X = np.random.randn(1000, 5)
y = np.random.randn(1000)
theta = gradient_descent_with_ewma(X, y, learning_rate=0.01, num_iterations=1000, alpha=0.9)
print("Optimized theta:", theta)
```

Slide 4: EWMA in Adam Optimizer

Adam (Adaptive Moment Estimation) is a popular optimization algorithm that uses EWMA for both the first and second moments of the gradients. Here's a simplified implementation:

```python
def adam_optimizer(params, grads, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = [np.zeros_like(param) for param in params]
    v = [np.zeros_like(param) for param in params]
    t = 0
    
    while True:
        t += 1
        for i, (param, grad) in enumerate(zip(params, grads)):
            m[i] = beta1 * m[i] + (1 - beta1) * grad
            v[i] = beta2 * v[i] + (1 - beta2) * (grad ** 2)
            
            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)
            
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        yield params

# Example usage
params = [np.random.randn(10, 10) for _ in range(3)]
grads = [np.random.randn(10, 10) for _ in range(3)]
optimizer = adam_optimizer(params, grads, learning_rate=0.001)
updated_params = next(optimizer)
```

Slide 5: EWMA for Feature Scaling

EWMA can be used for online feature scaling in deep learning, especially when dealing with streaming data. This technique helps normalize features dynamically:

```python
class EWMAScaler:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.mean = None
        self.variance = None
    
    def update(self, x):
        if self.mean is None:
            self.mean = x
            self.variance = np.zeros_like(x)
        else:
            diff = x - self.mean
            incr = self.alpha * diff
            self.mean += incr
            self.variance = (1 - self.alpha) * (self.variance + diff * incr)
    
    def transform(self, x):
        return (x - self.mean) / (np.sqrt(self.variance) + 1e-8)

# Example usage
scaler = EWMAScaler()
data_stream = np.random.randn(1000, 5)
scaled_data = []

for sample in data_stream:
    scaler.update(sample)
    scaled_sample = scaler.transform(sample)
    scaled_data.append(scaled_sample)

scaled_data = np.array(scaled_data)
print("Scaled data shape:", scaled_data.shape)
```

Slide 6: EWMA in Time Series Forecasting

EWMA is widely used in time series forecasting. Here's an example of using EWMA for simple time series prediction:

```python
import pandas as pd

def ewma_forecast(data, alpha, forecast_horizon):
    ewma = pd.Series.ewm(data, alpha=alpha).mean()
    forecast = [ewma.iloc[-1]] * forecast_horizon
    return pd.Series(forecast, index=pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon))

# Generate sample time series data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.cumsum(np.random.randn(len(dates))) + 100
ts_data = pd.Series(values, index=dates)

# Forecast next 30 days
forecast = ewma_forecast(ts_data, alpha=0.3, forecast_horizon=30)

# Plot results
plt.figure(figsize=(12, 6))
ts_data.plot(label='Original Data')
forecast.plot(label='EWMA Forecast')
plt.legend()
plt.title('EWMA Time Series Forecast')
plt.show()
```

Slide 7: EWMA for Noise Reduction in Sensor Data

EWMA can be effective in reducing noise in sensor data, which is crucial in many deep learning applications involving IoT devices. Here's an example:

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_noisy_sensor(n_samples, true_value, noise_std):
    return true_value + np.random.normal(0, noise_std, n_samples)

def ewma_filter(data, alpha):
    filtered = np.zeros_like(data)
    filtered[0] = data[0]
    for i in range(1, len(data)):
        filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i-1]
    return filtered

# Simulate noisy sensor data
n_samples = 1000
true_value = 25  # e.g., temperature in Celsius
noise_std = 2
noisy_data = simulate_noisy_sensor(n_samples, true_value, noise_std)

# Apply EWMA filter
alpha = 0.1
filtered_data = ewma_filter(noisy_data, alpha)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(noisy_data, label='Noisy Sensor Data', alpha=0.5)
plt.plot(filtered_data, label='EWMA Filtered Data', linewidth=2)
plt.axhline(y=true_value, color='r', linestyle='--', label='True Value')
plt.legend()
plt.title('EWMA for Noise Reduction in Sensor Data')
plt.ylabel('Temperature (°C)')
plt.xlabel('Time')
plt.show()
```

Slide 8: EWMA in Batch Normalization

EWMA can be used in batch normalization to compute running statistics. This is particularly useful during inference when batch statistics are not available:

```python
import torch
import torch.nn as nn

class EWMABatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x):
        if self.training:
            mean = x.mean([0, 2, 3])
            var = x.var([0, 2, 3], unbiased=False)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return x

# Example usage
batch_norm = EWMABatchNorm(64)
dummy_input = torch.randn(32, 64, 28, 28)
output = batch_norm(dummy_input)
print("Output shape:", output.shape)
```

Slide 9: EWMA in Reinforcement Learning

In reinforcement learning, EWMA is often used to estimate the value function or Q-values. Here's an example of using EWMA in Q-learning:

```python
import numpy as np

class EWMAQLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha  # EWMA factor
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

# Example usage
n_states, n_actions = 10, 4
agent = EWMAQLearning(n_states, n_actions)

# Simulate one step
state = 0
action = agent.choose_action(state)
reward = 1
next_state = 1

agent.update(state, action, reward, next_state)
print("Updated Q-value:", agent.Q[state, action])
```

Slide 10: EWMA for Anomaly Detection

EWMA can be used for anomaly detection in time series data, which is crucial in many deep learning applications. Here's an example:

```python
import numpy as np
import matplotlib.pyplot as plt

def ewma_anomaly_detection(data, alpha, threshold):
    ewma = np.zeros_like(data)
    ewma[0] = data[0]
    anomalies = []
    
    for i in range(1, len(data)):
        ewma[i] = alpha * data[i] + (1 - alpha) * ewma[i-1]
        if abs(data[i] - ewma[i]) > threshold:
            anomalies.append(i)
    
    return ewma, anomalies

# Generate sample data with anomalies
np.random.seed(42)
n_samples = 1000
data = np.cumsum(np.random.randn(n_samples))
data[500:520] += 10  # Introduce anomaly

# Detect anomalies
alpha = 0.1
threshold = 3
ewma, anomalies = ewma_anomaly_detection(data, alpha, threshold)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data, label='Original Data')
plt.plot(ewma, label='EWMA', linewidth=2)
plt.scatter(anomalies, data[anomalies], color='r', label='Anomalies')
plt.legend()
plt.title('EWMA for Anomaly Detection')
plt.show()
```

Slide 11: EWMA in Natural Language Processing

EWMA can be used in NLP tasks, such as smoothing word embeddings or tracking topic trends. Here's an example of using EWMA to smooth word embeddings:

```python
import numpy as np
from gensim.models import Word2Vec

# Sample sentences
sentences = [
    ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
    ["a", "quick", "brown", "dog", "jumps", "over", "the", "lazy", "fox"]
]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

def ewma_word_embeddings(model, alpha=0.1):
    vocab = list(model.wv.key_to_index.keys())
    smoothed_embeddings = {}
    
    for word in vocab:
        vector = model.wv[word]
        if word not in smoothed_embeddings:
            smoothed_embeddings[word] = vector
        else:
            smoothed_embeddings[word] = alpha * vector + (1 - alpha) * smoothed_embeddings[word]
    
    return smoothed_embeddings

# Apply EWMA smoothing
smoothed_embeddings = ewma_word_embeddings(model)

# Compare original and smoothed embeddings
word = "quick"
print(f"Original embedding for '{word}':", model.wv[word][:5])
print(f"Smoothed embedding for '{word}':", smoothed_embeddings[word][:5])
```

Slide 12: EWMA in Computer Vision

In computer vision, EWMA can be used for various tasks such as background subtraction or tracking object motion. Here's an example of using EWMA for simple background subtraction:

```python
import numpy as np

def ewma_background_subtraction(frames, alpha=0.01, threshold=30):
    background = frames[0].astype(float)
    foreground_masks = []
    
    for frame in frames[1:]:
        # Update background using EWMA
        background = alpha * frame + (1 - alpha) * background
        
        # Compute difference
        diff = np.abs(frame - background.astype(np.uint8))
        
        # Threshold difference
        mask = np.mean(diff, axis=2) > threshold
        foreground_masks.append(mask)
    
    return background.astype(np.uint8), foreground_masks

# Example usage
frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(100)]
background, foreground_masks = ewma_background_subtraction(frames)

print("Background shape:", background.shape)
print("Number of foreground masks:", len(foreground_masks))
```

Slide 13: EWMA in Audio Processing

EWMA can be applied in audio processing for tasks like noise reduction or spectral analysis. Here's a simple example of using EWMA for smoothing audio spectrograms:

```python
import numpy as np
import librosa
import matplotlib.pyplot as plt

def ewma_spectrogram_smoothing(audio_path, alpha=0.1):
    # Load audio file
    y, sr = librosa.load(audio_path)
    
    # Compute spectrogram
    D = librosa.stft(y)
    S = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Apply EWMA smoothing
    smoothed_S = np.zeros_like(S)
    smoothed_S[:, 0] = S[:, 0]
    
    for i in range(1, S.shape[1]):
        smoothed_S[:, i] = alpha * S[:, i] + (1 - alpha) * smoothed_S[:, i-1]
    
    return S, smoothed_S

# Example usage (using random data instead of actual audio file)
random_audio = np.random.randn(22050 * 5)  # 5 seconds of random audio at 22050 Hz
S, smoothed_S = ewma_spectrogram_smoothing(random_audio)

plt.figure(figsize=(12, 8))
plt.subplot(211)
librosa.display.specshow(S, sr=22050, x_axis='time', y_axis='hz')
plt.title('Original Spectrogram')
plt.subplot(212)
librosa.display.specshow(smoothed_S, sr=22050, x_axis='time', y_axis='hz')
plt.title('EWMA Smoothed Spectrogram')
plt.tight_layout()
plt.show()
```

Slide 14: EWMA in Generative Models

EWMA can be used in generative models to stabilize training or smooth generated outputs. Here's a conceptual example of using EWMA in a simple generative model:

```python
import torch
import torch.nn as nn

class EWMAGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, alpha=0.999):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )
        self.register_buffer('ema_weights', None)
        self.alpha = alpha
    
    def forward(self, z):
        return self.generator(z)
    
    def update_ema(self):
        if self.ema_weights is None:
            self.ema_weights = {k: v.data.clone() for k, v in self.generator.named_parameters()}
        else:
            for name, param in self.generator.named_parameters():
                self.ema_weights[name] = self.alpha * self.ema_weights[name] + (1 - self.alpha) * param.data
    
    def generate_with_ema(self, z):
        original_weights = {k: v.data.clone() for k, v in self.generator.named_parameters()}
        for name, param in self.generator.named_parameters():
            param.data._(self.ema_weights[name])
        
        with torch.no_grad():
            output = self.generator(z)
        
        for name, param in self.generator.named_parameters():
            param.data._(original_weights[name])
        
        return output

# Example usage
latent_dim, output_dim = 64, 784
generator = EWMAGenerator(latent_dim, output_dim)
z = torch.randn(10, latent_dim)
output = generator(z)
generator.update_ema()
ema_output = generator.generate_with_ema(z)

print("Output shape:", output.shape)
print("EMA output shape:", ema_output.shape)
```

Slide 15: Additional Resources

For further exploration of Exponentially Weighted Moving Averages in Deep Learning, consider the following resources:

1. "Adam: A Method for Stochastic Optimization" by Kingma and Ba (2014) ArXiv: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
2. "On the Convergence of Adam and Beyond" by Reddi et al. (2018) ArXiv: [https://arxiv.org/abs/1904.09237](https://arxiv.org/abs/1904.09237)
3. "Exponential Moving Average Normalization for Self-supervised and Semi-supervised Learning" by Cai et al. (2021) ArXiv: [https://arxiv.org/abs/2101.08482](https://arxiv.org/abs/2101.08482)

These papers provide in-depth discussions on the applications and theoretical foundations of EWMA in deep learning contexts.

