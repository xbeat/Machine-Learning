## Bias in Stochastic Gradient Descent for Neural Network Architectures
Slide 1: Bias in Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent is a fundamental optimization algorithm in machine learning. While it's highly effective, it can introduce bias in the training process. This bias can affect model performance and generalization. Let's explore the nature of this bias and its implications.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

# Plot the data
plt.scatter(X, y)
plt.title("Sample Data for Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

Slide 2: Understanding SGD Bias

SGD's bias stems from its stochastic nature. By updating parameters based on mini-batches rather than the entire dataset, SGD introduces variance in parameter updates. This variance can lead to bias in the final model, especially with small batch sizes or high learning rates.

```python
def sgd_step(X, y, w, b, learning_rate):
    N = len(y)
    y_pred = np.dot(X, w) + b
    dw = (1/N) * np.dot(X.T, (y_pred - y))
    db = (1/N) * np.sum(y_pred - y)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

# Initialize parameters
w = np.random.randn(1, 1)
b = 0
learning_rate = 0.01

# Perform SGD steps
for _ in range(1000):
    w, b = sgd_step(X, y, w, b, learning_rate)

print(f"Learned parameters: w = {w[0][0]:.4f}, b = {b:.4f}")
```

Slide 3: Batch Size Impact on Bias

The batch size in SGD significantly influences the bias-variance tradeoff. Smaller batch sizes introduce more noise in parameter updates, potentially leading to higher bias. Larger batch sizes reduce noise but may slow down convergence.

```python
def train_sgd(X, y, batch_size, epochs):
    w = np.random.randn(1, 1)
    b = 0
    learning_rate = 0.01
    N = len(y)
    
    for _ in range(epochs):
        for i in range(0, N, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            w, b = sgd_step(X_batch, y_batch, w, b, learning_rate)
    
    return w, b

batch_sizes = [1, 10, 50, 100]
results = []

for batch_size in batch_sizes:
    w, b = train_sgd(X, y, batch_size, epochs=100)
    results.append((batch_size, w[0][0], b))

for batch_size, w, b in results:
    print(f"Batch size: {batch_size}, w = {w:.4f}, b = {b:.4f}")
```

Slide 4: Learning Rate and Bias

The learning rate in SGD also plays a crucial role in determining bias. A high learning rate can cause parameter updates to overshoot, leading to increased bias and potentially unstable training. Conversely, a low learning rate may result in slow convergence and getting stuck in suboptimal solutions.

```python
def train_sgd_multi_lr(X, y, learning_rates):
    results = []
    for lr in learning_rates:
        w = np.random.randn(1, 1)
        b = 0
        
        for _ in range(1000):
            w, b = sgd_step(X, y, w, b, lr)
        
        results.append((lr, w[0][0], b))
    
    return results

learning_rates = [0.001, 0.01, 0.1, 1.0]
lr_results = train_sgd_multi_lr(X, y, learning_rates)

for lr, w, b in lr_results:
    print(f"Learning rate: {lr}, w = {w:.4f}, b = {b:.4f}")
```

Slide 5: Momentum to Reduce Bias

Momentum is a technique used to reduce bias in SGD by accumulating a moving average of past gradients. This helps smooth out parameter updates and can lead to faster convergence and reduced bias, especially in scenarios with sparse data or high curvature.

```python
def sgd_momentum_step(X, y, w, b, v_w, v_b, learning_rate, momentum):
    N = len(y)
    y_pred = np.dot(X, w) + b
    dw = (1/N) * np.dot(X.T, (y_pred - y))
    db = (1/N) * np.sum(y_pred - y)
    
    v_w = momentum * v_w + learning_rate * dw
    v_b = momentum * v_b + learning_rate * db
    
    w -= v_w
    b -= v_b
    
    return w, b, v_w, v_b

# Initialize parameters
w = np.random.randn(1, 1)
b = 0
v_w = np.zeros_like(w)
v_b = 0
learning_rate = 0.01
momentum = 0.9

# Perform SGD with momentum steps
for _ in range(1000):
    w, b, v_w, v_b = sgd_momentum_step(X, y, w, b, v_w, v_b, learning_rate, momentum)

print(f"Learned parameters with momentum: w = {w[0][0]:.4f}, b = {b:.4f}")
```

Slide 6: Adaptive Learning Rates

Adaptive learning rate methods like Adam or RMSprop can help mitigate bias by adjusting the learning rate for each parameter. These methods can be particularly effective in scenarios with sparse gradients or when dealing with non-stationary objectives.

```python
def adam_step(X, y, w, b, m_w, m_b, v_w, v_b, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    N = len(y)
    y_pred = np.dot(X, w) + b
    dw = (1/N) * np.dot(X.T, (y_pred - y))
    db = (1/N) * np.sum(y_pred - y)
    
    m_w = beta1 * m_w + (1 - beta1) * dw
    m_b = beta1 * m_b + (1 - beta1) * db
    v_w = beta2 * v_w + (1 - beta2) * (dw**2)
    v_b = beta2 * v_b + (1 - beta2) * (db**2)
    
    m_w_hat = m_w / (1 - beta1**t)
    m_b_hat = m_b / (1 - beta1**t)
    v_w_hat = v_w / (1 - beta2**t)
    v_b_hat = v_b / (1 - beta2**t)
    
    w -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
    b -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
    
    return w, b, m_w, m_b, v_w, v_b

# Initialize parameters for Adam
w = np.random.randn(1, 1)
b = 0
m_w, m_b, v_w, v_b = 0, 0, 0, 0
learning_rate = 0.01

# Perform Adam optimization steps
for t in range(1, 1001):
    w, b, m_w, m_b, v_w, v_b = adam_step(X, y, w, b, m_w, m_b, v_w, v_b, t, learning_rate)

print(f"Learned parameters with Adam: w = {w[0][0]:.4f}, b = {b:.4f}")
```

Slide 7: Regularization to Combat Bias

Regularization techniques like L1 and L2 regularization can help reduce bias by adding a penalty term to the loss function. This encourages simpler models and can prevent overfitting, which is often a symptom of bias in the training process.

```python
def sgd_step_with_regularization(X, y, w, b, learning_rate, l2_lambda):
    N = len(y)
    y_pred = np.dot(X, w) + b
    dw = (1/N) * np.dot(X.T, (y_pred - y)) + l2_lambda * w
    db = (1/N) * np.sum(y_pred - y)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

# Initialize parameters
w = np.random.randn(1, 1)
b = 0
learning_rate = 0.01
l2_lambda = 0.1

# Perform SGD steps with L2 regularization
for _ in range(1000):
    w, b = sgd_step_with_regularization(X, y, w, b, learning_rate, l2_lambda)

print(f"Learned parameters with L2 regularization: w = {w[0][0]:.4f}, b = {b:.4f}")
```

Slide 8: Cross-Validation to Assess Bias

Cross-validation is a powerful technique to assess and mitigate bias in SGD. By training on different subsets of the data and evaluating on held-out sets, we can get a more robust estimate of model performance and detect potential biases.

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def cross_validate_sgd(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        w = np.random.randn(1, 1)
        b = 0

        for _ in range(1000):
            w, b = sgd_step(X_train, y_train, w, b, learning_rate=0.01)

        y_pred = np.dot(X_val, w) + b
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)

    return np.mean(mse_scores), np.std(mse_scores)

mean_mse, std_mse = cross_validate_sgd(X, y)
print(f"Cross-validation MSE: {mean_mse:.4f} (+/- {std_mse:.4f})")
```

Slide 9: Ensemble Methods to Reduce Bias

Ensemble methods, such as bagging and boosting, can help reduce bias by combining multiple models. These techniques leverage the idea that different models may capture different aspects of the data, potentially canceling out individual biases.

```python
def train_sgd_ensemble(X, y, n_models=5):
    models = []
    for _ in range(n_models):
        w = np.random.randn(1, 1)
        b = 0
        for _ in range(1000):
            w, b = sgd_step(X, y, w, b, learning_rate=0.01)
        models.append((w, b))
    return models

def predict_ensemble(X, models):
    predictions = []
    for w, b in models:
        y_pred = np.dot(X, w) + b
        predictions.append(y_pred)
    return np.mean(predictions, axis=0)

ensemble_models = train_sgd_ensemble(X, y)
ensemble_predictions = predict_ensemble(X, ensemble_models)

mse = mean_squared_error(y, ensemble_predictions)
print(f"Ensemble MSE: {mse:.4f}")
```

Slide 10: Learning Rate Schedules

Learning rate schedules can help reduce bias by adapting the learning rate during training. Common strategies include step decay, exponential decay, and cosine annealing. These schedules can help the optimization process navigate the loss landscape more effectively.

```python
def sgd_with_lr_schedule(X, y, epochs, initial_lr, schedule='step', step_size=500, decay=0.1):
    w = np.random.randn(1, 1)
    b = 0
    
    for epoch in range(epochs):
        if schedule == 'step':
            lr = initial_lr * (decay ** (epoch // step_size))
        elif schedule == 'exponential':
            lr = initial_lr * (decay ** epoch)
        elif schedule == 'cosine':
            lr = initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        
        w, b = sgd_step(X, y, w, b, lr)
    
    return w, b

schedules = ['step', 'exponential', 'cosine']
results = []

for schedule in schedules:
    w, b = sgd_with_lr_schedule(X, y, epochs=1000, initial_lr=0.1, schedule=schedule)
    results.append((schedule, w[0][0], b))

for schedule, w, b in results:
    print(f"Schedule: {schedule}, w = {w:.4f}, b = {b:.4f}")
```

Slide 11: Batch Normalization

Batch Normalization is a technique that can help reduce internal covariate shift and mitigate bias in deep neural networks. By normalizing the inputs to each layer, it can stabilize the learning process and potentially improve generalization.

```python
import torch
import torch.nn as nn

class BatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.bn1 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Create and train the model
model = BatchNormModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

print(f"Final loss: {loss.item():.4f}")
```

Slide 12: Real-Life Example: Image Classification

In image classification tasks, SGD bias can manifest as poor performance on certain classes or types of images. For example, a model trained to classify animals might show bias towards more common species or struggle with images taken from unusual angles.

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

# Define a simple CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train the model
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # Just 2 epochs for demonstration
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, loss: {running_loss / len(trainloader):.3f}')

print('Finished Training')
```

Slide 13: Real-Life Example: Natural Language Processing

In NLP tasks, SGD bias can lead to models that perform poorly on certain types of text or exhibit unwanted biases. For instance, a sentiment analysis model might struggle with sarcasm or show bias towards certain demographic groups.

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, text_lengths)
        packed_output, hidden = self.rnn(packed_embedded)
        output, output_lengths = pad_packed_sequence(packed_output)
        return self.fc(hidden.squeeze(0))

# Pseudo-training loop
vocab_size = 10000
embedding_dim = 100
hidden_dim = 256
output_dim = 2  # Binary sentiment

model = RNN(vocab_size, embedding_dim, hidden_dim, output_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):  # 5 epochs for demonstration
    for batch in range(100):  # Assume 100 batches per epoch
        # In a real scenario, you'd load actual text data here
        text = torch.randint(0, vocab_size, (20, 32))  # (seq_len, batch_size)
        text_lengths = torch.randint(1, 21, (32,))
        labels = torch.randint(0, 2, (32,))
        
        optimizer.zero_grad()
        predictions = model(text, text_lengths)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

print('Training complete')
```

Slide 14: Mitigating Bias in SGD

To mitigate bias in SGD, consider the following strategies:

1. Use larger batch sizes or mini-batch SGD
2. Implement adaptive learning rate methods (Adam, RMSprop)
3. Apply regularization techniques (L1, L2, dropout)
4. Employ cross-validation for hyperparameter tuning
5. Use ensemble methods to combine multiple models
6. Implement learning rate schedules
7. Apply batch normalization in deep networks
8. Carefully preprocess and balance your dataset
9. Regularly evaluate your model on diverse test sets
10. Be aware of potential biases in your training data

These techniques can help reduce the impact of SGD bias and improve the overall performance and fairness of your models.

Slide 15: Additional Resources

For more information on SGD bias and optimization techniques, consider the following resources:

1. "Optimization Methods for Large-Scale Machine Learning" by Léon Bottou, Frank E. Curtis, and Jorge Nocedal ArXiv: [https://arxiv.org/abs/1606.04838](https://arxiv.org/abs/1606.04838)
2. "Adam: A Method for Stochastic Optimization" by Diederik P. Kingma and Jimmy Ba ArXiv: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
3. "On the Convergence of Adam and Beyond" by Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar ArXiv: [https://arxiv.org/abs/1904.09237](https://arxiv.org/abs/1904.09237)

These papers provide in-depth analyses of SGD, its variants, and their convergence properties, offering valuable insights into the nature of optimization bias in machine learning.

