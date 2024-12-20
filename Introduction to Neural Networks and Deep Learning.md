## Introduction to Neural Networks and Deep Learning
Slide 1: Neural Network Fundamentals

A neural network is a computational model inspired by biological neurons. The fundamental building block is the perceptron, which takes multiple inputs, applies weights, adds a bias, and produces an output through an activation function. This implementation demonstrates a basic perceptron class.

```python
import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize weights randomly and bias to zero
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0
        self.learning_rate = learning_rate
    
    def activate(self, x):
        # Step activation function
        return 1 if x > 0 else 0
    
    def predict(self, inputs):
        # Calculate weighted sum and apply activation
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activate(weighted_sum)
    
    def train(self, X, y, epochs=100):
        for _ in range(epochs):
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)
                # Update weights and bias
                error = target - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND gate
perceptron = Perceptron(input_size=2)
perceptron.train(X, y)
```

Slide 2: Activation Functions and Their Mathematics

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Each activation function has unique properties affecting gradient flow and network performance. Common functions include ReLU, Sigmoid, and Tanh.

```python
import numpy as np
import matplotlib.pyplot as plt

class ActivationFunctions:
    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x, derivative=False):
        sigmoid_x = 1 / (1 + np.exp(-x))
        if derivative:
            return sigmoid_x * (1 - sigmoid_x)
        return sigmoid_x
    
    @staticmethod
    def tanh(x, derivative=False):
        if derivative:
            return 1 - np.tanh(x)**2
        return np.tanh(x)

# Mathematical formulas (in LaTeX format):
"""
ReLU: $$f(x) = max(0, x)$$
Sigmoid: $$f(x) = \frac{1}{1 + e^{-x}}$$
Tanh: $$f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
"""
```

Slide 3: Feedforward Neural Network Implementation

The feedforward neural network propagates information through multiple layers sequentially. This implementation creates a flexible neural network architecture with configurable layers and neurons, using numpy for efficient matrix operations.

```python
import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))
        self.output = None
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes)-1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1]))
    
    def forward(self, X):
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

# Example usage
nn = NeuralNetwork([2, 4, 1])
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output = nn.forward(X)
```

Slide 4: Backpropagation Algorithm

Backpropagation is the cornerstone of neural network training, using the chain rule to calculate gradients and update weights. The algorithm propagates error backwards through the network, adjusting parameters to minimize the loss function.

```python
class NeuralNetworkWithBackprop:
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers)-1):
            self.layers.append({
                'weights': np.random.randn(layers[i], layers[i+1]) * 0.01,
                'biases': np.zeros((1, layers[i+1])),
            })
    
    def forward(self, X):
        activations = [X]
        for layer in self.layers:
            net = np.dot(activations[-1], layer['weights']) + layer['biases']
            activations.append(self.sigmoid(net))
        return activations
    
    def backward(self, X, y, activations, learning_rate=0.1):
        m = X.shape[0]
        delta = (activations[-1] - y) * self.sigmoid(activations[-1], derivative=True)
        
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            layer['weights'] -= learning_rate * np.dot(activations[i].T, delta) / m
            layer['biases'] -= learning_rate * np.sum(delta, axis=0, keepdims=True) / m
            if i > 0:
                delta = np.dot(delta, layer['weights'].T) * self.sigmoid(activations[i], derivative=True)
    
    @staticmethod
    def sigmoid(x, derivative=False):
        sigmoid_x = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return sigmoid_x * (1 - sigmoid_x) if derivative else sigmoid_x
```

Slide 5: Loss Functions and Optimization

Loss functions measure the difference between predicted and actual values, guiding the network's learning process. Different tasks require specific loss functions - MSE for regression, Cross-Entropy for classification. This implementation showcases common loss functions and their gradients.

```python
import numpy as np

class LossFunctions:
    @staticmethod
    def mse(y_true, y_pred, derivative=False):
        if derivative:
            return 2 * (y_pred - y_true) / y_true.shape[0]
        return np.mean(np.square(y_pred - y_true))
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred, derivative=False):
        epsilon = 1e-15  # Prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        if derivative:
            return -(y_true/y_pred - (1-y_true)/(1-y_pred))
        return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred, derivative=False):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        if derivative:
            return -y_true/y_pred
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# Loss function formulas in LaTeX:
"""
MSE: $$L = \frac{1}{n}\sum_{i=1}^n(y_i - \hat{y}_i)^2$$
Binary Cross-Entropy: $$L = -\frac{1}{n}\sum_{i=1}^n[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$
"""
```

Slide 6: Gradient Descent Optimization

Gradient descent optimizes neural network parameters by iteratively moving in the direction that minimizes the loss function. This implementation shows different variants including standard, mini-batch, and stochastic gradient descent.

```python
class GradientDescent:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params, gradients):
        if self.velocity is None:
            self.velocity = [np.zeros_like(param) for param in params]
        
        updated_params = []
        for i, (param, grad) in enumerate(zip(params, gradients)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
            updated_params.append(param + self.velocity[i])
            
        return updated_params

class MiniBatchGD:
    def train(self, X, y, model, batch_size=32, epochs=100):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:min(i + batch_size, n_samples)]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # Forward pass
                predictions = model.forward(X_batch)
                
                # Backward pass
                model.backward(X_batch, y_batch)
```

Slide 7: Weight Initialization Techniques

Proper weight initialization is crucial for neural network convergence. Different techniques like Xavier/Glorot and He initialization help maintain appropriate signal magnitude through layers, preventing vanishing or exploding gradients.

```python
class WeightInitialization:
    @staticmethod
    def xavier_init(n_inputs, n_outputs):
        """
        Xavier/Glorot initialization for tanh activation
        Variance = 2/(n_inputs + n_outputs)
        """
        limit = np.sqrt(6 / (n_inputs + n_outputs))
        return np.random.uniform(-limit, limit, (n_inputs, n_outputs))
    
    @staticmethod
    def he_init(n_inputs, n_outputs):
        """
        He initialization for ReLU activation
        Variance = 2/n_inputs
        """
        std = np.sqrt(2 / n_inputs)
        return np.random.normal(0, std, (n_inputs, n_outputs))
    
    @staticmethod
    def lecun_init(n_inputs, n_outputs):
        """
        LeCun initialization
        Variance = 1/n_inputs
        """
        std = np.sqrt(1 / n_inputs)
        return np.random.normal(0, std, (n_inputs, n_outputs))

# Mathematical formulas:
"""
Xavier: $$\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}$$
He: $$\sigma = \sqrt{\frac{2}{n_{in}}}$$
LeCun: $$\sigma = \sqrt{\frac{1}{n_{in}}}$$
"""
```

Slide 8: Regularization Techniques

Regularization helps prevent overfitting by adding constraints to the learning process. This implementation demonstrates L1/L2 regularization, dropout, and early stopping techniques for better model generalization.

```python
class Regularization:
    @staticmethod
    def l1_regularization(weights, lambda_reg):
        """L1 regularization (Lasso)"""
        return lambda_reg * np.sum(np.abs(weights))
    
    @staticmethod
    def l2_regularization(weights, lambda_reg):
        """L2 regularization (Ridge)"""
        return lambda_reg * np.sum(np.square(weights))
    
    @staticmethod
    def dropout(layer_output, dropout_rate, training=True):
        if not training:
            return layer_output
        
        mask = np.random.binomial(1, 1-dropout_rate, size=layer_output.shape)
        return (layer_output * mask) / (1 - dropout_rate)
    
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        
    def should_stop(self, validation_loss):
        if self.best_loss is None:
            self.best_loss = validation_loss
            return False
            
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience
```

Slide 9: Convolutional Neural Network Implementation

Convolutional Neural Networks excel at processing grid-like data, particularly images. This implementation demonstrates the core components: convolution operations, pooling layers, and the forward pass through a CNN architecture.

```python
import numpy as np

class ConvLayer:
    def __init__(self, num_filters, filter_size, input_shape):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_shape = input_shape
        
        # Initialize filters with He initialization
        self.filters = np.random.randn(
            num_filters, filter_size, filter_size) * np.sqrt(2.0/filter_size**2)
        self.biases = np.zeros(num_filters)
        
    def forward(self, input_data):
        self.input = input_data
        height, width = input_data.shape
        output_height = height - self.filter_size + 1
        output_width = width - self.filter_size + 1
        
        output = np.zeros((output_height, output_width, self.num_filters))
        
        for k in range(self.num_filters):
            for i in range(output_height):
                for j in range(output_width):
                    output[i, j, k] = np.sum(
                        input_data[i:i+self.filter_size, j:j+self.filter_size] * 
                        self.filters[k]) + self.biases[k]
        
        return output

class MaxPooling:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        
    def forward(self, input_data):
        self.input = input_data
        height, width, channels = input_data.shape
        
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        
        output = np.zeros((output_height, output_width, channels))
        
        for h in range(output_height):
            for w in range(output_width):
                h_start = h * self.stride
                h_end = h_start + self.pool_size
                w_start = w * self.stride
                w_end = w_start + self.pool_size
                
                output[h, w] = np.max(
                    input_data[h_start:h_end, w_start:w_end], axis=(0, 1))
                
        return output
```

Slide 10: LSTM Implementation - Long Short-Term Memory

Long Short-Term Memory networks are designed to handle sequential data by maintaining internal states and using gates to control information flow. This implementation shows the core LSTM architecture.

```python
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        # Initialize weights and biases for gates
        self.hidden_size = hidden_size
        scale = 1.0 / np.sqrt(hidden_size)
        
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * scale
        
        self.bf = np.zeros((1, hidden_size))
        self.bi = np.zeros((1, hidden_size))
        self.bc = np.zeros((1, hidden_size))
        self.bo = np.zeros((1, hidden_size))
        
    def forward(self, x, prev_h, prev_c):
        # Concatenate input and previous hidden state
        combined = np.concatenate((x, prev_h), axis=1)
        
        # Gate computations
        f = self.sigmoid(np.dot(combined, self.Wf) + self.bf)  # Forget gate
        i = self.sigmoid(np.dot(combined, self.Wi) + self.bi)  # Input gate
        c_tilde = np.tanh(np.dot(combined, self.Wc) + self.bc)  # Candidate
        o = self.sigmoid(np.dot(combined, self.Wo) + self.bo)  # Output gate
        
        # Update cell state and hidden state
        c = f * prev_c + i * c_tilde
        h = o * np.tanh(c)
        
        return h, c
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

"""
LSTM equations in LaTeX:
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$c_t = f_t * c_{t-1} + i_t * \tilde{c}_t$$
$$h_t = o_t * \tanh(c_t)$$
"""
```

Slide 11: Real-world Application - Image Classification

Implementation of a complete image classification system using a CNN, including data preprocessing, model training, and evaluation on the MNIST dataset.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ImageClassifier:
    def __init__(self):
        self.conv1 = ConvLayer(num_filters=32, filter_size=3, input_shape=(28, 28))
        self.pool1 = MaxPooling(pool_size=2, stride=2)
        self.conv2 = ConvLayer(num_filters=64, filter_size=3, input_shape=(13, 13))
        self.pool2 = MaxPooling(pool_size=2, stride=2)
        self.fc1 = Dense(64*5*5, 128)
        self.fc2 = Dense(128, 10)
    
    def preprocess_data(self, X):
        # Normalize and reshape data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))
        return X_scaled.reshape(X.shape)
    
    def forward(self, X):
        x = self.conv1.forward(X)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.pool2.forward(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1.forward(x)
        x = self.fc2.forward(x)
        return x

    def train(self, X, y, epochs=10, batch_size=32):
        X = self.preprocess_data(X)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                # Forward pass
                output = self.forward(batch_X)
                
                # Backward pass and optimization code here
                # (Implementation details omitted for brevity)
```

Slide 12: Real-world Application - Time Series Prediction

Implementation of a complete LSTM-based system for time series forecasting, demonstrating data preparation, sequence processing, and prediction on financial data.

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesPredictor:
    def __init__(self, sequence_length, n_features):
        self.sequence_length = sequence_length
        self.lstm = LSTMCell(input_size=n_features, hidden_size=64)
        self.dense = Dense(64, 1)
        self.scaler = MinMaxScaler()
    
    def prepare_sequences(self, data):
        # Create sequences for LSTM
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:(i + self.sequence_length)]
            target = data[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
    
    def train(self, data, epochs=100, batch_size=32):
        # Scale data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        # Prepare sequences
        X, y = self.prepare_sequences(scaled_data)
        
        # Initialize states
        h = np.zeros((batch_size, 64))
        c = np.zeros((batch_size, 64))
        
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                # Forward pass through LSTM
                for t in range(self.sequence_length):
                    h, c = self.lstm.forward(batch_X[:, t], h, c)
                
                # Final prediction
                predictions = self.dense.forward(h)
                
                # Calculate loss (MSE)
                loss = np.mean((predictions - batch_y) ** 2)
                total_loss += loss
                
                # Backward pass and optimization would go here
                
            print(f"Epoch {epoch + 1}, Loss: {total_loss/len(X):.4f}")
    
    def predict(self, sequence):
        # Scale input sequence
        scaled_seq = self.scaler.transform(sequence.reshape(-1, 1))
        
        # Initialize states
        h = np.zeros((1, 64))
        c = np.zeros((1, 64))
        
        # Forward pass
        for t in range(self.sequence_length):
            h, c = self.lstm.forward(scaled_seq[t].reshape(1, -1), h, c)
        
        prediction = self.dense.forward(h)
        
        # Inverse transform prediction
        return self.scaler.inverse_transform(prediction)
```

Slide 13: Model Evaluation and Metrics

Comprehensive implementation of evaluation metrics for both classification and regression tasks, essential for assessing model performance and comparing different architectures.

```python
class ModelEvaluation:
    @staticmethod
    def classification_metrics(y_true, y_pred):
        # Convert probabilities to class predictions
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred_classes == y_true)
        
        # Calculate precision, recall, and F1 for each class
        unique_classes = np.unique(y_true)
        metrics = {}
        
        for cls in unique_classes:
            true_positives = np.sum((y_true == cls) & (y_pred_classes == cls))
            false_positives = np.sum((y_true != cls) & (y_pred_classes == cls))
            false_negatives = np.sum((y_true == cls) & (y_pred_classes != cls))
            
            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            metrics[f'class_{cls}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # Calculate confusion matrix
        n_classes = len(unique_classes)
        confusion_matrix = np.zeros((n_classes, n_classes))
        for i in range(len(y_true)):
            confusion_matrix[y_true[i]][y_pred_classes[i]] += 1
            
        return {
            'accuracy': accuracy,
            'class_metrics': metrics,
            'confusion_matrix': confusion_matrix
        }
    
    @staticmethod
    def regression_metrics(y_true, y_pred):
        # Mean Squared Error
        mse = np.mean((y_true - y_pred) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
```

Slide 14: Additional Resources

arXiv papers for further reading:

*   "Deep Learning" by LeCun, Bengio, and Hinton: [https://arxiv.org/abs/1505.01497](https://arxiv.org/abs/1505.01497)
*   "Random Search for Hyper-Parameter Optimization" by Bergstra and Bengio: [https://arxiv.org/abs/1312.6055](https://arxiv.org/abs/1312.6055)
*   "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" by Srivastava et al.: [https://arxiv.org/abs/1207.0580](https://arxiv.org/abs/1207.0580)
*   "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" by Ioffe and Szegedy: [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
*   "Adam: A Method for Stochastic Optimization" by Kingma and Ba: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

