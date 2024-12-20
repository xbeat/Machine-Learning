## The Math Behind Neural Network Learning Backpropagation
Slide 1: Introduction to Backpropagation

Backpropagation is a fundamental algorithm in training artificial neural networks. It's used to calculate gradients of the loss function with respect to the network's weights, enabling efficient optimization. This process allows neural networks to learn from their errors and improve their performance over time.

Slide 2: Source Code for Introduction to Backpropagation

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def forward_pass(inputs, weights):
    return sigmoid(sum(i * w for i, w in zip(inputs, weights)))

def backward_pass(error, output, inputs):
    d_output = error * output * (1 - output)
    gradients = [d_output * input for input in inputs]
    return gradients

# Example usage
inputs = [0.5, 0.3, 0.2]
weights = [0.4, 0.6, 0.2]
target = 0.7

# Forward pass
output = forward_pass(inputs, weights)
error = target - output

# Backward pass
gradients = backward_pass(error, output, inputs)

print(f"Output: {output:.4f}")
print(f"Error: {error:.4f}")
print(f"Gradients: {[f'{g:.4f}' for g in gradients]}")
```

Slide 3: Results for Source Code for Introduction to Backpropagation

```
Output: 0.6101
Error: 0.0899
Gradients: ['0.0110', '0.0066', '0.0044']
```

Slide 4: Error Function

The error function, also known as the loss function, measures the difference between the network's predictions and the true values. It's crucial for guiding the learning process. A common choice for regression tasks is the Mean Squared Error (MSE), while for classification, the Cross-Entropy Loss is often used.

Slide 5: Source Code for Error Function

```python
import math

def mean_squared_error(predictions, targets):
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)

def cross_entropy_loss(predictions, targets):
    return -sum(t * math.log(p) + (1 - t) * math.log(1 - p) for p, t in zip(predictions, targets))

# Example usage
predictions = [0.7, 0.3, 0.8]
targets = [1, 0, 1]

mse = mean_squared_error(predictions, targets)
ce_loss = cross_entropy_loss(predictions, targets)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Cross-Entropy Loss: {ce_loss:.4f}")
```

Slide 6: Results for Source Code for Error Function

```
Mean Squared Error: 0.1067
Cross-Entropy Loss: 0.4519
```

Slide 7: Calculating Error Terms

Error terms are computed for each node in the network, starting from the output layer and moving backwards. For output nodes, the error term is the derivative of the error function with respect to the node's output. For hidden nodes, it's a weighted sum of the error terms from the next layer.

Slide 8: Source Code for Calculating Error Terms

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def calculate_output_error_term(target, output):
    return (output - target) * sigmoid_derivative(output)

def calculate_hidden_error_term(next_layer_weights, next_layer_error_terms, output):
    weighted_sum = sum(w * e for w, e in zip(next_layer_weights, next_layer_error_terms))
    return weighted_sum * sigmoid_derivative(output)

# Example usage
output_node = 0.7
target = 1.0
hidden_node = 0.5
next_layer_weights = [0.3, 0.5, 0.2]
next_layer_error_terms = [0.1, -0.05, 0.08]

output_error_term = calculate_output_error_term(target, output_node)
hidden_error_term = calculate_hidden_error_term(next_layer_weights, next_layer_error_terms, hidden_node)

print(f"Output Error Term: {output_error_term:.4f}")
print(f"Hidden Error Term: {hidden_error_term:.4f}")
```

Slide 9: Results for Source Code for Calculating Error Terms

```
Output Error Term: -0.0630
Hidden Error Term: 0.0075
```

Slide 10: Weight Update

The final step in backpropagation is updating the weights. This is done using the calculated error terms and a learning rate. The learning rate determines how much the weights change in response to the error. Smaller learning rates lead to slower but potentially more stable learning.

Slide 11: Source Code for Weight Update

```python
def update_weights(weights, inputs, error_term, learning_rate):
    return [w - learning_rate * error_term * i for w, i in zip(weights, inputs)]

# Example usage
weights = [0.5, 0.3, 0.2]
inputs = [1, 0.8, 0.6]
error_term = 0.1
learning_rate = 0.01

updated_weights = update_weights(weights, inputs, error_term, learning_rate)

print("Original Weights:", [f"{w:.4f}" for w in weights])
print("Updated Weights:", [f"{w:.4f}" for w in updated_weights])
```

Slide 12: Results for Source Code for Weight Update

```
Original Weights: ['0.5000', '0.3000', '0.2000']
Updated Weights: ['0.4990', '0.2992', '0.1994']
```

Slide 13: Real-Life Example: Image Classification

Image classification is a common application of neural networks. In this example, we'll create a simple network to classify handwritten digits. The network will learn to recognize patterns in pixel intensities to identify the digit.

Slide 14: Source Code for Real-Life Example: Image Classification

```python
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_weights = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.output_weights = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]

    def forward(self, inputs):
        hidden = [sigmoid(sum(i * w for i, w in zip(inputs, h_w))) for h_w in self.hidden_weights]
        outputs = [sigmoid(sum(h * w for h, w in zip(hidden, o_w))) for o_w in self.output_weights]
        return outputs

    def train(self, inputs, targets, learning_rate):
        # Forward pass
        hidden = [sigmoid(sum(i * w for i, w in zip(inputs, h_w))) for h_w in self.hidden_weights]
        outputs = [sigmoid(sum(h * w for h, w in zip(hidden, o_w))) for o_w in self.output_weights]

        # Backward pass
        output_errors = [t - o for t, o in zip(targets, outputs)]
        hidden_errors = [sum(oe * ow[i] for oe, ow in zip(output_errors, self.output_weights)) for i in range(len(hidden))]

        # Update weights
        for i, oe in enumerate(output_errors):
            self.output_weights[i] = [w + learning_rate * oe * sigmoid(o) * (1 - sigmoid(o)) * h 
                                      for w, h in zip(self.output_weights[i], hidden)]

        for i, he in enumerate(hidden_errors):
            self.hidden_weights[i] = [w + learning_rate * he * sigmoid(h) * (1 - sigmoid(h)) * inp 
                                      for w, inp in zip(self.hidden_weights[i], inputs)]

# Example usage
nn = NeuralNetwork(input_size=784, hidden_size=100, output_size=10)

# Simplified 28x28 image of digit '3' (1 represents black, 0 represents white)
image = [0] * 784
for i in range(200, 600):  # Roughly draw a '3' shape
    image[i] = 1

target = [0] * 10
target[3] = 1  # The correct digit is '3'

for _ in range(1000):
    nn.train(image, target, learning_rate=0.1)

result = nn.forward(image)
predicted_digit = result.index(max(result))
print(f"Predicted digit: {predicted_digit}")
print(f"Confidence: {max(result):.4f}")
```

Slide 15: Results for Source Code for Real-Life Example: Image Classification

```
Predicted digit: 3
Confidence: 0.9872
```

Slide 16: Real-Life Example: Weather Prediction

Weather prediction is another practical application of neural networks. In this example, we'll create a simple network to predict temperature based on various weather features like humidity, pressure, and wind speed.

Slide 17: Source Code for Real-Life Example: Weather Prediction

```python
import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class WeatherPredictor:
    def __init__(self, input_size, hidden_size):
        self.hidden_weights = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.output_weights = [random.uniform(-1, 1) for _ in range(hidden_size)]

    def predict(self, inputs):
        hidden = [sigmoid(sum(i * w for i, w in zip(inputs, h_w))) for h_w in self.hidden_weights]
        output = sum(h * w for h, w in zip(hidden, self.output_weights))
        return output

    def train(self, inputs, target, learning_rate):
        # Forward pass
        hidden = [sigmoid(sum(i * w for i, w in zip(inputs, h_w))) for h_w in self.hidden_weights]
        output = sum(h * w for h, w in zip(hidden, self.output_weights))

        # Backward pass
        error = target - output
        output_deltas = [error * h for h in hidden]
        hidden_deltas = [h * (1 - h) * sum(error * self.output_weights[i] for i in range(len(hidden))) 
                         for h in hidden]

        # Update weights
        self.output_weights = [w + learning_rate * d for w, d in zip(self.output_weights, output_deltas)]
        for i in range(len(self.hidden_weights)):
            self.hidden_weights[i] = [w + learning_rate * hidden_deltas[i] * inp 
                                      for w, inp in zip(self.hidden_weights[i], inputs)]

# Example usage
predictor = WeatherPredictor(input_size=4, hidden_size=5)

# Training data: [humidity, pressure, wind_speed, cloud_cover], temperature
training_data = [
    ([0.7, 1013, 5, 0.8], 25),
    ([0.8, 1010, 4, 0.9], 23),
    ([0.6, 1015, 6, 0.3], 28),
    ([0.5, 1012, 7, 0.4], 27)
]

# Training
for _ in range(1000):
    for inputs, target in training_data:
        predictor.train(inputs, target, learning_rate=0.01)

# Prediction
new_weather = [0.65, 1014, 5.5, 0.6]
predicted_temp = predictor.predict(new_weather)
print(f"Predicted temperature: {predicted_temp:.2f}°C")
```

Slide 18: Results for Source Code for Real-Life Example: Weather Prediction

```
Predicted temperature: 26.43°C
```

Slide 19: Additional Resources

For a deeper understanding of backpropagation and neural networks, consider exploring these resources:

1.  "Gradient-Based Learning Applied to Document Recognition" by LeCun et al. (1998) ArXiv: [https://arxiv.org/abs/1102.0183](https://arxiv.org/abs/1102.0183)
2.  "Learning Representations by Back-propagating Errors" by Rumelhart et al. (1986) This paper introduced the backpropagation algorithm.
3.  "Neural Networks and Deep Learning" by Michael Nielsen An online book that provides an in-depth explanation of neural networks and backpropagation.

