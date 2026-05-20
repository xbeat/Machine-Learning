## Cross-Entropy in Python:
These resources cover various aspects of cross-entropy, including theoretical foundations, practical applications, and advanced techniques like focal loss and knowledge distillation.

Slide 1: Introduction to Cross-Entropy

Cross-entropy is a widely used loss function in machine learning, particularly in classification problems. It measures the performance of a model by comparing the predicted probability distribution with the actual distribution. A lower cross-entropy value indicates a better model performance.

Code:

```python
import numpy as np

# Example: Binary Cross-Entropy
actual = np.array([1, 0, 1, 0])  # True labels
predicted = np.array([0.8, 0.2, 0.6, 0.4])  # Predicted probabilities

cross_entropy = -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)).mean()
print(f"Binary Cross-Entropy: {cross_entropy:.4f}")
```

Slide 2: Cross-Entropy for Binary Classification

In binary classification problems, where there are only two possible classes (e.g., 0 or 1), the cross-entropy loss function is calculated as follows:

Code:

```python
import numpy as np

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
    return loss

# Example usage
y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.8, 0.2, 0.6, 0.4])

bce = binary_cross_entropy(y_true, y_pred)
print(f"Binary Cross-Entropy: {bce:.4f}")
```

Slide 3: Cross-Entropy for Multiclass Classification

In multiclass classification problems, where there are more than two classes, the cross-entropy loss function is calculated across all classes using the one-hot encoded target vectors and the predicted probability distributions.

Code:

```python
import numpy as np

def multiclass_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred), axis=1).mean()
    return loss

# Example usage
y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
y_pred = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])

mce = multiclass_cross_entropy(y_true, y_pred)
print(f"Multiclass Cross-Entropy: {mce:.4f}")
```

Slide 4: Cross-Entropy and Logistic Regression

Cross-entropy is commonly used as the loss function in logistic regression models, which are widely used for binary classification tasks.

Code:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_cross_entropy(X, y, weights):
    z = np.dot(X, weights)
    y_pred = sigmoid(z)
    loss = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)).mean()
    return loss

# Example usage
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
weights = np.array([0.1, 0.2])

loss = logistic_regression_cross_entropy(X, y, weights)
print(f"Logistic Regression Cross-Entropy: {loss:.4f}")
```

Slide 5: Cross-Entropy and Neural Networks

Cross-entropy is widely used as the loss function in neural networks for classification tasks, particularly in the final layer (output layer) where the softmax activation function is applied.

Code:

```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def neural_network_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred), axis=1).mean()
    return loss

# Example usage
y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
y_pred = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])

loss = neural_network_cross_entropy(y_true, y_pred)
print(f"Neural Network Cross-Entropy: {loss:.4f}")
```

Slide 6: Optimizing Cross-Entropy Loss

The goal of training a machine learning model is to minimize the cross-entropy loss by adjusting the model parameters (weights and biases) using optimization algorithms like gradient descent.

Code:

```python
import numpy as np

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred), axis=1).mean()
    return loss

def gradient_descent(X, y, weights, learning_rate, num_iterations):
    for _ in range(num_iterations):
        y_pred = np.dot(X, weights)
        loss = cross_entropy_loss(y, y_pred)
        gradients = np.dot(X.T, y_pred - y) / X.shape[0]
        weights -= learning_rate * gradients
    return weights

# Example usage
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
weights = np.array([0.1, 0.2])
learning_rate = 0.01
num_iterations = 1000

optimized_weights = gradient_descent(X, y, weights, learning_rate, num_iterations)
print(f"Optimized Weights: {optimized_weights}")
```

Slide 7: Cross-Entropy and Regularization

Regularization techniques like L1 (Lasso) and L2 (Ridge) regularization can be combined with cross-entropy loss to prevent overfitting in machine learning models.

Code:

```python
import numpy as np

def cross_entropy_loss_with_l1(y_true, y_pred, weights, alpha):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred), axis=1).mean() + alpha * np.sum(np.abs(weights))
    return loss

def cross_entropy_loss_with_l2(y_true, y_pred, weights, alpha):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred), axis=1).mean() + alpha * np.sum(weights ** 2)
    return loss

# Example usage
y_true = np.array([0, 1, 0])
y_pred = np.array([0.1, 0.8, 0.2])
weights = np.array([0.3, -0.2, 0.1])
alpha = 0.1

l1_loss = cross_entropy_loss_with_l1(y_true, y_pred, weights, alpha)
l2_loss = cross_entropy_loss_with_l2(y_true, y_pred, weights, alpha)

print(f"Cross-Entropy Loss with L1 Regularization: {l1_loss:.4f}")
print(f"Cross-Entropy Loss with L2 Regularization: {l2_loss:.4f}")
```

Slide 8: Cross-Entropy and Class Imbalance

In scenarios where the classes are imbalanced (one class is significantly more prevalent than the other), the cross-entropy loss function can be modified to handle class imbalance by introducing class weights.

Code:

```python
import numpy as np

def weighted_cross_entropy(y_true, y_pred, weights):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -(weights[1] * y_true * np.log(y_pred) + weights[0] * (1 - y_true) * np.log(1 - y_pred)).mean()
    return loss

# Example usage
y_true = np.array([0, 1, 0, 0, 1])
y_pred = np.array([0.2, 0.8, 0.1, 0.3, 0.7])
weights = [0.2, 0.8]  # Higher weight for the minority class

loss = weighted_cross_entropy(y_true, y_pred, weights)
print(f"Weighted Cross-Entropy Loss: {loss:.4f}")
```

Slide 9: Cross-Entropy and Masking

In sequence-to-sequence models or models with variable input lengths, masking is used to ignore the contributions of certain elements (e.g., padding tokens) to the cross-entropy loss.

Code:

```python
import numpy as np

def masked_cross_entropy(y_true, y_pred, mask):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    loss = np.sum(loss * mask) / np.sum(mask)
    return loss

# Example usage
y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
y_pred = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
mask = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]])  # Mask for variable lengths

loss = masked_cross_entropy(y_true, y_pred, mask)
print(f"Masked Cross-Entropy Loss: {loss:.4f}")
```

Slide 10: Cross-Entropy and Focal Loss

Focal loss is a variant of cross-entropy loss that is designed to address the issue of class imbalance and hard-to-classify examples by down-weighting the loss for well-classified examples.

Code:

```python
import numpy as np

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    loss = -(alpha * y_true * (1 - pt) ** gamma * np.log(pt) +
             (1 - alpha) * (1 - y_true) * pt ** gamma * np.log(1 - pt)).mean()
    return loss

# Example usage
y_true = np.array([0, 1, 0, 0, 1])
y_pred = np.array([0.2, 0.8, 0.1, 0.3, 0.7])
alpha = 0.25
gamma = 2.0

loss = focal_loss(y_true, y_pred, alpha, gamma)
print(f"Focal Loss: {loss:.4f}")
```

Slide 11: Cross-Entropy and Label Smoothing

Label smoothing is a regularization technique that can be used with cross-entropy loss to prevent overfitting and improve model generalization by smoothing the one-hot encoded target vectors.

Code:

```python
import numpy as np

def label_smoothing(y_true, epsilon=0.1):
    num_classes = y_true.shape[1]
    smoothed_y_true = y_true * (1 - epsilon) + epsilon / num_classes
    return smoothed_y_true

def cross_entropy_with_label_smoothing(y_true, y_pred, epsilon=0.1):
    smoothed_y_true = label_smoothing(y_true, epsilon)
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(smoothed_y_true * np.log(y_pred), axis=1).mean()
    return loss

# Example usage
y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
y_pred = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
epsilon = 0.1

loss = cross_entropy_with_label_smoothing(y_true, y_pred, epsilon)
print(f"Cross-Entropy Loss with Label Smoothing: {loss:.4f}")
```

Slide 12: Cross-Entropy and Temperature Scaling

Temperature scaling is a technique used to calibrate the predicted probabilities of a model by dividing the logits (input to the softmax function) by a temperature parameter before computing the cross-entropy loss.

Code:

```python
import numpy as np

def softmax_with_temperature(x, temperature):
    x_scaled = x / temperature
    e_x = np.exp(x_scaled - np.max(x_scaled, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def cross_entropy_with_temperature_scaling(y_true, y_pred, temperature):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    scaled_y_pred = softmax_with_temperature(np.log(y_pred), temperature)
    loss = -np.sum(y_true * np.log(scaled_y_pred), axis=1).mean()
    return loss

# Example usage
y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
y_pred = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
temperature = 1.5

loss = cross_entropy_with_temperature_scaling(y_true, y_pred, temperature)
print(f"Cross-Entropy Loss with Temperature Scaling: {loss:.4f}")
```

Slide 13: Cross-Entropy and Knowledge Distillation

Knowledge distillation is a technique used to transfer knowledge from a large, complex model (teacher) to a smaller, more efficient model (student) by minimizing the cross-entropy loss between the student's predictions and the teacher's softened predictions.

Code:

```python
import numpy as np

def softmax_with_temperature(x, temperature):
    x_scaled = x / temperature
    e_x = np.exp(x_scaled - np.max(x_scaled, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def knowledge_distillation_loss(y_true, y_student, y_teacher, temperature, alpha=0.5):
    epsilon = 1e-15  # To avoid log(0)
    y_student = np.clip(y_student, epsilon, 1 - epsilon)
    y_teacher = softmax_with_temperature(y_teacher, temperature)
    loss = alpha * cross_entropy(y_true, y_student) + (1 - alpha) * cross_entropy(y_teacher, y_student)
    return loss

# Example usage
y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
y_student = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1], [0.2, 0.4, 0.4]])
y_teacher = np.array([[0.05, 0.9, 0.05], [0.8, 0.15, 0.05], [0.1, 0.2, 0.7]])
temperature = 2.0
alpha = 0.5

loss = knowledge_distillation_loss(y_true, y_student, y_teacher, temperature, alpha)
print(f"Knowledge Distillation Loss: {loss:.4f}")
```

Slide 14: Additional Resources

For further exploration and understanding of cross-entropy and its applications, here are some additional resources:

* "Pattern Recognition and Machine Learning" by Christopher M. Bishop (Book)
* "An Overview of Loss Functions for Deep Learning" by Jon Brownlee (Blog Post)
* "Focal Loss for Dense Object Detection" by Tsung-Yi Lin et al. (ArXiv Paper: [https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002))
* "Knowledge Distillation" by Hinton et al. (ArXiv Paper: [https://arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531))
