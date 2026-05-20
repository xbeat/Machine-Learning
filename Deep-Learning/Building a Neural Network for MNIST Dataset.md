## Building a Neural Network for MNIST Dataset
Slide 1: Building a Neural Network from Scratch for MNIST

In this presentation, we'll explore the process of creating a neural network from the ground up and applying it to the MNIST dataset. We'll cover data preparation, network architecture, training, and evaluation, all implemented in Python. This hands-on approach will provide insights into the inner workings of neural networks and their application to real-world problems.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target
X = X.astype('float32') / 255.0
y = y.astype('int')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Slide 2: Data Preparation

The first step in our journey is preparing the MNIST dataset for training. We load the dataset using scikit-learn's fetch\_openml function, which provides easy access to the MNIST images. Each image is a 28x28 pixel grayscale representation of a handwritten digit. We normalize the pixel values to range from 0 to 1 by dividing by 255, which helps in faster convergence during training.

```python
def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train_encoded = one_hot_encode(y_train)
y_test_encoded = one_hot_encode(y_test)

print("Shape of training data:", X_train.shape)
print("Shape of training labels:", y_train_encoded.shape)
```

Slide 3: Network Architecture

Our neural network consists of an input layer with 784 neurons (28x28 pixels), two hidden layers with ReLU activation, and an output layer with 10 neurons (one for each digit) using softmax activation. This architecture allows the network to learn complex patterns in the input data and make accurate predictions.

```python
class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.W1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.maximum(0, self.z2)  # ReLU activation
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        exp_scores = np.exp(self.z3)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # Softmax
        return self.probs
```

Slide 4: Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns. We use ReLU (Rectified Linear Unit) for the hidden layers and Softmax for the output layer. ReLU helps mitigate the vanishing gradient problem, while Softmax converts the output into a probability distribution over the classes.

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# Visualize ReLU function
x = np.linspace(-5, 5, 100)
y = relu(x)
plt.plot(x, y)
plt.title("ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()
```

Slide 5: Loss Function

The loss function quantifies how well our model is performing. For multi-class classification problems like MNIST, we use the categorical cross-entropy loss. This loss function penalizes the model more heavily for confident misclassifications, encouraging it to output probabilities that closely match the true labels.

```python
def categorical_cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss

# Example usage
y_true = np.array([[0, 1, 0], [1, 0, 0]])
y_pred = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
print("Cross-entropy loss:", categorical_cross_entropy(y_true, y_pred))
```

Slide 6: Backpropagation

Backpropagation is the heart of neural network training. It computes the gradient of the loss function with respect to each weight by applying the chain rule of calculus. These gradients indicate how to adjust the weights to minimize the loss. Our implementation calculates these gradients for each layer, moving backwards from the output.

```python
def backward(self, X, y, learning_rate):
    m = X.shape[0]
    
    # Backpropagation
    dz3 = self.probs - y
    dW3 = np.dot(self.a2.T, dz3) / m
    db3 = np.sum(dz3, axis=0, keepdims=True) / m
    
    dz2 = np.dot(dz3, self.W3.T) * relu_derivative(self.z2)
    dW2 = np.dot(self.a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    
    dz1 = np.dot(dz2, self.W2.T) * relu_derivative(self.z1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    
    # Update weights and biases
    self.W3 -= learning_rate * dW3
    self.b3 -= learning_rate * db3
    self.W2 -= learning_rate * dW2
    self.b2 -= learning_rate * db2
    self.W1 -= learning_rate * dW1
    self.b1 -= learning_rate * db1
```

Slide 7: Adam Optimizer

We implement the Adam (Adaptive Moment Estimation) optimizer, which adapts the learning rate for each parameter. Adam combines ideas from RMSprop and momentum optimization, resulting in faster convergence and better performance on a wide range of problems. It maintains a per-parameter learning rate, which is adapted based on the first and second moments of the gradients.

```python
class AdamOptimizer:
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
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1
        lr_t = self.learning_rate * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.epsilon)

        return params
```

Slide 8: Training Loop

The training loop is where all the pieces come together. We iterate over the dataset multiple times (epochs), performing forward propagation, calculating the loss, and then using backpropagation to update the weights. We use mini-batch gradient descent, processing small batches of data at a time to balance computational efficiency and update frequency.

```python
def train(model, X_train, y_train, epochs, batch_size, learning_rate):
    n_samples = X_train.shape[0]
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward pass
            probs = model.forward(X_batch)
            
            # Compute loss
            loss = categorical_cross_entropy(y_batch, probs)
            epoch_loss += loss

            # Backward pass
            model.backward(X_batch, y_batch, learning_rate)

        epoch_loss /= (n_samples // batch_size)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    return losses

# Train the model
model = NeuralNetwork(784, 128, 64, 10)
losses = train(model, X_train, y_train_encoded, epochs=10, batch_size=32, learning_rate=0.001)

# Plot the loss curve
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
```

Slide 9: Model Evaluation

After training, we evaluate our model's performance on the test set. This gives us an unbiased estimate of how well our network generalizes to unseen data. We calculate the accuracy, which is the proportion of correct predictions among the total number of cases examined.

```python
def predict(model, X):
    probs = model.forward(X)
    return np.argmax(probs, axis=1)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Make predictions on test set
y_pred = predict(model, X_test)
test_accuracy = accuracy(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Visualize some predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]}, True: {y_test[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 10: Confusion Matrix

A confusion matrix provides a detailed breakdown of our model's performance for each class. It shows how many instances of each digit were correctly classified and where misclassifications occurred. This helps identify which digits the model struggles with most, guiding further improvements.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Calculate per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)
for i, acc in enumerate(class_accuracy):
    print(f"Accuracy for digit {i}: {acc:.4f}")
```

Slide 11: Hyperparameter Tuning

Hyperparameters are configuration settings for our neural network that are not learned from the data. They include the learning rate, number of hidden layers, number of neurons in each layer, batch size, and number of epochs. Tuning these parameters can significantly impact model performance. We'll implement a simple grid search to find optimal hyperparameters.

```python
def grid_search(X_train, y_train, X_val, y_val):
    learning_rates = [0.001, 0.01, 0.1]
    hidden_sizes = [(64, 32), (128, 64), (256, 128)]
    best_accuracy = 0
    best_params = {}

    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            model = NeuralNetwork(784, hidden_size[0], hidden_size[1], 10)
            train(model, X_train, y_train, epochs=5, batch_size=32, learning_rate=lr)
            
            y_pred = predict(model, X_val)
            val_accuracy = accuracy(y_val, y_pred)
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_params = {'learning_rate': lr, 'hidden_size': hidden_size}

    print(f"Best parameters: {best_params}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")

# Assuming you've split your data into train, validation, and test sets
grid_search(X_train, y_train_encoded, X_val, y_val_encoded)
```

Slide 12: Real-Life Application: Handwritten Digit Recognition

One practical application of our MNIST-trained neural network is in automated mail sorting systems. Post offices can use similar models to recognize handwritten zip codes on envelopes, significantly speeding up the sorting process. This technology enables faster and more accurate mail delivery, reducing human error and processing time.

```python
def simulate_mail_sorting(model, image_path):
    # Load and preprocess the image (assuming it's already isolated and sized correctly)
    image = plt.imread(image_path)
    image = image.reshape(1, 784).astype('float32') / 255.0

    # Make prediction
    prediction = predict(model, image)[0]
    
    print(f"Predicted ZIP code digit: {prediction}")
    
    # Display the image
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted Digit: {prediction}")
    plt.axis('off')
    plt.show()

# Example usage (you would need an actual image file)
simulate_mail_sorting(model, 'path_to_zip_code_image.png')
```

Slide 13: Real-Life Application: Handwritten Note Digitization

Another application is digitizing handwritten notes. Students or professionals can use this technology to convert their handwritten notes into digital text, making them searchable and easier to organize. This application would require extending our model to recognize letters as well as digits, but the core principles remain the same.

```python
def digitize_notes(model, image_path):
    # Load and preprocess the image
    image = plt.imread(image_path)
    characters = segment_image(image)  # Segment image into individual characters
    
    digitized_text = ""
    for char in characters:
        char_image = preprocess_character(char)
        prediction = predict(model, char_image)
        digitized_text += convert_to_character(prediction)
    
    return digitized_text

# Example usage
image_path = "handwritten_note.png"
digitized_text = digitize_notes(model, image_path)
print("Digitized text:", digitized_text)

# Display original image
plt.imshow(plt.imread(image_path))
plt.title("Original Handwritten Note")
plt.axis('off')
plt.show()
```

Slide 14: Challenges and Future Improvements

While our neural network performs well on MNIST, there are several areas for improvement:

1. Data Augmentation: Generating additional training samples through rotations, shifts, and other transformations can improve model robustness.
2. Deeper Architectures: Exploring deeper networks or modern architectures like CNNs could potentially increase accuracy.
3. Regularization Techniques: Implementing dropout or L2 regularization can help prevent overfitting.
4. Learning Rate Scheduling: Dynamically adjusting the learning rate during training can lead to faster convergence and better performance.

```python
def data_augmentation(image):
    # Example of simple augmentation: random rotation
    angle = np.random.uniform(-15, 15)
    rotated = ndimage.rotate(image.reshape(28, 28), angle, reshape=False)
    return rotated.reshape(784)

# Augment training data
augmented_X = np.array([data_augmentation(img) for img in X_train])
augmented_y = y_train.()

# Combine original and augmented data
X_train_extended = np.vstack((X_train, augmented_X))
y_train_extended = np.hstack((y_train, augmented_y))

print("Original training set size:", X_train.shape[0])
print("Extended training set size:", X_train_extended.shape[0])
```

Slide 15: Additional Resources

For those interested in diving deeper into neural networks and their applications, here are some valuable resources:

1. "Neural Networks and Deep Learning" by Michael Nielsen - A comprehensive online book that explains neural networks from first principles.
2. ArXiv paper: "Gradient-Based Learning Applied to Document Recognition" by LeCun et al. (1998) - The original paper introducing the MNIST dataset and convolutional neural networks. ArXiv URL: [https://arxiv.org/abs/1998.01274](https://arxiv.org/abs/1998.01274)
3. deeplearning.ai courses on Coursera - A series of courses covering various aspects of deep learning, from basics to advanced topics.
4. TensorFlow and PyTorch documentation - Official guides for popular deep learning frameworks, offering tutorials and examples for implementing neural networks.
5. ArXiv paper: "Adam: A Method for Stochastic Optimization" by Kingma and Ba (2014) - The original paper describing the Adam optimizer. ArXiv URL: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

These resources provide a solid foundation for further exploration and experimentation with neural networks and deep learning techniques.

