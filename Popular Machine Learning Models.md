## Popular Machine Learning Models
Slide 1: Introduction to Popular Machine Learning Models

Machine learning models are algorithms that learn patterns from data to make predictions or decisions. They form the backbone of artificial intelligence applications across various industries. This presentation will cover several key machine learning models, their applications, and practical implementations using Python.

Slide 2: Linear Regression

Linear regression is a fundamental model for predicting continuous values. It assumes a linear relationship between input features and the target variable.

Slide 3: Source Code for Linear Regression

```python
import random

# Generate sample data
X = [random.uniform(0, 10) for _ in range(100)]
y = [2*x + 1 + random.gauss(0, 1) for x in X]

# Implement linear regression
def linear_regression(X, y):
    n = len(X)
    sum_x = sum(X)
    sum_y = sum(y)
    sum_xy = sum(x*y for x, y in zip(X, y))
    sum_x_squared = sum(x**2 for x in X)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept

# Fit the model
slope, intercept = linear_regression(X, y)
print(f"Slope: {slope:.2f}, Intercept: {intercept:.2f}")

# Make predictions
X_test = [5, 7, 9]
predictions = [slope * x + intercept for x in X_test]
print("Predictions:", [f"{pred:.2f}" for pred in predictions])
```

Slide 4: Results for Linear Regression

```
Slope: 2.03, Intercept: 0.98
Predictions: ['11.13', '15.19', '19.25']
```

Slide 5: Decision Trees

Decision trees are versatile models used for both classification and regression tasks. They make decisions by splitting the data based on feature values, forming a tree-like structure.

Slide 6: Source Code for Decision Trees

```python
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - sum(p**2 for p in probabilities)

def split_data(X, y, feature, threshold):
    left_mask = X[:, feature] <= threshold
    return X[left_mask], y[left_mask], X[~left_mask], y[~left_mask]

def find_best_split(X, y):
    best_gini = float('inf')
    best_split = None
    
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_data(X, y, feature, threshold)
            gini = (len(y_left) * gini_impurity(y_left) + len(y_right) * gini_impurity(y_right)) / len(y)
            
            if gini < best_gini:
                best_gini = gini
                best_split = (feature, threshold)
    
    return best_split

def build_tree(X, y, max_depth=3, depth=0):
    if depth == max_depth or len(np.unique(y)) == 1:
        return Node(value=np.argmax(np.bincount(y)))
    
    feature, threshold = find_best_split(X, y)
    X_left, y_left, X_right, y_right = split_data(X, y, feature, threshold)
    
    return Node(
        feature=feature,
        threshold=threshold,
        left=build_tree(X_left, y_left, max_depth, depth+1),
        right=build_tree(X_right, y_right, max_depth, depth+1)
    )

# Example usage
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 0, 1, 1])
tree = build_tree(X, y)
```

Slide 7: K-Nearest Neighbors (KNN)

KNN is a simple yet effective algorithm for classification and regression. It makes predictions based on the majority class or average value of the k nearest data points.

Slide 8: Source Code for K-Nearest Neighbors

```python
import math

def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(most_common)
        return predictions

# Example usage
X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train = [0, 0, 1, 1]
knn = KNN(k=3)
knn.fit(X_train, y_train)

X_test = [[2.5, 3.5], [4.5, 5.5]]
predictions = knn.predict(X_test)
print("Predictions:", predictions)
```

Slide 9: Results for K-Nearest Neighbors

```
Predictions: [0, 1]
```

Slide 10: Support Vector Machines (SVM)

SVMs are powerful classifiers that find the optimal hyperplane to separate classes in high-dimensional space. They can handle both linear and non-linear classification tasks.

Slide 11: Source Code for Support Vector Machines

```python
import random

def linear_kernel(x1, x2):
    return sum(a*b for a, b in zip(x1, x2))

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.w = [0] * n_features
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (linear_kernel(self.w, x_i) + self.b) >= 1
                if condition:
                    self.w = [w - self.lr * (2 * self.lambda_param * w) for w in self.w]
                else:
                    self.w = [w - self.lr * (2 * self.lambda_param * w - y[idx] * x) for w, x in zip(self.w, x_i)]
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        return [1 if linear_kernel(self.w, x) + self.b >= 0 else -1 for x in X]

# Generate sample data
X = [[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(100)]
y = [1 if x[0] + x[1] >= 0 else -1 for x in X]

# Train SVM
svm = SVM()
svm.fit(X, y)

# Make predictions
X_test = [[1, 2], [-1, -2], [3, -4]]
predictions = svm.predict(X_test)
print("Predictions:", predictions)
```

Slide 12: Results for Support Vector Machines

```
Predictions: [1, -1, 1]
```

Slide 13: Neural Networks

Neural networks are versatile models inspired by the human brain. They consist of interconnected layers of neurons and can learn complex patterns in data.

Slide 14: Source Code for Neural Networks

```python
import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights_ih = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.bias_h = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.weights_ho = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
        self.bias_o = [random.uniform(-1, 1) for _ in range(output_size)]

    def forward(self, inputs):
        hidden = [sigmoid(sum(w * x for w, x in zip(weights, inputs)) + b) for weights, b in zip(self.weights_ih, self.bias_h)]
        outputs = [sigmoid(sum(w * h for w, h in zip(weights, hidden)) + b) for weights, b in zip(self.weights_ho, self.bias_o)]
        return outputs

    def train(self, inputs, targets, learning_rate=0.1):
        # Forward pass
        hidden = [sigmoid(sum(w * x for w, x in zip(weights, inputs)) + b) for weights, b in zip(self.weights_ih, self.bias_h)]
        outputs = [sigmoid(sum(w * h for w, h in zip(weights, hidden)) + b) for weights, b in zip(self.weights_ho, self.bias_o)]

        # Backpropagation
        output_errors = [t - o for t, o in zip(targets, outputs)]
        hidden_errors = [sum(oe * w for oe, w in zip(output_errors, [w[i] for w in self.weights_ho])) for i in range(self.hidden_size)]

        # Update weights and biases
        for i in range(self.output_size):
            for j in range(self.hidden_size):
                self.weights_ho[i][j] += learning_rate * output_errors[i] * sigmoid_derivative(outputs[i]) * hidden[j]
            self.bias_o[i] += learning_rate * output_errors[i] * sigmoid_derivative(outputs[i])

        for i in range(self.hidden_size):
            for j in range(self.input_size):
                self.weights_ih[i][j] += learning_rate * hidden_errors[i] * sigmoid_derivative(hidden[i]) * inputs[j]
            self.bias_h[i] += learning_rate * hidden_errors[i] * sigmoid_derivative(hidden[i])

# Example usage
nn = NeuralNetwork(2, 4, 1)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

for _ in range(10000):
    for inputs, targets in zip(X, y):
        nn.train(inputs, targets)

for inputs in X:
    prediction = nn.forward(inputs)
    print(f"Input: {inputs}, Prediction: {prediction[0]:.4f}")
```

Slide 15: Results for Neural Networks

```
Input: [0, 0], Prediction: 0.0321
Input: [0, 1], Prediction: 0.9678
Input: [1, 0], Prediction: 0.9679
Input: [1, 1], Prediction: 0.0322
```

Slide 16: Real-Life Example: Image Classification

Image classification is a common application of machine learning models. For instance, a convolutional neural network (CNN) can be trained to recognize different types of animals in photographs.

Slide 17: Real-Life Example: Natural Language Processing

Machine learning models are widely used in natural language processing tasks. For example, recurrent neural networks (RNNs) can be employed for sentiment analysis of product reviews or for language translation.

Slide 18: Additional Resources

For more in-depth information on machine learning models and algorithms, consider exploring these resources:

1.  "A Survey of Deep Learning Techniques for Neural Machine Translation" - ArXiv:1703.01619 [https://arxiv.org/abs/1703.01619](https://arxiv.org/abs/1703.01619)
2.  "Deep Learning in Neural Networks: An Overview" - ArXiv:1404.7828 [https://arxiv.org/abs/1404.7828](https://arxiv.org/abs/1404.7828)
3.  "Machine Learning: A Probabilistic Perspective" by Kevin P. Murphy (This is a comprehensive textbook on machine learning)

