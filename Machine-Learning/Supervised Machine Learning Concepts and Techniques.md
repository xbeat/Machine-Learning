## Supervised Machine Learning Concepts and Techniques
Slide 1: Introduction to Supervised Machine Learning

Supervised Machine Learning is a fundamental concept in artificial intelligence where an algorithm learns from labeled training data to make predictions or decisions on new, unseen data. This process involves a dataset with input features and corresponding target variables, which the model uses to learn patterns and relationships.

```python
# Simple example of supervised learning using linear regression
import random

# Generate synthetic data
X = [i for i in range(100)]
y = [2*x + random.uniform(-10, 10) for x in X]

# Linear regression implementation
def linear_regression(X, y):
    n = len(X)
    sum_x = sum(X)
    sum_y = sum(y)
    sum_xy = sum(x*y for x, y in zip(X, y))
    sum_xx = sum(x*x for x in X)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept

# Train the model
slope, intercept = linear_regression(X, y)

# Make predictions
def predict(x):
    return slope * x + intercept

# Test the model
test_x = 150
prediction = predict(test_x)
print(f"Prediction for x={test_x}: {prediction}")
```

Slide 2: Linear Regression

Linear regression is a simple yet powerful supervised learning algorithm used for predicting a continuous target variable based on one or more input features. It assumes a linear relationship between the input variables and the target variable, making it easy to interpret and implement.

```python
import random

# Generate synthetic data
X = [[1, x] for x in range(100)]  # Add bias term
y = [3 + 2*x[1] + random.uniform(-10, 10) for x in X]

# Multivariate linear regression implementation
def multivariate_linear_regression(X, y):
    X_transpose = list(zip(*X))
    X_transpose_X = [[sum(a*b for a, b in zip(X_transpose_row, X_col)) 
                      for X_col in zip(*X)] for X_transpose_row in X_transpose]
    X_transpose_y = [sum(a*b for a, b in zip(X_transpose_row, y)) 
                     for X_transpose_row in X_transpose]
    
    # Solve linear system using Gaussian elimination
    n = len(X_transpose_X)
    for i in range(n):
        max_element = abs(X_transpose_X[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(X_transpose_X[k][i]) > max_element:
                max_element = abs(X_transpose_X[k][i])
                max_row = k
        X_transpose_X[i], X_transpose_X[max_row] = X_transpose_X[max_row], X_transpose_X[i]
        X_transpose_y[i], X_transpose_y[max_row] = X_transpose_y[max_row], X_transpose_y[i]
        
        for k in range(i + 1, n):
            c = -X_transpose_X[k][i] / X_transpose_X[i][i]
            for j in range(i, n):
                if i == j:
                    X_transpose_X[k][j] = 0
                else:
                    X_transpose_X[k][j] += c * X_transpose_X[i][j]
            X_transpose_y[k] += c * X_transpose_y[i]
    
    # Back substitution
    coefficients = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        coefficients[i] = X_transpose_y[i]
        for j in range(i + 1, n):
            coefficients[i] -= X_transpose_X[i][j] * coefficients[j]
        coefficients[i] /= X_transpose_X[i][i]
    
    return coefficients

# Train the model
coefficients = multivariate_linear_regression(X, y)

# Make predictions
def predict(x):
    return sum(a*b for a, b in zip(coefficients, x))

# Test the model
test_x = [1, 150]  # Add bias term
prediction = predict(test_x)
print(f"Prediction for x={test_x[1]}: {prediction}")
```

Slide 3: Generalized Linear Models

Generalized Linear Models (GLMs) extend linear regression to handle various types of target variables and error distributions. They allow for non-linear relationships between predictors and the target variable through a link function, making them versatile for different types of data and prediction tasks.

```python
import math
import random

# Logistic regression (a type of GLM) implementation
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    m, n = len(X), len(X[0])
    theta = [0] * n
    
    for _ in range(epochs):
        for i in range(m):
            h = sigmoid(sum(theta[j] * X[i][j] for j in range(n)))
            error = h - y[i]
            for j in range(n):
                theta[j] -= learning_rate * error * X[i][j]
    
    return theta

# Generate synthetic data
X = [[1, random.uniform(0, 10), random.uniform(0, 10)] for _ in range(100)]
y = [1 if x[1] + x[2] > 10 else 0 for x in X]

# Train the model
theta = logistic_regression(X, y)

# Make predictions
def predict(x):
    return sigmoid(sum(a*b for a, b in zip(theta, x)))

# Test the model
test_x = [1, 7, 5]
prediction = predict(test_x)
print(f"Probability for x={test_x[1:]}: {prediction}")
```

Slide 4: Gaussian Discriminant Analysis

Gaussian Discriminant Analysis (GDA) is a probabilistic classification method that models the distribution of each class using a Gaussian distribution. It's particularly useful when the classes are well-separated and follow a normal distribution, offering a balance between simplicity and effectiveness.

```python
import math
import random

def gaussian_pdf(x, mean, variance):
    return (1 / math.sqrt(2 * math.pi * variance)) * math.exp(-((x - mean) ** 2) / (2 * variance))

class GDA:
    def fit(self, X, y):
        self.classes = list(set(y))
        self.class_means = {c: [0] * len(X[0]) for c in self.classes}
        self.class_variances = {c: [0] * len(X[0]) for c in self.classes}
        self.class_priors = {c: 0 for c in self.classes}
        
        for x, label in zip(X, y):
            self.class_priors[label] += 1
            for i, feature in enumerate(x):
                self.class_means[label][i] += feature
        
        for c in self.classes:
            self.class_priors[c] /= len(y)
            self.class_means[c] = [mean / self.class_priors[c] / len(y) for mean in self.class_means[c]]
        
        for x, label in zip(X, y):
            for i, feature in enumerate(x):
                self.class_variances[label][i] += (feature - self.class_means[label][i]) ** 2
        
        for c in self.classes:
            self.class_variances[c] = [var / (self.class_priors[c] * len(y)) for var in self.class_variances[c]]
    
    def predict(self, x):
        probs = {}
        for c in self.classes:
            probs[c] = math.log(self.class_priors[c])
            for i, feature in enumerate(x):
                probs[c] += math.log(gaussian_pdf(feature, self.class_means[c][i], self.class_variances[c][i]))
        return max(probs, key=probs.get)

# Generate synthetic data
X = [[random.gauss(5, 2), random.gauss(5, 2)] for _ in range(50)] + \
    [[random.gauss(10, 2), random.gauss(10, 2)] for _ in range(50)]
y = [0] * 50 + [1] * 50

# Train the model
gda = GDA()
gda.fit(X, y)

# Test the model
test_x = [7, 7]
prediction = gda.predict(test_x)
print(f"Prediction for x={test_x}: Class {prediction}")
```

Slide 5: Tree-based Methods

Tree-based methods are a family of powerful algorithms used for both classification and regression tasks. They work by recursively partitioning the feature space into regions, creating a hierarchical structure that resembles a tree. Decision trees are the foundation of this approach, while ensemble methods like Random Forests and Gradient Boosting build upon this concept to create more robust and accurate models.

```python
import random

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def gini_impurity(y):
    classes = set(y)
    impurity = 1.0
    for c in classes:
        p = y.count(c) / len(y)
        impurity -= p ** 2
    return impurity

def split_data(X, y, feature_index, threshold):
    left_X, left_y, right_X, right_y = [], [], [], []
    for x, label in zip(X, y):
        if x[feature_index] <= threshold:
            left_X.append(x)
            left_y.append(label)
        else:
            right_X.append(x)
            right_y.append(label)
    return left_X, left_y, right_X, right_y

def find_best_split(X, y):
    best_gini = float('inf')
    best_feature = None
    best_threshold = None
    
    for feature_index in range(len(X[0])):
        thresholds = sorted(set(x[feature_index] for x in X))
        for threshold in thresholds:
            left_X, left_y, right_X, right_y = split_data(X, y, feature_index, threshold)
            gini = (len(left_y) * gini_impurity(left_y) + len(right_y) * gini_impurity(right_y)) / len(y)
            if gini < best_gini:
                best_gini = gini
                best_feature = feature_index
                best_threshold = threshold
    
    return best_feature, best_threshold

def build_tree(X, y, max_depth, current_depth=0):
    if current_depth == max_depth or len(set(y)) == 1:
        return DecisionTreeNode(value=max(set(y), key=y.count))
    
    feature_index, threshold = find_best_split(X, y)
    if feature_index is None:
        return DecisionTreeNode(value=max(set(y), key=y.count))
    
    left_X, left_y, right_X, right_y = split_data(X, y, feature_index, threshold)
    
    left_subtree = build_tree(left_X, left_y, max_depth, current_depth + 1)
    right_subtree = build_tree(right_X, right_y, max_depth, current_depth + 1)
    
    return DecisionTreeNode(feature_index, threshold, left_subtree, right_subtree)

def predict(node, x):
    if node.value is not None:
        return node.value
    if x[node.feature_index] <= node.threshold:
        return predict(node.left, x)
    else:
        return predict(node.right, x)

# Generate synthetic data
X = [[random.uniform(0, 10), random.uniform(0, 10)] for _ in range(100)]
y = [1 if x[0] + x[1] > 10 else 0 for x in X]

# Train the model
tree = build_tree(X, y, max_depth=5)

# Test the model
test_x = [7, 5]
prediction = predict(tree, test_x)
print(f"Prediction for x={test_x}: {prediction}")
```

Slide 6: Ensemble Methods

Ensemble methods combine multiple models to create a more powerful predictive model. These techniques often lead to improved accuracy and robustness compared to individual models. Common ensemble methods include bagging (e.g., Random Forests) and boosting (e.g., Gradient Boosting Machines).

```python
import random
from collections import Counter

def bootstrap_sample(X, y):
    n_samples = len(X)
    idxs = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
    return [X[i] for i in idxs], [y[i] for i in idxs]

def random_forest(X, y, n_trees, tree_params, n_features):
    forest = []
    features = list(range(len(X[0])))
    
    for _ in range(n_trees):
        X_sample, y_sample = bootstrap_sample(X, y)
        
        def find_best_split(X, y):
            best_gini = float('inf')
            best_feature = None
            best_threshold = None
            
            feature_subset = random.sample(features, n_features)
            
            for feature_index in feature_subset:
                thresholds = sorted(set(x[feature_index] for x in X))
                for threshold in thresholds:
                    left_X, left_y, right_X, right_y = split_data(X, y, feature_index, threshold)
                    gini = (len(left_y) * gini_impurity(left_y) + len(right_y) * gini_impurity(right_y)) / len(y)
                    if gini < best_gini:
                        best_gini = gini
                        best_feature = feature_index
                        best_threshold = threshold
            
            return best_feature, best_threshold
        
        tree = build_tree(X_sample, y_sample, find_best_split=find_best_split, **tree_params)
        forest.append(tree)
    
    return forest

def random_forest_predict(forest, x):
    predictions = [predict(tree, x) for tree in forest]
    return Counter(predictions).most_common(1)[0][0]

# Generate synthetic data
X = [[random.uniform(0, 10), random.uniform(0, 10)] for _ in range(100)]
y = [1 if x[0] + x[1] > 10 else 0 for x in X]

# Train the model
n_trees = 10
tree_params = {'max_depth': 5}
n_features = 1  # Number of features to consider for each split
forest = random_forest(X, y, n_trees, tree_params, n_features)

# Test the model
test_x = [7, 5]
prediction = random_forest_predict(forest, test_x)
print(f"Random Forest prediction for x={test_x}: {prediction}")
```

Slide 7: Introduction to Unsupervised Machine Learning

Unsupervised Machine Learning focuses on discovering hidden patterns and structures in unlabeled data. Unlike supervised learning, there are no predefined target variables. The primary goals include clustering similar data points, reducing dimensionality, and identifying anomalies. This approach is particularly useful when dealing with large datasets where manual labeling is impractical or when exploring unknown patterns in data.

```python
import random

def euclidean_distance(point1, point2):
    return sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5

def k_means(data, k, max_iterations=100):
    # Randomly initialize centroids
    centroids = random.sample(data, k)
    
    for _ in range(max_iterations):
        # Assign points to nearest centroid
        clusters = [[] for _ in range(k)]
        for point in data:
            closest_centroid = min(range(k), key=lambda i: euclidean_distance(point, centroids[i]))
            clusters[closest_centroid].append(point)
        
        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = [sum(coord) / len(cluster) for coord in zip(*cluster)]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(random.choice(data))
        
        # Check for convergence
        if new_centroids == centroids:
            break
        centroids = new_centroids
    
    return centroids, clusters

# Generate sample data
data = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(100)]

# Apply k-means clustering
k = 3
centroids, clusters = k_means(data, k)

print(f"Final centroids: {centroids}")
print(f"Number of points in each cluster: {[len(cluster) for cluster in clusters]}")
```

Slide 8: Principal Component Analysis (PCA)

Principal Component Analysis is a dimensionality reduction technique that identifies the principal components of the data, which are orthogonal vectors that capture the most variance. PCA is widely used for feature extraction, data compression, and visualization of high-dimensional data.

```python
def pca(X, num_components):
    # Center the data
    mean = [sum(x) / len(X) for x in zip(*X)]
    X_centered = [[x_i - m_i for x_i, m_i in zip(x, mean)] for x in X]

    # Compute covariance matrix
    cov_matrix = [[sum(a * b for a, b in zip(X_centered[i], X_centered[j])) / (len(X) - 1)
                   for j in range(len(X[0]))] for i in range(len(X[0]))]

    # Compute eigenvalues and eigenvectors (using power iteration method)
    def power_iteration(A, num_iterations=100):
        n = len(A)
        b_k = [random.random() for _ in range(n)]
        for _ in range(num_iterations):
            b_k1 = [sum(A[i][j] * b_k[j] for j in range(n)) for i in range(n)]
            b_k1_norm = sum(x*x for x in b_k1) ** 0.5
            b_k = [x / b_k1_norm for x in b_k1]
        return b_k

    eigenvectors = []
    for _ in range(num_components):
        eigenvector = power_iteration(cov_matrix)
        eigenvectors.append(eigenvector)
        # Deflate the covariance matrix
        cov_matrix = [[cov_matrix[i][j] - eigenvector[i] * eigenvector[j] 
                       for j in range(len(cov_matrix))] for i in range(len(cov_matrix))]

    # Project data onto principal components
    projected_data = [[sum(x_i * v_j for x_i, v_j in zip(x, v)) for v in eigenvectors] for x in X_centered]

    return projected_data, eigenvectors

# Generate sample data
X = [[random.gauss(0, 1) for _ in range(5)] for _ in range(100)]

# Apply PCA
num_components = 2
projected_data, principal_components = pca(X, num_components)

print(f"First few projected data points: {projected_data[:5]}")
print(f"Principal components: {principal_components}")
```

Slide 9: K-means Clustering

K-means clustering is an unsupervised learning algorithm that partitions data into K clusters based on similarity. It iteratively assigns data points to the nearest cluster centroid and updates the centroids until convergence. K-means is widely used for customer segmentation, image compression, and anomaly detection.

```python
import random

def euclidean_distance(point1, point2):
    return sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5

def k_means(data, k, max_iterations=100):
    # Randomly initialize centroids
    centroids = random.sample(data, k)
    
    for _ in range(max_iterations):
        # Assign points to nearest centroid
        clusters = [[] for _ in range(k)]
        for point in data:
            closest_centroid = min(range(k), key=lambda i: euclidean_distance(point, centroids[i]))
            clusters[closest_centroid].append(point)
        
        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = [sum(coord) / len(cluster) for coord in zip(*cluster)]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(random.choice(data))
        
        # Check for convergence
        if new_centroids == centroids:
            break
        centroids = new_centroids
    
    return centroids, clusters

# Generate sample data
data = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(100)]

# Apply k-means clustering
k = 3
centroids, clusters = k_means(data, k)

print(f"Final centroids: {centroids}")
print(f"Number of points in each cluster: {[len(cluster) for cluster in clusters]}")
```

Slide 10: Hierarchical Clustering

Hierarchical clustering is an unsupervised learning method that creates a tree-like hierarchy of clusters. It can be performed using either an agglomerative (bottom-up) or divisive (top-down) approach. This technique is particularly useful when the number of clusters is unknown and when a hierarchical structure in the data is of interest.

```python
def euclidean_distance(point1, point2):
    return sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5

def agglomerative_clustering(data, distance_threshold):
    # Initialize each point as a cluster
    clusters = [[point] for point in data]
    
    while len(clusters) > 1:
        min_distance = float('inf')
        merge_indices = None
        
        # Find the two closest clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = min(euclidean_distance(p1, p2) 
                               for p1 in clusters[i] for p2 in clusters[j])
                if distance < min_distance:
                    min_distance = distance
                    merge_indices = (i, j)
        
        # If the minimum distance is above the threshold, stop merging
        if min_distance > distance_threshold:
            break
        
        # Merge the two closest clusters
        i, j = merge_indices
        clusters[i].extend(clusters[j])
        clusters.pop(j)
    
    return clusters

# Generate sample data
data = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(20)]

# Apply agglomerative clustering
distance_threshold = 2.0
clusters = agglomerative_clustering(data, distance_threshold)

print(f"Number of clusters: {len(clusters)}")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {len(cluster)} points")
```

Slide 11: Expectation-Maximization Algorithm

The Expectation-Maximization (EM) algorithm is a powerful method for estimating parameters in statistical models with latent variables. It alternates between two steps: the expectation step (E-step) computes the expected value of the log-likelihood function, and the maximization step (M-step) updates the parameters to maximize this expected log-likelihood.

```python
import math
import random

def gaussian(x, mu, sigma):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def em_gaussian_mixture(data, k, max_iterations=100, tolerance=1e-6):
    # Initialize parameters
    means = random.sample(data, k)
    variances = [1] * k
    weights = [1 / k] * k
    
    for _ in range(max_iterations):
        # E-step: Compute responsibilities
        responsibilities = []
        for x in data:
            resp = [w * gaussian(x, mu, math.sqrt(var)) for w, mu, var in zip(weights, means, variances)]
            total = sum(resp)
            responsibilities.append([r / total for r in resp])
        
        # M-step: Update parameters
        N = [sum(r[j] for r in responsibilities) for j in range(k)]
        
        new_weights = [n / len(data) for n in N]
        new_means = [sum(r[j] * x for r, x in zip(responsibilities, data)) / N[j] for j in range(k)]
        new_variances = [sum(r[j] * (x - new_means[j])**2 for r, x in zip(responsibilities, data)) / N[j] for j in range(k)]
        
        # Check for convergence
        if all(abs(w1 - w2) < tolerance for w1, w2 in zip(weights, new_weights)):
            break
        
        weights, means, variances = new_weights, new_means, new_variances
    
    return weights, means, variances

# Generate sample data
data = [random.gauss(0, 1) for _ in range(100)] + [random.gauss(5, 1) for _ in range(100)]

# Apply EM algorithm for Gaussian Mixture Model
k = 2
weights, means, variances = em_gaussian_mixture(data, k)

print("Gaussian Mixture Model parameters:")
for i in range(k):
    print(f"Component {i + 1}: weight = {weights[i]:.2f}, mean = {means[i]:.2f}, variance = {variances[i]:.2f}")
```

Slide 12: Clustering Evaluation Metrics

Evaluating the quality of clustering results is crucial in unsupervised learning. Various metrics exist to assess cluster quality, including internal measures (e.g., silhouette score, Calinski-Harabasz index) and external measures when ground truth labels are available (e.g., adjusted Rand index, normalized mutual information).

```python
import math

def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def silhouette_score(data, labels):
    n = len(data)
    silhouette_values = []
    
    for i in range(n):
        a = 0  # Average distance to points in the same cluster
        b = float('inf')  # Minimum average distance to points in different cluster
        
        for j in range(n):
            if i != j:
                distance = euclidean_distance(data[i], data[j])
                if labels[i] == labels[j]:
                    a += distance
                else:
                    b = min(b, distance)
        
        if labels[i] != -1:  # Ignore noise points
            a /= labels.count(labels[i]) - 1
            silhouette = (b - a) / max(a, b)
            silhouette_values.append(silhouette)
    
    return sum(silhouette_values) / len(silhouette_values)

# Generate sample data and labels
data = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(100)]
labels = [random.randint(0, 2) for _ in range(100)]

# Calculate silhouette score
score = silhouette_score(data, labels)
print(f"Silhouette Score: {score:.4f}")
```

Slide 13: Introduction to Deep Learning

Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to learn hierarchical representations of data. These networks can automatically learn features from raw input, making them particularly powerful for tasks such as image recognition, natural language processing, and speech recognition.

```python
import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.bias1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.weights2 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
        self.bias2 = [random.uniform(-1, 1) for _ in range(output_size)]
    
    def forward(self, x):
        # Hidden layer
        hidden = [sigmoid(sum(w * i for w, i in zip(weights, x)) + b) 
                  for weights, b in zip(self.weights1, self.bias1)]
        
        # Output layer
        output = [sigmoid(sum(w * h for w, h in zip(weights, hidden)) + b) 
                  for weights, b in zip(self.weights2, self.bias2)]
        
        return output

# Create a simple neural network
nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)

# Test the neural network
input_data = [0.5, 0.8]
output = nn.forward(input_data)
print(f"Input: {input_data}")
print(f"Output: {output[0]:.4f}")
```

Slide 14: Convolutional Neural Networks (CNNs)

Convolutional Neural Networks are specialized deep learning models designed for processing grid-like data, particularly images. They use convolutional layers to automatically learn spatial hierarchies of features. CNNs have revolutionized computer vision tasks, including image classification, object detection, and image segmentation.

```python
import random

class ConvLayer:
    def __init__(self, input_shape, num_filters, filter_size, stride=1):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.filters = [[[random.uniform(-1, 1) for _ in range(filter_size)] 
                         for _ in range(filter_size)] 
                        for _ in range(num_filters)]
    
    def convolve(self, input_data):
        height, width = self.input_shape
        out_height = (height - self.filter_size) // self.stride + 1
        out_width = (width - self.filter_size) // self.stride + 1
        output = [[[0 for _ in range(out_width)] 
                   for _ in range(out_height)] 
                  for _ in range(self.num_filters)]
        
        for f in range(self.num_filters):
            for i in range(0, height - self.filter_size + 1, self.stride):
                for j in range(0, width - self.filter_size + 1, self.stride):
                    sum = 0
                    for x in range(self.filter_size):
                        for y in range(self.filter_size):
                            sum += input_data[i+x][j+y] * self.filters[f][x][y]
                    output[f][i//self.stride][j//self.stride] = sum
        return output

# Example usage
input_data = [[random.random() for _ in range(28)] for _ in range(28)]
conv_layer = ConvLayer(input_shape=(28, 28), num_filters=3, filter_size=3, stride=1)
output = conv_layer.convolve(input_data)
print(f"Output shape: {len(output)} x {len(output[0])} x {len(output[0][0])}")
```

Slide 15: Recurrent Neural Networks (RNNs)

Recurrent Neural Networks are a class of neural networks designed to work with sequential data. They maintain an internal state (memory) that allows them to process sequences of inputs. RNNs are particularly useful for tasks such as natural language processing, time series analysis, and speech recognition.

```python
import math
import random

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.Wxh = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.Whh = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.Why = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
        self.bh = [0 for _ in range(hidden_size)]
        self.by = [0 for _ in range(output_size)]

    def forward(self, inputs):
        h = [0 for _ in range(self.hidden_size)]
        outputs = []
        
        for x in inputs:
            # Update hidden state
            h_new = []
            for i in range(self.hidden_size):
                sum_wxh = sum(self.Wxh[i][j] * x[j] for j in range(len(x)))
                sum_whh = sum(self.Whh[i][j] * h[j] for j in range(self.hidden_size))
                h_new.append(tanh(sum_wxh + sum_whh + self.bh[i]))
            h = h_new
            
            # Compute output
            y = []
            for i in range(len(self.Why)):
                y.append(sum(self.Why[i][j] * h[j] for j in range(self.hidden_size)) + self.by[i])
            outputs.append(y)
        
        return outputs

# Example usage
rnn = SimpleRNN(input_size=3, hidden_size=4, output_size=2)
input_sequence = [[random.random() for _ in range(3)] for _ in range(5)]
outputs = rnn.forward(input_sequence)
print(f"Output sequence length: {len(outputs)}")
print(f"Output dimensionality: {len(outputs[0])}")
```

Slide 16: Reinforcement Learning

Reinforcement Learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties for its actions, aiming to maximize cumulative reward over time. This approach is particularly useful for problems involving sequential decision-making, such as game playing, robotics, and autonomous systems.

```python
import random

class SimpleQLearning:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = [[0 for _ in range(num_actions)] for _ in range(num_states)]
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.q_table[state]) - 1)
        else:
            return self.q_table[state].index(max(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        best_next_action = max(self.q_table[next_state])
        td_target = reward + self.discount_factor * best_next_action
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

# Simple environment simulation
def simple_environment(state, action):
    if state == 0 and action == 1:
        return 1, 10  # Next state, reward
    elif state == 1 and action == 0:
        return 0, 5
    else:
        return state, -1

# Training loop
agent = SimpleQLearning(num_states=2, num_actions=2)
for episode in range(1000):
    state = 0
    total_reward = 0
    
    for _ in range(10):  # 10 steps per episode
        action = agent.choose_action(state)
        next_state, reward = simple_environment(state, action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

print("Final Q-table:")
for state, actions in enumerate(agent.q_table):
    print(f"State {state}: {actions}")
```

Slide 17: Model Selection and Evaluation

Model selection and evaluation are crucial steps in the machine learning pipeline. They involve choosing the best model from a set of candidate models and assessing its performance on unseen data. Techniques such as cross-validation, regularization, and various performance metrics help in selecting robust models that generalize well to new data.

```python
import random

def train_test_split(data, test_size=0.2):
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]

def mean_squared_error(y_true, y_pred):
    return sum((y1 - y2) ** 2 for y1, y2 in zip(y_true, y_pred)) / len(y_true)

def k_fold_cross_validation(data, k=5):
    fold_size = len(data) // k
    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_fold = data[test_start:test_end]
        train_fold = data[:test_start] + data[test_end:]
        yield train_fold, test_fold

# Example: Linear Regression with k-fold cross-validation
def linear_regression(X, y):
    n = len(X)
    sum_x = sum(X)
    sum_y = sum(y)
    sum_xy = sum(x*y for x, y in zip(X, y))
    sum_xx = sum(x*x for x in X)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept

# Generate sample data
data = [(x, 2*x + random.uniform(-1, 1)) for x in range(100)]

# Perform k-fold cross-validation
k = 5
mse_scores = []

for train_fold, test_fold in k_fold_cross_validation(data, k):
    X_train, y_train = zip(*train_fold)
    X_test, y_true = zip(*test_fold)
    
    slope, intercept = linear_regression(X_train, y_train)
    y_pred = [slope * x + intercept for x in X_test]
    
    mse = mean_squared_error(y_true, y_pred)
    mse_scores.append(mse)

print(f"Cross-validation MSE scores: {mse_scores}")
print(f"Average MSE: {sum(mse_scores) / len(mse_scores)}")
```

Slide 18: Bias-Variance Tradeoff

The bias-variance tradeoff is a fundamental concept in machine learning that helps understand the balance between model complexity and generalization. Bias refers to the error introduced by approximating a real-world problem with a simplified model, while variance refers to the model's sensitivity to fluctuations in the training data. Understanding this tradeoff is crucial for building models that perform well on unseen data.

```python
import random
import math

def generate_data(n_samples, noise=0.1):
    X = [random.uniform(0, 10) for _ in range(n_samples)]
    y = [math.sin(x) + random.gauss(0, noise) for x in X]
    return X, y

def polynomial_regression(X, y, degree):
    n = len(X)
    X_poly = [[x**i for i in range(degree + 1)] for x in X]
    
    # Compute (X^T * X)^-1 * X^T * y
    XT_X = [[sum(a*b for a, b in zip(row1, row2)) for row2 in zip(*X_poly)] for row1 in X_poly]
    XT_y = [sum(x*y for x, y in zip(row, y)) for row in X_poly]
    
    # Solve using Gaussian elimination (simplified)
    n = len(XT_X)
    for i in range(n):
        pivot = XT_X[i][i]
        for j in range(i + 1, n):
            factor = XT_X[j][i] / pivot
            for k in range(i, n):
                XT_X[j][k] -= factor * XT_X[i][k]
            XT_y[j] -= factor * XT_y[i]
    
    # Back-substitution
    coeffs = [0] * n
    for i in range(n - 1, -1, -1):
        coeffs[i] = XT_y[i]
        for j in range(i + 1, n):
            coeffs[i] -= XT_X[i][j] * coeffs[j]
        coeffs[i] /= XT_X[i][i]
    
    return coeffs

def predict(X, coeffs):
    return [sum(coeff * x**i for i, coeff in enumerate(coeffs)) for x in X]

# Generate data
X_train, y_train = generate_data(100)
X_test, y_true = generate_data(50)

# Train models with different degrees
degrees = [1, 3, 10]
for degree in degrees:
    coeffs = polynomial_regression(X_train, y_train, degree)
    y_pred = predict(X_test, coeffs)
    mse = sum((y1 - y2)**2 for y1, y2 in zip(y_true, y_pred)) / len(y_true)
    print(f"Degree {degree} - Test MSE: {mse:.4f}")
```

Slide 19: Additional Resources

For further exploration of machine learning concepts and techniques, consider the following resources:

1.  Online Courses:
    *   Coursera: Machine Learning by Andrew Ng
    *   edX: Introduction to Artificial Intelligence by MIT
2.  Textbooks:
    *   "Pattern Recognition and Machine Learning" by Christopher Bishop
    *   "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
3.  Research Papers:
    *   "A Few Useful Things to Know About Machine Learning" by Pedro Domingos
    *   "Deep Learning in Neural Networks: An Overview" by Jürgen Schmidhuber
4.  Open-source Libraries:
    *   scikit-learn: Machine learning in Python
    *   TensorFlow and PyTorch: Deep learning frameworks
5.  Competitions and Datasets:
    *   Kaggle: Platform for data science competitions
    *   UCI Machine Learning Repository: Collection of datasets

These resources provide a mix of theoretical foundations and practical applications in machine learning, catering to various skill levels and interests.

