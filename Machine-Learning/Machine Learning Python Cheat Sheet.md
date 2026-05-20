## Machine Learning Python Cheat Sheet

Slide 1: Introduction to Machine Learning

Machine Learning is a branch of artificial intelligence that focuses on developing algorithms and models that enable computers to learn from and make predictions or decisions based on data. It's a powerful tool for solving complex problems and extracting insights from large datasets.

Slide 2: Source Code for Introduction to Machine Learning

```python
# Simple example of machine learning: Linear Regression
import random

# Generate sample data
X = [i for i in range(100)]
y = [2*x + random.uniform(-10, 10) for x in X]

# Calculate mean of X and y
mean_x = sum(X) / len(X)
mean_y = sum(y) / len(y)

# Calculate slope (m) and y-intercept (b)
numerator = sum((X[i] - mean_x) * (y[i] - mean_y) for i in range(len(X)))
denominator = sum((x - mean_x)**2 for x in X)
m = numerator / denominator
b = mean_y - m * mean_x

print(f"Linear equation: y = {m:.2f}x + {b:.2f}")

# Predict y for x = 150
x_new = 150
y_pred = m * x_new + b
print(f"Predicted y for x = 150: {y_pred:.2f}")
```

Slide 3: Types of Machine Learning

Machine Learning can be categorized into three main types: Supervised Learning, Unsupervised Learning, and Reinforcement Learning. Supervised Learning involves training models on labeled data to make predictions. Unsupervised Learning finds patterns in unlabeled data. Reinforcement Learning trains agents to make decisions in dynamic environments.

Slide 4: Source Code for Types of Machine Learning

```python
# Example of Supervised Learning: Simple Classification
def classify_fruit(weight, texture):
    if weight > 150 and texture == 'bumpy':
        return 'orange'
    elif weight < 150 and texture == 'smooth':
        return 'apple'
    else:
        return 'unknown'

# Example of Unsupervised Learning: Simple Clustering
def simple_kmeans(points, k, max_iterations=100):
    # Initialize centroids randomly
    centroids = random.sample(points, k)
    
    for _ in range(max_iterations):
        # Assign points to nearest centroid
        clusters = [[] for _ in range(k)]
        for point in points:
            distances = [((point[0]-c[0])**2 + (point[1]-c[1])**2)**0.5 for c in centroids]
            closest_centroid = distances.index(min(distances))
            clusters[closest_centroid].append(point)
        
        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroids.append((sum(p[0] for p in cluster)/len(cluster),
                                      sum(p[1] for p in cluster)/len(cluster)))
            else:
                new_centroids.append(random.choice(points))
        
        if new_centroids == centroids:
            break
        centroids = new_centroids
    
    return clusters, centroids

# Example usage
points = [(1, 2), (2, 1), (3, 1), (5, 4), (5, 5), (6, 5)]
clusters, centroids = simple_kmeans(points, k=2)
print("Clusters:", clusters)
print("Centroids:", centroids)
```

Slide 5: Feature Engineering

Feature engineering is the process of selecting, creating, or transforming raw data into features that can be used in machine learning models. It's a crucial step that often requires domain knowledge and can significantly impact model performance.

Slide 6: Source Code for Feature Engineering

```python
import math

# Sample dataset: House information
houses = [
    {"size": 1500, "bedrooms": 3, "location": "suburb"},
    {"size": 2200, "bedrooms": 4, "location": "city"},
    {"size": 1800, "bedrooms": 3, "location": "rural"}
]

def engineer_features(house):
    # Create new features
    house["size_per_bedroom"] = house["size"] / house["bedrooms"]
    house["is_large"] = house["size"] > 2000
    
    # Encode categorical variable
    location_encoding = {"suburb": 1, "city": 2, "rural": 0}
    house["location_code"] = location_encoding[house["location"]]
    
    # Normalize size feature
    max_size = max(h["size"] for h in houses)
    house["normalized_size"] = house["size"] / max_size
    
    return house

# Apply feature engineering to all houses
engineered_houses = [engineer_features(house) for house in houses]

# Print the first engineered house
print(engineered_houses[0])
```

Slide 7: Model Selection

Model selection is the process of choosing the most appropriate machine learning algorithm for a given problem. Factors to consider include the nature of the data, the problem type (classification, regression, clustering, etc.), computational resources, and desired outcomes.

Slide 8: Source Code for Model Selection

```python
import random

# Simple dataset
X = [[random.uniform(0, 10)] for _ in range(100)]
y = [x[0] * 2 + random.uniform(-1, 1) for x in X]

# Simple Linear Regression
def linear_regression(X, y):
    n = len(X)
    sum_x = sum(x[0] for x in X)
    sum_y = sum(y)
    sum_xy = sum(X[i][0] * y[i] for i in range(n))
    sum_xx = sum(x[0]**2 for x in X)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2)
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept

# Decision Tree (very simplified version)
def decision_tree(X, y, max_depth=3):
    def split(X, y):
        best_split = None
        best_gini = float('inf')
        for i in range(len(X)):
            left = [j for j in range(len(X)) if X[j][0] <= X[i][0]]
            right = [j for j in range(len(X)) if X[j][0] > X[i][0]]
            gini = (len(left) * gini_impurity([y[j] for j in left]) +
                    len(right) * gini_impurity([y[j] for j in right])) / len(X)
            if gini < best_gini:
                best_gini = gini
                best_split = (X[i][0], left, right)
        return best_split

    def gini_impurity(y):
        classes = set(y)
        return 1 - sum((y.count(c) / len(y))**2 for c in classes)

    def build_tree(X, y, depth=0):
        if depth == max_depth or len(set(y)) == 1:
            return max(set(y), key=y.count)
        split_value, left, right = split(X, y)
        return {
            'split_value': split_value,
            'left': build_tree([X[i] for i in left], [y[i] for i in left], depth+1),
            'right': build_tree([X[i] for i in right], [y[i] for i in right], depth+1)
        }

    return build_tree(X, y)

# Model selection
lr_slope, lr_intercept = linear_regression(X, y)
dt = decision_tree(X, y)

print("Linear Regression: y =", lr_slope, "* x +", lr_intercept)
print("Decision Tree:", dt)
```

Slide 9: Model Training and Evaluation

Model training involves using algorithms to learn patterns from data, while evaluation assesses the model's performance. Common evaluation metrics include accuracy, precision, recall, and F1-score for classification tasks, and mean squared error or R-squared for regression tasks.

Slide 10: Source Code for Model Training and Evaluation

```python
import random

# Generate synthetic data
X = [[random.uniform(0, 10)] for _ in range(1000)]
y = [1 if x[0] > 5 else 0 for x in X]

# Split data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Simple logistic regression model
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = [random.uniform(-1, 1)]
        self.bias = random.uniform(-1, 1)
    
    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                y_pred = sigmoid(self.weights[0] * xi[0] + self.bias)
                error = y_pred - yi
                self.weights[0] -= self.lr * error * xi[0]
                self.bias -= self.lr * error
    
    def predict(self, X):
        return [1 if sigmoid(self.weights[0] * xi[0] + self.bias) > 0.5 else 0 for xi in X]

# Train model
model = LogisticRegression()
model.train(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = sum(1 for yp, yt in zip(y_pred, y_test) if yp == yt) / len(y_test)
print(f"Accuracy: {accuracy:.2f}")
```

Slide 11: Hyperparameter Tuning

Hyperparameter tuning is the process of optimizing the configuration of a machine learning model to improve its performance. This involves adjusting parameters that are not learned from the data but set before training begins, such as learning rate, number of hidden layers, or regularization strength.

Slide 12: Source Code for Hyperparameter Tuning

```python
import random

# Generate synthetic data
X = [[random.uniform(0, 10)] for _ in range(1000)]
y = [1 if x[0] > 5 else 0 for x in X]

# Split data
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

class LogisticRegression:
    def __init__(self, learning_rate, epochs):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = [random.uniform(-1, 1)]
        self.bias = random.uniform(-1, 1)
    
    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                y_pred = 1 / (1 + math.exp(-(self.weights[0] * xi[0] + self.bias)))
                error = y_pred - yi
                self.weights[0] -= self.lr * error * xi[0]
                self.bias -= self.lr * error
    
    def predict(self, X):
        return [1 if 1 / (1 + math.exp(-(self.weights[0] * xi[0] + self.bias))) > 0.5 else 0 for xi in X]

def evaluate(model, X, y):
    y_pred = model.predict(X)
    return sum(1 for yp, yt in zip(y_pred, y) if yp == yt) / len(y)

# Grid search for hyperparameter tuning
learning_rates = [0.001, 0.01, 0.1]
epochs_list = [50, 100, 200]

best_accuracy = 0
best_params = None

for lr in learning_rates:
    for epochs in epochs_list:
        model = LogisticRegression(learning_rate=lr, epochs=epochs)
        model.train(X_train, y_train)
        accuracy = evaluate(model, X_val, y_val)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (lr, epochs)

print(f"Best parameters: Learning Rate = {best_params[0]}, Epochs = {best_params[1]}")
print(f"Best validation accuracy: {best_accuracy:.2f}")
```

Slide 13: Real-Life Example: Spam Detection

Spam detection is a common application of machine learning in email filtering. It involves classifying incoming emails as either spam or legitimate (ham) based on their content and metadata. This task typically uses techniques from natural language processing and classification algorithms.

Slide 14: Source Code for Real-Life Example: Spam Detection

```python
import re
import math
import random

# Sample emails (content only)
emails = [
    ("Get rich quick! Buy now!", "spam"),
    ("Meeting at 3pm tomorrow", "ham"),
    ("You've won a free iPhone!", "spam"),
    ("Project report due next week", "ham"),
    ("Congratulations! You're our lucky winner", "spam"),
    ("Reminder: dentist appointment on Friday", "ham")
]

def preprocess(text):
    return re.findall(r'\w+', text.lower())

def create_bow(emails):
    word_set = set(word for email, _ in emails for word in preprocess(email))
    return {word: i for i, word in enumerate(word_set)}

def vectorize(email, bow):
    words = preprocess(email)
    return [words.count(word) for word in bow]

def train_naive_bayes(X, y):
    n_spam = sum(1 for label in y if label == "spam")
    n_ham = len(y) - n_spam
    spam_word_counts = [1] * len(X[0])
    ham_word_counts = [1] * len(X[0])
    
    for features, label in zip(X, y):
        if label == "spam":
            spam_word_counts = [s + f for s, f in zip(spam_word_counts, features)]
        else:
            ham_word_counts = [h + f for h, f in zip(ham_word_counts, features)]
    
    spam_probs = [count / (n_spam + 2) for count in spam_word_counts]
    ham_probs = [count / (n_ham + 2) for count in ham_word_counts]
    return spam_probs, ham_probs, n_spam / len(y)

def classify(email, bow, spam_probs, ham_probs, spam_prior):
    features = vectorize(email, bow)
    spam_score = math.log(spam_prior)
    ham_score = math.log(1 - spam_prior)
    
    for f, sp, hp in zip(features, spam_probs, ham_probs):
        if f > 0:
            spam_score += math.log(sp)
            ham_score += math.log(hp)
    
    return "spam" if spam_score > ham_score else "ham"

# Prepare data and train model
random.shuffle(emails)
bow = create_bow(emails)
X = [vectorize(email, bow) for email, _ in emails]
y = [label for _, label in emails]
spam_probs, ham_probs, spam_prior = train_naive_bayes(X, y)

# Test classification
test_email = "Congratulations! You've won a million dollars!"
result = classify(test_email, bow, spam_probs, ham_probs, spam_prior)
print(f"The email '{test_email}' is classified as: {result}")
```

Slide 15: Real-Life Example: Image Classification

Image classification is a fundamental task in computer vision, with applications ranging from facial recognition to medical diagnosis. It involves categorizing images into predefined classes based on their visual content. While deep learning approaches like Convolutional Neural Networks (CNNs) are commonly used for this task, we'll implement a simplified version using a basic algorithm.

Slide 16: Source Code for Real-Life Example: Image Classification

```python
import random

# Simulate a simple 5x5 grayscale image
def create_image(is_circle=False):
    image = [[0 for _ in range(5)] for _ in range(5)]
    if is_circle:
        # Create a rough circle shape
        circle_pixels = [(1,1), (1,2), (1,3), (2,1), (2,3), (3,1), (3,2), (3,3)]
        for x, y in circle_pixels:
            image[x][y] = 1
    else:
        # Create a rough square shape
        for i in range(1, 4):
            for j in range(1, 4):
                image[i][j] = 1
    return image

# Generate dataset
def generate_dataset(n_samples):
    images = []
    labels = []
    for _ in range(n_samples):
        is_circle = random.choice([True, False])
        images.append(create_image(is_circle))
        labels.append("circle" if is_circle else "square")
    return images, labels

# Simple feature extraction: count non-zero pixels
def extract_features(image):
    return sum(sum(row) for row in image)

# Train a simple threshold classifier
def train_classifier(X, y):
    circle_counts = [x for x, label in zip(X, y) if label == "circle"]
    square_counts = [x for x, label in zip(X, y) if label == "square"]
    threshold = (sum(circle_counts) / len(circle_counts) + 
                 sum(square_counts) / len(square_counts)) / 2
    return threshold

# Classify new image
def classify_image(image, threshold):
    feature = extract_features(image)
    return "circle" if feature < threshold else "square"

# Generate dataset and train classifier
X_train, y_train = generate_dataset(100)
X_train = [extract_features(img) for img in X_train]
threshold = train_classifier(X_train, y_train)

# Test classification
test_image = create_image(is_circle=True)
result = classify_image(test_image, threshold)
print(f"The test image is classified as: {result}")

# Print the test image
for row in test_image:
    print(" ".join(["■" if pixel else "□" for pixel in row]))
```

Slide 17: Additional Resources

For those interested in diving deeper into machine learning, here are some valuable resources:

1.  ArXiv.org - A repository of electronic preprints of scientific papers: [https://arxiv.org/list/cs.LG/recent](https://arxiv.org/list/cs.LG/recent)
2.  "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville: [https://arxiv.org/abs/1607.06416](https://arxiv.org/abs/1607.06416)
3.  "Machine Learning: A Probabilistic Perspective" by Kevin P. Murphy: [https://mitpress.mit.edu/books/machine-learning-1](https://mitpress.mit.edu/books/machine-learning-1)
4.  "Pattern Recognition and Machine Learning" by Christopher M. Bishop: [https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)

These resources provide in-depth coverage of machine learning concepts, algorithms, and applications, suitable for both beginners and advanced practitioners.

