## 25 Key Mathematical Definitions for Data Science
Slide 1: Gradient Descent

Gradient descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of steepest descent. It's widely used in machine learning for training models.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 5*np.sin(x)

def df(x):
    return 2*x + 5*np.cos(x)

def gradient_descent(start, learn_rate, num_iter):
    x = start
    x_history = [x]
    
    for _ in range(num_iter):
        x = x - learn_rate * df(x)
        x_history.append(x)
    
    return x, x_history

x_min, x_history = gradient_descent(start=2, learn_rate=0.1, num_iter=50)

x = np.linspace(-5, 5, 100)
plt.plot(x, f(x))
plt.scatter(x_history, [f(x) for x in x_history], c='r', s=20)
plt.title('Gradient Descent Optimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

print(f"Minimum found at x = {x_min:.2f}")
```

Slide 2: Normal Distribution

The normal distribution, also known as the Gaussian distribution, is a probability distribution that is symmetric about the mean. It's characterized by its bell-shaped curve and is fundamental in statistics and data science.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data points
x = np.linspace(-4, 4, 100)

# Calculate probability density function
y = norm.pdf(x, loc=0, scale=1)

# Plot the distribution
plt.plot(x, y)
plt.title('Standard Normal Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()

# Generate random samples from the distribution
samples = np.random.normal(loc=0, scale=1, size=1000)

# Plot histogram of samples
plt.hist(samples, bins=30, density=True, alpha=0.7)
plt.plot(x, y, 'r-', lw=2)
plt.title('Histogram of Samples vs. Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```

Slide 3: Sigmoid Function

The sigmoid function is an S-shaped curve that maps any input to a value between 0 and 1. It's commonly used in logistic regression and neural networks as an activation function.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.plot(x, y)
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid(True)
plt.show()

# Example: Using sigmoid in a simple neural network
def neural_network(input_layer, weights, bias):
    layer = np.dot(input_layer, weights) + bias
    return sigmoid(layer)

input_layer = np.array([0.1, 0.2, 0.3])
weights = np.array([0.4, 0.5, 0.6])
bias = 0.1

output = neural_network(input_layer, weights, bias)
print(f"Neural network output: {output}")
```

Slide 4: Correlation

Correlation measures the strength and direction of the relationship between two variables. It ranges from -1 to 1, where 1 indicates a perfect positive correlation, -1 a perfect negative correlation, and 0 no linear correlation.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate correlated data
x = np.random.normal(0, 1, 1000)
y = 2 * x + np.random.normal(0, 1, 1000)

# Calculate correlation coefficient
correlation, _ = stats.pearsonr(x, y)

# Plot the data
plt.scatter(x, y, alpha=0.5)
plt.title(f'Scatter Plot (Correlation: {correlation:.2f})')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Generate correlation matrix for multiple variables
data = np.column_stack((x, y, 2*y + np.random.normal(0, 1, 1000)))
correlation_matrix = np.corrcoef(data.T)

plt.imshow(correlation_matrix, cmap='coolwarm')
plt.colorbar()
plt.title('Correlation Matrix')
plt.xticks(range(3), ['X', 'Y', 'Z'])
plt.yticks(range(3), ['X', 'Y', 'Z'])
plt.show()
```

Slide 5: Cosine Similarity

Cosine similarity measures the cosine of the angle between two non-zero vectors in an inner product space. It's often used in text analysis and recommendation systems to determine how similar two documents or items are.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Example: Document similarity
doc1 = np.array([1, 1, 0, 1, 0, 1])
doc2 = np.array([1, 1, 1, 0, 1, 0])
doc3 = np.array([1, 1, 0, 1, 0, 1])

print("Cosine similarity between doc1 and doc2:", cosine_sim(doc1, doc2))
print("Cosine similarity between doc1 and doc3:", cosine_sim(doc1, doc3))

# Using sklearn for multiple documents
docs = np.array([doc1, doc2, doc3])
similarity_matrix = cosine_similarity(docs)
print("\nSimilarity matrix:")
print(similarity_matrix)
```

Slide 6: Naive Bayes

Naive Bayes is a probabilistic classifier based on Bayes' theorem with the "naive" assumption of independence between features. It's particularly useful for text classification tasks.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
texts = [
    "I love this movie", "Great film", "Terrible movie", "I hate this film",
    "Awesome picture", "Boring film", "Excellent movie", "Worst ever"
]
labels = [1, 1, 0, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a bag of words representation
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Make predictions
y_pred = clf.predict(X_test_vec)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Predict a new review
new_review = ["This movie is amazing"]
new_review_vec = vectorizer.transform(new_review)
prediction = clf.predict(new_review_vec)
print("\nPrediction for new review:", "Positive" if prediction[0] == 1 else "Negative")
```

Slide 7: F1 Score

The F1 score is the harmonic mean of precision and recall, providing a single score that balances both metrics. It's particularly useful for evaluating classification models with imbalanced datasets.

```python
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

def calculate_f1(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1, precision, recall

# Example: Binary classification results
y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])

f1, precision, recall = calculate_f1(y_true, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Using sklearn's implementation
sklearn_f1 = f1_score(y_true, y_pred)
print(f"Sklearn F1 Score: {sklearn_f1:.2f}")

# Visualize confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

Slide 8: ReLU (Rectified Linear Unit)

ReLU is an activation function commonly used in neural networks. It outputs the input directly if it's positive, otherwise, it outputs zero. This non-linearity allows neural networks to learn complex patterns.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-10, 10, 100)
y = relu(x)

plt.plot(x, y)
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.grid(True)
plt.show()

# Example: Using ReLU in a simple neural network layer
def neural_network_layer(input_data, weights, bias):
    layer = np.dot(input_data, weights) + bias
    return relu(layer)

input_data = np.array([1, 2, 3, 4, 5])
weights = np.random.randn(5, 3)
bias = np.random.randn(3)

output = neural_network_layer(input_data, weights, bias)
print("Neural network layer output:")
print(output)
```

Slide 9: Softmax Function

The softmax function is used to convert a vector of numbers into a probability distribution. It's commonly used as the final activation function in multi-class classification problems.

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

# Example: Converting logits to probabilities
logits = np.array([2.0, 1.0, 0.1])
probabilities = softmax(logits)

print("Logits:", logits)
print("Probabilities:", probabilities)
print("Sum of probabilities:", np.sum(probabilities))

# Visualize softmax for different temperature values
def softmax_with_temperature(x, temperature):
    return softmax(x / temperature)

x = np.linspace(-5, 5, 100)
temperatures = [0.5, 1.0, 2.0]

plt.figure(figsize=(10, 6))
for temp in temperatures:
    y = softmax_with_temperature(x[:, np.newaxis], temp)
    plt.plot(x, y[:, 0], label=f'T={temp}')

plt.title('Softmax Function with Different Temperatures')
plt.xlabel('Input')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 10: Mean Squared Error (MSE)

Mean Squared Error is a common loss function used in regression problems. It measures the average squared difference between the predicted and actual values.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y_true = 2 * X + 1 + np.random.normal(0, 1, 100)
y_pred = 2.2 * X + 0.8  # Slightly off predictions

# Calculate MSE
mse_value = mse(y_true, y_pred)
sklearn_mse = mean_squared_error(y_true, y_pred)

print(f"Custom MSE: {mse_value:.4f}")
print(f"Sklearn MSE: {sklearn_mse:.4f}")

# Visualize the data and predictions
plt.scatter(X, y_true, label='True values')
plt.plot(X, y_pred, color='red', label='Predictions')
plt.title(f'True vs Predicted Values (MSE: {mse_value:.4f})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Visualize the squared errors
squared_errors = (y_true - y_pred) ** 2
plt.scatter(X, squared_errors)
plt.title('Squared Errors')
plt.xlabel('X')
plt.ylabel('Squared Error')
plt.show()
```

Slide 11: MSE with L2 Regularization

L2 regularization, also known as ridge regression, adds a penalty term to the loss function to prevent overfitting. This is particularly useful when dealing with high-dimensional data or when features are correlated.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.normal(0, 0.1, (100, 1))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models with different alpha values
alphas = [0, 0.1, 1, 10]
for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    
    print(f"Alpha: {alpha}, Test MSE: {mse:.4f}")

    plt.scatter(X_test, y_test, color='blue', label='True values')
    plt.plot(X_test, y_pred_test, color='red', label=f'Predictions (α={alpha})')
    plt.title(f'Ridge Regression (α={alpha})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()
```

Slide 12: K-Means Clustering

K-Means is an unsupervised learning algorithm used for clustering data into K groups. It works by iteratively assigning data points to the nearest centroid and updating the centroids based on the assigned points.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2)
X[:100, 0] += 2
X[100:200, 0] -= 2
X[200:, 1] += 2

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='r', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Print cluster centers
print("Cluster centers:")
print(kmeans.cluster_centers_)
```

Slide 13: Linear Regression

Linear regression is a supervised learning algorithm used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Coefficients: {model.coef_[0][0]:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"Mean squared error: {mse:.4f}")
print(f"R-squared score: {r2:.4f}")

# Plot the results
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Linear regression')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

Slide 14: Support Vector Machine (SVM)

SVM is a powerful supervised learning algorithm used for classification and regression tasks. It aims to find the hyperplane that best separates different classes in the feature space.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=1, n_clusters_per_class=1)

# Create and fit the SVM model
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')

# Create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Classification')
plt.show()

# Print support vectors
print("Number of support vectors:", len(clf.support_vectors_))
print("Support vectors:")
print(clf.support_vectors_)
```

Slide 15: Log Loss (Binary Cross-Entropy)

Log loss, also known as binary cross-entropy, is a common loss function used in binary classification problems. It measures the performance of a model whose output is a probability value between 0 and 1.

```python
import numpy as np
import matplotlib.pyplot as plt

def log_loss(y_true, y_pred):
    epsilon = 1e-15  # Small value to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Generate sample data
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.linspace(0, 1, 100)

# Calculate log loss for each prediction
losses = [log_loss(y_true, np.full_like(y_true, p)) for p in y_pred]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_pred, losses)
plt.title('Log Loss')
plt.xlabel('Predicted Probability')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Example calculations
print("Log loss for perfect predictions:")
print(log_loss(y_true, y_true))

print("\nLog loss for worst predictions:")
print(log_loss(y_true, 1 - y_true))

print("\nLog loss for random guessing:")
print(log_loss(y_true, np.full_like(y_true, 0.5)))
```

Slide 16: Additional Resources

For more in-depth information on these topics and advanced mathematical concepts in data science, consider exploring the following resources:

1. ArXiv.org: A repository of electronic preprints of scientific papers in various fields, including mathematics and computer science. URL: [https://arxiv.org/](https://arxiv.org/)
2. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville ArXiv reference: [https://arxiv.org/abs/1607.06036](https://arxiv.org/abs/1607.06036)
3. "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman Stanford University resource: [https://web.stanford.edu/~hastie/ElemStatLearn/](https://web.stanford.edu/~hastie/ElemStatLearn/)
4. "Pattern Recognition and Machine Learning" by Christopher Bishop Microsoft Research resource: [https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)

These resources provide comprehensive coverage of mathematical foundations and advanced techniques in data science and machine learning.

