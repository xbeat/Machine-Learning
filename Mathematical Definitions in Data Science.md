## Mathematical Definitions in Data Science
Slide 1: Gradient Descent

Gradient Descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of steepest descent. In data science, it's commonly used to train machine learning models by updating parameters to minimize the cost function.

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(start, gradient, learn_rate, n_iter=100, tolerance=1e-06):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector

# Example: Finding the minimum of f(x) = x^2 + 5
gradient = lambda x: 2 * x
start = np.array([10.0])
result = gradient_descent(start, gradient, learn_rate=0.1)

print(f"Minimum found at: {result[0]:.2f}")

# Plotting
x = np.linspace(-15, 15, 100)
y = x**2 + 5
plt.plot(x, y)
plt.plot(result, result**2 + 5, 'ro')
plt.title('Gradient Descent Example')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
```

Slide 2: Normal Distribution

The Normal (or Gaussian) distribution is a probability distribution that is symmetric about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean. It's characterized by its mean (μ) and standard deviation (σ).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def normal_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

x = np.linspace(-5, 5, 100)
mu, sigma = 0, 1

plt.figure(figsize=(10, 6))
plt.plot(x, normal_pdf(x, mu, sigma), label='PDF')
plt.plot(x, norm.cdf(x, mu, sigma), label='CDF')
plt.title('Normal Distribution (μ=0, σ=1)')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()

# Example: Calculating probabilities
prob_within_1_sigma = norm.cdf(1, mu, sigma) - norm.cdf(-1, mu, sigma)
print(f"Probability within 1 sigma: {prob_within_1_sigma:.4f}")
```

Slide 3: Z-score

The Z-score represents the number of standard deviations by which an observation is above or below the mean. It's used to compare values from different normal distributions and to identify outliers.

```python
import numpy as np
from scipy import stats

def calculate_z_score(x, data):
    return (x - np.mean(data)) / np.std(data)

# Example: Student test scores
scores = np.array([75, 80, 85, 90, 95, 100])
student_score = 92

z_score = calculate_z_score(student_score, scores)
print(f"Z-score for {student_score}: {z_score:.2f}")

# Interpreting Z-score
percentile = stats.norm.cdf(z_score) * 100
print(f"Percentile: {percentile:.2f}%")

# Visualizing Z-score
plt.figure(figsize=(10, 6))
plt.hist(scores, bins=10, density=True, alpha=0.7, color='skyblue')
plt.axvline(student_score, color='red', linestyle='dashed', linewidth=2)
plt.title(f'Distribution of Scores (Z-score = {z_score:.2f})')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()
```

Slide 4: Sigmoid Function

The Sigmoid function, also known as the logistic function, is an S-shaped curve that maps any input value to a value between 0 and 1. It's commonly used in machine learning, particularly in logistic regression and neural networks, as an activation function.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.grid(True)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axvline(x=0, color='r', linestyle='--')
plt.text(0, 0.55, 'y = 0.5', fontsize=12, color='red')
plt.text(0.5, 0.05, 'x = 0', fontsize=12, color='red')
plt.show()

# Example: Using sigmoid in binary classification
input_values = np.array([-2, 0, 2])
probabilities = sigmoid(input_values)
print("Input values:", input_values)
print("Probabilities:", probabilities)
```

Slide 5: Correlation

Correlation measures the strength and direction of the linear relationship between two variables. It ranges from -1 to 1, where 1 indicates a perfect positive correlation, -1 a perfect negative correlation, and 0 no linear correlation.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def correlation(x, y):
    return np.cov(x, y)[0, 1] / (np.std(x) * np.std(y))

# Generate correlated data
np.random.seed(42)
x = np.random.normal(0, 1, 1000)
y = 0.8 * x + np.random.normal(0, 0.5, 1000)

corr = correlation(x, y)
print(f"Correlation: {corr:.4f}")

# Visualize correlation
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)
plt.title(f'Scatter Plot (Correlation: {corr:.4f})')
plt.xlabel('X')
plt.ylabel('Y')

# Add regression line
slope, intercept, _, _, _ = stats.linregress(x, y)
line = slope * x + intercept
plt.plot(x, line, color='red', label='Regression Line')
plt.legend()
plt.show()
```

Slide 6: Cosine Similarity

Cosine Similarity measures the cosine of the angle between two non-zero vectors in an inner product space. It's often used in text analysis and recommendation systems to determine how similar two documents or items are, regardless of their magnitude.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example: Document similarity
doc1 = np.array([1, 1, 0, 1, 0, 1])
doc2 = np.array([1, 1, 1, 0, 1, 0])
doc3 = np.array([1, 1, 0, 1, 0, 1])

print("Cosine similarity between doc1 and doc2:", cosine_sim(doc1, doc2))
print("Cosine similarity between doc1 and doc3:", cosine_sim(doc1, doc3))

# Using sklearn for multiple documents
docs = np.array([doc1, doc2, doc3])
similarity_matrix = cosine_similarity(docs)

print("\nSimilarity Matrix:")
print(similarity_matrix)

# Visualize similarity matrix
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(similarity_matrix, annot=True, cmap='YlGnBu')
plt.title('Document Similarity Heatmap')
plt.show()
```

Slide 7: Naive Bayes

Naive Bayes is a probabilistic classifier based on Bayes' theorem with an assumption of independence between features. It's widely used in text classification, spam filtering, and recommendation systems due to its simplicity and effectiveness.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Example: Text classification
texts = [
    "I love this sandwich", "This is an amazing place",
    "I feel very good about these shoes", "This is my best work",
    "What an awesome view", "I do not like this restaurant",
    "I am tired of this stuff", "I can't deal with this",
    "He is my sworn enemy", "My boss is horrible"
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1: positive, 0: negative

# Vectorize the text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train the model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example prediction
new_text = ["This product is amazing"]
new_X = vectorizer.transform(new_text)
prediction = clf.predict(new_X)
print("\nPrediction for 'This product is amazing':", "Positive" if prediction[0] == 1 else "Negative")
```

Slide 8: Maximum Likelihood Estimation (MLE)

Maximum Likelihood Estimation is a method of estimating the parameters of a probability distribution by maximizing a likelihood function. It's widely used in statistics and machine learning for parameter estimation in various models.

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(42)
true_mu, true_sigma = 5, 2
data = np.random.normal(true_mu, true_sigma, 1000)

# Define negative log-likelihood function
def neg_log_likelihood(params, data):
    mu, sigma = params
    return -np.sum(norm.logpdf(data, mu, sigma))

# Perform MLE
initial_guess = [0, 1]
result = minimize(neg_log_likelihood, initial_guess, args=(data,), method='Nelder-Mead')

estimated_mu, estimated_sigma = result.x
print(f"Estimated μ: {estimated_mu:.2f}, Estimated σ: {estimated_sigma:.2f}")
print(f"True μ: {true_mu}, True σ: {true_sigma}")

# Visualize results
x = np.linspace(min(data), max(data), 100)
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue')
plt.plot(x, norm.pdf(x, estimated_mu, estimated_sigma), 'r-', lw=2, label='Estimated PDF')
plt.plot(x, norm.pdf(x, true_mu, true_sigma), 'g--', lw=2, label='True PDF')
plt.title('MLE for Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()
```

Slide 9: Ordinary Least Squares (OLS)

Ordinary Least Squares is a method for estimating the unknown parameters in a linear regression model. It minimizes the sum of the squares of the differences between the observed and predicted values of the dependent variable.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_test = np.array([[0], [2]])
y_pred = model.predict(X_test)

# Print results
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Coefficient: {model.coef_[0][0]:.2f}")
print(f"R-squared: {r2_score(y, model.predict(X)):.2f}")
print(f"Mean squared error: {mean_squared_error(y, model.predict(X)):.2f}")

# Visualize results
plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('Ordinary Least Squares Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Manual OLS calculation
X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
beta_manual = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
print(f"\nManual OLS calculation:")
print(f"Intercept: {beta_manual[0][0]:.2f}")
print(f"Coefficient: {beta_manual[1][0]:.2f}")
```

Slide 10: F1 Score

The F1 Score is the harmonic mean of precision and recall, providing a single score that balances both metrics. It's particularly useful when you have an uneven class distribution and is commonly used in classification tasks to measure a model's accuracy.

```python
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example: Binary classification results
y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])

# Calculate metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Visualize confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

Slide 11: ReLU (Rectified Linear Unit)

ReLU is an activation function commonly used in neural networks. It returns 0 for negative inputs and the input value for positive inputs, introducing non-linearity to the model without causing vanishing gradient problems.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-10, 10, 100)
y = relu(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='ReLU')
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.legend()
plt.grid(True)
plt.show()

# Example: Applying ReLU to an array
input_array = np.array([-2, -1, 0, 1, 2])
output_array = relu(input_array)
print("Input:", input_array)
print("Output:", output_array)
```

Slide 12: Softmax Function

The Softmax function is used in multi-class classification problems to convert a vector of raw scores into a probability distribution. It ensures all output values are between 0 and 1 and sum to 1.

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum()

# Example: Softmax for multi-class classification
scores = np.array([2.0, 1.0, 0.1])
probabilities = softmax(scores)

print("Scores:", scores)
print("Probabilities:", probabilities)
print("Sum of probabilities:", np.sum(probabilities))

# Visualize softmax output
plt.figure(figsize=(10, 6))
plt.bar(range(len(probabilities)), probabilities)
plt.title('Softmax Output')
plt.xlabel('Class')
plt.ylabel('Probability')
plt.xticks(range(len(probabilities)), ['Class 1', 'Class 2', 'Class 3'])
plt.show()
```

Slide 13: R² Score (Coefficient of Determination)

The R² score, also known as the coefficient of determination, measures the proportion of variance in the dependent variable that is predictable from the independent variable(s). It ranges from 0 to 1, with 1 indicating perfect prediction.

```python
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y_true = 3*x + 2 + np.random.normal(0, 1, 100)
y_pred = 3*x + 2

# Calculate R² score
r2 = r2_score(y_true, y_pred)

print(f"R² Score: {r2:.4f}")

# Visualize the data and prediction
plt.figure(figsize=(10, 6))
plt.scatter(x, y_true, alpha=0.5, label='True data')
plt.plot(x, y_pred, color='red', label='Prediction')
plt.title(f'Data vs. Prediction (R² = {r2:.4f})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

Slide 14: Mean Squared Error (MSE)

Mean Squared Error is a common loss function used in regression problems. It measures the average squared difference between the estimated values and the actual value, penalizing larger errors more heavily.

```python
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
y_true = np.array([3, 2, 7, 1, 5, 4, 6, 8])
y_pred = np.array([2.5, 3.5, 6, 1.5, 4.5, 4, 5.5, 7])

# Calculate MSE
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Visualize predictions vs true values
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
plt.title(f'True vs Predicted Values (MSE = {mse:.4f})')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.show()

# Visualize errors
errors = y_pred - y_true
plt.figure(figsize=(10, 6))
plt.bar(range(len(errors)), errors)
plt.title('Prediction Errors')
plt.xlabel('Sample')
plt.ylabel('Error')
plt.show()
```

Slide 15: MSE with L2 Regularization

MSE with L2 Regularization, also known as Ridge Regression, adds a penalty term to the standard MSE loss function. This helps prevent overfitting by discouraging large coefficient values.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models with different regularization strengths
alphas = [0, 0.1, 1, 10]
models = [Ridge(alpha=alpha).fit(X_train, y_train) for alpha in alphas]

# Plot results
plt.figure(figsize=(12, 8))
for i, model in enumerate(models):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    plt.subplot(2, 2, i+1)
    plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='True')
    plt.plot(X_test, y_pred, color='red', label='Predicted')
    plt.title(f'Alpha = {alphas[i]}, MSE = {mse:.4f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()

plt.tight_layout()
plt.show()
```

Slide 16: Eigenvectors

Eigenvectors are vectors that, when a linear transformation is applied, change only by a scalar factor. This concept is crucial in many areas of data science, including Principal Component Analysis (PCA) and spectral clustering.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a 2x2 matrix
A = np.array([[4, 1],
              [2, 3]])

# Calculate eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print("\nEigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)

# Visualize eigenvectors
plt.figure(figsize=(8, 8))
plt.axvline(x=0, color='k', linestyle='--')
plt.axhline(y=0, color='k', linestyle='--')

# Plot original vectors
plt.quiver(0, 0, A[0, 0], A[1, 0], angles='xy', scale_units='xy', scale=1, color='r', label='Original vector 1')
plt.quiver(0, 0, A[0, 1], A[1, 1], angles='xy', scale_units='xy', scale=1, color='b', label='Original vector 2')

# Plot eigenvectors
for i in range(2):
    plt.quiver(0, 0, eigenvectors[0, i], eigenvectors[1, i], angles='xy', scale_units='xy', scale=1, color='g', label=f'Eigenvector {i+1}')

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.legend()
plt.title('Eigenvectors of Matrix A')
plt.show()
```

Slide 17: Entropy

Entropy is a measure of the average amount of information contained in a message. In data science, it's often used in decision trees to determine the best splitting criteria and in information theory for compression algorithms.

```python
import numpy as np
import matplotlib.pyplot as plt

def entropy(p):
    return -np.sum(p * np.log2(p + 1e-10))  # Add small value to avoid log(0)

# Calculate entropy for different probability distributions
p_values = np.linspace(0.01, 0.99, 100)
entropies = [entropy(np.array([p, 1-p])) for p in p_values]

plt.figure(figsize=(10, 6))
plt.plot(p_values, entropies)
plt.title('Entropy of a Binary Event')
plt.xlabel('Probability of Event')
plt.ylabel('Entropy')
plt.grid(True)
plt.show()

# Example: Calculate entropy of a dice roll
dice_probs = np.array([1/6] * 6)
dice_entropy = entropy(dice_probs)
print(f"Entropy of a fair dice roll: {dice_entropy:.4f} bits")
```

Slide 18: K-Means Clustering

K-Means is an unsupervised learning algorithm used for clustering data into K groups. It aims to minimize the within-cluster sum of squares by iteratively assigning data points to the nearest centroid and updating centroids.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
n_samples = 300
n_clusters = 3
X, y = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=0.60, random_state=42)

# Perform K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Calculate and print inertia (within-cluster sum of squares)
print(f"Inertia: {kmeans.inertia_:.2f}")
```

Slide 19: KL Divergence

Kullback-Leibler Divergence measures the difference between two probability distributions. It's often used in machine learning for comparing distributions, particularly in variational inference and model selection.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# Generate two normal distributions
x = np.linspace(-5, 5, 1000)
p = norm.pdf(x, loc=0, scale=1)
q = norm.pdf(x, loc=1, scale=1.5)

# Calculate KL divergence
kl_pq = kl_divergence(p, q)
kl_qp = kl_divergence(q, p)

# Plot distributions
plt.figure(figsize=(10, 6))
plt.plot(x, p, label='P ~ N(0, 1)')
plt.plot(x, q, label='Q ~ N(1, 1.5)')
plt.title(f'KL(P||Q) = {kl_pq:.4f}, KL(Q||P) = {kl_qp:.4f}')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

print(f"KL(P||Q) = {kl_pq:.4f}")
print(f"KL(Q||P) = {kl_qp:.4f}")
```

Slide 20: Log-loss (Binary Cross-Entropy)

Log-loss, also known as binary cross-entropy, is a common loss function used in binary classification problems. It measures the performance of a classification model whose output is a probability value between 0 and 1.

```python
import numpy as np
import matplotlib.pyplot as plt

def log_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Generate sample predictions
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.9, 0.2, 0.1])

loss = log_loss(y_true, y_pred)
print(f"Log-loss: {loss:.4f}")

# Visualize log-loss for different prediction probabilities
p = np.linspace(0.01, 0.99, 100)
loss_1 = -np.log(p)
loss_0 = -np.log(1 - p)

plt.figure(figsize=(10, 6))
plt.plot(p, loss_1, label='True class = 1')
plt.plot(p, loss_0, label='True class = 0')
plt.title('Log-loss for Binary Classification')
plt.xlabel('Predicted Probability')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 21: Support Vector Machine (SVM)

Support Vector Machine is a powerful supervised learning algorithm used for classification and regression tasks. It aims to find the hyperplane that best separates different classes in the feature space, maximizing the margin between classes.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

# Fit the SVM model
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')

# Plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

# Plot decision boundary and margins
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Plot support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='k')

plt.title('Support Vector Machine Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

Slide 22: Linear Regression

Linear Regression is a fundamental statistical and machine learning technique used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_test = np.array([[0], [2]])
y_pred = model.predict(X_test)

# Print results
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Coefficient: {model.coef_[0][0]:.2f}")
print(f"R-squared: {r2_score(y, model.predict(X)):.2f}")

# Visualize results
plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
```

Slide 23: Singular Value Decomposition (SVD)

Singular Value Decomposition is a matrix factorization method that decomposes a matrix into three matrices: U, Σ (Sigma), and V^T. It's widely used in dimensionality reduction, data compression, and recommender systems.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a sample matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Perform SVD
U, s, VT = np.linalg.svd(A)

print("Original matrix A:")
print(A)
print("\nLeft singular vectors (U):")
print(U)
print("\nSingular values (s):")
print(s)
print("\nRight singular vectors (V^T):")
print(VT)

# Reconstruct the matrix using different numbers of singular values
k_values = [1, 2, 3]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, k in enumerate(k_values):
    A_reconstructed = U[:, :k] @ np.diag(s[:k]) @ VT[:k, :]
    axes[i].imshow(A_reconstructed, cmap='viridis')
    axes[i].set_title(f'Rank-{k} Approximation')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

Slide 24: Lagrange Multiplier

The Lagrange Multiplier method is used to find the local maxima and minima of a function subject to equality constraints. In machine learning, it's often used in optimization problems, such as finding the optimal parameters in Support Vector Machines.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return x**2 + y**2

def constraint(x, y):
    return x + y - 1

# Create a grid of points
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Calculate Z values
Z = f(X, Y)

# Create 3D plot
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('Function f(x,y) = x² + y²')

# Plot constraint
ax2 = fig.add_subplot(122)
ax2.contour(X, Y, Z, levels=20, cmap='viridis')
ax2.plot(x, 1-x, 'r-', label='Constraint: x + y = 1')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Contour plot with constraint')
ax2.legend()

plt.tight_layout()
plt.show()

# Solve using Lagrange multiplier (analytically)
# The solution is x = y = 1/2
x_opt, y_opt = 0.5, 0.5
print(f"Optimal point: x = {x_opt}, y = {y_opt}")
print(f"Optimal value: f({x_opt}, {y_opt}) = {f(x_opt, y_opt)}")

# Demonstrate the Lagrange multiplier function
def lagrange(x, y, lambda_):
    return f(x, y) - lambda_ * (constraint(x, y))

# Plot the Lagrange function for different lambda values
lambda_values = [0, 0.5, 1, 2]
plt.figure(figsize=(10, 6))

for lambda_ in lambda_values:
    L = lagrange(X, Y, lambda_)
    plt.contour(X, Y, L, levels=20, alpha=0.5)

plt.plot(x, 1-x, 'r-', label='Constraint: x + y = 1')
plt.plot(x_opt, y_opt, 'ro', label='Optimal point')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Lagrange Multiplier Method')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 25: Additional Resources

For further exploration of mathematical concepts in data science, consider the following resources:

1. ArXiv.org - A repository of electronic preprints of scientific papers: [https://arxiv.org/list/stat.ML/recent](https://arxiv.org/list/stat.ML/recent) (Machine Learning section)
2. "Mathematics for Machine Learning" by Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong: ArXiv link: [https://arxiv.org/abs/1811.03175](https://arxiv.org/abs/1811.03175)
3. "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman: Available at: [https://web.stanford.edu/~hastie/ElemStatLearn/](https://web.stanford.edu/~hastie/ElemStatLearn/)

These resources provide in-depth coverage of the mathematical foundations underlying modern data science and machine learning techniques.

Slide 26: Mathematical Formulas (LaTeX-style)

This slide presents the LaTeX representations of the mathematical formulas mentioned earlier in the presentation.

1. Gradient Descent: \[ \\theta\_{j+1} = \\theta\_j - \\alpha \\nabla J(\\theta\_j) \]
2. Normal Distribution: \[ f(x|\\mu,\\sigma^2) = \\frac{1}{\\sigma\\sqrt{2\\pi}} \\exp\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right) \]
3. Z-score: \[ z = \\frac{x - \\mu}{\\sigma} \]
4. Sigmoid Function: \[ \\sigma(x) = \\frac{1}{1 + e^{-x}} \]
5. Correlation: \[ \\text{Correlation} = \\frac{\\text{Cov}(X,Y)}{\\text{Std}(X) \\cdot \\text{Std}(Y)} \]
6. Cosine Similarity: \[ \\text{similarity} = \\frac{A \\cdot B}{|A| |B|} \]
7. Naive Bayes: \[ P(y|x\_1,...,x\_n) = \\frac{P(y) \\prod\_{i=1}^n P(x\_i|y)}{P(x\_1,...,x\_n)} \]
8. Maximum Likelihood Estimation: \[ \\arg\\max\_\\theta \\prod\_{i=1}^n P(x\_i|\\theta) \]
9. Ordinary Least Squares: \[ \\hat{\\beta} = (X^T X)^{-1} X^T y \]
10. F1 Score: \[ \\frac{2 \\cdot P \\cdot R}{P + R} \]
11. ReLU: \[ \\max(0, x) \]
12. Softmax: \[ P(y = j|x) = \\frac{e^{x^T w\_j}}{\\sum\_{k=1}^K e^{x^T w\_k}} \]
13. R² Score: \[ R^2 = 1 - \\frac{\\sum\_{i=1}^n (y\_i - \\hat{y}*i)^2}{\\sum*{i=1}^n (y\_i - \\bar{y})^2} \]
14. Mean Squared Error: \[ \\text{MSE} = \\frac{1}{n} \\sum\_{i=1}^n (y\_i - \\hat{y}\_i)^2 \]
15. MSE with L2 Regularization: \[ \\text{MSE}*{\\text{regularized}} = \\frac{1}{n} \\sum*{i=1}^n (y\_i - \\hat{y}*i)^2 + \\lambda \\sum*{j=1}^p \\beta\_j^2 \]
16. Eigenvectors: \[ Av = \\lambda v \]
17. Entropy: \[ \\text{Entropy} = -\\sum\_i p\_i \\log\_2(p\_i) \]
18. K-Means: \[ \\arg\\min\_S \\sum\_{i=1}^K \\sum\_{x \\in S\_i} |x - \\mu\_i|^2 \]
19. KL Divergence: \[ D\_{KL}(P|Q) = \\sum\_{x \\in X} P(x) \\log\\left(\\frac{P(x)}{Q(x)}\\right) \]
20. Log-loss: \[ -\\frac{1}{N} \\sum\_{i=1}^N (y\_i \\log(\\hat{y}\_i) + (1 - y\_i) \\log(1 - \\hat{y}\_i)) \]
21. Support Vector Machine: \[ \\min\_{w,b} \\frac{1}{2}|w|^2 + C \\sum\_{i=1}^n \\max(0, 1 - y\_i(w \\cdot x\_i - b)) \]
22. Linear Regression: \[ y = \\beta\_0 + \\beta\_1x\_1 + \\beta\_2x\_2 + ... + \\beta\_nx\_n + \\epsilon \]
23. Singular Value Decomposition: \[ A = U\\Sigma V^T \]
24. Lagrange Multiplier: \[ \\max f(x) \\text{ subject to } g(x) = 0 \] \[ L(x,\\lambda) = f(x) - \\lambda \\cdot g(x) \]
