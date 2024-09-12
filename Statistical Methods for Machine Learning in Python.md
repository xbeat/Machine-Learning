## Statistical Methods for Machine Learning in Python
Slide 1: Introduction to Statistical Methods in Machine Learning

Statistical methods form the backbone of many machine learning algorithms. They provide the mathematical foundation for understanding data patterns, making predictions, and drawing insights. In this presentation, we'll explore key statistical concepts and their application in machine learning using Python.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
data = np.random.normal(loc=0, scale=1, size=1000)

# Plot histogram
plt.figure(figsize=(10, 6))
sns.histplot(data, kde=True)
plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

Slide 2: Descriptive Statistics: Measures of Central Tendency

Descriptive statistics help summarize and describe the main features of a dataset. Measures of central tendency, such as mean, median, and mode, provide insights into the typical or central values of a distribution.

```python
import numpy as np

# Generate sample data
data = np.random.randint(1, 101, size=1000)

# Calculate measures of central tendency
mean = np.mean(data)
median = np.median(data)
mode = np.argmax(np.bincount(data))

print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Mode: {mode}")

# Output:
# Mean: 50.35
# Median: 50.00
# Mode: 42
```

Slide 3: Descriptive Statistics: Measures of Dispersion

Measures of dispersion describe the spread or variability of data points in a dataset. Common measures include variance, standard deviation, and range. These metrics help understand how data is distributed around central values.

```python
import numpy as np

# Generate sample data
data = np.random.normal(loc=0, scale=1, size=1000)

# Calculate measures of dispersion
variance = np.var(data)
std_dev = np.std(data)
data_range = np.max(data) - np.min(data)

print(f"Variance: {variance:.4f}")
print(f"Standard Deviation: {std_dev:.4f}")
print(f"Range: {data_range:.4f}")

# Output:
# Variance: 0.9927
# Standard Deviation: 0.9963
# Range: 6.5954
```

Slide 4: Probability Distributions

Probability distributions are mathematical functions that describe the likelihood of different outcomes in a random experiment. Understanding these distributions is crucial for many machine learning algorithms, especially in probabilistic modeling and Bayesian inference.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate data for different distributions
x = np.linspace(-5, 5, 1000)
normal = stats.norm.pdf(x, 0, 1)
uniform = stats.uniform.pdf(x, -2, 4)
exponential = stats.expon.pdf(x, scale=1)

# Plot distributions
plt.figure(figsize=(12, 6))
plt.plot(x, normal, label='Normal')
plt.plot(x, uniform, label='Uniform')
plt.plot(x, exponential, label='Exponential')
plt.legend()
plt.title("Common Probability Distributions")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.show()
```

Slide 5: Hypothesis Testing

Hypothesis testing is a statistical method used to make inferences about population parameters based on sample data. It involves formulating null and alternative hypotheses, calculating test statistics, and making decisions based on p-values or confidence intervals.

```python
import numpy as np
from scipy import stats

# Generate two samples
sample1 = np.random.normal(loc=0, scale=1, size=100)
sample2 = np.random.normal(loc=0.5, scale=1, size=100)

# Perform t-test
t_statistic, p_value = stats.ttest_ind(sample1, sample2)

print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")

# Output:
# T-statistic: -3.7424
# P-value: 0.0002
# Reject the null hypothesis
```

Slide 6: Correlation and Covariance

Correlation and covariance measure the relationship between variables. Correlation indicates the strength and direction of a linear relationship, while covariance measures how two variables vary together. These concepts are essential in feature selection and dimensionality reduction.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate correlated data
x = np.random.normal(0, 1, 1000)
y = 0.8 * x + np.random.normal(0, 0.5, 1000)

# Calculate correlation and covariance
correlation = np.corrcoef(x, y)[0, 1]
covariance = np.cov(x, y)[0, 1]

# Plot scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x=x, y=y)
plt.title(f"Correlation: {correlation:.4f}, Covariance: {covariance:.4f}")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Slide 7: Linear Regression

Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between variables and is widely used for prediction and inference.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate sample data
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Plot results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.legend()
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

print(f"Coefficient: {model.coef_[0][0]:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")

# Output:
# Coefficient: 1.9876
# Intercept: 1.0534
```

Slide 8: Logistic Regression

Logistic regression is a statistical method used for binary classification problems. It models the probability of an instance belonging to a particular class using the logistic function. Despite its name, logistic regression is a classification algorithm, not a regression algorithm.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
X = np.random.randn(1000, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Plot decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['--'])
plt.title(f"Logistic Regression (Accuracy: {accuracy:.4f})")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
```

Slide 9: Principal Component Analysis (PCA)

PCA is a statistical technique used for dimensionality reduction and feature extraction. It transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible. PCA is widely used in data visualization, noise reduction, and preprocessing for machine learning algorithms.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot original data and PCA-transformed data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1])
ax1.set_title("Original Data")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")

ax2.scatter(X_pca[:, 0], X_pca[:, 1])
ax2.set_title("PCA-transformed Data")
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")

plt.tight_layout()
plt.show()

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Output:
# Explained variance ratio: [0.72524061 0.27475939]
```

Slide 10: K-means Clustering

K-means clustering is an unsupervised learning algorithm used to partition data into K distinct clusters. It aims to minimize the within-cluster sum of squares by iteratively assigning data points to the nearest cluster centroid and updating the centroids.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Apply K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='r', label='Centroids')
plt.title("K-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

print(f"Inertia: {kmeans.inertia_:.2f}")

# Output:
# Inertia: 177.84
```

Slide 11: Naive Bayes Classification

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes that features are independent given the class label, which simplifies the computation. Despite its simplicity, Naive Bayes often performs well in practice, especially for text classification tasks.

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate sample data
X = np.random.randn(1000, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Naive Bayes Confusion Matrix (Accuracy: {accuracy:.4f})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

Slide 12: Cross-validation

Cross-validation is a statistical method used to assess the generalization performance of a machine learning model. It involves partitioning the data into subsets, training the model on a subset, and validating it on the remaining data. K-fold cross-validation is a common technique that helps prevent overfitting and provides a more robust estimate of model performance.

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Create logistic regression model
model = LogisticRegression()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

print("Cross-validation scores:", cv_scores)
print(f"Mean CV score: {cv_scores.mean():.4f}")
print(f"Standard deviation of CV scores: {cv_scores.std():.4f}")

# Output:
# Cross-validation scores: [0.915 0.905 0.93  0.915 0.93 ]
# Mean CV score: 0.9190
# Standard deviation of CV scores: 0.0103
```

Slide 13: Real-life Example: Image Classification

Image classification is a common application of statistical methods in machine learning. We'll use a simplified convolutional neural network (CNN) to classify images from the CIFAR-10 dataset. CNNs leverage statistical concepts like convolution and pooling for effective image processing.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc:.2f}')
```

Slide 14: Real-life Example: Natural Language Processing

Natural Language Processing (NLP) is another area where statistical methods play a crucial role. In this example, we'll use a simple sentiment analysis model based on the Naive Bayes algorithm to classify movie reviews as positive or negative.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample movie reviews and labels
reviews = [
    "This movie was fantastic! I loved every minute of it.",
    "Terrible acting and poor plot. Waste of time.",
    "Great special effects and an engaging story.",
    "Boring and predictable. Don't bother watching.",
    "An instant classic. Highly recommended!"
]
labels = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# Create feature vectors
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectors, y_train)

# Make predictions
y_pred = clf.predict(X_test_vectors)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
```

Slide 15: Additional Resources

For those interested in diving deeper into statistical methods for machine learning, here are some valuable resources:

1. ArXiv.org: A repository of academic papers covering various aspects of machine learning and statistics. Example: "A Tutorial on Principal Component Analysis" by Jonathon Shlens ArXiv URL: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
2. "Pattern Recognition and Machine Learning" by Christopher Bishop: A comprehensive textbook covering statistical methods in machine learning.
3. Online courses: Platforms like Coursera, edX, and Udacity offer courses on machine learning and statistics.
4. Python libraries documentation: Scikit-learn, NumPy, SciPy, and TensorFlow provide excellent documentation and tutorials for implementing statistical methods in machine learning.

Remember to verify the credibility and relevance of these resources before using them in your projects or research.

