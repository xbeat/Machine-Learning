## Supervised vs Unsupervised Learning in Python
Slide 1: Introduction to Supervised and Unsupervised Learning

Supervised and unsupervised learning are two fundamental paradigms in machine learning. Supervised learning uses labeled data to train models, while unsupervised learning discovers patterns in unlabeled data. This slideshow explores their differences, applications, and implementations using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Supervised Learning Example
X_supervised = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_supervised = np.array([1, 2, 3, 4])

# Unsupervised Learning Example
X_unsupervised = np.array([[1, 2], [2, 3], [8, 7], [9, 8]])

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_supervised[:, 0], X_supervised[:, 1], c=y_supervised)
plt.title("Supervised Learning")
plt.subplot(122)
plt.scatter(X_unsupervised[:, 0], X_unsupervised[:, 1])
plt.title("Unsupervised Learning")
plt.show()
```

Slide 2: Supervised Learning - Definition and Characteristics

Supervised learning involves training a model on a labeled dataset, where each input is associated with a corresponding output. The goal is to learn a mapping function that can predict the output for new, unseen inputs. This approach requires a well-defined target variable and is suitable for tasks like classification and regression.

```python
from sklearn.linear_model import LinearRegression

# Training data
X_train = np.array([[1], [2], [3], [4]])
y_train = np.array([2, 4, 6, 8])

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([[5], [6]])
y_pred = model.predict(X_test)

print(f"Predictions: {y_pred}")
```

Slide 3: Unsupervised Learning - Definition and Characteristics

Unsupervised learning deals with unlabeled data, aiming to discover hidden patterns or structures within the dataset. Unlike supervised learning, there's no predefined target variable. Common tasks include clustering, dimensionality reduction, and anomaly detection. This approach is particularly useful when exploring large datasets without prior knowledge of the underlying structure.

```python
from sklearn.cluster import KMeans
import numpy as np

# Generate random data
np.random.seed(42)
X = np.random.rand(100, 2)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.title("K-means Clustering (Unsupervised Learning)")
plt.show()
```

Slide 4: Key Differences - Data Requirements

Supervised learning requires labeled data, where each input is paired with its corresponding output. This labeling process can be time-consuming and expensive. Unsupervised learning, on the other hand, works with unlabeled data, which is often more abundant and easier to collect.

```python
import pandas as pd

# Supervised learning dataset
supervised_data = pd.DataFrame({
    'feature1': [1, 2, 3, 4],
    'feature2': [2, 3, 4, 5],
    'label': ['A', 'B', 'A', 'B']
})

# Unsupervised learning dataset
unsupervised_data = pd.DataFrame({
    'feature1': [1, 2, 3, 4],
    'feature2': [2, 3, 4, 5]
})

print("Supervised Learning Dataset:")
print(supervised_data)
print("\nUnsupervised Learning Dataset:")
print(unsupervised_data)
```

Slide 5: Key Differences - Learning Process

In supervised learning, the model learns by comparing its predictions to the actual labels, adjusting its parameters to minimize the error. Unsupervised learning algorithms, however, seek to identify inherent structures in the data without external guidance.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# Supervised learning (Logistic Regression)
X_supervised = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_supervised = np.array([0, 0, 1, 1])

supervised_model = LogisticRegression()
supervised_model.fit(X_supervised, y_supervised)

# Unsupervised learning (K-means)
X_unsupervised = np.array([[1, 2], [2, 3], [8, 7], [9, 8]])

unsupervised_model = KMeans(n_clusters=2)
unsupervised_model.fit(X_unsupervised)

print("Supervised Model Coefficients:", supervised_model.coef_)
print("Unsupervised Model Cluster Centers:", unsupervised_model.cluster_centers_)
```

Slide 6: Key Differences - Evaluation Metrics

Supervised learning models are typically evaluated using metrics like accuracy, precision, recall, or mean squared error, which compare predictions to known labels. Unsupervised learning evaluation is more challenging and often relies on intrinsic metrics like silhouette score or inertia.

```python
from sklearn.metrics import accuracy_score, silhouette_score

# Supervised learning evaluation
y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]
accuracy = accuracy_score(y_true, y_pred)

# Unsupervised learning evaluation
X = np.array([[1, 2], [2, 3], [8, 7], [9, 8]])
kmeans = KMeans(n_clusters=2).fit(X)
silhouette = silhouette_score(X, kmeans.labels_)

print(f"Supervised Learning Accuracy: {accuracy}")
print(f"Unsupervised Learning Silhouette Score: {silhouette}")
```

Slide 7: Supervised Learning - Classification Example

Classification is a common supervised learning task where the goal is to predict discrete class labels. Here's an example using logistic regression to classify iris flowers based on their sepal and petal measurements.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Classification Accuracy: {accuracy:.2f}")
```

Slide 8: Supervised Learning - Regression Example

Regression is another supervised learning task where the goal is to predict continuous numerical values. This example demonstrates linear regression to predict house prices based on features like square footage and number of bedrooms.

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=2, noise=0.1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
```

Slide 9: Unsupervised Learning - Clustering Example

Clustering is a fundamental unsupervised learning task that groups similar data points together. This example uses K-means clustering to identify groups in a dataset of customer behavior.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate synthetic customer data
np.random.seed(42)
X = np.random.rand(100, 2)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title("Customer Segmentation using K-means Clustering")
plt.show()
```

Slide 10: Unsupervised Learning - Dimensionality Reduction Example

Dimensionality reduction is another important unsupervised learning task, often used for data visualization and preprocessing. This example uses Principal Component Analysis (PCA) to reduce the dimensionality of the iris dataset.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X = iris.data

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize the results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis')
plt.title("PCA of Iris Dataset")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

Slide 11: Real-Life Example - Supervised Learning in Image Classification

Image classification is a popular application of supervised learning. Convolutional Neural Networks (CNNs) are often used for this task. Here's a simple example using the MNIST dataset of handwritten digits.

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load and preprocess the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((60000, 28, 28, 1)) / 255.0
X_test = X_test.reshape((10000, 28, 28, 1)) / 255.0

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

Slide 12: Real-Life Example - Unsupervised Learning in Anomaly Detection

Anomaly detection is a crucial application of unsupervised learning, often used in cybersecurity, fraud detection, and quality control. This example demonstrates using Isolation Forest for detecting anomalies in a dataset of network traffic.

```python
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic network traffic data
np.random.seed(42)
X = np.random.randn(1000, 2)
X[:50] += [5, 5]  # Add some anomalies

# Train the Isolation Forest model
clf = IsolationForest(contamination=0.1, random_state=42)
y_pred = clf.fit_predict(X)

# Visualize the results
plt.scatter(X[:, 0], X[y_pred == 1, 1], c='blue', label='Normal')
plt.scatter(X[:, 0][y_pred == -1], X[:, 1][y_pred == -1], c='red', label='Anomaly')
plt.title("Network Traffic Anomaly Detection")
plt.legend()
plt.show()

print(f"Number of detected anomalies: {sum(y_pred == -1)}")
```

Slide 13: Hybrid Approaches - Semi-Supervised Learning

Semi-supervised learning combines aspects of both supervised and unsupervised learning, using a small amount of labeled data along with a large amount of unlabeled data. This approach can be particularly useful when labeling data is expensive or time-consuming.

```python
from sklearn.semi_supervised import LabelSpreading
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(200, 2)
y = np.zeros(200)
y[:100] = 1

# Mask some labels as unlabeled (-1)
mask = np.random.randint(0, 2, 200).astype(bool)
y_partial = np.(y)
y_partial[mask] = -1

# Train the model
model = LabelSpreading()
model.fit(X, y_partial)

# Evaluate accuracy on the originally labeled points
accuracy = model.score(X[~mask], y[~mask])
print(f"Accuracy on labeled points: {accuracy:.2f}")

# Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=model.predict(X), cmap='viridis')
plt.title("Semi-Supervised Learning Results")
plt.show()
```

Slide 14: Choosing Between Supervised and Unsupervised Learning

The choice between supervised and unsupervised learning depends on various factors, including the availability of labeled data, the specific problem at hand, and the desired outcomes. Supervised learning is ideal when you have a clear target variable and sufficient labeled data. Unsupervised learning is useful for exploring data structure, discovering patterns, and preprocessing. In many real-world applications, a combination of both approaches may yield the best results.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
X = np.random.randn(300, 2)
X[:100] += [2, 2]
X[100:200] += [-2, -2]
y = np.zeros(300)
y[:100] = 1
y[100:200] = 2

# Visualize data
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1])
plt.title("Raw Data")

plt.subplot(132)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title("Supervised View")

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X)
plt.subplot(133)
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.title("Unsupervised View")

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in deepening their understanding of supervised and unsupervised learning, here are some valuable resources:

1. "A Survey of Deep Learning Techniques for Neural Machine Translation" by Shuohang Wang and Jing Jiang (2017). Available at: [https://arxiv.org/abs/1703.03906](https://arxiv.org/abs/1703.03906)
2. "A Survey of Clustering With Deep Learning: From the Perspective of Network Architecture" by Xifeng Guo et al. (2018). Available at: [https://arxiv.org/abs/1801.07648](https://arxiv.org/abs/1801.07648)
3. "Deep Learning" by Yann LeCun, Yoshua Bengio, and Geoffrey Hinton (2015). Nature

