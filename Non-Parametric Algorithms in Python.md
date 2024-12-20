## Non-Parametric Algorithms in Python
Slide 1: Introduction to Non-Parametric Algorithms

Non-parametric algorithms are statistical methods that don't make assumptions about the underlying data distribution. They are flexible and can handle various types of data, making them useful in many real-world scenarios.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
data = np.random.normal(loc=0, scale=1, size=1000)

# Plot histogram
plt.hist(data, bins=30, edgecolor='black')
plt.title('Histogram of Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 2: K-Nearest Neighbors (KNN)

KNN is a simple non-parametric algorithm used for classification and regression. It predicts based on the majority class or average of the k nearest neighbors in the feature space.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

print(f"Accuracy: {knn.score(X_test, y_test):.2f}")
```

Slide 3: Decision Trees

Decision trees are non-parametric algorithms that learn simple decision rules inferred from the data features. They can be used for both classification and regression tasks.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train decision tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

print(f"Accuracy: {dt.score(X_test, y_test):.2f}")
```

Slide 4: Kernel Density Estimation (KDE)

KDE is a non-parametric way to estimate the probability density function of a random variable. It's useful for visualizing the distribution of data and for density-based clustering.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate sample data
np.random.seed(42)
data = np.concatenate([np.random.normal(0, 1, 1000), np.random.normal(4, 1.5, 500)])

# Compute KDE
kde = gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 1000)
y_kde = kde(x_range)

# Plot results
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.5, label='Histogram')
plt.plot(x_range, y_kde, label='KDE')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.title('Kernel Density Estimation')
plt.show()
```

Slide 5: Support Vector Machines (SVM)

SVMs are versatile algorithms used for classification, regression, and outlier detection. They can handle non-linear decision boundaries using the kernel trick.

```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train SVM model
svm = SVC(kernel='rbf', C=1.0, random_state=42)
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)

print(f"Accuracy: {svm.score(X_test, y_test):.2f}")
```

Slide 6: Random Forests

Random Forests are an ensemble learning method that constructs multiple decision trees and merges their predictions. They are robust and can handle high-dimensional data.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load breast cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

print(f"Accuracy: {rf.score(X_test, y_test):.2f}")
```

Slide 7: Naive Bayes Classifier

Naive Bayes is a probabilistic classifier based on Bayes' theorem. It's particularly useful for text classification and spam filtering tasks.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Naive Bayes model
nb = GaussianNB()
nb.fit(X_train, y_train)

# Make predictions
y_pred = nb.predict(X_test)

print(f"Accuracy: {nb.score(X_test, y_test):.2f}")
```

Slide 8: K-Means Clustering

K-Means is a popular unsupervised learning algorithm used for clustering. It partitions n observations into k clusters, where each observation belongs to the cluster with the nearest mean.

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.concatenate([np.random.normal(0, 1, (100, 2)), np.random.normal(5, 1, (100, 2))])

# Create and fit K-Means model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title('K-Means Clustering')
plt.show()
```

Slide 9: Real-Life Example: Image Classification

Non-parametric algorithms like KNN can be used for image classification tasks. Here's a simple example using the MNIST dataset of handwritten digits.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load MNIST dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

print(f"Accuracy: {knn.score(X_test, y_test):.2f}")

# Display a sample prediction
sample_idx = 0
plt.imshow(X_test[sample_idx].reshape(8, 8), cmap='gray')
plt.title(f"Predicted: {y_pred[sample_idx]}, Actual: {y_test[sample_idx]}")
plt.show()
```

Slide 10: Real-Life Example: Customer Segmentation

K-Means clustering can be used for customer segmentation in marketing. This example demonstrates clustering customers based on their annual income and spending score.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate sample customer data
np.random.seed(42)
n_customers = 200
annual_income = np.random.normal(50000, 15000, n_customers)
spending_score = np.random.normal(50, 25, n_customers)
X = np.column_stack((annual_income, spending_score))

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Visualize results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(annual_income, spending_score, c=clusters, cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.colorbar(scatter)
plt.show()

print("Cluster Centers:")
print(scaler.inverse_transform(kmeans.cluster_centers_))
```

Slide 11: Advantages of Non-Parametric Algorithms

Non-parametric algorithms offer several benefits in data analysis and machine learning. They are flexible and can adapt to various data distributions without making strong assumptions about the underlying structure.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Generate data from different distributions
np.random.seed(42)
n_samples = 1000
gaussian = np.random.normal(0, 1, n_samples)
bimodal = np.concatenate([np.random.normal(-2, 0.5, n_samples//2), np.random.normal(2, 0.5, n_samples//2)])

# Estimate density using KDE
kde_gaussian = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(gaussian.reshape(-1, 1))
kde_bimodal = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(bimodal.reshape(-1, 1))

# Plot results
x = np.linspace(-5, 5, 1000).reshape(-1, 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(gaussian, bins=30, density=True, alpha=0.5)
ax1.plot(x, np.exp(kde_gaussian.score_samples(x)))
ax1.set_title('Gaussian Distribution')

ax2.hist(bimodal, bins=30, density=True, alpha=0.5)
ax2.plot(x, np.exp(kde_bimodal.score_samples(x)))
ax2.set_title('Bimodal Distribution')

plt.tight_layout()
plt.show()
```

Slide 12: Limitations of Non-Parametric Algorithms

While non-parametric algorithms are versatile, they have some limitations. They often require larger datasets to achieve good performance and can be computationally expensive for large-scale problems.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import learning_curve

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 1000).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Create KNN regressor
knn = KNeighborsRegressor(n_neighbors=5)

# Calculate learning curve
train_sizes, train_scores, test_scores = learning_curve(
    knn, X, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=5)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curve for KNN Regressor')
plt.legend(loc='best')
plt.show()
```

Slide 13: Choosing the Right Non-Parametric Algorithm

Selecting the appropriate non-parametric algorithm depends on the specific problem, dataset characteristics, and computational resources. This decision tree can guide you in choosing the right algorithm for your task.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a decision tree graph
G = nx.DiGraph()
G.add_edge("Start", "Classification?")
G.add_edge("Classification?", "KNN", label="Yes")
G.add_edge("Classification?", "Regression?", label="No")
G.add_edge("Regression?", "SVR", label="Yes")
G.add_edge("Regression?", "Clustering?", label="No")
G.add_edge("Clustering?", "K-Means", label="Yes")
G.add_edge("Clustering?", "Density Estimation?", label="No")
G.add_edge("Density Estimation?", "KDE", label="Yes")
G.add_edge("Density Estimation?", "Other Methods", label="No")

# Set up the plot
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.9, iterations=50)

# Draw the graph
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=3000, font_size=10, font_weight='bold')

# Add edge labels
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Decision Tree for Choosing Non-Parametric Algorithms")
plt.axis('off')
plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

For further exploration of non-parametric algorithms, consider these resources:

1. "Nonparametric Statistics: A Step-by-Step Approach" by Gregory W. Corder and Dale I. Foreman
2. "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman
3. ArXiv paper: "A Survey of Deep Learning Techniques for Neural Machine Translation" ([https://arxiv.org/abs/1703.01619](https://arxiv.org/abs/1703.01619))
4. Scikit-learn documentation: [https://scikit-learn.org/stable/modules/neighbors.html](https://scikit-learn.org/stable/modules/neighbors.html)

These resources provide in-depth coverage of various non-parametric techniques and their applications in machine learning and statistics.

