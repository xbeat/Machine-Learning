## Normalizing vs. Scaling in Python
Slide 1: Normalizing vs. Scaling: What's the Difference?

Normalizing and scaling are two crucial data preprocessing techniques in machine learning and data analysis. While often used interchangeably, they serve distinct purposes and have different mathematical foundations. This presentation will explore their differences, implementations, and practical applications.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create sample data
data = np.random.randn(1000)

# Plot histogram of original data
plt.hist(data, bins=30, alpha=0.5, label='Original')
plt.title('Distribution of Original Data')
plt.legend()
plt.show()
```

Slide 2: Normalization: Bringing Data to a Common Scale

Normalization typically refers to scaling features to a fixed range, often between 0 and 1. This process is crucial when features have different scales and you want to treat them equally in certain algorithms.

```python
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

normalized_data = normalize(data)

plt.hist(normalized_data, bins=30, alpha=0.5, label='Normalized')
plt.title('Distribution of Normalized Data')
plt.legend()
plt.show()
```

Slide 3: Scaling: Adjusting the Range of Values

Scaling, in a broader sense, refers to any transformation that changes the range of a dataset. This can include normalization, but also encompasses other techniques like standardization or robust scaling.

```python
def standardize(x):
    return (x - np.mean(x)) / np.std(x)

scaled_data = standardize(data)

plt.hist(scaled_data, bins=30, alpha=0.5, label='Scaled (Standardized)')
plt.title('Distribution of Scaled Data')
plt.legend()
plt.show()
```

Slide 4: Key Differences: Normalization vs. Scaling

Normalization guarantees all features will have the exact same scale, while general scaling doesn't. Normalization preserves zero values in sparse data, whereas standardization doesn't. Standardization may be preferable when you want to maintain the general distribution shape of your data.

```python
# Compare distributions
plt.hist(normalized_data, bins=30, alpha=0.5, label='Normalized')
plt.hist(scaled_data, bins=30, alpha=0.5, label='Scaled')
plt.title('Normalized vs Scaled Data Distribution')
plt.legend()
plt.show()
```

Slide 5: When to Use Normalization

Normalization is particularly useful when you want to preserve zero values in sparse data or when you know the distribution of your data doesn't follow a Gaussian distribution. It's often used in neural networks and image processing.

```python
from sklearn.preprocessing import MinMaxScaler

# Create sparse data
sparse_data = np.random.choice([0, 1, 2, 3], size=1000, p=[0.7, 0.1, 0.1, 0.1])

# Normalize
scaler = MinMaxScaler()
normalized_sparse = scaler.fit_transform(sparse_data.reshape(-1, 1))

plt.scatter(range(100), sparse_data[:100], alpha=0.5, label='Original')
plt.scatter(range(100), normalized_sparse[:100], alpha=0.5, label='Normalized')
plt.title('Sparse Data: Original vs Normalized')
plt.legend()
plt.show()
```

Slide 6: When to Use Scaling (Standardization)

Standardization is preferred when you want to maintain the shape of the original distribution and when your machine learning algorithm makes assumptions about the distribution of your data, such as Gaussian with zero mean and unit variance.

```python
from sklearn.preprocessing import StandardScaler

# Generate data with different scales
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(0, 10, 1000)

# Combine data
combined_data = np.column_stack((data1, data2))

# Standardize
scaler = StandardScaler()
standardized_data = scaler.fit_transform(combined_data)

plt.scatter(combined_data[:, 0], combined_data[:, 1], alpha=0.5, label='Original')
plt.scatter(standardized_data[:, 0], standardized_data[:, 1], alpha=0.5, label='Standardized')
plt.title('Original vs Standardized Data')
plt.legend()
plt.show()
```

Slide 7: Implementing Normalization in Python

Let's implement normalization using both NumPy and Scikit-learn. We'll use a dataset representing the heights (in cm) of different plant species.

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Sample dataset: heights of different plant species (in cm)
heights = np.array([10, 25, 50, 80, 120, 200, 300]).reshape(-1, 1)

# NumPy implementation
normalized_np = (heights - heights.min()) / (heights.max() - heights.min())

# Scikit-learn implementation
scaler = MinMaxScaler()
normalized_sklearn = scaler.fit_transform(heights)

print("Original heights:", heights.flatten())
print("Normalized (NumPy):", normalized_np.flatten())
print("Normalized (Scikit-learn):", normalized_sklearn.flatten())
```

Slide 8: Implementing Scaling (Standardization) in Python

Now, let's implement standardization using both NumPy and Scikit-learn. We'll use the same dataset of plant heights.

```python
from sklearn.preprocessing import StandardScaler

# NumPy implementation
standardized_np = (heights - np.mean(heights)) / np.std(heights)

# Scikit-learn implementation
scaler = StandardScaler()
standardized_sklearn = scaler.fit_transform(heights)

print("Original heights:", heights.flatten())
print("Standardized (NumPy):", standardized_np.flatten())
print("Standardized (Scikit-learn):", standardized_sklearn.flatten())
```

Slide 9: Real-Life Example: Image Processing

In image processing, normalization is often used to adjust pixel intensities. Let's normalize an image to improve its contrast.

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load an image (replace with your image path)
img = Image.open('sample_image.jpg')
img_array = np.array(img)

# Normalize the image
normalized_img = (img_array - img_array.min()) / (img_array.max() - img_array.min())

# Display original and normalized images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(img_array)
ax1.set_title('Original Image')
ax2.imshow(normalized_img)
ax2.set_title('Normalized Image')
plt.show()
```

Slide 10: Real-Life Example: Feature Scaling in Machine Learning

In machine learning, feature scaling is crucial when dealing with features of different magnitudes. Let's use a simple dataset to demonstrate how scaling affects the performance of k-Nearest Neighbors (k-NN) algorithm.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate without scaling
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy without scaling:", accuracy_score(y_test, y_pred))

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate with scaling
knn_scaled = KNeighborsClassifier()
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)
print("Accuracy with scaling:", accuracy_score(y_test, y_pred_scaled))
```

Slide 11: Choosing Between Normalization and Scaling

The choice between normalization and scaling depends on your data and the requirements of your machine learning algorithm. Here's a simple decision tree to help you choose:

```python
import networkx as nx
import matplotlib.pyplot as plt

def create_decision_tree():
    G = nx.DiGraph()
    G.add_edge("Start", "Sparse Data?")
    G.add_edge("Sparse Data?", "Normalization", label="Yes")
    G.add_edge("Sparse Data?", "Gaussian Distribution?", label="No")
    G.add_edge("Gaussian Distribution?", "Standardization", label="Yes")
    G.add_edge("Gaussian Distribution?", "Algorithm Specific", label="No")
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title("Decision Tree: Normalization vs Scaling")
    plt.axis('off')
    plt.show()

create_decision_tree()
```

Slide 12: Pitfalls and Considerations

While normalization and scaling are powerful techniques, they come with potential pitfalls. Be aware of these considerations:

1. Outliers can significantly affect both normalization and standardization.
2. Scaling may destroy sparsity in your data.
3. Some algorithms (e.g., decision trees) are invariant to monotonic transformations and don't require scaling.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data with outliers
data = np.random.normal(0, 1, 1000)
data[0] = 100  # Add an outlier

# Plot original, normalized, and standardized data
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.hist(data, bins=30)
plt.title('Original Data')

plt.subplot(132)
plt.hist((data - np.min(data)) / (np.max(data) - np.min(data)), bins=30)
plt.title('Normalized Data')

plt.subplot(133)
plt.hist((data - np.mean(data)) / np.std(data), bins=30)
plt.title('Standardized Data')

plt.tight_layout()
plt.show()
```

Slide 13: Handling Outliers: Robust Scaling

When dealing with outliers, robust scaling techniques like Robust Scaler in Scikit-learn can be useful. These methods use statistics that are robust to outliers.

```python
from sklearn.preprocessing import RobustScaler
import numpy as np
import matplotlib.pyplot as plt

# Generate data with outliers
data = np.random.normal(0, 1, 1000)
data[0] = 100  # Add an outlier

# Apply robust scaling
robust_scaler = RobustScaler()
data_robust = robust_scaler.fit_transform(data.reshape(-1, 1))

# Plot original vs robustly scaled data
plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.hist(data, bins=30)
plt.title('Original Data')

plt.subplot(122)
plt.hist(data_robust, bins=30)
plt.title('Robustly Scaled Data')

plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

For further exploration of normalization and scaling techniques:

1. Scikit-learn Preprocessing Guide: [https://scikit-learn.org/stable/modules/preprocessing.html](https://scikit-learn.org/stable/modules/preprocessing.html)
2. "A Survey on Data Preprocessing Methods in Data Mining" by García et al. (2015): [https://arxiv.org/abs/1511.03980](https://arxiv.org/abs/1511.03980)
3. "Normalization vs Standardization — Quantitative analysis" by Shivam Bansal (2020): [https://arxiv.org/abs/2001.04608](https://arxiv.org/abs/2001.04608)

These resources provide in-depth discussions on various preprocessing techniques, their mathematical foundations, and their applications in different domains of machine learning and data analysis.

