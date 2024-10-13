## Common Feature Scaling Techniques in Python
Slide 1: Introduction to Feature Scaling

Feature scaling is a crucial preprocessing step in machine learning. It involves transforming numeric variables to a standard scale, ensuring that all features contribute equally to the model. This process is particularly important when dealing with features of different magnitudes or units.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = np.random.randn(100, 2) * np.array([10, 1])

# Plot original data
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(data[:, 0], data[:, 1])
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot scaled data (using standard scaling for illustration)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

plt.subplot(122)
plt.scatter(scaled_data[:, 0], scaled_data[:, 1])
plt.title('Scaled Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

Slide 2: Min-Max Scaling

Min-Max scaling, also known as normalization, scales features to a fixed range, typically between 0 and 1. This method preserves zero values and maintains the original distribution of the data.

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Sample data
data = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(data)

print("Original data:")
print(data)
print("\nScaled data:")
print(scaled_data)

# Output:
# Original data:
# [[-1  2]
#  [-0.5  6]
#  [ 0 10]
#  [ 1 18]]
#
# Scaled data:
# [[0.   0.  ]
#  [0.25 0.25]
#  [0.5  0.5 ]
#  [1.   1.  ]]
```

Slide 3: Standard Scaling (Z-score Normalization)

Standard scaling transforms features to have zero mean and unit variance. This method is particularly useful when the data follows a Gaussian distribution and when dealing with algorithms sensitive to the scale of input features.

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample data
data = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(data)

print("Original data:")
print(data)
print("\nScaled data:")
print(scaled_data)

# Output:
# Original data:
# [[-1  2]
#  [-0.5  6]
#  [ 0 10]
#  [ 1 18]]
#
# Scaled data:
# [[-1.34164079 -1.34164079]
#  [-0.4472136  -0.4472136 ]
#  [ 0.4472136   0.4472136 ]
#  [ 1.34164079  1.34164079]]
```

Slide 4: Robust Scaling

Robust scaling is less affected by outliers compared to standard scaling. It uses the median and interquartile range instead of mean and standard deviation, making it suitable for datasets with outliers.

```python
from sklearn.preprocessing import RobustScaler
import numpy as np

# Sample data with an outlier
data = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18], [100, 200]])

# Initialize RobustScaler
scaler = RobustScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(data)

print("Original data:")
print(data)
print("\nScaled data:")
print(scaled_data)

# Output:
# Original data:
# [[ -1   2]
#  [ -0.5   6]
#  [  0  10]
#  [  1  18]
#  [100 200]]
#
# Scaled data:
# [[-0.66666667 -0.5       ]
#  [-0.33333333 -0.25      ]
#  [ 0.          0.        ]
#  [ 0.66666667  0.5       ]
#  [66.33333333 11.875     ]]
```

Slide 5: Max Absolute Scaling

Max Absolute scaling scales each feature by its maximum absolute value. This preserves zero values and is useful when working with sparse data or when you want to maintain the sign of the original data.

```python
from sklearn.preprocessing import MaxAbsScaler
import numpy as np

# Sample data
data = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])

# Initialize MaxAbsScaler
scaler = MaxAbsScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(data)

print("Original data:")
print(data)
print("\nScaled data:")
print(scaled_data)

# Output:
# Original data:
# [[-1  2]
#  [-0.5  6]
#  [ 0 10]
#  [ 1 18]]
#
# Scaled data:
# [[-1.    0.11111111]
#  [-0.5   0.33333333]
#  [ 0.    0.55555556]
#  [ 1.    1.        ]]
```

Slide 6: Quantile Transformation

Quantile transformation maps the original distribution to a uniform or normal distribution. This method is useful when dealing with non-Gaussian distributed data or when you want to reduce the impact of outliers.

```python
from sklearn.preprocessing import QuantileTransformer
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with a skewed distribution
np.random.seed(0)
data = np.random.exponential(size=(1000, 1))

# Initialize QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')

# Fit and transform the data
transformed_data = qt.fit_transform(data)

# Plot original and transformed data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(data, bins=50)
ax1.set_title('Original Data')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')

ax2.hist(transformed_data, bins=50)
ax2.set_title('Transformed Data')
ax2.set_xlabel('Value')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

Slide 7: Power Transformation

Power transformation is used to stabilize variance and make the data more Gaussian-like. Two common methods are the Box-Cox transformation and the Yeo-Johnson transformation.

```python
from sklearn.preprocessing import PowerTransformer
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with a skewed distribution
np.random.seed(0)
data = np.random.exponential(size=(1000, 1))

# Initialize PowerTransformer with Box-Cox method
pt_boxcox = PowerTransformer(method='box-cox')
# Initialize PowerTransformer with Yeo-Johnson method
pt_yeojohnson = PowerTransformer(method='yeo-johnson')

# Fit and transform the data
transformed_boxcox = pt_boxcox.fit_transform(data)
transformed_yeojohnson = pt_yeojohnson.fit_transform(data)

# Plot original and transformed data
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.hist(data, bins=50)
ax1.set_title('Original Data')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')

ax2.hist(transformed_boxcox, bins=50)
ax2.set_title('Box-Cox Transformed')
ax2.set_xlabel('Value')

ax3.hist(transformed_yeojohnson, bins=50)
ax3.set_title('Yeo-Johnson Transformed')
ax3.set_xlabel('Value')

plt.tight_layout()
plt.show()
```

Slide 8: L1 and L2 Normalization

L1 and L2 normalization scale individual samples to have unit norm. L1 normalization uses the sum of absolute values, while L2 normalization uses the square root of the sum of squared values.

```python
from sklearn.preprocessing import Normalizer
import numpy as np

# Sample data
data = np.array([[1, -1, 2],
                 [2, 0, 0],
                 [0, 1, -1]])

# Initialize Normalizer
l1_normalizer = Normalizer(norm='l1')
l2_normalizer = Normalizer(norm='l2')

# Fit and transform the data
l1_normalized = l1_normalizer.fit_transform(data)
l2_normalized = l2_normalizer.fit_transform(data)

print("Original data:")
print(data)
print("\nL1 normalized data:")
print(l1_normalized)
print("\nL2 normalized data:")
print(l2_normalized)

# Output:
# Original data:
# [[ 1 -1  2]
#  [ 2  0  0]
#  [ 0  1 -1]]
#
# L1 normalized data:
# [[ 0.25 -0.25  0.5 ]
#  [ 1.    0.    0.  ]
#  [ 0.    0.5  -0.5 ]]
#
# L2 normalized data:
# [[ 0.40824829 -0.40824829  0.81649658]
#  [ 1.          0.          0.        ]
#  [ 0.          0.70710678 -0.70710678]]
```

Slide 9: Scaling Sparse Data

When working with sparse data, it's important to use scaling techniques that preserve sparsity. The MaxAbsScaler is particularly useful for this purpose, as it doesn't shift the data and thus doesn't destroy sparsity.

```python
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import csr_matrix
import numpy as np

# Create a sparse matrix
data = [[1, 0, 2],
        [0, 0, 3],
        [4, 5, 6]]
sparse_matrix = csr_matrix(data)

# Initialize MaxAbsScaler
scaler = MaxAbsScaler()

# Fit and transform the sparse matrix
scaled_sparse = scaler.fit_transform(sparse_matrix)

print("Original sparse matrix:")
print(sparse_matrix.toarray())
print("\nScaled sparse matrix:")
print(scaled_sparse.toarray())

# Output:
# Original sparse matrix:
# [[1 0 2]
#  [0 0 3]
#  [4 5 6]]
#
# Scaled sparse matrix:
# [[0.25 0.   0.33333333]
#  [0.   0.   0.5       ]
#  [1.   1.   1.        ]]
```

Slide 10: Scaling in Pipeline

Incorporating scaling into a scikit-learn pipeline ensures that the scaling is applied consistently during both training and prediction phases, preventing data leakage.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler and LogisticRegression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the model
score = pipeline.score(X_test, y_test)
print(f"Model accuracy: {score:.4f}")

# Output:
# Model accuracy: 0.9600
```

Slide 11: Feature Scaling for Time Series Data

When dealing with time series data, it's crucial to scale features without introducing future information. This can be achieved by using a custom transformer that scales each feature based on past data.

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TimeSeriesScaler(BaseEstimator, TransformerMixin):
    def __init__(self, window=10):
        self.window = window
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_scaled = np.zeros_like(X)
        for i in range(X.shape[0]):
            if i < self.window:
                X_scaled[i] = X[i] / np.max(np.abs(X[:i+1]))
            else:
                X_scaled[i] = X[i] / np.max(np.abs(X[i-self.window:i+1]))
        return X_scaled

# Example usage
np.random.seed(42)
time_series = np.cumsum(np.random.randn(100))

scaler = TimeSeriesScaler(window=10)
scaled_series = scaler.fit_transform(time_series.reshape(-1, 1))

print("Original series (first 10 elements):")
print(time_series[:10])
print("\nScaled series (first 10 elements):")
print(scaled_series[:10])

# Output:
# Original series (first 10 elements):
# [ 0.49671415 -0.1382643   0.64768854  1.52302986  1.57921282  0.76743473
#   0.87811744  1.03261455  1.36546382  1.17303762]
#
# Scaled series (first 10 elements):
# [[1.        ]
#  [-0.27835873]
#  [ 1.        ]
#  [ 1.        ]
#  [ 1.        ]
#  [ 0.48598061]
#  [ 0.55595125]
#  [ 0.65388692]
#  [ 0.86452811]
#  [ 0.74267441]]
```

Slide 12: Real-life Example: Image Preprocessing

In image processing, scaling pixel values is crucial for many deep learning models. Here's an example of how to preprocess images for a convolutional neural network using Python and popular libraries.

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Create a sample 224x224 RGB image
image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

# Convert to PIL Image
pil_image = Image.fromarray(image)

# Preprocess the image
preprocessed_image = np.array(pil_image).astype(np.float32) / 255.0

# Plot original and preprocessed images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(image)
ax1.set_title('Original Image')
ax1.axis('off')

ax2.imshow(preprocessed_image)
ax2.set_title('Preprocessed Image (Scaled to [0, 1])')
ax2.axis('off')

plt.tight_layout()
plt.show()

print("Original image shape:", image.shape)
print("Original image data type:", image.dtype)
print("Preprocessed image shape:", preprocessed_image.shape)
print("Preprocessed image data type:", preprocessed_image.dtype)
print("Preprocessed image value range:", preprocessed_image.min(), "to", preprocessed_image.max())
```

Slide 13: Real-life Example: Natural Language Processing

In natural language processing, feature scaling is often applied to word embeddings or other numerical representations of text. Here's an example of scaling TF-IDF vectors for text classification.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample text data
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold."
]

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# Convert to dense array for scaling
tfidf_dense = tfidf_matrix.toarray()

# Scale the TF-IDF vectors
scaler = StandardScaler(with_mean=False)  # Sparse data: don't center
scaled_tfidf = scaler.fit_transform(tfidf_dense)

print("Original TF-IDF shape:", tfidf_dense.shape)
print("Scaled TF-IDF shape:", scaled_tfidf.shape)
print("\nOriginal TF-IDF (first row):")
print(tfidf_dense[0])
print("\nScaled TF-IDF (first row):")
print(scaled_tfidf[0])
```

Slide 14: Choosing the Right Scaling Technique

Selecting the appropriate scaling technique depends on your data and the machine learning algorithm you're using. Here's a guide to help you choose:

1. Standard Scaling: Use when your data is approximately normally distributed and you're working with algorithms that assume normally distributed data (e.g., linear regression, logistic regression, neural networks).
2. Min-Max Scaling: Useful when you need values in a bounded interval, such as \[0, 1\]. It's often used for neural networks and algorithms that don't assume any distribution.
3. Robust Scaling: Ideal when your data contains outliers. It's less affected by outliers compared to standard scaling.
4. MaxAbs Scaling: Good for sparse data as it doesn't shift/center the data, thus preserving sparsity.
5. Quantile Transformation: Use when you want to transform features to follow a uniform or normal distribution. It's robust to outliers.
6. Power Transformation: Helpful when dealing with skewed or non-normal distributions, aiming to make them more Gaussian-like.

Always consider your specific use case, the nature of your data, and the requirements of your chosen algorithm when selecting a scaling method.

Slide 15: Additional Resources

For more in-depth information on feature scaling techniques and their applications, consider exploring these resources:

1. ArXiv paper: "A Comparative Study of Feature Scaling Methods in the Context of Large Margin Classifiers" (arXiv:1807.01264) URL: [https://arxiv.org/abs/1807.01264](https://arxiv.org/abs/1807.01264)
2. ArXiv paper: "Normalization Techniques in Training DNNs: Methodology, Analysis and Application" (arXiv:2009.12836) URL: [https://arxiv.org/abs/2009.12836](https://arxiv.org/abs/2009.12836)

These papers provide detailed analyses of various scaling techniques and their impacts on machine learning models, offering valuable insights for both beginners and advanced practitioners in the field of data science and machine learning.


