## Data Preprocessing Techniques in Machine Learning
Slide 1: StandardScaler Implementation and Applications

StandardScaler transforms features by removing the mean and scaling to unit variance, resulting in a distribution with a mean of 0 and standard deviation of 1. This technique is crucial when features have varying scales and for algorithms sensitive to feature magnitudes like gradient descent-based methods.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Create sample dataset
data = np.array([[1, 2000, 3],
                 [2, 3000, 4],
                 [3, 4000, 5],
                 [4, 5000, 6]])

# Initialize and fit StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Show original vs scaled data
df = pd.DataFrame({
    'Original_F1': data[:, 0],
    'Original_F2': data[:, 1],
    'Original_F3': data[:, 2],
    'Scaled_F1': scaled_data[:, 0],
    'Scaled_F2': scaled_data[:, 1],
    'Scaled_F3': scaled_data[:, 2]
})

print("Original vs Scaled Data:")
print(df)
print("\nMean of scaled features:", scaled_data.mean(axis=0))
print("Std of scaled features:", scaled_data.std(axis=0))
```

Slide 2: MinMaxScaler Implementation

MinMaxScaler transforms features by scaling them to a given range, typically \[0,1\]. This technique preserves zero entries and is optimal when the distribution is not Gaussian or when standard deviation is small.

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Generate sample data with different scales
X = np.array([[1, -1, 2],
              [2, 0, 0],
              [0, 1, -1]])

# Initialize and apply MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Display results
print("Original data:\n", X)
print("\nScaled data:\n", X_scaled)
print("\nFeature ranges:")
print("Min values:", X_scaled.min(axis=0))
print("Max values:", X_scaled.max(axis=0))
```

Slide 3: RobustScaler for Outlier Handling

RobustScaler uses statistics that are robust to outliers. It removes the median and scales data according to the IQR (Interquartile Range). This approach makes it particularly useful for datasets where standard scaling might be influenced by extreme values.

```python
import numpy as np
from sklearn.preprocessing import RobustScaler

# Create dataset with outliers
data = np.array([[1], [2], [3], [1000], [4], [5]])

# Apply RobustScaler
robust_scaler = RobustScaler()
data_robust = robust_scaler.fit_transform(data)

# Compare with StandardScaler
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
data_standard = standard_scaler.fit_transform(data)

print("Original data:", data.ravel())
print("RobustScaler:", data_robust.ravel())
print("StandardScaler:", data_standard.ravel())
```

Slide 4: Normalizer Implementation

Normalizer scales individual samples independently of each other to have unit norm. This preprocessing step is useful for text classification and clustering, where the relative frequencies of features are more important than absolute frequencies.

```python
import numpy as np
from sklearn.preprocessing import Normalizer

# Create sample data
X = np.array([[4, 1, 2, 2],
              [1, 3, 9, 3],
              [5, 7, 5, 1]])

# Initialize and apply Normalizer
normalizer = Normalizer(norm='l2')  # l2 norm
X_normalized = normalizer.fit_transform(X)

# Calculate and display norms
original_norms = np.linalg.norm(X, axis=1)
normalized_norms = np.linalg.norm(X_normalized, axis=1)

print("Original data:\n", X)
print("\nNormalized data:\n", X_normalized)
print("\nOriginal norms:", original_norms)
print("Normalized norms:", normalized_norms)
```

Slide 5: Binarizer with Custom Thresholds

Binarizer transforms numerical features into binary values based on a threshold. This transformation is particularly useful in feature engineering when you need to convert continuous variables into binary flags or when implementing decision boundaries.

```python
import numpy as np
from sklearn.preprocessing import Binarizer

# Create sample continuous data
X = np.array([[0.1, 2.5, 3.8],
              [1.5, -0.2, 4.2],
              [3.2, 1.8, -0.5]])

# Initialize and apply Binarizer with different thresholds
binarizer_default = Binarizer(threshold=2.0)
X_binary = binarizer_default.fit_transform(X)

print("Original data:\n", X)
print("\nBinarized data (threshold=2.0):\n", X_binary)

# Example with multiple thresholds
thresholds = [1.0, 2.0, 3.0]
for threshold in thresholds:
    binarizer = Binarizer(threshold=threshold)
    print(f"\nBinarized data (threshold={threshold}):\n",
          binarizer.fit_transform(X))
```

Slide 6: LabelEncoder Advanced Usage

LabelEncoder transforms categorical labels into numerical values, enabling categorical data processing in machine learning algorithms. This implementation demonstrates handling multiple categories and dealing with unknown labels.

```python
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Sample categorical data
categories = np.array(['cat', 'dog', 'bird', 'cat', 'fish', 'dog', 'bird'])

# Initialize and fit LabelEncoder
label_encoder = LabelEncoder()
encoded_data = label_encoder.fit_transform(categories)

# Create mapping dictionary
label_mapping = dict(zip(label_encoder.classes_, 
                        label_encoder.transform(label_encoder.classes_)))

print("Original categories:", categories)
print("Encoded categories:", encoded_data)
print("\nLabel mapping:", label_mapping)

# Handle new/unknown categories
try:
    new_categories = np.array(['cat', 'dog', 'elephant'])
    print("\nTrying to transform new category:")
    print(label_encoder.transform(new_categories))
except ValueError as e:
    print("\nError handling unknown category:", str(e))
```

Slide 7: OneHotEncoder with Sparse Matrix

OneHotEncoder converts categorical features into a binary matrix format, creating new binary columns for each unique category. This implementation shows handling sparse matrices and dealing with unknown categories in production.

```python
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp

# Sample categorical data
X = np.array([['Red', 'Small'], 
              ['Blue', 'Medium'],
              ['Green', 'Large'],
              ['Red', 'Medium']])

# Initialize OneHotEncoder with sparse matrix
encoder = OneHotEncoder(sparse=True, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

print("Original shape:", X.shape)
print("Encoded shape:", X_encoded.shape)
print("\nFeature names:", encoder.get_feature_names_out())
print("\nSparse matrix:\n", X_encoded.toarray())

# Handle unknown categories
new_data = np.array([['Yellow', 'Extra Large']])
print("\nEncoding unknown category:")
print(encoder.transform(new_data).toarray())
```

Slide 8: PolynomialFeatures Implementation

PolynomialFeatures generates polynomial and interaction features, expanding the feature space to capture non-linear relationships. This implementation demonstrates different degrees of polynomial expansion and feature selection.

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Create sample data
X = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# Generate polynomial features
for degree in [2, 3]:
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X)
    
    print(f"\nDegree {degree} polynomial features:")
    print("Feature names:", poly.get_feature_names_out())
    print("Transformed data:\n", X_poly)
    print("Shape:", X_poly.shape)

# Example with interaction terms only
poly_interaction = PolynomialFeatures(degree=2, 
                                    interaction_only=True, 
                                    include_bias=False)
X_interaction = poly_interaction.fit_transform(X)
print("\nInteraction terms only:")
print("Feature names:", poly_interaction.get_feature_names_out())
print("Transformed data:\n", X_interaction)
```

Slide 9: Advanced SimpleImputer Techniques

SimpleImputer handles missing values through various strategies including mean, median, most\_frequent, and constant. This implementation demonstrates multiple imputation strategies and handling different data types simultaneously.

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Create dataset with missing values
data = np.array([
    [np.nan, 2, 3, np.nan],
    [4, np.nan, 6, 8],
    [10, 11, np.nan, 13],
    [14, 15, 16, np.nan]
])

# Different imputation strategies
strategies = ['mean', 'median', 'most_frequent', 'constant']
imputed_data = {}

for strategy in strategies:
    imputer = SimpleImputer(
        missing_values=np.nan,
        strategy=strategy,
        fill_value=999 if strategy == 'constant' else None
    )
    imputed_data[strategy] = imputer.fit_transform(data)

# Display results
for strategy, imp_data in imputed_data.items():
    print(f"\nImputed data using {strategy} strategy:")
    print(imp_data)
    print(f"Statistics for {strategy}:")
    print("Mean:", np.mean(imp_data, axis=0))
    print("Median:", np.median(imp_data, axis=0))
```

Slide 10: KNNImputer with Distance-Based Imputation

KNNImputer provides sophisticated missing value imputation using k-nearest neighbors algorithm. This implementation shows how to handle missing values based on feature similarity and distance metrics.

```python
import numpy as np
from sklearn.impute import KNNImputer
import pandas as pd

# Create dataset with missing values
X = np.array([
    [1.0, np.nan, 3.0, 4.0],
    [4.0, 2.0, np.nan, 1.0],
    [np.nan, 5.0, 6.0, 8.0],
    [2.0, 3.0, 4.0, np.nan],
    [7.0, 8.0, 9.0, 1.0]
])

# Initialize and apply KNNImputer with different configurations
imputers = {
    'uniform': KNNImputer(n_neighbors=2, weights='uniform'),
    'distance': KNNImputer(n_neighbors=2, weights='distance')
}

results = {}
for name, imputer in imputers.items():
    results[name] = imputer.fit_transform(X)
    
# Display and compare results
print("Original data with missing values:\n", X)
for name, imputed_data in results.items():
    print(f"\nImputed data using {name} weights:")
    print(imputed_data)
    print(f"Imputation statistics for {name}:")
    print("Mean:", np.mean(imputed_data, axis=0))
    print("Std:", np.std(imputed_data, axis=0))
```

Slide 11: PowerTransformer Implementation

PowerTransformer applies power transformations to make data more Gaussian-like. This implementation demonstrates both Box-Cox and Yeo-Johnson transformations with visualization of the distribution changes.

```python
import numpy as np
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt

# Generate skewed data
np.random.seed(42)
X_skewed = np.random.exponential(size=(1000, 1))

# Apply different power transformations
pt_boxcox = PowerTransformer(method='box-cox')
pt_yeojohnson = PowerTransformer(method='yeo-johnson')

# Transform data
X_boxcox = pt_boxcox.fit_transform(X_skewed + 1)  # Box-Cox requires positive values
X_yeojohnson = pt_yeojohnson.fit_transform(X_skewed)

# Print transformation parameters
print("Box-Cox lambda:", pt_boxcox.lambdas_)
print("Yeo-Johnson lambda:", pt_yeojohnson.lambdas_)

# Calculate and print normality statistics
from scipy import stats
print("\nSkewness:")
print("Original:", stats.skew(X_skewed))
print("Box-Cox:", stats.skew(X_boxcox))
print("Yeo-Johnson:", stats.skew(X_yeojohnson))

# Plot histograms
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.hist(X_skewed, bins=50)
ax1.set_title('Original Data')
ax2.hist(X_boxcox, bins=50)
ax2.set_title('Box-Cox Transformed')
ax3.hist(X_yeojohnson, bins=50)
ax3.set_title('Yeo-Johnson Transformed')
plt.tight_layout()
```

Slide 12: QuantileTransformer Advanced Usage

QuantileTransformer implements non-linear transformations to map data to a uniform or normal distribution. This implementation shows both distributions and demonstrates handling outliers while preserving feature rankings.

```python
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt

# Generate sample data with outliers
rng = np.random.RandomState(0)
n_samples = 1000
X = np.concatenate([
    rng.normal(0, 1, int(0.3 * n_samples)),
    rng.normal(5, 1, int(0.7 * n_samples))
])[:, np.newaxis]

# Initialize transformers
qt_uniform = QuantileTransformer(output_distribution='uniform', random_state=0)
qt_normal = QuantileTransformer(output_distribution='normal', random_state=0)

# Transform data
X_uniform = qt_uniform.fit_transform(X)
X_normal = qt_normal.fit_transform(X)

# Calculate statistics
print("Original Data Statistics:")
print(f"Mean: {X.mean():.2f}, Std: {X.std():.2f}")
print("\nUniform Transform Statistics:")
print(f"Mean: {X_uniform.mean():.2f}, Std: {X_uniform.std():.2f}")
print("\nNormal Transform Statistics:")
print(f"Mean: {X_normal.mean():.2f}, Std: {X_normal.std():.2f}")

# Verify rank preservation
original_ranks = np.argsort(X.ravel())
uniform_ranks = np.argsort(X_uniform.ravel())
normal_ranks = np.argsort(X_normal.ravel())

print("\nRank preservation check:")
print("Uniform transform:", np.array_equal(original_ranks, uniform_ranks))
print("Normal transform:", np.array_equal(original_ranks, normal_ranks))
```

Slide 13: MaxAbsScaler with Sparse Data

MaxAbsScaler scales each feature by its maximum absolute value, making it particularly useful for sparse data and maintaining zero values. This implementation demonstrates its application on both dense and sparse matrices.

```python
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import csr_matrix

# Create sample dense and sparse data
X_dense = np.array([[ 1., -1.,  2.],
                   [ 2.,  0.,  0.],
                   [ 0.,  1., -1.]])

X_sparse = csr_matrix([[ 1.,  0.,  2.],
                      [ 2.,  0.,  0.],
                      [ 0.,  1.,  0.]])

# Initialize and fit scaler
scaler = MaxAbsScaler()

# Transform both dense and sparse data
X_dense_scaled = scaler.fit_transform(X_dense)
X_sparse_scaled = scaler.fit_transform(X_sparse)

# Display results
print("Dense Data:")
print("Original:\n", X_dense)
print("\nScaled:\n", X_dense_scaled)
print("\nScaling factors:", scaler.scale_)

print("\nSparse Data:")
print("Original:\n", X_sparse.toarray())
print("\nScaled:\n", X_sparse_scaled.toarray())

# Verify zero preservation
print("\nZero Preservation Check:")
print("Dense zeros maintained:", 
      np.array_equal((X_dense == 0), (X_dense_scaled == 0)))
print("Sparse zeros maintained:", 
      np.array_equal((X_sparse.toarray() == 0), 
                    (X_sparse_scaled.toarray() == 0)))
```

Slide 14: Additional Resources

*   "A Survey on Data Preprocessing Methods" - [https://arxiv.org/abs/2106.00624](https://arxiv.org/abs/2106.00624)
*   "Comparative Study of Data Scaling Techniques in Machine Learning" - [https://arxiv.org/abs/1908.02839](https://arxiv.org/abs/1908.02839)
*   "Impact of Feature Scaling on Deep Neural Network Training" - [https://arxiv.org/abs/1912.01939](https://arxiv.org/abs/1912.01939)
*   For more detailed implementations and tutorials:
    *   scikit-learn documentation: [https://scikit-learn.org/stable/modules/preprocessing.html](https://scikit-learn.org/stable/modules/preprocessing.html)
    *   Towards Data Science: [https://towardsdatascience.com/preprocessing-techniques](https://towardsdatascience.com/preprocessing-techniques)
    *   Machine Learning Mastery: [https://machinelearningmastery.com/preprocessing-data](https://machinelearningmastery.com/preprocessing-data)

