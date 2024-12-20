## Introduction to Minimum Description Length Principle in Python
Slide 1: Introduction to the Minimum Description Length Principle

The Minimum Description Length (MDL) principle is a formalization of Occam's Razor in which the best hypothesis for a given set of data is the one that leads to the best compression of the data. MDL is used for model selection, statistical inference, and machine learning.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.normal(0, 1, 100)

# Plot the data
plt.scatter(x, y, label='Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sample Data for MDL Principle')
plt.legend()
plt.show()
```

Slide 2: The Basics of MDL

MDL balances model complexity with goodness of fit. It suggests that the best model is one that provides the shortest description of the data, including the model itself and the data given the model.

```python
import math

def model_cost(k):
    return k * math.log2(100)  # Assuming 100 data points

def data_cost(y, y_pred):
    return sum((y_i - y_pred_i)**2 for y_i, y_pred_i in zip(y, y_pred))

def total_cost(k, y, y_pred):
    return model_cost(k) + data_cost(y, y_pred)
```

Slide 3: Two-Part MDL

The two-part MDL focuses on finding a model that minimizes the sum of the description length of the model and the description length of the data when encoded with the help of the model.

```python
def two_part_mdl(data, models):
    best_model = None
    min_cost = float('inf')
    
    for model in models:
        model_desc_length = len(str(model))  # Simplified
        data_desc_length = len(str(data - model(data)))  # Simplified
        total_length = model_desc_length + data_desc_length
        
        if total_length < min_cost:
            min_cost = total_length
            best_model = model
    
    return best_model
```

Slide 4: Practical Example: Polynomial Regression

Let's use MDL to select the best degree for polynomial regression on a dataset.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def mdl_polynomial_regression(X, y, max_degree):
    best_degree = 1
    min_mdl = float('inf')
    
    for degree in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X.reshape(-1, 1))
        
        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        mse = mean_squared_error(y, y_pred)
        mdl = len(model.coef_) * np.log2(len(X)) + len(X) * np.log2(mse)
        
        if mdl < min_mdl:
            min_mdl = mdl
            best_degree = degree
    
    return best_degree

# Example usage
X = np.linspace(0, 10, 100)
y = 3 * X**2 + 2 * X + 1 + np.random.normal(0, 5, 100)

best_degree = mdl_polynomial_regression(X, y, 5)
print(f"Best polynomial degree according to MDL: {best_degree}")
```

Slide 5: MDL for Feature Selection

MDL can be used for feature selection in machine learning, helping to choose the most relevant features while avoiding overfitting.

```python
from sklearn.feature_selection import mutual_info_regression

def mdl_feature_selection(X, y, threshold=0.1):
    mi_scores = mutual_info_regression(X, y)
    selected_features = [i for i, score in enumerate(mi_scores) if score > threshold]
    
    mdl = len(selected_features) * np.log2(X.shape[1])  # Model description length
    mdl += X.shape[0] * np.log2(np.var(y - X[:, selected_features].mean(axis=1)))  # Data description length
    
    return selected_features, mdl

# Example usage
X = np.random.rand(100, 10)
y = 2 * X[:, 0] + 3 * X[:, 2] + np.random.normal(0, 0.1, 100)

selected_features, mdl_score = mdl_feature_selection(X, y)
print(f"Selected features: {selected_features}")
print(f"MDL score: {mdl_score}")
```

Slide 6: MDL for Model Selection

MDL provides a principled approach to model selection, balancing model complexity with goodness of fit.

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split

def mdl_model_selection(X, y, models):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    best_model = None
    min_mdl = float('inf')
    
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        model_complexity = len(model.coef_) * np.log2(len(X))
        mdl = model_complexity + len(X) * np.log2(mse)
        
        if mdl < min_mdl:
            min_mdl = mdl
            best_model = model
    
    return best_model

# Example usage
models = [LinearRegression(), Ridge(), Lasso()]
best_model = mdl_model_selection(X, y, models)
print(f"Best model according to MDL: {type(best_model).__name__}")
```

Slide 7: MDL for Time Series Analysis

MDL can be applied to time series analysis for model order selection in autoregressive (AR) models.

```python
from statsmodels.tsa.ar_model import AutoReg

def mdl_ar_order_selection(time_series, max_order):
    best_order = 0
    min_mdl = float('inf')
    
    for order in range(1, max_order + 1):
        model = AutoReg(time_series, lags=order).fit()
        
        aic = model.aic
        mdl = order * np.log2(len(time_series)) + aic
        
        if mdl < min_mdl:
            min_mdl = mdl
            best_order = order
    
    return best_order

# Example usage
np.random.seed(0)
time_series = np.cumsum(np.random.normal(0, 1, 1000))
best_order = mdl_ar_order_selection(time_series, max_order=10)
print(f"Best AR order according to MDL: {best_order}")
```

Slide 8: MDL for Clustering

MDL can be used to determine the optimal number of clusters in clustering algorithms.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def mdl_clustering(X, max_clusters):
    best_n_clusters = 2
    min_mdl = float('inf')
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        
        cluster_desc_length = n_clusters * X.shape[1] * np.log2(X.shape[0])
        data_desc_length = -np.sum(silhouette_score(X, kmeans.labels_) * np.log2(X.shape[0]))
        
        mdl = cluster_desc_length + data_desc_length
        
        if mdl < min_mdl:
            min_mdl = mdl
            best_n_clusters = n_clusters
    
    return best_n_clusters

# Example usage
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
best_n_clusters = mdl_clustering(X, max_clusters=10)
print(f"Best number of clusters according to MDL: {best_n_clusters}")
```

Slide 9: MDL for Neural Network Architecture Selection

MDL can guide the selection of neural network architectures by balancing model complexity with performance.

```python
import tensorflow as tf

def mdl_nn_architecture(X, y, architectures):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    best_architecture = None
    min_mdl = float('inf')
    
    for architecture in architectures:
        model = tf.keras.Sequential(architecture)
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=100, verbose=0)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        model_complexity = sum(layer.count_params() for layer in model.layers) * np.log2(len(X))
        mdl = model_complexity + len(X) * np.log2(mse)
        
        if mdl < min_mdl:
            min_mdl = mdl
            best_architecture = architecture
    
    return best_architecture

# Example usage
architectures = [
    [tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(1)],
    [tf.keras.layers.Dense(20, activation='relu'), tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(1)],
    [tf.keras.layers.Dense(30, activation='relu'), tf.keras.layers.Dense(20, activation='relu'), tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(1)]
]

best_architecture = mdl_nn_architecture(X, y, architectures)
print(f"Best neural network architecture according to MDL: {len(best_architecture)} layers")
```

Slide 10: MDL for Image Compression

MDL principles can be applied to image compression, balancing compression ratio with image quality.

```python
from PIL import Image
import io

def mdl_image_compression(image_path, quality_range):
    original_image = Image.open(image_path)
    best_quality = 0
    min_mdl = float('inf')
    
    for quality in quality_range:
        buffer = io.BytesIO()
        original_image.save(buffer, format="JPEG", quality=quality)
        compressed_size = buffer.getbuffer().nbytes
        
        # Simplified MDL calculation
        mdl = compressed_size + abs(quality - 100) * np.log2(original_image.size[0] * original_image.size[1])
        
        if mdl < min_mdl:
            min_mdl = mdl
            best_quality = quality
    
    return best_quality

# Example usage
best_quality = mdl_image_compression("example_image.jpg", range(1, 101, 5))
print(f"Best JPEG quality according to MDL: {best_quality}")
```

Slide 11: MDL for Text Classification

MDL can be used in text classification to select the most relevant features (words) for categorizing documents.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

def mdl_text_classification(texts, labels, max_features_range):
    best_max_features = 0
    min_mdl = float('inf')
    
    for max_features in max_features_range:
        vectorizer = CountVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(texts)
        
        model = MultinomialNB()
        scores = cross_val_score(model, X, labels, cv=5, scoring='neg_log_loss')
        
        mdl = max_features * np.log2(len(texts)) - np.mean(scores) * len(texts)
        
        if mdl < min_mdl:
            min_mdl = mdl
            best_max_features = max_features
    
    return best_max_features

# Example usage
texts = ["This is a positive review", "Negative sentiment here", "Another positive one"]
labels = [1, 0, 1]
best_max_features = mdl_text_classification(texts, labels, range(1, 11))
print(f"Best number of features for text classification according to MDL: {best_max_features}")
```

Slide 12: MDL for Anomaly Detection

MDL can be applied to anomaly detection by identifying data points that require more bits to encode, indicating they are outliers.

```python
from scipy.stats import norm

def mdl_anomaly_detection(data, threshold=2):
    mean = np.mean(data)
    std = np.std(data)
    
    def encode_length(x):
        p = norm.pdf(x, mean, std)
        return -np.log2(p)
    
    encoding_lengths = [encode_length(x) for x in data]
    mdl_scores = np.array(encoding_lengths)
    
    anomalies = data[mdl_scores > threshold * np.mean(mdl_scores)]
    return anomalies

# Example usage
data = np.concatenate([np.random.normal(0, 1, 1000), np.random.normal(5, 1, 10)])
anomalies = mdl_anomaly_detection(data)
print(f"Number of anomalies detected: {len(anomalies)}")
```

Slide 13: MDL for Decision Tree Pruning

MDL can be used to prune decision trees, balancing tree complexity with its predictive power.

```python
from sklearn.tree import DecisionTreeClassifier

def mdl_tree_pruning(X, y, max_depth_range):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    best_depth = 0
    min_mdl = float('inf')
    
    for max_depth in max_depth_range:
        tree = DecisionTreeClassifier(max_depth=max_depth)
        tree.fit(X_train, y_train)
        
        y_pred = tree.predict(X_test)
        misclassification = np.sum(y_test != y_pred)
        
        tree_complexity = tree.tree_.node_count * np.log2(len(X))
        mdl = tree_complexity + misclassification * np.log2(len(X))
        
        if mdl < min_mdl:
            min_mdl = mdl
            best_depth = max_depth
    
    return best_depth

# Example usage
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=42)
best_depth = mdl_tree_pruning(X, y, range(1, 21))
print(f"Best decision tree depth according to MDL: {best_depth}")
```

Slide 14: MDL for Model Averaging

MDL can be used to assign weights to different models in an ensemble, based on their complexity and performance.

```python
def mdl_model_averaging(X, y, models):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    mdl_scores = []
    predictions = []
    
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Simplified MDL calculation
        complexity = len(str(model))  # Proxy for model complexity
        mdl = complexity * np.log2(len(X)) + len(X) * np.log2(mse)
        
        mdl_scores.append(mdl)
        predictions.append(y_pred)
    
    weights = 1 / np.array(mdl_scores)
    weights /= np.sum(weights)
    
    final_prediction = np.average(predictions, axis=0, weights=weights)
    return final_prediction, weights

# Example usage
models = [LinearRegression(), RandomForestRegressor(), GradientBoostingRegressor()]
final_pred, model_weights = mdl_model_averaging(X, y, models)
print("Model weights based on MDL:", model_weights)
```

Slide 15: Limitations and Considerations of MDL

While MDL is a powerful principle, it's important to be aware of its limitations and considerations in practical applications.

```python
def mdl_limitations_demo():
    # Small sample size limitation
    small_data = np.random.rand(10, 5)
    small_target = np.random.rand(10)
    
    # Computational complexity for large feature spaces
    large_data = np.random.rand(1000, 1000)
    large_target = np.random.rand(1000)
    
    # Sensitivity to data representation
    binary_data = np.random.choice([0, 1], size=(100, 10))
    continuous_data = np.random.rand(100, 10)
    
    print("Small sample size might lead to overfitting in MDL")
    print("Large feature spaces can be computationally expensive")
    print("Different data representations can affect MDL scores")

mdl_limitations_demo()
```

Slide 16: Additional Resources

For further exploration of the Minimum Description Length principle, consider the following resources:

1. Grünwald, P. D. (2007). The Minimum Description Length Principle. MIT Press. ArXiv: [https://arxiv.org/abs/math/0406077](https://arxiv.org/abs/math/0406077)
2. Rissanen, J. (1978). Modeling by shortest data description. Automatica, 14(5), 465-471. DOI: 10.1016/0005-1098(78)90005-5
3. Myung, I. J., Navarro, D. J., & Pitt, M. A. (2006). Model selection by normalized maximum likelihood. Journal of Mathematical Psychology, 50(2), 167-179. ArXiv: [https://arxiv.org/abs/math/0412033](https://arxiv.org/abs/math/0412033)
4. Lee, P. M. (2012). Bayesian Statistics: An Introduction. Wiley. ISBN: 978-1118332573

These resources provide in-depth discussions on the theoretical foundations and practical applications of the MDL principle in various fields of study.

