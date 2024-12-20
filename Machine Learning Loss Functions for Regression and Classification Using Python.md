## Machine Learning Loss Functions for Regression and Classification Using Python
Slide 1: Mean Squared Error (MSE) for Regression

Mean Squared Error is a common loss function for regression tasks. It measures the average squared difference between predicted and actual values, penalizing larger errors more heavily.

```python
import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Example usage
y_true = np.array([3, 2, 5, 1, 7])
y_pred = np.array([2.5, 3.1, 4.8, 0.9, 6.3])

mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse}")
```

Slide 2: Mean Absolute Error (MAE) for Regression

Mean Absolute Error is another loss function for regression that measures the average absolute difference between predicted and actual values. It's less sensitive to outliers compared to MSE.

```python
import numpy as np

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Example usage
y_true = np.array([3, 2, 5, 1, 7])
y_pred = np.array([2.5, 3.1, 4.8, 0.9, 6.3])

mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Absolute Error: {mae}")
```

Slide 3: Binary Cross-Entropy for Binary Classification

Binary Cross-Entropy is the standard loss function for binary classification problems. It measures the performance of a model whose output is a probability value between 0 and 1.

```python
import numpy as np

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Small value to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example usage
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.3])

bce = binary_cross_entropy(y_true, y_pred)
print(f"Binary Cross-Entropy: {bce}")
```

Slide 4: Categorical Cross-Entropy for Multi-class Classification

Categorical Cross-Entropy is used for multi-class classification problems where each sample belongs to one of several classes.

```python
import numpy as np

def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# Example usage
y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

cce = categorical_cross_entropy(y_true, y_pred)
print(f"Categorical Cross-Entropy: {cce}")
```

Slide 5: Sparse Categorical Cross-Entropy

Sparse Categorical Cross-Entropy is similar to Categorical Cross-Entropy but is used when the true labels are provided as integers rather than one-hot encoded vectors.

```python
import numpy as np

def sparse_categorical_cross_entropy(y_true, y_pred):
    num_samples = y_true.shape[0]
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(np.log(y_pred[np.arange(num_samples), y_true])) / num_samples

# Example usage
y_true = np.array([0, 1, 2])
y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

scce = sparse_categorical_cross_entropy(y_true, y_pred)
print(f"Sparse Categorical Cross-Entropy: {scce}")
```

Slide 6: Hinge Loss for SVM Classification

Hinge Loss is commonly used in Support Vector Machines (SVM) for classification tasks. It encourages the correct class to have a score higher than the incorrect classes by a margin.

```python
import numpy as np

def hinge_loss(y_true, y_pred):
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

# Example usage
y_true = np.array([1, -1, 1, -1])
y_pred = np.array([0.9, -0.3, 0.2, -0.8])

hl = hinge_loss(y_true, y_pred)
print(f"Hinge Loss: {hl}")
```

Slide 7: Huber Loss for Regression

Huber Loss combines the best properties of MSE and MAE. It's less sensitive to outliers than MSE and provides a more useful gradient than MAE for small errors.

```python
import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))

# Example usage
y_true = np.array([3, 2, 5, 1, 7])
y_pred = np.array([2.5, 3.1, 4.8, 0.9, 6.3])

hl = huber_loss(y_true, y_pred)
print(f"Huber Loss: {hl}")
```

Slide 8: Focal Loss for Imbalanced Classification

Focal Loss is designed to address class imbalance problems in classification tasks. It down-weights the loss contribution from easy examples and focuses on hard ones.

```python
import numpy as np

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    pt = np.where(y_true == 1, y_pred, 1 - y_pred)
    return -np.mean(alpha * (1 - pt) ** gamma * np.log(pt))

# Example usage
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.3])

fl = focal_loss(y_true, y_pred)
print(f"Focal Loss: {fl}")
```

Slide 9: Wasserstein Loss for GANs

Wasserstein Loss is used in Wasserstein GANs (WGANs) to measure the distance between the real and generated probability distributions.

```python
import numpy as np

def wasserstein_loss(y_true, y_pred):
    return np.mean(y_true * y_pred)

# Example usage
# Assume y_true is 1 for real samples and -1 for generated samples
y_true = np.array([1, -1, 1, -1, 1])
y_pred = np.array([0.9, -0.8, 0.7, -0.6, 0.8])

wl = wasserstein_loss(y_true, y_pred)
print(f"Wasserstein Loss: {wl}")
```

Slide 10: Contrastive Loss for Siamese Networks

Contrastive Loss is used in Siamese networks to learn similarities between pairs of samples. It brings similar samples closer and pushes dissimilar samples apart in the embedding space.

```python
import numpy as np

def contrastive_loss(y_true, distance, margin=1.0):
    return np.mean(y_true * distance**2 + (1 - y_true) * np.maximum(0, margin - distance)**2)

# Example usage
y_true = np.array([1, 0, 1])  # 1 for similar pairs, 0 for dissimilar
distance = np.array([0.2, 0.8, 0.3])  # Euclidean distance between pairs

cl = contrastive_loss(y_true, distance)
print(f"Contrastive Loss: {cl}")
```

Slide 11: Triplet Loss for Face Recognition

Triplet Loss is commonly used in face recognition tasks. It aims to minimize the distance between an anchor and a positive sample while maximizing the distance between the anchor and a negative sample.

```python
import numpy as np

def triplet_loss(anchor, positive, negative, margin=0.2):
    pos_dist = np.sum((anchor - positive)**2)
    neg_dist = np.sum((anchor - negative)**2)
    loss = np.maximum(0, pos_dist - neg_dist + margin)
    return loss

# Example usage
anchor = np.array([1, 2, 3])
positive = np.array([1.1, 2.1, 3.1])
negative = np.array([4, 5, 6])

tl = triplet_loss(anchor, positive, negative)
print(f"Triplet Loss: {tl}")
```

Slide 12: Kullback-Leibler Divergence Loss

KL Divergence Loss measures the difference between two probability distributions. It's often used in variational autoencoders and other generative models.

```python
import numpy as np

def kl_divergence(p, q):
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return np.sum(p * np.log(p / q))

# Example usage
p = np.array([0.3, 0.4, 0.3])
q = np.array([0.25, 0.5, 0.25])

kl_div = kl_divergence(p, q)
print(f"KL Divergence: {kl_div}")
```

Slide 13: Cosine Proximity Loss

Cosine Proximity Loss measures the cosine of the angle between two vectors. It's useful when you're more interested in the direction of the vectors rather than their magnitude.

```python
import numpy as np

def cosine_proximity(y_true, y_pred):
    dot_product = np.sum(y_true * y_pred)
    norm_true = np.sqrt(np.sum(y_true**2))
    norm_pred = np.sqrt(np.sum(y_pred**2))
    return -dot_product / (norm_true * norm_pred)

# Example usage
y_true = np.array([1, 2, 3])
y_pred = np.array([2, 4, 6])

cp = cosine_proximity(y_true, y_pred)
print(f"Cosine Proximity: {cp}")
```

Slide 14: Additional Resources

For more in-depth information on loss functions and their applications in machine learning, consider exploring these resources:

1. "A Survey of Loss Functions for Semantic Segmentation" (ArXiv:2006.14822)
2. "Focal Loss for Dense Object Detection" (ArXiv:1708.02002)
3. "Understanding the difficulty of training deep feedforward neural networks" (ArXiv:1502.01852)
4. "Wasserstein GAN" (ArXiv:1701.07875)
5. "FaceNet: A Unified Embedding for Face Recognition and Clustering" (ArXiv:1503.03832)

These papers provide detailed insights into various loss functions and their applications in different machine learning tasks.

