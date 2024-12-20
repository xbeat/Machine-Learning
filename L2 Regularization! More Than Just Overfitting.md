## L2 Regularization! More Than Just Overfitting
Slide 1: L2 Regularization: Beyond Overfitting

L2 regularization is a powerful technique in machine learning, often misunderstood as solely a tool for reducing overfitting. While it does serve this purpose, its capabilities extend far beyond. This presentation will explore the lesser-known but equally important applications of L2 regularization, particularly its role in addressing multicollinearity in linear models.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression

# Generate correlated features
np.random.seed(42)
X = np.random.randn(100, 2)
X[:, 1] = X[:, 0] + np.random.randn(100) * 0.1
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100)

# Fit models with and without L2 regularization
model_no_reg = LinearRegression().fit(X, y)
model_l2 = Ridge(alpha=1.0).fit(X, y)

print("Coefficients without L2:", model_no_reg.coef_)
print("Coefficients with L2:", model_l2.coef_)
```

Slide 2: Understanding Multicollinearity

Multicollinearity occurs when two or more features in a dataset are highly correlated or when one feature can be predicted from others. This phenomenon can lead to unstable and unreliable coefficient estimates in linear models. L2 regularization offers a solution to this problem by adding a penalty term to the model's loss function.

```python
# Visualize correlation between features
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1])
plt.title("Correlation between features")
plt.xlabel("Feature A")
plt.ylabel("Feature B")
plt.show()

# Calculate and print correlation coefficient
correlation = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
print(f"Correlation coefficient: {correlation:.4f}")
```

Slide 3: The Impact of Multicollinearity

When multicollinearity is present, the model's parameter estimates become sensitive to small changes in the data. This instability can lead to inflated standard errors and unreliable p-values, making it difficult to determine which features are truly important predictors of the target variable.

```python
# Function to create datasets with varying levels of multicollinearity
def create_multicollinear_data(n_samples, correlation):
    X1 = np.random.randn(n_samples)
    X2 = correlation * X1 + np.sqrt(1 - correlation**2) * np.random.randn(n_samples)
    y = 2 * X1 + 3 * X2 + np.random.randn(n_samples)
    return np.column_stack((X1, X2)), y

# Compare coefficient stability for different correlation levels
correlations = [0.5, 0.9, 0.99]
for corr in correlations:
    X, y = create_multicollinear_data(1000, corr)
    model = LinearRegression().fit(X, y)
    print(f"Correlation: {corr:.2f}, Coefficients: {model.coef_}")
```

Slide 4: L2 Regularization to the Rescue

L2 regularization, also known as Ridge regression, adds a penalty term to the loss function that is proportional to the square of the magnitude of the coefficients. This penalty encourages the model to use all features more equally, rather than relying heavily on a subset of highly correlated features.

```python
# Compare models with and without L2 regularization
X, y = create_multicollinear_data(1000, 0.99)

model_no_reg = LinearRegression().fit(X, y)
model_l2 = Ridge(alpha=1.0).fit(X, y)

print("Coefficients without L2:", model_no_reg.coef_)
print("Coefficients with L2:", model_l2.coef_)
```

Slide 5: Visualizing the Effect of L2 Regularization

To understand how L2 regularization affects the parameter space, we can visualize the residual sum of squares (RSS) for different combinations of model parameters. Without L2 regularization, we often see a valley in the RSS surface, indicating multiple near-optimal solutions. L2 regularization eliminates this valley, providing a unique global minimum.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rss(X, y, theta, alpha=0):
    return np.sum((y - X.dot(theta))**2) + alpha * np.sum(theta**2)

# Generate data
np.random.seed(42)
X = np.random.randn(100, 2)
X[:, 1] = X[:, 0] + np.random.randn(100) * 0.1
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100)

# Create grid of parameter values
theta1 = np.linspace(-5, 5, 100)
theta2 = np.linspace(-5, 5, 100)
T1, T2 = np.meshgrid(theta1, theta2)

# Calculate RSS for each parameter combination
Z_no_reg = np.array([rss(X, y, [t1, t2]) for t1, t2 in zip(T1.ravel(), T2.ravel())]).reshape(T1.shape)
Z_l2 = np.array([rss(X, y, [t1, t2], alpha=1.0) for t1, t2 in zip(T1.ravel(), T2.ravel())]).reshape(T1.shape)

# Plot RSS surfaces
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(T1, T2, Z_no_reg, cmap='viridis')
ax1.set_title('RSS without L2 Regularization')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(T1, T2, Z_l2, cmap='viridis')
ax2.set_title('RSS with L2 Regularization')

plt.tight_layout()
plt.show()
```

Slide 6: The Mathematics Behind L2 Regularization

L2 regularization modifies the objective function of linear regression by adding a penalty term. The new objective function becomes:

J(θ) = RSS(θ) + λ \* Σ(θ\_i^2)

Where RSS(θ) is the residual sum of squares, λ is the regularization strength, and Σ(θ\_i^2) is the sum of squared coefficients. This penalty term encourages smaller coefficient values, leading to a more stable and generalizable model.

```python
def objective_function(X, y, theta, lambda_):
    rss = np.sum((y - X.dot(theta))**2)
    l2_penalty = lambda_ * np.sum(theta**2)
    return rss + l2_penalty

# Example usage
lambda_ = 1.0
theta = np.array([2, 3])
obj_value = objective_function(X, y, theta, lambda_)
print(f"Objective function value: {obj_value:.4f}")
```

Slide 7: Tuning the Regularization Strength

The regularization strength, controlled by the λ parameter (often called 'alpha' in libraries like scikit-learn), determines how much we penalize large coefficients. Choosing the right value for λ is crucial: too small, and we don't address multicollinearity effectively; too large, and we risk underfitting the data.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = [0.001, 0.01, 0.1, 1, 10, 100]
train_errors, test_errors = [], []

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_errors.append(mean_squared_error(y_train, train_pred))
    test_errors.append(mean_squared_error(y_test, test_pred))

plt.figure(figsize=(10, 6))
plt.semilogx(alphas, train_errors, label='Train')
plt.semilogx(alphas, test_errors, label='Test')
plt.xlabel('Alpha (regularization strength)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Impact of Regularization Strength on Model Performance')
plt.show()
```

Slide 8: L2 Regularization vs. Feature Selection

While feature selection aims to identify and remove less important features, L2 regularization takes a different approach. It keeps all features but reduces the impact of less important ones by shrinking their coefficients. This can be particularly useful when features are correlated, as removing one might lose information captured by the interaction between features.

```python
from sklearn.feature_selection import SelectKBest, f_regression

# Generate data with correlated features
np.random.seed(42)
X = np.random.randn(1000, 5)
X[:, 3] = X[:, 0] + X[:, 1] + np.random.randn(1000) * 0.1
X[:, 4] = X[:, 2] + np.random.randn(1000) * 0.1
y = 2 * X[:, 0] + 3 * X[:, 1] + 4 * X[:, 2] + 5 * X[:, 3] + 6 * X[:, 4] + np.random.randn(1000)

# Feature selection
selector = SelectKBest(f_regression, k=3)
X_selected = selector.fit_transform(X, y)

# L2 regularization
model_l2 = Ridge(alpha=1.0).fit(X, y)

print("Selected features:", selector.get_support())
print("L2 regularized coefficients:", model_l2.coef_)
```

Slide 9: L2 Regularization in Neural Networks

L2 regularization isn't limited to linear models; it's widely used in neural networks as well. In this context, it's often called "weight decay" and helps prevent the network from relying too heavily on any particular input or neuron, leading to better generalization.

```python
import tensorflow as tf

def build_model(l2_strength):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_strength)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_strength)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Example usage
model_no_reg = build_model(0)
model_l2 = build_model(0.01)

# Train and evaluate models (not shown for brevity)
```

Slide 10: Real-Life Example: Housing Price Prediction

Consider a housing price prediction model where features like the number of rooms, square footage, and lot size are highly correlated. L2 regularization can help create a more robust model that doesn't overly rely on any single feature.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and prepare data
housing = fetch_california_housing()
X, y = housing.data, housing.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
model_no_reg = LinearRegression().fit(X_train, y_train)
model_l2 = Ridge(alpha=1.0).fit(X_train, y_train)

# Compare performance
print("No regularization R^2:", model_no_reg.score(X_test, y_test))
print("L2 regularization R^2:", model_l2.score(X_test, y_test))

# Compare coefficients
for name, coef_no_reg, coef_l2 in zip(housing.feature_names, model_no_reg.coef_, model_l2.coef_):
    print(f"{name}: No reg = {coef_no_reg:.4f}, L2 = {coef_l2:.4f}")
```

Slide 11: Real-Life Example: Image Classification

In image classification tasks, features (pixel values) are often highly correlated. L2 regularization helps prevent the model from relying too heavily on specific pixels or patterns, leading to better generalization across different images.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn(l2_strength):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_strength), input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_strength)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_strength)),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_strength)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Create and train models
model_no_reg = create_cnn(0)
model_l2 = create_cnn(0.001)

# Train models (not shown for brevity)
# Compare performance (not shown for brevity)
```

Slide 12: Limitations and Considerations

While L2 regularization is powerful, it's not a silver bullet. It may not perform well when true coefficients are large or when there are many irrelevant features. In such cases, other techniques like L1 regularization (Lasso) or Elastic Net might be more appropriate. Always consider the nature of your data and problem when choosing a regularization technique.

```python
from sklearn.linear_model import Lasso, ElasticNet

# Generate data with some large coefficients and irrelevant features
np.random.seed(42)
X = np.random.randn(1000, 10)
y = 10 * X[:, 0] + 20 * X[:, 1] + 0.1 * np.random.randn(1000)

# Compare different regularization techniques
models = {
    'No regularization': LinearRegression(),
    'L2 (Ridge)': Ridge(alpha=1.0),
    'L1 (Lasso)': Lasso(alpha=1.0),
    'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5)
}

for name, model in models.items():
    model.fit(X, y)
    print(f"{name} coefficients:", model.coef_)
```

Slide 13: Conclusion and Best Practices

L2 regularization is a versatile tool that goes beyond reducing overfitting. It effectively addresses multicollinearity, stabilizes model coefficients, and improves generalization. To make the most of L2 regularization, consider these best practices: standardize your features, use cross-validation to tune the regularization strength, and always compare the performance of regularized models against non-regularized ones.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 5)
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(1000) * 0.1

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compare models with cross-validation
model_no_reg = LinearRegression()
model_l2 = Ridge(alpha=1.0)

cv_scores_no_reg = cross_val_score(model_no_reg, X_scaled, y, cv=5)
cv_scores_l2 = cross_val_score(model_l2, X_scaled, y, cv=5)

print("No regularization CV scores:", cv_scores_no_reg.mean())
print("L2 regularization CV scores:", cv_scores_l2.mean())
```

Slide 14: Additional Resources

For those interested in diving deeper into L2 regularization and its applications, consider exploring these resources:

1. "Regularization for Machine Learning: An Overview" - ArXiv:1803.09111 This paper provides a comprehensive review of regularization techniques, including L2 regularization.
2. "Understanding the Bias-Variance Tradeoff" - ArXiv:1812.11118 This article explores the relationship between regularization and the bias-variance tradeoff.
3. "An Overview of Regularization Techniques in Deep Learning" - ArXiv:1905.05100 For those interested in L2 regularization in the context of neural networks, this paper offers valuable insights.

These resources offer a more in-depth exploration of the topics covered in this presentation, providing theoretical foundations and practical applications of L2 regularization in various machine learning contexts.

