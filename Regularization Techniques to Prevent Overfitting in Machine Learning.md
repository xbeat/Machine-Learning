## Regularization Techniques to Prevent Overfitting in Machine Learning

Slide 1: Regularization in Machine Learning

Regularization is a crucial technique in machine learning used to prevent overfitting. Overfitting occurs when a model learns the training data too well, including its noise and peculiarities, leading to poor performance on new, unseen data. Regularization adds a penalty to the model's loss function, encouraging simpler models that generalize better.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 2

# Fit models
lr = LinearRegression().fit(X, y)
ridge = Ridge(alpha=1.0).fit(X, y)
lasso = Lasso(alpha=0.1).fit(X, y)

# Plot results
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, lr.predict(X), color='red', label='Linear Regression')
plt.plot(X, ridge.predict(X), color='green', label='Ridge')
plt.plot(X, lasso.predict(X), color='orange', label='Lasso')
plt.legend()
plt.title('Regularization Comparison')
plt.show()
```

Slide 2: Understanding Overfitting

Overfitting is a common problem in machine learning where a model learns to fit the training data too closely, capturing noise and random fluctuations rather than the underlying pattern. This results in poor generalization to new, unseen data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(0)
X = np.sort(np.random.rand(20, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + np.random.randn(20) * 0.1

# Create and fit models
X_test = np.linspace(0, 1, 100).reshape(-1, 1)
plt.figure(figsize=(12, 4))

for i, degree in enumerate([1, 3, 15]):
    plt.subplot(1, 3, i + 1)
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_poly = model.predict(poly_features.transform(X_test))
    
    plt.scatter(X, y, color='blue', label='Training data')
    plt.plot(X_test, y_poly, color='red', label=f'Degree {degree}')
    plt.title(f'Polynomial Degree {degree}')
    plt.legend()

plt.tight_layout()
plt.show()
```

Slide 3: Introduction to Regularization

Regularization is a technique used to prevent overfitting by adding a penalty term to the model's loss function. This penalty encourages simpler models by constraining the model's parameters. The most common types of regularization are L1 (Lasso) and L2 (Ridge) regularization.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 20)
y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr = LinearRegression().fit(X_train, y_train)
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)

# Evaluate models
models = [lr, ridge, lasso]
names = ['Linear Regression', 'Ridge', 'Lasso']

for name, model in zip(names, models):
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"{name}: Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
```

Slide 4: L2 Regularization (Ridge)

Ridge regression, also known as L2 regularization, adds a penalty term to the loss function based on the sum of squared coefficients. This encourages smaller and more balanced weights across all features, making it effective for handling multicollinearity.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 50)
true_weights = np.random.randn(50) * 0.5
y = X.dot(true_weights) + np.random.randn(1000) * 0.1

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge models with different alpha values
alphas = [0.01, 0.1, 1, 10, 100]
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    train_mse = mean_squared_error(y_train, ridge.predict(X_train_scaled))
    test_mse = mean_squared_error(y_test, ridge.predict(X_test_scaled))
    print(f"Alpha: {alpha}, Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")

# Plot coefficient values for different alphas
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    plt.plot(range(50), ridge.coef_, label=f'Alpha: {alpha}')

plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regression Coefficients for Different Alpha Values')
plt.legend()
plt.show()
```

Slide 5: L1 Regularization (Lasso)

Lasso regression, or L1 regularization, adds a penalty term based on the sum of absolute coefficient values. This encourages sparsity by driving some coefficients to exactly zero, effectively performing feature selection. Lasso is particularly useful for high-dimensional datasets with many irrelevant features.

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 100)
true_weights = np.zeros(100)
true_weights[:10] = np.random.randn(10)
y = X.dot(true_weights) + np.random.randn(1000) * 0.1

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Lasso models with different alpha values
alphas = [0.001, 0.01, 0.1, 1, 10]
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_scaled, y_train)
    train_mse = mean_squared_error(y_train, lasso.predict(X_train_scaled))
    test_mse = mean_squared_error(y_test, lasso.predict(X_test_scaled))
    non_zero_coefs = np.sum(lasso.coef_ != 0)
    print(f"Alpha: {alpha}, Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}, Non-zero coefficients: {non_zero_coefs}")

# Plot coefficient values for different alphas
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_scaled, y_train)
    plt.plot(range(100), lasso.coef_, label=f'Alpha: {alpha}')

plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Lasso Regression Coefficients for Different Alpha Values')
plt.legend()
plt.show()
```

Slide 6: Elastic Net Regularization

Elastic Net combines L1 and L2 penalties, balancing feature selection and handling correlated features. It's useful for datasets with many features and multicollinearity, offering a compromise between Lasso and Ridge regression.

```python
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 100)
true_weights = np.zeros(100)
true_weights[:20] = np.random.randn(20)
y = X.dot(true_weights) + np.random.randn(1000) * 0.1

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Elastic Net models with different l1_ratio values
alphas = [0.1, 1, 10]
l1_ratios = [0.1, 0.5, 0.9]

for alpha in alphas:
    for l1_ratio in l1_ratios:
        elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        elastic_net.fit(X_train_scaled, y_train)
        train_mse = mean_squared_error(y_train, elastic_net.predict(X_train_scaled))
        test_mse = mean_squared_error(y_test, elastic_net.predict(X_test_scaled))
        non_zero_coefs = np.sum(elastic_net.coef_ != 0)
        print(f"Alpha: {alpha}, L1 ratio: {l1_ratio}, Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}, Non-zero coefficients: {non_zero_coefs}")

# Plot coefficient values for different l1_ratio values
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
alpha = 1.0
for l1_ratio in l1_ratios:
    elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    elastic_net.fit(X_train_scaled, y_train)
    plt.plot(range(100), elastic_net.coef_, label=f'L1 ratio: {l1_ratio}')

plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Elastic Net Coefficients for Different L1 Ratios (Alpha = 1.0)')
plt.legend()
plt.show()
```

Slide 7: Regularization in Neural Networks

Regularization techniques are also crucial in deep learning to prevent overfitting. Common methods include L1/L2 regularization, dropout, and early stopping. Here's an example of implementing L2 regularization in a neural network using TensorFlow and Keras.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model with L2 regularization
def create_model(l2_lambda):
    model = Sequential([
        Dense(64, activation='relu', kernel_regularizer=l2(l2_lambda), input_shape=(20,)),
        Dense(32, activation='relu', kernel_regularizer=l2(l2_lambda)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train models with different L2 lambda values
l2_lambdas = [0, 0.01, 0.1]
for l2_lambda in l2_lambdas:
    model = create_model(l2_lambda)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"L2 lambda: {l2_lambda}")
    print(f"Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")
    print(f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
    print()

# Plot training history for different L2 lambda values
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
for i, l2_lambda in enumerate(l2_lambdas):
    model = create_model(l2_lambda)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    plt.subplot(1, 3, i+1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'L2 Lambda: {l2_lambda}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

plt.tight_layout()
plt.show()
```

Slide 8: Cross-validation for Hyperparameter Tuning

Cross-validation is essential for finding the optimal regularization strength. It helps prevent overfitting to the validation set and provides a more robust estimate of model performance. Here's an example using scikit-learn's GridSearchCV to find the best alpha value for Ridge regression.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# Generate sample data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define parameter grid
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Create Ridge model
ridge = Ridge()

# Perform grid search with cross-validation
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_scaled, y)

# Print results
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", -grid_search.best_score_)

# Train final model with best parameters
best_ridge = Ridge(alpha=grid_search.best_params_['alpha'])
best_ridge.fit(X_scaled, y)

# Print coefficients of the best model
print("Number of non-zero coefficients:", np.sum(best_ridge.coef_ != 0))
```

Slide 9: Regularization in Decision Trees

Decision trees can also benefit from regularization to prevent overfitting. Common techniques include limiting tree depth, setting a minimum number of samples per leaf, and pruning. Here's an example using scikit-learn's DecisionTreeRegressor with different regularization parameters.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision trees with different max_depth values
max_depths = [2, 5, 10, None]
X_plot = np.linspace(0, 5, 100).reshape(-1, 1)

plt.figure(figsize=(14, 4))
for i, max_depth in enumerate(max_depths):
    regressor = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    regressor.fit(X_train, y_train)
    
    y_pred_train = regressor.predict(X_train)
    y_pred_test = regressor.predict(X_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    plt.subplot(1, 4, i + 1)
    plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_plot, regressor.predict(X_plot), color="cornflowerblue", label="prediction", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title(f"max_depth={max_depth}\nMSE train: {mse_train:.2f}, test: {mse_test:.2f}")
    plt.legend()

plt.tight_layout()
plt.show()
```

Slide 10: Early Stopping

Early stopping is a form of regularization that prevents overfitting by stopping the training process when the model's performance on a validation set starts to degrade. This technique is particularly useful in iterative learning algorithms like neural networks and gradient boosting.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 20)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(1000) * 0.1

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create model
model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, 
                    validation_split=0.2, callbacks=[early_stopping], verbose=0)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axvline(x=early_stopping.stopped_epoch, color='r', linestyle='--', label='Early Stopping Point')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training History with Early Stopping')
plt.legend()
plt.show()

print(f"Training stopped at epoch {early_stopping.stopped_epoch}")
```

Slide 11: Dropout Regularization

Dropout is a powerful regularization technique commonly used in neural networks. It randomly "drops out" a proportion of neurons during training, which helps prevent overfitting by reducing complex co-adaptations between neurons.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 20)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(1000) * 0.1

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create models with and without dropout
def create_model(use_dropout=False):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(20,)),
        Dropout(0.5) if use_dropout else Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dropout(0.3) if use_dropout else Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Train models
model_without_dropout = create_model(use_dropout=False)
history_without_dropout = model_without_dropout.fit(X_train_scaled, y_train, epochs=100, 
                                                    batch_size=32, validation_split=0.2, verbose=0)

model_with_dropout = create_model(use_dropout=True)
history_with_dropout = model_with_dropout.fit(X_train_scaled, y_train, epochs=100, 
                                              batch_size=32, validation_split=0.2, verbose=0)

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history_without_dropout.history['loss'], label='Without Dropout - Training')
plt.plot(history_without_dropout.history['val_loss'], label='Without Dropout - Validation')
plt.plot(history_with_dropout.history['loss'], label='With Dropout - Training')
plt.plot(history_with_dropout.history['val_loss'], label='With Dropout - Validation')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training History: With vs Without Dropout')
plt.legend()
plt.show()
```

Slide 12: Regularization in Practice: Image Classification

Let's apply regularization techniques to a real-world problem: image classification. We'll use a convolutional neural network (CNN) with L2 regularization and dropout to classify images from the CIFAR-10 dataset.

```python
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Create CNN model with regularization
def create_cnn_model(l2_lambda=0.01, dropout_rate=0.5):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_lambda), input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dropout(dropout_rate),
        layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# Train model
model = create_cnn_model()
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 13: Regularization in Natural Language Processing

Regularization is crucial in NLP tasks, especially when dealing with large vocabularies and complex models. Here's an example of using L2 regularization and dropout in a simple text classification task using TensorFlow and Keras.

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess data
max_features = 10000
maxlen = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Create model with regularization
def create_model(l2_lambda=0.01, dropout_rate=0.5):
    model = Sequential([
        Embedding(max_features, 128, input_length=maxlen),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu', kernel_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train model
model = create_model()
history = model.fit(x_train, y_train, epochs=10, batch_size=32, 
                    validation_split=0.2, verbose=1)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

For more information on regularization techniques in machine learning, consider exploring the following resources:

1.  "Regularization for Machine Learning: An Overview" by Girish Kaushik ArXiv link: [https://arxiv.org/abs/1803.10747](https://arxiv.org/abs/1803.10747)
2.  "A Survey of Deep Learning Techniques for Neural Machine Translation" by Shuohang Wang and Jing Jiang ArXiv link: [https://arxiv.org/abs/1804.09849](https://arxiv.org/abs/1804.09849)
3.  "An Overview of Regularization Techniques in Deep Learning" by Prakash Chandra Chhipa ArXiv link: [https://arxiv.org/abs/1901.10566](https://arxiv.org/abs/1901.10566)

These papers provide in-depth discussions on various regularization techniques and their applications in different domains of machine learning and deep learning.

