## Mastering Bias, Variance, Overfitting, and Underfitting in Python
Slide 1: Understanding Bias and Variance

Bias and variance are fundamental concepts in machine learning that help us understand model performance. Bias refers to the error introduced by approximating a real-world problem with a simplified model, while variance is the model's sensitivity to fluctuations in the training data.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = 3*X + 2 + np.random.randn(100) * 2

# Fit models with different degrees
degrees = [1, 5, 15]
plt.figure(figsize=(12, 4))

for i, degree in enumerate(degrees):
    plt.subplot(1, 3, i+1)
    poly_features = np.polynomial.polynomial.polyvander(X, degree)
    coef = np.linalg.lstsq(poly_features, y, rcond=None)[0]
    y_pred = np.dot(poly_features, coef)
    plt.scatter(X, y, alpha=0.7)
    plt.plot(X, y_pred, color='r')
    plt.title(f'Degree {degree} Polynomial')

plt.tight_layout()
plt.show()
```

Slide 2: The Bias-Variance Tradeoff

The bias-variance tradeoff is a central problem in supervised learning. As we increase model complexity, bias tends to decrease, but variance increases. The goal is to find the sweet spot that minimizes both bias and variance, resulting in the best generalization performance.

```python
import numpy as np
import matplotlib.pyplot as plt

def bias_variance_demo(n_samples=100, n_features=1):
    np.random.seed(0)
    X = np.random.rand(n_samples, n_features)
    y = 3 * X.sum(axis=1) + 2 + np.random.randn(n_samples) * 0.5

    degrees = range(1, 11)
    train_errors, test_errors = [], []

    for degree in degrees:
        poly_features = np.polynomial.polynomial.polyvander(X.flatten(), degree)
        model = np.linalg.lstsq(poly_features, y, rcond=None)[0]
        
        train_pred = np.dot(poly_features, model)
        train_error = np.mean((y - train_pred) ** 2)
        train_errors.append(train_error)

        X_test = np.random.rand(n_samples, n_features)
        y_test = 3 * X_test.sum(axis=1) + 2 + np.random.randn(n_samples) * 0.5
        poly_features_test = np.polynomial.polynomial.polyvander(X_test.flatten(), degree)
        test_pred = np.dot(poly_features_test, model)
        test_error = np.mean((y_test - test_pred) ** 2)
        test_errors.append(test_error)

    plt.plot(degrees, train_errors, label='Training Error')
    plt.plot(degrees, test_errors, label='Test Error')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.title('Bias-Variance Tradeoff')
    plt.show()

bias_variance_demo()
```

Slide 3: Overfitting: When Models Learn Too Much

Overfitting occurs when a model learns the training data too well, including its noise and peculiarities. This results in poor generalization to new, unseen data. Overfitted models have low bias but high variance.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train models with different degrees
degrees = [1, 4, 15]
plt.figure(figsize=(14, 4))

for i, degree in enumerate(degrees):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_train = poly_features.fit_transform(X_train)
    X_poly_test = poly_features.transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    y_train_pred = model.predict(X_poly_train)
    y_test_pred = model.predict(X_poly_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    plt.subplot(1, 3, i+1)
    plt.scatter(X_train, y_train, color='b', label='Training data')
    plt.scatter(X_test, y_test, color='r', label='Test data')
    plt.plot(X, model.predict(poly_features.transform(X)), color='g', label='Prediction')
    plt.title(f'Degree {degree} Polynomial\nTrain MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}')
    plt.legend()

plt.tight_layout()
plt.show()
```

Slide 4: Underfitting: When Models Learn Too Little

Underfitting happens when a model is too simple to capture the underlying pattern in the data. Underfitted models have high bias but low variance. They perform poorly on both training and test data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Generate non-linear data
np.random.seed(0)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Fit linear and quadratic models
models = [
    ("Linear", LinearRegression()),
    ("Quadratic", LinearRegression())
]

plt.figure(figsize=(12, 5))

for i, (name, model) in enumerate(models):
    if name == "Quadratic":
        X_poly = PolynomialFeatures(degree=2).fit_transform(X)
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
    else:
        model.fit(X, y)
        y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)

    plt.subplot(1, 2, i+1)
    plt.scatter(X, y, color='b', label='Data')
    plt.plot(X, y_pred, color='r', label='Prediction')
    plt.title(f'{name} Model\nMSE: {mse:.4f}')
    plt.legend()

plt.tight_layout()
plt.show()
```

Slide 5: Regularization: Balancing Complexity

Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function. This encourages simpler models and helps strike a balance between bias and variance.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Create and fit models
X_plot = np.linspace(0, 5, 100)[:, np.newaxis]
models = [
    ("Ridge", Ridge(alpha=1, random_state=0)),
    ("Lasso", Lasso(alpha=0.1, random_state=0))
]

plt.figure(figsize=(12, 5))

for i, (name, model) in enumerate(models):
    plt.subplot(1, 2, i+1)
    for degree in [3, 10]:
        model = make_pipeline(PolynomialFeatures(degree), model)
        model.fit(X, y)
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, label=f'degree={degree}')
    
    plt.scatter(X, y, color='navy', s=30, marker='o', label="training points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{name} Regression")
    plt.legend(loc="best")

plt.tight_layout()
plt.show()
```

Slide 6: Cross-Validation: Assessing Model Performance

Cross-validation is a crucial technique for evaluating model performance and detecting overfitting. It involves partitioning the data into subsets, training on a portion, and validating on the held-out set.

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Perform cross-validation for different polynomial degrees
degrees = range(1, 11)
cv_scores = []

for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

# Plot results
plt.plot(degrees, cv_scores, marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Cross-Validation Scores for Different Polynomial Degrees')
plt.show()
```

Slide 7: Feature Selection: Choosing Relevant Variables

Feature selection helps reduce overfitting by identifying the most important variables in your dataset. This process can improve model performance and interpretability.

```python
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data with 20 features, only 5 of which are informative
X, y = make_regression(n_samples=100, n_features=20, n_informative=5, 
                       noise=0.1, random_state=42)

# Perform feature selection
selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X, y)

# Get feature importance scores
scores = selector.scores_

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(scores)), scores)
plt.xlabel('Feature Index')
plt.ylabel('F-score')
plt.title('Feature Importance')

# Highlight selected features
selected_features = selector.get_support()
plt.bar(range(len(scores)), scores, color=['red' if selected else 'blue' for selected in selected_features])

plt.show()

print(f"Number of original features: {X.shape[1]}")
print(f"Number of selected features: {X_selected.shape[1]}")
```

Slide 8: Learning Curves: Visualizing Model Performance

Learning curves help diagnose bias and variance problems by showing how model performance changes with increasing training set size.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")

plt.figure(figsize=(12, 5))

plt.subplot(121)
plot_learning_curve(LinearRegression(), X, y, "Learning Curve (Linear Model)")

plt.subplot(122)
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
plot_learning_curve(LinearRegression(), X_poly, y, "Learning Curve (Polynomial Model)")

plt.tight_layout()
plt.show()
```

Slide 9: Ensemble Methods: Combining Models

Ensemble methods, such as Random Forests and Gradient Boosting, can help reduce overfitting by combining multiple models. These techniques often provide a good balance between bias and variance.

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Make predictions
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_rf = rf.predict(X_plot)
y_gb = gb.predict(X_plot)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(X, y, alpha=0.5)
plt.plot(X_plot, y_rf, label='Random Forest', color='r')
plt.title('Random Forest Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.subplot(122)
plt.scatter(X, y, alpha=0.5)
plt.plot(X_plot, y_gb, label='Gradient Boosting', color='g')
plt.title('Gradient Boosting Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()

# Print MSE for both models
print(f"Random Forest MSE: {mean_squared_error(y_test, rf.predict(X_test)):.4f}")
print(f"Gradient Boosting MSE: {mean_squared_error(y_test, gb.predict(X_test)):.4f}")
```

Slide 10: Hyperparameter Tuning: Optimizing Model Performance

Hyperparameter tuning is crucial for finding the right balance between bias and variance. Techniques like Grid Search and Random Search can help identify the best hyperparameters for your model.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best negative MSE:", -grid_search.best_score_)

# Plot results
results = grid_search.cv_results_
plt.figure(figsize=(12, 6))
plt.plot(results['param_n_estimators'], -results['mean_test_score'], 'bo-')
plt.xlabel('Number of estimators')
plt.ylabel('Mean Squared Error')
plt.title('Grid Search Results: n_estimators vs MSE')
plt.show()
```

Slide 11: Regularization in Neural Networks: Dropout

Dropout is a powerful regularization technique used in neural networks to prevent overfitting. It randomly "drops out" a portion of neurons during training, forcing the network to learn more robust features.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create models with and without dropout
model_with_dropout = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_without_dropout = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile models
model_with_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_without_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train models
history_with_dropout = model_with_dropout.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)
history_without_dropout = model_without_dropout.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(history_with_dropout.history['accuracy'], label='Train (with dropout)')
plt.plot(history_with_dropout.history['val_accuracy'], label='Validation (with dropout)')
plt.plot(history_without_dropout.history['accuracy'], label='Train (without dropout)')
plt.plot(history_without_dropout.history['val_accuracy'], label='Validation (without dropout)')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(122)
plt.plot(history_with_dropout.history['loss'], label='Train (with dropout)')
plt.plot(history_with_dropout.history['val_loss'], label='Validation (with dropout)')
plt.plot(history_without_dropout.history['loss'], label='Train (without dropout)')
plt.plot(history_without_dropout.history['val_loss'], label='Validation (without dropout)')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 12: Real-life Example: Image Classification

In this example, we'll use a Convolutional Neural Network (CNN) to classify images of cats and dogs. We'll demonstrate how data augmentation and dropout can help prevent overfitting in image classification tasks.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess the data
train_generator = train_datagen.flow_from_directory(
    'path/to/train/directory',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    'path/to/validation/directory',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

# Plot the training history
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(122)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 13: Real-life Example: Time Series Forecasting

In this example, we'll use an LSTM (Long Short-Term Memory) network to forecast time series data. We'll demonstrate how to use early stopping to prevent overfitting in time series prediction tasks.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Generate sample time series data
np.random.seed(0)
dates = pd.date_range(start='2010-01-01', end='2023-01-01', freq='D')
ts = pd.Series(np.cumsum(np.random.randn(len(dates))), index=dates)

# Prepare data for LSTM
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(ts.values.reshape(-1, 1))

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(scaled_data, seq_length)

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(ts.index[seq_length:train_size], y_train_inv, label='Actual (Train)')
plt.plot(ts.index[seq_length:train_size], train_predict, label='Predicted (Train)')
plt.plot(ts.index[train_size+seq_length:], y_test_inv, label='Actual (Test)')
plt.plot(ts.index[train_size+seq_length:], test_predict, label='Predicted (Test)')
plt.title('Time Series Forecasting with LSTM')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
```

Slide 14: Additional Resources

For more in-depth information on bias, variance, overfitting, and underfitting, consider exploring the following resources:

1. "Understanding the Bias-Variance Tradeoff" by Scott Fortmann-Roe ([http://scott.fortmann-roe.com/docs/BiasVariance.html](http://scott.fortmann-roe.com/docs/BiasVariance.html))
2. "Regularization for Deep Learning: A Taxonomy" by Kukacev et al. ([https://arxiv.org/abs/1710.10686](https://arxiv.org/abs/1710.10686))
3. "An Overview of Deep Learning Regularization" by Goodfellow et al. ([https://arxiv.org/abs/1512.07108](https://arxiv.org/abs/1512.07108))
4. "A Unified Approach to Interpreting Model Predictions" by Lundberg and Lee ([https://arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874))
5. "XGBoost: A Scalable Tree Boosting System" by Chen and Guestrin ([https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754))

These resources provide a mix of theoretical foundations and practical applications in machine learning and deep learning. They can help deepen your understanding of model performance optimization and regularization techniques.

