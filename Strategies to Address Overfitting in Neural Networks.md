## Strategies to Address Overfitting in Neural Networks
Slide 1: Understanding Overfitting in Neural Networks

Overfitting occurs when a neural network learns the training data too well, including noise and outliers, leading to poor generalization on unseen data. This phenomenon is characterized by high accuracy on the training set but significantly lower performance on the test set.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

# Fit polynomials of different degrees
degrees = [1, 3, 15]
plt.figure(figsize=(15, 5))

for i, degree in enumerate(degrees):
    plt.subplot(1, 3, i+1)
    coeffs = np.polyfit(X.ravel(), y.ravel(), degree)
    y_pred = np.polyval(coeffs, X.ravel())
    plt.scatter(X, y, alpha=0.7)
    plt.plot(X, y_pred, color='r')
    plt.title(f'Degree {degree} Polynomial')
    plt.ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()
```

Slide 2: Regularization - L1 and L2

Regularization techniques add a penalty term to the loss function, discouraging complex models. L1 (Lasso) regularization promotes sparsity, while L2 (Ridge) regularization prevents large weight values.

```python
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

# Prepare data
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# L1 Regularization (Lasso)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

# L2 Regularization (Ridge)
ridge = Ridge(alpha=0.1)
ridge.fit(X_train_scaled, y_train)

print("Lasso coefficients:", lasso.coef_)
print("Ridge coefficients:", ridge.coef_)
```

Slide 3: Dropout

Dropout is a regularization technique that randomly deactivates a fraction of neurons during training, reducing co-adaptation and improving generalization.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

Slide 4: Early Stopping

Early stopping monitors the model's performance on a validation set during training and stops when the performance starts to degrade, preventing overfitting.

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=1000,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

Slide 5: Data Augmentation

Data augmentation artificially increases the size of the training set by applying transformations to existing data, helping the model learn invariant features and reduce overfitting.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Assume X_train is a 4D tensor of images
augmented_data = datagen.flow(X_train, y_train, batch_size=32)

# Visualize augmented images
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
for i, (img, _) in enumerate(augmented_data.take(6)):
    ax = axs[i // 3, i % 3]
    ax.imshow(img[0])
    ax.axis('off')
plt.show()
```

Slide 6: Cross-Validation

Cross-validation helps assess the model's performance and generalization by splitting the data into multiple train-test sets and averaging the results.

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scores = []
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model = tf.keras.Sequential([Dense(64, activation='relu', input_shape=(1,)),
                                 Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, verbose=0)
    
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mse_scores.append(mse)

print(f"Average MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
```

Slide 7: Ensemble Methods

Ensemble methods combine predictions from multiple models to reduce overfitting and improve generalization. Common techniques include bagging, boosting, and stacking.

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Train individual models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
lr = LinearRegression()

models = [rf, gb, lr]
for model in models:
    model.fit(X_train, y_train)

# Make predictions
predictions = np.column_stack([model.predict(X_test) for model in models])

# Simple averaging ensemble
ensemble_pred = np.mean(predictions, axis=1)

mse = mean_squared_error(y_test, ensemble_pred)
print(f"Ensemble MSE: {mse:.4f}")
```

Slide 8: Batch Normalization

Batch normalization normalizes the inputs of each layer, reducing internal covariate shift and allowing higher learning rates, which can help prevent overfitting.

```python
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Dense(64, input_shape=(10,)),
    BatchNormalization(),
    Activation('relu'),
    Dense(32),
    BatchNormalization(),
    Activation('relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

Slide 9: Weight Decay

Weight decay is a form of L2 regularization that adds a penalty term to the loss function, encouraging smaller weights and reducing model complexity.

```python
from tensorflow.keras.regularizers import l2

model = Sequential([
    Dense(64, activation='relu', input_shape=(10,), kernel_regularizer=l2(0.01)),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

Slide 10: Pruning

Pruning removes unnecessary connections or neurons from a trained network, reducing model complexity and potentially improving generalization.

```python
import tensorflow_model_optimization as tfmot

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=0.5,
        begin_step=0, end_step=1000)
}

model = Sequential([
    tfmot.sparsity.keras.prune_low_magnitude(
        Dense(64, activation='relu', input_shape=(10,)),
        **pruning_params),
    tfmot.sparsity.keras.prune_low_magnitude(
        Dense(32, activation='relu'),
        **pruning_params),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

Slide 11: Real-Life Example: Image Classification

In image classification tasks, overfitting can occur when the model memorizes specific training images instead of learning general features. Let's explore how data augmentation can help address this issue.

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Assume we have train and test directories with image data
train_generator = train_datagen.flow_from_directory(
    'train_dir', target_size=(224, 224), batch_size=32, class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    'test_dir', target_size=(224, 224), batch_size=32, class_mode='categorical')

history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=test_generator,
    validation_steps=50
)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

Slide 12: Real-Life Example: Natural Language Processing

In NLP tasks, overfitting can occur when the model learns specific patterns in the training text that don't generalize well. Let's explore how dropout and early stopping can help mitigate this issue in a sentiment analysis task.

Slide 13: Real-Life Example: Natural Language Processing

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Assume we have X_train, y_train, X_test, y_test as text data and labels

# Tokenize the text
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Build the model
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train_pad, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

Slide 14: Additional Resources

For further exploration of strategies to address overfitting in neural networks, consider the following resources:

1. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" by Srivastava et al. (2014) ArXiv: [https://arxiv.org/abs/1207.0580](https://arxiv.org/abs/1207.0580)
2. "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" by Ioffe and Szegedy (2015) ArXiv: [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
3. "Deep Learning" by Goodfellow, Bengio, and Courville (2016) Book website: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
4. "A Disciplined Approach to Neural Network Hyper-Parameters" by Leslie N. Smith (2018) ArXiv: [https://arxiv.org/abs/1803.09820](https://arxiv.org/abs/1803.09820)

These resources provide in-depth explanations and advanced techniques for addressing overfitting in various neural network architectures and applications.

