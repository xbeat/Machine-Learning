## Improving Unsatisfactory Model Performance with Python

Slide 1: Model Performance Evaluation

After training a machine learning model, it's crucial to evaluate its performance. If the metrics are unsatisfactory, there are several steps you can take to improve your model. Let's explore these strategies in detail.

```python
# Example of evaluating model performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = [0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
```

Output:

```
Accuracy: 0.67
Precision: 0.67
Recall: 0.67
F1 Score: 0.67
```

Slide 2: Data Quality Check

One of the first steps to improve model performance is to review your data quality. Ensure that your dataset is clean, balanced, and representative of the problem you're trying to solve.

```python
import pandas as pd

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Check class distribution (for classification problems)
print("\nClass distribution:")
print(df['target'].value_counts(normalize=True))

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")
```

Slide 3: Feature Engineering

Feature engineering can significantly impact model performance. Create new features or transform existing ones to capture more information from your data.

```python
import pandas as pd
import numpy as np

# Example of feature engineering
df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'income': [50000, 60000, 75000, 90000]
})

# Create a new feature
df['income_per_age'] = df['income'] / df['age']

# Bin a continuous feature
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 100], labels=['young', 'middle', 'senior'])

print(df)
```

Output:

```
   age  income  income_per_age age_group
0   25   50000          2000.0     young
1   30   60000          2000.0     young
2   35   75000          2142.9    middle
3   40   90000          2250.0    middle
```

Slide 4: Hyperparameter Tuning

Adjusting your model's hyperparameters can lead to significant improvements in performance. Use techniques like grid search or random search to find optimal hyperparameters.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Define the model and parameter grid
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

Slide 5: Cross-Validation

Implement proper cross-validation to ensure your model generalizes well to unseen data and to get a more robust estimate of its performance.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Create and train the model
rf = RandomForestClassifier(random_state=42)

# Perform 5-fold cross-validation
scores = cross_val_score(rf, X, y, cv=5)

print("Cross-validation scores:", scores)
print("Mean score:", scores.mean())
print("Standard deviation:", scores.std())
```

Slide 6: Ensemble Methods

Combine multiple models to create a more robust and accurate predictor. Techniques like bagging, boosting, and stacking can often improve performance.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# Create a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train individual models
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
lr = LogisticRegression(random_state=42)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Make predictions
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)
lr_pred = lr.predict(X_test)

# Ensemble prediction (simple majority voting)
ensemble_pred = (rf_pred + gb_pred + lr_pred) > 1.5

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_pred))
```

Slide 7: Regularization

Apply regularization techniques to prevent overfitting and improve model generalization. Common methods include L1 (Lasso) and L2 (Ridge) regularization.

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import numpy as np

# Create a sample dataset
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models with different regularization
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# Make predictions
ridge_pred = ridge.predict(X_test)
lasso_pred = lasso.predict(X_test)

print("Ridge MSE:", mean_squared_error(y_test, ridge_pred))
print("Lasso MSE:", mean_squared_error(y_test, lasso_pred))

print("Ridge coefficients:", np.sum(ridge.coef_ != 0))
print("Lasso coefficients:", np.sum(lasso.coef_ != 0))
```

Slide 8: Learning Curves

Analyze learning curves to diagnose whether your model is suffering from high bias or high variance. This can guide your next steps for improvement.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.datasets import load_digits

# Load a sample dataset
digits = load_digits()
X, y = digits.data, digits.target

# Calculate learning curves
train_sizes, train_scores, val_scores = learning_curve(
    SVC(kernel='rbf', gamma=0.001), X, y, train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5, scoring='accuracy'
)

# Calculate mean and standard deviation
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, val_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend()
plt.show()
```

Slide 9: Feature Selection

Identify and select the most important features to reduce noise and improve model performance. This can be especially helpful when dealing with high-dimensional data.

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_breast_cancer
import numpy as np

# Load a sample dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Perform feature selection
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)

# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)

# Print the names of the selected features
selected_features = [data.feature_names[i] for i in selected_feature_indices]
print("Selected features:", selected_features)

# Print the scores of the selected features
scores = selector.scores_
for feature, score in zip(selected_features, scores[selected_feature_indices]):
    print(f"{feature}: {score:.2f}")
```

Slide 10: Data Augmentation

For certain types of data, such as images, augmenting your dataset can help improve model performance by increasing the diversity of your training data.

```python
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Create a sample image (28x28 grayscale)
image = np.random.rand(28, 28, 1)

# Create an ImageDataGenerator with various augmentation options
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generate augmented images
augmented_images = [datagen.random_transform(image) for _ in range(9)]

# Plot the original and augmented images
plt.figure(figsize=(10, 10))
for i, aug_image in enumerate(augmented_images, 1):
    plt.subplot(3, 3, i)
    plt.imshow(aug_image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(f'Augmented {i}')
plt.tight_layout()
plt.show()
```

Slide 11: Transfer Learning

For complex tasks like image classification, using pre-trained models and fine-tuning them on your specific dataset can often lead to better performance than training from scratch.

```python
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten

# Load pre-trained VGG16 model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
```

Slide 12: Real-Life Example: Image Classification

Let's consider a real-life example of improving a convolutional neural network (CNN) for classifying images of different types of vehicles.

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# Define the initial model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')  # Assuming 5 vehicle classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# After training, if performance is unsatisfactory:

# 1. Add more layers and increase complexity
improved_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

# 2. Use a different optimizer and learning rate
improved_model.compile(optimizer=Adam(learning_rate=0.0001), 
                       loss='categorical_crossentropy', 
                       metrics=['accuracy'])

# 3. Implement data augmentation (as shown in a previous slide)

# 4. Use transfer learning (as shown in a previous slide)
```

Slide 13: Real-Life Example: Text Classification

Let's improve a text classification model for sentiment analysis of product reviews.

```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Sample data
texts = ["This product is amazing!", "Terrible experience, don't buy it", "Average quality, nothing special"]
labels = [1, 0, 0.5]  # 1 for positive, 0 for negative, 0.5 for neutral

# Tokenize the text
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=20, padding='post', truncating='post')

# Improved model
improved_model = Sequential([
    Embedding(1000, 32, input_length=20),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

improved_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model (assuming you have more data)
# improved_model.fit(padded, labels, epochs=10, validation_split=0.2)

# Further improvements:
# 1. Use pre-trained word embeddings (e.g., GloVe)
# 2. Implement data augmentation for text (e.g., synonym replacement)
# 3. Try different architectures (e.g., Transformer-based models)
# 4. Experiment with different optimizers and learning rates
```

Slide 14: Model Interpretability

Understanding why your model makes certain predictions can help you identify areas for improvement and build trust in your model.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_
feature_names = iris.feature_names

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

# Print feature importances
for f, imp in zip([feature_names[i] for i in indices], importances[indices]):
    print(f"{f}: {imp:.4f}")
```

Slide 15: Error Analysis

Carefully analyzing the errors your model makes can provide insights into where it's failing and guide your improvement efforts.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming X and y are your features and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Analyze misclassified samples
misclassified = X_test[y_test != y_pred]
misclassified_labels = y_test[y_test != y_pred]
predicted_labels = y_pred[y_test != y_pred]

for i in range(min(5, len(misclassified))):
    print(f"Sample {i+1}:")
    print(f"True label: {misclassified_labels[i]}")
    print(f"Predicted label: {predicted_labels[i]}")
    print(f"Features: {misclassified[i]}\n")
```

Slide 16: Additional Resources

For further reading on improving model performance, consider these resources:

1.  "Neural Network Architectures" by Christopher Olah (distill.pub/2019/computing-receptive-fields)
2.  "A Disciplined Approach to Neural Network Hyper-Parameters" by Leslie N. Smith (arxiv.org/abs/1803.09820)
3.  "Practical Recommendations for Gradient-Based Training of Deep Architectures" by Yoshua Bengio (arxiv.org/abs/1206.5533)

These papers provide in-depth discussions on various aspects of model improvement and optimization techniques.

