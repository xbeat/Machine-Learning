## Seven Most Used Machine Learning Algorithms

Slide 1: Seven Most Used Machine Learning Algorithms

Machine learning engineers rely on a set of core algorithms to solve various problems. This presentation will cover seven fundamental algorithms: Logistic Regression, Linear Regression, Decision Trees, Support Vector Machines, k-Nearest Neighbors, K-Means Clustering, and Random Forests. We'll explore each algorithm's key concepts, use cases, and provide practical Python code examples.

```python
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection, linear_model, tree, svm, neighbors, cluster, ensemble

# Load a sample dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)

# Plot the first two features
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='black')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Iris Dataset - First Two Features')
plt.show()
```

Slide 2: Logistic Regression

Logistic Regression is a statistical method for predicting binary outcomes. It's widely used in classification problems where the goal is to determine the probability of an instance belonging to a particular class.

```python
logistic = linear_model.LogisticRegression()
logistic.fit(X_train, y_train)

# Predict on test data
y_pred = logistic.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Logistic Regression Accuracy: {accuracy:.2f}")

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = logistic.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='black')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Logistic Regression Decision Boundary')
plt.show()
```

Slide 3: Linear Regression

Linear Regression is used to model the relationship between a dependent variable and one or more independent variables. It's commonly applied in predicting continuous values, such as house prices or sales forecasts.

```python
X_regression = X[:, np.newaxis, 0]  # Use only one feature for simplicity
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_regression, y, test_size=0.3, random_state=42)

linear = linear_model.LinearRegression()
linear.fit(X_train, y_train)

# Predict on test data
y_pred = linear.predict(X_test)

# Calculate mean squared error
mse = np.mean((y_pred - y_test) ** 2)
print(f"Linear Regression Mean Squared Error: {mse:.2f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Sepal length')
plt.ylabel('Target')
plt.title('Linear Regression')
plt.show()
```

Slide 4: Decision Trees

Decision Trees are versatile algorithms used for both classification and regression tasks. They work by creating a tree-like model of decisions based on features in the dataset.

```python
decision_tree = tree.DecisionTreeClassifier(max_depth=3)
decision_tree.fit(X_train, y_train)

# Predict on test data
y_pred = decision_tree.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Decision Tree Accuracy: {accuracy:.2f}")

# Visualize the tree
plt.figure(figsize=(20, 10))
tree.plot_tree(decision_tree, feature_names=iris.feature_names, 
               class_names=iris.target_names, filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
```

Slide 5: Support Vector Machines (SVM)

Support Vector Machines are powerful algorithms used for classification and regression tasks. They work by finding the hyperplane that best separates different classes in the feature space.

```python
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Predict on test data
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"SVM Accuracy: {accuracy:.2f}")

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='black')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM Decision Boundary')
plt.show()
```

Slide 6: k-Nearest Neighbors (k-NN)

k-Nearest Neighbors is a simple yet effective algorithm used for classification and regression. It works by finding the k nearest data points to a given query point and making predictions based on their values.

```python
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"k-NN Accuracy: {accuracy:.2f}")

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='black')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('k-NN Decision Boundary')
plt.show()
```

Slide 7: K-Means Clustering

K-Means Clustering is an unsupervised learning algorithm used to partition data into K distinct clusters. It's commonly used for customer segmentation, image compression, and anomaly detection.

```python
kmeans = cluster.KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Predict cluster labels
y_kmeans = kmeans.predict(X)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap=plt.cm.Set1, edgecolor='black')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('K-Means Clustering')
plt.show()
```

Slide 8: Random Forests

Random Forests are an ensemble learning method that combines multiple decision trees to create a more robust and accurate model. They're widely used for both classification and regression tasks.

```python
random_forest = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Predict on test data
y_pred = random_forest.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Random Forest Accuracy: {accuracy:.2f}")

# Feature importance
importances = random_forest.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [iris.feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()
```

Slide 9: Real-Life Example: Image Classification

In this example, we'll use a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes.

```python
from tensorflow.keras import datasets, layers, models

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc:.2f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 10: Real-Life Example: Sentiment Analysis

In this example, we'll use a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) cells to perform sentiment analysis on movie reviews from the IMDB dataset.

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load the IMDB dataset
max_features = 10000
maxlen = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to ensure uniform length
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_features, 128),
    tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.2f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Example prediction
def predict_sentiment(text):
    # Convert text to sequence of word indices
    word_index = imdb.get_word_index()
    text = text.lower().split()
    text = [word_index.get(word, 0) for word in text]
    text = sequence.pad_sequences([text], maxlen=maxlen)
    
    # Make prediction
    prediction = model.predict(text)[0][0]
    return "Positive" if prediction > 0.5 else "Negative", prediction

# Test the prediction function
sample_review = "This movie was fantastic! The acting was great and the plot kept me engaged throughout."
sentiment, score = predict_sentiment(sample_review)
print(f"Sentiment: {sentiment}")
print(f"Confidence: {score:.2f}")
```

Slide 11: Hyperparameter Tuning

Hyperparameter tuning is crucial for optimizing machine learning models. We'll use GridSearchCV to find the best hyperparameters for a Support Vector Machine classifier on the breast cancer dataset.

```python
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the breast cancer dataset and split it
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

# Create a base model
svm = SVC(random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluate on the test set
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test_scaled, y_test)
print("Test set score:", test_score)
```

Slide 12: Feature Engineering

Feature engineering is the process of creating new features or transforming existing ones to improve model performance. We'll demonstrate this using the California Housing dataset.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the California Housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a function for feature engineering
def engineer_features(X):
    # Create new features
    X_new = np.column_stack((
        X,
        X[:, 0] * X[:, 1],  # Interaction between median income and average house age
        np.log1p(X[:, 0]),  # Log transform of median income
        X[:, 2] ** 2        # Square of average rooms
    ))
    return X_new

# Apply feature engineering
X_train_eng = engineer_features(X_train)
X_test_eng = engineer_features(X_test)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_eng)
X_test_scaled = scaler.transform(X_test_eng)

# Train and evaluate models
model_original = LinearRegression().fit(X_train, y_train)
model_engineered = LinearRegression().fit(X_train_scaled, y_train)

print("Original features R^2 score:", model_original.score(X_test, y_test))
print("Engineered features R^2 score:", model_engineered.score(X_test_scaled, y_test))
```

Slide 13: Ensemble Methods

Ensemble methods combine multiple models to create a more robust and accurate predictor. We'll implement a simple voting classifier using different base models.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create base models
log_clf = LogisticRegression(random_state=42)
tree_clf = DecisionTreeClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)

# Create and train the voting classifier
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('dt', tree_clf), ('svc', svm_clf)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# Make predictions
y_pred = voting_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Voting Classifier Accuracy: {accuracy:.4f}")

# Compare with individual model performances
for clf in (log_clf, tree_clf, svm_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{clf.__class__.__name__} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

Slide 14: Model Interpretation

Understanding model decisions is crucial in many applications. We'll use SHAP (SHapley Additive exPlanations) to interpret a Random Forest model.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

# Load Boston Housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Explain the model's predictions using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualize feature importances
shap.summary_plot(shap_values, X, plot_type="bar", feature_names=boston.feature_names)
plt.title("Feature Importances (SHAP values)")
plt.tight_layout()
plt.show()

# Visualize SHAP values for a single prediction
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X[0,:], feature_names=boston.feature_names)
```

Slide 15: Additional Resources

For further exploration of machine learning algorithms and techniques, consider the following resources:

1. "Machine Learning" course by Andrew Ng on Coursera
2. "Deep Learning" book by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
3. scikit-learn documentation: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
4. TensorFlow tutorials: [https://www.tensorflow.org/tutorials](https://www.tensorflow.org/tutorials)
5. "Pattern Recognition and Machine Learning" by Christopher Bishop
6. arXiv.org for the latest research papers in machine learning: [https://arxiv.org/list/cs.LG/recent](https://arxiv.org/list/cs.LG/recent)

These resources provide a mix of theoretical foundations and practical implementations to deepen your understanding of machine learning algorithms and their applications.


