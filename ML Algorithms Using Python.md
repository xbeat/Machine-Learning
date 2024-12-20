## ML Algorithms Using Python
Slide 1: Linear Regression

Linear Regression is a fundamental algorithm in machine learning used for predicting continuous numerical values based on input features. It models the relationship between variables by fitting a linear equation to observed data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict values
X_test = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y_pred = model.predict(X_test)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', label='Predicted line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
```

Slide 2: Natural Language Processing (NLP) Models

NLP models like BERT and GPT are designed to understand and generate human language. These models have revolutionized tasks such as language translation, sentiment analysis, and chatbots.

```python
from transformers import pipeline

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Analyze sentiment of a given text
text = "I love learning about machine learning algorithms!"
result = sentiment_analyzer(text)

print(f"Text: {text}")
print(f"Sentiment: {result[0]['label']}")
print(f"Confidence: {result[0]['score']:.2f}")
```

Slide 3: Long Short-Term Memory (LSTM)

LSTM is a type of recurrent neural network architecture designed to handle sequence data where long-term dependencies are important. It's commonly used in tasks like language translation and speech synthesis.

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate sample sequence data
X = np.array([[i/100] for i in range(100)])
y = np.sin(X * 2 * np.pi)

# Create and compile the model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Reshape input for LSTM [samples, time steps, features]
X = X.reshape((X.shape[0], 1, 1))

# Train the model
model.fit(X, y, epochs=100, verbose=0)

# Make predictions
X_test = np.array([[i/100] for i in range(100, 200)])
X_test = X_test.reshape((X_test.shape[0], 1, 1))
predictions = model.predict(X_test)

print("First 5 predictions:")
print(predictions[:5].flatten())
```

Slide 4: Logistic Regression

Logistic Regression is used for binary classification problems. It's commonly applied in spam email detection, customer churn prediction, and medical diagnosis.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Generate sample data
X = np.random.randn(1000, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Example prediction
new_data = np.array([[0.5, 0.5]])
prediction = model.predict(new_data)
print(f"Prediction for [0.5, 0.5]: {prediction[0]}")
```

Slide 5: K-Means Clustering

K-Means is an unsupervised learning algorithm used for clustering data into groups based on similarity. It's often applied in customer segmentation and image compression.

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2)

# Create and fit the model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.title('K-Means Clustering')
plt.show()

print("Cluster centers:")
print(centers)
```

Slide 6: Gradient Boosting Algorithms

Gradient Boosting algorithms like XGBoost and LightGBM are powerful ensemble methods used for regression and classification tasks. They're known for their high performance in various applications.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions
y_pred = gb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Feature importance
feature_importance = gb_model.feature_importances_
print("Top 5 important features:")
for i in range(5):
    print(f"Feature {i}: {feature_importance[i]:.4f}")
```

Slide 7: Decision Trees

Decision Trees are versatile algorithms used for both classification and regression tasks. They're particularly useful in healthcare for disease diagnosis and in finance for fraud detection.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create and train the model
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X, y)

# Visualize the tree
plt.figure(figsize=(20,10))
plot_tree(tree_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

# Make a prediction
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example iris flower measurements
prediction = tree_model.predict(sample)
print(f"Predicted class: {iris.target_names[prediction[0]]}")
```

Slide 8: Hierarchical Clustering

Hierarchical Clustering is an algorithm that builds a hierarchy of clusters, often used in biology for taxonomy and in business for market segmentation.

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(50, 2)

# Perform hierarchical clustering
linked = linkage(X, 'ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Get cluster assignments
from scipy.cluster.hierarchy import fcluster
cluster_labels = fcluster(linked, t=1.5, criterion='distance')
print("Cluster assignments:")
print(cluster_labels)
```

Slide 9: Reinforcement Learning Algorithms

Reinforcement Learning algorithms, such as Q-Learning and Deep Q-Networks, are used in game playing, robotics, and recommendation systems. They learn optimal actions through trial and error.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple Q-Learning implementation
class QLearning:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.Q = np.zeros((states, actions))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.Q.shape[1])  # Explore
        else:
            return np.argmax(self.Q[state, :])  # Exploit

    def learn(self, state, action, reward, next_state):
        predict = self.Q[state, action]
        target = reward + self.gamma * np.max(self.Q[next_state, :])
        self.Q[state, action] += self.alpha * (target - predict)

# Example usage
q_learner = QLearning(states=5, actions=3)
rewards = []

for episode in range(1000):
    state = 0
    total_reward = 0
    
    for _ in range(100):  # 100 steps per episode
        action = q_learner.choose_action(state)
        next_state = np.random.randint(5)  # Simplified environment
        reward = np.random.randn()  # Random reward
        q_learner.learn(state, action, reward, next_state)
        
        state = next_state
        total_reward += reward
    
    rewards.append(total_reward)

plt.plot(rewards)
plt.title('Q-Learning Rewards Over Time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

print("Final Q-Table:")
print(q_learner.Q)
```

Slide 10: Random Forest

Random Forest is an ensemble learning method that constructs multiple decision trees and merges them for more accurate and stable predictions. It's widely used for classification and regression tasks.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Generate a random dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(20), importances[indices])
plt.xticks(range(20), indices)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()
```

Slide 11: Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique used to simplify complex datasets while retaining important information. It's commonly applied in data visualization and feature extraction.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the results
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i, c in zip(range(3), colors):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], c=c, label=iris.target_names[i])

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.title('PCA of Iris Dataset')
plt.show()

# Print explained variance ratio
print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
```

Slide 12: Word Embeddings

Word Embeddings are dense vector representations of words that capture semantic relationships. They're crucial for many NLP tasks, including sentiment analysis and document clustering.

```python
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

# Sample sentences
sentences = [
    ['I', 'love', 'machine', 'learning'],
    ['AI', 'is', 'fascinating'],
    ['Python', 'is', 'great', 'for', 'data', 'science'],
    ['Neural', 'networks', 'are', 'powerful']
]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Function to plot word vectors
def plot_words(model, words):
    vectors = [model.wv[word] for word in words]
    x = [v[0] for v in vectors]
    y = [v[1] for v in vectors]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y)
    
    for i, word in enumerate(words):
        plt.annotate(word, xy=(x[i], y[i]))
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Word Embeddings Visualization')
    plt.show()

# Plot some words
words_to_plot = ['machine', 'learning', 'AI', 'Python', 'data', 'science', 'neural']
plot_words(model, words_to_plot)

# Find similar words
print("Words similar to 'learning':")
print(model.wv.most_similar('learning', topn=3))
```

Slide 13: Support Vector Machines (SVM)

SVMs are powerful classifiers that find the optimal hyperplane to separate different classes. They're effective for both linear and non-linear classification tasks, as well as anomaly detection.

```python
from sklearn import svm
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
svm_model = svm.SVC(kernel='rbf', C=1)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualize the decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("SVM Decision Boundary")
    plt.show()

plot_decision_boundary(svm_model, X, y)
```

Slide 14: Neural Networks (Deep Learning)

Neural Networks, especially deep learning models, have revolutionized various domains including image and speech recognition, natural language processing, and more.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.2f}")

# Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

Slide 15: Autoencoders

Autoencoders are neural networks used for unsupervised learning of efficient data codings. They're particularly useful for anomaly detection, data compression, and feature learning.

```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = np.random.rand(1000, 10)

# Define the autoencoder architecture
input_dim = 10
encoding_dim = 3

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = Model(input_layer, decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(data, data, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Plot the loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Use the encoder to get the compressed representation
encoder = Model(input_layer, encoder)
encoded_data = encoder.predict(data)

# Visualize the compressed data (first 2 dimensions)
plt.scatter(encoded_data[:, 0], encoded_data[:, 1])
plt.title('2D Visualization of Encoded Data')
plt.xlabel('Encoded Dimension 1')
plt.ylabel('Encoded Dimension 2')
plt.show()
```

Slide 16: Additional Resources

For those interested in diving deeper into machine learning algorithms and their applications, here are some valuable resources:

1. ArXiv.org: A repository of electronic preprints of scientific papers, including many on machine learning and AI. URL: [https://arxiv.org/list/cs.LG/recent](https://arxiv.org/list/cs.LG/recent)
2. Machine Learning Mastery: A blog with practical tutorials and examples. URL: [https://machinelearningmastery.com/](https://machinelearningmastery.com/)
3. Towards Data Science: A Medium publication with articles on data science and machine learning. URL: [https://towardsdatascience.com/](https://towardsdatascience.com/)
4. Scikit-learn Documentation: Comprehensive guide to using scikit-learn for machine learning in Python. URL: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
5. TensorFlow Tutorials: Official tutorials for deep learning with TensorFlow. URL: [https://www.tensorflow.org/tutorials](https://www.tensorflow.org/tutorials)

Remember to verify the accuracy and relevance of information from these sources, as the field of machine learning is rapidly evolving.

