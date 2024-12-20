## Discriminative Models in Python
Slide 1: Introduction to Discriminative Models

Discriminative models are a class of machine learning algorithms that learn to distinguish between different classes or categories. These models focus on predicting the conditional probability of a target variable given input features. Unlike generative models, discriminative models don't attempt to model the underlying distribution of the data.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Example of a simple discriminative model: Logistic Regression
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# Predict the probability of class 1 for a new point
new_point = np.array([[2.5, 3.5]])
probability = model.predict_proba(new_point)[0][1]
print(f"Probability of class 1: {probability:.2f}")
```

Slide 2: Linear Discriminant Analysis (LDA)

Linear Discriminant Analysis is a dimensionality reduction technique that also serves as a discriminative model. It aims to find a linear combination of features that best separates two or more classes. LDA assumes that the data for each class is normally distributed with equal covariance matrices.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_redundant=0, random_state=42)

# Create and fit the LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

# Transform the data
X_transformed = lda.transform(X)

print(f"Original shape: {X.shape}")
print(f"Transformed shape: {X_transformed.shape}")
```

Slide 3: Support Vector Machines (SVM)

Support Vector Machines are powerful discriminative models that find the optimal hyperplane to separate classes in high-dimensional space. SVMs can handle both linear and non-linear classification tasks by using different kernel functions.

```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate a non-linearly separable dataset
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# Create and fit the SVM model
svm = SVC(kernel='rbf', C=1.0, random_state=42)
svm.fit(X, y)

# Plot the decision boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                     np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.title("SVM Decision Boundary")
plt.show()
```

Slide 4: Decision Trees

Decision Trees are versatile discriminative models that make decisions based on a series of questions about the input features. They can handle both classification and regression tasks and are easy to interpret.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create and fit the Decision Tree model
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

# Visualize the tree
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree for Iris Dataset")
plt.show()
```

Slide 5: Random Forests

Random Forests are an ensemble of decision trees, combining their predictions to create a more robust and accurate model. They reduce overfitting by introducing randomness in the tree-building process and feature selection.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
accuracy = rf.score(X_test, y_test)
print(f"Random Forest Accuracy: {accuracy:.2f}")
```

Slide 6: Gradient Boosting Machines

Gradient Boosting Machines (GBM) are another ensemble method that builds a series of weak learners (typically decision trees) sequentially. Each new model tries to correct the errors made by the previous models, resulting in a powerful discriminative model.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Gradient Boosting model
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)

# Evaluate the model
accuracy = gb.score(X_test, y_test)
print(f"Gradient Boosting Accuracy: {accuracy:.2f}")
```

Slide 7: Neural Networks

Neural Networks are versatile discriminative models inspired by the human brain. They can learn complex non-linear relationships between inputs and outputs, making them suitable for a wide range of tasks, including image and speech recognition.

```python
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Generate a sample dataset
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")
```

Slide 8: K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a simple yet effective discriminative model that classifies new data points based on the majority class of their k nearest neighbors in the feature space. It's non-parametric and doesn't make assumptions about the underlying data distribution.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate the model
accuracy = knn.score(X_test, y_test)
print(f"KNN Accuracy: {accuracy:.2f}")

# Predict a new sample
new_sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = knn.predict(new_sample)
print(f"Predicted class: {iris.target_names[prediction[0]]}")
```

Slide 9: Naive Bayes

Naive Bayes is a probabilistic classifier based on Bayes' theorem with the "naive" assumption of feature independence. Despite its simplicity, it often performs well, especially in text classification tasks.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Naive Bayes model
nb = GaussianNB()
nb.fit(X_train, y_train)

# Evaluate the model
accuracy = nb.score(X_test, y_test)
print(f"Naive Bayes Accuracy: {accuracy:.2f}")

# Predict probabilities for a new sample
new_sample = [[5.1, 3.5, 1.4, 0.2]]
probabilities = nb.predict_proba(new_sample)
print(f"Class probabilities: {probabilities[0]}")
```

Slide 10: Conditional Random Fields (CRF)

Conditional Random Fields are discriminative models used for structured prediction tasks, such as sequence labeling. They're particularly useful in natural language processing for tasks like named entity recognition and part-of-speech tagging.

```python
from sklearn_crfsuite import CRF
import nltk
from nltk.corpus import conll2002

# Download the CoNLL 2002 dataset
nltk.download('conll2002')

# Prepare the data
def word2features(sent, i):
    word = sent[i][0]
    return {
        'word': word,
        'is_first': i == 0,
        'is_last': i == len(sent) - 1,
        'is_capitalized': word[0].upper() == word[0],
        'is_all_caps': word.upper() == word,
        'is_all_lower': word.lower() == word,
        'prefix-1': word[0],
        'prefix-2': word[:2],
        'suffix-1': word[-1],
        'suffix-2': word[-2:],
        'prev_word': '' if i == 0 else sent[i-1][0],
        'next_word': '' if i == len(sent)-1 else sent[i+1][0],
        'has_hyphen': '-' in word,
        'is_numeric': word.isdigit(),
    }

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

# Load the data
train_sents = list(conll2002.iob_sents('esp.train'))
test_sents = list(conll2002.iob_sents('esp.testb'))

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

# Train the CRF model
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# Evaluate the model
y_pred = crf.predict(X_test)
accuracy = crf.score(X_test, y_test)
print(f"CRF Accuracy: {accuracy:.2f}")
```

Slide 11: Real-life Example: Spam Detection

Spam detection is a common application of discriminative models. We'll use a Naive Bayes classifier to distinguish between spam and non-spam emails based on their content.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Sample email data
emails = [
    ("Free gift! Click now!", "spam"),
    ("Meeting at 3 PM today", "ham"),
    ("Win a new iPhone!", "spam"),
    ("Project deadline reminder", "ham"),
    ("Congratulations! You've won!", "spam"),
    ("Lunch with the team tomorrow", "ham")
]

X, y = zip(*emails)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a bag of words representation
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Evaluate the model
accuracy = clf.score(X_test_vectorized, y_test)
print(f"Spam Detection Accuracy: {accuracy:.2f}")

# Classify a new email
new_email = ["Hey, how are you doing?"]
new_email_vectorized = vectorizer.transform(new_email)
prediction = clf.predict(new_email_vectorized)
print(f"New email classification: {prediction[0]}")
```

Slide 12: Real-life Example: Image Classification

Image classification is a common application of discriminative models. We'll use a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes.

```python
import tensorflow as tf
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
model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')
```

Slide 13: Comparison of Discriminative Models

Discriminative models have different strengths and weaknesses. Here's a brief comparison:

1. Logistic Regression: Simple, interpretable, but limited to linear decision boundaries.
2. SVM: Effective in high-dimensional spaces, but can be slow for large datasets.
3. Decision Trees: Easily interpretable, but prone to overfitting.
4. Random Forests: Robust to overfitting, but less interpretable than single trees.
5. Neural Networks: Highly flexible, but require large amounts of data and computational resources.
6. KNN: Simple and effective, but slow for large datasets and sensitive to irrelevant features.
7. Naive Bayes: Fast and efficient, especially for text classification, but assumes feature independence.

The choice of model depends on the specific problem, dataset characteristics, and computational constraints.

Slide 14: Challenges and Considerations

When working with discriminative models, consider the following:

1. Feature selection: Choose relevant features to improve model performance and reduce overfitting.
2. Hyperparameter tuning: Use techniques like cross-validation to optimize model parameters.
3. Handling imbalanced data: Apply techniques such as oversampling, undersampling, or using weighted loss functions.
4. Interpretability: Some models (e.g., decision trees) are more interpretable than others (e.g., neural networks).
5. Scalability: Consider computational requirements for large datasets or real-time predictions.
6. Generalization: Ensure the model performs well on unseen data by using proper validation techniques.

Addressing these challenges is crucial for developing effective and robust discriminative models in real-world applications.

Slide 15: Additional Resources

For further exploration of discriminative models, consider the following resources:

1. "Pattern Recognition and Machine Learning" by Christopher M. Bishop (2006) ArXiv: [https://arxiv.org/abs/2106.08903](https://arxiv.org/abs/2106.08903) (review of the book)
2. "The Elements of Statistical Learning" by Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009) ArXiv: [https://arxiv.org/abs/2103.05622](https://arxiv.org/abs/2103.05622) (review of the book)
3. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016) ArXiv: [https://arxiv.org/abs/1601.06615](https://arxiv.org/abs/1601.06615) (review of the book)
4. "Support Vector Machines and Kernel Methods: The New Generation of Learning Machines" by Bernhard Schölkopf and Alexander J. Smola ArXiv: [https://arxiv.org/abs/1902.03703](https://arxiv.org/abs/1902.03703)
5. "Random Forests" by Leo Breiman (2001) ArXiv: [https://arxiv.org/abs/1501.07788](https://arxiv.org/abs/1501.07788) (review of random forests)

These resources provide in-depth coverage of various discriminative models and their applications in machine learning and pattern recognition.

