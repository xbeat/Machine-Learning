## Top 10 Python AI Libraries and Their Uses

Slide 1: Top 10 Python Libraries for AI

The field of Artificial Intelligence (AI) has been revolutionized by Python libraries that simplify complex tasks and accelerate development. This presentation will explore the top 10 Python libraries for AI, focusing on their key features and practical applications. We'll dive into code examples to demonstrate how these libraries can be used in real-world scenarios.

Slide 2: TensorFlow

TensorFlow is an open-source library for numerical computation and large-scale machine learning. Developed by Google Brain team, it offers a flexible ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML-powered applications.

Slide 3: Source Code for TensorFlow

```python
import tensorflow as tf

# Create a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate some dummy data
import numpy as np
x_train = np.random.random((1000, 10))
y_train = np.random.randint(2, size=(1000, 1))

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Make predictions
x_test = np.random.random((100, 10))
predictions = model.predict(x_test)
print(predictions[:5])
```

Slide 4: PyTorch

PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It provides a seamless path from research prototyping to production deployment. PyTorch is known for its dynamic computational graphs and imperative programming style, which allows for more intuitive debugging and development.

Slide 5: Source Code for PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Create the model, loss function, and optimizer
model = SimpleNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Generate some dummy data
x_train = torch.randn(1000, 10)
y_train = torch.randint(0, 2, (1000, 1)).float()

# Train the model
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Make predictions
x_test = torch.randn(100, 10)
with torch.no_grad():
    predictions = model(x_test)
print(predictions[:5])
```

Slide 6: Scikit-learn

Scikit-learn is a machine learning library for Python that provides a wide range of supervised and unsupervised learning algorithms. It's built on NumPy, SciPy, and matplotlib, making it easy to integrate into data science workflows. Scikit-learn is known for its consistent API, extensive documentation, and focus on ease of use.

Slide 7: Source Code for Scikit-learn

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate a random binary classification problem
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Feature importance
feature_importance = rf_classifier.feature_importances_
for i, importance in enumerate(feature_importance):
    print(f"Feature {i+1} importance: {importance:.4f}")
```

Slide 8: Keras

Keras is a high-level neural networks API that can run on top of TensorFlow, Theano, or CNTK. It was developed with a focus on enabling fast experimentation and ease of use. Keras allows for easy and fast prototyping through its user-friendly, modular, and extensible design.

Slide 9: Source Code for Keras

```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Create a sequential model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate some dummy data
X_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
X_test = np.random.random((200, 20))
y_test = np.random.randint(2, size=(200, 1))

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Make predictions
predictions = model.predict(X_test[:5])
print("Predictions for first 5 test samples:")
print(predictions)
```

Slide 10: NumPy

NumPy is the fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently. NumPy is the foundation for many other scientific and machine learning libraries in Python.

Slide 11: Source Code for NumPy

```python
import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original array:")
print(arr)

# Basic operations
print("\nSum of all elements:", np.sum(arr))
print("Mean of all elements:", np.mean(arr))
print("Standard deviation:", np.std(arr))

# Array manipulation
print("\nTransposed array:")
print(arr.T)

# Matrix multiplication
arr2 = np.array([[2, 0, 1], [1, 2, 1], [1, 1, 0]])
print("\nMatrix multiplication:")
print(np.dot(arr, arr2))

# Element-wise operations
print("\nElement-wise square root:")
print(np.sqrt(arr))

# Broadcasting
print("\nAdding a vector to each row of the matrix:")
vector = np.array([10, 20, 30])
print(arr + vector)

# Random number generation
print("\nRandom 3x3 matrix with values between 0 and 1:")
print(np.random.rand(3, 3))
```

Slide 12: Pandas

Pandas is a fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation tool. It provides data structures like DataFrames for efficiently handling large datasets and tools for reading and writing data between in-memory data structures and different file formats.

Slide 13: Source Code for Pandas

```python
import pandas as pd
import numpy as np

# Create a DataFrame
df = pd.DataFrame({
    'A': np.random.randn(5),
    'B': np.random.randn(5),
    'C': np.random.randn(5),
    'D': np.random.choice(['X', 'Y', 'Z'], 5)
})

print("Original DataFrame:")
print(df)

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Filtering
print("\nFiltering rows where A > 0:")
print(df[df['A'] > 0])

# Grouping and aggregation
print("\nMean values grouped by D:")
print(df.groupby('D').mean())

# Adding a new column
df['E'] = df['A'] + df['B']
print("\nDataFrame with new column E:")
print(df)

# Handling missing values
df.loc[2, 'B'] = np.nan
print("\nDataFrame with a missing value:")
print(df)
print("\nDropping rows with missing values:")
print(df.dropna())

# Reading from and writing to CSV
df.to_csv('example.csv', index=False)
read_df = pd.read_csv('example.csv')
print("\nDataFrame read from CSV:")
print(read_df)
```

Slide 14: NLTK (Natural Language Toolkit)

NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.

Slide 15: Source Code for NLTK

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from nltk import FreqDist

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Sample text
text = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."

# Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens[:10])

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("\nTokens without stopwords:", filtered_tokens[:10])

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print("\nStemmed tokens:", stemmed_tokens[:10])

# Part-of-speech tagging
pos_tags = pos_tag(tokens)
print("\nPOS tags:", pos_tags[:10])

# Frequency distribution
fdist = FreqDist(filtered_tokens)
print("\nMost common words:")
print(fdist.most_common(5))

# Concordance (requires downloading the text)
# nltk.download('inaugural')
# from nltk.text import Text
# inaugural_text = Text(nltk.corpus.inaugural.words())
# print("\nConcordance for 'liberty':")
# inaugural_text.concordance("liberty", lines=5)
```

Slide 16: OpenCV

OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in commercial products. OpenCV has more than 2500 optimized algorithms for real-time computer vision tasks.

Slide 17: Source Code for OpenCV

```python
import cv2
import numpy as np

# Read an image
img = cv2.imread('example_image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Face detection using Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read an image with faces
img = cv2.imread('faces_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the result
cv2.imshow('Faces Detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Slide 18: Matplotlib

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It provides a MATLAB-like interface for creating plots, histograms, power spectra, bar charts, errorcharts, scatterplots, etc., with just a few lines of code.

Slide 19: Source Code for Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a figure and axis objects
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot sine wave
ax1.plot(x, y1, label='Sine')
ax1.set_title('Sine Wave')
ax1.set_xlabel('x')
ax1.set_ylabel('sin(x)')
ax1.legend()
ax1.grid(True)

# Plot cosine wave
ax2.plot(x, y2, label='Cosine', color='red')
ax2.set_title('Cosine Wave')
ax2.set_xlabel('x')
ax2.set_ylabel('cos(x)')
ax2.legend()
ax2.grid(True)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# Create a scatter plot
fig, ax = plt.subplots(figsize=(8, 6))

# Generate random data
n = 50
x = np.random.rand(n)
y = np.random.rand(n)
colors = np.random.rand(n)
sizes = 1000 * np.random.rand(n)

# Create scatter plot
scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='viridis')

# Add colorbar
plt.colorbar(scatter)

# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Scatter Plot with Color and Size Variation')

# Display the plot
plt.show()
```

Slide 20: Real-Life Example: Sentiment Analysis

Sentiment analysis is a common application of AI in social media monitoring and customer feedback analysis. We'll use NLTK and scikit-learn to perform sentiment analysis on movie reviews.

Slide 21: Source Code for Sentiment Analysis

```python
import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')

# Prepare the data
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
import random
random.shuffle(documents)

# Preprocess the text
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(' '.join(text))
    return ' '.join([word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words])

# Split into features and labels
X = [preprocess(doc) for doc, category in documents]
y = [category for doc, category in documents]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = clf.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
```

Slide 22: Real-Life Example: Image Classification

Image classification is widely used in various applications, from facial recognition to medical imaging. We'll use a pre-trained CNN model to classify images.

Slide 23: Source Code for Image Classification

```python
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Function to preprocess and predict image
def classify_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make prediction
    preds = model.predict(x)
    
    # Decode and print predictions
    print('Predicted:', decode_predictions(preds, top=3)[0])

# Example usage
classify_image('example_image.jpg')
```

Slide 24: Additional Resources

For more in-depth information on AI and machine learning libraries, consider exploring these resources:

1.  TensorFlow Documentation: [https://www.tensorflow.org/docs](https://www.tensorflow.org/docs)
2.  PyTorch Tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
3.  Scikit-learn User Guide: [https://scikit-learn.org/stable/user\_guide.html](https://scikit-learn.org/stable/user_guide.html)
4.  Keras Documentation: [https://keras.io/docs/](https://keras.io/docs/)
5.  NLTK Book: [https://www.nltk.org/book/](https://www.nltk.org/book/)
6.  OpenCV Tutorials: [https://docs.opencv.org/master/d9/df8/tutorial\_root.html](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
7.  Matplotlib Tutorials: [https://matplotlib.org/stable/tutorials/index.html](https://matplotlib.org/stable/tutorials/index.html)

For academic papers on AI and machine learning, visit ArXiv.org and search for topics of interest. For example:

*   "Attention Is All You Need" by Vaswani et al. (2017): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   "Deep Residual Learning for Image Recognition" by He et al. (2015): [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

