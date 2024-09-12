## Hashing Trick for Machine Learning in Python
Slide 1: The Hashing Trick in Machine Learning

The hashing trick, also known as feature hashing, is a technique used in machine learning to reduce the dimensionality of feature vectors. It's particularly useful when dealing with high-dimensional sparse data, such as text processing or large-scale online learning. By mapping input features to a fixed-size vector using a hash function, we can efficiently handle large feature spaces without explicitly storing feature names.

```python
import hashlib

def hash_feature(feature, n_buckets):
    # Use MD5 hash function to map the feature to a bucket
    hash_value = hashlib.md5(feature.encode()).hexdigest()
    # Convert the hexadecimal hash to an integer and take modulo
    return int(hash_value, 16) % n_buckets

# Example usage
feature = "example_feature"
n_buckets = 1000
bucket = hash_feature(feature, n_buckets)
print(f"Feature '{feature}' is mapped to bucket {bucket}")
```

Slide 2: Basic Concept of Feature Hashing

Feature hashing works by applying a hash function to the input features and using the hash values as indices in a fixed-size vector. This process effectively reduces the dimensionality of the feature space while preserving most of the information. The technique is particularly useful when the number of potential features is very large or unknown, as it allows us to work with a fixed-size feature vector regardless of the input size.

```python
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# Sample text data
texts = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Create a HashingVectorizer with 8 features
vectorizer = HashingVectorizer(n_features=8, alternate_sign=False)

# Transform the text data into feature vectors
X = vectorizer.transform(texts)

print("Feature matrix shape:", X.shape)
print("Feature matrix:")
print(X.toarray())
```

Slide 3: Advantages of the Hashing Trick

The hashing trick offers several advantages in machine learning applications. Firstly, it provides a constant-time operation for both training and prediction, making it highly efficient for large-scale problems. Secondly, it requires minimal memory usage since we don't need to store a dictionary of features. Lastly, it allows for online learning scenarios where new features can be incorporated on-the-fly without modifying the existing model.

```python
import time
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer

# Generate a large number of unique words
words = [f"word_{i}" for i in range(1000000)]
text = " ".join(words)

# Measure time for HashingVectorizer
start_time = time.time()
hashing_vectorizer = HashingVectorizer(n_features=1000)
hashing_vectorizer.transform([text])
hashing_time = time.time() - start_time

# Measure time for CountVectorizer
start_time = time.time()
count_vectorizer = CountVectorizer()
count_vectorizer.fit([text])
count_vectorizer.transform([text])
count_time = time.time() - start_time

print(f"HashingVectorizer time: {hashing_time:.4f} seconds")
print(f"CountVectorizer time: {count_time:.4f} seconds")
```

Slide 4: Collision Handling in Feature Hashing

One challenge with feature hashing is the possibility of collisions, where different features are mapped to the same bucket. While this can lead to some loss of information, the impact is often minimal in practice, especially with a sufficiently large number of buckets. To mitigate the effects of collisions, we can use signed hashing, where features are assigned a random sign (+1 or -1) based on a secondary hash function.

```python
import numpy as np

def signed_hash(feature, n_buckets):
    # Primary hash for bucket index
    bucket = hash(feature) % n_buckets
    # Secondary hash for sign
    sign = 1 if hash(feature + "_sign") % 2 == 0 else -1
    return bucket, sign

def feature_vector(features, n_buckets):
    vector = np.zeros(n_buckets)
    for feature in features:
        bucket, sign = signed_hash(feature, n_buckets)
        vector[bucket] += sign
    return vector

# Example usage
features = ["apple", "banana", "cherry", "date"]
n_buckets = 10
result = feature_vector(features, n_buckets)
print("Feature vector:", result)
```

Slide 5: Implementing a Simple Text Classifier with Feature Hashing

Let's implement a basic text classifier using the hashing trick. We'll use the HashingVectorizer from scikit-learn to transform our text data and a LogisticRegression model for classification. This example demonstrates how feature hashing can be applied to a real-world task like sentiment analysis.

```python
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data: movie reviews with sentiment labels
reviews = [
    "This movie was great!", "Terrible film, don't watch it",
    "I loved every minute of it", "Boring and predictable",
    "A masterpiece of cinema", "Waste of time and money"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# Create a HashingVectorizer
vectorizer = HashingVectorizer(n_features=1000)

# Transform the text data
X_train_hashed = vectorizer.transform(X_train)
X_test_hashed = vectorizer.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_hashed, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_hashed)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Predict sentiment for a new review
new_review = ["This movie exceeded all my expectations!"]
new_review_hashed = vectorizer.transform(new_review)
prediction = model.predict(new_review_hashed)
print(f"Predicted sentiment: {'Positive' if prediction[0] == 1 else 'Negative'}")
```

Slide 6: Feature Hashing for Large-Scale Learning

Feature hashing is particularly useful in large-scale machine learning scenarios where the feature space is enormous. It allows us to handle millions of features efficiently without explicitly storing them. Let's demonstrate this by creating a simple online learning algorithm that uses feature hashing to classify streaming text data.

```python
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

class OnlineClassifier:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = np.zeros(n_features)
        self.learning_rate = learning_rate
        self.vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False)
    
    def predict(self, text):
        features = self.vectorizer.transform([text]).toarray()[0]
        return 1 if np.dot(self.weights, features) > 0 else 0
    
    def update(self, text, true_label):
        features = self.vectorizer.transform([text]).toarray()[0]
        predicted = self.predict(text)
        self.weights += self.learning_rate * (true_label - predicted) * features

# Example usage
classifier = OnlineClassifier(n_features=1000)

# Simulate streaming data
stream = [
    ("The product works great", 1),
    ("Terrible customer service", 0),
    ("Highly recommended", 1),
    ("Doesn't work as advertised", 0)
]

for text, label in stream:
    prediction = classifier.predict(text)
    print(f"Text: '{text}', Predicted: {prediction}, Actual: {label}")
    classifier.update(text, label)

# Test on a new example
new_text = "Amazing product, exceeded expectations"
prediction = classifier.predict(new_text)
print(f"New text: '{new_text}', Predicted: {prediction}")
```

Slide 7: Handling Categorical Variables with Feature Hashing

Feature hashing is not limited to text data; it can also be used to handle categorical variables efficiently. This is particularly useful when dealing with high-cardinality categorical features. Let's implement a simple example that demonstrates how to use feature hashing for encoding categorical variables in a dataset.

```python
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset with categorical variables
data = [
    {"color": "red", "shape": "circle", "size": "large"},
    {"color": "blue", "shape": "square", "size": "small"},
    {"color": "green", "shape": "triangle", "size": "medium"},
    {"color": "red", "shape": "square", "size": "small"},
    {"color": "blue", "shape": "circle", "size": "large"},
]
labels = [0, 1, 1, 0, 1]

# Convert dictionary items to list of tuples
data = [[(key, value) for key, value in item.items()] for item in data]

# Create a FeatureHasher
hasher = FeatureHasher(n_features=10, input_type="pair")

# Transform the data
X = hasher.transform(data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Predict for a new sample
new_sample = [("color", "green"), ("shape", "circle"), ("size", "small")]
new_sample_hashed = hasher.transform([new_sample])
prediction = model.predict(new_sample_hashed)
print(f"Prediction for new sample: {prediction[0]}")
```

Slide 8: Feature Hashing in Natural Language Processing

Feature hashing is widely used in Natural Language Processing (NLP) tasks, especially when dealing with large vocabularies or streaming text data. Let's implement a simple n-gram feature extractor using the hashing trick, which can be useful for tasks like text classification or language modeling.

```python
import numpy as np
from collections import defaultdict

def hash_ngrams(text, n, num_buckets):
    words = text.lower().split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    feature_vector = defaultdict(int)
    for ngram in ngrams:
        bucket = hash(ngram) % num_buckets
        feature_vector[bucket] += 1
    
    return feature_vector

def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = np.sqrt(sum1) * np.sqrt(sum2)
    
    return numerator / denominator if denominator != 0 else 0

# Example usage
text1 = "The quick brown fox jumps over the lazy dog"
text2 = "The lazy dog is jumped over by the quick brown fox"
text3 = "A completely different sentence about cats and mice"

vec1 = hash_ngrams(text1, n=2, num_buckets=100)
vec2 = hash_ngrams(text2, n=2, num_buckets=100)
vec3 = hash_ngrams(text3, n=2, num_buckets=100)

print(f"Similarity between text1 and text2: {cosine_similarity(vec1, vec2):.4f}")
print(f"Similarity between text1 and text3: {cosine_similarity(vec1, vec3):.4f}")
print(f"Similarity between text2 and text3: {cosine_similarity(vec2, vec3):.4f}")
```

Slide 9: Feature Hashing for Anomaly Detection

Feature hashing can be applied to anomaly detection tasks, especially when dealing with high-dimensional data or streaming scenarios. Let's implement a simple anomaly detection system using feature hashing and the Isolation Forest algorithm, which is particularly effective for detecting anomalies in high-dimensional spaces.

```python
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

# Generate normal data
np.random.seed(42)
n_samples = 1000
n_features = 100
normal_data = np.random.rand(n_samples, n_features)

# Generate anomalous data
n_anomalies = 50
anomalies = np.random.rand(n_anomalies, n_features) * 2 - 1  # Values outside [0, 1]

# Combine normal and anomalous data
X = np.vstack([normal_data, anomalies])
y_true = np.hstack([np.ones(n_samples), -np.ones(n_anomalies)])

# Convert to dictionary format for FeatureHasher
X_dict = [{f"feature_{i}": val for i, val in enumerate(row)} for row in X]

# Apply feature hashing
hasher = FeatureHasher(n_features=50, input_type="dict")
X_hashed = hasher.transform(X_dict)

# Train Isolation Forest
clf = IsolationForest(contamination=n_anomalies / (n_samples + n_anomalies), random_state=42)
y_pred = clf.fit_predict(X_hashed)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Anomaly detection accuracy: {accuracy:.4f}")

# Detect anomalies in new data
new_data = np.random.rand(5, n_features)
new_data[2] = np.random.rand(n_features) * 2 - 1  # Make one sample anomalous
new_data_dict = [{f"feature_{i}": val for i, val in enumerate(row)} for row in new_data]
new_data_hashed = hasher.transform(new_data_dict)

new_predictions = clf.predict(new_data_hashed)
print("Predictions for new data (-1 indicates anomaly):")
print(new_predictions)
```

Slide 10: Real-Life Example: Email Spam Detection

Let's apply the hashing trick to a practical problem: email spam detection. We'll create a simple spam classifier using feature hashing to process email content efficiently. This example demonstrates how feature hashing can be used in real-world scenarios to handle large-scale text classification tasks.

```python
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample email data (content, label)
emails = [
    ("Get rich quick! Buy our amazing product now!", 1),
    ("Meeting scheduled for tomorrow at 2 PM", 0),
    ("Congratulations! You've won a free iPhone!", 1),
    ("Project report due by end of week", 0),
    ("Increase your followers on social media instantly!", 1),
    ("Reminder: Team lunch next Friday", 0),
    ("Limited time offer: 90% off luxury watches!", 1),
    ("Please review the attached document", 0),
    ("You're our lucky winner! Claim your prize now!", 1),
    ("Agenda for next week's planning session", 0)
]

# Split data into content and labels
X, y = zip(*emails)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create HashingVectorizer
vectorizer = HashingVectorizer(n_features=2**10, alternate_sign=False)

# Transform the email content
X_train_hashed = vectorizer.transform(X_train)
X_test_hashed = vectorizer.transform(X_test)

# Train a classifier
clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
clf.fit(X_train_hashed, y_train)

# Make predictions
y_pred = clf.predict(X_test_hashed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Predict on new emails
new_emails = [
    "Don't miss out on this incredible opportunity!",
    "The quarterly report is now available for review"
]
new_emails_hashed = vectorizer.transform(new_emails)
predictions = clf.predict(new_emails_hashed)

for email, pred in zip(new_emails, predictions):
    print(f"Email: '{email}'")
    print(f"Prediction: {'Spam' if pred == 1 else 'Ham'}\n")
```

Slide 11: Handling Out-of-Memory Scenarios with Feature Hashing

Feature hashing is particularly useful when dealing with datasets that are too large to fit in memory. Let's explore how to process a large text dataset using feature hashing and batch processing. This approach allows us to train a model on data that exceeds available RAM.

```python
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

def batch_generator(file_path, batch_size):
    while True:
        batch = []
        labels = []
        with open(file_path, 'r') as f:
            for _ in range(batch_size):
                line = f.readline().strip()
                if not line:
                    break
                label, text = line.split('\t')
                batch.append(text)
                labels.append(int(label))
        if not batch:
            break
        yield batch, labels

# Initialize HashingVectorizer and classifier
vectorizer = HashingVectorizer(n_features=2**15, alternate_sign=False)
clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)

# Training loop
batch_size = 1000
n_iterations = 10  # Number of passes over the data

for epoch in range(n_iterations):
    for batch, labels in batch_generator('large_dataset.txt', batch_size):
        X_batch = vectorizer.transform(batch)
        clf.partial_fit(X_batch, labels, classes=[0, 1])

    # Optional: Evaluate on a validation set
    # ...

print("Training complete")

# Example prediction
new_text = "Click here for a special offer!"
new_text_hashed = vectorizer.transform([new_text])
prediction = clf.predict(new_text_hashed)
print(f"Prediction for '{new_text}': {'Spam' if prediction[0] == 1 else 'Ham'}")
```

Slide 12: Feature Hashing for Recommendation Systems

Feature hashing can be applied to build simple recommendation systems, especially when dealing with large-scale user-item interaction data. Let's implement a basic collaborative filtering recommender using feature hashing to handle user and item features efficiently.

```python
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDRegressor

# Sample user-item interaction data
interactions = [
    {"user_id": "user1", "item_id": "item1", "category": "electronics", "rating": 4.5},
    {"user_id": "user2", "item_id": "item2", "category": "books", "rating": 3.0},
    {"user_id": "user1", "item_id": "item3", "category": "clothing", "rating": 5.0},
    # ... more interactions ...
]

# Feature hasher
hasher = FeatureHasher(n_features=2**10, input_type='string')

# Prepare data
X = []
y = []
for interaction in interactions:
    features = [
        f"user_{interaction['user_id']}",
        f"item_{interaction['item_id']}",
        f"category_{interaction['category']}"
    ]
    X.append(features)
    y.append(interaction['rating'])

X_hashed = hasher.transform(X)
y = np.array(y)

# Train a model
model = SGDRegressor(max_iter=1000, tol=1e-3)
model.fit(X_hashed, y)

# Predict rating for a new user-item pair
new_interaction = ["user_newuser", "item_newitem", "category_electronics"]
new_interaction_hashed = hasher.transform([new_interaction])
predicted_rating = model.predict(new_interaction_hashed)

print(f"Predicted rating: {predicted_rating[0]:.2f}")
```

Slide 13: Limitations and Considerations of Feature Hashing

While feature hashing is a powerful technique, it's important to be aware of its limitations and considerations for effective use in machine learning projects. The primary challenge is the potential for hash collisions, which can impact model performance. Additionally, the lack of feature interpretability can make it difficult to understand which specific features are most influential in the model's decisions.

To mitigate these issues, consider the following strategies:

1. Choose an appropriate hash size to balance between memory usage and collision probability.
2. Use signed hashing to reduce the impact of collisions.
3. Combine feature hashing with other dimensionality reduction techniques for improved performance.
4. Monitor model performance and adjust the number of hash buckets if necessary.

Here's a simple example demonstrating how to evaluate the impact of different hash sizes:

```python
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Sample data
X = [
    "This is a positive review",
    "Negative sentiment in this text",
    "Another positive example",
    "Very disappointed, negative experience",
    # ... more text data ...
]
y = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# Test different hash sizes
hash_sizes = [2**i for i in range(5, 15)]
mean_scores = []

for n_features in hash_sizes:
    vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False)
    X_hashed = vectorizer.transform(X)
    
    clf = LogisticRegression(random_state=42)
    scores = cross_val_score(clf, X_hashed, y, cv=3)
    mean_scores.append(np.mean(scores))

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.semilogx(hash_sizes, mean_scores, marker='o')
plt.xlabel('Number of Hash Buckets')
plt.ylabel('Mean Cross-Validation Score')
plt.title('Impact of Hash Size on Model Performance')
plt.grid(True)
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into feature hashing and its applications in machine learning, here are some valuable resources:

1. "Feature Hashing for Large Scale Multitask Learning" by Weinberger et al. (2009) ArXiv link: [https://arxiv.org/abs/0902.2206](https://arxiv.org/abs/0902.2206)
2. "Random Features for Large-Scale Kernel Machines" by Rahimi and Recht (2007) ArXiv link: [https://arxiv.org/abs/0702099](https://arxiv.org/abs/0702099)
3. "Bloom Filters in Probabilistic Verification" by Dillinger and Manolios (2004) ArXiv link: [https://arxiv.org/abs/cs/0411014](https://arxiv.org/abs/cs/0411014)

These papers provide in-depth theoretical foundations and practical insights into feature hashing and related techniques. They offer a comprehensive understanding of the subject matter and its various applications in machine learning and data processing.

