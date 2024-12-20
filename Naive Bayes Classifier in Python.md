## Naive Bayes Classifier in Python
Slide 1: Introduction to Naive Bayes

Naive Bayes is a probabilistic machine learning algorithm based on Bayes' theorem. It's widely used for classification tasks, particularly in text classification and spam filtering. The algorithm assumes that features are independent of each other, hence the term "naive".

```python
# Simple example of Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a random dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")
```

Slide 2: Bayes' Theorem

Bayes' theorem is the foundation of Naive Bayes. It describes the probability of an event based on prior knowledge of conditions that might be related to the event.

```python
import numpy as np

def bayes_theorem(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

# Example: probability of having a disease given a positive test result
prior = 0.01  # 1% of population has the disease
likelihood = 0.95  # 95% chance of positive test if you have the disease
false_positive = 0.1  # 10% chance of false positive
evidence = likelihood * prior + false_positive * (1 - prior)

posterior = bayes_theorem(prior, likelihood, evidence)
print(f"Probability of having the disease given a positive test: {posterior:.2f}")
```

Slide 3: Types of Naive Bayes

There are three main types of Naive Bayes classifiers: Gaussian, Multinomial, and Bernoulli. The choice depends on the nature of the features in your dataset.

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import numpy as np

# Example data
X = np.random.rand(100, 5)
y = np.random.randint(2, size=100)

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X, y)
print("Gaussian NB score:", gnb.score(X, y))

# Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X, y)
print("Multinomial NB score:", mnb.score(X, y))

# Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X, y)
print("Bernoulli NB score:", bnb.score(X, y))
```

Slide 4: Gaussian Naive Bayes

Gaussian Naive Bayes assumes that the features follow a normal distribution. It's suitable for continuous data and is often used in classification problems where features are continuous.

```python
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train Gaussian Naive Bayes
model = GaussianNB()
model.fit(X, y)

# Create a mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Make predictions on the mesh grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.title("Gaussian Naive Bayes Decision Boundary")
plt.show()
```

Slide 5: Multinomial Naive Bayes

Multinomial Naive Bayes is typically used for discrete counts, such as word counts in text classification. It's particularly effective for document classification and spam filtering.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Sample text data
texts = [
    "I love this movie",
    "This movie is awful",
    "Great acting and plot",
    "Terrible script and acting"
]
labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# Convert text to word count vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X, labels)

# Predict a new text
new_text = ["This movie is amazing"]
new_X = vectorizer.transform(new_text)
prediction = model.predict(new_X)

print("Prediction:", "Positive" if prediction[0] == 1 else "Negative")
print("Probability:", model.predict_proba(new_X)[0])
```

Slide 6: Bernoulli Naive Bayes

Bernoulli Naive Bayes is designed for binary/boolean features. It's often used in text classification with 'bag of words' model where we're only concerned with whether a word occurs in the document or not.

```python
from sklearn.naive_bayes import BernoulliNB
import numpy as np

# Sample data: each feature is binary (0 or 1)
# Let's say we're classifying animals based on features:
# [has_fur, has_feathers, lays_eggs, can_fly]
X = np.array([
    [1, 0, 0, 0],  # dog
    [1, 0, 0, 0],  # cat
    [0, 1, 1, 1],  # bird
    [0, 1, 1, 1],  # penguin
    [1, 0, 1, 0],  # platypus
])
y = np.array(['mammal', 'mammal', 'bird', 'bird', 'mammal'])

# Train Bernoulli Naive Bayes
model = BernoulliNB()
model.fit(X, y)

# Predict a new animal
new_animal = np.array([[1, 0, 1, 0]])  # has fur, lays eggs, can't fly
prediction = model.predict(new_animal)
probabilities = model.predict_proba(new_animal)

print("Prediction:", prediction[0])
print("Probabilities:", dict(zip(model.classes_, probabilities[0])))
```

Slide 7: Feature Independence Assumption

Naive Bayes assumes that features are independent of each other, which is often not true in real-world scenarios. Despite this "naive" assumption, the algorithm often performs well in practice.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Generate correlated data
np.random.seed(42)
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]
X = np.random.multivariate_normal(mean, cov, 1000)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train Naive Bayes
model = GaussianNB()
model.fit(X, y)

# Plot the data and decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.title("Naive Bayes with Correlated Features")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

Slide 8: Handling Continuous Features

When dealing with continuous features in Naive Bayes, we often assume they follow a Gaussian distribution. This is the basis for Gaussian Naive Bayes.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate sample data
np.random.seed(42)
data = np.random.normal(loc=5, scale=2, size=1000)

# Calculate mean and standard deviation
mean = np.mean(data)
std = np.std(data)

# Plot histogram and Gaussian curve
plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Gaussian Distribution of a Continuous Feature")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {std:.2f}")
```

Slide 9: Laplace Smoothing

Laplace smoothing (or additive smoothing) is a technique used to handle the zero probability problem in Naive Bayes. It adds a small count to all feature occurrences to prevent zero probabilities.

```python
import numpy as np

def naive_bayes_predict(X_train, y_train, x_test, alpha=1.0):
    # Count occurrences of each class
    class_counts = np.bincount(y_train)
    
    # Count occurrences of features for each class
    feature_counts = np.array([np.bincount(X_train[y_train == c], minlength=2) 
                               for c in range(len(class_counts))])
    
    # Apply Laplace smoothing
    smoothed_counts = feature_counts + alpha
    smoothed_probs = smoothed_counts / smoothed_counts.sum(axis=1, keepdims=True)
    
    # Calculate log probabilities
    log_probs = np.log(smoothed_probs)
    log_class_probs = np.log(class_counts / len(y_train))
    
    # Predict
    return np.argmax(log_class_probs + (x_test * log_probs[:, 1] + 
                                        (1 - x_test) * log_probs[:, 0]).sum(axis=1))

# Example usage
X_train = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
y_train = np.array([0, 1, 1])
x_test = np.array([1, 0, 1])

prediction = naive_bayes_predict(X_train, y_train, x_test)
print("Prediction:", prediction)
```

Slide 10: Text Classification with Naive Bayes

Text classification is one of the most common applications of Naive Bayes. It's particularly effective for spam detection, sentiment analysis, and document categorization.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample data
texts = [
    "I love this product, it's amazing",
    "This is terrible, worst purchase ever",
    "Great customer service and fast delivery",
    "Poor quality and overpriced",
    "Excellent value for money",
    "Disappointing performance and unreliable"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

# Split the data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Vectorize the text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train and predict
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# Evaluate
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
```

Slide 11: Real-Life Example: Email Spam Classification

Email spam classification is a practical application of Naive Bayes. This example demonstrates how to build a simple spam classifier using the Enron-Spam dataset.

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset (assuming you have it in a CSV file)
data = pd.read_csv('spam_data.csv')
X = data['text']
y = data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# Example prediction
new_email = ["Get rich quick! Limited time offer!"]
new_email_vec = vectorizer.transform(new_email)
prediction = model.predict(new_email_vec)
print("Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")
```

Slide 12: Real-Life Example: Sentiment Analysis of Product Reviews

Sentiment analysis is another common application of Naive Bayes, often used to analyze customer feedback and product reviews.

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample product review data
reviews = [
    "This smartphone has an excellent camera and long battery life",
    "The laptop's performance is disappointing and it overheats quickly",
    "Great sound quality on these headphones, very comfortable to wear",
    "The tablet's screen is too small and the interface is confusing",
    "Impressed with the TV's picture quality and smart features",
    "The smartwatch's fitness tracking is inaccurate and battery drains fast"
]
sentiments = [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

# Split the data
X_train, X_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=0.3, random_state=42)

# Vectorize the text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Print classification report
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Example prediction
new_review = ["This product exceeded my expectations, highly recommended!"]
new_review_vec = vectorizer.transform(new_review)
prediction = model.predict(new_review_vec)
print("Sentiment:", "Positive" if prediction[0] == 1 else "Negative")
```

Slide 13: Advantages and Disadvantages of Naive Bayes

Naive Bayes is a simple yet powerful algorithm with both strengths and limitations. Understanding these can help in deciding when to use this algorithm.

```python
import matplotlib.pyplot as plt

advantages = [
    "Simple and fast",
    "Works well with high-dimensional data",
    "Effective with small training sets",
    "Easy to implement"
]

disadvantages = [
    "Assumes feature independence",
    "Sensitive to feature selection",
    "Can be outperformed by other algorithms",
    "Requires smoothing techniques"
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.barh(range(len(advantages)), [1]*len(advantages), align='center')
ax1.set_yticks(range(len(advantages)))
ax1.set_yticklabels(advantages)
ax1.set_title("Advantages")

ax2.barh(range(len(disadvantages)), [1]*len(disadvantages), align='center')
ax2.set_yticks(range(len(disadvantages)))
ax2.set_yticklabels(disadvantages)
ax2.set_title("Disadvantages")

plt.tight_layout()
plt.show()
```

Slide 14: Naive Bayes vs. Other Classifiers

Comparing Naive Bayes to other popular classifiers can provide insights into its relative performance and use cases.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# Generate a random dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Define classifiers
classifiers = {
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Perform cross-validation
results = {}
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X, y, cv=5)
    results[name] = scores

# Plot results
plt.figure(figsize=(10, 6))
plt.boxplot([results[name] for name in classifiers.keys()], labels=classifiers.keys())
plt.title('Classifier Comparison')
plt.ylabel('Accuracy')
plt.show()

# Print mean accuracies
for name, scores in results.items():
    print(f"{name}: Mean accuracy = {np.mean(scores):.3f}")
```

Slide 15: Additional Resources

For those interested in delving deeper into Naive Bayes and its applications, here are some valuable resources:

1. "A Tutorial on Naive Bayes Classification" by Irina Rish (2001) ArXiv URL: [https://arxiv.org/abs/1410.5329](https://arxiv.org/abs/1410.5329)
2. "An Introduction to Naive Bayes" by Sebastian Raschka (2014) ArXiv URL: [https://arxiv.org/abs/1410.5329](https://arxiv.org/abs/1410.5329)
3. "Naive Bayes and Text Classification" by Charu C. Aggarwal and ChengXiang Zhai (2012) ArXiv URL: [https://arxiv.org/abs/1410.5329](https://arxiv.org/abs/1410.5329)

These papers provide in-depth explanations of Naive Bayes algorithms, their mathematical foundations, and practical applications in various domains.

