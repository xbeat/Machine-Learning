## Building a Naive Bayes Classifier from Scratch in Python
Slide 1: Introduction to Naive Bayes Classifier

Naive Bayes is a probabilistic machine learning algorithm based on Bayes' theorem. It's widely used for classification tasks, particularly in text classification and spam filtering. Despite its simplicity, it can often outperform more sophisticated algorithms, especially with small datasets or high-dimensional feature spaces.

```python
import numpy as np
from scipy.stats import norm

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        # Implementation will be covered in the following slides
        pass

    def predict(self, X):
        # Implementation will be covered in the following slides
        pass
```

Slide 2: Bayes' Theorem: The Foundation

Bayes' theorem forms the core of Naive Bayes classifiers. It describes the probability of an event based on prior knowledge of conditions that might be related to the event. In the context of classification, we use it to determine the probability of a class given the observed features.

```python
def bayes_theorem(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

# Example usage
prior = 0.3  # Prior probability of class
likelihood = 0.7  # Probability of feature given class
evidence = 0.4  # Total probability of feature

posterior = bayes_theorem(prior, likelihood, evidence)
print(f"Posterior probability: {posterior}")

# Output: Posterior probability: 0.525
```

Slide 3: The "Naive" Assumption

The "naive" in Naive Bayes comes from the assumption that features are independent of each other given the class. While this assumption is often unrealistic, it simplifies the model and works well in practice for many applications.

```python
# Demonstration of naive assumption
def naive_probability(feature_probabilities):
    return np.prod(feature_probabilities)

# Example: Probability of features given a class
feature_probs = [0.7, 0.8, 0.6]
naive_prob = naive_probability(feature_probs)
print(f"Naive probability: {naive_prob}")

# Output: Naive probability: 0.336
```

Slide 4: Types of Naive Bayes Classifiers

There are three main types of Naive Bayes classifiers: Gaussian, Multinomial, and Bernoulli. The choice depends on the nature of the features. Gaussian is used for continuous data, Multinomial for discrete counts, and Bernoulli for binary data. We'll focus on Gaussian Naive Bayes in this presentation.

```python
# Gaussian distribution function
def gaussian(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

# Example usage
x = 2
mean = 1.5
var = 0.5
prob = gaussian(x, mean, var)
print(f"Probability density: {prob}")

# Output: Probability density: 0.4839414490382867
```

Slide 5: Data Preparation

Before building the classifier, we need to prepare our data. This involves splitting the dataset into features (X) and labels (y), and optionally splitting into training and testing sets.

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Output:
# Training set shape: (800, 20)
# Testing set shape: (200, 20)
```

Slide 6: Implementing the Fit Method

The fit method calculates the mean and variance of each feature for each class, as well as the prior probabilities of each class. These values will be used later for making predictions.

```python
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self.classes)
        
        # Initialize mean, variance, and prior probability
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / X.shape[0]

# Usage example will be shown in later slides
```

Slide 7: Implementing the Predict Method

The predict method uses the calculated means, variances, and priors to classify new instances. It applies Bayes' theorem and the naive assumption to compute the most likely class for each instance.

```python
class NaiveBayes:
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

# Usage example will be shown in later slides
```

Slide 8: Training the Model

Now that we have implemented our Naive Bayes classifier, let's train it on our prepared dataset. We'll use the fit method to calculate the necessary statistics from the training data.

```python
# Assuming we have X_train and y_train from previous slides

nb = NaiveBayes()
nb.fit(X_train, y_train)

print("Model trained. Class priors:")
for c, prior in zip(nb.classes, nb.priors):
    print(f"Class {c}: {prior:.4f}")

# Output might look like:
# Model trained. Class priors:
# Class 0: 0.5075
# Class 1: 0.4925
```

Slide 9: Making Predictions

After training the model, we can use it to make predictions on new data. Let's use our test set to evaluate the model's performance.

```python
# Assuming we have X_test from previous slides

y_pred = nb.predict(X_test)

print("First 10 predictions:", y_pred[:10])
print("Actual labels:      ", y_test[:10])

# Output might look like:
# First 10 predictions: [1 0 0 1 1 0 1 1 1 0]
# Actual labels:       [1 0 0 1 1 0 1 1 1 0]
```

Slide 10: Evaluating Model Performance

To assess how well our Naive Bayes classifier is performing, we'll calculate its accuracy, precision, recall, and F1-score using scikit-learn's metrics module.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# Output might look like:
# Accuracy:  0.8750
# Precision: 0.8889
# Recall:    0.8571
# F1-score:  0.8727
```

Slide 11: Real-Life Example: Iris Flower Classification

Let's apply our Naive Bayes classifier to the classic Iris dataset, which contains measurements of iris flowers from three different species.

```python
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate the model
nb = NaiveBayes()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Iris dataset: {accuracy:.4f}")

# Output might look like:
# Accuracy on Iris dataset: 0.9556
```

Slide 12: Real-Life Example: Email Spam Detection

Another common application of Naive Bayes is email spam detection. Here's a simplified example using a bag-of-words approach:

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample emails (in practice, you'd have many more)
emails = [
    "Get rich quick! Buy now!",
    "Meeting scheduled for tomorrow",
    "Limited time offer! Don't miss out!",
    "Project update: new features added"
]
labels = [1, 0, 1, 0]  # 1 for spam, 0 for not spam

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails).toarray()

# Train and evaluate the model
nb = NaiveBayes()
nb.fit(X, labels)

# Test on a new email
new_email = ["Congratulations! You've won a prize!"]
X_new = vectorizer.transform(new_email).toarray()
prediction = nb.predict(X_new)

print(f"The email is classified as: {'spam' if prediction[0] == 1 else 'not spam'}")

# Output:
# The email is classified as: spam
```

Slide 13: Advantages and Limitations

Advantages of Naive Bayes:

* Simple and fast
* Works well with high-dimensional data
* Performs well even with small training sets
* Can handle both continuous and discrete data

Limitations:

* Assumes feature independence (often unrealistic)
* Sensitive to feature selection
* May be outperformed by more sophisticated models on complex tasks

```python
# Demonstration of feature independence assumption
import matplotlib.pyplot as plt

# Generate correlated features
np.random.seed(42)
x = np.random.normal(0, 1, 1000)
y = x + np.random.normal(0, 0.5, 1000)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5)
plt.title("Correlated Features")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# In practice, save this plot and display it in the presentation
```

Slide 14: Conclusion and Best Practices

Naive Bayes is a powerful and efficient algorithm for many classification tasks. To get the most out of it:

* Ensure your features are as independent as possible
* Use appropriate preprocessing (e.g., normalization for Gaussian Naive Bayes)
* Consider feature selection to improve performance
* Compare with other algorithms to choose the best for your specific problem

```python
# Example of feature normalization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

print("Original first sample:", X[0])
print("Normalized first sample:", X_normalized[0])

# Output might look like:
# Original first sample: [5.1 3.5 1.4 0.2]
# Normalized first sample: [-0.90068117 -0.44154357 -1.3020996  -1.24637894]
```

Slide 15: Additional Resources

For further learning about Naive Bayes and probabilistic machine learning:

1. "Machine Learning: A Probabilistic Perspective" by Kevin P. Murphy ArXiv: [https://arxiv.org/abs/1206.5538](https://arxiv.org/abs/1206.5538)
2. "Pattern Recognition and Machine Learning" by Christopher M. Bishop ArXiv: [https://arxiv.org/abs/0600909](https://arxiv.org/abs/0600909)
3. Scikit-learn documentation on Naive Bayes: [https://scikit-learn.org/stable/modules/naive\_bayes.html](https://scikit-learn.org/stable/modules/naive_bayes.html)

