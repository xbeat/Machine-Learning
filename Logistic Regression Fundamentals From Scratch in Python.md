## Logistic Regression Fundamentals From Scratch in Python

Slide 1: Where We Can Use Logistic Regression

Logistic Regression is a versatile statistical method used for predicting binary outcomes. It's commonly applied in various fields:

1.  Medical diagnosis: Predicting the likelihood of a disease based on symptoms.
2.  Customer behavior: Estimating the probability of a customer making a purchase.
3.  Spam detection: Classifying emails as spam or not spam.
4.  Quality control: Predicting product defects in manufacturing.

Here's a simple example of logistic regression for medical diagnosis:

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def predict_disease(age, blood_pressure):
    # Simplified coefficients (would be learned from data)
    w0, w1, w2 = -8, 0.1, 0.05
    z = w0 + w1 * age + w2 * blood_pressure
    probability = sigmoid(z)
    return probability

# Example usage
age = 60
blood_pressure = 130
risk = predict_disease(age, blood_pressure)
print(f"Disease risk for a {age}-year-old with blood pressure {blood_pressure}: {risk:.2%}")
```

This code demonstrates a basic logistic regression model for predicting disease risk based on age and blood pressure.

Slide 2: Logistic Regression vs Linear Regression

While both logistic and linear regression are used for prediction, they differ in key aspects:

1.  Output: Linear regression predicts continuous values, while logistic regression predicts probabilities between 0 and 1.
2.  Function: Linear regression uses a linear function, logistic regression uses the sigmoid function.
3.  Error measurement: Linear regression typically uses mean squared error, logistic regression uses log loss.

Let's compare them with a simple example:

```python
import math

def linear_regression(x):
    # Simplified linear model: y = 2x + 1
    return 2 * x + 1

def logistic_regression(x):
    # Simplified logistic model: sigmoid(2x - 4)
    return 1 / (1 + math.exp(-(2 * x - 4)))

# Compare predictions
x_values = [1, 2, 3, 4, 5]
linear_predictions = [linear_regression(x) for x in x_values]
logistic_predictions = [logistic_regression(x) for x in x_values]

print("x\tLinear\tLogistic")
for x, linear, logistic in zip(x_values, linear_predictions, logistic_predictions):
    print(f"{x}\t{linear:.2f}\t{logistic:.2f}")
```

This code demonstrates how linear regression produces unbounded outputs, while logistic regression outputs are always between 0 and 1.

Slide 3: How Logistic Regression Works

Logistic Regression models the probability of an input belonging to a particular class. It uses the logistic function to transform a linear combination of inputs into a probability.

The process involves:

1.  Combining input features linearly
2.  Applying the sigmoid function to the linear combination
3.  Interpreting the output as a probability
4.  Using a threshold (usually 0.5) to make binary predictions

Here's a simple implementation:

```python
import math

def logistic_regression(features, weights):
    # Linear combination of features and weights
    z = sum(f * w for f, w in zip(features, weights))
    # Apply sigmoid function
    probability = 1 / (1 + math.exp(-z))
    return probability

def predict_class(features, weights, threshold=0.5):
    prob = logistic_regression(features, weights)
    return 1 if prob > threshold else 0

# Example usage
features = [1, 2, 3]  # Input features
weights = [0.1, 0.2, 0.3]  # Model weights
probability = logistic_regression(features, weights)
prediction = predict_class(features, weights)

print(f"Probability: {probability:.4f}")
print(f"Predicted class: {prediction}")
```

This code demonstrates the core mechanics of logistic regression, from combining features to making a binary prediction.

Slide 4: Sigmoid Function in Logistic Regression

The sigmoid function is crucial in logistic regression. It maps any real-valued number to a value between 0 and 1, which can be interpreted as a probability.

The sigmoid function is defined as:

σ(x)\=11+e−x\\sigma(x) = \\frac{1}{1 + e^{-x}}σ(x)\=1+e−x1​

Key properties:

*   Output range: (0, 1)
*   S-shaped curve
*   Centered at x = 0, where σ(0) = 0.5

Let's visualize the sigmoid function:

```python
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

x = [i * 0.1 for i in range(-100, 101)]
y = [sigmoid(i) for i in x]

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("σ(x)")
plt.grid(True)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axvline(x=0, color='r', linestyle='--')
plt.text(0.5, 0.5, 'σ(0) = 0.5', verticalalignment='bottom')
plt.savefig('sigmoid_function.png')
plt.close()

print("Sigmoid function visualization saved as 'sigmoid_function.png'")
```

This code generates a plot of the sigmoid function, highlighting its S-shape and key properties.

Slide 5: Explanation with 'SUV Data Analysis' Dataset

Let's analyze a hypothetical SUV purchase dataset to predict if a customer will buy an SUV based on age and estimated salary.

Dataset structure:

*   Features: Age, EstimatedSalary
*   Target: Purchased (0 = No, 1 = Yes)

We'll implement logistic regression from scratch:

```python
import math
import random

# Sample dataset
data = [
    (35, 50000, 0), (45, 70000, 1), (25, 35000, 0),
    (55, 80000, 1), (30, 45000, 0), (50, 75000, 1),
    (40, 60000, 1), (60, 90000, 1), (28, 40000, 0),
    (52, 65000, 1)
]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def predict(features, weights):
    z = sum(f * w for f, w in zip(features, weights))
    return sigmoid(z)

def train_logistic_regression(data, learning_rate=0.01, epochs=1000):
    weights = [random.random() for _ in range(3)]  # Including bias
    
    for _ in range(epochs):
        for age, salary, actual in data:
            features = [1, age / 100, salary / 100000]  # Normalize and add bias term
            prediction = predict(features, weights)
            error = actual - prediction
            weights = [w + learning_rate * error * f for w, f in zip(weights, features)]
    
    return weights

# Train the model
trained_weights = train_logistic_regression(data)

# Test the model
test_data = [(38, 55000), (48, 72000)]
for age, salary in test_data:
    features = [1, age / 100, salary / 100000]
    prob = predict(features, trained_weights)
    print(f"Age: {age}, Salary: ${salary}")
    print(f"Probability of purchasing SUV: {prob:.2%}")
    print(f"Prediction: {'Will purchase' if prob > 0.5 else 'Will not purchase'}\n")
```

This code demonstrates a simple logistic regression implementation for the SUV dataset, including training and prediction.

Slide 6: Hyperparameters of Logistic Regression

Hyperparameters are configuration settings for the learning algorithm that are not learned from the data. Key hyperparameters in logistic regression include:

1.  Learning rate: Controls the step size in gradient descent.
2.  Number of iterations: Determines how long the algorithm trains.
3.  Regularization strength: Helps prevent overfitting.
4.  Solver: Algorithm used for optimization (e.g., 'liblinear', 'newton-cg', 'lbfgs').
5.  Threshold: Cut-off for binary classification (default is usually 0.5).

Let's implement a logistic regression model with adjustable hyperparameters:

```python
import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, regularization=0.1):
        self.lr = learning_rate
        self.iterations = iterations
        self.reg = regularization
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [random.random() for _ in range(n_features + 1)]

        for _ in range(self.iterations):
            for xi, yi in zip(X, y):
                xi = [1] + xi  # Add bias term
                y_pred = sigmoid(sum(w * x for w, x in zip(self.weights, xi)))
                error = yi - y_pred
                self.weights = [w + self.lr * (error * x - self.reg * w) 
                                for w, x in zip(self.weights, xi)]

    def predict_proba(self, X):
        return [sigmoid(sum(w * x for w, x in zip(self.weights, [1] + xi))) 
                for xi in X]

    def predict(self, X, threshold=0.5):
        return [1 if p > threshold else 0 for p in self.predict_proba(X)]

# Example usage
X = [[1, 2], [2, 3], [3, 4], [4, 5]]
y = [0, 0, 1, 1]

model = LogisticRegression(learning_rate=0.1, iterations=1000, regularization=0.01)
model.fit(X, y)

print("Predictions:")
for xi, yi, y_pred in zip(X, y, model.predict(X)):
    print(f"Input: {xi}, Actual: {yi}, Predicted: {y_pred}")
```

This implementation allows you to adjust learning rate, number of iterations, and regularization strength.

Slide 7: Metrics Used in Logistic Regression

Evaluating logistic regression models involves several key metrics:

1.  Accuracy: Proportion of correct predictions.
2.  Precision: Proportion of true positives among positive predictions.
3.  Recall: Proportion of true positives among actual positives.
4.  F1 Score: Harmonic mean of precision and recall.
5.  ROC AUC: Area under the Receiver Operating Characteristic curve.

Let's implement these metrics:

```python
def calculate_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Example usage
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 1, 0, 1, 1]

metrics = calculate_metrics(y_true, y_pred)
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

This code calculates and displays key metrics for evaluating logistic regression models.

Slide 8: Explaining AUC-ROC Curve

The AUC-ROC (Area Under the Curve - Receiver Operating Characteristic) curve is a performance measurement for classification problems at various thresholds settings.

Key points:

*   ROC curve plots True Positive Rate vs False Positive Rate
*   AUC represents the degree of separability between classes
*   AUC ranges from 0 to 1, with 1 being perfect classification

Let's implement and visualize the AUC-ROC curve:

```python
import random
import matplotlib.pyplot as plt

def calculate_tpr_fpr(y_true, y_scores, threshold):
    tp = sum(1 for t, s in zip(y_true, y_scores) if t == 1 and s >= threshold)
    fp = sum(1 for t, s in zip(y_true, y_scores) if t == 0 and s >= threshold)
    fn = sum(1 for t, s in zip(y_true, y_scores) if t == 1 and s < threshold)
    tn = sum(1 for t, s in zip(y_true, y_scores) if t == 0 and s < threshold)
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr

def plot_roc_curve(y_true, y_scores):
    thresholds = sorted(set(y_scores), reverse=True)
    tpr_fpr_pairs = [calculate_tpr_fpr(y_true, y_scores, t) for t in thresholds]
    tpr_fpr_pairs.append((1, 1))  # Add (1,1) point
    tpr_fpr_pairs.sort(key=lambda x: x[1])  # Sort by FPR
    
    tprs, fprs = zip(*tpr_fpr_pairs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fprs, tprs, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Calculate AUC
    auc = sum((fprs[i] - fprs[i-1]) * (tprs[i] + tprs[i-1]) / 2 
              for i in range(1, len(fprs)))
    
    print(f"AUC: {auc:.4f}")
    print("ROC curve saved as 'roc_curve.png'")

# Generate example data
random.seed(42)
y_true = [random.choice([0, 1]) for _ in range(100)]
y_scores = [random.random() for _ in range(100)]

plot_roc_curve(y_true, y_scores)
```

Slide 9: Predicting the Output

In logistic regression, predicting the output involves three key steps:

1.  Calculating the linear combination of features and weights
2.  Applying the sigmoid function to get a probability
3.  Using a threshold to make a binary decision

Let's implement a complete prediction pipeline:

```python
import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [random.random() for _ in range(n_features)]
        self.bias = random.random()

        for _ in range(epochs):
            for xi, yi in zip(X, y):
                linear_pred = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                y_pred = sigmoid(linear_pred)
                
                # Update weights and bias
                self.weights = [w - learning_rate * (y_pred - yi) * x 
                                for w, x in zip(self.weights, xi)]
                self.bias -= learning_rate * (y_pred - yi)

    def predict_proba(self, X):
        return [sigmoid(sum(w * x for w, x in zip(self.weights, xi)) + self.bias) 
                for xi in X]

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return [1 if p > threshold else 0 for p in probas]

# Example usage
X_train = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y_train = [0, 0, 1, 1, 1]

model = LogisticRegression()
model.fit(X_train, y_train)

X_test = [[2, 2], [3, 3], [4, 4]]
probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test)

for x, prob, pred in zip(X_test, probabilities, predictions):
    print(f"Input: {x}, Probability: {prob:.4f}, Prediction: {pred}")
```

Slide 10: Real-Life Example: Email Spam Detection

Logistic regression is commonly used for email spam detection. Let's implement a simple spam classifier:

```python
import re
from collections import defaultdict

class SpamClassifier:
    def __init__(self):
        self.word_counts = defaultdict(lambda: {'spam': 0, 'ham': 0})
        self.class_counts = {'spam': 0, 'ham': 0}

    def train(self, emails, labels):
        for email, label in zip(emails, labels):
            words = self._tokenize(email)
            self.class_counts[label] += 1
            for word in set(words):  # Count each word only once per email
                self.word_counts[word][label] += 1

    def predict(self, email):
        words = self._tokenize(email)
        spam_score = self._calculate_score(words, 'spam')
        ham_score = self._calculate_score(words, 'ham')
        return 'spam' if spam_score > ham_score else 'ham'

    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def _calculate_score(self, words, class_label):
        score = 0
        total_emails = sum(self.class_counts.values())
        class_prob = self.class_counts[class_label] / total_emails
        for word in words:
            word_in_class = self.word_counts[word][class_label]
            score += math.log((word_in_class + 1) / (self.class_counts[class_label] + 2))
        return score + math.log(class_prob)

# Example usage
emails = [
    "Get rich quick! Buy now!",
    "Meeting scheduled for tomorrow",
    "Claim your prize today!",
    "Project update: new features added"
]
labels = ['spam', 'ham', 'spam', 'ham']

classifier = SpamClassifier()
classifier.train(emails, labels)

test_email = "Win a free vacation now!"
prediction = classifier.predict(test_email)
print(f"Email: '{test_email}'")
print(f"Prediction: {prediction}")
```

Slide 11: Real-Life Example: Medical Diagnosis

Logistic regression is valuable in medical diagnosis for predicting the likelihood of diseases. Here's a simplified example for diabetes prediction:

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class DiabetesPredictor:
    def __init__(self):
        # Pre-trained weights (in practice, these would be learned from data)
        self.weights = {
            'glucose': 0.05,
            'bmi': 0.08,
            'age': 0.03,
            'bias': -6.0
        }

    def predict_diabetes_risk(self, glucose, bmi, age):
        z = (self.weights['glucose'] * glucose +
             self.weights['bmi'] * bmi +
             self.weights['age'] * age +
             self.weights['bias'])
        risk = sigmoid(z)
        return risk

# Example usage
predictor = DiabetesPredictor()

patients = [
    {'glucose': 85, 'bmi': 22, 'age': 25},
    {'glucose': 168, 'bmi': 33, 'age': 50},
    {'glucose': 120, 'bmi': 27, 'age': 35}
]

for i, patient in enumerate(patients, 1):
    risk = predictor.predict_diabetes_risk(**patient)
    print(f"Patient {i}:")
    print(f"  Glucose: {patient['glucose']}, BMI: {patient['bmi']}, Age: {patient['age']}")
    print(f"  Diabetes Risk: {risk:.2%}")
    print(f"  Assessment: {'High risk' if risk > 0.5 else 'Low risk'}\n")
```

Slide 12: Handling Imbalanced Datasets

In many real-world scenarios, datasets are imbalanced, with one class significantly outnumbering the other. This can bias logistic regression models. Techniques to address this include:

1.  Oversampling the minority class
2.  Undersampling the majority class
3.  Synthetic data generation (e.g., SMOTE)
4.  Adjusting class weights

Let's implement a simple weighted logistic regression:

```python
import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class WeightedLogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y, class_weights={0: 1, 1: 1}):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [random.random() for _ in range(n_features)]
        self.bias = random.random()

        for _ in range(self.iterations):
            for xi, yi in zip(X, y):
                linear_pred = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                y_pred = sigmoid(linear_pred)
                
                # Apply class weight
                weight = class_weights[yi]
                error = weight * (yi - y_pred)
                
                # Update weights and bias
                self.weights = [w + self.lr * error * x for w, x in zip(self.weights, xi)]
                self.bias += self.lr * error

    def predict(self, X):
        return [1 if sigmoid(sum(w * x for w, x in zip(self.weights, xi)) + self.bias) > 0.5 else 0 
                for xi in X]

# Example with imbalanced dataset
X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]]
y = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  # Imbalanced: 7 negative, 3 positive

model = WeightedLogisticRegression()
model.fit(X, y, class_weights={0: 1, 1: 2.33})  # Weighting positive class more

predictions = model.predict(X)
accuracy = sum(1 for true, pred in zip(y, predictions) if true == pred) / len(y)
print(f"Accuracy on imbalanced dataset: {accuracy:.2%}")
```

Slide 13: Regularization in Logistic Regression

Regularization helps prevent overfitting in logistic regression by adding a penalty term to the loss function. Common types are:

1.  L1 (Lasso): Encourages sparsity
2.  L2 (Ridge): Prevents large weight values

Let's implement L2 regularization:

```python
import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class RegularizedLogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, lambda_param=0.1):
        self.lr = learning_rate
        self.iterations = iterations
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [random.random() for _ in range(n_features)]
        self.bias = random.random()

        for _ in range(self.iterations):
            for xi, yi in zip(X, y):
                linear_pred = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                y_pred = sigmoid(linear_pred)
                
                # Compute gradients with L2 regularization
                dw = [(y_pred - yi) * x + (self.lambda_param * w) / n_samples 
                      for x, w in zip(xi, self.weights)]
                db = (y_pred - yi)
                
                # Update weights and bias
                self.weights = [w - self.lr * dw_i for w, dw_i in zip(self.weights, dw)]
                self.bias -= self.lr * db

    def predict(self, X):
        return [1 if sigmoid(sum(w * x for w, x in zip(self.weights, xi)) + self.bias) > 0.5 else 0 
                for xi in X]

# Example usage
X = [[1, 2], [2, 3], [3, 4], [4, 5]]
y = [0, 0, 1, 1]

model = RegularizedLogisticRegression(lambda_param=0.1)
model.fit(X, y)

test_X = [[2, 2], [3, 3], [4, 4]]
predictions = model.predict(test_X)

for x, pred in zip(test_X, predictions):
    print(f"Input: {x}, Prediction: {pred}")
```

Slide 14: Additional Resources

For those interested in diving deeper into logistic regression and related topics, here are some valuable resources:

1.  "Logistic Regression: From Basic to Advanced" by Hosmer and Lemeshow (book)
2.  Stanford CS229 Machine Learning Course Notes: [http://cs229.stanford.edu/notes/cs229-notes1.pdf](http://cs229.stanford.edu/notes/cs229-notes1.pdf)
3.  "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani (free PDF available online)
4.  ArXiv paper: "A Survey of Logistic Regression Techniques for Binary Classification" by Tolles and Meurer (2016) ArXiv link: [https://arxiv.org/abs/1609.00451](https://arxiv.org/abs/1609.00451)

These resources provide in-depth explanations, mathematical foundations, and advanced techniques related to logistic regression and its applications in machine learning and statistics.

