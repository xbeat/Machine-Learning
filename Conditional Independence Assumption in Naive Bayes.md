## Conditional Independence Assumption in Naive Bayes
Slide 1: Conditional Independence Assumption

Naive Bayes assumes that features are conditionally independent given the class label, meaning the presence of one feature does not affect the presence of another. This fundamental assumption simplifies probability calculations by allowing multiplication of individual feature probabilities.

```python
# Mathematical representation of conditional independence
'''
P(x₁,x₂,...,xn|y) = P(x₁|y) × P(x₂|y) × ... × P(xn|y)

Where:
- x₁,x₂,...,xn are features
- y is the class label
'''

# Basic implementation of conditional independence check
import numpy as np
from scipy.stats import chi2_contingency

def test_conditional_independence(data, feature1, feature2, class_label):
    contingency_tables = {}
    
    for label in np.unique(data[class_label]):
        subset = data[data[class_label] == label]
        cont_table = pd.crosstab(subset[feature1], subset[feature2])
        contingency_tables[label] = cont_table
        
    return contingency_tables
```

Slide 2: Implementing Naive Bayes from Scratch

A pure Python implementation of Naive Bayes demonstrates how the algorithm calculates class probabilities by leveraging the conditional independence assumption. This implementation handles both continuous and categorical features through different probability density functions.

```python
import numpy as np
from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.feature_params = defaultdict(dict)
    
    def fit(self, X, y):
        n_samples = len(y)
        self.classes = np.unique(y)
        
        # Calculate class probabilities
        for c in self.classes:
            self.class_probs[c] = np.sum(y == c) / n_samples
            
        # Calculate feature parameters
        for feature in range(X.shape[1]):
            for c in self.classes:
                feature_values = X[y == c, feature]
                self.feature_params[feature][c] = {
                    'mean': np.mean(feature_values),
                    'std': np.std(feature_values) + 1e-6
                }
    
    def _calculate_feature_probability(self, x, mean, std):
        return (1 / (np.sqrt(2 * np.pi) * std)) * \
               np.exp(-((x - mean) ** 2) / (2 * std ** 2))
```

Slide 3: Probability Calculations in Naive Bayes

The core of Naive Bayes lies in its probability calculations, which combine prior probabilities with likelihood estimates. The algorithm uses logarithmic probabilities to prevent numerical underflow when multiplying many small probabilities.

```python
def predict_proba(self, X):
    probabilities = np.zeros((X.shape[0], len(self.classes)))
    
    for i, x in enumerate(X):
        for j, c in enumerate(self.classes):
            # Start with log prior probability
            class_probability = np.log(self.class_probs[c])
            
            # Add log likelihood for each feature
            for feature in range(len(x)):
                params = self.feature_params[feature][c]
                feature_prob = self._calculate_feature_probability(
                    x[feature], 
                    params['mean'], 
                    params['std']
                )
                class_probability += np.log(feature_prob + 1e-10)
            
            probabilities[i, j] = class_probability
    
    # Convert log probabilities to normal probabilities
    probabilities = np.exp(probabilities)
    return probabilities / np.sum(probabilities, axis=1, keepdims=True)
```

Slide 4: Text Classification Example

Text classification represents a common application of Naive Bayes, where the conditional independence assumption works particularly well with word frequencies. This implementation demonstrates document classification using bag-of-words features.

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample text data
documents = [
    "machine learning is fascinating",
    "natural language processing rocks",
    "neural networks are complex",
    "deep learning transforms AI"
]
labels = ['ML', 'NLP', 'ML', 'DL']

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# Train Naive Bayes
nb = NaiveBayes()
nb.fit(X.toarray(), labels)

# Predict new document
new_doc = ["learning about neural networks"]
X_new = vectorizer.transform(new_doc)
predictions = nb.predict_proba(X_new.toarray())
```

Slide 5: Handling Missing Values in Naive Bayes

Missing values require special attention in Naive Bayes since they can affect probability calculations. This implementation demonstrates how to handle missing values through multiple strategies while maintaining the conditional independence assumption.

```python
class RobustNaiveBayes:
    def handle_missing_values(self, X, strategy='mean'):
        X_processed = X.copy()
        
        if strategy == 'mean':
            # Replace missing values with feature means
            for feature in range(X.shape[1]):
                mask = np.isnan(X[:, feature])
                if mask.any():
                    feature_mean = np.nanmean(X[:, feature])
                    X_processed[mask, feature] = feature_mean
        
        elif strategy == 'class_conditional':
            # Replace missing values with class-conditional means
            for c in self.classes:
                class_mask = (self.y == c)
                for feature in range(X.shape[1]):
                    mask = np.isnan(X[class_mask, feature])
                    if mask.any():
                        feature_mean = np.nanmean(X[class_mask, feature])
                        X_processed[class_mask & np.isnan(X[:, feature]), feature] = feature_mean
                        
        return X_processed
```

Slide 6: Laplace Smoothing Implementation

Laplace smoothing prevents zero probabilities in the model by adding a small constant to all counts. This technique is crucial for handling unseen features and maintaining numerical stability in probability calculations.

```python
def apply_laplace_smoothing(self, X, alpha=1.0):
    n_features = X.shape[1]
    
    # Add smoothing to feature parameters
    for feature in range(n_features):
        for c in self.classes:
            feature_values = X[self.y == c, feature]
            count = len(feature_values)
            
            # Update parameters with smoothing
            self.feature_params[feature][c] = {
                'mean': (np.sum(feature_values) + alpha) / (count + alpha * n_features),
                'std': np.std(feature_values) + alpha
            }
    
    # Update class probabilities with smoothing
    n_samples = len(self.y)
    n_classes = len(self.classes)
    for c in self.classes:
        self.class_probs[c] = (np.sum(self.y == c) + alpha) / (n_samples + alpha * n_classes)
```

Slide 7: Real-world Example - Spam Classification

Spam classification demonstrates the practical application of Naive Bayes in email filtering. This implementation includes text preprocessing, feature extraction, and model evaluation using real email data.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Example email dataset
emails = [
    {'text': 'Buy now! Limited offer', 'label': 'spam'},
    {'text': 'Meeting at 3pm tomorrow', 'label': 'ham'},
    {'text': 'Win free prizes now!!!', 'label': 'spam'},
    {'text': 'Project deadline reminder', 'label': 'ham'}
]

# Preprocess and vectorize
def preprocess_emails(emails):
    texts = [email['text'].lower() for email in emails]
    labels = [email['label'] for email in emails]
    
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    return X.toarray(), np.array(labels), vectorizer

X, y, vectorizer = preprocess_emails(emails)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and evaluate
nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
print(classification_report(y_test, predictions))
```

Slide 8: Feature Selection for Naive Bayes

Feature selection improves model performance by identifying the most informative features while maintaining the conditional independence assumption. This implementation uses mutual information to select relevant features.

```python
from sklearn.feature_selection import mutual_info_classif

class NaiveBayesWithFeatureSelection:
    def select_features(self, X, y, k=10):
        # Calculate mutual information between features and target
        mi_scores = mutual_info_classif(X, y)
        
        # Select top k features
        top_features = np.argsort(mi_scores)[-k:]
        
        # Create feature mask
        self.feature_mask = np.zeros(X.shape[1], dtype=bool)
        self.feature_mask[top_features] = True
        
        return X[:, self.feature_mask]
    
    def fit(self, X, y):
        # Select features before training
        X_selected = self.select_features(X, y)
        super().fit(X_selected, y)
    
    def predict(self, X):
        # Apply feature mask before prediction
        X_selected = X[:, self.feature_mask]
        return super().predict(X_selected)
```

Slide 9: Multiclass Classification Implementation

Naive Bayes naturally extends to multiclass problems by calculating probabilities for each class. This implementation shows how to handle multiple classes while maintaining computational efficiency through vectorized operations.

```python
class MulticlassNaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.feature_params = {}
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        # Initialize parameters for each class
        for c in self.classes:
            class_mask = (y == c)
            self.class_probs[c] = np.mean(class_mask)
            
            # Calculate parameters for each feature per class
            class_samples = X[class_mask]
            self.feature_params[c] = {
                'means': np.mean(class_samples, axis=0),
                'vars': np.var(class_samples, axis=0) + 1e-9
            }
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, len(self.classes)))
        
        for i, c in enumerate(self.classes):
            # Vectorized probability calculation
            means = self.feature_params[c]['means']
            vars = self.feature_params[c]['vars']
            
            # Calculate log probabilities
            log_probs = -0.5 * np.sum(
                np.log(2 * np.pi * vars) + 
                (X - means) ** 2 / vars, 
                axis=1
            )
            probs[:, i] = log_probs + np.log(self.class_probs[c])
            
        # Normalize probabilities
        probs = np.exp(probs - np.max(probs, axis=1, keepdims=True))
        return probs / np.sum(probs, axis=1, keepdims=True)
```

Slide 10: Real-world Example - Document Classification

Document classification using Naive Bayes demonstrates its effectiveness in natural language processing tasks. This implementation includes advanced text preprocessing and evaluation metrics.

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Load dataset
categories = ['comp.graphics', 'sci.med', 'rec.sport.baseball']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)

# Preprocess text data
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(newsgroups.data).toarray()
y = newsgroups.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train and evaluate
nb_classifier = MulticlassNaiveBayes()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)

# Calculate and visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=categories,
            yticklabels=categories)
```

Slide 11: Gaussian Naive Bayes Implementation

Gaussian Naive Bayes assumes features follow a normal distribution within each class. This implementation includes probability density function calculations and numerical stability improvements.

```python
class GaussianNaiveBayes:
    def _calculate_gaussian_probability(self, x, mean, var):
        """Calculate Gaussian probability density function"""
        return np.exp(-0.5 * ((x - mean) ** 2) / var) / np.sqrt(2 * np.pi * var)
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.parameters[c] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0) + 1e-9,  # Add small constant for stability
                'prior': len(X_c) / len(X)
            }
    
    def predict(self, X):
        posteriors = []
        
        for x in X:
            class_posteriors = []
            for c in self.classes:
                prior = np.log(self.parameters[c]['prior'])
                mean = self.parameters[c]['mean']
                var = self.parameters[c]['var']
                
                # Sum of log probabilities for numerical stability
                posterior = prior + np.sum(
                    np.log(self._calculate_gaussian_probability(x, mean, var))
                )
                class_posteriors.append(posterior)
            
            # Select class with highest posterior probability
            posteriors.append(self.classes[np.argmax(class_posteriors)])
            
        return np.array(posteriors)
```

Slide 12: Performance Optimization Techniques

This implementation focuses on computational efficiency through vectorization and memory optimization. By avoiding loops and implementing batch processing, we significantly improve processing speed for large datasets.

```python
class OptimizedNaiveBayes:
    def __init__(self, batch_size=1000):
        self.batch_size = batch_size
        
    def _batch_process(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        # Process data in batches
        for i in range(0, n_samples, self.batch_size):
            batch = X[i:i + self.batch_size]
            log_probs = np.zeros((batch.shape[0], len(self.classes)))
            
            # Vectorized computations for each class
            for j, c in enumerate(self.classes):
                params = self.parameters[c]
                log_probs[:, j] = (
                    np.log(params['prior']) +
                    np.sum(
                        -0.5 * np.log(2 * np.pi * params['var']) -
                        0.5 * ((batch - params['mean'])**2) / params['var'],
                        axis=1
                    )
                )
            
            predictions[i:i + self.batch_size] = self.classes[np.argmax(log_probs, axis=1)]
        
        return predictions
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        
        # Pre-allocate memory for parameters
        self.parameters = {
            c: {
                'mean': np.zeros(n_features),
                'var': np.zeros(n_features),
                'prior': 0.0
            } for c in self.classes
        }
        
        # Compute parameters efficiently
        for c in self.classes:
            X_c = X[y == c]
            self.parameters[c]['mean'] = np.mean(X_c, axis=0)
            self.parameters[c]['var'] = np.var(X_c, axis=0) + 1e-9
            self.parameters[c]['prior'] = len(X_c) / len(X)
```

Slide 13: Cross-Validation and Model Evaluation

Comprehensive model evaluation implementation including k-fold cross-validation, performance metrics, and learning curves to assess model stability and generalization.

```python
class ModelEvaluator:
    def __init__(self, model, n_folds=5):
        self.model = model
        self.n_folds = n_folds
        
    def cross_validate(self, X, y):
        fold_size = len(X) // self.n_folds
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for i in range(self.n_folds):
            # Create train-test split for this fold
            test_start = i * fold_size
            test_end = (i + 1) * fold_size
            
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            X_train = np.concatenate([X[:test_start], X[test_end:]])
            y_train = np.concatenate([y[:test_start], y[test_end:]])
            
            # Train and evaluate
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
            metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
            metrics['f1'].append(f1_score(y_test, y_pred, average='weighted'))
        
        return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}
```

Slide 14: Additional Resources

*   "A Tutorial on Naive Bayes Classification" - [https://arxiv.org/abs/1410.5329](https://arxiv.org/abs/1410.5329)
*   "On Discriminative vs. Generative Classifiers: A comparison of logistic regression and naive Bayes" - [https://proceedings.neurips.cc/paper/2001/file/7b7a53e239400a13bd6be6c91c4f6c4e-Paper.pdf](https://proceedings.neurips.cc/paper/2001/file/7b7a53e239400a13bd6be6c91c4f6c4e-Paper.pdf)
*   "Naive Bayes and Text Classification" - Search on Google Scholar for "naive bayes text classification survey"
*   "Feature Selection for Naive Bayes Classification" - [https://www.jmlr.org/papers/volume3/mccallum03a/mccallum03a.pdf](https://www.jmlr.org/papers/volume3/mccallum03a/mccallum03a.pdf)
*   "An Empirical Study of the Naive Bayes Classifier" - Search on Google Scholar for "empirical study naive bayes classifier"

