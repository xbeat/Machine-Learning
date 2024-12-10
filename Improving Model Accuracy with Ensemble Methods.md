## Improving Model Accuracy with Ensemble Methods
Slide 1: Understanding Bagging Fundamentals

Bagging, or Bootstrap Aggregating, combines predictions from multiple models to reduce variance and overfitting. This ensemble method creates diverse training datasets through bootstrap sampling, training independent models and aggregating their predictions through voting or averaging.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

class SimpleBagging:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.estimators = []
    
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]
```

Slide 2: Implementing Core Bagging Methods

The implementation focuses on creating multiple bootstrapped samples and training individual decision trees. Each tree learns from a different subset of data, promoting diversity in the ensemble and improving generalization capabilities.

```python
    def fit(self, X, y):
        self.estimators = []
        for _ in range(self.n_estimators):
            # Create bootstrap sample
            X_sample, y_sample = self.bootstrap_sample(X, y)
            
            # Train decision tree
            tree = DecisionTreeClassifier()
            tree.fit(X_sample, y_sample)
            self.estimators.append(tree)
```

Slide 3: Prediction Aggregation in Bagging

The prediction phase combines individual model outputs through majority voting for classification tasks. This aggregation mechanism helps smooth out individual model errors and produces more robust predictions.

```python
    def predict(self, X):
        predictions = np.array([est.predict(X) for est in self.estimators])
        # Majority voting for classification
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=0, 
            arr=predictions
        )
```

Slide 4: Mathematical Foundation of Bagging

Understanding the mathematical principles behind bagging helps explain its effectiveness in reducing model variance while maintaining the bias level of individual models.

```python
# Mathematical representation of bagging
'''
$$\hat{f}_{bag}(x) = \frac{1}{B}\sum_{b=1}^{B}\hat{f}^{*b}(x)$$

Where:
$$B$$ = number of bootstrap samples
$$\hat{f}^{*b}(x)$$ = prediction of model trained on bootstrap sample b
'''
```

Slide 5: Real-world Application: Credit Risk Assessment

Implementing bagging for credit risk prediction demonstrates its practical application in financial modeling, where robust and accurate predictions are crucial for decision-making.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic credit data
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=15, n_redundant=5,
                          random_state=42)

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

Slide 6: Source Code for Credit Risk Assessment

```python
# Initialize and train bagging classifier
bagging = SimpleBagging(n_estimators=50)
bagging.fit(X_train, y_train)

# Make predictions
y_pred = bagging.predict(X_test)

# Calculate performance metrics
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

Slide 7: Out-of-Bag Error Estimation

Out-of-bag error provides an unbiased estimate of the prediction error without requiring a separate validation set, making it a valuable tool for model evaluation.

```python
class BaggingWithOOB:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.estimators = []
        self.oob_scores = []
    
    def get_oob_samples(self, X, bootstrap_indices):
        all_indices = set(range(len(X)))
        oob_indices = list(all_indices - set(bootstrap_indices))
        return oob_indices
```

Slide 8: Source Code for OOB Implementation

```python
    def fit_with_oob(self, X, y):
        n_samples = X.shape[0]
        oob_predictions = np.zeros((n_samples,))
        oob_counts = np.zeros(n_samples)
        
        for _ in range(self.n_estimators):
            # Bootstrap sample indices
            bootstrap_indices = np.random.choice(
                n_samples, size=n_samples, replace=True
            )
            
            # Get OOB indices
            oob_indices = self.get_oob_samples(X, bootstrap_indices)
            
            # Train model
            tree = DecisionTreeClassifier()
            tree.fit(X[bootstrap_indices], y[bootstrap_indices])
            self.estimators.append(tree)
            
            # OOB predictions
            oob_predictions[oob_indices] += tree.predict(X[oob_indices])
            oob_counts[oob_indices] += 1
        
        # Calculate OOB score
        valid_indices = oob_counts > 0
        oob_predictions[valid_indices] /= oob_counts[valid_indices]
        self.oob_score = accuracy_score(
            y[valid_indices], 
            (oob_predictions[valid_indices] > 0.5).astype(int)
        )
```

Slide 9: Feature Importance Analysis

Bagging enables reliable feature importance estimation by averaging importance scores across multiple models, providing stable rankings that are less sensitive to data perturbations.

```python
def get_feature_importance(bagging_model, feature_names=None):
    importances = np.zeros(len(feature_names))
    for tree in bagging_model.estimators:
        importances += tree.feature_importances_
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importances / len(bagging_model.estimators)
    }).sort_values('importance', ascending=False)
```

Slide 10: Real-world Application: Spam Detection

Implementing bagging for email classification demonstrates its effectiveness in handling high-dimensional text data with complex feature interactions.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Simulated email data
emails = [
    "Special offer! Buy now!", 
    "Meeting scheduled for tomorrow",
    "Get rich quick! Limited time!",
    "Project update: milestone completed"
]
labels = np.array([1, 0, 1, 0])  # 1: spam, 0: not spam

# Feature extraction
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(emails).toarray()
```

Slide 11: Source Code for Spam Detection

```python
# Initialize and train spam detector
spam_detector = SimpleBagging(n_estimators=100)
spam_detector.fit(X, labels)

# New emails for prediction
new_emails = [
    "Congratulations! You've won a prize!",
    "Please review the attached document"
]
X_new = vectorizer.transform(new_emails).toarray()
predictions = spam_detector.predict(X_new)

# Print results
for email, pred in zip(new_emails, predictions):
    print(f"Email: {email}")
    print(f"Prediction: {'Spam' if pred == 1 else 'Not Spam'}\n")
```

Slide 12: Parallel Implementation

Parallel processing significantly improves bagging performance by distributing model training across multiple cores, essential for large-scale applications.

```python
from joblib import Parallel, delayed

class ParallelBagging:
    def __init__(self, n_estimators=10, n_jobs=-1):
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
    
    def _train_estimator(self, X, y):
        X_sample, y_sample = self.bootstrap_sample(X, y)
        tree = DecisionTreeClassifier()
        tree.fit(X_sample, y_sample)
        return tree
```

Slide 13: Source Code for Parallel Implementation

```python
    def fit(self, X, y):
        self.estimators = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_estimator)(X, y)
            for _ in range(self.n_estimators)
        )
        
    def predict(self, X):
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(estimator.predict)(X)
            for estimator in self.estimators
        )
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=np.array(predictions)
        )
```

Slide 14: Handling Model Persistence

Implementing model persistence ensures trained bagging ensembles can be saved and loaded efficiently for production deployment.

```python
import pickle

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Example usage
bagging_model = SimpleBagging(n_estimators=50)
bagging_model.fit(X_train, y_train)
save_model(bagging_model, 'bagging_model.pkl')
```

Slide 15: Additional Resources

*   "Random Forests" by Leo Breiman: [https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
*   "Bagging Predictors" by Leo Breiman: [https://statistics.berkeley.edu/sites/default/files/tech-reports/421.pdf](https://statistics.berkeley.edu/sites/default/files/tech-reports/421.pdf)
*   "A Survey of Bagging Methods" - Search on Google Scholar for comprehensive reviews
*   "Machine Learning with Random Forests and Decision Trees" - Available on various academic databases
*   "Ensemble Methods in Machine Learning" - Search IEEE Xplore for detailed implementations

