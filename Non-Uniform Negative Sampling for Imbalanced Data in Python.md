## Non-Uniform Negative Sampling for Imbalanced Data in Python
Slide 1: Introduction to Non-Uniform Negative Sampling

Non-uniform negative sampling is a technique used to address imbalanced datasets in machine learning. It involves selectively choosing negative examples to improve model performance. This approach is particularly useful when dealing with binary classification problems where one class significantly outnumbers the other.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate imbalanced dataset
np.random.seed(42)
positive_samples = np.random.randn(100, 2) + np.array([2, 2])
negative_samples = np.random.randn(1000, 2)

plt.scatter(positive_samples[:, 0], positive_samples[:, 1], c='r', label='Positive')
plt.scatter(negative_samples[:, 0], negative_samples[:, 1], c='b', label='Negative')
plt.legend()
plt.title('Imbalanced Dataset')
plt.show()
```

Slide 2: The Problem with Imbalanced Datasets

Imbalanced datasets can lead to biased models that perform poorly on minority classes. Traditional sampling methods may not effectively capture the distribution of the minority class, resulting in suboptimal performance. Non-uniform negative sampling addresses this issue by strategically selecting negative examples.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# Combine positive and negative samples
X = np.vstack((positive_samples, negative_samples))
y = np.hstack((np.ones(len(positive_samples)), np.zeros(len(negative_samples))))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

Slide 3: Non-Uniform Negative Sampling: The Concept

Non-uniform negative sampling involves selecting negative examples based on their relevance or difficulty. This approach aims to focus the model's attention on more informative negative samples, improving its ability to distinguish between classes. The sampling probability is often based on the similarity or distance between positive and negative examples.

```python
def non_uniform_sampling(positive_samples, negative_samples, num_samples):
    distances = np.sum(np.abs(negative_samples[:, np.newaxis] - positive_samples), axis=2)
    probabilities = 1 / (distances.min(axis=1) + 1e-5)
    probabilities /= probabilities.sum()
    
    sampled_indices = np.random.choice(
        len(negative_samples), 
        size=num_samples, 
        replace=False, 
        p=probabilities
    )
    return negative_samples[sampled_indices]

# Sample negative examples
sampled_negatives = non_uniform_sampling(positive_samples, negative_samples, 100)

plt.scatter(positive_samples[:, 0], positive_samples[:, 1], c='r', label='Positive')
plt.scatter(sampled_negatives[:, 0], sampled_negatives[:, 1], c='b', label='Sampled Negative')
plt.legend()
plt.title('Non-Uniform Negative Sampling')
plt.show()
```

Slide 4: Implementing Non-Uniform Negative Sampling

To implement non-uniform negative sampling, we first calculate the distances between positive and negative samples. We then compute sampling probabilities based on these distances, giving higher probabilities to negative samples that are closer to positive ones. Finally, we use these probabilities to sample a subset of negative examples.

```python
def non_uniform_sampling_with_probabilities(positive_samples, negative_samples, num_samples):
    distances = np.sum(np.abs(negative_samples[:, np.newaxis] - positive_samples), axis=2)
    probabilities = 1 / (distances.min(axis=1) + 1e-5)
    probabilities /= probabilities.sum()
    
    sampled_indices = np.random.choice(
        len(negative_samples), 
        size=num_samples, 
        replace=False, 
        p=probabilities
    )
    return negative_samples[sampled_indices], probabilities[sampled_indices]

sampled_negatives, sample_probabilities = non_uniform_sampling_with_probabilities(
    positive_samples, negative_samples, 100
)

plt.scatter(sampled_negatives[:, 0], sampled_negatives[:, 1], 
            c=sample_probabilities, cmap='viridis', s=50)
plt.colorbar(label='Sampling Probability')
plt.title('Non-Uniform Negative Sampling Probabilities')
plt.show()
```

Slide 5: Training with Non-Uniform Negative Sampling

When training a model with non-uniform negative sampling, we combine the positive samples with the sampled negative examples. This creates a more balanced dataset that focuses on the most informative negative samples. We can then train our model on this new dataset.

```python
# Combine positive samples with sampled negative samples
X_sampled = np.vstack((positive_samples, sampled_negatives))
y_sampled = np.hstack((np.ones(len(positive_samples)), np.zeros(len(sampled_negatives))))

# Split the sampled data
X_train_sampled, X_test_sampled, y_train_sampled, y_test_sampled = train_test_split(
    X_sampled, y_sampled, test_size=0.2, random_state=42
)

# Train a new model on the sampled data
model_sampled = LogisticRegression()
model_sampled.fit(X_train_sampled, y_train_sampled)

# Evaluate the new model
y_pred_sampled = model_sampled.predict(X_test)
print(classification_report(y_test, y_pred_sampled))
```

Slide 6: Comparing Results: Before and After Sampling

To understand the impact of non-uniform negative sampling, we compare the performance of models trained on the original imbalanced dataset and the sampled dataset. This comparison helps us assess the effectiveness of the sampling technique in improving classification performance.

```python
from sklearn.metrics import roc_curve, auc

# Original model
y_scores = model.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Sampled model
y_scores_sampled = model_sampled.decision_function(X_test)
fpr_sampled, tpr_sampled, _ = roc_curve(y_test, y_scores_sampled)
roc_auc_sampled = auc(fpr_sampled, tpr_sampled)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Original ROC (AUC = {roc_auc:.2f})')
plt.plot(fpr_sampled, tpr_sampled, color='green', lw=2, 
         label=f'Sampled ROC (AUC = {roc_auc_sampled:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

Slide 7: Real-Life Example: Spam Detection

In email spam detection, legitimate emails (ham) often outnumber spam emails. Non-uniform negative sampling can be used to create a more balanced training set by selecting the most relevant non-spam emails. This approach helps improve the model's ability to distinguish between spam and legitimate emails.

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Simulate a spam detection dataset
np.random.seed(42)
emails = [
    "Buy now! Limited offer!",
    "Meeting at 3 PM",
    "Claim your prize today!",
    "Project deadline reminder",
    "Free gift waiting for you!",
    "Lunch with the team tomorrow",
    # ... (add more examples)
]
labels = [1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for ham

# Create a DataFrame
df = pd.DataFrame({'email': emails, 'label': labels})

# Vectorize the emails
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['email'])
y = df['label']

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, y)

# Predict on new emails
new_emails = [
    "Great discount on watches!",
    "Team meeting rescheduled to 4 PM"
]
X_new = vectorizer.transform(new_emails)
predictions = clf.predict(X_new)

for email, pred in zip(new_emails, predictions):
    print(f"Email: {email}")
    print(f"Prediction: {'Spam' if pred == 1 else 'Ham'}\n")
```

Slide 8: Real-Life Example: Anomaly Detection in Sensor Data

Anomaly detection in sensor data often deals with imbalanced datasets, where normal readings vastly outnumber anomalies. Non-uniform negative sampling can be applied to select a subset of normal readings that are most similar to known anomalies, improving the model's ability to detect subtle deviations.

```python
from sklearn.ensemble import IsolationForest

# Simulate sensor data
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 2))
anomalies = np.random.normal(3, 0.5, (50, 2))

# Combine normal data and anomalies
X = np.vstack((normal_data, anomalies))
y = np.hstack((np.zeros(len(normal_data)), np.ones(len(anomalies))))

# Apply non-uniform sampling to normal data
sampled_normal = non_uniform_sampling(anomalies, normal_data, 100)

# Combine sampled normal data with anomalies
X_sampled = np.vstack((sampled_normal, anomalies))

# Train an Isolation Forest
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X_sampled)

# Generate a grid for visualization
xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the results
plt.contourf(xx, yy, Z, cmap=plt.cm.YlOrRd)
plt.scatter(normal_data[:, 0], normal_data[:, 1], c='blue', label='Normal')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label='Anomaly')
plt.scatter(sampled_normal[:, 0], sampled_normal[:, 1], c='green', label='Sampled Normal')
plt.legend()
plt.title('Anomaly Detection with Non-Uniform Sampling')
plt.show()
```

Slide 9: Challenges and Considerations

While non-uniform negative sampling can be effective, it's important to consider potential challenges. These may include increased computational complexity, the risk of overfitting to sampled negatives, and the need for careful selection of sampling parameters. It's crucial to validate the approach on a held-out test set and compare it with other techniques.

```python
def evaluate_sampling_approaches(X, y, n_iterations=10):
    results = {'No Sampling': [], 'Uniform Sampling': [], 'Non-Uniform Sampling': []}
    
    for _ in range(n_iterations):
        # No sampling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        results['No Sampling'].append(model.score(X_test, y_test))
        
        # Uniform sampling
        negative_indices = np.where(y_train == 0)[0]
        sampled_negative_indices = np.random.choice(negative_indices, size=sum(y_train == 1), replace=False)
        X_train_uniform = np.vstack((X_train[y_train == 1], X_train[sampled_negative_indices]))
        y_train_uniform = np.hstack((np.ones(sum(y_train == 1)), np.zeros(sum(y_train == 1))))
        model_uniform = LogisticRegression()
        model_uniform.fit(X_train_uniform, y_train_uniform)
        results['Uniform Sampling'].append(model_uniform.score(X_test, y_test))
        
        # Non-uniform sampling
        X_train_non_uniform = np.vstack((
            X_train[y_train == 1],
            non_uniform_sampling(X_train[y_train == 1], X_train[y_train == 0], sum(y_train == 1))
        ))
        y_train_non_uniform = np.hstack((np.ones(sum(y_train == 1)), np.zeros(sum(y_train == 1))))
        model_non_uniform = LogisticRegression()
        model_non_uniform.fit(X_train_non_uniform, y_train_non_uniform)
        results['Non-Uniform Sampling'].append(model_non_uniform.score(X_test, y_test))
    
    return results

evaluation_results = evaluate_sampling_approaches(X, y)

plt.boxplot([evaluation_results[key] for key in evaluation_results.keys()])
plt.xticks(range(1, 4), evaluation_results.keys())
plt.ylabel('Accuracy')
plt.title('Comparison of Sampling Approaches')
plt.show()
```

Slide 10: Hyperparameter Tuning for Non-Uniform Sampling

Optimizing the non-uniform sampling process often involves tuning hyperparameters such as the number of samples to select and the distance metric used. Grid search or random search can be employed to find the best combination of parameters for a given problem.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

class NonUniformSamplingClassifier:
    def __init__(self, base_estimator, n_samples, distance_metric='euclidean'):
        self.base_estimator = base_estimator
        self.n_samples = n_samples
        self.distance_metric = distance_metric
    
    def fit(self, X, y):
        positive_samples = X[y == 1]
        negative_samples = X[y == 0]
        
        # Calculate distances and probabilities
        distances = self._calculate_distances(negative_samples, positive_samples)
        probabilities = self._calculate_probabilities(distances)
        
        # Sample negative examples
        sampled_indices = np.random.choice(
            len(negative_samples), 
            size=min(self.n_samples, len(negative_samples)), 
            replace=False, 
            p=probabilities
        )
        
        # Combine samples and train base estimator
        X_sampled = np.vstack((positive_samples, negative_samples[sampled_indices]))
        y_sampled = np.hstack((np.ones(len(positive_samples)), np.zeros(len(sampled_indices))))
        self.base_estimator.fit(X_sampled, y_sampled)
    
    def predict(self, X):
        return self.base_estimator.predict(X)
    
    def _calculate_distances(self, negative_samples, positive_samples):
        if self.distance_metric == 'euclidean':
            return np.sum((negative_samples[:, np.newaxis] - positive_samples) ** 2, axis=2)
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(negative_samples[:, np.newaxis] - positive_samples), axis=2)
    
    def _calculate_probabilities(self, distances):
        probabilities = 1 / (distances.min(axis=1) + 1e-5)
        return probabilities / probabilities.sum()

# Example usage
clf = NonUniformSamplingClassifier(LogisticRegression(), n_samples=100)
param_dist = {"n_samples": randint(50, 200), "distance_metric": ['euclidean', 'manhattan']}
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, cv=5)
random_search.fit(X, y)

print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)
```

Slide 11: Handling High-Dimensional Data

When dealing with high-dimensional data, non-uniform negative sampling can become computationally expensive. Dimensionality reduction techniques can be applied before sampling to improve efficiency while maintaining the effectiveness of the sampling process.

```python
from sklearn.decomposition import PCA

def non_uniform_sampling_high_dim(positive_samples, negative_samples, num_samples, n_components=10):
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    positive_reduced = pca.fit_transform(positive_samples)
    negative_reduced = pca.transform(negative_samples)
    
    # Calculate distances in reduced space
    distances = np.sum((negative_reduced[:, np.newaxis] - positive_reduced) ** 2, axis=2)
    
    # Calculate sampling probabilities
    probabilities = 1 / (distances.min(axis=1) + 1e-5)
    probabilities /= probabilities.sum()
    
    # Sample negative examples
    sampled_indices = np.random.choice(
        len(negative_samples), 
        size=num_samples, 
        replace=False, 
        p=probabilities
    )
    
    return negative_samples[sampled_indices]

# Example usage
high_dim_positive = np.random.randn(100, 1000)  # 100 samples, 1000 features
high_dim_negative = np.random.randn(1000, 1000)  # 1000 samples, 1000 features

sampled_negatives = non_uniform_sampling_high_dim(high_dim_positive, high_dim_negative, 100)
print("Shape of sampled negatives:", sampled_negatives.shape)
```

Slide 12: Adaptive Non-Uniform Sampling

Adaptive non-uniform sampling adjusts the sampling strategy during training based on the model's performance. This approach can lead to more effective sampling by focusing on the most challenging negative examples as the model improves.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

class AdaptiveNonUniformSampler(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, n_samples, n_iterations=5):
        self.base_estimator = base_estimator
        self.n_samples = n_samples
        self.n_iterations = n_iterations
    
    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        for _ in range(self.n_iterations):
            # Sample negative examples
            negative_samples = X_train[y_train == 0]
            positive_samples = X_train[y_train == 1]
            
            probabilities = self._calculate_probabilities(negative_samples, positive_samples)
            sampled_indices = np.random.choice(
                len(negative_samples), 
                size=min(self.n_samples, len(negative_samples)), 
                replace=False, 
                p=probabilities
            )
            
            # Train the model
            X_sampled = np.vstack((positive_samples, negative_samples[sampled_indices]))
            y_sampled = np.hstack((np.ones(len(positive_samples)), np.zeros(len(sampled_indices))))
            self.base_estimator.fit(X_sampled, y_sampled)
            
            # Update sampling probabilities based on model performance
            y_pred = self.base_estimator.predict(X_val)
            incorrect_indices = np.where(y_pred != y_val)[0]
            if len(incorrect_indices) > 0:
                probabilities[incorrect_indices] *= 1.5
                probabilities /= probabilities.sum()
        
        return self
    
    def predict(self, X):
        return self.base_estimator.predict(X)
    
    def _calculate_probabilities(self, negative_samples, positive_samples):
        distances = np.sum((negative_samples[:, np.newaxis] - positive_samples) ** 2, axis=2)
        probabilities = 1 / (distances.min(axis=1) + 1e-5)
        return probabilities / probabilities.sum()

# Example usage
adaptive_sampler = AdaptiveNonUniformSampler(LogisticRegression(), n_samples=100)
adaptive_sampler.fit(X, y)
y_pred = adaptive_sampler.predict(X_test)
print("Accuracy:", np.mean(y_pred == y_test))
```

Slide 13: Combining Non-Uniform Sampling with Other Techniques

Non-uniform negative sampling can be combined with other techniques for handling imbalanced data, such as oversampling the minority class or using ensemble methods. This hybrid approach can lead to more robust models that perform well on imbalanced datasets.

```python
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

def hybrid_sampling_approach(X, y, n_samples):
    # Step 1: Non-uniform negative sampling
    positive_samples = X[y == 1]
    negative_samples = X[y == 0]
    sampled_negatives = non_uniform_sampling(positive_samples, negative_samples, n_samples)
    
    # Step 2: Combine with positive samples
    X_sampled = np.vstack((positive_samples, sampled_negatives))
    y_sampled = np.hstack((np.ones(len(positive_samples)), np.zeros(len(sampled_negatives))))
    
    # Step 3: Apply SMOTE to oversample the minority class
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_sampled, y_sampled)
    
    # Step 4: Train a BalancedRandomForestClassifier
    clf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_resampled, y_resampled)
    
    return clf

# Example usage
hybrid_model = hybrid_sampling_approach(X, y, n_samples=100)
y_pred = hybrid_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
```

Slide 14: Evaluating Non-Uniform Sampling Performance

To assess the effectiveness of non-uniform negative sampling, it's important to use appropriate evaluation metrics for imbalanced datasets. Metrics such as precision, recall, F1-score, and the area under the Precision-Recall curve (AUC-PR) provide a more comprehensive view of model performance than accuracy alone.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def evaluate_sampling_methods(X, y, X_test, y_test):
    # No sampling
    clf_no_sampling = LogisticRegression()
    clf_no_sampling.fit(X, y)
    y_pred_no_sampling = clf_no_sampling.predict_proba(X_test)[:, 1]
    
    # Uniform sampling
    negative_samples = X[y == 0]
    positive_samples = X[y == 1]
    uniform_sampled_negatives = negative_samples[np.random.choice(len(negative_samples), len(positive_samples), replace=False)]
    X_uniform = np.vstack((positive_samples, uniform_sampled_negatives))
    y_uniform = np.hstack((np.ones(len(positive_samples)), np.zeros(len(uniform_sampled_negatives))))
    clf_uniform = LogisticRegression()
    clf_uniform.fit(X_uniform, y_uniform)
    y_pred_uniform = clf_uniform.predict_proba(X_test)[:, 1]
    
    # Non-uniform sampling
    non_uniform_sampled_negatives = non_uniform_sampling(positive_samples, negative_samples, len(positive_samples))
    X_non_uniform = np.vstack((positive_samples, non_uniform_sampled_negatives))
    y_non_uniform = np.hstack((np.ones(len(positive_samples)), np.zeros(len(non_uniform_sampled_negatives))))
    clf_non_uniform = LogisticRegression()
    clf_non_uniform.fit(X_non_uniform, y_non_uniform)
    y_pred_non_uniform = clf_non_uniform.predict_proba(X_test)[:, 1]
    
    # Calculate Precision-Recall curves
    precision_no_sampling, recall_no_sampling, _ = precision_recall_curve(y_test, y_pred_no_sampling)
    precision_uniform, recall_uniform, _ = precision_recall_curve(y_test, y_pred_uniform)
    precision_non_uniform, recall_non_uniform, _ = precision_recall_curve(y_test, y_pred_non_uniform)
    
    # Calculate average precision scores
    ap_no_sampling = average_precision_score(y_test, y_pred_no_sampling)
    ap_uniform = average_precision_score(y_test, y_pred_uniform)
    ap_non_uniform = average_precision_score(y_test, y_pred_non_uniform)
    
    # Plot Precision-Recall curves
    plt.figure(figsize=(10, 6))
    plt.plot(recall_no_sampling, precision_no_sampling, label=f'No Sampling (AP={ap_no_sampling:.2f})')
    plt.plot(recall_uniform, precision_uniform, label=f'Uniform Sampling (AP={ap_uniform:.2f})')
    plt.plot(recall_non_uniform, precision_non_uniform, label=f'Non-Uniform Sampling (AP={ap_non_uniform:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Different Sampling Methods')
    plt.legend()
    plt.show()

# Example usage
evaluate_sampling_methods(X, y, X_test, y_test)
```

Slide 15: Additional Resources

For those interested in diving deeper into non-uniform negative sampling and related techniques for handling imbalanced data, here are some valuable resources:

1. "Learning from Imbalanced Data" by He and Garcia (2009) ArXiv: [https://arxiv.org/abs/1505.01658](https://arxiv.org/abs/1505.01658)
2. "SMOTE: Synthetic Minority Over-sampling Technique" by Chawla et al. (2002) ArXiv: [https://arxiv.org/abs/1106.1813](https://arxiv.org/abs/1106.1813)
3. "A Survey of Predictive Modelling under Imbalanced Distributions" by Branco et al. (2016) ArXiv: [https://arxiv.org/abs/1505.01658](https://arxiv.org/abs/1505.01658)

These papers provide comprehensive overviews of various techniques for dealing with imbalanced datasets, including non-uniform sampling approaches and their applications in different domains.

