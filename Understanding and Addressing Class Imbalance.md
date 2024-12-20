## Understanding and Addressing Class Imbalance

Slide 1: Class Imbalance: An Introduction

Class imbalance occurs when one class in a dataset significantly outnumbers others. This phenomenon is common in real-world scenarios, such as fraud detection or disease diagnosis, where the positive class (minority) represents rare events while the negative class (majority) dominates. Understanding and addressing class imbalance is crucial for developing effective machine learning models.

```python
import matplotlib.pyplot as plt

# Generate imbalanced dataset
np.random.seed(42)
majority = np.random.normal(loc=0, scale=1, size=(1000, 2))
minority = np.random.normal(loc=2, scale=1, size=(100, 2))

# Visualize the imbalanced dataset
plt.figure(figsize=(10, 6))
plt.scatter(majority[:, 0], majority[:, 1], label='Majority Class', alpha=0.6)
plt.scatter(minority[:, 0], minority[:, 1], label='Minority Class', alpha=0.6)
plt.title('Imbalanced Dataset Visualization')
plt.legend()
plt.show()
```

Slide 2: The Impact of Class Imbalance

Class imbalance can significantly affect model performance. Models trained on imbalanced datasets often exhibit bias towards the majority class, leading to poor performance on the minority class. This is particularly problematic when the minority class represents critical events that we aim to predict accurately.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Prepare data
X = np.vstack((majority, minority))
y = np.hstack((np.zeros(1000), np.ones(100)))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

Slide 3: Evaluating Imbalanced Datasets

When dealing with imbalanced datasets, traditional accuracy metrics can be misleading. A model that always predicts the majority class might achieve high accuracy but completely fail to identify the minority class. To accurately assess model performance, we need to use more appropriate metrics.

```python
import seaborn as sns

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Calculate and plot ROC curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

Slide 4: Resampling Techniques: Oversampling

One approach to address class imbalance is oversampling the minority class. Synthetic Minority Over-sampling Technique (SMOTE) is a popular method that creates synthetic examples of the minority class to balance the dataset.

```python
from collections import Counter

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Compare class distribution before and after SMOTE
print("Before SMOTE:", Counter(y_train))
print("After SMOTE:", Counter(y_resampled))

# Visualize resampled data
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='Majority', alpha=0.6)
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='Minority', alpha=0.6)
plt.title('Original Data')
plt.legend()

plt.subplot(122)
plt.scatter(X_resampled[y_resampled == 0][:, 0], X_resampled[y_resampled == 0][:, 1], label='Majority', alpha=0.6)
plt.scatter(X_resampled[y_resampled == 1][:, 0], X_resampled[y_resampled == 1][:, 1], label='Minority', alpha=0.6)
plt.title('SMOTE Resampled Data')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 5: Resampling Techniques: Undersampling

Undersampling is another approach to balance the dataset by reducing the number of instances in the majority class. Random undersampling is a simple method that randomly removes samples from the majority class.

```python

# Apply random undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# Compare class distribution before and after undersampling
print("Before undersampling:", Counter(y_train))
print("After undersampling:", Counter(y_resampled))

# Visualize resampled data
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='Majority', alpha=0.6)
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='Minority', alpha=0.6)
plt.title('Original Data')
plt.legend()

plt.subplot(122)
plt.scatter(X_resampled[y_resampled == 0][:, 0], X_resampled[y_resampled == 0][:, 1], label='Majority', alpha=0.6)
plt.scatter(X_resampled[y_resampled == 1][:, 0], X_resampled[y_resampled == 1][:, 1], label='Minority', alpha=0.6)
plt.title('Undersampled Data')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 6: Class Weighting

Many machine learning algorithms allow for class weighting, which assigns higher importance to the minority class during training. This approach helps the model pay more attention to the underrepresented class without modifying the original dataset.

```python

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

print("Class weights:", class_weight_dict)

# Train a logistic regression model with class weights
weighted_model = LogisticRegression(class_weight=class_weight_dict)
weighted_model.fit(X_train, y_train)

# Evaluate the weighted model
y_pred_weighted = weighted_model.predict(X_test)
print(classification_report(y_test, y_pred_weighted))
```

Slide 7: Ensemble Methods for Imbalanced Data

Ensemble methods, such as Random Forest and Gradient Boosting, can be effective in handling imbalanced datasets. These algorithms combine multiple weak learners to create a strong classifier, often performing well on imbalanced data without explicit resampling.

```python
from sklearn.model_selection import cross_val_score

# Train a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the Random Forest model
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# Perform cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='f1')
print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean F1 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

Slide 8: Real-Life Example: Rare Disease Detection

Consider a dataset of medical records where only 1% of patients have a rare disease. This imbalance can lead to challenges in developing an accurate diagnostic model. Let's simulate this scenario and apply techniques to improve model performance.

```python
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Simulate a medical dataset
np.random.seed(42)
n_samples = 10000
n_features = 5

X = np.random.randn(n_samples, n_features)
y = np.zeros(n_samples)
y[:100] = 1  # 1% positive cases

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with SMOTE and Random Forest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
```

Slide 9: Real-Life Example: Anomaly Detection in Manufacturing

In a manufacturing process, defective products are typically rare. Let's simulate a quality control dataset where only 2% of products are defective and apply techniques to improve defect detection.

```python
from imblearn.combine import SMOTETomek

# Simulate manufacturing quality control data
np.random.seed(42)
n_samples = 5000
n_features = 4

X = np.random.randn(n_samples, n_features)
y = np.zeros(n_samples)
y[:100] = 1  # 2% defective products

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with SMOTETomek and SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote_tomek', SMOTETomek(random_state=42)),
    ('classifier', SVC(kernel='rbf', class_weight='balanced', random_state=42))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
```

Slide 10: Choosing the Right Approach

Selecting the appropriate technique for handling class imbalance depends on various factors, including the nature of the problem, the size of the dataset, and the specific requirements of the application. It's often beneficial to experiment with multiple approaches and compare their performance.

```python

# Define parameter grid
param_grid = {
    'smote__k_neighbors': [3, 5, 7],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None]
}

# Create a grid search object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best F1 score:", grid_search.best_score_)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))
```

Slide 11: Threshold Adjustment

In addition to resampling and algorithm-specific techniques, adjusting the classification threshold can help improve performance on imbalanced datasets. By varying the threshold, we can control the trade-off between precision and recall.

```python

# Get prediction probabilities
y_scores = pipeline.predict_proba(X_test)[:, 1]

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot precision-recall curve
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# Find the optimal threshold
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

plt.axvline(recall[optimal_idx], color='r', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.2f}')
plt.legend()
plt.show()

# Apply the optimal threshold
y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
print(classification_report(y_test, y_pred_optimal))
```

Slide 12: Combining Multiple Techniques

Often, the best approach to handling class imbalance involves combining multiple techniques. This can include using resampling methods, ensemble algorithms, and threshold adjustment together to achieve optimal performance.

```python

# Create a pipeline with SMOTE and Balanced Random Forest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', BalancedRandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Get prediction probabilities
y_scores = pipeline.predict_proba(X_test)[:, 1]

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Apply optimal threshold
y_pred_optimal = (y_scores >= optimal_threshold).astype(int)

print("Classification Report with Combined Techniques:")
print(classification_report(y_test, y_pred_optimal))
```

Slide 13: Monitoring and Adapting to Changing Imbalance

In real-world applications, class imbalance ratios may change over time. Implementing a monitoring system to detect and adapt to these changes is crucial for maintaining model performance. This can involve periodically retraining the model or adjusting the sampling techniques based on the current data distribution.

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class AdaptiveImbalanceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier, imbalance_threshold=0.2):
        self.base_classifier = base_classifier
        self.imbalance_threshold = imbalance_threshold
        self.current_imbalance_ratio = None

    def fit(self, X, y):
        self.current_imbalance_ratio = np.mean(y)
        self.base_classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.base_classifier.predict(X)

    def update(self, X_new, y_new):
        new_imbalance_ratio = np.mean(y_new)
        if abs(new_imbalance_ratio - self.current_imbalance_ratio) > self.imbalance_threshold:
            print("Significant change in imbalance detected. Retraining model...")
            self.fit(X_new, y_new)
        else:
            print("No significant change in imbalance. Continuing with current model.")

# Usage example (pseudocode):
# model = AdaptiveImbalanceClassifier(RandomForestClassifier())
# model.fit(X_initial, y_initial)
# 
# # Periodically:
# new_data = get_new_data()
# model.update(new_data['X'], new_data['y'])
```

Slide 14: Evaluating Long-Term Performance

When dealing with imbalanced datasets, it's important to track model performance over time. This helps identify degradation in minority class prediction and ensures the chosen techniques remain effective as data distributions evolve.

```python
from sklearn.metrics import f1_score

def plot_performance_over_time(y_true_list, y_pred_list, timestamps):
    f1_scores = [f1_score(y_true, y_pred) for y_true, y_pred in zip(y_true_list, y_pred_list)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, f1_scores, marker='o')
    plt.title('Model Performance Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.show()

# Example usage (using dummy data):
np.random.seed(42)
timestamps = pd.date_range(start='2023-01-01', periods=10, freq='M')
y_true_list = [np.random.choice([0, 1], size=100, p=[0.9, 0.1]) for _ in range(10)]
y_pred_list = [np.random.choice([0, 1], size=100, p=[0.85, 0.15]) for _ in range(10)]

plot_performance_over_time(y_true_list, y_pred_list, timestamps)
```

Slide 15: Additional Resources

For those interested in delving deeper into class imbalance and related techniques, consider exploring these resources:

1. "Learning from Imbalanced Data" by Haibo He and Edwardo A. Garcia (2009) ArXiv: [https://arxiv.org/abs/1505.01658](https://arxiv.org/abs/1505.01658)
2. "A Survey of Predictive Modelling under Imbalanced Distributions" by Paula Branco, Luís Torgo, and Rita P. Ribeiro (2016) ArXiv: [https://arxiv.org/abs/1505.01658](https://arxiv.org/abs/1505.01658)
3. "SMOTE: Synthetic Minority Over-sampling Technique" by Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, and W. Philip Kegelmeyer (2002) Journal of Artificial Intelligence Research
4. "Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning" by Guillaume Lemaître, Fernando Nogueira, and Christos K. Aridas (2017) Journal of Machine Learning Research

These resources provide in-depth discussions on class imbalance, its effects on machine learning models, and various techniques to address the challenges it presents.


