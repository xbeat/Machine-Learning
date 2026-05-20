## Imbalanced Stratification Techniques for Machine Learning using Python
Slide 1: Imbalanced Classification: An Introduction

Imbalanced classification is a common problem in machine learning where one class significantly outnumbers the other(s). This can lead to biased models that perform poorly on minority classes. Let's explore techniques to address this issue.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate an imbalanced dataset
np.random.seed(42)
majority = np.random.normal(0, 1, (1000, 2))
minority = np.random.normal(3, 1, (100, 2))

plt.scatter(majority[:, 0], majority[:, 1], label='Majority')
plt.scatter(minority[:, 0], minority[:, 1], label='Minority')
plt.legend()
plt.title('Imbalanced Dataset')
plt.show()
```

Slide 2: Random Over-sampling

Random over-sampling is a simple technique that involves randomly duplicating examples from the minority class to balance the dataset. This method can be effective but may lead to overfitting.

```python
from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification

# Create an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Apply random over-sampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = rosություն.fit_resample(X, y)

print(f"Original dataset shape: {np.bincount(y)}")
print(f"Resampled dataset shape: {np.bincount(y_resampled)}")
```

Slide 3: Random Under-sampling

Random under-sampling involves randomly removing examples from the majority class to balance the dataset. This method can be useful when you have a large dataset, but it may discard potentially useful information.

```python
from imblearn.under_sampling import RandomUnderSampler

# Apply random under-sampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

print(f"Original dataset shape: {np.bincount(y)}")
print(f"Resampled dataset shape: {np.bincount(y_resampled)}")
```

Slide 4: SMOTE: Synthetic Minority Over-sampling Technique

SMOTE is a popular technique that creates synthetic examples in the minority class. It works by selecting examples that are close in the feature space, drawing a line between the examples, and drawing a new sample at a point along that line.

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"Original dataset shape: {np.bincount(y)}")
print(f"Resampled dataset shape: {np.bincount(y_resampled)}")

# Visualize SMOTE results
plt.scatter(X_resampled[y_resampled==0][:, 0], X_resampled[y_resampled==0][:, 1], label='Majority')
plt.scatter(X_resampled[y_resampled==1][:, 0], X_resampled[y_resampled==1][:, 1], label='Minority (SMOTE)')
plt.legend()
plt.title('SMOTE Resampled Dataset')
plt.show()
```

Slide 5: ADASYN: Adaptive Synthetic Sampling

ADASYN is similar to SMOTE but focuses on generating samples in the areas where the minority class samples are harder to learn. It does this by generating more synthetic data for minority class samples that are harder to learn.

```python
from imblearn.over_sampling import ADASYN

# Apply ADASYN
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

print(f"Original dataset shape: {np.bincount(y)}")
print(f"Resampled dataset shape: {np.bincount(y_resampled)}")
```

Slide 6: Tomek Links

Tomek links are pairs of nearest neighbors of opposite classes. Removing the majority instance of a Tomek link can help clean the overlapping regions between classes and potentially improve classification performance.

```python
from imblearn.under_sampling import TomekLinks

# Apply Tomek Links
tl = TomekLinks()
X_resampled, y_resampled = tl.fit_resample(X, y)

print(f"Original dataset shape: {np.bincount(y)}")
print(f"Resampled dataset shape: {np.bincount(y_resampled)}")
```

Slide 7: Combining Over-sampling and Under-sampling

We can combine over-sampling and under-sampling techniques to achieve a balanced dataset. This approach can help mitigate the drawbacks of using either technique alone.

```python
from imblearn.combine import SMOTETomek

# Apply SMOTE + Tomek Links
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X, y)

print(f"Original dataset shape: {np.bincount(y)}")
print(f"Resampled dataset shape: {np.bincount(y_resampled)}")
```

Slide 8: Class Weights

Instead of resampling, we can adjust the importance of each class during model training. Many machine learning algorithms in scikit-learn support a 'class\_weight' parameter.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Create a logistic regression model with balanced class weights
clf = LogisticRegression(class_weight='balanced', random_state=42)

# Train the model
clf.fit(X, y)

# Make predictions
y_pred = clf.predict(X)

# Print classification report
print(classification_report(y, y_pred))
```

Slide 9: Ensemble Methods

Ensemble methods like Random Forest or Gradient Boosting can handle imbalanced datasets better than single models. They often perform well without explicit resampling.

```python
from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Train the model
rf.fit(X, y)

# Make predictions
y_pred = rf.predict(X)

# Print classification report
print(classification_report(y, y_pred))
```

Slide 10: Anomaly Detection

For extremely imbalanced datasets, treating the problem as anomaly detection can be effective. One-class SVM is a popular method for this approach.

```python
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix

# Create and train a one-class SVM
ocsvm = OneClassSVM(kernel='rbf', nu=0.1)
ocsvm.fit(X[y==0])  # Train on majority class only

# Predict
y_pred = ocsvm.predict(X)
y_pred = np.where(y_pred == 1, 0, 1)  # Convert predictions to binary

# Print confusion matrix
print(confusion_matrix(y, y_pred))
```

Slide 11: Evaluation Metrics for Imbalanced Classification

When dealing with imbalanced datasets, accuracy can be misleading. Other metrics like precision, recall, F1-score, and ROC AUC are more informative.

```python
from sklearn.metrics import roc_auc_score, average_precision_score

# Assuming we have true labels (y_true) and predicted probabilities (y_prob)
y_true = y
y_prob = clf.predict_proba(X)[:, 1]

# Calculate ROC AUC
roc_auc = roc_auc_score(y_true, y_prob)

# Calculate average precision score
ap_score = average_precision_score(y_true, y_prob)

print(f"ROC AUC: {roc_auc:.3f}")
print(f"Average Precision Score: {ap_score:.3f}")
```

Slide 12: Cross-Validation for Imbalanced Datasets

When performing cross-validation on imbalanced datasets, it's important to maintain the class distribution across folds. Stratified K-Fold can help achieve this.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Create a stratified k-fold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    cv_scores.append(roc_auc_score(y_test, y_prob))

print(f"Mean ROC AUC: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores)*2:.3f})")
```

Slide 13: Real-Life Example: Credit Card Fraud Detection

Credit card fraud detection is a classic example of imbalanced classification. Fraudulent transactions are typically rare compared to legitimate ones.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Load the dataset (you would need to download this)
df = pd.read_csv('creditcard.csv')

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_resampled, y_train_resampled)

# Evaluate the model
y_pred = rf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

Slide 14: Real-Life Example: Rare Disease Diagnosis

Diagnosing rare diseases is another area where imbalanced classification techniques are crucial. The majority of patients will not have the rare disease, creating a significant class imbalance.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Simulate a rare disease dataset
X, y = make_classification(n_samples=10000, n_classes=2, weights=[0.99, 0.01], 
                           n_informative=5, n_redundant=0, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE + Tomek Links
smt = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)

# Train an SVM classifier
svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
svm.fit(X_train_resampled, y_train_resampled)

# Evaluate the model
y_pred = svm.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Slide 15: Additional Resources

For further exploration of imbalanced classification techniques, consider these resources:

1. He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284. ArXiv: [https://arxiv.org/abs/1505.01658](https://arxiv.org/abs/1505.01658)
2. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357. ArXiv: [https://arxiv.org/abs/1106.1813](https://arxiv.org/abs/1106.1813)
3. Lemaitre, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning. Journal of Machine Learning Research, 18(17), 1-5. ArXiv: [https://arxiv.org/abs/1609.06570](https://arxiv.org/abs/1609.06570)

