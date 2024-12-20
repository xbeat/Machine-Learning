## Avoiding Pitfalls of Random Splitting in Machine Learning
Slide 1: Random Splitting in Machine Learning: A Cautionary Tale

Random splitting of datasets is a common practice in machine learning, but it can lead to unexpected and potentially severe consequences if not done carefully. This presentation explores the pitfalls of random splitting and offers solutions to mitigate its risks.

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Generate a sample dataset
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

# Perform random splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
```

Slide 2: The Illusion of Independence

Random splitting assumes that data points are independent and identically distributed (i.i.d.). However, real-world datasets often violate this assumption, leading to biased model performance estimates.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Create a time series dataset
dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
values = np.cumsum(np.random.randn(len(dates))) + 100
df = pd.DataFrame({'date': dates, 'value': values})

# Plot the time series
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['value'])
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Perform random splitting
train, test = train_test_split(df, test_size=0.2, random_state=42)

print("Train set date range:", train['date'].min(), "to", train['date'].max())
print("Test set date range:", test['date'].min(), "to", test['date'].max())
```

Slide 3: Data Leakage: The Silent Killer

Random splitting can inadvertently introduce data leakage, where information from the test set influences the training process. This can lead to overly optimistic performance estimates and poor generalization.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create a dataset with a unique identifier
df = pd.DataFrame({
    'id': range(1000),
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'target': np.random.randint(0, 2, 1000)
})

# Perform random splitting
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Scale features using information from both train and test sets (incorrect!)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['feature1', 'feature2']])

print("Data leakage: Test set information used in preprocessing")
```

Slide 4: Imbalanced Classes: A Recipe for Disaster

Random splitting can exacerbate class imbalance issues, especially in smaller datasets. This can lead to models that perform poorly on minority classes or fail to learn them altogether.

```python
from sklearn.datasets import make_classification
from collections import Counter

# Generate an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Perform random splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Original class distribution:", Counter(y))
print("Train set class distribution:", Counter(y_train))
print("Test set class distribution:", Counter(y_test))

# Visualize class imbalance
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.pie(Counter(y_train).values(), labels=Counter(y_train).keys(), autopct='%1.1f%%')
plt.title('Train Set Class Distribution')
plt.subplot(1, 2, 2)
plt.pie(Counter(y_test).values(), labels=Counter(y_test).keys(), autopct='%1.1f%%')
plt.title('Test Set Class Distribution')
plt.show()
```

Slide 5: Real-Life Example: Medical Diagnosis

In a medical diagnosis scenario, random splitting can lead to biased model performance and potentially dangerous outcomes. Consider a dataset of patient records for disease detection:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Create a sample medical dataset
np.random.seed(42)
n_samples = 1000
age = np.random.randint(18, 90, n_samples)
gender = np.random.choice(['M', 'F'], n_samples)
symptom1 = np.random.randint(0, 10, n_samples)
symptom2 = np.random.randint(0, 10, n_samples)
disease = (symptom1 + symptom2 > 10).astype(int)

df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'symptom1': symptom1,
    'symptom2': symptom2,
    'disease': disease
})

# Perform random splitting
train, test = train_test_split(df, test_size=0.2, random_state=42)

print("Train set age range:", train['age'].min(), "to", train['age'].max())
print("Test set age range:", test['age'].min(), "to", test['age'].max())
print("\nTrain set gender distribution:")
print(train['gender'].value_counts(normalize=True))
print("\nTest set gender distribution:")
print(test['gender'].value_counts(normalize=True))
```

Slide 6: Stratified Splitting: A Step in the Right Direction

Stratified splitting helps maintain the proportion of classes in both train and test sets, addressing some issues of random splitting for classification tasks.

```python
from sklearn.model_selection import train_test_split

# Generate an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Perform stratified splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("Original class distribution:", Counter(y))
print("Train set class distribution:", Counter(y_train))
print("Test set class distribution:", Counter(y_test))

# Visualize stratified split results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.pie(Counter(y_train).values(), labels=Counter(y_train).keys(), autopct='%1.1f%%')
plt.title('Train Set Class Distribution (Stratified)')
plt.subplot(1, 2, 2)
plt.pie(Counter(y_test).values(), labels=Counter(y_test).keys(), autopct='%1.1f%%')
plt.title('Test Set Class Distribution (Stratified)')
plt.show()
```

Slide 7: Time-Based Splitting: Respecting Temporal Order

For time series data, time-based splitting preserves the temporal structure and avoids using future information to predict the past.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Create a time series dataset
dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
values = np.cumsum(np.random.randn(len(dates))) + 100
df = pd.DataFrame({'date': dates, 'value': values})

# Perform time-based splitting
split_date = '2022-10-01'
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]

# Visualize the split
plt.figure(figsize=(12, 6))
plt.plot(train['date'], train['value'], label='Train')
plt.plot(test['date'], test['value'], label='Test')
plt.axvline(x=pd.to_datetime(split_date), color='r', linestyle='--', label='Split Point')
plt.title('Time-Based Splitting')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

print("Train set date range:", train['date'].min(), "to", train['date'].max())
print("Test set date range:", test['date'].min(), "to", test['date'].max())
```

Slide 8: Group-Based Splitting: Preserving Data Integrity

When dealing with grouped data (e.g., multiple samples per patient), group-based splitting ensures that all samples from a group are in either the train or test set, preventing data leakage.

```python
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Create a sample dataset with grouped data
np.random.seed(42)
n_patients = 100
n_samples_per_patient = 5

patient_ids = np.repeat(range(n_patients), n_samples_per_patient)
feature1 = np.random.randn(n_patients * n_samples_per_patient)
feature2 = np.random.randn(n_patients * n_samples_per_patient)
target = np.random.randint(0, 2, n_patients * n_samples_per_patient)

df = pd.DataFrame({
    'patient_id': patient_ids,
    'feature1': feature1,
    'feature2': feature2,
    'target': target
})

# Perform group-based splitting
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['patient_id']))

train = df.iloc[train_idx]
test = df.iloc[test_idx]

print("Number of unique patients in train:", train['patient_id'].nunique())
print("Number of unique patients in test:", test['patient_id'].nunique())
print("Any patient in both train and test?", 
      any(set(train['patient_id']) & set(test['patient_id'])))
```

Slide 9: Cross-Validation: A More Robust Evaluation

Cross-validation provides a more reliable estimate of model performance by using multiple train-test splits. However, it's crucial to apply the appropriate splitting strategy within the cross-validation process.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform cross-validation
model = LogisticRegression()
cv_scores = cross_val_score(model, X_scaled, y, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Standard deviation of CV scores:", cv_scores.std())

# Visualize cross-validation results
plt.figure(figsize=(8, 6))
plt.boxplot(cv_scores)
plt.title('Cross-Validation Scores Distribution')
plt.ylabel('Accuracy')
plt.show()
```

Slide 10: Nested Cross-Validation: Unbiased Model Selection

Nested cross-validation separates model selection from performance estimation, providing an unbiased estimate of the model's generalization ability.

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)

# Define the parameter grid for SVM
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# Outer cross-validation loop
outer_cv = 5
inner_cv = 3

# Perform nested cross-validation
cv_scores = cross_val_score(
    GridSearchCV(SVC(), param_grid, cv=inner_cv),
    X, y, cv=outer_cv, scoring='accuracy'
)

print("Nested CV scores:", cv_scores)
print("Mean nested CV score:", cv_scores.mean())
print("Standard deviation of nested CV scores:", cv_scores.std())

# Visualize nested CV results
plt.figure(figsize=(8, 6))
plt.boxplot(cv_scores)
plt.title('Nested Cross-Validation Scores Distribution')
plt.ylabel('Accuracy')
plt.show()
```

Slide 11: Real-Life Example: Customer Churn Prediction

In a customer churn prediction scenario, random splitting can lead to biased results due to temporal dependencies and group-based structures. Let's explore a more appropriate splitting strategy:

```python
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Create a sample customer churn dataset
np.random.seed(42)
n_customers = 1000
dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
customer_ids = np.random.randint(1, 101, n_customers)
purchase_amounts = np.random.exponential(scale=50, size=n_customers)
days_since_last_purchase = np.random.randint(1, 365, n_customers)
churn = (days_since_last_purchase > 180).astype(int)

df = pd.DataFrame({
    'date': np.random.choice(dates, n_customers),
    'customer_id': customer_ids,
    'purchase_amount': purchase_amounts,
    'days_since_last_purchase': days_since_last_purchase,
    'churn': churn
})

# Sort the dataset by date
df = df.sort_values('date')

# Prepare features and target
X = df[['purchase_amount', 'days_since_last_purchase']]
y = df['churn']

# Perform time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
model = RandomForestClassifier(random_state=42)

cv_scores = []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv_scores.append(accuracy_score(y_test, y_pred))

print("Time series CV scores:", cv_scores)
print("Mean CV score:", np.mean(cv_scores))
print("Standard deviation of CV scores:", np.std(cv_scores))

# Visualize time series CV results
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o')
plt.title('Time Series Cross-Validation Scores')
plt.xlabel('CV Fold')
plt.ylabel('Accuracy')
plt.show()
```

Slide 12: Data Augmentation: Mitigating Splitting Issues

Data augmentation can help address some of the issues associated with random splitting by increasing dataset size and diversity. This technique is particularly useful for image and text data.

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create a sample 28x28 image
image = np.random.randint(0, 255, (28, 28, 1), dtype=np.uint8)

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generate augmented images
aug_iter = datagen.flow(image.reshape((1, 28, 28, 1)), batch_size=1)

# Display original and augmented images
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
axes[0].imshow(image.squeeze(), cmap='gray')
axes[0].set_title('Original')
for i in range(4):
    axes[i+1].imshow(next(aug_iter)[0].squeeze(), cmap='gray')
    axes[i+1].set_title(f'Augmented {i+1}')
plt.tight_layout()
plt.show()
```

Slide 13: Ensemble Methods: Leveraging Multiple Splits

Ensemble methods can help mitigate the impact of a single random split by combining predictions from models trained on different data subsets.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Create a BaggingClassifier
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=10,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    bootstrap_features=True,
    random_state=42
)

# Perform cross-validation
cv_scores = cross_val_score(bagging, X, y, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Standard deviation of CV scores:", cv_scores.std())

# Visualize individual tree predictions vs ensemble prediction
bagging.fit(X, y)
tree_preds = np.array([tree.predict(X) for tree in bagging.estimators_])
ensemble_pred = bagging.predict(X)

plt.figure(figsize=(10, 6))
plt.imshow(tree_preds, aspect='auto', cmap='binary', alpha=0.5)
plt.plot(ensemble_pred, 'r-', linewidth=2, label='Ensemble prediction')
plt.title('Individual Tree Predictions vs Ensemble Prediction')
plt.xlabel('Samples')
plt.ylabel('Trees')
plt.legend()
plt.show()
```

Slide 14: Best Practices for Dataset Splitting

To minimize the risks associated with random splitting, follow these best practices:

1. Understand your data: Identify temporal dependencies, group structures, and class distributions.
2. Choose appropriate splitting strategies: Use time-based, group-based, or stratified splitting when applicable.
3. Implement cross-validation: Use techniques like k-fold or time series cross-validation for more robust evaluation.
4. Consider data augmentation: Increase dataset size and diversity to reduce the impact of splitting.
5. Use ensemble methods: Combine predictions from multiple models trained on different data subsets.
6. Monitor and validate: Regularly check for data leakage and biases in your train-test splits.

```python
# Pseudocode for a robust splitting and evaluation pipeline

def robust_split_and_evaluate(data, target, splitting_strategy, model, cv_method):
    # Preprocess the data
    preprocessed_data = preprocess(data)
    
    # Perform appropriate splitting
    if splitting_strategy == 'time_based':
        splits = time_based_split(preprocessed_data)
    elif splitting_strategy == 'group_based':
        splits = group_based_split(preprocessed_data)
    elif splitting_strategy == 'stratified':
        splits = stratified_split(preprocessed_data, target)
    
    # Perform cross-validation
    cv_scores = cross_validate(model, splits, cv_method)
    
    # Evaluate final model on hold-out test set
    final_model = train_model(model, splits['train'])
    test_score = evaluate_model(final_model, splits['test'])
    
    return cv_scores, test_score
```

Slide 15: Additional Resources

For more information on dataset splitting and its impacts on machine learning models, consider the following resources:

1. "A Survey on Data Collection for Machine Learning: A Big Data - AI Integration Perspective" (arXiv:1811.03402) URL: [https://arxiv.org/abs/1811.03402](https://arxiv.org/abs/1811.03402)
2. "A Survey of Cross-Validation Procedures for Model Selection" (arXiv:0907.4728) URL: [https://arxiv.org/abs/0907.4728](https://arxiv.org/abs/0907.4728)
3. "Pitfalls of Data Splitting for Machine Learning Model Evaluation" (arXiv:2107.13962) URL: [https://arxiv.org/abs/2107.13962](https://arxiv.org/abs/2107.13962)

These papers provide in-depth discussions on data collection, model selection, and evaluation techniques, offering valuable insights into the challenges and best practices of dataset splitting in machine learning.

