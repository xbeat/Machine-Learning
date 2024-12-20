## Stratified K-Fold Cross-Validation in Python
Slide 1: Introduction to Stratified K-Fold Cross-Validation

Stratified K-Fold cross-validation is an essential technique in machine learning for evaluating model performance. It addresses the limitations of simple K-Fold cross-validation by ensuring that each fold maintains the same proportion of samples for each class as in the complete dataset. This method is particularly useful for imbalanced datasets, providing a more robust and reliable estimate of model performance.

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification

# Create a sample dataset
X, y = make_classification(n_samples=1000, n_classes=3, weights=[0.1, 0.3, 0.6], random_state=42)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Iterate through the folds
for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}:")
    print(f"  Train set size: {len(train_index)}")
    print(f"  Validation set size: {len(val_index)}")
    print(f"  Class distribution in validation set: {np.bincount(y[val_index])}")
```

Slide 2: The Importance of Stratification

Stratification ensures that each fold represents the overall class distribution of the dataset. This is crucial when dealing with imbalanced datasets, where one class may have significantly fewer samples than others. By maintaining the class proportions, we reduce the risk of introducing bias in our model evaluation process.

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Compare class distributions in StratifiedKFold vs regular KFold
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for ax, cv, title in zip([ax1, ax2], [skf, kf], ['StratifiedKFold', 'KFold']):
    for fold, (_, val_index) in enumerate(cv.split(X, y), 1):
        ax.bar(f'Fold {fold}', np.bincount(y[val_index]), alpha=0.7)
    
    ax.set_title(title)
    ax.set_ylabel('Class count')
    ax.legend(['Class 0', 'Class 1', 'Class 2'])

plt.tight_layout()
plt.show()
```

Slide 3: Implementing Stratified K-Fold with Scikit-learn

Scikit-learn provides a convenient implementation of Stratified K-Fold cross-validation. Let's explore how to use it in a complete machine learning pipeline.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load the Iris dataset
X, y = load_iris(return_X_y=True)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize an empty list to store fold accuracies
fold_accuracies = []

# Iterate through the folds
for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Train a model (SVM in this case)
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    fold_accuracies.append(accuracy)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")

print(f"\nMean Accuracy: {np.mean(fold_accuracies):.4f}")
print(f"Standard Deviation: {np.std(fold_accuracies):.4f}")
```

Slide 4: Handling Imbalanced Datasets

Stratified K-Fold is particularly useful for imbalanced datasets. Let's create an imbalanced dataset and see how Stratified K-Fold maintains class proportions.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Create an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Print class distribution for each fold
for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
    train_distribution = np.bincount(y[train_index]) / len(train_index)
    val_distribution = np.bincount(y[val_index]) / len(val_index)
    
    print(f"Fold {fold}:")
    print(f"  Train set class distribution: {train_distribution}")
    print(f"  Validation set class distribution: {val_distribution}")
    print()

# Overall dataset distribution
overall_distribution = np.bincount(y) / len(y)
print(f"Overall dataset class distribution: {overall_distribution}")
```

Slide 5: Stratified K-Fold vs. Regular K-Fold

Let's compare Stratified K-Fold with regular K-Fold to highlight the differences in maintaining class proportions, especially for imbalanced datasets.

```python
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt

# Create an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Initialize KFold and StratifiedKFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Function to calculate class proportions
def get_class_proportions(y):
    return np.bincount(y) / len(y)

# Plot class proportions for each fold
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, cv, title in zip([ax1, ax2], [kf, skf], ['KFold', 'StratifiedKFold']):
    for fold, (_, val_index) in enumerate(cv.split(X, y), 1):
        proportions = get_class_proportions(y[val_index])
        ax.bar(fold, proportions[1], alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Proportion of minority class')
    ax.set_ylim(0, 0.2)

plt.tight_layout()
plt.show()
```

Slide 6: Stratified K-Fold for Multi-class Problems

Stratified K-Fold is not limited to binary classification; it works equally well for multi-class problems. Let's demonstrate this using a multi-class dataset.

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt

# Load the Wine dataset (multi-class)
X, y = load_wine(return_X_y=True)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Plot class distribution for each fold
fig, ax = plt.subplots(figsize=(10, 6))

overall_distribution = np.bincount(y) / len(y)
ax.axhline(y=overall_distribution[0], color='r', linestyle='--', label='Class 0 Overall')
ax.axhline(y=overall_distribution[1], color='g', linestyle='--', label='Class 1 Overall')
ax.axhline(y=overall_distribution[2], color='b', linestyle='--', label='Class 2 Overall')

for fold, (_, val_index) in enumerate(skf.split(X, y), 1):
    fold_distribution = np.bincount(y[val_index]) / len(val_index)
    ax.scatter([fold] * 3, fold_distribution, c=['r', 'g', 'b'], alpha=0.7)

ax.set_xlabel('Fold')
ax.set_ylabel('Class Proportion')
ax.set_title('Class Distribution Across Folds (Stratified K-Fold)')
ax.legend()
plt.tight_layout()
plt.show()
```

Slide 7: Stratified K-Fold with Cross-Validation Scoring

We can use Stratified K-Fold with scikit-learn's cross-validation scoring functions to easily evaluate multiple metrics across folds.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import numpy as np

# Load the Breast Cancer dataset
X, y = load_breast_cancer(return_X_y=True)

# Initialize the model and StratifiedKFold
model = SVC(kernel='rbf', random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation with multiple metrics
cv_results = cross_validate(model, X, y, cv=cv, 
                            scoring=['accuracy', 'precision', 'recall', 'f1'],
                            return_train_score=True)

# Print results
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    train_scores = cv_results[f'train_{metric}']
    test_scores = cv_results[f'test_{metric}']
    print(f"{metric.capitalize()}:")
    print(f"  Train: {np.mean(train_scores):.4f} (+/- {np.std(train_scores):.4f})")
    print(f"  Test:  {np.mean(test_scores):.4f} (+/- {np.std(test_scores):.4f})")
    print()
```

Slide 8: Stratified K-Fold for Hyperparameter Tuning

Stratified K-Fold can be used in combination with GridSearchCV or RandomizedSearchCV for hyperparameter tuning, ensuring that each fold in the inner cross-validation maintains class balance.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
import numpy as np

# Load the Iris dataset
X, y = load_iris(return_X_y=True)

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}

# Initialize StratifiedKFold for both outer and inner cross-validation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Initialize GridSearchCV with StratifiedKFold
grid_search = GridSearchCV(SVC(), param_grid, cv=inner_cv, scoring='accuracy')

# Perform nested cross-validation
outer_scores = []

for fold, (train_index, test_index) in enumerate(outer_cv.split(X, y), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Evaluate best model on test set
    best_model = grid_search.best_estimator_
    score = best_model.score(X_test, y_test)
    outer_scores.append(score)
    
    print(f"Fold {fold}:")
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Test accuracy: {score:.4f}")
    print()

print(f"Mean accuracy: {np.mean(outer_scores):.4f} (+/- {np.std(outer_scores):.4f})")
```

Slide 9: Stratified K-Fold for Time Series Data

While Stratified K-Fold is not typically used for time series data, we can adapt it for certain time series problems where maintaining class balance is important.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# Create a synthetic time series dataset with binary labels
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
values = np.cumsum(np.random.randn(len(dates)))
labels = (values > values.mean()).astype(int)

df = pd.DataFrame({'date': dates, 'value': values, 'label': labels})

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=False)

# Plot the time series data with fold assignments
plt.figure(figsize=(12, 6))
for fold, (_, val_index) in enumerate(skf.split(df, df['label']), 1):
    plt.scatter(df.iloc[val_index]['date'], df.iloc[val_index]['value'], 
                label=f'Fold {fold}', alpha=0.7)

plt.title('Time Series Data with Stratified K-Fold Assignments')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()

# Print class distribution for each fold
for fold, (train_index, val_index) in enumerate(skf.split(df, df['label']), 1):
    train_dist = df.iloc[train_index]['label'].value_counts(normalize=True)
    val_dist = df.iloc[val_index]['label'].value_counts(normalize=True)
    print(f"Fold {fold}:")
    print(f"  Train set class distribution: {train_dist.values}")
    print(f"  Validation set class distribution: {val_dist.values}")
    print()
```

Slide 10: Real-Life Example: Sentiment Analysis

Let's apply Stratified K-Fold to a sentiment analysis task using a movie review dataset.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Sample movie review dataset (for demonstration purposes)
reviews = [
    "This movie was fantastic! I loved every minute of it.",
    "Terrible acting and poor plot. Waste of time.",
    "Great special effects but the story was lacking.",
    "A masterpiece of cinema. Highly recommended!",
    "Boring and predictable. Don't bother watching.",
    "Decent movie, but nothing special.",
    "Absolutely brilliant! A must-see film.",
    "Disappointing sequel. Doesn't live up to the original.",
    "Solid performances by the cast. Enjoyable overall.",
    "Worst movie I've seen in years. Avoid at all costs."
]
sentiments = [1, 0, 1, 1, 0, 1, 1, 0, 1, 0]

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews)
y = np.array(sentiments)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Print results
    print(f"Fold {fold}:")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(classification_report(y_val, y_pred))
    print()
```

Slide 11: Real-Life Example: Image Classification

In this example, we'll use Stratified K-Fold for an image classification task on a subset of the CIFAR-10 dataset.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import fetch_openml

# Load a subset of CIFAR-10 data
X, y = fetch_openml('CIFAR_10', version=1, return_X_y=True, as_frame=False)
X = X[:1000]  # Use only 1000 samples for demonstration
y = y[:1000]

# Convert labels to numeric
y = y.astype(int)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
fold_accuracies = []

for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Train the model (using SVM for simplicity)
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    fold_accuracies.append(accuracy)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")

print(f"\nMean Accuracy: {np.mean(fold_accuracies):.4f}")
print(f"Standard Deviation: {np.std(fold_accuracies):.4f}")
```

Slide 12: Stratified K-Fold with Data Augmentation

In some cases, we might want to combine Stratified K-Fold with data augmentation techniques. Here's an example of how to do this for image data.

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Simulated image data and labels
np.random.seed(42)
X = np.random.rand(1000, 32, 32, 3)  # 1000 32x32 RGB images
y = np.random.randint(0, 5, 1000)  # 5 classes

# Simple image augmentation function (for demonstration)
def augment_image(image):
    # Simulate image augmentation (e.g., flipping)
    return np.flip(image, axis=1)

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation with data augmentation
fold_accuracies = []

for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Augment training data
    X_train_aug = np.vstack([X_train, np.array([augment_image(img) for img in X_train])])
    y_train_aug = np.hstack([y_train, y_train])
    
    # Flatten images for SVM
    X_train_flat = X_train_aug.reshape(X_train_aug.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    # Train the model
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train_flat, y_train_aug)
    
    # Make predictions
    y_pred = model.predict(X_val_flat)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    fold_accuracies.append(accuracy)
    
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")

print(f"\nMean Accuracy: {np.mean(fold_accuracies):.4f}")
print(f"Standard Deviation: {np.std(fold_accuracies):.4f}")
```

Slide 13: Stratified K-Fold for Regression Tasks

While Stratified K-Fold is typically used for classification, it can be adapted for regression tasks by binning the continuous target variable.

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer

# Generate regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Bin the continuous target variable
n_bins = 5
kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
y_binned = kbd.fit_transform(y.reshape(-1, 1)).ravel()

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
fold_mse = []

for fold, (train_index, val_index) in enumerate(skf.split(X, y_binned), 1):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate MSE
    mse = mean_squared_error(y_val, y_pred)
    fold_mse.append(mse)
    
    print(f"Fold {fold} MSE: {mse:.4f}")

print(f"\nMean MSE: {np.mean(fold_mse):.4f}")
print(f"Standard Deviation: {np.std(fold_mse):.4f}")
```

Slide 14: Additional Resources

For those interested in diving deeper into Stratified K-Fold cross-validation and related topics, here are some valuable resources:

1. Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. In Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI), 2, 1137-1143. ArXiv: [https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf](https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf)
2. Cawley, G. C., & Talbot, N. L. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. Journal of Machine Learning Research, 11, 2079-2107. ArXiv: [https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf)
3. Raschka, S. (2018). Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning. arXiv preprint arXiv:1811.12808. ArXiv: [https://arxiv.org/abs/1811.12808](https://arxiv.org/abs/1811.12808)

These papers provide in-depth discussions on cross-validation techniques, their implications, and best practices in model evaluation and selection.

