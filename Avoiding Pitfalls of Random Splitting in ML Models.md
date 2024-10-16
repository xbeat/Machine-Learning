## Avoiding Pitfalls of Random Splitting in ML Models
Slide 1: Understanding Random Splitting in Machine Learning

Random splitting is a crucial technique in machine learning for dividing datasets into training and testing sets. However, when not handled properly, it can lead to significant issues in model performance and generalization. Let's explore why random splitting can be problematic and how to mitigate its potential pitfalls.

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Generate a sample dataset
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

# Perform random splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
```

Slide 2: The Importance of Data Distribution

When randomly splitting data, we assume that the resulting subsets will have similar distributions. However, this assumption doesn't always hold, especially for small or imbalanced datasets. Uneven distribution can lead to models that perform well on the training set but poorly on the test set or in real-world scenarios.

```python
import matplotlib.pyplot as plt

# Visualize the distribution of a feature in both sets
plt.figure(figsize=(10, 5))
plt.hist(X_train[:, 0], bins=30, alpha=0.5, label='Training')
plt.hist(X_test[:, 0], bins=30, alpha=0.5, label='Testing')
plt.legend()
plt.title("Distribution of Feature 0 in Training and Testing Sets")
plt.show()
```

Slide 3: Class Imbalance Issues

Random splitting can exacerbate class imbalance problems, especially in datasets with rare classes. This can result in models that perform poorly on minority classes or fail to learn them altogether.

```python
from collections import Counter

# Check class distribution in both sets
train_class_dist = Counter(y_train)
test_class_dist = Counter(y_test)

print("Training set class distribution:", train_class_dist)
print("Testing set class distribution:", test_class_dist)

# Visualize class distribution
plt.figure(figsize=(10, 5))
plt.bar(train_class_dist.keys(), train_class_dist.values(), alpha=0.5, label='Training')
plt.bar(test_class_dist.keys(), test_class_dist.values(), alpha=0.5, label='Testing')
plt.legend()
plt.title("Class Distribution in Training and Testing Sets")
plt.show()
```

Slide 4: Temporal Dependencies

For time-series data or datasets with temporal dependencies, random splitting can break the natural order of events, leading to data leakage and overly optimistic performance estimates.

```python
import pandas as pd

# Create a time-series dataset
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.cumsum(np.random.randn(len(dates)))
df = pd.DataFrame({'date': dates, 'value': values})

# Incorrect: Random splitting
train_random, test_random = train_test_split(df, test_size=0.2, random_state=42)

# Correct: Temporal splitting
train_temporal = df[df['date'] < '2023-11-01']
test_temporal = df[df['date'] >= '2023-11-01']

print("Random split - Train:", train_random['date'].min(), "to", train_random['date'].max())
print("Random split - Test:", test_random['date'].min(), "to", test_random['date'].max())
print("Temporal split - Train:", train_temporal['date'].min(), "to", train_temporal['date'].max())
print("Temporal split - Test:", test_temporal['date'].min(), "to", test_temporal['date'].max())
```

Slide 5: Spatial Dependencies

In geospatial data, random splitting can separate nearby points, potentially leading to spatial autocorrelation issues and inflated performance metrics.

```python
import geopandas as gpd
from shapely.geometry import Point

# Create a sample geospatial dataset
np.random.seed(42)
lat = np.random.uniform(40, 41, 1000)
lon = np.random.uniform(-74, -73, 1000)
geometry = [Point(xy) for xy in zip(lon, lat)]
gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

# Visualize the spatial distribution
world = gpd.read_file(gpd.datasets.get_path('usa_zip_codes'))
ax = world.plot(figsize=(10, 10), color='white', edgecolor='black')
gdf.plot(ax=ax, color='red', markersize=5)
plt.title("Spatial Distribution of Data Points")
plt.show()

# Implement spatial cross-validation instead of random splitting
from sklearn.model_selection import KFold
spatial_cv = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in spatial_cv.split(gdf):
    train_gdf = gdf.iloc[train_index]
    test_gdf = gdf.iloc[test_index]
    print(f"Training set size: {len(train_gdf)}, Testing set size: {len(test_gdf)}")
```

Slide 6: Feature Correlation and Multicollinearity

Random splitting might not preserve the correlation structure between features, potentially leading to models that fail to capture important relationships or suffer from multicollinearity issues.

```python
import seaborn as sns

# Generate correlated features
cov_matrix = np.array([[1, 0.8], [0.8, 1]])
X_corr = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=1000)

# Split the data
X_train_corr, X_test_corr = train_test_split(X_corr, test_size=0.2, random_state=42)

# Visualize correlation in both sets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(np.corrcoef(X_train_corr.T), ax=ax1, annot=True, cmap='coolwarm')
ax1.set_title("Feature Correlation in Training Set")
sns.heatmap(np.corrcoef(X_test_corr.T), ax=ax2, annot=True, cmap='coolwarm')
ax2.set_title("Feature Correlation in Testing Set")
plt.tight_layout()
plt.show()
```

Slide 7: Sample Size and Representativeness

Small sample sizes can lead to non-representative splits, especially when dealing with high-dimensional data or complex patterns. This can result in unreliable model evaluations and poor generalization.

```python
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generate a small dataset
X_small, y_small = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=10, n_classes=2, random_state=42)

# Perform multiple random splits and evaluate
n_splits = 100
accuracies = []

for _ in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, test_size=0.2)
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(10, 5))
plt.hist(accuracies, bins=20)
plt.title("Distribution of Accuracy Scores Across Multiple Random Splits")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.show()

print(f"Mean accuracy: {np.mean(accuracies):.2f}")
print(f"Standard deviation: {np.std(accuracies):.2f}")
```

Slide 8: Stratified Sampling

To mitigate some of the issues with random splitting, stratified sampling can be used to maintain the proportion of classes in both training and testing sets.

```python
from sklearn.model_selection import StratifiedShuffleSplit

# Generate an imbalanced dataset
X_imb, y_imb = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Perform stratified sampling
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in sss.split(X_imb, y_imb):
    X_train_strat, X_test_strat = X_imb[train_index], X_imb[test_index]
    y_train_strat, y_test_strat = y_imb[train_index], y_imb[test_index]

print("Original class distribution:", Counter(y_imb))
print("Training set class distribution:", Counter(y_train_strat))
print("Testing set class distribution:", Counter(y_test_strat))
```

Slide 9: Cross-Validation

Cross-validation can help mitigate the impact of random splitting by using multiple train-test splits and averaging the results.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Generate a dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# Perform cross-validation
model = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5)

print("Cross-validation scores:", cv_scores)
print(f"Mean CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# Compare with a single split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
single_split_score = model.score(X_test, y_test)
print(f"Single split score: {single_split_score:.2f}")
```

Slide 10: Time Series Cross-Validation

For time series data, specialized cross-validation techniques can be used to maintain temporal order and avoid data leakage.

```python
from sklearn.model_selection import TimeSeriesSplit

# Generate a time series dataset
n_samples = 100
time_index = pd.date_range('2023-01-01', periods=n_samples, freq='D')
series = pd.Series(np.cumsum(np.random.randn(n_samples)), index=time_index)

# Perform time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

plt.figure(figsize=(10, 8))
for i, (train_index, test_index) in enumerate(tscv.split(series)):
    train = series.iloc[train_index]
    test = series.iloc[test_index]
    
    plt.subplot(5, 1, i+1)
    plt.plot(train.index, train.values, label='Training')
    plt.plot(test.index, test.values, label='Testing')
    plt.title(f'Split {i+1}')
    plt.legend(loc='best')

plt.tight_layout()
plt.show()
```

Slide 11: Nested Cross-Validation

Nested cross-validation can be used to perform both model selection and unbiased performance estimation, reducing the impact of random splitting on hyperparameter tuning.

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC

# Generate a dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# Define the parameter grid
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# Perform nested cross-validation
cv_outer = KFold(n_splits=5, shuffle=True, random_state=42)
cv_inner = KFold(n_splits=3, shuffle=True, random_state=42)

nested_scores = []

for train_index, test_index in cv_outer.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf = GridSearchCV(SVC(), param_grid, cv=cv_inner)
    clf.fit(X_train, y_train)
    nested_scores.append(clf.score(X_test, y_test))

print("Nested CV scores:", nested_scores)
print(f"Mean nested CV score: {np.mean(nested_scores):.2f} (+/- {np.std(nested_scores) * 2:.2f})")
```

Slide 12: Real-Life Example: Image Classification

In image classification tasks, random splitting can lead to biased results if similar images end up in both training and testing sets. Let's simulate this scenario using a simple image dataset.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Perform random splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Visualize some misclassified images
misclassified = X_test[y_test != y_pred]
mis_true = y_test[y_test != y_pred]
mis_pred = y_pred[y_test != y_pred]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    if i < len(misclassified):
        ax.imshow(misclassified[i].reshape(8, 8), cmap='gray')
        ax.set_title(f"True: {mis_true[i]}, Pred: {mis_pred[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 13: Real-Life Example: Text Classification

In text classification, random splitting can separate related documents, potentially leading to overfitting or underfitting. Let's demonstrate this using a simple sentiment analysis task.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# Sample dataset
texts = [
    "I love this product", "Great service", "Terrible experience",
    "Awful customer support", "Amazing quality", "Highly recommended",
    "Waste of money", "Don't buy this", "Excellent purchase",
    "Disappointing results"
]
labels = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0]  # 1 for positive, 0 for negative

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Perform random splitting
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train and evaluate the model
model = MultinomialNB()
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training accuracy: {train_score:.2f}")
print(f"Testing accuracy: {test_score:.2f}")

# Perform cross-validation
cv_scores = cross_val_score(model, X, labels, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
```

Slide 14: Mitigating Random Splitting Issues

To address the potential problems of random splitting, consider these strategies:

1. Use stratified sampling to maintain class distributions.
2. Implement cross-validation for more robust performance estimates.
3. For time series data, use time-based splitting or specialized cross-validation techniques.
4. In spatial data analysis, consider spatial cross-validation methods.
5. For small datasets, use techniques like bootstrapping or leave-one-out cross-validation.
6. When dealing with imbalanced datasets, consider oversampling, undersampling, or synthetic data generation techniques.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample

# Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in skf.split(X, labels):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]
    # Train and evaluate model here

# Bootstrapping for small datasets
n_iterations = 1000
n_samples = X.shape[0]
bootstrapped_scores = []
for _ in range(n_iterations):
    X_resampled, y_resampled = resample(X, labels, n_samples=n_samples, random_state=42)
    # Train and evaluate model, append score to bootstrapped_scores

print(f"Bootstrap 95% CI: ({np.percentile(bootstrapped_scores, 2.5):.2f}, {np.percentile(bootstrapped_scores, 97.5):.2f})")
```

Slide 15: Additional Resources

For more information on the challenges of random splitting in machine learning and advanced techniques to address them, consider exploring these resources:

1. "A Survey on Data Collection for Machine Learning: A Big Data - AI Integration Perspective" (ArXiv:1811.03402)
2. "Learning from Imbalanced Data" (ArXiv:1901.10698)
3. "A Survey of Cross-Validation Procedures for Model Selection" (ArXiv:0907.4728)
4. "Spatial Data Analysis in the Context of Machine Learning" (ArXiv:2011.05150)

These papers provide in-depth discussions on data splitting strategies, handling imbalanced datasets, and advanced cross-validation techniques for various types of data.

