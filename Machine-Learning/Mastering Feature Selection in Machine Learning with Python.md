## Mastering Feature Selection in Machine Learning with Python in Python

Slide 1: Introduction to Feature Selection

Feature selection is a crucial step in machine learning that involves identifying the most relevant features for your model. It helps improve model performance, reduce overfitting, and decrease computational costs.

```python
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

# Load dataset
data = pd.read_csv('dataset.csv')
X = data.drop('target', axis=1)
y = data['target']

# Select top 5 features
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)

print("Original feature count:", X.shape[1])
print("Selected feature count:", X_selected.shape[1])
```

Slide 2: Filter Methods: Correlation-based Selection

Correlation-based selection removes highly correlated features to reduce redundancy. This method calculates the correlation between features and removes one of each pair that exceeds a threshold.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation matrix
corr_matrix = X.corr().abs()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Create a heatmap of the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Remove upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
corr_matrix[mask] = np.nan

# Find features with correlation greater than 0.8
to_drop = [column for column in corr_matrix.columns if any(corr_matrix[column] > 0.8)]

print("Features to drop:", to_drop)
```

Slide 3: Filter Methods: Variance Threshold

Variance threshold is a simple technique that removes features with low variance. It's particularly useful for removing constant or near-constant features.

```python
from sklearn.feature_selection import VarianceThreshold

# Create a variance threshold object
selector = VarianceThreshold(threshold=0.1)

# Fit the selector to your data
X_selected = selector.fit_transform(X)

# Get the selected feature names
selected_features = X.columns[selector.get_support()]

print("Original feature count:", X.shape[1])
print("Selected feature count:", X_selected.shape[1])
print("Selected features:", selected_features.tolist())
```

Slide 4: Wrapper Methods: Recursive Feature Elimination

Recursive Feature Elimination (RFE) works by recursively removing attributes and building a model on those attributes that remain. It uses the model accuracy to identify which attributes contribute the most to predicting the target variable.

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Create RFE selector
selector = RFE(estimator=rf, n_features_to_select=5, step=1)

# Fit the selector
selector = selector.fit(X, y)

# Get the selected feature names
selected_features = X.columns[selector.support_]

print("Selected features:", selected_features.tolist())
```

Slide 5: Embedded Methods: Lasso Regularization

Lasso (L1 regularization) can be used for feature selection as it encourages sparsity in the model coefficients, effectively setting some feature coefficients to zero.

```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit the Lasso model
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

# Get feature importances
feature_importance = pd.Series(abs(lasso.coef_), index=X.columns)
selected_features = feature_importance[feature_importance > 0].index

print("Selected features:", selected_features.tolist())
```

Slide 6: Tree-based Feature Importance

Decision trees and ensemble methods like Random Forests can provide feature importance scores, which can be used for feature selection.

```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Create and fit the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
importances.plot(kind='bar')
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()
```

Slide 7: Mutual Information

Mutual Information measures the mutual dependence between two variables. It can be used to select features that have the strongest relationship with the target variable.

```python
from sklearn.feature_selection import mutual_info_classif

# Calculate mutual information scores
mi_scores = mutual_info_classif(X, y)

# Create a dataframe of features and their mutual information scores
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print("Top 5 features by Mutual Information:")
print(mi_scores.head())

# Select top 5 features
top_features = mi_scores.nlargest(5).index.tolist()
X_selected = X[top_features]

print("\nSelected features:", top_features)
```

Slide 8: Principal Component Analysis (PCA)

While not strictly a feature selection method, PCA is a dimensionality reduction technique that can be used to create new features that capture the most variance in the data.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_scaled)

print("Original number of features:", X.shape[1])
print("Number of PCA components:", X_pca.shape[1])

# Plot explained variance ratio
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()
```

Slide 9: Feature Agglomeration

Feature Agglomeration is a hierarchical clustering method that merges features that are similar, reducing the number of features while retaining information.

```python
from sklearn.cluster import FeatureAgglomeration

# Create and fit FeatureAgglomeration
agglo = FeatureAgglomeration(n_clusters=5)
X_reduced = agglo.fit_transform(X)

print("Original number of features:", X.shape[1])
print("Number of features after agglomeration:", X_reduced.shape[1])

# Get cluster labels for each feature
cluster_labels = agglo.labels_

# Print features in each cluster
for i in range(5):
    cluster_features = X.columns[cluster_labels == i].tolist()
    print(f"Cluster {i}: {cluster_features}")
```

Slide 10: Sequential Feature Selection

Sequential Feature Selection algorithms are wrapper methods that add or remove features to form a feature subset in a greedy fashion.

```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

# Create a logistic regression model
lr = LogisticRegression(random_state=42)

# Create forward sequential feature selector
sfs = SequentialFeatureSelector(lr, n_features_to_select=5, direction='forward')

# Fit the selector
sfs.fit(X, y)

# Get selected feature names
selected_features = X.columns[sfs.get_support()].tolist()

print("Selected features:", selected_features)
```

Slide 11: Boruta Algorithm

The Boruta algorithm is a wrapper method that determines relevance by comparing original attributes' importance with importance achievable at random, estimated using their permuted copies.

```python
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

# Create Boruta feature selector
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

# Fit the selector
feat_selector.fit(np.array(X), np.array(y))

# Get selected feature names
selected_features = X.columns[feat_selector.support_].tolist()

print("Selected features:", selected_features)
```

Slide 12: Cross-validation in Feature Selection

Cross-validation is crucial in feature selection to ensure that the selected features generalize well to unseen data and to avoid overfitting.

```python
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Create RFECV object
selector = RFECV(estimator=rf, step=1, cv=5, scoring='accuracy')

# Fit the selector
selector = selector.fit(X, y)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.show()

print("Optimal number of features:", selector.n_features_)
print("Best cross-validation score:", selector.grid_scores_.max())
```

Slide 13: Putting It All Together: A Feature Selection Pipeline

Combining multiple feature selection methods can lead to more robust feature sets. Here's an example of a feature selection pipeline.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('variance_threshold', VarianceThreshold(threshold=0.1)),
    ('univariate', SelectKBest(score_func=f_classif, k=10)),
    ('rfe', RFE(estimator=RandomForestClassifier(), n_features_to_select=5))
])

# Fit the pipeline
pipeline.fit(X, y)

# Get selected feature names
selected_features = X.columns[pipeline.named_steps['rfe'].support_].tolist()

print("Selected features:", selected_features)
```

Slide 14: Additional Resources

1. "Feature Selection for Machine Learning" by Jundong Li et al. (2018) ArXiv: [https://arxiv.org/abs/1601.07996](https://arxiv.org/abs/1601.07996)
2. "An Introduction to Variable and Feature Selection" by Isabelle Guyon and André Elisseeff (2003) Journal of Machine Learning Research
3. "Feature Selection: A Data Perspective" by Jundong Li et al. (2017) ArXiv: [https://arxiv.org/abs/1704.08103](https://arxiv.org/abs/1704.08103)

These resources provide in-depth information on various feature selection techniques and their applications in machine learning.

