## Feature selection preprocessing

Slide 1: Introduction to Feature Selection

Feature selection is a crucial preprocessing step in machine learning. It involves choosing the most relevant features from a dataset to improve model performance and reduce overfitting. While the original statement suggests that fewer features are always better, this isn't necessarily true. The goal is to find the optimal set of features that best represent the underlying patterns in the data.

```python
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

# Load sample dataset
data = pd.DataFrame({
    'feature1': ['A', 'B', 'A', 'C', 'B', 'C'],
    'feature2': [1, 2, 1, 3, 2, 3],
    'target': [0, 1, 0, 1, 1, 0]
})

print(data.head())
```

Slide 2: Understanding Chi-Square (χ²) Test

The Chi-Square test is a statistical method used to determine the independence between two categorical variables. In feature selection, it helps identify the features that have a significant relationship with the target variable. The test calculates a statistic that measures the difference between observed and expected frequencies.

```python
X = data[['feature1', 'feature2']]
y = data['target']

# Encode categorical variables
le = LabelEncoder()
X['feature1'] = le.fit_transform(X['feature1'])

# Perform Chi-Square test
selector = SelectKBest(score_func=chi2, k=2)
X_new = selector.fit_transform(X, y)

# Get feature scores
scores = selector.scores_
print("Feature scores:", scores)
```

Slide 3: Interpreting Chi-Square Results

The Chi-Square test produces a score for each feature. Higher scores indicate a stronger relationship between the feature and the target variable. We can use these scores to rank features and select the most relevant ones for our model.

```python
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': scores
})

# Sort features by score in descending order
feature_scores = feature_scores.sort_values('Score', ascending=False)
print(feature_scores)

# Select top k features
k = 1  # Change this value to select more or fewer features
selected_features = feature_scores.head(k)['Feature'].tolist()
print("Selected features:", selected_features)
```

Slide 4: Applying Feature Selection to the Dataset

Once we've identified the most relevant features, we can create a new dataset containing only these features. This reduced dataset can then be used for training our machine learning model.

```python
X_selected = X[selected_features]
print("Original dataset shape:", X.shape)
print("Selected dataset shape:", X_selected.shape)

# Combine selected features with target variable
final_dataset = pd.concat([X_selected, y], axis=1)
print("\nFinal dataset:")
print(final_dataset.head())
```

Slide 5: Limitations of Chi-Square Test

While the Chi-Square test is useful for categorical data, it has limitations. It assumes independence between observations and requires a sufficient sample size. For continuous features or more complex relationships, other feature selection methods might be more appropriate.

```python
data['continuous_feature'] = [0.1, 0.5, 0.2, 0.8, 0.3, 0.7]

# Attempt to use Chi-Square with continuous data (this will raise a warning)
X_cont = data[['continuous_feature']]
try:
    chi2(X_cont, y)
except ValueError as e:
    print("Error:", str(e))

# Proper handling of continuous features
from sklearn.feature_selection import f_regression

f_scores, _ = f_regression(X_cont, y)
print("F-regression score for continuous feature:", f_scores)
```

Slide 6: Real-Life Example: Text Classification

Let's consider a text classification problem where we want to categorize customer reviews as positive or negative. We'll use the Chi-Square test to select the most relevant words for our classification task.

```python

# Sample customer reviews
reviews = [
    "Great product, highly recommended!",
    "Terrible experience, avoid at all costs.",
    "Average product, nothing special.",
    "Excellent service and quality.",
    "Disappointing performance, would not buy again."
]
labels = [1, 0, 0, 1, 0]  # 1 for positive, 0 for negative

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)

# Perform Chi-Square test
selector = SelectKBest(score_func=chi2, k=5)
X_new = selector.fit_transform(X, labels)

# Get selected feature names
feature_names = vectorizer.get_feature_names_out()
selected_features = feature_names[selector.get_support()]

print("Selected words:", selected_features)
```

Slide 7: Real-Life Example: Image Classification

In image classification tasks, we often deal with high-dimensional data. Feature selection can help reduce the number of pixels or extracted features we use for classification. Here's a simplified example using random pixel values:

```python
from sklearn.feature_selection import mutual_info_classif

# Generate random image data (100 images, 1000 pixels each)
X = np.random.rand(100, 1000)
y = np.random.randint(0, 2, 100)  # Binary classification

# Calculate mutual information between each pixel and the target
mi_scores = mutual_info_classif(X, y)

# Select top 100 pixels
top_pixels = np.argsort(mi_scores)[-100:]

# Create a new dataset with selected pixels
X_selected = X[:, top_pixels]

print("Original image shape:", X.shape)
print("Selected image shape:", X_selected.shape)
```

Slide 8: Feature Selection vs. Dimensionality Reduction

It's important to distinguish between feature selection and dimensionality reduction techniques like PCA. Feature selection chooses a subset of original features, while dimensionality reduction creates new features that are combinations of the original ones.

```python

# Feature Selection (using previous example)
X_selected = X[:, top_pixels]

# Dimensionality Reduction with PCA
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)

print("Feature Selection shape:", X_selected.shape)
print("PCA shape:", X_pca.shape)

# Compare first few values
print("\nFirst 5 values after Feature Selection:")
print(X_selected[0, :5])
print("\nFirst 5 values after PCA:")
print(X_pca[0, :5])
```

Slide 9: Wrapper Methods for Feature Selection

While filter methods like Chi-Square are computationally efficient, wrapper methods can sometimes yield better results by considering the specific model being used. Here's an example using Recursive Feature Elimination (RFE):

```python
from sklearn.linear_model import LogisticRegression

# Generate sample data
X = np.random.rand(100, 20)
y = np.random.randint(0, 2, 100)

# Create a logistic regression model
model = LogisticRegression()

# Perform RFE
rfe = RFE(estimator=model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)

# Get selected feature indices
selected_features = np.where(rfe.support_)[0]

print("Selected features:", selected_features)
print("Feature ranking:", rfe.ranking_)
```

Slide 10: Embedded Methods: L1 Regularization

Embedded methods perform feature selection as part of the model training process. L1 regularization (Lasso) is a popular technique that can lead to sparse models by driving some feature coefficients to zero.

```python
from sklearn.preprocessing import StandardScaler

# Generate sample data
X = np.random.rand(100, 20)
y = np.dot(X, np.random.rand(20)) + np.random.randn(100) * 0.1

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Lasso model
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

# Print non-zero coefficients
non_zero = np.abs(lasso.coef_) > 1e-5
print("Selected features:", np.where(non_zero)[0])
print("Number of selected features:", np.sum(non_zero))
```

Slide 11: Cross-Validation in Feature Selection

It's crucial to use cross-validation when performing feature selection to avoid overfitting. Here's an example using cross-validated recursive feature elimination (RFECV):

```python
from sklearn.model_selection import StratifiedKFold

# Generate sample data
X = np.random.rand(100, 20)
y = np.random.randint(0, 2, 100)

# Create a logistic regression model
model = LogisticRegression()

# Perform RFECV
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features:", rfecv.n_features_)
print("Feature ranking:", rfecv.ranking_)

# Plot number of features VS. cross-validation scores
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
```

Slide 12: Handling Multicollinearity

Multicollinearity, where features are highly correlated with each other, can affect feature selection. Here's how to detect and handle it using Variance Inflation Factor (VIF):

```python

# Generate sample data with multicollinearity
X = np.random.rand(100, 5)
X[:, 4] = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1

# Calculate VIF for each feature
vif = pd.DataFrame()
vif["Feature"] = range(X.shape[1])
vif["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

print(vif)

# Remove features with high VIF
high_vif = vif[vif["VIF"] > 5]["Feature"].tolist()
X_reduced = np.delete(X, high_vif, axis=1)

print("Shape after removing high VIF features:", X_reduced.shape)
```

Slide 13: Ensemble Feature Selection

Combining multiple feature selection methods can lead to more robust results. Here's an example that combines the results of different methods:

```python

# Generate sample data
X = np.random.rand(100, 20)
y = np.random.randint(0, 2, 100)

# Chi-Square
chi2_selector = SelectKBest(chi2, k=10)
chi2_support = chi2_selector.fit(X, y).get_support()

# Mutual Information
mi_selector = SelectKBest(mutual_info_classif, k=10)
mi_support = mi_selector.fit(X, y).get_support()

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
rf_support = rf.feature_importances_ > np.mean(rf.feature_importances_)

# Combine results
combined_support = chi2_support & mi_support & rf_support
selected_features = np.where(combined_support)[0]

print("Selected features:", selected_features)
print("Number of selected features:", len(selected_features))
```

Slide 14: Feature Selection in Practice: Tips and Considerations

1. Start with domain knowledge to identify potentially relevant features.
2. Use a combination of different feature selection methods.
3. Consider the interpretability of your model when selecting features.
4. Be aware of the assumptions and limitations of each feature selection method.
5. Always validate your feature selection results using cross-validation.
6. Remember that more features aren't always better - aim for the optimal set.
7. Consider the computational cost of your feature selection method, especially for large datasets.

```python
import time

X = np.random.rand(1000, 100)
y = np.random.randint(0, 2, 1000)

methods = [
    ("Chi-Square", SelectKBest(chi2, k=10)),
    ("Mutual Information", SelectKBest(mutual_info_classif, k=10)),
    ("Random Forest", RandomForestClassifier(n_estimators=100))
]

for name, method in methods:
    start_time = time.time()
    method.fit(X, y)
    elapsed_time = time.time() - start_time
    print(f"{name} took {elapsed_time:.4f} seconds")
```

Slide 15: Additional Resources

For those interested in diving deeper into feature selection techniques and their applications, here are some valuable resources:

1. Guyon, I., & Elisseeff, A. (2003). An Introduction to Variable and Feature Selection. Journal of Machine Learning Research, 3, 1157-1182. ArXiv: [https://arxiv.org/abs/cs/0701072](https://arxiv.org/abs/cs/0701072)
2. Li, J., Cheng, K., Wang, S., Morstatter, F., Trevino, R. P., Tang, J., & Liu, H. (2017). Feature Selection: A Data Perspective. ACM Computing Surveys, 50(6), 1-45. ArXiv: [https://arxiv.org/abs/1601.07996](https://arxiv.org/abs/1601.07996)
3. Chandrashekar, G., & Sahin, F. (2014). A survey on feature selection methods. Computers & Electrical Engineering, 40(1), 16-28. DOI: 10.1016/j.compeleceng.2013.11.024

These papers provide comprehensive overviews of various feature selection techniques, their theoretical foundations, and practical applications in machine learning.


