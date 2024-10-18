## 8 Advanced Feature Engineering for Machine Learning

Slide 1: Polynomial Features

Polynomial features capture nonlinear relationships in data. They create new features by combining existing ones, raising them to powers, or multiplying them together. This technique can reveal complex patterns that linear models might miss.

```python
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[1, 2], [3, 4], [5, 6]])
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

print("Original features:")
print(X)
print("\nPolynomial features:")
print(X_poly)
```

This code transforms a 2D feature space into a 6D polynomial feature space, including bias term, linear terms, and quadratic terms.

Slide 2: Target Encoding

Target encoding replaces categorical variables with the mean target value for each category. This method can effectively handle high-cardinality features and capture category-target relationships.

```python
from category_encoders import TargetEncoder

data = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B', 'C'],
    'target': [1, 0, 1, 1, 0, 0]
})

encoder = TargetEncoder()
encoded_data = encoder.fit_transform(data['category'], data['target'])

print("Original data:")
print(data)
print("\nEncoded data:")
print(encoded_data)
```

This example encodes the 'category' column based on the mean of the 'target' column for each category.

Slide 3: Feature Hashing

Feature hashing is a dimensionality reduction technique that maps high-dimensional feature spaces to lower-dimensional spaces. It's particularly useful for text data or high-cardinality categorical variables.

```python

data = [
    {'fruit': 'apple', 'color': 'red'},
    {'fruit': 'banana', 'color': 'yellow'},
    {'fruit': 'apple', 'color': 'green'}
]

hasher = FeatureHasher(n_features=4, input_type='dict')
hashed_features = hasher.transform(data)

print("Hashed features:")
print(hashed_features.toarray())
```

This code hashes dictionary features into a 4-dimensional space, demonstrating how feature hashing can reduce dimensionality.

Slide 4: Lag Features

Lag features are crucial in time series analysis, capturing temporal dependencies by including past values of a variable as features for predicting future values.

```python

dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
data = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, index=dates)

data['lag_1'] = data['value'].shift(1)
data['lag_2'] = data['value'].shift(2)

print(data)
```

This example creates lag features for the previous 1 and 2 days, useful for time series forecasting.

Slide 5: Binning

Binning groups continuous variables into discrete intervals, which can help capture nonlinear relationships and reduce the impact of minor observation errors.

```python
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame({'feature': np.random.normal(0, 1, 1000)})

data['binned'] = pd.cut(data['feature'], bins=5)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(data['feature'], bins=30)
plt.title('Original Distribution')
plt.subplot(1, 2, 2)
data['binned'].value_counts().sort_index().plot(kind='bar')
plt.title('Binned Distribution')
plt.tight_layout()
plt.show()
```

This code bins a normally distributed feature into 5 categories and visualizes the original and binned distributions.

Slide 6: Feature Interactions

Feature interactions capture the combined effect of multiple features, which can be more informative than individual features alone.

```python
import numpy as np

np.random.seed(0)
data = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100)
})

data['interaction'] = data['feature1'] * data['feature2']

print(data.head())

# Visualize the interaction
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['feature1'], data['feature2'], data['interaction'])
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Interaction')
plt.title('Feature Interaction Visualization')
plt.show()
```

This example creates an interaction feature by multiplying two existing features and visualizes the result in 3D space.

Slide 7: Dimensionality Reduction

Dimensionality reduction techniques like PCA (Principal Component Analysis) can improve computational efficiency and reduce overfitting by capturing the most important aspects of high-dimensional data.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
X = np.random.randn(200, 2)
X[:100, 0] += 2
X[100:, 0] -= 2

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot original and PCA-transformed data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], c='b', alpha=0.5)
ax1.set_title('Original Data')

ax2.scatter(X_pca[:, 0], X_pca[:, 1], c='r', alpha=0.5)
ax2.set_title('PCA Transformed Data')

plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

This code applies PCA to a 2D dataset and visualizes both the original and transformed data, showing how PCA can find the principal components of variation.

Slide 8: Group Aggregation

Group aggregation involves computing summary statistics for features within groups, which can reveal patterns and relationships at different levels of granularity.

```python
import numpy as np

# Create sample data
np.random.seed(0)
data = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'value1': np.random.rand(1000),
    'value2': np.random.randint(1, 100, 1000)
})

# Perform group aggregation
agg_data = data.groupby('category').agg({
    'value1': ['mean', 'std'],
    'value2': ['sum', 'max', 'min']
})

print(agg_data)

# Visualize aggregated data
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

agg_data['value1']['mean'].plot(kind='bar', ax=ax1, yerr=agg_data['value1']['std'])
ax1.set_title('Mean and Std Dev of Value1 by Category')

agg_data['value2'][['sum', 'max', 'min']].plot(kind='bar', ax=ax2)
ax2.set_title('Sum, Max, and Min of Value2 by Category')

plt.tight_layout()
plt.show()
```

This example demonstrates group aggregation on a dataset with categories, computing various statistics and visualizing the results.

Slide 9: Real-Life Example: Customer Churn Prediction

Let's apply some of these feature engineering techniques to predict customer churn in a telecommunications company.

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate sample data
np.random.seed(0)
data = pd.DataFrame({
    'usage_minutes': np.random.randint(0, 1000, 1000),
    'contract_length': np.random.choice([12, 24, 36], 1000),
    'customer_service_calls': np.random.randint(0, 10, 1000),
    'churn': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
})

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data[['usage_minutes', 'customer_service_calls']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names(['usage_minutes', 'customer_service_calls']))

# Combine with original features
final_data = pd.concat([data, poly_df], axis=1)

# Split data
X = final_data.drop('churn', axis=1)
y = final_data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Feature importance
importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
importance = importance.sort_values('importance', ascending=False).head(10)
importance.plot(x='feature', y='importance', kind='bar', figsize=(10, 5))
plt.title('Top 10 Important Features')
plt.tight_layout()
plt.show()
```

This example demonstrates the use of polynomial features in a customer churn prediction model, showcasing how feature engineering can improve model performance.

Slide 10: Real-Life Example: Text Classification

Let's apply feature hashing to a text classification task, such as sentiment analysis of movie reviews.

```python
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample movie reviews and sentiments
reviews = [
    "This movie was fantastic! I loved every minute.",
    "Terrible plot, bad acting. Waste of time.",
    "Great performances by the entire cast.",
    "Boring and predictable. Don't bother.",
    "A masterpiece of modern cinema.",
    "I fell asleep halfway through. Very disappointing."
]
sentiments = [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

# Apply feature hashing
vectorizer = HashingVectorizer(n_features=2**8)  # 256 features
X = vectorizer.fit_transform(reviews)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Predict sentiment for a new review
new_review = ["The movie had its moments, but overall it was just okay."]
new_review_hashed = vectorizer.transform(new_review)
prediction = model.predict(new_review_hashed)
print(f"\nPredicted sentiment for new review: {'Positive' if prediction[0] == 1 else 'Negative'}")
```

This example demonstrates how feature hashing can be used to convert text data into a fixed-size feature vector for sentiment analysis.

Slide 11: Advanced Techniques: Automated Feature Engineering

While manual feature engineering is powerful, automated techniques can discover complex features that humans might miss. Libraries like featuretools can automate this process.

```python
import pandas as pd
import numpy as np

# Create sample datasets
customers = pd.DataFrame({
    "customer_id": range(5),
    "join_date": pd.date_range("2023-01-01", periods=5)
})

transactions = pd.DataFrame({
    "transaction_id": range(20),
    "customer_id": np.random.choice(range(5), 20),
    "amount": np.random.randint(10, 100, 20),
    "transaction_date": pd.date_range("2023-01-01", periods=20)
})

# Create entity set
es = ft.EntitySet(id="customer_transactions")
es = es.add_dataframe(
    dataframe_name="customers",
    dataframe=customers,
    index="customer_id",
    time_index="join_date"
)
es = es.add_dataframe(
    dataframe_name="transactions",
    dataframe=transactions,
    index="transaction_id",
    time_index="transaction_date"
)

# Add relationship
es = es.add_relationship("customers", "customer_id", "transactions", "customer_id")

# Generate features
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="customers",
    agg_primitives=["sum", "mean", "max"],
    trans_primitives=["day", "month", "year"],
    max_depth=2
)

print("Generated features:")
print(feature_matrix.head())
print("\nFeature definitions:")
for feature in feature_defs:
    print(feature)
```

This example uses featuretools to automatically generate complex features from relational data, demonstrating how automated feature engineering can uncover intricate patterns.

Slide 12: Feature Selection

Feature selection improves model performance and interpretability by choosing the most relevant features. This process can reduce overfitting and computational complexity.

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=100, n_informative=5, 
                           random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select top K features
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Train models
rf_full = RandomForestClassifier(random_state=42).fit(X_train, y_train)
rf_selected = RandomForestClassifier(random_state=42).fit(X_train_selected, y_train)

# Evaluate models
acc_full = accuracy_score(y_test, rf_full.predict(X_test))
acc_selected = accuracy_score(y_test, rf_selected.predict(X_test_selected))

print(f"Accuracy with all features: {acc_full:.4f}")
print(f"Accuracy with selected features: {acc_selected:.4f}")

# Plot feature importances
plt.figure(figsize=(10, 5))
plt.bar(range(10), selector.scores_[:10])
plt.title('Top 10 Feature Importances')
plt.xlabel('Feature Index')
plt.ylabel('Importance Score')
plt.show()
```

This example demonstrates how feature selection can improve model performance by focusing on the most informative features.

Slide 13: Cross-Validation in Feature Engineering

Cross-validation ensures that our feature engineering techniques generalize well to unseen data and aren't overfitting to the training set.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.sort(np.random.rand(100, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, X.shape[0])

degrees = [1, 4, 15]
cv_scores = []

for degree in degrees:
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = make_pipeline(polynomial_features, linear_regression)
    scores = cross_val_score(pipeline, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_scores.append(-scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(degrees, cv_scores, marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Cross-Validation Scores for Different Polynomial Degrees')
plt.show()

print("MSE for each degree:", cv_scores)
print("Best degree:", degrees[np.argmin(cv_scores)])
```

This example shows how to use cross-validation to select the optimal degree for polynomial features, balancing model complexity and performance.

Slide 14: Additional Resources

For those interested in diving deeper into advanced feature engineering techniques, consider exploring these peer-reviewed articles:

1. "A Survey on Feature Selection Methods" (arXiv:1904.02368) URL: [https://arxiv.org/abs/1904.02368](https://arxiv.org/abs/1904.02368)
2. "Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists" (arXiv:1808.03368) URL: [https://arxiv.org/abs/1808.03368](https://arxiv.org/abs/1808.03368)

These resources provide comprehensive overviews of various feature engineering and selection methods, offering insights into state-of-the-art techniques and their applications in machine learning.


