## Feature Engineering Techniques for Data Science
Slide 1: Feature Engineering Techniques for Data Scientists

Feature engineering is the process of transforming raw data into meaningful features that enhance machine learning model performance. It involves selecting, manipulating, and creating new features to improve the predictive power of models. This slideshow will cover various feature engineering techniques, including imputation, discretization, categorical encoding, feature splitting, handling outliers, variable transformations, scaling, and feature creation.

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Load sample data
data = pd.DataFrame({
    'age': [25, 30, np.nan, 40, 35],
    'income': [50000, 60000, 75000, np.nan, 80000],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master']
})

print(data)
```

Slide 2: Imputation

Imputation is the process of replacing missing values in a dataset with estimated values. This technique is crucial for maintaining data integrity and avoiding the loss of valuable information. Common imputation methods include mean, median, and mode imputation for numerical data, and most frequent value imputation for categorical data.

```python
# Impute missing values
imputer = SimpleImputer(strategy='mean')
data[['age', 'income']] = imputer.fit_transform(data[['age', 'income']])

print("Data after imputation:")
print(data)
```

Slide 3: Discretization

Discretization is the process of converting continuous numerical variables into discrete categorical variables. This technique can help capture non-linear relationships and reduce the impact of outliers. Common discretization methods include equal-width binning, equal-frequency binning, and custom binning based on domain knowledge.

```python
# Discretize 'age' into 3 bins
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
data['age_group'] = discretizer.fit_transform(data[['age']])

print("Data after discretization:")
print(data)
```

Slide 4: Categorical Encoding

Categorical encoding is the process of converting categorical variables into numerical representations that can be used in machine learning models. Common encoding techniques include one-hot encoding, label encoding, and ordinal encoding. One-hot encoding creates binary columns for each category, while label encoding assigns a unique integer to each category.

```python
# One-hot encode 'education' column
encoder = OneHotEncoder(sparse=False)
education_encoded = encoder.fit_transform(data[['education']])
education_columns = encoder.get_feature_names(['education'])

# Add encoded columns to the dataframe
data = pd.concat([data, pd.DataFrame(education_encoded, columns=education_columns)], axis=1)

print("Data after categorical encoding:")
print(data)
```

Slide 5: Feature Splitting

Feature splitting involves breaking down complex features into simpler, more informative components. This technique can help capture important information hidden within composite features. Common examples include splitting datetime features into separate components like year, month, and day, or extracting domain-specific information from text fields.

```python
# Sample data with datetime column
data['date'] = pd.to_datetime(['2023-01-15', '2023-02-28', '2023-03-10', '2023-04-05', '2023-05-20'])

# Split datetime into year, month, and day
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

print("Data after feature splitting:")
print(data)
```

Slide 6: Handling Outliers

Outliers are data points that significantly deviate from other observations. Handling outliers is crucial for maintaining the integrity of statistical analyses and machine learning models. Common techniques for handling outliers include removing them, capping (winsorization), or transforming the data using methods like log transformation or Box-Cox transformation.

```python
import matplotlib.pyplot as plt

# Generate sample data with outliers
np.random.seed(42)
data = np.random.normal(0, 1, 1000)
outliers = np.random.uniform(10, 15, 5)
data = np.concatenate([data, outliers])

# Function to cap outliers
def cap_outliers(x, lower_percentile=1, upper_percentile=99):
    lower, upper = np.percentile(x, [lower_percentile, upper_percentile])
    return np.clip(x, lower, upper)

# Cap outliers
data_capped = cap_outliers(data)

# Plot original and capped data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(data, bins=50)
ax1.set_title("Original Data")
ax2.hist(data_capped, bins=50)
ax2.set_title("Data after Capping Outliers")
plt.tight_layout()
plt.show()
```

Slide 7: Variable Transformations

Variable transformations modify the distribution or scale of features to improve model performance or meet certain assumptions. Common transformations include logarithmic, square root, and power transformations. These techniques can help normalize skewed distributions, stabilize variance, or linearize relationships between variables.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with exponential distribution
np.random.seed(42)
data = np.random.exponential(scale=2, size=1000)

# Apply log transformation
data_log = np.log1p(data)  # log1p is used to handle zero values

# Plot original and transformed data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(data, bins=50)
ax1.set_title("Original Data (Exponential)")
ax2.hist(data_log, bins=50)
ax2.set_title("Log-transformed Data")
plt.tight_layout()
plt.show()

print("Skewness before transformation:", np.round(pd.Series(data).skew(), 2))
print("Skewness after log transformation:", np.round(pd.Series(data_log).skew(), 2))
```

Slide 8: Scaling

Scaling is the process of normalizing the range of features in a dataset. This technique is essential when dealing with features of different scales, as it ensures that all features contribute equally to the model. Common scaling methods include standardization (z-score scaling) and min-max scaling.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Sample data
data = np.array([[1, 1000], [2, 2000], [3, 3000], [4, 4000], [5, 5000]])

# Standardization
scaler_standard = StandardScaler()
data_standardized = scaler_standard.fit_transform(data)

# Min-Max scaling
scaler_minmax = MinMaxScaler()
data_minmax = scaler_minmax.fit_transform(data)

print("Original data:")
print(data)
print("\nStandardized data:")
print(data_standardized)
print("\nMin-Max scaled data:")
print(data_minmax)
```

Slide 9: Feature Creation in Machine Learning

Feature creation involves generating new features from existing ones to capture complex relationships or domain-specific knowledge. This technique can significantly improve model performance by providing more informative inputs. Common approaches include polynomial features, interaction terms, and domain-specific engineered features.

```python
from sklearn.preprocessing import PolynomialFeatures

# Sample data
X = np.array([[1, 2], [3, 4], [5, 6]])

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print("Original features:")
print(X)
print("\nPolynomial features (degree 2):")
print(X_poly)
print("\nFeature names:")
print(poly.get_feature_names(['x1', 'x2']))
```

Slide 10: Real-Life Example - Customer Churn Prediction

In this example, we'll demonstrate feature engineering techniques for predicting customer churn in a telecommunications company. We'll use a subset of features from a typical customer dataset.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample customer data
data = pd.DataFrame({
    'tenure': [12, 24, 6, 36, 18],
    'monthly_charges': [50.5, 70.2, 45.0, 80.5, 65.0],
    'total_charges': [600, 1680, np.nan, 2900, 1170],
    'contract_type': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],
    'churn': ['No', 'No', 'Yes', 'No', 'Yes']
})

# Define preprocessing steps
numeric_features = ['tenure', 'monthly_charges', 'total_charges']
categorical_features = ['contract_type']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit and transform the data
X = data.drop('churn', axis=1)
y = data['churn']
X_processed = preprocessor.fit_transform(X)

print("Original data:")
print(data)
print("\nProcessed features:")
print(X_processed)
```

Slide 11: Real-Life Example - Text Classification

In this example, we'll demonstrate feature engineering techniques for text classification using a sample dataset of product reviews. We'll use natural language processing techniques to extract meaningful features from the text data.

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder

# Sample product review data
data = pd.DataFrame({
    'review_text': [
        "This product is amazing and works great!",
        "Disappointed with the quality, not worth the price.",
        "Average product, nothing special but does the job.",
        "Absolutely love it, highly recommended!",
        "Terrible experience, avoid at all costs."
    ],
    'sentiment': ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative']
})

# Text preprocessing pipeline
text_pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer())
])

# Encode sentiment labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['sentiment'])

# Process text features
X = text_pipeline.fit_transform(data['review_text'])

print("Original data:")
print(data)
print("\nProcessed text features (TF-IDF):")
print(X.toarray())
print("\nFeature names:")
print(text_pipeline.named_steps['vect'].get_feature_names_out())
print("\nEncoded sentiment labels:")
print(y)
```

Slide 12: Feature Selection

Feature selection is the process of choosing a subset of relevant features for use in model construction. It helps reduce overfitting, improve model performance, and reduce training time. Common techniques include filter methods (e.g., correlation-based selection), wrapper methods (e.g., recursive feature elimination), and embedded methods (e.g., L1 regularization).

```python
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectKBest, f_regression

# Load Boston housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Select top 5 features based on F-statistic
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = [boston.feature_names[i] for i in selector.get_support(indices=True)]

print("Original number of features:", X.shape[1])
print("Number of features after selection:", X_selected.shape[1])
print("Selected features:", selected_features)
```

Slide 13: Dimensionality Reduction

Dimensionality reduction techniques aim to reduce the number of features while preserving important information. These methods are useful for visualizing high-dimensional data, reducing computational complexity, and mitigating the curse of dimensionality. Common techniques include Principal Component Analysis (PCA) and t-SNE.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Iris Dataset')
plt.colorbar(scatter)
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Slide 14: Additional Resources

For further exploration of feature engineering techniques and advanced topics in machine learning, consider the following resources:

1. "Feature Engineering for Machine Learning" by Alice Zheng and Amanda Casari (O'Reilly Media)
2. "Automated Machine Learning: Methods, Systems, Challenges" by Frank Hutter, Lars Kotthoff, and Joaquin Vanschoren (Springer)
3. "Feature Engineering and Selection: A Practical Approach for Predictive Models" by Max Kuhn and Kjell Johnson (Chapman and Hall/CRC)

For recent research papers on feature engineering and related topics, visit ArXiv.org and search for "feature engineering" or "feature selection" in the Computer Science > Machine Learning category. Some relevant papers include:

1. "A Survey on Feature Selection Methods" (arXiv:1904.02368)
2. "AutoML: A Survey of the State-of-the-Art" (arXiv:1908.00709)

Remember to verify the sources and check for the most up-to-date information, as the field of machine learning is rapidly evolving.

