## Ultimate Cheatcode to Feature Engineering

Slide 1: Introduction to Feature Engineering

Feature engineering is the process of transforming raw data into meaningful features that can improve machine learning model performance. It's a crucial step in the data science pipeline, often making the difference between mediocre and excellent models.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load a sample dataset
data = pd.read_csv('sample_data.csv')

# Basic feature engineering: create a new feature
data['age_squared'] = data['age'] ** 2

# Standardize features
scaler = StandardScaler()
data[['age', 'age_squared']] = scaler.fit_transform(data[['age', 'age_squared']])

print(data.head())
```

Slide 2: Handling Missing Data

Missing data can significantly impact model performance. Imputation is a common technique to handle missing values.

```python

# Create an imputer object
imputer = SimpleImputer(strategy='mean')

# Impute missing values
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

print("Before imputation:")
print(data.isnull().sum())
print("\nAfter imputation:")
print(data_imputed.isnull().sum())
```

Slide 3: Encoding Categorical Variables

Categorical variables need to be converted to numerical format for most machine learning algorithms.

```python

# One-hot encoding
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(data[['category']])

# Create a new dataframe with encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names(['category']))

print(encoded_df.head())
```

Slide 4: Feature Scaling

Feature scaling ensures that all features contribute equally to the model's decision-making process.

```python

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Scale numerical features
scaled_features = scaler.fit_transform(data[['age', 'income']])

# Create a new dataframe with scaled features
scaled_df = pd.DataFrame(scaled_features, columns=['age_scaled', 'income_scaled'])

print(scaled_df.head())
```

Slide 5: Feature Creation

Creating new features can capture complex relationships in the data that may not be apparent in the original features.

```python
data['BMI'] = data['weight'] / ((data['height'] / 100) ** 2)

# Create interaction features
data['age_income_interaction'] = data['age'] * data['income']

print(data[['weight', 'height', 'BMI', 'age', 'income', 'age_income_interaction']].head())
```

Slide 6: Handling Date and Time Features

Date and time features can be rich sources of information when properly engineered.

```python

# Sample dataset with a date column
data = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
    'value': [10, 20, 15, 30, 25]
})

# Extract various components from the date
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['day_of_week'] = data['date'].dt.dayofweek

print(data)
```

Slide 7: Text Feature Engineering

Text data requires special preprocessing techniques to convert it into a format suitable for machine learning models.

```python

# Sample text data
texts = [
    "I love machine learning",
    "Feature engineering is crucial",
    "Python is great for data science"
]

# Create a bag of words representation
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(texts)

# Convert to dataframe for better visualization
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())

print(bow_df)
```

Slide 8: Binning Continuous Variables

Binning can help capture non-linear relationships and reduce the impact of outliers.

```python
import numpy as np

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({'age': np.random.randint(18, 80, 1000)})

# Create bins
data['age_group'] = pd.cut(data['age'], bins=[0, 25, 35, 50, 65, 100], labels=['18-25', '26-35', '36-50', '51-65', '65+'])

print(data.head(10))
print("\nValue counts of age groups:")
print(data['age_group'].value_counts())
```

Slide 9: Polynomial Features

Polynomial features can capture non-linear relationships between features.

```python

# Sample data
X = np.array([[1, 2], [3, 4], [5, 6]])

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Create a dataframe for better visualization
poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names(['x1', 'x2']))

print(poly_df)
```

Slide 10: Feature Selection

Feature selection helps identify the most relevant features, reducing noise and improving model performance.

```python
from sklearn.datasets import make_regression

# Generate a random regression problem
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

# Select top 5 features
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X, y)

# Get selected feature indices
selected_features = selector.get_support(indices=True)

print("Selected feature indices:", selected_features)
print("Shape of X before selection:", X.shape)
print("Shape of X after selection:", X_selected.shape)
```

Slide 11: Handling Imbalanced Datasets

Imbalanced datasets can lead to biased models. Techniques like oversampling or undersampling can help address this issue.

```python
from sklearn.datasets import make_classification

# Generate an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Original dataset shape:", dict(pd.Series(y).value_counts()))
print("Resampled dataset shape:", dict(pd.Series(y_resampled).value_counts()))
```

Slide 12: Real-life Example: House Price Prediction

Let's apply feature engineering techniques to a house price prediction problem.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset (assuming we have a CSV file with house data)
df = pd.read_csv('house_data.csv')

# Create new features
df['age'] = 2023 - df['year_built']
df['total_sqft'] = df['living_area'] + df['basement_area']
df['price_per_sqft'] = df['price'] / df['total_sqft']

# Encode categorical variables
df = pd.get_dummies(df, columns=['neighborhood', 'house_type'])

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['age', 'total_sqft', 'bedrooms', 'bathrooms']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print(df.head())
```

Slide 13: Real-life Example: Customer Churn Prediction

Let's apply feature engineering to a customer churn prediction problem in a telecommunications company.

```python
from sklearn.preprocessing import LabelEncoder

# Load the dataset (assuming we have a CSV file with customer data)
df = pd.read_csv('telecom_customers.csv')

# Create new features
df['total_charges'] = df['monthly_charges'] * df['tenure']
df['charges_to_tenure_ratio'] = df['total_charges'] / df['tenure']

# Encode categorical variables
le = LabelEncoder()
categorical_features = ['contract', 'internet_service', 'payment_method']
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# Create interaction features
df['contract_tenure_interaction'] = df['contract'] * df['tenure']

print(df.head())
```

Slide 14: Feature Engineering Pipeline

Combining all the feature engineering steps into a single pipeline can streamline the process and make it more reproducible.

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define preprocessing for numerical columns (scale them)
numeric_features = ['age', 'income', 'credit_score']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical columns (encode them)
categorical_features = ['gender', 'occupation', 'marital_status']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create and run the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_transformed = pipeline.fit_transform(X)

print("Shape of transformed features:", X_transformed.shape)
```

Slide 15: Additional Resources

For further exploration of feature engineering techniques and best practices, consider the following resources:

1. "Feature Engineering for Machine Learning" by Alice Zheng and Amanda Casari
2. "Automated Feature Engineering in Python" tutorial on Machine Learning Mastery
3. "Feature Engineering and Selection: A Practical Approach for Predictive Models" by Max Kuhn and Kjell Johnson
4. ArXiv paper: "A Survey on Automated Feature Engineering for Tabular Data" ([https://arxiv.org/abs/2106.01751](https://arxiv.org/abs/2106.01751))

Remember that feature engineering is both an art and a science. Experimentation and domain knowledge are key to creating effective features for your specific problem.


