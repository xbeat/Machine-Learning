## Feature Engineering for Machine Learning using python
Slide 1: 

Introduction to Feature Engineering

Feature engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive models, resulting in improved model accuracy on unseen data. It is a crucial step in the machine learning pipeline and can make a significant difference in model performance.

Code:

```python
# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_boston

# Load the Boston Housing dataset
data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
```

Slide 2: 

Missing Value Imputation

Missing values in datasets can lead to biased results or cause errors during model training. Imputation techniques aim to fill in these missing values with estimated values based on the available data.

Code:

```python
from sklearn.impute import SimpleImputer

# Create an imputer object
imputer = SimpleImputer(strategy='mean')

# Fit and transform the data
X_imputed = imputer.fit_transform(X)
```

Slide 3: 

Categorical Variable Encoding

Many machine learning algorithms require input features to be numerical. Categorical variables need to be encoded into a format that the algorithms can understand. Common encoding techniques include one-hot encoding and label encoding.

Code:

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label encoding for a single categorical feature
encoder = LabelEncoder()
X['category'] = encoder.fit_transform(X['category'])

# One-hot encoding for multiple categorical features
ohe = OneHotEncoder(sparse=False)
X_encoded = ohe.fit_transform(X[['feature1', 'feature2']])
```

Slide 4: 

Feature Scaling

Many machine learning algorithms are sensitive to the scale of the input features. Feature scaling is the process of rescaling the features to a common range, such as 0 to 1 or -1 to 1, to prevent features with larger values from dominating the objective function.

Code:

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Min-max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Slide 5: 

Feature Selection

Feature selection is the process of selecting the most relevant features from the dataset to improve model performance, reduce overfitting, and increase interpretability. It can be performed using various techniques like filter methods, wrapper methods, or embedded methods.

Code:

```python
from sklearn.feature_selection import SelectKBest, f_regression

# Filter method: Select the top k features
selector = SelectKBest(f_regression, k=5)
X_selected = selector.fit_transform(X, y)
```

Slide 6: 

Polynomial Features

Polynomial features are created by raising the original features to specific powers or combining them through multiplication. This can help capture nonlinear relationships between the features and the target variable.

Code:

```python
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

Slide 7: 

Interactions and Feature Crosses

Interactions and feature crosses involve combining multiple features into a new feature by applying operations like multiplication or division. This can help capture the relationships between different features and improve model performance.

Code:

```python
import numpy as np

# Create interaction features
X['interaction'] = X['feature1'] * X['feature2']

# Create feature cross
X_cross = np.c_[X, X['feature1'] / X['feature2']]
```

Slide 8: 

Binning and Discretization

Binning and discretization involve transforming continuous numerical features into discrete categories or bins. This can help simplify the data, reduce the impact of outliers, and make the relationships between features and the target variable more interpretable.

Code:

```python
import pandas as pd

# Equal-width binning
bins = pd.cut(X['feature'], bins=5)

# Equal-frequency binning
bins = pd.qcut(X['feature'], q=5)
```

Slide 9: 

Logarithmic Transformation

Logarithmic transformations can be applied to features with skewed distributions or features that span a large range of values. This transformation can help reduce the impact of outliers and make the data more symmetrical.

Code:

```python
import numpy as np

# Apply logarithmic transformation
X['log_feature'] = np.log1p(X['feature'])
```

Slide 10: 

Ordinal Encoding

Ordinal encoding is a technique used for encoding ordered categorical variables. It assigns a numerical value to each category based on its order or rank, preserving the ordinal relationship between the categories.

Code:

```python
from sklearn.preprocessing import OrdinalEncoder

# Create an ordinal encoder object
encoder = OrdinalEncoder()

# Fit and transform the ordinal feature
X['ordinal_feature'] = encoder.fit_transform(X[['ordinal_feature']])
```

Slide 11: 

Feature Extraction

Feature extraction is the process of transforming the original features into a reduced set of features that capture the most relevant information. Common techniques include Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA).

Code:

```python
from sklearn.decomposition import PCA

# Create a PCA object
pca = PCA(n_components=3)

# Fit and transform the data
X_pca = pca.fit_transform(X)
```

Slide 12: 

Feature Generation

Feature generation involves creating new features from the existing ones based on domain knowledge or intuition. This can help capture additional information and improve model performance.

Code:

```python
import numpy as np

# Generate a new feature
X['new_feature'] = np.sqrt(X['feature1']) / X['feature2']
```

Slide 13: 

Handling Text Data

When dealing with text data, it needs to be transformed into a numerical representation that can be used by machine learning algorithms. Common techniques include bag-of-words, TF-IDF, and word embeddings.

Code:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer object
vectorizer = TfidfVectorizer()

# Fit and transform the text data
X_tfidf = vectorizer.fit_transform(X['text_data'])
```

Slide 14: 

Feature Engineering Pipeline

In practice, feature engineering often involves combining multiple techniques into a pipeline. This ensures consistent and reproducible transformations on both the training and test data.

Code:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Create a pipeline
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = Pipeline(steps=[
    ('imputer', numeric_transformer),
    ('encoder', categorical_transformer)
])

# Fit and transform the data
X_transformed = preprocessor.fit_transform(X)
```

These slides cover various feature engineering techniques, including missing value imputation, categorical variable encoding, feature scaling, feature selection, polynomial features, interactions and feature crosses, binning and discretization, logarithmic transformation, ordinal encoding, feature extraction, feature generation, handling text data, and creating a feature engineering pipeline. The examples provide code snippets to demonstrate the implementation of each technique using Python and popular machine learning libraries.

