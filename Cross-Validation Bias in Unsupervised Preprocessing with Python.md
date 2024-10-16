## Cross-Validation Bias in Unsupervised Preprocessing with Python
Slide 1: Introduction to Cross-Validation Bias in Unsupervised Preprocessing

Cross-validation is a widely used technique for model evaluation and selection. However, when combined with unsupervised preprocessing, it can lead to biased results. This presentation explores the potential pitfalls and solutions to this problem.

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Generate sample data
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

# Incorrect preprocessing (applied before cross-validation)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Biased cross-validation
model = LogisticRegression()
biased_scores = cross_val_score(model, X_scaled, y, cv=5)

print(f"Biased CV scores: {biased_scores.mean():.3f} ± {biased_scores.std():.3f}")
```

Slide 2: Understanding Unsupervised Preprocessing

Unsupervised preprocessing techniques, such as standardization or normalization, are commonly used to improve model performance. These methods don't require labels and are applied to the entire dataset before model training.

```python
from sklearn.preprocessing import MinMaxScaler

# Example of unsupervised preprocessing
X_raw = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

print("Raw data:")
print(X_raw)
print("\nScaled data:")
print(X_scaled)
```

Slide 3: The Problem with Naive Cross-Validation

When preprocessing is applied before cross-validation, information from the test set leaks into the training process, leading to overly optimistic performance estimates.

```python
from sklearn.model_selection import train_test_split

# Generate data
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

# Naive approach (incorrect)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data after scaling (information leakage)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
naive_score = model.score(X_test, y_test)

print(f"Naive approach score: {naive_score:.3f}")
```

Slide 4: Correct Cross-Validation with Preprocessing

To avoid bias, preprocessing should be performed within each fold of the cross-validation process, ensuring that the test set remains truly unseen.

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Create a pipeline that includes preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Correct cross-validation
correct_scores = cross_val_score(pipeline, X, y, cv=5)

print(f"Correct CV scores: {correct_scores.mean():.3f} ± {correct_scores.std():.3f}")
```

Slide 5: Impact of Biased Cross-Validation

The bias in cross-validation can lead to overestimating model performance, potentially resulting in poor generalization to new, unseen data.

```python
import matplotlib.pyplot as plt

# Generate larger dataset
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

# Biased approach
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
biased_scores = cross_val_score(LogisticRegression(), X_scaled, y, cv=10)

# Correct approach
pipeline = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
correct_scores = cross_val_score(pipeline, X, y, cv=10)

plt.boxplot([biased_scores, correct_scores], labels=['Biased', 'Correct'])
plt.title('Comparison of Biased and Correct Cross-Validation')
plt.ylabel('Accuracy')
plt.show()
```

Slide 6: Real-Life Example: Image Classification

In image classification tasks, preprocessing steps like normalization are crucial. Applying these steps incorrectly can lead to biased performance estimates.

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# Load digit dataset
digits = load_digits()
X, y = digits.data, digits.target

# Incorrect approach: PCA before cross-validation
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)

biased_scores = cross_val_score(LogisticRegression(), X_pca, y, cv=5)

# Correct approach: PCA within cross-validation
pipeline = Pipeline([('pca', PCA(n_components=0.95)), ('clf', LogisticRegression())])
correct_scores = cross_val_score(pipeline, X, y, cv=5)

print(f"Biased scores: {biased_scores.mean():.3f} ± {biased_scores.std():.3f}")
print(f"Correct scores: {correct_scores.mean():.3f} ± {correct_scores.std():.3f}")
```

Slide 7: Real-Life Example: Text Classification

Text classification often involves preprocessing steps like TF-IDF transformation. Applying these steps incorrectly can lead to information leakage and biased results.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

# Load text data
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# Incorrect approach: TF-IDF before cross-validation
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(twenty_train.data)

biased_scores = cross_val_score(LogisticRegression(), X_tfidf, twenty_train.target, cv=5)

# Correct approach: TF-IDF within cross-validation
pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
correct_scores = cross_val_score(pipeline, twenty_train.data, twenty_train.target, cv=5)

print(f"Biased scores: {biased_scores.mean():.3f} ± {biased_scores.std():.3f}")
print(f"Correct scores: {correct_scores.mean():.3f} ± {correct_scores.std():.3f}")
```

Slide 8: Detecting Cross-Validation Bias

To detect potential bias, compare the performance of models trained with preprocessing applied before and within cross-validation. Significant differences may indicate bias.

```python
from sklearn.model_selection import KFold

def detect_cv_bias(X, y, preprocessor, model, cv=5):
    # Biased approach
    X_preprocessed = preprocessor.fit_transform(X)
    biased_scores = cross_val_score(model, X_preprocessed, y, cv=cv)
    
    # Correct approach
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    correct_scores = cross_val_score(pipeline, X, y, cv=cv)
    
    print(f"Biased scores: {biased_scores.mean():.3f} ± {biased_scores.std():.3f}")
    print(f"Correct scores: {correct_scores.mean():.3f} ± {correct_scores.std():.3f}")
    
    if np.abs(biased_scores.mean() - correct_scores.mean()) > 0.05:
        print("Warning: Potential cross-validation bias detected!")

# Example usage
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)
detect_cv_bias(X, y, StandardScaler(), LogisticRegression())
```

Slide 9: Nested Cross-Validation

Nested cross-validation can help mitigate bias when both model selection and evaluation are needed. It involves an outer loop for evaluation and an inner loop for model selection.

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

def nested_cv(X, y, preprocessor, model, param_grid, cv_outer=5, cv_inner=3):
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    
    # Inner loop
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv_inner)
    
    # Outer loop
    nested_scores = cross_val_score(grid_search, X, y, cv=cv_outer)
    
    return nested_scores

# Example usage
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

param_grid = {'model__C': [0.1, 1, 10]}
nested_scores = nested_cv(X, y, StandardScaler(), LogisticRegression(), param_grid)

print(f"Nested CV scores: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")
```

Slide 10: Feature Selection and Cross-Validation

Feature selection is another preprocessing step that can introduce bias if not handled correctly within cross-validation.

```python
from sklearn.feature_selection import SelectKBest, f_classif

def biased_feature_selection(X, y, k=5):
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    return cross_val_score(LogisticRegression(), X_selected, y, cv=5)

def correct_feature_selection(X, y, k=5):
    pipeline = Pipeline([
        ('selector', SelectKBest(f_classif, k=k)),
        ('classifier', LogisticRegression())
    ])
    return cross_val_score(pipeline, X, y, cv=5)

# Generate data
X = np.random.rand(1000, 20)
y = np.random.randint(0, 2, 1000)

biased_scores = biased_feature_selection(X, y)
correct_scores = correct_feature_selection(X, y)

print(f"Biased feature selection: {biased_scores.mean():.3f} ± {biased_scores.std():.3f}")
print(f"Correct feature selection: {correct_scores.mean():.3f} ± {correct_scores.std():.3f}")
```

Slide 11: Handling Imbalanced Datasets

When dealing with imbalanced datasets, preprocessing techniques like oversampling or undersampling can introduce bias if not properly incorporated into the cross-validation process.

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Create imbalanced dataset
X = np.random.rand(1000, 10)
y = np.concatenate([np.ones(950), np.zeros(50)])

# Biased approach (incorrect)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
biased_scores = cross_val_score(LogisticRegression(), X_resampled, y_resampled, cv=5)

# Correct approach
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression())
])
correct_scores = cross_val_score(pipeline, X, y, cv=5)

print(f"Biased scores: {biased_scores.mean():.3f} ± {biased_scores.std():.3f}")
print(f"Correct scores: {correct_scores.mean():.3f} ± {correct_scores.std():.3f}")
```

Slide 12: Time Series Cross-Validation

For time series data, special care must be taken to preserve the temporal order and avoid using future information in preprocessing steps.

```python
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Generate time series data
dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
X = pd.DataFrame({'value': np.random.randn(len(dates))}, index=dates)
y = (X['value'].rolling(window=7).mean() > 0).astype(int)

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

def time_series_cv_score(X, y, pipeline):
    scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        pipeline.fit(X_train, y_train)
        scores.append(pipeline.score(X_test, y_test))
    
    return np.mean(scores), np.std(scores)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

mean_score, std_score = time_series_cv_score(X, y, pipeline)
print(f"Time series CV score: {mean_score:.3f} ± {std_score:.3f}")
```

Slide 13: Best Practices for Unbiased Cross-Validation

To ensure unbiased cross-validation when using unsupervised preprocessing:

1. Use scikit-learn's Pipeline class to combine preprocessing steps with the model.
2. Perform cross-validation on the entire pipeline, not just the model.
3. Be cautious when using feature selection or dimensionality reduction techniques.
4. For time series data, use appropriate cross-validation strategies that respect temporal order.
5. Regularly compare results with and without preprocessing to detect potential bias.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Example of a complex pipeline with multiple preprocessing steps
numeric_features = ['age', 'bmi', 'blood_pressure']
categorical_features = ['gender', 'smoker']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Now you can safely use cross_val_score or cross_validate on this pipeline
```

Slide 14: Additional Resources

For further reading on cross-validation bias and related topics, consider the following resources:

1. Cawley, G. C., & Talbot, N. L. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. Journal of Machine Learning Research, 11(Jul), 2079-2107. ArXiv: [https://arxiv.org/abs/0810.5576](https://arxiv.org/abs/0810.5576)
2. Krstajic, D., Buturovic, L. J., Leahy, D. E., & Thomas, S. (2014). Cross-validation pitfalls when selecting and assessing regression and classification models. Journal of cheminformatics, 6(1), 10. DOI: [https://doi.org/10.1186/1758-2946-6-10](https://doi.org/10.1186/1758-2946-6-10)
3. Tsamardinos, I., Greasidou, E., & Borboudakis, G. (2018

