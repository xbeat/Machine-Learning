## Lasso Regression in scikit-learn
Slide 1: Lasso Regression Overview

Linear regression with L1 regularization, also known as Lasso (Least Absolute Shrinkage and Selection Operator), adds a penalty term to the loss function that encourages sparse solutions by driving some coefficients exactly to zero, effectively performing feature selection.

```python
from sklearn.linear_model import Lasso
import numpy as np

# Initialize Lasso regression with alpha (regularization strength)
lasso = Lasso(alpha=1.0)

# Mathematical formulation (not rendered):
# $$\min_{w} \frac{1}{2n} ||y - Xw||_2^2 + \alpha ||w||_1$$
```

Slide 2: Basic Lasso Implementation

Implementing Lasso regression with sklearn involves data preparation, model training, and prediction phases. The alpha parameter controls the strength of regularization, with higher values producing sparser solutions.

```python
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train Lasso model
lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train, y_train)

# Make predictions
y_pred = lasso.predict(X_test)
```

Slide 3: Feature Selection with Lasso

Lasso regression naturally performs feature selection by shrinking less important feature coefficients to exactly zero. This property makes it particularly useful for high-dimensional datasets where feature selection is crucial.

```python
import pandas as pd

# Create feature names
feature_names = [f'Feature_{i}' for i in range(20)]

# Get non-zero coefficients
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lasso.coef_
})

# Display non-zero coefficients
selected_features = coef_df[coef_df['Coefficient'] != 0]
print("Selected features and their coefficients:")
print(selected_features)
```

Slide 4: Cross-Validation for Alpha Selection

The optimal regularization parameter alpha is crucial for Lasso's performance. Cross-validation helps select the best alpha value by evaluating model performance across different regularization strengths.

```python
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

# Create LassoCV object
lasso_cv = LassoCV(cv=5, random_state=42, alphas=np.logspace(-4, 1, 100))

# Fit model
lasso_cv.fit(X_train, y_train)

print(f"Best alpha: {lasso_cv.alpha_}")
print(f"MSE: {mean_squared_error(y_test, lasso_cv.predict(X_test))}")
```

Slide 5: Real-world Example - Housing Prices

A practical implementation of Lasso regression for predicting housing prices, demonstrating data preprocessing, model training, and evaluation on real estate data.

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load housing dataset (example with synthetic data)
data = pd.DataFrame({
    'price': np.random.normal(200000, 50000, 1000),
    'sqft': np.random.normal(1500, 500, 1000),
    'bedrooms': np.random.randint(1, 6, 1000),
    'bathrooms': np.random.randint(1, 4, 1000),
    'age': np.random.normal(20, 10, 1000)
})

# Prepare features and target
X = data.drop('price', axis=1)
y = data['price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Slide 6: Source Code for Housing Price Model

```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Create and train model with cross-validation
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)

# Make predictions
y_pred = lasso_cv.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = lasso_cv.score(X_test, y_test)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print("\nFeature Coefficients:")
for name, coef in zip(X.columns, lasso_cv.coef_):
    print(f"{name}: {coef:.4f}")
```

Slide 7: Elastic Net Integration

ElasticNet combines L1 and L2 regularization, providing a middle ground between Lasso and Ridge regression. This hybrid approach helps handle correlated features while maintaining the feature selection capability of Lasso.

```python
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

# Create ElasticNetCV model
elastic_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
    alphas=np.logspace(-4, 1, 100),
    cv=5,
    random_state=42
)

# Fit model
elastic_cv.fit(X_train, y_train)

print(f"Best alpha: {elastic_cv.alpha_}")
print(f"Best l1_ratio: {elastic_cv.l1_ratio_}")
```

Slide 8: Comparing Lasso with Other Regularization Methods

Implementation comparing Lasso against Ridge and ElasticNet regression to understand their different effects on feature coefficients and model performance.

```python
from sklearn.linear_model import Ridge, ElasticNet
import matplotlib.pyplot as plt

# Initialize models
models = {
    'Lasso': Lasso(alpha=0.1),
    'Ridge': Ridge(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# Train and collect coefficients
coefs = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    coefs[name] = model.coef_

# Plot coefficient comparison
plt.figure(figsize=(12, 6))
for name, coef in coefs.items():
    plt.plot(range(len(coef)), coef, label=name, marker='o')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.legend()
plt.title('Coefficient Comparison Across Models')
plt.grid(True)
```

Slide 9: Real-world Example - Gene Expression Analysis

Implementing Lasso regression for gene expression data analysis, where feature selection is crucial due to the high dimensionality of genetic data.

```python
# Generate synthetic gene expression data
n_samples = 100
n_features = 1000
n_informative = 50

# Create synthetic gene expression dataset
np.random.seed(42)
X_genes = np.random.normal(0, 1, (n_samples, n_features))
informative_features = np.random.choice(n_features, n_informative, replace=False)
beta = np.zeros(n_features)
beta[informative_features] = np.random.normal(0, 1, n_informative)
y_expression = np.dot(X_genes, beta) + np.random.normal(0, 0.1, n_samples)
```

Slide 10: Source Code for Gene Expression Analysis

```python
# Scale features
scaler = StandardScaler()
X_genes_scaled = scaler.fit_transform(X_genes)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_genes_scaled, y_expression, test_size=0.2, random_state=42
)

# Create and train LassoCV
lasso_genes = LassoCV(
    cv=5,
    random_state=42,
    max_iter=10000,
    n_jobs=-1
)

# Fit model
lasso_genes.fit(X_train, y_train)

# Get selected features
selected_genes = np.where(lasso_genes.coef_ != 0)[0]
print(f"Number of selected genes: {len(selected_genes)}")
print(f"Selected gene indices: {selected_genes[:10]}...")
```

Slide 11: Stability Selection with Lasso

Implementation of stability selection to improve feature selection reliability by running Lasso multiple times on bootstrapped samples.

```python
from sklearn.utils import resample

def stability_selection(X, y, n_bootstraps=100, threshold=0.5):
    feature_selection_counts = np.zeros(X.shape[1])
    
    for _ in range(n_bootstraps):
        # Bootstrap sample
        X_boot, y_boot = resample(X, y)
        
        # Fit Lasso
        lasso = Lasso(alpha=0.1, random_state=42)
        lasso.fit(X_boot, y_boot)
        
        # Count selected features
        feature_selection_counts += (lasso.coef_ != 0).astype(int)
    
    # Calculate selection probability
    selection_probability = feature_selection_counts / n_bootstraps
    stable_features = np.where(selection_probability >= threshold)[0]
    
    return stable_features, selection_probability

# Run stability selection
stable_features, probabilities = stability_selection(X_train, y_train)
print(f"Number of stable features: {len(stable_features)}")
```

Slide 12: Lasso Path Visualization

The Lasso path shows how feature coefficients change with different regularization strengths, providing insights into feature importance and selection order as regularization increases or decreases.

```python
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt

# Compute Lasso path
alphas, coefs, _ = lasso_path(X_train, y_train, alphas=np.logspace(-4, 1, 100))

# Plot Lasso paths
plt.figure(figsize=(12, 6))
for coef_path in coefs:
    plt.plot(np.log10(alphas), coef_path, alpha=0.5)
    
plt.xlabel('log(alpha)')
plt.ylabel('Coefficients')
plt.title('Lasso Path')
plt.grid(True)
plt.axvline(np.log10(lasso_cv.alpha_), color='k', linestyle='--')
```

Slide 13: Lasso with Sparse Input

Implementation demonstrating Lasso's effectiveness with sparse matrices, commonly encountered in text processing and high-dimensional data analysis.

```python
from scipy.sparse import csr_matrix
import numpy as np

# Create sparse matrix
n_samples, n_features = 1000, 5000
density = 0.01

# Generate sparse feature matrix
X_sparse = csr_matrix((n_samples, n_features))
for i in range(n_samples):
    n_nonzero = int(n_features * density)
    indices = np.random.choice(n_features, n_nonzero, replace=False)
    values = np.random.randn(n_nonzero)
    X_sparse[i, indices] = values

# Generate target variable
true_coef = np.zeros(n_features)
true_coef[np.random.choice(n_features, 10, replace=False)] = np.random.randn(10)
y_sparse = X_sparse.dot(true_coef) + np.random.normal(0, 0.1, n_samples)
```

Slide 14: Source Code for Sparse Data Analysis

```python
# Split sparse data
X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse = train_test_split(
    X_sparse, y_sparse, test_size=0.2, random_state=42
)

# Train Lasso on sparse data
lasso_sparse = Lasso(alpha=0.1, random_state=42)
lasso_sparse.fit(X_train_sparse, y_train_sparse)

# Evaluate performance
sparse_pred = lasso_sparse.predict(X_test_sparse)
sparse_mse = mean_squared_error(y_test_sparse, sparse_pred)
print(f"MSE on sparse data: {sparse_mse:.4f}")
print(f"Number of non-zero coefficients: {np.sum(lasso_sparse.coef_ != 0)}")
```

Slide 15: Additional Resources

*   "The Elements of Statistical Learning": [https://web.stanford.edu/~hastie/ElemStatLearn/](https://web.stanford.edu/~hastie/ElemStatLearn/)
*   "High-Dimensional Data Analysis with Lasso": [https://arxiv.org/abs/1903.01122](https://arxiv.org/abs/1903.01122)
*   "Regularization Paths for Generalized Linear Models": [https://arxiv.org/abs/1712.01947](https://arxiv.org/abs/1712.01947)
*   "Sparse Recovery with L1 Optimization": [https://arxiv.org/abs/1805.09411](https://arxiv.org/abs/1805.09411)
*   Resource for practical implementation: [https://scikit-learn.org/stable/modules/linear\_model.html](https://scikit-learn.org/stable/modules/linear_model.html)

