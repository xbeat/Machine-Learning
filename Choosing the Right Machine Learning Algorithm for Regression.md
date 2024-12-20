## Choosing the Right Machine Learning Algorithm for Regression
Slide 1: Regression Analysis Overview

Regression analysis forms the foundation of predictive modeling, enabling us to understand relationships between variables and make quantitative predictions. We'll explore implementing multiple regression techniques using Python's scikit-learn library, focusing on practical implementation with real-world datasets.

```python
# Basic regression analysis setup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample dataset
np.random.seed(42)
X = np.random.randn(100, 3)
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(100)*0.1

# Preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Mathematical representation of linear regression
'''
$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon$$
Where:
$$\beta_0$$ is the intercept
$$\beta_i$$ are the coefficients
$$\epsilon$$ is the error term
'''
```

Slide 2: Stochastic Gradient Descent Implementation

Stochastic Gradient Descent (SGD) is an efficient optimization method for fitting linear regression models on large datasets. It updates model parameters iteratively using individual training examples, making it memory-efficient and suitable for online learning scenarios.

```python
from sklearn.linear_model import SGDRegressor

# Initialize and train SGD regressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty='l2', eta0=0.01)
sgd_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_sgd = sgd_reg.predict(X_test_scaled)

# Evaluate performance
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
r2_sgd = r2_score(y_test, y_pred_sgd)

print(f"MSE: {mse_sgd:.4f}")
print(f"R2 Score: {r2_sgd:.4f}")
```

Slide 3: Least Angle Regression (LARS) Implementation

LARS provides a highly efficient method for computing the entire Lasso path with the same computational cost as a single least squares fit. It's particularly useful when dealing with high-dimensional data where the number of features exceeds observations.

```python
from sklearn.linear_model import LarsCV
import matplotlib.pyplot as plt

# Initialize and train LARS with cross-validation
lars_cv = LarsCV(cv=5, max_iter=100)
lars_cv.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred_lars = lars_cv.predict(X_test_scaled)

# Plot coefficients path
plt.figure(figsize=(10, 6))
plt.plot(lars_cv.coef_path_.T)
plt.xlabel('Step')
plt.ylabel('Coefficients')
plt.title('LARS Coefficient Path')
plt.show()

print(f"Best alpha: {lars_cv.alpha_}")
print(f"R2 Score: {r2_score(y_test, y_pred_lars):.4f}")
```

Slide 4: Lasso and Elastic Net Implementation

Lasso and Elastic Net combine L1 and L2 regularization to handle multicollinearity and perform feature selection. These methods are essential for high-dimensional datasets where feature selection and model interpretability are crucial.

```python
from sklearn.linear_model import LassoCV, ElasticNetCV

# Initialize models with cross-validation
lasso_cv = LassoCV(cv=5, random_state=42)
elastic_cv = ElasticNetCV(cv=5, random_state=42)

# Train models
lasso_cv.fit(X_train_scaled, y_train)
elastic_cv.fit(X_train_scaled, y_train)

# Predictions
y_pred_lasso = lasso_cv.predict(X_test_scaled)
y_pred_elastic = elastic_cv.predict(X_test_scaled)

# Compare results
results = pd.DataFrame({
    'Method': ['Lasso', 'Elastic Net'],
    'R2 Score': [
        r2_score(y_test, y_pred_lasso),
        r2_score(y_test, y_pred_elastic)
    ],
    'Alpha': [lasso_cv.alpha_, elastic_cv.alpha_]
})
print(results)
```

Slide 5: Ridge Regression Implementation

Ridge regression addresses multicollinearity by adding an L2 penalty term to the ordinary least squares objective function. This technique helps prevent overfitting and stabilizes the model when predictors are highly correlated.

```python
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt

# Initialize Ridge regression with cross-validation
alphas = np.logspace(-6, 6, 100)
ridge_cv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)

# Predictions
y_pred_ridge = ridge_cv.predict(X_test_scaled)

# Plot alpha vs MSE
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, ridge_cv.cv_values_.mean(axis=0) * -1)
plt.xlabel('Alpha (regularization strength)')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Regression: Alpha vs MSE')
plt.grid(True)
plt.show()

print(f"Best alpha: {ridge_cv.alpha_}")
print(f"R2 Score: {r2_score(y_test, y_pred_ridge):.4f}")
```

Slide 6: Support Vector Regressor with Linear Kernel

SVR with a linear kernel performs regression using linear support vectors, making it effective for problems where the relationship between features and target is approximately linear while maintaining robust prediction capabilities.

```python
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Initialize Linear SVR
linear_svr = SVR(kernel='linear')

# Parameter grid for optimization
param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.3]
}

# Grid search with cross-validation
grid_search = GridSearchCV(linear_svr, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Best model predictions
y_pred_linear_svr = grid_search.predict(X_test_scaled)

print("Best parameters:", grid_search.best_params_)
print(f"R2 Score: {r2_score(y_test, y_pred_linear_svr):.4f}")
```

Slide 7: Support Vector Regressor with RBF Kernel

The RBF kernel transforms the feature space non-linearly, enabling SVR to capture complex patterns in the data. This implementation demonstrates how to optimize hyperparameters for non-linear regression tasks.

```python
# Initialize RBF SVR
rbf_svr = SVR(kernel='rbf')

# Extended parameter grid for RBF kernel
param_grid_rbf = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 1],
    'epsilon': [0.1, 0.2, 0.3]
}

# Grid search for RBF kernel
grid_search_rbf = GridSearchCV(rbf_svr, param_grid_rbf, cv=5, scoring='neg_mean_squared_error')
grid_search_rbf.fit(X_train_scaled, y_train)

# Predictions with best model
y_pred_rbf_svr = grid_search_rbf.predict(X_test_scaled)

# Compare performance metrics
print("Best parameters:", grid_search_rbf.best_params_)
print(f"R2 Score: {r2_score(y_test, y_pred_rbf_svr):.4f}")

# Mathematical representation of RBF kernel
'''
$$K(x, x') = \exp(-\gamma ||x - x'||^2)$$
Where:
$$\gamma$$ is the kernel coefficient
$$||x - x'||^2$$ is the squared Euclidean distance
'''
```

Slide 8: Decision Tree and Ensemble Methods

Decision trees and ensemble methods combine multiple models to create robust predictors. This implementation showcases Random Forests and Gradient Boosting, two powerful ensemble techniques for regression tasks.

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Initialize models
dt_reg = DecisionTreeRegressor(random_state=42)
rf_reg = RandomForestRegressor(random_state=42)
gb_reg = GradientBoostingRegressor(random_state=42)

# Train models
models = {
    'Decision Tree': dt_reg,
    'Random Forest': rf_reg,
    'Gradient Boosting': gb_reg
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        'R2 Score': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred)
    }

# Display results
results_df = pd.DataFrame(results).T
print(results_df)
```

Slide 9: Ordinary Least Squares Implementation

Ordinary Least Squares (OLS) provides the foundation for linear regression by minimizing the sum of squared residuals. This implementation includes diagnostic tools and statistical tests to evaluate model assumptions and fit quality.

```python
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

# Implement OLS using both scikit-learn and statsmodels
# Scikit-learn implementation
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_ols = lr.predict(X_test_scaled)

# Statsmodels implementation for detailed statistics
X_train_sm = sm.add_constant(X_train_scaled)
model = sm.OLS(y_train, X_train_sm)
results = model.fit()

# Calculate residuals and perform diagnostic tests
residuals = y_test - y_pred_ols
residuals_standardized = (residuals - residuals.mean()) / residuals.std()

# Diagnostic plots and tests
normality_test = stats.normaltest(residuals_standardized)
print(results.summary())
print(f"\nNormality test p-value: {normality_test.pvalue:.4f}")
```

Slide 10: Linear Support Vector Classification

Linear SVC implements support vector classification using a linear kernel, offering efficient classification for linearly separable data with built-in regularization and margin optimization capabilities.

```python
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Generate classification dataset
np.random.seed(42)
X_class = np.random.randn(200, 2)
y_class = (X_class[:, 0] + X_class[:, 1] > 0).astype(int)

# Split and scale data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class)
scaler = StandardScaler()
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

# Train Linear SVC
linear_svc = LinearSVC(dual=False, random_state=42)
linear_svc.fit(X_train_c_scaled, y_train_c)

# Predictions and evaluation
y_pred_svc = linear_svc.predict(X_test_c_scaled)

print("Classification Report:")
print(classification_report(y_test_c, y_pred_svc))

# Mathematical representation
'''
$$\min_{w, b} \frac{1}{2} ||w||^2 + C \sum_{i=1}^n \max(0, 1 - y_i(w^Tx_i + b))$$
Where:
$$w$$ is the normal vector to the hyperplane
$$b$$ is the bias term
$$C$$ is the penalty parameter
'''
```

Slide 11: Naive Bayes Implementation

Naive Bayes classifiers implement Bayes' theorem with strong independence assumptions between features. This implementation shows Gaussian, Multinomial, and Bernoulli variants for different data distributions.

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import MinMaxScaler

# Initialize different Naive Bayes classifiers
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# For MultinomialNB and BernoulliNB, we need non-negative features
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train_c)
X_test_minmax = minmax_scaler.transform(X_test_c)

# Train and evaluate each classifier
classifiers = {
    'Gaussian NB': (gnb, X_train_c_scaled),
    'Multinomial NB': (mnb, X_train_minmax),
    'Bernoulli NB': (bnb, X_train_minmax)
}

results = {}
for name, (clf, X_train_transformed) in classifiers.items():
    clf.fit(X_train_transformed, y_train_c)
    y_pred = clf.predict(X_test_minmax if name != 'Gaussian NB' else X_test_c_scaled)
    results[name] = classification_report(y_test_c, y_pred, output_dict=True)

print(pd.DataFrame(results).round(3))
```

Slide 12: K-Nearest Neighbors Classifier

K-Nearest Neighbors is a versatile non-parametric classifier that makes predictions based on the majority class of the k nearest training samples. This implementation includes distance metrics optimization and neighbor weighting schemes.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Initialize arrays for storing cross-validation scores
k_range = range(1, 31)
cv_scores = []
cv_std = []

# Evaluate different k values
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores = cross_val_score(knn, X_train_c_scaled, y_train_c, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    cv_std.append(scores.std())

# Find optimal k
optimal_k = k_range[np.argmax(cv_scores)]
best_knn = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
best_knn.fit(X_train_c_scaled, y_train_c)

# Mathematical representation
'''
$$d(x, x') = \sqrt{\sum_{i=1}^n (x_i - x'_i)^2}$$
For weighted voting:
$$w_i = \frac{1}{d(x, x_i)^2}$$
'''

# Plot k vs accuracy
plt.figure(figsize=(10, 6))
plt.errorbar(k_range, cv_scores, yerr=cv_std, capsize=5)
plt.xlabel('k value')
plt.ylabel('Cross-validation accuracy')
plt.title('KNN: k vs Classification Accuracy')
print(f"Optimal k: {optimal_k}")
print(f"Best cross-validation score: {max(cv_scores):.4f}")
```

Slide 13: SVC with RBF Kernel Implementation

Support Vector Classification with RBF kernel enables non-linear decision boundaries through implicit feature space transformation. This implementation focuses on kernel parameter optimization and boundary visualization.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

# Initialize SVC with RBF kernel
svc_rbf = SVC(kernel='rbf', probability=True)

# Parameter grid for optimization
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1],
}

# Grid search with cross-validation
grid_search = GridSearchCV(svc_rbf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_c_scaled, y_train_c)

# Best model evaluation
best_svc = grid_search.best_estimator_
y_pred_rbf = best_svc.predict(X_test_c_scaled)
y_prob_rbf = best_svc.predict_proba(X_test_c_scaled)

# Calculate and plot decision boundary
def plot_decision_boundary(model, X, y):
    h = 0.02  # step size in mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('RBF SVC Decision Boundary')

plot_decision_boundary(best_svc, X_test_c_scaled, y_test_c)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Accuracy score: {grid_search.best_score_:.4f}")
```

Slide 14: Additional Resources

*   "A Tutorial on Support Vector Machines for Pattern Recognition" - [https://www.research.microsoft.com/pubs/67119/svmtutorial.pdf](https://www.research.microsoft.com/pubs/67119/svmtutorial.pdf)
*   "Random Forests" by Leo Breiman - [https://link.springer.com/article/10.1023/A:1010933404324](https://link.springer.com/article/10.1023/A:1010933404324)
*   "Gradient Boosting Machines: A Tutorial" - [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
*   "An Introduction to Statistical Learning" - [https://www.statlearning.com/](https://www.statlearning.com/)
*   "Pattern Recognition and Machine Learning" - [https://www.springer.com/gp/book/9780387310732](https://www.springer.com/gp/book/9780387310732)

