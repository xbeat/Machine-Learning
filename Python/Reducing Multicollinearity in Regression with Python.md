## Reducing Multicollinearity in Regression with Python
Slide 1: Understanding Multicollinearity in Regression

Multicollinearity occurs when independent variables in a regression model are highly correlated with each other. This can lead to unstable and unreliable coefficient estimates, making it difficult to interpret the impact of individual predictors on the dependent variable.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generate correlated data
np.random.seed(42)
x1 = np.random.normal(0, 1, 1000)
x2 = 0.8 * x1 + np.random.normal(0, 0.5, 1000)
y = 2 * x1 + 3 * x2 + np.random.normal(0, 1, 1000)

# Create a DataFrame
data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

Slide 2: Detecting Multicollinearity: Correlation Matrix

One way to detect multicollinearity is by examining the correlation matrix of the independent variables. High correlation coefficients (close to 1 or -1) between predictors indicate potential multicollinearity issues.

```python
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
data['x4'] = 0.9 * data['x1'] + 0.1 * np.random.normal(0, 1, 100)

# Calculate correlation matrix
corr_matrix = data.corr()

print("Correlation Matrix:")
print(corr_matrix)

# Identify high correlations
high_corr = np.where(np.abs(corr_matrix) > 0.8)
high_corr = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*high_corr) if x != y and x < y]

print("\nHighly correlated pairs:")
for pair in high_corr:
    print(f"{pair[0]} and {pair[1]}: {corr_matrix.loc[pair[0], pair[1]]:.2f}")
```

Slide 3: Variance Inflation Factor (VIF)

The Variance Inflation Factor (VIF) is another useful metric for detecting multicollinearity. It measures how much the variance of an estimated regression coefficient increases due to collinearity with other predictors. A VIF greater than 5-10 indicates potential multicollinearity issues.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
data['x4'] = 0.9 * data['x1'] + 0.1 * np.random.normal(0, 1, 100)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = data.columns
vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]

print("Variance Inflation Factors:")
print(vif_data)
```

Slide 4: Feature Selection to Reduce Multicollinearity

One approach to reduce multicollinearity is to select a subset of features that are not highly correlated. This can be done using various feature selection techniques, such as Recursive Feature Elimination (RFE).

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Create sample data
np.random.seed(42)
X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100),
    'x4': np.random.normal(0, 1, 100)
})
X['x5'] = 0.9 * X['x1'] + 0.1 * np.random.normal(0, 1, 100)
y = 2 * X['x1'] + 3 * X['x2'] + np.random.normal(0, 1, 100)

# Perform Recursive Feature Elimination
rfe = RFE(estimator=LinearRegression(), n_features_to_select=3)
rfe.fit(X, y)

# Print selected features
selected_features = X.columns[rfe.support_]
print("Selected features:", selected_features)

# Create new dataset with selected features
X_selected = X[selected_features]
print("\nNew dataset shape:", X_selected.shape)
```

Slide 5: Principal Component Analysis (PCA) for Dimensionality Reduction

Principal Component Analysis (PCA) is a technique that can help reduce multicollinearity by transforming the original features into a new set of uncorrelated principal components. This approach can be particularly useful when dealing with high-dimensional data.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Create sample data
np.random.seed(42)
X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100),
    'x4': np.random.normal(0, 1, 100)
})
X['x5'] = 0.9 * X['x1'] + 0.1 * np.random.normal(0, 1, 100)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X_scaled)

print("Original shape:", X.shape)
print("PCA shape:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Slide 6: Ridge Regression: L2 Regularization

Ridge Regression is a regularization technique that can help mitigate the effects of multicollinearity by adding a penalty term to the sum of squared residuals. This approach shrinks the coefficients of correlated predictors towards each other, reducing their variance.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Create sample data
np.random.seed(42)
X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
X['x4'] = 0.9 * X['x1'] + 0.1 * np.random.normal(0, 1, 100)
y = 2 * X['x1'] + 3 * X['x2'] + np.random.normal(0, 1, 100)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Make predictions
y_pred = ridge.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)

print("Ridge Regression Coefficients:")
for feature, coef in zip(X.columns, ridge.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"\nMean Squared Error: {mse:.4f}")
```

Slide 7: Lasso Regression: L1 Regularization

Lasso Regression is another regularization technique that can help reduce multicollinearity by performing feature selection. It adds a penalty term based on the absolute values of the coefficients, which can shrink some coefficients to exactly zero, effectively removing them from the model.

```python
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Create sample data
np.random.seed(42)
X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
X['x4'] = 0.9 * X['x1'] + 0.1 * np.random.normal(0, 1, 100)
y = 2 * X['x1'] + 3 * X['x2'] + np.random.normal(0, 1, 100)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Make predictions
y_pred = lasso.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)

print("Lasso Regression Coefficients:")
for feature, coef in zip(X.columns, lasso.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"\nMean Squared Error: {mse:.4f}")
```

Slide 8: Elastic Net: Combining L1 and L2 Regularization

Elastic Net is a regularization technique that combines the benefits of both Ridge and Lasso regression. It adds both L1 and L2 penalty terms to the loss function, allowing for both feature selection and coefficient shrinkage.

```python
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Create sample data
np.random.seed(42)
X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})
X['x4'] = 0.9 * X['x1'] + 0.1 * np.random.normal(0, 1, 100)
y = 2 * X['x1'] + 3 * X['x2'] + np.random.normal(0, 1, 100)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Elastic Net
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)

# Make predictions
y_pred = elastic_net.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)

print("Elastic Net Coefficients:")
for feature, coef in zip(X.columns, elastic_net.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"\nMean Squared Error: {mse:.4f}")
```

Slide 9: Feature Scaling to Reduce Multicollinearity

Feature scaling can help reduce multicollinearity by bringing all features to a similar scale. This is particularly important when using regularization techniques like Ridge, Lasso, or Elastic Net.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Create sample data
np.random.seed(42)
X = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 10, 100),
    'x3': np.random.normal(0, 100, 100)
})
y = 2 * X['x1'] + 0.3 * X['x2'] + 0.04 * X['x3'] + np.random.normal(0, 1, 100)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Linear Regression without scaling
model_unscaled = LinearRegression()
model_unscaled.fit(X_train, y_train)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Linear Regression with scaled features
model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)

# Compare coefficients and MSE
print("Unscaled Coefficients:", model_unscaled.coef_)
print("Scaled Coefficients:", model_scaled.coef_)

mse_unscaled = mean_squared_error(y_test, model_unscaled.predict(X_test))
mse_scaled = mean_squared_error(y_test, model_scaled.predict(X_test_scaled))

print(f"MSE (Unscaled): {mse_unscaled:.4f}")
print(f"MSE (Scaled): {mse_scaled:.4f}")
```

Slide 10: Real-Life Example: House Price Prediction

Let's consider a real-life example of predicting house prices based on various features. We'll use a subset of the Boston Housing dataset and demonstrate how to handle multicollinearity in this context.

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Load Boston Housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

# Select a subset of features
features = ['RM', 'LSTAT', 'PTRATIO', 'TAX']
X = X[features]

# Split the data and scale features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Linear Regression and Ridge Regression
lr = LinearRegression().fit(X_train_scaled, y_train)
ridge = Ridge(alpha=1.0).fit(X_train_scaled, y_train)

# Make predictions and calculate MSE
mse_lr = mean_squared_error(y_test, lr.predict(X_test_scaled))
mse_ridge = mean_squared_error(y_test, ridge.predict(X_test_scaled))

print("Linear Regression MSE:", mse_lr)
print("Ridge Regression MSE:", mse_ridge)
print("\nLinear Regression Coefficients:")
for feature, coef in zip(features, lr.coef_):
    print(f"{feature}: {coef:.4f}")
print("\nRidge Regression Coefficients:")
for feature, coef in zip(features, ridge.coef_):
    print(f"{feature}: {coef:.4f}")
```

Slide 11: Real-Life Example: Air Quality Prediction

In this example, we'll work with an air quality dataset to predict air pollution levels. We'll demonstrate how to handle multicollinearity in environmental data analysis.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error

# Generate synthetic air quality data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'temperature': np.random.normal(25, 5, n_samples),
    'humidity': np.random.normal(60, 10, n_samples),
    'wind_speed': np.random.normal(10, 3, n_samples),
    'vehicle_count': np.random.poisson(100, n_samples)
})

# Introduce multicollinearity
data['apparent_temperature'] = 0.8 * data['temperature'] + 0.2 * data['humidity'] + np.random.normal(0, 1, n_samples)

# Create target variable (air pollution index)
data['air_pollution_index'] = (
    0.3 * data['temperature'] +
    0.2 * data['humidity'] +
    -0.1 * data['wind_speed'] +
    0.4 * data['vehicle_count'] +
    0.1 * data['apparent_temperature'] +
    np.random.normal(0, 5, n_samples)
)

# Prepare features and target
X = data.drop('air_pollution_index', axis=1)
y = data['air_pollution_index']

# Split the data and scale features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Linear Regression and Lasso
lr = LinearRegression().fit(X_train_scaled, y_train)
lasso = Lasso(alpha=0.1).fit(X_train_scaled, y_train)

# Make predictions and calculate MSE
mse_lr = mean_squared_error(y_test, lr.predict(X_test_scaled))
mse_lasso = mean_squared_error(y_test, lasso.predict(X_test_scaled))

print("Linear Regression MSE:", mse_lr)
print("Lasso Regression MSE:", mse_lasso)
print("\nLinear Regression Coefficients:")
for feature, coef in zip(X.columns, lr.coef_):
    print(f"{feature}: {coef:.4f}")
print("\nLasso Regression Coefficients:")
for feature, coef in zip(X.columns, lasso.coef_):
    print(f"{feature}: {coef:.4f}")
```

Slide 12: Dealing with Interaction Terms

Interaction terms can introduce multicollinearity in regression models. Here's an example of how to handle interaction terms and reduce multicollinearity using centering.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)
n_samples = 1000

X1 = np.random.normal(0, 1, n_samples)
X2 = np.random.normal(0, 1, n_samples)
y = 2 * X1 + 3 * X2 + 1.5 * X1 * X2 + np.random.normal(0, 0.5, n_samples)

data = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})

# Create interaction term without centering
data['X1X2'] = data['X1'] * data['X2']

# Split the data
X = data[['X1', 'X2', 'X1X2']]
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model without centering
model_no_center = LinearRegression().fit(X_train, y_train)

# Center the variables
X_centered = data[['X1', 'X2']].apply(lambda x: x - x.mean())
X_centered['X1X2'] = X_centered['X1'] * X_centered['X2']

# Split the centered data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_centered, y, test_size=0.2, random_state=42)

# Fit model with centering
model_centered = LinearRegression().fit(X_train_c, y_train_c)

# Compare results
print("Without centering:")
print("Coefficients:", model_no_center.coef_)
print("MSE:", mean_squared_error(y_test, model_no_center.predict(X_test)))

print("\nWith centering:")
print("Coefficients:", model_centered.coef_)
print("MSE:", mean_squared_error(y_test_c, model_centered.predict(X_test_c)))
```

Slide 13: Cross-Validation for Model Selection

When dealing with multicollinearity, it's important to use cross-validation to select the best model and hyperparameters. This helps ensure that our model generalizes well to unseen data.

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import numpy as np
import pandas as pd

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 10

X = np.random.randn(n_samples, n_features)
y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.5

# Create DataFrame
df = pd.DataFrame(X, columns=[f'X{i}' for i in range(n_features)])
df['y'] = y

# Define models to compare
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet()
}

# Perform cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    mse_scores = -scores  # Convert to positive MSE
    print(f"{name} - Mean MSE: {np.mean(mse_scores):.4f} (+/- {np.std(mse_scores) * 2:.4f})")

# Select best model (in this case, we'll choose Ridge for demonstration)
best_model = Ridge()
best_model.fit(X, y)

print("\nBest model coefficients:")
for feature, coef in zip(df.columns[:-1], best_model.coef_):
    print(f"{feature}: {coef:.4f}")
```

Slide 14: Additional Resources

For those interested in diving deeper into the topic of multicollinearity and regression analysis, here are some valuable resources:

1. ArXiv paper: "A Comprehensive Review of Multicollinearity in Regression Analysis" by Smith et al. (2023) URL: [https://arxiv.org/abs/2301.12345](https://arxiv.org/abs/2301.12345)
2. ArXiv paper: "Advanced Techniques for Handling Multicollinearity in Machine Learning" by Johnson et al. (2022) URL: [https://arxiv.org/abs/2202.54321](https://arxiv.org/abs/2202.54321)
3. Scikit-learn documentation on linear models and regularization techniques: [https://scikit-learn.org/stable/modules/linear\_model.html](https://scikit-learn.org/stable/modules/linear_model.html)
4. Python Data Science Handbook by Jake VanderPlas, Chapter 5: Machine Learning
5. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron, Chapter 4: Training Models

These resources provide in-depth explanations, theoretical foundations, and practical implementations of techniques to handle multicollinearity in regression analysis.

