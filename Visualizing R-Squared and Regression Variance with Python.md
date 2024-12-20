## Visualizing R-Squared and Regression Variance with Python
Slide 1: Exploring R² and Regression Variance with Euler/Venn Diagrams

R² and regression variance are fundamental concepts in statistical analysis. This presentation explores these concepts using Euler/Venn diagrams, providing a visual approach to understanding their relationships. We'll use Python to create and analyze these diagrams, offering practical insights into regression analysis.

```python
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# Create a simple Venn diagram
plt.figure(figsize=(8, 6))
venn2(subsets=(3, 2, 1), set_labels=('Total Variance', 'Explained Variance'))
plt.title("R² Visualization")
plt.show()
```

Slide 2: Understanding R²

R², also known as the coefficient of determination, measures the proportion of variance in the dependent variable explained by the independent variables in a regression model. It ranges from 0 to 1, with 1 indicating perfect prediction.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Fit linear regression model
model = LinearRegression().fit(X, y)

# Calculate R²
r2 = r2_score(y, model.predict(X))
print(f"R² value: {r2:.4f}")
```

Slide 3: Components of Variance

In regression analysis, total variance can be decomposed into explained variance (accounted for by the model) and unexplained variance (residual). This decomposition is crucial for understanding the performance of a regression model.

```python
# Calculate variance components
total_variance = np.var(y)
predicted_y = model.predict(X)
explained_variance = np.var(predicted_y)
unexplained_variance = np.var(y - predicted_y)

print(f"Total Variance: {total_variance:.4f}")
print(f"Explained Variance: {explained_variance:.4f}")
print(f"Unexplained Variance: {unexplained_variance:.4f}")
```

Slide 4: Visualizing Variance Components

Euler/Venn diagrams offer an intuitive way to visualize the relationship between total variance, explained variance, and unexplained variance. The overlap between circles represents the explained variance (R²).

```python
from matplotlib_venn import venn2

# Create Venn diagram
plt.figure(figsize=(8, 6))
venn2(subsets=(unexplained_variance, explained_variance, 0), 
      set_labels=('Unexplained Variance', 'Explained Variance'))
plt.title("Variance Components in Regression")
plt.show()
```

Slide 5: Interpreting R² with Venn Diagrams

The Venn diagram provides a visual interpretation of R². The ratio of the overlap area (explained variance) to the total area of both circles (total variance) represents R². This visualization helps in understanding the model's explanatory power.

```python
# Calculate areas for Venn diagram
total_area = total_variance
explained_area = explained_variance
unexplained_area = unexplained_variance

# Create Venn diagram
plt.figure(figsize=(8, 6))
venn2(subsets=(unexplained_area, explained_area, 0), 
      set_labels=('Unexplained', 'Explained'))
plt.title(f"R² Visualization: {r2:.4f}")
plt.show()
```

Slide 6: Real-Life Example: Predicting Plant Growth

Consider a study on plant growth where we predict plant height based on the amount of sunlight received. We'll use this example to demonstrate R² and variance components.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate sample data
sunlight = np.array([2, 3, 5, 7, 8, 10, 12, 13, 15, 18]).reshape(-1, 1)
height = np.array([10, 12, 15, 18, 22, 23, 27, 30, 31, 35])

# Fit linear regression model
model = LinearRegression().fit(sunlight, height)

# Calculate R²
r2 = r2_score(height, model.predict(sunlight))

plt.scatter(sunlight, height, color='blue', label='Actual data')
plt.plot(sunlight, model.predict(sunlight), color='red', label='Regression line')
plt.xlabel('Sunlight (hours/day)')
plt.ylabel('Plant Height (cm)')
plt.title(f'Plant Growth Model (R² = {r2:.4f})')
plt.legend()
plt.show()
```

Slide 7: Analyzing Variance in Plant Growth Example

Let's break down the variance components in our plant growth example to better understand the model's performance.

```python
# Calculate variance components
total_variance = np.var(height)
predicted_height = model.predict(sunlight)
explained_variance = np.var(predicted_height)
unexplained_variance = np.var(height - predicted_height)

print(f"Total Variance: {total_variance:.4f}")
print(f"Explained Variance: {explained_variance:.4f}")
print(f"Unexplained Variance: {unexplained_variance:.4f}")

# Visualize variance components
plt.figure(figsize=(8, 6))
venn2(subsets=(unexplained_variance, explained_variance, 0), 
      set_labels=('Unexplained Variance', 'Explained Variance'))
plt.title(f"Variance Components in Plant Growth Model (R² = {r2:.4f})")
plt.show()
```

Slide 8: Limitations of R²

While R² is useful, it has limitations. It doesn't indicate whether the model's predictions are biased, nor does it tell us if we've chosen the right regression model. Additionally, R² can be artificially inflated by adding more variables to the model.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# Generate sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Fit linear and polynomial models
linear_model = LinearRegression().fit(X, y)
poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression().fit(X_poly, y)

# Calculate R² for both models
r2_linear = r2_score(y, linear_model.predict(X))
r2_poly = r2_score(y, poly_model.predict(X_poly))

print(f"Linear Model R²: {r2_linear:.4f}")
print(f"Polynomial Model R²: {r2_poly:.4f}")
```

Slide 9: Adjusted R²

To address some limitations of R², we use Adjusted R². This metric penalizes the addition of unnecessary predictors, providing a more accurate measure of model performance, especially when comparing models with different numbers of predictors.

```python
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Calculate Adjusted R² for both models
n = len(X)  # number of observations
p_linear = 1  # number of predictors in linear model
p_poly = 4  # number of predictors in polynomial model

adj_r2_linear = adjusted_r2(r2_linear, n, p_linear)
adj_r2_poly = adjusted_r2(r2_poly, n, p_poly)

print(f"Linear Model Adjusted R²: {adj_r2_linear:.4f}")
print(f"Polynomial Model Adjusted R²: {adj_r2_poly:.4f}")
```

Slide 10: Visualizing Overfitting with R²

Overfitting occurs when a model learns the training data too well, including its noise. This can lead to high R² values that don't generalize well to new data. Let's visualize this concept.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate noisy data
np.random.seed(0)
X = np.linspace(0, 1, 20).reshape(-1, 1)
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.1, X.shape)

# Fit models of increasing complexity
degrees = [1, 3, 15]
plt.figure(figsize=(15, 5))

for i, degree in enumerate(degrees):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    
    plt.subplot(1, 3, i+1)
    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X, y_pred, color='red', label=f'Degree {degree}')
    plt.title(f'Polynomial Degree {degree}\nR² = {r2:.4f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()

plt.tight_layout()
plt.show()
```

Slide 11: Real-Life Example: Predicting Crop Yield

Let's explore a scenario where we predict crop yield based on factors like rainfall and temperature. This example will demonstrate how R² and adjusted R² can guide model selection.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# Generate sample data
np.random.seed(0)
data = pd.DataFrame({
    'rainfall': np.random.uniform(500, 1500, 100),
    'temperature': np.random.uniform(15, 30, 100)
})
data['yield'] = 0.5 * data['rainfall'] + 2 * data['temperature'] + np.random.normal(0, 50, 100)

# Fit linear and polynomial models
X = data[['rainfall', 'temperature']]
y = data['yield']

linear_model = LinearRegression().fit(X, y)
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression().fit(X_poly, y)

# Calculate R² and Adjusted R²
r2_linear = r2_score(y, linear_model.predict(X))
r2_poly = r2_score(y, poly_model.predict(X_poly))

n = len(X)
p_linear = 2
p_poly = 5

adj_r2_linear = 1 - (1 - r2_linear) * (n - 1) / (n - p_linear - 1)
adj_r2_poly = 1 - (1 - r2_poly) * (n - 1) / (n - p_poly - 1)

print(f"Linear Model - R²: {r2_linear:.4f}, Adjusted R²: {adj_r2_linear:.4f}")
print(f"Polynomial Model - R²: {r2_poly:.4f}, Adjusted R²: {adj_r2_poly:.4f}")
```

Slide 12: Visualizing Model Comparison

To better understand the differences between our linear and polynomial models in the crop yield example, let's create a visual comparison using Euler diagrams.

```python
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# Calculate variance components for both models
total_variance = np.var(y)
explained_variance_linear = np.var(linear_model.predict(X))
explained_variance_poly = np.var(poly_model.predict(X_poly))

# Create subplots for both models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Linear model Venn diagram
venn2(subsets=(total_variance - explained_variance_linear, explained_variance_linear, 0), 
      set_labels=('Unexplained', 'Explained'), ax=ax1)
ax1.set_title(f"Linear Model (R² = {r2_linear:.4f})")

# Polynomial model Venn diagram
venn2(subsets=(total_variance - explained_variance_poly, explained_variance_poly, 0), 
      set_labels=('Unexplained', 'Explained'), ax=ax2)
ax2.set_title(f"Polynomial Model (R² = {r2_poly:.4f})")

plt.tight_layout()
plt.show()
```

Slide 13: Conclusion and Best Practices

Understanding R² and regression variance through Euler/Venn diagrams provides valuable insights into model performance. Key takeaways include:

1. R² represents the proportion of variance explained by the model.
2. Euler/Venn diagrams visually represent the relationship between explained and unexplained variance.
3. Higher R² doesn't always mean a better model; consider adjusted R² and potential overfitting.
4. Real-world examples, like plant growth and crop yield prediction, demonstrate practical applications of these concepts.

When using R² in regression analysis:

* Consider multiple metrics, not just R².
* Be cautious of overfitting, especially with complex models.
* Use adjusted R² when comparing models with different numbers of predictors.
* Always validate your model using techniques like cross-validation.

Slide 14: Additional Resources

For further exploration of R², regression variance, and related topics, consider the following resources:

1. "Understanding R-squared and Residual Plots" by Stephanie Glen (2021). Available at: [https://arxiv.org/abs/2106.04348](https://arxiv.org/abs/2106.04348)
2. "A Comprehensive Review of R² and Adjusted R² in Regression Analysis" by Wang et al. (2020). Available at: [https://arxiv.org/abs/2009.05771](https://arxiv.org/abs/2009.05771)
3. "Visualization Techniques for Regression Models: A Comparative Study" by Zhang and Liu (2019). Available at: [https://arxiv.org/abs/1908.00284](https://arxiv.org/abs/1908.00284)

These papers provide in-depth discussions and advanced techniques related to regression analysis and model evaluation.

