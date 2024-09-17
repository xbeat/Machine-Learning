## Introducing Simple Linear Regression
Slide 1: Simple Linear Regression

Simple Linear Regression predicts a dependent variable using one independent variable. It's ideal for straightforward relationships that can be represented by a line.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_test = np.array([6, 7, 8]).reshape(-1, 1)
y_pred = model.predict(X_test)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.scatter(X_test, y_pred, color='green', label='Predictions')
plt.legend()
plt.show()

print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
```

Mathematical formula (LaTeX): [y = \\beta\_0 + \\beta\_1x + \\epsilon]

Slide 2: Polynomial Linear Regression

Polynomial Regression models non-linear relationships by adding polynomial features to a linear model, making it suitable for curvy trends.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Generate sample data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = 2 * X**2 + X + 3 + np.random.randn(8, 1) * 5

# Create and fit the model
degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X, y)

# Generate points for smooth curve
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', label='Polynomial regression')
plt.legend()
plt.show()

print(f"Model coefficients: {model.named_steps['linearregression'].coef_}")
print(f"Intercept: {model.named_steps['linearregression'].intercept_[0]:.2f}")
```

Mathematical formula (LaTeX): [y = \\beta\_0 + \\beta\_1x + \\beta\_2x^2 + ... + \\beta\_nx^n + \\epsilon]

Slide 3: Multiple Linear Regression

Multiple Linear Regression predicts an outcome using several independent variables, allowing for more complex modeling of relationships.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(0)
X = np.random.rand(100, 3)
y = 2 * X[:, 0] + 0.5 * X[:, 1] - 1 * X[:, 2] + np.random.randn(100) * 0.1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Mean squared error: {mse:.2f}")
print(f"R-squared score: {r2:.2f}")
```

Mathematical formula (LaTeX): [y = \\beta\_0 + \\beta\_1x\_1 + \\beta\_2x\_2 + ... + \\beta\_nx\_n + \\epsilon]

Slide 4: Ridge Regression (L2 Regularization)

Ridge Regression applies L2 regularization to reduce overfitting while keeping all features. It's useful when you want to retain all variables but minimize their impact.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(0)
X = np.random.randn(100, 20)
y = np.sum(X[:, :5], axis=1) + np.random.randn(100) * 0.1

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit the model
alpha = 1.0
model = Ridge(alpha=alpha)
model.fit(X_train_scaled, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean squared error: {mse:.4f}")
print(f"Number of non-zero coefficients: {np.sum(model.coef_ != 0)}")
```

Mathematical formula (LaTeX): [\\min\_{\\beta} |y - X\\beta|^2\_2 + \\alpha|\\beta|^2\_2]

Slide 5: Lasso Regression (L1 Regularization)

Lasso Regression uses L1 regularization to perform feature selection by driving some coefficients to zero, effectively selecting only the most important features.

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(0)
X = np.random.randn(100, 20)
y = np.sum(X[:, :5], axis=1) + np.random.randn(100) * 0.1

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit the model
alpha = 0.1
model = Lasso(alpha=alpha)
model.fit(X_train_scaled, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean squared error: {mse:.4f}")
print(f"Number of non-zero coefficients: {np.sum(model.coef_ != 0)}")
```

Mathematical formula (LaTeX): [\\min\_{\\beta} |y - X\\beta|^2\_2 + \\alpha|\\beta|\_1]

Slide 6: Elastic Net Regression

Elastic Net combines L1 and L2 regularization, providing both feature selection and regularization. It's useful when you want a balance between Ridge and Lasso regression.

```python
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(0)
X = np.random.randn(100, 20)
y = np.sum(X[:, :5], axis=1) + np.random.randn(100) * 0.1

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit the model
alpha = 0.1
l1_ratio = 0.5  # Balance between L1 and L2
model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
model.fit(X_train_scaled, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean squared error: {mse:.4f}")
print(f"Number of non-zero coefficients: {np.sum(model.coef_ != 0)}")
```

Mathematical formula (LaTeX): [\\min\_{\\beta} |y - X\\beta|^2\_2 + \\alpha\\rho|\\beta|\_1 + \\frac{\\alpha(1-\\rho)}{2}|\\beta|^2\_2]

Slide 7: Logistic Regression

Logistic Regression is used for binary classification problems, estimating the probability of an instance belonging to a particular class.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Generate sample data
np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# Plot decision boundary
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                       np.arange(x2_min, x2_max, 0.1))
Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression Decision Boundary")
plt.show()
```

Mathematical formula (LaTeX): [P(y=1|x) = \\frac{1}{1 + e^{-(\\beta\_0 + \\beta\_1x\_1 + ... + \\beta\_nx\_n)}}]

Slide 8: Multinomial Logistic Regression

Multinomial Logistic Regression extends logistic regression to handle classification problems with more than two categories.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit the model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(class_report)

# Display probabilities for a sample
sample = X_test_scaled[0].reshape(1, -1)
probs = model.predict_proba(sample)
print("\nProbabilities for a sample:")
for i, prob in enumerate(probs[0]):
    print(f"{iris.target_names[i]}: {prob:.4f}")
```

Mathematical formula (LaTeX): [P(y=k|x) = \\frac{e^{\\beta\_k^T x}}{\\sum\_{j=1}^K e^{\\beta\_j^T x}}]

Slide 9: Real-Life Example: Simple Linear Regression

Let's use Simple Linear Regression to predict a person's height based on their shoe size.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample data: shoe sizes and heights
shoe_sizes = np.array([7, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12])
heights = np.array([65, 67, 68, 69, 70, 72, 73, 74, 75, 76])

X = shoe_sizes.reshape(-1, 1)
y = heights

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean squared error: {mse:.2f}")
print(f"R-squared score: {r2:.2f}")
print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('Shoe Size')
plt.ylabel('Height (inches)')
plt.title('Height Prediction based on Shoe Size')
plt.legend()
plt.show()

new_shoe_size = np.array([[10.75]])
predicted_height = model.predict(new_shoe_size)
print(f"Predicted height for shoe size 10.75: {predicted_height[0]:.2f} inches")
```

This example demonstrates how Simple Linear Regression can be used to predict a person's height based on their shoe size, showing a practical application of the technique in anthropometry.

Slide 10: Real-Life Example: Multiple Linear Regression

Let's use Multiple Linear Regression to predict fuel efficiency (mpg) based on a car's characteristics.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample data: car characteristics and mpg
data = {
    'weight': [2800, 3100, 2900, 3200, 2950, 3150, 3000, 3300, 2850, 3250],
    'horsepower': [130, 165, 140, 190, 150, 180, 160, 200, 135, 185],
    'engine_size': [2.0, 2.5, 2.2, 3.0, 2.3, 2.8, 2.4, 3.2, 2.1, 2.9],
    'mpg': [28, 25, 26, 22, 27, 24, 25, 21, 29, 23]
}

df = pd.DataFrame(data)

X = df[['weight', 'horsepower', 'engine_size']]
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean squared error: {mse:.2f}")
print(f"R-squared score: {r2:.2f}")
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

new_car = np.array([[3050, 170, 2.6]])
predicted_mpg = model.predict(new_car)
print(f"Predicted MPG for new car: {predicted_mpg[0]:.2f}")
```

This example shows how Multiple Linear Regression can be used to predict a car's fuel efficiency based on multiple characteristics, demonstrating its application in automotive engineering and environmental studies.

Slide 11: Polynomial Regression in Climate Science

Polynomial Regression can be used to model non-linear trends in climate data, such as the relationship between atmospheric CO2 levels and global temperature anomalies.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Sample data: CO2 levels (ppm) and temperature anomalies (°C)
co2_levels = np.array([315, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365, 370])
temp_anomalies = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9])

X = co2_levels.reshape(-1, 1)
y = temp_anomalies

degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X, y)

X_seq = np.linspace(310, 375, 100).reshape(-1, 1)
y_pred = model.predict(X_seq)

plt.scatter(X, y, color='blue', label='Observed data')
plt.plot(X_seq, y_pred, color='red', label=f'Polynomial regression (degree {degree})')
plt.xlabel('CO2 Levels (ppm)')
plt.ylabel('Temperature Anomaly (°C)')
plt.title('CO2 Levels vs Temperature Anomalies')
plt.legend()
plt.show()

new_co2_level = np.array([[380]])
predicted_temp_anomaly = model.predict(new_co2_level)
print(f"Predicted temperature anomaly for CO2 level of 380 ppm: {predicted_temp_anomaly[0]:.2f}°C")
```

This example demonstrates how Polynomial Regression can be applied to model the non-linear relationship between CO2 levels and temperature anomalies, which is crucial in climate science research.

Slide 12: Logistic Regression in Medical Diagnosis

Logistic Regression can be used in medical diagnosis to predict the likelihood of a disease based on various symptoms or risk factors.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample data: patient characteristics and disease presence
# Features: age, blood pressure, cholesterol level
X = np.array([
    [45, 120, 180], [50, 140, 200], [35, 110, 160],
    [55, 130, 220], [60, 150, 240], [40, 125, 190],
    [65, 160, 260], [38, 115, 170], [58, 145, 230],
    [42, 135, 210]
])
# Target: disease presence (0: absent, 1: present)
y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

new_patient = np.array([[52, 135, 215]])
disease_probability = model.predict_proba(new_patient)[0][1]
print(f"Probability of disease for new patient: {disease_probability:.2f}")
```

This example shows how Logistic Regression can be applied in medical diagnosis to predict the likelihood of a disease based on patient characteristics, demonstrating its importance in healthcare decision-making.

Slide 13: Ridge Regression in Genomics

Ridge Regression can be useful in genomics for predicting traits based on gene expression data, especially when dealing with many genes (features) and relatively few samples.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Simulated gene expression data
np.random.seed(42)
n_samples, n_features = 100, 1000
X = np.random.randn(n_samples, n_features)
true_coef = np.random.randn(n_features) * 0.5
true_coef[:-50] = 0  # Assume only 50 genes are actually relevant
y = np.dot(X, true_coef) + np.random.randn(n_samples) * 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

alpha = 1.0
model = Ridge(alpha=alpha)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean squared error: {mse:.4f}")
print(f"R-squared score: {r2:.4f}")
print(f"Number of non-zero coefficients: {np.sum(np.abs(model.coef_) > 1e-5)}")

# Identify top genes
top_genes = np.argsort(np.abs(model.coef_))[-10:]
print("Top 10 most influential genes (indices):")
print(top_genes)
```

This example demonstrates how Ridge Regression can be applied in genomics to predict traits based on gene expression data while handling the high-dimensionality challenge often encountered in this field.

Slide 14: Additional Resources

For those interested in diving deeper into regression techniques and their applications, here are some valuable resources:

1. "A Comprehensive Guide to Linear Regression" by Gareth James et al. (2013) arXiv:1309.6886 \[stat.ML\]
2. "Regularization Paths for Generalized Linear Models via Coordinate Descent" by Jerome Friedman et al. (2010) arXiv:0708.1485 \[stat.ML\]
3. "An Introduction to Statistical Learning" by Gareth James et al. (2013) Available at: [https://www.statlearning.com/](https://www.statlearning.com/)
4. "The Elements of Statistical Learning" by Trevor Hastie et al. (2009) Available at: [https://web.stanford.edu/~hastie/ElemStatLearn/](https://web.stanford.edu/~hastie/ElemStatLearn/)

These resources provide in-depth explanations of various regression techniques, their mathematical foundations, and practical applications in different fields.

