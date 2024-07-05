## Correlation, Regression, and Curve Fitting in Machine Learning with Python

Slide 1: Introduction to Correlation

Introduction to Correlation

Correlation measures the strength and direction of the relationship between two variables. It's a fundamental concept in statistics and machine learning, particularly useful in exploratory data analysis and feature selection.

Code:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate correlated data
x = np.random.randn(100)
y = 2*x + np.random.randn(100)*0.5

# Calculate correlation coefficient
correlation = np.corrcoef(x, y)[0, 1]

# Plot the data
plt.scatter(x, y)
plt.title(f"Correlation: {correlation:.2f}")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Slide 2: Types of Correlation

Types of Correlation

There are three main types of correlation: positive, negative, and no correlation. Positive correlation means as one variable increases, the other tends to increase. Negative correlation means as one variable increases, the other tends to decrease. No correlation means there's no clear relationship between the variables.

Code:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data for different types of correlation
x = np.linspace(0, 10, 100)
y_positive = x + np.random.randn(100)
y_negative = -x + np.random.randn(100)
y_no_corr = np.random.randn(100)

# Plot the data
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.scatter(x, y_positive)
ax1.set_title("Positive Correlation")

ax2.scatter(x, y_negative)
ax2.set_title("Negative Correlation")

ax3.scatter(x, y_no_corr)
ax3.set_title("No Correlation")

plt.tight_layout()
plt.show()
```

Slide 3: Pearson Correlation Coefficient

Pearson Correlation Coefficient

The Pearson correlation coefficient is the most common measure of correlation. It ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no linear correlation.

Code:

```python
import numpy as np
from scipy import stats

# Generate data
x = np.random.randn(100)
y = 2*x + np.random.randn(100)*0.5

# Calculate Pearson correlation coefficient
pearson_corr, _ = stats.pearsonr(x, y)

print(f"Pearson correlation coefficient: {pearson_corr:.2f}")
```

Slide 4: Spearman Rank Correlation

Spearman Rank Correlation

Spearman rank correlation assesses monotonic relationships between two variables. It's useful when the relationship between variables is not necessarily linear but follows a monotonic function.

Code:

```python
import numpy as np
from scipy import stats

# Generate non-linear but monotonic data
x = np.random.rand(100)
y = np.exp(x) + np.random.randn(100)*0.1

# Calculate Spearman rank correlation
spearman_corr, _ = stats.spearmanr(x, y)

print(f"Spearman rank correlation: {spearman_corr:.2f}")
```

Slide 5: Correlation Matrix

Correlation Matrix

A correlation matrix shows the correlation coefficients between multiple variables. It's particularly useful in multivariate analysis and feature selection for machine learning models.

Code:

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generate multivariate data
data = np.random.randn(100, 4)
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])

# Calculate correlation matrix
corr_matrix = df.corr()

# Plot heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
```

Slide 6: Introduction to Regression

Introduction to Regression

Regression analysis is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It's widely used in predictive modeling and machine learning.

Code:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = 2*X + 1 + np.random.randn(5, 1)*0.5

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Plot data and regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

Slide 7: Simple Linear Regression

Simple Linear Regression

Simple linear regression models the relationship between two variables using a linear equation. It's the simplest form of regression and serves as a foundation for more complex regression techniques.

Code:

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = 2*X + 1 + np.random.randn(5, 1)*0.5

# Fit model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Coefficient: {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Mean squared error: {mse:.2f}")
print(f"R-squared score: {r2:.2f}")
```

Slide 8: Multiple Linear Regression

Multiple Linear Regression

Multiple linear regression extends simple linear regression to include multiple independent variables. It's useful when trying to predict a dependent variable based on multiple factors.

Code:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate data
X = np.random.rand(100, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100)*0.1

# Fit model
model = LinearRegression()
model.fit(X, y)

# Print coefficients
for i, coef in enumerate(model.coef_):
    print(f"Coefficient for X{i+1}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
```

Slide 9: Polynomial Regression

Polynomial Regression

Polynomial regression is used when the relationship between variables is non-linear. It fits a polynomial equation to the data, allowing for more complex relationships to be modeled.

Code:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Generate non-linear data
X = np.linspace(0, 5, 100).reshape(-1, 1)
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1) * 0.5

# Create and fit the polynomial regression model
model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(X, y)

# Plot the results
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title("Polynomial Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

Slide 10: Logistic Regression

Logistic Regression

Logistic regression is used for binary classification problems. Despite its name, it's a classification algorithm, not a regression algorithm. It predicts the probability of an instance belonging to a particular class.

Code:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate binary classification data
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

Slide 11: Introduction to Curve Fitting

Introduction to Curve Fitting

Curve fitting is the process of constructing a curve or mathematical function that best fits a set of data points. It's used in various fields, including machine learning, for modeling complex relationships.

Code:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the function to fit
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Generate noisy data
x = np.linspace(0, 4, 50)
y = func(x, 2.5, 1.3, 0.5) + 0.2 * np.random.normal(size=len(x))

# Fit the function
popt, _ = curve_fit(func, x, y)

# Plot the results
plt.scatter(x, y, label='data')
plt.plot(x, func(x, *popt), 'r-', label='fit')
plt.legend()
plt.show()

print(f"Optimal parameters: a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f}")
```

Slide 12: Non-linear Least Squares Fitting

Non-linear Least Squares Fitting

Non-linear least squares fitting is a form of curve fitting where the function is not required to be linear in the parameters. It's used when the relationship between variables is known to be non-linear.

Code:

```python
import numpy as np
from scipy.optimize import least_squares

# Define the model function
def model(x, params):
    a, b, c = params
    return a * np.exp(-b * x) + c

# Define the residual function
def residual(params, x, y):
    return model(x, params) - y

# Generate synthetic data
x = np.linspace(0, 10, 100)
true_params = [2.5, 0.5, 1.0]
y_true = model(x, true_params)
y = y_true + 0.1 * np.random.randn(len(x))

# Perform the fit
initial_guess = [1.0, 1.0, 0.0]
result = least_squares(residual, initial_guess, args=(x, y))

print("Fitted parameters:", result.x)
```

Slide 13: Real-life Example: Housing Price Prediction

Real-life Example: Housing Price Prediction

Let's use multiple linear regression to predict housing prices based on various features such as size, number of bedrooms, and location.

Code:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data (assuming we have a CSV file with housing data)
data = pd.read_csv('housing_data.csv')

# Prepare the features and target
X = data[['size', 'bedrooms', 'location']]
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean squared error: {mse:.2f}")
print(f"R-squared score: {r2:.2f}")

# Example prediction
new_house = [[2000, 3, 1]]  # size: 2000 sq ft, 3 bedrooms, location code: 1
predicted_price = model.predict(new_house)
print(f"Predicted price: ${predicted_price[0]:,.2f}")
```

Slide 14: Real-life Example: Customer Churn Prediction

Real-life Example: Customer Churn Prediction

Let's use logistic regression to predict customer churn based on features such as usage, customer service calls, and contract length.

Code:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the data (assuming we have a CSV file with customer data)
data = pd.read_csv('customer_data.csv')

# Prepare the features and target
X = data[['usage', 'customer_service_calls', 'contract_length']]
y = data['churned']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example prediction
new_customer = [[100, 2, 12]]  # usage: 100, customer service calls: 2, contract length: 12 months
churn_probability = model.predict_proba(new_customer)[0][1]
print(f"Churn probability: {churn_probability:.2f}")
```

Slide 15: Additional Resources

Additional Resources

For more in-depth study on correlation, regression, and curve fitting in machine learning, consider exploring these resources:

1. "An Introduction to Statistical Learning" by Gareth James et al. (Available on ArXiv: [https://arxiv.org/abs/1501.07274](https://arxiv.org/abs/1501.07274))
2. "The Elements of Statistical Learning" by Trevor Hastie et al. (Available on ArXiv: [https://arxiv.org/abs/2001.00323](https://arxiv.org/abs/2001.00323))
3. Scikit-learn Documentation: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
4. SciPy Documentation: [https://docs.scipy.org/doc/scipy/reference/](https://docs.scipy.org/doc/scipy/reference/)

These resources provide comprehensive coverage of the topics discussed in this presentation and can help deepen your understanding of these fundamental machine learning concepts.

