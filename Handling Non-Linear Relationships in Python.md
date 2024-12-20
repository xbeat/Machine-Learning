## Handling Non-Linear Relationships in Python

Slide 1: Understanding Non-Linear Relationships in Data

Non-linear relationships are common in real-world data, where the relationship between variables doesn't follow a straight line. This slideshow will explore techniques to handle and analyze non-linear relationships using Python, providing practical examples and code snippets.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate non-linear data
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)

# Plot the data
plt.scatter(x, y)
plt.title("Example of Non-Linear Relationship")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Slide 2: Visualizing Non-Linear Relationships

Visualization is crucial for understanding non-linear relationships. We'll use scatter plots and line plots to identify patterns in the data.

```python
import seaborn as sns

# Create a more complex non-linear relationship
x = np.linspace(0, 10, 200)
y = 2 * np.sin(x) + 0.5 * x**2 + np.random.normal(0, 0.5, 200)

# Create scatter plot with trend line
sns.regplot(x=x, y=y, scatter_kws={"alpha": 0.5}, order=2)
plt.title("Non-Linear Relationship with Quadratic Trend")
plt.show()
```

Slide 3: Polynomial Regression

Polynomial regression is a technique to model non-linear relationships by fitting a polynomial equation to the data.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Create and fit the polynomial regression model
degree = 3
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(x.reshape(-1, 1), y)

# Generate predictions
x_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = model.predict(x_test)

# Plot the results
plt.scatter(x, y, alpha=0.5)
plt.plot(x_test, y_pred, color='r', label=f'Polynomial (degree={degree})')
plt.legend()
plt.title("Polynomial Regression")
plt.show()
```

Slide 4: Spline Regression

Spline regression uses piecewise polynomial functions to model non-linear relationships, offering more flexibility than simple polynomial regression.

```python
from scipy.interpolate import UnivariateSpline

# Fit a spline
spline = UnivariateSpline(x, y, k=3, s=1)

# Generate predictions
x_test = np.linspace(0, 10, 100)
y_pred = spline(x_test)

# Plot the results
plt.scatter(x, y, alpha=0.5)
plt.plot(x_test, y_pred, color='g', label='Spline Regression')
plt.legend()
plt.title("Spline Regression")
plt.show()
```

Slide 5: Logarithmic Transformation

Logarithmic transformation can help linearize certain types of non-linear relationships, especially when dealing with exponential growth or decay.

```python
# Generate data with exponential relationship
x = np.linspace(1, 10, 100)
y = 2 ** x + np.random.normal(0, 0.5 * 2**x, 100)

# Apply logarithmic transformation
y_log = np.log2(y)

# Plot original and transformed data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(x, y)
ax1.set_title("Original Data")
ax2.scatter(x, y_log)
ax2.set_title("Log-Transformed Data")
plt.tight_layout()
plt.show()
```

Slide 6: Gaussian Process Regression

Gaussian Process Regression is a powerful non-parametric method for modeling non-linear relationships, particularly useful when the underlying function is complex or unknown.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Define the kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

# Create and fit the GP model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(x.reshape(-1, 1), y)

# Make predictions
x_pred = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred, sigma = gp.predict(x_pred, return_std=True)

# Plot the results
plt.scatter(x, y, label='Observations')
plt.plot(x_pred, y_pred, label='Mean prediction')
plt.fill_between(x_pred.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma,
                 alpha=0.2, label='95% confidence interval')
plt.legend()
plt.title("Gaussian Process Regression")
plt.show()
```

Slide 7: Decision Trees for Non-Linear Relationships

Decision trees can capture non-linear relationships by recursively partitioning the feature space.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

# Create and fit the decision tree model
tree = DecisionTreeRegressor(max_depth=5)
tree.fit(x.reshape(-1, 1), y)

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(tree, filled=True, feature_names=['X'])
plt.title("Decision Tree for Non-Linear Relationship")
plt.show()

# Make predictions
y_pred = tree.predict(x.reshape(-1, 1))

# Plot the results
plt.scatter(x, y, label='Data')
plt.plot(x, y_pred, color='r', label='Decision Tree Prediction')
plt.legend()
plt.title("Decision Tree Regression")
plt.show()
```

Slide 8: Random Forest for Complex Non-Linear Relationships

Random Forests, an ensemble of decision trees, can model highly complex non-linear relationships by averaging predictions from multiple trees.

```python
from sklearn.ensemble import RandomForestRegressor

# Create and fit the random forest model
rf = RandomForestRegressor(n_estimators=100, max_depth=10)
rf.fit(x.reshape(-1, 1), y)

# Make predictions
y_pred = rf.predict(x.reshape(-1, 1))

# Plot the results
plt.scatter(x, y, label='Data')
plt.plot(x, y_pred, color='r', label='Random Forest Prediction')
plt.legend()
plt.title("Random Forest Regression")
plt.show()
```

Slide 9: Support Vector Regression with Non-Linear Kernel

Support Vector Regression with non-linear kernels can effectively model complex non-linear relationships by implicitly mapping the input space to a higher-dimensional feature space.

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x.reshape(-1, 1))

# Create and fit the SVR model with RBF kernel
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_scaled, y)

# Make predictions
X_pred = np.linspace(min(x), max(x), 100).reshape(-1, 1)
X_pred_scaled = scaler.transform(X_pred)
y_pred = svr.predict(X_pred_scaled)

# Plot the results
plt.scatter(x, y, label='Data')
plt.plot(X_pred, y_pred, color='r', label='SVR Prediction')
plt.legend()
plt.title("Support Vector Regression with RBF Kernel")
plt.show()
```

Slide 10: Artificial Neural Networks for Non-Linear Relationships

Artificial Neural Networks, particularly deep learning models, can capture complex non-linear relationships through multiple layers of non-linear transformations.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create and compile the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Fit the model
history = model.fit(x, y, epochs=200, validation_split=0.2, verbose=0)

# Make predictions
y_pred = model.predict(x)

# Plot the results
plt.scatter(x, y, label='Data')
plt.plot(x, y_pred, color='r', label='Neural Network Prediction')
plt.legend()
plt.title("Neural Network Regression")
plt.show()

# Plot the learning curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title("Learning Curves")
plt.show()
```

Slide 11: Real-Life Example: Temperature vs. Altitude

Let's explore the non-linear relationship between temperature and altitude using polynomial regression.

```python
# Generate sample data (altitude in meters, temperature in Celsius)
altitude = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
temperature = np.array([25, 22, 18, 15, 11, 8, 4, 1, -2, -5, -8])

# Create polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(altitude.reshape(-1, 1))

# Fit polynomial regression
model = LinearRegression()
model.fit(X_poly, temperature)

# Generate points for smooth curve
X_smooth = np.linspace(altitude.min(), altitude.max(), 100).reshape(-1, 1)
X_smooth_poly = poly_features.transform(X_smooth)
temperature_poly = model.predict(X_smooth_poly)

# Plot the results
plt.scatter(altitude, temperature, color='blue', label='Actual data')
plt.plot(X_smooth, temperature_poly, color='red', label='Polynomial regression')
plt.xlabel('Altitude (meters)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature vs. Altitude')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 12: Real-Life Example: Plant Growth Over Time

Let's model the non-linear growth of a plant over time using a logistic growth curve.

```python
from scipy.optimize import curve_fit

# Generate sample data
time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
height = np.array([2, 3, 4, 7, 12, 18, 25, 30, 33, 35, 36])

# Define logistic function
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# Fit the logistic function to the data
popt, _ = curve_fit(logistic, time, height)

# Generate smooth curve for plotting
time_smooth = np.linspace(0, 10, 100)
height_smooth = logistic(time_smooth, *popt)

# Plot the results
plt.scatter(time, height, label='Actual data')
plt.plot(time_smooth, height_smooth, 'r-', label='Logistic growth model')
plt.xlabel('Time (weeks)')
plt.ylabel('Plant height (cm)')
plt.title('Plant Growth Over Time')
plt.legend()
plt.grid(True)
plt.show()

print(f"Estimated maximum height: {popt[0]:.2f} cm")
print(f"Growth rate: {popt[1]:.2f}")
print(f"Time of maximum growth: {popt[2]:.2f} weeks")
```

Slide 13: Handling Non-Linear Relationships with Feature Engineering

Feature engineering can help capture non-linear relationships by creating new features that express non-linear transformations of the original features.

```python
# Generate sample data
x = np.linspace(0, 10, 100)
y = 3 * x**2 + 2 * np.sin(x) + np.random.normal(0, 5, 100)

# Create polynomial and trigonometric features
X = pd.DataFrame({'x': x})
X['x_squared'] = X['x']**2
X['sin_x'] = np.sin(X['x'])

# Fit linear regression with engineered features
model = LinearRegression()
model.fit(X, y)

# Generate predictions
X_test = pd.DataFrame({'x': np.linspace(0, 10, 200)})
X_test['x_squared'] = X_test['x']**2
X_test['sin_x'] = np.sin(X_test['x'])
y_pred = model.predict(X_test)

# Plot the results
plt.scatter(x, y, label='Data')
plt.plot(X_test['x'], y_pred, color='r', label='Prediction')
plt.legend()
plt.title("Linear Regression with Engineered Features")
plt.show()

print("Model coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.2f}")
```

Slide 14: Cross-Validation for Non-Linear Models

Cross-validation is crucial for assessing the performance of non-linear models and avoiding overfitting. Let's use k-fold cross-validation to evaluate a polynomial regression model.

```python
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate sample data
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Create models with different polynomial degrees
degrees = [1, 2, 3, 4, 5]
scores = []

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    scores.append(-score.mean())

# Plot the results
plt.plot(degrees, scores, marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Cross-Validation Scores for Polynomial Regression')
plt.show()

best_degree = degrees[np.argmin(scores)]
print(f"Best polynomial degree: {best_degree}")
```

Slide 15: Additional Resources

For further exploration of handling non-linear relationships in data using Python, consider the following resources:

1. "Gaussian Processes for Machine Learning" by Carl Edward Rasmussen and Christopher K. I. Williams (2006) - Available on ArXiv: [https://arxiv.org/abs/1505.02965](https://arxiv.org/abs/1505.02965)
2. "A Tutorial on Support Vector Regression" by Alex J. Smola and Bernhard Schölkopf (2004) - Available on ArXiv: [https://arxiv.org/abs/physics/0407053](https://arxiv.org/abs/physics/0407053)
3. "Random Forests" by Leo Breiman (2001) - Available on ArXiv: [https://arxiv.org/abs/1201.0490](https://arxiv.org/abs/1201.0490)
4. Scikit-learn documentation on non-linear models: [https://scikit-learn.org/stable/modules/nonlinear\_transformation.html](https://scikit-learn.org/stable/modules/nonlinear_transformation.html)

These resources provide in-depth explanations and mathematical foundations for the techniques covered in this slideshow.

