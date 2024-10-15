## Simple Regression Model to Predict Height from Weight
Slide 1: Simple Regression Model to Predict Height from Weight

A simple regression model is a statistical method used to analyze the relationship between two variables. In this case, we'll explore how to predict a person's height based on their weight using Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data (weight in kg, height in cm)
weight = np.array([70, 75, 80, 85, 90, 95, 100]).reshape(-1, 1)
height = np.array([170, 172, 175, 178, 180, 182, 185])

# Create and fit the model
model = LinearRegression()
model.fit(weight, height)

# Plot the data and regression line
plt.scatter(weight, height, color='blue', label='Data points')
plt.plot(weight, model.predict(weight), color='red', label='Regression line')
plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.title('Height vs Weight Regression')
plt.legend()
plt.show()
```

Slide 2: Data Collection and Preparation

To build our regression model, we first need to collect and prepare our data. In this example, we'll use a small dataset of weight and height measurements.

```python
import numpy as np

# Sample data (weight in kg, height in cm)
weight = np.array([70, 75, 80, 85, 90, 95, 100])
height = np.array([170, 172, 175, 178, 180, 182, 185])

# Reshape weight array for sklearn
weight = weight.reshape(-1, 1)

print("Weight data shape:", weight.shape)
print("Height data shape:", height.shape)
```

Slide 3: Visualizing the Data

Before building our model, it's crucial to visualize the data to understand the relationship between weight and height.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(weight, height, color='blue')
plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.title('Height vs Weight Scatter Plot')
plt.grid(True)
plt.show()
```

Slide 4: Creating the Linear Regression Model

We'll use scikit-learn's LinearRegression class to create our simple regression model.

```python
from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Fit the model to our data
model.fit(weight, height)

print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)
```

Slide 5: Making Predictions

Now that we have trained our model, we can use it to make predictions.

```python
# Predict height for a weight of 82 kg
new_weight = np.array([[82]])
predicted_height = model.predict(new_weight)

print(f"Predicted height for a weight of 82 kg: {predicted_height[0]:.2f} cm")

# Predict heights for a range of weights
weight_range = np.arange(60, 110, 5).reshape(-1, 1)
predicted_heights = model.predict(weight_range)

for w, h in zip(weight_range, predicted_heights):
    print(f"Weight: {w[0]} kg, Predicted Height: {h:.2f} cm")
```

Slide 6: Evaluating the Model

To assess the performance of our model, we'll calculate the R-squared score and mean squared error.

```python
from sklearn.metrics import r2_score, mean_squared_error

# Calculate R-squared score
r2 = r2_score(height, model.predict(weight))

# Calculate mean squared error
mse = mean_squared_error(height, model.predict(weight))

print(f"R-squared score: {r2:.4f}")
print(f"Mean squared error: {mse:.4f}")
```

Slide 7: Visualizing the Regression Line

Let's visualize our data points along with the regression line to see how well our model fits the data.

```python
plt.figure(figsize=(10, 6))
plt.scatter(weight, height, color='blue', label='Data points')
plt.plot(weight, model.predict(weight), color='red', label='Regression line')
plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.title('Height vs Weight Regression')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 8: Interpreting the Results

The slope (coefficient) of our regression line represents the change in height for each unit increase in weight. The intercept represents the predicted height when the weight is zero (which may not be meaningful in this context).

```python
slope = model.coef_[0]
intercept = model.intercept_

print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")

print(f"For every 1 kg increase in weight, height increases by {slope:.2f} cm")
print(f"The predicted height for a weight of 0 kg (intercept) is {intercept:.2f} cm")
```

Slide 9: Limitations of the Model

It's important to understand the limitations of our simple regression model:

1. Assumes a linear relationship between weight and height
2. Doesn't account for other factors that may influence height
3. May not be accurate for extreme values or outside the range of our data

```python
# Demonstrating potential inaccuracy for extreme values
extreme_low = np.array([[30]])
extreme_high = np.array([[200]])

print(f"Predicted height for 30 kg: {model.predict(extreme_low)[0]:.2f} cm")
print(f"Predicted height for 200 kg: {model.predict(extreme_high)[0]:.2f} cm")
```

Slide 10: Real-Life Example 1 - Predicting Plant Growth

Let's apply our simple regression model to predict plant height based on the amount of water given.

```python
# Sample data (water in ml, plant height in cm)
water = np.array([10, 20, 30, 40, 50, 60, 70]).reshape(-1, 1)
plant_height = np.array([5, 7, 10, 12, 15, 17, 20])

# Create and fit the model
plant_model = LinearRegression()
plant_model.fit(water, plant_height)

# Predict height for 45 ml of water
new_water = np.array([[45]])
predicted_plant_height = plant_model.predict(new_water)

print(f"Predicted plant height for 45 ml of water: {predicted_plant_height[0]:.2f} cm")

# Visualize the data and regression line
plt.scatter(water, plant_height, color='green', label='Data points')
plt.plot(water, plant_model.predict(water), color='blue', label='Regression line')
plt.xlabel('Water (ml)')
plt.ylabel('Plant Height (cm)')
plt.title('Plant Growth: Height vs Water')
plt.legend()
plt.show()
```

Slide 11: Real-Life Example 2 - Predicting Fuel Efficiency

Now, let's use our simple regression model to predict a car's fuel efficiency (miles per gallon) based on its weight.

```python
# Sample data (car weight in tons, fuel efficiency in mpg)
car_weight = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]).reshape(-1, 1)
fuel_efficiency = np.array([35, 30, 25, 22, 18, 15, 12])

# Create and fit the model
car_model = LinearRegression()
car_model.fit(car_weight, fuel_efficiency)

# Predict fuel efficiency for a 2.7-ton car
new_car_weight = np.array([[2.7]])
predicted_efficiency = car_model.predict(new_car_weight)

print(f"Predicted fuel efficiency for a 2.7-ton car: {predicted_efficiency[0]:.2f} mpg")

# Visualize the data and regression line
plt.scatter(car_weight, fuel_efficiency, color='orange', label='Data points')
plt.plot(car_weight, car_model.predict(car_weight), color='purple', label='Regression line')
plt.xlabel('Car Weight (tons)')
plt.ylabel('Fuel Efficiency (mpg)')
plt.title('Car Fuel Efficiency: MPG vs Weight')
plt.legend()
plt.show()
```

Slide 12: Improving the Model

To enhance our simple regression model, we can consider the following approaches:

1. Collect more data to improve accuracy
2. Use multiple features (multiple regression)
3. Apply non-linear regression techniques

```python
# Example of multiple regression
from sklearn.preprocessing import PolynomialFeatures

# Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
weight_poly = poly.fit_transform(weight)

# Create and fit the model
model_poly = LinearRegression()
model_poly.fit(weight_poly, height)

# Visualize the results
plt.scatter(weight, height, color='blue', label='Data points')
plt.plot(weight, model.predict(weight), color='red', label='Linear regression')
plt.plot(weight, model_poly.predict(weight_poly), color='green', label='Polynomial regression')
plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.title('Height vs Weight: Linear and Polynomial Regression')
plt.legend()
plt.show()
```

Slide 13: Conclusion and Next Steps

Simple regression models provide a foundation for understanding the relationship between variables. To further improve your skills in predictive modeling:

1. Explore multiple regression and polynomial regression
2. Learn about regularization techniques (Lasso, Ridge)
3. Study more advanced machine learning algorithms
4. Practice with larger, real-world datasets

```python
# Example of using scikit-learn's train_test_split for model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(weight, height, test_size=0.2, random_state=42)

model_eval = LinearRegression()
model_eval.fit(X_train, y_train)

y_pred = model_eval.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error on test set: {mae:.2f}")
```

Slide 14: Additional Resources

For further learning on simple regression and related topics:

1. "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani ArXiv URL: [https://arxiv.org/abs/1320.7907](https://arxiv.org/abs/1320.7907)
2. "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman ArXiv URL: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
3. Scikit-learn documentation on Linear Models: [https://scikit-learn.org/stable/modules/linear\_model.html](https://scikit-learn.org/stable/modules/linear_model.html)
4. "Python for Data Analysis" by Wes McKinney - A comprehensive guide to using Python for data analysis, including regression techniques.
5. "Machine Learning: A Probabilistic Perspective" by Kevin P. Murphy - Provides a deeper understanding of machine learning concepts, including regression models.

