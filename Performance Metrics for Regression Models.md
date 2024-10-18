## Performance Metrics for Regression Models

Slide 1: Introduction to Performance Metrics in Regression

Performance metrics in regression are essential tools for evaluating the accuracy and effectiveness of our predictive models. These metrics help us understand how well our model's predictions align with the actual observed values. In this presentation, we'll explore various performance metrics commonly used in regression analysis, their significance, and how to implement them using Python.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Example data
y_true = np.array([3, 5, 2, 7, 9])
y_pred = np.array([2.8, 4.5, 2.2, 7.1, 8.7])

# Plotting actual vs predicted values
plt.scatter(y_true, y_pred)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
```

Slide 2: Mean Absolute Error (MAE)

Mean Absolute Error (MAE) is a straightforward metric that measures the average magnitude of errors in a set of predictions, without considering their direction. It's calculated by taking the average of the absolute differences between the predicted values and the actual values. MAE is particularly useful when we want to treat all errors equally, regardless of their magnitude.

```python
mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Absolute Error: {mae:.4f}")

# Manual calculation of MAE
mae_manual = np.mean(np.abs(y_true - y_pred))
print(f"Manually calculated MAE: {mae_manual:.4f}")

# Visualize MAE
plt.bar(range(len(y_true)), np.abs(y_true - y_pred))
plt.axhline(y=mae, color='r', linestyle='--', label=f'MAE: {mae:.4f}')
plt.xlabel('Sample')
plt.ylabel('Absolute Error')
plt.title('Mean Absolute Error Visualization')
plt.legend()
plt.show()
```

Slide 3: Mean Squared Error (MSE)

Mean Squared Error (MSE) is another commonly used metric in regression analysis. It calculates the average of the squared differences between predicted and actual values. MSE gives more weight to larger errors due to the squaring operation, making it particularly sensitive to outliers. This property can be beneficial when larger errors are disproportionately undesirable in your specific application.

```python
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Manual calculation of MSE
mse_manual = np.mean((y_true - y_pred)**2)
print(f"Manually calculated MSE: {mse_manual:.4f}")

# Visualize MSE
plt.bar(range(len(y_true)), (y_true - y_pred)**2)
plt.axhline(y=mse, color='r', linestyle='--', label=f'MSE: {mse:.4f}')
plt.xlabel('Sample')
plt.ylabel('Squared Error')
plt.title('Mean Squared Error Visualization')
plt.legend()
plt.show()
```

Slide 4: Root Mean Squared Error (RMSE)

Root Mean Squared Error (RMSE) is the square root of the Mean Squared Error. It's widely used because it provides an error metric in the same units as the target variable, making it more interpretable than MSE. RMSE represents the standard deviation of the residuals (prediction errors) and gives a measure of how spread out these residuals are.

```python
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"Root Mean Squared Error: {rmse:.4f}")

# Manual calculation of RMSE
rmse_manual = np.sqrt(np.mean((y_true - y_pred)**2))
print(f"Manually calculated RMSE: {rmse_manual:.4f}")

# Visualize RMSE
plt.bar(range(len(y_true)), np.abs(y_true - y_pred))
plt.axhline(y=rmse, color='r', linestyle='--', label=f'RMSE: {rmse:.4f}')
plt.xlabel('Sample')
plt.ylabel('Absolute Error')
plt.title('Root Mean Squared Error Visualization')
plt.legend()
plt.show()
```

Slide 5: Mean Absolute Percentage Error (MAPE)

Mean Absolute Percentage Error (MAPE) measures the average of the absolute percentage differences between predicted and actual values. It's useful for understanding the relative magnitude of errors, especially when dealing with data on different scales. However, MAPE can be problematic when dealing with values close to or equal to zero.

```python
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate MAPE
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

# Visualize MAPE
plt.bar(range(len(y_true)), np.abs((y_true - y_pred) / y_true) * 100)
plt.axhline(y=mape, color='r', linestyle='--', label=f'MAPE: {mape:.2f}%')
plt.xlabel('Sample')
plt.ylabel('Absolute Percentage Error')
plt.title('Mean Absolute Percentage Error Visualization')
plt.legend()
plt.show()
```

Slide 6: R-squared (Coefficient of Determination)

R-squared, also known as the coefficient of determination, measures the proportion of variance in the dependent variable that is predictable from the independent variable(s). It ranges from 0 to 1, where 1 indicates that the model explains all the variability of the response data around its mean, and 0 indicates that the model explains none of the variability.

```python
r2 = r2_score(y_true, y_pred)
print(f"R-squared: {r2:.4f}")

# Manual calculation of R-squared
ss_total = np.sum((y_true - np.mean(y_true))**2)
ss_residual = np.sum((y_true - y_pred)**2)
r2_manual = 1 - (ss_residual / ss_total)
print(f"Manually calculated R-squared: {r2_manual:.4f}")

# Visualize R-squared
plt.scatter(y_true, y_pred)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
plt.text(0.1, 0.9, f'R-squared: {r2:.4f}', transform=plt.gca().transAxes)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('R-squared Visualization')
plt.show()
```

Slide 7: Adjusted R-squared

Adjusted R-squared is a modified version of R-squared that adjusts for the number of predictors in the model. It increases only if the new term improves the model more than would be expected by chance. This metric is particularly useful when comparing models with different numbers of predictors.

```python
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Assuming we have 5 samples and 2 predictors
n = 5  # number of samples
p = 2  # number of predictors

adj_r2 = adjusted_r2(r2, n, p)
print(f"Adjusted R-squared: {adj_r2:.4f}")

# Visualize Adjusted R-squared vs R-squared
predictors = range(1, 6)
r2_values = [r2] * 5
adj_r2_values = [adjusted_r2(r2, n, i) for i in predictors]

plt.plot(predictors, r2_values, label='R-squared')
plt.plot(predictors, adj_r2_values, label='Adjusted R-squared')
plt.xlabel('Number of Predictors')
plt.ylabel('R-squared / Adjusted R-squared')
plt.title('R-squared vs Adjusted R-squared')
plt.legend()
plt.show()
```

Slide 8: Real-Life Example: Predicting Energy Consumption

Let's consider a real-world scenario where we're predicting energy consumption for a building based on factors such as temperature, humidity, and time of day. We'll use a simple linear regression model and evaluate its performance using the metrics we've discussed.

```python
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 3)  # 3 features: temperature, humidity, time of day
y = 2 * X[:, 0] + 3 * X[:, 1] + 1.5 * X[:, 2] + np.random.randn(100) * 0.1

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")

# Visualize actual vs predicted values
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Energy Consumption')
plt.ylabel('Predicted Energy Consumption')
plt.title('Actual vs Predicted Energy Consumption')
plt.show()
```

Slide 9: Interpreting the Results

In our energy consumption prediction example, we can interpret the results as follows:

1. MAE tells us the average absolute difference between predicted and actual energy consumption.
2. RMSE gives us the standard deviation of prediction errors, in the same units as energy consumption.
3. R-squared indicates how much of the variance in energy consumption our model explains based on the input features.

These metrics help us understand the model's performance and can guide decisions on whether to use this model or explore more complex alternatives.

```python
percentage_errors = np.abs((y_test - y_pred) / y_test) * 100

plt.hist(percentage_errors, bins=20)
plt.xlabel('Percentage Error')
plt.ylabel('Frequency')
plt.title('Distribution of Percentage Errors')
plt.show()

print(f"Average Percentage Error: {np.mean(percentage_errors):.2f}%")
print(f"Median Percentage Error: {np.median(percentage_errors):.2f}%")
```

Slide 10: Choosing the Right Metric

The choice of performance metric depends on your specific problem and goals. Here are some guidelines:

1. Use MAE when you want to treat all errors equally.
2. Use MSE or RMSE when larger errors are more problematic.
3. Use MAPE when you want to measure relative errors.
4. Use R-squared to understand how well your model explains the variance in the data.

Consider using multiple metrics to get a comprehensive view of your model's performance.

```python
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R-squared': r2, 'MAPE': mape}

# Calculate metrics for our example
metrics = calculate_metrics(y_test, y_pred)

# Visualize metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics.keys(), metrics.values())
plt.title('Comparison of Different Metrics')
plt.ylabel('Value')
plt.show()
```

Slide 11: Real-Life Example: Predicting Crop Yield

Let's consider another real-world scenario where we're predicting crop yield based on factors such as rainfall, temperature, and soil quality. We'll use a polynomial regression model and evaluate its performance using the metrics we've discussed.

```python
from sklearn.pipeline import make_pipeline

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 3)  # 3 features: rainfall, temperature, soil quality
y = 2 * X[:, 0]**2 + 3 * X[:, 1] + 1.5 * X[:, 2]**3 + np.random.randn(100) * 0.1

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = make_pipeline(PolynomialFeatures(3), LinearRegression())
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate and print metrics
metrics = calculate_metrics(y_test, y_pred)
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Visualize actual vs predicted values
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Crop Yield')
plt.ylabel('Predicted Crop Yield')
plt.title('Actual vs Predicted Crop Yield')
plt.show()
```

Slide 12: Residual Analysis

Residual analysis is a crucial step in evaluating regression models. Residuals are the differences between observed and predicted values. Analyzing residuals can help identify patterns or issues in the model's predictions.

```python

plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Q-Q plot for normality check
from scipy import stats

fig, ax = plt.subplots()
_, (__, ___, r) = stats.probplot(residuals, plot=ax, fit=True)
ax.set_title("Q-Q plot")
plt.show()

print(f"Shapiro-Wilk test p-value: {stats.shapiro(residuals)[1]:.4f}")
```

Slide 13: Cross-Validation for Robust Evaluation

Cross-validation is a technique used to assess how the results of a statistical analysis will generalize to an independent data set. It's particularly useful when you have a limited amount of data and want to make the most of it for both training and validation.

```python

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Convert MSE to RMSE
rmse_scores = np.sqrt(-cv_scores)

print(f"Cross-validation RMSE scores: {rmse_scores}")
print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
print(f"Standard deviation of RMSE: {np.std(rmse_scores):.4f}")

# Visualize cross-validation results
plt.boxplot(rmse_scores)
plt.title('Cross-validation RMSE Scores')
plt.ylabel('RMSE')
plt.show()
```

Slide 14: Feature Importance Analysis

Understanding which features contribute most to your model's predictions can provide valuable insights. For linear models, we can examine the coefficients to determine feature importance.

```python
feature_names = ['Rainfall', 'Temperature', 'Soil Quality']
coefficients = model.named_steps['linearregression'].coef_

# Sort features by absolute coefficient value
feature_importance = sorted(zip(feature_names, np.abs(coefficients)), key=lambda x: x[1], reverse=True)

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.bar([x[0] for x in feature_importance], [x[1] for x in feature_importance])
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

for feature, importance in feature_importance:
    print(f"{feature}: {importance:.4f}")
```

Slide 15: Additional Resources

For those interested in diving deeper into regression analysis and performance metrics, here are some valuable resources:

1. "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani (Available at: [https://arxiv.org/abs/2103.05254](https://arxiv.org/abs/2103.05254))
2. "Evaluating Machine Learning Models" by Alice Zheng (O'Reilly)
3. Scikit-learn documentation on model evaluation: [https://scikit-learn.org/stable/modules/model\_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)

These resources provide in-depth explanations of various regression techniques, performance metrics, and best practices for model evaluation. They can help you further enhance your understanding and application of regression analysis in real-world scenarios.


