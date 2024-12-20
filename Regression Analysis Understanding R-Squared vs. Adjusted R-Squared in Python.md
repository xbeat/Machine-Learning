## Regression Analysis: Understanding R-Squared vs. Adjusted R-Squared in Python
Slide 1: Introduction to Regression Analysis

Regression analysis is a powerful statistical method used to model relationships between variables. It's widely applied in various fields, from economics to machine learning. In this presentation, we'll explore two key metrics for evaluating regression models: R-squared and Adjusted R-squared. We'll use Python to demonstrate these concepts.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1) * 0.1

# Plot the data
plt.scatter(X, y)
plt.title("Sample Data for Regression Analysis")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

Slide 2: Simple Linear Regression

Simple linear regression models the relationship between two variables using a straight line. It's the foundation for understanding more complex regression techniques. Let's create a simple linear regression model using scikit-learn.

```python
# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the results
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.title("Simple Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Coefficient: {model.coef_[0][0]:.2f}")
```

Slide 3: Understanding R-squared

R-squared, also known as the coefficient of determination, measures the proportion of variance in the dependent variable that is predictable from the independent variable(s). It ranges from 0 to 1, where 1 indicates a perfect fit. Let's calculate R-squared for our model.

```python
# Calculate R-squared
r2 = r2_score(y, y_pred)

print(f"R-squared: {r2:.4f}")

# Visualize R-squared
plt.scatter(X, y)
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.fill_between(X.flatten(), y_pred.flatten(), y.flatten(), alpha=0.3)
plt.title(f"Visualizing R-squared: {r2:.4f}")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

Slide 4: Interpreting R-squared

R-squared interpretation depends on the context of the problem. In our example, an R-squared of 0.9891 means that approximately 98.91% of the variance in y can be explained by X. While this seems high, it's important to remember that R-squared alone doesn't guarantee a good model fit or predictive power.

```python
# Generate predictions for new data
X_new = np.array([[0.1], [0.5], [0.9]])
y_new_pred = model.predict(X_new)

print("Predictions for new data:")
for x, y in zip(X_new, y_new_pred):
    print(f"X = {x[0]:.1f}, Predicted y = {y[0]:.2f}")

# Plot new predictions
plt.scatter(X, y, label='Original Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.scatter(X_new, y_new_pred, color='green', s=100, label='New Predictions')
plt.title("Predictions Using R-squared Model")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

Slide 5: Limitations of R-squared

R-squared has limitations, particularly when dealing with multiple predictors. It always increases or remains the same when adding more variables, even if they don't improve the model's predictive power. This can lead to overfitting, where the model performs well on training data but poorly on new, unseen data.

```python
# Demonstrate R-squared limitation
X_multi = np.column_stack((X, np.random.rand(100, 1)))  # Add a random feature
model_multi = LinearRegression().fit(X_multi, y)
y_pred_multi = model_multi.predict(X_multi)

r2_multi = r2_score(y, y_pred_multi)

print(f"R-squared (original): {r2:.4f}")
print(f"R-squared (with random feature): {r2_multi:.4f}")

# Visualize the comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.title(f"Original Model\nR-squared: {r2:.4f}")
plt.subplot(1, 2, 2)
plt.scatter(X, y)
plt.plot(X, y_pred_multi, color='red')
plt.title(f"Model with Random Feature\nR-squared: {r2_multi:.4f}")
plt.tight_layout()
plt.show()
```

Slide 6: Introduction to Adjusted R-squared

Adjusted R-squared addresses the limitation of R-squared by penalizing the addition of variables that don't improve the model's explanatory power. It adjusts for the number of predictors in the model relative to the sample size. Let's calculate Adjusted R-squared for our models.

```python
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

n = len(X)  # Sample size
p1 = 1  # Number of predictors in original model
p2 = 2  # Number of predictors in model with random feature

adj_r2 = adjusted_r2(r2, n, p1)
adj_r2_multi = adjusted_r2(r2_multi, n, p2)

print(f"Adjusted R-squared (original): {adj_r2:.4f}")
print(f"Adjusted R-squared (with random feature): {adj_r2_multi:.4f}")
```

Slide 7: Interpreting Adjusted R-squared

Adjusted R-squared provides a more realistic assessment of the model's fit, especially when comparing models with different numbers of predictors. In our example, we see that the Adjusted R-squared for the model with the random feature is lower than the original model, indicating that the additional feature doesn't improve the model's explanatory power.

```python
# Visualize the comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(['R-squared', 'Adjusted R-squared'], [r2, adj_r2])
ax1.set_title("Original Model")
ax1.set_ylim(0, 1)

ax2.bar(['R-squared', 'Adjusted R-squared'], [r2_multi, adj_r2_multi])
ax2.set_title("Model with Random Feature")
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.show()
```

Slide 8: When to Use R-squared vs. Adjusted R-squared

R-squared is suitable for simple linear regression with a single predictor. Adjusted R-squared is preferable when dealing with multiple linear regression or when comparing models with different numbers of predictors. It helps prevent overfitting by penalizing unnecessary complexity.

```python
def compare_models(X, y, X_extended):
    model1 = LinearRegression().fit(X, y)
    model2 = LinearRegression().fit(X_extended, y)
    
    r2_1 = r2_score(y, model1.predict(X))
    r2_2 = r2_score(y, model2.predict(X_extended))
    
    adj_r2_1 = adjusted_r2(r2_1, len(X), X.shape[1])
    adj_r2_2 = adjusted_r2(r2_2, len(X), X_extended.shape[1])
    
    return r2_1, r2_2, adj_r2_1, adj_r2_2

X_extended = np.column_stack((X, np.sin(X), np.cos(X)))
r2_1, r2_2, adj_r2_1, adj_r2_2 = compare_models(X, y, X_extended)

print(f"Model 1 - R-squared: {r2_1:.4f}, Adjusted R-squared: {adj_r2_1:.4f}")
print(f"Model 2 - R-squared: {r2_2:.4f}, Adjusted R-squared: {adj_r2_2:.4f}")
```

Slide 9: Real-life Example: House Price Prediction

Let's apply our knowledge to a real-life scenario: predicting house prices based on various features. We'll create a dataset with multiple predictors and compare R-squared and Adjusted R-squared.

```python
np.random.seed(42)
n_samples = 1000

# Generate features
area = np.random.uniform(1000, 5000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
age = np.random.uniform(0, 50, n_samples)

# Generate target variable (price) with some noise
price = 100000 + 200 * area + 25000 * bedrooms - 1000 * age + np.random.normal(0, 50000, n_samples)

# Create feature matrix and target vector
X = np.column_stack((area, bedrooms, age))
y = price

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression().fit(X_train, y_train)

# Calculate R-squared and Adjusted R-squared
r2 = r2_score(y_test, model.predict(X_test))
adj_r2 = adjusted_r2(r2, len(y_test), X_test.shape[1])

print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adj_r2:.4f}")
```

Slide 10: Analyzing the House Price Prediction Model

In our house price prediction model, we see that both R-squared and Adjusted R-squared are relatively high, indicating a good fit. The small difference between them suggests that all features contribute meaningfully to the model's predictive power.

```python
# Feature importance
feature_names = ['Area', 'Bedrooms', 'Age']
importance = np.abs(model.coef_)
sorted_idx = np.argsort(importance)

plt.barh(range(len(importance)), importance[sorted_idx])
plt.yticks(range(len(importance)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance in House Price Prediction')
plt.show()

# Predictions vs. Actual
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted House Prices')
plt.show()
```

Slide 11: Real-life Example: Stock Market Analysis

Another common application of regression analysis is in stock market prediction. Let's create a simple model to predict stock prices based on historical data and market indicators.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Generate synthetic stock market data
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
n_samples = len(dates)

price = 100 + np.cumsum(np.random.normal(0, 1, n_samples))
volume = np.random.lognormal(10, 1, n_samples)
market_index = 1000 + np.cumsum(np.random.normal(0, 5, n_samples))

df = pd.DataFrame({
    'Date': dates,
    'Price': price,
    'Volume': volume,
    'Market_Index': market_index
})

# Prepare features and target
X = df[['Volume', 'Market_Index']].values
y = df['Price'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression().fit(X_train, y_train)

# Calculate R-squared and Adjusted R-squared
r2 = r2_score(y_test, model.predict(X_test))
adj_r2 = adjusted_r2(r2, len(y_test), X_test.shape[1])

print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adj_r2:.4f}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price'], label='Actual Price')
plt.plot(df['Date'][-len(y_test):], model.predict(X_test), label='Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
```

Slide 12: Comparing R-squared and Adjusted R-squared

Let's compare R-squared and Adjusted R-squared across different models to understand their behavior. We'll create models with varying numbers of features and observe how these metrics change.

```python
def create_model(X, y, n_features):
    X_subset = X[:, :n_features]
    model = LinearRegression().fit(X_subset, y)
    y_pred = model.predict(X_subset)
    r2 = r2_score(y, y_pred)
    adj_r2 = adjusted_r2(r2, len(y), n_features)
    return r2, adj_r2

# Generate data with 10 features
n_samples = 1000
n_features = 10
X = np.random.randn(n_samples, n_features)
y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(n_samples)

r2_scores = []
adj_r2_scores = []

for i in range(1, n_features + 1):
    r2, adj_r2 = create_model(X, y, i)
    r2_scores.append(r2)
    adj_r2_scores.append(adj_r2)

plt.plot(range(1, n_features + 1), r2_scores, label='R-squared')
plt.plot(range(1, n_features + 1), adj_r2_scores, label='Adjusted R-squared')
plt.xlabel('Number of Features')
plt.ylabel('Score')
plt.title('R-squared vs Adjusted R-squared')
plt.legend()
plt.show()
```

Slide 13: Conclusion and Best Practices

R-squared and Adjusted R-squared are valuable tools for evaluating regression models, but they should be used in conjunction with other metrics and domain knowledge. Best practices include:

1. Use Adjusted R-squared when comparing models with different numbers of predictors.
2. Consider other metrics like Mean Squared Error (MSE) or Mean Absolute Error (MAE) for a complete evaluation.
3. Always validate your model on unseen data to assess its generalization ability.
4. Be cautious of overfitting, especially when R-squared is very high.
5. Remember that a high R-squared doesn't necessarily mean a good model or imply causation.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Assuming we have X_test and y_test from previous examples
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adj_r2 = adjusted_r2(r2, len(y_test), X_test.shape[1])

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adj_r2:.4f}")

# Visualize the metrics
metrics = ['MSE', 'MAE', 'R-squared', 'Adj R-squared']
values = [mse, mae, r2, adj_r2]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values)
plt.title('Model Evaluation Metrics')
plt.ylabel('Value')
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into regression analysis and model evaluation, here are some valuable resources:

1. "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani - A comprehensive guide to statistical learning methods.
2. "Regression Analysis by Example" by Chatterjee and Hadi - Offers practical examples and in-depth explanations of regression techniques.
3. ArXiv paper: "A Survey of R-squared Measures for Regression Models" by Sriboonchitta et al. (2021) ArXiv URL: [https://arxiv.org/abs/2101.09426](https://arxiv.org/abs/2101.09426)
4. Scikit-learn Documentation - Provides detailed information on implementing regression models and evaluating their performance in Python.
5. StatQuest YouTube channel - Offers clear, intuitive explanations of statistical concepts, including regression analysis and model evaluation.

Remember to verify the accuracy and relevance of these resources, as statistical methodologies and best practices may evolve over time.

