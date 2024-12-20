## Understanding R-squared and Adjusted R-squared in Regression Models
Slide 1: R-squared (R²) and Adjusted R-squared

R-squared (R²) and adjusted R-squared are statistical measures used to evaluate the goodness of fit of regression models. They provide insights into how well a model explains the variability in the dependent variable based on the independent variables. While both metrics serve similar purposes, they have distinct characteristics and use cases.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1) * 0.1

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Calculate R-squared
r2 = r2_score(y, model.predict(X))

print(f"R-squared: {r2:.4f}")

# Plot the data and regression line
plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.title(f"Linear Regression (R² = {r2:.4f})")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

Slide 2: Understanding R-squared (R²)

R-squared, also known as the coefficient of determination, measures the proportion of variance in the dependent variable that is predictable from the independent variable(s). It ranges from 0 to 1, where 0 indicates that the model explains none of the variability, and 1 indicates perfect prediction. R-squared is calculated as the ratio of the explained variance to the total variance.

```python
def calculate_r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

# Using the data from the previous slide
y_pred = model.predict(X)
r_squared = calculate_r_squared(y, y_pred)

print(f"Calculated R-squared: {r_squared:.4f}")
print(f"Sklearn R-squared: {r2:.4f}")
```

Slide 3: Interpreting R-squared

R-squared interpretation depends on the context and field of study. In general, higher values indicate better fit, but the acceptable range varies. For example, in social sciences, an R-squared of 0.3 might be considered good, while in physical sciences, values above 0.9 are often expected. It's crucial to consider other factors and not rely solely on R-squared when evaluating model performance.

```python
def interpret_r_squared(r_squared):
    if r_squared < 0.3:
        return "Weak fit"
    elif r_squared < 0.5:
        return "Moderate fit"
    elif r_squared < 0.7:
        return "Good fit"
    else:
        return "Strong fit"

print(f"R-squared: {r_squared:.4f}")
print(f"Interpretation: {interpret_r_squared(r_squared)}")
```

Slide 4: Limitations of R-squared

While R-squared is widely used, it has limitations. It doesn't indicate whether the coefficient estimates and predictions are biased. R-squared always increases when more predictors are added to the model, even if they don't improve the model's predictive power. This can lead to overfitting, especially with small sample sizes or many predictors.

```python
# Demonstrate R-squared increasing with irrelevant predictors
X_expanded = np.column_stack([X, np.random.rand(100, 5)])  # Add 5 random predictors

model_expanded = LinearRegression()
model_expanded.fit(X_expanded, y)

r2_expanded = r2_score(y, model_expanded.predict(X_expanded))

print(f"Original R-squared: {r2:.4f}")
print(f"R-squared with irrelevant predictors: {r2_expanded:.4f}")
```

Slide 5: Introduction to Adjusted R-squared

Adjusted R-squared addresses some limitations of R-squared by penalizing the addition of predictors that don't improve the model significantly. It adjusts the R-squared based on the number of predictors relative to the sample size. Unlike R-squared, adjusted R-squared can decrease when unnecessary predictors are added, helping to identify overfitting.

```python
def adjusted_r_squared(r_squared, n, p):
    return 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

n = len(X)  # Sample size
p = X.shape[1]  # Number of predictors

adj_r2 = adjusted_r_squared(r2, n, p)
adj_r2_expanded = adjusted_r_squared(r2_expanded, n, X_expanded.shape[1])

print(f"Original Adjusted R-squared: {adj_r2:.4f}")
print(f"Adjusted R-squared with irrelevant predictors: {adj_r2_expanded:.4f}")
```

Slide 6: Calculating Adjusted R-squared

Adjusted R-squared is calculated using the formula: 1 - \[(1 - R²) \* (n - 1) / (n - p - 1)\], where n is the sample size and p is the number of predictors. This adjustment allows for a fairer comparison between models with different numbers of predictors.

```python
from sklearn.datasets import make_regression

# Generate dataset with multiple features
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Fit models with different numbers of features
r2_scores = []
adj_r2_scores = []

for i in range(1, X.shape[1] + 1):
    model = LinearRegression()
    model.fit(X[:, :i], y)
    r2 = r2_score(y, model.predict(X[:, :i]))
    adj_r2 = adjusted_r_squared(r2, len(y), i)
    
    r2_scores.append(r2)
    adj_r2_scores.append(adj_r2)

# Plot R-squared and Adjusted R-squared
plt.plot(range(1, 6), r2_scores, label='R-squared')
plt.plot(range(1, 6), adj_r2_scores, label='Adjusted R-squared')
plt.xlabel('Number of features')
plt.ylabel('Score')
plt.legend()
plt.title('R-squared vs Adjusted R-squared')
plt.show()
```

Slide 7: Comparing R-squared and Adjusted R-squared

R-squared and adjusted R-squared often provide different insights. R-squared always increases or remains the same when adding predictors, while adjusted R-squared may decrease if the added predictor doesn't improve the model significantly. This property makes adjusted R-squared useful for feature selection and preventing overfitting.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 1 + 2 * X + 0.5 * X**2 + np.random.randn(100, 1) * 0.1

# Fit models with increasing polynomial degrees
degrees = range(1, 6)
r2_scores = []
adj_r2_scores = []

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    adj_r2 = adjusted_r_squared(r2, len(y), degree + 1)
    
    r2_scores.append(r2)
    adj_r2_scores.append(adj_r2)

# Plot R-squared and Adjusted R-squared
plt.plot(degrees, r2_scores, label='R-squared')
plt.plot(degrees, adj_r2_scores, label='Adjusted R-squared')
plt.xlabel('Polynomial Degree')
plt.ylabel('Score')
plt.legend()
plt.title('R-squared vs Adjusted R-squared for Polynomial Regression')
plt.show()
```

Slide 8: When to Use R-squared vs Adjusted R-squared

R-squared is suitable for simple linear regression or when comparing models with the same number of predictors. Adjusted R-squared is preferable when comparing models with different numbers of predictors or when dealing with multiple regression. It helps in selecting the most parsimonious model that explains the data well without overfitting.

```python
from sklearn.model_selection import train_test_split

# Generate dataset
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models with different numbers of features
results = []

for i in range(1, X.shape[1] + 1):
    model = LinearRegression()
    model.fit(X_train[:, :i], y_train)
    
    train_r2 = r2_score(y_train, model.predict(X_train[:, :i]))
    test_r2 = r2_score(y_test, model.predict(X_test[:, :i]))
    
    train_adj_r2 = adjusted_r_squared(train_r2, len(y_train), i)
    test_adj_r2 = adjusted_r_squared(test_r2, len(y_test), i)
    
    results.append((i, train_r2, test_r2, train_adj_r2, test_adj_r2))

# Print results
for i, train_r2, test_r2, train_adj_r2, test_adj_r2 in results:
    print(f"Features: {i}, Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, "
          f"Train Adj R²: {train_adj_r2:.4f}, Test Adj R²: {test_adj_r2:.4f}")
```

Slide 9: Real-life Example: Predicting House Prices

In this example, we'll use R-squared and adjusted R-squared to evaluate a model predicting house prices based on various features such as square footage, number of bedrooms, and location. This scenario demonstrates how these metrics can be applied in a practical context.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Calculate R-squared and adjusted R-squared
train_r2 = r2_score(y_train, model.predict(X_train_scaled))
test_r2 = r2_score(y_test, model.predict(X_test_scaled))

train_adj_r2 = adjusted_r_squared(train_r2, len(y_train), X_train.shape[1])
test_adj_r2 = adjusted_r_squared(test_r2, len(y_test), X_test.shape[1])

print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
print(f"Train Adjusted R²: {train_adj_r2:.4f}, Test Adjusted R²: {test_adj_r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': housing.feature_names,
    'importance': abs(model.coef_)
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)
```

Slide 10: Real-life Example: Analyzing Factors Affecting Crop Yield

In this example, we'll use R-squared and adjusted R-squared to evaluate a model predicting crop yield based on various environmental factors such as temperature, rainfall, soil quality, and fertilizer use. This scenario showcases how these metrics can be applied in agricultural research.

Slide 11: Real-life Example: Analyzing Factors Affecting Crop Yield

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic crop yield data
np.random.seed(42)
n_samples = 1000

temperature = np.random.normal(25, 5, n_samples)
rainfall = np.random.normal(1000, 200, n_samples)
soil_quality = np.random.uniform(0, 10, n_samples)
fertilizer = np.random.uniform(50, 150, n_samples)

# Create target variable (crop yield) with some noise
crop_yield = (0.5 * temperature + 0.3 * rainfall + 0.15 * soil_quality + 0.05 * fertilizer +
              np.random.normal(0, 10, n_samples))

# Create DataFrame
df = pd.DataFrame({
    'Temperature': temperature,
    'Rainfall': rainfall,
    'Soil_Quality': soil_quality,
    'Fertilizer': fertilizer,
    'Crop_Yield': crop_yield
})

# Split the data
X = df.drop('Crop_Yield', axis=1)
y = df['Crop_Yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Calculate R-squared and adjusted R-squared
train_r2 = model.score(X_train_scaled, y_train)
test_r2 = model.score(X_test_scaled, y_test)

train_adj_r2 = adjusted_r_squared(train_r2, len(y_train), X_train.shape[1])
test_adj_r2 = adjusted_r_squared(test_r2, len(y_test), X_test.shape[1])

print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
print(f"Train Adjusted R²: {train_adj_r2:.4f}, Test Adjusted R²: {test_adj_r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': abs(model.coef_)
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)
```

Slide 12: Overfitting and Underfitting

R-squared and adjusted R-squared can help identify overfitting and underfitting in models. Overfitting occurs when a model learns the training data too well, including noise, leading to poor generalization. Underfitting happens when a model is too simple to capture the underlying patterns in the data. Both R-squared and adjusted R-squared can provide insights into these issues.

Slide 13: Overfitting and Underfitting

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Generate sample data
np.random.seed(42)
X = np.sort(np.random.rand(100, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Create and fit models with different complexities
degrees = [1, 3, 15]
plt.figure(figsize=(14, 4))

for i, degree in enumerate(degrees):
    ax = plt.subplot(1, 3, i + 1)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    
    X_test = np.linspace(0, 1, 100).reshape(-1, 1)
    plt.scatter(X, y, color='blue', s=10, alpha=0.5)
    plt.plot(X_test, model.predict(X_test), color='red')
    plt.ylim((-2, 2))
    plt.title(f"Degree {degree}")

    r2 = model.score(X, y)
    adj_r2 = adjusted_r_squared(r2, len(y), degree + 1)
    plt.text(0.05, -1.5, f"R²: {r2:.4f}\nAdj R²: {adj_r2:.4f}")

plt.tight_layout()
plt.show()
```

Slide 14: Cross-validation and R-squared

Cross-validation is a technique used to assess how well a model generalizes to unseen data. By combining cross-validation with R-squared, we can get a more robust estimate of a model's performance and avoid overfitting. This approach helps in selecting the best model and hyperparameters.

```python
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate sample data
np.random.seed(42)
X = np.sort(np.random.rand(100, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Perform cross-validation for different polynomial degrees
degrees = range(1, 11)
cv_r2_scores = []

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_r2_scores.append(np.mean(cv_scores))

# Plot results
plt.plot(degrees, cv_r2_scores, marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Cross-Validated R²')
plt.title('Cross-Validated R² for Different Polynomial Degrees')
plt.show()

# Find the best degree
best_degree = degrees[np.argmax(cv_r2_scores)]
print(f"Best polynomial degree: {best_degree}")
print(f"Best mean cross-validated R²: {max(cv_r2_scores):.4f}")
```

Slide 15: Limitations and Alternatives

While R-squared and adjusted R-squared are useful metrics, they have limitations. They don't indicate whether the model's predictions are biased, and they can be sensitive to outliers. Alternative metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), or Mean Absolute Error (MAE) can provide additional insights into model performance.

Slide 16: Limitations and Alternatives

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1)
y_true = 2 + 3 * X + np.random.randn(100, 1) * 0.1
y_pred = 2.1 + 2.9 * X  # Slightly biased predictions

# Calculate various metrics
r2 = r2_score(y_true, y_pred)
adj_r2 = adjusted_r_squared(r2, len(y_true), 1)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)

print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adj_r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Visualize the predictions
plt.scatter(X, y_true, color='blue', alpha=0.5, label='True')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('True vs Predicted Values')
plt.show()
```

Slide 17: Conclusion and Best Practices

R-squared and adjusted R-squared are valuable tools for assessing model performance in regression analysis. To use them effectively:

1. Consider both R-squared and adjusted R-squared when comparing models with different numbers of predictors.
2. Use cross-validation to get more robust estimates of model performance.
3. Don't rely solely on these metrics; consider other evaluation measures and domain knowledge.
4. Be cautious of overfitting, especially with small sample sizes or many predictors.
5. Remember that a high R-squared doesn't necessarily mean a good model; always validate predictions and check for practical significance.

Slide 18: Conclusion and Best Practices

```python
# Pseudocode for model evaluation best practices

def evaluate_model(X, y, model):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Calculate R-squared and adjusted R-squared
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    train_adj_r2 = adjusted_r_squared(train_r2, len(y_train), X_train.shape[1])
    test_adj_r2 = adjusted_r_squared(test_r2, len(y_test), X_test.shape[1])
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    # Calculate additional metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Return all evaluation metrics
    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_adj_r2': train_adj_r2,
        'test_adj_r2': test_adj_r2,
        'cv_r2_mean': np.mean(cv_scores),
        'cv_r2_std': np.std(cv_scores),
        'mse': mse,
        'mae': mae
    }

# Use this function to evaluate and compare different models
```

Slide 95: Additional Resources

For those interested in diving deeper into R-squared, adjusted R-squared, and related topics in regression analysis, here are some valuable resources:

1. ArXiv paper: "On the Use and Interpretation of R-squared" by Aris Spanos and Deborah G. Mayo ([https://arxiv.org/abs/1511.03246](https://arxiv.org/abs/1511.03246))
2. ArXiv paper: "A Comprehensive Survey of Regression Based Loss Functions" by Shriram Sankaran et al. ([https://arxiv.org/abs/2103.15278](https://arxiv.org/abs/2103.15278))

These papers provide in-depth discussions on the theoretical foundations and practical applications of R-squared and related concepts in statistical modeling and machine learning.

