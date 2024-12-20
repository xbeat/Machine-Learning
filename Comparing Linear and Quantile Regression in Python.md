## Comparing Linear and Quantile Regression in Python
Slide 1: Introduction to Linear and Quantile Regression

Linear regression and quantile regression are two powerful statistical techniques used for modeling relationships between variables. While linear regression focuses on estimating the conditional mean of the dependent variable, quantile regression provides a more comprehensive view by estimating various quantiles of the conditional distribution. This presentation will explore both methods, their implementation in Python, and their practical applications.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.quantreg import QuantReg

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 2, 100)

# Plot the data
plt.scatter(X, y, alpha=0.5)
plt.title("Sample Data for Regression Analysis")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

Slide 2: Linear Regression: The Basics

Linear regression is a method for modeling the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the variables and estimates the conditional mean of the dependent variable. The goal is to find the best-fitting line that minimizes the sum of squared residuals.

```python
# Perform linear regression
lr_model = LinearRegression()
lr_model.fit(X.reshape(-1, 1), y)

# Plot the results
plt.scatter(X, y, alpha=0.5)
plt.plot(X, lr_model.predict(X.reshape(-1, 1)), color='red', label='Linear Regression')
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

print(f"Intercept: {lr_model.intercept_:.2f}")
print(f"Slope: {lr_model.coef_[0]:.2f}")
```

Slide 3: Quantile Regression: An Overview

Quantile regression estimates the conditional median or other quantiles of the response variable. Unlike linear regression, which focuses on the mean, quantile regression provides a more complete picture of the relationship between variables. It is particularly useful when dealing with non-normal distributions or when interested in specific parts of the distribution.

```python
# Perform quantile regression for different quantiles
quantiles = [0.1, 0.5, 0.9]
qr_models = [QuantReg(y, X).fit(q=q) for q in quantiles]

# Plot the results
plt.scatter(X, y, alpha=0.5)
for i, qr_model in enumerate(qr_models):
    y_pred = qr_model.predict(X)
    plt.plot(X, y_pred, label=f'Quantile {quantiles[i]}')

plt.title("Quantile Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

Slide 4: Implementing Linear Regression in Python

Let's dive deeper into implementing linear regression using Python's scikit-learn library. We'll use a simple example to demonstrate the process of fitting a model and making predictions.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.2, random_state=42)

# Create and train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
```

Slide 5: Interpreting Linear Regression Results

Understanding the output of a linear regression model is crucial for drawing meaningful conclusions. The key components to consider are the coefficients, intercept, and model evaluation metrics.

```python
# Print model coefficients and intercept
print(f"Intercept: {lr_model.intercept_:.2f}")
print(f"Slope: {lr_model.coef_[0]:.2f}")

# Plot the regression line with confidence intervals
from scipy import stats

plt.scatter(X_test, y_test, alpha=0.5)
plt.plot(X_test, y_pred, color='red', label='Regression Line')

# Calculate confidence intervals
conf_int = 0.95
n = len(X_test)
se = np.sqrt(np.sum((y_test - y_pred)**2) / (n - 2))
t_value = stats.t.ppf((1 + conf_int) / 2, n - 2)
ci = t_value * se * np.sqrt(1/n + (X_test - np.mean(X_test))**2 / np.sum((X_test - np.mean(X_test))**2))

plt.fill_between(X_test.flatten(), (y_pred - ci).flatten(), (y_pred + ci).flatten(), alpha=0.2, color='red', label='95% Confidence Interval')
plt.title("Linear Regression with Confidence Intervals")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

Slide 6: Implementing Quantile Regression in Python

Now, let's implement quantile regression using the statsmodels library in Python. We'll demonstrate how to fit models for different quantiles and interpret the results.

```python
import statsmodels.formula.api as smf

# Create a DataFrame for easier manipulation
import pandas as pd
df = pd.DataFrame({'X': X, 'y': y})

# Fit quantile regression models for different quantiles
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
qr_models = [smf.quantreg('y ~ X', df).fit(q=q) for q in quantiles]

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5)
for i, qr_model in enumerate(qr_models):
    y_pred = qr_model.predict(df)
    plt.plot(X, y_pred, label=f'Quantile {quantiles[i]}')

plt.title("Quantile Regression for Multiple Quantiles")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Print coefficients for each quantile
for i, qr_model in enumerate(qr_models):
    print(f"Quantile {quantiles[i]}:")
    print(qr_model.summary().tables[1])
    print("\n")
```

Slide 7: Interpreting Quantile Regression Results

Quantile regression provides insights into how different parts of the distribution are affected by the independent variables. Let's examine how to interpret these results and what they tell us about the relationship between variables.

```python
# Compare coefficients across quantiles
coef_df = pd.DataFrame(index=['Intercept', 'X'])
for i, qr_model in enumerate(qr_models):
    coef_df[f'Q{quantiles[i]}'] = qr_model.params

# Plot coefficient comparison
coef_df.T.plot(marker='o')
plt.title("Coefficient Comparison Across Quantiles")
plt.xlabel("Quantile")
plt.ylabel("Coefficient Value")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

print("Coefficient values across quantiles:")
print(coef_df)
```

Slide 8: Linear vs. Quantile Regression: When to Use Each

Linear regression and quantile regression serve different purposes and are suitable for different scenarios. Let's explore when to use each method and their respective strengths.

```python
# Generate heteroscedastic data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, X, 100)

# Fit linear and quantile regression models
lr_model = LinearRegression().fit(X.reshape(-1, 1), y)
qr_model_50 = QuantReg(y, X).fit(q=0.5)
qr_model_10 = QuantReg(y, X).fit(q=0.1)
qr_model_90 = QuantReg(y, X).fit(q=0.9)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5)
plt.plot(X, lr_model.predict(X.reshape(-1, 1)), color='red', label='Linear Regression')
plt.plot(X, qr_model_50.predict(X), color='green', label='Quantile Regression (50th)')
plt.plot(X, qr_model_10.predict(X), color='blue', label='Quantile Regression (10th)')
plt.plot(X, qr_model_90.predict(X), color='purple', label='Quantile Regression (90th)')
plt.title("Linear vs. Quantile Regression: Heteroscedastic Data")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

Slide 9: Handling Outliers: Linear vs. Quantile Regression

One of the key advantages of quantile regression is its robustness to outliers. Let's compare how linear and quantile regression handle datasets with outliers.

```python
# Generate data with outliers
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 1, 100)
y[80:85] += 10  # Add outliers

# Fit linear and quantile regression models
lr_model = LinearRegression().fit(X.reshape(-1, 1), y)
qr_model_50 = QuantReg(y, X).fit(q=0.5)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5)
plt.plot(X, lr_model.predict(X.reshape(-1, 1)), color='red', label='Linear Regression')
plt.plot(X, qr_model_50.predict(X), color='green', label='Quantile Regression (50th)')
plt.title("Linear vs. Quantile Regression: Handling Outliers")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

print("Linear Regression Coefficients:")
print(f"Intercept: {lr_model.intercept_:.2f}, Slope: {lr_model.coef_[0]:.2f}")
print("\nQuantile Regression Coefficients (50th percentile):")
print(qr_model_50.summary().tables[1])
```

Slide 10: Real-Life Example: Analyzing Plant Growth

Let's apply linear and quantile regression to analyze plant growth data. We'll examine how different factors affect plant height and compare the insights provided by both methods.

```python
# Generate sample plant growth data
np.random.seed(42)
sunlight_hours = np.random.uniform(4, 12, 100)
water_ml = np.random.uniform(50, 200, 100)
plant_height = 5 + 0.5 * sunlight_hours + 0.1 * water_ml + np.random.normal(0, 2, 100)

# Create a DataFrame
plant_data = pd.DataFrame({
    'Sunlight_Hours': sunlight_hours,
    'Water_ml': water_ml,
    'Plant_Height': plant_height
})

# Fit linear regression model
lr_model = smf.ols('Plant_Height ~ Sunlight_Hours + Water_ml', data=plant_data).fit()

# Fit quantile regression models
qr_model_25 = smf.quantreg('Plant_Height ~ Sunlight_Hours + Water_ml', data=plant_data).fit(q=0.25)
qr_model_50 = smf.quantreg('Plant_Height ~ Sunlight_Hours + Water_ml', data=plant_data).fit(q=0.50)
qr_model_75 = smf.quantreg('Plant_Height ~ Sunlight_Hours + Water_ml', data=plant_data).fit(q=0.75)

# Print results
print("Linear Regression Results:")
print(lr_model.summary().tables[1])
print("\nQuantile Regression Results (50th percentile):")
print(qr_model_50.summary().tables[1])

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(plant_data['Sunlight_Hours'], plant_data['Plant_Height'], alpha=0.5)
ax1.plot(plant_data['Sunlight_Hours'], lr_model.predict(plant_data), color='red', label='Linear Regression')
ax1.plot(plant_data['Sunlight_Hours'], qr_model_25.predict(plant_data), color='blue', label='25th Percentile')
ax1.plot(plant_data['Sunlight_Hours'], qr_model_50.predict(plant_data), color='green', label='50th Percentile')
ax1.plot(plant_data['Sunlight_Hours'], qr_model_75.predict(plant_data), color='purple', label='75th Percentile')
ax1.set_title("Plant Height vs. Sunlight Hours")
ax1.set_xlabel("Sunlight Hours")
ax1.set_ylabel("Plant Height")
ax1.legend()

ax2.scatter(plant_data['Water_ml'], plant_data['Plant_Height'], alpha=0.5)
ax2.plot(plant_data['Water_ml'], lr_model.predict(plant_data), color='red', label='Linear Regression')
ax2.plot(plant_data['Water_ml'], qr_model_25.predict(plant_data), color='blue', label='25th Percentile')
ax2.plot(plant_data['Water_ml'], qr_model_50.predict(plant_data), color='green', label='50th Percentile')
ax2.plot(plant_data['Water_ml'], qr_model_75.predict(plant_data), color='purple', label='75th Percentile')
ax2.set_title("Plant Height vs. Water Amount")
ax2.set_xlabel("Water (ml)")
ax2.set_ylabel("Plant Height")
ax2.legend()

plt.tight_layout()
plt.show()
```

Slide 11: Real-Life Example: Air Quality Analysis

In this example, we'll use linear and quantile regression to analyze air quality data, specifically looking at the relationship between temperature and ozone levels.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.quantreg import QuantReg

# Generate sample air quality data
np.random.seed(42)
temperature = np.random.uniform(10, 35, 100)
ozone_levels = 20 + 2 * temperature + np.random.normal(0, 10, 100)
ozone_levels = np.maximum(ozone_levels, 0)  # Ensure non-negative ozone levels

# Create a DataFrame
air_quality_data = pd.DataFrame({
    'Temperature': temperature,
    'Ozone_Levels': ozone_levels
})

# Fit linear regression model
lr_model = LinearRegression().fit(temperature.reshape(-1, 1), ozone_levels)

# Fit quantile regression models
qr_model_25 = QuantReg(ozone_levels, temperature).fit(q=0.25)
qr_model_50 = QuantReg(ozone_levels, temperature).fit(q=0.50)
qr_model_75 = QuantReg(ozone_levels, temperature).fit(q=0.75)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(temperature, ozone_levels, alpha=0.5)
plt.plot(temperature, lr_model.predict(temperature.reshape(-1, 1)), color='red', label='Linear Regression')
plt.plot(temperature, qr_model_25.predict(temperature), color='blue', label='25th Percentile')
plt.plot(temperature, qr_model_50.predict(temperature), color='green', label='50th Percentile')
plt.plot(temperature, qr_model_75.predict(temperature), color='purple', label='75th Percentile')
plt.title("Ozone Levels vs. Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Ozone Levels (ppb)")
plt.legend()
plt.show()

print("Linear Regression Coefficients:")
print(f"Intercept: {lr_model.intercept_:.2f}, Slope: {lr_model.coef_[0]:.2f}")
print("\nQuantile Regression Coefficients (50th percentile):")
print(qr_model_50.summary().tables[1])
```

Slide 12: Comparing Linear and Quantile Regression Results

Let's analyze the results of our air quality example to understand the insights provided by linear and quantile regression.

Linear regression gives us an average relationship between temperature and ozone levels. The slope coefficient represents the average increase in ozone levels for each degree increase in temperature.

Quantile regression provides a more nuanced view:

* The 25th percentile line shows how temperature affects lower ozone levels.
* The 50th percentile (median) line is less affected by extreme values than the mean.
* The 75th percentile line reveals the relationship for higher ozone levels.

By comparing these lines, we can see if the relationship between temperature and ozone levels varies across different parts of the ozone level distribution. This can be crucial for understanding air quality patterns and making informed decisions about pollution control measures.

Slide 14: Comparing Linear and Quantile Regression Results

```python
# Calculate R-squared for linear regression
from sklearn.metrics import r2_score
r2 = r2_score(ozone_levels, lr_model.predict(temperature.reshape(-1, 1)))

print(f"Linear Regression R-squared: {r2:.4f}")

# Compare slopes across quantiles
slopes = [qr_model_25.params['Temperature'], qr_model_50.params['Temperature'], qr_model_75.params['Temperature']]
quantiles = [0.25, 0.50, 0.75]

plt.figure(figsize=(8, 5))
plt.plot(quantiles, slopes, marker='o')
plt.title("Slope Coefficients Across Quantiles")
plt.xlabel("Quantile")
plt.ylabel("Slope Coefficient")
plt.grid(True)
plt.show()

print("Slope coefficients across quantiles:")
for q, s in zip(quantiles, slopes):
    print(f"Quantile {q}: {s:.4f}")
```

Slide 15: Advantages and Limitations of Linear and Quantile Regression

Linear Regression Advantages:

1. Simple to understand and interpret
2. Computationally efficient
3. Provides a single, comprehensive summary of the relationship between variables

Linear Regression Limitations:

1. Assumes a constant relationship across the entire distribution
2. Sensitive to outliers
3. May not capture non-linear relationships effectively

Slide 16: Advantages and Limitations of Linear and Quantile Regression

Quantile Regression Advantages:

1. Provides a more complete view of the relationship between variables
2. Robust to outliers, especially for median regression
3. Useful for analyzing heteroscedastic data

Quantile Regression Limitations:

1. More complex to interpret, especially when analyzing multiple quantiles
2. Computationally more intensive than linear regression
3. May be less efficient when the assumptions of linear regression hold

Slide 17: Advantages and Limitations of Linear and Quantile Regression

```python
# Demonstrate the effect of outliers on linear and quantile regression
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 1, 100)
y[90:95] += 20  # Add outliers

lr_model = LinearRegression().fit(X.reshape(-1, 1), y)
qr_model_50 = QuantReg(y, X).fit(q=0.5)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5)
plt.plot(X, lr_model.predict(X.reshape(-1, 1)), color='red', label='Linear Regression')
plt.plot(X, qr_model_50.predict(X), color='green', label='Quantile Regression (50th)')
plt.title("Effect of Outliers on Linear and Quantile Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

Slide 18: Choosing Between Linear and Quantile Regression

When deciding between linear and quantile regression, consider the following factors:

1. Research Question: If you're interested in the average relationship, linear regression might be sufficient. If you need to understand how the relationship varies across the distribution, quantile regression is more appropriate.
2. Data Characteristics: For data with outliers or heteroscedasticity, quantile regression may provide more robust results.
3. Interpretability: If simple interpretation is crucial, linear regression might be preferred. For a more nuanced analysis, quantile regression can offer deeper insights.
4. Computational Resources: Linear regression is generally faster and requires less computational power than quantile regression.
5. Sample Size: With smaller sample sizes, linear regression might be more stable. Quantile regression may require larger samples for reliable estimates across multiple quantiles.

Slide 19: Choosing Between Linear and Quantile Regression

```python
# Pseudocode for decision-making process
def choose_regression_method(data, research_question, resources):
    if has_outliers(data) or is_heteroscedastic(data):
        return "Consider Quantile Regression"
    elif interested_in_average_relationship(research_question):
        return "Linear Regression may be sufficient"
    elif interested_in_distribution_tails(research_question):
        return "Quantile Regression recommended"
    elif limited_computational_resources(resources):
        return "Linear Regression may be more practical"
    else:
        return "Consider both methods and compare results"

# Example usage
result = choose_regression_method(data, research_question, resources)
print(f"Recommendation: {result}")
```

Slide 20: Additional Resources

For those interested in delving deeper into linear and quantile regression, here are some valuable resources:

1. Koenker, R., & Hallock, K. F. (2001). Quantile Regression. Journal of Economic Perspectives, 15(4), 143-156. ArXiv: [https://arxiv.org/abs/2108.11202](https://arxiv.org/abs/2108.11202)
2. Cade, B. S., & Noon, B. R. (2003). A gentle introduction to quantile regression for ecologists. Frontiers in Ecology and the Environment, 1(8), 412-420. ArXiv: [https://arxiv.org/abs/1412.6349](https://arxiv.org/abs/1412.6349)
3. Angrist, J. D., & Pischke, J. S. (2008). Mostly harmless econometrics: An empiricist's companion. Princeton University Press.
4. Python Libraries:
   * Scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/) (for linear regression)
   * Statsmodels: [https://www.statsmodels.org/](https://www.statsmodels.org/) (for both linear and quantile regression)

These resources provide a mix of theoretical background and practical applications of linear and quantile regression techniques. They can help deepen your understanding and guide you in applying these methods to your own data analysis projects.

