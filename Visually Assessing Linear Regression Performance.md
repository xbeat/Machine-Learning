## Visually Assessing Linear Regression Performance
Slide 1: Assessing Linear Regression Performance with Residual Distribution Plots

Linear regression is a fundamental statistical technique used to model the relationship between variables. While the regression line itself provides valuable insights, assessing the model's performance requires a deeper look. One powerful yet underrated tool for this purpose is the residual distribution plot.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 1, 100)

# Perform linear regression
coeffs = np.polyfit(X, y, 1)
y_pred = np.polyval(coeffs, X)

# Calculate residuals
residuals = y - y_pred

# Plot residual distribution
plt.hist(residuals, bins=20, edgecolor='black')
plt.title('Residual Distribution Plot')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 2: Understanding Residuals in Linear Regression

Residuals are the differences between the observed values and the predicted values in a regression model. They play a crucial role in assessing how well the model fits the data. In an ideal scenario, residuals should be randomly distributed around zero, indicating that the model captures the underlying relationship well.

```python
# Visualize residuals
plt.scatter(X, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Independent Variable')
plt.xlabel('X')
plt.ylabel('Residual')
plt.show()
```

Slide 3: The Importance of Normally Distributed Residuals

One key assumption of linear regression is that the residuals follow a normal distribution. This assumption is crucial because it underpins the validity of statistical inferences drawn from the model. A normal distribution of residuals suggests that the model's errors are random and not systematically biased.

```python
import scipy.stats as stats

# Q-Q plot to check normality
fig, ax = plt.subplots()
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Q-Q plot of residuals")
plt.show()
```

Slide 4: Characteristics of a Good Residual Distribution Plot

A well-performing linear regression model should produce a residual distribution plot that:

1.  Follows a normal distribution, appearing symmetrical and bell-shaped.
2.  Is centered around zero, indicating unbiased predictions.
3.  Shows no clear patterns or trends when plotted against predicted values or independent variables.

```python
# Residual plot against predicted values
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()
```

Slide 5: Red Flags in Residual Distribution Plots

Certain patterns in residual plots can indicate issues with the model:

1.  Skewness: Asymmetry in the distribution suggests non-linearity or the presence of outliers.
2.  Heavy tails: Excess kurtosis may indicate the presence of outliers or heteroscedasticity.
3.  Multimodality: Multiple peaks in the distribution could suggest the need for additional predictors or non-linear terms.

```python
# Generate non-linear data
X_nl = np.linspace(0, 10, 100)
y_nl = 2 * X_nl**2 + 1 + np.random.normal(0, 5, 100)

# Fit linear model to non-linear data
coeffs_nl = np.polyfit(X_nl, y_nl, 1)
y_pred_nl = np.polyval(coeffs_nl, X_nl)
residuals_nl = y_nl - y_pred_nl

# Plot residual distribution for non-linear data
plt.hist(residuals_nl, bins=20, edgecolor='black')
plt.title('Residual Distribution Plot (Non-linear Data)')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 6: Detecting Heteroscedasticity

Heteroscedasticity occurs when the variability of residuals is not constant across all levels of the independent variables. This violation can lead to unreliable standard errors and confidence intervals. A residual plot can help detect this issue by revealing a fan or cone shape.

```python
# Generate heteroscedastic data
X_hetero = np.linspace(0, 10, 100)
y_hetero = 2 * X_hetero + np.random.normal(0, 0.5 * X_hetero, 100)

# Fit linear model
coeffs_hetero = np.polyfit(X_hetero, y_hetero, 1)
y_pred_hetero = np.polyval(coeffs_hetero, X_hetero)
residuals_hetero = y_hetero - y_pred_hetero

# Plot residuals
plt.scatter(X_hetero, residuals_hetero)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. X (Heteroscedastic)')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.show()
```

Slide 7: Dealing with Non-Normality: Transformations

When residuals are not normally distributed, transformations of the dependent or independent variables can sometimes help. Common transformations include logarithmic, square root, and Box-Cox transformations. These can help linearize relationships and stabilize variance.

```python
# Log transformation example
y_log = np.log(y_nl)
coeffs_log = np.polyfit(X_nl, y_log, 1)
y_pred_log = np.polyval(coeffs_log, X_nl)
residuals_log = y_log - y_pred_log

# Plot transformed residuals
plt.hist(residuals_log, bins=20, edgecolor='black')
plt.title('Residual Distribution Plot (Log-transformed)')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 8: Residual Plots for High-Dimensional Data

In high-dimensional datasets, visualizing the regression line becomes challenging. However, the residual distribution plot remains a powerful tool as it condenses the model's performance into a one-dimensional representation, regardless of the number of predictors.

```python
# Generate high-dimensional data
np.random.seed(42)
X_high_dim = np.random.rand(100, 5)  # 5 predictors
y_high_dim = np.sum(X_high_dim, axis=1) + np.random.normal(0, 0.5, 100)

# Fit linear model
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_high_dim, y_high_dim)
y_pred_high_dim = model.predict(X_high_dim)
residuals_high_dim = y_high_dim - y_pred_high_dim

# Plot residual distribution
plt.hist(residuals_high_dim, bins=20, edgecolor='black')
plt.title('Residual Distribution Plot (High-Dimensional Data)')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 9: Interpreting Residual Plots: A Case Study

Let's examine a real-world scenario where residual plots reveal model inadequacies. Consider a study on the relationship between a city's population and its crime rate. Initial linear regression might seem satisfactory, but residual analysis tells a different story.

```python
# Simulated city data
np.random.seed(42)
population = np.linspace(10000, 1000000, 100)
crime_rate = 0.05 * np.sqrt(population) + np.random.normal(0, 2, 100)

# Linear regression
coeffs = np.polyfit(population, crime_rate, 1)
crime_rate_pred = np.polyval(coeffs, population)
residuals = crime_rate - crime_rate_pred

# Residual plot
plt.scatter(population, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Population')
plt.xlabel('Population')
plt.ylabel('Residuals')
plt.show()
```

Slide 10: Interpreting the Case Study Results

The residual plot from our city crime rate example shows a clear curved pattern, indicating that the relationship between population and crime rate is not linear. This suggests that our initial linear model is inadequate and fails to capture the true relationship between the variables.

```python
# Histogram of residuals
plt.hist(residuals, bins=20, edgecolor='black')
plt.title('Residual Distribution (Crime Rate Model)')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.show()

# Q-Q plot
fig, ax = plt.subplots()
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Q-Q plot of residuals (Crime Rate Model)")
plt.show()
```

Slide 11: Improving the Model Based on Residual Analysis

Based on the residual analysis, we can improve our model by considering a non-linear relationship. In this case, a square root transformation of the population might be appropriate.

```python
# Improved model with square root transformation
population_sqrt = np.sqrt(population)
coeffs_improved = np.polyfit(population_sqrt, crime_rate, 1)
crime_rate_pred_improved = np.polyval(coeffs_improved, population_sqrt)
residuals_improved = crime_rate - crime_rate_pred_improved

# Residual plot for improved model
plt.scatter(population, residuals_improved)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Population (Improved Model)')
plt.xlabel('Population')
plt.ylabel('Residuals')
plt.show()
```

Slide 12: Comparing Original and Improved Models

By comparing the residual plots of the original and improved models, we can see a significant improvement in the distribution of residuals. The improved model shows a more random scatter around zero, indicating a better fit to the data.

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original model residuals
ax1.hist(residuals, bins=20, edgecolor='black')
ax1.set_title('Original Model Residuals')
ax1.set_xlabel('Residual Value')
ax1.set_ylabel('Frequency')

# Improved model residuals
ax2.hist(residuals_improved, bins=20, edgecolor='black')
ax2.set_title('Improved Model Residuals')
ax2.set_xlabel('Residual Value')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

Slide 13: Real-Life Example: Predicting House Prices

Let's consider another real-life example: predicting house prices based on square footage. This example demonstrates how residual analysis can reveal the need for additional predictors or non-linear terms in the model.

```python
# Simulated house price data
np.random.seed(42)
sqft = np.linspace(1000, 5000, 200)
price = 100000 + 150 * sqft + 0.05 * sqft**2 + np.random.normal(0, 50000, 200)

# Linear regression
coeffs = np.polyfit(sqft, price, 1)
price_pred = np.polyval(coeffs, sqft)
residuals = price - price_pred

# Residual plot
plt.scatter(sqft, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Square Footage (House Prices)')
plt.xlabel('Square Footage')
plt.ylabel('Residuals')
plt.show()
```

Slide 14: Interpreting the House Price Model Residuals

The residual plot for our house price model shows a clear quadratic pattern, indicating that a simple linear model is insufficient. This suggests that the relationship between square footage and price is non-linear, possibly due to factors like location or housing market dynamics.

```python
# Improved model with quadratic term
coeffs_quad = np.polyfit(sqft, price, 2)
price_pred_quad = np.polyval(coeffs_quad, sqft)
residuals_quad = price - price_pred_quad

# Residual plot for improved model
plt.scatter(sqft, residuals_quad)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Square Footage (Improved Model)')
plt.xlabel('Square Footage')
plt.ylabel('Residuals')
plt.show()
```

Slide 15: Conclusion and Best Practices

Residual distribution plots are powerful tools for assessing linear regression performance. They help identify violations of model assumptions and guide improvements. Best practices include:

1.  Always plot residuals against predicted values and independent variables.
2.  Use Q-Q plots to assess normality.
3.  Consider transformations or additional predictors when residuals show patterns.
4.  Remember that a good residual plot doesn't guarantee a perfect model, but a bad one almost always indicates problems.

By incorporating residual analysis into your regression workflow, you can build more reliable and accurate models, leading to better insights and predictions.

Slide 16: Additional Resources

For those interested in diving deeper into residual analysis and linear regression diagnostics, the following resources are recommended:

1.  Gelman, A., & Hill, J. (2006). Data Analysis Using Regression and Multilevel/Hierarchical Models. Cambridge University Press.
2.  Cook, R. D., & Weisberg, S. (1982). Residuals and Influence in Regression. Chapman and Hall.
3.  ArXiv paper: "Diagnostic Plots for the Quality of Linear Regression Models" by M. Friendly and D. Denis. Available at: [https://arxiv.org/abs/stat.AP/0406049](https://arxiv.org/abs/stat.AP/0406049)

These resources provide in-depth discussions on the theory and application of residual analysis in linear regression and other statistical models.

