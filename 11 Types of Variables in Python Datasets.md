## 11 Types of Variables in Python Datasets
Slide 1: Independent Variables

Independent variables are features used as input to predict an outcome. They are manipulated or controlled in experiments to observe their effect on the dependent variable. In data analysis, these are the variables we use to make predictions or explain relationships.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Create a sample dataset
data = {
    'temperature': [20, 25, 30, 35, 40],
    'ice_cream_sales': [100, 150, 200, 250, 300]
}
df = pd.DataFrame(data)

# Plot the relationship
plt.figure(figsize=(10, 6))
plt.scatter(df['temperature'], df['ice_cream_sales'])
plt.xlabel('Temperature (°C)')
plt.ylabel('Ice Cream Sales')
plt.title('Effect of Temperature on Ice Cream Sales')
plt.show()

# The temperature is the independent variable in this example
```

Slide 2: Dependent Variables

The dependent variable is the outcome being predicted or measured in an experiment or analysis. It's influenced by changes in the independent variables. In our previous example, ice cream sales were the dependent variable affected by temperature changes.

```python
# Continuing from the previous example
from sklearn.linear_model import LinearRegression

# Prepare the data
X = df[['temperature']]
y = df['ice_cream_sales']

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
new_temperatures = [[22], [28], [33]]
predictions = model.predict(new_temperatures)

print("Predicted ice cream sales:")
for temp, sales in zip(new_temperatures, predictions):
    print(f"Temperature: {temp[0]}°C, Predicted Sales: {sales:.2f}")
```

Slide 3: Interaction Variables

Interaction variables represent the combined effect of two or more independent variables on the dependent variable. They are crucial in understanding complex relationships where the impact of one variable depends on the level of another.

```python
import numpy as np

# Create a sample dataset with interaction
np.random.seed(42)
hours_studied = np.random.randint(1, 11, 100)
hours_slept = np.random.randint(4, 10, 100)
exam_score = 50 + 2 * hours_studied + 3 * hours_slept + 0.5 * hours_studied * hours_slept + np.random.normal(0, 5, 100)

data = pd.DataFrame({
    'hours_studied': hours_studied,
    'hours_slept': hours_slept,
    'exam_score': exam_score
})

# Create interaction term
data['interaction'] = data['hours_studied'] * data['hours_slept']

# Fit linear regression model
X = data[['hours_studied', 'hours_slept', 'interaction']]
y = data['exam_score']
model = LinearRegression()
model.fit(X, y)

print("Coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
```

Slide 4: Latent Variables

Latent variables are not directly observed but inferred from other observed variables. They often represent underlying constructs or hidden factors in a dataset. Clustering algorithms are commonly used to uncover latent structures in data.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2)
X[:100] += 3
X[100:200] -= 3

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.title('K-means Clustering: Uncovering Latent Structure')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()

# The cluster labels represent a latent variable
```

Slide 5: Confounding Variables

Confounding variables are factors that influence both the independent and dependent variables, potentially leading to misleading conclusions about the relationship between them. Identifying and controlling for confounders is crucial in causal inference studies.

```python
import seaborn as sns

# Generate sample data with a confounding variable
np.random.seed(42)
n_samples = 1000

temperature = np.random.normal(25, 5, n_samples)
ice_cream_sales = 100 + 2 * temperature + np.random.normal(0, 10, n_samples)
beach_visits = 50 + 1.5 * temperature + np.random.normal(0, 10, n_samples)

data = pd.DataFrame({
    'Temperature': temperature,
    'Ice Cream Sales': ice_cream_sales,
    'Beach Visits': beach_visits
})

# Visualize the relationships
sns.pairplot(data, height=4)
plt.suptitle('Relationships between Variables', y=1.02)
plt.tight_layout()
plt.show()

# Calculate correlations
correlations = data.corr()
print("Correlations:")
print(correlations)

# Temperature is a confounding variable affecting both ice cream sales and beach visits
```

Slide 6: Correlated Variables

Correlated variables are those that have a statistical relationship with each other. High correlation between independent variables can lead to multicollinearity issues in regression analysis, affecting the model's interpretability and stability.

```python
from scipy import stats

# Continuing from the previous example
correlation, p_value = stats.pearsonr(data['Ice Cream Sales'], data['Beach Visits'])

print(f"Correlation between Ice Cream Sales and Beach Visits: {correlation:.2f}")
print(f"P-value: {p_value:.4f}")

# Visualize the correlation
plt.figure(figsize=(10, 6))
sns.regplot(x='Ice Cream Sales', y='Beach Visits', data=data)
plt.title('Correlation between Ice Cream Sales and Beach Visits')
plt.show()

# Ice cream sales and beach visits are correlated due to their common relationship with temperature
```

Slide 7: Control Variables

Control variables are factors held constant in an experiment or analysis to isolate the effect of the independent variable on the dependent variable. They help reduce confounding effects and improve the validity of causal inferences.

```python
from statsmodels.formula.api import ols

# Continuing from the previous example
# Let's control for temperature to examine the relationship between ice cream sales and beach visits

# Fit a model without controlling for temperature
model_without_control = ols('Beach_Visits ~ Ice_Cream_Sales', data=data).fit()

# Fit a model controlling for temperature
model_with_control = ols('Beach_Visits ~ Ice_Cream_Sales + Temperature', data=data).fit()

print("Model without controlling for temperature:")
print(model_without_control.summary().tables[1])

print("\nModel controlling for temperature:")
print(model_with_control.summary().tables[1])

# Notice how the coefficient for Ice_Cream_Sales changes when we control for temperature
```

Slide 8: Leaky Variables

Leaky variables unintentionally provide information about the target variable that would not be available at the time of prediction. They can lead to overly optimistic model performance and should be identified and removed to ensure realistic evaluations.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(42)
n_samples = 1000

X = np.random.randn(n_samples, 2)
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n_samples)

# Create a leaky variable (future information)
leaky_var = y + np.random.randn(n_samples) * 0.1

# Add the leaky variable to the feature set
X_with_leak = np.column_stack((X, leaky_var))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_leak_train, X_leak_test, _, _ = train_test_split(X_with_leak, y, test_size=0.2, random_state=42)

# Train and evaluate models
model_without_leak = LinearRegression().fit(X_train, y_train)
model_with_leak = LinearRegression().fit(X_leak_train, y_train)

mse_without_leak = mean_squared_error(y_test, model_without_leak.predict(X_test))
mse_with_leak = mean_squared_error(y_test, model_with_leak.predict(X_leak_test))

print(f"MSE without leak: {mse_without_leak:.4f}")
print(f"MSE with leak: {mse_with_leak:.4f}")

# The model with the leaky variable shows unrealistically good performance
```

Slide 9: Stationary Variables

Stationary variables have statistical properties (mean, variance) that do not change over time. They are crucial in time series analysis and modeling, as many statistical techniques assume stationarity.

```python
from statsmodels.tsa.stattools import adfuller

# Generate a stationary time series
np.random.seed(42)
stationary_series = np.random.randn(1000)

# Generate a non-stationary time series (random walk)
non_stationary_series = np.cumsum(np.random.randn(1000))

# Function to test stationarity
def test_stationarity(series, series_name):
    result = adfuller(series)
    print(f"ADF test for {series_name}:")
    print(f"Test statistic: {result[0]:.4f}")
    print(f"P-value: {result[1]:.4f}")
    print(f"Stationary: {result[1] < 0.05}")

# Test both series
test_stationarity(stationary_series, "Stationary Series")
print("\n")
test_stationarity(non_stationary_series, "Non-stationary Series")

# Plot both series
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(stationary_series)
ax1.set_title("Stationary Series")
ax2.plot(non_stationary_series)
ax2.set_title("Non-stationary Series")
plt.tight_layout()
plt.show()
```

Slide 10: Non-stationary Variables

Non-stationary variables have statistical properties that change over time. They are common in real-world time series data and often require transformation or differencing to achieve stationarity for analysis.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate a non-stationary time series with trend and seasonality
t = np.arange(1000)
trend = 0.02 * t
seasonality = 10 * np.sin(2 * np.pi * t / 365)
noise = np.random.randn(1000)
non_stationary_series = trend + seasonality + noise

# Decompose the series
decomposition = seasonal_decompose(non_stationary_series, model='additive', period=365)

# Plot the decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
decomposition.observed.plot(ax=ax1)
ax1.set_title("Original Non-stationary Series")
decomposition.trend.plot(ax=ax2)
ax2.set_title("Trend")
decomposition.seasonal.plot(ax=ax3)
ax3.set_title("Seasonality")
decomposition.resid.plot(ax=ax4)
ax4.set_title("Residuals")
plt.tight_layout()
plt.show()

# Test stationarity of the residuals
test_stationarity(decomposition.resid.dropna(), "Residuals")
```

Slide 11: Lagged Variables

Lagged variables represent previous time points' values of a given variable, shifting the data series by a specified number of periods. They are essential in time series analysis and forecasting, capturing temporal dependencies in the data.

```python
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Generate a sample time series
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
values = np.cumsum(np.random.randn(365)) + np.sin(np.arange(365) * 2 * np.pi / 365) * 10
ts = pd.Series(values, index=dates)

# Create lagged variables
df = pd.DataFrame({'y': ts})
for i in range(1, 4):
    df[f'lag_{i}'] = df['y'].shift(i)

# Display the first few rows
print(df.head())

# Plot ACF and PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plot_acf(ts, ax=ax1, lags=20)
ax1.set_title("Autocorrelation Function (ACF)")
plot_pacf(ts, ax=ax2, lags=20)
ax2.set_title("Partial Autocorrelation Function (PACF)")
plt.tight_layout()
plt.show()

# The ACF and PACF plots help identify significant lag relationships
```

Slide 12: Real-life Example: Weather Prediction

In this example, we'll use various types of variables to predict daily maximum temperature based on historical weather data.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample weather data
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
n_samples = len(dates)

data = pd.DataFrame({
    'date': dates,
    'max_temp': 20 + 15 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.randn(n_samples) * 3,
    'min_temp': 10 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.randn(n_samples) * 2,
    'humidity': np.random.randint(30, 90, n_samples),
    'wind_speed': np.random.randint(0, 30, n_samples)
})

# Create lagged variables
data['max_temp_lag1'] = data['max_temp'].shift(1)
data['max_temp_lag2'] = data['max_temp'].shift(2)

# Create interaction variable
data['temp_humidity_interaction'] = data['min_temp'] * data['humidity']

# Add day of year as a feature (to capture seasonality)
data['day_of_year'] = data['date'].dt.dayofyear

# Prepare features and target
features = ['min_temp', 'humidity', 'wind_speed', 'max_temp_lag1', 'max_temp_lag2', 'temp_humidity_interaction', 'day_of_year']
X = data.dropna()[features]
y = data.dropna()['max_temp']

# Split the data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Print feature importances
for feature, importance in zip(features, model.coef_):
    print(f"{feature}: {importance:.4f}")
```

Slide 13: Real-life Example: Crop Yield Prediction

In this example, we'll use various types of variables to predict crop yield based on environmental factors.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample crop yield data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'temperature': np.random.uniform(15, 35, n_samples),
    'rainfall': np.random.uniform(500, 1500, n_samples),
    'soil_quality': np.random.uniform(0, 1, n_samples),
    'fertilizer_amount': np.random.uniform(50, 200, n_samples),
    'pest_resistance': np.random.uniform(0, 1, n_samples)
})

# Create interaction variable
data['temp_rain_interaction'] = data['temperature'] * data['rainfall']

# Create a non-linear relationship for yield
data['yield'] = (
    0.5 * data['temperature'] +
    0.3 * data['rainfall'] +
    10 * data['soil_quality'] +
    0.1 * data['fertilizer_amount'] +
    5 * data['pest_resistance'] +
    0.001 * data['temp_rain_interaction'] +
    np.random.normal(0, 2, n_samples)
)

# Prepare features and target
X = data.drop('yield', axis=1)
y = data['yield']

# Split the data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Print feature importances
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")
```

Slide 14: Additional Resources

For those interested in diving deeper into the topic of variables in datasets and their applications in data science and machine learning, here are some valuable resources:

1. "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani - This book provides an excellent overview of statistical learning methods and their applications.
2. "Feature Engineering for Machine Learning" by Alice Zheng and Amanda Casari - This book focuses on the art and science of creating effective features for machine learning models.
3. "Time Series Analysis and Its Applications" by Shumway and Stoffer - For those interested in time series analysis and working with lagged variables, this book is an excellent resource.
4. ArXiv.org - This open-access repository contains numerous research papers on advanced topics related to variable selection, feature engineering, and machine learning. Some relevant papers include:
   * "A Survey on Feature Selection Methods" (arXiv:1904.02368)
   * "An Overview of Deep Learning in Medical Imaging Focusing on MRI" (arXiv:1811.10052)

Remember to verify the accuracy and relevance of these resources, as the field of data science is rapidly evolving.

