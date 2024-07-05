## Mastering Variables Datasets Using Python
Slide 1: Independent Variables

Independent variables are factors that can be manipulated or controlled in an experiment or study. They are the potential causes or influences on the dependent variable. In data analysis, independent variables are typically used as predictors or features.

```python
import pandas as pd
import numpy as np

# Create a dataset with independent variables
data = {
    'temperature': np.random.uniform(0, 30, 100),
    'humidity': np.random.uniform(30, 90, 100),
    'wind_speed': np.random.uniform(0, 20, 100)
}

df = pd.DataFrame(data)
print(df.head())
```

Slide 2: Dependent Variables

Dependent variables are the outcomes or results that are influenced by the independent variables. They are what we measure or observe in response to changes in the independent variables. In statistical modeling, the dependent variable is typically what we're trying to predict or explain.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Add a dependent variable to our dataset
df['plant_growth'] = 2 * df['temperature'] - 0.5 * df['humidity'] + 0.3 * df['wind_speed'] + np.random.normal(0, 2, 100)

# Perform linear regression
X = df[['temperature', 'humidity', 'wind_speed']]
y = df['plant_growth']

model = LinearRegression().fit(X, y)
print("R-squared:", model.score(X, y))
```

Slide 3: Interaction Variables

Interaction variables represent the combined effect of two or more independent variables on the dependent variable. They capture situations where the effect of one variable depends on the level of another variable.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create dataset with interaction
df['temp_humidity_interaction'] = df['temperature'] * df['humidity']

# Visualize interaction
plt.figure(figsize=(10, 6))
plt.scatter(df['temperature'], df['plant_growth'], c=df['humidity'], cmap='viridis')
plt.colorbar(label='Humidity')
plt.xlabel('Temperature')
plt.ylabel('Plant Growth')
plt.title('Interaction between Temperature and Humidity on Plant Growth')
plt.show()
```

Slide 4: Latent Variables

Latent variables are hidden or unobserved factors that influence observed variables. They can't be directly measured but are inferred from other observable variables. Latent variables are often used in factor analysis and structural equation modeling.

```python
from sklearn.decomposition import FactorAnalysis
import numpy as np
import pandas as pd

# Generate data with latent structure
n_samples = 1000
n_features = 10
n_latent = 2

# Create latent variables
latent = np.random.randn(n_samples, n_latent)

# Create observed variables influenced by latent variables
observed = np.dot(latent, np.random.randn(n_latent, n_features)) + np.random.randn(n_samples, n_features)

# Perform factor analysis
fa = FactorAnalysis(n_components=n_latent, random_state=0)
fa.fit(observed)

# Get the latent variables
latent_variables = fa.transform(observed)
print("Extracted latent variables shape:", latent_variables.shape)
```

Slide 5: Confounding Variables

Confounding variables are factors that influence both the independent and dependent variables, potentially leading to spurious correlations. They can distort the true relationship between variables of interest if not properly controlled for.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Create dataset with a confounding variable
n = 1000
df = pd.DataFrame({
    'exercise': np.random.normal(0, 1, n),
    'age': np.random.uniform(20, 70, n)
})

# Age affects both exercise level and heart health
df['heart_health'] = -0.5 * df['age'] + 2 * df['exercise'] + np.random.normal(0, 1, n)

# Naive regression without controlling for age
model1 = LinearRegression().fit(df[['exercise']], df['heart_health'])

# Regression controlling for age
model2 = LinearRegression().fit(df[['exercise', 'age']], df['heart_health'])

print("Exercise coefficient without controlling for age:", model1.coef_[0])
print("Exercise coefficient controlling for age:", model2.coef_[0])
```

Slide 6: Correlated Variables

Correlated variables are those that have a statistical relationship with each other. High correlation between independent variables can lead to multicollinearity, which can cause issues in some statistical analyses.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create dataset with correlated variables
n = 1000
df = pd.DataFrame({
    'height': np.random.normal(170, 10, n),
    'weight': np.random.normal(70, 15, n)
})

# Introduce correlation
df['weight'] = df['weight'] + 0.6 * (df['height'] - 170)

# Visualize correlation
plt.figure(figsize=(8, 6))
sns.scatterplot(x='height', y='weight', data=df)
plt.title('Correlation between Height and Weight')
plt.show()

# Calculate correlation coefficient
correlation = df['height'].corr(df['weight'])
print(f"Correlation coefficient: {correlation:.2f}")
```

Slide 7: Control Variables

Control variables are factors that are kept constant during an experiment or analysis to isolate the effect of the independent variable on the dependent variable. They help reduce the influence of extraneous factors on the results.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Create dataset with a control variable
n = 1000
df = pd.DataFrame({
    'study_time': np.random.uniform(1, 5, n),
    'intelligence': np.random.normal(100, 15, n),
    'test_score': np.zeros(n)
})

# Generate test scores
df['test_score'] = 10 + 5 * df['study_time'] + 0.5 * df['intelligence'] + np.random.normal(0, 5, n)

# Regression without controlling for intelligence
model1 = LinearRegression().fit(df[['study_time']], df['test_score'])

# Regression controlling for intelligence
model2 = LinearRegression().fit(df[['study_time', 'intelligence']], df['test_score'])

print("Study time coefficient without control:", model1.coef_[0])
print("Study time coefficient with control:", model2.coef_[0])
```

Slide 8: Leaky Variables

Leaky variables are features that inadvertently include information about the target variable, leading to overly optimistic model performance. They often result from data preprocessing errors or including future information in historical data.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Create dataset with a leaky variable
n = 1000
df = pd.DataFrame({
    'feature1': np.random.normal(0, 1, n),
    'feature2': np.random.normal(0, 1, n),
    'target': np.random.choice([0, 1], n)
})

# Create a leaky variable
df['leaky'] = df['target'] + np.random.normal(0, 0.1, n)

# Split data and train models
X_safe = df[['feature1', 'feature2']]
X_leaky = df[['feature1', 'feature2', 'leaky']]
y = df['target']

X_safe_train, X_safe_test, X_leaky_train, X_leaky_test, y_train, y_test = train_test_split(
    X_safe, X_leaky, y, test_size=0.2, random_state=42)

model_safe = LogisticRegression().fit(X_safe_train, y_train)
model_leaky = LogisticRegression().fit(X_leaky_train, y_train)

print("Safe model accuracy:", model_safe.score(X_safe_test, y_test))
print("Leaky model accuracy:", model_leaky.score(X_leaky_test, y_test))
```

Slide 9: Stationary Variables

Stationary variables have statistical properties that do not change over time. In time series analysis, stationarity is often a key assumption for many models. A stationary series has constant mean, variance, and autocorrelation structure over time.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Generate stationary time series
np.random.seed(42)
stationary_series = np.random.normal(0, 1, 1000)

# Plot the series
plt.figure(figsize=(10, 5))
plt.plot(stationary_series)
plt.title('Stationary Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Perform Augmented Dickey-Fuller test
result = adfuller(stationary_series)
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```

Slide 10: Non-Stationary Variables

Non-stationary variables have statistical properties that change over time. Common forms of non-stationarity include trends, seasonality, and changing variance. Many real-world time series are non-stationary and require transformation or differencing to achieve stationarity.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Generate non-stationary time series with trend and seasonality
t = np.arange(1000)
trend = 0.1 * t
seasonality = 10 * np.sin(2 * np.pi * t / 365)
noise = np.random.normal(0, 1, 1000)
non_stationary_series = trend + seasonality + noise

# Plot the series
plt.figure(figsize=(10, 5))
plt.plot(non_stationary_series)
plt.title('Non-Stationary Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Perform Augmented Dickey-Fuller test
result = adfuller(non_stationary_series)
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```

Slide 11: Lagged Variables

Lagged variables represent past values of a variable. They are commonly used in time series analysis and forecasting to capture temporal dependencies. Lagged variables can help predict future values based on historical patterns.

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

# Create a time series
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
series = pd.Series(np.cumsum(np.random.randn(100)), index=dates)

# Create lagged variables
df = pd.DataFrame({'y': series})
for i in range(1, 4):
    df[f'lag_{i}'] = df['y'].shift(i)

# Remove NaN values
df = df.dropna()

# Fit autoregressive model
model = AutoReg(df['y'], lags=3)
results = model.fit()

print(results.summary())

# Forecast next 5 values
forecast = results.forecast(steps=5)
print("\nForecast for next 5 periods:")
print(forecast)
```

Slide 12: Additional Resources

For further exploration of variables in datasets and their applications in data science and machine learning, consider the following resources:

1. "A Survey on Feature Selection Methods" by Jundong Li et al. (2018) arXiv: [https://arxiv.org/abs/1904.00970](https://arxiv.org/abs/1904.00970)
2. "Causal Inference in Statistics: A Primer" by Judea Pearl et al. (2016) Book reference: Pearl, J., Glymour, M., & Jewell, N. P. (2016). Causal inference in statistics: A primer. John Wiley & Sons.
3. "Time Series Analysis: Forecasting and Control" by George E. P. Box et al. (2015) Book reference: Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control. John Wiley & Sons.

