## Time Series Regression Models in Python
Slide 1: Time Series Regression in Python

Time series regression is a statistical method used to analyze and predict time-dependent data. In Python, various regression techniques can be applied to time series data, each with its own strengths and use cases. This presentation will explore different regression methods for time series analysis, providing code examples and practical applications.

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load sample time series data
data = pd.read_csv('time_series_data.csv', parse_dates=['date'], index_col='date')

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()
```

Slide 2: Linear Regression for Time Series

Linear regression is a simple yet effective method for modeling trends in time series data. It assumes a linear relationship between the dependent variable and time.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare data
X = np.array(range(len(data))).reshape(-1, 1)
y = data.values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, predictions, color='red', label='Predicted')
plt.title('Linear Regression for Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
```

Slide 3: Autoregressive (AR) Model

The Autoregressive model predicts future values based on past values. It's useful for time series with a strong dependency on recent observations.

```python
from statsmodels.tsa.ar_model import AutoReg

# Fit AR model
model = AutoReg(data, lags=5)
results = model.fit()

# Make predictions
predictions = results.predict(start=len(data), end=len(data)+10)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual')
plt.plot(predictions, color='red', label='Predicted')
plt.title('Autoregressive (AR) Model')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

print(results.summary())
```

Slide 4: Moving Average (MA) Model

The Moving Average model uses past forecast errors in a regression-like model. It's suitable for time series with short-term fluctuations.

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit MA model (ARIMA with p=0, d=0)
model = ARIMA(data, order=(0, 0, 1))
results = model.fit()

# Make predictions
predictions = results.forecast(steps=10)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual')
plt.plot(predictions, color='red', label='Predicted')
plt.title('Moving Average (MA) Model')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

print(results.summary())
```

Slide 5: ARIMA Model

ARIMA (AutoRegressive Integrated Moving Average) combines AR and MA models with differencing to handle non-stationary data. It's versatile for various time series patterns.

```python
from pmdarima import auto_arima

# Find optimal ARIMA order
model = auto_arima(data, start_p=0, start_q=0, max_p=5, max_q=5, seasonal=False)

# Fit ARIMA model
results = model.fit(data)

# Make predictions
predictions = results.predict(n_periods=10)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual')
plt.plot(pd.date_range(start=data.index[-1], periods=11, freq='D')[1:], predictions, color='red', label='Predicted')
plt.title('ARIMA Model')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

print(results.summary())
```

Slide 6: Seasonal ARIMA (SARIMA) Model

SARIMA extends ARIMA to include seasonal components, making it ideal for time series with recurring patterns at fixed intervals.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit SARIMA model
model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Make predictions
predictions = results.forecast(steps=24)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual')
plt.plot(predictions, color='red', label='Predicted')
plt.title('Seasonal ARIMA (SARIMA) Model')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

print(results.summary())
```

Slide 7: Prophet Model

Facebook's Prophet is designed for forecasting time series data with strong seasonal effects and multiple seasonalities. It's robust to missing data and shifts in trends.

```python
from fbprophet import Prophet

# Prepare data for Prophet
df = data.reset_index().rename(columns={'date': 'ds', 'value': 'y'})

# Fit Prophet model
model = Prophet()
model.fit(df)

# Make predictions
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Plot results
fig = model.plot(forecast)
plt.title('Prophet Model Forecast')
plt.show()

# Plot components
fig = model.plot_components(forecast)
plt.show()
```

Slide 8: LSTM Neural Networks

Long Short-Term Memory (LSTM) networks are powerful for capturing long-term dependencies in time series data. They're particularly useful for complex, non-linear patterns.

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Prepare data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_sequences(scaled_data, seq_length)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build and train LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, batch_size=32)

# Make predictions
last_sequence = scaled_data[-seq_length:]
next_prediction = model.predict(np.array([last_sequence]))
next_prediction = scaler.inverse_transform(next_prediction)

print(f"Next predicted value: {next_prediction[0][0]}")
```

Slide 9: Real-Life Example: Weather Forecasting

Weather forecasting is a common application of time series regression. Let's use temperature data to predict future temperatures.

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Load temperature data (assuming daily temperatures for a year)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
temperatures = np.random.normal(loc=20, scale=5, size=365) + 5 * np.sin(np.arange(365) * 2 * np.pi / 365)
data = pd.Series(temperatures, index=dates)

# Fit SARIMA model
model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Forecast next 30 days
forecast = results.forecast(steps=30)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data, label='Historical Temperatures')
plt.plot(forecast, color='red', label='Forecasted Temperatures')
plt.title('Temperature Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

print(forecast)
```

Slide 10: Real-Life Example: Energy Consumption Prediction

Predicting energy consumption is crucial for efficient resource management. Let's use historical energy consumption data to forecast future usage.

```python
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt

# Generate sample energy consumption data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
consumption = np.random.normal(loc=1000, scale=100, size=365) + 200 * np.sin(np.arange(365) * 2 * np.pi / 365)
df = pd.DataFrame({'ds': dates, 'y': consumption})

# Fit Prophet model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model.fit(df)

# Make future predictions
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# Plot results
fig = model.plot(forecast)
plt.title('Energy Consumption Forecast')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (kWh)')
plt.show()

# Plot components
fig = model.plot_components(forecast)
plt.show()
```

Slide 11: Choosing the Right Regression Model

Selecting the appropriate regression model depends on various factors:

1. Data characteristics (trend, seasonality, stationarity)
2. Forecast horizon (short-term vs. long-term)
3. Model complexity and interpretability
4. Computational resources
5. Amount of historical data available

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load sample data
data = pd.read_csv('time_series_data.csv', parse_dates=['date'], index_col='date')

# Perform seasonal decomposition
result = seasonal_decompose(data, model='additive')

# Plot decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
result.observed.plot(ax=ax1)
ax1.set_title('Observed')
result.trend.plot(ax=ax2)
ax2.set_title('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()
```

Slide 12: Model Evaluation and Diagnostics

Evaluating the performance of time series regression models is crucial for ensuring accurate predictions. Common metrics and diagnostic tools include:

1. Mean Absolute Error (MAE)
2. Root Mean Square Error (RMSE)
3. Mean Absolute Percentage Error (MAPE)
4. Autocorrelation Function (ACF) plot
5. Partial Autocorrelation Function (PACF) plot
6. Residual analysis

```python
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Assuming 'results' is a fitted time series model
residuals = results.resid

# Calculate performance metrics
mae = np.mean(np.abs(residuals))
rmse = np.sqrt(np.mean(residuals**2))
mape = np.mean(np.abs(residuals / data)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Plot ACF and PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plot_acf(residuals, ax=ax1, lags=40)
ax1.set_title('Autocorrelation Function (ACF)')
plot_pacf(residuals, ax=ax2, lags=40)
ax2.set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()
```

Slide 13: Handling Non-Stationary Data

Many time series regression techniques assume stationary data. When dealing with non-stationary series, consider:

1. Differencing
2. Detrending
3. Seasonal adjustment
4. Transformation (e.g., log transformation)

```python
from statsmodels.tsa.stattools import adfuller

# Perform Augmented Dickey-Fuller test
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# Test original series
print("Original Series:")
adf_test(data)

# Difference series
diff_data = data.diff().dropna()

# Test differenced series
print("\nDifferenced Series:")
adf_test(diff_data)

# Plot original and differenced series
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
data.plot(ax=ax1)
ax1.set_title('Original Series')
diff_data.plot(ax=ax2)
ax2.set_title('Differenced Series')
plt.tight_layout()
plt.show()
```

Slide 14: Advanced Techniques and Future Directions

As time series analysis evolves, new techniques and hybrid models emerge:

1. Ensemble methods combining multiple models
2. Deep learning approaches (e.g., Temporal Convolutional Networks)
3. Bayesian methods for uncertainty quantification
4. Multivariate time series analysis
5. Transfer learning for time series forecasting

Slide 15: Advanced Techniques and Future Directions

As time series analysis evolves, new techniques and hybrid models emerge:

1. Ensemble methods combining multiple models
2. Deep learning approaches (e.g., Temporal Convolutional Networks)
3. Bayesian methods for uncertainty quantification
4. Multivariate time series analysis
5. Transfer learning for time series forecasting

Slide 16: Advanced Techniques and Future Directions

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Generate synthetic time series data
np.random.seed(42)
t = np.arange(0, 1000)
y = 50 + 0.5 * t + 10 * np.sin(0.1 * t) + 5 * np.random.randn(1000)

# Prepare data for Random Forest
X = t.reshape(-1, 1)
y = y.reshape(-1, 1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train.ravel())

# Make predictions
y_pred = rf_model.predict(X_test)

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual')
plt.scatter(X_test, y_pred, color='red', alpha=0.5, label='Predicted')
plt.title('Random Forest Time Series Regression')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Calculate and print performance metrics
mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Square Error: {rmse:.2f}")
```

Slide 17: Conclusion and Best Practices

When working with time series regression in Python:

1. Understand your data's characteristics (trend, seasonality, stationarity)
2. Preprocess data appropriately (handling missing values, outliers)
3. Experiment with multiple models and compare their performance
4. Regularly update and retrain models with new data
5. Consider domain expertise when interpreting results
6. Be aware of potential limitations and uncertainties in predictions

Slide 18: Conclusion and Best Practices

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate sample time series data
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
values = pd.Series(np.random.randn(365).cumsum(), index=dates)

# Perform seasonal decomposition
result = seasonal_decompose(values, model='additive', period=30)

# Plot the decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
result.observed.plot(ax=ax1)
ax1.set_title('Observed')
result.trend.plot(ax=ax2)
ax2.set_title('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()
```

Slide 19: Additional Resources

For further exploration of time series regression techniques in Python:

1. "Forecasting: Principles and Practice" by Rob J Hyndman and George Athanasopoulos Available online: [https://otexts.com/fpp3/](https://otexts.com/fpp3/)
2. "Time Series Analysis and Its Applications" by Robert H. Shumway and David S. Stoffer ArXiv link: [https://arxiv.org/abs/1707.07799](https://arxiv.org/abs/1707.07799)
3. StatsModels documentation for time series analysis: [https://www.statsmodels.org/stable/tsa.html](https://www.statsmodels.org/stable/tsa.html)
4. Prophet documentation and tutorials: [https://facebook.github.io/prophet/](https://facebook.github.io/prophet/)
5. Scikit-learn time series documentation: [https://scikit-learn.org/stable/modules/time\_series.html](https://scikit-learn.org/stable/modules/time_series.html)

