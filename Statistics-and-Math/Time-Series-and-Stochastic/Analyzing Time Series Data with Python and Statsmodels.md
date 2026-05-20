## Analyzing Time Series Data with Python and Statsmodels
Slide 1: Introduction to Time Series Analysis with Statsmodels

Time series analysis involves studying data points collected over time to identify patterns, trends, and relationships. Statsmodels provides comprehensive tools for time series analysis in Python, offering methods for decomposition, statistical testing, and forecasting with minimal setup required.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from datetime import datetime

# Generate sample time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.random.normal(loc=100, scale=10, size=len(dates))
values = values + np.linspace(0, 50, len(dates))  # Adding trend

# Create DataFrame
ts_data = pd.DataFrame({'value': values}, index=dates)

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data['value'])
plt.title('Sample Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()
```

Slide 2: Time Series Components and Decomposition

Understanding the fundamental components of time series data is crucial for analysis. These components include trend, seasonality, and residuals. Statsmodels provides tools to decompose time series data into these constituent parts using various methods.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Add seasonal component to our data
seasonal_pattern = 10 * np.sin(2 * np.pi * np.arange(len(dates))/365.25)
ts_data['value'] = ts_data['value'] + seasonal_pattern

# Perform decomposition
decomposition = seasonal_decompose(ts_data['value'], period=365)

# Plot decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=ax1)
ax1.set_title('Original Time Series')
decomposition.trend.plot(ax=ax2)
ax2.set_title('Trend')
decomposition.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
decomposition.resid.plot(ax=ax4)
ax4.set_title('Residuals')
plt.tight_layout()
plt.show()
```

Slide 3: Testing for Stationarity

A crucial concept in time series analysis is stationarity, where statistical properties remain constant over time. The Augmented Dickey-Fuller test helps determine if a time series is stationary, which is essential for many forecasting methods.

```python
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    # Perform Augmented Dickey-Fuller test
    result = adfuller(timeseries)
    
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

# Test original series
print("Testing original series:")
test_stationarity(ts_data['value'])

# Test differenced series
print("\nTesting differenced series:")
test_stationarity(ts_data['value'].diff().dropna())
```

Slide 4: Granger Causality Analysis

Granger causality determines whether one time series can predict another. This statistical concept helps identify relationships between different variables and their temporal dependencies, which is valuable for multivariate time series analysis.

```python
from statsmodels.tsa.stattools import grangercausalitytests

# Create a second time series with lagged relationship
ts_data['value2'] = ts_data['value'].shift(30) + np.random.normal(0, 5, len(ts_data))
ts_data = ts_data.dropna()

# Prepare data for Granger test
data = np.column_stack([ts_data['value'], ts_data['value2']])

# Perform Granger Causality test
max_lag = 5
test_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)

# Display results
for lag in range(1, max_lag + 1):
    print(f"\nLag {lag} test results:")
    print(f"F-test p-value: {test_result[lag][0]['ssr_chi2test'][1]:.4f}")
```

Slide 5: Exponential Smoothing Techniques

Exponential smoothing is a time series forecasting method that gives more weight to recent observations and less weight to older ones. Simple exponential smoothing is particularly effective for data without clear trends or seasonal patterns.

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit simple exponential smoothing
model = SimpleExpSmoothing(ts_data['value'])
fit_model = model.fit(smoothing_level=0.2, optimized=False)

# Make predictions
forecast = fit_model.forecast(30)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data['value'], label='Original')
plt.plot(fit_model.fittedvalues.index, fit_model.fittedvalues, 
         label='Fitted', color='red')
plt.plot(forecast.index, forecast, label='Forecast', color='green')
plt.title('Simple Exponential Smoothing')
plt.legend()
plt.show()
```

Slide 6: Holt-Winters Method Implementation

The Holt-Winters method extends exponential smoothing to handle both trend and seasonality in time series data. This triple exponential smoothing approach is particularly useful for data with clear seasonal patterns.

```python
# Implement Holt-Winters seasonal method
hw_model = ExponentialSmoothing(ts_data['value'],
                               seasonal_periods=365,
                               trend='add',
                               seasonal='add')
fitted_hw = hw_model.fit()

# Generate forecasts
hw_forecast = fitted_hw.forecast(steps=60)

# Calculate error metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(ts_data['value'], fitted_hw.fittedvalues)
mae = mean_absolute_error(ts_data['value'], fitted_hw.fittedvalues)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data['value'], label='Actual')
plt.plot(fitted_hw.fittedvalues.index, fitted_hw.fittedvalues, 
         label='Fitted', color='red')
plt.plot(hw_forecast.index, hw_forecast, label='Forecast', color='green')
plt.title(f'Holt-Winters Method (MSE: {mse:.2f}, MAE: {mae:.2f})')
plt.legend()
plt.show()
```

Slide 7: ARIMA Model Implementation

ARIMA (AutoRegressive Integrated Moving Average) models are sophisticated time series forecasting tools that capture various aspects of time series behavior including autoregression, differencing, and moving average components.

```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Fit ARIMA model
arima_model = ARIMA(ts_data['value'], order=(1, 1, 1))
arima_results = arima_model.fit()

# Make predictions
predictions = arima_results.predict(start=len(ts_data)-30, 
                                  end=len(ts_data)+30)

# Calculate error metrics
mse = mean_squared_error(ts_data['value'][-30:], predictions[:30])
rmse = np.sqrt(mse)

print(f'Model Summary:\n{arima_results.summary().tables[1]}')
print(f'\nRoot Mean Squared Error: {rmse:.2f}')

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data['value'], label='Actual')
plt.plot(predictions.index, predictions, label='ARIMA Predictions', 
         color='red')
plt.title('ARIMA Model Predictions')
plt.legend()
plt.show()
```

Slide 8: Advanced Time Series Decomposition

Seasonal-Trend decomposition using LOESS (STL) provides a more robust approach to decomposing time series data, handling non-linear trends and complex seasonal patterns better than classical decomposition methods.

```python
from statsmodels.tsa.seasonal import STL

# Perform STL decomposition
stl = STL(ts_data['value'], period=365)
result = stl.fit()

# Extract components
trend = result.trend
seasonal = result.seasonal
resid = result.resid

# Calculate strength of trend and seasonality
F_t = max(0, 1 - np.var(resid) / np.var(trend + resid))
F_s = max(0, 1 - np.var(resid) / np.var(seasonal + resid))

# Plot components
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
ax1.plot(ts_data.index, ts_data['value'])
ax1.set_title('Original Time Series')
ax2.plot(ts_data.index, trend)
ax2.set_title(f'Trend (Strength: {F_t:.3f})')
ax3.plot(ts_data.index, seasonal)
ax3.set_title(f'Seasonal (Strength: {F_s:.3f})')
ax4.plot(ts_data.index, resid)
ax4.set_title('Residuals')
plt.tight_layout()
plt.show()
```

Slide 9: Autocorrelation Analysis

Autocorrelation analysis helps identify patterns and dependencies in time series data by measuring the correlation between observations at different time lags. This is crucial for understanding cyclic patterns and selecting appropriate model parameters.

```python
from statsmodels.stats.diagnostic import acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Calculate and plot ACF and PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot ACF
plot_acf(ts_data['value'], lags=40, ax=ax1)
ax1.set_title('Autocorrelation Function')

# Plot PACF
plot_pacf(ts_data['value'], lags=40, ax=ax2)
ax2.set_title('Partial Autocorrelation Function')

# Add confidence intervals calculation
acf_values = acf(ts_data['value'], nlags=40)
confidence_interval = 1.96/np.sqrt(len(ts_data))

plt.tight_layout()
plt.show()

print(f"95% Confidence Interval: ±{confidence_interval:.3f}")
```

Slide 10: Rolling Statistics Analysis

Rolling statistics provide insights into the evolving characteristics of time series data. This analysis helps identify changes in statistical properties over time and assess stationarity visually.

```python
# Calculate rolling statistics
window_size = 30
rolling_mean = ts_data['value'].rolling(window=window_size).mean()
rolling_std = ts_data['value'].rolling(window=window_size).std()
rolling_skew = ts_data['value'].rolling(window=window_size).skew()

# Plot rolling statistics
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Plot original data with rolling mean
ax1.plot(ts_data.index, ts_data['value'], label='Original')
ax1.plot(rolling_mean.index, rolling_mean, label=f'{window_size}-day Rolling Mean')
ax1.set_title('Original Data vs Rolling Mean')
ax1.legend()

# Plot rolling standard deviation
ax2.plot(rolling_std.index, rolling_std, color='orange')
ax2.set_title(f'{window_size}-day Rolling Standard Deviation')

# Plot rolling skewness
ax3.plot(rolling_skew.index, rolling_skew, color='green')
ax3.set_title(f'{window_size}-day Rolling Skewness')

plt.tight_layout()
plt.show()
```

Slide 11: Prophet Model Implementation

Facebook's Prophet model is designed to handle time series data with strong seasonal patterns and missing values. It automatically detects changepoints and incorporates holiday effects.

```python
from fbprophet import Prophet
import pandas as pd

# Prepare data for Prophet
prophet_data = ts_data.reset_index()
prophet_data.columns = ['ds', 'y']

# Create and fit Prophet model
prophet = Prophet(yearly_seasonality=True,
                 weekly_seasonality=True,
                 daily_seasonality=False,
                 changepoint_prior_scale=0.05)
prophet.fit(prophet_data)

# Make future predictions
future_dates = prophet.make_future_dataframe(periods=60)
forecast = prophet.predict(future_dates)

# Plot results
fig = prophet.plot(forecast)
plt.title('Prophet Model Forecast')

# Plot components
fig2 = prophet.plot_components(forecast)
plt.show()

# Print performance metrics
from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(prophet_data['y'], 
                                    forecast['yhat'][:len(prophet_data)])
print(f'Mean Absolute Percentage Error: {mape:.2%}')
```

Slide 12: Advanced Feature Engineering for Time Series

Feature engineering in time series analysis involves creating meaningful temporal features that can improve model performance. This includes lag features, rolling statistics, and time-based features.

```python
def create_time_features(df):
    # Create copy of dataframe
    df = df.copy()
    
    # Extract datetime features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    
    # Create lag features
    df['lag1'] = df['value'].shift(1)
    df['lag7'] = df['value'].shift(7)
    
    # Create rolling features
    df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
    df['rolling_std_7'] = df['value'].rolling(window=7).std()
    
    # Create difference features
    df['diff1'] = df['value'].diff(1)
    df['diff7'] = df['value'].diff(7)
    
    return df

# Apply feature engineering
engineered_data = create_time_features(ts_data)
print("Features created:")
print(engineered_data.columns.tolist())

# Display correlation matrix
correlation_matrix = engineered_data.corr()
plt.figure(figsize=(12, 8))
plt.imshow(correlation_matrix, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), 
           correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), 
           correlation_matrix.columns)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
```

Slide 13: Time Series Cross-Validation

Time series cross-validation differs from traditional cross-validation as it respects the temporal order of observations. This implementation demonstrates how to properly evaluate time series models using rolling forecast origin evaluation.

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def time_series_cv(data, n_splits=5):
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Create arrays to store results
    train_scores = []
    test_scores = []
    
    # Perform rolling window cross-validation
    for train_idx, test_idx in tscv.split(data):
        # Split data
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        
        # Fit model (using ARIMA as example)
        model = ARIMA(train['value'], order=(1,1,1))
        fitted_model = model.fit()
        
        # Make predictions
        predictions = fitted_model.forecast(steps=len(test))
        
        # Calculate error metrics
        train_rmse = np.sqrt(mean_squared_error(
            train['value'], fitted_model.fittedvalues))
        test_rmse = np.sqrt(mean_squared_error(
            test['value'], predictions))
        
        train_scores.append(train_rmse)
        test_scores.append(test_rmse)
    
    return train_scores, test_scores

# Perform cross-validation
train_scores, test_scores = time_series_cv(ts_data)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_scores) + 1), train_scores, 
         label='Train RMSE', marker='o')
plt.plot(range(1, len(test_scores) + 1), test_scores, 
         label='Test RMSE', marker='o')
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.title('Time Series Cross-Validation Results')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 14: Spectral Analysis and Periodogram

Spectral analysis helps identify cyclical patterns and periodicities in time series data by decomposing the signal into its frequency components. This is particularly useful for detecting hidden periodic patterns.

```python
from scipy import signal
import numpy as np

def perform_spectral_analysis(data):
    # Calculate periodogram
    frequencies, spectrum = signal.periodogram(
        data['value'].values,
        fs=1.0,  # 1 sample per day
        window='hann',
        detrend='linear'
    )
    
    # Convert frequencies to periods
    periods = 1/frequencies[1:]  # Skip first element (0 frequency)
    power = spectrum[1:]
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot periodogram
    plt.subplot(2, 1, 1)
    plt.plot(frequencies[1:], spectrum[1:])
    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density')
    plt.title('Periodogram')
    plt.grid(True)
    
    # Plot period spectrum
    plt.subplot(2, 1, 2)
    plt.plot(periods, power)
    plt.xlabel('Period (days)')
    plt.ylabel('Power Spectral Density')
    plt.title('Period Spectrum')
    plt.grid(True)
    
    # Find top periods
    top_k = 5
    top_periods_idx = np.argsort(power)[-top_k:]
    top_periods = periods[top_periods_idx]
    
    plt.tight_layout()
    plt.show()
    
    return top_periods

# Perform spectral analysis
top_periods = perform_spectral_analysis(ts_data)
print("\nTop periods detected (in days):")
for i, period in enumerate(sorted(top_periods), 1):
    print(f"{i}. {period:.2f}")
```

Slide 15: Additional Resources

*   Research Papers and Documentation:
    *   "Time Series Analysis Using Python" - [https://arxiv.org/abs/2104.05937](https://arxiv.org/abs/2104.05937)
    *   "Modern Time Series Forecasting with Python" - [https://arxiv.org/abs/2203.12138](https://arxiv.org/abs/2203.12138)
    *   "Deep Learning for Time Series Forecasting" - [https://arxiv.org/abs/1901.00596](https://arxiv.org/abs/1901.00596)
*   Useful Resources:
    *   statsmodels documentation: [https://www.statsmodels.org/stable/tsa.html](https://www.statsmodels.org/stable/tsa.html)
    *   Python Time Series Analysis: [https://www.python.org/doc/time-series](https://www.python.org/doc/time-series)
    *   Time Series with Python Tutorial: [https://machinelearningmastery.com/time-series-forecasting-python-mini-course/](https://machinelearningmastery.com/time-series-forecasting-python-mini-course/)
*   Advanced Topics:
    *   Neural Prophet documentation: [https://neuralprophet.com/](https://neuralprophet.com/)
    *   Time Series Feature Extraction: [https://tsfresh.readthedocs.io/](https://tsfresh.readthedocs.io/)
    *   Statistical Forecasting: [https://otexts.com/fpp3/](https://otexts.com/fpp3/)

