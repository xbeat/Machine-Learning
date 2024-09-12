## PACF Plots for Mixed ARMA Models in Python
Slide 1: Introduction to PACF Plots for Mixed ARMA Models

Partial Autocorrelation Function (PACF) plots are crucial tools in time series analysis, particularly for identifying the order of autoregressive (AR) components in mixed Autoregressive Moving Average (ARMA) models. These plots help analysts understand the direct relationship between an observation and its lags, without the influence of intermediate lags.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_pacf

# Generate an ARMA(2,1) process
ar = np.array([1, -0.6, 0.2])
ma = np.array([1, 0.3])
arma_process = ArmaProcess(ar, ma)
y = arma_process.generate_sample(nsample=1000)

# Plot PACF
fig, ax = plt.subplots(figsize=(10, 6))
plot_pacf(y, ax=ax, lags=20)
plt.title("PACF Plot of ARMA(2,1) Process")
plt.show()
```

Slide 2: Understanding PACF

The Partial Autocorrelation Function measures the correlation between an observation and its lag, while controlling for the effects of intermediate lags. In ARMA models, PACF helps identify the order of the autoregressive (AR) component by showing significant correlations at specific lags.

```python
def pacf_manual(y, nlags):
    pacf_values = []
    for lag in range(1, nlags + 1):
        y_lagged = np.roll(y, lag)
        y_lagged[:lag] = np.nan
        
        X = np.column_stack([y_lagged] + [np.roll(y, i)[:lag] for i in range(1, lag)])
        X = X[lag:, :]
        y_subset = y[lag:]
        
        beta = np.linalg.lstsq(X, y_subset, rcond=None)[0]
        pacf_values.append(beta[0])
    
    return np.array(pacf_values)

# Calculate and plot manual PACF
manual_pacf = pacf_manual(y, 20)
plt.figure(figsize=(10, 6))
plt.bar(range(1, 21), manual_pacf)
plt.title("Manual PACF Calculation")
plt.xlabel("Lag")
plt.ylabel("PACF")
plt.show()
```

Slide 3: PACF vs ACF

While Autocorrelation Function (ACF) shows the overall correlation structure of a time series, PACF focuses on the direct effects of each lag. This distinction is crucial for identifying the order of AR and MA components in ARMA models.

```python
from statsmodels.graphics.tsaplots import plot_acf

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

plot_acf(y, ax=ax1, lags=20)
ax1.set_title("ACF Plot")

plot_pacf(y, ax=ax2, lags=20)
ax2.set_title("PACF Plot")

plt.tight_layout()
plt.show()
```

Slide 4: Interpreting PACF Plots

In PACF plots, significant spikes indicate the order of the AR process. For pure AR(p) processes, PACF shows significant spikes up to lag p and cuts off after that. For mixed ARMA models, the interpretation becomes more complex, but PACF still provides valuable insights into the AR structure.

```python
def interpret_pacf(pacf_values, confidence=0.95):
    n = len(pacf_values)
    se = np.sqrt(1/n)  # Standard error
    threshold = se * stats.norm.ppf((1 + confidence) / 2)
    
    significant_lags = np.where(np.abs(pacf_values) > threshold)[0] + 1
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n+1), pacf_values)
    plt.axhline(y=threshold, color='r', linestyle='--')
    plt.axhline(y=-threshold, color='r', linestyle='--')
    plt.title("PACF with Significance Threshold")
    plt.xlabel("Lag")
    plt.ylabel("PACF")
    
    for lag in significant_lags:
        plt.annotate(f"Lag {lag}", (lag, pacf_values[lag-1]))
    
    plt.show()
    
    return significant_lags

significant_lags = interpret_pacf(manual_pacf)
print(f"Significant lags: {significant_lags}")
```

Slide 5: PACF for AR(p) Models

For pure AR(p) models, PACF plots show clear cutoffs after lag p. This property makes PACF particularly useful for identifying the order of AR processes.

```python
# Generate AR(2) process
ar_params = np.array([1, -0.6, 0.3])
ar_process = ArmaProcess(ar_params, ma=np.array([1]))
ar_data = ar_process.generate_sample(nsample=1000)

fig, ax = plt.subplots(figsize=(10, 6))
plot_pacf(ar_data, ax=ax, lags=20)
plt.title("PACF Plot of AR(2) Process")
plt.show()

# Interpret PACF
ar_pacf = pacf_manual(ar_data, 20)
significant_lags_ar = interpret_pacf(ar_pacf)
print(f"Significant lags for AR(2): {significant_lags_ar}")
```

Slide 6: PACF for MA(q) Models

For pure MA(q) processes, PACF doesn't show a clear cutoff but rather a decay pattern. This behavior distinguishes MA processes from AR processes in PACF plots.

```python
# Generate MA(2) process
ma_params = np.array([1, 0.6, 0.3])
ma_process = ArmaProcess(ar=np.array([1]), ma=ma_params)
ma_data = ma_process.generate_sample(nsample=1000)

fig, ax = plt.subplots(figsize=(10, 6))
plot_pacf(ma_data, ax=ax, lags=20)
plt.title("PACF Plot of MA(2) Process")
plt.show()

# Interpret PACF
ma_pacf = pacf_manual(ma_data, 20)
significant_lags_ma = interpret_pacf(ma_pacf)
print(f"Significant lags for MA(2): {significant_lags_ma}")
```

Slide 7: PACF for Mixed ARMA(p,q) Models

In mixed ARMA(p,q) models, PACF plots show a combination of AR and MA characteristics. The interpretation becomes more challenging, often requiring additional analysis techniques.

```python
# Generate ARMA(2,1) process
arma_ar = np.array([1, -0.6, 0.2])
arma_ma = np.array([1, 0.3])
arma_process = ArmaProcess(arma_ar, arma_ma)
arma_data = arma_process.generate_sample(nsample=1000)

fig, ax = plt.subplots(figsize=(10, 6))
plot_pacf(arma_data, ax=ax, lags=20)
plt.title("PACF Plot of ARMA(2,1) Process")
plt.show()

# Interpret PACF
arma_pacf = pacf_manual(arma_data, 20)
significant_lags_arma = interpret_pacf(arma_pacf)
print(f"Significant lags for ARMA(2,1): {significant_lags_arma}")
```

Slide 8: PACF and Model Selection

PACF plots play a crucial role in model selection for ARMA processes. By examining the PACF, analysts can make informed decisions about the appropriate order of the AR component in the model.

```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def select_arima_order(data, max_p, max_q):
    best_aic = np.inf
    best_order = None
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(data, order=(p, 0, q))
                results = model.fit()
                aic = results.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, 0, q)
            except:
                continue
    
    return best_order

best_order = select_arima_order(arma_data, max_p=5, max_q=5)
print(f"Best ARIMA order based on AIC: {best_order}")

# Fit the best model
best_model = ARIMA(arma_data, order=best_order)
best_results = best_model.fit()

# Plot residuals PACF
residuals = best_results.resid
fig, ax = plt.subplots(figsize=(10, 6))
plot_pacf(residuals, ax=ax, lags=20)
plt.title(f"PACF of Residuals - ARIMA{best_order}")
plt.show()
```

Slide 9: Real-life Example: Temperature Forecasting

Let's apply PACF analysis to a real-world scenario: forecasting daily temperatures. We'll use synthetic data to simulate temperature readings and demonstrate how PACF can help in model selection.

```python
import pandas as pd

# Generate synthetic temperature data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31')
temp_data = 20 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 2, len(dates))
temp_series = pd.Series(temp_data, index=dates)

# Plot the temperature series
plt.figure(figsize=(12, 6))
temp_series.plot()
plt.title("Daily Temperature Data")
plt.ylabel("Temperature (°C)")
plt.show()

# Plot PACF
fig, ax = plt.subplots(figsize=(10, 6))
plot_pacf(temp_series, ax=ax, lags=30)
plt.title("PACF of Daily Temperature Data")
plt.show()

# Select best ARIMA order
best_order = select_arima_order(temp_series, max_p=5, max_q=5)
print(f"Best ARIMA order for temperature data: {best_order}")

# Fit the model and forecast
model = ARIMA(temp_series, order=best_order)
results = model.fit()
forecast = results.forecast(steps=30)

# Plot forecast
plt.figure(figsize=(12, 6))
temp_series.plot(label='Observed')
forecast.plot(label='Forecast')
plt.title("Temperature Forecast")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()
```

Slide 10: Real-life Example: Traffic Flow Prediction

Another practical application of PACF analysis is in traffic flow prediction. We'll use synthetic data to simulate hourly traffic volume and demonstrate how PACF can aid in model selection for traffic forecasting.

```python
# Generate synthetic hourly traffic data
np.random.seed(42)
hours = pd.date_range(start='2023-01-01', end='2023-01-31 23:00:00', freq='H')
base_traffic = 100 + 50 * np.sin(np.arange(len(hours)) * 2 * np.pi / 24)
weekly_pattern = 20 * np.sin(np.arange(len(hours)) * 2 * np.pi / (24 * 7))
noise = np.random.normal(0, 10, len(hours))
traffic_data = base_traffic + weekly_pattern + noise
traffic_series = pd.Series(traffic_data, index=hours)

# Plot the traffic series
plt.figure(figsize=(12, 6))
traffic_series.plot()
plt.title("Hourly Traffic Volume")
plt.ylabel("Number of Vehicles")
plt.show()

# Plot PACF
fig, ax = plt.subplots(figsize=(10, 6))
plot_pacf(traffic_series, ax=ax, lags=48)
plt.title("PACF of Hourly Traffic Volume")
plt.show()

# Select best ARIMA order
best_order = select_arima_order(traffic_series, max_p=5, max_q=5)
print(f"Best ARIMA order for traffic data: {best_order}")

# Fit the model and forecast
model = ARIMA(traffic_series, order=best_order)
results = model.fit()
forecast = results.forecast(steps=24)

# Plot forecast
plt.figure(figsize=(12, 6))
traffic_series[-48:].plot(label='Observed')
forecast.plot(label='Forecast')
plt.title("Traffic Volume Forecast")
plt.ylabel("Number of Vehicles")
plt.legend()
plt.show()
```

Slide 11: Challenges in PACF Interpretation

While PACF is a powerful tool, its interpretation can be challenging, especially for complex ARMA models. Factors such as seasonality, non-stationarity, and model misspecification can affect PACF plots.

```python
# Generate a complex ARMA process with seasonality
np.random.seed(42)
n = 1000
t = np.arange(n)
trend = 0.01 * t
seasonal = 10 * np.sin(2 * np.pi * t / 365)
ar_component = np.random.normal(0, 1, n)
for i in range(2, n):
    ar_component[i] += 0.6 * ar_component[i-1] - 0.2 * ar_component[i-2]
ma_component = np.random.normal(0, 1, n)
for i in range(1, n):
    ma_component[i] += 0.3 * ma_component[i-1]

complex_series = trend + seasonal + ar_component + ma_component

# Plot the complex series
plt.figure(figsize=(12, 6))
plt.plot(complex_series)
plt.title("Complex Time Series with Trend, Seasonality, and ARMA Components")
plt.show()

# Plot PACF
fig, ax = plt.subplots(figsize=(10, 6))
plot_pacf(complex_series, ax=ax, lags=50)
plt.title("PACF of Complex Time Series")
plt.show()

# Interpret PACF
complex_pacf = pacf_manual(complex_series, 50)
significant_lags_complex = interpret_pacf(complex_pacf)
print(f"Significant lags for complex series: {significant_lags_complex}")
```

Slide 12: Advanced PACF Techniques

To address challenges in PACF interpretation, advanced techniques such as seasonal PACF, differencing, and spectral analysis can be employed. These methods help in handling complex time series structures.

```python
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Perform ADF test
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# Seasonal decomposition
decomposition = seasonal_decompose(complex_series, model='additive', period=365)

# Plot decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
decomposition.observed.plot(ax=ax1)
ax1.set_title('Observed')
decomposition.trend.plot(ax=ax2)
ax2.set_title('Trend')
decomposition.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
decomposition.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()

# Perform ADF test on the residuals
adf_test(decomposition.resid.dropna())

# Plot PACF of residuals
fig, ax = plt.subplots(figsize=(10, 6))
plot_pacf(decomposition.resid.dropna(), ax=ax, lags=50)
plt.title("PACF of Residuals after Seasonal Decomposition")
plt.show()
```

Slide 13: PACF in Non-stationary Time Series

Non-stationary time series can lead to misleading PACF plots. Differencing is a common technique to achieve stationarity before applying PACF analysis.

```python
# Generate a non-stationary series
np.random.seed(42)
n = 1000
t = np.arange(n)
non_stationary = 0.1 * t + np.cumsum(np.random.normal(0, 1, n))

# Plot the non-stationary series
plt.figure(figsize=(12, 6))
plt.plot(non_stationary)
plt.title("Non-stationary Time Series")
plt.show()

# Plot PACF of non-stationary series
fig, ax = plt.subplots(figsize=(10, 6))
plot_pacf(non_stationary, ax=ax, lags=50)
plt.title("PACF of Non-stationary Series")
plt.show()

# Difference the series
diff_series = np.diff(non_stationary)

# Plot the differenced series
plt.figure(figsize=(12, 6))
plt.plot(diff_series)
plt.title("Differenced Time Series")
plt.show()

# Plot PACF of differenced series
fig, ax = plt.subplots(figsize=(10, 6))
plot_pacf(diff_series, ax=ax, lags=50)
plt.title("PACF of Differenced Series")
plt.show()
```

Slide 14: Practical Tips for PACF Analysis

When using PACF for mixed ARMA model identification:

1. Always check for stationarity before interpreting PACF.
2. Consider both ACF and PACF plots for a comprehensive analysis.
3. Use information criteria (AIC, BIC) alongside PACF for model selection.
4. Be aware of potential seasonal patterns that may affect PACF interpretation.
5. Cross-validate your models to ensure robustness of the selected order.

```python
def pacf_analysis_workflow(series, max_lags=50):
    # Check stationarity
    adf_result = adfuller(series)
    is_stationary = adf_result[1] < 0.05
    
    if not is_stationary:
        print("Series is not stationary. Differencing...")
        series = np.diff(series)
    
    # Plot ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    plot_acf(series, ax=ax1, lags=max_lags)
    ax1.set_title("ACF")
    plot_pacf(series, ax=ax2, lags=max_lags)
    ax2.set_title("PACF")
    plt.tight_layout()
    plt.show()
    
    # Model selection using AIC
    best_order = select_arima_order(series, max_p=5, max_q=5)
    print(f"Best ARIMA order based on AIC: {best_order}")
    
    return best_order

# Example usage
best_order = pacf_analysis_workflow(complex_series)
```

Slide 15: Additional Resources

For those interested in diving deeper into PACF analysis and time series modeling, the following resources are recommended:

1. "Time Series Analysis: Forecasting and Control" by Box, Jenkins, Reinsel, and Ljung (2015)
2. "Practical Time Series Forecasting with R: A Hands-On Guide" by Shmueli and Lichtendahl (2016)
3. "Introduction to Time Series and Forecasting" by Brockwell and Davis (2016)
4. ArXiv paper: "A Review of Partial Autocorrelation Functions in Time Series Analysis" by S. Kang and S. Cho (2020). ArXiv:2007.07041 \[stat.ME\]

These resources provide in-depth explanations of PACF theory, implementation, and applications in various fields of study.

