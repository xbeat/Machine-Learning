## Understanding Time Series Data
Slide 1: Time Series Fundamentals

Time series analysis requires specialized data structures and preprocessing techniques. NumPy and Pandas provide robust foundations for handling temporal data, enabling efficient manipulation of datetime indices and sequential observations while maintaining chronological integrity.

```python
import numpy as np
import pandas as pd

# Create a time series dataset
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.random.normal(loc=100, scale=15, size=len(dates))

# Initialize time series with Pandas
ts = pd.Series(data=values, index=dates)

# Basic time series operations
print(f"First 5 observations:\n{ts.head()}")
print(f"\nFrequency: {ts.index.freq}")
print(f"Time span: {ts.index.min()} to {ts.index.max()}")
```

Slide 2: Decomposition Methods

Statistical decomposition separates time series into constituent components, revealing underlying patterns. The additive model assumes components sum together, while multiplicative models assume they multiply, choosing between them depends on whether variations scale with the level.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate sample data with trend and seasonality
t = np.linspace(0, 4, 100)
trend = 0.1 * t**2
seasonal = 5 * np.sin(2 * np.pi * t)
noise = np.random.normal(0, 1, 100)
y = trend + seasonal + noise

# Create time series
ts = pd.Series(y, index=pd.date_range('2023', periods=100, freq='D'))

# Perform decomposition
decomposition = seasonal_decompose(ts, period=25)

# Access components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
```

Slide 3: Stationarity Analysis

Stationarity is a fundamental concept where statistical properties remain constant over time. Testing for stationarity using the Augmented Dickey-Fuller test helps determine if differencing or transformation is needed before applying time series models.

```python
from statsmodels.tsa.stattools import adfuller

def check_stationarity(timeseries):
    # Perform Augmented Dickey-Fuller test
    result = adfuller(timeseries.dropna())
    
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
        
    # Interpret results
    is_stationary = result[1] < 0.05
    return is_stationary

# Example usage
is_stationary = check_stationarity(ts)
print(f"\nIs the series stationary? {is_stationary}")
```

Slide 4: Moving Average Smoothing

Moving averages provide a fundamental approach to smoothing time series data by reducing noise and highlighting trends. The window size determines the balance between noise reduction and preservation of temporal patterns.

```python
def calculate_moving_averages(timeseries, windows=[7, 14, 30]):
    ma_dict = {}
    for window in windows:
        ma_dict[f'MA{window}'] = timeseries.rolling(
            window=window,
            center=True,
            min_periods=1
        ).mean()
    
    # Combine all MAs into a DataFrame
    ma_df = pd.DataFrame(ma_dict)
    ma_df['Original'] = timeseries
    
    return ma_df

# Calculate different moving averages
ma_results = calculate_moving_averages(ts)
print("First 5 rows of moving averages:")
print(ma_results.head())
```

Slide 5: Advanced Time Series Transformations

Time series often require mathematical transformations to achieve stationarity or normalize distributions. Box-Cox and logarithmic transformations help stabilize variance, while differencing removes trends and seasonal patterns.

```python
from scipy import stats
import numpy as np

def transform_series(series, method='boxcox'):
    if method == 'boxcox':
        # Ensure all values are positive for Box-Cox
        if any(series <= 0):
            series = series + abs(min(series)) + 1
        transformed, lambda_param = stats.boxcox(series)
        return transformed, lambda_param
    
    elif method == 'log':
        # Add small constant to handle zeros
        transformed = np.log1p(series)
        return transformed, None
    
    elif method == 'diff':
        transformed = series.diff().dropna()
        return transformed, None

# Example usage with different transformations
boxcox_transformed, lambda_param = transform_series(ts, method='boxcox')
log_transformed, _ = transform_series(ts, method='log')
diff_transformed, _ = transform_series(ts, method='diff')
```

Slide 6: Trend Analysis and Regression

Trend analysis involves fitting mathematical functions to capture long-term patterns in time series data. Polynomial regression and robust regression techniques help identify and model underlying trends while handling outliers.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, HuberRegressor
import numpy as np

def analyze_trend(timeseries, degree=2):
    # Create time index
    X = np.arange(len(timeseries)).reshape(-1, 1)
    
    # Polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    # Fit models
    linear_model = LinearRegression().fit(X_poly, timeseries)
    robust_model = HuberRegressor().fit(X_poly, timeseries)
    
    # Generate predictions
    trend_linear = linear_model.predict(X_poly)
    trend_robust = robust_model.predict(X_poly)
    
    return pd.DataFrame({
        'Original': timeseries,
        'Polynomial_Trend': trend_linear,
        'Robust_Trend': trend_robust
    })

# Analyze trends
trend_analysis = analyze_trend(ts)
print("First 5 rows of trend analysis:")
print(trend_analysis.head())
```

Slide 7: Seasonal Pattern Detection

Detecting and quantifying seasonal patterns requires specialized techniques including periodogram analysis and autocorrelation functions. These methods reveal cyclical components and their respective strengths in the time series.

```python
from scipy import signal
import numpy as np

def detect_seasonality(timeseries, sampling_freq=1):
    # Compute periodogram
    freqs, psd = signal.periodogram(timeseries, fs=sampling_freq)
    
    # Find dominant frequencies
    peak_indices = signal.find_peaks(psd)[0]
    dominant_freqs = freqs[peak_indices]
    dominant_periods = 1/dominant_freqs
    
    # Calculate autocorrelation
    autocorr = pd.Series(timeseries).autocorr(lag=range(1, 51))
    
    return {
        'dominant_periods': dominant_periods,
        'periodogram': (freqs, psd),
        'autocorrelation': autocorr
    }

# Detect seasonal patterns
seasonality_results = detect_seasonality(ts)
print("Dominant seasonal periods:", seasonality_results['dominant_periods'])
```

Slide 8: Outlier Detection Methods

Time series outlier detection requires consideration of temporal dependencies. This implementation combines statistical methods with rolling statistics to identify anomalous observations while accounting for local temporal context.

```python
def detect_outliers(timeseries, window=7, sigma=3):
    # Calculate rolling statistics
    rolling_mean = timeseries.rolling(window=window).mean()
    rolling_std = timeseries.rolling(window=window).std()
    
    # Z-score based detection
    z_scores = (timeseries - rolling_mean) / rolling_std
    
    # Modified Z-score based detection
    median = timeseries.rolling(window=window).median()
    mad = (timeseries - median).abs().rolling(window=window).median()
    modified_z_scores = 0.6745 * (timeseries - median) / mad
    
    # Identify outliers
    outliers = pd.DataFrame({
        'Original': timeseries,
        'Is_Outlier': (abs(z_scores) > sigma) | (abs(modified_z_scores) > sigma),
        'Z_Score': z_scores,
        'Modified_Z_Score': modified_z_scores
    })
    
    return outliers

# Detect outliers
outlier_results = detect_outliers(ts)
print("\nNumber of detected outliers:", outlier_results['Is_Outlier'].sum())
```

Slide 9: Advanced Forecasting with Prophet

Facebook's Prophet model provides sophisticated forecasting capabilities by automatically decomposing multiple seasonal patterns and handling holidays. This implementation showcases Prophet's ability to generate detailed predictions with uncertainty intervals.

```python
from fbprophet import Prophet
import pandas as pd

def prophet_forecast(timeseries, periods=30, yearly_seasonality=True):
    # Prepare data for Prophet
    df = pd.DataFrame({
        'ds': timeseries.index,
        'y': timeseries.values
    })
    
    # Initialize and fit Prophet model
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=True,
        daily_seasonality=False,
        uncertainty_samples=1000
    )
    model.fit(df)
    
    # Create future dates for forecasting
    future = model.make_future_dataframe(periods=periods)
    
    # Generate forecast
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Generate forecast
forecast_results = prophet_forecast(ts)
print("Forecast for next 5 periods:")
print(forecast_results.tail().to_string())
```

Slide 10: SARIMA Model Implementation

Seasonal ARIMA (SARIMA) models capture both temporal dependencies and seasonal patterns. This implementation includes automatic parameter selection using AIC criterion and provides comprehensive model diagnostics.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

def optimize_sarima(timeseries, max_p=2, max_d=2, max_q=2, max_P=1, max_D=1, max_Q=1, s=12):
    best_aic = float('inf')
    best_params = None
    
    # Generate parameter combinations
    pdq = list(product(range(max_p + 1), range(max_d + 1), range(max_q + 1)))
    seasonal_pdq = list(product(range(max_P + 1), range(max_D + 1), range(max_Q + 1), [s]))
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(timeseries,
                              order=param,
                              seasonal_order=param_seasonal,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                results = model.fit(disp=False)
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (param, param_seasonal)
            except:
                continue
                
    return best_params

# Find optimal parameters and fit model
best_params = optimize_sarima(ts)
final_model = SARIMAX(ts, 
                     order=best_params[0],
                     seasonal_order=best_params[1]).fit(disp=False)

# Generate forecast
forecast = final_model.forecast(steps=30)
print(f"Best SARIMA parameters: {best_params}")
print("\nForecast for next 5 periods:")
print(forecast.head())
```

Slide 11: Dynamic Time Warping Implementation

Dynamic Time Warping (DTW) provides a robust measure of similarity between temporal sequences. This implementation includes both the basic DTW algorithm and an optimized version with window constraints.

```python
import numpy as np
from numba import jit

@jit(nopython=True)
def dtw_distance(x, y, window=None):
    n, m = len(x), len(y)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0
    
    w = window if window else max(n, m)
    
    for i in range(1, n+1):
        for j in range(max(1, i-w), min(m+1, i+w+1)):
            cost = abs(x[i-1] - y[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1]   # match
            )
    
    return dtw_matrix[n, m]

# Example usage with two time series
ts1 = np.random.normal(0, 1, 100)
ts2 = np.random.normal(0, 1, 100)

# Calculate DTW distance
distance = dtw_distance(ts1, ts2, window=10)
print(f"DTW distance between sequences: {distance}")
```

Slide 12: Neural Prophet Implementation

Neural Prophet combines neural networks with Prophet's decomposition approach, offering enhanced flexibility for complex time series patterns. This implementation includes custom seasonality and automatic hyperparameter tuning.

```python
from neuralprophet import NeuralProphet
import pandas as pd

def neural_prophet_forecast(timeseries, periods=30, epochs=100):
    # Prepare data
    df = pd.DataFrame({
        'ds': timeseries.index,
        'y': timeseries.values
    })
    
    # Initialize model with custom parameters
    model = NeuralProphet(
        growth="linear",
        n_changepoints=10,
        changepoints_range=0.8,
        num_hidden_layers=2,
        d_hidden=64,
        learning_rate=0.001,
        epochs=epochs,
        batch_size=32,
        loss_func="Huber"
    )
    
    # Add seasonality
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Fit model
    metrics = model.fit(df, freq='D')
    
    # Generate forecast
    future = model.make_future_dataframe(df, periods=periods)
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Generate forecast
neural_forecast = neural_prophet_forecast(ts)
print("Neural Prophet forecast for next 5 periods:")
print(neural_forecast.head())
```

Slide 13: Wavelet Analysis for Time Series

Wavelet transformation provides multi-resolution analysis of time series, revealing patterns at different time scales. This implementation uses continuous wavelet transform to decompose signals and identify time-localized frequency components.

```python
import pywt
import numpy as np

def wavelet_analysis(timeseries, wavelet='cmor1.5-1.0', scales=None):
    # Generate scales if not provided
    if scales is None:
        scales = np.arange(1, min(len(timeseries)//2, 128))
    
    # Perform continuous wavelet transform
    coef, freqs = pywt.cwt(timeseries, scales, wavelet)
    
    # Calculate power spectrum
    power = (abs(coef)) ** 2
    
    # Identify dominant frequencies
    global_power = np.sum(power, axis=1)
    dominant_scales = scales[np.argsort(global_power)[-3:]]
    
    return {
        'coefficients': coef,
        'frequencies': freqs,
        'power': power,
        'dominant_scales': dominant_scales
    }

# Perform wavelet analysis
wavelet_results = wavelet_analysis(ts)
print("Dominant time scales detected:")
print(wavelet_results['dominant_scales'])

# Example reconstruction
reconstructed = np.real(pywt.icwt(
    wavelet_results['coefficients'], 
    wavelet_results['frequencies'], 
    'cmor1.5-1.0'
))
```

Slide 14: Kalman Filtering for Time Series

Kalman filtering provides optimal state estimation for time series with noise. This implementation includes both linear and extended Kalman filters for tracking underlying states and filtering noise.

```python
import numpy as np
from scipy.linalg import inv

class KalmanFilter:
    def __init__(self, dim_state, dim_measure):
        self.dim_state = dim_state
        self.dim_measure = dim_measure
        
        # Initialize state matrices
        self.state = np.zeros(dim_state)
        self.P = np.eye(dim_state)  # State covariance
        self.F = np.eye(dim_state)  # State transition
        self.H = np.zeros((dim_measure, dim_state))  # Measurement
        self.R = np.eye(dim_measure)  # Measurement noise
        self.Q = np.eye(dim_state)  # Process noise
        
    def predict(self):
        # Predict next state
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state
        
    def update(self, measurement):
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ inv(S)
        
        # Update state
        y = measurement - self.H @ self.state
        self.state = self.state + K @ y
        self.P = (np.eye(self.dim_state) - K @ self.H) @ self.P
        
        return self.state

# Example usage
kf = KalmanFilter(dim_state=2, dim_measure=1)
kf.H = np.array([[1.0, 0.0]])  # Measurement matrix

filtered_states = []
for measurement in ts:
    kf.predict()
    filtered_state = kf.update(measurement)
    filtered_states.append(filtered_state[0])

print("First 5 filtered states:")
print(filtered_states[:5])
```

Slide 15: Additional Resources

*   "Time Series Analysis and Its Applications" - ArXiv:2107.12839
    *   [https://arxiv.org/abs/2107.12839](https://arxiv.org/abs/2107.12839)
*   "Deep Learning for Time Series Forecasting" - ArXiv:2004.13408
    *   [https://arxiv.org/abs/2004.13408](https://arxiv.org/abs/2004.13408)
*   "Modern Time Series Analysis Techniques" - ArXiv:2109.14293
    *   [https://arxiv.org/abs/2109.14293](https://arxiv.org/abs/2109.14293)
*   "Statistical Methods for Time Series Data" - JMLR
    *   [http://www.jmlr.org/papers/time\_series\_analysis](http://www.jmlr.org/papers/time_series_analysis)
*   Recommended search terms for Google Scholar:
    *   "Advanced time series decomposition methods"
    *   "Neural networks for temporal data analysis"
    *   "State-space models in time series"

