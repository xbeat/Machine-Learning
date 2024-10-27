## Time Series Forecasting Project Walkthrough
Slide 1: Setting Up the Environment and Data Loading

Time series analysis requires specific Python libraries for statistical computations, data manipulation, and visualization. We'll import essential packages and set up our environment for forecasting analysis, ensuring we have all necessary tools for comprehensive time series modeling.

```python
# Import required libraries for time series analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
def load_stock_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Example usage
df = load_stock_data('stock_prices.csv')
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
```

Slide 2: Data Exploration and Summary Statistics

Comprehensive data exploration involves analyzing statistical properties, checking data types, and understanding the distribution of values. This step is crucial for identifying potential issues and gaining insights into the time series characteristics.

```python
# Perform initial data analysis
def explore_dataset(df):
    # Display basic information
    print("Dataset Information:")
    print(df.info())
    
    # Calculate summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Time series characteristics
    print("\nTime Series Properties:")
    print(f"Start Date: {df.index.min()}")
    print(f"End Date: {df.index.max()}")
    print(f"Time Series Length: {len(df)} observations")

# Execute exploration
explore_dataset(df)
```

Slide 3: Time Series Visualization

Visualizing time series data helps identify patterns, trends, and potential anomalies. We'll create comprehensive plots showing the stock price evolution over time, including moving averages to highlight underlying trends.

```python
def visualize_time_series(df, column='Close'):
    plt.figure(figsize=(15, 7))
    
    # Plot original time series
    plt.plot(df.index, df[column], label='Original Series', alpha=0.8)
    
    # Add moving averages
    df['MA50'] = df[column].rolling(window=50).mean()
    df['MA200'] = df[column].rolling(window=200).mean()
    
    plt.plot(df.index, df['MA50'], label='50-day MA', alpha=0.6)
    plt.plot(df.index, df['MA200'], label='200-day MA', alpha=0.6)
    
    plt.title('Stock Price Time Series with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

visualize_time_series(df)
```

Slide 4: Stationarity Testing

Understanding stationarity is crucial for time series analysis. We'll implement both Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests to thoroughly assess the stationarity of our series.

```python
def check_stationarity(series):
    # ADF Test
    adf_result = adfuller(series, autolag='AIC')
    print("ADF Test Results:")
    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f'\t{key}: {value}')
    
    # KPSS Test
    kpss_result = kpss(series, regression='c')
    print("\nKPSS Test Results:")
    print(f'KPSS Statistic: {kpss_result[0]}')
    print(f'p-value: {kpss_result[1]}')
    
# Test stationarity
check_stationarity(df['Close'])
```

Slide 5: Seasonal Pattern Analysis

Fourier transformation helps identify hidden periodicities in the time series data. We'll implement spectral analysis to determine the dominant frequencies and potential seasonal patterns.

```python
def analyze_seasonality(series):
    # Compute Fourier Transform
    n = len(series)
    fft = np.fft.fft(series)
    freq = np.fft.fftfreq(n)
    
    # Get power spectrum
    power = np.abs(fft) ** 2
    
    # Plot periodogram
    plt.figure(figsize=(15, 7))
    plt.plot(freq[1:n//2], power[1:n//2])
    plt.title('Periodogram: Frequency Domain Analysis')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.grid(True)
    plt.show()
    
    # Find dominant frequencies
    top_frequencies = sorted(zip(power[1:n//2], freq[1:n//2]), reverse=True)[:5]
    print("\nDominant Periods (days):")
    for power, freq in top_frequencies:
        if freq != 0:
            period = 1/abs(freq)
            print(f"Period: {period:.2f} days (Power: {power:.2e})")

analyze_seasonality(df['Close'])
```

Slide 6: ACF and PACF Analysis

Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots are essential tools for identifying the order of ARIMA models and detecting seasonality patterns in the time series data.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def analyze_correlations(series, lags=40):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot ACF
    plot_acf(series, lags=lags, ax=ax1)
    ax1.set_title('Autocorrelation Function')
    
    # Plot PACF
    plot_pacf(series, lags=lags, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate significant lags
    def get_significant_lags(series, lags):
        acf_values = pd.plotting.autocorrelation_plot(series)
        confidence_interval = 1.96/np.sqrt(len(series))
        significant_lags = [i for i in range(1, lags) 
                          if abs(acf_values.values[i]) > confidence_interval]
        return significant_lags

    sig_lags = get_significant_lags(series, lags)
    print(f"Significant lags: {sig_lags}")

# Analyze correlations for the differenced series
analyze_correlations(df['Close'].diff().dropna())
```

Slide 7: Time Series Decomposition

Decomposing the time series into its fundamental components - trend, seasonality, and residuals - provides insights into the underlying patterns and helps in choosing appropriate forecasting models.

```python
def perform_decomposition(series):
    # Additive decomposition
    decomposition_add = seasonal_decompose(series, period=252, model='additive')
    
    # Plot components
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
    
    decomposition_add.observed.plot(ax=ax1)
    ax1.set_title('Original Series')
    ax1.grid(True)
    
    decomposition_add.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    ax2.grid(True)
    
    decomposition_add.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    ax3.grid(True)
    
    decomposition_add.resid.plot(ax=ax4)
    ax4.set_title('Residuals')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate component strengths
    total_variance = np.var(series)
    trend_strength = np.var(decomposition_add.trend) / total_variance
    seasonal_strength = np.var(decomposition_add.seasonal) / total_variance
    residual_strength = np.var(decomposition_add.resid.dropna()) / total_variance
    
    print(f"Component Strengths:")
    print(f"Trend: {trend_strength:.2%}")
    print(f"Seasonal: {seasonal_strength:.2%}")
    print(f"Residual: {residual_strength:.2%}")

perform_decomposition(df['Close'])
```

Slide 8: Data Preprocessing and Train-Test Split

Proper data preprocessing and splitting are crucial for model evaluation. We'll implement robust preprocessing steps including scaling and create training and testing sets while maintaining temporal order.

```python
from sklearn.preprocessing import MinMaxScaler

def preprocess_and_split(df, train_size=0.8):
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    
    # Create DataFrame with scaled data
    scaled_df = pd.DataFrame(scaled_data, 
                           index=df.index, 
                           columns=['Scaled_Close'])
    
    # Calculate split point
    split_idx = int(len(scaled_df) * train_size)
    
    # Split the data
    train = scaled_df[:split_idx]
    test = scaled_df[split_idx:]
    
    print(f"Training set size: {len(train)} observations")
    print(f"Testing set size: {len(test)} observations")
    print(f"Training period: {train.index[0]} to {train.index[-1]}")
    print(f"Testing period: {test.index[0]} to {test.index[-1]}")
    
    return train, test, scaler

# Preprocess and split the data
train_data, test_data, scaler = preprocess_and_split(df)

# Visualize the split
plt.figure(figsize=(15, 7))
plt.plot(train_data.index, train_data['Scaled_Close'], label='Training Data')
plt.plot(test_data.index, test_data['Scaled_Close'], label='Testing Data')
plt.title('Train-Test Split Visualization')
plt.xlabel('Date')
plt.ylabel('Scaled Price')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 9: ARIMA Model Implementation

ARIMA (AutoRegressive Integrated Moving Average) modeling requires careful parameter selection. We'll implement a systematic approach to determine optimal parameters using both manual and automated methods.

```python
def implement_arima(train, test):
    # Determine optimal parameters using auto_arima
    model = auto_arima(train, start_p=0, start_q=0, max_p=5, max_q=5,
                      m=1, start_P=0, seasonal=False, d=1, D=1,
                      trace=True, error_action='ignore',
                      suppress_warnings=True, stepwise=True)
    
    # Print model summary
    print(model.summary())
    
    # Fit the model
    model.fit(train)
    
    # Make predictions
    forecast = model.predict(n_periods=len(test))
    
    # Calculate error metrics
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    
    print(f"\nModel Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return forecast, mae, rmse

# Implement ARIMA
forecast_arima, mae_arima, rmse_arima = implement_arima(
    train_data['Scaled_Close'], 
    test_data['Scaled_Close']
)
```

Slide 10: SARIMA Model Development

SARIMA extends ARIMA by incorporating seasonal components. We'll implement a comprehensive SARIMA model with automatic parameter selection and validation through information criteria.

```python
def implement_sarima(train, test):
    # Auto SARIMA parameter selection
    model = auto_arima(train, start_p=0, start_q=0, max_p=5, max_q=5,
                      m=12, start_P=0, seasonal=True, d=1, D=1,
                      trace=True, error_action='ignore',
                      suppress_warnings=True, stepwise=True)
    
    # Fit the model and make predictions
    model.fit(train)
    forecast = model.predict(n_periods=len(test))
    
    # Plot results
    plt.figure(figsize=(15, 7))
    plt.plot(test.index, test, label='Actual')
    plt.plot(test.index, forecast, label='SARIMA Forecast')
    plt.title('SARIMA Forecast vs Actual Values')
    plt.xlabel('Date')
    plt.ylabel('Scaled Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate metrics
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test)) * 100
    
    print(f"\nSARIMA Performance Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    return forecast, model

# Implement SARIMA
forecast_sarima, sarima_model = implement_sarima(
    train_data['Scaled_Close'],
    test_data['Scaled_Close']
)
```

Slide 11: Prophet Model Implementation

Facebook Prophet is designed for forecasting time series data with strong seasonal patterns and multiple seasonality levels. We'll implement a Prophet model with custom seasonality parameters.

```python
def implement_prophet(train, test):
    # Prepare data for Prophet
    df_prophet = pd.DataFrame({
        'ds': train.index,
        'y': train.values.flatten()
    })
    
    # Initialize and fit Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    model.fit(df_prophet)
    
    # Create future dataframe for forecasting
    future = pd.DataFrame({'ds': test.index})
    
    # Make predictions
    forecast = model.predict(future)
    
    # Plot results
    fig = model.plot(forecast)
    plt.title('Prophet Forecast')
    plt.show()
    
    # Plot components
    fig = model.plot_components(forecast)
    plt.show()
    
    # Calculate metrics
    mae = mean_absolute_error(test.values, forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(test.values, forecast['yhat']))
    mape = np.mean(np.abs((test.values - forecast['yhat']) / test.values)) * 100
    
    print(f"\nProphet Performance Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    return forecast, model

# Implement Prophet
forecast_prophet, prophet_model = implement_prophet(
    train_data['Scaled_Close'],
    test_data['Scaled_Close']
)
```

Slide 12: Model Comparison and Evaluation

A comprehensive comparison of all implemented models helps identify the most suitable approach for our time series forecasting task. We'll evaluate using multiple metrics and visual analysis.

```python
def compare_models(test, arima_forecast, sarima_forecast, prophet_forecast):
    # Create comparison DataFrame
    results = pd.DataFrame({
        'Actual': test.values.flatten(),
        'ARIMA': arima_forecast,
        'SARIMA': sarima_forecast,
        'Prophet': prophet_forecast['yhat'].values
    }, index=test.index)
    
    # Calculate metrics for each model
    metrics = {}
    models = ['ARIMA', 'SARIMA', 'Prophet']
    
    for model in models:
        metrics[model] = {
            'MAE': mean_absolute_error(results['Actual'], results[model]),
            'RMSE': np.sqrt(mean_squared_error(results['Actual'], results[model])),
            'MAPE': np.mean(np.abs((results['Actual'] - results[model]) / 
                                 results['Actual'])) * 100
        }
    
    # Plot comparison
    plt.figure(figsize=(15, 7))
    plt.plot(results.index, results['Actual'], label='Actual', linewidth=2)
    for model in models:
        plt.plot(results.index, results[model], label=model, alpha=0.7)
    plt.title('Model Comparison')
    plt.xlabel('Date')
    plt.ylabel('Scaled Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print metrics comparison
    print("\nModel Comparison Metrics:")
    metrics_df = pd.DataFrame(metrics).round(4)
    print(metrics_df)
    
    return metrics_df

# Compare all models
metrics_comparison = compare_models(
    test_data['Scaled_Close'],
    forecast_arima,
    forecast_sarima,
    forecast_prophet
)
```

Slide 13: Forecasting Future Values

After model evaluation, we'll use the best performing model to generate future predictions, including confidence intervals and uncertainty estimates for more reliable forecasting results.

```python
def forecast_future(best_model, scaler, periods=30):
    # Generate future dates
    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=periods,
        freq='B'
    )
    
    # Make predictions
    if isinstance(best_model, Prophet):
        future = pd.DataFrame({'ds': future_dates})
        forecast = best_model.predict(future)
        predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        # Inverse transform the predictions
        predictions['yhat'] = scaler.inverse_transform(
            predictions[['yhat']])
        predictions['yhat_lower'] = scaler.inverse_transform(
            predictions[['yhat_lower']])
        predictions['yhat_upper'] = scaler.inverse_transform(
            predictions[['yhat_upper']])
    else:
        # For SARIMA/ARIMA models
        forecast = best_model.predict(n_periods=periods, 
                                    return_conf_int=True)
        predictions = pd.DataFrame(
            index=future_dates,
            data={
                'forecast': scaler.inverse_transform(
                    forecast[0].reshape(-1, 1)).flatten(),
                'lower': scaler.inverse_transform(
                    forecast[1][:, 0].reshape(-1, 1)).flatten(),
                'upper': scaler.inverse_transform(
                    forecast[1][:, 1].reshape(-1, 1)).flatten()
            }
        )
    
    # Plotting
    plt.figure(figsize=(15, 7))
    plt.plot(df.index[-100:], df['Close'][-100:], 
             label='Historical Data', color='blue')
    plt.plot(predictions.index, predictions['forecast'], 
             label='Forecast', color='red')
    plt.fill_between(predictions.index,
                     predictions['lower'],
                     predictions['upper'],
                     color='red', alpha=0.1, label='Confidence Interval')
    plt.title('Future Price Forecast with Confidence Intervals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return predictions

# Generate future predictions
future_predictions = forecast_future(prophet_model, scaler)
```

Slide 14: Performance Analysis and Model Diagnostics

Comprehensive model diagnostics help understand model behavior and validate assumptions. We'll implement various diagnostic tests and visualizations to ensure model reliability.

```python
def perform_diagnostics(model, residuals):
    # Residual Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time plot of residuals
    ax1.plot(residuals)
    ax1.set_title('Residuals over Time')
    ax1.grid(True)
    
    # Histogram of residuals
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.set_title('Residual Distribution')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot')
    
    # Residual ACF
    plot_acf(residuals, ax=ax4, lags=40)
    ax4.set_title('Residual ACF')
    
    plt.tight_layout()
    plt.show()
    
    # Statistical Tests
    print("\nDiagnostic Tests:")
    
    # Ljung-Box Test for autocorrelation
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(residuals, lags=10)
    print("\nLjung-Box Test:")
    print(f"p-value: {lb_test.iloc[0, 1]:.4f}")
    
    # Jarque-Bera Test for normality
    from scipy.stats import jarque_bera
    jb_stat, jb_pval = jarque_bera(residuals)
    print("\nJarque-Bera Test:")
    print(f"p-value: {jb_pval:.4f}")
    
    # Heteroscedasticity test
    from statsmodels.stats.diagnostic import het_white
    white_test = het_white(residuals.reshape(-1, 1), 
                          np.ones((len(residuals), 2)))
    print("\nWhite's Test for Heteroscedasticity:")
    print(f"p-value: {white_test[1]:.4f}")

# Calculate residuals and perform diagnostics
residuals = test_data['Scaled_Close'] - forecast_prophet['yhat']
perform_diagnostics(prophet_model, residuals)
```

Slide 15: Additional Resources

```text
Recent papers and resources for further reading:

1. "Deep Learning for Time Series Forecasting: A Survey"
https://arxiv.org/abs/2004.13408

2. "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
https://arxiv.org/abs/1905.10437

3. "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
https://arxiv.org/abs/1912.09363

4. "Probabilistic Forecasting with Temporal Convolutional Neural Networks"
https://arxiv.org/abs/1906.04397

5. "Meta-learning framework with applications to zero-shot time-series forecasting"
https://arxiv.org/abs/2002.02887
```

