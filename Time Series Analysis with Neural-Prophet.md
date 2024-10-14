## Time Series Analysis with Neural-Prophet
Slide 1: Introduction to Time Series Analysis and Neural-Prophet

Time series analysis is a crucial technique for understanding and predicting patterns in sequential data. Neural-Prophet, an advanced forecasting model, combines the strengths of traditional time series methods with neural networks. This powerful tool is particularly useful for predicting energy prices, helping businesses and policymakers make informed decisions about resource allocation and pricing strategies.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample time series data
np.random.seed(42)
time = np.arange(0, 100, 1)
energy_prices = 50 + 10 * np.sin(time / 10) + np.random.normal(0, 5, 100)

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(time, energy_prices)
plt.title('Energy Prices Time Series')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
```

Slide 2: Components of Time Series Data

Time series data typically consists of several components: trend, seasonality, and residuals. Understanding these components is essential for accurate forecasting. The trend represents the long-term direction of the data, seasonality captures recurring patterns, and residuals are the unexplained variations.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Convert the data to a pandas Series
import pandas as pd
energy_prices_series = pd.Series(energy_prices, index=pd.date_range(start='2023-01-01', periods=100, freq='D'))

# Perform seasonal decomposition
result = seasonal_decompose(energy_prices_series, model='additive', period=30)

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

Slide 3: Introduction to Neural-Prophet

Neural-Prophet is a novel forecasting model that combines the interpretability of Facebook's Prophet with the flexibility of neural networks. It is designed to handle complex time series data with multiple seasonalities, holidays, and external regressors. Neural-Prophet is particularly well-suited for energy price forecasting due to its ability to capture intricate patterns and relationships in the data.

```python
from neuralprophet import NeuralProphet
import pandas as pd

# Prepare data for Neural-Prophet
df = pd.DataFrame({'ds': pd.date_range(start='2023-01-01', periods=100, freq='D'),
                   'y': energy_prices})

# Initialize and fit the model
model = NeuralProphet()
model.fit(df, freq='D')

# Make future predictions
future = model.make_future_dataframe(df, periods=30)
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title('Energy Price Forecast')
plt.show()
```

Slide 4: Key Features of Neural-Prophet

Neural-Prophet offers several key features that make it a powerful tool for energy price forecasting. These include automatic seasonality detection, the ability to incorporate external regressors, and handling of missing data. The model also provides uncertainty estimates, which are crucial for risk assessment in energy markets.

```python
# Add external regressor (e.g., temperature)
np.random.seed(42)
temperature = 20 + 10 * np.sin(time / 10) + np.random.normal(0, 2, 100)
df['temperature'] = temperature

# Initialize model with external regressor
model = NeuralProphet()
model.add_lagged_regressor(name='temperature')

# Fit the model
model.fit(df, freq='D')

# Make future predictions
future = model.make_future_dataframe(df, periods=30)
future['temperature'] = 20 + 10 * np.sin(np.arange(130) / 10) + np.random.normal(0, 2, 130)
forecast = model.predict(future)

# Plot the forecast with uncertainty
fig = model.plot(forecast)
plt.title('Energy Price Forecast with Temperature as Regressor')
plt.show()
```

Slide 5: Data Preprocessing for Neural-Prophet

Before applying Neural-Prophet to energy price data, it's essential to preprocess the data properly. This includes handling missing values, detecting and removing outliers, and normalizing the data if necessary. Proper preprocessing ensures that the model can learn meaningful patterns from the data.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Generate sample data with missing values and outliers
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
prices = 50 + 10 * np.sin(np.arange(100) / 10) + np.random.normal(0, 5, 100)
prices[10:15] = np.nan  # Add missing values
prices[50] = 200  # Add an outlier

df = pd.DataFrame({'ds': dates, 'y': prices})

# Handle missing values
df['y'] = df['y'].interpolate()

# Remove outliers using IQR method
Q1 = df['y'].quantile(0.25)
Q3 = df['y'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['y'] >= Q1 - 1.5 * IQR) & (df['y'] <= Q3 + 1.5 * IQR)]

# Normalize the data
scaler = MinMaxScaler()
df['y_normalized'] = scaler.fit_transform(df[['y']])

# Plot original and preprocessed data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
ax1.plot(df['ds'], df['y'])
ax1.set_title('Original Data')
ax2.plot(df['ds'], df['y_normalized'])
ax2.set_title('Preprocessed Data')
plt.tight_layout()
plt.show()
```

Slide 6: Model Architecture of Neural-Prophet

Neural-Prophet's architecture combines traditional time series components with neural networks. It uses a Fourier series to model seasonality and a linear layer to capture trends. The model also incorporates a feed-forward neural network to learn complex patterns and interactions between features. This hybrid approach allows Neural-Prophet to balance interpretability with predictive power.

Slide 7: Model Architecture of Neural-Prophet

```python
import torch
import torch.nn as nn

class SimplifiedNeuralProphet(nn.Module):
    def __init__(self, n_forecasts=1, n_lags=0, n_hidden=32):
        super().__init__()
        self.n_forecasts = n_forecasts
        self.n_lags = n_lags
        
        # Trend component
        self.trend = nn.Linear(1, 1)
        
        # Seasonality component (simplified)
        self.seasonality = nn.Linear(2, 1)  # Using sin and cos
        
        # AR component
        if n_lags > 0:
            self.ar = nn.Linear(n_lags, n_hidden)
        
        # Feed-forward neural network
        self.ffn = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_forecasts)
        )
    
    def forward(self, time, lags=None):
        # Trend
        trend = self.trend(time.unsqueeze(1))
        
        # Seasonality (simplified)
        s = torch.stack([torch.sin(time), torch.cos(time)], dim=1)
        seasonality = self.seasonality(s)
        
        # AR component
        if self.n_lags > 0 and lags is not None:
            ar = self.ar(lags)
        else:
            ar = torch.zeros(time.size(0), self.n_hidden)
        
        # Combine components
        x = trend + seasonality + ar
        
        # Feed-forward network
        return self.ffn(x)

# Create a simple example
model = SimplifiedNeuralProphet(n_lags=5, n_hidden=32)
print(model)
```

Slide 8: Training Neural-Prophet Models

Training a Neural-Prophet model involves specifying the model parameters, preparing the data, and fitting the model to the training set. The model uses backpropagation and stochastic gradient descent to optimize its parameters. It's important to monitor the training process and use techniques like early stopping to prevent overfitting.

Slide 9: Training Neural-Prophet Models

```python
from neuralprophet import NeuralProphet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
values = 50 + 10 * np.sin(np.arange(365) / 30) + np.random.normal(0, 5, 365)
df = pd.DataFrame({'ds': dates, 'y': values})

# Split data into training and testing sets
train_df = df[:300]
test_df = df[300:]

# Initialize and train the model
model = NeuralProphet(
    n_forecasts=7,
    n_lags=14,
    num_hidden_layers=2,
    d_hidden=32
)

metrics = model.fit(train_df, freq='D', validation_df=test_df)

# Plot training metrics
plt.figure(figsize=(10, 6))
plt.plot(metrics['train_loss'], label='Training Loss')
plt.plot(metrics['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Make predictions
future = model.make_future_dataframe(df, periods=30)
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title('Energy Price Forecast')
plt.show()
```

Slide 10: Handling Seasonality in Energy Prices

Energy prices often exhibit multiple seasonalities, such as daily, weekly, and yearly patterns. Neural-Prophet can automatically detect and model these seasonal patterns, improving the accuracy of forecasts. Understanding and incorporating seasonality is crucial for predicting energy prices accurately.

Slide 11: Handling Seasonality in Energy Prices

```python
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt

# Generate sample data with multiple seasonalities
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=365*2, freq='H')
hourly_pattern = np.sin(np.arange(365*2*24) * (2*np.pi/24))
weekly_pattern = np.sin(np.arange(365*2*24) * (2*np.pi/(7*24)))
yearly_pattern = np.sin(np.arange(365*2*24) * (2*np.pi/(365*24)))
values = 50 + 10*hourly_pattern + 20*weekly_pattern + 30*yearly_pattern + np.random.normal(0, 5, 365*2*24)

df = pd.DataFrame({'ds': dates, 'y': values})

# Initialize and train the model
model = NeuralProphet(
    n_forecasts=24,
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)

model.fit(df, freq='H')

# Plot the components
fig = model.plot_components(model.predict(df))
plt.show()

# Make future predictions
future = model.make_future_dataframe(df, periods=7*24)  # Forecast for a week
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title('Energy Price Forecast with Multiple Seasonalities')
plt.show()
```

Slide 12: Incorporating External Factors

Energy prices are often influenced by external factors such as weather conditions, economic indicators, or geopolitical events. Neural-Prophet allows the incorporation of these external regressors to improve forecast accuracy. This feature is particularly useful for energy price prediction, where multiple factors can impact the market.

Slide 13: Incorporating External Factors

```python
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt

# Generate sample data with external factors
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
base_price = 50 + 10 * np.sin(np.arange(365) / 30)
temperature = 20 + 15 * np.sin(np.arange(365) / 365 * 2 * np.pi)  # Yearly temperature cycle
economic_index = np.cumsum(np.random.normal(0, 0.1, 365))  # Random walk for economic index

energy_prices = base_price + 0.5 * temperature + 5 * economic_index + np.random.normal(0, 5, 365)

df = pd.DataFrame({
    'ds': dates,
    'y': energy_prices,
    'temperature': temperature,
    'economic_index': economic_index
})

# Initialize and train the model with external regressors
model = NeuralProphet(
    n_forecasts=30,
    n_lags=14,
    daily_seasonality=True
)

model.add_lagged_regressor(name='temperature')
model.add_lagged_regressor(name='economic_index')

model.fit(df, freq='D')

# Make future predictions
future = model.make_future_dataframe(df, periods=60)
future['temperature'] = 20 + 15 * np.sin(np.arange(365+60) / 365 * 2 * np.pi)[-60:]
future['economic_index'] = np.cumsum(np.random.normal(0, 0.1, 60)) + economic_index[-1]

forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title('Energy Price Forecast with External Factors')
plt.show()

# Plot the impact of external regressors
fig = model.plot_components(forecast)
plt.show()
```

Slide 14: Model Evaluation and Performance Metrics

Evaluating the performance of Neural-Prophet models is crucial for ensuring accurate energy price forecasts. Common metrics include Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Mean Absolute Percentage Error (MAPE). It's also important to compare the model's performance against benchmark models and to assess its ability to capture trends and seasonality.

Slide 15: Model Evaluation and Performance Metrics

```python
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
values = 50 + 10 * np.sin(np.arange(365) / 30) + np.random.normal(0, 5, 365)
df = pd.DataFrame({'ds': dates, 'y': values})

# Split data into training and testing sets
train_df = df[:300]
test_df = df[300:]

# Train the model
model = NeuralProphet(n_forecasts=1, n_lags=14)
model.fit(train_df, freq='D')

# Make predictions on the test set
forecast = model.predict(test_df)

# Calculate performance metrics
mae = mean_absolute_error(test_df['y'], forecast['yhat1'])
rmse = np.sqrt(mean_squared_error(test_df['y'], forecast['yhat1']))
mape = mean_absolute_percentage_error(test_df['y'], forecast['yhat1'])

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Square Error: {rmse:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(test_df['ds'], test_df['y'], label='Actual')
plt.plot(test_df['ds'], forecast['yhat1'], label='Predicted')
plt.title('Actual vs Predicted Energy Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
```

Slide 16: Handling Uncertainty in Energy Price Forecasts

Energy price forecasting inherently involves uncertainty due to various factors such as market volatility, policy changes, and unexpected events. Neural-Prophet provides uncertainty estimates in its forecasts, which are crucial for risk assessment and decision-making in energy markets. Understanding and visualizing these uncertainty intervals can help stakeholders make more informed decisions.

Slide 17: Handling Uncertainty in Energy Price Forecasts

```python
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
values = 50 + 10 * np.sin(np.arange(365) / 30) + np.random.normal(0, 5, 365)
df = pd.DataFrame({'ds': dates, 'y': values})

# Train the model
model = NeuralProphet(n_forecasts=30, n_lags=14, quantiles=[0.1, 0.9])
model.fit(df, freq='D')

# Make future predictions
future = model.make_future_dataframe(df, periods=60)
forecast = model.predict(future)

# Plot the forecast with uncertainty intervals
plt.figure(figsize=(12, 6))
plt.plot(df['ds'], df['y'], label='Historical Data')
plt.plot(forecast['ds'], forecast['yhat1'], label='Forecast')
plt.fill_between(forecast['ds'], forecast['yhat1_10'], forecast['yhat1_90'], alpha=0.3, label='80% Confidence Interval')
plt.title('Energy Price Forecast with Uncertainty')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
```

Slide 18: Real-life Example: Predicting Solar Energy Production

Solar energy production forecasting is crucial for grid management and energy trading. Neural-Prophet can be used to predict solar energy output based on historical data and weather forecasts. This example demonstrates how to incorporate weather data as an external regressor to improve solar energy production forecasts.

Slide 19: Real-life Example: Predicting Solar Energy Production

```python
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt

# Generate sample data for solar energy production
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
base_production = 100 + 50 * np.sin(np.arange(365) / 365 * 2 * np.pi)  # Yearly cycle
weather_factor = np.random.uniform(0.5, 1.5, 365)  # Random weather impact
solar_production = base_production * weather_factor + np.random.normal(0, 10, 365)

df = pd.DataFrame({
    'ds': dates,
    'y': solar_production,
    'weather': weather_factor
})

# Train the model with weather as an external regressor
model = NeuralProphet(
    n_forecasts=7,
    n_lags=14,
    daily_seasonality=True,
    yearly_seasonality=True
)

model.add_lagged_regressor(name='weather')
model.fit(df, freq='D')

# Generate future weather data (in practice, this would come from weather forecasts)
future_weather = np.random.uniform(0.5, 1.5, 30)

# Make future predictions
future = model.make_future_dataframe(df, periods=30)
future['weather'] = future_weather
forecast = model.predict(future)

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(df['ds'], df['y'], label='Historical Production')
plt.plot(forecast['ds'], forecast['yhat1'], label='Forecast')
plt.title('Solar Energy Production Forecast')
plt.xlabel('Date')
plt.ylabel('Energy Production (MWh)')
plt.legend()
plt.show()
```

Slide 20: Real-life Example: Wind Energy Forecasting

Wind energy forecasting is essential for efficient grid integration and market operations. Neural-Prophet can be applied to predict wind power output by incorporating wind speed forecasts as external regressors. This example demonstrates how to use Neural-Prophet for wind energy forecasting, considering the non-linear relationship between wind speed and power output.

Slide 21: Real-life Example: Wind Energy Forecasting

```python
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt

# Generate sample data for wind energy production
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=365, freq='H')
wind_speed = 5 + 3 * np.sin(np.arange(365*24) / (24*7) * 2 * np.pi) + np.random.normal(0, 1, 365*24)
wind_power = np.where(wind_speed < 3, 0, np.where(wind_speed > 25, 100, (wind_speed - 3)**3 / 20))
wind_power += np.random.normal(0, 5, 365*24)

df = pd.DataFrame({
    'ds': dates,
    'y': wind_power,
    'wind_speed': wind_speed
})

# Train the model with wind speed as an external regressor
model = NeuralProphet(
    n_forecasts=24,
    n_lags=48,
    daily_seasonality=True,
    weekly_seasonality=True
)

model.add_lagged_regressor(name='wind_speed')
model.fit(df, freq='H')

# Generate future wind speed data (in practice, this would come from weather forecasts)
future_wind_speed = 5 + 3 * np.sin(np.arange(365*24, 365*24+7*24) / (24*7) * 2 * np.pi) + np.random.normal(0, 1, 7*24)

# Make future predictions
future = model.make_future_dataframe(df, periods=7*24)
future['wind_speed'] = future_wind_speed
forecast = model.predict(future)

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(df['ds'][-7*24:], df['y'][-7*24:], label='Historical Production')
plt.plot(forecast['ds'], forecast['yhat1'], label='Forecast')
plt.title('Wind Energy Production Forecast')
plt.xlabel('Date')
plt.ylabel('Energy Production (MWh)')
plt.legend()
plt.show()
```

Slide 22: Challenges and Limitations of Neural-Prophet in Energy Forecasting

While Neural-Prophet is a powerful tool for energy price and production forecasting, it's important to be aware of its challenges and limitations. These include the need for sufficient historical data, sensitivity to hyperparameter tuning, and potential difficulties in capturing sudden market shifts or extreme events. Understanding these limitations helps in proper model application and interpretation of results.

Slide 23: Challenges and Limitations of Neural-Prophet in Energy Forecasting

```python
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt

# Generate sample data with a sudden shift
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
values = 50 + 10 * np.sin(np.arange(365) / 30) + np.random.normal(0, 5, 365)
values[180:] += 30  # Sudden shift after 6 months

df = pd.DataFrame({'ds': dates, 'y': values})

# Train the model
model = NeuralProphet(n_forecasts=30, n_lags=14)
model.fit(df[:180], freq='D')  # Train on data before the shift

# Make predictions
future = model.make_future_dataframe(df, periods=185)
forecast = model.predict(future)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df['ds'], df['y'], label='Actual Data')
plt.plot(forecast['ds'], forecast['yhat1'], label='Forecast')
plt.axvline(x=dates[180], color='r', linestyle='--', label='Sudden Shift')
plt.title('Neural-Prophet Forecast with Sudden Market Shift')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
```

Slide 24: Additional Resources

For those interested in diving deeper into Neural-Prophet and its applications in energy forecasting, the following resources are recommended:

1. Neural-Prophet GitHub Repository: [https://github.com/ourownstory/neural\_prophet](https://github.com/ourownstory/neural_prophet)
2. "Forecasting at Scale" by Sean J. Taylor and Benjamin Letham (Facebook Prophet paper): [https://peerj.com/preprints/3190/](https://peerj.com/preprints/3190/)
3. "A Review of Machine Learning Techniques for Forecasting Electricity Prices" by Javier Contreras et al. (ArXiv:2106.09440): [https://arxiv.org/abs/2106.09440](https://arxiv.org/abs/2106.09440)
4. "Deep Learning for Time Series Forecasting: The Electric Load Case" by Filippo Maria Bianchi et al. (ArXiv:1907.09207): [https://arxiv.org/abs/1907.09207](https://arxiv.org/abs/1907.09207)

