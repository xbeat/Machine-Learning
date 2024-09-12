## Exploring NeuralForecast Using Python
Slide 1: Introduction to NeuralForecast

NeuralForecast is a powerful Python library designed for time series forecasting using neural networks. It provides a collection of state-of-the-art models and tools to tackle complex forecasting tasks. This library is particularly useful for data scientists and researchers working with large-scale time series data.

```python
import neuralforecast
from neuralforecast.models import NBEATS, NHITS, TFT

print(f"NeuralForecast version: {neuralforecast.__version__}")
```

Slide 2: Key Features of NeuralForecast

NeuralForecast offers a wide range of features, including support for multiple neural network architectures, handling of multivariate time series, and built-in data preprocessing capabilities. It also provides tools for model evaluation and comparison, making it easier to select the best model for your specific forecasting task.

```python
from neuralforecast.utils import AirPassengersDataset

# Load a sample dataset
Y_df, _ = AirPassengersDataset.load()
print(Y_df.head())

# Display available models
print(neuralforecast.models.__all__)
```

Slide 3: Data Preparation

Before training a model, it's essential to prepare your data properly. NeuralForecast provides utilities to handle common preprocessing tasks such as scaling, encoding categorical variables, and creating lagged features.

```python
from neuralforecast.preprocessing import TimeSeriesFeaturizer

# Create a featurizer
featurizer = TimeSeriesFeaturizer(
    num_lags=12,
    categorical_columns=['month'],
    num_categories=12
)

# Prepare features
X = featurizer.fit_transform(Y_df)
print(X.head())
```

Slide 4: Model Selection

NeuralForecast offers various neural network architectures for time series forecasting. Some popular models include NBEATS, NHITS, and Temporal Fusion Transformers (TFT). Let's create an instance of the NBEATS model.

```python
from neuralforecast.models import NBEATS

# Create an NBEATS model
model = NBEATS(
    input_size=12,
    forecast_horizon=6,
    stack_types=['trend', 'seasonality'],
    num_blocks=[3, 3],
    num_layers=[4, 4],
    layer_widths=[512, 512],
    sharing='no_sharing'
)

print(model)
```

Slide 5: Model Training

Training a model in NeuralForecast is straightforward. The library handles the creation of training and validation sets, as well as the optimization process. Let's train our NBEATS model on the AirPassengers dataset.

```python
# Train the model
history = model.fit(
    Y_df,
    val_size=12,
    epochs=100,
    batch_size=32,
    verbose=True
)

# Plot training history
import matplotlib.pyplot as plt
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
```

Slide 6: Making Predictions

Once your model is trained, you can use it to make forecasts. NeuralForecast provides methods to generate point forecasts as well as probabilistic forecasts.

```python
# Generate forecasts
forecasts = model.predict(Y_df)

# Plot the forecasts
plt.figure(figsize=(12, 6))
plt.plot(Y_df.index, Y_df['y'], label='Actual')
plt.plot(forecasts.index, forecasts['y'], label='Forecast')
plt.title('Air Passengers Forecast')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.legend()
plt.show()
```

Slide 7: Model Evaluation

NeuralForecast provides various metrics to evaluate the performance of your forecasting models. These include Mean Absolute Error (MAE), Mean Squared Error (MSE), and Mean Absolute Percentage Error (MAPE).

```python
from neuralforecast.metrics import mae, mse, mape

# Calculate evaluation metrics
mae_value = mae(Y_df['y'], forecasts['y'])
mse_value = mse(Y_df['y'], forecasts['y'])
mape_value = mape(Y_df['y'], forecasts['y'])

print(f"MAE: {mae_value:.2f}")
print(f"MSE: {mse_value:.2f}")
print(f"MAPE: {mape_value:.2f}%")
```

Slide 8: Handling Multiple Time Series

NeuralForecast can handle multiple time series simultaneously, which is useful for forecasting related time series or when dealing with hierarchical data.

```python
import pandas as pd
import numpy as np

# Create a multi-series dataset
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
series1 = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
series2 = np.cos(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))

multi_df = pd.DataFrame({
    'date': dates,
    'series_id': np.repeat(['A', 'B'], len(dates)),
    'y': np.concatenate([series1, series2])
})

print(multi_df.head(10))

# Train a model on multiple series
multi_model = NBEATS(input_size=30, forecast_horizon=7)
multi_model.fit(multi_df, val_size=30, epochs=50)

# Generate forecasts for multiple series
multi_forecasts = multi_model.predict(multi_df)
print(multi_forecasts.head(10))
```

Slide 9: Temporal Fusion Transformers (TFT)

The Temporal Fusion Transformer is a powerful model in NeuralForecast that can handle multiple related time series and incorporate static metadata. It's particularly useful for complex forecasting tasks with multiple inputs.

```python
from neuralforecast.models import TFT

# Create a TFT model
tft_model = TFT(
    input_size=30,
    forecast_horizon=7,
    hidden_size=64,
    lstm_layers=2,
    num_attention_heads=4,
    dropout=0.1,
    static_categoricals=['series_id'],
    static_reals=[],
    time_varying_known_reals=['month', 'day'],
    time_varying_unknown_reals=['y']
)

# Add time features to the dataset
multi_df['month'] = multi_df['date'].dt.month
multi_df['day'] = multi_df['date'].dt.day

# Train the TFT model
tft_model.fit(multi_df, val_size=30, epochs=50)

# Generate forecasts
tft_forecasts = tft_model.predict(multi_df)
print(tft_forecasts.head(10))
```

Slide 10: Handling Missing Data

In real-world scenarios, time series data often contains missing values. NeuralForecast provides tools to handle missing data effectively.

```python
import numpy as np

# Introduce missing values
multi_df_missing = multi_df.()
mask = np.random.random(len(multi_df_missing)) > 0.9
multi_df_missing.loc[mask, 'y'] = np.nan

print("Data with missing values:")
print(multi_df_missing[mask].head())

# Train a model with missing data
model_missing = NBEATS(input_size=30, forecast_horizon=7)
model_missing.fit(multi_df_missing, val_size=30, epochs=50)

# Generate forecasts
forecasts_missing = model_missing.predict(multi_df_missing)
print("\nForecasts for data with missing values:")
print(forecasts_missing.head())
```

Slide 11: Ensemble Forecasting

Ensemble methods can often improve forecasting accuracy by combining predictions from multiple models. NeuralForecast supports creating ensembles of different models.

```python
from neuralforecast.models import NHITS, RNN

# Create multiple models
models = [
    NBEATS(input_size=30, forecast_horizon=7),
    NHITS(input_size=30, forecast_horizon=7),
    RNN(input_size=30, forecast_horizon=7, model='LSTM')
]

# Train all models
for model in models:
    model.fit(Y_df, val_size=30, epochs=50)

# Generate forecasts from all models
forecasts = [model.predict(Y_df) for model in models]

# Combine forecasts (simple average)
ensemble_forecast = sum(forecasts) / len(forecasts)

print("Ensemble forecast:")
print(ensemble_forecast.head())
```

Slide 12: Hyperparameter Tuning

Optimizing model hyperparameters can significantly improve forecasting performance. NeuralForecast can be integrated with hyperparameter optimization libraries like Optuna.

```python
import optuna

def objective(trial):
    model = NBEATS(
        input_size=trial.suggest_int('input_size', 7, 30),
        forecast_horizon=7,
        num_blocks=trial.suggest_int('num_blocks', 1, 5),
        num_layers=trial.suggest_int('num_layers', 1, 4),
        layer_widths=trial.suggest_int('layer_widths', 32, 512)
    )
    
    model.fit(Y_df, val_size=30, epochs=50)
    forecasts = model.predict(Y_df)
    return mse(Y_df['y'], forecasts['y'])

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("Best hyperparameters:", study.best_params)
print("Best MSE:", study.best_value)
```

Slide 13: Real-life Example: Sales Forecasting

Let's use NeuralForecast to predict future sales for a retail company. We'll use a dataset containing daily sales data for multiple products.

```python
import pandas as pd
from neuralforecast.models import NBEATS

# Load sales data (assuming we have a CSV file)
sales_data = pd.read_csv('sales_data.csv')
sales_data['date'] = pd.to_datetime(sales_data['date'])

# Create an NBEATS model
sales_model = NBEATS(
    input_size=30,
    forecast_horizon=7,
    stack_types=['trend', 'seasonality'],
    num_blocks=[3, 3],
    num_layers=[4, 4],
    layer_widths=[64, 64]
)

# Train the model
sales_model.fit(sales_data, val_size=30, epochs=100)

# Generate 7-day sales forecast
sales_forecast = sales_model.predict(sales_data)

print("7-day sales forecast:")
print(sales_forecast.tail(7))
```

Slide 14: Real-life Example: Energy Consumption Prediction

In this example, we'll use NeuralForecast to predict energy consumption for a smart grid system. We'll incorporate weather data as exogenous variables to improve our predictions.

```python
import pandas as pd
from neuralforecast.models import TFT

# Load energy consumption and weather data
energy_data = pd.read_csv('energy_data.csv')
energy_data['date'] = pd.to_datetime(energy_data['date'])

# Create a TFT model
energy_model = TFT(
    input_size=24,
    forecast_horizon=12,
    hidden_size=64,
    lstm_layers=2,
    num_attention_heads=4,
    dropout=0.1,
    static_categoricals=['grid_id'],
    static_reals=[],
    time_varying_known_reals=['temperature', 'humidity', 'wind_speed'],
    time_varying_unknown_reals=['consumption']
)

# Train the model
energy_model.fit(energy_data, val_size=24*7, epochs=100)

# Generate 12-hour energy consumption forecast
energy_forecast = energy_model.predict(energy_data)

print("12-hour energy consumption forecast:")
print(energy_forecast.tail(12))
```

Slide 15: Additional Resources

For more information on NeuralForecast and time series forecasting with neural networks, consider exploring these resources:

1. NeuralForecast GitHub repository: [https://github.com/Nixtla/neuralforecast](https://github.com/Nixtla/neuralforecast)
2. "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" by Lim et al. (2019): [https://arxiv.org/abs/1912.09363](https://arxiv.org/abs/1912.09363)
3. "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting" by Oreshkin et al. (2019): [https://arxiv.org/abs/1905.10437](https://arxiv.org/abs/1905.10437)
4. "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks" by Salinas et al. (2017): [https://arxiv.org/abs/1704.04110](https://arxiv.org/abs/1704.04110)

These resources provide in-depth explanations of the underlying algorithms and techniques used in NeuralForecast, as well as best practices for time series forecasting with neural networks.

