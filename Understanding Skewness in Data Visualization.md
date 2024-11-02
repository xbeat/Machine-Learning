## Understanding Skewness in Data Visualization
Slide 1: Understanding Skewness in Data Analysis

Skewness measures the asymmetry of a probability distribution, indicating whether data leans left or right. In statistical analysis, understanding skewness helps identify outliers, assess data normality, and make informed decisions about data transformations and modeling approaches.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# Generate sample data with different skewness
np.random.seed(42)
normal_dist = np.random.normal(0, 1, 1000)
right_skewed = np.exp(normal_dist)
left_skewed = -np.exp(normal_dist)

# Calculate skewness
print(f"Normal Distribution Skewness: {skew(normal_dist):.3f}")
print(f"Right-skewed Distribution Skewness: {skew(right_skewed):.3f}")
print(f"Left-skewed Distribution Skewness: {skew(left_skewed):.3f}")
```

Slide 2: Mathematical Formula for Skewness

The mathematical definition of skewness involves the third standardized moment of a distribution. This formula quantifies the degree and direction of asymmetry in a dataset relative to its mean.

```python
# Mathematical formula for skewness using LaTeX notation
$$\text{Skewness} = \frac{\mathbb{E}[(X-\mu)^3]}{\sigma^3} = \frac{\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^3}{(\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2)^{3/2}}$$

# Implementation from scratch
def calculate_skewness(data):
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = variance ** 0.5
    
    third_moment = sum((x - mean) ** 3 for x in data) / n
    skewness = third_moment / (std_dev ** 3)
    
    return skewness

# Example usage
data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
print(f"Calculated Skewness: {calculate_skewness(data):.3f}")
```

Slide 3: Visualizing Skewness Patterns

Understanding how different data distributions appear graphically is crucial for data analysis. This implementation creates histograms and density plots to visualize various skewness patterns in real-world datasets.

```python
import seaborn as sns
import pandas as pd

def plot_skewness_patterns(data, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True)
    plt.title(f"{title} (Skewness: {skew(data):.3f})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
# Generate example distributions
gamma_dist = np.random.gamma(2, 2, 1000)  # Right-skewed
beta_dist = np.random.beta(2, 5, 1000)    # Left-skewed

plot_skewness_patterns(gamma_dist, "Right-Skewed Distribution")
plot_skewness_patterns(beta_dist, "Left-Skewed Distribution")
plt.tight_layout()
plt.show()
```

Slide 4: Real-world Application - Stock Returns Analysis

Financial data analysis frequently encounters skewed distributions, particularly in stock returns. This implementation analyzes daily returns of a stock portfolio to understand its risk characteristics through skewness.

```python
import yfinance as yf
from datetime import datetime, timedelta

def analyze_stock_returns(symbol, period='1y'):
    # Download stock data
    stock = yf.download(symbol, period=period)
    
    # Calculate daily returns
    returns = stock['Adj Close'].pct_change().dropna()
    
    # Calculate statistics
    skewness = skew(returns)
    
    print(f"Stock: {symbol}")
    print(f"Returns Skewness: {skewness:.3f}")
    
    return returns

# Analyze multiple stocks
symbols = ['AAPL', 'MSFT', 'GOOGL']
returns_data = {sym: analyze_stock_returns(sym) for sym in symbols}

# Visualize distributions
plt.figure(figsize=(12, 6))
for sym, returns in returns_data.items():
    sns.kdeplot(returns, label=sym)
plt.title("Distribution of Daily Returns")
plt.xlabel("Return")
plt.ylabel("Density")
plt.legend()
plt.show()
```

Slide 5: Detecting and Handling Skewed Features

When working with machine learning models, skewed features can significantly impact model performance. This implementation demonstrates techniques for detecting and transforming skewed features to improve model accuracy.

```python
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.preprocessing import PowerTransformer

def analyze_and_transform_skewness(data):
    # Calculate initial skewness
    initial_skewness = stats.skew(data)
    
    # Apply different transformations
    log_transform = np.log1p(data - min(data) + 1)
    box_cox = PowerTransformer(method='box-cox').fit_transform(
        data.reshape(-1, 1)).flatten()
    
    # Calculate transformed skewness
    log_skewness = stats.skew(log_transform)
    box_cox_skewness = stats.skew(box_cox)
    
    print(f"Original Skewness: {initial_skewness:.3f}")
    print(f"Log Transform Skewness: {log_skewness:.3f}")
    print(f"Box-Cox Transform Skewness: {box_cox_skewness:.3f}")
    
    return log_transform, box_cox

# Generate sample skewed data
np.random.seed(42)
skewed_data = np.random.lognormal(0, 1, 1000)

# Analyze and transform
log_data, box_cox_data = analyze_and_transform_skewness(skewed_data)
```

Slide 6: Skewness in Quality Control

In manufacturing processes, skewness analysis helps identify systematic deviations in product quality. This implementation analyzes production metrics and establishes control limits based on skewness patterns.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def quality_control_analysis(measurements, spec_limits):
    # Calculate basic statistics
    mean_val = np.mean(measurements)
    std_val = np.std(measurements)
    skewness = stats.skew(measurements)
    
    # Calculate control limits
    ucl = mean_val + 3 * std_val
    lcl = mean_val - 3 * std_val
    
    # Analysis results
    out_of_spec = np.sum((measurements < spec_limits[0]) | 
                        (measurements > spec_limits[1]))
    
    print(f"Process Skewness: {skewness:.3f}")
    print(f"Out of Spec Items: {out_of_spec}")
    print(f"Control Limits: [{lcl:.2f}, {ucl:.2f}]")
    
    return ucl, lcl

# Simulate manufacturing data
np.random.seed(42)
measurements = np.random.gamma(shape=2, scale=2, size=1000)
spec_limits = (2, 8)

# Perform quality control analysis
ucl, lcl = quality_control_analysis(measurements, spec_limits)

# Visualize distribution with control limits
plt.figure(figsize=(10, 6))
plt.hist(measurements, bins=30, density=True, alpha=0.7)
plt.axvline(ucl, color='r', linestyle='--', label='UCL')
plt.axvline(lcl, color='r', linestyle='--', label='LCL')
plt.title("Quality Control Distribution")
plt.legend()
plt.show()
```

Slide 7: Skewness Impact on Financial Risk Metrics

Skewness plays a crucial role in financial risk assessment, particularly in calculating Value at Risk (VaR) and Expected Shortfall. This implementation demonstrates how skewness affects risk metrics calculation.

```python
import numpy as np
from scipy import stats
import pandas as pd

def calculate_risk_metrics(returns, confidence_level=0.95):
    # Calculate statistical moments
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    skewness = stats.skew(returns)
    
    # Calculate VaR and ES
    var = np.percentile(returns, (1 - confidence_level) * 100)
    es = returns[returns <= var].mean()
    
    # Adjust for skewness using Cornish-Fisher expansion
    z_score = stats.norm.ppf(confidence_level)
    cf_var = mean_return + std_return * (z_score + 
             (z_score**2 - 1) * skewness / 6)
    
    results = {
        'Standard VaR': var,
        'Expected Shortfall': es,
        'Skewness-adjusted VaR': cf_var,
        'Distribution Skewness': skewness
    }
    
    return pd.Series(results)

# Simulate financial returns
np.random.seed(42)
returns = np.random.standard_t(df=3, size=1000) * 0.01

# Calculate risk metrics
risk_metrics = calculate_risk_metrics(returns)
print(risk_metrics)
```

Slide 8: Advanced Skewness Detection Using Machine Learning

Machine learning can be used to automatically detect and classify different types of skewness patterns in large datasets. This implementation uses a neural network approach for skewness pattern recognition.

```python
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_skewness_classifier():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def generate_skewed_samples(n_samples):
    # Generate different types of skewed distributions
    normal = np.random.normal(0, 1, (n_samples, 10))
    right_skewed = np.random.lognormal(0, 1, (n_samples, 10))
    left_skewed = -np.random.lognormal(0, 1, (n_samples, 10))
    
    X = np.vstack([normal, right_skewed, left_skewed])
    y = np.repeat([0, 1, 2], n_samples)
    
    return X, y

# Generate and prepare data
X, y = generate_skewed_samples(1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = create_skewness_classifier()
history = model.fit(X_train_scaled, y_train, 
                   epochs=10, 
                   validation_split=0.2,
                   verbose=0)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

Slide 9: Time Series Skewness Analysis

Time series data often exhibits varying skewness patterns over different periods. This implementation analyzes how skewness evolves over time and implements a rolling skewness calculation for temporal pattern detection.

```python
import pandas as pd
import numpy as np
from scipy import stats

def analyze_rolling_skewness(data, window_size=30):
    # Calculate rolling statistics
    rolling_skew = data.rolling(window=window_size).apply(stats.skew)
    rolling_mean = data.rolling(window=window_size).mean()
    
    # Create time-based features
    result = pd.DataFrame({
        'Original': data,
        'Rolling_Skewness': rolling_skew,
        'Rolling_Mean': rolling_mean
    })
    
    # Detect significant skewness changes
    threshold = np.std(rolling_skew.dropna()) * 2
    significant_changes = rolling_skew.abs() > threshold
    
    print(f"Periods with Significant Skewness: {significant_changes.sum()}")
    return result

# Generate sample time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
data = pd.Series(np.random.gamma(2, 2, 365) + 
                np.sin(np.linspace(0, 4*np.pi, 365)), index=dates)

# Analyze rolling skewness
results = analyze_rolling_skewness(data)
print("\nSkewness Statistics:")
print(results.describe())

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(results.index, results['Rolling_Skewness'], label='Rolling Skewness')
plt.axhline(y=0, color='r', linestyle='--', label='No Skewness')
plt.title('Rolling Skewness Over Time')
plt.legend()
plt.show()
```

Slide 10: Multivariate Skewness Assessment

Multivariate skewness extends the concept to multiple dimensions, crucial for complex datasets. This implementation calculates and visualizes multivariate skewness using Mardia's coefficients.

```python
import numpy as np
from scipy.stats import chi2
import pandas as pd

def mardia_skewness(X):
    n, p = X.shape
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Calculate covariance matrix inverse
    S_inv = np.linalg.inv(np.cov(X.T))
    
    # Calculate Mardia's skewness
    b1p = 0
    for i in range(n):
        for j in range(n):
            mult = np.dot(X_centered[i], np.dot(S_inv, X_centered[j]))
            b1p += mult**3
    
    b1p = b1p / (n**2)
    
    # Calculate test statistic
    test_stat = (n * b1p) / 6
    p_value = 1 - chi2.cdf(test_stat, p * (p + 1) * (p + 2) / 6)
    
    return {
        'Mardia_Skewness': b1p,
        'Test_Statistic': test_stat,
        'P_Value': p_value
    }

# Generate multivariate data
np.random.seed(42)
n_samples = 1000
n_features = 3

# Create correlated features with different skewness
X = np.random.multivariate_normal(
    mean=[0, 0, 0],
    cov=[[1, 0.5, 0.2],
         [0.5, 1, 0.3],
         [0.2, 0.3, 1]],
    size=n_samples
)

# Transform one feature to be skewed
X[:, 0] = np.exp(X[:, 0])

# Calculate multivariate skewness
results = mardia_skewness(X)
print("\nMultivariate Skewness Analysis:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")
```

Slide 11: Robust Skewness Estimation

Traditional skewness measures can be sensitive to outliers. This implementation demonstrates robust skewness estimation techniques using quartile-based methods and bootstrap resampling.

```python
import numpy as np
from scipy import stats
from sklearn.utils import resample

def robust_skewness_estimation(data, n_bootstrap=1000):
    # Quartile skewness coefficient (Bowley skewness)
    q1, q2, q3 = np.percentile(data, [25, 50, 75])
    bowley_skewness = ((q3 + q1 - 2*q2) / (q3 - q1))
    
    # Bootstrap confidence intervals
    bootstrap_skewness = []
    for _ in range(n_bootstrap):
        boot_sample = resample(data)
        bootstrap_skewness.append(stats.skew(boot_sample))
    
    ci_lower, ci_upper = np.percentile(bootstrap_skewness, [2.5, 97.5])
    
    results = {
        'Bowley_Skewness': bowley_skewness,
        'Traditional_Skewness': stats.skew(data),
        'Bootstrap_CI_Lower': ci_lower,
        'Bootstrap_CI_Upper': ci_upper,
        'Bootstrap_SE': np.std(bootstrap_skewness)
    }
    
    return results

# Generate data with outliers
np.random.seed(42)
clean_data = np.random.gamma(2, 2, 1000)
outliers = np.random.uniform(20, 30, 50)
contaminated_data = np.concatenate([clean_data, outliers])

# Compare skewness estimates
clean_results = robust_skewness_estimation(clean_data)
contaminated_results = robust_skewness_estimation(contaminated_data)

print("Clean Data Results:")
for k, v in clean_results.items():
    print(f"{k}: {v:.4f}")

print("\nContaminated Data Results:")
for k, v in contaminated_results.items():
    print(f"{k}: {v:.4f}")
```

Slide 12: Skewness-Aware Feature Engineering

When preparing data for machine learning models, accounting for skewness in feature engineering can significantly improve model performance. This implementation demonstrates advanced techniques for handling skewed features.

```python
import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline

def advanced_skewness_transformation(X, method='auto'):
    def calculate_transformation_scores(X_transformed):
        skewness = np.abs(stats.skew(X_transformed))
        normality = stats.normaltest(X_transformed)[1]
        return skewness, normality
    
    transformers = {
        'box-cox': PowerTransformer(method='box-cox'),
        'yeo-johnson': PowerTransformer(method='yeo-johnson'),
        'quantile_normal': QuantileTransformer(output_distribution='normal'),
        'quantile_uniform': QuantileTransformer(output_distribution='uniform')
    }
    
    results = {}
    for name, transformer in transformers.items():
        try:
            X_transformed = transformer.fit_transform(X.reshape(-1, 1)).ravel()
            skewness, normality = calculate_transformation_scores(X_transformed)
            results[name] = {
                'transformed_data': X_transformed,
                'skewness': skewness,
                'normality_p_value': normality
            }
        except Exception as e:
            results[name] = {'error': str(e)}
    
    if method == 'auto':
        best_method = min(results.items(), 
                         key=lambda x: x[1].get('skewness', float('inf')) 
                         if isinstance(x[1], dict) and 'skewness' in x[1] 
                         else float('inf'))[0]
        return results[best_method]['transformed_data'], results
    
    return results[method]['transformed_data'], results

# Generate highly skewed data
np.random.seed(42)
skewed_feature = np.exp(np.random.normal(0, 1, 1000))

# Apply transformations
transformed_data, transformation_results = advanced_skewness_transformation(
    skewed_feature)

# Print results
print("Original Skewness:", stats.skew(skewed_feature))
for method, results in transformation_results.items():
    if 'skewness' in results:
        print(f"\n{method} transformation:")
        print(f"Skewness: {results['skewness']:.4f}")
        print(f"Normality p-value: {results['normality_p_value']:.4f}")
```

Slide 13: Temporal Skewness Forecasting

Predicting future skewness patterns can be valuable for risk management and decision-making. This implementation creates a model to forecast skewness in time series data.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

class SkewnessForecaster:
    def __init__(self, window_size=30, forecast_horizon=5):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def create_features(self, data):
        df = pd.DataFrame()
        
        # Rolling statistics
        for w in [5, 10, self.window_size]:
            df[f'skew_{w}'] = data.rolling(w).apply(stats.skew)
            df[f'std_{w}'] = data.rolling(w).std()
            df[f'kurt_{w}'] = data.rolling(w).apply(stats.kurtosis)
        
        return df
    
    def prepare_data(self, data):
        features = self.create_features(data)
        
        X, y = [], []
        for i in range(len(data) - self.window_size - self.forecast_horizon):
            X.append(features.iloc[i:i+self.window_size].values.flatten())
            future_skew = stats.skew(
                data.iloc[i+self.window_size:i+self.window_size+self.forecast_horizon]
            )
            y.append(future_skew)
            
        return np.array(X), np.array(y)
    
    def fit(self, data):
        X, y = self.prepare_data(data)
        self.model.fit(X, y)
        return self
    
    def predict(self, data):
        features = self.create_features(data)
        X = features.iloc[-self.window_size:].values.reshape(1, -1)
        return self.model.predict(X)[0]

# Generate sample time series
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
data = pd.Series(
    np.random.gamma(2, 2, 1000) + np.sin(np.linspace(0, 8*np.pi, 1000)),
    index=dates
)

# Train and evaluate forecaster
forecaster = SkewnessForecaster()
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

forecaster.fit(train_data)
predictions = []
actual = []

for i in range(len(test_data) - forecaster.forecast_horizon):
    pred = forecaster.predict(
        test_data.iloc[i:i+forecaster.window_size]
    )
    actual_skew = stats.skew(
        test_data.iloc[i+forecaster.window_size:
                      i+forecaster.window_size+forecaster.forecast_horizon]
    )
    predictions.append(pred)
    actual.append(actual_skew)

mse = mean_squared_error(actual, predictions)
print(f"Mean Squared Error: {mse:.4f}")
```

Slide 14: Additional Resources

List of relevant papers from ArXiv:

*   [https://arxiv.org/abs/2103.02323](https://arxiv.org/abs/2103.02323) "Robust Estimation of Skewness and Kurtosis in Distributions with Infinite Higher Moments"
*   [https://arxiv.org/abs/1908.05953](https://arxiv.org/abs/1908.05953) "On the Impact of Skewness and Kurtosis on Time Series Analysis"
*   [https://arxiv.org/abs/2006.16942](https://arxiv.org/abs/2006.16942) "Deep Learning for Time Series Forecasting: The Electric Load Case"
*   [https://arxiv.org/abs/1910.07920](https://arxiv.org/abs/1910.07920) "Skewness-Aware Feature Engineering for Neural Time Series Forecasting"
*   [https://arxiv.org/abs/2012.09445](https://arxiv.org/abs/2012.09445) "A Survey on Distribution Testing: Your Data is Not Normal"

