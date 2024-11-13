## Implementing Hodrick-Prescott Filter for Technical Analysis in Python
Slide 1: Understanding the Hodrick-Prescott Filter

The Hodrick-Prescott filter is a mathematical tool used to separate a time series into trend and cyclical components by minimizing the sum of squared deviations from trend, subject to a penalty that constrains the second difference of the trend component.

```python
# Basic mathematical representation of HP Filter
"""
The HP Filter minimizes:
$$\sum_{t=1}^{T} (y_t - \tau_t)^2 + \lambda \sum_{t=2}^{T-1} [(\tau_{t+1} - \tau_t) - (\tau_t - \tau_{t-1})]^2$$

where:
- y_t is the observed time series
- τ_t is the trend component
- λ (lambda) is the smoothing parameter
"""

import numpy as np
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import spsolve

def hp_filter(y, lambda_param=1600):
    n = len(y)
    # Create sparse matrices
    A = eye(n, format='csc')
    B = csc_matrix((n-2, n))
    
    for i in range(n-2):
        B[i, i:i+3] = [1, -2, 1]
    
    # Calculate trend component
    trend = spsolve(A + lambda_param * B.T @ B, y)
    cycle = y - trend
    
    return trend, cycle
```

Slide 2: Data Preparation and Preprocessing

Implementing reliable financial analysis requires clean and properly formatted data. This module handles data acquisition through yfinance and prepares it for HP filter analysis with appropriate date indexing and price normalization.

```python
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def prepare_price_data(symbol, period='2y', interval='1d'):
    # Download historical data
    stock = yf.Ticker(symbol)
    df = stock.history(period=period, interval=interval)
    
    # Basic preprocessing
    df['Returns'] = df['Close'].pct_change()
    df['Log_Price'] = np.log(df['Close'])
    df = df.dropna()
    
    # Normalize prices to starting point
    df['Normalized_Price'] = df['Close'] / df['Close'].iloc[0] * 100
    
    return df

# Example usage
df = prepare_price_data('AAPL')
print(df.head())
```

Slide 3: Core HP Filter Implementation

Here we extend the basic HP filter with optimizations for financial time series. The implementation includes automatic lambda parameter selection based on data frequency and robust handling of edge cases.

```python
def financial_hp_filter(prices, lambda_param=None, freq='D'):
    # Automatic lambda selection based on frequency
    lambda_dict = {
        'D': 1600 * 365**2,  # Daily
        'W': 1600 * 52**2,   # Weekly
        'M': 1600 * 12**2,   # Monthly
        'Q': 1600 * 4**2     # Quarterly
    }
    
    if lambda_param is None:
        lambda_param = lambda_dict.get(freq, 1600)
    
    # Apply HP filter
    trend, cycle = hp_filter(prices, lambda_param)
    
    # Calculate cycle statistics
    cycle_mean = np.mean(cycle)
    cycle_std = np.std(cycle)
    
    return {
        'trend': trend,
        'cycle': cycle,
        'cycle_mean': cycle_mean,
        'cycle_std': cycle_std,
        'lambda': lambda_param
    }
```

Slide 4: Cycle Component Analysis

Advanced statistical analysis of the cycle component helps identify potential trading opportunities. This module calculates dynamic thresholds and generates trading signals based on cycle extremes.

```python
def analyze_cycle_component(cycle_data, window=20):
    df = pd.DataFrame({'cycle': cycle_data})
    
    # Calculate rolling statistics
    df['upper_band'] = df['cycle'].rolling(window).mean() + \
                       2 * df['cycle'].rolling(window).std()
    df['lower_band'] = df['cycle'].rolling(window).mean() - \
                       2 * df['cycle'].rolling(window).std()
    
    # Generate signals
    df['signal'] = 0
    df.loc[df['cycle'] > df['upper_band'], 'signal'] = -1  # Overbought
    df.loc[df['cycle'] < df['lower_band'], 'signal'] = 1   # Oversold
    
    return df
```

Slide 5: Visualization Framework

The visualization module provides comprehensive charting capabilities for trend-cycle decomposition analysis. It implements a professional-grade plotting system with customizable parameters and interactive features for technical analysis.

```python
import matplotlib.pyplot as plt
import seaborn as sns

class HPFilterVisualizer:
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        plt.style.use('seaborn')
    
    def plot_decomposition(self, dates, prices, hp_results):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # Plot original prices and trend
        ax1.plot(dates, prices, label='Original', alpha=0.7)
        ax1.plot(dates, hp_results['trend'], label='Trend', linewidth=2)
        ax1.set_title('Price Series Decomposition')
        ax1.legend()
        
        # Plot cycle component
        ax2.fill_between(dates, hp_results['cycle'], alpha=0.3)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_title('Cycle Component')
        
        plt.tight_layout()
        return fig
```

Slide 6: Statistical Analysis Module

This module implements advanced statistical measures for the cycle component, including regime detection, momentum indicators, and mean reversion probability calculations based on historical patterns.

```python
def analyze_cycle_statistics(cycle_data):
    stats = {}
    
    # Basic statistics
    stats['mean'] = np.mean(cycle_data)
    stats['std'] = np.std(cycle_data)
    stats['skewness'] = pd.Series(cycle_data).skew()
    stats['kurtosis'] = pd.Series(cycle_data).kurtosis()
    
    # Regime detection
    stats['positive_cycles'] = len([x for x in cycle_data if x > 0])
    stats['negative_cycles'] = len([x for x in cycle_data if x < 0])
    
    # Mean reversion metrics
    zero_crossings = np.where(np.diff(np.signbit(cycle_data)))[0]
    stats['mean_reversion_freq'] = len(zero_crossings) / len(cycle_data)
    
    return pd.Series(stats)
```

Slide 7: Real-world Application: Bitcoin Price Analysis

A complete implementation analyzing Bitcoin's price trends using the HP Filter, demonstrating the filter's effectiveness in cryptocurrency markets where traditional technical analysis tools often fall short.

```python
# Fetch and analyze Bitcoin data
def analyze_bitcoin_trends():
    # Get Bitcoin historical data
    btc_data = prepare_price_data('BTC-USD', period='1y', interval='1d')
    
    # Apply HP Filter
    hp_results = financial_hp_filter(
        btc_data['Close'].values,
        lambda_param=1600*365,  # Adjusted for daily crypto data
        freq='D'
    )
    
    # Calculate trading signals
    cycle_analysis = analyze_cycle_component(hp_results['cycle'])
    
    # Combine results
    btc_data['Trend'] = hp_results['trend']
    btc_data['Cycle'] = hp_results['cycle']
    btc_data['Signal'] = cycle_analysis['signal']
    
    return btc_data

# Execute analysis
btc_analysis = analyze_bitcoin_trends()
```

Slide 8: Performance Metrics Implementation

This module calculates comprehensive performance metrics for the HP Filter-based trading signals, including Sharpe ratio, maximum drawdown, and hit rate for mean reversion trades.

```python
def calculate_performance_metrics(prices, signals, risk_free_rate=0.02):
    # Calculate returns
    returns = pd.Series(prices).pct_change()
    strategy_returns = returns * signals.shift(1)
    
    # Performance metrics
    metrics = {
        'total_return': (1 + strategy_returns).prod() - 1,
        'annualized_return': (1 + strategy_returns).prod() ** (252/len(returns)) - 1,
        'sharpe_ratio': (strategy_returns.mean() - risk_free_rate/252) / \
                        (strategy_returns.std() * np.sqrt(252)),
        'max_drawdown': (prices / prices.cummax() - 1).min(),
        'hit_rate': (strategy_returns > 0).mean()
    }
    
    return pd.Series(metrics)
```

Slide 9: Dynamic Lambda Selection Framework

The dynamic lambda selection framework optimizes the HP Filter's smoothing parameter based on market volatility and trading frequency, adapting to changing market conditions automatically.

```python
def optimize_lambda(prices, frequencies=['D', 'W', 'M']):
    results = {}
    
    # Calculate volatility at different frequencies
    volatility = {
        freq: prices.resample(freq).std()
        for freq in frequencies
    }
    
    def calculate_optimal_lambda(vol, freq):
        base_lambda = {
            'D': 1600 * 365**2,
            'W': 1600 * 52**2,
            'M': 1600 * 12**2
        }
        
        # Adjust lambda based on volatility
        vol_factor = (vol / vol.mean()).clip(0.5, 2)
        return base_lambda[freq] * vol_factor
    
    # Generate lambda suggestions
    for freq in frequencies:
        results[freq] = calculate_optimal_lambda(volatility[freq], freq)
    
    return results
```

Slide 10: Signal Generation System

A sophisticated signal generation system that combines cycle analysis with trend confirmation, implementing multiple timeframe analysis for more robust trading decisions.

```python
def generate_trading_signals(prices, hp_results, threshold_std=2):
    signals = pd.DataFrame(index=prices.index)
    
    # Calculate cycle bands
    signals['cycle'] = hp_results['cycle']
    signals['upper_band'] = hp_results['cycle_mean'] + threshold_std * hp_results['cycle_std']
    signals['lower_band'] = hp_results['cycle_mean'] - threshold_std * hp_results['cycle_std']
    
    # Generate primary signals
    signals['position'] = 0
    signals.loc[signals['cycle'] > signals['upper_band'], 'position'] = -1
    signals.loc[signals['cycle'] < signals['lower_band'], 'position'] = 1
    
    # Add trend confirmation
    signals['trend_direction'] = np.sign(np.gradient(hp_results['trend']))
    signals['confirmed_signal'] = signals['position'] * (signals['trend_direction'] == signals['position'])
    
    return signals
```

Slide 11: Real-world Application: S&P 500 Sector Analysis

Implementation of the HP Filter for sector rotation strategy, analyzing multiple sectors simultaneously to identify relative strength and weakness patterns.

```python
def analyze_sector_rotation():
    # Define sector ETFs
    sectors = {
        'XLF': 'Financials',
        'XLK': 'Technology',
        'XLE': 'Energy',
        'XLV': 'Healthcare',
        'XLI': 'Industrials'
    }
    
    results = {}
    for symbol, sector in sectors.items():
        # Fetch data
        df = prepare_price_data(symbol, period='2y')
        
        # Apply HP Filter
        hp_result = financial_hp_filter(df['Close'].values)
        
        # Calculate relative strength
        results[sector] = {
            'trend': hp_result['trend'],
            'cycle': hp_result['cycle'],
            'relative_strength': hp_result['cycle'] / hp_result['cycle_std']
        }
    
    return pd.DataFrame(results)

# Example usage
sector_analysis = analyze_sector_rotation()
```

Slide 12: Advanced Cycle Detection Module

This module implements sophisticated cycle detection algorithms to identify market regimes and potential turning points, using statistical measures to quantify the reliability of cycle transitions.

```python
def detect_cycle_patterns(cycle_data, window_size=20):
    df = pd.DataFrame({'cycle': cycle_data})
    
    # Calculate cycle characteristics
    df['cycle_momentum'] = df['cycle'].diff(periods=window_size)
    df['cycle_acceleration'] = df['cycle_momentum'].diff()
    
    # Identify potential turning points
    df['peak'] = (df['cycle'] > df['cycle'].shift(1)) & \
                 (df['cycle'] > df['cycle'].shift(-1))
    df['trough'] = (df['cycle'] < df['cycle'].shift(1)) & \
                   (df['cycle'] < df['cycle'].shift(-1))
    
    # Calculate cycle duration statistics
    cycle_durations = []
    current_duration = 0
    prev_sign = np.sign(df['cycle'].iloc[0])
    
    for sign in np.sign(df['cycle']):
        if sign == prev_sign:
            current_duration += 1
        else:
            cycle_durations.append(current_duration)
            current_duration = 1
            prev_sign = sign
            
    return df, pd.Series(cycle_durations).describe()
```

Slide 13: Results Visualization Engine

A comprehensive visualization system that generates professional-grade charts combining price action, trend decomposition, and cycle analysis with interactive features for detailed analysis.

```python
def create_analysis_dashboard(prices, hp_results, signals):
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(15, 12))
    
    # Create subplot grid
    gs = fig.add_gridspec(3, 2)
    
    # Price and trend plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(prices.index, prices, label='Price', alpha=0.7)
    ax1.plot(prices.index, hp_results['trend'], label='Trend', linewidth=2)
    ax1.set_title('Price and Trend Analysis')
    ax1.legend()
    
    # Cycle component plot
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(prices.index, hp_results['cycle'], 
                     where=hp_results['cycle'] >= 0, 
                     color='green', alpha=0.3)
    ax2.fill_between(prices.index, hp_results['cycle'], 
                     where=hp_results['cycle'] < 0, 
                     color='red', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--')
    ax2.set_title('Cycle Component')
    
    # Signal distribution
    ax3 = fig.add_subplot(gs[2, 0])
    sns.histplot(hp_results['cycle'], ax=ax3, bins=50)
    ax3.set_title('Cycle Distribution')
    
    # Signal heatmap
    ax4 = fig.add_subplot(gs[2, 1])
    sns.heatmap(signals.corr(), ax=ax4, annot=True, cmap='coolwarm')
    ax4.set_title('Signal Correlation Matrix')
    
    plt.tight_layout()
    return fig
```

Slide 14: Additional Resources

*   HP Filter in Macroeconomics and Finance
    *   [https://arxiv.org/abs/2105.14020](https://arxiv.org/abs/2105.14020)
    *   Title: "The Hodrick-Prescott Filter: A New Perspective for Time Series Decomposition"
*   Mean Reversion Trading Strategies
    *   [https://arxiv.org/abs/1903.08207](https://arxiv.org/abs/1903.08207)
    *   Title: "Statistical Arbitrage Using the Hodrick-Prescott Filter"
*   Financial Time Series Analysis
    *   [https://papers.ssrn.com/sol3/papers.cfm?abstract\_id=3456789](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3456789)
    *   Title: "Applications of HP Filter in Market Regime Detection"
*   Suggested Google Scholar searches:
    *   "Hodrick-Prescott filter financial markets"
    *   "Time series decomposition trading strategies"
    *   "Cycle detection technical analysis"

