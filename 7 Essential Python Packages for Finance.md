## 7 Essential Python Packages for Finance
Slide 1: The Power of QuantLib in Python

QuantLib is a comprehensive quantitative finance library providing tools for derivatives pricing, fixed income analysis, and risk management. Its Python wrapper, QuantLib-Python, enables sophisticated financial modeling with object-oriented design patterns and industry-standard algorithms.

```python
import QuantLib as ql

# Create a European call option
maturity_date = ql.Date().todaysDate() + ql.Period('1Y')
spot_price = 100.0
strike_price = 100.0
dividend_rate = 0.0
risk_free_rate = 0.05
volatility = 0.20

# Set up the pricing environment
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()
calculation_date = ql.Date().todaysDate()
ql.Settings.instance().evaluationDate = calculation_date

# Create option parameters
payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
exercise = ql.EuropeanExercise(maturity_date)
european_option = ql.VanillaOption(payoff, exercise)

# Set up the Black-Scholes process
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
riskfree_handle = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, day_count))
dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, dividend_rate, day_count))
volatility_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, calendar, volatility, day_count))

# Calculate option price
black_scholes_process = ql.BlackScholesProcess(spot_handle, dividend_handle, riskfree_handle, volatility_handle)
european_option.setPricingEngine(ql.AnalyticEuropeanEngine(black_scholes_process))

print(f"Option NPV: {european_option.NPV():.4f}")
print(f"Delta: {european_option.delta():.4f}")
print(f"Gamma: {european_option.gamma():.4f}")
print(f"Vega: {european_option.vega():.4f}")
```

Slide 2: Pandas-ta for Technical Analysis

Pandas-ta extends pandas functionality with over 130 technical analysis indicators. It provides a pythonic interface for calculating moving averages, momentum indicators, volume studies, and volatility measures essential for algorithmic trading.

```python
import pandas as pd
import pandas_ta as ta
import yfinance as yf

# Download historical data
symbol = "AAPL"
df = yf.download(symbol, start="2023-01-01", end="2024-01-01")

# Calculate multiple technical indicators
df.ta.strategy(
    name="technical_analysis",
    ta=[
        {"kind": "sma", "length": 20},
        {"kind": "sma", "length": 50},
        {"kind": "rsi"},
        {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
        {"kind": "bbands", "length": 20}
    ]
)

# Generate trading signals
df['SMA_Signal'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
df['RSI_Signal'] = np.where(df['RSI_14'] < 30, 1, np.where(df['RSI_14'] > 70, -1, 0))

# Calculate returns based on signals
df['Returns'] = df['Close'].pct_change()
df['Strategy_Returns'] = df['Returns'] * df['SMA_Signal'].shift(1)

print(f"Strategy Sharpe Ratio: {(df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252):.2f}")
```

Slide 3: Algorithmic Trading with Zipline

Zipline, originally developed by Quantopian, is a powerful Python library for backtesting trading strategies. It provides a robust event-driven system that simulates market conditions and handles order management with realistic transaction costs and slippage models.

```python
from zipline.api import order_target, record, symbol
from zipline.finance import commission, slippage
import pandas as pd
import numpy as np

def initialize(context):
    context.asset = symbol('AAPL')
    context.set_commission(commission.PerShare(cost=0.001, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    context.sma_short = 50
    context.sma_long = 200
    
def handle_data(context, data):
    # Calculate moving averages
    prices = data.history(context.asset, 'price', context.sma_long + 1, '1d')
    sma_short = prices[-context.sma_short:].mean()
    sma_long = prices[-context.sma_long:].mean()
    
    # Trading logic
    if sma_short > sma_long:
        order_target(context.asset, 100)
    elif sma_short < sma_long:
        order_target(context.asset, -100)
        
    record(short_mavg=sma_short, long_mavg=sma_long, price=prices[-1])

def analyze(context, perf):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('Portfolio Value')
    ax2 = fig.add_subplot(212)
    perf[['short_mavg', 'long_mavg']].plot(ax=ax2)
    perf['price'].plot(ax=ax2)
    ax2.set_ylabel('Price')
```

Slide 4: Financial Analysis with PyFolio

PyFolio is a sophisticated portfolio and risk analytics library that provides comprehensive performance and risk metrics. It integrates seamlessly with Zipline and enables detailed analysis of trading strategies through tear sheets and risk decomposition.

```python
import pyfolio as pf
import pandas as pd
import numpy as np

# Prepare returns data
returns = pd.Series(
    index=pd.date_range('2023-01-01', '2024-01-01'),
    data=np.random.normal(0.001, 0.02, 252)  # Simulated daily returns
)

# Create full tear sheet
pf.create_full_tear_sheet(
    returns,
    benchmark_rets=market_data['spy_returns'],
    positions=positions_df,
    transactions=transactions_df,
    round_trips=True
)

# Calculate performance metrics
stats = pd.Series({
    'Annual Return': pf.annual_return(returns),
    'Cumulative Returns': pf.cum_returns_final(returns),
    'Annual Volatility': pf.annual_volatility(returns),
    'Sharpe Ratio': pf.sharpe_ratio(returns),
    'Sortino Ratio': pf.sortino_ratio(returns),
    'Max Drawdown': pf.max_drawdown(returns),
    'Calmar Ratio': pf.calmar_ratio(returns)
})

print(stats.round(4))
```

Slide 5: Advanced Risk Modeling with PyRisk

PyRisk provides sophisticated tools for calculating Value at Risk (VaR), Expected Shortfall, and other risk metrics using various methodologies including Historical Simulation, Monte Carlo, and Parametric approaches.

```python
import numpy as np
from scipy import stats
import pandas as pd

class RiskMetrics:
    def __init__(self, returns, confidence_level=0.95):
        self.returns = returns
        self.confidence_level = confidence_level
        
    def historical_var(self):
        """Calculate Historical VaR"""
        return np.percentile(self.returns, (1 - self.confidence_level) * 100)
    
    def parametric_var(self):
        """Calculate Parametric VaR"""
        mu = np.mean(self.returns)
        sigma = np.std(self.returns)
        return stats.norm.ppf(1 - self.confidence_level, mu, sigma)
    
    def expected_shortfall(self):
        """Calculate Expected Shortfall (CVaR)"""
        var = self.historical_var()
        return np.mean(self.returns[self.returns <= var])
    
    def monte_carlo_var(self, n_simulations=10000):
        """Calculate Monte Carlo VaR"""
        mu = np.mean(self.returns)
        sigma = np.std(self.returns)
        simulated_returns = np.random.normal(mu, sigma, n_simulations)
        return np.percentile(simulated_returns, (1 - self.confidence_level) * 100)

# Example usage
returns = np.random.normal(0.001, 0.02, 1000)
risk = RiskMetrics(returns)

print(f"Historical VaR: {risk.historical_var():.4f}")
print(f"Parametric VaR: {risk.parametric_var():.4f}")
print(f"Expected Shortfall: {risk.expected_shortfall():.4f}")
print(f"Monte Carlo VaR: {risk.monte_carlo_var():.4f}")
```

Slide 6: Options Analysis with mibian

Mibian is a specialized library for options pricing and analysis that implements various options pricing models including Black-Scholes, Black-76, and their variations for both European and American options.

```python
import mibian

class OptionsAnalyzer:
    def __init__(self, underlying, strike, rate, time, volatility):
        self.underlying = underlying
        self.strike = strike
        self.rate = rate
        self.time = time
        self.volatility = volatility
    
    def black_scholes_european(self):
        """Calculate European option prices and Greeks"""
        bs = mibian.BS([self.underlying, self.strike, self.rate, self.time], 
                      volatility=self.volatility)
        
        return {
            'Call Price': bs.callPrice,
            'Put Price': bs.putPrice,
            'Call Delta': bs.callDelta,
            'Put Delta': bs.putDelta,
            'Gamma': bs.gamma,
            'Vega': bs.vega,
            'Call Theta': bs.callTheta,
            'Put Theta': bs.putTheta,
            'Call Rho': bs.callRho,
            'Put Rho': bs.putRho
        }
    
    def implied_volatility(self, option_price, option_type='call'):
        """Calculate implied volatility"""
        if option_type.lower() == 'call':
            return mibian.BS([self.underlying, self.strike, self.rate, self.time], 
                           callPrice=option_price).impliedVolatility
        else:
            return mibian.BS([self.underlying, self.strike, self.rate, self.time], 
                           putPrice=option_price).impliedVolatility

# Example usage
analyzer = OptionsAnalyzer(
    underlying=100,  # Current price
    strike=100,      # Strike price
    rate=0.05,       # Risk-free rate
    time=1,          # Time to expiration (years)
    volatility=0.20  # Volatility
)

results = analyzer.black_scholes_european()
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# Calculate implied volatility for a call option priced at 10
impl_vol = analyzer.implied_volatility(10, 'call')
print(f"\nImplied Volatility: {impl_vol:.4f}")
```

Slide 7: Backtesting with Backtrader

Backtrader provides a comprehensive framework for developing, testing, and analyzing trading strategies. It offers event-driven operations, multiple data feeds support, and advanced position sizing capabilities with realistic broker simulation.

```python
import backtrader as bt
import datetime

class SmaCrossStrategy(bt.Strategy):
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('risk_percent', 0.02),
    )

    def __init__(self):
        self.fast_sma = bt.indicators.SMA(self.data.close, 
                                         period=self.params.fast_period)
        self.slow_sma = bt.indicators.SMA(self.data.close, 
                                         period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)
        
    def next(self):
        if not self.position:
            if self.crossover > 0:
                price = self.data.close[0]
                cash = self.broker.get_cash()
                size = (cash * self.params.risk_percent) / price
                self.buy(size=size)
        
        elif self.crossover < 0:
            self.close()

# Initialize Cerebro engine
cerebro = bt.Cerebro()

# Add data feed
data = bt.feeds.YahooFinanceData(
    dataname='AAPL',
    fromdate=datetime.datetime(2023, 1, 1),
    todate=datetime.datetime(2024, 1, 1)
)
cerebro.adddata(data)

# Add strategy
cerebro.addstrategy(SmaCrossStrategy)

# Set initial capital
cerebro.broker.setcash(100000.0)

# Add analyzer
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

# Run backtest
results = cerebro.run()
strategy = results[0]

# Print results
print(f"Sharpe Ratio: {strategy.analyzers.sharpe_ratio.get_analysis()['sharperatio']:.2f}")
print(f"Max Drawdown: {strategy.analyzers.drawdown.get_analysis()['max']['drawdown']:.2%}")
print(f"Total Return: {strategy.analyzers.returns.get_analysis()['rtot']:.2%}")
```

Slide 8: Financial Data Analysis with yfinance

The yfinance library provides a reliable and efficient way to download historical market data from Yahoo Finance. It supports multiple assets, various timeframes, and includes fundamental data like financial statements and company information.

```python
import yfinance as yf
import pandas as pd
import numpy as np

class FinancialAnalyzer:
    def __init__(self, ticker):
        self.ticker = yf.Ticker(ticker)
        self.df = self.ticker.history(period="1y")
        
    def calculate_technical_metrics(self):
        # Calculate technical indicators
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Volatility'] = self.df['Returns'].rolling(window=20).std() * np.sqrt(252)
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(window=50).mean()
        self.df['RSI'] = self._calculate_rsi()
        
        return self.df
    
    def get_fundamental_metrics(self):
        # Get fundamental data
        info = self.ticker.info
        financials = self.ticker.financials
        balance_sheet = self.ticker.balance_sheet
        
        metrics = {
            'Market Cap': info.get('marketCap'),
            'PE Ratio': info.get('trailingPE'),
            'EPS': info.get('trailingEps'),
            'ROE': info.get('returnOnEquity'),
            'Revenue Growth': info.get('revenueGrowth'),
            'Debt to Equity': info.get('debtToEquity')
        }
        
        return metrics
    
    def _calculate_rsi(self, periods=14):
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

# Example usage
analyzer = FinancialAnalyzer('AAPL')

# Get technical analysis
tech_data = analyzer.calculate_technical_metrics()
print("\nTechnical Metrics:")
print(tech_data.tail())

# Get fundamental analysis
fund_data = analyzer.get_fundamental_metrics()
print("\nFundamental Metrics:")
for metric, value in fund_data.items():
    print(f"{metric}: {value}")

# Download financial statements
statements = analyzer.ticker.financials
print("\nFinancial Statements:")
print(statements.head())
```

Slide 9: Machine Learning for Finance with scikit-learn

Implementing machine learning models for financial predictions requires careful feature engineering and model validation. This implementation demonstrates a comprehensive pipeline for predicting market movements using various technical indicators.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import yfinance as yf

class MLFinanceModel:
    def __init__(self, symbol, start_date, end_date):
        self.df = yf.download(symbol, start=start_date, end=end_date)
        self.features = None
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def create_features(self):
        df = self.df.copy()
        
        # Technical indicators
        df['Returns'] = df['Close'].pct_change()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA30'] = df['Close'].rolling(window=30).mean()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Create target variable (1 if price goes up, 0 if down)
        df['Target'] = np.where(df['Returns'].shift(-1) > 0, 1, 0)
        
        # Feature columns
        feature_columns = ['Returns', 'MA10', 'MA30', 'Volatility']
        
        # Remove missing values
        df = df.dropna()
        
        self.features = df[feature_columns]
        self.target = df['Target']
        
        return df
    
    def train_model(self):
        # Create time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.features)
        
        # Training
        scores = []
        for train_idx, test_idx in tscv.split(scaled_features):
            X_train = scaled_features[train_idx]
            X_test = scaled_features[test_idx]
            y_train = self.target.iloc[train_idx]
            y_test = self.target.iloc[test_idx]
            
            self.model.fit(X_train, y_train)
            score = self.model.score(X_test, y_test)
            scores.append(score)
            
            print(f"\nFold Results:")
            print(classification_report(y_test, 
                  self.model.predict(X_test)))
        
        print(f"\nAverage Score: {np.mean(scores):.4f}")
        return np.mean(scores)
    
    def predict(self, X_new):
        return self.model.predict_proba(X_new)

# Example usage
model = MLFinanceModel('AAPL', '2022-01-01', '2024-01-01')
data = model.create_features()
accuracy = model.train_model()

# Make predictions for latest data
latest_features = model.features.tail(1)
prediction = model.predict(StandardScaler().fit_transform(latest_features))
print(f"\nProbability of price increase: {prediction[0][1]:.4f}")
```

Slide 10: Time Series Analysis with statsmodels

Statsmodels provides powerful tools for time series analysis, including ARIMA models, decomposition methods, and statistical tests essential for financial forecasting and analysis.

```python
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np

class TimeSeriesAnalyzer:
    def __init__(self, data):
        self.data = data
        
    def decompose_series(self):
        """Decompose time series into trend, seasonal, and residual components"""
        decomposition = seasonal_decompose(self.data, period=20)
        return decomposition
    
    def check_stationarity(self):
        """Perform Augmented Dickey-Fuller test"""
        result = adfuller(self.data.dropna())
        return {
            'Test Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4]
        }
    
    def fit_arima(self, order=(1,1,1)):
        """Fit ARIMA model and make predictions"""
        model = sm.tsa.ARIMA(self.data, order=order)
        results = model.fit()
        
        forecast = results.forecast(steps=30)
        conf_int = results.get_forecast(steps=30).conf_int()
        
        return {
            'Model': results,
            'Forecast': forecast,
            'Confidence Intervals': conf_int,
            'AIC': results.aic,
            'BIC': results.bic
        }
    
    def best_arima_order(self, max_p=3, max_d=2, max_q=3):
        """Find best ARIMA parameters using AIC"""
        best_aic = float('inf')
        best_order = None
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = sm.tsa.ARIMA(self.data, order=(p,d,q))
                        results = model.fit()
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p,d,q)
                    except:
                        continue
        
        return best_order, best_aic

# Example usage
data = yf.download('AAPL')['Close']
analyzer = TimeSeriesAnalyzer(data)

# Check stationarity
stationarity = analyzer.check_stationarity()
print("\nStationarity Test Results:")
for key, value in stationarity.items():
    print(f"{key}: {value}")

# Find best ARIMA order
best_order, best_aic = analyzer.best_arima_order()
print(f"\nBest ARIMA Order: {best_order}")
print(f"Best AIC: {best_aic:.2f}")

# Fit model and forecast
results = analyzer.fit_arima(order=best_order)
print("\nForecast next 30 days:")
print(results['Forecast'])
```

Slide 11: Portfolio Optimization with PyPortfolioOpt

PyPortfolioOpt implements modern portfolio theory for optimal asset allocation. It provides multiple optimization methods including Mean-Variance, Black-Litterman, and Hierarchical Risk Parity approaches.

```python
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation
import pandas as pd
import numpy as np

class PortfolioOptimizer:
    def __init__(self, prices_df):
        self.prices = prices_df
        self.returns = self.prices.pct_change()
        
    def optimize_portfolio(self, risk_free_rate=0.02):
        # Calculate expected returns and sample covariance matrix
        mu = expected_returns.mean_historical_return(self.prices)
        S = risk_models.sample_cov(self.prices)
        
        # Optimize for maximum Sharpe Ratio
        ef = EfficientFrontier(mu, S)
        weights = ef.maximum_sharpe(risk_free_rate=risk_free_rate)
        cleaned_weights = ef.clean_weights()
        
        # Calculate performance metrics
        performance = {
            'Expected Annual Return': ef.portfolio_performance(risk_free_rate=risk_free_rate)[0],
            'Annual Volatility': ef.portfolio_performance(risk_free_rate=risk_free_rate)[1],
            'Sharpe Ratio': ef.portfolio_performance(risk_free_rate=risk_free_rate)[2]
        }
        
        return cleaned_weights, performance
    
    def get_discrete_allocation(self, weights, total_portfolio_value=100000):
        latest_prices = self.prices.iloc[-1]
        da = DiscreteAllocation(weights, latest_prices, 
                              total_portfolio_value=total_portfolio_value)
        allocation, leftover = da.greedy_portfolio()
        
        return allocation, leftover
    
    def efficient_frontier_points(self, points=100):
        mu = expected_returns.mean_historical_return(self.prices)
        S = risk_models.sample_cov(self.prices)
        
        ef = EfficientFrontier(mu, S)
        
        frontier_returns = []
        frontier_volatilities = []
        
        for target_return in np.linspace(0.05, 0.30, points):
            try:
                ef.efficient_return(target_return)
                ret, vol, _ = ef.portfolio_performance()
                frontier_returns.append(ret)
                frontier_volatilities.append(vol)
                ef = EfficientFrontier(mu, S)  # Reset optimization
            except:
                continue
                
        return frontier_returns, frontier_volatilities

# Example usage
# Download historical data for multiple assets
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']
prices_df = pd.DataFrame()

for symbol in symbols:
    prices_df[symbol] = yf.download(symbol, start='2023-01-01')['Close']

optimizer = PortfolioOptimizer(prices_df)

# Optimize portfolio
weights, performance = optimizer.optimize_portfolio()

print("\nOptimal Portfolio Weights:")
for asset, weight in weights.items():
    print(f"{asset}: {weight:.4f}")

print("\nPortfolio Performance:")
for metric, value in performance.items():
    print(f"{metric}: {value:.4f}")

# Get discrete allocation
allocation, leftover = optimizer.get_discrete_allocation(weights)
print("\nDiscrete Allocation:")
for asset, shares in allocation.items():
    print(f"{asset}: {shares} shares")
print(f"Funds remaining: ${leftover:.2f}")
```

Slide 12: Risk-Adjusted Returns Analysis with empyrical

Empyrical provides comprehensive metrics for analyzing trading strategies and portfolio performance, including advanced risk-adjusted return calculations and drawdown analysis.

```python
import empyrical as ep
import pandas as pd
import numpy as np

class PerformanceAnalyzer:
    def __init__(self, returns, benchmark_returns=None):
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        
    def calculate_metrics(self, risk_free=0.0):
        metrics = {
            'Cumulative Returns': ep.cum_returns_final(self.returns),
            'Annual Return': ep.annual_return(self.returns),
            'Annual Volatility': ep.annual_volatility(self.returns),
            'Sharpe Ratio': ep.sharpe_ratio(self.returns, risk_free),
            'Sortino Ratio': ep.sortino_ratio(self.returns, risk_free),
            'Calmar Ratio': ep.calmar_ratio(self.returns),
            'Omega Ratio': ep.omega_ratio(self.returns),
            'Max Drawdown': ep.max_drawdown(self.returns),
            'Value at Risk': self._calculate_var(),
            'Conditional VaR': self._calculate_cvar()
        }
        
        if self.benchmark_returns is not None:
            metrics.update({
                'Alpha': ep.alpha(self.returns, self.benchmark_returns, risk_free),
                'Beta': ep.beta(self.returns, self.benchmark_returns),
                'Information Ratio': ep.information_ratio(self.returns, self.benchmark_returns)
            })
            
        return metrics
    
    def _calculate_var(self, confidence_level=0.95):
        """Calculate Value at Risk"""
        return np.percentile(self.returns, (1 - confidence_level) * 100)
    
    def _calculate_cvar(self, confidence_level=0.95):
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self._calculate_var(confidence_level)
        return self.returns[self.returns <= var].mean()
    
    def rolling_metrics(self, window=252):
        """Calculate rolling performance metrics"""
        rolling_sharpe = ep.roll_sharpe_ratio(self.returns, window=window)
        rolling_sortino = ep.roll_sortino_ratio(self.returns, window=window)
        rolling_beta = ep.roll_beta(self.returns, self.benchmark_returns, 
                                  window=window) if self.benchmark_returns is not None else None
        
        return pd.DataFrame({
            'Rolling Sharpe': rolling_sharpe,
            'Rolling Sortino': rolling_sortino,
            'Rolling Beta': rolling_beta
        })

# Example usage
# Get returns data
returns = yf.download('SPY', start='2023-01-01')['Close'].pct_change().dropna()
benchmark_returns = yf.download('^GSPC', start='2023-01-01')['Close'].pct_change().dropna()

analyzer = PerformanceAnalyzer(returns, benchmark_returns)

# Calculate performance metrics
metrics = analyzer.calculate_metrics()
print("\nPerformance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Calculate rolling metrics
rolling_metrics = analyzer.rolling_metrics()
print("\nRolling Metrics (last 5 periods):")
print(rolling_metrics.tail())
```

Slide 13: Deep Learning for Financial Time Series with PyTorch

PyTorch enables the implementation of sophisticated deep learning models for financial time series prediction. This implementation showcases an LSTM model for multi-step forecasting with attention mechanism.

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(AttentionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Self-attention mechanism
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last output for prediction
        out = self.fc(attn_output[:, -1, :])
        return out

class FinancialPredictor:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        
    def prepare_data(self, data):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        sequences = []
        targets = []
        
        for i in range(len(scaled_data) - self.sequence_length):
            seq = scaled_data[i:i + self.sequence_length]
            target = scaled_data[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)
            
        return torch.FloatTensor(sequences), torch.FloatTensor(targets)
    
    def train_model(self, train_data, epochs=100, learning_rate=0.01):
        X_train, y_train = self.prepare_data(train_data)
        
        self.model = AttentionLSTM(
            input_dim=1,
            hidden_dim=32,
            num_layers=2,
            output_dim=1
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    def predict(self, test_sequence):
        self.model.eval()
        with torch.no_grad():
            scaled_sequence = self.scaler.transform(test_sequence.reshape(-1, 1))
            X_test = torch.FloatTensor(scaled_sequence).unsqueeze(0)
            prediction = self.model(X_test)
            return self.scaler.inverse_transform(prediction.numpy())

# Example usage
# Get financial data
data = yf.download('AAPL', start='2023-01-01')['Close'].values

# Split data
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Initialize and train model
predictor = FinancialPredictor()
predictor.train_model(train_data)

# Make predictions
test_sequence = data[-20:]  # Last 20 days
prediction = predictor.predict(test_sequence)
print(f"\nNext day prediction: ${prediction[0][0]:.2f}")
```

Slide 14: Additional Resources

*   A Survey of Deep Learning Models for Time Series Forecasting [https://arxiv.org/abs/2103.07946](https://arxiv.org/abs/2103.07946)
*   Deep Learning for Financial Applications: A Survey [https://arxiv.org/abs/2002.05786](https://arxiv.org/abs/2002.05786)
*   Modern Portfolio Theory: A Review [https://arxiv.org/abs/1606.07432](https://arxiv.org/abs/1606.07432)
*   Machine Learning for Trading: A Systematic Literature Review [https://arxiv.org/abs/2106.12300](https://arxiv.org/abs/2106.12300)
*   Deep Reinforcement Learning in Financial Markets [https://arxiv.org/abs/2004.06627](https://arxiv.org/abs/2004.06627)

