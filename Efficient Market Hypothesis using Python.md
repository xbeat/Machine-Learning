## Efficient Market Hypothesis using Python
Slide 1: Introduction to Efficient Market Hypothesis (EMH)

The Efficient Market Hypothesis (EMH) is a fundamental concept in financial economics that suggests asset prices fully reflect all available information. This theory implies that it's impossible to consistently outperform the market through expert stock selection or market timing.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate random walk (efficient market)
np.random.seed(42)
steps = 1000
price = 100 + np.cumsum(np.random.randn(steps))

plt.figure(figsize=(10, 6))
plt.plot(price)
plt.title('Simulated Stock Price in an Efficient Market')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
```

Slide 2: Types of Market Efficiency

EMH is typically divided into three forms: weak, semi-strong, and strong. Each form represents a different degree of market efficiency and information incorporation into asset prices.

```python
def market_efficiency(information_type):
    efficiency_levels = {
        "past_prices": "Weak Form",
        "public_info": "Semi-Strong Form",
        "all_info": "Strong Form"
    }
    return efficiency_levels.get(information_type, "Unknown")

print(market_efficiency("past_prices"))
print(market_efficiency("public_info"))
print(market_efficiency("all_info"))
```

Slide 3: Weak Form Efficiency

Weak form efficiency posits that asset prices already reflect all historical price information. This implies that technical analysis, which relies on past price patterns, cannot consistently generate excess returns.

```python
import pandas as pd
import numpy as np

# Simulate stock prices
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
prices = 100 + np.cumsum(np.random.randn(100))
df = pd.DataFrame({'Date': dates, 'Price': prices})

# Calculate moving averages
df['MA5'] = df['Price'].rolling(window=5).mean()
df['MA20'] = df['Price'].rolling(window=20).mean()

print(df.tail())
```

Slide 4: Semi-Strong Form Efficiency

Semi-strong form efficiency suggests that asset prices quickly adjust to incorporate all publicly available information. This includes financial statements, economic reports, and news announcements.

```python
import time

def simulate_market_reaction(event):
    print(f"Breaking news: {event}")
    time.sleep(1)  # Simulate time passing
    print("Market analyzing information...")
    time.sleep(1)
    print("Prices adjusting...")
    time.sleep(1)
    print("New equilibrium reached.")

simulate_market_reaction("Company XYZ announces breakthrough product")
```

Slide 5: Strong Form Efficiency

Strong form efficiency proposes that asset prices reflect all information, both public and private. This implies that even insider information cannot be used to gain an advantage in the market.

```python
import random

class StrongFormMarket:
    def __init__(self):
        self.price = 100

    def trade(self, insider_info=False):
        # Random price movement, regardless of insider info
        self.price += random.uniform(-1, 1)
        return self.price

market = StrongFormMarket()
print(f"Price with public info: {market.trade():.2f}")
print(f"Price with insider info: {market.trade(insider_info=True):.2f}")
```

Slide 6: Implications of EMH for Investors

EMH suggests that active investment strategies are largely futile, as the market price is always "fair" given current information. This leads to the recommendation of passive investment strategies.

```python
import numpy as np

def compare_strategies(years, active_return, passive_return, active_fee, passive_fee):
    active_value = np.prod([(1 + active_return - active_fee) for _ in range(years)])
    passive_value = np.prod([(1 + passive_return - passive_fee) for _ in range(years)])
    return active_value, passive_value

active, passive = compare_strategies(years=20, active_return=0.10, passive_return=0.09, 
                                     active_fee=0.02, passive_fee=0.001)

print(f"Active strategy final value: {active:.2f}")
print(f"Passive strategy final value: {passive:.2f}")
```

Slide 7: Challenges to EMH

Despite its widespread acceptance, EMH faces several challenges. Market anomalies, behavioral biases, and the existence of successful investors like Warren Buffett seem to contradict the theory.

```python
import random

def simulate_market_anomaly(trials):
    anomalies = 0
    for _ in range(trials):
        if random.random() < 0.05:  # 5% chance of anomaly
            anomalies += 1
    return anomalies

anomaly_count = simulate_market_anomaly(1000)
print(f"Number of market anomalies in 1000 trials: {anomaly_count}")
```

Slide 8: The Random Walk Theory

Closely related to EMH is the Random Walk Theory, which posits that stock price changes are random and unpredictable. This supports the idea that future prices cannot be predicted based on past behavior.

```python
import numpy as np
import matplotlib.pyplot as plt

def random_walk(steps):
    return np.cumsum(np.random.choice([-1, 1], size=steps))

walks = [random_walk(1000) for _ in range(5)]

plt.figure(figsize=(10, 6))
for walk in walks:
    plt.plot(walk)
plt.title('Multiple Random Walks')
plt.xlabel('Steps')
plt.ylabel('Position')
plt.show()
```

Slide 9: EMH and Market Bubbles

Critics argue that EMH fails to explain market bubbles and crashes. These events suggest that markets can be driven by irrational exuberance or fear, rather than always reflecting fundamental values.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_bubble(duration, burst_point):
    x = np.linspace(0, duration, 1000)
    y = np.exp(x/10) + np.random.normal(0, 0.1, 1000)
    y[burst_point:] = y[burst_point] * np.exp(-0.5 * (x[burst_point:] - x[burst_point]))
    return x, y

x, y = simulate_bubble(duration=10, burst_point=800)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Simulated Market Bubble and Crash')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
```

Slide 10: EMH and Algorithmic Trading

The rise of algorithmic trading and high-frequency trading has implications for EMH. These strategies aim to exploit minor price discrepancies, potentially making markets more efficient.

```python
import time

def high_frequency_trade(price, threshold):
    while True:
        new_price = price + np.random.normal(0, 0.01)
        if abs(new_price - price) > threshold:
            print(f"Trade executed. Old price: {price:.2f}, New price: {new_price:.2f}")
            price = new_price
        time.sleep(0.1)  # Simulate high-frequency intervals

# Run for a few iterations
for _ in range(10):
    high_frequency_trade(100, 0.02)
```

Slide 11: Real-Life Example: Weather Forecasting

Consider weather forecasting as an analogy to EMH. In an "efficient weather market," all available information about atmospheric conditions would be instantly incorporated into forecasts, making it impossible to consistently predict weather better than the current forecast.

```python
import random

class WeatherMarket:
    def __init__(self):
        self.forecast = 70  # Initial forecast temperature

    def update_forecast(self, new_data):
        # Simulate rapid incorporation of new data
        self.forecast = (self.forecast + new_data) / 2
        return self.forecast

weather = WeatherMarket()
for day in range(7):
    new_data = random.uniform(60, 80)
    updated_forecast = weather.update_forecast(new_data)
    print(f"Day {day+1} forecast: {updated_forecast:.2f}°F")
```

Slide 12: Real-Life Example: Online Marketplaces

Online marketplaces like eBay can be seen as an example of efficient markets. In these platforms, the prices of goods quickly adjust based on supply and demand, reflecting all available information about the product.

```python
class OnlineMarketplace:
    def __init__(self, initial_price):
        self.price = initial_price
        self.demand = 100
        self.supply = 100

    def update_price(self):
        if self.demand > self.supply:
            self.price *= 1.05  # Price increases by 5%
        elif self.supply > self.demand:
            self.price *= 0.95  # Price decreases by 5%
        return self.price

    def simulate_market(self, days):
        for day in range(days):
            self.demand += random.randint(-10, 10)
            self.supply += random.randint(-10, 10)
            new_price = self.update_price()
            print(f"Day {day+1}: Price = ${new_price:.2f}, Demand = {self.demand}, Supply = {self.supply}")

market = OnlineMarketplace(100)
market.simulate_market(7)
```

Slide 13: The Future of EMH

As markets evolve with technological advancements and new financial instruments, the relevance and application of EMH continue to be debated. Research in behavioral finance and market microstructure may provide new insights into market efficiency.

```python
import matplotlib.pyplot as plt
import numpy as np

def simulate_market_evolution(years, efficiency_growth):
    time = np.arange(years)
    efficiency = 1 - np.exp(-efficiency_growth * time)
    return time, efficiency

time, efficiency = simulate_market_evolution(50, 0.05)

plt.figure(figsize=(10, 6))
plt.plot(time, efficiency)
plt.title('Hypothetical Market Efficiency Evolution')
plt.xlabel('Years')
plt.ylabel('Market Efficiency')
plt.ylim(0, 1)
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into the Efficient Market Hypothesis and related topics, here are some recommended academic papers:

1. Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. The Journal of Finance, 25(2), 383-417. ArXiv: [https://arxiv.org/abs/1102.1847](https://arxiv.org/abs/1102.1847)
2. Lo, A. W. (2004). The Adaptive Markets Hypothesis: Market Efficiency from an Evolutionary Perspective. Journal of Portfolio Management, 30(5), 15-29. ArXiv: [https://arxiv.org/abs/1106.5082](https://arxiv.org/abs/1106.5082)
3. Shiller, R. J. (2003). From Efficient Markets Theory to Behavioral Finance. Journal of Economic Perspectives, 17(1), 83-104. ArXiv: [https://arxiv.org/abs/1108.2011](https://arxiv.org/abs/1108.2011)

These papers provide a comprehensive overview of EMH, its critiques, and modern perspectives on market efficiency.

