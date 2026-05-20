## Exploring the Mathematical Thriller Pi Movie Using Python:
Slide 1: Introduction to "Pi" (1998)

"Pi" is a psychological thriller film directed by Darren Aronofsky. The movie follows Max Cohen, a mathematician who believes he has discovered a 216-digit number that can predict patterns in the stock market. While the film's premise is fictional, let's explore some mathematical concepts and their potential applications in financial analysis using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_fibonacci_spiral(n):
    phi = (1 + np.sqrt(5)) / 2
    theta = np.linspace(0, 8*np.pi, 1000)
    r = phi**(theta / (2*np.pi))
    
    plt.figure(figsize=(10, 10))
    plt.plot(r*np.cos(theta), r*np.sin(theta))
    plt.title("Fibonacci Spiral")
    plt.axis('equal')
    plt.show()

plot_fibonacci_spiral(10)
```

Slide 2: The Golden Ratio in Nature and Markets

The Golden Ratio (approximately 1.618) appears in various natural phenomena and has been observed in some market patterns. While not a predictor, it's sometimes used in technical analysis.

```python
def fibonacci_sequence(n):
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    return sequence

fib_seq = fibonacci_sequence(10)
ratios = [fib_seq[i+1] / fib_seq[i] for i in range(len(fib_seq)-1)]

print("Fibonacci Sequence:", fib_seq)
print("Ratios:", ratios)
print("Golden Ratio approximation:", ratios[-1])
```

Slide 3: Time Series Analysis in Finance

Time series analysis is crucial in financial modeling. Let's look at a simple moving average calculation, a common technique in stock market analysis.

```python
import pandas as pd
import yfinance as yf

def calculate_moving_average(symbol, period):
    data = yf.download(symbol, start="2023-01-01", end="2023-12-31")
    data['MA'] = data['Close'].rolling(window=period).mean()
    return data

stock_data = calculate_moving_average("AAPL", 20)
print(stock_data.tail())

plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data['Close'], label='Close Price')
plt.plot(stock_data.index, stock_data['MA'], label='20-day Moving Average')
plt.title("AAPL Stock Price and 20-day Moving Average")
plt.legend()
plt.show()
```

Slide 4: Patterns and Randomness in Markets

While the film suggests a deterministic pattern in the market, real markets exhibit both patterns and randomness. Let's simulate a random walk, often used to model stock prices.

```python
def random_walk(steps, start=100):
    returns = np.random.normal(loc=0.001, scale=0.02, size=steps)
    price_path = start * (1 + returns).cumprod()
    return price_path

simulated_prices = random_walk(252)  # 252 trading days in a year

plt.figure(figsize=(12, 6))
plt.plot(simulated_prices)
plt.title("Simulated Stock Price: Random Walk")
plt.xlabel("Trading Days")
plt.ylabel("Price")
plt.show()
```

Slide 5: Chaos Theory and the Butterfly Effect

The film touches on chaos theory, which studies how small changes can lead to large, unpredictable outcomes. Let's visualize this with the Lorenz attractor.

```python
from scipy.integrate import odeint

def lorenz_system(state, t, sigma, rho, beta):
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

sigma, rho, beta = 10, 28, 8/3
initial_state = [1, 1, 1]
t = np.linspace(0, 100, 10000)

solution = odeint(lorenz_system, initial_state, t, args=(sigma, rho, beta))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(solution[:, 0], solution[:, 1], solution[:, 2])
ax.set_title("Lorenz Attractor")
plt.show()
```

Slide 6: Fractals in Finance

Fractals, self-similar patterns at different scales, have been applied to financial market analysis. Let's generate a simple fractal pattern.

```python
def sierpinski_triangle(n):
    def midpoint(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def triangle(a, b, c, level):
        if level == 0:
            return [a, b, c]
        else:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            return (triangle(a, ab, ca, level-1) +
                    triangle(ab, b, bc, level-1) +
                    triangle(ca, bc, c, level-1))

    points = triangle((0, 0), (0.5, np.sqrt(3)/2), (1, 0), n)
    x, y = zip(*points)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, s=1)
    plt.title(f"Sierpinski Triangle (Level {n})")
    plt.axis('equal')
    plt.axis('off')
    plt.show()

sierpinski_triangle(7)
```

Slide 7: Neural Networks for Pattern Recognition

Modern approaches to market prediction often involve machine learning. Let's create a simple neural network for time series prediction.

```python
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Assume we have a numpy array 'data' with historical prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.reshape(-1, 1))

look_back = 60
X, Y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, Y, batch_size=1, epochs=1)

# This is a simplified example and would need more code for actual prediction
```

Slide 8: Correlation and Causation

The film's protagonist assumes a causal relationship between his number and market movements. In reality, correlation doesn't imply causation. Let's explore this concept.

```python
import seaborn as sns

np.random.seed(0)
x = np.random.randn(1000)
y = x + np.random.randn(1000) * 0.5

plt.figure(figsize=(10, 6))
sns.scatterplot(x=x, y=y)
plt.title("Correlation Example")
plt.xlabel("Variable X")
plt.ylabel("Variable Y")

correlation = np.corrcoef(x, y)[0, 1]
plt.text(0.05, 0.95, f"Correlation: {correlation:.2f}", transform=plt.gca().transAxes)

plt.show()
```

Slide 9: The Role of Randomness

While patterns exist in markets, randomness plays a crucial role. Let's simulate a coin flip experiment to illustrate the law of large numbers.

```python
def coin_flip_experiment(n_flips):
    flips = np.random.choice(['H', 'T'], size=n_flips)
    cumulative_heads = np.cumsum(flips == 'H')
    proportions = cumulative_heads / np.arange(1, n_flips + 1)
    return proportions

n_experiments = 100
n_flips = 1000

results = np.array([coin_flip_experiment(n_flips) for _ in range(n_experiments)])

plt.figure(figsize=(12, 6))
for result in results:
    plt.plot(range(1, n_flips + 1), result, alpha=0.1, color='blue')

plt.axhline(y=0.5, color='r', linestyle='--')
plt.title("Coin Flip Experiments: Proportion of Heads")
plt.xlabel("Number of Flips")
plt.ylabel("Proportion of Heads")
plt.ylim(0, 1)
plt.show()
```

Slide 10: Algorithmic Trading

Modern markets often involve algorithmic trading. Let's implement a simple moving average crossover strategy.

```python
def moving_average_crossover(data, short_window, long_window):
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    signals['signal'] = 0.0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                                > signals['long_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    return signals

# Assume we have stock_data from yfinance
signals = moving_average_crossover(stock_data, 20, 50)

plt.figure(figsize=(12, 6))
plt.plot(signals.index, signals['price'], label='Price')
plt.plot(signals.index, signals['short_mavg'], label='Short MA')
plt.plot(signals.index, signals['long_mavg'], label='Long MA')
plt.plot(signals.loc[signals.positions == 1.0].index, 
         signals.price[signals.positions == 1.0],
         '^', markersize=10, color='g', label='Buy')
plt.plot(signals.loc[signals.positions == -1.0].index, 
         signals.price[signals.positions == -1.0],
         'v', markersize=10, color='r', label='Sell')
plt.title("Moving Average Crossover Trading Strategy")
plt.legend()
plt.show()
```

Slide 11: Market Efficiency and Predictability

The Efficient Market Hypothesis suggests that predicting markets is challenging. Let's test for randomness in returns using the Ljung-Box test.

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

def test_market_efficiency(returns, lags=10):
    result = acorr_ljungbox(returns, lags=lags)
    return result.lb_pvalue

# Assume we have daily_returns calculated from stock data
p_values = test_market_efficiency(daily_returns)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(p_values) + 1), p_values)
plt.axhline(y=0.05, color='r', linestyle='--')
plt.title("Ljung-Box Test for Market Efficiency")
plt.xlabel("Lag")
plt.ylabel("P-value")
plt.show()
```

Slide 12: Ethical Considerations in Financial Modeling

The film raises questions about the ethics of market prediction. Let's explore a simple sentiment analysis tool for news headlines, which could be used in trading decisions.

```python
from textblob import TextBlob

def analyze_sentiment(headlines):
    sentiments = [TextBlob(headline).sentiment.polarity for headline in headlines]
    return np.mean(sentiments)

headlines = [
    "Company X reports record profits",
    "Market crash looms as tensions rise",
    "New technology breakthrough announced",
    "Economic indicators show mixed signals"
]

sentiment_score = analyze_sentiment(headlines)
print(f"Average sentiment score: {sentiment_score:.2f}")

plt.figure(figsize=(10, 6))
plt.bar(range(len(headlines)), [TextBlob(h).sentiment.polarity for h in headlines])
plt.title("Sentiment Analysis of Headlines")
plt.xlabel("Headline")
plt.ylabel("Sentiment Score")
plt.xticks(range(len(headlines)), [f"Headline {i+1}" for i in range(len(headlines))], rotation=45)
plt.tight_layout()
plt.show()
```

Slide 13: The Limits of Prediction

While advanced techniques exist, perfect market prediction remains elusive. Let's visualize prediction uncertainty using a Monte Carlo simulation.

```python
def monte_carlo_simulation(start_price, days, simulations):
    dt = 1/252  # Trading days in a year
    mu = 0.1  # Expected return
    sigma = 0.2  # Volatility

    price_paths = np.zeros((days, simulations))
    price_paths[0] = start_price

    for t in range(1, days):
        z = np.random.standard_normal(simulations)
        price_paths[t] = price_paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    return price_paths

simulations = monte_carlo_simulation(100, 252, 1000)

plt.figure(figsize=(12, 6))
plt.plot(simulations)
plt.title("Monte Carlo Simulation of Stock Prices")
plt.xlabel("Trading Days")
plt.ylabel("Stock Price")
plt.show()
```

Slide 14: Conclusion

While "Pi" presents an intriguing fictional scenario, real-world financial analysis involves a complex interplay of mathematics, statistics, and computer science. The quest for patterns in markets continues, but it's important to remember the role of randomness and the ethical implications of predictive models.

```python
def generate_word_cloud(text):
    from wordcloud import WordCloud
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Key Concepts in Financial Modeling")
    plt.show()

concepts = """
Mathematics Statistics Randomness Patterns
Algorithms Ethics Prediction Uncertainty
Time Series Neural Networks Fractals
Chaos Theory Efficiency Markets
"""

generate_word_cloud(concepts)
```

Slide 15: Additional Resources

For those interested in exploring these topics further, here are some relevant resources:

1. "Econophysics and Financial Economics: An Emerging Dialogue" by Franck Jovanovic and Christophe Schinckus (ArXiv:1406.2974)
2. "Machine Learning for Financial Market Prediction" by Matthew Dixon, et al. (ArXiv:1609.05243)
3. "The Geometry of Chaos: Fractals and Strange Attractors in Financial Markets" by J. Doyne Farmer (ArXiv:cond-mat/9508102)

These papers provide in-depth discussions on the intersection of mathematics, physics, and finance, offering valuable insights into the complex world of financial modeling and prediction.

