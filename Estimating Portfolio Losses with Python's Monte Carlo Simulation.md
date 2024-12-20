## Estimating Portfolio Losses with Python's Monte Carlo Simulation
Slide 1: Introduction to Monte Carlo Simulation

Monte Carlo simulation is a powerful statistical technique used to model complex systems and estimate probabilities. It's named after the famous casino in Monaco, reflecting its reliance on repeated random sampling. This method is particularly useful when dealing with problems that have many variables or uncertain inputs.

```python
import random
import matplotlib.pyplot as plt

def simple_monte_carlo(num_simulations):
    inside_circle = 0
    total_points = num_simulations

    for _ in range(total_points):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside_circle += 1

    pi_estimate = 4 * inside_circle / total_points
    return pi_estimate

# Run simulation
pi_approx = simple_monte_carlo(100000)
print(f"Estimated value of pi: {pi_approx}")
```

Slide 2: Value at Risk (VaR) Concept

Value at Risk (VaR) is a risk management metric that quantifies the potential loss in value of a risky asset or portfolio over a defined period for a given confidence interval. It answers the question: "What is the maximum loss we can expect with X% confidence over the next N days?"

```python
import random
import numpy as np

def calculate_var(returns, confidence_level):
    sorted_returns = sorted(returns)
    index = int(len(returns) * (1 - confidence_level))
    return -sorted_returns[index]

# Generate sample returns
np.random.seed(42)
returns = np.random.normal(0, 0.01, 1000)

# Calculate 95% VaR
var_95 = calculate_var(returns, 0.95)
print(f"95% VaR: {var_95:.2%}")
```

Slide 3: Setting Up the Portfolio

We start by defining our initial portfolio value and the key parameters for our simulation: the expected daily return (mu) and the daily volatility (sigma). These parameters are crucial for modeling the behavior of our portfolio over time.

```python
import numpy as np
import matplotlib.pyplot as plt

# Portfolio setup
initial_portfolio = 1_000_000  # $1,000,000 initial portfolio
mu = 0.001  # Expected daily return (0.1%)
sigma = 0.02  # Daily volatility (2%)

print(f"Initial Portfolio: ${initial_portfolio:,}")
print(f"Expected Daily Return: {mu:.2%}")
print(f"Daily Volatility: {sigma:.2%}")
```

Slide 4: Geometric Brownian Motion

Geometric Brownian Motion (GBM) is a continuous-time stochastic process often used to model stock prices. It assumes that the percentage changes in the stock price follow a normal distribution. We use GBM to generate possible future portfolio values.

```python
def geometric_brownian_motion(S0, mu, sigma, dt, T):
    """
    S0: initial stock price
    mu: drift (expected return)
    sigma: volatility
    dt: time step
    T: total time
    """
    steps = int(T / dt)
    t = np.linspace(0, T, steps)
    W = np.random.standard_normal(size=steps)
    W = np.cumsum(W) * np.sqrt(dt)  # Brownian motion
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)  # Geometric Brownian motion
    return S

# Example usage
S = geometric_brownian_motion(initial_portfolio, mu, sigma, 1, 252)
plt.plot(S)
plt.title("Sample GBM Path for Portfolio Value")
plt.xlabel("Trading Days")
plt.ylabel("Portfolio Value ($)")
plt.show()
```

Slide 5: Running the Monte Carlo Simulation

We now use the GBM model to simulate multiple paths for our portfolio value. Each path represents a possible scenario of how our portfolio might perform over the given time period.

```python
def run_simulation(initial_value, mu, sigma, days, num_simulations):
    dt = 1  # 1 day
    simulation_results = np.zeros((num_simulations, days))
    
    for i in range(num_simulations):
        simulation_results[i] = geometric_brownian_motion(initial_value, mu, sigma, dt, days)
    
    return simulation_results

# Run simulation
num_simulations = 100_000
days = 1  # We're interested in 1-day VaR
simulations = run_simulation(initial_portfolio, mu, sigma, days, num_simulations)

print(f"Number of simulations: {num_simulations:,}")
print(f"Shape of results: {simulations.shape}")
```

Slide 6: Calculating Profit and Loss (P&L)

To calculate VaR, we need to determine the potential profit or loss for each simulation. We do this by comparing the end-of-period value to our initial portfolio value.

```python
# Calculate P&L
pnl = simulations[:, -1] - initial_portfolio

# Plot P&L distribution
plt.figure(figsize=(10, 6))
plt.hist(pnl, bins=100, edgecolor='black')
plt.title("Distribution of Simulated P&L")
plt.xlabel("Profit/Loss ($)")
plt.ylabel("Frequency")
plt.show()

print(f"Minimum P&L: ${pnl.min():,.2f}")
print(f"Maximum P&L: ${pnl.max():,.2f}")
print(f"Mean P&L: ${pnl.mean():,.2f}")
```

Slide 7: Calculating Value at Risk (VaR)

Now we can calculate the VaR at our chosen confidence level (95% in this case). The VaR represents the maximum expected loss with 95% confidence.

```python
confidence_level = 0.95
var_95 = np.percentile(pnl, 100 * (1 - confidence_level))

print(f"95% VaR: ${-var_95:,.2f}")

# Visualize VaR
plt.figure(figsize=(10, 6))
plt.hist(pnl, bins=100, edgecolor='black')
plt.axvline(var_95, color='r', linestyle='dashed', linewidth=2)
plt.title("Distribution of Simulated P&L with 95% VaR")
plt.xlabel("Profit/Loss ($)")
plt.ylabel("Frequency")
plt.text(var_95, plt.ylim()[1], f'95% VaR: ${-var_95:,.2f}', 
         horizontalalignment='right', verticalalignment='top')
plt.show()
```

Slide 8: Interpreting the Results

The 95% VaR tells us that with 95% confidence, our portfolio's maximum loss over the next day will not exceed the calculated amount. This information is crucial for risk management and decision-making.

```python
total_simulations = len(pnl)
losses_beyond_var = sum(pnl < var_95)
percentage_beyond_var = (losses_beyond_var / total_simulations) * 100

print(f"Number of simulations with losses beyond VaR: {losses_beyond_var}")
print(f"Percentage of simulations beyond VaR: {percentage_beyond_var:.2f}%")
print(f"This aligns with our 95% confidence level (5% tail)")
```

Slide 9: Sensitivity Analysis: Changing Volatility

Let's explore how changes in volatility affect our VaR calculation. We'll run simulations with different volatility levels and compare the results.

```python
volatilities = [0.01, 0.02, 0.03, 0.04, 0.05]
var_results = []

for vol in volatilities:
    simulations = run_simulation(initial_portfolio, mu, vol, days, num_simulations)
    pnl = simulations[:, -1] - initial_portfolio
    var_95 = np.percentile(pnl, 100 * (1 - confidence_level))
    var_results.append(-var_95)

plt.figure(figsize=(10, 6))
plt.plot(volatilities, var_results, marker='o')
plt.title("95% VaR vs Volatility")
plt.xlabel("Volatility")
plt.ylabel("95% VaR ($)")
plt.grid(True)
plt.show()

for vol, var in zip(volatilities, var_results):
    print(f"Volatility: {vol:.2f}, 95% VaR: ${var:,.2f}")
```

Slide 10: Real-Life Example: Weather Prediction

Monte Carlo simulations are widely used in weather forecasting. Meteorologists use these simulations to predict the probability of different weather scenarios.

```python
import random

def simulate_temperature(base_temp, volatility, days):
    temp = base_temp
    temperatures = [temp]
    for _ in range(days - 1):
        change = random.gauss(0, volatility)
        temp += change
        temperatures.append(temp)
    return temperatures

# Simulate 1000 temperature scenarios
scenarios = 1000
days = 7
base_temp = 20  # Starting temperature in Celsius
volatility = 2  # Daily temperature volatility

all_scenarios = [simulate_temperature(base_temp, volatility, days) for _ in range(scenarios)]

# Calculate probability of temperature exceeding 25°C on day 7
exceed_25 = sum(scenario[-1] > 25 for scenario in all_scenarios)
probability = exceed_25 / scenarios

print(f"Probability of temperature exceeding 25°C on day 7: {probability:.2%}")
```

Slide 11: Real-Life Example: Project Management

Monte Carlo simulations can be used in project management to estimate project completion times and costs, considering uncertainties in task durations and resource availability.

```python
import random

def simulate_project():
    tasks = {
        'A': (10, 20),  # (min_days, max_days)
        'B': (15, 25),
        'C': (5, 15),
        'D': (8, 18)
    }
    
    total_days = 0
    for task, (min_days, max_days) in tasks.items():
        days = random.uniform(min_days, max_days)
        total_days += days
    
    return total_days

# Run 10,000 simulations
simulations = 10000
project_durations = [simulate_project() for _ in range(simulations)]

# Calculate 90% confidence interval
confidence_level = 0.9
lower_bound = np.percentile(project_durations, (1 - confidence_level) / 2 * 100)
upper_bound = np.percentile(project_durations, (1 + confidence_level) / 2 * 100)

print(f"90% confidence interval for project duration: {lower_bound:.1f} to {upper_bound:.1f} days")
```

Slide 12: Limitations and Considerations

While Monte Carlo simulations are powerful, they have limitations. The accuracy of results depends on the quality of input parameters and assumptions about the underlying distributions.

```python
def demonstrate_input_sensitivity(initial_value, mu, sigma, days, num_simulations):
    base_var = calculate_var(initial_value, mu, sigma, days, num_simulations)
    
    # Vary mu
    var_high_mu = calculate_var(initial_value, mu * 1.1, sigma, days, num_simulations)
    var_low_mu = calculate_var(initial_value, mu * 0.9, sigma, days, num_simulations)
    
    # Vary sigma
    var_high_sigma = calculate_var(initial_value, mu, sigma * 1.1, days, num_simulations)
    var_low_sigma = calculate_var(initial_value, mu, sigma * 0.9, days, num_simulations)
    
    print(f"Base VaR: ${base_var:,.2f}")
    print(f"VaR with 10% higher mu: ${var_high_mu:,.2f}")
    print(f"VaR with 10% lower mu: ${var_low_mu:,.2f}")
    print(f"VaR with 10% higher sigma: ${var_high_sigma:,.2f}")
    print(f"VaR with 10% lower sigma: ${var_low_sigma:,.2f}")

def calculate_var(initial_value, mu, sigma, days, num_simulations):
    simulations = run_simulation(initial_value, mu, sigma, days, num_simulations)
    pnl = simulations[:, -1] - initial_value
    return -np.percentile(pnl, 5)

demonstrate_input_sensitivity(initial_portfolio, mu, sigma, days, num_simulations)
```

Slide 13: Conclusion and Best Practices

Monte Carlo simulations are a powerful tool for risk assessment and decision-making under uncertainty. To make the most of this technique:

1.  Use reliable data sources for input parameters.
2.  Run a sufficient number of simulations for statistical significance.
3.  Regularly update and validate your models.
4.  Combine Monte Carlo results with other analytical methods for a comprehensive risk assessment.

```python
def monte_carlo_best_practices(num_simulations, confidence_level):
    # Example of checking if number of simulations is sufficient
    if num_simulations < 10000:
        print("Warning: Consider increasing the number of simulations for better accuracy.")
    
    # Example of validating confidence level
    if confidence_level < 0.9 or confidence_level > 0.99:
        print("Warning: Unusual confidence level. Common levels are between 90% and 99%.")
    
    # Example of suggesting model updates
    print("Reminder: Regularly update your model parameters based on recent market data.")
    
    # Example of recommending complementary analysis
    print("Best Practice: Combine Monte Carlo results with stress testing and scenario analysis.")

monte_carlo_best_practices(num_simulations, confidence_level)
```

Slide 14: Additional Resources

For those interested in diving deeper into Monte Carlo simulations and their applications in finance and risk management, here are some valuable resources:

1.  "Monte Carlo Methods in Financial Engineering" by Paul Glasserman (Springer, 2003)
2.  "Financial Risk Forecasting" by Jon Danielsson (Wiley, 2011)
3.  "Implementing Value at Risk" by Philip Best (Wiley, 2000)
4.  ArXiv.org paper: "Monte Carlo Methods for Risk Analysis in Financial Markets" ([https://arxiv.org/abs/q-fin/0410016](https://arxiv.org/abs/q-fin/0410016))
5.  ArXiv.org paper: "A Survey of Monte Carlo Methods for Value-at-Risk" ([https://arxiv.org/abs/1404.3303](https://arxiv.org/abs/1404.3303))

These resources provide in-depth explanations of the theoretical foundations and practical implementations of Monte Carlo simulations in financial contexts.

