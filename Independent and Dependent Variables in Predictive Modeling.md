## Independent and Dependent Variables in Predictive Modeling

Slide 1: Independent Variables

Independent variables are features used as input to predict an outcome. They are the factors that we manipulate or control in an experiment or analysis to observe their effect on the dependent variable.

```python
import random

# Simulate the effect of study time on exam score
def simulate_exam_score(study_time):
    base_score = 60
    improvement = study_time * 2
    random_factor = random.randint(-5, 5)
    return min(100, max(0, base_score + improvement + random_factor))

study_times = [2, 4, 6, 8, 10]
exam_scores = [simulate_exam_score(time) for time in study_times]

print("Study Time (hours) | Exam Score")
for time, score in zip(study_times, exam_scores):
    print(f"{time:17} | {score}")
```

Slide 2: Dependent Variables

The dependent variable is the outcome being predicted or measured in an experiment or analysis. It depends on or is influenced by the independent variables.

```python
import random

# Simulate the effect of temperature on ice cream sales
def simulate_ice_cream_sales(temperature):
    base_sales = 100
    temperature_effect = temperature * 5
    random_factor = random.randint(-20, 20)
    return max(0, base_sales + temperature_effect + random_factor)

temperatures = [20, 25, 30, 35, 40]
sales = [simulate_ice_cream_sales(temp) for temp in temperatures]

print("Temperature (°C) | Ice Cream Sales")
for temp, sale in zip(temperatures, sales):
    print(f"{temp:15} | {sale}")
```

Slide 3: Interaction Variables

Interaction variables represent the combined effect of two or more independent variables on the dependent variable. They are often used in regression analysis to capture non-additive relationships.

```python
# Simulate the effect of fertilizer and water on plant growth
def plant_growth(fertilizer, water):
    base_growth = 10
    fertilizer_effect = fertilizer * 2
    water_effect = water * 1.5
    interaction_effect = fertilizer * water * 0.5
    return base_growth + fertilizer_effect + water_effect + interaction_effect

fertilizer_levels = [1, 2, 3]
water_levels = [1, 2, 3]

print("Fertilizer | Water | Plant Growth")
for f in fertilizer_levels:
    for w in water_levels:
        growth = plant_growth(f, w)
        print(f"{f:9} | {w:5} | {growth:.2f}")
```

Slide 4: Latent Variables

Latent variables are not directly observed but inferred from other observed variables. They are often used in statistical modeling to represent underlying constructs or hidden factors.

```python
import random

# Simulate customer satisfaction (latent variable) based on observable factors
def simulate_customer_satisfaction(service_quality, product_quality):
    latent_satisfaction = (service_quality + product_quality) / 2
    observed_rating = round(latent_satisfaction + random.uniform(-0.5, 0.5), 1)
    return min(5, max(1, observed_rating))

customers = 5
service_qualities = [random.uniform(3, 5) for _ in range(customers)]
product_qualities = [random.uniform(3, 5) for _ in range(customers)]

print("Customer | Service Quality | Product Quality | Observed Rating")
for i in range(customers):
    rating = simulate_customer_satisfaction(service_qualities[i], product_qualities[i])
    print(f"{i+1:8} | {service_qualities[i]:.2f} | {product_qualities[i]:.2f} | {rating:.1f}")
```

Slide 5: Confounding Variables

Confounding variables are factors that influence both the independent and dependent variables in a study, potentially leading to misleading conclusions about the relationship between the variables of interest.

```python
import random

# Simulate the effect of exercise on health, with age as a confounding variable
def simulate_health_score(exercise_hours, age):
    base_health = 70
    exercise_effect = exercise_hours * 2
    age_effect = -age * 0.5
    random_factor = random.randint(-5, 5)
    return min(100, max(0, base_health + exercise_effect + age_effect + random_factor))

people = 5
exercise_hours = [random.randint(1, 10) for _ in range(people)]
ages = [random.randint(20, 60) for _ in range(people)]

print("Person | Exercise (hours) | Age | Health Score")
for i in range(people):
    health = simulate_health_score(exercise_hours[i], ages[i])
    print(f"{i+1:6} | {exercise_hours[i]:17} | {ages[i]:3} | {health}")
```

Slide 6: Correlated Variables

Correlated variables are those that have a statistical relationship with each other. This relationship can be positive (both increase together) or negative (one increases as the other decreases).

```python
import random

# Simulate correlated variables: study time and exam score
def generate_correlated_data(n, correlation):
    x = [random.gauss(0, 1) for _ in range(n)]
    y = [correlation * xi + random.gauss(0, (1 - correlation**2)**0.5) for xi in x]
    return x, y

study_time, exam_score = generate_correlated_data(10, 0.8)

print("Student | Study Time (z-score) | Exam Score (z-score)")
for i, (time, score) in enumerate(zip(study_time, exam_score), 1):
    print(f"{i:7} | {time:20.2f} | {score:20.2f}")
```

Slide 7: Control Variables

Control variables are factors that are kept constant during an experiment to isolate the effect of the independent variable on the dependent variable.

```python
import random

# Simulate plant growth experiment with temperature as a control variable
def plant_growth_experiment(water_amount, temperature=25):
    base_growth = 5
    water_effect = water_amount * 0.5
    temp_effect = (temperature - 25) * 0.2
    random_factor = random.uniform(-1, 1)
    return max(0, base_growth + water_effect + temp_effect + random_factor)

water_amounts = [10, 20, 30, 40, 50]

print("Water Amount (ml) | Plant Growth (cm)")
for water in water_amounts:
    growth = plant_growth_experiment(water)
    print(f"{water:17} | {growth:.2f}")
```

Slide 8: Leaky Variables

Leaky variables unintentionally provide information about the target variable that would not be available at the time of prediction, potentially leading to overly optimistic model performance.

```python
import random

# Simulate a house price prediction model with a leaky variable
def simulate_house_price(size, location, future_development=None):
    base_price = 200000
    size_effect = size * 1000
    location_effect = 50000 if location == "urban" else 20000
    
    # Leaky variable: future development plans
    if future_development:
        future_effect = 100000 if future_development == "positive" else -50000
    else:
        future_effect = 0
    
    random_factor = random.randint(-20000, 20000)
    return base_price + size_effect + location_effect + future_effect + random_factor

# Correct way: without using the leaky variable
print("Correct Prediction:")
price = simulate_house_price(150, "urban")
print(f"Predicted Price: ${price}")

# Incorrect way: using the leaky variable
print("\nIncorrect Prediction (with leaky variable):")
price_leaky = simulate_house_price(150, "urban", "positive")
print(f"Predicted Price: ${price_leaky}")
```

Slide 9: Stationary Variables

Stationary variables have statistical properties that do not change over time. They are crucial in many statistical models as they ensure consistent relationships between variables.

```python
import random

# Simulate a stationary time series
def generate_stationary_series(n, mean=0, std_dev=1):
    return [random.gauss(mean, std_dev) for _ in range(n)]

# Generate and analyze a stationary series
series = generate_stationary_series(100)

# Calculate rolling mean and variance
window = 20
rolling_mean = [sum(series[i:i+window]) / window for i in range(len(series) - window + 1)]
rolling_var = [sum((x - m)**2 for x in series[i:i+window]) / window 
               for i, m in enumerate(rolling_mean)]

print("Time | Value | Rolling Mean | Rolling Variance")
for i in range(0, 100, 10):
    if i < len(rolling_mean):
        print(f"{i:4} | {series[i]:5.2f} | {rolling_mean[i]:12.2f} | {rolling_var[i]:17.2f}")
    else:
        print(f"{i:4} | {series[i]:5.2f} | {'N/A':12} | {'N/A':17}")
```

Slide 10: Non-stationary Variables

Non-stationary variables have statistical properties that change over time. They can lead to unreliable predictions if not properly handled in time series analysis.

```python
import random

# Simulate a non-stationary time series with trend and seasonality
def generate_non_stationary_series(n):
    trend = [0.1 * i for i in range(n)]
    seasonality = [20 * (1 + (i % 12)) for i in range(n)]
    noise = [random.gauss(0, 5) for _ in range(n)]
    return [t + s + n for t, s, n in zip(trend, seasonality, noise)]

# Generate and analyze a non-stationary series
series = generate_non_stationary_series(100)

# Calculate rolling mean and variance
window = 20
rolling_mean = [sum(series[i:i+window]) / window for i in range(len(series) - window + 1)]
rolling_var = [sum((x - m)**2 for x in series[i:i+window]) / window 
               for i, m in enumerate(rolling_mean)]

print("Time | Value | Rolling Mean | Rolling Variance")
for i in range(0, 100, 10):
    if i < len(rolling_mean):
        print(f"{i:4} | {series[i]:5.2f} | {rolling_mean[i]:12.2f} | {rolling_var[i]:17.2f}")
    else:
        print(f"{i:4} | {series[i]:5.2f} | {'N/A':12} | {'N/A':17}")
```

Slide 11: Lagged Variables

Lagged variables represent previous time points' values of a given variable, shifting the data series by a specified number of periods. They are often used in time series analysis and forecasting.

```python
# Simulate a time series and create lagged variables
def create_lagged_series(series, lag):
    return [None] * lag + series[:-lag]

# Generate a sample time series
time_series = [10, 12, 15, 18, 20, 22, 25, 28, 30, 32]

# Create lagged variables
lag_1 = create_lagged_series(time_series, 1)
lag_2 = create_lagged_series(time_series, 2)

print("Time | Original | Lag 1 | Lag 2")
for t, orig, l1, l2 in zip(range(len(time_series)), time_series, lag_1, lag_2):
    print(f"{t:4} | {orig:8} | {l1 if l1 is not None else 'N/A':5} | {l2 if l2 is not None else 'N/A':5}")
```

Slide 12: Real-life Example: Weather Prediction

This example demonstrates the use of various variable types in a weather prediction scenario.

```python
import random

def predict_weather(temperature, humidity, pressure, yesterday_weather):
    # Independent variables: temperature, humidity, pressure
    # Lagged variable: yesterday_weather
    # Interaction variable: temp_humidity_interaction
    
    temp_effect = temperature * 0.3
    humidity_effect = humidity * -0.2
    pressure_effect = (pressure - 1000) * 0.1
    yesterday_effect = 0.5 if yesterday_weather == "Sunny" else -0.5
    
    # Interaction between temperature and humidity
    temp_humidity_interaction = (temperature * humidity) * -0.001
    
    weather_score = temp_effect + humidity_effect + pressure_effect + yesterday_effect + temp_humidity_interaction
    
    if weather_score > 5:
        return "Sunny"
    elif weather_score < -5:
        return "Rainy"
    else:
        return "Cloudy"

# Simulate weather for 5 days
for day in range(1, 6):
    temperature = random.uniform(15, 30)
    humidity = random.uniform(30, 80)
    pressure = random.uniform(990, 1020)
    yesterday_weather = "Sunny" if day == 1 else predict_weather(temperature, humidity, pressure, yesterday_weather)
    
    prediction = predict_weather(temperature, humidity, pressure, yesterday_weather)
    
    print(f"Day {day}:")
    print(f"Temperature: {temperature:.1f}°C")
    print(f"Humidity: {humidity:.1f}%")
    print(f"Pressure: {pressure:.1f} hPa")
    print(f"Yesterday's Weather: {yesterday_weather}")
    print(f"Predicted Weather: {prediction}")
    print()
```

Slide 13: Real-life Example: Customer Churn Prediction

This example illustrates the use of various variable types in predicting customer churn for a subscription-based service.

```python
import random

def predict_churn(usage_time, support_calls, subscription_length, competitor_prices):
    # Independent variables: usage_time, support_calls, subscription_length
    # Confounding variable: competitor_prices
    
    base_churn_prob = 0.1
    
    usage_effect = -0.01 * usage_time  # More usage, less likely to churn
    support_effect = 0.05 * support_calls  # More support calls, more likely to churn
    loyalty_effect = -0.02 * subscription_length  # Longer subscription, less likely to churn
    
    # Confounding effect: competitor prices affect both subscription length and churn probability
    competitor_effect = 0.001 * competitor_prices
    
    churn_probability = base_churn_prob + usage_effect + support_effect + loyalty_effect + competitor_effect
    churn_probability = max(0, min(1, churn_probability))  # Ensure probability is between 0 and 1
    
    return random.random() < churn_probability

# Simulate churn prediction for 10 customers
print("Customer | Usage (h) | Support Calls | Subscription (months) | Competitor Prices | Churn Prediction")
for customer in range(1, 11):
    usage_time = random.randint(0, 100)
    support_calls = random.randint(0, 5)
    subscription_length = random.randint(1, 24)
    competitor_prices = random.randint(50, 150)
    
    churn = predict_churn(usage_time, support_calls, subscription_length, competitor_prices)
    
    print(f"{customer:8} | {usage_time:9} | {support_calls:13} | {subscription_length:23} | {competitor_prices:17} | {'Yes' if churn else 'No'}")
```

Slide 14: Additional Resources

For further exploration of the topics covered in this presentation, consider the following resources:

1.  "Understanding Machine Learning: From Theory to Algorithms" by Shai Shalev-Shwartz and Shai Ben-David (arXiv:1406.0923)
2.  "An Introduction to Statistical Learning" by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (Available at: [https://www.statlearning.com/](https://www.statlearning.com/))
3.  "Causality: Models, Reasoning, and Inference" by Judea Pearl (For a deeper understanding of confounding variables and causal inference)
4.  "Time Series Analysis and Its Applications: With R Examples" by Robert H. Shumway and David S. Stoffer (For more on stationary and non-stationary variables)
5.  "Feature Engineering and Selection: A Practical Approach for Predictive Models" by Max Kuhn and Kjell Johnson (For insights on interaction variables and feature selection)

These resources provide comprehensive coverage of the variable types discussed and their applications in data science and machine learning. They offer both theoretical foundations and practical implementations to deepen your understanding of these concepts.

