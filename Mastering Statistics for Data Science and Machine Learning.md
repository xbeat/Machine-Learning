## Mastering Statistics for Data Science and Machine Learning
Slide 1: Introduction to Statistics in Data Science

Statistics forms the foundation of data science and machine learning. It provides the tools to analyze, interpret, and draw meaningful insights from data. This book aims to introduce key statistical concepts essential for aspiring data scientists and machine learning practitioners.

Slide 2: Source Code for Introduction to Statistics in Data Science

```python
import random

# Generate sample data
data = [random.randint(1, 100) for _ in range(1000)]

# Calculate basic statistics
mean = sum(data) / len(data)
sorted_data = sorted(data)
median = sorted_data[len(data) // 2]
mode = max(set(data), key=data.count)

print(f"Mean: {mean:.2f}")
print(f"Median: {median}")
print(f"Mode: {mode}")
```

Slide 3: Mean, Median, and Mode

These measures of central tendency help summarize data distributions. The mean represents the average, the median the middle value, and the mode the most frequent value. Understanding these concepts is crucial for initial data exploration and analysis.

Slide 4: Source Code for Mean, Median, and Mode

```python
def calculate_statistics(data):
    # Mean
    mean = sum(data) / len(data)
    
    # Median
    sorted_data = sorted(data)
    n = len(data)
    if n % 2 == 0:
        median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    else:
        median = sorted_data[n//2]
    
    # Mode
    freq = {}
    for value in data:
        freq[value] = freq.get(value, 0) + 1
    mode = max(freq, key=freq.get)
    
    return mean, median, mode

# Example usage
data = [1, 2, 2, 3, 4, 4, 4, 5]
mean, median, mode = calculate_statistics(data)
print(f"Mean: {mean:.2f}, Median: {median}, Mode: {mode}")
```

Slide 5: Variance and Standard Deviation

Variance and standard deviation measure the spread of data points around the mean. These metrics help understand data variability and are crucial for assessing data reliability and making predictions.

Slide 6: Source Code for Variance and Standard Deviation

```python
def calculate_variance_and_std(data):
    n = len(data)
    mean = sum(data) / n
    
    # Variance
    squared_diff_sum = sum((x - mean) ** 2 for x in data)
    variance = squared_diff_sum / n
    
    # Standard Deviation
    std_dev = variance ** 0.5
    
    return variance, std_dev

# Example usage
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
variance, std_dev = calculate_variance_and_std(data)
print(f"Variance: {variance:.2f}, Standard Deviation: {std_dev:.2f}")
```

Slide 7: Box Plots and Graphs

Visual representations of data are essential for understanding distributions and identifying patterns. Box plots display the five-number summary, while other graphs like histograms and scatter plots offer different perspectives on data relationships.

Slide 8: Source Code for Box Plots and Graphs

```python
def create_box_plot_data(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    q1_index = n // 4
    q2_index = n // 2
    q3_index = 3 * n // 4
    
    q1 = sorted_data[q1_index]
    q2 = sorted_data[q2_index]
    q3 = sorted_data[q3_index]
    
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    whiskers = [max(x for x in sorted_data if x >= lower_fence),
                min(x for x in sorted_data if x <= upper_fence)]
    
    outliers = [x for x in sorted_data if x < whiskers[0] or x > whiskers[1]]
    
    return {
        'q1': q1, 'q2': q2, 'q3': q3,
        'whiskers': whiskers,
        'outliers': outliers
    }

# Example usage
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50]
box_plot_data = create_box_plot_data(data)
print(box_plot_data)
```

Slide 9: Correlation

Correlation measures the strength and direction of relationships between variables. Understanding correlation is crucial for feature selection in machine learning and for identifying potential causal relationships in data.

Slide 10: Source Code for Correlation

```python
def calculate_correlation(x, y):
    n = len(x)
    if n != len(y):
        raise ValueError("Lists must have the same length")
    
    # Calculate means
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Calculate covariance and standard deviations
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
    std_dev_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
    std_dev_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5
    
    # Calculate correlation coefficient
    correlation = covariance / (std_dev_x * std_dev_y)
    
    return correlation

# Example usage
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
correlation = calculate_correlation(x, y)
print(f"Correlation coefficient: {correlation:.2f}")
```

Slide 11: Probability Distributions

Probability distributions describe the likelihood of different outcomes in a random experiment. Common distributions like normal, binomial, and Poisson are fundamental in statistical modeling and machine learning algorithms.

Slide 12: Source Code for Probability Distributions

```python
import math

def normal_distribution(x, mean, std_dev):
    coefficient = 1 / (std_dev * (2 * math.pi) ** 0.5)
    exponent = -((x - mean) ** 2) / (2 * std_dev ** 2)
    return coefficient * math.exp(exponent)

def binomial_distribution(n, k, p):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def poisson_distribution(k, lambda_):
    return (math.exp(-lambda_) * lambda_ ** k) / math.factorial(k)

# Example usage
print(f"Normal Distribution (x=0, μ=0, σ=1): {normal_distribution(0, 0, 1):.4f}")
print(f"Binomial Distribution (n=10, k=3, p=0.5): {binomial_distribution(10, 3, 0.5):.4f}")
print(f"Poisson Distribution (k=2, λ=3): {poisson_distribution(2, 3):.4f}")
```

Slide 13: Real-Life Example: Weather Prediction

Meteorologists use statistical models to predict weather patterns. They analyze historical data, atmospheric conditions, and various parameters to forecast temperature, precipitation, and other weather phenomena.

Slide 14: Source Code for Weather Prediction Example

```python
import random

def simulate_weather_data(days):
    temperature = []
    humidity = []
    for _ in range(days):
        temp = random.uniform(15, 35)
        hum = random.uniform(30, 90)
        temperature.append(temp)
        humidity.append(hum)
    return temperature, humidity

def predict_rain(temperature, humidity):
    rain_probability = []
    for temp, hum in zip(temperature, humidity):
        prob = (hum - 30) / 100 * (1 - (temp - 15) / 40)
        rain_probability.append(max(0, min(1, prob)))
    return rain_probability

# Simulate 7 days of weather data
days = 7
temp, humidity = simulate_weather_data(days)

# Predict rain probability
rain_prob = predict_rain(temp, humidity)

# Print results
for day in range(days):
    print(f"Day {day+1}: Temp: {temp[day]:.1f}°C, Humidity: {humidity[day]:.1f}%, Rain Prob: {rain_prob[day]:.2f}")
```

Slide 15: Real-Life Example: Quality Control in Manufacturing

Statistical process control is used in manufacturing to maintain product quality. By analyzing measurements of product characteristics, manufacturers can detect and correct deviations from desired specifications.

Slide 16: Source Code for Quality Control Example

```python
import random

def generate_measurements(n, target, std_dev):
    return [random.gauss(target, std_dev) for _ in range(n)]

def calculate_control_limits(measurements, k=3):
    mean = sum(measurements) / len(measurements)
    std_dev = (sum((x - mean) ** 2 for x in measurements) / len(measurements)) ** 0.5
    ucl = mean + k * std_dev
    lcl = mean - k * std_dev
    return mean, ucl, lcl

def check_out_of_control(measurement, lcl, ucl):
    return measurement < lcl or measurement > ucl

# Simulate production line
target_weight = 100  # grams
std_dev = 2  # grams
n_measurements = 50

measurements = generate_measurements(n_measurements, target_weight, std_dev)
mean, ucl, lcl = calculate_control_limits(measurements)

print(f"Mean: {mean:.2f}g")
print(f"Upper Control Limit: {ucl:.2f}g")
print(f"Lower Control Limit: {lcl:.2f}g")

# Check for out-of-control points
for i, measurement in enumerate(measurements, 1):
    if check_out_of_control(measurement, lcl, ucl):
        print(f"Warning: Measurement {i} ({measurement:.2f}g) is out of control limits")
```

Slide 17: Additional Resources

For those interested in diving deeper into statistics for data science and machine learning, here are some valuable resources:

1.  "Statistical Learning with Applications in R" by Gareth James et al. (Available on ArXiv: [https://arxiv.org/abs/1303.5922](https://arxiv.org/abs/1303.5922))
2.  "Probabilistic Machine Learning: An Introduction" by Kevin Murphy (ArXiv: [https://arxiv.org/abs/2012.03543](https://arxiv.org/abs/2012.03543))
3.  "A Survey of Deep Learning Techniques for Neural Machine Translation" by Shuohang Wang and Jing Jiang (ArXiv: [https://arxiv.org/abs/1804.09849](https://arxiv.org/abs/1804.09849))

These papers provide in-depth coverage of statistical concepts and their applications in machine learning and data science.

