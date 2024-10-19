## Python User-Defined Aggregate Functions (UDAFs) for Data Analysis

Slide 1: Understanding User-Defined Aggregate Functions (UDAFs) in Python

User-Defined Aggregate Functions (UDAFs) in Python allow developers to create custom functions that summarize data according to specific needs. These functions extend beyond built-in aggregates like sum or count, enabling more flexible and tailored data analysis.

```python
def custom_aggregate(data):
    return sum(data) / len(data) if data else 0

# Using the custom aggregate
numbers = [1, 2, 3, 4, 5]
result = custom_aggregate(numbers)
print(f"Custom aggregate result: {result}")
```

Slide 2: Creating a Basic UDAF: Weighted Average

Let's create a UDAF to calculate the weighted average of a dataset. This function takes two lists: one for values and another for their corresponding weights.

```python
    if len(values) != len(weights):
        raise ValueError("Values and weights must have the same length")
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)

# Example usage
values = [80, 90, 95]
weights = [0.3, 0.4, 0.3]
result = weighted_average(values, weights)
print(f"Weighted average: {result}")
```

Slide 3: Implementing a UDAF in a Class-Based Structure

For more complex UDAFs, a class-based structure can be beneficial. This approach allows for maintaining state between function calls and provides a clearer organization of the aggregate function's logic.

```python
    def __init__(self):
        self.count = 0
        self.total = 0
    
    def step(self, value):
        self.count += 1
        self.total += value
    
    def finalize(self):
        return self.total / self.count if self.count > 0 else 0

# Usage
ra = RunningAverage()
for num in [1, 2, 3, 4, 5]:
    ra.step(num)
result = ra.finalize()
print(f"Running average: {result}")
```

Slide 4: UDAF for Mode Calculation

Let's create a UDAF to find the mode (most frequent value) in a dataset. This example demonstrates handling more complex logic within a custom aggregate function.

```python

def mode(data):
    if not data:
        return None
    counter = Counter(data)
    max_count = max(counter.values())
    modes = [k for k, v in counter.items() if v == max_count]
    return modes[0] if len(modes) == 1 else modes

# Example usage
numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
result = mode(numbers)
print(f"Mode: {result}")
```

Slide 5: UDAF for Data Normalization

This UDAF normalizes a dataset by scaling values to a range between 0 and 1. It's useful in various data preprocessing scenarios.

```python
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def step(self, value):
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
    
    def finalize(self, data):
        range_val = self.max_val - self.min_val
        return [(x - self.min_val) / range_val for x in data] if range_val != 0 else [0] * len(data)

# Usage
normalizer = Normalizer()
data = [10, 20, 30, 40, 50]
for value in data:
    normalizer.step(value)
normalized_data = normalizer.finalize(data)
print(f"Normalized data: {normalized_data}")
```

Slide 6: UDAF for Moving Average

Implementing a moving average UDAF can be useful for smoothing time series data or identifying trends over time.

```python

class MovingAverage:
    def __init__(self, window_size):
        self.window = deque(maxlen=window_size)
    
    def step(self, value):
        self.window.append(value)
    
    def finalize(self):
        return sum(self.window) / len(self.window) if self.window else 0

# Usage
ma = MovingAverage(window_size=3)
data = [1, 3, 5, 2, 8, 4, 6]
moving_averages = []
for value in data:
    ma.step(value)
    moving_averages.append(ma.finalize())
print(f"Moving averages: {moving_averages}")
```

Slide 7: UDAF for Variance Calculation

Creating a UDAF to calculate variance demonstrates how to handle multi-pass aggregations where we need to compute intermediate results.

```python

class VarianceCalculator:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.sum_sq = 0
    
    def step(self, value):
        self.count += 1
        self.sum += value
        self.sum_sq += value ** 2
    
    def finalize(self):
        if self.count < 2:
            return 0
        mean = self.sum / self.count
        return (self.sum_sq / self.count) - (mean ** 2)

# Usage
vc = VarianceCalculator()
data = [2, 4, 4, 4, 5, 5, 7, 9]
for value in data:
    vc.step(value)
variance = vc.finalize()
std_dev = math.sqrt(variance)
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev}")
```

Slide 8: UDAF for Percentile Calculation

This UDAF calculates a specified percentile of a dataset, which is useful for understanding data distribution and identifying outliers.

```python

class PercentileCalculator:
    def __init__(self, percentile):
        self.percentile = percentile
        self.values = []
    
    def step(self, value):
        self.values.append(value)
    
    def finalize(self):
        if not self.values:
            return None
        sorted_values = sorted(self.values)
        index = (len(self.values) - 1) * self.percentile / 100
        lower_index = math.floor(index)
        upper_index = math.ceil(index)
        if lower_index == upper_index:
            return sorted_values[int(index)]
        lower_value = sorted_values[lower_index]
        upper_value = sorted_values[upper_index]
        return lower_value + (upper_value - lower_value) * (index - lower_index)

# Usage
pc = PercentileCalculator(percentile=75)
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for value in data:
    pc.step(value)
result = pc.finalize()
print(f"75th percentile: {result}")
```

Slide 9: UDAF for String Concatenation

This UDAF demonstrates how custom aggregates can be applied to non-numeric data, such as concatenating strings with a custom separator.

```python
    def __init__(self, separator=', '):
        self.separator = separator
        self.strings = []
    
    def step(self, value):
        self.strings.append(str(value))
    
    def finalize(self):
        return self.separator.join(self.strings)

# Usage
sc = StringConcatenator(separator=' | ')
words = ['Python', 'is', 'awesome', 'for', 'data', 'analysis']
for word in words:
    sc.step(word)
result = sc.finalize()
print(f"Concatenated string: {result}")
```

Slide 10: Real-Life Example: UDAF for Environmental Data Analysis

Suppose we're analyzing temperature data from various weather stations. We want to create a UDAF that calculates the daily temperature range (difference between max and min temperatures) and flags days with extreme variations.

```python
    def __init__(self, extreme_threshold):
        self.extreme_threshold = extreme_threshold
        self.daily_min = float('inf')
        self.daily_max = float('-inf')
    
    def step(self, temperature):
        self.daily_min = min(self.daily_min, temperature)
        self.daily_max = max(self.daily_max, temperature)
    
    def finalize(self):
        temp_range = self.daily_max - self.daily_min
        is_extreme = temp_range > self.extreme_threshold
        return {
            'min_temp': self.daily_min,
            'max_temp': self.daily_max,
            'temp_range': temp_range,
            'is_extreme': is_extreme
        }

# Usage
tra = TemperatureRangeAnalyzer(extreme_threshold=20)
daily_temperatures = [15, 18, 22, 25, 30, 28, 20]
for temp in daily_temperatures:
    tra.step(temp)
result = tra.finalize()
print(f"Temperature analysis: {result}")
```

Slide 11: Real-Life Example: UDAF for Text Sentiment Analysis

This UDAF performs a simple sentiment analysis on text data, counting positive and negative words to determine overall sentiment.

```python
    def __init__(self):
        self.positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful'])
        self.negative_words = set(['bad', 'poor', 'terrible', 'awful', 'horrible'])
        self.positive_count = 0
        self.negative_count = 0
        self.total_words = 0
    
    def step(self, text):
        words = text.lower().split()
        self.total_words += len(words)
        self.positive_count += sum(1 for word in words if word in self.positive_words)
        self.negative_count += sum(1 for word in words if word in self.negative_words)
    
    def finalize(self):
        if self.total_words == 0:
            return 'Neutral'
        sentiment_score = (self.positive_count - self.negative_count) / self.total_words
        if sentiment_score > 0.05:
            return 'Positive'
        elif sentiment_score < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

# Usage
sa = SentimentAnalyzer()
reviews = [
    "This product is amazing and works great!",
    "I had a terrible experience with customer service.",
    "The quality is good but the price is a bit high."
]
for review in reviews:
    sa.step(review)
overall_sentiment = sa.finalize()
print(f"Overall sentiment: {overall_sentiment}")
```

Slide 12: Optimizing UDAFs for Large Datasets

When working with large datasets, it's crucial to optimize UDAFs for memory efficiency and performance. Here's an example of a memory-efficient UDAF for calculating the median of a large dataset.

```python

class MedianCalculator:
    def __init__(self):
        self.smaller = []  # max heap
        self.larger = []   # min heap
    
    def step(self, value):
        if len(self.smaller) == len(self.larger):
            heapq.heappush(self.larger, -heapq.heappushpop(self.smaller, -value))
        else:
            heapq.heappush(self.smaller, -heapq.heappushpop(self.larger, value))
    
    def finalize(self):
        if len(self.smaller) == len(self.larger):
            return (-self.smaller[0] + self.larger[0]) / 2
        else:
            return self.larger[0]

# Usage
mc = MedianCalculator()
large_dataset = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
for value in large_dataset:
    mc.step(value)
median = mc.finalize()
print(f"Median of the dataset: {median}")
```

Slide 13: Combining Multiple UDAFs

In real-world scenarios, you might need to combine multiple UDAFs to perform complex analyses. Here's an example that combines several UDAFs to analyze a dataset of product reviews.

```python
    def __init__(self):
        self.total_reviews = 0
        self.total_rating = 0
        self.word_count = 0
        self.positive_words = set(['good', 'great', 'excellent', 'amazing'])
        self.negative_words = set(['bad', 'poor', 'terrible', 'awful'])
        self.sentiment_score = 0
    
    def step(self, review, rating):
        self.total_reviews += 1
        self.total_rating += rating
        words = review.lower().split()
        self.word_count += len(words)
        self.sentiment_score += sum(1 for word in words if word in self.positive_words)
        self.sentiment_score -= sum(1 for word in words if word in self.negative_words)
    
    def finalize(self):
        avg_rating = self.total_rating / self.total_reviews if self.total_reviews > 0 else 0
        avg_word_count = self.word_count / self.total_reviews if self.total_reviews > 0 else 0
        overall_sentiment = 'Positive' if self.sentiment_score > 0 else 'Negative' if self.sentiment_score < 0 else 'Neutral'
        return {
            'total_reviews': self.total_reviews,
            'average_rating': avg_rating,
            'average_word_count': avg_word_count,
            'overall_sentiment': overall_sentiment
        }

# Usage
ra = ReviewAnalyzer()
reviews = [
    ("This product is amazing!", 5),
    ("Not worth the money, terrible quality.", 1),
    ("Good product, but a bit overpriced.", 3)
]
for review, rating in reviews:
    ra.step(review, rating)
analysis_result = ra.finalize()
print(f"Review analysis: {analysis_result}")
```

Slide 14: Best Practices for Creating UDAFs

When creating UDAFs, consider the following best practices:

1. Ensure memory efficiency, especially for large datasets.
2. Implement clear error handling and input validation.
3. Use descriptive names for functions and variables.
4. Document your UDAFs thoroughly, including expected inputs and outputs.
5. Test your UDAFs with various edge cases and large datasets.

Slide 15: Best Practices for Creating UDAFs

Here's an example incorporating these practices:

```python
    """
    A UDAF that calculates robust statistics (median and IQR) for a dataset.
    """
    def __init__(self):
        self.data = []
    
    def step(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Input must be a number")
        self.data.append(value)
    
    def finalize(self):
        if not self.data:
            raise ValueError("Dataset is empty")
        sorted_data = sorted(self.data)
        n = len(sorted_data)
        median = sorted_data[n // 2] if n % 2 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        q1 = sorted_data[n // 4]
        q3 = sorted_data[3 * n // 4]
        iqr = q3 - q1
        return {"median": median, "IQR": iqr}

# Usage
calc = RobustStatCalculator()
dataset = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
for value in dataset:
    calc.step(value)
result = calc.finalize()
print(f"Robust statistics: {result}")
```

Slide 16: Integrating UDAFs with Data Processing Frameworks

UDAFs can be integrated with popular data processing frameworks like pandas or PySpark for more efficient data analysis. Here's an example using pandas:

```python

def custom_udaf(group):
    return pd.Series({
        'mean': group.mean(),
        'median': group.median(),
        'range': group.max() - group.min()
    })

# Sample data
data = {
    'category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'value': [10, 15, 20, 25, 30, 35]
}
df = pd.DataFrame(data)

# Apply UDAF
result = df.groupby('category')['value'].apply(custom_udaf)
print(result)
```

Slide 17: Additional Resources

For those interested in diving deeper into UDAFs and advanced Python data processing:

1. "Python for Data Analysis" by Wes McKinney (O'Reilly Media)
2. "Fluent Python" by Luciano Ramalho (O'Reilly Media)
3. "Effective Pandas" by Matt Harrison (available online)
4. ArXiv paper: "Efficient Aggregation Algorithms for Probabilistic Data" (arXiv:1703.02614)
5. PEP 450 - Adding a Statistics Module to the Standard Library (python.org/dev/peps/pep-0450/)

These resources provide in-depth explanations and advanced techniques for working with data in Python.


