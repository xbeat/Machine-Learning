## Python for Data Science Simplicity and Power

Slide 1: Introduction to Python for Data Science

Python has indeed become a popular language for data science and analysis. Its simplicity, readability, and extensive ecosystem of libraries make it an excellent tool for handling and visualizing data. In this presentation, we'll explore key Python techniques and libraries for data manipulation and visualization, focusing on built-in functionalities and implementing core concepts from scratch.

```python
# A simple example demonstrating Python's readability and power
data = [1, 2, 3, 4, 5]
average = sum(data) / len(data)
squared_diff = [(x - average) ** 2 for x in data]
variance = sum(squared_diff) / len(data)

print(f"Average: {average}")
print(f"Variance: {variance}")
```

Slide 2: Loading and Cleaning Data

When working with data, the first step is often loading and cleaning it. While libraries like Pandas are popular for this task, we can achieve similar results using Python's built-in functionalities. Let's create a simple CSV reader and cleaner from scratch.

```python
import csv
from collections import defaultdict

def load_and_clean_csv(file_path):
    data = defaultdict(list)
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key, value in row.items():
                # Basic cleaning: strip whitespace and handle empty values
                cleaned_value = value.strip() if value else None
                data[key].append(cleaned_value)
    return dict(data)

# Usage
file_path = 'data.csv'
cleaned_data = load_and_clean_csv(file_path)
print(cleaned_data)
```

Slide 3: Data Transformation

Data transformation is crucial for extracting meaningful insights. Let's implement some basic transformation techniques using pure Python.

```python
def transform_data(data):
    # Filter: Keep only even numbers
    filtered_data = [x for x in data if x % 2 == 0]
    
    # Sort: Arrange in descending order
    sorted_data = sorted(filtered_data, reverse=True)
    
    # Aggregate: Calculate the sum
    total = sum(sorted_data)
    
    return filtered_data, sorted_data, total

# Example usage
raw_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
filtered, sorted_result, sum_result = transform_data(raw_data)

print(f"Filtered data: {filtered}")
print(f"Sorted data: {sorted_result}")
print(f"Sum: {sum_result}")
```

Slide 4: Feature Engineering

Feature engineering involves creating new features or transforming existing ones to improve model performance. Let's implement a simple feature engineering technique from scratch.

```python
import math

def engineer_features(data):
    engineered_data = []
    for item in data:
        # Create new features
        log_feature = math.log(item) if item > 0 else 0
        square_feature = item ** 2
        sqrt_feature = math.sqrt(abs(item))
        
        engineered_data.append({
            'original': item,
            'log': log_feature,
            'square': square_feature,
            'sqrt': sqrt_feature
        })
    return engineered_data

# Example usage
original_data = [1, 2, 3, 4, 5]
engineered_features = engineer_features(original_data)
for item in engineered_features:
    print(item)
```

Slide 5: Implementing NumPy-like Functionality

While NumPy is a powerful library for numerical operations, we can implement basic array operations from scratch. Let's create a simple NumPy-like array class.

```python
class Array:
    def __init__(self, data):
        self.data = list(data)
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Array([x + other for x in self.data])
        elif isinstance(other, Array) and len(self) == len(other):
            return Array([x + y for x, y in zip(self.data, other.data)])
        else:
            raise ValueError("Invalid addition")
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Array([x * other for x in self.data])
        elif isinstance(other, Array) and len(self) == len(other):
            return Array([x * y for x, y in zip(self.data, other.data)])
        else:
            raise ValueError("Invalid multiplication")
    
    def __len__(self):
        return len(self.data)
    
    def __str__(self):
        return str(self.data)

# Usage
a = Array([1, 2, 3])
b = Array([4, 5, 6])
print(f"a + b = {a + b}")
print(f"a * 2 = {a * 2}")
print(f"a * b = {a * b}")
```

Slide 6: Implementing Pandas-like Functionality

Pandas is a powerful data analysis library. Let's create a simple DataFrame-like class to demonstrate some of its core functionalities.

```python
class DataFrame:
    def __init__(self, data):
        self.data = data
    
    def head(self, n=5):
        return {k: v[:n] for k, v in self.data.items()}
    
    def describe(self):
        stats = {}
        for column, values in self.data.items():
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if numeric_values:
                stats[column] = {
                    'count': len(numeric_values),
                    'mean': sum(numeric_values) / len(numeric_values),
                    'min': min(numeric_values),
                    'max': max(numeric_values)
                }
        return stats

# Usage
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': ['a', 'b', 'c', 'd', 'e']
}
df = DataFrame(data)
print("First 3 rows:")
print(df.head(3))
print("\nDescriptive statistics:")
print(df.describe())
```

Slide 7: Basic Data Visualization

While libraries like Matplotlib and Seaborn are commonly used for data visualization, we can create simple visualizations using ASCII characters. Let's implement a basic bar chart function.

```python
def ascii_bar_chart(data, max_width=50):
    max_value = max(data.values())
    for label, value in data.items():
        bar_length = int((value / max_value) * max_width)
        bar = '#' * bar_length
        print(f"{label.ljust(10)}: {bar} ({value})")

# Example usage
data = {'A': 5, 'B': 7, 'C': 3, 'D': 8}
print("ASCII Bar Chart:")
ascii_bar_chart(data)
```

Slide 8: Time Series Analysis

Time series analysis is crucial in many data science applications. Let's implement a simple moving average function from scratch.

```python
from datetime import datetime, timedelta

def moving_average(data, window_size):
    results = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        average = sum(window) / window_size
        results.append(average)
    return results

# Generate sample time series data
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(10)]
values = [10, 12, 15, 14, 16, 18, 17, 20, 22, 21]

# Calculate 3-day moving average
ma_values = moving_average(values, 3)

print("Date\t\tValue\tMoving Average")
for i, (date, value) in enumerate(zip(dates, values)):
    ma = ma_values[i-2] if i >= 2 else None
    print(f"{date.strftime('%Y-%m-%d')}\t{value}\t{ma:.2f if ma else 'N/A'}")
```

Slide 9: Basic Machine Learning: Linear Regression

Linear regression is a fundamental machine learning algorithm. Let's implement a simple linear regression model from scratch using the least squares method.

```python
def linear_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x_squared = sum(xi ** 2 for xi in x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept

# Example usage
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

slope, intercept = linear_regression(x, y)
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")

# Make predictions
x_test = [6, 7, 8]
y_pred = [slope * xi + intercept for xi in x_test]
print("Predictions:")
for xi, yi in zip(x_test, y_pred):
    print(f"x = {xi}, predicted y = {yi:.2f}")
```

Slide 10: Data Structures for Machine Learning: Decision Tree

Decision trees are widely used in machine learning. Let's implement a simple decision tree node structure from scratch.

```python
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(data, target, max_depth=3):
    if max_depth == 0 or len(set(target)) == 1:
        return DecisionTreeNode(value=max(set(target), key=target.count))
    
    best_feature = None
    best_threshold = None
    best_gain = 0
    
    for feature in range(len(data[0])):
        thresholds = sorted(set(row[feature] for row in data))
        for threshold in thresholds:
            gain = calculate_information_gain(data, target, feature, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    
    if best_gain == 0:
        return DecisionTreeNode(value=max(set(target), key=target.count))
    
    left_data, left_target, right_data, right_target = split_data(data, target, best_feature, best_threshold)
    
    left_subtree = build_tree(left_data, left_target, max_depth - 1)
    right_subtree = build_tree(right_data, right_target, max_depth - 1)
    
    return DecisionTreeNode(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

# Helper functions (not implemented for brevity)
def calculate_information_gain(data, target, feature, threshold):
    # Calculate information gain
    pass

def split_data(data, target, feature, threshold):
    # Split data based on feature and threshold
    pass

# Example usage
data = [[1, 2], [2, 3], [3, 1], [4, 4]]
target = [0, 0, 1, 1]
tree = build_tree(data, target)
print("Decision tree built successfully")
```

Slide 11: Natural Language Processing: Basic Text Analysis

Natural Language Processing (NLP) is a crucial part of data science. Let's implement some basic text analysis functions from scratch.

```python
import re
from collections import Counter

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def calculate_word_frequency(tokens):
    return Counter(tokens)

def calculate_tf_idf(documents):
    doc_count = len(documents)
    word_doc_count = Counter()
    doc_word_count = []
    
    for doc in documents:
        tokens = set(tokenize(doc))
        word_doc_count.update(tokens)
        doc_word_count.append(Counter(tokenize(doc)))
    
    tf_idf = []
    for doc_words in doc_word_count:
        doc_tf_idf = {}
        for word, count in doc_words.items():
            tf = count / sum(doc_words.values())
            idf = math.log(doc_count / (word_doc_count[word] + 1))
            doc_tf_idf[word] = tf * idf
        tf_idf.append(doc_tf_idf)
    
    return tf_idf

# Example usage
text = "This is a sample text. This text is used for demonstration."
tokens = tokenize(text)
word_freq = calculate_word_frequency(tokens)

print("Tokens:", tokens)
print("Word Frequency:", dict(word_freq))

documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
]
tf_idf_scores = calculate_tf_idf(documents)
print("\nTF-IDF Scores:")
for i, doc_scores in enumerate(tf_idf_scores):
    print(f"Document {i + 1}:")
    for word, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {word}: {score:.4f}")
```

Slide 12: Real-Life Example: Weather Data Analysis

Let's apply our data manipulation and visualization techniques to analyze weather data. We'll create a simple weather data generator and perform basic analysis.

```python
import random
from datetime import datetime, timedelta

def generate_weather_data(start_date, num_days):
    data = []
    current_date = start_date
    for _ in range(num_days):
        temp = random.uniform(10, 30)
        humidity = random.uniform(30, 80)
        pressure = random.uniform(995, 1015)
        data.append({
            'date': current_date,
            'temperature': round(temp, 1),
            'humidity': round(humidity, 1),
            'pressure': round(pressure, 1)
        })
        current_date += timedelta(days=1)
    return data

def analyze_weather_data(data):
    temp_values = [day['temperature'] for day in data]
    humid_values = [day['humidity'] for day in data]
    pressure_values = [day['pressure'] for day in data]
    
    avg_temp = sum(temp_values) / len(temp_values)
    avg_humid = sum(humid_values) / len(humid_values)
    avg_pressure = sum(pressure_values) / len(pressure_values)
    
    max_temp_day = max(data, key=lambda x: x['temperature'])
    min_temp_day = min(data, key=lambda x: x['temperature'])
    
    return {
        'avg_temp': round(avg_temp, 1),
        'avg_humid': round(avg_humid, 1),
        'avg_pressure': round(avg_pressure, 1),
        'max_temp_day': max_temp_day,
        'min_temp_day': min_temp_day
    }

# Generate and analyze weather data
start_date = datetime(2023, 1, 1)
weather_data = generate_weather_data(start_date, 30)
analysis_results = analyze_weather_data(weather_data)

print("Weather Data Analysis:")
print(f"Average Temperature: {analysis_results['avg_temp']}°C")
print(f"Average Humidity: {analysis_results['avg_humid']}%")
print(f"Average Pressure: {analysis_results['avg_pressure']} hPa")
print(f"Hottest Day: {analysis_results['max_temp_day']['date']} ({analysis_results['max_temp_day']['temperature']}°C)")
print(f"Coldest Day: {analysis_results['min_temp_day']['date']} ({analysis_results['min_temp_day']['temperature']}°C)")
```

Slide 13: Real-Life Example: Text Classification

Let's implement a simple text classification system using a bag-of-words approach and a naive Bayes classifier. This example demonstrates how to process text data and build a basic machine learning model.

```python
import re
from collections import defaultdict

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.word_probs = defaultdict(lambda: defaultdict(float))

    def train(self, texts, labels):
        word_counts = defaultdict(lambda: defaultdict(int))
        class_counts = defaultdict(int)

        for text, label in zip(texts, labels):
            tokens = tokenize(text)
            class_counts[label] += 1
            for word in set(tokens):
                word_counts[label][word] += 1

        total_docs = sum(class_counts.values())
        for label, count in class_counts.items():
            self.class_probs[label] = count / total_docs

        for label, words in word_counts.items():
            total_words = sum(words.values())
            for word, count in words.items():
                self.word_probs[label][word] = (count + 1) / (total_words + len(words))

    def classify(self, text):
        tokens = tokenize(text)
        scores = {}
        for label in self.class_probs:
            score = self.class_probs[label]
            for word in tokens:
                score *= self.word_probs[label][word]
            scores[label] = score
        return max(scores, key=scores.get)

# Example usage
texts = [
    "Python is a great programming language",
    "Data science is an exciting field",
    "Machine learning models can be powerful",
    "Natural language processing is fascinating"
]
labels = ["programming", "data_science", "machine_learning", "nlp"]

classifier = NaiveBayesClassifier()
classifier.train(texts, labels)

test_text = "I love working with Python for data analysis"
predicted_label = classifier.classify(test_text)
print(f"Predicted label for '{test_text}': {predicted_label}")
```

Slide 14: Data Visualization: Creating a Simple Scatter Plot

While specialized libraries are often used for data visualization, we can create basic plots using ASCII characters. Here's an example of a simple scatter plot function.

```python
def ascii_scatter_plot(x, y, width=60, height=20):
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    
    plot = [[' ' for _ in range(width)] for _ in range(height)]
    
    for xi, yi in zip(x, y):
        plot_x = int((xi - min_x) / (max_x - min_x) * (width - 1))
        plot_y = int((yi - min_y) / (max_y - min_y) * (height - 1))
        plot[height - 1 - plot_y][plot_x] = '*'
    
    for row in plot:
        print(''.join(row))
    
    print('-' * width)
    print(f"{min_x:.2f}{' ' * (width - 12)}{max_x:.2f}")

# Example usage
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 5, 4, 5, 6, 7, 8, 7, 9]

print("ASCII Scatter Plot:")
ascii_scatter_plot(x, y)
```

Slide 15: Additional Resources

For those interested in diving deeper into Python for data science, here are some valuable resources:

1.  "Python for Data Analysis" by Wes McKinney
2.  "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron
3.  Python Data Science Handbook: [https://jakevdp.github.io/PythonDataScienceHandbook/](https://jakevdp.github.io/PythonDataScienceHandbook/)
4.  ArXiv.org for latest research papers: [https://arxiv.org/list/cs.LG/recent](https://arxiv.org/list/cs.LG/recent) (Machine Learning category)
5.  Official Python documentation: [https://docs.python.org/3/](https://docs.python.org/3/)

These resources provide in-depth coverage of various data science topics and advanced Python techniques.

