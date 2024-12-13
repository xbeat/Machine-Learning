## List Unpacking in Python
Slide 1: Basic List Unpacking

List unpacking in Python allows extracting individual elements from sequences into separate variables using a simple assignment syntax. This fundamental feature enables cleaner and more readable code by directly mapping sequence items to meaningful variable names during assignment operations.

```python
# Basic list unpacking into separate variables
numbers = [1, 2, 3]
first, second, third = numbers

print(f"First: {first}")    # Output: First: 1
print(f"Second: {second}")  # Output: Second: 2
print(f"Third: {third}")    # Output: Third: 3
```

Slide 2: Extended Unpacking with Asterisk Operator

The asterisk operator enables capturing multiple elements into a single variable during unpacking. This powerful feature allows flexible assignment patterns where you can extract both individual elements and subsequences from the original list simultaneously.

```python
# Extended unpacking with * operator
values = [1, 2, 3, 4, 5, 6]
first, *middle, last = values

print(f"First: {first}")   # Output: First: 1
print(f"Middle: {middle}") # Output: Middle: [2, 3, 4, 5]
print(f"Last: {last}")     # Output: Last: 6
```

Slide 3: Selective Unpacking with Underscore

When certain elements in a sequence are not needed, Python convention uses underscore as a placeholder variable. This technique helps improve code readability by explicitly indicating which values are intentionally ignored during unpacking operations.

```python
# Unpacking with placeholder for unwanted values
data = [100, 200, 300, 400, 500]
first, _, third, *_ = data

print(f"First: {first}")  # Output: First: 100
print(f"Third: {third}")  # Output: Third: 300
```

Slide 4: Function Argument Unpacking

Arguments can be unpacked directly into function calls using the asterisk operator. This feature enables passing sequence elements as separate positional arguments, making function calls more flexible and reducing code verbosity.

```python
def calculate_stats(x, y, z):
    return min(x, y, z), max(x, y, z)

values = [10, 5, 8]
minimum, maximum = calculate_stats(*values)

print(f"Minimum: {minimum}")  # Output: Minimum: 5
print(f"Maximum: {maximum}")  # Output: Maximum: 10
```

Slide 5: Dictionary Unpacking

Dictionary unpacking extends the concept to key-value pairs, allowing the merging of multiple dictionaries or passing dictionary items as keyword arguments to functions using the double asterisk operator.

```python
# Dictionary unpacking for merging and function calls
defaults = {'host': 'localhost', 'port': 8000}
custom = {'port': 9000, 'timeout': 30}

# Merge dictionaries with unpacking
config = {**defaults, **custom}
print(f"Config: {config}")  
# Output: Config: {'host': 'localhost', 'port': 9000, 'timeout': 30}
```

Slide 6: Parallel Assignment with Unpacking

Unpacking enables elegant parallel assignment operations, allowing simultaneous value swapping and multiple variable assignments in a single line. This feature significantly improves code readability in scenarios involving multiple variable updates.

```python
# Parallel assignment and swapping
a, b = 1, 2
print(f"Before: a={a}, b={b}")  # Output: Before: a=1, b=2

# Swap values using unpacking
a, b = b, a
print(f"After: a={a}, b={b}")   # Output: After: a=2, b=1

# Multiple assignments
x, y, z = [1, 2, 3]
print(f"x={x}, y={y}, z={z}")   # Output: x=1, y=2, z=3
```

Slide 7: Real-world Example - Data Processing

List unpacking proves invaluable in data processing scenarios, particularly when working with structured data like CSV records. This example demonstrates parsing and processing customer transaction records using various unpacking techniques.

```python
# Processing customer transaction records
transactions = [
    ('C001', 'Electronics', 499.99, '2024-01-15'),
    ('C002', 'Books', 29.99, '2024-01-16'),
    ('C003', 'Clothing', 89.99, '2024-01-16')
]

def process_transaction(customer_id, category, amount, date):
    return {
        'customer_id': customer_id,
        'category': category,
        'amount': amount,
        'date': date,
        'processed': True
    }

# Process transactions using unpacking
processed_records = [process_transaction(*transaction) 
                    for transaction in transactions]

print(processed_records[0])
# Output: {'customer_id': 'C001', 'category': 'Electronics', 
#          'amount': 499.99, 'date': '2024-01-15', 'processed': True}
```

Slide 8: Advanced Pattern Matching with Unpacking

Python's unpacking syntax combines powerfully with structural pattern matching, enabling sophisticated data extraction patterns. This feature is particularly useful when working with nested data structures and complex object hierarchies.

```python
# Advanced pattern matching with unpacking
def analyze_data_structure(data):
    match data:
        case [first, *rest] if isinstance(first, dict):
            print(f"Dictionary head: {first}")
            print(f"Remaining items: {rest}")
        case [x, y, *_] if isinstance(x, (int, float)):
            print(f"Numeric sequence starting with: {x}, {y}")
        case _:
            print("Unmatched pattern")

# Example usage
data1 = [{'id': 1}, 2, 3, 4]
data2 = [1.0, 2.0, 3.0, 4.0]

analyze_data_structure(data1)
# Output: Dictionary head: {'id': 1}
#         Remaining items: [2, 3, 4]

analyze_data_structure(data2)
# Output: Numeric sequence starting with: 1.0, 2.0
```

Slide 9: Nested Unpacking Patterns

Nested unpacking allows extraction of values from complex nested data structures in a single assignment statement. This advanced technique is particularly useful when working with JSON-like data structures or nested collections returned from APIs.

```python
# Nested unpacking with complex data structures
user_data = [
    "John Doe",
    [
        ("email", "john@example.com"),
        ("phone", "555-0123")
    ],
    {"age": 30, "active": True}
]

name, [(contact_type1, contact1), (contact_type2, contact2)], details = user_data

print(f"Name: {name}")           # Output: Name: John Doe
print(f"{contact_type1}: {contact1}")  # Output: email: john@example.com
print(f"Age: {details['age']}")  # Output: Age: 30
```

Slide 10: Generator Unpacking

Generator unpacking demonstrates how to efficiently process large datasets by unpacking iterator elements without loading the entire sequence into memory. This technique is crucial for memory-efficient data processing.

```python
def number_generator(start, end):
    for i in range(start, end):
        yield i ** 2

# Unpacking first few elements from generator
first, second, *rest = number_generator(1, 6)

print(f"First square: {first}")   # Output: First square: 1
print(f"Second square: {second}") # Output: Second square: 4
print(f"Rest squares: {rest}")    # Output: Rest squares: [9, 16, 25]

# Memory-efficient processing of large sequences
def process_large_dataset():
    gen = (x for x in range(1000000))
    head, *_ = gen  # Only first value is materialized
    return head

print(process_large_dataset())  # Output: 0
```

Slide 11: Real-world Example - Time Series Processing

This example demonstrates practical application of unpacking in time series data analysis, showing how to efficiently process and analyze sequential financial data points.

```python
# Time series data processing with unpacking
time_series = [
    (1641034800, 45000.0),  # Bitcoin price data
    (1641121200, 46500.0),
    (1641207600, 47200.0),
    (1641294000, 46800.0),
]

def analyze_price_movement(current, next_point):
    _, current_price = current
    _, next_price = next_point
    change = ((next_price - current_price) / current_price) * 100
    return f"{change:.2f}%"

# Analyze consecutive price movements using unpacking
*pairs, last = time_series
movements = []

for current, next_point in zip(pairs, time_series[1:]):
    movement = analyze_price_movement(current, next_point)
    movements.append(movement)

print(f"Price movements: {movements}")
# Output: Price movements: ['3.33%', '1.51%', '-0.85%']
```

Slide 12: Unpacking in List Comprehensions

List comprehensions combined with unpacking provide a powerful way to transform and filter complex data structures. This technique enables concise yet readable data transformations while maintaining code elegance.

```python
# Complex data transformation using unpacking in list comprehension
records = [
    ('AAPL', [150.5, 151.2, 149.8, 152.0]),
    ('GOOGL', [2800.0, 2795.5, 2810.2, 2805.8]),
    ('MSFT', [310.2, 312.5, 308.8, 315.0])
]

# Calculate daily returns using unpacking
daily_returns = [
    (symbol, [((prices[i+1] - prices[i])/prices[i])*100 
             for i in range(len(prices)-1)])
    for symbol, prices in records
]

for symbol, returns in daily_returns:
    print(f"{symbol} daily returns: {[f'{r:.2f}%' for r in returns]}")
# Output:
# AAPL daily returns: ['0.47%', '-0.93%', '1.47%']
# GOOGL daily returns: ['-0.16%', '0.52%', '-0.16%']
# MSFT daily returns: ['0.74%', '-1.18%', '2.01%']
```

Slide 13: Advanced Function Parameter Unpacking

Extended unpacking patterns in function parameters enable flexible argument handling and default value assignment. This advanced usage allows for more robust and maintainable function definitions.

```python
def analyze_metrics(*series, window_size=3, **kwargs):
    """Analyze time series with flexible parameter unpacking"""
    baseline, *variations = series
    
    # Process kwargs with unpacking
    config = {
        'normalize': False,
        'ignore_outliers': True,
        **kwargs
    }
    
    # Simulate analysis
    results = {
        'baseline': sum(baseline[:window_size])/window_size,
        'variations': [sum(var[:window_size])/window_size 
                      for var in variations],
        'config': config
    }
    return results

# Example usage
base = [1, 2, 3, 4]
var1 = [2, 3, 4, 5]
var2 = [3, 4, 5, 6]

result = analyze_metrics(base, var1, var2, window_size=2, 
                        normalize=True)
print(result)
# Output: {
#   'baseline': 1.5, 
#   'variations': [2.5, 3.5], 
#   'config': {'normalize': True, 'ignore_outliers': True}
# }
```

Slide 14: Additional Resources

*   Understanding Python's Unpacking Operators - [https://realpython.com/python-packing-unpacking-operators/](https://realpython.com/python-packing-unpacking-operators/)
*   Advanced Python Unpacking Techniques - [https://docs.python.org/3/tutorial/datastructures.html](https://docs.python.org/3/tutorial/datastructures.html)
*   Python Pattern Matching and Unpacking - [https://peps.python.org/pep-0636/](https://peps.python.org/pep-0636/)
*   ArXiv Paper: "Advanced Pattern Matching in Python" - [https://arxiv.org/search/cs?query=python+pattern+matching](https://arxiv.org/search/cs?query=python+pattern+matching)
*   Modern Python Features - [https://github.com/topics/python-features](https://github.com/topics/python-features)

