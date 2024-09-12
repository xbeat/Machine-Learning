## Data Preparation with Python's singledispatch
Slide 1: Introduction to singledispatch

The singledispatch decorator in Python allows you to create generic functions that can handle different types of input. It's particularly useful for data preparation tasks where you need to process various data types uniformly. Let's explore how to use this powerful feature.

```python
from functools import singledispatch

@singledispatch
def process_data(data):
    raise NotImplementedError("Unsupported data type")
```

Slide 2: Basic Usage of singledispatch

The singledispatch decorator is applied to a base function, which serves as a default implementation or fallback. We then register specialized functions for specific types using the `register` method.

```python
@process_data.register(str)
def _(data: str):
    return f"Processing string: {data.upper()}"

@process_data.register(int)
def _(data: int):
    return f"Processing integer: {data * 2}"
```

Slide 3: Handling Multiple Data Types

With singledispatch, we can easily handle different data types without complex if-else statements. This leads to cleaner and more maintainable code, especially when dealing with various data formats in data preparation tasks.

```python
print(process_data("hello"))  # Output: Processing string: HELLO
print(process_data(5))        # Output: Processing integer: 10
print(process_data([1, 2, 3]))  # Raises NotImplementedError
```

Slide 4: Registering List Handlers

Let's add support for processing lists. We can define different behaviors based on the list contents, demonstrating the flexibility of singledispatch.

```python
@process_data.register(list)
def _(data: list):
    if all(isinstance(item, str) for item in data):
        return f"Processing string list: {', '.join(data).upper()}"
    elif all(isinstance(item, int) for item in data):
        return f"Processing integer list: {sum(data)}"
    else:
        return "Mixed list: unable to process"
```

Slide 5: Handling Complex Data Structures

In real-world scenarios, we often deal with more complex data structures. Let's add support for dictionaries, which are common in data preparation tasks.

```python
@process_data.register(dict)
def _(data: dict):
    processed = {k.upper(): process_data(v) for k, v in data.items()}
    return f"Processed dictionary: {processed}"

print(process_data({"name": "john", "age": 30}))
# Output: Processed dictionary: {'NAME': 'Processing string: JOHN', 'AGE': 'Processing integer: 60'}
```

Slide 6: Handling Missing Data

Data preparation often involves dealing with missing or null values. Let's add support for None type and show how to handle it gracefully.

```python
@process_data.register(type(None))
def _(data: None):
    return "Missing data: value is None"

print(process_data(None))  # Output: Missing data: value is None
```

Slide 7: Processing Numerical Data

When preparing data for analysis or machine learning, we often need to normalize or scale numerical data. Let's add support for float types with a simple normalization function.

```python
@process_data.register(float)
def _(data: float):
    def normalize(value, min_value=0, max_value=1):
        return (value - min_value) / (max_value - min_value)
    
    return f"Normalized float: {normalize(data):.2f}"

print(process_data(0.75))  # Output: Normalized float: 0.75
```

Slide 8: Handling Date and Time Data

Date and time data are common in many datasets. Let's add support for datetime objects to demonstrate how singledispatch can handle complex types.

```python
from datetime import datetime

@process_data.register(datetime)
def _(data: datetime):
    return f"Formatted date: {data.strftime('%Y-%m-%d %H:%M:%S')}"

print(process_data(datetime.now()))
# Output: Formatted date: 2024-07-10 12:34:56
```

Slide 9: Processing Nested Structures

Real-world data often contains nested structures. Let's enhance our list handler to process nested lists recursively.

```python
@process_data.register(list)
def _(data: list):
    return [process_data(item) for item in data]

nested_data = [1, "hello", [2, 3], {"key": "value"}]
print(process_data(nested_data))
# Output: ['Processing integer: 2', 'Processing string: HELLO', [4, 6], "Processed dictionary: {'KEY': 'Processing string: VALUE'}"]
```

Slide 10: Handling Custom Objects

In data preparation, we might encounter custom objects. Let's create a simple Person class and add support for it using singledispatch.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

@process_data.register(Person)
def _(data: Person):
    return f"Person: {process_data(data.name)}, Age: {process_data(data.age)}"

print(process_data(Person("Alice", 28)))
# Output: Person: Processing string: ALICE, Age: Processing integer: 56
```

Slide 11: Error Handling and Logging

Robust data preparation requires good error handling and logging. Let's enhance our process\_data function to include these features.

```python
import logging

@singledispatch
def process_data(data):
    logging.warning(f"Unsupported data type: {type(data)}")
    return f"Unable to process data of type: {type(data)}"

# Other registered functions remain the same

logging.basicConfig(level=logging.INFO)
print(process_data(complex(1, 2)))
# Logs a warning and outputs: Unable to process data of type: <class 'complex'>
```

Slide 12: Performance Considerations

When processing large datasets, performance becomes crucial. Let's compare the performance of singledispatch with a traditional if-else approach.

```python
import time

def traditional_process(data):
    if isinstance(data, str):
        return data.upper()
    elif isinstance(data, int):
        return data * 2
    # ... other type checks

start = time.time()
for _ in range(1000000):
    traditional_process("test")
print(f"Traditional: {time.time() - start:.4f} seconds")

start = time.time()
for _ in range(1000000):
    process_data("test")
print(f"Singledispatch: {time.time() - start:.4f} seconds")
```

Slide 13: Real-world Example: Data Cleaning Pipeline

Let's create a simple data cleaning pipeline using singledispatch to handle different types of data in a CSV file.

```python
import csv
from typing import Any

@singledispatch
def clean_data(data: Any) -> Any:
    return data  # Default: return as-is if no specific handler

@clean_data.register(str)
def _(data: str) -> str:
    return data.strip().lower()

@clean_data.register(float)
def _(data: float) -> float:
    return round(data, 2)

def process_csv(input_file: str, output_file: str):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            cleaned_row = [clean_data(item) for item in row]
            writer.writerow(cleaned_row)

# Usage:
# process_csv('input.csv', 'output.csv')
```

Slide 14: Real-world Example: Data Transformation for Machine Learning

In this example, we'll use singledispatch to preprocess different types of features for a machine learning model.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

@singledispatch
def preprocess_feature(feature):
    raise ValueError(f"Unsupported feature type: {type(feature)}")

@preprocess_feature.register(np.ndarray)
def _(feature: np.ndarray):
    if feature.dtype.kind in 'iuf':  # integer or float
        return StandardScaler().fit_transform(feature.reshape(-1, 1))
    elif feature.dtype.kind in 'OSU':  # string or unicode
        return OneHotEncoder(sparse=False).fit_transform(feature.reshape(-1, 1))

@preprocess_feature.register(list)
def _(feature: list):
    return preprocess_feature(np.array(feature))

# Usage
numeric_feature = [1, 2, 3, 4, 5]
categorical_feature = ['red', 'blue', 'green', 'red', 'blue']

print(preprocess_feature(numeric_feature))
print(preprocess_feature(categorical_feature))
```

Slide 15: Additional Resources

For more information on singledispatch and its applications in data preparation, consider exploring the following resources:

1. Python's official documentation on singledispatch: [https://docs.python.org/3/library/functools.html#functools.singledispatch](https://docs.python.org/3/library/functools.html#functools.singledispatch)
2. "Effective Python: 90 Specific Ways to Write Better Python" by Brett Slatkin, which covers advanced uses of singledispatch.
3. The paper "Generic Programming in Python with More Protocols" by Luciano Ramalho, available on arXiv: [https://arxiv.org/abs/2004.06058](https://arxiv.org/abs/2004.06058)

These resources provide deeper insights into the singledispatch mechanism and its practical applications in various programming scenarios, including data preparation tasks.

