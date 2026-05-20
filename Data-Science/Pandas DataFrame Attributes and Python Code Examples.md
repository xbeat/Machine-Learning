## Pandas DataFrame Attributes and Python Code Examples
Slide 1: Pandas DataFrame Attributes

DataFrames are the most commonly used data structure in pandas. They are two-dimensional labeled data structures with columns of potentially different types. Understanding DataFrame attributes is crucial for effective data manipulation and analysis.

```python
import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Paris']
})

print(df)
```

Slide 2: DataFrame.shape

The shape attribute returns a tuple representing the dimensionality of the DataFrame. It provides the number of rows and columns in the DataFrame.

```python
# Get the shape of the DataFrame
shape = df.shape

print(f"Number of rows: {shape[0]}")
print(f"Number of columns: {shape[1]}")
```

Slide 3: DataFrame.dtypes

The dtypes attribute returns the data types of each column in the DataFrame. This is essential for understanding the nature of your data and performing appropriate operations.

```python
# Display the data types of each column
print(df.dtypes)

# Change the data type of a column
df['Age'] = df['Age'].astype(float)
print(df.dtypes)
```

Slide 4: DataFrame.index

The index attribute represents the row labels of the DataFrame. It can be customized to use meaningful identifiers instead of default integer indices.

```python
# Display the current index
print(df.index)

# Set a custom index
df.set_index('Name', inplace=True)
print(df)
print(df.index)
```

Slide 5: DataFrame.columns

The columns attribute returns the column labels of the DataFrame. It can be used to access, modify, or rename columns.

```python
# Display column names
print(df.columns)

# Rename columns
df.columns = ['Years', 'Location']
print(df)
```

Slide 6: DataFrame.values

The values attribute returns a NumPy array containing the data in the DataFrame. This is useful when you need to perform operations that require a pure NumPy array.

```python
# Get the values as a NumPy array
array_data = df.values
print(array_data)
print(type(array_data))
```

Slide 7: DataFrame.empty

The empty attribute returns a boolean indicating whether the DataFrame is empty (contains no data). This is useful for error checking and flow control in data processing pipelines.

```python
# Check if the DataFrame is empty
print(f"Is the DataFrame empty? {df.empty}")

# Create an empty DataFrame
empty_df = pd.DataFrame()
print(f"Is the new DataFrame empty? {empty_df.empty}")
```

Slide 8: DataFrame.size

The size attribute returns the total number of elements in the DataFrame. It is equal to the number of rows multiplied by the number of columns.

```python
# Get the size of the DataFrame
print(f"Total number of elements: {df.size}")

# Verify the calculation
total_elements = df.shape[0] * df.shape[1]
print(f"Calculated total elements: {total_elements}")
```

Slide 9: DataFrame.ndim

The ndim attribute returns the number of dimensions of the DataFrame. For a standard DataFrame, this will always be 2 (rows and columns).

```python
# Get the number of dimensions
print(f"Number of dimensions: {df.ndim}")

# Create a Series (1-dimensional) for comparison
series = pd.Series([1, 2, 3])
print(f"Number of dimensions in a Series: {series.ndim}")
```

Slide 10: DataFrame.axes

The axes attribute returns a list of the row axis labels and column axis labels. This can be useful for understanding the structure of your DataFrame.

```python
# Get the axes of the DataFrame
axes = df.axes
print(f"Row labels: {axes[0]}")
print(f"Column labels: {axes[1]}")
```

Slide 11: DataFrame.info()

While not strictly an attribute, the info() method provides a concise summary of the DataFrame, including the index dtype and column dtypes, non-null values, and memory usage.

```python
# Display DataFrame info
df.info()

# Display DataFrame info with memory usage
df.info(memory_usage="deep")
```

Slide 12: Real-life Example: Weather Data Analysis

Let's use DataFrame attributes to analyze weather data for different cities.

```python
import pandas as pd
import numpy as np

# Create a DataFrame with weather data
weather_data = pd.DataFrame({
    'City': ['Tokyo', 'New York', 'London', 'Paris'],
    'Temperature': [25.5, 22.1, 18.7, 20.3],
    'Humidity': [60, 55, 70, 65],
    'Wind_Speed': [10.2, 8.5, 12.1, 9.8]
})

print(weather_data)
print(f"\nShape: {weather_data.shape}")
print(f"\nData Types:\n{weather_data.dtypes}")
print(f"\nColumn Names: {weather_data.columns}")
```

Slide 13: Real-life Example: Student Performance Analysis

Let's use DataFrame attributes to analyze student performance data.

```python
# Create a DataFrame with student performance data
student_data = pd.DataFrame({
    'Student_ID': ['S001', 'S002', 'S003', 'S004', 'S005'],
    'Math_Score': [85, 92, 78, 95, 88],
    'Science_Score': [90, 88, 82, 96, 85],
    'Literature_Score': [75, 85, 92, 88, 91]
})

student_data.set_index('Student_ID', inplace=True)
print(student_data)
print(f"\nIndex: {student_data.index}")
print(f"\nSize: {student_data.size}")
print(f"\nMean Scores:\n{student_data.mean()}")
```

Slide 14: Additional Resources

For more advanced topics and in-depth explanations of pandas DataFrame attributes, consider exploring the following resources:

1. Official pandas documentation: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
2. "Effective Pandas" by Matt Harrison: [https://github.com/mattharrison/effective\_pandas](https://github.com/mattharrison/effective_pandas)
3. "Python for Data Analysis" by Wes McKinney (creator of pandas): [https://wesmckinney.com/book/](https://wesmckinney.com/book/)

These resources provide comprehensive coverage of pandas and its capabilities, helping you master DataFrame manipulation and analysis.

