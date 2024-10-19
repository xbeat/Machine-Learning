## Identifying and Removing Duplicates in Pandas
Slide 1: Understanding Duplicates in Pandas

Duplicates in Pandas DataFrames can significantly impact data analysis by skewing results and leading to incorrect conclusions. This presentation will explore methods to identify and remove duplicates, ensuring data integrity and accuracy in your analysis.

```python
import pandas as pd
import numpy as np

# Create a sample DataFrame with duplicates
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David', 'Bob'],
    'Age': [25, 30, 35, 25, 40, 30],
    'City': ['New York', 'London', 'Paris', 'New York', 'Tokyo', 'London']
}
df = pd.DataFrame(data)
print(df)
```

Slide 2: The duplicated() Method

The duplicated() method in Pandas is used to identify duplicate rows in a DataFrame. It returns a Boolean Series where True indicates a duplicate row.

```python
# Identify duplicate rows
duplicates = df.duplicated()
print(duplicates)

# Display rows marked as duplicates
print(df[duplicates])
```

Slide 3: Customizing duplicated() Parameters

The duplicated() method offers parameters to fine-tune duplicate detection:

*   subset: Specify columns to consider for duplicates
*   keep: Choose which duplicate to mark ('first', 'last', or False)

```python
# Check duplicates based on 'Name' and 'Age' columns
name_age_dupes = df.duplicated(subset=['Name', 'Age'], keep='last')
print(df[name_age_dupes])
```

Slide 4: The drop\_duplicates() Method

The drop\_duplicates() method removes duplicate rows from a DataFrame, returning a new DataFrame with unique rows.

```python
# Remove duplicate rows
df_unique = df.drop_duplicates()
print(df_unique)
```

Slide 5: Customizing drop\_duplicates() Parameters

Similar to duplicated(), drop\_duplicates() offers parameters for fine-tuning:

*   subset: Specify columns to consider for duplicates
*   keep: Choose which duplicate to keep ('first', 'last', or False)
*   inplace: Modify the original DataFrame instead of creating a copy
*   ignore\_index: Reset the index of the resulting DataFrame

```python
# Remove duplicates based on 'Name' and 'Age', keeping the last occurrence
df_unique_custom = df.drop_duplicates(subset=['Name', 'Age'], keep='last', ignore_index=True)
print(df_unique_custom)
```

Slide 6: Handling Duplicates in Large DataFrames

For large DataFrames, it's crucial to consider performance. Using a subset of columns can significantly speed up the process.

```python
# Generate a large DataFrame with duplicates
large_df = pd.DataFrame({
    'ID': np.random.randint(1, 1000, 100000),
    'Value': np.random.rand(100000)
})

# Time the operation with and without subset
import time

start = time.time()
large_df.drop_duplicates()
print(f"Without subset: {time.time() - start:.4f} seconds")

start = time.time()
large_df.drop_duplicates(subset=['ID'])
print(f"With subset: {time.time() - start:.4f} seconds")
```

Slide 7: Identifying Duplicate Patterns

Understanding the pattern of duplicates can provide insights into data quality issues. Let's explore how to analyze duplicate occurrences.

```python
# Count occurrences of each row
value_counts = df.groupby(df.columns.tolist()).size().reset_index(name='count')
print(value_counts[value_counts['count'] > 1])
```

Slide 8: Visualizing Duplicates

Visualizing the distribution of duplicates can help in understanding the scale of the problem.

```python
import matplotlib.pyplot as plt

# Count duplicates per column
duplicate_counts = df.duplicated(subset=df.columns, keep=False).sum()

plt.figure(figsize=(10, 6))
duplicate_counts.plot(kind='bar')
plt.title('Number of Duplicates per Column')
plt.xlabel('Columns')
plt.ylabel('Number of Duplicates')
plt.tight_layout()
plt.show()
```

Slide 9: Handling Partial Duplicates

Sometimes, rows may be partially duplicate. Let's explore how to handle such cases.

```python
# Create a DataFrame with partial duplicates
partial_dupes = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Alice', 'Charlie'],
    'Age': [25, 30, 25, 35],
    'City': ['New York', 'London', 'Chicago', 'Paris']
})

# Find rows with duplicate Names and Ages, but different Cities
partial_dupes[partial_dupes.duplicated(subset=['Name', 'Age'], keep=False) & 
              ~partial_dupes.duplicated(keep=False)]
```

Slide 10: Real-Life Example: Weather Data

Consider a weather dataset with potential duplicate readings. We'll clean the data and prepare it for analysis.

```python
# Sample weather data with duplicates
weather_data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=10).repeat(2),
    'Temperature': [20, 20, 22, 22, 19, 19, 21, 21, 23, 23, 18, 18, 20, 20, 22, 22, 21, 21, 19, 19],
    'Humidity': [50, 50, 55, 54, 52, 52, 53, 53, 51, 51, 49, 49, 50, 50, 54, 54, 53, 53, 52, 52]
})

# Remove exact duplicates
clean_weather = weather_data.drop_duplicates()

# Remove duplicates based on Date and Temperature, keeping the first occurrence
clean_weather = clean_weather.drop_duplicates(subset=['Date', 'Temperature'], keep='first')

print(clean_weather)
```

Slide 11: Real-Life Example: Sensor Data

Let's examine a dataset from multiple sensors, where some sensors might report duplicate readings.

```python
# Sample sensor data with duplicates
sensor_data = pd.DataFrame({
    'Timestamp': pd.date_range(start='2023-01-01', periods=12, freq='H').repeat(3),
    'Sensor_ID': ['A', 'B', 'C'] * 12,
    'Reading': np.random.rand(36).round(2)
})

# Add some duplicates
sensor_data.loc[5:7, 'Reading'] = sensor_data.loc[2:4, 'Reading'].values

# Remove duplicates, keeping the first reading for each Sensor_ID at each Timestamp
clean_sensor_data = sensor_data.drop_duplicates(subset=['Timestamp', 'Sensor_ID'], keep='first')

print(clean_sensor_data)
```

Slide 12: Handling Time-Based Duplicates

In time-series data, we might want to remove duplicates within a specific time window.

```python
import pandas as pd

# Sample time-series data with close timestamps
time_data = pd.DataFrame({
    'Timestamp': pd.to_datetime(['2023-01-01 12:00:00', '2023-01-01 12:00:01', 
                                 '2023-01-01 12:05:00', '2023-01-01 12:05:02']),
    'Value': [100, 101, 105, 106]
})

# Function to remove duplicates within a 1-minute window
def remove_time_duplicates(df, time_window='1min'):
    return df.resample(time_window, on='Timestamp').first().dropna().reset_index()

cleaned_time_data = remove_time_duplicates(time_data)
print(cleaned_time_data)
```

Slide 13: Efficiency Considerations

When working with large datasets, memory usage becomes crucial. Let's explore an efficient way to handle duplicates in chunks.

```python
import pandas as pd

# Function to process large CSV files in chunks
def process_large_csv(file_path, chunk_size=10000):
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Remove duplicates in each chunk
        chunk_no_dupes = chunk.drop_duplicates()
        chunks.append(chunk_no_dupes)
    
    # Combine all chunks and remove any remaining duplicates
    result = pd.concat(chunks, ignore_index=True)
    return result.drop_duplicates()

# Usage (commented out as we don't have the actual file)
# clean_data = process_large_csv('large_data.csv')
# print(clean_data.shape)
```

Slide 14: Additional Resources

For more in-depth information on handling duplicates and data cleaning in Pandas, consider exploring these resources:

1.  Pandas Documentation on Duplicate Handling: [https://pandas.pydata.org/pandas-docs/stable/user\_guide/duplicates.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/duplicates.html)
2.  "Efficient Data Cleaning in Python" by Sarah Guido: [https://arxiv.org/abs/2006.01661](https://arxiv.org/abs/2006.01661)
3.  "A Survey on Data Preprocessing for Data Mining" by S. García et al.: [https://arxiv.org/abs/1511.03980](https://arxiv.org/abs/1511.03980)

These resources provide comprehensive guides and research on data cleaning techniques, including advanced methods for handling duplicates in various scenarios.

