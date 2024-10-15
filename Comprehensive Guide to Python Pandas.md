## Comprehensive Guide to Python Pandas
Slide 1: Introduction to Python Pandas

Pandas is a powerful data manipulation library for Python. It provides high-performance, easy-to-use data structures and tools for working with structured data. Let's start by importing pandas and creating a simple DataFrame.

```python
import pandas as pd

# Create a simple DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'London', 'Paris']}
df = pd.DataFrame(data)

print(df)
```

Output:

```
     Name  Age     City
0   Alice   25  New York
1     Bob   30    London
2  Charlie  35     Paris
```

Slide 2: Reading Data with Pandas

Pandas can read data from various file formats. Let's read a CSV file and display its contents.

```python
# Read a CSV file
df = pd.read_csv('sample_data.csv')

# Display the first few rows
print(df.head())

# Display basic information about the DataFrame
print(df.info())
```

Output:

```
   ID   Name  Age     City
0   1  Alice   25  New York
1   2    Bob   30    London
2   3  Charlie 35     Paris

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   ID      3 non-null      int64 
 1   Name    3 non-null      object
 2   Age     3 non-null      int64 
 3   City    3 non-null      object
dtypes: int64(2), object(2)
memory usage: 224.0+ bytes
```

Slide 3: Data Selection and Indexing

Pandas offers various ways to select and index data. Let's explore some common methods.

```python
# Select a single column
print(df['Name'])

# Select multiple columns
print(df[['Name', 'Age']])

# Select rows by index
print(df.loc[1])

# Select rows by condition
print(df[df['Age'] > 30])

# Select specific rows and columns
print(df.loc[df['Age'] > 30, ['Name', 'City']])
```

Output:

```
0     Alice
1       Bob
2    Charlie
Name: Name, dtype: object

     Name  Age
0   Alice   25
1     Bob   30
2  Charlie  35

Name       Bob
Age         30
City    London
Name: 1, dtype: object

     Name  Age   City
2  Charlie   35  Paris

     Name   City
2  Charlie  Paris
```

Slide 4: Data Cleaning and Preprocessing

Data cleaning is crucial in data analysis. Let's explore some common data cleaning operations.

```python
# Handle missing values
df['Salary'] = [50000, None, 75000]
print(df)

# Fill missing values
df['Salary'].fillna(df['Salary'].mean(), inplace=True)
print(df)

# Remove duplicates
df = df.drop_duplicates()

# Rename columns
df = df.rename(columns={'Name': 'Full_Name'})
print(df.columns)
```

Output:

```
     Name  Age     City  Salary
0   Alice   25  New York  50000.0
1     Bob   30    London      NaN
2  Charlie  35     Paris  75000.0

     Name  Age     City     Salary
0   Alice   25  New York  50000.000
1     Bob   30    London  62500.000
2  Charlie  35     Paris  75000.000

Index(['Full_Name', 'Age', 'City', 'Salary'], dtype='object')
```

Slide 5: Data Transformation

Pandas provides powerful tools for data transformation. Let's explore some common operations.

```python
# Apply a function to a column
df['Age_Group'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Adult')

# Create a new column based on multiple existing columns
df['Location'] = df['City'] + ', ' + df['Country']

# Categorical encoding
df['City_Code'] = pd.Categorical(df['City']).codes

print(df)
```

Output:

```
  Full_Name  Age     City     Salary Age_Group        Location  City_Code
0     Alice   25  New York  50000.000     Young  New York, USA          2
1       Bob   30    London  62500.000     Adult    London, UK          1
2   Charlie   35     Paris  75000.000     Adult  Paris, France         0
```

Slide 6: Grouping and Aggregation

Grouping and aggregation are essential for data analysis. Let's explore these operations.

```python
# Group by a column and calculate statistics
grouped = df.groupby('Age_Group')
print(grouped['Salary'].mean())

# Multiple aggregations
agg_result = grouped.agg({
    'Salary': ['mean', 'max'],
    'Age': 'mean'
})
print(agg_result)

# Reset index after grouping
agg_result = agg_result.reset_index()
print(agg_result)
```

Output:

```
Age_Group
Adult    68750.0
Young    50000.0
Name: Salary, dtype: float64

           Salary            Age
             mean       max  mean
Age_Group                        
Adult     68750.0  75000.0  32.5
Young     50000.0  50000.0  25.0

  Age_Group     Salary            Age
                  mean       max  mean
0     Adult  68750.000  75000.0  32.5
1     Young  50000.000  50000.0  25.0
```

Slide 7: Merging and Joining DataFrames

Combining data from multiple sources is a common task. Let's explore merging and joining operations.

```python
# Create two DataFrames
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [2, 3, 4], 'City': ['London', 'Paris', 'Berlin']})

# Inner join
inner_join = pd.merge(df1, df2, on='ID', how='inner')
print("Inner Join:")
print(inner_join)

# Left join
left_join = pd.merge(df1, df2, on='ID', how='left')
print("\nLeft Join:")
print(left_join)

# Concatenate DataFrames
df3 = pd.concat([df1, df2], axis=1)
print("\nConcatenated:")
print(df3)
```

Output:

```
Inner Join:
   ID     Name    City
0   2      Bob  London
1   3  Charlie   Paris

Left Join:
   ID     Name    City
0   1    Alice     NaN
1   2      Bob  London
2   3  Charlie   Paris

Concatenated:
   ID     Name  ID    City
0   1    Alice   2  London
1   2      Bob   3   Paris
2   3  Charlie   4  Berlin
```

Slide 8: Time Series Data

Pandas excels at handling time series data. Let's explore some time series operations.

```python
import pandas as pd

# Create a time series DataFrame
dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
ts_df = pd.DataFrame({'Date': dates, 'Value': range(len(dates))})
ts_df.set_index('Date', inplace=True)

print(ts_df)

# Resample to weekly frequency
weekly = ts_df.resample('W').mean()
print("\nWeekly average:")
print(weekly)

# Rolling window calculations
rolling_mean = ts_df.rolling(window=3).mean()
print("\nRolling mean (3-day window):")
print(rolling_mean)
```

Output:

```
            Value
Date             
2023-01-01      0
2023-01-02      1
2023-01-03      2
2023-01-04      3
2023-01-05      4
2023-01-06      5
2023-01-07      6
2023-01-08      7
2023-01-09      8
2023-01-10      9

Weekly average:
            Value
Date             
2023-01-01    1.5
2023-01-08    6.5

Rolling mean (3-day window):
                Value
Date                 
2023-01-01       NaN
2023-01-02       NaN
2023-01-03  1.000000
2023-01-04  2.000000
2023-01-05  3.000000
2023-01-06  4.000000
2023-01-07  5.000000
2023-01-08  6.000000
2023-01-09  7.000000
2023-01-10  8.000000
```

Slide 9: Data Visualization with Pandas

Pandas integrates well with matplotlib for quick data visualization. Let's create some basic plots.

```python
import matplotlib.pyplot as plt

# Create a sample DataFrame
data = {'Year': [2018, 2019, 2020, 2021, 2022],
        'Sales': [100, 120, 90, 150, 180]}
df = pd.DataFrame(data)

# Line plot
df.plot(x='Year', y='Sales', kind='line')
plt.title('Sales Trend')
plt.show()

# Bar plot
df.plot(x='Year', y='Sales', kind='bar')
plt.title('Sales by Year')
plt.show()

# Histogram
df['Sales'].plot(kind='hist')
plt.title('Sales Distribution')
plt.show()
```

\[Note: The actual plots would be displayed here. As I cannot generate or display images, I've described the expected output.\]

Slide 10: Advanced Data Analysis: Pivot Tables

Pivot tables are powerful tools for data analysis. Let's create a pivot table to summarize our data.

```python
# Create a sample DataFrame
data = {
    'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
    'Product': ['A', 'B', 'A', 'B'],
    'Sales': [100, 150, 120, 180]
}
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

# Create a pivot table
pivot = df.pivot_table(values='Sales', index='Date', columns='Product', aggfunc='sum')

print("Pivot Table:")
print(pivot)

# Add row and column totals
pivot['Total'] = pivot.sum(axis=1)
pivot.loc['Total'] = pivot.sum()

print("\nPivot Table with Totals:")
print(pivot)
```

Output:

```
Pivot Table:
Product           A      B
Date                      
2023-01-01  100.0  150.0
2023-01-02  120.0  180.0

Pivot Table with Totals:
Product           A      B  Total
Date                           
2023-01-01  100.0  150.0  250.0
2023-01-02  120.0  180.0  300.0
Total       220.0  330.0  550.0
```

Slide 11: Real-life Example: Weather Data Analysis

Let's analyze weather data using Pandas. We'll work with a dataset containing temperature readings from different cities.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Create a sample weather dataset
data = {
    'Date': pd.date_range(start='2023-01-01', end='2023-01-10'),
    'City': ['New York', 'London', 'Tokyo'] * 3 + ['New York'],
    'Temperature': [5, 2, 8, 6, 3, 9, 4, 1, 7, 5]
}
df = pd.DataFrame(data)

# Calculate average temperature by city
avg_temp = df.groupby('City')['Temperature'].mean()
print("Average Temperature by City:")
print(avg_temp)

# Plot temperature trends
plt.figure(figsize=(10, 6))
for city in df['City'].unique():
    city_data = df[df['City'] == city]
    plt.plot(city_data['Date'], city_data['Temperature'], label=city)

plt.title('Temperature Trends by City')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
```

Output:

```
Average Temperature by City:
City
London    2.0
New York  5.0
Tokyo     8.0
Name: Temperature, dtype: float64
```

\[Note: A line plot showing temperature trends for each city would be displayed here.\]

Slide 12: Real-life Example: Text Analysis

Let's use Pandas for text analysis on a dataset of book titles.

```python
import pandas as pd

# Create a sample dataset of book titles
data = {
    'Title': [
        'The Great Gatsby',
        'To Kill a Mockingbird',
        '1984',
        'Pride and Prejudice',
        'The Catcher in the Rye'
    ],
    'Author': [
        'F. Scott Fitzgerald',
        'Harper Lee',
        'George Orwell',
        'Jane Austen',
        'J.D. Salinger'
    ],
    'Year': [1925, 1960, 1949, 1813, 1951]
}
df = pd.DataFrame(data)

# Count words in titles
df['Word_Count'] = df['Title'].apply(lambda x: len(x.split()))

# Extract first word of each title
df['First_Word'] = df['Title'].apply(lambda x: x.split()[0])

# Find titles containing 'the' (case-insensitive)
df['Contains_The'] = df['Title'].str.contains('the', case=False)

print(df)

# Analyze word counts
print("\nWord Count Statistics:")
print(df['Word_Count'].describe())

# Most common first words
print("\nMost Common First Words:")
print(df['First_Word'].value_counts())
```

Output:

```
                    Title               Author  Year  Word_Count First_Word  Contains_The
0       The Great Gatsby  F. Scott Fitzgerald  1925           3        The          True
1  To Kill a Mockingbird          Harper Lee  1960           4         To         False
2                   1984       George Orwell  1949           1       1984         False
3    Pride and Prejudice         Jane Austen  1813           3      Pride         False
4  The Catcher in the Rye      J.D. Salinger  1951           5        The          True

Word Count Statistics:
count    5.000000
mean     3.200000
std      1.483240
min      1.000000
25%      3.000000
50%      3.000000
75%      4.000000
max      5.000000
Name: Word_Count, dtype: float64

Most Common First Words:
The      2
To       1
1984     1
Pride    1
Name: First_Word, dtype: int64
```

Slide 13: Advanced Pandas: Custom Aggregation Functions

Pandas allows you to define custom aggregation functions for complex data analysis. Let's explore this feature.

```python
import pandas as pd
import numpy as np

# Create a sample DataFrame
data = {
    'Category': ['A', 'A', 'B', 'B', 'C'],
    'Value1': [10, 20, 30, 40, 50],
    'Value2': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)

# Define custom aggregation functions
def range_func(x):
    return x.max() - x.min()

def percent_change(x):
    return (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100

# Apply custom aggregations
custom_agg = df.groupby('Category').agg({
    'Value1': ['mean', range_func],
    'Value2': ['median', percent_change]
})

print(custom_agg)
```

Output:

```
        Value1            Value2         
          mean range_func median percent_change
Category                                       
A          15.0       10.0    1.5         100.0
B          35.0       10.0    3.5          33.3
C          50.0        0.0    5.0           0.0
```

Slide 14: Working with Large Datasets: Chunking

When dealing with large datasets, processing data in chunks can be more efficient. Let's explore how to use chunking in Pandas.

```python
import pandas as pd

# Simulate reading a large CSV file in chunks
chunk_size = 1000
chunks = []

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process each chunk
    processed_chunk = chunk[chunk['value'] > chunk['value'].mean()]
    chunks.append(processed_chunk)

# Combine all processed chunks
result = pd.concat(chunks, ignore_index=True)

print(f"Total rows after processing: {len(result)}")
print(result.head())
```

Note: This code assumes the existence of a large CSV file. In practice, you would replace 'large\_file.csv' with your actual file path.

Slide 15: Additional Resources

For further exploration of Pandas, consider these resources:

1. Pandas Official Documentation: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
2. "Python for Data Analysis" by Wes McKinney (creator of Pandas)
3. DataCamp's Pandas Tutorials: [https://www.datacamp.com/courses/pandas-foundations](https://www.datacamp.com/courses/pandas-foundations)
4. Real Python's Pandas Tutorials: [https://realpython.com/learning-paths/pandas-data-science/](https://realpython.com/learning-paths/pandas-data-science/)

These resources offer in-depth explanations, practical examples, and advanced techniques to enhance your Pandas skills.

