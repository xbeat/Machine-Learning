## Pandas for Data Wrangling and Analysis
Slide 1: Introduction to Pandas

Pandas is a powerful Python library for data manipulation and analysis. It provides data structures like DataFrame and Series, which allow for efficient handling of structured data. Pandas simplifies tasks such as data cleaning, transformation, merging, and analysis, making it an essential tool for data scientists and analysts.

```python
import pandas as pd

# Create a simple DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'London', 'Paris']}
df = pd.DataFrame(data)

print(df)
```

Slide 2: Creating DataFrames

DataFrames are the primary data structure in Pandas. They represent tabular data with labeled rows and columns. You can create DataFrames from various data sources, including dictionaries, lists, or external files.

```python
# Create a DataFrame from a dictionary
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)

# Create a DataFrame from a list of lists
data = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
df = pd.DataFrame(data, columns=['A', 'B', 'C'])

print(df)
```

Slide 3: Loading Data from External Sources

Pandas provides functions to read data from various file formats, including CSV, Excel, and JSON. This allows you to easily import data from external sources into your Python environment for analysis.

```python
# Read data from a CSV file
df_csv = pd.read_csv('data.csv')

# Read data from an Excel file
df_excel = pd.read_excel('data.xlsx')

# Read data from a JSON file
df_json = pd.read_json('data.json')

print(df_csv.head())
```

Slide 4: Data Inspection

After loading data, it's crucial to inspect it to understand its structure and content. Pandas offers several methods to quickly examine your DataFrame.

```python
# Display the first few rows
print(df.head())

# Get basic information about the DataFrame
print(df.info())

# Display summary statistics
print(df.describe())

# Check the shape of the DataFrame
print(df.shape)
```

Slide 5: Handling Missing Data

Missing data is a common issue in real-world datasets. Pandas provides methods to identify and handle missing values effectively.

```python
# Create a DataFrame with missing values
df = pd.DataFrame({'A': [1, 2, None, 4], 'B': [5, None, 7, 8]})

# Fill missing values with a specific value
df_filled = df.fillna(0)

# Drop rows with missing values
df_dropped = df.dropna()

print("Original DataFrame:")
print(df)
print("\nFilled DataFrame:")
print(df_filled)
print("\nDropped DataFrame:")
print(df_dropped)
```

Slide 6: Data Selection and Indexing

Pandas offers powerful ways to select and index data within a DataFrame. You can access data by label, position, or boolean indexing.

```python
# Create a sample DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]},
                  index=['x', 'y', 'z'])

# Select a column
print(df['A'])

# Select multiple columns
print(df[['A', 'B']])

# Select rows by label
print(df.loc['x'])

# Select rows and columns by position
print(df.iloc[0, 1])

# Boolean indexing
print(df[df['A'] > 1])
```

Slide 7: Data Transformation

Data transformation is a crucial step in data analysis. Pandas provides various methods to modify and reshape your data.

```python
# Create a sample DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Apply a function to a column
df['C'] = df['A'].apply(lambda x: x * 2)

# Rename columns
df = df.rename(columns={'A': 'X', 'B': 'Y'})

# Add a new column based on existing ones
df['Z'] = df['X'] + df['Y']

print(df)
```

Slide 8: Grouping and Aggregation

Grouping and aggregation are powerful techniques for summarizing data. Pandas makes it easy to group data by one or more columns and apply aggregate functions.

```python
# Create a sample DataFrame
df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'A'],
    'Value': [10, 20, 30, 40, 50]
})

# Group by Category and calculate mean
grouped = df.groupby('Category')['Value'].mean()

# Group by Category and apply multiple aggregations
agg_funcs = {'Value': ['mean', 'sum', 'count']}
result = df.groupby('Category').agg(agg_funcs)

print("Grouped mean:")
print(grouped)
print("\nMultiple aggregations:")
print(result)
```

Slide 9: Merging and Joining DataFrames

Combining data from multiple sources is a common task in data analysis. Pandas provides various methods to merge and join DataFrames based on common columns or indices.

```python
# Create two sample DataFrames
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})

# Merge DataFrames on the 'key' column
merged = pd.merge(df1, df2, on='key', how='outer')

# Join DataFrames based on index
df3 = pd.DataFrame({'value': [7, 8, 9]}, index=['A', 'B', 'E'])
joined = df1.set_index('key').join(df3, how='outer')

print("Merged DataFrame:")
print(merged)
print("\nJoined DataFrame:")
print(joined)
```

Slide 10: Pivot Tables and Reshaping Data

Pivot tables are useful for summarizing and analyzing data. Pandas provides functions to create pivot tables and reshape data between wide and long formats.

```python
# Create a sample DataFrame
df = pd.DataFrame({
    'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
    'Product': ['A', 'B', 'A', 'B'],
    'Sales': [100, 150, 120, 180]
})

# Create a pivot table
pivot = df.pivot_table(values='Sales', index='Date', columns='Product', aggfunc='sum')

# Melt the DataFrame from wide to long format
melted = pd.melt(df, id_vars=['Date'], value_vars=['Sales'], var_name='Metric', value_name='Value')

print("Pivot Table:")
print(pivot)
print("\nMelted DataFrame:")
print(melted)
```

Slide 11: Time Series Analysis

Pandas excels at handling time series data. It provides powerful tools for working with dates, times, and time-based operations.

```python
# Create a time series DataFrame
dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
ts = pd.DataFrame({'Value': range(len(dates))}, index=dates)

# Resample to weekly frequency
weekly = ts.resample('W').sum()

# Shift the time series
shifted = ts.shift(periods=2)

# Calculate rolling mean
rolling_mean = ts.rolling(window=3).mean()

print("Original Time Series:")
print(ts)
print("\nWeekly Resampled:")
print(weekly)
print("\nShifted Time Series:")
print(shifted)
print("\nRolling Mean:")
print(rolling_mean)
```

Slide 12: Data Visualization with Pandas

Pandas integrates well with matplotlib, allowing you to create quick visualizations directly from your DataFrames.

```python
import matplotlib.pyplot as plt

# Create a sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [2, 4, 6, 8, 10],
    'C': [3, 6, 9, 12, 15]
})

# Create a line plot
df.plot(kind='line')
plt.title('Line Plot')
plt.show()

# Create a bar plot
df.plot(kind='bar')
plt.title('Bar Plot')
plt.show()

# Create a scatter plot
df.plot(kind='scatter', x='A', y='B')
plt.title('Scatter Plot')
plt.show()
```

Slide 13: Real-Life Example: Weather Data Analysis

Let's analyze a weather dataset to demonstrate Pandas' capabilities in a real-world scenario.

```python
# Load weather data
weather_data = pd.read_csv('weather_data.csv')

# Display basic information about the dataset
print(weather_data.info())

# Calculate average temperature by month
weather_data['Date'] = pd.to_datetime(weather_data['Date'])
monthly_temp = weather_data.groupby(weather_data['Date'].dt.to_period('M'))['Temperature'].mean()

# Find the hottest and coldest days
hottest_day = weather_data.loc[weather_data['Temperature'].idxmax()]
coldest_day = weather_data.loc[weather_data['Temperature'].idxmin()]

print("\nAverage Monthly Temperature:")
print(monthly_temp)
print("\nHottest Day:")
print(hottest_day)
print("\nColdest Day:")
print(coldest_day)
```

Slide 14: Real-Life Example: Product Inventory Management

This example demonstrates how Pandas can be used to manage and analyze product inventory data.

```python
# Create a sample inventory DataFrame
inventory = pd.DataFrame({
    'Product': ['Widget A', 'Widget B', 'Widget C', 'Widget D'],
    'Quantity': [100, 150, 200, 75],
    'Price': [10.99, 15.99, 8.99, 12.99],
    'Category': ['Electronics', 'Tools', 'Electronics', 'Tools']
})

# Calculate total value of inventory
inventory['Total Value'] = inventory['Quantity'] * inventory['Price']

# Find products with low stock (less than 100)
low_stock = inventory[inventory['Quantity'] < 100]

# Calculate average price by category
avg_price_by_category = inventory.groupby('Category')['Price'].mean()

print("Inventory Summary:")
print(inventory)
print("\nLow Stock Products:")
print(low_stock)
print("\nAverage Price by Category:")
print(avg_price_by_category)
```

Slide 15: Additional Resources

To further expand your knowledge of Pandas and data analysis with Python, consider exploring these resources:

1.  Official Pandas Documentation: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
2.  "Python for Data Analysis" by Wes McKinney (creator of Pandas)
3.  DataCamp's Pandas Tutorials: [https://www.datacamp.com/courses/data-manipulation-with-pandas](https://www.datacamp.com/courses/data-manipulation-with-pandas)
4.  "Effective Pandas" by Matt Harrison (available on GitHub)
5.  Kaggle's Pandas Microcourse: [https://www.kaggle.com/learn/pandas](https://www.kaggle.com/learn/pandas)

Remember to practice regularly with real-world datasets to strengthen your Pandas skills.

