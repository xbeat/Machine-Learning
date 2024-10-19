## Pandas Library! Key Methods for Data Science

Slide 1: Introduction to Pandas

Pandas is a powerful Python library for data manipulation and analysis. It provides data structures and functions to efficiently handle structured data, making it an essential tool for data scientists and analysts.

```python

# Create a simple DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'San Francisco', 'London']}
df = pd.DataFrame(data)
print(df)
```

Slide 2: Data Structures: Series and DataFrame

Pandas offers two main data structures: Series (1-dimensional) and DataFrame (2-dimensional). A Series is similar to a column in a spreadsheet, while a DataFrame is like a table with rows and columns.

```python
s = pd.Series([1, 3, 5, 7, 9], name='Odd Numbers')
print("Series:\n", s)

# Create a DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
print("\nDataFrame:\n", df)
```

Slide 3: Data Creation: Reading from Files

Pandas can read data from various file formats, including CSV and Excel. This feature allows easy import of external data into your Python environment.

```python
df_csv = pd.read_csv('example.csv')
print("CSV Data:\n", df_csv.head())

# Reading an Excel file
df_excel = pd.read_excel('example.xlsx')
print("\nExcel Data:\n", df_excel.head())
```

Slide 4: Data Inspection

Inspecting your data is crucial for understanding its structure and content. Pandas provides methods like head(), info(), and describe() for this purpose.

```python

# Create a sample DataFrame
df = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])

# Display the first few rows
print(df.head())

# Get information about the DataFrame
print(df.info())

# Get statistical summary
print(df.describe())
```

Slide 5: Data Manipulation: Adding and Removing Columns

Pandas makes it easy to add new columns or remove existing ones from a DataFrame. This is essential for feature engineering and data preprocessing.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Add a new column
df['C'] = df['A'] + df['B']
print("After adding column C:\n", df)

# Remove a column
df = df.drop('B', axis=1)
print("\nAfter removing column B:\n", df)
```

Slide 6: Time Series Methods

Pandas excels at handling time series data, providing methods to work with dates and times efficiently.

```python
dates = pd.date_range('20230101', periods=6)
ts = pd.Series(np.random.randn(6), index=dates)
print("Time Series:\n", ts)

# Resample to monthly frequency
monthly = ts.resample('M').mean()
print("\nMonthly Resampled:\n", monthly)
```

Slide 7: Grouping and Aggregation

Grouping and aggregating data are common operations in data analysis. Pandas provides powerful methods for these tasks.

```python
df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Value': [10, 20, 30, 40, 50, 60]
})

# Group by Category and calculate mean
grouped = df.groupby('Category').mean()
print("Grouped by Category:\n", grouped)

# Multiple aggregations
agg_functions = {'Value': ['sum', 'mean', 'max']}
multi_agg = df.groupby('Category').agg(agg_functions)
print("\nMultiple Aggregations:\n", multi_agg)
```

Slide 8: Visualization with Pandas

Pandas integrates well with Matplotlib, allowing for quick and easy data visualization directly from DataFrames.

```python

# Create a sample DataFrame
df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))

# Create a line plot
df.plot(figsize=(10, 6))
plt.title('Line Plot of Random Data')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()

# Create a histogram
df['A'].hist(bins=20)
plt.title('Histogram of Column A')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 9: Input and Output Methods

Pandas provides various methods to read from and write to different file formats, facilitating data exchange between different systems.

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Write to CSV
df.to_csv('output.csv', index=False)
print("Data written to CSV")

# Write to Excel
df.to_excel('output.xlsx', index=False)
print("Data written to Excel")

# Read back the CSV file
df_csv = pd.read_csv('output.csv')
print("\nData read from CSV:\n", df_csv)
```

Slide 10: Data Cleaning: Handling Missing Values

Data cleaning is a crucial step in data analysis. Pandas offers methods to handle missing values effectively.

```python
df = pd.DataFrame({'A': [1, 2, np.nan, 4],
                   'B': [5, np.nan, np.nan, 8],
                   'C': [9, 10, 11, 12]})

print("Original DataFrame:\n", df)

# Fill missing values with a specific value
df_filled = df.fillna(0)
print("\nFilled with 0:\n", df_filled)

# Fill missing values with the mean of the column
df_mean_filled = df.fillna(df.mean())
print("\nFilled with column mean:\n", df_mean_filled)

# Drop rows with any missing values
df_dropped = df.dropna()
print("\nRows with missing values dropped:\n", df_dropped)
```

Slide 11: Descriptive Statistics

Pandas provides built-in methods for calculating descriptive statistics, helping you understand the central tendencies and variability in your data.

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
})

# Calculate basic statistics
print("Mean:\n", df.mean())
print("\nMedian:\n", df.median())
print("\nStandard Deviation:\n", df.std())

# Get a comprehensive statistical summary
print("\nDescriptive Statistics:\n", df.describe())

# Calculate correlation between columns
print("\nCorrelation Matrix:\n", df.corr())
```

Slide 12: Data Selection and Indexing

Pandas offers powerful methods for selecting and indexing data, allowing you to extract specific subsets of your data efficiently.

```python
df = pd.DataFrame({
    'A': range(1, 6),
    'B': range(10, 60, 10),
    'C': ['P', 'Q', 'R', 'S', 'T']
})

print("Original DataFrame:\n", df)

# Select a single column
print("\nColumn A:\n", df['A'])

# Select multiple columns
print("\nColumns A and C:\n", df[['A', 'C']])

# Select rows by index
print("\nRows 1 to 3:\n", df.loc[1:3])

# Select rows by condition
print("\nRows where A > 2:\n", df[df['A'] > 2])

# Select specific cells
print("\nValue at row 2, column 'B':", df.at[2, 'B'])
```

Slide 13: Real-life Example: Weather Data Analysis

Let's analyze a dataset containing daily weather information for a city over a year.

```python
import matplotlib.pyplot as plt

# Create a sample weather dataset
dates = pd.date_range('20230101', periods=365)
weather_data = pd.DataFrame({
    'Date': dates,
    'Temperature': np.random.normal(15, 5, 365),
    'Humidity': np.random.normal(60, 10, 365),
    'Rainfall': np.random.exponential(1, 365)
})

# Set Date as index
weather_data.set_index('Date', inplace=True)

# Calculate monthly average temperature
monthly_temp = weather_data['Temperature'].resample('M').mean()

# Plot monthly average temperature
plt.figure(figsize=(12, 6))
monthly_temp.plot(kind='bar')
plt.title('Monthly Average Temperature')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Find the day with maximum rainfall
max_rainfall_day = weather_data['Rainfall'].idxmax()
print(f"Day with maximum rainfall: {max_rainfall_day}")
print(f"Rainfall amount: {weather_data.loc[max_rainfall_day, 'Rainfall']:.2f} mm")
```

Slide 14: Real-life Example: Product Inventory Analysis

Let's analyze an inventory dataset for a small retail store.

```python
import matplotlib.pyplot as plt

# Create a sample inventory dataset
inventory_data = pd.DataFrame({
    'Product': ['A', 'B', 'C', 'D', 'E'] * 12,
    'Date': pd.date_range('20230101', periods=60, freq='D').repeat(5),
    'Quantity': np.random.randint(10, 100, 60)
})

# Set multi-index
inventory_data.set_index(['Date', 'Product'], inplace=True)

# Calculate total inventory by product
total_inventory = inventory_data.groupby('Product')['Quantity'].sum()

# Plot total inventory by product
plt.figure(figsize=(10, 6))
total_inventory.plot(kind='bar')
plt.title('Total Inventory by Product')
plt.xlabel('Product')
plt.ylabel('Total Quantity')
plt.tight_layout()
plt.show()

# Find the product with the highest average daily inventory
avg_daily_inventory = inventory_data.groupby('Product')['Quantity'].mean()
highest_avg_product = avg_daily_inventory.idxmax()
print(f"Product with highest average daily inventory: {highest_avg_product}")
print(f"Average daily inventory: {avg_daily_inventory[highest_avg_product]:.2f}")
```

Slide 15: Additional Resources

For more advanced topics and in-depth understanding of Pandas, consider exploring these resources:

1. Official Pandas Documentation: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
2. "Python for Data Analysis" by Wes McKinney (creator of Pandas)
3. Pandas Cheat Sheet: [https://pandas.pydata.org/Pandas\_Cheat\_Sheet.pdf](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
4. DataCamp's Pandas Tutorials: [https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python](https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python)

Remember to always refer to the most up-to-date documentation as Pandas is continuously evolving.


