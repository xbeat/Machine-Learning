## Data Wrangling with Pandas in Python
Slide 1: Introduction to Data Wrangling with Pandas

Data wrangling is the process of cleaning, structuring, and enriching raw data into a desired format for better decision making in less time. Pandas is a powerful Python library that provides high-performance, easy-to-use data structures and data analysis tools for handling structured data.

```python
import pandas as pd
import numpy as np

# Create a sample dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']
}

# Create a DataFrame
df = pd.DataFrame(data)

print(df)
```

Slide 2: Loading Data with Pandas

Pandas can read data from various file formats, including CSV, Excel, JSON, and SQL databases. Let's explore how to load data from a CSV file.

```python
# Load data from a CSV file
df = pd.read_csv('sample_data.csv')

# Display the first few rows
print(df.head())

# Get basic information about the dataset
print(df.info())
```

Slide 3: Data Exploration

After loading the data, it's essential to explore its structure and contents. Pandas provides several methods to quickly understand your dataset.

```python
# Display basic statistics of numerical columns
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Display unique values in a column
print(df['Category'].unique())

# Get the shape of the DataFrame
print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
```

Slide 4: Data Cleaning - Handling Missing Values

Missing values can significantly impact your analysis. Pandas offers various methods to handle them effectively.

```python
# Fill missing values with a specific value
df['Column_A'].fillna(0, inplace=True)

# Fill missing values with the mean of the column
df['Column_B'].fillna(df['Column_B'].mean(), inplace=True)

# Drop rows with any missing values
df_cleaned = df.dropna()

# Drop columns with more than 50% missing values
df_cleaned = df.dropna(thresh=len(df) * 0.5, axis=1)

print(df_cleaned.isnull().sum())
```

Slide 5: Data Transformation - Renaming and Reordering Columns

Organizing your data can improve readability and analysis efficiency. Let's explore how to rename and reorder columns.

```python
# Rename columns
df = df.rename(columns={'old_name1': 'new_name1', 'old_name2': 'new_name2'})

# Reorder columns
desired_order = ['column3', 'column1', 'column2']
df = df[desired_order]

print(df.head())
```

Slide 6: Data Filtering and Selection

Pandas provides powerful ways to filter and select data based on various conditions.

```python
# Select specific columns
selected_columns = df[['Name', 'Age', 'City']]

# Filter rows based on a condition
adults = df[df['Age'] >= 18]

# Filter using multiple conditions
target_group = df[(df['Age'] >= 25) & (df['City'] == 'New York')]

# Select data using .loc and .iloc
specific_data = df.loc[df['Name'] == 'Alice', 'Age']
first_two_rows = df.iloc[:2, :3]

print(target_group)
print(specific_data)
print(first_two_rows)
```

Slide 7: Grouping and Aggregation

Grouping data allows you to perform operations on subsets of your dataset. Let's explore grouping and aggregation techniques.

```python
# Group by a column and calculate mean
mean_age_by_city = df.groupby('City')['Age'].mean()

# Multiple aggregations
agg_results = df.groupby('City').agg({
    'Age': ['mean', 'max', 'min'],
    'Salary': ['mean', 'median']
})

# Reset index to make grouped column a regular column
agg_results = agg_results.reset_index()

print(mean_age_by_city)
print(agg_results)
```

Slide 8: Merging and Joining DataFrames

Combining data from multiple sources is a common task in data wrangling. Pandas provides various methods to merge and join DataFrames.

```python
# Create two sample DataFrames
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [2, 3, 4], 'Age': [25, 30, 35]})

# Inner join
inner_join = pd.merge(df1, df2, on='ID', how='inner')

# Left join
left_join = pd.merge(df1, df2, on='ID', how='left')

# Concatenate DataFrames vertically
df3 = pd.DataFrame({'ID': [5, 6], 'Name': ['David', 'Eve']})
concatenated = pd.concat([df1, df3], ignore_index=True)

print("Inner Join:")
print(inner_join)
print("\nLeft Join:")
print(left_join)
print("\nConcatenated:")
print(concatenated)
```

Slide 9: Handling Duplicate Data

Duplicate data can skew your analysis. Let's explore how to identify and remove duplicates.

```python
# Create a sample DataFrame with duplicates
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David'],
    'Age': [25, 30, 35, 25, 40]
}
df = pd.DataFrame(data)

# Identify duplicate rows
print("Duplicate rows:")
print(df[df.duplicated()])

# Remove duplicate rows
df_unique = df.drop_duplicates()

# Remove duplicates based on specific columns
df_unique_name = df.drop_duplicates(subset=['Name'])

print("\nDataFrame after removing all duplicates:")
print(df_unique)
print("\nDataFrame after removing duplicates based on 'Name':")
print(df_unique_name)
```

Slide 10: Data Type Conversion

Ensuring correct data types is crucial for accurate analysis and efficient memory usage.

```python
# Create a sample DataFrame
df = pd.DataFrame({
    'ID': ['1', '2', '3'],
    'Value': ['10.5', '20.0', '30.7'],
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03']
})

print("Original DataFrame:")
print(df.dtypes)

# Convert 'ID' to integer
df['ID'] = df['ID'].astype(int)

# Convert 'Value' to float
df['Value'] = df['Value'].astype(float)

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

print("\nDataFrame after type conversion:")
print(df.dtypes)
print(df)
```

Slide 11: Handling Categorical Data

Categorical data requires special treatment. Pandas provides tools to encode and manipulate categorical variables.

```python
# Create a sample DataFrame with categorical data
df = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red']
})

# Convert 'Color' to categorical type
df['Color'] = df['Color'].astype('category')

# Get category codes
df['Color_Code'] = df['Color'].cat.codes

# One-hot encoding
color_dummies = pd.get_dummies(df['Color'], prefix='Color')

# Combine with original DataFrame
df_encoded = pd.concat([df, color_dummies], axis=1)

print("DataFrame with categorical data:")
print(df)
print("\nDataFrame with one-hot encoding:")
print(df_encoded)
```

Slide 12: Real-life Example: Analyzing Product Reviews

Let's analyze a dataset of product reviews to gain insights into customer satisfaction.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (assuming we have a CSV file with product reviews)
df = pd.read_csv('product_reviews.csv')

# Clean the data
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df = df.dropna(subset=['Rating', 'Review'])

# Calculate average rating per product
avg_ratings = df.groupby('ProductID')['Rating'].mean().sort_values(ascending=False)

# Get the top 10 products by rating
top_10_products = avg_ratings.head(10)

# Plot the results
plt.figure(figsize=(12, 6))
top_10_products.plot(kind='bar')
plt.title('Top 10 Products by Average Rating')
plt.xlabel('Product ID')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Top 10 Products by Average Rating:")
print(top_10_products)
```

Slide 13: Real-life Example: Analyzing Weather Data

Let's analyze historical weather data to identify trends and patterns.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the weather dataset (assuming we have a CSV file with daily weather data)
df = pd.read_csv('weather_data.csv')

# Convert date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set date as index
df.set_index('Date', inplace=True)

# Resample data to monthly average temperature
monthly_temp = df['Temperature'].resample('M').mean()

# Plot the monthly average temperature
plt.figure(figsize=(12, 6))
monthly_temp.plot()
plt.title('Monthly Average Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate year-over-year temperature change
yearly_temp = df['Temperature'].resample('Y').mean()
temp_change = yearly_temp.pct_change() * 100

print("Year-over-Year Temperature Change (%):")
print(temp_change)
```

Slide 14: Additional Resources

For further learning and exploration of data wrangling with Pandas, consider the following resources:

1. Pandas Official Documentation: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
2. "Python for Data Analysis" by Wes McKinney (creator of Pandas)
3. DataCamp's Pandas Tutorials: [https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python](https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python)
4. Real Python's Pandas Tutorials: [https://realpython.com/learning-paths/pandas-data-science/](https://realpython.com/learning-paths/pandas-data-science/)
5. ArXiv paper: "Pandas: Powerful Python Data Analysis Toolkit" by Wes McKinney (2018) - [https://arxiv.org/abs/1801.01323](https://arxiv.org/abs/1801.01323)

These resources provide in-depth explanations, examples, and best practices for using Pandas effectively in your data wrangling projects.

