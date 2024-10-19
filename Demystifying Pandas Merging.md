## Demystifying Pandas Merging
Slide 1: Understanding Merging in Pandas

Merging in Pandas is often misunderstood, but it's a powerful tool for combining datasets. It allows you to join DataFrames based on common columns or indices, similar to SQL joins. Let's explore the different types of merges and their applications.

Slide 2: Source Code for Understanding Merging in Pandas

```python
import pandas as pd

# Create sample DataFrames
df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value1': [1, 2, 3, 4]})
df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'], 'value2': [5, 6, 7, 8]})

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)
```

Slide 3: Results for: Understanding Merging in Pandas

```
DataFrame 1:
  key  value1
0   A       1
1   B       2
2   C       3
3   D       4

DataFrame 2:
  key  value2
0   B       5
1   D       6
2   E       7
3   F       8
```

Slide 4: Inner Merge

Inner merge is the default merge operation in Pandas. It returns only the rows with matching keys in both DataFrames. This is useful when you want to combine data only for entries that exist in both datasets.

Slide 5: Source Code for Inner Merge

```python
# Perform inner merge
inner_merge = pd.merge(df1, df2, on='key')

print("Inner Merge Result:")
print(inner_merge)
```

Slide 6: Results for: Inner Merge

```
Inner Merge Result:
  key  value1  value2
0   B       2       5
1   D       4       6
```

Slide 7: Outer Merge

Outer merge returns all rows from both DataFrames, filling in NaN for missing values. This is useful when you want to preserve all data from both datasets, even if there's no match.

Slide 8: Source Code for Outer Merge

```python
# Perform outer merge
outer_merge = pd.merge(df1, df2, on='key', how='outer')

print("Outer Merge Result:")
print(outer_merge)
```

Slide 9: Results for: Outer Merge

```
Outer Merge Result:
  key  value1  value2
0   A     1.0    NaN
1   B     2.0    5.0
2   C     3.0    NaN
3   D     4.0    6.0
4   E     NaN    7.0
5   F     NaN    8.0
```

Slide 10: Left and Right Merge

Left merge keeps all rows from the left DataFrame and only matching rows from the right DataFrame. Right merge does the opposite. These are useful when you want to prioritize one dataset over the other.

Slide 11: Source Code for Left and Right Merge

```python
# Perform left merge
left_merge = pd.merge(df1, df2, on='key', how='left')

# Perform right merge
right_merge = pd.merge(df1, df2, on='key', how='right')

print("Left Merge Result:")
print(left_merge)
print("\nRight Merge Result:")
print(right_merge)
```

Slide 12: Results for: Left and Right Merge

```
Left Merge Result:
  key  value1  value2
0   A       1    NaN
1   B       2    5.0
2   C       3    NaN
3   D       4    6.0

Right Merge Result:
  key  value1  value2
0   B     2.0       5
1   D     4.0       6
2   E     NaN       7
3   F     NaN       8
```

Slide 13: Real-Life Example: Merging Customer and Order Data

Imagine you have two datasets: one with customer information and another with order details. You want to combine these to analyze customer purchasing behavior.

Slide 14: Source Code for Real-Life Example: Merging Customer and Order Data

```python
# Create sample customer and order DataFrames
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'email': ['alice@email.com', 'bob@email.com', 'charlie@email.com', 'david@email.com']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105],
    'customer_id': [1, 2, 2, 3, 5],
    'product': ['Widget A', 'Widget B', 'Widget C', 'Widget A', 'Widget D'],
    'quantity': [2, 1, 3, 1, 2]
})

# Merge customer and order data
customer_orders = pd.merge(customers, orders, on='customer_id', how='left')

print("Merged Customer Orders:")
print(customer_orders)
```

Slide 15: Results for: Real-Life Example: Merging Customer and Order Data

```
Merged Customer Orders:
   customer_id    name             email  order_id product  quantity
0            1   Alice   alice@email.com     101.0  Widget A       2.0
1            2     Bob     bob@email.com     102.0  Widget B       1.0
2            2     Bob     bob@email.com     103.0  Widget C       3.0
3            3 Charlie charlie@email.com     104.0  Widget A       1.0
4            4   David   david@email.com       NaN      NaN       NaN
```

Slide 16: Real-Life Example: Merging Weather and Pollution Data

Consider merging weather data with pollution measurements to study the relationship between weather conditions and air quality.

Slide 17: Source Code for Real-Life Example: Merging Weather and Pollution Data

```python
# Create sample weather and pollution DataFrames
weather = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=5),
    'temperature': [20, 22, 19, 21, 23],
    'humidity': [60, 55, 65, 58, 52]
})

pollution = pd.DataFrame({
    'date': pd.date_range(start='2023-01-02', periods=5),
    'pm25': [15, 18, 12, 20, 16],
    'no2': [30, 35, 28, 40, 32]
})

# Merge weather and pollution data
weather_pollution = pd.merge(weather, pollution, on='date', how='outer')

print("Merged Weather and Pollution Data:")
print(weather_pollution)
```

Slide 18: Results for: Real-Life Example: Merging Weather and Pollution Data

```
Merged Weather and Pollution Data:
        date  temperature  humidity  pm25   no2
0 2023-01-01         20.0      60.0   NaN   NaN
1 2023-01-02         22.0      55.0  15.0  30.0
2 2023-01-03         19.0      65.0  18.0  35.0
3 2023-01-04         21.0      58.0  12.0  28.0
4 2023-01-05         23.0      52.0  20.0  40.0
5 2023-01-06          NaN       NaN  16.0  32.0
```

Slide 19: Common Misunderstandings about Merging in Pandas

One common misunderstanding is assuming that merge always combines DataFrames based on index. By default, merge uses common column names. Another misconception is that merge always keeps all rows, which is only true for outer merge.

Slide 20: Source Code for Common Misunderstandings about Merging in Pandas

```python
# Create sample DataFrames with different indices
df1 = pd.DataFrame({'value': [1, 2, 3]}, index=['A', 'B', 'C'])
df2 = pd.DataFrame({'value': [4, 5, 6]}, index=['B', 'C', 'D'])

# Attempt to merge on index (this won't work as expected)
incorrect_merge = pd.merge(df1, df2, left_index=True, right_index=True)

# Correct way to merge on index
correct_merge = pd.merge(df1, df2, left_index=True, right_index=True, how='outer')

print("Incorrect Merge Result:")
print(incorrect_merge)
print("\nCorrect Merge Result:")
print(correct_merge)
```

Slide 21: Results for: Common Misunderstandings about Merging in Pandas

```
Incorrect Merge Result:
   value_x  value_y
B       2       4
C       3       5

Correct Merge Result:
   value_x  value_y
A     1.0     NaN
B     2.0     4.0
C     3.0     5.0
D     NaN     6.0
```

Slide 22: Additional Resources

For more information on merging in Pandas, refer to the following resources:

1.  Pandas official documentation on merging: [https://pandas.pydata.org/pandas-docs/stable/user\_guide/merging.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html)
2.  "Efficient DataFrame Manipulations in Pandas" by Wes McKinney (creator of Pandas): arXiv:2101.00140

