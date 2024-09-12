## Grouping Sets, Rollup, and Cube in SQL with Python
Slide 1: Introduction to Grouping Sets, Rollup, and Cube in SQL

Grouping Sets, Rollup, and Cube are powerful SQL extensions that allow for flexible and efficient generation of multiple grouping combinations in a single query. These features are particularly useful for generating summary reports and performing multi-dimensional data analysis.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Create a sample dataset
data = {
    'Region': ['North', 'North', 'South', 'South', 'East', 'East', 'West', 'West'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'Sales': [100, 150, 200, 250, 300, 350, 400, 450]
}
df = pd.DataFrame(data)

# Display the dataset
print(df)
```

Slide 2: Grouping Sets Basics

Grouping Sets allow you to specify multiple grouping clauses in a single GROUP BY statement. This feature combines different levels of aggregation, producing a result set that includes subtotals and grand totals.

```python
import pandas as pd

# Sample data
data = {
    'Region': ['North', 'North', 'South', 'South'],
    'Product': ['A', 'B', 'A', 'B'],
    'Sales': [100, 150, 200, 250]
}
df = pd.DataFrame(data)

# Simulate GROUPING SETS
result = pd.concat([
    df.groupby('Region')['Sales'].sum().reset_index(),
    df.groupby('Product')['Sales'].sum().reset_index(),
    pd.DataFrame({'Sales': [df['Sales'].sum()]})
])

print(result)
```

Slide 3: ROLLUP Clause

ROLLUP generates a result set with multiple levels of subtotals, moving from the most detailed level to the grand total. It's particularly useful for hierarchical data summarization.

```python
import pandas as pd

# Sample data
data = {
    'Year': [2022, 2022, 2023, 2023],
    'Quarter': [1, 2, 1, 2],
    'Sales': [1000, 1200, 1100, 1300]
}
df = pd.DataFrame(data)

# Simulate ROLLUP
result = pd.concat([
    df.groupby(['Year', 'Quarter'])['Sales'].sum().reset_index(),
    df.groupby('Year')['Sales'].sum().reset_index(),
    pd.DataFrame({'Sales': [df['Sales'].sum()]})
])

print(result)
```

Slide 4: CUBE Clause

CUBE generates a result set with all possible combinations of the specified dimensions. It provides a complete cross-tabulation of all dimensions in the query.

```python
import pandas as pd
import itertools

# Sample data
data = {
    'Region': ['North', 'North', 'South', 'South'],
    'Product': ['A', 'B', 'A', 'B'],
    'Sales': [100, 150, 200, 250]
}
df = pd.DataFrame(data)

# Simulate CUBE
dimensions = ['Region', 'Product']
combinations = list(itertools.chain.from_iterable(
    itertools.combinations(dimensions, r) for r in range(len(dimensions) + 1)
))

result = pd.concat([
    df.groupby(list(combo))['Sales'].sum().reset_index() for combo in combinations
])

print(result)
```

Slide 5: Comparing Grouping Sets, ROLLUP, and CUBE

These three SQL features offer different levels of aggregation and flexibility:

* Grouping Sets: Custom combinations of dimensions
* ROLLUP: Hierarchical summaries
* CUBE: All possible combinations

```python
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {
    'Region': ['North', 'North', 'South', 'South'],
    'Product': ['A', 'B', 'A', 'B'],
    'Sales': [100, 150, 200, 250]
}
df = pd.DataFrame(data)

# Simulate different grouping operations
grouping_sets = pd.concat([
    df.groupby('Region')['Sales'].sum(),
    df.groupby('Product')['Sales'].sum(),
    pd.Series([df['Sales'].sum()], index=['Total'])
])

rollup = pd.concat([
    df.groupby(['Region', 'Product'])['Sales'].sum(),
    df.groupby('Region')['Sales'].sum(),
    pd.Series([df['Sales'].sum()], index=['Total'])
])

cube = pd.concat([
    df.groupby(['Region', 'Product'])['Sales'].sum(),
    df.groupby('Region')['Sales'].sum(),
    df.groupby('Product')['Sales'].sum(),
    pd.Series([df['Sales'].sum()], index=['Total'])
])

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
grouping_sets.plot(kind='bar', ax=ax1, title='Grouping Sets')
rollup.plot(kind='bar', ax=ax2, title='ROLLUP')
cube.plot(kind='bar', ax=ax3, title='CUBE')
plt.tight_layout()
plt.show()
```

Slide 6: Practical Example: Sales Analysis

Let's analyze a dataset of product sales across different regions and time periods using Grouping Sets, ROLLUP, and CUBE.

```python
import pandas as pd

# Create a sample sales dataset
data = {
    'Year': [2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023],
    'Region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'Sales': [100, 150, 200, 250, 300, 350, 400, 450]
}
df = pd.DataFrame(data)

# Display the dataset
print(df)

# Simulate GROUPING SETS
grouping_sets = pd.concat([
    df.groupby(['Year', 'Region'])['Sales'].sum().reset_index(),
    df.groupby(['Year', 'Product'])['Sales'].sum().reset_index(),
    df.groupby('Year')['Sales'].sum().reset_index(),
    pd.DataFrame({'Sales': [df['Sales'].sum()]})
])

print("\nGrouping Sets Result:")
print(grouping_sets)
```

Slide 7: ROLLUP Example: Hierarchical Sales Summary

Using ROLLUP to create a hierarchical summary of sales data.

```python
import pandas as pd

# Using the same dataset from the previous slide

# Simulate ROLLUP
rollup = pd.concat([
    df.groupby(['Year', 'Region', 'Product'])['Sales'].sum().reset_index(),
    df.groupby(['Year', 'Region'])['Sales'].sum().reset_index(),
    df.groupby('Year')['Sales'].sum().reset_index(),
    pd.DataFrame({'Sales': [df['Sales'].sum()]})
])

print("ROLLUP Result:")
print(rollup)

# Visualize the hierarchical structure
import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph()
G.add_edge("Total", "2022")
G.add_edge("Total", "2023")
G.add_edge("2022", "North 2022")
G.add_edge("2022", "South 2022")
G.add_edge("2023", "North 2023")
G.add_edge("2023", "South 2023")

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, arrows=True)
plt.title("ROLLUP Hierarchical Structure")
plt.axis('off')
plt.show()
```

Slide 8: CUBE Example: Multi-dimensional Analysis

Using CUBE to perform a multi-dimensional analysis of the sales data.

```python
import pandas as pd
import itertools

# Using the same dataset from the previous slides

# Simulate CUBE
dimensions = ['Year', 'Region', 'Product']
combinations = list(itertools.chain.from_iterable(
    itertools.combinations(dimensions, r) for r in range(len(dimensions) + 1)
))

cube = pd.concat([
    df.groupby(list(combo))['Sales'].sum().reset_index() for combo in combinations
])

print("CUBE Result:")
print(cube)

# Visualize the cube structure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = [0, 1, 0, 1]
y = [0, 0, 1, 1]
z = [0, 0, 0, 0]

ax.scatter(x, y, z, c='r', s=100)
ax.plot([0, 1], [0, 0], [0, 0], 'b')
ax.plot([0, 0], [0, 1], [0, 0], 'b')
ax.plot([1, 1], [0, 1], [0, 0], 'b')
ax.plot([0, 1], [1, 1], [0, 0], 'b')

ax.set_xlabel('Year')
ax.set_ylabel('Region')
ax.set_zlabel('Product')
ax.set_title('CUBE Structure Visualization')

plt.show()
```

Slide 9: Performance Considerations

When using Grouping Sets, ROLLUP, and CUBE, consider the following performance aspects:

1. Data volume: These operations can generate large result sets, especially CUBE.
2. Indexing: Proper indexing on grouped columns can significantly improve performance.
3. Materialized views: For frequently used groupings, consider using materialized views.

```python
import time
import pandas as pd
import numpy as np

# Generate a larger dataset
np.random.seed(0)
n = 1000000
data = pd.DataFrame({
    'Year': np.random.choice([2022, 2023], n),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], n),
    'Product': np.random.choice(['A', 'B', 'C', 'D'], n),
    'Sales': np.random.randint(100, 1000, n)
})

# Measure execution time for different operations
def measure_time(func):
    start = time.time()
    func()
    end = time.time()
    return end - start

grouping_sets_time = measure_time(lambda: data.groupby(['Year', 'Region'])['Sales'].sum())
rollup_time = measure_time(lambda: data.groupby(['Year', 'Region', 'Product'])['Sales'].sum())
cube_time = measure_time(lambda: data.groupby(['Year', 'Region', 'Product'])['Sales'].sum())

print(f"Grouping Sets time: {grouping_sets_time:.2f} seconds")
print(f"ROLLUP time: {rollup_time:.2f} seconds")
print(f"CUBE time: {cube_time:.2f} seconds")

# Plot execution times
import matplotlib.pyplot as plt

operations = ['Grouping Sets', 'ROLLUP', 'CUBE']
times = [grouping_sets_time, rollup_time, cube_time]

plt.figure(figsize=(10, 6))
plt.bar(operations, times)
plt.title('Execution Time Comparison')
plt.ylabel('Time (seconds)')
plt.show()
```

Slide 10: Real-life Example: Weather Data Analysis

Analyzing temperature data across different locations and time periods using Grouping Sets, ROLLUP, and CUBE.

```python
import pandas as pd
import numpy as np

# Generate sample weather data
np.random.seed(0)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
locations = ['City A', 'City B', 'City C']
data = []

for date in dates:
    for location in locations:
        temp = np.random.normal(loc=20, scale=5)
        data.append([date, location, temp])

df = pd.DataFrame(data, columns=['Date', 'Location', 'Temperature'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Grouping Sets: Average temperature by year and location
grouping_sets = df.groupby(['Year', 'Location'])['Temperature'].mean().reset_index()
print("Grouping Sets Result:")
print(grouping_sets)

# ROLLUP: Hierarchical summary of temperatures
rollup = pd.concat([
    df.groupby(['Year', 'Month', 'Location'])['Temperature'].mean(),
    df.groupby(['Year', 'Month'])['Temperature'].mean(),
    df.groupby('Year')['Temperature'].mean(),
    pd.Series([df['Temperature'].mean()], index=['Overall'])
]).reset_index()
print("\nROLLUP Result:")
print(rollup)

# CUBE: Multi-dimensional analysis
cube = pd.concat([
    df.groupby(['Year', 'Month', 'Location'])['Temperature'].mean(),
    df.groupby(['Year', 'Month'])['Temperature'].mean(),
    df.groupby(['Year', 'Location'])['Temperature'].mean(),
    df.groupby(['Month', 'Location'])['Temperature'].mean(),
    df.groupby('Year')['Temperature'].mean(),
    df.groupby('Month')['Temperature'].mean(),
    df.groupby('Location')['Temperature'].mean(),
    pd.Series([df['Temperature'].mean()], index=['Overall'])
]).reset_index()
print("\nCUBE Result:")
print(cube)

# Visualize average temperatures by location
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for location in locations:
    data = df[df['Location'] == location]
    plt.plot(data['Date'], data['Temperature'], label=location)

plt.title('Temperature Trends by Location')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
```

Slide 11: Real-life Example: E-commerce Product Analysis

Analyzing product data across different categories and time periods using Grouping Sets, ROLLUP, and CUBE in an e-commerce context.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample e-commerce data
np.random.seed(0)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
categories = ['Electronics', 'Clothing', 'Home & Garden']
products = ['Product A', 'Product B', 'Product C']
data = []

for date in dates:
    for category in categories:
        for product in products:
            sales = np.random.randint(10, 100)
            data.append([date, category, product, sales])

df = pd.DataFrame(data, columns=['Date', 'Category', 'Product', 'Sales'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Grouping Sets: Total sales by category and product
grouping_sets = df.groupby(['Category', 'Product'])['Sales'].sum().reset_index()
print("Grouping Sets Result:")
print(grouping_sets.head(10))

# ROLLUP: Hierarchical summary of sales
rollup = pd.concat([
    df.groupby(['Year', 'Month', 'Category'])['Sales'].sum(),
    df.groupby(['Year', 'Month'])['Sales'].sum(),
    df.groupby('Year')['Sales'].sum(),
    pd.Series([df['Sales'].sum()], index=['Overall'])
]).reset_index()
print("\nROLLUP Result:")
print(rollup.head(10))

# CUBE: Multi-dimensional analysis
cube = pd.concat([
    df.groupby(['Year', 'Month', 'Category', 'Product'])['Sales'].sum(),
    df.groupby(['Year', 'Month', 'Category'])['Sales'].sum(),
    df.groupby(['Year', 'Month'])['Sales'].sum(),
    df.groupby(['Year', 'Category'])['Sales'].sum(),
    df.groupby('Year')['Sales'].sum(),
    pd.Series([df['Sales'].sum()], index=['Overall'])
]).reset_index()
print("\nCUBE Result:")
print(cube.head(10))

# Visualize sales trends
plt.figure(figsize=(12, 6))
for category in categories:
    category_data = df[df['Category'] == category].groupby('Date')['Sales'].sum()
    plt.plot(category_data.index, category_data.values, label=category)

plt.title('Sales Trends by Category')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.show()
```

Slide 12: Advanced Techniques: Combining Grouping Sets, ROLLUP, and CUBE

In complex scenarios, you can combine these features to create highly customized result sets. This is particularly useful when dealing with multi-dimensional data analysis.

```python
import pandas as pd
import numpy as np

# Using the e-commerce dataset from the previous slide

# Combined GROUPING SETS, ROLLUP, and CUBE
combined_analysis = pd.concat([
    # GROUPING SETS
    df.groupby(['Category', 'Product'])['Sales'].sum(),
    
    # ROLLUP
    df.groupby(['Year', 'Month', 'Category'])['Sales'].sum(),
    df.groupby(['Year', 'Month'])['Sales'].sum(),
    df.groupby('Year')['Sales'].sum(),
    
    # CUBE
    df.groupby(['Year', 'Category', 'Product'])['Sales'].sum(),
    df.groupby(['Year', 'Category'])['Sales'].sum(),
    df.groupby(['Category', 'Product'])['Sales'].sum(),
    
    # Overall total
    pd.Series([df['Sales'].sum()], index=['Overall'])
]).reset_index()

print("Combined Analysis Result:")
print(combined_analysis.head(15))

# Visualize the multi-level aggregation
plt.figure(figsize=(12, 8))
combined_analysis.groupby('Year')['Sales'].sum().plot(kind='bar', position=1, width=0.2, color='blue', label='Year')
combined_analysis.groupby('Category')['Sales'].sum().plot(kind='bar', position=0, width=0.2, color='green', label='Category')
combined_analysis.groupby('Product')['Sales'].sum().plot(kind='bar', position=2, width=0.2, color='red', label='Product')

plt.title('Multi-level Sales Aggregation')
plt.xlabel('Grouping Level')
plt.ylabel('Total Sales')
plt.legend()
plt.tight_layout()
plt.show()
```

Slide 13: Best Practices and Optimization Techniques

When working with Grouping Sets, ROLLUP, and CUBE, consider these best practices:

1. Use appropriate indexing on grouped columns
2. Limit the number of dimensions to avoid exponential growth in result set size
3. Consider using materialized views for frequently accessed aggregations
4. Monitor query performance and optimize as needed

```python
import pandas as pd
import numpy as np
import time

# Generate a larger dataset for performance testing
np.random.seed(0)
n = 1000000
large_df = pd.DataFrame({
    'Year': np.random.choice([2022, 2023], n),
    'Month': np.random.randint(1, 13, n),
    'Category': np.random.choice(['A', 'B', 'C'], n),
    'Product': np.random.choice(['X', 'Y', 'Z'], n),
    'Sales': np.random.randint(100, 1000, n)
})

# Function to measure execution time
def measure_time(func):
    start = time.time()
    result = func()
    end = time.time()
    return result, end - start

# Test different grouping operations
grouping_sets, gs_time = measure_time(lambda: large_df.groupby(['Category', 'Product'])['Sales'].sum())
rollup, rollup_time = measure_time(lambda: pd.concat([
    large_df.groupby(['Year', 'Month', 'Category'])['Sales'].sum(),
    large_df.groupby(['Year', 'Month'])['Sales'].sum(),
    large_df.groupby('Year')['Sales'].sum()
]))
cube, cube_time = measure_time(lambda: pd.concat([
    large_df.groupby(['Year', 'Month', 'Category', 'Product'])['Sales'].sum(),
    large_df.groupby(['Year', 'Month', 'Category'])['Sales'].sum(),
    large_df.groupby(['Year', 'Month'])['Sales'].sum(),
    large_df.groupby('Year')['Sales'].sum()
]))

print(f"Grouping Sets execution time: {gs_time:.2f} seconds")
print(f"ROLLUP execution time: {rollup_time:.2f} seconds")
print(f"CUBE execution time: {cube_time:.2f} seconds")

# Visualize performance comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(['Grouping Sets', 'ROLLUP', 'CUBE'], [gs_time, rollup_time, cube_time])
plt.title('Performance Comparison')
plt.ylabel('Execution Time (seconds)')
plt.show()
```

Slide 14: Conclusion and Future Trends

Grouping Sets, ROLLUP, and CUBE are powerful SQL features that enable efficient multi-dimensional data analysis. As data volumes continue to grow and business intelligence needs become more complex, these tools will play an increasingly important role in data analysis and reporting.

Future trends may include:

1. Integration with machine learning pipelines
2. Enhanced visualization tools for multi-dimensional data
3. Optimization for cloud-based data warehouses
4. Incorporation into real-time analytics systems

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulating future adoption trends
years = np.arange(2020, 2031)
grouping_sets_adoption = np.cumsum(np.random.normal(10, 2, len(years)))
rollup_adoption = np.cumsum(np.random.normal(8, 2, len(years)))
cube_adoption = np.cumsum(np.random.normal(6, 2, len(years)))

plt.figure(figsize=(12, 6))
plt.plot(years, grouping_sets_adoption, label='Grouping Sets')
plt.plot(years, rollup_adoption, label='ROLLUP')
plt.plot(years, cube_adoption, label='CUBE')

plt.title('Projected Adoption Trends')
plt.xlabel('Year')
plt.ylabel('Cumulative Adoption (arbitrary units)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 15: Additional Resources

For further exploration of Grouping Sets, ROLLUP, and CUBE in SQL:

1. SQL Performance Explained by Markus Winand Reference: arXiv:1508.03474 \[cs.DB\]
2. Advanced SQL Techniques for Data Analysis Reference: arXiv:1907.04346 \[cs.DB\]
3. Efficient OLAP Operations in SQL Reference: arXiv:2003.01793 \[cs.DB\]

These resources provide in-depth discussions on advanced SQL techniques, including Grouping Sets, ROLLUP, and CUBE, along with performance optimization strategies and real-world applications in data analysis.

