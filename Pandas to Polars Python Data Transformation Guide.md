## Pandas to Polars Python Data Transformation Guide
Slide 1: Basic Pandas to Polars DataFrame Creation

Converting Pandas DataFrame creation patterns to Polars involves understanding the fundamental differences in syntax while maintaining similar functionality. Polars emphasizes performance and memory efficiency through its columnar data structure.

```python
# Pandas DataFrame Creation
import pandas as pd
df_pandas = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['a', 'b', 'c']
})

# Polars DataFrame Creation
import polars as pl
df_polars = pl.DataFrame({
    'A': [1, 2, 3],
    'B': ['a', 'b', 'c']
})

# Output examples
print("Pandas DataFrame:")
print(df_pandas)
print("\nPolars DataFrame:")
print(df_polars)
```

Slide 2: Reading Data Files

Polars provides optimized methods for reading various file formats, offering significant performance improvements over Pandas, especially for large datasets. The syntax remains intuitive while leveraging parallel processing.

```python
# Pandas CSV reading
import pandas as pd
df_pandas = pd.read_csv('data.csv')

# Polars CSV reading with optimization
import polars as pl
df_polars = pl.read_csv('data.csv',
                        use_streaming=True,
                        n_threads=4)

# Reading parquet files
df_pandas_parquet = pd.read_parquet('data.parquet')
df_polars_parquet = pl.read_parquet('data.parquet')
```

Slide 3: Column Operations and Selection

Polars introduces a more explicit and chainable syntax for column operations, replacing Pandas' bracket notation with a more functional approach using the select and with\_columns methods.

```python
import polars as pl
import pandas as pd

# Pandas column selection
df_pandas = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
result_pandas = df_pandas[['A']]

# Polars column selection
df_polars = pl.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
result_polars = df_polars.select(['A'])

# Column transformation
# Pandas
df_pandas['C'] = df_pandas['A'] * 2

# Polars
df_polars = df_polars.with_columns(
    pl.col('A') * 2).alias('C')
```

Slide 4: Filtering and Boolean Operations

Polars implements a more expressive syntax for filtering operations, using the filter method with column expressions that provide better performance and clearer intent than Pandas' boolean indexing.

```python
# Pandas filtering
df_pandas = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': ['w', 'x', 'y', 'z']
})
filtered_pandas = df_pandas[df_pandas['A'] > 2]

# Polars filtering
df_polars = pl.DataFrame({
    'A': [1, 2, 3, 4],
    'B': ['w', 'x', 'y', 'z']
})
filtered_polars = df_polars.filter(pl.col('A') > 2)

# Multiple conditions
# Polars
complex_filter = df_polars.filter(
    (pl.col('A') > 2) & (pl.col('B') != 'z')
)
```

Slide 5: Aggregations and Grouping

The translation from Pandas to Polars groupby operations involves understanding Polars' expression-based syntax, which offers improved performance and more intuitive chaining of operations.

```python
import polars as pl
import pandas as pd

# Pandas groupby and aggregation
df_pandas = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B'],
    'value': [1, 2, 3, 4]
})
result_pandas = df_pandas.groupby('group')['value'].agg(['mean', 'sum'])

# Polars groupby and aggregation
df_polars = pl.DataFrame({
    'group': ['A', 'A', 'B', 'B'],
    'value': [1, 2, 3, 4]
})
result_polars = df_polars.groupby('group').agg([
    pl.col('value').mean().alias('mean'),
    pl.col('value').sum().alias('sum')
])
```

Slide 6: Time Series Operations

Polars introduces a more efficient approach to time series manipulations compared to Pandas, with built-in temporal functions that leverage Arrow's datetime capabilities for faster processing and memory efficiency.

```python
import polars as pl
import pandas as pd
from datetime import datetime

# Pandas time series
dates_pandas = pd.date_range('2023-01-01', periods=3)
df_pandas = pd.DataFrame({
    'date': dates_pandas,
    'value': [1, 2, 3]
})
result_pandas = df_pandas.set_index('date').resample('D').mean()

# Polars time series
df_polars = pl.DataFrame({
    'date': pl.date_range(
        datetime(2023, 1, 1),
        datetime(2023, 1, 3),
        interval='1d'
    ),
    'value': [1, 2, 3]
})
result_polars = df_polars.groupby_dynamic(
    'date',
    every='1d'
).agg(pl.col('value').mean())
```

Slide 7: Join Operations

Polars provides more explicit join syntax with improved performance characteristics over Pandas, especially for large datasets. The join operations maintain similar semantics while offering better memory management.

```python
# Pandas joins
df1_pandas = pd.DataFrame({
    'key': ['A', 'B', 'C'],
    'value1': [1, 2, 3]
})
df2_pandas = pd.DataFrame({
    'key': ['A', 'B', 'D'],
    'value2': [4, 5, 6]
})
merged_pandas = df1_pandas.merge(
    df2_pandas,
    on='key',
    how='left'
)

# Polars joins
df1_polars = pl.DataFrame({
    'key': ['A', 'B', 'C'],
    'value1': [1, 2, 3]
})
df2_polars = pl.DataFrame({
    'key': ['A', 'B', 'D'],
    'value2': [4, 5, 6]
})
merged_polars = df1_polars.join(
    df2_polars,
    on='key',
    how='left'
)
```

Slide 8: Window Functions

Polars window functions offer a more intuitive syntax compared to Pandas, with better performance for complex calculations over sliding windows and grouped data.

```python
# Pandas window functions
df_pandas = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B'],
    'value': [1, 2, 3, 4]
})
df_pandas['rolling_mean'] = df_pandas.groupby('group')['value'].transform(
    lambda x: x.rolling(window=2).mean()
)

# Polars window functions
df_polars = pl.DataFrame({
    'group': ['A', 'A', 'B', 'B'],
    'value': [1, 2, 3, 4]
})
result_polars = df_polars.with_columns([
    pl.col('value')
    .rolling_mean(window_size=2)
    .over('group')
    .alias('rolling_mean')
])
```

Slide 9: Data Type Handling

Polars emphasizes strict data typing and offers more explicit type conversion methods compared to Pandas, resulting in better memory usage and performance characteristics.

```python
import polars as pl
import pandas as pd

# Pandas type conversion
df_pandas = pd.DataFrame({
    'string_col': ['1', '2', '3'],
    'float_col': [1.1, 2.2, 3.3]
})
df_pandas['string_col'] = df_pandas['string_col'].astype(int)
df_pandas['float_col'] = df_pandas['float_col'].astype('float32')

# Polars type conversion
df_polars = pl.DataFrame({
    'string_col': ['1', '2', '3'],
    'float_col': [1.1, 2.2, 3.3]
})
df_polars = df_polars.with_columns([
    pl.col('string_col').cast(pl.Int64),
    pl.col('float_col').cast(pl.Float32)
])

# Check dtypes
print("Polars dtypes:", df_polars.dtypes)
```

Slide 10: Missing Value Handling

Polars handles null values differently from Pandas, using a more memory-efficient representation and providing clearer syntax for dealing with missing data through explicit null operations.

```python
import polars as pl
import pandas as pd

# Pandas missing value handling
df_pandas = pd.DataFrame({
    'A': [1, None, 3],
    'B': [4, 5, None]
})
cleaned_pandas = df_pandas.fillna(0)
dropped_pandas = df_pandas.dropna()

# Polars missing value handling
df_polars = pl.DataFrame({
    'A': [1, None, 3],
    'B': [4, 5, None]
})
cleaned_polars = df_polars.fill_null(0)
dropped_polars = df_polars.drop_nulls()

# More complex null handling in Polars
result_polars = df_polars.with_columns([
    pl.col('A').fill_null(pl.col('A').mean()),
    pl.col('B').fill_null(strategy='forward')
])
```

Slide 11: Advanced Data Transformation

Polars offers powerful expressions for complex data transformations, replacing Pandas' apply and transform methods with more efficient vectorized operations.

```python
import polars as pl
import pandas as pd

# Pandas transformation
df_pandas = pd.DataFrame({
    'values': [1, 2, 3, 4, 5]
})
df_pandas['scaled'] = df_pandas['values'].apply(
    lambda x: (x - x.mean()) / x.std()
)

# Polars transformation
df_polars = pl.DataFrame({
    'values': [1, 2, 3, 4, 5]
})
df_polars = df_polars.with_columns([
    ((pl.col('values') - pl.col('values').mean()) / 
     pl.col('values').std()).alias('scaled')
])

# Complex transformation with multiple columns
df_polars = df_polars.with_columns([
    pl.when(pl.col('values') > 3)
    .then(pl.col('values') * 2)
    .otherwise(pl.col('values'))
    .alias('conditional_transform')
])
```

Slide 12: Real-world Example - Time Series Analysis

A practical example demonstrating the migration from Pandas to Polars for analyzing financial time series data, showing significant performance improvements.

```python
import polars as pl
import pandas as pd
from datetime import datetime, timedelta

# Generate sample data
dates = [datetime(2023, 1, 1) + timedelta(days=x) for x in range(1000)]
values = list(range(1000))

# Pandas implementation
df_pandas = pd.DataFrame({
    'date': dates,
    'value': values
})
result_pandas = (df_pandas
    .set_index('date')
    .rolling(window='7d')
    .agg({'value': ['mean', 'std']})
)

# Polars implementation
df_polars = pl.DataFrame({
    'date': dates,
    'value': values
})
result_polars = df_polars.groupby_dynamic(
    'date',
    every='1d',
    period='7d'
).agg([
    pl.col('value').mean().alias('mean'),
    pl.col('value').std().alias('std')
])

# Performance comparison
print("Polars shape:", result_polars.shape)
print("Memory usage:", result_polars.estimated_size())
```

Slide 13: Real-world Example - Data Cleaning Pipeline

This example demonstrates a complete data cleaning pipeline translation from Pandas to Polars, showing how to handle multiple data quality issues efficiently in a production environment.

```python
import polars as pl
import pandas as pd
import numpy as np

# Sample dirty data
data = {
    'id': ['A1', 'A2', None, 'A4', 'A5'],
    'value': ['10.5', 'invalid', '15.7', '20.1', None],
    'category': ['X', 'Y', 'X', None, 'Z'],
    'date': ['2023-01-01', '2023-13-01', '2023-01-03', '2023-01-04', '2023-01-05']
}

# Pandas implementation
def clean_pandas(df):
    return (df
        .dropna(subset=['id'])
        .assign(
            value=pd.to_numeric(df['value'], errors='coerce'),
            date=pd.to_datetime(df['date'], errors='coerce')
        )
        .fillna({
            'category': 'UNKNOWN',
            'value': 0
        })
        .query('date.notnull()')
    )

# Polars implementation
def clean_polars(df):
    return (df
        .filter(pl.col('id').is_not_null())
        .with_columns([
            pl.col('value').cast(pl.Float64, strict=False),
            pl.col('date').str.strptime(pl.Date, fmt='%Y-%m-%d', strict=False)
        ])
        .with_columns([
            pl.col('category').fill_null('UNKNOWN'),
            pl.col('value').fill_null(0)
        ])
        .filter(pl.col('date').is_not_null())
    )

# Execute and compare
df_pandas = pd.DataFrame(data)
df_polars = pl.DataFrame(data)

clean_pandas_result = clean_pandas(df_pandas)
clean_polars_result = clean_polars(df_polars)

print("Polars Result:")
print(clean_polars_result)
```

Slide 14: Performance Optimization Techniques

Converting Pandas optimization patterns to Polars requires understanding its unique approach to memory management and parallel processing capabilities for maximum performance.

```python
import polars as pl
import pandas as pd
import numpy as np

# Large dataset simulation
n_rows = 1000000
data = {
    'id': range(n_rows),
    'value': np.random.randn(n_rows),
    'category': np.random.choice(['A', 'B', 'C'], n_rows)
}

# Pandas optimization
def optimize_pandas(df):
    return (df
        .memory_usage(deep=True)  # Check memory usage
        .pipe(lambda x: x.astype({'id': 'int32'}))  # Downcast types
        .groupby('category')
        .agg({'value': ['mean', 'std']})
    )

# Polars optimization
def optimize_polars(df):
    return (df
        .with_columns([
            pl.col('id').cast(pl.Int32)
        ])
        .groupby('category')
        .agg([
            pl.col('value').mean(),
            pl.col('value').std()
        ])
        .collect(streaming=True)  # Use streaming for large datasets
    )

# Lazy evaluation in Polars
lazy_query = (pl.DataFrame(data)
    .lazy()
    .with_columns([
        pl.col('id').cast(pl.Int32)
    ])
    .groupby('category')
    .agg([
        pl.col('value').mean(),
        pl.col('value').std()
    ])
)

# Execute optimized query
result = lazy_query.collect()
```

Slide 15: Additional Resources

*   Polars Official Documentation
    *   [https://pola-rs.github.io/polars/](https://pola-rs.github.io/polars/)
*   High-Performance Data Processing with Polars
    *   [https://towardsdatascience.com/polars-vs-pandas](https://towardsdatascience.com/polars-vs-pandas)
*   Performance Comparison Studies
    *   [https://h2oai.github.io/db-benchmark/](https://h2oai.github.io/db-benchmark/)
*   Best Practices and Migration Guide
    *   [https://python.polars.tech/migration/pandas](https://python.polars.tech/migration/pandas)
*   Community Resources and Examples
    *   [https://github.com/pola-rs/polars/tree/master/examples](https://github.com/pola-rs/polars/tree/master/examples)
*   Reference Articles
    *   Rust, Python and the Future of Data Science
        *   [https://www.nature.com/articles/s41592-023-02036-1](https://www.nature.com/articles/s41592-023-02036-1)
    *   Modern Data Processing with Apache Arrow
        *   [https://arrow.apache.org/blog/](https://arrow.apache.org/blog/)
    *   Getting Started with Polars for Data Science
        *   [https://databricks.com/blog/polars-integration](https://databricks.com/blog/polars-integration)

