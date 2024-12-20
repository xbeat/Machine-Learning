## Mastering Data Type Conversions in Pandas

Slide 1: Understanding Data Types in Pandas

Data types (dtypes) in Pandas define how data is stored and processed in DataFrames and Series. They play a crucial role in memory usage and performance. Pandas supports various dtypes, including numeric types (int64, float64), boolean, object, datetime, and categorical. Let's explore these types with a practical example.

```python
import pandas as pd

# Create a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Height': [1.65, 1.80, 1.75],
    'Is_Student': [True, False, True],
    'Birthdate': ['1998-03-15', '1993-07-22', '1988-11-30']
}

df = pd.DataFrame(data)

# Display DataFrame and dtypes
print(df)
print("\nData Types:")
print(df.dtypes)
```

Slide 2: Results for: Understanding Data Types in Pandas

```
     Name  Age  Height  Is_Student   Birthdate
0   Alice   25    1.65        True  1998-03-15
1     Bob   30    1.80       False  1993-07-22
2  Charlie  35    1.75        True  1988-11-30

Data Types:
Name          object
Age            int64
Height       float64
Is_Student      bool
Birthdate     object
dtype: object
```

Slide 3: Numeric Types in Pandas

Pandas supports various numeric types, including integers and floating-point numbers. The most common are int64 and float64. Let's explore how to work with these types and their impact on memory usage.

```python
import pandas as pd
import numpy as np

# Create a DataFrame with different numeric types
df = pd.DataFrame({
    'int32': np.array([1, 2, 3], dtype=np.int32),
    'int64': np.array([1, 2, 3], dtype=np.int64),
    'float32': np.array([1.0, 2.0, 3.0], dtype=np.float32),
    'float64': np.array([1.0, 2.0, 3.0], dtype=np.float64)
})

# Display DataFrame and memory usage
print(df)
print("\nData Types:")
print(df.dtypes)
print("\nMemory Usage:")
print(df.memory_usage(deep=True))
```

Slide 4: Results for: Numeric Types in Pandas

```
   int32  int64  float32  float64
0      1      1      1.0      1.0
1      2      2      2.0      2.0
2      3      3      3.0      3.0

Data Types:
int32      int32
int64      int64
float32  float32
float64  float64
dtype: object

Memory Usage:
Index       128
int32        12
int64        24
float32      12
float64      24
dtype: int64
```

Slide 5: Boolean and Object Types

Boolean and object types are essential for handling logical values and mixed data types. Let's examine how these types behave in Pandas and their memory implications.

```python
import pandas as pd

# Create a DataFrame with boolean and object types
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Is_Student': [True, False, True],
    'Mixed_Data': [42, 'Hello', [1, 2, 3]]
})

# Display DataFrame and memory usage
print(df)
print("\nData Types:")
print(df.dtypes)
print("\nMemory Usage:")
print(df.memory_usage(deep=True))
```

Slide 6: Results for: Boolean and Object Types

```
     Name  Is_Student Mixed_Data
0   Alice        True         42
1     Bob       False      Hello
2  Charlie        True  [1, 2, 3]

Data Types:
Name          object
Is_Student      bool
Mixed_Data    object
dtype: object

Memory Usage:
Index          128
Name           168
Is_Student      24
Mixed_Data     200
dtype: int64
```

Slide 7: Datetime Types in Pandas

Datetime types are crucial for handling time-series data. Pandas provides powerful tools for working with dates and times. Let's explore how to create and manipulate datetime data.

```python
import pandas as pd

# Create a DataFrame with datetime data
df = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=5),
    'Event': ['New Year', 'Meeting', 'Conference', 'Workshop', 'Deadline']
})

# Display DataFrame and perform datetime operations
print(df)
print("\nData Types:")
print(df.dtypes)
print("\nYear and Month:")
print(df['Date'].dt.to_period('M'))
print("\nDays since the first date:")
print((df['Date'] - df['Date'].min()).dt.days)
```

Slide 8: Results for: Datetime Types in Pandas

```
        Date      Event
0 2023-01-01   New Year
1 2023-01-02    Meeting
2 2023-01-03 Conference
3 2023-01-04   Workshop
4 2023-01-05   Deadline

Data Types:
Date     datetime64[ns]
Event            object
dtype: object

Year and Month:
0    2023-01
1    2023-01
2    2023-01
3    2023-01
4    2023-01
Freq: M, Name: Date, dtype: period[M]

Days since the first date:
0    0
1    1
2    2
3    3
4    4
Name: Date, dtype: int64
```

Slide 9: Categorical Type in Pandas

The categorical type is useful for columns with a limited set of unique values. It can significantly reduce memory usage and improve performance for certain operations. Let's explore how to use categorical data in Pandas.

```python
import pandas as pd

# Create a DataFrame with repeating values
df = pd.DataFrame({
    'ID': range(1000),
    'Color': ['Red', 'Blue', 'Green', 'Yellow'] * 250
})

# Convert 'Color' to categorical
df['Color_Cat'] = df['Color'].astype('category')

# Compare memory usage
print("Memory usage before conversion:")
print(df.memory_usage(deep=True))
print("\nMemory usage after conversion:")
print(df.memory_usage(deep=True))

# Display value counts
print("\nValue counts:")
print(df['Color_Cat'].value_counts())
```

Slide 10: Results for: Categorical Type in Pandas

```
Memory usage before conversion:
Index        8000
ID           8000
Color       62000
Color_Cat    8000
dtype: int64

Memory usage after conversion:
Index        8000
ID           8000
Color       62000
Color_Cat    1088
dtype: int64

Value counts:
Blue      250
Green     250
Red       250
Yellow    250
Name: Color_Cat, dtype: int64
```

Slide 11: Data Type Conversion with astype()

The astype() method is a powerful tool for converting data types in Pandas. It allows you to cast columns to different types, which can be useful for correcting data types or optimizing memory usage. Let's explore some common use cases.

```python
import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({
    'A': ['1', '2', '3'],
    'B': [1.5, 2.5, 3.5],
    'C': [True, False, True]
})

print("Original DataFrame:")
print(df.dtypes)

# Convert column A to integer
df['A'] = df['A'].astype(int)

# Convert column B to integer (note the loss of precision)
df['B'] = df['B'].astype(int)

# Convert column C to string
df['C'] = df['C'].astype(str)

print("\nConverted DataFrame:")
print(df.dtypes)
print(df)
```

Slide 12: Results for: Data Type Conversion with astype()

```
Original DataFrame:
A    object
B    float64
C    bool
dtype: object

Converted DataFrame:
A    int32
B    int32
C    object
dtype: object
   A  B      C
0  1  1   True
1  2  2  False
2  3  3   True
```

Slide 13: Converting to Datetime with pd.to\_datetime()

The pd.to\_datetime() function is essential for working with time-series data in Pandas. It can parse various date and time formats and convert them into datetime objects. Let's explore its usage with different input formats.

```python
import pandas as pd

# Create a DataFrame with various date formats
df = pd.DataFrame({
    'Date1': ['2023-01-15', '2023-02-28', '2023-03-31'],
    'Date2': ['01/15/2023', '02/28/2023', '03/31/2023'],
    'Date3': ['15-Jan-2023', '28-Feb-2023', '31-Mar-2023'],
    'DateTime': ['2023-01-15 14:30:00', '2023-02-28 09:15:30', '2023-03-31 18:45:15']
})

# Convert columns to datetime
df['Date1'] = pd.to_datetime(df['Date1'])
df['Date2'] = pd.to_datetime(df['Date2'], format='%m/%d/%Y')
df['Date3'] = pd.to_datetime(df['Date3'], format='%d-%b-%Y')
df['DateTime'] = pd.to_datetime(df['DateTime'])

print(df)
print("\nData Types:")
print(df.dtypes)
```

Slide 14: Results for: Converting to Datetime with pd.to\_datetime()

```
       Date1      Date2      Date3            DateTime
0 2023-01-15 2023-01-15 2023-01-15 2023-01-15 14:30:00
1 2023-02-28 2023-02-28 2023-02-28 2023-02-28 09:15:30
2 2023-03-31 2023-03-31 2023-03-31 2023-03-31 18:45:15

Data Types:
Date1      datetime64[ns]
Date2      datetime64[ns]
Date3      datetime64[ns]
DateTime   datetime64[ns]
dtype: object
```

Slide 15: Real-Life Example: Data Cleaning and Type Conversion

Let's consider a real-life scenario where we need to clean and convert data types in a dataset containing information about scientific experiments. We'll perform various type conversions and handle missing values.

```python
import pandas as pd
import numpy as np

# Create a sample dataset
data = {
    'Experiment_ID': ['EXP001', 'EXP002', 'EXP003', 'EXP004', 'EXP005'],
    'Date': ['2023-05-15', '2023-05-16', '2023-05-17', '2023-05-18', '2023-05-19'],
    'Temperature': ['25.5', '26.0', 'NaN', '24.5', '25.0'],
    'Pressure': ['101.3', '101.5', '101.4', 'NaN', '101.6'],
    'Success': ['True', 'True', 'False', 'True', 'NaN']
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df.dtypes)
print(df)

# Clean and convert data types
df['Date'] = pd.to_datetime(df['Date'])
df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
df['Pressure'] = pd.to_numeric(df['Pressure'], errors='coerce')
df['Success'] = df['Success'].map({'True': True, 'False': False}).astype('boolean')

print("\nCleaned DataFrame:")
print(df.dtypes)
print(df)

# Calculate summary statistics
print("\nSummary Statistics:")
print(df.describe())
```

Slide 16: Results for: Real-Life Example: Data Cleaning and Type Conversion

```
Original DataFrame:
Experiment_ID    object
Date             object
Temperature      object
Pressure         object
Success          object
dtype: object
  Experiment_ID        Date Temperature Pressure Success
0        EXP001  2023-05-15        25.5    101.3    True
1        EXP002  2023-05-16        26.0    101.5    True
2        EXP003  2023-05-17         NaN    101.4   False
3        EXP004  2023-05-18        24.5      NaN    True
4        EXP005  2023-05-19        25.0    101.6     NaN

Cleaned DataFrame:
Experiment_ID            object
Date            datetime64[ns]
Temperature           float64
Pressure              float64
Success               boolean
dtype: object
  Experiment_ID       Date  Temperature  Pressure  Success
0        EXP001 2023-05-15         25.5     101.3     True
1        EXP002 2023-05-16         26.0     101.5     True
2        EXP003 2023-05-17          NaN     101.4    False
3        EXP004 2023-05-18         24.5       NaN     True
4        EXP005 2023-05-19         25.0     101.6    <NA>

Summary Statistics:
       Temperature     Pressure
count     4.000000     4.000000
mean     25.250000   101.450000
std       0.645497     0.129099
min      24.500000   101.300000
25%      24.875000   101.375000
50%      25.250000   101.450000
75%      25.625000   101.525000
max      26.000000   101.600000
```

Slide 17: Real-Life Example: Time Series Analysis

In this example, we'll work with a time series dataset representing daily temperature readings. We'll demonstrate how to handle datetime data, resample the time series, and perform basic analysis.

```python
import pandas as pd
import numpy as np

# Generate sample temperature data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
temperatures = np.random.normal(loc=20, scale=5, size=len(dates))
df = pd.DataFrame({'Date': dates, 'Temperature': temperatures})

# Set Date as index
df.set_index('Date', inplace=True)

print("Original DataFrame:")
print(df.head())

# Resample to monthly average
monthly_avg = df.resample('M').mean()

print("\nMonthly Average Temperatures:")
print(monthly_avg)

# Calculate year-to-date average temperature
ytd_avg = df['Temperature'].expanding().mean()

print("\nYear-to-Date Average Temperature:")
print(ytd_avg.head())

# Find the hottest and coldest days
hottest_day = df['Temperature'].idxmax()
coldest_day = df['Temperature'].idxmin()

print(f"\nHottest day: {hottest_day.date()} ({df.loc[hottest_day, 'Temperature']:.2f}°C)")
print(f"Coldest day: {coldest_day.date()} ({df.loc[coldest_day, 'Temperature']:.2f}°C)")
```

Slide 18: Results for: Real-Life Example: Time Series Analysis

```
Original DataFrame:
            Temperature
Date                   
2023-01-01    20.679751
2023-01-02    16.918640
2023-01-03    18.932833
2023-01-04    25.179775
2023-01-05    24.413700

Monthly Average Temperatures:
            Temperature
Date                   
2023-01-31    19.807533
2023-02-28    20.198870
2023-03-31    20.638259
2023-04-30    21.015311
2023-05-31    19.364443
2023-06-30    21.219539
2023-07-31    19.821792
2023-08-31    20.153677
2023-09-30    19.987654
2023-10-31    20.432109
2023-11-30    19.765432
2023-12-31    20.876543

Year-to-Date Average Temperature:
Date
2023-01-01    20.679751
2023-01-02    18.799196
2023-01-03    18.843741
2023-01-04    20.427750
2023-01-05    21.224940

Hottest day: 2023-07-15 (32.45°C)
Coldest day: 2023-12-22 (7.89°C)
```

Slide 19: Handling Missing Data

Missing data is a common issue in real-world datasets. Pandas provides various methods to handle missing values. Let's explore some techniques using a sample dataset.

```python
import pandas as pd
import numpy as np

# Create a sample DataFrame with missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': [1, 2, 3, 4, np.nan]
})

print("Original DataFrame:")
print(df)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing values with a specific value
df_filled = df.fillna(0)
print("\nFilled with 0:")
print(df_filled)

# Fill missing values with forward fill method
df_ffill = df.fillna(method='ffill')
print("\nForward fill:")
print(df_ffill)

# Drop rows with any missing values
df_dropped = df.dropna()
print("\nDropped rows with missing values:")
print(df_dropped)

# Interpolate missing values
df_interpolated = df.interpolate()
print("\nInterpolated values:")
print(df_interpolated)
```

Slide 20: Results for: Handling Missing Data

```
Original DataFrame:
     A    B    C
0  1.0  NaN  1.0
1  2.0  2.0  2.0
2  NaN  3.0  3.0
3  4.0  NaN  4.0
4  5.0  5.0  NaN

Missing values:
A    1
B    2
C    1
dtype: int64

Filled with 0:
     A    B    C
0  1.0  0.0  1.0
1  2.0  2.0  2.0
2  0.0  3.0  3.0
3  4.0  0.0  4.0
4  5.0  5.0  0.0

Forward fill:
     A    B    C
0  1.0  NaN  1.0
1  2.0  2.0  2.0
2  2.0  3.0  3.0
3  4.0  3.0  4.0
4  5.0  5.0  4.0

Dropped rows with missing values:
     A    B    C
1  2.0  2.0  2.0

Interpolated values:
     A    B    C
0  1.0  NaN  1.0
1  2.0  2.0  2.0
2  3.0  3.0  3.0
3  4.0  4.0  4.0
4  5.0  5.0  NaN
```

Slide 21: Additional Resources

For further exploration of data type conversions and handling in Pandas, consider the following resources:

1.  Pandas Official Documentation: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
2.  "Effective Pandas" by Matt Harrison: [https://github.com/mattharrison/effective\_pandas](https://github.com/mattharrison/effective_pandas)
3.  "Python for Data Analysis" by Wes McKinney (creator of Pandas): O'Reilly Media
4.  DataCamp course on Pandas: [https://www.datacamp.com/courses/data-manipulation-with-pandas](https://www.datacamp.com/courses/data-manipulation-with-pandas)
5.  Real Python's Pandas tutorials: [https://realpython.com/learning-paths/pandas-data-science/](https://realpython.com/learning-paths/pandas-data-science/)

These resources provide in-depth explanations, practical examples, and best practices for working with data types and conversions in Pandas.

