## Introduction to Data Cleaning with Pandas and Python

Slide 1: Introduction to Data Cleaning with Pandas

Data cleaning is a crucial step in data analysis, ensuring the data is accurate, consistent, and ready for analysis. Pandas, a powerful Python library, provides various tools and functions to handle data cleaning tasks efficiently.

Code:

```python
import pandas as pd
```

Slide 2: Handling Missing Data

Missing data is a common issue in datasets. Pandas provides several methods to handle missing values, such as dropping rows or columns, filling with a specific value, or using interpolation techniques.

Code:

```python
# Drop rows with missing values
df.dropna(inplace=True)

# Fill missing values with a specific value
df.fillna(0, inplace=True)

# Fill missing values with the mean of the column
df['column_name'] = df['column_name'].fillna(df['column_name'].mean())
```

Slide 3: Removing Duplicates

Duplicate data can lead to inaccurate analysis and skewed results. Pandas offers methods to identify and remove duplicate rows or columns from a DataFrame.

Code:

```python
# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Remove duplicate rows based on specific columns
df.drop_duplicates(subset=['column1', 'column2'], inplace=True)
```

Slide 4: Data Transformation

Data transformation involves converting data into a more suitable format for analysis. Pandas provides functions to perform operations like data type conversion, string manipulation, and date/time handling.

Code:

```python
# Convert data types
df['column_name'] = df['column_name'].astype('int')

# String manipulation
df['column_name'] = df['column_name'].str.lower()

# Date/time handling
df['date_column'] = pd.to_datetime(df['date_column'])
```

Slide 5: Handling Outliers

Outliers can significantly impact the analysis results. Pandas offers various techniques to identify and handle outliers, such as using statistical methods or applying domain-specific rules.

Code:

```python
# Identify outliers using z-scores
z_scores = np.abs(df['column_name'] - df['column_name'].mean()) / df['column_name'].std()
outliers = df[z_scores > 3]

# Replace outliers with a specific value
df.loc[z_scores > 3, 'column_name'] = df['column_name'].median()
```

Slide 6: Data Filtering

Data filtering is the process of selecting a subset of data based on specific criteria. Pandas provides powerful filtering capabilities using boolean indexing and conditional statements.

Code:

```python
# Filter rows based on a condition
filtered_df = df[df['column_name'] > 10]

# Filter rows based on multiple conditions
filtered_df = df[(df['column1'] > 5) & (df['column2'] == 'value')]
```

Slide 7: Handling Categorical Data

Categorical data represents distinct categories or groups. Pandas offers tools to work with categorical data, such as encoding categorical variables and performing operations like grouping and aggregation.

Code:

```python
# Convert a column to categorical data type
df['column_name'] = df['column_name'].astype('category')

# Encode categorical data
encoded_df = pd.get_dummies(df, columns=['column_name'])
```

Slide 8: Data Merging and Joining

Merging and joining data from multiple sources is a common task in data analysis. Pandas provides methods to combine datasets based on common columns or indexes.

Code:

```python
# Merge two DataFrames based on a common column
merged_df = pd.merge(df1, df2, on='common_column')

# Join two DataFrames based on indexes
joined_df = df1.join(df2, how='inner')
```

Slide 9: Data Reshaping

Data reshaping involves transforming the structure of a DataFrame, such as pivoting or unpivoting data. Pandas offers functions like `melt` and `pivot` to reshape data for better analysis.

Code:

```python
# Unpivot (melt) data
melted_df = pd.melt(df, id_vars=['column1', 'column2'], var_name='variable', value_name='value')

# Pivot data
pivoted_df = df.pivot(index='column1', columns='column2', values='column3')
```

Slide 10: Data Imputation

Data imputation is the process of replacing missing data with substituted values. Pandas provides various imputation techniques, such as mean, median, or mode imputation, as well as more advanced methods like regression imputation.

Code:

```python
# Mean imputation
df['column_name'] = df['column_name'].fillna(df['column_name'].mean())

# Regression imputation
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
df['column_name'] = df['column_name'].fillna(regressor.predict(X_test))
```

Slide 11: Data Normalization

Data normalization is a technique used to rescale data to a common range, often between 0 and 1 or -1 and 1. This can be useful for certain machine learning algorithms or when dealing with different scales of data.

Code:

```python
# Min-max normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Standardization (z-score normalization)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
standardized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
```

Slide 12: Data Validation

Data validation is the process of ensuring that data adheres to specific rules, constraints, or formats. Pandas provides methods to validate data and handle violations, such as raising errors or applying custom functions.

Code:

```python
# Validate data types
df = df.astype({'column1': 'int', 'column2': 'float'})

# Apply custom validation function
def validate_age(age):
    if age < 0 or age > 120:
        raise ValueError('Invalid age')
    return age

df['age'] = df['age'].apply(validate_age)
```

Slide 13: Data Profiling

Data profiling involves summarizing and understanding the characteristics of a dataset. Pandas offers various methods to generate descriptive statistics, identify data types, and detect missing values or outliers.

Code:

```python
# Generate descriptive statistics
df.describe()

# Identify data types
df.dtypes

# Detect missing values
df.isnull().sum()

# Detect duplicates
df.duplicated().sum()
```

Slide 14: Conclusion

Data cleaning is an essential step in the data analysis process. Pandas provides a powerful and flexible toolset to handle various data cleaning tasks, from handling missing data and duplicates to data transformation, filtering, and reshaping. By mastering these techniques, you can ensure your data is accurate, consistent, and ready for meaningful analysis.
