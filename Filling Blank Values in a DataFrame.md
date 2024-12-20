## Filling Blank Values in a DataFrame
Slide 1: Filling NaN Values in a DataFrame

To fill all blank (NaN) values in a DataFrame with zero using Python, we can use the fillna() method. This method is part of the pandas library, which provides powerful data manipulation tools for Python.

Slide 2: Source Code for Filling NaN Values in a DataFrame

```python
import pandas as pd
import numpy as np

# Create a sample DataFrame with NaN values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, 7, 8],
    'C': [9, 10, 11, np.nan]
})

print("Original DataFrame:")
print(df)

# Fill NaN values with zero
df_filled = df.fillna(0)

print("\nDataFrame with NaN values filled with zero:")
print(df_filled)
```

Slide 3: Results for: Filling NaN Values in a DataFrame

```
Original DataFrame:
     A    B     C
0  1.0  5.0   9.0
1  2.0  NaN  10.0
2  NaN  7.0  11.0
3  4.0  8.0   NaN

DataFrame with NaN values filled with zero:
     A    B     C
0  1.0  5.0   9.0
1  2.0  0.0  10.0
2  0.0  7.0  11.0
3  4.0  8.0   0.0
```

Slide 4: Understanding the fillna() Method

The fillna() method is versatile and can be used in various ways to fill NaN values. It can replace NaN values with a specific value, a dictionary of values, or even use forward or backward fill methods.

Slide 5: Source Code for Understanding the fillna() Method

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, 7, 8],
    'C': [9, 10, 11, np.nan]
})

# Fill NaN with different values for each column
df_custom = df.fillna({'A': 100, 'B': 200, 'C': 300})

print("DataFrame with custom NaN fill values:")
print(df_custom)
```

Slide 6: Filling NaN Values with Column Mean

Sometimes, it's more appropriate to fill NaN values with the mean of the column rather than a fixed value like zero. This can help maintain the statistical properties of the data.

Slide 7: Source Code for Filling NaN Values with Column Mean

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, 7, 8],
    'C': [9, 10, 11, np.nan]
})

# Fill NaN values with column mean
df_mean_filled = df.fillna(df.mean())

print("DataFrame with NaN values filled with column mean:")
print(df_mean_filled)
```

Slide 8: Forward Fill Method

The forward fill method propagates the last valid observation forward to next valid backfill. This is useful when dealing with time series data where you want to carry forward the last known value.

Slide 9: Source Code for Forward Fill Method

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, 7, 8],
    'C': [9, 10, 11, np.nan]
})

# Forward fill NaN values
df_ffill = df.fillna(method='ffill')

print("DataFrame with forward fill:")
print(df_ffill)
```

Slide 10: Backward Fill Method

The backward fill method works similarly to forward fill, but it fills NaN values with the next valid value instead of the previous one. This can be useful when you want to backfill missing data.

Slide 11: Source Code for Backward Fill Method

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, 7, 8],
    'C': [9, 10, 11, np.nan]
})

# Backward fill NaN values
df_bfill = df.fillna(method='bfill')

print("DataFrame with backward fill:")
print(df_bfill)
```

Slide 12: Filling NaN Values in Specific Columns

Sometimes you may want to fill NaN values only in specific columns of your DataFrame. This can be achieved by selecting the columns before applying the fillna() method.

Slide 13: Source Code for Filling NaN Values in Specific Columns

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, 7, 8],
    'C': [9, 10, 11, np.nan],
    'D': [np.nan, 13, 14, 15]
})

# Fill NaN values only in columns A and B
df[['A', 'B']] = df[['A', 'B']].fillna(0)

print("DataFrame with NaN values filled in specific columns:")
print(df)
```

Slide 14: Real-Life Example: Weather Data

In weather data analysis, we often encounter missing values due to sensor malfunctions or data transmission issues. Let's see how we can handle NaN values in a weather dataset.

Slide 15: Source Code for Real-Life Example: Weather Data

```python
import pandas as pd
import numpy as np

# Create a sample weather DataFrame
weather_data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=5),
    'Temperature': [25.5, np.nan, 24.0, 26.5, np.nan],
    'Humidity': [60, 65, np.nan, 70, 68],
    'Wind_Speed': [10, 12, 15, np.nan, 11]
})

print("Original Weather Data:")
print(weather_data)

# Fill NaN values: Temperature with mean, Humidity and Wind_Speed with 0
weather_data['Temperature'] = weather_data['Temperature'].fillna(weather_data['Temperature'].mean())
weather_data[['Humidity', 'Wind_Speed']] = weather_data[['Humidity', 'Wind_Speed']].fillna(0)

print("\nWeather Data after filling NaN values:")
print(weather_data)
```

Slide 16: Real-Life Example: Survey Responses

In survey analysis, missing responses are common. Let's see how we can handle NaN values in a survey dataset, using forward fill for categorical data and mean fill for numerical data.

Slide 17: Source Code for Real-Life Example: Survey Responses

```python
import pandas as pd
import numpy as np

# Create a sample survey DataFrame
survey_data = pd.DataFrame({
    'Respondent': range(1, 6),
    'Age': [25, 30, np.nan, 35, 28],
    'Gender': ['M', 'F', np.nan, 'M', 'F'],
    'Satisfaction': [4, np.nan, 3, 5, np.nan]
})

print("Original Survey Data:")
print(survey_data)

# Fill NaN values: Age with mean, Gender with forward fill, Satisfaction with median
survey_data['Age'] = survey_data['Age'].fillna(survey_data['Age'].mean())
survey_data['Gender'] = survey_data['Gender'].fillna(method='ffill')
survey_data['Satisfaction'] = survey_data['Satisfaction'].fillna(survey_data['Satisfaction'].median())

print("\nSurvey Data after filling NaN values:")
print(survey_data)
```

Slide 18: Additional Resources

For more information on handling missing data in pandas, refer to the following resources:

1.  Pandas Official Documentation on Working with Missing Data: [https://pandas.pydata.org/pandas-docs/stable/user\_guide/missing\_data.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html)
2.  "Handling Missing Values in Data" by Rahul Agarwal on Towards Data Science: [https://towardsdatascience.com/handling-missing-values-in-data-modeling-56e0ba6e7e0d](https://towardsdatascience.com/handling-missing-values-in-data-modeling-56e0ba6e7e0d)
3.  "Dealing with Missing Data" by Jason Brownlee on Machine Learning Mastery: [https://machinelearningmastery.com/handle-missing-data-python/](https://machinelearningmastery.com/handle-missing-data-python/)

These resources provide in-depth explanations and additional techniques for handling missing data in various scenarios.

