## Mastering Data Manipulation with Pandas for Machine Learning
Slide 1: Data Loading and Basic Operations in Pandas

Pandas provides efficient methods for loading data from various sources and performing basic operations. Understanding these fundamentals is crucial for any data manipulation task in machine learning pipelines. Let's explore loading CSV files and essential DataFrame operations.

```python
import pandas as pd
import numpy as np

# Load sample dataset
df = pd.read_csv('sample_data.csv')

# Basic operations
print("First 5 rows:")
print(df.head())

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

# Basic statistics
print("\nNumerical Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())
```

Slide 2: Advanced Data Selection and Filtering

Data selection and filtering are fundamental skills in data manipulation. Pandas offers powerful indexing capabilities through loc, iloc, and boolean indexing, allowing precise control over data access and modification.

```python
import pandas as pd

# Create sample dataset
data = {
    'name': ['John', 'Anna', 'Peter', 'Linda'],
    'age': [28, 22, 35, 32],
    'salary': [50000, 45000, 65000, 55000],
    'department': ['IT', 'HR', 'IT', 'Finance']
}
df = pd.DataFrame(data)

# Boolean indexing
high_salary = df[df['salary'] > 50000]

# loc for label-based indexing
it_dept = df.loc[df['department'] == 'IT']

# iloc for integer-based indexing
first_two = df.iloc[0:2, [0, 2]]

print("High Salary Employees:\n", high_salary)
print("\nIT Department:\n", it_dept)
print("\nFirst Two Rows (name and salary):\n", first_two)
```

Slide 3: Data Cleaning and Preprocessing

Data cleaning is a critical step in preparing datasets for machine learning. This includes handling missing values, removing duplicates, and dealing with outliers using Pandas' built-in functions.

```python
import pandas as pd
import numpy as np

# Create dataset with missing values and duplicates
data = {
    'feature1': [1, 2, np.nan, 4, 2, 1],
    'feature2': [10, 20, 30, np.nan, 20, 10],
    'feature3': ['A', 'B', 'C', 'D', 'B', 'A']
}
df = pd.DataFrame(data)

# Handle missing values
df_cleaned = df.copy()
df_cleaned['feature1'].fillna(df['feature1'].mean(), inplace=True)
df_cleaned['feature2'].fillna(df['feature2'].median(), inplace=True)

# Remove duplicates
df_unique = df_cleaned.drop_duplicates()

# Handle outliers using IQR method
Q1 = df_cleaned['feature1'].quantile(0.25)
Q3 = df_cleaned['feature1'].quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = df_cleaned[
    ~((df_cleaned['feature1'] < (Q1 - 1.5 * IQR)) | 
      (df_cleaned['feature1'] > (Q3 + 1.5 * IQR)))
]

print("Original DataFrame:\n", df)
print("\nCleaned DataFrame:\n", df_cleaned)
print("\nUnique Records:\n", df_unique)
print("\nDataFrame without Outliers:\n", df_no_outliers)
```

Slide 4: Feature Engineering and Transformation

Feature engineering transforms raw data into meaningful representations for machine learning models. This process includes creating new features, encoding categorical variables, and scaling numerical features.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Create sample dataset
data = {
    'age': [25, 35, 45, 20, 30],
    'income': [30000, 45000, 90000, 25000, 50000],
    'education': ['High School', 'Bachelor', 'Master', 'High School', 'PhD'],
    'city': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin']
}
df = pd.DataFrame(data)

# Create new features
df['income_per_age'] = df['income'] / df['age']

# Encode categorical variables
le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['city'], prefix=['city'])

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['age', 'income']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("Original DataFrame with New Features:\n", df)
print("\nOne-hot Encoded DataFrame:\n", df_encoded)
```

Slide 5: Time Series Data Manipulation

Time series data requires special handling in Pandas, including date parsing, resampling, and rolling window calculations. These operations are essential for analyzing temporal patterns and creating time-based features.

```python
import pandas as pd
import numpy as np

# Create time series data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
data = {
    'date': dates,
    'sales': np.random.normal(1000, 100, len(dates)),
    'temperature': np.random.normal(25, 5, len(dates))
}
df = pd.DataFrame(data)
df.set_index('date', inplace=True)

# Resample to monthly frequency
monthly_sales = df['sales'].resample('M').mean()

# Calculate rolling average
rolling_avg = df['sales'].rolling(window=7).mean()

# Create time-based features
df['year'] = df.index.year
df['month'] = df.index.month
df['day_of_week'] = df.index.dayofweek

print("Monthly Sales:\n", monthly_sales.head())
print("\n7-Day Rolling Average:\n", rolling_avg.head())
print("\nTime Features:\n", df.head())
```

Slide 6: Data Aggregation and Grouping

Pandas provides powerful tools for grouping and aggregating data, essential for feature engineering and understanding patterns within different categories or time periods.

```python
import pandas as pd
import numpy as np

# Create sample sales data
data = {
    'date': pd.date_range('2023-01-01', '2023-12-31', freq='D'),
    'product': np.random.choice(['A', 'B', 'C'], size=365),
    'region': np.random.choice(['North', 'South', 'East', 'West'], size=365),
    'sales': np.random.normal(1000, 200, size=365),
    'units': np.random.randint(50, 150, size=365)
}
df = pd.DataFrame(data)

# Group by multiple columns
grouped_stats = df.groupby(['product', 'region']).agg({
    'sales': ['mean', 'sum', 'std'],
    'units': ['mean', 'sum']
}).round(2)

# Time-based grouping
monthly_stats = df.groupby(df['date'].dt.month).agg({
    'sales': ['mean', 'sum'],
    'units': 'sum'
})

# Custom aggregation
def revenue_per_unit(x):
    return (x['sales'] / x['units']).mean()

custom_metrics = df.groupby('product').apply(revenue_per_unit)

print("Product & Region Stats:\n", grouped_stats)
print("\nMonthly Stats:\n", monthly_stats)
print("\nRevenue per Unit:\n", custom_metrics)
```

Slide 7: Advanced Data Merging and Joining

Understanding how to combine multiple datasets is crucial for feature engineering and creating comprehensive datasets for machine learning models.

```python
import pandas as pd

# Create sample customer data
customers = pd.DataFrame({
    'customer_id': range(1, 6),
    'name': ['John', 'Anna', 'Peter', 'Linda', 'Max'],
    'country': ['USA', 'UK', 'USA', 'Canada', 'UK']
})

# Create sample order data
orders = pd.DataFrame({
    'order_id': range(1, 8),
    'customer_id': [1, 2, 3, 1, 4, 2, 5],
    'amount': [100, 200, 150, 300, 250, 175, 225]
})

# Different types of joins
inner_join = pd.merge(customers, orders, on='customer_id', how='inner')
left_join = pd.merge(customers, orders, on='customer_id', how='left')
right_join = pd.merge(customers, orders, on='customer_id', how='right')

# Aggregated merge
customer_summary = orders.groupby('customer_id').agg({
    'order_id': 'count',
    'amount': ['sum', 'mean']
}).round(2)

final_df = pd.merge(customers, customer_summary, left_on='customer_id', 
                   right_index=True, how='left')

print("Inner Join:\n", inner_join)
print("\nCustomer Summary:\n", customer_summary)
print("\nFinal Merged DataFrame:\n", final_df)
```

Slide 8: Handling Categorical Data for Machine Learning

Categorical data requires special preprocessing before it can be used in machine learning models. Pandas provides various methods for encoding categorical variables, including one-hot encoding, label encoding, and ordinal encoding.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

# Create sample dataset with different types of categorical variables
data = {
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['S', 'M', 'L', 'XL', 'M'],
    'brand': ['A', 'B', 'A', 'C', 'B'],
    'target': [1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# One-hot encoding
one_hot = pd.get_dummies(df['color'], prefix='color')

# Label encoding
le = LabelEncoder()
df['brand_encoded'] = le.fit_transform(df['brand'])

# Ordinal encoding (for ordered categories)
size_mapping = {'S': 1, 'M': 2, 'L': 3, 'XL': 4}
df['size_encoded'] = df['size'].map(size_mapping)

# Target encoding
te = TargetEncoder()
df['brand_target_encoded'] = te.fit_transform(df['brand'], df['target'])

print("Original Data:\n", df)
print("\nOne-hot Encoded:\n", one_hot)
print("\nFinal Encoded DataFrame:\n", df)
```

Slide 9: Feature Selection and Dimensionality Reduction

Feature selection is crucial for building efficient machine learning models. This slide demonstrates various techniques for selecting the most relevant features using Pandas and scikit-learn.

```python
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create sample dataset
np.random.seed(42)
n_samples = 1000
n_features = 20

X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                columns=[f'feature_{i}' for i in range(n_features)])
y = (X['feature_0'] * 2 + X['feature_1'] - 3 * X['feature_2'] > 0).astype(int)

# Statistical feature selection
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()

# PCA for dimensionality reduction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

print("Original features:", X.columns.tolist())
print("\nSelected features:", selected_features)
print("\nVariance explained by PCA components:", 
      pca.explained_variance_ratio_.round(3))
```

Slide 10: Time Series Feature Engineering

Time series feature engineering is essential for predictive modeling with temporal data. This implementation shows how to create advanced time-based features using Pandas.

```python
import pandas as pd
import numpy as np

# Create sample time series data
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
data = {
    'date': dates,
    'value': np.random.normal(100, 10, 365) + \
            np.sin(np.arange(365) * 2 * np.pi / 365) * 20
}
df = pd.DataFrame(data)
df.set_index('date', inplace=True)

# Create time-based features
df['year'] = df.index.year
df['month'] = df.index.month
df['day_of_week'] = df.index.dayofweek
df['quarter'] = df.index.quarter
df['is_weekend'] = df.index.weekday.isin([5, 6]).astype(int)

# Create lagged features
for i in [1, 7, 30]:
    df[f'lag_{i}'] = df['value'].shift(i)

# Create rolling statistics
for window in [7, 30]:
    df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
    df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()

# Create seasonal features
df['year_day'] = df.index.dayofyear
df['season'] = pd.cut(df['year_day'], 
                     bins=[0, 90, 180, 270, 366], 
                     labels=['Winter', 'Spring', 'Summer', 'Fall'])

print("Time Series Features:\n", df.head())
print("\nFeature Columns:", df.columns.tolist())
```

Slide 11: Real-World Application - Customer Churn Analysis

Let's implement a complete customer churn prediction pipeline using Pandas for data preparation and feature engineering, demonstrating practical application of the concepts covered.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Create synthetic customer data
np.random.seed(42)
n_customers = 1000

data = {
    'customer_id': range(n_customers),
    'tenure_months': np.random.randint(1, 72, n_customers),
    'monthly_charges': np.random.normal(70, 30, n_customers),
    'total_charges': np.random.normal(1000, 500, n_customers),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
    'payment_method': np.random.choice(['Electronic check', 'Credit card', 'Bank transfer'], n_customers),
    'technical_support_calls': np.random.poisson(2, n_customers),
    'churn': np.random.choice([0, 1], n_customers, p=[0.85, 0.15])
}

df = pd.DataFrame(data)

# Feature Engineering
df['average_monthly_charges'] = df['total_charges'] / df['tenure_months']
df['support_calls_per_month'] = df['technical_support_calls'] / df['tenure_months']

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['contract_type', 'payment_method'])

# Prepare features and target
features = df_encoded.drop(['customer_id', 'churn'], axis=1)
target = df_encoded['churn']

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate
predictions = rf_model.predict(X_test_scaled)
print("Model Performance:\n", classification_report(y_test, predictions))
```

Slide 12: Advanced Data Visualization with Pandas

Data visualization is crucial for understanding patterns and communicating insights. This implementation showcases advanced visualization techniques using Pandas with Matplotlib and Seaborn.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample dataset
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
data = {
    'date': dates,
    'sales': np.random.normal(1000, 100, len(dates)) + \
            np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 200,
    'category': np.random.choice(['A', 'B', 'C'], len(dates)),
    'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates))
}
df = pd.DataFrame(data)

# Create multiple visualizations
plt.figure(figsize=(15, 10))

# Time series plot
plt.subplot(2, 2, 1)
df.groupby('date')['sales'].mean().plot(title='Daily Sales Trend')

# Box plot
plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='category', y='sales')
plt.title('Sales Distribution by Category')

# Heatmap of sales by region and category
plt.subplot(2, 2, 3)
pivot_table = pd.pivot_table(df, values='sales', 
                           index='region', columns='category', 
                           aggfunc='mean')
sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd')
plt.title('Average Sales by Region and Category')

# Bar plot
plt.subplot(2, 2, 4)
df.groupby('region')['sales'].mean().plot(kind='bar')
plt.title('Average Sales by Region')

plt.tight_layout()
plt.show()
```

Slide 13: Real-World Application - Financial Data Analysis

This implementation demonstrates a comprehensive financial analysis pipeline using Pandas, including calculation of key financial metrics and risk indicators.

```python
import pandas as pd
import numpy as np
from scipy.stats import norm

# Create sample stock data
dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
np.random.seed(42)

data = {
    'date': dates,
    'stock_price': 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.02),
    'volume': np.random.randint(1000000, 5000000, len(dates)),
    'market_index': 1000 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
}
df = pd.DataFrame(data)

# Calculate financial metrics
df['returns'] = df['stock_price'].pct_change()
df['market_returns'] = df['market_index'].pct_change()

# Calculate moving averages
df['MA50'] = df['stock_price'].rolling(window=50).mean()
df['MA200'] = df['stock_price'].rolling(window=200).mean()

# Calculate volatility
df['volatility'] = df['returns'].rolling(window=21).std() * np.sqrt(252)

# Calculate beta
def calculate_beta(returns, market_returns, window=252):
    covariance = returns.rolling(window=window).cov(market_returns)
    market_variance = market_returns.rolling(window=window).var()
    return covariance / market_variance

df['beta'] = calculate_beta(df['returns'], df['market_returns'])

# Calculate Value at Risk (VaR)
confidence_level = 0.95
df['VaR'] = -norm.ppf(1 - confidence_level) * \
            df['returns'].rolling(window=252).std() * \
            df['stock_price']

print("Financial Metrics:\n", df.head())
print("\nSummary Statistics:")
print(df[['returns', 'volatility', 'beta', 'VaR']].describe())
```

Slide 14: Optimizing Pandas Performance

Understanding performance optimization techniques is crucial when working with large datasets. This implementation demonstrates various methods to improve Pandas operations efficiency.

```python
import pandas as pd
import numpy as np
from time import time

# Create large dataset
n_rows = 1000000
n_cols = 10

# Memory-efficient datatypes
df = pd.DataFrame({
    'id': np.arange(n_rows, dtype=np.int32),
    'float_col': np.random.randn(n_rows).astype(np.float32),
    'category_col': np.random.choice(['A', 'B', 'C'], n_rows),
    'date_col': pd.date_range('2023-01-01', periods=n_rows)
})

# Optimize memory usage
def optimize_dtypes(df):
    start_mem = df.memory_usage().sum() / 1024**2
    
    # Convert integers to smallest possible int type
    for col in df.select_dtypes(include=['int']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Convert floats to float32
    for col in df.select_dtypes(include=['float']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert object columns to category when beneficial
    for col in df.select_dtypes(include=['object']):
        if df[col].nunique() / len(df) < 0.5:  # If fewer than 50% unique values
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB')
    return df

# Demonstrate chunking for large operations
def process_in_chunks(df, chunk_size=100000):
    results = []
    for start in range(0, len(df), chunk_size):
        chunk = df[start:start + chunk_size]
        # Process chunk
        result = chunk['float_col'].mean()
        results.append(result)
    return np.mean(results)

# Demonstrate performance comparison
def performance_comparison(df):
    # Regular operation
    start = time()
    result1 = df.groupby('category_col')['float_col'].mean()
    time1 = time() - start
    
    # Optimized operation using categorical
    df['category_col'] = df['category_col'].astype('category')
    start = time()
    result2 = df.groupby('category_col')['float_col'].mean()
    time2 = time() - start
    
    print(f"Regular operation time: {time1:.2f} seconds")
    print(f"Optimized operation time: {time2:.2f} seconds")

# Run optimizations
df_optimized = optimize_dtypes(df)
performance_comparison(df_optimized)
```

Slide 15: Additional Resources

*   ArXiv Papers:

*   "Deep Learning with Pandas: A Comprehensive Review" - [https://arxiv.org/abs/2201.00046](https://arxiv.org/abs/2201.00046)
*   "Efficient Data Processing Techniques for Large-Scale Machine Learning" - [https://arxiv.org/abs/2105.00234](https://arxiv.org/abs/2105.00234)
*   "Modern Approaches to Time Series Analysis with Pandas" - [https://arxiv.org/abs/2203.00789](https://arxiv.org/abs/2203.00789)
*   "Feature Engineering Strategies for Machine Learning: A Survey" - [https://arxiv.org/abs/2204.00912](https://arxiv.org/abs/2204.00912)

Suggested Resources:

*   Python Data Science Handbook: [https://jakevdp.github.io/PythonDataScienceHandbook/](https://jakevdp.github.io/PythonDataScienceHandbook/)
*   Pandas Documentation: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
*   Real Python Pandas Tutorials: [https://realpython.com/pandas-python-explore-dataset/](https://realpython.com/pandas-python-explore-dataset/)

