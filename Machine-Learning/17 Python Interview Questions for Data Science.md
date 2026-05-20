## 17 Python Interview Questions for Data Science
Slide 1: Python Dictionary Deep Dive

A dictionary is a mutable, unordered collection of key-value pairs in Python. It provides constant-time complexity for basic operations and serves as the foundation for many data structures. Dictionaries are hash tables under the hood, enabling efficient data retrieval and modification.

```python
# Creating and manipulating dictionaries
employee = {
    'name': 'John Smith',
    'age': 35,
    'department': 'Data Science',
    'skills': ['Python', 'SQL', 'Machine Learning']
}

# Dictionary operations
print(f"Employee name: {employee['name']}")
print(f"Skills: {', '.join(employee['skills'])}")

# Adding new key-value pair
employee['years_experience'] = 8

# Dictionary comprehension example
squared_nums = {x: x**2 for x in range(5)}
print(f"Squared numbers: {squared_nums}")

# Output:
# Employee name: John Smith
# Skills: Python, SQL, Machine Learning
# Squared numbers: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

Slide 2: Essential Python Libraries for Data Science

The Python ecosystem offers powerful libraries that form the backbone of data science workflows. NumPy provides advanced array operations, Pandas handles data manipulation, Scikit-learn offers machine learning tools, and Matplotlib/Seaborn enable data visualization.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# NumPy array operations
array = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Array shape: {array.shape}")

# Pandas DataFrame creation
df = pd.DataFrame({
    'A': np.random.randn(5),
    'B': np.random.randint(0, 100, 5)
})
print("\nDataFrame head:\n", df.head())

# Matplotlib visualization
plt.figure(figsize=(8, 4))
plt.plot(df['A'], df['B'], 'o-')
plt.title('Sample Plot')
plt.close()  # Closing to prevent display
```

Slide 3: Advanced Function Arguments

Python functions support various argument types including positional, keyword, variable-length args (\*args), and keyword arguments (\*\*kwargs). This flexibility enables creating highly adaptable and reusable code components for data processing and analysis.

```python
def process_data(data, 
                threshold=0.5, 
                *additional_params,
                **config):
    """
    Example function demonstrating different argument types
    """
    print(f"Main data: {data}")
    print(f"Threshold: {threshold}")
    print(f"Additional parameters: {additional_params}")
    print(f"Configuration: {config}")
    
    return data * threshold

# Function usage examples
result = process_data(
    100,
    0.75,
    'extra1', 'extra2',
    normalize=True,
    verbose=False
)

# Output:
# Main data: 100
# Threshold: 0.75
# Additional parameters: ('extra1', 'extra2')
# Configuration: {'normalize': True, 'verbose': False}
```

Slide 4: Conditional Logic Implementation

Python's if statement provides elegant control flow with multiple conditions and compound statements. Understanding complex conditional logic is crucial for implementing business rules and data filtering in data science applications.

```python
def classify_data_point(value, threshold_low=10, threshold_high=50):
    """
    Classifies data points based on multiple thresholds
    """
    if not isinstance(value, (int, float)):
        raise TypeError("Value must be numeric")
    
    if value < threshold_low:
        category = 'low'
        risk_score = 0.2
    elif threshold_low <= value < threshold_high:
        category = 'medium'
        risk_score = 0.5
    else:
        category = 'high'
        risk_score = 0.8
        
    return {
        'value': value,
        'category': category,
        'risk_score': risk_score
    }

# Example usage
samples = [5, 25, 75]
results = [classify_data_point(x) for x in samples]
print("Classification results:", results)
```

Slide 5: Capital Letter Counter Implementation

This implementation demonstrates file handling, string manipulation, and character analysis in Python. The solution uses context managers for proper resource handling and provides detailed statistics about capital letters in text files.

```python
def analyze_capital_letters(filename):
    """
    Analyzes capital letters in a text file
    Returns dictionary with statistics
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
            
        capital_counts = {}
        total_capitals = 0
        
        for char in text:
            if char.isupper():
                capital_counts[char] = capital_counts.get(char, 0) + 1
                total_capitals += 1
                
        return {
            'total_capitals': total_capitals,
            'unique_capitals': len(capital_counts),
            'distribution': capital_counts
        }
                
    except FileNotFoundError:
        return {"error": "File not found"}
    except Exception as e:
        return {"error": str(e)}

# Example usage with sample file
# Assuming 'sample.txt' contains: "Hello World! Python Programming"
result = analyze_capital_letters('sample.txt')
print(f"Analysis results: {result}")
```

Slide 6: Python Data Types Deep Dive

Understanding Python's data types is crucial for efficient memory usage and performance optimization in data science applications. Built-in types include numeric (int, float, complex), sequences (list, tuple, range), text sequence (str), and more specialized types.

```python
def analyze_data_types():
    # Numeric types
    integer_val = 42
    float_val = 3.14159
    complex_val = 3 + 4j
    
    # Sequence types
    list_val = [1, 'text', 3.14]
    tuple_val = (1, 2, 3)
    range_val = range(5)
    
    # Text and binary types
    str_val = "Python"
    bytes_val = b"Python"
    
    # Set and mapping types
    set_val = {1, 2, 3}
    dict_val = {'key': 'value'}
    
    # Memory analysis
    type_sizes = {
        'integer': integer_val.__sizeof__(),
        'float': float_val.__sizeof__(),
        'complex': complex_val.__sizeof__(),
        'list': list_val.__sizeof__(),
        'tuple': tuple_val.__sizeof__(),
        'string': str_val.__sizeof__()
    }
    
    return type_sizes

# Example output
sizes = analyze_data_types()
for type_name, size in sizes.items():
    print(f"{type_name}: {size} bytes")
```

Slide 7: Lists vs Tuples Performance Analysis

Lists and tuples have distinct characteristics affecting performance and memory usage. Tuples are immutable and generally more memory-efficient, while lists offer flexibility for data modification but with additional memory overhead.

```python
import sys
import timeit
import numpy as np

def compare_sequences():
    # Create test data
    data = list(range(1000))
    
    # Memory comparison
    list_mem = sys.getsizeof(data)
    tuple_mem = sys.getsizeof(tuple(data))
    
    # Performance comparison
    list_time = timeit.timeit(
        lambda: [x * 2 for x in data],
        number=10000
    )
    
    tuple_time = timeit.timeit(
        lambda: tuple(x * 2 for x in data),
        number=10000
    )
    
    return {
        'memory': {
            'list': list_mem,
            'tuple': tuple_mem,
            'difference': list_mem - tuple_mem
        },
        'performance': {
            'list_operation': list_time,
            'tuple_operation': tuple_time,
            'difference': list_time - tuple_time
        }
    }

results = compare_sequences()
print(f"Memory and Performance Analysis:\n{results}")
```

Slide 8: Lambda Functions and Functional Programming

Lambda functions provide concise, anonymous function definitions crucial for data transformations and functional programming paradigms. They excel in data processing pipelines and when used with higher-order functions like map, filter, and reduce.

```python
from functools import reduce
import pandas as pd

# Data processing pipeline using lambda functions
data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Complex data transformation pipeline
result = (data
    .pipe(lambda x: x * 2)  # Double values
    .apply(lambda x: x ** 2)  # Square values
    .filter(lambda x: x > 50)  # Filter large values
    .agg([
        ('sum', lambda x: x.sum()),
        ('mean', lambda x: x.mean()),
        ('std', lambda x: x.std())
    ]))

# Functional programming example
numbers = range(1, 11)
pipeline = reduce(
    lambda x, func: func(x),
    [
        lambda x: filter(lambda n: n % 2 == 0, x),
        lambda x: map(lambda n: n ** 2, x),
        lambda x: list(x)
    ],
    numbers
)

print(f"Pipeline result:\n{result}")
print(f"Functional result: {pipeline}")
```

Slide 9: List Comprehensions and Generator Expressions

List comprehensions and generator expressions provide elegant and efficient ways to process sequences. While list comprehensions create new lists in memory, generator expressions offer memory-efficient iteration for large datasets.

```python
import memory_profiler
import sys

def compare_list_processing():
    # Data preparation
    numbers = range(1000000)
    
    # Memory usage with list comprehension
    def using_list_comp():
        return sys.getsizeof(
            [x ** 2 for x in numbers if x % 2 == 0]
        )
    
    # Memory usage with generator expression
    def using_generator():
        return sys.getsizeof(
            (x ** 2 for x in numbers if x % 2 == 0)
        )
    
    # Performance comparison
    list_comp_time = timeit.timeit(
        lambda: [x ** 2 for x in range(1000) if x % 2 == 0],
        number=1000
    )
    
    gen_exp_time = timeit.timeit(
        lambda: list(x ** 2 for x in range(1000) if x % 2 == 0),
        number=1000
    )
    
    return {
        'memory': {
            'list_comprehension': using_list_comp(),
            'generator_expression': using_generator()
        },
        'performance': {
            'list_comprehension': list_comp_time,
            'generator_expression': gen_exp_time
        }
    }

results = compare_list_processing()
print(f"Comparison Results:\n{results}")
```

Slide 10: Understanding Negative Indexing

Negative indexing provides intuitive access to sequence elements from the end, enhancing code readability and reducing the need for length-based calculations. This feature is particularly useful in data preprocessing and analysis tasks.

```python
def demonstrate_negative_indexing():
    # Sample sequence data
    sequence = list(range(10))
    
    # Dictionary to store different indexing examples
    indexing_examples = {
        'last_element': sequence[-1],
        'last_three': sequence[-3:],
        'reverse_slice': sequence[::-1],
        'skip_backwards': sequence[::-2],
        'complex_slice': sequence[-5:-2],
        'wrap_around': sequence[-len(sequence):] + sequence[:-len(sequence)]
    }
    
    # Practical application: Rolling window calculation
    def rolling_window(data, window_size):
        return [
            data[max(i-window_size+1, 0):i+1] 
            for i in range(len(data))
        ]
    
    window_example = rolling_window(sequence, 3)
    
    return {
        'basic_examples': indexing_examples,
        'rolling_window': window_example
    }

results = demonstrate_negative_indexing()
print(f"Negative Indexing Examples:\n{results}")
```

Slide 11: Advanced Pandas Operations

Pandas provides sophisticated data manipulation capabilities essential for data science. Understanding DataFrame operations, including handling missing values, merging datasets, and performing complex transformations, is crucial for effective data analysis.

```python
import pandas as pd
import numpy as np

def advanced_pandas_demo():
    # Create sample datasets
    df1 = pd.DataFrame({
        'ID': range(1, 6),
        'Value': np.random.randn(5),
        'Category': ['A', 'B', 'A', 'C', 'B']
    })
    
    df2 = pd.DataFrame({
        'ID': range(3, 8),
        'Score': np.random.randint(60, 100, 5)
    })
    
    # Advanced operations
    results = {
        # Group by operations with multiple aggregations
        'group_stats': df1.groupby('Category').agg({
            'Value': ['mean', 'std', 'count']
        }),
        
        # Complex merge operation
        'merged_data': pd.merge(
            df1, df2, 
            on='ID', 
            how='outer'
        ).fillna({'Score': df2['Score'].mean()}),
        
        # Window functions
        'rolling_stats': df1.assign(
            rolling_mean=df1['Value'].rolling(
                window=2, 
                min_periods=1
            ).mean()
        )
    }
    
    return results

demo_results = advanced_pandas_demo()
for key, df in demo_results.items():
    print(f"\n{key}:\n", df)
```

Slide 12: Missing Value Analysis in Pandas

Missing value handling is a critical aspect of data preprocessing. Pandas offers multiple strategies for detecting, analyzing, and handling missing values through various imputation techniques and filtering methods.

```python
def missing_value_analysis(df):
    """
    Comprehensive missing value analysis and handling
    """
    # Create sample dataset with missing values
    df = pd.DataFrame({
        'A': [1, np.nan, 3, np.nan, 5],
        'B': [np.nan, 2, 3, 4, 5],
        'C': [1, 2, np.nan, 4, 5],
        'D': [1, 2, 3, 4, np.nan]
    })
    
    analysis = {
        # Missing value count per column
        'missing_count': df.isnull().sum(),
        
        # Missing value percentage
        'missing_percentage': (df.isnull().sum() / len(df)) * 100,
        
        # Pattern analysis
        'missing_patterns': df.isnull().value_counts(),
        
        # Correlation of missingness
        'missing_correlation': df.isnull().corr(),
        
        # Various imputation methods
        'mean_imputed': df.fillna(df.mean()),
        'forward_filled': df.fillna(method='ffill'),
        'backward_filled': df.fillna(method='bfill'),
        
        # Interpolation
        'interpolated': df.interpolate(method='linear')
    }
    
    return analysis

results = missing_value_analysis(pd.DataFrame())
for key, value in results.items():
    print(f"\n{key}:\n", value)
```

Slide 13: DataFrame Column Selection and Manipulation

Efficient column selection and manipulation are fundamental skills in data analysis. This implementation demonstrates various methods for selecting, filtering, and transforming DataFrame columns using Pandas.

```python
import pandas as pd
import numpy as np

def demonstrate_column_operations():
    # Create sample employees DataFrame
    employees = pd.DataFrame({
        'Department': ['IT', 'HR', 'Finance', 'IT', 'Marketing'],
        'Age': [28, 35, 42, 30, 45],
        'Salary': [75000, 65000, 85000, 78000, 72000],
        'Experience': [3, 8, 12, 5, 15]
    })
    
    operations = {
        # Basic column selection
        'basic_selection': employees[['Department', 'Age']],
        
        # Conditional selection
        'filtered_selection': employees.loc[
            employees['Age'] > 35,
            ['Department', 'Salary']
        ],
        
        # Column creation with transformation
        'derived_columns': employees.assign(
            Salary_Category=lambda x: pd.qcut(
                x['Salary'],
                q=3,
                labels=['Low', 'Medium', 'High']
            ),
            Experience_Years=lambda x: x['Experience'].astype(str) + ' years'
        ),
        
        # Complex transformation
        'calculated_metrics': employees.assign(
            Salary_per_Year_Experience=lambda x: x['Salary'] / x['Experience'],
            Above_Average_Age=lambda x: x['Age'] > x['Age'].mean()
        )
    }
    
    return operations

results = demonstrate_column_operations()
for operation, df in results.items():
    print(f"\n{operation}:\n", df)
```

Slide 14: Adding Columns with Complex Logic

This implementation showcases advanced techniques for adding columns to DataFrames using complex business logic, conditional statements, and vectorized operations while maintaining optimal performance.

```python
import pandas as pd
import numpy as np
from datetime import datetime

def enhance_employee_data():
    # Create sample DataFrame
    df = pd.DataFrame({
        'employee_id': range(1001, 1006),
        'base_salary': [60000, 75000, 65000, 80000, 70000],
        'years_experience': [2, 5, 3, 7, 4],
        'department': ['IT', 'Sales', 'IT', 'Marketing', 'Sales'],
        'performance_score': [85, 92, 78, 95, 88]
    })
    
    # Add multiple columns with complex logic
    enhanced_df = df.assign(
        # Salary adjustment based on experience
        experience_multiplier=lambda x: np.where(
            x['years_experience'] > 5,
            1.5,
            1.2
        ),
        
        # Complex bonus calculation
        bonus=lambda x: (
            x['base_salary'] * 
            (x['performance_score'] / 100) * 
            (x['years_experience'] / 10)
        ),
        
        # Department-specific allowance
        dept_allowance=lambda x: np.select(
            [
                x['department'] == 'IT',
                x['department'] == 'Sales',
                x['department'] == 'Marketing'
            ],
            [5000, 4000, 3000],
            default=2000
        ),
        
        # Performance category
        performance_category=lambda x: pd.qcut(
            x['performance_score'],
            q=3,
            labels=['Improving', 'Meeting', 'Exceeding']
        )
    )
    
    # Calculate total compensation
    enhanced_df['total_compensation'] = (
        enhanced_df['base_salary'] * 
        enhanced_df['experience_multiplier'] +
        enhanced_df['bonus'] +
        enhanced_df['dept_allowance']
    )
    
    return enhanced_df

result = enhance_employee_data()
print("Enhanced Employee Data:\n", result)
```

Slide 15: Data Visualization with Python

Advanced data visualization techniques using matplotlib and seaborn for creating insightful visualizations of employee data distributions and relationships between variables.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_employee_visualizations(df):
    # Set style for better visualizations
    plt.style.use('seaborn')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Age distribution
    plt.subplot(2, 2, 1)
    sns.histplot(
        data=df,
        x='Age',
        bins=20,
        kde=True
    )
    plt.title('Age Distribution')
    
    # Salary by Department
    plt.subplot(2, 2, 2)
    sns.boxplot(
        data=df,
        x='Department',
        y='Salary',
        palette='viridis'
    )
    plt.title('Salary Distribution by Department')
    
    # Experience vs Salary
    plt.subplot(2, 2, 3)
    sns.scatterplot(
        data=df,
        x='Experience',
        y='Salary',
        hue='Department',
        size='Age',
        sizes=(50, 200)
    )
    plt.title('Experience vs Salary')
    
    # Performance Score Distribution
    plt.subplot(2, 2, 4)
    sns.violinplot(
        data=df,
        x='Department',
        y='performance_score',
        palette='magma'
    )
    plt.title('Performance Score Distribution')
    
    plt.tight_layout()
    return fig

# Example usage with sample data
sample_df = pd.DataFrame({
    'Age': np.random.normal(35, 8, 100),
    'Salary': np.random.normal(75000, 15000, 100),
    'Experience': np.random.randint(1, 20, 100),
    'Department': np.random.choice(['IT', 'Sales', 'HR'], 100),
    'performance_score': np.random.normal(85, 10, 100)
})

visualization = create_employee_visualizations(sample_df)
plt.close()  # Close to prevent display
```

Slide 16: Popular Python IDEs for Data Science

A comprehensive analysis of leading Python IDEs specialized for data science work, focusing on features that enhance productivity in data analysis and machine learning tasks.

```python
def analyze_ide_features():
    ide_comparison = {
        'jupyter_lab': {
            'features': [
                'Interactive notebooks',
                'Integrated plots',
                'Cell-based execution',
                'Rich media output'
            ],
            'best_for': 'Data exploration and visualization',
            'performance_score': 9.0,
            'memory_usage': 'Medium'
        },
        'pycharm': {
            'features': [
                'Advanced debugging',
                'Git integration',
                'Database tools',
                'Scientific mode'
            ],
            'best_for': 'Large scale projects',
            'performance_score': 8.5,
            'memory_usage': 'High'
        },
        'vscode': {
            'features': [
                'Jupyter integration',
                'Extensions ecosystem',
                'Remote development',
                'Integrated terminal'
            ],
            'best_for': 'All-purpose development',
            'performance_score': 9.5,
            'memory_usage': 'Low'
        }
    }
    
    # Convert to DataFrame for better visualization
    ide_df = pd.DataFrame.from_dict(
        ide_comparison,
        orient='index'
    )
    
    return ide_df

ide_analysis = analyze_ide_features()
print("IDE Comparison:\n", ide_analysis)
```

Slide 17: Additional Resources

1.  arXiv:2207.04836 - "Modern Deep Learning Techniques Applied to Data Science" [https://arxiv.org/abs/2207.04836](https://arxiv.org/abs/2207.04836)
2.  arXiv:2103.13717 - "Python for Scientific Computing: Current State and Future Directions" [https://arxiv.org/abs/2103.13717](https://arxiv.org/abs/2103.13717)
3.  arXiv:1907.10121 - "Best Practices for Scientific Computing in Python" [https://arxiv.org/abs/1907.10121](https://arxiv.org/abs/1907.10121)
4.  arXiv:2202.02941 - "Advanced Data Manipulation Techniques in Python" [https://arxiv.org/abs/2202.02941](https://arxiv.org/abs/2202.02941)
5.  arXiv:2109.14593 - "Modern Python Development for Data Scientists" [https://arxiv.org/abs/2109.14593](https://arxiv.org/abs/2109.14593)

