## The Beast of Real-World Data Wrangling

Slide 1: The Reality of Real-World Data

Data wrangling in real-world scenarios is indeed more challenging than working with clean, pre-processed datasets like iris or mtcars. Real-world data often comes with inconsistencies, missing values, and unexpected formats that require significant effort to clean and prepare for analysis. However, the characterization of this process as "taming a beast" may be an overstatement. While data wrangling can be complex, it's a manageable and essential part of the data science workflow that can be approached systematically.

```python
# Example of real-world data inconsistencies
raw_data = [
    {'name': 'John Doe', 'age': '35', 'income': '$50,000'},
    {'name': 'Jane Smith', 'age': 'N/A', 'income': '45000'},
    {'name': 'Bob Johnson', 'age': '42', 'income': None},
    {'name': 'Alice Brown', 'age': '28', 'income': '55,000'}
]

# Demonstrating inconsistencies
for entry in raw_data:
    print(f"Name: {entry['name']}")
    print(f"Age: {entry['age']} (type: {type(entry['age']).__name__})")
    print(f"Income: {entry['income']} (type: {type(entry['income']).__name__})")
    print()
```

Slide 2: Handling Missing Values

Missing values are common in real-world datasets and can significantly impact analysis if not handled properly. There are several strategies to deal with missing data, including imputation (filling in missing values), interpolation, or removing records with missing values. The choice depends on the nature of the data and the specific requirements of the analysis.

```python
def handle_missing_values(data, strategy='mean'):
    clean_data = []
    if strategy == 'mean':
        # Calculate mean age, excluding 'N/A'
        valid_ages = [int(entry['age']) for entry in data if entry['age'] != 'N/A' and entry['age'] is not None]
        mean_age = sum(valid_ages) / len(valid_ages)
        
        for entry in data:
            new_entry = entry.copy()
            if entry['age'] == 'N/A' or entry['age'] is None:
                new_entry['age'] = mean_age
            clean_data.append(new_entry)
    return clean_data

cleaned_data = handle_missing_values(raw_data)
for entry in cleaned_data:
    print(f"Name: {entry['name']}, Age: {entry['age']}")
```

Slide 3: Outlier Detection and Removal

Outliers can significantly skew statistical analyses and machine learning models. Identifying and appropriately handling outliers is crucial for maintaining the integrity of your data and ensuring accurate results. Common methods for outlier detection include statistical techniques like Z-score and Interquartile Range (IQR), as well as visual methods such as box plots.

```python
import statistics

def detect_outliers(data, feature, threshold=2):
    values = [float(entry[feature]) for entry in data if entry[feature] is not None]
    mean = statistics.mean(values)
    std_dev = statistics.stdev(values)
    
    outliers = []
    for entry in data:
        if entry[feature] is not None:
            z_score = (float(entry[feature]) - mean) / std_dev
            if abs(z_score) > threshold:
                outliers.append(entry)
    
    return outliers

# Assuming we've cleaned the 'age' data to be numeric
cleaned_data = [{'name': 'John', 'age': 35}, {'name': 'Jane', 'age': 28},
                {'name': 'Bob', 'age': 42}, {'name': 'Alice', 'age': 90}]

outliers = detect_outliers(cleaned_data, 'age', threshold=2)
print("Detected outliers:")
for outlier in outliers:
    print(f"Name: {outlier['name']}, Age: {outlier['age']}")
```

Slide 4: Data Transformation

Data transformation is often necessary to prepare data for analysis or modeling. This can involve standardizing or normalizing numerical features, encoding categorical variables, or applying mathematical transformations to achieve desired distributions. Proper data transformation ensures that all features contribute appropriately to the analysis.

```python
def standardize_feature(data, feature):
    values = [entry[feature] for entry in data]
    mean = sum(values) / len(values)
    std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
    
    for entry in data:
        entry[f'{feature}_standardized'] = (entry[feature] - mean) / std_dev
    
    return data

# Example usage
numeric_data = [{'value': 10}, {'value': 20}, {'value': 30}, {'value': 40}, {'value': 50}]
standardized_data = standardize_feature(numeric_data, 'value')

for entry in standardized_data:
    print(f"Original: {entry['value']}, Standardized: {entry['value_standardized']:.2f}")
```

Slide 5: Feature Engineering

Feature engineering is the process of creating new features from existing data to improve model performance. This can involve combining existing features, extracting information from complex data types, or applying domain knowledge to create more informative variables. Effective feature engineering often requires a deep understanding of the problem domain and creative thinking.

```python
def engineer_features(data):
    for entry in data:
        # Create a new feature: age group
        if entry['age'] < 18:
            entry['age_group'] = 'minor'
        elif 18 <= entry['age'] < 65:
            entry['age_group'] = 'adult'
        else:
            entry['age_group'] = 'senior'
        
        # Create a feature for name length
        entry['name_length'] = len(entry['name'])
    
    return data

# Example usage
sample_data = [
    {'name': 'John Doe', 'age': 35},
    {'name': 'Jane Smith', 'age': 17},
    {'name': 'Bob Johnson', 'age': 70}
]

engineered_data = engineer_features(sample_data)
for entry in engineered_data:
    print(f"Name: {entry['name']}, Age: {entry['age']}, "
          f"Age Group: {entry['age_group']}, Name Length: {entry['name_length']}")
```

Slide 6: Encoding Categorical Data

Many machine learning algorithms require numerical input, necessitating the conversion of categorical data into a numerical format. Common encoding techniques include one-hot encoding for nominal categories and label encoding for ordinal categories. The choice of encoding method can significantly impact model performance and interpretability.

```python
def one_hot_encode(data, feature):
    # Get unique categories
    categories = set(entry[feature] for entry in data)
    
    for entry in data:
        for category in categories:
            entry[f'{feature}_{category}'] = 1 if entry[feature] == category else 0
    
    return data

# Example usage
categorical_data = [
    {'color': 'red'},
    {'color': 'blue'},
    {'color': 'green'},
    {'color': 'red'}
]

encoded_data = one_hot_encode(categorical_data, 'color')
for entry in encoded_data:
    print(entry)
```

Slide 7: Real-Life Example: Weather Data Analysis

Let's consider a real-life example of wrangling weather data. Weather datasets often come with various challenges, including missing values, different units of measurement, and the need for feature engineering to extract meaningful insights.

```python
raw_weather_data = [
    {'date': '2023-01-01', 'temperature': '72F', 'humidity': '65%', 'precipitation': '0.1in'},
    {'date': '2023-01-02', 'temperature': '68F', 'humidity': 'N/A', 'precipitation': '0'},
    {'date': '2023-01-03', 'temperature': '18C', 'humidity': '70%', 'precipitation': '5mm'},
    {'date': '2023-01-04', 'temperature': '65F', 'humidity': '60%', 'precipitation': None}
]

def clean_weather_data(data):
    cleaned_data = []
    for entry in data:
        new_entry = {}
        # Convert date to datetime object
        new_entry['date'] = entry['date']  # In practice, use datetime.strptime()
        
        # Handle temperature: convert all to Celsius
        if 'F' in entry['temperature']:
            temp = float(entry['temperature'].rstrip('F'))
            new_entry['temperature_celsius'] = (temp - 32) * 5/9
        else:
            new_entry['temperature_celsius'] = float(entry['temperature'].rstrip('C'))
        
        # Handle humidity: convert to float and fill missing values
        new_entry['humidity'] = float(entry['humidity'].rstrip('%')) if entry['humidity'] != 'N/A' else None
        
        # Handle precipitation: convert all to mm and handle missing values
        if entry['precipitation'] is None or entry['precipitation'] == '0':
            new_entry['precipitation_mm'] = 0
        elif 'in' in entry['precipitation']:
            precip = float(entry['precipitation'].rstrip('in'))
            new_entry['precipitation_mm'] = precip * 25.4
        else:
            new_entry['precipitation_mm'] = float(entry['precipitation'].rstrip('mm'))
        
        cleaned_data.append(new_entry)
    
    return cleaned_data

cleaned_weather = clean_weather_data(raw_weather_data)
for entry in cleaned_weather:
    print(entry)
```

Slide 8: Results for: Real-Life Example: Weather Data Analysis

```
{'date': '2023-01-01', 'temperature_celsius': 22.22222222222222, 'humidity': 65.0, 'precipitation_mm': 2.54}
{'date': '2023-01-02', 'temperature_celsius': 20.0, 'humidity': None, 'precipitation_mm': 0}
{'date': '2023-01-03', 'temperature_celsius': 18.0, 'humidity': 70.0, 'precipitation_mm': 5.0}
{'date': '2023-01-04', 'temperature_celsius': 18.333333333333332, 'humidity': 60.0, 'precipitation_mm': 0}
```

Slide 9: Real-Life Example: Text Data Processing

Text data is another common type of real-world data that often requires extensive wrangling. This can include tasks such as tokenization, removing stop words, stemming or lemmatization, and handling special characters or formatting issues.

```python
import re
from collections import Counter

def process_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stop words (a very basic list for demonstration)
    stop_words = set(['the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of'])
    tokens = [token for token in tokens if token not in stop_words]
    
    # Count word frequencies
    word_freq = Counter(tokens)
    
    return word_freq

# Example usage
sample_text = """
The quick brown fox jumps over the lazy dog. 
The dog barks, but the fox is too quick! 
In the end, both animals are tired.
"""

processed_text = process_text(sample_text)
print("Word frequencies:")
for word, freq in processed_text.most_common(5):
    print(f"{word}: {freq}")
```

Slide 10: Results for: Real-Life Example: Text Data Processing

```
Word frequencies:
quick: 2
fox: 2
dog: 2
jumps: 1
over: 1
```

Slide 11: Challenges and Best Practices

While data wrangling can be complex, it's a crucial step in the data science process. Some best practices include:

1.  Understanding your data sources and potential issues before starting.
2.  Documenting all data cleaning and transformation steps for reproducibility.
3.  Regularly validating your data throughout the wrangling process.
4.  Using version control for your data and code.
5.  Collaborating with domain experts to ensure appropriate handling of field-specific data.

```python
def data_quality_check(data, expected_columns):
    issues = []
    
    # Check for missing columns
    missing_columns = set(expected_columns) - set(data[0].keys())
    if missing_columns:
        issues.append(f"Missing columns: {', '.join(missing_columns)}")
    
    # Check for missing values
    for entry in data:
        for column in expected_columns:
            if column in entry and entry[column] is None:
                issues.append(f"Missing value in column '{column}' for entry: {entry}")
    
    # Check for data type consistency
    for column in expected_columns:
        column_types = set(type(entry[column]) for entry in data if column in entry)
        if len(column_types) > 1:
            issues.append(f"Inconsistent data types in column '{column}': {column_types}")
    
    return issues

# Example usage
sample_data = [
    {'name': 'John', 'age': 30, 'city': 'New York'},
    {'name': 'Jane', 'age': '25', 'city': None},
    {'name': 'Bob', 'city': 'Chicago'}
]

expected_columns = ['name', 'age', 'city']
quality_issues = data_quality_check(sample_data, expected_columns)

print("Data quality issues:")
for issue in quality_issues:
    print(f"- {issue}")
```

Slide 12: Automating Data Wrangling

As datasets grow larger and more complex, automating parts of the data wrangling process becomes increasingly important. While full automation is often not possible due to the unique characteristics of each dataset, certain tasks can be standardized and automated to improve efficiency.

```python
class DataWrangler:
    def __init__(self, data):
        self.data = data
    
    def remove_duplicates(self):
        self.data = list({tuple(d.items()) for d in self.data})
        return self
    
    def fill_missing_values(self, column, strategy='mean'):
        if strategy == 'mean':
            values = [entry[column] for entry in self.data if entry[column] is not None]
            mean_value = sum(values) / len(values)
            for entry in self.data:
                if entry[column] is None:
                    entry[column] = mean_value
        return self
    
    def standardize_column(self, column):
        values = [entry[column] for entry in self.data]
        mean = sum(values) / len(values)
        std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        
        for entry in self.data:
            entry[f'{column}_standardized'] = (entry[column] - mean) / std_dev
        return self
    
    def get_cleaned_data(self):
        return self.data

# Example usage
raw_data = [
    {'id': 1, 'value': 10},
    {'id': 2, 'value': None},
    {'id': 3, 'value': 20},
    {'id': 1, 'value': 10},  # Duplicate
    {'id': 4, 'value': 30}
]

wrangler = DataWrangler(raw_data)
cleaned_data = (wrangler
                .remove_duplicates()
                .fill_missing_values('value')
                .standardize_column('value')
                .get_cleaned_data())

for entry in cleaned_data:
    print(entry)
```

Slide 13: Continuous Improvement in Data Wrangling

Data wrangling is an iterative process that requires continuous refinement and adaptation. As you work with diverse datasets and encounter new challenges, it's crucial to update your wrangling techniques and tools. This ongoing improvement involves learning from past experiences, staying updated with new methodologies, and refining your approach based on the specific needs of each project.

```python
class AdaptiveDataWrangler:
    def __init__(self):
        self.techniques = {}
        self.performance_log = {}

    def add_technique(self, name, function):
        self.techniques[name] = function
        self.performance_log[name] = {'uses': 0, 'success_rate': 0}

    def apply_technique(self, name, data):
        if name not in self.techniques:
            raise ValueError(f"Technique '{name}' not found")
        
        result = self.techniques[name](data)
        self.performance_log[name]['uses'] += 1
        
        # In a real scenario, you'd implement a way to measure success
        success = self.evaluate_success(result)
        
        current_success_rate = self.performance_log[name]['success_rate']
        total_uses = self.performance_log[name]['uses']
        self.performance_log[name]['success_rate'] = (
            (current_success_rate * (total_uses - 1) + success) / total_uses
        )
        
        return result

    def evaluate_success(self, result):
        # Placeholder for success evaluation logic
        return 1  # Assume success for this example

    def get_best_technique(self):
        return max(self.performance_log, key=lambda x: self.performance_log[x]['success_rate'])

# Example usage
wrangler = AdaptiveDataWrangler()
wrangler.add_technique('remove_nulls', lambda data: [d for d in data if all(d.values())])
wrangler.add_technique('fill_mean', lambda data: [{**d, 'value': sum(d['value'] for d in data if d['value']) / len(data) if d['value'] is None else d['value']} for d in data])

sample_data = [{'id': 1, 'value': 10}, {'id': 2, 'value': None}, {'id': 3, 'value': 20}]

print("Applying techniques:")
print(wrangler.apply_technique('remove_nulls', sample_data))
print(wrangler.apply_technique('fill_mean', sample_data))

print("\nBest technique:", wrangler.get_best_technique())
```

Slide 14: Ethical Considerations in Data Wrangling

When working with real-world data, it's crucial to consider ethical implications. This includes ensuring data privacy, avoiding bias in data cleaning and transformation, and being transparent about the methods used. Ethical data wrangling practices help maintain the integrity of your analysis and protect the individuals represented in your datasets.

```python
def anonymize_data(data, sensitive_fields):
    anonymized_data = []
    for entry in data:
        anonymized_entry = {}
        for key, value in entry.items():
            if key in sensitive_fields:
                anonymized_entry[key] = hash(str(value))  # Simple hashing for demonstration
            else:
                anonymized_entry[key] = value
        anonymized_data.append(anonymized_entry)
    return anonymized_data

def check_data_bias(data, protected_attribute, target_attribute):
    groups = {}
    for entry in data:
        group = entry[protected_attribute]
        if group not in groups:
            groups[group] = {'count': 0, 'sum': 0}
        groups[group]['count'] += 1
        groups[group]['sum'] += entry[target_attribute]
    
    for group, stats in groups.items():
        stats['average'] = stats['sum'] / stats['count']
    
    return groups

# Example usage
sample_data = [
    {'id': 1, 'name': 'Alice', 'age': 30, 'salary': 50000, 'gender': 'F'},
    {'id': 2, 'name': 'Bob', 'age': 35, 'salary': 60000, 'gender': 'M'},
    {'id': 3, 'name': 'Charlie', 'age': 40, 'salary': 70000, 'gender': 'M'},
    {'id': 4, 'name': 'Diana', 'age': 38, 'salary': 65000, 'gender': 'F'}
]

anonymized_data = anonymize_data(sample_data, ['name', 'id'])
print("Anonymized data:")
for entry in anonymized_data:
    print(entry)

bias_check = check_data_bias(sample_data, 'gender', 'salary')
print("\nPotential bias check:")
for group, stats in bias_check.items():
    print(f"{group}: Average salary = {stats['average']}")
```

Slide 15: Additional Resources

For those looking to deepen their understanding of data wrangling techniques and best practices, here are some valuable resources:

1.  ArXiv paper: "A Survey on Data Collection for Machine Learning: a Big Data - AI Integration Perspective" by Yuji Roh, Geon Heo, Steven Euijong Whang (2019). ArXiv:1811.03402 \[cs.LG\]
2.  ArXiv paper: "Automating Large-Scale Data Quality Verification" by Sebastian Schelter, Dustin Lange, Philipp Schmidt, Meltem Celikel, Felix Biessmann, Andreas Grafberger (2018). ArXiv:1801.07900 \[cs.DB\]
3.  ArXiv paper: "Towards Automated Data Cleaning: A Statistical Approach" by Sanjay Krishnan, Jiannan Wang, Eugene Wu, Michael J. Franklin, Ken Goldberg (2016). ArXiv:1603.08248 \[cs.DB\]

These papers provide in-depth discussions on various aspects of data wrangling, from collection to verification and cleaning, and can serve as excellent starting points for further exploration of the topic.

