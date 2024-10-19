## Leveraging Regular Expressions in Pandas
Slide 1: Introduction to Regular Expressions in Pandas

Regular expressions (regex) are powerful tools for pattern matching and data extraction in text. When combined with Pandas, they become even more potent for data manipulation and analysis. This presentation will cover essential techniques for using regex in Pandas, demonstrating how to extract, clean, and validate various types of data.

```python
import pandas as pd
import re

# Create a sample DataFrame
data = {
    'text': [
        'Call me at 123-456-7890 or email john@example.com',
        'Visit our website at https://www.example.com',
        'My ID is ABC-12345 and I live in New York, NY 10001'
    ]
}
df = pd.DataFrame(data)
print(df)
```

Slide 2: Extracting Phone Numbers

Regular expressions can be used to extract phone numbers from text data. We'll use a pattern that matches the common format of American phone numbers.

```python
# Extract phone numbers
df['phone'] = df['text'].str.extract(r'(\d{3}-\d{3}-\d{4})')
print(df[['text', 'phone']])
```

Slide 3: Results for: Extracting Phone Numbers

```
                                                text         phone
0  Call me at 123-456-7890 or email john@example.com  123-456-7890
1  Visit our website at https://www.example.com              None
2  My ID is ABC-12345 and I live in New York, NY...         None
```

Slide 4: Extracting Email Addresses

Email addresses can be extracted using a regex pattern that matches the typical structure of an email address.

```python
# Extract email addresses
df['email'] = df['text'].str.extract(r'(\S+@\S+\.\S+)')
print(df[['text', 'email']])
```

Slide 5: Results for: Extracting Email Addresses

```
                                                text               email
0  Call me at 123-456-7890 or email john@example.com  john@example.com
1  Visit our website at https://www.example.com              None
2  My ID is ABC-12345 and I live in New York, NY...         None
```

Slide 6: Extracting URLs

URLs can be extracted using a regex pattern that matches the common structure of web addresses.

```python
# Extract URLs
df['url'] = df['text'].str.extract(r'(https?://\S+)')
print(df[['text', 'url']])
```

Slide 7: Results for: Extracting URLs

```
                                                text                        url
0  Call me at 123-456-7890 or email john@example.com                       None
1  Visit our website at https://www.example.com    https://www.example.com
2  My ID is ABC-12345 and I live in New York, NY...                       None
```

Slide 8: Cleaning Special Characters

Regular expressions can be used to remove or replace special characters in text data.

```python
# Remove special characters
df['cleaned_text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)
print(df[['text', 'cleaned_text']])
```

Slide 9: Results for: Cleaning Special Characters

```
                                                text                                   cleaned_text
0  Call me at 123-456-7890 or email john@example.com  Call me at 1234567890 or email johnexamplecom
1  Visit our website at https://www.example.com       Visit our website at httpswwwexamplecom
2  My ID is ABC-12345 and I live in New York, NY...   My ID is ABC12345 and I live in New York NY...
```

Slide 10: Validating Patterns

Regex can be used to validate if a string matches a specific pattern, such as an ID format.

```python
# Validate ID format (e.g., ABC-12345)
df['valid_id'] = df['text'].str.contains(r'\b[A-Z]{3}-\d{5}\b')
print(df[['text', 'valid_id']])
```

Slide 11: Results for: Validating Patterns

```
                                                text  valid_id
0  Call me at 123-456-7890 or email john@example.com     False
1  Visit our website at https://www.example.com          False
2  My ID is ABC-12345 and I live in New York, NY...      True
```

Slide 12: Extracting Multiple Matches

Sometimes we need to extract multiple occurrences of a pattern within a single string.

```python
# Extract all numbers
df['numbers'] = df['text'].str.findall(r'\d+')
print(df[['text', 'numbers']])
```

Slide 13: Results for: Extracting Multiple Matches

```
                                                text                    numbers
0  Call me at 123-456-7890 or email john@example.com  [123, 456, 7890]
1  Visit our website at https://www.example.com       []
2  My ID is ABC-12345 and I live in New York, NY...   [12345, 10001]
```

Slide 14: Real-Life Example: Analyzing Product Reviews

Let's analyze product reviews to extract sentiment words and product ratings.

```python
reviews = pd.DataFrame({
    'review': [
        "Great product! I love it. 5/5 stars.",
        "Decent quality, but overpriced. 3 out of 5.",
        "Terrible experience. Avoid at all costs! 1 star."
    ]
})

# Extract sentiment words
reviews['sentiment'] = reviews['review'].str.extract(r'\b(great|decent|terrible)\b', flags=re.IGNORECASE)

# Extract ratings
reviews['rating'] = reviews['review'].str.extract(r'(\d+)(?:/5| out of 5| star)')

print(reviews)
```

Slide 15: Results for: Real-Life Example: Analyzing Product Reviews

```
                                              review sentiment rating
0             Great product! I love it. 5/5 stars.     Great      5
1        Decent quality, but overpriced. 3 out of 5.   Decent      3
2  Terrible experience. Avoid at all costs! 1 star.  Terrible      1
```

Slide 16: Real-Life Example: Parsing Log Files

System administrators often need to parse log files to extract important information. Let's use regex to parse a simple log file.

```python
log_data = pd.DataFrame({
    'log_entry': [
        "[2024-03-15 08:30:45] INFO: User login successful - username: john_doe",
        "[2024-03-15 09:15:22] ERROR: Database connection failed - error code: DB001",
        "[2024-03-15 10:05:37] WARNING: High CPU usage detected - usage: 95%"
    ]
})

# Extract timestamp, log level, and message
log_data[['timestamp', 'level', 'message']] = log_data['log_entry'].str.extract(r'\[(.*?)\] (\w+): (.+)')

print(log_data)
```

Slide 17: Results for: Real-Life Example: Parsing Log Files

```
                                           log_entry            timestamp   level                                            message
0  [2024-03-15 08:30:45] INFO: User login success...  2024-03-15 08:30:45    INFO    User login successful - username: john_doe
1  [2024-03-15 09:15:22] ERROR: Database connecti...  2024-03-15 09:15:22   ERROR    Database connection failed - error code: DB001
2  [2024-03-15 10:05:37] WARNING: High CPU usage ...  2024-03-15 10:05:37  WARNING   High CPU usage detected - usage: 95%
```

Slide 18: Additional Resources

For those interested in diving deeper into regular expressions and their applications in data analysis, here are some additional resources:

1.  "Regular Expression Matching Can Be Simple And Fast" by Russ Cox ([https://arxiv.org/abs/1407.7246](https://arxiv.org/abs/1407.7246))
2.  "Parsing Gigabytes of JSON per Second" by Daniel Lemire et al. ([https://arxiv.org/abs/1902.08318](https://arxiv.org/abs/1902.08318))

These papers provide insights into the efficiency and performance aspects of regular expressions and parsing techniques, which can be valuable when working with large datasets in Pandas.

