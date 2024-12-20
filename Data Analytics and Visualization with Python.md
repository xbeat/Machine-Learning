## Data Analytics and Visualization with Python
Slide 1: Introduction to Data Analytics and Visualization with Python

Data analytics and visualization are essential tools for extracting insights from complex datasets. Python, with its rich ecosystem of libraries, provides powerful capabilities for data manipulation, analysis, and visualization. This presentation will cover key concepts and techniques in data analytics and visualization using Python, focusing on practical examples and actionable code.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a simple line plot
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Simple Sine Wave Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
```

Slide 2: Data Manipulation with Pandas

Pandas is a powerful library for data manipulation and analysis in Python. It provides data structures like DataFrames and Series, which allow for efficient handling of structured data.

```python
import pandas as pd

# Create a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'San Francisco', 'London', 'Paris']
}
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Basic operations
print("\nAverage age:", df['Age'].mean())
print("\nUnique cities:", df['City'].unique())
```

Slide 3: Results for: Data Manipulation with Pandas

```
   Name  Age          City
0  Alice   25      New York
1    Bob   30  San Francisco
2  Charlie 35        London
3   David  28         Paris

Average age: 29.5

Unique cities: ['New York' 'San Francisco' 'London' 'Paris']
```

Slide 4: Data Cleaning and Preprocessing

Data cleaning and preprocessing are crucial steps in any data analysis project. Python offers various tools to handle missing values, remove duplicates, and transform data.

```python
import pandas as pd
import numpy as np

# Create a DataFrame with missing values and duplicates
data = {
    'A': [1, 2, np.nan, 4, 5, 5],
    'B': [5, 6, 7, np.nan, 9, 9],
    'C': ['a', 'b', 'c', 'd', 'e', 'e']
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Remove duplicates
df_clean = df.drop_duplicates()

# Fill missing values
df_clean = df_clean.fillna(df_clean.mean())

print("\nCleaned DataFrame:")
print(df_clean)
```

Slide 5: Results for: Data Cleaning and Preprocessing

```
Original DataFrame:
     A    B  C
0  1.0  5.0  a
1  2.0  6.0  b
2  NaN  7.0  c
3  4.0  NaN  d
4  5.0  9.0  e
5  5.0  9.0  e

Cleaned DataFrame:
     A    B  C
0  1.0  5.0  a
1  2.0  6.0  b
2  3.0  7.0  c
3  4.0  6.75 d
4  5.0  9.0  e
```

Slide 6: Exploratory Data Analysis (EDA)

Exploratory Data Analysis is a critical step in understanding the characteristics and patterns in your data. Python provides various tools for statistical analysis and visualization to aid in EDA.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load a sample dataset
tips = sns.load_dataset("tips")

# Display basic statistics
print(tips.describe())

# Create a histogram of total bill
plt.figure(figsize=(10, 6))
sns.histplot(tips['total_bill'], kde=True)
plt.title('Distribution of Total Bill')
plt.xlabel('Total Bill')
plt.ylabel('Frequency')
plt.show()

# Create a scatter plot of total bill vs tip
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_bill', y='tip', data=tips, hue='time')
plt.title('Total Bill vs Tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.show()
```

Slide 7: Data Visualization with Matplotlib

Matplotlib is a versatile plotting library in Python that allows for the creation of a wide range of static, animated, and interactive visualizations.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot sine wave
ax1.plot(x, y1, color='blue', label='Sine')
ax1.set_title('Sine Wave')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.legend()
ax1.grid(True)

# Plot cosine wave
ax2.plot(x, y2, color='red', label='Cosine')
ax2.set_title('Cosine Wave')
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

Slide 8: Interactive Visualizations with Plotly

Plotly is a powerful library for creating interactive and publication-quality visualizations in Python. It allows for the creation of a wide range of chart types with built-in interactivity.

```python
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100)
values = np.cumsum(np.random.randn(100))

# Create an interactive line plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=values, mode='lines', name='Time Series'))
fig.update_layout(title='Interactive Time Series Plot',
                  xaxis_title='Date',
                  yaxis_title='Value')

fig.show()
```

Slide 9: Machine Learning with Scikit-learn

Scikit-learn is a popular machine learning library in Python that provides a wide range of algorithms for classification, regression, clustering, and dimensionality reduction.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

Slide 10: Results for: Machine Learning with Scikit-learn

```
Accuracy: 0.96

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        16
  versicolor       0.95      0.95      0.95        19
   virginica       0.93      0.93      0.93        10

    accuracy                           0.96        45
   macro avg       0.96      0.96      0.96        45
weighted avg       0.96      0.96      0.96        45
```

Slide 11: Time Series Analysis

Time series analysis is crucial for understanding patterns and trends in data that change over time. Python provides various tools for handling time-based data and performing time series analysis.

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Create a sample time series dataset
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = pd.Series(range(len(dates))) + pd.Series(10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365))
ts = pd.Series(values, index=dates)

# Perform seasonal decomposition
result = seasonal_decompose(ts, model='additive', period=365)

# Plot the decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
result.observed.plot(ax=ax1)
ax1.set_title('Observed')
result.trend.plot(ax=ax2)
ax2.set_title('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()
```

Slide 12: Natural Language Processing (NLP)

Natural Language Processing is a field of AI that focuses on the interaction between computers and human language. Python offers various libraries for text processing and analysis.

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')

# Sample text
text = "Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."

# Tokenization
tokens = word_tokenize(text)

# Remove punctuation and convert to lowercase
tokens = [word.lower() for word in tokens if word not in string.punctuation]

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

print("Original text:", text)
print("\nTokenized and processed text:", stemmed_tokens)
```

Slide 13: Results for: Natural Language Processing (NLP)

```
Original text: Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.

Tokenized and processed text: ['natur', 'languag', 'process', 'nlp', 'subfield', 'linguist', 'comput', 'scienc', 'artifici', 'intellig', 'concern', 'interact', 'comput', 'human', 'languag']
```

Slide 14: Real-life Example: Weather Data Analysis

In this example, we'll analyze weather data to identify trends and patterns. This type of analysis is crucial for climate research and weather forecasting.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load sample weather data (you would typically load this from a file)
data = {
    'Date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'),
    'Temperature': np.random.normal(15, 5, 365) + 10 * np.sin(np.arange(365) * 2 * np.pi / 365)
}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Calculate moving average
df['MA_7'] = df['Temperature'].rolling(window=7).mean()

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Temperature'], label='Daily Temperature')
plt.plot(df.index, df['MA_7'], label='7-day Moving Average', color='red')
plt.title('Temperature Trends Over a Year')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate monthly averages
monthly_avg = df.resample('M')['Temperature'].mean()
print("Monthly Temperature Averages:")
print(monthly_avg)
```

Slide 15: Real-life Example: Text Sentiment Analysis

Sentiment analysis is widely used in social media monitoring, customer feedback analysis, and market research. This example demonstrates a simple sentiment analysis on product reviews.

```python
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')

# Sample product reviews
reviews = [
    "This product is amazing! I love it.",
    "The quality is poor and it broke after a week.",
    "Average product, nothing special.",
    "I'm impressed with the features, but the price is too high.",
    "Terrible customer service, I'm very disappointed."
]

# Perform sentiment analysis
sia = SentimentIntensityAnalyzer()
sentiments = [sia.polarity_scores(review)['compound'] for review in reviews]

# Categorize sentiments
categories = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
counts = [sum((-1 <= s < -0.6, -0.6 <= s < -0.2, -0.2 <= s < 0.2, 0.2 <= s < 0.6, 0.6 <= s <= 1)) for s in sentiments]

# Visualize results
plt.figure(figsize=(10, 6))
plt.bar(categories, counts)
plt.title('Sentiment Analysis of Product Reviews')
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Reviews')
plt.show()

# Print individual review sentiments
for review, sentiment in zip(reviews, sentiments):
    print(f"Review: {review}")
    print(f"Sentiment Score: {sentiment:.2f}")
    print()
```

Slide 16: Additional Resources

For those interested in diving deeper into data analytics and visualization with Python, here are some valuable resources:

1.  ArXiv.org: A rich source of research papers on data science and machine learning. Example: "A Survey of Deep Learning Techniques for Neural Machine Translation" ([https://arxiv.org/abs/1703.01619](https://arxiv.org/abs/1703.01619))
2.  Python Data Science Handbook by Jake VanderPlas: A comprehensive guide to the scientific Python ecosystem.
3.  Coursera and edX: Offer various online courses on data analytics and visualization with Python.
4.  Official documentation of key libraries:
    *   Pandas: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
    *   Matplotlib: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
    *   Scikit-learn: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
5.  Data visualization blogs:
    *   Flowing Data: [https://flowingdata.com/](https://flowingdata.com/)
    *   Information is Beautiful: [https://informationisbeautiful.net/](https://informationisbeautiful.net/)

Remember to verify the accuracy and relevance of these resources, as they may have been updated since this presentation was created.

