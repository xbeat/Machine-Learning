## Python Data Science Cheatsheet

Slide 1: Python for Data Science: Introduction

Data Science with Python involves using powerful libraries and tools to analyze, visualize, and interpret data. This cheatsheet covers essential concepts and techniques for beginners and intermediate users, focusing on practical, actionable examples.

Slide 2: Source Code for Python for Data Science: Introduction

```python
# Basic data science workflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('dataset.csv')

# Perform analysis
mean_value = np.mean(data['column_name'])

# Visualize results
plt.plot(data['x'], data['y'])
plt.show()
```

Slide 3: Data Loading and Exploration

Data scientists often start by loading and exploring datasets. Pandas is a popular library for this purpose, offering powerful tools for data manipulation and analysis.

Slide 4: Source Code for Data Loading and Exploration

```python
import pandas as pd

# Load CSV file
df = pd.read_csv('dataset.csv')

# Display first few rows
print(df.head())

# Get basic information about the dataset
print(df.info())

# Calculate summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())
```

Slide 5: Data Cleaning and Preprocessing

Raw data often requires cleaning and preprocessing before analysis. This involves handling missing values, removing duplicates, and transforming data types.

Slide 6: Source Code for Data Cleaning and Preprocessing

```python
import pandas as pd

# Handle missing values
df['column'].fillna(df['column'].mean(), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert data types
df['date_column'] = pd.to_datetime(df['date_column'])

# Normalize numerical columns
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['num_col1', 'num_col2']] = scaler.fit_transform(df[['num_col1', 'num_col2']])
```

Slide 7: Data Visualization

Visualizing data helps in understanding patterns, trends, and relationships. Matplotlib and Seaborn are popular libraries for creating various types of plots.

Slide 8: Source Code for Data Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='x_column', y='y_column', data=df)
plt.title('Scatter Plot')
plt.show()

# Create a histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['column'], kde=True)
plt.title('Histogram')
plt.show()

# Create a heatmap of correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

Slide 9: Statistical Analysis

Statistical analysis is crucial in data science for hypothesis testing, inferencing, and understanding data distributions.

Slide 10: Source Code for Statistical Analysis

```python
import scipy.stats as stats

# Perform t-test
group1 = df[df['category'] == 'A']['value']
group2 = df[df['category'] == 'B']['value']
t_statistic, p_value = stats.ttest_ind(group1, group2)

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

# Calculate correlation
correlation = df['column1'].corr(df['column2'])
print(f"Correlation: {correlation}")

# Perform ANOVA
categories = [group for _, group in df.groupby('category')['value']]
f_statistic, p_value = stats.f_oneway(*categories)
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")
```

Slide 11: Machine Learning: Model Training

Machine learning is a core component of data science. Scikit-learn provides a wide range of algorithms for classification, regression, and clustering tasks.

Slide 12: Source Code for Machine Learning: Model Training

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Prepare data
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
```

Slide 13: Feature Engineering

Feature engineering is the process of creating new features or transforming existing ones to improve model performance.

Slide 14: Source Code for Feature Engineering

```python
import pandas as pd
import numpy as np

# Create interaction features
df['interaction'] = df['feature1'] * df['feature2']

# Bin continuous variables
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '65+'])

# Create dummy variables for categorical features
df_encoded = pd.get_dummies(df, columns=['category'])

# Apply logarithmic transformation
df['log_income'] = np.log(df['income'] + 1)  # Adding 1 to handle zero values

# Create time-based features
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
```

Slide 15: Real-Life Example: Customer Churn Prediction

In this example, we'll predict customer churn for a telecom company using a logistic regression model.

Slide 16: Source Code for Real-Life Example: Customer Churn Prediction

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the telecom customer churn dataset
df = pd.read_csv('telecom_churn.csv')

# Prepare features and target
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn'].map({'Yes': 1, 'No': 0})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
```

Slide 17: Real-Life Example: Sentiment Analysis

In this example, we'll perform sentiment analysis on product reviews using natural language processing techniques.

Slide 18: Source Code for Real-Life Example: Sentiment Analysis

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the product reviews dataset
df = pd.read_csv('product_reviews.csv')

# Prepare features and target
X = df['review_text']
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
```

Slide 19: Additional Resources

To deepen your understanding of data science with Python, explore these valuable resources:

1.  ArXiv.org: "Deep Learning for Time Series Forecasting: A Survey" ([https://arxiv.org/abs/2004.13408](https://arxiv.org/abs/2004.13408)) This comprehensive survey covers various deep learning techniques applied to time series forecasting, a crucial area in data science.
2.  ArXiv.org: "A Survey on Explainable Artificial Intelligence (XAI): Towards Medical XAI" ([https://arxiv.org/abs/1907.07374](https://arxiv.org/abs/1907.07374)) This paper provides insights into explainable AI, focusing on its applications in healthcare and medical research.
3.  Python Data Science Handbook by Jake VanderPlas (available online) This free resource offers in-depth coverage of essential libraries like NumPy, Pandas, Matplotlib, and Scikit-learn.
4.  Official documentation for key Python libraries:
    *   Pandas: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
    *   NumPy: [https://numpy.org/doc/](https://numpy.org/doc/)
    *   Scikit-learn: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
    *   Matplotlib: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)

These resources provide a mix of theoretical foundations and practical implementations to enhance your data science skills with Python.

