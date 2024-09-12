## Natural Language Processing with NLTK and Pandas:
Slide 1: Introduction to Natural Language Toolkit (NLTK) NLTK is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for the Python programming language. It provides easy-to-use interfaces to over 50 corpora and lexical resources, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

text = "Hello, world! This is an example sentence."
print(sent_tokenize(text))
print(word_tokenize(text))
```

Slide 2: Importing NLTK and Downloading Resources Before working with NLTK, you need to import the necessary modules and download the required resources. NLTK provides a convenient interface to download and install various data packages, including corpora, models, and other resources.

```python
import nltk
nltk.download('all')
```

Slide 3: Tokenization Tokenization is the process of breaking a text into smaller units, such as words, sentences, or even subwords. NLTK provides various tokenizers for different levels of tokenization.

```python
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Hello, world! This is an example sentence."
words = word_tokenize(text)
sentences = sent_tokenize(text)

print("Words:", words)
print("Sentences:", sentences)
```

Slide 4: Stemming and Lemmatization Stemming and lemmatization are text normalization techniques used to reduce words to their base or root form. NLTK provides various stemmers and lemmatizers for different languages.

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

word = "running"
print("Stemmed word:", stemmer.stem(word))
print("Lemmatized word:", lemmatizer.lemmatize(word))
```

Slide 5: Part-of-Speech Tagging Part-of-Speech (POS) tagging is the process of assigning a part-of-speech tag (noun, verb, adjective, etc.) to each word in a text. NLTK provides pre-trained models and tools for POS tagging.

```python
import nltk
from nltk import pos_tag, word_tokenize

text = "The quick brown fox jumps over the lazy dog."
tokens = word_tokenize(text)
tagged = pos_tag(tokens)

print(tagged)
```

Slide 6: Named Entity Recognition (NER) Named Entity Recognition (NER) is the process of identifying and classifying named entities, such as person names, organizations, locations, and numerical expressions, in unstructured text. NLTK provides pre-trained models and tools for NER.

```python
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize

text = "John Smith is the CEO of Acme Corp. located in New York."
tokens = word_tokenize(text)
tagged = pos_tag(tokens)
entities = ne_chunk(tagged)

print(entities)
```

Slide 7: Text Classification Text classification is the process of assigning predefined categories or labels to text documents based on their content. NLTK provides tools and utilities for building and evaluating text classifiers.

```python
import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier

# Load and preprocess data
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Train the classifier
train_data = documents[:1000]
test_data = documents[1000:]
classifier = NaiveBayesClassifier.train(train_data)

# Evaluate the classifier
print(nltk.classify.accuracy(classifier, test_data))
```

Slide 8: Introduction to Pandas Pandas is a powerful open-source Python library for data manipulation and analysis. It provides data structures and data analysis tools for working with structured (tabular, multidimensional, potentially heterogeneous) and time series data.

```python
import pandas as pd

# Creating a DataFrame
data = {'Name': ['John', 'Jane', 'Bob'],
        'Age': [25, 30, 35]}
df = pd.DataFrame(data)

print(df)
```

Slide 9: Importing and Exporting Data with Pandas Pandas provides convenient functions for reading and writing data in various formats, such as CSV, Excel, SQL databases, and more.

```python
import pandas as pd

# Reading data from a CSV file
df = pd.read_csv('data.csv')

# Writing data to an Excel file
df.to_excel('output.xlsx', index=False)
```

Slide 10: Data Selection and Manipulation Pandas provides powerful tools for selecting, filtering, and manipulating data in a DataFrame or Series.

```python
import pandas as pd

# Creating a DataFrame
data = {'Name': ['John', 'Jane', 'Bob', 'Alice'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'London', 'Paris', 'Tokyo']}
df = pd.DataFrame(data)

# Selecting columns
print(df[['Name', 'Age']])

# Filtering rows
print(df[df['Age'] > 30])
```

Slide 11: Data Cleaning and Preprocessing Pandas provides functions and methods for cleaning and preprocessing data, such as handling missing values, removing duplicates, and data type conversions.

```python
import pandas as pd
import numpy as np

# Creating a DataFrame with missing values
data = {'Name': ['John', 'Jane', 'Bob', np.nan],
        'Age': [25, np.nan, 35, 40]}
df = pd.DataFrame(data)

# Handling missing values
print(df.dropna())  # Drop rows with missing values
print(df.fillna(0))  # Fill missing values with 0
```

Slide 12: Data Visualization with Pandas Pandas integrates with popular data visualization libraries like Matplotlib and Seaborn, providing convenient methods for creating various types of plots directly from DataFrame objects.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Creating a DataFrame
data = {'Name': ['John', 'Jane', 'Bob', 'Alice'],
        'Age': [25, 30, 35, 40]}
df = pd.DataFrame(data)

# Creating a bar plot
df.plot(kind='bar', x='Name', y='Age')
plt.show()
```

Slide 13: Grouping and Aggregating Data Pandas provides powerful grouping and aggregation functions for performing operations on grouped data, such as computing group statistics or applying transformations.

```python
import pandas as pd

# Creating a DataFrame
data = {'Name': ['John', 'Jane', 'Bob', 'Alice', 'John', 'Jane'],
        'Age': [25, 30, 35, 40, 27, 32],
        'City': ['New York', 'London', 'Paris', 'Tokyo', 'New York', 'London']}
df = pd.DataFrame(data)

# Grouping data by 'City' and computing the mean age
print(df.groupby('City')['Age'].mean())
```

Slide 14: Additional Resources Here are some additional resources for learning NLTK and Pandas:

* NLTK Book: [https://www.nltk.org/book/](https://www.nltk.org/book/) (Reference: [https://arxiv.org/abs/0205028](https://arxiv.org/abs/0205028))
* Pandas Documentation: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
* "Python for Data Analysis" by Wes McKinney (ArXiv: [https://arxiv.org/abs/1012.3620](https://arxiv.org/abs/1012.3620))

