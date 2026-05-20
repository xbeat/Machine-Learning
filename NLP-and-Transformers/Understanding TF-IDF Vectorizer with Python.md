## Understanding TF-IDF Vectorizer with Python
Slide 1: Introduction to TF-IDF Vectorizer

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic used to reflect the importance of words in a document within a collection or corpus. The TF-IDF Vectorizer transforms text into a vector representation, making it suitable for machine learning algorithms. This technique is widely used in information retrieval and text mining.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog sleeps all day",
    "The quick brown fox is quick and brown"
]

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the corpus
tfidf_matrix = vectorizer.fit_transform(corpus)

print(tfidf_matrix.shape)
# Output: (3, 11) - 3 documents, 11 unique words
```

Slide 2: Term Frequency (TF)

Term Frequency measures how often a term appears in a document. It's calculated by dividing the number of occurrences of a term by the total number of terms in the document. This helps to normalize the frequency across documents of different lengths.

```python
import pandas as pd

def calculate_tf(document):
    words = document.lower().split()
    word_count = len(words)
    tf_dict = {}
    for word in set(words):
        tf_dict[word] = words.count(word) / word_count
    return tf_dict

document = "The quick brown fox jumps over the lazy dog"
tf = calculate_tf(document)
print(pd.DataFrame([tf]).T)
```

Slide 3: Inverse Document Frequency (IDF)

Inverse Document Frequency measures how important a term is across the entire corpus. It's calculated as the logarithm of the total number of documents divided by the number of documents containing the term. IDF reduces the weight of common words and increases the weight of rare words.

```python
import math

def calculate_idf(corpus):
    N = len(corpus)
    idf_dict = {}
    for document in corpus:
        for word in set(document.lower().split()):
            if word in idf_dict:
                idf_dict[word] += 1
            else:
                idf_dict[word] = 1
    
    for word, count in idf_dict.items():
        idf_dict[word] = math.log(N / count)
    
    return idf_dict

corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog sleeps all day",
    "The quick brown fox is quick and brown"
]

idf = calculate_idf(corpus)
print(pd.DataFrame([idf]).T)
```

Slide 4: TF-IDF Calculation

TF-IDF is calculated by multiplying the Term Frequency (TF) and Inverse Document Frequency (IDF) for each term in a document. This combination gives higher weights to terms that are frequent in a specific document but rare across the corpus.

```python
def calculate_tfidf(tf, idf):
    tfidf = {}
    for word, tf_value in tf.items():
        tfidf[word] = tf_value * idf.get(word, 0)
    return tfidf

document = "The quick brown fox jumps over the lazy dog"
tf = calculate_tf(document)
idf = calculate_idf([document])
tfidf = calculate_tfidf(tf, idf)

print(pd.DataFrame([tfidf]).T)
```

Slide 5: TF-IDF Vectorizer in Scikit-learn

Scikit-learn provides a TfidfVectorizer class that combines all the steps of TF-IDF calculation and vectorization. It tokenizes the text, learns the vocabulary, and computes the TF-IDF representation for each document in the corpus.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog sleeps all day",
    "The quick brown fox is quick and brown"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
print("Feature Names:", vectorizer.get_feature_names_out())
```

Slide 6: Understanding TF-IDF Matrix

The TF-IDF matrix is a sparse matrix where each row represents a document, and each column represents a unique term in the corpus. The values in the matrix are the TF-IDF scores for each term in each document.

```python
import numpy as np

# Convert sparse matrix to dense array for visualization
dense_matrix = tfidf_matrix.toarray()

# Create a DataFrame for better visualization
df = pd.DataFrame(dense_matrix, columns=vectorizer.get_feature_names_out())
print(df)
```

Slide 7: Interpreting TF-IDF Scores

TF-IDF scores help identify the most important terms in each document. Higher scores indicate terms that are both frequent in the document and relatively rare in the corpus. This allows us to extract key features or topics from documents.

```python
def top_tfidf_terms(tfidf_matrix, feature_names, top_n=5):
    for i, doc in enumerate(tfidf_matrix):
        feature_index = doc.nonzero()[1]
        tfidf_scores = zip(feature_index, [doc[0, x] for x in feature_index])
        top_terms = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:top_n]
        print(f"Document {i + 1} top terms:")
        for idx, score in top_terms:
            print(f"{feature_names[idx]}: {score:.4f}")
        print()

top_tfidf_terms(tfidf_matrix, vectorizer.get_feature_names_out())
```

Slide 8: Preprocessing with TF-IDF Vectorizer

TfidfVectorizer offers various preprocessing options, such as lowercasing, removing stop words, and setting minimum and maximum document frequency thresholds. These options help in cleaning and refining the text data before vectorization.

```python
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.9  # Ignore terms that appear in more than 90% of documents
)

tfidf_matrix = vectorizer.fit_transform(corpus)
print("Feature Names:", vectorizer.get_feature_names_out())
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
```

Slide 9: N-grams in TF-IDF Vectorizer

N-grams are contiguous sequences of n items from a given text. TfidfVectorizer can generate n-grams to capture multi-word phrases, which can be useful for preserving context and identifying important combinations of words.

```python
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Unigrams and bigrams
tfidf_matrix = vectorizer.fit_transform(corpus)

print("Feature Names:", vectorizer.get_feature_names_out())
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
```

Slide 10: Real-life Example: Document Classification

TF-IDF Vectorizer is commonly used in document classification tasks. Here's an example of using TF-IDF features with a Naive Bayes classifier to categorize news articles.

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
news_data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(news_data.data, news_data.target, test_size=0.3, random_state=42)

# Vectorize and classify
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"Classification Accuracy: {accuracy:.2f}")
```

Slide 11: Real-life Example: Information Retrieval

TF-IDF is widely used in information retrieval systems, such as search engines. Here's a simple example of how TF-IDF can be used to rank documents based on their relevance to a query.

```python
from sklearn.metrics.pairwise import cosine_similarity

# Sample corpus
corpus = [
    "Climate change is a global issue affecting our planet",
    "Renewable energy sources can help combat climate change",
    "Artificial intelligence is revolutionizing various industries",
    "Machine learning algorithms are used in AI applications"
]

# Query
query = "climate change impact"

# Vectorize corpus and query
vectorizer = TfidfVectorizer()
corpus_tfidf = vectorizer.fit_transform(corpus)
query_tfidf = vectorizer.transform([query])

# Calculate cosine similarity between query and documents
similarities = cosine_similarity(query_tfidf, corpus_tfidf).flatten()

# Rank documents based on similarity
ranked_docs = sorted(zip(similarities, corpus), reverse=True)

print("Ranked documents:")
for score, doc in ranked_docs:
    print(f"Score: {score:.4f} - Document: {doc}")
```

Slide 12: Limitations and Considerations

While TF-IDF is powerful, it has limitations. It doesn't capture word order or semantics, treats words independently, and can struggle with out-of-vocabulary words. For more advanced text representation, consider techniques like word embeddings (Word2Vec, GloVe) or contextual embeddings (BERT, GPT).

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulate TF-IDF scores for different scenarios
scenarios = ['Common Words', 'Rare Words', 'Domain-Specific', 'Out-of-Vocabulary']
tfidf_scores = [0.1, 0.8, 0.6, 0]

plt.figure(figsize=(10, 6))
plt.bar(scenarios, tfidf_scores)
plt.title('TF-IDF Scores in Different Scenarios')
plt.ylabel('TF-IDF Score')
plt.ylim(0, 1)

for i, v in enumerate(tfidf_scores):
    plt.text(i, v + 0.05, f'{v:.1f}', ha='center')

plt.show()
```

Slide 13: Advanced TF-IDF Techniques

There are several advanced techniques to improve TF-IDF performance, such as using sublinear TF scaling, applying L2 normalization, or incorporating domain-specific knowledge. These modifications can help address some limitations and improve the effectiveness of TF-IDF in specific applications.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog sleeps all day",
    "The quick brown fox is quick and brown"
]

# Advanced TF-IDF configuration
advanced_vectorizer = TfidfVectorizer(
    sublinear_tf=True,  # Apply sublinear TF scaling
    norm='l2',          # L2 normalization
    analyzer='word',    # Analyze at word level
    token_pattern=r'\b\w+\b',  # Custom token pattern
    stop_words='english',
    ngram_range=(1, 2),  # Include unigrams and bigrams
    max_features=1000    # Limit vocabulary size
)

advanced_tfidf_matrix = advanced_vectorizer.fit_transform(corpus)
print("Advanced TF-IDF Matrix Shape:", advanced_tfidf_matrix.shape)
print("Feature Names:", advanced_vectorizer.get_feature_names_out())
```

Slide 14: Additional Resources

For further exploration of TF-IDF and related techniques, consider the following resources:

1. "From Frequency to Meaning: Vector Space Models of Semantics" by Peter D. Turney and Patrick Pantel (arXiv:1003.1141)
2. "A Survey of Text Classification Algorithms" by Charu C. Aggarwal and ChengXiang Zhai (arXiv:1904.08067)
3. "Efficient Estimation of Word Representations in Vector Space" by Tomas Mikolov et al. (arXiv:1301.3781)

These papers provide in-depth discussions on text representation techniques, including TF-IDF and more advanced methods.

