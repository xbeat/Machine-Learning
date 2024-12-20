## TF-IDF in Natural Language Processing

Slide 1: Introduction to TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic used to reflect the importance of a word in a document within a collection or corpus. It's widely used in information retrieval and text mining.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.shape)
```

Slide 2: Term Frequency (TF)

Term Frequency measures how frequently a term appears in a document. It's calculated as the number of occurrences of a term in a document divided by the total number of terms in that document.

```python
def term_frequency(term, document):
    words = document.lower().split()
    return words.count(term.lower()) / len(words)

document = "This is a sample document to calculate term frequency."
term = "document"
tf = term_frequency(term, document)
print(f"TF of '{term}': {tf}")
```

Slide 3: Inverse Document Frequency (IDF)

IDF measures the importance of a term across the entire corpus. It's calculated as the logarithm of the total number of documents divided by the number of documents containing the term.

```python
import math

def inverse_document_frequency(term, documents):
    num_documents = len(documents)
    num_documents_with_term = sum(1 for doc in documents if term.lower() in doc.lower())
    return math.log(num_documents / (1 + num_documents_with_term))

documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
term = "document"
idf = inverse_document_frequency(term, documents)
print(f"IDF of '{term}': {idf}")
```

Slide 4: Calculating TF-IDF

TF-IDF is calculated by multiplying the Term Frequency (TF) and Inverse Document Frequency (IDF) for each term in a document.

```python
def tf_idf(term, document, documents):
    tf = term_frequency(term, document)
    idf = inverse_document_frequency(term, documents)
    return tf * idf

documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
term = "document"
document = documents[1]
tfidf = tf_idf(term, document, documents)
print(f"TF-IDF of '{term}' in document: {tfidf}")
```

Slide 5: TF-IDF with scikit-learn

Scikit-learn provides a convenient TfidfVectorizer class to compute TF-IDF scores for a collection of documents.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()
print("Feature names:", feature_names)
print("TF-IDF matrix shape:", tfidf_matrix.shape)
```

Slide 6: Analyzing TF-IDF Scores

After calculating TF-IDF scores, we can analyze which terms are most important in each document.

```python
import pandas as pd

def display_tfidf_scores(tfidf_matrix, feature_names, document_index):
    df = pd.DataFrame(tfidf_matrix[document_index].T.todense(), index=feature_names, columns=["TF-IDF"])
    df = df.sort_values("TF-IDF", ascending=False)
    return df.head(5)

document_index = 1
top_terms = display_tfidf_scores(tfidf_matrix, feature_names, document_index)
print(f"Top 5 terms in document {document_index}:")
print(top_terms)
```

Slide 7: Document Similarity with TF-IDF

TF-IDF can be used to measure similarity between documents using cosine similarity.

```python
from sklearn.metrics.pairwise import cosine_similarity

def document_similarity(tfidf_matrix, doc1_index, doc2_index):
    return cosine_similarity(tfidf_matrix[doc1_index], tfidf_matrix[doc2_index])[0][0]

doc1_index, doc2_index = 0, 3
similarity = document_similarity(tfidf_matrix, doc1_index, doc2_index)
print(f"Similarity between document {doc1_index} and {doc2_index}: {similarity}")
```

Slide 8: TF-IDF for Text Classification

TF-IDF features can be used as input for text classification algorithms.

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

X = tfidf_matrix
y = [0, 1, 2, 0]  # Example labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification accuracy: {accuracy}")
```

Slide 9: Handling Stop Words

Stop words are common words that don't carry much meaning. We can remove them to improve TF-IDF results.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

vectorizer = TfidfVectorizer(stop_words=stop_words)
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()
print("Feature names without stop words:", feature_names)
```

Slide 10: N-grams in TF-IDF

N-grams are contiguous sequences of n items from a given text. We can include them in TF-IDF calculations.

```python
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()
print("Feature names with unigrams and bigrams:", feature_names)
```

Slide 11: TF-IDF Normalization

TF-IDF scores can be normalized to account for document length differences.

```python
vectorizer = TfidfVectorizer(norm='l2')
tfidf_matrix = vectorizer.fit_transform(documents)

def print_normalized_vector(tfidf_matrix, document_index):
    vector = tfidf_matrix[document_index].toarray()[0]
    norm = sum(v**2 for v in vector)**0.5
    print(f"L2 norm of document {document_index}: {norm}")

print_normalized_vector(tfidf_matrix, 0)
print_normalized_vector(tfidf_matrix, 1)
```

Slide 12: TF-IDF for Document Summarization

TF-IDF can be used to identify the most important sentences in a document for summarization.

```python
import numpy as np

def summarize(text, num_sentences=2):
    sentences = text.split('.')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1)
    top_indices = sentence_scores.argsort(axis=0)[-num_sentences:]
    summary = '. '.join([sentences[i[0]] for i in sorted(top_indices)])
    return summary

text = "TF-IDF is an important concept in NLP. It helps identify relevant words in documents. TF-IDF is widely used in search engines and recommendation systems. It can also be applied to document classification and clustering."
summary = summarize(text)
print("Summary:", summary)
```

Slide 13: TF-IDF for Keyword Extraction

TF-IDF scores can be used to extract keywords from a document.

```python
def extract_keywords(text, num_keywords=5):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
    sorted_scores = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_scores[:num_keywords]]

text = "TF-IDF is a numerical statistic used to reflect the importance of a word in a document within a collection or corpus. It's commonly used in information retrieval and text mining tasks."
keywords = extract_keywords(text)
print("Extracted keywords:", keywords)
```

Slide 14: Additional Resources

1. "TF-IDF for Document Ranking in Information Retrieval: A Survey" by Charu C. Aggarwal and ChengXiang Zhai arXiv:1905.07405 \[cs.IR\] [https://arxiv.org/abs/1905.07405](https://arxiv.org/abs/1905.07405)
2. "A Comparative Study of TF-IDF Algorithms in Text Classification" by Yun-tao Zhang, Ling Gong, and Yong-cheng Wang arXiv:1905.07405 \[cs.CL\] [https://arxiv.org/abs/1708.03851](https://arxiv.org/abs/1708.03851)
3. "Improvements to TF-IDF for Text Classification" by Xiaoying Liu, Huiling Wang, and Xiangdong Zhou arXiv:2010.02539 \[cs.IR\] [https://arxiv.org/abs/2010.02539](https://arxiv.org/abs/2010.02539)

