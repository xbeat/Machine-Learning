## Understanding TF-IDF! A Key Text Analysis Technique in Python
Slide 1: Understanding TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic used to reflect the importance of a word in a document within a collection or corpus. It's widely used in information retrieval and text mining.

```python
import math
from collections import Counter

def tf_idf(term, document, corpus):
    # Calculate term frequency
    tf = document.count(term) / len(document)
    
    # Calculate inverse document frequency
    idf = math.log(len(corpus) / sum(1 for doc in corpus if term in doc))
    
    return tf * idf

# Example usage
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog sleeps all day",
    "The quick brown fox is quick and brown"
]

term = "quick"
document = corpus[0].split()

result = tf_idf(term, document, corpus)
print(f"TF-IDF score for '{term}': {result}")
```

Slide 2: Term Frequency (TF)

Term Frequency measures how frequently a term occurs in a document. It's calculated by dividing the number of times a term appears in a document by the total number of terms in that document.

```python
def term_frequency(term, document):
    return document.count(term) / len(document)

# Example usage
document = "The quick brown fox jumps over the lazy dog".split()
term = "quick"

tf = term_frequency(term, document)
print(f"Term Frequency of '{term}': {tf}")
```

Slide 3: Inverse Document Frequency (IDF)

IDF measures how important a term is across the entire corpus. It's calculated as the logarithm of the total number of documents divided by the number of documents containing the term.

```python
import math

def inverse_document_frequency(term, corpus):
    num_documents = len(corpus)
    num_documents_with_term = sum(1 for doc in corpus if term in doc)
    return math.log(num_documents / num_documents_with_term)

# Example usage
corpus = [
    "The quick brown fox jumps over the lazy dog".split(),
    "The lazy dog sleeps all day".split(),
    "The quick brown fox is quick and brown".split()
]
term = "quick"

idf = inverse_document_frequency(term, corpus)
print(f"Inverse Document Frequency of '{term}': {idf}")
```

Slide 4: Combining TF and IDF

The TF-IDF score is calculated by multiplying the term frequency by the inverse document frequency. This combination helps to balance the importance of a term within a document and across the corpus.

```python
def tf_idf(term, document, corpus):
    tf = term_frequency(term, document)
    idf = inverse_document_frequency(term, corpus)
    return tf * idf

# Example usage
corpus = [
    "The quick brown fox jumps over the lazy dog".split(),
    "The lazy dog sleeps all day".split(),
    "The quick brown fox is quick and brown".split()
]
document = corpus[0]
term = "quick"

score = tf_idf(term, document, corpus)
print(f"TF-IDF score for '{term}': {score}")
```

Slide 5: Implementing TF-IDF with scikit-learn

Scikit-learn provides an efficient implementation of TF-IDF through its TfidfVectorizer class. This class converts a collection of raw documents to a matrix of TF-IDF features.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog sleeps all day",
    "The quick brown fox is quick and brown"
]

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(corpus)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Print TF-IDF scores for each document
for i, doc in enumerate(tfidf_matrix.toarray()):
    print(f"Document {i+1}:")
    for j, score in enumerate(doc):
        if score > 0:
            print(f"  {feature_names[j]}: {score:.4f}")
```

Slide 6: Visualizing TF-IDF Scores

Visualizing TF-IDF scores can help in understanding the importance of different terms across documents. Let's create a heatmap to visualize the TF-IDF scores.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming we have the tfidf_matrix and feature_names from the previous slide

# Convert sparse matrix to dense array
tfidf_dense = tfidf_matrix.toarray()

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(tfidf_dense, annot=True, cmap='YlOrRd', xticklabels=feature_names, yticklabels=[f'Doc {i+1}' for i in range(len(corpus))])
plt.title('TF-IDF Scores Heatmap')
plt.tight_layout()
plt.show()
```

Slide 7: Document Similarity with TF-IDF

TF-IDF can be used to measure document similarity. By representing documents as TF-IDF vectors, we can compute their cosine similarity to find how similar they are.

```python
from sklearn.metrics.pairwise import cosine_similarity

# Assuming we have the tfidf_matrix from previous slides

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Print similarity matrix
print("Document Similarity Matrix:")
print(cosine_sim)

# Find most similar document pairs
for i in range(len(corpus)):
    for j in range(i+1, len(corpus)):
        print(f"Similarity between Document {i+1} and Document {j+1}: {cosine_sim[i][j]:.4f}")
```

Slide 8: Keyword Extraction with TF-IDF

TF-IDF is useful for extracting important keywords from documents. Words with high TF-IDF scores are likely to be important or characteristic of the document.

```python
import numpy as np

def extract_keywords(tfidf_matrix, feature_names, doc_index, top_n=5):
    # Get the TF-IDF scores for the specified document
    doc_tfidf = tfidf_matrix[doc_index].toarray()[0]
    
    # Sort the scores in descending order and get the top N indices
    top_indices = doc_tfidf.argsort()[::-1][:top_n]
    
    # Return the top N keywords and their scores
    return [(feature_names[i], doc_tfidf[i]) for i in top_indices]

# Assuming we have tfidf_matrix and feature_names from previous slides

# Extract top 5 keywords for each document
for i in range(len(corpus)):
    print(f"Top keywords for Document {i+1}:")
    keywords = extract_keywords(tfidf_matrix, feature_names, i)
    for word, score in keywords:
        print(f"  {word}: {score:.4f}")
```

Slide 9: Text Classification with TF-IDF

TF-IDF is often used as a feature extraction method in text classification tasks. Let's use it with a simple classifier to categorize text.

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Sample dataset
texts = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outpaces a quick fox",
    "The lazy cat sleeps all day",
    "Lazy dogs and cats nap in the sun",
    "Foxes are quick and cunning animals"
]
labels = [0, 0, 1, 1, 0]  # 0 for fox-related, 1 for lazy animal

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = clf.predict(X_test_tfidf)

# Print classification report
print(classification_report(y_test, y_pred))
```

Slide 10: TF-IDF for Text Summarization

TF-IDF can be used in extractive text summarization by identifying the most important sentences in a document based on the TF-IDF scores of their words.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def summarize_text(text, num_sentences=3):
    # Split the text into sentences
    sentences = text.split('.')
    sentences = [sent.strip() for sent in sentences if sent.strip()]

    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Calculate sentence scores
    sentence_scores = tfidf_matrix.sum(axis=1).A1

    # Get indices of top sentences
    top_indices = sentence_scores.argsort()[-num_sentences:][::-1]

    # Return summary
    summary = '. '.join([sentences[i] for i in sorted(top_indices)])
    return summary + '.'

# Example usage
text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. Many different classes of machine learning algorithms have been applied to natural language processing tasks. These algorithms take as input a large set of "features" that are generated from the input data. Some of the earliest-used algorithms, such as decision trees, produced systems of hard if-then rules similar to existing hand-written rules. However, part-of-speech tagging introduced the use of hidden Markov models to natural language processing, and increasingly, research has focused on statistical models, which make soft, probabilistic decisions based on attaching real-valued weights to the features making up the input data.
"""

summary = summarize_text(text)
print("Summary:")
print(summary)
```

Slide 11: TF-IDF for Information Retrieval

TF-IDF is widely used in information retrieval systems, such as search engines, to rank documents based on their relevance to a query.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def search_documents(query, documents):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform documents
    doc_vectors = vectorizer.fit_transform(documents)
    
    # Transform query
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity between query and documents
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    
    # Sort documents by similarity
    ranked_docs = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    
    return ranked_docs

# Example usage
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outpaces a quick fox",
    "The lazy cat sleeps all day",
    "Lazy dogs and cats nap in the sun",
    "Foxes are quick and cunning animals"
]

query = "quick animals"

results = search_documents(query, documents)

print("Search Results:")
for i, score in results:
    print(f"Document {i+1} (Score: {score:.4f}): {documents[i]}")
```

Slide 12: Real-Life Example: Spam Detection

TF-IDF is commonly used in email spam detection systems. Let's implement a simple spam detector using TF-IDF and a Naive Bayes classifier.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample dataset (email subjects and labels)
subjects = [
    "Get rich quick! Guaranteed results!",
    "Meeting schedule for next week",
    "Lose weight fast with this one trick",
    "Project update: deadline approaching",
    "You've won a free vacation!",
    "Reminder: submit your expense reports"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for not spam

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(subjects, labels, test_size=0.3, random_state=42)

# Create and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = clf.predict(X_test_tfidf)

# Print classification report
print("Spam Detection Results:")
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

# Test with new emails
new_emails = [
    "Congratulations! You've been selected for a special offer!",
    "Team meeting rescheduled to 3 PM tomorrow"
]
new_emails_tfidf = vectorizer.transform(new_emails)
predictions = clf.predict(new_emails_tfidf)

print("\nPredictions for new emails:")
for email, prediction in zip(new_emails, predictions):
    print(f"Email: {email}")
    print(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}\n")
```

Slide 13: Real-Life Example: Content-Based Recommendation System

TF-IDF can be used in content-based recommendation systems to suggest items based on their textual descriptions. Let's create a simple movie recommender.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie database
movies = {
    "The Shawshank Redemption": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
    "The Godfather": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
    "The Dark Knight": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
    "12 Angry Men": "A jury holdout attempts to prevent a miscarriage of justice by forcing his colleagues to reconsider the evidence.",
    "Schindler's List": "In German-occupied Poland during World War II, industrialist Oskar Schindler gradually becomes concerned for his Jewish workforce after witnessing their persecution by the Nazis."
}

def recommend_movies(liked_movie, movies_db, top_n=3):
    vectorizer = TfidfVectorizer()
    movie_vectors = vectorizer.fit_transform(movies_db.values())
    liked_movie_vector = movie_vectors[list(movies_db.keys()).index(liked_movie)]
    
    similarities = cosine_similarity(liked_movie_vector, movie_vectors).flatten()
    similar_indices = similarities.argsort()[::-1][1:top_n+1]
    
    return [list(movies_db.keys())[i] for i in similar_indices]

# Example usage
liked_movie = "The Dark Knight"
recommendations = recommend_movies(liked_movie, movies)
print(f"If you liked '{liked_movie}', you might also enjoy:")
for movie in recommendations:
    print(f"- {movie}")
```

Slide 14: Limitations and Considerations of TF-IDF

While TF-IDF is a powerful technique, it has some limitations to consider:

1. Ignores word order and context
2. Doesn't capture semantics or meaning
3. May give high scores to rare but irrelevant terms

To address these issues, more advanced techniques like word embeddings (e.g., Word2Vec) or transformer models (e.g., BERT) can be used in conjunction with or as alternatives to TF-IDF.


Slide 15: Limitations and Considerations of TF-IDF

```python
# Pseudocode for potential improvements

# 1. Combine TF-IDF with Word2Vec
def improved_vector(document):
    tfidf_vector = compute_tfidf(document)
    word2vec_vector = compute_word2vec(document)
    return combine_vectors(tfidf_vector, word2vec_vector)

# 2. Use TF-IDF as input features for a neural network
def train_neural_network(documents, labels):
    tfidf_features = compute_tfidf_for_corpus(documents)
    model = create_neural_network()
    model.train(tfidf_features, labels)
    return model

# 3. Implement a custom IDF formula
def custom_idf(term, corpus):
    # Implement a more sophisticated IDF calculation
    # that takes into account term relevance or domain knowledge
    pass
```

Slide 16: Additional Resources

For those interested in diving deeper into TF-IDF and related techniques, here are some valuable resources:

1. "Introduction to Information Retrieval" by Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze ArXiv: [https://arxiv.org/abs/cs/0412098](https://arxiv.org/abs/cs/0412098)
2. "Feature extraction and machine learning for information retrieval" by Yiming Yang ArXiv: [https://arxiv.org/abs/cs/9811006](https://arxiv.org/abs/cs/9811006)
3. "A Study of Smoothing Methods for Language Models Applied to Information Retrieval" by Chengxiang Zhai and John Lafferty ArXiv: [https://arxiv.org/abs/cs/0308031](https://arxiv.org/abs/cs/0308031)

These resources provide in-depth explanations of TF-IDF, its applications, and advanced techniques in information retrieval and text analysis.

