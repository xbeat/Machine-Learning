## Bag of Words in NLP using Python
Slide 1: Introduction to Bag of Words (BoW) in NLP

Bag of Words is a fundamental technique in Natural Language Processing that represents text as a collection of words, disregarding grammar and word order. This method is used to create feature vectors for text classification, sentiment analysis, and information retrieval tasks.

```python
from collections import Counter

text = "The quick brown fox jumps over the lazy dog"
bow = Counter(text.lower().split())
print(bow)
```

Slide 2: Tokenization: The First Step

Tokenization is the process of breaking down text into individual words or tokens. It's a crucial step in creating a Bag of Words representation. We'll use the NLTK library for more advanced tokenization.

```python
import nltk
nltk.download('punkt')

text = "The quick brown fox, jumps over the lazy dog!"
tokens = nltk.word_tokenize(text)
print(tokens)
```

Slide 3: Creating a Vocabulary

After tokenization, we need to create a vocabulary of unique words. This vocabulary will be used to create our feature vectors.

```python
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog sleeps all day",
    "The quick rabbit runs fast"
]

vocabulary = set()
for sentence in corpus:
    vocabulary.update(sentence.lower().split())

print(f"Vocabulary size: {len(vocabulary)}")
print(f"Vocabulary: {vocabulary}")
```

Slide 4: Encoding Text as BoW Vectors

Once we have our vocabulary, we can encode each text as a vector of word frequencies.

```python
def bow_encoding(text, vocabulary):
    vector = {word: 0 for word in vocabulary}
    for word in text.lower().split():
        if word in vector:
            vector[word] += 1
    return vector

vocabulary = list(vocabulary)  # Convert set to list for consistent ordering
encoded_texts = [bow_encoding(text, vocabulary) for text in corpus]

for i, encoded_text in enumerate(encoded_texts):
    print(f"Text {i + 1}: {encoded_text}")
```

Slide 5: Handling Stop Words

Stop words are common words that often don't contribute much to the meaning of a text. Removing them can improve the performance of NLP models.

```python
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

text = "The quick brown fox jumps over the lazy dog"
tokens = nltk.word_tokenize(text.lower())
filtered_tokens = [word for word in tokens if word not in stop_words]

print(f"Original: {tokens}")
print(f"Filtered: {filtered_tokens}")
```

Slide 6: Stemming and Lemmatization

Stemming and lemmatization reduce words to their base or root form, which can help in creating more meaningful BoW representations.

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "runs", "ran", "easily", "fairly"]

print("Original | Stemmed | Lemmatized")
for word in words:
    print(f"{word:9} | {stemmer.stem(word):7} | {lemmatizer.lemmatize(word)}")
```

Slide 7: TF-IDF: Improving BoW

Term Frequency-Inverse Document Frequency (TF-IDF) is an improvement over simple BoW. It considers the importance of words across the entire corpus.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog sleeps all day",
    "The quick rabbit runs fast"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print("TF-IDF matrix:")
print(X.toarray())
print("\nFeature names:")
print(vectorizer.get_feature_names_out())
```

Slide 8: Implementing BoW for Text Classification

Let's use BoW for a simple text classification task using the Naive Bayes classifier.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Sample data
X = [
    "I love this product",
    "This is terrible",
    "Great customer service",
    "Poor quality",
    "Excellent experience"
]
y = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create BoW representation
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train and evaluate the model
clf = MultinomialNB()
clf.fit(X_train_bow, y_train)
print(f"Accuracy: {clf.score(X_test_bow, y_test)}")
```

Slide 9: Visualizing BoW with Word Clouds

Word clouds provide a visual representation of word frequencies in a corpus, which can be useful for understanding the most common terms in a BoW model.

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = "The quick brown fox jumps over the lazy dog. " * 10
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud Representation of BoW')
plt.show()
```

Slide 10: Handling Out-of-Vocabulary Words

When applying a BoW model to new text, we may encounter words that weren't in our original vocabulary. Let's explore how to handle this situation.

```python
def bow_encoding(text, vocabulary):
    vector = {word: 0 for word in vocabulary}
    for word in text.lower().split():
        if word in vector:
            vector[word] += 1
        else:
            vector['<UNK>'] = vector.get('<UNK>', 0) + 1
    return vector

vocabulary = set(['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog', '<UNK>'])
new_text = "The fast red fox leaps over the sleepy cat"

encoded_text = bow_encoding(new_text, vocabulary)
print(encoded_text)
```

Slide 11: N-grams: Capturing Word Order

N-grams extend the BoW model by considering sequences of N words, which can capture some word order information.

```python
from nltk import ngrams

text = "The quick brown fox jumps over the lazy dog"
tokens = text.split()

print("Unigrams:", list(ngrams(tokens, 1)))
print("Bigrams:", list(ngrams(tokens, 2)))
print("Trigrams:", list(ngrams(tokens, 3)))
```

Slide 12: Real-Life Example: Spam Detection

Let's use BoW for a practical spam detection task using a public dataset.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load the SMS Spam Collection dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
data = pd.read_csv(url, sep='\t', names=['label', 'message'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Create BoW representation
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train and evaluate the model
clf = MultinomialNB()
clf.fit(X_train_bow, y_train)
y_pred = clf.predict(X_test_bow)

print(classification_report(y_test, y_pred))
```

Slide 13: Real-Life Example: Document Similarity

BoW can be used to measure similarity between documents, which is useful for tasks like recommendation systems or plagiarism detection.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "The cat sits on the mat",
    "The dog jumps over the fence",
    "The cat chases the mouse",
    "The bird flies in the sky"
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)

similarity_matrix = cosine_similarity(bow_matrix)

print("Document Similarity Matrix:")
print(similarity_matrix)

# Find the most similar pair of documents
max_similarity = 0
max_pair = None
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        if similarity_matrix[i][j] > max_similarity:
            max_similarity = similarity_matrix[i][j]
            max_pair = (i, j)

print(f"\nMost similar documents: {max_pair}")
print(f"Similarity score: {max_similarity}")
print(f"Doc 1: {documents[max_pair[0]]}")
print(f"Doc 2: {documents[max_pair[1]]}")
```

Slide 14: Limitations of Bag of Words

While BoW is simple and effective, it has limitations:

1. Loss of word order information
2. Inability to capture semantics
3. High dimensionality for large vocabularies
4. Sensitivity to vocabulary choice

These limitations have led to the development of more advanced techniques like word embeddings (e.g., Word2Vec, GloVe) and transformer-based models (e.g., BERT, GPT).

```python
# Demonstrating loss of word order
sentence1 = "The cat chases the mouse"
sentence2 = "The mouse chases the cat"

bow1 = Counter(sentence1.lower().split())
bow2 = Counter(sentence2.lower().split())

print("BoW for sentence 1:", bow1)
print("BoW for sentence 2:", bow2)
print("Are the BoW representations identical?", bow1 == bow2)
```

Slide 15: Additional Resources

For further exploration of Bag of Words and related NLP techniques, consider the following resources:

1. "Efficient Estimation of Word Representations in Vector Space" by Mikolov et al. (2013) ArXiv: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
2. "GloVe: Global Vectors for Word Representation" by Pennington et al. (2014) ArXiv: [https://arxiv.org/abs/1405.4053](https://arxiv.org/abs/1405.4053)
3. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

These papers introduce more advanced techniques that address some of the limitations of the basic Bag of Words model.

