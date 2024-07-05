## Sentiment Analysis in Python A Hands-On Guide
Slide 1: 

Introduction to Sentiment Analysis

Sentiment analysis, also known as opinion mining, is a natural language processing (NLP) technique that involves analyzing and categorizing subjective information from text data. It aims to determine the sentiment or opinion expressed in a piece of text, whether it is positive, negative, or neutral. Sentiment analysis has numerous applications, such as monitoring social media sentiment, analyzing customer feedback, and understanding public opinion on various topics.

```python
import nltk

# Sample text
text = "I really enjoyed the movie. The acting was superb, and the plot kept me engaged throughout."

# Tokenize the text
tokens = nltk.word_tokenize(text)

# Initialize a sentiment intensity analyzer
sia = nltk.sentiment.vader.SentimentIntensityAnalyzer()

# Calculate the sentiment scores
sentiment_scores = sia.polarity_scores(text)

# Print the sentiment scores
print(sentiment_scores)
```

Slide 2: 

Importing Required Libraries

Before starting with sentiment analysis, you need to import the necessary libraries. The Natural Language Toolkit (NLTK) is a popular Python library for natural language processing tasks, including sentiment analysis. Additionally, you may need to install and download specific NLTK resources for sentiment analysis.

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
```

Slide 3: 

Text Preprocessing

Text preprocessing is an essential step in sentiment analysis. It involves cleaning and preparing the text data for analysis. Common preprocessing steps include tokenization, removing stop words, stemming or lemmatization, and handling special characters or punctuation.

```python
# Sample text
text = "I really enjoyed the movie, but the ending was disappointing."

# Tokenize the text
tokens = nltk.word_tokenize(text)

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Perform stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

print("Tokenized text:", tokens)
print("Filtered tokens:", filtered_tokens)
print("Stemmed tokens:", stemmed_tokens)
```

Slide 4: 

Bag-of-Words (BoW) Approach

The bag-of-words (BoW) approach is a simple and widely used technique for representing text data in sentiment analysis. It involves counting the occurrences of each word in the text and creating a feature vector based on these counts.

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample text data
corpus = [
    "This movie was amazing! I loved the acting and the plot.",
    "The customer service was terrible. I will never use this product again.",
    "I had a great experience with this product. Highly recommended."
]

# Create a bag-of-words representation
vectorizer = CountVectorizer()
bow_repr = vectorizer.fit_transform(corpus)

# Print the feature names (vocabulary)
print("Feature names:", vectorizer.get_feature_names_out())

# Print the bag-of-words representation
print("Bag-of-words representation:\n", bow_repr.toarray())
```

Slide 5: 

TF-IDF (Term Frequency-Inverse Document Frequency)

The TF-IDF (Term Frequency-Inverse Document Frequency) is another popular technique for representing text data in sentiment analysis. It considers not only the frequency of a word in a document but also its importance across the entire corpus.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
corpus = [
    "This movie was amazing! I loved the acting and the plot.",
    "The customer service was terrible. I will never use this product again.",
    "I had a great experience with this product. Highly recommended."
]

# Create a TF-IDF representation
vectorizer = TfidfVectorizer()
tfidf_repr = vectorizer.fit_transform(corpus)

# Print the feature names (vocabulary)
print("Feature names:", vectorizer.get_feature_names_out())

# Print the TF-IDF representation
print("TF-IDF representation:\n", tfidf_repr.toarray())
```

Slide 6: 

Naive Bayes Classifier

The Naive Bayes classifier is a popular machine learning algorithm for sentiment analysis. It is based on Bayes' theorem and makes the assumption that features are independent of each other, which simplifies the calculations. Despite this naive assumption, it often performs well in practice.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
train_data = [
    ("This movie was amazing! I loved the acting and the plot.", "positive"),
    ("The customer service was terrible. I will never use this product again.", "negative"),
    ("I had a great experience with this product. Highly recommended.", "positive")
]

# Split the data into text and labels
train_texts, train_labels = zip(*train_data)

# Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the pipeline
pipeline.fit(train_texts, train_labels)

# Test the model
test_text = "This product is amazing! I highly recommend it."
prediction = pipeline.predict([test_text])[0]
print("Predicted sentiment:", prediction)
```

Slide 7: 

Logistic Regression

Logistic regression is another popular machine learning algorithm for sentiment analysis. It models the probability of a text belonging to a particular sentiment class (positive or negative) based on the feature values.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
train_data = [
    ("This movie was amazing! I loved the acting and the plot.", "positive"),
    ("The customer service was terrible. I will never use this product again.", "negative"),
    ("I had a great experience with this product. Highly recommended.", "positive")
]

# Split the data into text and labels
train_texts, train_labels = zip(*train_data)

# Create a pipeline with TF-IDF vectorizer and Logistic Regression
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Train the pipeline
pipeline.fit(train_texts, train_labels)

# Test the model
test_text = "This product is amazing! I highly recommend it."
prediction = pipeline.predict([test_text])[0]
print("Predicted sentiment:", prediction)
```

Slide 8: 

Support Vector Machines (SVM)

Support Vector Machines (SVM) are a powerful machine learning algorithm that can be used for sentiment analysis. SVMs construct a hyperplane in a high-dimensional space to separate the classes (positive and negative sentiment) with the maximum margin.

```python
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
train_data = [
    ("This movie was amazing! I loved the acting and the plot.", "positive"),
    ("The customer service was terrible. I will never use this product again.", "negative"),
    ("I had a great experience with this product. Highly recommended.", "positive")
]

# Split the data into text and labels
train_texts, train_labels = zip(*train_data)

# Create a pipeline with TF-IDF vectorizer and Linear SVM
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LinearSVC())
])

# Train the pipeline
pipeline.fit(train_texts, train_labels)

# Test the model
test_text = "This product is amazing! I highly recommend it."
prediction = pipeline.predict([test_text])[0]
print("Predicted sentiment:", prediction)
```

Slide 9: 

Word Embeddings

Word embeddings are a technique for representing words as dense vectors in a continuous vector space. These vectors capture semantic and syntactic relationships between words, making them useful for various NLP tasks, including sentiment analysis. Popular word embedding models include Word2Vec and GloVe.

```python
import gensim

# Load pre-trained Word2Vec model
model = gensim.models.KeyedVectors.load_word2vec_format('path/to/googlenews-vectors-negative300.bin.gz', binary=True)

# Get the word vector for 'amazing'
amazing_vector = model.wv['amazing']
print("Vector for 'amazing':", amazing_vector)

# Find the most similar words to 'amazing'
similar_words = model.wv.most_similar(positive=['amazing'], topn=5)
print("Most similar words to 'amazing':", similar_words)
```

Slide 10: 

Recurrent Neural Networks (RNNs) for Sentiment Analysis

Recurrent Neural Networks (RNNs) are a type of neural network architecture well-suited for sequential data like text. They can capture long-range dependencies and are commonly used for sentiment analysis tasks. Popular variants of RNNs include Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU).

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Sample text data
texts = [
    "This movie was amazing! I loved the acting and the plot.",
    "The customer service was terrible. I will never use this product again.",
    "I had a great experience with this product. Highly recommended."
]

# Tokenize the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=100)

# Define the model
model = Sequential([
    Embedding(len(tokenizer.word_index) + 1, 100),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, [0, 1, 0], epochs=5, batch_size=1)
```

Slide 11: 

Convolutional Neural Networks (CNNs) for Sentiment Analysis

Convolutional Neural Networks (CNNs) are another type of neural network architecture that has been successfully applied to sentiment analysis tasks. CNNs are particularly effective at capturing local patterns and features in text data, making them well-suited for tasks like sentiment classification.

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

# Sample text data
texts = [
    "This movie was amazing! I loved the acting and the plot.",
    "The customer service was terrible. I will never use this product again.",
    "I had a great experience with this product. Highly recommended."
]

# Tokenize the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=100)

# Define the model
model = Sequential([
    Embedding(len(tokenizer.word_index) + 1, 100),
    Conv1D(64, 3, padding='same', activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, [0, 1, 0], epochs=5, batch_size=1)
```

Slide 12: 

Transfer Learning for Sentiment Analysis

Transfer learning is a technique where a pre-trained model on a large dataset is fine-tuned on a specific task, such as sentiment analysis. This approach can significantly improve performance, especially when the target task has limited training data. Popular pre-trained language models like BERT, RoBERTa, and GPT can be fine-tuned for sentiment analysis tasks.

```python
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer

# Load pre-trained BERT model and tokenizer
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sample text data
texts = [
    "This movie was amazing! I loved the acting and the plot.",
    "The customer service was terrible. I will never use this product again.",
    "I had a great experience with this product. Highly recommended."
]

# Tokenize and encode the texts
input_ids = []
attention_masks = []
for text in texts:
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )
    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])

# Fine-tune the model
model.train_model(
    input_ids=input_ids,
    attention_mask=attention_masks,
    labels=[0, 1, 0]  # Replace with your labels
)

# Evaluate the model
output = model(input_ids, attention_mask=attention_masks)
predictions = output.logits.argmax(-1)
print("Predictions:", predictions)
```

