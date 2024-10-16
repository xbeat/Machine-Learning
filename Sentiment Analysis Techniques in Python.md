## Sentiment Analysis Techniques in Python
Slide 1: Introduction to Sentiment Analysis

Sentiment Analysis is a natural language processing technique used to determine the emotional tone behind a piece of text. It's widely used in social media monitoring, customer feedback analysis, and market research. This slideshow will cover various methods of sentiment analysis, from basic to advanced, using Python.

```python
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Example text
text = "I love this product! It's amazing and works perfectly."

# Perform sentiment analysis
sentiment_scores = sia.polarity_scores(text)

print(f"Sentiment scores: {sentiment_scores}")
print(f"Overall sentiment: {'Positive' if sentiment_scores['compound'] > 0 else 'Negative' if sentiment_scores['compound'] < 0 else 'Neutral'}")
```

Slide 2: Tokenization and Basic Preprocessing

Tokenization is the process of breaking down text into individual words or tokens. It's a crucial step in sentiment analysis as it allows us to analyze the text at a granular level. We'll also cover basic preprocessing techniques such as lowercasing and removing punctuation.

```python
import re
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    return tokens

# Example usage
text = "Hello! How are you doing today? It's a beautiful day!"
preprocessed_tokens = preprocess_text(text)
print(f"Original text: {text}")
print(f"Preprocessed tokens: {preprocessed_tokens}")
```

Slide 3: Bag of Words (BoW) Representation

The Bag of Words model is a simple yet effective method for representing text data. It creates a vocabulary of all unique words in the corpus and represents each document as a vector of word frequencies.

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "I love this movie",
    "This movie is awful",
    "The acting is great"
]

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Fit and transform the documents
bow_matrix = vectorizer.fit_transform(documents)

# Get the vocabulary
vocabulary = vectorizer.get_feature_names_out()

print("Vocabulary:", vocabulary)
print("BoW matrix:\n", bow_matrix.toarray())
```

Slide 4: TF-IDF Representation

TF-IDF (Term Frequency-Inverse Document Frequency) is an improvement over the simple Bag of Words model. It considers both the frequency of a word in a document and its importance across the entire corpus.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "I love this movie",
    "This movie is awful",
    "The acting is great"
]

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the vocabulary
vocabulary = vectorizer.get_feature_names_out()

print("Vocabulary:", vocabulary)
print("TF-IDF matrix:\n", tfidf_matrix.toarray())
```

Slide 5: Naive Bayes Classifier for Sentiment Analysis

Naive Bayes is a probabilistic classifier often used for sentiment analysis due to its simplicity and effectiveness. We'll use the MultinomialNB classifier from scikit-learn to classify movie reviews as positive or negative.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Sample data
reviews = [
    "This movie is fantastic",
    "I hated this film",
    "Great acting and plot",
    "Terrible screenplay and direction"
]
labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# Split the data
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# Create a CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Make predictions
predictions = clf.predict(X_test_vec)
print("Predictions:", predictions)
```

Slide 6: Real-life Example: Analyzing Product Reviews

Let's apply our Naive Bayes classifier to analyze sentiment in product reviews. This example demonstrates how sentiment analysis can be used to gauge customer satisfaction and identify areas for improvement.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample product review data
reviews = [
    "This smartphone is amazing! Great battery life and camera quality.",
    "The laptop keeps crashing. Terrible customer support.",
    "Love the new headphones. Crystal clear sound and comfortable fit.",
    "Disappointed with the tablet. Slow performance and frequent glitches.",
    "The smartwatch exceeded my expectations. Accurate fitness tracking and stylish design."
]
sentiments = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

# Create a DataFrame
df = pd.DataFrame({'review': reviews, 'sentiment': sentiments})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the classifier
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Make predictions
y_pred = clf.predict(X_test_vec)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Analyze a new review
new_review = "The camera quality is excellent, but the battery life is disappointing."
new_review_vec = vectorizer.transform([new_review])
prediction = clf.predict(new_review_vec)
print(f"\nSentiment of new review: {'Positive' if prediction[0] == 1 else 'Negative'}")
```

Slide 7: Word Embeddings: Word2Vec

Word embeddings are dense vector representations of words that capture semantic relationships. Word2Vec is a popular method for creating word embeddings. We'll use the Gensim library to train a Word2Vec model on our corpus.

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Sample corpus
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be that is the question",
    "All that glitters is not gold"
]

# Tokenize the corpus
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Find similar words
similar_words = model.wv.most_similar("journey", topn=3)
print("Words similar to 'journey':", similar_words)

# Get vector for a word
word_vector = model.wv["fox"]
print("Vector for 'fox':", word_vector[:5])  # Showing first 5 dimensions
```

Slide 8: Recurrent Neural Networks (RNN) for Sentiment Analysis

RNNs are a class of neural networks designed to work with sequential data, making them suitable for natural language processing tasks like sentiment analysis. We'll use a simple RNN model with TensorFlow and Keras.

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
texts = [
    "This movie is great",
    "I hated this film",
    "Amazing performance by the actors",
    "Terrible screenplay and direction"
]
labels = np.array([1, 0, 1, 0])  # 1 for positive, 0 for negative

# Tokenize the texts
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
padded_sequences = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')

# Build the model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=10),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=10, verbose=0)

# Make predictions
new_texts = ["This movie is fantastic", "I didn't enjoy this film at all"]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=10, padding='post', truncating='post')
predictions = model.predict(new_padded)

for text, pred in zip(new_texts, predictions):
    print(f"Text: {text}")
    print(f"Sentiment: {'Positive' if pred > 0.5 else 'Negative'}")
    print(f"Confidence: {pred[0]:.2f}")
    print()
```

Slide 9: Long Short-Term Memory (LSTM) Networks

LSTMs are a type of RNN that can capture long-term dependencies in sequential data. They are particularly effective for sentiment analysis tasks involving longer texts.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample data
texts = [
    "This movie is absolutely fantastic. The plot is engaging and the acting is superb.",
    "I was thoroughly disappointed with this film. The story was confusing and the pacing was off.",
    "An incredible cinematic experience. The visuals are breathtaking and the soundtrack is mesmerizing.",
    "This movie was a complete waste of time. The characters were poorly developed and the dialogue was cringeworthy."
]
labels = np.array([1, 0, 1, 0])  # 1 for positive, 0 for negative

# Tokenize the texts
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
padded_sequences = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')

# Build the LSTM model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=50),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=50, verbose=0)

# Make predictions
new_texts = [
    "This film exceeded all my expectations. A true masterpiece.",
    "I couldn't even finish watching this movie. It was that bad."
]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=50, padding='post', truncating='post')
predictions = model.predict(new_padded)

for text, pred in zip(new_texts, predictions):
    print(f"Text: {text}")
    print(f"Sentiment: {'Positive' if pred > 0.5 else 'Negative'}")
    print(f"Confidence: {pred[0]:.2f}")
    print()
```

Slide 10: Bidirectional LSTM for Enhanced Context

Bidirectional LSTMs process input sequences in both forward and backward directions, allowing the network to capture context from both past and future tokens. This can lead to improved performance in sentiment analysis tasks.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample data
texts = [
    "Despite its flaws, the movie manages to be entertaining and thought-provoking.",
    "While the visuals are stunning, the plot is convoluted and hard to follow.",
    "The film starts slow but builds to a satisfying and emotional conclusion.",
    "Although well-acted, the movie suffers from pacing issues and a weak script."
]
labels = np.array([1, 0, 1, 0])  # 1 for positive, 0 for negative

# Tokenize the texts
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
padded_sequences = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')

# Build the Bidirectional LSTM model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=50),
    Bidirectional(LSTM(32)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=50, verbose=0)

# Make predictions
new_texts = [
    "The movie has its moments, but ultimately fails to deliver on its promising premise.",
    "A cinematic triumph that pushes the boundaries of storytelling and visual effects."
]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=50, padding='post', truncating='post')
predictions = model.predict(new_padded)

for text, pred in zip(new_texts, predictions):
    print(f"Text: {text}")
    print(f"Sentiment: {'Positive' if pred > 0.5 else 'Negative'}")
    print(f"Confidence: {pred[0]:.2f}")
    print()
```

Slide 11: Attention Mechanism for Sentiment Analysis

The attention mechanism allows the model to focus on specific parts of the input sequence when making predictions. This can be particularly useful for sentiment analysis, as it helps the model identify key phrases or words that strongly influence the overall sentiment.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dot, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample data
texts = [
    "The movie's stunning visuals can't make up for its weak plot and poor character development.",
    "An emotional rollercoaster with brilliant performances and a touching storyline.",
    "Despite a strong start, the film loses its way in the second half and ends disappointingly.",
    "A masterpiece of modern cinema that will be remembered for years to come."
]
labels = np.array([0, 1,
```

## Response:
Slide 10: Bidirectional LSTM for Enhanced Context

Bidirectional LSTMs process input sequences in both forward and backward directions, allowing the network to capture context from both past and future tokens. This can lead to improved performance in sentiment analysis tasks.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample data
texts = [
    "Despite its flaws, the movie manages to be entertaining and thought-provoking.",
    "While the visuals are stunning, the plot is convoluted and hard to follow.",
    "The film starts slow but builds to a satisfying and emotional conclusion.",
    "Although well-acted, the movie suffers from pacing issues and a weak script."
]
labels = np.array([1, 0, 1, 0])  # 1 for positive, 0 for negative

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')

# Build the Bidirectional LSTM model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=50),
    Bidirectional(LSTM(32)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=50, verbose=0)

# Make predictions
new_texts = [
    "The movie has its moments, but ultimately fails to deliver on its promising premise.",
    "A cinematic triumph that pushes the boundaries of storytelling and visual effects."
]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=50, padding='post', truncating='post')
predictions = model.predict(new_padded)

for text, pred in zip(new_texts, predictions):
    print(f"Text: {text}")
    print(f"Sentiment: {'Positive' if pred > 0.5 else 'Negative'}")
    print(f"Confidence: {pred[0]:.2f}")
    print()
```

Slide 11: Attention Mechanism for Sentiment Analysis

The attention mechanism allows the model to focus on specific parts of the input sequence when making predictions. This can be particularly useful for sentiment analysis, as it helps the model identify key phrases or words that strongly influence the overall sentiment.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dot, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample data
texts = [
    "The movie's stunning visuals can't make up for its weak plot and poor character development.",
    "An emotional rollercoaster with brilliant performances and a touching storyline.",
    "Despite a strong start, the film loses its way in the second half and ends disappointingly.",
    "A masterpiece of modern cinema that will be remembered for years to come."
]
labels = np.array([0, 1, 0, 1])  # 0 for negative, 1 for positive

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')

# Build the model with attention
inputs = Input(shape=(50,))
embedding = Embedding(input_dim=1000, output_dim=16, input_length=50)(inputs)
lstm = Bidirectional(LSTM(32, return_sequences=True))(embedding)
attention = Dot(axes=[2, 2])([lstm, lstm])
attention = Activation('softmax')(attention)
context = Dot(axes=[2, 1])([attention, lstm])
output = Dense(1, activation='sigmoid')(context)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=50, verbose=0)

# Make predictions
new_texts = [
    "A disappointing sequel that fails to capture the magic of the original.",
    "An innovative and captivating film that leaves a lasting impression."
]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=50, padding='post', truncating='post')
predictions = model.predict(new_padded)

for text, pred in zip(new_texts, predictions):
    print(f"Text: {text}")
    print(f"Sentiment: {'Positive' if pred > 0.5 else 'Negative'}")
    print(f"Confidence: {pred[0]:.2f}")
    print()
```

Slide 12: Transfer Learning with Pre-trained Models

Transfer learning allows us to leverage knowledge from pre-trained models on large datasets. We'll use the BERT (Bidirectional Encoder Representations from Transformers) model for sentiment analysis.

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Sample text for sentiment analysis
text = "This movie is a rollercoaster of emotions, with stunning visuals and exceptional performances."

# Tokenize and prepare input
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    predictions = softmax(outputs.logits, dim=-1)

# Get sentiment label
sentiment_label = "Positive" if predictions[0][1] > predictions[0][0] else "Negative"
confidence = predictions[0][1] if sentiment_label == "Positive" else predictions[0][0]

print(f"Text: {text}")
print(f"Sentiment: {sentiment_label}")
print(f"Confidence: {confidence.item():.2f}")
```

Slide 13: Real-life Example: Social Media Sentiment Analysis

Let's apply our sentiment analysis techniques to analyze public opinion on a social media platform about a recent product launch.

```python
import pandas as pd
from transformers import pipeline

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Sample social media data
tweets = [
    "Just got my hands on the new XYZ phone. The camera quality is mind-blowing! #XYZlaunch",
    "Disappointed with the battery life of the new XYZ phone. Expected better. #XYZlaunch",
    "The XYZ phone's UI is so intuitive. Loving the user experience! #XYZlaunch",
    "XYZ's customer support is terrible. Still waiting for a response about my order. #XYZlaunch",
    "The new XYZ phone is blazing fast. Apps load instantly! #XYZlaunch"
]

# Perform sentiment analysis
results = sentiment_pipeline(tweets)

# Create a DataFrame with results
df = pd.DataFrame({
    'Tweet': tweets,
    'Sentiment': [r['label'] for r in results],
    'Score': [r['score'] for r in results]
})

# Print results
print(df)

# Calculate overall sentiment
positive_count = sum(df['Sentiment'] == 'POSITIVE')
negative_count = sum(df['Sentiment'] == 'NEGATIVE')
total_count = len(df)

print(f"\nOverall Sentiment:")
print(f"Positive: {positive_count/total_count:.2%}")
print(f"Negative: {negative_count/total_count:.2%}")
```

Slide 14: Challenges and Future Directions in Sentiment Analysis

Sentiment analysis faces several challenges, including:

1. Sarcasm and irony detection
2. Context-dependent sentiment
3. Multilingual sentiment analysis
4. Handling emojis and informal language

Future directions in sentiment analysis include:

1. Multimodal sentiment analysis (text, image, audio)
2. Fine-grained emotion detection
3. Aspect-based sentiment analysis
4. Real-time sentiment analysis for streaming data

Researchers are exploring advanced deep learning architectures and combining different modalities to address these challenges and push the boundaries of sentiment analysis capabilities.

Slide 15: Additional Resources

For those interested in diving deeper into sentiment analysis, here are some valuable resources:

1. ArXiv paper: "A Survey on Deep Learning for Named Entity Recognition" URL: [https://arxiv.org/abs/1812.09449](https://arxiv.org/abs/1812.09449)
2. ArXiv paper: "Deep Learning Based Text Classification: A Comprehensive Review" URL: [https://arxiv.org/abs/2004.03705](https://arxiv.org/abs/2004.03705)
3. ArXiv paper: "Attention Is All You Need" (Transformer architecture) URL: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
4. NLTK documentation for sentiment analysis: [https://www.nltk.org/howto/sentiment.html](https://www.nltk.org/howto/sentiment.html)
5. Scikit-learn documentation for text feature extraction: [https://scikit-learn.org/stable/modules/feature\_extraction.html#text-feature-extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

These resources provide in-depth information on various aspects of sentiment analysis and natural language processing, from foundational concepts to cutting-edge techniques.

## Response:
undefined

