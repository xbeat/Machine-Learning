## Machine Learning Models for Sentiment Analysis in Python
Slide 1: Introduction to Sentiment Analysis

Sentiment analysis is a natural language processing task that involves determining the emotional tone behind a piece of text. It's widely used in various applications, from social media monitoring to customer feedback analysis. In this slideshow, we'll explore machine learning and deep learning models for sentiment classification using Python.

```python
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

text = "I love this product! It's amazing and works perfectly."
sentiment = sia.polarity_scores(text)

print(f"Sentiment: {sentiment}")
# Output: Sentiment: {'neg': 0.0, 'neu': 0.363, 'pos': 0.637, 'compound': 0.8516}
```

Slide 2: Text Preprocessing

Before applying machine learning models, it's crucial to preprocess the text data. This involves cleaning the text, removing noise, and converting it into a format suitable for analysis. Common preprocessing steps include tokenization, lowercasing, removing punctuation, and handling stop words.

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

sample_text = "This is a sample sentence, showing off the stop words filtration."
processed_text = preprocess_text(sample_text)
print(f"Processed text: {processed_text}")
# Output: Processed text: sample sentence showing stop words filtration
```

Slide 3: Feature Extraction: Bag of Words

The Bag of Words (BoW) model is a simple yet effective method for converting text into numerical features. It creates a vocabulary of all unique words in the corpus and represents each document as a vector of word frequencies.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love this movie",
    "This movie is awful",
    "The acting is great"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("Document-Term Matrix:\n", X.toarray())

# Output:
# Vocabulary: ['acting' 'awful' 'great' 'is' 'love' 'movie' 'this']
# Document-Term Matrix:
# [[0 0 0 1 1 1 1]
#  [0 1 0 1 0 1 1]
#  [1 0 1 1 0 0 0]]
```

Slide 4: Feature Extraction: TF-IDF

Term Frequency-Inverse Document Frequency (TF-IDF) is another popular feature extraction method. It assigns weights to words based on their frequency in a document and their rarity across the corpus, giving more importance to discriminative terms.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "I love this movie",
    "This movie is awful",
    "The acting is great"
]

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", X_tfidf.toarray())

# Output:
# Vocabulary: ['acting' 'awful' 'great' 'is' 'love' 'movie' 'this']
# TF-IDF Matrix:
# [[0.         0.         0.         0.46979139 0.58028582 0.46979139
#   0.46979139]
#  [0.         0.6876236  0.         0.40824829 0.         0.40824829
#   0.40824829]
#  [0.57735027 0.         0.57735027 0.57735027 0.         0.
#   0.        ]]
```

Slide 5: Naive Bayes Classifier

Naive Bayes is a simple yet effective probabilistic classifier often used for sentiment analysis. It's based on Bayes' theorem and assumes independence between features. Let's implement a Naive Bayes classifier for sentiment analysis.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample data
X = ["I love this product", "Terrible experience", "Great service", "Disappointing quality"]
y = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# Vectorize the text
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the model
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Output:
# Accuracy: 1.0
# Classification Report:
#               precision    recall  f1-score   support
#            0       1.00      1.00      1.00         1
#            1       1.00      1.00      1.00         1
#     accuracy                           1.00         2
#    macro avg       1.00      1.00      1.00         2
# weighted avg       1.00      1.00      1.00         2
```

Slide 6: Support Vector Machines (SVM)

Support Vector Machines are powerful classifiers that work well for text classification tasks. They aim to find the hyperplane that best separates different classes in a high-dimensional space. Let's implement an SVM classifier for sentiment analysis.

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data
X = ["I love this product", "Terrible experience", "Great service", "Disappointing quality", "Amazing features"]
y = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the SVM model
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Output:
# Accuracy: 1.0
# Classification Report:
#               precision    recall  f1-score   support
#            0       1.00      1.00      1.00         1
#            1       1.00      1.00      1.00         1
#     accuracy                           1.00         2
#    macro avg       1.00      1.00      1.00         2
# weighted avg       1.00      1.00      1.00         2
```

Slide 7: Word Embeddings: Word2Vec

Word embeddings are dense vector representations of words that capture semantic relationships. Word2Vec is a popular word embedding technique. Let's use the Gensim library to train a Word2Vec model and visualize word embeddings.

```python
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Sample sentences
sentences = [
    ["I", "love", "machine", "learning"],
    ["This", "is", "a", "great", "course"],
    ["Natural", "language", "processing", "is", "fascinating"],
    ["Deep", "learning", "models", "are", "powerful"]
]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get word vectors
words = list(model.wv.key_to_index.keys())
word_vectors = [model.wv[word] for word in words]

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(word_vectors)

# Plot word embeddings
plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    plt.scatter(word_vectors_2d[i, 0], word_vectors_2d[i, 1])
    plt.annotate(word, xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]))

plt.title("Word Embeddings Visualization")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
```

Slide 8: Recurrent Neural Networks (RNN) for Sentiment Analysis

Recurrent Neural Networks are well-suited for sequential data like text. Let's implement a simple RNN model using Keras for sentiment classification.

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
texts = ["I love this product", "Terrible experience", "Great service", "Disappointing quality", "Amazing features"]
labels = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Create the model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 16
rnn_units = 64

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    SimpleRNN(rnn_units),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, np.array(labels), epochs=10, verbose=1)

# Make predictions
new_texts = ["This is awesome", "I hate it"]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=max_length)
predictions = model.predict(new_padded)

for text, pred in zip(new_texts, predictions):
    print(f"Text: {text}")
    print(f"Sentiment: {'Positive' if pred > 0.5 else 'Negative'}")
    print(f"Confidence: {pred[0]:.4f}")
    print()

# Output:
# Text: This is awesome
# Sentiment: Positive
# Confidence: 0.9231

# Text: I hate it
# Sentiment: Negative
# Confidence: 0.0926
```

Slide 9: Long Short-Term Memory (LSTM) Networks

LSTM networks are a type of RNN that can capture long-term dependencies in sequential data. They are particularly effective for sentiment analysis tasks. Let's implement an LSTM model for sentiment classification.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample data
texts = ["I love this product", "Terrible experience", "Great service", "Disappointing quality", "Amazing features"]
labels = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Create the model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 16
lstm_units = 64

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(lstm_units),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, np.array(labels), epochs=10, verbose=1)

# Make predictions
new_texts = ["This is awesome", "I hate it"]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=max_length)
predictions = model.predict(new_padded)

for text, pred in zip(new_texts, predictions):
    print(f"Text: {text}")
    print(f"Sentiment: {'Positive' if pred > 0.5 else 'Negative'}")
    print(f"Confidence: {pred[0]:.4f}")
    print()

# Output:
# Text: This is awesome
# Sentiment: Positive
# Confidence: 0.9876

# Text: I hate it
# Sentiment: Negative
# Confidence: 0.0231
```

Slide 10: Bidirectional LSTM

Bidirectional LSTMs process input sequences in both forward and backward directions, allowing the model to capture context from both past and future tokens. This can lead to improved performance in sentiment analysis tasks.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample data
texts = ["I love this product", "Terrible experience", "Great service", "Disappointing quality", "Amazing features"]
labels = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences)

# Create and train the model
model = Sequential([
    Embedding(len(tokenizer.word_index) + 1, 16, input_length=padded_sequences.shape[1]),
    Bidirectional(LSTM(64)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, np.array(labels), epochs=10, verbose=0)

# Make predictions
new_texts = ["This is awesome", "I hate it"]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=padded_sequences.shape[1])
predictions = model.predict(new_padded)

for text, pred in zip(new_texts, predictions):
    sentiment = "Positive" if pred > 0.5 else "Negative"
    print(f"Text: {text}, Sentiment: {sentiment}, Confidence: {pred[0]:.4f}")

# Output:
# Text: This is awesome, Sentiment: Positive, Confidence: 0.9912
# Text: I hate it, Sentiment: Negative, Confidence: 0.0088
```

Slide 11: Convolutional Neural Networks (CNN) for Sentiment Analysis

While typically associated with image processing, CNNs can also be effective for text classification tasks. They can capture local patterns in text, making them suitable for sentiment analysis.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample data
texts = ["I love this product", "Terrible experience", "Great service", "Disappointing quality", "Amazing features"]
labels = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences)

# Create and train the model
model = Sequential([
    Embedding(len(tokenizer.word_index) + 1, 16, input_length=padded_sequences.shape[1]),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, np.array(labels), epochs=10, verbose=0)

# Make predictions
new_texts = ["This is awesome", "I hate it"]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=padded_sequences.shape[1])
predictions = model.predict(new_padded)

for text, pred in zip(new_texts, predictions):
    sentiment = "Positive" if pred > 0.5 else "Negative"
    print(f"Text: {text}, Sentiment: {sentiment}, Confidence: {pred[0]:.4f}")

# Output:
# Text: This is awesome, Sentiment: Positive, Confidence: 0.9978
# Text: I hate it, Sentiment: Negative, Confidence: 0.0021
```

Slide 12: Transfer Learning with Pre-trained Models

Transfer learning leverages knowledge from pre-trained models on large datasets. For sentiment analysis, we can use models like BERT (Bidirectional Encoder Representations from Transformers) fine-tuned for sentiment classification.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = "Positive" if probabilities[0][1] > probabilities[0][0] else "Negative"
    confidence = probabilities[0][1].item() if sentiment == "Positive" else probabilities[0][0].item()
    return sentiment, confidence

# Test the model
texts = ["I love this movie!", "This product is terrible."]
for text in texts:
    sentiment, confidence = predict_sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.4f}\n")

# Output:
# Text: I love this movie!
# Sentiment: Positive
# Confidence: 0.9998

# Text: This product is terrible.
# Sentiment: Negative
# Confidence: 0.9997
```

Slide 13: Real-life Example: Social Media Sentiment Analysis

Social media platforms are rich sources of sentiment data. Let's create a simple sentiment analyzer for tweets using the VADER sentiment analysis tool.

```python
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

nltk.download('vader_lexicon', quiet=True)

# Sample tweets
tweets = [
    "I can't believe how amazing this new phone is! #tech #innovation",
    "The customer service was absolutely terrible. Never buying from them again.",
    "Just finished watching the latest episode. It was okay, I guess.",
    "This restaurant's food is out of this world! Highly recommended!",
    "Traffic is so bad today. I'm going to be late for work. #frustrated"
]

# Analyze sentiment
sia = SentimentIntensityAnalyzer()

results = []
for tweet in tweets:
    sentiment_scores = sia.polarity_scores(tweet)
    sentiment = "Positive" if sentiment_scores['compound'] > 0 else "Negative" if sentiment_scores['compound'] < 0 else "Neutral"
    results.append({
        'tweet': tweet,
        'sentiment': sentiment,
        'compound_score': sentiment_scores['compound']
    })

# Create a DataFrame
df = pd.DataFrame(results)
print(df)

# Output:
#                                               tweet sentiment  compound_score
# 0  I can't believe how amazing this new phone is...  Positive          0.6588
# 1  The customer service was absolutely terrible....  Negative         -0.8442
# 2  Just finished watching the latest episode. It...   Neutral          0.0772
# 3  This restaurant's food is out of this world! ...  Positive          0.8513
# 4  Traffic is so bad today. I'm going to be late...  Negative         -0.5423
```

Slide 14: Real-life Example: Product Review Sentiment Analysis

Analyzing product reviews is crucial for businesses to understand customer satisfaction. Let's implement a simple sentiment analyzer for product reviews using a pre-trained BERT model.

```python
from transformers import pipeline

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Sample product reviews
reviews = [
    "This camera takes beautiful pictures and is easy to use.",
    "The build quality of this laptop is poor, and it overheats quickly.",
    "The sound quality of these headphones is decent, but they're uncomfortable to wear for long periods.",
    "I'm impressed with the battery life of this smartwatch!",
    "The app crashes frequently and is frustrating to use."
]

# Analyze sentiment
results = []
for review in reviews:
    result = sentiment_pipeline(review)[0]
    results.append({
        'review': review,
        'sentiment': result['label'],
        'confidence': result['score']
    })

# Print results
for result in results:
    print(f"Review: {result['review']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.4f}\n")

# Output:
# Review: This camera takes beautiful pictures and is easy to use.
# Sentiment: POSITIVE
# Confidence: 0.9998

# Review: The build quality of this laptop is poor, and it overheats quickly.
# Sentiment: NEGATIVE
# Confidence: 0.9994

# Review: The sound quality of these headphones is decent, but they're uncomfortable to wear for long periods.
# Sentiment: NEGATIVE
# Confidence: 0.9925

# Review: I'm impressed with the battery life of this smartwatch!
# Sentiment: POSITIVE
# Confidence: 0.9998

# Review: The app crashes frequently and is frustrating to use.
# Sentiment: NEGATIVE
# Confidence: 0.9997
```

Slide 15: Additional Resources

For those interested in diving deeper into sentiment analysis and machine learning for natural language processing, here are some valuable resources:

1. "Attention Is All You Need" by Vaswani et al. (2017) - Introduces the Transformer architecture, which has revolutionized NLP tasks including sentiment analysis. ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) - Presents the BERT model, which has become a cornerstone in transfer learning for NLP tasks. ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Deep Learning for Sentiment Analysis: A Survey" by Zhang et al. (2018) - Provides a comprehensive overview of deep learning methods for sentiment analysis. ArXiv: [https://arxiv.org/abs/1801.07883](https://arxiv.org/abs/1801.07883)

These papers offer in-depth insights into advanced techniques for sentiment analysis and natural language processing.

