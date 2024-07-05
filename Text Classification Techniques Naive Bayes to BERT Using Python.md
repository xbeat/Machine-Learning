## Text Classification Techniques Naive Bayes to BERT Using Python

Slide 1: Introduction to Text Classification

Text classification is the task of assigning predefined categories to text documents. It has various applications, including spam detection, sentiment analysis, and topic categorization. In this presentation, we'll explore different techniques for text classification, ranging from traditional methods like Naive Bayes to modern deep learning approaches like BERT.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Example text and labels
texts = ["This movie is great", "I hate this book", "The weather is nice"]
labels = ["positive", "negative", "neutral"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
```

Slide 2: Text Preprocessing

Before applying any classification algorithm, it's crucial to preprocess the text data. This step involves tokenization, lowercasing, removing punctuation, and handling stop words. Proper preprocessing can significantly improve the performance of text classification models.

```python
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Tokenize and remove stop words
    tokens = [word for word in text.split() if word not in stop_words]
    
    return ' '.join(tokens)

preprocessed_texts = [preprocess_text(text) for text in texts]
```

Slide 3: Feature Extraction - Bag of Words

The Bag of Words (BoW) model is a simple yet effective method for representing text data as numerical features. It creates a vocabulary of unique words and represents each document as a vector of word frequencies.

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preprocessed_texts)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

Slide 4: Naive Bayes Classifier

Naive Bayes is a probabilistic classifier based on Bayes' theorem. It's particularly effective for text classification due to its simplicity and ability to handle high-dimensional data. The MultinomialNB variant is commonly used for text data.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

print("Accuracy:", nb_classifier.score(X_test, y_test))
```

Slide 5: TF-IDF Vectorization

Term Frequency-Inverse Document Frequency (TF-IDF) is an improvement over the simple Bag of Words model. It takes into account the importance of words across the entire corpus, giving higher weight to rare words that are more informative.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(preprocessed_texts)

print(tfidf_vectorizer.get_feature_names_out())
print(X_tfidf.toarray())
```

Slide 6: Support Vector Machines (SVM)

Support Vector Machines are powerful classifiers that work well for text classification. They aim to find the hyperplane that best separates the classes in a high-dimensional space. SVMs can handle non-linear decision boundaries using kernel tricks.

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.2)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

print("Accuracy:", svm_classifier.score(X_test, y_test))
```

Slide 7: Word Embeddings - Word2Vec

Word embeddings are dense vector representations of words that capture semantic relationships. Word2Vec is a popular method for creating word embeddings. These embeddings can be used as features for text classification tasks.

```python
from gensim.models import Word2Vec

# Tokenize the texts
tokenized_texts = [text.split() for text in preprocessed_texts]

# Train Word2Vec model
w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

# Get word vector
print(w2v_model.wv['movie'])
```

Slide 8: Recurrent Neural Networks (RNN)

Recurrent Neural Networks are a class of neural networks designed to handle sequential data, making them suitable for text classification. They can capture long-range dependencies in text.

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_texts)
sequences = tokenizer.texts_to_sequences(preprocessed_texts)
X_padded = pad_sequences(sequences, maxlen=50)

# Build RNN model
model = tf.keras.Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=32, input_length=50),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

Slide 9: Long Short-Term Memory (LSTM)

LSTM networks are an advanced type of RNN that can better capture long-term dependencies in text. They are particularly effective for tasks that require understanding context over longer sequences.

```python
from tensorflow.keras.layers import LSTM

# Build LSTM model
lstm_model = tf.keras.Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=32, input_length=50),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

Slide 10: Convolutional Neural Networks (CNN) for Text

While primarily used for image processing, CNNs have shown great performance in text classification tasks. They can capture local patterns in text, similar to n-grams, but in a more flexible and learnable manner.

```python
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D

# Build CNN model
cnn_model = tf.keras.Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=32, input_length=50),
    Conv1D(32, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

Slide 11: Transfer Learning with Pre-trained Word Embeddings

Transfer learning allows us to leverage knowledge from pre-trained models. Using pre-trained word embeddings like GloVe can significantly improve the performance of text classification models, especially when working with limited labeled data.

```python
import numpy as np
from tensorflow.keras.layers import Embedding

# Load pre-trained GloVe embeddings
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Use pre-trained embeddings in the model
model = tf.keras.Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_dim, 
              weights=[embedding_matrix], input_length=50, trainable=False),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

Slide 12: Transformer Architecture

The Transformer architecture, introduced in the "Attention is All You Need" paper, revolutionized natural language processing. It relies solely on attention mechanisms, allowing for efficient parallel processing of sequential data.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim),])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Build Transformer model
inputs = Input(shape=(50,))
embedding_layer = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=32, input_length=50)(inputs)
transformer_block = TransformerBlock(32, 2, 32)(embedding_layer)
x = GlobalAveragePooling1D()(transformer_block)
outputs = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

Slide 13: BERT - Bidirectional Encoder Representations from Transformers

BERT is a state-of-the-art model for various NLP tasks, including text classification. It uses bidirectional training of the Transformer architecture to develop a deep understanding of language context. Fine-tuning BERT on specific tasks often yields excellent results.

```python
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize and encode the text data
inputs = tokenizer(preprocessed_texts, padding=True, truncation=True, return_tensors="tf")

# Prepare the model for fine-tuning
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Fine-tune the model
model.fit(inputs.data, labels, epochs=3, batch_size=16)
```

Slide 14: Evaluation Metrics for Text Classification

Evaluating text classification models is crucial for understanding their performance. Common metrics include accuracy, precision, recall, F1-score, and confusion matrix. For imbalanced datasets, consider using metrics like ROC-AUC or precision-recall curves.

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have predictions from a model
y_pred = model.predict(X_test)

# Generate classification report
print(classification_report(y_test, y_pred))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

Slide 15: Additional Resources

For further exploration of text classification techniques and advanced NLP models, consider the following resources:

1. "Attention Is All You Need" by Vaswani et al. (2017) arXiv:1706.03762
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) arXiv:1810.04805
3. "XLNet: Generalized Autoregressive Pretraining for Language Understanding" by Yang et al. (2019) arXiv:1906.08237
4. "RoBERTa: A Robustly Optimized BERT Pretraining Approach" by Liu et al. (2019) arXiv:1907.11692
5. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" by Raffel et al. (2019) arXiv:1910.10683

