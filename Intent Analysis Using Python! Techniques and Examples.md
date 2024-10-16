## Intent Analysis Using Python! Techniques and Examples
Slide 1: Understanding Intent Analysis

Intent analysis is a crucial component of natural language processing (NLP) that focuses on determining the underlying purpose or goal behind a user's text input. This technique is widely used in chatbots, virtual assistants, and customer service applications to interpret and respond to user queries effectively.

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

user_input = "How do I reset my password?"
processed_input = preprocess_text(user_input)
print(processed_input)
```

Slide 2: Text Preprocessing for Intent Analysis

Before analyzing intent, it's essential to preprocess the text input. This involves tokenization, lowercasing, and removing stopwords to focus on the most meaningful words.

```python
from sklearn.feature_extraction.text import CountVectorizer

def vectorize_text(texts):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(texts), vectorizer

sample_texts = [
    "How do I reset my password?",
    "What are your business hours?",
    "I need to cancel my subscription"
]

vector_matrix, vectorizer = vectorize_text(sample_texts)
print(vector_matrix.toarray())
print(vectorizer.get_feature_names_out())
```

Slide 3: Feature Extraction

Feature extraction is a critical step in intent analysis. It involves converting text data into numerical features that machine learning models can understand. The CountVectorizer is a simple yet effective method for this purpose.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

intents = ['password_reset', 'business_hours', 'cancel_subscription']
X_train, X_test, y_train, y_test = train_test_split(vector_matrix, intents, test_size=0.2, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)

print(f"Model accuracy: {clf.score(X_test, y_test)}")
```

Slide 4: Training a Simple Intent Classifier

Using the extracted features, we can train a machine learning model to classify intents. In this example, we use a Multinomial Naive Bayes classifier, which is often effective for text classification tasks.

```python
def predict_intent(text, clf, vectorizer):
    text_vector = vectorizer.transform([text])
    intent = clf.predict(text_vector)[0]
    return intent

new_text = "What time do you close?"
predicted_intent = predict_intent(new_text, clf, vectorizer)
print(f"Predicted intent: {predicted_intent}")
```

Slide 5: Predicting Intent

Once the model is trained, we can use it to predict the intent of new, unseen text inputs. This function takes a text input, transforms it using the same vectorizer, and then uses the trained classifier to predict the intent.

```python
from sklearn.metrics import classification_report

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

Slide 6: Evaluating the Intent Classifier

To ensure our intent classifier is performing well, we need to evaluate its performance. The classification report provides a comprehensive view of the model's performance, including precision, recall, and F1-score for each intent class.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

sample_text = "I need to reset my Google account password by tomorrow."
print(extract_entities(sample_text))
```

Slide 7: Entity Extraction in Intent Analysis

Entity extraction is often used alongside intent classification to provide more context and information. This example uses SpaCy, a popular NLP library, to extract named entities from the text.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = [
    "How do I reset my password?",
    "What are your business hours?",
    "I need to cancel my subscription",
    "Can you help me with my account?",
    "Where is your nearest store?"
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=10, padding='post')

print(padded)
```

Slide 8: Preparing Data for Deep Learning Models

When using deep learning models for intent analysis, we need to prepare our text data differently. This example shows how to tokenize and pad sequences for use with neural networks.

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(100, 16, input_length=10),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

Slide 9: Building a Deep Learning Model for Intent Classification

This slide demonstrates how to build a simple deep learning model using TensorFlow for intent classification. The model uses an embedding layer, followed by a global average pooling layer and dense layers.

```python
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

svm_clf = SVC(kernel='rbf')
scores = cross_val_score(svm_clf, vector_matrix, intents, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.2f}")
```

Slide 10: Cross-Validation for Intent Classification

Cross-validation is an important technique to assess how well our intent classification model generalizes to unseen data. This example uses 5-fold cross-validation with a Support Vector Machine classifier.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sample_texts)

print(tfidf_matrix.toarray())
print(tfidf_vectorizer.get_feature_names_out())
```

Slide 11: TF-IDF for Intent Analysis

TF-IDF (Term Frequency-Inverse Document Frequency) is another popular method for feature extraction in intent analysis. It often performs better than simple count vectorization by considering the importance of words across the entire corpus.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_intent(query, intents, vectorizer):
    query_vec = vectorizer.transform([query])
    intent_vecs = vectorizer.transform(intents)
    similarities = cosine_similarity(query_vec, intent_vecs)
    most_similar_idx = np.argmax(similarities)
    return intents[most_similar_idx], similarities[0][most_similar_idx]

query = "How can I change my account settings?"
intents = [
    "How do I reset my password?",
    "What are your business hours?",
    "I need to cancel my subscription"
]

similar_intent, similarity_score = find_similar_intent(query, intents, tfidf_vectorizer)
print(f"Most similar intent: {similar_intent}")
print(f"Similarity score: {similarity_score:.2f}")
```

Slide 12: Intent Similarity Using Cosine Similarity

In some cases, we might want to find the most similar intent from a predefined set. This example demonstrates how to use cosine similarity to find the most similar intent to a given query.

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

sequence = "I want to return an item I purchased last week."
candidate_labels = ['refund request', 'product inquiry', 'complaint', 'track order']

result = classifier(sequence, candidate_labels)
print(result)
```

Slide 13: Zero-Shot Intent Classification

Zero-shot learning allows us to classify intents without training on specific examples. This approach is particularly useful when we have a dynamic set of intents or limited training data.

Slide 14: Real-Life Example: Smart Home Assistant

Consider a smart home assistant that needs to interpret user commands. The assistant needs to differentiate between various intents such as controlling lights, adjusting temperature, or setting alarms.

```python
import random

intents = {
    'lights_on': ['turn on the lights', 'lights on', 'illuminate the room'],
    'lights_off': ['turn off the lights', 'lights off', 'darken the room'],
    'set_temperature': ['set temperature to', 'adjust thermostat to', 'change temperature to'],
    'set_alarm': ['set an alarm for', 'wake me up at', 'alarm for']
}

def simple_intent_classifier(user_input):
    user_input = user_input.lower()
    for intent, phrases in intents.items():
        if any(phrase in user_input for phrase in phrases):
            return intent
    return 'unknown_intent'

# Simulate user interactions
user_commands = [
    "Turn on the lights in the living room",
    "Set temperature to 72 degrees",
    "Wake me up at 7 AM tomorrow",
    "What's the weather like?"
]

for command in user_commands:
    intent = simple_intent_classifier(command)
    print(f"User: {command}")
    print(f"Assistant: Detected intent: {intent}")
    print()
```

Slide 15: Real-Life Example: Customer Support Chatbot

Another common application of intent analysis is in customer support chatbots. These bots need to understand the user's intent to provide appropriate responses or route the query to the right department.

```python
import re

intents = {
    'greeting': r'\b(hi|hello|hey)\b',
    'farewell': r'\b(bye|goodbye|see you)\b',
    'order_status': r'\b(order|status|tracking)\b',
    'product_info': r'\b(product|item|details)\b',
    'technical_support': r'\b(not working|broken|error|issue)\b'
}

def regex_intent_classifier(user_input):
    user_input = user_input.lower()
    for intent, pattern in intents.items():
        if re.search(pattern, user_input):
            return intent
    return 'unknown_intent'

# Simulate customer interactions
customer_queries = [
    "Hello, I need help with my order",
    "What are the details of the new smartphone?",
    "My laptop is not working properly",
    "Goodbye and thank you for your help"
]

for query in customer_queries:
    intent = regex_intent_classifier(query)
    print(f"Customer: {query}")
    print(f"Chatbot: Detected intent: {intent}")
    print()
```

Slide 16: Additional Resources

For those interested in diving deeper into intent analysis and natural language processing, here are some valuable resources:

1. "Attention Is All You Need" by Vaswani et al. (2017) - Introduces the Transformer architecture, which has revolutionized NLP. ArXiv link: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) - Presents BERT, a powerful language model for various NLP tasks. ArXiv link: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Few-Shot Text Classification with Pre-Trained Language Models" by Gao et al. (2020) - Explores techniques for intent classification with limited training data. ArXiv link: [https://arxiv.org/abs/2108.08466](https://arxiv.org/abs/2108.08466)

These papers provide a solid foundation for understanding advanced techniques in NLP and intent analysis.

