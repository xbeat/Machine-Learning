## NLP Fundamentals! Transformers, Context, and Python
Slide 1: Introduction to NLP

Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a valuable way. NLP combines computational linguistics, machine learning, and deep learning to process and analyze large amounts of natural language data.

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = "Natural Language Processing is fascinating!"
tokens = word_tokenize(text)
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print(f"Original text: {text}")
print(f"Tokenized and filtered: {filtered_tokens}")
```

Slide 2: Tokenization

Tokenization is the process of breaking down text into smaller units called tokens. These tokens can be words, characters, or subwords. Tokenization is a fundamental step in many NLP tasks, as it helps in understanding the structure of the text and preparing it for further analysis.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)

tokens = [token.text for token in doc]
print(f"Tokens: {tokens}")

sentences = [sent.text for sent in doc.sents]
print(f"Sentences: {sentences}")
```

Slide 3: Part-of-Speech Tagging

Part-of-Speech (POS) tagging is the process of assigning grammatical categories (such as noun, verb, adjective) to each word in a text. This information is crucial for understanding the syntactic structure of sentences and can be used in various NLP applications.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "The cat sat on the mat."
doc = nlp(text)

pos_tags = [(token.text, token.pos_) for token in doc]
print(f"POS tags: {pos_tags}")
```

Slide 4: Named Entity Recognition

Named Entity Recognition (NER) is the task of identifying and classifying named entities in text into predefined categories such as person names, organizations, locations, and more. NER is essential for information extraction and text understanding.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
doc = nlp(text)

entities = [(ent.text, ent.label_) for ent in doc.ents]
print(f"Named Entities: {entities}")
```

Slide 5: Sentiment Analysis

Sentiment Analysis is the process of determining the emotional tone behind a piece of text. It can be used to identify and extract subjective information from source materials, helping businesses understand customer opinions and feedback.

```python
from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

text = "I love this product! It's amazing."
sentiment = analyze_sentiment(text)
print(f"Text: {text}")
print(f"Sentiment: {sentiment}")
```

Slide 6: Word Embeddings

Word embeddings are dense vector representations of words that capture semantic meaning. They allow words with similar meanings to have similar vector representations, enabling machines to understand relationships between words and concepts.

```python
import gensim.downloader as api

# Load pre-trained word2vec embeddings
word2vec_model = api.load("word2vec-google-news-300")

# Find similar words
similar_words = word2vec_model.most_similar("computer")
print("Words similar to 'computer':")
for word, score in similar_words:
    print(f"{word}: {score:.2f}")

# Word analogy
result = word2vec_model.most_similar(positive=['woman', 'king'], negative=['man'])
print("\nWord analogy (woman - man + king):")
print(f"Result: {result[0][0]}")
```

Slide 7: Transformers

Transformers are a type of neural network architecture that has revolutionized NLP. They use self-attention mechanisms to process input sequences in parallel, capturing long-range dependencies more effectively than traditional sequential models like RNNs.

```python
import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize input text
text = "Transformers have revolutionized NLP."
inputs = tokenizer(text, return_tensors='pt')

# Get BERT embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Extract the [CLS] token embedding
sentence_embedding = outputs.last_hidden_state[:, 0, :]
print(f"Sentence embedding shape: {sentence_embedding.shape}")
```

Slide 8: Context in NLP

Context in NLP refers to the surrounding information that helps in understanding the meaning of words or phrases. Contextual information is crucial for tasks like word sense disambiguation, coreference resolution, and language generation.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def resolve_pronouns(text):
    doc = nlp(text)
    resolved_text = []
    for token in doc:
        if token.pos_ == "PRON" and token.dep_ == "nsubj":
            if token.head.pos_ == "VERB":
                for ent in doc.ents:
                    if ent.root.dep_ == "nsubj" and ent.root.head == token.head:
                        resolved_text.append(ent.text)
                        break
                else:
                    resolved_text.append(token.text)
            else:
                resolved_text.append(token.text)
        else:
            resolved_text.append(token.text)
    return " ".join(resolved_text)

text = "John went to the store. He bought some milk."
resolved = resolve_pronouns(text)
print(f"Original: {text}")
print(f"Resolved: {resolved}")
```

Slide 9: Intent Recognition

Intent recognition is the task of identifying the purpose or goal behind a user's input in natural language. It's a crucial component in building conversational AI systems and chatbots.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
intents = ["greeting", "weather", "goodbye"]
X = [
    "hello", "hi", "hey",
    "what's the weather like", "is it raining",
    "bye", "see you later", "goodbye"
]
y = [0, 0, 0, 1, 1, 2, 2, 2]

# Create a bag-of-words model
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_vec, y)

# Predict intent
def predict_intent(text):
    text_vec = vectorizer.transform([text])
    intent_id = clf.predict(text_vec)[0]
    return intents[intent_id]

# Test the model
test_texts = ["hello there", "what's the temperature today", "goodbye friend"]
for text in test_texts:
    print(f"Text: '{text}' - Intent: {predict_intent(text)}")
```

Slide 10: Natural Language Generation (NLG)

Natural Language Generation (NLG) is the process of producing human-readable text from structured data or other input. It involves tasks such as text summarization, translation, and content creation.

```python
import random

def generate_weather_report(temperature, condition):
    templates = [
        "Today's weather is {condition} with a temperature of {temp}°C.",
        "Expect {condition} conditions and a temperature around {temp}°C today.",
        "The weather forecast shows {condition} skies and {temp}°C."
    ]
    report = random.choice(templates)
    return report.format(condition=condition, temp=temperature)

# Generate weather reports
temperatures = [20, 25, 15]
conditions = ["sunny", "cloudy", "rainy"]

for temp, cond in zip(temperatures, conditions):
    report = generate_weather_report(temp, cond)
    print(report)
```

Slide 11: Text Summarization

Text summarization is the process of creating a concise and coherent summary of a longer text while preserving its key information. It can be extractive (selecting important sentences) or abstractive (generating new text).

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

def extractive_summarize(text, num_sentences=3):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Calculate word frequencies
    freq = FreqDist(words)
    
    # Score sentences based on word frequencies
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                if i in sentence_scores:
                    sentence_scores[i] += freq[word]
                else:
                    sentence_scores[i] = freq[word]
    
    # Get the top N sentences
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    summary = ' '.join([sentences[i] for i in sorted(summarized_sentences)])
    
    return summary

# Example usage
text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language, but at the time not articulated as a problem separate from artificial intelligence.
"""

summary = extractive_summarize(text)
print("Original text length:", len(text))
print("Summary length:", len(summary))
print("\nSummary:")
print(summary)
```

Slide 12: Language Translation

Language translation is the task of converting text from one language to another while preserving its meaning. Modern translation systems use neural machine translation techniques based on sequence-to-sequence models and transformers.

```python
from transformers import MarianMTModel, MarianTokenizer

def translate(text, source_lang, target_lang):
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    return translated_text

# Example usage
english_text = "Hello, how are you?"
spanish_translation = translate(english_text, "en", "es")
french_translation = translate(english_text, "en", "fr")

print(f"English: {english_text}")
print(f"Spanish: {spanish_translation}")
print(f"French: {french_translation}")
```

Slide 13: Real-life Example: Chatbot

Chatbots are a common application of NLP, combining various techniques like intent recognition, entity extraction, and natural language generation to create interactive conversational agents.

```python
import random

# Simple rule-based chatbot
greetings = ["hello", "hi", "hey", "greetings"]
farewells = ["bye", "goodbye", "see you", "farewell"]

def simple_chatbot(user_input):
    user_input = user_input.lower()
    
    if any(word in user_input for word in greetings):
        return random.choice(["Hello!", "Hi there!", "Greetings!"])
    elif any(word in user_input for word in farewells):
        return random.choice(["Goodbye!", "See you later!", "Take care!"])
    elif "name" in user_input:
        return "My name is ChatBot. Nice to meet you!"
    elif "weather" in user_input:
        return "I'm sorry, I don't have access to real-time weather information."
    else:
        return "I'm not sure how to respond to that. Can you please rephrase or ask something else?"

# Simulate a conversation
print("ChatBot: Hi! I'm a simple chatbot. Type 'bye' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in farewells:
        print("ChatBot: Goodbye!")
        break
    response = simple_chatbot(user_input)
    print("ChatBot:", response)
```

Slide 14: Real-life Example: Text Classification

Text classification is a fundamental NLP task used in various applications, such as spam detection, sentiment analysis, and topic categorization.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample dataset
texts = [
    "I love this product, it's amazing!",
    "This is the worst experience ever.",
    "The customer service was excellent.",
    "I'm very disappointed with the quality.",
    "Highly recommended, great value for money!",
    "Don't waste your time, it's terrible.",
    "Average product, nothing special.",
    "Exceeded my expectations, very satisfied!",
]
labels = ["positive", "negative", "positive", "negative", "positive", "negative", "neutral", "positive"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

# Train the model
text_clf.fit(X_train, y_train)

# Make predictions
y_pred = text_clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Test with new examples
new_texts = [
    "This product is fantastic!",
    "I regret buying this, it's awful.",
    "It's okay, but could be better."
]

predictions = text_clf.predict(new_texts)
for text, prediction in zip(new_texts, predictions):
    print(f"Text: '{text}'")
    print(f"Predicted sentiment: {prediction}\n")
```

Slide 15: Additional Resources

For those interested in diving deeper into NLP, here are some valuable resources:

1. ArXiv.org NLP papers: [https://arxiv.org/list/cs.CL/recent](https://arxiv.org/list/cs.CL/recent)
2. "Attention Is All You Need" (Transformer paper): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
3. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding": [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
4. "Natural Language Processing with Transformers" by Lewis Tunstall, Leandro

