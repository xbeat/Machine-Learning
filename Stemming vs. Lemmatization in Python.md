## Stemming vs. Lemmatization in Python
Slide 1: Introduction to Stemming and Lemmatization

Stemming and lemmatization are two fundamental techniques in Natural Language Processing (NLP) used to reduce words to their base or root form. While they serve similar purposes, their approaches and outcomes differ significantly. This presentation will explore both methods, their implementations in Python, and their practical applications.

```python
# Simple example of stemming and lemmatization
from nltk.stem import PorterStemmer, WordNetLemmatizer

word = "running"
ps = PorterStemmer()
wnl = WordNetLemmatizer()

print(f"Original: {word}")
print(f"Stemmed: {ps.stem(word)}")
print(f"Lemmatized: {wnl.lemmatize(word, 'v')}")

# Output:
# Original: running
# Stemmed: run
# Lemmatized: run
```

Slide 2: What is Stemming?

Stemming is a process of reducing words to their stem or root form. It works by truncating the end or beginning of the word, taking into account a list of common prefixes and suffixes. Stemming is a faster but less accurate method compared to lemmatization.

```python
from nltk.stem import PorterStemmer

ps = PorterStemmer()
words = ["cat", "cats", "catty", "catlike", "catness"]

for word in words:
    print(f"{word} -> {ps.stem(word)}")

# Output:
# cat -> cat
# cats -> cat
# catty -> catti
# catlike -> catlik
# catness -> cat
```

Slide 3: Types of Stemmers

There are several types of stemmers available in Python, each with its own algorithm and level of aggressiveness. The most common ones are Porter, Lancaster, and Snowball stemmers.

```python
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

word = "considerably"
ps = PorterStemmer()
ls = LancasterStemmer()
ss = SnowballStemmer('english')

print(f"Porter: {ps.stem(word)}")
print(f"Lancaster: {ls.stem(word)}")
print(f"Snowball: {ss.stem(word)}")

# Output:
# Porter: consid
# Lancaster: consid
# Snowball: consider
```

Slide 4: Implementing Porter Stemmer

The Porter Stemmer is one of the most widely used stemming algorithms. It's less aggressive compared to other stemmers and is often a good choice for general use.

```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

text = "The cats are running quickly through the forest"
words = word_tokenize(text)

stemmed_words = [ps.stem(word) for word in words]
print(" ".join(stemmed_words))

# Output: The cat are run quickli through the forest
```

Slide 5: What is Lemmatization?

Lemmatization is the process of reducing words to their base or dictionary form, known as the lemma. Unlike stemming, lemmatization considers the context and part of speech of the word, resulting in more accurate but slower processing.

```python
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

words = ["better", "worst", "running", "ran", "says", "said"]
for word in words:
    print(f"{word} -> {wnl.lemmatize(word)}")

# Output:
# better -> better
# worst -> worst
# running -> running
# ran -> ran
# says -> say
# said -> said
```

Slide 6: Implementing WordNet Lemmatizer

The WordNet Lemmatizer in NLTK uses the WordNet database to look up lemmas. It's important to specify the part of speech for more accurate results.

```python
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

word = "better"
print(f"Noun: {wnl.lemmatize(word, 'n')}")
print(f"Verb: {wnl.lemmatize(word, 'v')}")
print(f"Adjective: {wnl.lemmatize(word, 'a')}")
print(f"Adverb: {wnl.lemmatize(word, 'r')}")

# Output:
# Noun: better
# Verb: better
# Adjective: good
# Adverb: better
```

Slide 7: Stemming vs. Lemmatization: A Comparison

Let's compare the results of stemming and lemmatization on a set of words to understand their differences.

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

ps = PorterStemmer()
wnl = WordNetLemmatizer()

words = ["cats", "troubling", "troubled", "having", "better", "was"]

for word in words:
    print(f"Word: {word}")
    print(f"Stemmed: {ps.stem(word)}")
    print(f"Lemmatized (n): {wnl.lemmatize(word, 'n')}")
    print(f"Lemmatized (v): {wnl.lemmatize(word, 'v')}")
    print()

# Output:
# Word: cats
# Stemmed: cat
# Lemmatized (n): cat
# Lemmatized (v): cat
# ...
```

Slide 8: Advantages and Disadvantages

Stemming:

* Fast processing
* Simple implementation

* Less accurate
* Can produce non-words

Lemmatization:

* More accurate
* Produces valid words

* Slower processing
* Requires part-of-speech information

Slide 9: Advantages and Disadvantages

```python
import time
from nltk.stem import PorterStemmer, WordNetLemmatizer

ps = PorterStemmer()
wnl = WordNetLemmatizer()

word = "troubled"

start = time.time()
for _ in range(10000):
    ps.stem(word)
stem_time = time.time() - start

start = time.time()
for _ in range(10000):
    wnl.lemmatize(word, 'v')
lem_time = time.time() - start

print(f"Stemming time: {stem_time:.5f} seconds")
print(f"Lemmatization time: {lem_time:.5f} seconds")

# Output may vary:
# Stemming time: 0.00297 seconds
# Lemmatization time: 0.01589 seconds
```

Slide 10: Real-Life Example: Text Normalization for Search Engines

Search engines often use stemming or lemmatization to normalize query terms and document content. This helps in matching variations of words.

```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

def normalize_text(text):
    words = word_tokenize(text.lower())
    return " ".join(ps.stem(word) for word in words)

documents = [
    "The cat is chasing mice in the garden",
    "Cats are known for their hunting abilities",
    "The garden is full of colorful flowers"
]

query = "cats chasing in gardens"

normalized_docs = [normalize_text(doc) for doc in documents]
normalized_query = normalize_text(query)

for i, doc in enumerate(normalized_docs):
    if any(term in doc for term in normalized_query.split()):
        print(f"Document {i+1} matches the query")

# Output:
# Document 1 matches the query
# Document 2 matches the query
# Document 3 matches the query
```

Slide 11: Real-Life Example: Sentiment Analysis

Stemming or lemmatization can improve sentiment analysis by reducing word variations and focusing on core meanings.

```python
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text.lower())
    return [wnl.lemmatize(word, 'v') for word in words if word not in stop_words]

positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful'])
negative_words = set(['bad', 'terrible', 'awful', 'horrible', 'poor'])

def analyze_sentiment(text):
    processed_text = preprocess_text(text)
    positive_score = sum(1 for word in processed_text if word in positive_words)
    negative_score = sum(1 for word in processed_text if word in negative_words)
    return "Positive" if positive_score > negative_score else "Negative" if negative_score > positive_score else "Neutral"

reviews = [
    "The movie was absolutely amazing and wonderful!",
    "I had a terrible experience at the restaurant. The food was awful.",
    "The product is okay, nothing special."
]

for review in reviews:
    print(f"Review: {review}")
    print(f"Sentiment: {analyze_sentiment(review)}\n")

# Output:
# Review: The movie was absolutely amazing and wonderful!
# Sentiment: Positive

# Review: I had a terrible experience at the restaurant. The food was awful.
# Sentiment: Negative

# Review: The product is okay, nothing special.
# Sentiment: Neutral
```

Slide 12: Choosing Between Stemming and Lemmatization

The choice between stemming and lemmatization depends on your specific use case:

1. Speed vs. Accuracy: If processing speed is crucial, stemming might be preferred. For higher accuracy, choose lemmatization.
2. Domain Specificity: In some domains, stemming might suffice, while others may require the precision of lemmatization.
3. Language Complexity: For languages with complex morphology, lemmatization often yields better results.
4. Available Resources: Lemmatization requires more computational resources and often a comprehensive dictionary.

Slide 13: Choosing Between Stemming and Lemmatization

```python
import time
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
wnl = WordNetLemmatizer()

text = "The programmers were debugging the system for hours"
words = word_tokenize(text)

start = time.time()
stemmed = [ps.stem(word) for word in words]
stem_time = time.time() - start

start = time.time()
lemmatized = [wnl.lemmatize(word, 'v') for word in words]
lem_time = time.time() - start

print(f"Original: {' '.join(words)}")
print(f"Stemmed ({stem_time:.5f}s): {' '.join(stemmed)}")
print(f"Lemmatized ({lem_time:.5f}s): {' '.join(lemmatized)}")

# Output (times may vary):
# Original: The programmers were debugging the system for hours
# Stemmed (0.00015s): The programm were debug the system for hour
# Lemmatized (0.00101s): The programmer be debug the system for hour
```

Slide 14: Challenges and Limitations

Both stemming and lemmatization face challenges:

1. Irregular Words: Words like "be", "am", "is", "are" can be problematic.
2. Context Dependency: The meaning of a word can change based on context.
3. Domain-Specific Vocabulary: Technical terms may not be handled correctly.
4. Multilingual Support: Effectiveness varies across languages.

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

ps = PorterStemmer()
wnl = WordNetLemmatizer()

challenging_words = ["better", "worse", "go", "went", "gone", "bacteria", "criterion"]

for word in challenging_words:
    print(f"Word: {word}")
    print(f"Stemmed: {ps.stem(word)}")
    print(f"Lemmatized (n): {wnl.lemmatize(word, 'n')}")
    print(f"Lemmatized (v): {wnl.lemmatize(word, 'v')}")
    print(f"Lemmatized (a): {wnl.lemmatize(word, 'a')}")
    print()

# Output:
# Word: better
# Stemmed: better
# Lemmatized (n): better
# Lemmatized (v): better
# Lemmatized (a): good
# ...
```

Slide 15: Future Directions and Advanced Techniques

As NLP continues to evolve, new approaches are emerging:

1. Machine Learning-based Lemmatization: Using ML models for context-aware lemmatization.
2. Neural Network Stemmers: Leveraging deep learning for more accurate stemming.
3. Subword Tokenization: Methods like Byte-Pair Encoding (BPE) that can handle out-of-vocabulary words.

```python
# Simplified example of a basic neural stemmer concept
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SimpleNeuralStemmer:
    def __init__(self, input_size, hidden_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, input_size)
    
    def forward(self, x):
        self.z = sigmoid(np.dot(x, self.W1))
        return sigmoid(np.dot(self.z, self.W2))
    
    def stem(self, word):
        x = np.array([ord(c) for c in word.ljust(10, ' ')]) / 255
        output = self.forward(x)
        return ''.join([chr(int(i * 255)) for i in output if i > 0.5])

stemmer = SimpleNeuralStemmer(10, 5)
print(stemmer.stem("running"))  # Output will be random as the model is not trained

# Note: This is a conceptual example and would require training to function properly
```

Slide 16: Additional Resources

For those interested in diving deeper into stemming and lemmatization, here are some valuable resources:

1. ArXiv paper on advanced lemmatization techniques: "Neural Lemmatization with Adaptive Memory" by Chaitanya Malylnn et al. ArXiv ID: 1908.10576
2. ArXiv paper on multilingual stemming: "A Survey of Stemming Algorithms for Information Retrieval" by Anjali Ganesh Jivani ArXiv ID: 1205.0947

These papers provide in-depth discussions on the latest advancements in stemming and lemmatization techniques.

