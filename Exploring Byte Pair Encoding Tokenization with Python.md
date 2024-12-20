## Exploring Byte Pair Encoding Tokenization with Python
Slide 1: Introduction to Byte Pair Encoding (BPE)

Byte Pair Encoding is a data compression technique that iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte. In the context of natural language processing, BPE is used for tokenization, breaking down text into subword units. This approach balances the benefits of character-level and word-level tokenization, allowing for efficient representation of both common and rare words.

```python
import re
from collections import defaultdict

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
num_merges = 10

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f"Merge #{i+1}: {best}")

print("\nFinal vocabulary:", vocab)
```

Slide 2: BPE Algorithm Explained

The BPE algorithm starts with a vocabulary of individual characters and iteratively merges the most frequent adjacent pairs. This process continues until a desired vocabulary size is reached or no more merges are possible. The resulting vocabulary contains both individual characters and subword units, allowing for efficient tokenization of text.

```python
def byte_pair_encoding(text, num_merges):
    # Initialize vocabulary with character-level tokens
    vocab = {' '.join(word) + ' </w>': 1 for word in text.split()}
    
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(f"Merge #{i+1}: {best}")
    
    return vocab

text = "low lower newest widest"
num_merges = 10

final_vocab = byte_pair_encoding(text, num_merges)
print("\nFinal vocabulary:", final_vocab)
```

Slide 3: Tokenization with BPE

Once we have our BPE vocabulary, we can use it to tokenize new text. The tokenization process involves splitting the text into words and then applying the learned merges to each word. This results in a sequence of subword tokens that efficiently represent the input text.

```python
def tokenize(text, vocab):
    words = text.split()
    tokens = []
    for word in words:
        word = ' '.join(word) + ' </w>'
        while True:
            subwords = word.split()
            if subwords == [word]:
                break
            for i in range(len(subwords) - 1):
                if ' '.join(subwords[i:i+2]) in vocab:
                    subwords = subwords[:i] + [' '.join(subwords[i:i+2])] + subwords[i+2:]
                    break
            word = ' '.join(subwords)
        tokens.extend(word.split())
    return tokens

# Using the vocabulary from the previous slide
text_to_tokenize = "lower newest wide"
tokens = tokenize(text_to_tokenize, final_vocab)
print("Tokenized text:", tokens)
```

Slide 4: Efficiency of BPE Tokenization

BPE tokenization offers several efficiency advantages. It creates a compact vocabulary that can represent both common and rare words effectively. This leads to shorter sequences of tokens compared to character-level tokenization, while still maintaining the ability to handle out-of-vocabulary words better than word-level tokenization.

```python
import matplotlib.pyplot as plt

def compare_tokenization_methods(text):
    # Character-level tokenization
    char_tokens = list(text.replace(" ", "_"))
    
    # Word-level tokenization
    word_tokens = text.split()
    
    # BPE tokenization (using our previous implementation)
    bpe_vocab = byte_pair_encoding(text, num_merges=10)
    bpe_tokens = tokenize(text, bpe_vocab)
    
    # Plot comparison
    methods = ['Character', 'Word', 'BPE']
    token_counts = [len(char_tokens), len(word_tokens), len(bpe_tokens)]
    
    plt.bar(methods, token_counts)
    plt.title('Comparison of Tokenization Methods')
    plt.xlabel('Tokenization Method')
    plt.ylabel('Number of Tokens')
    plt.show()

sample_text = "the quick brown fox jumps over the lazy dog"
compare_tokenization_methods(sample_text)
```

Slide 5: Handling Out-of-Vocabulary Words

One of the key advantages of BPE is its ability to handle out-of-vocabulary (OOV) words. By breaking words into subword units, BPE can often represent new or rare words using combinations of known subwords. This makes it particularly useful for languages with rich morphology or for domains with specialized vocabulary.

```python
def handle_oov(word, vocab):
    if word in vocab:
        return [word]
    
    tokens = []
    while word:
        found = False
        for i in range(len(word), 0, -1):
            if word[:i] in vocab:
                tokens.append(word[:i])
                word = word[i:]
                found = True
                break
        if not found:
            tokens.append(word[0])
            word = word[1:]
    return tokens

# Example vocabulary
vocab = {'un': 1, 'common': 1, 'word': 1, 'sub': 1}

# Test with in-vocabulary and out-of-vocabulary words
words = ['uncommon', 'subword', 'unknown']

for word in words:
    tokens = handle_oov(word, vocab)
    print(f"'{word}' tokenized as: {tokens}")
```

Slide 6: BPE for Multilingual Tokenization

BPE is particularly useful for multilingual tokenization as it can effectively handle different languages with a single vocabulary. By learning subword units that are common across languages, BPE can create a compact, shared vocabulary that works well for multiple languages simultaneously.

```python
def learn_multilingual_bpe(texts, num_merges):
    # Initialize vocabulary with character-level tokens for all languages
    vocab = {}
    for lang, text in texts.items():
        for word in text.split():
            token = ' '.join(word) + ' </w>'
            vocab[token] = vocab.get(token, 0) + 1
    
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(f"Merge #{i+1}: {best}")
    
    return vocab

# Example texts in different languages
texts = {
    'english': "the quick brown fox jumps over the lazy dog",
    'spanish': "el rápido zorro marrón salta sobre el perro perezoso",
    'german': "der schnelle braune fuchs springt über den faulen hund"
}

multilingual_vocab = learn_multilingual_bpe(texts, num_merges=20)
print("\nMultilingual vocabulary size:", len(multilingual_vocab))
```

Slide 7: Compression Efficiency of BPE

BPE achieves compression by replacing frequent pairs of tokens with single tokens. This results in a more compact representation of the text, which can lead to significant space savings, especially for large corpora. Let's examine the compression ratio achieved by BPE on a sample text.

```python
def calculate_compression_ratio(original_text, bpe_vocab):
    original_size = len(original_text)
    
    # Tokenize the text using BPE
    tokens = tokenize(original_text, bpe_vocab)
    
    # Calculate the size of the compressed representation
    compressed_size = sum(len(token) for token in tokens)
    
    compression_ratio = original_size / compressed_size
    return compression_ratio

sample_text = "the quick brown fox jumps over the lazy dog " * 100
bpe_vocab = byte_pair_encoding(sample_text, num_merges=50)

compression_ratio = calculate_compression_ratio(sample_text, bpe_vocab)
print(f"Compression ratio: {compression_ratio:.2f}")
print(f"Space savings: {(1 - 1/compression_ratio) * 100:.2f}%")
```

Slide 8: BPE for Sentiment Analysis

BPE tokenization can be particularly useful for sentiment analysis tasks. By breaking words into subword units, it can capture sentiment-related morphemes and handle variations of words more effectively. Let's implement a simple sentiment analyzer using BPE tokenization.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def bpe_sentiment_analyzer(train_texts, train_labels, test_texts, num_merges=100):
    # Learn BPE vocabulary
    bpe_vocab = byte_pair_encoding(' '.join(train_texts), num_merges)
    
    # Tokenize texts using BPE
    tokenized_train = [' '.join(tokenize(text, bpe_vocab)) for text in train_texts]
    tokenized_test = [' '.join(tokenize(text, bpe_vocab)) for text in test_texts]
    
    # Create feature vectors
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(tokenized_train)
    X_test = vectorizer.transform(tokenized_test)
    
    # Train classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, train_labels)
    
    # Predict sentiments
    predictions = classifier.predict(X_test)
    return predictions

# Example usage
train_texts = ["I love this movie", "This film is terrible", "Great acting and plot"]
train_labels = [1, 0, 1]  # 1 for positive, 0 for negative
test_texts = ["The actors were amazing", "I didn't enjoy the story"]

predictions = bpe_sentiment_analyzer(train_texts, train_labels, test_texts)
print("Predicted sentiments:", predictions)
```

Slide 9: BPE for Machine Translation

BPE is widely used in machine translation systems to handle vocabulary differences between languages. It allows for a shared vocabulary across source and target languages, which can improve translation quality, especially for morphologically rich languages or languages with compound words.

```python
def create_parallel_bpe_vocab(source_texts, target_texts, num_merges):
    # Combine source and target texts
    all_texts = source_texts + target_texts
    
    # Learn BPE vocabulary on combined texts
    combined_vocab = byte_pair_encoding(' '.join(all_texts), num_merges)
    
    return combined_vocab

# Example parallel texts
source_texts = [
    "The cat is on the mat",
    "I love to eat pizza",
    "She runs every morning"
]
target_texts = [
    "Le chat est sur le tapis",
    "J'aime manger de la pizza",
    "Elle court tous les matins"
]

shared_vocab = create_parallel_bpe_vocab(source_texts, target_texts, num_merges=50)

# Tokenize a new sentence using the shared vocabulary
new_sentence = "The dog barks loudly"
tokens = tokenize(new_sentence, shared_vocab)
print(f"Tokenized sentence: {tokens}")
```

Slide 10: BPE for Text Generation

BPE tokenization can enhance text generation models by providing a balance between character-level and word-level generation. This allows the model to generate both common and rare words effectively. Let's implement a simple n-gram based text generator using BPE tokens.

```python
import random

def generate_text_with_bpe(seed_text, vocab, n=3, length=50):
    # Tokenize the seed text
    tokens = tokenize(seed_text, vocab)
    
    # Generate text
    generated_tokens = tokens.()
    for _ in range(length):
        ngram = tuple(generated_tokens[-n:])
        candidates = [token for token, _ in vocab.items() if token.startswith(' '.join(ngram))]
        if not candidates:
            break
        next_token = random.choice(candidates)
        generated_tokens.append(next_token.split()[-1])
    
    # Join tokens back into text
    generated_text = ' '.join(generated_tokens).replace(' </w>', '').replace(' ', '')
    return generated_text

# Example usage
seed_text = "The quick brown fox"
generated_text = generate_text_with_bpe(seed_text, final_vocab, n=2, length=20)
print("Generated text:", generated_text)
```

Slide 11: BPE for Data Augmentation

BPE can be used for data augmentation in natural language processing tasks. By manipulating BPE tokens, we can create variations of existing sentences, which can help improve the robustness of machine learning models. Let's implement a simple data augmentation technique using BPE.

```python
import random

def augment_text_with_bpe(text, vocab, num_augmentations=5):
    tokens = tokenize(text, vocab)
    augmented_texts = []
    
    for _ in range(num_augmentations):
        augmented_tokens = tokens.()
        
        # Randomly replace some tokens with other tokens from the vocabulary
        for i in range(len(augmented_tokens)):
            if random.random() < 0.2:  # 20% chance of replacement
                augmented_tokens[i] = random.choice(list(vocab.keys()))
        
        augmented_text = ' '.join(augmented_tokens).replace(' </w>', '').replace(' ', '')
        augmented_texts.append(augmented_text)
    
    return augmented_texts

# Example usage
original_text = "The quick brown fox jumps over the lazy dog"
augmented_texts = augment_text_with_bpe(original_text, final_vocab)

print("Original text:", original_text)
print("Augmented texts:")
for i, text in enumerate(augmented_texts, 1):
    print(f"{i}. {text}")
```

Slide 12: BPE for Language Identification

BPE can be useful for language identification tasks, as it can capture language-specific subword patterns. Let's implement a simple language identifier using BPE tokenization and a Naive Bayes classifier.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def train_language_identifier(texts, languages, num_merges=100):
    # Learn BPE vocabulary
    all_text = ' '.join(texts)
    bpe_vocab = byte_pair_encoding(all_text, num_merges)
    
    # Tokenize texts using BPE
    tokenized_texts = [' '.join(tokenize(text, bpe_vocab)) for text in texts]
    
    # Create feature vectors
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(tokenized_texts)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, languages, test_size=0.2, random_state=42)
    
    # Train classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    
    # Evaluate classifier
    accuracy = classifier.score(X_test, y_test)
    
    return vectorizer, classifier, bpe_vocab, accuracy

# Example usage
texts = [
    "The quick brown fox jumps over the lazy dog",
    "Le renard brun rapide saute par-dessus le chien paresseux",
    "Der schnelle braune Fuchs springt über den faulen Hund",
    "The cat is on the mat",
    "Le chat est sur le tapis",
    "Die Katze ist auf der Matte"
]
languages = ["en", "fr", "de", "en", "fr", "de"]

vectorizer, classifier, bpe_vocab, accuracy = train_language_identifier(texts, languages)
print(f"Classifier accuracy: {accuracy:.2f}")

# Identify language of a new text
new_text = "Hello, how are you today?"
tokenized_new_text = ' '.join(tokenize(new_text, bpe_vocab))
new_text_vector = vectorizer.transform([tokenized_new_text])
predicted_language = classifier.predict(new_text_vector)[0]
print(f"Predicted language: {predicted_language}")
```

Slide 13: Real-life Example: BPE in Neural Machine Translation

BPE tokenization is widely used in neural machine translation systems. It helps handle the vocabulary challenge in translation tasks, especially for languages with rich morphology or compound words. Let's look at a simplified example of how BPE might be used in a translation pipeline.

```python
import numpy as np

def simple_translation_model(source_tokens, source_to_target_vocab, target_vocab_size):
    # This is a highly simplified model for illustration purposes
    translated_indices = [source_to_target_vocab.get(token, np.random.randint(target_vocab_size)) for token in source_tokens]
    return translated_indices

# Simulate BPE vocabulary and translation model
source_bpe_vocab = {"the": 0, "qu": 1, "ick": 2, "brown": 3, "fox": 4}
target_bpe_vocab = {"le": 0, "rap": 1, "ide": 2, "renard": 3, "brun": 4}
source_to_target_vocab = {"the": 0, "qu": 1, "ick": 2, "brown": 4, "fox": 3}

source_sentence = "the quick brown fox"
source_tokens = tokenize(source_sentence, source_bpe_vocab)
print("Source tokens:", source_tokens)

translated_indices = simple_translation_model(source_tokens, source_to_target_vocab, len(target_bpe_vocab))
translated_tokens = [list(target_bpe_vocab.keys())[i] for i in translated_indices]
translated_sentence = "".join(translated_tokens)

print("Translated sentence:", translated_sentence)
```

Slide 14: Real-life Example: BPE in Text Classification

BPE tokenization can improve text classification tasks by capturing meaningful subword units. This is particularly useful for domains with specialized vocabulary or for handling out-of-vocabulary words. Let's implement a simple text classifier using BPE tokenization.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

def train_text_classifier(texts, labels, num_merges=100):
    # Learn BPE vocabulary
    bpe_vocab = byte_pair_encoding(' '.join(texts), num_merges)
    
    # Tokenize texts using BPE
    tokenized_texts = [' '.join(tokenize(text, bpe_vocab)) for text in texts]
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(tokenized_texts)
    
    # Train classifier
    classifier = SVC(kernel='linear')
    classifier.fit(X, labels)
    
    return vectorizer, classifier, bpe_vocab

# Example usage
texts = [
    "The new product launch was a great success",
    "Our customer service team handled the complaint efficiently",
    "The quarterly financial report shows positive growth",
    "Employee satisfaction has improved significantly this year"
]
labels = ["Marketing", "Support", "Finance", "HR"]

vectorizer, classifier, bpe_vocab = train_text_classifier(texts, labels)

# Classify a new text
new_text = "We need to improve our product marketing strategy"
tokenized_new_text = ' '.join(tokenize(new_text, bpe_vocab))
new_text_vector = vectorizer.transform([tokenized_new_text])
predicted_category = classifier.predict(new_text_vector)[0]
print(f"Predicted category: {predicted_category}")
```

Slide 15: Additional Resources

For those interested in diving deeper into Byte Pair Encoding and its applications in natural language processing, here are some valuable resources:

1. "Neural Machine Translation of Rare Words with Subword Units" by Sennrich et al. (2016) ArXiv URL: [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)
2. "BPE-Dropout: Simple and Effective Subword Regularization" by Provilkov et al. (2020) ArXiv URL: [https://arxiv.org/abs/1910.13267](https://arxiv.org/abs/1910.13267)
3. "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates" by Kudo (2018) ArXiv URL: [https://arxiv.org/abs/1804.10959](https://arxiv.org/abs/1804.10959)

These papers provide in-depth explanations of BPE and its variants, as well as their applications in various NLP tasks. They offer valuable insights into the theoretical foundations and practical implementations of BPE tokenization techniques.

