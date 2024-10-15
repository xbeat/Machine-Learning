## NLP FastText! Word Embedding with Python
Slide 1: Introduction to FastText

FastText is an open-source library developed by Facebook's AI Research lab for efficient learning of word representations and sentence classification. It extends the word2vec model by representing each word as a bag of character n-grams, allowing it to capture subword information and handle out-of-vocabulary words effectively.

```python
import fasttext

# Train a FastText model
model = fasttext.train_unsupervised('corpus.txt', model='skipgram')

# Get word vector
word_vector = model.get_word_vector('example')
print(word_vector)
```

Slide 2: Word Embeddings Basics

Word embeddings are dense vector representations of words in a continuous vector space. They capture semantic relationships between words, allowing similar words to have similar vector representations. FastText builds upon this concept by incorporating subword information.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simplified word embedding visualization
words = ['king', 'queen', 'man', 'woman']
embeddings = np.random.rand(4, 2)  # 2D for visualization

plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, (embeddings[i, 0], embeddings[i, 1]))
plt.title('Word Embeddings Visualization')
plt.show()
```

Slide 3: FastText Model Architecture

FastText uses a shallow neural network with an input layer, a hidden layer, and an output layer. The input layer represents words or n-grams, the hidden layer learns the embeddings, and the output layer predicts the context words or labels for classification tasks.

```python
import torch.nn as nn

class SimplifiedFastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SimplifiedFastText, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        output = self.linear(embeds.mean(dim=1))
        return output

# Usage
model = SimplifiedFastText(vocab_size=10000, embedding_dim=100)
```

Slide 4: Subword Information

FastText represents words as bags of character n-grams, allowing it to capture morphological information and handle out-of-vocabulary words. This approach is particularly useful for languages with rich morphology or for dealing with rare words.

```python
def get_ngrams(word, n_min=3, n_max=6):
    ngrams = []
    word = "<" + word + ">"
    for n in range(n_min, min(len(word), n_max) + 1):
        for i in range(len(word) - n + 1):
            ngrams.append(word[i:i+n])
    return ngrams

word = "example"
print(f"N-grams for '{word}':", get_ngrams(word))
```

Slide 5: Training a FastText Model

Training a FastText model involves preparing a corpus, setting hyperparameters, and using either the supervised or unsupervised learning mode. The unsupervised mode learns word representations, while the supervised mode is used for text classification.

```python
import fasttext

# Prepare corpus (one sentence per line)
with open('corpus.txt', 'w') as f:
    f.write("This is an example sentence.\n")
    f.write("Another sentence for training.\n")

# Train unsupervised model
model = fasttext.train_unsupervised('corpus.txt', 
                                    model='skipgram',
                                    dim=100,
                                    epoch=5,
                                    lr=0.1)

# Save the model
model.save_model("fasttext_model.bin")
```

Slide 6: Word Similarity and Analogies

FastText embeddings can be used to find similar words and solve word analogies. This is useful for various NLP tasks and applications, such as recommendation systems or language understanding.

```python
import fasttext

# Load pre-trained model
model = fasttext.load_model("fasttext_model.bin")

# Find similar words
similar_words = model.get_nearest_neighbors("computer", k=5)
print("Words similar to 'computer':", similar_words)

# Word analogy
result = model.get_analogies("king", "man", "woman")
print("king - man + woman =", result)
```

Slide 7: Text Classification with FastText

FastText can be used for efficient text classification tasks. It's particularly useful for large-scale problems with many categories. The model can handle both single-label and multi-label classification.

```python
import fasttext

# Prepare labeled data (format: __label__category text)
with open('train.txt', 'w') as f:
    f.write("__label__positive This movie is great!\n")
    f.write("__label__negative I didn't like the book.\n")

# Train classifier
classifier = fasttext.train_supervised('train.txt', lr=0.5, epoch=25)

# Predict
text = "I enjoyed watching this film."
predictions = classifier.predict(text)
print(f"Text: {text}")
print(f"Predicted label: {predictions[0][0]}, Probability: {predictions[1][0]:.2f}")
```

Slide 8: Handling Out-of-Vocabulary Words

FastText's use of subword information allows it to generate embeddings for words not seen during training. This is a significant advantage over traditional word embedding methods.

```python
import fasttext

# Load pre-trained model
model = fasttext.load_model("fasttext_model.bin")

# Get vector for an out-of-vocabulary word
oov_word = "untrainedword"
oov_vector = model.get_word_vector(oov_word)

print(f"Vector for '{oov_word}':")
print(oov_vector[:10])  # Print first 10 elements

# Find nearest neighbors for the OOV word
nearest_neighbors = model.get_nearest_neighbors(oov_word, k=5)
print(f"Nearest neighbors for '{oov_word}':", nearest_neighbors)
```

Slide 9: FastText vs Word2Vec

FastText builds upon Word2Vec by incorporating subword information. This comparison highlights the key differences and advantages of FastText over traditional word embedding techniques.

```python
import fasttext
import gensim

# FastText
fasttext_model = fasttext.train_unsupervised('corpus.txt', model='skipgram')

# Word2Vec
sentences = [line.split() for line in open('corpus.txt', 'r')]
word2vec_model = gensim.models.Word2Vec(sentences, min_count=1)

# Compare embeddings
word = "example"
print("FastText embedding:", fasttext_model.get_word_vector(word)[:5])
print("Word2Vec embedding:", word2vec_model.wv[word][:5])

# Out-of-vocabulary word
oov_word = "unseeword"
print("FastText OOV:", fasttext_model.get_word_vector(oov_word)[:5])
# Word2Vec will raise KeyError for OOV words
```

Slide 10: Preprocessing for FastText

Proper text preprocessing is crucial for optimal FastText performance. This includes tokenization, lowercasing, and handling special characters. FastText's subword approach reduces the need for extensive preprocessing compared to other models.

```python
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Join tokens
    return ' '.join(tokens)

# Example usage
raw_text = "Hello, world! This is an example."
processed_text = preprocess_text(raw_text)
print("Raw text:", raw_text)
print("Processed text:", processed_text)
```

Slide 11: FastText for Multilingual Applications

FastText's ability to handle subword information makes it particularly useful for multilingual applications. It can generate meaningful embeddings for languages with complex morphology or limited training data.

```python
import fasttext

# Train multilingual model
model = fasttext.train_unsupervised(
    'multilingual_corpus.txt',
    model='skipgram',
    dim=100,
    minn=2,
    maxn=5
)

# Get embeddings for words in different languages
languages = ['english', 'spanish', 'french', 'german']
word = 'hello'

for lang in languages:
    vector = model.get_word_vector(f"{word}_{lang}")
    print(f"Embedding for '{word}' in {lang}:", vector[:5])
```

Slide 12: Real-Life Example: Sentiment Analysis

FastText can be effectively used for sentiment analysis tasks, such as analyzing customer reviews or social media posts. This example demonstrates how to train and use a FastText model for sentiment classification.

```python
import fasttext

# Prepare labeled data
with open('reviews.txt', 'w') as f:
    f.write("__label__positive The product exceeded my expectations.\n")
    f.write("__label__negative The service was disappointing.\n")
    f.write("__label__neutral It's an average product, nothing special.\n")

# Train the model
model = fasttext.train_supervised('reviews.txt', lr=0.5, epoch=25)

# Analyze new reviews
new_reviews = [
    "I love this product!",
    "The quality is terrible.",
    "It's okay, but could be better."
]

for review in new_reviews:
    label, prob = model.predict(review)
    print(f"Review: {review}")
    print(f"Sentiment: {label[0]}, Probability: {prob[0]:.2f}\n")
```

Slide 13: Real-Life Example: Language Identification

FastText can be used for language identification, which is useful for processing multilingual content. This example shows how to train and use a FastText model for identifying languages in short text snippets.

```python
import fasttext

# Prepare training data
with open('lang_data.txt', 'w') as f:
    f.write("__label__en This is an English sentence.\n")
    f.write("__label__es Esta es una oración en español.\n")
    f.write("__label__fr Ceci est une phrase en français.\n")

# Train the model
model = fasttext.train_supervised('lang_data.txt', lr=0.5, epoch=25)

# Identify languages
texts = [
    "Hello, how are you?",
    "Bonjour, comment allez-vous?",
    "Hola, ¿cómo estás?"
]

for text in texts:
    lang, prob = model.predict(text)
    print(f"Text: {text}")
    print(f"Detected language: {lang[0]}, Probability: {prob[0]:.2f}\n")
```

Slide 14: FastText Optimization and Performance Tuning

Optimizing FastText models involves tuning hyperparameters and considering trade-offs between model size, training speed, and performance. Key parameters include learning rate, embedding dimension, and n-gram sizes.

```python
import fasttext
import time

def train_and_evaluate(params):
    start_time = time.time()
    model = fasttext.train_supervised('train_data.txt', **params)
    train_time = time.time() - start_time
    
    accuracy = model.test('test_data.txt')[1]
    model_size = model.get_input_matrix().size * 4 / (1024 * 1024)  # Size in MB
    
    return accuracy, train_time, model_size

# Different configurations
configs = [
    {'dim': 100, 'epoch': 5, 'lr': 0.1, 'wordNgrams': 2},
    {'dim': 200, 'epoch': 10, 'lr': 0.05, 'wordNgrams': 3},
    {'dim': 300, 'epoch': 15, 'lr': 0.01, 'wordNgrams': 4}
]

for i, config in enumerate(configs):
    accuracy, train_time, model_size = train_and_evaluate(config)
    print(f"Config {i+1}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Training Time: {train_time:.2f} seconds")
    print(f"  Model Size: {model_size:.2f} MB\n")
```

Slide 15: Additional Resources

For more information on FastText and its applications, consider exploring the following resources:

1. FastText official documentation: [https://fasttext.cc/docs/en/support.html](https://fasttext.cc/docs/en/support.html)
2. "Enriching Word Vectors with Subword Information" by P. Bojanowski et al. (2017): [https://arxiv.org/abs/1607.04606](https://arxiv.org/abs/1607.04606)
3. "Bag of Tricks for Efficient Text Classification" by A. Joulin et al. (2016): [https://arxiv.org/abs/1607.01759](https://arxiv.org/abs/1607.01759)
4. "FastText.zip: Compressing text classification models" by A. Joulin et al. (2016): [https://arxiv.org/abs/1612.03651](https://arxiv.org/abs/1612.03651)

These resources provide in-depth information on the FastText algorithm, its implementation, and various applications in natural language processing tasks.

